#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._external_runtime import NativeWorkerCall, NativeWorkerIdentity
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._assembly import MeshAssembly, MeshPart
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._coupling import OversetCoupling
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._scope import MeshingScope
from ._worker import ProviderWorker


_CELL_KINDS = ("tetrahedron", "pyramid", "prism", "hexahedron")
_STENCIL_WIDTHS = (4, 5, 6, 8)
_MAXIMUM_STENCIL = 8
_RANK_ARRAYS = {
    "parts": np.dtype(np.int64),
    "node_iblank": np.dtype(np.int32),
    "cell_iblank": np.dtype(np.int32),
    "donor_parts": np.dtype(np.int64),
    "receptor_parts": np.dtype(np.int64),
    "receptor_nodes": np.dtype(np.int64),
    "donor_cells": np.dtype(np.int64),
    "stencil_offsets": np.dtype(np.int64),
    "stencil_nodes": np.dtype(np.int64),
    "stencil_weights": np.dtype(np.float64),
}
_DONOR_ARRAYS = tuple(_RANK_ARRAYS)[3:]


class TiogaOptions(StrictModule, NonTrainableState):
    """Persistent TIOGA worker options; no native dependency is loaded on import.

    Build ``native/providers/tioga`` against a real TIOGA installation and put
    ``phydrax-tioga-worker`` on PATH, set PHYDRAX_TIOGA_WORKER, or pass
    ``executable``. ``ranks`` distributes complete parts round-robin, not cells
    within a part; more than one rank launches the worker through
    ``mpi_launcher`` with ``mpi_arguments`` and ``-n ranks``, whose MPI
    implementation must match the linked one. Linear nodal interpolation is not
    conservative overlap remapping or high-order transfer.
    """

    executable: str | None = eqx.field(static=True)
    mpi_launcher: str = eqx.field(static=True)
    mpi_arguments: tuple[str, ...] = eqx.field(static=True)
    ranks: int = eqx.field(static=True)
    fringe_layers: int = eqx.field(static=True)
    exclusion_layers: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        executable: str | None = None,
        mpi_launcher: str = "mpiexec",
        mpi_arguments: tuple[str, ...] = (),
        ranks: int = 1,
        fringe_layers: int = 1,
        exclusion_layers: int = 3,
        tolerance: float = 1e-9,
    ) -> None:
        for name, value, minimum in (
            ("ranks", ranks, 1),
            ("fringe_layers", fringe_layers, 1),
            ("exclusion_layers", exclusion_layers, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not minimum <= value <= np.iinfo(np.int32).max
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum} fitting TIOGA int32."
                )
        if executable is not None and (
            not isinstance(executable, str) or not executable.strip()
        ):
            raise ValueError("executable must be a nonempty path or None.")
        if not isinstance(mpi_launcher, str) or not mpi_launcher.strip():
            raise ValueError("mpi_launcher must be a nonempty executable path.")
        if not isinstance(mpi_arguments, tuple) or any(
            not isinstance(value, str) or not value for value in mpi_arguments
        ):
            raise ValueError("mpi_arguments must be a tuple of nonempty argv tokens.")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must be finite and between zero and one.")
        self.executable, self.mpi_launcher = executable, mpi_launcher
        self.mpi_arguments = mpi_arguments
        self.ranks, self.fringe_layers, self.exclusion_layers = (
            int(ranks),
            int(fringe_layers),
            int(exclusion_layers),
        )
        self.tolerance = float(tolerance)
        self.options_id = canonical_fingerprint(
            {
                "kind": "tioga-options",
                "executable": executable,
                "launcher": mpi_launcher,
                "launcher_arguments": mpi_arguments,
                "ranks": self.ranks,
                "fringe": self.fringe_layers,
                "exclude": self.exclusion_layers,
                "tolerance": self.tolerance,
            }
        )


class TiogaPartBlanking(StrictModule, NonTrainableState):
    """Unmodified TIOGA IBLANK arrays in original mesh row order.

    Node 1 means active, 0 means hole, and negative means receptor. Cell 1
    means active, 0 hole, -1 fringe. IDs retain the original part namespace.
    """

    part_name: str = eqx.field(static=True)
    part_id: str = eqx.field(static=True)
    node_ids: Array
    node_iblank: Array
    cell_ids: Array
    cell_iblank: Array
    report_id: str = eqx.field(static=True)

    def __init__(
        self, part: MeshPart, node_iblank: ArrayLike, cell_iblank: ArrayLike, /
    ) -> None:
        if not isinstance(part, MeshPart) or not isinstance(
            part.carrier, CellMeshingResult
        ):
            raise TypeError("TIOGA blanking requires a cell mesh part.")
        mesh = part.carrier.mesh
        nodes, cells = np.asarray(node_iblank), np.asarray(cell_iblank)
        cell_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        if (
            nodes.dtype.kind not in "iu"
            or cells.dtype.kind not in "iu"
            or nodes.shape != mesh.vertex_global_ids.shape
            or cells.shape != cell_ids.shape
            or np.any(nodes > 1)
            or np.any(~np.isin(cells, (-1, 0, 1)))
        ):
            raise ValueError("TIOGA returned invalid node/cell IBLANK arrays.")
        self.part_name, self.part_id = part.name, part.part_id
        self.node_ids, self.cell_ids = mesh.vertex_global_ids, jnp.asarray(cell_ids)
        self.node_iblank, self.cell_iblank = jnp.asarray(nodes), jnp.asarray(cells)
        self.report_id = canonical_fingerprint(
            {
                "kind": "tioga-blanking",
                "part": part.part_id,
                "nodes": array_tree_fingerprint(nodes),
                "cells": array_tree_fingerprint(cells),
            }
        )


class TiogaDonorEvidence(StrictModule, NonTrainableState):
    """Actual native donor cells and raw weights aligned to coupling receptor rows.

    The generic positive coupling clips only roundoff-negative weights within
    tolerance, then normalizes. These raw weights preserve the native evidence.
    """

    coupling_id: str = eqx.field(static=True)
    donor_cell_scope: MeshingScope
    donor_cell_ids: Array
    raw_weights: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        coupling: OversetCoupling,
        donor_cell_ids: ArrayLike,
        raw_weights: ArrayLike,
        /,
    ) -> None:
        cells, weights = (
            np.asarray(donor_cell_ids),
            np.asarray(raw_weights, dtype=np.float64),
        )
        if (
            cells.dtype.kind not in "iu"
            or cells.shape != (coupling.target_scope.entity_ids.size,)
            or weights.shape != coupling.donor_weights.shape
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("TIOGA donor evidence has incompatible rows.")
        source.require_scope(coupling.source_scope)
        self.coupling_id = coupling.coupling_id
        self.donor_cell_scope = source.scope(3, np.unique(cells))
        self.donor_cell_ids, self.raw_weights = jnp.asarray(cells), jnp.asarray(weights)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "tioga-donors",
                "coupling": coupling.coupling_id,
                "cells": array_tree_fingerprint(cells),
                "weights": array_tree_fingerprint(weights),
            }
        )


class TiogaRegistration(StrictModule, NonTrainableState):
    """Worker-resident TIOGA registration state an assembly result describes.

    ``session_id`` names the worker session holding the registration, and
    ``state`` the exact registered coordinates (every motion update advances
    it). Only a result whose registration is still resident in that session at
    that state can be moved; anything else fails explicitly instead of
    registering again. Boundary node IDs follow ``part_names`` order.
    """

    session_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)
    registration: int = eqx.field(static=True)
    state: int = eqx.field(static=True)
    part_names: tuple[str, ...] = eqx.field(static=True)
    wall_node_ids: tuple[Array, ...]
    overset_node_ids: tuple[Array, ...]
    registration_id: str = eqx.field(static=True)

    def __init__(
        self,
        session_id: str,
        identity_id: str,
        registration: int,
        state: int,
        part_names: tuple[str, ...],
        wall_node_ids: tuple[ArrayLike, ...],
        overset_node_ids: tuple[ArrayLike, ...],
        /,
    ) -> None:
        if not all(
            isinstance(value, str) and value for value in (session_id, identity_id)
        ):
            raise ValueError("TIOGA registrations require worker session identities.")
        if (
            type(registration) is not int
            or type(state) is not int
            or not 0 < registration <= state
        ):
            raise ValueError("TIOGA registration and state must be ordered sequences.")
        names = tuple(part_names)
        walls = tuple(np.asarray(values, dtype=np.int64) for values in wall_node_ids)
        overset = tuple(np.asarray(values, dtype=np.int64) for values in overset_node_ids)
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or not all(isinstance(name, str) for name in names)
            or len(walls) != len(names)
            or len(overset) != len(names)
            or any(values.ndim != 1 for values in (*walls, *overset))
        ):
            raise ValueError("TIOGA registration boundaries must follow its parts.")
        self.session_id, self.identity_id = session_id, identity_id
        self.registration, self.state = registration, state
        self.part_names = names
        self.wall_node_ids = tuple(jnp.asarray(values) for values in walls)
        self.overset_node_ids = tuple(jnp.asarray(values) for values in overset)
        self.registration_id = canonical_fingerprint(
            {
                "kind": "tioga-registration",
                "session": session_id,
                "identity": identity_id,
                "registration": registration,
                "state": state,
                "parts": names,
                "walls": [array_tree_fingerprint(values) for values in walls],
                "overset": [array_tree_fingerprint(values) for values in overset],
            }
        )


class TiogaAssemblyResult(StrictModule, NonTrainableState):
    assembly: MeshAssembly
    blanking: tuple[TiogaPartBlanking, ...]
    donors: tuple[TiogaDonorEvidence, ...]
    registration: TiogaRegistration
    provider: MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    provenance: SemanticProvenance
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        assembly: MeshAssembly,
        blanking: tuple[TiogaPartBlanking, ...],
        donors: tuple[TiogaDonorEvidence, ...],
        registration: TiogaRegistration,
        provider: MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        provenance: SemanticProvenance,
        /,
    ) -> None:
        if {item.part_id for item in blanking} != {
            part.part_id for part in assembly.parts
        } or len(blanking) != len(assembly.parts):
            raise ValueError(
                "TIOGA result requires blanking for every exact assembly part."
            )
        links = {
            link.coupling_id
            for link in assembly.couplings
            if isinstance(link, OversetCoupling)
        }
        if {item.coupling_id for item in donors} != links or len(donors) != len(links):
            raise ValueError("TIOGA result requires evidence for every overset coupling.")
        if not isinstance(registration, TiogaRegistration) or registration.part_names != (
            tuple(part.name for part in assembly.parts)
        ):
            raise ValueError("TIOGA result registration must follow the assembly parts.")
        self.assembly, self.blanking, self.donors = (
            assembly,
            tuple(blanking),
            tuple(donors),
        )
        self.registration = registration
        self.provider, self.runtime, self.provenance = provider, runtime, provenance
        self.derivative_mode = MeshingDerivativeMode.NONDIFFERENTIABLE
        self.result_id = canonical_fingerprint(
            {
                "kind": "tioga-assembly-result",
                "assembly": assembly.assembly_id,
                "blanking": [item.report_id for item in blanking],
                "donors": [item.evidence_id for item in donors],
                "registration": registration.registration_id,
                "runtime": runtime.runtime_id,
                "provenance": provenance.semantic_id,
            }
        )


@dataclass(frozen=True, slots=True)
class _PartTables:
    """Host row tables of one registered part in original mesh row order."""

    vertex_ids: np.ndarray
    cell_ids: np.ndarray
    cell_nodes: np.ndarray
    coordinates: np.ndarray


def _conversion(message: str, /) -> MeshingFailure:
    return MeshingFailure(MeshingFailureCategory.CONVERSION_FAILED, message)


def _tables(part: MeshPart, /) -> _PartTables:
    # ty: ignore[unresolved-attribute]
    mesh = part.carrier.mesh
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    cell_nodes = np.full((cell_ids.size, _MAXIMUM_STENCIL), -1, dtype=np.int64)
    start = 0
    for block in mesh.blocks:
        cell_nodes[start : start + block.cell_count, : block.arity] = np.asarray(
            block.vertices, dtype=np.int64
        )
        start += block.cell_count
    return _PartTables(
        np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        cell_ids,
        cell_nodes,
        np.asarray(mesh.coordinates, dtype=np.float64),
    )


def _rows(vertex_ids: np.ndarray, identifiers: np.ndarray, /) -> np.ndarray:
    """Rows of known vertex IDs (membership is guaranteed by scope checks)."""
    order = np.argsort(vertex_ids, kind="stable")
    return order[np.searchsorted(vertex_ids[order], identifiers)].astype(np.int32)


def _admit_parts(assembly: MeshAssembly, limits: MeshingLimits, /) -> None:
    if len(assembly.parts) < 2:
        raise ValueError("TIOGA requires at least two parts.")
    if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
        raise ValueError("Remove previous overset overlays before reassembling.")
    totals = np.zeros(3, dtype=np.int64)
    for part in assembly.parts:
        if (
            not isinstance(part.carrier, CellMeshingResult)
            or part.intrinsic_dimension != 3
            or part.ambient_dimension != 3
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "TIOGA requires certified 3D CellMesh parts, not implicit tessellation of other carriers.",
            )
        mesh = part.carrier.mesh
        if any(block.cell_kind not in _CELL_KINDS for block in mesh.blocks):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "TIOGA supports tetrahedra, pyramids, prisms and hexahedra, not arbitrary polyhedra.",
            )
        elements, routes, coordinates = part.carrier.geometry.resolve(mesh)
        if not np.array_equal(
            np.asarray(coordinates), np.asarray(mesh.coordinates)
        ) or any(
            element.local_dof_count != block.arity
            or not np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "TIOGA adapter requires vertex-linear geometry; high-order geometry cannot be silently dropped.",
            )
        connectivity = sum(block.vertices.size for block in mesh.blocks)
        if (
            mesh.coordinates.size > np.iinfo(np.int32).max
            or connectivity > np.iinfo(np.int32).max
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Mesh exceeds TIOGA int32 local indexing capacity.",
            )
        totals += (
            mesh.coordinates.shape[0],
            sum(block.cell_count for block in mesh.blocks),
            connectivity,
        )
    if (
        totals[0] > limits.maximum_vertices
        or totals[1] > limits.maximum_cells
        or totals[2] > limits.maximum_connectivity_entries
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "TIOGA input exceeds configured entity or connectivity limits.",
        )


def _boundaries(
    assembly: MeshAssembly, scopes: tuple[MeshingScope, ...], /
) -> tuple[np.ndarray, ...]:
    """Unique boundary node IDs of every part, in assembly part order."""
    grouped: dict[str, list[np.ndarray]] = {part.name: [] for part in assembly.parts}
    for scope in scopes:
        if not isinstance(scope, MeshingScope) or scope.entity_dimension != 0:
            raise TypeError(
                "TIOGA boundaries require revision-bound vertex MeshingScope values."
            )
        part = assembly.part(scope.source_id)
        part.require_scope(scope)
        grouped[part.name].append(np.asarray(scope.entity_ids, dtype=np.int64))
    return tuple(
        np.unique(np.concatenate(grouped[part.name], dtype=np.int64))
        if grouped[part.name]
        else np.zeros(0, dtype=np.int64)
        for part in assembly.parts
    )


def _boundary_arrays(
    tables: tuple[_PartTables, ...], identifiers: tuple[np.ndarray, ...], /
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        _rows(table.vertex_ids, values)
        for table, values in zip(tables, identifiers, strict=True)
    ]
    offsets = np.concatenate(([0], np.cumsum([row.size for row in rows]))).astype(
        np.int64
    )
    return offsets, np.concatenate(rows, dtype=np.int32)


def _registration_arrays(
    assembly: MeshAssembly,
    tables: tuple[_PartTables, ...],
    walls: tuple[np.ndarray, ...],
    overset: tuple[np.ndarray, ...],
    /,
) -> dict[str, np.ndarray]:
    blocks = [
        (index, block)
        for index, part in enumerate(assembly.parts)
        # ty: ignore[unresolved-attribute]
        for block in part.carrier.mesh.blocks
    ]
    wall_offsets, wall_nodes = _boundary_arrays(tables, walls)
    overset_offsets, overset_nodes = _boundary_arrays(tables, overset)
    return {
        "part_nodes": np.asarray(
            [table.vertex_ids.size for table in tables], dtype=np.int64
        ),
        "part_cells": np.asarray(
            [table.cell_ids.size for table in tables], dtype=np.int64
        ),
        "coordinates": np.concatenate([table.coordinates for table in tables]),
        "block_parts": np.asarray([index for index, _ in blocks], dtype=np.int64),
        "block_arities": np.asarray([block.arity for _, block in blocks], dtype=np.int64),
        "block_cells": np.asarray(
            [block.cell_count for _, block in blocks], dtype=np.int64
        ),
        "connectivity": np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32).ravel() for _, block in blocks]
        ),
        "wall_offsets": wall_offsets,
        "wall_nodes": wall_nodes,
        "overset_offsets": overset_offsets,
        "overset_nodes": overset_nodes,
    }


def _require_input_bytes(
    arrays: Mapping[str, np.ndarray], limits: MeshingLimits, /
) -> None:
    if sum(value.nbytes for value in arrays.values()) > limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "TIOGA input exceeds maximum_data_bytes.",
        )


@dataclass(frozen=True, slots=True)
class _Donors:
    """Validated donor records of every rank, one row per receptor."""

    donor_parts: np.ndarray
    receptor_parts: np.ndarray
    receptor_nodes: np.ndarray
    donor_cells: np.ndarray
    stencil_rows: np.ndarray
    raw_weights: np.ndarray


def _rank_outputs(
    call: NativeWorkerCall,
    tables: tuple[_PartTables, ...],
    ranks: int,
    /,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    count = len(tables)
    if call.arrays or set(call.parts) != {f"rank-{rank}" for rank in range(ranks)}:
        raise _conversion("TIOGA rank outputs are incomplete.")
    nodes = np.asarray([table.vertex_ids.size for table in tables], dtype=np.int64)
    cells = np.asarray([table.cell_ids.size for table in tables], dtype=np.int64)
    node_iblank: list[np.ndarray] = [np.zeros(0, dtype=np.int32)] * count
    cell_iblank: list[np.ndarray] = [np.zeros(0, dtype=np.int32)] * count
    records: dict[str, list[np.ndarray]] = {name: [] for name in _DONOR_ARRAYS}
    stencil_base = 0
    for rank in range(ranks):
        arrays = call.parts[f"rank-{rank}"]
        owned = np.arange(rank, count, ranks, dtype=np.int64)
        if {name: value.dtype for name, value in arrays.items()} != _RANK_ARRAYS or (
            not np.array_equal(arrays["parts"], owned)
        ):
            raise _conversion("TIOGA rank output ownership is incomplete.")
        donors = arrays["donor_parts"].size
        offsets = arrays["stencil_offsets"]
        if (
            arrays["node_iblank"].shape != (np.sum(nodes[owned]),)
            or arrays["cell_iblank"].shape != (np.sum(cells[owned]),)
            or any(
                arrays[name].shape != (donors,)
                for name in ("receptor_parts", "receptor_nodes", "donor_cells")
            )
            or offsets.shape != (donors + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or arrays["stencil_nodes"].shape != (offsets[-1],)
            or arrays["stencil_weights"].shape != (offsets[-1],)
            or np.any(np.remainder(arrays["donor_parts"], ranks) != rank)
        ):
            raise _conversion("TIOGA rank output arrays are inconsistent.")
        for part, values in zip(
            owned, np.split(arrays["node_iblank"], np.cumsum(nodes[owned])[:-1])
        ):
            node_iblank[part] = np.array(values, dtype=np.int32)
        for part, values in zip(
            owned, np.split(arrays["cell_iblank"], np.cumsum(cells[owned])[:-1])
        ):
            cell_iblank[part] = np.array(values, dtype=np.int32)
        for name in records:
            values = arrays[name]
            records[name].append(
                values[1:] + stencil_base if name == "stencil_offsets" else values
            )
        stencil_base += offsets[-1]
    merged = {name: np.concatenate(values) for name, values in records.items()}
    merged["stencil_offsets"] = np.concatenate(
        ([0], merged["stencil_offsets"]), dtype=np.int64
    )
    return node_iblank, cell_iblank, merged


def _donors(
    records: dict[str, np.ndarray],
    tables: tuple[_PartTables, ...],
    node_iblank: list[np.ndarray],
    limits: MeshingLimits,
    tolerance: float,
    /,
) -> _Donors:
    """Validate native donor records without per-donor Python work."""
    donor_parts, receptor_parts = records["donor_parts"], records["receptor_parts"]
    receptor_nodes, donor_cells = records["receptor_nodes"], records["donor_cells"]
    offsets, stencil_nodes = records["stencil_offsets"], records["stencil_nodes"]
    count = donor_parts.size
    if (
        count > limits.maximum_vertices
        or stencil_nodes.size > limits.maximum_connectivity_entries
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "TIOGA donor inventory exceeds resource limits.",
        )
    nodes = np.asarray([table.vertex_ids.size for table in tables], dtype=np.int64)
    cells = np.asarray([table.cell_ids.size for table in tables], dtype=np.int64)
    widths = np.diff(offsets)
    if (
        np.any(~np.isin(widths, _STENCIL_WIDTHS))
        or np.any((donor_parts < 0) | (donor_parts >= nodes.size))
        or np.any((receptor_parts < 0) | (receptor_parts >= nodes.size))
        or np.any(receptor_parts == donor_parts)
        or np.any((receptor_nodes < 0) | (receptor_nodes >= nodes[receptor_parts]))
        or np.any((donor_cells < 0) | (donor_cells >= cells[donor_parts]))
    ):
        raise _conversion("TIOGA returned invalid donor/receptor routing.")
    owner = np.repeat(np.arange(count), widths)
    if np.any((stencil_nodes < 0) | (stencil_nodes >= nodes[donor_parts[owner]])):
        raise _conversion("TIOGA returned an invalid donor stencil node.")
    column = np.arange(stencil_nodes.size) - offsets[:-1][owner]
    rows = np.full((count, _MAXIMUM_STENCIL), -1, dtype=np.int64)
    raw = np.zeros((count, _MAXIMUM_STENCIL), dtype=np.float64)
    rows[owner, column] = stencil_nodes
    raw[owner, column] = records["stencil_weights"]
    # Every stencil must be exactly the node set of its reported donor cell.
    for part, table in enumerate(tables):
        selected = donor_parts == part
        if not np.array_equal(
            np.sort(table.cell_nodes[donor_cells[selected]], axis=1),
            np.sort(rows[selected], axis=1),
        ):
            raise _conversion(
                "TIOGA donor stencil does not belong to its reported source cell."
            )
    keys = np.sort(receptor_parts * np.max(nodes) + receptor_nodes)
    if np.any(keys[1:] == keys[:-1]):
        raise _conversion("TIOGA assigned multiple donors to one receptor.")
    for part, values in enumerate(node_iblank):
        receptors = np.zeros(values.size, dtype=np.bool_)
        receptors[receptor_nodes[receptor_parts == part]] = True
        if not np.array_equal(receptors, values < 0):
            raise _conversion("TIOGA receptor blanking and donor records disagree.")
    if (
        not np.all(np.isfinite(raw))
        or np.any(raw < -tolerance)
        or np.any(np.abs(raw.sum(axis=1) - 1) > tolerance)
    ):
        raise _conversion(
            "TIOGA stencil is not positive partition-of-unity within tolerance."
        )
    return _Donors(donor_parts, receptor_parts, receptor_nodes, donor_cells, rows, raw)


def _coupling(
    source: MeshPart,
    target: MeshPart,
    source_table: _PartTables,
    target_table: _PartTables,
    donors: _Donors,
    selected: np.ndarray,
    hole_scope: MeshingScope | None,
    tolerance: float,
    /,
) -> tuple[OversetCoupling, TiogaDonorEvidence]:
    target_ids = target_table.vertex_ids[donors.receptor_nodes[selected]]
    order = np.argsort(target_ids, kind="stable")
    selected, target_ids = selected[order], target_ids[order]
    rows, raw = donors.stencil_rows[selected], donors.raw_weights[selected]
    valid = rows >= 0
    weights = np.maximum(raw, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    donor_points = np.where(
        valid[..., None], source_table.coordinates[np.maximum(rows, 0)], 0
    )
    reconstructed = np.sum(weights[..., None] * donor_points, axis=1)
    expected = target_table.coordinates[donors.receptor_nodes[selected]]
    used = source_table.coordinates[np.unique(rows[valid])]
    scale = max(
        np.max(np.abs(used)),
        np.max(np.abs(expected)),
        np.max(np.ptp(used, axis=0)),
        np.finfo(np.float64).tiny,
    )
    if not np.allclose(reconstructed, expected, rtol=0, atol=10 * tolerance * scale):
        raise _conversion("TIOGA donor weights do not reproduce receptor coordinates.")
    identifiers = np.where(valid, source_table.vertex_ids[np.maximum(rows, 0)], -1)
    link = OversetCoupling(
        source,
        target,
        source.scope(0, np.unique(identifiers[valid])),
        target.scope(0, target_ids),
        identifiers,
        weights,
        hole_scope=hole_scope,
        tolerance=tolerance,
    )
    cells = source_table.cell_ids[donors.donor_cells[selected]]
    return link, TiogaDonorEvidence(source, link, cells, raw)


def _assembly_outputs(
    call: NativeWorkerCall,
    parts: tuple[MeshPart, ...],
    registration: TiogaRegistration,
    options: TiogaOptions,
    limits: MeshingLimits,
    /,
) -> tuple[tuple[TiogaPartBlanking, ...], tuple[OversetCoupling, ...], tuple]:
    tables = tuple(_tables(part) for part in parts)
    node_iblank, cell_iblank, records = _rank_outputs(call, tables, options.ranks)
    donors = _donors(records, tables, node_iblank, limits, options.tolerance)
    blanking = tuple(
        TiogaPartBlanking(part, nodes, cells)
        for part, nodes, cells in zip(parts, node_iblank, cell_iblank, strict=True)
    )
    for part, table, values, walls, overset in zip(
        parts,
        tables,
        node_iblank,
        registration.wall_node_ids,
        registration.overset_node_ids,
        strict=True,
    ):
        if np.any(values[_rows(table.vertex_ids, np.asarray(overset))] == 1):
            raise _conversion(
                f"TIOGA left orphan overset boundary receptors in {part.name!r}."
            )
        if np.any(values[_rows(table.vertex_ids, np.asarray(walls))] == 0):
            raise _conversion(f"TIOGA blanked solid wall nodes in {part.name!r}.")
    holes = tuple(
        part.scope(0, table.vertex_ids[values == 0]) if np.any(values == 0) else None
        for part, table, values in zip(parts, tables, node_iblank, strict=True)
    )
    count = len(parts)
    pairs = donors.donor_parts * count + donors.receptor_parts
    links, evidence = [], []
    for key in np.unique(pairs):
        source, target = divmod(int(key), count)
        link, record = _coupling(
            parts[source],
            parts[target],
            tables[source],
            tables[target],
            donors,
            np.flatnonzero(pairs == key),
            holes[target],
            options.tolerance,
        )
        links.append(link)
        evidence.append(record)
    return blanking, tuple(links), tuple(evidence)


def _provider_info(identity: NativeWorkerIdentity, ranks: int, /) -> MeshingProviderInfo:
    reported = identity.reported
    revision = reported.get("revision")
    if (
        reported.get("provider") != "tioga"
        or not isinstance(revision, str)
        or not revision
        or any(character.isspace() for character in revision)
        or reported.get("node_global_ids") is not True
        or not isinstance(reported.get("operations"), list)
        or not {"move", "register"}.issubset(reported["operations"])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Unsupported TIOGA worker identity.",
            stage="startup",
        )
    if identity.ranks != ranks:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"The TIOGA worker started {identity.ranks} ranks instead of {ranks}.",
            stage="startup",
        )
    return MeshingProviderInfo(
        "tioga",
        revision,
        "BSD-3-Clause",
        operations=(MeshingOperation.ASSEMBLE_OVERSET,),
        source_kinds=(MeshingSourceKind.MESH_ASSEMBLY,),
        capabilities=(
            MeshingCapability.MIXED_CELLS,
            MeshingCapability.PARALLEL,
            MeshingCapability.DISTRIBUTED,
        ),
        cell_kinds=_CELL_KINDS,
        dimensions=(3,),
        execution_modes=(MeshingExecutionMode.SUBPROCESS,),
    )


def _moved_part(part: MeshPart, coordinates: ArrayLike, /) -> MeshPart:
    carrier = part.carrier
    points = np.asarray(coordinates)
    if points.dtype.kind not in "iuf":
        raise TypeError("TIOGA motion coordinates must be real arrays.")
    points = points.astype(np.float64)
    # ty: ignore[unresolved-attribute]
    current = np.asarray(carrier.mesh.coordinates, dtype=np.float64)
    if points.shape != current.shape or not np.all(np.isfinite(points)):
        raise ValueError(
            f"TIOGA motion of {part.name!r} requires finite coordinates of shape {current.shape}."
        )
    if np.array_equal(points, current):
        return part
    # ty: ignore[unresolved-attribute]
    if carrier.boundary is not None or carrier.associations:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            f"TIOGA motion cannot carry the geometry-bound boundary or associations of {part.name!r}.",
        )
    # ty: ignore[unresolved-attribute]
    mesh = carrier.mesh.with_coordinates(
        points,
        numeric_version=canonical_fingerprint(
            {
                "kind": "tioga-motion",
                # ty: ignore[unresolved-attribute]
                "mesh": carrier.mesh.mesh_id,
                "coordinates": array_tree_fingerprint(points),
            }
        ),
    )
    return MeshPart(
        part.name,
        certify_cell_mesh(
            mesh,
            part.coordinate_contract,
            # ty: ignore[unresolved-attribute]
            patches=carrier.patches,
            # ty: ignore[unresolved-attribute]
            zones=carrier.zones,
            # ty: ignore[unresolved-attribute]
            labels=carrier.labels,
            # ty: ignore[unresolved-attribute]
            attributes=carrier.attributes,
        ),
    )


class TiogaProvider:
    """TIOGA overset assembly through one persistent collective worker session.

    The worker identity is probed once per session; every ``execute`` and
    ``move`` reuses the session. ``move`` updates the resident registration in
    place, so only a result produced by the live session can be moved.
    """

    def __init__(self, options: TiogaOptions | None = None) -> None:
        self.options = TiogaOptions() if options is None else options
        if not isinstance(self.options, TiogaOptions):
            raise TypeError("options must be TiogaOptions.")
        launcher = (
            (
                self.options.mpi_launcher,
                *self.options.mpi_arguments,
                "-n",
                str(self.options.ranks),
            )
            if self.options.ranks > 1
            else ()
        )
        self.worker = ProviderWorker(
            "tioga",
            executable=self.options.executable,
            environment_variable="PHYDRAX_TIOGA_WORKER",
            default_executable="phydrax-tioga-worker",
            build_hint=(
                "Build native/providers/tioga and set PHYDRAX_TIOGA_WORKER or "
                "TiogaOptions.executable."
            ),
            launcher=launcher,
        )

    def info(self) -> MeshingProviderInfo:
        return _provider_info(self.worker.identity, self.options.ranks)

    def close(self) -> None:
        self.worker.close()

    def __enter__(self) -> TiogaProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def execute(
        self,
        assembly: MeshAssembly,
        /,
        *,
        wall_scopes: tuple[MeshingScope, ...] = (),
        overset_scopes: tuple[MeshingScope, ...] = (),
        limits: MeshingLimits | None = None,
    ) -> TiogaAssemblyResult:
        """Register and assemble 3D vertex-linear cell meshes without altering identities.

        Boundary scopes contain original global NODE IDs. Walls must describe
        closed solid boundaries, and overset nodes mark mandatory interpolation
        boundaries. Unspecified boundaries remain ordinary exterior boundaries.
        Any orphan mandatory receptor fails instead of inventing a donor.
        Existing non-overset overlays and all input audits are retained
        verbatim. The registration replaces any earlier one of this session.
        """
        if not isinstance(assembly, MeshAssembly):
            raise TypeError("assembly must be MeshAssembly.")
        limits_ = _limits(limits)
        if self.options.ranks > len(assembly.parts):
            raise ValueError("TIOGA requires ranks <= part count.")
        _admit_parts(assembly, limits_)
        walls = _boundaries(assembly, wall_scopes)
        overset = _boundaries(assembly, overset_scopes)
        if any(
            np.intersect1d(wall, boundary).size
            for wall, boundary in zip(walls, overset, strict=True)
        ):
            raise ValueError("Wall and overset boundary node scopes must be disjoint.")
        tables = tuple(_tables(part) for part in assembly.parts)
        arrays = _registration_arrays(assembly, tables, walls, overset)
        _require_input_bytes(arrays, limits_)
        provider = self.info()
        call = self.worker.call(
            "register",
            {
                "fringe_layers": self.options.fringe_layers,
                "exclusion_layers": self.options.exclusion_layers,
                "maximum_vertices": limits_.maximum_vertices,
                "maximum_cells": limits_.maximum_cells,
                "maximum_connectivity_entries": limits_.maximum_connectivity_entries,
            },
            arrays,
            limits=limits_,
        )
        return self._result(
            assembly.parts,
            assembly.couplings,
            walls,
            overset,
            call,
            provider,
            limits_,
            {"operation": "register", "source_assembly": assembly.assembly_id},
        )

    def move(
        self,
        previous: TiogaAssemblyResult,
        coordinates: Mapping[str, ArrayLike],
        /,
        *,
        limits: MeshingLimits | None = None,
    ) -> TiogaAssemblyResult:
        """Move named parts of a resident registration and rerun connectivity.

        ``coordinates`` maps part names to new coordinates in mesh row order;
        connectivity and every node/cell ID are unchanged. The worker rewrites
        the registered coordinates in place without restarting. A result whose
        registration is no longer resident at its exact state (worker restart,
        a later registration, or a later motion) fails explicitly. Moved parts
        are recertified; unmoved parts and their overlays are kept verbatim.
        """
        if not isinstance(previous, TiogaAssemblyResult):
            raise TypeError("previous must be TiogaAssemblyResult.")
        if not isinstance(coordinates, Mapping) or not coordinates:
            raise TypeError("coordinates must map moved part names to coordinates.")
        limits_ = _limits(limits)
        registration = previous.registration
        names = registration.part_names
        if any(name not in names for name in coordinates):
            raise ValueError("TIOGA motion names parts outside the registration.")
        moved = np.asarray(
            sorted(names.index(name) for name in coordinates), dtype=np.int64
        )
        tioga_links = {item.coupling_id for item in previous.donors}
        retained = tuple(
            link
            for link in previous.assembly.couplings
            if link.coupling_id not in tioga_links
        )
        if any(
            scope.source_id in coordinates
            for link in retained
            for scope in (link.source_scope, link.target_scope)
        ):
            raise ValueError(
                "Non-overset overlays of moved parts cannot follow a TIOGA motion."
            )
        identity = self.worker.identity
        if (
            identity.session_id != registration.session_id
            or identity.identity_id != registration.identity_id
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "The TIOGA registration belongs to a worker session that is no longer "
                "active; execute the assembly again instead of moving it.",
                stage="move",
            )
        provider = _provider_info(identity, self.options.ranks)
        parts = tuple(
            _moved_part(part, coordinates[part.name])
            if part.name in coordinates
            else part
            for part in previous.assembly.parts
        )
        moved_coordinates = []
        for index in moved.tolist():
            carrier = parts[int(index)].carrier
            if not isinstance(carrier, CellMeshingResult):
                raise TypeError("TIOGA motion requires cell mesh parts.")
            moved_coordinates.append(
                np.asarray(carrier.mesh.coordinates, dtype=np.float64)
            )
        arrays = {
            "moved_parts": moved,
            "coordinates": np.concatenate(moved_coordinates),
        }
        _require_input_bytes(arrays, limits_)
        call = self.worker.call(
            "move",
            {"registration": registration.registration, "state": registration.state},
            arrays,
            limits=limits_,
        )
        return self._result(
            parts,
            retained,
            tuple(np.asarray(values) for values in registration.wall_node_ids),
            tuple(np.asarray(values) for values in registration.overset_node_ids),
            call,
            provider,
            limits_,
            {
                "operation": "move",
                "previous_result": previous.result_id,
                "moved_parts": sorted(coordinates),
            },
        )

    def _result(
        self,
        parts: tuple[MeshPart, ...],
        retained: tuple,
        walls: tuple[np.ndarray, ...],
        overset: tuple[np.ndarray, ...],
        call: NativeWorkerCall,
        provider: MeshingProviderInfo,
        limits: MeshingLimits,
        content: dict,
        /,
    ) -> TiogaAssemblyResult:
        identity = self.worker.identity
        result = call.result
        if (
            set(result) != {"registration", "state"}
            or type(result["registration"]) is not int
            or type(result["state"]) is not int
            or result["state"] != call.sequence
        ):
            raise _conversion("TIOGA worker returned an invalid registration record.")
        registration = TiogaRegistration(
            identity.session_id,
            identity.identity_id,
            result["registration"],
            result["state"],
            tuple(part.name for part in parts),
            walls,
            overset,
        )
        # Provider output becomes domain objects here; any contract violation
        # the explicit checks above did not name is still a conversion failure.
        try:
            blanking, links, evidence = _assembly_outputs(
                call, parts, registration, self.options, limits
            )
            assembly = MeshAssembly(parts, couplings=(*retained, *links))
        except ValueError as error:
            raise _conversion(str(error)) from error
        provenance = SemanticProvenance(
            {
                "kind": "tioga-overset-assembly",
                **content,
                "parts": [part.part_id for part in parts],
                "registration": registration.registration_id,
                "input_manifest_sha256": call.evidence["input_manifest_sha256"],
                "output_manifest_sha256": call.evidence["output_manifest_sha256"],
                "worker_sequence": call.sequence,
                "worker_peak_rss_bytes": call.evidence["peak_rss_bytes"],
                "upstream_revision": provider.version,
                "options": self.options.options_id,
                "rank_distribution": "whole-parts-round-robin",
                "native_node_ids": "collision-free-part-namespaced; source IDs retained",
                "weights": "raw evidence retained; negative roundoff clipped then normalized",
                "topology_change": "none",
                "conservative_transfer": False,
            },
            resource_ids={
                "worker_identity": identity.identity_id,
                "worker_session": identity.session_id,
            },
        )
        runtime = MeshingRuntimeInfo(
            provider.provider_id,
            provider.version,
            MeshingExecutionMode.SUBPROCESS,
            deterministic=False,
            enforced_limits=(
                "wall_seconds",
                "input_entities",
                "input_connectivity_entries",
                "input_bytes",
                "output_bytes",
                "local_int32_indexing",
                "global_uint64_node_namespace",
                self.worker.memory_limit_evidence(),
            ),
            unenforced_limits=(
                "provider_internal_workspace",
                "native_donor_generation_preallocation",
            ),
        )
        return TiogaAssemblyResult(
            assembly, blanking, evidence, registration, provider, runtime, provenance
        )


def _limits(limits: MeshingLimits | None, /) -> MeshingLimits:
    limits_ = MeshingLimits() if limits is None else limits
    if not isinstance(limits_, MeshingLimits):
        raise TypeError("limits must be MeshingLimits.")
    if (
        limits_.maximum_vertices > 10_000_000
        or limits_.maximum_cells > 20_000_000
        or limits_.maximum_connectivity_entries > 500_000_000
        or limits_.maximum_data_bytes > 4_000_000_000
    ):
        raise ValueError("limits exceed the TIOGA worker hard bounds.")
    return limits_


__all__ = [
    "TiogaAssemblyResult",
    "TiogaDonorEvidence",
    "TiogaOptions",
    "TiogaPartBlanking",
    "TiogaProvider",
    "TiogaRegistration",
]
