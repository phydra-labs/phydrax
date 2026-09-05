#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import struct
import subprocess
import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._assembly import MeshAssembly, MeshPart
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._coupling import OversetCoupling
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._scope import MeshingScope


_CELL_KINDS = ("tetrahedron", "pyramid", "prism", "hexahedron")


class TiogaOptions(StrictModule, NonTrainableState):
    """Process-isolated TIOGA options; no native dependency is loaded on import.

    Build ``native/tioga`` against a real TIOGA installation and put
    ``phydrax-tioga`` on PATH, set PHYDRAX_TIOGA_EXECUTABLE, or pass executable.
    ``ranks`` distributes complete parts round-robin, not cells within a part.
    The linked MPI implementation must match ``mpi_launcher``. Linear nodal
    interpolation is not conservative overlap remapping or high-order transfer.
    """

    executable: str | None = eqx.field(static=True)
    mpi_launcher: str = eqx.field(static=True)
    mpi_arguments: tuple[str, ...] = eqx.field(static=True)
    ranks: int = eqx.field(static=True)
    fringe_layers: int = eqx.field(static=True)
    exclusion_layers: int = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
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
        timeout_seconds: float = 300.0,
        tolerance: float = 1e-9,
    ):
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
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite.")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must be finite and between zero and one.")
        self.executable, self.mpi_launcher = executable, mpi_launcher
        self.mpi_arguments = mpi_arguments
        self.ranks, self.fringe_layers, self.exclusion_layers = (
            int(ranks),
            int(fringe_layers),
            int(exclusion_layers),
        )
        self.timeout_seconds, self.tolerance = float(timeout_seconds), float(tolerance)
        self.options_id = canonical_fingerprint(
            {
                "kind": "tioga-options",
                "executable": executable,
                "launcher": mpi_launcher,
                "launcher_arguments": mpi_arguments,
                "ranks": self.ranks,
                "fringe": self.fringe_layers,
                "exclude": self.exclusion_layers,
                "timeout": self.timeout_seconds,
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

    def __init__(self, part: MeshPart, node_iblank: ArrayLike, cell_iblank: ArrayLike, /):
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
    ):
        cells, weights = np.asarray(donor_cell_ids), np.asarray(raw_weights, dtype=float)
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


class TiogaAssemblyResult(StrictModule, NonTrainableState):
    assembly: MeshAssembly
    blanking: tuple[TiogaPartBlanking, ...]
    donors: tuple[TiogaDonorEvidence, ...]
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
        provider: MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        provenance: SemanticProvenance,
        /,
    ):
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
        self.assembly, self.blanking, self.donors = (
            assembly,
            tuple(blanking),
            tuple(donors),
        )
        self.provider, self.runtime, self.provenance = provider, runtime, provenance
        self.derivative_mode = MeshingDerivativeMode.NONDIFFERENTIABLE
        self.result_id = canonical_fingerprint(
            {
                "kind": "tioga-assembly-result",
                "assembly": assembly.assembly_id,
                "blanking": [item.report_id for item in blanking],
                "donors": [item.evidence_id for item in donors],
                "runtime": runtime.runtime_id,
                "provenance": provenance.semantic_id,
            }
        )


def _run(command: list[str], timeout: float, *, cwd: str | None = None) -> str:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE, f"Cannot launch TIOGA: {error}"
        ) from error
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "TIOGA assembly exceeded its wall-time limit.",
        ) from error
    if process.returncode != 0:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"TIOGA exited with status {process.returncode}: {output[-12000:]}",
        )
    return output


def _executable(options: TiogaOptions) -> str:
    candidate = options.executable or os.environ.get(
        "PHYDRAX_TIOGA_EXECUTABLE", "phydrax-tioga"
    )
    executable = shutil.which(candidate)
    if executable is None:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Build native/tioga and set PHYDRAX_TIOGA_EXECUTABLE or TiogaOptions.executable.",
        )
    return str(Path(executable).resolve())


def _info(executable: str, timeout: float) -> MeshingProviderInfo:
    version = _run([executable, "--version"], timeout).strip()
    prefix = "phydrax-tioga/1 tioga/"
    if (
        not version.startswith(prefix)
        or not version[len(prefix) :]
        or any(char.isspace() for char in version[len(prefix) :])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Unsupported phydrax-tioga native bridge protocol/version.",
        )
    return MeshingProviderInfo(
        "tioga",
        version[len(prefix) :],
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


def _boundaries(
    assembly: MeshAssembly, scopes: tuple[MeshingScope, ...]
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for scope in scopes:
        if not isinstance(scope, MeshingScope) or scope.entity_dimension != 0:
            raise TypeError(
                "TIOGA boundaries require revision-bound vertex MeshingScope values."
            )
        part = assembly.part(scope.source_id)
        part.require_scope(scope)
        grouped.setdefault(part.name, []).append(np.asarray(scope.entity_ids))
    return {name: np.unique(np.concatenate(rows)) for name, rows in grouped.items()}


def _write_input(
    path: Path,
    assembly: MeshAssembly,
    walls: dict[str, np.ndarray],
    overset: dict[str, np.ndarray],
    options: TiogaOptions,
) -> None:
    with path.open("wb") as stream:
        stream.write(b"PXTIOGA1")
        stream.write(
            struct.pack(
                "=iii",
                len(assembly.parts),
                options.fringe_layers,
                options.exclusion_layers,
            )
        )
        node_offset = 0
        for part in assembly.parts:
            assert isinstance(part.carrier, CellMeshingResult)
            mesh = part.carrier.mesh
            vertices = np.asarray(mesh.vertex_global_ids)
            rows = {
                int(identifier): index + 1 for index, identifier in enumerate(vertices)
            }
            wall_rows = np.asarray(
                [rows[int(value)] for value in walls.get(part.name, ())], dtype=np.int32
            )
            overset_rows = np.asarray(
                [rows[int(value)] for value in overset.get(part.name, ())], dtype=np.int32
            )
            start = stream.tell()
            stream.write(struct.pack("=Q", 0))
            stream.write(
                struct.pack(
                    "=iiii",
                    vertices.size,
                    len(mesh.blocks),
                    wall_rows.size,
                    overset_rows.size,
                )
            )
            np.asarray(mesh.coordinates, dtype=np.float64).tofile(stream)
            vertices.astype(np.uint64, copy=False).tofile(stream)
            wall_rows.tofile(stream)
            overset_rows.tofile(stream)
            stream.write(struct.pack("=Q", node_offset))
            node_offset += vertices.size
            for block in mesh.blocks:
                stream.write(struct.pack("=ii", block.arity, block.cell_count))
                (np.asarray(block.vertices, dtype=np.int32) + 1).tofile(stream)
                np.asarray(block.global_ids, dtype=np.uint64).tofile(stream)
            end = stream.tell()
            stream.seek(start)
            stream.write(struct.pack("=Q", end - start - 8))
            stream.seek(end)


def _read_array(stream, dtype, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("Negative TIOGA output count.")
    values = np.fromfile(stream, dtype=dtype, count=count)
    if values.size != count:
        raise ValueError("Truncated TIOGA output.")
    return values


def _read_outputs(prefix: Path, assembly: MeshAssembly, ranks: int):
    blanking = {}
    records: dict[tuple[int, int], list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
    for rank in range(ranks):
        with Path(f"{prefix}.{rank}").open("rb") as stream:
            if stream.read(8) != b"PXTIOGR1":
                raise ValueError("Invalid TIOGA output protocol.")
            count = int(_read_array(stream, np.int32, 1)[0])
            if count != len(range(rank, len(assembly.parts), ranks)):
                raise ValueError("TIOGA output rank ownership is incomplete.")
            for _ in range(count):
                source, nodes, cells = (
                    int(value) for value in _read_array(stream, np.int32, 3)
                )
                if (
                    source not in range(rank, len(assembly.parts), ranks)
                    or source in blanking
                ):
                    raise ValueError("TIOGA returned duplicate/foreign mesh parts.")
                part = assembly.parts[source]
                assert isinstance(part.carrier, CellMeshingResult)
                mesh = part.carrier.mesh
                if nodes != mesh.coordinates.shape[0] or cells != sum(
                    block.cell_count for block in mesh.blocks
                ):
                    raise ValueError("TIOGA changed the input mesh cardinality.")
                blanking[source] = TiogaPartBlanking(
                    part,
                    _read_array(stream, np.int32, nodes),
                    _read_array(stream, np.int32, cells),
                )
                donor_count = int(_read_array(stream, np.int32, 1)[0])
                if donor_count < 0:
                    raise ValueError("Invalid TIOGA donor count.")
                for _ in range(donor_count):
                    target, receptor, width = (
                        int(value) for value in _read_array(stream, np.int32, 3)
                    )
                    if (
                        target not in range(len(assembly.parts))
                        or source == target
                        or width not in (4, 5, 6, 8)
                    ):
                        raise ValueError("Invalid TIOGA donor/receptor routing.")
                    target_part = assembly.parts[target]
                    assert isinstance(target_part.carrier, CellMeshingResult)
                    target_mesh = target_part.carrier.mesh
                    if not 0 <= receptor < target_mesh.coordinates.shape[0]:
                        raise ValueError("Unknown TIOGA receptor node.")
                    cell = int(_read_array(stream, np.uint64, 1)[0])
                    stencil = _read_array(
                        stream, np.dtype([("id", "=u8"), ("weight", "=f8")]), width
                    )
                    if np.any(stencil["id"] > np.iinfo(np.int64).max):
                        raise ValueError("Native donor ID exceeds canonical int64 range.")
                    records.setdefault((source, target), []).append(
                        (
                            int(np.asarray(target_mesh.vertex_global_ids)[receptor]),
                            cell,
                            stencil["id"].astype(np.int64),
                            stencil["weight"],
                        )
                    )
            if stream.read(1):
                raise ValueError("Unexpected trailing TIOGA output.")
    return tuple(blanking[index] for index in range(len(assembly.parts))), records


def _couplings(
    assembly: MeshAssembly,
    blanking: tuple[TiogaPartBlanking, ...],
    records,
    tolerance: float,
):
    links, evidence = [], []
    receptors = [set() for _ in assembly.parts]
    for (source_index, target_index), rows in sorted(records.items()):
        source, target = assembly.parts[source_index], assembly.parts[target_index]
        rows.sort(key=lambda item: item[0])
        target_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        if len(set(target_ids)) != len(rows) or receptors[target_index].intersection(
            target_ids
        ):
            raise ValueError("TIOGA assigned multiple donors to one receptor.")
        receptors[target_index].update(target_ids)
        width = max(len(row[2]) for row in rows)
        ids, raw = (
            np.full((len(rows), width), -1, dtype=np.int64),
            np.zeros((len(rows), width)),
        )
        cells = np.asarray([row[1] for row in rows], dtype=np.int64)
        source_mesh = source.carrier.mesh
        cell_nodes = {
            int(identifier): np.asarray(source_mesh.vertex_global_ids)[vertices]
            for block in source_mesh.blocks
            for identifier, vertices in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        }
        for index, (_, cell, nodes, weights) in enumerate(rows):
            if cell not in cell_nodes or not np.array_equal(
                np.sort(nodes), np.sort(cell_nodes[cell])
            ):
                raise ValueError(
                    "TIOGA donor stencil does not belong to its reported source cell."
                )
            ids[index, : len(nodes)], raw[index, : len(nodes)] = nodes, weights
        if (
            not np.all(np.isfinite(raw))
            or np.any(raw < -tolerance)
            or np.any(np.abs(raw.sum(axis=1) - 1) > tolerance)
        ):
            raise ValueError(
                "TIOGA stencil is not positive partition-of-unity within tolerance."
            )
        weights = np.maximum(raw, 0)
        weights /= weights.sum(axis=1, keepdims=True)
        holes = np.asarray(blanking[target_index].node_ids)[
            np.asarray(blanking[target_index].node_iblank) == 0
        ]
        link = OversetCoupling(
            source,
            target,
            source.scope(0, np.unique(ids[ids >= 0])),
            target.scope(0, target_ids),
            ids,
            weights,
            hole_scope=target.scope(0, holes) if holes.size else None,
            tolerance=tolerance,
        )
        coordinates = np.asarray(source.point_coordinates(link.source_scope))
        expected = np.asarray(target.point_coordinates(link.target_scope))
        reconstructed = np.asarray(link.transfer(coordinates))
        scale = max(
            float(np.max(np.abs(coordinates))),
            float(np.max(np.abs(expected))),
            float(np.max(np.ptp(coordinates, axis=0))),
            np.finfo(float).tiny,
        )
        if not np.allclose(reconstructed, expected, rtol=0, atol=10 * tolerance * scale):
            raise ValueError("TIOGA donor weights do not reproduce receptor coordinates.")
        links.append(link)
        evidence.append(TiogaDonorEvidence(source, link, cells, raw))
    for index, status in enumerate(blanking):
        expected = set(
            np.asarray(status.node_ids)[np.asarray(status.node_iblank) < 0].tolist()
        )
        if expected != receptors[index]:
            raise ValueError("TIOGA receptor blanking and donor records disagree.")
    return tuple(links), tuple(evidence)


class TiogaProvider:
    def __init__(self, options: TiogaOptions | None = None):
        self.options = TiogaOptions() if options is None else options
        if not isinstance(self.options, TiogaOptions):
            raise TypeError("options must be TiogaOptions.")

    def info(self) -> MeshingProviderInfo:
        return _info(_executable(self.options), self.options.timeout_seconds)

    def execute(
        self,
        assembly: MeshAssembly,
        /,
        *,
        wall_scopes: tuple[MeshingScope, ...] = (),
        overset_scopes: tuple[MeshingScope, ...] = (),
    ) -> TiogaAssemblyResult:
        """Assemble 3D vertex-linear cell meshes without altering their identities.

        Boundary scopes contain original global NODE IDs. Walls must describe
        closed solid boundaries, and overset nodes mark mandatory interpolation
        boundaries. Unspecified boundaries remain ordinary exterior boundaries.
        Any orphan mandatory receptor fails instead of inventing a donor.
        Existing non-overset overlays and all input audits are retained verbatim.
        """
        if not isinstance(assembly, MeshAssembly):
            raise TypeError("assembly must be MeshAssembly.")
        if len(assembly.parts) < 2 or self.options.ranks > len(assembly.parts):
            raise ValueError("TIOGA requires at least two parts and ranks <= part count.")
        if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
            raise ValueError("Remove previous overset overlays before reassembling.")
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
                for block, element, route in zip(
                    mesh.blocks, elements, routes, strict=True
                )
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                    "TIOGA adapter requires vertex-linear geometry; high-order geometry cannot be silently dropped.",
                )
            if (
                mesh.coordinates.size > np.iinfo(np.int32).max
                or sum(block.vertices.size for block in mesh.blocks)
                > np.iinfo(np.int32).max
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Mesh exceeds TIOGA int32 local indexing capacity.",
                )
        walls, overset = (
            _boundaries(assembly, wall_scopes),
            _boundaries(assembly, overset_scopes),
        )
        for name in walls.keys() & overset.keys():
            if np.intersect1d(walls[name], overset[name]).size:
                raise ValueError(
                    "Wall and overset boundary node scopes must be disjoint."
                )
        executable = _executable(self.options)
        provider = _info(executable, self.options.timeout_seconds)
        command = [executable]
        if self.options.ranks > 1:
            launcher = shutil.which(self.options.mpi_launcher)
            if launcher is None:
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                    "TIOGA MPI launcher is unavailable.",
                )
            command = [
                str(Path(launcher).resolve()),
                *self.options.mpi_arguments,
                "-n",
                str(self.options.ranks),
                executable,
            ]
        with tempfile.TemporaryDirectory(prefix="phydrax-tioga-") as directory:
            path, prefix = Path(directory) / "input.bin", Path(directory) / "output.bin"
            _write_input(path, assembly, walls, overset, self.options)
            with path.open("rb") as stream:
                input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            _run(
                [*command, str(path), str(prefix)],
                self.options.timeout_seconds,
                cwd=directory,
            )
            try:
                blanking, records = _read_outputs(prefix, assembly, self.options.ranks)
                links, evidence = _couplings(
                    assembly, blanking, records, self.options.tolerance
                )
                for status in blanking:
                    ids, values = (
                        np.asarray(status.node_ids),
                        np.asarray(status.node_iblank),
                    )
                    if np.any(
                        values[np.isin(ids, overset.get(status.part_name, ()))] == 1
                    ):
                        raise ValueError(
                            f"TIOGA left orphan overset boundary receptors in {status.part_name!r}."
                        )
                    if np.any(values[np.isin(ids, walls.get(status.part_name, ()))] == 0):
                        raise ValueError(
                            f"TIOGA blanked solid wall nodes in {status.part_name!r}."
                        )
                result_assembly = MeshAssembly(
                    assembly.parts, couplings=(*assembly.couplings, *links)
                )
            except (ValueError, OSError, OverflowError) as error:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED, str(error)
                ) from error
        provenance = SemanticProvenance(
            {
                "kind": "tioga-overset-assembly",
                "source_assembly": assembly.assembly_id,
                "input_sha256": input_digest,
                "upstream_revision": provider.version,
                "options": self.options.options_id,
                "rank_distribution": "whole-parts-round-robin",
                "native_node_ids": "collision-free-part-namespaced; source IDs retained",
                "weights": "raw evidence retained; negative roundoff clipped then normalized",
                "topology_change": "none",
                "conservative_transfer": False,
            }
        )
        runtime = MeshingRuntimeInfo(
            provider.provider_id,
            provider.version,
            MeshingExecutionMode.SUBPROCESS,
            deterministic=False,
            enforced_limits=("wall_seconds", "local_int32_indexing"),
            unenforced_limits=("memory",),
        )
        return TiogaAssemblyResult(
            result_assembly, blanking, evidence, provider, runtime, provenance
        )


__all__ = [
    "TiogaAssemblyResult",
    "TiogaDonorEvidence",
    "TiogaOptions",
    "TiogaPartBlanking",
    "TiogaProvider",
]
