#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-routed physical finite-volume overlap and conservative transport."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_compact_cell_common_refinement,
    PreparedCommonRefinement,
)


if TYPE_CHECKING:
    from .._cell_geometry_transition_types import CellGeometryTransition
    from .._coordinate_enclosure import ExactScalar
    from .._view_support import OwnerLocalCellGeometry


_COVERAGE_ROUNDOFF_ULPS = 256.0


def _group_layout(group: ExecutionGroup, axis_name: str, /) -> tuple[Mesh, int, int]:
    if not isinstance(group, ExecutionGroup):
        raise TypeError("execution_group must be an ExecutionGroup.")
    axis = str(axis_name).strip()
    if not axis:
        raise ValueError("axis_name must be non-empty.")
    processes = tuple(group.spec.process_indices)
    devices = tuple(group.devices)
    if (
        not processes
        or len(devices) != len(processes)
        or tuple(device.process_index for device in devices) != processes
        or len(set(processes)) != len(processes)
        or not group.is_member
        or jax.process_index() not in processes
    ):
        raise ValueError(
            "Owner-local remapping requires exactly one ordered execution-group "
            "device per participating process."
        )
    rank = processes.index(jax.process_index())
    if len(devices) > np.iinfo(np.int32).max:
        raise ValueError("Execution group exceeds owner-rank capacity.")
    return Mesh(np.asarray(devices, dtype=object), (axis,)), rank, len(devices)


def _local_array(value: Any, mesh: Mesh, axis: str, owner_count: int, /) -> Array:
    local = jnp.asarray(value)
    return jax.make_array_from_process_local_data(
        NamedSharding(mesh, PartitionSpec(axis)),
        local[None, ...],
        global_shape=(owner_count, *local.shape),
    )


def _all_gather_local(
    value: Any,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    /,
) -> np.ndarray:
    distributed = _local_array(value, mesh, axis, owner_count)

    def gather(local: Array, /) -> Array:
        local_value = local[0]
        return jax.lax.all_gather(
            local_value,
            axis,
            axis=0,
            tiled=local_value.ndim != 0,
        )

    gathered = jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(distributed)
    shards = gathered.addressable_shards
    if len(shards) != 1:
        raise ValueError("Owner-local remapping requires one addressable group shard.")
    return np.asarray(jax.device_get(shards[0].data))


def _all_to_all_local(
    value: Any,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    /,
) -> np.ndarray:
    local = jnp.asarray(value)
    if local.ndim == 0 or local.shape[0] != owner_count:
        raise ValueError("Owner-routed packets need one leading row per group owner.")
    distributed = _local_array(local, mesh, axis, owner_count)

    def exchange(local_packet: Array, /) -> Array:
        received = jax.lax.all_to_all(
            local_packet[0],
            axis,
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        return received[None, ...]

    received = jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(axis),
        check_vma=False,
    )(distributed)
    shards = received.addressable_shards
    if len(shards) != 1:
        raise ValueError("Owner-local remapping requires one addressable group shard.")
    return np.asarray(jax.device_get(shards[0].data))[0]


def _group_max(value: int, mesh: Mesh, axis: str, owner_count: int, /) -> int:
    gathered = _all_gather_local(
        np.asarray(value, dtype=np.int64), mesh, axis, owner_count
    )
    return int(np.max(gathered))


def _group_sum(value: Array, mesh: Mesh, axis: str, owner_count: int, /) -> Array:
    local = jnp.asarray(value)
    distributed = _local_array(local, mesh, axis, owner_count)

    def reduce(local_value: Array, /) -> Array:
        return jax.lax.psum(local_value[0], axis)[None, ...]

    result = jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(axis),
        check_vma=False,
    )(distributed)
    shards = result.addressable_shards
    if len(shards) != 1:
        raise ValueError("Owner-local remapping requires one addressable group shard.")
    return jnp.asarray(jax.device_get(shards[0].data))[0]


def _synchronize_error(
    error: Exception | None,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    stage: str,
    /,
) -> None:
    failed = _all_gather_local(
        np.asarray(error is not None, dtype=np.int32), mesh, axis, owner_count
    )
    if np.any(failed):
        detail = (
            str(error) if error is not None else "another owner rejected its local input"
        )
        raise ValueError(
            f"Owner-local conservative remap {stage} failed collectively: {detail}"
        )


def _fingerprint_word(identity: str, /) -> np.int64:
    return np.int64(int(canonical_fingerprint(identity)[:15], 16))


def _require_group_identity(
    identity: Any, mesh: Mesh, axis: str, owner_count: int, name: str, /
) -> None:
    word = _fingerprint_word(canonical_fingerprint(identity))
    gathered = _all_gather_local(
        np.asarray(word, dtype=np.int64), mesh, axis, owner_count
    )
    if np.all(gathered == gathered[0]):
        return
    # Every rank observed the mismatch, so naming the differing components is
    # itself collective over one fixed-width row per rank.
    components: dict[str, Any] = {}
    if isinstance(identity, dict):
        for key, value in identity.items():
            if isinstance(value, dict):
                for inner, inner_value in value.items():
                    components[f"{key}.{inner}"] = inner_value
            else:
                components[str(key)] = value
    names = tuple(sorted(components))
    widths = _all_gather_local(
        np.asarray(len(names), dtype=np.int64), mesh, axis, owner_count
    )
    differing: tuple[str, ...] = ()
    if names and np.all(widths == widths[0]):
        words = np.asarray(
            _all_gather_local(
                np.asarray(
                    [
                        _fingerprint_word(canonical_fingerprint(components[key]))
                        for key in names
                    ],
                    dtype=np.int64,
                ),
                mesh,
                axis,
                owner_count,
            )
        ).reshape((owner_count, len(names)))
        differing = tuple(
            key for index, key in enumerate(names) if np.unique(words[:, index]).size > 1
        )
    detail = f": {', '.join(differing)}" if differing else ""
    raise ValueError(f"Owner-local {name} differs across execution-group ranks{detail}.")


def _resolve_group_cell_kind(
    kind: str, dimension: int, mesh: Mesh, axis: str, owner_count: int, name: str, /
) -> str:
    supported = (
        ("triangle", "quadrilateral", "polygon")
        if dimension == 2
        else ("tetrahedron", "polyhedron")
    )
    code = -1 if not kind else supported.index(kind)
    codes = _all_gather_local(
        np.asarray(code, dtype=np.int32), mesh, axis, owner_count
    ).reshape((-1,))
    present = np.unique(codes[codes >= 0])
    if present.size != 1:
        raise ValueError(
            f"Owner-local {name} cell family is absent or differs across ranks."
        )
    return supported[int(present[0])]


def _view_maps(
    view: Any, /
) -> tuple[
    str,
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, ...], ...] | None,
    bool,
    tuple[tuple[tuple[ExactScalar, ...], ...], ...],
]:
    from .._cell_geometry import CellVertexGeometryElement
    from .._reference_cell import reference_cell_topology
    from .._view_support import OwnerLocalCellGeometry
    from ..fem._reference import FiniteElementSpec

    if not isinstance(view, OwnerLocalCellGeometry):
        raise TypeError("Owner-local remapping requires OwnerLocalCellGeometry views.")
    dimension = int(view.topological_dimension)
    ambient = int(np.asarray(view.coordinates).shape[1])
    if dimension not in (2, 3) or ambient != dimension:
        raise ValueError(
            "Owner-local remapping requires a 2D or 3D physical cell-map view."
        )
    identifiers = np.asarray(view.cell_ids, dtype=np.int64)
    owners = np.asarray(view.cell_owner_ranks, dtype=np.int32)
    owned = np.asarray(view.owned, dtype=np.bool_)
    measures = np.asarray(view.cell_measures, dtype=np.float64)
    if (
        owners.shape != identifiers.shape
        or owned.shape != identifiers.shape
        or measures.shape != identifiers.shape
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError("Owner-local stable cell and ownership rows are malformed.")
    if not view.blocks:
        if identifiers.size:
            raise ValueError("Owner-local cells lack their straight-sided map blocks.")
        return "", (), identifiers, owners, owned, measures, None, True, ()
    kinds = {block.cell_kind for block in view.blocks}
    if len(kinds) != 1:
        raise ValueError(
            "Owner-local common refinement requires one straight-sided cell family per view."
        )
    kind = kinds.pop()
    supported = (
        ("triangle", "quadrilateral", "polygon")
        if dimension == 2
        else ("tetrahedron", "polyhedron")
    )
    if kind not in supported:
        raise ValueError(
            "Owner-local physical overlap supports triangles, quadrilaterals, "
            "straight-sided polygons, tetrahedra, and authored polyhedra."
        )
    rows_by_id = {int(identifier): row for row, identifier in enumerate(identifiers)}
    physical: list[np.ndarray | None] = [None] * identifiers.size
    cell_faces: list[tuple[np.ndarray, ...] | None] = [None] * identifiers.size
    exact_physical: list[tuple[tuple[ExactScalar, ...], ...] | None] = [
        None
    ] * identifiers.size
    coordinates = np.asarray(view.coordinates, dtype=np.float64)
    lattice = (
        None
        if view.periodic_lift_matrix is None
        else np.asarray(view.periodic_lift_matrix, dtype=np.float64)
    )
    periodic_rank = (
        0
        if view.vertex_windings is None
        else int(np.asarray(view.vertex_windings).shape[1])
    )
    exact_lattice = (
        None
        if lattice is None
        else tuple(
            tuple(Fraction.from_float(float(value)) for value in row) for row in lattice
        )
    )
    straight_map = True
    vertex_coordinates = np.asarray(view.vertex_coordinates, dtype=np.float64)
    raw_exact_vertices = view.exact_vertex_coordinates
    exact_vertex_bank = (
        tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in vertex_coordinates
        )
        if raw_exact_vertices is None
        else tuple(tuple(value for value in row) for row in raw_exact_vertices)
    )
    if len(exact_vertex_bank) != vertex_coordinates.shape[0] or any(
        len(row) != dimension for row in exact_vertex_bank
    ):
        raise ValueError("Exact owner-local vertex coordinates lost their source rows.")
    for block in view.blocks:
        if block.cell_kind in ("polygon", "polyhedron"):
            straight_block = (
                isinstance(block.element, CellVertexGeometryElement)
                and block.element.cell_kind == kind
                and block.chart_arguments is None
            )
            if not straight_block:
                raise ValueError(
                    "Variable-topology owner-local cells require vertex geometry maps."
                )
        else:
            topology = reference_cell_topology(kind)
            straight_block = (
                isinstance(block.element, FiniteElementSpec)
                and block.element.family == "Lagrange"
                and block.element.degree == 1
                and block.element.cell_kind == kind
                and block.element.local_dof_count == len(topology.vertices)
                and block.chart_arguments is None
                and block.element.form_basis is None
                and np.array_equal(
                    np.asarray(block.element.reference_nodes, dtype=np.float64),
                    np.asarray(topology.vertices, dtype=np.float64),
                )
            )
            if not straight_block and (dimension != 3 or kind != "tetrahedron"):
                raise ValueError(
                    "Curved common-root owner-local remapping currently requires 3D tetrahedra."
                )
            if not straight_block:
                straight_map = False
        cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
        routes = np.asarray(block.coordinate_routes, dtype=np.int64)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if (
            routes.ndim != 2
            or routes.shape[0] != cell_ids.size
            or routes.shape[1] != block.element.local_dof_count
            or np.any(routes < 0)
            or np.any(routes >= coordinates.shape[0])
            or vertices.shape != valid.shape
            or valid.shape[0] != cell_ids.size
            or np.any(vertices[~valid] != -1)
            or np.any(vertices[valid] < 0)
            or np.any(vertices[valid] >= vertex_coordinates.shape[0])
            or np.any(np.sum(valid, axis=1) < (3 if dimension == 2 else 4))
        ):
            raise ValueError(
                "Owner-local cell vertex and coordinate routes are malformed."
            )
        if straight_block and routes.shape != valid.shape:
            raise ValueError(
                "Straight-sided maps require one coordinate route per cell vertex."
            )
        shifts = (
            None
            if block.cell_vertex_shifts is None
            else np.asarray(block.cell_vertex_shifts, dtype=np.int64)
        )
        if (
            periodic_rank
            and not view.coordinates_are_lifted
            and (
                shifts is None
                or lattice is None
                or shifts.shape != (*valid.shape, periodic_rank)
            )
        ):
            raise ValueError(
                "Periodic affine cells require explicit lift shifts and lattice."
            )
        for local_row, identifier in enumerate(cell_ids):
            row = rows_by_id.get(int(identifier))
            if row is None or physical[row] is not None:
                raise ValueError(
                    "Owner-local block cell routes lost stable cell identity."
                )
            slots = np.flatnonzero(valid[local_row])
            points = vertex_coordinates[vertices[local_row, slots]]
            if periodic_rank and not view.coordinates_are_lifted:
                if shifts is None or lattice is None:
                    raise ValueError(
                        "Periodic cell lift lost its validated shifts or lattice."
                    )
                points = points + shifts[local_row, slots] @ lattice
            if not np.all(np.isfinite(points)):
                raise ValueError(
                    "Owner-local physical cell map contains non-finite vertices."
                )
            exact_points = tuple(
                exact_vertex_bank[int(vertex)] for vertex in vertices[local_row, slots]
            )
            if periodic_rank and not view.coordinates_are_lifted:
                if shifts is None or exact_lattice is None:
                    raise ValueError(
                        "Periodic exact vertex lift lost its source lattice."
                    )
                shift_rows = shifts[local_row, slots]
                exact_points = tuple(
                    tuple(
                        point[axis]
                        + sum(
                            (
                                Fraction(int(shift_rows[slot_row, lattice_axis]))
                                * exact_lattice[lattice_axis][axis]
                                for lattice_axis in range(periodic_rank)
                            ),
                            Fraction(0),
                        )
                        for axis in range(dimension)
                    )
                    for slot_row, point in enumerate(exact_points)
                )
            exact_physical[row] = exact_points
            physical[row] = points
            if kind == "polyhedron":
                loops = block.polyhedron_faces[local_row]
                signs = np.asarray(block.polyhedron_face_signs[local_row], dtype=np.int8)
                if len(loops) != signs.size:
                    raise ValueError(
                        "Authored polyhedron face loops and orientations differ."
                    )
                normalized = []
                slot_to_local = {
                    int(vertices[local_row, slot]): index
                    for index, slot in enumerate(slots)
                }
                for loop, sign in zip(loops, signs, strict=True):
                    face = np.asarray(loop, dtype=np.int32)
                    try:
                        mapped = np.asarray(
                            [slot_to_local[int(vertex)] for vertex in face],
                            dtype=np.int32,
                        )
                    except KeyError as error:
                        raise ValueError(
                            "Authored polyhedron face references a padded vertex."
                        ) from error
                    normalized.append(mapped if sign > 0 else mapped[::-1])
                cell_faces[row] = tuple(normalized)
    if any(row is None for row in physical):
        raise ValueError("Owner-local physical vertex source rows are incomplete.")
    cells = tuple(row for row in physical if row is not None)
    if any(row is None for row in exact_physical):
        raise ValueError("Owner-local exact vertex source rows are incomplete.")
    exact_cells = tuple(row for row in exact_physical if row is not None)
    faces = (
        tuple(row for row in cell_faces if row is not None)
        if kind == "polyhedron"
        else None
    )
    return (
        kind,
        cells,
        identifiers,
        owners,
        owned,
        measures,
        faces,
        straight_map,
        exact_cells,
    )


def _cell_coverage_tolerances(
    vertices: tuple[np.ndarray, ...],
    measures: np.ndarray,
    tolerance: float,
    /,
    *,
    dimension: int | None = None,
) -> np.ndarray:
    dimension_ = (
        int(dimension)
        if dimension is not None
        else vertices[0].shape[1]
        if vertices
        else 2
    )
    diameter = np.asarray(
        [np.linalg.norm(np.max(row, axis=0) - np.min(row, axis=0)) for row in vertices],
        dtype=np.float64,
    )
    return (
        tolerance * measures
        + _COVERAGE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * diameter**dimension_
    )


def _view_binding(view: Any, /) -> dict[str, Any]:
    directory = view.owner_directory_data
    return {
        "part_name": view.part_name,
        "distribution_id": view.distribution_id,
        "owner_rank": view.owner_rank,
        "part_id": view.part_id,
        "source_id": view.source_id,
        "source_revision": view.source_revision,
        "source_mesh_id": view.source_mesh_id,
        "source_topology_id": view.source_topology_id,
        "source_geometry_id": view.source_geometry_id,
        "source_map_id": view.source_map_id,
        "source_context_id": view.source_context_id,
        "source_manifest_id": directory.source_manifest_id,
        "directory_group_id": directory.group_id,
        "source_directory_id": directory.source_directory_id,
        "cell_directory_id": directory.cell_directory_id,
        "vertex_directory_id": directory.vertex_directory_id,
        "geometry_layout_id": view.geometry_layout_id,
        "coordinate_contract_id": view.coordinate_contract_id,
        "periodic_topology_id": view.periodic_topology_id,
        "periodic_lift_id": view.periodic_lift_id,
        "periodic_lift_matrix": None
        if view.periodic_lift_matrix is None
        else np.asarray(view.periodic_lift_matrix, dtype=np.float64),
        "local_mesh_id": view.local_mesh_id,
        "local_topology_id": view.local_topology_id,
        "local_geometry_id": view.local_geometry_id,
        "local_view_id": view.local_view_id,
    }


def _validate_accepted_view(view: Any, /) -> None:
    source_value = view.source_value
    if source_value is not None and (
        getattr(source_value, "source_id", None) != view.source_id
        or getattr(source_value, "source_revision", None) != view.source_revision
    ):
        raise ValueError(
            "Owner-local remap source value differs from its exact identity."
        )
    authority = view.source_geometry_authority
    if authority is not None and (
        getattr(authority, "source_id", view.source_id) != view.source_id
        or getattr(authority, "source_revision", view.source_revision)
        != view.source_revision
        or getattr(authority, "source_map_id", None) != view.source_map_id
    ):
        raise ValueError(
            "Owner-local remap geometry authority differs from its exact source identity."
        )


def _validate_transition(transition: Any, source: Any, target: Any, /) -> str | None:
    _validate_accepted_view(source)
    _validate_accepted_view(target)
    if transition is None:
        return None
    from .._cell_geometry_transition_types import CellGeometryTransition
    from .._cell_geometry_validity import cell_geometry_id

    if not isinstance(transition, CellGeometryTransition):
        raise TypeError("transition must be a CellGeometryTransition or None.")
    source_mesh, source_geometry = source.reconstruct_local_geometry()
    target_mesh, target_geometry = target.reconstruct_local_geometry()

    def endpoint_matches(
        endpoint_mesh: Any,
        endpoint_geometry: Any,
        local_mesh: Any,
        local_geometry: Any,
        view: Any,
        /,
    ) -> tuple[bool, bool]:
        local_match = (
            local_mesh is not None
            and local_geometry is not None
            and endpoint_mesh.mesh_id == view.local_mesh_id
            and endpoint_mesh.topology_id == view.local_topology_id
            and cell_geometry_id(endpoint_geometry) == view.local_geometry_id
            and endpoint_mesh.mesh_id == local_mesh.mesh_id
            and endpoint_mesh.topology_id == local_mesh.topology_id
            and cell_geometry_id(endpoint_geometry) == cell_geometry_id(local_geometry)
            and endpoint_geometry.source_map_id == local_geometry.source_map_id
            and endpoint_geometry.source_map_id == view.source_map_id
        )
        global_match = (
            endpoint_mesh.topology_id == view.source_topology_id
            and cell_geometry_id(endpoint_geometry) == view.source_geometry_id
            and endpoint_geometry.source_map_id == view.source_map_id
        )
        return bool(local_match), bool(global_match)

    source_local, source_global = endpoint_matches(
        transition.source_mesh,
        transition.source_geometry,
        source_mesh,
        source_geometry,
        source,
    )
    target_local, target_global = endpoint_matches(
        transition.target_mesh,
        transition.target_geometry,
        target_mesh,
        target_geometry,
        target,
    )
    if not (source_local or source_global) or not (target_local or target_global):
        raise ValueError(
            "CellGeometryTransition does not bind the current accepted global or owner-local endpoints."
        )
    if source_local and target_local:
        transition.require_bound(
            source_mesh, source_geometry, target_mesh, target_geometry
        )
    else:
        transition = CellGeometryTransition.from_stored_fields(transition)
    return transition.transition_id


class OwnerLocalConservativeRemapEvidence(StrictModule, NonTrainableState):
    """Local CSR geometry, owner coverage, and action conservation evidence."""

    status: CommonRefinementStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    coverage_complete: bool = eqx.field(static=True)
    coverage_evidence_id: str = eqx.field(static=True)
    common_refinement_evidence: CommonRefinementEvidence
    curved_integration_evidence: Any | None
    source_coverage_defects: Array
    source_coverage_tolerances: Array
    target_coverage_defects: Array
    target_coverage_tolerances: Array
    source_measure: Array
    target_measure: Array
    uncovered_source_measure: Array
    uncovered_target_measure: Array
    source_double_coverage_measure: Array
    target_double_coverage_measure: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: CommonRefinementStatus,
        reason: str,
        coverage_complete: bool,
        common_refinement_evidence: CommonRefinementEvidence,
        source_coverage_defects: Any,
        source_coverage_tolerances: Any,
        target_coverage_defects: Any,
        target_coverage_tolerances: Any,
        source_measure: float,
        target_measure: float,
        curved_integration_evidence: Any | None = None,
    ) -> None:
        if not isinstance(status, CommonRefinementStatus) or not isinstance(
            common_refinement_evidence, CommonRefinementEvidence
        ):
            raise TypeError("Remap evidence needs common-refinement status and evidence.")
        curved_evidence_id = None
        if isinstance(curved_integration_evidence, tuple):
            from ...geometry._common_root_volume_overlap import (
                CurvedOverlapIntegrationEvidence,
            )

            if not curved_integration_evidence or not all(
                isinstance(value, CurvedOverlapIntegrationEvidence)
                and isinstance(value.evidence_id, str)
                and value.evidence_id
                for value in curved_integration_evidence
            ):
                raise TypeError(
                    "Curved integration evidence tuples must contain registered evidence records."
                )
            curved_evidence_id = canonical_fingerprint(
                {
                    "kind": "owner-local-curved-integration-evidence-set",
                    "children": tuple(
                        sorted(value.evidence_id for value in curved_integration_evidence)
                    ),
                }
            )
        elif curved_integration_evidence is not None:
            curved_evidence_id = getattr(curved_integration_evidence, "evidence_id", None)
            if not isinstance(curved_evidence_id, str) or not curved_evidence_id:
                raise TypeError(
                    "Curved remap evidence must expose a non-empty evidence ID."
                )
        source_defects = np.asarray(source_coverage_defects, dtype=np.float64)
        source_limits = np.asarray(source_coverage_tolerances, dtype=np.float64)
        target_defects = np.asarray(target_coverage_defects, dtype=np.float64)
        target_limits = np.asarray(target_coverage_tolerances, dtype=np.float64)
        if (
            source_defects.shape != source_limits.shape
            or target_defects.shape != target_limits.shape
        ):
            raise ValueError("Owner-local coverage defects and tolerances must align.")
        self.status = status
        self.reason = str(reason)
        self.coverage_complete = bool(coverage_complete)
        self.coverage_evidence_id = canonical_fingerprint(
            {
                "kind": "owner-local-conservative-coverage",
                "native": common_refinement_evidence.evidence_id,
                "source_defects": array_tree_fingerprint(source_defects),
                "source_limits": array_tree_fingerprint(source_limits),
                "target_defects": array_tree_fingerprint(target_defects),
                "target_limits": array_tree_fingerprint(target_limits),
                "complete": bool(coverage_complete),
                "curved_integration": curved_evidence_id,
            }
        )
        self.common_refinement_evidence = common_refinement_evidence
        self.curved_integration_evidence = curved_integration_evidence
        self.source_coverage_defects = jnp.asarray(source_defects)
        self.source_coverage_tolerances = jnp.asarray(source_limits)
        self.target_coverage_defects = jnp.asarray(target_defects)
        self.target_coverage_tolerances = jnp.asarray(target_limits)
        self.source_measure = jnp.asarray(source_measure, dtype=jnp.float64)
        self.target_measure = jnp.asarray(target_measure, dtype=jnp.float64)
        self.uncovered_source_measure = jnp.asarray(
            np.sum(np.maximum(-source_defects, 0.0)), dtype=jnp.float64
        )
        self.uncovered_target_measure = jnp.asarray(
            np.sum(np.maximum(-target_defects, 0.0)), dtype=jnp.float64
        )
        self.source_double_coverage_measure = jnp.asarray(
            np.sum(np.maximum(source_defects, 0.0)), dtype=jnp.float64
        )
        self.target_double_coverage_measure = jnp.asarray(
            np.sum(np.maximum(target_defects, 0.0)), dtype=jnp.float64
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "owner-local-conservative-remap-evidence",
                "status": int(status),
                "reason": self.reason,
                "coverage": self.coverage_evidence_id,
                "source_measure": float(source_measure),
                "target_measure": float(target_measure),
            }
        )


class OwnerLocalConservativeRemapResult(StrictModule):
    """Owner-local target values with fresh target ghosts and conservation ledger."""

    owned_values: Array
    local_values: Array
    owned_target_cell_ids: Array
    local_target_cell_ids: Array
    quantity: str = eqx.field(static=True)
    source_integral: Array
    target_integral: Array
    conservation_residual: Array
    evidence_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        owned_values: Any,
        local_values: Any,
        owned_target_cell_ids: Any,
        local_target_cell_ids: Any,
        quantity: str,
        source_integral: Any,
        target_integral: Any,
        evidence_id: str,
        /,
    ) -> None:
        owned = jnp.asarray(owned_values)
        local = jnp.asarray(local_values)
        ids = np.asarray(owned_target_cell_ids, dtype=np.int64)
        local_ids = np.asarray(local_target_cell_ids, dtype=np.int64)
        kind = str(quantity)
        source_total, target_total = (
            jnp.asarray(source_integral),
            jnp.asarray(target_integral),
        )
        if (
            owned.ndim == 0
            or local.ndim != owned.ndim
            or owned.shape[1:] != local.shape[1:]
            or owned.shape[0] != ids.size
            or local.shape[0] != local_ids.size
            or kind not in ("average", "extensive")
            or source_total.shape != target_total.shape
            or not evidence_id
        ):
            raise ValueError("Owner-local remap action result arrays are malformed.")
        residual = target_total - source_total
        self.owned_values = owned
        self.local_values = local
        self.owned_target_cell_ids = jnp.asarray(ids)
        self.local_target_cell_ids = jnp.asarray(local_ids)
        self.quantity = kind
        self.source_integral = source_total
        self.target_integral = target_total
        self.conservation_residual = residual
        self.evidence_id = str(evidence_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "owner-local-conservative-remap-action",
                "quantity": kind,
                "target_ids": array_tree_fingerprint(ids),
                "local_target_ids": array_tree_fingerprint(local_ids),
                "source_integral": array_tree_fingerprint(np.asarray(source_total)),
                "target_integral": array_tree_fingerprint(np.asarray(target_total)),
                "evidence": self.evidence_id,
            }
        )


class PreparedOwnerLocalConservativeRemap(StrictModule, NonTrainableState):
    """Owner-routed physical overlap CSR for one local source/target view pair.

    CSR target rows are owned by this process. Every overlap retains its stable
    source cell ID and owner rank. Forward actions request only source values used
    by this CSR; target ghost values are refreshed from their actual owners.
    """

    group_id: str = eqx.field(static=True)
    group_process_indices: tuple[int, ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    local_rank: int = eqx.field(static=True)
    source_bindings: Any
    target_bindings: Any
    transition_id: str | None = eqx.field(static=True)
    advance_id: str | None = eqx.field(static=True)
    source_local_cell_ids: Array
    source_local_cell_owner_ranks: Array
    source_local_cell_owned: Array
    source_cell_ids: Array
    source_cell_owner_ranks: Array
    source_volumes: Array
    target_local_cell_ids: Array
    target_local_cell_owner_ranks: Array
    target_local_cell_owned: Array
    target_cell_ids: Array
    target_cell_owner_ranks: Array
    target_volumes: Array
    target_owned_view_rows: Array
    target_ghost_view_rows: Array
    target_ghost_owners: Array
    target_ghost_slots: Array
    target_ghost_request_ids: Array
    target_ghost_request_valid: Array
    target_ghost_response_rows: Array
    target_offsets: Array
    target_routes: Array
    overlap_source_cell_ids: Array
    overlap_source_owner_ranks: Array
    overlap_source_periodic_cell_ids: Array
    overlap_source_cell_windings: Array
    overlap_source_measures: Array
    overlap_target_cell_ids: Array
    overlap_measures: Array
    overlap_first_moments: Array
    overlap_second_moments: Array
    overlap_request_slots: Array
    source_request_ids: Array
    source_request_valid: Array
    source_response_rows: Array
    source_request_capacity: int = eqx.field(static=True)
    target_ghost_capacity: int = eqx.field(static=True)
    source_cell_kind: str = eqx.field(static=True)
    target_cell_kind: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    require_complete: bool = eqx.field(static=True)
    status: CommonRefinementStatus = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    source_geometry_evidence_id: str = eqx.field(static=True)
    target_geometry_evidence_id: str = eqx.field(static=True)
    evidence: OwnerLocalConservativeRemapEvidence
    remap_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        axis_name: str,
        local_rank: int,
        transition_id: str | None,
        advance_id: str | None,
        source_kind: str,
        target_kind: str,
        source_local_cell_ids: Any,
        source_local_cell_owner_ranks: Any,
        source_local_cell_owned: Any,
        source_cell_ids: Any,
        source_volumes: Any,
        target_local_cell_ids: Any,
        target_local_cell_owner_ranks: Any,
        target_local_cell_owned: Any,
        target_cell_ids: Any,
        target_volumes: Any,
        target_owned_view_rows: Any,
        target_ghost_view_rows: Any,
        target_ghost_owners: Any,
        target_ghost_slots: Any,
        target_ghost_request_ids: Any,
        target_ghost_request_valid: Any,
        target_ghost_response_rows: Any,
        refinement: PreparedCommonRefinement,
        source_candidate_ids: Any,
        source_candidate_owners: Any,
        source_candidate_periodic_ids: Any,
        source_candidate_windings: Any,
        source_candidate_measures: Any,
        overlap_request_slots: Any,
        source_request_ids: Any,
        source_request_valid: Any,
        source_response_rows: Any,
        source_request_capacity: int,
        target_ghost_capacity: int,
        source_coverage_defects: Any,
        source_coverage_tolerances: Any,
        target_coverage_defects: Any,
        target_coverage_tolerances: Any,
        tolerance: float,
        require_complete: bool,
        coverage_complete: bool,
        status: CommonRefinementStatus,
        reason: str,
        source_geometry_evidence: CommonRefinementEvidence,
        target_geometry_evidence: CommonRefinementEvidence,
        source_measure: float,
        target_measure: float,
        curved_integration_evidence: Any | None = None,
        target_fragment_routes: Any | None = None,
    ) -> None:
        fragment_routes = (
            np.arange(refinement.target_cell_count, dtype=np.int32)
            if target_fragment_routes is None
            else np.asarray(target_fragment_routes, dtype=np.int32)
        )
        target_fragment_ids = np.asarray(refinement.target_cells, dtype=np.int32)
        source_routes = np.asarray(refinement.source_cells, dtype=np.int32)
        if (
            fragment_routes.shape != (refinement.target_cell_count,)
            or np.any(fragment_routes < 0)
            or np.any(fragment_routes >= len(target_cell_ids))
            or np.any(target_fragment_ids < 0)
            or np.any(target_fragment_ids >= refinement.target_cell_count)
        ):
            raise ValueError("Native target-fragment routes are malformed.")
        target_routes = fragment_routes[target_fragment_ids]
        if (
            target_routes.shape != source_routes.shape
            or target_routes.size != refinement.volumes.shape[0]
            or np.any(source_routes < 0)
            or np.any(source_routes >= len(source_candidate_ids))
            or np.any(target_routes < 0)
            or np.any(target_routes >= len(target_cell_ids))
        ):
            raise ValueError("Native owner-local refinement routes are malformed.")
        candidate_ids = np.asarray(source_candidate_ids, dtype=np.int64)
        candidate_owners = np.asarray(source_candidate_owners, dtype=np.int32)
        candidate_periodic_ids = np.asarray(source_candidate_periodic_ids, dtype=np.int64)
        candidate_windings = np.asarray(source_candidate_windings, dtype=np.int64)
        candidate_measures = np.asarray(source_candidate_measures, dtype=np.float64)
        if (
            candidate_owners.shape != candidate_ids.shape
            or candidate_periodic_ids.shape != candidate_ids.shape
            or candidate_windings.ndim != 2
            or candidate_windings.shape[0] != candidate_ids.size
            or candidate_measures.shape != candidate_ids.shape
        ):
            raise ValueError("Owner-local source candidate identities are malformed.")
        overlap_source_ids = candidate_ids[source_routes]
        overlap_source_owners = candidate_owners[source_routes]
        overlap_target_ids = np.asarray(target_cell_ids, dtype=np.int64)[target_routes]
        overlap_request_slot_array = np.asarray(overlap_request_slots, dtype=np.int32)
        if overlap_request_slot_array.shape != overlap_source_ids.shape:
            raise ValueError("Owner-local overlap value-request slots are malformed.")
        refinement_offsets = np.asarray(refinement.target_offsets, dtype=np.int32)
        measures = np.asarray(refinement.volumes, dtype=np.float64)
        if (
            refinement_offsets.shape != (refinement.target_cell_count + 1,)
            or refinement_offsets[-1] != measures.size
        ):
            raise ValueError("Native fragment overlap CSR is malformed.")
        entry_order = np.lexsort((overlap_source_ids, target_routes))
        source_routes = source_routes[entry_order]
        target_routes = target_routes[entry_order]
        overlap_source_ids = overlap_source_ids[entry_order]
        overlap_source_owners = overlap_source_owners[entry_order]
        overlap_target_ids = overlap_target_ids[entry_order]
        overlap_request_slot_array = overlap_request_slot_array[entry_order]
        measures = measures[entry_order]
        first_moments = np.asarray(refinement.first_moments, dtype=np.float64)[
            entry_order
        ]
        second_moments = (
            None
            if refinement.second_moments is None
            else np.asarray(refinement.second_moments, dtype=np.float64)[entry_order]
        )
        offsets = np.zeros((len(target_cell_ids) + 1,), dtype=np.int32)
        np.cumsum(
            np.bincount(target_routes, minlength=len(target_cell_ids)),
            out=offsets[1:],
        )
        source_binding, target_binding = (
            _view_binding(source_view),
            _view_binding(target_view),
        )
        arrays = (
            source_local_cell_ids,
            source_local_cell_owner_ranks,
            source_local_cell_owned,
            source_cell_ids,
            source_volumes,
            target_local_cell_ids,
            target_local_cell_owner_ranks,
            target_local_cell_owned,
            target_cell_ids,
            target_volumes,
            target_owned_view_rows,
            target_ghost_view_rows,
            target_ghost_owners,
            target_ghost_slots,
            target_ghost_request_ids,
            target_ghost_request_valid,
            target_ghost_response_rows,
            source_request_ids,
            source_request_valid,
            source_response_rows,
        )
        source_local_ids = np.asarray(source_local_cell_ids, dtype=np.int64)
        source_local_owners = np.asarray(source_local_cell_owner_ranks, dtype=np.int32)
        source_local_owned_mask = np.asarray(source_local_cell_owned, dtype=np.bool_)
        source_owned_ids = np.asarray(source_cell_ids, dtype=np.int64)
        source_measure_array = np.asarray(source_volumes, dtype=np.float64)
        target_local_ids = np.asarray(target_local_cell_ids, dtype=np.int64)
        target_local_owners = np.asarray(target_local_cell_owner_ranks, dtype=np.int32)
        target_local_owned_mask = np.asarray(target_local_cell_owned, dtype=np.bool_)
        target_owned_ids = np.asarray(target_cell_ids, dtype=np.int64)
        target_measure_array = np.asarray(target_volumes, dtype=np.float64)
        target_view_rows = np.asarray(target_owned_view_rows, dtype=np.int32)
        ghost_view_rows = np.asarray(target_ghost_view_rows, dtype=np.int32)
        ghost_owners = np.asarray(target_ghost_owners, dtype=np.int32)
        ghost_slots = np.asarray(target_ghost_slots, dtype=np.int32)
        ghost_ids = np.asarray(target_ghost_request_ids, dtype=np.int64)
        ghost_valid = np.asarray(target_ghost_request_valid, dtype=np.bool_)
        ghost_responses = np.asarray(target_ghost_response_rows, dtype=np.int32)
        req_ids = np.asarray(source_request_ids, dtype=np.int64)
        req_valid = np.asarray(source_request_valid, dtype=np.bool_)
        source_responses = np.asarray(source_response_rows, dtype=np.int32)
        if (
            source_local_ids.shape != source_local_owners.shape
            or source_local_ids.shape != source_local_owned_mask.shape
            or source_owned_ids.shape != source_measure_array.shape
            or target_local_ids.shape != target_local_owners.shape
            or target_local_ids.shape != target_local_owned_mask.shape
            or target_owned_ids.shape != target_measure_array.shape
            or target_view_rows.shape != target_owned_ids.shape
            or ghost_view_rows.shape != ghost_owners.shape
            or ghost_view_rows.shape != ghost_slots.shape
            or ghost_ids.shape != ghost_valid.shape
            or ghost_responses.shape != ghost_valid.shape
            or req_ids.shape != req_valid.shape
            or req_ids.shape != source_responses.shape
            or len(arrays) != 20
        ):
            raise ValueError(
                "Owner-local remap row identities or communication plan are malformed."
            )
        evidence = OwnerLocalConservativeRemapEvidence(
            status=status,
            reason=reason,
            coverage_complete=coverage_complete,
            curved_integration_evidence=curved_integration_evidence,
            common_refinement_evidence=refinement.evidence,
            source_coverage_defects=source_coverage_defects,
            source_coverage_tolerances=source_coverage_tolerances,
            target_coverage_defects=target_coverage_defects,
            target_coverage_tolerances=target_coverage_tolerances,
            source_measure=source_measure,
            target_measure=target_measure,
        )
        self.group_id = execution_group.spec.group_id
        self.group_process_indices = tuple(execution_group.spec.process_indices)
        self.axis_name = str(axis_name)
        self.local_rank = int(local_rank)
        self.source_bindings = source_binding
        self.target_bindings = target_binding
        self.transition_id = transition_id
        self.advance_id = advance_id
        self.source_local_cell_ids = jnp.asarray(source_local_ids)
        self.source_local_cell_owner_ranks = jnp.asarray(source_local_owners)
        self.source_local_cell_owned = jnp.asarray(source_local_owned_mask)
        self.source_cell_ids = jnp.asarray(source_owned_ids)
        self.source_cell_owner_ranks = jnp.full(
            source_owned_ids.shape, local_rank, dtype=jnp.int32
        )
        self.source_volumes = jnp.asarray(source_measure_array)
        self.target_local_cell_ids = jnp.asarray(target_local_ids)
        self.target_local_cell_owner_ranks = jnp.asarray(target_local_owners)
        self.target_local_cell_owned = jnp.asarray(target_local_owned_mask)
        self.target_cell_ids = jnp.asarray(target_owned_ids)
        self.target_cell_owner_ranks = jnp.full(
            target_owned_ids.shape, local_rank, dtype=jnp.int32
        )
        self.target_volumes = jnp.asarray(target_measure_array)
        self.target_owned_view_rows = jnp.asarray(target_view_rows)
        self.target_ghost_view_rows = jnp.asarray(ghost_view_rows)
        self.target_ghost_owners = jnp.asarray(ghost_owners)
        self.target_ghost_slots = jnp.asarray(ghost_slots)
        self.target_ghost_request_ids = jnp.asarray(ghost_ids)
        self.target_ghost_request_valid = jnp.asarray(ghost_valid)
        self.target_ghost_response_rows = jnp.asarray(ghost_responses)
        self.target_offsets = jnp.asarray(offsets)
        self.target_routes = jnp.asarray(target_routes)
        self.overlap_source_cell_ids = jnp.asarray(overlap_source_ids)
        self.overlap_source_owner_ranks = jnp.asarray(overlap_source_owners)
        self.overlap_source_periodic_cell_ids = jnp.asarray(
            candidate_periodic_ids[source_routes]
        )
        self.overlap_source_cell_windings = jnp.asarray(candidate_windings[source_routes])
        self.overlap_source_measures = jnp.asarray(candidate_measures[source_routes])
        self.overlap_target_cell_ids = jnp.asarray(overlap_target_ids)
        self.overlap_measures = jnp.asarray(measures)
        self.overlap_first_moments = jnp.asarray(first_moments)
        if second_moments is None:
            raise ValueError(
                "Owner-local conservative remapping requires overlap second moments."
            )
        self.overlap_second_moments = jnp.asarray(second_moments)
        self.overlap_request_slots = jnp.asarray(overlap_request_slot_array)
        self.source_request_ids = jnp.asarray(req_ids)
        self.source_request_valid = jnp.asarray(req_valid)
        self.source_response_rows = jnp.asarray(source_responses)
        self.source_request_capacity = int(source_request_capacity)
        self.target_ghost_capacity = int(target_ghost_capacity)
        self.source_cell_kind = str(source_kind)
        self.target_cell_kind = str(target_kind)
        self.tolerance = float(tolerance)
        self.require_complete = bool(require_complete)
        self.status = status
        self.refinement_id = refinement.refinement_id
        self.source_geometry_evidence_id = source_geometry_evidence.evidence_id
        self.target_geometry_evidence_id = target_geometry_evidence.evidence_id
        self.evidence = evidence
        self.remap_id = canonical_fingerprint(
            {
                "kind": "prepared-owner-local-conservative-remap",
                "group": self.group_id,
                "rank": self.local_rank,
                "source": source_binding,
                "target": target_binding,
                "transition": transition_id,
                "advance": advance_id,
                "refinement": refinement.refinement_id,
                "source_cell_kind": source_kind,
                "target_cell_kind": target_kind,
                "tolerance": float(tolerance),
                "require_complete": bool(require_complete),
                "status": int(status),
                "source_geometry_evidence": source_geometry_evidence.evidence_id,
                "target_geometry_evidence": target_geometry_evidence.evidence_id,
                "source_local_cell_ids": array_tree_fingerprint(source_local_ids),
                "source_local_cell_owners": array_tree_fingerprint(source_local_owners),
                "source_local_cell_owned": array_tree_fingerprint(
                    source_local_owned_mask
                ),
                "source_volumes": array_tree_fingerprint(source_measure_array),
                "target_local_cell_ids": array_tree_fingerprint(target_local_ids),
                "target_local_cell_owners": array_tree_fingerprint(target_local_owners),
                "target_local_cell_owned": array_tree_fingerprint(
                    target_local_owned_mask
                ),
                "target_volumes": array_tree_fingerprint(target_measure_array),
                "target_owned_view_rows": array_tree_fingerprint(target_view_rows),
                "target_ghost_view_rows": array_tree_fingerprint(ghost_view_rows),
                "target_ghost_owners": array_tree_fingerprint(ghost_owners),
                "target_ghost_slots": array_tree_fingerprint(ghost_slots),
                "target_ghost_request_ids": array_tree_fingerprint(ghost_ids),
                "target_ghost_request_valid": array_tree_fingerprint(ghost_valid),
                "target_ghost_response_rows": array_tree_fingerprint(ghost_responses),
                "overlap_source_periodic_ids": array_tree_fingerprint(
                    candidate_periodic_ids[source_routes]
                ),
                "overlap_source_windings": array_tree_fingerprint(
                    candidate_windings[source_routes]
                ),
                "overlap_first_moments": array_tree_fingerprint(
                    np.asarray(refinement.first_moments)
                ),
                "overlap_second_moments": array_tree_fingerprint(
                    np.asarray(refinement.second_moments)
                ),
                "overlap_request_slots": array_tree_fingerprint(
                    overlap_request_slot_array
                ),
                "source_request_ids": array_tree_fingerprint(req_ids),
                "source_request_valid": array_tree_fingerprint(req_valid),
                "source_response_rows": array_tree_fingerprint(source_responses),
                "source_cells": array_tree_fingerprint(source_owned_ids),
                "target_cells": array_tree_fingerprint(target_owned_ids),
                "target_offsets": array_tree_fingerprint(offsets),
                "overlap_sources": array_tree_fingerprint(overlap_source_ids),
                "overlap_source_owners": array_tree_fingerprint(overlap_source_owners),
                "overlap_targets": array_tree_fingerprint(overlap_target_ids),
                "overlap_measures": array_tree_fingerprint(measures),
                "evidence": evidence.evidence_id,
            }
        )

    @classmethod
    def from_stored_fields(
        cls,
        stored: PreparedOwnerLocalConservativeRemap,
        source_view: OwnerLocalCellGeometry,
        target_view: OwnerLocalCellGeometry,
        execution_group: ExecutionGroup,
        /,
        *,
        transition: CellGeometryTransition | None = None,
        advance_id: str | None = None,
        axis_name: str | None = None,
    ) -> PreparedOwnerLocalConservativeRemap:
        """Cold-replay a stored owner remap from current accepted view owners.

        Stored CSR is evidence to compare, never a fallback for missing current
        geometry or routes.
        """
        axis = (
            str(axis_name)
            if axis_name is not None
            else str(getattr(stored, "axis_name", "owner_local_remap"))
        )
        mesh, rank, owner_count = _group_layout(execution_group, axis)
        local_error: Exception | None = None
        try:
            transition_id = None if transition is None else transition.transition_id
            if (
                not isinstance(stored, cls)
                or execution_group.spec.group_id != stored.group_id
                or tuple(execution_group.spec.process_indices)
                != stored.group_process_indices
                or axis != stored.axis_name
                or rank != stored.local_rank
                or transition_id != stored.transition_id
                or advance_id != stored.advance_id
                or canonical_fingerprint(_view_binding(source_view))
                != canonical_fingerprint(stored.source_bindings)
                or canonical_fingerprint(_view_binding(target_view))
                != canonical_fingerprint(stored.target_bindings)
            ):
                raise ValueError(
                    "Stored owner-local remap no longer binds the current views, "
                    "owner group, or motion transition."
                )
        except Exception as caught:
            local_error = caught
        _synchronize_error(local_error, mesh, axis, owner_count, "stored binding replay")
        rebuilt = prepare_owner_local_conservative_remap(
            source_view,
            target_view,
            execution_group,
            transition=transition,
            advance_id=advance_id,
            axis_name=axis,
            tolerance=stored.tolerance,
            require_complete=stored.require_complete,
        )
        replay_error = (
            None
            if rebuilt.remap_id == stored.remap_id
            else ValueError(
                "Stored owner-local remap differs from its native geometry replay."
            )
        )
        _synchronize_error(
            replay_error, mesh, axis, owner_count, "stored geometry replay"
        )
        return rebuilt

    def _validate_binding(
        self,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        axis_name: str | None,
        /,
    ) -> tuple[Mesh, int, int, str]:
        axis = self.axis_name if axis_name is None else str(axis_name)
        mesh, rank, owner_count = _group_layout(execution_group, axis)
        local_error: Exception | None = None
        try:
            if (
                execution_group.spec.group_id != self.group_id
                or tuple(execution_group.spec.process_indices)
                != self.group_process_indices
                or axis != self.axis_name
                or rank != self.local_rank
                or canonical_fingerprint(_view_binding(source_view))
                != canonical_fingerprint(self.source_bindings)
                or canonical_fingerprint(_view_binding(target_view))
                != canonical_fingerprint(self.target_bindings)
            ):
                raise ValueError(
                    "Execution group, owner-local view authority, geometry, map, or context is stale."
                )
            _validate_accepted_view(source_view)
            _validate_accepted_view(target_view)
        except Exception as caught:
            local_error = caught
        _synchronize_error(local_error, mesh, axis, owner_count, "action binding")
        source_view.require_owner_directory_current(
            execution_group, axis_name=f"{axis}_source_directory"
        )
        target_view.require_owner_directory_current(
            execution_group, axis_name=f"{axis}_target_directory"
        )
        return mesh, rank, owner_count, axis

    def _apply_linear(
        self,
        source_values: Any,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        quantity: str,
        /,
        *,
        axis_name: str | None = None,
    ) -> OwnerLocalConservativeRemapResult:
        mesh, rank, owner_count, axis = self._validate_binding(
            source_view, target_view, execution_group, axis_name
        )
        values = jnp.asarray(source_values)
        shape_error = None
        if (
            values.ndim == 0
            or values.shape[0] != self.source_local_cell_ids.size
            or not jnp.issubdtype(values.dtype, jnp.inexact)
        ):
            shape_error = ValueError(
                "Source values must be floating or complex and align with this owner-local source view."
            )
        _synchronize_error(shape_error, mesh, axis, owner_count, "field layout")
        nonfinite_count = _group_sum(
            jnp.asarray(jnp.any(~jnp.isfinite(values)), dtype=jnp.int32),
            mesh,
            axis,
            owner_count,
        )
        nonfinite_error = (
            ValueError("Conservative remap source values must be finite.")
            if int(np.asarray(jax.device_get(nonfinite_count))) > 0
            else None
        )
        _synchronize_error(nonfinite_error, mesh, axis, owner_count, "field values")
        local_error: Exception | None = None
        if not bool(np.asarray(self.evidence.coverage_complete)):
            local_error = ValueError(
                "Owner-local conservative transport refuses incomplete source/target coverage."
            )
        elif self.status is not CommonRefinementStatus.SUCCESS:
            local_error = ValueError(
                f"Owner-local common refinement failed with {self.status.name}: {self.evidence.reason}"
            )
        _synchronize_error(local_error, mesh, axis, owner_count, "field action")

        trailing = values.shape[1:]
        source_response_rows = np.asarray(self.source_response_rows, dtype=np.int32)
        source_response_valid = source_response_rows >= 0
        safe_source_rows = jnp.asarray(
            np.maximum(source_response_rows, 0), dtype=jnp.int32
        )
        if values.shape[0]:
            source_packet = values[safe_source_rows]
        else:
            source_packet = jnp.zeros(
                source_response_rows.shape + trailing, dtype=values.dtype
            )
        source_mask = jnp.asarray(source_response_valid).reshape(
            source_response_valid.shape + (1,) * len(trailing)
        )
        source_packet = jnp.where(
            source_mask, source_packet, jnp.zeros((), dtype=values.dtype)
        )

        requested_values = _all_to_all_local(source_packet, mesh, axis, owner_count)
        overlap_owner = self.overlap_source_owner_ranks.astype(jnp.int32)
        overlap_slot = self.overlap_request_slots.astype(jnp.int32)
        overlap_values = requested_values[overlap_owner, overlap_slot]
        overlap_measure = self.overlap_measures.astype(values.dtype).reshape(
            (-1,) + (1,) * len(trailing)
        )
        if quantity == "average":
            contributions = overlap_values * overlap_measure
        else:
            source_volume = self.overlap_source_measures.astype(values.dtype).reshape(
                (-1,) + (1,) * len(trailing)
            )
            contributions = overlap_values * (overlap_measure / source_volume)
        owned = (
            jnp.zeros((self.target_cell_ids.size,) + trailing, dtype=values.dtype)
            .at[self.target_routes]
            .add(contributions)
        )
        if quantity == "average":
            target_volume = self.target_volumes.astype(values.dtype).reshape(
                (-1,) + (1,) * len(trailing)
            )
            owned = owned / target_volume
        ghost_response_rows = np.asarray(self.target_ghost_response_rows, dtype=np.int32)

        ghost_response_valid = ghost_response_rows >= 0
        safe_ghost_rows = jnp.asarray(np.maximum(ghost_response_rows, 0), dtype=jnp.int32)
        if owned.shape[0]:
            ghost_packet = owned[safe_ghost_rows]
        else:
            ghost_packet = jnp.zeros(
                ghost_response_rows.shape + trailing, dtype=owned.dtype
            )
        ghost_mask = jnp.asarray(ghost_response_valid).reshape(
            ghost_response_valid.shape + (1,) * len(trailing)
        )
        ghost_packet = jnp.where(
            ghost_mask, ghost_packet, jnp.zeros((), dtype=owned.dtype)
        )
        ghost_received = _all_to_all_local(ghost_packet, mesh, axis, owner_count)
        local_target = jnp.zeros(
            (self.target_local_cell_ids.size,) + trailing, dtype=owned.dtype
        )
        local_target = local_target.at[self.target_owned_view_rows].set(owned)
        if self.target_ghost_view_rows.size:
            ghost_values = ghost_received[
                self.target_ghost_owners.astype(jnp.int32),
                self.target_ghost_slots.astype(jnp.int32),
            ]
            local_target = local_target.at[self.target_ghost_view_rows].set(ghost_values)

        owned_mask = self.source_local_cell_owned.reshape(
            self.source_local_cell_owned.shape + (1,) * len(trailing)
        )
        if quantity == "average":
            source_terms = values * self._source_local_volumes(
                source_view, values.dtype
            ).reshape((-1,) + (1,) * len(trailing))
            source_total_local = jnp.sum(jnp.where(owned_mask, source_terms, 0.0), axis=0)
            target_total_local = jnp.sum(
                owned
                * self.target_volumes.astype(owned.dtype).reshape(
                    (-1,) + (1,) * len(trailing)
                ),
                axis=0,
            )
        else:
            source_total_local = jnp.sum(jnp.where(owned_mask, values, 0.0), axis=0)
            target_total_local = jnp.sum(owned, axis=0)
        source_total = _group_sum(source_total_local, mesh, axis, owner_count)
        target_total = _group_sum(target_total_local, mesh, axis, owner_count)
        return OwnerLocalConservativeRemapResult(
            owned,
            local_target,
            self.target_cell_ids,
            self.target_local_cell_ids,
            quantity,
            source_total,
            target_total,
            self.evidence.evidence_id,
        )

    def _source_local_volumes(self, source_view: Any, dtype: Any, /) -> Array:
        view_ids = np.asarray(source_view.cell_ids, dtype=np.int64)
        owned_ids = np.asarray(self.source_cell_ids, dtype=np.int64)
        owned_volumes = np.asarray(self.source_volumes, dtype=np.float64)
        volume_by_id = {
            int(identifier): float(volume)
            for identifier, volume in zip(owned_ids, owned_volumes, strict=True)
        }
        volumes = np.asarray(
            [volume_by_id.get(int(identifier), 0.0) for identifier in view_ids],
            dtype=np.float64,
        )
        return jnp.asarray(volumes, dtype=dtype)

    def apply_averages(
        self,
        source_cell_averages: Any,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str | None = None,
    ) -> OwnerLocalConservativeRemapResult:
        """Owner-route source averages and return target averages plus fresh ghosts."""
        return self._apply_linear(
            source_cell_averages,
            source_view,
            target_view,
            execution_group,
            "average",
            axis_name=axis_name,
        )

    def apply_content(
        self,
        source_extensive_content: Any,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str | None = None,
    ) -> OwnerLocalConservativeRemapResult:
        """Owner-route extensive content without converting the target to an average."""
        return self._apply_linear(
            source_extensive_content,
            source_view,
            target_view,
            execution_group,
            "extensive",
            axis_name=axis_name,
        )

    def transpose(
        self,
        target_cotangent: Any,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str | None = None,
    ) -> Array:
        """Apply the raw algebraic transpose of the cell-average CSR map."""
        mesh, rank, owner_count, axis = self._validate_binding(
            source_view, target_view, execution_group, axis_name
        )
        target = jnp.asarray(target_cotangent)
        trailing = target.shape[1:]
        error = None
        if (
            target.ndim == 0
            or target.shape[0] != self.target_cell_ids.size
            or not bool(np.asarray(self.evidence.coverage_complete))
            or self.status is not CommonRefinementStatus.SUCCESS
        ):
            error = ValueError(
                "Raw remap transpose requires a successful complete local map."
            )
        _synchronize_error(error, mesh, axis, owner_count, "raw transpose")
        weights = (
            self.overlap_measures.astype(target.dtype)
            / self.target_volumes.astype(target.dtype)[self.target_routes]
        )
        weights = weights.reshape((-1,) + (1,) * len(trailing))
        contributions = target[self.target_routes] * weights
        packets = (
            jnp.zeros(
                (owner_count, self.source_request_capacity) + trailing, dtype=target.dtype
            )
            .at[
                self.overlap_source_owner_ranks.astype(jnp.int32),
                self.overlap_request_slots.astype(jnp.int32),
            ]
            .add(contributions)
        )
        received = _all_to_all_local(packets, mesh, axis, owner_count)
        return self._assemble_source_rows(received, source_view, target.dtype, trailing)

    def weighted_adjoint(
        self,
        target_cotangent: Any,
        source_view: Any,
        target_view: Any,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str | None = None,
    ) -> Array:
        """Apply ``M_source^-1 R.T M_target`` under FV cell-volume pairings."""
        mesh, rank, owner_count, axis = self._validate_binding(
            source_view, target_view, execution_group, axis_name
        )
        target = jnp.asarray(target_cotangent)
        trailing = target.shape[1:]
        error = None
        if (
            target.ndim == 0
            or target.shape[0] != self.target_cell_ids.size
            or not bool(np.asarray(self.evidence.coverage_complete))
            or self.status is not CommonRefinementStatus.SUCCESS
        ):
            error = ValueError(
                "Weighted remap adjoint requires a successful complete local map."
            )
        _synchronize_error(error, mesh, axis, owner_count, "weighted adjoint")
        source_volume = self.overlap_source_measures.astype(target.dtype)
        weights = (self.overlap_measures.astype(target.dtype) / source_volume).reshape(
            (-1,) + (1,) * len(trailing)
        )
        contributions = target[self.target_routes] * weights
        packets = (
            jnp.zeros(
                (owner_count, self.source_request_capacity) + trailing, dtype=target.dtype
            )
            .at[
                self.overlap_source_owner_ranks.astype(jnp.int32),
                self.overlap_request_slots.astype(jnp.int32),
            ]
            .add(contributions)
        )
        received = _all_to_all_local(packets, mesh, axis, owner_count)
        return self._assemble_source_rows(received, source_view, target.dtype, trailing)

    def _assemble_source_rows(
        self,
        received: np.ndarray,
        source_view: Any,
        dtype: Any,
        trailing: tuple[int, ...],
        /,
    ) -> Array:
        del source_view
        result = jnp.zeros((self.source_local_cell_ids.size,) + trailing, dtype=dtype)
        response_rows = np.asarray(self.source_response_rows, dtype=np.int32)
        response_valid = response_rows >= 0
        packets = jnp.asarray(received, dtype=dtype)
        for requester in range(len(self.group_process_indices)):
            valid = np.flatnonzero(response_valid[requester])
            if valid.size:
                rows = jnp.asarray(response_rows[requester, valid], dtype=jnp.int32)
                result = result.at[rows].add(packets[requester, valid])
        return result


# ---------------------------------------------------------------- preparation


def _periodic_cells(
    view: Any, owned_rows: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    periodic_ids = (
        np.full((owned_rows.size,), -1, dtype=np.int64)
        if view.periodic_cell_ids is None
        else np.asarray(view.periodic_cell_ids, dtype=np.int64)[owned_rows]
    )
    windings = (
        np.empty((owned_rows.size, 0), dtype=np.int64)
        if view.cell_windings is None
        else np.asarray(view.cell_windings, dtype=np.int64)[owned_rows]
    )
    return periodic_ids, windings


def _periodic_overlap_lattice(
    source_view: Any, target_view: Any, dimension: int, /
) -> np.ndarray | None:
    if (
        source_view.periodic_lift_id is None
        or source_view.periodic_lift_id != target_view.periodic_lift_id
        or source_view.periodic_lift_matrix is None
        or target_view.periodic_lift_matrix is None
    ):
        return None
    source_lattice = np.asarray(source_view.periodic_lift_matrix, dtype=np.float64)
    target_lattice = np.asarray(target_view.periodic_lift_matrix, dtype=np.float64)
    if (
        source_lattice.shape != (dimension, dimension)
        or target_lattice.shape != source_lattice.shape
        or not np.array_equal(source_lattice, target_lattice)
        or abs(float(np.linalg.det(source_lattice))) <= np.finfo(np.float64).tiny
    ):
        return None
    return source_lattice


def _periodic_view_origin(
    view: Any, mesh: Mesh, axis: str, owner_count: int, /
) -> np.ndarray | None:
    from .._periodic_cell import PeriodicCell

    dimension = int(view.coordinates.shape[1])
    local_origin = np.zeros((dimension,), dtype=np.float64)
    local_valid = 0.0
    local_error: Exception | None = None
    try:
        local_mesh, _ = view.reconstruct_local_geometry()
        if local_mesh is not None:
            topology = local_mesh.periodic_topology
            if topology is None or not isinstance(topology.cell, PeriodicCell):
                raise ValueError("Periodic owner-local geometry has no PeriodicCell.")
            cell = topology.cell
            lattice = np.asarray(view.periodic_lift_matrix, dtype=np.float64)
            if (
                cell.cell_id != view.periodic_lift_id
                or lattice.shape != (dimension, dimension)
                or not np.array_equal(np.asarray(cell.vectors), lattice)
                or cell.periodic_axes != (True,) * dimension
            ):
                raise ValueError(
                    "Periodic domain origin is not bound to the accepted cell lift."
                )
            local_origin = np.asarray(cell.origin, dtype=np.float64)
            local_valid = 1.0
    except Exception as caught:
        local_error = caught
    _synchronize_error(
        local_error, mesh, axis, owner_count, "periodic domain origin binding"
    )
    local_record = np.concatenate(
        (
            np.asarray((local_valid, float(view.cell_ids.size)), dtype=np.float64),
            local_origin,
        )
    )
    gathered = _all_gather_local(local_record, mesh, axis, owner_count).reshape(
        (owner_count, dimension + 2)
    )
    selected = gathered[gathered[:, 0] == 1.0, 2:]
    if not selected.size:
        if np.any(gathered[:, 1] > 0.0):
            _synchronize_error(
                ValueError("Periodic owner cells lack an exact domain origin."),
                mesh,
                axis,
                owner_count,
                "periodic domain origin gather",
            )
        return None
    origin = selected[0]
    origin_error = (
        ValueError("Periodic owner ranks disagree on their accepted domain origin.")
        if any(not np.array_equal(row, origin) for row in selected[1:])
        else None
    )
    _synchronize_error(
        origin_error, mesh, axis, owner_count, "periodic domain origin agreement"
    )
    return origin


def _clean_clipped_polyhedron(
    vertices: np.ndarray,
    faces: tuple[np.ndarray, ...],
    face_labels: np.ndarray,
    /,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Deduplicate native clip vertices while retaining every labeled face subdivision."""
    points = np.asarray(vertices, dtype=np.float64)
    labels = np.asarray(face_labels, dtype=np.int32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 4
        or np.any(~np.isfinite(points))
        or labels.shape != (len(faces),)
    ):
        raise ValueError("Clipped polyhedron vertices or face labels are malformed.")
    coordinate_rows: dict[tuple[float, float, float], int] = {}
    unique_points: list[np.ndarray] = []
    vertex_map = np.empty((points.shape[0],), dtype=np.int32)
    for row in range(points.shape[0]):
        key = (
            float(points[row, 0]),
            float(points[row, 1]),
            float(points[row, 2]),
        )
        index = coordinate_rows.get(key)
        if index is None:
            index = len(unique_points)
            coordinate_rows[key] = index
            unique_points.append(np.asarray(key, dtype=np.float64))
        vertex_map[row] = index
    unique_coordinates = np.asarray(unique_points, dtype=np.float64)
    if unique_coordinates.shape[0] < 4:
        raise ValueError(
            "Positive-volume clipped image has fewer than four exact vertices."
        )
    output_faces: list[np.ndarray] = []
    output_labels: list[int] = []
    for raw_face, label in zip(faces, labels, strict=True):
        raw = np.asarray(raw_face, dtype=np.int64)
        if raw.ndim != 1 or np.any(raw < 0) or np.any(raw >= points.shape[0]):
            raise ValueError("Native clipped face loop has invalid vertex rows.")
        loop: list[int] = []
        for row in vertex_map[raw]:
            vertex = int(row)
            if not loop or vertex != loop[-1]:
                loop.append(vertex)
        if len(loop) > 1 and loop[0] == loop[-1]:
            loop.pop()
        if len(set(loop)) < 3:
            continue
        output_faces.append(np.asarray(loop, dtype=np.int32))
        output_labels.append(int(label))
    if not output_faces:
        raise ValueError(
            "Native positive-volume clip returned no polygonal face patches."
        )
    used = np.asarray(
        sorted({int(vertex) for face in output_faces for vertex in face}),
        dtype=np.int32,
    )
    compact_rows = {int(vertex): row for row, vertex in enumerate(used.tolist())}
    compact_faces = tuple(
        np.asarray([compact_rows[int(vertex)] for vertex in face], dtype=np.int32)
        for face in output_faces
    )
    return (
        unique_coordinates[used],
        compact_faces,
        np.asarray(output_labels, dtype=np.int32),
    )


def _merge_coplanar_clipped_faces(
    vertices: np.ndarray,
    faces: tuple[np.ndarray, ...],
    face_labels: np.ndarray,
    exact_planes: tuple[tuple[tuple[ExactScalar, ...], ExactScalar], ...],
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    native_measure: float,
    /,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], float, float]:
    """Rebuild a clipped convex shell from exact source/domain halfspaces.

    Native coordinates bind active-plane labels and measure their realization
    error; they never define support planes. Returned vertices are the binary64
    realization of the exact halfspace shell, and the measure is exact for that
    shell. The error bounds its exact-to-realized symmetric difference and
    includes the observed native measure discrepancy.
    """
    from itertools import combinations

    from ...geometry._exact_polyhedral_geometry import projected_convex_hull
    from ...linalg._exact import solve_exact

    points = np.asarray(vertices, dtype=np.float64)
    labels = np.asarray(face_labels, dtype=np.int32)
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 4
        or labels.shape != (len(faces),)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(~np.isfinite(points))
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or not np.isfinite(native_measure)
        or native_measure <= 0.0
    ):
        raise ValueError("Clipped support-plane provenance is malformed.")

    def dot(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...]
    ) -> ExactScalar:
        return sum(
            (left * right for left, right in zip(first, second, strict=True)),
            Fraction(0),
        )

    def cross(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...]
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    def subtract(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...]
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        return (
            first[0] - second[0],
            first[1] - second[1],
            first[2] - second[2],
        )

    def determinant(
        first: tuple[ExactScalar, ...],
        second: tuple[ExactScalar, ...],
        third: tuple[ExactScalar, ...],
    ) -> ExactScalar:
        return dot(first, cross(second, third))

    def plane_key(
        normal: tuple[ExactScalar, ...], offset: ExactScalar
    ) -> tuple[tuple[ExactScalar, ...], int]:
        pivot_axis = next((axis for axis, value in enumerate(normal) if value != 0), None)
        if pivot_axis is None:
            raise ValueError("Clipped support-plane provenance has a zero normal.")
        pivot = normal[pivot_axis]
        orientation = 1 if pivot > 0 else -1
        key = tuple(value / pivot for value in normal) + (offset / pivot,)
        return key, orientation

    # Exact source/domain labels are physical; negative native labels are
    # AABB-shadow provenance and may never define a positive-area shell face.
    plane_normals: dict[tuple[ExactScalar, ...], tuple[ExactScalar, ...]] = {}
    plane_offsets: dict[tuple[ExactScalar, ...], ExactScalar] = {}
    plane_orientations: dict[tuple[ExactScalar, ...], int] = {}
    plane_source_labels: dict[tuple[ExactScalar, ...], set[int]] = {}
    plane_box_labels: dict[tuple[ExactScalar, ...], set[int]] = {}
    label_keys: dict[int, tuple[ExactScalar, ...]] = {}

    def add_plane(
        normal: tuple[ExactScalar, ...],
        offset: ExactScalar,
        label: int,
        *,
        physical: bool,
    ) -> tuple[ExactScalar, ...]:
        key, orientation = plane_key(normal, offset)
        previous = plane_orientations.get(key)
        if previous is not None and previous != orientation:
            raise ValueError(
                "Coincident clipped support planes disagree on outward orientation."
            )
        plane_orientations[key] = orientation
        plane_normals.setdefault(key, normal)
        plane_offsets.setdefault(key, offset)
        label_keys[label] = key
        (plane_source_labels if physical else plane_box_labels).setdefault(
            key, set()
        ).add(label)
        return key

    for label, (normal, offset) in enumerate(exact_planes):
        if len(normal) != 3:
            raise ValueError("Exact source/domain halfspace has the wrong dimension.")
        add_plane(normal, offset, label, physical=True)
    for axis in range(3):
        lower_normal = tuple(
            Fraction(-1 if coordinate == axis else 0) for coordinate in range(3)
        )
        upper_normal = tuple(
            Fraction(1 if coordinate == axis else 0) for coordinate in range(3)
        )
        add_plane(
            lower_normal,
            -Fraction.from_float(float(lower[axis])),
            -1 - 2 * axis,
            physical=False,
        )
        add_plane(
            upper_normal,
            Fraction.from_float(float(upper[axis])),
            -1 - (2 * axis + 1),
            physical=False,
        )

    native_vertex_planes: list[set[tuple[ExactScalar, ...]]] = [set() for _ in points]
    native_face_keys: set[tuple[ExactScalar, ...]] = set()
    eps = np.finfo(np.float64).eps
    for face, raw_label in zip(faces, labels, strict=True):
        label = int(raw_label)
        key = label_keys.get(label)
        if key is None:
            raise ValueError("Native clipped face names an absent exact halfspace.")
        rows = np.asarray(face, dtype=np.int64)
        if (
            rows.ndim != 1
            or rows.size < 3
            or np.any(rows < 0)
            or np.any(rows >= points.shape[0])
        ):
            raise ValueError("Native clipped face label references invalid vertices.")
        native_face_keys.add(key)
        normal = plane_normals[key]
        offset = plane_offsets[key]
        normal_float = np.asarray([float(value) for value in normal], dtype=np.float64)
        normal_scale = float(np.linalg.norm(normal_float))
        if not np.isfinite(normal_scale) or normal_scale == 0.0:
            raise ValueError(
                "Exact clipped support plane cannot be realized in binary64."
            )
        for row in rows:
            point = tuple(Fraction.from_float(float(value)) for value in points[int(row)])
            residual = abs(dot(normal, point) - offset)
            residual_float = float(residual)
            scale = max(
                1.0,
                abs(float(offset)),
                float(np.sum(np.abs(normal_float * points[int(row)]))),
            )
            if (
                not np.isfinite(residual_float)
                or not np.isfinite(scale)
                or residual_float > _COVERAGE_ROUNDOFF_ULPS * eps * scale * normal_scale
            ):
                raise ValueError(
                    "Native clipped face realization disagrees with its "
                    "exact plane label."
                )
            native_vertex_planes[int(row)].add(key)

    def has_rank_three(active_keys: set[tuple[ExactScalar, ...]]) -> bool:
        return any(
            determinant(*(plane_normals[key] for key in triple)) != 0
            for triple in combinations(sorted(active_keys, key=repr), 3)
        )

    native_vertex_has_rank_three = [
        has_rank_three(active_keys) for active_keys in native_vertex_planes
    ]
    exact_vertex_set: set[tuple[ExactScalar, ...]] = set()
    for active_keys, has_full_rank in zip(
        native_vertex_planes, native_vertex_has_rank_three, strict=True
    ):
        ordered_keys = sorted(active_keys, key=repr)
        if not has_full_rank:
            if len(active_keys) >= 3:
                raise ValueError(
                    "Native clipped vertex labels do not determine an exact point."
                )
            continue
        for key_triple in combinations(ordered_keys, 3):
            normals = [plane_normals[key] for key in key_triple]
            if determinant(*normals) == 0:
                continue
            matrix = [[value for value in normal] for normal in normals]
            right = [[plane_offsets[key]] for key in key_triple]
            solution = solve_exact(matrix, right)
            candidate = tuple(solution[axis][0] for axis in range(3))
            if any(
                dot(normal, candidate) > plane_offsets[key]
                for key, normal in plane_normals.items()
            ):
                continue
            exact_vertex_set.add(candidate)
    exact_vertices = tuple(sorted(exact_vertex_set))
    if len(exact_vertices) < 4:
        raise ValueError(
            "Exact clipped halfspaces do not enclose a positive-volume shell."
        )

    scale = max(1.0, float(np.max(np.abs(points))))
    coordinate_limit = _COVERAGE_ROUNDOFF_ULPS * eps * scale
    exact_active: list[set[tuple[ExactScalar, ...]]] = []
    for point in exact_vertices:
        exact_active.append(
            {
                key
                for key, normal in plane_normals.items()
                if dot(normal, point) == plane_offsets[key]
            }
        )
    native_match_counts = [0] * len(exact_vertices)
    native_realization = 0.0
    for row in range(points.shape[0]):
        native_exact = tuple(
            Fraction.from_float(float(points[row, axis])) for axis in range(3)
        )
        candidates = [
            index
            for index, exact_point in enumerate(exact_vertices)
            if native_vertex_has_rank_three[row]
            and native_vertex_planes[row].issubset(exact_active[index])
            and max(
                abs(float(native_exact[axis] - exact_point[axis])) for axis in range(3)
            )
            <= coordinate_limit
        ]
        if len(candidates) > 1:
            raise ValueError(
                "Native clipped vertex realization ambiguously matches exact vertices."
            )
        if candidates:
            index = candidates[0]
            native_match_counts[index] += 1
            native_realization = max(
                native_realization,
                max(
                    abs(float(native_exact[axis] - exact_vertices[index][axis]))
                    for axis in range(3)
                ),
            )
        elif len(native_vertex_planes[row]) >= 3:
            raise ValueError(
                "Native clipped vertex has no exact active-plane realization."
            )
    if any(count == 0 for count in native_match_counts):
        raise ValueError(
            "Exact clipped halfspaces contain a vertex absent from native clipping."
        )

    merged_faces: list[np.ndarray] = []
    for key in sorted(plane_normals, key=repr):
        normal = plane_normals[key]
        face_rows = tuple(
            row
            for row, point in enumerate(exact_vertices)
            if dot(normal, point) == plane_offsets[key]
        )
        if len(face_rows) < 3:
            continue
        first = exact_vertices[face_rows[0]]
        second_row = next(
            (row for row in face_rows[1:] if exact_vertices[row] != first), None
        )
        if second_row is None:
            continue
        first_edge = subtract(exact_vertices[second_row], first)
        has_area = any(
            cross(
                first_edge,
                subtract(exact_vertices[row], first),
            )
            != (Fraction(0), Fraction(0), Fraction(0))
            for row in face_rows[1:]
        )
        if not has_area:
            continue
        if key not in native_face_keys:
            raise ValueError(
                "Exact clipped support plane is absent from native face provenance."
            )
        if not plane_source_labels.get(key):
            if plane_box_labels.get(key):
                raise ValueError(
                    "Native AABB shadow clipping truncated the exact periodic domain."
                )
            raise ValueError("Exact clipped facet lost its source/domain provenance.")
        hull = projected_convex_hull(
            face_rows, exact_vertices, normal, plane_offsets[key]
        )
        loop = tuple(hull)
        polygon_normal = (Fraction(0), Fraction(0), Fraction(0))
        for first_row, second_row in zip(loop, (*loop[1:], loop[0]), strict=True):
            polygon_normal = tuple(
                a + b
                for a, b in zip(
                    polygon_normal,
                    cross(exact_vertices[first_row], exact_vertices[second_row]),
                    strict=True,
                )
            )
        orientation = dot(polygon_normal, normal)
        if orientation == 0:
            raise ValueError("Exact clipped support-plane hull is degenerate.")
        if orientation < 0:
            loop = tuple(reversed(loop))
        merged_faces.append(np.asarray(loop, dtype=np.int32))
    if len(merged_faces) < 4:
        raise ValueError("Exact support-plane consolidation left an open clipped shell.")

    edge_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(merged_faces):
        for row, raw_start in enumerate(face):
            start = int(raw_start)
            stop = int(face[(row + 1) % face.size])
            if start == stop:
                raise ValueError("Exact clipped shell contains a zero-length edge.")
            edge_uses.setdefault((min(start, stop), max(start, stop)), []).append(
                (face_index, start, stop)
            )
    if any(
        len(uses) != 2 or uses[0][1] != uses[1][2] or uses[0][2] != uses[1][1]
        for uses in edge_uses.values()
    ):
        raise ValueError(
            "Exact clipped support-plane hulls do not form one closed oriented shell."
        )

    center = tuple(
        sum((point[axis] for point in exact_vertices), Fraction(0)) / len(exact_vertices)
        for axis in range(3)
    )
    exact_volume = Fraction(0)
    for face in merged_faces:
        points_on_face = [exact_vertices[int(row)] for row in face]
        first = subtract(points_on_face[0], center)
        for row in range(1, len(points_on_face) - 1):
            second = subtract(points_on_face[row], center)
            third = subtract(points_on_face[row + 1], center)
            exact_volume += abs(determinant(first, second, third)) / 6
    if exact_volume <= 0:
        raise ValueError("Exact clipped support planes enclose no positive measure.")

    realized_vertices = np.asarray(
        [[float(value) for value in point] for point in exact_vertices],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(realized_vertices)):
        raise ValueError("Exact clipped vertices cannot be realized in binary64.")
    max_realization = max(
        (
            abs(
                float(
                    Fraction.from_float(float(realized_vertices[row, axis]))
                    - exact_vertices[row][axis]
                )
            )
            for row in range(len(exact_vertices))
            for axis in range(3)
        ),
        default=0.0,
    )
    if max_realization > coordinate_limit:
        raise ValueError("Exact clipped shell exceeds its binary64 realization bound.")
    realized_exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in realized_vertices
    )
    realized_center = tuple(
        sum((point[axis] for point in realized_exact), Fraction(0)) / len(realized_exact)
        for axis in range(3)
    )
    realized_volume = Fraction(0)
    for face in merged_faces:
        points_on_face = [realized_exact[int(row)] for row in face]
        first = subtract(points_on_face[0], realized_center)
        for row in range(1, len(points_on_face) - 1):
            second = subtract(points_on_face[row], realized_center)
            third = subtract(points_on_face[row + 1], realized_center)
            realized_volume += abs(determinant(first, second, third)) / 6
    exact_measure_error = abs(Fraction.from_float(float(native_measure)) - exact_volume)
    realization_measure_error = abs(realized_volume - exact_volume)
    displacement = np.sqrt(3.0) * max_realization
    diameter = (
        float(
            np.linalg.norm(
                np.max(realized_vertices, axis=0) - np.min(realized_vertices, axis=0)
            )
        )
        + 2.0 * displacement
    )
    native_displacement = np.sqrt(3.0) * native_realization
    native_diameter = diameter + 2.0 * native_displacement

    def symmetric_difference_bound(diameter: float, radius: float, /) -> float:
        return 2.0 * (
            np.pi * diameter**2 * radius
            + 2.0 * np.pi * diameter * radius**2
            + (4.0 * np.pi / 3.0) * radius**3
        )

    native_measure_limit = symmetric_difference_bound(
        native_diameter, native_displacement
    ) + _COVERAGE_ROUNDOFF_ULPS * eps * max(1.0, native_diameter**3)
    if exact_measure_error > Fraction.from_float(native_measure_limit):
        raise ValueError(
            "Native clipped measure disagrees with the exact halfspace shell."
        )
    geometric_measure_error_bound = symmetric_difference_bound(diameter, displacement)
    measured_error = float(exact_measure_error + realization_measure_error) + (
        geometric_measure_error_bound
    )
    return (
        realized_vertices,
        tuple(merged_faces),
        float(exact_volume),
        measured_error,
    )


def _periodic_domain_cell_images(
    kind: str,
    cell_ids: np.ndarray,
    cell_owners: np.ndarray,
    periodic_ids: np.ndarray,
    cell_windings: np.ndarray,
    cell_vertices: tuple[np.ndarray, ...],
    exact_cell_vertices: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    cell_faces: tuple[tuple[np.ndarray, ...], ...] | None,
    lattice: np.ndarray,
    origin: np.ndarray | None,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, ...], ...],
    np.ndarray,
]:
    from itertools import product

    from ... import _meshcore
    from ...linalg._exact import solve_exact
    from .._exact_algebraic_geometry import coefficient_endpoint

    identifiers = np.asarray(cell_ids, dtype=np.int64)
    owners = np.asarray(cell_owners, dtype=np.int32)
    periodic = np.asarray(periodic_ids, dtype=np.int64)
    windings = np.asarray(cell_windings, dtype=np.int64)
    matrix = np.asarray(lattice, dtype=np.float64)
    if (
        not identifiers.size
        and not cell_vertices
        and not exact_cell_vertices
        and (cell_faces is None or not cell_faces)
    ):
        empty = np.empty((0,), dtype=np.int64)
        return (
            empty,
            empty.copy(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            (),
            empty.copy(),
            np.empty((0, matrix.shape[0]), dtype=np.int64),
            (),
            np.empty((0,), dtype=np.float64),
        )
    dimension = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if (
        kind not in ("tetrahedron", "polyhedron")
        or matrix.shape != (dimension, dimension)
        or dimension != 3
        or origin is None
        or np.asarray(origin).shape != (dimension,)
        or identifiers.ndim != 1
        or owners.shape != identifiers.shape
        or periodic.shape != identifiers.shape
        or windings.shape != (identifiers.size, dimension)
        or len(cell_vertices) != identifiers.size
        or len(exact_cell_vertices) != identifiers.size
        or cell_faces is not None
        and len(cell_faces) != identifiers.size
        or kind == "polyhedron"
        and cell_faces is None
        or np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(origin))
        or abs(float(np.linalg.det(matrix))) <= np.finfo(np.float64).tiny
    ):
        raise ValueError("Periodic cell-image clipping inputs are malformed.")
    origin_array = np.asarray(origin, dtype=np.float64)
    exact_matrix = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in matrix
    )
    exact_origin = tuple(Fraction.from_float(float(value)) for value in origin_array)

    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    try:
        exact_inverse = tuple(tuple(row) for row in solve_exact(exact_matrix, identity))
    except ValueError:
        raise ValueError("Periodic image lattice is exactly singular.") from None

    from math import ceil, floor

    def exact_floor(value: ExactScalar, /) -> int:
        return floor(value)

    def exact_ceil(value: ExactScalar, /) -> int:
        return ceil(value)

    def exact_dot(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...]
    ) -> ExactScalar:
        return sum(
            (left * right for left, right in zip(first, second, strict=True)),
            Fraction(0),
        )

    def exact_cross(
        first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...]
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    exact_domain_planes: list[tuple[tuple[ExactScalar, ...], ExactScalar]] = []
    for fractional_axis in range(dimension):
        normal = tuple(
            exact_inverse[coordinate][fractional_axis] for coordinate in range(dimension)
        )
        origin_offset = exact_dot(normal, exact_origin)
        exact_domain_planes.append((normal, origin_offset + 1))
        exact_domain_planes.append((tuple(-value for value in normal), -origin_offset))

    exact_domain_corners = tuple(
        tuple(
            exact_origin[axis]
            + sum(
                (
                    Fraction(int(bits[row])) * exact_matrix[row][axis]
                    for row in range(dimension)
                ),
                Fraction(0),
            )
            for axis in range(dimension)
        )
        for bits in product((0, 1), repeat=dimension)
    )
    domain_lower = tuple(
        min(point[axis] for point in exact_domain_corners) for axis in range(dimension)
    )
    domain_upper = tuple(
        max(point[axis] for point in exact_domain_corners) for axis in range(dimension)
    )

    def exact_source_planes(
        points: tuple[tuple[ExactScalar, ...], ...],
        faces: tuple[np.ndarray, ...],
    ) -> tuple[tuple[tuple[ExactScalar, ...], ExactScalar], ...]:
        if len(points) < 4 or any(len(point) != dimension for point in points):
            raise ValueError("Exact periodic source cell vertices are malformed.")
        center = tuple(
            sum((point[axis] for point in points), Fraction(0)) / len(points)
            for axis in range(dimension)
        )
        result = []
        for face in faces:
            rows = np.asarray(face, dtype=np.int64)
            if (
                rows.ndim != 1
                or rows.size < 3
                or np.any(rows < 0)
                or np.any(rows >= len(points))
            ):
                raise ValueError("Exact periodic source face loop is malformed.")
            anchor = points[int(rows[0])]
            normal = None
            for first_row in range(1, rows.size - 1):
                first = tuple(
                    points[int(rows[first_row])][axis] - anchor[axis]
                    for axis in range(dimension)
                )
                for second_row in range(first_row + 1, rows.size):
                    second = tuple(
                        points[int(rows[second_row])][axis] - anchor[axis]
                        for axis in range(dimension)
                    )
                    candidate = exact_cross(first, second)
                    if candidate != (Fraction(0), Fraction(0), Fraction(0)):
                        normal = candidate
                        break
                if normal is not None:
                    break
            if normal is None:
                raise ValueError("Exact periodic source face is degenerate.")
            offset = exact_dot(normal, anchor)
            if any(exact_dot(normal, points[int(row)]) != offset for row in rows):
                raise ValueError("Exact periodic source face is not planar.")
            if exact_dot(normal, center) > offset:
                normal = tuple(-value for value in normal)
                offset = -offset
            if any(exact_dot(normal, point) > offset for point in points):
                raise ValueError("Exact periodic source cell is not convex.")
            result.append((normal, offset))
        return tuple(result)

    faces_by_cell = (
        cell_faces
        if cell_faces is not None
        else tuple(
            tuple(
                np.asarray(face, dtype=np.int32)
                for face in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
            )
            for _ in identifiers
        )
    )
    if kind == "tetrahedron" and any(
        vertices.shape != (4, dimension) for vertices in cell_vertices
    ):
        raise ValueError("Periodic tetrahedra require four physical vertices.")

    image_stable_ids: list[int] = []
    image_owners: list[int] = []
    image_periodic_ids: list[int] = []
    image_windings: list[np.ndarray] = []
    image_exact_planes: list[tuple[tuple[tuple[ExactScalar, ...], ExactScalar], ...]] = []
    image_intersection_lowers: list[tuple[ExactScalar, ...]] = []
    image_intersection_uppers: list[tuple[ExactScalar, ...]] = []
    for (
        identifier,
        owner,
        periodic_id,
        winding,
        vertices,
        exact_points,
        faces,
    ) in zip(
        identifiers,
        owners,
        periodic,
        windings,
        cell_vertices,
        exact_cell_vertices,
        faces_by_cell,
        strict=True,
    ):
        points = np.asarray(vertices, dtype=np.float64)
        if points.shape != (len(exact_points), dimension):
            raise ValueError("Exact periodic source vertices lost their binary rows.")
        source_planes = exact_source_planes(exact_points, faces)
        exact_binary_points = np.asarray(
            [[float(value) for value in point] for point in exact_points],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(exact_binary_points)):
            raise ValueError("Exact periodic source vertices cannot be realized.")
        source_roundoff = (
            _COVERAGE_ROUNDOFF_ULPS
            * np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(points))))
        )
        if np.max(np.abs(exact_binary_points - points)) > source_roundoff:
            raise ValueError(
                "Owner-local source vertices disagree with their exact geometry bank."
            )
        fractional_vertices = []
        for point in exact_points:
            delta = tuple(
                point[coordinate] - exact_origin[coordinate]
                for coordinate in range(dimension)
            )
            fractional_vertices.append(
                tuple(
                    sum(
                        (
                            delta[coordinate] * exact_inverse[coordinate][fractional_axis]
                            for coordinate in range(dimension)
                        ),
                        Fraction(0),
                    )
                    for fractional_axis in range(dimension)
                )
            )
        fractional_lower = tuple(
            min(row[coordinate] for row in fractional_vertices)
            for coordinate in range(dimension)
        )
        fractional_upper = tuple(
            max(row[coordinate] for row in fractional_vertices)
            for coordinate in range(dimension)
        )
        lower_shift = tuple(exact_ceil(-value) for value in fractional_upper)
        upper_shift = tuple(
            exact_floor(Fraction(1) - value) for value in fractional_lower
        )
        if any(
            lower > upper for lower, upper in zip(lower_shift, upper_shift, strict=True)
        ):
            continue
        for shift_values in product(
            *(
                range(lower, upper + 1)
                for lower, upper in zip(lower_shift, upper_shift, strict=True)
            )
        ):
            if any(
                fractional_upper[axis] + shift_values[axis] <= 0
                or fractional_lower[axis] + shift_values[axis] >= 1
                for axis in range(dimension)
            ):
                continue
            shift = np.asarray(shift_values, dtype=np.int64)
            exact_translation = tuple(
                sum(
                    (
                        Fraction(int(shift_values[row])) * exact_matrix[row][axis]
                        for row in range(dimension)
                    ),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            source_lower = tuple(
                min(point[axis] for point in exact_points) + exact_translation[axis]
                for axis in range(dimension)
            )
            source_upper = tuple(
                max(point[axis] for point in exact_points) + exact_translation[axis]
                for axis in range(dimension)
            )
            intersection_lower = tuple(
                max(source_lower[axis], domain_lower[axis]) for axis in range(dimension)
            )
            intersection_upper = tuple(
                min(source_upper[axis], domain_upper[axis]) for axis in range(dimension)
            )
            if any(
                lower >= upper
                for lower, upper in zip(
                    intersection_lower, intersection_upper, strict=True
                )
            ):
                continue
            image_intersection_lowers.append(intersection_lower)
            image_intersection_uppers.append(intersection_upper)
            shifted_planes = tuple(
                (
                    normal,
                    offset + exact_dot(normal, exact_translation),
                )
                for normal, offset in source_planes
            ) + tuple(exact_domain_planes)
            image_stable_ids.append(int(identifier))
            image_owners.append(int(owner))
            image_periodic_ids.append(int(periodic_id))
            image_windings.append(winding + shift)
            image_exact_planes.append(shifted_planes)
    if not image_exact_planes:
        empty = np.empty((0,), dtype=np.int64)
        return (
            empty,
            empty.copy(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            (),
            empty.copy(),
            np.empty((0, dimension), dtype=np.int64),
            (),
            np.empty((0,), dtype=np.float64),
        )

    exact_box_lower = tuple(
        min(bounds[axis] for bounds in image_intersection_lowers)
        for axis in range(dimension)
    )
    exact_box_upper = tuple(
        max(bounds[axis] for bounds in image_intersection_uppers)
        for axis in range(dimension)
    )
    box_lower = np.asarray(
        [coefficient_endpoint(value, upper=False) for value in exact_box_lower],
        dtype=np.float64,
    )
    box_upper = np.asarray(
        [coefficient_endpoint(value, upper=True) for value in exact_box_upper],
        dtype=np.float64,
    )
    plane_capacity = max(len(planes) for planes in image_exact_planes)
    normals = np.zeros(
        (len(image_exact_planes), plane_capacity, dimension), dtype=np.float64
    )
    offsets = np.zeros((len(image_exact_planes), plane_capacity), dtype=np.float64)
    plane_counts = np.zeros((len(image_exact_planes),), dtype=np.int32)
    for row, planes in enumerate(image_exact_planes):
        for plane, (exact_normal, exact_offset) in enumerate(planes):
            largest = max(abs(value) for value in exact_normal)
            if largest == 0:
                raise ValueError("Exact periodic clipping plane has a zero normal.")
            normal = np.asarray(
                [float(value / largest) for value in exact_normal],
                dtype=np.float64,
            )
            length = float(np.linalg.norm(normal))
            if not np.isfinite(length) or length == 0.0:
                raise ValueError("Exact periodic clipping plane cannot be normalized.")
            normals[row, plane] = normal / length
            offsets[row, plane] = float(exact_offset / largest) / length
            if not np.isfinite(offsets[row, plane]):
                raise ValueError("Exact periodic clipping offset is not finite.")
        plane_counts[row] = len(planes)

    (
        clipped_vertices,
        vertex_counts,
        face_offsets,
        clipped_face_labels,
        face_vertices,
        face_counts,
        clipped_measures,
        _,
        statuses,
    ) = _meshcore.clip_box_halfspaces(
        box_lower,
        box_upper,
        normals,
        offsets,
        plane_counts,
    )
    output_rows = []
    output_vertices = []
    output_faces = []
    output_measures = []
    output_errors = []
    for row, (measure, status) in enumerate(zip(clipped_measures, statuses, strict=True)):
        if int(status) != 0 or not np.isfinite(measure) or measure < 0.0:
            raise ValueError(
                "Native periodic-domain clipping did not certify a cell image."
            )
        if measure == 0.0:
            continue
        vertex_count = int(vertex_counts[row])
        face_count = int(face_counts[row])
        if vertex_count < 4 or face_count < 4:
            raise ValueError(
                "Native periodic-domain clipping returned an open cell image."
            )
        raw_faces = tuple(
            face_vertices[
                row,
                int(face_offsets[row, face_row]) : int(face_offsets[row, face_row + 1]),
            ].astype(np.int32, copy=True)
            for face_row in range(face_count)
        )
        raw_face_labels = clipped_face_labels[row, :face_count].astype(
            np.int32, copy=True
        )
        clean_vertices, clean_faces, clean_labels = _clean_clipped_polyhedron(
            clipped_vertices[row, :vertex_count], raw_faces, raw_face_labels
        )
        clean_vertices, clean_faces, exact_measure, measure_error = (
            _merge_coplanar_clipped_faces(
                clean_vertices,
                clean_faces,
                clean_labels,
                image_exact_planes[row],
                box_lower,
                box_upper,
                float(measure),
            )
        )
        if clean_vertices.shape[0] < 4:
            raise ValueError("Periodic cell clipping collapsed a positive-volume image.")
        output_rows.append(row)
        output_vertices.append(clean_vertices)
        output_faces.append(clean_faces)
        output_measures.append(exact_measure)
        output_errors.append(measure_error)
    selected = np.asarray(output_rows, dtype=np.int64)
    return (
        np.arange(selected.size, dtype=np.int64),
        np.asarray(image_stable_ids, dtype=np.int64)[selected],
        np.asarray(image_owners, dtype=np.int32)[selected],
        np.asarray(output_measures, dtype=np.float64),
        tuple(output_vertices),
        np.asarray(image_periodic_ids, dtype=np.int64)[selected],
        np.asarray(image_windings, dtype=np.int64)[selected],
        tuple(output_faces),
        np.asarray(output_errors, dtype=np.float64),
    )


def _aggregate_periodic_fragment_measures(
    stable_ids: np.ndarray,
    fragment_stable_ids: np.ndarray,
    fragment_measures: np.ndarray,
    /,
) -> np.ndarray:
    stable = np.asarray(stable_ids, dtype=np.int64)
    fragment_ids = np.asarray(fragment_stable_ids, dtype=np.int64)
    fragment_volume = np.asarray(fragment_measures, dtype=np.float64)
    if (
        stable.ndim != 1
        or np.unique(stable).size != stable.size
        or fragment_ids.shape != fragment_volume.shape
        or np.any(~np.isfinite(fragment_volume) | (fragment_volume <= 0.0))
    ):
        raise ValueError("Periodic fragment measures do not align with stable cells.")
    rows = {int(identifier): row for row, identifier in enumerate(stable)}
    result = np.zeros(stable.shape, dtype=np.float64)
    for identifier, volume in zip(fragment_ids, fragment_volume, strict=True):
        row = rows.get(int(identifier))
        if row is None:
            raise ValueError("Periodic clipping returned an unknown stable cell.")
        result[row] += float(volume)
    if np.any(~np.isfinite(result) | (result <= 0.0)):
        raise ValueError("Periodic clipping lost a positive-measure stable cell.")
    return result


def _aggregate_periodic_fragment_errors(
    stable_ids: np.ndarray,
    fragment_stable_ids: np.ndarray,
    fragment_errors: np.ndarray,
    /,
) -> np.ndarray:
    stable = np.asarray(stable_ids, dtype=np.int64)
    fragment_ids = np.asarray(fragment_stable_ids, dtype=np.int64)
    errors = np.asarray(fragment_errors, dtype=np.float64)
    if (
        stable.ndim != 1
        or np.unique(stable).size != stable.size
        or fragment_ids.shape != errors.shape
        or np.any(~np.isfinite(errors) | (errors < 0.0))
    ):
        raise ValueError("Periodic fragment errors do not align with stable cells.")
    rows = {int(identifier): row for row, identifier in enumerate(stable)}
    result = np.zeros(stable.shape, dtype=np.float64)
    for identifier, error in zip(fragment_ids, errors, strict=True):
        row = rows.get(int(identifier))
        if row is None:
            raise ValueError("Periodic clipping returned an unknown error cell.")
        result[row] += float(error)
    if np.any(~np.isfinite(result)):
        raise ValueError("Periodic fragment errors overflowed their stable cells.")
    return result


def _prepare_variable_polyhedral_common_refinement(
    source_kind: str,
    source_vertices: tuple[np.ndarray, ...],
    source_ids: np.ndarray,
    source_faces: tuple[tuple[np.ndarray, ...], ...] | None,
    target_kind: str,
    target_vertices: tuple[np.ndarray, ...],
    target_ids: np.ndarray,
    target_faces: tuple[tuple[np.ndarray, ...], ...] | None,
    policy: CommonRefinementPolicy,
    /,
) -> PreparedCommonRefinement:
    from ...geometry import _supermesh as supermesh_owner
    from .._cell_mesh import CellBlock, CellMesh

    def compact(
        vertices: tuple[np.ndarray, ...],
        identifiers: np.ndarray,
        kind: str,
        faces: tuple[tuple[np.ndarray, ...], ...] | None,
        side: str,
        /,
    ) -> tuple[CellMesh | None, np.ndarray]:
        if not vertices:
            return None, identifiers
        counts = np.asarray([row.shape[0] for row in vertices], dtype=np.int64)
        if kind == "polyhedron":
            if faces is None or len(faces) != len(vertices):
                raise ValueError(f"{side} polyhedra require aligned face loops.")
            order = np.argsort(counts, kind="stable")
        elif kind == "tetrahedron":
            if faces is not None or np.any(counts != 4):
                raise ValueError(
                    f"{side} tetrahedra require four vertices and no face loops."
                )
            order = np.arange(len(vertices), dtype=np.int64)
        else:
            raise ValueError(
                "Variable polyhedral refinement requires tetrahedron or polyhedron cells."
            )
        internal_ids = np.empty(identifiers.shape, dtype=np.int64)
        internal_ids[order] = np.arange(identifiers.size, dtype=np.int64)
        offsets = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum(counts, dtype=np.int64),
            )
        )
        coordinates = np.concatenate(vertices, axis=0)
        if kind == "polyhedron":
            if faces is None:
                raise ValueError(f"{side} polyhedron face loops are missing.")
            cells = tuple(
                tuple(
                    np.asarray(face, dtype=np.int32) + int(offsets[row])
                    for face in cell_faces
                )
                for row, cell_faces in enumerate(faces)
            )
            mesh = CellMesh.from_polyhedra(
                coordinates,
                cells,
                cell_global_ids=internal_ids,
                block_name=f"owner-local-{side}-polyhedra",
            )
        else:
            connectivity = np.stack(
                [
                    np.arange(offsets[row], offsets[row + 1], dtype=np.int32)
                    for row in range(identifiers.size)
                ]
            )
            mesh = CellMesh(
                coordinates,
                (
                    CellBlock(
                        f"owner-local-{side}-tetrahedra",
                        "tetrahedron",
                        connectivity,
                        global_ids=internal_ids,
                    ),
                ),
            )
        if not np.array_equal(
            np.asarray(
                mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
            ),
            np.arange(identifiers.size, dtype=np.int64),
        ):
            raise ValueError(f"{side} compact-cell order is not stable.")
        return mesh, identifiers[order]

    source_mesh, source_global_ids = compact(
        source_vertices, source_ids, source_kind, source_faces, "source"
    )
    target_mesh, target_global_ids = compact(
        target_vertices, target_ids, target_kind, target_faces, "target"
    )
    if source_mesh is None or target_mesh is None:
        return prepare_compact_cell_common_refinement(
            source_vertices,
            source_ids,
            target_vertices,
            target_ids,
            source_cell_kind=source_kind,
            target_cell_kind=target_kind,
            source_cell_faces=source_faces,
            target_cell_faces=target_faces,
            policy=policy,
        )
    internal = supermesh_owner.prepare_common_refinement(
        source_mesh, target_mesh, policy=policy
    )
    outside = supermesh_owner._outside_exact_domain(source_mesh) or (
        supermesh_owner._outside_exact_domain(target_mesh)
    )
    mode = supermesh_owner.PredicateMode.FILTERED if outside else policy.predicate_mode
    source = supermesh_owner._decompose(source_mesh, mode)._replace(
        global_ids=source_global_ids
    )
    target = supermesh_owner._decompose(target_mesh, mode)._replace(
        global_ids=target_global_ids
    )
    entries = supermesh_owner._Entries(
        np.asarray(internal.target_cells, dtype=np.int64),
        np.asarray(internal.source_cells, dtype=np.int64),
        np.asarray(internal.volumes, dtype=np.float64),
        np.asarray(internal.first_moments, dtype=np.float64),
        None
        if internal.second_moments is None
        else np.asarray(internal.second_moments, dtype=np.float64),
        None
        if internal.simplex_offsets is None
        else np.asarray(internal.simplex_offsets, dtype=np.int64),
        None
        if internal.simplices is None
        else np.asarray(internal.simplices, dtype=np.float64),
    )
    return PreparedCommonRefinement(
        source=source,
        target=target,
        entries=entries,
        identities=(
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            source_mesh.topology_id,
            target_mesh.topology_id,
        ),
        policy=policy,
        evidence=internal.evidence,
    )


def _local_geometry_certificate(
    kind: str,
    vertices: tuple[np.ndarray, ...],
    identifiers: np.ndarray,
    declared_measures: np.ndarray,
    tolerance: float,
    /,
    *,
    faces: tuple[tuple[np.ndarray, ...], ...] | None = None,
) -> tuple[PreparedCommonRefinement, np.ndarray, np.ndarray]:
    policy = CommonRefinementPolicy(
        coverage=CommonRefinementCoverage.COMPLETE,
        coverage_tolerance=tolerance,
    )
    if kind == "polyhedron":
        refinement = _prepare_variable_polyhedral_common_refinement(
            kind,
            vertices,
            identifiers,
            faces,
            kind,
            vertices,
            identifiers,
            faces,
            policy,
        )
    else:
        refinement = prepare_compact_cell_common_refinement(
            vertices,
            identifiers,
            vertices,
            identifiers,
            source_cell_kind=kind,
            target_cell_kind=kind,
            source_cell_faces=faces,
            target_cell_faces=faces,
            policy=policy,
        )
    native_by_id = {
        int(identifier): float(measure)
        for identifier, measure in zip(
            np.asarray(refinement.source_cell_global_ids, dtype=np.int64),
            np.asarray(refinement.source_measures, dtype=np.float64),
            strict=True,
        )
    }
    native = np.asarray([native_by_id[int(identifier)] for identifier in identifiers])
    limits = _cell_coverage_tolerances(
        vertices,
        native,
        tolerance,
        dimension=2 if kind in ("triangle", "quadrilateral", "polygon") else 3,
    )
    if not refinement.succeeded:
        raise ValueError(
            "Owner-local source/target cell geometry is not a certified "
            f"non-overlapping mesh ({refinement.status.name}: {refinement.evidence.reason})."
        )
    if (
        native.shape != declared_measures.shape
        or np.any(~np.isfinite(native) | (native <= 0.0))
        or np.any(np.abs(native - declared_measures) > limits)
    ):
        raise ValueError(
            "Owner-local cell measures do not agree with their native physical geometry."
        )
    return refinement, native, limits


def _identity_root_reference_maps(mesh: Any, /) -> dict[str, Any]:
    from .._nested_reference_types import ReferenceCellMapWitnesses
    from .._reference_cell import reference_cell_topology

    result = {}
    for block in mesh.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        corners = np.asarray(
            reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
        )
        result[block.name] = ReferenceCellMapWitnesses(
            np.arange(block.cell_count, dtype=np.int64),
            identifiers,
            block.cell_kind,
            block.cell_kind,
            np.broadcast_to(corners, (block.cell_count, *corners.shape)).copy(),
        )
    return result


def _subset_owner_local_geometry(
    mesh: Any, geometry: Any, cell_ids: Any, /
) -> tuple[Any, Any]:
    from .._cell_geometry import (
        CellGeometrySpec,
        PerCellRestrictedCellGeometryElement,
    )
    from .._cell_mesh import CellBlock, CellMesh

    selected_ids = np.asarray(cell_ids, dtype=np.int64)
    if (
        selected_ids.ndim != 1
        or np.unique(selected_ids).size != selected_ids.size
        or mesh.periodic_topology is not None
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
    ):
        raise ValueError(
            "Shared-root remapping requires unique nonperiodic tetrahedral endpoints."
        )
    elements, routes, coefficients = geometry.resolve(mesh)
    selected_blocks = []
    selected_elements = {}
    selected_routes = {}
    remaining = set(map(int, selected_ids))
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        rows = np.flatnonzero(np.isin(identifiers, selected_ids))
        if not rows.size:
            continue
        remaining.difference_update(map(int, identifiers[rows]))
        selected_blocks.append(
            CellBlock(
                block.name,
                block.cell_kind,
                np.asarray(block.vertices)[rows],
                vertex_valid=np.asarray(block.vertex_valid)[rows],
                global_ids=identifiers[rows],
            )
        )
        selected_element = element
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            selected_element = PerCellRestrictedCellGeometryElement(
                element.source_element,
                element.cell_kind,
                identifiers[rows],
                np.asarray(element.source_cell_ids, dtype=np.int64)[rows],
                tuple(element.chart_arguments(int(row)) for row in rows),
            )
        selected_elements[block.name] = selected_element
        selected_routes[block.name] = np.asarray(route)[rows]
    if remaining or not selected_blocks:
        raise ValueError("Shared-root target subset lost stable owner-local cell IDs.")
    subset_mesh = CellMesh(
        np.asarray(mesh.coordinates, dtype=np.float64),
        tuple(selected_blocks),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        numeric_version=mesh.numeric_version,
    )
    source_map = geometry.source_map
    subset_geometry = CellGeometrySpec(
        selected_elements,
        selected_routes,
        np.asarray(coefficients, dtype=np.float64),
        source_map=source_map,
        source_mesh=subset_mesh if source_map is not None else None,
    )
    return subset_mesh, subset_geometry


class _ExactCoordinateSourceProjection:
    __slots__ = ("source_id", "source_revision", "source_map_id", "_coordinates")

    def __init__(self, source_map: Any, coordinates: Any, /) -> None:
        self.source_id = source_map.source_id
        self.source_revision = source_map.source_revision
        self.source_map_id = source_map.source_map_id
        self._coordinates = tuple(tuple(value for value in row) for row in coordinates)

    def source_coordinates(self, /) -> tuple[tuple[Any, ...], ...]:
        return self._coordinates


def _root_projected_endpoint(
    root_mesh: Any,
    root_geometry: Any,
    map_rows: tuple[tuple[Any, np.ndarray], ...],
    /,
) -> tuple[Any, Any, dict[str, Any]]:
    from ...geometry._common_root_volume_overlap import (
        _as_tetrahedron,
        _chart_arguments,
        _coordinate_expression_rows,
        _root_cell_rows,
    )
    from .._cell_geometry import (
        CellGeometrySpec,
        PerCellRestrictedCellGeometryElement,
    )
    from .._cell_mesh import CellBlock, CellMesh
    from .._coordinate_enclosure import compose_expression, evaluate_expression
    from .._nested_reference_types import (
        exact_reference_cell_vertices,
        freeze_reference_chart_arguments,
        ReferenceCellMapWitnesses,
    )
    from .._reference_cell import reference_cell_topology

    if not map_rows:
        raise ValueError("An exact root restriction needs at least one endpoint map.")
    maps = tuple(value[0] for value in map_rows)
    vertex_ids = tuple(np.asarray(value[1], dtype=np.int64) for value in map_rows)
    root_ids = {
        int(np.asarray(witness.source_cell_ids, dtype=np.int64)[0]) for witness in maps
    }
    target_kinds = {witness.target_cell_kind for witness in maps}
    source_kinds = {witness.source_cell_kind for witness in maps}
    if (
        len(root_ids) != 1
        or target_kinds != {"tetrahedron"}
        or source_kinds != {"tetrahedron"}
        or any(
            np.asarray(witness.target_cell_ids).shape != (1,)
            or np.asarray(witness.source_cell_ids).shape != (1,)
            or ids.shape != (4,)
            or np.unique(ids).size != 4
            for witness, ids in zip(maps, vertex_ids, strict=True)
        )
    ):
        raise ValueError("Root restriction packets must be single-cell tetrahedron maps.")
    root_id = next(iter(root_ids))
    root_rows = _root_cell_rows(root_mesh)
    if root_id not in root_rows:
        raise ValueError(f"Exact root map names absent root cell {root_id}.")
    root_block_index, root_cell_row = root_rows[root_id]
    root_blocks = root_mesh.blocks
    root_block = root_blocks[root_block_index]
    if root_block.cell_kind != "tetrahedron":
        raise ValueError("Curved root restrictions currently require tetrahedron roots.")
    root_elements, root_routes, coefficients = root_geometry.resolve(root_mesh)
    root_element = root_elements[root_block_index]
    root_route = np.asarray(root_routes[root_block_index], dtype=np.int32)[root_cell_row]
    root_expression_rows = _coordinate_expression_rows(
        root_mesh, root_geometry, root_elements, root_routes
    )
    root_global_row = (
        sum(block.cell_count for block in root_blocks[:root_block_index]) + root_cell_row
    )
    root_expressions = root_expression_rows[root_global_row]
    target_topology = reference_cell_topology("tetrahedron")
    target_corners = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in target_topology.vertices
    )
    exact_map_rows = []
    chart_rows = []
    physical_vertex_rows = []
    for witness, ids in zip(maps, vertex_ids, strict=True):
        exact_vertices = exact_reference_cell_vertices(witness)[0]
        chart_bank = witness.source_reference_chart_arguments
        chart = (
            _chart_arguments(_as_tetrahedron(exact_vertices))
            if chart_bank is None
            else chart_bank[0]
        )
        mapped_corners = tuple(
            tuple(evaluate_expression(argument, corner) for argument in chart)
            for corner in target_corners
        )
        if mapped_corners != exact_vertices:
            raise ValueError(
                "Transferred exact root chart does not reproduce its reference-map corners."
            )
        exact_map_rows.append(exact_vertices)
        chart_rows.append(tuple(chart))
        physical_vertex_rows.append(
            tuple(
                tuple(
                    evaluate_expression(expression, point)
                    for expression in root_expressions
                )
                for point in mapped_corners
            )
        )
    point_by_id: dict[int, tuple[Any, ...]] = {}
    for ids, points in zip(vertex_ids, physical_vertex_rows, strict=True):
        for identifier, point in zip(ids.tolist(), points, strict=True):
            key = int(identifier)
            previous = point_by_id.get(key)
            if previous is not None and previous != point:
                raise ValueError(
                    "Adjacent exact root restrictions disagree at a shared endpoint vertex."
                )
            point_by_id[key] = point
    ordered_vertex_ids = np.asarray(sorted(point_by_id), dtype=np.int64)
    vertex_row_by_id = {
        int(identifier): row for row, identifier in enumerate(ordered_vertex_ids)
    }
    connectivity = np.asarray(
        [[vertex_row_by_id[int(identifier)] for identifier in ids] for ids in vertex_ids],
        dtype=np.int32,
    )
    coordinates = np.asarray(
        [
            [float(value) for value in point_by_id[int(identifier)]]
            for identifier in ordered_vertex_ids
        ],
        dtype=np.float64,
    )
    block_name = root_block.name
    target_cell_ids = np.asarray(
        [int(np.asarray(witness.target_cell_ids, dtype=np.int64)[0]) for witness in maps],
        dtype=np.int64,
    )
    source_cell_ids = np.full(target_cell_ids.shape, root_id, dtype=np.int64)
    endpoint_base = root_element
    element_source_cell_ids = source_cell_ids
    endpoint_charts = tuple(chart_rows)
    if isinstance(root_element, PerCellRestrictedCellGeometryElement):
        # A restricted root carrier composes into one restriction of its base element.
        root_chart = root_element.chart_arguments(root_cell_row)
        endpoint_base = root_element.source_element
        element_source_cell_ids = np.full(
            target_cell_ids.shape,
            int(np.asarray(root_element.source_cell_ids, dtype=np.int64)[root_cell_row]),
            dtype=np.int64,
        )
        endpoint_charts = tuple(
            tuple(compose_expression(argument, chart) for argument in root_chart)
            for chart in chart_rows
        )
    map_target_ids = np.arange(target_cell_ids.size, dtype=np.int64)
    endpoint_mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                block_name,
                "tetrahedron",
                connectivity,
                global_ids=target_cell_ids,
            ),
        ),
        vertex_global_ids=ordered_vertex_ids,
        numeric_version=root_mesh.numeric_version,
    )
    endpoint_element = PerCellRestrictedCellGeometryElement(
        endpoint_base,
        "tetrahedron",
        target_cell_ids,
        element_source_cell_ids,
        endpoint_charts,
    )
    source_map = root_geometry.source_map
    if source_map is not None:
        source_map = _ExactCoordinateSourceProjection(
            source_map, root_geometry.source_coordinates()
        )
    endpoint_geometry = CellGeometrySpec(
        {block_name: endpoint_element},
        {block_name: np.tile(root_route[None, :], (target_cell_ids.size, 1))},
        np.asarray(coefficients, dtype=np.float64),
        source_map=source_map,
        source_mesh=endpoint_mesh if source_map is not None else None,
    )
    endpoint_map = ReferenceCellMapWitnesses(
        map_target_ids,
        source_cell_ids,
        "tetrahedron",
        "tetrahedron",
        np.asarray(
            [
                [[float(value) for value in point] for point in vertices]
                for vertices in exact_map_rows
            ],
            dtype=np.float64,
        ),
        tuple(exact_map_rows),
        freeze_reference_chart_arguments(tuple(chart_rows)),
    )
    return endpoint_mesh, endpoint_geometry, {block_name: endpoint_map}


def _root_owner_overlap(
    root_view: Any,
    map_rows: tuple[tuple[Any, np.ndarray], ...],
    root_role: str,
    tolerance: float,
    require_complete: bool,
    integration_policy: Any,
    /,
) -> tuple[Any, Any]:
    from ...geometry._common_root_volume_overlap import (
        CommonRootVolumeWitness,
        prepare_common_root_volume_refinement,
    )
    from ...geometry._supermesh import CommonRefinementCoverage

    if root_role not in ("source", "target"):
        raise ValueError("Curved overlap root role must be source or target.")
    root_mesh, root_geometry = root_view.reconstruct_local_geometry()
    if root_mesh is None or root_geometry is None:
        raise ValueError("The declared root owner has no resident exact root carrier.")
    root_ids = {
        int(np.asarray(witness.source_cell_ids, dtype=np.int64)[0])
        for witness, _ in map_rows
    }
    if len(root_ids) != 1:
        raise ValueError("One root-owner overlap batch must retain one exact root cell.")
    root_id = next(iter(root_ids))
    root_subset_mesh, root_subset_geometry = _subset_owner_local_geometry(
        root_mesh, root_geometry, np.asarray((root_id,), dtype=np.int64)
    )
    endpoint_mesh, endpoint_geometry, endpoint_maps = _root_projected_endpoint(
        root_subset_mesh, root_subset_geometry, map_rows
    )
    identity_maps = _identity_root_reference_maps(root_subset_mesh)
    if root_role == "source":
        source_mesh, source_geometry = root_subset_mesh, root_subset_geometry
        target_mesh, target_geometry = endpoint_mesh, endpoint_geometry
        source_maps, target_maps = identity_maps, endpoint_maps
    else:
        source_mesh, source_geometry = endpoint_mesh, endpoint_geometry
        target_mesh, target_geometry = root_subset_mesh, root_subset_geometry
        source_maps, target_maps = endpoint_maps, identity_maps
    witness = CommonRootVolumeWitness(
        root_subset_mesh,
        root_subset_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        source_maps,
        target_maps,
    )
    coverage = (
        CommonRefinementCoverage.TARGET
        if require_complete
        else CommonRefinementCoverage.PARTIAL
    )
    return prepare_common_root_volume_refinement(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(
            coverage=coverage,
            coverage_tolerance=tolerance,
            second_moments=True,
        ),
        integration_policy=integration_policy,
    )


def _transition_reference_maps(transition: Any, /) -> tuple[str, tuple[Any, ...]] | None:
    if transition is None:
        return None
    from .._cell_geometry_transition_types import _nested_reference_groups
    from .._nested_reference_types import ReferenceCellMapWitnesses

    if transition.coarsening is not None:
        return (
            "target",
            _nested_reference_groups(
                transition.coarsening,
                transition.source_mesh,
                transition.target_mesh,
                reverse=True,
            ),
        )
    if transition.reference_map is not None:
        value = transition.reference_map
        groups = (value,) if isinstance(value, ReferenceCellMapWitnesses) else value
        return "source", tuple(groups)
    if transition.refinement is not None:
        return (
            "source",
            _nested_reference_groups(
                transition.refinement, transition.source_mesh, transition.target_mesh
            ),
        )
    return None


class _OwnerLocalCurvedRefinement(NamedTuple):
    status: CommonRefinementStatus
    source_cell_global_ids: np.ndarray
    target_cell_global_ids: np.ndarray
    source_measures: np.ndarray
    target_measures: np.ndarray
    target_offsets: np.ndarray
    target_cells: np.ndarray
    source_cells: np.ndarray
    volumes: np.ndarray
    first_moments: np.ndarray
    second_moments: np.ndarray
    evidence: CommonRefinementEvidence
    refinement_id: str
    target_cell_count: int

    @property
    def entry_count(self) -> int:
        return int(self.volumes.size)


def _owner_view_authority_identity(view: Any, /) -> tuple[Any, ...]:
    binding = _view_binding(view)
    return tuple(
        binding[name]
        for name in (
            "part_name",
            "part_id",
            "distribution_id",
            "source_id",
            "source_revision",
            "source_mesh_id",
            "source_topology_id",
            "source_geometry_id",
            "source_map_id",
            "source_context_id",
            "source_manifest_id",
            "directory_group_id",
            "source_directory_id",
            "cell_directory_id",
            "vertex_directory_id",
            "geometry_layout_id",
            "coordinate_contract_id",
        )
    )


def _same_reference_cell_map(first: Any, second: Any, /) -> bool:
    from .._nested_reference_types import (
        exact_reference_cell_vertices,
        ReferenceCellMapWitnesses,
    )

    if not isinstance(first, ReferenceCellMapWitnesses) or not isinstance(
        second, ReferenceCellMapWitnesses
    ):
        return False
    return bool(
        first.target_cell_kind == second.target_cell_kind
        and first.source_cell_kind == second.source_cell_kind
        and np.array_equal(
            np.asarray(first.target_cell_ids, dtype=np.int64),
            np.asarray(second.target_cell_ids, dtype=np.int64),
        )
        and np.array_equal(
            np.asarray(first.source_cell_ids, dtype=np.int64),
            np.asarray(second.source_cell_ids, dtype=np.int64),
        )
        and np.array_equal(
            np.asarray(first.source_reference_vertices, dtype=np.float64),
            np.asarray(second.source_reference_vertices, dtype=np.float64),
        )
        and exact_reference_cell_vertices(first) == exact_reference_cell_vertices(second)
        and first.source_reference_chart_arguments
        == second.source_reference_chart_arguments
    )


def _reference_map_row_for_cell(groups: tuple[Any, ...], cell_id: int, /) -> Any | None:
    from .._nested_reference_types import (
        exact_reference_cell_vertices,
        ReferenceCellMapWitnesses,
    )

    found = None
    for witness in groups:
        if not isinstance(witness, ReferenceCellMapWitnesses):
            raise TypeError(
                "Curved root routing requires exact ReferenceCellMapWitnesses."
            )
        target_ids = np.asarray(witness.target_cell_ids, dtype=np.int64)
        rows = np.flatnonzero(target_ids == int(cell_id))
        if rows.size > 1 or rows.size and found is not None:
            raise ValueError("A stable endpoint cell has duplicate exact root maps.")
        if not rows.size:
            continue
        row = int(rows[0])
        exact = exact_reference_cell_vertices(witness)
        chart_bank = witness.source_reference_chart_arguments
        found = ReferenceCellMapWitnesses(
            target_ids[row : row + 1],
            np.asarray(witness.source_cell_ids, dtype=np.int64)[row : row + 1],
            witness.target_cell_kind,
            witness.source_cell_kind,
            np.asarray(witness.source_reference_vertices, dtype=np.float64)[
                row : row + 1
            ],
            (exact[row],),
            None if chart_bank is None else (chart_bank[row],),
        )
    return found


def _owner_cell_vertex_ids(view: Any, cell_id: int, /) -> np.ndarray:
    for block in view.blocks:
        identifiers = np.asarray(block.cell_ids, dtype=np.int64)
        rows = np.flatnonzero(identifiers == int(cell_id))
        if not rows.size:
            continue
        if rows.size != 1:
            raise ValueError("A stable owner-local cell ID occurs more than once.")
        row = int(rows[0])
        vertices = np.asarray(block.vertices, dtype=np.int64)[row]
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)[row]
        local_vertices = vertices[valid]
        if local_vertices.size != 4:
            raise ValueError(
                "Curved root-map routing requires tetrahedral endpoint vertices."
            )
        return np.asarray(view.vertex_ids, dtype=np.int64)[local_vertices]
    raise ValueError(f"Endpoint cell {int(cell_id)} has no local vertex map.")


def _model_recipe_packet_bytes(value: Any, /) -> bytes:
    import tempfile
    from pathlib import Path

    from ..._array_archive import write_array_archive
    from ..._model._structure import (
        model_archive_projection,
        model_structure_recipe,
        pack_model_array_tree,
        validate_model_structure_recipe,
    )
    from ...meshing._model_codec import (
        _MODEL_ARRAYS,
        _MODEL_KIND,
        _register_model_archive_types,
    )

    _register_model_archive_types()
    projected = model_archive_projection(value)
    recipe = model_structure_recipe(projected)
    validate_model_structure_recipe(recipe, limits=None)
    arrays = pack_model_array_tree(projected, recipe, prefix=_MODEL_ARRAYS, limits=None)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "owner-local-map-packet.phx"
        write_array_archive(
            path,
            manifest={"kind": _MODEL_KIND, "model_recipe": recipe},
            arrays=arrays,
            limits=None,
        )
        return path.read_bytes()


def _model_recipe_packet_from_bytes(value: bytes, /) -> Any:
    import tempfile
    from pathlib import Path

    from ..._array_archive import read_array_archive
    from ..._model._structure import (
        model_from_array_recipe,
        model_structure_recipe,
        validate_model_structure_recipe,
    )
    from ..._typing_plan import validate_tree
    from ...meshing._model_codec import (
        _MODEL_ARRAYS,
        _MODEL_KIND,
        _register_model_archive_types,
    )

    _register_model_archive_types()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "owner-local-map-packet.phx"
        path.write_bytes(value)
        manifest, arrays = read_array_archive(path, limits=None)
    if (
        set(manifest) != {"arrays", "kind", "model_recipe"}
        or manifest["kind"] != _MODEL_KIND
    ):
        raise ValueError("Curved owner-local map packet has a noncanonical model recipe.")
    recipe = manifest["model_recipe"]
    validate_model_structure_recipe(recipe, limits=None)
    rebuilt = model_from_array_recipe(
        recipe, arrays, prefix=_MODEL_ARRAYS, limits=None, validate=False
    )
    if model_structure_recipe(rebuilt) != recipe:
        raise ValueError(
            "Curved owner-local map packet did not reproduce its model recipe."
        )
    validate_tree(rebuilt)
    return rebuilt


def _exchange_curved_root_map_packets(
    records_by_owner: tuple[tuple[Any, ...], ...],
    local_rank: int,
    execution_group: ExecutionGroup,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    /,
) -> tuple[tuple[Any, ...], dict[int, tuple[Any, ...]]]:
    from ...meshing._distribution import _owner_exchange_message_banks

    if len(records_by_owner) != owner_count or records_by_owner[local_rank]:
        raise ValueError("Curved root-map packets must exclude the local owner channel.")
    encoded: dict[int, bytes] = {}
    encode_error: Exception | None = None
    try:
        for owner, records in enumerate(records_by_owner):
            if records:
                encoded[owner] = _model_recipe_packet_bytes(
                    ("owner-local-curved-root-map-packets-v1", records)
                )
    except Exception as caught:
        encode_error = caught
    _synchronize_error(
        encode_error, mesh, axis, owner_count, "curved root-map packet encoding"
    )
    width = 8 + _group_max(
        max((len(value) for value in encoded.values()), default=0),
        mesh,
        axis,
        owner_count,
    )
    outgoing = []
    for owner in range(owner_count):
        payload = encoded.get(owner)
        if payload is None:
            outgoing.append(np.zeros((0, width), dtype=np.uint8))
            continue
        row = np.zeros((1, width), dtype=np.uint8)
        row[0, :8] = np.frombuffer(len(payload).to_bytes(8, "little"), dtype=np.uint8)
        row[0, 8 : 8 + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        outgoing.append(row)
    incoming_wire = _owner_exchange_message_banks(
        tuple(outgoing), local_rank, execution_group, axis
    )
    incoming: dict[int, tuple[Any, ...]] = {}
    decode_error: Exception | None = None
    try:
        for sender, bank in incoming_wire.items():
            if bank.shape != (1, width):
                raise ValueError("Curved root-map packet has an invalid row layout.")
            length = int.from_bytes(
                np.asarray(bank[0, :8], dtype=np.uint8).tobytes(), "little"
            )
            if not 0 < length <= width - 8:
                raise ValueError("Curved root-map packet has an invalid payload length.")
            payload = _model_recipe_packet_from_bytes(
                np.asarray(bank[0, 8 : 8 + length], dtype=np.uint8).tobytes()
            )
            if (
                not isinstance(payload, tuple)
                or len(payload) != 2
                or payload[0] != "owner-local-curved-root-map-packets-v1"
                or not isinstance(payload[1], tuple)
            ):
                raise ValueError("Curved root-map packet has an unknown schema.")
            incoming[int(sender)] = payload[1]
    except Exception as caught:
        decode_error = caught
    _synchronize_error(
        decode_error, mesh, axis, owner_count, "curved root-map packet decoding"
    )
    return records_by_owner[local_rank], incoming


def _curved_self_cell_geometry(
    view: Any,
    cell_kind: str,
    empty_refinement: PreparedCommonRefinement,
    tolerance: float,
    integration_policy: Any,
    /,
) -> tuple[
    PreparedCommonRefinement,
    Any | None,
    dict[int, float],
    dict[int, float],
    dict[int, np.ndarray],
]:
    from ...geometry._common_root_volume_overlap import (
        CommonRootVolumeWitness,
        prepare_common_root_volume_refinement,
    )

    mesh, geometry = view.reconstruct_local_geometry()
    if mesh is None or geometry is None:
        if np.asarray(view.cell_ids).size:
            raise ValueError(
                "Curved owner-local cells lack their exact geometry carrier."
            )
        return empty_refinement, None, {}, {}, {}
    if not mesh.blocks:
        if np.asarray(view.cell_ids).size:
            raise ValueError("Curved owner-local cells have no typed geometry blocks.")
        return empty_refinement, None, {}, {}, {}
    if any(block.cell_kind != cell_kind for block in mesh.blocks):
        raise ValueError("Curved owner-local geometry changed its certified cell family.")
    identity_maps = _identity_root_reference_maps(mesh)
    witness = CommonRootVolumeWitness(
        mesh,
        geometry,
        mesh,
        geometry,
        mesh,
        geometry,
        identity_maps,
        identity_maps,
    )
    refinement, integration = prepare_common_root_volume_refinement(
        mesh,
        geometry,
        mesh,
        geometry,
        witness,
        policy=CommonRefinementPolicy(
            coverage=CommonRefinementCoverage.COMPLETE,
            coverage_tolerance=tolerance,
            second_moments=True,
        ),
        integration_policy=integration_policy,
    )
    if not refinement.succeeded:
        raise ValueError(
            "Curved local cell self-overlap is not a certified complete geometry."
        )
    source_ids = np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
    source_measures = np.asarray(refinement.source_measures, dtype=np.float64)
    source_error_rows = np.asarray(integration.source_errors, dtype=np.float64)
    source_bounds_rows = np.asarray(
        integration.source_coordinate_bounds, dtype=np.float64
    )
    if (
        source_measures.shape != source_ids.shape
        or source_error_rows.ndim != 2
        or source_error_rows.shape[0] != source_ids.size
        or source_error_rows.shape[1] < 1
        or source_bounds_rows.shape != (source_ids.size, 2, 3)
    ):
        raise ValueError("Curved self-integration evidence lost its cell row shapes.")
    native: dict[int, float] = {}
    errors: dict[int, float] = {}
    bounds: dict[int, np.ndarray] = {}
    for row in range(source_ids.size):
        identifier = int(source_ids[row])
        native[identifier] = float(source_measures[row])
        errors[identifier] = float(source_error_rows[row, 0])
        bounds[identifier] = np.array(
            source_bounds_rows[row], dtype=np.float64, copy=True
        )
    declared_by_id = {
        int(identifier): float(measure)
        for identifier, measure in zip(
            np.asarray(view.cell_ids, dtype=np.int64),
            np.asarray(view.cell_measures, dtype=np.float64),
            strict=True,
        )
    }
    for identifier, measure in native.items():
        declared = declared_by_id.get(identifier)
        if declared is None:
            raise ValueError("Curved local geometry lost an accepted cell measure row.")
        diameter = float(np.linalg.norm(np.diff(bounds[identifier], axis=0)))
        limit = (
            integration_policy.relative_tolerance * measure
            + integration_policy.absolute_tolerance
            + errors[identifier]
            + _COVERAGE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * diameter**3
        )
        if abs(measure - declared) > limit:
            raise ValueError(
                "Owner-local measures do not agree with their exact curved geometry."
            )
    return refinement, integration, native, errors, bounds


def _owner_local_boxes(
    cell_vertices: tuple[np.ndarray, ...], dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    if not cell_vertices:
        empty = np.empty((0, dimension), dtype=np.float64)
        return empty, empty.copy()
    return (
        np.stack([np.min(vertices, axis=0) for vertices in cell_vertices]),
        np.stack([np.max(vertices, axis=0) for vertices in cell_vertices]),
    )


def _rank_source_bounds(
    vertices: tuple[np.ndarray, ...], dimension: int, /
) -> np.ndarray:
    if not vertices:
        return np.concatenate(
            (np.full((dimension,), np.inf), np.full((dimension,), -np.inf))
        )
    return np.concatenate(
        (
            np.min(np.stack([np.min(cell, axis=0) for cell in vertices]), axis=0),
            np.max(np.stack([np.max(cell, axis=0) for cell in vertices]), axis=0),
        )
    )


def _target_queries(
    target_ids: np.ndarray,
    target_vertices: tuple[np.ndarray, ...],
    dimension: int,
    rank_bounds: np.ndarray,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    periodic_lattice: np.ndarray | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    periodic_inverse = (
        None
        if periodic_lattice is None
        else np.linalg.inv(np.asarray(periodic_lattice, dtype=np.float64))
    )
    query_vertices = (
        target_vertices
        if periodic_inverse is None
        else tuple(vertices @ periodic_inverse for vertices in target_vertices)
    )
    target_min, target_max = _owner_local_boxes(query_vertices, dimension)
    gathered = _all_gather_local(rank_bounds, mesh, axis, owner_count).reshape(
        (owner_count, 2 * dimension)
    )
    target_capacity = max(1, _group_max(target_ids.size, mesh, axis, owner_count))
    request_ids = np.full((owner_count, target_capacity), -1, dtype=np.int64)
    request_bounds = np.zeros(
        (owner_count, target_capacity, 2 * dimension), dtype=np.float64
    )
    valid = np.zeros((owner_count, target_capacity), dtype=np.bool_)
    for target_row, (lower, upper) in enumerate(zip(target_min, target_max, strict=True)):
        for source_owner, bounds in enumerate(gathered):
            if not np.all(np.isfinite(bounds)):
                continue
            source_lower, source_upper = bounds[:dimension], bounds[dimension:]
            if periodic_inverse is None:
                overlaps = np.all((source_upper > lower) & (source_lower < upper))
            else:
                shift_lower = np.ceil(np.nextafter(lower - source_upper, -np.inf))
                shift_upper = np.floor(np.nextafter(upper - source_lower, np.inf))
                overlaps = bool(np.all(shift_lower <= shift_upper))
            if overlaps:
                request_ids[source_owner, target_row] = target_ids[target_row]
                request_bounds[source_owner, target_row] = np.concatenate((lower, upper))
                valid[source_owner, target_row] = True
    return request_ids, request_bounds, valid


def _route_candidate_source_cells(
    source_ids: np.ndarray,
    source_owners: np.ndarray,
    source_measures: np.ndarray,
    source_vertices: tuple[np.ndarray, ...],
    source_faces: tuple[tuple[np.ndarray, ...], ...] | None,
    source_periodic_ids: np.ndarray,
    source_windings: np.ndarray,
    request_ids: np.ndarray,
    request_bounds: np.ndarray,
    request_valid: np.ndarray,
    dimension: int,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    periodic_lattice: np.ndarray | None,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, ...], ...] | None,
]:
    if (
        source_owners.shape != source_ids.shape
        or source_measures.shape != source_ids.shape
        or source_periodic_ids.shape != source_ids.shape
        or source_windings.ndim != 2
        or source_windings.shape[0] != source_ids.size
        or source_faces is not None
        and len(source_faces) != source_ids.size
    ):
        raise ValueError("Owner-local source cell-map identities do not align.")
    received_ids = _all_to_all_local(request_ids, mesh, axis, owner_count)
    received_bounds = _all_to_all_local(request_bounds, mesh, axis, owner_count)
    received_valid = _all_to_all_local(request_valid, mesh, axis, owner_count).astype(
        np.bool_
    )
    if np.any(received_valid & (received_ids < 0)):
        raise ValueError("AABB requests lost their target stable-cell identities.")
    periodic_inverse = (
        None
        if periodic_lattice is None
        else np.linalg.inv(np.asarray(periodic_lattice, dtype=np.float64))
    )
    if periodic_inverse is None:
        source_query_vertices = source_vertices
    else:
        source_query_vertices = tuple(
            vertices @ periodic_inverse for vertices in source_vertices
        )
    source_min, source_max = _owner_local_boxes(source_query_vertices, dimension)
    selected_by_target_owner: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
    from itertools import product

    candidate_counts = np.zeros((owner_count,), dtype=np.int32)
    for target_owner in range(owner_count):
        query_rows = np.flatnonzero(received_valid[target_owner])
        selected: set[tuple[int, tuple[int, ...]]] = set()
        if source_ids.size and query_rows.size:
            bounds = received_bounds[target_owner, query_rows]
            qlower, qupper = bounds[:, :dimension], bounds[:, dimension:]
            if periodic_inverse is None:
                overlap = np.all(
                    (source_max[:, None, :] > qlower[None, :, :])
                    & (source_min[:, None, :] < qupper[None, :, :]),
                    axis=2,
                )
                selected.update(
                    (int(row), (0,) * dimension)
                    for row in np.flatnonzero(np.any(overlap, axis=1))
                )
            else:
                if source_windings.shape[1] != dimension:
                    raise ValueError("Periodic source image routing lost cell windings.")
                for source_row in range(source_ids.size):
                    for query_row in range(query_rows.size):
                        lower_shift = np.ceil(
                            np.nextafter(
                                qlower[query_row] - source_max[source_row], -np.inf
                            )
                        ).astype(np.int64)
                        upper_shift = np.floor(
                            np.nextafter(
                                qupper[query_row] - source_min[source_row], np.inf
                            )
                        ).astype(np.int64)
                        if np.any(lower_shift > upper_shift):
                            continue
                        for shift in product(
                            *(
                                range(int(lower), int(upper) + 1)
                                for lower, upper in zip(
                                    lower_shift, upper_shift, strict=True
                                )
                            )
                        ):
                            image = np.asarray(shift, dtype=np.int64)
                            if np.all(
                                source_max[source_row] + image > qlower[query_row]
                            ) and np.all(
                                source_min[source_row] + image < qupper[query_row]
                            ):
                                selected.add(
                                    (
                                        source_row,
                                        tuple(int(value) for value in image),
                                    )
                                )
        selected_rows = tuple(
            sorted(
                selected,
                key=lambda item: (
                    int(source_ids[item[0]]),
                    int(source_periodic_ids[item[0]]),
                    tuple(
                        int(value)
                        for value in (
                            source_windings[item[0]] + np.asarray(item[1], dtype=np.int64)
                        )
                    ),
                ),
            )
        )
        selected_by_target_owner.append(selected_rows)
        candidate_counts[target_owner] = len(selected_rows)
    received_counts = _all_to_all_local(
        candidate_counts.reshape((owner_count, 1)), mesh, axis, owner_count
    )[:, 0].astype(np.int64)
    capacity = max(
        1,
        _group_max(int(np.sum(received_counts)), mesh, axis, owner_count),
    )
    local_arity = max((row.shape[0] for row in source_vertices), default=0)
    arity = max(1, _group_max(local_arity, mesh, axis, owner_count))
    local_faces = (
        0 if source_faces is None else max((len(row) for row in source_faces), default=0)
    )
    face_capacity = max(1, _group_max(local_faces, mesh, axis, owner_count))
    local_face_arity = (
        0
        if source_faces is None
        else max((face.size for cell in source_faces for face in cell), default=0)
    )
    face_arity = max(1, _group_max(local_face_arity, mesh, axis, owner_count))
    outgoing_ids = np.full((owner_count, capacity), -1, dtype=np.int64)
    outgoing_measures = np.zeros((owner_count, capacity), dtype=np.float64)
    outgoing_vertex_counts = np.zeros((owner_count, capacity), dtype=np.int32)
    outgoing_vertices = np.zeros(
        (owner_count, capacity, arity, dimension), dtype=np.float64
    )
    outgoing_periodic_ids = np.full((owner_count, capacity), -1, dtype=np.int64)
    outgoing_valid = np.zeros((owner_count, capacity), dtype=np.bool_)
    winding_width = (
        source_windings.shape[1]
        if source_windings.shape[1]
        else dimension
        if periodic_lattice is not None
        else 0
    )
    if periodic_lattice is not None and winding_width != dimension:
        raise ValueError("Periodic source windings do not span the lattice rank.")
    outgoing_windings = (
        np.zeros((owner_count, capacity, winding_width), dtype=np.int64)
        if winding_width
        else None
    )
    outgoing_face_counts = (
        np.zeros((owner_count, capacity), dtype=np.int32)
        if source_faces is not None
        else None
    )
    outgoing_face_vertex_counts = (
        np.zeros((owner_count, capacity, face_capacity), dtype=np.int32)
        if source_faces is not None
        else None
    )
    outgoing_face_vertices = (
        np.full(
            (owner_count, capacity, face_capacity, face_arity),
            -1,
            dtype=np.int32,
        )
        if source_faces is not None
        else None
    )
    for target_owner, owner_rows in enumerate(selected_by_target_owner):
        for slot, (source_row, winding_shift) in enumerate(owner_rows):
            points = source_vertices[int(source_row)]
            winding_shift_array = np.asarray(winding_shift, dtype=np.int64)
            if periodic_lattice is not None:
                points = points + winding_shift_array @ periodic_lattice
            outgoing_ids[target_owner, slot] = source_ids[source_row]
            outgoing_measures[target_owner, slot] = source_measures[source_row]
            outgoing_vertex_counts[target_owner, slot] = points.shape[0]
            outgoing_vertices[target_owner, slot, : points.shape[0]] = points
            outgoing_periodic_ids[target_owner, slot] = source_periodic_ids[source_row]
            outgoing_valid[target_owner, slot] = True
            if outgoing_windings is not None:
                outgoing_windings[target_owner, slot] = (
                    source_windings[source_row] + winding_shift_array
                )
            if source_faces is not None:
                if (
                    outgoing_face_counts is None
                    or outgoing_face_vertex_counts is None
                    or outgoing_face_vertices is None
                ):
                    raise RuntimeError(
                        "Polyhedral source face transport arrays are missing."
                    )
                cell_faces = source_faces[int(source_row)]
                outgoing_face_counts[target_owner, slot] = len(cell_faces)
                for face_row, loop in enumerate(cell_faces):
                    outgoing_face_vertex_counts[target_owner, slot, face_row] = loop.size
                    outgoing_face_vertices[target_owner, slot, face_row, : loop.size] = (
                        loop
                    )
    candidate_ids = _all_to_all_local(outgoing_ids, mesh, axis, owner_count)
    candidate_measures = _all_to_all_local(outgoing_measures, mesh, axis, owner_count)
    candidate_vertex_counts = _all_to_all_local(
        outgoing_vertex_counts, mesh, axis, owner_count
    )
    candidate_vertices = _all_to_all_local(outgoing_vertices, mesh, axis, owner_count)
    candidate_periodic_ids = _all_to_all_local(
        outgoing_periodic_ids, mesh, axis, owner_count
    )
    candidate_valid = _all_to_all_local(outgoing_valid, mesh, axis, owner_count).astype(
        np.bool_
    )
    candidate_windings = (
        _all_to_all_local(outgoing_windings, mesh, axis, owner_count)
        if outgoing_windings is not None
        else np.empty((owner_count, capacity, 0), dtype=np.int64)
    )
    candidate_face_counts = (
        _all_to_all_local(outgoing_face_counts, mesh, axis, owner_count)
        if outgoing_face_counts is not None
        else None
    )
    candidate_face_vertex_counts = (
        _all_to_all_local(outgoing_face_vertex_counts, mesh, axis, owner_count)
        if outgoing_face_vertex_counts is not None
        else None
    )
    candidate_face_vertices = (
        _all_to_all_local(outgoing_face_vertices, mesh, axis, owner_count)
        if outgoing_face_vertices is not None
        else None
    )
    rows = [
        (owner, slot)
        for owner in range(owner_count)
        for slot in np.flatnonzero(candidate_valid[owner])
    ]
    rows.sort(
        key=lambda item: (
            item[0],
            int(candidate_ids[item]),
            int(candidate_periodic_ids[item]),
            tuple(int(value) for value in candidate_windings[item]),
        )
    )
    owners = np.asarray([owner for owner, _ in rows], dtype=np.int32)
    slots = np.asarray([slot for _, slot in rows], dtype=np.int32)
    ids = candidate_ids[owners, slots]
    periodic_ids = candidate_periodic_ids[owners, slots]
    windings = candidate_windings[owners, slots]
    measures = candidate_measures[owners, slots]
    vertices = tuple(
        candidate_vertices[owner, slot, : int(candidate_vertex_counts[owner, slot])]
        for owner, slot in rows
    )
    faces = None
    if candidate_face_counts is not None:
        if candidate_face_vertex_counts is None or candidate_face_vertices is None:
            raise RuntimeError(
                "Received polyhedral source faces lack aligned vertex banks."
            )
        face_rows = []
        for owner, slot in rows:
            cell_faces = []
            for face_row in range(int(candidate_face_counts[owner, slot])):
                count = int(candidate_face_vertex_counts[owner, slot, face_row])
                cell_faces.append(
                    candidate_face_vertices[owner, slot, face_row, :count].astype(
                        np.int32
                    )
                )
            face_rows.append(tuple(cell_faces))
        faces = tuple(face_rows)
    identities = tuple(
        (
            int(owners[row]),
            int(ids[row]),
            int(periodic_ids[row]),
            tuple(int(windings[row, axis]) for axis in range(windings.shape[1])),
        )
        for row in range(ids.size)
    )
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Source owner directory returned duplicate physical cell identities."
        )
    geometry_ids = np.arange(ids.size, dtype=np.int64)
    return owners, geometry_ids, ids, measures, vertices, periodic_ids, windings, faces


def _exchange_index_requests(
    request_ids: np.ndarray,
    request_valid: np.ndarray,
    local_cell_ids: np.ndarray,
    local_owned: np.ndarray,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    stage: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    received_ids = _all_to_all_local(request_ids, mesh, axis, owner_count)
    received_valid = _all_to_all_local(request_valid, mesh, axis, owner_count).astype(
        np.bool_
    )
    owned_lookup = {
        int(identifier): row
        for row, (identifier, owned) in enumerate(
            zip(local_cell_ids, local_owned, strict=True)
        )
        if owned
    }
    response_rows = np.full(received_ids.shape, -1, dtype=np.int32)
    error: Exception | None = None
    try:
        for requester, slot in zip(*np.nonzero(received_valid), strict=True):
            identifier = int(received_ids[requester, slot])
            if identifier not in owned_lookup:
                raise ValueError(
                    f"Requested {stage} cell {identifier} is not locally owned."
                )
            response_rows[requester, slot] = owned_lookup[identifier]
    except Exception as caught:
        error = caught
    _synchronize_error(error, mesh, axis, owner_count, f"{stage} owner lookup")
    return received_ids, response_rows


def _build_request_table(
    overlap_cell_ids: np.ndarray,
    overlap_owners: np.ndarray,
    local_source_cell_ids: np.ndarray,
    local_source_owned: np.ndarray,
    owner_count: int,
    mesh: Mesh,
    axis: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pairs = sorted(
        set(
            (int(owner), int(identifier))
            for owner, identifier in zip(overlap_owners, overlap_cell_ids, strict=True)
        )
    )
    by_owner: list[list[int]] = [[] for _ in range(owner_count)]
    for owner, identifier in pairs:
        if not 0 <= owner < owner_count:
            raise ValueError("Overlap source owner is outside the execution group.")
        by_owner[owner].append(identifier)
    capacity = max(1, _group_max(len(pairs), mesh, axis, owner_count))
    request_ids = np.full((owner_count, capacity), -1, dtype=np.int64)
    request_valid = np.zeros((owner_count, capacity), dtype=np.bool_)
    slot_by_key: dict[tuple[int, int], int] = {}
    for owner, owner_ids in enumerate(by_owner):
        for slot, identifier in enumerate(sorted(set(owner_ids))):
            request_ids[owner, slot] = identifier
            request_valid[owner, slot] = True
            slot_by_key[(owner, identifier)] = slot
    overlap_slots = np.asarray(
        [
            slot_by_key[(int(owner), int(identifier))]
            for owner, identifier in zip(overlap_owners, overlap_cell_ids, strict=True)
        ],
        dtype=np.int32,
    )
    _, source_response_rows = _exchange_index_requests(
        request_ids,
        request_valid,
        local_source_cell_ids,
        local_source_owned,
        mesh,
        axis,
        owner_count,
        "source",
    )
    return request_ids, request_valid, overlap_slots, source_response_rows, capacity


def _build_ghost_requests(
    target_view: Any,
    target_owned_output_ids: np.ndarray,
    local_rank: int,
    owner_count: int,
    mesh: Mesh,
    axis: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    view_ids = np.asarray(target_view.cell_ids, dtype=np.int64)
    view_owners = np.asarray(target_view.cell_owner_ranks, dtype=np.int32)
    view_owned = np.asarray(target_view.owned, dtype=np.bool_)
    ghost_rows = np.flatnonzero(~view_owned).astype(np.int32)
    ghost_owners = view_owners[ghost_rows]
    if np.any(ghost_owners == local_rank) or np.any(
        (ghost_owners < 0) | (ghost_owners >= owner_count)
    ):
        raise ValueError("Target ghost rows have invalid stable-cell owners.")
    unique_by_owner = [
        sorted(set(int(view_ids[row]) for row in ghost_rows if view_owners[row] == owner))
        for owner in range(owner_count)
    ]
    capacity = max(
        1,
        _group_max(sum(map(len, unique_by_owner)), mesh, axis, owner_count),
    )
    request_ids = np.full((owner_count, capacity), -1, dtype=np.int64)
    request_valid = np.zeros((owner_count, capacity), dtype=np.bool_)
    slot_by_key: dict[tuple[int, int], int] = {}
    for owner, identifiers in enumerate(unique_by_owner):
        for slot, identifier in enumerate(identifiers):
            request_ids[owner, slot] = identifier
            request_valid[owner, slot] = True
            slot_by_key[(owner, identifier)] = slot
    ghost_slots = np.asarray(
        [
            slot_by_key[(int(owner), int(view_ids[row]))]
            for row, owner in zip(ghost_rows, ghost_owners, strict=True)
        ],
        dtype=np.int32,
    )
    received_ids, response_view_rows = _exchange_index_requests(
        request_ids,
        request_valid,
        view_ids,
        view_owned,
        mesh,
        axis,
        owner_count,
        "target ghost",
    )
    output_rows = {
        int(identifier): row for row, identifier in enumerate(target_owned_output_ids)
    }
    response_rows = np.full(response_view_rows.shape, -1, dtype=np.int32)
    for requester, slot in zip(*np.nonzero(response_view_rows >= 0), strict=True):
        identifier = int(received_ids[requester, slot])
        if identifier not in output_rows:
            raise ValueError(
                "Target ghost request is absent from the certified target CSR."
            )
        response_rows[requester, slot] = output_rows[identifier]
    return (
        request_ids,
        request_valid,
        response_rows,
        ghost_rows,
        ghost_owners,
        ghost_slots,
        capacity,
    )


def _source_coverage_exchange(
    overlap_source_ids: np.ndarray,
    overlap_source_owners: np.ndarray,
    overlap_measures: np.ndarray,
    local_source_ids: np.ndarray,
    local_source_native_measures: np.ndarray,
    source_vertices: tuple[np.ndarray, ...],
    dimension: int,
    tolerance: float,
    mesh: Mesh,
    axis: str,
    owner_count: int,
    /,
    *,
    overlap_errors: Any | None = None,
    local_source_errors: Any | None = None,
    local_source_bounds: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap_values = np.asarray(overlap_measures, dtype=np.float64)
    overlap_error_values = (
        np.zeros(overlap_values.shape, dtype=np.float64)
        if overlap_errors is None
        else np.asarray(overlap_errors, dtype=np.float64)
    )
    source_errors = (
        np.zeros(local_source_ids.shape, dtype=np.float64)
        if local_source_errors is None
        else np.asarray(local_source_errors, dtype=np.float64)
    )
    source_bounds = (
        None
        if local_source_bounds is None
        else np.asarray(local_source_bounds, dtype=np.float64)
    )
    input_error: Exception | None = None
    if (
        overlap_source_owners.shape != overlap_source_ids.shape
        or overlap_values.shape != overlap_source_ids.shape
        or overlap_error_values.shape != overlap_values.shape
        or local_source_native_measures.shape != local_source_ids.shape
        or source_errors.shape != local_source_ids.shape
        or np.unique(local_source_ids).size != local_source_ids.size
        or np.any((overlap_source_owners < 0) | (overlap_source_owners >= owner_count))
        or np.any(overlap_source_ids < 0)
        or np.any(~np.isfinite(overlap_values) | (overlap_values < 0.0))
        or np.any(~np.isfinite(overlap_error_values) | (overlap_error_values < 0.0))
        or np.any(
            ~np.isfinite(local_source_native_measures)
            | (local_source_native_measures <= 0.0)
        )
        or np.any(~np.isfinite(source_errors) | (source_errors < 0.0))
        or source_bounds is not None
        and (
            source_bounds.shape != (local_source_ids.size, 2, dimension)
            or np.any(~np.isfinite(source_bounds))
            or np.any(source_bounds[:, 0, :] > source_bounds[:, 1, :])
        )
    ):
        input_error = ValueError(
            "Source coverage contributions or evidence are malformed."
        )
    _synchronize_error(input_error, mesh, axis, owner_count, "source coverage inputs")

    totals: dict[tuple[int, int], tuple[float, float]] = {}
    for owner, identifier, measure, error_bound in zip(
        overlap_source_owners,
        overlap_source_ids,
        overlap_values,
        overlap_error_values,
        strict=True,
    ):
        key = (int(owner), int(identifier))
        previous_measure, previous_error = totals.get(key, (0.0, 0.0))
        totals[key] = (
            previous_measure + float(measure),
            previous_error + float(error_bound),
        )
    by_owner: list[list[tuple[int, float, float]]] = [[] for _ in range(owner_count)]
    for (owner, identifier), (measure, error_bound) in sorted(totals.items()):
        by_owner[owner].append((identifier, measure, error_bound))
    capacity = max(1, _group_max(sum(map(len, by_owner)), mesh, axis, owner_count))
    outgoing_ids = np.full((owner_count, capacity), -1, dtype=np.int64)
    outgoing_measures = np.zeros((owner_count, capacity), dtype=np.float64)
    outgoing_errors = np.zeros((owner_count, capacity), dtype=np.float64)
    outgoing_valid = np.zeros((owner_count, capacity), dtype=np.bool_)
    for owner, entries in enumerate(by_owner):
        for slot, (identifier, measure, error_bound) in enumerate(entries):
            outgoing_ids[owner, slot] = identifier
            outgoing_measures[owner, slot] = measure
            outgoing_errors[owner, slot] = error_bound
            outgoing_valid[owner, slot] = True
    incoming_ids = _all_to_all_local(outgoing_ids, mesh, axis, owner_count)
    incoming_measures = _all_to_all_local(outgoing_measures, mesh, axis, owner_count)
    incoming_errors = _all_to_all_local(outgoing_errors, mesh, axis, owner_count)
    incoming_valid = _all_to_all_local(outgoing_valid, mesh, axis, owner_count).astype(
        np.bool_
    )
    local_slot = {int(identifier): row for row, identifier in enumerate(local_source_ids)}
    covered = np.zeros(local_source_ids.shape, dtype=np.float64)
    covered_errors = np.zeros(local_source_ids.shape, dtype=np.float64)
    error: Exception | None = None
    try:
        for target_owner, slot in zip(*np.nonzero(incoming_valid), strict=True):
            identifier = int(incoming_ids[target_owner, slot])
            row = local_slot.get(identifier)
            if row is None:
                raise ValueError(
                    f"Source coverage contribution names unknown cell {identifier}."
                )
            covered[row] += incoming_measures[target_owner, slot]
            covered_errors[row] += incoming_errors[target_owner, slot]
    except Exception as caught:
        error = caught
    _synchronize_error(error, mesh, axis, owner_count, "source coverage ledger")
    defects = covered - local_source_native_measures
    if source_bounds is None:
        limits = _cell_coverage_tolerances(
            source_vertices,
            local_source_native_measures,
            tolerance,
            dimension=dimension,
        )
    else:
        diameter = np.linalg.norm(source_bounds[:, 1, :] - source_bounds[:, 0, :], axis=1)
        limits = (
            tolerance * local_source_native_measures
            + _COVERAGE_ROUNDOFF_ULPS * np.finfo(np.float64).eps * diameter**dimension
        )
    limits = limits + source_errors + covered_errors
    return defects, limits, covered


def prepare_owner_local_conservative_remap(
    source_view: OwnerLocalCellGeometry,
    target_view: OwnerLocalCellGeometry,
    execution_group: ExecutionGroup,
    /,
    *,
    transition: CellGeometryTransition | None = None,
    advance_id: str | None = None,
    axis_name: str = "owner_local_remap",
    tolerance: float = 1.0e-10,
    require_complete: bool = True,
) -> PreparedOwnerLocalConservativeRemap:
    """Prepare owner-routed native overlap CSR from compact local cell views.

    Straight-sided candidates are routed by compact physical bounds and returned
    only from their owning source ranks. Nonperiodic curved tetrahedra use exact
    transition maps into a shared root and the certified common-root integrator;
    corner maps never substitute for curved geometry. Neither path gathers a
    global mesh, field, or candidate-pair collection.
    """
    from .._view_support import OwnerLocalCellGeometry

    if not isinstance(source_view, OwnerLocalCellGeometry) or not isinstance(
        target_view, OwnerLocalCellGeometry
    ):
        raise TypeError("source_view and target_view must be OwnerLocalCellGeometry.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or tolerance_ >= 1.0:
        raise ValueError("tolerance must be finite and in (0, 1).")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be bool.")
    if advance_id is not None and (not isinstance(advance_id, str) or not advance_id):
        raise ValueError("advance_id must be a non-empty ID.")
    axis = str(axis_name).strip()
    mesh, local_rank, owner_count = _group_layout(execution_group, axis)
    metadata_error: Exception | None = None
    try:
        if (
            source_view.owner_rank != local_rank
            or target_view.owner_rank != local_rank
            or source_view.owner_directory_data.rank != local_rank
            or target_view.owner_directory_data.rank != local_rank
            or source_view.owner_directory_data.part_count != owner_count
            or target_view.owner_directory_data.part_count != owner_count
            or source_view.serial_source_view
            and owner_count > 1
            or target_view.serial_source_view
            and owner_count > 1
        ):
            raise ValueError(
                "Owner-local views do not belong to this execution-group rank."
            )
        if (
            advance_id is not None
            and transition is None
            and (
                source_view.local_mesh_id is not None
                or target_view.local_mesh_id is not None
            )
        ):
            raise ValueError(
                "A nonempty owner-local motion endpoint requires its CellGeometryTransition."
            )
        if (
            source_view.coordinate_contract_id != target_view.coordinate_contract_id
            or source_view.topological_dimension != target_view.topological_dimension
            or np.asarray(source_view.coordinates).shape[1]
            != np.asarray(target_view.coordinates).shape[1]
        ):
            raise ValueError(
                "Source and target views use different physical coordinate frames."
            )
        # Periodic endpoint quotients/lifts may differ. Each cell map is resolved
        # into this shared physical coordinate contract before AABB routing.
        transition_id = _validate_transition(transition, source_view, target_view)
        (
            source_kind,
            source_vertices_all,
            source_ids_all,
            source_owners_all,
            source_owned_all,
            source_declared,
            source_faces_all,
            source_straight,
            source_exact_vertices_all,
        ) = _view_maps(source_view)
        (
            target_kind,
            target_vertices_all,
            target_ids_all,
            target_owners_all,
            target_owned_all,
            target_declared,
            target_faces_all,
            target_straight,
            target_exact_vertices_all,
        ) = _view_maps(target_view)
        if source_view.topological_dimension != target_view.topological_dimension:
            raise ValueError("Source and target local cell dimensions differ.")
        source_binding = _view_binding(source_view)
        target_binding = _view_binding(target_view)
        # Exact coordinate maps and their source contexts are bound to each rank's
        # resident mesh and support set (an empty owner binds its source value), so
        # their IDs are rank-local evidence checked against the rank's own geometry
        # authority; the group compares the published source, manifest, and
        # directory identities.
        rank_local_binding_fields = (
            "local_mesh_id",
            "local_topology_id",
            "local_geometry_id",
            "local_view_id",
            "owner_rank",
            "source_map_id",
            "source_context_id",
        )
        source_signature = {
            key: value
            for key, value in source_binding.items()
            if key not in rank_local_binding_fields
        }
        target_signature = {
            key: value
            for key, value in target_binding.items()
            if key not in rank_local_binding_fields
        }
        global_signature = {
            "source": source_signature,
            "target": target_signature,
            "coordinate_contract": source_view.coordinate_contract_id,
            "cell_kinds": (source_kind, target_kind),
            "dimension": source_view.topological_dimension,
            "advance_id": advance_id,
            "execution_group_id": execution_group.spec.group_id,
        }
    except Exception as caught:
        metadata_error = caught
        transition_id = None
        source_kind = target_kind = ""
        source_vertices_all = target_vertices_all = ()
        source_ids_all = target_ids_all = np.empty((0,), dtype=np.int64)
        source_owners_all = target_owners_all = np.empty((0,), dtype=np.int32)
        source_owned_all = target_owned_all = np.empty((0,), dtype=np.bool_)
        source_declared = target_declared = np.empty((0,), dtype=np.float64)
        source_faces_all = target_faces_all = None
        source_exact_vertices_all = target_exact_vertices_all = ()
        source_straight = target_straight = True
        global_signature = None
    _synchronize_error(metadata_error, mesh, axis, owner_count, "view validation")
    if global_signature is None:
        raise RuntimeError("Owner-local metadata validation omitted the group signature.")
    dimension = int(source_view.topological_dimension)
    source_kind = _resolve_group_cell_kind(
        source_kind, dimension, mesh, axis, owner_count, "source"
    )
    target_kind = _resolve_group_cell_kind(
        target_kind, dimension, mesh, axis, owner_count, "target"
    )
    if source_kind == "polyhedron" and source_faces_all is None:
        source_faces_all = ()
    if target_kind == "polyhedron" and target_faces_all is None:
        target_faces_all = ()
    local_curved_map = bool(
        (source_ids_all.size and not source_straight)
        or (target_ids_all.size and not target_straight)
    )
    curved_map_group = bool(
        np.any(
            _all_gather_local(
                np.asarray(local_curved_map, dtype=np.int32),
                mesh,
                axis,
                owner_count,
            )
        )
    )
    global_signature["curved_map_group"] = curved_map_group
    global_signature["cell_kinds"] = (source_kind, target_kind)
    _require_group_identity(global_signature, mesh, axis, owner_count, "view identity")
    source_view.require_owner_directory_current(
        execution_group, axis_name=f"{axis}_source_directory"
    )
    target_view.require_owner_directory_current(
        execution_group, axis_name=f"{axis}_target_directory"
    )

    owned_source_rows = np.flatnonzero(source_owned_all).astype(np.int32)
    owned_target_rows = np.flatnonzero(target_owned_all).astype(np.int32)
    source_owned_ids = source_ids_all[owned_source_rows]
    target_owned_ids = target_ids_all[owned_target_rows]
    source_owned_vertices = tuple(source_vertices_all[row] for row in owned_source_rows)
    target_owned_vertices = tuple(target_vertices_all[row] for row in owned_target_rows)
    source_owned_exact_vertices = tuple(
        source_exact_vertices_all[row] for row in owned_source_rows
    )
    target_owned_exact_vertices = tuple(
        target_exact_vertices_all[row] for row in owned_target_rows
    )
    source_owned_faces = (
        None
        if source_faces_all is None
        else tuple(source_faces_all[row] for row in owned_source_rows)
    )
    target_owned_faces = (
        None
        if target_faces_all is None
        else tuple(target_faces_all[row] for row in owned_target_rows)
    )
    source_owned_declared = source_declared[owned_source_rows]
    target_owned_declared = target_declared[owned_target_rows]
    source_cell_owners = source_owners_all[owned_source_rows]
    target_owned_view_rows = owned_target_rows.copy()
    target_refinement_to_owned_rows: np.ndarray | None = None
    periodic_domain_clipping = False
    source_geometry_error: Exception | None = None
    target_geometry_error: Exception | None = None
    source_geometry_refinement: PreparedCommonRefinement | None = None
    target_geometry_refinement: PreparedCommonRefinement | None = None
    source_owned_measures = np.empty(source_owned_ids.shape, dtype=np.float64)
    target_owned_measures = np.empty(target_owned_ids.shape, dtype=np.float64)
    source_geometry_measure_errors = np.empty((0,), dtype=np.float64)
    target_geometry_measure_errors = np.empty((0,), dtype=np.float64)
    source_coverage_errors: np.ndarray | None = None
    source_coverage_bounds: np.ndarray | None = None
    overlap_integration_errors: np.ndarray | None = None
    curved_integration_evidence = None
    candidate_result: tuple[Any, ...] | None = None
    if curved_map_group:
        curved_result: tuple[Any, ...] | None = None
        curved_error: Exception | None = None
        try:
            from ._owner_local_curved_root import (
                prepare_owner_local_curved_root_refinement,
            )

            curved_result = prepare_owner_local_curved_root_refinement(
                source_view,
                target_view,
                source_kind,
                target_kind,
                source_ids_all,
                source_owners_all,
                source_declared,
                target_ids_all,
                target_owned_ids,
                target_declared,
                source_owned_ids,
                transition,
                tolerance_,
                require_complete,
                execution_group,
                mesh,
                axis,
                local_rank,
                owner_count,
            )
        except Exception as caught:
            curved_error = caught
        _synchronize_error(
            curved_error, mesh, axis, owner_count, "curved shared-root preparation"
        )
        if curved_result is None:
            raise RuntimeError("Curved owner-local preparation omitted its evidence.")
        (
            local_refinement,
            source_geometry_refinement,
            target_geometry_refinement,
            curved_integration_evidence,
            candidate_ids,
            candidate_owners,
            candidate_measures,
            candidate_vertices,
            candidate_periodic_ids,
            candidate_windings,
            candidate_faces,
            source_owned_measures,
            target_owned_measures,
            source_coverage_errors,
            source_coverage_bounds,
            overlap_integration_errors,
        ) = curved_result
        candidate_result = (
            candidate_owners,
            candidate_ids.copy(),
            candidate_ids,
            candidate_measures,
            candidate_vertices,
            candidate_periodic_ids,
            candidate_windings,
            candidate_faces,
        )
    else:
        source_periodic_ids, source_windings = _periodic_cells(
            source_view, owned_source_rows
        )
        target_periodic_ids, target_windings = _periodic_cells(
            target_view, owned_target_rows
        )
        periodic_lattice = _periodic_overlap_lattice(source_view, target_view, dimension)
        periodic_domain_clipping = bool(
            dimension == 3
            and source_view.periodic_lift_id is not None
            and target_view.periodic_lift_id is not None
            and source_view.periodic_lift_id != target_view.periodic_lift_id
            and source_view.periodic_lift_matrix is not None
            and target_view.periodic_lift_matrix is not None
            and np.asarray(source_view.periodic_lift_matrix).shape
            == (dimension, dimension)
            and np.asarray(target_view.periodic_lift_matrix).shape
            == (dimension, dimension)
        )
        source_geometry_kind = source_kind
        target_geometry_kind = target_kind
        source_geometry_ids = source_owned_ids
        source_geometry_stable_ids = source_owned_ids
        source_geometry_owners = source_cell_owners
        source_geometry_periodic_ids = source_periodic_ids
        source_geometry_windings = source_windings
        source_geometry_vertices = source_owned_vertices
        source_geometry_faces = source_owned_faces
        source_geometry_declared = source_owned_declared
        target_geometry_ids = target_owned_ids
        target_geometry_stable_ids = target_owned_ids
        target_geometry_vertices = target_owned_vertices
        target_geometry_faces = target_owned_faces
        target_geometry_declared = target_owned_declared
        if periodic_domain_clipping:
            source_origin = _periodic_view_origin(source_view, mesh, axis, owner_count)
            target_origin = _periodic_view_origin(target_view, mesh, axis, owner_count)
            clipping_error: Exception | None = None
            try:
                (
                    source_geometry_ids,
                    source_geometry_stable_ids,
                    source_geometry_owners,
                    source_geometry_declared,
                    source_geometry_vertices,
                    source_geometry_periodic_ids,
                    source_geometry_windings,
                    source_geometry_faces,
                    source_geometry_measure_errors,
                ) = _periodic_domain_cell_images(
                    source_kind,
                    source_owned_ids,
                    source_cell_owners,
                    source_periodic_ids,
                    source_windings,
                    source_owned_vertices,
                    source_owned_exact_vertices,
                    source_owned_faces,
                    np.asarray(source_view.periodic_lift_matrix, dtype=np.float64),
                    source_origin,
                )
                (
                    target_geometry_ids,
                    target_geometry_stable_ids,
                    _,
                    target_geometry_declared,
                    target_geometry_vertices,
                    _,
                    _,
                    target_geometry_faces,
                    target_geometry_measure_errors,
                ) = _periodic_domain_cell_images(
                    target_kind,
                    target_owned_ids,
                    target_owners_all[owned_target_rows],
                    target_periodic_ids,
                    target_windings,
                    target_owned_vertices,
                    target_owned_exact_vertices,
                    target_owned_faces,
                    np.asarray(target_view.periodic_lift_matrix, dtype=np.float64),
                    target_origin,
                )
                source_geometry_kind = "polyhedron"
                target_geometry_kind = "polyhedron"
                periodic_lattice = None
            except Exception as caught:
                clipping_error = caught
            _synchronize_error(
                clipping_error, mesh, axis, owner_count, "periodic domain clipping"
            )

        source_native_fragments: np.ndarray | None = None
        target_native_fragments: np.ndarray | None = None
        target_coverage_errors: np.ndarray | None = None
        try:
            source_geometry_refinement, source_native_fragments, _ = (
                _local_geometry_certificate(
                    source_geometry_kind,
                    source_geometry_vertices,
                    source_geometry_ids,
                    source_geometry_declared,
                    tolerance_,
                    faces=source_geometry_faces,
                )
            )
            source_owned_measures = (
                _aggregate_periodic_fragment_measures(
                    source_owned_ids,
                    source_geometry_stable_ids,
                    source_native_fragments,
                )
                if periodic_domain_clipping
                else source_native_fragments
            )
            if periodic_domain_clipping:
                source_coverage_errors = _aggregate_periodic_fragment_errors(
                    source_owned_ids,
                    source_geometry_stable_ids,
                    source_geometry_measure_errors,
                )
                source_limits = (
                    _cell_coverage_tolerances(
                        source_owned_vertices,
                        source_owned_declared,
                        tolerance_,
                        dimension=dimension,
                    )
                    + source_coverage_errors
                )
                if np.any(
                    np.abs(source_owned_measures - source_owned_declared) > source_limits
                ):
                    raise ValueError(
                        "Periodic source images do not reproduce the accepted cell measures."
                    )
        except Exception as caught:
            source_geometry_error = caught
        try:
            target_geometry_refinement, target_native_fragments, _ = (
                _local_geometry_certificate(
                    target_geometry_kind,
                    target_geometry_vertices,
                    target_geometry_ids,
                    target_geometry_declared,
                    tolerance_,
                    faces=target_geometry_faces,
                )
            )
            target_owned_measures = (
                _aggregate_periodic_fragment_measures(
                    target_owned_ids,
                    target_geometry_stable_ids,
                    target_native_fragments,
                )
                if periodic_domain_clipping
                else target_native_fragments
            )
            if periodic_domain_clipping:
                target_coverage_errors = _aggregate_periodic_fragment_errors(
                    target_owned_ids,
                    target_geometry_stable_ids,
                    target_geometry_measure_errors,
                )
                target_limits = (
                    _cell_coverage_tolerances(
                        target_owned_vertices,
                        target_owned_declared,
                        tolerance_,
                        dimension=dimension,
                    )
                    + target_coverage_errors
                )
                if np.any(
                    np.abs(target_owned_measures - target_owned_declared) > target_limits
                ):
                    raise ValueError(
                        "Periodic target images do not reproduce the accepted cell measures."
                    )
        except Exception as caught:
            target_geometry_error = caught
        _synchronize_error(
            source_geometry_error or target_geometry_error,
            mesh,
            axis,
            owner_count,
            "native cell-map certification",
        )
        if source_geometry_refinement is None or target_geometry_refinement is None:
            raise RuntimeError(
                "Owner-local cell-map certification omitted its native evidence."
            )

        source_measure_by_id = {
            int(identifier): float(measure)
            for identifier, measure in zip(
                source_owned_ids, source_owned_measures, strict=True
            )
        }
        source_candidate_measures = np.asarray(
            [
                source_measure_by_id[int(identifier)]
                for identifier in source_geometry_stable_ids
            ],
            dtype=np.float64,
        )
        periodic_inverse = (
            None if periodic_lattice is None else np.linalg.inv(periodic_lattice)
        )
        source_query_vertices = (
            source_geometry_vertices
            if periodic_inverse is None
            else tuple(
                vertices @ periodic_inverse for vertices in source_geometry_vertices
            )
        )
        rank_bounds = _rank_source_bounds(source_query_vertices, dimension)
        target_request_ids, target_request_bounds, target_request_valid = _target_queries(
            target_geometry_ids,
            target_geometry_vertices,
            dimension,
            rank_bounds,
            mesh,
            axis,
            owner_count,
            periodic_lattice,
        )
        candidate_error: Exception | None = None
        try:
            candidate_result = _route_candidate_source_cells(
                source_geometry_stable_ids,
                source_geometry_owners,
                source_candidate_measures,
                source_geometry_vertices,
                source_geometry_faces,
                source_geometry_periodic_ids,
                source_geometry_windings,
                target_request_ids,
                target_request_bounds,
                target_request_valid,
                dimension,
                mesh,
                axis,
                owner_count,
                periodic_lattice,
            )
        except Exception as caught:
            candidate_error = caught
        _synchronize_error(
            candidate_error, mesh, axis, owner_count, "AABB source-map routing"
        )
    if candidate_result is None:
        raise RuntimeError(
            "Owner-local source-map routing omitted its candidate records."
        )
    (
        candidate_owners,
        candidate_geometry_ids,
        candidate_ids,
        candidate_measures,
        candidate_vertices,
        candidate_periodic_ids,
        candidate_windings,
        candidate_faces,
    ) = candidate_result

    if not curved_map_group:
        refinement_error: Exception | None = None
        local_refinement: PreparedCommonRefinement | None = None
        try:
            policy = CommonRefinementPolicy(
                coverage=(
                    CommonRefinementCoverage.TARGET
                    if require_complete
                    else CommonRefinementCoverage.PARTIAL
                ),
                coverage_tolerance=tolerance_,
                second_moments=True,
            )
            if (
                source_geometry_kind == "polyhedron"
                or target_geometry_kind == "polyhedron"
            ):
                local_refinement = _prepare_variable_polyhedral_common_refinement(
                    source_geometry_kind,
                    candidate_vertices,
                    candidate_geometry_ids,
                    candidate_faces,
                    target_geometry_kind,
                    target_geometry_vertices,
                    target_geometry_ids,
                    target_geometry_faces,
                    policy,
                )
            else:
                local_refinement = prepare_compact_cell_common_refinement(
                    candidate_vertices,
                    candidate_geometry_ids,
                    target_geometry_vertices,
                    target_geometry_ids,
                    source_cell_kind=source_geometry_kind,
                    target_cell_kind=target_geometry_kind,
                    source_cell_faces=candidate_faces,
                    target_cell_faces=target_geometry_faces,
                    policy=policy,
                )
        except Exception as caught:
            refinement_error = caught
        _synchronize_error(
            refinement_error, mesh, axis, owner_count, "native physical intersection"
        )
        if local_refinement is None:
            raise RuntimeError("Native owner-local intersection omitted its evidence.")

    source_refinement_ids = np.asarray(
        local_refinement.source_cell_global_ids, dtype=np.int64
    )
    target_refinement_ids = np.asarray(
        local_refinement.target_cell_global_ids, dtype=np.int64
    )
    candidate_index = {
        int(identifier): row for row, identifier in enumerate(candidate_geometry_ids)
    }
    target_index = {
        int(identifier): row for row, identifier in enumerate(target_owned_ids)
    }
    target_map_error: Exception | None = None
    try:
        candidate_order = np.asarray(
            [candidate_index[int(identifier)] for identifier in source_refinement_ids],
            dtype=np.int32,
        )
        if periodic_domain_clipping:
            stable_by_geometry_id = {
                int(geometry_id): int(stable_id)
                for geometry_id, stable_id in zip(
                    target_geometry_ids, target_geometry_stable_ids, strict=True
                )
            }
            target_refinement_to_owned_rows = np.asarray(
                [
                    target_index[stable_by_geometry_id[int(identifier)]]
                    for identifier in target_refinement_ids
                ],
                dtype=np.int32,
            )
        else:
            target_order = np.asarray(
                [target_index[int(identifier)] for identifier in target_refinement_ids],
                dtype=np.int32,
            )
    except KeyError as caught:
        target_map_error = ValueError(
            "Native compact mesh lost stable source/target cell identity."
        )
        target_map_error.__cause__ = caught
    _synchronize_error(
        target_map_error, mesh, axis, owner_count, "native stable-cell routing"
    )
    candidate_geometry_ids = candidate_geometry_ids[candidate_order]
    candidate_ids = candidate_ids[candidate_order]
    candidate_owners = candidate_owners[candidate_order]
    candidate_measures = candidate_measures[candidate_order]
    candidate_periodic_ids = candidate_periodic_ids[candidate_order]
    candidate_windings = candidate_windings[candidate_order]
    if not periodic_domain_clipping:
        target_owned_ids = target_owned_ids[target_order]
        target_owned_measures = target_owned_measures[target_order]
        target_owned_view_rows = owned_target_rows[target_order]

    target_defects = np.asarray(
        local_refinement.evidence.target_coverage_defects, dtype=np.float64
    )
    target_limits = np.asarray(
        local_refinement.evidence.target_coverage_tolerances, dtype=np.float64
    )
    source_indices = np.asarray(local_refinement.source_cells, dtype=np.int32)
    local_source_ids_by_overlap = candidate_ids[source_indices]
    local_source_owners_by_overlap = candidate_owners[source_indices]
    if periodic_domain_clipping:
        if target_refinement_to_owned_rows is None:
            raise RuntimeError(
                "Periodic target clipping omitted fragment-to-cell routes."
            )
        target_fragment_rows = np.asarray(target_refinement_to_owned_rows, dtype=np.intp)
        if (
            target_fragment_rows.shape != target_defects.shape
            or target_limits.shape != target_defects.shape
        ):
            raise ValueError(
                "Periodic target fragments lost their coverage evidence rows."
            )
        target_defects = np.bincount(
            target_fragment_rows,
            weights=np.asarray(target_defects, dtype=np.float64),
            minlength=target_owned_ids.size,
        )
        target_limits = np.bincount(
            target_fragment_rows,
            weights=np.asarray(target_limits, dtype=np.float64),
            minlength=target_owned_ids.size,
        )
        if source_coverage_errors is None or target_coverage_errors is None:
            raise RuntimeError("Periodic clipping omitted realization errors.")
        target_limits += target_coverage_errors
        stable_by_geometry_id = {
            int(geometry_id): int(stable_id)
            for geometry_id, stable_id in zip(
                target_geometry_ids, target_geometry_stable_ids, strict=True
            )
        }
        source_error_by_id = {
            int(identifier): float(error)
            for identifier, error in zip(
                source_owned_ids, source_coverage_errors, strict=True
            )
        }
        target_error_by_id = {
            int(identifier): float(error)
            for identifier, error in zip(
                target_owned_ids, target_coverage_errors, strict=True
            )
        }
        target_indices = np.asarray(local_refinement.target_cells, dtype=np.int32)
        if (
            target_indices.shape != local_source_ids_by_overlap.shape
            or np.any(target_indices < 0)
            or np.any(target_indices >= target_refinement_ids.size)
        ):
            raise ValueError("Periodic overlap rows lost their target-fragment routes.")
        overlap_target_geometry_ids = target_refinement_ids[target_indices]
        overlap_target_stable_ids = np.asarray(
            [
                stable_by_geometry_id[int(identifier)]
                for identifier in overlap_target_geometry_ids
            ],
            dtype=np.int64,
        )
        overlap_integration_errors = np.asarray(
            [
                source_error_by_id[int(source_id)] + target_error_by_id[int(target_id)]
                for source_id, target_id in zip(
                    local_source_ids_by_overlap,
                    overlap_target_stable_ids,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
    source_defects, source_limits, _ = _source_coverage_exchange(
        local_source_ids_by_overlap,
        local_source_owners_by_overlap,
        np.asarray(local_refinement.volumes, dtype=np.float64),
        source_owned_ids,
        source_owned_measures,
        source_owned_vertices,
        dimension,
        tolerance_,
        mesh,
        axis,
        owner_count,
        overlap_errors=overlap_integration_errors,
        local_source_errors=source_coverage_errors,
        local_source_bounds=source_coverage_bounds,
    )
    source_output_ids = source_owned_ids
    source_output_volumes = np.asarray(source_owned_measures, dtype=np.float64)

    (
        request_ids_local,
        request_valid_local,
        overlap_request_slots,
        source_response_rows,
        source_request_capacity,
    ) = _build_request_table(
        local_source_ids_by_overlap,
        local_source_owners_by_overlap,
        source_ids_all,
        source_owned_all,
        owner_count,
        mesh,
        axis,
    )
    (
        ghost_request_ids,
        ghost_request_valid,
        ghost_response_rows,
        ghost_view_rows,
        ghost_owners,
        ghost_slots,
        ghost_capacity,
    ) = _build_ghost_requests(
        target_view,
        target_owned_ids,
        local_rank,
        owner_count,
        mesh,
        axis,
    )

    local_source_bad = bool(np.any(np.abs(source_defects) > source_limits))
    local_target_bad = bool(np.any(np.abs(target_defects) > target_limits))
    local_double = bool(
        np.any(source_defects > source_limits) or np.any(target_defects > target_limits)
    )
    if local_double:
        local_status = CommonRefinementStatus.DOUBLE_COVERAGE
        local_reason = "source or target cells are covered more than once"
    elif local_refinement.status is not CommonRefinementStatus.SUCCESS:
        local_status = local_refinement.status
        local_reason = local_refinement.evidence.reason
    elif local_source_bad or local_target_bad:
        local_status = CommonRefinementStatus.COVERAGE_GAP
        local_reason = "complete source/target physical coverage requirement is not met"
    else:
        local_status = CommonRefinementStatus.SUCCESS
        local_reason = "certified owner-local physical common refinement"
    statuses = _all_gather_local(
        np.asarray(int(local_status), dtype=np.int32), mesh, axis, owner_count
    )
    status_priority = {
        CommonRefinementStatus.SUCCESS: 0,
        CommonRefinementStatus.COVERAGE_GAP: 1,
        CommonRefinementStatus.DOUBLE_COVERAGE: 2,
        CommonRefinementStatus.INTERSECTION_FAILURE: 3,
        CommonRefinementStatus.PREDICATE_UNCERTAIN: 4,
        CommonRefinementStatus.INVALID_GEOMETRY: 5,
    }
    status = max(
        (CommonRefinementStatus(int(value)) for value in statuses),
        key=status_priority.__getitem__,
    )
    local_coverage_bad = _all_gather_local(
        np.asarray(local_source_bad or local_target_bad, dtype=np.int32),
        mesh,
        axis,
        owner_count,
    )
    coverage_complete = bool(
        not np.any(local_coverage_bad) and status is CommonRefinementStatus.SUCCESS
    )
    reason = (
        local_reason
        if status is local_status
        else f"group common refinement status {status.name}; {local_reason}"
    )
    source_measure_global = float(
        np.sum(
            _all_gather_local(
                np.asarray(np.sum(source_output_volumes), dtype=np.float64),
                mesh,
                axis,
                owner_count,
            )
        )
    )
    target_measure_global = float(
        np.sum(
            _all_gather_local(
                np.asarray(np.sum(target_owned_measures), dtype=np.float64),
                mesh,
                axis,
                owner_count,
            )
        )
    )
    return PreparedOwnerLocalConservativeRemap(
        advance_id=advance_id,
        source_view=source_view,
        target_view=target_view,
        execution_group=execution_group,
        axis_name=axis,
        local_rank=local_rank,
        transition_id=transition_id,
        source_kind=source_kind,
        target_kind=target_kind,
        source_local_cell_ids=source_ids_all,
        source_local_cell_owner_ranks=source_owners_all,
        source_local_cell_owned=source_owned_all,
        source_cell_ids=source_output_ids,
        source_volumes=source_output_volumes,
        target_local_cell_ids=target_ids_all,
        target_local_cell_owner_ranks=target_owners_all,
        target_local_cell_owned=target_owned_all,
        target_cell_ids=target_owned_ids,
        target_volumes=target_owned_measures,
        target_owned_view_rows=target_owned_view_rows,
        target_ghost_view_rows=ghost_view_rows,
        target_ghost_owners=ghost_owners,
        target_ghost_slots=ghost_slots,
        target_ghost_request_ids=ghost_request_ids,
        target_ghost_request_valid=ghost_request_valid,
        target_ghost_response_rows=ghost_response_rows,
        refinement=local_refinement,
        source_candidate_ids=candidate_ids,
        source_candidate_owners=candidate_owners,
        source_candidate_periodic_ids=candidate_periodic_ids,
        source_candidate_windings=candidate_windings,
        source_candidate_measures=candidate_measures,
        overlap_request_slots=overlap_request_slots,
        source_request_ids=request_ids_local,
        source_request_valid=request_valid_local,
        source_response_rows=source_response_rows,
        source_request_capacity=source_request_capacity,
        target_ghost_capacity=ghost_capacity,
        source_coverage_defects=source_defects,
        source_coverage_tolerances=source_limits,
        target_coverage_defects=target_defects,
        target_coverage_tolerances=target_limits,
        coverage_complete=coverage_complete,
        status=status,
        reason=reason,
        source_geometry_evidence=source_geometry_refinement.evidence,
        target_geometry_evidence=target_geometry_refinement.evidence,
        source_measure=source_measure_global,
        target_measure=target_measure_global,
        tolerance=tolerance_,
        require_complete=require_complete,
        curved_integration_evidence=curved_integration_evidence,
        target_fragment_routes=target_refinement_to_owned_rows,
    )


__all__ = (
    "OwnerLocalConservativeRemapEvidence",
    "OwnerLocalConservativeRemapResult",
    "PreparedOwnerLocalConservativeRemap",
    "prepare_owner_local_conservative_remap",
)
