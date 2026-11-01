#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native advancing boundary layers grown from triangle and quadrilateral walls.

The front is a column complex. Every wall vertex carries one column per smooth
sector (sectors split at convex ridges under the FAN corner policy), fan columns
interpolate across convex ridges, and corner patches close vertices where three
or more ridges meet. Columns point along the visibility-optimal direction
``argmax_{|d|=1} min_i n_i.d`` over the sector's face normals: the normalized
minimum-norm point of their convex hull, found exactly by enumerating the hull
faces of at most three normals and certified by a closed weak-duality gap.
Column heights follow
the explicit schedule, scaled by smooth concave curvature and by BVH-nearest
medial-crossing probes. Every layer is certified by Bernstein validity and by
exact-predicate intersection of its cell faces against earlier layers, the wall,
and the obstacles before it is accepted; collisions resolve only through the
control's explicit collision policy, and every rejection carries the offending
wall vertices and locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._bvh import bvh_nearest_items, bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import load_meshcore
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..discretization import (
    CellBlock,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
    point_triangle_distance,
)
from ..geometry._source_identity import source_entity_id
from ..typing import checked
from ._association import _incidence_pairs, GeometryAssociation
from ._audit_topology import _orient3d, _triangle_pairs_intersect
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import (
    BoundaryLayerCollisionPolicy,
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
)
from ._scope import MeshingEntityKind
from ._trace import MeshingStageKind


_STAGE = MeshingStageKind.LAYER_GENERATION.value
_BLOCK_NAMES = {
    "tetrahedron": "tetrahedra",
    "pyramid": "pyramids",
    "prism": "prisms",
    "hexahedron": "hexahedra",
    "triangle": "triangles",
    "quadrilateral": "quadrilaterals",
}
_VOLUME_KINDS = ("tetrahedron", "pyramid", "prism", "hexahedron")
_CELL_ARITY = {"tetrahedron": 4, "pyramid": 5, "prism": 6, "hexahedron": 8}
_TETRAHEDRON_FACE_ROUTES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
# Outward face loops per cell kind (phydrax reference orientation).
_FACE_ROUTES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tetrahedron": _TETRAHEDRON_FACE_ROUTES,
    "simplex:3": _TETRAHEDRON_FACE_ROUTES,
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    "prism": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}
_WALL, _STRIP, _CORNER = 0, 1, 2
_SECTOR, _FAN, _CENTER = 0, 1, 2


# ---------------------------------------------------------------- public records


class BoundaryLayerPolicy(StrictModule, NonTrainableState):
    """Numerical controls of native advancing-layer generation.

    ``proximity_fraction`` and ``curvature_fraction`` bound column heights by the
    medial crossing found with ``proximity_samples`` BVH-nearest probes per
    column and by the smooth concave radius of curvature. A foreign surface
    crosses a probe when it is nearer than ``1 - medial_tolerance`` times the
    distance to the column's own wall. Collision resolution scales colliding
    columns by ``reduction_factor`` until certified, at the physical thickness
    floor, or unable to make representable progress.
    ``simplex_cap`` closes quadrilateral cap faces with transition pyramids so
    the cap is a triangle surface for simplex core fill.
    """

    proximity_samples: int = eqx.field(static=True)
    proximity_fraction: float = eqx.field(static=True)
    curvature_fraction: float = eqx.field(static=True)
    medial_tolerance: float = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    simplex_cap: bool = eqx.field(static=True)
    validity: CellValidityPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        proximity_samples: int = 16,
        proximity_fraction: float = 0.8,
        curvature_fraction: float = 0.8,
        medial_tolerance: float = 0.05,
        reduction_factor: float = 0.7,
        simplex_cap: bool = True,
        validity: CellValidityPolicy | None = None,
    ) -> None:
        if isinstance(proximity_samples, bool) or not isinstance(proximity_samples, int):
            raise TypeError("proximity_samples must be an integer.")
        if proximity_samples < 1:
            raise ValueError("proximity_samples must be positive.")
        fractions = {
            name: finite_real_scalar(value, name)
            for name, value in (
                ("proximity_fraction", proximity_fraction),
                ("curvature_fraction", curvature_fraction),
                ("reduction_factor", reduction_factor),
            )
        }
        if any(not 0.0 < value < 1.0 for value in fractions.values()):
            raise ValueError(
                "proximity_fraction, curvature_fraction, and reduction_factor must lie in (0, 1)."
            )
        tolerance = finite_real_scalar(medial_tolerance, "medial_tolerance")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("medial_tolerance must lie in [0, 1).")
        if not isinstance(simplex_cap, bool):
            raise TypeError("simplex_cap must be a bool.")
        validity_ = CellValidityPolicy() if validity is None else validity
        if not isinstance(validity_, CellValidityPolicy):
            raise TypeError("validity must be CellValidityPolicy or None.")
        self.proximity_samples = proximity_samples
        self.proximity_fraction = fractions["proximity_fraction"]
        self.curvature_fraction = fractions["curvature_fraction"]
        self.medial_tolerance = tolerance
        self.reduction_factor = fractions["reduction_factor"]
        self.simplex_cap = simplex_cap
        self.validity = validity_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-policy",
                "proximity_samples": proximity_samples,
                **fractions,
                "medial_tolerance": tolerance,
                "simplex_cap": simplex_cap,
                "validity": validity_.policy_id,
            }
        )


class BoundaryLayerEvidence(StrictModule, NonTrainableState):
    """Measured layer realization and collision-resolution evidence.

    Thicknesses are measured as increments of the exact distance from each
    column's layer vertices to the wall surface (BVH-nearest point-triangle
    distances); per-layer statistics cover every column that carries the layer.
    ``layer_active[k]`` is ``True`` iff at least one surviving column carries
    requested layer ``k``. A requested layer carried by no surviving column has
    ``layer_active[k] == False`` and NaN achieved, minimum, and maximum
    thickness; every growth rate touching it
    (``achieved_growth_rates[k] = achieved_thicknesses[k + 1] /
    achieved_thicknesses[k]``) is NaN. The explicit ``terminated_vertex_count``,
    ``reduced_vertex_count``, and ``merged_vertex_count`` record why columns
    stopped short of or shrank the requested schedule.
    """

    requested_thicknesses: tuple[float, ...] = eqx.field(static=True)
    achieved_thicknesses: Array
    minimum_thicknesses: Array
    maximum_thicknesses: Array
    achieved_growth_rates: Array
    layer_active: Array
    column_count: int = eqx.field(static=True)
    fan_column_count: int = eqx.field(static=True)
    corner_patch_count: int = eqx.field(static=True)
    convex_ridge_count: int = eqx.field(static=True)
    concave_ridge_count: int = eqx.field(static=True)
    rim_vertex_count: int = eqx.field(static=True)
    minimum_visibility: float = eqx.field(static=True)
    maximum_stretch: float = eqx.field(static=True)
    collision_policy: BoundaryLayerCollisionPolicy = eqx.field(static=True)
    predicted_collision_vertex_count: int = eqx.field(static=True)
    detected_collision_count: int = eqx.field(static=True)
    resolution_iterations: int = eqx.field(static=True)
    reduced_vertex_count: int = eqx.field(static=True)
    terminated_vertex_count: int = eqx.field(static=True)
    merged_vertex_count: int = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    cell_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    certified_valid_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, **values: Any) -> None:
        requested = tuple(float(value) for value in values["requested_thicknesses"])
        if (
            not requested
            or not np.all(np.isfinite(requested))
            or np.any(np.asarray(requested) <= 0.0)
        ):
            raise ValueError("requested_thicknesses must be finite and positive.")
        layer_active = np.asarray(values["layer_active"], dtype=np.bool_)
        if layer_active.shape != (len(requested),):
            raise ValueError("layer_active must hold one entry per requested layer.")
        arrays = {
            name: np.asarray(values[name], dtype=np.float64)
            for name in (
                "achieved_thicknesses",
                "minimum_thicknesses",
                "maximum_thicknesses",
                "achieved_growth_rates",
            )
        }
        layer_shape = (len(requested),)
        if any(arrays[name].shape != layer_shape for name in tuple(arrays)[:3]) or arrays[
            "achieved_growth_rates"
        ].shape != (len(requested) - 1,):
            raise ValueError(
                "Boundary-layer thickness and growth arrays have invalid shapes."
            )
        for name in (
            "achieved_thicknesses",
            "minimum_thicknesses",
            "maximum_thicknesses",
        ):
            array = arrays[name]
            if np.any(~np.isfinite(array[layer_active])) or np.any(
                ~np.isnan(array[~layer_active])
            ):
                raise ValueError(
                    f"{name} must be finite on active layers and NaN otherwise."
                )
        growth_active = layer_active[:-1] & layer_active[1:]
        growth = arrays["achieved_growth_rates"]
        if np.any(~np.isfinite(growth[growth_active])) or np.any(
            ~np.isnan(growth[~growth_active])
        ):
            raise ValueError(
                "achieved_growth_rates must be finite between active layers and NaN otherwise."
            )
        for name, array in arrays.items():
            setattr(self, name, jnp.asarray(array))
        self.layer_active = jnp.asarray(layer_active)
        self.requested_thicknesses = requested
        integers = (
            "column_count",
            "fan_column_count",
            "corner_patch_count",
            "convex_ridge_count",
            "concave_ridge_count",
            "rim_vertex_count",
            "predicted_collision_vertex_count",
            "detected_collision_count",
            "resolution_iterations",
            "reduced_vertex_count",
            "terminated_vertex_count",
            "merged_vertex_count",
            "certified_valid_count",
        )
        for name in integers:
            setattr(self, name, int(values[name]))
        for name in ("minimum_visibility", "maximum_stretch", "minimum_scale"):
            setattr(self, name, float(values[name]))
        policy = values["collision_policy"]
        if not isinstance(policy, BoundaryLayerCollisionPolicy):
            raise TypeError("collision_policy must be BoundaryLayerCollisionPolicy.")
        self.collision_policy = policy
        self.cell_counts = tuple(
            (str(kind), int(count)) for kind, count in values["cell_counts"]
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-evidence",
                "requested": self.requested_thicknesses,
                "arrays": {
                    **{name: array_tree_fingerprint(arrays[name]) for name in arrays},
                    "layer_active": array_tree_fingerprint(layer_active),
                },
                **{name: int(values[name]) for name in integers},
                "minimum_visibility": self.minimum_visibility,
                "maximum_stretch": self.maximum_stretch,
                "minimum_scale": self.minimum_scale,
                "collision_policy": policy.value,
                "cell_counts": self.cell_counts,
            }
        )


class BoundaryLayerMesh(StrictModule, NonTrainableState):
    """Certified layer cells, the exact cap surface, and their evidence.

    ``cap`` is the outer front: a surface mesh whose vertex ``i`` is layer-mesh
    vertex ``cap_vertices[i]`` with bitwise-identical coordinates, oriented away
    from the layers, or ``None`` when merged fronts leave no free front. A core
    fill must keep it fixed. ``wall_vertices`` is exact row ancestry from the
    input wall; ``source_wall`` and ``wall_association`` retain its authoritative
    source binding. ``layer_index`` and ``column_index`` identify every output
    cell's schedule interval and front-row column in global cell-ID order.
    """

    mesh: CellMesh
    cap: CellMesh | None
    wall_vertices: Array
    cap_vertices: Array
    layer_index: Array
    column_index: Array
    column_vertex_sources: Array
    column_vertex_ids: Array
    column_vertex_identity: tuple[str, ...] = eqx.field(static=True)
    column_source_geometry_entity_ids: tuple[tuple[str, ...], ...] = eqx.field(
        static=True
    )
    column_source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    column_source_face_offsets: Array
    column_source_face_ids: Array
    column_source_face_orientations: Array
    column_source_face_weights: Array
    column_source_face_roles: Array
    column_parent_offsets: Array
    column_parent_ids: Array
    column_parent_weights: Array
    column_construction_kinds: tuple[str, ...] = eqx.field(static=True)
    front_row_ids: Array
    front_row_identity: tuple[str, ...] = eqx.field(static=True)
    front_row_source_piece_indices: Array
    front_row_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...] = (
        eqx.field(static=True)
    )
    front_row_column_ids: Array
    front_row_source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    front_row_source_entity_kinds: tuple[str, ...] = eqx.field(static=True)
    front_row_source_global_ids: Array
    front_row_orientations: Array
    vertex_source_vertex_ids: Array
    vertex_source_entity_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vertex_column_parent_offsets: Array
    vertex_column_parent_ids: Array
    vertex_column_parent_weights: Array
    vertex_column_ids: Array
    vertex_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...] = eqx.field(
        static=True
    )
    source_wall: CellMesh
    wall_association: GeometryAssociation | None
    validity: CellValidityCertificate
    evidence: BoundaryLayerEvidence
    control_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        cap: CellMesh | None,
        wall_vertices: Any,
        cap_vertices: Any,
        layer_index: Any,
        validity: CellValidityCertificate,
        evidence: BoundaryLayerEvidence,
        /,
        *,
        source_wall: CellMesh,
        wall_association: GeometryAssociation | None,
        column_index: Any,
        column_vertex_sources: Any,
        column_vertex_ids: Any,
        column_vertex_identity: tuple[str, ...],
        column_source_geometry_entity_ids: tuple[tuple[str, ...], ...],
        column_source_occurrence_paths: tuple[tuple[str, ...], ...],
        column_source_face_offsets: Any,
        column_source_face_ids: Any,
        column_source_face_orientations: Any,
        column_source_face_weights: Any,
        column_source_face_roles: Any,
        column_parent_offsets: Any,
        column_parent_ids: Any,
        column_parent_weights: Any,
        column_construction_kinds: tuple[str, ...],
        front_row_ids: Any,
        front_row_identity: tuple[str, ...],
        front_row_source_piece_indices: Any,
        front_row_column_ids: Any,
        front_row_source_entity_ids: tuple[str, ...],
        front_row_source_entity_kinds: tuple[str, ...],
        front_row_source_global_ids: Any,
        front_row_orientations: Any,
        front_row_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
        vertex_source_vertex_ids: Any,
        vertex_source_entity_ids: tuple[tuple[str, ...], ...],
        vertex_column_ids: Any,
        vertex_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
        vertex_column_parent_offsets: Any,
        vertex_column_parent_ids: Any,
        vertex_column_parent_weights: Any,
        control_id: str,
        policy_id: str,
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(cap, (CellMesh, type(None))):
            raise TypeError("mesh must be CellMesh and cap CellMesh or None.")
        if source_wall.topological_dimension != 2 or source_wall.ambient_dimension != 3:
            raise ValueError("source_wall must be an ambient-3 surface mesh.")
        cell_count = sum(block.cell_count for block in mesh.blocks)
        cell_global_ids = np.concatenate(
            tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
        )
        if not np.array_equal(cell_global_ids, np.sort(cell_global_ids)):
            raise ValueError(
                "Boundary-layer cell banks must be stored in global-cell-ID order."
            )
        if wall_association is not None:
            if not isinstance(wall_association, GeometryAssociation):
                raise TypeError("wall_association must be GeometryAssociation or None.")
            vertex_set = source_wall.entity_set(0)
            if (
                wall_association.target_entity_set_id != vertex_set.entity_set_id
                or not np.array_equal(
                    np.sort(
                        np.asarray(wall_association.target_global_ids, dtype=np.int64)
                    ),
                    np.sort(np.asarray(vertex_set.entity_ids, dtype=np.int64)),
                )
            ):
                raise ValueError(
                    "wall_association must bind the exact source wall vertices."
                )
        raw_columns = np.asarray(column_index)
        if (
            raw_columns.shape != (cell_count,)
            or not np.issubdtype(raw_columns.dtype, np.integer)
            or np.any(raw_columns < 0)
        ):
            raise ValueError(
                "column_index must give every layer cell a nonnegative column ID."
            )
        columns = raw_columns.astype(np.int64, copy=False)
        source_vertex_ids = np.asarray(source_wall.vertex_global_ids, dtype=np.int64)
        column_sources = np.asarray(column_vertex_sources)
        column_ids = np.asarray(column_vertex_ids)
        if (
            column_sources.ndim != 1
            or not np.issubdtype(column_sources.dtype, np.integer)
            or column_ids.shape != column_sources.shape
            or not np.issubdtype(column_ids.dtype, np.integer)
            or np.any(column_ids < 0)
            or np.unique(column_ids).size != column_ids.size
            or np.any(~np.isin(column_sources, source_vertex_ids))
        ):
            raise ValueError(
                "Column roots and identities must name unique source-wall columns."
            )
        column_sources = column_sources.astype(np.int64, copy=False)
        column_ids = column_ids.astype(np.int64, copy=False)
        if not np.array_equal(np.sort(column_ids), np.arange(column_ids.size)):
            raise ValueError(
                "Column IDs must be canonical dense ranks of source identities."
            )
        column_keys = tuple(column_vertex_identity)
        column_geometry_ids = tuple(
            tuple(str(value) for value in row)
            for row in column_source_geometry_entity_ids
        )
        column_occurrences = tuple(tuple(path) for path in column_source_occurrence_paths)
        column_kinds = tuple(str(value) for value in column_construction_kinds)
        if (
            len(column_keys) != column_ids.size
            or len(set(column_keys)) != len(column_keys)
            or any(not value for value in column_keys)
            or column_keys != tuple(sorted(column_keys))
            or len(column_geometry_ids) != column_ids.size
            or len(column_occurrences) != column_ids.size
            or len(column_kinds) != column_ids.size
            or any(value not in ("sector", "fan", "corner") for value in column_kinds)
        ):
            raise ValueError(
                "Column source descriptors must align with the column ID bank."
            )
        face_offsets = np.asarray(column_source_face_offsets)
        face_ids = np.asarray(column_source_face_ids)
        face_orientations = np.asarray(column_source_face_orientations)
        face_weights = np.asarray(column_source_face_weights, dtype=np.float64)
        face_roles = np.asarray(column_source_face_roles)
        if (
            face_offsets.shape != (column_ids.size + 1,)
            or not np.issubdtype(face_offsets.dtype, np.integer)
            or face_offsets[0] != 0
            or face_offsets[-1] != face_ids.size
            or np.any(np.diff(face_offsets) < 0)
            or not np.issubdtype(face_ids.dtype, np.integer)
            or face_orientations.shape != face_ids.shape
            or not np.issubdtype(face_orientations.dtype, np.integer)
            or face_weights.shape != face_ids.shape
            or not np.all(np.isfinite(face_weights))
            or face_roles.shape != face_ids.shape
            or not np.issubdtype(face_roles.dtype, np.integer)
            or np.any(~np.isin(face_roles, (0, 1)))
            or np.any(~np.isin(face_orientations, (-1, 1)))
            or np.any(
                ~np.isin(face_ids, np.asarray(source_wall.entity_set(2).entity_ids))
            )
            or np.any(np.diff(face_offsets) == 0)
        ):
            raise ValueError("Column incident source-face CSR is invalid or incomplete.")
        parent_offsets = np.asarray(column_parent_offsets)
        parent_ids = np.asarray(column_parent_ids)
        parent_weights = np.asarray(column_parent_weights, dtype=np.float64)
        if (
            parent_offsets.shape != (column_ids.size + 1,)
            or not np.issubdtype(parent_offsets.dtype, np.integer)
            or parent_offsets[0] != 0
            or parent_offsets[-1] != parent_ids.size
            or np.any(np.diff(parent_offsets) < 0)
            or parent_weights.shape != parent_ids.shape
            or not np.all(np.isfinite(parent_weights))
            or np.any(~np.isin(parent_ids, column_ids))
        ):
            raise ValueError("Column parent ancestry CSR is invalid.")
        has_parents = np.diff(parent_offsets) > 0
        parent_sums = np.asarray(
            [
                np.sum(parent_weights[start:stop])
                for start, stop in zip(
                    parent_offsets[:-1], parent_offsets[1:], strict=True
                )
            ],
            dtype=np.float64,
        )
        if np.any(np.abs(parent_sums[has_parents] - 1.0) > 64.0 * np.finfo(float).eps):
            raise ValueError("Column parent interpolation weights must sum to one.")
        row_ids = np.asarray(front_row_ids)
        row_identity = tuple(front_row_identity)
        row_pieces = np.asarray(front_row_source_piece_indices)
        row_columns = np.asarray(front_row_column_ids)
        row_source_ids = tuple(str(value) for value in front_row_source_entity_ids)
        row_kinds = tuple(str(value) for value in front_row_source_entity_kinds)
        row_global_ids = np.asarray(front_row_source_global_ids)
        row_orientations = np.asarray(front_row_orientations)
        row_occurrences = tuple(
            tuple(tuple(path) for path in row)
            for row in front_row_source_occurrence_paths
        )
        row_count = row_ids.size
        if (
            row_ids.ndim != 1
            or not np.issubdtype(row_ids.dtype, np.integer)
            or np.any(row_ids < 0)
            or np.unique(row_ids).size != row_count
            or len(row_identity) != row_count
            or len(set(row_identity)) != row_count
            or any(not value for value in row_identity)
            or row_pieces.shape != (row_count,)
            or not np.issubdtype(row_pieces.dtype, np.integer)
            or np.any(row_pieces < 0)
            or row_columns.shape != (row_count, 4)
            or np.any(row_columns < -1)
            or np.any(~np.isin(row_columns[row_columns >= 0], column_ids))
            or np.any(np.sum(row_columns >= 0, axis=1) < 3)
            or len(row_source_ids) != row_count
            or any(not value for value in row_source_ids)
            or len(row_kinds) != row_count
            or any(value not in ("face", "edge", "vertex") for value in row_kinds)
            or row_global_ids.shape != (row_count,)
            or not np.issubdtype(row_global_ids.dtype, np.integer)
            or row_orientations.shape != (row_count,)
            or not np.issubdtype(row_orientations.dtype, np.integer)
            or np.any(~np.isin(row_orientations, (-1, 0, 1)))
            or len(row_occurrences) != row_count
        ):
            raise ValueError("Front-row ancestry must bind every oriented source row.")
        if not np.array_equal(np.sort(row_ids), np.arange(row_count)):
            raise ValueError(
                "Front-row IDs must be canonical dense ranks of source identities."
            )
        entity_ids_by_kind = {
            "face": np.asarray(source_wall.entity_set(2).entity_ids, dtype=np.int64),
            "edge": np.asarray(source_wall.entity_set(1).entity_ids, dtype=np.int64),
            "vertex": source_vertex_ids,
        }
        for kind, identifier, entity_key, orientation in zip(
            row_kinds, row_global_ids, row_source_ids, row_orientations, strict=True
        ):
            if int(identifier) not in entity_ids_by_kind[kind]:
                raise ValueError(
                    "Front-row source IDs must resolve in the exact source-wall entity set."
                )
            expected_key = source_entity_id(
                source_wall.mesh_id,
                str(source_wall.numeric_version),
                f"mesh-{kind}",
                int(identifier),
            )
            if entity_key != expected_key:
                raise ValueError(
                    "Front-row source entity IDs must bind the exact source-wall entity."
                )
            if (kind == "vertex") != (int(orientation) == 0):
                raise ValueError(
                    "Front-row orientations must be defined only for face/edge sources."
                )
        if any(
            any(not isinstance(part, str) or not part for part in path)
            for row in row_occurrences
            for path in row
        ):
            raise ValueError(
                "Front-row occurrence paths must be exact identifier tuples."
            )
        if np.any(~np.isin(columns, row_ids)):
            raise ValueError("Every layer-cell column ID must name a front-row source.")
        vertex_sources = np.asarray(vertex_source_vertex_ids)
        vertex_columns = np.asarray(vertex_column_ids)
        vertex_entities = tuple(
            tuple(str(value) for value in row) for row in vertex_source_entity_ids
        )
        vertex_occurrences = tuple(
            tuple(tuple(path) for path in paths)
            for paths in vertex_source_occurrence_paths
        )
        vertex_parent_offsets = np.asarray(vertex_column_parent_offsets)
        vertex_parent_ids = np.asarray(vertex_column_parent_ids)
        vertex_parent_weights = np.asarray(vertex_column_parent_weights, dtype=np.float64)
        vertex_count = mesh.coordinates.shape[0]
        if (
            vertex_sources.shape != (vertex_count,)
            or not np.issubdtype(vertex_sources.dtype, np.integer)
            or np.any((vertex_sources < 0) & (vertex_sources != -1))
            or np.any(~np.isin(vertex_sources[vertex_sources >= 0], source_vertex_ids))
            or vertex_columns.shape != (vertex_count,)
            or not np.issubdtype(vertex_columns.dtype, np.integer)
            or np.any((vertex_columns < 0) & (vertex_columns != -1))
            or np.any(~np.isin(vertex_columns[vertex_columns >= 0], column_ids))
            or len(vertex_entities) != vertex_count
            or len(vertex_occurrences) != vertex_count
            or vertex_parent_offsets.shape != (vertex_count + 1,)
            or not np.issubdtype(vertex_parent_offsets.dtype, np.integer)
            or vertex_parent_offsets[0] != 0
            or vertex_parent_offsets[-1] != vertex_parent_ids.size
            or np.any(np.diff(vertex_parent_offsets) < 0)
            or vertex_parent_weights.shape != vertex_parent_ids.shape
            or not np.all(np.isfinite(vertex_parent_weights))
            or np.any(~np.isin(vertex_parent_ids, column_ids))
        ):
            raise ValueError("Layer-vertex source and column ancestry is invalid.")
        if np.any((vertex_columns < 0) & (vertex_sources < 0)):
            raise ValueError(
                "Uncurved wall vertices must retain their source-wall identity."
            )
        wall_map = np.asarray(wall_vertices)
        if (
            wall_map.shape != (source_wall.coordinates.shape[0],)
            or not np.issubdtype(wall_map.dtype, np.integer)
            or np.any(wall_map < -1)
            or np.any(wall_map >= mesh.coordinates.shape[0])
            or np.unique(wall_map[wall_map >= 0]).size != np.count_nonzero(wall_map >= 0)
        ):
            raise ValueError(
                "wall_vertices must preserve unique source-wall vertex rows."
            )
        retained_rows = np.flatnonzero(wall_map >= 0)
        retained_vertices = wall_map[retained_rows].astype(np.int64, copy=False)
        if not np.array_equal(
            np.asarray(mesh.coordinates, dtype=np.float64)[retained_vertices].view(
                np.uint64
            ),
            np.asarray(source_wall.coordinates, dtype=np.float64)[retained_rows].view(
                np.uint64
            ),
        ):
            raise ValueError(
                "Retained wall vertices must preserve exact source coordinates."
            )
        if (
            not np.array_equal(
                vertex_sources[retained_vertices], source_vertex_ids[retained_rows]
            )
            or np.any(vertex_columns[retained_vertices] != -1)
            or np.any(np.diff(vertex_parent_offsets)[retained_vertices] != 0)
        ):
            raise ValueError(
                "Retained wall vertices must preserve identity without a layer column."
            )
        retained_mask = np.zeros((vertex_count,), dtype=np.bool_)
        retained_mask[retained_vertices] = True
        if not np.array_equal(vertex_columns == -1, retained_mask):
            raise ValueError(
                "Only unchanged retained wall vertices may omit a column ID."
            )
        vertex_parent_counts = np.diff(vertex_parent_offsets)
        if np.any((vertex_columns >= 0) & (vertex_parent_counts == 0)):
            raise ValueError("Every generated layer vertex must name its source columns.")
        vertex_parent_sums = np.asarray(
            [
                np.sum(vertex_parent_weights[start:stop])
                for start, stop in zip(
                    vertex_parent_offsets[:-1], vertex_parent_offsets[1:], strict=True
                )
            ],
            dtype=np.float64,
        )
        if np.any(
            np.abs(vertex_parent_sums[vertex_parent_counts > 0] - 1.0)
            > 64.0 * np.finfo(float).eps
        ):
            raise ValueError("Layer-vertex source-column weights must sum to one.")
        cap_ids = np.asarray(cap_vertices)
        if (
            cap_ids.ndim != 1
            or not np.issubdtype(cap_ids.dtype, np.integer)
            or np.any(cap_ids < 0)
            or np.any(cap_ids >= mesh.coordinates.shape[0])
        ):
            raise ValueError("cap_vertices must identify layer-mesh vertex rows.")
        cap_ids = cap_ids.astype(np.int64, copy=False)
        layers = np.asarray(layer_index, dtype=np.int32)
        if layers.shape != (cell_count,):
            raise ValueError("layer_index must hold one entry per layer cell.")
        cap_points = np.empty((0, 3)) if cap is None else np.asarray(cap.coordinates)
        if cap_ids.shape != (cap_points.shape[0],) or not np.array_equal(
            np.asarray(mesh.coordinates)[cap_ids], cap_points
        ):
            raise ValueError("Cap vertices must be bitwise layer-mesh vertices.")
        if validity.certified_valid_count != cell_count:
            raise ValueError("Every boundary-layer cell must be certified valid.")
        self.mesh = mesh
        self.cap = cap
        self.wall_vertices = jnp.asarray(wall_map, dtype=jnp.int64)
        self.cap_vertices = jnp.asarray(cap_ids)
        self.layer_index = jnp.asarray(layers)
        self.column_vertex_sources = jnp.asarray(column_sources, dtype=jnp.int64)
        self.column_vertex_ids = jnp.asarray(column_ids, dtype=jnp.int64)
        self.column_vertex_identity = column_keys
        self.column_source_geometry_entity_ids = column_geometry_ids
        self.column_source_occurrence_paths = column_occurrences
        self.column_source_face_offsets = jnp.asarray(face_offsets, dtype=jnp.int64)
        self.column_source_face_ids = jnp.asarray(face_ids, dtype=jnp.int64)
        self.column_source_face_orientations = jnp.asarray(
            face_orientations, dtype=jnp.int8
        )
        self.column_source_face_weights = jnp.asarray(face_weights, dtype=jnp.float64)
        self.column_source_face_roles = jnp.asarray(face_roles, dtype=jnp.int8)
        self.column_parent_offsets = jnp.asarray(parent_offsets, dtype=jnp.int64)
        self.column_parent_ids = jnp.asarray(parent_ids, dtype=jnp.int64)
        self.column_parent_weights = jnp.asarray(parent_weights, dtype=jnp.float64)
        self.column_construction_kinds = column_kinds
        self.front_row_ids = jnp.asarray(row_ids, dtype=jnp.int64)
        self.front_row_identity = row_identity
        self.front_row_source_piece_indices = jnp.asarray(row_pieces, dtype=jnp.int64)
        self.front_row_column_ids = jnp.asarray(row_columns, dtype=jnp.int64)
        self.front_row_source_entity_ids = row_source_ids
        self.front_row_source_entity_kinds = row_kinds
        self.front_row_source_global_ids = jnp.asarray(row_global_ids, dtype=jnp.int64)
        self.front_row_orientations = jnp.asarray(row_orientations, dtype=jnp.int8)
        self.front_row_source_occurrence_paths = row_occurrences
        self.vertex_source_vertex_ids = jnp.asarray(vertex_sources, dtype=jnp.int64)
        self.vertex_source_entity_ids = vertex_entities
        self.vertex_column_ids = jnp.asarray(vertex_columns, dtype=jnp.int64)
        self.vertex_source_occurrence_paths = vertex_occurrences
        self.vertex_column_parent_offsets = jnp.asarray(
            vertex_parent_offsets, dtype=jnp.int64
        )
        self.vertex_column_parent_ids = jnp.asarray(vertex_parent_ids, dtype=jnp.int64)
        self.vertex_column_parent_weights = jnp.asarray(
            vertex_parent_weights, dtype=jnp.float64
        )
        self.column_index = jnp.asarray(columns)
        self.source_wall = source_wall
        self.wall_association = wall_association
        self.validity = validity
        self.evidence = evidence
        self.control_id = str(control_id)
        self.policy_id = str(policy_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-mesh",
                "mesh": mesh.mesh_id,
                "cap": None if cap is None else cap.mesh_id,
                "layers": array_tree_fingerprint(layers),
                "columns": array_tree_fingerprint(columns),
                "source_wall": source_wall.mesh_id,
                "source_wall_revision": source_wall.numeric_version,
                "wall_association": (
                    None if wall_association is None else wall_association.association_id
                ),
                "column_vertex_sources": array_tree_fingerprint(column_sources),
                "column_vertex_ids": array_tree_fingerprint(column_ids),
                "column_vertex_identity": column_keys,
                "column_source_geometry_entities": column_geometry_ids,
                "column_source_occurrences": column_occurrences,
                "column_source_faces": {
                    "offsets": array_tree_fingerprint(face_offsets),
                    "ids": array_tree_fingerprint(face_ids),
                    "orientations": array_tree_fingerprint(face_orientations),
                    "weights": array_tree_fingerprint(face_weights),
                    "roles": array_tree_fingerprint(face_roles),
                },
                "column_parents": {
                    "offsets": array_tree_fingerprint(parent_offsets),
                    "ids": array_tree_fingerprint(parent_ids),
                    "weights": array_tree_fingerprint(parent_weights),
                },
                "front_rows": {
                    "ids": array_tree_fingerprint(row_ids),
                    "identity": row_identity,
                    "source_pieces": array_tree_fingerprint(row_pieces),
                    "columns": array_tree_fingerprint(row_columns),
                    "source_entities": row_source_ids,
                    "source_kinds": row_kinds,
                    "source_global_ids": array_tree_fingerprint(row_global_ids),
                    "orientations": array_tree_fingerprint(row_orientations),
                    "occurrences": row_occurrences,
                },
                "vertex_sources": array_tree_fingerprint(vertex_sources),
                "vertex_source_entities": vertex_entities,
                "vertex_columns": array_tree_fingerprint(vertex_columns),
                "vertex_occurrences": vertex_occurrences,
                "vertex_column_parents": {
                    "offsets": array_tree_fingerprint(vertex_parent_offsets),
                    "ids": array_tree_fingerprint(vertex_parent_ids),
                    "weights": array_tree_fingerprint(vertex_parent_weights),
                },
                "evidence": evidence.evidence_id,
                "validity": validity.certificate_id,
                "control": self.control_id,
                "policy": self.policy_id,
            }
        )

    @property
    def closed_cap(self) -> bool:
        """Whether a cap exists and every cap edge is shared by exactly two faces."""
        if self.cap is None:
            return False
        faces, arity = _mesh_faces(self.cap)
        _, counts = _edge_table(faces, arity)[1:3]
        return bool(np.all(counts == 2))


# ---------------------------------------------------------------- failures


def _failure(
    category: MeshingFailureCategory,
    message: str,
    vertices: np.ndarray,
    points: np.ndarray,
    /,
) -> MeshingFailure:
    identifiers = np.unique(np.asarray(vertices, dtype=np.int64))
    return MeshingFailure(
        category,
        message,
        stage=_STAGE,
        entity_ids=tuple(int(value) for value in identifiers),
        locations=tuple(tuple(float(x) for x in points[value]) for value in identifiers),
    )


# ---------------------------------------------------------------- surface topology


def _mesh_faces(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    """Faces of a triangle/quadrilateral surface mesh in global-ID order."""
    rows = []
    identifiers = []
    for block in mesh.blocks:
        match block.cell_kind:
            case "triangle":
                vertices = np.asarray(block.vertices, dtype=np.int64)
                rows.append(np.pad(vertices, ((0, 0), (0, 1)), constant_values=-1))
            case "quadrilateral":
                rows.append(np.asarray(block.vertices, dtype=np.int64))
            case kind:
                raise ValueError(
                    f"Boundary-layer surfaces admit triangles and quadrilaterals, not {kind!r}."
                )
        identifiers.append(np.asarray(block.global_ids, dtype=np.int64))
    faces = np.concatenate(rows)
    order = np.argsort(np.concatenate(identifiers), kind="stable")
    faces = faces[order]
    return faces, np.where(faces[:, 3] < 0, 3, 4).astype(np.int64)


def _half_edges(faces: np.ndarray, arity: np.ndarray, /) -> Any:
    local = np.arange(4)
    valid = local[None, :] < arity[:, None]
    following = np.where(local[None, :] + 1 < arity[:, None], local[None, :] + 1, 0)
    start = faces[valid]
    end = np.take_along_axis(faces, following, axis=1)[valid]
    owner = np.nonzero(valid)[0]
    return start, end, owner


def _edge_table(faces: np.ndarray, arity: np.ndarray, /) -> Any:
    start, end, owner = _half_edges(faces, arity)
    keys = np.sort(np.stack((start, end), axis=1), axis=1)
    edges, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    return (start, end, owner), edges, counts, inverse.reshape(-1)


def _canonical_quads(quads: np.ndarray, /) -> np.ndarray:
    """Rotate quads so their smallest vertex leads; splits then agree across cells."""
    shift = np.argmin(quads, axis=1)
    index = (shift[:, None] + np.arange(4)[None, :]) % 4
    return np.take_along_axis(quads, index, axis=1)


def _split_polygons(
    rows: np.ndarray, arity: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate triangles/quads along the diagonal through the smallest vertex."""
    triangles = rows[arity == 3, :3]
    quads = _canonical_quads(rows[arity == 4])
    owners = np.concatenate(
        (
            np.flatnonzero(arity == 3),
            np.repeat(np.flatnonzero(arity == 4), 2),
        )
    )
    split = np.stack((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]), axis=1).reshape(-1, 3)
    return np.concatenate((triangles, split)), owners


def _unit(vectors: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=-1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return vectors / safe[..., None], norms


def _newell_normals(points: np.ndarray, faces: np.ndarray, arity: np.ndarray, /) -> Any:
    local = np.arange(4)
    following = np.where(local[None, :] + 1 < arity[:, None], local[None, :] + 1, 0)
    valid = local[None, :] < arity[:, None]
    first = points[np.where(valid, faces, faces[:, :1])]
    second = points[
        np.take_along_axis(np.where(valid, faces, faces[:, :1]), following, 1)
    ]
    normal = np.sum(np.where(valid[..., None], np.cross(first, second), 0.0), axis=1)
    return _unit(normal)


# ---------------------------------------------------------------- visibility optimum


# Duality-gap tolerance of the certified visibility optimum (unit normals).
_VISIBILITY_GAP = 1.0e-9


def _minimum_norm_point(points: np.ndarray, /) -> tuple[list[int], np.ndarray]:
    """Wolfe's minimum-norm point of ``conv(points)``: support rows and weights.

    Each major cycle adds the point most opposed to the current iterate; each
    minor cycle moves to the affine minimum-norm point of the corral, or to the
    last convex point on the way and drops the points whose weight vanishes.
    Corrals stay affinely independent, so at most four points are active in
    R^3 and every cycle is a small dense solve. Termination is finite; the
    iteration bound only guards rounding, and the caller certifies the result.
    """
    support = [0]
    weights = np.ones((1,))
    for _ in range(8 * points.shape[0] + 8):
        current = weights @ points[support]
        squared = float(current @ current)
        if squared <= 1.0e-24:
            break
        products = points @ current
        candidate = int(np.argmin(products))
        if products[candidate] >= squared - 1.0e-15 or candidate in support:
            break
        support.append(candidate)
        weights = np.append(weights, 0.0)
        while True:
            corral = points[support]
            size = len(support)
            # Affine minimum-norm point corral[0] + edges^T c by least squares on
            # the edges themselves (not their Gram), which keeps clustered
            # normals well conditioned.
            edges = corral[1:] - corral[0]
            offsets = np.linalg.lstsq(edges.T, -corral[0], rcond=None)[0]
            affine = np.concatenate(((1.0 - np.sum(offsets),), offsets))
            if np.all(affine > 0.0):
                weights = affine
                break
            blocked = (affine <= 0.0) & (weights > affine)
            ratios = np.full((size,), np.inf)
            ratios[blocked] = weights[blocked] / (weights[blocked] - affine[blocked])
            # A non-positive affine weight at a zero convex weight blocks at once.
            ratios[(affine <= 0.0) & ~blocked] = 0.0
            exit_row = int(np.argmin(ratios))
            weights = weights + min(float(ratios[exit_row]), 1.0) * (affine - weights)
            keep = weights > 0.0
            keep[exit_row] = False
            if not np.any(keep):
                keep[int(np.argmax(weights))] = True
            support = [row for row, kept in zip(support, keep, strict=True) if kept]
            weights = weights[keep] / np.sum(weights[keep])
            if len(support) <= 1:
                break
    return support, weights


def _visibility_optimum(
    normals: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Certified visibility-optimal direction of each column's padded unit normals.

    ``max_{|d|=1} min_i n_i.d`` equals the norm of the minimum-norm point ``p``
    of ``conv{n_i}`` and is attained at ``d = p / |p|`` whenever the origin lies
    outside the hull. Wolfe's algorithm finds ``p`` over each column's distinct
    normals, at a cost linear in their number per cycle. ``|p|`` bounds the
    optimum from above, and every tried direction ``d`` (``p / |p|`` and the
    normals themselves) bounds it from below by ``min_i n_i.d``. A column is
    certified when this duality gap closes to ``_VISIBILITY_GAP``. A
    non-positive lower bound with a vanishing ``p`` proves the origin lies in
    the hull: no visible direction exists.

    Returns per-slot convex weights, the best direction's visibility, and the
    duality gap of each column.
    """
    count, width, _ = normals.shape
    weights = np.zeros((count, width))
    lower = np.empty((count,))
    upper = np.empty((count,))
    for column in range(count):
        distinct, slots = np.unique(normals[column], axis=0, return_index=True)
        support, convex = _minimum_norm_point(distinct)
        point = convex @ distinct[support]
        norm = float(np.linalg.norm(point))
        directions = distinct if norm == 0.0 else np.vstack((point / norm, distinct))
        lower[column] = float(np.max(np.min(directions @ distinct.T, axis=1)))
        upper[column] = norm
        weights[column, slots[support]] = convex
    return weights, lower, np.maximum(upper - lower, 0.0)


def _padded_groups(groups: np.ndarray, members: np.ndarray, count: int, /) -> Any:
    """Pad per-group member lists (grouped rows) by repeating each group's first member."""
    order = np.argsort(groups, kind="stable")
    groups_ = groups[order]
    members_ = members[order]
    sizes = np.bincount(groups_, minlength=count)
    width = max(int(sizes.max(initial=1)), 1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    position = np.arange(groups_.size) - starts[groups_]
    # Empty groups (never read) borrow the last member to keep the table rectangular.
    lead = members_[np.minimum(starts, max(members_.size - 1, 0))]
    padded = np.repeat(lead[:, None], width, axis=1)
    padded[groups_, position] = members_
    return padded, sizes


def _polygon_centroids(points: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    valid = faces >= 0
    total = np.sum(np.where(valid[..., None], points[np.maximum(faces, 0)], 0.0), axis=1)
    return total / np.sum(valid, axis=1)[:, None]


# ---------------------------------------------------------------- wall analysis


@dataclass(frozen=True, slots=True)
class _Wall:
    points: np.ndarray
    faces: np.ndarray
    arity: np.ndarray
    grow: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    edge_faces: np.ndarray
    interior: np.ndarray
    rim: np.ndarray
    attached: np.ndarray
    theta: np.ndarray
    bend: np.ndarray
    convex: np.ndarray
    concave: np.ndarray
    wall_vertices: np.ndarray
    rim_vertex: np.ndarray


def _analyze_wall(
    points: np.ndarray,
    faces: np.ndarray,
    arity: np.ndarray,
    grow: np.ndarray,
    feature_angle: float,
    /,
) -> _Wall:
    normals, areas = _newell_normals(points, faces, arity)
    if np.any(areas[grow] <= 0.0):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Boundary-layer walls contain degenerate faces.",
            faces[grow & (areas <= 0.0)][:, :3].reshape(-1),
            points,
        )
    (start, _, owner), edges, counts, inverse = _edge_table(faces, arity)
    if np.any(counts > 2):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Boundary-layer surfaces must be edge-manifold.",
            edges[counts > 2].reshape(-1),
            points,
        )
    order = np.lexsort((owner, inverse))
    sorted_edges = inverse[order]
    first = np.concatenate(([True], sorted_edges[1:] != sorted_edges[:-1]))
    slot = np.where(first, 0, 1)
    edge_faces = np.full((edges.shape[0], 2), -1, dtype=np.int64)
    edge_starts = np.full((edges.shape[0], 2), -1, dtype=np.int64)
    edge_faces[sorted_edges, slot] = owner[order]
    edge_starts[sorted_edges, slot] = start[order]
    present = edge_faces >= 0
    grown = present & grow[np.maximum(edge_faces, 0)]
    grow_count = np.sum(grown, axis=1)
    interior = grow_count == 2
    rim = grow_count == 1
    attached = rim & np.all(present, axis=1)
    inconsistent = interior & (edge_starts[:, 0] == edge_starts[:, 1])
    if np.any(inconsistent):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Boundary-layer wall faces must be consistently oriented.",
            edges[inconsistent].reshape(-1),
            points,
        )
    theta = np.zeros((edges.shape[0],), dtype=np.float64)
    bend = np.zeros((edges.shape[0],), dtype=np.float64)
    f, g = edge_faces[interior].T
    centroids = _polygon_centroids(points, faces)
    theta[interior] = np.arccos(
        np.clip(np.sum(normals[f] * normals[g], axis=1), -1.0, 1.0)
    )
    # Positive bend: the neighbor rises toward the growth side (fronts converge).
    bend[interior] = np.sum(normals[f] * (centroids[g] - centroids[f]), axis=1) + np.sum(
        normals[g] * (centroids[f] - centroids[g]), axis=1
    )
    feature = interior & (theta > feature_angle)
    wall_vertices = np.unique(faces[grow][faces[grow] >= 0])
    rim_vertex = np.zeros((points.shape[0],), dtype=np.bool_)
    rim_vertex[edges[rim].reshape(-1)] = True
    return _Wall(
        points,
        faces,
        arity,
        grow,
        normals,
        edges,
        edge_faces,
        interior,
        rim,
        attached,
        theta,
        bend,
        feature & (bend < 0.0),
        feature & (bend >= 0.0),
        wall_vertices,
        rim_vertex,
    )


def _split_ridges(wall: _Wall, corner: BoundaryLayerCornerPolicy, /) -> np.ndarray:
    feature = wall.convex | wall.concave
    match corner:
        case BoundaryLayerCornerPolicy.REJECT:
            if np.any(feature):
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "The REJECT corner policy refuses wall feature edges beyond the feature angle.",
                    wall.edges[feature].reshape(-1),
                    wall.points,
                )
            return np.zeros_like(feature)
        case BoundaryLayerCornerPolicy.SMOOTH:
            return np.zeros_like(feature)
        case BoundaryLayerCornerPolicy.FAN:
            split = wall.convex.copy()
            closed = ~wall.rim_vertex
            # Dangling ridge ends inside closed fans cannot separate sectors.
            for _ in range(int(np.count_nonzero(split)) + 1):
                degree = np.bincount(
                    wall.edges[split].reshape(-1), minlength=wall.points.shape[0]
                )
                dangling = (degree == 1) & closed
                remove = split & np.any(dangling[wall.edges], axis=1)
                if not np.any(remove):
                    break
                split &= ~remove
            return split
        case _:
            raise TypeError("corner must be BoundaryLayerCornerPolicy.")


@dataclass(frozen=True, slots=True)
class _Sectors:
    incidence_keys: np.ndarray
    incidence_sector: np.ndarray
    sector_vertex: np.ndarray
    face_count: int

    def at(self, vertices: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
        """Sector of each (vertex, wall face) incidence; absent pairs return garbage."""
        keys = vertices * self.face_count + faces
        position = np.searchsorted(self.incidence_keys, keys)
        return self.incidence_sector[np.minimum(position, self.incidence_keys.size - 1)]


def _sectors(wall: _Wall, split: np.ndarray, /) -> _Sectors:
    local = np.arange(4)
    valid = (local[None, :] < wall.arity[:, None]) & wall.grow[:, None]
    vertices = wall.faces[valid]
    owners = np.nonzero(valid)[0]
    face_count = wall.faces.shape[0]
    keys = vertices * face_count + owners
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    vertices = vertices[order]
    joined = wall.interior & ~split
    first_face, second_face = wall.edge_faces[joined].T
    rows = []
    columns = []
    for endpoint in wall.edges[joined].T:
        rows.append(np.searchsorted(keys, endpoint * face_count + first_face))
        columns.append(np.searchsorted(keys, endpoint * face_count + second_face))
    rows_ = np.concatenate(rows)
    columns_ = np.concatenate(columns)
    graph = coo_matrix(
        (np.ones(rows_.size, dtype=np.int8), (rows_, columns_)),
        shape=(keys.size, keys.size),
    )
    _, labels = connected_components(graph, directed=False)
    # Canonical sector order: first incidence in (vertex, face) order.
    _, first, relabel = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty_like(first)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    sector = rank[relabel.reshape(-1)]
    sector_vertex = np.empty((first.size,), dtype=np.int64)
    sector_vertex[sector] = vertices
    return _Sectors(keys, sector, sector_vertex, face_count)


# ---------------------------------------------------------------- column directions


@dataclass(frozen=True, slots=True)
class _Directions:
    direction: np.ndarray
    visibility: np.ndarray
    normals: np.ndarray
    source_face_rows: np.ndarray
    source_face_counts: np.ndarray
    source_face_weights: np.ndarray
    constraint_face_offsets: np.ndarray
    constraint_face_rows: np.ndarray
    constraint_face_orientations: np.ndarray


@dataclass(frozen=True, slots=True)
class _Constraints:
    """Distinct adjacent-surface normals of attached rim sector columns (0-2 each)."""

    count: np.ndarray
    first: np.ndarray
    second: np.ndarray


def _constraint_normals(wall: _Wall, sectors: _Sectors, count: int, /) -> _Constraints:
    """Adjacent non-wall face normals constraining attached rim sector columns."""
    faces = wall.edge_faces[wall.attached]
    grow_first = wall.grow[faces[:, 0]]
    wall_face = np.where(grow_first, faces[:, 0], faces[:, 1])
    side_normal = wall.normals[np.where(grow_first, faces[:, 1], faces[:, 0])]
    ends = wall.edges[wall.attached]
    columns = np.concatenate([sectors.at(ends[:, side], wall_face) for side in (0, 1)])
    normals = np.concatenate((side_normal, side_normal))
    # Planes are sign-free: canonicalize the leading nonzero component positive.
    lead = np.argmax(np.abs(normals) > 1.0e-12, axis=1)
    sign = np.sign(normals[np.arange(normals.shape[0]), lead])
    canonical = normals * sign[:, None]
    keys = np.concatenate((columns[:, None], np.round(canonical * 1.0e9)), axis=1)
    _, unique = np.unique(keys, axis=0, return_index=True)
    columns, canonical = columns[unique], canonical[unique]
    order = np.argsort(columns, kind="stable")
    columns, canonical = columns[order], canonical[order]
    counts = np.bincount(columns, minlength=count)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    first = np.zeros((count, 3))
    second = np.zeros((count, 3))
    rank = np.arange(columns.size) - starts[columns]
    first[columns[rank == 0]] = canonical[rank == 0]
    second[columns[rank == 1]] = canonical[rank == 1]
    return _Constraints(counts, first, second)


def _constraint_face_ancestry(
    wall: _Wall, sectors: _Sectors, count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR of attached source faces and their canonical plane orientations."""
    faces = wall.edge_faces[wall.attached]
    if not faces.size:
        return (
            np.zeros((count + 1,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int8),
        )
    grow_first = wall.grow[faces[:, 0]]
    wall_face = np.where(grow_first, faces[:, 0], faces[:, 1])
    side_face = np.where(grow_first, faces[:, 1], faces[:, 0])
    ends = wall.edges[wall.attached]
    columns = np.concatenate([sectors.at(ends[:, side], wall_face) for side in (0, 1)])
    source_faces = np.tile(side_face, 2)
    normals = wall.normals[source_faces]
    lead = np.argmax(np.abs(normals) > 0.0, axis=1)
    orientation = np.sign(normals[np.arange(normals.shape[0]), lead]).astype(np.int8)
    pairs = np.stack((columns, source_faces), axis=1)
    _, unique = np.unique(pairs, axis=0, return_index=True)
    pairs = pairs[unique]
    orientation = orientation[unique]
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    orientation = orientation[order]
    counts = np.bincount(pairs[:, 0], minlength=count)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return offsets, pairs[:, 1].astype(np.int64), orientation


def _apply_constraints(direction: np.ndarray, constraints: _Constraints, /) -> np.ndarray:
    """Project columns onto one adjacent plane or the line shared by two."""
    plane = (
        direction
        - np.sum(direction * constraints.first, axis=1)[:, None] * constraints.first
    )
    line = np.cross(constraints.first, constraints.second)
    line = np.sign(np.sum(line * direction, axis=1))[:, None] * line
    value = np.where(
        (constraints.count == 0)[:, None],
        direction,
        np.where((constraints.count == 1)[:, None], plane, line),
    )
    value = np.where((constraints.count > 2)[:, None], 0.0, value)
    result, _ = _unit(value)
    return result


def _sector_directions(
    wall: _Wall,
    sectors: _Sectors,
    control: BoundaryLayerControl,
    /,
) -> _Directions:
    count = sectors.sector_vertex.size
    faces = sectors.incidence_keys % sectors.face_count
    padded, face_counts = _padded_groups(sectors.incidence_sector, faces, count)
    normals = wall.normals[padded]
    weights, optimum, gap = _visibility_optimum(normals)
    hidden = ~(optimum > 0.0)
    if np.any(hidden):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "No visible column direction exists: the incident face normals of "
            f"{int(np.count_nonzero(hidden))} column(s) enclose the origin "
            f"(best visibility {float(np.min(optimum)):.6g}).",
            sectors.sector_vertex[hidden],
            wall.points,
        )
    uncertified = gap > _VISIBILITY_GAP
    if np.any(uncertified):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "The visibility optimum of "
            f"{int(np.count_nonzero(uncertified))} column(s) is not certified "
            f"(largest duality gap {float(np.max(gap)):.3g}).",
            sectors.sector_vertex[uncertified],
            wall.points,
        )
    hull = np.sum(weights[..., None] * normals, axis=1)
    direction, _ = _unit(hull)
    constraints = _constraint_normals(wall, sectors, count)
    if np.any(constraints.count > 2):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Attached rim columns meet more than two adjacent surfaces.",
            sectors.sector_vertex[constraints.count > 2],
            wall.points,
        )
    direction = _apply_constraints(direction, constraints)
    visibility = np.min(np.sum(normals * direction[:, None, :], axis=-1), axis=1)
    if np.any(visibility <= 0.0):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "No visible column direction exists at concave corners or folds "
            f"(minimum visibility {float(np.min(visibility)):.6g}).",
            sectors.sector_vertex[visibility <= 0.0],
            wall.points,
        )
    direction = _smooth_directions(
        wall, sectors, direction, visibility, normals, constraints, control
    )
    visibility = np.min(np.sum(normals * direction[:, None, :], axis=-1), axis=1)
    constraint_offsets, constraint_faces, constraint_orientations = (
        _constraint_face_ancestry(wall, sectors, count)
    )
    return _Directions(
        direction,
        visibility,
        normals,
        padded,
        face_counts,
        weights,
        constraint_offsets,
        constraint_faces,
        constraint_orientations,
    )


def _sector_graph(wall: _Wall, sectors: _Sectors, /) -> tuple[np.ndarray, np.ndarray]:
    joined = wall.interior | wall.rim
    faces = wall.edge_faces[joined]
    face = np.where(wall.grow[faces[:, 0]], faces[:, 0], faces[:, 1])
    first = sectors.at(wall.edges[joined, 0], face)
    second = sectors.at(wall.edges[joined, 1], face)
    return first, second


def _smooth_directions(
    wall: Any,
    sectors: Any,
    direction: Any,
    reference: Any,
    normals: Any,
    constraints: Any,
    control: Any,
) -> Any:
    """Weighted Laplacian smoothing that never loses more than 5% visibility."""
    first, second = _sector_graph(wall, sectors)
    lengths = np.linalg.norm(
        wall.points[sectors.sector_vertex[first]]
        - wall.points[sectors.sector_vertex[second]],
        axis=1,
    )
    weight = 1.0 / np.maximum(lengths, np.finfo(np.float64).tiny)
    count = direction.shape[0]
    total = np.bincount(first, weight, count) + np.bincount(second, weight, count)
    for _ in range(control.smoothing_iterations):
        accumulated = np.zeros_like(direction)
        np.add.at(accumulated, first, weight[:, None] * direction[second])
        np.add.at(accumulated, second, weight[:, None] * direction[first])
        average = np.where(
            total[:, None] > 0.0,
            accumulated / np.maximum(total, 1.0e-300)[:, None],
            direction,
        )
        candidate, _ = _unit(direction + 0.5 * (average - direction))
        candidate = _apply_constraints(candidate, constraints)
        visibility = np.min(np.sum(normals * candidate[:, None, :], axis=-1), axis=1)
        accept = visibility >= 0.95 * reference
        direction = np.where(accept[:, None], candidate, direction)
    return direction


def _slerp(first: np.ndarray, second: np.ndarray, fraction: np.ndarray, /) -> np.ndarray:
    cosine = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
    angle = np.arccos(cosine)
    small = angle < 1.0e-9
    sine = np.where(small, 1.0, np.sin(angle))
    left = np.where(small, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / sine)
    right = np.where(small, fraction, np.sin(fraction * angle) / sine)
    value, _ = _unit(left[..., None] * first + right[..., None] * second)
    return value


def _slerp_weights(
    first: np.ndarray, second: np.ndarray, fraction: float, /
) -> tuple[float, float]:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-9:
        left, right = 1.0 - fraction, fraction
    else:
        sine = np.sin(angle)
        left = np.sin((1.0 - fraction) * angle) / sine
        right = np.sin(fraction * angle) / sine
    total = left + right
    return float(left / total), float(right / total)


def _source_edge_global_id_map(source_wall: CellMesh, /) -> dict[tuple[int, int], int]:
    """Map stable source-wall endpoint IDs to their authored edge IDs."""
    incidences = np.asarray(_incidence_pairs(source_wall, 0, 1), dtype=np.int64)
    edge_count = source_wall.entity_set(1).count
    counts = np.bincount(incidences[:, 1], minlength=edge_count)
    if counts.shape != (edge_count,) or np.any(counts != 2):
        raise ValueError("Source-wall edge ancestry requires two endpoint incidences.")
    order = np.lexsort((incidences[:, 0], incidences[:, 1]))
    endpoints = np.asarray(source_wall.vertex_global_ids, dtype=np.int64)[
        incidences[order, 0]
    ].reshape(edge_count, 2)
    identifiers = np.asarray(source_wall.entity_set(1).entity_ids, dtype=np.int64)
    result: dict[tuple[int, int], int] = {}
    for pair, identifier in zip(endpoints, identifiers, strict=True):
        key = (int(min(pair[0], pair[1])), int(max(pair[0], pair[1])))
        if key in result:
            raise ValueError(
                "Source-wall edge endpoint IDs must identify unique authored edges."
            )
        result[key] = int(identifier)
    return result


# ---------------------------------------------------------------- front complex


@dataclass(frozen=True, slots=True)
class _Front:
    column_vertex: np.ndarray
    column_direction: np.ndarray
    column_kind: np.ndarray
    faces: np.ndarray
    arity: np.ndarray
    kind: np.ndarray
    wall_face: np.ndarray
    fan_column_count: int
    corner_patch_count: int
    ancestry: _FrontAncestry


@dataclass(frozen=True, slots=True)
class _FrontAncestry:
    column_vertex_sources: np.ndarray
    column_vertex_ids: np.ndarray
    column_ids_by_front_vertex: np.ndarray
    column_vertex_identity: tuple[str, ...]
    column_source_geometry_entity_ids: tuple[tuple[str, ...], ...]
    column_source_occurrence_paths: tuple[tuple[str, ...], ...]
    column_source_face_offsets: np.ndarray
    column_source_face_ids: np.ndarray
    column_source_face_orientations: np.ndarray
    column_source_face_weights: np.ndarray
    column_source_face_roles: np.ndarray
    column_parent_offsets: np.ndarray
    column_parent_ids: np.ndarray
    column_parent_weights: np.ndarray
    column_construction_kinds: tuple[str, ...]
    front_row_ids: np.ndarray
    front_row_identity: tuple[str, ...]
    front_row_source_piece_indices: np.ndarray
    front_row_column_ids: np.ndarray
    front_row_source_entity_ids: tuple[str, ...]
    front_row_source_entity_kinds: tuple[str, ...]
    front_row_source_global_ids: np.ndarray
    front_row_orientations: np.ndarray
    front_row_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...]


def _fan_lists(
    wall: _Wall,
    sectors: _Sectors,
    split: np.ndarray,
    direction: np.ndarray,
    feature_angle: float,
    /,
) -> Any:
    """Ordered fan column lists of every split ridge at both endpoints."""
    ridges = np.flatnonzero(split)
    ridge_faces = wall.edge_faces[ridges]
    ends = wall.edges[ridges]
    sector_a = np.stack(
        [sectors.at(ends[:, side], ridge_faces[:, 0]) for side in (0, 1)], axis=1
    )
    sector_b = np.stack(
        [sectors.at(ends[:, side], ridge_faces[:, 1]) for side in (0, 1)], axis=1
    )
    if np.any(sector_a == sector_b):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A convex ridge does not separate wall sectors.",
            ends[np.any(sector_a == sector_b, axis=1)].reshape(-1),
            wall.points,
        )
    angle = np.arccos(
        np.clip(np.sum(direction[sector_a] * direction[sector_b], axis=-1), -1.0, 1.0)
    )
    if np.any(angle >= np.pi - 1.0e-6):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "A convex ridge folds its sector directions back onto each other.",
            ends[np.any(angle >= np.pi - 1.0e-6, axis=1)].reshape(-1),
            wall.points,
        )
    need = np.maximum(np.ceil(angle / feature_angle).astype(np.int64) - 1, 0)
    # Ridge chains continue through closed-fan vertices with exactly two ridges.
    degree = np.bincount(ends.reshape(-1), minlength=wall.points.shape[0])
    chain_vertex = (degree == 2) & ~wall.rim_vertex
    incidence_vertex = ends.reshape(-1)
    incidence_ridge = np.repeat(np.arange(ridges.size), 2)
    continuing = chain_vertex[incidence_vertex]
    order = np.argsort(incidence_vertex[continuing], kind="stable")
    paired = incidence_ridge[continuing][order].reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(paired.shape[0], dtype=np.int8), (paired[:, 0], paired[:, 1])),
        shape=(ridges.size, ridges.size),
    )
    _, chain = connected_components(graph, directed=False)
    chain_need = np.zeros((chain.max(initial=-1) + 1,), dtype=np.int64)
    np.maximum.at(chain_need, chain, np.max(need, axis=1))
    subdivisions = chain_need[chain]
    return ridges, ends, sector_a, sector_b, subdivisions


def _build_front(
    wall: _Wall,
    sectors: _Sectors,
    split: np.ndarray,
    directions: _Directions,
    control: BoundaryLayerControl,
    source_wall: CellMesh,
    wall_association: GeometryAssociation | None,
    /,
) -> _Front:
    sector_count = sectors.sector_vertex.size
    column_vertex = [sectors.sector_vertex]
    column_direction = [directions.direction]
    column_kind = [np.full((sector_count,), _SECTOR, dtype=np.int64)]
    parent_local: list[tuple[int, ...]] = [() for _ in range(sector_count)]
    parent_weights: list[tuple[float, ...]] = [() for _ in range(sector_count)]
    fan_fractions: list[tuple[int, int] | None] = [None for _ in range(sector_count)]
    next_column = sector_count
    ridges, ends, sector_a, sector_b, subdivisions = _fan_lists(
        wall, sectors, split, directions.direction, control.feature_angle
    )
    fan_groups: dict[tuple[int, int, int], np.ndarray] = {}
    lists: dict[tuple[int, int], list[int]] = {}
    for index in range(ridges.size):
        for side in (0, 1):
            vertex = int(ends[index, side])
            first = int(sector_a[index, side])
            second = int(sector_b[index, side])
            low, high = min(first, second), max(first, second)
            key = (vertex, low, high)
            if key not in fan_groups:
                count = int(subdivisions[index])
                fractions = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
                values = _slerp(
                    np.repeat(directions.direction[low][None], count, 0),
                    np.repeat(directions.direction[high][None], count, 0),
                    fractions,
                )
                fan_groups[key] = np.arange(next_column, next_column + count)
                column_vertex.append(np.full((count,), vertex, dtype=np.int64))
                column_direction.append(values)
                column_kind.append(np.full((count,), _FAN, dtype=np.int64))
                for position, fraction in enumerate(fractions):
                    parent_local.append((low, high))
                    parent_weights.append(
                        _slerp_weights(
                            directions.direction[low],
                            directions.direction[high],
                            float(fraction),
                        )
                    )
                    fan_fractions.append((position + 1, count + 1))
                next_column += count
            fans = fan_groups[key]
            ordered = fans if first == low else fans[::-1]
            lists[index, side] = [first, *(int(value) for value in ordered), second]
    column_direction_ = np.concatenate(column_direction)
    column_vertex_ = np.concatenate(column_vertex)
    strips = []
    strip_ridges = []
    strip_pieces = []
    for index in range(ridges.size):
        left = lists[index, 0]
        right = lists[index, 1]
        for position in range(len(left) - 1):
            strips.append(
                (left[position], left[position + 1], right[position + 1], right[position])
            )
            strip_ridges.append(int(ridges[index]))
            strip_pieces.append(position)
    strips_ = np.asarray(strips, dtype=np.int64).reshape(-1, 4)
    if strips_.size:
        # Orient strips like the wall faces: normal along the growth direction.
        tops = (
            wall.points[column_vertex_[strips_]] + column_direction_[strips_]
        ).reshape(-1, 3)
        normal, _ = _newell_normals(
            tops,
            np.arange(tops.shape[0]).reshape(-1, 4),
            np.full((strips_.shape[0],), 4),
        )
        reverse = (
            np.sum(normal * np.sum(column_direction_[strips_], axis=1), axis=1) < 0.0
        )
        strips_[reverse] = strips_[reverse][:, (1, 0, 3, 2)]
    corners = _corner_patches(wall, ends, sector_a, sector_b, lists)
    corner_rows = []
    corner_source_vertices = []
    corner_pieces = []
    extra_vertex = []
    extra_direction = []
    for vertex, cycle in corners:
        center = next_column
        next_column += 1
        value, _ = _unit(np.sum(column_direction_[cycle], axis=0))
        extra_vertex.append(vertex)
        extra_direction.append(value)
        parent_local.append(tuple(int(column) for column in cycle))
        parent_weights.append(tuple(1.0 / len(cycle) for _ in cycle))
        fan_fractions.append(None)
        ring = np.asarray(cycle, dtype=np.int64)
        rows = np.stack((np.full(ring.shape, center), ring, np.roll(ring, -1)), axis=1)
        first = column_direction_[ring]
        second = column_direction_[np.roll(ring, -1)]
        # Corner fans face away from the vertex: normals along the center direction.
        if np.sum(np.cross(first - value, second - value) @ value) < 0.0:
            rows = rows[:, (0, 2, 1)]
        corner_rows.append(rows)
        corner_source_vertices.extend((vertex,) * len(cycle))
        corner_pieces.extend(range(len(cycle)))
    if extra_vertex:
        column_vertex_ = np.concatenate(
            (column_vertex_, np.asarray(extra_vertex, dtype=np.int64))
        )
        column_direction_ = np.concatenate(
            (column_direction_, np.asarray(extra_direction))
        )
        column_kind.append(np.full((len(extra_vertex),), _CENTER, dtype=np.int64))
    wall_rows = np.flatnonzero(wall.grow)
    wall_faces = wall.faces[wall_rows]
    wall_columns = np.where(
        wall_faces >= 0,
        sectors.at(np.maximum(wall_faces, 0), np.repeat(wall_rows[:, None], 4, axis=1)),
        -1,
    )
    corner_faces = (
        np.concatenate(corner_rows) if corner_rows else np.empty((0, 3), dtype=np.int64)
    )
    faces = np.concatenate(
        (
            wall_columns,
            strips_,
            np.pad(corner_faces, ((0, 0), (0, 1)), constant_values=-1),
        )
    )
    arity = np.concatenate(
        (
            wall.arity[wall_rows],
            np.full((strips_.shape[0],), 4, dtype=np.int64),
            np.full((corner_faces.shape[0],), 3, dtype=np.int64),
        )
    )
    kind = np.concatenate(
        (
            np.full((wall_rows.size,), _WALL),
            np.full((strips_.shape[0],), _STRIP),
            np.full((corner_faces.shape[0],), _CORNER),
        )
    )
    wall_face = np.concatenate(
        (wall_rows, np.full((strips_.shape[0] + corner_faces.shape[0],), -1))
    )
    source_vertex_ids = np.asarray(source_wall.vertex_global_ids, dtype=np.int64)
    source_face_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source_wall.blocks
        )
    )
    source_face_ids = source_face_ids[np.argsort(source_face_ids, kind="stable")]
    edge_ids = _source_edge_global_id_map(source_wall)
    strip_global_edges = []
    strip_orientations = []
    for ridge in strip_ridges:
        endpoints = source_vertex_ids[wall.edges[ridge]]
        key = (int(min(endpoints[0], endpoints[1])), int(max(endpoints[0], endpoints[1])))
        strip_global_edges.append(edge_ids[key])
        strip_orientations.append(1 if endpoints[0] < endpoints[1] else -1)
    row_source_kinds = (
        ("face",) * wall_rows.size
        + ("edge",) * len(strip_ridges)
        + ("vertex",) * len(corner_source_vertices)
    )
    row_source_global_ids = np.concatenate(
        (
            source_face_ids[wall_rows],
            np.asarray(strip_global_edges, dtype=np.int64),
            source_vertex_ids[np.asarray(corner_source_vertices, dtype=np.int64)],
        )
    )
    row_orientations = np.concatenate(
        (
            np.ones((wall_rows.size,), dtype=np.int8),
            np.asarray(strip_orientations, dtype=np.int8),
            np.zeros((len(corner_source_vertices),), dtype=np.int8),
        )
    )
    row_pieces = np.concatenate(
        (
            np.zeros((wall_rows.size,), dtype=np.int64),
            np.asarray(strip_pieces, dtype=np.int64),
            np.asarray(corner_pieces, dtype=np.int64),
        )
    )
    column_kind_ = np.concatenate(column_kind)
    ancestry = _front_source_ancestry(
        source_wall,
        wall_association,
        directions,
        column_vertex_,
        column_kind_,
        tuple(parent_local),
        tuple(parent_weights),
        tuple(fan_fractions),
        faces,
        row_source_kinds,
        row_source_global_ids,
        row_orientations,
        row_pieces,
    )
    return _Front(
        column_vertex_,
        column_direction_,
        column_kind_,
        faces,
        arity,
        kind,
        wall_face,
        int(np.count_nonzero(column_kind_ == _FAN)),
        len(corners),
        ancestry,
    )


def _corner_patches(
    wall: Any, ends: Any, sector_a: Any, sector_b: Any, lists: Any, /
) -> Any:
    """Cycles of fan and sector columns around closed vertices with three or more ridges."""
    degree = np.bincount(ends.reshape(-1), minlength=wall.points.shape[0])
    corners = []
    for vertex in np.flatnonzero((degree >= 3) & ~wall.rim_vertex):
        incident = [
            (index, side) for index, side in zip(*np.nonzero(ends == vertex), strict=True)
        ]
        by_sector: dict[int, list[tuple[int, int]]] = {}
        for index, side in incident:
            by_sector.setdefault(int(sector_a[index, side]), []).append((index, side))
            by_sector.setdefault(int(sector_b[index, side]), []).append((index, side))
        if any(len(values) != 2 for values in by_sector.values()):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A wall corner does not form one closed cycle of ridges and sectors.",
                np.asarray([vertex]),
                wall.points,
            )
        start = incident[0]
        current = start
        sector = int(sector_a[start])
        cycle: list[int] = []
        for _ in range(len(incident)):
            fan = lists[current]
            ordered = fan if fan[0] == sector else fan[::-1]
            cycle.extend(ordered[:-1])
            sector = ordered[-1]
            following = [value for value in by_sector[sector] if value != current]
            current = following[0]
        if current != start or len(cycle) < 3:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A wall corner does not form one closed cycle of ridges and sectors.",
                np.asarray([vertex]),
                wall.points,
            )
        corners.append((int(vertex), cycle))
    return corners


def _front_source_ancestry(
    source_wall: CellMesh,
    wall_association: GeometryAssociation | None,
    directions: _Directions,
    column_vertex: np.ndarray,
    column_kind: np.ndarray,
    parent_local: tuple[tuple[int, ...], ...],
    parent_weights: tuple[tuple[float, ...], ...],
    fan_fractions: tuple[tuple[int, int] | None, ...],
    front_faces: np.ndarray,
    row_source_kinds: tuple[str, ...],
    row_source_global_ids: np.ndarray,
    row_orientations: np.ndarray,
    row_pieces: np.ndarray,
    /,
) -> _FrontAncestry:
    """Bind each front/column to exact source topology, not output coordinates."""
    source_vertex_ids = np.asarray(source_wall.vertex_global_ids, dtype=np.int64)
    source_face_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source_wall.blocks
        )
    )
    source_face_ids = source_face_ids[np.argsort(source_face_ids, kind="stable")]
    column_count = column_vertex.size
    root_ids = source_vertex_ids[column_vertex]
    face_records: list[list[tuple[int, int, int, float]]] = []
    for column in range(column_count):
        records: dict[tuple[int, int, int], float] = {}
        if column_kind[column] == _SECTOR:
            for slot in range(int(directions.source_face_counts[column])):
                face_row = int(directions.source_face_rows[column, slot])
                face_id = int(source_face_ids[face_row])
                key = (face_id, 1, 0)
                records[key] = records.get(key, 0.0) + float(
                    directions.source_face_weights[column, slot]
                )
            start = int(directions.constraint_face_offsets[column])
            stop = int(directions.constraint_face_offsets[column + 1])
            for index in range(start, stop):
                face_row = int(directions.constraint_face_rows[index])
                key = (
                    int(source_face_ids[face_row]),
                    int(directions.constraint_face_orientations[index]),
                    1,
                )
                records.setdefault(key, 0.0)
        else:
            for parent, fraction in zip(
                parent_local[column], parent_weights[column], strict=True
            ):
                for face_id, orientation, role, weight in face_records[parent]:
                    key = (face_id, orientation, role)
                    records[key] = records.get(key, 0.0) + float(fraction) * weight
        if not records:
            raise ValueError(
                "Every generated front column must retain incident source faces."
            )
        face_records.append([(*key, weight) for key, weight in sorted(records.items())])

    association_rows: dict[int, int] = {}
    if wall_association is not None:
        association_rows = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(wall_association.target_global_ids, dtype=np.int64)
            )
        }
    column_geometry_entities: list[tuple[str, ...]] = []
    column_occurrences: list[tuple[str, ...]] = []
    column_identities: list[str] = []
    for column in range(column_count):
        association_row = association_rows.get(int(root_ids[column]))
        if association_row is None or wall_association is None:
            geometry_entities: tuple[str, ...] = ()
            occurrence: tuple[str, ...] = ()
        else:
            geometry_entities = (wall_association.source_entity_ids[association_row],)
            occurrence = wall_association.source_occurrence_paths[association_row]
        column_geometry_entities.append(geometry_entities)
        column_occurrences.append(occurrence)
        source_vertex = source_entity_id(
            source_wall.mesh_id,
            str(source_wall.numeric_version),
            "mesh-vertex",
            int(root_ids[column]),
            occurrence_path=occurrence,
        )
        incident_faces = tuple(sorted({record[0] for record in face_records[column]}))
        parents = parent_local[column]
        if column_kind[column] == _SECTOR:
            recipe = {"kind": "sector", "incident_faces": incident_faces}
        elif column_kind[column] == _FAN:
            fan_fraction = fan_fractions[column]
            if fan_fraction is None:
                raise ValueError(
                    "Boundary-layer fan ancestry lost its exact rational fraction."
                )
            numerator, denominator = fan_fraction
            first, second = parents
            if column_identities[first] <= column_identities[second]:
                fraction = (numerator, denominator)
                ordered_parents = (column_identities[first], column_identities[second])
            else:
                fraction = (denominator - numerator, denominator)
                ordered_parents = (column_identities[second], column_identities[first])
            recipe = {
                "kind": "fan",
                "parents": ordered_parents,
                "fraction": fraction,
            }
        else:
            recipe = {
                "kind": "corner",
                "parents": tuple(sorted(column_identities[parent] for parent in parents)),
            }
        column_identities.append(
            canonical_fingerprint(
                {
                    "kind": "boundary-layer-column",
                    "source_vertex": source_vertex,
                    "geometry_entities": geometry_entities,
                    "recipe": recipe,
                }
            )
        )

    column_order = np.argsort(np.asarray(column_identities), kind="stable")
    column_ids = np.empty((column_count,), dtype=np.int64)
    column_ids[column_order] = np.arange(column_count, dtype=np.int64)
    face_offsets = [0]
    flat_face_ids: list[int] = []
    flat_face_orientations: list[int] = []
    flat_face_weights: list[float] = []
    flat_face_roles: list[int] = []
    parent_offsets = [0]
    flat_parent_ids: list[int] = []
    flat_parent_weights: list[float] = []
    for column in column_order:
        for face_id, orientation, role, weight in face_records[int(column)]:
            flat_face_ids.append(face_id)
            flat_face_orientations.append(orientation)
            flat_face_weights.append(weight)
            flat_face_roles.append(role)
        face_offsets.append(len(flat_face_ids))
        parent_pairs = sorted(
            (
                int(column_ids[parent]),
                float(weight),
            )
            for parent, weight in zip(
                parent_local[int(column)], parent_weights[int(column)], strict=True
            )
        )
        flat_parent_ids.extend(parent for parent, _ in parent_pairs)
        flat_parent_weights.extend(weight for _, weight in parent_pairs)
        parent_offsets.append(len(flat_parent_ids))

    row_count = front_faces.shape[0]
    row_occurrences: list[tuple[tuple[str, ...], ...]] = []
    row_source_entity_ids: list[str] = []
    row_identity_ids: list[str] = []
    for row in range(row_count):
        paths = tuple(
            sorted(
                {
                    column_occurrences[column]
                    for column in front_faces[row]
                    if column >= 0 and column_occurrences[column]
                }
            )
        )
        row_occurrences.append(paths)
        row_kind = f"mesh-{row_source_kinds[row]}"
        entity_id = int(row_source_global_ids[row])
        row_source_entity_ids.append(
            source_entity_id(
                source_wall.mesh_id,
                str(source_wall.numeric_version),
                row_kind,
                entity_id,
            )
        )
        path_keys = tuple(
            source_entity_id(
                source_wall.mesh_id,
                str(source_wall.numeric_version),
                row_kind,
                entity_id,
                occurrence_path=path,
            )
            for path in paths
        )
        row_identity_ids.append(
            source_entity_id(
                source_wall.mesh_id,
                str(source_wall.numeric_version),
                "layer-front-row",
                canonical_fingerprint(
                    {
                        "source_entities": path_keys or row_source_entity_ids[-1:],
                        "columns": tuple(
                            column_identities[column]
                            for column in front_faces[row]
                            if column >= 0
                        ),
                        "orientation": int(row_orientations[row]),
                        "piece": int(row_pieces[row]),
                    }
                ),
                occurrence_path=paths[0] if len(paths) == 1 else (),
            )
        )
    row_order = np.argsort(np.asarray(row_identity_ids), kind="stable")
    row_ids = np.empty((row_count,), dtype=np.int64)
    row_ids[row_order] = np.arange(row_count, dtype=np.int64)
    row_columns = np.where(front_faces >= 0, column_ids[np.maximum(front_faces, 0)], -1)
    return _FrontAncestry(
        root_ids[column_order],
        np.arange(column_count, dtype=np.int64),
        column_ids,
        tuple(column_identities[column] for column in column_order),
        tuple(column_geometry_entities[column] for column in column_order),
        tuple(column_occurrences[column] for column in column_order),
        np.asarray(face_offsets, dtype=np.int64),
        np.asarray(flat_face_ids, dtype=np.int64),
        np.asarray(flat_face_orientations, dtype=np.int8),
        np.asarray(flat_face_weights, dtype=np.float64),
        np.asarray(flat_face_roles, dtype=np.int8),
        np.asarray(parent_offsets, dtype=np.int64),
        np.asarray(flat_parent_ids, dtype=np.int64),
        np.asarray(flat_parent_weights, dtype=np.float64),
        tuple(
            "sector"
            if column_kind[column] == _SECTOR
            else "fan"
            if column_kind[column] == _FAN
            else "corner"
            for column in column_order
        ),
        row_ids,
        tuple(row_identity_ids),
        np.asarray(row_pieces, dtype=np.int64),
        row_columns,
        tuple(row_source_entity_ids),
        tuple(row_source_kinds),
        np.asarray(row_source_global_ids, dtype=np.int64),
        np.asarray(row_orientations, dtype=np.int8),
        tuple(row_occurrences),
    )


def _incident_grown_triangles(
    wall: _Wall, vertices: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grown wall triangles and their ``(row, triangle)`` incidence on ``vertices``.

    Pairs are flat rather than padded to the widest vertex fan, so pole and
    cap-center fans cost their own size instead of widening every column.
    """
    grow_triangles, _ = _split_polygons(wall.faces[wall.grow], wall.arity[wall.grow])
    incidence = np.unique(
        np.stack(
            (
                grow_triangles.reshape(-1),
                np.repeat(np.arange(grow_triangles.shape[0]), 3),
            ),
            axis=1,
        ),
        axis=0,
    )
    starts = np.searchsorted(incidence[:, 0], vertices, side="left")
    counts = np.searchsorted(incidence[:, 0], vertices, side="right") - starts
    rows = np.repeat(np.arange(vertices.size), counts)
    within = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return grow_triangles, rows, incidence[np.repeat(starts, counts) + within, 1]


def _probe_stretch(wall: _Wall, front: _Front, /) -> np.ndarray:
    """Height factor that makes each column's local wall distance equal its level."""
    grow_triangles, rows, items = _incident_grown_triangles(wall, front.column_vertex)
    lengths = np.linalg.norm(
        wall.points[wall.edges[:, 0]] - wall.points[wall.edges[:, 1]], axis=1
    )
    shortest = np.full((wall.points.shape[0],), np.inf)
    np.minimum.at(shortest, wall.edges.reshape(-1), np.repeat(lengths, 2))
    epsilon = 1.0e-3 * shortest[front.column_vertex]
    probe = wall.points[front.column_vertex] + epsilon[:, None] * front.column_direction
    corners = wall.points[grow_triangles[items]]
    distance = point_triangle_distance(
        jnp.asarray(probe[rows]),
        jnp.asarray(corners[:, 0]),
        jnp.asarray(corners[:, 1]),
        jnp.asarray(corners[:, 2]),
    )
    squared = np.full((front.column_vertex.size,), np.inf)
    np.minimum.at(squared, rows, np.asarray(distance.squared_distance))
    nearest = np.sqrt(squared)
    return epsilon / np.maximum(nearest, np.finfo(np.float64).tiny)


# ---------------------------------------------------------------- height limits


def _curvature_limits(wall: _Wall, policy: BoundaryLayerPolicy, /) -> np.ndarray:
    """Admissible column height from the smooth concave radius of curvature."""
    smooth_concave = (
        wall.interior
        & ~(wall.convex | wall.concave)
        & (wall.theta > 1.0e-9)
        & (wall.bend > 0.0)
    )
    edges = wall.edges[smooth_concave]
    centroids = _polygon_centroids(wall.points, wall.faces)
    first, second = wall.edge_faces[smooth_concave].T
    spacing = np.linalg.norm(centroids[first] - centroids[second], axis=1)
    # Circle through adjacent face centroids turning by theta: d / (2 sin(theta / 2)).
    radii = spacing / (2.0 * np.sin(0.5 * wall.theta[smooth_concave]))
    radius = np.full((wall.points.shape[0],), np.inf)
    np.minimum.at(radius, edges.reshape(-1), np.repeat(radii, 2))
    return policy.curvature_fraction * radius


def _triangle_distance(triangles: np.ndarray) -> Any:
    first = jnp.asarray(triangles[:, 0])
    second = jnp.asarray(triangles[:, 1])
    third = jnp.asarray(triangles[:, 2])

    def distance(point: Any, items: Any) -> Any:
        corner = first[items]
        return point_triangle_distance(
            jnp.broadcast_to(point, corner.shape), corner, second[items], third[items]
        ).squared_distance

    return distance


def _nearest_distances(triangles: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Exact nearest point-triangle distances through a float64 BVH."""
    bvh = prepare_bvh(
        np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=jnp.float64
    )
    result = bvh_nearest_items(
        bvh,
        jnp.asarray(queries),
        k=1,
        item_distance_squared=_triangle_distance(triangles),
    )
    return np.sqrt(np.asarray(result.distance_squared, dtype=np.float64)[:, 0])


def _proximity_limits(
    wall: _Wall,
    front: _Front,
    stretch: np.ndarray,
    total: float,
    obstacle_triangles: np.ndarray,
    policy: BoundaryLayerPolicy,
    /,
) -> np.ndarray:
    """Largest safe column height before a foreign surface crosses the medial surface."""
    columns = np.flatnonzero(~_attached_vertex(wall)[front.column_vertex])
    limits = np.full((front.column_vertex.size,), np.inf)
    if not columns.size:
        return limits
    all_triangles, _ = _split_polygons(wall.faces, wall.arity)
    triangles = np.concatenate((wall.points[all_triangles], obstacle_triangles))
    grow_triangles, rows, items = _incident_grown_triangles(
        wall, front.column_vertex[columns]
    )
    samples = policy.proximity_samples
    reach = stretch[columns] * total / policy.proximity_fraction
    fractions = np.arange(1, samples + 1, dtype=np.float64) / samples
    heights = reach[:, None] * fractions[None, :]
    origin = wall.points[front.column_vertex[columns]]
    probes = (
        origin[:, None, :]
        + heights[..., None] * front.column_direction[columns][:, None, :]
    )
    nearest = _nearest_distances(triangles, probes.reshape(-1, 3)).reshape(
        columns.size, samples
    )
    own_corners = wall.points[grow_triangles[items]]
    shape = (rows.size, samples, 3)
    own = point_triangle_distance(
        jnp.asarray(probes[rows]),
        *(
            jnp.asarray(np.broadcast_to(own_corners[:, None, corner], shape))
            for corner in range(3)
        ),
    )
    own_squared = np.full((columns.size, samples), np.inf)
    np.minimum.at(own_squared, rows, np.asarray(own.squared_distance))
    own_distance = np.sqrt(own_squared)
    crossed = nearest < (1.0 - policy.medial_tolerance) * own_distance
    first = np.where(np.any(crossed, axis=1), np.argmax(crossed, axis=1), samples)
    safe = np.where(
        first > 0, heights[np.arange(columns.size), np.maximum(first - 1, 0)], 0.0
    )
    limits[columns] = np.where(first < samples, policy.proximity_fraction * safe, np.inf)
    return limits


def _attached_vertex(wall: _Wall, /) -> np.ndarray:
    attached = np.zeros((wall.points.shape[0],), dtype=np.bool_)
    attached[wall.edges[wall.attached].reshape(-1)] = True
    return attached


# ---------------------------------------------------------------- vertex graph


def _vertex_graph(wall: _Wall, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = wall.edges[wall.interior | wall.rim]
    lengths = np.linalg.norm(wall.points[edges[:, 0]] - wall.points[edges[:, 1]], axis=1)
    return edges[:, 0], edges[:, 1], 1.0 / np.maximum(lengths, np.finfo(np.float64).tiny)


def _smooth_scale(scale: np.ndarray, graph: Any, iterations: int, /) -> np.ndarray:
    """Monotone weighted Laplacian smoothing: scales only ever decrease."""
    first, second, weight = graph
    total = np.bincount(first, weight, scale.size) + np.bincount(
        second, weight, scale.size
    )
    active = total > 0.0
    for _ in range(iterations):
        accumulated = np.bincount(
            first, weight * scale[second], scale.size
        ) + np.bincount(second, weight * scale[first], scale.size)
        average = np.where(active, accumulated / np.where(active, total, 1.0), scale)
        scale = np.minimum(scale, 0.5 * (scale + average))
    return scale


def _smooth_counts(counts: np.ndarray, graph: Any, /) -> np.ndarray:
    """Neighboring layer counts differ by at most one layer."""
    first, second, _ = graph
    for _ in range(int(counts.max(initial=0)) + 1):
        bound = counts.copy()
        np.minimum.at(bound, first, counts[second] + 1)
        np.minimum.at(bound, second, counts[first] + 1)
        if np.array_equal(bound, counts):
            break
        counts = bound
    return counts


# ---------------------------------------------------------------- cells


@dataclass(frozen=True, slots=True)
class _Columns:
    """Per-column heights; vertex scale and count are shared by a vertex's columns."""

    levels: np.ndarray
    stretch: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    merged_top: np.ndarray


def _level_ids(front: _Front, columns: _Columns, base: int, level: int, /) -> np.ndarray:
    total = front.column_vertex.size
    vertex_count = columns.count[front.column_vertex]
    effective = np.minimum(level, vertex_count)
    ids = np.where(
        effective == 0,
        front.column_vertex,
        base + (np.maximum(effective, 1) - 1) * total + np.arange(total),
    )
    layers = columns.levels.size - 1
    if level >= layers:
        ids = np.where(
            columns.merged_top >= 0, base + (layers - 1) * total + columns.merged_top, ids
        )
    return ids


def _positions(front: _Front, columns: _Columns, points: np.ndarray, /) -> np.ndarray:
    layers = columns.levels.size - 1
    origin = points[front.column_vertex]
    height = (
        columns.stretch[:, None]
        * columns.scale[front.column_vertex][:, None]
        * columns.levels[None, 1:]
    )
    level_points = (
        origin[:, None, :] + height[..., None] * front.column_direction[:, None, :]
    )
    merged = columns.merged_top >= 0
    if np.any(merged):
        partner = columns.merged_top[merged]
        middle = 0.5 * (origin[merged] + origin[partner])
        level_points[merged, layers - 1] = middle
        level_points[partner, layers - 1] = middle
    return np.concatenate((points, level_points.transpose(1, 0, 2).reshape(-1, 3)))


def _front_rows(
    front: _Front, columns: _Columns, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Front faces and their exact ancestry, splitting unevenly terminated quads."""
    counts = columns.count[front.column_vertex]
    face_counts = np.where(front.faces >= 0, counts[np.maximum(front.faces, 0)], -1)
    uniform = np.all((face_counts == face_counts[:, :1]) | (front.faces < 0), axis=1)
    split = (front.kind == _WALL) & (front.arity == 4) & ~uniform
    keep = ~split
    quads = front.faces[split]
    lead = np.argmin(front.column_vertex[quads], axis=1)
    rotated = np.take_along_axis(
        quads, (lead[:, None] + np.arange(4)[None, :]) % 4, axis=1
    )
    halves = np.stack((rotated[:, (0, 1, 2)], rotated[:, (0, 2, 3)]), axis=1).reshape(
        -1, 3
    )
    rows = np.concatenate(
        (front.faces[keep], np.pad(halves, ((0, 0), (0, 1)), constant_values=-1))
    )
    arity = np.concatenate((front.arity[keep], np.full((halves.shape[0],), 3)))
    source_columns = np.arange(front.faces.shape[0], dtype=np.int64)
    source_columns = np.concatenate(
        (source_columns[keep], np.repeat(source_columns[split], 2))
    )
    return rows, arity, source_columns


def _layer_cells(
    rows: np.ndarray,
    arity: np.ndarray,
    bottom_ids: np.ndarray,
    top_ids: np.ndarray,
    source_columns: np.ndarray,
    /,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create mixed cells and carry each exact front-row column into its cells."""
    cell_rows: dict[str, list[np.ndarray]] = {name: [] for name in _VOLUME_KINDS}
    cell_columns: dict[str, list[np.ndarray]] = {name: [] for name in _VOLUME_KINDS}

    def append(
        name: str, values: np.ndarray, mask: np.ndarray, owners: np.ndarray, /
    ) -> None:
        cell_rows[name].append(values[mask])
        cell_columns[name].append(owners[mask])

    safe = np.maximum(rows, 0)
    bottom = bottom_ids[safe]
    top = top_ids[safe]
    vertical = bottom == top
    triangle = arity == 3
    b, t, v = bottom[triangle][:, :3], top[triangle][:, :3], vertical[triangle][:, :3]
    owners = source_columns[triangle]
    collapsed = np.sum(v, axis=1)
    pinched = (b[:, 0] == b[:, 1]) & (b[:, 1] == b[:, 2])
    append(
        "prism",
        np.concatenate((b, t), axis=1),
        (collapsed == 0) & ~pinched,
        owners,
    )
    corner = (collapsed == 0) & pinched
    append(
        "tetrahedron",
        np.stack((t[:, 0], t[:, 2], t[:, 1], b[:, 0]), axis=1),
        corner,
        owners,
    )
    for lead in range(3):
        i, j, k = lead, (lead + 1) % 3, (lead + 2) % 3
        one = (collapsed == 1) & v[:, i] & ~pinched
        append(
            "pyramid",
            np.stack((b[:, j], t[:, j], t[:, k], b[:, k], b[:, i]), axis=1),
            one,
            owners,
        )
        two = (collapsed == 2) & ~v[:, k] & ~pinched
        append(
            "tetrahedron",
            np.stack((b[:, i], b[:, j], b[:, k], t[:, k]), axis=1),
            two,
            owners,
        )
    quad = arity == 4
    b, t, v = bottom[quad], top[quad], vertical[quad]
    owners = source_columns[quad]
    pair_v = v[:, 0] & v[:, 1]
    pair_w = v[:, 2] & v[:, 3]
    none = ~np.any(v, axis=1)
    pinched = (b[:, 0] == b[:, 1]) & (b[:, 2] == b[:, 3])
    append(
        "hexahedron",
        np.concatenate((b, t), axis=1),
        none & ~pinched,
        owners,
    )
    append(
        "prism",
        np.stack((b[:, 0], t[:, 0], t[:, 1], b[:, 3], t[:, 3], t[:, 2]), axis=1),
        none & pinched,
        owners,
    )
    append(
        "tetrahedron",
        np.stack((b[:, 2], t[:, 2], t[:, 3], b[:, 0]), axis=1),
        pinched & pair_v & ~pair_w,
        owners,
    )
    append(
        "tetrahedron",
        np.stack((b[:, 0], t[:, 0], t[:, 1], b[:, 2]), axis=1),
        pinched & pair_w & ~pair_v,
        owners,
    )
    append(
        "prism",
        np.stack((b[:, 0], b[:, 3], t[:, 3], b[:, 1], b[:, 2], t[:, 2]), axis=1),
        ~pinched & pair_v & ~pair_w,
        owners,
    )
    append(
        "prism",
        np.stack((b[:, 2], b[:, 1], t[:, 1], b[:, 3], b[:, 0], t[:, 0]), axis=1),
        ~pinched & pair_w & ~pair_v,
        owners,
    )
    half = (b[:, 0] == b[:, 1]) ^ (b[:, 2] == b[:, 3])
    irregular = half | (
        ~none & ~(pair_v & pair_w) & ~((pair_v ^ pair_w) & (np.sum(v, axis=1) == 2))
    )
    if np.any(irregular):
        raise ValueError("Quadrilateral front faces admit only paired side collapses.")
    return (
        {
            name: np.concatenate(values).reshape(-1, _CELL_ARITY[name]).astype(np.int64)
            for name, values in cell_rows.items()
        },
        {
            name: np.concatenate(values).astype(np.int64, copy=False)
            for name, values in cell_columns.items()
        },
    )


def _cell_triangles(
    cells: dict[str, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulated cell faces with owning (kind, row) pairs."""
    triangles = []
    kinds = []
    owners = []
    for kind_index, name in enumerate(_VOLUME_KINDS):
        rows = cells[name]
        if not rows.size:
            continue
        for route in _FACE_ROUTES[name]:
            face = rows[:, route]
            if len(route) == 4:
                face = _canonical_quads(face)
                split = np.stack(
                    (face[:, (0, 1, 2)], face[:, (0, 2, 3)]), axis=1
                ).reshape(-1, 3)
                owner = np.repeat(np.arange(rows.shape[0]), 2)
            else:
                split = face
                owner = np.arange(rows.shape[0])
            triangles.append(split)
            owners.append(owner)
            kinds.append(np.full(owner.shape, kind_index))
    if not triangles:
        empty = np.empty((0,), dtype=np.int64)
        return np.empty((0, 3), dtype=np.int64), empty, empty
    return np.concatenate(triangles), np.concatenate(kinds), np.concatenate(owners)


def _cell_mesh(
    points: np.ndarray, cells: dict[str, np.ndarray], /
) -> tuple[CellMesh, np.ndarray]:
    used = np.unique(np.concatenate([rows.reshape(-1) for rows in cells.values()]))
    remap = np.full((points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    blocks = []
    cursor = 0
    for name in _VOLUME_KINDS:
        rows = cells[name]
        if not rows.size:
            continue
        blocks.append(
            CellBlock(
                _BLOCK_NAMES[name],
                name,
                remap[rows],
                global_ids=np.arange(cursor, cursor + rows.shape[0], dtype=np.int64),
            )
        )
        cursor += rows.shape[0]
    return CellMesh(points[used], tuple(blocks)), used


# ---------------------------------------------------------------- certification


@dataclass(frozen=True, slots=True)
class _Environment:
    """Fixed triangles the layers must not cross.

    Wall triangles index the leading wall vertices of the layer points (so shared
    wall vertices are recognized as welded); obstacle triangles index
    ``obstacle_points``. ``side`` marks non-grown wall triangles on which rim
    sheets may lie.
    """

    wall_triangles: np.ndarray
    side: np.ndarray
    obstacle_points: np.ndarray
    obstacle_triangles: np.ndarray


def _certify_cells(
    points: np.ndarray,
    cells: dict[str, np.ndarray],
    accepted: list[dict[str, np.ndarray]],
    environment: _Environment,
    policy: BoundaryLayerPolicy,
    /,
) -> tuple[dict[str, np.ndarray], int, CellValidityCertificate | None]:
    """Invalid or intersecting new cells (per kind masks), certified hit count and
    the validity certificate of the new cells (``None`` without new cells)."""
    bad = {
        name: np.zeros((rows.shape[0],), dtype=np.bool_) for name, rows in cells.items()
    }
    if not any(rows.size for rows in cells.values()):
        return bad, 0, None
    mesh, _ = _cell_mesh(points, cells)
    certificate = certify_cell_geometry_validity(mesh, policy=policy.validity)
    status = np.asarray(certificate.status)
    cursor = 0
    for name in _VOLUME_KINDS:
        count = cells[name].shape[0]
        bad[name] |= status[cursor : cursor + count] != int(
            CellValidityStatus.CERTIFIED_VALID
        )
        cursor += count
    new_triangles, new_kinds, new_owners = _cell_triangles(cells)
    old = [_cell_triangles(values)[0] for values in accepted]
    offset = points.shape[0]
    fixed = np.concatenate(
        (environment.wall_triangles, environment.obstacle_triangles + offset)
    )
    combined_points = np.concatenate((points, environment.obstacle_points))
    triangles = np.concatenate((new_triangles, *old, fixed))
    new_count = new_triangles.shape[0]
    fixed_start = triangles.shape[0] - fixed.shape[0]
    corners = combined_points[triangles]
    lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
    side = np.zeros((triangles.shape[0],), dtype=np.bool_)
    side[fixed_start : fixed_start + environment.side.size] = environment.side
    count = 0
    # Only pairs with a new triangle are tested, so fixed and accepted fans
    # (e.g. pole fans) never pair among themselves; pairs stream in bounded
    # blocks instead of being materialized at once.
    for first, second in bvh_overlap_pair_blocks(
        prepare_bvh(lower[:new_count], upper[:new_count], dtype=jnp.float64),
        prepare_bvh(lower, upper, dtype=jnp.float64),
        include_touching=True,
    ):
        keep = first < second
        first, second = first[keep], second[keep]
        if not first.size:
            continue
        hit, certain = _triangle_pairs_intersect(
            combined_points, triangles[first], triangles[second]
        )
        sheet = side[second]
        if np.any(sheet & hit):
            rows = np.flatnonzero(sheet & hit)
            plane = triangles[second[rows]]
            above = np.zeros((rows.size,), dtype=np.bool_)
            below = np.zeros((rows.size,), dtype=np.bool_)
            known_all = np.ones((rows.size,), dtype=np.bool_)
            for corner in range(3):
                sign, known = _orient3d(
                    combined_points[plane[:, 0]],
                    combined_points[plane[:, 1]],
                    combined_points[plane[:, 2]],
                    combined_points[triangles[first[rows], corner]],
                )
                above |= sign > 0
                below |= sign < 0
                known_all &= known
            # Rim sheets slide on the adjacent surface: an intersection that does
            # not strictly straddle the surface plane is contact on it, not
            # penetration.
            hit[rows[known_all & ~(above & below)]] = False
        colliding = hit | ~certain
        count += int(np.count_nonzero(hit & certain))
        for index in (first[colliding], second[colliding]):
            index = index[index < new_count]
            for kind_index, name in enumerate(_VOLUME_KINDS):
                selected = new_owners[index[new_kinds[index] == kind_index]]
                bad[name][selected] = True
    return bad, count, certificate


# ---------------------------------------------------------------- merge template


def _merge_pairs(
    wall: _Wall,
    front: _Front,
    candidates: np.ndarray,
    medial: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutually paired opposing sector columns and their gap lengths."""
    sector_of_vertex = np.full((wall.points.shape[0],), -1, dtype=np.int64)
    sectors = np.flatnonzero(front.column_kind == _SECTOR)
    counts = np.bincount(front.column_vertex[sectors], minlength=wall.points.shape[0])
    sector_of_vertex[front.column_vertex[sectors]] = sectors
    vertices = np.flatnonzero(candidates)
    unpaired = vertices[counts[vertices] != 1]
    if unpaired.size:
        raise _failure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "MERGE pairs smooth single-column fronts only; colliding feature vertices remain.",
            unpaired,
            wall.points,
        )
    columns = sector_of_vertex[vertices]
    origin = wall.points[vertices]
    direction = front.column_direction[columns]
    probe = origin + 2.0 * medial[vertices][:, None] * direction
    candidate_points = wall.points[vertices]
    bvh = prepare_bvh(candidate_points, candidate_points, dtype=jnp.float64)
    nearest = np.asarray(bvh_nearest_items(bvh, jnp.asarray(probe), k=1).items)[:, 0]
    partner = vertices[nearest]
    back = np.full((wall.points.shape[0],), -1, dtype=np.int64)
    back[vertices] = partner
    offset = wall.points[partner] - origin
    gap = np.sum(offset * direction, axis=1)
    lateral = np.linalg.norm(offset - gap[:, None] * direction, axis=1)
    opposing = np.sum(
        direction * front.column_direction[sector_of_vertex[partner]], axis=1
    )
    valid = (
        (back[partner] == vertices)
        & (gap > 0.0)
        & (lateral <= 1.0e-6 * np.maximum(gap, 1.0e-300))
        & (opposing <= -1.0 + 1.0e-6)
    )
    if not np.all(valid):
        raise _failure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "MERGE requires mutually paired, antiparallel, aligned opposing columns.",
            vertices[~valid],
            wall.points,
        )
    grow_rows = np.flatnonzero(wall.grow)
    faces = wall.faces[grow_rows]
    lookup = {
        tuple(sorted(int(value) for value in row if value >= 0)): int(index)
        for index, row in zip(grow_rows, faces, strict=True)
    }
    for row in faces:
        members = row[row >= 0]
        if np.all(candidates[members]):
            image = tuple(sorted(int(back[value]) for value in members))
            if image not in lookup:
                raise _failure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    "MERGE requires face-to-face paired opposing fronts.",
                    members,
                    wall.points,
                )
    gaps = np.full((wall.points.shape[0],), np.nan)
    gaps[vertices] = gap
    return back, gaps


# ---------------------------------------------------------------- growth driver


@dataclass(frozen=True, slots=True)
class _Resolution:
    columns: _Columns
    cells: list[dict[str, np.ndarray]]
    cell_columns: list[dict[str, np.ndarray]]
    points: np.ndarray
    detected: int
    iterations: int
    # Validity proofs of the accepted layers' cells, reused by assembly.
    validity: tuple[CellValidityCertificate, ...]


def _initial_columns(
    wall: _Wall,
    front: _Front,
    stretch: np.ndarray,
    limits: np.ndarray,
    medial: np.ndarray,
    curvature_limited: np.ndarray,
    control: BoundaryLayerControl,
    graph: Any,
    /,
) -> tuple[_Columns, int]:
    thicknesses = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    levels = np.concatenate(([0.0], np.cumsum(thicknesses)))
    total = float(levels[-1])
    layers = thicknesses.size
    vertex_count = wall.points.shape[0]
    column_height = stretch * total
    ratio = np.full((vertex_count,), np.inf)
    np.minimum.at(ratio, front.column_vertex, limits / column_height)
    predicted = ratio < 1.0
    scale = np.ones((vertex_count,))
    count = np.full((vertex_count,), layers, dtype=np.int64)
    merged_top = np.full((front.column_vertex.size,), -1, dtype=np.int64)
    fraction = control.minimum_thickness_fraction
    match control.collision:
        case BoundaryLayerCollisionPolicy.FAIL:
            if np.any(predicted):
                raise _failure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    "Boundary-layer fronts collide with other walls or themselves "
                    f"(minimum admissible scale {float(np.min(ratio)):.6g}).",
                    np.flatnonzero(predicted),
                    wall.points,
                )
        case BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
            scale = _smooth_scale(
                np.minimum(1.0, ratio), graph, control.smoothing_iterations
            )
            if np.any(scale[wall.wall_vertices] < fraction):
                raise _failure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    "Thickness reduction below the minimum thickness fraction "
                    f"(required scale {float(np.min(scale[wall.wall_vertices])):.6g} < {fraction:.6g}).",
                    wall.wall_vertices[scale[wall.wall_vertices] < fraction],
                    wall.points,
                )
        case BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
            allowed = np.sum(
                levels[None, 1:] <= ratio[:, None] * total * (1.0 + 1.0e-12), axis=1
            )
            count = _smooth_counts(np.minimum(count, allowed), graph)
            _require_first_layer(wall, count)
        case BoundaryLayerCollisionPolicy.MERGE:
            limited = predicted & curvature_limited
            if np.any(limited):
                raise _failure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    "MERGE joins opposing fronts only; curvature-limited fronts collide.",
                    np.flatnonzero(limited),
                    wall.points,
                )
            if np.any(predicted):
                _merge_template(
                    wall,
                    front,
                    predicted,
                    medial,
                    stretch * total,
                    fraction,
                    scale,
                    merged_top,
                )
        case _:
            raise TypeError("collision must be BoundaryLayerCollisionPolicy.")
    return (
        _Columns(levels, stretch, scale, count, merged_top),
        int(np.count_nonzero(predicted)),
    )


def _merge_template(
    wall: Any,
    front: Any,
    predicted: Any,
    medial: Any,
    column_height: Any,
    fraction: Any,
    scale: Any,
    merged_top: Any,
) -> None:
    """Scale paired opposing columns to meet at their midpoint and share its vertex."""
    partner, gap = _merge_pairs(wall, front, predicted, medial)
    sectors = np.flatnonzero(front.column_kind == _SECTOR)
    column_of = np.full((wall.points.shape[0],), -1, dtype=np.int64)
    column_of[front.column_vertex[sectors]] = sectors
    vertices = np.flatnonzero(predicted)
    scale[vertices] = 0.5 * gap[vertices] / column_height[column_of[vertices]]
    if np.any(scale[vertices] < fraction):
        raise _failure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "Merged fronts compress layers below the minimum thickness fraction.",
            vertices[scale[vertices] < fraction],
            wall.points,
        )
    lower = vertices[vertices < partner[vertices]]
    merged_top[column_of[partner[lower]]] = column_of[lower]


def _grow_layers(
    wall: _Wall,
    front: _Front,
    columns: _Columns,
    environment: _Environment,
    control: BoundaryLayerControl,
    policy: BoundaryLayerPolicy,
    graph: Any,
    /,
) -> _Resolution:
    layers = columns.levels.size - 1
    base = wall.points.shape[0]
    detected = 0
    iterations = 0
    while True:
        accepted: list[dict[str, np.ndarray]] = []
        accepted_columns: list[dict[str, np.ndarray]] = []
        accepted_validity: list[CellValidityCertificate] = []
        points = _positions(front, columns, wall.points)
        rows, arity, source_columns = _front_rows(front, columns)
        for layer in range(layers):
            cells, cell_columns = _layer_cells(
                rows,
                arity,
                _level_ids(front, columns, base, layer),
                _level_ids(front, columns, base, layer + 1),
                source_columns,
            )
            bad, hits, certificate = _certify_cells(
                points, cells, accepted, environment, policy
            )
            if any(np.any(mask) for mask in bad.values()):
                break
            accepted.append(cells)
            accepted_columns.append(cell_columns)
            if certificate is not None:
                accepted_validity.append(certificate)
        else:
            return _Resolution(
                columns,
                accepted,
                accepted_columns,
                points,
                detected,
                iterations,
                tuple(accepted_validity),
            )
        iterations += 1
        offending = np.unique(
            np.concatenate([cells[name][bad[name]].reshape(-1) for name in _VOLUME_KINDS])
        )
        column = (offending - base) % front.column_vertex.size
        vertices = np.unique(
            np.where(offending < base, offending, front.column_vertex[column])
        )
        vertices = vertices[np.isin(vertices, wall.wall_vertices)]
        match control.collision:
            case BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
                count = columns.count.copy()
                count[vertices] = np.minimum(count[vertices], layer)
                count = _smooth_counts(count, graph)
                _require_first_layer(wall, count)
                if np.array_equal(count, columns.count):
                    raise _failure(
                        MeshingFailureCategory.CONTROL_CONFLICT,
                        "Local termination cannot remove another colliding layer.",
                        vertices,
                        wall.points,
                    )
                columns = _Columns(
                    columns.levels,
                    columns.stretch,
                    columns.scale,
                    count,
                    columns.merged_top,
                )
            case BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
                scale = columns.scale.copy()
                scale[vertices] = np.maximum(
                    scale[vertices] * policy.reduction_factor,
                    control.minimum_thickness_fraction,
                )
                scale = _smooth_scale(scale, graph, control.smoothing_iterations)
                if np.any(scale[wall.wall_vertices] < control.minimum_thickness_fraction):
                    raise _failure(
                        MeshingFailureCategory.CONTROL_CONFLICT,
                        "Certified collisions persist at the minimum thickness fraction.",
                        vertices,
                        wall.points,
                    )
                if np.array_equal(scale, columns.scale):
                    raise _failure(
                        MeshingFailureCategory.CONTROL_CONFLICT,
                        "Certified collisions persist without a representable thickness reduction.",
                        vertices,
                        wall.points,
                    )
                columns = _Columns(
                    columns.levels,
                    columns.stretch,
                    scale,
                    columns.count,
                    columns.merged_top,
                )
            case BoundaryLayerCollisionPolicy.FAIL | BoundaryLayerCollisionPolicy.MERGE:
                raise _failure(
                    MeshingFailureCategory.CONTROL_CONFLICT,
                    f"Layer {layer} cells intersect or are not certified valid "
                    f"({hits} certified intersections).",
                    vertices,
                    wall.points,
                )
            case _:
                raise TypeError("collision must be BoundaryLayerCollisionPolicy.")


def _require_first_layer(wall: _Wall, count: np.ndarray, /) -> None:
    """Local termination keeps at least the first layer at every wall vertex."""
    empty = wall.wall_vertices[count[wall.wall_vertices] < 1]
    if empty.size:
        raise _failure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "Local termination would remove the first layer; the first layer itself collides.",
            empty,
            wall.points,
        )


# ---------------------------------------------------------------- cap and pyramids


def _cap_faces(
    front: _Front, columns: _Columns, base: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, arity, source_columns = _front_rows(front, columns)
    layers = columns.levels.size - 1
    ids = _level_ids(front, columns, base, layers)
    top = np.where(rows >= 0, ids[np.maximum(rows, 0)], -1)
    faces = []
    face_columns = []
    for row, size, column in zip(top, arity, source_columns, strict=True):
        loop = [int(value) for value in row[:size]]
        distinct = []
        for value in loop:
            if not distinct or distinct[-1] != value:
                distinct.append(value)
        if len(distinct) > 1 and distinct[0] == distinct[-1]:
            distinct.pop()
        if len(distinct) >= 3:
            faces.append(distinct + [-1] * (4 - len(distinct)))
            face_columns.append(int(column))
    cap = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    cap_columns = np.asarray(face_columns, dtype=np.int64)
    keys = np.sort(np.where(cap >= 0, cap, np.iinfo(np.int64).max), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    single = counts[inverse.reshape(-1)] == 1
    cap = cap[single]
    cap_columns = cap_columns[single]
    return cap, np.where(cap[:, 3] < 0, 3, 4), cap_columns


def _cap_pyramids(
    points: np.ndarray,
    cap: np.ndarray,
    arity: np.ndarray,
    accepted: list[dict[str, np.ndarray]],
    environment: _Environment,
    policy: BoundaryLayerPolicy,
    /,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    tuple[CellValidityCertificate, ...],
]:
    quads = cap[arity == 4]
    empty = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    if not quads.size:
        return points, empty, cap, arity, ()
    corners = points[quads]
    normal, _ = _newell_normals(points, quads, np.full((quads.shape[0],), 4))
    edge = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
    height = 0.5 * np.mean(edge, axis=1)
    centroid = np.mean(corners, axis=1)
    apex_ids = points.shape[0] + np.arange(quads.shape[0])
    while True:
        apex = centroid + height[:, None] * normal
        candidate = np.concatenate((points, apex))
        # The cap quad faces away from the layer; the pyramid base must face its apex.
        pyramids = np.concatenate((quads[:, (0, 1, 2, 3)], apex_ids[:, None]), axis=1)
        cells = dict(empty)
        cells["pyramid"] = pyramids
        bad, _, certificate = _certify_cells(
            candidate, cells, accepted, environment, policy
        )
        if not np.any(bad["pyramid"]):
            sides = np.concatenate(
                [
                    np.stack((quads[:, i], quads[:, (i + 1) % 4], apex_ids), axis=1)
                    for i in range(4)
                ]
            )
            triangles = cap[arity == 3]
            faces = np.concatenate(
                (
                    np.pad(triangles[:, :3], ((0, 0), (0, 1)), constant_values=-1),
                    np.pad(sides, ((0, 0), (0, 1)), constant_values=-1),
                )
            )
            return (
                candidate,
                cells,
                faces,
                np.full((faces.shape[0],), 3),
                () if certificate is None else (certificate,),
            )
        reduced = np.where(bad["pyramid"], 0.5 * height, height)
        if np.any(bad["pyramid"] & ((reduced == height) | (reduced == 0.0))):
            raise _failure(
                MeshingFailureCategory.CONTROL_CONFLICT,
                "Transition pyramids cannot resolve collisions with a smaller positive height.",
                quads[bad["pyramid"]].reshape(-1),
                points,
            )
        height = reduced


# ---------------------------------------------------------------- evidence


def _measured_thicknesses(
    wall: _Wall, front: _Front, columns: _Columns, points: np.ndarray, /
) -> Any:
    layers = columns.levels.size - 1
    grow_triangles, _ = _split_polygons(wall.faces[wall.grow], wall.arity[wall.grow])
    total = front.column_vertex.size
    level_points = points[wall.points.shape[0] :].reshape(layers, total, 3)
    distance = _nearest_distances(
        wall.points[grow_triangles], level_points.reshape(-1, 3)
    ).reshape(layers, total)
    distance = np.concatenate((np.zeros((1, total)), distance))
    carried = (
        np.arange(1, layers + 1)[:, None] <= columns.count[front.column_vertex][None, :]
    )
    active = np.any(carried, axis=1)
    increments = np.diff(distance, axis=0)
    mean = np.full((layers,), np.nan)
    minimum = np.full((layers,), np.nan)
    maximum = np.full((layers,), np.nan)
    for layer in range(layers):
        if active[layer]:
            values = increments[layer, carried[layer]]
            mean[layer] = np.mean(values)
            minimum[layer] = np.min(values)
            maximum[layer] = np.max(values)
    return mean, minimum, maximum, active


# ---------------------------------------------------------------- entry points


def _grow_boundary_layers(
    points: np.ndarray,
    faces: np.ndarray,
    arity: np.ndarray,
    grow: np.ndarray,
    obstacle_points: np.ndarray,
    obstacle_triangles: np.ndarray,
    control: BoundaryLayerControl,
    policy: BoundaryLayerPolicy,
    source_wall: CellMesh,
    wall_association: GeometryAssociation | None,
    /,
) -> BoundaryLayerMesh:
    """Grow certified layers from ``grow`` faces of one oriented surface mesh."""
    if not np.any(grow):
        raise ValueError("Boundary layers require at least one wall face.")
    # Layer certification decides exact coplanar contacts; no filtered fallback.
    load_meshcore()
    wall = _analyze_wall(points, faces, arity, grow, control.feature_angle)
    split = _split_ridges(wall, control.corner)
    sectors = _sectors(wall, split)
    directions = _sector_directions(wall, sectors, control)
    front = _build_front(
        wall, sectors, split, directions, control, source_wall, wall_association
    )
    stretch = _probe_stretch(wall, front)
    too_stretched = stretch > control.maximum_corner_stretch
    if np.any(too_stretched):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Concave corner columns need a height stretch of "
            f"{float(np.max(stretch)):.6g} > maximum_corner_stretch "
            f"{control.maximum_corner_stretch:.6g}.",
            front.column_vertex[too_stretched],
            points,
        )
    obstacles = obstacle_points[obstacle_triangles]
    limits, curvature_limited, medial = _column_limits(
        wall, front, stretch, obstacles, control, policy
    )
    graph = _vertex_graph(wall)
    columns, predicted = _initial_columns(
        wall, front, stretch, limits, medial, curvature_limited, control, graph
    )
    wall_triangles, _ = _split_polygons(faces, arity)
    grown = np.concatenate((grow[arity == 3], np.repeat(grow[arity == 4], 2)))
    environment = _Environment(
        wall_triangles, ~grown, obstacle_points, obstacle_triangles
    )
    resolution = _grow_layers(wall, front, columns, environment, control, policy, graph)
    return _assemble(
        wall,
        front,
        directions,
        stretch,
        predicted,
        resolution,
        environment,
        control,
        policy,
        source_wall,
        wall_association,
    )


def _column_limits(
    wall: Any, front: Any, stretch: Any, obstacles: Any, control: Any, policy: Any, /
) -> Any:
    """Per-column height limits, curvature-limited vertices, and medial heights."""
    total = control.schedule.total_thickness
    proximity = _proximity_limits(wall, front, stretch, total, obstacles, policy)
    curvature = _curvature_limits(wall, policy)[front.column_vertex]
    count = wall.points.shape[0]
    curvature_limited = np.zeros((count,), dtype=np.bool_)
    np.logical_or.at(curvature_limited, front.column_vertex, curvature < stretch * total)
    medial = np.full((count,), np.inf)
    np.minimum.at(medial, front.column_vertex, proximity / policy.proximity_fraction)
    return np.minimum(proximity, curvature), curvature_limited, medial


def _cap_mesh(points: np.ndarray, cap: np.ndarray, arity: np.ndarray, /) -> Any:
    vertices = np.unique(cap[cap >= 0])
    if not vertices.size:
        return None, vertices
    local = np.full((points.shape[0],), -1, dtype=np.int64)
    local[vertices] = np.arange(vertices.size)
    blocks = []
    cursor = 0
    for name, size in (("triangle", 3), ("quadrilateral", 4)):
        rows = cap[arity == size][:, :size]
        if rows.size:
            blocks.append(
                CellBlock(
                    _BLOCK_NAMES[name],
                    name,
                    local[rows],
                    global_ids=np.arange(cursor, cursor + rows.shape[0], dtype=np.int64),
                )
            )
            cursor += rows.shape[0]
    return CellMesh(points[vertices], tuple(blocks)), vertices


def _output_vertex_ancestry(
    source_wall: CellMesh,
    wall_association: GeometryAssociation | None,
    front: _Front,
    columns: _Columns,
    used: np.ndarray,
    pyramid_front_rows: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    tuple[tuple[str, ...], ...],
    np.ndarray,
    tuple[tuple[tuple[str, ...], ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return exact source-root and column lineage for the used coordinate bank."""
    source_vertex_ids = np.asarray(source_wall.vertex_global_ids, dtype=np.int64)
    column_ids = front.ancestry.column_ids_by_front_vertex
    roots = source_vertex_ids[front.column_vertex]
    base_count = source_vertex_ids.size
    column_count = front.column_vertex.size
    layer_count = columns.levels.size - 1
    point_count = base_count + layer_count * column_count + pyramid_front_rows.size
    point_source_ids = np.full((point_count,), -1, dtype=np.int64)
    point_column_ids = np.full((point_count,), -1, dtype=np.int64)
    point_parents: list[tuple[tuple[int, float], ...]] = [() for _ in range(point_count)]
    for row, source_id in enumerate(source_vertex_ids):
        point_source_ids[row] = source_id
    merged: dict[int, list[int]] = {}
    for partner, lower in enumerate(columns.merged_top):
        if lower >= 0:
            merged.setdefault(int(lower), [int(lower)]).append(partner)
    for level in range(layer_count):
        for column in range(column_count):
            point = base_count + level * column_count + column
            members = (
                merged.get(column, [column]) if level == layer_count - 1 else [column]
            )
            member_ids = sorted(int(column_ids[member]) for member in members)
            source_ids = {int(roots[member]) for member in members}
            point_source_ids[point] = (
                next(iter(source_ids)) if len(source_ids) == 1 else -1
            )
            point_column_ids[point] = member_ids[0]
            weight = 1.0 / len(member_ids)
            point_parents[point] = tuple(
                (identifier, weight) for identifier in member_ids
            )
    for apex, front_row in enumerate(pyramid_front_rows):
        point = base_count + layer_count * column_count + apex
        parent_ids = tuple(
            int(value)
            for value in front.ancestry.front_row_column_ids[int(front_row)]
            if value >= 0
        )
        if not parent_ids:
            raise ValueError("A cap transition apex must inherit a nonempty front row.")
        source_ids = {
            int(front.ancestry.column_vertex_sources[identifier])
            for identifier in parent_ids
        }
        point_source_ids[point] = next(iter(source_ids)) if len(source_ids) == 1 else -1
        point_column_ids[point] = min(parent_ids)
        weight = 1.0 / len(parent_ids)
        point_parents[point] = tuple((identifier, weight) for identifier in parent_ids)
    if used.size and int(np.max(used)) >= point_count:
        raise ValueError("Layer coordinate ancestry does not cover every output vertex.")

    association_rows: dict[int, int] = {}
    if wall_association is not None:
        association_rows = {
            int(identifier): row
            for row, identifier in enumerate(
                np.asarray(wall_association.target_global_ids, dtype=np.int64)
            )
        }

    def source_entities(
        source_ids: set[int],
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        entities: set[str] = set()
        occurrences: set[tuple[str, ...]] = set()
        if wall_association is not None:
            for source_id in source_ids:
                row = association_rows.get(source_id)
                if row is not None:
                    entities.add(wall_association.source_entity_ids[row])
                    path = wall_association.source_occurrence_paths[row]
                    if path:
                        occurrences.add(path)
        return tuple(sorted(entities)), tuple(sorted(occurrences))

    vertex_source_entities = []
    vertex_occurrences = []
    for point in used:
        source_id = int(point_source_ids[point])
        if source_id >= 0:
            source_ids = {source_id}
        else:
            source_ids = {
                int(front.ancestry.column_vertex_sources[column])
                for column, _ in point_parents[int(point)]
            }
        entities, occurrences = source_entities(source_ids)
        vertex_source_entities.append(entities)
        vertex_occurrences.append(occurrences)
    parent_offsets = [0]
    parent_ids = []
    parent_weights = []
    for point in used:
        for identifier, weight in point_parents[int(point)]:
            parent_ids.append(identifier)
            parent_weights.append(weight)
        parent_offsets.append(len(parent_ids))
    return (
        point_source_ids[used],
        tuple(vertex_source_entities),
        point_column_ids[used],
        tuple(vertex_occurrences),
        np.asarray(parent_offsets, dtype=np.int64),
        np.asarray(parent_ids, dtype=np.int64),
        np.asarray(parent_weights, dtype=np.float64),
    )


def _assemble(
    wall: _Wall,
    front: _Front,
    directions: _Directions,
    stretch: np.ndarray,
    predicted: int,
    resolution: _Resolution,
    environment: _Environment,
    control: BoundaryLayerControl,
    policy: BoundaryLayerPolicy,
    source_wall: CellMesh,
    wall_association: GeometryAssociation | None,
    /,
) -> BoundaryLayerMesh:
    columns = resolution.columns
    layers = columns.levels.size - 1
    cap, cap_arity, cap_columns = _cap_faces(front, columns, wall.points.shape[0])
    all_points = resolution.points
    pyramids = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    pyramid_columns = np.empty((0,), dtype=np.int64)
    pyramid_validity: tuple[CellValidityCertificate, ...] = ()
    if policy.simplex_cap:
        pyramid_columns = cap_columns[cap_arity == 4]
        all_points, pyramids, cap, cap_arity, pyramid_validity = _cap_pyramids(
            all_points, cap, cap_arity, resolution.cells, environment, policy
        )
    cells = {}
    layer_index = []
    column_index = []
    for name in _VOLUME_KINDS:
        rows = [values[name] for values in resolution.cells] + [pyramids[name]]
        column_rows = [
            front.ancestry.front_row_ids[values[name]]
            for values in resolution.cell_columns
        ] + [
            front.ancestry.front_row_ids[pyramid_columns]
            if name == "pyramid"
            else np.empty((0,), dtype=np.int64)
        ]
        cells[name] = np.concatenate(rows)
        column_index.extend(column_rows)
        layer_index.extend(
            np.full((values.shape[0],), layer) for layer, values in enumerate(rows)
        )
    mesh, used = _cell_mesh(all_points, cells)
    remap = np.full((all_points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    # Layer and pyramid cells keep the exact maps their acceptance proved.
    certificate = certify_cell_geometry_validity(
        mesh,
        policy=policy.validity,
        parent=resolution.validity + pyramid_validity,
    )
    invalid = np.asarray(certificate.status) != int(CellValidityStatus.CERTIFIED_VALID)
    if np.any(invalid):
        offsets = np.cumsum([0, *(cells[name].shape[0] for name in _VOLUME_KINDS)])
        offending = np.concatenate(
            [
                cells[name][invalid[offsets[index] : offsets[index + 1]]].reshape(-1)
                for index, name in enumerate(_VOLUME_KINDS)
            ]
        )
        raise _failure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Assembled boundary-layer cells are not all certified valid.",
            offending,
            all_points,
        )
    cap_mesh, cap_vertices = _cap_mesh(all_points, cap, cap_arity)
    mean, minimum, maximum, active = _measured_thicknesses(
        wall, front, columns, resolution.points
    )
    wall_scale = columns.scale[wall.wall_vertices]
    evidence = BoundaryLayerEvidence(
        requested_thicknesses=control.schedule.thicknesses,
        achieved_thicknesses=mean,
        minimum_thicknesses=minimum,
        maximum_thicknesses=maximum,
        achieved_growth_rates=mean[1:] / mean[:-1],
        layer_active=active,
        column_count=front.column_vertex.size,
        fan_column_count=front.fan_column_count,
        corner_patch_count=front.corner_patch_count,
        convex_ridge_count=np.count_nonzero(wall.convex),
        concave_ridge_count=np.count_nonzero(wall.concave),
        rim_vertex_count=np.count_nonzero(wall.rim_vertex),
        minimum_visibility=np.min(directions.visibility),
        maximum_stretch=np.max(stretch),
        collision_policy=control.collision,
        predicted_collision_vertex_count=predicted,
        detected_collision_count=resolution.detected,
        resolution_iterations=resolution.iterations,
        reduced_vertex_count=np.count_nonzero(wall_scale < 1.0),
        terminated_vertex_count=np.count_nonzero(
            columns.count[wall.wall_vertices] < layers
        ),
        merged_vertex_count=2 * np.count_nonzero(columns.merged_top >= 0),
        minimum_scale=np.min(wall_scale),
        cell_counts=tuple((block.cell_kind, block.cell_count) for block in mesh.blocks),
        certified_valid_count=certificate.certified_valid_count,
    )
    (
        vertex_source_vertex_ids,
        vertex_source_entity_ids,
        vertex_column_ids,
        vertex_source_occurrence_paths,
        vertex_column_parent_offsets,
        vertex_column_parent_ids,
        vertex_column_parent_weights,
    ) = _output_vertex_ancestry(
        source_wall,
        wall_association,
        front,
        columns,
        used,
        pyramid_columns,
    )
    return BoundaryLayerMesh(
        mesh,
        cap_mesh,
        remap[: wall.points.shape[0]],
        remap[cap_vertices],
        np.concatenate(layer_index),
        certificate,
        evidence,
        source_wall=source_wall,
        wall_association=wall_association,
        column_index=np.concatenate(column_index),
        column_vertex_sources=front.ancestry.column_vertex_sources,
        column_vertex_ids=front.ancestry.column_vertex_ids,
        column_vertex_identity=front.ancestry.column_vertex_identity,
        column_source_geometry_entity_ids=(
            front.ancestry.column_source_geometry_entity_ids
        ),
        column_source_occurrence_paths=front.ancestry.column_source_occurrence_paths,
        column_source_face_offsets=front.ancestry.column_source_face_offsets,
        column_source_face_ids=front.ancestry.column_source_face_ids,
        column_source_face_orientations=front.ancestry.column_source_face_orientations,
        column_source_face_weights=front.ancestry.column_source_face_weights,
        column_source_face_roles=front.ancestry.column_source_face_roles,
        column_parent_offsets=front.ancestry.column_parent_offsets,
        column_parent_ids=front.ancestry.column_parent_ids,
        column_parent_weights=front.ancestry.column_parent_weights,
        column_construction_kinds=front.ancestry.column_construction_kinds,
        front_row_ids=front.ancestry.front_row_ids,
        front_row_identity=front.ancestry.front_row_identity,
        front_row_source_piece_indices=front.ancestry.front_row_source_piece_indices,
        front_row_column_ids=front.ancestry.front_row_column_ids,
        front_row_source_entity_ids=front.ancestry.front_row_source_entity_ids,
        front_row_source_entity_kinds=front.ancestry.front_row_source_entity_kinds,
        front_row_source_global_ids=front.ancestry.front_row_source_global_ids,
        front_row_orientations=front.ancestry.front_row_orientations,
        front_row_source_occurrence_paths=(
            front.ancestry.front_row_source_occurrence_paths
        ),
        vertex_source_vertex_ids=vertex_source_vertex_ids,
        vertex_source_entity_ids=vertex_source_entity_ids,
        vertex_column_ids=vertex_column_ids,
        vertex_source_occurrence_paths=vertex_source_occurrence_paths,
        vertex_column_parent_offsets=vertex_column_parent_offsets,
        vertex_column_parent_ids=vertex_column_parent_ids,
        vertex_column_parent_weights=vertex_column_parent_weights,
        control_id=control.control_id,
        policy_id=policy.policy_id,
    )


def prepare_boundary_layers(
    wall: CellMesh,
    control: BoundaryLayerControl,
    /,
    *,
    obstacles: CellMesh | None = None,
    wall_association: GeometryAssociation | None = None,
    policy: BoundaryLayerPolicy | None = None,
) -> BoundaryLayerMesh:
    """Grow the ADVANCING layers of ``control`` from an oriented surface mesh.

    ``wall`` is a triangle/quadrilateral surface in three dimensions whose face
    orientation points in the growth direction. ``control.wall_scope`` must bind
    its cells; unselected cells are fixed adjacent surfaces along which rim
    columns slide. ``obstacles`` are further fixed surfaces the layers must not
    cross (their orientation is irrelevant).
    """
    if not isinstance(wall, CellMesh):
        raise TypeError("wall must be CellMesh.")
    if not isinstance(control, BoundaryLayerControl):
        raise TypeError("control must be BoundaryLayerControl.")
    if control.route is not BoundaryLayerRoute.ADVANCING:
        raise ValueError("prepare_boundary_layers realizes ADVANCING controls only.")
    if obstacles is not None and not isinstance(obstacles, CellMesh):
        raise TypeError("obstacles must be CellMesh or None.")
    policy_ = BoundaryLayerPolicy() if policy is None else policy
    if not isinstance(policy_, BoundaryLayerPolicy):
        raise TypeError("policy must be BoundaryLayerPolicy or None.")
    if wall.topological_dimension != 2 or wall.ambient_dimension != 3:
        raise ValueError("Boundary-layer walls are surfaces in three dimensions.")
    if wall_association is not None:
        if not isinstance(wall_association, GeometryAssociation):
            raise TypeError("wall_association must be GeometryAssociation or None.")
        vertex_set = wall.entity_set(0)
        if (
            wall_association.target_entity_set_id != vertex_set.entity_set_id
            or not np.array_equal(
                np.sort(np.asarray(wall_association.target_global_ids, dtype=np.int64)),
                np.sort(np.asarray(vertex_set.entity_ids, dtype=np.int64)),
            )
        ):
            raise ValueError("wall_association must bind the exact source wall vertices.")
    scope = control.wall_scope
    cells = wall.entity_set(2)
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.source_id != wall.mesh_id
        or scope.source_revision != wall.numeric_version
        or scope.entity_set_id != cells.entity_set_id
    ):
        raise ValueError("control.wall_scope must bind the cells of the wall mesh.")
    selected = np.asarray(scope.entity_ids, dtype=np.int64)
    identifiers = np.sort(np.asarray(cells.entity_ids, dtype=np.int64))
    if np.setdiff1d(selected, identifiers).size:
        raise ValueError("control.wall_scope selects cells absent from the wall mesh.")
    faces, arity = _mesh_faces(wall)
    grow = np.isin(identifiers, selected)
    if obstacles is None:
        obstacle_points = np.empty((0, 3))
        obstacle_triangles = np.empty((0, 3), dtype=np.int64)
    else:
        if obstacles.topological_dimension != 2 or obstacles.ambient_dimension != 3:
            raise ValueError("Obstacles are surfaces in three dimensions.")
        obstacle_faces, obstacle_arity = _mesh_faces(obstacles)
        obstacle_points = np.asarray(obstacles.coordinates, dtype=np.float64)
        obstacle_triangles, _ = _split_polygons(obstacle_faces, obstacle_arity)
    return _grow_boundary_layers(
        np.asarray(wall.coordinates, dtype=np.float64),
        faces,
        arity,
        grow,
        obstacle_points,
        obstacle_triangles,
        control,
        policy_,
        wall,
        wall_association,
    )


__all__ = [
    "BoundaryLayerEvidence",
    "BoundaryLayerMesh",
    "BoundaryLayerPolicy",
    "prepare_boundary_layers",
]
