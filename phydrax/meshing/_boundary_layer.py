#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native advancing boundary layers grown from triangle and quadrilateral walls.

The front is a column complex. Every wall vertex carries one column per smooth
sector (sectors split at convex ridges under the FAN corner policy), fan columns
interpolate across convex ridges, and corner patches close vertices where three
or more ridges meet. Columns point along the visibility-optimal direction, the
normalized minimum-norm point of the convex hull of the sector's face normals,
solved as a simplex-constrained QP by ``phydrax.optim``. Column heights follow
the explicit schedule, scaled by smooth concave curvature and by BVH-nearest
medial-crossing probes. Every layer is certified by Bernstein validity and by
exact-predicate intersection of its cell faces against earlier layers, the wall,
and the obstacles before it is accepted; collisions resolve only through the
control's explicit collision policy, and every rejection carries the offending
wall vertices and locations.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._bvh import bvh_nearest_items, bvh_overlap_pairs_host, prepare_bvh
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
from ..optim import (
    AcceleratedProximalGradient,
    OptimizationStatus,
    OptimizationTermination,
    proximal_minimize,
    SimplexIndicator,
)
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
# Outward face loops per cell kind (phydrax reference orientation).
_FACE_ROUTES = {
    "tetrahedron": ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
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
    """Execution bounds of native advancing-layer generation.

    ``proximity_fraction`` and ``curvature_fraction`` bound column heights by the
    medial crossing found with ``proximity_samples`` BVH-nearest probes per
    column and by the smooth concave radius of curvature. A foreign surface
    crosses a probe when it is nearer than ``1 - medial_tolerance`` times the
    distance to the column's own wall. Collision resolution retries at most
    ``maximum_collision_iterations`` times per layer (restarts for thickness
    reduction), scaling colliding columns by ``reduction_factor``.
    ``simplex_cap`` closes quadrilateral cap faces with transition pyramids so
    the cap is a triangle surface for simplex core fill.
    """

    visibility_iterations: int = eqx.field(static=True)
    proximity_samples: int = eqx.field(static=True)
    proximity_fraction: float = eqx.field(static=True)
    curvature_fraction: float = eqx.field(static=True)
    medial_tolerance: float = eqx.field(static=True)
    maximum_collision_iterations: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    simplex_cap: bool = eqx.field(static=True)
    validity: CellValidityPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        visibility_iterations: int = 2000,
        proximity_samples: int = 16,
        proximity_fraction: float = 0.8,
        curvature_fraction: float = 0.8,
        medial_tolerance: float = 0.05,
        maximum_collision_iterations: int = 8,
        reduction_factor: float = 0.7,
        simplex_cap: bool = True,
        validity: CellValidityPolicy | None = None,
    ):
        counts = (
            ("visibility_iterations", visibility_iterations),
            ("proximity_samples", proximity_samples),
            ("maximum_collision_iterations", maximum_collision_iterations),
        )
        for name, value in counts:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
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
        self.visibility_iterations = visibility_iterations
        self.proximity_samples = proximity_samples
        self.proximity_fraction = fractions["proximity_fraction"]
        self.curvature_fraction = fractions["curvature_fraction"]
        self.medial_tolerance = tolerance
        self.maximum_collision_iterations = maximum_collision_iterations
        self.reduction_factor = fractions["reduction_factor"]
        self.simplex_cap = simplex_cap
        self.validity = validity_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-policy",
                "visibility_iterations": visibility_iterations,
                "proximity_samples": proximity_samples,
                **fractions,
                "medial_tolerance": tolerance,
                "maximum_collision_iterations": maximum_collision_iterations,
                "simplex_cap": simplex_cap,
                "validity": validity_.policy_id,
            }
        )


class BoundaryLayerEvidence(StrictModule, NonTrainableState):
    """Measured layer realization and collision-resolution evidence.

    Thicknesses are measured as increments of the exact distance from each
    column's layer vertices to the wall surface (BVH-nearest point-triangle
    distances); per-layer statistics cover every column that carries the layer.
    """

    requested_thicknesses: tuple[float, ...] = eqx.field(static=True)
    achieved_thicknesses: Array
    minimum_thicknesses: Array
    maximum_thicknesses: Array
    achieved_growth_rates: Array
    column_count: int = eqx.field(static=True)
    fan_column_count: int = eqx.field(static=True)
    corner_patch_count: int = eqx.field(static=True)
    convex_ridge_count: int = eqx.field(static=True)
    concave_ridge_count: int = eqx.field(static=True)
    rim_vertex_count: int = eqx.field(static=True)
    minimum_visibility: float = eqx.field(static=True)
    maximum_stretch: float = eqx.field(static=True)
    unconverged_visibility_count: int = eqx.field(static=True)
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

    def __init__(self, **values):
        arrays = (
            "achieved_thicknesses",
            "minimum_thicknesses",
            "maximum_thicknesses",
            "achieved_growth_rates",
        )
        for name in arrays:
            setattr(self, name, jnp.asarray(np.asarray(values[name], dtype=np.float64)))
        self.requested_thicknesses = tuple(values["requested_thicknesses"])
        integers = (
            "column_count",
            "fan_column_count",
            "corner_patch_count",
            "convex_ridge_count",
            "concave_ridge_count",
            "rim_vertex_count",
            "unconverged_visibility_count",
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
                    name: array_tree_fingerprint(np.asarray(values[name]))
                    for name in arrays
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
    fill must keep it fixed. ``wall_vertices`` maps every
    input wall-mesh vertex to its layer-mesh vertex (``-1`` when unused), and
    ``layer_index`` gives each layer cell (global ID order) its schedule layer;
    transition pyramids closing quadrilateral cap faces carry the layer count.
    """

    mesh: CellMesh
    cap: CellMesh | None
    wall_vertices: Array
    cap_vertices: Array
    layer_index: Array
    validity: CellValidityCertificate
    evidence: BoundaryLayerEvidence
    control_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        cap: CellMesh | None,
        wall_vertices,
        cap_vertices,
        layer_index,
        validity: CellValidityCertificate,
        evidence: BoundaryLayerEvidence,
        /,
        *,
        control_id: str,
        policy_id: str,
    ):
        if not isinstance(mesh, CellMesh) or not isinstance(cap, (CellMesh, type(None))):
            raise TypeError("mesh must be CellMesh and cap CellMesh or None.")
        if not isinstance(validity, CellValidityCertificate):
            raise TypeError("validity must be CellValidityCertificate.")
        if not isinstance(evidence, BoundaryLayerEvidence):
            raise TypeError("evidence must be BoundaryLayerEvidence.")
        cap_ids = np.asarray(cap_vertices, dtype=np.int64)
        layers = np.asarray(layer_index, dtype=np.int32)
        cell_count = sum(block.cell_count for block in mesh.blocks)
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
        self.wall_vertices = jnp.asarray(np.asarray(wall_vertices, dtype=np.int64))
        self.cap_vertices = jnp.asarray(cap_ids)
        self.layer_index = jnp.asarray(layers)
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


def _half_edges(faces: np.ndarray, arity: np.ndarray, /):
    local = np.arange(4)
    valid = local[None, :] < arity[:, None]
    following = np.where(local[None, :] + 1 < arity[:, None], local[None, :] + 1, 0)
    start = faces[valid]
    end = np.take_along_axis(faces, following, axis=1)[valid]
    owner = np.nonzero(valid)[0]
    return start, end, owner


def _edge_table(faces: np.ndarray, arity: np.ndarray, /):
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


def _newell_normals(points: np.ndarray, faces: np.ndarray, arity: np.ndarray, /):
    local = np.arange(4)
    following = np.where(local[None, :] + 1 < arity[:, None], local[None, :] + 1, 0)
    valid = local[None, :] < arity[:, None]
    first = points[np.where(valid, faces, faces[:, :1])]
    second = points[
        np.take_along_axis(np.where(valid, faces, faces[:, :1]), following, 1)
    ]
    normal = np.sum(np.where(valid[..., None], np.cross(first, second), 0.0), axis=1)
    return _unit(normal)


# ---------------------------------------------------------------- visibility QP


def _hull_norm(weights, normals):
    point = weights @ normals
    return 0.5 * jnp.sum(point * point)


_VISIBILITY_METHOD = AcceleratedProximalGradient()


@functools.partial(jax.jit, static_argnames=("maximum_steps",))
def _visibility_weights(normals, *, maximum_steps: int):
    """Minimum-norm convex combination of each column's padded face normals."""
    termination = OptimizationTermination(
        absolute_optimality=1.0e-13,
        relative_optimality=1.0e-13,
        maximum_steps=maximum_steps,
    )

    def solve(block):
        count = block.shape[0]
        result = proximal_minimize(
            _hull_norm,
            jnp.full((count,), 1.0 / count, dtype=block.dtype),
            nonsmooth=SimplexIndicator(1.0),
            method=_VISIBILITY_METHOD,
            termination=termination,
            args=block,
        )
        return result.parameters, result.status

    return jax.vmap(solve)(normals)


def _padded_groups(groups: np.ndarray, members: np.ndarray, count: int, /):
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
    unconverged: int


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
    policy: BoundaryLayerPolicy,
    /,
) -> _Directions:
    count = sectors.sector_vertex.size
    faces = sectors.incidence_keys % sectors.face_count
    padded, _ = _padded_groups(sectors.incidence_sector, faces, count)
    normals = wall.normals[padded]
    weights, status = _visibility_weights(
        jnp.asarray(normals), maximum_steps=policy.visibility_iterations
    )
    weights = np.asarray(weights, dtype=np.float64)
    unconverged = int(
        np.count_nonzero(np.asarray(status) != int(OptimizationStatus.SUCCESS))
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
    return _Directions(direction, visibility, normals, unconverged)


def _sector_graph(wall: _Wall, sectors: _Sectors, /) -> tuple[np.ndarray, np.ndarray]:
    joined = wall.interior | wall.rim
    faces = wall.edge_faces[joined]
    face = np.where(wall.grow[faces[:, 0]], faces[:, 0], faces[:, 1])
    first = sectors.at(wall.edges[joined, 0], face)
    second = sectors.at(wall.edges[joined, 1], face)
    return first, second


def _smooth_directions(
    wall, sectors, direction, reference, normals, constraints, control
):
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


def _fan_lists(
    wall: _Wall,
    sectors: _Sectors,
    split: np.ndarray,
    direction: np.ndarray,
    feature_angle: float,
    /,
):
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
    /,
) -> _Front:
    sector_count = sectors.sector_vertex.size
    column_vertex = [sectors.sector_vertex]
    column_direction = [directions.direction]
    column_kind = [np.full((sector_count,), _SECTOR, dtype=np.int64)]
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
                next_column += count
            fans = fan_groups[key]
            ordered = fans if first == low else fans[::-1]
            lists[index, side] = [first, *(int(value) for value in ordered), second]
    column_direction_ = np.concatenate(column_direction)
    column_vertex_ = np.concatenate(column_vertex)
    strips = []
    for index in range(ridges.size):
        left = lists[index, 0]
        right = lists[index, 1]
        for position in range(len(left) - 1):
            strips.append(
                (left[position], left[position + 1], right[position + 1], right[position])
            )
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
    extra_vertex = []
    extra_direction = []
    for vertex, cycle in corners:
        center = next_column
        next_column += 1
        value, _ = _unit(np.sum(column_direction_[cycle], axis=0))
        extra_vertex.append(vertex)
        extra_direction.append(value)
        ring = np.asarray(cycle, dtype=np.int64)
        rows = np.stack((np.full(ring.shape, center), ring, np.roll(ring, -1)), axis=1)
        first = column_direction_[ring]
        second = column_direction_[np.roll(ring, -1)]
        # Corner fans face away from the vertex: normals along the center direction.
        if np.sum(np.cross(first - value, second - value) @ value) < 0.0:
            rows = rows[:, (0, 2, 1)]
        corner_rows.append(rows)
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
    return _Front(
        column_vertex_,
        column_direction_,
        np.concatenate(column_kind),
        faces,
        arity,
        kind,
        wall_face,
        int(np.count_nonzero(np.concatenate(column_kind) == _FAN)),
        len(corners),
    )


def _corner_patches(wall, ends, sector_a, sector_b, lists, /):
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


def _probe_stretch(wall: _Wall, front: _Front, /) -> np.ndarray:
    """Height factor that makes each column's local wall distance equal its level."""
    grow_triangles, _ = _split_polygons(wall.faces[wall.grow], wall.arity[wall.grow])
    incident_vertex = grow_triangles.reshape(-1)
    incident_triangle = np.repeat(np.arange(grow_triangles.shape[0]), 3)
    pairs = np.unique(np.stack((incident_vertex, incident_triangle), axis=1), axis=0)
    padded, _ = _padded_groups(pairs[:, 0], pairs[:, 1], wall.points.shape[0])
    local = padded[front.column_vertex]
    lengths = np.linalg.norm(
        wall.points[wall.edges[:, 0]] - wall.points[wall.edges[:, 1]], axis=1
    )
    shortest = np.full((wall.points.shape[0],), np.inf)
    np.minimum.at(shortest, wall.edges.reshape(-1), np.repeat(lengths, 2))
    epsilon = 1.0e-3 * shortest[front.column_vertex]
    probe = wall.points[front.column_vertex] + epsilon[:, None] * front.column_direction
    corners = wall.points[grow_triangles[local]]
    distance = point_triangle_distance(
        jnp.broadcast_to(jnp.asarray(probe)[:, None, :], corners.shape[:2] + (3,)),
        jnp.asarray(corners[:, :, 0]),
        jnp.asarray(corners[:, :, 1]),
        jnp.asarray(corners[:, :, 2]),
    )
    nearest = np.sqrt(np.min(np.asarray(distance.squared_distance), axis=1))
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


def _triangle_distance(triangles: np.ndarray):
    first = jnp.asarray(triangles[:, 0])
    second = jnp.asarray(triangles[:, 1])
    third = jnp.asarray(triangles[:, 2])

    def distance(point, items):
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
    grow_triangles, _ = _split_polygons(wall.faces[wall.grow], wall.arity[wall.grow])
    incident = np.unique(
        np.stack(
            (
                grow_triangles.reshape(-1),
                np.repeat(np.arange(grow_triangles.shape[0]), 3),
            ),
            1,
        ),
        axis=0,
    )
    padded, _ = _padded_groups(incident[:, 0], incident[:, 1], wall.points.shape[0])
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
    own_corners = wall.points[grow_triangles[padded[front.column_vertex[columns]]]]
    shape = probes.shape[:2] + own_corners.shape[1:2] + (3,)
    own = point_triangle_distance(
        jnp.asarray(np.broadcast_to(probes[:, :, None, :], shape)),
        *(
            jnp.asarray(np.broadcast_to(own_corners[:, None, :, corner], shape))
            for corner in range(3)
        ),
    )
    own_distance = np.sqrt(np.min(np.asarray(own.squared_distance), axis=2))
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


def _smooth_scale(scale: np.ndarray, graph, iterations: int, /) -> np.ndarray:
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


def _smooth_counts(counts: np.ndarray, graph, /) -> np.ndarray:
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


def _front_rows(front: _Front, columns: _Columns, /):
    """Front faces, splitting wall quads whose vertices carry different layer counts."""
    counts = columns.count[front.column_vertex]
    face_counts = np.where(front.faces >= 0, counts[np.maximum(front.faces, 0)], -1)
    uniform = np.all((face_counts == face_counts[:, :1]) | (front.faces < 0), axis=1)
    split = (front.kind == _WALL) & (front.arity == 4) & ~uniform
    keep = ~split
    quads = front.faces[split]
    # Split along the diagonal through the smallest wall vertex, matching wall triangulation.
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
    return rows, arity


def _layer_cells(rows, arity, bottom_ids, top_ids, /) -> dict[str, np.ndarray]:
    """Standard cells of one layer from front faces and their collapse pattern."""
    cells: dict[str, list[np.ndarray]] = {name: [] for name in _VOLUME_KINDS}
    safe = np.maximum(rows, 0)
    bottom = bottom_ids[safe]
    top = top_ids[safe]
    vertical = bottom == top
    triangle = arity == 3
    b, t, v = bottom[triangle][:, :3], top[triangle][:, :3], vertical[triangle][:, :3]
    collapsed = np.sum(v, axis=1)
    pinched = (b[:, 0] == b[:, 1]) & (b[:, 1] == b[:, 2])
    cells["prism"].append(np.concatenate((b, t), axis=1)[(collapsed == 0) & ~pinched])
    corner = (collapsed == 0) & pinched
    cells["tetrahedron"].append(
        np.stack((t[:, 0], t[:, 2], t[:, 1], b[:, 0]), axis=1)[corner]
    )
    for lead in range(3):
        i, j, k = lead, (lead + 1) % 3, (lead + 2) % 3
        one = (collapsed == 1) & v[:, i] & ~pinched
        cells["pyramid"].append(
            np.stack((b[:, j], t[:, j], t[:, k], b[:, k], b[:, i]), axis=1)[one]
        )
        two = (collapsed == 2) & ~v[:, k] & ~pinched
        cells["tetrahedron"].append(
            np.stack((b[:, i], b[:, j], b[:, k], t[:, k]), axis=1)[two]
        )
    quad = arity == 4
    b, t, v = bottom[quad], top[quad], vertical[quad]
    pair_v = v[:, 0] & v[:, 1]
    pair_w = v[:, 2] & v[:, 3]
    none = ~np.any(v, axis=1)
    pinched = (b[:, 0] == b[:, 1]) & (b[:, 2] == b[:, 3])
    cells["hexahedron"].append(np.concatenate((b, t), axis=1)[none & ~pinched])
    cells["prism"].append(
        np.stack((b[:, 0], t[:, 0], t[:, 1], b[:, 3], t[:, 3], t[:, 2]), axis=1)[
            none & pinched
        ]
    )
    cells["tetrahedron"].append(
        np.stack((b[:, 2], t[:, 2], t[:, 3], b[:, 0]), axis=1)[pinched & pair_v & ~pair_w]
    )
    cells["tetrahedron"].append(
        np.stack((b[:, 0], t[:, 0], t[:, 1], b[:, 2]), axis=1)[pinched & pair_w & ~pair_v]
    )
    cells["prism"].append(
        np.stack((b[:, 0], b[:, 3], t[:, 3], b[:, 1], b[:, 2], t[:, 2]), axis=1)[
            ~pinched & pair_v & ~pair_w
        ]
    )
    cells["prism"].append(
        np.stack((b[:, 2], b[:, 1], t[:, 1], b[:, 3], b[:, 0], t[:, 0]), axis=1)[
            ~pinched & pair_w & ~pair_v
        ]
    )
    half = (b[:, 0] == b[:, 1]) ^ (b[:, 2] == b[:, 3])
    irregular = half | (
        ~none & ~(pair_v & pair_w) & ~((pair_v ^ pair_w) & (np.sum(v, axis=1) == 2))
    )
    if np.any(irregular):
        raise ValueError("Quadrilateral front faces admit only paired side collapses.")
    return {
        name: np.concatenate(values).reshape(-1, _CELL_ARITY[name]).astype(np.int64)
        for name, values in cells.items()
    }


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
) -> tuple[dict[str, np.ndarray], int]:
    """Invalid or intersecting new cells (per kind masks) and certified hit count."""
    bad = {
        name: np.zeros((rows.shape[0],), dtype=np.bool_) for name, rows in cells.items()
    }
    if not any(rows.size for rows in cells.values()):
        return bad, 0
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
    bvh = prepare_bvh(np.min(corners, axis=1), np.max(corners, axis=1), dtype=jnp.float64)
    first, second = bvh_overlap_pairs_host(bvh, bvh, include_touching=True)
    keep = (first < second) & (first < new_count)
    first, second = first[keep], second[keep]
    if not first.size:
        return bad, 0
    hit, certain = _triangle_pairs_intersect(
        combined_points, triangles[first], triangles[second]
    )
    side = np.zeros((triangles.shape[0],), dtype=np.bool_)
    side[fixed_start : fixed_start + environment.side.size] = environment.side
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
        # Rim sheets slide on the adjacent surface: an intersection that does not
        # strictly straddle the surface plane is contact on it, not penetration.
        hit[rows[known_all & ~(above & below)]] = False
    colliding = hit | ~certain
    count = int(np.count_nonzero(hit & certain))
    for index in (first[colliding], second[colliding]):
        index = index[index < new_count]
        for kind_index, name in enumerate(_VOLUME_KINDS):
            selected = new_owners[index[new_kinds[index] == kind_index]]
            bad[name][selected] = True
    return bad, count


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
    points: np.ndarray
    detected: int
    iterations: int


def _initial_columns(
    wall: _Wall,
    front: _Front,
    stretch: np.ndarray,
    limits: np.ndarray,
    medial: np.ndarray,
    curvature_limited: np.ndarray,
    control: BoundaryLayerControl,
    graph,
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
    wall, front, predicted, medial, column_height, fraction, scale, merged_top
):
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
    graph,
    /,
) -> _Resolution:
    layers = columns.levels.size - 1
    base = wall.points.shape[0]
    detected = 0
    iterations = 0
    for _ in range(policy.maximum_collision_iterations + 1):
        accepted: list[dict[str, np.ndarray]] = []
        points = _positions(front, columns, wall.points)
        rows, arity = _front_rows(front, columns)
        for layer in range(layers):
            cells = _layer_cells(
                rows,
                arity,
                _level_ids(front, columns, base, layer),
                _level_ids(front, columns, base, layer + 1),
            )
            bad, hits = _certify_cells(points, cells, accepted, environment, policy)
            if any(np.any(mask) for mask in bad.values()):
                break
            accepted.append(cells)
        else:
            return _Resolution(columns, accepted, points, detected, iterations)
        detected += hits
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
                columns = _Columns(
                    columns.levels,
                    columns.stretch,
                    columns.scale,
                    count,
                    columns.merged_top,
                )
            case BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
                scale = columns.scale.copy()
                scale[vertices] *= policy.reduction_factor
                scale = _smooth_scale(scale, graph, control.smoothing_iterations)
                if np.any(scale[wall.wall_vertices] < control.minimum_thickness_fraction):
                    raise _failure(
                        MeshingFailureCategory.CONTROL_CONFLICT,
                        "Certified collisions persist at the minimum thickness fraction.",
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
    raise _failure(
        MeshingFailureCategory.CONTROL_CONFLICT,
        "Collision resolution did not converge within the iteration limit.",
        wall.wall_vertices,
        wall.points,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    rows, arity = _front_rows(front, columns)
    layers = columns.levels.size - 1
    ids = _level_ids(front, columns, base, layers)
    top = np.where(rows >= 0, ids[np.maximum(rows, 0)], -1)
    faces = []
    for row, size in zip(top, arity, strict=True):
        loop = [int(value) for value in row[:size]]
        distinct = []
        for value in loop:
            if not distinct or distinct[-1] != value:
                distinct.append(value)
        if len(distinct) > 1 and distinct[0] == distinct[-1]:
            distinct.pop()
        if len(distinct) >= 3:
            faces.append(distinct + [-1] * (4 - len(distinct)))
    cap = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    keys = np.sort(np.where(cap >= 0, cap, np.iinfo(np.int64).max), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    # Merged opposing fronts share their top faces, which become interior.
    single = counts[inverse.reshape(-1)] == 1
    cap = cap[single]
    return cap, np.where(cap[:, 3] < 0, 3, 4)


def _cap_pyramids(
    points: np.ndarray,
    cap: np.ndarray,
    arity: np.ndarray,
    accepted: list[dict[str, np.ndarray]],
    environment: _Environment,
    policy: BoundaryLayerPolicy,
    /,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    quads = cap[arity == 4]
    empty = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    if not quads.size:
        return points, empty, cap, arity
    corners = points[quads]
    normal, _ = _newell_normals(points, quads, np.full((quads.shape[0],), 4))
    edge = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
    height = 0.5 * np.mean(edge, axis=1)
    centroid = np.mean(corners, axis=1)
    apex_ids = points.shape[0] + np.arange(quads.shape[0])
    for _ in range(policy.maximum_collision_iterations + 1):
        apex = centroid + height[:, None] * normal
        candidate = np.concatenate((points, apex))
        # The cap quad faces away from the layer; the pyramid base must face its apex.
        pyramids = np.concatenate((quads[:, (0, 1, 2, 3)], apex_ids[:, None]), axis=1)
        cells = dict(empty)
        cells["pyramid"] = pyramids
        bad, _ = _certify_cells(candidate, cells, accepted, environment, policy)
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
            return candidate, cells, faces, np.full((faces.shape[0],), 3)
        height = np.where(bad["pyramid"], 0.5 * height, height)
    raise _failure(
        MeshingFailureCategory.CONTROL_CONFLICT,
        "Transition pyramids on the layer cap collide or are not certified valid.",
        quads[bad["pyramid"]].reshape(-1),
        points,
    )


# ---------------------------------------------------------------- evidence


def _measured_thicknesses(
    wall: _Wall, front: _Front, columns: _Columns, points: np.ndarray, /
):
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
    increments = np.diff(distance, axis=0)
    mean = np.full((layers,), np.nan)
    minimum = np.full((layers,), np.nan)
    maximum = np.full((layers,), np.nan)
    for layer in range(layers):
        values = increments[layer, carried[layer]]
        if values.size:
            mean[layer] = np.mean(values)
            minimum[layer] = np.min(values)
            maximum[layer] = np.max(values)
    return mean, minimum, maximum


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
    directions = _sector_directions(wall, sectors, control, policy)
    front = _build_front(wall, sectors, split, directions, control)
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
    )


def _column_limits(wall, front, stretch, obstacles, control, policy, /):
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


def _cap_mesh(points: np.ndarray, cap: np.ndarray, arity: np.ndarray, /):
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
    /,
) -> BoundaryLayerMesh:
    columns = resolution.columns
    layers = columns.levels.size - 1
    cap, cap_arity = _cap_faces(front, columns, wall.points.shape[0])
    all_points = resolution.points
    pyramids = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    if policy.simplex_cap:
        all_points, pyramids, cap, cap_arity = _cap_pyramids(
            all_points, cap, cap_arity, resolution.cells, environment, policy
        )
    cells = {}
    layer_index = []
    for name in _VOLUME_KINDS:
        rows = [values[name] for values in resolution.cells] + [pyramids[name]]
        cells[name] = np.concatenate(rows)
        # Global IDs follow kind order, then layer order within a kind.
        layer_index.extend(
            np.full((values.shape[0],), layer) for layer, values in enumerate(rows)
        )
    mesh, used = _cell_mesh(all_points, cells)
    remap = np.full((all_points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    certificate = certify_cell_geometry_validity(mesh, policy=policy.validity)
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
    mean, minimum, maximum = _measured_thicknesses(
        wall, front, columns, resolution.points
    )
    wall_scale = columns.scale[wall.wall_vertices]
    evidence = BoundaryLayerEvidence(
        requested_thicknesses=control.schedule.thicknesses,
        achieved_thicknesses=mean,
        minimum_thicknesses=minimum,
        maximum_thicknesses=maximum,
        achieved_growth_rates=mean[1:] / mean[:-1],
        column_count=front.column_vertex.size,
        fan_column_count=front.fan_column_count,
        corner_patch_count=front.corner_patch_count,
        convex_ridge_count=np.count_nonzero(wall.convex),
        concave_ridge_count=np.count_nonzero(wall.concave),
        rim_vertex_count=np.count_nonzero(wall.rim_vertex),
        minimum_visibility=np.min(directions.visibility),
        maximum_stretch=np.max(stretch),
        unconverged_visibility_count=directions.unconverged,
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
    return BoundaryLayerMesh(
        mesh,
        cap_mesh,
        remap[: wall.points.shape[0]],
        remap[cap_vertices],
        np.concatenate(layer_index),
        certificate,
        evidence,
        control_id=control.control_id,
        policy_id=policy.policy_id,
    )


def prepare_boundary_layers(
    wall: CellMesh,
    control: BoundaryLayerControl,
    /,
    *,
    obstacles: CellMesh | None = None,
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
    )


__all__ = [
    "BoundaryLayerEvidence",
    "BoundaryLayerMesh",
    "BoundaryLayerPolicy",
    "prepare_boundary_layers",
]
