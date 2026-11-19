#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Isosurface-stuffed tetrahedralization of analytic implicit profiles."""

from __future__ import annotations

import math
from typing import final, NamedTuple

import equinox as eqx
import jax
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    MeshcoreStatus,
    PlcBoundaryPolicy,
    TetMesh3D,
    TetMeshEditStatus,
    verify_volume,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityPolicy,
    ComputedCellValidity,
)
from ..discretization._cell_mesh import CellMesh, TetrahedralConnectivity
from ..geometry._global_embedding import ValidatedGlobalEmbedding
from ..geometry._interval_enclosure import prepare_interval_function
from ..geometry._interval_native import NativeIntervalProgram
from ..geometry._rational import (
    binary64_difference_magnitude,
    binary64_norm_lower,
    binary64_norm_upper,
)
from ..geometry._source_boundary_records import (
    SourceBoundaryDistanceWitness,
)
from ..geometry.implicit import AnalyticImplicitProfile
from ..geometry.implicit._tubular_projection_coverage import (
    TubularProjectionBoundaryWitness,
)
from ..geometry.surface import SurfaceMetadata, SurfaceModel
from ..typing import checked
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from ._certification_schedule import MeshCertificationSchedule
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from ._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from ._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from ._result import MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import SizeControlStrength, UniformSizeControl
from ._volume_generation import _shape_targets_gated, NativeVolumeSchedule


_FACE_CORNERS = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
# Lattice repair: each rejected stuffing shrinks the spacing by this factor.
_STUFFING_REFINEMENT_FACTOR = 0.8
_STUFFING_REFINEMENTS = 7
# Largest stuffed boundary-triangle diameter per lattice spacing that the
# first spacing provides for: warping moves a lattice vertex by at most 0.36
# spacings, and measured stuffed surfaces stay below 1.42.
_STUFFED_DIAMETER_PER_SPACING = 1.5
# The first lattice aims at this fraction of the fidelity tolerance.
_FIRST_LATTICE_FIDELITY_FRACTION = 0.95
# Coarse lattice cells span at most this fraction of the boundary's smallest
# extent.
_GRADING_EXTENT_FRACTION = 0.125
# Stuffed cells skip the native quality pass when their binary64 shape
# estimates clear the schedule by these margins.
_SCHEDULE_RATIO_MARGIN = 0.99
_SCHEDULE_DETERMINANT_MARGIN = 2.0
_SCHEDULE_DIHEDRAL_MARGIN = 0.5


def _failure(
    category: MeshingFailureCategory, message: str, stage: str, /
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_implicit_stuffing",
        stage=stage,
    )


def _scope_matches(
    candidate: MeshingScope, profile: AnalyticImplicitProfile, dimension: int, /
) -> bool:
    return (
        candidate.source_id == profile.source_id
        and candidate.source_revision == profile.source_revision
        and candidate.entity_kind is MeshingEntityKind.GEOMETRY
        and candidate.entity_dimension == dimension
        and candidate.entity_set_id == profile.scope_entity_set_id(dimension)
        and np.array_equal(
            np.asarray(candidate.entity_ids, dtype=np.int64),
            np.asarray(profile.scope_entity_ids(dimension), dtype=np.int64),
        )
    )


def _support_issues(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[str, ...]:
    issues: list[str] = []
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"tetrahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("affine tetrahedra in ambient dimension three")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("the simplex volume-fill strategy")
    if not _scope_matches(specification.boundary_scope, profile, 2):
        issues.append("the exact profile-owned zero-set boundary scope")
    if len(profile.region_ids) != 1 or profile.region_ids != ("interior",):
        issues.append("one source-owned analytic interior region")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append("nonuniform volume size controls")
        elif not _scope_matches(control.scope, profile, 2):
            issues.append("size controls outside the exact profile zero-set scope")
        elif control.maximum_growth_rate is not None:
            issues.append("maximum-growth-rate controls without an authored size field")
    if specification.protected_features:
        issues.append("protected features on the implicit volume boundary")
    if specification.region_controls:
        issues.append("independent region controls on the profile-owned interior")
    if specification.region_seeds:
        issues.append("user region seeds outside the profile-owned interior witness")
    if specification.hole_seeds:
        issues.append("additional void regions not present in the analytic profile")
    if specification.patch_controls:
        issues.append("patch controls on the derived stuffed boundary")
    if specification.periodic_constraints:
        issues.append("periodic constraints on the analytic profile domain")
    if specification.layer_controls:
        issues.append("boundary layers through the layer-core route")
    return tuple(sorted(set(issues)))


def _stuffing_bounds(profile: AnalyticImplicitProfile) -> np.ndarray:
    # The stuffing lattice pads this box by one coarse cell on each side and
    # refuses a solid that reaches the padding.
    bounds = np.array(profile.boundary_bounds, dtype=np.float64, copy=True)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] <= bounds[0])
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The analytic profile boundary must have finite bounds of positive extent.",
            "source_inspection",
        )
    return bounds


def _cover_radius(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[float, float]:
    requested = profile.tube_radius
    if specification.maximum_deviation is not None:
        requested = min(requested, float(specification.maximum_deviation))
    if not math.isfinite(requested) or requested <= 0.0:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Stuffed analytic volume construction requires a positive physical fidelity tolerance.",
            "source_inspection",
        )
    boundary_size = min(
        (
            control.target_size
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        default=profile.reach_lower,
    )
    cover_radius = min(requested / 4.0, boundary_size / 2.0)
    if not math.isfinite(cover_radius) or cover_radius <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The analytic profile must admit a positive representable physical cover radius.",
            "source_inspection",
        )
    return requested, cover_radius


def _topology_matches(
    profile: AnalyticImplicitProfile,
    oriented_facets: np.ndarray,
    /,
) -> bool:
    """Whether oriented triangles close up with the profile's boundary topology.

    Every directed edge must occur once and its reverse once (a watertight,
    consistently oriented, edge-manifold surface), with the profile's number
    of edge-connected components and Euler characteristic.
    """
    facets = np.asarray(oriented_facets, dtype=np.int64)
    if facets.ndim != 2 or facets.shape[1] != 3 or facets.shape[0] == 0:
        return False
    vertices, inverse = np.unique(facets, return_inverse=True)
    local = inverse.reshape(facets.shape)
    count = vertices.size
    tails = local.reshape((-1,))
    heads = local[:, [1, 2, 0]].reshape((-1,))
    directed = tails * count + heads
    order = np.argsort(directed)
    ordered = directed[order]
    if np.any(tails == heads) or np.any(ordered[1:] == ordered[:-1]):
        return False
    position = np.minimum(
        np.searchsorted(ordered, heads * count + tails), ordered.size - 1
    )
    if np.any(ordered[position] != heads * count + tails):
        return False
    face_count = facets.shape[0]
    halfedge_faces = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    adjacency = coo_matrix(
        (
            np.ones((halfedge_faces.size,), dtype=np.int8),
            (halfedge_faces, halfedge_faces[order[position]]),
        ),
        shape=(face_count, face_count),
    )
    components = connected_components(adjacency, directed=False, return_labels=False)
    euler = count - directed.size // 2 + face_count
    return (
        components == profile.component_count
        and euler == profile.boundary_euler_characteristic
    )


def _boundary_faces(cells: np.ndarray, vertex_count: int, /) -> np.ndarray:
    """Outward faces of positive tetrahedra that no other cell shares.

    Faces are enumerated per local face (one opposite each corner), all cells
    for each; the boundary keeps that order.
    """
    faces = np.asarray(cells, dtype=np.int64)[:, np.asarray(_FACE_CORNERS)]
    faces = faces.transpose((1, 0, 2)).reshape((-1, 3))
    ordered = np.sort(faces, axis=1)
    if vertex_count < 2**21:
        keys = (ordered[:, 0] * vertex_count + ordered[:, 1]) * vertex_count
        keys += ordered[:, 2]
        order = np.argsort(keys)
        sorted_keys = keys[order]
    else:
        # Three packed vertex indices would overflow int64.
        order = np.lexsort((ordered[:, 2], ordered[:, 1], ordered[:, 0]))
        sorted_keys = ordered[order]
    distinct = np.ones((faces.shape[0] + 1,), dtype=np.bool_)
    if sorted_keys.ndim == 1:
        distinct[1:-1] = sorted_keys[1:] != sorted_keys[:-1]
    else:
        distinct[1:-1] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    single = np.empty((faces.shape[0],), dtype=np.bool_)
    single[order] = distinct[:-1] & distinct[1:]
    return faces[single]


def _meets_schedule(
    points: np.ndarray, tetrahedra: np.ndarray, schedule: NativeVolumeSchedule, /
) -> bool:
    """Whether every cell meets the schedule's shape targets with margin.

    Binary64 estimates of the radius-edge ratio, the smallest dihedral angle
    and the relative determinant (as the native quality pass measures them)
    with a margin far above their rounding error: a proposal, which the
    certified compliance of `execute()` checks again.
    """
    corners = np.asarray(points, dtype=np.float64)[np.asarray(tetrahedra, dtype=np.int64)]
    origin = corners[:, 0]
    a, b, c = (corners[:, k] - origin for k in (1, 2, 3))
    bc, ca, ab = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    determinant = np.einsum("ij,ij->i", a, bc)
    if np.any(~(determinant > 0.0)):
        return False
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64)
    lengths = np.linalg.norm(corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]], axis=2)
    if schedule.radius_edge_bound is not None:
        center = (
            np.einsum("ij,ij->i", a, a)[:, None] * bc
            + np.einsum("ij,ij->i", b, b)[:, None] * ca
            + np.einsum("ij,ij->i", c, c)[:, None] * ab
        ) / (2.0 * determinant[:, None])
        ratio = np.linalg.norm(center, axis=1) / np.min(lengths, axis=1)
        if np.max(ratio) > _SCHEDULE_RATIO_MARGIN * schedule.radius_edge_bound:
            return False
    # Products of the three edge lengths at each corner.
    corner_edges = np.asarray(
        ((0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)), dtype=np.int64
    )
    products = np.max(np.prod(lengths[:, corner_edges], axis=2), axis=1)
    floor = max(schedule.minimum_relative_determinant, 2.0**-22)
    if np.min(determinant / products) <= _SCHEDULE_DETERMINANT_MARGIN * floor:
        return False
    # Outward face normals; the interior dihedral angle of two faces has
    # cosine -n_i . n_j.
    face_normals = np.stack(
        [
            np.cross(corners[:, j] - corners[:, i], corners[:, k] - corners[:, i])
            for i, j, k in _FACE_CORNERS
        ],
        axis=1,
    )
    face_normals /= np.linalg.norm(face_normals, axis=2, keepdims=True)
    cosine = -np.einsum(
        "ijk,ijk->ij", face_normals[:, pairs[:, 0]], face_normals[:, pairs[:, 1]]
    )
    smallest = math.degrees(math.acos(min(float(np.max(cosine)), 1.0)))
    return smallest >= schedule.minimum_dihedral_degrees + _SCHEDULE_DIHEDRAL_MARGIN


def _tetra_quality(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    center_bounds: np.ndarray,
    /,
) -> tuple[np.ndarray, float]:
    """Radius-edge upper bounds per tetrahedron and a minimum-dihedral lower bound.

    Every rounded operation is directed: center-to-vertex and edge distances use
    the binary64 norm bounds, and face normals and their cosines are interval
    products, sums, and quotients. The dihedral angle increases with the cosine
    of its face normals, so the smallest certified cosine lower bound gives one
    rigorous angle bound.
    """
    from ..geometry._interval_enclosure import (
        interval_add,
        interval_divide,
        interval_multiply,
        interval_sqrt,
        interval_square,
        interval_subtract,
    )

    vertices = np.asarray(points, dtype=np.float64)[
        np.asarray(tetrahedra, dtype=np.int64)
    ]
    centers = np.asarray(center_bounds, dtype=np.float64)
    center_lower, center_upper = centers[:, 0], centers[:, 1]
    # The farthest center-box point from a vertex is a box corner: per axis the
    # larger of the gaps to both box endpoints.
    center_gap = np.maximum(
        binary64_difference_magnitude(center_lower[:, None, :], vertices)[1],
        binary64_difference_magnitude(center_upper[:, None, :], vertices)[1],
    )
    radius_upper = np.max(binary64_norm_upper(center_gap), axis=1)
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
    )
    edge_lower = np.min(
        binary64_norm_lower(
            binary64_difference_magnitude(
                vertices[:, edge_pairs[:, 0]], vertices[:, edge_pairs[:, 1]]
            )[0]
        ),
        axis=1,
    )
    if np.any(edge_lower <= 0.0) or np.any(~np.isfinite(radius_upper)):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron edge and circumradius bounds must remain finite and positive.",
            "specification_compliance",
        )
    radius_edge = np.nextafter(radius_upper / edge_lower, np.inf)
    face_vertices = vertices[:, np.asarray(_FACE_CORNERS, dtype=np.int64)]
    origins = (face_vertices[:, :, 0], face_vertices[:, :, 0])
    first_edges = _float_interval(
        interval_subtract((face_vertices[:, :, 1], face_vertices[:, :, 1]), origins)
    )
    second_edges = _float_interval(
        interval_subtract((face_vertices[:, :, 2], face_vertices[:, :, 2]), origins)
    )
    normal_scale = np.max(
        np.abs(np.concatenate((*first_edges, *second_edges), axis=2)), axis=2
    )
    if np.any(~np.isfinite(normal_scale)) or np.any(normal_scale <= 0.0):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron face-normal quality is unresolved at binary64 scale.",
            "specification_compliance",
        )
    # A power-of-two scale keeps the normal products in range; interval
    # products round any scaled endpoint that is not exact outward.
    scale = np.ldexp(1.0, -np.frexp(normal_scale)[1])[:, :, None]
    first_edges = _float_interval(interval_multiply(first_edges, (scale, scale)))
    second_edges = _float_interval(interval_multiply(second_edges, (scale, scale)))
    following = np.asarray((1, 2, 0), dtype=np.int64)
    preceding = np.asarray((2, 0, 1), dtype=np.int64)
    face_normals = _float_interval(
        interval_subtract(
            interval_multiply(
                (first_edges[0][..., following], first_edges[1][..., following]),
                (second_edges[0][..., preceding], second_edges[1][..., preceding]),
            ),
            interval_multiply(
                (first_edges[0][..., preceding], first_edges[1][..., preceding]),
                (second_edges[0][..., following], second_edges[1][..., following]),
            ),
        )
    )
    normal_squares = _float_interval(interval_square(face_normals))
    normal_lengths = _float_interval(
        interval_sqrt(
            interval_add(
                interval_add(
                    (normal_squares[0][..., 0], normal_squares[1][..., 0]),
                    (normal_squares[0][..., 1], normal_squares[1][..., 1]),
                ),
                (normal_squares[0][..., 2], normal_squares[1][..., 2]),
            )
        )
    )
    if np.any(~np.isfinite(normal_lengths[1])) or np.any(normal_lengths[0] <= 0.0):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron dihedral quality is unresolved at binary64 normal precision.",
            "specification_compliance",
        )
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64)
    products = _float_interval(
        interval_multiply(
            (face_normals[0][:, pairs[:, 0]], face_normals[1][:, pairs[:, 0]]),
            (face_normals[0][:, pairs[:, 1]], face_normals[1][:, pairs[:, 1]]),
        )
    )
    dots = interval_add(
        interval_add(
            (products[0][..., 0], products[1][..., 0]),
            (products[0][..., 1], products[1][..., 1]),
        ),
        (products[0][..., 2], products[1][..., 2]),
    )
    cosine_lower = _float_interval(
        interval_divide(
            dots,
            interval_multiply(
                (normal_lengths[0][:, pairs[:, 0]], normal_lengths[1][:, pairs[:, 0]]),
                (normal_lengths[0][:, pairs[:, 1]], normal_lengths[1][:, pairs[:, 1]]),
            ),
        )
    )[0]
    if cosine_lower.size == 0:
        return radius_edge, 180.0
    return radius_edge, _dihedral_degrees_lower(float(np.min(cosine_lower)))


def _float_interval(value: tuple[object, object], /) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(value[0], dtype=np.float64),
        np.asarray(value[1], dtype=np.float64),
    )


def _dihedral_degrees_lower(cosine_lower: float, /) -> float:
    """Binary64 lower bound of ``180 - degrees(acos(c))`` over ``c >= cosine_lower``.

    ``acos(c) = atan2(sqrt(1 - c**2), c)`` is evaluated in outward mpmath interval
    arithmetic, and the float candidate is checked against the enclosure exactly.
    """
    from ..geometry.brep._intersection_curve import interval_context

    context = interval_context(40)
    cosine = context.mpf(min(max(cosine_lower, -1.0), 1.0))
    degrees = 180 - 180 * context.atan2(context.sqrt(1 - cosine * cosine), cosine) / (
        context.pi
    )
    bound = float(degrees.a)
    while not context.mpf(bound) <= degrees.a:
        bound = math.nextafter(bound, -math.inf)
    return max(bound, 0.0)


class StuffedImplicitConstruction(NamedTuple):
    """The constructed tetrahedral state of the stuffing route, before publication.

    ``points`` rows are the vertices; positively oriented ``tetrahedra`` and
    outward ``boundary_faces`` (one per source boundary facet, in stuffing
    order) index those rows. ``vertex_ids`` and ``cell_ids`` are the global
    identities `PreparedStuffedImplicitVolume.execute` publishes for the rows:
    the row numbers of the stuffed lattice, or the identities the native
    fixed-boundary quality pass carries from them.
    """

    points: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    vertex_ids: np.ndarray
    cell_ids: np.ndarray


def _rows_of(identifiers: np.ndarray, query: np.ndarray, /) -> np.ndarray | None:
    """Rows of unique ``identifiers`` holding each ``query`` value, or None."""
    order = np.argsort(identifiers, kind="stable")
    ordered = identifiers[order]
    position = np.minimum(np.searchsorted(ordered, query), ordered.size - 1)
    if ordered.size == 0 or np.any(ordered[position] != query):
        return None
    return order[position]


def _native_fixed_boundary_quality_pass(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    boundary_faces: np.ndarray,
    schedule: NativeVolumeSchedule,
    /,
) -> StuffedImplicitConstruction:
    """Improve a source-constrained tetrahedral state with its boundary fixed.

    ``points`` and ``tetrahedra`` are the primal arrays; vertex and cell
    identities are their row numbers, which the native state carries.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    cell_ids = np.arange(tetrahedra.shape[0], dtype=np.int64)
    vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    facets = np.asarray(boundary_faces, dtype=np.int32).reshape((-1, 3))
    if (
        facets.shape[0] == 0
        or facets.size
        and (np.any(facets < 0) or np.any(facets >= coordinates.shape[0]))
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Analytic volume boundary facets must index the current target mesh.",
            "implicit_quality_preparation",
        )
    if facets.shape[0] > np.iinfo(np.int32).max:
        raise OverflowError("Analytic volume facet labels exceed native int32 indices.")
    face_sources = np.arange(facets.shape[0], dtype=np.int32)
    face_orientations = np.ones((facets.shape[0],), dtype=np.int8)
    segments = np.empty((0, 2), dtype=np.int32)
    segment_sources = np.empty((0,), dtype=np.int32)
    regions = np.zeros((tetrahedra.shape[0],), dtype=np.int32)
    source_boundary_ids = vertex_ids[facets]
    state = TetMesh3D(
        coordinates,
        tetrahedra,
        regions,
        facets,
        face_sources,
        segments,
        segment_sources,
        boundary_policy=PlcBoundaryPolicy.FIXED,
        vertex_ids=np.asarray(vertex_ids, dtype=np.uint64),
        cell_ids=np.asarray(cell_ids, dtype=np.uint64),
        face_orientations=face_orientations,
    )
    try:
        refinement = state.refine(radius_edge_bound=schedule.radius_edge_bound)
        if refinement.edit_status is not TetMeshEditStatus.APPLIED:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native fixed-boundary radius refinement returned edit status {refinement.edit_status.name}.",
                "implicit_quality_refinement",
            )
        if refinement.status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native fixed-boundary radius refinement failed with {refinement.status.name}.",
                "implicit_quality_refinement",
            )
        improvement = state.improve(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        if improvement.edit_status is not TetMeshEditStatus.APPLIED:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native joint quality improvement returned edit status {improvement.edit_status.name}.",
                "implicit_quality_improvement",
            )
        if improvement.status is MeshcoreStatus.UNRESOLVED:
            unmet = state.unmet()
            residuals = tuple(
                (int(criterion), float(value))
                for criterion, value in zip(
                    unmet.criteria.tolist(), unmet.values.tolist(), strict=True
                )
            )
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Native fixed-boundary quality improvement remains unresolved "
                f"for {improvement.unmet_cells} cell(s): {residuals}.",
                "implicit_quality_improvement",
            )
        if improvement.status is not MeshcoreStatus.OK or improvement.unmet_cells != 0:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Native joint quality improvement returned inconsistent success status "
                f"{improvement.status.name} with {improvement.unmet_cells} unmet cell(s).",
                "implicit_quality_improvement",
            )
        arrays = state.arrays()
    finally:
        state.close()

    native_vertex_ids = np.asarray(arrays.vertex_ids, dtype=np.uint64)
    native_cell_ids = np.asarray(arrays.cell_ids, dtype=np.uint64)
    if np.any(native_vertex_ids > np.iinfo(np.int64).max) or np.any(
        native_cell_ids > np.iinfo(np.int64).max
    ):
        raise OverflowError(
            "Native quality identities exceed canonical CellMesh int64 IDs."
        )
    if (
        arrays.tetrahedron_regions.shape != (arrays.tetrahedra.shape[0],)
        or np.any(arrays.tetrahedron_regions != 0)
        or arrays.face_sources.shape != (facets.shape[0],)
        or arrays.faces.shape != facets.shape
        or arrays.segments.shape != (0, 2)
        or arrays.segment_sources.size != 0
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native joint quality edits changed target regions or fixed source-constraint banks.",
            "implicit_quality_materialization",
        )
    # Every source keeps exactly its facet: the sources are a permutation of
    # the input facet rows, and each carries the same vertex identities.
    actual_sources = np.asarray(arrays.face_sources, dtype=np.int64)
    by_source = np.argsort(actual_sources, kind="stable")
    actual_faces = native_vertex_ids.astype(np.int64)[
        np.asarray(arrays.faces, dtype=np.int64)[by_source]
    ]
    if not np.array_equal(
        actual_sources[by_source], np.arange(facets.shape[0], dtype=np.int64)
    ) or not np.array_equal(
        np.sort(actual_faces, axis=1), np.sort(source_boundary_ids, axis=1)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native joint quality edits changed a fixed source-boundary facet.",
            "implicit_quality_materialization",
        )
    output_vertex_ids = native_vertex_ids.astype(np.int64)
    output_boundary_faces = _rows_of(output_vertex_ids, source_boundary_ids)
    if output_boundary_faces is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native fixed-boundary improvement removed an authored boundary vertex.",
            "implicit_quality_materialization",
        )
    return StuffedImplicitConstruction(
        np.asarray(arrays.points, dtype=np.float64),
        np.asarray(arrays.tetrahedra, dtype=np.int32),
        output_boundary_faces.astype(np.int32),
        output_vertex_ids,
        native_cell_ids.astype(np.int64),
    )


def _published_mesh(
    construction: StuffedImplicitConstruction, numeric_version: str, /
) -> tuple[CellMesh, np.ndarray, np.ndarray]:
    """The canonical mesh of a construction, its boundary face IDs and rows.

    The published boundary must be exactly the construction's boundary
    facets. Returns the mesh, the global IDs of its boundary faces in mesh
    order, and the boundary facets as mesh vertex rows in construction order.
    """
    output = CellMesh.from_tetrahedra(
        construction.points,
        construction.tetrahedra,
        vertex_global_ids=construction.vertex_ids,
        cell_global_ids=construction.cell_ids,
        numeric_version=numeric_version,
    )
    connectivity = output.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native joint quality output lost tetrahedral connectivity.",
            "implicit_quality_materialization",
        )
    output_vertex_ids = np.asarray(output.vertex_global_ids, dtype=np.int64)
    source_boundary_ids = np.asarray(construction.vertex_ids, dtype=np.int64)[
        np.asarray(construction.boundary_faces, dtype=np.int64)
    ]
    output_boundary_faces = _rows_of(output_vertex_ids, source_boundary_ids)
    if output_boundary_faces is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native fixed-boundary improvement removed an authored boundary vertex.",
            "implicit_quality_materialization",
        )
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    output_faces = np.asarray(connectivity.faces, dtype=np.int64)
    expected_boundary = np.unique(np.sort(source_boundary_ids, axis=1), axis=0)
    actual_boundary = np.unique(
        np.sort(output_vertex_ids[output_faces[boundary_mask]], axis=1), axis=0
    )
    if not np.array_equal(actual_boundary, expected_boundary):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native fixed-boundary improvement changed the embedded target boundary.",
            "implicit_quality_materialization",
        )
    face_ids = np.asarray(output.entity_set(2).entity_ids, dtype=np.int64)
    return (
        output,
        face_ids[boundary_mask],
        output_boundary_faces.astype(np.int32),
    )


def _organization(
    profile: AnalyticImplicitProfile,
    contract: SpatialCoordinateContract,
    mesh: CellMesh,
    boundary_face_ids: np.ndarray,
    boundary_faces: np.ndarray,
    deviation: float,
    /,
) -> NativeMeshOrganization:
    from .providers._native_volume import _faces_connected, _mesh_scope

    region_name = profile.region_ids[0]
    all_vertices = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    all_edges = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    all_faces = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    all_cells = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    boundary_face_mask = np.isin(all_faces, boundary_face_ids)
    boundary_vertex_rows = np.unique(
        np.asarray(boundary_faces, dtype=np.int64).reshape((-1,))
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    boundary_vertex_mask = np.isin(all_vertices, vertex_ids[boundary_vertex_rows])
    boundary_edge_keys = {
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
        for face in np.asarray(boundary_faces, dtype=np.int64)
        for index in range(3)
    }
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "A mesh created from restricted tetrahedra must retain tetrahedral connectivity.",
            "implicit_mesh_assembly",
        )
    edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    boundary_edge_mask = np.asarray(
        [
            tuple(sorted(int(value) for value in edge)) in boundary_edge_keys
            for edge in edge_vertices
        ],
        dtype=np.bool_,
    )
    surface_key = profile.source_entity_key_for_scope(
        2, int(profile.scope_entity_ids(2)[0])
    )
    region_key = profile.source_entity_key_for_scope(
        3, int(profile.scope_entity_ids(3)[0])
    )

    def association(
        dimension: int,
        identifiers: np.ndarray,
        boundary_mask: np.ndarray,
        /,
        *,
        exact: bool,
    ) -> GeometryAssociation:
        keys = tuple(
            surface_key if boundary else region_key for boundary in boundary_mask.tolist()
        )
        return GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            profile.source_id,
            profile.source_revision,
            mesh.entity_set(dimension).entity_set_id,
            identifiers,
            tuple(key[0] for key in keys),
            np.where(boundary_mask, deviation, 0.0),
            exact=exact,
            source_roles=tuple(
                GeometrySourceEntityRole.FACET
                if boundary
                else GeometrySourceEntityRole.REGION
                for boundary in boundary_mask.tolist()
            ),
            source_occurrence_paths=tuple(key[1] for key in keys),
        )

    associations = (
        association(0, all_vertices, boundary_vertex_mask, exact=False),
        association(1, all_edges, boundary_edge_mask, exact=False),
        association(2, all_faces, boundary_face_mask, exact=False),
        association(
            3, all_cells, np.zeros((all_cells.size,), dtype=np.bool_), exact=True
        ),
    )
    cell_scope = _mesh_scope(mesh, 3, all_cells)
    boundary_scope = _mesh_scope(mesh, 2, boundary_face_ids)
    zone = MeshZone(region_name, MeshZoneRole.REGION, cell_scope)
    patch = MeshPatch(
        "implicit-zero-set",
        boundary_scope,
        connected=_faces_connected(np.asarray(boundary_faces, dtype=np.int64)),
        adjacent_zone_ids=(zone.zone_id,),
        source_adjacent_region_ids=(region_key[0], None),
        source_adjacent_region_occurrence_paths=(region_key[1], None),
    )
    used = np.unique(np.asarray(boundary_faces, dtype=np.int64).reshape((-1,)))
    inverse = np.full((np.asarray(mesh.coordinates).shape[0],), -1, dtype=np.int64)
    inverse[used] = np.arange(used.size, dtype=np.int64)
    surface_mesh = CellMesh.from_triangles(
        np.asarray(mesh.coordinates, dtype=np.float64)[used],
        inverse[np.asarray(boundary_faces, dtype=np.int64)].astype(np.int32, copy=False),
        vertex_global_ids=vertex_ids[used],
        numeric_version=profile.source_revision,
    )
    boundary_model = SurfaceModel(
        surface_mesh,
        SurfaceMetadata(
            source_id=profile.source_id,
            source_revision=profile.source_revision,
            coordinate_contract=contract,
            provenance=(
                "geometry-owned-stuffed-implicit-volume-boundary",
                profile.profile_id,
                profile.domain_id,
            ),
            cell_tags=("implicit-zero-set",) * np.asarray(boundary_faces).shape[0],
        ),
    )
    return NativeMeshOrganization(
        boundary=boundary_model,
        patches=(patch,),
        zones=(zone,),
        labels=(
            MeshLabel(f"region:{region_name}", cell_scope),
            MeshLabel("implicit-zero-set", boundary_scope),
        ),
        associations=associations,
    )


def _compliance(
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    mesh: CellMesh,
    boundary_faces: np.ndarray,
    fidelity_tolerance: float,
    fidelity_bound: float,
    /,
) -> MeshingComplianceReport:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    # Recompute center bounds through the native exact Delaunay route before
    # compliance; this function's caller has already checked every cell.
    from .._meshcore import restricted_centers_3d

    centers, statuses = restricted_centers_3d(coordinates, cells.astype(np.int32))
    if np.any(np.asarray(statuses) != int(MeshcoreStatus.OK)):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Published restricted tetrahedra require successful native circumcenter bounds.",
            "specification_compliance",
        )
    radius_edge, minimum_dihedral = _tetra_quality(coordinates, cells, centers)
    maximum_radius_edge = float(np.max(radius_edge, initial=0.0))
    requested: list[tuple[str, float]] = []
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", schedule.radius_edge_bound))
    requested.extend(
        (
            ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
            ("source_fidelity_tolerance", fidelity_tolerance),
        )
    )
    achieved: list[tuple[str, float]] = [
        ("maximum_radius_edge_ratio", maximum_radius_edge),
        ("minimum_dihedral_degrees", minimum_dihedral),
        ("source_fidelity_bound", fidelity_bound),
    ]
    issues: list[str] = []
    if schedule.radius_edge_bound is None and not np.isfinite(maximum_radius_edge):
        issues.append("nonrepresentable_radius_edge")
    if schedule.minimum_dihedral_degrees == 0.0 and not np.isfinite(minimum_dihedral):
        issues.append("nonrepresentable_dihedral")
    if (
        schedule.radius_edge_bound is not None
        and maximum_radius_edge > schedule.radius_edge_bound
    ):
        issues.append("radius_edge_bound")
    if minimum_dihedral < schedule.minimum_dihedral_degrees:
        issues.append("minimum_dihedral_degrees")
    if fidelity_bound > fidelity_tolerance:
        issues.append("source_fidelity")
    edge_keys = {
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
        for face in np.asarray(boundary_faces, dtype=np.int64)
        for index in range(3)
    }
    edges = np.asarray(sorted(edge_keys), dtype=np.int64)
    lengths = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(f"size:{control.control_id}:unsupported")
            continue
        key = f"size:{control.control_id}"
        for statistic in specification.size_compliance.target_statistics:
            quantile = 0.5 if statistic == "p50" else 0.95
            value = float(np.quantile(lengths, quantile))
            name = f"{key}:{statistic}"
            requested.append((name, control.target_size))
            achieved.append((name, value))
            allowed = (
                control.target_size
                * (1.0 + specification.size_compliance.relative_tolerance)
                + specification.size_compliance.absolute_tolerance
            )
            if control.strength is SizeControlStrength.HARD and value > allowed:
                issues.append(name)
        if control.minimum_size is not None:
            value = float(np.min(lengths))
            name = f"{key}:minimum"
            requested.append((name, control.minimum_size))
            achieved.append((name, value))
            if (
                control.strength is SizeControlStrength.HARD
                and value < control.minimum_size
            ):
                issues.append(name)
        if control.maximum_size is not None:
            value = float(np.max(lengths))
            name = f"{key}:maximum"
            requested.append((name, control.maximum_size))
            achieved.append((name, value))
            if (
                control.strength is SizeControlStrength.HARD
                and value > control.maximum_size
            ):
                issues.append(name)
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _stuffing_program(profile: AnalyticImplicitProfile, /) -> NativeIntervalProgram:
    """Native point program of the profile's boundary field, one point per row."""
    kernel, state = profile.geometry.kernel, profile.geometry.state

    def field(point: jax.Array) -> jax.Array:
        return kernel.boundary_field(state, point[None, :])

    program = prepare_interval_function(
        field, 3, batch_capacity=1, execution="batched"
    ).native_program
    if program is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The analytic boundary field must lower to a native interval program.",
            "source_preparation",
        )
    return program


def _stuffed_primal_mesh(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    program: NativeIntervalProgram,
    bounds: np.ndarray,
    fidelity_tolerance: float,
    spacing: float,
    levels: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Stuff the profile interior on one lattice, or return None for a finer lattice.

    The floating-point construction proposes; the exact volume certificate,
    the profile topology and the certified facet deviation accept it.
    Returns the points, the positive tetrahedra and the outward boundary
    triangles.
    """
    try:
        # The profile field is a signed distance: Lipschitz constant one.
        stuffed = program.stuff_tetrahedra(
            bounds[0], bounds[1], spacing, levels=levels, lipschitz=1.0
        )
    except ValueError as error:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"The analytic interior must lie inside its boundary bounds: {error}",
            "implicit_stuffing",
        ) from error
    points, cells = stuffed.points, stuffed.cells
    if stuffed.degenerate_count or cells.shape[0] == 0:
        return None
    boundary = _boundary_faces(cells, points.shape[0])
    if not verify_volume(points, cells, boundary).certified:
        return None
    if not _topology_matches(profile, boundary):
        return None
    if np.any(profile.triangle_distance_upper(points[boundary]) > fidelity_tolerance):
        return None
    if not specification.size_controls:
        return points, cells, boundary.astype(np.int32)
    edges = np.unique(
        np.sort(
            np.concatenate(
                (boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]])
            ),
            axis=1,
        ),
        axis=0,
    )
    edge_lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    for control in specification.size_controls:
        if (
            not isinstance(control, UniformSizeControl)
            or control.strength is not SizeControlStrength.HARD
        ):
            continue
        allowed = control.target_size * (
            1.0 + specification.size_compliance.relative_tolerance
        )
        allowed += specification.size_compliance.absolute_tolerance
        if control.maximum_size is not None:
            allowed = min(allowed, control.maximum_size)
        if (
            control.minimum_size is not None
            and float(np.min(edge_lengths)) < control.minimum_size
        ):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Hard minimum boundary size for control {control.control_id!r} is unmet by the stuffed boundary.",
                "specification_compliance",
            )
        if float(np.max(edge_lengths)) > allowed:
            return None
        for statistic in specification.size_compliance.target_statistics:
            quantile = 0.5 if statistic == "p50" else 0.95
            if float(np.quantile(edge_lengths, quantile)) > allowed:
                return None
    return points, cells, boundary.astype(np.int32, copy=False)


@final
class PreparedStuffedImplicitVolume(StrictModule, NonTrainableState):
    """Prepared analytic-profile volume route by isosurface stuffing.

    ``program`` is the profile field's native point program, lowered once at
    preparation. ``bounds`` is the boundary's bounding box, which the lattice
    pads; ``spacing`` the first lattice spacing near the zero set and
    ``levels`` the number of halvings of the lattice away from it.
    """

    profile: AnalyticImplicitProfile
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    coordinate_contract: SpatialCoordinateContract
    program: NativeIntervalProgram
    fidelity_tolerance: float = eqx.field(static=True)
    seed_cover_radius: float = eqx.field(static=True)
    bounds: np.ndarray
    spacing: float = eqx.field(static=True)
    levels: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: AnalyticImplicitProfile,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        coordinate_contract: SpatialCoordinateContract,
        program: NativeIntervalProgram,
        fidelity_tolerance: float,
        seed_cover_radius: float,
        bounds: np.ndarray,
        spacing: float,
        levels: int,
        prepared_id: str,
        /,
    ) -> None:
        self.profile = profile
        self.specification = specification
        self.schedule = schedule
        self.coordinate_contract = coordinate_contract
        self.program = program
        self.fidelity_tolerance = float(fidelity_tolerance)
        self.seed_cover_radius = float(seed_cover_radius)
        box = np.array(bounds, dtype=np.float64, copy=True)
        box.setflags(write=False)
        self.bounds = box
        self.spacing = float(spacing)
        self.levels = levels
        self.prepared_id = prepared_id

    def construct(self, /) -> StuffedImplicitConstruction:
        """Stuff, accept and improve the lattice: the arrays before publication.

        Acceptance of a lattice uses the exact volume certificate, the profile
        topology and the certified facet deviation. Stuffed cells that meet
        the schedule's shape targets with margin are kept as they are; others
        go through the native fixed-boundary quality pass. `execute` publishes
        the arrays as a `CellMesh` and runs the source-fidelity, coverage,
        compliance and embedding certificates on it.
        """
        profile = self.profile
        spacing = self.spacing
        # A rejected lattice is repaired by a finer one; the bounded number of
        # refinements keeps the cell count within 64 times the first attempt's.
        for _ in range(_STUFFING_REFINEMENTS):
            attempt = _stuffed_primal_mesh(
                profile,
                self.specification,
                self.program,
                self.bounds,
                self.fidelity_tolerance,
                spacing,
                self.levels,
            )
            if attempt is not None:
                break
            spacing *= _STUFFING_REFINEMENT_FACTOR
        else:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "No stuffed lattice met the certified volume, topology, fidelity and size controls.",
                "implicit_stuffing",
            )
        points, cells, boundary = attempt
        if _meets_schedule(points, cells, self.schedule):
            return StuffedImplicitConstruction(
                points,
                cells,
                boundary,
                np.arange(points.shape[0], dtype=np.int64),
                np.arange(cells.shape[0], dtype=np.int64),
            )
        return _native_fixed_boundary_quality_pass(points, cells, boundary, self.schedule)

    def execute(self, /) -> NativeMeshConstruction:
        profile = self.profile
        specification = self.specification
        profile.require_bound(profile.geometry, self.coordinate_contract)
        mesh, target_face_global_ids, boundary_faces = _published_mesh(
            self.construct(), profile.source_revision
        )
        samples = profile.boundary_cover(self.seed_cover_radius)
        if (
            not samples.complete
            or samples.semantics != "certified"
            or samples.points.shape[0] == 0
            or np.any(samples.covering_radius > self.seed_cover_radius)
        ):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The profile must provide a complete certified boundary cover at the requested physical radius.",
                "source_preparation",
            )
        cover_radius = self.seed_cover_radius
        sample_count = samples.points.shape[0]
        del samples
        if not _topology_matches(profile, boundary_faces):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The native fixed-boundary quality pass changed analytic boundary topology.",
                "implicit_quality_materialization",
            )
        facet_deviation = profile.triangle_distance_upper(
            np.asarray(mesh.coordinates, dtype=np.float64)[boundary_faces]
        )
        if np.any(facet_deviation > self.fidelity_tolerance):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The native fixed-boundary quality pass violated analytic source fidelity.",
                "implicit_quality_materialization",
            )
        deviation_upper = float(np.max(facet_deviation, initial=0.0))
        geometry = CellGeometrySpec.affine(mesh)
        fidelity_witness = SourceBoundaryDistanceWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=profile.source_id,
            source_revision=profile.source_revision,
            source_query_id=profile.profile_id,
            sample_count=sample_count,
            cover_radius=cover_radius,
            target_facet_global_ids=target_face_global_ids,
        )
        coverage_witness = TubularProjectionBoundaryWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            profile_id=profile.profile_id,
            source_id=profile.source_id,
            source_revision=profile.source_revision,
            domain_id=profile.domain_id,
            distance_witness_id=fidelity_witness.witness_id,
        )
        compliance = _compliance(
            specification,
            self.schedule,
            mesh,
            boundary_faces,
            self.fidelity_tolerance,
            deviation_upper,
        )
        if not compliance.passed:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Stuffed analytic volume construction does not meet requested controls: "
                + "; ".join(compliance.issues),
                "specification_compliance",
            )
        cell_regions = np.zeros((mesh.blocks[0].cell_count,), dtype=np.int32)
        # The exact global embedding is computed once here and passed to
        # acceptance, which checks its inputs before the coverage proof uses it.
        computed_validity = ComputedCellValidity(
            geometry, mesh, CellValidityPolicy(), None
        )
        validity = computed_validity.certificate
        validated_embedding = (
            ValidatedGlobalEmbedding(
                mesh, geometry, validity, validated_validity=computed_validity
            )
            if validity.all_certified
            else None
        )
        organization = _organization(
            profile,
            self.coordinate_contract,
            mesh,
            target_face_global_ids,
            boundary_faces,
            deviation_upper,
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_implicit"),
            profile.source_id,
            profile.source_revision,
            domain=profile,
            cell_regions=cell_regions,
            fidelity_source=profile,
            fidelity_tolerance=self.fidelity_tolerance,
            fidelity_witness=fidelity_witness,
            feature_source=profile,
            source_coverage_witness=coverage_witness,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            validated_global_embedding=validated_embedding,
            organization=organization,
        )


@checked
def prepare_stuffed_implicit_volume(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedStuffedImplicitVolume:
    """Prepare the isosurface-stuffing route of an analytic profile."""
    issues = _support_issues(profile, specification)
    if not _shape_targets_gated(schedule):
        # Its native quality pass refuses any unresolved cell before acceptance.
        issues += ("hard NativeVolumeSchedule shape strength",)
    if issues:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Stuffed analytic volume controls are unsupported: " + "; ".join(issues),
            "source_inspection",
        )
    try:
        profile.require_bound(profile.geometry, coordinate_contract)
    except (TypeError, ValueError, OverflowError) as error:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"Analytic profile is not bound to the requested compiled geometry and coordinate contract: {error}",
            "source_inspection",
        ) from error
    fidelity, cover = _cover_radius(profile, specification)
    target_size = min(
        (
            control.target_size
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        default=math.inf,
    )
    # The certified facet deviation of a triangle of diameter d on the zero
    # set of a signed distance of reach R is d^2 / (6 (R - d)); the first
    # lattice holds it below a fraction of the tolerance for the largest
    # stuffed triangle diameter per spacing.
    reach = profile.reach_lower
    aim = _FIRST_LATTICE_FIDELITY_FRACTION * fidelity
    diameter = math.sqrt(9.0 * aim * aim + 6.0 * aim * reach) - 3.0 * aim
    spacing = min(target_size, diameter / _STUFFED_DIAMETER_PER_SPACING)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The profile's reach and requested volume size must admit a finite positive lattice spacing.",
            "source_inspection",
        )
    bounds = _stuffing_bounds(profile)
    smallest_extent = float(np.min(bounds[1] - bounds[0]))
    levels = min(
        16,
        max(
            0, math.floor(math.log2(_GRADING_EXTENT_FRACTION * smallest_extent / spacing))
        ),
    )
    identity = canonical_fingerprint(
        {
            "kind": "prepared-restricted-implicit-volume",
            "profile": profile.profile_id,
            "domain": profile.domain_id,
            "specification": specification.specification_id,
            "schedule": schedule.schedule_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "fidelity_tolerance": fidelity,
            "cover_radius": cover,
            "bounds": bounds,
            "spacing": spacing,
            "levels": levels,
        }
    )
    return PreparedStuffedImplicitVolume(
        profile,
        specification,
        schedule,
        coordinate_contract,
        _stuffing_program(profile),
        fidelity,
        cover,
        bounds,
        spacing,
        levels,
        identity,
    )


__all__ = ["PreparedStuffedImplicitVolume", "prepare_stuffed_implicit_volume"]
