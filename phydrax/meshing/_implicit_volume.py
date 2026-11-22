#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Restricted Delaunay tetrahedralization of analytic implicit profiles.

The zero set is meshed first by restricted Delaunay refinement, and the
interior from the same triangulation.
"""

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
    RestrictedDelaunayCriteria,
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
# The restricted facets aim at this fraction of the fidelity tolerance.
_FIDELITY_FRACTION = 0.95
# Constructed cells aim at the schedule's shape targets with these margins.
_SCHEDULE_RATIO_MARGIN = 0.99
_SCHEDULE_DETERMINANT_MARGIN = 2.0
_SCHEDULE_DIHEDRAL_MARGIN = 0.5
# Restricted Delaunay construction: Boissonnat-Oudot's 30 degree facet angle
# bound, at least the surface-to-volume rows' 15 degree dihedral angle, box
# roots closer than this fraction of the facet edge bound to an earlier seed
# left to the refinement, and the facet-deviation repair rounds.
_RESTRICTED_FACET_ANGLE_DEGREES = 30.0
_RESTRICTED_DIHEDRAL_DEGREES = 15.0
_RESTRICTED_SEED_SPACING = 0.8
_RESTRICTED_REPAIRS = 3


def _failure(
    category: MeshingFailureCategory, message: str, stage: str, /
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_implicit_restricted_delaunay",
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
        issues.append("patch controls on the derived restricted boundary")
    if specification.periodic_constraints:
        issues.append("periodic constraints on the analytic profile domain")
    if specification.layer_controls:
        issues.append("boundary layers through the layer-core route")
    return tuple(sorted(set(issues)))


def _boundary_bounds(profile: AnalyticImplicitProfile) -> np.ndarray:
    # The restricted construction's box tree pads this box, which holds the
    # zero set.
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


def _fidelity_tolerance(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    /,
) -> float:
    requested = profile.tube_radius
    if specification.maximum_deviation is not None:
        requested = min(requested, float(specification.maximum_deviation))
    if not math.isfinite(requested) or requested <= 0.0:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Restricted analytic volume construction requires a positive physical fidelity tolerance.",
            "source_inspection",
        )
    return requested


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


class RestrictedImplicitConstruction(NamedTuple):
    """The constructed tetrahedral state of the restricted route, before publication.

    ``points`` rows are the vertices; positively oriented ``tetrahedra`` and
    outward ``boundary_faces`` (the restricted surface) index those rows.
    ``vertex_ids`` and ``cell_ids`` are the global identities
    `PreparedRestrictedImplicitVolume.execute` publishes for the rows: their
    row numbers.
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


def _published_mesh(
    construction: RestrictedImplicitConstruction, numeric_version: str, /
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
            "The published restricted construction lost tetrahedral connectivity.",
            "implicit_publication",
        )
    output_vertex_ids = np.asarray(output.vertex_global_ids, dtype=np.int64)
    source_boundary_ids = np.asarray(construction.vertex_ids, dtype=np.int64)[
        np.asarray(construction.boundary_faces, dtype=np.int64)
    ]
    output_boundary_faces = _rows_of(output_vertex_ids, source_boundary_ids)
    if output_boundary_faces is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The published mesh lost a boundary vertex of the restricted construction.",
            "implicit_publication",
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
            "The published mesh boundary differs from the restricted construction's surface.",
            "implicit_publication",
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
                "geometry-owned-restricted-implicit-volume-boundary",
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


def _field_program(profile: AnalyticImplicitProfile, /) -> NativeIntervalProgram:
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


@final
class PreparedRestrictedImplicitVolume(StrictModule, NonTrainableState):
    """Prepared analytic-profile volume route by restricted Delaunay refinement.

    ``program`` is the profile field's native program, lowered once at
    preparation, and ``bounds`` the boundary's bounding box, which holds the
    zero set. Restricted facets keep their longest edge within
    ``facet_edge`` and cells their circumradius within ``cell_size`` (None:
    no size).
    """

    profile: AnalyticImplicitProfile
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    coordinate_contract: SpatialCoordinateContract
    program: NativeIntervalProgram
    fidelity_tolerance: float = eqx.field(static=True)
    bounds: np.ndarray
    facet_edge: float = eqx.field(static=True)
    cell_size: float | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: AnalyticImplicitProfile,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        coordinate_contract: SpatialCoordinateContract,
        program: NativeIntervalProgram,
        fidelity_tolerance: float,
        bounds: np.ndarray,
        facet_edge: float,
        cell_size: float | None,
        prepared_id: str,
        /,
    ) -> None:
        self.profile = profile
        self.specification = specification
        self.schedule = schedule
        self.coordinate_contract = coordinate_contract
        self.program = program
        self.fidelity_tolerance = float(fidelity_tolerance)
        box = np.array(bounds, dtype=np.float64, copy=True)
        box.setflags(write=False)
        self.bounds = box
        self.facet_edge = float(facet_edge)
        self.cell_size = cell_size
        self.prepared_id = prepared_id

    def construct(self, /) -> RestrictedImplicitConstruction:
        """The constructed arrays before publication.

        `execute` publishes them as a `CellMesh` and runs the source-fidelity,
        coverage, compliance and embedding certificates on it.
        """
        return self._constructed(np.empty((0, 3), dtype=np.float64))

    def _constructed(self, extra_seeds: np.ndarray, /) -> RestrictedImplicitConstruction:
        """Restricted Delaunay refinement of the zero set, then of the interior.

        ``extra_seeds`` are points near the zero set that start the surface
        beside the box-tree roots (facet repair). Cells reach the schedule's
        margined shape targets, with at least the restricted dihedral angle,
        by flips and relocations, else the native fixed-boundary quality pass.
        """
        schedule = self.schedule
        ratio = (
            None
            if schedule.radius_edge_bound is None
            else _SCHEDULE_RATIO_MARGIN * schedule.radius_edge_bound
        )
        criteria = RestrictedDelaunayCriteria(
            self.facet_edge,
            self.fidelity_tolerance,
            _RESTRICTED_FACET_ANGLE_DEGREES,
            math.inf if self.cell_size is None else self.cell_size,
            ratio,
            max(schedule.minimum_dihedral_degrees, _RESTRICTED_DIHEDRAL_DEGREES)
            + _SCHEDULE_DIHEDRAL_MARGIN,
            _SCHEDULE_DETERMINANT_MARGIN
            * max(schedule.minimum_relative_determinant, 2.0**-22),
            _RESTRICTED_SEED_SPACING,
        )
        volume = self.program.restricted_tetrahedra(
            self.bounds[0],
            self.bounds[1],
            criteria,
            self.profile.component_count,
            self.profile.boundary_euler_characteristic,
            extra_seeds,
        )
        if volume.status is MeshcoreStatus.UNRESOLVED:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted Delaunay refinement is unresolved at its separation bound "
                f"({volume.refused_facets} refused facet(s) still violate the criteria).",
                "implicit_restricted_delaunay",
            )
        if volume.unmet_cells:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted Delaunay construction left "
                f"{volume.unmet_cells} cell(s) below the shape targets.",
                "implicit_restricted_delaunay",
            )
        return RestrictedImplicitConstruction(
            volume.points,
            volume.cells,
            volume.boundary_faces,
            np.arange(volume.points.shape[0], dtype=np.int64),
            np.arange(volume.cells.shape[0], dtype=np.int64),
        )

    def _repaired(
        self, construction: RestrictedImplicitConstruction, /
    ) -> RestrictedImplicitConstruction:
        """The construction with restricted facets above the tolerance repaired.

        A restricted facet whose certified deviation exceeds the tolerance is
        split: its centroid, moved onto the zero set, seeds the refinement
        again, for a bounded number of rounds.
        """
        seeds = np.empty((0, 3), dtype=np.float64)
        for _ in range(_RESTRICTED_REPAIRS):
            facets = construction.points[construction.boundary_faces]
            failing = (
                self.profile.triangle_distance_upper(facets) > self.fidelity_tolerance
            )
            if not np.any(failing):
                break
            seeds = np.concatenate((seeds, np.mean(facets[failing], axis=1)))
            construction = self._constructed(seeds)
        return construction

    def execute(self, /) -> NativeMeshConstruction:
        profile = self.profile
        specification = self.specification
        profile.require_bound(profile.geometry, self.coordinate_contract)
        mesh, target_face_global_ids, boundary_faces = _published_mesh(
            self._repaired(self.construct()), profile.source_revision
        )
        if not _topology_matches(profile, boundary_faces):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The restricted construction's surface does not have the analytic boundary topology.",
                "implicit_restricted_delaunay",
            )
        facet_deviation = profile.triangle_distance_upper(
            np.asarray(mesh.coordinates, dtype=np.float64)[boundary_faces]
        )
        if np.any(facet_deviation > self.fidelity_tolerance):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The restricted construction's surface exceeds the analytic source fidelity after repair.",
                "implicit_restricted_delaunay",
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
            reverse_tolerance=self.fidelity_tolerance,
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
                "Restricted analytic volume construction does not meet requested controls: "
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
def prepare_restricted_implicit_volume(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedRestrictedImplicitVolume:
    """Prepare the restricted Delaunay volume route of an analytic profile.

    The facet edge bound follows from the profile's certified reach and the
    fidelity tolerance, and the cell size from the uniform size controls.
    """
    issues = _support_issues(profile, specification)
    if not _shape_targets_gated(schedule):
        # The construction refuses any cell it leaves below the shape targets.
        issues += ("hard NativeVolumeSchedule shape strength",)
    if issues:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Restricted analytic volume controls are unsupported: " + "; ".join(issues),
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
    fidelity = _fidelity_tolerance(profile, specification)
    target_size = min(
        (
            control.target_size
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        default=math.inf,
    )
    # The certified facet deviation of a triangle of diameter d on the zero
    # set of a signed distance of reach R is d^2 / (6 (R - d)); restricted
    # facets hold it below a fraction of the tolerance.
    reach = profile.reach_lower
    aim = _FIDELITY_FRACTION * fidelity
    diameter = math.sqrt(9.0 * aim * aim + 6.0 * aim * reach) - 3.0 * aim
    facet_edge = min(target_size, diameter)
    if not math.isfinite(facet_edge) or facet_edge <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The profile's reach and requested volume size must admit a finite positive facet edge.",
            "source_inspection",
        )
    # The circumradius of the regular tetrahedron of the target edge.
    cell_size = None if math.isinf(target_size) else target_size * math.sqrt(6.0) / 4.0
    bounds = _boundary_bounds(profile)
    identity = canonical_fingerprint(
        {
            "kind": "prepared-restricted-implicit-volume",
            "profile": profile.profile_id,
            "domain": profile.domain_id,
            "specification": specification.specification_id,
            "schedule": schedule.schedule_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "fidelity_tolerance": fidelity,
            "bounds": bounds,
            "facet_edge": facet_edge,
            "cell_size": cell_size,
        }
    )
    return PreparedRestrictedImplicitVolume(
        profile,
        specification,
        schedule,
        coordinate_contract,
        _field_program(profile),
        fidelity,
        bounds,
        facet_edge,
        cell_size,
        identity,
    )


__all__ = ["PreparedRestrictedImplicitVolume", "prepare_restricted_implicit_volume"]
