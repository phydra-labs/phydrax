#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Established-reach restricted-Delaunay tetrahedralization of analytic profiles."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    delaunay_3d,
    exact_orient3d,
    MeshcoreStatus,
    PlcBoundaryPolicy,
    restricted_centers_3d,
    restricted_dual_3d,
    restricted_rays_3d,
    RestrictedDual3D,
    TetMesh3D,
    TetMeshEditStatus,
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
from ..geometry._source_boundary_records import (
    SourceBoundaryDistanceWitness,
    SourceBoundarySamples,
)
from ..geometry.implicit import AnalyticImplicitProfile
from ..geometry.implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
    validate_restricted_delaunay_boundary_witness,
)
from ..geometry.simplicial._topology import TriangleTopology
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


_CLASS_INSIDE = 0
_CLASS_OUTSIDE = 1
_CLASS_BOUNDARY = 2
_SIDE_INTERIOR = 0
_SIDE_EXTERIOR = 1
_FACE_CORNERS = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_ROOT_EDGE_BATCH_SIZE = 2048


def _failure(
    category: MeshingFailureCategory, message: str, stage: str, /
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_implicit_restricted_delaunay",
        stage=stage,
    )


def _stable_id(value: object, /) -> int:
    token = canonical_fingerprint(value)
    raw = hashlib.blake2b(token.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(raw, "little") & np.iinfo(np.int64).max


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
        issues.append("periodic constraints on the restricted analytic domain")
    if specification.layer_controls:
        issues.append("boundary layers through the layer-core route")
    return tuple(sorted(set(issues)))


def _carrier_bounds(profile: AnalyticImplicitProfile) -> np.ndarray:
    bounds = np.asarray(profile.boundary_bounds, dtype=np.float64)
    extent = bounds[1] - bounds[0]
    margin = max(profile.reach_lower, float(np.max(extent)) / 8.0)
    carrier = np.stack(
        (
            np.nextafter(bounds[0] - margin, -np.inf),
            np.nextafter(bounds[1] + margin, np.inf),
        )
    )
    if (
        not np.all(np.isfinite(carrier))
        or np.any(carrier[0] >= bounds[0])
        or np.any(carrier[1] <= bounds[1])
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The analytic carrier must finitely enclose the profile boundary and its established-reach neighborhood.",
            "source_inspection",
        )
    return carrier


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
            "Restricted analytic volume construction requires a positive physical fidelity tolerance.",
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


def _carrier_lattice(
    profile: AnalyticImplicitProfile,
    carrier: np.ndarray,
    spacing: float,
    cover_radius: float,
    /,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    bounds = np.asarray(carrier, dtype=np.float64)
    if not math.isfinite(float(spacing)) or spacing <= 0.0:
        raise ValueError("carrier lattice spacing must be finite and positive.")
    extent = bounds[1] - bounds[0]
    ratios = extent / spacing
    if not np.all(np.isfinite(ratios)) or np.any(ratios > np.iinfo(np.intp).max):
        raise OverflowError("The analytic carrier lattice exceeds array index capacity.")
    counts = tuple(max(1, math.ceil(float(value))) for value in ratios)
    total = math.prod(counts)
    if total > np.iinfo(np.intp).max // 3:
        raise OverflowError("The analytic carrier lattice exceeds array index capacity.")
    axes = tuple(
        bounds[0, axis]
        + (np.arange(counts[axis], dtype=np.float64) + 0.5) * extent[axis] / counts[axis]
        for axis in range(3)
    )
    retained: list[np.ndarray] = []
    keys: list[tuple[int, int, int]] = []
    sides: list[int] = []
    batch = 4096
    for start in range(0, total, batch):
        stop = min(start + batch, total)
        flat = np.arange(start, stop, dtype=np.int64)
        first = flat // (counts[1] * counts[2])
        second = (flat // counts[2]) % counts[1]
        third = flat % counts[2]
        rows = np.stack((axes[0][first], axes[1][second], axes[2][third]), axis=1)
        lower, upper = profile.field_bounds.point_values(rows)
        inside = upper < -cover_radius
        outside = lower > cover_radius
        keep = inside | outside
        if np.any(keep):
            retained.append(rows[keep])
            keys.extend(
                (int(i), int(j), int(k))
                for i, j, k, selected in zip(
                    first, second, third, keep.tolist(), strict=True
                )
                if selected
            )
            sides.extend(
                _SIDE_INTERIOR if is_inside else _SIDE_EXTERIOR
                for is_inside, selected in zip(
                    inside.tolist(), keep.tolist(), strict=True
                )
                if selected
            )
    if not retained:
        return np.empty((0, 3), dtype=np.float64), (), ()
    return np.concatenate(retained, axis=0), tuple(keys), tuple(sides)


def _initial_sites(
    profile: AnalyticImplicitProfile,
    carrier: np.ndarray,
    samples: SourceBoundarySamples,
    spacing: float,
    cover_radius: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ),
        dtype=np.int8,
    )
    hull = np.where(corners, carrier[1], carrier[0])
    if np.any(samples.covering_radius > cover_radius):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The retained source cover exceeds its requested physical radius.",
            "source_preparation",
        )
    sample_radius = np.nextafter(samples.covering_radius + samples.point_error, np.inf)
    site_clearance = float(np.nextafter(np.max(sample_radius), np.inf))
    hull_lower, _hull_upper = profile.field_bounds.point_values(hull)
    if np.any(hull_lower <= site_clearance):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted carrier corners must be strictly exterior beyond the certified source-site separation radius.",
            "source_preparation",
        )
    grid, grid_keys, grid_sides = _carrier_lattice(
        profile, carrier, spacing, site_clearance
    )
    witness = np.asarray(profile.interior_witness, dtype=np.float64)
    witness_lower, witness_upper = profile.field_bounds.point_values(witness[None, :])
    if (
        np.asarray(witness_lower).shape != (1,)
        or np.asarray(witness_upper).shape != (1,)
        or not np.isfinite(np.asarray(witness_lower)[0])
        or not np.isfinite(np.asarray(witness_upper)[0])
        or float(np.asarray(witness_upper)[0]) >= -site_clearance
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored interior witness needs a strict field-side enclosure beyond the source-site separation.",
            "source_preparation",
        )
    distinct_grid_rows = ~np.all(grid == witness, axis=1)
    if np.any(~distinct_grid_rows):
        grid = grid[distinct_grid_rows]
        grid_keys = tuple(
            key
            for key, selected in zip(grid_keys, distinct_grid_rows.tolist(), strict=True)
            if selected
        )
        grid_sides = tuple(
            side
            for side, selected in zip(
                grid_sides, distinct_grid_rows.tolist(), strict=True
            )
            if selected
        )
    site_count = 9 + samples.points.shape[0] + grid.shape[0]
    if site_count > np.iinfo(np.intp).max // 3:
        raise OverflowError("The restricted Delaunay site array exceeds index capacity.")
    points = np.concatenate((hull, samples.points, grid, witness[None, :]), axis=0)
    if np.unique(points, axis=0).shape[0] != points.shape[0]:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted Delaunay raw sites must be numerically unique before triangulation.",
            "source_preparation",
        )
    identities = [
        _stable_id(
            ("restricted-carrier-corner", profile.profile_id, tuple(int(v) for v in key))
        )
        for key in corners
    ]
    identities.extend(
        _stable_id(("analytic-boundary-cover-sample", profile.profile_id, row))
        for row in range(samples.points.shape[0])
    )
    identities.extend(
        _stable_id(("restricted-certified-region-site", profile.profile_id, side, key))
        for key, side in zip(grid_keys, grid_sides, strict=True)
    )
    identities.append(_stable_id(("restricted-interior-witness", profile.profile_id)))
    point_ids = np.asarray(identities, dtype=np.int64)
    if np.unique(point_ids).size != point_ids.size:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted Delaunay sites require unique deterministic numerical identities.",
            "source_inspection",
        )
    sample_input_rows = np.arange(8, 8 + samples.points.shape[0], dtype=np.int64)
    return points, point_ids, sample_input_rows


def _delaunay_snapshot(
    input_points: np.ndarray,
    input_ids: np.ndarray,
    sample_input_rows: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tetrahedra, vertex_map = delaunay_3d(input_points)
    mapping = np.asarray(vertex_map, dtype=np.int64)
    if (
        mapping.shape != (input_points.shape[0],)
        or np.any(mapping < 0)
        or np.unique(mapping).size != mapping.size
        or not tetrahedra.size
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "The native exact Delaunay constructor must retain every input site in a nonempty three-dimensional triangulation.",
            "restricted_delaunay",
        )
    point_count = int(np.max(mapping)) + 1
    source_rows = np.full((point_count,), -1, dtype=np.int64)
    for input_row, point_row in enumerate(mapping.tolist()):
        if source_rows[point_row] < 0:
            source_rows[point_row] = input_row
    if np.any(source_rows < 0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "The native Delaunay vertex map must retain every numerical point identity.",
            "restricted_delaunay",
        )
    points = np.asarray(input_points[source_rows], dtype=np.float64)
    point_ids = np.asarray(input_ids[source_rows], dtype=np.int64)
    cells = np.asarray(tetrahedra, dtype=np.int32)
    sample_rows = mapping[np.asarray(sample_input_rows, dtype=np.int64)]
    return points, point_ids, cells, sample_rows


def _triangle_key(first: int, second: int, third: int, /) -> tuple[int, int, int]:
    ordered = sorted((first, second, third))
    return ordered[0], ordered[1], ordered[2]


def _tetrahedron_faces(
    tetrahedra: np.ndarray, /
) -> dict[tuple[int, int, int], list[int]]:
    incidence: dict[tuple[int, int, int], list[int]] = {}
    cells = np.asarray(tetrahedra, dtype=np.int64).reshape((-1, 4)).tolist()
    for row, cell in enumerate(cells):
        for first, second, third in _FACE_CORNERS:
            key = _triangle_key(
                int(cell[first]),
                int(cell[second]),
                int(cell[third]),
            )
            rows = incidence.get(key)
            if rows is None:
                incidence[key] = [row]
            else:
                rows.append(row)
    if any(len(rows) > 2 for rows in incidence.values()):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "The native Delaunay complex must be face-manifold.",
            "restricted_delaunay",
        )
    return incidence


def _inside_seed_rows(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    /,
) -> tuple[int, ...]:
    vertices = np.asarray(points, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    witness = np.asarray(profile.interior_witness, dtype=np.float64)
    witness_rows = np.flatnonzero(np.all(vertices == witness, axis=1))
    if witness_rows.shape != (1,):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored interior witness must be retained exactly once in the Delaunay vertex bank.",
            "restricted_seed_star",
        )
    witness_row = int(witness_rows[0])
    incident_rows = np.flatnonzero(np.any(cells == witness_row, axis=1))
    if not incident_rows.size:
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored interior witness must have a nonempty Delaunay cell star.",
            "restricted_seed_star",
        )
    star_vertices = vertices[cells[incident_rows]]
    from ..geometry._interval_enclosure import (
        interval_add,
        interval_sqrt,
        interval_square,
        interval_subtract,
    )

    displacement_bounds = interval_subtract(
        (star_vertices, star_vertices),
        (witness[None, None, :], witness[None, None, :]),
    )
    displacement_lower = np.asarray(displacement_bounds[0], dtype=np.float64)
    displacement_upper = np.asarray(displacement_bounds[1], dtype=np.float64)
    if (
        displacement_lower.shape != star_vertices.shape
        or displacement_upper.shape != star_vertices.shape
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored witness star requires rank-preserving interval displacements.",
            "restricted_seed_star",
        )
    squared_bounds = interval_square((displacement_lower, displacement_upper))
    squared_lower = np.asarray(squared_bounds[0], dtype=np.float64)
    squared_upper = np.asarray(squared_bounds[1], dtype=np.float64)
    if (
        squared_lower.shape != star_vertices.shape
        or squared_upper.shape != star_vertices.shape
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored witness star requires rank-preserving squared bounds.",
            "restricted_seed_star",
        )
    squared01_bounds = interval_add(
        (squared_lower[:, :, 0], squared_upper[:, :, 0]),
        (squared_lower[:, :, 1], squared_upper[:, :, 1]),
    )
    squared01_lower = np.asarray(squared01_bounds[0], dtype=np.float64)
    squared01_upper = np.asarray(squared01_bounds[1], dtype=np.float64)
    expected_star_shape = star_vertices.shape[:2]
    if (
        squared01_lower.shape != expected_star_shape
        or squared01_upper.shape != expected_star_shape
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored witness star requires rank-preserving partial squared bounds.",
            "restricted_seed_star",
        )
    squared012_bounds = interval_add(
        (squared01_lower, squared01_upper),
        (squared_lower[:, :, 2], squared_upper[:, :, 2]),
    )
    squared012_lower = np.asarray(squared012_bounds[0], dtype=np.float64)
    squared012_upper = np.asarray(squared012_bounds[1], dtype=np.float64)
    if (
        squared012_lower.shape != expected_star_shape
        or squared012_upper.shape != expected_star_shape
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored witness star requires rank-preserving full squared bounds.",
            "restricted_seed_star",
        )
    radius_bounds = interval_sqrt((squared012_lower, squared012_upper))
    radius_upper_cells = np.asarray(radius_bounds[1], dtype=np.float64)
    if radius_upper_cells.shape != expected_star_shape:
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored witness star requires rank-preserving radius bounds.",
            "restricted_seed_star",
        )
    radius_upper = np.max(radius_upper_cells, axis=1)
    distance = profile.boundary_distance(witness[None, :])
    if distance.semantics != "certified":
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "The authored interior witness needs a certified source-boundary distance.",
            "restricted_seed_star",
        )
    lower = np.asarray(distance.lower, dtype=np.float64)
    if (
        lower.shape != (1,)
        or not np.isfinite(lower[0])
        or np.any(radius_upper >= lower[0])
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Every tetrahedron in the authored interior witness star must lie inside its certified boundary-free ball.",
            "restricted_seed_star",
        )
    return tuple(int(row) for row in incident_rows.tolist())


def _field_box_bounds(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(points, dtype=np.float64)[
        np.asarray(tetrahedra, dtype=np.int64)
    ]
    lower = np.min(vertices, axis=1)
    upper = np.max(vertices, axis=1)
    boxes = profile.field_bounds.boxes(lower, upper)
    classification = np.full((vertices.shape[0],), _CLASS_BOUNDARY, dtype=np.int8)
    classification[boxes.value_upper < 0.0] = _CLASS_INSIDE
    classification[boxes.value_lower > 0.0] = _CLASS_OUTSIDE
    return lower, upper, boxes.value_lower, boxes.value_upper, classification, vertices


def _segment_interval(
    endpoint_bounds: np.ndarray,
    first: float,
    second: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ..geometry._interval_enclosure import (
        interval_add,
        interval_multiply,
        interval_subtract,
    )

    bounds = np.asarray(endpoint_bounds, dtype=np.float64)
    origin = (bounds[0, 0], bounds[0, 1])
    destination = (bounds[1, 0], bounds[1, 1])
    direction = interval_subtract(destination, origin)
    parameter = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    coordinates = interval_add(origin, interval_multiply(parameter, direction))
    return (
        np.asarray(coordinates[0], dtype=np.float64),
        np.asarray(coordinates[1], dtype=np.float64),
        np.asarray(direction[0], dtype=np.float64),
        np.asarray(direction[1], dtype=np.float64),
    )


def _directional_bounds(
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    direction_lower: np.ndarray,
    direction_upper: np.ndarray,
    /,
) -> tuple[float, float]:
    from ..geometry._interval_enclosure import interval_add, interval_multiply

    product_lower, product_upper = interval_multiply(
        (gradient_lower, gradient_upper), (direction_lower, direction_upper)
    )
    lower_products = np.asarray(product_lower, dtype=np.float64).reshape((-1,))
    upper_products = np.asarray(product_upper, dtype=np.float64).reshape((-1,))
    result = (
        np.asarray(lower_products[0], dtype=np.float64),
        np.asarray(upper_products[0], dtype=np.float64),
    )
    for position in range(1, lower_products.size):
        result = interval_add(
            result,
            (
                np.asarray(lower_products[position], dtype=np.float64),
                np.asarray(upper_products[position], dtype=np.float64),
            ),
        )
    return float(np.asarray(result[0])), float(np.asarray(result[1]))


def _segment_interval_bank(
    endpoint_bounds: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ..geometry._interval_enclosure import (
        interval_add,
        interval_multiply,
        interval_subtract,
    )

    bounds = np.asarray(endpoint_bounds, dtype=np.float64)
    if bounds.ndim != 4 or bounds.shape[1:] != (2, 2, 3):
        raise ValueError("segment endpoint bounds must have shape (N, 2, 2, 3).")
    count = bounds.shape[0]
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != (count,) or second_values.shape != (count,):
        raise ValueError("segment parameters must have one scalar per endpoint pair.")
    origin = (bounds[:, 0, 0], bounds[:, 0, 1])
    destination = (bounds[:, 1, 0], bounds[:, 1, 1])
    raw_direction = interval_subtract(destination, origin)
    direction_lower = np.asarray(raw_direction[0], dtype=np.float64)
    direction_upper = np.asarray(raw_direction[1], dtype=np.float64)
    expected_shape = (count, 3)
    if direction_lower.shape != expected_shape or direction_upper.shape != expected_shape:
        raise ValueError("segment direction interval lost its (N, 3) shape.")
    parameter = (first_values[:, None], second_values[:, None])
    raw_relative = interval_multiply(parameter, (direction_lower, direction_upper))
    relative_lower = np.asarray(raw_relative[0], dtype=np.float64)
    relative_upper = np.asarray(raw_relative[1], dtype=np.float64)
    if relative_lower.shape != expected_shape or relative_upper.shape != expected_shape:
        raise ValueError("segment parameter interval lost its (N, 3) shape.")
    raw_coordinates = interval_add(origin, (relative_lower, relative_upper))
    coordinate_lower = np.asarray(raw_coordinates[0], dtype=np.float64)
    coordinate_upper = np.asarray(raw_coordinates[1], dtype=np.float64)
    if (
        coordinate_lower.shape != expected_shape
        or coordinate_upper.shape != expected_shape
    ):
        raise ValueError("segment coordinate interval lost its (N, 3) shape.")
    return coordinate_lower, coordinate_upper, direction_lower, direction_upper


def _directional_bounds_bank(
    gradient_lower: np.ndarray,
    gradient_upper: np.ndarray,
    direction_lower: np.ndarray,
    direction_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    from ..geometry._interval_enclosure import interval_add, interval_multiply

    product_lower, product_upper = interval_multiply(
        (gradient_lower, gradient_upper), (direction_lower, direction_upper)
    )
    product_lower = np.asarray(product_lower, dtype=np.float64)
    product_upper = np.asarray(product_upper, dtype=np.float64)
    result_lower = product_lower[:, 0]
    result_upper = product_upper[:, 0]
    for axis in (1, 2):
        result_lower, result_upper = interval_add(
            (result_lower, result_upper),
            (product_lower[:, axis], product_upper[:, axis]),
        )
    return (
        np.asarray(result_lower, dtype=np.float64),
        np.asarray(result_upper, dtype=np.float64),
    )


def _isolate_monotone_root_bank(
    profile: AnalyticImplicitProfile,
    endpoint_bounds: np.ndarray,
    spatial_resolution: float,
    initial_enclosure: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    /,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...]:
    from ..geometry._interval_enclosure import interval_divide, interval_subtract

    bounds = np.asarray(endpoint_bounds, dtype=np.float64)
    count = bounds.shape[0]
    root_lower = np.zeros((count,), dtype=np.float64)
    root_upper = np.ones((count,), dtype=np.float64)
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None] = [
        None
    ] * count
    active = np.arange(count, dtype=np.int64)
    first_iteration = True

    while active.size:
        if first_iteration:
            (
                value_lower,
                value_upper,
                derivative_lower,
                derivative_upper,
                coordinate_lower,
                coordinate_upper,
            ) = tuple(values[active] for values in initial_enclosure)
            first_iteration = False
        else:
            segment_lower, segment_upper, direction_lower, direction_upper = (
                _segment_interval_bank(
                    bounds[active],
                    root_lower[active],
                    root_upper[active],
                )
            )
            values = profile.field_bounds.boxes(segment_lower, segment_upper)
            value_lower = np.asarray(values.value_lower, dtype=np.float64)
            value_upper = np.asarray(values.value_upper, dtype=np.float64)
            derivative_lower, derivative_upper = _directional_bounds_bank(
                values.gradient_lower,
                values.gradient_upper,
                direction_lower,
                direction_upper,
            )
            coordinate_lower = segment_lower
            coordinate_upper = segment_upper

        if np.any((value_lower > 0.0) | (value_upper < 0.0)):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Interval refinement excludes a previously bracketed restricted-Delaunay root.",
                "restricted_root_isolation",
            )
        derivative_is_strict = (
            np.isfinite(derivative_lower)
            & np.isfinite(derivative_upper)
            & ((derivative_lower > 0.0) | (derivative_upper < 0.0))
        )
        if not np.all(derivative_is_strict):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A restricted-Delaunay root bracket must retain a strict interval derivative sign.",
                "restricted_root_isolation",
            )

        extent = np.nextafter(np.abs(coordinate_upper - coordinate_lower), np.inf)
        diagonal_xy = np.nextafter(np.hypot(extent[:, 0], extent[:, 1]), np.inf)
        diameter = np.nextafter(np.hypot(diagonal_xy, extent[:, 2]), np.inf)
        finished = diameter <= spatial_resolution
        for position in np.flatnonzero(finished).tolist():
            row = int(active[position])
            parameter_bounds = np.asarray(
                (root_lower[row], root_upper[row]), dtype=np.float64
            )
            coordinate_bounds = np.stack(
                (coordinate_lower[position], coordinate_upper[position])
            )
            value_bounds = np.asarray(
                (value_lower[position], value_upper[position]), dtype=np.float64
            )
            derivative_bounds = np.asarray(
                (derivative_lower[position], derivative_upper[position]),
                dtype=np.float64,
            )
            if not all(
                np.all(np.isfinite(values))
                for values in (
                    parameter_bounds,
                    coordinate_bounds,
                    value_bounds,
                    derivative_bounds,
                )
            ):
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Restricted dual root intervals must retain finite outward field and derivative bounds.",
                    "restricted_root_isolation",
                )
            results[row] = (
                parameter_bounds,
                coordinate_bounds,
                value_bounds,
                derivative_bounds,
                1 if derivative_lower[position] > 0.0 else -1,
            )

        remaining = np.flatnonzero(~finished)
        if not remaining.size:
            break
        rows = active[remaining]
        active = rows
        first = root_lower[rows]
        last = root_upper[rows]
        midpoint = first + 0.5 * (last - first)
        if np.any((midpoint <= first) | (midpoint >= last)):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted dual root isolation reached binary64 parameter resolution before meeting the physical root tolerance.",
                "restricted_root_isolation",
            )

        midpoint_lower, midpoint_upper, _direction_lower, _direction_upper = (
            _segment_interval_bank(bounds[rows], midpoint, midpoint)
        )
        midpoint_values = profile.field_bounds.boxes(midpoint_lower, midpoint_upper)
        newton_image = interval_subtract(
            (midpoint, midpoint),
            interval_divide(
                (
                    np.asarray(midpoint_values.value_lower, dtype=np.float64),
                    np.asarray(midpoint_values.value_upper, dtype=np.float64),
                ),
                (
                    derivative_lower[remaining],
                    derivative_upper[remaining],
                ),
            ),
        )
        image_lower = np.asarray(newton_image[0], dtype=np.float64)
        image_upper = np.asarray(newton_image[1], dtype=np.float64)
        new_lower = np.maximum(first, image_lower)
        new_upper = np.minimum(last, image_upper)
        if np.any(new_lower > new_upper):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Interval Newton image is disjoint from a bracketed restricted-Delaunay root.",
                "restricted_root_isolation",
            )
        contracts = (
            np.isfinite(new_lower)
            & np.isfinite(new_upper)
            & ((new_upper - new_lower) < (last - first))
        )
        root_lower[rows[contracts]] = new_lower[contracts]
        root_upper[rows[contracts]] = new_upper[contracts]

        unresolved = np.flatnonzero(~contracts)
        if not unresolved.size:
            continue
        fallback_rows = rows[unresolved]
        fallback_first = first[unresolved]
        fallback_last = last[unresolved]
        fallback_midpoint = midpoint[unresolved]
        span = fallback_last - fallback_first
        candidate_parameters = np.stack(
            (
                fallback_midpoint,
                fallback_first + span / 3.0,
                fallback_first + 2.0 * span / 3.0,
                fallback_first + span / 4.0,
                fallback_first + 3.0 * span / 4.0,
            ),
            axis=1,
        )
        candidate_bounds = np.repeat(bounds[fallback_rows], 5, axis=0)
        flat_parameters = candidate_parameters.reshape((-1,))
        point_lower, point_upper, _point_direction_lower, _point_direction_upper = (
            _segment_interval_bank(
                candidate_bounds,
                flat_parameters,
                flat_parameters,
            )
        )
        candidate_values = profile.field_bounds.boxes(point_lower, point_upper)
        candidate_value_lower = np.asarray(
            candidate_values.value_lower, dtype=np.float64
        ).reshape((-1,))
        candidate_value_upper = np.asarray(
            candidate_values.value_upper, dtype=np.float64
        ).reshape((-1,))
        candidate_signs = np.zeros((flat_parameters.size,), dtype=np.int8)
        candidate_signs[candidate_value_upper < 0.0] = -1
        candidate_signs[candidate_value_lower > 0.0] = 1
        interior = (candidate_parameters > fallback_first[:, None]) & (
            candidate_parameters < fallback_last[:, None]
        )
        candidate_signs = np.where(interior.reshape((-1,)), candidate_signs, 0)
        candidate_signs = candidate_signs.reshape((-1, 5))
        has_sign = np.any(candidate_signs != 0, axis=1)
        if not np.all(has_sign):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A bracketed restricted-Delaunay root cannot be narrowed with certified point signs.",
                "restricted_root_isolation",
            )
        selected_columns = np.argmax(candidate_signs != 0, axis=1)
        selected_signs = candidate_signs[np.arange(fallback_rows.size), selected_columns]
        selected_parameters = candidate_parameters[
            np.arange(fallback_rows.size), selected_columns
        ]
        increasing = derivative_lower[remaining][unresolved] > 0.0
        root_is_left = np.where(
            increasing,
            selected_signs > 0,
            selected_signs < 0,
        )
        root_upper[fallback_rows[root_is_left]] = selected_parameters[root_is_left]
        root_lower[fallback_rows[~root_is_left]] = selected_parameters[~root_is_left]

    if any(result is None for result in results):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Every monotone restricted-Delaunay root must retain a finite strict interval proof.",
            "restricted_root_isolation",
        )
    return tuple(result for result in results if result is not None)


def _root_edge(
    profile: AnalyticImplicitProfile,
    endpoint_bounds: np.ndarray,
    spatial_resolution: float,
    /,
    *,
    initial_enclosure: (
        tuple[float, float, float, float, np.ndarray, np.ndarray] | None
    ) = None,
    endpoint_signs: tuple[int, int] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...]:
    if not profile.field_bounds.gradient_rigorous:
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted dual roots require the profile's rigorous interval-gradient enclosure.",
            "restricted_root_isolation",
        )

    def enclosure(
        lower: float, upper: float
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        if lower == 0.0 and upper == 1.0 and initial_enclosure is not None:
            return initial_enclosure
        x_lower, x_upper, d_lower, d_upper = _segment_interval(
            endpoint_bounds, lower, upper
        )
        values = profile.field_bounds.boxes(x_lower[None, :], x_upper[None, :])
        derivative_lower, derivative_upper = _directional_bounds(
            values.gradient_lower[0], values.gradient_upper[0], d_lower, d_upper
        )
        return (
            float(values.value_lower[0]),
            float(values.value_upper[0]),
            derivative_lower,
            derivative_upper,
            x_lower,
            x_upper,
        )

    known_signs = (
        {} if endpoint_signs is None else {0.0: endpoint_signs[0], 1.0: endpoint_signs[1]}
    )

    def endpoint_sign(parameter: float) -> int:
        known = known_signs.get(parameter)
        if known is not None:
            return known
        value_lower, value_upper, _dl, _du, _xl, _xu = enclosure(parameter, parameter)
        if value_upper < 0.0:
            sign = -1
        elif value_lower > 0.0:
            sign = 1
        else:
            sign = 0
        known_signs[parameter] = sign
        return sign

    from ..geometry._interval_enclosure import interval_divide, interval_subtract

    pending: list[tuple[float, float]] = [(0.0, 1.0)]
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    while pending:
        lower, upper = pending.pop()
        (
            value_lower,
            value_upper,
            derivative_lower,
            derivative_upper,
            x_lower,
            x_upper,
        ) = enclosure(lower, upper)
        if value_lower > 0.0 or value_upper < 0.0:
            continue
        monotone = (
            math.isfinite(derivative_lower)
            and math.isfinite(derivative_upper)
            and (derivative_lower > 0.0 or derivative_upper < 0.0)
        )
        first_sign = endpoint_sign(lower)
        second_sign = endpoint_sign(upper)
        if monotone and first_sign != 0 and second_sign != 0:
            if first_sign == second_sign:
                continue
            root_lower, root_upper = lower, upper
            while True:
                (
                    root_value_lower,
                    root_value_upper,
                    root_derivative_lower,
                    root_derivative_upper,
                    root_x_lower,
                    root_x_upper,
                ) = enclosure(root_lower, root_upper)
                if root_value_lower > 0.0 or root_value_upper < 0.0:
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "Interval refinement excludes a previously bracketed restricted-Delaunay root.",
                        "restricted_root_isolation",
                    )
                if not (
                    math.isfinite(root_derivative_lower)
                    and math.isfinite(root_derivative_upper)
                    and (root_derivative_lower > 0.0 or root_derivative_upper < 0.0)
                ):
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "A restricted-Delaunay root bracket must retain a strict interval derivative sign.",
                        "restricted_root_isolation",
                    )
                extent = np.nextafter(np.abs(root_x_upper - root_x_lower), np.inf)
                diameter = float(np.nextafter(math.hypot(*extent.tolist()), np.inf))
                if diameter <= spatial_resolution:
                    if not np.all(
                        np.isfinite(
                            np.asarray(
                                (
                                    root_value_lower,
                                    root_value_upper,
                                    root_derivative_lower,
                                    root_derivative_upper,
                                ),
                                dtype=np.float64,
                            )
                        )
                    ):
                        raise _failure(
                            MeshingFailureCategory.COMPLIANCE_FAILED,
                            "Restricted dual root intervals must retain finite outward field and derivative bounds.",
                            "restricted_root_isolation",
                        )
                    roots.append(
                        (
                            np.asarray((root_lower, root_upper), dtype=np.float64),
                            np.stack((root_x_lower, root_x_upper)),
                            np.asarray(
                                (root_value_lower, root_value_upper), dtype=np.float64
                            ),
                            np.asarray(
                                (root_derivative_lower, root_derivative_upper),
                                dtype=np.float64,
                            ),
                            1 if root_derivative_lower > 0.0 else -1,
                        )
                    )
                    break
                midpoint = root_lower + 0.5 * (root_upper - root_lower)
                if midpoint <= root_lower or midpoint >= root_upper:
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "Restricted dual root isolation reached binary64 parameter resolution before meeting the physical root tolerance.",
                        "restricted_root_isolation",
                    )
                # Opposite endpoint signs prove a root; interval Newton contracts
                # even when a bisection midpoint itself encloses that root.
                midpoint_bounds = enclosure(midpoint, midpoint)
                newton_image = interval_subtract(
                    (np.asarray(midpoint), np.asarray(midpoint)),
                    interval_divide(
                        (
                            np.asarray(midpoint_bounds[0]),
                            np.asarray(midpoint_bounds[1]),
                        ),
                        (
                            np.asarray(root_derivative_lower),
                            np.asarray(root_derivative_upper),
                        ),
                    ),
                )
                image_lower = float(np.asarray(newton_image[0]))
                image_upper = float(np.asarray(newton_image[1]))
                new_lower = max(root_lower, image_lower)
                new_upper = min(root_upper, image_upper)
                if new_lower > new_upper:
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "Interval Newton image is disjoint from a bracketed restricted-Delaunay root.",
                        "restricted_root_isolation",
                    )
                if (
                    math.isfinite(new_lower)
                    and math.isfinite(new_upper)
                    and new_upper - new_lower < root_upper - root_lower
                ):
                    root_lower, root_upper = new_lower, new_upper
                    continue
                candidate_parameters = (
                    midpoint,
                    root_lower + (root_upper - root_lower) / 3.0,
                    root_lower + 2.0 * (root_upper - root_lower) / 3.0,
                    root_lower + (root_upper - root_lower) / 4.0,
                    root_lower + 3.0 * (root_upper - root_lower) / 4.0,
                )
                selected_parameter: float | None = None
                selected_sign = 0
                for candidate in candidate_parameters:
                    if not root_lower < candidate < root_upper:
                        continue
                    sign = endpoint_sign(candidate)
                    if sign != 0:
                        selected_parameter, selected_sign = candidate, sign
                        break
                if selected_parameter is None:
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "A bracketed restricted-Delaunay root cannot be narrowed with certified point signs.",
                        "restricted_root_isolation",
                    )
                root_is_left = (
                    selected_sign > 0
                    if root_derivative_lower > 0.0
                    else selected_sign < 0
                )
                if root_is_left:
                    root_upper = selected_parameter
                else:
                    root_lower = selected_parameter
            continue
        if monotone and first_sign != 0 and second_sign != 0:
            continue
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint <= lower or midpoint >= upper:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted dual root isolation reached binary64 parameter resolution before proving a unique transverse intersection.",
                "restricted_root_isolation",
            )
        split = midpoint
        if endpoint_sign(midpoint) == 0:
            selected_split = None
            for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0, 3.0 / 4.0):
                candidate = lower + fraction * (upper - lower)
                if not lower < candidate < upper:
                    continue
                if endpoint_sign(candidate) != 0:
                    selected_split = candidate
                    break
            if selected_split is None:
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Restricted dual root isolation could not split an interval using a certified source-field sign.",
                    "restricted_root_isolation",
                )
            split = selected_split
        pending.append((split, upper))
        pending.append((lower, split))
    roots.sort(key=lambda item: float(item[0][0]))
    return tuple(roots)


def _root_edge_bank(
    profile: AnalyticImplicitProfile,
    endpoint_bounds: np.ndarray,
    spatial_resolution: float,
    /,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...], ...]:
    if not profile.field_bounds.gradient_rigorous:
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted dual roots require the profile's rigorous interval-gradient enclosure.",
            "restricted_root_isolation",
        )

    bounds = np.asarray(endpoint_bounds, dtype=np.float64)
    if bounds.ndim != 4 or bounds.shape[1:] != (2, 2, 3):
        raise ValueError(
            "Restricted root-edge endpoints require shape (edge, endpoint, bound, axis)."
        )
    roots_by_edge: list[
        tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...]
    ] = [()] * bounds.shape[0]
    if not bounds.shape[0]:
        return tuple(roots_by_edge)

    from ..geometry._interval_enclosure import (
        interval_add,
        interval_multiply,
    )

    for start in range(0, bounds.shape[0], _ROOT_EDGE_BATCH_SIZE):
        stop = min(start + _ROOT_EDGE_BATCH_SIZE, bounds.shape[0])
        block = bounds[start:stop]
        lower, upper, direction_lower, direction_upper = _segment_interval_bank(
            block,
            np.zeros((block.shape[0],), dtype=np.float64),
            np.ones((block.shape[0],), dtype=np.float64),
        )
        whole_segment = (lower, upper)
        direction = (direction_lower, direction_upper)
        whole_values = profile.field_bounds.boxes(whole_segment[0], whole_segment[1])
        value_lower = np.asarray(whole_values.value_lower, dtype=np.float64)
        value_upper = np.asarray(whole_values.value_upper, dtype=np.float64)
        possible = ~((value_lower > 0.0) | (value_upper < 0.0))
        possible_rows = np.flatnonzero(possible.reshape((-1,)))
        if not possible_rows.size:
            continue

        candidate_bounds = block[possible_rows]
        candidate_count = possible_rows.size
        endpoint_lower = np.concatenate(
            (candidate_bounds[:, 0, 0], candidate_bounds[:, 1, 0]), axis=0
        )
        endpoint_upper = np.concatenate(
            (candidate_bounds[:, 0, 1], candidate_bounds[:, 1, 1]), axis=0
        )
        endpoint_values = profile.field_bounds.boxes(endpoint_lower, endpoint_upper)
        endpoint_value_lower = np.asarray(
            endpoint_values.value_lower, dtype=np.float64
        ).reshape((-1,))
        endpoint_value_upper = np.asarray(
            endpoint_values.value_upper, dtype=np.float64
        ).reshape((-1,))
        signs = np.zeros((2 * candidate_count,), dtype=np.int8)
        signs[endpoint_value_upper < 0.0] = -1
        signs[endpoint_value_lower > 0.0] = 1

        products = interval_multiply(
            (
                whole_values.gradient_lower[possible_rows],
                whole_values.gradient_upper[possible_rows],
            ),
            (
                direction[0][possible_rows],
                direction[1][possible_rows],
            ),
        )
        product_lower = np.asarray(products[0], dtype=np.float64)
        product_upper = np.asarray(products[1], dtype=np.float64)
        derivative_lower = product_lower[:, 0]
        derivative_upper = product_upper[:, 0]
        for axis in (1, 2):
            summed = interval_add(
                (derivative_lower, derivative_upper),
                (product_lower[:, axis], product_upper[:, axis]),
            )
            derivative_lower = np.asarray(summed[0], dtype=np.float64)
            derivative_upper = np.asarray(summed[1], dtype=np.float64)
        monotone = (
            np.isfinite(derivative_lower)
            & np.isfinite(derivative_upper)
            & ((derivative_lower > 0.0) | (derivative_upper < 0.0))
        )
        start_signs = signs[:candidate_count]
        end_signs = signs[candidate_count:]
        no_root = monotone & (start_signs != 0) & (start_signs == end_signs)
        crossing = (
            monotone & (start_signs != 0) & (end_signs != 0) & (start_signs != end_signs)
        )
        crossing_positions = np.flatnonzero(crossing)
        if crossing_positions.size:
            crossing_rows = possible_rows[crossing_positions]
            crossing_roots = _isolate_monotone_root_bank(
                profile,
                candidate_bounds[crossing_positions],
                spatial_resolution,
                (
                    value_lower[crossing_rows],
                    value_upper[crossing_rows],
                    derivative_lower[crossing_positions],
                    derivative_upper[crossing_positions],
                    whole_segment[0][crossing_rows],
                    whole_segment[1][crossing_rows],
                ),
            )
            for candidate_position, root in zip(
                crossing_positions.tolist(), crossing_roots, strict=True
            ):
                local_row = int(possible_rows[candidate_position])
                roots_by_edge[start + local_row] = (root,)
        fallback_positions = np.flatnonzero(~no_root & ~crossing)
        for candidate_position in fallback_positions.tolist():
            local_row = int(possible_rows[candidate_position])
            values = (
                float(value_lower[local_row]),
                float(value_upper[local_row]),
                float(derivative_lower[candidate_position]),
                float(derivative_upper[candidate_position]),
                np.asarray(whole_segment[0][local_row], dtype=np.float64),
                np.asarray(whole_segment[1][local_row], dtype=np.float64),
            )
            roots_by_edge[start + local_row] = _root_edge(
                profile,
                candidate_bounds[candidate_position],
                spatial_resolution,
                initial_enclosure=values,
                endpoint_signs=(
                    int(start_signs[candidate_position]),
                    int(end_signs[candidate_position]),
                ),
            )
    return tuple(roots_by_edge)


def _dual_data(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    carrier: np.ndarray,
    spatial_resolution: float,
    /,
) -> tuple[
    RestrictedDual3D,
    np.ndarray,
    np.ndarray,
    tuple[tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ...], ...],
    tuple[int, ...],
]:
    dual = restricted_dual_3d(points, tetrahedra, carrier[0], carrier[1])
    center_bounds, center_status = restricted_centers_3d(points, tetrahedra)
    if np.any(np.asarray(center_status) != int(MeshcoreStatus.OK)) or np.any(
        np.asarray(dual.status) != int(MeshcoreStatus.OK)
    ):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Every native restricted-Delaunay center and dual row must have a successful numerical status before interval proof.",
            "restricted_dual_query",
        )
    hull_rows = np.flatnonzero(np.asarray(dual.cells)[:, 1] < 0)
    ray_bounds = np.empty((0, 2, 2, 3), dtype=np.float64)
    if hull_rows.size:
        ray_bounds, _ray_kinds, ray_status = restricted_rays_3d(
            points,
            tetrahedra,
            center_bounds,
            np.asarray(dual.facets)[hull_rows],
            np.asarray(dual.cells)[hull_rows],
            carrier[0],
            carrier[1],
        )
        if np.any(np.asarray(ray_status) != int(MeshcoreStatus.OK)):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Every native restricted-Delaunay carrier ray must have a successful numerical status before interval proof.",
                "restricted_dual_query",
            )
    hull_position = {
        int(row): position for position, row in enumerate(hull_rows.tolist())
    }
    endpoint_bounds = np.empty(
        (np.asarray(dual.facets).shape[0], 2, 2, 3), dtype=np.float64
    )
    for row, cells in enumerate(np.asarray(dual.cells, dtype=np.int64).tolist()):
        if cells[1] >= 0:
            endpoint_bounds[row] = center_bounds[cells]
        else:
            endpoint_bounds[row] = ray_bounds[hull_position[row]]
    roots_by_row = _root_edge_bank(
        profile,
        endpoint_bounds,
        spatial_resolution,
    )
    root_counts = tuple(len(roots) for roots in roots_by_row)
    return (
        dual,
        np.asarray(center_bounds),
        endpoint_bounds,
        tuple(roots_by_row),
        root_counts,
    )


def _classify_sides(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    classification: np.ndarray,
    face_incidence: dict[tuple[int, int, int], list[int]],
    selected_keys: set[tuple[int, int, int]],
    /,
) -> np.ndarray:
    inside_seed_rows = _inside_seed_rows(profile, points, tetrahedra)
    hull_rows = {rows[0] for key, rows in face_incidence.items() if len(rows) == 1}
    if not hull_rows or any(classification[row] != _CLASS_OUTSIDE for row in hull_rows):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Every carrier-hull tetrahedron must have a strict outside field enclosure.",
            "restricted_topology",
        )
    adjacency: list[list[int]] = [[] for _ in range(tetrahedra.shape[0])]
    for key, rows in face_incidence.items():
        if len(rows) != 2 or key in selected_keys:
            continue
        first, second = rows
        adjacency[first].append(second)
        adjacency[second].append(first)
    labels = np.full((tetrahedra.shape[0],), -1, dtype=np.int8)
    seeds = tuple((row, _SIDE_INTERIOR) for row in inside_seed_rows) + tuple(
        (row, _SIDE_EXTERIOR) for row in sorted(hull_rows)
    )
    for seed, side in seeds:
        if labels[seed] not in (-1, side):
            raise _failure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "Restricted source facets must separate the authored interior witness from every carrier-hull component.",
                "restricted_topology",
            )
        queue = deque((seed,))
        while queue:
            row = queue.popleft()
            if labels[row] not in (-1, side):
                raise _failure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "Restricted source facets produce contradictory primal-region flood labels.",
                    "restricted_topology",
                )
            if labels[row] == side:
                continue
            labels[row] = side
            queue.extend(adjacency[row])
    if np.any(labels < 0):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Every raw Delaunay component must be reached from the profile interior witness or the carrier-hull exterior.",
            "restricted_topology",
        )
    if np.any((classification == _CLASS_INSIDE) & (labels != _SIDE_INTERIOR)) or np.any(
        (classification == _CLASS_OUTSIDE) & (labels != _SIDE_EXTERIOR)
    ):
        raise _failure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Primal region flooding must agree with all strict analytic whole-tetrahedron enclosures.",
            "restricted_topology",
        )
    return labels


def _orient_facets(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    facets: np.ndarray,
    cells: np.ndarray,
    labels: np.ndarray,
    /,
) -> np.ndarray:
    oriented_rows = np.asarray(facets, dtype=np.int64).reshape((-1, 3)).tolist()
    owner_rows = np.asarray(cells, dtype=np.int64).reshape((-1, 2)).tolist()
    tetrahedron_rows = np.asarray(tetrahedra, dtype=np.int64).reshape((-1, 4)).tolist()
    for row, (face_row, owners_row) in enumerate(
        zip(oriented_rows, owner_rows, strict=True)
    ):
        face = (int(face_row[0]), int(face_row[1]), int(face_row[2]))
        owners = (int(owners_row[0]), int(owners_row[1]))
        if owners[1] < 0:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A restricted source facet must separate two finite primal tetrahedra.",
                "restricted_topology",
            )
        if labels[owners[0]] == labels[owners[1]]:
            raise _failure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "Each selected restricted source facet must separate the flooded interior and exterior.",
                "restricted_topology",
            )
        inside_cell = owners[0] if labels[owners[0]] == _SIDE_INTERIOR else owners[1]
        face_vertices = set(face)
        apex_candidates = [
            int(value)
            for value in tetrahedron_rows[inside_cell]
            if int(value) not in face_vertices
        ]
        if len(apex_candidates) != 1:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted source facets must have one opposite primal tetrahedron vertex.",
                "restricted_topology",
            )
        apex = apex_candidates[0]
        sign = int(
            np.asarray(
                exact_orient3d(
                    points[face[0]],
                    points[face[1]],
                    points[face[2]],
                    points[apex],
                )
            )
        )
        if sign == 0:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Restricted source facets must be nondegenerate primal triangles.",
                "restricted_topology",
            )
        if sign > 0:
            oriented_rows[row] = [face[0], face[2], face[1]]
    return np.asarray(oriented_rows, dtype=np.int64).reshape((-1, 3))


def _topology_matches(
    profile: AnalyticImplicitProfile,
    oriented_facets: np.ndarray,
    /,
) -> bool:
    facets = np.asarray(oriented_facets, dtype=np.int64)
    if not facets.size:
        return False
    vertices, inverse = np.unique(facets, return_inverse=True)
    topology = TriangleTopology(
        jnp.asarray(inverse.reshape(facets.shape)),
        num_vertices=vertices.size,
    )
    return (
        topology.watertight
        and topology.num_face_components == profile.component_count
        and topology.euler_characteristic == profile.boundary_euler_characteristic
    )


def _facet_deviation(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    facets: np.ndarray,
    /,
) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    faces = np.asarray(facets, dtype=np.int64)
    used, inverse = np.unique(faces, return_inverse=True)
    value_lower, value_upper = profile.field_bounds.point_values(vertices[used])
    vertex_error = np.maximum(np.abs(value_lower), np.abs(value_upper))
    triangles = vertices[faces]
    diameter = np.maximum.reduce(
        tuple(
            np.nextafter(
                np.linalg.norm(
                    np.nextafter(triangles[:, first] - triangles[:, second], np.inf),
                    axis=1,
                ),
                np.inf,
            )
            for first, second in ((0, 1), (1, 2), (2, 0))
        )
    )
    error = np.max(vertex_error[inverse.reshape(faces.shape)], axis=1)
    in_tube = np.nextafter(error + diameter, np.inf) < profile.tube_radius
    interpolation = np.nextafter(
        error
        + np.nextafter(
            profile.hessian_upper * np.nextafter(diameter * diameter, np.inf) / 6.0,
            np.inf,
        ),
        np.inf,
    )
    boxes = profile.field_bounds.boxes(
        np.min(triangles, axis=1), np.max(triangles, axis=1)
    )
    box_bound = np.maximum(np.abs(boxes.value_lower), np.abs(boxes.value_upper))
    return np.where(in_tube, np.minimum(box_bound, interpolation), box_bound)


def _tetra_quality(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    center_bounds: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(points, dtype=np.float64)[
        np.asarray(tetrahedra, dtype=np.int64)
    ]
    centers = np.asarray(center_bounds, dtype=np.float64)
    center_lower, center_upper = centers[:, 0], centers[:, 1]
    center_mid = center_lower + 0.5 * (center_upper - center_lower)
    center_radius = np.linalg.norm(
        np.maximum(
            np.abs(center_mid[:, None, :] - vertices),
            np.abs(center_upper[:, None, :] - vertices),
        ),
        axis=2,
    )
    center_radius = np.maximum(
        center_radius,
        np.linalg.norm(center_lower[:, None, :] - vertices, axis=2),
    )
    radius_upper = np.nextafter(np.max(center_radius, axis=1), np.inf)
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
    )
    edge_vectors = vertices[:, edge_pairs[:, 0]] - vertices[:, edge_pairs[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    edge_lower = np.nextafter(
        np.min(np.nextafter(edge_lengths, -np.inf), axis=1), -np.inf
    )
    if np.any(edge_lower <= 0.0) or np.any(~np.isfinite(radius_upper)):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron edge and circumradius bounds must remain finite and positive.",
            "specification_compliance",
        )
    radius_edge = np.nextafter(radius_upper / edge_lower, np.inf)
    face_vertices = vertices[:, np.asarray(_FACE_CORNERS, dtype=np.int64)]
    first_edges = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    second_edges = face_vertices[:, :, 2] - face_vertices[:, :, 0]
    normal_scale = np.maximum(
        np.max(np.abs(first_edges), axis=2),
        np.max(np.abs(second_edges), axis=2),
    )
    if np.any(~np.isfinite(normal_scale)) or np.any(normal_scale <= 0.0):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron face-normal quality is unresolved at binary64 scale.",
            "specification_compliance",
        )
    face_normals = np.cross(
        first_edges / normal_scale[:, :, None],
        second_edges / normal_scale[:, :, None],
    )
    normal_lengths = np.linalg.norm(face_normals, axis=2)
    if np.any(~np.isfinite(normal_lengths)) or np.any(normal_lengths <= 0.0):
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted tetrahedron dihedral quality is unresolved at binary64 normal precision.",
            "specification_compliance",
        )
    unit = face_normals / normal_lengths[:, :, None]
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64)
    cosine = np.sum(unit[:, pairs[:, 0]] * unit[:, pairs[:, 1]], axis=2)
    dihedral = 180.0 - np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return radius_edge, np.min(dihedral, axis=1)


def _strict_region_site_mask(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    carrier: np.ndarray,
    /,
) -> np.ndarray:
    candidates = np.asarray(points, dtype=np.float64)
    lower, upper = profile.field_bounds.point_values(candidates)
    inside_carrier = np.all(
        (candidates > carrier[0]) & (candidates < carrier[1]),
        axis=1,
    )
    return inside_carrier & ((upper < 0.0) | (lower > 0.0))


def _face_parity(reference: np.ndarray, oriented: np.ndarray, /) -> int:
    positions = tuple(int(np.flatnonzero(reference == value)[0]) for value in oriented)
    inversions = sum(
        positions[first] > positions[second]
        for first in range(3)
        for second in range(first + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _build_mesh(
    profile: AnalyticImplicitProfile,
    points: np.ndarray,
    point_ids: np.ndarray,
    tetrahedra: np.ndarray,
    tetrahedron_ids: np.ndarray,
    inside_rows: np.ndarray,
    selected_facets: np.ndarray,
    selected_cells: np.ndarray,
    dual_facets: np.ndarray,
    flood_labels: np.ndarray,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray]:
    inside_rows = np.asarray(inside_rows, dtype=np.int64)
    used_points = np.unique(
        np.asarray(tetrahedra, dtype=np.int64)[inside_rows].reshape((-1,))
    )
    reindex = np.full((points.shape[0],), -1, dtype=np.int64)
    reindex[used_points] = np.arange(used_points.size, dtype=np.int64)
    target_cells = reindex[np.asarray(tetrahedra, dtype=np.int64)[inside_rows]].astype(
        np.int32, copy=False
    )
    cell_ids = np.asarray(tetrahedron_ids, dtype=np.int64)[inside_rows]
    if np.unique(cell_ids).size != cell_ids.size:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted volume cells require unique stable identities inherited from raw primal tetrahedra.",
            "restricted_mesh_assembly",
        )
    mesh = CellMesh.from_tetrahedra(
        points[used_points],
        target_cells,
        vertex_global_ids=point_ids[used_points],
        cell_global_ids=cell_ids,
        numeric_version=profile.source_revision,
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "A mesh created from restricted tetrahedra must retain tetrahedral connectivity.",
            "restricted_mesh_assembly",
        )
    face_array = np.asarray(connectivity.faces, dtype=np.int64)
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    boundary_faces = face_array[boundary_mask]
    boundary_face_rows = np.flatnonzero(boundary_mask)
    cell_faces = (
        np.asarray(connectivity.cell_faces, dtype=np.int64).reshape((-1, 4)).tolist()
    )
    cell_face_signs = (
        np.asarray(connectivity.cell_face_signs, dtype=np.int64).reshape((-1, 4)).tolist()
    )
    face_owner_signs = np.zeros((face_array.shape[0],), dtype=np.int8)
    for cell_row, cell_face_rows in enumerate(cell_faces):
        for local_face, face_row in enumerate(cell_face_rows):
            if boundary_mask[face_row]:
                if face_owner_signs[face_row] != 0:
                    raise _failure(
                        MeshingFailureCategory.INVALID_SOURCE,
                        "A target boundary face must have exactly one oriented interior-cell incidence.",
                        "restricted_mesh_assembly",
                    )
                face_owner_signs[face_row] = int(cell_face_signs[cell_row][local_face])
    if np.any(np.abs(face_owner_signs[boundary_mask]) != 1):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Target exterior faces require signed cell-boundary incidence.",
            "restricted_mesh_assembly",
        )
    boundary_face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)[
        boundary_mask
    ]
    selected_rows = np.asarray(selected_facets, dtype=np.int64).reshape((-1, 3)).tolist()
    selected_by_key = {
        _triangle_key(int(face[0]), int(face[1]), int(face[2])): position
        for position, face in enumerate(selected_rows)
    }
    raw_oriented = _orient_facets(
        points,
        tetrahedra,
        dual_facets,
        selected_cells,
        flood_labels,
    )
    oriented_rows = raw_oriented.tolist()
    oriented_by_key = {
        _triangle_key(int(face[0]), int(face[1]), int(face[2])): face
        for face in oriented_rows
    }
    local_oriented: list[np.ndarray] = []
    matched_facets = 0
    for boundary_face_row, face_values in zip(
        boundary_face_rows.tolist(),
        np.asarray(boundary_faces, dtype=np.int64).reshape((-1, 3)).tolist(),
        strict=True,
    ):
        raw_face = used_points[np.asarray(face_values, dtype=np.int64)]
        key = _triangle_key(int(raw_face[0]), int(raw_face[1]), int(raw_face[2]))
        source_position = selected_by_key.get(key)
        if source_position is None or key not in oriented_by_key:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Flooded target exterior faces must be exactly the selected restricted-Delaunay source facets.",
                "restricted_mesh_assembly",
            )
        source_facet = np.asarray(selected_facets[source_position], dtype=np.int64)
        oriented_raw = np.asarray(oriented_by_key[key], dtype=np.int64)
        local_oriented.append(reindex[oriented_raw].astype(np.int32, copy=False))
        matched_facets += 1
        target_orientation = int(face_owner_signs[boundary_face_row]) * _face_parity(
            source_facet, raw_face
        )
        surface_orientation = _face_parity(source_facet, oriented_raw)
        if target_orientation != surface_orientation:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The target cell-boundary chain and outward restricted source surface must share orientation.",
                "restricted_mesh_assembly",
            )
    if matched_facets != np.asarray(selected_facets).shape[0]:
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Every selected restricted facet must appear exactly once on the flooded target boundary.",
            "restricted_mesh_assembly",
        )
    return mesh, boundary_face_ids, np.asarray(local_oriented, dtype=np.int32)


def _native_fixed_boundary_quality_pass(
    mesh: CellMesh,
    boundary_faces: np.ndarray,
    profile: AnalyticImplicitProfile,
    schedule: NativeVolumeSchedule,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray]:
    """Improve a source-constrained tetrahedral state and rebind its unchanged boundary."""
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "tetrahedron":
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Restricted analytic quality improvement requires one tetrahedral block.",
            "restricted_quality_preparation",
        )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    tetrahedra = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    facets = np.asarray(boundary_faces, dtype=np.int32).reshape((-1, 3))
    if (
        facets.shape[0] == 0
        or facets.size
        and (np.any(facets < 0) or np.any(facets >= coordinates.shape[0]))
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted analytic boundary facets must index the current target mesh.",
            "restricted_quality_preparation",
        )
    if facets.shape[0] > np.iinfo(np.int32).max:
        raise OverflowError(
            "Restricted analytic facet labels exceed native int32 indices."
        )
    if np.any(vertex_ids < 0) or np.any(cell_ids < 0):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Restricted analytic quality state requires nonnegative stable vertex and cell IDs.",
            "restricted_quality_preparation",
        )
    face_sources = np.arange(facets.shape[0], dtype=np.int32)
    face_orientations = np.ones((facets.shape[0],), dtype=np.int8)
    segments = np.empty((0, 2), dtype=np.int32)
    segment_sources = np.empty((0,), dtype=np.int32)
    regions = np.zeros((tetrahedra.shape[0],), dtype=np.int32)
    expected_faces_by_source = {
        int(source): tuple(sorted(int(vertex_ids[value]) for value in face))
        for source, face in zip(face_sources.tolist(), facets.tolist(), strict=True)
    }
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
                "restricted_quality_refinement",
            )
        if refinement.status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native fixed-boundary radius refinement failed with {refinement.status.name}.",
                "restricted_quality_refinement",
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
                "restricted_quality_improvement",
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
                "restricted_quality_improvement",
            )
        if improvement.status is not MeshcoreStatus.OK or improvement.unmet_cells != 0:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Native joint quality improvement returned inconsistent success status "
                f"{improvement.status.name} with {improvement.unmet_cells} unmet cell(s).",
                "restricted_quality_improvement",
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
            "restricted_quality_materialization",
        )
    actual_faces_by_source = {
        int(source): tuple(sorted(int(native_vertex_ids[value]) for value in face))
        for source, face in zip(
            np.asarray(arrays.face_sources, dtype=np.int64).tolist(),
            np.asarray(arrays.faces, dtype=np.int64).tolist(),
            strict=True,
        )
    }
    if actual_faces_by_source != expected_faces_by_source:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native joint quality edits changed a fixed restricted-Delaunay source facet.",
            "restricted_quality_materialization",
        )
    output = CellMesh.from_tetrahedra(
        arrays.points,
        arrays.tetrahedra,
        vertex_global_ids=np.asarray(native_vertex_ids, dtype=np.int64),
        cell_global_ids=np.asarray(native_cell_ids, dtype=np.int64),
        numeric_version=profile.source_revision,
    )
    connectivity = output.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native joint quality output lost tetrahedral connectivity.",
            "restricted_quality_materialization",
        )
    output_vertex_ids = np.asarray(output.vertex_global_ids, dtype=np.int64)
    output_row_by_id = {
        int(identifier): row for row, identifier in enumerate(output_vertex_ids.tolist())
    }
    source_boundary_ids = vertex_ids[facets]
    try:
        output_boundary_faces = np.asarray(
            [
                [output_row_by_id[int(identifier)] for identifier in face]
                for face in source_boundary_ids.tolist()
            ],
            dtype=np.int32,
        )
    except KeyError as error:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native fixed-boundary improvement removed an authored boundary vertex.",
            "restricted_quality_materialization",
        ) from error
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    output_faces = np.asarray(connectivity.faces, dtype=np.int64)
    expected_boundary = {
        tuple(sorted(int(value) for value in face))
        for face in source_boundary_ids.tolist()
    }
    actual_boundary = {
        tuple(sorted(int(output_vertex_ids[value]) for value in face))
        for face in output_faces[boundary_mask].tolist()
    }
    if actual_boundary != expected_boundary:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native fixed-boundary improvement changed the embedded target boundary.",
            "restricted_quality_materialization",
        )
    face_ids = np.asarray(output.entity_set(2).entity_ids, dtype=np.int64)
    boundary_face_ids = face_ids[boundary_mask]
    return output, boundary_face_ids, output_boundary_faces


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
            "restricted_mesh_assembly",
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
                "geometry-owned-restricted-delaunay-implicit-volume-boundary",
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
    radius_edge, dihedral = _tetra_quality(coordinates, cells, centers)
    maximum_radius_edge = float(np.max(radius_edge, initial=0.0))
    minimum_dihedral = float(np.min(dihedral, initial=180.0))
    requested: list[tuple[str, float]] = []
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", schedule.radius_edge_bound))
    requested.extend((
        ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
        ("source_fidelity_tolerance", fidelity_tolerance),
    ))
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
    if schedule.radius_edge_bound is not None and maximum_radius_edge > schedule.radius_edge_bound:
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


def _restricted_primal_mesh(
    profile: AnalyticImplicitProfile,
    specification: VolumeMeshingSpec,
    carrier_bounds: np.ndarray,
    fidelity_tolerance: float,
    samples: SourceBoundarySamples,
    site_points: np.ndarray,
    site_ids: np.ndarray,
    sample_input_rows: np.ndarray,
    spatial_resolution: float,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray] | Literal["cover", "grid"]:
    """Assemble one restricted-Delaunay primal mesh, or name the refinement it needs.

    `"cover"` asks for a finer source cover and `"grid"` for a finer site grid.
    The Delaunay, dual, flood and incidence arrays are local to this attempt and
    are released when it returns; only the assembled volume mesh survives.
    """
    points, point_ids, tetrahedra, sample_rows = _delaunay_snapshot(
        site_points, site_ids, sample_input_rows
    )
    if sample_rows.shape != (samples.points.shape[0],):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Every analytic cover sample must retain one native Delaunay vertex row.",
            "restricted_delaunay",
        )
    for sample_row, primal_row in enumerate(sample_rows.tolist()):
        if not np.array_equal(points[primal_row], samples.points[sample_row]):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Every analytic source cover sample must map through the actual native Delaunay vertex map.",
                "restricted_delaunay",
            )
    dual, _, _, _, root_counts = _dual_data(
        profile,
        points,
        tetrahedra,
        carrier_bounds,
        spatial_resolution,
    )
    raw_facets = np.asarray(dual.facets, dtype=np.int64)
    raw_cells = np.asarray(dual.cells, dtype=np.int64)
    del dual
    if raw_facets.shape[0] != len(root_counts):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Restricted dual rows and interval root records must align exactly.",
            "restricted_dual_query",
        )
    face_incidence = _tetrahedron_faces(tetrahedra)
    for face, owners in zip(raw_facets.tolist(), raw_cells.tolist(), strict=True):
        actual = face_incidence.get(
            _triangle_key(int(face[0]), int(face[1]), int(face[2]))
        )
        expected = tuple(sorted(int(value) for value in owners if value >= 0))
        if actual is None or tuple(sorted(actual)) != expected:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Native restricted dual candidates must retain their actual primal facet and incident tetrahedra.",
                "restricted_dual_query",
            )
    selected_rows = np.asarray(
        [row for row, count in enumerate(root_counts) if count == 1],
        dtype=np.int64,
    )
    selected_facets = raw_facets[selected_rows]
    selected_cells = raw_cells[selected_rows]
    if any(count > 1 for count in root_counts):
        return "cover"
    if selected_rows.size:
        # Root-certified facets may use strict-side Steiner sites;
        # complete source coverage is a separate proof.
        selected_points = points[selected_facets]
        face_diameters = np.maximum.reduce(
            tuple(
                np.nextafter(
                    np.linalg.norm(
                        np.nextafter(
                            selected_points[:, first] - selected_points[:, second],
                            np.inf,
                        ),
                        axis=1,
                    ),
                    np.inf,
                )
                for first, second in ((0, 1), (1, 2), (2, 0))
            )
        )
        if np.any(face_diameters >= profile.reach_lower / 4.0):
            return "cover"
    selected_keys = {
        _triangle_key(int(face[0]), int(face[1]), int(face[2]))
        for face in selected_facets.tolist()
    }
    classification = _field_box_bounds(profile, points, tetrahedra)[4]
    hull_dual_rows = [
        row for row, count in enumerate(root_counts) if count and raw_cells[row, 1] < 0
    ]
    if hull_dual_rows:
        raise _failure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The analytic source must be enclosed by the restricted primal carrier; a carrier-hull dual ray intersects its zero set.",
            "restricted_topology",
        )
    # Dual-root topology and complete source projection coverage are
    # independent proofs; only hull cells need a strict exterior box.
    try:
        flood_labels = _classify_sides(
            profile,
            points,
            tetrahedra,
            classification,
            face_incidence,
            selected_keys,
        )
        del face_incidence, classification
        if not selected_rows.size:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The analytic profile must have a nonempty restricted-Delaunay zero-set boundary.",
                "restricted_topology",
            )
        for cells, root_count in zip(raw_cells.tolist(), root_counts, strict=True):
            if cells[1] < 0:
                if flood_labels[cells[0]] != _SIDE_EXTERIOR:
                    raise _failure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "Every zero-free restricted carrier ray must remain on the exterior region side.",
                        "restricted_topology",
                    )
                continue
            same_side = flood_labels[cells[0]] == flood_labels[cells[1]]
            if same_side != (root_count == 0):
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Restricted dual root counts must agree with the flooded region-side transition.",
                    "restricted_topology",
                )
        oriented_facets = _orient_facets(
            points,
            tetrahedra,
            selected_facets,
            selected_cells,
            flood_labels,
        )
    except MeshingFailure as error:
        return "grid" if error.stage == "restricted_seed_star" else "cover"
    del raw_facets, raw_cells
    if not _topology_matches(profile, oriented_facets):
        return "cover"
    facet_deviation = _facet_deviation(profile, points, oriented_facets)
    if np.any(facet_deviation > fidelity_tolerance):
        return "cover"
    inside_rows = np.flatnonzero(flood_labels == _SIDE_INTERIOR)
    edge_keys = {
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
        for face in selected_facets
        for index in range(3)
    }
    boundary_edges = np.asarray(sorted(edge_keys), dtype=np.int64)
    edge_lengths = np.linalg.norm(
        points[boundary_edges[:, 1]] - points[boundary_edges[:, 0]], axis=1
    )
    size_unmet = False
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
                f"Hard minimum boundary size for control {control.control_id!r} is unmet by the restricted primal facets.",
                "specification_compliance",
            )
        if float(np.max(edge_lengths)) > allowed:
            size_unmet = True
        for statistic in specification.size_compliance.target_statistics:
            quantile = 0.5 if statistic == "p50" else 0.95
            if float(np.quantile(edge_lengths, quantile)) > allowed:
                size_unmet = True
    if size_unmet:
        return "cover"
    primal_tet_ids = np.asarray(
        [
            _stable_id(
                (
                    "restricted-primal-tetrahedron",
                    profile.profile_id,
                    tuple(sorted(int(point_ids[value]) for value in tet)),
                )
            )
            for tet in tetrahedra
        ],
        dtype=np.int64,
    )
    if np.unique(primal_tet_ids).size != primal_tet_ids.size:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Raw Delaunay tetrahedra require unique deterministic ancestry identities.",
            "restricted_mesh_assembly",
        )
    return _build_mesh(
        profile,
        points,
        point_ids,
        tetrahedra,
        primal_tet_ids,
        inside_rows,
        selected_facets,
        selected_cells,
        selected_facets,
        flood_labels,
    )


@final
class PreparedRestrictedImplicitVolume(StrictModule, NonTrainableState):
    """Prepared analytic-profile volume route using restricted Delaunay."""

    profile: AnalyticImplicitProfile
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    coordinate_contract: SpatialCoordinateContract
    fidelity_tolerance: float = eqx.field(static=True)
    seed_cover_radius: float = eqx.field(static=True)
    carrier_bounds: np.ndarray
    spacing: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: AnalyticImplicitProfile,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        coordinate_contract: SpatialCoordinateContract,
        fidelity_tolerance: float,
        seed_cover_radius: float,
        carrier_bounds: np.ndarray,
        spacing: float,
        prepared_id: str,
        /,
    ) -> None:
        self.profile = profile
        self.specification = specification
        self.schedule = schedule
        self.coordinate_contract = coordinate_contract
        self.fidelity_tolerance = float(fidelity_tolerance)
        self.seed_cover_radius = float(seed_cover_radius)
        carrier = np.array(carrier_bounds, dtype=np.float64, copy=True)
        carrier.setflags(write=False)
        self.carrier_bounds = carrier
        self.spacing = float(spacing)
        self.prepared_id = prepared_id

    def execute(self, /) -> NativeMeshConstruction:
        profile = self.profile
        specification = self.specification
        profile.require_bound(profile.geometry, self.coordinate_contract)
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
        site_points, site_ids, sample_input_rows = _initial_sites(
            profile,
            self.carrier_bounds,
            samples,
            self.spacing,
            self.seed_cover_radius,
        )
        spatial_resolution = min(
            self.seed_cover_radius / 8.0, self.fidelity_tolerance / 32.0
        )
        iteration = 0

        cover_radius = self.seed_cover_radius
        spacing = self.spacing

        def refresh_cover() -> None:
            nonlocal samples, cover_radius, site_points, site_ids, sample_input_rows
            nonlocal spatial_resolution, iteration
            next_radius = cover_radius / 2.0
            if (
                not math.isfinite(next_radius)
                or next_radius <= 0.0
                or next_radius >= cover_radius
            ):
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "The restricted analytic source cover reached binary64 physical-radius resolution.",
                    "restricted_refinement",
                )
            updated = profile.boundary_cover(next_radius)
            if (
                not updated.complete
                or updated.semantics != "certified"
                or updated.points.shape[0] == 0
                or np.any(updated.covering_radius > next_radius)
                or (
                    np.array_equal(updated.points, samples.points)
                    and np.array_equal(updated.covering_radius, samples.covering_radius)
                )
            ):
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "The complete analytic source cover cannot be refined at its requested physical radius.",
                    "restricted_refinement",
                )
            samples = updated
            cover_radius = next_radius
            site_points, site_ids, sample_input_rows = _initial_sites(
                profile, self.carrier_bounds, samples, spacing, cover_radius
            )
            spatial_resolution = min(cover_radius / 8.0, self.fidelity_tolerance / 32.0)
            iteration += 1

        def refine_grid() -> None:
            nonlocal spacing, site_points, site_ids, sample_input_rows, iteration
            nonlocal samples, cover_radius
            next_spacing = spacing / 2.0
            if (
                not math.isfinite(next_spacing)
                or next_spacing <= 0.0
                or next_spacing >= spacing
            ):
                raise _failure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "The restricted volume site spacing reached binary64 representability.",
                    "restricted_refinement",
                )
            spacing = next_spacing
            site_points, site_ids, sample_input_rows = _initial_sites(
                profile, self.carrier_bounds, samples, spacing, cover_radius
            )
            iteration += 1

        while isinstance(
            attempt := _restricted_primal_mesh(
                profile,
                specification,
                self.carrier_bounds,
                self.fidelity_tolerance,
                samples,
                site_points,
                site_ids,
                sample_input_rows,
                spatial_resolution,
            ),
            str,
        ):
            if attempt == "cover":
                refresh_cover()
            else:
                refine_grid()
        sample_count = samples.points.shape[0]
        del samples, site_points, site_ids, sample_input_rows, refresh_cover, refine_grid
        # The primal mesh is only the quality pass input; dropping the attempt
        # releases it as soon as the improved mesh exists.
        primal_mesh, _, primal_boundary_faces = attempt
        del attempt, _
        mesh, target_face_global_ids, boundary_faces = (
            _native_fixed_boundary_quality_pass(
                primal_mesh, primal_boundary_faces, profile, self.schedule
            )
        )
        del primal_mesh, primal_boundary_faces
        if not _topology_matches(profile, boundary_faces):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The native fixed-boundary quality pass changed analytic boundary topology.",
                "restricted_quality_materialization",
            )
        facet_deviation = _facet_deviation(
            profile, np.asarray(mesh.coordinates, dtype=np.float64), boundary_faces
        )
        if np.any(facet_deviation > self.fidelity_tolerance):
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The native fixed-boundary quality pass violated analytic source fidelity.",
                "restricted_quality_materialization",
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
        coverage_witness = RestrictedDelaunayBoundaryWitness(
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
        # the coverage replay and to acceptance, which check its inputs.
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
        embedding = (
            None if validated_embedding is None else validated_embedding.certificate
        )
        validate_restricted_delaunay_boundary_witness(
            mesh,
            geometry,
            profile,
            cell_regions,
            fidelity_witness,
            coverage_witness,
            embedding=validated_embedding,
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
            global_embedding=embedding,
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
    """Prepare the real established-reach restricted-Delaunay profile route."""
    issues = _support_issues(profile, specification)
    if not _shape_targets_gated(schedule):
        # Its native quality pass refuses any unresolved cell before acceptance.
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
    fidelity, cover = _cover_radius(profile, specification)
    target_size = min(
        (
            control.target_size
            for control in specification.size_controls
            if isinstance(control, UniformSizeControl)
        ),
        default=profile.reach_lower,
    )
    spacing = min(target_size, profile.reach_lower / 2.0)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The profile's established reach and requested volume size must admit a finite positive interior spacing.",
            "source_inspection",
        )
    carrier = _carrier_bounds(profile)
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
            "carrier": carrier,
            "spacing": spacing,
        }
    )
    return PreparedRestrictedImplicitVolume(
        profile,
        specification,
        schedule,
        coordinate_contract,
        fidelity,
        cover,
        carrier,
        spacing,
        identity,
    )


__all__ = ["PreparedRestrictedImplicitVolume", "prepare_restricted_implicit_volume"]
