#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry-owned implicit surface extraction and restricted volume preparation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner

from ..._bvh import bvh_overlap_pair_blocks, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    IntervalBoxProgram,
    MeshcoreError,
    zero_set_remesh,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec, PreparedTensorGrid
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry._contracts import CompiledGeometry
from ...geometry._interval_enclosure import (
    IntervalExtensionError,
    prepare_interval_function,
)
from ...geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
    SourceBoundaryProjectionResult,
    SourceBoundaryQuery,
    SourceBoundarySamples,
)
from ...geometry._source_identity import source_entity_id
from ...geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
    discover_implicit_surface,
    ImplicitSurfacePlan,
    ImplicitSurfacePolicy,
    ImplicitVolumeClass,
)
from ...geometry.implicit._adaptive_discovery import AdaptiveImplicitBoundarySource
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...typing import checked
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import (
    FeatureKind,
    PatchControl,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from .._implicit_volume import (
    prepare_restricted_implicit_volume,
    PreparedRestrictedImplicitVolume,
)
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole, RegionRole
from .._plc_source import PiecewiseLinearComplex
from .._result import MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import SizeControlStrength, UniformSizeControl
from .._volume_generation import (
    _shape_targets_gated,
    _soft_shape_target_evidence,
    _split_shape_target_rows,
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from ._native_options import NativeMeshingOptions
from ._native_plc_geometry import simplex_volume_entities, volume_cell_entity_rows
from ._native_sources import NativeImplicitSource, NativePlcSource
from ._native_volume import (
    _boundary_face_entities,
    _faces_connected,
)


# Full-score minimum angle of the carrier zero-set remesh; a face at this angle
# has radius-edge ratio 1, the floor of every FIXED-fill schedule bound.
_CARRIER_MINIMUM_ANGLE_DEGREES = 30.0


def _failure(
    category: MeshingFailureCategory,
    message: str,
    /,
    *,
    provider_code: str,
    stage: str,
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code=provider_code,
        stage=stage,
    )


def _scope_ids(source: NativeImplicitSource, /) -> np.ndarray:
    identifiers = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    if identifiers.shape != (1,):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The native implicit source must expose its one stable zero-set identity.",
            provider_code="native_implicit_identity",
            stage="source_inspection",
        )
    return identifiers


def _scope_matches(
    scope: MeshingScope, source: NativeImplicitSource, identifiers: np.ndarray, /
) -> bool:
    return (
        scope.source_id == source.source_id
        and scope.source_revision == source.source_revision
        and scope.entity_kind is MeshingEntityKind.GEOMETRY
        and scope.entity_dimension == 2
        and scope.entity_set_id == source.scope_entity_set_id(2)
        and np.array_equal(np.asarray(scope.entity_ids, dtype=np.int64), identifiers)
    )


def _region_scope_ids(source: NativeImplicitSource, /) -> np.ndarray:
    identifiers = np.asarray(source.scope_entity_ids(3), dtype=np.int64)
    if identifiers.shape != (1,):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "A bounded native implicit region must expose exactly its authored interior scope.",
            provider_code="native_implicit_region_identity",
            stage="source_inspection",
        )
    return identifiers


def _region_scope_matches(
    scope: MeshingScope, source: NativeImplicitSource, identifiers: np.ndarray, /
) -> bool:
    return (
        scope.source_id == source.source_id
        and scope.source_revision == source.source_revision
        and scope.entity_kind is MeshingEntityKind.GEOMETRY
        and scope.entity_dimension == 3
        and scope.entity_set_id == source.scope_entity_set_id(3)
        and np.array_equal(np.asarray(scope.entity_ids, dtype=np.int64), identifiers)
    )


def _implicit_protected_feature_issues(
    source: NativeImplicitSource,
    features: tuple[ProtectedFeature, ...],
    /,
) -> list[str]:
    identifiers = _scope_ids(source)
    return [
        f"protected feature {feature.feature_id} must name the complete authored implicit surface"
        for feature in features
        if feature.feature_kind is not FeatureKind.SURFACE
        or not _scope_matches(feature.scope, source, identifiers)
    ]


def implicit_support_issues(
    source: NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    """Report request semantics this implicit adaptive surface route cannot honor."""
    unsupported: list[str] = []
    target = specification.target
    family = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*family.required, *family.preferred)) != {"triangle"}
        or family.allow_mixed
        or family.allowed_transitions
    ):
        unsupported.append("affine triangular cells in ambient dimension three")
    if options.route not in (None, "implicit_surface"):
        unsupported.append("the selected implicit_surface native route")
    if options.implicit_policy is not None and not isinstance(
        options.implicit_policy, (AdaptiveImplicitSurfacePolicy, ImplicitSurfacePolicy)
    ):
        unsupported.append("an implicit surface policy")
    if specification.planar_embedding is not None:
        unsupported.append("planar embeddings")
    if specification.background_metric is not None:
        unsupported.append("background metrics")
    if specification.maximum_normal_deviation is not None:
        unsupported.append("maximum-normal-deviation controls")
    unsupported.extend(
        _implicit_protected_feature_issues(source, specification.protected_features)
    )
    if specification.region_controls:
        unsupported.append("region controls")
    if specification.patch_controls:
        unsupported.append("patch controls")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    identifiers = _scope_ids(source)
    if not _scope_matches(specification.scope, source, identifiers):
        unsupported.append("the source-owned implicit zero-set scope")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            unsupported.append(
                f"size-control family {type(control).__name__} outside uniform isotropic sizing"
            )
        elif not _scope_matches(control.scope, source, identifiers):
            unsupported.append("local or foreign-entity size-control scopes")
    return list(dict.fromkeys(unsupported))


def implicit_volume_support_issues(
    source: NativeImplicitSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> list[str]:
    """Report volume controls this implicit-to-PLC carrier cannot enforce."""
    unsupported: list[str] = []
    target = specification.target
    family = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*family.required, *family.preferred)) != {"tetrahedron"}
        or family.allow_mixed
        or family.allowed_transitions
    ):
        unsupported.append("affine tetrahedra in ambient dimension three")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        unsupported.append("the simplex volume-fill strategy")
    if options.route == "surface_envelope_tetrahedral":
        unsupported.append(
            "surface_envelope_tetrahedral requires an explicit reconstructed surface source"
        )
    elif options.route not in (None, "implicit_adaptive_tetrahedral"):
        unsupported.append("implicit_adaptive_tetrahedral")
    if options.implicit_policy is not None and not isinstance(
        options.implicit_policy, AdaptiveImplicitSurfacePolicy
    ):
        unsupported.append("an AdaptiveImplicitSurfacePolicy schedule")
    if options.volume_schedule is not None and not isinstance(
        options.volume_schedule, NativeVolumeSchedule
    ):
        unsupported.append("a NativeVolumeSchedule")
    identifiers = _scope_ids(source)
    if not _scope_matches(specification.boundary_scope, source, identifiers):
        unsupported.append("the source-owned implicit zero-set boundary scope")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            unsupported.append(
                f"volume size-control family {type(control).__name__} outside uniform isotropic sizing"
            )
        elif not _scope_matches(control.scope, source, identifiers):
            unsupported.append("local or foreign-entity implicit boundary size scopes")
    unsupported.extend(
        _implicit_protected_feature_issues(source, specification.protected_features)
    )
    if len(specification.region_controls) > 1:
        unsupported.append(
            "one source-owned region control for the single implicit interior"
        )
    region_identifiers = _region_scope_ids(source)
    controls_by_identity: set[tuple[str, str, RegionRole]] = set()
    for control in specification.region_controls:
        if not _region_scope_matches(control.scope, source, region_identifiers):
            unsupported.append("foreign or partial implicit region-control scope")
        if control.region_name != "interior":
            unsupported.append(
                "a region control matching the source-declared bounded interior"
            )
        if not control.meshing_enabled:
            unsupported.append("meshing the complete bounded implicit interior")
        controls_by_identity.add((control.region_name, control.material_id, control.role))
    if len(controls_by_identity) > 1:
        unsupported.append(
            "one material and role identity for the single implicit interior"
        )
    seed_identities = {
        (seed.region_name, seed.material_id, seed.role)
        for seed in specification.region_seeds
    }
    if any(seed.region_name != "interior" for seed in specification.region_seeds):
        unsupported.append("region seeds naming the source-declared bounded interior")
    if len(seed_identities) > 1:
        unsupported.append(
            "consistent material and role identities for the single implicit region"
        )
    if (
        controls_by_identity
        and seed_identities
        and controls_by_identity != seed_identities
    ):
        unsupported.append("matching implicit region-control and region-seed identities")
    if len(specification.patch_controls) > 1:
        unsupported.append("one patch control for the single implicit zero-set scope")
    for control in specification.patch_controls:
        if not _scope_matches(control.scope, source, identifiers):
            unsupported.append(
                "a patch covering the source-owned implicit zero-set scope"
            )
        if control.adjacent_region_names != ("interior",):
            unsupported.append("patch adjacency to the source-declared implicit interior")
    if specification.hole_seeds:
        unsupported.append(
            "hole seeds without independently classified bounded implicit components"
        )
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    return list(dict.fromkeys(unsupported))


@checked
def normalize_analytic_implicit_source(
    profile: AnalyticImplicitProfile,
    policy: AdaptiveImplicitSurfacePolicy | None,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec | None = None,
) -> NativeImplicitSource:
    """Bind an exact analytic profile to the existing adaptive implicit route.

    The profile is the source of identity and coordinate semantics. Its signed
    compiled geometry remains the discovery authority; a padded analytic
    boundary box supplies the finite search domain without inferring source
    identity from coordinates. With a ``specification``, discovery runs at the
    policy the route resolves for it, so the route reuses this discovery.
    """
    if policy is not None and not isinstance(policy, AdaptiveImplicitSurfacePolicy):
        raise TypeError("policy must be an AdaptiveImplicitSurfacePolicy or None.")
    try:
        profile.require_bound(profile.geometry, coordinate_contract)
    except (TypeError, ValueError, OverflowError) as error:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            f"Analytic implicit profile is not bound to this compiled geometry and coordinate contract: {error}",
            provider_code="native_implicit_profile",
            stage="source_inspection",
        ) from error

    bounds = np.asarray(profile.boundary_bounds, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[0] >= bounds[1])
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Analytic implicit profile must retain finite ordered boundary bounds.",
            provider_code="native_implicit_profile",
            stage="source_inspection",
        )
    extent = bounds[1] - bounds[0]
    padding = float(np.max(extent)) / 8.0
    if not np.all(np.isfinite(extent)) or not math.isfinite(padding) or padding <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Analytic implicit boundary bounds must admit a finite positive adaptive search margin.",
            provider_code="native_implicit_profile",
            stage="source_inspection",
        )
    domain = np.stack(
        (
            np.nextafter(bounds[0] - padding, -np.inf),
            np.nextafter(bounds[1] + padding, np.inf),
        )
    )
    if (
        not np.all(np.isfinite(domain))
        or np.any(domain[0] >= bounds[0])
        or np.any(domain[1] <= bounds[1])
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The analytic implicit search domain must strictly enclose its certified boundary bounds.",
            provider_code="native_implicit_profile",
            stage="source_inspection",
        )
    selected_policy = (
        (AdaptiveImplicitSurfacePolicy() if policy is None else policy)
        if specification is None
        else _effective_policy(policy, specification, domain)
    )
    try:
        discovery = discover_adaptive_implicit_surface(
            profile.geometry,
            domain=domain,
            policy=selected_policy,
            source_id=profile.source_id,
        )
        source = NativeImplicitSource(
            profile.geometry,
            profile.source_id,
            profile.source_revision,
            discovery=discovery,
            authoring_profile=profile,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"Analytic profile adaptive implicit discovery failed: {error}",
            provider_code="native_implicit_profile_discovery",
            stage="surface_meshing",
        ) from error
    _validate_discovery(source, discovery)
    return source


def _grid_domain(source: NativeImplicitSource, /) -> np.ndarray:
    grid = source.grid
    if grid is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit discovery requires a retained adaptive result or a bounded three-axis tensor grid.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    if len(grid.axes) != 3:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Adaptive implicit surface discovery requires a three-dimensional tensor-grid domain.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    lower = np.empty((3,), dtype=np.float64)
    upper = np.empty((3,), dtype=np.float64)
    for dimension, axis in enumerate(grid.axes):
        bounds = axis.bounds
        if axis.periodic or bounds is None:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Implicit adaptive discovery requires finite nonperiodic bounds on all three tensor-grid axes.",
                provider_code="native_implicit_domain",
                stage="source_inspection",
            )
        values = np.asarray(bounds, dtype=np.float64)
        if (
            values.shape != (2,)
            or not np.all(np.isfinite(values))
            or values[0] >= values[1]
        ):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The implicit tensor-grid axes must retain finite increasing physical bounds.",
                provider_code="native_implicit_domain",
                stage="source_inspection",
            )
        lower[dimension], upper[dimension] = values
    return np.stack((lower, upper))


def _effective_policy(
    policy: AdaptiveImplicitSurfacePolicy | None,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: np.ndarray,
    /,
) -> AdaptiveImplicitSurfacePolicy:
    """Discovery policy for ``specification`` on the declared ``domain``.

    An explicit policy is only tightened by the specification's deviation and
    hard size allowances. Without one, the specification's deviation and uniform
    size targets set the resolution; the policy default applies only when the
    specification declares neither.
    """
    diagonal = float(np.linalg.norm(domain[1] - domain[0]))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The implicit discovery domain must have a finite positive diagonal.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    selected = AdaptiveImplicitSurfacePolicy() if policy is None else policy
    resolutions: list[float] = [] if policy is None else [policy.relative_cell_diameter]
    root_tolerance = selected.root_tolerance
    maximum_deviation = specification.maximum_deviation
    if maximum_deviation is not None:
        if maximum_deviation <= 0.0:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A strictly positive maximum_deviation is required for adaptive implicit surface fidelity.",
                provider_code="native_implicit_preflight",
                stage="source_inspection",
            )
        root_tolerance = min(root_tolerance, maximum_deviation * 0.25)
        resolutions.append(0.5 * maximum_deviation / diagonal)
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        if control.strength is SizeControlStrength.HARD:
            allowance = control.target_size + (
                specification.size_compliance.absolute_tolerance
                + specification.size_compliance.relative_tolerance * control.target_size
            )
            if control.maximum_size is not None:
                allowance = min(
                    allowance,
                    control.maximum_size
                    + specification.size_compliance.absolute_tolerance
                    + specification.size_compliance.relative_tolerance
                    * control.maximum_size,
                )
            resolutions.append(allowance / diagonal)
        elif policy is None:
            resolutions.append(control.target_size / diagonal)
    return replace(
        selected,
        relative_cell_diameter=min(resolutions, default=selected.relative_cell_diameter),
        root_tolerance=root_tolerance,
    )


_DISCOVERY_VALIDATION_BATCH = 16384


def _discovery_roots_match(
    roots: np.ndarray, errors: np.ndarray, points: np.ndarray, point_errors: np.ndarray, /
) -> bool:
    """Match byte-exact points and numerically equal errors, including duplicates."""
    if len(roots) != len(errors):
        raise ValueError("zip() argument 2 is shorter or longer than argument 1")
    root_bits = tuple(roots[:, axis].view(np.uint64) for axis in range(3))
    order = None
    for start in range(0, len(points), _DISCOVERY_VALIDATION_BATCH):
        stop = min(start + _DISCOVERY_VALIDATION_BATCH, len(points))
        query = points[start:stop]
        query_errors = point_errors[start:stop]
        bits = tuple(query[:, axis].view(np.uint64) for axis in range(3))
        # The common aligned case needs no global permutation at all.
        if (
            stop <= len(roots)
            and all(
                np.array_equal(bits[axis], root_bits[axis][start:stop])
                for axis in range(3)
            )
            and np.array_equal(query_errors, errors[start:stop])
        ):
            continue
        if not len(roots):
            return False
        if order is None:
            # Only the permutation is proportional to the root count; all key
            # comparisons and search scratch below are bounded batches.
            order = np.lexsort((errors, root_bits[2], root_bits[1], root_bits[0]))
        low = np.zeros(len(query), dtype=np.int64)
        high = np.full(len(query), len(roots), dtype=np.int64)
        while np.any(low < high):
            active = low < high
            mid = (low[active] + high[active]) // 2
            rows = order[mid]
            equal = np.ones(len(rows), dtype=np.bool_)
            less = np.zeros(len(rows), dtype=np.bool_)
            for axis in range(3):
                left = root_bits[axis][rows]
                right = bits[axis][active]
                less |= equal & (left < right)
                equal &= left == right
            less |= equal & (errors[rows] < query_errors[active])
            active_rows = np.flatnonzero(active)
            low[active_rows[less]] = mid[less] + 1
            high[active_rows[~less]] = mid[~less]
        valid = low < len(roots)
        if not np.all(valid):
            return False
        rows = order[low]
        if any(
            not np.array_equal(root_bits[axis][rows], bits[axis]) for axis in range(3)
        ) or not np.array_equal(errors[rows], query_errors):
            return False
    return True


def _discovery_possible_boxes(cover: Any, /) -> Iterator[np.ndarray]:
    """Yield possible-zero boxes in cover order using bounded scratch."""
    for start in range(0, len(cover.value_lower), _DISCOVERY_VALIDATION_BATCH):
        stop = min(start + _DISCOVERY_VALIDATION_BATCH, len(cover.value_lower))
        possible = (cover.value_lower[start:stop] <= 0.0) & (
            cover.value_upper[start:stop] >= 0.0
        )
        yield np.asarray(cover.boxes[start:stop][possible], dtype=np.float64)


def _direct_discovery_cover_matches(
    cover: Any,
    points: np.ndarray,
    errors: np.ndarray,
    boxes: np.ndarray,
    radius: float,
    /,
) -> bool:
    """Prove the direct unique-box sample case without materializing the samples."""
    count = len(points)
    if (
        points.shape != (count, 3)
        or errors.shape != (count,)
        or boxes.shape != (count, 2, 3)
        or not math.isfinite(radius)
        or radius <= 0
    ):
        return False
    dtype = np.dtype([(f"coordinate_{axis}", np.float64) for axis in range(6)])
    keys = np.ascontiguousarray(boxes).reshape((-1, 6)).view(dtype).reshape(-1)
    order = np.argsort(keys, kind="stable")
    for start in range(0, count, _DISCOVERY_VALIDATION_BATCH):
        stop = min(start + _DISCOVERY_VALIDATION_BATCH, count)
        selected = keys[order[start:stop]]
        if (start and keys[order[start - 1]] == selected[0]) or np.any(
            selected[1:] == selected[:-1]
        ):
            return False
    used = np.zeros((count,), dtype=np.bool_)
    total = 0
    for possible in _discovery_possible_boxes(cover):
        n = len(possible)
        if not n:
            continue
        query = np.ascontiguousarray(possible).reshape((-1, 6)).view(dtype).reshape(-1)
        positions = np.searchsorted(keys, query, side="left", sorter=order)
        if np.any(positions >= count):
            return False
        rows = order[positions]
        if np.any(keys[rows] != query) or np.any(used[rows]) or len(np.unique(rows)) != n:
            return False
        used[rows] = True
        extents = np.nextafter(possible[:, 1] - possible[:, 0], np.inf)
        squared = np.nextafter(extents * extents, np.inf)
        radii = np.nextafter(
            np.sqrt(np.nextafter(np.sum(squared, axis=1), np.inf)), np.inf
        )
        first = np.nextafter(np.abs(points[rows] - possible[:, 0]), np.inf)
        second = np.nextafter(np.abs(points[rows] - possible[:, 1]), np.inf)
        delta = np.maximum(first, second)
        distance_squared = np.zeros((n,), dtype=np.float64)
        for axis in range(3):
            distance_squared = np.nextafter(
                distance_squared + np.nextafter(delta[:, axis] * delta[:, axis], np.inf),
                np.inf,
            )
        distances = np.nextafter(np.sqrt(np.nextafter(distance_squared, np.inf)), np.inf)
        if (
            np.any(~np.isfinite(radii))
            or np.any(radii > radius)
            or np.any(distances > radii)
        ):
            return False
        total += n
    return total == count and bool(np.all(used))


def _validate_discovery(
    source: NativeImplicitSource,
    surface: AdaptiveImplicitSurface,
    /,
) -> None:
    cover = surface.cover
    if (
        surface.source_id != source.source_id
        or surface.volume.source_id != source.source_id
        or not surface.volume.certified
        or cover is None
        or cover.source_id != source.source_id
        or not cover.complete
        or not cover.certified
        or cover.enclosure == "sampled"
        or surface.evidence.accuracy == "sampled"
        or surface.evidence.unresolved_count != 0
        or surface.evidence.status_flags != AdaptiveImplicitSurfaceStatus.SUCCESS
        or surface.mesh is None
        or not surface.mesh.topology.watertight
        or np.asarray(surface.mesh.faces).shape[0] == 0
        or surface.boundary_witnesses.shape[0] == 0
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit adaptive discovery must retain a complete rigorous cover and a positive watertight extraction without unresolved boxes.",
            provider_code="native_implicit_discovery_incomplete",
            stage="source_inspection",
        )

    witness_points = np.asarray(surface.boundary_witnesses, dtype=np.float64)
    witness_errors = np.asarray(surface.boundary_errors, dtype=np.float64)
    witness_boxes = np.asarray(surface.boundary_witness_boxes, dtype=np.float64)
    maximum_radius = -np.inf
    for start in range(0, len(witness_boxes), _DISCOVERY_VALIDATION_BATCH):
        boxes = witness_boxes[start : start + _DISCOVERY_VALIDATION_BATCH]
        extents = np.nextafter(boxes[:, 1] - boxes[:, 0], np.inf)
        squared = np.nextafter(extents * extents, np.inf)
        radii = np.nextafter(
            np.sqrt(np.nextafter(np.sum(squared, axis=1), np.inf)), np.inf
        )
        maximum_radius = np.maximum(maximum_radius, np.max(radii))
    if not len(witness_boxes):
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity"
        )
    maximum_radius = float(maximum_radius)
    if type(surface) is AdaptiveImplicitSurface and _direct_discovery_cover_matches(
        cover, witness_points, witness_errors, witness_boxes, maximum_radius
    ):
        return
    samples = surface.boundary_source.boundary_cover(maximum_radius)
    sample_points = np.asarray(samples.points, dtype=np.float64)
    sample_errors = np.asarray(samples.point_error, dtype=np.float64)
    sample_radii = np.asarray(samples.covering_radius, dtype=np.float64)
    if len(witness_points) != len(witness_errors):
        raise ValueError("zip() argument 2 is shorter or longer than argument 1")
    possible_count = sum(len(boxes) for boxes in _discovery_possible_boxes(cover))
    sample_roots_match = (
        sample_points.shape == (possible_count, 3)
        and sample_errors.shape == (possible_count,)
        and _discovery_roots_match(
            witness_points, witness_errors, sample_points, sample_errors
        )
    )
    if not sample_roots_match:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit adaptive cover samples must retain one current strict root for each possible-zero box.",
            provider_code="native_implicit_discovery_incomplete",
            stage="source_inspection",
        )
    box_radii_match = True
    offset = 0
    for boxes in _discovery_possible_boxes(cover):
        count = len(boxes)
        points = sample_points[offset : offset + count]
        first_delta = np.nextafter(np.abs(points - boxes[:, 0]), np.inf)
        second_delta = np.nextafter(np.abs(points - boxes[:, 1]), np.inf)
        delta = np.maximum(first_delta, second_delta)
        distance_squared = np.zeros((count,), dtype=np.float64)
        for axis in range(3):
            distance_squared = np.nextafter(
                distance_squared + np.nextafter(delta[:, axis] * delta[:, axis], np.inf),
                np.inf,
            )
        box_cover_upper = np.nextafter(
            np.sqrt(np.nextafter(distance_squared, np.inf)), np.inf
        )
        if np.any(box_cover_upper > sample_radii[offset : offset + count]):
            box_radii_match = False
        offset += count
    if (
        samples.semantics != "certified"
        or not samples.complete
        or not np.all(np.isfinite(sample_radii))
        or np.any(sample_radii > maximum_radius)
        or not box_radii_match
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit adaptive discovery must retain a complete certified per-box root cover.",
            provider_code="native_implicit_discovery_incomplete",
            stage="source_inspection",
        )


def _discover(
    source: NativeImplicitSource,
    domain: np.ndarray,
    policy: AdaptiveImplicitSurfacePolicy,
    /,
) -> AdaptiveImplicitSurface:
    try:
        surface = discover_adaptive_implicit_surface(
            source.geometry,
            domain=domain,
            policy=policy,
            source_id=source.source_id,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"Geometry-owned adaptive implicit discovery failed: {error}",
            provider_code="native_implicit_discovery",
            stage="surface_meshing",
        ) from error
    _validate_discovery(source, surface)
    return surface


def _resolve_discovery(
    source: NativeImplicitSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    policy: AdaptiveImplicitSurfacePolicy | None,
    /,
) -> AdaptiveImplicitSurface:
    if source.discovery is not None:
        _validate_discovery(source, source.discovery)
        if source.discovery.cover is None:
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "A retained implicit discovery must retain its complete domain cover.",
                provider_code="native_implicit_discovery_incomplete",
                stage="source_inspection",
            )
        # The cover spans the discovery lattice; the policy is declared on the
        # caller's domain, which a re-anchored lattice extends.
        domain = np.asarray(source.discovery.evidence.declared_domain, dtype=np.float64)
    else:
        domain = _grid_domain(source)
    effective_policy = _effective_policy(policy, specification, domain)
    if source.discovery is not None and source.discovery.policy == effective_policy:
        return source.discovery
    return _discover(source, domain, effective_policy)


@dataclass(slots=True, eq=False, init=False)
class _NativeImplicitProjectionResult:
    """Retain adaptive candidates under their native source/frame contract."""

    adaptive_projection: SourceBoundaryProjectionResult
    source_id: str
    source_revision: str
    frame_id: str
    adaptive_source_revision: str
    queries: ArrayLike
    points: ArrayLike
    parameters: ArrayLike
    residuals: ArrayLike
    distance_lower_bounds: ArrayLike
    status: ArrayLike
    normals: ArrayLike
    tangents: ArrayLike
    occurrence_paths: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        adaptive_projection: SourceBoundaryProjectionResult,
        source_id: str,
        source_revision: str,
        frame_id: str,
        adaptive_source_revision: str,
        /,
    ) -> None:
        if (
            adaptive_projection.source_id != source_id
            or adaptive_projection.source_revision != adaptive_source_revision
            or adaptive_projection.frame_id != frame_id
            or not source_id
            or not source_revision
            or not frame_id
            or not adaptive_source_revision
        ):
            raise ValueError(
                "Adaptive projection evidence does not match its native source adapter."
            )
        self.adaptive_projection = adaptive_projection
        self.source_id = source_id
        self.source_revision = source_revision
        self.frame_id = frame_id
        self.adaptive_source_revision = adaptive_source_revision
        self.queries = adaptive_projection.queries
        self.points = adaptive_projection.points
        self.parameters = adaptive_projection.parameters
        self.residuals = adaptive_projection.residuals
        self.distance_lower_bounds = adaptive_projection.distance_lower_bounds
        self.status = adaptive_projection.status
        self.normals = adaptive_projection.normals
        self.tangents = adaptive_projection.tangents
        self.occurrence_paths = adaptive_projection.occurrence_paths

    @property
    def candidate_root_rows(self) -> np.ndarray:
        return np.asarray(
            getattr(self.adaptive_projection, "candidate_root_rows"), dtype=np.int64
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(getattr(self.adaptive_projection, "reasons"))

    def entity_ids(self, /) -> tuple[str, ...]:
        surface_key = source_entity_id(
            self.source_id, self.source_revision, "implicit-surface", "zero-set"
        )
        return (surface_key,) * len(self.occurrence_paths)


@final
class _NativeImplicitBoundaryQuery(StrictModule, NonTrainableState):
    """Forward adaptive boundary evidence under the native source revision."""

    query: AdaptiveImplicitBoundarySource
    authoring_profile: AnalyticImplicitProfile | None
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    boundary_source_revision: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        query: AdaptiveImplicitBoundarySource,
        source_id: str,
        source_revision: str,
        frame_id: str,
        /,
        *,
        authoring_profile: AnalyticImplicitProfile | None = None,
    ) -> None:
        if not isinstance(query, AdaptiveImplicitBoundarySource):
            raise TypeError("query must be an AdaptiveImplicitBoundarySource.")
        if query.source_id != source_id:
            raise ValueError("query.source_id must match the native implicit source.")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(
                "The native implicit boundary query requires an exact spatial contract ID."
            )
        if query.frame_id and query.frame_id != frame_id:
            raise ValueError(
                "Adaptive implicit boundary source is bound to a different spatial contract."
            )
        if authoring_profile is not None:
            if not isinstance(authoring_profile, AnalyticImplicitProfile):
                raise TypeError(
                    "authoring_profile must be AnalyticImplicitProfile or None."
                )
            if (
                authoring_profile.source_id != source_id
                or authoring_profile.source_revision != source_revision
                or authoring_profile.coordinate_contract.spatial_id != frame_id
            ):
                raise ValueError(
                    "authoring_profile must bind the native source identity and frame."
                )
            authoring_profile.require_bound(
                query.geometry, authoring_profile.coordinate_contract
            )
        bound_query = (
            query
            if query.frame_id == frame_id
            else AdaptiveImplicitBoundarySource(query.surface, frame_id)
        )
        bound_query.require_current()
        self.query = bound_query
        self.authoring_profile = authoring_profile
        self.source_id = source_id
        self.source_revision = source_revision
        self.frame_id = frame_id
        self.boundary_source_revision = bound_query.source_revision
        self.query_id = canonical_fingerprint(
            {
                "kind": "native-adaptive-implicit-boundary-query",
                "source": (source_id, source_revision),
                "boundary_source_revision": bound_query.source_revision,
                "frame_id": frame_id,
                "authoring_profile": (
                    None if authoring_profile is None else authoring_profile.profile_id
                ),
            }
        )

    @property
    def adaptive_source(self) -> AdaptiveImplicitBoundarySource:
        """Return the actual geometry-owned adaptive source authority."""
        return self.query

    @property
    def ambient_dimension(self) -> int:
        return self.query.ambient_dimension

    def require_current(self, /) -> None:
        self.query.require_current()
        if (
            self.query.source_id != self.source_id
            or self.query.source_revision != self.boundary_source_revision
            or self.query.frame_id != self.frame_id
        ):
            raise ValueError(
                "Native implicit boundary query changed its source or frame binding."
            )
        profile = self.authoring_profile
        if profile is not None:
            if (
                profile.source_id != self.source_id
                or profile.source_revision != self.source_revision
                or profile.coordinate_contract.spatial_id != self.frame_id
            ):
                raise ValueError(
                    "Native implicit boundary query changed its authoring-profile binding."
                )
            profile.require_bound(self.query.geometry, profile.coordinate_contract)

    def project_boundary(
        self,
        points: np.ndarray,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> SourceBoundaryProjectionResult:
        surface_key = source_entity_id(
            self.source_id, self.source_revision, "implicit-surface", "zero-set"
        )
        if entity_ids is not None and (
            not entity_ids or any(identifier != surface_key for identifier in entity_ids)
        ):
            raise ValueError(
                "Implicit projection scope must name its complete zero-set entity."
            )
        if occurrence_paths is not None and any(path for path in occurrence_paths):
            raise ValueError("Implicit zero-set projection has no occurrence path.")
        if (
            entity_ids is not None
            and occurrence_paths is not None
            and (len(entity_ids) != len(occurrence_paths))
        ):
            raise ValueError(
                "Implicit projection entity IDs and occurrence paths must align."
            )
        projection = self.query.project_boundary(
            points,
            frame_id=self.frame_id,
        )
        return _NativeImplicitProjectionResult(
            projection,
            self.source_id,
            self.source_revision,
            self.frame_id,
            self.boundary_source_revision,
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.query.boundary_distance(points)

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        return self.query.boundary_cover(maximum_radius)


def _mesh_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        identifiers,
    )


def _mesh_connected(faces: np.ndarray, /) -> bool:
    face_count = faces.shape[0]
    if face_count <= 1:
        return True
    owners = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    edges = np.sort(
        np.stack(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
            axis=1,
        ).reshape((-1, 2)),
        axis=1,
    )
    order = np.lexsort((owners, edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_owners = owners[order]
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.asarray((True,)),
                np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1),
            )
        )
    )
    parent = np.arange(face_count, dtype=np.int64)

    def root(value: int, /) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    stops = np.concatenate((starts[1:], np.asarray((sorted_owners.size,))))
    for first, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        if stop - first == 2:
            first_root = root(int(sorted_owners[first]))
            second_root = root(int(sorted_owners[first + 1]))
            if first_root != second_root:
                parent[second_root] = first_root
    first_root = root(0)
    return all(root(face) == first_root for face in range(1, face_count))


def _minimum_angle(vertices: np.ndarray, faces: np.ndarray, /) -> float:
    minimum = np.float64(np.inf)
    if not len(faces):
        return float(np.min(np.empty((0,), dtype=np.float64)))
    for start in range(0, len(faces), _DISCOVERY_VALIDATION_BATCH):
        minimum = np.minimum(
            minimum,
            _minimum_angle_batch(
                vertices, faces[start : start + _DISCOVERY_VALIDATION_BATCH]
            ),
        )
    return float(minimum)


def _minimum_angle_batch(vertices: np.ndarray, faces: np.ndarray, /) -> float:
    triangles = vertices[faces]
    side_a = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    side_b = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    side_c = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    angle_a = np.arccos(
        np.clip(
            (side_b**2 + side_c**2 - side_a**2) / (2.0 * side_b * side_c),
            -1.0,
            1.0,
        )
    )
    angle_b = np.arccos(
        np.clip(
            (side_a**2 + side_c**2 - side_b**2) / (2.0 * side_a * side_c),
            -1.0,
            1.0,
        )
    )
    angle_c = np.arccos(
        np.clip(
            (side_a**2 + side_b**2 - side_c**2) / (2.0 * side_a * side_b),
            -1.0,
            1.0,
        )
    )
    return float(np.min(np.minimum(np.minimum(angle_a, angle_b), angle_c)))


def _compliance(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    fidelity_bound: float,
    /,
) -> MeshingComplianceReport:
    edges = np.unique(
        np.sort(
            np.concatenate(
                (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
            ),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Adaptive implicit extraction produced no finite positive triangle edges.",
            provider_code="native_implicit_quality",
            stage="specification_compliance",
        )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = [("source_fidelity_bound", fidelity_bound)]
    issues: list[str] = []
    policy = specification.size_compliance
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.maximum_growth_rate is not None:
            requested.append((f"{key}:maximum_growth_rate", control.maximum_growth_rate))
        statistics = {
            name: float(np.quantile(lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(lengths))),
                (f"{key}:maximum_edge", float(np.max(lengths))),
                (f"{key}:size_field_growth_rate", 1.0),
                (
                    f"{key}:maximum_edge_to_target_size",
                    float(np.max(lengths) / control.target_size),
                ),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if control.minimum_size is not None and float(np.min(lengths)) < (
                control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if control.maximum_size is not None and float(np.max(lengths)) > (
                control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
            if np.any(lengths > control.target_size + tolerance):
                issues.append(f"local_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and 1.0 > control.maximum_growth_rate + tolerance
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.append(
            (f"protected:{feature.feature_id}:source_fidelity_bound", fidelity_bound)
        )
        if feature.hard and fidelity_bound > feature.maximum_deviation:
            issues.append(f"protected:{feature.feature_id}:maximum_deviation")
    if specification.maximum_deviation is not None:
        requested.append(("maximum_deviation", specification.maximum_deviation))
        if fidelity_bound > specification.maximum_deviation:
            issues.append("maximum_deviation")
    if specification.quality_target is not None:
        target = specification.quality_target
        minimum_angle = _minimum_angle(vertices, faces)
        requested.append(("minimum_angle", target.minimum_angle))
        achieved.append(("minimum_angle", minimum_angle))
        if target.hard and minimum_angle < target.minimum_angle:
            issues.append(f"minimum_angle:{target.target_id}")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _organization(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    coordinate_contract: SpatialCoordinateContract,
    mesh: cell_mesh_owner.CellMesh,
    faces: np.ndarray,
    fidelity_bound: float,
    /,
    *,
    vertex_deviations: np.ndarray | None = None,
    fixed_plan_id: str | None = None,
) -> NativeMeshOrganization:
    cover = discovery.cover
    if cover is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit surface publication requires its retained adaptive cover.",
            provider_code="native_implicit_cover_missing",
            stage="geometry_association",
        )
    identifiers = _scope_ids(source)
    source_key = source.source_entity_key_for_scope(2, int(identifiers[0]))
    cell_set = mesh.entity_set(2)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    if (
        cell_ids.size != faces.shape[0]
        or vertex_ids.size != np.asarray(mesh.coordinates).shape[0]
    ):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Cell-mesh entity sets do not align with adaptive extraction coordinates and triangles.",
            provider_code="native_implicit_organization",
            stage="geometry_association",
        )
    cell_scope = _mesh_scope(mesh, 2, cell_ids)
    label = MeshLabel("implicit-zero-set", cell_scope)
    patch = MeshPatch(
        "implicit-zero-set",
        cell_scope,
        connected=_mesh_connected(faces),
    )
    associations = (
        GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            source.source_id,
            source.source_revision,
            cell_set.entity_set_id,
            cell_ids,
            (source_key[0],) * cell_ids.size,
            np.full((cell_ids.size,), fidelity_bound, dtype=np.float64),
            exact=False,
            source_roles=(GeometrySourceEntityRole.FACET,) * cell_ids.size,
            source_occurrence_paths=(source_key[1],) * cell_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            source.source_id,
            source.source_revision,
            vertex_set.entity_set_id,
            vertex_ids,
            (source_key[0],) * vertex_ids.size,
            (
                np.full(
                    (vertex_ids.size,), discovery.policy.root_tolerance, dtype=np.float64
                )
                if vertex_deviations is None
                else vertex_deviations
            ),
            source_roles=(GeometrySourceEntityRole.FACET,) * vertex_ids.size,
            source_occurrence_paths=(source_key[1],) * vertex_ids.size,
        ),
    )
    tags = ("implicit-zero-set",) * faces.shape[0]
    boundary = SurfaceModel(
        mesh,
        SurfaceMetadata(
            source_id=source.source_id,
            source_revision=source.source_revision,
            coordinate_contract=coordinate_contract,
            provenance=(
                (
                    "geometry-owned-adaptive-implicit-surface"
                    if fixed_plan_id is None
                    else "geometry-owned-fixed-lattice-implicit-surface"
                ),
                source.binding_id,
                discovery.evidence.evidence_id,
                cover.cover_id,
                *((fixed_plan_id,) if fixed_plan_id is not None else ()),
            ),
            cell_tags=tags,
        ),
    )
    return NativeMeshOrganization(
        boundary=boundary,
        patches=(patch,),
        labels=(label,),
        associations=associations,
    )


def _implicit_region_authority(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    /,
) -> tuple[str, tuple[str, ...], tuple[str, tuple[str, ...]]]:
    domain_id = getattr(discovery, "domain_id", None)
    region_ids = tuple(getattr(discovery, "region_ids", ()))
    if (
        not isinstance(domain_id, str)
        or not domain_id
        or not region_ids
        or len(set(region_ids)) != len(region_ids)
        or any(not isinstance(value, str) or not value for value in region_ids)
        or discovery.source_id != source.source_id
        or array_tree_fingerprint(discovery.geometry)
        != array_tree_fingerprint(source.geometry)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume meshing requires an immutable source-bound discovery domain with authored region identities.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    if len(region_ids) != 1:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "The native implicit PLC carrier supports one certified interior region.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    region_paths = tuple(getattr(discovery, "region_occurrence_paths", ((),)))
    if len(region_paths) != len(region_ids):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit discovery region identities and occurrence paths must align.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    scope_ids = np.asarray(source.scope_entity_ids(3), dtype=np.int64)
    if (
        scope_ids.shape != (len(region_ids),)
        or np.unique(scope_ids).size != scope_ids.size
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Native implicit region scopes must bijectively match the adaptive domain region table.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    region_key = source.source_entity_key_for_scope(3, int(scope_ids[0]))
    if region_key[1] != region_paths[0]:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit source-region occurrence paths must match the original source scope.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    return domain_id, region_ids, region_key


def _implicit_interior_seed(
    discovery: AdaptiveImplicitSurface,
    region_name: str,
    /,
    *,
    material_id: str | None = None,
    role: RegionRole = RegionRole.USER,
) -> RegionSeed:
    cover = discovery.cover
    if cover is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit region seeding requires the complete adaptive value-enclosure cover.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    interior_boxes = np.flatnonzero(cover.value_upper < 0.0)
    if not interior_boxes.size:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The adaptive cover contains no rigorously classified interior seed box.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    box = cover.boxes[int(interior_boxes[0])]
    point = box[0] + 0.5 * (box[1] - box[0])
    classification = discovery.volume.classify_points(point[None, :])
    if not classification.certified or int(classification.classes[0]) != int(
        ImplicitVolumeClass.INSIDE
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The implicit volume seed must retain certified negative-inside classification.",
            provider_code="native_implicit_domain",
            stage="source_inspection",
        )
    material = region_name if material_id is None else material_id
    return RegionSeed(point, region_name, material, role)


def _validated_implicit_region_seeds(
    discovery: AdaptiveImplicitSurface,
    specification: VolumeMeshingSpec,
    region_name: str,
    /,
) -> tuple[RegionSeed, ...]:
    control = specification.region_controls[0] if specification.region_controls else None
    if not specification.region_seeds:
        return (
            _implicit_interior_seed(
                discovery,
                region_name,
                material_id=None if control is None else control.material_id,
                role=RegionRole.USER if control is None else control.role,
            ),
        )
    seeds = tuple(specification.region_seeds)
    points = np.asarray([np.asarray(seed.point, dtype=np.float64) for seed in seeds])
    if points.shape != (len(seeds), 3):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit region seeds must be three-dimensional source coordinates.",
            provider_code="native_implicit_region_seed",
            stage="source_inspection",
        )
    if any(seed.region_name != region_name for seed in seeds):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit region seeds must name the source-declared interior region.",
            provider_code="native_implicit_region_seed",
            stage="source_inspection",
        )
    if control is not None and any(
        (seed.material_id, seed.role) != (control.material_id, control.role)
        for seed in seeds
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit region seeds and source-region control disagree on material or role.",
            provider_code="native_implicit_region_seed",
            stage="source_inspection",
        )
    classification = discovery.volume.classify_points(points)
    if not classification.certified or np.any(
        np.asarray(classification.classes) != int(ImplicitVolumeClass.INSIDE)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Every implicit region seed must have certified inside classification.",
            provider_code="native_implicit_region_seed",
            stage="source_inspection",
        )
    return seeds


class _ImplicitCarrier(NamedTuple):
    """The FIXED-fill PLC carrier and the ancestry its chart witness needs."""

    carrier: NativePlcSource
    specification: VolumeMeshingSpec
    triangles: np.ndarray
    face_scaffold_rows: np.ndarray
    """Scaffold triangle row bound by each carrier triangle's chart."""
    distance: float
    """Certified carrier-to-zero-set distance bound."""
    evidence: tuple[tuple[str, float], ...]
    """Carrier quality before/after the zero-set remesh, and whether it ran."""


def _face_radius_edge_ratios(vertices: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    """Circumradius over shortest edge per triangle (+inf when degenerate).

    A tetrahedron on a FIXED boundary face has a radius-edge ratio at least the
    face's, so a face above the schedule bound makes the fill unmeetable.
    """
    triangles = vertices[faces]
    sides = np.stack(
        (
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    shortest = np.min(sides, axis=1)
    valid = (twice_area > 0.0) & (shortest > 0.0)
    ratios = np.full((faces.shape[0],), np.inf, dtype=np.float64)
    ratios[valid] = np.prod(sides[valid], axis=1) / (
        2.0 * twice_area[valid] * shortest[valid]
    )
    return ratios


def _maximum_face_radius_edge_ratio(vertices: np.ndarray, faces: np.ndarray, /) -> float:
    if not len(faces):
        return float(np.max(np.empty((0,), dtype=np.float64)))
    maximum = np.float64(-np.inf)
    for start in range(0, len(faces), _DISCOVERY_VALIDATION_BATCH):
        maximum = np.maximum(
            maximum,
            np.max(
                _face_radius_edge_ratios(
                    vertices, faces[start : start + _DISCOVERY_VALIDATION_BATCH]
                )
            ),
        )
    return float(maximum)


def _carrier_quality(
    vertices: np.ndarray, faces: np.ndarray, suffix: str, /
) -> tuple[tuple[str, float], ...]:
    return (
        (
            f"carrier_maximum_radius_edge_ratio_{suffix}",
            _maximum_face_radius_edge_ratio(vertices, faces),
        ),
        (
            f"carrier_minimum_angle_degrees_{suffix}",
            float(np.rad2deg(_minimum_angle(vertices, faces))),
        ),
    )


def _implicit_field_box_programs(
    geometry: CompiledGeometry, /
) -> tuple[IntervalBoxProgram, IntervalBoxProgram]:
    """Lower f and grad f of the implicit source to one-box native interval programs."""
    kernel, state = geometry.kernel, geometry.state

    def value(point: Array) -> Array:
        return kernel.boundary_field(state, point[None, :])[0]

    try:
        programs = tuple(
            prepare_interval_function(function, 3, batch_capacity=1, execution="native")
            for function in (value, jax.grad(value))
        )
    except IntervalExtensionError as error:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "The implicit carrier zero-set remesh requires a native interval lowering "
            f"of the field and its gradient: {error}",
            provider_code="native_implicit_carrier_remesh",
            stage="source_inspection",
        ) from error
    value_program, gradient_program = (program.native_program for program in programs)
    if value_program is None or gradient_program is None:
        raise RuntimeError("Native interval preparation returned no native program.")
    return value_program.box_program(), gradient_program.box_program()


def _carrier_edge_length(
    vertices: np.ndarray, faces: np.ndarray, specification: VolumeMeshingSpec, /
) -> float:
    """Requested uniform target size, else the carrier's mean edge length."""
    targets = [
        control.target_size
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    ]
    if targets:
        return float(min(targets))
    edges = np.unique(
        np.sort(
            np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)])), axis=1
        ),
        axis=0,
    )
    return float(
        np.mean(np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1))
    )


def _nearest_scaffold_rows(
    scaffold_vertices: np.ndarray,
    scaffold_faces: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
) -> np.ndarray:
    """Find exact nearest scaffold centroids in bounded spatial shards.

    A sampled centroid supplies an actual upper bound. Each shard's canonical
    packed BVH enumerates every centroid in an outward-rounded box around that
    bound; squared distances and the lowest-row tie are then evaluated exactly
    as in the exhaustive NumPy scan. The outward ULP before the square root
    includes distances whose squares round to zero. Nonfinite cases use the same
    exhaustive reduction in bounded blocks.
    """

    def triangle_centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        first = np.mean(points[triangles[:4096]], axis=1)
        result = np.empty((len(triangles), *first.shape[1:]), dtype=first.dtype)
        result[: len(first)] = first
        for start in range(len(first), len(triangles), 4096):
            stop = min(start + 4096, len(triangles))
            result[start:stop] = np.mean(points[triangles[start:stop]], axis=1)
        return result

    first_scaffold = np.mean(scaffold_vertices[scaffold_faces[:16384]], axis=1)
    centroids = triangle_centroids(vertices, faces)
    rows = np.full(len(centroids), -1, dtype=np.int64)
    if not len(scaffold_faces):
        if len(centroids):
            raise ValueError("attempt to get argmin of an empty sequence")
        return rows
    best = np.full(len(centroids), np.inf, dtype=np.float64)
    finite_points = np.all(np.isfinite(centroids), axis=1)
    tiny = np.finfo(np.float64).tiny
    sample_rows = np.unique(np.linspace(0, len(scaffold_faces) - 1, 64).astype(np.int64))
    sample = np.mean(scaffold_vertices[scaffold_faces[sample_rows]], axis=1)
    for start in range(0, len(centroids), 32):
        stop = min(start + 32, len(centroids))
        squared = np.sum(
            (centroids[start:stop, None, :] - sample[None, :, :]) ** 2, axis=2
        )
        local = np.argmin(squared, axis=1)
        rows[start:stop] = sample_rows[local]
        best[start:stop] = squared[np.arange(stop - start), local]

    def exhaustive(points: np.ndarray, source: np.ndarray) -> np.ndarray:
        squared = np.sum((points[:, None, :] - source[None, :, :]) ** 2, axis=2)
        return np.argmin(squared, axis=1)

    for source_start in range(0, len(scaffold_faces), 16384):
        source_stop = min(source_start + 16384, len(scaffold_faces))
        source = (
            first_scaffold
            if source_start == 0
            else np.mean(
                scaffold_vertices[scaffold_faces[source_start:source_stop]], axis=1
            )
        )
        source_finite = bool(np.all(np.isfinite(source)))
        use_bvh = source_finite and source.dtype == centroids.dtype == np.dtype(
            np.float64
        )
        active = (~np.isnan(best)) | (~finite_points & (rows > source_start))
        if not source_finite:
            active |= rows > source_start
        if use_bvh:
            lower, upper = np.min(source, axis=0), np.max(source, axis=0)
            reach = (
                np.nextafter(
                    np.sqrt(np.nextafter(best, np.inf)) * (1.0 + 2.0**-30), np.inf
                )
                + tiny
            )
            overlap = np.all(
                (centroids + reach[:, None] >= lower)
                & (centroids - reach[:, None] <= upper),
                axis=1,
            )
            active &= (rows < 0) | ~np.isfinite(best) | ~finite_points | overlap
        pending = np.flatnonzero(active)
        if not len(pending):
            continue
        source_bvh = (
            prepare_bvh(source, source, dtype=np.float64)
            if use_bvh and np.any(finite_points[pending])
            else None
        )
        for start in range(0, len(pending), 32):
            selected = pending[start : start + 32]
            points = centroids[selected]
            accelerated = finite_points[selected] & use_bvh & np.isfinite(best[selected])
            local = np.full(len(selected), -1, dtype=np.int64)
            if np.any(accelerated):
                subset = np.flatnonzero(accelerated)
                radius = reach[selected[subset], None]
                low = np.nextafter(points[subset] - radius, -np.inf)
                high = np.nextafter(points[subset] + radius, np.inf)
                finite_box = np.all(np.isfinite(low) & np.isfinite(high), axis=1)
                if np.any(~finite_box):
                    selected_fallback = subset[~finite_box]
                    local[selected_fallback] = exhaustive(
                        points[selected_fallback], source
                    )
                if np.any(finite_box):
                    if source_bvh is None:
                        raise RuntimeError("Finite BVH candidates require a source BVH.")
                    selected_bvh = subset[finite_box]
                    query_bvh = prepare_bvh(
                        low[finite_box], high[finite_box], dtype=np.float64
                    )
                    least = np.full(len(selected_bvh), np.inf, dtype=np.float64)
                    nearest = np.full(
                        len(selected_bvh), np.iinfo(np.int64).max, dtype=np.int64
                    )
                    for owners, columns in bvh_overlap_pair_blocks(
                        query_bvh, source_bvh, include_touching=True, item_block=4096
                    ):
                        squared = np.sum(
                            (points[selected_bvh[owners]] - source[columns]) ** 2,
                            axis=1,
                        )
                        prior = least.copy()
                        np.minimum.at(least, owners, squared)
                        nearest[least < prior] = np.iinfo(np.int64).max
                        equal = squared == least[owners]
                        np.minimum.at(nearest, owners[equal], columns[equal])
                    del query_bvh
                    found_local = nearest != np.iinfo(np.int64).max
                    local[selected_bvh[found_local]] = nearest[found_local]
            if np.any(~accelerated):
                local[~accelerated] = exhaustive(points[~accelerated], source)
            found = local >= 0
            if np.any(found):
                found_rows = selected[found]
                found_local = local[found]
                squared = np.sum(
                    (centroids[found_rows] - source[found_local]) ** 2, axis=1
                )
                candidate_rows = source_start + found_local
                replace = (
                    (rows[found_rows] < 0)
                    | (squared < best[found_rows])
                    | (np.isnan(squared) & ~np.isnan(best[found_rows]))
                    | (
                        (
                            (squared == best[found_rows])
                            | (np.isnan(squared) & np.isnan(best[found_rows]))
                        )
                        & (candidate_rows < rows[found_rows])
                    )
                )
                changed = found_rows[replace]
                rows[changed] = candidate_rows[replace]
                best[changed] = squared[replace]
        del source_bvh
    return rows


def _implicit_carrier_admission_tolerance(
    source: NativeImplicitSource, specification: VolumeMeshingSpec, /
) -> float:
    """Resolve an authored whole-surface bound for newly remeshed faces."""
    features = specification.protected_features
    if features:
        issues = _implicit_protected_feature_issues(source, features)
        if issues:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(issues),
                provider_code="native_implicit_carrier_remesh",
                stage="source_inspection",
            )
    declared = [feature.maximum_deviation for feature in features if feature.hard]
    if specification.maximum_deviation is not None:
        declared.append(specification.maximum_deviation)
    if not declared or min(declared) <= 0.0:
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Implicit carrier remeshing requires an explicitly positive whole-surface "
            "fidelity bound; exact or unspecified fidelity cannot establish one.",
            provider_code="native_implicit_carrier_remesh",
            stage="source_inspection",
        )
    profile = source.authoring_profile
    if profile is not None:
        declared.append(profile.tube_radius)
    return float(min(declared))


def _remesh_implicit_carrier(
    source: NativeImplicitSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    vertices: np.ndarray,
    faces: np.ndarray,
    carrier_distance: float,
    admission_tolerance: float,
    incidence: tuple[np.ndarray, np.ndarray] | None = None,
    /,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Remesh the scaffold carrier under the native zero-set band admission.

    Every created triangle is certified within ``admission_tolerance`` of the
    zero set and transversal to the field gradient; untouched scaffold
    triangles keep their independently established ``carrier_distance``. The
    result must meet the FIXED-fill radius-edge bound or the route refuses.
    """
    value, gradient = _implicit_field_box_programs(source.geometry)
    edge_length = _carrier_edge_length(vertices, faces, specification)
    try:
        remeshed = zero_set_remesh(
            vertices,
            faces,
            value,
            gradient,
            admission_tolerance,
            carrier_distance,
            edge_length,
            _CARRIER_MINIMUM_ANGLE_DEGREES,
            incidence=incidence,
            admission_limited_coarsening=True,
        )
    except (MeshcoreError, ValueError, OverflowError) as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"The implicit carrier zero-set remesh could not be certified: {error}",
            provider_code="native_implicit_carrier_remesh",
            stage="surface_meshing",
        ) from error
    worst = _maximum_face_radius_edge_ratio(remeshed.vertices, remeshed.triangles)
    if schedule.radius_edge_bound is None and not np.isfinite(worst):
        raise _failure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "The certified zero-set remesh has a nonrepresentable carrier triangle.",
            provider_code="native_implicit_carrier_remesh",
            stage="surface_meshing",
        )
    # A soft schedule keeps the remeshed carrier and publishes the resulting
    # tetrahedral shape rows; a nonrepresentable triangle stays a refusal.
    if (
        schedule.radius_edge_bound is not None
        and not worst <= schedule.radius_edge_bound
        and (_shape_targets_gated(schedule) or not np.isfinite(worst))
    ):
        raise _failure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "The certified zero-set remesh of the implicit carrier leaves a boundary "
            f"triangle with radius-edge ratio {worst:.6g} above the FIXED-fill bound "
            f"{schedule.radius_edge_bound:.6g}; no tetrahedron on it can meet the bound.",
            provider_code="native_implicit_carrier_remesh",
            stage="surface_meshing",
        )
    return (
        remeshed.vertices,
        remeshed.triangles,
        remeshed.distance,
        remeshed.counters,
        edge_length,
    )


def _scaffold_carrier_identities(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    vertices: np.ndarray,
    triangles: np.ndarray,
    /,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Carrier, vertex, and facet identities of the unmodified scaffold carrier."""
    carrier_id = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-volume-carrier",
            "source": source.binding_id,
            "discovery": discovery.evidence.evidence_id,
        }
    )
    scaffold = discovery.scaffold
    if (
        scaffold.triangle_root_edge_rows.shape != triangles.shape
        or scaffold.triangle_cell_rows.shape != (triangles.shape[0],)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit carrier triangles lack complete source-root ancestry.",
            provider_code="native_implicit_carrier_ancestry",
            stage="source_inspection",
        )
    vertex_root_rows = np.full(vertices.shape[0], -1, dtype=np.int64)
    for triangle_row in range(triangles.shape[0]):
        for vertex_column in range(3):
            vertex_row = int(triangles[triangle_row, vertex_column])
            root_row = int(scaffold.triangle_root_edge_rows[triangle_row, vertex_column])
            previous = int(vertex_root_rows[vertex_row])
            if previous not in (-1, root_row):
                raise _failure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "An implicit carrier vertex has conflicting source-root edge identities.",
                    provider_code="native_implicit_carrier_ancestry",
                    stage="source_inspection",
                )
            vertex_root_rows[vertex_row] = root_row
    if np.any(vertex_root_rows < 0):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "An implicit carrier vertex has no source-root edge identity.",
            provider_code="native_implicit_carrier_ancestry",
            stage="source_inspection",
        )
    vertex_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "implicit-volume-carrier-vertex",
                "carrier": carrier_id,
                "source_root_edge": tuple(
                    int(value)
                    for value in scaffold.root_edge_ids[int(root_row)].reshape((-1,))
                ),
            }
        )
        for root_row in vertex_root_rows
    )
    if len(set(vertex_ids)) != len(vertex_ids):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Distinct implicit carrier vertices share a source-root edge identity.",
            provider_code="native_implicit_carrier",
            stage="source_inspection",
        )
    source_triangle_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "adaptive-implicit-carrier-facet",
                "source": (source.source_id, source.source_revision),
                "cell": tuple(
                    int(value)
                    for value in scaffold.cell_indices[
                        int(scaffold.triangle_cell_rows[row])
                    ]
                ),
                "root_edges": tuple(
                    tuple(
                        int(value)
                        for value in scaffold.root_edge_ids[int(root_row)].reshape((-1,))
                    )
                    for root_row in scaffold.triangle_root_edge_rows[row]
                ),
            }
        )
        for row in range(triangles.shape[0])
    )
    if len(set(source_triangle_ids)) != triangles.shape[0]:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit carrier facets must retain unique source leaf/root-edge identities.",
            provider_code="native_implicit_carrier_ancestry",
            stage="source_inspection",
        )
    return carrier_id, vertex_ids, source_triangle_ids


def _implicit_remeshed_carrier_identity(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    edge_length: float,
    admission_tolerance: float,
    scaffold_distance: float,
    /,
) -> str:
    """Bind the actual source, remeshing controls, and both distance bounds."""
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-volume-carrier",
            "source": source.binding_id,
            "discovery": discovery.evidence.evidence_id,
            "zero_set_remesh": (
                edge_length,
                _CARRIER_MINIMUM_ANGLE_DEGREES,
                admission_tolerance,
                scaffold_distance,
            ),
        }
    )


def _implicit_plc_carrier(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
) -> _ImplicitCarrier:
    _domain_id, region_ids, _region_key = _implicit_region_authority(source, discovery)
    region_name = region_ids[0]
    carrier_region_seeds = _validated_implicit_region_seeds(
        discovery, specification, region_name
    )
    extracted = discovery.mesh
    if extracted is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume construction requires the complete adaptive triangle extraction.",
            provider_code="native_implicit_discovery_incomplete",
            stage="source_inspection",
        )
    vertices = np.asarray(extracted.vertices, dtype=np.float64)
    triangles = np.asarray(extracted.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume carrier coordinates and triangles must be three-dimensional.",
            provider_code="native_implicit_carrier",
            stage="source_inspection",
        )
    scaffold_distance = (
        discovery.evidence.maximum_surface_box_diagonal + discovery.policy.root_tolerance
    )
    before = _carrier_quality(vertices, triangles, "before")
    if np.isfinite(before[0][1]) and (
        schedule.radius_edge_bound is None or before[0][1] <= schedule.radius_edge_bound
    ):
        triangles = np.asarray(triangles, dtype=np.int64)
        carrier_id, vertex_ids, source_triangle_ids = _scaffold_carrier_identities(
            source, discovery, vertices, triangles
        )
        face_scaffold_rows = np.arange(triangles.shape[0], dtype=np.int64)
        carrier_distance = scaffold_distance
        evidence = (("carrier_remeshed", 0.0), *before)
    else:
        # The scaffold carrier provably cannot meet the FIXED-fill bound.
        admission_tolerance = _implicit_carrier_admission_tolerance(source, specification)
        scaffold_vertices, scaffold_triangles = vertices, triangles
        incidence = None
        if 3 * len(triangles) <= np.iinfo(np.int32).max:
            topology = extracted.topology
            incidence = (
                np.asarray(topology.vertex_face_offsets),
                np.asarray(topology.vertex_faces),
            )
        vertices, triangles, carrier_distance, counters, edge_length = (
            _remesh_implicit_carrier(
                source,
                specification,
                schedule,
                vertices,
                triangles,
                scaffold_distance,
                admission_tolerance,
                incidence,
            )
        )
        face_scaffold_rows = _nearest_scaffold_rows(
            scaffold_vertices, scaffold_triangles, vertices, triangles
        )
        carrier_id = _implicit_remeshed_carrier_identity(
            source, discovery, edge_length, admission_tolerance, scaffold_distance
        )
        vertex_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "implicit-volume-remeshed-carrier-vertex",
                    "carrier": carrier_id,
                    "row": row,
                }
            )
            for row in range(vertices.shape[0])
        )
        source_triangle_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "adaptive-implicit-remeshed-carrier-facet",
                    "source": (source.source_id, source.source_revision),
                    "carrier": carrier_id,
                    "row": row,
                }
            )
            for row in range(triangles.shape[0])
        )
        evidence = (
            ("carrier_remeshed", 1.0),
            *before,
            *_carrier_quality(vertices, triangles, "after"),
            ("carrier_remesh_edge_length", edge_length),
            ("carrier_remesh_admission_tolerance", admission_tolerance),
            ("carrier_remesh_unmet_count", float(counters[4])),
            ("carrier_remesh_transversality_refusals", float(counters[6])),
            ("carrier_remesh_distance_refusals", float(counters[7])),
        )
    carrier_revision = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-volume-carrier-revision",
            "carrier": carrier_id,
            "source_revision": source.source_revision,
        }
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        triangles,
        np.arange(triangles.shape[0], dtype=np.int64),
        np.tile(np.asarray(((-1, 0),), dtype=np.int64), (triangles.shape[0], 1)),
        region_ids,
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=tuple(() for _ in vertex_ids),
        facet_ids=source_triangle_ids,
        facet_occurrence_paths=tuple(() for _ in source_triangle_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=tuple(
            getattr(discovery, "region_occurrence_paths", ((),))
        ),
        boundary="fixed",
    )
    carrier = NativePlcSource(
        complex_,
        carrier_id,
        carrier_revision,
    )
    carrier_scope = MeshingScope(
        carrier.source_id,
        carrier.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        carrier.scope_entity_set_id(2),
        carrier.scope_entity_ids(2),
    )
    carrier_region_scope = MeshingScope(
        carrier.source_id,
        carrier.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        carrier.scope_entity_set_id(3),
        carrier.scope_entity_ids(3),
    )
    carrier_region_controls = tuple(
        RegionControl(
            carrier_region_scope,
            control.region_name,
            control.material_id,
            control.role,
            meshing_enabled=control.meshing_enabled,
        )
        for control in specification.region_controls
    )
    carrier_patch_controls = tuple(
        PatchControl(
            control.name,
            carrier_scope,
            control.adjacent_region_names,
            required=control.required,
        )
        for control in specification.patch_controls
    )
    carrier_controls = tuple(
        UniformSizeControl(
            carrier_scope,
            control.target_size,
            minimum_size=control.minimum_size,
            maximum_size=control.maximum_size,
            maximum_growth_rate=control.maximum_growth_rate,
            strength=control.strength,
            priority=control.priority,
        )
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    carrier_specification = VolumeMeshingSpec(
        specification.target,
        carrier_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=carrier_controls,
        region_controls=carrier_region_controls,
        patch_controls=carrier_patch_controls,
        region_seeds=carrier_region_seeds,
        size_compliance=specification.size_compliance,
        size_combination=specification.size_combination,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )
    face_scaffold_rows.setflags(write=False)
    return _ImplicitCarrier(
        carrier,
        carrier_specification,
        triangles,
        face_scaffold_rows,
        carrier_distance,
        evidence,
    )


def _implicit_surface_chart_witness(
    source_id: str,
    source_revision: str,
    discovery: AdaptiveImplicitSurface,
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: AdaptiveImplicitBoundarySource,
    /,
) -> SourceBoundaryChartWitness:
    """Bind each extracted triangle to its leaf and the three canonical roots."""
    scaffold = discovery.scaffold
    triangle_count = sum(block.cell_count for block in mesh.blocks)
    cell_rows = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    root_rows = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    if (
        cell_rows.shape != (triangle_count,)
        or root_rows.shape != (triangle_count, 3)
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
    ):
        raise ValueError(
            "Implicit triangle ancestry must cover every output cell and root edge."
        )
    cell_keys = np.asarray(scaffold.cell_indices, dtype=np.int64)[cell_rows]
    root_edge_keys = np.asarray(scaffold.root_edge_ids, dtype=np.int64)[root_rows]
    if cell_keys.shape != (triangle_count, 4) or root_edge_keys.shape != (
        triangle_count,
        3,
        2,
        3,
    ):
        raise ValueError(
            "Implicit source chart keys must retain exact leaf and root-edge dimensions."
        )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    parameters = np.broadcast_to(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (triangle_count, 3, 2),
    )
    chart_id_rows: list[str] = []
    for triangle_row in range(triangle_count):
        cell_key = tuple(
            int(cell_keys[triangle_row, axis]) for axis in range(cell_keys.shape[1])
        )
        triangle_edges = tuple(
            (
                tuple(
                    int(root_edge_keys[triangle_row, edge, 0, axis]) for axis in range(3)
                ),
                tuple(
                    int(root_edge_keys[triangle_row, edge, 1, axis]) for axis in range(3)
                ),
            )
            for edge in range(3)
        )
        chart_id_rows.append(
            canonical_fingerprint(
                {
                    "kind": "adaptive-implicit-scaffold-triangle",
                    "source": (query.source_id, query.source_revision),
                    "cell_key": cell_key,
                    "root_edges": triangle_edges,
                }
            )
        )
    chart_ids = tuple(chart_id_rows)
    surface_entity_id = source_entity_id(
        source_id, source_revision, "implicit-surface", "zero-set"
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source_id,
        source_revision=source_revision,
        source_query_id=query.source_revision,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=cell_ids,
        entity_global_ids=cell_ids,
        local_entity_indices=np.full(triangle_count, -1, dtype=np.int32),
        target_reference_vertices=parameters,
        source_chart_ids=chart_ids,
        source_entity_ids=(surface_entity_id,) * triangle_count,
        occurrence_paths=((),) * triangle_count,
        source_parameters=parameters,
        orientations=np.ones((triangle_count,), dtype=np.int8),
        implicit_cell_keys=cell_keys,
        implicit_root_edge_keys=root_edge_keys,
    )


def _implicit_layer_wall_chart_witness(
    source_id: str,
    source_revision: str,
    source_query: SourceBoundaryQuery,
    wall: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    """Bind an exact source wall to its adaptive root-scaffold triangles."""
    if not isinstance(source_query, SourceBoundaryQuery):
        raise TypeError("source_query must implement SourceBoundaryQuery.")
    if (source_query.source_id, source_query.source_revision) != (
        source_id,
        source_revision,
    ):
        raise ValueError(
            "Implicit layer source query must bind the source wall identity."
        )
    adaptive_query = (
        source_query
        if isinstance(source_query, AdaptiveImplicitBoundarySource)
        else getattr(source_query, "adaptive_source", None)
    )
    if not isinstance(adaptive_query, AdaptiveImplicitBoundarySource):
        raise ValueError(
            "Implicit layer curving requires the original adaptive boundary query."
        )
    adaptive_query.require_current()
    discovery = adaptive_query.surface
    source_mesh = discovery.mesh
    if source_mesh is None or not isinstance(wall, cell_mesh_owner.CellMesh):
        raise ValueError(
            "Implicit layer root correspondence requires its retained source mesh."
        )
    if (
        wall.topological_dimension != 2
        or wall.ambient_dimension != 3
        or wall.numeric_version != source_revision
        or len(wall.blocks) != 1
        or wall.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError("Implicit source wall changed its exact triangle-mesh identity.")
    source_points = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    wall_faces = np.asarray(wall.blocks[0].vertices, dtype=np.int64)
    expected_wall = cell_mesh_owner.CellMesh.from_triangles(
        source_points,
        source_faces,
        numeric_version=source_revision,
    )
    if (
        wall.mesh_id != expected_wall.mesh_id
        or wall.topology_id != expected_wall.topology_id
        or wall.geometry_id != expected_wall.geometry_id
        or not np.array_equal(wall_faces, source_faces)
        or not np.array_equal(wall_points, source_points)
        or not np.array_equal(
            np.asarray(wall.vertex_global_ids),
            np.asarray(expected_wall.vertex_global_ids),
        )
        or not np.array_equal(
            np.asarray(wall.blocks[0].global_ids),
            np.asarray(expected_wall.blocks[0].global_ids),
        )
    ):
        raise ValueError(
            "Implicit source wall must preserve the exact discovery mesh, cell order, and entity IDs."
        )
    witness = _implicit_surface_chart_witness(
        source_id,
        source_revision,
        discovery,
        wall,
        geometry,
        adaptive_query,
    )
    return witness


def _implicit_volume_chart_witness(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: AdaptiveImplicitBoundarySource,
    boundary_faces: np.ndarray,
    face_sources: np.ndarray,
    input_vertices: np.ndarray,
    target_face_ids: np.ndarray,
    carrier_triangles: np.ndarray,
    face_scaffold_rows: np.ndarray,
    /,
) -> SourceBoundaryChartWitness:
    """Bind every filled-volume exterior face to its implicit leaf/root edges.

    Carrier triangle ``k`` charts through scaffold triangle
    ``face_scaffold_rows[k]`` (itself for the scaffold carrier, the nearest one
    for a zero-set remeshed carrier).
    """
    from ...discretization._reference_cell import reference_cell_topology

    scaffold = discovery.scaffold
    original = discovery.mesh
    if original is None:
        raise ValueError(
            "Implicit volume witness requires the retained source triangle carrier."
        )
    source_triangles = np.asarray(carrier_triangles, dtype=np.int64)
    scaffold_rows = np.asarray(face_scaffold_rows, dtype=np.int64)
    scaffold_cell_rows = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    scaffold_root_rows = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    scaffold_count = np.asarray(original.faces).shape[0]
    if (
        scaffold_cell_rows.shape != (scaffold_count,)
        or scaffold_root_rows.shape != (scaffold_count, 3)
        or source_triangles.ndim != 2
        or source_triangles.shape[1] != 3
        or scaffold_rows.shape != (source_triangles.shape[0],)
        or np.any(scaffold_rows < 0)
        or np.any(scaffold_rows >= scaffold_count)
    ):
        raise ValueError("Implicit volume carrier lost triangle-to-root ancestry.")
    source_cell_rows = scaffold_cell_rows[scaffold_rows]
    source_root_rows = scaffold_root_rows[scaffold_rows]
    boundary_faces = np.asarray(boundary_faces, dtype=np.int64)
    face_sources = np.asarray(face_sources, dtype=np.int64)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or face_sources.shape != (boundary_faces.shape[0],)
        or np.any(face_sources < 0)
        or np.any(face_sources >= source_triangles.shape[0])
        or not np.array_equal(
            np.sort(face_sources), np.arange(source_triangles.shape[0], dtype=np.int64)
        )
    ):
        raise ValueError(
            "Implicit volume boundary must retain every original carrier facet."
        )

    input_vertices = np.asarray(input_vertices, dtype=np.int64)
    if input_vertices.shape != (mesh.coordinates.shape[0],):
        raise ValueError(
            "Implicit volume vertices lost their retained carrier-vertex ancestry."
        )

    reference = reference_cell_topology("tetrahedron")
    reference_vertices = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(reference.vertices, dtype=np.float64)
    )
    reference_faces = tuple(
        tuple(int(value) for value in row) for row in reference.entities[2]
    )
    if (
        mesh.topological_dimension != 3
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind not in ("simplex:3", "tetrahedron")
        or mesh.blocks[0].arity != 4
    ):
        raise ValueError("Implicit volume chart witnesses require tetrahedral cells.")
    block = mesh.blocks[0]
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    _, cell_faces = volume_cell_entity_rows(mesh, cell_ids)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    boundary_mask = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    face_owner: dict[int, tuple[int, int, int, tuple[int, ...]]] = {}
    for cell_row in range(cell_ids.size):
        for local_facet in range(cell_faces.shape[1]):
            face_row = int(cell_faces[cell_row, local_facet])
            if boundary_mask[face_row]:
                face_owner[int(face_ids[face_row])] = (
                    int(cell_ids[cell_row]),
                    cell_row,
                    local_facet,
                    reference_faces[local_facet],
                )
    target_face_ids = np.asarray(target_face_ids, dtype=np.int64)
    if target_face_ids.shape != (boundary_faces.shape[0],):
        raise ValueError(
            "Implicit volume boundary face IDs must align with retained triangles."
        )

    target_reference: list[np.ndarray] = []
    source_parameters: list[np.ndarray] = []
    parent_cells: list[int] = []
    local_facets: list[int] = []
    chart_ids: list[str] = []
    implicit_cells: list[np.ndarray] = []
    implicit_edges: list[np.ndarray] = []
    orientations: list[int] = []
    for face_row in range(boundary_faces.shape[0]):
        face = (
            int(boundary_faces[face_row, 0]),
            int(boundary_faces[face_row, 1]),
            int(boundary_faces[face_row, 2]),
        )
        source_facet = int(face_sources[face_row])
        entity_id = int(target_face_ids[face_row])
        owner = face_owner.get(int(entity_id))
        if owner is None:
            raise ValueError(
                "Implicit volume boundary face has no exterior cell/facet owner."
            )
        parent_id, cell_row, local_facet, local_vertices = owner
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
        if set(face) != {int(cell_vertices[local]) for local in local_vertices}:
            raise ValueError(
                "Implicit boundary face ID does not match its vertex incidence."
            )
        source_vertices = (
            int(source_triangles[source_facet, 0]),
            int(source_triangles[source_facet, 1]),
            int(source_triangles[source_facet, 2]),
        )
        volume_vertex_by_source_vertex: dict[int, int] = {}
        for volume_vertex in face:
            source_vertex = int(input_vertices[volume_vertex])
            if source_vertex < 0 or source_vertex in volume_vertex_by_source_vertex:
                raise ValueError(
                    "Fixed implicit boundary facets must retain original carrier vertices."
                )
            volume_vertex_by_source_vertex[source_vertex] = volume_vertex
        if set(volume_vertex_by_source_vertex) != set(source_vertices):
            raise ValueError(
                "Implicit boundary facet vertices do not retain their source triangle identity."
            )
        source_ordered_vertices = tuple(
            volume_vertex_by_source_vertex[value] for value in source_vertices
        )
        target_local = tuple(
            int(np.flatnonzero(cell_vertices == int(vertex))[0])
            for vertex in source_ordered_vertices
        )
        local_to_reference = tuple(reference_vertices[index] for index in target_local)
        target_reference.append(np.asarray(local_to_reference, dtype=np.float64))
        source_chart = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
        source_parameters.append(source_chart)
        permutation = tuple(
            local_vertices.index(local_vertex) for local_vertex in target_local
        )
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(3)
            for second in range(first + 1, 3)
        )
        target_orientation = -1 if inversions % 2 else 1
        parent_cells.append(parent_id)
        local_facets.append(local_facet)
        triangle_cell_row = int(source_cell_rows[int(source_facet)])
        cell_key = np.asarray(scaffold.cell_indices, dtype=np.int64)[triangle_cell_row]
        triangle_root_keys = np.asarray(scaffold.root_edge_ids, dtype=np.int64)[
            source_root_rows[int(source_facet)]
        ]
        implicit_cells.append(cell_key)
        implicit_edges.append(triangle_root_keys)
        cell_key_tuple = tuple(int(cell_key[axis]) for axis in range(4))
        triangle_root_edges = tuple(
            (
                (
                    int(triangle_root_keys[edge, 0, 0]),
                    int(triangle_root_keys[edge, 0, 1]),
                    int(triangle_root_keys[edge, 0, 2]),
                ),
                (
                    int(triangle_root_keys[edge, 1, 0]),
                    int(triangle_root_keys[edge, 1, 1]),
                    int(triangle_root_keys[edge, 1, 2]),
                ),
            )
            for edge in range(3)
        )
        chart_ids.append(
            canonical_fingerprint(
                {
                    "kind": "adaptive-implicit-scaffold-triangle",
                    "source": (query.source_id, query.source_revision),
                    "cell_key": cell_key_tuple,
                    "root_edges": triangle_root_edges,
                }
            )
        )
        orientations.append(target_orientation)
    surface_entity_id = source_entity_id(
        source.source_id, source.source_revision, "implicit-surface", "zero-set"
    )
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_revision,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(parent_cells, dtype=np.int64),
        entity_global_ids=target_face_ids,
        local_entity_indices=np.asarray(local_facets, dtype=np.int32),
        target_reference_vertices=np.asarray(target_reference, dtype=np.float64),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=(surface_entity_id,) * len(chart_ids),
        occurrence_paths=((),) * len(chart_ids),
        source_parameters=np.asarray(source_parameters, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.int8),
        implicit_cell_keys=np.asarray(implicit_cells, dtype=np.int64),
        implicit_root_edge_keys=np.asarray(implicit_edges, dtype=np.int64),
    )


def _implicit_remeshed_volume_chart_witness(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    """Propose root ancestry for the actual target boundary; proof remains separate."""
    scaffold = discovery.mesh
    if scaffold is None:
        raise ValueError(
            "Implicit volume witness requires the retained source triangle carrier."
        )
    from ...discretization._reference_cell import reference_cell_topology

    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind not in (
        "tetrahedron",
        "simplex:3",
    ):
        raise ValueError("Implicit volume chart witnesses require tetrahedral cells.")
    block = mesh.blocks[0]
    _, cell_faces = volume_cell_entity_rows(
        mesh, np.asarray(block.global_ids, dtype=np.int64)
    )
    mask = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if face_ids.shape != mask.shape:
        raise ValueError("Implicit volume boundary incidence is not aligned.")
    oriented = np.empty((len(mask), 3), dtype=np.int64)
    cells = np.asarray(block.vertices, dtype=np.int64)
    for local, corners in enumerate(reference_cell_topology(block.cell_kind).entities[2]):
        face_rows = cell_faces[:, local]
        exterior = mask[face_rows]
        oriented[face_rows[exterior]] = cells[exterior][:, corners]
    boundary = oriented[mask]
    rows = _nearest_scaffold_rows(
        np.asarray(scaffold.vertices, dtype=np.float64),
        np.asarray(scaffold.faces),
        np.asarray(mesh.coordinates, dtype=np.float64),
        boundary,
    )
    return _implicit_volume_chart_witness(
        source,
        discovery,
        mesh,
        geometry,
        discovery.boundary_source,
        boundary,
        np.arange(len(boundary), dtype=np.int64),
        np.arange(len(mesh.coordinates), dtype=np.int64),
        face_ids[mask],
        boundary,
        rows,
    )


def _fixed_source_contract_identity(
    source: NativeImplicitSource,
    policy: ImplicitSurfacePolicy,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> str:
    from ..._model._structure import model_structure_recipe
    from .._model_codec import _register_model_archive_types

    if source.grid is None:
        raise ValueError("Fixed implicit source identity requires its authored grid.")
    _register_model_archive_types()
    return canonical_fingerprint(
        {
            "kind": "fixed-implicit-source-contract",
            "source": source.source_id,
            "geometry_structure": model_structure_recipe(source.geometry, path="fixed.source.geometry"),
            "geometry_arrays": array_tree_fingerprint(source.geometry),
            "grid_structure": model_structure_recipe(source.grid, path="fixed.source.grid"),
            "grid_arrays": array_tree_fingerprint(source.grid),
            "policy": asdict(policy),
            "coordinate_contract": coordinate_contract.spatial_id,
        }
    )


def _fixed_preparation_identity(
    origin_geometry: CompiledGeometry,
    origin_grid: PreparedTensorGrid,
    surface_plan: ImplicitSurfacePlan,
    discovery_recipe: tuple[object, ...],
    specification: SurfaceMeshingSpec,
    revision_bindings: tuple[tuple[str, str], ...],
    /,
) -> str:
    """Bind the complete fixed plan and its scientific discovery inputs."""
    from ..._model._structure import model_structure_recipe
    from .._model_codec import _register_model_archive_types

    _register_model_archive_types()
    projection = surface_plan.projection
    return canonical_fingerprint(
        {
            "kind": "native-fixed-implicit-preparation",
            "discovery_structure": model_structure_recipe(discovery_recipe, path="fixed.discovery"),
            "discovery_arrays": array_tree_fingerprint(discovery_recipe),
            "specification_structure": model_structure_recipe(specification, path="fixed.specification"),
            "specification_arrays": array_tree_fingerprint(specification),
            "revision_bindings": revision_bindings,
            "origin_geometry_structure": model_structure_recipe(origin_geometry, path="fixed.origin"),
            "origin_geometry_arrays": array_tree_fingerprint(origin_geometry),
            "origin_grid_structure": model_structure_recipe(origin_grid, path="fixed.grid"),
            "origin_grid_arrays": array_tree_fingerprint(origin_grid),
            "surface_kernel": model_structure_recipe(surface_plan.kernel, path="fixed.kernel"),
            "projection_kernel": model_structure_recipe(projection.kernel, path="fixed.projection.kernel"),
            "surface_arrays": array_tree_fingerprint(surface_plan),
            "surface_schema": surface_plan.schema.schema_id,
            "surface_tolerance": asdict(surface_plan.tolerance),
            "surface_policy": asdict(surface_plan.policy),
            "surface_source": surface_plan.source_id,
            "surface_topology": surface_plan.topology_id,
            "surface_plan": surface_plan.plan_id,
            "projection_schema": projection.schema.schema_id,
            "projection_tolerance": asdict(projection.tolerance),
            "projection_policy": asdict(projection.policy),
            "projection_source": projection.source_id,
            "projection_plan": projection.plan_id,
        }
    )


@final
class _PreparedFixedImplicitRoute(StrictModule, NonTrainableState):
    """Fixed-lattice dual realization with independently certified source charts."""

    source: NativeImplicitSource
    specification: SurfaceMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    discovery_recipe: tuple[object, ...]
    surface_plan: ImplicitSurfacePlan
    origin_geometry: CompiledGeometry
    origin_grid: PreparedTensorGrid
    revision_bindings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativeImplicitSource,
        specification: SurfaceMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        discovery: AdaptiveImplicitSurface,
        surface_plan: ImplicitSurfacePlan,
        /,
        *,
        origin_geometry: CompiledGeometry | None = None,
        origin_grid: PreparedTensorGrid | None = None,
        revision_bindings: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        grid = source.grid if origin_grid is None else origin_grid
        origin = source.geometry if origin_geometry is None else origin_geometry
        if grid is None:
            raise ValueError("Fixed implicit preparation requires an authored grid.")
        from ...geometry._mesh_certificates import _same_tree_content

        domain = _grid_domain(source)
        regenerated_discovery = _discover(
            source, domain, _effective_policy(None, specification, domain)
        )
        recipe = discovery.archive_recipe
        discovery_data = tuple(getattr(recipe, name) for name in recipe.__dataclass_fields__)
        fresh_recipe = regenerated_discovery.archive_recipe
        fresh_data = tuple(
            getattr(fresh_recipe, name) for name in fresh_recipe.__dataclass_fields__
        )
        if not _same_tree_content(discovery_data, fresh_data):
            raise ValueError("Fixed implicit discovery is not the current source authority.")
        del regenerated_discovery
        contract_id = _fixed_source_contract_identity(source, surface_plan.policy, coordinate_contract)
        bindings = (
            ((source.source_revision, contract_id),)
            if revision_bindings is None else tuple(sorted(revision_bindings))
        )
        if (
            len({revision for revision, _ in bindings}) != len(bindings)
            or any(not revision or not identity for revision, identity in bindings)
            or dict(bindings).get(source.source_revision) != contract_id
        ):
            raise ValueError("Fixed implicit revision bindings do not match the current source.")
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.discovery_recipe = discovery_data
        self.surface_plan = surface_plan
        self.origin_geometry = origin
        self.origin_grid = grid
        self.revision_bindings = bindings
        self.preparation_id = _fixed_preparation_identity(
            origin, grid, surface_plan, self.discovery_recipe, specification, bindings
        )

    def require_current(self, /) -> AdaptiveImplicitSurface:
        from ...geometry._mesh_certificates import _same_tree_content

        source = self.source
        rebuilt = NativeImplicitSource(
            source.geometry, source.source_id, source.source_revision,
            grid=source.grid, discovery=source.discovery,
            authoring_profile=source.authoring_profile,
        )
        if not _same_tree_content(rebuilt, source):
            raise ValueError("Fixed implicit source binding is no longer current.")
        if (
            source.grid is None
            or not _same_tree_content(source.grid, self.origin_grid)
            or not _same_tree_content(source.geometry.kernel, self.origin_geometry.kernel)
            or not _same_tree_content(source.geometry.kernel, self.surface_plan.kernel)
            or not _same_tree_content(self.surface_plan.kernel, self.surface_plan.projection.kernel)
            or self.revision_bindings != tuple(sorted(self.revision_bindings))
            or dict(self.revision_bindings).get(source.source_revision)
            != _fixed_source_contract_identity(source, self.surface_plan.policy, self.coordinate_contract)
            or self.preparation_id != _fixed_preparation_identity(
                self.origin_geometry, self.origin_grid, self.surface_plan,
                self.discovery_recipe, self.specification, self.revision_bindings,
            )
        ):
            raise ValueError(
                "Fixed implicit preparation or discovery authority no longer matches "
                "its current source and plan."
            )
        regenerated = discover_implicit_surface(
            self.origin_geometry, self.origin_grid,
            policy=self.surface_plan.policy, source_id=self.surface_plan.source_id,
        )
        if not _same_tree_content(regenerated, self.surface_plan):
            raise ValueError("Fixed implicit plan no longer matches its owning discovery.")
        domain = _grid_domain(source)
        discovery = _discover(
            source, domain, _effective_policy(None, self.specification, domain)
        )
        recipe = discovery.archive_recipe
        current_data = tuple(getattr(recipe, name) for name in recipe.__dataclass_fields__)
        if not _same_tree_content(current_data, self.discovery_recipe):
            raise ValueError("Fixed implicit discovery authority is no longer current.")
        return discovery

    def execute(self, /) -> NativeMeshConstruction:
        from ._native_layer import _implicit_carrier_root_witness

        discovery = self.require_current()
        source = self.source
        realization = self.surface_plan.realize(source.geometry.state)
        if not bool(np.asarray(realization.accepted)):
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Fixed-lattice implicit realization failed with status {int(np.asarray(realization.evidence.status))}.",
                provider_code="native_implicit_fixed_realization",
                stage="surface_meshing",
            )
        vertices = np.asarray(realization.vertices, dtype=np.float64)
        faces = np.asarray(self.surface_plan.faces, dtype=np.int64)
        mesh = cell_mesh_owner.CellMesh.from_triangles(
            vertices, faces.astype(np.int32, copy=False), numeric_version=source.source_revision
        )
        geometry = CellGeometrySpec.affine(mesh)
        query = _NativeImplicitBoundaryQuery(
            discovery.boundary_source,
            source.source_id,
            source.source_revision,
            self.coordinate_contract.spatial_id,
            authoring_profile=source.authoring_profile,
        )
        distance = query.boundary_distance(vertices)
        if (
            distance.semantics != "certified"
            or not np.all(np.isfinite(distance.upper))
            or not np.array_equal(mesh.entity_set(0).entity_ids, mesh.vertex_global_ids)
        ):
            raise ValueError("Fixed implicit vertices require aligned certified source-distance bounds.")
        vertex_bound = np.asarray(distance.upper, dtype=np.float64)
        # Distance to a closed set is 1-Lipschitz.  The L1 diameter bounds
        # the displacement from a corner to any point of its triangle.
        corners = vertices[faces]
        diameter = max(
            math.nextafter(
                math.fsum(
                    math.nextafter(abs(float(a) - float(b)), math.inf)
                    for a, b in zip(first, second, strict=True)
                ),
                math.inf,
            )
            for triangle in corners
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        )
        fidelity_bound = math.nextafter(float(np.max(vertex_bound)) + diameter, math.inf)
        compliance = _compliance(self.specification, vertices, faces, fidelity_bound)
        if not compliance.passed:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "; ".join(compliance.issues),
                provider_code="native_implicit_compliance",
                stage="specification_compliance",
            )
        organization = _organization(
            source, discovery, self.coordinate_contract, mesh, faces, fidelity_bound,
            vertex_deviations=vertex_bound, fixed_plan_id=self.surface_plan.plan_id,
        )
        witness = _implicit_carrier_root_witness(
            source.source_id, source.source_revision, query, discovery.boundary_source,
            mesh, geometry,
        )
        scoped = tuple(
            (query, witness, feature.maximum_deviation)
            for feature in self.specification.protected_features if feature.hard
        )
        tolerance = self.specification.maximum_deviation
        if tolerance is None:
            tolerance = fidelity_bound
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            source.source_id,
            source.source_revision,
            fidelity_source=query,
            fidelity_tolerance=tolerance,
            protected_features=self.specification.protected_features,
            fidelity_witness=witness,
            scoped_fidelity=scoped,
            feature_source=source,
        )
        return NativeMeshConstruction(
            mesh, self.coordinate_contract, compliance, certification,
            geometry=geometry, organization=organization,
        )


@final
class _PreparedImplicitRoute(StrictModule, NonTrainableState):
    source: NativeImplicitSource
    specification: SurfaceMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    discovery: AdaptiveImplicitSurface

    def __init__(
        self,
        source: NativeImplicitSource,
        specification: SurfaceMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        discovery: AdaptiveImplicitSurface,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.discovery = discovery

    def execute(self, /) -> NativeMeshConstruction:
        source, specification, discovery = self.source, self.specification, self.discovery
        extracted = discovery.mesh
        if extracted is None:
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Adaptive implicit discovery has no positive triangle extraction.",
                provider_code="native_implicit_discovery_incomplete",
                stage="surface_meshing",
            )
        vertices = np.asarray(extracted.vertices, dtype=np.float64)
        faces = np.asarray(extracted.faces, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or vertices.shape[0] == 0
            or faces.shape[0] == 0
            or not np.all(np.isfinite(vertices))
        ):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Adaptive implicit extraction must contain finite coordinates and positive triangle connectivity.",
                provider_code="native_implicit_discovery_incomplete",
                stage="surface_meshing",
            )
        fidelity_bound = math.nextafter(
            discovery.evidence.maximum_surface_box_diagonal
            + discovery.policy.root_tolerance,
            math.inf,
        )
        if not math.isfinite(fidelity_bound) or fidelity_bound <= 0.0:
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Adaptive implicit extraction must retain a finite positive continuous source-fidelity bound.",
                provider_code="native_implicit_fidelity",
                stage="geometry_association",
            )
        compliance = _compliance(
            specification,
            vertices,
            faces,
            fidelity_bound,
        )
        if not compliance.passed:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Adaptive implicit surface extraction does not meet the requested physical controls: "
                + "; ".join(compliance.issues),
                provider_code="native_implicit_compliance",
                stage="specification_compliance",
            )
        mesh = cell_mesh_owner.CellMesh.from_triangles(
            vertices,
            faces.astype(np.int32, copy=False),
            numeric_version=source.source_revision,
        )
        organization = _organization(
            source,
            discovery,
            self.coordinate_contract,
            mesh,
            faces,
            fidelity_bound,
        )
        query = discovery.boundary_source
        fidelity_source: SourceBoundaryQuery = _NativeImplicitBoundaryQuery(
            query,
            source.source_id,
            source.source_revision,
            self.coordinate_contract.spatial_id,
            authoring_profile=source.authoring_profile,
        )
        geometry = CellGeometrySpec.affine(mesh)
        fidelity_witness = _implicit_surface_chart_witness(
            source.source_id,
            source.source_revision,
            discovery,
            mesh,
            geometry,
            query,
        )
        scoped_fidelity = tuple(
            (fidelity_source, fidelity_witness, feature.maximum_deviation)
            for feature in specification.protected_features
            if feature.hard
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            source.source_id,
            source.source_revision,
            fidelity_source=fidelity_source,
            fidelity_tolerance=specification.maximum_deviation,
            protected_features=specification.protected_features,
            fidelity_witness=fidelity_witness,
            scoped_fidelity=scoped_fidelity,
            feature_source=source,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
        )


def _implicit_volume_fidelity_bound(
    carrier_distance: float,
    construction: VolumeConstruction,
    /,
) -> float:
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or not boundary_faces.size
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit tetrahedral filling must retain a positive triangular boundary.",
            provider_code="native_implicit_volume_boundary",
            stage="geometry_association",
        )
    boundary_vertices = np.unique(boundary_faces.reshape((-1,)))
    deviations = np.asarray(construction.witness_deviations, dtype=np.float64)
    if (
        deviations.shape != (construction.mesh.coordinates.shape[0],)
        or np.any(~np.isfinite(deviations))
        or np.any(deviations < 0.0)
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume boundary witnesses must retain finite nonnegative carrier deviations.",
            provider_code="native_implicit_volume_fidelity",
            stage="geometry_association",
        )
    carrier_bound = float(np.max(deviations[boundary_vertices], initial=0.0))
    return float(np.nextafter(carrier_distance + carrier_bound, math.inf))


def _implicit_tetrahedron_quality(
    coordinates: np.ndarray,
    tetrahedra: np.ndarray,
    /,
) -> tuple[float, float]:
    points = np.asarray(coordinates, dtype=np.float64)[
        np.asarray(tetrahedra, dtype=np.int64)
    ]
    if points.ndim != 3 or points.shape[1:] != (4, 3) or not points.shape[0]:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Implicit tetrahedral construction must contain positive four-node cells.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        )
    edge_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=np.int64,
    )
    edge_lengths = np.linalg.norm(
        points[:, edge_pairs[:, 1]] - points[:, edge_pairs[:, 0]], axis=2
    )
    minimum_edges = np.min(edge_lengths, axis=1)
    if np.any(~np.isfinite(edge_lengths)) or np.any(minimum_edges <= 0.0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Implicit tetrahedral cells must retain finite positive edge lengths.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        )
    offsets = points[:, 1:] - points[:, :1]
    try:
        centers = np.linalg.solve(
            2.0 * offsets,
            np.sum(offsets * offsets, axis=2)[..., None],
        )[..., 0]
    except np.linalg.LinAlgError as error:
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Implicit tetrahedral cells must retain a finite circumsphere.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        ) from error
    radius_edge = float(
        np.max(np.linalg.norm(centers, axis=1) / minimum_edges, initial=0.0)
    )
    face_vertices = np.asarray(
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
        dtype=np.int64,
    )
    triangles = points[:, face_vertices]
    normals = np.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0],
    )
    centers_of_faces = np.mean(triangles, axis=2)
    centers_of_cells = np.mean(points, axis=1)
    inward = (
        np.sum(
            normals * (centers_of_faces - centers_of_cells[:, None, :]),
            axis=2,
        )
        < 0.0
    )
    normals = np.where(inward[:, :, None], -normals, normals)
    normal_lengths = np.linalg.norm(normals, axis=2)
    if np.any(~np.isfinite(normal_lengths)) or np.any(normal_lengths <= 0.0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Implicit tetrahedral cells must retain finite face normals.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        )
    unit_normals = normals / normal_lengths[:, :, None]
    face_pairs = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        dtype=np.int64,
    )
    cosines = np.sum(
        unit_normals[:, face_pairs[:, 0]] * unit_normals[:, face_pairs[:, 1]],
        axis=2,
    )
    dihedral = 180.0 - np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return radius_edge, float(np.min(dihedral))


def _implicit_volume_compliance(
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    construction: VolumeConstruction,
    fidelity_bound: float,
    carrier_evidence: tuple[tuple[str, float], ...],
    /,
) -> MeshingComplianceReport:
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    edge_pairs = np.unique(
        np.sort(
            np.concatenate(
                (
                    boundary_faces[:, (0, 1)],
                    boundary_faces[:, (1, 2)],
                    boundary_faces[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.linalg.norm(
        coordinates[edge_pairs[:, 1]] - coordinates[edge_pairs[:, 0]], axis=1
    )
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise _failure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Implicit tetrahedral filling produced no finite positive boundary edges.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        )
    if not construction.mesh.blocks or any(
        block.cell_kind not in ("simplex:3", "tetrahedron") or block.arity != 4
        for block in construction.mesh.blocks
    ):
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume schedules require tetrahedral carrier output.",
            provider_code="native_implicit_volume_quality",
            stage="specification_compliance",
        )
    tetrahedra = np.concatenate(
        [
            np.asarray(block.vertices, dtype=np.int64)
            for block in construction.mesh.blocks
        ],
        axis=0,
    )
    maximum_radius_edge, minimum_dihedral = _implicit_tetrahedron_quality(
        coordinates,
        tetrahedra,
    )
    requested: list[tuple[str, float]] = []
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", schedule.radius_edge_bound))
    requested.append(("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees))
    achieved: list[tuple[str, float]] = [
        ("maximum_radius_edge_ratio", maximum_radius_edge),
        ("minimum_dihedral_degrees", minimum_dihedral),
        ("refinement_status", float(int(construction.refinement_status))),
        ("improvement_status", float(int(construction.improvement_status))),
        ("refinement_unmet_count", float(len(construction.refinement_unmet))),
        ("improvement_unmet_count", float(len(construction.improvement_unmet))),
        (
            "refinement_inserted_vertices",
            float(construction.refinement_inserted_vertices),
        ),
        ("final_quality_unmet_count", float(len(construction.final_quality_unmet))),
        ("source_fidelity_bound", fidelity_bound),
        *carrier_evidence,
    ]
    gated_rows, soft_rows = _split_shape_target_rows(
        schedule, construction.final_quality_schedule_unmet
    )
    shape_gated = _shape_targets_gated(schedule)
    if not shape_gated:
        achieved.extend(_soft_shape_target_evidence(construction, soft_rows))
    issues: list[str] = []
    issues.extend(
        f"native_final_quality:{criterion}"
        for criterion, _reason, _cell_id, _value in gated_rows
    )
    if schedule.radius_edge_bound is None and not np.isfinite(maximum_radius_edge):
        issues.append("nonrepresentable_radius_edge")
    if schedule.minimum_dihedral_degrees == 0.0 and not np.isfinite(minimum_dihedral):
        issues.append("nonrepresentable_dihedral")
    if (
        schedule.radius_edge_bound is not None
        and maximum_radius_edge > schedule.radius_edge_bound
        and (shape_gated or not np.isfinite(maximum_radius_edge))
    ):
        issues.append("radius_edge_bound")
    if shape_gated and minimum_dihedral < schedule.minimum_dihedral_degrees:
        issues.append("minimum_dihedral_degrees")
    size_policy = specification.size_compliance
    cell_edges = np.unique(
        np.sort(
            np.concatenate(
                [tetrahedra[:, (i, j)] for i in range(4) for j in range(i + 1, 4)]
            ),
            axis=1,
        ),
        axis=0,
    )
    maximum_cell_edge = float(
        np.max(
            np.linalg.norm(
                coordinates[cell_edges[:, 1]] - coordinates[cell_edges[:, 0]], axis=1
            )
        )
    )
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        key = f"size:{control.control_id}"
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.maximum_growth_rate is not None:
            requested.append((f"{key}:maximum_growth_rate", control.maximum_growth_rate))
        statistics = {
            name: float(np.quantile(lengths, 0.5 if name == "p50" else 0.95))
            for name in size_policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(lengths))),
                (f"{key}:maximum_edge", float(np.max(lengths))),
                (f"{key}:maximum_cell_edge", maximum_cell_edge),
                (f"{key}:size_field_growth_rate", 1.0),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            size_policy.absolute_tolerance
            + size_policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if control.minimum_size is not None and float(np.min(lengths)) < (
                control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            # Donor acceptance (tools/meshing_qualification/scenarios/implicit.py
            # lines 560-574 of feature/native-meshing-20260929): under the
            # circumradius fill every tetrahedron edge, not only the carrier's,
            # is at most the maximum size widened by the compliance tolerance on
            # that maximum.
            if control.maximum_size is not None and maximum_cell_edge > (
                control.maximum_size
                + size_policy.absolute_tolerance
                + size_policy.relative_tolerance * control.maximum_size
            ):
                issues.append(f"maximum_size:{control.control_id}")
            if np.any(lengths > control.target_size + tolerance):
                issues.append(f"local_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and 1.0 > control.maximum_growth_rate + tolerance
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.append(
            (f"protected:{feature.feature_id}:source_fidelity_bound", fidelity_bound)
        )
        if feature.hard and fidelity_bound > feature.maximum_deviation:
            issues.append(f"protected:{feature.feature_id}:maximum_deviation")
    if specification.maximum_deviation is not None:
        requested.append(("maximum_deviation", specification.maximum_deviation))
        if fidelity_bound > specification.maximum_deviation:
            issues.append("maximum_deviation")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _implicit_volume_organization(
    source: NativeImplicitSource,
    discovery: AdaptiveImplicitSurface,
    specification: VolumeMeshingSpec,
    contract: SpatialCoordinateContract,
    construction: VolumeConstruction,
    fidelity_bound: float,
    /,
) -> NativeMeshOrganization:
    cover = discovery.cover
    if cover is None:
        raise _failure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Implicit volume publication requires its retained adaptive cover.",
            provider_code="native_implicit_cover_missing",
            stage="geometry_association",
        )
    domain_id, region_names, region_key = _implicit_region_authority(source, discovery)
    del domain_id
    region_name = region_names[0]
    mesh = construction.mesh
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    face_set = mesh.entity_set(2)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    boundary_face_ids = _boundary_face_entities(mesh, construction)
    boundary_face_mask = np.isin(face_ids, boundary_face_ids)
    edge_set = mesh.entity_set(1)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
    edge_vertices, _ = simplex_volume_entities(mesh)
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    boundary_edge_keys = {
        tuple(sorted((int(face[offset]), int(face[(offset + 1) % 3]))))
        for face in boundary_faces.tolist()
        for offset in range(3)
    }
    boundary_edge_mask = np.asarray(
        [tuple(sorted(edge)) in boundary_edge_keys for edge in edge_vertices.tolist()],
        dtype=np.bool_,
    )
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    boundary_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.unique(boundary_faces.reshape((-1,)))
    ]
    boundary_vertex_mask = np.isin(vertex_ids, boundary_vertex_ids)
    surface_key = source.source_entity_key_for_scope(2, int(_scope_ids(source)[0]))

    def association(
        dimension: int,
        targets: np.ndarray,
        boundary_mask: np.ndarray,
        /,
        *,
        exact: bool,
    ) -> GeometryAssociation:
        source_ids = tuple(
            surface_key[0] if boundary else region_key[0]
            for boundary in boundary_mask.tolist()
        )
        roles = tuple(
            GeometrySourceEntityRole.FACET
            if boundary
            else GeometrySourceEntityRole.REGION
            for boundary in boundary_mask.tolist()
        )
        paths = tuple(
            surface_key[1] if boundary else region_key[1]
            for boundary in boundary_mask.tolist()
        )
        residuals = np.where(boundary_mask, fidelity_bound, 0.0)
        return GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            source.source_id,
            source.source_revision,
            mesh.entity_set(dimension).entity_set_id,
            targets,
            source_ids,
            residuals,
            exact=exact,
            source_roles=roles,
            source_occurrence_paths=paths,
        )

    associations = (
        association(0, vertex_ids, boundary_vertex_mask, exact=False),
        association(1, edge_ids, boundary_edge_mask, exact=False),
        association(2, face_ids, boundary_face_mask, exact=False),
        association(
            3,
            cell_ids,
            np.zeros((cell_ids.size,), dtype=np.bool_),
            exact=True,
        ),
    )
    cell_scope = _mesh_scope(mesh, 3, cell_ids)
    boundary_scope = _mesh_scope(mesh, 2, boundary_face_ids)
    region_control = (
        specification.region_controls[0] if specification.region_controls else None
    )
    region_seed = specification.region_seeds[0] if specification.region_seeds else None
    material_id = (
        region_control.material_id
        if region_control is not None
        else region_name
        if region_seed is None
        else region_seed.material_id
    )
    region_role = (
        region_control.role
        if region_control is not None
        else RegionRole.USER
        if region_seed is None
        else region_seed.role
    )
    zone = MeshZone(
        region_name,
        MeshZoneRole.REGION,
        cell_scope,
        material_id=material_id,
        region_role=region_role,
    )
    patch_controls = specification.patch_controls
    patches = tuple(
        MeshPatch(
            control.name,
            boundary_scope,
            connected=_faces_connected(boundary_faces),
            adjacent_zone_ids=(zone.zone_id,),
            source_adjacent_region_ids=(region_key[0], None),
            source_adjacent_region_occurrence_paths=(region_key[1], None),
        )
        for control in patch_controls
    ) or (
        MeshPatch(
            "implicit-zero-set",
            boundary_scope,
            connected=_faces_connected(boundary_faces),
            adjacent_zone_ids=(zone.zone_id,),
            source_adjacent_region_ids=(region_key[0], None),
            source_adjacent_region_occurrence_paths=(region_key[1], None),
        ),
    )
    used, remapped = np.unique(boundary_faces.reshape((-1,)), return_inverse=True)
    boundary_mesh = cell_mesh_owner.CellMesh.from_triangles(
        np.asarray(mesh.coordinates, dtype=np.float64)[used],
        remapped.reshape((-1, 3)).astype(np.int32, copy=False),
        numeric_version=source.source_revision,
    )
    boundary_model = SurfaceModel(
        boundary_mesh,
        SurfaceMetadata(
            source_id=source.source_id,
            source_revision=source.source_revision,
            coordinate_contract=contract,
            provenance=(
                "geometry-owned-adaptive-implicit-volume-boundary",
                source.binding_id,
                discovery.evidence.evidence_id,
                cover.cover_id,
            ),
            cell_tags=("implicit-zero-set",) * boundary_faces.shape[0],
        ),
    )
    labels = (
        MeshLabel(f"region:{region_name}", cell_scope),
        *(MeshLabel(patch.name, boundary_scope) for patch in patches),
    )
    return NativeMeshOrganization(
        boundary=boundary_model,
        patches=patches,
        zones=(zone,),
        labels=labels,
        associations=associations,
    )


@final
class _PreparedImplicitVolumeRoute(StrictModule, NonTrainableState):
    source: NativeImplicitSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    discovery: AdaptiveImplicitSurface
    carrier: NativePlcSource
    carrier_specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    carrier_triangles: np.ndarray
    face_scaffold_rows: np.ndarray
    carrier_distance: float = eqx.field(static=True)
    carrier_evidence: tuple[tuple[str, float], ...] = eqx.field(static=True)

    def __init__(
        self,
        source: NativeImplicitSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        discovery: AdaptiveImplicitSurface,
        schedule: NativeVolumeSchedule,
        carrier: _ImplicitCarrier,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.discovery = discovery
        self.carrier = carrier.carrier
        self.carrier_specification = carrier.specification
        self.schedule = schedule
        triangles = np.array(carrier.triangles, dtype=np.int64, copy=True)
        rows = np.array(carrier.face_scaffold_rows, dtype=np.int64, copy=True)
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or rows.shape != (triangles.shape[0],)
        ):
            raise ValueError(
                "Implicit carrier scaffold rows must align with carrier triangles."
            )
        triangles.setflags(write=False)
        rows.setflags(write=False)
        self.carrier_triangles = triangles
        self.face_scaffold_rows = rows
        self.carrier_distance = carrier.distance
        self.carrier_evidence = carrier.evidence

    def execute(self, /) -> NativeMeshConstruction:
        mapped_geometry = (
            None
            if self.carrier.mapped_domain is None
            else self.carrier.mapped_domain.source_geometry
        )
        source_map_id = (
            ""
            if mapped_geometry is None or mapped_geometry.source_map_id is None
            else mapped_geometry.source_map_id
        )
        try:
            construction = generate_plc_volume(
                self.carrier.complex,
                self.carrier_specification,
                self.schedule,
                source_id=self.carrier.source_id,
                source_revision=self.carrier.source_revision,
                source_map_id=source_map_id,
                frame_id=self.coordinate_contract.spatial_id,
                # The donor's implicit volume refined to circumradius h; the
                # maximum size is checked at compliance, not capped in the fill.
                size_convention="circumradius",
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise _failure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Native implicit PLC volume filling failed: {error}",
                provider_code="native_implicit_volume_fill",
                stage="volume_meshing",
            ) from error
        mesh = construction.mesh
        if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Native implicit volume filling must return a tetrahedral three-dimensional mesh.",
                provider_code="native_implicit_volume_fill",
                stage="volume_meshing",
            )
        region_ids = tuple(getattr(self.discovery, "region_ids", ()))
        cell_regions = np.asarray(construction.cell_regions, dtype=np.int64)
        cell_count = sum(block.cell_count for block in mesh.blocks)
        if (
            tuple(construction.region_ids) != region_ids
            or cell_regions.shape != (cell_count,)
            or np.any(cell_regions != 0)
        ):
            raise _failure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Implicit tetrahedral cells must cover only the authored adaptive interior region.",
                provider_code="native_implicit_volume_regions",
                stage="domain_coverage",
            )
        fidelity_bound = _implicit_volume_fidelity_bound(
            self.carrier_distance,
            construction,
        )
        compliance = _implicit_volume_compliance(
            self.specification,
            self.schedule,
            construction,
            fidelity_bound,
            self.carrier_evidence,
        )
        if not compliance.passed:
            raise _failure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Adaptive implicit volume construction does not meet requested physical controls: "
                + "; ".join(compliance.issues),
                provider_code="native_implicit_volume_compliance",
                stage="specification_compliance",
            )
        organization = _implicit_volume_organization(
            self.source,
            self.discovery,
            self.specification,
            self.coordinate_contract,
            construction,
            fidelity_bound,
        )
        adaptive_source = self.discovery.boundary_source
        fidelity_source: SourceBoundaryQuery = _NativeImplicitBoundaryQuery(
            adaptive_source,
            self.source.source_id,
            self.source.source_revision,
            self.coordinate_contract.spatial_id,
            authoring_profile=self.source.authoring_profile,
        )
        geometry = CellGeometrySpec.affine(mesh)
        fidelity_witness = _implicit_volume_chart_witness(
            self.source,
            self.discovery,
            mesh,
            geometry,
            adaptive_source,
            construction.boundary_faces,
            construction.face_source_facets,
            construction.input_vertices,
            _boundary_face_entities(mesh, construction),
            self.carrier_triangles,
            self.face_scaffold_rows,
        )
        scoped_fidelity = tuple(
            (fidelity_source, fidelity_witness, feature.maximum_deviation)
            for feature in self.specification.protected_features
            if feature.hard
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_implicit"),
            self.source.source_id,
            self.source.source_revision,
            domain=self.discovery,
            cell_regions=construction.cell_regions,
            fidelity_source=fidelity_source,
            fidelity_tolerance=self.specification.maximum_deviation,
            protected_features=self.specification.protected_features,
            fidelity_witness=fidelity_witness,
            scoped_fidelity=scoped_fidelity,
            feature_source=self.source,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
        )


def prepare_native_fixed_rebind(
    source: NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    prior: _PreparedFixedImplicitRoute,
    /,
) -> _PreparedFixedImplicitRoute:
    """Rebind a qualified fixed lattice to a new source revision, without rediscovery."""
    from ...geometry._mesh_certificates import _same_tree_content

    prior.require_current()
    surface_plan = prior.surface_plan
    if (
        source.source_revision == prior.source.source_revision
        and not _same_tree_content(source.geometry.state, prior.source.geometry.state)
    ):
        raise ValueError("Changed fixed implicit geometry requires a new source revision.")
    policy = options.implicit_policy
    if not isinstance(policy, ImplicitSurfacePolicy) or source.grid is None:
        raise ValueError("Fixed implicit rebind requires its authored fixed-grid policy.")
    if implicit_support_issues(source, specification, options):
        raise ValueError("Fixed implicit rebind does not support the requested surface controls.")
    axes = tuple(np.asarray(axis.point_coordinates) for axis in source.grid.structured_axes)
    grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    if (
        surface_plan.source_id != source.source_id
        or surface_plan.schema != source.geometry.schema
        or surface_plan.tolerance != source.geometry.tolerance
        or surface_plan.policy != policy
        or not _same_tree_content(surface_plan.kernel, source.geometry.kernel)
        or not _same_tree_content(prior.origin_grid, source.grid)
        or not np.array_equal(np.asarray(surface_plan.grid_points), grid_points)
    ):
        raise ValueError("Fixed implicit rebind changed its source kernel, schema, lattice, or policy.")
    contract_id = _fixed_source_contract_identity(source, policy, coordinate_contract)
    bindings = dict(prior.revision_bindings)
    previous = bindings.get(source.source_revision)
    if previous is not None and previous != contract_id:
        raise ValueError("A fixed implicit source revision cannot be reused for different scientific content.")
    bindings[source.source_revision] = contract_id
    realization = surface_plan.realize(source.geometry.state)
    if not bool(np.asarray(realization.accepted)):
        raise ValueError(
            f"Fixed implicit rebind crossed a stopped event: status {int(np.asarray(realization.evidence.status))}."
        )
    discovery = _resolve_discovery(source, specification, None)
    return _PreparedFixedImplicitRoute(
        source, specification, coordinate_contract, discovery, surface_plan,
        origin_geometry=prior.origin_geometry, origin_grid=prior.origin_grid,
        revision_bindings=tuple(sorted(bindings.items())),
    )


@checked
def prepare_native_implicit(
    source: NativeImplicitSource | AnalyticImplicitProfile,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> (
    _PreparedImplicitRoute
    | _PreparedFixedImplicitRoute
    | _PreparedImplicitVolumeRoute
    | PreparedRestrictedImplicitVolume
):
    """Prepare adaptive surface/volume or direct restricted analytic volume data."""
    if isinstance(source, AnalyticImplicitProfile):
        if isinstance(specification, VolumeMeshingSpec) and options.route in (
            None,
            "implicit_restricted_delaunay",
        ):
            if options.implicit_policy is not None:
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "AdaptiveImplicitSurfacePolicy cannot alter the established-reach restricted-Delaunay route.",
                    provider_code="native_implicit_preflight",
                    stage="source_inspection",
                )
            schedule = (
                NativeVolumeSchedule()
                if options.volume_schedule is None
                else options.volume_schedule
            )
            if not isinstance(schedule, NativeVolumeSchedule):
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Restricted analytic volume generation requires a NativeVolumeSchedule.",
                    provider_code="native_implicit_preflight",
                    stage="source_inspection",
                )
            return prepare_restricted_implicit_volume(
                source,
                specification,
                schedule,
                coordinate_contract,
            )
        if isinstance(specification, VolumeMeshingSpec):
            if options.route != "implicit_adaptive_tetrahedral":
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "An analytic volume profile uses the established-reach restricted-Delaunay route by default; adaptive discovery requires explicit implicit_adaptive_tetrahedral selection.",
                    provider_code="native_implicit_preflight",
                    stage="source_inspection",
                )
        elif isinstance(specification, SurfaceMeshingSpec):
            if options.route not in (None, "implicit_surface"):
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Analytic surface profiles require the implicit_surface adaptive extraction route.",
                    provider_code="native_implicit_preflight",
                    stage="source_inspection",
                )
        adaptive_policy = options.implicit_policy
        if adaptive_policy is not None and not isinstance(
            adaptive_policy, AdaptiveImplicitSurfacePolicy
        ):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Adaptive implicit construction requires AdaptiveImplicitSurfacePolicy.",
                provider_code="native_implicit_preflight",
                stage="source_inspection",
            )
        source = normalize_analytic_implicit_source(
            source, adaptive_policy, coordinate_contract, specification=specification
        )
    if isinstance(specification, SurfaceMeshingSpec):
        unsupported = implicit_support_issues(source, specification, options)
        if unsupported:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(unsupported),
                provider_code="native_implicit_preflight",
                stage="source_inspection",
            )
        if isinstance(options.implicit_policy, ImplicitSurfacePolicy):
            if source.grid is None:
                raise ValueError("Fixed-lattice implicit meshing requires the authored grid.")
            surface_plan = discover_implicit_surface(
                source.geometry, source.grid,
                policy=options.implicit_policy, source_id=source.source_id,
            )
            discovery = _resolve_discovery(source, specification, None)
            return _PreparedFixedImplicitRoute(
                source, specification, coordinate_contract, discovery, surface_plan
            )
        discovery = _resolve_discovery(source, specification, options.implicit_policy)
        return _PreparedImplicitRoute(
            source,
            specification,
            coordinate_contract,
            discovery,
        )
    if isinstance(specification, VolumeMeshingSpec):
        unsupported = implicit_volume_support_issues(source, specification, options)
        if unsupported:
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(unsupported),
                provider_code="native_implicit_volume_preflight",
                stage="source_inspection",
            )
        policy = (
            source.discovery.policy
            if source.discovery is not None and options.implicit_policy is None
            else options.implicit_policy
        )
        if policy is not None and not isinstance(policy, AdaptiveImplicitSurfacePolicy):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Implicit volume generation requires AdaptiveImplicitSurfacePolicy.",
                provider_code="native_implicit_volume_preflight",
                stage="source_inspection",
            )
        discovery = _resolve_discovery(source, specification, policy)
        schedule = (
            NativeVolumeSchedule()
            if options.volume_schedule is None
            else options.volume_schedule
        )
        if not isinstance(schedule, NativeVolumeSchedule):
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Implicit volume generation requires a NativeVolumeSchedule.",
                provider_code="native_implicit_volume_preflight",
                stage="source_inspection",
            )
        return _PreparedImplicitVolumeRoute(
            source,
            specification,
            coordinate_contract,
            discovery,
            schedule,
            _implicit_plc_carrier(source, discovery, specification, schedule),
        )
    raise TypeError("specification must be SurfaceMeshingSpec or VolumeMeshingSpec.")


__all__ = [
    "implicit_support_issues",
    "implicit_volume_support_issues",
    "normalize_analytic_implicit_source",
    "prepare_native_implicit",
]
