#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope

from .._bvh import bvh_nearest_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_finite_float
from ..typing import checked, parse


class SizeControlStrength(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class SizeFieldDomain(StrEnum):
    EUCLIDEAN_VOLUME = "euclidean_volume"
    SURFACE_GEODESIC = "surface_geodesic"
    MESH_GEODESIC = "mesh_geodesic"
    BACKGROUND_GRID = "background_grid"
    SAMPLE_CLOUD = "sample_cloud"


class SizeCombinationPolicy(StrEnum):
    REJECT_HARD_CONFLICTS = "reject_hard_conflicts"
    EXPLICIT_PRIORITY = "explicit_priority"


type SizeStatistic = Literal["p50", "p95"]


def size_statistic_quantile(statistic: SizeStatistic, /) -> float:
    """Return the quantile corresponding to a size-compliance statistic."""
    selected = parse(statistic, SizeStatistic, "SizeCompliancePolicy.target_statistics")
    match selected:
        case "p50":
            return 0.5
        case "p95":
            return 0.95
        case _:
            assert_never(selected)


class SizeCompliancePolicy(StrictModule, NonTrainableState):
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    target_statistics: tuple[SizeStatistic, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        target_statistics: tuple[SizeStatistic, ...] = ("p50", "p95"),
    ) -> None:
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        normalized_statistics = tuple(str(value).strip() for value in target_statistics)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError(
                f"SizeCompliancePolicy.tolerances must be finite and non-negative; got {(absolute, relative)!r}"
            )
        if not normalized_statistics or len(set(normalized_statistics)) != len(
            normalized_statistics
        ):
            raise ValueError(
                f"SizeCompliancePolicy.target_statistics must contain distinct supported statistics; got {normalized_statistics!r}"
            )
        statistics = tuple(
            parse(value, SizeStatistic, "SizeCompliancePolicy.target_statistics")
            for value in normalized_statistics
        )
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.target_statistics = statistics
        self.policy_id = canonical_fingerprint(
            {
                "kind": "size-compliance-policy",
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "target_statistics": statistics,
            }
        )

    def tolerance(self, value: float, /) -> float:
        """Absolute size tolerance explicitly requested at this magnitude."""
        return self.absolute_tolerance + self.relative_tolerance * abs(value)


class UniformSizeControl(StrictModule, NonTrainableState):
    """Bind a physical edge-size target and optional bounds to a declared scope.

    A hard target requires each statistic selected by ``SizeCompliancePolicy``
    to equal the target within its declared tolerance, including for coarse
    requests. It is not merely an upper edge-length bound. Seed preparation
    may add intervals required by source topology, but cannot exempt their
    published lengths from these obligations.
    """

    scope: _dependency_phydrax_meshing__scope.MeshingScope
    target_size: float = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    maximum_growth_rate: float | None = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        target_size: float,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        maximum_growth_rate: float | None = None,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        target = positive_finite_float(target_size, "target_size")
        minimum = (
            None
            if minimum_size is None
            else positive_finite_float(minimum_size, "minimum_size")
        )
        maximum = (
            None
            if maximum_size is None
            else positive_finite_float(maximum_size, "maximum_size")
        )
        growth = None if maximum_growth_rate is None else float(maximum_growth_rate)
        if minimum is not None and minimum > target:
            raise ValueError(
                f"UniformSizeControl.minimum_size must not exceed target_size when supplied; got minimum={minimum!r}, target={target!r}"
            )
        if maximum is not None and target > maximum:
            raise ValueError(
                f"UniformSizeControl.maximum_size must not be below target_size when supplied; got maximum={maximum!r}, target={target!r}"
            )
        if growth is not None and (not np.isfinite(growth) or growth < 1.0):
            raise ValueError(
                f"UniformSizeControl.maximum_growth_rate must be finite and at least one; got {growth!r}"
            )
        strength = parse(strength, SizeControlStrength, "SizeControl.strength")
        self.scope = scope
        self.target_size = target
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.maximum_growth_rate = growth
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "uniform-size-control",
                "scope": scope.scope_id,
                "target_size": target,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "maximum_growth_rate": growth,
                "strength": strength.value,
                "priority": int(priority),
            }
        )


class CurvatureSizeControl(StrictModule, NonTrainableState):
    scope: _dependency_phydrax_meshing__scope.MeshingScope
    normal_angle: float = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    use_faceted_curvature: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        normal_angle: float,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        use_faceted_curvature: bool = False,
        strength: SizeControlStrength = SizeControlStrength.SOFT,
        priority: int = 0,
    ) -> None:
        angle = float(normal_angle)
        minimum = (
            None
            if minimum_size is None
            else positive_finite_float(minimum_size, "minimum_size")
        )
        maximum = (
            None
            if maximum_size is None
            else positive_finite_float(maximum_size, "maximum_size")
        )
        if not np.isfinite(angle) or angle <= 0.0 or angle >= np.pi:
            raise ValueError(
                f"CurvatureSizeControl.normal_angle must lie strictly between zero and pi; got {angle!r}"
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"SizeControl.minimum_size must not exceed maximum_size; got minimum={minimum!r}, maximum={maximum!r}"
            )
        strength = parse(strength, SizeControlStrength, "SizeControl.strength")
        self.scope = scope
        self.normal_angle = angle
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.use_faceted_curvature = bool(use_faceted_curvature)
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "curvature-size-control",
                "scope": scope.scope_id,
                "normal_angle": angle,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "use_faceted_curvature": bool(use_faceted_curvature),
                "strength": strength.value,
                "priority": int(priority),
            }
        )


class ProximitySizeControl(StrictModule, NonTrainableState):
    source_scope: _dependency_phydrax_meshing__scope.MeshingScope
    target_scope: _dependency_phydrax_meshing__scope.MeshingScope
    elements_per_gap: int = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    opposite_normals_only: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_scope: _dependency_phydrax_meshing__scope.MeshingScope,
        target_scope: _dependency_phydrax_meshing__scope.MeshingScope,
        elements_per_gap: int,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        opposite_normals_only: bool = True,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        binding = (
            source_scope.source_id,
            source_scope.source_revision,
            source_scope.entity_kind,
            source_scope.entity_dimension,
            source_scope.entity_set_id,
        )
        target_binding = (
            target_scope.source_id,
            target_scope.source_revision,
            target_scope.entity_kind,
            target_scope.entity_dimension,
            target_scope.entity_set_id,
        )
        if binding != target_binding:
            raise ValueError(
                f"ProximitySizeControl.scopes must share one entity binding; got source={binding!r}, target={target_binding!r}"
            )
        count = int(elements_per_gap)
        minimum = (
            None
            if minimum_size is None
            else positive_finite_float(minimum_size, "minimum_size")
        )
        maximum = (
            None
            if maximum_size is None
            else positive_finite_float(maximum_size, "maximum_size")
        )
        if count <= 0:
            raise ValueError(
                f"ProximitySizeControl.elements_per_gap must be positive; got {count!r}"
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"SizeControl.minimum_size must not exceed maximum_size; got minimum={minimum!r}, maximum={maximum!r}"
            )
        strength = parse(strength, SizeControlStrength, "SizeControl.strength")
        self.source_scope = source_scope
        self.target_scope = target_scope
        self.elements_per_gap = count
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.opposite_normals_only = bool(opposite_normals_only)
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "proximity-size-control",
                "source_scope": source_scope.scope_id,
                "target_scope": target_scope.scope_id,
                "elements_per_gap": count,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "opposite_normals_only": bool(opposite_normals_only),
                "strength": strength.value,
                "priority": int(priority),
            }
        )


class SourceProximityGapEvidence(StrictModule, NonTrainableState):
    """Source-bound lower gap enclosures, not arbitrary nearest sample distances."""

    source_scope: _dependency_phydrax_meshing__scope.MeshingScope
    target_scope: _dependency_phydrax_meshing__scope.MeshingScope
    sample_entity_ids: Array
    sample_points: Array
    lower_bounds: Array
    complete: Array
    control_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        control: ProximitySizeControl,
        sample_entity_ids: ArrayLike,
        sample_points: ArrayLike,
        lower_bounds: ArrayLike,
        complete: ArrayLike,
        /,
    ) -> None:
        identifiers = np.asarray(sample_entity_ids)
        points = np.asarray(sample_points, dtype=np.float64)
        lower = np.asarray(lower_bounds, dtype=np.float64)
        decided = np.asarray(complete)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise ValueError(
                f"SourceProximityGapEvidence.sample_entity_ids must be one integer vector; got shape={identifiers.shape}, dtype={identifiers.dtype}"
            )
        if (
            points.ndim != 2
            or points.shape[0] != identifiers.size
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                f"SourceProximityGapEvidence.sample_points must be one aligned finite physical-coordinate matrix; got shape={points.shape}, identifier_count={identifiers.size}"
            )
        if (
            lower.shape != identifiers.shape
            or np.any(np.isnan(lower))
            or np.any(lower < 0)
        ):
            raise ValueError(
                f"SourceProximityGapEvidence.lower_bounds must be aligned and nonnegative; got shape={lower.shape}, identifier_shape={identifiers.shape}"
            )
        if decided.dtype != np.dtype(np.bool_) or decided.shape != identifiers.shape:
            raise ValueError(
                f"SourceProximityGapEvidence.complete must be one aligned Boolean vector; got shape={decided.shape}, dtype={decided.dtype}"
            )
        self.source_scope, self.target_scope = control.source_scope, control.target_scope
        _host_sample_entity_ids = np.asarray(identifiers, dtype=np.int64)
        _host_sample_points = np.asarray(points, dtype=np.float64)
        _host_lower_bounds = np.asarray(lower, dtype=np.float64)
        _host_complete = np.asarray(decided, dtype=np.bool_)
        (
            self.sample_entity_ids,
            self.sample_points,
            self.lower_bounds,
            self.complete,
        ) = jax.device_put(
            (
                _host_sample_entity_ids,
                _host_sample_points,
                _host_lower_bounds,
                _host_complete,
            )
        )
        self.control_id = control.control_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "source-proximity-gap-evidence",
                "control": control.control_id,
                "source": control.source_scope.scope_id,
                "target": control.target_scope.scope_id,
                "samples": array_tree_fingerprint((identifiers, points, lower, decided)),
            }
        )

    def require_current(
        self,
        control: ProximitySizeControl,
        identifiers: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
        /,
    ) -> np.ndarray:
        if (
            self.control_id != control.control_id
            or self.source_scope.scope_id != control.source_scope.scope_id
            or self.target_scope.scope_id != control.target_scope.scope_id
            or not np.array_equal(np.asarray(self.sample_entity_ids), identifiers)
            or not np.array_equal(np.asarray(self.sample_points), points)
        ):
            raise ValueError(
                f"SourceProximityGapEvidence.bindings must retain the control and query workset; got stale control_id={self.control_id!r}"
            )
        if not np.all(np.asarray(self.complete)[mask]):
            raise ValueError(
                "SourceProximityGapEvidence.complete must certify every selected query; got unresolved lower-gap enclosure"
            )
        lower = np.asarray(self.lower_bounds, dtype=np.float64)
        if control.strength is SizeControlStrength.HARD and np.any(lower[mask] <= 0):
            raise ValueError(
                f"SourceProximityGapEvidence.lower_bounds must be strictly positive for hard proximity controls; got nonpositive bounds for control_id={control.control_id!r}"
            )
        return lower


SizeControl = UniformSizeControl | CurvatureSizeControl | ProximitySizeControl


class ResolvedSizeField(StrictModule, NonTrainableState):
    """Isotropic target sizes at explicitly identified sample entities."""

    domain: SizeFieldDomain = eqx.field(static=True)
    sample_points: Array
    sample_entity_ids: Array
    values: Array
    source_control_ids: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: SizeFieldDomain,
        sample_points: ArrayLike,
        values: ArrayLike,
        /,
        *,
        sample_entity_ids: ArrayLike,
        source_control_ids: tuple[str, ...],
    ) -> None:
        domain = parse(domain, SizeFieldDomain, "ResolvedSizeField.domain")
        points = np.asarray(sample_points, dtype=np.float64)
        sizes = np.asarray(values, dtype=np.float64)
        identifiers = np.asarray(sample_entity_ids)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ValueError(
                f"ResolvedSizeField.sample_points must be one non-empty finite matrix; got shape={points.shape}"
            )
        if (
            sizes.shape != (points.shape[0],)
            or np.any(~np.isfinite(sizes))
            or np.any(sizes <= 0)
        ):
            raise ValueError(
                f"ResolvedSizeField.values must be positive and match samples; got size_shape={sizes.shape}, point_shape={points.shape}"
            )
        if identifiers.shape != (points.shape[0],) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise ValueError(
                f"ResolvedSizeField.sample_entity_ids must be integer IDs aligned with samples; got shape={identifiers.shape}, dtype={identifiers.dtype}"
            )
        controls = tuple(str(value) for value in source_control_ids)
        if not controls or any(not value for value in controls):
            raise ValueError(
                f"ResolvedSizeField.source_control_ids must contain non-empty identities; got {controls!r}"
            )
        identifiers = identifiers.astype(np.int64, copy=False)
        self.domain = domain
        _host_sample_points = np.asarray(points)
        _host_sample_entity_ids = np.asarray(identifiers)
        _host_values = np.asarray(sizes)
        (
            self.sample_points,
            self.sample_entity_ids,
            self.values,
        ) = jax.device_put(
            (
                _host_sample_points,
                _host_sample_entity_ids,
                _host_values,
            )
        )
        self.source_control_ids = controls
        self.field_id = canonical_fingerprint(
            {
                "kind": "resolved-size-field",
                "domain": domain.value,
                "sample_points": array_tree_fingerprint(points),
                "sample_entity_ids": array_tree_fingerprint(identifiers),
                "values": array_tree_fingerprint(sizes),
                "source_controls": controls,
            }
        )


class SizeResolutionReport(StrictModule, NonTrainableState):
    """Retain size-resolution decisions and edge-length-aware growth evidence.

    ``graded_count`` counts samples lowered by hard growth limits and
    ``maximum_gradation_violation`` is the a-posteriori relative excess of any
    graded size over its growth bound (zero when every limited edge is satisfied).
    """

    control_ids: tuple[str, ...] = eqx.field(static=True)
    winning_control_ids: tuple[str, ...] = eqx.field(static=True)
    overlapping_scope_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    clamped: bool = eqx.field(static=True)
    provider_resolved: bool = eqx.field(static=True)
    graded_count: int = eqx.field(static=True)
    maximum_gradation_violation: float = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        controls: tuple[SizeControl, ...],
        field: ResolvedSizeField,
        /,
        *,
        winning_control_ids: tuple[str, ...],
        overlapping_scope_pairs: tuple[tuple[str, str], ...] = (),
        clamped: bool = False,
        provider_resolved: bool = False,
        graded_count: int = 0,
        maximum_gradation_violation: float = 0.0,
    ) -> None:
        if not controls or not all(
            isinstance(
                control,
                (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
            )
            for control in controls
        ):
            raise TypeError(
                f"SizeResolutionReport.controls must contain supported size controls; got {tuple(type(control).__name__ for control in controls)!r}"
            )
        identifiers = tuple(control.control_id for control in controls)
        winners = tuple(str(identifier) for identifier in winning_control_ids)
        if len(winners) != field.values.shape[0] or any(
            identifier not in identifiers for identifier in winners
        ):
            raise ValueError(
                f"SizeResolutionReport.winning_control_ids must identify one control per sample; got {winners!r}"
            )
        overlaps = tuple(
            (str(first), str(second)) for first, second in overlapping_scope_pairs
        )
        graded = int(graded_count)
        violation = float(maximum_gradation_violation)
        if graded < 0 or not np.isfinite(violation) or violation < 0.0:
            raise ValueError(
                f"SizeResolutionReport.gradation must be finite and non-negative; got graded_count={graded!r}, maximum_violation={violation!r}"
            )
        self.control_ids = identifiers
        self.winning_control_ids = winners
        self.overlapping_scope_pairs = overlaps
        self.clamped = bool(clamped)
        self.provider_resolved = bool(provider_resolved)
        self.graded_count = graded
        self.maximum_gradation_violation = violation
        self.field_id = field.field_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "size-resolution-report",
                "controls": identifiers,
                "winners": winners,
                "overlaps": overlaps,
                "clamped": bool(clamped),
                "provider_resolved": bool(provider_resolved),
                "graded_count": graded,
                "maximum_gradation_violation": violation,
                "field": field.field_id,
            }
        )


@checked
def control_scopes(
    control: SizeControl, /
) -> tuple[_dependency_phydrax_meshing__scope.MeshingScope, ...]:
    """Return the source scopes participating in a size control."""
    match control:
        case ProximitySizeControl():
            return control.source_scope, control.target_scope
        case UniformSizeControl() | CurvatureSizeControl():
            return (control.scope,)
        case _:
            raise TypeError(
                f"control_scopes.control must be a supported size control; got {type(control).__name__}"
            )


def _nearest_rows(
    points: np.ndarray, queries: np.ndarray, candidates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest candidate row and distance for each BVH query row."""
    candidate_points = points[candidates]
    hierarchy = prepare_bvh(candidate_points, candidate_points, dtype=jnp.float64)
    nearest = bvh_nearest_items(hierarchy, points[queries])
    items = np.asarray(nearest.items[:, 0])
    return candidates[items], np.sqrt(np.asarray(nearest.distance_squared[:, 0]))


def _proximity_gaps(
    control: ProximitySizeControl,
    points: np.ndarray,
    identifiers: np.ndarray,
    normals: np.ndarray | None,
    /,
) -> np.ndarray:
    """Gap to the nearest sample on the opposite scope's distinct entities.

    Source samples measure to target-only entities and vice versa, so a surface
    never measures a gap to itself. With ``opposite_normals_only`` a gap counts
    only when the nearest opposite sample faces the query (negative normal dot
    product); other samples receive an infinite gap.
    """
    source = np.isin(identifiers, np.asarray(control.source_scope.global_entity_ids))
    target = np.isin(identifiers, np.asarray(control.target_scope.global_entity_ids))
    source_only = np.flatnonzero(source & ~target)
    target_only = np.flatnonzero(target & ~source)
    if not source_only.size or not target_only.size:
        raise ValueError(
            f"resolve_size_controls.proximity_samples must cover disjoint source and target entities; got source_only={source_only.size}, target_only={target_only.size}"
        )
    if control.opposite_normals_only and normals is None:
        raise ValueError(
            "resolve_size_controls.normals must be supplied for opposite-normal proximity controls; got None"
        )
    gaps = np.full((points.shape[0],), np.inf)
    for queries, candidates in (
        (np.flatnonzero(source), target_only),
        (np.flatnonzero(target), source_only),
    ):
        nearest, distance = _nearest_rows(points, queries, candidates)
        if control.opposite_normals_only and normals is not None:
            facing = np.sum(normals[queries] * normals[nearest], axis=1) < 0.0
            distance = np.where(facing, distance, np.inf)
        gaps[queries] = np.minimum(gaps[queries], distance)
    return gaps


def _control_candidate(
    control: SizeControl,
    mask: np.ndarray,
    points: np.ndarray,
    identifiers: np.ndarray,
    curvature: np.ndarray | None,
    normals: np.ndarray | None,
    proximity: SourceProximityGapEvidence | None,
    /,
) -> np.ndarray:
    match control:
        case UniformSizeControl():
            source_bound = proximity is not None
            value = np.full((points.shape[0],), control.target_size)
        case CurvatureSizeControl():
            source_bound = True
            if curvature is None or curvature.shape != identifiers.shape:
                raise ValueError(
                    f"resolve_size_controls.curvature must supply aligned curvature samples; got {None if curvature is None else curvature.shape!r}"
                )
            if np.any(~np.isfinite(curvature[mask])) or np.any(curvature[mask] < 0.0):
                raise ValueError(
                    f"resolve_size_controls.curvature must be finite and non-negative; got invalid samples for control_id={control.control_id!r}"
                )
            value = np.divide(
                2.0 * np.sin(0.5 * control.normal_angle),
                curvature,
                out=np.full(curvature.shape, np.inf),
                where=curvature > 0.0,
            )
        case ProximitySizeControl():
            source_bound = proximity is not None
            gap = (
                _proximity_gaps(control, points, identifiers, normals)
                if proximity is None
                else proximity.require_current(control, identifiers, mask, points)
            )
            value = gap / control.elements_per_gap
        case _:
            raise TypeError(
                f"resolve_size_controls.controls must use a supported size-control kind; got {type(control).__name__}"
            )
    if control.minimum_size is not None:
        if (
            control.strength is SizeControlStrength.HARD
            and source_bound
            and np.any(value[mask] < control.minimum_size)
        ):
            raise ValueError(
                f"resolve_size_controls.minimum_size must not contradict an active hard curvature or gap bound; got minimum_size={control.minimum_size!r} for control_id={control.control_id!r}"
            )
        value = np.maximum(value, control.minimum_size)
    if control.maximum_size is not None:
        value = np.minimum(value, control.maximum_size)
    return value


def _hard_growth(
    controls: tuple[SizeControl, ...],
    masks: np.ndarray,
    edges: np.ndarray,
    /,
) -> np.ndarray:
    growth = np.full((edges.shape[0],), np.inf)
    for index, control in enumerate(controls):
        if (
            isinstance(control, UniformSizeControl)
            and control.strength is SizeControlStrength.HARD
            and control.maximum_growth_rate is not None
        ):
            within = masks[index, edges[:, 0]] & masks[index, edges[:, 1]]
            growth[within] = np.minimum(growth[within], control.maximum_growth_rate)
    return growth


def select_control(
    candidates: np.ndarray,
    active_mask: np.ndarray,
    hard: np.ndarray,
    priorities: np.ndarray,
    interval: tuple[float, float],
    combination: SizeCombinationPolicy,
    /,
) -> int:
    """Winning control of one sample under the explicit combination policy."""
    active = np.flatnonzero(active_mask)
    if not active.size:
        raise ValueError(
            "resolve_size_controls.controls must cover every sample; got an uncovered sample"
        )
    active_hard = active[hard[active]]
    pool = active_hard if active_hard.size else active
    match combination:
        case SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
            choices = np.clip(candidates[pool], *interval)
            if active_hard.size and np.any(choices != choices[0]):
                raise ValueError(
                    f"resolve_size_controls.controls must agree on overlapping hard size-control targets; got {choices!r}"
                )
            return int(pool[int(np.argmin(choices))])
        case SizeCombinationPolicy.EXPLICIT_PRIORITY:
            finalists = pool[priorities[pool] == np.max(priorities[pool])]
            choices = np.clip(candidates[finalists], *interval)
            if np.any(choices != choices[0]):
                raise ValueError(
                    f"resolve_size_controls.controls must agree on equal-priority size-control targets for one sample; got {choices!r}"
                )
            return int(finalists[0])
        case _:
            raise ValueError(
                f"resolve_size_controls.combination must be a supported size-combination policy; got {combination!r}"
            )


def _apply_hard_growth(
    controls: tuple[SizeControl, ...],
    masks: np.ndarray,
    adjacency: ArrayLike | None,
    points: np.ndarray,
    raw: np.ndarray,
    admissible_minimum: np.ndarray,
    /,
) -> tuple[np.ndarray, int, float]:
    """Apply edge-length-aware growth limits through the metric owner."""
    if adjacency is None:
        return raw, 0, 0.0
    edges = np.asarray(adjacency)
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError(
            f"resolve_size_controls.adjacency must contain integer sample rows; got dtype={edges.dtype}"
        )
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= points.shape[0])
    ):
        raise ValueError(
            f"resolve_size_controls.adjacency must have shape (edges, 2) with valid sample rows; got shape={edges.shape}"
        )
    growth = _hard_growth(controls, masks, edges)
    limited = np.isfinite(growth)
    limited_edges = edges[limited].astype(np.int64, copy=False)
    lengths = np.linalg.norm(
        points[limited_edges[:, 1]] - points[limited_edges[:, 0]], axis=1
    )
    resolved, _, _ = _dependency_phydrax_meshing__metric.grade_scalar_sizes(
        raw,
        limited_edges,
        lengths,
        growth[limited],
        _dependency_phydrax_meshing__metric.MetricGradationKind.PHYSICAL,
    )
    if np.any(resolved < admissible_minimum):
        raise ValueError(
            "resolve_size_controls.adjacency must admit the hard size intervals under its growth limits; got resolved sizes below admissible minima"
        )
    violation = _dependency_phydrax_meshing__metric.scalar_violation(
        resolved,
        limited_edges,
        lengths,
        growth[limited],
        _dependency_phydrax_meshing__metric.MetricGradationKind.PHYSICAL,
    )
    return resolved, int(np.count_nonzero(resolved < raw)), violation


@final
class _SizeCandidates(StrictModule, NonTrainableState):
    candidates: np.ndarray
    masks: np.ndarray
    hard: np.ndarray
    priorities: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    overlaps: tuple[tuple[str, str], ...]


def _prepare_size_candidates(
    controls_: tuple[SizeControl, ...],
    points: np.ndarray,
    identifiers: np.ndarray,
    curvature_values: np.ndarray | None,
    normal_values: np.ndarray | None,
    supplied_gaps: dict[str, SourceProximityGapEvidence],
) -> _SizeCandidates:
    candidates = []
    masks = []
    lower_bounds = []
    upper_bounds = []
    overlaps = []
    for left, control in enumerate(controls_):
        scopes = control_scopes(control)
        entity_ids = np.unique(
            np.concatenate(
                tuple(
                    np.asarray(scope.global_entity_ids, dtype=np.int64)
                    for scope in scopes
                )
            )
        )
        mask = np.isin(identifiers, entity_ids)
        if not np.any(mask):
            raise ValueError(
                f"resolve_size_controls.controls must select supplied sample entities; got control_id={control.control_id!r} with no selected samples"
            )
        for second in controls_[left + 1 :]:
            if any(
                first_scope.entity_set_id == second_scope.entity_set_id
                and np.intersect1d(
                    first_scope.global_entity_ids, second_scope.global_entity_ids
                ).size
                for first_scope in scopes
                for second_scope in control_scopes(second)
            ):
                overlaps.append((control.control_id, second.control_id))
        candidates.append(
            _control_candidate(
                control,
                mask,
                points,
                identifiers,
                curvature_values,
                normal_values,
                supplied_gaps.get(control.control_id),
            )
        )
        source_gap = supplied_gaps.get(control.control_id)
        if (
            isinstance(control, ProximitySizeControl)
            and source_gap is not None
            and control.maximum_size is None
        ):
            # Infinite certified gap means no facing source feature: there is
            # no active geometric bound, rather than an infinite hard target.
            mask &= np.isfinite(np.asarray(source_gap.lower_bounds))
        masks.append(mask)
        lower_bounds.append(
            0.0
            if control.strength is SizeControlStrength.SOFT
            or control.minimum_size is None
            else control.minimum_size
        )
        upper_bounds.append(
            np.inf
            if control.strength is SizeControlStrength.SOFT
            or control.maximum_size is None
            else control.maximum_size
        )
    candidate_array = np.stack(candidates)
    mask_array = np.stack(masks)
    hard_array = np.asarray(
        tuple(control.strength is SizeControlStrength.HARD for control in controls_)
    )
    priority_array = np.asarray(tuple(control.priority for control in controls_))
    lower_array = np.asarray(lower_bounds, dtype=np.float64)[:, None]
    upper_array = np.asarray(upper_bounds, dtype=np.float64)[:, None]
    hard_mask = mask_array & hard_array[:, None]
    admissible_minimum = np.max(np.where(hard_mask, lower_array, 0.0), axis=0)
    admissible_maximum = np.min(np.where(hard_mask, upper_array, np.inf), axis=0)
    if np.any(admissible_minimum > admissible_maximum):
        raise ValueError(
            "resolve_size_controls.controls must have intersecting active hard size intervals; got an empty intersection"
        )
    return _SizeCandidates(
        candidate_array,
        mask_array,
        hard_array,
        priority_array,
        admissible_minimum,
        admissible_maximum,
        tuple(overlaps),
    )


def resolve_size_controls(
    controls: tuple[SizeControl, ...],
    sample_points: ArrayLike,
    sample_entity_ids: ArrayLike,
    domain: SizeFieldDomain,
    /,
    *,
    curvature: ArrayLike | None = None,
    normals: ArrayLike | None = None,
    adjacency: ArrayLike | None = None,
    proximity_gaps: Mapping[str, SourceProximityGapEvidence] | None = None,
    combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
) -> tuple[ResolvedSizeField, SizeResolutionReport]:
    """Resolve targets only after intersecting every active hard size interval.

    Without ``proximity_gaps``, gaps are BVH nearest distances between supplied
    samples; they do not certify a continuous source gap. Supplied lower-gap
    evidence binds the declared control, scopes, revisions and sample workset.
    Unresolved evidence is refused.
    Hard growth limits on ``adjacency`` enforce
    ``h_i <= h_j + (rate - 1) |x_i - x_j|`` through the metric owner.
    """

    controls_ = tuple(controls)
    if not controls_:
        raise ValueError(
            "resolve_size_controls.controls must contain at least one size control; got an empty sequence"
        )
    if not all(
        isinstance(
            control,
            (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
        )
        for control in controls_
    ):
        raise TypeError(
            f"resolve_size_controls.controls must contain supported size controls; got {tuple(type(control).__name__ for control in controls_)!r}"
        )
    supplied_gaps = {} if proximity_gaps is None else dict(proximity_gaps)
    known_gaps = {
        control.control_id
        for control in controls_
        if isinstance(control, ProximitySizeControl)
    }
    if set(supplied_gaps) - known_gaps or not all(
        isinstance(value, SourceProximityGapEvidence) for value in supplied_gaps.values()
    ):
        raise ValueError(
            f"resolve_size_controls.proximity_gaps must bind known proximity controls; got evidence_keys={tuple(supplied_gaps)!r}"
        )
    domain = parse(domain, SizeFieldDomain, "ResolvedSizeField.domain")
    combination = parse(
        combination, SizeCombinationPolicy, "resolve_size_controls.combination"
    )
    points = np.asarray(sample_points, dtype=np.float64)
    identifiers = np.asarray(sample_entity_ids, dtype=np.int64)
    if (
        points.ndim != 2
        or not np.all(np.isfinite(points))
        or identifiers.shape != (points.shape[0],)
    ):
        raise ValueError(
            f"resolve_size_controls.samples must have finite points and aligned entity IDs; got point_shape={points.shape}, identifier_shape={identifiers.shape}"
        )
    curvature_values = (
        None if curvature is None else np.asarray(curvature, dtype=np.float64)
    )
    normal_values = None if normals is None else np.asarray(normals, dtype=np.float64)
    if normal_values is not None and (
        normal_values.shape != points.shape or not np.all(np.isfinite(normal_values))
    ):
        raise ValueError(
            f"resolve_size_controls.normals must be finite and aligned with samples; got normal_shape={normal_values.shape}, point_shape={points.shape}"
        )
    prepared = _prepare_size_candidates(
        controls_, points, identifiers, curvature_values, normal_values, supplied_gaps
    )
    candidate_array, mask_array = prepared.candidates, prepared.masks
    hard_array, priority_array = prepared.hard, prepared.priorities
    admissible_minimum, admissible_maximum = prepared.minimum, prepared.maximum
    overlaps = prepared.overlaps

    preferred = np.empty((points.shape[0],), dtype=np.float64)
    raw = np.empty((points.shape[0],), dtype=np.float64)
    winners = []
    for sample in range(points.shape[0]):
        interval = (admissible_minimum[sample], admissible_maximum[sample])
        selected = select_control(
            candidate_array[:, sample],
            mask_array[:, sample],
            hard_array,
            priority_array,
            interval,
            combination,
        )
        preferred[sample] = candidate_array[selected, sample]
        raw[sample] = np.clip(preferred[sample], *interval)
        winners.append(controls_[selected].control_id)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(
            f"resolve_size_controls.targets must be positive and finite; got {raw.size} targets including an invalid value"
        )

    resolved, graded_count, violation = _apply_hard_growth(
        controls_, mask_array, adjacency, points, raw, admissible_minimum
    )
    field = ResolvedSizeField(
        domain,
        points,
        resolved,
        sample_entity_ids=identifiers,
        source_control_ids=tuple(control.control_id for control in controls_),
    )
    return field, SizeResolutionReport(
        controls_,
        field,
        winning_control_ids=tuple(winners),
        overlapping_scope_pairs=tuple(overlaps),
        clamped=not np.array_equal(resolved, preferred),
        graded_count=graded_count,
        maximum_gradation_violation=violation,
    )


@checked
def size_field_metric(
    field: ResolvedSizeField, scope: _dependency_phydrax_meshing__scope.MeshingScope, /
) -> _dependency_phydrax_meshing__metric.MeshMetricField:
    """Compile a resolved size field into isotropic metric constraints.

    Row ``i`` becomes ``h_i**-2 I`` for ``scope.global_entity_ids[i]``, which
    must equal the field's sample entity IDs in order. Declared size bounds are
    the resolved extremes, so the metric can be combined with solution-adaptive
    metrics through :func:`combine_mesh_metrics`. The field's hard growth limits
    are already resolved into the sizes; requested execution gradation belongs to
    a `MetricGradationPolicy` or the provider options.
    """
    if not np.array_equal(
        np.asarray(scope.global_entity_ids, dtype=np.int64),
        np.asarray(field.sample_entity_ids),
    ):
        raise ValueError(
            f"size_field_metric.scope must have entity IDs equal to the size-field sample entity IDs; got scope_id={scope.scope_id!r}, field_id={field.field_id!r}"
        )
    sizes = np.asarray(field.values, dtype=np.float64)
    dimension = field.sample_points.shape[1]
    values = np.eye(dimension)[None, :, :] / sizes[:, None, None] ** 2
    return _dependency_phydrax_meshing__metric.MeshMetricField(
        scope,
        values,
        minimum_size=float(np.min(sizes)),
        maximum_size=float(np.max(sizes)),
        maximum_anisotropy=1.0,
    )


__all__ = [
    "CurvatureSizeControl",
    "ProximitySizeControl",
    "ResolvedSizeField",
    "SizeCombinationPolicy",
    "SizeCompliancePolicy",
    "SizeControl",
    "SizeControlStrength",
    "SizeFieldDomain",
    "SizeResolutionReport",
    "SizeStatistic",
    "SourceProximityGapEvidence",
    "UniformSizeControl",
    "resolve_size_controls",
    "size_field_metric",
]
