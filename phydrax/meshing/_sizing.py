#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._bvh import bvh_nearest_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._metric import (
    _grade_scalar_sizes,
    _scalar_violation,
    MeshMetricField,
    MetricGradationKind,
)
from ._scope import MeshingScope


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


class SizeCompliancePolicy(StrictModule, NonTrainableState):
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    target_statistics: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        target_statistics: tuple[str, ...] = ("p50", "p95"),
    ) -> None:
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        statistics = tuple(str(value).strip() for value in target_statistics)
        supported = {"p50", "p95"}
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError(
                "Size compliance tolerances must be finite and non-negative."
            )
        if (
            not statistics
            or len(set(statistics)) != len(statistics)
            or not set(statistics) <= supported
        ):
            raise ValueError(
                "target_statistics must contain distinct supported statistics p50/p95."
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


def _size(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


class UniformSizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
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
        scope: MeshingScope,
        target_size: float,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        maximum_growth_rate: float | None = None,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        target = _size(target_size, "target_size")
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        growth = None if maximum_growth_rate is None else float(maximum_growth_rate)
        if minimum is not None and minimum > target:
            raise ValueError("Sizes must satisfy minimum <= target when supplied.")
        if maximum is not None and target > maximum:
            raise ValueError("Sizes must satisfy target <= maximum when supplied.")
        if growth is not None and (not np.isfinite(growth) or growth < 1.0):
            raise ValueError("maximum_growth_rate must be finite and at least one.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
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
    scope: MeshingScope
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
        scope: MeshingScope,
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
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        if not np.isfinite(angle) or angle <= 0.0 or angle >= np.pi:
            raise ValueError("normal_angle must lie strictly between zero and pi.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
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
    source_scope: MeshingScope
    target_scope: MeshingScope
    elements_per_gap: int = eqx.field(static=True)
    minimum_size: float | None = eqx.field(static=True)
    maximum_size: float | None = eqx.field(static=True)
    opposite_normals_only: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scope: MeshingScope,
        target_scope: MeshingScope,
        elements_per_gap: int,
        /,
        *,
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        opposite_normals_only: bool = True,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ) -> None:
        if not isinstance(source_scope, MeshingScope) or not isinstance(
            target_scope, MeshingScope
        ):
            raise TypeError("source_scope and target_scope must be MeshingScope.")
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
            raise ValueError("Proximity scopes must share one exact entity binding.")
        count = int(elements_per_gap)
        minimum = None if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = None if maximum_size is None else _size(maximum_size, "maximum_size")
        if count <= 0:
            raise ValueError("elements_per_gap must be positive.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
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
        if not isinstance(domain, SizeFieldDomain):
            raise TypeError("domain must be SizeFieldDomain.")
        points = np.asarray(sample_points, dtype=np.float64)
        sizes = np.asarray(values, dtype=np.float64)
        identifiers = np.asarray(sample_entity_ids)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ValueError("sample_points must be one non-empty finite matrix.")
        if (
            sizes.shape != (points.shape[0],)
            or np.any(~np.isfinite(sizes))
            or np.any(sizes <= 0)
        ):
            raise ValueError("Resolved size values must be positive and match samples.")
        if identifiers.shape != (points.shape[0],) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise ValueError(
                "sample_entity_ids must be integer IDs aligned with samples."
            )
        controls = tuple(str(value) for value in source_control_ids)
        if not controls or any(not value for value in controls):
            raise ValueError("Resolved size fields require source control identities.")
        identifiers = identifiers.astype(np.int64, copy=False)
        self.domain = domain
        self.sample_points = jnp.asarray(points)
        self.sample_entity_ids = jnp.asarray(identifiers)
        self.values = jnp.asarray(sizes)
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
    """Resolution decisions, including exact edge-length-aware growth evidence.

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
            raise TypeError("controls must contain supported size controls.")
        identifiers = tuple(control.control_id for control in controls)
        winners = tuple(str(identifier) for identifier in winning_control_ids)
        if len(winners) != field.values.shape[0] or any(
            identifier not in identifiers for identifier in winners
        ):
            raise ValueError("winning_control_ids must identify one control per sample.")
        overlaps = tuple(
            (str(first), str(second)) for first, second in overlapping_scope_pairs
        )
        graded = int(graded_count)
        violation = float(maximum_gradation_violation)
        if graded < 0 or not np.isfinite(violation) or violation < 0.0:
            raise ValueError("Gradation evidence must be finite and non-negative.")
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


def _control_scopes(control: SizeControl, /) -> tuple[MeshingScope, ...]:
    if isinstance(control, ProximitySizeControl):
        return control.source_scope, control.target_scope
    return (control.scope,)


def _nearest_rows(
    points: np.ndarray, queries: np.ndarray, candidates: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest candidate row and distance for each query row (BVH)."""
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
    source = np.isin(identifiers, np.asarray(control.source_scope.entity_ids))
    target = np.isin(identifiers, np.asarray(control.target_scope.entity_ids))
    source_only = np.flatnonzero(source & ~target)
    target_only = np.flatnonzero(target & ~source)
    if not source_only.size or not target_only.size:
        raise ValueError(
            "Proximity controls require samples on disjoint source and target entities."
        )
    if control.opposite_normals_only and normals is None:
        raise ValueError("Opposite-normal proximity controls require sample normals.")
    gaps = np.full((points.shape[0],), np.inf)
    for queries, candidates in (
        (np.flatnonzero(source), target_only),
        (np.flatnonzero(target), source_only),
    ):
        nearest, distance = _nearest_rows(points, queries, candidates)
        if control.opposite_normals_only:
            assert normals is not None
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
    /,
) -> np.ndarray:
    if isinstance(control, UniformSizeControl):
        value = np.full((points.shape[0],), control.target_size)
    elif isinstance(control, CurvatureSizeControl):
        if curvature is None or curvature.shape != identifiers.shape:
            raise ValueError("Curvature controls require aligned curvature samples.")
        if np.any(~np.isfinite(curvature[mask])) or np.any(curvature[mask] < 0.0):
            raise ValueError("Curvature samples must be finite and non-negative.")
        value = np.divide(
            2.0 * np.sin(0.5 * control.normal_angle),
            curvature,
            out=np.full(curvature.shape, np.inf),
            where=curvature > 0.0,
        )
    elif isinstance(control, ProximitySizeControl):
        value = (
            _proximity_gaps(control, points, identifiers, normals)
            / control.elements_per_gap
        )
    else:
        raise TypeError("Unsupported size control.")
    if control.minimum_size is not None:
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


def _select_control(
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
        raise ValueError("Size controls do not cover every sample.")
    active_hard = active[hard[active]]
    pool = active_hard if active_hard.size else active
    match combination:
        case SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
            choices = np.clip(candidates[pool], *interval)
            if active_hard.size and np.any(choices != choices[0]):
                raise ValueError("Overlapping hard size-control targets conflict.")
            return int(pool[int(np.argmin(choices))])
        case SizeCombinationPolicy.EXPLICIT_PRIORITY:
            finalists = pool[priorities[pool] == np.max(priorities[pool])]
            choices = np.clip(candidates[finalists], *interval)
            if np.any(choices != choices[0]):
                raise ValueError(
                    "Equal-priority size-control targets conflict on one sample."
                )
            return int(finalists[0])
        case _:
            raise ValueError(f"Unsupported size combination policy {combination!r}.")


def _apply_hard_growth(
    controls: tuple[SizeControl, ...],
    masks: np.ndarray,
    adjacency: ArrayLike | None,
    points: np.ndarray,
    raw: np.ndarray,
    admissible_minimum: np.ndarray,
    /,
) -> tuple[np.ndarray, int, float]:
    """Exact edge-length-aware growth limits through the metric owner."""
    if adjacency is None:
        return raw, 0, 0.0
    edges = np.asarray(adjacency)
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("Size-field adjacency must contain integer sample rows.")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or np.any(edges < 0)
        or np.any(edges >= points.shape[0])
    ):
        raise ValueError("Size-field adjacency must have shape (edges, 2).")
    growth = _hard_growth(controls, masks, edges)
    limited = np.isfinite(growth)
    limited_edges = edges[limited].astype(np.int64, copy=False)
    lengths = np.linalg.norm(
        points[limited_edges[:, 1]] - points[limited_edges[:, 0]], axis=1
    )
    resolved, _, _ = _grade_scalar_sizes(
        raw, limited_edges, lengths, growth[limited], MetricGradationKind.PHYSICAL
    )
    if np.any(resolved < admissible_minimum):
        raise ValueError("Hard growth limits are incompatible with hard size intervals.")
    violation = _scalar_violation(
        resolved, limited_edges, lengths, growth[limited], MetricGradationKind.PHYSICAL
    )
    return resolved, int(np.count_nonzero(resolved < raw)), violation


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
    combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
) -> tuple[ResolvedSizeField, SizeResolutionReport]:
    """Resolve targets only after intersecting every active hard size interval.

    Proximity gaps are measured by exact BVH nearest queries between samples of
    the source and target scopes. Hard growth limits on ``adjacency`` edges
    enforce ``h_i <= h_j + (rate - 1) |x_i - x_j|`` exactly through the metric
    owner's minimum-first relaxation.
    """

    controls_ = tuple(controls)
    if not controls_:
        raise ValueError("At least one size control is required.")
    if not all(
        isinstance(
            control,
            (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
        )
        for control in controls_
    ):
        raise TypeError("controls must contain supported size controls.")
    if not isinstance(domain, SizeFieldDomain):
        raise TypeError("domain must be SizeFieldDomain.")
    if not isinstance(combination, SizeCombinationPolicy):
        raise TypeError("combination must be SizeCombinationPolicy.")
    points = np.asarray(sample_points, dtype=np.float64)
    identifiers = np.asarray(sample_entity_ids, dtype=np.int64)
    if (
        points.ndim != 2
        or not np.all(np.isfinite(points))
        or identifiers.shape != (points.shape[0],)
    ):
        raise ValueError("Size samples and entity IDs must be finite and aligned.")
    curvature_values = (
        None if curvature is None else np.asarray(curvature, dtype=np.float64)
    )
    normal_values = None if normals is None else np.asarray(normals, dtype=np.float64)
    if normal_values is not None and (
        normal_values.shape != points.shape or not np.all(np.isfinite(normal_values))
    ):
        raise ValueError("Sample normals must be finite and aligned with samples.")
    candidates = []
    masks = []
    lower_bounds = []
    upper_bounds = []
    overlaps = []
    for left, control in enumerate(controls_):
        scopes = _control_scopes(control)
        entity_ids = np.unique(
            np.concatenate(
                tuple(np.asarray(scope.entity_ids, dtype=np.int64) for scope in scopes)
            )
        )
        mask = np.isin(identifiers, entity_ids)
        if not np.any(mask):
            raise ValueError("A size control resolves to no supplied sample entities.")
        for second in controls_[left + 1 :]:
            if any(
                first_scope.entity_set_id == second_scope.entity_set_id
                and np.intersect1d(first_scope.entity_ids, second_scope.entity_ids).size
                for first_scope in scopes
                for second_scope in _control_scopes(second)
            ):
                overlaps.append((control.control_id, second.control_id))
        candidates.append(
            _control_candidate(
                control, mask, points, identifiers, curvature_values, normal_values
            )
        )
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
        raise ValueError("Active hard size intervals have an empty intersection.")

    preferred = np.empty((points.shape[0],), dtype=np.float64)
    raw = np.empty((points.shape[0],), dtype=np.float64)
    winners = []
    for sample in range(points.shape[0]):
        interval = (admissible_minimum[sample], admissible_maximum[sample])
        selected = _select_control(
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
        raise ValueError("Resolved size targets must be positive and finite.")

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


def size_field_metric(
    field: ResolvedSizeField, scope: MeshingScope, /
) -> MeshMetricField:
    """Compile a resolved size field into isotropic metric constraints.

    Row ``i`` becomes ``h_i**-2 I`` for the entity ``scope.entity_ids[i]``, which
    must equal the field's sample entity IDs in order. Declared size bounds are
    the resolved extremes, so the metric can be combined with solution-adaptive
    metrics through :func:`combine_mesh_metrics`. The field's hard growth limits
    are already resolved into the sizes; requested execution gradation belongs to
    a `MetricGradationPolicy` or the provider options.
    """
    if not isinstance(field, ResolvedSizeField):
        raise TypeError("field must be ResolvedSizeField.")
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    if not np.array_equal(
        np.asarray(scope.entity_ids, dtype=np.int64),
        np.asarray(field.sample_entity_ids),
    ):
        raise ValueError("Scope entity IDs must equal the size-field sample entity IDs.")
    sizes = np.asarray(field.values, dtype=np.float64)
    dimension = field.sample_points.shape[1]
    values = np.eye(dimension)[None, :, :] / sizes[:, None, None] ** 2
    return MeshMetricField(
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
    "UniformSizeControl",
    "resolve_size_controls",
    "size_field_metric",
]
