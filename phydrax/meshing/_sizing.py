#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
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
    ):
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
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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
    ):
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
    domain: SizeFieldDomain = eqx.field(static=True)
    sample_points: Array
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
        source_control_ids: tuple[str, ...],
    ):
        if not isinstance(domain, SizeFieldDomain):
            raise TypeError("domain must be SizeFieldDomain.")
        points = np.asarray(sample_points, dtype=float)
        sizes = np.asarray(values, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise ValueError("sample_points must be one non-empty finite matrix.")
        if (
            sizes.shape != (points.shape[0],)
            or np.any(~np.isfinite(sizes))
            or np.any(sizes <= 0)
        ):
            raise ValueError("Resolved size values must be positive and match samples.")
        controls = tuple(str(value) for value in source_control_ids)
        if not controls or any(not value for value in controls):
            raise ValueError("Resolved size fields require source control identities.")
        self.domain = domain
        self.sample_points = jnp.asarray(points)
        self.values = jnp.asarray(sizes)
        self.source_control_ids = controls
        self.field_id = canonical_fingerprint(
            {
                "kind": "resolved-size-field",
                "domain": domain.value,
                "sample_points": array_tree_fingerprint(points),
                "values": array_tree_fingerprint(sizes),
                "source_controls": controls,
            }
        )


class SizeResolutionReport(StrictModule, NonTrainableState):
    control_ids: tuple[str, ...] = eqx.field(static=True)
    winning_control_ids: tuple[str, ...] = eqx.field(static=True)
    overlapping_scope_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    clamped: bool = eqx.field(static=True)
    provider_resolved: bool = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

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
    ):
        if not controls or not all(
            isinstance(
                control,
                (UniformSizeControl, CurvatureSizeControl, ProximitySizeControl),
            )
            for control in controls
        ):
            raise TypeError("controls must contain supported size controls.")
        if not isinstance(field, ResolvedSizeField):
            raise TypeError("field must be ResolvedSizeField.")
        identifiers = tuple(control.control_id for control in controls)
        winners = tuple(str(identifier) for identifier in winning_control_ids)
        if len(winners) != field.values.shape[0] or any(
            identifier not in identifiers for identifier in winners
        ):
            raise ValueError("winning_control_ids must identify one control per sample.")
        overlaps = tuple(
            (str(first), str(second)) for first, second in overlapping_scope_pairs
        )
        self.control_ids = identifiers
        self.winning_control_ids = winners
        self.overlapping_scope_pairs = overlaps
        self.clamped = bool(clamped)
        self.provider_resolved = bool(provider_resolved)
        self.field_id = field.field_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "size-resolution-report",
                "controls": identifiers,
                "winners": winners,
                "overlaps": overlaps,
                "clamped": bool(clamped),
                "provider_resolved": bool(provider_resolved),
                "field": field.field_id,
            }
        )


class MeshMetricField(StrictModule, NonTrainableState):
    """Vertex-associated SPD Riemannian metric with explicit bounds."""

    scope: MeshingScope
    values: Array
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    maximum_gradation: float = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_anisotropy: float = 100.0,
        maximum_gradation: float = 1.3,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        metric = np.asarray(values, dtype=float)
        minimum = _size(minimum_size, "minimum_size")
        maximum = _size(maximum_size, "maximum_size")
        anisotropy = float(maximum_anisotropy)
        gradation = float(maximum_gradation)
        if minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if (
            metric.ndim != 3
            or metric.shape[0] != scope.entity_ids.shape[0]
            or metric.shape[1] != metric.shape[2]
            or not np.all(np.isfinite(metric))
        ):
            raise ValueError("Metric values must be finite aligned square matrices.")
        if not np.allclose(metric, np.swapaxes(metric, -1, -2), atol=1.0e-12, rtol=0.0):
            raise ValueError("Mesh metrics must be symmetric.")
        if np.any(np.linalg.eigvalsh(metric) <= 0.0):
            raise ValueError("Mesh metrics must be positive definite.")
        if not np.isfinite(anisotropy) or anisotropy < 1.0:
            raise ValueError("maximum_anisotropy must be finite and at least one.")
        if not np.isfinite(gradation) or gradation < 1.0:
            raise ValueError("maximum_gradation must be finite and at least one.")
        self.scope = scope
        self.values = jnp.asarray(metric)
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.maximum_anisotropy = anisotropy
        self.maximum_gradation = gradation
        self.metric_id = canonical_fingerprint(
            {
                "kind": "mesh-metric-field",
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(metric),
                "minimum_size": minimum,
                "maximum_size": maximum,
                "maximum_anisotropy": anisotropy,
                "maximum_gradation": gradation,
            }
        )


def resolve_size_controls(
    controls: tuple[SizeControl, ...],
    sample_points: ArrayLike,
    sample_entity_ids: ArrayLike,
    domain: SizeFieldDomain,
    /,
    *,
    curvature: ArrayLike | None = None,
    proximity: ArrayLike | None = None,
    adjacency: ArrayLike | None = None,
    combination: SizeCombinationPolicy = SizeCombinationPolicy.REJECT_HARD_CONFLICTS,
) -> tuple[ResolvedSizeField, SizeResolutionReport]:
    """Resolve targets only after intersecting every active hard size interval."""

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
    points = np.asarray(sample_points, dtype=float)
    identifiers = np.asarray(sample_entity_ids, dtype=np.int64)
    if (
        points.ndim != 2
        or not np.all(np.isfinite(points))
        or identifiers.shape != (points.shape[0],)
    ):
        raise ValueError("Size samples and entity IDs must be finite and aligned.")
    curvature_values = None if curvature is None else np.asarray(curvature, dtype=float)
    proximity_values = None if proximity is None else np.asarray(proximity, dtype=float)
    candidates = []
    masks = []
    lower_bounds = []
    upper_bounds = []
    overlaps = []
    for left, control in enumerate(controls_):
        scopes = (
            (control.source_scope, control.target_scope)
            if isinstance(control, ProximitySizeControl)
            else (control.scope,)
        )
        entity_ids = np.unique(
            np.concatenate(
                tuple(np.asarray(scope.entity_ids, dtype=np.int64) for scope in scopes)
            )
        )
        mask = np.isin(identifiers, entity_ids)
        if not np.any(mask):
            raise ValueError("A size control resolves to no supplied sample entities.")
        for second in controls_[left + 1 :]:
            second_scopes = (
                (second.source_scope, second.target_scope)
                if isinstance(second, ProximitySizeControl)
                else (second.scope,)
            )
            if any(
                first_scope.entity_set_id == second_scope.entity_set_id
                and np.intersect1d(first_scope.entity_ids, second_scope.entity_ids).size
                for first_scope in scopes
                for second_scope in second_scopes
            ):
                overlaps.append((control.control_id, second.control_id))
        if isinstance(control, UniformSizeControl):
            value = np.full((points.shape[0],), control.target_size)
        elif isinstance(control, CurvatureSizeControl):
            if curvature_values is None or curvature_values.shape != identifiers.shape:
                raise ValueError("Curvature controls require aligned curvature samples.")
            if np.any(~np.isfinite(curvature_values[mask])) or np.any(
                curvature_values[mask] < 0.0
            ):
                raise ValueError("Curvature samples must be finite and non-negative.")
            value = np.divide(
                2.0 * np.sin(0.5 * control.normal_angle),
                curvature_values,
                out=np.full(curvature_values.shape, np.inf),
                where=curvature_values > 0.0,
            )
        elif isinstance(control, ProximitySizeControl):
            if proximity_values is None or proximity_values.shape != identifiers.shape:
                raise ValueError("Proximity controls require aligned gap samples.")
            if np.any(~np.isfinite(proximity_values[mask])) or np.any(
                proximity_values[mask] < 0.0
            ):
                raise ValueError("Gap samples must be finite and non-negative.")
            value = proximity_values / control.elements_per_gap
        else:
            raise TypeError("Unsupported size control.")
        if control.minimum_size is not None:
            value = np.maximum(value, control.minimum_size)
        if control.maximum_size is not None:
            value = np.minimum(value, control.maximum_size)
        candidates.append(value)
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
    lower_array = np.asarray(lower_bounds, dtype=float)[:, None]
    upper_array = np.asarray(upper_bounds, dtype=float)[:, None]
    hard_mask = mask_array & hard_array[:, None]
    admissible_minimum = np.max(np.where(hard_mask, lower_array, 0.0), axis=0)
    admissible_maximum = np.min(np.where(hard_mask, upper_array, np.inf), axis=0)
    if np.any(admissible_minimum > admissible_maximum):
        raise ValueError("Active hard size intervals have an empty intersection.")

    preferred = np.empty((points.shape[0],), dtype=float)
    raw = np.empty((points.shape[0],), dtype=float)
    winners = []
    for sample in range(points.shape[0]):
        active = np.flatnonzero(mask_array[:, sample])
        if not active.size:
            raise ValueError("Size controls do not cover every sample.")
        active_hard = active[hard_array[active]]
        pool = active_hard if active_hard.size else active
        choices = np.clip(
            candidate_array[pool, sample],
            admissible_minimum[sample],
            admissible_maximum[sample],
        )
        if combination is SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
            if active_hard.size and np.any(choices != choices[0]):
                raise ValueError("Overlapping hard size-control targets conflict.")
            selected = int(pool[int(np.argmin(choices))])
        else:
            highest = np.max(priority_array[pool])
            finalists = pool[priority_array[pool] == highest]
            finalist_choices = np.clip(
                candidate_array[finalists, sample],
                admissible_minimum[sample],
                admissible_maximum[sample],
            )
            if np.any(finalist_choices != finalist_choices[0]):
                raise ValueError(
                    "Equal-priority size-control targets conflict on one sample."
                )
            selected = int(finalists[0])
        preferred[sample] = candidate_array[selected, sample]
        raw[sample] = np.clip(
            preferred[sample],
            admissible_minimum[sample],
            admissible_maximum[sample],
        )
        winners.append(controls_[selected].control_id)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("Resolved size targets must be positive and finite.")

    resolved = raw.copy()
    if adjacency is not None:
        edges = np.asarray(adjacency, dtype=np.int32)
        if (
            edges.ndim != 2
            or edges.shape[1] != 2
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError("Size-field adjacency must have shape (edges, 2).")
        edge_growth = np.full((edges.shape[0],), np.inf)
        for index, control in enumerate(controls_):
            if (
                isinstance(control, UniformSizeControl)
                and control.strength is SizeControlStrength.HARD
                and control.maximum_growth_rate is not None
            ):
                within = mask_array[index, edges[:, 0]] & mask_array[index, edges[:, 1]]
                edge_growth[within] = np.minimum(
                    edge_growth[within], control.maximum_growth_rate
                )
        constrained = np.isfinite(edge_growth)
        if np.any(constrained):
            constrained_edges = edges[constrained]
            rates = edge_growth[constrained]
            for _ in range(points.shape[0]):
                previous = resolved.copy()
                first = constrained_edges[:, 0]
                second = constrained_edges[:, 1]
                np.minimum.at(resolved, first, previous[second] * rates)
                np.minimum.at(resolved, second, previous[first] * rates)
                if np.any(resolved < admissible_minimum):
                    raise ValueError(
                        "Hard growth limits are incompatible with hard size intervals."
                    )
                if np.array_equal(previous, resolved):
                    break
    field = ResolvedSizeField(
        domain,
        points,
        resolved,
        source_control_ids=tuple(control.control_id for control in controls_),
    )
    return field, SizeResolutionReport(
        controls_,
        field,
        winning_control_ids=tuple(winners),
        overlapping_scope_pairs=tuple(overlaps),
        clamped=not np.array_equal(resolved, preferred),
    )


def normalize_mesh_metric(
    metric: MeshMetricField,
    /,
    *,
    target_complexity: float | None = None,
    adjacency: ArrayLike | None = None,
) -> MeshMetricField:
    """Clamp SPD eigenvalues, anisotropy, complexity, and scalar gradation."""

    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField.")
    values = np.asarray(metric.values, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    lower = 1.0 / metric.maximum_size**2
    upper = 1.0 / metric.minimum_size**2
    eigenvalues = np.clip(eigenvalues, lower, upper)
    minimum_eigenvalue = eigenvalues[:, :1]
    eigenvalues = np.minimum(
        eigenvalues,
        minimum_eigenvalue * metric.maximum_anisotropy**2,
    )
    dimension = values.shape[-1]
    if target_complexity is not None:
        target = float(target_complexity)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_complexity must be positive and finite.")
        current = float(np.sum(np.sqrt(np.prod(eigenvalues, axis=1))))
        eigenvalues *= (target / current) ** (2.0 / dimension)
        eigenvalues = np.clip(eigenvalues, lower, upper)
    if adjacency is not None:
        edges = np.asarray(adjacency, dtype=np.int32)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Metric adjacency must have shape (edges, 2).")
        sizes = np.power(np.prod(eigenvalues, axis=1), -1.0 / (2.0 * dimension))
        for _ in range(min(sizes.size, 64)):
            previous = sizes.copy()
            first = edges[:, 0]
            second = edges[:, 1]
            np.minimum.at(sizes, first, previous[second] * metric.maximum_gradation)
            np.minimum.at(sizes, second, previous[first] * metric.maximum_gradation)
            if np.array_equal(previous, sizes):
                break
        current_sizes = np.power(np.prod(eigenvalues, axis=1), -1.0 / (2.0 * dimension))
        eigenvalues *= (current_sizes / sizes)[:, None] ** 2
        eigenvalues = np.clip(eigenvalues, lower, upper)
    normalized = contract(
        "nij,nj,nkj->nik",
        eigenvectors,
        eigenvalues,
        eigenvectors,
        optimize=True,
    )
    return MeshMetricField(
        metric.scope,
        normalized,
        minimum_size=metric.minimum_size,
        maximum_size=metric.maximum_size,
        maximum_anisotropy=metric.maximum_anisotropy,
        maximum_gradation=metric.maximum_gradation,
    )


__all__ = [
    "CurvatureSizeControl",
    "MeshMetricField",
    "ProximitySizeControl",
    "ResolvedSizeField",
    "SizeCombinationPolicy",
    "SizeCompliancePolicy",
    "SizeControl",
    "SizeControlStrength",
    "SizeFieldDomain",
    "SizeResolutionReport",
    "UniformSizeControl",
    "normalize_mesh_metric",
    "resolve_size_controls",
]
