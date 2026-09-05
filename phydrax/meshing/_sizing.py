#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

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


def _size(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


class UniformSizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    target_size: float = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_growth_rate: float = eqx.field(static=True)
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
        maximum_growth_rate: float = 1.3,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        target = _size(target_size, "target_size")
        minimum = target if minimum_size is None else _size(minimum_size, "minimum_size")
        maximum = target if maximum_size is None else _size(maximum_size, "maximum_size")
        growth = float(maximum_growth_rate)
        if minimum > target or target > maximum:
            raise ValueError("Sizes must satisfy minimum <= target <= maximum.")
        if not np.isfinite(growth) or growth < 1.0:
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
                "sizes": [minimum, target, maximum],
                "growth": growth,
                "strength": strength.value,
                "priority": int(priority),
            }
        )


class CurvatureSizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    normal_angle: float = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    use_faceted_curvature: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        normal_angle: float,
        minimum_size: float,
        maximum_size: float,
        /,
        *,
        use_faceted_curvature: bool = False,
        strength: SizeControlStrength = SizeControlStrength.SOFT,
        priority: int = 0,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        angle = float(normal_angle)
        minimum = _size(minimum_size, "minimum_size")
        maximum = _size(maximum_size, "maximum_size")
        if not np.isfinite(angle) or angle <= 0.0 or angle >= np.pi:
            raise ValueError("normal_angle must lie strictly between zero and pi.")
        if minimum > maximum:
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
                "sizes": [minimum, maximum],
                "use_faceted_curvature": bool(use_faceted_curvature),
                "strength": strength.value,
                "priority": int(priority),
            }
        )


class ProximitySizeControl(StrictModule, NonTrainableState):
    scope: MeshingScope
    elements_per_gap: int = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    include_self_proximity: bool = eqx.field(static=True)
    opposite_normals_only: bool = eqx.field(static=True)
    strength: SizeControlStrength = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: MeshingScope,
        elements_per_gap: int,
        minimum_size: float,
        maximum_size: float,
        /,
        *,
        include_self_proximity: bool = False,
        opposite_normals_only: bool = True,
        strength: SizeControlStrength = SizeControlStrength.HARD,
        priority: int = 0,
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        count = int(elements_per_gap)
        minimum = _size(minimum_size, "minimum_size")
        maximum = _size(maximum_size, "maximum_size")
        if count <= 0:
            raise ValueError("elements_per_gap must be positive.")
        if minimum > maximum:
            raise ValueError("minimum_size cannot exceed maximum_size.")
        if not isinstance(strength, SizeControlStrength):
            raise TypeError("strength must be SizeControlStrength.")
        self.scope = scope
        self.elements_per_gap = count
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.include_self_proximity = bool(include_self_proximity)
        self.opposite_normals_only = bool(opposite_normals_only)
        self.strength = strength
        self.priority = int(priority)
        self.control_id = canonical_fingerprint(
            {
                "kind": "proximity-size-control",
                "scope": scope.scope_id,
                "elements_per_gap": count,
                "sizes": [minimum, maximum],
                "include_self_proximity": bool(include_self_proximity),
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
        overlaps = tuple(
            (str(first), str(second)) for first, second in overlapping_scope_pairs
        )
        self.control_ids = identifiers
        self.overlapping_scope_pairs = overlaps
        self.clamped = bool(clamped)
        self.provider_resolved = bool(provider_resolved)
        self.field_id = field.field_id
        self.report_id = canonical_fingerprint(
            {
                "kind": "size-resolution-report",
                "controls": identifiers,
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
    """Compile scoped uniform/curvature/proximity controls into one size field."""

    if not controls:
        raise ValueError("At least one size control is required.")
    if not isinstance(domain, SizeFieldDomain):
        raise TypeError("domain must be SizeFieldDomain.")
    if not isinstance(combination, SizeCombinationPolicy):
        raise TypeError("combination must be SizeCombinationPolicy.")
    points = np.asarray(sample_points, dtype=float)
    identifiers = np.asarray(sample_entity_ids, dtype=np.int64)
    if points.ndim != 2 or identifiers.shape != (points.shape[0],):
        raise ValueError("Size samples and entity IDs must align.")
    curvature_values = None if curvature is None else np.asarray(curvature, dtype=float)
    proximity_values = None if proximity is None else np.asarray(proximity, dtype=float)
    candidates = []
    masks = []
    hard = []
    overlaps = []
    for left, first in enumerate(controls):
        for second in controls[left + 1 :]:
            if (
                first.scope.entity_set_id == second.scope.entity_set_id
                and np.intersect1d(first.scope.entity_ids, second.scope.entity_ids).size
            ):
                overlaps.append((first.control_id, second.control_id))
        mask = np.isin(identifiers, np.asarray(first.scope.entity_ids, dtype=np.int64))
        if not np.any(mask):
            raise ValueError("A size control resolves to no supplied sample entities.")
        if isinstance(first, UniformSizeControl):
            value = np.full((points.shape[0],), first.target_size)
        elif isinstance(first, CurvatureSizeControl):
            if curvature_values is None or curvature_values.shape != identifiers.shape:
                raise ValueError("Curvature controls require aligned curvature samples.")
            radius = 1.0 / np.maximum(curvature_values, np.finfo(float).tiny)
            value = np.clip(
                2.0 * radius * np.sin(0.5 * first.normal_angle),
                first.minimum_size,
                first.maximum_size,
            )
        elif isinstance(first, ProximitySizeControl):
            if proximity_values is None or proximity_values.shape != identifiers.shape:
                raise ValueError("Proximity controls require aligned gap samples.")
            value = np.clip(
                proximity_values / first.elements_per_gap,
                first.minimum_size,
                first.maximum_size,
            )
        else:
            raise TypeError("Unsupported size control.")
        candidates.append(value)
        masks.append(mask)
        hard.append(first.strength is SizeControlStrength.HARD)
    candidate_array = np.stack(candidates)
    mask_array = np.stack(masks)
    if combination is SizeCombinationPolicy.REJECT_HARD_CONFLICTS:
        for left in range(len(controls)):
            if not hard[left]:
                continue
            for right in range(left + 1, len(controls)):
                overlap = mask_array[left] & mask_array[right]
                if hard[right] and np.any(
                    np.abs(
                        candidate_array[left, overlap] - candidate_array[right, overlap]
                    )
                    > 1.0e-12
                ):
                    raise ValueError("Overlapping hard size controls conflict.")
    raw = np.min(np.where(mask_array, candidate_array, np.inf), axis=0)
    if np.any(~np.isfinite(raw)):
        raise ValueError("Size controls do not cover every sample.")
    resolved = raw.copy()
    growth = min(
        (
            control.maximum_growth_rate
            for control in controls
            if isinstance(control, UniformSizeControl)
        ),
        default=1.0e300,
    )
    if adjacency is not None:
        edges = np.asarray(adjacency, dtype=np.int32)
        if (
            edges.ndim != 2
            or edges.shape[1] != 2
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError("Size-field adjacency must have shape (edges, 2).")
        for _ in range(min(points.shape[0], 64)):
            previous = resolved.copy()
            first = edges[:, 0]
            second = edges[:, 1]
            np.minimum.at(resolved, first, previous[second] * growth)
            np.minimum.at(resolved, second, previous[first] * growth)
            if np.array_equal(previous, resolved):
                break
    field = ResolvedSizeField(
        domain,
        points,
        resolved,
        source_control_ids=tuple(control.control_id for control in controls),
    )
    return field, SizeResolutionReport(
        controls,
        field,
        overlapping_scope_pairs=tuple(overlaps),
        clamped=not np.array_equal(resolved, raw),
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
    "normalize_mesh_metric",
    "ProximitySizeControl",
    "ResolvedSizeField",
    "SizeCombinationPolicy",
    "SizeControl",
    "SizeControlStrength",
    "SizeFieldDomain",
    "SizeResolutionReport",
    "resolve_size_controls",
    "UniformSizeControl",
]
