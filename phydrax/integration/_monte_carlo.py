#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._sampling import design_capabilities, design_name, materialize_design
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    Boundary,
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._domain import RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._probability import _open_unit_interval, ProbabilityDomain
from ..domain._scalar import _AbstractScalarDomain
from ..domain._structure import PointsBatch, ProductStructure
from ..domain.geometry2d._from_cad import Geometry2DFromCAD
from ..domain.geometry3d._mesh import Geometry3DFromCAD
from ._batches import PointIntegrationBatch, WeightedSampleBatch
from ._estimates import (
    AntitheticDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
    MonteCarloDiagnostics,
    RandomizedQMCDiagnostics,
    StratifiedDiagnostics,
)
from ._lowering import (
    axes_for_over,
    component_factor_fields,
    materialize_sampled_component,
)
from ._plans import (
    AntitheticDesign,
    ControlVariateEstimator,
    IIDDesign,
    ImportanceSamplingPlan,
    LatinHypercubeDesign,
    MonteCarloPlan,
    QuasiMonteCarloPlan,
    RandomizedQMCDesign,
    StratifiedDesign,
    StratifiedMonteCarloPlan,
)
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget, ProbabilityTarget


def _unwrap(factor: Any, /) -> Any:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _component_base(target: ComponentTarget | DensityTarget, /) -> ComponentTarget:
    if isinstance(target, DensityTarget):
        if not isinstance(target.base, ComponentTarget):
            raise TypeError("This sampling plan requires a component target.")
        return target.base
    return target


def _target_domain(target: Any, /) -> Any:
    base = target.base if isinstance(target, DensityTarget) else target
    if isinstance(base, ComponentTarget):
        if isinstance(base.component, DomainComponentUnion):
            raise TypeError(
                "Monte Carlo component unions must be integrated term by term."
            )
        return base.component.domain
    if isinstance(base, ProbabilityTarget):
        return base.probability
    raise TypeError(f"Target {type(target).__name__} has no domain for sampling.")


def _as_domain_function(value: Any, domain: Any, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=domain, deps=(), func=value)


def _default_structure(component: DomainComponent, /) -> ProductStructure:
    varying = tuple(
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.component_for(label), (FixedStart, FixedEnd, Fixed)
        )
    )
    if not varying:
        raise ValueError("Monte Carlo sampling requires at least one non-fixed label.")
    return ProductStructure((varying,))


def _design_sampler(design: Any, /) -> str:
    return design_name(design)


def _materialize_probability(
    target: ProbabilityTarget,
    num_samples: int,
    sampler: str,
    key: Key[Array, ""],
    /,
) -> PointIntegrationBatch:
    probability = target.probability
    values = probability.sample(num_samples, sampler=sampler, key=key)
    structure = ProductStructure(((probability.label,),)).canonicalize(
        (probability.label,)
    )
    axis = structure.axis_for(probability.label)
    if axis is None:
        raise RuntimeError("Probability sample structure has no axis.")
    points = PointsBatch(
        frozendict(
            {
                probability.label: cx.Field(
                    jnp.asarray(values).reshape((num_samples,)), dims=(axis,)
                )
            }
        ),
        structure,
    )
    weights = cx.Field(jnp.full((num_samples,), 1.0 / float(num_samples)), dims=(axis,))
    return PointIntegrationBatch(
        points,
        weights,
        axes=(axis,),
        target_mass=jnp.asarray(1.0),
        provenance=f"monte-carlo:{sampler}",
    )


def _materialize_direct_once(
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    plan: MonteCarloPlan | QuasiMonteCarloPlan,
    design: Any,
    key: Key[Array, ""],
    /,
) -> PointIntegrationBatch:
    sampler = _design_sampler(design)
    base = target.base if isinstance(target, DensityTarget) else target
    if isinstance(base, ProbabilityTarget):
        return _materialize_probability(base, plan.num_samples, sampler, key)
    if not isinstance(target, (ComponentTarget, DensityTarget)):
        raise TypeError("Component sampling requires a component-backed target.")
    component_target = _component_base(target)
    if isinstance(component_target.component, DomainComponentUnion):
        raise TypeError("Materialize component-union Monte Carlo terms separately.")
    structure = _default_structure(component_target.component)
    points = component_target.component.sample(
        plan.num_samples,
        structure=structure,
        sampler=sampler,
        key=key,
    )
    batch = materialize_sampled_component(component_target, points)
    if not isinstance(batch, PointIntegrationBatch):
        raise TypeError("Monte Carlo sampling requires a paired point batch.")
    return PointIntegrationBatch(
        batch.points,
        batch.weights,
        axes=batch.axes,
        mask=batch.mask,
        target_mass=batch.target_mass,
        provenance=f"monte-carlo:{sampler}",
    )


def _fixed_point(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, _AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Expected a fixed scalar selector.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, _AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.var_dim,)),
            dims=(None,),
        )
    raise TypeError("Unsupported fixed factor in sample design.")


def _materialize_antithetic(
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    plan: MonteCarloPlan,
    design: AntitheticDesign,
    key: Key[Array, ""],
    /,
) -> PointIntegrationBatch:
    if plan.num_samples % 2:
        raise ValueError("Antithetic designs require an even num_samples.")
    domain = _target_domain(target)
    if isinstance(domain, ProbabilityDomain):
        labels = (domain.label,)
        factors = (domain,)
        selectors = (None,)
        fixed_labels = frozenset()
    else:
        if not isinstance(target, (ComponentTarget, DensityTarget)):
            raise TypeError("Antithetic sampling requires a supported target.")
        component = _component_base(target).component
        if isinstance(component, DomainComponentUnion):
            raise TypeError("Antithetic component unions are not supported.")
        labels = component.domain.labels
        factors = tuple(_unwrap(component.domain.factor(label)) for label in labels)
        selectors = tuple(component.spec.component_for(label) for label in labels)
        fixed_labels = frozenset(
            label
            for label, selector in zip(labels, selectors, strict=True)
            if isinstance(selector, (FixedStart, FixedEnd, Fixed))
        )
    varying = tuple(label for label in labels if label not in fixed_labels)
    if not isinstance(domain, ProbabilityDomain):
        unsupported = tuple(
            label
            for label, selector in zip(labels, selectors, strict=True)
            if label in varying and not isinstance(selector, Interior)
        )
        if unsupported:
            raise TypeError(
                "Antithetic sampling supports only Interior() or fixed component "
                f"selectors; unsupported labels: {unsupported!r}."
            )
    varying_factors = tuple(
        factor for label, factor in zip(labels, factors, strict=True) if label in varying
    )
    if any(not isinstance(factor, _AbstractScalarDomain) for factor in varying_factors):
        raise TypeError(
            "The default antithetic map supports scalar and probability domains only; "
            "supply external paired samples for general geometry."
        )
    pairs = plan.num_samples // 2
    capabilities = design_capabilities(design.base)
    design_key = key if capabilities.randomized else None
    unit = materialize_design(
        design.base,
        count=pairs,
        dimension=len(varying),
        key=design_key,
    )
    if design.involution is None:
        reflected = 1.0 - unit
    else:
        reflected = jnp.asarray(design.involution(unit), dtype=float)
        if reflected.shape != unit.shape:
            raise ValueError("Antithetic involution must preserve the design shape.")
    paired = jnp.concatenate((unit, reflected), axis=0)
    structure = ProductStructure((varying,)).canonicalize(
        labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(varying[0])
    if axis is None:
        raise RuntimeError("Antithetic sample structure has no axis.")
    integration_axes = (
        (axis,)
        if isinstance(target, ProbabilityTarget)
        else axes_for_over(structure, _component_base(target).axes)
    )
    points: dict[str, cx.Field] = {}
    varying_index = {label: index for index, label in enumerate(varying)}
    for label, factor, selector in zip(labels, factors, selectors, strict=True):
        if label in fixed_labels:
            points[label] = _fixed_point(factor, selector)
            continue
        coordinate = paired[:, varying_index[label]]
        if isinstance(factor, ProbabilityDomain):
            values = factor.distribution.icdf(_open_unit_interval(coordinate))
        else:
            values = factor.fixed("start") + coordinate * (
                factor.fixed("end") - factor.fixed("start")
            )
        points[label] = cx.Field(jnp.asarray(values), dims=(axis,))
    batch_points = PointsBatch(frozendict(points), structure)
    weights = cx.Field(
        jnp.full((plan.num_samples,), 1.0 / float(plan.num_samples)), dims=(axis,)
    )
    if isinstance(domain, ProbabilityDomain):
        mass = jnp.asarray(1.0)
    else:
        if not isinstance(target, (ComponentTarget, DensityTarget)):
            raise TypeError("Antithetic sampling requires a supported target.")
        mass = _component_base(target).component.measure()
    return PointIntegrationBatch(
        batch_points,
        weights,
        axes=integration_axes,
        target_mass=mass,
        provenance="monte-carlo:antithetic",
    )


def materialize_monte_carlo(
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    plan: MonteCarloPlan | QuasiMonteCarloPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> PointIntegrationBatch | tuple[PointIntegrationBatch, ...]:
    """Materialize direct stochastic samples while preserving design provenance."""
    design = plan.design
    if isinstance(design, AntitheticDesign):
        if not isinstance(plan, MonteCarloPlan):
            raise TypeError("Antithetic designs require a MonteCarloPlan.")
        return _materialize_antithetic(target, plan, design, key)
    if isinstance(design, RandomizedQMCDesign) and design.num_replicates > 1:
        keys = jr.split(key, design.num_replicates)
        return tuple(
            _materialize_direct_once(target, plan, design, replicate_key)
            for replicate_key in keys
        )
    return _materialize_direct_once(target, plan, design, key)


def _stratified_partition(
    component: DomainComponent,
    design: StratifiedDesign,
    /,
) -> tuple[str, Any]:
    if design.partition is not None:
        nonfixed = tuple(
            label
            for label in component.domain.labels
            if not isinstance(
                component.spec.component_for(label), (FixedStart, FixedEnd, Fixed)
            )
        )
        if len(nonfixed) != 1:
            raise ValueError("Explicit stratification requires one non-fixed label.")
        return nonfixed[0], design.partition
    candidates: list[tuple[str, Any]] = []
    for label in component.domain.labels:
        selector = component.spec.component_for(label)
        if isinstance(selector, (FixedStart, FixedEnd, Fixed)):
            continue
        factor = _unwrap(component.domain.factor(label))
        if isinstance(factor, Geometry2DFromCAD):
            partition = (
                factor.boundary_measure_partition
                if isinstance(selector, Boundary)
                else factor.interior_measure_partition
            )
            candidates.append((label, partition))
        elif isinstance(factor, Geometry3DFromCAD) and isinstance(selector, Boundary):
            candidates.append((label, factor.boundary_measure_partition))
    if len(candidates) != 1:
        raise ValueError(
            "Stratified Monte Carlo requires one explicit GeometryMeasurePartition."
        )
    return candidates[0]


def materialize_stratified(
    target: ComponentTarget | DensityTarget,
    plan: StratifiedMonteCarloPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> PointIntegrationBatch:
    """Materialize a physical-measure stratified component batch."""
    component_target = _component_base(target)
    component = component_target.component
    if isinstance(component, DomainComponentUnion):
        raise TypeError("Stratified component unions must be integrated term by term.")
    label, partition = _stratified_partition(component, plan.design)
    target_mass = partition.measures / partition.total_measure
    if plan.design.allocation == "proportional":
        allocation_weights = None
    elif plan.design.allocation == "equal":
        allocation_weights = 1.0 / target_mass
    else:
        requested = jnp.asarray(plan.design.allocation_weights, dtype=float)
        if requested.shape != target_mass.shape:
            raise ValueError("Explicit allocation weights must match num_strata.")
        allocation_weights = requested / target_mass
    points_array, strata, represented = partition.sample(
        plan.num_samples,
        key=key,
        stratum_weights=allocation_weights,
        minimum_per_stratum=2,
    )
    fixed_labels = frozenset(
        other
        for other in component.domain.labels
        if isinstance(component.spec.component_for(other), (FixedStart, FixedEnd, Fixed))
    )
    structure = ProductStructure(((label,),)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    axis = structure.axis_for(label)
    if axis is None:
        raise RuntimeError("Stratified sample structure has no axis.")
    values: dict[str, cx.Field] = {}
    for other in component.domain.labels:
        selector = component.spec.component_for(other)
        factor = component.domain.factor(other)
        if other == label:
            raw = jnp.asarray(points_array)
            dims = (axis,) + (None,) * (raw.ndim - 1)
            values[other] = cx.Field(raw, dims=dims)
        else:
            values[other] = _fixed_point(factor, selector)
    point_batch = PointsBatch(frozendict(values), structure)
    weights = cx.Field(jnp.asarray(represented) * partition.total_measure, dims=(axis,))
    return PointIntegrationBatch(
        point_batch,
        weights,
        axes=(axis,),
        target_mass=partition.total_measure,
        stratum_indices=strata,
        num_strata=partition.num_strata,
        provenance=f"stratified:{plan.design.allocation}",
    )


def _expand_and_flatten(
    field: cx.Field,
    batch: PointIntegrationBatch,
    /,
) -> tuple[Array, tuple[Any, ...]]:
    template = cx.Field(jnp.ones(batch.weights.data.shape), dims=batch.weights.dims)
    expanded = field * template
    positions = tuple(expanded.dims.index(axis) for axis in batch.axes)
    data = jnp.moveaxis(expanded.data, positions, tuple(range(len(positions))))
    sample_count = 1
    for size in batch.weights.data.shape:
        sample_count *= int(size)
    output_shape = data.shape[len(positions) :]
    output_dims = tuple(dim for dim in expanded.dims if dim not in batch.axes)
    return jnp.reshape(data, (sample_count,) + output_shape), output_dims


def _sample_values(
    integrand: Any,
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    batch: PointIntegrationBatch,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[Array, Array, Array | None, Array, tuple[Any, ...]]:
    domain = _target_domain(target)
    function = _as_domain_function(integrand, domain)
    value_field = function(batch.points, key=key, **kwargs)
    if not isinstance(value_field, cx.Field):
        raise TypeError("Monte Carlo integrands must evaluate to coordax.Field.")
    values, output_dims = _expand_and_flatten(value_field, batch)
    base = target.base if isinstance(target, DensityTarget) else target
    if isinstance(base, ComponentTarget):
        if isinstance(base.component, DomainComponentUnion):
            raise TypeError("Component unions must be sampled term by term.")
        mask, modifier = component_factor_fields(
            base.component, batch.points, key=key, kwargs=kwargs
        )
        base_factor_field = mask * modifier
    else:
        base_factor_field = cx.Field(jnp.asarray(1.0), dims=())
    if batch.mask is not None:
        base_factor_field = base_factor_field * batch.mask
    factor_field = base_factor_field
    if isinstance(target, DensityTarget):
        density_function = _as_domain_function(target.log_density, domain)
        log_density = density_function(batch.points, key=key, **kwargs)
        factor_field = factor_field * cx.Field(
            jnp.exp(jnp.asarray(log_density.data)), dims=log_density.dims
        )
    normalizer_field = None
    if isinstance(target, DensityTarget):
        if target.normalized:
            normalizer_field = factor_field
        elif isinstance(base, ComponentTarget) and base.normalized:
            normalizer_field = base_factor_field
    elif isinstance(target, (ComponentTarget, ProbabilityTarget)) and target.normalized:
        normalizer_field = base_factor_field
    factors, factor_dims = _expand_and_flatten(factor_field, batch)
    if factor_dims or factors.ndim != 1:
        raise ValueError("Stochastic target weights must be scalar per sample.")
    normalizer_factors = None
    if normalizer_field is not None:
        normalizer_factors, normalizer_dims = _expand_and_flatten(normalizer_field, batch)
        if normalizer_dims or normalizer_factors.ndim != 1:
            raise ValueError("Stochastic normalizers must be scalar per sample.")
    base_weights = jnp.reshape(jnp.asarray(batch.weights.data), (-1,))
    return values, factors, normalizer_factors, base_weights, output_dims


def _control_values(
    control: Any,
    target: Any,
    batch: PointIntegrationBatch,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> Array:
    function = _as_domain_function(control, _target_domain(target))
    field = function(batch.points, key=key, **kwargs)
    values, dims = _expand_and_flatten(field, batch)
    if dims or values.ndim != 1:
        raise ValueError("Each control variate must be scalar-valued.")
    return values


def _apply_control_variate(
    values: Array,
    target: Any,
    batch: PointIntegrationBatch,
    control: ControlVariateEstimator,
    /,
    *,
    independent_pilot: bool,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[Array, int]:
    controls = jnp.stack(
        tuple(
            _control_values(item, target, batch, key=key, kwargs=kwargs)
            for item in control.controls
        ),
        axis=1,
    )
    expected = jnp.asarray(control.expectations, dtype=controls.dtype)
    flat_values = jnp.reshape(values, (values.shape[0], -1))
    production_start = 0
    if control.coefficients is None:
        if control.same_sample_asymptotic:
            fit_controls = controls
            fit_values = flat_values
        else:
            if not independent_pilot:
                raise ValueError(
                    "Independent fitted control coefficients require an IID design; "
                    "supply coefficients or opt into same_sample_asymptotic=True."
                )
            production_start = control.pilot_samples
            if production_start >= values.shape[0]:
                raise ValueError(
                    "control pilot_samples must be smaller than plan.num_samples."
                )
            fit_controls = controls[:production_start]
            fit_values = flat_values[:production_start]
        centered_controls = fit_controls - jnp.mean(fit_controls, axis=0, keepdims=True)
        centered_values = fit_values - jnp.mean(fit_values, axis=0, keepdims=True)
        gram = centered_controls.T @ centered_controls
        gram = gram + control.regularization * jnp.eye(gram.shape[0])
        coefficients = jnp.linalg.solve(gram, centered_controls.T @ centered_values)
    else:
        coefficients = jnp.asarray(control.coefficients, dtype=flat_values.dtype)
        if coefficients.ndim == 0:
            coefficients = coefficients.reshape((1, 1))
        if coefficients.ndim == 1:
            coefficients = coefficients[:, None]
        if coefficients.shape != (controls.shape[1], flat_values.shape[1]):
            raise ValueError(
                "Control coefficients must have shape (num_controls, output_size)."
            )
    production_values = flat_values[production_start:]
    production_controls = controls[production_start:]
    corrected = production_values - (production_controls - expected) @ coefficients
    output_shape = (corrected.shape[0],) + values.shape[1:]
    return jnp.reshape(corrected, output_shape), production_start


def _mean_and_error(
    values: Array,
    factors: Array,
    normalizer_factors: Array | None,
    base_weights: Array,
    mass: Array,
    /,
) -> tuple[Array, Array, Array]:
    weighted = base_weights * factors
    numerator = jnp.tensordot(weighted, values, axes=(0, 0))
    factor_shape = (-1,) + (1,) * (values.ndim - 1)
    if normalizer_factors is not None:
        denominator = jnp.sum(base_weights * normalizer_factors)
        estimate = numerator / denominator
        mean_normalizer = jnp.mean(normalizer_factors)
        influence = (
            factors.reshape(factor_shape) * values
            - normalizer_factors.reshape(factor_shape) * estimate
        ) / mean_normalizer
        valid_mass = jnp.isfinite(denominator) & (denominator != 0.0)
    else:
        estimate = numerator
        influence = mass * factors.reshape(factor_shape) * values
        valid_mass = jnp.asarray(True)
    count = values.shape[0]
    centered = influence - jnp.mean(influence, axis=0)
    variance = jnp.sum(jnp.real(centered * jnp.conj(centered)), axis=0) / max(
        count - 1, 1
    )
    standard_error = jnp.sqrt(variance / count)
    return estimate, standard_error, valid_mass


def _error_norm(error: Array, /) -> Array:
    return jnp.max(jnp.asarray(error))


def _integrate_stratified_samples(
    values: Array,
    factors: Array,
    normalizer_factors: Array | None,
    base_weights: Array,
    batch: PointIntegrationBatch,
    output_dims: tuple[Any, ...],
    /,
) -> IntegrationEstimate:
    if batch.stratum_indices is None or batch.num_strata is None:
        raise RuntimeError("Stratified batch is missing stratum metadata.")
    strata = batch.stratum_indices
    count = batch.num_strata
    membership = jax.nn.one_hot(strata, count, dtype=float).T
    counts = jnp.sum(membership, axis=1)
    masses = membership @ base_weights
    factor_shape = (-1,) + (1,) * (values.ndim - 1)
    effective = factors.reshape(factor_shape) * values
    numerator = jnp.tensordot(base_weights, effective, axes=(0, 0))
    normalized = normalizer_factors is not None
    if normalizer_factors is not None:
        denominator = jnp.sum(base_weights * normalizer_factors)
        estimate = numerator / denominator
        error_samples = (
            effective - normalizer_factors.reshape(factor_shape) * estimate
        ) / denominator
    else:
        denominator = jnp.sum(base_weights)
        estimate = numerator
        error_samples = effective
    expand = (slice(None),) + (None,) * (values.ndim - 1)
    stratum_means = jnp.einsum("hn,n...->h...", membership, effective) / counts[expand]
    error_means = jnp.einsum("hn,n...->h...", membership, error_samples) / counts[expand]
    centered = error_samples - error_means[strata]
    stratum_variances = (
        jnp.einsum(
            "hn,n...->h...",
            membership,
            jnp.real(centered * jnp.conj(centered)),
        )
        / jnp.maximum(counts - 1.0, 1.0)[expand]
    )
    variance = jnp.sum(masses[expand] ** 2 * stratum_variances / counts[expand], axis=0)
    standard_error = jnp.sqrt(variance)
    contributions = masses[expand] * stratum_means
    if normalized:
        contributions = contributions / denominator
    status = jnp.where(
        jnp.all(counts > 0),
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.UNSAMPLED_STRATUM),
    )
    valid_mass = jnp.isfinite(denominator) & (denominator != 0.0)
    if normalized:
        status = jnp.where(
            valid_mass,
            status,
            int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
        )
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(factors))
        & jnp.all(jnp.isfinite(base_weights))
        & (
            True
            if normalizer_factors is None
            else jnp.all(jnp.isfinite(normalizer_factors))
        )
    )
    status = jnp.where(
        finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    diagnostics = StratifiedDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(values.shape[0], dtype=jnp.int32),
        standard_error=standard_error,
        samples_per_stratum=counts.astype(jnp.int32),
        stratum_estimates=stratum_means,
        stratum_variances=stratum_variances,
        stratum_contributions=contributions,
    )
    return IntegrationEstimate(
        cx.Field(estimate, dims=output_dims),
        status=status,
        num_evaluations=values.shape[0],
        error_estimate=_error_norm(standard_error),
        error_kind="stratified-standard-error",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "stratified-monte-carlo", "component", batch.provenance
        ),
    )


def _integrate_antithetic_samples(
    values: Array,
    factors: Array,
    normalizer_factors: Array | None,
    batch: PointIntegrationBatch,
    output_dims: tuple[Any, ...],
    /,
    *,
    uncertainty_supported: bool,
) -> IntegrationEstimate:
    count = values.shape[0]
    pairs = count // 2
    factor_shape = (-1,) + (1,) * (values.ndim - 1)
    effective = factors.reshape(factor_shape) * values
    normalized = normalizer_factors is not None
    if normalizer_factors is not None:
        denominator = jnp.mean(normalizer_factors)
        estimate = jnp.mean(effective, axis=0) / denominator
        observations = (
            effective - normalizer_factors.reshape(factor_shape) * estimate
        ) / denominator + estimate
    else:
        if batch.target_mass is None:
            raise RuntimeError("Antithetic component batches require target_mass.")
        denominator = jnp.asarray(1.0)
        observations = batch.target_mass * effective
        estimate = jnp.mean(observations, axis=0)
    first = observations[:pairs]
    second = observations[pairs:]
    if uncertainty_supported:
        pair_means = 0.5 * (first + second)
        centered_pairs = pair_means - jnp.mean(pair_means, axis=0)
        pair_variance = jnp.sum(
            jnp.real(centered_pairs * jnp.conj(centered_pairs)), axis=0
        ) / (pairs - 1)
        standard_error = jnp.sqrt(pair_variance / pairs)
        centered_first = first - jnp.mean(first, axis=0)
        centered_second = second - jnp.mean(second, axis=0)
        covariance = jnp.sum(
            jnp.real(centered_first * jnp.conj(centered_second)), axis=0
        ) / (pairs - 1)
        centered_all = observations - jnp.mean(observations, axis=0)
        ordinary_variance = jnp.sum(
            jnp.real(centered_all * jnp.conj(centered_all)), axis=0
        ) / (count - 1)
        ordinary_estimator_variance = ordinary_variance / count
        antithetic_estimator_variance = pair_variance / pairs
        reduction = ordinary_estimator_variance / jnp.maximum(
            antithetic_estimator_variance, jnp.finfo(float).tiny
        )
        reported_error = _error_norm(standard_error)
        error_kind = "antithetic-pair-standard-error"
    else:
        standard_error = None
        covariance = None
        reduction = None
        reported_error = None
        error_kind = None
    valid_mass = jnp.isfinite(denominator) & (denominator != 0.0)
    status = jnp.where(
        valid_mass,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    )
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(factors))
        & (
            True
            if normalizer_factors is None
            else jnp.all(jnp.isfinite(normalizer_factors))
        )
    )
    status = jnp.where(
        finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    diagnostics = AntitheticDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(count, dtype=jnp.int32),
        standard_error=standard_error,
        num_pairs=jnp.asarray(pairs, dtype=jnp.int32),
        pair_covariance=covariance,
        variance_reduction_factor=reduction,
    )
    return IntegrationEstimate(
        cx.Field(estimate, dims=output_dims),
        status=status,
        num_evaluations=count,
        error_estimate=reported_error,
        error_kind=error_kind,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "antithetic-monte-carlo", "component", batch.provenance
        ),
    )


def integrate_monte_carlo_batch(
    integrand: Any,
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    batch: PointIntegrationBatch,
    /,
    *,
    plan: MonteCarloPlan | QuasiMonteCarloPlan | StratifiedMonteCarloPlan | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce a materialized stochastic point batch with method-correct error."""
    callback_kwargs = {} if kwargs is None else kwargs
    values, factors, normalizer_factors, base_weights, output_dims = _sample_values(
        integrand, target, batch, key=key, kwargs=callback_kwargs
    )
    if (
        isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan))
        and plan.control_variate is not None
    ):
        values, production_start = _apply_control_variate(
            values,
            target,
            batch,
            plan.control_variate,
            independent_pilot=isinstance(plan.design, IIDDesign),
            key=key,
            kwargs=callback_kwargs,
        )
        if production_start:
            original_mass = jnp.sum(base_weights)
            factors = factors[production_start:]
            if normalizer_factors is not None:
                normalizer_factors = normalizer_factors[production_start:]
            base_weights = base_weights[production_start:]
            base_weights = base_weights * (original_mass / jnp.sum(base_weights))
    if batch.stratum_indices is not None:
        return _integrate_stratified_samples(
            values,
            factors,
            normalizer_factors,
            base_weights,
            batch,
            output_dims,
        )
    if batch.provenance == "monte-carlo:antithetic":
        design = plan.design if isinstance(plan, MonteCarloPlan) else None
        uncertainty_supported = (
            isinstance(design, AntitheticDesign)
            and isinstance(design.base, IIDDesign)
            and values.shape[0] // 2 >= 2
        )
        return _integrate_antithetic_samples(
            values,
            factors,
            normalizer_factors,
            batch,
            output_dims,
            uncertainty_supported=uncertainty_supported,
        )
    mass = jnp.asarray(1.0) if batch.target_mass is None else batch.target_mass
    estimate, standard_error, valid_mass = _mean_and_error(
        values,
        factors,
        normalizer_factors,
        base_weights,
        mass,
    )
    normalized = normalizer_factors is not None
    status = jnp.where(
        valid_mass | (not normalized),
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    )
    status = jnp.where(
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(factors))
        & jnp.all(jnp.isfinite(base_weights))
        & (
            True
            if normalizer_factors is None
            else jnp.all(jnp.isfinite(normalizer_factors))
        ),
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    design = (
        plan.design if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)) else None
    )
    qmc_design = design if isinstance(design, RandomizedQMCDesign) else None
    uncertainty_supported = qmc_design is None and not isinstance(
        design, LatinHypercubeDesign
    )
    reported_standard_error = standard_error if uncertainty_supported else None
    reported_error = (
        _error_norm(reported_standard_error)
        if reported_standard_error is not None
        else None
    )
    error_kind = "iid-standard-error" if uncertainty_supported else None
    reduction_finite = jnp.all(jnp.isfinite(estimate))
    if reported_standard_error is not None:
        reduction_finite = reduction_finite & jnp.all(
            jnp.isfinite(reported_standard_error)
        )
    status = jnp.where(
        (status != int(IntegrationStatus.CONVERGED)) | reduction_finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    diagnostics = MonteCarloDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(values.shape[0], dtype=jnp.int32),
        standard_error=reported_standard_error,
        num_samples=jnp.asarray(values.shape[0], dtype=jnp.int32),
        num_independent_replicates=jnp.asarray(1, dtype=jnp.int32),
        target_mass=batch.target_mass,
    )
    return IntegrationEstimate(
        cx.Field(estimate, dims=output_dims),
        status=status,
        num_evaluations=values.shape[0],
        error_estimate=reported_error,
        error_kind=error_kind,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "quasi-monte-carlo" if qmc_design is not None else "monte-carlo",
            "sampled",
            batch.provenance,
        ),
    )


def _require_point_batch(batch: object, /) -> PointIntegrationBatch:
    if not isinstance(batch, PointIntegrationBatch):
        raise TypeError("Expected a point integration batch.")
    return batch


def integrate_monte_carlo(
    integrand: Any,
    target: ComponentTarget | DensityTarget | ProbabilityTarget,
    realization: PointIntegrationBatch | tuple[PointIntegrationBatch, ...],
    /,
    *,
    plan: MonteCarloPlan | QuasiMonteCarloPlan | StratifiedMonteCarloPlan | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce one sample batch or independent randomized-QMC replicates."""
    if not isinstance(realization, tuple):
        return integrate_monte_carlo_batch(
            integrand, target, realization, plan=plan, key=key, kwargs=kwargs
        )
    batches = tuple(_require_point_batch(batch) for batch in realization)
    keys = jr.split(key, len(batches))
    estimates = tuple(
        integrate_monte_carlo_batch(
            integrand,
            target,
            batch,
            plan=plan,
            key=keys[index],
            kwargs=kwargs,
        )
        for index, batch in enumerate(batches)
    )
    replicate_values = jnp.stack(tuple(estimate.value.data for estimate in estimates))
    count = len(estimates)
    mean = jnp.mean(replicate_values, axis=0)
    if count > 1:
        standard_error = jnp.std(replicate_values, axis=0, ddof=1) / jnp.sqrt(count)
        error = _error_norm(standard_error)
    else:
        standard_error = None
        error = None
    status = jnp.max(jnp.stack(tuple(estimate.status for estimate in estimates)))
    design = (
        plan.design if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)) else None
    )
    if not isinstance(design, RandomizedQMCDesign):
        raise TypeError("Replicated batches require RandomizedQMCDesign provenance.")
    total_evaluations = sum(jnp.asarray(batch.weights.data).size for batch in batches)
    diagnostics = RandomizedQMCDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(total_evaluations, dtype=jnp.int32),
        standard_error=standard_error,
        num_samples_per_replicate=jnp.asarray(
            jnp.asarray(batches[0].weights.data).size, dtype=jnp.int32
        ),
        num_independent_replicates=jnp.asarray(count, dtype=jnp.int32),
        replicate_estimates=replicate_values,
        scrambled=design.scrambled,
        sequence=design.sequence,
    )
    return IntegrationEstimate(
        cx.Field(mean, dims=estimates[0].value.dims),
        status=status,
        num_evaluations=total_evaluations,
        error_estimate=error,
        error_kind="randomized-qmc-replicate-error" if count > 1 else None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("quasi-monte-carlo", "sampled", design.sequence),
    )


def _proposal_covers_probability(probability: Any, proposal: Any, /) -> Array:
    distribution = probability.distribution
    quantiles = _open_unit_interval(jnp.linspace(0.0, 1.0, 257))
    probes = distribution.icdf(quantiles)
    covered = jnp.all(proposal.contains(probes))
    target_support = distribution.support
    proposal_support = proposal.support
    if target_support is None:
        if proposal_support is not None:
            covered = jnp.asarray(False)
    else:
        endpoints = jnp.stack(tuple(jnp.asarray(value) for value in target_support))
        covered = covered & jnp.all(proposal.contains(endpoints))
    return covered


def materialize_importance(
    target: ProbabilityTarget | DensityTarget,
    plan: ImportanceSamplingPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> WeightedSampleBatch:
    """Draw proposal samples and retain raw target-to-proposal log ratios."""
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ProbabilityTarget):
        raise TypeError(
            "Importance sampling currently requires a ProbabilityTarget base."
        )
    probability = base.probability
    support_valid = _proposal_covers_probability(probability, plan.proposal)
    samples = jnp.asarray(
        plan.proposal.sample(key, sample_shape=(plan.num_samples,)), dtype=float
    ).reshape((plan.num_samples,))
    structure = ProductStructure(((probability.label,),)).canonicalize(
        (probability.label,)
    )
    axis = structure.axis_for(probability.label)
    if axis is None:
        raise RuntimeError("Importance sample structure has no axis.")
    points = PointsBatch(
        frozendict({probability.label: cx.Field(samples, dims=(axis,))}), structure
    )
    log_weights = probability.distribution.log_prob(samples) - plan.proposal.log_prob(
        samples
    )
    if isinstance(target, DensityTarget):
        density_function = _as_domain_function(target.log_density, probability)
        density_values = density_function(points, key=key)
        log_weights = log_weights + density_values.data
    return WeightedSampleBatch(
        points,
        log_weights,
        target_mass=None,
        support_valid=support_valid,
        independent=True,
        provenance=f"importance:{type(plan.proposal).__name__}",
    )


__all__ = [
    "integrate_monte_carlo",
    "integrate_monte_carlo_batch",
    "materialize_importance",
    "materialize_monte_carlo",
    "materialize_stratified",
]
