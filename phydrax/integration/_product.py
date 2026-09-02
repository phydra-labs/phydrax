#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.domain import (
    AbstractGeometry,
    AbstractScalarDomain,
    Boundary,
    ComponentSum,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    PointBatch,
    ProbabilityDomain,
    reference_transport,
    SampleLayout,
)

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._sampling import (
    derive_key,
    DESIGN_ALGORITHM_VERSION,
    design_name,
    materialize_design,
    SampleAddress,
)
from .._strict import StrictModule
from ._batches import PointIntegrationBatch
from ._estimates import (
    IntegrationEstimate,
    IntegrationProvenance,
    ProductIntegrationDiagnostics,
)
from ._fixed import integrate_fixed_component, integrate_fixed_density
from ._lowering import (
    _block_measure,
    _cubature_factor_data,
    _scalar_interior_rule_data,
    axes_for_over,
    component_factor_fields,
    sum_over,
)
from ._plans import (
    AntitheticDesign,
    FixedQuadraturePlan,
    IIDDesign,
    MonteCarloPlan,
    ProductIntegrationPlan,
    QuasiMonteCarloPlan,
    RandomizedQMCDesign,
    SparseGridPlan,
)
from ._precision import IntegrationPrecisionPolicy
from ._rules import (
    ClenshawCurtisRule,
    CubatureRule,
    GaussianCubatureRule,
    TanhSinhRule,
)
from ._sparse_grid import _smolyak_rule
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


class ProductIntegrationRealization(StrictModule):
    """One or more full product batches sharing a named-axis plan."""

    batches: tuple[PointIntegrationBatch, ...]
    factor_plans: tuple[Any, ...]
    stochastic_axes: tuple[str, ...] = eqx.field(static=True)
    deterministic_axes: tuple[str, ...] = eqx.field(static=True)
    randomized_qmc: bool = eqx.field(static=True)


def _unwrap(factor: Any, /) -> Any:
    return factor


def _fixed_field(factor: Any, selector: Any, /) -> cx.Field:
    factor = _unwrap(factor)
    if isinstance(factor, AbstractScalarDomain):
        if isinstance(selector, FixedStart):
            value = factor.fixed("start")
        elif isinstance(selector, FixedEnd):
            value = factor.fixed("end")
        elif isinstance(selector, Fixed):
            value = selector.value
        else:
            raise TypeError("Expected a fixed scalar selector.")
        return cx.Field(jnp.asarray(value, dtype=float).reshape(()), dims=())
    if isinstance(factor, AbstractGeometry) and isinstance(selector, Fixed):
        return cx.Field(
            jnp.asarray(selector.value, dtype=float).reshape((factor.spatial_dim,)),
            dims=(None,),
        )
    raise TypeError("Unsupported fixed product-plan factor.")


def _groups(
    component: Any,
    product_plan: ProductIntegrationPlan,
    /,
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    groups: list[tuple[tuple[str, ...], Any]] = []
    seen: set[str] = set()
    for key, factor_plan in product_plan.plans.items():
        labels = (key,) if isinstance(key, str) else tuple(key)
        if not labels:
            raise ValueError("Product plan axis groups cannot be empty.")
        unknown = tuple(label for label in labels if label not in component.domain.labels)
        if unknown:
            raise ValueError(f"Unknown product-plan labels {unknown!r}.")
        overlap = tuple(label for label in labels if label in seen)
        if overlap:
            raise ValueError(f"Product-plan labels occur more than once: {overlap!r}.")
        for label in labels:
            selector = component.spec.selection_for(label)
            if isinstance(selector, (FixedStart, FixedEnd, Fixed)):
                raise ValueError(
                    f"Product-plan label {label!r} must select a non-fixed component."
                )
            deterministic = isinstance(factor_plan, (FixedQuadraturePlan, SparseGridPlan))
            native_cubature = isinstance(factor_plan, FixedQuadraturePlan) and isinstance(
                factor_plan.rule, CubatureRule
            )
            valid_selector = (
                isinstance(selector, (Interior, Boundary))
                if native_cubature
                else isinstance(selector, Interior)
            )
            if deterministic and not valid_selector:
                expected = "Interior() or Boundary()" if native_cubature else "Interior()"
                raise ValueError(
                    f"Deterministic product-plan label {label!r} must select {expected}."
                )
        seen.update(labels)
        groups.append((labels, factor_plan))
    nonfixed = {
        label
        for label in component.domain.labels
        if not isinstance(
            component.spec.selection_for(label), (FixedStart, FixedEnd, Fixed)
        )
    }
    if seen != nonfixed:
        missing = tuple(sorted(nonfixed - seen))
        extra = tuple(sorted(seen - nonfixed))
        raise ValueError(
            "Product plans must cover every interior label and other non-fixed "
            f"labels exactly once; missing={missing!r}, extra={extra!r}."
        )
    return tuple(groups)


def _map_canonical(factor: Any, node: Array, /) -> tuple[Array, Array]:
    if isinstance(factor, ProbabilityDomain):
        unit = 0.5 * (node + 1.0)
        return factor.distribution.icdf(unit), jnp.asarray(0.5)
    if isinstance(factor, AbstractScalarDomain):
        lower = factor.fixed("start")
        upper = factor.fixed("end")
        return (
            0.5 * (upper - lower) * node + 0.5 * (upper + lower),
            jnp.asarray(0.5 * (upper - lower)),
        )
    raise TypeError("Product deterministic plans support scalar/probability factors.")


def _validate_sparse_factor(
    factor: Any,
    rule: str,
    label: str,
    /,
) -> None:
    if rule == "clenshaw-curtis":
        if (
            isinstance(factor, ProbabilityDomain)
            and factor.reference_transport.reference_measure != "uniform"
        ):
            raise ValueError(
                f"Clenshaw--Curtis sparse-grid axis {label!r} requires bounded "
                "probability support with a uniform reference transport."
            )
        return
    if rule != "gauss-hermite":
        raise ValueError(f"Unsupported sparse-grid axis rule {rule!r}.")
    if not isinstance(factor, ProbabilityDomain):
        raise TypeError(
            f"Gauss--Hermite sparse-grid axis {label!r} requires a probability factor."
        )
    if factor.reference_transport.reference_measure != "standard-normal":
        raise ValueError(
            f"Gauss--Hermite sparse-grid axis {label!r} requires a "
            "standard-normal reference transport."
        )


def _map_sparse_canonical(
    factor: Any,
    node: Array,
    rule: str,
    label: str,
    /,
) -> tuple[Array, Array]:
    _validate_sparse_factor(factor, rule, label)
    if rule == "gauss-hermite":
        return factor.reference_transport.from_reference(node), jnp.asarray(1.0)
    return _map_canonical(factor, node)


def _replicate_count(groups: tuple[tuple[tuple[str, ...], Any], ...], /) -> int:
    counts = {
        plan.design.num_replicates
        for _, plan in groups
        if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan))
        and isinstance(plan.design, RandomizedQMCDesign)
        and plan.design.num_replicates > 1
    }
    if len(counts) > 1:
        raise ValueError("Randomized-QMC product factors must use one replicate count.")
    return 1 if not counts else counts.pop()


def materialize_product(
    target: ComponentTarget | DensityTarget,
    plan: ProductIntegrationPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> ProductIntegrationRealization:
    """Materialize fixed/sparse and one stochastic named-axis product plan."""
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Product plans require a component target.")
    component = base.component
    if isinstance(component, ComponentSum):
        raise TypeError("Product-plan component unions must be integrated term by term.")
    groups = _groups(component, plan)
    stochastic_groups = tuple(
        (labels, factor_plan)
        for labels, factor_plan in groups
        if isinstance(factor_plan, (MonteCarloPlan, QuasiMonteCarloPlan))
    )
    unsupported = tuple(
        type(factor_plan).__name__
        for _, factor_plan in groups
        if not isinstance(
            factor_plan,
            (FixedQuadraturePlan, SparseGridPlan, MonteCarloPlan, QuasiMonteCarloPlan),
        )
    )
    if unsupported:
        raise TypeError(f"Unsupported product factor plans {unsupported!r}.")
    controlled_groups = tuple(
        labels
        for labels, factor_plan in groups
        if isinstance(factor_plan, (MonteCarloPlan, QuasiMonteCarloPlan))
        and factor_plan.control_variate is not None
    )
    if controlled_groups:
        raise ValueError(
            "Product integration does not support control variates; "
            "use a direct Monte Carlo or quasi-Monte Carlo plan."
        )
    fixed_labels = frozenset(
        label
        for label in component.domain.labels
        if isinstance(component.spec.selection_for(label), (FixedStart, FixedEnd, Fixed))
    )
    blocks: list[tuple[str, ...]] = []
    for labels, factor_plan in groups:
        if isinstance(factor_plan, FixedQuadraturePlan) and not isinstance(
            factor_plan.rule, GaussianCubatureRule
        ):
            blocks.extend((label,) for label in labels)
        else:
            blocks.append(labels)
    structure = SampleLayout(tuple(blocks)).canonicalize(
        component.domain.labels, fixed_labels=fixed_labels
    )
    reduction_axes = axes_for_over(structure, base.axes)
    stochastic_axes_list: list[str] = []
    for labels, _ in stochastic_groups:
        axis = structure.axis_for(labels[0])
        if axis is None:
            raise RuntimeError("Stochastic product factor has no axis.")
        if axis in reduction_axes:
            stochastic_axes_list.append(axis)
    stochastic_axes = tuple(stochastic_axes_list)
    integrated_stochastic = bool(stochastic_axes)
    replicas = _replicate_count(groups) if integrated_stochastic else 1
    batches: list[PointIntegrationBatch] = []
    deterministic_axes: list[str] = []
    factor_plans = tuple(factor_plan for _, factor_plan in groups)
    for replica in range(replicas):
        points: dict[str, Any] = {}
        weights_by_axis: dict[str, cx.Field] = {}
        for label in fixed_labels:
            points[label] = _fixed_field(
                component.domain.factor(label), component.spec.selection_for(label)
            )
        for group_index, (labels, factor_plan) in enumerate(groups):
            factors = tuple(_unwrap(component.domain.factor(label)) for label in labels)
            endpoint_factors = (
                tuple(
                    factor
                    for factor, rule in zip(
                        factors,
                        factor_plan.axis_rules,
                        strict=False,
                    )
                    if rule == "clenshaw-curtis"
                )
                if isinstance(factor_plan, SparseGridPlan)
                else factors
            )
            endpoint_inclusive = isinstance(factor_plan, SparseGridPlan) or (
                isinstance(factor_plan, FixedQuadraturePlan)
                and isinstance(factor_plan.rule, (ClenshawCurtisRule, TanhSinhRule))
            )
            if endpoint_inclusive and any(
                isinstance(factor, ProbabilityDomain)
                and getattr(factor.distribution, "support", None) is None
                for factor in endpoint_factors
            ):
                raise ValueError(
                    "Endpoint-inclusive product rules require bounded probability "
                    "support."
                )
            if isinstance(factor_plan, FixedQuadraturePlan):
                if isinstance(factor_plan.rule, GaussianCubatureRule):
                    rule = factor_plan.rule
                    if len(labels) != rule.dimension:
                        raise ValueError(
                            "Gaussian cubature dimension must match its product labels."
                        )
                    axis = structure.axis_for(labels[0])
                    if axis is None:
                        raise RuntimeError(
                            "Gaussian cubature product factor has no axis."
                        )
                    for column, (label, factor) in enumerate(
                        zip(labels, factors, strict=True)
                    ):
                        if not isinstance(factor, ProbabilityDomain):
                            raise TypeError(
                                "Gaussian cubature product factors must be probability domains."
                            )
                        transport = factor.reference_transport
                        if transport.reference_measure != "standard-normal":
                            raise ValueError(
                                "Gaussian cubature product factors require "
                                "standard-normal reference transports."
                            )
                        points[label] = cx.Field(
                            transport.from_reference(rule.prepared.points[:, column]),
                            dims=(axis,),
                        )
                    weights_by_axis[axis] = cx.Field(rule.prepared.weights, dims=(axis,))
                    if axis in reduction_axes and axis not in deterministic_axes:
                        deterministic_axes.append(axis)
                    continue
                if isinstance(factor_plan.rule, CubatureRule):
                    if len(labels) != 1 or not isinstance(factors[0], AbstractGeometry):
                        raise TypeError(
                            "Native cubature product factors require one geometry label."
                        )
                    label = labels[0]
                    axis = structure.axis_for(label)
                    if axis is None:
                        raise RuntimeError("Cubature product factor has no axis.")
                    mapped, weights = _cubature_factor_data(
                        factors[0],
                        component.spec.selection_for(label),
                        factor_plan.rule,
                    )
                    points[label] = cx.Field(mapped, dims=(axis, None))
                    weights_by_axis[axis] = cx.Field(weights, dims=(axis,))
                    if axis in reduction_axes and axis not in deterministic_axes:
                        deterministic_axes.append(axis)
                    continue
                for label, factor in zip(labels, factors, strict=True):
                    axis = structure.axis_for(label)
                    if axis is None:
                        raise RuntimeError("Fixed product factor has no axis.")
                    mapped, weights = _scalar_interior_rule_data(factor, factor_plan.rule)
                    points[label] = cx.Field(jnp.asarray(mapped), dims=(axis,))
                    weights_by_axis[axis] = cx.Field(jnp.asarray(weights), dims=(axis,))
                    if axis in reduction_axes and axis not in deterministic_axes:
                        deterministic_axes.append(axis)
                continue
            if isinstance(factor_plan, SparseGridPlan):
                if factor_plan.dimension != len(labels):
                    raise ValueError(
                        "Sparse-grid factor dimension must match its label group."
                    )
                for label, factor, rule in zip(
                    labels,
                    factors,
                    factor_plan.axis_rules,
                    strict=True,
                ):
                    _validate_sparse_factor(factor, rule, label)
                nodes, raw_weights = _smolyak_rule(
                    factor_plan.dimension,
                    factor_plan.level,
                    factor_plan.anisotropy,
                    factor_plan.axis_rules,
                )
                axis = structure.axis_for(labels[0])
                if axis is None:
                    raise RuntimeError("Sparse-grid product factor has no axis.")
                scale = jnp.asarray(1.0)
                for column, (label, factor, rule) in enumerate(
                    zip(
                        labels,
                        factors,
                        factor_plan.axis_rules,
                        strict=True,
                    )
                ):
                    mapped, local_scale = _map_sparse_canonical(
                        factor,
                        jnp.asarray(nodes[:, column]),
                        rule,
                        label,
                    )
                    points[label] = cx.Field(jnp.asarray(mapped), dims=(axis,))
                    scale = scale * local_scale
                weights_by_axis[axis] = cx.Field(
                    scale * jnp.asarray(raw_weights), dims=(axis,)
                )
                if axis in reduction_axes and axis not in deterministic_axes:
                    deterministic_axes.append(axis)
                continue
            design = factor_plan.design
            count = factor_plan.num_samples
            transports = tuple(
                reference_transport(factor, component.spec.selection_for(label))
                for label, factor in zip(labels, factors, strict=True)
            )
            unsupported_labels = tuple(
                label
                for label, transport in zip(labels, transports, strict=True)
                if transport is None
            )
            if unsupported_labels:
                raise TypeError(
                    "Product stochastic plans require exact target-measure reference "
                    f"transports; unsupported labels={unsupported_labels!r}."
                )
            reference_dimension = sum(
                transport.reference_dimension
                for transport in transports
                if transport is not None
            )
            base_design = design.base if isinstance(design, AntitheticDesign) else design
            name = design_name(base_design)
            address = SampleAddress(
                "integration",
                "product-group",
                algorithm_version=DESIGN_ALGORITHM_VERSION,
                target=labels,
                role=name,
            )
            design_key = derive_key(key, address, replica)
            if isinstance(design, AntitheticDesign):
                if count % 2:
                    raise ValueError(
                        "Antithetic product factors require even num_samples."
                    )
                pair_count = count // 2
                first = materialize_design(
                    base_design,
                    count=pair_count,
                    dimension=reference_dimension,
                    key=design_key,
                )
                second = (
                    1.0 - first
                    if design.involution is None
                    else jnp.asarray(design.involution(first), dtype=float)
                )
                if second.shape != first.shape:
                    raise ValueError(
                        "Antithetic involution must preserve the design shape."
                    )
                unit = jnp.concatenate((first, second), axis=0)
            else:
                unit = materialize_design(
                    design,
                    count=count,
                    dimension=reference_dimension,
                    key=design_key,
                )
            axis = structure.axis_for(labels[0])
            if axis is None:
                raise RuntimeError("Stochastic product factor has no axis.")
            offset = 0
            for label, transport in zip(labels, transports, strict=True):
                if transport is None:
                    raise RuntimeError("Validated reference transport is unavailable.")
                next_offset = offset + transport.reference_dimension
                mapped = transport.map(unit[:, offset:next_offset])

                def _sample_field(value):
                    array = jnp.asarray(value)
                    return cx.Field(
                        array,
                        dims=(axis,) + (None,) * (array.ndim - 1),
                    )

                points[label] = jax.tree_util.tree_map(_sample_field, mapped)
                offset = next_offset
            group_mass = _block_measure(component, labels)
            weights_by_axis[axis] = cx.Field(
                jnp.full((count,), group_mass / float(count)), dims=(axis,)
            )
        total_weight = cx.Field(jnp.asarray(1.0), dims=())
        axis_names = structure.axis_names
        if axis_names is None:
            raise RuntimeError("Product structure is not canonicalized.")
        for block, axis in zip(structure.blocks, axis_names, strict=True):
            if axis not in reduction_axes:
                continue
            total_weight = total_weight * weights_by_axis[axis]
        mass = total_weight
        for axis in reduction_axes:
            mass = sum_over(mass, axis)
        target_mass = jnp.asarray(mass.data)
        point_batch = PointBatch(
            frozendict({label: points[label] for label in component.domain.labels}),
            structure,
        )
        batches.append(
            PointIntegrationBatch(
                point_batch,
                total_weight,
                axes=reduction_axes,
                target_mass=target_mass,
                provenance="product",
            )
        )
    randomized_qmc = replicas > 1
    return ProductIntegrationRealization(
        tuple(batches),
        factor_plans,
        stochastic_axes=stochastic_axes,
        deterministic_axes=tuple(deterministic_axes),
        randomized_qmc=randomized_qmc,
    )


def _as_function(value: Any, component: Any, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


def _reduce_stochastic_product(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    batch: PointIntegrationBatch,
    stochastic_axis: str,
    design: Any,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
    precision: IntegrationPrecisionPolicy,
) -> tuple[cx.Field, Array | None, Array, int]:
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, ComponentTarget):
        raise TypeError("Product integration requires a component-backed target.")
    component = base.component
    if isinstance(component, ComponentSum):
        raise TypeError("Product component unions are unsupported.")
    function = _as_function(integrand, component)
    values = function(batch.points, key=key, **kwargs)
    if not isinstance(values, cx.Field):
        raise TypeError("Product integrands must evaluate to coordax.Field.")
    values = cx.Field(
        precision.evaluation(values.data),
        dims=values.dims,
    )
    mask, modifier = component_factor_fields(
        component, batch.points, key=key, kwargs=kwargs
    )
    base_weight = batch.weights * mask * modifier
    base_weight = cx.Field(
        precision.accumulation(base_weight.data),
        dims=base_weight.dims,
    )
    weight = base_weight
    if isinstance(target, DensityTarget):
        density_function = _as_function(target.log_density, component)
        log_density = density_function(batch.points, key=key, **kwargs)
        log_data = precision.evaluation(log_density.data)
        weight = weight * cx.Field(jnp.exp(log_data), dims=log_density.dims)
    weight = cx.Field(precision.accumulation(weight.data), dims=weight.dims)
    normalizer_weight = None
    if isinstance(target, DensityTarget):
        if target.normalized:
            normalizer_weight = weight
        elif base.normalized:
            normalizer_weight = base_weight
    elif target.normalized:
        normalizer_weight = base_weight
    numerator = weight * values
    for axis in batch.axes:
        if axis == stochastic_axis:
            continue
        numerator = sum_over(
            numerator,
            axis,
            accumulation_dtype=precision.accumulation_dtype,
        )
        if normalizer_weight is not None:
            normalizer_weight = sum_over(
                normalizer_weight,
                axis,
                accumulation_dtype=precision.accumulation_dtype,
            )
    sample_position = numerator.dims.index(stochastic_axis)
    numerator_samples = precision.accumulation(
        jnp.moveaxis(jnp.asarray(numerator.data), sample_position, 0)
    )
    numerator_total = precision.accumulation(jnp.sum(numerator_samples, axis=0))
    normalized = normalizer_weight is not None
    if normalizer_weight is not None:
        denominator_position = normalizer_weight.dims.index(stochastic_axis)
        denominator_samples = precision.accumulation(
            jnp.moveaxis(
                jnp.asarray(normalizer_weight.data),
                denominator_position,
                0,
            )
        )
        denominator_total = precision.accumulation(jnp.sum(denominator_samples, axis=0))
        value_data = numerator_total / denominator_total
    else:
        denominator_samples = None
        denominator_total = precision.accumulation(jnp.asarray(1.0))
        value_data = numerator_total
    count = numerator_samples.shape[0]
    if denominator_samples is not None:
        observations = (
            count
            * (
                numerator_samples
                - value_data
                * denominator_samples.reshape(
                    (count,) + (1,) * (numerator_samples.ndim - 1)
                )
            )
            / denominator_total
        )
    else:
        observations = count * numerator_samples
    centered = observations - jnp.mean(observations, axis=0)
    if (
        isinstance(design, AntitheticDesign)
        and isinstance(design.base, IIDDesign)
        and count // 2 >= 2
    ):
        pairs = count // 2
        pair_means = 0.5 * (observations[:pairs] + observations[pairs:])
        centered_pairs = pair_means - jnp.mean(pair_means, axis=0)
        variance = jnp.sum(
            jnp.real(centered_pairs * jnp.conj(centered_pairs)), axis=0
        ) / (pairs - 1)
        standard_error = precision.decision(jnp.sqrt(variance / pairs))
        error = precision.decision(jnp.max(standard_error))
    elif isinstance(design, IIDDesign):
        variance = jnp.sum(jnp.real(centered * jnp.conj(centered)), axis=0) / max(
            count - 1, 1
        )
        standard_error = precision.decision(jnp.sqrt(variance / count))
        error = precision.decision(jnp.max(standard_error))
    else:
        error = None
    valid_mass = jnp.all(jnp.isfinite(denominator_total)) & jnp.all(
        denominator_total != 0.0
    )
    status = jnp.where(
        valid_mass | (not normalized),
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
    )
    finite = jnp.all(jnp.isfinite(numerator_samples)) & (
        True
        if denominator_samples is None
        else jnp.all(jnp.isfinite(denominator_samples))
    )
    status = jnp.where(
        finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    output_dims = tuple(dim for dim in numerator.dims if dim != stochastic_axis)
    return cx.Field(value_data, dims=output_dims), error, status, count


def integrate_product(
    integrand: Any,
    target: ComponentTarget | DensityTarget,
    realization: ProductIntegrationRealization,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Reduce a deterministic or mixed deterministic/stochastic product plan."""
    callback_kwargs = {} if kwargs is None else kwargs
    precision_ = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    if not realization.stochastic_axes:
        if isinstance(target, DensityTarget):
            estimate = integrate_fixed_density(
                integrand,
                target,
                realization.batches[0],
                key=key,
                kwargs=callback_kwargs,
                precision=precision_,
            )
        else:
            estimate = integrate_fixed_component(
                integrand,
                target,
                realization.batches[0],
                key=key,
                kwargs=callback_kwargs,
                precision=precision_,
            )
        diagnostics = ProductIntegrationDiagnostics(
            status=estimate.status,
            num_evaluations=estimate.num_evaluations,
            error_estimate=None,
            factors=realization.factor_plans,
        )
        return IntegrationEstimate(
            estimate.value,
            status=estimate.status,
            num_evaluations=estimate.num_evaluations,
            error_estimate=None,
            error_kind=None,
            diagnostics=diagnostics,
            provenance=IntegrationProvenance("product", "component", "deterministic"),
        )
    if len(realization.stochastic_axes) > 1:
        estimates = tuple(
            (
                integrate_fixed_density(
                    integrand,
                    target,
                    batch,
                    key=jr.fold_in(key, index),
                    kwargs=callback_kwargs,
                    precision=precision_,
                )
                if isinstance(target, DensityTarget)
                else integrate_fixed_component(
                    integrand,
                    target,
                    batch,
                    key=jr.fold_in(key, index),
                    kwargs=callback_kwargs,
                    precision=precision_,
                )
            )
            for index, batch in enumerate(realization.batches)
        )
        values = precision_.accumulation(
            jnp.stack(tuple(jnp.asarray(estimate.value.data) for estimate in estimates))
        )
        if len(estimates) > 1:
            value_data = jnp.mean(values, axis=0)
            error = precision_.decision(
                jnp.max(jnp.std(values, axis=0, ddof=1) / jnp.sqrt(len(estimates)))
            )
            error_kind = "randomized-qmc-replicate-error"
        else:
            value_data = estimates[0].value.data
            error = None
            error_kind = None
        status = jnp.max(
            jnp.stack(tuple(jnp.asarray(estimate.status) for estimate in estimates))
        )
        evaluations = sum(int(batch.weights.data.size) for batch in realization.batches)
        diagnostics = ProductIntegrationDiagnostics(
            status=status,
            num_evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            error_estimate=error,
            factors=realization.factor_plans,
        )
        return IntegrationEstimate(
            cx.Field(value_data, dims=estimates[0].value.dims),
            status=status,
            num_evaluations=evaluations,
            error_estimate=error,
            error_kind=error_kind,
            diagnostics=diagnostics,
            provenance=IntegrationProvenance("product", "component", "mixed"),
        )

    stochastic_axis = realization.stochastic_axes[0]
    stochastic_plan = next(
        factor_plan
        for factor_plan in realization.factor_plans
        if isinstance(factor_plan, (MonteCarloPlan, QuasiMonteCarloPlan))
    )
    design = stochastic_plan.design
    reductions = tuple(
        _reduce_stochastic_product(
            integrand,
            target,
            batch,
            stochastic_axis,
            design,
            key=jr.fold_in(key, index),
            kwargs=callback_kwargs,
            precision=precision_,
        )
        for index, batch in enumerate(realization.batches)
    )
    values = precision_.accumulation(
        jnp.stack(tuple(jnp.asarray(item[0].data) for item in reductions))
    )
    if len(reductions) > 1:
        value_data = jnp.mean(values, axis=0)
        error = precision_.decision(
            jnp.max(jnp.std(values, axis=0, ddof=1) / jnp.sqrt(len(reductions)))
        )
        error_kind = "randomized-qmc-replicate-error"
    else:
        value_data = reductions[0][0].data
        error = reductions[0][1]
        if error is not None and isinstance(design, AntitheticDesign):
            error_kind = "antithetic-pair-standard-error"
        elif error is not None and isinstance(design, IIDDesign):
            error_kind = "iid-standard-error"
        else:
            error_kind = None
    status = jnp.max(jnp.stack(tuple(item[2] for item in reductions)))
    evaluations = sum(int(batch.weights.data.size) for batch in realization.batches)
    diagnostics = ProductIntegrationDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
        error_estimate=error,
        factors=realization.factor_plans,
    )
    return IntegrationEstimate(
        cx.Field(value_data, dims=reductions[0][0].dims),
        status=status,
        num_evaluations=evaluations,
        error_estimate=error,
        error_kind=error_kind,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("product", "component", "mixed"),
    )


__all__ = [
    "ProductIntegrationRealization",
    "integrate_product",
    "materialize_product",
]
