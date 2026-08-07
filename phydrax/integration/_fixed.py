#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.domain import ComponentSum, DomainComponent, DomainFunction

from .._doc import DOC_KEY0
from ._batches import PointIntegrationBatch, SeparableIntegrationBatch
from ._estimates import (
    FixedQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._lowering import _component_base_mass, component_factor_fields, sum_over
from ._status import IntegrationStatus
from ._targets import ComponentTarget, DensityTarget


def _batch_weight(
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    /,
) -> cx.Field:
    if isinstance(batch, PointIntegrationBatch):
        weight = batch.weights
        if batch.mask is not None:
            weight = weight * batch.mask
        return weight
    weight = batch.total_weight()
    if batch.mask is not None:
        weight = weight * batch.mask
    return weight


def _num_evaluations(
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    /,
) -> int:
    if isinstance(batch, PointIntegrationBatch):
        size = 1
        for axis in batch.axes:
            size *= int(batch.weights.named_shape[axis])
        return size
    size = 1
    for axis in batch.axes:
        size *= int(batch.weights_by_axis[axis].named_shape[axis])
    return size


def _as_domain_function(value: Any, component: DomainComponent, /) -> DomainFunction:
    if isinstance(value, DomainFunction):
        return value
    return DomainFunction(domain=component.domain, deps=(), func=value)


def _require_fixed_batch(
    batch: object, /
) -> PointIntegrationBatch | SeparableIntegrationBatch:
    if not isinstance(batch, (PointIntegrationBatch, SeparableIntegrationBatch)):
        raise TypeError("Expected a fixed point or separable integration batch.")
    return batch


def _component_moments(
    integrand: Any,
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    component: DomainComponent,
    /,
    *,
    log_density: Any | None,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[cx.Field, cx.Field, cx.Field, Array]:
    points = batch.points
    function = _as_domain_function(integrand, component)
    values = function(points, key=key, **kwargs)
    if not isinstance(values, cx.Field):
        raise TypeError("An integration DomainFunction must evaluate to coordax.Field.")
    mask, modifier = component_factor_fields(
        component,
        points,
        key=key,
        kwargs=kwargs,
    )
    base_weight = _batch_weight(batch) * mask * modifier
    finite_inputs = jnp.all(jnp.isfinite(jnp.asarray(values.data)))
    weight = base_weight
    if log_density is not None:
        log_density_function = _as_domain_function(log_density, component)
        log_values = log_density_function(points, key=key, **kwargs)
        if not isinstance(log_values, cx.Field):
            raise TypeError("log_density must evaluate to coordax.Field.")
        density_values = jnp.exp(jnp.asarray(log_values.data))
        finite_inputs = finite_inputs & jnp.all(jnp.isfinite(density_values))
        weight = weight * cx.Field(density_values, dims=log_values.dims)
    numerator = weight * values
    denominator = weight
    base_denominator = base_weight
    for axis in batch.axes:
        numerator = sum_over(numerator, axis)
        denominator = sum_over(denominator, axis)
        base_denominator = sum_over(base_denominator, axis)
    return numerator, denominator, base_denominator, finite_inputs


def _finish_fixed(
    numerator: cx.Field,
    denominator: cx.Field,
    /,
    *,
    normalized: bool,
    num_evaluations: int,
    target_mass: Array | None,
    provenance: str,
    base_normalization_mass: cx.Field | None = None,
    finite_inputs: Array | None = None,
) -> IntegrationEstimate:
    numerator_data = jnp.asarray(numerator.data)
    denominator_data = jnp.asarray(denominator.data)
    valid_denominator = jnp.all(jnp.isfinite(denominator_data)) & jnp.all(
        denominator_data != 0
    )
    finite_numerator = jnp.all(jnp.isfinite(numerator_data))
    if base_normalization_mass is not None:
        normalization_data = jnp.asarray(base_normalization_mass.data)
        valid_normalization = jnp.all(jnp.isfinite(normalization_data)) & jnp.all(
            normalization_data != 0
        )
        value = numerator / base_normalization_mass
        data = jnp.asarray(value.data)
        finite_value = jnp.all(jnp.isfinite(data))
        genuine_inputs_finite = (
            finite_numerator if finite_inputs is None else finite_inputs
        )
        status = jnp.where(
            ~genuine_inputs_finite,
            int(IntegrationStatus.NONFINITE_INTEGRAND),
            jnp.where(
                ~valid_normalization,
                int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
                jnp.where(
                    finite_numerator & finite_value,
                    int(IntegrationStatus.CONVERGED),
                    int(IntegrationStatus.NONFINITE_INTEGRAND),
                ),
            ),
        )
    elif normalized:
        data = numerator_data / denominator_data
        value = cx.Field(data, dims=numerator.dims)
        status = jnp.where(
            valid_denominator,
            int(IntegrationStatus.CONVERGED),
            int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
        )
        status = jnp.where(
            finite_numerator | (~valid_denominator),
            status,
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        )
    else:
        value = numerator
        status = jnp.where(
            finite_numerator,
            int(IntegrationStatus.CONVERGED),
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        )
    diagnostics = FixedQuadratureDiagnostics(
        status=jnp.asarray(status, dtype=jnp.int32),
        num_evaluations=jnp.asarray(num_evaluations, dtype=jnp.int32),
        target_mass=target_mass,
        rule=provenance,
    )
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=num_evaluations,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("fixed", "component", provenance),
    )


def integrate_fixed_component(
    integrand: Any,
    target: ComponentTarget,
    batch: PointIntegrationBatch
    | SeparableIntegrationBatch
    | tuple[PointIntegrationBatch | SeparableIntegrationBatch, ...],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce a component target over a typed fixed or sampled batch."""
    callback_kwargs = {} if kwargs is None else kwargs
    if isinstance(target.component, ComponentSum):
        if not isinstance(batch, tuple) or len(batch) != len(target.component.terms):
            raise ValueError(
                "Union targets require one aligned integration batch per term."
            )
        keys = jr.split(key, len(target.component.terms))
        numerators: list[cx.Field] = []
        denominators: list[cx.Field] = []
        evaluations = 0
        for component, term_batch, term_key in zip(
            target.component.terms, batch, keys, strict=True
        ):
            term_batch = _require_fixed_batch(term_batch)
            numerator, denominator, _, _ = _component_moments(
                integrand,
                term_batch,
                component,
                log_density=None,
                key=term_key,
                kwargs=callback_kwargs,
            )
            numerators.append(numerator)
            denominators.append(denominator)
            evaluations += _num_evaluations(term_batch)
        numerator = numerators[0]
        denominator = denominators[0]
        for term_numerator, term_denominator in zip(
            numerators[1:], denominators[1:], strict=True
        ):
            numerator = numerator + term_numerator
            denominator = denominator + term_denominator
        return _finish_fixed(
            numerator,
            denominator,
            normalized=target.normalized,
            num_evaluations=evaluations,
            target_mass=_component_base_mass(target.component),
            provenance="component-sum",
        )
    if isinstance(batch, tuple):
        raise TypeError("A single component target requires one integration batch.")
    numerator, denominator, _, _ = _component_moments(
        integrand,
        batch,
        target.component,
        log_density=None,
        key=key,
        kwargs=callback_kwargs,
    )
    return _finish_fixed(
        numerator,
        denominator,
        normalized=target.normalized,
        num_evaluations=_num_evaluations(batch),
        target_mass=batch.target_mass,
        provenance=batch.provenance,
    )


def integrate_fixed_density(
    integrand: Any,
    target: DensityTarget,
    batch: PointIntegrationBatch
    | SeparableIntegrationBatch
    | tuple[PointIntegrationBatch | SeparableIntegrationBatch, ...],
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Reduce a deterministic density target relative to a component measure."""
    if not isinstance(target.base, ComponentTarget):
        raise TypeError("Fixed density integration requires a ComponentTarget base.")
    callback_kwargs = {} if kwargs is None else kwargs
    if isinstance(target.base.component, ComponentSum):
        if not isinstance(batch, tuple) or len(batch) != len(target.base.component.terms):
            raise ValueError(
                "Density union targets require one aligned integration batch per term."
            )
        keys = jr.split(key, len(target.base.component.terms))
        numerators: list[cx.Field] = []
        denominators: list[cx.Field] = []
        base_denominators: list[cx.Field] = []
        finite_inputs = jnp.asarray(True)
        evaluations = 0
        for component, term_batch, term_key in zip(
            target.base.component.terms, batch, keys, strict=True
        ):
            term_batch = _require_fixed_batch(term_batch)
            numerator, denominator, base_denominator, term_finite_inputs = (
                _component_moments(
                    integrand,
                    term_batch,
                    component,
                    log_density=target.log_density,
                    key=term_key,
                    kwargs=callback_kwargs,
                )
            )
            numerators.append(numerator)
            denominators.append(denominator)
            base_denominators.append(base_denominator)
            finite_inputs = finite_inputs & term_finite_inputs
            evaluations += _num_evaluations(term_batch)
        numerator = numerators[0]
        denominator = denominators[0]
        base_denominator = base_denominators[0]
        for term_numerator, term_denominator, term_base_denominator in zip(
            numerators[1:],
            denominators[1:],
            base_denominators[1:],
            strict=True,
        ):
            numerator = numerator + term_numerator
            denominator = denominator + term_denominator
            base_denominator = base_denominator + term_base_denominator
        base_normalization_mass = None
        if target.base.normalized and not target.normalized:
            base_normalization_mass = base_denominator
        return _finish_fixed(
            numerator,
            denominator,
            normalized=target.normalized,
            num_evaluations=evaluations,
            target_mass=_component_base_mass(target.base.component),
            provenance="density:component-sum",
            base_normalization_mass=base_normalization_mass,
            finite_inputs=finite_inputs,
        )
    if isinstance(batch, tuple):
        raise TypeError("A single density component target requires one batch.")
    numerator, denominator, base_denominator, finite_inputs = _component_moments(
        integrand,
        batch,
        target.base.component,
        log_density=target.log_density,
        key=key,
        kwargs=callback_kwargs,
    )
    base_normalization_mass = None
    if target.base.normalized and not target.normalized:
        base_normalization_mass = base_denominator
    return _finish_fixed(
        numerator,
        denominator,
        normalized=target.normalized,
        num_evaluations=_num_evaluations(batch),
        target_mass=batch.target_mass,
        provenance=f"density:{batch.provenance}",
        base_normalization_mass=base_normalization_mass,
        finite_inputs=finite_inputs,
    )


__all__ = ["integrate_fixed_component", "integrate_fixed_density"]
