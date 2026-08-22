#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from ._batches import MappedIntegrationBatch
from ._estimates import (
    IntegrationEstimate,
    IntegrationProvenance,
    MappedIntegrationDiagnostics,
)
from ._plans import CellQuadraturePlan, FixedQuadraturePlan
from ._rules import CubatureRule, reference_rule_data
from ._status import IntegrationStatus
from ._targets import DensityTarget, MappedTarget


def materialize_mapped(
    target: MappedTarget,
    plan: CellQuadraturePlan | FixedQuadraturePlan,
    /,
) -> MappedIntegrationBatch:
    """Apply a reference rule and supplied physical map/Jacobian."""
    rule = target.reference_rule
    if isinstance(plan, CellQuadraturePlan):
        rule = plan.rule
    data = reference_rule_data(rule)
    reference_points = data.points
    points = target.mapping(reference_points)
    jacobian = jnp.asarray(target.jacobian(reference_points), dtype=float).reshape((-1,))
    if jacobian.shape != data.weights.shape:
        raise ValueError("Mapped Jacobian must return one measure scale per point.")
    if target.mask is None:
        mask = jnp.ones(data.weights.shape, dtype=bool)
    elif callable(target.mask):
        mask = jnp.asarray(target.mask(reference_points), dtype=bool).reshape((-1,))
    else:
        mask = jnp.asarray(target.mask, dtype=bool).reshape((-1,))
    if mask.shape != data.weights.shape:
        raise ValueError("Mapped target mask must have one entry per reference point.")
    raw_weights = data.weights * jnp.abs(jacobian)
    represented_mass = jnp.sum(jnp.where(mask, raw_weights, 0.0))
    if target.target_mass is None:
        weights = raw_weights
        target_mass = represented_mass
    else:
        target_mass = jnp.asarray(target.target_mass, dtype=float).reshape(())
        scale = jnp.where(
            represented_mass != 0.0,
            target_mass / represented_mass,
            jnp.asarray(jnp.nan),
        )
        weights = raw_weights * scale
    provenance = (
        rule.rule_id
        if isinstance(rule, CubatureRule)
        else f"{data.cell}:{type(rule).__name__}"
    )
    return MappedIntegrationBatch(
        reference_points,
        points,
        weights,
        mask=mask,
        target_mass=target_mass,
        cell=data.cell,
        provenance=provenance,
    )


def _mapped_values(
    value: Any,
    batch: MappedIntegrationBatch,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> tuple[Array, tuple[Any, ...]]:
    if isinstance(value, DomainFunction):
        result = value(batch.points, key=key, **kwargs)
    elif callable(value):
        function = _ensure_special_kwonly_args(value)
        result = function(batch.points, key=key, **kwargs)
    else:
        result = value
    if isinstance(result, cx.Field):
        if batch.axis not in result.named_dims:
            raise ValueError(
                f"Mapped callback output is missing point axis {batch.axis!r}."
            )
        position = result.dims.index(batch.axis)
        values = jnp.moveaxis(jnp.asarray(result.data), position, 0)
        output_dims = tuple(dim for dim in result.dims if dim != batch.axis)
    else:
        values = jnp.asarray(result)
        output_dims = (None,) * max(values.ndim - 1, 0)
    count = int(batch.weights.shape[0])
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (count,))
        output_dims = ()
    if values.shape[0] != count:
        raise ValueError("Mapped callback values must have a leading point axis.")
    return values, output_dims


def integrate_mapped(
    integrand: Any,
    target: MappedTarget | DensityTarget,
    batch: MappedIntegrationBatch,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
) -> IntegrationEstimate:
    """Evaluate and reduce a mapped reference-cell target."""
    callback_kwargs = {} if kwargs is None else kwargs
    base = target.base if isinstance(target, DensityTarget) else target
    if not isinstance(base, MappedTarget):
        raise TypeError("Mapped integration requires a mapped target base.")
    values, output_dims = _mapped_values(
        integrand, batch, key=key, kwargs=callback_kwargs
    )
    weights = jnp.where(batch.mask, batch.weights, 0.0)
    if isinstance(target, DensityTarget):
        log_values, log_dims = _mapped_values(
            target.log_density, batch, key=key, kwargs=callback_kwargs
        )
        if log_dims or log_values.ndim != 1:
            raise ValueError("Mapped log density must be scalar-valued per point.")
        weights = weights * jnp.exp(log_values)
        normalized = target.normalized
    else:
        normalized = False
    count = int(batch.weights.shape[0])
    expanded = jnp.reshape(weights, (count,) + (1,) * (values.ndim - 1))
    numerator = jnp.sum(expanded * values, axis=0)
    mass = jnp.sum(weights)
    finite_operands = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights))
    if normalized:
        value = numerator / mass
        zero_mass = jnp.isfinite(mass) & (mass == 0.0)
    else:
        value = numerator
        zero_mass = jnp.asarray(False)
    finite_reduction = jnp.all(jnp.isfinite(numerator)) & jnp.all(jnp.isfinite(value))
    status = jnp.where(
        finite_operands & zero_mass,
        int(IntegrationStatus.INVALID_NORMALIZATION_MASS),
        jnp.where(
            finite_operands & finite_reduction,
            int(IntegrationStatus.CONVERGED),
            int(IntegrationStatus.NONFINITE_INTEGRAND),
        ),
    )
    diagnostics = MappedIntegrationDiagnostics(
        status=jnp.asarray(status, dtype=jnp.int32),
        num_evaluations=jnp.asarray(count, dtype=jnp.int32),
        target_mass=batch.target_mass,
        num_active_points=jnp.sum(batch.mask, dtype=jnp.int32),
        cell=batch.cell,
    )
    return IntegrationEstimate(
        cx.Field(value, dims=output_dims),
        status=status,
        num_evaluations=count,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("mapped-fixed", "mapped", batch.provenance),
    )


__all__ = ["integrate_mapped", "materialize_mapped"]
