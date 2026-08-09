#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.conditions import AbstractMomentCondition
from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._term import AbstractEvaluatedScalarTerm, TermEvaluation
from ..integration import (
    AdaptiveIntegration,
    CallerIntegration,
    FixedIntegration,
    IntegrationRealization,
    IntegrationSource,
    PerStepIntegration,
    reduce,
)
from ._integrated import (
    checked_estimate_field,
    resolve_term_realization,
    validate_condition_source,
)


_SOURCE_TYPES = (
    PerStepIntegration,
    FixedIntegration,
    CallerIntegration,
)


class MomentPenalty(AbstractEvaluatedScalarTerm):
    """Squared mismatch between an integrated field moment and its target."""

    condition: AbstractMomentCondition
    source: IntegrationSource
    fields: tuple[str, ...] = eqx.field(static=True)
    scale: Array
    weight: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        condition: AbstractMomentCondition,
        source: IntegrationSource,
        /,
        *,
        scale: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(condition, AbstractMomentCondition):
            raise TypeError("MomentPenalty requires an AbstractMomentCondition.")
        if isinstance(source, AdaptiveIntegration):
            raise TypeError(
                "MomentPenalty does not support AdaptiveIntegration; "
                "solver-managed adaptive collocation requires ResidualPenalty."
            )
        if not isinstance(source, _SOURCE_TYPES):
            raise TypeError("MomentPenalty requires a typed IntegrationSource.")
        validate_condition_source(condition.on, source)
        coefficient = jnp.asarray(scale, dtype=float)
        if coefficient.shape != ():
            raise ValueError("Term scale must be a scalar.")
        if not bool(jnp.isfinite(coefficient)) or float(coefficient) < 0.0:
            raise ValueError("Term scale must be finite and nonnegative.")
        self.condition = condition
        self.source = source
        self.fields = condition.fields
        self.scale = coefficient.reshape(())
        self.weight = self.scale
        self.label = condition.label if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> TermEvaluation:
        resolved = resolve_term_realization(
            self.source,
            key=key,
            realization=realization,
        )
        runtime_kwargs = dict(kwargs)
        if iter_ is not None:
            runtime_kwargs["iter_"] = iter_
        estimate = reduce(
            self.condition.integrand(functions),
            resolved,
            **runtime_kwargs,
        )
        field = checked_estimate_field(estimate)
        named_dims = tuple(dim for dim in field.dims if dim is not None)
        if named_dims:
            raise ValueError(
                "MomentPenalty integration left sampling dimensions "
                f"{named_dims!r}; the source must integrate them all."
            )
        integrated = jnp.asarray(field.data)
        target = jnp.asarray(self.condition.target)
        if jnp.broadcast_shapes(integrated.shape, target.shape) != integrated.shape:
            raise ValueError(
                f"Moment target shape {target.shape} cannot broadcast to "
                f"integrated shape {integrated.shape}."
            )
        difference = integrated - target
        mismatch = jnp.sum(jnp.real(jnp.conj(difference) * difference))
        value = self.scale * jnp.asarray(mismatch, dtype=float).reshape(())
        return TermEvaluation(value, diagnostics=estimate)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        realization: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(
            functions,
            key=key,
            iter_=iter_,
            realization=realization,
            **kwargs,
        ).value


__all__ = ["MomentPenalty"]
