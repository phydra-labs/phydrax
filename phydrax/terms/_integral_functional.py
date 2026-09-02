#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._term import AbstractSamplingTerm
from ..integration import (
    AdaptiveIntegration,
    CallerIntegration,
    FixedIntegration,
    IntegrationRealization,
    IntegrationSource,
    IntegrationStatus,
    PerStepIntegration,
    reduce,
    resolve_integration,
)
from ..integration._adaptive_signed import AdaptiveSignedEstimator


class IntegralFunctional(AbstractSamplingTerm):
    """A raw signed scalar objective executed by any integration target and plan."""

    objective_vars: tuple[str, ...]
    source: IntegrationSource
    weight: float = eqx.field(static=True)
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction] | DomainFunction
    label: str | None = eqx.field(static=True)
    nonfinite_integrand: Literal["raise", "propagate"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: IntegrationSource,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        nonfinite_integrand: Literal["raise", "propagate"] = "raise",
    ):
        if not isinstance(integrand, DomainFunction) and not callable(integrand):
            raise TypeError("integrand must be a DomainFunction or callable.")
        if not isinstance(
            source,
            (
                PerStepIntegration,
                FixedIntegration,
                CallerIntegration,
                AdaptiveIntegration,
            ),
        ):
            raise TypeError("source must be an IntegrationSource.")
        if isinstance(source, AdaptiveIntegration):
            if not isinstance(source.policy, AdaptiveSignedEstimator):
                raise TypeError(
                    "IntegralFunctional AdaptiveIntegration requires an "
                    "AdaptiveSignedEstimator policy."
                )
        nonfinite_policy = str(nonfinite_integrand).lower()
        if nonfinite_policy not in ("raise", "propagate"):
            raise ValueError("nonfinite_integrand must be 'raise' or 'propagate'.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.source = source
        self.integrand = integrand
        weight_value = float(weight)
        if not math.isfinite(weight_value):
            raise ValueError("weight must be finite.")
        self.weight = weight_value
        self.label = None if label is None else str(label)
        self.nonfinite_integrand = nonfinite_policy

    @classmethod
    def from_operator(
        cls,
        *,
        source: IntegrationSource,
        operator: Callable[..., DomainFunction],
        objective_vars: str | Sequence[str],
        weight: ArrayLike = 1.0,
        label: str | None = None,
        nonfinite_integrand: Literal["raise", "propagate"] = "raise",
    ) -> "IntegralFunctional":
        """Build an integral functional from an operator on named solver fields."""
        variables = (
            (objective_vars,)
            if isinstance(objective_vars, str)
            else tuple(objective_vars)
        )

        def integrand(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in variables))

        return cls(
            source=source,
            integrand=integrand,
            objective_vars=variables,
            weight=weight,
            label=label,
            nonfinite_integrand=nonfinite_integrand,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> IntegrationRealization | None:
        """Resolve one realization according to the typed integration source."""
        if isinstance(self.source, CallerIntegration):
            return None
        return resolve_integration(self.source, key=key)

    def _integrand_function(
        self, functions: Mapping[str, DomainFunction], /
    ) -> DomainFunction:
        value = self.integrand
        integrand = value if isinstance(value, DomainFunction) else value(functions)
        if not isinstance(integrand, DomainFunction):
            raise TypeError(
                "IntegralFunctional integrand must produce a DomainFunction; "
                f"got {type(integrand).__name__}."
            )
        return integrand

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> Array:
        """Execute and return the raw signed scalar integral."""
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        realization = batch
        if realization is None:
            if isinstance(self.source, CallerIntegration):
                raise ValueError(
                    "Caller-managed IntegralFunctional requires "
                    "batch=IntegrationRealization."
                )
            realization = self.sample(key=key)
        elif isinstance(self.source, CallerIntegration):
            realization = resolve_integration(
                self.source,
                key=key,
                realization=realization,
            )
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("IntegralFunctional batch must be an IntegrationRealization.")
        estimate = reduce(self._integrand_function(functions), realization, **kwargs)
        if estimate.value.dims != ():
            raise ValueError(
                "IntegralFunctional must reduce to a scalar Field, "
                f"got dims={estimate.value.dims}."
            )
        value = jnp.asarray(estimate.value.data).reshape(())
        if jnp.iscomplexobj(value):
            raise TypeError(
                "IntegralFunctional requires a real scalar integrand; "
                "use real_part(...) to select an explicitly real objective."
            )
        nonfinite = estimate.status == int(IntegrationStatus.NONFINITE_INTEGRAND)
        if self.nonfinite_integrand == "propagate":
            failed = (estimate.status != int(IntegrationStatus.CONVERGED)) & ~nonfinite
            value = eqx.error_if(
                value,
                failed,
                "IntegralFunctional integration did not converge.",
            )
            value = jnp.where(nonfinite, jnp.nan, value)
        else:
            value = eqx.error_if(
                value,
                estimate.status != int(IntegrationStatus.CONVERGED),
                "IntegralFunctional integration did not converge.",
            )
        return self.weight * value


__all__ = ["IntegralFunctional"]
