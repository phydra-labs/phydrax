#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._objective import AbstractSamplingObjectiveTerm
from ..domain._function import DomainFunction
from ..integration import (
    IntegrationRealization,
    IntegrationStatus,
    materialize,
    reduce,
)
from ..integration._api import _requires_random_key


class IntegralFunctional(AbstractSamplingObjectiveTerm):
    """A raw signed scalar objective executed by any integration target and plan."""

    objective_vars: tuple[str, ...]
    target: Any
    plan: Any
    weight: Array
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction] | DomainFunction
    fixed_realization: IntegrationRealization | None
    label: str | None = eqx.field(static=True)
    materialization_policy: Literal["fixed", "per_step", "caller"] = eqx.field(
        static=True
    )

    def __init__(
        self,
        *,
        target: Any,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        plan: Any = None,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        materialization_policy: Literal["fixed", "per_step", "caller"] = "per_step",
        fixed_realization: IntegrationRealization | None = None,
        fixed_key: Key[Array, ""] | None = None,
    ):
        if not isinstance(integrand, DomainFunction) and not callable(integrand):
            raise TypeError("integrand must be a DomainFunction or callable.")
        policy = str(materialization_policy).lower()
        if policy not in ("fixed", "per_step", "caller"):
            raise ValueError(
                "materialization_policy must be 'fixed', 'per_step', or 'caller'."
            )
        if policy == "fixed":
            if fixed_realization is None:
                if _requires_random_key(plan):
                    if fixed_key is None:
                        raise ValueError(
                            "A randomized fixed IntegralFunctional requires fixed_key=."
                        )
                    fixed_realization = materialize(target, plan, key=fixed_key)
                else:
                    if fixed_key is not None:
                        raise ValueError(
                            "A deterministic fixed IntegralFunctional does not consume "
                            "fixed_key=."
                        )
                    fixed_realization = materialize(target, plan)
        elif fixed_realization is not None or fixed_key is not None:
            raise ValueError(
                "fixed_realization/fixed_key require materialization_policy='fixed'."
            )
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.target = target
        self.plan = plan
        self.integrand = integrand
        self.weight = jnp.asarray(weight, dtype=float).reshape(())
        self.label = None if label is None else str(label)
        self.materialization_policy = policy
        self.fixed_realization = fixed_realization

    @classmethod
    def from_operator(
        cls,
        *,
        target: Any,
        operator: Callable[..., DomainFunction],
        objective_vars: str | Sequence[str],
        plan: Any = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        materialization_policy: Literal["fixed", "per_step", "caller"] = "per_step",
        fixed_realization: IntegrationRealization | None = None,
        fixed_key: Key[Array, ""] | None = None,
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
            target=target,
            plan=plan,
            integrand=integrand,
            objective_vars=variables,
            weight=weight,
            label=label,
            materialization_policy=materialization_policy,
            fixed_realization=fixed_realization,
            fixed_key=fixed_key,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> IntegrationRealization | None:
        """Materialize according to the objective's refresh policy."""
        if self.materialization_policy == "fixed":
            if self.fixed_realization is None:
                raise RuntimeError("Fixed IntegralFunctional has no realization.")
            return self.fixed_realization
        if self.materialization_policy == "caller":
            return None
        if _requires_random_key(self.plan):
            return materialize(self.target, self.plan, key=key)
        return materialize(self.target, self.plan)

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
        realization = batch
        if realization is None:
            if self.materialization_policy == "caller":
                raise ValueError(
                    "Caller-managed IntegralFunctional requires batch=IntegrationRealization."
                )
            realization = self.sample(key=key)
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
        value = eqx.error_if(
            value,
            estimate.status != int(IntegrationStatus.CONVERGED),
            "IntegralFunctional integration did not converge.",
        )
        return self.weight * value


__all__ = ["IntegralFunctional"]
