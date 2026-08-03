#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._objective import AbstractObjectiveTerm
from ..domain._components import DomainComponent
from ..domain._function import DomainFunction
from ..operators.integral._adaptive import (
    adaptive_integral,
    AdaptiveQuadratureConfig,
)


class AdaptiveIntegralFunctional(AbstractObjectiveTerm):
    """A raw signed objective evaluated by one-dimensional adaptive quadrature."""

    objective_vars: tuple[str, ...]
    component: DomainComponent
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction] | DomainFunction
    quadrature: AdaptiveQuadratureConfig
    weight: Array
    variable: str | None
    label: str | None

    def __init__(
        self,
        *,
        component: DomainComponent,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        objective_vars: Sequence[str] | None = None,
        quadrature: AdaptiveQuadratureConfig | None = None,
        weight: ArrayLike = 1.0,
        variable: str | None = None,
        label: str | None = None,
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("component must be a DomainComponent.")
        if not isinstance(integrand, DomainFunction) and not callable(integrand):
            raise TypeError("integrand must be a DomainFunction or callable.")
        if quadrature is not None and not isinstance(
            quadrature, AdaptiveQuadratureConfig
        ):
            raise TypeError("quadrature must be an AdaptiveQuadratureConfig or None.")

        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.component = component
        self.integrand = integrand
        self.quadrature = AdaptiveQuadratureConfig() if quadrature is None else quadrature
        self.weight = jnp.asarray(weight, dtype=float).reshape(())
        self.variable = None if variable is None else str(variable)
        self.label = None if label is None else str(label)

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent,
        operator: Callable[..., DomainFunction],
        objective_vars: str | Sequence[str],
        quadrature: AdaptiveQuadratureConfig | None = None,
        weight: ArrayLike = 1.0,
        variable: str | None = None,
        label: str | None = None,
    ) -> "AdaptiveIntegralFunctional":
        """Build an adaptive functional from an operator on named solver fields."""
        vars_tuple = (
            (objective_vars,)
            if isinstance(objective_vars, str)
            else tuple(objective_vars)
        )

        def integrand(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in vars_tuple))

        return cls(
            component=component,
            integrand=integrand,
            objective_vars=vars_tuple,
            quadrature=quadrature,
            weight=weight,
            variable=variable,
            label=label,
        )

    def _integrand_function(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        value = self.integrand
        integrand = value if isinstance(value, DomainFunction) else value(functions)
        if not isinstance(integrand, DomainFunction):
            raise TypeError(
                "AdaptiveIntegralFunctional integrand must produce a DomainFunction; "
                f"got {type(integrand).__name__}."
            )
        return integrand

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        """Evaluate and return the raw signed adaptive integral."""
        result = adaptive_integral(
            self._integrand_function(functions),
            component=self.component,
            variable=self.variable,
            quadrature=self.quadrature,
            key=key,
            **kwargs,
        )
        if not isinstance(result.value, cx.Field):
            raise TypeError("Expected adaptive_integral to return a coordax.Field value.")
        if result.value.dims != ():
            raise ValueError(
                "AdaptiveIntegralFunctional requires a scalar integral; "
                f"got dims={result.value.dims}."
            )

        value = jnp.asarray(result.value.data).reshape(())
        if jnp.iscomplexobj(value):
            raise TypeError(
                "AdaptiveIntegralFunctional requires a real scalar integrand; "
                "use real_part(...) to select an explicitly real objective."
            )
        value = eqx.error_if(
            value,
            result.status != 0,
            "AdaptiveIntegralFunctional quadrature did not converge.",
        )
        return self.weight * value


__all__ = ["AdaptiveIntegralFunctional"]
