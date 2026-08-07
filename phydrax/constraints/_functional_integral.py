#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    ComponentSum,
    DomainComponent,
    DomainFunction,
    GridBatch,
    GridSampling,
    PointBatch,
    PointSampling,
)

from .._doc import DOC_KEY0
from ..integration import from_samples, over, reduce
from ._base import AbstractSamplingConstraint


class IntegralEqualityConstraint(AbstractSamplingConstraint):
    r"""A constraint enforcing an integral equality.

    Given an integrand `DomainFunction` $f(z)$ on a component $\Omega_{\text{comp}}$,
    this enforces the scalar equality

    $$
    \int_{\Omega_{\text{comp}}} f(z)\,d\mu(z) = c,
    $$

    by minimizing the squared error

    $$
    \ell = w\left\|\int_{\Omega_{\text{comp}}} f(z)\,d\mu(z) - c\right\|_2^2,
    $$

    where $w$ is `weight` and $c$ is `equal_to`.
    """

    constraint_vars: tuple[str, ...]
    component: DomainComponent | ComponentSum
    sampling: GridSampling | PointSampling | tuple[PointSampling, ...]
    weight: Array
    label: str | None
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "integral"]
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
    equal_to: Array

    def __init__(
        self,
        *,
        component: DomainComponent | ComponentSum,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        equal_to: ArrayLike = 0.0,
        sampling: GridSampling | PointSampling | tuple[PointSampling, ...],
        constraint_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
    ):
        """Create an integral equality constraint from an integrand callable."""
        self.constraint_vars = () if constraint_vars is None else tuple(constraint_vars)
        self.component = component
        self.integrand = integrand
        self.equal_to = jnp.asarray(equal_to, dtype=float)
        if isinstance(component, ComponentSum):
            if isinstance(sampling, GridSampling):
                raise TypeError("ComponentSum does not support GridSampling.")
        elif isinstance(sampling, tuple):
            raise TypeError(
                "Per-term PointSampling tuples require a ComponentSum."
            )
        self.sampling = sampling
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.over = over
        self.reduction = "integral"

    @classmethod
    def from_integrand(
        cls,
        *,
        component: DomainComponent | ComponentSum,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        equal_to: ArrayLike = 0.0,
        sampling: GridSampling | PointSampling | tuple[PointSampling, ...],
        constraint_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
    ) -> "IntegralEqualityConstraint":
        """Build an `IntegralEqualityConstraint` from an integrand callable or `DomainFunction`."""
        if isinstance(integrand, DomainFunction):

            def _fn(_: Mapping[str, DomainFunction], /) -> DomainFunction:
                return integrand

            integrand_fn = _fn
        else:
            integrand_fn = integrand

        return cls(
            component=component,
            integrand=integrand_fn,
            equal_to=equal_to,
            sampling=sampling,
            constraint_vars=constraint_vars,
            weight=weight,
            label=label,
            over=over,
        )

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent | ComponentSum,
        operator: Callable[..., DomainFunction],
        constraint_vars: str | Sequence[str],
        equal_to: ArrayLike = 0.0,
        sampling: GridSampling | PointSampling | tuple[PointSampling, ...],
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
    ) -> "IntegralEqualityConstraint":
        """Build an `IntegralEqualityConstraint` from an operator applied to named fields."""
        vars_tuple = (
            (constraint_vars,)
            if isinstance(constraint_vars, str)
            else tuple(constraint_vars)
        )

        def integrand(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in vars_tuple))

        return cls(
            component=component,
            integrand=integrand,
            equal_to=equal_to,
            sampling=sampling,
            constraint_vars=vars_tuple,
            weight=weight,
            label=label,
            over=over,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointBatch | GridBatch | tuple[PointBatch, ...]:
        """Sample points for estimating the integral."""
        component = self.component
        sampling = self.sampling
        if isinstance(component, ComponentSum):
            if isinstance(sampling, GridSampling):
                raise RuntimeError("ComponentSum sampling invariant was violated.")
            return component.sample(sampling, key=key)
        if isinstance(sampling, tuple):
            raise RuntimeError("DomainComponent sampling invariant was violated.")
        if isinstance(sampling, PointSampling):
            return component.sample(sampling, key=key)
        return component.sample(sampling, key=key)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: PointBatch | GridBatch | tuple[PointBatch, ...] | None = None,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate the squared integral mismatch loss.

        Computes the integral estimate $\widehat{I} \approx \int f\,d\mu$ and returns
        $w\|\widehat{I}-c\|_2^2$.
        """
        f = self.integrand(functions)
        if not isinstance(f, DomainFunction):
            base = None
            if self.constraint_vars:
                base = functions.get(self.constraint_vars[0])
            if base is None:
                for fn in functions.values():
                    if isinstance(fn, DomainFunction):
                        base = fn
                        break
            domain = base.domain if base is not None else self.component.domain
            if callable(f):
                deps = base.deps if base is not None else domain.labels
                f = DomainFunction(domain=domain, deps=deps, func=f, metadata={})
            else:
                f = DomainFunction(domain=domain, deps=(), func=f, metadata={})

        if batch is None:
            sampling_key, evaluation_key = jr.split(key)
            batch_ = self.sample(key=sampling_key)
        else:
            batch_ = batch
            evaluation_key = key
        target = over(self.component, axes=self.over)
        realization = from_samples(target, batch_, key=evaluation_key)
        out = reduce(f, realization, **kwargs).value
        if not isinstance(out, cx.Field):
            raise TypeError("Expected integral to return a coordax.Field.")

        diff = jnp.asarray(out.data, dtype=float) - self.equal_to
        sq = jnp.sum(diff * diff)
        return self.weight * sq.reshape(())
