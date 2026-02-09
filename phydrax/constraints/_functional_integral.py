#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from ..domain._components import (
    DomainComponent,
    DomainComponentUnion,
)
from ..domain._function import DomainFunction
from ..domain._structure import (
    CoordSeparableBatch,
    PointsBatch,
    ProductStructure,
)
from ..operators.integral._batch_ops import integral
from ._base import AbstractSamplingConstraint
from ._sampling_spec import (
    CoordSamplingMap,
    parse_sampling_num_points,
    SamplingNumPoints,
)


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
    component: DomainComponent | DomainComponentUnion
    structure: ProductStructure
    coord_sampling: CoordSamplingMap | None
    dense_structure: ProductStructure | None
    num_points: Any
    sampler: str
    weight: Array
    label: str | None
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "integral"]
    integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
    equal_to: Array

    def __init__(
        self,
        *,
        component: DomainComponent | DomainComponentUnion,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        equal_to: ArrayLike = 0.0,
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        constraint_vars: Sequence[str] | None = None,
        sampler: str = "latin_hypercube",
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
    ):
        """Create an integral equality constraint from an integrand callable."""
        self.constraint_vars = () if constraint_vars is None else tuple(constraint_vars)
        self.component = component
        self.integrand = integrand
        self.equal_to = jnp.asarray(equal_to, dtype=float)
        self.structure = structure
        dense_num_points, coord_sampling, dense_structure_out = parse_sampling_num_points(
            component,
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
        )
        self.num_points = dense_num_points
        self.coord_sampling = coord_sampling
        self.dense_structure = dense_structure_out
        self.sampler = str(sampler)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.over = over
        self.reduction = "integral"

    @classmethod
    def from_integrand(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        equal_to: ArrayLike = 0.0,
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        constraint_vars: Sequence[str] | None = None,
        sampler: str = "latin_hypercube",
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
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
            constraint_vars=constraint_vars,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
        )

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        operator: Callable[..., DomainFunction],
        constraint_vars: str | Sequence[str],
        equal_to: ArrayLike = 0.0,
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        sampler: str = "latin_hypercube",
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
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
            constraint_vars=vars_tuple,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        """Sample points for estimating the integral."""
        if self.coord_sampling is not None:
            if isinstance(self.component, DomainComponentUnion):
                raise ValueError(
                    "coord-separable sampling is not supported for DomainComponentUnion."
                )
            return self.component.sample_coord_separable(
                self.coord_sampling,
                num_points=self.num_points,
                dense_structure=self.dense_structure,
                sampler=self.sampler,
                key=key,
            )
        return self.component.sample(
            self.num_points,
            structure=self.structure,
            sampler=self.sampler,
            key=key,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None = None,
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

        batch_ = self.sample(key=key) if batch is None else batch
        out = integral(
            f,
            batch_,
            component=self.component,
            over=self.over,
            key=key,
            **kwargs,
        )
        if not isinstance(out, cx.Field):
            raise TypeError("Expected integral to return a coordax.Field.")

        diff = jnp.asarray(out.data, dtype=float) - self.equal_to
        sq = jnp.sum(diff * diff)
        return self.weight * sq.reshape(())
