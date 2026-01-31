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
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._function import DomainFunction
from ..domain._structure import (
    CoordSeparableBatch,
    NumPoints,
    PointsBatch,
    ProductStructure,
)
from ..operators.integral._batch_ops import integral
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
    component: DomainComponent | DomainComponentUnion
    structure: ProductStructure
    coord_separable: Mapping[str, Any] | None
    dense_structure: ProductStructure | None
    num_points: NumPoints | tuple[Any, ...]
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
        num_points: NumPoints | tuple[Any, ...],
        structure: ProductStructure,
        coord_separable: Mapping[str, Any] | None = None,
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
        self.num_points = num_points
        self.structure = structure
        self.coord_separable = coord_separable
        self.dense_structure = dense_structure
        self.sampler = str(sampler)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.over = over
        self.reduction = "integral"
        self._ensure_coord_separable_interior()

    def _ensure_coord_separable_interior(self) -> None:
        if self.coord_separable is None:
            return
        if isinstance(self.component, DomainComponentUnion):
            raise ValueError(
                "coord_separable sampling is not supported for DomainComponentUnion."
            )
        coord_labels = tuple(self.coord_separable)
        bad = tuple(
            lbl
            for lbl in coord_labels
            if not isinstance(self.component.spec.component_for(lbl), Interior)
        )
        if bad:
            raise ValueError(
                f"coord_separable labels must use Interior() components; got {bad!r}."
            )

    @classmethod
    def from_integrand(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        integrand: Callable[[Mapping[str, DomainFunction]], DomainFunction]
        | DomainFunction,
        equal_to: ArrayLike = 0.0,
        num_points: NumPoints | tuple[Any, ...],
        structure: ProductStructure,
        coord_separable: Mapping[str, Any] | None = None,
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
            coord_separable=coord_separable,
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
        num_points: NumPoints | tuple[Any, ...],
        structure: ProductStructure,
        coord_separable: Mapping[str, Any] | None = None,
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
            coord_separable=coord_separable,
            dense_structure=dense_structure,
            constraint_vars=vars_tuple,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
        )

    def _dense_structure_for_coord_separable(self) -> ProductStructure | None:
        if self.dense_structure is not None:
            return self.dense_structure
        if self.coord_separable is None:
            return None
        if isinstance(self.component, DomainComponentUnion):
            raise ValueError(
                "coord_separable sampling is not supported for DomainComponentUnion."
            )
        coord_labels = set(self.coord_separable)
        fixed_labels = frozenset(
            lbl
            for lbl in self.component.domain.labels
            if isinstance(
                self.component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed)
            )
        )
        dense_labels = [
            lbl
            for lbl in self.component.domain.labels
            if lbl not in coord_labels and lbl not in fixed_labels
        ]
        if not dense_labels:
            return ProductStructure(blocks=())
        blocks: list[tuple[str, ...]] = []
        for block in self.structure.blocks:
            filtered = tuple(
                lbl
                for lbl in block
                if lbl not in coord_labels and lbl not in fixed_labels
            )
            if filtered:
                blocks.append(filtered)
        covered = set(lbl for block in blocks for lbl in block)
        missing = [lbl for lbl in dense_labels if lbl not in covered]
        if missing:
            raise ValueError(
                "coord_separable requires dense_structure to cover non-separable labels; "
                f"missing {tuple(missing)!r}."
            )
        return ProductStructure(tuple(blocks))

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        """Sample points for estimating the integral."""
        if self.coord_separable is not None:
            if isinstance(self.component, DomainComponentUnion):
                raise ValueError(
                    "coord_separable sampling is not supported for DomainComponentUnion."
                )
            return self.component.sample_coord_separable(
                self.coord_separable,
                num_points=self.num_points,
                dense_structure=self._dense_structure_for_coord_separable(),
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
