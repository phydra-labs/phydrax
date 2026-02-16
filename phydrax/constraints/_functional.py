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
from .._strict import StrictModule
from ..domain._components import DomainComponent, DomainComponentUnion
from ..domain._function import DomainFunction
from ..domain._structure import (
    CoordSeparableBatch,
    PointsBatch,
    ProductStructure,
)
from ..operators.integral._batch_ops import integral, mean
from ._base import AbstractSamplingConstraint
from ._sampling_spec import (
    CoordSamplingMap,
    parse_sampling_num_points,
    SamplingNumPoints,
)


class _SquaredFrobeniusResidual(StrictModule):
    residual: DomainFunction

    def __init__(self, residual: DomainFunction):
        self.residual = residual

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        y = jnp.asarray(self.residual.func(*args, key=key, **kwargs))
        return jnp.sum(y * y)


class FunctionalConstraint(AbstractSamplingConstraint):
    r"""A sampled objective term defined by a residual `DomainFunction`.

    A `FunctionalConstraint` represents one term in a physics/data objective. It is
    defined by:

    - a `DomainComponent` (or union) describing the integration/sampling region
      $\Omega_{\text{comp}}$ and measure $\mu$;
    - a residual operator producing a `DomainFunction` $r(z)$ from the current set of
      field functions.

    The pointwise squared residual is taken as a Frobenius norm:

    $$
    \rho(z) = \|r(z)\|_F^2 = \sum_{i} r_i(z)^2,
    $$

    and the scalar loss is computed using either reduction mode.

    If `weight` is a scalar/array-like, it is treated as a global multiplier $w$.
    If `weight` is a `DomainFunction`, it is applied pointwise inside the reduction.

    For `reduction="mean"` with scalar weight:

    $$
    \ell = w\,\frac{1}{\mu(\Omega_{\text{comp}})}\int_{\Omega_{\text{comp}}} \rho(z)\,d\mu(z),
    $$

    For `reduction="integral"` with scalar weight:

    $$
    \ell = w\int_{\Omega_{\text{comp}}} \rho(z)\,d\mu(z),
    $$

    where $w$ is the scalar global `weight`.

    Sampling is performed according to `structure` (paired blocks) or coord-separable
    mapping specs encoded directly in `num_points`.
    """

    constraint_vars: tuple[str, ...]
    component: DomainComponent | DomainComponentUnion
    structure: ProductStructure
    coord_sampling: CoordSamplingMap | None
    dense_structure: ProductStructure | None
    num_points: Any
    sampler: str
    weight: Array
    pointwise_weight: DomainFunction | None
    label: str | None
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "integral"]
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction]

    def __init__(
        self,
        *,
        component: DomainComponent | DomainComponentUnion,
        residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        constraint_vars: Sequence[str] | None = None,
        sampler: str = "latin_hypercube",
        weight: DomainFunction | ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
        reduction: Literal["mean", "integral"] = "mean",
    ):
        self.constraint_vars = () if constraint_vars is None else tuple(constraint_vars)
        self.component = component
        self.residual = residual
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
        if isinstance(weight, DomainFunction):
            self.weight = jnp.asarray(1.0, dtype=float)
            self.pointwise_weight = weight
        else:
            self.weight = jnp.asarray(weight, dtype=float)
            self.pointwise_weight = None
        self.label = None if label is None else str(label)
        self.over = over
        self.reduction = reduction

    @classmethod
    def from_operator(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        operator: Callable[..., DomainFunction],
        constraint_vars: str | Sequence[str],
        num_points: SamplingNumPoints,
        structure: ProductStructure,
        dense_structure: ProductStructure | None = None,
        sampler: str = "latin_hypercube",
        weight: DomainFunction | ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
        reduction: Literal["mean", "integral"] = "mean",
    ) -> "FunctionalConstraint":
        r"""Create a `FunctionalConstraint` from an operator mapping `DomainFunction`s to a residual.

        This wraps an `operator(u1, u2, ...) -> r` into a residual callable
        `residual(functions) -> r` using the provided `constraint_vars`.
        """
        vars_tuple = (
            (constraint_vars,)
            if isinstance(constraint_vars, str)
            else tuple(constraint_vars)
        )

        def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in vars_tuple))

        return cls(
            component=component,
            residual=residual,
            num_points=num_points,
            structure=structure,
            dense_structure=dense_structure,
            constraint_vars=vars_tuple,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
            reduction=reduction,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
        r"""Sample points from the configured component.

        - Returns a `PointsBatch` for paired sampling.
        - Returns a `CoordSeparableBatch` when `num_points` requested coord-separable sampling.
        - Returns a tuple of `PointsBatch` when sampling from a `DomainComponentUnion`.
        """
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
        r"""Evaluate the scalar loss for this constraint.

        This samples the configured component, evaluates the residual, forms a squared
        Frobenius norm, and reduces via `mean(...)` or `integral(...)` depending on
        `reduction` and `over`.

        This:
        1) builds the residual `DomainFunction` $r$ from `functions`,
        2) samples points $z_i$ on `component`,
        3) computes $\rho(z_i)=\|r(z_i)\|_F^2$,
        4) reduces using either a mean or an integral estimator.
        """
        res = self.residual(functions)
        if not isinstance(res, DomainFunction):
            base = None
            if self.constraint_vars:
                base = functions.get(self.constraint_vars[0])
            if base is None:
                for fn in functions.values():
                    if isinstance(fn, DomainFunction):
                        base = fn
                        break
            domain = base.domain if base is not None else self.component.domain
            if callable(res):
                deps = base.deps if base is not None else domain.labels
                res = DomainFunction(domain=domain, deps=deps, func=res, metadata={})
            else:
                res = DomainFunction(domain=domain, deps=(), func=res, metadata={})

        batch_ = self.sample(key=key) if batch is None else batch
        f = DomainFunction(
            domain=res.domain,
            deps=res.deps,
            func=_SquaredFrobeniusResidual(res),
            metadata=res.metadata,
        )
        if self.pointwise_weight is not None:
            w = self.pointwise_weight
            if w.domain.labels != f.domain.labels:
                w = w.promote(f.domain)
            f = w * f
        if self.reduction == "mean":
            out = mean(
                f,
                batch_,
                component=self.component,
                over=self.over,
                key=key,
                **kwargs,
            )
        else:
            out = integral(
                f,
                batch_,
                component=self.component,
                over=self.over,
                key=key,
                **kwargs,
            )
        if not isinstance(out, cx.Field):
            raise TypeError("Expected reduction to return a coordax.Field.")
        if out.dims != ():
            raise ValueError(
                f"Constraint reduction must return a scalar Field, got dims={out.dims}."
            )
        return self.weight * jnp.asarray(out.data, dtype=float).reshape(())
