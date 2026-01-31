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
from ..operators.integral._batch_ops import integral, mean
from ._base import AbstractSamplingConstraint


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

    For `reduction="mean"`:

    $$
    \ell = w\,\frac{1}{\mu(\Omega_{\text{comp}})}\int_{\Omega_{\text{comp}}} \rho(z)\,d\mu(z),
    $$

    For `reduction="integral"`:

    $$
    \ell = w\int_{\Omega_{\text{comp}}} \rho(z)\,d\mu(z),
    $$

    where $w$ is the scalar `weight`.

    Sampling is performed according to `structure` (paired blocks) and optionally
    `coord_separable`.
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
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction]

    def __init__(
        self,
        *,
        component: DomainComponent | DomainComponentUnion,
        residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        num_points: NumPoints | tuple[Any, ...],
        structure: ProductStructure,
        coord_separable: Mapping[str, Any] | None = None,
        dense_structure: ProductStructure | None = None,
        constraint_vars: Sequence[str] | None = None,
        sampler: str = "latin_hypercube",
        weight: ArrayLike = 1.0,
        label: str | None = None,
        over: str | tuple[str, ...] | None = None,
        reduction: Literal["mean", "integral"] = "mean",
    ):
        self.constraint_vars = () if constraint_vars is None else tuple(constraint_vars)
        self.component = component
        self.residual = residual
        self.num_points = num_points
        self.structure = structure
        self.coord_separable = coord_separable
        self.dense_structure = dense_structure
        self.sampler = str(sampler)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.over = over
        self.reduction = reduction
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
    def from_operator(
        cls,
        *,
        component: DomainComponent | DomainComponentUnion,
        operator: Callable[..., DomainFunction],
        constraint_vars: str | Sequence[str],
        num_points: NumPoints | tuple[Any, ...],
        structure: ProductStructure,
        coord_separable: Mapping[str, int | Sequence[int]] | None = None,
        dense_structure: ProductStructure | None = None,
        sampler: str = "latin_hypercube",
        weight: ArrayLike = 1.0,
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
            coord_separable=coord_separable,
            dense_structure=dense_structure,
            constraint_vars=vars_tuple,
            sampler=sampler,
            weight=weight,
            label=label,
            over=over,
            reduction=reduction,
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
        r"""Sample points from the configured component.

        - Returns a `PointsBatch` for paired sampling (`coord_separable=None`).
        - Returns a `CoordSeparableBatch` when `coord_separable` is enabled.
        - Returns a tuple of `PointsBatch` when sampling from a `DomainComponentUnion`.
        """
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
