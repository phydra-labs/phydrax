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
from ..domain._function import batch_aware_callable, BatchAwareCallable, DomainFunction
from ..domain._structure import (
    CoordSeparableBatch,
    PointsBatch,
    ProductStructure,
)
from ..operators.integral._batch_ops import integral, mean
from ._adaptive import AbstractCollocationPolicy
from ._base import AbstractSamplingConstraint
from ._data_metrics import supervised_data_metrics
from ._sampling_spec import (
    CoordSamplingMap,
    parse_sampling_num_points,
    SamplingNumPoints,
)


def _validate_batch_weight(
    batch: PointsBatch | CoordSeparableBatch,
    weight: cx.Field,
) -> None:
    if any(dim is None for dim in weight.dims):
        raise ValueError("Adaptive batch weights may use only named sampling axes.")
    if isinstance(batch, PointsBatch):
        axes = batch.structure.axis_names
        if axes is None:
            raise ValueError("PointsBatch.structure must be canonicalized.")
        allowed = frozenset(axes)
    else:
        dense_axes = batch.dense_structure.axis_names
        if dense_axes is None:
            raise ValueError("CoordSeparableBatch.dense_structure must be canonicalized.")
        allowed = frozenset(
            axis
            for axes_for_label in batch.coord_axes_by_label.values()
            for axis in axes_for_label
        ) | frozenset(dense_axes)
    unknown = tuple(dim for dim in weight.dims if dim not in allowed)
    if unknown:
        raise ValueError(
            f"Adaptive batch weight uses axes {unknown!r} absent from the batch."
        )


class _BatchWeightedSquaredResidual(StrictModule, BatchAwareCallable):
    residual: DomainFunction
    weight: cx.Field

    def __init__(self, residual: DomainFunction, weight: cx.Field):
        self.residual = residual
        self.weight = weight

    def use_batch_call(self) -> bool:
        return True

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        squared = _SquaredFrobeniusResidual(self.residual).__call_batch__(
            batch, key=key, **kwargs
        )
        return self.weight * squared

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        raise TypeError("Adaptive batch weights require structured batch evaluation.")


class _SquaredFrobeniusResidual(StrictModule, BatchAwareCallable):
    residual: DomainFunction

    def __init__(self, residual: DomainFunction):
        self.residual = residual

    def use_batch_call(self) -> bool:
        return batch_aware_callable(self.residual.func) is not None

    def __call_batch__(
        self,
        batch: PointsBatch | CoordSeparableBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        y = self.residual(batch, key=key, **kwargs)
        if not isinstance(y, cx.Field):
            raise TypeError("Expected residual to return a coordax.Field.")

        data = jnp.asarray(y.data)
        dims = y.dims
        squared = jnp.real(jnp.conj(data) * data)
        reduction_axes = [i for i, dim in enumerate(dims) if dim is None]
        for axis in reversed(reduction_axes):
            squared = jnp.sum(squared, axis=axis)
            dims = dims[:axis] + dims[axis + 1 :]
        return cx.Field(squared, dims=dims)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        y = jnp.asarray(self.residual.func(*args, key=key, **kwargs))
        return jnp.sum(jnp.real(jnp.conj(y) * y))


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
    \rho(z) = \|r(z)\|_F^2 = \sum_{i} |r_i(z)|^2.
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

    Sampling policy is controlled by `sampling_mode`:

    - `"resample"`: draw a new batch every loss evaluation (default).
    - `"fixed"`: build one batch once (from `fixed_batch` or `fixed_batch_key`)
      and reuse it.
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
    sampling_mode: Literal["resample", "fixed"]
    fixed_batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction]
    data_constraint_var: str | None
    data_target: DomainFunction | None
    data_accuracy_eps: Array
    collocation_policy: AbstractCollocationPolicy | None

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
        sampling_mode: Literal["resample", "fixed"] = "resample",
        fixed_batch: (
            PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
        ) = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
        data_constraint_var: str | None = None,
        data_target: DomainFunction | None = None,
        data_accuracy_eps: float = 1e-12,
        collocation_policy: AbstractCollocationPolicy | None = None,
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
        sampling_mode_str = str(sampling_mode).lower()
        if sampling_mode_str not in ("resample", "fixed"):
            raise ValueError("sampling_mode must be either 'resample' or 'fixed'.")
        if sampling_mode_str == "fixed":
            self.sampling_mode = "fixed"
            self.fixed_batch = (
                self._sample_once(key=fixed_batch_key)
                if fixed_batch is None
                else fixed_batch
            )
        else:
            self.sampling_mode = "resample"
            if fixed_batch is not None:
                raise ValueError("fixed_batch is only valid when sampling_mode='fixed'.")
            self.fixed_batch = None
        self.data_constraint_var = (
            None if data_constraint_var is None else str(data_constraint_var)
        )
        self.data_target = data_target
        self.data_accuracy_eps = jnp.asarray(float(data_accuracy_eps), dtype=float)
        self.collocation_policy = collocation_policy

    def _sample_once(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...]:
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
        sampling_mode: Literal["resample", "fixed"] = "resample",
        fixed_batch: (
            PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None
        ) = None,
        fixed_batch_key: Key[Array, ""] = DOC_KEY0,
        data_constraint_var: str | None = None,
        data_target: DomainFunction | None = None,
        data_accuracy_eps: float = 1e-12,
        collocation_policy: AbstractCollocationPolicy | None = None,
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
            sampling_mode=sampling_mode,
            fixed_batch=fixed_batch,
            fixed_batch_key=fixed_batch_key,
            data_constraint_var=data_constraint_var,
            data_target=data_target,
            data_accuracy_eps=data_accuracy_eps,
            collocation_policy=collocation_policy,
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
        - In `sampling_mode="fixed"`, this returns the same stored batch every call.
        """
        if self.sampling_mode == "fixed":
            if self.fixed_batch is None:
                raise ValueError("sampling_mode='fixed' requires fixed_batch to be set.")
            return self.fixed_batch
        return self._sample_once(key=key)

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Evaluate supervised-data diagnostics for sampled data-fit constraints."""
        if self.data_constraint_var is None or self.data_target is None:
            return {}

        batch_ = self.sample(key=key) if batch is None else batch
        if isinstance(batch_, tuple):
            raise TypeError("Data metrics require a single sampled batch.")

        prediction = functions[self.data_constraint_var](batch_, key=key, **kwargs)
        target = self.data_target(batch_, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError("Expected data prediction to return a coordax.Field.")
        if not isinstance(target, cx.Field):
            raise TypeError("Expected data target to return a coordax.Field.")

        return supervised_data_metrics(
            jnp.asarray(prediction.data, dtype=float),
            jnp.asarray(target.data, dtype=float),
            eps=self.data_accuracy_eps,
        )

    def _residual_function(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        res = self.residual(functions)
        if isinstance(res, DomainFunction):
            return res
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
            return DomainFunction(domain=domain, deps=deps, func=res, metadata={})
        return DomainFunction(domain=domain, deps=(), func=res, metadata={})

    def pointwise_loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        batch: PointsBatch | CoordSeparableBatch,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        """Return the unreduced squared Frobenius residual on ``batch``."""
        residual = self._residual_function(functions)
        out = _SquaredFrobeniusResidual(residual).__call_batch__(
            batch, key=key, **kwargs
        )
        if self.pointwise_weight is not None:
            weight = self.pointwise_weight
            if weight.domain.labels != residual.domain.labels:
                weight = weight.promote(residual.domain)
            evaluated = weight(batch, key=key, **kwargs)
            if not isinstance(evaluated, cx.Field):
                raise TypeError("Pointwise weight must evaluate to a coordax.Field.")
            out = evaluated * out
        return out

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...] | None = None,
        batch_weight: cx.Field | None = None,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate the scalar loss for this constraint.

        This samples the configured component, evaluates the residual, forms a squared
        Frobenius norm, and reduces via `mean(...)` or `integral(...)` depending on
        `reduction` and `over`.

        If `batch` is provided, it is used directly (overriding `sampling_mode`).

        This:
        1) builds the residual `DomainFunction` $r$ from `functions`,
        2) samples points $z_i$ on `component`,
        3) computes $\rho(z_i)=\|r(z_i)\|_F^2$,
        4) reduces using either a mean or an integral estimator.
        """
        res = self._residual_function(functions)
        batch_ = self.sample(key=key) if batch is None else batch
        residual_callable: BatchAwareCallable
        if batch_weight is None:
            residual_callable = _SquaredFrobeniusResidual(res)
        else:
            if not isinstance(batch_, (PointsBatch, CoordSeparableBatch)):
                raise TypeError(
                    "Adaptive batch weights require a single structured point batch."
                )
            _validate_batch_weight(batch_, batch_weight)
            residual_callable = _BatchWeightedSquaredResidual(res, batch_weight)
        f = DomainFunction(
            domain=res.domain,
            deps=res.deps,
            func=residual_callable,
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
