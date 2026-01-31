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
from .._frozendict import frozendict
from ..domain._base import _AbstractGeometry
from ..domain._components import DomainComponent, Fixed, FixedEnd, FixedStart
from ..domain._domain import _AbstractDomain, RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._scalar import _AbstractScalarDomain
from ..domain._structure import Points, PointsBatch, ProductStructure
from ._base import AbstractConstraint
from ._functional import _SquaredFrobeniusResidual


def _unwrap_factor(factor: object, /) -> object:
    if isinstance(factor, RelabeledDomain):
        return factor.base
    return factor


def _fixed_labels(component: DomainComponent, /) -> frozenset[str]:
    fixed: set[str] = set()
    for lbl in component.domain.labels:
        if isinstance(component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed)):
            fixed.add(lbl)
    return frozenset(fixed)


def _as_point_array(domain: _AbstractDomain, label: str, x: Array, /) -> Array:
    factor = _unwrap_factor(domain.factor(label))
    arr = jnp.asarray(x, dtype=float)

    if isinstance(factor, _AbstractGeometry):
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        if arr.ndim != 2:
            raise ValueError(
                f"Geometry points for {label!r} must have shape (N,d), got {arr.shape}."
            )
        d = int(factor.var_dim)
        if arr.shape[1] != d:
            raise ValueError(
                f"Geometry points for {label!r} must have d={d}, got {arr.shape[1]}."
            )
        return arr

    if isinstance(factor, _AbstractScalarDomain):
        if arr.ndim == 0:
            return arr.reshape((1,))
        if arr.ndim == 1:
            return arr.reshape((-1,))
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape((-1,))
        raise ValueError(
            f"Scalar points for {label!r} must have shape (N,), got {arr.shape}."
        )

    raise TypeError(
        f"Unsupported unary domain factor {type(factor).__name__} for label {label!r}."
    )


def _split_stacked_points(
    domain: _AbstractDomain,
    labels: tuple[str, ...],
    points: ArrayLike,
    /,
) -> frozendict[str, Array]:
    pts = jnp.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError(f"Expected stacked points to have shape (N,D), got {pts.shape}.")

    widths: list[int] = []
    for lbl in labels:
        factor = _unwrap_factor(domain.factor(lbl))
        if isinstance(factor, _AbstractGeometry):
            widths.append(int(factor.var_dim))
            continue
        if isinstance(factor, _AbstractScalarDomain):
            widths.append(1)
            continue
        raise TypeError(
            f"Unsupported unary domain factor {type(factor).__name__} for label {lbl!r}."
        )

    total = int(sum(widths))
    if int(pts.shape[1]) != total:
        raise ValueError(
            f"Expected stacked points dimension D={total}, got {pts.shape[1]}."
        )

    out: dict[str, Array] = {}
    offset = 0
    for lbl, w in zip(labels, widths, strict=True):
        if w == 1:
            out[lbl] = pts[:, offset].reshape((-1,))
        else:
            out[lbl] = pts[:, offset : offset + w]
        offset += w
    return frozendict(out)


def points_batch_from_points(
    component: DomainComponent,
    points: Mapping[str, ArrayLike] | ArrayLike,
    /,
) -> PointsBatch:
    domain = component.domain
    fixed_labels = _fixed_labels(component)
    non_fixed_labels = tuple(lbl for lbl in domain.labels if lbl not in fixed_labels)

    if non_fixed_labels:
        structure = ProductStructure((non_fixed_labels,)).canonicalize(
            domain.labels, fixed_labels=fixed_labels
        )
        axis_names = structure.axis_names
        if axis_names is None:
            raise ValueError(
                "points_batch_from_points requires a canonicalized ProductStructure."
            )
        axis = axis_names[0]
    else:
        structure = ProductStructure(()).canonicalize(
            domain.labels, fixed_labels=fixed_labels
        )
        axis = None

    if isinstance(points, Mapping):
        raw = frozendict({k: jnp.asarray(v, dtype=float) for k, v in points.items()})
    else:
        raw = _split_stacked_points(domain, non_fixed_labels, points)

    mapped: dict[str, Any] = {}
    for lbl in domain.labels:
        comp = component.spec.component_for(lbl)
        factor = _unwrap_factor(domain.factor(lbl))

        if lbl in fixed_labels:
            if isinstance(factor, _AbstractScalarDomain):
                if isinstance(comp, FixedStart):
                    val = factor.fixed("start")
                elif isinstance(comp, FixedEnd):
                    val = factor.fixed("end")
                else:
                    assert isinstance(comp, Fixed)
                    val = jnp.asarray(comp.value, dtype=float).reshape(())
                mapped[lbl] = cx.Field(jnp.asarray(val, dtype=float).reshape(()), dims=())
                continue

            if isinstance(factor, _AbstractGeometry):
                assert isinstance(comp, Fixed)
                val = jnp.asarray(comp.value, dtype=float).reshape((int(factor.var_dim),))
                mapped[lbl] = cx.Field(val, dims=(None,))
                continue

            raise TypeError(
                f"Unsupported unary domain factor {type(factor).__name__} for label {lbl!r}."
            )

        if lbl not in raw:
            raise KeyError(
                f"Missing point coordinates for non-fixed label {lbl!r}; expected at least {non_fixed_labels}."
            )
        if axis is None:
            raise ValueError(
                "Internal error: missing sampling axis for non-fixed points."
            )

        arr = _as_point_array(domain, lbl, raw[lbl])
        if isinstance(factor, _AbstractGeometry):
            mapped[lbl] = cx.Field(arr, dims=(axis, None))
        elif isinstance(factor, _AbstractScalarDomain):
            mapped[lbl] = cx.Field(arr.reshape((-1,)), dims=(axis,))
        else:
            raise TypeError(
                f"Unsupported unary domain factor {type(factor).__name__} for label {lbl!r}."
            )

    pts: Points = frozendict(mapped)
    return PointsBatch(points=pts, structure=structure)


class PointSetConstraint(AbstractConstraint):
    r"""A constraint evaluated on an explicit finite set of points.

    Given a fixed collection of points $\{z_i\}_{i=1}^N$ (encoded as a `PointsBatch`)
    and a residual `DomainFunction` $r(z)$, the pointwise squared residual is

    $$
    \rho(z_i)=\|r(z_i)\|_F^2.
    $$

    The scalar loss is then

    For `reduction="mean"`:

    $$
    \ell = w\,\frac{1}{N}\sum_{i=1}^N \rho(z_i),
    $$

    For `reduction="sum"`:

    $$
    \ell = w\sum_{i=1}^N \rho(z_i),
    $$

    where $w$ is the scalar `weight`.
    """

    constraint_vars: tuple[str, ...]
    points: PointsBatch
    weight: Array
    label: str | None
    reduction: Literal["mean", "sum"]
    residual: Callable[[Mapping[str, DomainFunction]], DomainFunction]

    def __init__(
        self,
        *,
        points: PointsBatch,
        residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        constraint_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        """Create a point-set constraint from points and a residual callable."""
        self.constraint_vars = () if constraint_vars is None else tuple(constraint_vars)
        self.points = points
        self.residual = residual
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        self.reduction = reduction

    @classmethod
    def from_points(
        cls,
        *,
        component: DomainComponent,
        points: Mapping[str, ArrayLike] | ArrayLike,
        residual: Callable[[Mapping[str, DomainFunction]], DomainFunction],
        constraint_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> "PointSetConstraint":
        """Build a `PointSetConstraint` from raw point coordinates."""
        batch = points_batch_from_points(component, points)
        return cls(
            points=batch,
            residual=residual,
            constraint_vars=constraint_vars,
            weight=weight,
            label=label,
            reduction=reduction,
        )

    @classmethod
    def from_operator(
        cls,
        *,
        points: PointsBatch,
        operator: Callable[..., DomainFunction],
        constraint_vars: str | Sequence[str],
        weight: ArrayLike = 1.0,
        label: str | None = None,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> "PointSetConstraint":
        """Build a `PointSetConstraint` from an operator applied to named fields."""
        vars_tuple = (
            (constraint_vars,)
            if isinstance(constraint_vars, str)
            else tuple(constraint_vars)
        )

        def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
            return operator(*(functions[name] for name in vars_tuple))

        return cls(
            points=points,
            residual=residual,
            constraint_vars=vars_tuple,
            weight=weight,
            label=label,
            reduction=reduction,
        )

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch:
        del key
        return self.points

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate the point-set loss.

        This evaluates the residual on the stored points and applies the configured
        reduction (`mean` or `sum`).
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
            if base is None:
                raise ValueError(
                    "PointSetConstraint cannot infer a DomainFunction to wrap residual output."
                )
            if callable(res):
                res = DomainFunction(
                    domain=base.domain, deps=base.deps, func=res, metadata={}
                )
            else:
                res = DomainFunction(domain=base.domain, deps=(), func=res, metadata={})

        f = DomainFunction(
            domain=res.domain,
            deps=res.deps,
            func=_SquaredFrobeniusResidual(res),
            metadata=res.metadata,
        )
        out = f(self.points, key=key, **kwargs)
        if not isinstance(out, cx.Field):
            raise TypeError("Expected pointset evaluation to return a coordax.Field.")

        data = jnp.asarray(out.data, dtype=float)
        if self.reduction == "sum":
            value = jnp.sum(data)
        else:
            value = jnp.mean(data)
        return self.weight * value.reshape(())
