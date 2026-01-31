#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._base import _AbstractGeometry
from ..domain._domain import _AbstractDomain, RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._scalar import _AbstractScalarDomain


def _unwrap_factor(factor: object, /) -> object:
    if isinstance(factor, RelabeledDomain):
        return factor.base
    return factor


def _as_anchor_array(domain: _AbstractDomain, label: str, x: Array, /) -> Array:
    factor = _unwrap_factor(domain.factor(label))
    arr = jnp.asarray(x, dtype=float)

    if isinstance(factor, _AbstractGeometry):
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        if arr.ndim != 2:
            raise ValueError(
                f"Geometry anchors for {label!r} must have shape (N,d), got {arr.shape}."
            )
        d = int(factor.var_dim)
        if arr.shape[1] != d:
            raise ValueError(
                f"Geometry anchors for {label!r} must have d={d}, got {arr.shape[1]}."
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
            f"Scalar anchors for {label!r} must have shape (N,), got {arr.shape}."
        )

    raise TypeError(
        f"Unsupported unary domain factor {type(factor).__name__} for label {label!r}."
    )


def _split_stacked(
    domain: _AbstractDomain,
    labels: tuple[str, ...],
    points: ArrayLike,
    /,
) -> frozendict[str, Array]:
    pts = jnp.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError(
            f"Expected stacked anchors to have shape (N,D), got {pts.shape}."
        )

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
            f"Expected stacked anchors dimension D={total}, got {pts.shape[1]}."
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


class _IDWInterpolant(StrictModule):
    anchors: frozendict[str, Array]
    values: Array
    labels: tuple[str, ...]
    idw_exponent: float
    eps: float
    eps_snap: float
    lengthscales: frozendict[str, float]

    def __init__(
        self,
        *,
        anchors: frozendict[str, Array],
        values: Array,
        labels: tuple[str, ...],
        idw_exponent: float,
        eps: float,
        eps_snap: float,
        lengthscales: frozendict[str, float],
    ):
        self.anchors = anchors
        self.values = values
        self.labels = labels
        self.idw_exponent = float(idw_exponent)
        self.eps = float(eps)
        self.eps_snap = float(eps_snap)
        self.lengthscales = lengthscales

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        del key, kwargs
        if len(args) != len(self.labels):
            raise ValueError("IDW interpolant called with wrong number of arguments.")

        n = int(next(iter(self.anchors.values())).shape[0])
        dist2 = jnp.zeros((n,), dtype=float)

        for lbl, x in zip(self.labels, args, strict=True):
            a = self.anchors[lbl]
            scale = float(self.lengthscales.get(lbl, 1.0))
            x_arr = jnp.asarray(x, dtype=float)
            if a.ndim == 2:
                x_vec = x_arr.reshape((-1,))
                diff = (a - x_vec[None, :]) / scale
                dist2 = dist2 + jnp.sum(diff * diff, axis=1)
            else:
                x_scalar = x_arr.reshape(())
                diff = (a - x_scalar) / scale
                dist2 = dist2 + diff * diff

        dmin = jnp.min(dist2)
        argmin = jnp.argmin(dist2)

        y = jnp.asarray(self.values, dtype=float)
        if y.ndim == 1:
            y_rows = y.reshape((n, 1))
        else:
            y_rows = y

        def _snap(_: Any):
            return y_rows[argmin]

        def _interp(_: Any):
            power = 0.5 * self.idw_exponent
            w_raw = (dist2 + self.eps) ** (-power)
            w = w_raw / (jnp.sum(w_raw) + self.eps)
            out = jnp.sum(w[:, None] * y_rows, axis=0)
            return out

        out = jax.lax.cond(dmin < self.eps_snap, _snap, _interp, operand=None)
        if y.ndim == 1:
            return out.reshape(())
        return out


def idw_interpolant(
    domain: _AbstractDomain,
    /,
    *,
    anchors: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    labels: tuple[str, ...] | None = None,
    lengthscales: Mapping[str, float] | None = None,
    idw_exponent: float = 2.0,
    eps: float = 1e-12,
    eps_snap: float = 1e-12,
) -> DomainFunction:
    deps = domain.labels if labels is None else tuple(labels)
    for lbl in deps:
        if lbl not in domain.labels:
            raise ValueError(
                f"Unknown label {lbl!r}; expected subset of {domain.labels}."
            )

    if isinstance(anchors, Mapping):
        anchors_map = frozendict(
            {k: jnp.asarray(v, dtype=float) for k, v in anchors.items()}
        )
    else:
        anchors_map = _split_stacked(domain, deps, anchors)

    anchors_arr = frozendict(
        {lbl: _as_anchor_array(domain, lbl, anchors_map[lbl]) for lbl in deps}
    )
    n = int(next(iter(anchors_arr.values())).shape[0])
    for lbl, arr in anchors_arr.items():
        if int(arr.shape[0]) != n:
            raise ValueError("All anchor arrays must share the same leading dimension N.")

    y = jnp.asarray(values, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape((-1,))

    if y.ndim == 1:
        if int(y.shape[0]) != n:
            raise ValueError("values must have leading dim N matching anchors.")
    elif y.ndim == 2:
        if int(y.shape[0]) != n:
            raise ValueError("values must have leading dim N matching anchors.")
    else:
        raise ValueError(f"values must be rank-1 or rank-2, got shape {y.shape}.")

    ls = frozendict(
        {} if lengthscales is None else {k: float(v) for k, v in lengthscales.items()}
    )
    interp = _IDWInterpolant(
        anchors=anchors_arr,
        values=y,
        labels=deps,
        idw_exponent=float(idw_exponent),
        eps=float(eps),
        eps_snap=float(eps_snap),
        lengthscales=ls,
    )
    return DomainFunction(domain=domain, deps=deps, func=interp)
