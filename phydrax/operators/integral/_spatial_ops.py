#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ...domain._function import DomainFunction
from ..differential._domain_ops import _factor_and_dim
from ._ctx import _compile_ctx_integrand


def spatial_integral(
    u: DomainFunction,
    /,
    *,
    quad: dict,
    kernel: Callable | None = None,
    nonlinearity: Callable | None = None,
    importance_weight: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    var: str = "x",
    time_var: str | None = None,
) -> DomainFunction:
    r"""Spatial integral operator with a fixed quadrature rule.

    Constructs a new function $v$ defined by

    $$
    v(x) = \int_{\Omega} K(x,y)\,g(u(y))\,dy,
    $$

    approximated using the provided quadrature nodes/weights `quad = {"points": y_j, "weights": w_j}`:

    $$
    v(x) \approx \sum_{j=1}^{N_y} w_j\,K(x,y_j)\,g(u(y_j)).
    $$

    **Arguments:**

    - `u`: Input field $u$.
    - `quad`: A dict with keys `"points"` (shape `(N_y, d)`) and `"weights"` (shape `(N_y,)`).
    - `kernel`: Optional kernel $K$. If provided, it is evaluated as
      `kernel(concat([x, y]))` with input in $\mathbb{R}^{2d}$.
    - `nonlinearity`: Optional nonlinearity $g$ applied to $u(y)$ before integration.
    - `importance_weight`: Optional extra factor $M(x,y)$ multiplied into the quadrature weights.
    - `var`: Label of the spatial variable (default `"x"`).
    - `time_var`: Optional time label to include as an additional dependency.

    **Notes:**

    - This operator does not support coord-separable inputs for `var`.
    """
    _, var_dim = _factor_and_dim(u, var)

    y = jnp.asarray(quad["points"], dtype=float)
    w = jnp.asarray(quad["weights"], dtype=float)
    if y.ndim != 2 or y.shape[1] != int(var_dim):
        raise ValueError(
            f"quad['points'] must have shape (Ny, {int(var_dim)}), got {y.shape}."
        )
    if w.ndim != 1 or w.shape[0] != y.shape[0]:
        raise ValueError(
            "quad['weights'] must be a 1D array matching quad['points'].shape[0]."
        )

    required: list[str] = []
    for lbl in u.domain.labels:
        if lbl in u.deps:
            required.append(lbl)
    if var not in required:
        required.append(var)
    if time_var is not None and time_var in u.domain.labels and time_var not in required:
        required.append(time_var)

    deps = tuple(lbl for lbl in u.domain.labels if lbl in required)
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u.deps)

    x_pos = idx.get(var)
    if x_pos is None:
        raise ValueError(
            f"spatial_integral requires var {var!r} to be present in dependencies."
        )

    t_pos = None
    if time_var is not None and time_var in deps:
        t_pos = idx[time_var]

    u_var_idx = u.deps.index(var) if var in u.deps else None

    def _eval_g(val: Any) -> Any:
        return nonlinearity(val) if callable(nonlinearity) else val

    def _op(*args, key=None, **kwargs):
        x = args[x_pos]
        if isinstance(x, tuple):
            raise ValueError("spatial_integral does not support coord-separable inputs.")
        x = jnp.asarray(x, dtype=float)

        t = None
        if t_pos is not None:
            t = jnp.asarray(args[t_pos], dtype=float).reshape(())

        u_args = [args[i] for i in u_pos]

        def per_y(y_sp):
            call_args = list(u_args)
            if u_var_idx is not None:
                call_args[u_var_idx] = y_sp
            return u.func(*call_args, key=key, **kwargs)

        uy = _eval_g(jax.vmap(per_y)(y))

        if kernel is None:
            K = jnp.ones((y.shape[0],), dtype=float)
        else:
            z = jnp.concatenate([jnp.broadcast_to(x[None, :], y.shape), y], axis=1)
            K = jax.vmap(kernel)(z)

        if importance_weight is None:
            Mw = 1.0
        else:
            Mw = jax.vmap(lambda yj: importance_weight(x, yj))(y)

        weights = w * K * Mw
        return jnp.tensordot(weights, uy, axes=(0, 0))

    return DomainFunction(domain=u.domain, deps=deps, func=_op, metadata=u.metadata)


def nonlocal_integral(
    u: DomainFunction,
    /,
    *,
    integrand: Callable,
    quad: dict,
    importance_weight: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    var: str = "x",
    time_var: str | None = None,
) -> DomainFunction:
    r"""Nonlocal integral operator with a context-aware integrand.

    Constructs a function

    $$
    v(x) = \int_{\Omega} \mathcal{I}(x,y,u(x),u(y),\dots)\,dy
    \approx \sum_{j=1}^{N_y} w_j\,\mathcal{I}(\cdot),
    $$

    where the integrand is provided as `integrand(ctx)` and is evaluated on a context
    dictionary containing (at least) the keys:

    - `"x"`: Full coordinate (including time if `time_var` is provided).
    - `"y"`: Full coordinate (including time if `time_var` is provided).
    - `"x_space"`: Spatial part of `x`.
    - `"y_space"`: Spatial part of `y`.
    - `"t"`: Scalar time or `None`.
    - `"ux"`: Value $u(x)$.
    - `"uy"`: Value $u(y)$.
    - `"du"`: $u(y)-u(x)$.
    - `"xi"`: displacement $y-x$ in space.

    **Arguments:**

    - `u`: Input function $u$.
    - `integrand`: A callable returning the integrand value given a `ctx` dict.
    - `quad`: A dict with `"points"` and `"weights"` as in `spatial_integral`.
    - `importance_weight`: Optional extra factor $M(x,y)$ multiplied into weights.
    - `var`: Spatial domain label used for the integral variable.
    - `time_var`: Optional time label included in `"x"`/`"y"` when present.
    """
    _, var_dim = _factor_and_dim(u, var)

    y_space = jnp.asarray(quad["points"], dtype=float)
    w = jnp.asarray(quad["weights"], dtype=float)
    if y_space.ndim != 2 or y_space.shape[1] != int(var_dim):
        raise ValueError(
            f"quad['points'] must have shape (Ny, {int(var_dim)}), got {y_space.shape}."
        )
    if w.ndim != 1 or w.shape[0] != y_space.shape[0]:
        raise ValueError(
            "quad['weights'] must be a 1D array matching quad['points'].shape[0]."
        )

    call_integrand = _compile_ctx_integrand(integrand)

    required: list[str] = []
    for lbl in u.domain.labels:
        if lbl in u.deps:
            required.append(lbl)
    if var not in required:
        required.append(var)
    if time_var is not None and time_var in u.domain.labels and time_var not in required:
        required.append(time_var)

    deps = tuple(lbl for lbl in u.domain.labels if lbl in required)
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u.deps)

    x_pos = idx.get(var)
    if x_pos is None:
        raise ValueError(
            f"nonlocal_integral requires var {var!r} to be present in dependencies."
        )

    t_pos = None
    if time_var is not None and time_var in deps:
        t_pos = idx[time_var]

    u_var_idx = u.deps.index(var) if var in u.deps else None

    def _op(*args, key=None, **kwargs):
        x_sp = args[x_pos]
        if isinstance(x_sp, tuple):
            raise ValueError("nonlocal_integral does not support coord-separable inputs.")
        x_sp = jnp.asarray(x_sp, dtype=float)

        t = None
        if t_pos is not None:
            t = jnp.asarray(args[t_pos], dtype=float).reshape(())

        u_args = [args[i] for i in u_pos]
        ux = u.func(*u_args, key=key, **kwargs)

        def per_y(y_sp):
            call_args = list(u_args)
            if u_var_idx is not None:
                call_args[u_var_idx] = y_sp
            return u.func(*call_args, key=key, **kwargs)

        uy = jax.vmap(per_y)(y_space)

        if importance_weight is None:
            Mw = 1.0
        else:
            Mw = jax.vmap(lambda yj: importance_weight(x_sp, yj))(y_space)

        def per_ctx(y_sp_i, uy_i):
            if t is None:
                x_full = x_sp
                y_full = y_sp_i
            else:
                x_full = jnp.concatenate([x_sp, jnp.atleast_1d(t)])
                y_full = jnp.concatenate([y_sp_i, jnp.atleast_1d(t)])

            ctx = {
                "x": x_full,
                "y": y_full,
                "x_space": x_sp,
                "y_space": y_sp_i,
                "t": t,
                "ux": ux,
                "uy": uy_i,
                "du": uy_i - ux,
                "xi": y_sp_i - x_sp,
            }
            return call_integrand(ctx)

        vals = jax.vmap(per_ctx)(y_space, uy)
        return jnp.tensordot(w * Mw, vals, axes=(0, 0))

    return DomainFunction(domain=u.domain, deps=deps, func=_op, metadata=u.metadata)


__all__ = [
    "nonlocal_integral",
    "spatial_integral",
]
