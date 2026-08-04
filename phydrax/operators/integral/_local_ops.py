#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ..._sampling import unit_design
from ...domain._function import DomainFunction
from ..differential._domain_ops import _factor_and_dim
from ._ctx import _compile_ctx_integrand


def _uniform_ball_rule(
    radius: float,
    dimension: int,
    count: int,
    /,
) -> dict[str, jax.Array]:
    """Build a deterministic equal-weight Halton rule on a Euclidean ball."""
    radius_ = float(radius)
    dimension_ = int(dimension)
    count_ = int(count)
    if radius_ <= 0.0 or dimension_ <= 0 or count_ <= 0:
        raise ValueError("radius, dimension, and count must be positive.")
    design = unit_design("halton", count=count_, dimension=dimension_ + 1, key=None)
    probability = jnp.clip(
        design[:, :dimension_],
        jnp.finfo(float).eps,
        1.0 - jnp.finfo(float).eps,
    )
    direction = jsp.special.ndtri(probability)
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    radial = radius_ * jnp.power(design[:, -1], 1.0 / dimension_)
    offsets = radial[:, None] * direction
    volume = (
        math.pi ** (0.5 * dimension_)
        * radius_**dimension_
        / math.gamma(0.5 * dimension_ + 1.0)
    )
    return {
        "offsets": offsets,
        "weights": jnp.full((count_,), volume / count_, dtype=float),
    }


def local_integral(
    u: DomainFunction,
    /,
    *,
    integrand: Callable,
    ball_quad: dict,
    var: str = "x",
    time_var: str | None = None,
) -> DomainFunction:
    r"""Local (ball) integral operator around each query point.

    Constructs a function of the form

    $$
    v(x) = \int_{B_\delta(0)} \mathcal{I}(x, x+\xi, u(x), u(x+\xi), \xi, \dots)\,d\xi,
    $$

    approximated using a fixed quadrature rule on offsets $\xi_j$:

    $$
    v(x) \approx \sum_{j=1}^{N_\xi} w_j\,\mathcal{I}(x, x+\xi_j, u(x), u(x+\xi_j), \xi_j, \dots).
    $$

    The integrand may receive the whole context through a single parameter named
    `ctx` or `context`, or select context values by matching parameter names. Available
    keys include `"x"`, `"y"`, `"ux"`, `"uy"`, `"du"`, and `"xi"`. For compatibility,
    `value`, `delta`/`delta_value`, and `displacement` select `"uy"`, `"du"`, and
    `"xi"`, respectively.

    **Notes:**

    - This operator does not support coord-separable inputs for `var`.
    """
    factor, var_dim = _factor_and_dim(u, var)
    del factor

    offsets = jnp.asarray(ball_quad["offsets"], dtype=float)  # (Ny, d)
    w = jnp.asarray(ball_quad["weights"], dtype=float)  # (Ny,)

    if offsets.ndim != 2 or offsets.shape[1] != int(var_dim):
        raise ValueError(
            f"ball_quad['offsets'] must have shape (Ny, {int(var_dim)}), got {offsets.shape}."
        )
    if w.ndim != 1 or w.shape[0] != offsets.shape[0]:
        raise ValueError(
            "ball_quad['weights'] must be a 1D array matching offsets.shape[0]."
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
            f"local_integral requires var {var!r} to be present in dependencies."
        )

    t_pos = None
    if time_var is not None and time_var in deps:
        t_pos = idx[time_var]

    u_var_idx = u.deps.index(var) if var in u.deps else None

    def _op(*args, key=None, **kwargs):
        x = args[x_pos]
        if isinstance(x, tuple):
            raise ValueError("local_integral does not support coord-separable inputs.")
        x = jnp.asarray(x, dtype=float)

        t = None
        if t_pos is not None:
            t = jnp.asarray(args[t_pos], dtype=float).reshape(())

        u_args = [args[i] for i in u_pos]
        ux = u.func(*u_args, key=key, **kwargs)

        y_space = x[None, :] + offsets

        def per_y(y_sp):
            call_args = list(u_args)
            if u_var_idx is not None:
                call_args[u_var_idx] = y_sp
            return u.func(*call_args, key=key, **kwargs)

        uy = jax.vmap(per_y)(y_space)

        def per_ctx(y_sp, uy_i):
            if t is None:
                x_full = x
                y_full = y_sp
            else:
                x_full = jnp.concatenate([x, jnp.atleast_1d(t)])
                y_full = jnp.concatenate([y_sp, jnp.atleast_1d(t)])

            ctx = {
                "x": x_full,
                "y": y_full,
                "x_space": x,
                "y_space": y_sp,
                "t": t,
                "ux": ux,
                "uy": uy_i,
                "du": uy_i - ux,
                "xi": y_sp - x,
            }
            return call_integrand(ctx)

        vals = jax.vmap(per_ctx)(y_space, uy)
        return jnp.tensordot(w, vals, axes=(0, 0))

    return DomainFunction(domain=u.domain, deps=deps, func=_op, metadata=u.metadata)


def local_integral_ball(
    u: DomainFunction,
    f_bond: Callable,
    *,
    ball_quad: dict,
    var: str = "x",
    time_var: str | None = None,
) -> DomainFunction:
    r"""Convenience wrapper for a bond-based local integral.

    Uses an integrand of the form $\mathcal{I} = f(\Delta u, \xi)$ with
    $\Delta u = u(y)-u(x)$ and $\xi=y-x$.
    """
    return local_integral(
        u,
        integrand=lambda du, xi: f_bond(du, xi),
        ball_quad=ball_quad,
        var=var,
        time_var=time_var,
    )


__all__ = [
    "local_integral",
    "local_integral_ball",
]
