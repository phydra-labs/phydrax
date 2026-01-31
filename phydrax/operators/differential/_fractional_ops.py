#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import opt_einsum as oe

from ..._doc import DOC_KEY0
from ...domain._function import DomainFunction
from ...domain._sampling import get_sampler
from ...domain._scalar import _AbstractScalarDomain
from ..integral._quadrature import build_ball_quadrature
from ._domain_ops import _factor_and_dim, _resolve_var, grad


def fractional_laplacian(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    var: str | None = None,
    radius: float | None = None,
    num_points: int = 4096,
    eps: float = 1e-6,
    desingularize: bool = False,
    angular_points: int | None = None,
) -> DomainFunction:
    r"""Fractional Laplacian via a truncated ball quadrature estimator.

    The (spectral/integral) fractional Laplacian in $\mathbb{R}^d$ admits an integral
    representation (up to a normalization constant $C_{d,\alpha}$):

    $$
    (-\Delta)^{\alpha/2}u(x)
    \propto \int_{\mathbb{R}^d}\frac{u(x)-u(y)}{\|x-y\|^{d+\alpha}}\,dy,
    \qquad 0<\alpha<2.
    $$

    This implementation uses a *truncated* ball integral with radius `radius` and a
    Monte Carlo-style quadrature rule on offsets $y=x+\xi$:

    $$
    \int_{B_R(0)} \frac{u(x)-u(x+\xi)}{\|\xi\|^{d+\alpha}}\,d\xi.
    $$

    This routine returns a quantity proportional to $(-\Delta)^{\alpha/2}u$; it does
    not apply the normalization constant $C_{d,\alpha}$ and truncates the integral
    to a finite ball.

    **Arguments:**

    - `u`: Input function $u$.
    - `alpha`: Fractional order $\alpha\in(0,2)$.
    - `var`: Geometry label to apply the operator to. If `None`, Phydrax infers a
      differentiable geometry label when possible.
    - `radius`: Truncation radius $R$ (defaults to an AABB-derived scale).
    - `num_points`: Number of ball quadrature offsets.
    - `eps`: Excludes offsets with $\|\xi\|\le \varepsilon$ to avoid the singularity.
    - `desingularize`: If `True` and $\alpha>1$, subtracts a first-order correction
      term involving $\nabla u$ to reduce variance.
    - `angular_points`: Reserved for future deterministic angular quadrature; currently unused.

    **Returns:**

    A `DomainFunction` representing the (truncated, unnormalized) fractional Laplacian.
    """
    del angular_points
    a = float(alpha)
    if not (0.0 < a < 2.0):
        raise ValueError("alpha must be in (0,2).")

    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "fractional_laplacian(var=...) requires a geometry variable, not a scalar variable."
        )

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            return jnp.zeros_like(y)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    if radius is None:
        bounds = jnp.asarray(factor.mesh_bounds, dtype=float)
        mins = bounds[0]
        maxs = bounds[1]
        radius = float(jnp.linalg.norm(maxs - mins) + 1e-12)
    R = float(radius)

    bq = build_ball_quadrature(radius=R, dim=int(var_dim), num_points=int(num_points))
    offsets = jnp.asarray(bq["offsets"], dtype=float)  # (N, d)
    w = jnp.asarray(bq["weights"], dtype=float)  # (N,)
    r_off = jnp.linalg.norm(offsets, axis=1)  # (N,)
    r_safe = jnp.maximum(r_off, 1e-12)
    kern = jnp.power(r_safe, -(var_dim + a)) * (r_off > float(eps)).astype(float)

    idx = u.deps.index(var)
    grad_u = grad(u, var=var, mode="forward") if desingularize and a > 1.0 else None

    def _op(*args, key=None, **kwargs):
        x = args[idx]
        if isinstance(x, tuple):
            raise ValueError(
                "fractional_laplacian does not support coord-separable inputs."
            )
        x = jnp.asarray(x, dtype=float)

        ux = u.func(*args, key=key, **kwargs)

        y_sp = x[None, :] + offsets

        def per_y(yv):
            call_args = list(args)
            call_args[idx] = yv
            return u.func(*call_args, key=key, **kwargs)

        uy = jax.vmap(per_y)(y_sp)

        if grad_u is not None:
            gu = jnp.asarray(grad_u.func(*args, key=key, **kwargs))
            corr = oe.contract("nd,...d->n...", offsets, gu)
            num = (ux - uy) + corr
        else:
            num = ux - uy

        weights = w * kern
        return jnp.tensordot(weights, num, axes=(0, 0))

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def _gmc_cdf(alpha: float, *, max_k: int = 100000, tol: float = 1e-12):
    if not (1.0 < alpha < 2.0):
        raise ValueError("GMC CDF requires alpha in (1,2).")
    p = []
    p1 = float(jnp.clip(2.0 - alpha, 1e-15, 1 - 1e-15))
    p.append(p1)
    total = p1
    m_prev = -alpha
    k = 2
    while total < 1.0 - tol and k <= max_k:
        mk = -m_prev * (alpha - (k - 1)) / k
        m_prev = mk
        mkp = float(jnp.maximum(mk, 0.0))
        if mkp < 1e-20:
            break
        p.append(mkp)
        total += mkp
        k += 1
    E = jnp.cumsum(jnp.array(p))
    if float(E[-1]) != 1.0:
        E = E.at[-1].add(1.0 - float(E[-1]))
    return E


def fractional_derivative_gl_mc(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    side: Literal["left", "right"] = "right",
    axis: int = 0,
    N: int = 100,
    K: int = 1,
    x_lb: float | None = None,
    x_ub: float | None = None,
    sampler: str = "sobol_scrambled",
    caputo: bool = False,
    var: str | None = None,
    time_var: str | None = "t",
) -> DomainFunction:
    r"""Grünwald–Letnikov fractional derivative (Monte Carlo / GMC estimator).

    For $\alpha\in(1,2)$, the Grünwald–Letnikov derivative can be viewed as a limit of
    weighted finite differences. This routine implements a Monte Carlo / generalized
    Monte Carlo (GMC) estimator for left- or right-sided fractional derivatives along
    a chosen spatial axis.

    The derivative is taken with respect to a geometry variable `var` along coordinate
    axis `axis`, using bounds `x_lb`/`x_ub` to define the one-sided interval.

    **Arguments:**

    - `u`: Input `DomainFunction` $u$.
    - `alpha`: Order $\alpha\in(1,2)$.
    - `side`: `"left"` or `"right"`.
    - `axis`: Spatial axis within `var` used for the one-sided derivative.
    - `N`: Number of sub-intervals used to define the step size $h$.
    - `K`: Sample multiplier; total Monte Carlo samples is `N * K`.
    - `x_lb`: Left bound (required when `side="left"`).
    - `x_ub`: Right bound (required when `side="right"`).
    - `sampler`: Sampler name for drawing the GMC random variates.
    - `caputo`: If `True`, uses a Caputo-style correction.
    - `var`: Geometry label to differentiate with respect to (defaults to an inferred geometry label).
    - `time_var`: Optional time label; when present in `u.deps`, the derivative is
      conditioned on the current time coordinate.
    """
    a = float(alpha)
    if not (1.0 < a < 2.0):
        raise ValueError("fractional_derivative_gl_mc currently supports alpha in (1,2).")

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    if side == "right" and x_ub is None:
        raise ValueError("x_ub is required for right-sided GMC derivative.")
    if side == "left" and x_lb is None:
        raise ValueError("x_lb is required for left-sided GMC derivative.")

    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "fractional_derivative_gl_mc(var=...) requires a geometry variable, not a scalar variable."
        )

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            return jnp.zeros_like(y)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    axis_i = int(axis)
    if not (0 <= axis_i < int(var_dim)):
        raise ValueError(f"axis must be in [0,{int(var_dim)}).")

    use_time = (time_var is not None) and (time_var in u.domain.labels)
    if use_time and time_var is None:
        raise ValueError("time_var cannot be None when use_time is True.")
    if use_time and time_var not in u.domain.labels:
        raise ValueError(f"time_var {time_var!r} not in domain.")

    if use_time and time_var not in u.deps:
        use_time = False

    E = _gmc_cdf(a)
    sampler_fn = get_sampler(sampler)

    def map_F_to_Y(F):
        F1 = jnp.squeeze(F, axis=-1)
        idx = jnp.searchsorted(E, F1, side="right")
        return idx + 1

    idx_x = u.deps.index(var)
    idx_t = u.deps.index(time_var) if use_time else None

    total_samples = int(N) * max(int(K), 1)

    def _per_point_base(*args, key, **kwargs):
        x = args[idx_x]
        if isinstance(x, tuple):
            raise ValueError(
                "fractional_derivative_gl_mc does not support coord-separable inputs."
            )
        x = jnp.asarray(x, dtype=float)

        t = None
        if use_time:
            assert idx_t is not None
            t = jnp.asarray(args[idx_t], dtype=float).reshape(())

        xi = x[axis_i]
        if side == "right":
            assert x_ub is not None
            h = (float(x_ub) - xi) / float(N)
            sgn = 1.0
        else:
            assert x_lb is not None
            h = (xi - float(x_lb)) / float(N)
            sgn = -1.0

        fx0 = u.func(*args, key=key, **kwargs)
        cond_bad = jnp.logical_or(~jnp.isfinite(h), h <= 0)

        def _zero():
            return jnp.zeros_like(fx0)

        def _cont():
            # deterministic x+h
            x_h = x.at[axis_i].set(xi + sgn * h)
            call_args = list(args)
            call_args[idx_x] = x_h
            fxh = u.func(*call_args, key=key, **kwargs)

            F = sampler_fn(total_samples, 1, key)
            Y = map_F_to_Y(F).astype(float)

            xs = jnp.repeat(x[None, :], total_samples, axis=0)
            xs = xs.at[:, axis_i].add(sgn * Y * h)

            def per_y(xy):
                call_args = list(args)
                call_args[idx_x] = xy
                return u.func(*call_args, key=key, **kwargs)

            fY = jax.vmap(per_y)(xs)
            mean_fY = jnp.mean(fY, axis=0)

            if not caputo:
                return (fx0 - 2.0 * fxh + mean_fY) / (h**a)

            e_axis = jnp.zeros_like(x).at[axis_i].set(1.0)

            def f_single(z):
                call_args = list(args)
                call_args[idx_x] = z
                return u.func(*call_args, key=key, **kwargs)

            fx_axis = jax.jvp(f_single, (x,), (e_axis,))[1]
            delta_h = sgn * h
            gxh = fxh - fx0 - fx_axis * delta_h
            deltas = sgn * Y * h
            deltas_b = deltas.reshape(deltas.shape + (1,) * jnp.ndim(fx_axis))
            gY = fY - fx0 - deltas_b * fx_axis
            mean_gY = jnp.mean(gY, axis=0)
            return (0.0 - 2.0 * gxh + mean_gY) / (h**a)

        return jax.lax.cond(cond_bad, _zero, _cont)

    def _op(*args, key=None, **kwargs):
        if key is None:
            key = DOC_KEY0
        return _per_point_base(*args, key=key, **kwargs)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def riesz_fractional_derivative_gl_mc(
    u: DomainFunction,
    /,
    *,
    beta: float,
    bounds: Sequence[tuple[float, float]],
    N: int = 100,
    K: int = 1,
    sampler: str = "sobol_scrambled",
    var: str | None = None,
    time_var: str | None = "t",
) -> DomainFunction:
    r"""Riesz fractional derivative assembled from one-sided GL derivatives.

    For $\beta\in(1,2)$, this constructs a Riesz-type fractional derivative by summing
    left and right Grünwald–Letnikov derivatives along each spatial axis and applying
    the standard normalization factor:

    $$
    \mathcal{D}^{\beta}u \propto \frac{1}{2\cos(\pi\beta/2)}\sum_{i=1}^{d}
    \left(D_{+,i}^{\beta}u + D_{-,i}^{\beta}u\right).
    $$

    The per-axis bounds are provided via `bounds`.

    **Arguments:**

    - `u`: Input `DomainFunction` $u$.
    - `beta`: Order $\beta\in(1,2)$.
    - `bounds`: Sequence of per-axis bounds `[(x_lb_0, x_ub_0), ..., (x_lb_{d-1}, x_ub_{d-1})]`.
    - `N`: Number of sub-intervals used to define step sizes $h$ per axis.
    - `K`: Sample multiplier; total Monte Carlo samples per one-sided derivative is `N * K`.
    - `sampler`: Sampler name for drawing the GMC random variates.
    - `var`: Geometry label to differentiate with respect to (defaults to an inferred geometry label).
    - `time_var`: Optional time label; when present in `u.deps`, the derivative is conditioned on time.
    """
    b = float(beta)
    if not (1.0 < b < 2.0):
        raise ValueError("beta must be in (1,2).")

    var = _resolve_var(u, var)
    _, var_dim = _factor_and_dim(u, var)
    if len(bounds) != int(var_dim):
        raise ValueError(f"bounds must have length {int(var_dim)}.")

    fac = 1.0 / (2.0 * jnp.cos(jnp.pi * b / 2.0))

    ops = []
    for ax, (lb, ub) in enumerate(bounds):
        D_left = fractional_derivative_gl_mc(
            u,
            alpha=b,
            side="left",
            axis=ax,
            N=N,
            K=K,
            x_lb=lb,
            x_ub=None,
            sampler=sampler,
            caputo=False,
            var=var,
            time_var=time_var,
        )
        D_right = fractional_derivative_gl_mc(
            u,
            alpha=b,
            side="right",
            axis=ax,
            N=N,
            K=K,
            x_lb=None,
            x_ub=ub,
            sampler=sampler,
            caputo=False,
            var=var,
            time_var=time_var,
        )
        ops.append(D_left + D_right)

    out = ops[0]
    for op in ops[1:]:
        out = out + op
    return fac * out
