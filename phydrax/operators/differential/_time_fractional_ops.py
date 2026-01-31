#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from ..._doc import DOC_KEY0
from ...domain._function import DomainFunction
from ...domain._sampling import get_sampler
from ...domain._scalar import _AbstractScalarDomain
from ._domain_ops import _unwrap_factor


def _time_start(u: DomainFunction, time_var: str) -> jax.Array:
    factor = _unwrap_factor(u.domain.factor(time_var))
    if isinstance(factor, _AbstractScalarDomain):
        return jnp.asarray(factor.fixed("start"), dtype=float)
    raise TypeError(f"time_var {time_var!r} is not a scalar domain label.")


def _legendre_nodes_weights(n: int) -> tuple[jax.Array, jax.Array]:
    from numpy.polynomial.legendre import leggauss

    xs, ws = leggauss(int(n))
    return jnp.asarray(xs, dtype=float), jnp.asarray(ws, dtype=float)


def caputo_time_fractional(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    time_var: str = "t",
    mode: str = "auto",
    sampler: str = "sobol_scrambled",
    order: int = 64,
    tau_epsilon: float | str = "auto",
    cluster_exponent: float | None = None,
) -> DomainFunction:
    r"""Caputo fractional derivative in time.

    For a time interval starting at $t_0$, the Caputo derivative of order
    $\alpha\in(0,2)$ is defined (for sufficiently smooth $u$) by

    For $0<\alpha<1$:

    $$
    {}^C D_t^{\alpha}u(t)=\frac{1}{\Gamma(1-\alpha)}\int_{t_0}^{t}\frac{u'(s)}{(t-s)^{\alpha}}\,ds;
    $$

    For $1<\alpha<2$:

    $$
    {}^C D_t^{\alpha}u(t)=\frac{1}{\Gamma(2-\alpha)}\int_{t_0}^{t}\frac{u''(s)}{(t-s)^{\alpha-1}}\,ds.
    $$

    This implementation uses quadrature/Monte Carlo estimators depending on `alpha`
    and `mode`.

    **Arguments:**

    - `alpha`: Fractional order $\alpha\in(0,2)$.
    - `time_var`: Time label (must correspond to a scalar domain factor).
    - `mode`: `"auto"` selects a reasonable default; `"gj"` uses Gauss–Jacobi nodes;
      `"gl"` uses Gauss–Legendre; other values fall back to sampling.
    - `order`: Number of quadrature/sampling points.
    - `tau_epsilon`: Regularization for the kernel singularity near $t=s$.
    - `cluster_exponent`: Optional clustering of Gauss–Legendre nodes toward $t_0$
      for the $0<\alpha<1$ branch.
    """
    a = float(alpha)
    if not (0.0 < a < 2.0):
        raise ValueError("alpha must be in (0,2).")
    if time_var not in u.domain.labels:
        raise KeyError(f"time_var {time_var!r} not in domain {u.domain.labels}.")

    t0 = _time_start(u, time_var)

    idx_t = u.deps.index(time_var) if time_var in u.deps else None

    def d_t_at(args, *, key, **kwargs):
        if idx_t is None:
            return jnp.zeros_like(u.func(*args, key=key, **kwargs))
        t = args[idx_t]

        def f(tt):
            call_args = list(args)
            call_args[idx_t] = tt
            return u.func(*call_args, key=key, **kwargs)

        _, dt_val = jax.jvp(f, (t,), (jnp.array(1.0, dtype=jnp.asarray(t).dtype),))
        return dt_val

    if 0.0 < a < 1.0:
        denom = jsp.gamma(1.0 - a)

        xs, ws = _legendre_nodes_weights(order)
        r = (xs + 1.0) / 2.0

        if cluster_exponent is not None and float(cluster_exponent) != 1.0:
            ce = float(cluster_exponent)
            r = jnp.power(r, ce)
            jac = 0.5 * ce * jnp.power(r, ce - 1.0)
            W = ws * jac
        else:
            W = ws * 0.5

        def _op(*args, key=None, **kwargs):
            if key is None:
                key = DOC_KEY0

            if idx_t is None:
                return jnp.zeros_like(u.func(*args, key=key, **kwargs))

            t = jnp.asarray(args[idx_t], dtype=float).reshape(())
            dt = t - t0
            dt = jnp.maximum(dt, 0.0)
            dt_safe = jnp.maximum(dt, 1e-12)

            s = t0 + dt_safe * r
            tau = t - s
            k_tau = jnp.power(jnp.maximum(tau, 1e-12), -a) / denom

            def du_at(si):
                call_args = list(args)
                call_args[idx_t] = si
                return d_t_at(tuple(call_args), key=key, **kwargs)

            du_vals = jax.vmap(du_at)(s)

            w_eff = (W * dt_safe) * k_tau
            return jnp.tensordot(w_eff, du_vals, axes=(0, 0))

        return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)

    if not (1.0 < a < 2.0):
        raise ValueError("alpha must be in (0,1) or (1,2) for Caputo derivative.")

    _mode = "gj" if mode == "auto" else str(mode)
    inv_gamma = 1.0 / jsp.gamma(2.0 - a)

    sampler_fn = get_sampler(sampler)

    def _tau_eps_val():
        if isinstance(tau_epsilon, str) and tau_epsilon == "auto":
            return 1e-3 * (2.0 - a) + 1e-8
        return float(tau_epsilon)

    def _samples_tau(key, n) -> jax.Array:
        U = sampler_fn(int(n), 1, key)
        return jnp.power(jnp.clip(U.squeeze(-1), 1e-12, 1.0 - 1e-12), 1.0 / (2.0 - a))

    def _gj_quadrature(M: int) -> tuple[jax.Array, jax.Array]:
        from scipy.special import roots_jacobi

        alpha_j = 1.0 - a
        beta_j = 0.0
        t, w = roots_jacobi(int(M), alpha_j, beta_j)
        x = (t + 1.0) / 2.0
        scale = 1.0 / (2.0 ** (2.0 - a))
        w_beta = w * scale
        w_expect = (2.0 - a) * w_beta
        return jnp.asarray(x, dtype=float), jnp.asarray(w_expect, dtype=float)

    tau_nodes: jax.Array | None = None
    tau_weights: jax.Array | None = None
    if _mode == "gj":
        tau_nodes, tau_weights = _gj_quadrature(int(order))

    def _gl_quadrature(M: int) -> tuple[jax.Array, jax.Array]:
        xs, ws = _legendre_nodes_weights(M)
        r = (xs + 1.0) / 2.0
        W = ws * 0.5
        return r, W

    def _per_point(*args, key, **kwargs):
        if idx_t is None:
            raise ValueError(
                "caputo_time_fractional(alpha>1) requires time_var in u.deps."
            )

        t = jnp.asarray(args[idx_t], dtype=float).reshape(())
        dt = t - t0
        dt = jnp.maximum(dt, 0.0)
        dt_safe = jnp.maximum(dt, 1e-12)

        fx = u.func(*args, key=key, **kwargs)
        dfx = d_t_at(args, key=key, **kwargs)

        args0 = list(args)
        args0[idx_t] = t0
        f0 = u.func(*args0, key=key, **kwargs)
        df0 = d_t_at(tuple(args0), key=key, **kwargs)

        if _mode == "gj":
            assert tau_nodes is not None and tau_weights is not None
            taus = tau_nodes
            wts = tau_weights
        elif _mode == "gl":
            taus, gl_W = _gl_quadrature(order)
            wts = gl_W
        else:
            taus = _samples_tau(key, order)
            wts = None

        dt_taus = dt_safe * taus
        eps = _tau_eps_val()
        denom = jnp.square(dt_safe * (taus + float(eps)))

        def u_at_tau(dt_tau):
            call_args = list(args)
            call_args[idx_t] = t - dt_tau
            return u.func(*call_args, key=key, **kwargs)

        f_tau = jax.vmap(u_at_tau)(dt_taus)

        dt_taus_b = dt_taus.reshape(dt_taus.shape + (1,) * jnp.ndim(dfx))
        numer = fx - f_tau - dt_taus_b * dfx
        denom_b = denom.reshape(denom.shape + (1,) * (numer.ndim - 1))
        q = numer / denom_b

        if _mode == "gl":
            pdf = (2.0 - a) * jnp.power(jnp.clip(taus, 1e-15, 1.0), 1.0 - a)
            assert wts is not None
            W_eff = wts * pdf
            W_eff_b = W_eff.reshape(W_eff.shape + (1,) * (q.ndim - 1))
            E_term = jnp.sum(W_eff_b * q, axis=0)
        elif wts is None:
            E_term = jnp.mean(q, axis=0)
        else:
            wts_b = wts.reshape(wts.shape + (1,) * (q.ndim - 1))
            E_term = jnp.sum(wts_b * q, axis=0)

        term1 = (dfx - df0) / jnp.power(dt_safe, a - 1.0)
        term2 = (a - 1.0) * (fx - f0 - dt * dfx) / jnp.power(dt_safe, a)
        term3 = (a * (a - 1.0) / (2.0 - a)) * jnp.power(dt_safe, 2.0 - a) * E_term
        return inv_gamma * (term1 - term2 - term3)

    def _op(*args, key=None, **kwargs):
        if key is None:
            key = DOC_KEY0
        return _per_point(*args, key=key, **kwargs)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_op, metadata=u.metadata)


def caputo_time_fractional_dw(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    time_var: str = "t",
    M: int = 128,
    sampler: str = "sobol_scrambled",
    tau_epsilon: float | str = "auto",
    mode: str = "gj",
) -> DomainFunction:
    r"""Convenience wrapper for `caputo_time_fractional` using `order=M`.

    Named for a common discretization setting in physics-informed fractional models.
    """
    return caputo_time_fractional(
        u,
        alpha=float(alpha),
        time_var=time_var,
        mode=mode,
        sampler=sampler,
        order=int(M),
        tau_epsilon=tau_epsilon,
    )
