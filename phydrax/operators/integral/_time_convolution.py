#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._doc import DOC_KEY0
from ...domain._domain import RelabeledDomain
from ...domain._function import DomainFunction
from ...domain._sampling import get_sampler
from ...domain._scalar import _AbstractScalarDomain


def _unwrap_factor(factor: object, /) -> object:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _time_start(u: DomainFunction, time_var: str) -> Array:
    factor = _unwrap_factor(u.domain.factor(time_var))
    if isinstance(factor, _AbstractScalarDomain):
        return jnp.asarray(factor.fixed("start"), dtype=float)
    return jnp.array(0.0, dtype=float)


def _legendre_nodes_weights(n: int) -> tuple[Array, Array]:
    from numpy.polynomial.legendre import leggauss

    xs, ws = leggauss(int(n))
    return jnp.asarray(xs, dtype=float), jnp.asarray(ws, dtype=float)


def time_convolution(
    k: Callable[[Array], ArrayLike],
    u: DomainFunction,
    /,
    *,
    time_var: str = "t",
    order: int = 48,
    cluster_exponent: float = 1.0,
    mode: Literal["gl", "qmc", "gmc_1d"] = "gl",
    sampler: str = "sobol_scrambled",
    kernel_exponent: float | None = None,
) -> DomainFunction:
    r"""Time convolution operator on a labeled time coordinate.

    Constructs the causal convolution

    $$
    (k * u)(t) = \int_{t_0}^{t} k(t-s)\,u(s)\,ds,
    $$

    where $t_0$ is the start of the time interval when `time_var` is a `TimeInterval`
    factor (otherwise $t_0=0$).

    The integral is approximated using one of:
    - `mode="gl"`: Gauss–Legendre quadrature on $[t_0,t]$ (optionally clustered near $t_0$);
    - `mode="qmc"`: quasi Monte Carlo on $[t_0,t]$;
    - `mode="gmc_1d"`: importance sampling for weakly singular kernels
      (requires `kernel_exponent` $\gamma$).

    **Arguments:**

    - `k`: Kernel $k(\tau)$ evaluated at $\tau=t-s$.
    - `u`: Input function $u(s)$.
    - `time_var`: Label for the time coordinate (default `"t"`).
    - `order`: Number of quadrature/MC samples.
    - `cluster_exponent`: When using Gauss–Legendre, applies a power-law transform to
      cluster nodes toward the start of the interval.
    - `sampler`: QMC sampler name used for `mode="qmc"` / `mode="gmc_1d"`.
    - `kernel_exponent`: Exponent $\gamma$ used in `mode="gmc_1d"` for sampling
      $\tau^{-\gamma}$-type singularities.
    """
    if time_var not in u.domain.labels:
        raise ValueError(
            f"time_convolution requires time_var {time_var!r} in the function domain."
        )

    t0 = _time_start(u, time_var)

    xs, ws = _legendre_nodes_weights(int(order))
    r = (xs + 1.0) / 2.0

    ce = float(cluster_exponent)
    if ce != 1.0:
        r = jnp.power(r, ce)
        jac = 0.5 * ce * jnp.power(r, ce - 1.0)
        W = ws * jac
    else:
        W = ws * 0.5

    sampler_fn = get_sampler(sampler)

    required = list(u.deps)
    if time_var not in required:
        required.append(time_var)
    deps = tuple(lbl for lbl in u.domain.labels if lbl in required)
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u.deps)
    t_pos = idx.get(time_var)
    if t_pos is None:
        raise ValueError(
            "time_convolution requires time_var to be present in dependencies."
        )

    u_time_idx = u.deps.index(time_var) if time_var in u.deps else None

    def _u_at_time(u_args: list[object], tt: Array, *, key, **kwargs):
        call_args = list(u_args)
        if u_time_idx is not None:
            call_args[u_time_idx] = tt
        return u.func(*call_args, key=key, **kwargs)

    def _op(*args, key=None, **kwargs):
        if key is None:
            key = DOC_KEY0

        t = jnp.asarray(args[t_pos], dtype=float).reshape(())
        dt = jnp.maximum(t - t0, 0.0)
        dt_safe = jnp.maximum(dt, 1e-12)

        u_args = [args[i] for i in u_pos]

        if mode == "gl":
            s = t0 + dt_safe * r
            tau = t - s
            U = jax.vmap(lambda si: _u_at_time(u_args, si, key=key, **kwargs))(s)
            K = jax.vmap(lambda ti: k(ti))(tau)
            w_eff = (W * dt_safe) * K
            return jnp.tensordot(w_eff, U, axes=(0, 0))

        if mode == "qmc":
            F = sampler_fn(int(order), 1, key).squeeze(-1)
            s = t0 + dt_safe * F
            tau = t - s
            U = jax.vmap(lambda si: _u_at_time(u_args, si, key=key, **kwargs))(s)
            K = jax.vmap(lambda ti: k(ti))(tau)
            Kb = K.reshape((K.shape[0],) + (1,) * (U.ndim - 1))
            return jnp.mean(U * Kb, axis=0) * dt_safe

        if mode == "gmc_1d":
            if kernel_exponent is None:
                raise ValueError("kernel_exponent is required for mode='gmc_1d'.")
            gamma = float(kernel_exponent)
            Uu = sampler_fn(int(order), 1, key).squeeze(-1)
            tau = dt_safe * jnp.power(jnp.clip(Uu, 1e-12, 1.0), 1.0 / (1.0 - gamma))
            s = t - tau
            U = jax.vmap(lambda si: _u_at_time(u_args, si, key=key, **kwargs))(s)
            q = (
                (1.0 - gamma)
                * jnp.power(jnp.maximum(tau, 1e-12), -gamma)
                / jnp.power(dt_safe, 1.0 - gamma)
            )
            K = jax.vmap(lambda ti: k(ti))(tau)
            Wi = K / (q + 1e-12)
            Wb = Wi.reshape((Wi.shape[0],) + (1,) * (U.ndim - 1))
            return jnp.mean(U * Wb, axis=0)

        raise ValueError(f"Unsupported mode {mode!r}.")

    return DomainFunction(domain=u.domain, deps=deps, func=_op, metadata=u.metadata)


__all__ = [
    "time_convolution",
]
