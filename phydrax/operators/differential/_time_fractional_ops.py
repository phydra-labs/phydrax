#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from phydrax.domain import AbstractScalarDomain, DomainFunction

from ..._doc import DOC_KEY0
from ...integration import GaussLegendreRule
from .._causal_quadrature import causal_reference_rule
from ._domain_ops import _unwrap_factor


def _time_start(u: DomainFunction, time_var: str) -> jax.Array:
    factor = _unwrap_factor(u.domain.factor(time_var))
    if isinstance(factor, AbstractScalarDomain):
        return jnp.asarray(factor.fixed("start"), dtype=float)
    raise TypeError(f"time_var {time_var!r} is not a scalar domain label.")


def caputo_time_fractional(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    time_var: str = "t",
    mode: Literal["auto", "gj", "gl"] = "auto",
    order: int = 64,
    cluster_exponent: float | None = None,
) -> DomainFunction:
    r"""Deterministic Caputo fractional derivative in time.

    For $0<\alpha<1$ this evaluates

    $$
    {}^C D_t^\alpha u(t)=\frac{1}{\Gamma(1-\alpha)}
    \int_{t_0}^{t}(t-s)^{-\alpha}u'(s)\,ds.
    $$

    The $1<\alpha<2$ branch applies the same deterministic quadrature to
    the second time derivative. ``mode="auto"`` selects Gauss--Jacobi quadrature.
    Randomized fractional estimators require an estimator-aware randomized residual
    term and are not hidden in this operator.
    """
    a = float(alpha)
    if not (0.0 < a < 2.0):
        raise ValueError("alpha must be in (0,2).")
    if time_var not in u.domain.labels:
        raise KeyError(f"time_var {time_var!r} not in domain {u.domain.labels}.")
    resolved_mode = "gj" if mode == "auto" else str(mode)
    if resolved_mode not in ("gj", "gl"):
        raise ValueError("mode must be 'auto', 'gj', or 'gl'.")
    count = int(order)
    if count < 1:
        raise ValueError("order must be positive.")
    cluster = 1.0 if cluster_exponent is None else float(cluster_exponent)
    if resolved_mode == "gj" and cluster != 1.0:
        raise ValueError("cluster_exponent is only valid with mode='gl'.")

    t0 = _time_start(u, time_var)
    time_position = u.deps.index(time_var) if time_var in u.deps else None

    def d_t_at(args, *, key, **kwargs):
        if time_position is None:
            return jnp.zeros_like(u.func(*args, key=key, **kwargs))
        time = args[time_position]

        def evaluate(candidate_time):
            call_args = list(args)
            call_args[time_position] = candidate_time
            return u.func(*call_args, key=key, **kwargs)

        _, derivative = jax.jvp(
            evaluate,
            (time,),
            (jnp.array(1.0, dtype=jnp.asarray(time).dtype),),
        )
        return derivative

    def d2_t_at(args, *, key, **kwargs):
        if time_position is None:
            return jnp.zeros_like(u.func(*args, key=key, **kwargs))
        time = args[time_position]

        def first_derivative(candidate_time):
            call_args = list(args)
            call_args[time_position] = candidate_time
            return d_t_at(tuple(call_args), key=key, **kwargs)

        _, derivative = jax.jvp(
            first_derivative,
            (time,),
            (jnp.array(1.0, dtype=jnp.asarray(time).dtype),),
        )
        return derivative

    def with_metadata(function, *, method: str) -> DomainFunction:
        metadata = dict(u.metadata)
        metadata.update(
            {
                "differential_operator": "caputo-time-fractional",
                "fractional_alpha": a,
                "fractional_time_var": time_var,
                "fractional_method": method,
                "fractional_order": count,
                "fractional_cluster_exponent": cluster,
                "fractional_randomized": False,
            }
        )
        return DomainFunction(
            domain=u.domain,
            deps=u.deps,
            func=function,
            metadata=metadata,
        )

    if 0.0 < a < 1.0:
        gamma = jsp.gamma(1.0 - a)
        if resolved_mode == "gj":
            from scipy.special import roots_jacobi

            raw_nodes, raw_weights = roots_jacobi(count, -a, 0.0)
            nodes = (jnp.asarray(raw_nodes, dtype=float) + 1.0) / 2.0
            weights = jnp.asarray(raw_weights, dtype=float) * (2.0 ** (a - 1.0))
        else:
            nodes, weights = causal_reference_rule(
                GaussLegendreRule(count),
                cluster_exponent=cluster,
            )

        def _op(*args, key=None, **kwargs):
            evaluation_key = DOC_KEY0 if key is None else key
            if time_position is None:
                return jnp.zeros_like(u.func(*args, key=evaluation_key, **kwargs))
            target_time = jnp.asarray(args[time_position], dtype=float).reshape(())
            duration = jnp.maximum(target_time - t0, 0.0)

            def positive(_):
                source_times = t0 + duration * nodes

                def derivative_at(source_time):
                    call_args = list(args)
                    call_args[time_position] = source_time
                    return d_t_at(
                        tuple(call_args),
                        key=evaluation_key,
                        **kwargs,
                    )

                derivatives = jax.vmap(derivative_at)(source_times)
                if resolved_mode == "gj":
                    integral = jnp.tensordot(weights, derivatives, axes=(0, 0))
                    return jnp.power(duration, 1.0 - a) * integral / gamma
                lags = target_time - source_times
                kernel = jnp.power(lags, -a) / gamma
                return jnp.tensordot(
                    duration * weights * kernel,
                    derivatives,
                    axes=(0, 0),
                )

            def zero(_):
                return jnp.zeros_like(u.func(*args, key=evaluation_key, **kwargs))

            return jax.lax.cond(duration > 0.0, positive, zero, operand=None)

        return with_metadata(_op, method=resolved_mode)

    gamma = jsp.gamma(2.0 - a)
    if resolved_mode == "gj":
        from scipy.special import roots_jacobi

        raw_nodes, raw_weights = roots_jacobi(count, 1.0 - a, 0.0)
        nodes = (jnp.asarray(raw_nodes, dtype=float) + 1.0) / 2.0
        weights = jnp.asarray(raw_weights, dtype=float) * (2.0 ** (a - 2.0))
    else:
        nodes, weights = causal_reference_rule(
            GaussLegendreRule(count),
            cluster_exponent=cluster,
        )

    def _op(*args, key=None, **kwargs):
        evaluation_key = DOC_KEY0 if key is None else key
        if time_position is None:
            return jnp.zeros_like(u.func(*args, key=evaluation_key, **kwargs))
        target_time = jnp.asarray(args[time_position], dtype=float).reshape(())
        duration = jnp.maximum(target_time - t0, 0.0)

        def positive(_):
            source_times = t0 + duration * nodes

            def second_derivative_at(source_time):
                call_args = list(args)
                call_args[time_position] = source_time
                return d2_t_at(
                    tuple(call_args),
                    key=evaluation_key,
                    **kwargs,
                )

            derivatives = jax.vmap(second_derivative_at)(source_times)
            if resolved_mode == "gj":
                integral = jnp.tensordot(weights, derivatives, axes=(0, 0))
                return jnp.power(duration, 2.0 - a) * integral / gamma
            lags = target_time - source_times
            kernel = jnp.power(lags, 1.0 - a) / gamma
            return jnp.tensordot(
                duration * weights * kernel,
                derivatives,
                axes=(0, 0),
            )

        def zero(_):
            return jnp.zeros_like(u.func(*args, key=evaluation_key, **kwargs))

        return jax.lax.cond(duration > 0.0, positive, zero, operand=None)

    return with_metadata(_op, method=resolved_mode)


def caputo_time_fractional_dw(
    u: DomainFunction,
    /,
    *,
    alpha: float,
    time_var: str = "t",
    M: int = 128,
    mode: Literal["auto", "gj", "gl"] = "gj",
) -> DomainFunction:
    r"""Convenience wrapper for `caputo_time_fractional` using `order=M`.

    Named for a common discretization setting in physics-informed fractional models.
    """
    return caputo_time_fractional(
        u,
        alpha=float(alpha),
        time_var=time_var,
        mode=mode,
        order=int(M),
    )
