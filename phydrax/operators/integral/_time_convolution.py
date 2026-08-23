#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.domain import AbstractScalarDomain, DomainFunction

from ..._doc import DOC_KEY0
from ...integration import GaussLegendreRule
from ...integration._rules import IntervalRule
from .._causal_quadrature import causal_reference_rule


def _unwrap_factor(factor: object, /) -> object:
    return factor


def _time_start(u: DomainFunction, time_var: str) -> Array:
    factor = _unwrap_factor(u.domain.factor(time_var))
    if isinstance(factor, AbstractScalarDomain):
        return jnp.asarray(factor.fixed("start"), dtype=float)
    return jnp.array(0.0, dtype=float)


def time_convolution(
    k: Callable[[Array], ArrayLike],
    u: DomainFunction,
    /,
    *,
    time_var: str = "t",
    rule: IntervalRule | None = None,
    cluster_exponent: float = 1.0,
) -> DomainFunction:
    r"""Deterministic time convolution on a labeled time coordinate.

    Constructs

    $$
    (k * u)(t) = \int_{t_0}^{t} k(t-s)\,u(s)\,ds,
    $$

    where $t_0$ is the start of the scalar time factor. The declared fixed
    interval rule is mapped independently onto each causal interval. Randomized
    integral estimators belong in an estimator-aware randomized term rather than
    this field-valued operator.
    """
    if not callable(k):
        raise TypeError("time_convolution kernel must be callable.")
    if time_var not in u.domain.labels:
        raise ValueError(
            f"time_convolution requires time_var {time_var!r} in the function domain."
        )

    resolved_rule = GaussLegendreRule(48) if rule is None else rule
    reference_nodes, reference_weights = causal_reference_rule(
        resolved_rule,
        cluster_exponent=cluster_exponent,
    )
    t0 = _time_start(u, time_var)

    required = list(u.deps)
    if time_var not in required:
        required.append(time_var)
    deps = tuple(label for label in u.domain.labels if label in required)
    positions = {label: index for index, label in enumerate(deps)}
    u_positions = tuple(positions[label] for label in u.deps)
    time_position = positions.get(time_var)
    if time_position is None:
        raise ValueError(
            "time_convolution requires time_var to be present in dependencies."
        )
    u_time_position = u.deps.index(time_var) if time_var in u.deps else None

    def _u_at_time(u_args: list[object], time: Array, *, key, **kwargs):
        call_args = list(u_args)
        if u_time_position is not None:
            call_args[u_time_position] = time
        return u.func(*call_args, key=key, **kwargs)

    def _op(*args, key=None, **kwargs):
        evaluation_key = DOC_KEY0 if key is None else key
        target_time = jnp.asarray(args[time_position], dtype=float).reshape(())
        duration = jnp.maximum(target_time - t0, 0.0)
        u_args = [args[index] for index in u_positions]

        def integrate(_):
            source_times = t0 + duration * reference_nodes
            lags = target_time - source_times
            values = jax.vmap(
                lambda source_time: _u_at_time(
                    u_args,
                    source_time,
                    key=evaluation_key,
                    **kwargs,
                )
            )(source_times)
            kernel_values = jnp.asarray(jax.vmap(k)(lags))
            if kernel_values.ndim != 1:
                raise ValueError(
                    "time_convolution kernel must return one scalar per lag."
                )
            effective_weights = duration * reference_weights * kernel_values
            return jnp.tensordot(effective_weights, values, axes=(0, 0))

        def zero(_):
            return jnp.zeros_like(_u_at_time(u_args, t0, key=evaluation_key, **kwargs))

        return jax.lax.cond(duration > 0.0, integrate, zero, operand=None)

    metadata = dict(u.metadata)
    metadata.update(
        {
            "integral_operator": "time-convolution",
            "integral_time_var": time_var,
            "integral_rule": type(resolved_rule).__name__,
            "integral_rule_order": int(resolved_rule.order),
            "integral_cluster_exponent": float(cluster_exponent),
            "integral_randomized": False,
        }
    )
    return DomainFunction(domain=u.domain, deps=deps, func=_op, metadata=metadata)


__all__ = [
    "time_convolution",
]
