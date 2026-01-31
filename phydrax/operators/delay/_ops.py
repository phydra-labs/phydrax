#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ...domain._function import DomainFunction


def delay(
    u: DomainFunction,
    /,
    tau: DomainFunction | ArrayLike,
    *,
    time_var: str = "t",
    clip_time_min: float | None = None,
) -> DomainFunction:
    r"""Delay operator along a labeled time coordinate.

    Defines a new function $v$ by shifting the time-like coordinate:

    $$
    v(t) = (\mathcal{D}_\tau u)(t) = u(t-\tau),
    $$

    where $\tau$ can be a constant or a `DomainFunction` (allowing spatially varying
    delays).

    **Arguments:**

    - `u`: Input function $u$.
    - `tau`: Delay $\tau$ (constant or `DomainFunction`).
    - `time_var`: Label of the time coordinate (default `"t"`).
    - `clip_time_min`: Optional lower bound for the delayed time; if set, uses
      `t_delayed = max(t - tau, clip_time_min)` to avoid evaluating before a minimum
      time (e.g. before the interval start).
    """
    if time_var not in u.domain.labels:
        raise ValueError(
            f"delay requires time_var {time_var!r} to be in the function domain."
        )

    tau_fn = (
        tau
        if isinstance(tau, DomainFunction)
        else DomainFunction(domain=u.domain, deps=(), func=tau)
    )
    joined = u.domain.join(tau_fn.domain)
    u2 = u.promote(joined)
    tau2 = tau_fn.promote(joined)

    needed = tuple(
        lbl
        for lbl in joined.labels
        if (lbl in u2.deps) or (lbl in tau2.deps) or (lbl == time_var)
    )

    idx = {lbl: i for i, lbl in enumerate(needed)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    tau_pos = tuple(idx[lbl] for lbl in tau2.deps)
    t_pos = idx.get(time_var)
    if t_pos is None:
        raise ValueError("delay requires time_var to be present in dependencies.")

    u_time_idx = u2.deps.index(time_var) if time_var in u2.deps else None

    def _op(*args, key=None, **kwargs):
        t = jnp.asarray(args[t_pos])

        tau_args = [args[i] for i in tau_pos]
        tau_val = jnp.asarray(tau2.func(*tau_args, key=key, **kwargs)).reshape(())
        t_delayed = t - tau_val
        if clip_time_min is not None:
            t_delayed = jnp.maximum(t_delayed, float(clip_time_min))

        u_args = [args[i] for i in u_pos]
        if u_time_idx is not None:
            u_args[u_time_idx] = t_delayed
        return u2.func(*u_args, key=key, **kwargs)

    return DomainFunction(domain=joined, deps=needed, func=_op, metadata=u.metadata)


__all__ = [
    "delay",
]
