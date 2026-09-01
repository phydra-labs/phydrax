#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._elements import ModifiedEquinoctialElements
from ._status import AstrodynamicsStatus


class DsstResult(StrictModule):
    times: Array
    mean_elements: Array
    osculating_elements: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class DsstPlan(StrictModule, NonTrainableState):
    """Fixed-grid semi-analytical propagation of mean equinoctial elements."""

    averaged_rates: tuple[Callable, ...]
    short_period_terms: tuple[Callable, ...]
    times: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        averaged_rates: tuple[Callable, ...],
        short_period_terms: tuple[Callable, ...],
        times: ArrayLike,
        /,
        *,
        model_ids: tuple[str, ...],
    ):
        rates = tuple(averaged_rates)
        short = tuple(short_period_terms)
        if not rates or any(not callable(value) for value in (*rates, *short)):
            raise ValueError("DSST force and short-period terms must be callable.")
        if len(model_ids) != len(rates) + len(short):
            raise ValueError("DSST model IDs must cover every contribution.")
        times_host = np.asarray(times, dtype=float)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(np.diff(times_host) <= 0.0)
            or np.any(~np.isfinite(times_host))
        ):
            raise ValueError("DSST times must be finite and strictly increasing.")
        self.averaged_rates = rates
        self.short_period_terms = short
        self.times = jnp.asarray(times_host)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsst-plan",
                "models": list(model_ids),
                "num_times": int(times_host.size),
            }
        )

    def _rate(self, time: Array, elements: Array, args: Any, /) -> Array:
        contributions = tuple(
            jnp.asarray(term(time, elements, args)) for term in self.averaged_rates
        )
        return jnp.sum(jnp.stack(contributions), axis=0)

    def _short_period(self, time: Array, elements: Array, args: Any, /) -> Array:
        if not self.short_period_terms:
            return jnp.zeros_like(elements)
        contributions = tuple(
            jnp.asarray(term(time, elements, args)) for term in self.short_period_terms
        )
        return jnp.sum(jnp.stack(contributions), axis=0)

    def propagate(
        self, initial: ModifiedEquinoctialElements, args: Any = None, /
    ) -> DsstResult:
        if not isinstance(initial, ModifiedEquinoctialElements):
            raise TypeError("initial must be ModifiedEquinoctialElements.")

        def step(elements, interval):
            start, end = interval
            dt = end - start
            k1 = self._rate(start, elements, args)
            k2 = self._rate(start + 0.5 * dt, elements + 0.5 * dt * k1, args)
            k3 = self._rate(start + 0.5 * dt, elements + 0.5 * dt * k2, args)
            k4 = self._rate(end, elements + dt * k3, args)
            next_elements = elements + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            valid = jnp.all(jnp.isfinite(next_elements)) & (next_elements[0] > 0.0)
            accepted = jnp.where(valid, next_elements, elements)
            return accepted, (accepted, valid)

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        _, outputs = jax.lax.scan(step, initial.values, intervals)
        mean = jnp.concatenate((initial.values[None], outputs[0]), axis=0)
        valid = jnp.concatenate((jnp.asarray(True)[None], outputs[1]))
        osculating = jax.vmap(
            lambda time, values: values + self._short_period(time, values, args)
        )(self.times, mean)
        valid = valid & jnp.all(jnp.isfinite(osculating), axis=-1)
        status = jnp.where(
            valid, int(AstrodynamicsStatus.SUCCESS), int(AstrodynamicsStatus.NONCONVERGED)
        ).astype(jnp.int32)
        return DsstResult(self.times, mean, osculating, valid, status, self.plan_id)


__all__ = ["DsstPlan", "DsstResult"]
