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
from ._status import AstrodynamicsStatus


class LightTimeResult(StrictModule):
    transmit_time: Array
    receive_time: Array
    geometric_range: Array
    shapiro_delay: Array
    iterations: Array
    residual: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class LightTimePlan(StrictModule, NonTrainableState):
    transmitter_state: Callable
    receiver_state: Callable
    gravitating_body_state: Callable
    body_mu: Array
    speed_of_light: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transmitter_provider_id: str = eqx.field(static=True)
    receiver_provider_id: str = eqx.field(static=True)
    gravitating_body_provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transmitter_state,
        receiver_state,
        gravitating_body_state,
        body_mu,
        /,
        *,
        speed_of_light=299792458.0,
        max_iterations=16,
        tolerance=1.0e-12,
        plan_id="one-way-light-time",
        transmitter_provider_id,
        receiver_provider_id,
        gravitating_body_provider_id,
    ):
        if not all(
            callable(value)
            for value in (transmitter_state, receiver_state, gravitating_body_state)
        ):
            raise TypeError("Light-time state providers must be callable.")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer.")
        tolerance_ = float(tolerance)
        body_mu_ = np.asarray(body_mu, dtype=np.float64)
        light_ = np.asarray(speed_of_light, dtype=np.float64)
        provider_ids = tuple(
            str(value).strip()
            for value in (
                transmitter_provider_id,
                receiver_provider_id,
                gravitating_body_provider_id,
            )
        )
        declared_id = str(plan_id).strip()
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if (
            not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or body_mu_.shape != ()
            or not np.isfinite(body_mu_)
            or body_mu_ <= 0.0
            or light_.shape != ()
            or not np.isfinite(light_)
            or light_ <= 0.0
        ):
            raise ValueError(
                "Light-time constants and tolerance must be finite and positive."
            )
        if any(not value for value in (*provider_ids, declared_id)):
            raise ValueError("Light-time provider and plan identities must be non-empty.")
        self.transmitter_state = transmitter_state
        self.receiver_state = receiver_state
        self.gravitating_body_state = gravitating_body_state
        self.body_mu = jnp.asarray(body_mu_)
        self.speed_of_light = jnp.asarray(light_)
        self.max_iterations = max_iterations
        self.tolerance = tolerance_
        (
            self.transmitter_provider_id,
            self.receiver_provider_id,
            self.gravitating_body_provider_id,
        ) = provider_ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "light-time-plan",
                "declared_id": declared_id,
                "providers": provider_ids,
                "body_mu": body_mu_,
                "speed_of_light": light_,
                "iterations": max_iterations,
                "tolerance": tolerance_,
            }
        )

    def solve(self, receive_time: ArrayLike, args: Any = None, /) -> LightTimeResult:
        receive = jnp.asarray(receive_time).reshape(())
        receiver = jnp.asarray(self.receiver_state(receive, args))
        if receiver.shape != (3,):
            raise ValueError("receiver_state must return one position three-vector.")

        def residual(transmit):
            transmitter = jnp.asarray(self.transmitter_state(transmit, args))
            body = jnp.asarray(self.gravitating_body_state(receive, args))
            if transmitter.shape != (3,) or body.shape != (3,):
                raise ValueError(
                    "Light-time state providers must return position three-vectors."
                )
            range_vector = receiver - transmitter
            distance = jnp.sqrt(jnp.sum(range_vector * range_vector))
            r_tx = jnp.sqrt(jnp.sum((transmitter - body) ** 2))
            r_rx = jnp.sqrt(jnp.sum((receiver - body) ** 2))
            shapiro = (
                2.0
                * self.body_mu
                / self.speed_of_light**3
                * jnp.log(
                    (r_tx + r_rx + distance)
                    / jnp.maximum(r_tx + r_rx - distance, 1.0e-30)
                )
            )
            value = receive - transmit - distance / self.speed_of_light - shapiro
            finite = (
                jnp.isfinite(receive)
                & jnp.all(jnp.isfinite(receiver))
                & jnp.all(jnp.isfinite(transmitter))
                & jnp.all(jnp.isfinite(body))
                & jnp.all(jnp.isfinite(jnp.asarray((value, distance, shapiro))))
            )
            return value, distance, shapiro, finite

        initial = receive

        def step(index, carry):
            transmit, converged, first = carry
            value, _, _, _ = residual(transmit)
            derivative = jax.grad(lambda time: residual(time)[0])(transmit)
            candidate = transmit - value / jnp.where(
                jnp.abs(derivative) > 0.0, derivative, 1.0
            )
            now = jnp.abs(value) <= self.tolerance
            update = ~converged & ~now & jnp.isfinite(candidate)
            return (
                jnp.where(update, candidate, transmit),
                converged | now,
                jnp.where((first < 0) & now, index + 1, first),
            )

        transmit, converged, iterations = jax.lax.fori_loop(
            0,
            self.max_iterations,
            step,
            (initial, jnp.asarray(False), jnp.asarray(-1, dtype=jnp.int32)),
        )
        value, distance, shapiro, finite = residual(transmit)
        converged = converged | (jnp.abs(value) <= self.tolerance)
        valid = finite & converged
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.NONCONVERGED),
            ),
        ).astype(jnp.int32)
        return LightTimeResult(
            transmit,
            receive,
            distance,
            shapiro,
            jnp.where(iterations >= 0, iterations, self.max_iterations),
            jnp.abs(value),
            valid,
            status,
            self.plan_id,
        )


__all__ = ["LightTimePlan", "LightTimeResult"]
