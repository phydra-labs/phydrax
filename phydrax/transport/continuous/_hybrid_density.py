#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ._density import ContinuousFlowLaw
from ._transport import ContinuousTransport


class ConditionalContinuousFlowLaw(AbstractProbabilityLaw):
    """State-conditional density for one fixed typed context/input layout."""

    flow_law: ContinuousFlowLaw
    input_signal: Any
    input_policy: Any
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: ContinuousTransport,
        input_signal: Any,
        /,
        *,
        input_policy: Any,
        context_id: str,
        max_exact_dimension: int = 32,
    ):
        if not context_id:
            raise ValueError("context_id must be non-empty.")
        # The supplied transport must already be prepared at this fixed context.  Its
        # divergence is state-only; no Jacobian with respect to input is introduced.
        self.flow_law = ContinuousFlowLaw(
            transport,
            max_exact_dimension=max_exact_dimension,
            flow_id=f"conditional:{context_id}:{transport.transport_id}",
        )
        self.input_signal = input_signal
        self.input_policy = input_policy
        self.context_id = context_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.flow_law.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.flow_law.batch_shape

    @property
    def density_measure_kind(self) -> str:
        return "lebesgue"

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        return self.flow_law.sample(key, sample_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.flow_law.log_prob(value)

    def contains(self, value: ArrayLike, /) -> Array:
        return self.flow_law.contains(value)


class PiecewiseFlowDensityResult(StrictModule):
    pre_event_state: Array
    continuous_log_prob: Array
    event_log_abs_determinant: Array
    event_count: Array
    log_prob: Array
    valid: Array
    status: Array
    schedule_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class PiecewiseContinuousFlowLaw(AbstractProbabilityLaw):
    """Exact finite event itinerary with dense square event log determinants."""

    flow_law: ContinuousFlowLaw
    prepared_schedule: Any
    forward_event_map: Any
    inverse_event_map: Any
    tape_provider: Any
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: ContinuousTransport,
        prepared_schedule: Any,
        /,
        *,
        forward_event_map: Any,
        inverse_event_map: Any,
        tape_provider: Any,
        max_exact_dimension: int = 32,
        law_id: str | None = None,
    ):
        if (
            not callable(forward_event_map)
            or not callable(inverse_event_map)
            or not callable(tape_provider)
        ):
            raise TypeError("event maps and tape_provider must be callable.")
        self.flow_law = ContinuousFlowLaw(
            transport,
            max_exact_dimension=max_exact_dimension,
            flow_id=f"piecewise-continuous:{transport.transport_id}",
        )
        self.prepared_schedule = prepared_schedule
        self.forward_event_map = forward_event_map
        self.inverse_event_map = inverse_event_map
        self.tape_provider = tape_provider
        self.law_id = canonical_fingerprint(
            {
                "kind": "piecewise-continuous-flow-law-v2",
                "transport": transport.transport_id,
                "schedule": prepared_schedule.schedule_id,
                "schedule_preparation": prepared_schedule.preparation_id,
                "replay_policy": prepared_schedule.replay_policy.policy_id,
                "maximum_dimension": max_exact_dimension,
                "requested_law_id": law_id,
            }
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.flow_law.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.flow_law.batch_shape

    @property
    def density_measure_kind(self) -> str:
        return "lebesgue"

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        continuous = self.flow_law.sample(key, sample_shape)
        return self.forward_event_map(continuous, self.prepared_schedule)

    def log_prob_with_diagnostics(
        self, value: ArrayLike, /
    ) -> PiecewiseFlowDensityResult:
        data = jnp.asarray(value)
        pre_event = self.inverse_event_map(data, self.prepared_schedule)
        continuous = self.flow_law.log_prob(pre_event)
        tape = self.tape_provider(data, pre_event, self.prepared_schedule)
        if (
            tape.schedule_id != self.prepared_schedule.schedule_id
            or tape.policy_id != self.prepared_schedule.replay_policy.policy_id
        ):
            raise ValueError(
                "Hybrid event tape itinerary or replay-policy identity differs "
                "from the prepared schedule."
            )
        event_valid = (
            jnp.all((~tape.active) | tape.log_jacobian_valid)
            & ~tape.capacity_exceeded
            & jnp.all(tape.saltation_valid | ~tape.active)
            & jnp.all(jnp.isfinite(tape.log_abs_determinants) | ~tape.active)
        )
        logdet = tape.total_log_abs_determinant
        density = continuous - logdet
        valid = (
            event_valid
            & jnp.all(jnp.isfinite(continuous))
            & jnp.all(jnp.isfinite(density))
        )
        return PiecewiseFlowDensityResult(
            pre_event_state=pre_event,
            continuous_log_prob=continuous,
            event_log_abs_determinant=logdet,
            event_count=tape.event_count,
            log_prob=jnp.where(valid, density, -jnp.inf),
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            schedule_id=self.prepared_schedule.schedule_id,
            reference_measure="lebesgue",
            law_id=self.law_id,
        )

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).log_prob

    def contains(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).valid


__all__ = [
    "ConditionalContinuousFlowLaw",
    "PiecewiseContinuousFlowLaw",
    "PiecewiseFlowDensityResult",
]
