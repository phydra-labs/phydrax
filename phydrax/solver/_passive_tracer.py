#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic fixed-step composition for a base method and one passive tracer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_passive_tracer import (
    PreparedMACPassiveTracerMacCormack,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class MACPassiveTracerContinuationState(StrictModule):
    """One base state and its collocated, fixed-shape passive tracer."""

    base_state: PyTree[Array]
    tracer: Array


class MACPassiveTracerFixedStepMethod(AbstractFixedStepMethod):
    """Atomically advance a base method and a nonconservative passive tracer.

    The carrier velocity is sampled once from the pre-step base state and remains
    frozen throughout midpoint tracing. Base and tracer candidates are committed
    together; failure of either branch rolls both back to the input state.
    """

    base_method: AbstractFixedStepMethod
    transport: PreparedMACPassiveTracerMacCormack
    velocity_from_state: Callable[[PyTree[Array]], FaceVelocity] = eqx.field(static=True)
    velocity_provider_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_method: AbstractFixedStepMethod,
        transport: PreparedMACPassiveTracerMacCormack,
        velocity_from_state: Callable[[PyTree[Array]], FaceVelocity],
        velocity_provider_id: str,
        /,
    ):
        if not isinstance(base_method, AbstractFixedStepMethod):
            raise TypeError("base_method must implement AbstractFixedStepMethod.")
        if not isinstance(transport, PreparedMACPassiveTracerMacCormack):
            raise TypeError("transport must be PreparedMACPassiveTracerMacCormack.")
        if not callable(velocity_from_state):
            raise TypeError("velocity_from_state must be callable.")
        provider_id = str(velocity_provider_id)
        if not provider_id:
            raise ValueError("velocity_provider_id must be non-empty.")
        self.base_method = base_method
        self.transport = transport
        self.velocity_from_state = velocity_from_state
        self.velocity_provider_id = provider_id
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-passive-tracer-fixed-step",
                "base_method": base_method.method_id,
                "transport": transport.prepared_id,
                "velocity_provider": provider_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: MACPassiveTracerContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, MACPassiveTracerContinuationState):
            raise TypeError("state must be MACPassiveTracerContinuationState.")
        frozen_velocity = self.velocity_from_state(state.base_state)
        base = self.base_method.step(
            step_index,
            time,
            state.base_state,
            step_size,
            args,
        )
        if not isinstance(base, FixedStepResult):
            raise TypeError("base_method.step must return FixedStepResult.")
        tracer = self.transport.advance(state.tracer, frozen_velocity, step_size)
        successful = base.successful & tracer.success

        candidate = MACPassiveTracerContinuationState(
            base_state=base.candidate_state,
            tracer=tracer.values,
        )
        accepted_base = jax.tree.map(
            lambda accepted, previous: jnp.where(successful, accepted, previous),
            base.accepted_state,
            state.base_state,
        )
        accepted = MACPassiveTracerContinuationState(
            base_state=accepted_base,
            tracer=jnp.where(successful, tracer.values, state.tracer),
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=jnp.maximum(base.residual, tracer.donor_bound_defect),
            iterations=base.iterations,
            work=base.work
            + jnp.asarray(self.transport.work_count, dtype=base.work.dtype),
            transform_applied=base.transform_applied,
            transform_correction_norm=base.transform_correction_norm,
        )


__all__ = [
    "MACPassiveTracerContinuationState",
    "MACPassiveTracerFixedStepMethod",
]
