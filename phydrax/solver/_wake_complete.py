#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.vortex._ring_sheet import ring_sheet_evidence, VortexRingSheetState
from ..discretization.vortex._source import VortexTargetState


WakeVelocity = Callable[[VortexTargetState, Array, Any], Array]


class WakeAdaptationCandidate(StrictModule):
    refine_edge_mask: Array
    coarsen_edge_mask: Array
    reconnect_pair_mask: Array
    requested_new_vertices: Array
    requested_removed_vertices: Array
    quality_successful: Array
    adaptation_id: str = eqx.field(static=True)


class WakeAdaptationPlan(StrictModule, NonTrainableState):
    maximum_edge_length: float = eqx.field(static=True)
    minimum_edge_length: float = eqx.field(static=True)
    coarsen_age: float = eqx.field(static=True)
    reconnection_distance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_edge_length: float,
        minimum_edge_length: float,
        /,
        *,
        coarsen_age: float,
        reconnection_distance: float,
    ):
        maximum, minimum, age, distance = (
            float(maximum_edge_length),
            float(minimum_edge_length),
            float(coarsen_age),
            float(reconnection_distance),
        )
        if minimum <= 0.0 or maximum <= minimum or age < 0.0 or distance <= 0.0:
            raise ValueError("Wake adaptation controls are invalid.")
        self.maximum_edge_length, self.minimum_edge_length = maximum, minimum
        self.coarsen_age, self.reconnection_distance = age, distance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wake-adaptation-plan",
                "maximum_edge_length": maximum,
                "minimum_edge_length": minimum,
                "coarsen_age": age,
                "reconnection_distance": distance,
            }
        )

    def evaluate(self, state: VortexRingSheetState, /) -> WakeAdaptationCandidate:
        start, end, _ = state.edge_geometry()
        length = jnp.linalg.norm(end - start, axis=-1)
        active = state.topology.edge_active
        refine = active & (length > self.maximum_edge_length)
        coarsen = (
            active
            & (length < self.minimum_edge_length)
            & (state.edge_age >= self.coarsen_age)
        )
        midpoint = 0.5 * (start + end)
        separation = jnp.linalg.norm(midpoint[:, None, :] - midpoint[None, :, :], axis=-1)
        share_vertex = (
            (state.topology.edge_start[:, None] == state.topology.edge_start[None, :])
            | (state.topology.edge_start[:, None] == state.topology.edge_end[None, :])
            | (state.topology.edge_end[:, None] == state.topology.edge_start[None, :])
            | (state.topology.edge_end[:, None] == state.topology.edge_end[None, :])
        )
        reconnect = (
            active[:, None]
            & active[None, :]
            & ~share_vertex
            & ~jnp.eye(state.topology.edge_capacity, dtype=bool)
            & (separation < self.reconnection_distance)
        )
        quality = jnp.all(jnp.where(active, jnp.isfinite(length) & (length > 0.0), True))
        return WakeAdaptationCandidate(
            refine,
            coarsen,
            reconnect,
            jnp.sum(refine, dtype=jnp.int32),
            jnp.sum(coarsen, dtype=jnp.int32),
            quality,
            self.plan_id,
        )


class WakeStepEvidence(StrictModule):
    circulation_before: Array
    circulation_after: Array
    circulation_residual: Array
    minimum_edge_length: Array
    maximum_edge_length: Array
    maximum_core_radius: Array
    finite: Array
    adaptation: WakeAdaptationCandidate | None


class WakeStepResult(StrictModule):
    candidate: VortexRingSheetState
    accepted: VortexRingSheetState
    evidence: WakeStepEvidence
    successful: Array
    method_id: str = eqx.field(static=True)


class VortexWakeIntegratorPlan(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    core_diffusivity: float = eqx.field(static=True)
    adaptation: WakeAdaptationPlan | None
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: str = "midpoint",
        /,
        *,
        core_diffusivity: float = 0.0,
        adaptation: WakeAdaptationPlan | None = None,
    ):
        method_, diffusivity = str(method), float(core_diffusivity)
        if method_ not in ("euler", "midpoint", "rk3") or diffusivity < 0.0:
            raise ValueError("Wake integrator method/diffusivity is invalid.")
        if adaptation is not None and not isinstance(adaptation, WakeAdaptationPlan):
            raise TypeError("adaptation must be WakeAdaptationPlan or None.")
        self.method, self.core_diffusivity, self.adaptation = (
            method_,
            diffusivity,
            adaptation,
        )
        self.method_id = canonical_fingerprint(
            {
                "kind": "vortex-wake-integrator",
                "method": method_,
                "core_diffusivity": diffusivity,
                "adaptation": None if adaptation is None else adaptation.plan_id,
            }
        )

    def step(
        self,
        state: VortexRingSheetState,
        velocity: WakeVelocity,
        time: ArrayLike,
        time_step: ArrayLike,
        args: Any = None,
        /,
    ) -> WakeStepResult:
        if not isinstance(state, VortexRingSheetState) or not callable(velocity):
            raise TypeError("Wake step requires state and velocity callback.")
        time_, dt = (
            jnp.asarray(time, dtype=state.vertices.dtype),
            jnp.asarray(time_step, dtype=state.vertices.dtype),
        )
        if time_.shape != () or dt.shape != ():
            raise ValueError("Wake time and step must be scalar.")

        def field(vertices, evaluation_time):
            value = jnp.asarray(
                velocity(VortexTargetState(vertices), evaluation_time, args),
                dtype=vertices.dtype,
            )
            if value.shape != vertices.shape:
                raise ValueError("Wake velocity callback must match vertex shape.")
            return value

        initial = state.vertices
        first = field(initial, time_)
        if self.method == "euler":
            candidate_vertices = initial + dt * first
        elif self.method == "midpoint":
            midpoint = initial + 0.5 * dt * first
            candidate_vertices = initial + dt * field(midpoint, time_ + 0.5 * dt)
        else:
            stage1 = initial + dt * first
            stage2 = 0.75 * initial + 0.25 * (stage1 + dt * field(stage1, time_ + dt))
            candidate_vertices = (1.0 / 3.0) * initial + (2.0 / 3.0) * (
                stage2 + dt * field(stage2, time_ + 0.5 * dt)
            )
        core = jnp.sqrt(state.edge_core_radius**2 + 4.0 * self.core_diffusivity * dt)
        candidate = VortexRingSheetState(
            state.topology,
            candidate_vertices,
            state.ring_circulation,
            core,
            state.edge_age + jnp.where(state.topology.edge_active, dt, 0.0),
        )
        before, after = ring_sheet_evidence(state), ring_sheet_evidence(candidate)
        adaptation = (
            None if self.adaptation is None else self.adaptation.evaluate(candidate)
        )
        finite = after.finite & jnp.isfinite(dt) & (dt > 0.0)
        successful = finite & (
            jnp.max(jnp.abs(candidate.ring_circulation - state.ring_circulation)) == 0.0
        )
        accepted = VortexRingSheetState(
            state.topology,
            jnp.where(successful, candidate.vertices, state.vertices),
            jnp.where(successful, candidate.ring_circulation, state.ring_circulation),
            jnp.where(successful, candidate.edge_core_radius, state.edge_core_radius),
            jnp.where(successful, candidate.edge_age, state.edge_age),
        )
        start, end, _ = accepted.edge_geometry()
        length = jnp.linalg.norm(end - start, axis=-1)
        circulation_before = jnp.sum(state.ring_circulation)
        circulation_after = jnp.sum(accepted.ring_circulation)
        evidence = WakeStepEvidence(
            circulation_before,
            circulation_after,
            circulation_after - circulation_before,
            jnp.min(jnp.where(state.topology.edge_active, length, jnp.inf)),
            jnp.max(jnp.where(state.topology.edge_active, length, 0.0)),
            jnp.max(
                jnp.where(state.topology.edge_active, accepted.edge_core_radius, 0.0)
            ),
            finite,
            adaptation,
        )
        return WakeStepResult(candidate, accepted, evidence, successful, self.method_id)


__all__ = [
    "VortexWakeIntegratorPlan",
    "WakeAdaptationCandidate",
    "WakeAdaptationPlan",
    "WakeStepEvidence",
    "WakeStepResult",
]
