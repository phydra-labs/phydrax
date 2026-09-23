#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._population import ParticlePopulationState
from ...solver._finite_particle_transport import (
    FiniteParticleTransportPlan,
    FiniteParticleTransportState,
)
from ._dld_geometry import DLDGeometryPlan
from ._dld_metrics import DLDMetricPlan, DLDOutletPlan, DLDSeparationMetrics
from ._dld_screening import DLDEmpiricalScreenPlan, DLDEmpiricalScreenResult


class DLDWorkflowResult(StrictModule):
    final_state: FiniteParticleTransportState
    metrics: DLDSeparationMetrics
    screening: DLDEmpiricalScreenResult | None
    terminal_times: Array
    header: AdmissibilityHeader
    successful: Array
    plan_id: str = eqx.field(static=True)


class DLDWorkflowPlan(StrictModule, NonTrainableState):
    """Fixed-topology flow-bound finite-particle DLD separation workflow."""

    geometry: DLDGeometryPlan
    transport: FiniteParticleTransportPlan
    outlets: DLDOutletPlan
    metrics: DLDMetricPlan
    particle_classes: Array
    screening: DLDEmpiricalScreenPlan | None
    step_count: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    flow_model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: DLDGeometryPlan,
        transport: FiniteParticleTransportPlan,
        outlets: DLDOutletPlan,
        metrics: DLDMetricPlan,
        particle_classes: ArrayLike,
        /,
        *,
        step_count: int,
        step_size: float,
        flow_model_id: str,
        screening: DLDEmpiricalScreenPlan | None = None,
    ) -> None:
        classes = np.asarray(particle_classes, dtype=np.int32)
        steps = int(step_count)
        step = float(step_size)
        flow = str(flow_model_id)
        if (
            not isinstance(geometry, DLDGeometryPlan)
            or not isinstance(transport, FiniteParticleTransportPlan)
            or not isinstance(outlets, DLDOutletPlan)
            or not isinstance(metrics, DLDMetricPlan)
            or (
                screening is not None
                and not isinstance(screening, DLDEmpiricalScreenPlan)
            )
            or transport.wall.plan_id != geometry.plan_id
            or transport.properties.capacity != geometry.topology.particle_capacity
            or outlets.outlet_count != geometry.topology.outlet_count
            or outlets.outlet_x != geometry.design.outlet_x
            or metrics.outlet_count != outlets.outlet_count
            or classes.shape != (geometry.topology.particle_capacity,)
            or np.any((classes < 0) | (classes >= metrics.class_count))
            or steps <= 0
            or not np.isfinite(step)
            or step <= 0.0
            or not flow
            or transport.velocity_field.provider_id != flow
        ):
            raise ValueError(
                "DLD workflow geometry, transport, metrics, or flow mismatch."
            )
        self.geometry = geometry
        self.transport = transport
        self.outlets = outlets
        self.metrics = metrics
        self.particle_classes = jnp.asarray(classes)
        self.screening = screening
        self.step_count = steps
        self.step_size = step
        self.flow_model_id = flow
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-workflow",
                "geometry": geometry.plan_id,
                "transport": transport.plan_id,
                "outlets": outlets.plan_id,
                "metrics": metrics.plan_id,
                "particle_classes": array_tree_fingerprint(classes),
                "screening": None if screening is None else screening.plan_id,
                "step_count": steps,
                "step_size": step,
                "flow_model": flow,
            }
        )

    def run(
        self,
        initial_state: FiniteParticleTransportState,
        flow_evidence: AdmissibilityHeader,
        /,
        *,
        volume_flow: ArrayLike,
        pressure_drop: ArrayLike,
    ) -> DLDWorkflowResult:
        if initial_state.runtime_id != self.transport.runtime_id:
            raise ValueError(
                "DLD initial particle state belongs to another transport plan."
            )
        if (
            not isinstance(flow_evidence, AdmissibilityHeader)
            or flow_evidence.model_id != self.flow_model_id
        ):
            raise ValueError("DLD flow evidence does not match the bound velocity field.")
        initial_active = initial_state.population.active
        terminal_times = jnp.full(
            initial_active.shape, jnp.inf, dtype=initial_state.position.dtype
        )
        state = initial_state
        cumulative = flow_evidence.globally_eligible
        for _ in range(self.step_count):
            step_result = self.transport.advance(state, self.step_size)
            state = step_result.accepted
            cumulative = cumulative & step_result.successful
            classification = self.outlets.classify(
                state.position, state.population.active
            )
            terminal = classification.terminal_mask
            deactivation = self.transport.population.deactivate(
                state.population, terminal
            )
            population = ParticlePopulationState(
                deactivation.accepted_state.active,
                deactivation.accepted_state.mass,
                deactivation.accepted_state.incarnation,
                deactivation.accepted_state.ever_occupied,
                deactivation.accepted_state.retired,
            )
            terminal_times = jnp.where(terminal, state.time, terminal_times)
            state = FiniteParticleTransportState(
                population,
                jnp.where(population.active[:, None], state.position, 0.0),
                jnp.where(population.active[:, None], state.velocity, 0.0),
                jnp.where(terminal, classification.outlet_code, state.terminal_code),
                state.event_count,
                state.time,
                state.accepted_steps,
                state.key,
                state.runtime_id,
            )
            cumulative = cumulative & deactivation.successful
        metric = self.metrics.evaluate(
            self.particle_classes,
            initial_active,
            state.terminal_code,
            terminal_times,
            volume_flow,
            pressure_drop,
        )
        screen = (
            None
            if self.screening is None
            else self.screening.evaluate(self.geometry.topology, self.geometry.design)
        )
        screen_ok = (
            jnp.asarray(True) if screen is None else screen.header.globally_eligible
        )
        successful = cumulative & metric.header.globally_eligible & screen_ok
        reasons = jnp.bitwise_or.reduce(
            jnp.ravel(flow_evidence.reason_bits)
        ) | jnp.bitwise_or.reduce(jnp.ravel(metric.header.reason_bits))
        if screen is not None:
            reasons = reasons | jnp.bitwise_or.reduce(
                jnp.ravel(screen.header.reason_bits)
            )
        reasons = jnp.where(
            successful,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(successful, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dld-workflow-evidence", "plan": self.plan_id}
            ),
        )
        return DLDWorkflowResult(
            state,
            metric,
            screen,
            terminal_times,
            header,
            successful,
            self.plan_id,
        )


__all__ = ["DLDWorkflowPlan", "DLDWorkflowResult"]
