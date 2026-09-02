#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._dynamics import PreparedFiniteVolumeDynamics
from ...equations._chemical_rates import ChemicalRateRuntime
from ...solver._finite_volume_content import FiniteVolumeConservativeContentState
from ...solver._finite_volume_runtime import (
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)
from ._mechanism import CompiledChemicalMechanism
from ._state import ReactiveConservedLayout, ReactiveEulerSystem


class ReactiveAdvanceState(StrictModule):
    """Restart-complete reacting state, including the accepted FV runtime tree."""

    time: Array
    conserved: Array
    transport_runtime_state: FiniteVolumeRuntimeState
    accepted_macro_steps: Array
    schedule_index: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        conserved: ArrayLike,
        transport_runtime_state: FiniteVolumeRuntimeState,
        /,
        *,
        accepted_macro_steps: ArrayLike = 0,
        schedule_index: ArrayLike = 0,
        state_id: str,
    ):
        time_ = jnp.asarray(time)
        conserved_ = jnp.asarray(conserved)
        accepted = jnp.asarray(accepted_macro_steps, dtype=jnp.int32)
        schedule = jnp.asarray(schedule_index, dtype=jnp.int32)
        if not isinstance(transport_runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("transport_runtime_state must be FiniteVolumeRuntimeState.")
        if time_.shape != () or accepted.shape != () or schedule.shape != ():
            raise ValueError("Reactive advance metadata must be scalar.")
        identifier = str(state_id)
        if not identifier:
            raise ValueError("state_id must be nonempty.")
        self.time = time_
        self.conserved = conserved_
        self.transport_runtime_state = transport_runtime_state
        self.accepted_macro_steps = accepted
        self.schedule_index = schedule
        self.state_id = identifier


class ReactiveAdvanceEvidence(StrictModule):
    attempted_time: Array
    attempted_step: Array
    transport_stage_successful: Array
    transport_runtime_status: Array
    minimum_transport_stability_margin: Array
    chemistry_stage_successful: Array
    final_cell_successful: Array
    maximum_element_defect: Array
    maximum_charge_defect: Array
    state_change_norm: Array
    rolled_back: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class ReactiveAdvanceResult(StrictModule):
    state: ReactiveAdvanceState
    evidence: ReactiveAdvanceEvidence


class ReactiveStrangPlan(StrictModule, NonTrainableState):
    """Fixed-schedule Strang split with atomic FV/runtime macro-step commit."""

    transport: PreparedFiniteVolumeRuntime
    mechanism: CompiledChemicalMechanism
    layout: ReactiveConservedLayout
    schedule_substeps: int = eqx.field(static=True)
    transport_substeps: int = eqx.field(static=True)
    chemistry_substeps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: PreparedFiniteVolumeRuntime,
        mechanism: CompiledChemicalMechanism,
        /,
        *,
        schedule_substeps: int = 1,
        transport_substeps: int = 1,
        chemistry_substeps: int = 1,
    ):
        if not isinstance(transport, PreparedFiniteVolumeRuntime):
            raise TypeError("transport must be PreparedFiniteVolumeRuntime.")
        if not isinstance(transport.dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("Reactive transport requires structured FV dynamics.")
        if not isinstance(transport.dynamics.system, ReactiveEulerSystem):
            raise TypeError("Reactive transport must bind a ReactiveEulerSystem.")
        if not isinstance(mechanism, CompiledChemicalMechanism):
            raise TypeError("mechanism must be CompiledChemicalMechanism.")
        layout = transport.dynamics.system.layout
        if layout.gas_model.model_id != mechanism.gas_model.model_id:
            raise ValueError("Transport and chemistry must bind the same gas model.")
        schedule = int(schedule_substeps)
        transport_count = int(transport_substeps)
        chemistry = int(chemistry_substeps)
        if min(schedule, transport_count, chemistry) < 1:
            raise ValueError("All fixed schedule counts must be positive.")
        self.transport = transport
        self.mechanism = mechanism
        self.layout = layout
        self.schedule_substeps = schedule
        self.transport_substeps = transport_count
        self.chemistry_substeps = chemistry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-fixed-schedule-strang",
                "transport_runtime": transport.runtime_id,
                "mechanism": mechanism.mechanism_id,
                "schedule_substeps": schedule,
                "transport_substeps": transport_count,
                "chemistry_substeps": chemistry,
            }
        )

    @property
    def dynamics(self) -> PreparedFiniteVolumeDynamics:
        return self.transport.dynamics

    def _runtime_average(self, runtime_state: FiniteVolumeRuntimeState, /) -> Array:
        return runtime_state.cell_average().reshape(
            self.transport.dynamics.discretization.state_shape
        )

    def _replace_runtime_average(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        conserved: Array,
        time: Array,
        /,
    ) -> FiniteVolumeRuntimeState:
        content = runtime_state.content_state
        flat = conserved.reshape(content.conservative_content.shape)
        replaced_content = FiniteVolumeConservativeContentState.from_cell_average(
            flat,
            content.effective_cell_volumes,
            content.active_cell_mask,
            time,
            topology_epoch_id=content.topology_epoch_id,
            geometry_family_id=content.geometry_family_id,
            geometry_layout_id=content.geometry_layout_id,
            geometry_version=content.geometry_version,
            evidence_policy_id=content.evidence_policy_id,
            evidence_version=content.evidence_version,
            precision=content.precision,
        )
        return FiniteVolumeRuntimeState(
            replaced_content,
            runtime_state.topology_journal,
            runtime_state.step_size,
            accepted_step=runtime_state.accepted_step,
            last_status=runtime_state.last_status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )

    def initial_state(
        self, conserved: ArrayLike, /, *, time: ArrayLike = 0.0
    ) -> ReactiveAdvanceState:
        value = jnp.asarray(conserved)
        if value.shape != self.transport.dynamics.discretization.state_shape:
            raise ValueError("Initial reactive state does not match FV geometry.")
        evidence = self.layout.evidence(value)
        if not bool(jnp.all(evidence.successful)):
            raise ValueError("Initial reactive state is inadmissible.")
        runtime_state = self.transport.initialize_state(
            value,
            time,
            jnp.asarray(1.0, dtype=value.dtype),
        )
        return ReactiveAdvanceState(
            runtime_state.time,
            value,
            runtime_state,
            state_id=canonical_fingerprint(
                {
                    "kind": "reactive-advance-state",
                    "plan": self.plan_id,
                    "shape": list(value.shape),
                }
            ),
        )

    def _transport_step(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        step: Array,
        args: Any,
        /,
    ) -> tuple[Array, FiniteVolumeRuntimeState, Array, Array, Array]:
        stage_step = step / self.transport_substeps
        state = runtime_state
        successful = jnp.asarray(True)
        minimum_margin = jnp.asarray(jnp.inf, dtype=step.dtype)
        status = runtime_state.last_status
        for _ in range(self.transport_substeps):
            scheduled = self.transport.advance_prescribed(state, stage_step, args)
            state = scheduled.runtime_state
            successful = successful & scheduled.accepted
            minimum_margin = jnp.minimum(minimum_margin, scheduled.stability_margin)
            status = state.last_status
        return self._runtime_average(state), state, successful, status, minimum_margin

    def _chemistry_source(
        self,
        conserved: Array,
        runtime: ChemicalRateRuntime | None,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        fields = self.layout.split(conserved)
        primitive = self.layout.primitive(conserved)
        chemical = self.mechanism.source_from_density_mass_fractions(
            fields.density,
            primitive.temperature,
            primitive.pressure,
            fields.mass_fractions,
            runtime=runtime,
        )
        source = self.layout.assemble(
            jnp.zeros_like(fields.density),
            chemical.species_mass_production_rate[..., :-1],
            jnp.zeros_like(fields.momentum),
            jnp.zeros_like(fields.density),
        )
        successful = primitive.evidence.successful & chemical.evidence.successful
        element = jnp.max(jnp.abs(chemical.evidence.element_residual), axis=-1)
        charge = jnp.abs(chemical.evidence.charge_residual)
        return source, successful, element, charge

    def _chemistry_step(
        self,
        conserved: Array,
        step: Array,
        runtime: ChemicalRateRuntime | None,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        stage_step = step / self.chemistry_substeps
        value = conserved
        successful = jnp.asarray(True)
        maximum_element = jnp.asarray(0.0, dtype=value.dtype)
        maximum_charge = jnp.asarray(0.0, dtype=value.dtype)
        for _ in range(self.chemistry_substeps):
            first, first_valid, first_element, first_charge = self._chemistry_source(
                value, runtime
            )
            midpoint = value + 0.5 * stage_step * first
            midpoint_valid = self.layout.evidence(midpoint).successful
            second, second_valid, second_element, second_charge = self._chemistry_source(
                midpoint, runtime
            )
            candidate = value + stage_step * second
            candidate_valid = self.layout.evidence(candidate).successful
            successful = (
                successful
                & jnp.all(first_valid)
                & jnp.all(midpoint_valid)
                & jnp.all(second_valid)
                & jnp.all(candidate_valid)
            )
            maximum_element = jnp.maximum(
                maximum_element,
                jnp.maximum(jnp.max(first_element), jnp.max(second_element)),
            )
            maximum_charge = jnp.maximum(
                maximum_charge,
                jnp.maximum(jnp.max(first_charge), jnp.max(second_charge)),
            )
            value = candidate
        return value, successful, maximum_element, maximum_charge

    def advance(
        self,
        state: ReactiveAdvanceState,
        step: ArrayLike,
        /,
        *,
        transport_args: Any = None,
        chemistry_runtime: ChemicalRateRuntime | None = None,
    ) -> ReactiveAdvanceResult:
        if not isinstance(state, ReactiveAdvanceState):
            raise TypeError("state must be ReactiveAdvanceState.")
        step_ = jnp.asarray(step, dtype=state.conserved.dtype)
        if step_.shape != ():
            raise ValueError("Reactive macro step must be scalar.")
        schedule_step = step_ / self.schedule_substeps
        candidate = state.conserved
        candidate_runtime = state.transport_runtime_state
        transport_success = jnp.asarray(True)
        chemistry_success = jnp.asarray(True)
        minimum_margin = jnp.asarray(jnp.inf, dtype=candidate.dtype)
        runtime_status = candidate_runtime.last_status
        maximum_element = jnp.asarray(0.0, dtype=candidate.dtype)
        maximum_charge = jnp.asarray(0.0, dtype=candidate.dtype)
        for _ in range(self.schedule_substeps):
            candidate, candidate_runtime, first_transport, status, margin = (
                self._transport_step(
                    candidate_runtime,
                    0.5 * schedule_step,
                    transport_args,
                )
            )
            candidate, chemical, element, charge = self._chemistry_step(
                candidate, schedule_step, chemistry_runtime
            )
            candidate_runtime = self._replace_runtime_average(
                candidate_runtime,
                candidate,
                candidate_runtime.time,
            )
            candidate, candidate_runtime, second_transport, status, second_margin = (
                self._transport_step(
                    candidate_runtime,
                    0.5 * schedule_step,
                    transport_args,
                )
            )
            transport_success = transport_success & first_transport & second_transport
            chemistry_success = chemistry_success & chemical
            minimum_margin = jnp.minimum(
                minimum_margin, jnp.minimum(margin, second_margin)
            )
            runtime_status = status
            maximum_element = jnp.maximum(maximum_element, element)
            maximum_charge = jnp.maximum(maximum_charge, charge)
        final_cells = self.layout.evidence(candidate).successful
        time_scale = jnp.maximum(jnp.abs(state.time + step_), 1.0)
        time_consistent = (
            jnp.abs(candidate_runtime.time - (state.time + step_))
            <= 32.0 * jnp.finfo(step_.dtype).eps * time_scale
        )
        accepted = (
            jnp.isfinite(step_)
            & (step_ > 0.0)
            & transport_success
            & chemistry_success
            & time_consistent
            & jnp.all(final_cells)
            & jnp.all(jnp.isfinite(candidate))
        )
        committed = ReactiveAdvanceState(
            candidate_runtime.time,
            candidate,
            candidate_runtime,
            accepted_macro_steps=state.accepted_macro_steps + 1,
            schedule_index=state.schedule_index + self.schedule_substeps,
            state_id=state.state_id,
        )
        output_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(accepted, new, old), committed, state
        )
        evidence = ReactiveAdvanceEvidence(
            state.time,
            step_,
            transport_success,
            runtime_status,
            minimum_margin,
            chemistry_success,
            final_cells,
            maximum_element,
            maximum_charge,
            jnp.sqrt(jnp.sum((candidate - state.conserved) ** 2)),
            ~accepted,
            accepted,
            self.plan_id,
        )
        return ReactiveAdvanceResult(output_state, evidence)


class ReactiveIMEXPlan(StrictModule, NonTrainableState):
    """Explicit FV-runtime transport and implicit trapezoidal chemistry."""

    strang: ReactiveStrangPlan
    nonlinear_iterations: int = eqx.field(static=True)
    nonlinear_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: PreparedFiniteVolumeRuntime,
        mechanism: CompiledChemicalMechanism,
        /,
        *,
        nonlinear_iterations: int = 12,
        nonlinear_tolerance: float = 1.0e-9,
    ):
        iterations = int(nonlinear_iterations)
        tolerance = float(nonlinear_tolerance)
        if iterations < 1 or not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("IMEX nonlinear controls are invalid.")
        self.strang = ReactiveStrangPlan(transport, mechanism)
        self.nonlinear_iterations = iterations
        self.nonlinear_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-reactive-imex",
                "transport_runtime": transport.runtime_id,
                "mechanism": mechanism.mechanism_id,
                "nonlinear_iterations": iterations,
                "nonlinear_tolerance": tolerance,
            }
        )

    def initial_state(
        self, conserved: ArrayLike, /, *, time: ArrayLike = 0.0
    ) -> ReactiveAdvanceState:
        base = self.strang.initial_state(conserved, time=time)
        return ReactiveAdvanceState(
            base.time,
            base.conserved,
            base.transport_runtime_state,
            state_id=canonical_fingerprint(
                {
                    "kind": "reactive-imex-state",
                    "plan": self.plan_id,
                    "shape": list(base.conserved.shape),
                }
            ),
        )

    def advance(
        self,
        state: ReactiveAdvanceState,
        step: ArrayLike,
        /,
        *,
        transport_args: Any = None,
        chemistry_runtime: ChemicalRateRuntime | None = None,
    ) -> ReactiveAdvanceResult:
        if not isinstance(state, ReactiveAdvanceState):
            raise TypeError("state must be ReactiveAdvanceState.")
        step_ = jnp.asarray(step, dtype=state.conserved.dtype)
        if step_.shape != ():
            raise ValueError("Reactive macro step must be scalar.")
        start = state.conserved
        scheduled = self.strang.transport.advance_prescribed(
            state.transport_runtime_state,
            step_,
            transport_args,
        )
        explicit_base = self.strang._runtime_average(scheduled.runtime_state)
        chemistry_start, chemistry_start_valid, element, charge = (
            self.strang._chemistry_source(start, chemistry_runtime)
        )
        candidate = explicit_base + step_ * chemistry_start
        chemistry_success = jnp.all(chemistry_start_valid)
        maximum_element = jnp.max(element)
        maximum_charge = jnp.max(charge)
        nonlinear_residual = jnp.asarray(jnp.inf, dtype=start.dtype)
        for _ in range(self.nonlinear_iterations):
            chemistry_end, chemistry_valid, element, charge = (
                self.strang._chemistry_source(candidate, chemistry_runtime)
            )
            updated = explicit_base + 0.5 * step_ * (chemistry_start + chemistry_end)
            nonlinear_residual = jnp.sqrt(jnp.sum((updated - candidate) ** 2))
            candidate = updated
            chemistry_success = chemistry_success & jnp.all(chemistry_valid)
            maximum_element = jnp.maximum(maximum_element, jnp.max(element))
            maximum_charge = jnp.maximum(maximum_charge, jnp.max(charge))
        final_cells = self.strang.layout.evidence(candidate).successful
        scale = jnp.maximum(jnp.sqrt(jnp.sum(candidate**2)), 1.0)
        converged = nonlinear_residual <= self.nonlinear_tolerance * scale
        candidate_runtime = self.strang._replace_runtime_average(
            scheduled.runtime_state,
            candidate,
            scheduled.runtime_state.time,
        )
        accepted = (
            scheduled.accepted
            & chemistry_success
            & converged
            & jnp.all(final_cells)
            & jnp.all(jnp.isfinite(candidate))
        )
        committed = ReactiveAdvanceState(
            candidate_runtime.time,
            candidate,
            candidate_runtime,
            accepted_macro_steps=state.accepted_macro_steps + 1,
            schedule_index=state.schedule_index + 1,
            state_id=state.state_id,
        )
        output_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(accepted, new, old), committed, state
        )
        evidence = ReactiveAdvanceEvidence(
            state.time,
            step_,
            scheduled.accepted,
            scheduled.runtime_state.last_status,
            scheduled.stability_margin,
            chemistry_success & converged,
            final_cells,
            maximum_element,
            maximum_charge,
            jnp.sqrt(jnp.sum((candidate - start) ** 2)),
            ~accepted,
            accepted,
            self.plan_id,
        )
        return ReactiveAdvanceResult(output_state, evidence)


__all__ = [
    "ReactiveAdvanceEvidence",
    "ReactiveAdvanceResult",
    "ReactiveAdvanceState",
    "ReactiveIMEXPlan",
    "ReactiveStrangPlan",
]
