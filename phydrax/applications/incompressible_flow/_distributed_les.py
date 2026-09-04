#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._ssp_runge_kutta import (
    ssprk33_step_with_evidence,
    ssprk54_step_with_evidence,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral._coordinates import HermitianSpectralCoordinates
from ...discretization.spectral._distributed_les import (
    DistributedPeriodicLESPlan,
    DistributedPeriodicLESStage,
    DistributedPeriodicLESStepRestriction,
    PreparedDistributedPeriodicLES,
)
from ...equations._incompressible import IncompressibleFlowProblem
from ...solver._etdrk import _etdrk_update
from ...solver._fixed_step import (
    AbstractFixedStepMethod,
    FixedStepResult,
    RetriedFixedStepResult,
    RobustRetryPolicy,
)
from ...solver._production_runtime import (
    ArtifactCheckpointStore,
    ProductionCaseManifest,
    ProductionRunPlan,
    ProductionRunResult,
    ProductionRunState,
    ProductionTriggerBinding,
)
from ...solver._runtime_lifecycle import (
    ByteBoundedAsyncPublisher,
    ExactTimeSchedule,
    RuntimeCheckpointEncodingPlan,
)
from ._forcing import ConstantPowerFourierForcingPlan
from ._production import (
    _output_schedule,
    _PreparedProductionRoute,
    _runtime_values,
    _statistics_moment,
    _statistics_window,
    StatisticsWeighting,
)


DistributedPeriodicLESMethod: TypeAlias = Literal[
    "etdrk2", "etdrk4", "ssprk33", "ssprk54"
]


class DistributedPeriodicLESRateComponents(StrictModule):
    """Term-resolved rates of one distributed periodic LES evaluation."""

    advective_rate: Array
    molecular_rate: Array
    algebraic_les_rate: Array
    forcing_rate: Array
    nonlinear_rate: Array
    total_rate: Array


class DistributedPeriodicIncompressibleStage(StrictModule):
    """One complete rotational incompressible LES equation evaluation."""

    rates: DistributedPeriodicLESRateComponents
    pressure_driving_unprojected_rate: Array
    algebraic_les: DistributedPeriodicLESStage
    forcing_active: Array
    forcing_successful: Array
    finite: Array
    sharding_preserved: bool = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)


class _DistributedPeriodicFullFlowDrift(StrictModule):
    problem: IncompressibleFlowProblem
    backend: PreparedDistributedPeriodicLES
    constant_power_forcing: ConstantPowerFourierForcingPlan | None
    forcing_id: str = eqx.field(static=True)
    nonlinear_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        backend: PreparedDistributedPeriodicLES,
        constant_power_forcing: ConstantPowerFourierForcingPlan | None,
        /,
    ):
        if problem.spatial_dimension != 3:
            raise ValueError(
                "Distributed periodic LES requires a three-dimensional problem."
            )
        forcing = constant_power_forcing
        if forcing is not None and not isinstance(
            forcing, ConstantPowerFourierForcingPlan
        ):
            raise TypeError("constant_power_forcing has the wrong type.")
        if forcing is not None and problem.forcing is not None:
            raise ValueError(
                "Compiled problem forcing and constant-power forcing are mutually exclusive."
            )
        discretization = backend.scientific.grid_filter.discretization
        projector = backend.scientific.projector
        if forcing is not None and (
            forcing.discretization_id != discretization.prepared_id
            or forcing.projector_id != projector.projector_id
        ):
            raise ValueError("Constant-power forcing belongs to another periodic grid.")
        forcing_id = forcing.forcing_id if forcing is not None else problem.forcing_id
        self.problem = problem
        self.backend = backend
        self.constant_power_forcing = forcing
        self.forcing_id = forcing_id
        self.nonlinear_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-incompressible-les-drift",
                "problem": problem.problem_id,
                "backend": backend.prepared_id,
                "forcing": forcing_id,
                "advection": "dealiased-rotational",
                "constraint": "distributed-leray",
                "sgs": backend.scientific.prepared_id,
            }
        )

    def _forcing(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[Array, Array, Array, Array]:
        forcing = self.constant_power_forcing
        if forcing is not None:
            result = forcing.evaluate(state)
            unprojected = self.backend.validate_state(
                result.forcing, owner="Distributed constant-power forcing"
            )
            return (
                unprojected,
                self.backend.project(unprojected),
                result.active,
                result.successful,
            )
        if self.problem.forcing is None:
            zero = jnp.zeros_like(state)
            return zero, zero, jnp.asarray(False), jnp.asarray(True)
        unprojected = self.backend.validate_state(
            self.problem.forcing(time, state, args),
            owner="Distributed modal forcing",
        )
        value = self.backend.project(unprojected)
        finite = self.backend.execution.global_all(
            jnp.all(jnp.isfinite(unprojected), axis=-1)
        )
        return unprojected, value, jnp.asarray(True), finite

    def stage(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> DistributedPeriodicIncompressibleStage:
        value = self.backend.validate_state(
            state, owner="Distributed periodic LES velocity"
        )
        advective_unprojected = self.backend.rotational_unprojected_rate(value)
        advective = self.backend.project(advective_unprojected)
        les_stage = self.backend.evaluate(value)
        (
            forcing_unprojected,
            forcing,
            forcing_active,
            forcing_successful,
        ) = self._forcing(jnp.asarray(time), value, args)
        pressure_driving = self.backend.zero_forbidden_modes(
            advective_unprojected + les_stage.unprojected_rate + forcing_unprojected
        )
        nonlinear = self.backend.project(pressure_driving)
        viscosity = self.problem.viscosity.astype(value.real.dtype)
        molecular = (
            -viscosity.astype(value.dtype)
            * jnp.real(self.backend.wavenumber_squared).astype(value.dtype)[..., None]
            * value
        )
        molecular = self.backend.zero_forbidden_modes(molecular)
        total = self.backend.zero_forbidden_modes(molecular + nonlinear)
        finite = (
            les_stage.finite
            & self.backend.execution.global_all(jnp.all(jnp.isfinite(total), axis=-1))
            & jnp.isfinite(viscosity)
        )
        return DistributedPeriodicIncompressibleStage(
            rates=DistributedPeriodicLESRateComponents(
                advective_rate=advective,
                molecular_rate=molecular,
                algebraic_les_rate=les_stage.projected_rate,
                forcing_rate=forcing,
                nonlinear_rate=nonlinear,
                total_rate=total,
            ),
            pressure_driving_unprojected_rate=pressure_driving,
            algebraic_les=les_stage,
            forcing_active=forcing_active,
            forcing_successful=forcing_successful,
            finite=finite,
            sharding_preserved=True,
            dynamics_id=self.nonlinear_id,
        )

    def nonlinear(self, time: Array, state: Array, args: Any, /) -> Array:
        return self.stage(time, state, args).rates.nonlinear_rate

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.stage(time, state, args).rates.total_rate


class CompiledDistributedPeriodicLESDynamics(StrictModule, NonTrainableState):
    """Full distributed rotational Navier--Stokes and algebraic-LES dynamics."""

    source_plan: DistributedPeriodicLESPlan
    backend: PreparedDistributedPeriodicLES
    problem: IncompressibleFlowProblem
    drift: _DistributedPeriodicFullFlowDrift
    diagonal: Array
    forcing_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    qualification_inherited: bool = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        source_plan: DistributedPeriodicLESPlan,
        /,
        *,
        constant_power_forcing: ConstantPowerFourierForcingPlan | None = None,
    ):
        if not isinstance(problem, IncompressibleFlowProblem):
            raise TypeError("problem must be an IncompressibleFlowProblem.")
        if not isinstance(source_plan, DistributedPeriodicLESPlan):
            raise TypeError("source_plan must be a DistributedPeriodicLESPlan.")
        backend = source_plan.prepare()
        drift = _DistributedPeriodicFullFlowDrift(
            problem, backend, constant_power_forcing
        )
        dtype = jnp.dtype(
            backend.scientific.grid_filter.discretization.plan.precision.coefficient_dtype
        )
        diagonal = backend.execution.place_batched(
            (
                -problem.viscosity.astype(dtype)
                * jnp.real(backend.wavenumber_squared).astype(dtype)[..., None]
            ),
            representation="modal",
        )
        compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-distributed-periodic-incompressible-les",
                "problem": problem.problem_id,
                "source_plan": source_plan.plan_id,
                "backend": backend.prepared_id,
                "execution": backend.execution.plan_id,
                "topology": backend.execution.topology.topology_id,
                "layout": backend.execution.modal_layout.layout_id,
                "nonlinear": drift.nonlinear_id,
                "linear": {
                    "kind": "molecular-fourier-diagonal",
                    "problem": problem.problem_id,
                },
                "forcing": drift.forcing_id,
                "qualification": "backend-specific-not-inherited",
            }
        )
        self.source_plan = source_plan
        self.backend = backend
        self.problem = problem
        self.drift = drift
        self.diagonal = diagonal
        self.forcing_id = drift.forcing_id
        self.compilation_id = compilation_id
        self.qualification_inherited = False

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.backend.execution.modal_layout.global_shape

    def validate_state(self, state: ArrayLike, /) -> Array:
        return self.backend.validate_state(
            state, owner="Compiled distributed periodic LES velocity"
        )

    def project_state(self, state: ArrayLike, /) -> Array:
        return self.backend.project(state)

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        value = self.backend.project(state)
        return jnp.real(self.backend.execution.to_physical_batched(value))

    def stage(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> DistributedPeriodicIncompressibleStage:
        return self.drift.stage(time, state, args)

    def step_restriction(
        self,
        state: ArrayLike,
        /,
        *,
        algebraic_les_stage: DistributedPeriodicLESStage | None = None,
    ) -> DistributedPeriodicLESStepRestriction:
        return self.backend.step_restriction(
            state,
            self.problem.viscosity,
            stage=algebraic_les_stage,
        )

    def nonlinear(self, time: Array, state: Array, args: Any, /) -> Array:
        return self.drift.nonlinear(time, state, args)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.drift(time, state, args)


def compile_distributed_periodic_les(
    problem: IncompressibleFlowProblem,
    source_plan: DistributedPeriodicLESPlan,
    /,
    *,
    constant_power_forcing: ConstantPowerFourierForcingPlan | None = None,
) -> CompiledDistributedPeriodicLESDynamics:
    """Compile one resource-preflighted distributed full-flow LES realization."""

    return CompiledDistributedPeriodicLESDynamics(
        problem,
        source_plan,
        constant_power_forcing=constant_power_forcing,
    )


class DistributedPeriodicLESMethodPlan(StrictModule, NonTrainableState):
    """Select a fixed-step method and its distributed current-state guard."""

    method: DistributedPeriodicLESMethod = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: DistributedPeriodicLESMethod = "etdrk4",
        /,
        *,
        safety_factor: float = 0.8,
    ):
        if method not in ("etdrk2", "etdrk4", "ssprk33", "ssprk54"):
            raise ValueError("Distributed LES method is unsupported.")
        safety = float(safety_factor)
        if not np.isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError("safety_factor must be finite and lie in (0, 1].")
        self.method = method
        self.safety_factor = safety
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-fixed-step-plan",
                "method": method,
                "safety_factor": safety,
                "restriction": "distributed-current-state-global",
            }
        )

    def prepare(
        self,
        dynamics: CompiledDistributedPeriodicLESDynamics,
        coordinates: HermitianSpectralCoordinates,
        /,
    ) -> PreparedDistributedPeriodicLESMethod:
        return PreparedDistributedPeriodicLESMethod(self, dynamics, coordinates)


class PreparedDistributedPeriodicLESMethod(AbstractFixedStepMethod):
    """ETDRK or SSPRK transition with exact distributed LES admission."""

    plan: DistributedPeriodicLESMethodPlan
    dynamics: CompiledDistributedPeriodicLESDynamics
    coordinates: HermitianSpectralCoordinates
    stage_count: int = eqx.field(static=True)
    method: DistributedPeriodicLESMethod = eqx.field(static=True)
    order: int = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DistributedPeriodicLESMethodPlan,
        dynamics: CompiledDistributedPeriodicLESDynamics,
        coordinates: HermitianSpectralCoordinates,
        /,
    ):
        if not isinstance(plan, DistributedPeriodicLESMethodPlan):
            raise TypeError("plan must be a DistributedPeriodicLESMethodPlan.")
        if not isinstance(dynamics, CompiledDistributedPeriodicLESDynamics):
            raise TypeError("dynamics has the wrong compiled type.")
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError("coordinates must be HermitianSpectralCoordinates.")
        if coordinates.state_shape != dynamics.state_shape:
            raise ValueError("Hermitian coordinates and distributed dynamics disagree.")
        if plan.method == "ssprk33":
            order = 3
            stage_count = 3
        elif plan.method == "ssprk54":
            order = 4
            stage_count = 5
        else:
            order = 2 if plan.method == "etdrk2" else 4
            stage_count = order
        self.plan = plan
        self.dynamics = dynamics
        self.coordinates = coordinates
        self.stage_count = stage_count
        self.method = plan.method
        self.order = order
        self.safety_factor = plan.safety_factor
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-periodic-les-fixed-step",
                "plan": plan.plan_id,
                "dynamics": dynamics.compilation_id,
                "backend": dynamics.backend.prepared_id,
                "coordinates": coordinates.coordinate_id,
                "execution": dynamics.backend.execution.plan_id,
                "topology": dynamics.backend.execution.topology.topology_id,
                "layout": dynamics.backend.execution.modal_layout.layout_id,
                "qualification": "backend-specific-not-inherited",
            }
        )

    def _boundary(self, state: ArrayLike, /) -> tuple[Array, Array, Array]:
        value = self.dynamics.validate_state(state)
        finite = self.dynamics.backend.execution.global_all(
            jnp.all(jnp.isfinite(value), axis=-1)
        )
        reality_defect = self.coordinates.reality_defect(value)
        hermitian = self.coordinates.project(value)
        projected = self.dynamics.project_state(hermitian)
        correction = projected - value
        correction_norm = jnp.sqrt(
            jnp.maximum(
                jnp.real(
                    self.dynamics.backend.execution.global_inner_product(
                        correction, correction
                    )
                ),
                0.0,
            )
        )
        valid = finite & (reality_defect <= self.coordinates.reality_tolerance)
        return projected, valid, jnp.maximum(reality_defect, correction_norm)

    def step_restriction(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> DistributedPeriodicLESStepRestriction:
        value = self.dynamics.validate_state(state)
        stage = self.dynamics.stage(time, value, args)
        return self.dynamics.step_restriction(
            value,
            algebraic_les_stage=stage.algebraic_les,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        value = self.dynamics.validate_state(state)
        step = jnp.asarray(step_size, dtype=value.real.dtype).reshape(())
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step > 0.0)),
            "Distributed LES step size must be finite and positive.",
        )
        start = jnp.asarray(time, dtype=step.dtype).reshape(())
        incoming, incoming_valid, incoming_defect = self._boundary(value)
        stage = self.dynamics.stage(start, incoming, args)
        restriction = self.dynamics.step_restriction(
            incoming,
            algebraic_les_stage=stage.algebraic_les,
        )
        selected = (
            restriction.etdrk_selected
            if self.method.startswith("etdrk")
            else restriction.fully_explicit_selected
        ).astype(step.dtype)
        allowed = jnp.asarray(self.safety_factor, dtype=step.dtype) * selected
        first_rate = (
            stage.rates.nonlinear_rate
            if self.method.startswith("etdrk")
            else stage.rates.total_rate
        )
        rate_finite = self.dynamics.backend.execution.global_all(
            jnp.all(jnp.isfinite(first_rate), axis=-1)
        )
        stable = (
            incoming_valid
            & restriction.finite
            & stage.finite
            & rate_finite
            & (allowed > 0.0)
            & (step <= allowed)
        )

        def advance(_: None) -> FixedStepResult:
            if self.method.startswith("etdrk"):
                candidate = _etdrk_update(
                    self.order,
                    self.dynamics.drift,
                    self.dynamics.diagonal,
                    start,
                    incoming,
                    step,
                    args,
                    None,
                    stage.rates.nonlinear_rate,
                )
                base_successful = jnp.asarray(True)
                work = jnp.asarray(self.order, dtype=jnp.int32)
            else:

                def stage_rate(
                    stage_time: Array, stage_state: Array, stage_args: Any
                ) -> Array:
                    return jax.lax.cond(
                        stage_time == start,
                        lambda _: stage.rates.total_rate,
                        lambda _: self.dynamics(stage_time, stage_state, stage_args),
                        operand=None,
                    )

                advance_ssprk = (
                    ssprk33_step_with_evidence
                    if self.method == "ssprk33"
                    else ssprk54_step_with_evidence
                )
                base = advance_ssprk(stage_rate, start, incoming, step, args)
                candidate = base.state
                base_successful = base.successful
                work = jnp.asarray(self.stage_count, dtype=jnp.int32)
            accepted_candidate, candidate_valid, defect = self._boundary(candidate)
            successful = base_successful & candidate_valid
            accepted = jnp.where(successful, accepted_candidate, value)
            correction = accepted_candidate - candidate
            correction_norm = jnp.sqrt(
                jnp.maximum(
                    jnp.real(
                        self.dynamics.backend.execution.global_inner_product(
                            correction, correction
                        )
                    ),
                    0.0,
                )
            )
            return FixedStepResult(
                candidate_state=candidate,
                accepted_state=accepted,
                successful=successful,
                residual=jnp.maximum(incoming_defect, defect),
                iterations=jnp.asarray(0, dtype=jnp.int32),
                work=work,
                transform_applied=successful & (correction_norm > 0.0),
                transform_correction_norm=jnp.where(
                    successful,
                    correction_norm,
                    jnp.zeros_like(correction_norm),
                ),
            )

        def reject(_: None) -> FixedStepResult:
            finite_limit = jnp.isfinite(allowed) & (allowed > 0.0)
            safe_limit = jnp.where(finite_limit, allowed, jnp.ones_like(allowed))
            violation = jnp.maximum(step / safe_limit - 1.0, 0.0)
            valid_evidence = (
                incoming_valid & restriction.finite & stage.finite & rate_finite
            )
            residual = jnp.where(
                valid_evidence,
                violation,
                jnp.asarray(jnp.inf, dtype=step.dtype),
            )
            return FixedStepResult(
                candidate_state=value,
                accepted_state=value,
                successful=jnp.asarray(False),
                residual=residual,
                iterations=jnp.asarray(0, dtype=jnp.int32),
                work=jnp.asarray(1, dtype=jnp.int32),
                transform_applied=jnp.asarray(False),
                transform_correction_norm=jnp.zeros((), dtype=step.dtype),
            )

        result = jax.lax.cond(stable, advance, reject, operand=None)
        candidate = self.dynamics.backend.execution.place_batched(
            result.candidate_state, representation="modal"
        )
        accepted = self.dynamics.backend.execution.place_batched(
            result.accepted_state, representation="modal"
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=result.successful,
            residual=result.residual,
            iterations=result.iterations,
            work=result.work,
            transform_applied=result.transform_applied,
            transform_correction_norm=result.transform_correction_norm,
        )


class DistributedPeriodicLESStatistics(StrictModule):
    """Globally reduced statistics evaluated without materializing a host field."""

    kinetic_energy: Array
    molecular_dissipation: Array
    advective_energy_rate: Array
    algebraic_les_energy_rate: Array
    forcing_power: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    imaginary_leakage: Array
    maximum_kinematic_viscosity: Array
    advective_step_limit: Array
    selected_step_limit: Array
    finite: Array
    successful: Array
    sharding_preserved: bool = eqx.field(static=True)
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    statistics_id: str = eqx.field(static=True)


class DistributedPeriodicLESStatisticsPlan(StrictModule, NonTrainableState):
    """Sharding-preserving scalar statistics for one compiled distributed flow."""

    dynamics: CompiledDistributedPeriodicLESDynamics
    coordinates: HermitianSpectralCoordinates
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledDistributedPeriodicLESDynamics,
        coordinates: HermitianSpectralCoordinates,
        /,
    ):
        if not isinstance(dynamics, CompiledDistributedPeriodicLESDynamics):
            raise TypeError("dynamics has the wrong compiled type.")
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError("coordinates must be HermitianSpectralCoordinates.")
        if coordinates.state_shape != dynamics.state_shape:
            raise ValueError("Statistics coordinates and dynamics disagree.")
        self.dynamics = dynamics
        self.coordinates = coordinates
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-statistics",
                "dynamics": dynamics.compilation_id,
                "backend": dynamics.backend.prepared_id,
                "execution": dynamics.backend.execution.plan_id,
                "coordinates": coordinates.coordinate_id,
                "reductions": dynamics.backend.execution.modal_layout.used_mesh_axes,
                "host_gather": False,
            }
        )

    def evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        stage: DistributedPeriodicIncompressibleStage | None = None,
        restriction: DistributedPeriodicLESStepRestriction | None = None,
        method: DistributedPeriodicLESMethod = "etdrk4",
    ) -> DistributedPeriodicLESStatistics:
        value = self.dynamics.validate_state(state)
        stage_ = self.dynamics.stage(time, value, args) if stage is None else stage
        if not isinstance(stage_, DistributedPeriodicIncompressibleStage):
            raise TypeError("stage must be a distributed incompressible stage or None.")
        if stage_.dynamics_id != self.dynamics.drift.nonlinear_id:
            raise ValueError("Stage belongs to another distributed dynamics identity.")
        restriction_ = (
            self.dynamics.step_restriction(
                value, algebraic_les_stage=stage_.algebraic_les
            )
            if restriction is None
            else restriction
        )
        if not isinstance(restriction_, DistributedPeriodicLESStepRestriction):
            raise TypeError("restriction has the wrong distributed type.")
        if restriction_.backend_id != self.dynamics.backend.prepared_id:
            raise ValueError("Restriction belongs to another distributed backend.")
        backend = self.dynamics.backend
        live = backend.grid_filter.apply(value)

        def energy_rate(rate: Array, /) -> Array:
            return jnp.real(backend.execution.global_inner_product(live, rate))

        kinetic_energy = 0.5 * jnp.real(
            backend.execution.global_inner_product(live, live)
        )
        advective_energy_rate = energy_rate(stage_.rates.advective_rate)
        molecular_energy_rate = energy_rate(stage_.rates.molecular_rate)
        algebraic_les_energy_rate = energy_rate(stage_.rates.algebraic_les_rate)
        forcing_power = energy_rate(stage_.rates.forcing_rate)
        semidiscrete_energy_rate = energy_rate(stage_.rates.total_rate)
        molecular_dissipation = -molecular_energy_rate
        energy_balance_defect = semidiscrete_energy_rate - (
            forcing_power
            - molecular_dissipation
            - stage_.algebraic_les.modeled_dissipation
        )
        divergence_norm = backend.execution.diagnostics_batched(
            backend.divergence(live)
        ).l2_norm
        imaginary_leakage = backend.execution.diagnostics_batched(
            jnp.imag(backend.execution.to_physical_batched(live)),
            representation="physical",
        ).maximum_absolute
        selected = (
            restriction_.etdrk_selected
            if method.startswith("etdrk")
            else restriction_.fully_explicit_selected
        )
        scalars = jnp.stack(
            (
                kinetic_energy,
                molecular_dissipation,
                advective_energy_rate,
                algebraic_les_energy_rate,
                forcing_power,
                semidiscrete_energy_rate,
                energy_balance_defect,
                divergence_norm,
                imaginary_leakage,
                stage_.algebraic_les.maximum_kinematic_viscosity,
            )
        )
        finite = (
            stage_.finite
            & restriction_.finite
            & backend.execution.global_all(jnp.all(jnp.isfinite(value), axis=-1))
            & jnp.all(jnp.isfinite(scalars))
            & (jnp.isfinite(restriction_.advective) | jnp.isinf(restriction_.advective))
            & (jnp.isfinite(selected) | jnp.isinf(selected))
        )
        successful = (
            finite
            & stage_.algebraic_les.dissipative
            & stage_.algebraic_les.energy_consistent
            & stage_.forcing_successful
        )
        return DistributedPeriodicLESStatistics(
            kinetic_energy=kinetic_energy,
            molecular_dissipation=molecular_dissipation,
            advective_energy_rate=advective_energy_rate,
            algebraic_les_energy_rate=algebraic_les_energy_rate,
            forcing_power=forcing_power,
            semidiscrete_energy_rate=semidiscrete_energy_rate,
            energy_balance_defect=energy_balance_defect,
            divergence_norm=divergence_norm,
            imaginary_leakage=imaginary_leakage,
            maximum_kinematic_viscosity=stage_.algebraic_les.maximum_kinematic_viscosity,
            advective_step_limit=restriction_.advective,
            selected_step_limit=selected,
            finite=finite,
            successful=successful,
            sharding_preserved=True,
            reduction_axes=backend.execution.modal_layout.used_mesh_axes,
            dynamics_id=self.dynamics.compilation_id,
            statistics_id=self.plan_id,
        )


class _DistributedPeriodicLESStatisticsEvaluator(StrictModule):
    method: PreparedDistributedPeriodicLESMethod
    statistics: DistributedPeriodicLESStatisticsPlan
    evaluator_id: str = eqx.field(static=True)
    value_size: int = eqx.field(static=True)

    def __init__(
        self,
        method: PreparedDistributedPeriodicLESMethod,
        statistics: DistributedPeriodicLESStatisticsPlan,
        /,
    ):
        self.method = method
        self.statistics = statistics
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-production-statistics",
                "method": method.method_id,
                "statistics": statistics.plan_id,
                "cadence": "every-accepted-step",
                "host_gather": False,
            }
        )
        self.value_size = 13

    def snapshot(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> DistributedPeriodicLESStatistics:
        return self.statistics.evaluate(time, state, args, method=self.method.method)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        result = self.snapshot(time, state, args)
        return jnp.stack(
            (
                result.kinetic_energy,
                result.molecular_dissipation,
                result.advective_energy_rate,
                result.algebraic_les_energy_rate,
                result.forcing_power,
                result.semidiscrete_energy_rate,
                result.energy_balance_defect,
                result.divergence_norm,
                result.imaginary_leakage,
                result.maximum_kinematic_viscosity,
                jnp.where(
                    jnp.isfinite(result.advective_step_limit),
                    result.advective_step_limit,
                    0.0,
                ),
                jnp.where(
                    jnp.isfinite(result.selected_step_limit),
                    result.selected_step_limit,
                    0.0,
                ),
                result.successful.astype(result.kinetic_energy.dtype),
            )
        )


class DistributedPeriodicLESProductionCase(StrictModule, NonTrainableState):
    """Exact scientific identity for one distributed periodic LES initial value."""

    case_id: str = eqx.field(static=True)
    source_problem_id: str = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    scientific_prepared_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    initial_condition_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledDistributedPeriodicLESDynamics,
        initial_velocity: ArrayLike,
        /,
        *,
        case_id: str,
    ):
        if not isinstance(dynamics, CompiledDistributedPeriodicLESDynamics):
            raise TypeError("dynamics has the wrong compiled distributed type.")
        label = str(case_id)
        if not label:
            raise ValueError("case_id must be nonempty.")
        velocity = dynamics.validate_state(initial_velocity)
        concrete = np.asarray(velocity)
        if np.any(~np.isfinite(concrete)):
            raise ValueError("Distributed initial velocity must be finite.")
        backend = dynamics.backend
        scientific = backend.scientific
        discretization = scientific.grid_filter.discretization
        filter_id = scientific.grid_filter.plan.resolved_filter.filter_id
        initial_condition_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-initial-condition",
                "source_plan": dynamics.source_plan.plan_id,
                "compilation": dynamics.compilation_id,
                "scientific_prepared": scientific.prepared_id,
                "backend": backend.prepared_id,
                "discretization": discretization.prepared_id,
                "filter": filter_id,
                "topology": backend.execution.topology.topology_id,
                "layout": backend.execution.modal_layout.layout_id,
                "field": array_tree_fingerprint(concrete),
            }
        )
        self.case_id = label
        self.source_problem_id = dynamics.problem.problem_id
        self.source_plan_id = dynamics.source_plan.plan_id
        self.compilation_id = dynamics.compilation_id
        self.scientific_prepared_id = scientific.prepared_id
        self.backend_id = backend.prepared_id
        self.discretization_id = discretization.prepared_id
        self.filter_id = filter_id
        self.topology_id = backend.execution.topology.topology_id
        self.layout_id = backend.execution.modal_layout.layout_id
        self.state_shape = tuple(velocity.shape)
        self.state_dtype = str(velocity.dtype)
        self.initial_condition_id = initial_condition_id
        self.identity_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-production-case",
                "case": label,
                "source_problem": self.source_problem_id,
                "source_plan": self.source_plan_id,
                "compilation": self.compilation_id,
                "scientific_prepared": self.scientific_prepared_id,
                "backend": self.backend_id,
                "discretization": self.discretization_id,
                "filter": self.filter_id,
                "topology": self.topology_id,
                "layout": self.layout_id,
                "initial_condition": initial_condition_id,
            }
        )

    def validate_initial_condition(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity)
        if tuple(value.shape) != self.state_shape or str(value.dtype) != self.state_dtype:
            raise ValueError(
                "Distributed initial velocity shape or dtype differs from the bound case."
            )
        candidate = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-initial-condition",
                "source_plan": self.source_plan_id,
                "compilation": self.compilation_id,
                "scientific_prepared": self.scientific_prepared_id,
                "backend": self.backend_id,
                "discretization": self.discretization_id,
                "filter": self.filter_id,
                "topology": self.topology_id,
                "layout": self.layout_id,
                "field": array_tree_fingerprint(value),
            }
        )
        if candidate != self.initial_condition_id:
            raise ValueError(
                "Distributed initial velocity differs from the bound production case."
            )
        return value


class DistributedPeriodicLESProductionPlan(StrictModule, NonTrainableState):
    """Exact production assembly consuming a distributed LES execution plan."""

    source_plan: DistributedPeriodicLESPlan
    dynamics: CompiledDistributedPeriodicLESDynamics
    case: DistributedPeriodicLESProductionCase
    method_plan: DistributedPeriodicLESMethodPlan
    method: PreparedDistributedPeriodicLESMethod
    coordinates: HermitianSpectralCoordinates
    statistics: DistributedPeriodicLESStatisticsPlan
    statistics_evaluator: _DistributedPeriodicLESStatisticsEvaluator
    manifest: ProductionCaseManifest
    runtime_plan: ProductionRunPlan
    checkpoint_encoding: RuntimeCheckpointEncodingPlan
    output_schedule: ExactTimeSchedule | None
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    source_problem_id: str = eqx.field(static=True)
    initial_condition_id: str = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    checkpoint_retention: int = eqx.field(static=True)
    qualification_inherited: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        source_plan: DistributedPeriodicLESPlan,
        method: DistributedPeriodicLESMethodPlan,
        case: DistributedPeriodicLESProductionCase,
        /,
        *,
        start_time: float,
        end_time: float,
        step_size: float,
        checkpoint_interval: int,
        constant_power_forcing: ConstantPowerFourierForcingPlan | None = None,
        output_times: ArrayLike | None = None,
        output_tolerance: float = 1.0e-12,
        maximum_steps: int | None = None,
        segment_steps: int = 32,
        checkpoint_retention: int = 3,
        retry_policy: RobustRetryPolicy | None = None,
        trigger_bindings: Sequence[ProductionTriggerBinding] = (),
        statistics_weighting: StatisticsWeighting = "time",
        statistics_window_start: float | None = None,
        statistics_window_end: float | None = None,
        statistics_batch_duration: float | None = None,
        maximum_statistics_batches: int = 0,
    ):
        if not isinstance(source_plan, DistributedPeriodicLESPlan):
            raise TypeError("source_plan must be a DistributedPeriodicLESPlan.")
        if source_plan.checkpoint_count < 1:
            raise ValueError(
                "Distributed production requires checkpoint_count>=1 in the resource plan."
            )
        if not isinstance(method, DistributedPeriodicLESMethodPlan):
            raise TypeError("method must be a DistributedPeriodicLESMethodPlan.")
        if not isinstance(case, DistributedPeriodicLESProductionCase):
            raise TypeError("case must be a DistributedPeriodicLESProductionCase.")
        dynamics = compile_distributed_periodic_les(
            problem,
            source_plan,
            constant_power_forcing=constant_power_forcing,
        )
        coordinates = HermitianSpectralCoordinates(
            dynamics.backend.scientific.grid_filter.discretization,
            component_shape=(3,),
        )
        scientific = dynamics.backend.scientific
        discretization = scientific.grid_filter.discretization
        filter_id = scientific.grid_filter.plan.resolved_filter.filter_id
        if (
            case.source_problem_id != problem.problem_id
            or case.source_plan_id != source_plan.plan_id
            or case.compilation_id != dynamics.compilation_id
            or case.scientific_prepared_id != scientific.prepared_id
            or case.backend_id != dynamics.backend.prepared_id
            or case.discretization_id != discretization.prepared_id
            or case.filter_id != filter_id
            or case.topology_id != dynamics.backend.execution.topology.topology_id
            or case.layout_id != dynamics.backend.execution.modal_layout.layout_id
        ):
            raise ValueError(
                "Distributed production case belongs to another compilation."
            )
        prepared_method = method.prepare(dynamics, coordinates)
        statistics = DistributedPeriodicLESStatisticsPlan(dynamics, coordinates)
        evaluator = _DistributedPeriodicLESStatisticsEvaluator(
            prepared_method, statistics
        )
        start, end, step, steps, interval, segment, retention = _runtime_values(
            start_time,
            end_time,
            step_size,
            maximum_steps,
            checkpoint_interval,
            segment_steps,
            checkpoint_retention,
        )
        output = _output_schedule(output_times, start, end, output_tolerance)
        window_start, window_end = _statistics_window(
            start,
            end,
            statistics_window_start,
            statistics_window_end,
        )
        moment = _statistics_moment(
            evaluator,
            evaluator.value_size,
            evaluator_id=evaluator.evaluator_id,
            weighting=statistics_weighting,
            window_start=window_start,
            window_end=window_end,
            batch_duration=statistics_batch_duration,
            maximum_batches=maximum_statistics_batches,
        )
        retry = (
            RobustRetryPolicy(maximum_retries=0) if retry_policy is None else retry_policy
        )
        bindings = tuple(trigger_bindings)
        runtime_plan = ProductionRunPlan(
            prepared_method,
            retry,
            step_size=step,
            end_time=end,
            maximum_steps=steps,
            checkpoint_interval=interval,
            segment_steps=segment,
            output_schedule=output,
            moments=(moment,),
            trigger_bindings=bindings,
            device_resident=True,
        )
        encoding = RuntimeCheckpointEncodingPlan()
        case_identity = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-production-run-case",
                "case": case.identity_id,
                "problem": problem.problem_id,
                "source_plan": source_plan.plan_id,
                "backend": dynamics.backend.prepared_id,
                "dynamics": dynamics.compilation_id,
                "method": prepared_method.method_id,
                "forcing": dynamics.forcing_id,
                "statistics": statistics.plan_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "resource": dynamics.backend.preparation.resource.report_id,
                "qualification": "backend-specific-not-inherited",
            }
        )
        precision = dynamics.backend.scientific.grid_filter.discretization.plan.precision
        manifest = ProductionCaseManifest(
            problem_id=case_identity,
            method_id=prepared_method.method_id,
            precision_id=precision.policy_id,
            topology_id=dynamics.backend.execution.topology.topology_id,
            geometry_layout_id=dynamics.backend.execution.modal_layout.layout_id,
            dtype=coordinates.evidence.source_dtype,
        )
        self.source_plan = source_plan
        self.dynamics = dynamics
        self.method_plan = method
        self.method = prepared_method
        self.coordinates = coordinates
        self.statistics = statistics
        self.statistics_evaluator = evaluator
        self.manifest = manifest
        self.runtime_plan = runtime_plan
        self.checkpoint_encoding = encoding
        self.output_schedule = output
        self.trigger_bindings = bindings
        self.case = case
        self.source_problem_id = problem.problem_id
        self.initial_condition_id = case.initial_condition_id
        self.start_time = start
        self.checkpoint_retention = retention
        self.qualification_inherited = False
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-production-plan",
                "case": case_identity,
                "source_plan": source_plan.plan_id,
                "backend": dynamics.backend.prepared_id,
                "manifest": manifest.manifest_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "resource": dynamics.backend.preparation.resource.report_id,
                "start_time": start,
                "checkpoint_retention": retention,
                "qualification": "backend-specific-not-inherited",
            }
        )

    def prepare(
        self,
        checkpoint: str | Path | ArtifactCheckpointStore,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ) -> PreparedDistributedPeriodicLESProduction:
        return PreparedDistributedPeriodicLESProduction(
            self,
            checkpoint,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )


class PreparedDistributedPeriodicLESProduction(_PreparedProductionRoute):
    """Thin distributed route over the shared production runtime."""

    _prepared_kind = "prepared-distributed-periodic-les-production"

    def __init__(
        self,
        plan: DistributedPeriodicLESProductionPlan,
        checkpoint: str | Path | ArtifactCheckpointStore,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if not isinstance(plan, DistributedPeriodicLESProductionPlan):
            raise TypeError("plan must be DistributedPeriodicLESProductionPlan.")
        self._bind_runtime(
            plan,
            checkpoint,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )

    def _place_run_state(self, state: ProductionRunState, /) -> ProductionRunState:
        if not isinstance(state, ProductionRunState):
            raise TypeError("state must be a ProductionRunState.")
        accepted = self.plan.dynamics.validate_state(state.accepted_state)
        return ProductionRunState(
            state.step_index,
            state.time,
            accepted,
            state.controller_state,
            state.rng_state,
            state.schedule_cursor,
            state.moment_states,
            state.trigger_states,
            state.output_cursor,
            state.status,
            state.last_checkpoint_id,
        )

    def step(
        self, state: ProductionRunState, /
    ) -> tuple[ProductionRunState, RetriedFixedStepResult]:
        following, transition = self.runtime.step(self._place_run_state(state))
        candidate = self.plan.dynamics.validate_state(transition.candidate_state)
        accepted = self.plan.dynamics.validate_state(transition.accepted_state)
        placed_transition = RetriedFixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=transition.successful,
            accepted_step_size=transition.accepted_step_size,
            retry_count=transition.retry_count,
            attempted_step_sizes=transition.attempted_step_sizes,
            decision_id=transition.decision_id,
        )
        return self._place_run_state(following), placed_transition

    def run(self, state: ProductionRunState, /) -> ProductionRunResult:
        result = self.runtime.run(self._place_run_state(state))
        return ProductionRunResult(
            state=self._place_run_state(result.state),
            successful=result.successful,
            failure=result.failure,
            run_id=result.run_id,
        )

    def checkpoint(self, state: ProductionRunState, /) -> ProductionRunState:
        placed = self._place_run_state(state)
        return self._place_run_state(self.runtime.checkpoint(placed))

    def initialize(
        self,
        modal_velocity: ArrayLike,
        /,
        *,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        velocity = self.plan.case.validate_initial_condition(
            self.plan.dynamics.validate_state(modal_velocity)
        )
        finite = self.plan.dynamics.backend.execution.global_all(
            jnp.all(jnp.isfinite(velocity), axis=-1)
        )
        defect = self.plan.coordinates.reality_defect(velocity)
        velocity = eqx.error_if(
            velocity,
            ~(finite & (defect <= self.plan.coordinates.reality_tolerance)),
            "Distributed production initial velocity is nonfinite or non-Hermitian.",
        )
        velocity = self.plan.dynamics.project_state(velocity)
        return self.runtime.initial_state(
            velocity,
            time=self.plan.start_time,
            controller_state=controller_state,
            rng_state=rng_state,
        )

    def resume(self, template: ProductionRunState, /) -> ProductionRunState:
        restored = self.runtime.resume(self._place_run_state(template))
        restored = self._place_run_state(restored)
        expected = self.plan.dynamics.backend.execution.modal_layout.sharding(
            self.plan.dynamics.backend.execution.topology
        )
        if restored.accepted_state.sharding != expected:
            raise RuntimeError(
                "Distributed artifact restart did not retain the bound NamedSharding."
            )
        return restored

    def restart_evidence(self, state: ProductionRunState, /):
        return self.plan.dynamics.backend.restart_evidence(state.accepted_state)

    def statistics_snapshot(
        self, time: ArrayLike, state: ArrayLike, /
    ) -> DistributedPeriodicLESStatistics:
        return self.plan.statistics_evaluator.snapshot(time, state, self.runtime.args)


__all__ = [
    "CompiledDistributedPeriodicLESDynamics",
    "DistributedPeriodicIncompressibleStage",
    "DistributedPeriodicLESMethod",
    "DistributedPeriodicLESMethodPlan",
    "DistributedPeriodicLESProductionCase",
    "DistributedPeriodicLESProductionPlan",
    "DistributedPeriodicLESRateComponents",
    "DistributedPeriodicLESStatistics",
    "DistributedPeriodicLESStatisticsPlan",
    "PreparedDistributedPeriodicLESMethod",
    "PreparedDistributedPeriodicLESProduction",
    "compile_distributed_periodic_les",
]
