#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import FaceVelocity, PreparedMACOperators
from ...discretization.spectral import HermitianSpectralCoordinates
from ...equations import CompiledMACIncompressibleDynamics
from ...equations._incompressible import _PeriodicRotationalDrift
from ...solver._channel_flow import (
    ChannelSBDF2State,
    PreparedChannelSBDF2Method,
)
from ...solver._etdrk import _etdrk_update, ETDRKMethod, PreparedETDRKMethod
from ...solver._fixed_step import (
    AbstractFixedStepMethod,
    FixedStepResult,
    RobustRetryPolicy,
)
from ...solver._production_runtime import (
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    PreparedProductionRun,
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
    RuntimeCheckpointLeafBinding,
    StreamingMomentPlan,
)
from ...solver._semilinear_drift import SemilinearDrift
from ...stochastic import OrnsteinUhlenbeckRealization
from ._forcing import (
    ConstantPowerFourierForcingPlan,
    SolenoidalOUForcingPlan,
    SolenoidalOUForcingState,
)
from ._statistics import (
    MACPlaneWallStatistics,
    MACPlaneWallStatisticsPlan,
    PeriodicModalTurbulenceStatistics,
    PeriodicModalTurbulenceStatisticsPlan,
    SpectralChannelStatistics,
    SpectralChannelStatisticsPlan,
)


StatisticsWeighting = Literal["sample", "time"]
ConstantPowerWiring = Literal["compiled", "adapter"]


def _required_identifier(value: str, role: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{role} must be nonempty.")
    return identifier


def _runtime_values(
    start_time: float,
    end_time: float,
    step_size: float,
    maximum_steps: int | None,
    checkpoint_interval: int,
    segment_steps: int,
    checkpoint_retention: int,
    /,
) -> tuple[float, float, float, int, int, int, int]:
    start = float(start_time)
    end = float(end_time)
    step = float(step_size)
    interval = int(checkpoint_interval)
    segment = int(segment_steps)
    retention = int(checkpoint_retention)
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
        or not math.isfinite(step)
        or step <= 0.0
        or interval <= 0
        or segment <= 0
        or retention <= 0
    ):
        raise ValueError("Production runtime configuration is invalid.")
    required_steps = math.ceil((end - start) / step)
    steps = required_steps if maximum_steps is None else int(maximum_steps)
    if steps < required_steps:
        raise ValueError("maximum_steps cannot reach the declared end_time.")
    return start, end, step, steps, interval, segment, retention


def _output_schedule(
    output_times: ArrayLike | None,
    start_time: float,
    end_time: float,
    tolerance: float,
    /,
) -> ExactTimeSchedule | None:
    if output_times is None:
        return None
    schedule = ExactTimeSchedule(output_times, tolerance=tolerance)
    targets = np.asarray(schedule.targets)
    if targets[0] <= start_time + schedule.tolerance:
        raise ValueError("Production output targets must follow start_time.")
    if targets[-1] > end_time + schedule.tolerance:
        raise ValueError("Production output targets cannot exceed end_time.")
    return schedule


def _statistics_window(
    start_time: float,
    end_time: float,
    window_start: float | None,
    window_end: float | None,
    /,
) -> tuple[float, float]:
    start = start_time if window_start is None else float(window_start)
    end = end_time if window_end is None else float(window_end)
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < start_time
        or end > end_time
        or end <= start
    ):
        raise ValueError("Production statistics window is outside the runtime horizon.")
    return start, end


def _statistics_moment(
    evaluator: Callable,
    value_size: int,
    /,
    *,
    evaluator_id: str,
    weighting: StatisticsWeighting,
    window_start: float,
    window_end: float,
    batch_duration: float | None,
    maximum_batches: int,
) -> StreamingMomentPlan:
    return StreamingMomentPlan(
        evaluator,
        value_shape=(int(value_size),),
        weighting=weighting,
        window_start=window_start,
        window_end=window_end,
        batch_duration=batch_duration,
        maximum_batches=maximum_batches,
        plan_id=evaluator_id,
    )


def _lattice_index(value: float, origin: float, step: float, role: str, /) -> int:
    raw = (float(value) - origin) / step
    nearest = round(raw)
    if raw < 0.0 or not np.isclose(raw, nearest, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{role} is not on the channel SBDF2 step lattice.")
    return int(nearest)


def _compiled_periodic_forcing_id(method: PreparedETDRKMethod, /) -> str | None:
    nonlinear = method.drift.nonlinear_drift
    if isinstance(nonlinear, _PeriodicRotationalDrift):
        return nonlinear.problem.forcing_id
    if isinstance(nonlinear, _ConstantPowerPeriodicNonlinearDrift):
        return nonlinear.forcing_id
    raise TypeError(
        "Compiled forcing wiring requires native periodic incompressible dynamics."
    )


class _ConstantPowerPeriodicNonlinearDrift(StrictModule):
    base: SemilinearDrift
    forcing: ConstantPowerFourierForcingPlan
    forcing_id: str = eqx.field(static=True)
    nonlinear_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: SemilinearDrift,
        forcing: ConstantPowerFourierForcingPlan,
        /,
    ):
        self.base = base
        self.forcing = forcing
        self.forcing_id = forcing.forcing_id
        self.nonlinear_id = canonical_fingerprint(
            {
                "kind": "constant-power-periodic-production-drift-v1",
                "base": base.nonlinear_id,
                "forcing": forcing.forcing_id,
            }
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return (
            self.base.nonlinear(time, state, args) + self.forcing.evaluate(state).forcing
        )


def prepare_constant_power_periodic_method(
    method: PreparedETDRKMethod,
    forcing: ConstantPowerFourierForcingPlan,
    /,
) -> PreparedETDRKMethod:
    """Explicitly add constant-power forcing to one already prepared ETDRK drift."""

    if not isinstance(method, PreparedETDRKMethod):
        raise TypeError("method must be PreparedETDRKMethod.")
    if not isinstance(forcing, ConstantPowerFourierForcingPlan):
        raise TypeError("forcing must be ConstantPowerFourierForcingPlan.")
    coordinates = method.coordinates
    if coordinates is None:
        raise ValueError("Periodic production ETDRK requires Hermitian coordinates.")
    if forcing.discretization_id != coordinates.discretization.prepared_id:
        raise ValueError("Constant-power forcing and ETDRK coordinates use another grid.")
    nonlinear = _ConstantPowerPeriodicNonlinearDrift(method.drift, forcing)
    base = method.drift
    drift = SemilinearDrift(
        base.linear_operator,
        nonlinear,
        state_shape=base.state_shape,
        operator_id=base.operator_id,
        nonlinear_id=nonlinear.nonlinear_id,
        mass_self_adjoint=base.mass_self_adjoint,
        mass_weights=base.mass_weights,
        spectral_bounds=base.spectral_bounds,
        spectral_representation=base.spectral_representation,
        compatible_noise_eigenvalues=base.compatible_noise_eigenvalues,
        compatible_noise_basis_id=base.compatible_noise_basis_id,
    )
    return ETDRKMethod(method.order).prepare(drift, coordinates=coordinates)


class OUForcedPeriodicState(StrictModule):
    """Periodic velocity and exact OU forcing continuation state."""

    velocity: Array
    forcing_state: SolenoidalOUForcingState


class PreparedOUForcedETDRKMethod(AbstractFixedStepMethod):
    """ETDRK with exact OU coefficient transitions at its stage abscissae."""

    base: PreparedETDRKMethod
    forcing: SolenoidalOUForcingPlan
    realization: OrnsteinUhlenbeckRealization
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: PreparedETDRKMethod,
        forcing: SolenoidalOUForcingPlan,
        realization: OrnsteinUhlenbeckRealization,
        /,
    ):
        if not isinstance(base, PreparedETDRKMethod):
            raise TypeError("base must be PreparedETDRKMethod.")
        if not isinstance(forcing, SolenoidalOUForcingPlan):
            raise TypeError("forcing must be SolenoidalOUForcingPlan.")
        if not isinstance(realization, OrnsteinUhlenbeckRealization):
            raise TypeError("realization must be OrnsteinUhlenbeckRealization.")
        if base.coordinates is None:
            raise ValueError("OU-forced periodic ETDRK requires Hermitian coordinates.")
        if forcing.discretization_id != base.coordinates.discretization.prepared_id:
            raise ValueError("OU forcing and ETDRK coordinates use another grid.")
        forcing._validate_realization(realization)
        self.base = base
        self.forcing = forcing
        self.realization = realization
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-ou-forced-etdrk-method-v1",
                "base": base.method_id,
                "forcing": forcing.forcing_id,
                "realization": realization.realization_id,
                "stage_forcing": (
                    "start,end" if base.order == 2 else "start,half,half,end"
                ),
            }
        )

    @property
    def coordinates(self) -> HermitianSpectralCoordinates:
        coordinates = self.base.coordinates
        if coordinates is None:
            raise RuntimeError("OU-forced ETDRK lost its Hermitian coordinates.")
        return coordinates

    @property
    def drift(self) -> SemilinearDrift:
        return self.base.drift

    @property
    def order(self) -> Literal[2, 4]:
        return self.base.order

    def initial_state(
        self,
        velocity: ArrayLike,
        time: ArrayLike,
        /,
        *,
        coefficients: ArrayLike | None = None,
    ) -> OUForcedPeriodicState:
        value = self.coordinates.validate_state(velocity)
        forcing_state = self.forcing.initialize(
            time,
            self.realization,
            coefficients=coefficients,
        )
        return OUForcedPeriodicState(value, forcing_state)

    def step(
        self,
        step_index: Array,
        time: Array,
        state: OUForcedPeriodicState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        if not isinstance(state, OUForcedPeriodicState):
            raise TypeError("OU-forced ETDRK state has the wrong type.")
        value = self.base._validate_state(state.velocity)
        step = jnp.asarray(step_size, dtype=value.real.dtype).reshape(())
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step > 0.0)),
            "OU-forced ETDRK step size must be finite and positive.",
        )
        scheduled_start = jnp.asarray(time, dtype=step.dtype).reshape(())
        start = state.forcing_state.time
        schedule_defect = jnp.abs(scheduled_start - start)
        schedule_tolerance = (
            32.0
            * jnp.finfo(step.dtype).eps
            * jnp.maximum(1.0, jnp.maximum(jnp.abs(scheduled_start), jnp.abs(start)))
        )
        schedule_valid = jnp.isfinite(scheduled_start) & (
            schedule_defect <= schedule_tolerance
        )
        advance = self.forcing.advance(
            state.forcing_state,
            start,
            start + step,
            self.realization,
        )
        stages = (
            (advance.start_forcing, advance.end_forcing)
            if self.order == 2
            else (
                advance.start_forcing,
                advance.half_forcing,
                advance.half_forcing,
                advance.end_forcing,
            )
        )
        _, incoming_valid, incoming_defect = self.base._boundary_evidence(value)
        candidate_velocity = _etdrk_update(
            self.order,
            self.drift,
            self.base.diagonal,
            start,
            value,
            step,
            args,
            stages,
        )
        projected, candidate_valid, candidate_defect = self.base._boundary_evidence(
            candidate_velocity
        )
        successful = (
            incoming_valid & candidate_valid & advance.successful & schedule_valid
        )
        accepted_velocity = jnp.where(successful, projected, value)
        accepted_forcing = SolenoidalOUForcingState(
            time=jnp.where(successful, advance.state.time, state.forcing_state.time),
            coefficients=jnp.where(
                successful,
                advance.state.coefficients,
                state.forcing_state.coefficients,
            ),
            basis_id=state.forcing_state.basis_id,
            realization_id=state.forcing_state.realization_id,
            forcing_id=state.forcing_state.forcing_id,
        )
        candidate_state = OUForcedPeriodicState(candidate_velocity, advance.state)
        accepted_state = OUForcedPeriodicState(accepted_velocity, accepted_forcing)
        correction = projected - candidate_velocity
        correction_norm = jnp.sqrt(jnp.sum(jnp.real(correction * jnp.conj(correction))))
        finite_transition = (
            jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.all(jnp.isfinite(advance.state.coefficients))
            & advance.finite
            & jnp.isfinite(schedule_defect)
        )
        residual = jnp.where(
            finite_transition,
            jnp.maximum(
                jnp.maximum(incoming_defect, candidate_defect),
                schedule_defect,
            ),
            jnp.asarray(jnp.inf, dtype=value.real.dtype),
        )
        return FixedStepResult(
            candidate_state=candidate_state,
            accepted_state=accepted_state,
            successful=successful,
            residual=residual,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            work=jnp.asarray(self.order, dtype=jnp.int32),
            transform_applied=successful & (correction_norm > 0.0),
            transform_correction_norm=jnp.where(
                successful,
                correction_norm,
                jnp.zeros((), dtype=correction_norm.dtype),
            ),
        )


def prepare_ou_forced_periodic_method(
    method: PreparedETDRKMethod,
    forcing: SolenoidalOUForcingPlan,
    realization: OrnsteinUhlenbeckRealization,
    /,
) -> PreparedOUForcedETDRKMethod:
    return PreparedOUForcedETDRKMethod(method, forcing, realization)


class MACConstantPressureGradientForcing(StrictModule, NonTrainableState):
    """One compiler-ready constant physical pressure-gradient acceleration."""

    operators: PreparedMACOperators
    pressure_gradient: Array
    density: float = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        pressure_gradient: ArrayLike,
        /,
        *,
        density: float,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        gradient = np.asarray(pressure_gradient, dtype=float)
        density_ = float(density)
        dimension = len(operators.discretization.cell_shape)
        if (
            gradient.shape != (dimension,)
            or np.any(~np.isfinite(gradient))
            or not np.any(gradient != 0.0)
            or not math.isfinite(density_)
            or density_ <= 0.0
        ):
            raise ValueError("Constant MAC pressure-gradient forcing is invalid.")
        self.operators = operators
        self.pressure_gradient = jnp.asarray(
            gradient, dtype=operators.pressure_space.dtype
        )
        self.density = density_
        self.forcing_id = canonical_fingerprint(
            {
                "kind": "mac-constant-pressure-gradient-forcing-v1",
                "operators": operators.prepared_id,
                "pressure_gradient": gradient.tolist(),
                "density": density_,
                "control": "none",
            }
        )

    def __call__(
        self,
        time: Array,
        velocity: FaceVelocity,
        args: Any,
    ) -> FaceVelocity:
        del time, args
        values = self.operators.validate_velocity(velocity)
        return tuple(
            jnp.full_like(value, -self.pressure_gradient[axis] / self.density)
            for axis, value in enumerate(values)
        )


class _PeriodicStatisticsEvaluator(StrictModule):
    method: PreparedETDRKMethod | PreparedOUForcedETDRKMethod
    statistics: PeriodicModalTurbulenceStatisticsPlan
    forcing: ConstantPowerFourierForcingPlan | None
    ou_forcing: SolenoidalOUForcingPlan | None
    evaluator_id: str = eqx.field(static=True)
    value_size: int = eqx.field(static=True)

    def __init__(
        self,
        method: PreparedETDRKMethod | PreparedOUForcedETDRKMethod,
        statistics: PeriodicModalTurbulenceStatisticsPlan,
        forcing: ConstantPowerFourierForcingPlan | None,
        ou_forcing: SolenoidalOUForcingPlan | None,
        /,
    ):
        self.method = method
        self.statistics = statistics
        self.forcing = forcing
        self.ou_forcing = ou_forcing
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "periodic-production-statistics-observer-v1",
                "method": method.method_id,
                "statistics": statistics.plan_id,
                "forcing": None if forcing is None else forcing.forcing_id,
                "ou_forcing": (None if ou_forcing is None else ou_forcing.forcing_id),
            }
        )
        self.value_size = 4 * statistics.geometry.bin_count + 21

    def snapshot(
        self,
        time: ArrayLike,
        state: ArrayLike | OUForcedPeriodicState,
        args: Any,
        /,
    ) -> PeriodicModalTurbulenceStatistics:
        if isinstance(state, OUForcedPeriodicState):
            if self.ou_forcing is None:
                raise TypeError("OU periodic state requires an OU statistics binding.")
            value = state.velocity
            force = self.ou_forcing.evaluate(state.forcing_state)
        else:
            if self.ou_forcing is not None:
                raise TypeError("OU statistics require OUForcedPeriodicState.")
            value = jnp.asarray(state)
            force = None if self.forcing is None else self.forcing.evaluate(value).forcing
        nonlinear = self.method.drift.nonlinear(jnp.asarray(time), value, args)
        if self.forcing is not None:
            nonlinear = nonlinear - force
        return self.statistics.evaluate(
            value,
            nonlinear_rate=nonlinear,
            forcing=force,
        )

    def __call__(
        self,
        time: Array,
        state: Array | OUForcedPeriodicState,
        args: Any,
    ) -> Array:
        result = self.snapshot(time, state, args)
        shells = (
            result.energy_shells.integral,
            result.dissipation_shells.integral,
            result.nonlinear_transfer_shells.integral,
            result.forcing_injection_shells.integral,
        )
        scalars = jnp.stack(
            (
                result.kinetic_energy,
                result.mean_kinetic_energy,
                result.dissipation,
                result.mean_dissipation,
                result.nonlinear_energy_rate,
                result.mean_nonlinear_energy_rate,
                result.forcing_power,
                result.mean_forcing_power,
                result.enstrophy,
                result.mean_enstrophy,
                result.helicity,
                result.mean_helicity,
                result.taylor_microscale,
                result.kolmogorov_scale,
                result.kmax_kolmogorov,
                result.integral_scale,
                result.energy_tail_fraction,
                result.dissipation_tail_fraction,
                result.divergence_norm,
                result.velocity_reality_defect,
                result.successful.astype(result.kinetic_energy.dtype),
            )
        )
        return jnp.concatenate((*shells, scalars))


class _ChannelStatisticsEvaluator(StrictModule):
    statistics: SpectralChannelStatisticsPlan
    evaluator_id: str = eqx.field(static=True)
    value_size: int = eqx.field(static=True)

    def __init__(self, statistics: SpectralChannelStatisticsPlan, /):
        self.statistics = statistics
        count = int(statistics.wall_normal_coordinates.size)
        self.value_size = 15 * count + 9
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "spectral-channel-production-statistics-observer-v1",
                "statistics": statistics.plan_id,
            }
        )

    def snapshot(self, state: ChannelSBDF2State, /) -> SpectralChannelStatistics:
        if not isinstance(state, ChannelSBDF2State):
            raise TypeError("Channel production statistics require ChannelSBDF2State.")
        return self.statistics.evaluate(state.current_velocity)

    def __call__(self, time: Array, state: ChannelSBDF2State, args: Any) -> Array:
        del time, args
        result = self.snapshot(state)
        profiles = (
            result.mean_streamwise_velocity,
            result.mean_wall_normal_velocity,
            result.mean_spanwise_velocity,
            result.raw_uu,
            result.raw_vv,
            result.raw_ww,
            result.raw_uv,
            result.raw_uw,
            result.raw_vw,
            result.reynolds_uu,
            result.reynolds_vv,
            result.reynolds_ww,
            result.reynolds_uv,
            result.reynolds_uw,
            result.reynolds_vw,
        )
        scalars = jnp.stack(
            (
                result.lower_wall_shear,
                result.upper_wall_shear,
                result.bulk_velocity,
                result.lower_friction_velocity,
                result.upper_friction_velocity,
                result.lower_friction_reynolds,
                result.upper_friction_reynolds,
                result.imaginary_leakage,
                result.successful.astype(result.bulk_velocity.dtype),
            )
        )
        return jnp.concatenate((*profiles, scalars))


class _MACStatisticsEvaluator(StrictModule):
    dynamics: CompiledMACIncompressibleDynamics
    statistics: MACPlaneWallStatisticsPlan
    pressure_gradient: MACConstantPressureGradientForcing | None
    evaluator_id: str = eqx.field(static=True)
    value_size: int = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        statistics: MACPlaneWallStatisticsPlan,
        pressure_gradient: MACConstantPressureGradientForcing | None,
        /,
    ):
        self.dynamics = dynamics
        self.statistics = statistics
        self.pressure_gradient = pressure_gradient
        dimension = len(statistics.operators.discretization.cell_shape)
        count = int(statistics.wall_normal_coordinates.size)
        self.value_size = count * dimension + 2 * count * dimension**2 + 3 * dimension + 8
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "structured-mac-production-statistics-observer-v1",
                "dynamics": dynamics.compilation_id,
                "statistics": statistics.plan_id,
                "pressure_gradient": None
                if pressure_gradient is None
                else pressure_gradient.forcing_id,
            }
        )

    def snapshot(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any,
        /,
    ) -> MACPlaneWallStatistics:
        velocity = self.dynamics.physical_state(time, state, args)
        forcing = (
            None
            if self.pressure_gradient is None
            else self.pressure_gradient(jnp.asarray(time), velocity, args)
        )
        return self.statistics.evaluate(velocity, forcing=forcing)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        result = self.snapshot(time, state, args)
        profiles = (
            result.mean_velocity.reshape((-1,)),
            result.raw_second_moment.reshape((-1,)),
            result.reynolds_stress.reshape((-1,)),
            result.lower_wall_shear.reshape((-1,)),
            result.upper_wall_shear.reshape((-1,)),
            result.bulk_velocity.reshape((-1,)),
        )
        scalars = jnp.stack(
            (
                result.lower_wall_normal_velocity,
                result.upper_wall_normal_velocity,
                result.kinetic_energy,
                result.mean_kinetic_energy,
                result.forcing_power,
                result.mean_forcing_power,
                result.divergence_norm,
                result.successful.astype(result.kinetic_energy.dtype),
            )
        )
        return jnp.concatenate((*profiles, scalars))


class _PreparedProductionRoute:
    def _bind_runtime(
        self,
        plan: Any,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any,
        args_id: str | None,
        publisher: ByteBoundedAsyncPublisher | None,
    ) -> None:
        store = DurableCheckpointStore(
            checkpoint_root,
            plan.manifest,
            CheckpointGenerationPolicy(plan.checkpoint_retention),
            encoding_plan=plan.checkpoint_encoding,
        )
        runtime = PreparedProductionRun(
            plan.manifest,
            plan.runtime_plan,
            store,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )
        self.plan = plan
        self.manifest = plan.manifest
        self.runtime_plan = plan.runtime_plan
        self.checkpoint_store = store
        self.runtime = runtime
        self.prepared_id = canonical_fingerprint(
            {
                "kind": self._prepared_kind,
                "plan": plan.plan_id,
                "runtime": runtime.run_id,
            }
        )

    def run(self, state: ProductionRunState, /) -> ProductionRunResult:
        return self.runtime.run(state)

    def resume(self, template: ProductionRunState, /) -> ProductionRunState:
        return self.runtime.resume(template)

    def step(self, state: ProductionRunState, /):
        return self.runtime.step(state)

    def checkpoint(self, state: ProductionRunState, /) -> ProductionRunState:
        return self.runtime.checkpoint(state)


class PeriodicSpectralProductionPlan(StrictModule, NonTrainableState):
    """Prepared-object assembly for periodic spectral turbulence production."""

    method: PreparedETDRKMethod | PreparedOUForcedETDRKMethod
    statistics: PeriodicModalTurbulenceStatisticsPlan
    constant_power_forcing: ConstantPowerFourierForcingPlan | None
    ou_forcing: SolenoidalOUForcingPlan | None
    ou_realization: OrnsteinUhlenbeckRealization | None
    statistics_evaluator: _PeriodicStatisticsEvaluator
    manifest: ProductionCaseManifest
    runtime_plan: ProductionRunPlan
    checkpoint_encoding: RuntimeCheckpointEncodingPlan
    output_schedule: ExactTimeSchedule | None
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    source_problem_id: str = eqx.field(static=True)
    constant_power_wiring: ConstantPowerWiring = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    checkpoint_retention: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: PreparedETDRKMethod,
        statistics: PeriodicModalTurbulenceStatisticsPlan,
        /,
        *,
        problem_id: str,
        start_time: float,
        end_time: float,
        step_size: float,
        checkpoint_interval: int,
        constant_power_forcing: ConstantPowerFourierForcingPlan | None = None,
        constant_power_wiring: ConstantPowerWiring = "compiled",
        ou_forcing: SolenoidalOUForcingPlan | None = None,
        ou_realization: OrnsteinUhlenbeckRealization | None = None,
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
        if not isinstance(method, PreparedETDRKMethod):
            raise TypeError("method must be PreparedETDRKMethod.")
        if not isinstance(statistics, PeriodicModalTurbulenceStatisticsPlan):
            raise TypeError("statistics must be PeriodicModalTurbulenceStatisticsPlan.")
        forcing = constant_power_forcing
        if forcing is not None and not isinstance(
            forcing, ConstantPowerFourierForcingPlan
        ):
            raise TypeError("constant_power_forcing has the wrong type.")
        if (ou_forcing is None) != (ou_realization is None):
            raise ValueError("ou_forcing and ou_realization must be supplied together.")
        if forcing is not None and ou_forcing is not None:
            raise ValueError("Constant-power and OU forcing are mutually exclusive.")
        if ou_forcing is not None and not isinstance(ou_forcing, SolenoidalOUForcingPlan):
            raise TypeError("ou_forcing has the wrong type.")
        if ou_realization is not None and not isinstance(
            ou_realization, OrnsteinUhlenbeckRealization
        ):
            raise TypeError("ou_realization has the wrong type.")
        if constant_power_wiring not in ("compiled", "adapter"):
            raise ValueError("constant_power_wiring must be 'compiled' or 'adapter'.")
        coordinates = method.coordinates
        if coordinates is None:
            raise ValueError(
                "Periodic production requires full-complex Hermitian ETDRK state."
            )
        discretization_id = coordinates.discretization.prepared_id
        if statistics.discretization_id != discretization_id:
            raise ValueError(
                "Periodic statistics and ETDRK coordinates use another grid."
            )
        if forcing is not None and forcing.discretization_id != discretization_id:
            raise ValueError("Periodic forcing and ETDRK coordinates use another grid.")
        if ou_forcing is not None and ou_forcing.discretization_id != discretization_id:
            raise ValueError("OU forcing and ETDRK coordinates use another grid.")
        selected_method: PreparedETDRKMethod | PreparedOUForcedETDRKMethod = method
        if ou_forcing is not None and ou_realization is not None:
            selected_method = prepare_ou_forced_periodic_method(
                method, ou_forcing, ou_realization
            )
        elif forcing is not None and constant_power_wiring == "adapter":
            selected_method = prepare_constant_power_periodic_method(method, forcing)
        elif forcing is not None:
            compiled_forcing_id = _compiled_periodic_forcing_id(method)
            if compiled_forcing_id != forcing.forcing_id:
                raise ValueError(
                    "The compiled periodic drift does not bind the declared "
                    "constant-power forcing."
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
        evaluator = _PeriodicStatisticsEvaluator(
            selected_method, statistics, forcing, ou_forcing
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
            selected_method,
            retry,
            step_size=step,
            end_time=end,
            maximum_steps=steps,
            checkpoint_interval=interval,
            segment_steps=segment,
            output_schedule=output,
            moments=(moment,),
            trigger_bindings=bindings,
        )
        encoding = RuntimeCheckpointEncodingPlan(
            (RuntimeCheckpointLeafBinding(0, coordinates, coordinates.evidence),)
        )
        source_problem = _required_identifier(problem_id, "problem_id")
        case_problem_id = canonical_fingerprint(
            {
                "kind": "periodic-spectral-production-case-v1",
                "problem": source_problem,
                "method": selected_method.method_id,
                "forcing": None if forcing is None else forcing.forcing_id,
                "forcing_wiring": constant_power_wiring,
                "ou_forcing": (None if ou_forcing is None else ou_forcing.forcing_id),
                "ou_realization": (
                    None if ou_realization is None else ou_realization.realization_id
                ),
                "statistics": statistics.plan_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "start_time": start,
                "checkpoint_retention": retention,
            }
        )
        discretization = coordinates.discretization
        manifest = ProductionCaseManifest(
            problem_id=case_problem_id,
            method_id=selected_method.method_id,
            precision_id=discretization.plan.precision.policy_id,
            topology_id=discretization.prepared_id,
            geometry_layout_id=coordinates.coordinate_id,
            dtype=coordinates.evidence.source_dtype,
        )
        self.method = selected_method
        self.statistics = statistics
        self.constant_power_forcing = forcing
        self.ou_forcing = ou_forcing
        self.ou_realization = ou_realization
        self.statistics_evaluator = evaluator
        self.manifest = manifest
        self.runtime_plan = runtime_plan
        self.checkpoint_encoding = encoding
        self.output_schedule = output
        self.trigger_bindings = bindings
        self.source_problem_id = source_problem
        self.constant_power_wiring = constant_power_wiring
        self.start_time = start
        self.checkpoint_retention = retention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-spectral-production-plan-v1",
                "manifest": manifest.manifest_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "start_time": start,
                "checkpoint_retention": retention,
            }
        )

    def prepare(
        self,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ) -> PreparedPeriodicSpectralProduction:
        return PreparedPeriodicSpectralProduction(
            self,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )


class PreparedPeriodicSpectralProduction(_PreparedProductionRoute):
    _prepared_kind = "prepared-periodic-spectral-production-v1"

    def __init__(
        self,
        plan: PeriodicSpectralProductionPlan,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if not isinstance(plan, PeriodicSpectralProductionPlan):
            raise TypeError("plan must be PeriodicSpectralProductionPlan.")
        self._bind_runtime(
            plan,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )

    def initialize(
        self,
        modal_velocity: ArrayLike,
        /,
        *,
        controller_state: Any = (),
        rng_state: Any = (),
        ou_coefficients: ArrayLike | None = None,
    ) -> ProductionRunState:
        coordinates = self.plan.method.coordinates
        velocity = coordinates.validate_state(modal_velocity)
        if isinstance(self.plan.method, PreparedOUForcedETDRKMethod):
            accepted_state: Array | OUForcedPeriodicState = (
                self.plan.method.initial_state(
                    velocity,
                    self.plan.start_time,
                    coefficients=ou_coefficients,
                )
            )
        else:
            if ou_coefficients is not None:
                raise ValueError(
                    "ou_coefficients require an OU-forced production method."
                )
            accepted_state = velocity
        return self.runtime.initial_state(
            accepted_state,
            time=self.plan.start_time,
            controller_state=controller_state,
            rng_state=rng_state,
        )

    def statistics_snapshot(
        self,
        time: ArrayLike,
        state: ArrayLike | OUForcedPeriodicState,
        /,
    ) -> PeriodicModalTurbulenceStatistics:
        return self.plan.statistics_evaluator.snapshot(
            time,
            state,
            self.runtime.args,
        )


class SpectralChannelProductionPlan(StrictModule, NonTrainableState):
    """Prepared-object assembly for fixed-lattice spectral channel production."""

    method: PreparedChannelSBDF2Method
    velocity_coordinates: HermitianSpectralCoordinates
    pressure_coordinates: HermitianSpectralCoordinates
    statistics: SpectralChannelStatisticsPlan
    statistics_evaluator: _ChannelStatisticsEvaluator
    manifest: ProductionCaseManifest
    runtime_plan: ProductionRunPlan
    checkpoint_encoding: RuntimeCheckpointEncodingPlan
    output_schedule: ExactTimeSchedule | None
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    source_problem_id: str = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    checkpoint_retention: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: PreparedChannelSBDF2Method,
        velocity_coordinates: HermitianSpectralCoordinates,
        pressure_coordinates: HermitianSpectralCoordinates,
        statistics: SpectralChannelStatisticsPlan,
        /,
        *,
        problem_id: str,
        start_time: float,
        end_time: float,
        checkpoint_interval: int,
        output_times: ArrayLike | None = None,
        output_tolerance: float = 1.0e-12,
        maximum_steps: int | None = None,
        segment_steps: int = 32,
        checkpoint_retention: int = 3,
        trigger_bindings: Sequence[ProductionTriggerBinding] = (),
        statistics_weighting: StatisticsWeighting = "time",
        statistics_window_start: float | None = None,
        statistics_window_end: float | None = None,
        statistics_batch_duration: float | None = None,
        maximum_statistics_batches: int = 0,
    ):
        if not isinstance(method, PreparedChannelSBDF2Method):
            raise TypeError("method must be PreparedChannelSBDF2Method.")
        if not isinstance(velocity_coordinates, HermitianSpectralCoordinates):
            raise TypeError("velocity_coordinates must be HermitianSpectralCoordinates.")
        if not isinstance(pressure_coordinates, HermitianSpectralCoordinates):
            raise TypeError("pressure_coordinates must be HermitianSpectralCoordinates.")
        if not isinstance(statistics, SpectralChannelStatisticsPlan):
            raise TypeError("statistics must be SpectralChannelStatisticsPlan.")
        discretization = method.dynamics.discretization
        if (
            velocity_coordinates.discretization.prepared_id != discretization.prepared_id
            or pressure_coordinates.discretization.prepared_id
            != discretization.prepared_id
            or statistics.discretization_id != discretization.prepared_id
            or velocity_coordinates.state_shape != method.dynamics.state_shape
            or pressure_coordinates.state_shape != discretization.modal_shape
        ):
            raise ValueError(
                "Channel method, coordinates, and statistics are incompatible."
            )
        step_size = method.required_step_size
        start, end, step, steps, interval, segment, retention = _runtime_values(
            start_time,
            end_time,
            step_size,
            maximum_steps,
            checkpoint_interval,
            segment_steps,
            checkpoint_retention,
        )
        end_index = _lattice_index(end, start, step, "end_time")
        if steps < end_index:
            raise ValueError(
                "maximum_steps cannot reach the channel end_time lattice point."
            )
        output = _output_schedule(output_times, start, end, output_tolerance)
        if output is not None:
            for target in np.asarray(output.targets):
                _lattice_index(float(target), start, step, "An output target")
        window_start, window_end = _statistics_window(
            start,
            end,
            statistics_window_start,
            statistics_window_end,
        )
        _lattice_index(window_start, start, step, "statistics_window_start")
        _lattice_index(window_end, start, step, "statistics_window_end")
        evaluator = _ChannelStatisticsEvaluator(statistics)
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
        bindings = tuple(trigger_bindings)
        runtime_plan = ProductionRunPlan(
            method,
            RobustRetryPolicy(maximum_retries=0),
            step_size=step,
            end_time=end,
            maximum_steps=steps,
            checkpoint_interval=interval,
            segment_steps=segment,
            output_schedule=output,
            moments=(moment,),
            trigger_bindings=bindings,
        )
        encoding = RuntimeCheckpointEncodingPlan(
            (
                RuntimeCheckpointLeafBinding(
                    0, velocity_coordinates, velocity_coordinates.evidence
                ),
                RuntimeCheckpointLeafBinding(
                    1, velocity_coordinates, velocity_coordinates.evidence
                ),
                RuntimeCheckpointLeafBinding(
                    2, velocity_coordinates, velocity_coordinates.evidence
                ),
                RuntimeCheckpointLeafBinding(
                    3, velocity_coordinates, velocity_coordinates.evidence
                ),
                RuntimeCheckpointLeafBinding(
                    4, pressure_coordinates, pressure_coordinates.evidence
                ),
            )
        )
        source_problem = _required_identifier(problem_id, "problem_id")
        case_problem_id = canonical_fingerprint(
            {
                "kind": "spectral-channel-production-case-v1",
                "problem": source_problem,
                "dynamics": method.dynamics.compilation_id,
                "statistics": statistics.plan_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "step_lattice": {"origin": start, "step": step, "end_index": end_index},
                "checkpoint_retention": retention,
            }
        )
        manifest = ProductionCaseManifest(
            problem_id=case_problem_id,
            method_id=method.method_id,
            precision_id=discretization.plan.precision.policy_id,
            topology_id=discretization.prepared_id,
            geometry_layout_id=velocity_coordinates.coordinate_id,
            dtype=velocity_coordinates.evidence.source_dtype,
        )
        self.method = method
        self.velocity_coordinates = velocity_coordinates
        self.pressure_coordinates = pressure_coordinates
        self.statistics = statistics
        self.statistics_evaluator = evaluator
        self.manifest = manifest
        self.runtime_plan = runtime_plan
        self.checkpoint_encoding = encoding
        self.output_schedule = output
        self.trigger_bindings = bindings
        self.source_problem_id = source_problem
        self.start_time = start
        self.checkpoint_retention = retention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-channel-production-plan-v1",
                "manifest": manifest.manifest_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "start_time": start,
                "checkpoint_retention": retention,
            }
        )

    def prepare(
        self,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ) -> PreparedSpectralChannelProduction:
        return PreparedSpectralChannelProduction(
            self,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )


class PreparedSpectralChannelProduction(_PreparedProductionRoute):
    _prepared_kind = "prepared-spectral-channel-production-v1"

    def __init__(
        self,
        plan: SpectralChannelProductionPlan,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if not isinstance(plan, SpectralChannelProductionPlan):
            raise TypeError("plan must be SpectralChannelProductionPlan.")
        self._bind_runtime(
            plan,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )

    def initialize(
        self,
        modal_velocity: ArrayLike,
        /,
        *,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        continuation = self.plan.method.initialize(
            modal_velocity,
            self.plan.start_time,
            self.runtime.args,
        )
        return self.runtime.initial_state(
            continuation,
            time=self.plan.start_time,
            controller_state=controller_state,
            rng_state=rng_state,
        )

    def statistics_snapshot(
        self,
        continuation: ChannelSBDF2State,
        /,
    ) -> SpectralChannelStatistics:
        return self.plan.statistics_evaluator.snapshot(continuation)


class StructuredMACProductionPlan(StrictModule, NonTrainableState):
    """Prepared-object assembly for fixed-step structured-MAC production."""

    method: AbstractFixedStepMethod
    dynamics: CompiledMACIncompressibleDynamics
    statistics: MACPlaneWallStatisticsPlan
    constant_pressure_gradient: MACConstantPressureGradientForcing | None
    statistics_evaluator: _MACStatisticsEvaluator
    manifest: ProductionCaseManifest
    runtime_plan: ProductionRunPlan
    checkpoint_encoding: RuntimeCheckpointEncodingPlan
    output_schedule: ExactTimeSchedule | None
    trigger_bindings: tuple[ProductionTriggerBinding, ...]
    start_time: float = eqx.field(static=True)
    checkpoint_retention: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        dynamics: CompiledMACIncompressibleDynamics,
        statistics: MACPlaneWallStatisticsPlan,
        /,
        *,
        start_time: float,
        end_time: float,
        step_size: float,
        checkpoint_interval: int,
        constant_pressure_gradient: MACConstantPressureGradientForcing | None = None,
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
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be AbstractFixedStepMethod.")
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(statistics, MACPlaneWallStatisticsPlan):
            raise TypeError("statistics must be MACPlaneWallStatisticsPlan.")
        pressure_gradient = constant_pressure_gradient
        if pressure_gradient is not None and not isinstance(
            pressure_gradient, MACConstantPressureGradientForcing
        ):
            raise TypeError("constant_pressure_gradient has the wrong type.")
        operators = dynamics.momentum.operators
        if statistics.operators_id != operators.prepared_id:
            raise ValueError("MAC dynamics and statistics use another operator plan.")
        if pressure_gradient is not None:
            if pressure_gradient.operators.prepared_id != operators.prepared_id:
                raise ValueError("MAC pressure-gradient forcing uses another grid.")
            if dynamics.problem.forcing_id != pressure_gradient.forcing_id:
                raise ValueError(
                    "Compiled MAC dynamics do not bind the declared pressure gradient."
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
        evaluator = _MACStatisticsEvaluator(dynamics, statistics, pressure_gradient)
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
            method,
            retry,
            step_size=step,
            end_time=end,
            maximum_steps=steps,
            checkpoint_interval=interval,
            segment_steps=segment,
            output_schedule=output,
            moments=(moment,),
            trigger_bindings=bindings,
        )
        encoding = RuntimeCheckpointEncodingPlan()
        forcing_id = (
            dynamics.problem.forcing_id
            if pressure_gradient is None
            else pressure_gradient.forcing_id
        )
        case_problem_id = canonical_fingerprint(
            {
                "kind": "structured-mac-production-case-v1",
                "problem": dynamics.problem.problem_id,
                "dynamics": dynamics.compilation_id,
                "forcing": forcing_id,
                "forcing_control": "constant-pressure-gradient"
                if pressure_gradient is not None
                else "compiled-or-none",
                "statistics": statistics.plan_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "start_time": start,
                "checkpoint_retention": retention,
            }
        )
        discretization = operators.discretization
        manifest = ProductionCaseManifest(
            problem_id=case_problem_id,
            method_id=method.method_id,
            precision_id=dynamics.momentum.precision.policy_id,
            topology_id=operators.prepared_id,
            geometry_layout_id=discretization.grid.prepared_id,
            dtype=str(jnp.dtype(operators.pressure_space.dtype)),
        )
        self.method = method
        self.dynamics = dynamics
        self.statistics = statistics
        self.constant_pressure_gradient = pressure_gradient
        self.statistics_evaluator = evaluator
        self.manifest = manifest
        self.runtime_plan = runtime_plan
        self.checkpoint_encoding = encoding
        self.output_schedule = output
        self.trigger_bindings = bindings
        self.start_time = start
        self.checkpoint_retention = retention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-mac-production-plan-v1",
                "manifest": manifest.manifest_id,
                "runtime": runtime_plan.plan_id,
                "checkpoint_encoding": encoding.encoding_id,
                "start_time": start,
                "checkpoint_retention": retention,
            }
        )

    def prepare(
        self,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ) -> PreparedStructuredMACProduction:
        return PreparedStructuredMACProduction(
            self,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )


class PreparedStructuredMACProduction(_PreparedProductionRoute):
    _prepared_kind = "prepared-structured-mac-production-v1"

    def __init__(
        self,
        plan: StructuredMACProductionPlan,
        checkpoint_root: str | Path,
        /,
        *,
        args: Any = None,
        args_id: str | None = None,
        publisher: ByteBoundedAsyncPublisher | None = None,
    ):
        if not isinstance(plan, StructuredMACProductionPlan):
            raise TypeError("plan must be StructuredMACProductionPlan.")
        self._bind_runtime(
            plan,
            checkpoint_root,
            args=args,
            args_id=args_id,
            publisher=publisher,
        )

    def initialize(
        self,
        velocity: ArrayLike | FaceVelocity,
        /,
        *,
        controller_state: Any = (),
        rng_state: Any = (),
    ) -> ProductionRunState:
        if isinstance(velocity, (tuple, list)):
            state = self.plan.dynamics.project_state(
                tuple(velocity),
                time=self.plan.start_time,
                args=self.runtime.args,
            )
        else:
            state = self.plan.dynamics.validate_state(velocity)
        return self.runtime.initial_state(
            state,
            time=self.plan.start_time,
            controller_state=controller_state,
            rng_state=rng_state,
        )

    def statistics_snapshot(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
    ) -> MACPlaneWallStatistics:
        return self.plan.statistics_evaluator.snapshot(
            time,
            state,
            self.runtime.args,
        )


__all__ = [
    "MACConstantPressureGradientForcing",
    "OUForcedPeriodicState",
    "PeriodicSpectralProductionPlan",
    "PreparedPeriodicSpectralProduction",
    "PreparedOUForcedETDRKMethod",
    "PreparedSpectralChannelProduction",
    "PreparedStructuredMACProduction",
    "SpectralChannelProductionPlan",
    "StructuredMACProductionPlan",
    "prepare_ou_forced_periodic_method",
    "prepare_constant_power_periodic_method",
]
