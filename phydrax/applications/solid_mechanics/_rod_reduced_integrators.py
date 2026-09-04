#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import BlockSpace
from ...nonlinear import (
    AbstractNonlinearMethod,
    NewtonKrylov,
    NonlinearDiagnostics,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._rod_loads import RodLoadLedger
from ._rod_reduced_dynamics import (
    PreparedReducedRodDynamics,
    ReducedRodDynamicsEvaluation,
    ReducedRodMaterialControl,
    ReducedRodMaterialState,
    ReducedRodSolveEvidence,
)
from ._rod_reduction import ReducedRodState


_INTERNAL_POWER_SOURCE_IDS = frozenset(("elastic", "kelvin_voigt"))


ReducedRodIntegratorRoute: TypeAlias = Literal[
    "semi-implicit-velocity-euler", "implicit-midpoint"
]


class ReducedRodStepStatus(IntEnum):
    """Portable acceptance status for one reduced rod integration attempt."""

    SUCCESS = 0
    STEP_OUT_OF_BOUNDS = 1
    SOURCE_INVALID = 2
    MASS_SOLVE_FAILED = 3
    NONLINEAR_SOLVE_FAILED = 4
    MATERIAL_TRIAL_FAILED = 5
    CANDIDATE_INVALID = 6
    LEDGER_INVALID = 7


class ReducedRodIntegrationState(StrictModule):
    """Complete integrator-owned reduced phase and committed material history."""

    reduced_state: ReducedRodState
    material_state: ReducedRodMaterialState
    time: Array
    step_index: Array

    def __init__(
        self,
        reduced_state: ReducedRodState,
        material_state: ReducedRodMaterialState,
        time: ArrayLike = 0,
        step_index: ArrayLike = 0,
        /,
    ):
        if not isinstance(reduced_state, ReducedRodState):
            raise TypeError("reduced_state must be a ReducedRodState.")
        if not isinstance(material_state, ReducedRodMaterialState):
            raise TypeError("material_state must be a ReducedRodMaterialState.")
        time_ = jnp.asarray(time, dtype=reduced_state.values.dtype)
        step_ = jnp.asarray(step_index, dtype=jnp.int32)
        if time_.shape != () or step_.shape != ():
            raise ValueError(
                "Reduced rod integration time and step_index must be scalar."
            )
        self.reduced_state = reduced_state
        self.material_state = material_state
        self.time = time_
        self.step_index = step_


class ReducedRodEnergyWorkLedger(StrictModule):
    """Source/channel work, mechanical energy, and constitutive dissipation."""

    source_power_before: Array
    source_power_after: Array
    source_work: Array
    channel_power_before: Array
    channel_power_after: Array
    channel_work: Array
    total_power_before: Array
    total_power_after: Array
    external_work: Array
    kinetic_energy_before: Array
    kinetic_energy_after: Array
    stored_energy_before: Array
    stored_energy_after: Array
    mechanical_energy_before: Array
    mechanical_energy_after: Array
    viscous_dissipation: Array
    balance_residual: Array
    balance_scale: Array
    finite: Array
    dissipation_nonnegative: Array
    balanced: Array
    valid: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)


class ReducedRodStepEvidence(StrictModule):
    """Complete route, solver, material, and ledger evidence for one attempt."""

    source_evaluation: ReducedRodDynamicsEvaluation
    candidate_evaluation: ReducedRodDynamicsEvaluation
    linear_solve_evidence: ReducedRodSolveEvidence | NonlinearDiagnostics
    nonlinear_solve_evidence: NonlinearResult | None
    ledger: ReducedRodEnergyWorkLedger
    step_finite: Array
    step_within_bound: Array
    source_valid: Array
    linear_solve_successful: Array
    nonlinear_solve_successful: Array
    material_trial_valid: Array
    candidate_valid: Array
    finite: Array
    accepted: Array
    status: Array
    backend_status: Array
    route: ReducedRodIntegratorRoute = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class ReducedRodStepResult(StrictModule):
    """Candidate plus atomically accepted or completely rolled-back state."""

    previous_state: ReducedRodIntegrationState
    candidate_state: ReducedRodIntegrationState
    accepted_state: ReducedRodIntegrationState
    attempted: Array
    successful: Array
    status: Array
    backend_status: Array
    evidence: ReducedRodStepEvidence
    policy_id: str = eqx.field(static=True)


class ReducedRodSemiImplicitVelocityEuler(StrictModule):
    """One explicit, bounded velocity-first Euler route with no route fallback."""

    maximum_step_size: float = eqx.field(static=True)
    energy_balance_tolerance: float = eqx.field(static=True)
    route: ReducedRodIntegratorRoute = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_step_size: float,
        energy_balance_tolerance: float = 1.0e-6,
    ):
        maximum = float(maximum_step_size)
        tolerance = float(energy_balance_tolerance)
        if not isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_size must be positive and finite.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_balance_tolerance must be finite and nonnegative.")
        route: ReducedRodIntegratorRoute = "semi-implicit-velocity-euler"
        self.maximum_step_size = maximum
        self.energy_balance_tolerance = tolerance
        self.route = route
        self.policy_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-integrator-policy",
                "route": route,
                "maximum_step_size": maximum,
                "energy_balance_tolerance": tolerance,
            }
        )


class ReducedRodImplicitMidpoint(StrictModule):
    """One bounded Newton midpoint route with explicit nonlinear work limits."""

    nonlinear_method: AbstractNonlinearMethod
    nonlinear_termination: NonlinearTermination
    maximum_step_size: float = eqx.field(static=True)
    energy_balance_tolerance: float = eqx.field(static=True)
    route: ReducedRodIntegratorRoute = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_step_size: float,
        nonlinear_method: AbstractNonlinearMethod | None = None,
        nonlinear_termination: NonlinearTermination | None = None,
        energy_balance_tolerance: float = 1.0e-6,
    ):
        maximum = float(maximum_step_size)
        tolerance = float(energy_balance_tolerance)
        if not isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_size must be positive and finite.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_balance_tolerance must be finite and nonnegative.")
        method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
        termination = (
            NonlinearTermination(
                absolute_residual=1.0e-6,
                relative_residual=1.0e-6,
                absolute_step=1.0e-8,
                relative_step=1.0e-6,
                maximum_steps=24,
            )
            if nonlinear_termination is None
            else nonlinear_termination
        )
        if not isinstance(method, AbstractNonlinearMethod):
            raise TypeError(
                "nonlinear_method must be an AbstractNonlinearMethod or None."
            )
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("nonlinear_termination must be NonlinearTermination or None.")
        route: ReducedRodIntegratorRoute = "implicit-midpoint"
        self.nonlinear_method = method
        self.nonlinear_termination = termination
        self.maximum_step_size = maximum
        self.energy_balance_tolerance = tolerance
        self.route = route
        self.policy_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-integrator-policy",
                "route": route,
                "maximum_step_size": maximum,
                "energy_balance_tolerance": tolerance,
                "nonlinear_method": method.method_id,
                "maximum_nonlinear_steps": termination.maximum_steps,
                "maximum_evaluations": termination.maximum_evaluations,
                "maximum_linear_iterations": termination.maximum_linear_iterations,
            }
        )


ReducedRodIntegratorPolicy: TypeAlias = (
    ReducedRodSemiImplicitVelocityEuler | ReducedRodImplicitMidpoint
)


class _MidpointResidual(StrictModule):
    dynamics: PreparedReducedRodDynamics
    source: ReducedRodIntegrationState
    material_control: ReducedRodMaterialControl | None
    native_loads: RodLoadLedger | None
    step_size: Array

    def __call__(self, state: tuple[Array, Array], _arguments: Any, /):
        q0 = self.source.reduced_state.coefficients
        v0 = self.source.reduced_state.coefficient_velocities
        q1, v1 = state
        midpoint = ReducedRodState(0.5 * (q0 + q1), 0.5 * (v0 + v1))
        acceleration = (v1 - v0) / self.step_size
        inverse = self.dynamics.inverse_dynamics(
            midpoint,
            acceleration,
            source_state=self.source.reduced_state,
            material_state=self.source.material_state,
            material_control=self.material_control,
            time=self.source.time + 0.5 * self.step_size,
            step_size=0.5 * self.step_size,
            native_loads=self.native_loads,
        )
        kinematic = q1 - q0 - 0.5 * self.step_size * (v0 + v1)
        return (kinematic, inverse.residual), inverse


def initialize_reduced_rod_integration_state(
    dynamics: PreparedReducedRodDynamics,
    reduced_state: ReducedRodState | None = None,
    material_state: ReducedRodMaterialState | None = None,
    /,
    *,
    time: ArrayLike = 0,
    step_index: ArrayLike = 0,
) -> ReducedRodIntegrationState:
    """Create a complete initial state without trialing or committing material data."""
    if not isinstance(dynamics, PreparedReducedRodDynamics):
        raise TypeError("dynamics must be PreparedReducedRodDynamics.")
    state = (
        dynamics.reduction.initialize_state() if reduced_state is None else reduced_state
    )
    dynamics.reduction.validate_state(state)
    history = (
        dynamics.initialize_material_state() if material_state is None else material_state
    )
    return ReducedRodIntegrationState(state, history, time, step_index)


def _validate_source(
    dynamics: PreparedReducedRodDynamics,
    source: ReducedRodIntegrationState,
    /,
) -> None:
    if not isinstance(dynamics, PreparedReducedRodDynamics):
        raise TypeError("dynamics must be PreparedReducedRodDynamics.")
    if not isinstance(source, ReducedRodIntegrationState):
        raise TypeError("source must be ReducedRodIntegrationState.")
    dynamics.reduction.validate_state(source.reduced_state)


def _selected_state(
    accepted: Array,
    candidate: ReducedRodIntegrationState,
    source: ReducedRodIntegrationState,
    /,
) -> ReducedRodIntegrationState:
    candidate_history = candidate.material_state
    source_history = source.material_state
    material = ReducedRodMaterialState(
        jnp.where(
            accepted,
            candidate_history.stretch_shear_history,
            source_history.stretch_shear_history,
        ),
        jnp.where(
            accepted,
            candidate_history.bend_twist_history,
            source_history.bend_twist_history,
        ),
    )
    return ReducedRodIntegrationState(
        ReducedRodState(
            jnp.where(
                accepted,
                candidate.reduced_state.coefficients,
                source.reduced_state.coefficients,
            ),
            jnp.where(
                accepted,
                candidate.reduced_state.coefficient_velocities,
                source.reduced_state.coefficient_velocities,
            ),
        ),
        material,
        jnp.where(accepted, candidate.time, source.time),
        jnp.where(accepted, candidate.step_index, source.step_index),
    )


def _energy_work_ledger(
    source_evaluation: ReducedRodDynamicsEvaluation,
    candidate_evaluation: ReducedRodDynamicsEvaluation,
    step_size: Array,
    tolerance: float,
    /,
) -> ReducedRodEnergyWorkLedger:
    source_forces = source_evaluation.forces
    candidate_forces = candidate_evaluation.forces
    if source_forces.source_ids != candidate_forces.source_ids:
        raise ValueError("Rod load source IDs changed during one integration step.")
    if source_forces.channel_names != candidate_forces.channel_names:
        raise ValueError("Rod load channel names changed during one integration step.")
    source_power = 0.5 * (source_forces.source_power + candidate_forces.source_power)
    channel_power = 0.5 * (source_forces.channel_power + candidate_forces.channel_power)
    source_work = step_size * source_power
    channel_work = step_size * channel_power
    internal_source_ids = _INTERNAL_POWER_SOURCE_IDS
    external_source_mask = jnp.asarray(
        tuple(
            source_id not in internal_source_ids for source_id in source_forces.source_ids
        ),
        dtype=bool,
    )
    external_power_before = jnp.sum(
        jnp.where(
            external_source_mask,
            source_forces.source_power,
            jnp.zeros_like(source_forces.source_power),
        )
    )
    external_power_after = jnp.sum(
        jnp.where(
            external_source_mask,
            candidate_forces.source_power,
            jnp.zeros_like(candidate_forces.source_power),
        )
    )
    external_work = step_size * 0.5 * (external_power_before + external_power_after)
    before = source_evaluation.energy
    after = candidate_evaluation.energy
    dissipation = after.viscous_dissipation
    balance = (
        after.total_mechanical_energy
        - before.total_mechanical_energy
        + dissipation
        - external_work
    )
    scalar_values = jnp.stack(
        (
            before.kinetic_energy,
            after.kinetic_energy,
            before.stored_energy,
            after.stored_energy,
            before.total_mechanical_energy,
            after.total_mechanical_energy,
            dissipation,
            external_work,
            balance,
        )
    )
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=balance.dtype), jnp.max(jnp.abs(scalar_values))
    )
    finite = (
        jnp.all(jnp.isfinite(scalar_values))
        & jnp.all(jnp.isfinite(source_forces.source_power))
        & jnp.all(jnp.isfinite(candidate_forces.source_power))
        & jnp.all(jnp.isfinite(source_work))
        & jnp.all(jnp.isfinite(channel_work))
    )
    nonnegative = dissipation >= -64.0 * jnp.finfo(balance.dtype).eps * scale
    balanced = finite & (jnp.abs(balance) <= tolerance * scale)
    valid = finite & nonnegative & balanced
    return ReducedRodEnergyWorkLedger(
        source_forces.source_power,
        candidate_forces.source_power,
        source_work,
        source_forces.channel_power,
        candidate_forces.channel_power,
        channel_work,
        source_forces.total_power,
        candidate_forces.total_power,
        external_work,
        before.kinetic_energy,
        after.kinetic_energy,
        before.stored_energy,
        after.stored_energy,
        before.total_mechanical_energy,
        after.total_mechanical_energy,
        dissipation,
        balance,
        scale,
        finite,
        nonnegative,
        balanced,
        valid,
        source_forces.source_ids,
        source_forces.channel_names,
    )


def _candidate_state(
    source: ReducedRodIntegrationState,
    reduced_state: ReducedRodState,
    material_state: ReducedRodMaterialState,
    step_size: Array,
    /,
) -> ReducedRodIntegrationState:
    return ReducedRodIntegrationState(
        reduced_state,
        material_state,
        source.time + step_size,
        source.step_index + jnp.asarray(1, dtype=jnp.int32),
    )


def _status(
    *,
    step_valid: Array,
    source_valid: Array,
    linear_valid: Array,
    nonlinear_attempted: bool,
    nonlinear_valid: Array,
    material_valid: Array,
    candidate_valid: Array,
    ledger_valid: Array,
) -> Array:
    status = jnp.asarray(ReducedRodStepStatus.SUCCESS, dtype=jnp.int32)
    status = jnp.where(
        ~ledger_valid,
        jnp.asarray(ReducedRodStepStatus.LEDGER_INVALID, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        ~candidate_valid,
        jnp.asarray(ReducedRodStepStatus.CANDIDATE_INVALID, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        ~material_valid,
        jnp.asarray(ReducedRodStepStatus.MATERIAL_TRIAL_FAILED, dtype=jnp.int32),
        status,
    )
    if nonlinear_attempted:
        status = jnp.where(
            ~nonlinear_valid,
            jnp.asarray(ReducedRodStepStatus.NONLINEAR_SOLVE_FAILED, dtype=jnp.int32),
            status,
        )
    else:
        status = jnp.where(
            ~linear_valid,
            jnp.asarray(ReducedRodStepStatus.MASS_SOLVE_FAILED, dtype=jnp.int32),
            status,
        )
    status = jnp.where(
        ~source_valid,
        jnp.asarray(ReducedRodStepStatus.SOURCE_INVALID, dtype=jnp.int32),
        status,
    )
    return jnp.where(
        ~step_valid,
        jnp.asarray(ReducedRodStepStatus.STEP_OUT_OF_BOUNDS, dtype=jnp.int32),
        status,
    )


def _source_mechanics_valid(evaluation: ReducedRodDynamicsEvaluation, /) -> Array:
    """Validate the source law independently of the selected mass-solve route."""
    return (
        evaluation.bias.finite
        & evaluation.forces.valid
        & evaluation.energy.valid
        & evaluation.stretch_shear_material_result.evidence.valid
        & evaluation.bend_twist_material_result.evidence.valid
    )


def _semi_implicit_step(
    dynamics: PreparedReducedRodDynamics,
    policy: ReducedRodSemiImplicitVelocityEuler,
    source: ReducedRodIntegrationState,
    step_size: Array,
    material_control: ReducedRodMaterialControl | None,
    native_loads: RodLoadLedger | None,
    /,
) -> ReducedRodStepResult:
    forward = dynamics.forward_dynamics(
        source.reduced_state,
        material_state=source.material_state,
        material_control=material_control,
        time=source.time,
        step_size=step_size,
        native_loads=native_loads,
    )
    q0 = source.reduced_state.coefficients
    v1 = source.reduced_state.coefficient_velocities + step_size * forward.acceleration
    candidate_reduced = ReducedRodState(q0 + step_size * v1, v1)
    candidate_evaluation = dynamics.evaluate(
        candidate_reduced,
        source_state=source.reduced_state,
        material_state=source.material_state,
        material_control=material_control,
        time=source.time + step_size,
        step_size=step_size,
        native_loads=native_loads,
    )
    candidate = _candidate_state(
        source,
        candidate_reduced,
        candidate_evaluation.candidate_material_state,
        step_size,
    )
    ledger = _energy_work_ledger(
        forward.evaluation,
        candidate_evaluation,
        step_size,
        policy.energy_balance_tolerance,
    )
    step_finite = jnp.isfinite(step_size) & (step_size > 0.0)
    within = step_finite & (step_size <= policy.maximum_step_size)
    source_valid = _source_mechanics_valid(forward.evaluation)
    linear_valid = forward.solve_evidence.successful
    material_valid = (
        candidate_evaluation.stretch_shear_material_result.evidence.valid
        & candidate_evaluation.bend_twist_material_result.evidence.valid
    )
    candidate_valid = candidate_evaluation.valid
    finite = forward.finite & candidate_evaluation.finite & ledger.finite
    accepted = (
        within
        & source_valid
        & linear_valid
        & material_valid
        & candidate_valid
        & ledger.valid
        & finite
    )
    status = _status(
        step_valid=within,
        source_valid=source_valid,
        linear_valid=linear_valid,
        nonlinear_attempted=False,
        nonlinear_valid=jnp.asarray(True),
        material_valid=material_valid,
        candidate_valid=candidate_valid,
        ledger_valid=ledger.valid & finite,
    )
    backend = jnp.asarray(forward.solve_evidence.status, dtype=jnp.int32)
    evidence = ReducedRodStepEvidence(
        forward.evaluation,
        candidate_evaluation,
        forward.solve_evidence,
        None,
        ledger,
        step_finite,
        within,
        source_valid,
        linear_valid,
        jnp.asarray(True),
        material_valid,
        candidate_valid,
        finite,
        accepted,
        status,
        backend,
        policy.route,
        policy.policy_id,
    )
    return ReducedRodStepResult(
        source,
        candidate,
        _selected_state(accepted, candidate, source),
        jnp.asarray(True),
        accepted,
        status,
        backend,
        evidence,
        policy.policy_id,
    )


def _implicit_midpoint_step(
    dynamics: PreparedReducedRodDynamics,
    policy: ReducedRodImplicitMidpoint,
    source: ReducedRodIntegrationState,
    step_size: Array,
    material_control: ReducedRodMaterialControl | None,
    native_loads: RodLoadLedger | None,
    /,
) -> ReducedRodStepResult:
    source_evaluation = dynamics.evaluate(
        source.reduced_state,
        material_state=source.material_state,
        material_control=material_control,
        time=source.time,
        step_size=step_size,
        native_loads=native_loads,
    )
    q0 = source.reduced_state.coefficients
    v0 = source.reduced_state.coefficient_velocities
    initial = (q0 + step_size * v0, v0)
    state_space = BlockSpace(
        (
            dynamics.reduction.coefficient_space,
            dynamics.reduction.coefficient_space,
        ),
        names=("configuration", "velocity"),
    )
    residual_space = BlockSpace(
        (
            dynamics.reduction.coefficient_space,
            dynamics.reduction.reduced_effort_space,
        ),
        names=("kinematic", "dynamic"),
    )
    residual = _MidpointResidual(
        dynamics, source, material_control, native_loads, step_size
    )
    problem = NonlinearSystemProblem(
        residual,
        state_space=state_space,
        residual_space=residual_space,
        has_aux=True,
        validity=lambda _state, _value, inverse, _args: inverse.valid,
        problem_id=f"reduced-rod-implicit-midpoint:{dynamics.dynamics_id}:{policy.policy_id}",
    )
    nonlinear = policy.nonlinear_method.solve(
        problem,
        initial,
        termination=policy.nonlinear_termination,
    )
    q1, v1 = state_space.validate(nonlinear.state)
    candidate_reduced = ReducedRodState(q1, v1)
    candidate_evaluation = dynamics.evaluate(
        candidate_reduced,
        source_state=source.reduced_state,
        material_state=source.material_state,
        material_control=material_control,
        time=source.time + step_size,
        step_size=step_size,
        native_loads=native_loads,
    )
    candidate = _candidate_state(
        source,
        candidate_reduced,
        candidate_evaluation.candidate_material_state,
        step_size,
    )
    ledger = _energy_work_ledger(
        source_evaluation,
        candidate_evaluation,
        step_size,
        policy.energy_balance_tolerance,
    )
    step_finite = jnp.isfinite(step_size) & (step_size > 0.0)
    within = step_finite & (step_size <= policy.maximum_step_size)
    source_valid = source_evaluation.valid
    nonlinear_valid = nonlinear.successful
    material_valid = (
        candidate_evaluation.stretch_shear_material_result.evidence.valid
        & candidate_evaluation.bend_twist_material_result.evidence.valid
    )
    candidate_valid = candidate_evaluation.valid & nonlinear.auxiliary.valid
    finite = (
        source_evaluation.finite
        & candidate_evaluation.finite
        & nonlinear.auxiliary.finite
        & jnp.all(jnp.isfinite(q1))
        & jnp.all(jnp.isfinite(v1))
        & ledger.finite
    )
    accepted = (
        within
        & source_valid
        & nonlinear_valid
        & material_valid
        & candidate_valid
        & ledger.valid
        & finite
    )
    status = _status(
        step_valid=within,
        source_valid=source_valid,
        linear_valid=jnp.asarray(True),
        nonlinear_attempted=True,
        nonlinear_valid=nonlinear_valid,
        material_valid=material_valid,
        candidate_valid=candidate_valid,
        ledger_valid=ledger.valid & finite,
    )
    backend = jnp.asarray(nonlinear.status, dtype=jnp.int32)
    evidence = ReducedRodStepEvidence(
        source_evaluation,
        candidate_evaluation,
        nonlinear.diagnostics,
        nonlinear,
        ledger,
        step_finite,
        within,
        source_valid,
        nonlinear.diagnostics.final_linear_converged,
        nonlinear_valid,
        material_valid,
        candidate_valid,
        finite,
        accepted,
        status,
        backend,
        policy.route,
        policy.policy_id,
    )
    return ReducedRodStepResult(
        source,
        candidate,
        _selected_state(accepted, candidate, source),
        jnp.asarray(True),
        accepted,
        status,
        backend,
        evidence,
        policy.policy_id,
    )


def integrate_reduced_rod_step(
    dynamics: PreparedReducedRodDynamics,
    policy: ReducedRodIntegratorPolicy,
    source: ReducedRodIntegrationState,
    step_size: ArrayLike,
    /,
    *,
    material_control: ReducedRodMaterialControl | None = None,
    native_loads: RodLoadLedger | None = None,
) -> ReducedRodStepResult:
    """Execute exactly the selected route and fail closed without route fallback."""
    _validate_source(dynamics, source)
    if native_loads is not None and not isinstance(native_loads, RodLoadLedger):
        raise TypeError("native_loads must be RodLoadLedger or None.")
    step = jnp.asarray(step_size, dtype=source.reduced_state.values.dtype)
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    safe_step = jnp.where(jnp.isfinite(step) & (step > 0.0), step, jnp.ones_like(step))
    if isinstance(policy, ReducedRodSemiImplicitVelocityEuler):
        result = _semi_implicit_step(
            dynamics, policy, source, safe_step, material_control, native_loads
        )
    elif isinstance(policy, ReducedRodImplicitMidpoint):
        result = _implicit_midpoint_step(
            dynamics, policy, source, safe_step, material_control, native_loads
        )
    else:
        raise TypeError("policy must select an explicit reduced rod integrator route.")
    requested_valid = jnp.isfinite(step) & (step > 0.0)
    if_requested = result.successful & requested_valid
    status = jnp.where(
        requested_valid,
        result.status,
        jnp.asarray(ReducedRodStepStatus.STEP_OUT_OF_BOUNDS, dtype=jnp.int32),
    )
    accepted_state = _selected_state(
        if_requested, result.candidate_state, result.previous_state
    )
    evidence = eqx.tree_at(
        lambda value: (value.step_finite, value.accepted, value.status),
        result.evidence,
        (requested_valid, if_requested, status),
    )
    return ReducedRodStepResult(
        result.previous_state,
        result.candidate_state,
        accepted_state,
        result.attempted,
        if_requested,
        status,
        result.backend_status,
        evidence,
        result.policy_id,
    )


__all__ = [
    "initialize_reduced_rod_integration_state",
    "integrate_reduced_rod_step",
    "ReducedRodEnergyWorkLedger",
    "ReducedRodImplicitMidpoint",
    "ReducedRodIntegrationState",
    "ReducedRodIntegratorPolicy",
    "ReducedRodIntegratorRoute",
    "ReducedRodSemiImplicitVelocityEuler",
    "ReducedRodStepEvidence",
    "ReducedRodStepResult",
    "ReducedRodStepStatus",
]
