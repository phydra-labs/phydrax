#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import SecondOrderDifferentialSystem
from ...equations import (
    CompiledFiniteElementProblem,
    FiniteElementExecutionContext,
    MaterialTransaction,
)
from ...equations.fem import FiniteElementMassPolicy
from ...linalg import ArraySpace
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearPrecisionPolicy,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)


class ImplicitNewmarkMethod(StrictModule, NonTrainableState):
    """Implicit Newmark kinematics with explicit stability evidence."""

    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    second_order: bool = eqx.field(static=True)
    unconditionally_stable: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float = 0.25,
        gamma: float = 0.5,
        /,
        *,
        method_id: str | None = None,
    ):
        beta_ = float(beta)
        gamma_ = float(gamma)
        if not isfinite(beta_) or not isfinite(gamma_) or beta_ <= 0.0 or gamma_ <= 0.0:
            raise ValueError("Newmark beta and gamma must be positive finite values.")
        second_order = gamma_ == 0.5
        stable = gamma_ >= 0.5 and beta_ >= 0.25 * (gamma_ + 0.5) ** 2
        generated = canonical_fingerprint(
            {
                "kind": "implicit-newmark-method",
                "beta": beta_.hex(),
                "gamma": gamma_.hex(),
                "second_order": second_order,
                "unconditionally_stable": stable,
            }
        )
        identifier = generated if method_id is None else str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty or None.")
        self.beta = beta_
        self.gamma = gamma_
        self.second_order = second_order
        self.unconditionally_stable = stable
        self.method_id = identifier

    def rates(
        self,
        displacement: Array,
        accepted: FiniteElementDynamicsState,
        step_size: Array,
        /,
    ) -> tuple[Array, Array]:
        dt = jnp.asarray(step_size, dtype=displacement.dtype)
        predicted_displacement = (
            accepted.displacement
            + dt * accepted.velocity
            + dt * dt * (0.5 - self.beta) * accepted.acceleration
        )
        acceleration = (displacement - predicted_displacement) / (self.beta * dt * dt)
        predicted_velocity = (
            accepted.velocity + dt * (1.0 - self.gamma) * accepted.acceleration
        )
        velocity = predicted_velocity + self.gamma * dt * acceleration
        return velocity, acceleration

    def predictor(
        self,
        accepted: FiniteElementDynamicsState,
        step_size: Array,
        /,
    ) -> Array:
        dt = jnp.asarray(step_size, dtype=accepted.displacement.dtype)
        return (
            accepted.displacement
            + dt * accepted.velocity
            + dt * dt * (0.5 - self.beta) * accepted.acceleration
        )

    def position_to_acceleration_scale(
        self, step_size: ArrayLike, dtype: Any, /
    ) -> Array:
        """Return the scalar derivative of Newmark acceleration with respect to position."""
        dt = jnp.asarray(step_size, dtype=dtype)
        return 1.0 / (self.beta * dt * dt)

    def position_to_velocity_scale(self, step_size: ArrayLike, dtype: Any, /) -> Array:
        """Return the scalar derivative of Newmark velocity with respect to position."""
        dt = jnp.asarray(step_size, dtype=dtype)
        return self.gamma / (self.beta * dt)


class FiniteElementDynamicsState(StrictModule, NonTrainableState):
    """Displacement, rate, acceleration, and committed material history."""

    displacement: Array
    velocity: Array
    acceleration: Array
    materials: MaterialTransaction | None
    time: Array
    step: Array
    state_version: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        displacement: ArrayLike,
        velocity: ArrayLike,
        acceleration: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        step: ArrayLike = 0,
        state_version: ArrayLike = 0,
        materials: MaterialTransaction | None = None,
    ):
        displacement_ = jnp.asarray(displacement)
        velocity_ = jnp.asarray(velocity)
        acceleration_ = jnp.asarray(acceleration)
        time_ = jnp.asarray(time, dtype=displacement_.real.dtype)
        step_ = jnp.asarray(step, dtype=jnp.int32)
        version = jnp.asarray(state_version, dtype=jnp.int32)
        if (
            displacement_.shape != velocity_.shape
            or displacement_.shape != acceleration_.shape
            or displacement_.ndim < 1
        ):
            raise ValueError(
                "FEM displacement, velocity, and acceleration shapes must agree."
            )
        if not jnp.issubdtype(displacement_.dtype, jnp.inexact):
            raise TypeError("FEM dynamic state arrays must use an inexact dtype.")
        if (
            velocity_.dtype != displacement_.dtype
            or acceleration_.dtype != displacement_.dtype
        ):
            raise TypeError("FEM dynamic state arrays must have one common dtype.")
        if time_.shape != () or step_.shape != () or version.shape != ():
            raise ValueError("FEM dynamic time, step, and version must be scalars.")
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("materials must be MaterialTransaction or None.")
        material_layout = None if materials is None else materials.layout_id
        self.displacement = displacement_
        self.velocity = velocity_
        self.acceleration = acceleration_
        self.materials = materials
        self.time = time_
        self.step = step_
        self.state_version = version
        self.layout_id = canonical_fingerprint(
            {
                "kind": "finite-element-dynamics-state-layout",
                "shape": list(displacement_.shape),
                "dtype": str(displacement_.dtype),
                "materials": material_layout,
            }
        )


class FiniteElementAdmissibilityEvidence(StrictModule):
    """Finite-state, orientation, constitutive, and user-hook evidence."""

    minimum_jacobian: Array
    jacobian_available: Array
    jacobian_valid: Array
    constitutive_valid: Array
    transaction_consistency_residual: Array
    custom_valid: Array
    finite: Array
    admissible: Array


class FiniteElementEnergyWorkLedger(StrictModule):
    """Endpoint energy and integrated work evidence for one candidate step."""

    kinetic_before: Array
    kinetic_after: Array
    potential_before: Array
    potential_after: Array
    external_work: Array
    damping_work: Array
    balance_residual: Array
    balance_scale: Array
    available: Array
    finite: Array
    balanced: Array


class FiniteElementDynamicsCandidate(StrictModule):
    """Uncommitted Newmark candidate and its physical evidence."""

    state: FiniteElementDynamicsState
    admissibility: FiniteElementAdmissibilityEvidence
    energy: FiniteElementEnergyWorkLedger


class FiniteElementDynamicsResult(StrictModule):
    """Candidate plus atomic promotion or rollback of every dynamic field."""

    previous: FiniteElementDynamicsState
    candidate: FiniteElementDynamicsCandidate
    accepted_state: FiniteElementDynamicsState
    accepted: Array
    rollback_applied: Array
    nonlinear: NonlinearResult
    plan_id: str = eqx.field(static=True)

    def promote(self, /) -> FiniteElementDynamicsState:
        """Apply the scalar host decision and increment committed material versions."""
        if not bool(self.accepted):
            return self.previous
        state = self.candidate.state
        materials = None if state.materials is None else state.materials.commit()
        return FiniteElementDynamicsState(
            state.displacement,
            state.velocity,
            state.acceleration,
            time=state.time,
            step=state.step,
            state_version=state.state_version,
            materials=materials,
        )


class FiniteElementDynamicsExecutionContext(StrictModule):
    """Accepted material history and user data visible to transient FE kernels."""

    materials: MaterialTransaction | None
    start_time: Array
    end_time: Array
    step_size: Array
    user_args: Any


class _FiniteElementStepArguments(StrictModule):
    accepted: FiniteElementDynamicsState
    step_size: Array
    end_time: Array
    user_args: Any


def _system_arguments(
    arguments: _FiniteElementStepArguments,
    /,
) -> FiniteElementDynamicsExecutionContext | FiniteElementExecutionContext:
    raw = arguments.user_args
    user_args = raw.user_args if isinstance(raw, FiniteElementExecutionContext) else raw
    dynamics = FiniteElementDynamicsExecutionContext(
        arguments.accepted.materials,
        arguments.accepted.time,
        arguments.end_time,
        arguments.step_size,
        user_args,
    )
    if not isinstance(raw, FiniteElementExecutionContext):
        return dynamics
    return FiniteElementExecutionContext(
        raw.runtime,
        time=raw.time,
        lift=raw.lift,
        lift_rate=raw.lift_rate,
        lift_acceleration=raw.lift_acceleration,
        metric_data=raw.metric_data,
        user_args=dynamics,
    )


class _NewmarkResidual(StrictModule, NonTrainableState):
    system: SecondOrderDifferentialSystem
    method: ImplicitNewmarkMethod

    def __call__(self, displacement: Array, arguments: _FiniteElementStepArguments):
        velocity, acceleration = self.method.rates(
            displacement, arguments.accepted, arguments.step_size
        )
        return self.system.evaluate(
            arguments.end_time,
            displacement,
            velocity,
            acceleration,
            _system_arguments(arguments),
        )


class _NewmarkValidity(StrictModule, NonTrainableState):
    method: ImplicitNewmarkMethod
    determinant_evaluator: Callable | None
    admissibility_evaluator: Callable | None
    minimum_jacobian: float = eqx.field(static=True)

    def __call__(
        self,
        displacement: Array,
        residual: Array,
        auxiliary: object,
        arguments: _FiniteElementStepArguments,
    ) -> Array:
        del auxiliary
        velocity, acceleration = self.method.rates(
            displacement, arguments.accepted, arguments.step_size
        )
        finite = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(acceleration))
            & jnp.all(jnp.isfinite(residual))
        )
        if self.determinant_evaluator is None:
            jacobian_valid = jnp.asarray(True)
        else:
            determinants = jnp.asarray(
                self.determinant_evaluator(
                    arguments.end_time,
                    displacement,
                    arguments.user_args,
                )
            )
            jacobian_valid = jnp.all(
                jnp.isfinite(determinants) & (determinants > self.minimum_jacobian)
            )
        if self.admissibility_evaluator is None:
            custom_valid = jnp.asarray(True)
        else:
            custom_valid = jnp.all(
                jnp.asarray(
                    self.admissibility_evaluator(
                        arguments.end_time,
                        displacement,
                        velocity,
                        acceleration,
                        arguments.user_args,
                    ),
                    dtype=bool,
                )
            )
        return finite & jacobian_valid & custom_valid


class FiniteElementDynamicsPlan(StrictModule, NonTrainableState):
    """Reusable symbolic Newmark solve and all declared physical hooks."""

    problem: CompiledFiniteElementProblem
    system: SecondOrderDifferentialSystem
    mass_problem: CompiledFiniteElementProblem
    method: ImplicitNewmarkMethod
    root_problem: NonlinearSystemProblem
    nonlinear_template: PreparedNonlinearSolve
    mass_coefficient: Array
    mass_policy: FiniteElementMassPolicy
    damping_coefficient: Array
    derivative_policy: ImplicitRootDerivativePolicy | None
    determinant_evaluator: Callable | None
    admissibility_evaluator: Callable | None
    material_update: Callable | None
    potential_energy: Callable | None
    external_work: Callable | None
    minimum_jacobian: float = eqx.field(static=True)
    transaction_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedFiniteElementDynamicsStep(StrictModule):
    """One numeric Newmark root refreshed against a reusable symbolic plan."""

    plan: FiniteElementDynamicsPlan
    arguments: _FiniteElementStepArguments
    nonlinear: PreparedNonlinearSolve


def _hook_id(value: Callable | None, identifier: str | None, owner: str, /) -> str | None:
    if value is None:
        if identifier is not None:
            raise ValueError(f"{owner}_id requires a corresponding callable.")
        return None
    if not callable(value):
        raise TypeError(f"{owner} must be callable or None.")
    if identifier is None or not str(identifier):
        raise ValueError(f"{owner}_id must be non-empty when {owner} is supplied.")
    return str(identifier)


def _finite_tree(value: object, /) -> Array:
    leaves = jax.tree.leaves(value)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves))
    )


def _validate_state(
    problem: CompiledFiniteElementProblem,
    state: FiniteElementDynamicsState,
    /,
) -> FiniteElementDynamicsState:
    if not isinstance(state, FiniteElementDynamicsState):
        raise TypeError("state must be FiniteElementDynamicsState.")
    problem.state_space.validate(state.displacement)
    problem.state_space.validate(state.velocity)
    problem.state_space.validate(state.acceleration)
    displacement = eqx.error_if(
        state.displacement,
        ~(
            _finite_tree(state.displacement)
            & _finite_tree(state.velocity)
            & _finite_tree(state.acceleration)
            & _finite_tree(state.materials)
            & jnp.isfinite(state.time)
            & (state.step >= 0)
            & (state.state_version >= 0)
        ),
        "Accepted FEM dynamic state must be finite with nonnegative counters.",
    )
    return eqx.tree_at(lambda value: value.displacement, state, displacement)


def prepare_finite_element_dynamics(
    problem: CompiledFiniteElementProblem,
    initial_state: FiniteElementDynamicsState,
    sample_step: ArrayLike,
    /,
    *,
    method: ImplicitNewmarkMethod | None = None,
    mass_coefficient: ArrayLike = 1.0,
    mass_policy: FiniteElementMassPolicy | None = None,
    damping_coefficient: ArrayLike = 0.0,
    nonlinear_method: AbstractNonlinearMethod | None = None,
    nonlinear_termination: NonlinearTermination | None = None,
    nonlinear_precision: NonlinearPrecisionPolicy | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
    args: object = None,
    determinant_evaluator: Callable | None = None,
    determinant_id: str | None = None,
    admissibility_evaluator: Callable | None = None,
    admissibility_id: str | None = None,
    material_update: Callable | None = None,
    material_update_id: str | None = None,
    potential_energy: Callable | None = None,
    potential_energy_id: str | None = None,
    external_work: Callable | None = None,
    external_work_id: str | None = None,
    minimum_jacobian: float = 0.0,
    transaction_tolerance: float = 1.0e-12,
    energy_tolerance: float = 1.0e-8,
    plan_id: str | None = None,
) -> FiniteElementDynamicsPlan:
    """Prepare a reusable implicit Newmark root over one compiled vector FE field."""
    if not isinstance(problem, CompiledFiniteElementProblem):
        raise TypeError("problem must be CompiledFiniteElementProblem.")
    if len(problem.form.field_names) != 1 or not isinstance(
        problem.state_space, ArraySpace
    ):
        raise ValueError("Transient FEM dynamics requires one array-valued FE field.")
    initial_state = _validate_state(problem, initial_state)
    selected_method = ImplicitNewmarkMethod() if method is None else method
    if not isinstance(selected_method, ImplicitNewmarkMethod):
        raise TypeError("method must be ImplicitNewmarkMethod or None.")
    selected_mass_policy = (
        FiniteElementMassPolicy() if mass_policy is None else mass_policy
    )
    if not isinstance(selected_mass_policy, FiniteElementMassPolicy):
        raise TypeError("mass_policy must be FiniteElementMassPolicy or None.")
    selected_nonlinear = NewtonKrylov() if nonlinear_method is None else nonlinear_method
    if not isinstance(selected_nonlinear, (NewtonKrylov, NewtonTrustRegion)):
        raise ValueError("FEM dynamics requires a prepared Newton nonlinear method.")
    termination = (
        NonlinearTermination() if nonlinear_termination is None else nonlinear_termination
    )
    precision = (
        NonlinearPrecisionPolicy() if nonlinear_precision is None else nonlinear_precision
    )
    if not isinstance(termination, NonlinearTermination):
        raise TypeError("nonlinear_termination must be NonlinearTermination or None.")
    if not isinstance(precision, NonlinearPrecisionPolicy):
        raise TypeError("nonlinear_precision must be NonlinearPrecisionPolicy or None.")
    if derivative_policy is not None and not isinstance(
        derivative_policy, ImplicitRootDerivativePolicy
    ):
        raise TypeError("derivative_policy must be ImplicitRootDerivativePolicy or None.")
    mass_host = np.asarray(mass_coefficient)
    damping_host = np.asarray(damping_coefficient)
    if mass_host.shape != () or not np.isfinite(mass_host) or mass_host <= 0.0:
        raise ValueError("mass_coefficient must be one positive finite scalar.")
    if damping_host.shape != () or not np.isfinite(damping_host) or damping_host < 0.0:
        raise ValueError("damping_coefficient must be one finite nonnegative scalar.")
    minimum = float(minimum_jacobian)
    transaction_limit = float(transaction_tolerance)
    energy_limit = float(energy_tolerance)
    if (
        not isfinite(minimum)
        or not isfinite(transaction_limit)
        or transaction_limit < 0.0
        or not isfinite(energy_limit)
        or energy_limit < 0.0
    ):
        raise ValueError("FEM admissibility and ledger tolerances are invalid.")
    determinant_name = _hook_id(
        determinant_evaluator, determinant_id, "determinant_evaluator"
    )
    admissibility_name = _hook_id(
        admissibility_evaluator, admissibility_id, "admissibility_evaluator"
    )
    material_name = _hook_id(material_update, material_update_id, "material_update")
    potential_name = _hook_id(potential_energy, potential_energy_id, "potential_energy")
    external_name = _hook_id(external_work, external_work_id, "external_work")
    if material_update is not None and initial_state.materials is None:
        raise ValueError("material_update requires an initial material transaction.")
    step_host = np.asarray(sample_step)
    if step_host.shape != () or not np.isfinite(step_host) or step_host <= 0.0:
        raise ValueError("sample_step must be one positive finite scalar.")
    generated = canonical_fingerprint(
        {
            "kind": "finite-element-transient-dynamics",
            "compilation": problem.compilation_id,
            "method": selected_method.method_id,
            "mass": float(mass_host).hex(),
            "mass_policy": selected_mass_policy.kind,
            "damping": float(damping_host).hex(),
            "nonlinear_method": selected_nonlinear.method_id,
            "precision": precision.policy_id,
            "hooks": {
                "determinant": determinant_name,
                "admissibility": admissibility_name,
                "material": material_name,
                "potential": potential_name,
                "external_work": external_name,
            },
            "minimum_jacobian": minimum.hex(),
            "transaction_tolerance": transaction_limit.hex(),
            "energy_tolerance": energy_limit.hex(),
        }
    )
    identifier = generated if plan_id is None else str(plan_id)
    if not identifier:
        raise ValueError("plan_id must be non-empty or None.")
    mass = jnp.asarray(mass_host, dtype=initial_state.displacement.dtype)
    damping = jnp.asarray(damping_host, dtype=initial_state.displacement.dtype)
    system = problem.as_second_order_system(
        mass_coefficient=float(mass_host),
        mass_policy=selected_mass_policy,
        damping_coefficient=float(damping_host),
        system_id=f"{identifier}:second-order",
    )
    mass_problem = problem._compile_unit_mass_problem(selected_mass_policy)
    root_problem = NonlinearSystemProblem(
        _NewmarkResidual(system, selected_method),
        state_space=problem.state_space,
        residual_space=problem.state_space,
        validity=_NewmarkValidity(
            selected_method,
            determinant_evaluator,
            admissibility_evaluator,
            minimum,
        ),
        problem_id=f"{identifier}:newmark-root",
    )
    sample_step_ = jnp.asarray(step_host, dtype=initial_state.displacement.dtype)
    sample_arguments = _FiniteElementStepArguments(
        initial_state,
        sample_step_,
        initial_state.time + sample_step_,
        args,
    )
    template = prepare_nonlinear(
        root_problem,
        selected_method.predictor(initial_state, sample_step_),
        method=selected_nonlinear,
        termination=termination,
        args=sample_arguments,
        precision=precision,
    )
    return FiniteElementDynamicsPlan(
        problem,
        system,
        mass_problem,
        selected_method,
        root_problem,
        template,
        mass,
        selected_mass_policy,
        damping,
        derivative_policy,
        determinant_evaluator,
        admissibility_evaluator,
        material_update,
        potential_energy,
        external_work,
        minimum,
        transaction_limit,
        energy_limit,
        identifier,
    )


def prepare_finite_element_dynamics_step(
    plan: FiniteElementDynamicsPlan,
    accepted: FiniteElementDynamicsState,
    step_size: ArrayLike,
    /,
    *,
    args: object = None,
) -> PreparedFiniteElementDynamicsStep:
    """Refresh one numeric Newmark step without changing its symbolic linear plan."""
    if not isinstance(plan, FiniteElementDynamicsPlan):
        raise TypeError("plan must be FiniteElementDynamicsPlan.")
    accepted = _validate_state(plan.problem, accepted)
    dt = jnp.asarray(step_size, dtype=accepted.displacement.dtype)
    if dt.shape != ():
        raise ValueError("step_size must be scalar.")
    dt = eqx.error_if(
        dt,
        ~jnp.isfinite(dt) | (dt <= 0.0),
        "step_size must be positive and finite.",
    )
    arguments = _FiniteElementStepArguments(
        accepted,
        dt,
        accepted.time + dt,
        args,
    )
    nonlinear = refresh_nonlinear(
        plan.nonlinear_template,
        plan.root_problem,
        plan.method.predictor(accepted, dt),
        args=arguments,
    )
    return PreparedFiniteElementDynamicsStep(plan, arguments, nonlinear)


def _material_transaction(
    plan: FiniteElementDynamicsPlan,
    arguments: _FiniteElementStepArguments,
    displacement: Array,
    velocity: Array,
    acceleration: Array,
    /,
) -> tuple[MaterialTransaction | None, Array, Array]:
    previous = arguments.accepted.materials
    if plan.material_update is None:
        return (
            previous,
            jnp.asarray(0.0, dtype=displacement.real.dtype),
            jnp.asarray(True),
        )
    candidate = plan.material_update(
        displacement,
        velocity,
        acceleration,
        arguments.end_time,
        arguments.step_size,
        previous,
        arguments.user_args,
    )
    if not isinstance(candidate, MaterialTransaction):
        raise TypeError("material_update must return MaterialTransaction.")
    if previous is None or previous.layout_id != candidate.layout_id:
        raise ValueError("Material update changed the transaction layout.")
    residual = jnp.asarray(0.0, dtype=displacement.real.dtype)
    for old, new in zip(previous.states, candidate.states, strict=True):
        if old.layout_id != new.layout_id or old.state_version != new.state_version:
            raise ValueError(
                "Material update changed a site identity, state layout, or version."
            )
        residual = jnp.maximum(
            residual,
            jnp.max(jnp.abs(new.committed - old.committed), initial=0.0),
        )
    finite = _finite_tree(candidate)
    return candidate, residual, finite


def _selected_materials(
    previous: MaterialTransaction | None,
    candidate: MaterialTransaction | None,
    accepted: Array,
    /,
) -> MaterialTransaction | None:
    if previous is None:
        return None
    if candidate is None:
        return previous
    trials = {
        old.site_id.key: jnp.where(accepted, new.trial, old.committed)
        for old, new in zip(previous.states, candidate.states, strict=True)
    }
    return previous.with_trials(trials)


def _admissibility(
    plan: FiniteElementDynamicsPlan,
    arguments: _FiniteElementStepArguments,
    displacement: Array,
    velocity: Array,
    acceleration: Array,
    materials: MaterialTransaction | None,
    transaction_residual: Array,
    material_finite: Array,
    /,
) -> FiniteElementAdmissibilityEvidence:
    if plan.determinant_evaluator is None:
        minimum = jnp.asarray(jnp.inf, dtype=displacement.real.dtype)
        jacobian_available = jnp.asarray(False)
        jacobian_valid = jnp.asarray(True)
    else:
        determinant = jnp.asarray(
            plan.determinant_evaluator(
                arguments.end_time, displacement, arguments.user_args
            )
        )
        minimum = jnp.min(determinant, initial=jnp.inf)
        jacobian_available = jnp.asarray(True)
        jacobian_valid = jnp.all(
            jnp.isfinite(determinant) & (determinant > plan.minimum_jacobian)
        )
    if plan.admissibility_evaluator is None:
        custom_valid = jnp.asarray(True)
    else:
        custom_valid = jnp.all(
            jnp.asarray(
                plan.admissibility_evaluator(
                    arguments.end_time,
                    displacement,
                    velocity,
                    acceleration,
                    arguments.user_args,
                ),
                dtype=bool,
            )
        )
    finite = (
        _finite_tree(displacement)
        & _finite_tree(velocity)
        & _finite_tree(acceleration)
        & material_finite
        & _finite_tree(materials)
    )
    constitutive_valid = material_finite & (
        transaction_residual <= plan.transaction_tolerance
    )
    admissible = finite & jacobian_valid & constitutive_valid & custom_valid
    return FiniteElementAdmissibilityEvidence(
        minimum,
        jacobian_available,
        jacobian_valid,
        constitutive_valid,
        transaction_residual,
        custom_valid,
        finite,
        admissible,
    )


def _scalar_hook(value: object, owner: str, dtype, /) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{owner} must return one scalar.")
    return result


def _energy_ledger(
    plan: FiniteElementDynamicsPlan,
    arguments: _FiniteElementStepArguments,
    candidate: FiniteElementDynamicsState,
    /,
) -> FiniteElementEnergyWorkLedger:
    previous = arguments.accepted
    dtype = candidate.displacement.real.dtype
    context = plan.problem._execution_context(arguments.user_args)
    _, mass = plan.problem._mass_operators(
        context,
        plan.mass_coefficient,
        plan.mass_policy,
        plan.mass_problem,
    )
    mass_velocity_before = plan.problem.state_space.inverse_riesz(
        mass.mv(previous.velocity)
    )
    mass_velocity_after = plan.problem.state_space.inverse_riesz(
        mass.mv(candidate.velocity)
    )
    kinetic_before = 0.5 * jnp.real(
        plan.problem.state_space.inner(previous.velocity, mass_velocity_before)
    )
    kinetic_after = 0.5 * jnp.real(
        plan.problem.state_space.inner(candidate.velocity, mass_velocity_after)
    )
    if plan.potential_energy is None:
        potential_before = jnp.asarray(0.0, dtype=dtype)
        potential_after = jnp.asarray(0.0, dtype=dtype)
    else:
        potential_before = _scalar_hook(
            plan.potential_energy(
                previous.time, previous.displacement, arguments.user_args
            ),
            "potential_energy",
            dtype,
        )
        potential_after = _scalar_hook(
            plan.potential_energy(
                candidate.time, candidate.displacement, arguments.user_args
            ),
            "potential_energy",
            dtype,
        )
    if plan.external_work is None:
        external_work = jnp.asarray(0.0, dtype=dtype)
    else:
        external_work = _scalar_hook(
            plan.external_work(previous, candidate, arguments.user_args),
            "external_work",
            dtype,
        )
    damping_power = plan.damping_coefficient * jnp.real(
        plan.problem.state_space.inner(candidate.velocity, mass_velocity_after)
    )
    damping_work = arguments.step_size * damping_power
    balance = (
        kinetic_after
        + potential_after
        - kinetic_before
        - potential_before
        + damping_work
        - external_work
    )
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.max(
            jnp.abs(
                jnp.stack(
                    (
                        kinetic_before,
                        kinetic_after,
                        potential_before,
                        potential_after,
                        external_work,
                        damping_work,
                    )
                )
            )
        ),
    )
    available = jnp.asarray(
        plan.potential_energy is not None and plan.external_work is not None
    )
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    kinetic_before,
                    kinetic_after,
                    potential_before,
                    potential_after,
                    external_work,
                    damping_work,
                    balance,
                    scale,
                )
            )
        )
    )
    balanced = available & finite & (jnp.abs(balance) <= plan.energy_tolerance * scale)
    return FiniteElementEnergyWorkLedger(
        kinetic_before,
        kinetic_after,
        potential_before,
        potential_after,
        external_work,
        damping_work,
        balance,
        scale,
        available,
        finite,
        balanced,
    )


def solve_finite_element_dynamics_step(
    prepared: PreparedFiniteElementDynamicsStep,
    /,
) -> FiniteElementDynamicsResult:
    """Solve, certify, and atomically commit or roll back one prepared step."""
    if not isinstance(prepared, PreparedFiniteElementDynamicsStep):
        raise TypeError("prepared must be PreparedFiniteElementDynamicsStep.")
    plan = prepared.plan
    arguments = prepared.arguments
    previous = arguments.accepted
    nonlinear = implicit_root_result(
        prepared.nonlinear,
        derivative_policy=plan.derivative_policy,
    )
    displacement = plan.problem.state_space.validate(nonlinear.state)
    velocity, acceleration = plan.method.rates(
        displacement, previous, arguments.step_size
    )
    materials, transaction_residual, material_finite = _material_transaction(
        plan,
        arguments,
        displacement,
        velocity,
        acceleration,
    )
    candidate_state = FiniteElementDynamicsState(
        displacement,
        velocity,
        acceleration,
        time=arguments.end_time,
        step=previous.step + 1,
        state_version=previous.state_version + 1,
        materials=materials,
    )
    admissibility = _admissibility(
        plan,
        arguments,
        displacement,
        velocity,
        acceleration,
        materials,
        transaction_residual,
        material_finite,
    )
    energy = _energy_ledger(plan, arguments, candidate_state)
    accepted = nonlinear.successful & admissibility.admissible
    committed_materials = _selected_materials(
        previous.materials,
        materials,
        accepted,
    )
    accepted_state = FiniteElementDynamicsState(
        jnp.where(accepted, displacement, previous.displacement),
        jnp.where(accepted, velocity, previous.velocity),
        jnp.where(accepted, acceleration, previous.acceleration),
        time=jnp.where(accepted, arguments.end_time, previous.time),
        step=jnp.where(accepted, previous.step + 1, previous.step),
        state_version=jnp.where(
            accepted, previous.state_version + 1, previous.state_version
        ),
        materials=committed_materials,
    )
    candidate = FiniteElementDynamicsCandidate(candidate_state, admissibility, energy)
    return FiniteElementDynamicsResult(
        previous,
        candidate,
        accepted_state,
        accepted,
        ~accepted,
        nonlinear,
        plan.plan_id,
    )


__all__ = [
    "FiniteElementAdmissibilityEvidence",
    "FiniteElementDynamicsCandidate",
    "FiniteElementDynamicsExecutionContext",
    "FiniteElementDynamicsPlan",
    "FiniteElementDynamicsResult",
    "FiniteElementDynamicsState",
    "FiniteElementEnergyWorkLedger",
    "ImplicitNewmarkMethod",
    "PreparedFiniteElementDynamicsStep",
    "prepare_finite_element_dynamics",
    "prepare_finite_element_dynamics_step",
    "solve_finite_element_dynamics_step",
]
