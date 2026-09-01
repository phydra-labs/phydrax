#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
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
from ...discretization.contact import (
    CertifiedAABBCCDPlan,
    collision_free_step_limit,
    ContactCandidateEpoch,
    ContactSafetyEvidence,
    InclusionCCDPlan,
    InversionStepEvidence,
    PreparedCollisionScene,
    simplex_inversion_step_limit,
    SimplexInversionStepPlan,
    SweepAndPruneContactSearchPlan,
)
from ...equations import CompiledFiniteElementProblem
from ...equations.fem import FiniteElementMassPolicy
from ...linalg import ArraySpace, FunctionLinearOperator, OperatorProperties
from ...optim import (
    solve_trust_region_subproblem,
    SteihaugToint,
    TrustRegionQuadraticProblem,
)
from ..solid_mechanics._fem_dynamics import (
    FiniteElementDynamicsState,
    ImplicitNewmarkMethod,
)
from ._friction import (
    ContactFrictionEvaluation,
    ContactFrictionState,
    PreparedLaggedCoulombFriction,
)
from ._potential import ContactPotentialEvaluation, PreparedConvergentContactPotential


ContactCCDPlan = InclusionCCDPlan | CertifiedAABBCCDPlan


class ContactRejectionReason(IntFlag):
    NONE = 0
    NONFINITE = 1 << 0
    SEARCH = 1 << 1
    CCD = 1 << 2
    INVERSION = 1 << 3
    NONLINEAR = 1 << 4
    LINE_SEARCH = 1 << 5
    CONTACT = 1 << 6
    ENERGY = 1 << 7
    FRICTION = 1 << 8


class ContactSolvePolicy(StrictModule, NonTrainableState):
    absolute_gradient: float = eqx.field(static=True)
    relative_gradient: float = eqx.field(static=True)
    absolute_step: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    armijo: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    initial_trust_radius: float = eqx.field(static=True)
    maximum_trust_radius: float = eqx.field(static=True)
    minimum_trust_radius: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_gradient: float = 1.0e-8,
        relative_gradient: float = 1.0e-8,
        absolute_step: float = 1.0e-12,
        maximum_iterations: int = 100,
        maximum_line_search_steps: int = 24,
        armijo: float = 1.0e-4,
        contraction: float = 0.5,
        initial_trust_radius: float = 1.0,
        maximum_trust_radius: float = 1.0e4,
        minimum_trust_radius: float = 1.0e-12,
    ):
        values = tuple(
            float(value)
            for value in (
                absolute_gradient,
                relative_gradient,
                absolute_step,
                armijo,
                contraction,
                initial_trust_radius,
                maximum_trust_radius,
                minimum_trust_radius,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Contact solve controls must be finite and positive.")
        if not values[3] < 1.0 or not values[4] < 1.0:
            raise ValueError("Armijo and contraction values must lie below one.")
        if not values[7] <= values[5] <= values[6]:
            raise ValueError("Trust radii are inconsistently ordered.")
        iterations = int(maximum_iterations)
        search_steps = int(maximum_line_search_steps)
        if iterations <= 0 or search_steps <= 0:
            raise ValueError("Contact iteration limits must be positive.")
        (
            self.absolute_gradient,
            self.relative_gradient,
            self.absolute_step,
            self.armijo,
            self.contraction,
            self.initial_trust_radius,
            self.maximum_trust_radius,
            self.minimum_trust_radius,
        ) = values
        self.maximum_iterations = iterations
        self.maximum_line_search_steps = search_steps
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contact-solve-policy",
                "values": tuple(value.hex() for value in values),
                "maximum_iterations": iterations,
                "maximum_line_search_steps": search_steps,
            }
        )


class ContactSolveDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    hessian_actions: Array
    linear_iterations: Array
    line_search_evaluations: Array
    direction_fallbacks: Array
    initial_gradient_norm: Array
    final_gradient_norm: Array
    final_step_norm: Array
    accepted_step_size: Array
    trust_radius: Array
    status_reason: Array
    lag_iterations: Array
    lag_residual: Array


class ContactEnergyLedger(StrictModule):
    initial_total: Array
    final_total: Array
    bulk_potential: Array
    contact_potential: Array
    inertial_potential: Array
    friction_potential: Array
    decrease: Array
    finite: Array


class ContactDynamicsState(StrictModule, NonTrainableState):
    mechanics: FiniteElementDynamicsState
    replay_epoch: ContactCandidateEpoch | None
    friction_state: ContactFrictionState | None
    state_version: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanics: FiniteElementDynamicsState,
        /,
        *,
        replay_epoch: ContactCandidateEpoch | None = None,
        friction_state: ContactFrictionState | None = None,
        state_version: ArrayLike = 0,
    ):
        if not isinstance(mechanics, FiniteElementDynamicsState):
            raise TypeError("mechanics must be FiniteElementDynamicsState.")
        if replay_epoch is not None and not isinstance(
            replay_epoch, ContactCandidateEpoch
        ):
            raise TypeError("replay_epoch must be ContactCandidateEpoch or None.")
        if friction_state is not None and not isinstance(
            friction_state, ContactFrictionState
        ):
            raise TypeError("friction_state must be ContactFrictionState or None.")
        version = jnp.asarray(state_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("state_version must be scalar.")
        self.mechanics = mechanics
        self.replay_epoch = replay_epoch
        self.friction_state = friction_state
        self.state_version = version
        self.state_id = canonical_fingerprint(
            {
                "kind": "contact-dynamics-state",
                "mechanics_layout": mechanics.layout_id,
                "replay": None if replay_epoch is None else replay_epoch.epoch_id,
                "friction": (
                    None if friction_state is None else friction_state.prepared_id
                ),
            }
        )


class FiniteElementContactDynamicsPlan(StrictModule, NonTrainableState):
    problem: CompiledFiniteElementProblem
    scene: PreparedCollisionScene
    contact: PreparedConvergentContactPotential
    search: SweepAndPruneContactSearchPlan
    friction: PreparedLaggedCoulombFriction | None
    ccd: ContactCCDPlan
    inversion: SimplexInversionStepPlan | None
    method: ImplicitNewmarkMethod
    mass_policy: FiniteElementMassPolicy
    mass_problem: CompiledFiniteElementProblem
    mass_coefficient: Array
    solve_policy: ContactSolvePolicy
    subproblem: SteihaugToint
    plan_id: str = eqx.field(static=True)


class PreparedFiniteElementContactStep(StrictModule, NonTrainableState):
    plan: FiniteElementContactDynamicsPlan
    accepted: ContactDynamicsState
    step_size: Array
    args: Any
    prepared_id: str = eqx.field(static=True)


class FiniteElementContactResult(StrictModule):
    previous: ContactDynamicsState
    candidate: ContactDynamicsState
    accepted_state: ContactDynamicsState
    accepted: Array
    rollback_applied: Array
    diagnostics: ContactSolveDiagnostics
    contact: ContactPotentialEvaluation
    friction: ContactFrictionEvaluation | None
    safety: ContactSafetyEvidence
    inversion: InversionStepEvidence | None
    energy: ContactEnergyLedger
    rejection_reasons: Array
    plan_id: str = eqx.field(static=True)

    def promote(self, /) -> ContactDynamicsState:
        return self.accepted_state


class FiniteElementContactEquilibriumPlan(StrictModule, NonTrainableState):
    problem: CompiledFiniteElementProblem
    scene: PreparedCollisionScene
    contact: PreparedConvergentContactPotential
    search: SweepAndPruneContactSearchPlan
    ccd: ContactCCDPlan
    inversion: SimplexInversionStepPlan | None
    solve_policy: ContactSolvePolicy
    subproblem: SteihaugToint
    plan_id: str = eqx.field(static=True)


class FiniteElementContactEquilibriumResult(StrictModule):
    previous: Array
    candidate: Array
    value: Array
    accepted: Array
    diagnostics: ContactSolveDiagnostics
    contact: ContactPotentialEvaluation
    safety: ContactSafetyEvidence
    inversion: InversionStepEvidence | None
    rejection_reasons: Array
    plan_id: str = eqx.field(static=True)


def _validate_common(
    problem: CompiledFiniteElementProblem,
    scene: PreparedCollisionScene,
    contact: PreparedConvergentContactPotential,
    search: SweepAndPruneContactSearchPlan,
    ccd: ContactCCDPlan,
    inversion: SimplexInversionStepPlan | None,
    /,
) -> ArraySpace:
    if not isinstance(problem, CompiledFiniteElementProblem):
        raise TypeError("problem must be CompiledFiniteElementProblem.")
    if not problem.potential_compatible or len(problem.form.field_names) != 1:
        raise ValueError(
            "Finite-element contact requires one field and a form compiled from "
            "variational.Functional."
        )
    if not isinstance(problem.state_space, ArraySpace):
        raise TypeError("Finite-element contact currently requires an ArraySpace state.")
    if not isinstance(scene, PreparedCollisionScene):
        raise TypeError("scene must be PreparedCollisionScene.")
    if not problem.state_space.compatible(scene.source_space):
        raise ValueError("Collision scene state space does not match the FE problem.")
    if (
        not isinstance(contact, PreparedConvergentContactPotential)
        or contact.scene.scene_id != scene.scene_id
    ):
        raise ValueError("Prepared contact potential must belong to the collision scene.")
    if not isinstance(search, SweepAndPruneContactSearchPlan):
        raise TypeError("search must be SweepAndPruneContactSearchPlan.")
    if search.activation_distance != contact.plan.activation_distance:
        raise ValueError("Search and contact activation distances must agree exactly.")
    if not isinstance(ccd, (InclusionCCDPlan, CertifiedAABBCCDPlan)):
        raise TypeError("ccd must be a concrete contact CCD plan.")
    if inversion is not None and not isinstance(inversion, SimplexInversionStepPlan):
        raise TypeError("inversion must be SimplexInversionStepPlan or None.")
    return problem.state_space


def prepare_finite_element_contact_dynamics(
    problem: CompiledFiniteElementProblem,
    initial_state: ContactDynamicsState,
    scene: PreparedCollisionScene,
    contact: PreparedConvergentContactPotential,
    search: SweepAndPruneContactSearchPlan,
    ccd: ContactCCDPlan,
    /,
    *,
    inversion: SimplexInversionStepPlan | None = None,
    friction: PreparedLaggedCoulombFriction | None = None,
    method: ImplicitNewmarkMethod | None = None,
    mass_coefficient: ArrayLike = 1.0,
    mass_policy: FiniteElementMassPolicy | None = None,
    solve_policy: ContactSolvePolicy | None = None,
    subproblem: SteihaugToint | None = None,
) -> FiniteElementContactDynamicsPlan:
    space = _validate_common(problem, scene, contact, search, ccd, inversion)
    if not isinstance(initial_state, ContactDynamicsState):
        raise TypeError("initial_state must be ContactDynamicsState.")
    for value in (
        initial_state.mechanics.displacement,
        initial_state.mechanics.velocity,
        initial_state.mechanics.acceleration,
    ):
        space.validate(value)
    if friction is not None and (
        not isinstance(friction, PreparedLaggedCoulombFriction)
        or friction.scene.scene_id != scene.scene_id
    ):
        raise ValueError("friction must be prepared for the contact collision scene.")
    method_ = ImplicitNewmarkMethod() if method is None else method
    mass_policy_ = FiniteElementMassPolicy() if mass_policy is None else mass_policy
    solve_policy_ = ContactSolvePolicy() if solve_policy is None else solve_policy
    subproblem_ = SteihaugToint() if subproblem is None else subproblem
    if not isinstance(method_, ImplicitNewmarkMethod):
        raise TypeError("method must be ImplicitNewmarkMethod or None.")
    if not isinstance(mass_policy_, FiniteElementMassPolicy):
        raise TypeError("mass_policy must be FiniteElementMassPolicy or None.")
    if not isinstance(solve_policy_, ContactSolvePolicy):
        raise TypeError("solve_policy must be ContactSolvePolicy or None.")
    if not isinstance(subproblem_, SteihaugToint):
        raise TypeError("subproblem must be SteihaugToint or None.")
    mass = jnp.asarray(mass_coefficient, dtype=space.dtype)
    if mass.shape != () or not bool(jnp.isfinite(mass)) or mass <= 0.0:
        raise ValueError("mass_coefficient must be one positive finite scalar.")
    mass_problem = problem._compile_unit_mass_problem(mass_policy_)
    identifier = canonical_fingerprint(
        {
            "kind": "finite-element-contact-dynamics-plan",
            "problem": problem.compilation_id,
            "scene": scene.scene_id,
            "contact": contact.prepared_id,
            "friction": None if friction is None else friction.prepared_id,
            "search": search.plan_id,
            "ccd": ccd.plan_id,
            "inversion": None if inversion is None else inversion.plan_id,
            "method": method_.method_id,
            "mass_policy": mass_policy_.kind,
            "mass": float(mass).hex(),
            "solve": solve_policy_.policy_id,
        }
    )
    return FiniteElementContactDynamicsPlan(
        problem,
        scene,
        contact,
        search,
        friction,
        ccd,
        inversion,
        method_,
        mass_policy_,
        mass_problem,
        mass,
        solve_policy_,
        subproblem_,
        identifier,
    )


def prepare_finite_element_contact_step(
    plan: FiniteElementContactDynamicsPlan,
    accepted: ContactDynamicsState,
    step_size: ArrayLike,
    /,
    *,
    args: Any = None,
) -> PreparedFiniteElementContactStep:
    if not isinstance(plan, FiniteElementContactDynamicsPlan):
        raise TypeError("plan must be FiniteElementContactDynamicsPlan.")
    if not isinstance(accepted, ContactDynamicsState):
        raise TypeError("accepted must be ContactDynamicsState.")
    dt = jnp.asarray(step_size, dtype=accepted.mechanics.displacement.dtype)
    if dt.shape != () or not bool(jnp.isfinite(dt)) or dt <= 0.0:
        raise ValueError("step_size must be one positive finite scalar.")
    return PreparedFiniteElementContactStep(
        plan,
        accepted,
        dt,
        args,
        canonical_fingerprint(
            {
                "kind": "prepared-finite-element-contact-step",
                "plan": plan.plan_id,
                "state": accepted.state_id,
                "step_size": float(dt).hex(),
            }
        ),
    )


def prepare_finite_element_contact_equilibrium(
    problem: CompiledFiniteElementProblem,
    scene: PreparedCollisionScene,
    contact: PreparedConvergentContactPotential,
    search: SweepAndPruneContactSearchPlan,
    ccd: ContactCCDPlan,
    /,
    *,
    inversion: SimplexInversionStepPlan | None = None,
    solve_policy: ContactSolvePolicy | None = None,
    subproblem: SteihaugToint | None = None,
) -> FiniteElementContactEquilibriumPlan:
    _validate_common(problem, scene, contact, search, ccd, inversion)
    solve_policy_ = ContactSolvePolicy() if solve_policy is None else solve_policy
    subproblem_ = SteihaugToint() if subproblem is None else subproblem
    if not isinstance(solve_policy_, ContactSolvePolicy) or not isinstance(
        subproblem_, SteihaugToint
    ):
        raise TypeError("solve_policy/subproblem have incompatible types.")
    identifier = canonical_fingerprint(
        {
            "kind": "finite-element-contact-equilibrium-plan",
            "problem": problem.compilation_id,
            "scene": scene.scene_id,
            "contact": contact.prepared_id,
            "search": search.plan_id,
            "ccd": ccd.plan_id,
            "inversion": None if inversion is None else inversion.plan_id,
            "solve": solve_policy_.policy_id,
        }
    )
    return FiniteElementContactEquilibriumPlan(
        problem,
        scene,
        contact,
        search,
        ccd,
        inversion,
        solve_policy_,
        subproblem_,
        identifier,
    )


def _array_norm(space: ArraySpace, value: Array, /) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.real(space.inner(value, value)), 0.0))


def _full_positions(
    problem: CompiledFiniteElementProblem, state: Array, args: Any, /
) -> Array:
    full = problem.expand(state, args)
    coordinates = problem._execution_context(args).runtime.coordinates
    if full.shape != coordinates.shape:
        raise ValueError(
            "Certified inversion stepping requires nodal displacement matching FE coordinates."
        )
    return coordinates + full


def _solve_contact_minimization(
    *,
    problem: CompiledFiniteElementProblem,
    scene: PreparedCollisionScene,
    contact: PreparedConvergentContactPotential,
    search: SweepAndPruneContactSearchPlan,
    ccd: ContactCCDPlan,
    inversion: SimplexInversionStepPlan | None,
    policy: ContactSolvePolicy,
    subproblem: SteihaugToint,
    initial: Array,
    args: Any,
    inertial_energy,
    extra_energy,
) -> tuple[
    Array,
    ContactCandidateEpoch,
    ContactSolveDiagnostics,
    ContactSafetyEvidence,
    InversionStepEvidence | None,
    ContactEnergyLedger,
    int,
]:
    space = problem.state_space
    if not isinstance(space, ArraySpace):
        raise TypeError("Contact minimization requires ArraySpace.")
    parameters = space.validate(initial)
    trust_radius = jnp.asarray(policy.initial_trust_radius, dtype=parameters.dtype)
    objective_evaluations = 0
    gradient_evaluations = 0
    hessian_actions = 0
    linear_iterations = 0
    line_evaluations = 0
    fallbacks = 0
    last_rate = jnp.asarray(0.0, dtype=parameters.dtype)
    last_step_norm = jnp.asarray(jnp.inf, dtype=parameters.dtype)
    last_safety: ContactSafetyEvidence | None = None
    last_inversion: InversionStepEvidence | None = None
    rejection = int(ContactRejectionReason.NONE)

    def build_epoch(value, end_value=None):
        positions = scene.positions(value)
        end_positions = None if end_value is None else scene.positions(end_value)
        return search.build(
            scene,
            np.asarray(positions),
            end_positions=None if end_positions is None else np.asarray(end_positions),
        )

    epoch = build_epoch(parameters)
    if not bool(epoch.successful):
        rejection |= int(ContactRejectionReason.SEARCH)

    def total_energy(value, current_epoch):
        return (
            inertial_energy(value)
            + problem.potential(value, args)
            + contact.energy(scene.positions(value), current_epoch)
            + extra_energy(value, current_epoch)
        )

    initial_energy = total_energy(parameters, epoch)
    objective_evaluations += 1
    initial_gradient_norm = jnp.asarray(jnp.inf, dtype=parameters.dtype)
    final_gradient_norm = initial_gradient_norm
    converged = False
    iteration_count = 0

    if rejection == int(ContactRejectionReason.NONE):
        for iteration in range(policy.maximum_iterations):
            iteration_count = iteration + 1
            objective = lambda value, current_epoch=epoch: total_energy(
                value, current_epoch
            )
            value, gradient = jax.value_and_grad(objective)(parameters)
            objective_evaluations += 1
            gradient_evaluations += 1
            gradient_norm = _array_norm(space, gradient)
            if iteration == 0:
                initial_gradient_norm = gradient_norm
            final_gradient_norm = gradient_norm
            threshold = (
                policy.absolute_gradient
                + policy.relative_gradient * initial_gradient_norm
            )
            if bool(
                jnp.isfinite(value)
                & jnp.isfinite(gradient_norm)
                & (gradient_norm <= threshold)
            ):
                converged = True
                break
            if not bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))):
                rejection |= int(ContactRejectionReason.NONFINITE)
                break
            _, hessian_action = jax.linearize(jax.grad(objective), parameters)
            hessian = FunctionLinearOperator(
                hessian_action,
                source=space,
                target=space,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
                operator_id=f"contact-objective-hessian/{epoch.epoch_id}",
                closure_convert=False,
            )
            trust = solve_trust_region_subproblem(
                TrustRegionQuadraticProblem(hessian, gradient, trust_radius),
                method=subproblem,
            )
            hessian_actions += int(trust.diagnostics.hessian_actions)
            linear_iterations += int(trust.diagnostics.iterations)
            direction = space.validate(trust.step)
            slope = jnp.real(space.inner(gradient, direction))
            if not bool(trust.successful & jnp.isfinite(slope) & (slope < 0.0)):
                direction = -gradient
                slope = -jnp.real(space.inner(gradient, gradient))
                fallbacks += 1
            proposed = parameters + direction
            swept_epoch = build_epoch(parameters, proposed)
            if not bool(swept_epoch.successful):
                rejection |= int(ContactRejectionReason.SEARCH)
                break
            start_surface = scene.positions(parameters)
            end_surface = scene.positions(proposed)
            safety = collision_free_step_limit(
                ccd,
                scene,
                swept_epoch,
                np.asarray(start_surface),
                np.asarray(end_surface),
            )
            last_safety = safety
            if not bool(safety.successful):
                rejection |= int(ContactRejectionReason.CCD)
                break
            safe_rate = safety.step_size
            if inversion is not None:
                start_full = _full_positions(problem, parameters, args)
                end_full = _full_positions(problem, proposed, args)
                inversion_evidence = simplex_inversion_step_limit(
                    inversion, np.asarray(start_full), np.asarray(end_full)
                )
                last_inversion = inversion_evidence
                if not bool(inversion_evidence.successful):
                    rejection |= int(ContactRejectionReason.INVERSION)
                    break
                safe_rate = jnp.minimum(safe_rate, inversion_evidence.step_size)
            rate = jnp.minimum(1.0, safe_rate)
            accepted = False
            accepted_epoch = epoch
            candidate = parameters
            for _ in range(policy.maximum_line_search_steps):
                line_evaluations += 1
                trial = parameters + rate * direction
                trial_epoch = build_epoch(trial)
                if not bool(trial_epoch.successful):
                    rejection |= int(ContactRejectionReason.SEARCH)
                    break
                trial_value = total_energy(trial, trial_epoch)
                objective_evaluations += 1
                armijo_bound = value + policy.armijo * rate * slope
                if bool(jnp.isfinite(trial_value) & (trial_value <= armijo_bound)):
                    accepted = True
                    candidate = trial
                    accepted_epoch = trial_epoch
                    break
                rate = policy.contraction * rate
            if rejection != int(ContactRejectionReason.NONE):
                break
            if not accepted:
                rejection |= int(ContactRejectionReason.LINE_SEARCH)
                break
            step = candidate - parameters
            last_step_norm = _array_norm(space, step)
            last_rate = rate
            parameters = candidate
            epoch = accepted_epoch
            if float(rate) >= 0.75 and bool(trust.diagnostics.boundary_hit):
                trust_radius = jnp.minimum(
                    policy.maximum_trust_radius, 2.0 * trust_radius
                )
            elif float(rate) < 0.25:
                trust_radius = jnp.maximum(
                    policy.minimum_trust_radius, 0.5 * trust_radius
                )
            if bool(last_step_norm <= policy.absolute_step):
                objective = lambda value, current_epoch=epoch: total_energy(
                    value, current_epoch
                )
                _, gradient = jax.value_and_grad(objective)(parameters)
                objective_evaluations += 1
                gradient_evaluations += 1
                final_gradient_norm = _array_norm(space, gradient)
                converged = bool(
                    final_gradient_norm
                    <= policy.absolute_gradient
                    + policy.relative_gradient * initial_gradient_norm
                )
                if not converged:
                    rejection |= int(ContactRejectionReason.NONLINEAR)
                break
        if not converged and rejection == int(ContactRejectionReason.NONE):
            rejection |= int(ContactRejectionReason.NONLINEAR)

    final_total = total_energy(parameters, epoch)
    objective_evaluations += 1
    bulk = problem.potential(parameters, args)
    contact_energy = contact.energy(scene.positions(parameters), epoch)
    inertia = inertial_energy(parameters)
    friction_energy = extra_energy(parameters, epoch)
    finite_energy = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    initial_energy,
                    final_total,
                    bulk,
                    contact_energy,
                    inertia,
                    friction_energy,
                )
            )
        )
    )
    ledger = ContactEnergyLedger(
        initial_energy,
        final_total,
        bulk,
        contact_energy,
        inertia,
        friction_energy,
        initial_energy - final_total,
        finite_energy,
    )
    if not bool(finite_energy):
        rejection |= int(ContactRejectionReason.ENERGY)
    diagnostics = ContactSolveDiagnostics(
        jnp.asarray(converged),
        jnp.asarray(iteration_count, dtype=jnp.int32),
        jnp.asarray(objective_evaluations, dtype=jnp.int32),
        jnp.asarray(gradient_evaluations, dtype=jnp.int32),
        jnp.asarray(hessian_actions, dtype=jnp.int32),
        jnp.asarray(linear_iterations, dtype=jnp.int32),
        jnp.asarray(line_evaluations, dtype=jnp.int32),
        jnp.asarray(fallbacks, dtype=jnp.int32),
        initial_gradient_norm,
        final_gradient_norm,
        last_step_norm,
        last_rate,
        trust_radius,
        jnp.asarray(rejection, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=parameters.dtype),
    )
    if last_safety is None:
        positions = scene.positions(parameters)
        static_epoch = search.build(
            scene, np.asarray(positions), end_positions=np.asarray(positions)
        )
        last_safety = collision_free_step_limit(
            ccd, scene, static_epoch, np.asarray(positions), np.asarray(positions)
        )
    return parameters, epoch, diagnostics, last_safety, last_inversion, ledger, rejection


def solve_finite_element_contact_step(
    prepared: PreparedFiniteElementContactStep,
    /,
) -> FiniteElementContactResult:
    if not isinstance(prepared, PreparedFiniteElementContactStep):
        raise TypeError("prepared must be PreparedFiniteElementContactStep.")
    plan = prepared.plan
    previous = prepared.accepted
    mechanics = previous.mechanics
    predictor = plan.method.predictor(mechanics, prepared.step_size)
    context = plan.problem._execution_context(prepared.args)
    _, reduced_mass = plan.problem._mass_operators(
        context,
        plan.mass_coefficient,
        plan.mass_policy,
        plan.mass_problem,
    )
    acceleration_scale = plan.method.position_to_acceleration_scale(
        prepared.step_size, mechanics.displacement.dtype
    )

    def inertial_energy(displacement):
        delta = displacement - predictor
        mass_delta = plan.problem.state_space.inverse_riesz(reduced_mass.mv(delta))
        return (
            0.5
            * acceleration_scale
            * jnp.real(plan.problem.state_space.inner(delta, mass_delta))
        )

    displacement = mechanics.displacement
    friction_state = previous.friction_state
    lag_iterations = 0
    lag_residual = jnp.asarray(0.0, dtype=displacement.dtype)
    friction_converged = plan.friction is None
    friction_evaluation = None
    maximum_lag_iterations = (
        1 if plan.friction is None else plan.friction.plan.maximum_lag_iterations
    )
    result = None
    for lag_iteration in range(maximum_lag_iterations):
        lag_iterations = lag_iteration + 1
        if plan.friction is not None and friction_state is None:
            initial_positions = plan.scene.positions(displacement)
            initial_epoch = plan.search.build(plan.scene, np.asarray(initial_positions))
            if not bool(initial_epoch.successful):
                raise ValueError(
                    "Initial friction lag state requires a complete contact epoch."
                )
            friction_state = plan.friction.build_state(
                initial_positions,
                initial_epoch,
                state_version=previous.state_version,
            )

        if plan.friction is None:
            extra_energy = lambda value, current_epoch: jnp.asarray(
                0.0, dtype=value.dtype
            )
        else:
            velocity_scale = plan.method.position_to_velocity_scale(
                prepared.step_size, displacement.dtype
            )
            active_friction_state = friction_state

            def extra_energy(
                value,
                current_epoch,
                lag_state=active_friction_state,
                scale=velocity_scale,
            ):
                del current_epoch
                velocity_value, _ = plan.method.rates(
                    value, mechanics, prepared.step_size
                )
                surface_velocity = plan.scene.map_values(velocity_value)
                return plan.friction.energy(surface_velocity, lag_state) / scale

        result = _solve_contact_minimization(
            problem=plan.problem,
            scene=plan.scene,
            contact=plan.contact,
            search=plan.search,
            ccd=plan.ccd,
            inversion=plan.inversion,
            policy=plan.solve_policy,
            subproblem=plan.subproblem,
            initial=displacement,
            args=prepared.args,
            inertial_energy=inertial_energy,
            extra_energy=extra_energy,
        )
        (
            displacement,
            epoch,
            diagnostics,
            safety,
            inversion,
            ledger,
            rejection,
        ) = result
        if rejection != int(ContactRejectionReason.NONE):
            break
        if plan.friction is None:
            friction_converged = True
            break
        velocity_value, _ = plan.method.rates(displacement, mechanics, prepared.step_size)
        candidate_friction_state = plan.friction.build_state(
            plan.scene.positions(displacement),
            epoch,
            state_version=previous.state_version + 1,
        )
        lag_residual = plan.friction.lag_residual(
            friction_state, candidate_friction_state
        )
        friction_evaluation = plan.friction.evaluate(
            plan.scene.map_values(velocity_value), friction_state
        )
        if bool(lag_residual <= plan.friction.plan.lag_tolerance):
            friction_state = candidate_friction_state
            friction_converged = True
            break
        friction_state = candidate_friction_state
    if result is None:
        raise RuntimeError("Contact lag solve did not execute.")
    diagnostics = eqx.tree_at(
        lambda value: value.lag_iterations,
        diagnostics,
        jnp.asarray(lag_iterations, dtype=jnp.int32),
    )
    diagnostics = eqx.tree_at(
        lambda value: value.lag_residual,
        diagnostics,
        lag_residual,
    )
    if not friction_converged:
        rejection |= int(ContactRejectionReason.FRICTION)
    velocity, acceleration = plan.method.rates(
        displacement, mechanics, prepared.step_size
    )
    candidate_mechanics = FiniteElementDynamicsState(
        displacement,
        velocity,
        acceleration,
        time=mechanics.time + prepared.step_size,
        step=mechanics.step + 1,
        state_version=mechanics.state_version + 1,
        materials=mechanics.materials,
    )
    candidate = ContactDynamicsState(
        candidate_mechanics,
        replay_epoch=epoch,
        friction_state=friction_state,
        state_version=previous.state_version + 1,
    )
    contact_evaluation = plan.contact.evaluate(plan.scene.positions(displacement), epoch)
    if not bool(contact_evaluation.successful):
        rejection |= int(ContactRejectionReason.CONTACT)
    accepted = diagnostics.converged & (rejection == int(ContactRejectionReason.NONE))
    accepted_state = candidate if bool(accepted) else previous
    return FiniteElementContactResult(
        previous,
        candidate,
        accepted_state,
        accepted,
        ~accepted,
        diagnostics,
        contact_evaluation,
        friction_evaluation,
        safety,
        inversion,
        ledger,
        jnp.asarray(rejection, dtype=jnp.int32),
        plan.plan_id,
    )


def solve_finite_element_contact_equilibrium(
    plan: FiniteElementContactEquilibriumPlan,
    initial_displacement: ArrayLike,
    /,
    *,
    args: Any = None,
) -> FiniteElementContactEquilibriumResult:
    if not isinstance(plan, FiniteElementContactEquilibriumPlan):
        raise TypeError("plan must be FiniteElementContactEquilibriumPlan.")
    initial = plan.problem.state_space.validate(initial_displacement)
    zero_inertial = lambda value: jnp.asarray(0.0, dtype=value.dtype)
    zero_extra = lambda value, current_epoch: jnp.asarray(0.0, dtype=value.dtype)
    result = _solve_contact_minimization(
        problem=plan.problem,
        scene=plan.scene,
        contact=plan.contact,
        search=plan.search,
        ccd=plan.ccd,
        inversion=plan.inversion,
        policy=plan.solve_policy,
        subproblem=plan.subproblem,
        initial=initial,
        args=args,
        inertial_energy=zero_inertial,
        extra_energy=zero_extra,
    )
    displacement, epoch, diagnostics, safety, inversion, _, rejection = result
    contact_evaluation = plan.contact.evaluate(plan.scene.positions(displacement), epoch)
    if not bool(contact_evaluation.successful):
        rejection |= int(ContactRejectionReason.CONTACT)
    accepted = diagnostics.converged & (rejection == int(ContactRejectionReason.NONE))
    value = plan.problem.potential(displacement, args) + contact_evaluation.energy
    return FiniteElementContactEquilibriumResult(
        initial,
        displacement,
        value,
        accepted,
        diagnostics,
        contact_evaluation,
        safety,
        inversion,
        jnp.asarray(rejection, dtype=jnp.int32),
        plan.plan_id,
    )


__all__ = [
    "ContactDynamicsState",
    "ContactEnergyLedger",
    "ContactRejectionReason",
    "ContactSolveDiagnostics",
    "ContactSolvePolicy",
    "FiniteElementContactDynamicsPlan",
    "FiniteElementContactEquilibriumPlan",
    "FiniteElementContactEquilibriumResult",
    "FiniteElementContactResult",
    "PreparedFiniteElementContactStep",
    "prepare_finite_element_contact_dynamics",
    "prepare_finite_element_contact_equilibrium",
    "prepare_finite_element_contact_step",
    "solve_finite_element_contact_equilibrium",
    "solve_finite_element_contact_step",
]
