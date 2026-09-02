#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import ParticlePopulationState
from ..discretization.pic import (
    PICChargeModelPlan,
    PICChargeState,
    PICParticleResponsePlan,
    PICParticleState,
    PreparedPICParticleCochainTransfer,
)
from ..linalg import (
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)
from ._maxwell import (
    CompatibleMaxwellState,
    MaxwellPrimaryState,
    PreparedCompatibleMaxwell,
)


class _SemiImplicitFieldAction(StrictModule):
    response: PICParticleResponsePlan
    response_state: object
    cochain: object
    theta_dt: Array

    def __call__(self, electric: Array, /) -> Array:
        response = self.response.apply(self.response_state, electric).current
        curl = self.cochain.exterior_derivative(1, electric)
        curl_curl = self.cochain.codifferential(2, curl)
        return electric + self.theta_dt * response + self.theta_dt**2 * curl_curl


class SemiImplicitPICState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    charge: PICChargeState
    maxwell: CompatibleMaxwellState
    time: Array


class SemiImplicitPICDiagnostics(StrictModule):
    linear_residual: Array
    linear_iterations: Array
    energy_defect: Array
    gauss_defect: Array
    magnetic_defect: Array
    response_finite: Array
    finite: Array
    successful: Array


class SemiImplicitPICResult(StrictModule):
    candidate_state: SemiImplicitPICState
    accepted_state: SemiImplicitPICState
    diagnostics: SemiImplicitPICDiagnostics
    successful: Array
    plan_id: str = eqx.field(static=True)


class SemiImplicitPICPlan(StrictModule, NonTrainableState):
    """Periodic nonrelativistic ECSIM response with a bounded GMRES field solve."""

    maxwell: PreparedCompatibleMaxwell
    transfer: PreparedPICParticleCochainTransfer
    response: PICParticleResponsePlan
    charge_model: PICChargeModelPlan
    theta: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maxwell: PreparedCompatibleMaxwell,
        transfer: PreparedPICParticleCochainTransfer,
        charge_model: PICChargeModelPlan,
        /,
        *,
        theta: float = 0.5,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ):
        if not isinstance(maxwell, PreparedCompatibleMaxwell):
            raise TypeError("maxwell must be PreparedCompatibleMaxwell.")
        if not isinstance(transfer, PreparedPICParticleCochainTransfer):
            raise TypeError("transfer must be a prepared PIC transfer.")
        if transfer.bridge.bridge_id != maxwell.plan.bridge.bridge_id:
            raise ValueError("Semi-implicit PIC transfer and Maxwell bridge differ.")
        theta_ = float(theta)
        if theta_ < 0.5 or theta_ > 1.0:
            raise ValueError("theta must lie in [0.5,1].")
        if (
            maxwell.pml is not None
            or maxwell.boundaries
            or maxwell.capabilities.dispersive
            or maxwell.capabilities.nonlinear
        ):
            raise ValueError(
                "Semi-implicit PIC initially requires periodic instantaneous Maxwell material."
            )
        self.maxwell = maxwell
        self.transfer = transfer
        self.response = PICParticleResponsePlan(transfer)
        self.charge_model = charge_model
        self.theta = theta_
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semi-implicit-pic",
                "maxwell": maxwell.prepared_id,
                "transfer": transfer.prepared_id,
                "charge_model": charge_model.plan_id,
                "theta": theta_,
                "tolerance": float(tolerance),
                "maximum_iterations": int(maximum_iterations),
            }
        )

    def step(
        self,
        state: SemiImplicitPICState,
        step_size: ArrayLike,
        /,
    ) -> SemiImplicitPICResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        velocity = state.particles.proper_velocity
        midpoint = state.particles.position + 0.5 * dt * velocity
        macrocharge = self.charge_model.macrocharge(state.population, state.charge)
        magnetic = self.maxwell.magnetic_field(state.maxwell)
        response_state = self.response.prepare_state(
            midpoint,
            velocity,
            magnetic,
            macrocharge,
            state.population.mass,
            state.population.active,
            dt,
        )
        known = self.response.known_current(response_state, macrocharge)
        cochain = self.maxwell.plan.bridge.cochain
        theta_dt = self.theta * dt
        action = _SemiImplicitFieldAction(
            self.response, response_state, cochain, theta_dt
        )
        electric_space = cochain.space(1).vector_space
        operator = FunctionLinearOperator(
            action,
            source=electric_space,
            target=electric_space,
            properties=OperatorProperties(
                self_adjoint=False,
                positive_definite=False,
                evidence={},
            ),
            operator_id=canonical_fingerprint(
                {"kind": "semi-implicit-pic-step-operator", "plan": self.plan_id}
            ),
        )
        policy = LinearSolvePolicy(
            GMRES(restart=min(50, self.maximum_iterations)),
            tolerance=TolerancePolicy(
                relative=self.tolerance,
                absolute=self.tolerance,
                max_steps=self.maximum_iterations,
            ),
        )
        prepared = prepare(LinearSystem(operator), policy)
        displacement = state.maxwell.primary.electric_displacement
        magnetic_flux = state.maxwell.primary.magnetic_flux
        rhs = (
            displacement
            + theta_dt * cochain.codifferential(2, magnetic_flux)
            - theta_dt * known.current
        )
        electric_old = self.maxwell.electric_field(state.maxwell)
        linear = solve(prepared, rhs, initial_guess=electric_old)
        electric_theta = linear.value
        response_current = self.response.apply(response_state, electric_theta)
        total_current = known.current + response_current.current
        magnetic_new = magnetic_flux - dt * cochain.exterior_derivative(1, electric_theta)
        magnetic_theta = (1.0 - self.theta) * magnetic_flux + self.theta * magnetic_new
        displacement_new = displacement + dt * (
            cochain.codifferential(2, magnetic_theta) - total_current
        )
        charge_new = state.maxwell.primary.charge - dt * cochain.codifferential(
            1, total_current
        )
        maxwell_candidate = CompatibleMaxwellState(
            MaxwellPrimaryState(displacement_new, magnetic_new, charge_new),
            state.maxwell.auxiliary,
            state.maxwell.observations,
        )
        gathered = self.transfer.gather_electric(
            response_state.routes, electric_theta
        ).values
        mean_velocity = (
            response_state.rotated_velocity
            + contract(
                "pij,pj->pi",
                response_state.alpha,
                gathered,
            )
            * (macrocharge * dt / (2.0 * jnp.maximum(state.population.mass, 1.0e-30)))[
                :, None
            ]
        )
        velocity_new = 2.0 * mean_velocity - velocity
        position_new = state.particles.position + dt * mean_velocity
        candidate = SemiImplicitPICState(
            PICParticleState(position_new, velocity_new),
            state.population,
            state.charge,
            maxwell_candidate,
            state.time + dt,
        )
        linear_residual = jnp.sqrt(jnp.sum((action(electric_theta) - rhs) ** 2))
        electric_constraint = self.maxwell.electric_constraint(maxwell_candidate)
        magnetic_constraint = self.maxwell.magnetic_constraint(maxwell_candidate)
        gauss = jnp.max(jnp.abs(electric_constraint), initial=0.0)
        magnetic_defect = jnp.max(jnp.abs(magnetic_constraint), initial=0.0)
        old_energy = self.maxwell.energy(state.maxwell) + 0.5 * jnp.sum(
            state.population.mass * jnp.sum(velocity**2, axis=-1)
        )
        new_energy = self.maxwell.energy(maxwell_candidate) + 0.5 * jnp.sum(
            state.population.mass * jnp.sum(velocity_new**2, axis=-1)
        )
        energy_defect = new_energy - old_energy
        finite = (
            jnp.all(jnp.isfinite(position_new))
            & jnp.all(jnp.isfinite(velocity_new))
            & jnp.isfinite(linear_residual + energy_defect + gauss + magnetic_defect)
        )
        successful = (
            linear.successful
            & known.successful
            & response_current.successful
            & finite
            & (
                linear_residual
                <= self.tolerance * jnp.maximum(1.0, jnp.sqrt(jnp.sum(rhs**2)))
            )
            & (gauss <= 10.0 * self.tolerance)
            & (magnetic_defect <= 10.0 * self.tolerance)
        )
        accepted = jax.tree.map(
            lambda proposed, old: jnp.where(successful, proposed, old), candidate, state
        )
        diagnostics = SemiImplicitPICDiagnostics(
            linear_residual,
            linear.diagnostics.iterations,
            energy_defect,
            gauss,
            magnetic_defect,
            known.finite & response_current.finite,
            finite,
            successful,
        )
        return SemiImplicitPICResult(
            candidate, accepted, diagnostics, successful, self.plan_id
        )


class PICGaussCorrectionResult(StrictModule):
    position: Array
    charge: Array
    residual: Array
    residual_norm: Array
    displacement_norm: Array
    converged: Array
    successful: Array


class PICGaussCorrectionPlan(StrictModule, NonTrainableState):
    transfer: PreparedPICParticleCochainTransfer
    iterations: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedPICParticleCochainTransfer,
        /,
        *,
        iterations: int = 4,
        learning_rate: float = 1.0e-3,
        tolerance: float = 1.0e-8,
    ):
        self.transfer = transfer
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)

    def correct(
        self,
        position: ArrayLike,
        macrocharge: ArrayLike,
        target_charge: ArrayLike,
        active_mask: ArrayLike,
        /,
    ) -> PICGaussCorrectionResult:
        initial = jnp.asarray(position)
        target = jnp.asarray(target_charge)
        active = jnp.asarray(active_mask, dtype=bool)

        def objective(value):
            routes = self.transfer.build(value, active_mask=active)
            deposited = self.transfer.deposit_macrocharge(routes, macrocharge)
            residual = deposited.cochain - target
            return 0.5 * jnp.sum(residual**2)

        value = initial
        for _ in range(self.iterations):
            gradient = jax.grad(objective)(value)
            value = jnp.where(
                active[:, None], value - self.learning_rate * gradient, value
            )
        routes = self.transfer.build(value, active_mask=active)
        charge = self.transfer.deposit_macrocharge(routes, macrocharge).cochain
        residual = charge - target
        norm = jnp.sqrt(jnp.sum(residual**2))
        displacement = jnp.sqrt(jnp.sum((value - initial) ** 2))
        finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(norm + displacement)
        converged = norm <= self.tolerance * jnp.maximum(
            1.0, jnp.sqrt(jnp.sum(target**2))
        )
        return PICGaussCorrectionResult(
            value, charge, residual, norm, displacement, converged, finite & converged
        )


__all__ = [
    "PICGaussCorrectionPlan",
    "PICGaussCorrectionResult",
    "SemiImplicitPICDiagnostics",
    "SemiImplicitPICPlan",
    "SemiImplicitPICResult",
    "SemiImplicitPICState",
]
