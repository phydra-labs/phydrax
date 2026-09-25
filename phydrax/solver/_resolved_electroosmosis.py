#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
)
from .._differentiation import DerivativeContract, DerivativeRoute, DerivativeSurface
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_momentum import PreparedMACMomentumOperators
from ._electrode_reaction import (
    MACReactiveElectrodeBinding,
    MACReactiveElectrodeEvaluation,
    ReactiveElectrodeState,
)
from ._electrohydrodynamic import (
    MACElectrohydrodynamicEvaluation,
    MACElectrohydrodynamicForcePlan,
)
from ._mac_poisson_nernst_planck import (
    MACPoissonNernstPlanckEvaluation,
    MACPoissonNernstPlanckPlan,
)
from ._structured_incompressible import (
    MACPressureProjectionPlan,
    MACPressureProjectionResult,
)


class ResolvedElectroosmoticState(StrictModule):
    concentrations: Array
    potential: Array
    velocity: FaceVelocity
    pressure: Array
    electrode: ReactiveElectrodeState | None
    time: Array
    accepted_steps: Array
    runtime_id: str = eqx.field(static=True)


class ResolvedElectroosmoticLedger(StrictModule):
    species_content_defect: Array
    charge_content_defect: Array
    divergence_norm: Array
    ionic_free_energy_change: Array
    kinetic_energy_change: Array
    viscous_dissipation: Array
    electrohydrodynamic_work: Array
    energy_balance_defect: Array
    finite: Array
    conservative: Array


# Implicit derivatives of the fixed model with its regime decisions frozen.
_DERIVATIVE_CONTRACT = DerivativeContract.smooth(
    (DerivativeSurface.PRIMAL_STATE, DerivativeSurface.PHYSICAL_PARAMETER),
    route=DerivativeRoute.IMPLICIT,
    conditions=("decisions-frozen",),
)


class ResolvedElectroosmoticStepResult(StrictModule):
    candidate: ResolvedElectroosmoticState
    accepted: ResolvedElectroosmoticState
    pnp: MACPoissonNernstPlanckEvaluation
    force: MACElectrohydrodynamicEvaluation
    projection: MACPressureProjectionResult
    electrode: MACReactiveElectrodeEvaluation | None
    ledger: ResolvedElectroosmoticLedger
    fixed_point_residual: Array
    iteration_count: Array
    header: AdmissibilityHeader
    derivative_contract: DerivativeContract
    successful: Array
    plan_id: str = eqx.field(static=True)


class ResolvedElectroosmoticStokesPlan(StrictModule, NonTrainableState):
    """Monolithic fixed-point PNP and quasi-steady MAC Stokes candidate solve."""

    pnp: MACPoissonNernstPlanckPlan
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    force: MACElectrohydrodynamicForcePlan
    electrode_binding: MACReactiveElectrodeBinding | None
    electrode_potential: float = eqx.field(static=True)
    electrode_temperature: float = eqx.field(static=True)
    electrode_pressure: float = eqx.field(static=True)
    density: float = eqx.field(static=True)
    kinematic_viscosity: float = eqx.field(static=True)
    hydrodynamic_relaxation: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pnp: MACPoissonNernstPlanckPlan,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        /,
        *,
        density: float,
        kinematic_viscosity: float,
        hydrodynamic_relaxation: float,
        maximum_iterations: int = 20,
        tolerance: float = 1.0e-8,
        conservation_tolerance: float = 1.0e-10,
        electrode_binding: MACReactiveElectrodeBinding | None = None,
        electrode_potential: float = 0.0,
        electrode_temperature: float = 300.0,
        electrode_pressure: float = 101325.0,
    ) -> None:
        density_ = float(density)
        viscosity = float(kinematic_viscosity)
        relaxation = float(hydrodynamic_relaxation)
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        conservation = float(conservation_tolerance)
        electrode_values = tuple(
            float(value)
            for value in (
                electrode_potential,
                electrode_temperature,
                electrode_pressure,
            )
        )
        operators_id = pnp.electrostatic.operators.prepared_id
        if (
            not isinstance(pnp, MACPoissonNernstPlanckPlan)
            or not isinstance(momentum, PreparedMACMomentumOperators)
            or not isinstance(projection, MACPressureProjectionPlan)
            or momentum.operators.prepared_id != operators_id
            or projection.operators.prepared_id != operators_id
            or momentum.boundaries.prepared_id != projection.boundaries.prepared_id
            or projection.density != density_
            or (
                electrode_binding is not None
                and (
                    not isinstance(electrode_binding, MACReactiveElectrodeBinding)
                    or electrode_binding.operators.prepared_id != operators_id
                    or electrode_binding.pnp_species_count
                    != pnp.parameters.schema.species_count
                )
            )
            or any(not np.isfinite(value) for value in electrode_values)
            or electrode_values[1] <= 0.0
            or electrode_values[2] <= 0.0
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in (
                    density_,
                    viscosity,
                    relaxation,
                    tolerance_,
                    conservation,
                )
            )
            or iterations <= 0
        ):
            raise ValueError("Resolved electroosmotic plans are incompatible.")
        self.pnp = pnp
        self.momentum = momentum
        self.projection = projection
        self.force = MACElectrohydrodynamicForcePlan(momentum.operators)
        self.electrode_binding = electrode_binding
        (
            self.electrode_potential,
            self.electrode_temperature,
            self.electrode_pressure,
        ) = electrode_values
        self.density = density_
        self.kinematic_viscosity = viscosity
        self.hydrodynamic_relaxation = relaxation
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.conservation_tolerance = conservation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resolved-electroosmotic-stokes",
                "pnp": pnp.plan_id,
                "momentum": momentum.prepared_id,
                "projection": projection.plan_id,
                "density": density_,
                "kinematic_viscosity": viscosity,
                "electrode_binding": (
                    None if electrode_binding is None else electrode_binding.plan_id
                ),
                "electrode_conditions": electrode_values,
                "hydrodynamic_relaxation": relaxation,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "conservation_tolerance": conservation,
            }
        )

    def _runtime_id(self, /) -> str:
        return canonical_fingerprint(
            {"kind": "resolved-electroosmotic-runtime", "plan": self.plan_id}
        )

    def initialize(
        self,
        concentrations: ArrayLike,
        velocity: FaceVelocity,
        /,
        *,
        initial_potential: ArrayLike | None = None,
        boundary_args: Any = None,
        electrode_state: ReactiveElectrodeState | None = None,
    ) -> ResolvedElectroosmoticState:
        concentration = jnp.asarray(concentrations)
        stage = self.momentum.boundaries.evaluate(jnp.asarray(0.0), boundary_args)
        velocity_ = self.momentum.boundaries.enforce(
            self.momentum.operators.validate_velocity(velocity), stage
        )
        projected = self.projection.project(
            velocity_, jnp.asarray(1.0, dtype=concentration.dtype), boundary_stage=stage
        )
        pnp = self.pnp.evaluate(
            concentration,
            face_velocity=projected.velocity,
            initial_potential=initial_potential,
        )
        successful = stage.successful & projected.converged & pnp.header.globally_eligible
        if (self.electrode_binding is None) != (electrode_state is None):
            raise ValueError(
                "electrode_state is required exactly when an electrode binding is configured."
            )
        if self.electrode_binding is not None:
            initial_electrode = self.electrode_binding.evaluate(
                concentration,
                electrode_state,
                pnp.electrostatic.potential,
                self.electrode_potential,
                self.electrode_temperature,
                self.electrode_pressure,
            )
            successful = successful & initial_electrode.header.globally_eligible
        if not bool(np.asarray(successful)):
            raise ValueError("Initial electroosmotic state failed physical admission.")
        return ResolvedElectroosmoticState(
            concentration,
            pnp.electrostatic.potential,
            projected.velocity,
            projected.pressure,
            electrode_state,
            jnp.asarray(0.0, dtype=concentration.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self._runtime_id(),
        )

    def advance(
        self,
        state: ResolvedElectroosmoticState,
        step_size: ArrayLike,
        boundary_args: Any = None,
        /,
    ) -> ResolvedElectroosmoticStepResult:
        if state.runtime_id != self._runtime_id():
            raise ValueError(
                "Resolved electroosmotic state does not belong to this plan."
            )
        step = jnp.asarray(step_size, dtype=state.concentrations.dtype)
        if step.shape != ():
            raise ValueError("Electroosmotic step size must be scalar.")
        stage = self.momentum.boundaries.evaluate(state.time + step, boundary_args)
        concentration = state.concentrations
        velocity = state.velocity
        potential = state.potential
        pressure = state.pressure
        electrode_state = state.electrode
        fixed_point_residual = jnp.asarray(jnp.inf, dtype=step.dtype)
        final_pnp = self.pnp.evaluate(
            concentration, face_velocity=velocity, initial_potential=potential
        )
        final_force = self.force.evaluate(final_pnp, face_velocity=velocity)
        final_projection = self.projection.project(
            velocity, self.hydrodynamic_relaxation, boundary_stage=stage
        )
        final_electrode = None

        for _ in range(self.maximum_iterations):
            pnp_iterate = self.pnp.evaluate(
                concentration,
                face_velocity=velocity,
                initial_potential=potential,
            )
            if self.electrode_binding is None:
                electrode_evaluation = None
                electrode_rate = jnp.zeros_like(concentration)
                electrode_candidate = None
                electrode_residual = jnp.asarray(0.0, dtype=step.dtype)
            else:
                electrode_evaluation = self.electrode_binding.evaluate(
                    concentration,
                    electrode_state,
                    pnp_iterate.electrostatic.potential,
                    self.electrode_potential,
                    self.electrode_temperature,
                    self.electrode_pressure,
                )
                electrode_rate = electrode_evaluation.concentration_rate
                electrode_plan = self.electrode_binding.electrode
                candidate_amount = (
                    state.electrode.surface_amount
                    + step * electrode_evaluation.electrode.surface_amount_rate
                )
                candidate_charge = (
                    state.electrode.surface_charge
                    - step * electrode_evaluation.electrode.faradaic_current
                )
                candidate_stern = candidate_charge / (
                    electrode_plan.capacitance_per_area * electrode_plan.face_measures
                )
                electrode_candidate = ReactiveElectrodeState(
                    candidate_amount,
                    candidate_charge,
                    candidate_stern,
                    state.electrode.state_id,
                )
                electrode_residual = jnp.maximum(
                    jnp.max(
                        jnp.abs(
                            electrode_candidate.surface_amount
                            - electrode_state.surface_amount
                        )
                    ),
                    jnp.max(
                        jnp.abs(
                            electrode_candidate.surface_charge
                            - electrode_state.surface_charge
                        )
                    ),
                )
            concentration_candidate = state.concentrations + step * (
                pnp_iterate.concentration_rate + electrode_rate
            )
            pnp_candidate = self.pnp.evaluate(
                concentration_candidate,
                face_velocity=velocity,
                initial_potential=pnp_iterate.electrostatic.potential,
            )
            force_candidate = self.force.evaluate(pnp_candidate, face_velocity=velocity)
            diffusion = self.momentum.laplacian(velocity, stage=stage)
            tentative = tuple(
                component
                + self.hydrodynamic_relaxation
                * (self.kinematic_viscosity * viscous + forcing / self.density)
                for component, viscous, forcing in zip(
                    velocity,
                    diffusion,
                    force_candidate.total_force,
                    strict=True,
                )
            )
            projected = self.projection.project(
                tentative,
                self.hydrodynamic_relaxation,
                boundary_stage=stage,
            )
            concentration_residual = jnp.max(
                jnp.abs(concentration_candidate - concentration)
            )
            velocity_residual = jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(jnp.abs(new - old))
                        for new, old in zip(projected.velocity, velocity, strict=True)
                    )
                )
            )
            fixed_point_residual = jnp.maximum(
                jnp.maximum(concentration_residual, velocity_residual),
                electrode_residual,
            )
            concentration = concentration_candidate
            velocity = projected.velocity
            potential = pnp_candidate.electrostatic.potential
            pressure = projected.pressure
            electrode_state = electrode_candidate
            final_pnp = pnp_candidate
            final_force = force_candidate
            final_projection = projected
            final_electrode = electrode_evaluation

        volumes = self.momentum.operators.discretization.cell_volumes.astype(step.dtype)
        spatial_axes = tuple(range(len(volumes.shape)))
        species_change = jnp.sum(
            volumes[..., None] * (concentration - state.concentrations),
            axis=spatial_axes,
        )
        if final_electrode is None:
            electrode_species_change = jnp.zeros_like(species_change)
            electrode_ok = jnp.asarray(True)
        else:
            electrode_species_change = step * jnp.sum(
                volumes[..., None] * final_electrode.concentration_rate,
                axis=spatial_axes,
            )
            electrode_ok = (
                final_electrode.header.globally_eligible
                & jnp.all(electrode_state.surface_amount >= 0.0)
                & jnp.all(jnp.isfinite(electrode_state.surface_charge))
            )
        species_defect = species_change - electrode_species_change
        charge_defect = contract(
            "s,s->",
            species_defect,
            self.pnp.parameters.schema.charges,
            backend="jax",
        )
        divergence = self.momentum.operators.divergence(velocity)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        before_pnp = self.pnp.evaluate(
            state.concentrations,
            face_velocity=state.velocity,
            initial_potential=state.potential,
        )
        ionic_change = final_pnp.total_free_energy - before_pnp.total_free_energy
        velocity_space = self.momentum.operators.velocity_space
        kinetic_before = (
            0.5 * self.density * velocity_space.inner(state.velocity, state.velocity)
        )
        kinetic_after = 0.5 * self.density * velocity_space.inner(velocity, velocity)
        kinetic_change = kinetic_after - kinetic_before
        diffusion = self.momentum.laplacian(velocity, stage=stage)
        viscous_dissipation = (
            -step
            * self.density
            * self.kinematic_viscosity
            * velocity_space.inner(velocity, diffusion)
        )
        ehd_work = step * final_force.fluid_power
        energy_defect = ionic_change + kinetic_change + viscous_dissipation - ehd_work
        content_scale = jnp.maximum(
            jnp.sum(
                volumes[..., None] * jnp.abs(state.concentrations), axis=spatial_axes
            ),
            1.0,
        )
        conservative = jnp.all(
            jnp.abs(species_defect) <= self.conservation_tolerance * content_scale
        )
        finite = (
            jnp.all(jnp.isfinite(concentration))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in velocity))
            )
            & jnp.isfinite(fixed_point_residual)
            & jnp.isfinite(energy_defect)
            & electrode_ok
        )
        converged = fixed_point_residual <= self.tolerance
        supported = (
            finite
            & (step > 0.0)
            & stage.successful
            & final_pnp.header.globally_eligible
            & final_force.header.globally_eligible
            & final_projection.converged
            & jnp.all(concentration > 0.0)
            & conservative
            & converged
        )
        ledger = ResolvedElectroosmoticLedger(
            species_defect,
            charge_defect,
            divergence_norm,
            ionic_change,
            kinetic_change,
            viscous_dissipation,
            ehd_work,
            energy_defect,
            finite,
            conservative,
        )
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            supported,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, self.tolerance - fixed_point_residual, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "resolved-electroosmotic-evidence", "plan": self.plan_id}
            ),
        )
        successful = supported & header.globally_eligible
        candidate = ResolvedElectroosmoticState(
            concentration,
            potential,
            velocity,
            pressure,
            electrode_state,
            state.time + step,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
            state.runtime_id,
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate,
            state,
        )
        return ResolvedElectroosmoticStepResult(
            candidate,
            accepted,
            final_pnp,
            final_force,
            final_projection,
            final_electrode,
            ledger,
            fixed_point_residual,
            jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            header,
            _DERIVATIVE_CONTRACT,
            successful,
            self.plan_id,
        )


__all__ = [
    "ResolvedElectroosmoticLedger",
    "ResolvedElectroosmoticState",
    "ResolvedElectroosmoticStepResult",
    "ResolvedElectroosmoticStokesPlan",
]
