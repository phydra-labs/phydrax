#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Spatial electrohydrodynamic–viscoelastic interface workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...electrohydrodynamics import (
    CoupledElectrohydrodynamicSolver,
    ElectrohydrodynamicState,
    maxwell_stress,
)
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ...rheology import SpatialConformationSolver


@dataclass(frozen=True, slots=True)
class SpatialElectroViscoelasticState:
    electrohydrodynamic: ElectrohydrodynamicState
    conformation_minus: Array
    conformation_plus: Array
    surface_charge_c_m2: Array


@dataclass(frozen=True, slots=True)
class SpatialElectroViscoelasticStep:
    state: SpatialElectroViscoelasticState
    interface_traction_pa: Array
    surface_charge_balance_residual_c: Array
    polymer_free_energy_j: Array
    polymer_force_n_m3: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialElectroViscoelasticWorkflow:
    ehd: CoupledElectrohydrodynamicSolver
    minus_rheology: SpatialConformationSolver
    plus_rheology: SpatialConformationSolver
    minus_phase_fraction: Array
    stress_divergence_operator_m_inv: Array
    velocity_gradient_operator_m_inv: Array
    interface_minus_cells: Array
    interface_plus_cells: Array
    interface_normals: Array
    interface_areas_m2: Array
    surface_charge_generator_s_inv: Array

    @classmethod
    def create(
        cls,
        ehd: CoupledElectrohydrodynamicSolver,
        minus_rheology: SpatialConformationSolver,
        plus_rheology: SpatialConformationSolver,
        minus_phase_fraction: ArrayLike,
        stress_divergence_operator_m_inv: ArrayLike,
        velocity_gradient_operator_m_inv: ArrayLike,
        interface_minus_cells: ArrayLike,
        interface_plus_cells: ArrayLike,
        interface_normals: ArrayLike,
        interface_areas_m2: ArrayLike,
        surface_charge_generator_s_inv: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> SpatialElectroViscoelasticWorkflow:
        cells = ehd.cell_volumes_m3.size
        dimension = ehd.spatial_dimension
        fraction = np.asarray(minus_phase_fraction, dtype=float)
        stress_divergence = np.asarray(stress_divergence_operator_m_inv, dtype=float)
        gradient = np.asarray(velocity_gradient_operator_m_inv, dtype=float)
        minus_cells = np.asarray(interface_minus_cells, dtype=np.int32)
        plus_cells = np.asarray(interface_plus_cells, dtype=np.int32)
        normals = np.asarray(interface_normals, dtype=float)
        areas = np.asarray(interface_areas_m2, dtype=float)
        surface_generator = np.asarray(surface_charge_generator_s_inv, dtype=float)
        surfaces = areas.size
        if fraction.shape != (cells,) or np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("Electroviscoelastic phase fractions must lie in [0, 1].")
        if stress_divergence.shape != (cells * dimension, cells * dimension * dimension):
            raise ValueError("Polymer-stress divergence operator has incompatible shape.")
        if gradient.shape != (cells * dimension * dimension, cells * dimension):
            raise ValueError("Velocity-gradient operator has incompatible shape.")
        if minus_cells.shape != (surfaces,) or plus_cells.shape != (surfaces,):
            raise ValueError("Interface cell maps must align with interface areas.")
        if (
            np.any(minus_cells < 0)
            or np.any(minus_cells >= cells)
            or np.any(plus_cells < 0)
            or np.any(plus_cells >= cells)
        ):
            raise ValueError("Interface cell map references an unknown cell.")
        if normals.shape != (surfaces, dimension) or np.any(areas <= 0):
            raise ValueError("Interface normals and areas have incompatible shapes.")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=tolerance, rtol=0):
            raise ValueError("Interface normals must be unit vectors.")
        if surface_generator.shape != (surfaces, surfaces):
            raise ValueError("Surface charge generator has incompatible shape.")
        if not np.allclose(areas @ surface_generator, 0, atol=tolerance, rtol=tolerance):
            raise ValueError(
                "Surface charge transport must conserve area-weighted charge."
            )
        if (
            minus_rheology.measure_weights.size != cells
            or plus_rheology.measure_weights.size != cells
        ):
            raise ValueError("Bulk rheology solvers must share the EHD cell topology.")
        return cls(
            ehd,
            minus_rheology,
            plus_rheology,
            jnp.asarray(fraction),
            jnp.asarray(stress_divergence),
            jnp.asarray(gradient),
            jnp.asarray(minus_cells),
            jnp.asarray(plus_cells),
            jnp.asarray(normals),
            jnp.asarray(areas),
            jnp.asarray(surface_generator),
        )

    def advance(
        self,
        state: SpatialElectroViscoelasticState,
        fixed_charge_density_c_m3: ArrayLike,
        step_size_s: float,
        /,
    ) -> SpatialElectroViscoelasticStep:
        if step_size_s <= 0:
            raise ValueError("Electroviscoelastic step size must be positive.")
        minus_stress = self.minus_rheology.law.stress(state.conformation_minus)
        plus_stress = self.plus_rheology.law.stress(state.conformation_plus)
        fraction = self.minus_phase_fraction[:, None, None]
        polymer_stress = fraction * minus_stress + (1 - fraction) * plus_stress
        polymer_force = (
            self.stress_divergence_operator_m_inv @ polymer_stress.reshape((-1,))
        ).reshape((-1, self.ehd.spatial_dimension))
        ehd = self.ehd.advance(
            state.electrohydrodynamic,
            fixed_charge_density_c_m3,
            step_size_s,
            external_body_force_n_m3=polymer_force,
        )
        velocity_gradient = (
            self.velocity_gradient_operator_m_inv @ ehd.state.velocity_m_s.reshape((-1,))
        ).reshape((-1, self.ehd.spatial_dimension, self.ehd.spatial_dimension))
        minus = self.minus_rheology.advance(
            state.conformation_minus, velocity_gradient, step_size_s
        )
        plus = self.plus_rheology.advance(
            state.conformation_plus, velocity_gradient, step_size_s
        )

        electric = ehd.electric_field_v_m
        minus_field = electric[self.interface_minus_cells]
        plus_field = electric[self.interface_plus_cells]
        normal_current_minus = self.ehd.conductivity_s_m[
            self.interface_minus_cells
        ] * contract("sd,sd->s", minus_field, self.interface_normals)
        normal_current_plus = self.ehd.conductivity_s_m[
            self.interface_plus_cells
        ] * contract("sd,sd->s", plus_field, self.interface_normals)
        current_jump = normal_current_minus - normal_current_plus
        surface_matrix = (
            jnp.eye(state.surface_charge_c_m2.size)
            - float(step_size_s) * self.surface_charge_generator_s_inv
        )
        surface_right = state.surface_charge_c_m2 + float(step_size_s) * current_jump
        surface = solve(
            LinearSystem(DenseLinearOperator(surface_matrix)),
            surface_right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        charge_balance = contract(
            "s,s->", self.interface_areas_m2, surface.value - state.surface_charge_c_m2
        ) - float(step_size_s) * contract("s,s->", self.interface_areas_m2, current_jump)

        electric_minus = maxwell_stress(
            minus_field, self.ehd.permittivity_f_m[self.interface_minus_cells]
        )
        electric_plus = maxwell_stress(
            plus_field, self.ehd.permittivity_f_m[self.interface_plus_cells]
        )
        total_minus = minus.polymer_stress_pa[self.interface_minus_cells] + electric_minus
        total_plus = plus.polymer_stress_pa[self.interface_plus_cells] + electric_plus
        traction = contract(
            "sij,sj->si", total_plus - total_minus, self.interface_normals
        )
        polymer_energy = contract(
            "q,q->",
            self.ehd.cell_volumes_m3,
            self.minus_phase_fraction
            * self.minus_rheology.law.free_energy_density(minus.conformation)
            + (1 - self.minus_phase_fraction)
            * self.plus_rheology.law.free_energy_density(plus.conformation),
        )
        next_state = SpatialElectroViscoelasticState(
            ehd.state,
            minus.conformation,
            plus.conformation,
            surface.value,
        )
        successful = (
            ehd.successful
            & minus.successful
            & plus.successful
            & surface.successful
            & jnp.all(jnp.isfinite(traction))
        )
        return SpatialElectroViscoelasticStep(
            next_state,
            traction,
            charge_balance,
            polymer_energy,
            polymer_force,
            successful,
        )


__all__ = [
    "SpatialElectroViscoelasticState",
    "SpatialElectroViscoelasticStep",
    "SpatialElectroViscoelasticWorkflow",
]
