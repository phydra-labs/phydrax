#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded isothermal finite-volume Doyle–Fuller–Newman battery model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ...qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class DFNParameters:
    negative_length_m: float
    separator_length_m: float
    positive_length_m: float
    negative_porosity: float
    separator_porosity: float
    positive_porosity: float
    negative_surface_area_m2_m3: float
    positive_surface_area_m2_m3: float
    negative_solid_conductivity_s_m: float
    positive_solid_conductivity_s_m: float
    electrolyte_conductivity_s_m: float
    electrolyte_diffusivity_m2_s: float
    negative_solid_diffusivity_m2_s: float
    positive_solid_diffusivity_m2_s: float
    negative_particle_radius_m: float
    positive_particle_radius_m: float
    negative_max_concentration_mol_m3: float
    positive_max_concentration_mol_m3: float
    negative_exchange_current_a_m2: float
    positive_exchange_current_a_m2: float
    transference_number: float
    temperature_k: float
    negative_ocp: Callable[[Array], Array]
    positive_ocp: Callable[[Array], Array]

    def __post_init__(self) -> None:
        positive = (
            self.negative_length_m,
            self.separator_length_m,
            self.positive_length_m,
            self.negative_porosity,
            self.separator_porosity,
            self.positive_porosity,
            self.negative_surface_area_m2_m3,
            self.positive_surface_area_m2_m3,
            self.negative_solid_conductivity_s_m,
            self.positive_solid_conductivity_s_m,
            self.electrolyte_conductivity_s_m,
            self.electrolyte_diffusivity_m2_s,
            self.negative_solid_diffusivity_m2_s,
            self.positive_solid_diffusivity_m2_s,
            self.negative_particle_radius_m,
            self.positive_particle_radius_m,
            self.negative_max_concentration_mol_m3,
            self.positive_max_concentration_mol_m3,
            self.negative_exchange_current_a_m2,
            self.positive_exchange_current_a_m2,
            self.temperature_k,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("DFN dimensional parameters must be finite and positive.")
        if not 0.0 < self.transference_number < 1.0:
            raise ValueError("DFN transference number must lie in (0, 1).")
        if not callable(self.negative_ocp) or not callable(self.positive_ocp):
            raise TypeError("DFN open-circuit potentials must be callable.")


@dataclass(frozen=True, slots=True)
class DFNState:
    electrolyte_concentration_mol_m3: Array
    negative_particle_concentration_mol_m3: Array
    positive_particle_concentration_mol_m3: Array
    time_s: Array


@dataclass(frozen=True, slots=True)
class DFNEvaluation:
    electrolyte_potential_v: Array
    negative_solid_potential_v: Array
    positive_solid_potential_v: Array
    negative_reaction_current_a_m2: Array
    positive_reaction_current_a_m2: Array
    voltage_v: Array
    residual_norm: Array
    converged: Array


@dataclass(frozen=True, slots=True)
class DFNStepResult:
    candidate_state: DFNState
    accepted_state: DFNState
    evaluation: DFNEvaluation
    accepted: Array
    minimum_concentration_mol_m3: Array


class IsothermalDFNPlan:
    """Small/medium isothermal DFN with explicit concentrations and implicit charge."""

    FARADAY = 96485.33212
    GAS_CONSTANT = 8.314462618

    def __init__(
        self,
        negative_cells: int,
        separator_cells: int,
        positive_cells: int,
        radial_cells: int,
        /,
        *,
        maximum_newton_steps: int = 12,
        residual_tolerance: float = 1.0e-4,
    ):
        counts = tuple(
            int(value)
            for value in (negative_cells, separator_cells, positive_cells, radial_cells)
        )
        if any(value < 2 for value in counts):
            raise ValueError(
                "DFN regions and radial particles require at least two cells."
            )
        (
            self.negative_cells,
            self.separator_cells,
            self.positive_cells,
            self.radial_cells,
        ) = counts
        self.maximum_newton_steps = int(maximum_newton_steps)
        self.residual_tolerance = float(residual_tolerance)
        if self.maximum_newton_steps <= 0 or self.residual_tolerance <= 0.0:
            raise ValueError("DFN Newton limits must be positive.")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isothermal-dfn-plan",
                "negative_cells": counts[0],
                "separator_cells": counts[1],
                "positive_cells": counts[2],
                "radial_cells": counts[3],
                "maximum_newton_steps": self.maximum_newton_steps,
                "residual_tolerance": self.residual_tolerance,
            }
        )

    @property
    def through_cells(self) -> int:
        return self.negative_cells + self.separator_cells + self.positive_cells

    def initial_state(
        self,
        parameters: DFNParameters,
        /,
        *,
        electrolyte_concentration_mol_m3: float,
        negative_stoichiometry: float,
        positive_stoichiometry: float,
    ) -> DFNState:
        if (
            not 0.0 < negative_stoichiometry < 1.0
            or not 0.0 < positive_stoichiometry < 1.0
        ):
            raise ValueError("Initial DFN stoichiometries must lie in (0, 1).")
        electrolyte = float(electrolyte_concentration_mol_m3)
        if not np.isfinite(electrolyte) or electrolyte <= 0.0:
            raise ValueError("Initial electrolyte concentration must be positive.")
        return DFNState(
            jnp.full((self.through_cells,), electrolyte),
            jnp.full(
                (self.negative_cells, self.radial_cells),
                negative_stoichiometry * parameters.negative_max_concentration_mol_m3,
            ),
            jnp.full(
                (self.positive_cells, self.radial_cells),
                positive_stoichiometry * parameters.positive_max_concentration_mol_m3,
            ),
            jnp.asarray(0.0),
        )

    def _geometry(self, parameters: DFNParameters):
        lengths = jnp.concatenate(
            (
                jnp.full(
                    (self.negative_cells,),
                    parameters.negative_length_m / self.negative_cells,
                ),
                jnp.full(
                    (self.separator_cells,),
                    parameters.separator_length_m / self.separator_cells,
                ),
                jnp.full(
                    (self.positive_cells,),
                    parameters.positive_length_m / self.positive_cells,
                ),
            )
        )
        centers = jnp.cumsum(lengths) - 0.5 * lengths
        return lengths, centers

    @staticmethod
    def _face_flux(potential, conductivity, centers, left, right):
        internal = -conductivity * jnp.diff(potential) / jnp.diff(centers)
        return jnp.concatenate((jnp.asarray([left]), internal, jnp.asarray([right])))

    def evaluate(
        self,
        state: DFNState,
        parameters: DFNParameters,
        current_density_a_m2: ArrayLike,
        /,
    ) -> DFNEvaluation:
        current = jnp.asarray(current_density_a_m2)
        concentration = state.electrolyte_concentration_mol_m3
        if concentration.shape != (self.through_cells,):
            raise ValueError("DFN electrolyte state has the wrong cell count.")
        lengths, centers = self._geometry(parameters)
        negative_slice = slice(0, self.negative_cells)
        positive_start = self.negative_cells + self.separator_cells
        positive_slice = slice(positive_start, self.through_cells)
        negative_surface = state.negative_particle_concentration_mol_m3[:, -1]
        positive_surface = state.positive_particle_concentration_mol_m3[:, -1]
        negative_theta = negative_surface / parameters.negative_max_concentration_mol_m3
        positive_theta = positive_surface / parameters.positive_max_concentration_mol_m3
        negative_ocp = parameters.negative_ocp(negative_theta)
        positive_ocp = parameters.positive_ocp(positive_theta)
        thermal = self.GAS_CONSTANT * parameters.temperature_k / self.FARADAY
        size = self.through_cells + 2 * self.negative_cells + 2 * self.positive_cells

        initial = jnp.concatenate(
            (
                jnp.zeros((self.through_cells,)),
                negative_ocp,
                positive_ocp,
                jnp.full(
                    (self.negative_cells,),
                    current
                    / (
                        parameters.negative_surface_area_m2_m3
                        * parameters.negative_length_m
                    ),
                ),
                jnp.full(
                    (self.positive_cells,),
                    -current
                    / (
                        parameters.positive_surface_area_m2_m3
                        * parameters.positive_length_m
                    ),
                ),
            )
        )
        if initial.shape != (size,):
            raise RuntimeError("DFN algebraic layout is inconsistent.")

        def unpack(vector):
            offset = 0
            phi_e = vector[offset : offset + self.through_cells]
            offset += self.through_cells
            phi_n = vector[offset : offset + self.negative_cells]
            offset += self.negative_cells
            phi_p = vector[offset : offset + self.positive_cells]
            offset += self.positive_cells
            j_n = vector[offset : offset + self.negative_cells]
            offset += self.negative_cells
            j_p = vector[offset : offset + self.positive_cells]
            return phi_e, phi_n, phi_p, j_n, j_p

        def residual(vector):
            phi_e, phi_n, phi_p, j_n, j_p = unpack(vector)
            i_n = self._face_flux(
                phi_n,
                parameters.negative_solid_conductivity_s_m,
                centers[negative_slice],
                current,
                0.0,
            )
            i_p = self._face_flux(
                phi_p,
                parameters.positive_solid_conductivity_s_m,
                centers[positive_slice],
                0.0,
                current,
            )
            solid_n = (
                jnp.diff(i_n) / lengths[negative_slice]
                + parameters.negative_surface_area_m2_m3 * j_n
            )
            solid_p = (
                jnp.diff(i_p) / lengths[positive_slice]
                + parameters.positive_surface_area_m2_m3 * j_p
            )
            log_concentration = jnp.log(concentration)
            diffusion_potential = 2.0 * thermal * (1.0 - parameters.transference_number)
            i_e_internal = -parameters.electrolyte_conductivity_s_m * jnp.diff(
                phi_e
            ) / jnp.diff(
                centers
            ) + parameters.electrolyte_conductivity_s_m * diffusion_potential * jnp.diff(
                log_concentration
            ) / jnp.diff(centers)
            i_e = jnp.concatenate((jnp.zeros((1,)), i_e_internal, jnp.zeros((1,))))
            source = jnp.concatenate(
                (
                    parameters.negative_surface_area_m2_m3 * j_n,
                    jnp.zeros((self.separator_cells,)),
                    parameters.positive_surface_area_m2_m3 * j_p,
                )
            )
            electrolyte = jnp.diff(i_e) / lengths - source
            electrolyte = electrolyte.at[0].set(phi_e[0])
            negative_exchange = parameters.negative_exchange_current_a_m2 * jnp.sqrt(
                jnp.maximum(negative_theta * (1.0 - negative_theta), 0.0)
                * concentration[negative_slice]
            )
            positive_exchange = parameters.positive_exchange_current_a_m2 * jnp.sqrt(
                jnp.maximum(positive_theta * (1.0 - positive_theta), 0.0)
                * concentration[positive_slice]
            )
            reaction_n = j_n - 2.0 * negative_exchange * jnp.sinh(
                (phi_n - phi_e[negative_slice] - negative_ocp) / (2.0 * thermal)
            )
            reaction_p = j_p - 2.0 * positive_exchange * jnp.sinh(
                (phi_p - phi_e[positive_slice] - positive_ocp) / (2.0 * thermal)
            )
            return jnp.concatenate(
                (electrolyte, solid_n, solid_p, reaction_n, reaction_p)
            )

        vector = initial
        for _ in range(self.maximum_newton_steps):
            linearization = la.prepare_linearization(residual, vector)
            jacobian = la.materialize(
                la.JacobianLinearOperator(linearization),
                la.MaterializationPolicy(
                    max_entries=int(vector.size) ** 2,
                    max_bytes=int(vector.size) ** 2 * vector.dtype.itemsize,
                ),
            )
            solve_result = la.solve(
                la.LinearSystem(la.DenseLinearOperator(jacobian)),
                -jnp.asarray(linearization.primal),
            )
            vector = vector + jnp.asarray(solve_result.value)
        defect_norm = jnp.linalg.norm(residual(vector))
        phi_e, phi_n, phi_p, j_n, j_p = unpack(vector)
        voltage = phi_p[-1] - phi_n[0]
        converged = jnp.isfinite(defect_norm) & (defect_norm <= self.residual_tolerance)
        return DFNEvaluation(
            phi_e, phi_n, phi_p, j_n, j_p, voltage, defect_norm, converged
        )

    @staticmethod
    def _particle_rate(concentration, diffusivity, radius, flux):
        radial_cells = concentration.shape[-1]
        step = radius / radial_cells
        inner = concentration[..., 1:] - concentration[..., :-1]
        face_flux = -diffusivity * inner / step
        outer = jnp.asarray(flux)
        if outer.ndim == concentration.ndim - 1:
            outer = outer[..., None]
        faces = jnp.concatenate((jnp.zeros_like(outer), face_flux, outer), axis=-1)
        outer_radius = jnp.arange(radial_cells + 1) * step
        area = outer_radius**2
        volume = (outer_radius[1:] ** 3 - outer_radius[:-1] ** 3) / 3.0
        return -(faces[..., 1:] * area[1:] - faces[..., :-1] * area[:-1]) / volume

    def step(
        self,
        state: DFNState,
        parameters: DFNParameters,
        current_density_a_m2: ArrayLike,
        step_size_s: ArrayLike,
        /,
    ) -> DFNStepResult:
        evaluation = self.evaluate(state, parameters, current_density_a_m2)
        step = jnp.asarray(step_size_s)
        lengths, centers = self._geometry(parameters)
        concentration = state.electrolyte_concentration_mol_m3
        porosity = jnp.concatenate(
            (
                jnp.full((self.negative_cells,), parameters.negative_porosity),
                jnp.full((self.separator_cells,), parameters.separator_porosity),
                jnp.full((self.positive_cells,), parameters.positive_porosity),
            )
        )
        internal = (
            -parameters.electrolyte_diffusivity_m2_s
            * jnp.diff(concentration)
            / jnp.diff(centers)
        )
        faces = jnp.concatenate((jnp.zeros((1,)), internal, jnp.zeros((1,))))
        source = jnp.concatenate(
            (
                parameters.negative_surface_area_m2_m3
                * evaluation.negative_reaction_current_a_m2,
                jnp.zeros((self.separator_cells,)),
                parameters.positive_surface_area_m2_m3
                * evaluation.positive_reaction_current_a_m2,
            )
        )
        electrolyte_rate = (
            -jnp.diff(faces) / lengths
            + (1.0 - parameters.transference_number) * source / self.FARADAY
        ) / porosity
        negative_flux = evaluation.negative_reaction_current_a_m2 / self.FARADAY
        positive_flux = evaluation.positive_reaction_current_a_m2 / self.FARADAY
        negative_rate = self._particle_rate(
            state.negative_particle_concentration_mol_m3,
            parameters.negative_solid_diffusivity_m2_s,
            parameters.negative_particle_radius_m,
            negative_flux,
        )
        positive_rate = self._particle_rate(
            state.positive_particle_concentration_mol_m3,
            parameters.positive_solid_diffusivity_m2_s,
            parameters.positive_particle_radius_m,
            positive_flux,
        )
        candidate = DFNState(
            concentration + step * electrolyte_rate,
            state.negative_particle_concentration_mol_m3 + step * negative_rate,
            state.positive_particle_concentration_mol_m3 + step * positive_rate,
            state.time_s + step,
        )
        minimum = jnp.min(
            jnp.concatenate(
                (
                    candidate.electrolyte_concentration_mol_m3.reshape((-1,)),
                    candidate.negative_particle_concentration_mol_m3.reshape((-1,)),
                    candidate.positive_particle_concentration_mol_m3.reshape((-1,)),
                )
            )
        )
        accepted = (
            evaluation.converged & jnp.isfinite(step) & (step > 0.0) & (minimum > 0.0)
        )
        accepted_state = DFNState(
            jnp.where(
                accepted,
                candidate.electrolyte_concentration_mol_m3,
                state.electrolyte_concentration_mol_m3,
            ),
            jnp.where(
                accepted,
                candidate.negative_particle_concentration_mol_m3,
                state.negative_particle_concentration_mol_m3,
            ),
            jnp.where(
                accepted,
                candidate.positive_particle_concentration_mol_m3,
                state.positive_particle_concentration_mol_m3,
            ),
            jnp.where(accepted, candidate.time_s, state.time_s),
        )
        return DFNStepResult(candidate, accepted_state, evaluation, accepted, minimum)


@dataclass(frozen=True, slots=True)
class SeriesBatteryPackState:
    cell_states: tuple[DFNState, ...]


@dataclass(frozen=True, slots=True)
class SeriesBatteryPackStep:
    state: SeriesBatteryPackState
    cell_results: tuple[DFNStepResult, ...]
    pack_voltage_v: Array
    accepted: Array


class SeriesBatteryPackPlan:
    """Series-connected native DFN cells with one shared current."""

    def __init__(
        self,
        cell_plans: Sequence[IsothermalDFNPlan],
        cell_parameters: Sequence[DFNParameters],
        /,
    ):
        plans = tuple(cell_plans)
        parameters = tuple(cell_parameters)
        if not plans or len(plans) != len(parameters):
            raise ValueError(
                "Series pack plans and parameters must be non-empty and aligned."
            )
        if any(not isinstance(plan, IsothermalDFNPlan) for plan in plans):
            raise TypeError(
                "Series packs currently require native IsothermalDFNPlan cells."
            )
        self.cell_plans = plans
        self.cell_parameters = parameters
        self.plan_id = canonical_fingerprint(
            {"kind": "series-dfn-pack", "cell_plan_ids": [plan.plan_id for plan in plans]}
        )

    def step(
        self,
        state: SeriesBatteryPackState,
        current_density_a_m2: ArrayLike,
        step_size_s: ArrayLike,
        /,
    ) -> SeriesBatteryPackStep:
        if len(state.cell_states) != len(self.cell_plans):
            raise ValueError("Series pack state does not match the cell count.")
        results = tuple(
            plan.step(cell_state, parameters, current_density_a_m2, step_size_s)
            for plan, parameters, cell_state in zip(
                self.cell_plans, self.cell_parameters, state.cell_states, strict=True
            )
        )
        accepted = jnp.all(jnp.stack(tuple(result.accepted for result in results)))
        next_state = SeriesBatteryPackState(
            tuple(
                DFNState(
                    jnp.where(
                        accepted,
                        result.accepted_state.electrolyte_concentration_mol_m3,
                        original.electrolyte_concentration_mol_m3,
                    ),
                    jnp.where(
                        accepted,
                        result.accepted_state.negative_particle_concentration_mol_m3,
                        original.negative_particle_concentration_mol_m3,
                    ),
                    jnp.where(
                        accepted,
                        result.accepted_state.positive_particle_concentration_mol_m3,
                        original.positive_particle_concentration_mol_m3,
                    ),
                    jnp.where(accepted, result.accepted_state.time_s, original.time_s),
                )
                for result, original in zip(results, state.cell_states, strict=True)
            )
        )
        voltage = jnp.sum(
            jnp.stack(tuple(result.evaluation.voltage_v for result in results))
        )
        return SeriesBatteryPackStep(next_state, results, voltage, accepted)


def battery_dfn_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return exact unreleased native DFN and series-pack candidates."""

    specifications = (
        (
            "battery.dfn.isothermal-finite-volume",
            {
                "thermal": "isothermal",
                "geometry": "one-dimensional-through-cell",
                "particles": "spherical-radial-finite-volume",
                "charge": "implicit-newton",
                "concentration": "explicit-candidate",
            },
        ),
        (
            "battery.pack.series-dfn",
            {
                "topology": "series",
                "cell_model": "native-isothermal-dfn",
                "current": "shared",
                "commit": "atomic-all-cells",
            },
        ),
    )
    gates = (
        "algebraic-residual",
        "mass-balance",
        "mesh-time-refinement",
        "reference-comparison",
        "resource-envelope",
        "public-workflow",
    )
    return tuple(
        CapabilityProfile(
            f"{capability}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(capability, attributes),),
            required_gates=gates,
        )
        for capability, attributes in specifications
    )


__all__ = [
    "battery_dfn_candidate_profiles",
    "DFNEvaluation",
    "DFNParameters",
    "DFNState",
    "DFNStepResult",
    "IsothermalDFNPlan",
    "SeriesBatteryPackPlan",
    "SeriesBatteryPackState",
    "SeriesBatteryPackStep",
]
