#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import PreparedMACOperators
from ..equations._chemical_mechanism import (
    ChemicalRateEvaluation,
    PreparedChemicalMechanism,
)
from ..equations._chemical_rates import ChemicalRateKind, ChemicalRateRuntime
from ..equations._chemical_species import ChemicalPhaseKind
from ..equations._electrochemistry import FARADAY_CONSTANT


class ReactiveElectrodeState(StrictModule):
    surface_amount: Array
    surface_charge: Array
    stern_potential: Array
    state_id: str = eqx.field(static=True)


class ReactiveElectrodeEvaluation(StrictModule):
    mechanism: ChemicalRateEvaluation
    full_concentrations: Array
    reaction_extent_rate: Array
    surface_amount_rate: Array
    bulk_boundary_flux: Array
    faradaic_current: Array
    site_margin: Array
    charge_current_defect: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactiveElectrodeStepResult(StrictModule):
    state: ReactiveElectrodeState
    bulk_boundary_flux: Array
    evaluation: ReactiveElectrodeEvaluation
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactiveElectrodePlan(StrictModule, NonTrainableState):
    mechanism: PreparedChemicalMechanism
    boundary_node_indices: Array
    face_measures: Array
    electron_transfer: Array
    surface_mask: Array
    capacitance_per_area: Array
    boundary_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        boundary_node_indices: ArrayLike,
        face_measures: ArrayLike,
        electron_transfer: ArrayLike,
        capacitance_per_area: ArrayLike,
        /,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        indices = np.asarray(boundary_node_indices)
        measures = np.asarray(face_measures, dtype=np.float64)
        electrons = np.asarray(electron_transfer)
        capacitance = np.asarray(capacitance_per_area, dtype=np.float64)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or measures.shape != indices.shape
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0.0)
            or electrons.shape != (mechanism.reaction_count,)
            or not np.issubdtype(electrons.dtype, np.integer)
            or capacitance.shape not in ((), indices.shape)
            or np.any(~np.isfinite(capacitance))
            or np.any(capacitance <= 0.0)
        ):
            raise ValueError("Reactive electrode geometry/charge inputs are invalid.")
        if not any(
            reaction.forward_rate.kind is ChemicalRateKind.BUTLER_VOLMER
            or (
                reaction.reverse_rate is not None
                and reaction.reverse_rate.kind is ChemicalRateKind.BUTLER_VOLMER
            )
            for reaction in mechanism.reactions
        ):
            raise ValueError("Reactive electrode mechanism has no Butler-Volmer rate.")
        mask = np.asarray(
            mechanism.schema.phase_mask(ChemicalPhaseKind.SURFACE), dtype=np.bool_
        )
        if not np.any(mask):
            raise ValueError("Reactive electrode mechanism requires surface species.")
        self.mechanism = mechanism
        self.boundary_node_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.face_measures = jnp.asarray(measures)
        self.electron_transfer = jnp.asarray(electrons, dtype=jnp.int32)
        self.surface_mask = jnp.asarray(mask)
        self.capacitance_per_area = jnp.broadcast_to(
            jnp.asarray(capacitance), measures.shape
        )
        self.boundary_count = indices.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-electrode",
                "mechanism": mechanism.mechanism_id,
                "indices": array_tree_fingerprint(indices),
                "measures": array_tree_fingerprint(measures),
                "electrons": array_tree_fingerprint(electrons),
                "capacitance": array_tree_fingerprint(
                    np.broadcast_to(capacitance, measures.shape)
                ),
            }
        )

    def initialize(
        self,
        surface_amount: ArrayLike,
        surface_charge: ArrayLike = 0.0,
        /,
    ) -> ReactiveElectrodeState:
        amount = jnp.asarray(surface_amount)
        charge = jnp.broadcast_to(
            jnp.asarray(surface_charge, dtype=amount.dtype),
            (self.boundary_count,),
        )
        if amount.shape != (
            self.boundary_count,
            self.mechanism.schema.species_count,
        ):
            raise ValueError("surface_amount must have boundary/species shape.")
        if not bool(jnp.all(jnp.where(self.surface_mask, amount >= 0.0, amount == 0.0))):
            raise ValueError("Only surface species may carry initial surface amount.")
        stern = charge / (
            self.capacitance_per_area.astype(amount.dtype) * self.face_measures
        )
        state_id = canonical_fingerprint(
            {
                "kind": "reactive-electrode-state",
                "plan": self.plan_id,
                "shape": list(amount.shape),
            }
        )
        return ReactiveElectrodeState(amount, charge, stern, state_id)

    def evaluate(
        self,
        boundary_concentrations: ArrayLike,
        state: ReactiveElectrodeState,
        electrolyte_potential: ArrayLike,
        electrode_potential: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> ReactiveElectrodeEvaluation:
        concentration = jnp.asarray(boundary_concentrations)
        if concentration.shape != state.surface_amount.shape:
            raise ValueError("boundary_concentrations must have boundary/species shape.")
        electrolyte = jnp.broadcast_to(
            jnp.asarray(electrolyte_potential, dtype=concentration.dtype),
            (self.boundary_count,),
        )
        electrode = jnp.broadcast_to(
            jnp.asarray(electrode_potential, dtype=concentration.dtype),
            (self.boundary_count,),
        )
        temperature_ = jnp.broadcast_to(
            jnp.asarray(temperature, dtype=concentration.dtype),
            (self.boundary_count,),
        )
        pressure_ = jnp.broadcast_to(
            jnp.asarray(pressure, dtype=concentration.dtype),
            (self.boundary_count,),
        )
        surface_concentration = state.surface_amount / self.face_measures[:, None]
        full = jnp.where(self.surface_mask, surface_concentration, concentration)
        overpotential = electrode - electrolyte - state.stern_potential
        runtime = ChemicalRateRuntime(
            jnp.zeros((0,), dtype=concentration.dtype),
            overpotential,
        )
        fields = self.mechanism.evaluate(
            full,
            temperature_,
            pressure_,
            runtime=runtime,
        )
        extent = fields.net_progress_rates * self.face_measures[:, None]
        amount_rate = contract("br,rs->bs", extent, self.mechanism.net_stoichiometry)
        surface_rate = jnp.where(self.surface_mask, amount_rate, 0.0)
        bulk_flux = jnp.where(self.surface_mask, 0.0, -amount_rate)
        current = FARADAY_CONSTANT * contract("br,r->b", extent, self.electron_transfer)
        capacity = jnp.zeros((self.boundary_count,), dtype=concentration.dtype)
        for phase in self.mechanism.schema.phase_specs:
            if phase.kind is ChemicalPhaseKind.SURFACE:
                if phase.site_density is None:
                    raise ValueError("Surface phase lost its site density.")
                capacity = capacity + phase.site_density * self.face_measures
        occupied = jnp.sum(state.surface_amount, axis=-1)
        site_margin = capacity - occupied
        charge_rate = -current
        charge_current_defect = charge_rate + current
        successful = (
            fields.successful
            & jnp.all(jnp.isfinite(extent))
            & jnp.all(jnp.isfinite(current))
            & jnp.all(site_margin >= 0.0)
            & jnp.all(jnp.abs(charge_current_defect) == 0.0)
        )
        return ReactiveElectrodeEvaluation(
            fields,
            full,
            extent,
            surface_rate,
            bulk_flux,
            current,
            site_margin,
            charge_current_defect,
            successful,
            self.plan_id,
        )

    def step(
        self,
        boundary_concentrations: ArrayLike,
        state: ReactiveElectrodeState,
        electrolyte_potential: ArrayLike,
        electrode_potential: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> ReactiveElectrodeStepResult:
        step = jnp.asarray(time_step)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        evaluation = self.evaluate(
            boundary_concentrations,
            state,
            electrolyte_potential,
            electrode_potential,
            temperature,
            pressure,
        )
        candidate_amount = state.surface_amount + step * evaluation.surface_amount_rate
        candidate_charge = state.surface_charge - step * evaluation.faradaic_current
        candidate_stern = candidate_charge / (
            self.capacitance_per_area * self.face_measures
        )
        candidate = ReactiveElectrodeState(
            candidate_amount,
            candidate_charge,
            candidate_stern,
            state.state_id,
        )
        successful = (
            evaluation.successful
            & jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(candidate_amount >= 0.0)
            & jnp.all(jnp.isfinite(candidate_charge))
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate,
            state,
        )
        return ReactiveElectrodeStepResult(
            accepted,
            jnp.where(successful, evaluation.bulk_boundary_flux, 0.0),
            evaluation,
            successful,
            self.plan_id,
        )


class MACReactiveElectrodeEvaluation(StrictModule):
    electrode: ReactiveElectrodeEvaluation
    concentration_rate: Array
    charge_current_defect: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class MACReactiveElectrodeBinding(StrictModule, NonTrainableState):
    """Bind reactive electrode slots to fixed MAC boundary-adjacent cells."""

    electrode: ReactiveElectrodePlan
    operators: PreparedMACOperators
    bulk_species_indices: Array
    fixed_mechanism_concentrations: Array
    pnp_species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrode: ReactiveElectrodePlan,
        operators: PreparedMACOperators,
        bulk_species_indices: ArrayLike,
        fixed_mechanism_concentrations: ArrayLike,
        /,
    ) -> None:
        indices = np.asarray(bulk_species_indices, dtype=np.int32)
        fixed = np.asarray(fixed_mechanism_concentrations, dtype=np.float64)
        cell_count = int(np.prod(operators.discretization.cell_shape))
        if (
            not isinstance(electrode, ReactiveElectrodePlan)
            or not isinstance(operators, PreparedMACOperators)
            or indices.ndim != 1
            or indices.size == 0
            or len({int(value) for value in indices}) != indices.size
            or np.any(
                (indices < 0) | (indices >= electrode.mechanism.schema.species_count)
            )
            or fixed.shape != (electrode.mechanism.schema.species_count,)
            or np.any(~np.isfinite(fixed))
            or np.any(fixed < 0.0)
            or np.any(np.asarray(electrode.boundary_node_indices) >= cell_count)
        ):
            raise ValueError("MAC reactive-electrode binding is invalid.")
        if np.any(np.asarray(electrode.surface_mask)[indices]):
            raise ValueError("PNP bulk species cannot map to surface species.")
        self.electrode = electrode
        self.operators = operators
        self.bulk_species_indices = jnp.asarray(indices)
        self.fixed_mechanism_concentrations = jnp.asarray(fixed)
        self.pnp_species_count = indices.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-reactive-electrode-binding",
                "electrode": electrode.plan_id,
                "operators": operators.prepared_id,
                "bulk_species_indices": tuple(indices),
                "fixed_mechanism_concentrations": array_tree_fingerprint(fixed),
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        electrode_state: ReactiveElectrodeState,
        electrolyte_potential: ArrayLike,
        electrode_potential: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> MACReactiveElectrodeEvaluation:
        concentration = jnp.asarray(concentrations)
        potential = jnp.asarray(electrolyte_potential, dtype=concentration.dtype)
        expected = self.operators.discretization.cell_shape + (self.pnp_species_count,)
        if concentration.shape != expected or potential.shape != expected[:-1]:
            raise ValueError(
                "MAC electrode concentrations or potential have wrong shape."
            )
        flat = concentration.reshape((-1, self.pnp_species_count))
        flat_potential = potential.reshape((-1,))
        cells = self.electrode.boundary_node_indices
        full = jnp.broadcast_to(
            self.fixed_mechanism_concentrations.astype(concentration.dtype),
            (
                self.electrode.boundary_count,
                self.electrode.mechanism.schema.species_count,
            ),
        )
        full = full.at[:, self.bulk_species_indices].set(flat[cells])
        evaluation = self.electrode.evaluate(
            full,
            electrode_state,
            flat_potential[cells],
            electrode_potential,
            temperature,
            pressure,
        )
        volume = self.operators.discretization.cell_volumes.reshape((-1,))[cells]
        bulk_flux = evaluation.bulk_boundary_flux[:, self.bulk_species_indices]
        flat_rate = jnp.zeros_like(flat).at[cells].add(-bulk_flux / volume[:, None])
        concentration_rate = flat_rate.reshape(expected)
        charges = self.electrode.mechanism.schema.charges[self.bulk_species_indices]
        bulk_current = FARADAY_CONSTANT * contract(
            "bs,s->b", bulk_flux, charges, backend="jax"
        )
        defect = bulk_current - evaluation.faradaic_current
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(bulk_current), jnp.abs(evaluation.faradaic_current)),
            1.0,
        )
        charge_closed = jnp.all(
            jnp.abs(defect) <= 512.0 * jnp.finfo(concentration.dtype).eps * scale
        )
        finite = (
            evaluation.successful
            & jnp.all(jnp.isfinite(concentration_rate))
            & jnp.all(jnp.isfinite(defect))
        )
        successful = finite & charge_closed
        reasons = jnp.where(
            successful,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(successful, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "mac-reactive-electrode-evidence", "plan": self.plan_id}
            ),
        )
        return MACReactiveElectrodeEvaluation(
            evaluation,
            concentration_rate,
            defect,
            header,
            self.plan_id,
        )


__all__ = [
    "MACReactiveElectrodeBinding",
    "MACReactiveElectrodeEvaluation",
    "ReactiveElectrodeEvaluation",
    "ReactiveElectrodePlan",
    "ReactiveElectrodeState",
    "ReactiveElectrodeStepResult",
]
