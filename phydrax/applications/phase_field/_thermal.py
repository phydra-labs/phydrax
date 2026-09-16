#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._multiphysics_ledger import EntropyProductionBalance


class NonisothermalPhaseEvaluation(StrictModule):
    grand_potential: Array
    composition: Array
    susceptibility: Array
    entropy: Array
    internal_energy: Array
    heat_capacity: Array
    finite: Array
    stable: Array
    phase_id: str = eqx.field(static=True)


class NonisothermalGrandPotentialPhase(StrictModule, NonTrainableState):
    """Constant-heat-capacity grand potential with exact thermal derivatives."""

    reference_temperature: Array
    reference_grand_potential: Array
    reference_entropy: Array
    reference_composition: Array
    susceptibility: Array
    heat_capacity: Array
    thermal_conductivity: Array
    phase_id: str = eqx.field(static=True)
    component_count: int = eqx.field(static=True)

    def __init__(
        self,
        phase_id: str,
        /,
        *,
        reference_temperature: ArrayLike,
        reference_grand_potential: ArrayLike,
        reference_entropy: ArrayLike,
        reference_composition: ArrayLike,
        susceptibility: ArrayLike,
        heat_capacity: ArrayLike,
        thermal_conductivity: ArrayLike,
    ):
        identifier = str(phase_id)
        scalars = tuple(
            np.asarray(value)
            for value in (
                reference_temperature,
                reference_grand_potential,
                reference_entropy,
                heat_capacity,
                thermal_conductivity,
            )
        )
        composition = np.asarray(reference_composition)
        response = np.asarray(susceptibility)
        if (
            not identifier
            or any(value.shape != () or not np.isfinite(value) for value in scalars)
            or scalars[0] <= 0.0
            or scalars[3] <= 0.0
            or scalars[4] <= 0.0
            or composition.ndim != 1
            or composition.size == 0
            or np.any(~np.isfinite(composition))
            or response.shape != (composition.size, composition.size)
            or np.any(~np.isfinite(response))
        ):
            raise ValueError("Nonisothermal phase thermodynamics are invalid.")
        symmetric = 0.5 * (response + response.T)
        scale = max(float(np.max(np.abs(symmetric))), 1.0)
        tolerance = 128.0 * np.finfo(symmetric.dtype).eps * scale
        if (
            np.max(np.abs(response - response.T)) > tolerance
            or np.min(np.linalg.eigvalsh(symmetric)) < -tolerance
        ):
            raise ValueError("Thermodynamic susceptibility must be symmetric PSD.")
        self.reference_temperature = jnp.asarray(scalars[0])
        self.reference_grand_potential = jnp.asarray(scalars[1])
        self.reference_entropy = jnp.asarray(scalars[2])
        self.reference_composition = jnp.asarray(composition)
        self.susceptibility = jnp.asarray(symmetric)
        self.heat_capacity = jnp.asarray(scalars[3])
        self.thermal_conductivity = jnp.asarray(scalars[4])
        self.component_count = composition.size
        self.phase_id = canonical_fingerprint(
            {
                "kind": "nonisothermal-grand-potential-phase",
                "declared_id": identifier,
                "reference_temperature": float(scalars[0]),
                "reference_grand_potential": float(scalars[1]),
                "reference_entropy": float(scalars[2]),
                "reference_composition": array_tree_fingerprint(composition),
                "susceptibility": array_tree_fingerprint(symmetric),
                "heat_capacity": float(scalars[3]),
                "thermal_conductivity": float(scalars[4]),
            }
        )

    def evaluate(
        self,
        chemical_potential: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> NonisothermalPhaseEvaluation:
        potential = jnp.asarray(chemical_potential)
        thermal = jnp.asarray(temperature, dtype=potential.dtype)
        if (
            potential.shape[-1] != self.component_count
            or thermal.shape != (potential.shape[:-1])
        ):
            raise ValueError("Nonisothermal phase evaluation shapes are incompatible.")
        thermal = eqx.error_if(
            thermal,
            ~jnp.isfinite(thermal) | (thermal <= 0.0),
            "Temperature must be positive and finite.",
        )
        reference_temperature = self.reference_temperature.astype(potential.dtype)
        delta = thermal - reference_temperature
        logarithm = jnp.log(thermal / reference_temperature)
        entropy = self.reference_entropy.astype(potential.dtype) + (
            self.heat_capacity.astype(potential.dtype) * logarithm
        )
        thermal_grand = (
            self.reference_grand_potential.astype(potential.dtype)
            - self.reference_entropy.astype(potential.dtype) * delta
            - self.heat_capacity.astype(potential.dtype)
            * (thermal * logarithm - thermal + reference_temperature)
        )
        composition = self.reference_composition.astype(potential.dtype) + ein.contract(
            "ij,...j->...i", self.susceptibility.astype(potential.dtype), potential
        )
        grand = (
            thermal_grand
            - ein.contract("i,...i->...", self.reference_composition, potential)
            - 0.5
            * ein.contract(
                "...i,ij,...j->...",
                potential,
                self.susceptibility.astype(potential.dtype),
                potential,
            )
        )
        internal = (
            grand
            + thermal * entropy
            + ein.contract("...i,...i->...", potential, composition)
        )
        heat_capacity = jnp.broadcast_to(
            self.heat_capacity.astype(potential.dtype), thermal.shape
        )
        eigenvalues = jnp.linalg.eigvalsh(self.susceptibility)
        finite = (
            jnp.all(jnp.isfinite(grand))
            & jnp.all(jnp.isfinite(composition))
            & jnp.all(jnp.isfinite(entropy))
            & jnp.all(jnp.isfinite(internal))
        )
        stable = (
            jnp.min(eigenvalues) >= -64.0 * jnp.finfo(eigenvalues.dtype).eps
        ) & jnp.all(heat_capacity > 0.0)
        return NonisothermalPhaseEvaluation(
            grand,
            composition,
            self.susceptibility,
            entropy,
            internal,
            heat_capacity,
            finite,
            stable,
            self.phase_id,
        )


class NonisothermalMaterialCatalog(StrictModule, NonTrainableState):
    phases: tuple[NonisothermalGrandPotentialPhase, ...]
    phase_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    catalog_id: str = eqx.field(static=True)

    def __init__(self, phases: Sequence[NonisothermalGrandPotentialPhase], /):
        values = tuple(phases)
        if (
            len(values) < 2
            or any(
                not isinstance(value, NonisothermalGrandPotentialPhase)
                for value in values
            )
            or len({value.phase_id for value in values}) != len(values)
            or len({value.component_count for value in values}) != 1
        ):
            raise ValueError("Nonisothermal phase catalog is incompatible.")
        self.phases = values
        self.phase_count = len(values)
        self.component_count = values[0].component_count
        self.catalog_id = canonical_fingerprint(
            {
                "kind": "nonisothermal-material-catalog",
                "phases": [value.phase_id for value in values],
            }
        )


class NonisothermalMixtureEvaluation(StrictModule):
    weights: Array
    grand_potential: Array
    composition: Array
    entropy: Array
    internal_energy: Array
    heat_capacity: Array
    thermal_conductivity: Array
    finite: Array
    stable: Array


class NonisothermalSolidificationModel(StrictModule, NonTrainableState):
    catalog: NonisothermalMaterialCatalog
    barrier_scale: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        catalog: NonisothermalMaterialCatalog,
        /,
        *,
        barrier_scale: ArrayLike,
    ):
        if not isinstance(catalog, NonisothermalMaterialCatalog):
            raise TypeError("catalog must be NonisothermalMaterialCatalog.")
        barrier = np.asarray(barrier_scale)
        if barrier.shape != () or not np.isfinite(barrier) or barrier < 0.0:
            raise ValueError("Solidification barrier scale must be nonnegative.")
        self.catalog = catalog
        self.barrier_scale = jnp.asarray(barrier)
        self.model_id = canonical_fingerprint(
            {
                "kind": "nonisothermal-solidification-model",
                "catalog": catalog.catalog_id,
                "barrier_scale": float(barrier),
            }
        )

    def evaluate(
        self,
        phase_logits: ArrayLike,
        chemical_potential: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> NonisothermalMixtureEvaluation:
        logits = jnp.asarray(phase_logits)
        potential = jnp.asarray(chemical_potential, dtype=logits.dtype)
        thermal = jnp.asarray(temperature, dtype=logits.dtype)
        if (
            logits.shape[-1] != self.catalog.phase_count
            or potential.shape[-1] != self.catalog.component_count
            or logits.shape[:-1] != potential.shape[:-1]
            or thermal.shape != logits.shape[:-1]
        ):
            raise ValueError("Nonisothermal mixture field shapes are incompatible.")
        weights = jax.nn.softmax(logits, axis=-1)
        evaluations = tuple(
            phase.evaluate(potential, thermal) for phase in self.catalog.phases
        )
        grand = ein.contract(
            "...p,...p->...",
            weights,
            jnp.stack(tuple(value.grand_potential for value in evaluations), axis=-1),
        )
        composition = ein.contract(
            "...p,...pc->...c",
            weights,
            jnp.stack(tuple(value.composition for value in evaluations), axis=-2),
        )
        entropy = ein.contract(
            "...p,...p->...",
            weights,
            jnp.stack(tuple(value.entropy for value in evaluations), axis=-1),
        )
        internal = ein.contract(
            "...p,...p->...",
            weights,
            jnp.stack(tuple(value.internal_energy for value in evaluations), axis=-1),
        )
        heat_capacity = ein.contract(
            "...p,...p->...",
            weights,
            jnp.stack(tuple(value.heat_capacity for value in evaluations), axis=-1),
        )
        conductivity = ein.contract(
            "...p,p->...",
            weights,
            jnp.stack(
                tuple(phase.thermal_conductivity for phase in self.catalog.phases)
            ).astype(logits.dtype),
        )
        barrier = self.barrier_scale.astype(logits.dtype) * jnp.sum(
            weights**2 * (1.0 - weights) ** 2, axis=-1
        )
        internal = internal + barrier
        grand = grand + barrier
        finite = jnp.all(
            jnp.stack(tuple(value.finite for value in evaluations))
        ) & jnp.all(jnp.isfinite(internal))
        stable = jnp.all(
            jnp.stack(tuple(value.stable for value in evaluations))
        ) & jnp.all(heat_capacity > 0.0)
        return NonisothermalMixtureEvaluation(
            weights,
            grand,
            composition,
            entropy,
            internal,
            heat_capacity,
            conductivity,
            finite,
            stable,
        )


class NonisothermalSolidificationState(StrictModule):
    phase_logits: Array
    chemical_potential: Array
    temperature: Array
    internal_energy: Array
    composition: Array
    entropy: Array


class ThermalSolidificationEvidence(StrictModule):
    energy_before: Array
    energy_after: Array
    heat_input: Array
    energy_defect: Array
    entropy: EntropyProductionBalance
    constitutive_residual: Array
    finite: Array
    successful: Array


class NonisothermalSolidificationPlan(StrictModule, NonTrainableState):
    """Homogeneous constitutive step used by FE/FV thermal residuals."""

    model: NonisothermalSolidificationModel
    absolute_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: NonisothermalSolidificationModel,
        /,
        *,
        absolute_tolerance: float = 1.0e-10,
        maximum_iterations: int = 32,
    ):
        if not isinstance(model, NonisothermalSolidificationModel):
            raise TypeError("model must be NonisothermalSolidificationModel.")
        tolerance = float(absolute_tolerance)
        iterations = int(maximum_iterations)
        if not np.isfinite(tolerance) or tolerance <= 0.0 or iterations < 1:
            raise ValueError("Nonisothermal solve policy is invalid.")
        self.model = model
        self.absolute_tolerance = tolerance
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonisothermal-solidification-plan",
                "model": model.model_id,
                "absolute_tolerance": tolerance,
                "maximum_iterations": iterations,
            }
        )

    def initialize(
        self,
        phase_logits: ArrayLike,
        chemical_potential: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> NonisothermalSolidificationState:
        logits = jnp.asarray(phase_logits)
        potential = jnp.asarray(chemical_potential, dtype=logits.dtype)
        thermal = jnp.asarray(temperature, dtype=logits.dtype)
        evaluation = self.model.evaluate(logits, potential, thermal)
        return NonisothermalSolidificationState(
            logits,
            potential,
            thermal,
            evaluation.internal_energy,
            evaluation.composition,
            evaluation.entropy,
        )

    def step(
        self,
        state: NonisothermalSolidificationState,
        phase_logits: ArrayLike,
        chemical_potential: ArrayLike,
        /,
        *,
        heat_input: ArrayLike,
        entropy_flux: ArrayLike = 0.0,
        entropy_production: ArrayLike = 0.0,
    ) -> tuple[NonisothermalSolidificationState, ThermalSolidificationEvidence]:
        logits = jnp.asarray(phase_logits, dtype=state.phase_logits.dtype)
        potential = jnp.asarray(chemical_potential, dtype=state.phase_logits.dtype)
        heat = jnp.asarray(heat_input, dtype=state.phase_logits.dtype)
        target = state.internal_energy + heat
        temperature = state.temperature
        for _ in range(self.maximum_iterations):
            evaluation = self.model.evaluate(logits, potential, temperature)
            residual = evaluation.internal_energy - target
            temperature = temperature - residual / evaluation.heat_capacity
            temperature = eqx.error_if(
                temperature,
                ~jnp.isfinite(temperature) | (temperature <= 0.0),
                "Thermal constitutive inversion left the positive-temperature domain.",
            )
        candidate_evaluation = self.model.evaluate(logits, potential, temperature)
        residual = candidate_evaluation.internal_energy - target
        entropy = EntropyProductionBalance(
            jnp.sum(state.entropy),
            jnp.sum(candidate_evaluation.entropy),
            entropy_flux=jnp.asarray(entropy_flux),
            production=jnp.asarray(entropy_production),
        )
        energy_before = jnp.sum(state.internal_energy)
        energy_after = jnp.sum(candidate_evaluation.internal_energy)
        heat_total = jnp.sum(heat)
        energy_defect = energy_after - energy_before - heat_total
        finite = (
            candidate_evaluation.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(energy_defect)
        )
        tolerance = self.absolute_tolerance * jnp.maximum(
            jnp.maximum(jnp.abs(energy_before), jnp.abs(energy_after)), 1.0
        )
        successful = (
            finite
            & candidate_evaluation.stable
            & (jnp.max(jnp.abs(residual)) <= tolerance)
            & (jnp.abs(energy_defect) <= tolerance)
            & entropy.nonnegative
        )
        candidate = NonisothermalSolidificationState(
            logits,
            potential,
            temperature,
            candidate_evaluation.internal_energy,
            candidate_evaluation.composition,
            candidate_evaluation.entropy,
        )
        evidence = ThermalSolidificationEvidence(
            energy_before,
            energy_after,
            heat_total,
            energy_defect,
            entropy,
            jnp.max(jnp.abs(residual)),
            finite,
            successful,
        )
        return candidate, evidence


__all__ = [
    "NonisothermalGrandPotentialPhase",
    "NonisothermalMaterialCatalog",
    "NonisothermalMixtureEvaluation",
    "NonisothermalPhaseEvaluation",
    "NonisothermalSolidificationModel",
    "NonisothermalSolidificationPlan",
    "NonisothermalSolidificationState",
    "ThermalSolidificationEvidence",
]
