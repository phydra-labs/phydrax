# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....units import conversion_factor
from ....uq import (
    bennett_acceptance_ratio,
    evaluate_targeted_work,
    free_energy_perturbation,
    FreeEnergyResult,
    multistate_bennett_acceptance_ratio,
    ReducedPotentialSamples,
    thermodynamic_integration,
)
from .._construct import _identifier
from ..experiments._models import ThermodynamicConvention


@dataclass(frozen=True, slots=True)
class ProteinFreeEnergyEstimate:
    native_result: FreeEnergyResult
    thermal_energy: float
    protocol_id: str
    state_ids: tuple[str, ...]
    bias_ids: tuple[str, ...]

    @property
    def free_energies(self):
        return self.native_result.free_energies * self.thermal_energy

    @property
    def standard_errors(self):
        return self.native_result.standard_errors * self.thermal_energy


@dataclass(frozen=True, slots=True)
class ProteinFreeEnergyWorkflow:
    """Domain admission around native FEP/TI/BAR/MBAR and targeted-map owners.

    State definitions and composition are independent source identities. Bias
    IDs identify every sampling bias; numeric inputs MUST include full reduced
    potentials/work for the stated thermodynamic states, not optimizer losses.
    Statistical decorrelation/overlap evidence remains in the native result and
    source ensemble artifact. Configuration weights never imply path weights.
    """

    state_ids: tuple[str, ...]
    composition_id: str
    temperature_kelvin: float
    convention: ThermodynamicConvention
    ensemble_source_id: str
    decorrelation_evidence_id: str
    bias_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.state_ids) < 2 or len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError(
                "At least two independently identified thermodynamic states are required."
            )
        for value in (
            *self.state_ids,
            self.composition_id,
            self.ensemble_source_id,
            self.decorrelation_evidence_id,
            *self.bias_ids,
        ):
            _identifier(value, "free-energy evidence")
        if not np.isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("Temperature must be positive Kelvin.")

    @property
    def thermal_energy(self):
        return self.convention.thermal_constant * self.temperature_kelvin

    def fingerprint(self):
        return canonical_fingerprint(
            {
                "kind": "protein-free-energy-protocol",
                "states": self.state_ids,
                "composition": self.composition_id,
                "temperature": self.temperature_kelvin,
                "unit": self.convention.energy_unit.unit_id,
                "source": self.ensemble_source_id,
                "decorrelation": self.decorrelation_evidence_id,
                "biases": self.bias_ids,
            }
        )

    def _wrap(self, result):
        return ProteinFreeEnergyEstimate(
            result, self.thermal_energy, self.fingerprint(), self.state_ids, self.bias_ids
        )

    def _reduced(self, work, unit):
        return (
            jnp.asarray(work)
            * float(conversion_factor(unit, self.convention.energy_unit))
            / self.thermal_energy
        )

    def fep(self, forward_work, *, energy_unit):
        if len(self.state_ids) != 2:
            raise ValueError("FEP requires exactly source and destination states.")
        return self._wrap(
            free_energy_perturbation(
                self._reduced(forward_work, energy_unit), source_id=self.fingerprint()
            )
        )

    def bar(self, forward_work, reverse_work, *, energy_unit, **solver_options):
        if len(self.state_ids) != 2:
            raise ValueError("BAR requires exactly source and destination states.")
        return self._wrap(
            bennett_acceptance_ratio(
                self._reduced(forward_work, energy_unit),
                self._reduced(reverse_work, energy_unit),
                source_id=self.fingerprint(),
                **solver_options,
            )
        )

    def ti(self, lambda_values, derivative_means, derivative_errors, *, energy_unit):
        if len(lambda_values) != len(self.state_ids):
            raise ValueError(
                "Every TI quadrature state requires an independent state identity."
            )
        return self._wrap(
            thermodynamic_integration(
                lambda_values,
                self._reduced(derivative_means, energy_unit),
                self._reduced(derivative_errors, energy_unit),
                source_id=self.fingerprint(),
            )
        )

    def mbar(
        self,
        potential_energies,
        state_counts,
        origin_states,
        *,
        energy_unit,
        **solver_options,
    ):
        values = self._reduced(potential_energies, energy_unit)
        if values.shape[0] != len(self.state_ids):
            raise ValueError("MBAR potentials must cover every declared state.")
        counts, origins = np.asarray(state_counts), np.asarray(origin_states)
        if (
            origins.ndim != 1
            or not np.issubdtype(origins.dtype, np.integer)
            or np.any(origins < 0)
            or np.any(origins >= len(self.state_ids))
            or not np.array_equal(
                np.bincount(origins, minlength=len(self.state_ids)), counts
            )
        ):
            raise ValueError(
                "MBAR origin labels must reproduce the declared state counts."
            )
        samples = ReducedPotentialSamples(
            values, state_counts, origin_states, source_id=self.fingerprint()
        )
        return self._wrap(multistate_bennett_acceptance_ratio(samples, **solver_options))

    def targeted(self, problem, source_samples, *, target_samples=None, **solver_options):
        if len(self.state_ids) != 2:
            raise ValueError("A targeted-map calculation requires two declared states.")
        work = evaluate_targeted_work(
            problem, source_samples, target_samples=target_samples
        )
        if not bool(work.valid):
            raise ValueError("Targeted mapping failed round-trip/support qualification.")
        result = (
            free_energy_perturbation(work.forward_work, source_id=self.fingerprint())
            if work.reverse_work is None
            else bennett_acceptance_ratio(
                work.forward_work,
                work.reverse_work,
                source_id=self.fingerprint(),
                **solver_options,
            )
        )
        return self._wrap(result), work


__all__ = ["ProteinFreeEnergyEstimate", "ProteinFreeEnergyWorkflow"]
