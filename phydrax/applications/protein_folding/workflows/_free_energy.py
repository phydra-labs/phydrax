# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....units import conversion_factor
from ....uq import (
    bennett_acceptance_ratio,
    free_energy_perturbation,
    FreeEnergyResult,
    FreeEnergySelectionEvidence,
    FreeEnergySelectionPlan,
    multistate_bennett_acceptance_ratio,
    ReducedPotentialDataset,
    ReducedWorkDataset,
    thermodynamic_integration,
    ThermodynamicDerivativeDataset,
)
from .._construct import _identifier
from ..experiments._models import ThermodynamicConvention


@dataclass(frozen=True, slots=True)
class ProteinFreeEnergyEstimate:
    native_result: FreeEnergyResult
    thermal_energy: float
    protocol_id: str
    state_ids: tuple[str, ...]
    bias_ids: tuple[str | None, ...]

    @property
    def free_energies(self):
        return self.native_result.free_energies * self.thermal_energy

    @property
    def covariance(self):
        return self.native_result.covariance * self.thermal_energy**2

    @property
    def differences(self):
        return self.native_result.differences * self.thermal_energy

    @property
    def standard_errors(self):
        return self.native_result.standard_errors * self.thermal_energy


@dataclass(frozen=True, slots=True)
class ProteinFreeEnergyWorkflow:
    """Protein-domain admission around authenticated FEP/TI/BAR/MBAR data.

    The workflow converts physical energy observations into canonical reduced
    datasets, then verifies every state, measure, run, producer and bias identity
    before dispatching an estimator. Configuration weights never imply path
    weights, and free-form decorrelation claims are not accepted.
    """

    state_ids: tuple[str, ...]
    composition_id: str
    temperature_kelvin: float
    convention: ThermodynamicConvention
    ensemble_source_id: str
    measure_id: str
    run_id: str
    bias_ids: tuple[str | None, ...] = ()
    selection_plan: FreeEnergySelectionPlan = field(
        default_factory=FreeEnergySelectionPlan
    )

    def __post_init__(self):
        states = tuple(self.state_ids)
        biases = (None,) * len(states) if not self.bias_ids else tuple(self.bias_ids)
        if len(states) < 2 or len(set(states)) != len(states):
            raise ValueError(
                "At least two independently identified thermodynamic states are required."
            )
        for value in (
            *states,
            self.composition_id,
            self.ensemble_source_id,
            self.measure_id,
            self.run_id,
            *(value for value in biases if value is not None),
        ):
            _identifier(value, "free-energy evidence")
        if len(biases) != len(states):
            raise ValueError(
                "Protein free-energy bias identities must align with states."
            )
        if not np.isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("Temperature must be positive Kelvin.")
        if not isinstance(self.convention, ThermodynamicConvention):
            raise TypeError("convention must be ThermodynamicConvention.")
        if not isinstance(self.selection_plan, FreeEnergySelectionPlan):
            raise TypeError("selection_plan must be FreeEnergySelectionPlan.")
        object.__setattr__(self, "state_ids", states)
        object.__setattr__(self, "bias_ids", biases)
        object.__setattr__(self, "temperature_kelvin", float(self.temperature_kelvin))

    @property
    def thermal_energy(self):
        return self.convention.thermal_constant * self.temperature_kelvin

    @property
    def inverse_temperature(self) -> float:
        return 1.0 / self.thermal_energy

    @property
    def reduced_convention_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "protein-reduced-potential-convention",
                "definition": "u=beta*U",
                "energy_unit": self.convention.energy_unit.unit_id,
                "thermal_constant": float(self.convention.thermal_constant).hex(),
            }
        )

    def fingerprint(self):
        return canonical_fingerprint(
            {
                "kind": "protein-free-energy-protocol",
                "states": self.state_ids,
                "composition": self.composition_id,
                "temperature": self.temperature_kelvin,
                "unit": self.convention.energy_unit.unit_id,
                "reduced_convention": self.reduced_convention_id,
                "source": self.ensemble_source_id,
                "measure": self.measure_id,
                "run": self.run_id,
                "biases": self.bias_ids,
                "selection_plan": self.selection_plan.plan_id,
            }
        )

    def _wrap(self, result: FreeEnergyResult, /) -> ProteinFreeEnergyEstimate:
        return ProteinFreeEnergyEstimate(
            result,
            self.thermal_energy,
            self.fingerprint(),
            result.state_ids,
            self.bias_ids,
        )

    def _reduced(self, values, unit):
        return (
            jnp.asarray(values)
            * float(conversion_factor(unit, self.convention.energy_unit))
            / self.thermal_energy
        )

    def reduced_work_dataset(
        self,
        values,
        coverage,
        sample_active,
        source_state,
        destination_state,
        chain_index,
        draw_index,
        repeat_index,
        dependence_group_index,
        /,
        *,
        potential_ids: Sequence[str],
        work_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        work_kind="equilibrium-difference",
        mapping_id: str | None = None,
    ) -> ReducedWorkDataset:
        if len(self.state_ids) != 2:
            raise ValueError("Reduced work datasets require exactly two workflow states.")
        return ReducedWorkDataset(
            values,
            coverage,
            sample_active,
            source_state,
            destination_state,
            chain_index,
            draw_index,
            repeat_index,
            dependence_group_index,
            state_ids=self.state_ids,
            measure_ids=(self.measure_id, self.measure_id),
            potential_ids=potential_ids,
            producer_id=self.ensemble_source_id,
            run_id=self.run_id,
            work_id=work_id,
            work_kind=work_kind,
            inverse_temperature=self.inverse_temperature,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            mapping_id=mapping_id,
            bias_ids=self.bias_ids,
            unit_id="1",
        )

    def work_dataset(
        self,
        values,
        coverage,
        sample_active,
        source_state,
        destination_state,
        chain_index,
        draw_index,
        repeat_index,
        dependence_group_index,
        /,
        *,
        energy_unit,
        potential_ids: Sequence[str],
        work_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        work_kind="equilibrium-difference",
        mapping_id: str | None = None,
    ) -> ReducedWorkDataset:
        """Convert physical directed work to an authenticated reduced dataset."""

        return self.reduced_work_dataset(
            self._reduced(values, energy_unit),
            coverage,
            sample_active,
            source_state,
            destination_state,
            chain_index,
            draw_index,
            repeat_index,
            dependence_group_index,
            work_id=work_id,
            potential_ids=potential_ids,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            work_kind=work_kind,
            mapping_id=mapping_id,
        )

    def potential_dataset(
        self,
        values,
        coverage,
        sample_active,
        origin_state,
        chain_index,
        draw_index,
        repeat_index,
        dependence_group_index,
        /,
        *,
        energy_unit,
        potential_ids: Sequence[str],
        inverse_temperatures,
        reduced_convention_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
    ) -> ReducedPotentialDataset:
        """Convert physical cross-potentials to an authenticated dense dataset."""
        raw_beta = np.asarray(inverse_temperatures)
        if not np.issubdtype(raw_beta.dtype, np.floating):
            raw_beta = raw_beta.astype("float64")
        beta = raw_beta.astype("float64", copy=False)
        expected_beta = self.inverse_temperature
        beta_tolerance = max(1.0e-12, 128.0 * np.finfo(raw_beta.dtype).eps)
        if beta.shape != (len(self.state_ids),) or not np.allclose(
            beta, expected_beta, rtol=beta_tolerance, atol=0.0
        ):
            raise ValueError(
                "Protein physical projection requires one beta matching workflow temperature."
            )
        if reduced_convention_id != self.reduced_convention_id:
            raise ValueError("Reduced-potential convention does not match this workflow.")

        return ReducedPotentialDataset(
            self._reduced(values, energy_unit),
            coverage,
            sample_active,
            origin_state,
            chain_index,
            draw_index,
            repeat_index,
            dependence_group_index,
            state_ids=self.state_ids,
            potential_ids=potential_ids,
            inverse_temperatures=beta,
            reduced_convention_id=reduced_convention_id,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            measure_id=self.measure_id,
            producer_id=self.ensemble_source_id,
            run_id=self.run_id,
            bias_ids=self.bias_ids,
            unit_id="1",
        )

    def derivative_dataset(
        self,
        values,
        coverage,
        sample_active,
        chain_index,
        draw_index,
        repeat_index,
        dependence_group_index,
        path_parameter,
        /,
        *,
        energy_unit,
        potential_ids: Sequence[str],
        derivative_id: str,
        control_path_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
    ) -> ThermodynamicDerivativeDataset:
        """Convert complete physical dU/dpath observations to reduced form."""

        return ThermodynamicDerivativeDataset(
            self._reduced(values, energy_unit),
            coverage,
            sample_active,
            chain_index,
            draw_index,
            repeat_index,
            dependence_group_index,
            path_parameter,
            state_ids=self.state_ids,
            potential_ids=potential_ids,
            measure_id=self.measure_id,
            producer_id=self.ensemble_source_id,
            run_id=self.run_id,
            derivative_id=derivative_id,
            control_path_id=control_path_id,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            bias_ids=self.bias_ids,
            unit_id="1",
        )

    def _verify(self, dataset, /) -> None:
        if dataset.state_ids != self.state_ids:
            raise ValueError(
                "Dataset state identities do not match this protein workflow."
            )
        if (
            dataset.producer_id != self.ensemble_source_id
            or dataset.run_id != self.run_id
        ):
            raise ValueError(
                "Dataset producer/run identity does not match this workflow."
            )
        if dataset.bias_ids != self.bias_ids:
            raise ValueError("Dataset bias identities do not match this workflow.")
        measures = (
            dataset.measure_ids
            if isinstance(dataset, ReducedWorkDataset)
            else (dataset.measure_id,) * len(dataset.state_ids)
        )
        if any(measure != self.measure_id for measure in measures):
            raise ValueError(
                "Dataset configurational measure does not match this workflow."
            )
        if isinstance(dataset, ReducedWorkDataset):
            beta = float(dataset.inverse_temperature)
            beta_tolerance = max(1.0e-12, 128.0 * np.finfo(np.float64).eps)
            if not np.isclose(
                beta,
                self.inverse_temperature,
                rtol=beta_tolerance,
                atol=0.0,
            ):
                raise ValueError("Reduced-work beta does not match this workflow.")
        if isinstance(dataset, ReducedPotentialDataset):
            beta = np.asarray(dataset.inverse_temperatures)
            beta_tolerance = max(1.0e-12, 128.0 * np.finfo(beta.dtype).eps)
            if (
                dataset.reduced_convention_id != self.reduced_convention_id
                or not np.allclose(
                    beta,
                    self.inverse_temperature,
                    rtol=beta_tolerance,
                    atol=0.0,
                )
            ):
                raise ValueError(
                    "Reduced-potential beta/convention does not match this workflow."
                )

    def _selection(self, selection, /):
        return self.selection_plan if selection is None else selection

    def fep(
        self,
        dataset: ReducedWorkDataset,
        selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
        /,
        *,
        key=None,
    ) -> ProteinFreeEnergyEstimate:
        if not isinstance(dataset, ReducedWorkDataset):
            raise TypeError("dataset must be ReducedWorkDataset.")
        self._verify(dataset)
        return self._wrap(
            free_energy_perturbation(dataset, self._selection(selection), key=key)
        )

    def bar(
        self,
        dataset: ReducedWorkDataset,
        selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
        /,
        *,
        key=None,
        **solver_options,
    ) -> ProteinFreeEnergyEstimate:
        if not isinstance(dataset, ReducedWorkDataset):
            raise TypeError("dataset must be ReducedWorkDataset.")
        self._verify(dataset)
        return self._wrap(
            bennett_acceptance_ratio(
                dataset,
                self._selection(selection),
                key=key,
                **solver_options,
            )
        )

    def ti(
        self,
        dataset: ThermodynamicDerivativeDataset,
        selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
        /,
        *,
        key=None,
    ) -> ProteinFreeEnergyEstimate:
        if not isinstance(dataset, ThermodynamicDerivativeDataset):
            raise TypeError("dataset must be ThermodynamicDerivativeDataset.")
        self._verify(dataset)
        return self._wrap(
            thermodynamic_integration(dataset, self._selection(selection), key=key)
        )

    def mbar(
        self,
        dataset: ReducedPotentialDataset,
        selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
        /,
        *,
        key=None,
        **solver_options,
    ) -> ProteinFreeEnergyEstimate:
        if not isinstance(dataset, ReducedPotentialDataset):
            raise TypeError("dataset must be ReducedPotentialDataset.")
        self._verify(dataset)
        return self._wrap(
            multistate_bennett_acceptance_ratio(
                dataset,
                self._selection(selection),
                key=key,
                **solver_options,
            )
        )

    def targeted(
        self,
        dataset: ReducedWorkDataset,
        selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
        /,
        *,
        key=None,
        **solver_options,
    ) -> ProteinFreeEnergyEstimate:
        """Analyze authenticated targeted-map work without reinterpreting raw arrays."""

        if (
            not isinstance(dataset, ReducedWorkDataset)
            or dataset.work_kind != "targeted-map"
        ):
            raise TypeError("dataset must be a targeted-map ReducedWorkDataset.")
        self._verify(dataset)
        has_reverse = bool(jnp.any(dataset.sample_active & (dataset.source_state == 1)))
        result = (
            bennett_acceptance_ratio(
                dataset,
                self._selection(selection),
                key=key,
                **solver_options,
            )
            if has_reverse
            else free_energy_perturbation(
                dataset,
                self._selection(selection),
                key=key,
            )
        )
        return self._wrap(result)


__all__ = ["ProteinFreeEnergyEstimate", "ProteinFreeEnergyWorkflow"]
