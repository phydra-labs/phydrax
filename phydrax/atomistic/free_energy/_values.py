#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authenticated free-energy projections and analytic corrections."""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._alchemical import AlchemicalControlKind, PreparedControlledHamiltonian
from .._thermodynamic import PreparedThermodynamicStateTable


DESTINATION_MINUS_SOURCE = "destination-minus-source"


class FreeEnergyCorrectionKind(StrEnum):
    RESTRAINT = "restraint"
    STANDARD_STATE = "standard-state"
    SYMMETRY = "symmetry"
    FINITE_SIZE = "finite-size"
    ANALYTIC = "analytic"


def _identity(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _estimate(value: ArrayLike, variance: ArrayLike, /) -> tuple[Array, Array]:
    estimate = jnp.asarray(value)
    uncertainty = jnp.asarray(variance, dtype=estimate.dtype)
    if estimate.shape != () or uncertainty.shape != () or estimate.dtype.kind != "f":
        raise TypeError("Free-energy values and variances must be floating scalars.")
    if not bool(jnp.isfinite(estimate)) or not bool(jnp.isfinite(uncertainty)):
        raise ValueError("Free-energy values and variances must be finite.")
    if float(uncertainty) < 0.0:
        raise ValueError("Free-energy variances must be non-negative.")
    return estimate, uncertainty


class FreeEnergyStatePlan(StrictModule, NonTrainableState):
    """One table state with system-derived charge and exact measure evidence."""

    hamiltonian: PreparedControlledHamiltonian
    thermodynamic: PreparedThermodynamicStateTable
    state_index: int = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    bias_id: str | None = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    neutral_control_evidence: bool = eqx.field(static=True)
    charge_evidence_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: PreparedControlledHamiltonian,
        thermodynamic: PreparedThermodynamicStateTable,
        state_index: int,
        /,
    ):
        if not isinstance(hamiltonian, PreparedControlledHamiltonian):
            raise TypeError("hamiltonian must be PreparedControlledHamiltonian.")
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be PreparedThermodynamicStateTable.")
        index = int(state_index)
        if index < 0 or index >= thermodynamic.state_count:
            raise ValueError("state_index must identify a prepared thermodynamic state.")
        if (
            thermodynamic.program_id != hamiltonian.prepared_id
            or thermodynamic.system_id != hamiltonian.system.prepared_id
            or thermodynamic.control_ids != hamiltonian.control_ids
            or thermodynamic.control_layout_id != hamiltonian.control_layout_id
            or thermodynamic.unit_system_id
            != hamiltonian.system.plan.units.unit_system_id
        ):
            raise ValueError("State table and controlled Hamiltonian identities differ.")
        neutral_flags = hamiltonian.partition.preparation.neutral_electrostatic_regions
        neutral = all(
            neutral_flags[control_index]
            for control_index, kind in enumerate(hamiltonian.plan.schedule.control_kinds)
            if kind is AlchemicalControlKind.ELECTROSTATICS
        )
        charge_evidence = canonical_fingerprint(
            {
                "kind": "alchemical-charge-evidence",
                "system": hamiltonian.system.prepared_id,
                "partition": hamiltonian.partition.prepared_id,
                "control_ids": list(hamiltonian.control_ids),
                "neutral_electrostatic_regions": list(neutral_flags),
                "charges": array_tree_fingerprint(
                    np.asarray(hamiltonian.system.plan.charges)
                ),
            }
        )
        self.hamiltonian = hamiltonian
        self.thermodynamic = thermodynamic
        self.state_index = index
        self.state_id = thermodynamic.state_ids[index]
        self.potential_id = thermodynamic.potential_ids[index]
        self.measure_id = thermodynamic.phase_space_measure_id
        self.bias_id = thermodynamic.bias_ids[index]
        self.unit_system_id = thermodynamic.unit_system_id
        self.system_id = thermodynamic.system_id
        self.neutral_control_evidence = neutral
        self.charge_evidence_id = charge_evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-energy-state-plan",
                "hamiltonian": hamiltonian.prepared_id,
                "thermodynamic": thermodynamic.table_id,
                "state_index": index,
                "state": self.state_id,
                "potential": self.potential_id,
                "measure": self.measure_id,
                "bias": self.bias_id,
                "units": self.unit_system_id,
                "charge_evidence": charge_evidence,
            }
        )

    def project(self, result: Any, dataset: Any, /) -> "FreeEnergyStateResult":
        (
            state_ids,
            potential_ids,
            measure_ids,
            bias_ids,
            unit_system_id,
            _,
            qualification_id,
            sampling_exact,
            sampling_bias_bound,
        ) = _authenticated_metadata(result, dataset)
        index = _validate_state_metadata(
            self,
            state_ids,
            potential_ids,
            measure_ids,
            bias_ids,
            unit_system_id,
        )
        value, variance = _estimate(
            result.free_energies[index], result.covariance[index, index]
        )
        identity = canonical_fingerprint(
            {
                "kind": "free-energy-state-result",
                "plan": self.plan_id,
                "analysis": result.analysis_id,
                "dataset": dataset.dataset_id,
                "qualification": qualification_id,
                "sampling_exact": sampling_exact,
                "sampling_bias_bound": float(sampling_bias_bound).hex(),
                "value": float(value).hex(),
                "variance": float(variance).hex(),
            }
        )
        return FreeEnergyStateResult(
            value=value,
            variance=variance,
            state_id=self.state_id,
            potential_id=self.potential_id,
            measure_id=self.measure_id,
            bias_id=self.bias_id,
            unit_system_id=self.unit_system_id,
            charge_evidence_id=self.charge_evidence_id,
            dataset_id=dataset.dataset_id,
            analysis_id=result.analysis_id,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            successful=True,
            result_id=identity,
        )


class FreeEnergyStateResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    state_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    bias_id: str | None = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    charge_evidence_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class FreeEnergyProtocolLegResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    source_state_id: str = eqx.field(static=True)
    destination_state_id: str = eqx.field(static=True)
    source_potential_id: str = eqx.field(static=True)
    destination_potential_id: str = eqx.field(static=True)
    source_measure_id: str = eqx.field(static=True)
    destination_measure_id: str = eqx.field(static=True)
    source_bias_id: str | None = eqx.field(static=True)
    destination_bias_id: str | None = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    source_charge_evidence_id: str = eqx.field(static=True)
    destination_charge_evidence_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    mapping_id: str | None = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _authenticated_metadata(result: Any, dataset: Any, /):
    from ...uq._free_energy import (
        FreeEnergyResult,
        FreeEnergyStatus,
        ReducedPotentialDataset,
        ReducedWorkDataset,
    )

    if not isinstance(result, FreeEnergyResult):
        raise TypeError("result must be an authenticated UQ FreeEnergyResult.")
    if not isinstance(dataset, (ReducedPotentialDataset, ReducedWorkDataset)):
        raise TypeError(
            "dataset must be an authenticated reduced-potential or work dataset."
        )
    if result.dataset_id != dataset.dataset_id:
        raise ValueError("Free-energy result and dataset identities differ.")
    if int(np.asarray(result.numerical_status)) != int(FreeEnergyStatus.SUCCESS) or int(
        np.asarray(result.statistical_status)
    ) != int(FreeEnergyStatus.SUCCESS):
        raise ValueError(
            "Free-energy result is not numerically and statistically successful."
        )
    if tuple(result.state_ids) != tuple(dataset.state_ids):
        raise ValueError("Free-energy result and dataset state identities differ.")
    if isinstance(dataset, ReducedPotentialDataset):
        measures = (dataset.measure_id,) * len(dataset.state_ids)
        mapping_id = None
        unit_system_id = dataset.unit_system_id
    else:
        measures = tuple(dataset.measure_ids)
        mapping_id = dataset.mapping_id
        unit_system_id = dataset.unit_system_id
    return (
        tuple(dataset.state_ids),
        tuple(dataset.potential_ids),
        measures,
        tuple(dataset.bias_ids),
        unit_system_id,
        mapping_id,
        dataset.qualification_id,
        dataset.sampling_exact,
        dataset.sampling_bias_bound,
    )


def _validate_state_metadata(
    plan: FreeEnergyStatePlan,
    state_ids,
    potential_ids,
    measure_ids,
    bias_ids,
    unit_system_id,
    /,
) -> int:
    if plan.state_id not in state_ids:
        raise ValueError("Bound state is absent from the authenticated analysis.")
    index = state_ids.index(plan.state_id)
    if (
        potential_ids[index] != plan.potential_id
        or measure_ids[index] != plan.measure_id
        or bias_ids[index] != plan.bias_id
        or unit_system_id != plan.unit_system_id
        or not plan.neutral_control_evidence
    ):
        raise ValueError(
            "Potential, measure, bias, unit, or neutral-charge evidence does not match."
        )
    return index


class FreeEnergyProtocolLegPlan(StrictModule, NonTrainableState):
    """Project one successful analysis as destination minus source."""

    source: FreeEnergyStatePlan
    destination: FreeEnergyStatePlan
    environment_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: FreeEnergyStatePlan,
        destination: FreeEnergyStatePlan,
        environment_id: str,
        /,
    ):
        if not isinstance(source, FreeEnergyStatePlan) or not isinstance(
            destination, FreeEnergyStatePlan
        ):
            raise TypeError("Protocol legs require bound source and destination states.")
        if source.state_id == destination.state_id:
            raise ValueError(
                "Protocol leg endpoints must have distinct state identities."
            )
        if source.system_id != destination.system_id:
            raise ValueError("Protocol leg endpoints must share one atomistic system.")
        if source.unit_system_id != destination.unit_system_id:
            raise ValueError("Protocol leg endpoints must share one exact unit system.")
        if (
            not source.neutral_control_evidence
            or not destination.neutral_control_evidence
        ):
            raise ValueError(
                "Protocol leg endpoints require neutral charge-change evidence."
            )
        environment = _identity(environment_id, "environment_id")
        self.source = source
        self.destination = destination
        self.environment_id = environment
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-energy-protocol-leg",
                "source": source.plan_id,
                "destination": destination.plan_id,
                "environment": environment,
                "orientation": DESTINATION_MINUS_SOURCE,
            }
        )

    def project(self, result: Any, dataset: Any, /) -> FreeEnergyProtocolLegResult:
        (
            state_ids,
            potential_ids,
            measure_ids,
            bias_ids,
            unit_system_id,
            mapping_id,
            qualification_id,
            sampling_exact,
            sampling_bias_bound,
        ) = _authenticated_metadata(result, dataset)
        source_index = _validate_state_metadata(
            self.source,
            state_ids,
            potential_ids,
            measure_ids,
            bias_ids,
            unit_system_id,
        )
        destination_index = _validate_state_metadata(
            self.destination,
            state_ids,
            potential_ids,
            measure_ids,
            bias_ids,
            unit_system_id,
        )
        if not bool(np.asarray(result.connectivity)[source_index, destination_index]):
            raise ValueError("Protocol leg endpoints are disconnected in the analysis.")
        value = result.differences[source_index, destination_index]
        variance = result.standard_errors[source_index, destination_index] ** 2
        estimate, uncertainty = _estimate(value, variance)
        result_id = canonical_fingerprint(
            {
                "kind": "free-energy-protocol-leg-result",
                "plan": self.plan_id,
                "analysis": result.analysis_id,
                "dataset": dataset.dataset_id,
                "source_index": source_index,
                "destination_index": destination_index,
                "value": float(estimate).hex(),
                "variance": float(uncertainty).hex(),
                "mapping": mapping_id,
                "qualification": qualification_id,
                "sampling_exact": sampling_exact,
                "sampling_bias_bound": float(sampling_bias_bound).hex(),
            }
        )
        return FreeEnergyProtocolLegResult(
            value=estimate,
            variance=uncertainty,
            source_state_id=self.source.state_id,
            destination_state_id=self.destination.state_id,
            source_potential_id=self.source.potential_id,
            destination_potential_id=self.destination.potential_id,
            source_measure_id=self.source.measure_id,
            destination_measure_id=self.destination.measure_id,
            source_bias_id=self.source.bias_id,
            destination_bias_id=self.destination.bias_id,
            unit_system_id=self.source.unit_system_id,
            source_charge_evidence_id=self.source.charge_evidence_id,
            destination_charge_evidence_id=self.destination.charge_evidence_id,
            dataset_id=dataset.dataset_id,
            analysis_id=result.analysis_id,
            mapping_id=mapping_id,
            orientation=DESTINATION_MINUS_SOURCE,
            qualification_id=qualification_id,
            sampling_exact=sampling_exact,
            sampling_bias_bound=sampling_bias_bound,
            successful=True,
            plan_id=self.plan_id,
            result_id=result_id,
        )


class FreeEnergyCorrectionResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    correction_kind: FreeEnergyCorrectionKind = eqx.field(static=True)
    correction_name: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class FreeEnergyCorrectionPlan(StrictModule, NonTrainableState):
    """Abstract explicit signed correction in dimensionless free-energy units."""

    __strict_abstract__ = True
    correction_kind: eqx.AbstractVar[FreeEnergyCorrectionKind]
    correction_name: eqx.AbstractVar[str]
    convention: eqx.AbstractVar[str]
    plan_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def result(
        self,
        value: ArrayLike,
        variance: ArrayLike = 0.0,
        /,
        *,
        evidence_id: str,
        successful: bool = True,
    ) -> FreeEnergyCorrectionResult:
        raise NotImplementedError


def _correction_fields(kind, name: str, convention: str, /):
    name_ = _identity(name, "correction_name")
    convention_ = _identity(convention, "convention")
    plan_id = canonical_fingerprint(
        {
            "kind": "free-energy-correction-plan",
            "correction_kind": kind.value,
            "name": name_,
            "convention": convention_,
            "orientation": DESTINATION_MINUS_SOURCE,
        }
    )
    return kind, name_, convention_, plan_id


def _correction_result(plan, value, variance, evidence_id, successful, /):
    estimate, uncertainty = _estimate(value, variance)
    evidence = _identity(evidence_id, "evidence_id")
    accepted = bool(successful)
    return FreeEnergyCorrectionResult(
        value=estimate,
        variance=uncertainty,
        correction_kind=plan.correction_kind,
        correction_name=plan.correction_name,
        evidence_id=evidence,
        orientation=DESTINATION_MINUS_SOURCE,
        successful=accepted,
        plan_id=plan.plan_id,
        result_id=canonical_fingerprint(
            {
                "kind": "free-energy-correction-result",
                "plan": plan.plan_id,
                "value": float(estimate).hex(),
                "variance": float(uncertainty).hex(),
                "evidence": evidence,
                "successful": accepted,
            }
        ),
    )


class RestraintCorrectionPlan(FreeEnergyCorrectionPlan):
    correction_kind: FreeEnergyCorrectionKind = eqx.field(static=True)
    correction_name: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, restraint_id: str, /, *, convention: str):
        (
            self.correction_kind,
            self.correction_name,
            self.convention,
            self.plan_id,
        ) = _correction_fields(
            FreeEnergyCorrectionKind.RESTRAINT,
            _identity(restraint_id, "restraint_id"),
            convention,
        )

    def result(self, value, variance=0.0, /, *, evidence_id, successful=True):
        return _correction_result(self, value, variance, evidence_id, successful)


class StandardStateCorrectionPlan(FreeEnergyCorrectionPlan):
    correction_kind: FreeEnergyCorrectionKind = eqx.field(static=True)
    correction_name: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, standard_state_id: str, /, *, convention: str):
        (
            self.correction_kind,
            self.correction_name,
            self.convention,
            self.plan_id,
        ) = _correction_fields(
            FreeEnergyCorrectionKind.STANDARD_STATE,
            _identity(standard_state_id, "standard_state_id"),
            convention,
        )

    def result(self, value, variance=0.0, /, *, evidence_id, successful=True):
        return _correction_result(self, value, variance, evidence_id, successful)


class SymmetryCorrectionPlan(FreeEnergyCorrectionPlan):
    correction_kind: FreeEnergyCorrectionKind = eqx.field(static=True)
    correction_name: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    symmetry_number: int = eqx.field(static=True)

    def __init__(self, symmetry_number: int, /, *, convention: str):
        number = int(symmetry_number)
        if number < 1:
            raise ValueError("symmetry_number must be positive.")
        (
            self.correction_kind,
            self.correction_name,
            self.convention,
            self.plan_id,
        ) = _correction_fields(
            FreeEnergyCorrectionKind.SYMMETRY,
            f"symmetry-{number}",
            convention,
        )
        self.symmetry_number = number

    def result(self, value, variance=0.0, /, *, evidence_id, successful=True):
        return _correction_result(self, value, variance, evidence_id, successful)

    def analytic_result(self, /, *, evidence_id: str) -> FreeEnergyCorrectionResult:
        return self.result(
            -np.log(float(self.symmetry_number)),
            0.0,
            evidence_id=evidence_id,
        )


__all__ = [
    "DESTINATION_MINUS_SOURCE",
    "FreeEnergyCorrectionKind",
    "FreeEnergyCorrectionPlan",
    "FreeEnergyCorrectionResult",
    "FreeEnergyProtocolLegPlan",
    "FreeEnergyProtocolLegResult",
    "FreeEnergyStatePlan",
    "FreeEnergyStateResult",
    "RestraintCorrectionPlan",
    "StandardStateCorrectionPlan",
    "SymmetryCorrectionPlan",
]
