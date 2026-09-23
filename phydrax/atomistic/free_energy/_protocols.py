#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Signed thermodynamic-cycle contracts with explicit covariance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._values import (
    DESTINATION_MINUS_SOURCE,
    FreeEnergyCorrectionKind,
    FreeEnergyCorrectionPlan,
    FreeEnergyCorrectionResult,
    FreeEnergyProtocolLegPlan,
    FreeEnergyProtocolLegResult,
    FreeEnergyStatePlan,
    FreeEnergyStateResult,
)


class SeparatedTopologyResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    component_values: Array
    covariance: Array
    weights: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    formula: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class NeutralAbsoluteSolvationResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    component_values: Array
    covariance: Array
    weights: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    formula: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class AbsoluteBindingResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    component_values: Array
    covariance: Array
    weights: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    formula: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MappedRelativeSolvationResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    component_values: Array
    covariance: Array
    weights: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    formula: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MappedRelativeBindingResult(StrictModule, NonTrainableState):
    value: Array
    variance: Array
    component_values: Array
    covariance: Array
    weights: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    component_result_ids: tuple[str, ...] = eqx.field(static=True)
    formula: str = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


_Result = TypeVar("_Result", bound=StrictModule)
_Component = Any


def _component_lineage(value, /) -> tuple | None:
    if isinstance(value, FreeEnergyStateResult):
        return (
            value.dataset_id,
            value.analysis_id,
            value.qualification_id,
            value.sampling_exact,
            value.sampling_bias_bound,
            value.unit_system_id,
            value.measure_id,
            None,
            None,
        )
    if isinstance(value, FreeEnergyProtocolLegResult):
        return (
            value.dataset_id,
            value.analysis_id,
            value.qualification_id,
            value.sampling_exact,
            value.sampling_bias_bound,
            value.unit_system_id,
            (value.source_measure_id, value.destination_measure_id),
            value.mapping_id,
            value.orientation,
        )
    if isinstance(value, MappedRelativeTransformationResult):
        return _component_lineage(value.leg)
    if isinstance(value, FreeEnergyCorrectionResult):
        return None
    raise TypeError(f"Unsupported free-energy protocol component {type(value).__name__}.")


def _combine(
    result_type: type[_Result],
    plan_id: str,
    component_names: Sequence[str],
    components: Sequence[_Component],
    weights: Sequence[float],
    covariance: ArrayLike,
    formula: str,
    /,
) -> _Result:
    names = tuple(component_names)
    values_ = tuple(components)
    factors = np.asarray(weights, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    count = len(values_)
    lineages = tuple(
        lineage for value in values_ if (lineage := _component_lineage(value)) is not None
    )
    if lineages and any(value != lineages[0] for value in lineages[1:]):
        raise ValueError(
            "Free-energy protocol components must share dataset, analysis, qualification, sampling, unit, measure, mapping, and orientation lineage."
        )
    if len(names) != count or factors.shape != (count,):
        raise ValueError("Protocol component names and weights must align.")
    if matrix.shape != (count, count):
        raise ValueError("Protocol covariance must be square over every component.")
    if np.any(~np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError("Protocol covariance must be finite and symmetric.")
    component_variances = np.asarray([float(value.variance) for value in values_])
    if not np.allclose(np.diag(matrix), component_variances, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError(
            "Protocol covariance diagonal must equal the authenticated component variances."
        )
    tolerance = (
        128.0 * np.finfo(matrix.dtype).eps * max(1.0, float(np.max(np.abs(matrix))))
    )
    if count and float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
        raise ValueError("Protocol covariance must be positive semidefinite.")
    component_values = jnp.stack(tuple(jnp.asarray(value.value) for value in values_))
    covariance_ = jnp.asarray(matrix, dtype=component_values.dtype)
    weights_ = jnp.asarray(factors, dtype=component_values.dtype)
    total = jnp.sum(weights_ * component_values)
    variance = contract("i,ij,j->", weights_, covariance_, weights_)
    accepted = all(value.successful for value in values_) and bool(
        jnp.isfinite(total) & jnp.isfinite(variance) & (variance >= 0.0)
    )
    result_ids = tuple(value.result_id for value in values_)
    result_id = canonical_fingerprint(
        {
            "kind": "free-energy-protocol-result",
            "plan": plan_id,
            "components": list(result_ids),
            "weights": factors.tolist(),
            "covariance": array_tree_fingerprint(matrix),
            "formula": formula,
            "orientation": DESTINATION_MINUS_SOURCE,
            "successful": accepted,
        }
    )
    return result_type(
        value=total,
        variance=variance,
        component_values=component_values,
        covariance=covariance_,
        weights=weights_,
        component_names=names,
        component_result_ids=result_ids,
        formula=formula,
        orientation=DESTINATION_MINUS_SOURCE,
        successful=accepted,
        plan_id=plan_id,
        result_id=result_id,
    )


def _corrections(
    plans: Sequence[FreeEnergyCorrectionPlan],
    results: Sequence[FreeEnergyCorrectionResult],
    /,
) -> tuple[FreeEnergyCorrectionResult, ...]:
    expected = tuple(plans)
    values = tuple(results)
    if len(expected) != len(values) or any(
        plan.plan_id != result.plan_id
        for plan, result in zip(expected, values, strict=True)
    ):
        raise ValueError("Correction results must match their protocol correction plans.")
    return values


class SeparatedTopologyPlan(StrictModule, NonTrainableState):
    """Environment decoupling D_env = f_off - f_on on one topology support."""

    coupled: FreeEnergyStatePlan
    decoupled: FreeEnergyStatePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coupled: FreeEnergyStatePlan,
        decoupled: FreeEnergyStatePlan,
        /,
    ):
        if not isinstance(coupled, FreeEnergyStatePlan) or not isinstance(
            decoupled, FreeEnergyStatePlan
        ):
            raise TypeError(
                "Separated topology requires coupled and decoupled state plans."
            )
        if coupled.measure_id != decoupled.measure_id:
            raise ValueError(
                "Separated topology states require one exact measure identity."
            )
        if not coupled.neutral_control_evidence or not decoupled.neutral_control_evidence:
            raise ValueError(
                "Separated topology states require neutral bound-system evidence."
            )
        self.coupled = coupled
        self.decoupled = decoupled
        self.plan_id = canonical_fingerprint(
            {
                "kind": "separated-topology-free-energy",
                "coupled": coupled.plan_id,
                "decoupled": decoupled.plan_id,
                "formula": "f_off - f_on",
            }
        )

    def evaluate(
        self,
        coupled: FreeEnergyStateResult,
        decoupled: FreeEnergyStateResult,
        covariance: ArrayLike,
        /,
    ) -> SeparatedTopologyResult:
        coupled_matches = (
            coupled.state_id == self.coupled.state_id
            and coupled.potential_id == self.coupled.potential_id
            and coupled.measure_id == self.coupled.measure_id
            and coupled.bias_id == self.coupled.bias_id
            and coupled.unit_system_id == self.coupled.unit_system_id
            and coupled.charge_evidence_id == self.coupled.charge_evidence_id
        )
        decoupled_matches = (
            decoupled.state_id == self.decoupled.state_id
            and decoupled.potential_id == self.decoupled.potential_id
            and decoupled.measure_id == self.decoupled.measure_id
            and decoupled.bias_id == self.decoupled.bias_id
            and decoupled.unit_system_id == self.decoupled.unit_system_id
            and decoupled.charge_evidence_id == self.decoupled.charge_evidence_id
        )
        if (
            not coupled_matches
            or not decoupled_matches
            or coupled.dataset_id != decoupled.dataset_id
            or coupled.analysis_id != decoupled.analysis_id
        ):
            raise ValueError("Separated-topology state results do not match the plan.")
        return _combine(
            SeparatedTopologyResult,
            self.plan_id,
            ("f_on", "f_off"),
            (coupled, decoupled),
            (-1.0, 1.0),
            covariance,
            "D_env = f_off - f_on",
        )


class NeutralAbsoluteSolvationPlan(StrictModule, NonTrainableState):
    """Neutral absolute solvation: D_vac - D_solv + corrections."""

    vacuum_decoupling: FreeEnergyProtocolLegPlan
    solvent_decoupling: FreeEnergyProtocolLegPlan
    corrections: tuple[FreeEnergyCorrectionPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vacuum_decoupling: FreeEnergyProtocolLegPlan,
        solvent_decoupling: FreeEnergyProtocolLegPlan,
        /,
        *,
        corrections: Sequence[FreeEnergyCorrectionPlan] = (),
    ):
        if not isinstance(vacuum_decoupling, FreeEnergyProtocolLegPlan) or not isinstance(
            solvent_decoupling, FreeEnergyProtocolLegPlan
        ):
            raise TypeError(
                "Absolute solvation requires vacuum and solvent protocol legs."
            )
        correction_plans = tuple(corrections)
        if any(
            not isinstance(value, FreeEnergyCorrectionPlan) for value in correction_plans
        ):
            raise TypeError("corrections must contain FreeEnergyCorrectionPlan values.")
        self.vacuum_decoupling = vacuum_decoupling
        self.solvent_decoupling = solvent_decoupling
        self.corrections = correction_plans
        self.plan_id = canonical_fingerprint(
            {
                "kind": "neutral-absolute-solvation-plan",
                "vacuum": vacuum_decoupling.plan_id,
                "solvent": solvent_decoupling.plan_id,
                "corrections": [value.plan_id for value in correction_plans],
                "formula": "D_vac - D_solv + C",
            }
        )

    def evaluate(
        self,
        vacuum_decoupling: FreeEnergyProtocolLegResult,
        solvent_decoupling: FreeEnergyProtocolLegResult,
        corrections: Sequence[FreeEnergyCorrectionResult],
        covariance: ArrayLike,
        /,
    ) -> NeutralAbsoluteSolvationResult:
        if (
            vacuum_decoupling.plan_id != self.vacuum_decoupling.plan_id
            or solvent_decoupling.plan_id != self.solvent_decoupling.plan_id
        ):
            raise ValueError("Absolute-solvation leg results do not match the plan.")
        correction_values = _corrections(self.corrections, corrections)
        components = (vacuum_decoupling, solvent_decoupling, *correction_values)
        return _combine(
            NeutralAbsoluteSolvationResult,
            self.plan_id,
            ("D_vac", "D_solv", *(value.correction_name for value in correction_values)),
            components,
            (1.0, -1.0, *(1.0 for _ in correction_values)),
            covariance,
            "solvation = D_vac - D_solv + C",
        )


class AbsoluteBindingPlan(StrictModule, NonTrainableState):
    """Neutral absolute binding with explicit restraint and standard-state terms."""

    solvent_decoupling: FreeEnergyProtocolLegPlan
    complex_decoupling: FreeEnergyProtocolLegPlan
    corrections: tuple[FreeEnergyCorrectionPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solvent_decoupling: FreeEnergyProtocolLegPlan,
        complex_decoupling: FreeEnergyProtocolLegPlan,
        corrections: Sequence[FreeEnergyCorrectionPlan],
        /,
    ):
        if not isinstance(
            solvent_decoupling, FreeEnergyProtocolLegPlan
        ) or not isinstance(complex_decoupling, FreeEnergyProtocolLegPlan):
            raise TypeError(
                "Absolute binding requires solvent and complex decoupling legs."
            )
        correction_plans = tuple(corrections)
        if any(
            not isinstance(value, FreeEnergyCorrectionPlan) for value in correction_plans
        ):
            raise TypeError("corrections must contain FreeEnergyCorrectionPlan values.")
        correction_kinds = {value.correction_kind for value in correction_plans}
        if (
            FreeEnergyCorrectionKind.RESTRAINT not in correction_kinds
            or FreeEnergyCorrectionKind.STANDARD_STATE not in correction_kinds
        ):
            raise ValueError(
                "Absolute binding requires explicit restraint and standard-state corrections."
            )
        self.solvent_decoupling = solvent_decoupling
        self.complex_decoupling = complex_decoupling
        self.corrections = correction_plans
        self.plan_id = canonical_fingerprint(
            {
                "kind": "absolute-binding-plan",
                "solvent": solvent_decoupling.plan_id,
                "complex": complex_decoupling.plan_id,
                "corrections": [value.plan_id for value in correction_plans],
                "formula": (
                    "D_solv - D_complex + C_restraint + C_standard_state + C_symmetry"
                ),
            }
        )

    def evaluate(
        self,
        solvent_decoupling: FreeEnergyProtocolLegResult,
        complex_decoupling: FreeEnergyProtocolLegResult,
        corrections: Sequence[FreeEnergyCorrectionResult],
        covariance: ArrayLike,
        /,
    ) -> AbsoluteBindingResult:
        if (
            solvent_decoupling.plan_id != self.solvent_decoupling.plan_id
            or complex_decoupling.plan_id != self.complex_decoupling.plan_id
        ):
            raise ValueError("Absolute-binding leg results do not match the plan.")
        correction_values = _corrections(self.corrections, corrections)
        components = (solvent_decoupling, complex_decoupling, *correction_values)
        return _combine(
            AbsoluteBindingResult,
            self.plan_id,
            (
                "D_solv",
                "D_complex",
                *(value.correction_name for value in correction_values),
            ),
            components,
            (1.0, -1.0, *(1.0 for _ in correction_values)),
            covariance,
            (
                "binding = D_solv - D_complex + C_restraint + C_standard_state + C_symmetry"
            ),
        )


class MappedRelativeTransformationResult(StrictModule, NonTrainableState):
    leg: FreeEnergyProtocolLegResult
    value: Array
    variance: Array
    mapping_id: str = eqx.field(static=True)
    mapping_plan_id: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MappedRelativeTransformationPlan(StrictModule, NonTrainableState):
    """One mapped neutral source-to-destination transformation leg."""

    leg: FreeEnergyProtocolLegPlan
    mapping_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        leg: FreeEnergyProtocolLegPlan,
        mapping_id: str,
        /,
    ):
        if not isinstance(leg, FreeEnergyProtocolLegPlan):
            raise TypeError("Mapped transformations require a protocol leg.")
        mapping = str(mapping_id).strip()
        if not mapping:
            raise ValueError("mapping_id must be non-empty.")
        if (
            not leg.source.neutral_control_evidence
            or not leg.destination.neutral_control_evidence
        ):
            raise ValueError(
                "Mapped relative transformations require neutral bound-system evidence."
            )
        self.leg = leg
        self.mapping_id = mapping
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-relative-transformation",
                "leg": leg.plan_id,
                "mapping": mapping,
                "orientation": DESTINATION_MINUS_SOURCE,
            }
        )

    def project(
        self,
        result,
        dataset,
        /,
    ) -> MappedRelativeTransformationResult:
        leg_result = self.leg.project(result, dataset)
        if leg_result.mapping_id != self.mapping_id:
            raise ValueError(
                "Authenticated dataset mapping does not match the mapped transformation."
            )
        result_id = canonical_fingerprint(
            {
                "kind": "mapped-relative-transformation-result",
                "plan": self.plan_id,
                "mapping": self.mapping_id,
                "leg_result": leg_result.result_id,
                "analysis": leg_result.analysis_id,
                "dataset": leg_result.dataset_id,
            }
        )
        return MappedRelativeTransformationResult(
            leg=leg_result,
            value=leg_result.value,
            variance=leg_result.variance,
            mapping_id=self.mapping_id,
            mapping_plan_id=self.plan_id,
            successful=leg_result.successful,
            result_id=result_id,
        )


class MappedRelativeSolvationPlan(StrictModule, NonTrainableState):
    """Relative solvation: T_solv - T_vac for one authenticated mapping."""

    solvent: MappedRelativeTransformationPlan
    vacuum: MappedRelativeTransformationPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solvent: MappedRelativeTransformationPlan,
        vacuum: MappedRelativeTransformationPlan,
        /,
    ):
        if not isinstance(solvent, MappedRelativeTransformationPlan) or not isinstance(
            vacuum, MappedRelativeTransformationPlan
        ):
            raise TypeError("Relative solvation requires solvent and vacuum mappings.")
        if solvent.mapping_id != vacuum.mapping_id:
            raise ValueError(
                "Relative-solvation legs require one exact mapping identity."
            )
        self.solvent = solvent
        self.vacuum = vacuum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-relative-solvation-plan",
                "solvent": solvent.plan_id,
                "vacuum": vacuum.plan_id,
                "formula": "T_solv - T_vac",
            }
        )

    def evaluate(
        self,
        solvent: MappedRelativeTransformationResult,
        vacuum: MappedRelativeTransformationResult,
        covariance: ArrayLike,
        /,
    ) -> MappedRelativeSolvationResult:
        if (
            solvent.mapping_plan_id != self.solvent.plan_id
            or vacuum.mapping_plan_id != self.vacuum.plan_id
            or solvent.mapping_id != self.solvent.mapping_id
            or vacuum.mapping_id != self.vacuum.mapping_id
        ):
            raise ValueError("Relative-solvation results do not match the mapped plans.")
        return _combine(
            MappedRelativeSolvationResult,
            self.plan_id,
            ("T_solv", "T_vac"),
            (solvent, vacuum),
            (1.0, -1.0),
            covariance,
            "relative solvation = T_solv - T_vac",
        )


class MappedRelativeBindingPlan(StrictModule, NonTrainableState):
    """Relative binding: T_complex - T_solv for one authenticated mapping."""

    complex: MappedRelativeTransformationPlan
    solvent: MappedRelativeTransformationPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: MappedRelativeTransformationPlan,
        solvent: MappedRelativeTransformationPlan,
        /,
    ):
        if not isinstance(complex, MappedRelativeTransformationPlan) or not isinstance(
            solvent, MappedRelativeTransformationPlan
        ):
            raise TypeError("Relative binding requires complex and solvent mappings.")
        if complex.mapping_id != solvent.mapping_id:
            raise ValueError("Relative-binding legs require one exact mapping identity.")
        self.complex = complex
        self.solvent = solvent
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-relative-binding-plan",
                "complex": complex.plan_id,
                "solvent": solvent.plan_id,
                "formula": "T_complex - T_solv",
            }
        )

    def evaluate(
        self,
        complex: MappedRelativeTransformationResult,
        solvent: MappedRelativeTransformationResult,
        covariance: ArrayLike,
        /,
    ) -> MappedRelativeBindingResult:
        if (
            complex.mapping_plan_id != self.complex.plan_id
            or solvent.mapping_plan_id != self.solvent.plan_id
            or complex.mapping_id != self.complex.mapping_id
            or solvent.mapping_id != self.solvent.mapping_id
        ):
            raise ValueError("Relative-binding results do not match the mapped plans.")
        return _combine(
            MappedRelativeBindingResult,
            self.plan_id,
            ("T_complex", "T_solv"),
            (complex, solvent),
            (1.0, -1.0),
            covariance,
            "relative binding = T_complex - T_solv",
        )


__all__ = [
    "AbsoluteBindingPlan",
    "AbsoluteBindingResult",
    "MappedRelativeBindingPlan",
    "MappedRelativeBindingResult",
    "MappedRelativeSolvationPlan",
    "MappedRelativeSolvationResult",
    "MappedRelativeTransformationPlan",
    "MappedRelativeTransformationResult",
    "NeutralAbsoluteSolvationPlan",
    "NeutralAbsoluteSolvationResult",
    "SeparatedTopologyPlan",
    "SeparatedTopologyResult",
]
