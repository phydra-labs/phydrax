#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native lifecycle records and archives for molecular electronic results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticUnitSystem
from ..execution import ExecutionPlan
from ..lifecycle import (
    AnalysisPlan,
    create,
    LifecycleArchive,
    ModelManifest,
    open as open_lifecycle,
    payload_digest,
    ResultManifest,
    RunRecord,
)
from ._calculation import ElectronicCalculationPlan
from ._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicEvaluation,
    ElectronicEvaluationHeader,
    ElectronicGroundStatePropertyEvaluation,
    ElectronicWorkEvidence,
)
from ._units import dipole_unit, hessian_unit


_SEMANTICS_KEY = "computational_chemistry"
_RESULT_TYPES = (
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicGroundStatePropertyEvaluation,
)


class ChemistryRunEnvelope(StrictModule, NonTrainableState):
    """Lifecycle records for one completed molecular electronic evaluation."""

    analysis: AnalysisPlan = eqx.field(static=True)
    execution: ExecutionPlan = eqx.field(static=True)
    run: RunRecord = eqx.field(static=True)
    result: ResultManifest = eqx.field(static=True)
    model: ModelManifest | None = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        analysis: AnalysisPlan,
        execution: ExecutionPlan,
        run: RunRecord,
        result: ResultManifest,
        /,
        *,
        model: ModelManifest | None = None,
    ):
        if not isinstance(analysis, AnalysisPlan):
            raise TypeError("analysis must be AnalysisPlan.")
        if not isinstance(execution, ExecutionPlan):
            raise TypeError("execution must be ExecutionPlan.")
        if not isinstance(run, RunRecord):
            raise TypeError("run must be RunRecord.")
        if not isinstance(result, ResultManifest):
            raise TypeError("result must be ResultManifest.")
        if model is not None and not isinstance(model, ModelManifest):
            raise TypeError("model must be ModelManifest or None.")
        if run.analysis_plan_id != analysis.analysis_plan_id:
            raise ValueError("Chemistry run and analysis identities differ.")
        if run.execution_plan_id != execution.execution_plan_id:
            raise ValueError("Chemistry run and execution identities differ.")
        if result.run_id != run.run_id:
            raise ValueError("Chemistry result and run identities differ.")
        self.analysis = analysis
        self.execution = execution
        self.run = run
        self.result = result
        self.model = model
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "chemistry-run-envelope",
                "analysis": analysis.plan_fingerprint,
                "execution": execution.plan_fingerprint,
                "run": run.record_id,
                "result": result.manifest_id,
                "model": None if model is None else model.manifest_id,
            }
        )


def _result_arrays(result: ElectronicEvaluation, /) -> dict[str, np.ndarray]:
    if not isinstance(result, _RESULT_TYPES):
        raise TypeError("result must be a supported electronic evaluation.")
    arrays = {
        "stable_particle_ids": np.asarray(result.header.stable_particle_ids),
        "active_mask": np.asarray(result.header.active_mask),
        "energy": np.asarray(result.energy),
    }
    if isinstance(
        result,
        (
            ElectronicEnergyForceEvaluation,
            ElectronicEnergyForceHessianEvaluation,
            ElectronicGroundStatePropertyEvaluation,
        ),
    ):
        arrays["forces"] = np.asarray(result.forces)
    if isinstance(result, ElectronicEnergyForceHessianEvaluation):
        arrays["hessian"] = np.asarray(result.hessian)
    if isinstance(result, ElectronicGroundStatePropertyEvaluation):
        arrays["dipole"] = np.asarray(result.dipole)
    return arrays


def _finite_or_none(value: object, /) -> float | None:
    number = float(np.asarray(value))
    return number if np.isfinite(number) else None


def _descriptor(result: ElectronicEvaluation, /) -> dict[str, object]:
    if isinstance(result, ElectronicEnergyForceHessianEvaluation):
        kind = "energy-force-hessian"
    elif isinstance(result, ElectronicGroundStatePropertyEvaluation):
        kind = "ground-state-properties"
    elif isinstance(result, ElectronicEnergyForceEvaluation):
        kind = "energy-force"
    elif isinstance(result, ElectronicEnergyEvaluation):
        kind = "energy"
    else:
        raise TypeError("result must be a supported electronic evaluation.")
    header = result.header
    return {
        "kind": kind,
        "result_id": result.result_id,
        "system_id": header.system_id,
        "geometry_id": header.geometry_id,
        "state_id": header.state_id,
        "model_chemistry_id": header.model_chemistry_id,
        "request_id": header.request_id,
        "provider_id": header.provider_id,
        "units": header.units.to_dict(),
        "source_unit_ids": [list(value) for value in header.source_unit_ids],
        "artifact_ids": list(header.artifact_ids),
        "status": int(header.status),
        "convergence": {
            "converged": bool(header.convergence.converged),
            "iterations": int(header.convergence.iterations),
            "energy_residual": _finite_or_none(header.convergence.energy_residual),
            "density_residual": _finite_or_none(header.convergence.density_residual),
            "message": header.convergence.message,
        },
        "work": {
            "energy_evaluations": int(header.work.energy_evaluations),
            "force_evaluations": int(header.work.force_evaluations),
            "hessian_evaluations": int(header.work.hessian_evaluations),
            "property_evaluations": int(header.work.property_evaluations),
        },
    }


def _manifest(
    result: ElectronicEvaluation,
    run_id: str,
    arrays: Mapping[str, np.ndarray],
    /,
) -> ResultManifest:
    units = result.header.units
    field_units = {
        "stable_particle_ids": "1",
        "active_mask": "1",
        "energy": units.scale.energy_unit.unit_id,
    }
    if "forces" in arrays:
        field_units["forces"] = units.scale.force_unit.unit_id
    if "hessian" in arrays:
        field_units["hessian"] = hessian_unit(units).unit_id
    if "dipole" in arrays:
        field_units["dipole"] = dipole_unit(units).unit_id
    return ResultManifest(
        result.result_id,
        run_id,
        field_units,
        {name: payload_digest(value) for name, value in arrays.items()},
        evidence_ids=(result.header.convergence.evidence_id, result.header.work.work_id),
        diagnostic_ids=result.header.artifact_ids,
        sampled_semantics={
            _SEMANTICS_KEY: json.dumps(
                _descriptor(result), allow_nan=False, separators=(",", ":"), sort_keys=True
            )
        },
    )


def chemistry_lifecycle(
    calculation: ElectronicCalculationPlan,
    result: ElectronicEvaluation,
    /,
    *,
    provider_capability_id: str,
    execution_backend: str = "host",
    solver_policy_id: str = "direct-electronic-evaluation",
) -> ChemistryRunEnvelope:
    if not isinstance(calculation, ElectronicCalculationPlan):
        raise TypeError("calculation must be ElectronicCalculationPlan.")
    if not isinstance(result, _RESULT_TYPES):
        raise TypeError("result must be a supported electronic evaluation.")
    if result.header.system_id != calculation.system.system_id:
        raise ValueError("Electronic result belongs to another calculation system.")
    capability = str(provider_capability_id).strip()
    if not capability:
        raise ValueError("provider_capability_id must be non-empty.")
    basis = calculation.model_chemistry.basis
    analysis = AnalysisPlan(
        calculation.calculation_id,
        result.header.provider_id,
        calculation.model_chemistry.model_chemistry_id if basis is None else basis.basis_id,
        (
            calculation.system.system_id,
            calculation.state.prepared_id,
            calculation.request.request_id,
        ),
        material_plan_id=calculation.system.system_id,
        capability_ids=(capability,),
    )
    execution_id = canonical_fingerprint(
        {
            "kind": "chemistry-execution",
            "calculation": calculation.calculation_id,
            "provider": result.header.provider_id,
            "geometry": result.header.geometry_id,
        }
    )
    execution = ExecutionPlan(
        execution_id,
        str(execution_backend),
        calculation.precision.policy_id,
        str(solver_policy_id),
    )
    run_id = canonical_fingerprint(
        {
            "kind": "chemistry-run",
            "calculation": calculation.calculation_id,
            "result": result.result_id,
        }
    )
    arrays = _result_arrays(result)
    manifest = _manifest(result, run_id, arrays)
    successful = bool(result.successful)
    run = RunRecord(
        run_id,
        analysis.analysis_plan_id,
        result.header.geometry_id,
        execution.execution_plan_id,
        "completed" if successful else "failed",
        result_ids=(manifest.manifest_id,),
        diagnostic_ids=(
            result.header.convergence.evidence_id,
            result.header.work.work_id,
            *result.header.artifact_ids,
        ),
    )
    artifact = calculation.model_chemistry.model_artifact
    model = (
        None
        if artifact is None
        else ModelManifest(
            calculation.model_chemistry.model_chemistry_id,
            analysis.analysis_plan_id,
            result.header.geometry_id,
            {"model_artifact": artifact.checksum},
            unit_contract_id=calculation.system.units.unit_system_id,
            association_ids=(artifact.manifest_id,),
        )
    )
    return ChemistryRunEnvelope(analysis, execution, run, manifest, model=model)


def write_electronic_result_archive(
    path: str | Path,
    result: ElectronicEvaluation,
    /,
    *,
    run_id: str,
) -> LifecycleArchive:
    if not isinstance(result, _RESULT_TYPES):
        raise TypeError("result must be a supported electronic evaluation.")
    run = str(run_id).strip()
    if not run:
        raise ValueError("run_id must be non-empty.")
    arrays = _result_arrays(result)
    return create(path, manifest=_manifest(result, run, arrays), arrays=arrays)


def read_electronic_result_archive(path: str | Path, /) -> ElectronicEvaluation:
    archive = open_lifecycle(path)
    manifest = archive.manifest
    if not isinstance(manifest, ResultManifest):
        raise ValueError("Chemistry result archives require a ResultManifest.")
    semantics = dict(manifest.sampled_semantics)
    if _SEMANTICS_KEY not in semantics:
        raise ValueError("Result archive has no computational chemistry semantics.")
    descriptor = json.loads(semantics[_SEMANTICS_KEY])
    arrays = archive.arrays
    units = AtomisticUnitSystem.from_dict(descriptor["units"])
    convergence_record = descriptor["convergence"]
    work_record = descriptor["work"]
    convergence = ElectronicConvergenceEvidence(
        convergence_record["converged"],
        iterations=convergence_record["iterations"],
        energy_residual=(
            jnp.nan
            if convergence_record["energy_residual"] is None
            else convergence_record["energy_residual"]
        ),
        density_residual=(
            jnp.nan
            if convergence_record["density_residual"] is None
            else convergence_record["density_residual"]
        ),
        message=convergence_record["message"],
    )
    work = ElectronicWorkEvidence(**work_record)
    header = ElectronicEvaluationHeader(
        arrays["stable_particle_ids"],
        arrays["active_mask"],
        units,
        convergence,
        work,
        ElectronicCalculationStatus(descriptor["status"]),
        system_id=descriptor["system_id"],
        geometry_id=descriptor["geometry_id"],
        state_id=descriptor["state_id"],
        model_chemistry_id=descriptor["model_chemistry_id"],
        request_id=descriptor["request_id"],
        provider_id=descriptor["provider_id"],
        source_unit_ids=tuple(tuple(value) for value in descriptor["source_unit_ids"]),
        artifact_ids=tuple(descriptor["artifact_ids"]),
    )
    kind = descriptor["kind"]
    if kind == "energy":
        result: ElectronicEvaluation = ElectronicEnergyEvaluation(header, arrays["energy"])
    elif kind == "energy-force":
        result = ElectronicEnergyForceEvaluation(
            header, arrays["energy"], arrays["forces"]
        )
    elif kind == "energy-force-hessian":
        result = ElectronicEnergyForceHessianEvaluation(
            header, arrays["energy"], arrays["forces"], arrays["hessian"]
        )
    elif kind == "ground-state-properties":
        result = ElectronicGroundStatePropertyEvaluation(
            header, arrays["energy"], arrays["forces"], arrays["dipole"]
        )
    else:
        raise ValueError("Unknown electronic result kind in archive.")
    if result.result_id != manifest.result_id or result.result_id != descriptor["result_id"]:
        raise ValueError("Electronic result archive identity does not match its payload.")
    return result


__all__ = [
    "ChemistryRunEnvelope",
    "chemistry_lifecycle",
    "read_electronic_result_archive",
    "write_electronic_result_archive",
]
