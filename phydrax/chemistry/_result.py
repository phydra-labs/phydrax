#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed electronic-structure values and execution evidence."""

from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticUnitSystem


class ElectronicCalculationStatus(IntEnum):
    SUCCESS = 0
    PROVIDER_UNAVAILABLE = 1
    UNSUPPORTED_CALCULATION = 2
    ELECTRONIC_NOT_CONVERGED = 3
    NONFINITE_RESULT = 4
    ATOM_ORDER_CHANGED = 5
    UNIT_CONVERSION_FAILED = 6
    BACKEND_FAILED = 7
    CANCELLED = 8


class ElectronicConvergenceEvidence(StrictModule, NonTrainableState):
    """Provider-neutral electronic convergence evidence."""

    converged: Array
    iterations: Array
    energy_residual: Array
    density_residual: Array
    message: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        converged: ArrayLike,
        /,
        *,
        iterations: ArrayLike = 0,
        energy_residual: ArrayLike = jnp.nan,
        density_residual: ArrayLike = jnp.nan,
        message: str = "not-reported",
    ):
        converged_ = jnp.asarray(converged, dtype=bool).reshape(())
        iterations_ = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        energy_ = jnp.asarray(energy_residual).reshape(())
        density_ = jnp.asarray(density_residual, dtype=energy_.dtype).reshape(())
        message_ = str(message).strip()
        if not message_:
            raise ValueError("Convergence evidence message must be non-empty.")
        if int(iterations_) < 0:
            raise ValueError("Electronic iteration count must be non-negative.")
        self.converged = converged_
        self.iterations = iterations_
        self.energy_residual = energy_
        self.density_residual = density_
        self.message = message_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "electronic-convergence-evidence",
                "arrays": array_tree_fingerprint(
                    {
                        "converged": np.asarray(converged_),
                        "iterations": np.asarray(iterations_),
                        "energy_residual": np.asarray(energy_),
                        "density_residual": np.asarray(density_),
                    }
                ),
                "message": message_,
            }
        )


class ElectronicWorkEvidence(StrictModule, NonTrainableState):
    """Exact provider-call accounting for one returned evaluation."""

    energy_evaluations: Array
    force_evaluations: Array
    hessian_evaluations: Array
    property_evaluations: Array
    work_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy_evaluations: int = 1,
        force_evaluations: int = 0,
        hessian_evaluations: int = 0,
        property_evaluations: int = 0,
    ):
        values = tuple(
            int(value)
            for value in (
                energy_evaluations,
                force_evaluations,
                hessian_evaluations,
                property_evaluations,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Electronic work counts must be non-negative.")
        arrays = tuple(jnp.asarray(value, dtype=jnp.int32) for value in values)
        (
            self.energy_evaluations,
            self.force_evaluations,
            self.hessian_evaluations,
            self.property_evaluations,
        ) = arrays
        self.work_id = canonical_fingerprint(
            {"kind": "electronic-work-evidence", "counts": list(values)}
        )


class ElectronicEvaluationHeader(StrictModule, NonTrainableState):
    """Identity, unit, status, and artifact envelope shared by result variants."""

    stable_particle_ids: Array
    active_mask: Array
    units: AtomisticUnitSystem
    convergence: ElectronicConvergenceEvidence
    work: ElectronicWorkEvidence
    status: Array
    system_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    model_chemistry_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_unit_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    artifact_ids: tuple[str, ...] = eqx.field(static=True)
    header_id: str = eqx.field(static=True)

    def __init__(
        self,
        stable_particle_ids: ArrayLike,
        active_mask: ArrayLike,
        units: AtomisticUnitSystem,
        convergence: ElectronicConvergenceEvidence,
        work: ElectronicWorkEvidence,
        status: ElectronicCalculationStatus | int | ArrayLike,
        /,
        *,
        system_id: str,
        geometry_id: str,
        state_id: str,
        model_chemistry_id: str,
        request_id: str,
        provider_id: str,
        source_unit_ids: tuple[tuple[str, str], ...],
        artifact_ids: tuple[str, ...] = (),
    ):
        ids = jnp.asarray(stable_particle_ids, dtype=jnp.int64)
        active = jnp.asarray(active_mask, dtype=bool)
        if ids.ndim != 1 or active.shape != ids.shape:
            raise ValueError("Header particle IDs and active mask must be aligned vectors.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if not isinstance(convergence, ElectronicConvergenceEvidence):
            raise TypeError("convergence must be ElectronicConvergenceEvidence.")
        if not isinstance(work, ElectronicWorkEvidence):
            raise TypeError("work must be ElectronicWorkEvidence.")
        status_ = jnp.asarray(status, dtype=jnp.int32).reshape(())
        ElectronicCalculationStatus(int(status_))
        identifiers = tuple(
            str(value).strip()
            for value in (
                system_id,
                geometry_id,
                state_id,
                model_chemistry_id,
                request_id,
                provider_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Electronic result identifiers must be non-empty.")
        source_units = tuple(sorted((str(name), str(unit)) for name, unit in source_unit_ids))
        if not source_units or any(not name or not unit for name, unit in source_units):
            raise ValueError("source_unit_ids must contain named non-empty unit IDs.")
        if len({name for name, _ in source_units}) != len(source_units):
            raise ValueError("source_unit_ids names must be unique.")
        artifacts = tuple(str(value).strip() for value in artifact_ids)
        if any(not value for value in artifacts) or len(set(artifacts)) != len(artifacts):
            raise ValueError("artifact_ids must be unique non-empty identifiers.")
        self.stable_particle_ids = ids
        self.active_mask = active
        self.units = units
        self.convergence = convergence
        self.work = work
        self.status = status_
        (
            self.system_id,
            self.geometry_id,
            self.state_id,
            self.model_chemistry_id,
            self.request_id,
            self.provider_id,
        ) = identifiers
        self.source_unit_ids = source_units
        self.artifact_ids = artifacts
        self.header_id = canonical_fingerprint(
            {
                "kind": "electronic-evaluation-header",
                "system": self.system_id,
                "geometry": self.geometry_id,
                "state": self.state_id,
                "model_chemistry": self.model_chemistry_id,
                "request": self.request_id,
                "provider": self.provider_id,
                "units": units.unit_system_id,
                "source_units": [list(value) for value in source_units],
                "particles": array_tree_fingerprint(
                    {"ids": np.asarray(ids), "active": np.asarray(active)}
                ),
                "convergence": convergence.evidence_id,
                "work": work.work_id,
                "status": int(status_),
                "artifacts": list(artifacts),
            }
        )

    @property
    def successful(self) -> Array:
        return self.convergence.converged & (
            self.status == int(ElectronicCalculationStatus.SUCCESS)
        )


class ElectronicEnergyEvaluation(StrictModule, NonTrainableState):
    header: ElectronicEvaluationHeader
    energy: Array
    result_id: str = eqx.field(static=True)

    def __init__(self, header: ElectronicEvaluationHeader, energy: ArrayLike, /):
        energy_ = jnp.asarray(energy).reshape(())
        _require_finite_if_successful(header, energy_)
        self.header = header
        self.energy = energy_
        self.result_id = _result_id("energy", header, {"energy": energy_})

    @property
    def successful(self) -> Array:
        return self.header.successful


class ElectronicEnergyForceEvaluation(StrictModule, NonTrainableState):
    header: ElectronicEvaluationHeader
    energy: Array
    forces: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        header: ElectronicEvaluationHeader,
        energy: ArrayLike,
        forces: ArrayLike,
        /,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = _particle_tensor(header, forces, rank=2, name="forces")
        _require_finite_if_successful(header, energy_, forces_)
        self.header = header
        self.energy = energy_
        self.forces = forces_
        self.result_id = _result_id(
            "energy-force", header, {"energy": energy_, "forces": forces_}
        )

    @property
    def successful(self) -> Array:
        return self.header.successful


class ElectronicEnergyForceHessianEvaluation(StrictModule, NonTrainableState):
    header: ElectronicEvaluationHeader
    energy: Array
    forces: Array
    hessian: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        header: ElectronicEvaluationHeader,
        energy: ArrayLike,
        forces: ArrayLike,
        hessian: ArrayLike,
        /,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = _particle_tensor(header, forces, rank=2, name="forces")
        hessian_ = _particle_tensor(header, hessian, rank=4, name="hessian")
        _require_finite_if_successful(header, energy_, forces_, hessian_)
        self.header = header
        self.energy = energy_
        self.forces = forces_
        self.hessian = hessian_
        self.result_id = _result_id(
            "energy-force-hessian",
            header,
            {"energy": energy_, "forces": forces_, "hessian": hessian_},
        )

    @property
    def successful(self) -> Array:
        return self.header.successful


class ElectronicGroundStatePropertyEvaluation(StrictModule, NonTrainableState):
    header: ElectronicEvaluationHeader
    energy: Array
    forces: Array
    dipole: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        header: ElectronicEvaluationHeader,
        energy: ArrayLike,
        forces: ArrayLike,
        dipole: ArrayLike,
        /,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = _particle_tensor(header, forces, rank=2, name="forces")
        dipole_ = jnp.asarray(dipole, dtype=energy_.dtype)
        if dipole_.shape != (3,):
            raise ValueError("dipole must have shape (3,).")
        _require_finite_if_successful(header, energy_, forces_, dipole_)
        self.header = header
        self.energy = energy_
        self.forces = forces_
        self.dipole = dipole_
        self.result_id = _result_id(
            "ground-state-properties",
            header,
            {"energy": energy_, "forces": forces_, "dipole": dipole_},
        )

    @property
    def successful(self) -> Array:
        return self.header.successful


def _particle_tensor(
    header: ElectronicEvaluationHeader,
    value: ArrayLike,
    /,
    *,
    rank: int,
    name: str,
) -> Array:
    array = jnp.asarray(value)
    capacity = int(header.stable_particle_ids.shape[0])
    expected = (capacity, 3) if rank == 2 else (capacity, 3, capacity, 3)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}.")
    return array


def _require_finite_if_successful(
    header: ElectronicEvaluationHeader, *values: Array
) -> None:
    if bool(header.successful) and any(not bool(jnp.all(jnp.isfinite(value))) for value in values):
        raise ValueError("A successful electronic result must contain finite values.")


def _result_id(kind: str, header: ElectronicEvaluationHeader, arrays: dict[str, Array]) -> str:
    return canonical_fingerprint(
        {
            "kind": f"electronic-{kind}-evaluation",
            "header": header.header_id,
            "arrays": array_tree_fingerprint(
                {name: np.asarray(value) for name, value in arrays.items()}
            ),
        }
    )


ElectronicEvaluation: TypeAlias = (
    ElectronicEnergyEvaluation
    | ElectronicEnergyForceEvaluation
    | ElectronicEnergyForceHessianEvaluation
    | ElectronicGroundStatePropertyEvaluation
)


__all__ = [
    "ElectronicCalculationStatus",
    "ElectronicConvergenceEvidence",
    "ElectronicEnergyEvaluation",
    "ElectronicEnergyForceEvaluation",
    "ElectronicEnergyForceHessianEvaluation",
    "ElectronicEvaluation",
    "ElectronicEvaluationHeader",
    "ElectronicGroundStatePropertyEvaluation",
    "ElectronicWorkEvidence",
]
