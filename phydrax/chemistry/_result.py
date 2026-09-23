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
from ..units import UnitDefinition


class ElectronicCalculationStatus(IntEnum):
    SUCCESS = 0
    PROVIDER_UNAVAILABLE = 1
    UNSUPPORTED_CALCULATION = 2
    ELECTRONIC_NOT_CONVERGED = 3
    NONFINITE_RESULT = 4
    ATOM_ORDER_CHANGED = 5
    UNIT_CONVERSION_FAILED = 6
    BACKEND_FAILED = 7
    CANCELED = 8


class ElectronicEnergyLedger(StrictModule, NonTrainableState):
    """Named electronic energy terms with an independently certified total."""

    components: Array
    reported_total: Array
    closure_residual: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    energy_unit: UnitDefinition
    ledger_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: tuple[str, ...],
        components: ArrayLike,
        reported_total: ArrayLike,
        energy_unit: UnitDefinition,
        /,
    ):
        names = tuple(str(value).strip() for value in component_names)
        values = jnp.asarray(components)
        total = jnp.asarray(reported_total, dtype=values.dtype).reshape(())
        if (
            not names
            or len(names) != values.size
            or values.ndim != 1
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Energy ledger names and components must align uniquely.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        residual = jnp.abs(jnp.sum(values) - total)
        self.components = values
        self.reported_total = total
        self.closure_residual = residual
        self.component_names = names
        self.energy_unit = energy_unit
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "electronic-energy-ledger",
                "component_names": list(names),
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "components": np.asarray(values),
                        "reported_total": np.asarray(total),
                        "closure_residual": np.asarray(residual),
                    }
                ),
            }
        )


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
        converged_host = np.asarray(converged)
        iterations_host = np.asarray(iterations)
        energy_host = np.asarray(energy_residual)
        density_host = np.asarray(density_residual)
        if converged_host.shape != ():
            raise ValueError("converged must be scalar.")
        if (
            iterations_host.shape != ()
            or iterations_host.dtype.kind not in "iu"
            or iterations_host.dtype.kind == "b"
            or int(iterations_host) < 0
        ):
            raise ValueError("Electronic iteration count must be a non-negative integer.")
        if energy_host.shape != () or density_host.shape != ():
            raise ValueError("Electronic convergence residuals must be scalar.")
        if bool(converged_host) and (
            not np.isfinite(energy_host)
            or float(energy_host) < 0.0
            or not np.isfinite(density_host)
            or float(density_host) < 0.0
        ):
            raise ValueError(
                "Converged electronic evidence requires finite non-negative residuals."
            )
        converged_ = jnp.asarray(converged_host, dtype=jnp.bool_).reshape(())
        iterations_ = jnp.asarray(iterations_host, dtype=jnp.int32).reshape(())
        energy_ = jnp.asarray(energy_host).reshape(())
        density_ = jnp.asarray(density_host, dtype=energy_.dtype).reshape(())
        message_ = str(message).strip()
        if not message_:
            raise ValueError("Convergence evidence message must be non-empty.")
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

    @property
    def valid(self) -> Array:
        residuals_valid = (
            jnp.isfinite(self.energy_residual)
            & (self.energy_residual >= 0.0)
            & jnp.isfinite(self.density_residual)
            & (self.density_residual >= 0.0)
        )
        return ~self.converged | residuals_valid


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
        values = (
            energy_evaluations,
            force_evaluations,
            hessian_evaluations,
            property_evaluations,
        )
        maximum = int(np.iinfo(np.int32).max)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            or int(value) > maximum
            for value in values
        ):
            raise ValueError(
                "Electronic work counts must be exact non-negative int32 values."
            )
        values = tuple(int(value) for value in values)
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
    task_id: str = eqx.field(static=True)
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
        task_id: str,
        provider_id: str,
        source_unit_ids: tuple[tuple[str, str], ...],
        artifact_ids: tuple[str, ...] = (),
    ):
        ids = jnp.asarray(stable_particle_ids, dtype=jnp.int64)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        if ids.ndim != 1 or active.shape != ids.shape:
            raise ValueError(
                "Header particle IDs and active mask must be aligned vectors."
            )
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
                task_id,
                provider_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Electronic result identifiers must be non-empty.")
        source_units = tuple(
            sorted((str(name), str(unit)) for name, unit in source_unit_ids)
        )
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
            self.task_id,
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
                "task": self.task_id,
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
        return (
            self.convergence.valid
            & self.convergence.converged
            & (self.status == int(ElectronicCalculationStatus.SUCCESS))
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


class ElectronicPeriodicEvaluation(StrictModule, NonTrainableState):
    """Provider-normalized periodic electronic observables."""

    header: ElectronicEvaluationHeader
    energy: Array
    forces: Array | None
    stress: Array | None
    band_energies: Array | None
    density_matrices: Array | None
    polarization: Array | None
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        header: ElectronicEvaluationHeader,
        energy: ArrayLike,
        /,
        *,
        forces: ArrayLike | None = None,
        stress: ArrayLike | None = None,
        band_energies: ArrayLike | None = None,
        density_matrices: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
    ):
        energy_ = jnp.asarray(energy).reshape(())
        forces_ = (
            None
            if forces is None
            else _particle_tensor(header, forces, rank=2, name="forces")
        )
        stress_ = None if stress is None else jnp.asarray(stress, dtype=energy_.dtype)
        bands_ = None if band_energies is None else jnp.asarray(band_energies)
        density_ = None if density_matrices is None else jnp.asarray(density_matrices)
        polarization_ = (
            None
            if polarization is None
            else jnp.asarray(polarization, dtype=energy_.real.dtype)
        )
        if stress_ is not None and stress_.shape != (3, 3):
            raise ValueError("stress must have shape (3, 3).")
        if bands_ is not None and bands_.ndim not in (2, 3):
            raise ValueError(
                "band_energies must have shape (k, band) or (spin, k, band)."
            )
        if density_ is not None and (
            density_.ndim not in (3, 4) or density_.shape[-1] != density_.shape[-2]
        ):
            raise ValueError(
                "density_matrices must carry square orbital axes and optional spin."
            )
        if polarization_ is not None and polarization_.shape != (3,):
            raise ValueError("polarization must have shape (3,).")
        arrays = {"energy": energy_}
        for name, value in (
            ("forces", forces_),
            ("stress", stress_),
            ("band_energies", bands_),
            ("density_matrices", density_),
            ("polarization", polarization_),
        ):
            if value is not None:
                arrays[name] = value
        _require_finite_if_successful(header, *arrays.values())
        self.header = header
        self.energy = energy_
        self.forces = forces_
        self.stress = stress_
        self.band_energies = bands_
        self.density_matrices = density_
        self.polarization = polarization_
        self.result_id = _result_id("periodic", header, arrays)

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
    capacity = header.stable_particle_ids.shape[0]
    expected = (capacity, 3) if rank == 2 else (capacity, 3, capacity, 3)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}.")
    return array


def _require_finite_if_successful(
    header: ElectronicEvaluationHeader, *values: Array
) -> None:
    if bool(header.successful) and any(
        not bool(jnp.all(jnp.isfinite(value))) for value in values
    ):
        raise ValueError("A successful electronic result must contain finite values.")


def _result_id(
    kind: str, header: ElectronicEvaluationHeader, arrays: dict[str, Array]
) -> str:
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
    | ElectronicPeriodicEvaluation
)


__all__ = [
    "ElectronicCalculationStatus",
    "ElectronicConvergenceEvidence",
    "ElectronicEnergyEvaluation",
    "ElectronicEnergyLedger",
    "ElectronicEnergyForceEvaluation",
    "ElectronicEnergyForceHessianEvaluation",
    "ElectronicEvaluation",
    "ElectronicEvaluationHeader",
    "ElectronicGroundStatePropertyEvaluation",
    "ElectronicPeriodicEvaluation",
    "ElectronicWorkEvidence",
]
