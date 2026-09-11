#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Loss-audited QCSchema mappings for finite molecular calculations."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...units import (
    BOHR,
    conversion_factor,
    DALTON,
    derived_unit,
    ELEMENTARY_CHARGE,
    HARTREE,
)
from .._calculation import (
    electronic_geometry_id,
    ElectronicCalculationPlan,
    make_electronic_evaluation,
)
from .._properties import ElectronicProperty
from .._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEvaluation,
    ElectronicWorkEvidence,
)


_SYMBOLS = (
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


def is_qcelemental_available() -> bool:
    return importlib.util.find_spec("qcelemental") is not None


def require_qcelemental():
    if not is_qcelemental_available():
        raise ImportError(
            "QCSchema object construction requires optional dependency 'qcelemental'."
        )
    return importlib.import_module("qcelemental")


def _driver(calculation: ElectronicCalculationPlan, /) -> str:
    request = calculation.request
    if request.requires(ElectronicProperty.HESSIAN):
        return "hessian"
    if request.requires(ElectronicProperty.FORCES):
        return "gradient"
    return "energy"


def electronic_calculation_to_qcschema(
    calculation: ElectronicCalculationPlan,
    positions,
    /,
    *,
    cell_vectors=None,
    keywords: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], AdapterReport]:
    """Return one QCSchema AtomicInput mapping and a lossless adapter report."""

    if not isinstance(calculation, ElectronicCalculationPlan):
        raise TypeError("calculation must be ElectronicCalculationPlan.")
    if calculation.system.cell is not None and any(calculation.system.cell.periodic_axes):
        raise ValueError("QCSchema molecular input currently supports nonperiodic systems.")
    coordinate = np.asarray(positions, dtype=np.dtype(calculation.system.coordinate_dtype))
    expected = (int(calculation.system.particle_ids.shape[0]), 3)
    if coordinate.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    active = np.asarray(calculation.system.active_mask, dtype=bool)
    numbers = np.asarray(calculation.system.atomic_numbers, dtype=np.int64)[active]
    if np.any(numbers <= 0) or np.any(numbers >= len(_SYMBOLS)):
        raise ValueError("QCSchema export requires supported positive atomic numbers.")
    length_factor = float(conversion_factor(calculation.system.units.scale.length_unit, BOHR))
    mass_factor = float(conversion_factor(calculation.system.units.mass_unit, DALTON))
    geometry_id = electronic_geometry_id(calculation.system, coordinate, cell_vectors)
    basis = calculation.model_chemistry.basis
    model: dict[str, Any] = {"method": calculation.model_chemistry.method.method}
    if basis is not None:
        model["basis"] = basis.name
    payload: dict[str, Any] = {
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "molecule": {
            "schema_name": "qcschema_molecule",
            "schema_version": 2,
            "symbols": [_SYMBOLS[int(value)] for value in numbers],
            "geometry": (coordinate[active] * length_factor).reshape(-1).tolist(),
            "masses": (
                np.asarray(calculation.system.masses)[active] * mass_factor
            ).tolist(),
            "molecular_charge": float(calculation.state.total_charge),
            "molecular_multiplicity": calculation.state.spin_multiplicity,
            "fix_com": True,
            "fix_orientation": True,
            "real": [True] * int(np.count_nonzero(active)),
            "extras": {
                "phydrax": {
                    "system_id": calculation.system.system_id,
                    "geometry_id": geometry_id,
                    "stable_particle_ids": np.asarray(
                        calculation.system.particle_ids
                    )[active].tolist(),
                    "state_id": calculation.state.prepared_id,
                }
            },
        },
        "driver": _driver(calculation),
        "model": model,
        "keywords": dict(keywords or {}),
        "extras": {
            "phydrax": {
                "calculation_id": calculation.calculation_id,
                "request_id": calculation.request.request_id,
                "model_chemistry_id": calculation.model_chemistry.model_chemistry_id,
            }
        },
    }
    target_id = canonical_fingerprint(
        {"kind": "qcschema-atomic-input", "payload": payload}
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "ElectronicCalculationPlan",
        "QCSchema AtomicInput",
        source_id=calculation.calculation_id,
        target_id=target_id,
        coordinate_mapping=(
            "active positions[atom,xyz] -> molecule.geometry[3*atom+xyz]",
            "stable particle order -> molecule symbol order",
        ),
        preserved_fields=(
            "atomic_numbers",
            "positions",
            "masses",
            "total_charge",
            "spin_multiplicity",
            "model_chemistry",
            "property_driver",
            "stable_particle_ids",
        ),
        assumptions=(
            "QCSchema molecular geometry is represented in bohr.",
            "QCSchema molecular masses are represented in dalton.",
            "Center-of-mass translation and orientation changes are disabled.",
        ),
    )
    return payload, report


def electronic_calculation_to_qcelemental(
    calculation: ElectronicCalculationPlan,
    positions,
    /,
    *,
    cell_vectors=None,
    keywords: Mapping[str, Any] | None = None,
):
    qcelemental = require_qcelemental()
    payload, report = electronic_calculation_to_qcschema(
        calculation, positions, cell_vectors=cell_vectors, keywords=keywords
    )
    return qcelemental.models.AtomicInput(**payload), report


def _energy(record: Mapping[str, Any], driver: str, /) -> float:
    if driver == "energy":
        return float(record["return_result"])
    properties = record.get("properties", {})
    if not isinstance(properties, Mapping) or "return_energy" not in properties:
        return float("nan")
    return float(properties["return_energy"])


def _source_units() -> tuple[tuple[str, str], ...]:
    gradient = derived_unit("hartree/bohr", ((HARTREE, 1), (BOHR, -1)))
    hessian = derived_unit("hartree/bohr^2", ((HARTREE, 1), (BOHR, -2)))
    return (
        ("energy", HARTREE.unit_id),
        ("gradient", gradient.unit_id),
        ("hessian", hessian.unit_id),
        ("length", BOHR.unit_id),
    )


def electronic_evaluation_from_qcschema(
    calculation: ElectronicCalculationPlan,
    provider_id: str,
    positions,
    record: Mapping[str, Any],
    /,
    *,
    cell_vectors=None,
) -> tuple[ElectronicEvaluation, AdapterReport]:
    """Convert one QCSchema AtomicResult mapping into native units and order."""

    if not isinstance(calculation, ElectronicCalculationPlan):
        raise TypeError("calculation must be ElectronicCalculationPlan.")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a QCSchema result mapping.")
    driver = _driver(calculation)
    success = bool(record.get("success", False))
    active = np.asarray(calculation.system.active_mask, dtype=bool)
    capacity = active.size
    count = int(np.count_nonzero(active))
    energy_factor = float(conversion_factor(HARTREE, calculation.system.units.scale.energy_unit))
    length_factor = float(conversion_factor(BOHR, calculation.system.units.scale.length_unit))
    force_factor = energy_factor / length_factor
    hessian_factor = energy_factor / length_factor**2
    energy = _energy(record, driver) * energy_factor if success else float("nan")
    forces = None
    hessian = None
    if calculation.request.requires(ElectronicProperty.FORCES):
        active_forces = np.full((count, 3), np.nan)
        if success:
            if driver == "gradient":
                gradient = np.asarray(record["return_result"], dtype=float).reshape((count, 3))
            else:
                properties = record.get("properties", {})
                gradient = np.asarray(properties["return_gradient"], dtype=float).reshape(
                    (count, 3)
                )
            active_forces = -force_factor * gradient
        forces = np.zeros((capacity, 3), dtype=float)
        forces[active] = active_forces
    if calculation.request.requires(ElectronicProperty.HESSIAN):
        active_hessian = np.full((count, 3, count, 3), np.nan)
        if success:
            active_hessian = (
                np.asarray(record["return_result"], dtype=float).reshape(
                    (count, 3, count, 3)
                )
                * hessian_factor
            )
        hessian = np.zeros((capacity, 3, capacity, 3), dtype=float)
        active_indices = np.flatnonzero(active)
        hessian[
            np.ix_(active_indices, np.arange(3), active_indices, np.arange(3))
        ] = active_hessian
    dipole = None
    if calculation.request.requires(ElectronicProperty.DIPOLE):
        properties = record.get("properties", {})
        if not isinstance(properties, Mapping) or "scf_dipole_moment" not in properties:
            raise ValueError("QCSchema result omitted the requested dipole moment.")
        dipole = np.asarray(properties["scf_dipole_moment"], dtype=float).reshape((3,))
        charge_factor = float(
            conversion_factor(
                ELEMENTARY_CHARGE,
                calculation.system.units.charge_unit,
            )
        )
        dipole = dipole * charge_factor * length_factor
    molecule = record.get("molecule", {})
    molecule_extras = (
        molecule.get("extras", {}) if isinstance(molecule, Mapping) else {}
    )
    extras = record.get("extras", {})
    phydrax = (
        molecule_extras.get("phydrax", {})
        if isinstance(molecule_extras, Mapping)
        else {}
    )
    if not phydrax and isinstance(extras, Mapping):
        phydrax = extras.get("phydrax", {})
    losses: tuple[AdapterLoss, ...] = ()
    expected_ids = np.asarray(calculation.system.particle_ids)[active].tolist()
    observed_ids = (
        phydrax.get("stable_particle_ids") if isinstance(phydrax, Mapping) else None
    )
    if observed_ids is None:
        losses = (
            AdapterLoss(
                "extras.phydrax.stable_particle_ids",
                "import",
                "synthesized",
                "QCSchema result did not retain native IDs; request order was used.",
                changes_interpretation=False,
            ),
        )
    elif list(observed_ids) != expected_ids:
        raise ValueError("QCSchema result changed native particle order.")
    provenance = record.get("provenance", {})
    provenance_id = canonical_fingerprint(
        {"kind": "qcschema-result-provenance", "content": provenance}
    )
    convergence = ElectronicConvergenceEvidence(
        success,
        message="qcschema-success" if success else "qcschema-failure",
    )
    work = ElectronicWorkEvidence(
        energy_evaluations=1,
        force_evaluations=int(calculation.request.requires(ElectronicProperty.FORCES)),
        hessian_evaluations=int(calculation.request.requires(ElectronicProperty.HESSIAN)),
        property_evaluations=int(calculation.request.requires(ElectronicProperty.DIPOLE)),
    )
    result = make_electronic_evaluation(
        calculation,
        provider_id,
        positions,
        energy,
        forces=forces,
        hessian=hessian,
        dipole=dipole,
        cell_vectors=cell_vectors,
        convergence=convergence,
        work=work,
        status=(
            ElectronicCalculationStatus.SUCCESS
            if success
            else ElectronicCalculationStatus.BACKEND_FAILED
        ),
        source_unit_ids=_source_units(),
        artifact_ids=(provenance_id,),
    )
    source_id = canonical_fingerprint(
        {"kind": "qcschema-atomic-result", "content": dict(record)}
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "QCSchema AtomicResult",
        type(result).__name__,
        source_id=source_id,
        target_id=result.result_id,
        coordinate_mapping=(
            "QCSchema gradient[atom,xyz] -> -forces[atom,xyz]",
            "active request order -> native stable particle order",
        ),
        preserved_fields=(
            "energy",
            "forces",
            "hessian",
            "total_charge",
            "spin_multiplicity",
            "model",
            "provenance",
        ),
        assumptions=("QCSchema result quantities use atomic units.",),
        losses=losses,
    )
    return result, report


__all__ = [
    "electronic_calculation_to_qcelemental",
    "electronic_calculation_to_qcschema",
    "electronic_evaluation_from_qcschema",
    "is_qcelemental_available",
    "require_qcelemental",
]
