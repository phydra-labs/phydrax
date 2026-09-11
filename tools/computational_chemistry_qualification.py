#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


def qualification():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    equilibrium = np.asarray(
        [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]]
    )
    structure = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        equilibrium + 0.02,
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[5, 7, 11],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0, 0]
    )
    stiffness = 2.0

    def surface_evaluator(positions, _cell):
        coordinate = jnp.asarray(positions)
        delta = coordinate - jnp.asarray(equilibrium)
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            0.5 * stiffness * jnp.sum(delta**2),
            -stiffness * delta,
            None,
            True,
            provider_id="qualified-harmonic-surface",
            source_result_id=phx.chemistry.electronic_geometry_id(system, coordinate),
        )

    surface = phx.chemistry.CallablePotentialEnergySurface(
        surface_evaluator,
        system.system_id,
        units,
        "qualified-harmonic-surface",
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )
    optimized = phx.chemistry.MolecularGeometryOptimizationPlan(
        system,
        surface,
        convergence=phx.chemistry.MolecularGeometryConvergencePlan(
            maximum_force=1.0e-8,
            rms_force=1.0e-8,
            maximum_steps=64,
            maximum_evaluations=256,
        ),
    ).run(structure)
    hessian = phx.chemistry.MolecularHessianPlan(
        system,
        surface,
        displacement=1.0e-4,
        antisymmetry_tolerance=1.0e-10,
    ).evaluate(optimized.final_structure)
    vibration = phx.chemistry.VibrationalAnalysisPlan(system).evaluate(
        optimized.final_structure, hessian
    )
    pressure = 101325.0 * float(
        phx.units.conversion_factor(phx.units.PASCAL, units.pressure_unit)
    )
    thermochemistry = phx.chemistry.HarmonicThermochemistryPlan(
        system,
        298.15,
        pressure,
        symmetry_number=2,
        electronic_degeneracy=1,
    ).evaluate(
        optimized.final_structure,
        vibration,
        optimized.final_evaluation.energy,
    )
    state = phx.chemistry.MolecularElectronicStatePlan(0, 1)
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.ElectronicMethodPlan(
            phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,
            "analytic-dipole",
            phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        )
    )
    dipole_calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        state,
        model,
        phx.chemistry.ElectronicPropertyRequest(
            (
                phx.chemistry.ElectronicProperty.ENERGY,
                phx.chemistry.ElectronicProperty.FORCES,
                phx.chemistry.ElectronicProperty.DIPOLE,
            )
        ),
    )
    charges = np.asarray([-0.8, 0.4, 0.4])

    def electronic(plan, positions, cell):
        del cell
        coordinate = np.asarray(positions)
        delta = coordinate - equilibrium
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "qualified-dipole-provider",
            coordinate,
            0.5 * stiffness * np.sum(delta**2),
            forces=-stiffness * delta,
            dipole=np.sum(charges[:, None] * coordinate, axis=0),
        )

    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        (
            phx.chemistry.ElectronicProperty.ENERGY,
            phx.chemistry.ElectronicProperty.FORCES,
            phx.chemistry.ElectronicProperty.DIPOLE,
        ),
        (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
    )
    prepared = phx.chemistry.CallableElectronicProvider(
        electronic, "qualified-dipole-provider", capabilities
    ).prepare(dipole_calculation)
    spectrum = phx.chemistry.IRSpectrumPlan(
        system,
        prepared,
        displacement=1.0e-4,
        maximum_wavenumber=5000.0,
        grid_size=501,
    ).evaluate(optimized.final_structure, vibration)
    with tempfile.TemporaryDirectory(prefix="phydrax-chemistry-") as directory:
        path = Path(directory) / "result.phx"
        electronic_result = prepared.evaluate(optimized.final_structure.positions)
        phx.chemistry.write_electronic_result_archive(
            path, electronic_result, run_id="qualification"
        )
        restored = phx.chemistry.read_electronic_result_archive(path)
        archive_roundtrip = restored.result_id == electronic_result.result_id
    hessian_error = float(
        np.max(
            np.abs(
                np.asarray(hessian.hessian).reshape((9, 9))
                - stiffness * np.eye(9)
            )
        )
    )
    cases = {
        "geometry": {
            "maximum_force": float(optimized.maximum_force),
            "provider_evaluations": int(optimized.provider_evaluations),
            "passed": bool(optimized.successful),
        },
        "hessian": {
            "maximum_absolute_error": hessian_error,
            "antisymmetry_residual": float(hessian.antisymmetry_residual),
            "passed": bool(hessian.successful) and hessian_error < 1.0e-8,
        },
        "vibration": {
            "external_mode_count": vibration.external_mode_count,
            "internal_mode_count": vibration.internal_mode_count,
            "stationary_point": vibration.stationary_point.value,
            "passed": bool(vibration.successful)
            and vibration.stationary_point is phx.chemistry.StationaryPointKind.MINIMUM,
        },
        "thermochemistry": {
            "gibbs_energy": float(thermochemistry.gibbs_energy),
            "passed": bool(thermochemistry.successful),
        },
        "infrared": {
            "line_count": int(spectrum.wavenumbers.size),
            "maximum_intensity": float(jnp.max(spectrum.intensity)),
            "passed": bool(spectrum.successful)
            and bool(jnp.any(spectrum.line_strengths > 0.0)),
        },
        "archive": {"passed": archive_roundtrip},
    }
    return {
        "environment": capture_environment().to_dict(),
        "system_id": system.system_id,
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/computational_chemistry_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
