#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _h2():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        [1.008, 1.008],
        units.scale,
        particle_ids=[11, 17],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        [_EXPONENTS, _EXPONENTS],
        [_COEFFICIENTS, _COEFFICIENTS],
        source_id="sto-3g-hydrogen-benchmark",
    ).prepare(system)
    plan = phx.chemistry.NativeRHFPlan(
        system,
        basis,
        2,
        convergence_tolerance=1.0e-9,
        damping=0.2,
    )
    positions_bohr = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    return structure, plan, positions_bohr


def _periodic():
    cell = phx.discretization.PeriodicCell(
        5.0 * np.eye(3), periodic_axes=(True, True, True)
    )
    mesh = phx.chemistry.KPointMeshPlan.monkhorst_pack((4, 1, 1))
    model = phx.chemistry.PeriodicAOModelPlan(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[[-0.2]], [[-1.0]], [[-0.2]]],
        [[[0.0]], [[1.0]], [[0.0]]],
        [0.0],
        [1.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    return phx.chemistry.NativePeriodicSCFPlan(
        cell,
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(1.0),
        model,
        smearing_energy=0.05,
    )


def benchmark(repeats: int):
    structure, rhf, positions_bohr = _h2()
    rhf_samples = []
    state = None
    for _ in range(repeats):
        started = time.perf_counter()
        state = rhf.solve_atomic_units(positions_bohr)
        rhf_samples.append(time.perf_counter() - started)
    started = time.perf_counter()
    manifold = phx.chemistry.rhf_tamm_dancoff(
        rhf,
        structure.positions,
        state,
        phx.chemistry.ExcitedStateManifoldPlan(1),
    ).solve()
    tda_seconds = time.perf_counter() - started
    started = time.perf_counter()
    spectrum = phx.chemistry.UVVisibleSpectrumPlan(
        phx.chemistry.SpectralAxis.ENERGY_EV,
        phx.chemistry.SpectralLineShape.GAUSSIAN,
        1.0,
        40.0,
        fwhm=0.2,
        grid_size=4001,
    ).evaluate(manifold)
    spectrum_seconds = time.perf_counter() - started
    periodic = _periodic()
    periodic_samples = []
    periodic_result = None
    for _ in range(repeats):
        started = time.perf_counter()
        periodic_result = periodic.evaluate()
        periodic_samples.append(time.perf_counter() - started)
    started = time.perf_counter()
    ewald = phx.chemistry.PeriodicEwaldPlan(
        0.8, real_shell=2, reciprocal_shell=2
    ).evaluate(
        np.asarray([[1.0, 2.5, 2.5], [4.0, 2.5, 2.5]]),
        np.asarray([1.0, -1.0]),
        5.0 * np.eye(3),
    )
    ewald_seconds = time.perf_counter() - started
    started = time.perf_counter()
    profile = phx.chemistry.SpectralProfilePlan(
        phx.chemistry.SpectralLineShape.VOIGT,
        -20.0,
        20.0,
        grid_size=10001,
        fwhm=0.4,
        lorentzian_fwhm=0.2,
    ).evaluate([0.0], [1.0])
    profile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    network = phx.chemistry.ReactionNetworkPlan([[0.0, 2.0], [1.0, 0.0]]).propagate(
        [1.0, 0.0], np.linspace(0.0, 2.0, 101)
    )
    network_seconds = time.perf_counter() - started
    campaigns = phx.chemistry.candidate_chemistry_qualification_campaigns()
    return {
        "native_rhf": {
            "samples_seconds": rhf_samples,
            "median_seconds": statistics.median(rhf_samples),
            "iterations": int(state.iterations),
            "energy_hartree": float(state.total_energy),
            "state_id": state.state_id,
            "successful": bool(state.converged),
        },
        "tda": {
            "seconds": tda_seconds,
            "roots": int(manifold.excitation_energies.size),
            "result_id": manifold.result_id,
            "successful": bool(manifold.successful),
        },
        "uv_visible": {
            "seconds": spectrum_seconds,
            "grid_size": int(spectrum.grid.size),
            "area_residual": float(spectrum.area_residual),
            "result_id": spectrum.result_id,
            "successful": bool(spectrum.successful),
        },
        "periodic_scf": {
            "samples_seconds": periodic_samples,
            "median_seconds": statistics.median(periodic_samples),
            "k_points": int(periodic.mesh.fractional_points.shape[0]),
            "iterations": int(periodic_result.iterations),
            "result_id": periodic_result.result_id,
            "successful": bool(periodic_result.successful),
        },
        "periodic_ewald": {
            "seconds": ewald_seconds,
            "energy": float(ewald.energy),
            "force_balance_residual": float(ewald.force_balance_residual),
            "stress_symmetry_residual": float(ewald.stress_symmetry_residual),
            "result_id": ewald.result_id,
            "successful": bool(ewald.successful),
        },
        "voigt_profile": {
            "seconds": profile_seconds,
            "grid_size": int(profile.grid.size),
            "area_residual": float(profile.area_residual),
            "result_id": profile.result_id,
            "successful": bool(profile.successful),
        },
        "reaction_network": {
            "seconds": network_seconds,
            "time_points": int(network.times.size),
            "conservation_residual": float(network.conservation_residual),
            "plan_id": network.plan_id,
            "successful": bool(network.successful),
        },
        "qualification_campaigns": {
            "count": len(campaigns),
            "campaign_ids": [value.campaign_id for value in campaigns],
            "successful": len(campaigns) == 4,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/production_computational_chemistry.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive")
    cases = benchmark(arguments.repeats)
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "successful": all(bool(value["successful"]) for value in cases.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
