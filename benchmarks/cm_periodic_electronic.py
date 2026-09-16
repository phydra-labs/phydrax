#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


periodic = phx.chemistry.periodic


def _measure(function, warmup: int, repeats: int):
    for _ in range(warmup):
        value = function()
        jax.block_until_ready(value)
    samples = []
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        jax.block_until_ready(value)
        samples.append(perf_counter() - start)
    values = np.asarray(samples)
    return value, {
        "minimum_seconds": float(np.min(values)),
        "median_seconds": float(np.median(values)),
        "maximum_seconds": float(np.max(values)),
        "repeats": repeats,
    }


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _cell():
    return phx.discretization.PeriodicCell(
        np.asarray([[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.3, 0.2, 5.2]]),
        periodic_axes=(True, True, True),
    )


def _manifest(source_id: str):
    return periodic.PeriodicProvenanceManifest.for_bytes(
        f"{source_id}:benchmark".encode(), source_id, "benchmark-generated"
    )


def _ewald_case(shell: int, warmup: int, repeats: int):
    prepared = periodic.PeriodicEwaldPlan(
        _cell(), _units(), 0.75, real_shell=shell, reciprocal_shell=shell
    ).prepare()
    positions = np.asarray([[0.2, 0.3, 0.4], [1.3, 0.7, 0.9]])
    result, timing = _measure(
        lambda: prepared.evaluate(positions, [1.0, -1.0]), warmup, repeats
    )
    return {
        "axes": {
            "particle_count": 2,
            "real_shell": shell,
            "reciprocal_shell": shell,
            "real_image_count": (2 * shell + 1) ** 3,
            "reciprocal_mode_count": (2 * shell + 1) ** 3 - 1,
        },
        "identity": {
            "plan_id": prepared.plan.plan_id,
            "prepared_id": prepared.prepared_id,
            "result_id": result.result_id,
        },
        "timing": timing,
        "scientific_residuals": {
            "energy_closure": float(result.evidence.energy_closure_residual),
            "force_balance": float(result.evidence.force_balance_residual),
            "stress_symmetry": float(result.evidence.stress_symmetry_residual),
        },
        "successful": bool(result.successful),
    }


def _gdf_case(orbital_count: int, warmup: int, repeats: int):
    manifest = _manifest(f"gdf-benchmark-{orbital_count}")
    factors = np.zeros((orbital_count, orbital_count, orbital_count))
    for index in range(orbital_count):
        factors[index, index, index] = 0.2
    tensor = phx.operators.quantum.gaussian.FactorizedERITensor(
        factors, 0.0, manifest.source_id, "gdf"
    )
    plan = periodic.GammaGDFPlan(
        np.diag(np.linspace(-1.0, 1.0, orbital_count)),
        np.eye(orbital_count),
        tensor,
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        manifest,
    )
    result, timing = _measure(plan.evaluate, warmup, repeats)
    return {
        "axes": {
            "orbital_count": orbital_count,
            "factor_rank": tensor.rank,
            "electron_count": 2,
        },
        "identity": {
            "plan_id": plan.plan_id,
            "result_id": result.result_id,
            "source_manifest_id": manifest.manifest_id,
        },
        "classification": result.classification,
        "timing": timing,
        "scientific_residuals": {
            "electron_count": float(result.evidence.electron_count_residual),
            "commutator": float(result.evidence.commutator_residual),
            "eigenpair": float(result.evidence.eigenpair_residual),
            "energy_closure": float(result.energy_ledger.closure_residual),
            "factorization_bound": float(result.evidence.factorization_residual),
        },
        "successful": bool(result.successful),
    }


def _fftdf_case(grid_size: int, warmup: int, repeats: int):
    manifest = _manifest(f"local-gth-grid-{grid_size}")
    pseudopotential = periodic.GTHPseudopotentialPlan(
        2.0,
        0.3,
        [0.1, -0.02, 0.0, 0.0],
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        manifest,
    )
    plan = periodic.GammaFFTDFPlan(
        _cell(),
        (grid_size, grid_size, grid_size),
        [[0.0, 0.0, 0.0]],
        (pseudopotential,),
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        smearing_energy=0.1,
        convergence_tolerance=1.0e-7,
        maximum_iterations=300,
        damping=0.5,
        maximum_grid_points=grid_size**3,
    )
    result, timing = _measure(plan.evaluate, warmup, repeats)
    return {
        "axes": {
            "grid_shape": [grid_size, grid_size, grid_size],
            "grid_points": grid_size**3,
            "electron_count": 2,
        },
        "identity": {
            "plan_id": plan.plan_id,
            "result_id": result.result_id,
            "source_manifest_id": manifest.manifest_id,
        },
        "classification": result.classification,
        "timing": timing,
        "scientific_residuals": {
            "electron_count": float(result.evidence.electron_count_residual),
            "commutator": float(result.evidence.commutator_residual),
            "eigenpair": float(result.evidence.eigenpair_residual),
            "energy_closure": float(result.energy_ledger.closure_residual),
        },
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    payload = {
        "benchmark": "periodic-electronic-production-and-candidate-kernels",
        "environment": capture_environment().to_dict(),
        "ewald": [
            _ewald_case(shell, arguments.warmup, arguments.repeats) for shell in (2, 3)
        ],
        "gamma_gdf": [
            _gdf_case(count, arguments.warmup, arguments.repeats) for count in (2, 4, 8)
        ],
        "gamma_local_gth_fftdf": [
            _fftdf_case(size, arguments.warmup, arguments.repeats) for size in (2, 3)
        ],
        "nonclaims": [
            "timing success is not qualification or release evidence",
            "finite Ewald shells require independent shell refinement",
            "supplied GDF integrals are not generated by this benchmark",
            "candidate local-GTH FFTDF omits nonlocal projectors and LDA correlation",
            "measured sizes do not imply an unrestricted resource envelope",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
