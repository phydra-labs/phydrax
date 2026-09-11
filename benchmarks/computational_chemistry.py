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
from ase.calculators.emt import EMT

import phydrax as phx
from benchmarks._runtime import capture_environment


def _case(repetitions: int):
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [29, 29],
        [[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]],
        [63.546, 63.546],
        units.scale,
        particle_ids=[1, 2],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.MolecularElectronicStatePlan(0, 1),
        phx.chemistry.ElectronicModelChemistryPlan(
            phx.chemistry.ElectronicMethodPlan(
                phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,
                "emt",
                phx.chemistry.ElectronicReferenceKind.RESTRICTED,
            )
        ),
        phx.chemistry.ElectronicPropertyRequest.energy_and_forces(),
    )
    provider = phx.chemistry.interchange.ASECalculatorProvider(
        EMT,
        "ase-emt",
        phx.chemistry.interchange.ASEElectronicStateBinding.invariant(),
        model_chemistry_id=calculation.model_chemistry.model_chemistry_id,
    )
    started = time.perf_counter()
    prepared = calculation.prepare(provider)
    prepare_seconds = time.perf_counter() - started
    samples = []
    last = None
    for _ in range(repetitions):
        started = time.perf_counter()
        last = prepared.evaluate(structure.positions)
        samples.append(time.perf_counter() - started)
    surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared)
    started = time.perf_counter()
    hessian = phx.chemistry.MolecularHessianPlan(
        system,
        surface,
        displacement=1.0e-3,
        antisymmetry_tolerance=2.0e-3,
    ).evaluate(structure)
    hessian_seconds = time.perf_counter() - started
    return {
        "provider_id": provider.provider_id,
        "prepare_seconds": prepare_seconds,
        "evaluation_seconds": samples,
        "evaluation_median_seconds": statistics.median(samples),
        "evaluations_per_second": repetitions / sum(samples),
        "hessian_seconds": hessian_seconds,
        "hessian_provider_evaluations": int(hessian.evaluation_count),
        "hessian_antisymmetry_residual": float(hessian.antisymmetry_residual),
        "energy": float(last.energy),
        "maximum_force": float(np.max(np.abs(np.asarray(last.forces)))),
        "successful": bool(last.successful) and bool(hessian.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/computational_chemistry.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive")
    payload = {
        "environment": capture_environment().to_dict(),
        "case": _case(arguments.repeats),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["case"]["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
