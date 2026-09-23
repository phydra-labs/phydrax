#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _surface(system, evaluator, provider_id):
    def wrapped(positions, _cell):
        energy, forces = evaluator(jnp.asarray(positions))
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            True,
            provider_id=provider_id,
            source_result_id=phx.chemistry.electronic_geometry_id(system, positions),
        )

    return phx.chemistry.CallablePotentialEnergySurface(
        wrapped,
        system.system_id,
        system.units,
        provider_id,
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )


def qualification():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    h2 = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        [1.008, 1.008],
        units.scale,
        particle_ids=[11, 17],
    )
    h2_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        h2, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        [_EXPONENTS, _EXPONENTS],
        [_COEFFICIENTS, _COEFFICIENTS],
        source_id="sto-3g-hydrogen-qualification",
    ).prepare(h2_system)
    rhf = phx.chemistry.NativeRHFPlan(
        h2_system,
        basis,
        2,
        convergence_tolerance=1.0e-9,
        damping=0.2,
    )
    positions_bohr = np.asarray(h2.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    rhf_state = rhf.solve_atomic_units(positions_bohr)
    manifold = phx.chemistry.rhf_tamm_dancoff(
        rhf,
        h2.positions,
        rhf_state,
        phx.chemistry.ExcitedStateManifoldPlan(1),
    ).solve()
    uv = phx.chemistry.UVVisibleSpectrumPlan(
        phx.chemistry.SpectralAxis.ENERGY_EV,
        phx.chemistry.SpectralLineShape.GAUSSIAN,
        1.0,
        40.0,
        fwhm=0.2,
        grid_size=4001,
        area_tolerance=1.0e-3,
    ).evaluate(manifold)

    water_positions = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
    water = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        water_positions,
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[2, 3, 5],
    )
    water_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        water, units, molecule_ids=[0, 0, 0]
    )
    hessian_array = np.eye(9).reshape((3, 3, 3, 3))
    hessian = phx.chemistry.MolecularHessianResult(
        hessian_array,
        hessian_array,
        0.0,
        1,
        True,
        units,
        water_system.system_id,
        water.structure_id,
        ("qualification-hessian",),
        "qualification-hessian-plan",
    )
    constrained = phx.chemistry.ConstrainedVibrationalAnalysisPlan(
        water_system,
        phx.chemistry.MolecularConstraintSetPlan.distances(
            water_system, [(2, 3)], [0.95]
        ),
    ).evaluate(water, hessian, np.zeros_like(water_positions))

    reactant = phx.atomistic.AtomicStructure(
        [1], [[-1.0, 0.0, 0.0]], [1.0], units.scale, particle_ids=[41]
    )
    product = phx.atomistic.AtomicStructure(
        [1], [[1.0, 0.0, 0.0]], [1.0], units.scale, particle_ids=[41]
    )
    reaction_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant, units, molecule_ids=[0]
    )

    def double_well(positions):
        x, y, z = positions[0]
        return (
            (x * x - 1.0) ** 2 + 0.5 * y * y + 0.5 * z * z,
            jnp.asarray([[-4.0 * x * (x * x - 1.0), -y, -z]]),
        )

    reaction = phx.chemistry.NudgedElasticBandPlan(
        reaction_system,
        _surface(reaction_system, double_well, "qualification-double-well"),
        image_count=5,
        climbing_start=0,
        force_tolerance=1.0e-8,
        maximum_steps=10,
    ).run(reactant, product)

    full = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[51, 53],
    )
    full_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        full, units, molecule_ids=[0, 0]
    )
    region = phx.chemistry.QuantumRegionPlan(
        full_system, [51], boundary_bonds=[(51, 53)]
    ).prepare()

    def quadratic(scale):
        return lambda positions: (
            0.5 * scale * jnp.sum(positions**2),
            -scale * positions,
        )

    qmmm = phx.chemistry.SubtractiveQMMMSurface(
        region,
        _surface(full_system, quadratic(1.0), "qualification-full-mm"),
        _surface(region.region_system, quadratic(2.0), "qualification-model-qm"),
        _surface(region.region_system, quadratic(1.0), "qualification-model-mm"),
    ).evaluate_components(full.positions)

    periodic_cell = phx.discretization.PeriodicCell(
        5.0 * np.eye(3), periodic_axes=(True, True, True)
    )
    periodic_basis = phx.chemistry.PeriodicOrbitalBasisPlan(
        periodic_cell,
        ("s",),
        [[0.0, 0.0, 0.0]],
        phx.units.ANGSTROM,
        phx.chemistry.PeriodicBlochGauge("lattice"),
    )
    periodic_family = (
        phx.operators.periodic.periodic_translation_family_from_dense_blocks(
            [[0, 0, 0]], np.asarray([-1.0]).reshape((1, 1, 1, 1, 1))
        )
    )
    periodic_pencil = phx.chemistry.PeriodicOrbitalPencilPlan.orthonormal(
        periodic_basis,
        periodic_family.plan,
        periodic_family.state,
        phx.units.ELECTRONVOLT,
    ).prepare()
    periodic_mean_field = phx.chemistry.PeriodicHubbardMeanFieldPlan(
        periodic_basis, [0.0], [2.0], 0.0, phx.units.ELECTRONVOLT
    )
    periodic = phx.chemistry.NativePeriodicSCFPlan(
        periodic_cell,
        phx.discretization.ReciprocalMeshPlan.monkhorst_pack(periodic_cell, (1, 1, 1)),
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        periodic_pencil,
        periodic_mean_field,
    ).evaluate()

    cases = {
        "native_rhf": {
            "energy_hartree": float(rhf_state.total_energy),
            "residual": float(rhf_state.residual),
            "passed": bool(rhf_state.converged)
            and abs(float(rhf_state.total_energy) + 1.11676) < 2.0e-4,
        },
        "excited_uv": {
            "excitation_hartree": float(manifold.excitation_energies[0]),
            "area_residual": float(uv.area_residual),
            "passed": bool(manifold.successful) and bool(uv.successful),
        },
        "constrained_modes": {
            "constraint_rank": constrained.constraint_rank,
            "mode_count": constrained.vibration.internal_mode_count,
            "hessian_system_id": hessian.system_id,
            "hessian_geometry_id": hessian.geometry_id,
            "passed": bool(constrained.successful)
            and hessian.system_id == water_system.system_id
            and hessian.geometry_id == water.structure_id
            and constrained.constraint_rank == 1
            and constrained.vibration.internal_mode_count == 2,
        },
        "reaction_path": {
            "climbing_energy": float(reaction.energies[reaction.climbing_image]),
            "passed": bool(reaction.successful),
        },
        "qmmm": {
            "energy": float(qmmm.total_energy),
            "passed": bool(qmmm.successful),
        },
        "periodic_scf": {
            "energy": float(periodic.energy),
            "electron_population": float(np.sum(np.asarray(periodic.populations))),
            "passed": bool(periodic.successful)
            and abs(float(periodic.energy) + 2.0) < 1.0e-12,
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(bool(value["passed"]) for value in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/production_computational_chemistry_qualification.json"),
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
