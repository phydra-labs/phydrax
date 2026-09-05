#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Synchronized reference-conditioned insertion benchmark, not biological calibration.

Run: .venv/bin/python benchmarks/protein_cotranslation.py --residues 6 --steps 8
"""

from __future__ import annotations

import argparse
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax.applications.protein_folding._construct import ProteinConstruct
from phydrax.applications.protein_folding.cotranslation import (
    CotranslationProtocol,
    CotranslationStage,
    RibosomeBoundaryPotential,
)
from phydrax.applications.protein_folding.cotranslation._protocol import _step
from phydrax.atomistic import (
    AtomisticDynamicsPlan,
    AtomisticPotentialProgram,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    HarmonicBondPotential,
    LennardJonesPotential,
    MolecularTopologyPlan,
    VelocityVerletPlan,
)
from phydrax.atomistic._topology_epoch import prepare_dormant_system
from phydrax.discretization import DenseParticleNeighborhoodPlan
from phydrax.qualification import ReferenceArtifactManifest


def prepare_case(residues: int, steps: int):
    ids = tuple(1001 + 17 * i for i in range(residues))
    bonds = tuple(zip(ids[:-1], ids[1:], strict=True))
    topology = MolecularTopologyPlan(
        bonds=bonds,
        pair_exceptions=bonds,
        lennard_jones_scales=np.zeros(residues - 1),
        electrostatic_scales=np.zeros(residues - 1),
    )
    material = AtomisticSystemPlan(
        ids,
        np.zeros(residues, dtype=int),
        np.ones(residues),
        AtomisticUnitSystem.reduced(),
        element_mask=np.zeros(residues, dtype=bool),
        atom_type_ids=np.zeros(residues, dtype=int),
        molecule_ids=np.zeros(residues, dtype=int),
        topology=topology,
    ).prepare()
    stages = []
    for count in range(1, residues + 1):
        system = prepare_dormant_system(material, ids[:count])
        potential = AtomisticPotentialProgram(
            [
                HarmonicBondPotential([4.0], [1.0]),
                LennardJonesPotential([0.02], [0.5], 3.0, switch_distance=2.5),
                RibosomeBoundaryPotential(
                    tether_particle_id=ids[count - 1],
                    anchor=[1.1 * (count - 1), 0, 0],
                    tether_stiffness=0.1,
                    sphere_centers=[[-2.0, 0, 0]],
                    sphere_radii=[1.0],
                    exclusion_stiffness=0.2,
                ),
            ]
        ).prepare(system)
        runtime = AtomisticDynamicsPlan(
            system,
            potential,
            DenseParticleNeighborhoodPlan(residues * (residues - 1) // 2).prepare(
                system.particles
            ),
            VelocityVerletPlan(1e-3),
        ).prepare()
        stages.append(
            CotranslationStage(
                runtime,
                count,
                steps,
                "GCU",
                f"analytical-codon-stage:{count}",
                ((1.1 * (count - 1), 0.0, 0.0),) if count > 1 else (),
                ((0.01, 0.0, 0.0),) if count > 1 else (),
            )
        )
    specification = json.dumps(
        {
            "ids": ids,
            "bond_k": 4.0,
            "bond_length": 1.0,
            "lj_epsilon": 0.02,
            "lj_sigma": 0.5,
            "step": 1e-3,
            "dwell_steps": steps,
            "geometry": "linear-1.1-spacing",
            "tether_k": 0.1,
            "sphere_center": [-2.0, 0, 0],
            "sphere_radius": 1.0,
            "sphere_k": 0.2,
        },
        sort_keys=True,
    ).encode()
    source = ReferenceArtifactManifest(
        "independent-cotranslation-numerical-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(specification).hexdigest(),
        size_bytes=len(specification),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"reduced-length": 1.0, "reduced-energy": 1.0},
        uncertainty=None,
        lineage_ids=("analytical-reference-conditioned-fixture",),
    )
    protocol = CotranslationProtocol(
        ProteinConstruct(("A",), ("A" * residues,)), ids, tuple(stages), source, source
    )
    initial = protocol.initialize(
        jnp.zeros((residues, 3)), jnp.zeros((residues, 3)), key=jax.random.key(91)
    )
    return protocol, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residues", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.residues < 2 or args.steps < 1 or args.repeats < 1:
        parser.error("residues >= 2, steps >= 1 and repeats >= 1 are required")
    jax.config.update("jax_enable_x64", True)
    (protocol, initial), preparation = measure_synchronized(
        lambda: prepare_case(args.residues, args.steps)
    )
    compilation = []
    positions = (
        jnp.zeros((args.residues, 3)).at[:, 0].set(1.1 * jnp.arange(args.residues))
    )
    for stage in protocol.stages:
        runtime = stage.runtime
        state = runtime.initialize_state(
            positions, momentum=jnp.zeros_like(positions), key=jax.random.key(91)
        )
        compiled, timing = measure_lower_and_compile(
            lambda: _step.lower(runtime, state),
            lambda lowered: lowered.compile(),
        )
        evaluated, elapsed = measure_repeated(
            lambda: compiled(runtime, state), warmup=1, repeats=args.repeats
        )
        if not bool(evaluated.successful):
            raise RuntimeError("Benchmark native epoch step was rejected.")
        compilation.append(
            {
                "active": stage.nascent_residue_count,
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "step_milliseconds": elapsed.to_dict(unit="milliseconds"),
            }
        )
    result, elapsed = measure_repeated(
        lambda: protocol.run(initial), warmup=1, repeats=args.repeats
    )
    if not result.successful:
        raise RuntimeError(result.refusal)
    energy = result.cursor.state.energy
    balance = (
        energy.total_energy
        - energy.initial_kinetic_energy
        - energy.initial_potential_energy
        - energy.external_work
        - energy.constraint_work
        - energy.thermostat_heat
    )
    print(
        json.dumps(
            {
                "environment": capture_environment().to_dict(),
                "capacity": args.residues,
                "active_sizes": [s.nascent_residue_count for s in protocol.stages],
                "preparation_seconds": preparation,
                "epoch_compilation": compilation,
                "protocol_milliseconds": elapsed.to_dict(unit="milliseconds"),
                "logical_array_bytes": logical_array_bytes(
                    (protocol.stages[-1].runtime, result.cursor.state)
                ),
                "accepted_native_steps": int(result.cursor.state.step_index),
                "inserted_mass": sum(float(x.mass_source) for x in result.insertions),
                "insertion_work": sum(float(x.external_work) for x in result.insertions),
                "conservation_balance": float(balance),
                "ledger_balance_residual": float(energy.cumulative_balance_residual),
                "reference_conditioned": True,
                "biological_timing_calibrated": False,
                "scientific_scope": "Caller-parameterized numerical fixture; no protein accuracy claim",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
