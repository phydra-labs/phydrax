#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native multistate and controlled-Hamiltonian benchmark"
    )
    parser.add_argument(
        "--scenario", choices=("runtime", "alchemy", "all"), default="all"
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _runtime_fixture(iterations: int) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        topology=phx.atomistic.MolecularTopologyPlan(bonds=[[10, 20]]),
    ).prepare()
    # The bonded-only Hamiltonian declares no neighborhood requirement: its
    # bundle is explicitly empty.
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.HarmonicBondPotential([100.0], [1.0])]
    ).prepare(
        system,
        neighborhood_bundle=phx.atomistic.AtomisticNeighborhoodBundlePlan((), (), ()),
    )
    # Canonical NVT replicas need a canonical thermostat; velocity Verlet is NVE.
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.BAOABLangevinPlan(1.0e-7, 1.0),
    ).prepare()
    measure = phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system)
    state_plans = tuple(
        phx.atomistic.AtomisticThermodynamicStatePlan(
            measure,
            ensemble="nvt",
            temperature=temperature,
            state_id=f"temperature-{temperature:g}",
        )
        for temperature in (1.0, 2.0)
    )
    table = phx.atomistic.PreparedThermodynamicStateTable(dynamics, state_plans)
    multistate = phx.atomistic.sampling.AtomisticMultistatePlan(
        table,
        (101, 202),
        qualification=phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
            dynamics,
            table,
            "multistate-benchmark-demonstration-qualification",
            sampling_exact=False,
            sampling_bias_bound=1.0,
        ),
        exchange=phx.atomistic.sampling.AtomisticReplicaExchangePlan(1),
        run_id="multistate-free-energy-benchmark",
    ).prepare(dynamics)
    positions = (
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
    )
    lanes = tuple(
        dynamics.initialize_state(
            value,
            table,
            state_index=index,
            velocity=jnp.zeros_like(value),
            key=jax.random.key(100 + index),
        )
        for index, value in enumerate(positions)
    )
    initial = multistate.initialize(lanes, (0, 1), jax.random.key(41))
    segment = phx.atomistic.sampling.AtomisticMultistateSegmentPlan(
        multistate,
        iterations,
        0,
        0,
        multistate.initial_continuation_id,
    )
    return segment, initial


def _runtime_benchmark(iterations: int, repeats: int) -> dict:
    segment, initial = _runtime_fixture(iterations)
    function = eqx.filter_jit(segment.run)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(initial), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(initial), warmup=1, repeats=repeats
    )
    dataset = phx.uq.reduced_potential_dataset_from_multistate(result)
    analysis = phx.uq.multistate_bennett_acceptance_ratio(
        dataset,
        phx.uq.FreeEnergySelectionPlan(block_length=1),
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    active = int(jnp.sum(result.sample_active))
    elapsed = execution.mean_seconds
    return {
        "configuration": {
            "iterations": iterations,
            "states": len(result.state_ids),
            "replicas": result.sample_active.shape[1],
            "dtype": str(result.reduced_potentials.dtype),
            "unit_id": result.unit_id,
            "sampling_exact": result.sampling_exact,
            "sampling_bias_bound": result.sampling_bias_bound,
        },
        "identities": {
            "producer": result.producer_id,
            "measure": result.measure_id,
            "run": result.run_id,
            "segment": result.segment_id,
            "dataset": dataset.dataset_id,
            "analysis": analysis.analysis_id,
            "qualification": result.qualification_id,
        },
        "compilation": asdict(compilation),
        "compiler": asdict(compiler),
        "execution": execution.to_dict(),
        "performance": {
            "accepted_draws_per_second": active / elapsed,
            "cross_potential_evaluations_per_second": (
                active * len(result.state_ids) / elapsed
            ),
        },
        "memory": {
            "initial_logical_bytes": logical_array_bytes(initial),
            "segment_logical_bytes": logical_array_bytes(result),
            "dataset_logical_bytes": logical_array_bytes(dataset),
        },
        "analysis": {
            "free_energies": np.asarray(analysis.free_energies).tolist(),
            "standard_errors": np.asarray(analysis.standard_errors).tolist(),
            "minimum_overlap": float(jnp.min(analysis.overlap)),
            "raw_effective_sample_size": np.asarray(
                analysis.raw_effective_sample_size
            ).tolist(),
            "successful": bool(analysis.successful),
            "numerical_status": int(analysis.numerical_status),
            "statistical_status": int(analysis.statistical_status),
        },
        "gates": {
            "segment_complete": int(result.count) == iterations,
            "dense_coverage": bool(jnp.all(result.coverage[result.iteration_valid])),
            "analysis_numerically_successful": int(analysis.numerical_status) == 0,
            "approximation_declared": (
                (not result.sampling_exact) and result.sampling_bias_bound > 0.0
            ),
            "finite": bool(jnp.all(jnp.isfinite(analysis.free_energies))),
        },
    }


def _controlled_fixture() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 1],
        charges=[1.0, -1.0, 0.0],
        region_ids=[1, 1, 0],
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential([0.4, 0.8], [1.0, 1.2], 3.0),
            phx.atomistic.DirectCoulombPotential(),
        ]
    )
    (sterics,) = tuple(
        value for value in program.requirements.neighborhoods if not value.complete
    )
    (coulomb,) = tuple(
        value for value in program.requirements.neighborhoods if value.complete
    )
    bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (sterics, coulomb),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "steric-pairs",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                (sterics,),
                phx.atomistic.AtomisticDenseImageSearch(maximum_dense_routes=9),
                phx.discretization.ParticleImageCapacityLadder(
                    (
                        phx.discretization.ParticleImageCapacity(
                            maximum_particles_per_cell=3,
                            maximum_edges=6,
                            maximum_degree=2,
                            maximum_images=1,
                            maximum_case_edges=6,
                        ),
                    )
                ),
                skin=0.3,
            ),
            phx.atomistic.AtomisticCompletePairSource(
                "coulomb-pairs",
                phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS,
                (coulomb,),
                maximum_pairs=3,
            ),
        ),
        (
            phx.atomistic.AtomisticRoleCapacityLadder(
                sterics,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=3, maximum_degree=2
                    ),
                ),
            ),
        ),
        frame=phx.discretization.PeriodicCell(
            16.0 * np.eye(3), origin=np.full((3,), -8.0), periodic_axes=(False,) * 3
        ),
    )
    force_field = phx.atomistic.AtomisticForceFieldPlan(
        system,
        program,
        phx.atomistic.AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        phx.atomistic.AtomisticForceFieldProvenance(
            "native", ("benchmark-parameters",), "benchmark", "controlled"
        ),
    ).prepare(neighborhood_bundle=bundle)
    schedule = phx.atomistic.AlchemicalControlSchedulePlan(
        ("coupled", "middle", "decoupled"),
        ("solute-sterics", "solute-electrostatics"),
        (
            phx.atomistic.AlchemicalControlKind.STERICS,
            phx.atomistic.AlchemicalControlKind.ELECTROSTATICS,
        ),
        jnp.asarray([[1.0, 1.0], [0.4, 0.6], [0.0, 0.0]]),
    )
    partition = phx.atomistic.AlchemicalInteractionPartitionPlan(
        schedule.control_ids,
        ([10, 20], [10, 20]),
        mapped_particle_ids=[[10, 10], [20, 20]],
    )
    controlled = phx.atomistic.ControlledHamiltonianPlan(
        force_field, schedule, partition
    ).prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.2, 0.2, 0.0]])
    prepared_bundle = controlled.neighborhood_bundle
    neighborhoods = prepared_bundle.initialize(prepared_bundle.geometry(positions[None]))
    return controlled, positions, neighborhoods


def _alchemy_benchmark(repeats: int) -> dict:
    controlled, positions, neighborhoods = _controlled_fixture()
    function = eqx.filter_jit(
        lambda value: controlled.evaluate(value, neighborhoods, state_index=1)
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(positions), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(positions), warmup=1, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "configuration": {
            "states": controlled.plan.schedule.state_count,
            "controls": controlled.plan.schedule.control_count,
            "particles": controlled.system.capacity,
            "dtype": str(positions.dtype),
        },
        "identities": {
            "hamiltonian": controlled.prepared_id,
            "schedule": controlled.plan.schedule.schedule_id,
            "partition": controlled.plan.partition.partition_id,
        },
        "compilation": asdict(compilation),
        "compiler": asdict(compiler),
        "execution": execution.to_dict(),
        "performance": {
            "evaluations_per_second": 1.0 / execution.mean_seconds,
        },
        "memory": {
            "input_logical_bytes": logical_array_bytes(positions),
            "result_logical_bytes": logical_array_bytes(result),
        },
        "physics": {
            "energy": float(result.energy),
            "control_derivative": np.asarray(result.dU_dcontrols).tolist(),
            "successful": bool(result.successful),
        },
        "gates": {
            "successful": bool(result.successful),
            "finite_energy": bool(jnp.isfinite(result.energy)),
            "finite_control_derivative": bool(jnp.all(jnp.isfinite(result.dU_dcontrols))),
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    iterations = 2 if arguments.smoke else arguments.iterations
    repeats = 1 if arguments.smoke else arguments.repeats
    if iterations < 1 or repeats < 1:
        raise ValueError("iterations and repeats must be positive")
    payload = {
        "runtime": (
            _runtime_benchmark(iterations, repeats)
            if arguments.scenario in ("runtime", "all")
            else None
        ),
        "alchemy": (
            _alchemy_benchmark(repeats)
            if arguments.scenario in ("alchemy", "all")
            else None
        ),
        "environment": asdict(capture_environment()),
    }
    active_gates = [
        value
        for scenario in (payload["runtime"], payload["alchemy"])
        if scenario is not None
        for value in scenario["gates"].values()
    ]
    payload["all_successful"] = all(active_gates)
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
