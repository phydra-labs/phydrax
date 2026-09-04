#!/usr/bin/env python3

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


SEEDS = (17, 29, 43)
CHEMICAL_ACCURACY_HARTREE = 1.593601e-3
STATISTICAL_SIGMA_GATE = 3.0
DETERMINANT_COUNT = 16
MAXIMUM_PAIR_ELEMENTS = 12
MAXIMUM_DETERMINANT_WORK = 128


@dataclass(frozen=True)
class MolecularCase:
    name: str
    charges: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    electron_count: int
    spin_up_count: int
    reference_energy_hartree: float


CASES = (
    MolecularCase("H", (1,), ((0.0, 0.0, 0.0),), 1, 1, -0.5),
    MolecularCase("He", (2,), ((0.0, 0.0, 0.0),), 2, 1, -2.9037243770341196),
    MolecularCase(
        "H2",
        (1, 1),
        ((-0.7, 0.0, 0.0), (0.7, 0.0, 0.0)),
        2,
        1,
        -1.174475714220443,
    ),
)


def _structure(case: MolecularCase):
    scale = phx.atomistic.AtomisticScaleContract(phx.units.BOHR, phx.units.HARTREE)
    return phx.atomistic.AtomicStructure(
        jnp.asarray(case.charges, dtype=jnp.int32),
        jnp.asarray(case.positions, dtype=jnp.float64),
        jnp.ones((len(case.charges),), dtype=jnp.float64),
        scale,
        name=case.name,
    )


def _problem(case: MolecularCase, seed: int):
    resource_plan = phx.operators.ElectronicVMCResourcePlan(
        case.electron_count,
        determinant_count=DETERMINANT_COUNT,
        maximum_pair_elements=MAXIMUM_PAIR_ELEMENTS,
        maximum_determinant_work=MAXIMUM_DETERMINANT_WORK,
    )
    nuclei = _structure(case)
    model = phx.nn.quantum.FermiNet(
        nuclei,
        case.electron_count,
        case.spin_up_count,
        hidden_features=64,
        pair_features=32,
        layer_count=4,
        determinant_count=resource_plan.determinant_count,
        compute_dtype="float64",
        resource_plan=resource_plan,
        key=jr.key(seed),
    )
    operator = phx.operators.ElectronicCoulombHamiltonian(
        nuclei,
        case.electron_count,
        kinetic=phx.operators.ElectronicKineticPolicy(
            trace_method="chunked-exact",
            coordinate_chunk_size=3,
            compute_dtype="float64",
        ),
        resource_plan=resource_plan,
    )
    walkers = phx.operators.electronic_initial_walkers(
        jr.fold_in(jr.key(seed), 1), nuclei, case.electron_count, 64
    )
    kernel = phx.sampling.MetropolisHastings(
        phx.operators.harmonic_mean_electron_proposal(
            nuclei, case.electron_count, step_size=0.2
        )
    )
    return phx.solver.VariationalMonteCarloProblem(
        model,
        operator,
        kernel,
        walkers,
        problem_id=f"electronic-benchmark:{case.name}:seed={seed}",
    )


def _policy():
    return phx.solver.VariationalMonteCarloPolicy(
        num_iterations=200,
        draws_per_iteration=32,
        steps_per_draw=4,
        warmup_steps=100,
        final_evaluation_draws=1024,
        learning_rate=0.03,
        damping=1e-3,
        max_update_norm=0.1,
        failure_mode="record",
        final_chain_diagnostics=True,
    )


def _diagnostic_scalar(tree, name):
    value = tree[name]
    return float(jnp.min(jnp.asarray(value)))


def _run(case: MolecularCase, seed: int, *, compilation_expected: bool):
    problem = _problem(case, seed)
    policy = _policy()
    started = perf_counter()
    result = phx.solver.solve_variational_monte_carlo(
        problem, policy, key=jr.fold_in(jr.key(seed), 2)
    )
    jax.block_until_ready(result.final_estimate.energy)
    elapsed = perf_counter() - started
    estimate = result.final_estimate
    diagnostics = estimate.chain_diagnostics
    if diagnostics is None:
        raise RuntimeError("Electronic benchmark requires final chain diagnostics.")
    bulk_ess = _diagnostic_scalar(diagnostics.bulk_ess, "local_energy_real")
    tail_ess = _diagnostic_scalar(diagnostics.tail_ess, "local_energy_real")
    rhat = float(jnp.max(jnp.asarray(diagnostics.rhat["local_energy_real"])))
    variance = float(estimate.variance)
    standard_error = float(np.sqrt(variance / bulk_ess))
    energy = float(estimate.physical_energy)
    absolute_error = abs(energy - case.reference_energy_hartree)
    statistical_tolerance = STATISTICAL_SIGMA_GATE * standard_error
    chemical_pass = absolute_error <= CHEMICAL_ACCURACY_HARTREE
    statistical_pass = absolute_error <= statistical_tolerance
    return {
        "case": case.name,
        "seed": seed,
        "energy_hartree": energy,
        "reference_energy_hartree": case.reference_energy_hartree,
        "absolute_error_hartree": absolute_error,
        "mc_standard_error_hartree": standard_error,
        "variance_hartree2": variance,
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "rhat": rhat,
        "acceptance": float(estimate.acceptance_rate),
        "compile_and_runtime_seconds"
        if compilation_expected
        else "runtime_seconds": elapsed,
        "parameter_count": problem.parameter_subspace.total_dimension,
        "status": phx.solver.vmc_status_name(estimate.status),
        "gates": {
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
            "chemical_pass": chemical_pass,
            "statistical_sigma": STATISTICAL_SIGMA_GATE,
            "statistical_tolerance_hartree": statistical_tolerance,
            "statistical_pass": statistical_pass,
            "release_pass": bool(chemical_pass and statistical_pass),
        },
        "provenance": {
            "operator_id": problem.operator.operator_id,
            "operator_method": estimate.local.method_id,
            "operator_compute_dtype": estimate.local.compute_dtype,
            "operator_coordinate_work_per_sample": int(
                jnp.max(estimate.local.work_count)
            ),
            "network_id": problem.model.network_id,
            "problem_id": problem.problem_id,
            "scale_id": problem.operator.nuclei.scale.scale_id,
            "proposal_id": problem.kernel.proposal.proposal_id,
            "kernel_id": problem.kernel.kernel_id,
            "trace_cost_claim": (
                "exact coordinate second derivatives; linear in coordinate count "
                "times derivative-action cost"
            ),
            "resource_admission": {
                "claim": problem.model.resource_plan.claim,
                "valid": bool(problem.model.resource_plan.valid),
                "electron_count": problem.model.resource_plan.electron_count,
                "determinant_count": problem.model.resource_plan.determinant_count,
                "pair_stream_elements": problem.model.resource_plan.pair_stream_elements,
                "determinant_work": problem.model.resource_plan.determinant_work,
                "maximum_pair_elements": (
                    problem.model.resource_plan.admitted_pair_elements
                ),
                "maximum_determinant_work": (
                    problem.model.resource_plan.admitted_determinant_work
                ),
            },
        },
    }


def main():
    records = []
    for case in CASES:
        for seed_index, seed in enumerate(SEEDS):
            records.append(_run(case, seed, compilation_expected=seed_index == 0))
    summaries = []
    for case in CASES:
        selected = [record for record in records if record["case"] == case.name]
        energies = np.asarray([record["energy_hartree"] for record in selected])
        summaries.append(
            {
                "case": case.name,
                "seed_count": len(selected),
                "mean_energy_hartree": float(np.mean(energies)),
                "seed_standard_deviation_hartree": float(np.std(energies, ddof=1)),
                "all_release_gates_pass": all(
                    record["gates"]["release_pass"] for record in selected
                ),
            }
        )
    payload = {
        "campaign": "fixed-multiseed-electronic-vmc",
        "seeds": list(SEEDS),
        "cases": [case.name for case in CASES],
        "records": records,
        "summaries": summaries,
        "provenance": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "finite_nonperiodic": True,
            "resource_limits": {
                "determinant_count": DETERMINANT_COUNT,
                "maximum_pair_elements": MAXIMUM_PAIR_ELEMENTS,
                "maximum_determinant_work": MAXIMUM_DETERMINANT_WORK,
            },
            "born_oppenheimer": True,
            "relativistic": False,
            "stochastic_trace": False,
            "energy_interpretation": (
                "finite-sample VMC estimate; upper-bound language requires "
                "separately established estimator conditions"
            ),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
