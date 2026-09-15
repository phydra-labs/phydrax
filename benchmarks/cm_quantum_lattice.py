#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    lower_quantum_lattice_to_vmc,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from phydrax.solver._quantum_lattice import (
    LocalHamiltonianQuantumLatticePolicy,
    lower_quantum_lattice_to_local_hamiltonian,
)
from phydrax.tensor_network._quantum_lattice import (
    lower_quantum_lattice_to_mpo,
    QuantumLatticeMPOPolicy,
)


def _model(mode_count: int):
    order = FermionModeOrder(tuple(f"m{index}" for index in range(mode_count)))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    local = {
        space.site_id: (
            LocalOperatorPlan(space, "create", create, (1,)),
            LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
        )
        for space in spaces
    }
    terms = tuple(
        QuantumLatticeTerm(
            (local[order.labels[index]][0], local[order.labels[index + 1]][1]),
            coefficient=-1.0,
            add_adjoint=True,
            label=f"hop:{index}:{index + 1}",
        )
        for index in range(mode_count - 1)
    )
    specification = QuantumLatticeSpecification(spaces, terms, fermion_mode_order=order)
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=2 * mode_count,
        maximum_factors_per_term=2,
        maximum_branches_per_input=8 * mode_count,
        maximum_sector_dimension=1_000_000,
        maximum_workspace_bytes=256 * 1024**2,
    )
    return order, specification, resources


def benchmark_case(mode_count: int, repeats: int):
    (order, specification, resources), model_seconds = measure_host(
        lambda: _model(mode_count)
    )
    prepared, prepare_seconds = measure_host(
        lambda: prepare_quantum_lattice(specification, resources)
    )
    basis = FixedCardinalityFermionBasis(
        order,
        mode_count // 2,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=1_000_000, maximum_table_bytes=16 * 1024**2
        ),
    )
    operator = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    vector = jnp.ones((basis.dimension,), dtype=jnp.complex128) / jnp.sqrt(
        basis.dimension
    )
    apply = eqx.filter_jit(lambda op, value: op.mv(value))
    executable, compilation = measure_lower_and_compile(
        lambda: apply.lower(operator, vector), lambda lowered: lowered.compile()
    )
    warm, warm_seconds = measure_synchronized(lambda: executable(operator, vector))
    steady_value, steady = measure_repeated(
        lambda: executable(operator, vector),
        warmup=0,
        repeats=repeats,
    )
    compiler = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiler-analysis",
    )
    local, local_seconds = measure_host(
        lambda: lower_quantum_lattice_to_local_hamiltonian(
            prepared,
            LocalHamiltonianQuantumLatticePolicy(maximum_term_matrix_elements=1_024),
        )
    )
    mpo, mpo_seconds = measure_host(
        lambda: lower_quantum_lattice_to_mpo(
            prepared,
            QuantumLatticeMPOPolicy(
                maximum_bond_dimension=2 * mode_count,
                maximum_tensor_elements=4_000_000,
            ),
        )
    )
    vmc, vmc_seconds = measure_host(lambda: lower_quantum_lattice_to_vmc(operator))
    configuration = basis.coordinate(0)
    _, vmc_connection_seconds = measure_synchronized(
        lambda: vmc.connections(configuration)
    )
    probe = jnp.arange(1, basis.dimension + 1, dtype=jnp.float64).astype(jnp.complex128)
    probe = probe / jnp.sqrt(jnp.vdot(probe, probe).real)
    hermiticity_residual = jnp.abs(
        jnp.vdot(probe, operator.mv(vector)) - jnp.vdot(operator.mv(probe), vector)
    )
    return {
        "axes": {
            "modes": mode_count,
            "particles": mode_count // 2,
            "sector_dimension": basis.dimension,
            "compiled_monomials": len(prepared.monomials),
        },
        "plan_id": prepared.plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "model_seconds": model_seconds,
        "prepare_seconds": prepare_seconds,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_seconds": warm_seconds,
        "steady": steady.to_seconds_dict(),
        "target_lowering_seconds": {
            "local_hamiltonian": local_seconds,
            "mpo": mpo_seconds,
            "vmc": vmc_seconds,
            "vmc_connections": vmc_connection_seconds,
        },
        "logical_operator_bytes": logical_array_bytes(operator),
        "logical_target_bytes": {
            "local_hamiltonian": logical_array_bytes(local),
            "mpo": logical_array_bytes(mpo),
            "vmc": logical_array_bytes(vmc),
        },
        "matrix_free_action_workspace_bytes": operator.action_workspace_bytes,
        "compiler": asdict(compiler),
        "operator_matvec_count": repeats + 4,
        "scientific_residual": float(hermiticity_residual),
        "output_norm": float(jnp.sqrt(jnp.vdot(steady_value, steady_value).real)),
        "successful": bool(
            jnp.all(jnp.isfinite(warm))
            and jnp.all(jnp.isfinite(steady_value))
            and jnp.isfinite(hermiticity_residual)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", type=int, default=(6, 10))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 2 or value % 2 for value in arguments.modes):
        raise ValueError("Benchmark mode counts must be positive even integers.")
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(mode_count, arguments.repeats)
            for mode_count in arguments.modes
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
