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
from phydrax.operators.quantum.lattice import (
    CharacterSectorPlan,
    FiniteGroupActionPlan,
    FixedSpinProjectionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    MonomialConfigurationGenerator,
    OrbitOperatorResourcePolicy,
    OrbitSectorResourcePolicy,
    prepare_finite_group_action,
    prepare_orbit_sector_basis,
    prepare_quantum_lattice,
    prepare_quantum_orbit_sector_operator,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    SectorBasisResourcePolicy,
)


def _ring(site_count: int):
    spaces = tuple(LocalSpacePlan.spin(f"s{index}", 1) for index in range(site_count))
    raising = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    lowering = raising.T
    terms = tuple(
        QuantumLatticeTerm(
            (
                LocalOperatorPlan(spaces[index], f"raise-{index}", raising, (2,)),
                LocalOperatorPlan(
                    spaces[(index + 1) % site_count],
                    f"lower-{(index + 1) % site_count}",
                    lowering,
                    (-2,),
                ),
            ),
            coefficient=1.0,
            add_adjoint=True,
            label=f"exchange-{index}",
        )
        for index in range(site_count)
    )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, terms),
        QuantumLatticeResourcePolicy(
            maximum_terms=4 * site_count,
            maximum_factors_per_term=2,
            maximum_branches_per_input=16 * site_count,
            maximum_sector_dimension=1_000_000,
            maximum_workspace_bytes=256 * 1024**2,
        ),
    )
    basis = FixedSpinProjectionBasis(
        tuple(space.site_id for space in spaces),
        (1,) * site_count,
        2 - site_count,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=1_000_000,
            maximum_table_bytes=16 * 1024**2,
        ),
    )
    translation = MonomialConfigurationGenerator(
        "translation",
        basis.site_ids,
        basis.site_dimensions,
        tuple((index + 1) % site_count for index in range(site_count)),
        order=site_count,
    )
    return prepared, basis, translation


def benchmark_case(site_count: int, repeats: int):
    (prepared, direct, translation), model_seconds = measure_host(
        lambda: _ring(site_count)
    )
    action, action_seconds = measure_host(
        lambda: prepare_finite_group_action(
            FiniteGroupActionPlan(
                direct,
                (translation,),
                OrbitSectorResourcePolicy(
                    maximum_group_order=site_count,
                    maximum_orbit_dimension=direct.dimension,
                    maximum_table_bytes=128 * 1024**2,
                ),
            ),
            CharacterSectorPlan("zero-momentum", {"translation": 1.0}),
        )
    )
    basis, basis_seconds = measure_host(lambda: prepare_orbit_sector_basis(action))
    operator, operator_seconds = measure_host(
        lambda: prepare_quantum_orbit_sector_operator(
            prepared,
            basis,
            OrbitOperatorResourcePolicy(
                maximum_routes=max(1, basis.dimension**2),
                maximum_workspace_bytes=256 * 1024**2,
            ),
        )
    )
    vector = jnp.ones((basis.dimension,), dtype=jnp.complex128) / jnp.sqrt(
        basis.dimension
    )
    apply = eqx.filter_jit(lambda op, value: op.mv(value))
    executable, compilation = measure_lower_and_compile(
        lambda: apply.lower(operator, vector), lambda lowered: lowered.compile()
    )
    warm, warm_seconds = measure_synchronized(lambda: executable(operator, vector))
    result, steady = measure_repeated(
        lambda: executable(operator, vector), warmup=0, repeats=repeats
    )
    compiler = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiler-analysis",
    )
    hermiticity = jnp.abs(
        jnp.vdot(vector, operator.mv(result)) - jnp.vdot(operator.mv(vector), result)
    )
    return {
        "axes": {
            "sites": site_count,
            "direct_dimension": direct.dimension,
            "orbit_dimension": basis.dimension,
            "group_order": action.group_order,
            "route_count": operator.evidence.route_count,
        },
        "ids": {
            "prepared": prepared.prepared_id,
            "action": action.prepared_id,
            "basis": basis.basis_id,
            "operator": operator.operator_id,
        },
        "host_seconds": {
            "model": model_seconds,
            "group_closure": action_seconds,
            "orbit_basis": basis_seconds,
            "operator_routes": operator_seconds,
        },
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_seconds": warm_seconds,
        "steady": steady.to_seconds_dict(),
        "logical_bytes": {
            "action": logical_array_bytes(action),
            "basis": logical_array_bytes(basis),
            "operator": logical_array_bytes(operator),
        },
        "action_workspace_bytes": operator.action_workspace_bytes,
        "compiler": asdict(compiler),
        "scientific_residuals": {
            "invariance": float(operator.evidence.maximum_invariance_residual),
            "hermiticity": float(operator.evidence.hermiticity_residual),
            "probe_hermiticity": float(hermiticity),
        },
        "output_norm": float(jnp.linalg.norm(result)),
        "successful": bool(
            jnp.all(jnp.isfinite(warm))
            and jnp.all(jnp.isfinite(result))
            and bool(operator.evidence.accepted)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, default=(4, 6))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 2 for value in arguments.sites):
        raise ValueError("Benchmark site counts must be at least two.")
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(site_count, arguments.repeats)
            for site_count in arguments.sites
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
