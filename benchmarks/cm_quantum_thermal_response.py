#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from math import isfinite, pi

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_synchronized,
)
from phydrax.linalg import (
    DifferentiationPolicy,
    MatrixFunctionPolicy,
    ShiftedSolvePolicy,
    ShiftedSolveResourcePolicy,
)
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum._response import (
    FiniteTemperatureResponsePlan,
    QuantumSectorProbe,
    ZeroTemperatureResponsePlan,
)
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from phydrax.solver._quantum_response import (
    finite_temperature_response,
    zero_temperature_response,
)
from phydrax.solver._thermal_pure_quantum import (
    prepare_thermal_pure_quantum,
    thermal_pure_quantum,
    ThermalPureQuantumPlan,
)


def _case(mode_count: int, krylov_dimension: int):
    order = FermionModeOrder(tuple(f"m{index}" for index in range(mode_count)))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    create_ops = tuple(
        LocalOperatorPlan(space, "create", create, (1,)) for space in spaces
    )
    number_terms = tuple(
        QuantumLatticeTerm(
            (
                create_ops[index],
                LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
            ),
            coefficient=float(index + 1),
            label=f"energy:{index}",
        )
        for index, space in enumerate(spaces)
    )
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=mode_count + 2,
        maximum_factors_per_term=2,
        maximum_branches_per_input=8 * mode_count,
        maximum_sector_dimension=1_000_000,
        maximum_workspace_bytes=256 * 1024**2,
    )
    hamiltonian = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, number_terms, fermion_mode_order=order),
        resources,
    )
    sector_resources = SectorBasisResourcePolicy(
        maximum_dimension=1_000_000, maximum_table_bytes=16 * 1024**2
    )
    source_basis = FixedCardinalityFermionBasis(order, 1, resources=sector_resources)
    target_basis = FixedCardinalityFermionBasis(order, 2, resources=sector_resources)
    source = QuantumSectorOperator(
        hamiltonian, SectorChargeMap(source_basis, source_basis, 0)
    )
    target = QuantumSectorOperator(
        hamiltonian, SectorChargeMap(target_basis, target_basis, 0)
    )
    probe_prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(
            spaces,
            (QuantumLatticeTerm((create_ops[1],), label="create:m1"),),
            fermion_mode_order=order,
        ),
        resources,
    )
    probe = QuantumSectorProbe(
        QuantumSectorOperator(
            probe_prepared, SectorChargeMap(source_basis, target_basis, 1)
        ),
        probe_id="create:m1",
    )
    matrix_policy = MatrixFunctionPolicy(
        "lanczos",
        max_dimension=krylov_dimension,
        error_tolerance=1e-9,
        differentiation=DifferentiationPolicy("none"),
    )
    return source, target, probe, matrix_policy


def benchmark_case(mode_count: int, probe_count: int, krylov_dimension: int, beta: float):
    (source, target, probe, matrix_policy), prepare_seconds = measure_host(
        lambda: _case(mode_count, krylov_dimension)
    )
    source_tpq_plan = ThermalPureQuantumPlan(
        beta,
        probe_count=probe_count,
        observable_count=0,
        matrix_function=matrix_policy,
        maximum_retained_bytes=256 * 1024**2,
        maximum_workspace_bytes=256 * 1024**2,
    )
    target_tpq_plan = ThermalPureQuantumPlan(
        beta,
        probe_count=probe_count,
        observable_count=0,
        matrix_function=matrix_policy,
        maximum_retained_bytes=256 * 1024**2,
        maximum_workspace_bytes=256 * 1024**2,
    )
    prepared_source, source_prepare_seconds = measure_host(
        lambda: prepare_thermal_pure_quantum(source_tpq_plan, source)
    )
    prepared_target, target_prepare_seconds = measure_host(
        lambda: prepare_thermal_pure_quantum(target_tpq_plan, target)
    )
    source_tpq, source_seconds = measure_synchronized(
        lambda: thermal_pure_quantum(prepared_source, (), key=jr.key(31))
    )
    target_tpq, target_seconds = measure_synchronized(
        lambda: thermal_pure_quantum(prepared_target, (), key=jr.key(47))
    )
    zero_plan = ZeroTemperatureResponsePlan(
        jnp.linspace(-1.0, float(mode_count + 2), 33),
        0.1,
        moment_count=4,
        shifted_solve=ShiftedSolvePolicy(
            "lanczos",
            max_dimension=krylov_dimension,
            relative_tolerance=1e-9,
            differentiation="none",
            resources=ShiftedSolveResourcePolicy(
                max_matvec_count=10_000,
                max_storage_bytes=64 * 1024**2,
                max_workspace_bytes=64 * 1024**2,
            ),
        ),
        maximum_frequency_points=64,
        maximum_result_bytes=8 * 1024**2,
    )
    ground = jnp.zeros((source.source.size,), dtype=jnp.complex128).at[-1].set(1.0)
    zero, zero_seconds = measure_synchronized(
        lambda: zero_temperature_response(zero_plan, source, target, probe, ground, 1.0)
    )
    times = jnp.linspace(-pi, pi, 17)
    finite_plan = FiniteTemperatureResponsePlan(
        times,
        jnp.linspace(-4.0, 4.0, 17),
        jnp.sin(0.5 * (times + pi)) ** 2,
        moment_count=4,
        matrix_function=matrix_policy,
        maximum_result_bytes=64 * 1024**2,
        maximum_workspace_bytes=256 * 1024**2,
        positivity_tolerance=1e-5,
        kms_tolerance=1e-2,
    )
    finite, finite_seconds = measure_synchronized(
        lambda: finite_temperature_response(
            finite_plan, source, target, probe, source_tpq, target_tpq
        )
    )
    return {
        "axes": {
            "modes": mode_count,
            "source_sector_dimension": source.source.size,
            "target_sector_dimension": target.source.size,
            "probes": probe_count,
            "beta": beta,
            "times": finite_plan.times.size,
            "frequencies": finite_plan.frequencies.size,
            "krylov_dimension": krylov_dimension,
        },
        "prepare_seconds": prepare_seconds,
        "tpq_prepare_seconds": {
            "source": source_prepare_seconds,
            "target": target_prepare_seconds,
        },
        "tpq_seconds": {"source": source_seconds, "target": target_seconds},
        "response_seconds": {
            "zero_temperature": zero_seconds,
            "finite_temperature": finite_seconds,
        },
        "logical_bytes": {
            "source_tpq": logical_array_bytes(source_tpq),
            "target_tpq": logical_array_bytes(target_tpq),
            "zero_response": logical_array_bytes(zero),
            "finite_response": logical_array_bytes(finite),
        },
        "raw_numerical_error": {
            "source_tpq_maximum": float(jnp.max(source_tpq.numerical_error_estimates)),
            "target_tpq_maximum": float(jnp.max(target_tpq.numerical_error_estimates)),
            "finite_maximum": float(finite.evidence.maximum_numerical_error),
        },
        "raw_statistical_error": {
            "source_partition": float(source_tpq.partition_standard_error),
            "target_partition": float(target_tpq.partition_standard_error),
            "finite_spectrum_maximum": float(
                jnp.max(finite.forward_spectrum_standard_error)
            ),
        },
        "scientific_residual": {
            "zero_positivity_violation": float(zero.evidence.positivity_violation),
            "finite_positivity_violation": float(
                finite.evidence.forward_positivity_violation
            ),
            "kms": float(finite.evidence.kms_residual),
        },
        "successful": bool(
            source_tpq.valid
            & target_tpq.valid
            & zero.evidence.valid
            & finite.evidence.valid
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--krylov-dimension", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if (
        arguments.modes < 2
        or arguments.probes < 2
        or arguments.krylov_dimension < 1
        or not isfinite(arguments.beta)
        or arguments.beta < 0.0
    ):
        raise ValueError("Thermal-response benchmark controls are invalid.")
    payload = {
        "environment": capture_environment().to_dict(),
        "case": benchmark_case(
            arguments.modes,
            arguments.probes,
            arguments.krylov_dimension,
            arguments.beta,
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
