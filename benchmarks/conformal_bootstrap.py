#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.conformal_bootstrap import (
    audit_pmp_samples,
    ConformalDataPlan,
    ConformalPolynomialMatrixProgram,
    CrossingChannel,
    DampedRationalPrefactor,
    ExchangedOperatorSector,
    ExternalScalarOperator,
    GlobalScalarBlockPlan,
    ising_sigma_crossing_evidence,
    PolynomialMatrixBlock,
    prepare_global_scalar_blocks,
)


def _data(dimension: float):
    external = tuple(
        ExternalScalarOperator(f"phi-{index}", 0.6, "scalar") for index in range(4)
    )
    return ConformalDataPlan(
        dimension,
        external,
        (
            ExchangedOperatorSector(
                "even",
                "scalar",
                (0, 2, 4),
                parity="even",
                minimum_dimension=max(0.1, (dimension - 2.0) / 2.0),
            ),
        ),
        (CrossingChannel("s-t", (2, 1, 0, 3), involutive=True),),
        category=None,
    )


def _pmp():
    block = PolynomialMatrixBlock(
        DampedRationalPrefactor("0.3678794411714423215955", "1"),
        (((("1", "0", "1"), ("0", "1")),),),
        sample_points=("0", "0.5", "1"),
        sample_scalings=("1", "1", "1"),
        reduced_sample_scalings=("1", "1", "1"),
    )
    return ConformalPolynomialMatrixProgram(
        ("0", "-1"),
        ("1", "0"),
        (block,),
        frontend_id="benchmark-exact-decimal-control",
        frontend_precision_bits=256,
    )


def benchmark_case(dimension: float, recursion_order: int, repeats: int):
    points = np.asarray(((0.2, 0.3), (0.3, 0.2), (0.25, 0.25)))
    plan, plan_seconds = measure_host(
        lambda: GlobalScalarBlockPlan(
            _data(dimension),
            points,
            (0, 2, 4),
            ((0, 0), (1, 0), (0, 1)),
            recursion_order=recursion_order,
            hypergeometric_order=192,
            maximum_evaluations=20_000_000,
        )
    )
    prepared, prepare_seconds = measure_host(lambda: prepare_global_scalar_blocks(plan))
    delta = jnp.asarray(max(2.5, dimension), dtype=jnp.float64)
    execute = eqx.filter_jit(lambda blocks, value: blocks.derivative_table(value, 0))
    executable, compilation = measure_lower_and_compile(
        lambda: execute.lower(prepared, delta), lambda lowered: lowered.compile()
    )
    _warm, warm_seconds = measure_synchronized(lambda: executable(prepared, delta))
    values, steady = measure_repeated(
        lambda: executable(prepared, delta), warmup=0, repeats=repeats
    )
    evidence, evidence_seconds = measure_synchronized(lambda: prepared.evidence(delta, 0))
    program, pmp_seconds = measure_host(_pmp)
    audit, audit_seconds = measure_host(
        lambda: audit_pmp_samples(program, (1.0, 0.0), (0.0, 0.5, 1.0))
    )
    virasoro, virasoro_seconds = measure_host(
        lambda: ising_sigma_crossing_evidence((0.2, 0.35, 0.65, 0.8))
    )
    return {
        "axes": {
            "spacetime_dimension": dimension,
            "recursion_order": recursion_order,
            "point_count": int(points.shape[0]),
            "derivative_count": len(plan.derivative_orders),
        },
        "ids": {
            "plan": plan.plan_id,
            "prepared": prepared.prepared_id,
            "pmp": program.pmp_id,
            "virasoro_crossing": virasoro.crossing_id,
        },
        "method": prepared.method,
        "host_seconds": {
            "plan": plan_seconds,
            "prepare": prepare_seconds,
            "pmp": pmp_seconds,
            "pmp_audit": audit_seconds,
            "virasoro_crossing": virasoro_seconds,
        },
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_seconds": warm_seconds,
        "steady": steady.to_seconds_dict(),
        "evidence_seconds": evidence_seconds,
        "logical_bytes": {
            "prepared_blocks": logical_array_bytes(prepared),
            "derivatives": logical_array_bytes(values),
        },
        "scientific_residuals": {
            "maximum_truncation_proxy": float(jnp.max(evidence.truncation_proxy)),
            "maximum_casimir_residual": float(jnp.max(evidence.casimir_residuals)),
            "pmp_normalization_residual": float(audit.normalization_residual),
            "pmp_minimum_eigenvalue": float(jnp.min(audit.minimum_eigenvalues)),
            "virasoro_crossing": float(virasoro.maximum_residual),
        },
        "successful": bool(evidence.finite and audit.accepted and virasoro.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", nargs="+", type=float, default=(2.0, 3.0))
    parser.add_argument("--recursion-order", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if arguments.recursion_order < 0 or arguments.repeats < 1:
        raise ValueError("Recursion order and repeats are invalid.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(dimension, arguments.recursion_order, arguments.repeats)
            for dimension in arguments.dimensions
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
