#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Executable compound and cold/prepared Hodge capacity campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.stages import Compiled

import phydrax.linalg as la
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _compiler(compiled: Compiled, /) -> dict[str, int | None]:
    evidence = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
    }


def _timing(
    function: Callable[[Array], Array], values: Array, repeats: int, /
) -> tuple[Array, dict[str, Any]]:
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(values), lambda lowered: lowered.compile()
    )
    result, first = measure_synchronized(lambda: compiled(values))
    result, warm = measure_repeated(lambda: compiled(values), warmup=1, repeats=repeats)
    return result, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_synchronized_seconds": first,
        "warm_execution": warm.to_dict(),
        "compiler": _compiler(compiled),
    }


def _ring(size: int, /) -> la.HilbertComplex:
    if size < 3:
        raise ValueError("Ring capacity must be at least three.")
    w0 = jnp.linspace(1.0, 2.0, size, dtype=jnp.float64)
    w1 = jnp.linspace(2.0, 3.0, size, dtype=jnp.float64)
    spaces = tuple(
        la.ArraySpace(
            (size,),
            dtype=jnp.float64,
            pairing=la.DiagonalPairing(weights),
            space_id=f"campaign:ring:{size}:{degree}",
        )
        for degree, weights in enumerate((w0, w1))
    )
    rows = jnp.repeat(jnp.arange(size, dtype=jnp.int32), 2)
    columns = jnp.stack(
        (
            jnp.arange(size, dtype=jnp.int32),
            (jnp.arange(size, dtype=jnp.int32) + 1) % size,
        ),
        axis=1,
    ).reshape((-1,))
    relation = EdgeRelation(columns, rows, source_size=size, target_size=size)
    differential = SparseCoordinateOperator(
        relation,
        jnp.tile(jnp.asarray([-1.0, 1.0], dtype=jnp.float64), size),
        source=spaces[0],
        target=spaces[1],
        operator_id=f"campaign:ring:d:{size}",
    )
    return la.HilbertComplex(spaces, (differential,), complex_id=f"campaign:ring:{size}")


def _hodge_case(size: int, repeats: int, /) -> dict[str, Any]:
    complex = _ring(size)
    lower, harmonic = tuple(
        la.harmonic_subspace(complex, degree, expected_dimension=1) for degree in (0, 1)
    )
    policy = la.HodgeDecompositionPolicy(
        solve_policy=la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(
                relative=1e-10, absolute=1e-12, max_steps=8 * size
            ),
        )
    )
    phase = jnp.arange(size, dtype=jnp.float64)
    values = jnp.sin(phase) + 1j * jnp.cos(phase)
    prepared, preparation = measure_synchronized(
        lambda: la.prepare_hodge_decomposition(
            complex, 1, harmonic=harmonic, lower_harmonic=lower, policy=policy
        )
    )

    def reuse(values: Array) -> Array:
        result = prepared.apply(values)
        components = tuple(
            jax.tree.leaves(component)[0].reshape((-1,))
            for component in (result.exact, result.coexact, result.harmonic)
        )
        return jnp.concatenate(
            (
                *components,
                jnp.stack(
                    (
                        result.orthogonality_defect,
                        result.reconstruction_defect,
                        result.solve_status.astype(jnp.float64),
                        result.valid.astype(jnp.float64),
                    )
                ),
            )
        )

    def cold(values: Array) -> Array:
        current = la.hodge_decomposition(
            complex, 1, values, harmonic=harmonic, lower_harmonic=lower, policy=policy
        )
        return jax.tree.leaves(current.exact)[0].reshape((-1,))

    result, reused = _timing(reuse, values, repeats)
    _, unprepared = _timing(cold, values, repeats)
    valid = bool(result[-1] == 1)
    reconstruction = jnp.linalg.norm(
        result[:size] + result[size : 2 * size] + result[2 * size : 3 * size] - values
    )
    if not valid or not bool(reconstruction < 1e-8):
        raise RuntimeError("Hodge campaign failed decomposition conservation/evidence.")
    return {
        "capacity": size,
        "preparation_seconds": preparation,
        "prepared": reused,
        "cold": unprepared,
        "retained_bytes": logical_array_bytes(prepared),
        "reconstruction_defect": float(reconstruction),
        "orthogonality_defect": float(jnp.real(result[-4])),
        "solve_status": int(jnp.real(result[-2])),
        "valid": valid,
    }


def _compound_case(size: int, repeats: int, /) -> dict[str, Any]:
    matrix = jax.random.normal(jax.random.key(size), (size, size), dtype=jnp.float64)

    def exterior_power(matrix: Array) -> Array:
        return la.compound_matrix(matrix, 2)

    result, timing = _timing(exterior_power, matrix, repeats)
    left = la.compound_matrix(matrix @ matrix, 2)
    defect = jnp.linalg.norm(left - result @ result)
    if not bool(defect <= 1e-8):
        raise RuntimeError("Compound campaign failed Cauchy-Binet.")
    return {
        "capacity": size,
        "degree": 2,
        "timing": timing,
        "retained_bytes": logical_array_bytes(result),
        "cauchy_binet_defect": float(defect),
    }


def _singular_derivative_case(repeats: int, /) -> dict[str, Any]:
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0, 0.0], dtype=jnp.float64))

    def determinant(matrix: Array) -> Array:
        return la.compound_matrix(matrix, 5)[0, 0]

    result, timing = _timing(jax.grad(determinant), matrix, repeats)
    error = jnp.linalg.norm(
        result - jnp.zeros((5, 5), dtype=jnp.float64).at[4, 4].set(24)
    )
    if not bool(error <= 1e-10):
        raise RuntimeError("Compound campaign failed rank-four adjugate derivative.")
    return {"degree": 5, "timing": timing, "adjugate_derivative_defect": float(error)}


def run_benchmarks(
    *, capacities: Sequence[int] = (8, 32, 128), repeats: int = 3
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    return {
        "environment": capture_environment().to_dict(),
        "compound": [_compound_case(size, repeats) for size in (3, 5, 7)],
        "singular_derivative": _singular_derivative_case(repeats),
        "hodge_decomposition": [_hodge_case(size, repeats) for size in capacities],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmarks(capacities=args.capacities, repeats=args.repeats), indent=2
        )
    )


if __name__ == "__main__":
    main()
