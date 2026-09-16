#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _timed(callable_, repeats):
    started = time.perf_counter()
    value = None
    for _ in range(repeats):
        value = callable_()
        jax.block_until_ready(value.reduced_state)
    return value, (time.perf_counter() - started) / repeats


def run(dimension: int, rank: int, repeats: int) -> dict[str, object]:
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(dimension, rank))
    basis_np, _ = np.linalg.qr(raw, mode="reduced")
    spectrum = np.linspace(1.0, 4.0, dimension)
    operator_np = np.diag(spectrum)
    reduced_truth = rng.normal(size=(rank,))
    truth = basis_np @ reduced_truth
    rhs_np = operator_np @ truth

    dtype = jnp.float64
    full = phx.linalg.ArraySpace((dimension,), dtype=dtype, space_id="rom-benchmark-full")
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            full,
            jnp.asarray(basis_np, dtype=dtype),
            orthonormal=True,
            subspace_id="rom-benchmark-subspace",
        ),
        role="state",
        state_contract_id="rom-benchmark-state",
        support_id="rom-benchmark-support",
        measure_id="rom-benchmark-measure",
        geometry_id="rom-benchmark-geometry",
        source_artifact_ids=("rom-benchmark-snapshots",),
    )
    reduction = phx.rom.trial_test_reduction_from_bases(basis)
    operator = phx.linalg.DenseLinearOperator(
        jnp.asarray(operator_np, dtype=dtype),
        source=full,
        target=phx.linalg.DualSpace(full),
        operator_id="rom-benchmark-operator",
    )
    problem = phx.rom.AffineLinearROMProblem(
        reduction,
        (operator,),
        (jnp.asarray(rhs_np, dtype=dtype),),
        operator_term_ids=("A",),
        right_hand_side_term_ids=("b",),
        source_artifact_ids=("rom-benchmark-family",),
    )
    coefficients = phx.rom.ArrayAffineCoefficientMap(
        jnp.zeros((1, 1), dtype=dtype),
        jnp.ones((1,), dtype=dtype),
        jnp.zeros((1, 1), dtype=dtype),
        jnp.ones((1,), dtype=dtype),
        operator_term_ids=("A",),
        right_hand_side_term_ids=("b",),
        lower=jnp.zeros((1,), dtype=dtype),
        upper=jnp.ones((1,), dtype=dtype),
        input_contract_id="rom-benchmark-input",
        unit_contract_id="dimensionless",
        support_id="rom-benchmark-support",
    )
    started = time.perf_counter()
    model = phx.rom.prepare_affine_linear_rom(problem, coefficients)
    preparation_seconds = time.perf_counter() - started
    inputs = jnp.asarray([0.5], dtype=dtype)
    reduced_evaluator = eqx.filter_jit(
        lambda value: model.evaluate_admitted(value, reconstruct=False)
    )
    reconstruction_evaluator = eqx.filter_jit(
        lambda value: model.evaluate_admitted(value, reconstruct=True)
    )
    jax.block_until_ready(reduced_evaluator(inputs).reduced_state)
    jax.block_until_ready(reconstruction_evaluator(inputs).reduced_state)
    reduced, reduced_seconds = _timed(lambda: reduced_evaluator(inputs), repeats)
    reconstructed, reconstruction_seconds = _timed(
        lambda: reconstruction_evaluator(inputs), repeats
    )
    started = time.perf_counter()
    for _ in range(repeats):
        direct = np.linalg.solve(operator_np, rhs_np)
    direct_seconds = (time.perf_counter() - started) / repeats
    error = float(np.linalg.norm(np.asarray(reconstructed.reconstructed_state) - direct))
    return {
        "dimension": dimension,
        "rank": rank,
        "affine_terms": 1,
        "repeats": repeats,
        "preparation_seconds": preparation_seconds,
        "reduced_solve_seconds": reduced_seconds,
        "solve_and_reconstruction_seconds": reconstruction_seconds,
        "full_solve_seconds": direct_seconds,
        "state_error": error,
        "reduced_valid": bool(reduced.valid),
        "model_id": model.model_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if (
        arguments.dimension <= 0
        or arguments.rank <= 0
        or arguments.rank > arguments.dimension
    ):
        raise ValueError("Require 0 < rank <= dimension.")
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive.")
    print(
        json.dumps(
            run(arguments.dimension, arguments.rank, arguments.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
