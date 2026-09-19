#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scale native Pfaffian and prepared local-determinant execution."""

from __future__ import annotations

import json
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


PFAFFIAN_CASES = ((8, 1), (16, 32), (32, 8), (64, 1))
LOW_RANK_CASES = (
    (16, 4),
    (32, 4),
    (64, 2),
    (64, 4),
    (64, 8),
    (64, 16),
    (128, 4),
    (256, 4),
    (512, 4),
)
WARMUP = 1
REPEATS = 10


@jax.jit
def _evaluate_pfaffian(matrix):
    return phx.linalg.evaluate_pfaffian(matrix)


@jax.jit
def _propose_row_update(sequence, update):
    proposal = phx.linalg.propose_low_rank_update(sequence, update)
    return (
        proposal.log_abs,
        proposal.status,
        proposal.compact_condition,
        proposal.aggregate_condition,
        proposal.candidate_prepared,
    )


@jax.jit
def _skew_determinant_log_abs(matrix):
    return 0.5 * jnp.linalg.slogdet(matrix)[1]


@jax.jit
def _full_row_update(matrix, index, row_delta, base_log_abs):
    # Intentional dense reference route for the benchmark comparison.
    updated = matrix.at[index, :].add(row_delta)
    return jnp.linalg.slogdet(updated)[1] - base_log_abs


@eqx.filter_jit
def _markov_step(target, kernel, state, key):
    return kernel.step(target, state, key)


def _plain_executable(compiled):
    return compiled


def _filtered_executable(compiled):
    return compiled.compiled


def _compiler_record(compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _measure(
    compiled_function,
    arguments,
    summarize,
    *,
    executable_view=_plain_executable,
):
    jax.clear_caches()
    compiled, compilation = measure_lower_and_compile(
        lambda: compiled_function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    result, execution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=WARMUP,
        repeats=REPEATS,
    )
    executable = executable_view(compiled)
    return {
        "compilation": asdict(compilation),
        "first_execution_seconds": first_seconds,
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(executable),
        "output_logical_bytes": logical_array_bytes(first),
        "output": summarize(result),
    }


def _skew_batch(dimension, batch_size, seed):
    generator = np.random.default_rng(seed)
    raw = generator.normal(size=(batch_size, dimension, dimension))
    skew = raw - np.swapaxes(raw, -1, -2)
    regularizer = np.kron(
        np.eye(dimension // 2),
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
    )
    return jnp.asarray(skew + 0.25 * regularizer)


def _pfaffian_summary(result):
    return {
        "sign": np.asarray(result.sign).tolist(),
        "log_abs": np.asarray(result.log_abs).tolist(),
        "status": np.asarray(result.status).tolist(),
        "successful": np.asarray(result.successful).tolist(),
        "singular": np.asarray(result.singular).tolist(),
        "value_finite": np.asarray(result.value_finite).tolist(),
    }


def _array_summary(result):
    return {"value": np.asarray(result).tolist()}


def _pfaffian_record(dimension, batch_size):
    matrix = _skew_batch(dimension, batch_size, 1000 + dimension + batch_size)
    argument = matrix[0] if batch_size == 1 else matrix
    pfaffian = _measure(_evaluate_pfaffian, (argument,), _pfaffian_summary)
    determinant = _measure(
        _skew_determinant_log_abs,
        (argument,),
        _array_summary,
    )
    pfaffian_value = np.asarray(pfaffian["output"]["log_abs"])
    determinant_value = np.asarray(determinant["output"]["value"])
    return {
        "operation": "portable-pfaffian",
        "dimension": dimension,
        "batch_size": batch_size,
        "input_logical_bytes": logical_array_bytes(argument),
        "absolute_log_magnitude_error": float(
            np.max(np.abs(pfaffian_value - determinant_value))
        ),
        "pfaffian": pfaffian,
        "determinant_magnitude_reference": determinant,
    }


def _sequence(dimension, capacity, seed):
    generator = np.random.default_rng(seed)
    raw = generator.normal(size=(dimension, dimension))
    matrix = jnp.asarray(raw @ raw.T + dimension * np.eye(dimension))
    policy = phx.linalg.LowRankSolvePolicy(
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
        base_nonsingularity="asserted",
        failure=phx.linalg.FailurePolicy("status"),
    )
    sequence = phx.linalg.prepare_low_rank_sequence(
        phx.linalg.DenseLinearOperator(matrix),
        capacity,
        policy,
    )
    row_delta = jnp.asarray(
        generator.normal(scale=1.0e-2, size=(1, dimension)),
        dtype=matrix.dtype,
    )
    index = jnp.asarray(dimension // 2, dtype=jnp.int32)
    update = phx.linalg.row_low_rank_update(index[None], row_delta)
    base_log_abs = phx.linalg.factorize(
        phx.linalg.DenseLinearOperator(matrix),
        phx.linalg.FactorizationPolicy("lu"),
    ).log_abs_determinant()
    return matrix, index, row_delta[0], base_log_abs, sequence, update


def _proposal_summary(result):
    log_abs, status, compact_condition, aggregate_condition, candidate = result
    successful = status == int(phx.linalg.LowRankDeterminantStatus.SUCCESS)
    return {
        "value": np.asarray(log_abs).tolist(),
        "status": int(np.asarray(status)),
        "successful": bool(np.asarray(successful)),
        "requires_rebase": not bool(np.asarray(successful)),
        "compact_condition": float(np.asarray(compact_condition)),
        "aggregate_condition": float(np.asarray(aggregate_condition)),
        "route": "row-indexed",
        "candidate_logical_bytes": logical_array_bytes(candidate),
    }


def _low_rank_record(dimension, capacity):
    matrix, index, row_delta, base_log_abs, sequence, update = _sequence(
        dimension,
        capacity,
        2000 + dimension + capacity,
    )
    proposed = _measure(_propose_row_update, (sequence, update), _proposal_summary)
    full = _measure(
        _full_row_update,
        (matrix, index, row_delta, base_log_abs),
        _array_summary,
    )
    proposed_value = float(proposed["output"]["value"])
    full_value = float(full["output"]["value"])
    return {
        "operation": "row-determinant-update",
        "dimension": dimension,
        "capacity": capacity,
        "sequence_logical_bytes": logical_array_bytes(sequence),
        "update_logical_bytes": logical_array_bytes(update),
        "absolute_log_ratio_error": abs(proposed_value - full_value),
        "prepared_sequence": proposed,
        "full_refactorization": full,
    }


def _local_pairing_matrix(coordinates):
    particle_count = coordinates.shape[0]
    even = jnp.arange(0, particle_count, 2)
    odd = even + 1
    base = jnp.zeros((particle_count, particle_count), dtype=coordinates.dtype)
    base = base.at[even, odd].set(1.0)
    base = base.at[odd, even].set(-1.0)
    coordinate = coordinates[:, 0]
    return base + 0.01 * (coordinate[:, None] - coordinate[None, :])


def _zero_jastrow(coordinates):
    return jnp.zeros((), dtype=coordinates.dtype)


def _model_log_target(model, position):
    return 2.0 * model(position).log_abs


def _incremental_step_summary(result):
    state, info = result
    return {
        "accepted": np.asarray(info.accepted).tolist(),
        "target_valid": np.asarray(info.target_valid).tolist(),
        "state_valid": np.asarray(state.valid).tolist(),
        "route": np.asarray(state.cache.route).tolist(),
        "low_rank_status": np.asarray(state.cache.low_rank_status).tolist(),
    }


def _full_step_summary(result):
    state, info = result
    return {
        "accepted": np.asarray(info.accepted).tolist(),
        "target_valid": np.asarray(info.target_valid).tolist(),
        "state_valid": np.asarray(state.valid).tolist(),
    }


def _target_step_record(particle_count=16, chain_count=4, capacity=4):
    model = phx.nn.quantum.PfaffianJastrowAmplitude(
        _local_pairing_matrix,
        _zero_jastrow,
        particle_count=particle_count,
        spatial_dimension=1,
        pairing_id="benchmark-local-pairing",
        cusp_id="benchmark-zero-jastrow",
        policy=phx.linalg.PfaffianPolicy(),
    )
    update_policy = phx.linalg.LowRankSolvePolicy(
        phx.linalg.LinearSolvePolicy(
            phx.linalg.DenseLU(),
            failure=phx.linalg.FailurePolicy("status"),
        ),
        base_nonsingularity="asserted",
        failure=phx.linalg.FailurePolicy("status"),
    )
    incremental_target = phx.nn.quantum.pfaffian_jastrow_incremental_target(
        model,
        capacity=capacity,
        update_policy=update_policy,
        maximum_chains=chain_count,
    )
    full_target = phx.sampling.FullMarkovTarget(
        eqx.Partial(_model_log_target, model),
        target_id="benchmark-full-pfaffian",
    )
    kernel = phx.sampling.MetropolisHastings(
        phx.sampling.SingleCoordinateGaussianProposal(0.02)
    )
    base = jnp.linspace(-1.0, 1.0, particle_count)[:, None]
    positions = jnp.stack(tuple(base + 0.03 * chain for chain in range(chain_count)))
    incremental_state = kernel.initialize(incremental_target, positions)
    full_state = kernel.initialize(full_target, positions)
    key = jr.key(991)
    incremental = _measure(
        _markov_step,
        (incremental_target, kernel, incremental_state, key),
        _incremental_step_summary,
        executable_view=_filtered_executable,
    )
    full = _measure(
        _markov_step,
        (full_target, kernel, full_state, key),
        _full_step_summary,
        executable_view=_filtered_executable,
    )
    return {
        "operation": "pfaffian-markov-step",
        "particle_count": particle_count,
        "chain_count": chain_count,
        "capacity": capacity,
        "incremental_state_logical_bytes": logical_array_bytes(incremental_state),
        "full_state_logical_bytes": logical_array_bytes(full_state),
        "incremental": incremental,
        "full": full,
        "accepted_equal": (
            incremental["output"]["accepted"] == full["output"]["accepted"]
        ),
    }


def main():
    records = [
        *(_pfaffian_record(dimension, batch) for dimension, batch in PFAFFIAN_CASES),
        *(
            _low_rank_record(dimension, capacity)
            for dimension, capacity in LOW_RANK_CASES
        ),
        _target_step_record(16, 4, 4),
        _target_step_record(128, 4, 4),
    ]
    payload = {
        "campaign": "native-matrix-invariant-scaling",
        "configuration": {"warmup": WARMUP, "repeats": REPEATS},
        "environment": capture_environment().to_dict(),
        "records": records,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
