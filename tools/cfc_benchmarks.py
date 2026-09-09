#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Key

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._trainable import combine_trainable, partition_trainable


Architecture = Literal["cfc", "gru-dt", "lstm-dt", "selective"]
_ARCHITECTURES: tuple[Architecture, ...] = (
    "cfc",
    "gru-dt",
    "lstm-dt",
    "selective",
)


@dataclass(frozen=True, slots=True)
class EventDataset:
    batch: phx.nn.layers.RecurrentBatch
    intervals: Array
    targets: Array


def _event_dataset(
    key: Key[Array, ""],
    /,
    *,
    cases: int,
    length: int,
    minimum_interval: float,
    maximum_interval: float,
) -> EventDataset:
    input_key, interval_key, length_key = jr.split(key, 3)
    inputs = jr.normal(input_key, (cases, length, 2), dtype=jnp.float32)
    tail_intervals = jr.uniform(
        interval_key,
        (cases, length - 1),
        minval=float(minimum_interval),
        maxval=float(maximum_interval),
        dtype=jnp.float32,
    )
    raw_intervals = jnp.concatenate(
        (jnp.zeros((cases, 1), dtype=jnp.float32), tail_intervals),
        axis=-1,
    )
    lengths = jr.randint(
        length_key,
        (cases,),
        minval=max(2, length // 2),
        maxval=length + 1,
    )
    positions = jnp.arange(length)[None, :]
    valid = positions < lengths[:, None]
    reset_index = lengths // 2
    reset = (
        ((jnp.arange(cases) % 2) == 1)[:, None]
        & (positions == reset_index[:, None])
        & valid
    )
    inputs = jnp.where(valid[..., None], inputs, 0.0)
    clock_intervals = jnp.where(valid, raw_intervals, 0.0)
    times = jnp.cumsum(clock_intervals, axis=-1)
    intervals = jnp.where(reset, 0.0, clock_intervals)

    candidate_one_weight = jnp.asarray(
        ((0.8, -0.3, 0.4, 0.1), (-0.2, 0.7, -0.1, 0.5)),
        dtype=jnp.float32,
    )
    candidate_two_weight = jnp.asarray(
        ((-0.4, 0.6, 0.2, -0.3), (0.5, 0.1, 0.3, 0.4)),
        dtype=jnp.float32,
    )
    rate_weight = jnp.asarray(
        ((0.5, -0.2, 0.3, 0.4), (-0.3, 0.4, 0.6, -0.1)),
        dtype=jnp.float32,
    )
    offset_weight = jnp.asarray(
        ((-0.1, 0.3, 0.2, -0.2), (0.4, -0.2, 0.1, 0.3)),
        dtype=jnp.float32,
    )
    candidate_one_bias = jnp.asarray((0.1, -0.2), dtype=jnp.float32)
    candidate_two_bias = jnp.asarray((-0.15, 0.05), dtype=jnp.float32)
    rate_bias = jnp.asarray((0.4, 0.2), dtype=jnp.float32)
    offset_bias = jnp.asarray((-0.3, 0.25), dtype=jnp.float32)

    scan_inputs = jnp.moveaxis(inputs, 1, 0)
    scan_intervals = jnp.moveaxis(intervals, 1, 0)
    scan_valid = jnp.moveaxis(valid, 1, 0)
    scan_reset = jnp.moveaxis(reset, 1, 0)

    def transition(state, event):
        values, interval, is_valid, is_reset = event
        entering = jnp.where(is_reset[:, None], 0.0, state)
        features = jnp.concatenate((values, entering), axis=-1)
        candidate_one = jnp.tanh(
            phx.ein.contract("oi,...i->...o", candidate_one_weight, features)
            + candidate_one_bias
        )
        candidate_two = jnp.tanh(
            phx.ein.contract("oi,...i->...o", candidate_two_weight, features)
            + candidate_two_bias
        )
        rate = phx.ein.contract("oi,...i->...o", rate_weight, features) + rate_bias
        offset = phx.ein.contract("oi,...i->...o", offset_weight, features) + offset_bias
        gate = jax.nn.sigmoid(rate * interval[:, None] + offset)
        proposed = candidate_one * (1.0 - gate) + candidate_two * gate
        next_state = jnp.where(is_valid[:, None], proposed, state)
        output = jnp.where(is_valid[:, None], next_state, 0.0)
        return next_state, output

    _, scan_targets = jax.lax.scan(
        transition,
        jnp.zeros((cases, 2), dtype=jnp.float32),
        (scan_inputs, scan_intervals, scan_valid, scan_reset),
    )
    targets = jnp.moveaxis(scan_targets, 0, 1)
    batch = phx.nn.layers.RecurrentBatch(inputs, valid, reset=reset, time=times)
    return EventDataset(batch, intervals, targets)


def _parameter_count(model: Any, /) -> int:
    trainable, _ = partition_trainable(model)
    return sum(int(leaf.size) for leaf in jax.tree.leaves(trainable))


def _build_model(
    architecture: Architecture,
    width: int,
    /,
    *,
    key: Key[Array, ""],
):
    cell_key, readout_key = jr.split(key)
    if architecture == "cfc":
        cell = phx.nn.layers.CfCCell(
            2,
            width,
            backbone_width=width,
            dtype=jnp.float32,
            key=cell_key,
        )
        readout = eqx.nn.Linear(width, 2, dtype=jnp.float32, key=readout_key)
        return phx.nn.models.RecurrentSequenceModel(cell, readout=readout)
    if architecture == "gru-dt":
        cell = phx.nn.layers.GRUCell(3, width, dtype=jnp.float32, key=cell_key)
        readout = eqx.nn.Linear(width, 2, dtype=jnp.float32, key=readout_key)
        return phx.nn.models.RecurrentSequenceModel(cell, readout=readout)
    if architecture == "lstm-dt":
        cell = phx.nn.layers.LSTMCell(3, width, dtype=jnp.float32, key=cell_key)
        readout = eqx.nn.Linear(width, 2, dtype=jnp.float32, key=readout_key)
        return phx.nn.models.RecurrentSequenceModel(cell, readout=readout)
    if architecture == "selective":
        return phx.nn.models.SelectiveSequenceModel(
            2,
            width,
            inner_size=4,
            depth=1,
            dtype=jnp.float32,
            key=key,
        )
    raise ValueError(f"Unknown architecture {architecture!r}.")


def _matched_widths(*, cfc_width: int) -> tuple[dict[Architecture, int], int]:
    reference = _build_model("cfc", cfc_width, key=jr.key(0))
    target = _parameter_count(reference)
    widths: dict[Architecture, int] = {"cfc": int(cfc_width)}
    for index, architecture in enumerate(_ARCHITECTURES[1:], start=1):
        candidates = tuple(
            (
                abs(
                    _parameter_count(
                        _build_model(
                            architecture,
                            width,
                            key=jr.fold_in(jr.key(index), width),
                        )
                    )
                    - target
                ),
                width,
            )
            for width in range(1, 65)
        )
        widths[architecture] = min(candidates)[1]
    return widths, target


def _model_batch(architecture: Architecture, data: EventDataset, /):
    if architecture not in ("gru-dt", "lstm-dt"):
        return data.batch
    values = jnp.concatenate(
        (data.batch.inputs, data.intervals[..., None]),
        axis=-1,
    )
    return phx.nn.layers.RecurrentBatch(
        values,
        data.batch.valid,
        reset=data.batch.reset,
    )


def _masked_mse(model: Any, batch: Any, targets: Array, valid: Array, /) -> Array:
    prediction = model(batch)
    error = jnp.where(valid[..., None], prediction - targets, 0.0)
    count = jnp.maximum(jnp.sum(valid) * targets.shape[-1], 1)
    return jnp.sum(error**2) / count


def _train_candidate(
    model: Any,
    train: EventDataset,
    validation: EventDataset,
    /,
    *,
    architecture: Architecture,
    learning_rate: float,
    steps: int,
) -> tuple[Any, dict[str, float]]:
    train_batch = _model_batch(architecture, train)
    validation_batch = _model_batch(architecture, validation)
    parameters, fixed = partition_trainable(model)

    def objective(candidate):
        current = combine_trainable(candidate, fixed)
        return _masked_mse(
            current,
            train_batch,
            train.targets,
            train.batch.valid,
        )

    optimizer = optax.adam(float(learning_rate))
    optimizer_state = optimizer.init(parameters)

    @eqx.filter_jit
    def train_step(current_parameters, current_state):
        loss, gradient = eqx.filter_value_and_grad(objective)(current_parameters)
        updates, next_state = optimizer.update(
            gradient,
            current_state,
            current_parameters,
        )
        return eqx.apply_updates(current_parameters, updates), next_state, loss

    initial_loss = float(objective(parameters))
    first_loss = initial_loss
    training_started = time.perf_counter()
    compilation_seconds = 0.0
    for step_index in range(int(steps)):
        step_started = time.perf_counter()
        parameters, optimizer_state, loss = train_step(parameters, optimizer_state)
        jax.block_until_ready(loss)
        if step_index == 0:
            compilation_seconds = time.perf_counter() - step_started
            first_loss = float(loss)
    training_seconds = time.perf_counter() - training_started
    trained = combine_trainable(parameters, fixed)
    final_loss = float(objective(parameters))
    validation_loss = float(
        _masked_mse(
            trained,
            validation_batch,
            validation.targets,
            validation.batch.valid,
        )
    )
    return trained, {
        "learning_rate": float(learning_rate),
        "initial_train_loss": initial_loss,
        "first_train_loss": first_loss,
        "final_train_loss": final_loss,
        "validation_loss": validation_loss,
        "training_seconds": training_seconds,
        "first_step_including_compile_seconds": compilation_seconds,
    }


def _inference_evidence(
    model: Any,
    batch: Any,
    /,
    *,
    repetitions: int,
) -> dict[str, Any]:
    parameters, fixed = partition_trainable(model)

    def predict(candidate):
        return combine_trainable(candidate, fixed)(batch)

    jitted = jax.jit(predict)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(parameters),
        lambda lowered: lowered.compile(),
    )
    _, first_execution = measure_synchronized(lambda: compiled(parameters))
    _, steady = measure_repeated(
        lambda: compiled(parameters),
        warmup=1,
        repeats=int(repetitions),
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first_execution,
        "steady": steady.to_seconds_dict(),
        "compiler": {
            "flops": evidence.flops,
            "bytes_accessed": evidence.bytes_accessed,
            "argument_bytes": evidence.argument_bytes,
            "output_bytes": evidence.output_bytes,
            "temporary_bytes": evidence.temporary_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
            "source": evidence.source,
            "unavailable_reason": evidence.unavailable_reason,
        },
    }


def _run_record(
    architecture: Architecture,
    /,
    *,
    seed: int,
    width: int,
    target_parameters: int,
    steps: int,
    learning_rates: tuple[float, ...],
    cases: int,
    length: int,
    repetitions: int,
) -> dict[str, Any]:
    root = jr.key(seed)
    train_key, validation_key, test_key, ood_key, model_key = jr.split(root, 5)
    train = _event_dataset(
        train_key,
        cases=cases,
        length=length,
        minimum_interval=0.05,
        maximum_interval=0.75,
    )
    validation = _event_dataset(
        validation_key,
        cases=max(8, cases // 2),
        length=length,
        minimum_interval=0.05,
        maximum_interval=0.75,
    )
    test = _event_dataset(
        test_key,
        cases=max(8, cases // 2),
        length=length,
        minimum_interval=0.05,
        maximum_interval=0.75,
    )
    ood = _event_dataset(
        ood_key,
        cases=max(8, cases // 2),
        length=length,
        minimum_interval=0.8,
        maximum_interval=2.0,
    )
    initial_model = _build_model(architecture, width, key=model_key)
    trials = tuple(
        _train_candidate(
            initial_model,
            train,
            validation,
            architecture=architecture,
            learning_rate=learning_rate,
            steps=steps,
        )
        for learning_rate in learning_rates
    )
    selected_model, selected_metrics = min(
        trials,
        key=lambda trial: trial[1]["validation_loss"],
    )
    test_loss = float(
        _masked_mse(
            selected_model,
            _model_batch(architecture, test),
            test.targets,
            test.batch.valid,
        )
    )
    ood_loss = float(
        _masked_mse(
            selected_model,
            _model_batch(architecture, ood),
            ood.targets,
            ood.batch.valid,
        )
    )
    timing = _inference_evidence(
        selected_model,
        _model_batch(architecture, test),
        repetitions=repetitions,
    )
    return {
        "architecture": architecture,
        "seed": int(seed),
        "width": int(width),
        "parameter_count": _parameter_count(initial_model),
        "target_parameter_count": int(target_parameters),
        "training_steps": int(steps),
        "selected_learning_rate": selected_metrics["learning_rate"],
        "learning_rate_trials": [metrics for _, metrics in trials],
        "test_rmse": math.sqrt(test_loss),
        "ood_interval_rmse": math.sqrt(ood_loss),
        "timing": timing,
    }


def _aggregate(records: tuple[dict[str, Any], ...]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for architecture in _ARCHITECTURES:
        selected = tuple(
            record for record in records if record["architecture"] == architecture
        )
        aggregate[architecture] = {
            "median_test_rmse": statistics.median(
                float(record["test_rmse"]) for record in selected
            ),
            "median_ood_interval_rmse": statistics.median(
                float(record["ood_interval_rmse"]) for record in selected
            ),
            "parameter_count": statistics.median(
                int(record["parameter_count"]) for record in selected
            ),
            "median_steady_seconds": statistics.median(
                float(record["timing"]["steady"]["median_seconds"]) for record in selected
            ),
        }
    return aggregate


def _dominates(left: dict[str, float], right: dict[str, float], /) -> bool:
    keys = (
        "median_ood_interval_rmse",
        "parameter_count",
        "median_steady_seconds",
    )
    return all(left[key] <= right[key] for key in keys) and any(
        left[key] < right[key] for key in keys
    )


def run_cfc_benchmarks(
    *,
    quick: bool = False,
    steps: int | None = None,
    repetitions: int | None = None,
) -> dict[str, Any]:
    seeds = (0,) if quick else (0, 1, 2, 3, 4)
    learning_rates = (1e-3,) if quick else (3e-3, 1e-3)
    resolved_steps = (2 if quick else 200) if steps is None else int(steps)
    resolved_repetitions = (
        (2 if quick else 10) if repetitions is None else int(repetitions)
    )
    cases = 8 if quick else 64
    length = 8 if quick else 24
    widths, target_parameters = _matched_widths(cfc_width=8)
    records = tuple(
        _run_record(
            architecture,
            seed=seed,
            width=widths[architecture],
            target_parameters=target_parameters,
            steps=resolved_steps,
            learning_rates=learning_rates,
            cases=cases,
            length=length,
            repetitions=resolved_repetitions,
        )
        for architecture in _ARCHITECTURES
        for seed in seeds
    )
    aggregate = _aggregate(records)
    finite = all(
        math.isfinite(float(record[key]))
        for record in records
        for key in ("test_rmse", "ood_interval_rmse")
    )
    cfc_dominated = any(
        _dominates(aggregate[architecture], aggregate["cfc"])
        for architecture in _ARCHITECTURES
        if architecture != "cfc"
    )
    passed = finite and (quick or not cfc_dominated)
    return {
        "benchmark": "capacity-controlled-cfc-event-relaxation",
        "quick": bool(quick),
        "environment": capture_environment().to_dict(),
        "configuration": {
            "seeds": list(seeds),
            "learning_rates": list(learning_rates),
            "steps": resolved_steps,
            "cases": cases,
            "sequence_length": length,
            "repetitions": resolved_repetitions,
            "training_interval_range": [0.05, 0.75],
            "ood_interval_range": [0.8, 2.0],
        },
        "scope": {
            "event_semantics": True,
            "continuous_flow_claim": False,
            "universal_superiority_claim": False,
            "promotion_rule": "CfC must be finite and nondominated on OOD error, parameters, and steady latency.",
        },
        "widths": dict(widths),
        "target_parameter_count": target_parameters,
        "records": list(records),
        "aggregate": aggregate,
        "cfc_dominated": cfc_dominated,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capacity-controlled CfC irregular-event qualification benchmark."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    arguments = parser.parse_args()
    report = run_cfc_benchmarks(
        quick=arguments.quick,
        steps=arguments.steps,
        repetitions=arguments.repetitions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
