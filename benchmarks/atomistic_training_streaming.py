# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Phase-separated streaming atomistic training cost.

Random hydrogen clusters of varying size are labeled by a fixed teacher PaiNN
and stored as immutable training/validation sources. The phases are measured
separately: source ingest, loader reads, host payload preparation (graph
layout/topology), the first committed update (trace/lower/compile and
execution, from JAX compile monitoring), warm accepted updates, repeated
epochs over the same payload layouts (the compile-count experiment), and one
validation evaluation. Workload evidence records atoms, active directed edges,
padding cases/rows, payload staging bytes, host allocation peaks and memory
high-water marks. Every fit time includes the validation evaluations the
policy performs (step 0 and the final step); the standalone evaluation time is
reported separately so it can be attributed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import host_allocation_peak, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    logical_array_bytes,
    measure_host,
    synchronize,
)


_COMPILE_PREFIX = "/jax/core/compile/"


class _CompileMonitor:
    """Process-wide counts and seconds of every JAX compile-phase event."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: dict[str, list[float]] = {}

    def __call__(self, event: str, duration_secs: float, **_: str | int) -> None:
        if event.startswith(_COMPILE_PREFIX):
            entry = self.events.setdefault(event[len(_COMPILE_PREFIX) :], [0.0, 0.0])
            entry[0] += 1.0
            entry[1] += duration_secs

    def snapshot(self) -> dict[str, tuple[float, float]]:
        return {name: (value[0], value[1]) for name, value in self.events.items()}

    @staticmethod
    def delta(
        before: dict[str, tuple[float, float]], after: dict[str, tuple[float, float]]
    ) -> dict[str, dict[str, float]]:
        return {
            name: {
                "count": int(count - before.get(name, (0.0, 0.0))[0]),
                "seconds": seconds - before.get(name, (0.0, 0.0))[1],
            }
            for name, (count, seconds) in sorted(after.items())
        }


_COMPILES = _CompileMonitor()
jax.monitoring.register_event_duration_secs_listener(_COMPILES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=32)
    parser.add_argument("--validation-structures", type=int, default=8)
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--payload-cases",
        type=int,
        default=4,
        help="Cases per payload; below --batch-size every update accumulates payloads.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Warm accepted updates.")
    parser.add_argument(
        "--repeat-epochs",
        type=int,
        default=2,
        help="Further epochs over the same payload layouts (compile-count experiment).",
    )
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--maximum-neighbors", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="atomistic-training-streaming")
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON record path (default: stdout)."
    )
    return parser


def _validate(arguments: argparse.Namespace, /) -> None:
    positive = (
        "structures",
        "validation_structures",
        "min_atoms",
        "batch_size",
        "payload_cases",
        "steps",
        "repeat_epochs",
        "features",
        "maximum_neighbors",
    )
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if arguments.max_atoms < arguments.min_atoms or arguments.min_atoms < 2:
        raise ValueError("Atom counts need 2 <= --min-atoms <= --max-atoms.")
    if not np.isfinite(arguments.cutoff) or arguments.cutoff <= 0.0:
        raise ValueError("--cutoff must be finite and positive.")


def _scale() -> phx.atomistic.AtomisticScaleContract:
    return phx.atomistic.AtomisticScaleContract(
        phx.units.ANGSTROM, phx.units.ELECTRONVOLT
    )


def _potential(arguments: argparse.Namespace, seed: int, /) -> Any:
    return phx.nn.atomistic.PaiNNPotential(
        _scale(),
        cutoff=arguments.cutoff,
        feature_count=arguments.features,
        interaction_count=2,
        radial_basis_count=6,
        key=jax.random.key(seed),
    )


def _clusters(
    count: int, arguments: argparse.Namespace, generator: np.random.Generator, prefix: str
) -> phx.atomistic.AtomisticBatch:
    """Hydrogen clusters with 0.7 Angstrom minimum separation, padded to max atoms."""
    capacity = arguments.max_atoms
    positions = np.zeros((count, capacity, 3))
    mask = np.zeros((count, capacity), dtype=np.bool_)
    for case in range(count):
        atoms = int(generator.integers(arguments.min_atoms, capacity + 1))
        side = 1.4 * atoms ** (1.0 / 3.0)
        placed: list[np.ndarray] = []
        while len(placed) < atoms:
            candidate = generator.uniform(0.0, side, size=3)
            if all(np.linalg.norm(candidate - point) >= 0.7 for point in placed):
                placed.append(candidate)
        positions[case, :atoms] = np.asarray(placed)
        mask[case, :atoms] = True
    return phx.atomistic.AtomisticBatch(
        np.where(mask, 1, 0).astype(np.int32),
        positions,
        np.ones((count, capacity)),
        _scale(),
        atom_mask=mask,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(count)),
    )


def _labeled_source(teacher: Any, execution: Any, batch: Any) -> Any:
    labels = phx.atomistic.energy_and_forces(teacher, batch, execution)
    if not bool(jnp.all(labels.valid)):
        raise RuntimeError("Teacher labels failed; raise --maximum-neighbors.")
    return phx.atomistic.atomistic_training_source(
        batch,
        mass_unit=phx.units.DALTON,
        energy=labels.energy,
        forces=labels.forces,
    )


def _training_loader(source: Any, arguments: argparse.Namespace, /) -> Any:
    canonical = phx.atomistic.atomistic_training_loader(
        source, batch_size=arguments.batch_size, seed=arguments.seed
    )
    atom_level = canonical.mixture.schema.atom_level
    bucket = phx.atomistic.AtomisticWorkloadBucket(
        "clusters",
        cases=arguments.payload_cases,
        levels=(
            phx.atomistic.AtomisticLevelCapacity(
                atom_level, layout="per-case", rows=arguments.max_atoms
            ),
        ),
        max_bytes=1 << 30,
    )
    return phx.atomistic.AtomisticStructureLoader(
        canonical.mixture,
        canonical.bindings,
        phx.atomistic.AtomisticWorkloadBuckets((bucket,)),
        batch_size=arguments.batch_size,
        stream="training",
    )


def _active_edges(batch: Any, execution: Any, cutoff: float, /) -> int:
    graph = (
        phx.atomistic.AtomisticGraphLayout.for_batch(batch, execution, cutoff=cutoff)
        .prepare(batch)
        .graph(cutoff=cutoff)
    )
    return int(jnp.sum(graph.neighbor_counts))


def _read_pass(loader: Any, /) -> dict[str, Any]:
    """Loader reads only: payload count, padding and per-payload host seconds."""
    atom_level = loader.mixture.schema.atom_level
    seconds: list[float] = []
    cases = valid_cases = rows = real_rows = 0
    with loader.epoch(loader.start(0)) as epoch:
        while True:
            item, elapsed = measure_host(lambda: next(epoch, None))
            if item is None:
                break
            seconds.append(elapsed)
            valid = np.asarray(item.payload.case_valid, dtype=np.bool_)
            row_mask = np.asarray(item.payload.level(atom_level).row_mask, dtype=np.bool_)
            cases += valid.size
            valid_cases += int(np.count_nonzero(valid))
            rows += row_mask.size
            real_rows += int(np.count_nonzero(row_mask.reshape((valid.size, -1))[valid]))
    return {
        "payloads": len(seconds),
        "payload_seconds": seconds,
        "read_seconds": float(np.sum(seconds)),
        "case_slots": cases,
        "valid_cases": valid_cases,
        "padding_cases": cases - valid_cases,
        "atom_row_slots": rows,
        "real_atom_rows": real_rows,
        "padding_atom_rows": rows - real_rows,
    }


def _prepare_pass(problem: Any, coordinate_dtype: Any, /) -> dict[str, Any]:
    """Host preparation of every payload: layout, topology and placement."""
    reduction = phx.atomistic.AtomisticLocalObjectiveReduction()
    loader = problem.training
    seconds: list[float] = []
    staged: list[int] = []
    with loader.epoch(loader.start(0)) as epoch:
        for item in epoch:
            batch, elapsed = measure_host(
                lambda item=item: synchronize(
                    problem.preparation.prepare_step(
                        item.payload,
                        item.provenance,
                        reduction,
                        coordinate_dtype=coordinate_dtype,
                    )
                )
            )
            seconds.append(elapsed)
            staged.append(logical_array_bytes(batch))
    return {
        "prepare_seconds": float(np.sum(seconds)),
        "payload_prepare_seconds": seconds,
        "payload_logical_bytes": staged,
        "maximum_payload_logical_bytes": max(staged),
    }


def _device_peak_bytes() -> int | None:
    stats = jax.devices()[0].memory_stats()
    if not stats or "peak_bytes_in_use" not in stats:
        return None
    return int(stats["peak_bytes_in_use"])


def _fit_phase(
    potential: Any,
    problem: Any,
    policy: Any,
    /,
    *,
    key: Any,
    continuation: Any = None,
) -> tuple[Any, dict[str, Any]]:
    before = _COMPILES.snapshot()
    updates_before = 0 if continuation is None else continuation.progress.update_step
    result, seconds = measure_host(
        lambda: synchronize(
            phx.atomistic.fit_atomistic_potential(
                potential, problem, policy, key=key, continuation=continuation
            )
        )
    )
    if not bool(result.successful):
        raise RuntimeError(f"Training phase failed: {result.termination}")
    accepted = result.progress.update_step - updates_before
    return result, {
        "seconds": seconds,
        "accepted_updates": accepted,
        "seconds_per_accepted_update": seconds / accepted if accepted else None,
        "skipped_windows": result.skipped_windows,
        "validations": int(result.validation_values.shape[0]),
        "compile_events": _COMPILES.delta(before, _COMPILES.snapshot()),
        "final_loss": float(result.final_loss),
    }


def _run(arguments: argparse.Namespace, /) -> dict[str, Any]:
    generator = np.random.default_rng(arguments.seed)
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        arguments.maximum_neighbors, maximum_dense_atoms=arguments.max_atoms * 8
    )
    teacher = _potential(arguments, arguments.seed + 101)
    training_batch = _clusters(arguments.structures, arguments, generator, "train")
    validation_batch = _clusters(
        arguments.validation_structures, arguments, generator, "validation"
    )
    (training_source, validation_source), label_seconds = measure_host(
        lambda: (
            _labeled_source(teacher, execution, training_batch),
            _labeled_source(teacher, execution, validation_batch),
        )
    )
    training = _training_loader(training_source, arguments)
    validation = phx.atomistic.atomistic_training_loader(
        validation_source, stream="validation"
    )
    problem = phx.atomistic.AtomisticTrainingProblem(
        training,
        execution,
        (
            phx.atomistic.AtomisticEnergyObjective(),
            phx.atomistic.AtomisticForceObjective(),
        ),
        scale=_scale(),
        cutoff=arguments.cutoff,
        validation=validation,
    )
    student = _potential(arguments, arguments.seed + 202)
    reads, read_allocation = host_allocation_peak(lambda: _read_pass(training))
    prepared, prepare_allocation = host_allocation_peak(
        lambda: _prepare_pass(problem, student.precision.coordinate_dtype)
    )
    windows = training.batch_count
    first_steps = 1
    warm_steps = first_steps + arguments.steps
    repeat_steps = warm_steps + arguments.repeat_epochs * windows

    def policy(steps: int) -> Any:
        return phx.atomistic.AtomisticTrainingPolicy(
            maximum_steps=steps, learning_rate=1e-3, validation_every=repeat_steps + 1
        )

    key = jax.random.key(arguments.seed + 303)
    first, first_phase = _fit_phase(student, problem, policy(first_steps), key=key)
    warm, warm_phase = _fit_phase(
        student, problem, policy(warm_steps), key=key, continuation=first
    )
    repeated, repeat_phase = _fit_phase(
        student, problem, policy(repeat_steps), key=key, continuation=warm
    )
    validation_preparation = problem.validation_preparation
    if validation_preparation is None:
        raise RuntimeError(
            "The streaming benchmark requires a prepared validation source."
        )
    evaluation, evaluation_seconds = measure_host(
        lambda: phx.atomistic.evaluate_atomistic_objectives(
            repeated.trained_tree,
            problem,
            repeated.normalization,
            validation_preparation,
        )
    )
    atoms = int(np.sum(np.asarray(training_batch.atom_mask)))
    return {
        "configuration": {
            name.replace("_", "-"): value
            for name, value in vars(arguments).items()
            if name not in ("output", "label")
        },
        "identities": {
            "problem_id": problem.problem_id,
            "execution_id": problem.execution_id,
            "loader_semantics_id": training.semantics_id,
            "loader_structure_id": training.structure_id,
        },
        "workload": {
            "structures": arguments.structures,
            "atoms": atoms,
            "active_directed_edges": _active_edges(
                training_batch, execution, arguments.cutoff
            ),
            "windows_per_epoch": windows,
            "payloads_per_window": [
                training.payload_count(0, index) for index in range(windows)
            ],
        },
        "phases": {
            "label_seconds": label_seconds,
            "read": {**reads, "host_allocation_peak_bytes": read_allocation},
            "prepare_topology": {
                **prepared,
                "host_allocation_peak_bytes": prepare_allocation,
            },
            "first_update": first_phase,
            "warm_updates": warm_phase,
            "repeated_payload_epochs": repeat_phase,
            "validation_evaluation": {
                "seconds": evaluation_seconds,
                "values": list(evaluation.values),
                "successful": evaluation.successful,
            },
        },
        "memory": {
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "device_peak_bytes_in_use": _device_peak_bytes(),
        },
        "result": {
            "result_id": repeated.result_id,
            "accepted_updates": repeated.progress.update_step,
            "training_values": np.asarray(repeated.training_values).tolist(),
            "best_loss": float(repeated.best_loss),
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    _validate(arguments)
    root = Path(__file__).resolve().parents[1]
    record = {
        "benchmark": "atomistic-training-streaming",
        "label": arguments.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            ("configuration", "identities", "memory", "phases", "result", "workload"),
        ).to_dict(),
        **_run(arguments),
    }
    if arguments.output is None:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, record)
        print(arguments.output)


if __name__ == "__main__":
    main()
