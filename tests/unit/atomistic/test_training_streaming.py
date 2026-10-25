#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Streaming atomistic training: windows, rollback, selection and continuation.

Every loader global batch is one accumulation window. These scenarios fit tiny
native PaiNN/MACE potentials on teacher labels and check the merged
numerator/support semantics, committed-boundary rollback and replay, skips,
schedules at committed updates, EMA/best/plateau selection state, parameter
and irrep-multiplicity adaptation, host stops inside a window, fresh-process
checkpoint continuation, and refusals before any optimizer state exists.
Bitwise continuation claims use ordered (deterministic) graph accumulation.
"""

from __future__ import annotations

import functools
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from phydrax._trainable import partition_parameters
from phydrax._training import ExponentialMovingAverageTargetPolicy, TrainingIterationKind
from phydrax.atomistic import (
    atomistic_model_revision,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    AtomisticStatus,
    energy_and_forces,
)
from phydrax.atomistic._training import (
    atomistic_irrep_linear_sites,
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticIrrepLowRankAdaptation,
    AtomisticParameterSelection,
    AtomisticPlateauPolicy,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    evaluate_atomistic_objectives,
    fit_atomistic_potential,
    read_atomistic_training_checkpoint,
    write_atomistic_training_checkpoint,
)
from phydrax.atomistic._training_data import (
    AtomisticLevelCapacity,
    AtomisticLoaderError,
    AtomisticStructureLoader,
    AtomisticWorkloadBucket,
    AtomisticWorkloadBuckets,
)
from phydrax.atomistic._training_objectives import (
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticObjectiveSchedule,
    AtomisticStressObjective,
)
from phydrax.execution import (
    CallableIterationHostControl,
    CallableIterationSink,
    IterationSession,
)
from phydrax.nn.atomistic import PaiNNPotential
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, BOHR, DALTON, ELECTRONVOLT
from tests._support.compiler_profile import isolated_in_fixed_compiler_profile
from tests._support.mace_deployment import tiny_mace


REPOSITORY = Path(__file__).resolve().parents[3]
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 2.0
FAR = [9.0, 9.0, 9.0]
TRAIN_POSITIONS = np.asarray(
    [
        [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], FAR],
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], FAR],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.1, 0.9, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.1]],
    ]
)
TRAIN_MASK = np.asarray(
    [[True, True, False], [True, True, False], [True, True, True], [True, True, True]]
)
# Validation structures have one neighbor per atom (used by the overflow test).
VALIDATION_POSITIONS = np.asarray(
    [[[0.0, 0.0, 0.0], [0.85, 0.05, 0.0]], [[0.0, 0.0, 0.0], [0.95, 0.0, 0.1]]]
)
VALIDATION_MASK = np.ones((2, 2), dtype=np.bool_)


def _execution(maximum_neighbors: int = 4) -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=64,
        streamed=StreamedRelationPlan(accumulation="deterministic"),
    )


def _potential(
    seed: int, *, cutoff: float = CUTOFF, scale: Any = SCALE
) -> PaiNNPotential:
    return PaiNNPotential(
        scale,
        cutoff=cutoff,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(seed),
    )


def _batch(positions: np.ndarray, mask: np.ndarray, prefix: str) -> AtomisticBatch:
    return AtomisticBatch(
        np.where(mask, 1, 0).astype(np.int32),
        positions,
        np.ones(mask.shape),
        SCALE,
        atom_mask=mask,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(mask.shape[0])),
    )


@functools.cache
def _labels(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    positions, mask = (
        (TRAIN_POSITIONS, TRAIN_MASK)
        if prefix == "train"
        else (VALIDATION_POSITIONS, VALIDATION_MASK)
    )
    teacher = energy_and_forces(
        _potential(91), _batch(positions, mask, prefix), _execution()
    )
    assert bool(jnp.all(teacher.valid))
    return np.asarray(teacher.energy), np.asarray(teacher.forces)


def _source(
    prefix: str = "train",
    *,
    energy: np.ndarray | None = None,
    energy_available: np.ndarray | None = None,
    lineage_ids: tuple[str, ...] | None = None,
    record_prefix: str | None = None,
    shift: float = 0.0,
) -> Any:
    positions, mask = (
        (TRAIN_POSITIONS, TRAIN_MASK)
        if prefix == "train"
        else (VALIDATION_POSITIONS, VALIDATION_MASK)
    )
    labels, forces = _labels(prefix)
    batch = _batch(positions + shift, mask, record_prefix or prefix)
    return atomistic_training_source(
        batch,
        mass_unit=DALTON,
        energy=labels if energy is None else energy,
        forces=forces,
        energy_available=energy_available,
        lineage_ids=lineage_ids,
    )


def _loader(
    source: Any,
    *,
    batch_size: int | None = None,
    payload_cases: int | None = None,
    stream: str = "training",
) -> AtomisticStructureLoader:
    """Ordered loader; ``payload_cases`` below the batch size accumulates a window."""
    full = atomistic_training_loader(
        source, batch_size=batch_size, stream=stream, shuffle=False
    )
    if payload_cases is None:
        return full
    atom_level = full.mixture.schema.atom_level
    rows = full.buckets.buckets[0].capacity(atom_level).rows
    bucket = AtomisticWorkloadBucket(
        "accumulated",
        cases=payload_cases,
        levels=(AtomisticLevelCapacity(atom_level, layout="per-case", rows=rows),),
        max_bytes=1 << 30,
    )
    return AtomisticStructureLoader(
        full.mixture,
        full.bindings,
        AtomisticWorkloadBuckets((bucket,)),
        batch_size=full.batch_size,
        stream=stream,
    )


def _validation() -> AtomisticStructureLoader:
    return _loader(_source("validation"), stream="validation")


def _objectives() -> tuple[Any, ...]:
    return (AtomisticEnergyObjective(scale=1.0), AtomisticForceObjective(scale=1.0))


def _problem(
    training: AtomisticStructureLoader,
    *,
    objectives: tuple[Any, ...] | None = None,
    execution: AtomisticGraphExecutionPlan | None = None,
    validation: AtomisticStructureLoader | None = None,
    weighting: Any = "literal",
) -> AtomisticTrainingProblem:
    return AtomisticTrainingProblem(
        training,
        _execution() if execution is None else execution,
        _objectives() if objectives is None else objectives,
        scale=SCALE,
        cutoff=CUTOFF,
        validation=_validation() if validation is None else validation,
        weighting=weighting,
    )


def _accumulated_problem() -> AtomisticTrainingProblem:
    """Two payloads of two cases per four-structure window (fresh-process builder)."""
    return _problem(_loader(_source(), payload_cases=2))


def _adam(steps: int, **options: Any) -> AtomisticTrainingPolicy:
    return AtomisticTrainingPolicy(maximum_steps=steps, learning_rate=1e-2, **options)


def _sgd(steps: int) -> AtomisticTrainingPolicy:
    return AtomisticTrainingPolicy(
        maximum_steps=steps, optimizer=optax.sgd(0.05), optimizer_id="optax.sgd:0.05"
    )


def _trainable(model: Any) -> list[np.ndarray]:
    trainable, _, _ = partition_parameters(model)
    return [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if eqx.is_inexact_array(leaf)
    ]


def _revision(model: Any) -> str:
    return atomistic_model_revision(model).revision_id


def _assert_trees_bitwise_equal(observed: Any, expected: Any) -> None:
    observed_leaves, observed_structure = jax.tree_util.tree_flatten(observed)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert observed_structure == expected_structure
    for left, right in zip(observed_leaves, expected_leaves, strict=True):
        if isinstance(left, jax.Array):
            assert bool(jnp.array_equal(left, right))
        else:
            assert left == right


def _event_kind(event: Any) -> TrainingIterationKind:
    return TrainingIterationKind(int(event.record.metrics.kind))


# ----------------------------------------------------------------- windows


def test_accumulated_window_merges_numerators_and_supports_like_one_full_batch() -> None:
    available = np.asarray([True, False, True, True])
    source = _source(energy_available=available)
    full = _problem(_loader(source))
    accumulated = _problem(_loader(source, payload_cases=2))
    assert full.problem_id == accumulated.problem_id
    assert full.execution_id != accumulated.execution_id
    assert full.training.payload_count(0, 0) == 1
    assert accumulated.training.payload_count(0, 0) == 2

    initial = _potential(3)
    one = fit_atomistic_potential(initial, full, _sgd(1), key=jr.key(4))
    two = fit_atomistic_potential(initial, accumulated, _sgd(1), key=jr.key(4))
    assert bool(one.successful) and bool(two.successful)

    # Independent float64 window objective at the initial parameters.
    prediction = energy_and_forces(
        initial, _batch(TRAIN_POSITIONS, TRAIN_MASK, "train"), _execution()
    )
    labels, forces = _labels("train")
    counts = np.sum(TRAIN_MASK, axis=1)
    energy_residual = (np.asarray(prediction.energy) - labels) / counts
    energy_reference = float(np.sum(energy_residual[available] ** 2) / np.sum(available))
    force_residual = (np.asarray(prediction.forces) - forces)[TRAIN_MASK]
    force_reference = float(np.sum(force_residual**2) / (3.0 * np.sum(TRAIN_MASK)))
    for result in (one, two):
        np.testing.assert_allclose(
            np.asarray(result.training_objectives)[0],
            [energy_reference, force_reference],
            rtol=1e-9,
        )
        assert float(result.training_values[0]) == pytest.approx(
            energy_reference + force_reference, rel=1e-9
        )
    # Averaging the two payload means would weight the lone available energy of
    # the first payload like two energies of the second: a different objective.
    mean_of_means = 0.5 * (
        energy_residual[0] ** 2 + float(np.mean(energy_residual[2:] ** 2))
    )
    assert not math.isclose(mean_of_means, energy_reference, rel_tol=1e-6)
    # One SGD step of the merged window gradient: identical updates up to the
    # (documented) reduction-grouping rounding.
    for left, right in zip(
        _trainable(one.potential), _trainable(two.potential), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-12)
    assert _revision(two.potential) != _revision(initial)


def test_nonfinite_label_rolls_back_to_the_last_committed_boundary_and_replays() -> None:
    labels, _ = _labels("train")
    poisoned = labels.copy()
    poisoned[3] = np.nan
    problem = _problem(_loader(_source(energy=poisoned), batch_size=2))
    initial = _potential(6)
    failed = fit_atomistic_potential(initial, problem, _adam(3), key=jr.key(7))

    assert int(failed.status) == int(AtomisticStatus.NONFINITE)
    assert not bool(failed.successful)
    assert failed.termination == "nonfinite_update_window"
    assert failed.progress.update_step == 1
    assert failed.training_values.shape == (1,)
    assert (
        failed.cursor.epoch,
        failed.cursor.batch_index,
        failed.cursor.payload_index,
    ) == (
        0,
        1,
        0,
    )
    reference = fit_atomistic_potential(initial, problem, _adam(1), key=jr.key(7))
    assert _revision(failed.potential) == _revision(reference.potential)

    replayed = fit_atomistic_potential(
        initial, problem, _adam(3), key=jr.key(8), continuation=failed
    )
    assert int(replayed.status) == int(AtomisticStatus.NONFINITE)
    assert replayed.progress.update_step == 1
    assert replayed.cursor.cursor_id == failed.cursor.cursor_id
    assert _revision(replayed.potential) == _revision(failed.potential)


def test_capacity_overflow_stops_at_the_boundary_and_a_grown_capacity_replays() -> None:
    training = _loader(_source(), batch_size=2)
    small = _problem(training, execution=_execution(1))
    grown = _problem(training, execution=_execution(4))
    assert small.problem_id == grown.problem_id
    assert small.execution_id != grown.execution_id
    initial = _potential(9)
    failed = fit_atomistic_potential(initial, small, _adam(3), key=jr.key(10))

    assert int(failed.status) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert failed.termination == "capacity_failure_in_update_window"
    assert failed.capacity_failures > 0
    assert failed.progress.update_step == 1
    assert failed.cursor.batch_index == 1

    continued = fit_atomistic_potential(
        initial, grown, _adam(3), key=jr.key(999), continuation=failed
    )
    uninterrupted = fit_atomistic_potential(initial, grown, _adam(3), key=jr.key(10))
    assert bool(continued.successful)
    assert continued.progress.update_step == 3
    np.testing.assert_allclose(
        continued.training_values, uninterrupted.training_values, rtol=1e-9
    )
    for left, right in zip(
        _trainable(continued.potential), _trainable(uninterrupted.potential), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-12)


def test_zero_support_window_is_a_committed_skip_that_never_advances_schedules() -> None:
    available = np.asarray([False, False, True, True])
    scheduled = AtomisticObjectiveSchedule.piecewise((1,), (1.0, 0.5))
    problem = _problem(
        _loader(_source(energy_available=available), batch_size=2),
        objectives=(
            AtomisticEnergyObjective(scale=1.0),
            AtomisticEnergyObjective(
                name="energy-scheduled", scale=1.0, schedule=scheduled
            ),
        ),
    )
    initial = _potential(11)
    result = fit_atomistic_potential(initial, problem, _adam(1), key=jr.key(12))

    assert bool(result.successful)
    assert result.skipped_windows == 1
    assert result.progress.update_step == 1
    assert result.training_values.shape == (1,)
    assert (result.cursor.epoch, result.cursor.batch_index) == (1, 0)
    objectives = np.asarray(result.training_objectives)[0]
    # The accepted update after the skip still sees committed step 0.
    assert objectives[1] == pytest.approx(objectives[0], rel=1e-14)
    assert _revision(result.potential) != _revision(initial)


def _unsupported_problem() -> AtomisticTrainingProblem:
    """Energy-only problem whose two-window training epoch has no energy label."""
    return _problem(
        _loader(_source(energy_available=np.zeros(4, dtype=np.bool_)), batch_size=2),
        objectives=(AtomisticEnergyObjective(scale=1.0),),
    )


def _collecting_session(
    name: str, events: list[TrainingIterationKind], stop: TrainingIterationKind | None
) -> IterationSession:
    return IterationSession(
        name,
        sinks=(
            CallableIterationSink(
                lambda event: events.append(_event_kind(event)), "collector"
            ),
        ),
        control=None
        if stop is None
        else CallableIterationHostControl(
            lambda event: _event_kind(event) is stop, f"stop-on-{stop.name.lower()}"
        ),
    )


def test_completely_unsupported_epoch_stops_at_its_boundary_without_any_update() -> None:
    problem = _unsupported_problem()
    assert problem.training.batch_count == 2
    initial = _potential(13)
    events: list[TrainingIterationKind] = []
    result = fit_atomistic_potential(
        initial,
        problem,
        _adam(3),
        key=jr.key(14),
        session=_collecting_session("atomistic-unsupported-epoch", events, None),
    )

    assert int(result.status) == int(AtomisticStatus.STOPPED_EARLY)
    assert result.termination == "unsupported_training_epoch"
    assert not bool(result.successful)
    assert result.progress.update_step == 0
    assert result.training_values.shape == (0,)
    assert result.skipped_windows == result.unsupported_windows == 2
    assert (
        result.cursor.epoch,
        result.cursor.batch_index,
        result.cursor.payload_index,
    ) == (1, 0, 0)
    assert events.count(TrainingIterationKind.SKIP) == 2
    assert TrainingIterationKind.UPDATE not in events
    assert TrainingIterationKind.FAILURE not in events
    assert events[-1] is TrainingIterationKind.RUN_TERMINAL

    # Independent control: the same run stopped before its first window holds
    # the kernel's initial state; two skips moved only the attempt bookkeeping.
    untouched = fit_atomistic_potential(
        initial,
        problem,
        _adam(3),
        key=jr.key(14),
        session=_collecting_session(
            "atomistic-unsupported-untouched", [], TrainingIterationKind.RUN_START
        ),
    )
    assert untouched.termination == "host_control_stop_before_first_update"
    for name in ("parameters", "model_state", "rule_state", "targets"):
        _assert_trees_bitwise_equal(
            getattr(result.training_state, name), getattr(untouched.training_state, name)
        )
    assert int(result.training_state.accepted_cursor) == 0
    assert int(result.training_state.attempt_cursor) == 2
    assert int(untouched.training_state.attempt_cursor) == 0
    assert _revision(result.potential) == _revision(initial)


def test_unsupported_epoch_termination_is_deterministic_across_resume_and_restore(
    tmp_path: Path,
) -> None:
    problem = _unsupported_problem()
    initial = _potential(15)
    uninterrupted = fit_atomistic_potential(initial, problem, _adam(3), key=jr.key(16))

    stopped = fit_atomistic_potential(
        initial,
        problem,
        _adam(3),
        key=jr.key(16),
        session=_collecting_session(
            "atomistic-unsupported-stop", [], TrainingIterationKind.SKIP
        ),
    )
    assert stopped.termination == "host_control_stop_after_skipped_window"
    assert bool(stopped.successful)
    assert stopped.skipped_windows == stopped.unsupported_windows == 1
    assert (stopped.cursor.epoch, stopped.cursor.batch_index) == (0, 1)

    resumed = fit_atomistic_potential(
        initial, problem, _adam(3), key=jr.key(999), continuation=stopped
    )
    assert (
        resumed.termination == uninterrupted.termination == "unsupported_training_epoch"
    )
    assert int(resumed.status) == int(AtomisticStatus.STOPPED_EARLY)
    assert resumed.skipped_windows == resumed.unsupported_windows == 2
    assert resumed.cursor.cursor_id == uninterrupted.cursor.cursor_id
    for name in (
        "parameters",
        "rule_state",
        "targets",
        "attempt_cursor",
        "accepted_cursor",
        "finite_rejections",
    ):
        _assert_trees_bitwise_equal(
            getattr(resumed.training_state, name),
            getattr(uninterrupted.training_state, name),
        )
    _assert_trees_bitwise_equal(resumed.trained_tree, uninterrupted.trained_tree)

    path = write_atomistic_training_checkpoint(
        tmp_path / "unsupported", uninterrupted, problem, _adam(3)
    )
    restored = read_atomistic_training_checkpoint(path, initial, problem, _adam(5))
    assert restored.result_id == uninterrupted.result_id
    assert restored.unsupported_windows == 2
    assert restored.termination == "unsupported_training_epoch"

    # Continuing over the unchanged problem refuses again without reading data.
    events: list[TrainingIterationKind] = []
    continued = fit_atomistic_potential(
        initial,
        problem,
        _adam(5),
        key=jr.key(999),
        continuation=restored,
        session=_collecting_session("atomistic-unsupported-continued", events, None),
    )
    assert continued.termination == "unsupported_training_epoch"
    assert not bool(continued.successful)
    assert continued.progress.update_step == 0
    assert continued.skipped_windows == continued.unsupported_windows == 2
    assert continued.cursor.cursor_id == uninterrupted.cursor.cursor_id
    assert int(continued.training_state.attempt_cursor) == 2
    assert TrainingIterationKind.SKIP not in events
    assert TrainingIterationKind.EPOCH_START not in events


def test_unsupported_windows_spanning_an_epoch_boundary_keep_training() -> None:
    # Batch size one: windows 0, 2, 3 lack energy. Skips 2, 3 of epoch 0 and
    # 0 of epoch 1 are consecutive across the boundary but never a whole epoch.
    available = np.asarray([False, True, False, False])
    problem = _problem(
        _loader(_source(energy_available=available), batch_size=1),
        objectives=(AtomisticEnergyObjective(scale=1.0),),
    )
    assert problem.training.batch_count == 4
    initial = _potential(17)
    result = fit_atomistic_potential(initial, problem, _adam(2), key=jr.key(18))

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert result.termination == "maximum_steps"
    assert bool(result.successful)
    assert result.progress.update_step == 2
    assert result.skipped_windows == 4
    assert result.unsupported_windows == 0
    assert (result.cursor.epoch, result.cursor.batch_index) == (1, 2)
    assert result.training_values.shape == (2,)
    assert int(result.training_state.accepted_cursor) == 2
    assert _revision(result.potential) != _revision(initial)


@pytest.mark.parametrize(
    ("schedule", "weights"),
    [
        (AtomisticObjectiveSchedule.constant(0.5), [0.5, 0.5, 0.5, 0.5]),
        (AtomisticObjectiveSchedule.linear(0.0, 1.0, steps=2), [0.0, 0.5, 1.0, 1.0]),
        (
            AtomisticObjectiveSchedule.cosine(1.0, 0.0, steps=3),
            [0.5 * (1.0 + math.cos(math.pi * min(k / 3.0, 1.0))) for k in range(4)],
        ),
        (
            AtomisticObjectiveSchedule.piecewise((1, 3), (1.0, 0.25, 0.5)),
            [1.0, 0.25, 0.25, 0.5],
        ),
    ],
    ids=["constant", "linear", "cosine", "piecewise"],
)
def test_objective_weights_follow_committed_updates_and_stay_fixed_in_a_window(
    schedule: AtomisticObjectiveSchedule, weights: list[float]
) -> None:
    # Two identical energy objectives over a two-payload window: their ratio is
    # exactly the scheduled weight at the committed update count; a weight that
    # advanced per payload would mix two steps inside one window.
    problem = _problem(
        _loader(_source(), payload_cases=2),
        objectives=(
            AtomisticEnergyObjective(scale=1.0),
            AtomisticEnergyObjective(
                name="energy-scheduled", scale=1.0, schedule=schedule
            ),
        ),
    )
    result = fit_atomistic_potential(_potential(13), problem, _adam(4), key=jr.key(14))
    assert bool(result.successful)
    objectives = np.asarray(result.training_objectives)
    expected = np.asarray(weights)
    np.testing.assert_allclose(objectives[:, 1], expected * objectives[:, 0], rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.training_values),
        objectives[:, 0] * (1.0 + expected),
        rtol=1e-12,
    )


def test_normalized_weighting_divides_by_the_schedule_sum_at_the_same_update() -> None:
    schedule = AtomisticObjectiveSchedule.piecewise((2,), (3.0, 1.0))

    def fit(weighting: Any) -> Any:
        problem = _problem(
            _loader(_source(), payload_cases=2),
            objectives=(
                AtomisticEnergyObjective(scale=1.0),
                AtomisticEnergyObjective(
                    name="energy-scheduled", scale=1.0, schedule=schedule
                ),
            ),
            weighting=weighting,
        )
        return problem, fit_atomistic_potential(
            _potential(15), problem, _adam(1), key=jr.key(16)
        )

    literal_problem, literal = fit("literal")
    normalized_problem, normalized = fit("normalized")
    assert literal_problem.problem_id != normalized_problem.problem_id
    literal_values = np.asarray(literal.training_objectives)[0]
    normalized_values = np.asarray(normalized.training_objectives)[0]
    # Same initial parameters: unweighted value v, weights (1, 3) at step 0.
    assert normalized_values[0] * 4.0 == pytest.approx(literal_values[0], rel=1e-12)
    assert normalized_values[1] == pytest.approx(3.0 * normalized_values[0], rel=1e-12)
    assert float(normalized.training_values[0]) == pytest.approx(
        float(literal_values[0]), rel=1e-12
    )
    with pytest.raises(ValueError, match="positive at every committed update"):
        _problem(
            _loader(_source()),
            objectives=(
                AtomisticEnergyObjective(
                    scale=1.0,
                    schedule=AtomisticObjectiveSchedule.linear(1.0, 0.0, steps=2),
                ),
            ),
            weighting="normalized",
        )


# ---------------------------------------------------------------- selection


def test_ema_target_view_drives_validation_and_the_evaluation_tree() -> None:
    decay = 0.75
    problem = _problem(_loader(_source()))
    initial = _potential(17)
    result = fit_atomistic_potential(
        initial,
        problem,
        _adam(
            1,
            target=ExponentialMovingAverageTargetPolicy(decay=decay),
            evaluation="target",
            select_best=False,
        ),
        key=jr.key(18),
    )
    assert bool(result.successful)
    for start, current, target in zip(
        _trainable(initial),
        _trainable(result.potential),
        _trainable(result.best_potential),
        strict=True,
    ):
        np.testing.assert_allclose(
            target, decay * start + (1.0 - decay) * current, rtol=1e-12, atol=1e-15
        )
    assert _revision(result.best_potential) != _revision(result.potential)
    validation = problem.validation_preparation
    assert validation is not None
    evaluation = evaluate_atomistic_objectives(
        result.best_tree, problem, result.normalization, validation
    )
    assert float(result.validation_values[-1]) == pytest.approx(
        sum(evaluation.values), rel=1e-10
    )


def test_best_selection_keeps_the_minimum_validation_view_and_step() -> None:
    problem = _problem(_loader(_source()))
    result = fit_atomistic_potential(
        _potential(19), problem, _adam(4, validation_every=1), key=jr.key(20)
    )
    assert bool(result.successful)
    values = np.asarray(result.validation_values)
    steps = np.asarray(result.validation_steps)
    np.testing.assert_array_equal(steps, [0, 1, 2, 3, 4])
    best = int(np.argmin(values))
    assert float(result.best_loss) == pytest.approx(float(values[best]), rel=1e-14)
    assert result.progress.best_step == int(steps[best])
    validation = problem.validation_preparation
    assert validation is not None
    evaluation = evaluate_atomistic_objectives(
        result.best_tree, problem, result.normalization, validation
    )
    assert sum(evaluation.values) == pytest.approx(float(result.best_loss), rel=1e-10)


def test_plateau_scaling_is_a_committed_rule_state_transition() -> None:
    plateau = AtomisticPlateauPolicy(factor=0.5, patience=0, threshold=1e6)
    problem = _problem(_loader(_source()))
    result = fit_atomistic_potential(
        _potential(21), problem, _adam(2, plateau=plateau), key=jr.key(22)
    )
    assert bool(result.successful)
    rule = result.training_state.rule_state
    # Validation 0 sets the best; validations 1 and 2 are stale (threshold) and
    # each halves the committed update scale.
    assert float(rule.update_scale) == 0.25
    assert int(rule.stale_validations) == 0
    assert float(rule.best_validation) == float(result.validation_values[0])
    assert _adam(2, plateau=plateau).continuation_id != _adam(2).continuation_id


# --------------------------------------------------------------- adaptation


def test_parameter_selection_trains_only_the_selected_leaves() -> None:
    initial = _potential(23)
    selected = tuple(
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(initial)[0]
        if path
        and path[0] == jax.tree_util.GetAttrKey("readout_energy")
        and eqx.is_inexact_array(leaf)
    )
    assert selected
    result = fit_atomistic_potential(
        initial,
        _problem(_loader(_source())),
        _adam(2, adaptation=AtomisticParameterSelection(leaf_paths=selected)),
        key=jr.key(24),
    )
    assert bool(result.successful)
    before = jax.tree_util.tree_flatten_with_path(
        eqx.filter(initial, eqx.is_inexact_array)
    )[0]
    after = jax.tree_util.tree_flatten_with_path(
        eqx.filter(result.potential, eqx.is_inexact_array)
    )[0]
    changed = False
    for (path, old), (_, new) in zip(before, after, strict=True):
        if jax.tree_util.keystr(path) in selected:
            changed = changed or not bool(jnp.array_equal(old, new))
        else:
            assert bool(jnp.array_equal(old, new)), jax.tree_util.keystr(path)
    assert changed


WATER = np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
ROTATION = np.asarray([[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]])


def _water_batch(
    offsets: list[float], prefix: str, *, rotation: Any = None
) -> AtomisticBatch:
    positions = np.stack(
        [
            WATER
            + offset * np.asarray([[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.0, -0.08, 0.1]])
            for offset in offsets
        ]
    )
    if rotation is not None:
        positions = positions @ np.asarray(rotation).T
    count = len(offsets)
    return AtomisticBatch(
        np.broadcast_to(np.asarray([8, 1, 1], dtype=np.int32), (count, 3)),
        positions,
        np.broadcast_to(np.asarray([15.999, 1.008, 1.008]), (count, 3)),
        SCALE,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(count)),
    )


def test_irrep_low_rank_adaptation_keeps_equivariance_and_refuses_unknown_sites() -> None:
    execution = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=8)
    teacher = tiny_mace(seed=11)

    def source(offsets: list[float], prefix: str) -> Any:
        batch = _water_batch(offsets, prefix)
        labels = energy_and_forces(teacher, batch, execution)
        return atomistic_training_source(
            batch,
            mass_unit=DALTON,
            energy=np.asarray(labels.energy),
            forces=np.asarray(labels.forces),
        )

    problem = AtomisticTrainingProblem(
        atomistic_training_loader(source([-1.0, 0.5, 1.5], "water-train"), shuffle=False),
        execution,
        _objectives(),
        scale=SCALE,
        cutoff=3.0,
        validation=atomistic_training_loader(
            source([0.25], "water-validation"), stream="validation"
        ),
    )
    initial = tiny_mace(seed=4)
    sites = atomistic_irrep_linear_sites(initial)
    assert sites
    adaptation = AtomisticIrrepLowRankAdaptation(sites[:2], rank=1, seed=3)
    result = fit_atomistic_potential(
        initial, problem, _adam(2, adaptation=adaptation), key=jr.key(5)
    )
    assert bool(result.successful)
    tree = result.trained_tree.model
    assert any(bool(jnp.any(factor != 0.0)) for site in tree.left for factor in site)
    assert _revision(result.trained_tree.base_potential()) == _revision(initial)
    assert _revision(result.potential) != _revision(initial)

    reference = energy_and_forces(
        result.potential, _water_batch([0.7], "probe"), execution
    )
    rotated = energy_and_forces(
        result.potential, _water_batch([0.7], "probe", rotation=ROTATION), execution
    )
    np.testing.assert_allclose(rotated.energy, reference.energy, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        rotated.forces, np.asarray(reference.forces) @ ROTATION.T, rtol=0.0, atol=1e-9
    )

    with pytest.raises(ValueError, match="Unknown O3IrrepLinear adaptation sites"):
        fit_atomistic_potential(
            initial,
            problem,
            _adam(
                1,
                adaptation=AtomisticIrrepLowRankAdaptation((".not_a_site",), rank=1),
            ),
        )


# ------------------------------------------------------------- continuation


def test_host_stop_inside_a_window_continues_like_an_uninterrupted_run() -> None:
    problem = _accumulated_problem()
    initial = _potential(25)
    events: list[TrainingIterationKind] = []
    session = IterationSession(
        "atomistic-mid-window-stop",
        sinks=(
            CallableIterationSink(
                lambda event: events.append(_event_kind(event)), "collector"
            ),
        ),
        control=CallableIterationHostControl(
            lambda event: _event_kind(event) is TrainingIterationKind.MICROSTEP,
            "stop-on-first-microstep",
        ),
    )
    stopped = fit_atomistic_potential(
        initial, problem, _adam(3), key=jr.key(26), session=session
    )
    assert int(stopped.status) == int(AtomisticStatus.STOPPED_EARLY)
    assert stopped.termination == "host_control_stop_inside_accumulation_window"
    assert stopped.progress.update_step == 0
    assert (stopped.cursor.batch_index, stopped.cursor.payload_index) == (0, 1)
    assert TrainingIterationKind.MICROSTEP in events
    assert TrainingIterationKind.UPDATE not in events

    continued = fit_atomistic_potential(
        initial, problem, _adam(3), key=jr.key(999), continuation=stopped
    )
    uninterrupted = fit_atomistic_potential(initial, problem, _adam(3), key=jr.key(26))
    assert int(continued.status) == int(AtomisticStatus.SUCCESS)
    assert continued.progress.update_step == 3
    _assert_trees_bitwise_equal(continued.trained_tree, uninterrupted.trained_tree)
    _assert_trees_bitwise_equal(continued.best_tree, uninterrupted.best_tree)
    np.testing.assert_array_equal(
        continued.training_values, uninterrupted.training_values
    )
    np.testing.assert_array_equal(
        continued.validation_values, uninterrupted.validation_values
    )
    assert continued.cursor.cursor_id == uninterrupted.cursor.cursor_id


_FRESH_CONTINUATION = """
import json
import sys

import jax
import numpy as np

from phydrax.atomistic import atomistic_model_revision
from phydrax.atomistic._training import (
    fit_atomistic_potential,
    read_atomistic_training_checkpoint,
)
from tests.unit.atomistic.test_training_streaming import (
    _accumulated_problem,
    _adam,
    _potential,
)

path, total = sys.argv[1], int(sys.argv[2])
problem = _accumulated_problem()
policy = _adam(total)
restored = read_atomistic_training_checkpoint(path, _potential(27), problem, policy)
continued = fit_atomistic_potential(
    _potential(27), problem, policy, key=jax.random.key(999), continuation=restored
)
print(json.dumps({
    "values": np.asarray(continued.training_values).tolist(),
    "validation": np.asarray(continued.validation_values).tolist(),
    "revision": atomistic_model_revision(continued.potential).revision_id,
    "cursor": continued.cursor.cursor_id,
    "updates": continued.progress.update_step,
}))
"""


def test_fresh_process_checkpoint_continues_like_an_uninterrupted_run(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    if isolated_in_fixed_compiler_profile(request):
        return
    problem = _accumulated_problem()
    partial = fit_atomistic_potential(_potential(27), problem, _adam(2), key=jr.key(28))
    assert bool(partial.successful)
    path = write_atomistic_training_checkpoint(
        tmp_path / "training-checkpoint", partial, problem, _adam(2)
    )
    uninterrupted = fit_atomistic_potential(
        _potential(27), problem, _adam(4), key=jr.key(28)
    )
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_FRESH_CONTINUATION), str(path), "4"],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    fresh = json.loads(completed.stdout.strip().splitlines()[-1])
    assert fresh["updates"] == 4
    np.testing.assert_array_equal(
        fresh["values"], np.asarray(uninterrupted.training_values)
    )
    np.testing.assert_array_equal(
        fresh["validation"], np.asarray(uninterrupted.validation_values)
    )
    assert fresh["revision"] == _revision(uninterrupted.potential)
    assert fresh["cursor"] == uninterrupted.cursor.cursor_id


# ------------------------------------------------------------------ refusal


def test_validation_lineage_leakage_is_refused() -> None:
    training = _loader(_source())
    shared_lineage = _loader(
        _source(
            record_prefix="held-out",
            lineage_ids=("train-0", "unit-1", "unit-2", "unit-3"),
            shift=0.01,
        ),
        stream="validation",
    )
    with pytest.raises(AtomisticLoaderError, match="lineage"):
        _problem(training, validation=shared_lineage)
    shared_record = _loader(
        _source(lineage_ids=tuple(f"unit-{index}" for index in range(4)), shift=0.01),
        stream="validation",
    )
    with pytest.raises(AtomisticLoaderError, match="record identity"):
        _problem(training, validation=shared_record)


def test_bad_model_support_and_resource_inputs_are_refused_before_optimizer_state() -> (
    None
):
    problem = _problem(_loader(_source()))

    def refused(potential: Any, policy: Any, match: str, **options: Any) -> None:
        events: list[Any] = []
        session = IterationSession(
            f"refusal-{match}",
            sinks=(CallableIterationSink(events.append, "collector"),),
        )
        with pytest.raises(ValueError, match=match):
            fit_atomistic_potential(
                potential, problem, policy, session=session, **options
            )
        # Refused before the kernel and its optimizer state: no event was emitted.
        assert events == []

    refused(
        _potential(1, scale=AtomisticScaleContract(BOHR, ELECTRONVOLT)),
        _adam(1),
        "scales differ",
    )
    refused(_potential(1, cutoff=2.5), _adam(1), "smaller than the model cutoff")
    refused(_potential(1), _adam(1), "unknown objectives", components={"missing": None})

    with pytest.raises(ValueError, match="EMA target"):
        _adam(1, evaluation="target")
    with pytest.raises(ValueError, match="no label slot 'stress'"):
        _problem(
            _loader(_source()),
            objectives=(AtomisticStressObjective(scale=1.0),),
        )
    with pytest.raises(ValueError, match="unique"):
        _problem(
            _loader(_source()),
            objectives=(AtomisticEnergyObjective(), AtomisticEnergyObjective()),
        )
    with pytest.raises(TypeError, match="nonempty"):
        _problem(_loader(_source()), objectives=())
    full = atomistic_training_loader(_source(), shuffle=False)
    atom_level = full.mixture.schema.atom_level
    with pytest.raises(AtomisticLoaderError):
        AtomisticStructureLoader(
            full.mixture,
            full.bindings,
            AtomisticWorkloadBuckets(
                (
                    AtomisticWorkloadBucket(
                        "too_small",
                        cases=2,
                        levels=(
                            AtomisticLevelCapacity(atom_level, layout="per-case", rows=2),
                        ),
                        max_bytes=1 << 30,
                    ),
                )
            ),
            batch_size=4,
            stream="training",
        )
