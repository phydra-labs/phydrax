from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._training import TrainingIterationKind
from phydrax.atomistic import (
    atomistic_model_revision,
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticBatch,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    AtomisticStatus,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    energy_and_forces,
    fit_atomistic_potential,
    load_rmd17_npz,
    split_rmd17,
)
from phydrax.execution import CallableIterationSink, IterationSession
from phydrax.nn.atomistic import NequIPPotential, PaiNNPotential
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import (
    ANGSTROM,
    DALTON,
    ELECTRONVOLT,
    KILOCALORIE_PER_MOLE,
)


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 2.0


def _execution(
    maximum_neighbors: Any = 3, *, streamed: StreamedRelationPlan | None = None
) -> Any:
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=256,
        streamed=streamed,
    )


def _batch(shift: float = 0.0, *, prefix: str = "train") -> Any:
    return AtomisticBatch(
        # ty: ignore[invalid-argument-type]
        [[1, 1], [1, 1], [1, 1]],
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]],
            ]
        )
        + shift,
        # ty: ignore[invalid-argument-type]
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        SCALE,
        structure_ids=tuple(f"{prefix}-{index}" for index in range(3)),
    )


def _potential(key: Any) -> Any:
    return PaiNNPotential(
        SCALE,
        cutoff=2.0,
        feature_count=6,
        interaction_count=1,
        radial_basis_count=4,
        key=key,
    )


def _targets(batch: Any) -> Any:
    teacher = _potential(jr.key(91))
    prediction = energy_and_forces(teacher, batch, _execution())
    return prediction.energy, prediction.forces


def _loader(
    batch: Any,
    *,
    energy: Any = None,
    forces: Any = None,
    stream: str = "training",
    **availability: Any,
) -> Any:
    return atomistic_training_loader(
        atomistic_training_source(
            batch, mass_unit=DALTON, energy=energy, forces=forces, **availability
        ),
        stream=stream,
    )


def _problem(
    batch: Any,
    *,
    energy: Any = None,
    forces: Any = None,
    execution: Any = None,
    validation: Any = None,
    energy_scale: float | None = None,
    **availability: Any,
) -> Any:
    """Full-batch problem with one objective per supplied label kind."""
    objectives = []
    if energy is not None:
        objectives.append(AtomisticEnergyObjective(scale=energy_scale))
    if forces is not None:
        objectives.append(AtomisticForceObjective())
    return AtomisticTrainingProblem(
        _loader(batch, energy=energy, forces=forces, **availability),
        _execution() if execution is None else execution,
        tuple(objectives),
        scale=SCALE,
        cutoff=CUTOFF,
        validation=validation,
    )


def _held_out(**availability: Any) -> Any:
    batch = _batch(0.2, prefix="validation")
    energy, _ = _targets(batch)
    return _loader(batch, energy=energy, stream="validation", **availability)


def _assert_trees_bitwise_equal(observed: Any, expected: Any) -> None:
    observed_leaves, observed_structure = jax.tree_util.tree_flatten(observed)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert observed_structure == expected_structure
    for observed_leaf, expected_leaf in zip(
        observed_leaves, expected_leaves, strict=True
    ):
        if isinstance(observed_leaf, jax.Array):
            assert bool(jnp.array_equal(observed_leaf, expected_leaf))
        else:
            assert observed_leaf == expected_leaf


def test_training_and_rmd17_scenario_1() -> None:
    for target_kind in ["energy", "force", "joint"]:
        batch = _batch()
        energy, forces = _targets(batch)
        problem = _problem(
            batch,
            energy=energy if target_kind != "force" else None,
            forces=forces if target_kind != "energy" else None,
        )
        policy = AtomisticTrainingPolicy(maximum_steps=8, learning_rate=2e-3)
        result = fit_atomistic_potential(
            _potential(jr.key(3)), problem, policy, key=jr.key(12)
        )
        objective_count = 2 if target_kind == "joint" else 1
        assert result.training_values.shape == (8,)
        assert result.training_objectives.shape == (8, objective_count)
        np.testing.assert_allclose(
            np.sum(result.training_objectives, axis=1), result.training_values, rtol=1e-12
        )
        assert float(result.training_values[-1]) < float(result.training_values[0])
        assert bool(result.successful)
    # Fitted normalization reads the training labels only.
    batch = _batch()
    energy, _ = _targets(batch)
    problem_a = _problem(
        batch,
        energy=energy,
        validation=_loader(
            _batch(0.2, prefix="validation"),
            energy=jnp.asarray([1e3, -1e3, 2e3]),
            stream="validation",
        ),
    )
    problem_b = _problem(
        batch,
        energy=energy,
        validation=_loader(
            _batch(0.2, prefix="validation"),
            energy=jnp.asarray([-4e8, 7e8, 9e8]),
            stream="validation",
        ),
    )
    assert problem_a.problem_id != problem_b.problem_id
    policy = AtomisticTrainingPolicy(maximum_steps=0)
    first = fit_atomistic_potential(_potential(jr.key(2)), problem_a, policy)
    second = fit_atomistic_potential(_potential(jr.key(2)), problem_b, policy)
    assert first.normalization.scales == second.normalization.scales
    assert first.normalization.fitted == (True,)
    assert first.normalization.fitted_from == problem_a.training.semantics_id
    assert float(first.validation_values[0]) != float(second.validation_values[0])
    # Bitwise continuation requires ordered accumulation: fast reductions keep
    # their nondeterministic rounding on accelerators.
    problem = _problem(
        batch,
        energy=energy,
        execution=_execution(streamed=StreamedRelationPlan(accumulation="deterministic")),
    )
    initial = _potential(jr.key(4))
    first = fit_atomistic_potential(
        initial, problem, AtomisticTrainingPolicy(maximum_steps=2), key=jr.key(5)
    )
    continued = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=5),
        key=jr.key(999),
        continuation=first,
    )
    uninterrupted = fit_atomistic_potential(
        initial, problem, AtomisticTrainingPolicy(maximum_steps=5), key=jr.key(5)
    )

    _assert_trees_bitwise_equal(continued, uninterrupted)
    assert first.progress.update_step == 2
    assert continued.progress.update_step == 5
    assert continued.progress == uninterrupted.progress
    for name in ("potential", "best_potential"):
        assert atomistic_model_revision(getattr(continued, name)).revision_id == (
            atomistic_model_revision(getattr(uninterrupted, name)).revision_id
        )
    assert continued.problem_id == uninterrupted.problem_id
    assert continued.policy_id == uninterrupted.policy_id
    assert continued.continuation_id == uninterrupted.continuation_id
    assert continued.result_id == uninterrupted.result_id


def test_training_and_rmd17_scenario_2() -> None:
    batch = _batch()
    energy, _ = _targets(batch)
    # An available nonfinite label is never treated as missing: the first
    # window is rejected and the result stays at the initial boundary.
    # The scale is declared: a fitted scale refuses nonfinite training labels.
    problem = _problem(
        batch,
        energy=energy.at[1].set(jnp.nan),
        validation=_held_out(),
        energy_scale=1.0,
    )
    result = fit_atomistic_potential(
        _potential(jr.key(6)), problem, AtomisticTrainingPolicy(maximum_steps=4)
    )
    assert int(result.status) == int(AtomisticStatus.NONFINITE)
    assert not bool(result.successful)
    assert result.termination == "nonfinite_update_window"
    assert result.training_values.shape == (0,)
    assert result.cursor.cursor_id == problem.training.start(0).cursor_id
    np.testing.assert_array_equal(result.validation_steps, [0])
    events = []
    session = IterationSession(
        "atomistic-training-test",
        sinks=(
            CallableIterationSink(
                lambda event: events.append(
                    TrainingIterationKind(int(event.record.metrics.kind))
                ),
                "collector",
            ),
        ),
    )

    fit_atomistic_potential(
        _potential(jr.key(8)),
        _problem(batch, energy=energy),
        AtomisticTrainingPolicy(maximum_steps=1),
        session=session,
    )
    assert events == [
        TrainingIterationKind.RUN_START,
        TrainingIterationKind.VALIDATION,
        TrainingIterationKind.EPOCH_START,
        TrainingIterationKind.UPDATE,
        TrainingIterationKind.VALIDATION,
        TrainingIterationKind.RUN_TERMINAL,
    ]
    first_problem = _problem(
        batch,
        energy=energy,
        validation=_held_out(energy_available=np.asarray([True, True, False])),
    )
    changed_problem = _problem(
        batch,
        energy=energy,
        validation=_held_out(energy_available=np.asarray([True, False, True])),
    )
    assert first_problem.problem_id != changed_problem.problem_id
    first = fit_atomistic_potential(
        _potential(jr.key(41)), first_problem, AtomisticTrainingPolicy(maximum_steps=0)
    )
    with pytest.raises(ValueError, match="different training problem"):
        fit_atomistic_potential(
            _potential(jr.key(41)),
            changed_problem,
            AtomisticTrainingPolicy(maximum_steps=1),
            continuation=first,
        )


def test_training_and_rmd17_scenario_3() -> None:
    batch = _batch()
    energy, forces = _targets(batch)
    energy = energy.at[1].set(jnp.nan)
    # Force availability is per atom; an unavailable atom's label is never read.
    force_available = np.ones((3, 2), dtype=np.bool_)
    force_available[2, 1] = False
    forces = forces.at[2, 1, 2].set(jnp.nan)
    problem = _problem(
        batch,
        energy=energy,
        forces=forces,
        energy_available=np.asarray([True, False, True]),
        force_available=force_available,
    )
    result = fit_atomistic_potential(
        _potential(jr.key(42)), problem, AtomisticTrainingPolicy(maximum_steps=1)
    )
    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert bool(jnp.all(jnp.isfinite(result.training_values)))
    batch = _batch()
    energy, _ = _targets(batch)
    result = fit_atomistic_potential(
        _potential(jr.key(43)),
        _problem(batch, energy=energy, execution=_execution(0)),
        AtomisticTrainingPolicy(maximum_steps=2),
    )
    # Without a validation loader the initial training-epoch evaluation already
    # meets the exhausted neighbor capacity.
    assert int(result.status) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert result.termination == "failed_validation_evaluation"
    assert result.training_values.shape == (0,)
    problem = _problem(batch, energy=energy)
    continuation = fit_atomistic_potential(
        _potential(jr.key(405)), problem, AtomisticTrainingPolicy(maximum_steps=0)
    )
    changed = PaiNNPotential(
        SCALE,
        cutoff=1.5,
        feature_count=6,
        interaction_count=1,
        radial_basis_count=4,
        key=jr.key(405),
    )
    with pytest.raises(ValueError, match="configuration"):
        fit_atomistic_potential(
            changed,
            problem,
            AtomisticTrainingPolicy(maximum_steps=1),
            continuation=continuation,
        )
    initial = _potential(jr.key(44))
    result = fit_atomistic_potential(
        initial, _problem(batch, energy=energy), AtomisticTrainingPolicy(maximum_steps=1)
    )
    assert int(result.validation_steps[0]) == 0
    assert result.progress.best_step in (0, 1)
    assert float(result.best_loss) <= float(result.validation_values[0])
    assert atomistic_model_revision(result.potential).revision_id != (
        atomistic_model_revision(initial).revision_id
    )


def test_local_rmd17_parser_and_split_are_explicit_disjoint_and_reproducible(
    tmp_path: Any,
) -> None:
    sample_count = 12
    atom_count = 3
    path = tmp_path / "rmd17_synthetic_layout.npz"
    coordinates = np.arange(sample_count * atom_count * 3, dtype="float64").reshape(
        sample_count, atom_count, 3
    )
    np.savez(
        path,
        nuclear_charges=np.asarray([8, 1, 1]),
        coords=coordinates,
        energies=np.arange(sample_count, dtype="float64"),
        forces=np.ones_like(coordinates),
        old_indices=np.arange(100, 100 + sample_count),
    )
    dataset = load_rmd17_npz(path)
    assert dataset.scale.length_unit == ANGSTROM
    assert dataset.scale.energy_unit == ELECTRONVOLT
    assert dataset.source_length_unit == ANGSTROM
    assert dataset.source_energy_unit == KILOCALORIE_PER_MOLE
    assert dataset.source_mass_unit == DALTON
    assert dataset.avogadro_constant_set_id == "codata-2018"
    np.testing.assert_allclose(dataset.energies[1], 0.04336410424180094)
    assert dataset.sample_count == sample_count
    split = split_rmd17(dataset, train_size=5, validation_size=3, test_size=4, seed=17)
    repeated = split_rmd17(dataset, train_size=5, validation_size=3, test_size=4, seed=17)
    assert split.split_id == repeated.split_id
    train = set(np.asarray(split.train_indices).tolist())
    validation = set(np.asarray(split.validation_indices).tolist())
    test = set(np.asarray(split.test_indices).tolist())
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    batch, energy, forces = dataset.take(split.train_indices)
    assert batch.positions.shape == (5, atom_count, 3)
    assert energy.shape == (5,)
    assert forces.shape == batch.positions.shape


def test_rmd17_parser_rejects_nonfinite_or_unsupported_local_data(tmp_path: Any) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        nuclear_charges=np.asarray([16]),
        coords=np.zeros((2, 1, 3)),
        energies=np.asarray([0.0, np.nan]),
        forces=np.zeros((2, 1, 3)),
    )
    with pytest.raises(ValueError, match="finite"):
        load_rmd17_npz(path)


def test_nequip_trains_through_existing_contract_on_synthetic_rmd17(
    tmp_path: Any,
) -> None:
    sample_count = 5
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.65 + 0.1 * index, 0.0, 0.0]]
            for index in range(sample_count)
        ]
    )
    path = tmp_path / "rmd17_synthetic_nequip.npz"
    np.savez(
        path,
        nuclear_charges=np.asarray([1, 1]),
        coords=coordinates,
        energies=np.zeros((sample_count,)),
        forces=np.zeros_like(coordinates),
    )
    dataset = load_rmd17_npz(path, scale=AtomisticScaleContract(ANGSTROM, ELECTRONVOLT))
    batch, _, _ = dataset.take(np.arange(sample_count))

    def potential(key: Any) -> Any:
        return NequIPPotential(
            dataset.scale,
            cutoff=2.0,
            feature_count=2,
            interaction_count=1,
            radial_basis_count=3,
            key=key,
        )

    teacher = energy_and_forces(potential(jr.key(301)), batch, _execution())
    result = fit_atomistic_potential(
        potential(jr.key(302)),
        AtomisticTrainingProblem(
            atomistic_training_loader(
                atomistic_training_source(
                    batch,
                    mass_unit=dataset.source_mass_unit,
                    energy=teacher.energy,
                    forces=teacher.forces,
                ),
                batch_size=2,
            ),
            _execution(),
            (AtomisticEnergyObjective(), AtomisticForceObjective()),
            scale=dataset.scale,
            cutoff=CUTOFF,
        ),
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-3),
        key=jr.key(303),
    )
    assert isinstance(result.potential, NequIPPotential)
    assert result.training_values.shape == (2,)
    prediction = energy_and_forces(result.best_potential, batch, _execution())
    assert bool(jnp.all(prediction.valid))
    assert prediction.energy.shape == (sample_count,)


def test_training_rejects_cross_family_continuation() -> None:
    for continuation_family in ["painn", "nequip"]:
        batch = _batch()
        energy, _ = _targets(batch)
        problem = _problem(batch, energy=energy)

        def nequip(key: Any) -> Any:
            return NequIPPotential(
                SCALE,
                cutoff=2.0,
                feature_count=2,
                interaction_count=1,
                radial_basis_count=3,
                key=key,
            )

        if continuation_family == "painn":
            resumed = _potential(jr.key(401))
            supplied = nequip(jr.key(402))
        else:
            resumed = nequip(jr.key(403))
            supplied = _potential(jr.key(404))
        continuation = fit_atomistic_potential(
            resumed, problem, AtomisticTrainingPolicy(maximum_steps=0)
        )
        with pytest.raises(ValueError, match="same concrete family"):
            fit_atomistic_potential(
                supplied,
                problem,
                AtomisticTrainingPolicy(maximum_steps=1),
                continuation=continuation,
            )
