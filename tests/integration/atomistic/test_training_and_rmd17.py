import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax.atomistic._training as atomistic_training
from phydrax.atomistic import (
    AtomisticBatch,
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
from phydrax.nn.atomistic import NequIPPotential, PaiNNPotential
from phydrax.units import (
    ANGSTROM,
    DALTON,
    ELECTRONVOLT,
    KILOCALORIE_PER_MOLE,
)


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _execution(maximum_neighbors=3):
    return AtomisticGraphExecutionPlan(
        maximum_neighbors,
        maximum_dense_atoms=256,
    )


def _batch():
    return AtomisticBatch(
        [[1, 1], [1, 1], [1, 1]],
        [
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]],
        ],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        SCALE,
    )


def _potential(key):
    return PaiNNPotential(
        SCALE,
        cutoff=2.0,
        feature_count=6,
        interaction_count=1,
        radial_basis_count=4,
        key=key,
    )


def _targets(batch):
    teacher = _potential(jr.key(91))
    prediction = energy_and_forces(teacher, batch, _execution())
    return prediction.energy, prediction.forces


def _assert_trees_bitwise_equal(observed, expected):
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


def _assert_optimizer_state_matches_checkpoint(result):
    def is_potential(node):
        return isinstance(node, atomistic_training.AbstractAtomisticPotential)

    optimizer_leaves = jax.tree_util.tree_leaves(
        result.optimizer_state, is_leaf=is_potential
    )
    optimizer_potentials = tuple(leaf for leaf in optimizer_leaves if is_potential(leaf))
    assert optimizer_potentials
    trainable, _ = atomistic_training.partition_trainable(result.potential)
    expected_structure = jax.tree_util.tree_structure(trainable)
    for optimizer_potential in optimizer_potentials:
        assert jax.tree_util.tree_structure(optimizer_potential) == expected_structure
        assert (
            optimizer_potential.parameter_state_id == result.potential.parameter_state_id
        )
        assert optimizer_potential.potential_id == result.potential.potential_id


@pytest.mark.parametrize("target_kind", ["energy", "force", "joint"])
def test_energy_force_and_joint_training_have_complete_decreasing_histories(target_kind):
    batch = _batch()
    energy, forces = _targets(batch)
    problem = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy if target_kind != "force" else None,
        training_forces=forces if target_kind != "energy" else None,
    )
    policy = AtomisticTrainingPolicy(
        maximum_steps=8,
        learning_rate=2e-3,
        energy_weight=1.0 if target_kind != "force" else 0.0,
        force_weight=1.0 if target_kind != "energy" else 0.0,
    )
    result = fit_atomistic_potential(
        _potential(jr.key(3)), problem, policy, key=jr.key(12)
    )
    assert result.training_loss_history.shape == (8,)
    assert result.energy_loss_history.shape == (8,)
    assert result.force_loss_history.shape == (8,)
    assert float(result.training_loss_history[-1]) < float(
        result.training_loss_history[0]
    )
    assert bool(result.successful)
    if target_kind == "energy":
        np.testing.assert_allclose(result.force_loss_history, 0.0)
    if target_kind == "force":
        np.testing.assert_allclose(result.energy_loss_history, 0.0)


def test_normalization_is_fitted_only_from_training_targets():
    batch = _batch()
    energy, _ = _targets(batch)
    validation = batch.with_positions(batch.positions + 0.2)
    problem_a = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy,
        validation_batch=validation,
        validation_energy=jnp.asarray([1e3, -1e3, 2e3]),
    )
    problem_b = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy,
        validation_batch=validation,
        validation_energy=jnp.asarray([-4e8, 7e8, 9e8]),
    )
    policy = AtomisticTrainingPolicy(maximum_steps=0, force_weight=0.0)
    first = fit_atomistic_potential(_potential(jr.key(2)), problem_a, policy)
    second = fit_atomistic_potential(_potential(jr.key(2)), problem_b, policy)
    np.testing.assert_allclose(
        first.normalization.energy_per_atom_scale,
        second.normalization.energy_per_atom_scale,
    )
    assert first.normalization.fitted_from_problem_id == problem_a.problem_id


def test_deterministic_continuation_matches_uninterrupted_training_and_selection():
    batch = _batch()
    energy, _ = _targets(batch)
    problem = AtomisticTrainingProblem(batch, _execution(), training_energy=energy)
    initial = _potential(jr.key(4))
    first = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, force_weight=0.0),
        key=jr.key(5),
    )
    continued = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=5, force_weight=0.0),
        key=jr.key(999),
        continuation=first,
    )
    uninterrupted = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=5, force_weight=0.0),
        key=jr.key(5),
    )

    _assert_trees_bitwise_equal(continued, uninterrupted)
    _assert_optimizer_state_matches_checkpoint(first)
    _assert_optimizer_state_matches_checkpoint(continued)
    assert first.progress.update_step == 2
    assert continued.progress.update_step == 5
    assert continued.progress == uninterrupted.progress
    assert continued.potential.parameter_state_id == (
        uninterrupted.potential.parameter_state_id
    )
    assert continued.potential.potential_id == uninterrupted.potential.potential_id
    assert continued.best_potential.parameter_state_id == (
        uninterrupted.best_potential.parameter_state_id
    )
    assert (
        continued.best_potential.potential_id == uninterrupted.best_potential.potential_id
    )
    assert continued.problem_id == uninterrupted.problem_id
    assert continued.policy_id == uninterrupted.policy_id
    assert continued.continuation_id == uninterrupted.continuation_id
    assert continued.result_id == uninterrupted.result_id


def test_nonfinite_supervision_terminates_with_typed_status():
    batch = _batch()
    energy, _ = _targets(batch)
    problem = AtomisticTrainingProblem(
        batch, _execution(), training_energy=energy.at[1].set(jnp.nan)
    )
    result = fit_atomistic_potential(
        _potential(jr.key(6)),
        problem,
        AtomisticTrainingPolicy(maximum_steps=4, force_weight=0.0),
    )
    assert int(result.status) == int(AtomisticStatus.NONFINITE)
    assert not bool(result.successful)
    assert "nonfinite" in result.termination
    assert result.training_loss_history.shape == (0,)
    np.testing.assert_array_equal(result.validation_steps, [0])


def test_callbacks_receive_start_update_validation_and_stop_events():
    batch = _batch()
    energy, _ = _targets(batch)
    events = []

    def callback(event):
        events.append(event.name)
        return False

    fit_atomistic_potential(
        _potential(jr.key(8)),
        AtomisticTrainingProblem(batch, _execution(), training_energy=energy),
        AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
        callbacks=(callback,),
    )
    assert events == ["start", "validation", "update", "validation", "stop"]


def test_validation_masks_are_part_of_continuation_identity():
    batch = _batch()
    energy, _ = _targets(batch)
    first_problem = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy,
        validation_batch=batch,
        validation_energy=energy,
        validation_energy_mask=[True, True, False],
    )
    changed_problem = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy,
        validation_batch=batch,
        validation_energy=energy,
        validation_energy_mask=[True, False, True],
    )
    assert first_problem.problem_id != changed_problem.problem_id
    first = fit_atomistic_potential(
        _potential(jr.key(41)),
        first_problem,
        AtomisticTrainingPolicy(maximum_steps=0, force_weight=0.0),
    )
    with pytest.raises(ValueError, match="different training problem"):
        fit_atomistic_potential(
            _potential(jr.key(41)),
            changed_problem,
            AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
            continuation=first,
        )


def test_masked_nonfinite_targets_are_inert_before_residual_squaring():
    batch = _batch()
    energy, forces = _targets(batch)
    energy = energy.at[1].set(jnp.nan)
    force_mask = jnp.ones_like(forces, dtype=bool).at[2, 1, 2].set(False)
    forces = forces.at[2, 1, 2].set(jnp.nan)
    problem = AtomisticTrainingProblem(
        batch,
        _execution(),
        training_energy=energy,
        training_forces=forces,
        training_energy_mask=[True, False, True],
        training_force_mask=force_mask,
    )
    result = fit_atomistic_potential(
        _potential(jr.key(42)),
        problem,
        AtomisticTrainingPolicy(maximum_steps=1),
    )
    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert bool(jnp.all(jnp.isfinite(result.training_loss_history)))


def test_training_reports_neighbor_overflow_without_nonfinite_conflation():
    batch = _batch()
    energy, _ = _targets(batch)
    result = fit_atomistic_potential(
        _potential(jr.key(43)),
        AtomisticTrainingProblem(batch, _execution(0), training_energy=energy),
        AtomisticTrainingPolicy(maximum_steps=2, force_weight=0.0),
    )
    assert int(result.status) == int(AtomisticStatus.NEIGHBOR_OVERFLOW)
    assert "neighbor_overflow" in result.termination
    assert result.training_loss_history.shape == (0,)


def test_initial_model_is_selected_at_step_zero_and_trained_state_is_refingerprinted():
    batch = _batch()
    energy, _ = _targets(batch)
    initial = _potential(jr.key(44))
    initial_parameter_state = initial.parameter_state_id
    result = fit_atomistic_potential(
        initial,
        AtomisticTrainingProblem(batch, _execution(), training_energy=energy),
        AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
    )
    assert int(result.validation_steps[0]) == 0
    assert result.progress.best_step in (0, 1)
    assert float(result.best_loss) <= float(result.validation_loss_history[0])
    assert result.potential.parameter_state_id != initial_parameter_state
    assert result.potential.potential_id != initial.potential_id


def test_local_rmd17_parser_and_split_are_explicit_disjoint_and_reproducible(tmp_path):
    sample_count = 12
    atom_count = 3
    path = tmp_path / "rmd17_synthetic_layout.npz"
    coordinates = np.arange(sample_count * atom_count * 3, dtype=float).reshape(
        sample_count, atom_count, 3
    )
    np.savez(
        path,
        nuclear_charges=np.asarray([8, 1, 1]),
        coords=coordinates,
        energies=np.arange(sample_count, dtype=float),
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


def test_rmd17_parser_rejects_nonfinite_or_unsupported_local_data(tmp_path):
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


def test_nequip_trains_through_existing_contract_on_synthetic_rmd17(tmp_path):
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

    def potential(key):
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
            batch,
            _execution(),
            training_energy=teacher.energy,
            training_forces=teacher.forces,
        ),
        AtomisticTrainingPolicy(
            maximum_steps=2,
            learning_rate=1e-3,
            energy_weight=1.0,
            force_weight=1.0,
        ),
        key=jr.key(303),
    )
    assert isinstance(result.final_potential, NequIPPotential)
    assert result.training_loss_history.shape == (2,)
    prediction = energy_and_forces(result.best_potential, batch, _execution())
    assert bool(jnp.all(prediction.valid))
    assert prediction.energy.shape == (sample_count,)


@pytest.mark.parametrize("continuation_family", ["painn", "nequip"])
def test_training_rejects_cross_family_continuation(continuation_family):
    batch = _batch()
    energy, _ = _targets(batch)
    problem = AtomisticTrainingProblem(batch, _execution(), training_energy=energy)

    def nequip(key):
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
        resumed,
        problem,
        AtomisticTrainingPolicy(maximum_steps=0, force_weight=0.0),
    )
    with pytest.raises(ValueError, match="same concrete family"):
        fit_atomistic_potential(
            supplied,
            problem,
            AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
            continuation=continuation,
        )


def test_training_rejects_same_family_continuation_with_changed_configuration():
    batch = _batch()
    energy, _ = _targets(batch)
    problem = AtomisticTrainingProblem(batch, _execution(), training_energy=energy)
    continuation = fit_atomistic_potential(
        _potential(jr.key(405)),
        problem,
        AtomisticTrainingPolicy(maximum_steps=0, force_weight=0.0),
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
            AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
            continuation=continuation,
        )


def test_training_hashes_parameter_state_only_for_returned_checkpoints(monkeypatch):
    batch = _batch()
    energy, _ = _targets(batch)
    calls = 0
    checkpoint = atomistic_training.checkpoint_atomistic_potential

    def counted_checkpoint(potential):
        nonlocal calls
        calls += 1
        return checkpoint(potential)

    monkeypatch.setattr(
        atomistic_training, "checkpoint_atomistic_potential", counted_checkpoint
    )
    result = fit_atomistic_potential(
        _potential(jr.key(406)),
        AtomisticTrainingProblem(
            batch,
            _execution(),
            training_energy=energy,
            validation_batch=batch,
            validation_energy=energy,
        ),
        AtomisticTrainingPolicy(
            maximum_steps=3,
            validation_every=1,
            force_weight=0.0,
        ),
    )
    assert int(result.progress.update_step) == 3
    assert calls == 2
