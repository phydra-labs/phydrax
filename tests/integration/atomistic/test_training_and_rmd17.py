import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticScaleContract,
    AtomisticStatus,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    energy_and_forces,
    fit_atomistic_potential,
    load_rmd17_npz,
    split_rmd17,
)
from phydrax.nn.atomistic import PaiNNPotential


SCALE = AtomisticScaleContract("angstrom", "electronvolt")


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


def _potential(key, *, maximum_neighbors=1):
    return PaiNNPotential(
        SCALE,
        cutoff=2.0,
        maximum_neighbors=maximum_neighbors,
        maximum_dense_atoms=2,
        feature_count=6,
        interaction_count=1,
        radial_basis_count=4,
        key=key,
    )


def _targets(batch):
    teacher = _potential(jr.key(91))
    prediction = energy_and_forces(teacher, batch)
    return prediction.energy, prediction.forces


@pytest.mark.parametrize("target_kind", ["energy", "force", "joint"])
def test_energy_force_and_joint_training_have_complete_decreasing_histories(target_kind):
    batch = _batch()
    energy, forces = _targets(batch)
    problem = AtomisticTrainingProblem(
        batch,
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
    assert float(result.training_loss_history[-1]) < float(result.training_loss_history[0])
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
        training_energy=energy,
        validation_batch=validation,
        validation_energy=jnp.asarray([1e3, -1e3, 2e3]),
    )
    problem_b = AtomisticTrainingProblem(
        batch,
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
    problem = AtomisticTrainingProblem(batch, training_energy=energy)
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
    np.testing.assert_allclose(
        continued.training_loss_history,
        uninterrupted.training_loss_history,
        rtol=1e-12,
        atol=1e-12,
    )
    continued_leaves = jax.tree_util.tree_leaves(continued.potential)
    uninterrupted_leaves = jax.tree_util.tree_leaves(uninterrupted.potential)
    for observed, expected in zip(continued_leaves, uninterrupted_leaves, strict=True):
        if isinstance(observed, jax.Array) and jnp.issubdtype(observed.dtype, jnp.inexact):
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert continued.progress.best_step <= continued.progress.update_step
    assert continued.best_potential is not None


def test_nonfinite_supervision_terminates_with_typed_status():
    batch = _batch()
    energy, _ = _targets(batch)
    problem = AtomisticTrainingProblem(
        batch, training_energy=energy.at[1].set(jnp.nan)
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
        AtomisticTrainingProblem(batch, training_energy=energy),
        AtomisticTrainingPolicy(maximum_steps=1, force_weight=0.0),
        callbacks=(callback,),
    )
    assert events == ["start", "validation", "update", "validation", "stop"]


def test_validation_masks_are_part_of_continuation_identity():
    batch = _batch()
    energy, _ = _targets(batch)
    first_problem = AtomisticTrainingProblem(
        batch,
        training_energy=energy,
        validation_batch=batch,
        validation_energy=energy,
        validation_energy_mask=[True, True, False],
    )
    changed_problem = AtomisticTrainingProblem(
        batch,
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
        _potential(jr.key(43), maximum_neighbors=0),
        AtomisticTrainingProblem(batch, training_energy=energy),
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
        AtomisticTrainingProblem(batch, training_energy=energy),
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
    assert dataset.scale.length_unit == "angstrom"
    assert dataset.scale.energy_unit == "kilocalorie_per_mole"
    assert dataset.sample_count == sample_count
    split = split_rmd17(
        dataset, train_size=5, validation_size=3, test_size=4, seed=17
    )
    repeated = split_rmd17(
        dataset, train_size=5, validation_size=3, test_size=4, seed=17
    )
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
