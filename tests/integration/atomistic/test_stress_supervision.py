#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.atomistic import (
    atomistic_potential_revision,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticStatus,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    energy_and_forces,
    fit_atomistic_potential,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic import MACEArchitecture, MACEPotential, PaiNNPotential
from phydrax.precision import ScalarPrecisionDType
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])


def _mace(key: Any) -> Any:
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=CUTOFF,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=4,
            hidden_degree=1,
            edge_degree=2,
            interactions=("real-agnostic", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(8,),
            readout_width=4,
            average_neighbor_count=2.0,
        ),
        # ty: ignore[invalid-argument-type]
        atomic_energies=[[-1.0, -2.0]],
        key=key,
    )


def _execution() -> Any:
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=125,
        ),
    )


def _periodic_batch(shift: float = 0.0, *, periodic: Any = (True, True, True)) -> Any:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]],
            [[0.1, 0.0, 0.0], [1.3, 0.5, -0.1]],
            [[0.0, 0.2, 0.1], [0.9, 1.2, 0.4]],
        ]
    )
    return AtomisticBatch(
        # ty: ignore[invalid-argument-type]
        [[1, 8], [1, 8], [1, 8]],
        positions + shift,
        # ty: ignore[invalid-argument-type]
        [[1.0, 16.0], [1.0, 16.0], [1.0, 16.0]],
        SCALE,
        cells=np.broadcast_to(CELL, (3, 3, 3)),
        periodic_axes=np.broadcast_to(np.asarray(periodic, dtype=np.bool_), (3, 3)),
    )


def _teacher_labels(batch: Any) -> Any:
    prediction = energy_and_forces(
        _mace(jr.key(91)), batch, _execution(), compute_stress=True
    )
    assert bool(jnp.all(prediction.valid))
    return prediction.energy, prediction.forces, prediction.stress


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


def test_mace_energy_force_stress_fit_continues_like_an_uninterrupted_run() -> None:
    batch = _periodic_batch()
    energy, forces, stress = _teacher_labels(batch)
    problem = AtomisticTrainingProblem(
        batch,
        _execution(),
        cutoff=CUTOFF,
        training_energy=energy,
        training_forces=forces,
        training_stress=stress,
    )

    def policy(steps: int) -> Any:
        return AtomisticTrainingPolicy(maximum_steps=steps, learning_rate=5e-3)

    initial = _mace(jr.key(3))
    first = fit_atomistic_potential(initial, problem, policy(2), key=jr.key(5))
    continued = fit_atomistic_potential(
        initial, problem, policy(4), key=jr.key(999), continuation=first
    )
    uninterrupted = fit_atomistic_potential(initial, problem, policy(4), key=jr.key(5))

    _assert_trees_bitwise_equal(continued, uninterrupted)
    assert continued.result_id == uninterrupted.result_id
    assert int(continued.status) == int(AtomisticStatus.SUCCESS)
    assert continued.progress.update_step == 4
    for history in (
        continued.energy_loss_history,
        continued.force_loss_history,
        continued.stress_loss_history,
    ):
        assert history.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(history) & (history > 0.0)))
    assert float(continued.training_loss_history[-1]) < float(
        continued.validation_loss_history[0]
    )
    assert atomistic_potential_revision(continued.potential).revision_id != (
        atomistic_potential_revision(initial).revision_id
    )
    fitted = energy_and_forces(
        continued.potential, batch, _execution(), compute_stress=True
    )
    assert bool(jnp.all(fitted.valid))


def test_stress_only_supervision_skips_masked_cases_and_moves_the_model() -> None:
    batch = _periodic_batch()
    _, _, stress = _teacher_labels(batch)
    stress_mask = np.ones((3, 3, 3), dtype=np.bool_)
    stress_mask[1] = False
    stress_mask[2, 0, 1] = False
    labels = stress.at[1].set(jnp.nan).at[2, 0, 1].set(jnp.inf)
    initial = _mace(jr.key(7))
    result = fit_atomistic_potential(
        initial,
        AtomisticTrainingProblem(
            batch,
            _execution(),
            cutoff=CUTOFF,
            training_stress=labels,
            training_stress_mask=stress_mask,
        ),
        AtomisticTrainingPolicy(maximum_steps=3, learning_rate=1e-2),
        key=jr.key(8),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    np.testing.assert_array_equal(result.energy_loss_history, np.zeros((3,)))
    np.testing.assert_array_equal(result.force_loss_history, np.zeros((3,)))
    assert bool(jnp.all(jnp.isfinite(result.stress_loss_history)))
    assert float(result.stress_loss_history[-1]) < float(
        result.validation_loss_history[0]
    )
    selected = np.asarray(stress)[stress_mask]
    np.testing.assert_allclose(
        result.normalization.stress_component_scale,
        np.sqrt(np.mean(selected * selected)),
    )
    assert atomistic_potential_revision(result.potential).revision_id != (
        atomistic_potential_revision(initial).revision_id
    )


def test_mixed_finite_and_periodic_cases_train_stress_only_where_labeled() -> None:
    periodic = _periodic_batch()
    cells = np.broadcast_to(CELL, (3, 3, 3)).copy()
    cells[1] = 0.0
    mixed = AtomisticBatch(
        periodic.atomic_numbers,
        periodic.positions,
        periodic.masses,
        SCALE,
        cells=cells,
        periodic_axes=np.asarray(
            [[True, True, True], [False, False, False], [True, True, True]]
        ),
    )
    stress_cases = np.asarray([True, False, True])
    teacher = energy_and_forces(
        _mace(jr.key(91)),
        mixed,
        _execution(),
        compute_stress=True,
        stress_case_mask=stress_cases,
    )
    assert bool(jnp.all(teacher.valid))
    assert teacher.stress is not None
    assert bool(jnp.all(jnp.isnan(teacher.stress[1])))
    problem = AtomisticTrainingProblem(
        mixed,
        _execution(),
        cutoff=CUTOFF,
        training_energy=teacher.energy,
        training_forces=teacher.forces,
        training_stress=teacher.stress,
    )
    assert problem.training.stress_mask is not None
    np.testing.assert_array_equal(
        np.asarray(problem.training.stress_mask)[:, 0, 0], stress_cases
    )
    initial = _mace(jr.key(15))
    result = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=5e-3),
        key=jr.key(16),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    for history in (
        result.energy_loss_history,
        result.force_loss_history,
        result.stress_loss_history,
    ):
        assert bool(jnp.all(jnp.isfinite(history) & (history > 0.0)))
    assert float(result.training_loss_history[-1]) < float(
        result.validation_loss_history[0]
    )
    assert atomistic_potential_revision(result.potential).revision_id != (
        atomistic_potential_revision(initial).revision_id
    )


def test_stress_normalization_is_fitted_from_training_labels_only() -> None:
    batch = _periodic_batch()
    validation_batch = _periodic_batch(0.05)
    _, _, stress = _teacher_labels(batch)

    def fit(validation_stress: Any) -> Any:
        problem = AtomisticTrainingProblem(
            batch,
            _execution(),
            cutoff=CUTOFF,
            training_stress=stress,
            validation_batch=validation_batch,
            validation_stress=validation_stress,
        )
        return problem, fit_atomistic_potential(
            _mace(jr.key(2)), problem, AtomisticTrainingPolicy(maximum_steps=0)
        )

    first_problem, first = fit(jnp.full((3, 3, 3), 1e3))
    second_problem, second = fit(jnp.full((3, 3, 3), -4e6))

    np.testing.assert_array_equal(
        first.normalization.stress_component_scale,
        second.normalization.stress_component_scale,
    )
    np.testing.assert_allclose(
        first.normalization.stress_component_scale,
        np.sqrt(np.mean(np.asarray(stress) ** 2)),
    )
    assert first.normalization.fitted_from_problem_id == first_problem.problem_id
    assert first_problem.problem_id != second_problem.problem_id
    assert float(first.validation_loss_history[0]) != float(
        second.validation_loss_history[0]
    )


def test_partial_pbc_stress_trains_against_the_declared_embedding_cell() -> None:
    slab = _periodic_batch(periodic=(True, True, False))
    _, _, stress = _teacher_labels(slab)
    problem = AtomisticTrainingProblem(
        slab, _execution(), cutoff=CUTOFF, training_stress=stress
    )
    assert problem.training.stress_mask is not None
    assert bool(jnp.all(problem.training.stress_mask))
    initial = _mace(jr.key(13))
    result = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-2),
        key=jr.key(14),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert bool(jnp.all(jnp.isfinite(result.stress_loss_history)))
    assert float(result.stress_loss_history[-1]) < float(
        result.validation_loss_history[0]
    )


def test_stress_supervision_refuses_cases_without_a_declared_cell_volume() -> None:
    batch = _periodic_batch()
    _, forces, stress = _teacher_labels(batch)
    finite = AtomisticBatch(
        batch.atomic_numbers,
        batch.positions,
        batch.masses,
        SCALE,
    )
    with pytest.raises(ValueError, match="finite nonsingular embedding cell"):
        AtomisticTrainingProblem(
            finite, _execution(), cutoff=CUTOFF, training_stress=stress
        )
    unperiodic = _periodic_batch(periodic=(False, False, False))
    with pytest.raises(ValueError, match="finite nonsingular embedding cell"):
        AtomisticTrainingProblem(
            unperiodic, _execution(), cutoff=CUTOFF, training_stress=stress
        )
    singular = np.broadcast_to(CELL, (3, 3, 3)).copy()
    singular[1, 2] = 0.0
    mixed = AtomisticBatch(
        batch.atomic_numbers,
        batch.positions,
        batch.masses,
        SCALE,
        cells=singular,
        periodic_axes=np.broadcast_to(np.asarray([True, True, False]), (3, 3)),
    )
    with pytest.raises(ValueError, match="finite nonsingular embedding cell"):
        AtomisticTrainingProblem(
            mixed,
            _execution(),
            cutoff=CUTOFF,
            training_forces=forces,
            training_stress=stress,
            training_stress_mask=np.ones((3, 3, 3), dtype=np.bool_),
        )
    defaulted = AtomisticTrainingProblem(
        mixed, _execution(), cutoff=CUTOFF, training_stress=stress
    )
    assert defaulted.training.stress_mask is not None
    np.testing.assert_array_equal(
        np.asarray(defaulted.training.stress_mask)[:, 0, 0], [True, False, True]
    )


def test_stress_supervision_refuses_mismatched_target_kinds() -> None:
    batch = _periodic_batch()
    _, forces, stress = _teacher_labels(batch)
    with pytest.raises(ValueError, match="same energy/force/stress target kinds"):
        AtomisticTrainingProblem(
            batch,
            _execution(),
            cutoff=CUTOFF,
            training_stress=stress,
            validation_batch=batch,
            validation_forces=forces,
        )


def test_stress_supervision_refuses_zero_total_target_weight() -> None:
    batch = _periodic_batch()
    _, _, stress = _teacher_labels(batch)
    stress_problem = AtomisticTrainingProblem(
        batch, _execution(), cutoff=CUTOFF, training_stress=stress
    )
    with pytest.raises(ValueError, match="zero weight"):
        fit_atomistic_potential(
            _mace(jr.key(4)),
            stress_problem,
            AtomisticTrainingPolicy(maximum_steps=1, stress_weight=0.0),
        )


def test_nonfinite_update_rolls_back_to_the_last_accepted_potential() -> None:
    batch = _periodic_batch()
    energy, forces, stress = _teacher_labels(batch)
    initial = _mace(jr.key(11))
    result = fit_atomistic_potential(
        initial,
        AtomisticTrainingProblem(
            batch,
            _execution(),
            cutoff=CUTOFF,
            training_energy=energy,
            training_forces=forces,
            training_stress=stress,
        ),
        AtomisticTrainingPolicy(maximum_steps=3, learning_rate=1e300),
        key=jr.key(12),
    )

    assert int(result.status) == int(AtomisticStatus.NONFINITE)
    assert not bool(result.successful)
    assert result.termination in (
        "nonfinite_training_loss_or_gradient",
        "nonfinite_updated_loss",
    )
    assert result.progress.update_step == 0
    assert bool(jnp.all(jnp.isnan(result.stress_loss_history)))
    assert result.stress_loss_history.shape == (1,)
    assert atomistic_potential_revision(result.potential).revision_id == (
        atomistic_potential_revision(initial).revision_id
    )
    _assert_trees_bitwise_equal(result.best_potential, result.potential)


def test_painn_force_training_agrees_on_dense_and_sparse_finite_topologies() -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.2]],
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [9.0, 9.0, 9.0]],
        ]
    )
    batch = AtomisticBatch(
        # ty: ignore[invalid-argument-type]
        [[1, 1, 1], [1, 1, 0]],
        positions,
        # ty: ignore[invalid-argument-type]
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        SCALE,
        # ty: ignore[invalid-argument-type]
        atom_mask=[[True, True, True], [True, True, False]],
    )

    def painn(key: Any) -> Any:
        return PaiNNPotential(
            SCALE,
            cutoff=2.0,
            feature_count=4,
            interaction_count=1,
            radial_basis_count=3,
            key=key,
        )

    dense = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=8)
    sparse = AtomisticGraphExecutionPlan(
        4,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=4,
            maximum_edges=64,
            maximum_degree=4,
            maximum_images=1,
        ),
    )
    teacher = energy_and_forces(painn(jr.key(21)), batch, dense)
    results = tuple(
        fit_atomistic_potential(
            painn(jr.key(22)),
            AtomisticTrainingProblem(
                batch,
                execution,
                cutoff=2.0,
                training_energy=teacher.energy,
                training_forces=teacher.forces,
            ),
            AtomisticTrainingPolicy(maximum_steps=3, learning_rate=2e-3),
            key=jr.key(23),
        )
        for execution in (dense, sparse)
    )

    assert all(bool(result.successful) for result in results)
    for name in (
        "training_loss_history",
        "energy_loss_history",
        "force_loss_history",
        "validation_loss_history",
    ):
        np.testing.assert_allclose(
            getattr(results[1], name), getattr(results[0], name), rtol=1e-10
        )
    np.testing.assert_array_equal(results[1].stress_loss_history, np.zeros((3,)))
    assert results[0].problem_id != results[1].problem_id


@pytest.mark.strict_jax
@pytest.mark.parametrize("output_dtype", ["float32", "float64"])
def test_float32_supervised_update_honors_reduction_precision(
    output_dtype: ScalarPrecisionDType,
) -> None:
    precision = AtomisticPrecisionPolicy(
        coordinate_dtype="float32",
        compute_dtype="float32",
        reduction_dtype="float64",
        output_dtype=output_dtype,
    )
    architecture = MACEArchitecture(
        species=(1,),
        cutoff=2.0,
        radial_basis_count=2,
        cutoff_power=5,
        channel_count=2,
        hidden_degree=0,
        edge_degree=0,
        interactions=("real-agnostic",),
        correlations=(2,),
        radial_widths=(4,),
        readout_width=None,
        average_neighbor_count=1.0,
    )

    def potential(seed: int) -> MACEPotential:
        return MACEPotential(
            SCALE,
            architecture,
            atomic_energies=np.asarray([[-1.0]], dtype=np.float32),
            precision=precision,
            key=jr.key(seed),
        )

    batch = AtomisticBatch(
        np.asarray([[1, 1]], dtype=np.int32),
        np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]], dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        SCALE,
        coordinate_dtype="float32",
    )
    execution = AtomisticGraphExecutionPlan(2, maximum_dense_atoms=2)
    teacher = energy_and_forces(potential(0), batch, execution)
    initial = potential(1)
    result = fit_atomistic_potential(
        initial,
        AtomisticTrainingProblem(
            batch,
            execution,
            cutoff=2.0,
            training_energy=teacher.energy,
            training_forces=teacher.forces,
        ),
        AtomisticTrainingPolicy(maximum_steps=1, energy_scale=1.0, force_scale=1.0),
        key=jr.key(5),
    )
    assert bool(result.successful)
    assert result.progress.update_step == 1
    assert bool(jnp.all(jnp.isfinite(result.training_loss_history)))
    assert atomistic_potential_revision(result.potential).revision_id != (
        atomistic_potential_revision(initial).revision_id
    )


@pytest.mark.parametrize(
    "mask_shape",
    [(), (3,), (3, 3), (3, 2, 1)],
    ids=["scalar", "cartesian-only", "atom-cartesian", "broadcast-cartesian"],
)
def test_force_supervision_refuses_implicit_mask_broadcast(
    mask_shape: tuple[int, ...],
) -> None:
    batch = _periodic_batch()
    with pytest.raises(ValueError):
        AtomisticTrainingProblem(
            batch,
            _execution(),
            cutoff=CUTOFF,
            training_forces=np.ones(batch.positions.shape, dtype=np.float64),
            training_force_mask=np.ones(mask_shape, dtype=np.bool_),
        )
