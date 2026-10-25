#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from phydrax.atomistic import (
    atomistic_model_revision,
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticBatch,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticGraphExecutionPlan,
    AtomisticObjectiveSchedule,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticStatus,
    AtomisticStressObjective,
    AtomisticTrainingFields,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    energy_and_forces,
    fit_atomistic_potential,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic import MACEArchitecture, MACEPotential, PaiNNPotential
from phydrax.precision import ScalarPrecisionDType
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, DALTON, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
PERIODIC_FIELDS = AtomisticTrainingFields(cell="cell", periodic="periodic")


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
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=key,
    )


def _execution(*, streamed: StreamedRelationPlan | None = None) -> Any:
    # Every training layout shares one declared search envelope: the cell.
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=125,
        ),
        streamed=streamed,
        search_cell=CELL,
        search_margin=0.1,
    )


def _periodic_batch(
    shift: float = 0.0, *, periodic: Any = (True, True, True), prefix: str = "train"
) -> Any:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]],
            [[0.1, 0.0, 0.0], [1.3, 0.5, -0.1]],
            [[0.0, 0.2, 0.1], [0.9, 1.2, 0.4]],
        ]
    )
    return AtomisticBatch(
        np.asarray([[1, 8], [1, 8], [1, 8]], dtype=np.int64),
        positions + shift,
        np.asarray([[1.0, 16.0], [1.0, 16.0], [1.0, 16.0]], dtype=np.float64),
        SCALE,
        cells=np.broadcast_to(CELL, (3, 3, 3)),
        periodic_axes=np.broadcast_to(np.asarray(periodic, dtype=np.bool_), (3, 3)),
        structure_ids=tuple(f"{prefix}-{index}" for index in range(3)),
    )


def _teacher_labels(batch: Any) -> Any:
    prediction = energy_and_forces(
        _mace(jr.key(91)), batch, _execution(), compute_stress=True
    )
    assert bool(jnp.all(prediction.valid))
    return prediction.energy, prediction.forces, prediction.stress


def _loader(batch: Any, *, stream: str = "training", **labels: Any) -> Any:
    return atomistic_training_loader(
        atomistic_training_source(batch, mass_unit=DALTON, **labels), stream=stream
    )


def _problem(
    batch: Any,
    *,
    objectives: tuple[Any, ...] | None = None,
    execution: Any = None,
    validation: Any = None,
    weighting: Any = "literal",
    **labels: Any,
) -> Any:
    """Problem over one full-batch source with the default E/F/S objectives."""
    if objectives is None:
        objectives = tuple(
            objective
            for name, objective in (
                ("energy", AtomisticEnergyObjective()),
                ("forces", AtomisticForceObjective()),
                ("stress", AtomisticStressObjective()),
            )
            if labels.get(name) is not None
        )
    return AtomisticTrainingProblem(
        _loader(batch, **labels),
        _execution() if execution is None else execution,
        objectives,
        scale=SCALE,
        cutoff=CUTOFF,
        fields=PERIODIC_FIELDS,
        validation=validation,
        weighting=weighting,
    )


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
    # Bitwise continuation requires ordered accumulation: fast reductions keep
    # their nondeterministic rounding on accelerators.
    problem = _problem(
        batch,
        execution=_execution(streamed=StreamedRelationPlan(accumulation="deterministic")),
        energy=energy,
        forces=forces,
        stress=stress,
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
    objectives = continued.training_objectives
    assert objectives.shape == (4, 3)
    assert bool(jnp.all(jnp.isfinite(objectives) & (objectives > 0.0)))
    assert float(continued.training_values[-1]) < float(continued.validation_values[0])
    assert atomistic_model_revision(continued.potential).revision_id != (
        atomistic_model_revision(initial).revision_id
    )
    fitted = energy_and_forces(
        continued.potential, batch, _execution(), compute_stress=True
    )
    assert bool(jnp.all(fitted.valid))


def test_stress_only_supervision_skips_unavailable_cases_and_masked_components() -> None:
    batch = _periodic_batch()
    _, _, stress = _teacher_labels(batch)
    # Case 1 is unavailable; component (0, 1) is excluded by the objective, so
    # the nonfinite label stored there is never read.
    components = ((True, False, True), (True, True, True), (True, True, True))
    component_mask = np.asarray(components, dtype=np.bool_)
    labels = stress.at[1].set(jnp.nan).at[2, 0, 1].set(jnp.inf)
    initial = _mace(jr.key(7))
    result = fit_atomistic_potential(
        initial,
        _problem(
            batch,
            objectives=(AtomisticStressObjective(components=components),),
            stress=labels,
            stress_available=np.asarray([True, False, True]),
        ),
        AtomisticTrainingPolicy(maximum_steps=3, learning_rate=1e-2),
        key=jr.key(8),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert result.training_objectives.shape == (3, 1)
    assert bool(jnp.all(jnp.isfinite(result.training_values)))
    assert float(result.training_values[-1]) < float(result.validation_values[0])
    selected = np.asarray(stress)[[0, 2]][:, component_mask]
    np.testing.assert_allclose(
        result.normalization.scale("stress"), np.sqrt(np.mean(selected * selected))
    )
    assert atomistic_model_revision(result.potential).revision_id != (
        atomistic_model_revision(initial).revision_id
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
        structure_ids=periodic.structure_ids,
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
    problem = _problem(
        mixed,
        energy=teacher.energy,
        forces=teacher.forces,
        stress=teacher.stress,
        stress_available=stress_cases,
    )
    initial = _mace(jr.key(15))
    result = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=5e-3),
        key=jr.key(16),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    objectives = result.training_objectives
    assert objectives.shape == (2, 3)
    assert bool(jnp.all(jnp.isfinite(objectives) & (objectives > 0.0)))
    assert float(result.training_values[-1]) < float(result.validation_values[0])
    assert atomistic_model_revision(result.potential).revision_id != (
        atomistic_model_revision(initial).revision_id
    )


def test_stress_normalization_is_fitted_from_training_labels_only() -> None:
    batch = _periodic_batch()
    validation_batch = _periodic_batch(0.05, prefix="validation")
    _, _, stress = _teacher_labels(batch)

    def fit(validation_stress: Any) -> Any:
        problem = _problem(
            batch,
            stress=stress,
            validation=_loader(
                validation_batch, stream="validation", stress=validation_stress
            ),
        )
        return problem, fit_atomistic_potential(
            _mace(jr.key(2)), problem, AtomisticTrainingPolicy(maximum_steps=0)
        )

    first_problem, first = fit(jnp.full((3, 3, 3), 1e3))
    second_problem, second = fit(jnp.full((3, 3, 3), -4e6))

    assert first.normalization.scales == second.normalization.scales
    np.testing.assert_allclose(
        first.normalization.scale("stress"), np.sqrt(np.mean(np.asarray(stress) ** 2))
    )
    assert first.normalization.fitted_from == first_problem.training.semantics_id
    assert first_problem.problem_id != second_problem.problem_id
    assert float(first.validation_values[0]) != float(second.validation_values[0])


def test_partial_pbc_stress_trains_against_the_declared_embedding_cell() -> None:
    slab = _periodic_batch(periodic=(True, True, False))
    _, _, stress = _teacher_labels(slab)
    problem = _problem(slab, stress=stress)
    initial = _mace(jr.key(13))
    result = fit_atomistic_potential(
        initial,
        problem,
        AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1e-2),
        key=jr.key(14),
    )

    assert int(result.status) == int(AtomisticStatus.SUCCESS)
    assert bool(jnp.all(jnp.isfinite(result.training_values)))
    assert float(result.training_values[-1]) < float(result.validation_values[0])


def test_stress_supervision_refuses_cases_without_a_declared_cell_volume() -> None:
    batch = _periodic_batch()
    _, forces, stress = _teacher_labels(batch)
    finite = AtomisticBatch(
        batch.atomic_numbers,
        batch.positions,
        batch.masses,
        SCALE,
    )
    with pytest.raises(ValueError, match="require batch cells and periodic axes"):
        atomistic_training_source(finite, mass_unit=DALTON, stress=stress)
    unperiodic = _periodic_batch(periodic=(False, False, False))
    with pytest.raises(ValueError, match="finite nonsingular cell"):
        fit_atomistic_potential(
            _mace(jr.key(4)),
            _problem(unperiodic, stress=stress),
            AtomisticTrainingPolicy(maximum_steps=1),
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
        structure_ids=batch.structure_ids,
    )
    with pytest.raises(ValueError, match="finite nonsingular cell"):
        fit_atomistic_potential(
            _mace(jr.key(4)),
            _problem(mixed, forces=forces, stress=stress),
            AtomisticTrainingPolicy(maximum_steps=1),
        )
    # Declaring the singular case's stress unavailable is the explicit remedy.
    declared = fit_atomistic_potential(
        _mace(jr.key(4)),
        _problem(
            mixed,
            forces=forces,
            stress=stress,
            stress_available=np.asarray([True, False, True]),
        ),
        AtomisticTrainingPolicy(maximum_steps=0),
    )
    assert bool(declared.successful)


def test_stress_supervision_refuses_mismatched_target_kinds() -> None:
    batch = _periodic_batch()
    _, forces, stress = _teacher_labels(batch)
    with pytest.raises(ValueError, match="binds no label slot 'stress'"):
        _problem(
            batch,
            stress=stress,
            validation=_loader(
                _periodic_batch(0.05, prefix="validation"),
                stream="validation",
                forces=forces,
            ),
        )


def test_stress_supervision_refuses_zero_total_normalized_weight() -> None:
    batch = _periodic_batch()
    _, _, stress = _teacher_labels(batch)
    with pytest.raises(ValueError, match="positive at every committed update"):
        _problem(
            batch,
            objectives=(
                AtomisticStressObjective(
                    schedule=AtomisticObjectiveSchedule.constant(0.0)
                ),
            ),
            weighting="normalized",
            stress=stress,
        )


def test_nonfinite_update_rolls_back_to_the_last_accepted_potential() -> None:
    batch = _periodic_batch()
    energy, forces, stress = _teacher_labels(batch)
    initial = _mace(jr.key(11))
    result = fit_atomistic_potential(
        initial,
        _problem(batch, energy=energy, forces=forces, stress=stress),
        AtomisticTrainingPolicy(
            maximum_steps=3,
            optimizer=optax.scale(float("nan")),
            optimizer_id="optax.scale:nan",
        ),
        key=jr.key(12),
    )

    assert int(result.status) == int(AtomisticStatus.NONFINITE)
    assert not bool(result.successful)
    assert result.termination == "nonfinite_update_window"
    assert result.progress.update_step == 0
    assert result.training_values.shape == (0,)
    assert atomistic_model_revision(result.potential).revision_id == (
        atomistic_model_revision(initial).revision_id
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
        np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.int64),
        positions,
        np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        SCALE,
        atom_mask=np.asarray([[True, True, True], [True, True, False]], dtype=np.bool_),
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
        search_cell=8.0 * np.eye(3),
    )
    teacher = energy_and_forces(painn(jr.key(21)), batch, dense)
    problems = tuple(
        AtomisticTrainingProblem(
            _loader(batch, energy=teacher.energy, forces=teacher.forces),
            execution,
            (AtomisticEnergyObjective(), AtomisticForceObjective()),
            scale=SCALE,
            cutoff=2.0,
        )
        for execution in (dense, sparse)
    )
    results = tuple(
        fit_atomistic_potential(
            painn(jr.key(22)),
            problem,
            AtomisticTrainingPolicy(maximum_steps=3, learning_rate=2e-3),
            key=jr.key(23),
        )
        for problem in problems
    )

    assert all(bool(result.successful) for result in results)
    for name in ("training_values", "training_objectives", "validation_values"):
        np.testing.assert_allclose(
            getattr(results[1], name), getattr(results[0], name), rtol=1e-10
        )
    # The graph route is execution, not scientific problem identity.
    assert problems[0].problem_id == problems[1].problem_id
    assert results[0].execution_id != results[1].execution_id


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
    # One structure has no fittable energy spread: the scales are declared.
    result = fit_atomistic_potential(
        initial,
        AtomisticTrainingProblem(
            _loader(batch, energy=teacher.energy, forces=teacher.forces),
            execution,
            (AtomisticEnergyObjective(scale=1.0), AtomisticForceObjective(scale=1.0)),
            scale=SCALE,
            cutoff=2.0,
        ),
        AtomisticTrainingPolicy(maximum_steps=1),
        key=jr.key(5),
    )
    assert bool(result.successful)
    assert result.progress.update_step == 1
    assert bool(jnp.all(jnp.isfinite(result.training_values)))
    assert atomistic_model_revision(result.potential).revision_id != (
        atomistic_model_revision(initial).revision_id
    )


@pytest.mark.parametrize(
    "mask_shape",
    [(), (3,), (3, 3), (3, 2, 1), (3, 2, 3)],
    ids=["scalar", "case-only", "wrong-atoms", "broadcast", "per-component"],
)
def test_force_availability_refuses_implicit_broadcast_and_component_masks(
    mask_shape: tuple[int, ...],
) -> None:
    batch = _periodic_batch()
    # Force availability is exactly per atom (case, atom); the removed
    # per-component masks have no equivalent and are refused.
    with pytest.raises(ValueError, match="force availability must have shape"):
        atomistic_training_source(
            batch,
            mass_unit=DALTON,
            forces=np.ones(batch.positions.shape, dtype=np.float64),
            force_available=np.ones(mask_shape, dtype=np.bool_),
        )
