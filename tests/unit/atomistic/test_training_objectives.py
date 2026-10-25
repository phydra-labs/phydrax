#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent float64 references for atomistic objective numerators/supports.

Every objective returns an additive numerator and support; these tests rebuild
both quantities with NumPy from their documented definitions (residual forms,
structure/atom/component balances, case weights, availability and padding) and
compare them with the device-side term functions. Fitting helpers (reference
energies, fitted normalization) are checked against closed-form solutions.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
)
from phydrax.atomistic._training import (
    atomistic_training_loader,
    atomistic_training_source,
    AtomisticTrainingFields,
    AtomisticTrainingProblem,
    fit_atomistic_normalization,
)
from phydrax.atomistic._training_objectives import (
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticObjectiveSchedule,
    AtomisticResidualLoss,
    AtomisticStressObjective,
    AtomisticTrainingPrecision,
    energy_objective_term,
    fit_atomistic_reference_energies,
    force_objective_term,
    stress_objective_term,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.linalg import RankPolicy
from phydrax.units import ANGSTROM, DALTON, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
LOSSES = (
    AtomisticResidualLoss("squared"),
    AtomisticResidualLoss("absolute"),
    AtomisticResidualLoss("huber", delta=0.3),
)
LOSS_IDS = ("mse", "mae", "huber")

# Four cases with capacity three; case 3 is padding (invalid, NaN values,
# unsuccessful), which must stay inert for every objective.
MASK = np.asarray(
    [[True, True, False], [True, True, True], [True, False, False], [True, True, True]],
    dtype=np.bool_,
)
VALID = np.asarray([True, True, True, False], dtype=np.bool_)
WEIGHT = np.asarray([1.0, 2.0, 0.5, 3.0])
SUCCESS = np.asarray([True, True, True, False], dtype=np.bool_)
NO_OVERFLOW = np.zeros((4,), dtype=np.bool_)
SPECIES = np.asarray([[8, 1, 0], [8, 1, 1], [1, 0, 0], [1, 1, 1]])


def _rho(residual: np.ndarray, loss: AtomisticResidualLoss) -> np.ndarray:
    magnitude = np.abs(residual)
    if loss.kind == "squared":
        return residual * residual
    if loss.kind == "absolute":
        return magnitude
    if loss.kind == "huber":
        assert loss.delta is not None
        delta = float(loss.delta)
        return np.where(
            magnitude <= delta,
            0.5 * residual * residual,
            delta * (magnitude - 0.5 * delta),
        )
    raise AssertionError(loss.kind)


def _rho_vector(residual: np.ndarray, loss: AtomisticResidualLoss) -> np.ndarray:
    if loss.kind == "euclidean":
        return np.sqrt(np.sum(residual * residual, axis=-1))
    return np.sum(_rho(residual, loss), axis=-1)


def _energy_term(
    objective: Any, energy: Any, labels: Any, available: Any, **kwargs: Any
) -> Any:
    return energy_objective_term(
        objective,
        jnp.asarray(energy),
        jnp.asarray(kwargs.pop("successful", SUCCESS)),
        jnp.asarray(kwargs.pop("overflow", NO_OVERFLOW)),
        labels=jnp.asarray(labels),
        available=jnp.asarray(available),
        case_valid=jnp.asarray(kwargs.pop("case_valid", VALID)),
        case_weight=jnp.asarray(kwargs.pop("case_weight", WEIGHT)),
        species=jnp.asarray(SPECIES, dtype=jnp.int32),
        atom_mask=jnp.asarray(MASK),
        scale=kwargs.pop("scale", 1.0),
        dtype=kwargs.pop("dtype", "float64"),
    )


def _force_term(
    objective: Any, forces: Any, labels: Any, available: Any, **kwargs: Any
) -> Any:
    return force_objective_term(
        objective,
        jnp.asarray(forces),
        jnp.asarray(kwargs.pop("successful", SUCCESS)),
        jnp.asarray(kwargs.pop("overflow", NO_OVERFLOW)),
        labels=jnp.asarray(labels),
        available=jnp.asarray(available),
        case_valid=jnp.asarray(VALID),
        case_weight=jnp.asarray(WEIGHT),
        atom_mask=jnp.asarray(MASK),
        scale=kwargs.pop("scale", 1.0),
        dtype="float64",
    )


# ------------------------------------------------------------- residual forms


def test_residual_forms_match_their_documented_penalties() -> None:
    residual = np.asarray([-1.2, -0.3, -0.1, 0.0, 0.25, 0.3, 0.31, 4.0])
    for loss in LOSSES:
        np.testing.assert_allclose(
            np.asarray(loss.pointwise(jnp.asarray(residual))),
            _rho(residual, loss),
            rtol=0.0,
            atol=1e-15,
        )
    huber = AtomisticResidualLoss("huber", delta=0.3)
    # The source Huber convention is continuous with a continuous slope at delta.
    inside = float(huber.pointwise(jnp.asarray(0.3)))
    assert inside == pytest.approx(0.5 * 0.3**2, abs=1e-15)
    assert float(huber.pointwise(jnp.asarray(1.3))) == pytest.approx(
        0.3 * (1.3 - 0.15), abs=1e-15
    )
    vectors = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, -2.0, 2.0]])
    euclidean = AtomisticResidualLoss("euclidean")
    np.testing.assert_allclose(
        np.asarray(euclidean.vector(jnp.asarray(vectors))), [5.0, 0.0, 3.0], atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(AtomisticResidualLoss().vector(jnp.asarray(vectors))),
        np.sum(vectors * vectors, axis=-1),
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="no componentwise penalty"):
        euclidean.pointwise(jnp.asarray(vectors))


def test_residual_and_objective_declarations_are_refused_when_ill_posed() -> None:
    with pytest.raises(ValueError, match="Exactly the huber"):
        AtomisticResidualLoss("huber")
    with pytest.raises(ValueError, match="Exactly the huber"):
        AtomisticResidualLoss("squared", delta=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        AtomisticResidualLoss("huber", delta=-1.0)
    euclidean = AtomisticResidualLoss("euclidean")
    with pytest.raises(ValueError, match="euclidean"):
        AtomisticEnergyObjective(loss=euclidean)
    with pytest.raises(ValueError, match="component reduction"):
        AtomisticForceObjective(loss=euclidean, reduction="component")
    with pytest.raises(ValueError, match="componentwise"):
        AtomisticStressObjective(loss=euclidean)
    with pytest.raises(ValueError, match=r"\(3, 3\) boolean mask"):
        AtomisticStressObjective(components=np.zeros((3, 3), dtype=np.bool_))
    with pytest.raises(ValueError, match="finite and positive"):
        AtomisticEnergyObjective(scale=0.0)
    assert (
        AtomisticForceObjective(reduction="atom").objective_id
        != AtomisticForceObjective(reduction="structure").objective_id
    )


# ------------------------------------------------------------------- energy


@pytest.mark.parametrize("loss", LOSSES, ids=LOSS_IDS)
@pytest.mark.parametrize(
    "reduction", ["per-atom-structure", "total-structure", "per-atom-atom-weighted"]
)
def test_energy_numerator_and_support_match_float64_reference(
    loss: AtomisticResidualLoss, reduction: Any
) -> None:
    objective = AtomisticEnergyObjective(reduction=reduction, loss=loss)
    energy = np.asarray([1.0, -2.0, 0.5, np.nan])
    labels = np.asarray([0.4, -1.1, 0.75, np.nan])
    available = np.asarray([True, True, False, True], dtype=np.bool_)
    scale = 0.7
    term = _energy_term(objective, energy, labels, available, scale=scale)

    counts = np.sum(MASK, axis=1).astype(np.float64)
    active = VALID & available
    residual = energy - labels
    if reduction != "total-structure":
        residual = residual / counts
    penalty = _rho(residual / scale, loss)
    weight = np.where(active, WEIGHT, 0.0)
    if reduction == "per-atom-atom-weighted":
        weight = weight * counts
    expected_numerator = float(np.sum(weight[active] * penalty[active]))
    expected_support = float(np.sum(weight))

    assert term.numerator.dtype == jnp.float64
    assert float(term.numerator) == pytest.approx(expected_numerator, rel=1e-14)
    assert float(term.support) == pytest.approx(expected_support, rel=1e-15)
    assert int(term.capacity_failures) == 0
    assert int(term.scientific_failures) == 0


def test_case_weights_scale_numerator_and_support_together() -> None:
    objective = AtomisticEnergyObjective()
    energy = np.asarray([1.0, -2.0, 0.5, 0.0])
    labels = np.asarray([0.4, -1.1, 0.75, 0.0])
    available = np.ones((4,), dtype=np.bool_)
    unit = _energy_term(objective, energy, labels, available, case_weight=np.ones((4,)))
    doubled = _energy_term(
        objective, energy, labels, available, case_weight=np.full((4,), 2.0)
    )
    assert float(doubled.numerator) == pytest.approx(
        2.0 * float(unit.numerator), rel=1e-15
    )
    assert float(doubled.support) == pytest.approx(2.0 * float(unit.support), rel=1e-15)
    # The normalized value is invariant to a uniform case weight.
    assert float(doubled.numerator) / float(doubled.support) == pytest.approx(
        float(unit.numerator) / float(unit.support), rel=1e-15
    )


def test_padding_and_unavailable_labels_are_inert_and_zero_support_is_exact() -> None:
    objective = AtomisticEnergyObjective()
    finite = np.asarray([1.0, -2.0, 0.5, 9.0])
    poisoned = np.asarray([1.0, -2.0, 0.5, np.nan])
    labels = np.asarray([0.4, -1.1, 0.75, np.inf])
    available = np.asarray([True, True, True, True])
    clean = _energy_term(objective, finite, labels, available)
    padded = _energy_term(objective, poisoned, labels, available)
    assert np.isfinite(float(padded.numerator))
    assert float(padded.numerator) == float(clean.numerator)
    assert float(padded.support) == float(clean.support) == pytest.approx(3.5)

    nothing = _energy_term(objective, poisoned, labels, np.zeros((4,), dtype=np.bool_))
    assert float(nothing.numerator) == 0.0
    assert float(nothing.support) == 0.0
    assert int(nothing.scientific_failures) == 0


def test_failed_valid_case_poisons_the_term_and_counts_failure_kind() -> None:
    objective = AtomisticEnergyObjective()
    energy = np.asarray([1.0, -2.0, 0.5, 0.0])
    labels = np.asarray([0.4, -1.1, 0.75, 0.0])
    # A failed valid case invalidates the update even when its label is absent.
    available = np.asarray([True, False, False, False])
    scientific = _energy_term(
        objective,
        energy,
        labels,
        available,
        successful=np.asarray([True, False, True, False]),
    )
    assert math.isnan(float(scientific.numerator))
    assert float(scientific.support) >= 1.0
    assert int(scientific.scientific_failures) == 1
    assert int(scientific.capacity_failures) == 0

    capacity = _energy_term(
        objective,
        energy,
        labels,
        np.zeros((4,), dtype=np.bool_),
        successful=np.asarray([False, True, True, False]),
        overflow=np.asarray([True, False, False, True]),
    )
    assert math.isnan(float(capacity.numerator))
    assert float(capacity.support) == 1.0
    # Overflow is a capacity failure only; the padding case never counts.
    assert int(capacity.capacity_failures) == 1
    assert int(capacity.scientific_failures) == 0


def test_lower_loss_precision_is_an_explicit_cast_at_the_residual() -> None:
    objective = AtomisticEnergyObjective(reduction="total-structure")
    offset = 2.0**-30
    labels = np.asarray([1.0 + offset, 1.0, 1.0, 1.0])
    energy = np.ones((4,), dtype=np.float64)
    available = np.asarray([True, False, False, False])
    wide = _energy_term(objective, energy, labels, available)
    narrow = _energy_term(objective, energy, labels, available, dtype="float32")
    assert wide.numerator.dtype == jnp.float64
    assert float(wide.numerator) == offset * offset
    # float32 cannot represent the label offset: the requested narrower loss
    # dtype rounds the label exactly once where the residual forms.
    assert narrow.numerator.dtype == jnp.float32
    assert float(narrow.numerator) == 0.0


def test_training_precision_is_declared_and_x64_is_required_for_float64() -> None:
    wide = AtomisticTrainingPrecision()
    narrow = AtomisticTrainingPrecision(
        loss_dtype="float32", accumulation_dtype="float64"
    )
    assert (wide.loss_dtype, wide.accumulation_dtype) == ("float64", "float64")
    assert narrow.loss_dtype == "float32"
    assert wide.precision_id != narrow.precision_id
    with pytest.raises((TypeError, ValueError)):
        AtomisticTrainingPrecision(loss_dtype="float16")
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)):
        with pytest.raises(ValueError, match="x64"):
            AtomisticTrainingPrecision()
        AtomisticTrainingPrecision(loss_dtype="float32", accumulation_dtype="float32")


# ------------------------------------------------------------------- forces


def _force_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    forces = generator.normal(size=(4, 3, 3))
    labels = generator.normal(size=(4, 3, 3))
    forces[3] = np.nan
    labels[~MASK] = np.nan
    available = MASK.copy()
    available[1, 2] = False
    available[2, 0] = False  # case 2 keeps no available atom.
    return forces, labels, available


@pytest.mark.parametrize(
    "loss",
    (*LOSSES, AtomisticResidualLoss("euclidean")),
    ids=(*LOSS_IDS, "euclidean"),
)
@pytest.mark.parametrize("reduction", ["component", "atom", "structure"])
def test_force_reductions_match_float64_reference(
    loss: AtomisticResidualLoss, reduction: Any
) -> None:
    if loss.kind == "euclidean" and reduction == "component":
        pytest.skip("Refused at construction (covered separately).")
    objective = AtomisticForceObjective(reduction=reduction, loss=loss)
    forces, labels, available = _force_data()
    scale = 0.4
    term = _force_term(objective, forces, labels, available, scale=scale)

    atoms = VALID[:, None] & available & MASK
    residual = np.where(atoms[:, :, None], (forces - labels) / scale, 0.0)
    weight = np.where(atoms, WEIGHT[:, None], 0.0)
    if reduction == "component":
        penalty = np.sum(np.where(atoms[:, :, None], _rho(residual, loss), 0.0), axis=-1)
        numerator = np.sum(weight * penalty)
        support = 3.0 * np.sum(weight)
    elif reduction == "atom":
        numerator = np.sum(np.where(atoms, weight * _rho_vector(residual, loss), 0.0))
        support = np.sum(weight)
    else:
        counts = np.sum(atoms, axis=1)
        present = counts > 0
        per_atom = np.where(atoms, _rho_vector(residual, loss), 0.0)
        means = np.sum(per_atom, axis=1) / np.where(present, counts, 1)
        structure_weight = np.where(VALID & present, WEIGHT, 0.0)
        numerator = np.sum(structure_weight * means)
        support = np.sum(structure_weight)

    assert float(term.numerator) == pytest.approx(float(numerator), rel=1e-13)
    assert float(term.support) == pytest.approx(float(support), rel=1e-15)
    assert int(term.scientific_failures) == 0


def test_force_balances_differ_on_unequal_structure_sizes() -> None:
    forces, labels, available = _force_data()
    # The general masked fixture has two available atoms in each contributing
    # case; admit the third atom here to exercise genuinely unequal supports.
    available[1, 2] = True
    values = {}
    for reduction in ("component", "atom", "structure"):
        term = _force_term(
            AtomisticForceObjective(reduction=reduction), forces, labels, available
        )
        values[reduction] = float(term.numerator) / float(term.support)
    # Component balance equals the atom balance divided by three components;
    # graph (structure) balance weights every structure equally instead.
    assert values["component"] == pytest.approx(values["atom"] / 3.0, rel=1e-14)
    assert values["structure"] != pytest.approx(values["atom"], rel=1e-6)


# ------------------------------------------------------------------- stress


@pytest.mark.parametrize("loss", LOSSES, ids=LOSS_IDS)
@pytest.mark.parametrize("reduction", ["component", "structure"])
def test_stress_component_masks_and_reductions_match_reference(
    loss: AtomisticResidualLoss, reduction: Any
) -> None:
    components = np.asarray(
        [[True, False, False], [True, True, False], [False, False, True]],
        dtype=np.bool_,
    )
    objective = AtomisticStressObjective(
        reduction=reduction, components=components, loss=loss
    )
    generator = np.random.default_rng(11)
    stress = generator.normal(size=(4, 3, 3))
    labels = generator.normal(size=(4, 3, 3))
    stress[~components[None].repeat(4, axis=0)] = np.nan
    available = np.asarray([True, True, False, True], dtype=np.bool_)
    requested = available.copy()
    scale = 0.05
    term = stress_objective_term(
        objective,
        jnp.asarray(stress),
        jnp.asarray(SUCCESS),
        jnp.asarray(NO_OVERFLOW),
        labels=jnp.asarray(labels),
        available=jnp.asarray(available),
        requested=jnp.asarray(requested),
        case_valid=jnp.asarray(VALID),
        case_weight=jnp.asarray(WEIGHT),
        scale=scale,
        dtype="float64",
    )
    selected = requested & VALID
    weight = np.where(selected, WEIGHT, 0.0)
    residual = np.where(
        selected[:, None, None] & components[None], (stress - labels) / scale, 0.0
    )
    penalty = np.sum(np.where(components[None], _rho(residual, loss), 0.0), axis=(1, 2))
    count = float(np.sum(components))
    if reduction == "component":
        numerator, support = np.sum(weight * penalty), count * np.sum(weight)
    else:
        numerator, support = np.sum(weight * penalty / count), np.sum(weight)
    assert float(term.numerator) == pytest.approx(float(numerator), rel=1e-13)
    assert float(term.support) == pytest.approx(float(support), rel=1e-15)


def test_available_stress_without_admissible_cell_is_a_scientific_failure() -> None:
    objective = AtomisticStressObjective()
    zeros = jnp.zeros((4, 3, 3), dtype=jnp.float64)
    available = jnp.asarray([True, True, False, False])
    term = stress_objective_term(
        objective,
        zeros,
        jnp.asarray(SUCCESS),
        jnp.asarray(NO_OVERFLOW),
        labels=zeros,
        available=available,
        requested=jnp.asarray([True, False, False, False]),
        case_valid=jnp.asarray(VALID),
        case_weight=jnp.asarray(WEIGHT),
        scale=1.0,
        dtype="float64",
    )
    assert math.isnan(float(term.numerator))
    assert int(term.scientific_failures) == 1


# ---------------------------------------------------------------- schedules


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (AtomisticObjectiveSchedule.constant(0.5), {0: 0.5, 3: 0.5, 100: 0.5}),
        (
            AtomisticObjectiveSchedule.linear(1.0, 0.25, steps=4),
            {0: 1.0, 1: 0.8125, 3: 0.4375, 4: 0.25, 9: 0.25},
        ),
        (
            AtomisticObjectiveSchedule.cosine(1.0, 0.0, steps=4),
            {
                0: 1.0,
                1: 0.5 * (1.0 + math.cos(math.pi / 4.0)),
                2: 0.5,
                4: 0.0,
                7: 0.0,
            },
        ),
        (
            AtomisticObjectiveSchedule.piecewise((2, 5), (1.0, 0.5, 0.1)),
            {0: 1.0, 1: 1.0, 2: 0.5, 4: 0.5, 5: 0.1, 50: 0.1},
        ),
    ],
    ids=["constant", "linear", "cosine", "piecewise"],
)
def test_schedules_hold_documented_values_at_committed_update_boundaries(
    schedule: AtomisticObjectiveSchedule, expected: dict[int, float]
) -> None:
    for step, value in expected.items():
        observed = schedule(jnp.asarray(step), dtype="float64")
        assert observed.dtype == jnp.float64
        assert float(observed) == pytest.approx(value, abs=1e-15)
    assert schedule.minimum == pytest.approx(min(expected.values()))


def test_schedule_declarations_are_refused_when_ill_posed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        AtomisticObjectiveSchedule.constant(-1.0)
    with pytest.raises(ValueError, match="positive steps"):
        AtomisticObjectiveSchedule.linear(1.0, 0.0, steps=0)
    with pytest.raises(ValueError, match="one more value"):
        AtomisticObjectiveSchedule.piecewise((2,), (1.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        AtomisticObjectiveSchedule.piecewise((3, 3), (1.0, 0.5, 0.1))
    with pytest.raises(ValueError, match="only start"):
        AtomisticObjectiveSchedule("constant", start=1.0, end=0.5)


# ------------------------------------------------------- sources and fitting


def _composition_source(
    numbers: list[list[int]],
    energies: list[float],
    *,
    available: list[bool] | None = None,
) -> Any:
    numbers_ = np.asarray(numbers, dtype=np.int32)
    mask = numbers_ > 0
    cases, atoms = numbers_.shape
    generator = np.random.default_rng(numbers_.size)
    positions = np.zeros((cases, atoms, 3))
    positions[:, :, 0] = np.arange(atoms)[None, :] * 1.1
    positions += 0.01 * generator.normal(size=positions.shape)
    batch = AtomisticBatch(
        numbers_,
        positions,
        np.where(mask, np.where(numbers_ == 8, 15.999, 1.008), 1.0),
        SCALE,
        atom_mask=mask,
        structure_ids=tuple(f"composition-{index}" for index in range(cases)),
    )
    return atomistic_training_source(
        batch,
        mass_unit=DALTON,
        energy=np.asarray(energies),
        energy_available=None if available is None else np.asarray(available),
    )


def test_reference_energy_fit_recovers_known_coefficients() -> None:
    hydrogen, oxygen = -13.6, -2041.3
    numbers = [[8, 1, 1, 0], [8, 1, 0, 0], [8, 8, 1, 1], [1, 1, 0, 0], [8, 1, 1, 1]]
    counts = np.asarray([[2, 1], [1, 1], [2, 2], [2, 0], [3, 1]], dtype=np.float64)
    energies = counts @ np.asarray([hydrogen, oxygen])
    energies[3] = 1e9  # unavailable: must not enter the fit.
    source = _composition_source(
        numbers, energies.tolist(), available=[True, True, True, False, True]
    )
    loader = atomistic_training_loader(source)
    fit = fit_atomistic_reference_energies(
        loader,
        species_slot="species",
        energy_slot="energy",
        species=(8, 1),
        split="training",
        head="default",
    )
    assert fit.species == (1, 8)
    np.testing.assert_allclose(np.asarray(fit.energies), [hydrogen, oxygen], rtol=1e-10)
    assert fit.rank == 2
    assert fit.record_count == 4
    # Accumulated normal equations cancel to roughly eps * sum(E**2).
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-3)
    assert (fit.split, fit.head, fit.convention) == (
        "training",
        "default",
        "total-energy",
    )
    assert fit.energy_unit.unit_id == ELECTRONVOLT.unit_id
    assert fit.source_ids == loader.mixture.source_ids
    # The fitted composition energy is subtracted from the label inside the term:
    # a model predicting exactly label - sum_i e(z_i) has zero residual.
    fitted = dict(zip(fit.species, np.asarray(fit.energies).tolist(), strict=True))
    offsets = np.asarray(
        [
            sum(fitted[int(code)] for code in row[active])
            for row, active in zip(SPECIES, MASK, strict=True)
        ]
    )
    total = np.asarray([-2068.5, -2070.0, -12.0, 0.0])
    objective = AtomisticEnergyObjective(reduction="total-structure", reference=fit)
    term = _energy_term(objective, total - offsets, total, np.asarray([True] * 4))
    assert float(term.numerator) == pytest.approx(0.0, abs=1e-12)
    assert int(term.scientific_failures) == 0
    assert (
        objective.objective_id
        != AtomisticEnergyObjective(reduction="total-structure").objective_id
    )
    with pytest.raises(ValueError, match="absent from the declared"):
        fit_atomistic_reference_energies(
            loader, species_slot="species", energy_slot="energy", species=(1,)
        )


def test_reference_energy_fit_reports_rank_deficient_composition_evidence() -> None:
    # Every structure has the composition H2O times an integer: the per-species
    # energies are not identifiable, only their H2O combination.
    numbers = [[8, 1, 1, 0, 0, 0], [8, 8, 1, 1, 1, 1], [8, 1, 1, 0, 0, 0]]
    composition = np.asarray([[2.0, 1.0], [4.0, 2.0], [2.0, 1.0]])
    energies = np.asarray([-76.0, -152.1, -76.2])
    loader = atomistic_training_loader(_composition_source(numbers, energies.tolist()))
    fit = fit_atomistic_reference_energies(
        loader, species_slot="species", energy_slot="energy", species=(1, 8)
    )
    assert fit.rank == 1
    singular = np.asarray(fit.gram_singular_values)
    assert singular.shape == (2,)
    assert float(np.min(singular)) <= 1e-10 * float(np.max(singular))
    # The minimum-norm least-squares solution, never an invented unique one.
    expected = np.linalg.pinv(composition) @ energies
    np.testing.assert_allclose(np.asarray(fit.energies), expected, rtol=1e-9)
    residual = composition @ expected - energies
    assert fit.residual_rms == pytest.approx(
        float(np.sqrt(np.mean(residual * residual))), rel=1e-6
    )
    with pytest.raises(ValueError, match="rank"):
        fit_atomistic_reference_energies(
            loader,
            species_slot="species",
            energy_slot="energy",
            species=(1, 8),
            rank_policy=RankPolicy(relative_cutoff=1e-12, require_full_rank=True),
        )


CELL = ((4.2, 0.0, 0.0), (0.3, 4.0, 0.0), (0.1, 0.2, 4.4))


def _periodic_execution() -> AtomisticGraphExecutionPlan:
    """Dense periodic plan; its declared search envelope is the shared cell."""
    return AtomisticGraphExecutionPlan(
        4,
        maximum_dense_atoms=16,
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=256,
            maximum_degree=64,
            maximum_images=343,
        ),
        search_cell=np.asarray(CELL),
        search_margin=0.1,
    )


def _labeled_periodic_source(
    energies: np.ndarray,
    forces: np.ndarray,
    stress: np.ndarray,
    *,
    prefix: str,
    shift: float = 0.0,
    force_available: np.ndarray | None = None,
    energy_available: np.ndarray | None = None,
) -> Any:
    numbers = np.asarray([[8, 1, 1], [8, 1, 0], [1, 1, 0]], dtype=np.int32)
    mask = numbers > 0
    positions = np.zeros((3, 3, 3))
    positions[:, 1, 0] = 0.96
    positions[:, 2, 1] = 0.93
    positions += 0.01 * np.arange(3)[:, None, None] + shift
    cell = np.asarray(CELL)
    batch = AtomisticBatch(
        numbers,
        positions,
        np.ones((3, 3)),
        SCALE,
        atom_mask=mask,
        cells=np.broadcast_to(cell, (3, 3, 3)),
        periodic_axes=np.ones((3, 3), dtype=np.bool_),
        structure_ids=tuple(f"{prefix}-{index}" for index in range(3)),
    )
    return atomistic_training_source(
        batch,
        mass_unit=DALTON,
        energy=energies,
        forces=forces,
        stress=stress,
        energy_available=energy_available,
        force_available=force_available,
    )


def test_fitted_normalization_uses_training_labels_only() -> None:
    generator = np.random.default_rng(3)
    energies = np.asarray([-3.0, 1.5, 0.25])
    forces = generator.normal(size=(3, 3, 3))
    stress = generator.normal(scale=0.01, size=(3, 3, 3))
    force_available = np.asarray(
        [[True, False, True], [True, True, False], [False, True, False]],
        dtype=np.bool_,
    )
    energy_available = np.asarray([True, True, False], dtype=np.bool_)
    training = atomistic_training_loader(
        _labeled_periodic_source(
            energies,
            forces,
            stress,
            prefix="train",
            force_available=force_available,
            energy_available=energy_available,
        )
    )

    def problem(validation_scale: float) -> AtomisticTrainingProblem:
        validation = atomistic_training_loader(
            _labeled_periodic_source(
                energies * validation_scale,
                forces * validation_scale + 3.0,
                stress * validation_scale,
                prefix=f"validation-{validation_scale}",
                shift=0.05,
            ),
            stream="validation",
        )
        return AtomisticTrainingProblem(
            training,
            _periodic_execution(),
            (
                AtomisticEnergyObjective(),
                AtomisticForceObjective(),
                AtomisticStressObjective(
                    components=[
                        [True, False, False],
                        [False, True, False],
                        [False, False, False],
                    ]
                ),
                AtomisticForceObjective(name="declared-forces", scale=0.5),
            ),
            scale=SCALE,
            cutoff=3.0,
            fields=AtomisticTrainingFields(cell="cell", periodic="periodic"),
            validation=validation,
        )

    first = fit_atomistic_normalization(problem(1e3), floor=1e-12)
    second = fit_atomistic_normalization(problem(-4e6), floor=1e-12)
    assert first.scales == second.scales
    assert first.fitted == (True, True, True, False)

    per_atom = (energies / np.asarray([3.0, 2.0, 2.0]))[energy_available]
    expected_energy = float(np.sqrt(np.mean(per_atom**2) - np.mean(per_atom) ** 2))
    selected_forces = forces[force_available]
    expected_forces = float(np.sqrt(np.mean(selected_forces**2)))
    selected_stress = np.stack((stress[:, 0, 0], stress[:, 1, 1]), axis=1)
    expected_stress = float(np.sqrt(np.mean(selected_stress**2)))
    assert first.scale("energy") == pytest.approx(expected_energy, rel=1e-12)
    assert first.scale("forces") == pytest.approx(expected_forces, rel=1e-12)
    assert first.scale("stress") == pytest.approx(expected_stress, rel=1e-12)
    assert first.scale("declared-forces") == 0.5
    assert first.counts[:3] == (2, 3 * int(np.sum(force_available)), 6)
    assert first.fitted_from == training.semantics_id


def test_degenerate_fitted_scales_are_refused_instead_of_clamped() -> None:
    generator = np.random.default_rng(5)
    counts = np.asarray([3.0, 2.0, 2.0])
    forces = generator.normal(size=(3, 3, 3))
    stress = generator.normal(scale=0.01, size=(3, 3, 3))

    def normalization(
        prefix: str, energies: np.ndarray, *, objective: Any = None, **options: Any
    ) -> Any:
        problem = AtomisticTrainingProblem(
            atomistic_training_loader(
                _labeled_periodic_source(
                    energies, forces, stress, prefix=prefix, **options
                )
            ),
            _periodic_execution(),
            (AtomisticEnergyObjective() if objective is None else objective,),
            scale=SCALE,
            cutoff=3.0,
            fields=AtomisticTrainingFields(cell="cell", periodic="periodic"),
        )
        return fit_atomistic_normalization(problem, floor=0.125)

    # Equal per-atom energies: a zero standard deviation is below the floor.
    with pytest.raises(ValueError, match="degenerate scale"):
        normalization("flat", np.full((3,), -6.0) * counts)
    # A declared scale needs no fit, so the same data is admitted.
    declared = normalization(
        "flat-declared",
        np.full((3,), -6.0) * counts,
        objective=AtomisticEnergyObjective(scale=2.0),
    )
    assert declared.scales == (2.0,) and declared.fitted == (False,)
    with pytest.raises(ValueError, match="no available training labels"):
        normalization(
            "unlabeled",
            np.asarray([-3.0, 1.5, 0.25]),
            energy_available=np.zeros((3,), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="nonfinite training labels"):
        normalization("nonfinite", np.asarray([-3.0, np.inf, 0.25]))


def test_float64_labels_require_x64_through_ingest() -> None:
    energies = np.asarray([-3.0 + 2.0**-40, 1.5, 0.25])
    loader = atomistic_training_loader(
        _labeled_periodic_source(
            energies, np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), prefix="precision"
        )
    )
    execution = _periodic_execution()
    fields = AtomisticTrainingFields(cell="cell", periodic="periodic")
    problem = AtomisticTrainingProblem(
        loader,
        execution,
        (AtomisticEnergyObjective(reduction="total-structure"),),
        scale=SCALE,
        cutoff=3.0,
        fields=fields,
    )
    with loader.epoch(loader.start(0)) as epoch:
        item = next(epoch)
        labels = problem.preparation.host_labels(item.payload)
        batch = problem.preparation.prepare_batch(
            item.payload, item.provenance, coordinate_dtype="float32"
        )
    values, _ = labels.label("energy")
    assert values.dtype == np.float64
    first = int(np.argmin(np.abs(values - energies[0])))
    assert values[first] == energies[0]
    prepared, available = batch.label("energy")
    # Labels keep their stored precision even with float32 coordinates.
    assert prepared.dtype == jnp.float64
    assert float(prepared[first]) == energies[0]
    assert bool(jnp.all(available))

    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)):
        narrow = AtomisticTrainingPrecision(
            loss_dtype="float32", accumulation_dtype="float32"
        )
        with pytest.raises(ValueError, match="enable JAX x64"):
            AtomisticTrainingProblem(
                loader,
                execution,
                (AtomisticEnergyObjective(),),
                scale=SCALE,
                cutoff=3.0,
                fields=fields,
                precision=narrow,
            )
