#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import ValuePort
from phydrax._strict import StrictModule
from phydrax.atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioningSpec,
)
from phydrax.atomistic._diffusion import (
    AtomisticLogCholeskyCellChart,
    AtomisticPeriodicCoordinateDiffusion,
    AtomisticPeriodicSamplerMethod,
    AtomisticSpeciesDiffusion,
    PreparedPeriodicDiffusionSampler,
    PreparedPeriodicFlowSampler,
)
from phydrax.atomistic._generation import (
    AbstractPreparedAtomisticSampler,
    AbstractProposalFilter,
    atomistic_proposal_id,
    AtomisticCellBounds,
    AtomisticChargeBalanceQualifier,
    AtomisticCompositionBounds,
    AtomisticConditionSchedule,
    AtomisticConditionTransform,
    AtomisticGeneratedProposal,
    AtomisticGenerationState,
    AtomisticGenerationStopReason,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticMemoryGenerationSink,
    AtomisticMinimumDistanceFilter,
    AtomisticProposalStatus,
    AtomisticSamplerLaneState,
    PreparedAtomisticGenerationStream,
)
from phydrax.domain import (
    Domain,
    DomainFunction,
    HyperRectangle,
    ScalarInterval,
    TimeInterval,
)
from phydrax.dynamics._evolution import EVOLUTION_SUCCESS
from phydrax.stochastic._categorical_diffusion import CategoricalDiffusionSchedule
from phydrax.stochastic._gaussian_diffusion import VarianceExplodingDiffusion
from phydrax.units import ANGSTROM, DIMENSIONLESS


INITIAL_SCALE = 0.01
TERMINAL_SCALE = 1.0
LOG_RATIO = float(np.log(TERMINAL_SCALE / INITIAL_SCALE))
SKEW_CELL = np.asarray([[4.0, 0.0, 0.0], [0.8, 3.8, 0.0], [0.3, 0.4, 4.2]])
SITES = np.asarray([[0.1, 0.2, 0.95], [0.6, 0.5, 0.45], [0.0, 0.0, 0.0]])
CHART_TARGET = np.asarray([0.05, -0.02, 0.1, 0.1, -0.05, 0.08])
MASS_PROVENANCE = "IUPAC standard atomic weights (test fixture)"


def _bounds() -> AtomisticCellBounds:
    return AtomisticCellBounds(
        minimum_height=2.0,
        maximum_condition=10.0,
        minimum_volume=10.0,
        maximum_volume=200.0,
    )


def _process(dimension: int) -> VarianceExplodingDiffusion:
    return VarianceExplodingDiffusion(
        dimension, initial_scale=INITIAL_SCALE, terminal_scale=TERMINAL_SCALE
    )


def _brute_force_log_density(delta: np.ndarray, variance: float) -> np.ndarray:
    images = np.arange(-40, 41, dtype=np.float64)
    shifted = delta[..., None] + images
    return np.log(np.sum(np.exp(-0.5 * shifted**2 / variance), axis=-1))


def _finite_difference_score(delta: np.ndarray, variance: float) -> np.ndarray:
    step = 1e-5
    return (
        _brute_force_log_density(delta + step, variance)
        - _brute_force_log_density(delta - step, variance)
    ) / (2.0 * step)


class _WrappedSiteScore(eqx.Module):
    """Exact marginal score of wrapped normals (scale ``INITIAL_SCALE``) at fixed sites.

    VE noise adds ``s0^2 (exp(2 lambda t) - 1)`` to the target variance, so the
    marginal is a wrapped normal of variance ``(s0 exp(lambda t))^2``; an
    optional chart block is an ordinary Gaussian around ``chart``.
    """

    sites: jax.Array
    chart: jax.Array | None

    def __call__(self, state: jax.Array, time: jax.Array) -> jax.Array:
        variance = (INITIAL_SCALE * jnp.exp(LOG_RATIO * time)) ** 2
        size = self.sites.shape[0]
        delta = state[:size] - self.sites
        delta = delta - jnp.round(delta)
        images = jnp.arange(-6, 7, dtype=state.dtype)
        shifted = delta[:, None] + images[None, :]
        weights = jax.nn.softmax(-0.5 * shifted**2 / variance, axis=-1)
        fractional = -jnp.sum(weights * shifted, axis=-1) / variance
        if self.chart is None:
            return fractional
        return jnp.concatenate((fractional, -(state[size:] - self.chart) / variance))


_TRACES: list[None] = []
_EVALUATIONS: list[None] = []


def _record_evaluation(value: jax.Array) -> None:
    del value
    _EVALUATIONS.append(None)


class _CountingSiteScore(eqx.Module):
    """`_WrappedSiteScore` that records traces (Python) and executed evaluations."""

    inner: _WrappedSiteScore

    def __call__(self, state: jax.Array, time: jax.Array) -> jax.Array:
        _TRACES.append(None)
        jax.debug.callback(_record_evaluation, state[0])
        return self.inner(state, time)


class _ShiftedSiteScore(eqx.Module):
    """Wrapped-site score of sites translated by the case condition ``site_shift``."""

    inner: _WrappedSiteScore

    def __call__(
        self, state: jax.Array, time: jax.Array, site_shift: jax.Array
    ) -> jax.Array:
        size = self.inner.sites.shape[0]
        return self.inner(state.at[:size].add(-site_shift), time)


def _state_domain(dimension: int) -> Domain:
    return HyperRectangle(
        jnp.full((dimension,), -1e3), jnp.full((dimension,), 1e3), label="x"
    ) @ TimeInterval(0.0, 1.0)


def _site_model(sites: np.ndarray, chart: np.ndarray | None) -> _WrappedSiteScore:
    return _WrappedSiteScore(
        jnp.asarray(sites.reshape(-1)), None if chart is None else jnp.asarray(chart)
    )


def _score(
    sites: np.ndarray,
    chart: np.ndarray | None = None,
    *,
    counting: bool = False,
) -> DomainFunction:
    model = _site_model(sites, chart)
    dimension = sites.size + (0 if chart is None else chart.size)
    return _state_domain(dimension).Function("x", "t")(
        _CountingSiteScore(model) if counting else model
    )


def _shifted_score(
    sites: np.ndarray,
    chart: np.ndarray | None = None,
) -> DomainFunction:
    dimension = sites.size + (0 if chart is None else chart.size)
    domain = _state_domain(dimension) @ ScalarInterval(-1.0, 1.0, label="site_shift")
    return domain.Function("x", "t", "site_shift")(
        _ShiftedSiteScore(_site_model(sites, chart))
    )


def _shift_spec() -> AtomisticConditioningSpec:
    return AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                "site_shift",
                ValuePort(
                    "test.periodic-generation.site-shift",
                    event_shape=(),
                    component_ids=("shift",),
                    dimensions=(DIMENSIONLESS,),
                    representation="declared-fractional-site-translation",
                    normalization_id="identity",
                ),
                minimum=-0.5,
                maximum=0.5,
            ),
        )
    )


def _fixed_support(
    *,
    cell_admission: str = "fixed-cell",
    bounds: AtomisticCellBounds | None = None,
    conditioning: AtomisticConditioningSpec | None = None,
) -> AtomisticGenerationSupport:
    variable = cell_admission == "variable-cell"
    return AtomisticGenerationSupport(
        atom_capacity=3,
        species_vocabulary=(8, 14),
        vocabulary_masses=(15.999, 28.085),
        mass_provenance=MASS_PROVENANCE,
        chemistry="fixed",
        length_unit=ANGSTROM,
        conditioning=conditioning,
        fixed_species=(14, 8, 0),
        fixed_mask=(True, True, False),
        cell_admission=cell_admission,
        cell=None if variable else SKEW_CELL,
        cell_bounds=_bounds() if bounds is None else bounds,
    )


def _run(
    sampler: AbstractPreparedAtomisticSampler,
    *,
    lanes: int,
    target: int,
    budget: int,
    filters: Sequence[AbstractProposalFilter] = (),
    seed: int = 7,
    conditions: AtomisticConditionSchedule | None = None,
) -> tuple[
    PreparedAtomisticGenerationStream,
    AtomisticGenerationState,
    tuple[AtomisticGeneratedProposal, ...],
]:
    stream = AtomisticGenerationStreamPlan(
        sampler,
        campaign_id="periodic-generation-test",
        lanes_per_member=lanes,
        pending_capacity=max(lanes, budget),
        filters=filters,
        conditions=conditions,
    ).prepare()
    state = stream.start(jr.key(seed), target=target, attempt_budget=budget)
    state = stream.segment(state, maximum_transitions=4 * budget + 4)
    return stream, state, stream.ready(state)


def _minimum_image_error(fractional: np.ndarray, sites: np.ndarray) -> float:
    delta = fractional - sites
    return float(np.max(np.abs(delta - np.round(delta))))


@pytest.mark.parametrize(("variance", "extent"), [(0.04, 3), (0.25, 4), (0.49, 1)])
def test_wrapped_normal_score_matches_brute_force_image_density(
    variance: float, extent: int
) -> None:
    diffusion = AtomisticPeriodicCoordinateDiffusion(2, _process(6), image_extent=extent)
    deltas = np.asarray([[-0.5, -0.31, -0.02], [0.07, 0.26, 0.49]])
    score, bound = diffusion.wrapped_normal_score(jnp.asarray(deltas), variance)
    reference = _finite_difference_score(deltas, variance)
    error = np.abs(np.asarray(score) - reference)

    assert np.all(error <= np.asarray(bound) + 1e-6)
    if extent >= 3:
        assert float(np.max(bound)) < 1e-6
    else:
        # A one-image truncation at a wide variance is visibly wrong, and the
        # declared bound still covers it.
        assert float(np.max(error)) > 1e-3


def test_periodic_conditional_score_and_perturbation_respect_mask() -> None:
    process = _process(9)
    diffusion = AtomisticPeriodicCoordinateDiffusion(3, process, image_extent=5)
    clean = jnp.asarray([[0.95, 0.02, 0.5], [0.3, 0.7, 0.1], [0.25, 0.25, 0.25]])
    mask = jnp.asarray([True, True, False])
    time = jnp.asarray(0.6)
    perturbed = diffusion.perturb(jr.key(0), clean, mask, time=time)
    fractional = np.asarray(perturbed.fractional)

    assert perturbed.chart is None
    assert np.all((fractional[:2] >= 0.0) & (fractional[:2] < 1.0))
    np.testing.assert_array_equal(fractional[2], np.asarray(clean)[2])
    result = diffusion.conditional_score(perturbed.fractional, clean, mask, time=time)
    variance = float(process.transition_variance(0.0, time))
    delta = fractional - np.asarray(clean)
    delta = delta - np.round(delta)
    expected = _finite_difference_score(delta, variance)

    np.testing.assert_allclose(
        np.asarray(result.fractional)[:2],
        expected[:2],
        atol=float(result.maximum_truncation_bound) + 1e-5,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(np.asarray(result.fractional)[2], 0.0)
    assert float(result.maximum_truncation_bound) < 1e-8
    with pytest.raises(ValueError, match="cell chart"):
        diffusion.perturb(jr.key(1), clean, mask, time=time, chart=jnp.zeros((6,)))


def test_cell_chart_round_trip_gauge_and_unrepaired_bounds() -> None:
    chart = AtomisticLogCholeskyCellChart(_bounds(), reference_length=4.0)
    coordinates = jnp.asarray([CHART_TARGET, np.zeros(6)])
    cells = np.asarray(chart.decode(coordinates))

    np.testing.assert_array_equal(np.triu(cells, 1), 0.0)
    assert np.all(np.linalg.det(cells) > 0.0)
    encoded, successful = chart.encode(jnp.asarray(cells))
    assert bool(jnp.all(successful))
    np.testing.assert_allclose(encoded, coordinates, atol=1e-12)
    first, second = 0.3, 0.7
    about_z = np.asarray(
        [
            [np.cos(first), -np.sin(first), 0.0],
            [np.sin(first), np.cos(first), 0.0],
            [0, 0, 1],
        ]
    )
    about_x = np.asarray(
        [
            [1, 0, 0],
            [0.0, np.cos(second), -np.sin(second)],
            [0.0, np.sin(second), np.cos(second)],
        ]
    )
    rotation = about_z @ about_x
    rotated, rotated_ok = chart.encode(jnp.asarray(cells @ rotation.T))
    assert bool(jnp.all(rotated_ok))
    np.testing.assert_allclose(rotated, coordinates, atol=1e-10)
    _, left_handed = chart.encode(jnp.asarray(cells[0] @ np.diag([1.0, 1.0, -1.0])))
    assert not bool(left_handed)

    evidence = chart.evaluate(coordinates)
    assert bool(jnp.all(evidence.admitted))
    squeezed = jnp.asarray([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rejected = chart.evaluate(squeezed)
    assert not bool(rejected.admitted)
    np.testing.assert_allclose(rejected.cell, chart.decode(squeezed))
    assert float(jnp.min(rejected.heights)) < 2.0


def test_species_diffusion_vacancy_encoding_infers_only_occupancy() -> None:
    schedule = CategoricalDiffusionSchedule.uniform(4, 3)
    species = AtomisticSpeciesDiffusion(schedule, (11, 17), vacancy=True)
    classes = species.encode(jnp.asarray([17, 0, 11]), jnp.asarray([True, False, True]))

    np.testing.assert_array_equal(classes, [1, 2, 0])
    numbers, mask, valid = species.decode(classes)
    np.testing.assert_array_equal(numbers, [17, 0, 11])
    np.testing.assert_array_equal(mask, [True, False, True])
    assert bool(jnp.all(valid))
    with pytest.raises(ValueError, match="one class per element"):
        AtomisticSpeciesDiffusion(schedule, (11, 17), vacancy=False)
    occupied = AtomisticSpeciesDiffusion(
        CategoricalDiffusionSchedule.uniform(4, 2), (11, 17), vacancy=False
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="vacancies"):
        occupied.encode(jnp.asarray([11, 0]), jnp.asarray([True, False]))


def test_periodic_samplers_refuse_undeclared_support_kinds() -> None:
    diffusion = AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4)
    nonperiodic = AtomisticGenerationSupport(
        atom_capacity=3,
        species_vocabulary=(8, 14),
        vocabulary_masses=(15.999, 28.085),
        mass_provenance=MASS_PROVENANCE,
        chemistry="fixed",
        length_unit=ANGSTROM,
        fixed_species=(14, 8, 0),
        fixed_mask=(True, True, False),
    )
    with pytest.raises(ValueError, match="periodic support"):
        PreparedPeriodicDiffusionSampler(
            nonperiodic,
            diffusion,
            _score(SITES),
            score_id="sites",
            dt0=0.01,
            wiener_tolerance=1e-4,
        )
    with pytest.raises(ValueError, match="cell-chart"):
        PreparedPeriodicDiffusionSampler(
            _fixed_support(cell_admission="variable-cell"),
            diffusion,
            _score(SITES),
            score_id="sites",
            dt0=0.01,
            wiener_tolerance=1e-4,
        )
    with pytest.raises(ValueError, match="Fixed chemistry"):
        PreparedPeriodicFlowSampler(
            _fixed_support(),
            diffusion,
            _score(SITES),
            score_id="sites",
            species=AtomisticSpeciesDiffusion(
                CategoricalDiffusionSchedule.uniform(4, 2), (8, 14), vacancy=False
            ),
        )


def test_periodic_reverse_diffusion_concentrates_on_fixed_cell_sites() -> None:
    diffusion = AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4)
    sampler = PreparedPeriodicDiffusionSampler(
        _fixed_support(),
        diffusion,
        _score(SITES),
        score_id="analytic-wrapped-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
    )
    assert sampler.idempotent and not sampler.stateful
    stream, state, proposals = _run(sampler, lanes=2, target=2, budget=2)

    assert state.stop_reason is AtomisticGenerationStopReason.TARGET_REACHED
    assert [item.ordinal for item in proposals] == [0, 1]
    for proposal in proposals:
        fractional = proposal.value("fractional_positions")
        assert proposal.status == 0
        assert bool(proposal.value("solver_valid"))
        assert bool(proposal.value("backend_successful"))
        np.testing.assert_array_equal(proposal.slots, [0, 1])
        np.testing.assert_array_equal(proposal.species, [14, 8])
        np.testing.assert_array_equal(proposal.cell, SKEW_CELL)
        assert _minimum_image_error(fractional[:2], SITES[:2]) < 0.05
        np.testing.assert_array_equal(fractional[2], 0.0)
        np.testing.assert_allclose(
            proposal.positions, fractional[:2] @ SKEW_CELL, atol=1e-12
        )
    assert not np.allclose(proposals[0].positions, proposals[1].positions)
    assert stream.shortfall(state) == 2


def test_periodic_probability_flow_concentrates_on_fixed_cell_sites() -> None:
    diffusion = AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4)
    sampler = PreparedPeriodicFlowSampler(
        _fixed_support(),
        diffusion,
        _score(SITES),
        score_id="analytic-wrapped-sites",
        rtol=1e-7,
        atol=1e-9,
    )
    _, _, proposals = _run(sampler, lanes=2, target=2, budget=2)

    assert len(proposals) == 2
    for proposal in proposals:
        fractional = proposal.value("fractional_positions")
        assert proposal.status == 0
        assert int(proposal.value("evolution_status")) == EVOLUTION_SUCCESS
        assert _minimum_image_error(fractional[:2], SITES[:2]) < 0.05
        np.testing.assert_allclose(
            proposal.positions, fractional[:2] @ SKEW_CELL, atol=1e-12
        )


def test_periodic_reverse_diffusion_recovers_variable_cell_chart_target() -> None:
    bounds = _bounds()
    chart = AtomisticLogCholeskyCellChart(bounds, reference_length=4.0)
    diffusion = AtomisticPeriodicCoordinateDiffusion(
        3, _process(15), image_extent=4, cell_chart=chart
    )
    sampler = PreparedPeriodicDiffusionSampler(
        _fixed_support(cell_admission="variable-cell", bounds=bounds),
        diffusion,
        _score(SITES, CHART_TARGET),
        score_id="analytic-wrapped-sites-and-chart",
        dt0=0.01,
        wiener_tolerance=1e-4,
    )
    _, _, proposals = _run(sampler, lanes=2, target=2, budget=2)
    target_cell = np.asarray(chart.decode(jnp.asarray(CHART_TARGET)))

    assert len(proposals) == 2
    for proposal in proposals:
        coordinates = proposal.value("cell_chart")
        fractional = proposal.value("fractional_positions")
        assert proposal.status == 0
        assert proposal.cell is not None
        np.testing.assert_allclose(coordinates, CHART_TARGET, atol=0.05)
        np.testing.assert_allclose(
            proposal.cell, chart.decode(jnp.asarray(coordinates)), atol=1e-12
        )
        np.testing.assert_allclose(proposal.cell, target_cell, atol=0.25)
        assert _minimum_image_error(fractional[:2], SITES[:2]) < 0.05
        np.testing.assert_allclose(
            proposal.positions, fractional[:2] @ proposal.cell, atol=1e-12
        )


def _species_logits(
    classes: jax.Array, timestep: jax.Array, *, key: jax.Array
) -> jax.Array:
    del timestep, key
    preferred = jnp.asarray([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    return jnp.broadcast_to(preferred, classes.shape + (3,))


def _chloride_logits(
    classes: jax.Array, timestep: jax.Array, *, key: jax.Array
) -> jax.Array:
    """Another composition law: both slots prefer chlorine."""
    del timestep, key
    preferred = jnp.asarray([[0.0, 1.5, 0.0], [0.0, 1.5, 0.0]])
    return jnp.broadcast_to(preferred, classes.shape + (3,))


PREFERRED_LOGITS = np.asarray([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])


class _PreferenceLogits(StrictModule):
    """Parameter-bearing denoiser: clean logits ``preferred`` for every state."""

    preferred: jax.Array

    def __call__(
        self, classes: jax.Array, timestep: jax.Array, *, key: jax.Array
    ) -> jax.Array:
        del timestep, key
        return jnp.broadcast_to(self.preferred, classes.shape + (3,))


def _composition_support() -> AtomisticGenerationSupport:
    return AtomisticGenerationSupport(
        atom_capacity=2,
        species_vocabulary=(11, 17),
        vocabulary_masses=(22.99, 35.45),
        mass_provenance=MASS_PROVENANCE,
        chemistry="variable-composition",
        length_unit=ANGSTROM,
        composition=AtomisticCompositionBounds(
            {11: (0, 2), 17: (0, 2)}, minimum_atoms=1, maximum_atoms=2
        ),
        cell_admission="fixed-cell",
        cell=np.diag([5.0, 5.0, 5.0]),
        cell_bounds=AtomisticCellBounds(
            minimum_height=2.0,
            maximum_condition=10.0,
            minimum_volume=10.0,
            maximum_volume=500.0,
        ),
    )


def _species_diffusion() -> AtomisticSpeciesDiffusion:
    return AtomisticSpeciesDiffusion(
        CategoricalDiffusionSchedule.uniform(8, 3), (11, 17), vacancy=True
    )


_COMPOSITION_SITES = np.asarray([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])


def _composition_sampler(
    *,
    lane_chunk: int = 1,
    counting: bool = False,
    predictor: Callable[..., jax.Array] = _species_logits,
    predictor_semantic_id: str | None = None,
    predictor_numeric_id: str | None = None,
) -> PreparedPeriodicDiffusionSampler:
    diffusion = AtomisticPeriodicCoordinateDiffusion(2, _process(6), image_extent=4)
    return PreparedPeriodicDiffusionSampler(
        _composition_support(),
        diffusion,
        _score(_COMPOSITION_SITES, counting=counting),
        score_id="analytic-wrapped-rock-salt-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
        species=_species_diffusion(),
        species_predictor=predictor,
        species_predictor_semantic_id=predictor_semantic_id,
        species_predictor_numeric_id=predictor_numeric_id,
        lane_chunk=lane_chunk,
    )


def _composition_flow_sampler(
    predictor: Callable[..., jax.Array],
) -> PreparedPeriodicFlowSampler:
    diffusion = AtomisticPeriodicCoordinateDiffusion(2, _process(6), image_extent=4)
    return PreparedPeriodicFlowSampler(
        _composition_support(),
        diffusion,
        _score(_COMPOSITION_SITES),
        score_id="analytic-wrapped-rock-salt-sites",
        species=_species_diffusion(),
        species_predictor=predictor,
        rtol=1e-7,
        atol=1e-9,
    )


def _capturing_predictor(captured: jax.Array) -> Callable[..., jax.Array]:
    """Opaque denoiser closing over ``captured`` (state hidden from the PyTree)."""

    def predictor(
        classes: jax.Array, timestep: jax.Array, *, key: jax.Array
    ) -> jax.Array:
        _ = captured
        return _species_logits(classes, timestep, key=key)

    return predictor


def _capturing_transform(
    captured: jax.Array,
) -> Callable[[Mapping[str, jax.Array], jax.Array], dict[str, jax.Array]]:
    """Opaque condition map closing over ``captured`` (state hidden from the PyTree)."""

    def transform(
        values: Mapping[str, jax.Array], keys: jax.Array
    ) -> dict[str, jax.Array]:
        _ = captured, keys
        return dict(values)

    return transform


class _TabulatedLogits(eqx.Module):
    """Opaque PyTree denoiser with visible integer-table and PRNG-key leaves."""

    table: jax.Array
    seed: jax.Array

    def __call__(
        self, classes: jax.Array, timestep: jax.Array, *, key: jax.Array
    ) -> jax.Array:
        del timestep, key
        logits = 1.5 * jax.nn.one_hot(self.table, 3, dtype=jnp.float64)
        return jnp.broadcast_to(logits, classes.shape + (3,))


class _StrictTabulatedLogits(StrictModule):
    """Content-addressed denoiser with visible integer-table and PRNG-key fields."""

    table: jax.Array
    seed: jax.Array

    def __call__(
        self, classes: jax.Array, timestep: jax.Array, *, key: jax.Array
    ) -> jax.Array:
        del timestep, key
        logits = 1.5 * jax.nn.one_hot(self.table, 3, dtype=jnp.float64)
        return jnp.broadcast_to(logits, classes.shape + (3,))


def _closest_image_distance(positions: np.ndarray, cell: np.ndarray) -> float:
    closest = np.inf
    for i, j in itertools.product(range(len(positions)), repeat=2):
        for shift in itertools.product(range(-2, 3), repeat=3):
            if i == j and not any(shift):
                continue
            image = positions[j] - positions[i] + np.asarray(shift) @ cell
            closest = min(closest, float(np.linalg.norm(image)))
    return closest


def _record(
    proposal: AtomisticGeneratedProposal,
) -> tuple[str, int, int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        proposal.proposal_id,
        proposal.ordinal,
        proposal.status,
        proposal.filter_bits,
        tuple(proposal.species.tolist()),
        tuple(proposal.slots.tolist()),
    )


def test_variable_composition_stream_qualifies_chemistry_and_is_lane_invariant() -> None:
    filters = (
        AtomisticChargeBalanceQualifier(
            {11: (1,), 17: (-1,)}, provenance="common oxidation states", maximum_atoms=2
        ),
        AtomisticMinimumDistanceFilter(2.0),
    )
    runs = [
        _run(
            _composition_sampler(lane_chunk=chunk),
            lanes=lanes,
            target=8,
            budget=8,
            filters=filters,
        )
        for lanes, chunk in ((1, 1), (3, 2), (4, 3))
    ]
    (_, single_state, single), *others = runs

    assert single_state.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED
    assert [item.ordinal for item in single] == list(range(8))
    for _, other_state, other in others:
        # Lane count and lane chunk are executable choices: ordinal-addressed
        # proposals, their keys and their reason bits do not depend on them.
        assert other_state.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED
        assert [_record(item) for item in single] == [_record(item) for item in other]
        for left, right in zip(single, other, strict=True):
            np.testing.assert_allclose(
                left.positions, right.positions, rtol=0.0, atol=1e-12
            )
            np.testing.assert_array_equal(
                left.value("species_classes"), right.value("species_classes")
            )

    statuses = AtomisticProposalStatus
    accepted = rejected = 0
    for proposal in single:
        numbers = proposal.species
        total = len(numbers)
        balanced = int(np.sum(numbers == 11)) == int(np.sum(numbers == 17))
        classes = proposal.value("species_classes")
        decoded = np.where(classes == 0, 11, np.where(classes == 1, 17, 0))
        np.testing.assert_array_equal(decoded[proposal.slots], numbers)
        assert np.all(classes[np.setdiff1d(np.arange(2), proposal.slots)] == 2)
        if not 1 <= total <= 2:
            assert proposal.status & int(statuses.COMPOSITION_INVALID)
            assert proposal.filter_bits == 0
        elif not balanced:
            assert proposal.status & int(statuses.CHEMISTRY_UNQUALIFIED)
            assert proposal.status & int(statuses.FILTER_REJECTED)
            assert proposal.filter_bits == 1
        else:
            assert proposal.cell is not None
            assert _closest_image_distance(proposal.positions, proposal.cell) >= 2.0
            assert proposal.status == 0
        accepted += proposal.status == 0
        rejected += proposal.status != 0
        # Rejected proposals keep their geometry and reason bits.
        assert proposal.positions.shape == (total, 3)
    assert accepted > 0 and rejected > 0


def test_zero_target_periodic_stream_is_empty_and_consumes_nothing() -> None:
    sampler = _composition_sampler(counting=True)
    stream = AtomisticGenerationStreamPlan(
        sampler,
        campaign_id="periodic-generation-test",
        lanes_per_member=2,
        pending_capacity=4,
    ).prepare()
    _TRACES.clear()
    _EVALUATIONS.clear()
    state = stream.start(jr.key(0), target=0, attempt_budget=5)
    state = stream.segment(state, maximum_transitions=3)
    jax.effects_barrier()

    assert state.stop_reason is AtomisticGenerationStopReason.ZERO_TARGET
    assert stream.ready(state) == ()
    assert stream.finished(state)
    assert stream.shortfall(state) == 0
    assert int(state.cursor.next_ordinal) == 0
    assert int(state.cursor.transitions) == 0
    assert not _TRACES and not _EVALUATIONS


def _counting_sampler(
    kind: str,
) -> PreparedPeriodicDiffusionSampler | PreparedPeriodicFlowSampler:
    diffusion = AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4)
    score = _score(SITES, counting=True)
    if kind == "reverse-sde":
        return PreparedPeriodicDiffusionSampler(
            _fixed_support(),
            diffusion,
            score,
            score_id="analytic-wrapped-sites",
            dt0=0.01,
            wiener_tolerance=1e-4,
        )
    return PreparedPeriodicFlowSampler(
        _fixed_support(),
        diffusion,
        score,
        score_id="analytic-wrapped-sites",
        rtol=1e-7,
        atol=1e-9,
    )


def _admitted(
    sampler: AbstractPreparedAtomisticSampler,
    selected: tuple[bool, ...],
) -> tuple[AtomisticSamplerLaneState, int]:
    lanes = len(selected)
    keys = jr.split(jr.key(21), lanes)
    ordinals = jnp.arange(lanes, dtype=jnp.int64)
    _EVALUATIONS.clear()
    state = sampler.admit(
        sampler.vacant(lanes), jnp.asarray(selected), keys, ordinals, None
    )
    jax.block_until_ready(state.method.fractional)
    jax.effects_barrier()
    return state, len(_EVALUATIONS)


def _rows_equal(
    left: AtomisticPeriodicSamplerMethod,
    right: AtomisticPeriodicSamplerMethod,
    row: int,
) -> None:
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(
            np.asarray(a)[row], np.asarray(b)[row]
        ),
        left,
        right,
    )


@pytest.mark.parametrize("kind", ["reverse-sde", "probability-flow"])
def test_periodic_admission_is_compiled_once_and_never_evaluates_vacant_lanes(
    kind: str,
) -> None:
    sampler = _counting_sampler(kind)
    vacant = sampler.vacant(4)
    _TRACES.clear()
    nothing, nothing_count = _admitted(sampler, (False, False, False, False))
    traced = len(_TRACES)
    first, first_count = _admitted(sampler, (True, False, False, False))
    third, third_count = _admitted(sampler, (False, False, True, False))
    both, both_count = _admitted(sampler, (True, False, True, False))

    # One stable compiled entrypoint: admissions with other masks, keys or
    # ordinals are dynamic data and never retrace the score.
    assert traced > 0
    assert len(_TRACES) == traced
    # Vacant lanes consume no score evaluation; admitted lanes cost exactly
    # their own solves.
    assert nothing_count == 0
    assert first_count > 0 and third_count > 0
    assert both_count == first_count + third_count
    assert not bool(jnp.any(nothing.active))
    _rows_equal(nothing.method, vacant.method, 0)
    for row in (1, 3):
        _rows_equal(both.method, vacant.method, row)
    _rows_equal(both.method, first.method, 0)
    _rows_equal(both.method, third.method, 2)
    np.testing.assert_array_equal(both.active, [True, False, True, False])
    np.testing.assert_array_equal(both.terminal, [True, False, True, False])
    assert bool(jnp.all(both.method.solver_valid[jnp.asarray([0, 2])]))


@pytest.mark.parametrize("cell_admission", ["fixed-cell", "variable-cell"])
def test_conditional_periodic_proposals_follow_their_ordinal_conditions(
    cell_admission: str,
) -> None:
    bounds = _bounds()
    variable = cell_admission == "variable-cell"
    chart = (
        AtomisticLogCholeskyCellChart(bounds, reference_length=4.0) if variable else None
    )
    diffusion = AtomisticPeriodicCoordinateDiffusion(
        3, _process(15 if variable else 9), image_extent=4, cell_chart=chart
    )
    spec = _shift_spec()
    sampler = PreparedPeriodicDiffusionSampler(
        _fixed_support(cell_admission=cell_admission, bounds=bounds, conditioning=spec),
        diffusion,
        _shifted_score(SITES, CHART_TARGET if variable else None),
        score_id="analytic-translated-wrapped-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
        lane_chunk=2,
    )
    shifts = np.asarray([0.0, 0.25, -0.2, 0.75])
    schedule = AtomisticConditionSchedule(spec, {"site_shift": shifts}, policy="bounded")
    _, state, proposals = _run(sampler, lanes=3, target=4, budget=4, conditions=schedule)

    assert [item.ordinal for item in proposals] == [0, 1, 2, 3]
    for proposal, shift in zip(proposals, shifts, strict=True):
        fractional = proposal.value("fractional_positions")
        assert bool(proposal.value("solver_valid"))
        assert _minimum_image_error(fractional[:2], SITES[:2] + shift) < 0.05
        np.testing.assert_allclose(
            proposal.positions, fractional[:2] @ proposal.cell, atol=1e-12
        )
        if variable:
            np.testing.assert_allclose(
                proposal.value("cell_chart"), CHART_TARGET, atol=0.05
            )
        else:
            np.testing.assert_array_equal(proposal.cell, SKEW_CELL)
        if abs(shift) <= 0.5:
            assert proposal.status == 0
        else:
            # The declared condition range is admission, not a silent clip.
            assert proposal.status & int(AtomisticProposalStatus.CONDITION_INVALID)
    assert state.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED


def test_periodic_stream_restarts_exactly_from_a_checkpoint(tmp_path: Path) -> None:
    plan = AtomisticGenerationStreamPlan(
        _composition_sampler(lane_chunk=2),
        campaign_id="periodic-generation-restart",
        lanes_per_member=2,
        pending_capacity=4,
    )
    stream = plan.prepare()
    reference = AtomisticMemoryGenerationSink("uninterrupted")
    stream.run(
        stream.start(jr.key(5), target=6, attempt_budget=6),
        reference,
        maximum_transitions=40,
    )

    sink = AtomisticMemoryGenerationSink("interrupted")
    state = stream.start(jr.key(5), target=6, attempt_budget=6)
    state = stream.segment(state, maximum_transitions=2)
    state, _ = stream.publish(state, sink)
    path = tmp_path / "periodic-generation.npz"
    stream.write_checkpoint(path, state)
    restored = stream.read_checkpoint(path)
    jax.tree.map(
        np.testing.assert_array_equal, jax.device_get(restored), jax.device_get(state)
    )
    resumed = stream.run(restored, sink, maximum_transitions=40)

    assert stream.finished(resumed)
    assert [_record(item) for item in sink.records] == [
        _record(item) for item in reference.records
    ]
    for left, right in zip(sink.records, reference.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)


def test_species_denoiser_identity_binds_both_periodic_samplers() -> None:
    sodium = _composition_sampler()
    chloride = _composition_sampler(predictor=_chloride_logits)

    # Another composition-generating function shares the categorical chain
    # identity (schedule, shape, terminal reference) but is another sampler:
    # its proposal identities and keys differ.
    assert sodium.species_chain is not None and chloride.species_chain is not None
    assert sodium.species_chain.sampler_id == chloride.species_chain.sampler_id
    assert sodium.species_predictor_id != chloride.species_predictor_id
    assert sodium.sampler_id != chloride.sampler_id
    assert atomistic_proposal_id(
        "campaign", sodium.support.support_id, sodium.sampler_id, 0
    ) != atomistic_proposal_id(
        "campaign", chloride.support.support_id, chloride.sampler_id, 0
    )
    again = _composition_sampler()
    assert again.sampler_id == sodium.sampler_id
    assert again.numeric_revision().revision_id == sodium.numeric_revision().revision_id

    flows = [
        _composition_flow_sampler(item) for item in (_species_logits, _chloride_logits)
    ]
    assert flows[0].species_predictor_id == sodium.species_predictor_id
    assert flows[0].sampler_id != flows[1].sampler_id
    weighted = [
        _composition_flow_sampler(_PreferenceLogits(jnp.asarray(values)))
        for values in (PREFERRED_LOGITS, PREFERRED_LOGITS[::-1])
    ]
    # Denoiser coefficients are numeric content, not semantic identity.
    assert weighted[0].sampler_id == weighted[1].sampler_id
    assert (
        weighted[0].numeric_revision().revision_id
        != weighted[1].numeric_revision().revision_id
    )

    # A closure or lambda is opaque: it declares both identities, which bind.
    def wrapped(classes: jax.Array, timestep: jax.Array, *, key: jax.Array) -> jax.Array:
        return _species_logits(classes, timestep, key=key)

    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        _composition_sampler(predictor=wrapped)
    declared = {
        (law, numeric): _composition_sampler(
            predictor=wrapped,
            predictor_semantic_id=law,
            predictor_numeric_id=numeric,
        )
        for law, numeric in (
            ("sodium-first-law", "fixed-preference-logits"),
            ("chlorine-first-law", "fixed-preference-logits"),
            ("sodium-first-law", "recalibrated-preference-logits"),
        )
    }
    first = declared["sodium-first-law", "fixed-preference-logits"]
    other_law = declared["chlorine-first-law", "fixed-preference-logits"]
    recalibrated = declared["sodium-first-law", "recalibrated-preference-logits"]
    assert first.sampler_id != other_law.sampler_id
    # A declared numeric ID binds the revision only: it never enters the
    # sampler's static structure or executable facts.
    assert recalibrated.sampler_id == first.sampler_id
    assert recalibrated.signature_facts() == first.signature_facts()
    assert (
        recalibrated.numeric_revision().revision_id
        != first.numeric_revision().revision_id
    )
    # Declared IDs never authorize numerical state hidden from the PyTree:
    # floating weights, integer lookup tables and typed PRNG keys alike.
    for hidden in (
        jnp.asarray(PREFERRED_LOGITS),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jr.key(11),
    ):
        with pytest.raises(ValueError, match="hides numerical arrays"):
            _composition_sampler(
                predictor=_capturing_predictor(hidden),
                predictor_semantic_id="hidden-state-law",
                predictor_numeric_id="hidden-state",
            )
    # Visible integer and key leaves of an opaque PyTree denoiser are admitted
    # and bound by the numeric revision, not by the semantic identity.
    tabulated = [
        _composition_sampler(
            predictor=_TabulatedLogits(jnp.asarray(table, dtype=jnp.int32), jr.key(seed)),
            predictor_semantic_id="tabulated-preference-law",
            predictor_numeric_id="tabulated-preferences",
        )
        for table, seed in (([0, 1], 0), ([1, 1], 0), ([0, 1], 1))
    ]
    assert len({item.sampler_id for item in tabulated}) == 1
    assert len({item.numeric_revision().revision_id for item in tabulated}) == 3
    content_addressed = [
        _composition_sampler(
            predictor=_StrictTabulatedLogits(
                jnp.asarray(table, dtype=jnp.int32), jr.key(seed)
            )
        )
        for table, seed in (([0, 1], 0), ([1, 1], 0), ([0, 1], 1))
    ]
    assert len({item.sampler_id for item in content_addressed}) == 1
    assert content_addressed[0].sampler_id != tabulated[0].sampler_id
    assert len({item.numeric_revision().revision_id for item in content_addressed}) == 3
    with pytest.raises(ValueError, match="Fixed chemistry"):
        PreparedPeriodicFlowSampler(
            _fixed_support(),
            AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4),
            _score(SITES),
            score_id="sites",
            species_predictor_semantic_id="undeclared-law",
        )


def test_species_denoiser_parameters_bind_exact_restart_and_reuse_executable(
    tmp_path: Path,
) -> None:
    sampler = _composition_sampler(
        lane_chunk=2,
        counting=True,
        predictor=_PreferenceLogits(jnp.asarray(PREFERRED_LOGITS)),
    )
    changed = eqx.tree_at(
        lambda value: value.species_chain.predictor.preferred,
        sampler,
        jnp.asarray(PREFERRED_LOGITS[::-1]),
    )
    assert changed.sampler_id == sampler.sampler_id
    assert (
        changed.numeric_revision().revision_id != sampler.numeric_revision().revision_id
    )
    # A numeric-only change reuses the compiled admission.
    _TRACES.clear()
    _admitted(sampler, (True, False))
    traced = len(_TRACES)
    _admitted(changed, (True, False))
    assert traced > 0
    assert len(_TRACES) == traced

    def prepared(
        member: PreparedPeriodicDiffusionSampler,
    ) -> PreparedAtomisticGenerationStream:
        return AtomisticGenerationStreamPlan(
            member,
            campaign_id="periodic-species-restart",
            lanes_per_member=2,
            pending_capacity=4,
        ).prepare()

    stream = prepared(sampler)
    reference = AtomisticMemoryGenerationSink("uninterrupted")
    stream.run(
        stream.start(jr.key(5), target=6, attempt_budget=6),
        reference,
        maximum_transitions=40,
    )
    state = stream.start(jr.key(5), target=6, attempt_budget=6)
    state = stream.segment(state, maximum_transitions=2)
    # Pending and active work drawn under the original coefficients is retained.
    assert int(state.cursor.next_ordinal) > int(state.cursor.watermark)
    path = tmp_path / "periodic-species.npz"
    stream.write_checkpoint(path, state)

    refused = prepared(changed)
    assert refused.stream_id == stream.stream_id
    assert refused.identity()["revisions"] != stream.identity()["revisions"]
    with pytest.raises(ValueError, match="caller-prepared structure"):
        refused.read_checkpoint(path)

    identical = prepared(
        _composition_sampler(
            lane_chunk=2,
            counting=True,
            predictor=_PreferenceLogits(jnp.asarray(PREFERRED_LOGITS)),
        )
    )
    assert identical.identity() == stream.identity()
    sink = AtomisticMemoryGenerationSink("interrupted")
    resumed = identical.run(identical.read_checkpoint(path), sink, maximum_transitions=40)

    assert identical.finished(resumed)
    assert [_record(item) for item in sink.records] == [
        _record(item) for item in reference.records
    ]
    for left, right in zip(sink.records, reference.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)


class _AffineShift(StrictModule):
    """Parameter-bearing condition map ``site_shift -> scale site_shift + offset``."""

    scale: jax.Array
    offset: jax.Array

    def __call__(
        self, values: Mapping[str, jax.Array], keys: jax.Array
    ) -> dict[str, jax.Array]:
        del keys
        return {"site_shift": self.scale * values["site_shift"] + self.offset}


def test_condition_source_and_transform_parameters_bind_exact_restart(
    tmp_path: Path,
) -> None:
    spec = _shift_spec()
    sampler = PreparedPeriodicDiffusionSampler(
        _fixed_support(conditioning=spec),
        AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4),
        _shifted_score(SITES),
        score_id="analytic-translated-wrapped-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
        lane_chunk=2,
    )
    shifts = np.asarray([0.0, 0.25, -0.2, 0.3])

    def prepared(rows: np.ndarray, offset: float) -> PreparedAtomisticGenerationStream:
        transform = AtomisticConditionTransform(
            spec,
            spec,
            _AffineShift(jnp.asarray(1.0), jnp.asarray(offset)),
            transform_id="declared-site-offset",
        )
        return AtomisticGenerationStreamPlan(
            sampler,
            campaign_id="periodic-condition-restart",
            lanes_per_member=2,
            pending_capacity=4,
            conditions=AtomisticConditionSchedule(
                spec, {"site_shift": rows}, policy="bounded"
            ),
            transforms=(transform,),
        ).prepare()

    stream = prepared(shifts, 0.1)
    reference = AtomisticMemoryGenerationSink("uninterrupted")
    stream.run(
        stream.start(jr.key(3), target=4, attempt_budget=4),
        reference,
        maximum_transitions=40,
    )
    assert [item.ordinal for item in reference.records] == [0, 1, 2, 3]
    for proposal, shift in zip(reference.records, shifts, strict=True):
        # Independent control: the executed transform translated the sites.
        fractional = proposal.value("fractional_positions")
        assert proposal.status == 0
        assert _minimum_image_error(fractional[:2], SITES[:2] + shift + 0.1) < 0.05

    state = stream.start(jr.key(3), target=4, attempt_budget=4)
    state = stream.segment(state, maximum_transitions=1)
    assert int(state.cursor.next_ordinal) > int(state.cursor.watermark)
    path = tmp_path / "periodic-conditions.npz"
    stream.write_checkpoint(path, state)

    # Other transform coefficients or condition rows keep the semantic stream
    # identity but change its numeric revision, so the continuation conditioned
    # under the old values is refused.
    for changed in (prepared(shifts, 0.2), prepared(shifts + 0.05, 0.1)):
        assert changed.stream_id == stream.stream_id
        assert changed.identity()["revisions"] != stream.identity()["revisions"]
        with pytest.raises(ValueError, match="caller-prepared structure"):
            changed.read_checkpoint(path)

    identical = prepared(shifts.copy(), 0.1)
    assert identical.identity() == stream.identity()
    sink = AtomisticMemoryGenerationSink("interrupted")
    resumed = identical.run(identical.read_checkpoint(path), sink, maximum_transitions=40)

    assert identical.finished(resumed)
    assert [_record(item) for item in sink.records] == [
        _record(item) for item in reference.records
    ]
    for left, right in zip(sink.records, reference.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)

    def passthrough(
        values: Mapping[str, jax.Array], keys: jax.Array
    ) -> dict[str, jax.Array]:
        del keys
        return dict(values)

    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        AtomisticConditionTransform(spec, spec, passthrough, transform_id="passthrough")
    declared = AtomisticConditionTransform(
        spec,
        spec,
        passthrough,
        transform_id="passthrough",
        function_semantic_id="identity-condition-map",
        function_numeric_id="no-parameters",
    )
    assert declared.numeric_revision().semantic_id == declared.transform_id
    for hidden in (
        jnp.asarray(0.1),
        jnp.asarray([1, -1], dtype=jnp.int32),
        jr.key(13),
    ):
        with pytest.raises(ValueError, match="hides numerical arrays"):
            AtomisticConditionTransform(
                spec,
                spec,
                _capturing_transform(hidden),
                transform_id="hidden-state-map",
                function_semantic_id="hidden-state-map",
                function_numeric_id="hidden-state",
            )
