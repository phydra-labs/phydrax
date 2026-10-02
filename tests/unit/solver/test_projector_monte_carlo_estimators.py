#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from phydrax._strict import StrictModule
from phydrax.operators.quantum._amplitude import LogAmplitude
from phydrax.operators.quantum.lattice._address import (
    QuantumAddress,
    QuantumConfigurationDomain,
)
from phydrax.operators.quantum.lattice._column import QuantumLatticeColumnOperator
from phydrax.operators.quantum.lattice._column_compile import (
    prepare_quantum_lattice_columns,
    QuantumColumnResourcePolicy,
)
from phydrax.operators.quantum.lattice._guide import QuantumGuide
from phydrax.operators.quantum.lattice._model import (
    LocalOperatorPlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.solver._projector_monte_carlo import (
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
)
from phydrax.solver._projector_monte_carlo_contracts import (
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloHistory,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloState,
)
from phydrax.solver._projector_monte_carlo_estimators import (
    analyze_projector_monte_carlo,
    ProjectorEstimatorPolicy,
    ProjectorWeightStatus,
)
from phydrax.solver._projector_monte_carlo_observables import observe_projector_state
from phydrax.typing import Float64, Scalar
from phydrax.units import derived_unit, JOULE, ONE
from phydrax.uq._correlated_ratio import CorrelatedRatioPolicy, CorrelatedRatioStatus
from tests._support.projector_monte_carlo import (
    complex_flux_control,
    control_keys,
    control_operator,
    fermionic_exterior_control,
    ProjectorFiniteControl,
    two_boson_two_site_control,
)


class _PositiveCoordinateGuide(StrictModule):
    __strict_contract__ = True

    domain: QuantumConfigurationDomain
    offset: Float64[Scalar]
    slope: Float64[Scalar]

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        coordinate = self.domain.decode(address.key_words)
        return LogAmplitude(self.offset + self.slope * coordinate[0].astype(jnp.float64))


def _guide(operator: QuantumLatticeColumnOperator, *, varying: bool) -> QuantumGuide:
    provider = _PositiveCoordinateGuide(
        domain=operator.domain,
        offset=jnp.log(jnp.asarray(3, dtype=jnp.float64)),
        slope=jnp.log(jnp.asarray(2 if varying else 1, dtype=jnp.float64)),
    )
    return QuantumGuide(
        operator.domain,
        provider,
        provider_id="positive-test-coordinate-snapshot",
        mapping_id="canonical-left-occupation",
        globally_positive=True,
    )


def _prepare(
    control: ProjectorFiniteControl,
    *,
    guide: QuantumGuide | None = None,
    operator: QuantumLatticeColumnOperator | None = None,
    coefficients: Array | None = None,
    observables: tuple[QuantumLatticeColumnOperator, ...] = (),
    replicas: int = 3,
    history_capacity: int = 512,
) -> tuple[PreparedProjectorMonteCarlo, ProjectorMonteCarloState]:
    original = control_operator(control) if operator is None else operator
    problem = ProjectorMonteCarloProblem(
        original,
        control_keys(original, control),
        control.ground_vector if coefficients is None else coefficients,
        dt=0.1,
        energy_unit=JOULE,
        inverse_energy_unit=derived_unit("J^-1", ((JOULE, -1),)),
        provenance_id=control.provenance,
        guide=guide,
        observables=observables,
        observable_units=(ONE,) * len(observables),
    )
    plan = ProjectorMonteCarloPlan(
        replicas=replicas,
        support_capacity=4,
        group_capacity=16,
        event_capacity=8,
        attempt_capacity=8,
        source_capacity=4,
        history_capacity=history_capacity,
        maximum_retained_bytes=10_000_000,
        maximum_workspace_bytes=10_000_000,
    )
    prepared = prepare_projector_monte_carlo(problem, plan)
    return prepared, initialize_projector_monte_carlo(prepared, jr.key(123))


def _manufactured_history(
    state: ProjectorMonteCarloState,
    projected_numerator: Array,
    projected_denominator: Array,
    *,
    shifts: Array | None = None,
    pair_numerators: Array | None = None,
    pair_denominators: Array | None = None,
) -> ProjectorMonteCarloHistory:
    replicas, draws = projected_numerator.shape
    history = state.history
    pairs = history.pair_denominators.shape[0]
    observables = history.pair_numerators.shape[-1]
    applied = (
        jnp.zeros((replicas, draws), dtype=jnp.float64) if shifts is None else shifts
    )
    pnum = (
        jnp.ones((pairs, draws, observables), dtype=jnp.complex128)
        if pair_numerators is None
        else pair_numerators
    )
    pden = (
        jnp.ones((pairs, draws), dtype=jnp.complex128)
        if pair_denominators is None
        else pair_denominators
    )
    return ProjectorMonteCarloHistory(
        applied_shifts=applied,
        populations=jnp.ones((replicas, draws), dtype=jnp.float64),
        projected_numerator=projected_numerator.astype(jnp.complex128),
        projected_denominator=projected_denominator.astype(jnp.complex128),
        pair_numerators=pnum.astype(jnp.complex128),
        pair_denominators=pden.astype(jnp.complex128),
        pre_annihilation_norm=jnp.ones((replicas, draws), dtype=jnp.float64),
        post_annihilation_norm=jnp.ones((replicas, draws), dtype=jnp.float64),
        valid=jnp.ones((draws,), dtype=jnp.bool_),
        count=jnp.asarray(draws, dtype=jnp.int64),
        scientific_id=history.scientific_id,
        domain_id=history.domain_id,
        operator_id=history.operator_id,
        guide_id=history.guide_id,
        metric_id=history.metric_id,
    )


@pytest.mark.parametrize("kind", ("boson", "fermion", "flux"))
def test_physical_observer_recovers_independent_analytic_ground_energy(kind: str) -> None:
    match kind:
        case "boson":
            control = two_boson_two_site_control()
        case "fermion":
            control = fermionic_exterior_control()
        case "flux":
            control = complex_flux_control()
        case _:
            raise ValueError(kind)
    prepared, state = _prepare(control)
    observation = observe_projector_state(prepared, state)
    assert bool(observation.finite)
    np.testing.assert_allclose(
        observation.projected_numerator / observation.projected_denominator,
        control.ground_energy,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        observation.pair_numerators[:, 0] / observation.pair_denominators,
        control.ground_energy,
        atol=1e-12,
    )


def test_original_outgoing_orientation_preserves_complex_flux_contraction() -> None:
    control = complex_flux_control()
    physical = np.asarray((0.5 + 0.2j, -0.3 + 0.7j, 1.2 - 0.4j), dtype=np.complex128)
    prepared, state = _prepare(control, coefficients=jnp.asarray(physical))
    observation = observe_projector_state(prepared, state)
    expected = np.vdot(physical, control.matrix @ physical)
    assert bool(observation.finite)
    np.testing.assert_allclose(observation.projected_numerator, expected, atol=1e-12)
    np.testing.assert_allclose(observation.pair_numerators[:, 0], expected, atol=1e-12)


def test_absent_trial_support_is_a_physical_sparse_zero_not_operator_failure() -> None:
    control = two_boson_two_site_control()
    prepared, state = _prepare(control)
    trial = jnp.asarray((1, 0, 0), dtype=jnp.complex128)
    prepared = eqx.tree_at(
        lambda p: (p.trial_coefficients, p.trial_active),
        prepared,
        (trial, jnp.asarray((True, False, False), dtype=jnp.bool_)),
    )
    observation = observe_projector_state(prepared, state)
    expected = control.matrix[0] @ control.ground_vector
    assert bool(observation.finite)
    np.testing.assert_allclose(observation.projected_numerator, expected)
    np.testing.assert_allclose(
        observation.projected_denominator, control.ground_vector[0]
    )


@pytest.mark.parametrize(
    "guided", (False, True), ids=("physical", "nonconstant-positive-guide")
)
def test_general_complex_observable_keeps_its_imaginary_physical_value(
    guided: bool,
) -> None:
    control = two_boson_two_site_control()
    original = control_operator(control)
    number = LocalOperatorPlan(
        control.specification.spaces[0],
        "left-number",
        np.diag(np.asarray((0, 1, 2), dtype=np.complex128)),
        (0,),
    )
    specification = QuantumLatticeSpecification(
        control.specification.spaces,
        (QuantumLatticeTerm((number,), coefficient=1j, label="imaginary-number"),),
    )
    resources = QuantumColumnResourcePolicy(
        maximum_monomials=8,
        maximum_factors_per_monomial=4,
        maximum_raw_routes=64,
        maximum_column_targets=64,
        maximum_workspace_bytes=1_000_000,
    )
    observable = QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(specification, resources), original.domain
    )
    prepared, state = _prepare(
        control,
        operator=original,
        guide=_guide(original, varying=True) if guided else None,
        observables=(observable,),
    )
    observation = observe_projector_state(prepared, state)
    expected = np.vdot(
        control.ground_vector,
        np.asarray((2j, 1j, 0j), dtype=np.complex128) * control.ground_vector,
    )
    assert bool(observation.finite)
    assert expected.imag > 0
    np.testing.assert_allclose(observation.pair_numerators[:, 1], expected, atol=1e-12)


def test_projected_estimate_is_ratio_of_means_not_mean_of_instantaneous_ratios() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    numerator = jnp.asarray(((1, 9, 1, 9), (1, 9, 1, 9)), dtype=jnp.float64)
    denominator = jnp.asarray(((1, 3, 1, 3), (1, 3, 1, 3)), dtype=jnp.float64)
    history = _manufactured_history(state, numerator, denominator)
    analysis = analyze_projector_monte_carlo(
        prepared, history, policy=ProjectorEstimatorPolicy(deterministic_records=True)
    )
    np.testing.assert_allclose(analysis.projected.exploratory_value, 2.5)
    assert not np.isclose(
        complex(analysis.projected.exploratory_value),
        np.mean(np.asarray(numerator / denominator)),
    )


def test_zero_projected_denominator_remains_recorded_and_statistically_unsafe() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    numerator = jnp.ones((2, 4), dtype=jnp.complex128)
    denominator = jnp.zeros_like(numerator)
    history = _manufactured_history(state, numerator, denominator)
    analysis = analyze_projector_monte_carlo(
        prepared, history, policy=ProjectorEstimatorPolicy(deterministic_records=True)
    )
    assert not bool(analysis.projected.statistically_valid)
    assert int(analysis.projected.status) & int(CorrelatedRatioStatus.DENOMINATOR_UNSAFE)
    assert bool(jnp.all(history.projected_denominator == 0))


def test_finite_history_uses_incoming_per_step_shifts_before_measured_state() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    x = jnp.asarray(((1, 2, 4, 8, 16), (3, 5, 7, 9, 11)), dtype=jnp.float64)
    y = jnp.ones_like(x)
    shifts = jnp.asarray(((1, 3, 7, 15, 31), (-2, -4, -8, -16, -32)), dtype=jnp.float64)
    history = _manufactured_history(state, x, y, shifts=shifts)
    policy = ProjectorEstimatorPolicy(
        burn_in=1,
        cadence=2,
        history_depths=(0, 1, 4),
        reference_energy=2,
        deterministic_records=True,
        minimum_weight_ess=1,
        minimum_weight_fraction=0,
    )
    analysis = analyze_projector_monte_carlo(prepared, history, policy=policy)
    np.testing.assert_array_equal(analysis.measured_state_numbers, (2, 4))
    h0, h1, h4 = analysis.reweighted
    np.testing.assert_allclose(
        h0.projected.exploratory_value, np.mean(np.asarray(x)[:, (1, 3)])
    )
    np.testing.assert_allclose(h0.projected_weights.effective_samples, 2)
    np.testing.assert_allclose(h0.projected_weights.effective_fraction, 1)
    assert bool(h0.projected_weights.successful)
    weights1 = np.exp(-0.1 * (np.asarray(shifts)[:, (1, 3)] - 2))
    expected1 = np.sum(weights1 * np.asarray(x)[:, (1, 3)]) / np.sum(weights1)
    np.testing.assert_allclose(h1.projected.exploratory_value, expected1)
    # h=4 excludes state 2; state 4 uses shifts 0..3, including pre-burn-in.
    weights4 = np.exp(-0.1 * np.sum(np.asarray(shifts)[:, :4] - 2, axis=1))
    expected4 = np.sum(weights4 * np.asarray(x)[:, 3]) / np.sum(weights4)
    np.testing.assert_array_equal(h4.measured_state_numbers, (4,))
    np.testing.assert_allclose(h4.projected.exploratory_value, expected4)
    assert h4.projected_weights.excluded_incomplete_windows == 1
    assert h4.history_horizon == pytest.approx(0.4)


def test_insufficient_complete_history_is_not_silently_shortened() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    history = _manufactured_history(state, data, data)
    analysis = analyze_projector_monte_carlo(
        prepared,
        history,
        policy=ProjectorEstimatorPolicy(history_depths=(5,), deterministic_records=True),
    )
    estimate = analysis.reweighted[0]
    assert int(estimate.projected_weights.status) & int(
        ProjectorWeightStatus.INSUFFICIENT_HISTORY
    )
    assert estimate.projected_weights.excluded_incomplete_windows == 4
    assert not bool(estimate.projected_statistically_valid)


def test_weight_concentration_is_separate_from_ratio_denominator_qualification() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 8), dtype=jnp.float64)
    shifts = jnp.asarray(((0, 100, 100, 100, 100, 100, 100, 100),) * 2, dtype=jnp.float64)
    history = _manufactured_history(state, data, data, shifts=shifts)
    analysis = analyze_projector_monte_carlo(
        prepared,
        history,
        policy=ProjectorEstimatorPolicy(
            history_depths=(1,), deterministic_records=True, minimum_weight_ess=2
        ),
    )
    estimate = analysis.reweighted[0]
    assert int(estimate.projected_weights.status) & int(
        ProjectorWeightStatus.CONCENTRATED
    )
    assert bool(estimate.projected.statistically_valid)
    assert not bool(estimate.projected_statistically_valid)


def test_constant_guide_three_recovers_one_ninth_physical_replica_metric() -> None:
    control = two_boson_two_site_control()
    original = control_operator(control)
    guide = _guide(original, varying=False)
    prepared, state = _prepare(
        control,
        operator=original,
        guide=guide,
        coefficients=jnp.asarray(control.ground_vector / 3),
    )
    observation = observe_projector_state(prepared, state)
    np.testing.assert_allclose(jnp.sum(jnp.abs(state.coefficients[0]) ** 2), 1)
    np.testing.assert_allclose(observation.pair_denominators, 1 / 9, atol=1e-12)
    np.testing.assert_allclose(
        observation.pair_numerators[:, 0], control.ground_energy / 9, atol=1e-12
    )
    assert bool(observation.finite)


def test_zero_physical_overlap_is_finite_observation_not_repaired_denominator() -> None:
    control = two_boson_two_site_control()
    prepared, state = _prepare(control, replicas=2)
    coefficients = jnp.asarray(((1, 0, 0, 0), (0, 0, 1, 0)), dtype=jnp.complex128)
    state = eqx.tree_at(lambda s: s.coefficients, state, coefficients)
    observation = observe_projector_state(prepared, state)
    assert bool(observation.finite)
    np.testing.assert_array_equal(observation.pair_denominators, (0, 0))


def test_operator_numerical_failure_is_reported_even_with_empty_bra_support() -> None:
    control = two_boson_two_site_control()
    prepared, state = _prepare(control)
    broken = eqx.tree_at(
        lambda op: op.prepared.monomials[0].coefficient,
        prepared.original_operator,
        jnp.asarray(jnp.nan, dtype=jnp.complex128),
    )
    prepared = eqx.tree_at(
        lambda p: (p.original_operator, p.trial_active),
        prepared,
        (broken, jnp.zeros_like(prepared.trial_active)),
    )
    observation = observe_projector_state(prepared, state)
    assert not bool(observation.finite)


def test_guide_numerical_failure_is_not_hidden_as_missing_physical_support() -> None:
    control = two_boson_two_site_control()
    original = control_operator(control)
    guide = _guide(original, varying=False)
    prepared, state = _prepare(control, operator=original, guide=guide)
    provider = _PositiveCoordinateGuide(
        domain=original.domain,
        offset=jnp.asarray(jnp.nan, dtype=jnp.float64),
        slope=jnp.asarray(0, dtype=jnp.float64),
    )
    broken = eqx.tree_at(lambda g: g.provider, guide, provider)
    prepared = eqx.tree_at(lambda p: p.guide, prepared, broken)
    assert not bool(observe_projector_state(prepared, state).finite)


def test_ordered_pair_reuse_retains_cross_covariance_in_one_time_stream() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=3)
    rng = np.random.default_rng(781)
    signals = rng.normal(size=(3, 512))
    x = np.stack(tuple(2 + signals[a] + signals[b] for a, b in prepared.pair_ids))
    y = np.stack(tuple(1 + 0.02 * signals[a] * signals[b] for a, b in prepared.pair_ids))
    history = _manufactured_history(
        state,
        jnp.asarray(2 + signals),
        jnp.ones((3, 512), dtype=jnp.float64),
        pair_numerators=jnp.asarray(x[:, :, None]),
        pair_denominators=jnp.asarray(y),
    )
    analysis = analyze_projector_monte_carlo(
        prepared,
        history,
        policy=ProjectorEstimatorPolicy(ratio_policy=CorrelatedRatioPolicy(max_lag=128)),
    )
    result = analysis.replicas[0]
    assert bool(result.statistically_valid)
    ids = np.asarray(result.block_index)[0]
    block_count = int(np.asarray(result.block_counts)[0])
    aggregate = np.stack(
        (np.sum(x, axis=0), np.zeros(512), np.sum(y, axis=0), np.zeros(512)), axis=1
    )
    block_means = np.stack(
        tuple(np.mean(aggregate[ids == k], axis=0) for k in range(block_count))
    )
    centered = block_means - np.mean(block_means, axis=0)
    expected_covariance = centered.T @ centered / (block_count * (block_count - 1))
    np.testing.assert_allclose(result.mean_covariance, expected_covariance, atol=1e-12)
    np.testing.assert_allclose(
        result.exploratory_value, np.sum(x) / np.sum(y), atol=1e-12
    )
    per_pair_blocks = np.stack(
        tuple(np.mean(x[:, ids == k], axis=1) for k in range(block_count))
    )
    per_pair_centered = per_pair_blocks - np.mean(per_pair_blocks, axis=0)
    false_independent_variance = np.sum(per_pair_centered * per_pair_centered) / (
        block_count * (block_count - 1)
    )
    assert (
        float(np.asarray(result.mean_covariance)[0, 0]) > 2 * false_independent_variance
    )


def test_stochastic_constant_history_does_not_fabricate_zero_uncertainty() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 32), dtype=jnp.float64)
    history = _manufactured_history(state, 2 * data, data)
    analysis = analyze_projector_monte_carlo(
        prepared, history, policy=ProjectorEstimatorPolicy()
    )
    assert int(analysis.projected.status) & int(
        CorrelatedRatioStatus.ZERO_VARIATION_UNRESOLVED
    )
    assert not bool(analysis.projected.statistically_valid)


def test_underflowed_history_weights_refuse_numerical_range_without_clipping() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    shifts = jnp.asarray(((0, 10000, 10000, 10000),) * 2, dtype=jnp.float64)
    history = _manufactured_history(state, data, data, shifts=shifts)
    analysis = analyze_projector_monte_carlo(
        prepared,
        history,
        policy=ProjectorEstimatorPolicy(history_depths=(1,), deterministic_records=True),
    )
    estimate = analysis.reweighted[0]
    assert int(estimate.projected_weights.status) & int(
        ProjectorWeightStatus.NUMERICAL_RANGE
    )
    assert not bool(estimate.projected_statistically_valid)


def test_history_with_missing_applied_shift_record_refuses_analysis() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    history = _manufactured_history(state, data, data)
    history = eqx.tree_at(
        lambda h: h.valid,
        history,
        jnp.asarray((True, False, True, True), dtype=jnp.bool_),
    )
    with pytest.raises(ValueError, match="without gaps"):
        analyze_projector_monte_carlo(
            prepared, history, policy=ProjectorEstimatorPolicy()
        )


def test_pair_history_weights_multiply_replica_weights_before_time_aggregation() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=3)
    shifts = np.asarray(((1, 2, 3, 4), (2, 5, 1, 6), (-1, 0, -3, 2)), dtype=np.float64)
    x = np.asarray(
        tuple(
            np.arange(1, 5, dtype=np.float64) * (a + 1) + b for a, b in prepared.pair_ids
        )
    )
    y = np.asarray(
        tuple(np.full(4, 1 + 0.1 * a, dtype=np.float64) for a, _ in prepared.pair_ids)
    )
    history = _manufactured_history(
        state,
        jnp.ones((3, 4), dtype=jnp.float64),
        jnp.ones((3, 4), dtype=jnp.float64),
        shifts=jnp.asarray(shifts),
        pair_numerators=jnp.asarray(x[:, :, None]),
        pair_denominators=jnp.asarray(y),
    )
    policy = ProjectorEstimatorPolicy(
        history_depths=(1,), deterministic_records=True, minimum_weight_ess=1
    )
    estimate = analyze_projector_monte_carlo(prepared, history, policy=policy).reweighted[
        0
    ]
    pair_weights = np.exp(
        np.asarray(tuple(-0.1 * (shifts[a] + shifts[b]) for a, b in prepared.pair_ids))
    )
    expected = np.sum(pair_weights * x) / np.sum(pair_weights * y)
    np.testing.assert_allclose(estimate.replicas[0].exploratory_value, expected)


def test_foreign_scientific_history_is_rejected_before_numerical_analysis() -> None:
    prepared, state = _prepare(two_boson_two_site_control(interaction=2), replicas=2)
    other, _ = _prepare(two_boson_two_site_control(interaction=3), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    history = _manufactured_history(state, data, data)
    with pytest.raises(ValueError, match="identity mismatch"):
        analyze_projector_monte_carlo(other, history, policy=ProjectorEstimatorPolicy())


def test_shift_mean_is_not_gated_by_physical_overlap_magnitude_floor() -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    history = _manufactured_history(state, data, data, shifts=3 * data)
    policy = ProjectorEstimatorPolicy(
        ratio_policy=CorrelatedRatioPolicy(minimum_denominator_magnitude=2),
        deterministic_records=True,
    )
    analysis = analyze_projector_monte_carlo(prepared, history, policy=policy)
    np.testing.assert_allclose(analysis.shifts.value, 3)
    assert bool(analysis.shifts.statistically_valid)
    assert not bool(analysis.projected.statistically_valid)


@pytest.mark.parametrize("reference_energy", (0.0, 100.0))
def test_normalized_history_denominator_floor_is_independent_of_weight_scale(
    reference_energy: float,
) -> None:
    prepared, state = _prepare(two_boson_two_site_control(), replicas=2)
    data = jnp.ones((2, 4), dtype=jnp.float64)
    shifts = jnp.asarray(((1, 3, 5, 7),) * 2, dtype=jnp.float64)
    history = _manufactured_history(state, 2 * data, data, shifts=shifts)
    policy = ProjectorEstimatorPolicy(
        ratio_policy=CorrelatedRatioPolicy(minimum_denominator_magnitude=0.9),
        history_depths=(1,),
        reference_energy=reference_energy,
        deterministic_records=True,
        minimum_weight_ess=1,
    )
    estimate = analyze_projector_monte_carlo(prepared, history, policy=policy).reweighted[
        0
    ]
    np.testing.assert_allclose(estimate.projected.denominator_mean, 1)
    np.testing.assert_allclose(estimate.projected.value, 2)
    assert bool(estimate.projected_statistically_valid)
