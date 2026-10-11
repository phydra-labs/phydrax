#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.incompressible_flow import (
    align_coupled_ensembles,
    componentwise_empirical_wasserstein,
    CoupledEnsemblePair,
    paired_coupling_upper_bound,
    PeriodicFlowEnsembleResult,
    PeriodicFlowInitialMeasure,
    PeriodicFlowInitialState,
    prepare_periodic_flow_ensemble,
    PreparedPeriodicFlowEnsemble,
    restrict_periodic_flow_initial_state,
    sliced_joint_marginal_distance,
    StructureFunctionScalingPlan,
    TaylorGreenFamily,
    TimeIntegratedStructureFunctionScalingEvidence,
)
from phydrax.applications.incompressible_flow._statistical_evidence import _member_labels
from phydrax.discretization import (
    AxisDomain,
    DiscretizationBundle,
    DiscretizationRecord,
    FourierBasisPlan,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import (
    PeriodicIncrementMomentResult,
    TensorSpectralDiscretization,
)
from phydrax.execution import MemoryBudgetError
from phydrax.solver import (
    FixedStepEnsemblePlan,
    FixedStepEnsembleResult,
    FixedStepProblem,
    SSPRK33FixedStepMethod,
)


def _space(count: int) -> TensorSpectralDiscretization:
    return TensorSpectralPlan((FourierBasisPlan(count),) * 2).prepare(
        (AxisDomain.periodic(0.3, 2.3),) * 2
    )


def _decay(time: Array, state: Array, args: Any, /) -> Array:
    del time, args
    return -state


def _solve(initial: PeriodicFlowInitialState) -> PeriodicFlowEnsembleResult:
    # One step, four unpadded lanes, and two saves: only the evidence substrate
    # is under test, not the incompressible solver's nonlinear compilation.
    problem = FixedStepProblem(
        SSPRK33FixedStepMethod(_decay),
        initial.velocity[0],
        t0=0.0,
        t1=0.01,
        step_size=0.01,
        discretization_bundle=DiscretizationBundle(
            (
                DiscretizationRecord(
                    initial.discretization.key,
                    "TensorSpectralDiscretization",
                    initial.discretization_id,
                ),
            )
        ),
    )
    prepared = FixedStepEnsemblePlan(
        problem,
        initial.velocity,
        _member_labels(initial),
        member_chunk_size=4,
    ).prepare()
    return PreparedPeriodicFlowEnsemble(prepared, initial).execute()


@pytest.fixture(scope="module")
def campaigns() -> tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult]:
    fine = _space(8)
    prepared = PeriodicFlowInitialMeasure(TaylorGreenFamily(2)).prepare(
        fine, member_chunk_size=4
    )
    initial = prepared.realize(jax.random.key(42), np.arange(4, dtype=np.uint32))
    return _solve(restrict_periodic_flow_initial_state(initial, _space(4))), _solve(
        initial
    )


def _reordered(
    evidence: PeriodicFlowEnsembleResult, order: np.ndarray
) -> PeriodicFlowEnsembleResult:
    result = evidence.result
    reordered = FixedStepEnsembleResult(
        result.times,
        jax.tree.map(lambda a: a[order], result.states),
        result.valid[order],
        result.status[order],
        jax.tree.map(lambda a: a[order], result.solutions),
        eqx.tree_at(
            lambda r: (r.weights, r.valid),
            result.reducer,
            (result.reducer.weights[order], result.reducer.valid[order]),
        ),
        result.evidence,
        tuple(result.path_ids[i] for i in order),
        tuple(result.realization_ids[i] for i in order),
        result.prepared_id,
        result.result_id,
        result.discretization_ids,
    )
    initial = eqx.tree_at(
        lambda s: (s.velocity, s.member_indices, s.parameters),
        evidence.initial_state,
        (
            evidence.initial_state.velocity[order],
            evidence.initial_state.member_indices[order],
            jax.tree.map(lambda a: a[order], evidence.initial_state.parameters),
        ),
    )
    return PeriodicFlowEnsembleResult(reordered, initial)


def test_alignment_restores_semantic_member_order(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, fine = campaigns
    order = np.asarray([2, 0, 3, 1], dtype=np.int32)
    aligned = align_coupled_ensembles(coarse, _reordered(fine, order))
    assert aligned.fine.result.result_id != fine.result.result_id
    assert (
        align_coupled_ensembles(coarse, _reordered(fine, order)).fine.result.result_id
        == aligned.fine.result.result_id
    )
    assert aligned.fine.result.path_ids == coarse.result.path_ids
    assert aligned.fine.result.realization_ids == fine.result.realization_ids
    np.testing.assert_array_equal(aligned.fine.result.states, fine.result.states)
    np.testing.assert_array_equal(aligned.fine.result.valid, fine.result.valid)
    np.testing.assert_array_equal(aligned.fine.result.status, fine.result.status)
    np.testing.assert_array_equal(
        aligned.fine.result.reducer.weights, fine.result.reducer.weights
    )
    np.testing.assert_array_equal(
        aligned.fine.initial_state.member_indices, fine.initial_state.member_indices
    )
    for actual, expected in zip(
        jax.tree.leaves(aligned.fine.result.solutions),
        jax.tree.leaves(fine.result.solutions),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "change", ["measure", "root", "paths", "times", "weights", "members"]
)
def test_alignment_refuses_incompatible_campaigns(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
    change: str,
) -> None:
    coarse, fine = campaigns
    result, initial = fine.result, fine.initial_state
    match change:
        case "measure" | "root":
            measure = PeriodicFlowInitialMeasure(
                TaylorGreenFamily(2, amplitude=2.0 if change == "measure" else 1.0)
            ).prepare(initial.discretization, member_chunk_size=4)
            initial = measure.realize(
                jax.random.key(43 if change == "root" else 42),
                np.arange(4, dtype=np.uint32),
            )
        case "paths":
            result = FixedStepEnsembleResult(
                result.times,
                result.states,
                result.valid,
                result.status,
                result.solutions,
                result.reducer,
                result.evidence,
                ("foreign", *result.path_ids[1:]),
                result.realization_ids,
                result.prepared_id,
                result.result_id,
                result.discretization_ids,
            )
        case "times":
            result = eqx.tree_at(lambda r: r.times, result, result.times + 0.001)
        case "weights":
            result = eqx.tree_at(
                lambda r: r.reducer.weights,
                result,
                jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64),
            )
        case "members":
            initial = eqx.tree_at(
                lambda s: s.member_indices, initial, initial.member_indices[::-1]
            )
        case _:
            raise ValueError("Unknown test case.")
    with pytest.raises(ValueError):
        align_coupled_ensembles(coarse, PeriodicFlowEnsembleResult(result, initial))


def _physical_values(grid: TensorSpectralDiscretization, mode: int) -> Array:
    x, y = jnp.meshgrid(
        *(2 * jnp.pi * (axis.nodes - 0.3) / 2 for axis in grid.axes),
        indexing="ij",
    )
    offsets = jnp.asarray([1.0, -2.0, 0.4, 3.0], dtype=jnp.float64)
    return jnp.stack(
        (
            offsets[:, None, None] + jnp.cos(mode * y)[None, :, :],
            -0.6 * offsets[:, None, None] + 0.7 * jnp.sin(mode * x)[None, :, :],
        ),
        axis=-1,
    )


def _with_physical_states(
    evidence: PeriodicFlowEnsembleResult, values: Array
) -> PeriodicFlowEnsembleResult:
    coefficients = jax.vmap(evidence.initial_state.discretization.project)(values)
    states = jnp.repeat(coefficients[:, None], evidence.result.times.size, axis=1)
    return PeriodicFlowEnsembleResult(
        eqx.tree_at(lambda r: r.states, evidence.result, states), evidence.initial_state
    )


def test_componentwise_w1_matches_sorted_native_node_oracle(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, fine = campaigns
    cv = _physical_values(coarse.initial_state.discretization, 1)
    # Mode three is retained only on the fine grid. Its injected values must
    # contribute even though a coarse-band modal truncation would remove it.
    fv = _physical_values(fine.initial_state.discretization, 3)
    pair = align_coupled_ensembles(
        _with_physical_states(coarse, cv), _with_physical_states(fine, fv)
    )
    injected = np.asarray(fv)[:, ::2, ::2, :]
    expected = np.mean(
        np.abs(np.sort(np.asarray(cv), axis=0) - np.sort(injected, axis=0)), axis=0
    ).reshape((-1, 2))
    quadrature = np.asarray(
        coarse.initial_state.discretization.quadrature_weights
    ).reshape(-1)
    lower = componentwise_empirical_wasserstein(pair, node_chunk_size=4)
    np.testing.assert_allclose(lower.per_node_component, expected, atol=2e-13)
    np.testing.assert_allclose(
        lower.value, quadrature @ expected.mean(axis=-1), atol=2e-13
    )
    # The fine high mode is invisible to truncation, but not to native injection.
    truncated = np.broadcast_to(
        np.asarray([1.0, -2.0, 0.4, 3.0])[:, None, None, None]
        * np.asarray([1.0, -0.6])[None, None, None, :],
        np.asarray(cv).shape,
    )
    truncated_w1 = np.mean(
        np.abs(np.sort(np.asarray(cv), axis=0) - np.sort(truncated, axis=0)), axis=0
    ).reshape((-1, 2))
    assert not np.isclose(float(lower.value), quadrature @ truncated_w1.mean(axis=-1))
    upper = paired_coupling_upper_bound(pair, node_chunk_size=4)
    cost = np.mean(np.linalg.norm(np.asarray(cv) - injected, axis=-1), axis=0).reshape(-1)
    np.testing.assert_allclose(upper.per_node_cost, cost, atol=2e-13)
    np.testing.assert_allclose(upper.value, quadrature @ cost, atol=2e-13)
    assert float(upper.value) >= float(lower.value)


def test_identical_campaign_has_zero_for_all_three_distances(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, _ = campaigns
    pair = align_coupled_ensembles(coarse, coarse)
    np.testing.assert_array_equal(componentwise_empirical_wasserstein(pair).value, 0.0)
    np.testing.assert_array_equal(paired_coupling_upper_bound(pair).value, 0.0)
    sliced = sliced_joint_marginal_distance(
        pair, ((0,), (3,)), key=jax.random.key(6), num_projections=8
    )
    np.testing.assert_array_equal(sliced.values, np.zeros(2, dtype=np.float64))
    np.testing.assert_array_equal(
        sliced.projection_distances, np.zeros((2, 8), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        sliced.projection_standard_error, np.zeros(2, dtype=np.float64)
    )


def test_sliced_distance_replays_fixed_key_and_reports_projection_error(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, fine = campaigns
    pair = align_coupled_ensembles(
        _with_physical_states(
            coarse, _physical_values(coarse.initial_state.discretization, 1)
        ),
        _with_physical_states(
            fine, _physical_values(fine.initial_state.discretization, 3)
        ),
    )
    tuples = ((0, 1), (2, 7))
    first = sliced_joint_marginal_distance(
        pair, tuples, key=jax.random.key(5), num_projections=12
    )
    replay = sliced_joint_marginal_distance(
        pair, tuples, key=jax.random.key(5), num_projections=12
    )
    np.testing.assert_array_equal(first.values, replay.values)
    np.testing.assert_array_equal(first.projection_distances, replay.projection_distances)
    np.testing.assert_array_equal(
        first.projection_standard_error, replay.projection_standard_error
    )
    np.testing.assert_allclose(
        first.values, np.mean(first.projection_distances, axis=-1), atol=1e-14
    )
    np.testing.assert_allclose(
        first.projection_standard_error,
        np.std(first.projection_distances, axis=-1, ddof=1) / np.sqrt(12),
        atol=1e-14,
    )
    assert first.node_tuples == tuples
    assert np.all(np.isfinite(first.values))
    assert np.any(np.asarray(first.values) > 0)


@pytest.mark.parametrize("metric", ["componentwise", "paired", "sliced"])
def test_metrics_refuse_invalid_selected_save_without_dropping_member(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
    metric: str,
) -> None:
    coarse, fine = campaigns
    fine = PeriodicFlowEnsembleResult(
        eqx.tree_at(
            lambda r: r.valid, fine.result, fine.result.valid.at[1, -1].set(False)
        ),
        fine.initial_state,
    )
    pair = align_coupled_ensembles(coarse, fine)
    with pytest.raises(ValueError, match="Invalid members"):
        match metric:
            case "componentwise":
                componentwise_empirical_wasserstein(pair, time_index=-1)
            case "paired":
                paired_coupling_upper_bound(pair, time_index=-1)
            case "sliced":
                sliced_joint_marginal_distance(
                    pair, ((0,),), key=jax.random.key(2), num_projections=4, time_index=-1
                )
            case _:
                raise ValueError("Unknown metric.")
    # Invalidity belongs to the selected save, not the entire trajectory.
    assert np.isfinite(
        float(componentwise_empirical_wasserstein(pair, time_index=0).value)
    )


def test_metrics_refuse_boolean_indices(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    pair = align_coupled_ensembles(*campaigns)
    with pytest.raises(TypeError):
        componentwise_empirical_wasserstein(pair, time_index=True)
    with pytest.raises(TypeError):
        sliced_joint_marginal_distance(
            pair,
            ((True,),),
            key=jax.random.key(2),
            num_projections=4,
        )


def _moment(radii: np.ndarray, values: np.ndarray) -> PeriodicIncrementMomentResult:
    members = values.shape[0]
    return PeriodicIncrementMomentResult(
        values=jnp.asarray(values, dtype=jnp.float64),
        forbidden_coefficient_energy=jnp.zeros((members,), dtype=jnp.float64),
        finite=jnp.asarray(np.all(np.isfinite(values), axis=1), dtype=jnp.bool_),
        radii=jnp.asarray(radii, dtype=jnp.float64),
        bin_counts=jnp.zeros((0,), dtype=jnp.int32),
        valid_bins=jnp.zeros((0,), dtype=jnp.bool_),
        exact=True,
        plan_id="synthetic-exact-second-moments",
    )


def test_exact_power_law_structure_function_scaling() -> None:
    radii = np.geomspace(0.3, 0.8, 6, dtype=np.float64)
    constant, exponent = 1.7, 0.65
    values = np.broadcast_to((constant * radii**exponent) ** 2, (4, radii.size)).copy()
    evidence = StructureFunctionScalingPlan((0.3, 0.8)).evaluate(
        (_moment(radii, values), _moment(radii, values)), (_space(16), _space(32))
    )
    for fit in evidence.fits:
        assert fit.status == "complete"
        assert fit.reason == ""
        np.testing.assert_allclose(fit.exponent, exponent, atol=2e-13)
        np.testing.assert_allclose(fit.constant, constant, atol=2e-13)
        np.testing.assert_allclose(fit.standard_error, 0.0, atol=2e-13)
    np.testing.assert_allclose(evidence.common_exponent, exponent, atol=2e-13)
    np.testing.assert_allclose(evidence.envelope_constant, constant, atol=2e-13)
    np.testing.assert_allclose(evidence.paired_exponent_standard_errors, 0.0, atol=2e-13)
    assert evidence.consistent_within_campaign
    assert "not certified" in evidence.nonclaim


def test_member_jackknife_matches_numpy_paired_delete_one_oracle() -> None:
    radii = np.geomspace(0.3, 0.8, 6, dtype=np.float64)
    design = np.column_stack((np.ones(radii.size, dtype=np.float64), np.log(radii)))
    amplitudes = np.asarray([0.7, 1.1, 1.6, 2.2], dtype=np.float64)
    powers = np.asarray([0.2, 0.45, 0.9, 1.2], dtype=np.float64)
    values = (
        (amplitudes[:, None] * radii[None, :] ** powers[:, None]) ** 2,
        ((amplitudes + 0.3)[:, None] * radii[None, :] ** (powers + 0.1)[:, None]) ** 2,
    )
    evidence = StructureFunctionScalingPlan((0.3, 0.8)).evaluate(
        tuple(_moment(radii, v) for v in values), (_space(16), _space(32))
    )
    replicates = np.empty((2, 4), dtype=np.float64)
    for resolution, (value, fit) in enumerate(zip(values, evidence.fits, strict=True)):
        assert fit.status == "complete"
        point = np.linalg.lstsq(design, 0.5 * np.log(np.mean(value, axis=0)), rcond=None)[
            0
        ]
        np.testing.assert_allclose(fit.exponent, point[1], atol=2e-12)
        for deleted in range(4):
            response = 0.5 * np.log(np.mean(np.delete(value, deleted, axis=0), axis=0))
            replicates[resolution, deleted] = np.linalg.lstsq(
                design, response, rcond=None
            )[0][1]
        np.testing.assert_allclose(
            fit.jackknife_exponents, replicates[resolution], atol=2e-12
        )
        se = np.sqrt(
            3 / 4 * np.sum((replicates[resolution] - replicates[resolution].mean()) ** 2)
        )
        np.testing.assert_allclose(fit.standard_error, se, atol=2e-12)
    differences = replicates[:, None, :] - replicates[None, :, :]
    paired_se = np.sqrt(
        3
        / 4
        * np.sum((differences - differences.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    )
    np.testing.assert_allclose(
        evidence.paired_exponent_standard_errors, paired_se, atol=2e-12
    )
    assert paired_se[0, 1] > 0


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("short", "Insufficient radii"),
        ("rank", "Rank-deficient"),
        ("rmin", "r_min"),
        ("members", "At least three"),
    ],
)
def test_scaling_abstains_with_explicit_reason(case: str, reason: str) -> None:
    radii = np.geomspace(0.3, 0.8, 6, dtype=np.float64)
    window = (0.3, 0.8)
    members = 4
    match case:
        case "short":
            window = (0.3, 0.4)
        case "rank":
            radii = np.full((6,), 0.4, dtype=np.float64)
        case "rmin":
            window = (0.1, 0.8)
        case "members":
            members = 2
        case _:
            raise ValueError("Unknown abstention case.")
    values = np.broadcast_to(radii**1.2, (members, radii.size)).copy()
    evidence = StructureFunctionScalingPlan(window).evaluate(
        (_moment(radii, values),), (_space(16),)
    )
    assert evidence.fits[0].status == "abstained"
    assert reason in evidence.fits[0].reason
    assert not evidence.consistent_within_campaign
    assert np.isnan(evidence.common_exponent)
    assert np.isnan(evidence.envelope_constant)


def test_explicit_time_quadrature_integrates_m2_before_fitting() -> None:
    radii = np.geomspace(0.3, 0.8, 6, dtype=np.float64)
    values = np.broadcast_to((1.3 * radii**0.7) ** 2, (4, radii.size)).copy()
    first, second = _moment(radii, values), _moment(radii, 4 * values)
    plan = StructureFunctionScalingPlan((0.3, 0.8))
    evidence = plan.evaluate_time_quadrature(
        ((first, second),), (_space(16),), np.asarray([0.25, 0.5], dtype=np.float64)
    )
    assert isinstance(evidence, TimeIntegratedStructureFunctionScalingEvidence)
    assert evidence.time_quadrature_weights == (0.25, 0.5)
    direct = plan.evaluate((_moment(radii, 2.25 * values),), (_space(16),))
    fit = evidence.fits[0]
    assert fit.status == "complete"
    np.testing.assert_allclose(fit.exponent, 0.7, atol=2e-13)
    np.testing.assert_allclose(fit.constant, 1.3 * np.sqrt(2.25), atol=2e-13)
    np.testing.assert_allclose(fit.constant, direct.fits[0].constant, atol=2e-13)
    np.testing.assert_allclose(
        fit.jackknife_exponents, direct.fits[0].jackknife_exponents, atol=2e-13
    )
    with pytest.raises(ValueError, match="Time quadrature"):
        plan.evaluate_time_quadrature(
            ((first, second),), (_space(16),), np.asarray([1.0], dtype=np.float64)
        )


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_sliced_distance_matches_numpy_projected_sorted_sample_oracle(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
    p: float,
) -> None:
    coarse, fine = campaigns
    cv = _physical_values(coarse.initial_state.discretization, 1)
    fv = _physical_values(fine.initial_state.discretization, 3)
    pair = align_coupled_ensembles(
        _with_physical_states(coarse, cv), _with_physical_states(fine, fv)
    )
    tuples = ((0, 1), (2, 7))
    result = sliced_joint_marginal_distance(
        pair, tuples, key=jax.random.key(5), num_projections=12, p=p
    )
    source = np.asarray(cv).reshape((4, -1, 2))
    target = np.asarray(fv)[:, ::2, ::2, :].reshape((4, -1, 2))
    distances = []
    for nodes, projections in zip(tuples, np.asarray(result.projections), strict=True):
        left = source[:, nodes].reshape((4, -1)) @ projections.T
        right = target[:, nodes].reshape((4, -1)) @ projections.T
        distances.append(
            np.mean(np.abs(np.sort(left, axis=0) - np.sort(right, axis=0)) ** p, axis=0)
            ** (1 / p)
        )
    projected = np.stack(distances)
    costs = projected**p
    value = np.mean(costs, axis=-1) ** (1 / p)
    error = np.std(costs, axis=-1, ddof=1) / np.sqrt(12) / (p * value ** (p - 1))
    np.testing.assert_allclose(result.projection_distances, projected, atol=2e-14)
    np.testing.assert_allclose(result.values, value, atol=2e-14)
    np.testing.assert_allclose(result.projection_standard_error, error, atol=2e-14)


def test_velocity_only_ensemble_refuses_transactional_dynamic_les() -> None:
    from tests.unit.applications.test_incompressible_production import (
        _periodic_dynamic_production_inputs,
    )

    dynamics, method, _, _, _ = _periodic_dynamic_production_inputs()
    initial = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(3))
        .prepare(dynamics.discretization)
        .realize(jax.random.key(4), np.arange(2, dtype=np.uint32))
    )
    with pytest.raises(ValueError, match="Transactional dynamic LES"):
        prepare_periodic_flow_ensemble(
            dynamics, method, initial, start_time=0.0, end_time=0.01, step_size=0.01
        )


def test_identity_pair_avoids_copies_and_direct_misalignment_is_refused(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, fine = campaigns
    pair = align_coupled_ensembles(coarse, fine)
    assert pair.coarse is coarse and pair.fine is fine
    with pytest.raises(ValueError, match="ordered"):
        CoupledEnsemblePair(
            coarse, _reordered(fine, np.asarray([1, 0, 2, 3], dtype=np.int32))
        )


def test_result_refuses_same_shape_foreign_initial_identity(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    _, fine = campaigns
    foreign = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(2, amplitude=2.0))
        .prepare(fine.initial_state.discretization)
        .realize(jax.random.key(42), np.arange(4, dtype=np.uint32))
    )
    with pytest.raises(ValueError, match="semantic paths"):
        PeriodicFlowEnsembleResult(fine.result, foreign)
    different_grid = eqx.tree_at(
        lambda s: s.discretization, fine.initial_state, _space(8)
    )
    # Identical prepared geometry is valid; a different domain is not.
    shifted = TensorSpectralPlan((FourierBasisPlan(8),) * 2).prepare(
        (AxisDomain.periodic(0.4, 2.4),) * 2
    )
    with pytest.raises(ValueError, match="identity"):
        PeriodicFlowEnsembleResult(
            fine.result, eqx.tree_at(lambda s: s.discretization, different_grid, shifted)
        )


def test_folded_shared_nodes_match_full_fine_reconstruction_oracle(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
) -> None:
    coarse, fine = campaigns
    coarse = _with_physical_states(
        coarse, _physical_values(coarse.initial_state.discretization, 1)
    )
    fine = _with_physical_states(
        fine, _physical_values(fine.initial_state.discretization, 3)
    )
    pair = align_coupled_ensembles(coarse, fine)
    actual = paired_coupling_upper_bound(pair, p=2.0, member_chunk_size=1)
    cv = np.asarray(
        jax.vmap(coarse.initial_state.discretization.reconstruct)(
            coarse.result.states[:, -1]
        )
    )
    fv = np.asarray(
        jax.vmap(fine.initial_state.discretization.reconstruct)(fine.result.states[:, -1])
    )[:, ::2, ::2]
    expected = np.mean(np.sum((cv - fv) ** 2, axis=-1), axis=0).reshape(-1)
    np.testing.assert_allclose(actual.per_node_cost, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(MemoryBudgetError) as refusal:
        paired_coupling_upper_bound(pair, maximum_workspace_bytes=1, member_chunk_size=1)
    assert refusal.value.budget == "maximum_workspace_bytes"
    assert refusal.value.required_bytes > refusal.value.limit_bytes


@pytest.mark.parametrize("foreign_coordinates", [False, True])
def test_prepared_etdrk_refuses_missing_or_foreign_coordinates(
    foreign_coordinates: bool,
) -> None:
    from phydrax.discretization import PaddingDealiasingPlan, PseudospectralMethodPlan
    from phydrax.discretization.spectral import HermitianSpectralCoordinates
    from phydrax.equations import (
        compile_periodic_incompressible_flow,
        IncompressibleFlowProblem,
    )
    from phydrax.solver import ETDRKMethod

    space = _space(4)
    dynamics = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(2, 0.0),
        space,
        PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2)),
    )
    initial = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
        .prepare(space)
        .realize(jax.random.key(0), np.arange(2, dtype=np.uint32))
    )
    shifted = TensorSpectralPlan((FourierBasisPlan(4),) * 2).prepare(
        (AxisDomain.periodic(0.4, 2.4),) * 2
    )
    coordinates = (
        HermitianSpectralCoordinates(shifted, component_shape=(2,))
        if foreign_coordinates
        else None
    )
    method = ETDRKMethod(4).prepare(dynamics.semilinear_drift, coordinates=coordinates)
    with pytest.raises(ValueError, match="coordinates"):
        prepare_periodic_flow_ensemble(
            dynamics, method, initial, start_time=0.0, end_time=0.01, step_size=0.01
        )


@pytest.mark.parametrize("metric", ["componentwise", "paired", "sliced"])
def test_metric_workspace_refusal_is_typed_and_covers_shared_values(
    campaigns: tuple[PeriodicFlowEnsembleResult, PeriodicFlowEnsembleResult],
    metric: str,
) -> None:
    pair = align_coupled_ensembles(*campaigns)
    with pytest.raises(MemoryBudgetError) as refusal:
        match metric:
            case "componentwise":
                componentwise_empirical_wasserstein(pair, maximum_workspace_bytes=1)
            case "paired":
                paired_coupling_upper_bound(pair, maximum_workspace_bytes=1)
            case "sliced":
                sliced_joint_marginal_distance(
                    pair,
                    ((0, 1),),
                    key=jax.random.key(1),
                    num_projections=2,
                    maximum_workspace_bytes=1,
                )
    assert refusal.value.budget == "maximum_workspace_bytes"
    assert refusal.value.required_bytes > refusal.value.limit_bytes
