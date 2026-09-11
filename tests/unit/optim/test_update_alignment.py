from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.optim import (
    ConflictFreeUpdatePolicy,
    ConflictFreeUpdateStatistics,
    ConflictFreeUpdateStatus,
    project_conflict_free_direction,
)


def test_projection_is_exactly_inactive_for_feasible_updates_and_jittable():
    gradients = (jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0)))
    proposal = jnp.asarray((1.0, 2.0))

    eager = project_conflict_free_direction(proposal, gradients)
    compiled = jax.jit(lambda value: project_conflict_free_direction(value, gradients))(
        proposal
    )

    assert int(eager.status) == int(ConflictFreeUpdateStatus.ALREADY_FEASIBLE)
    assert not bool(eager.projected)
    assert bool(eager.successful)
    assert jnp.array_equal(eager.direction, proposal)
    assert jnp.array_equal(compiled.direction, proposal)
    np.testing.assert_allclose(eager.multipliers, 0.0, atol=0.0)


def test_exact_active_set_projects_edges_and_opposing_constraints():
    orthant = project_conflict_free_direction(
        jnp.asarray((-1.0, 2.0)),
        (jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))),
    )
    opposing = project_conflict_free_direction(
        jnp.asarray((1.0, 2.0)),
        (jnp.asarray((1.0, 0.0)), jnp.asarray((-1.0, 0.0))),
    )

    np.testing.assert_allclose(orthant.direction, (0.0, 2.0), atol=1e-10)
    np.testing.assert_allclose(opposing.direction, (0.0, 2.0), atol=1e-10)
    assert bool(orthant.successful & opposing.successful)
    assert bool(orthant.projected & opposing.projected)
    assert not bool(jnp.any(orthant.aligned_conflicts))
    assert not bool(jnp.any(opposing.aligned_conflicts))


def test_diagonal_metric_changes_closest_feasible_direction():
    gradient = (jnp.asarray((1.0, 1.0)),)
    proposal = jnp.asarray((-2.0, 0.0))

    euclidean = project_conflict_free_direction(proposal, gradient)
    diagonal = project_conflict_free_direction(
        proposal,
        gradient,
        metric_diagonal=jnp.asarray((1.0, 0.25)),
    )

    np.testing.assert_allclose(euclidean.direction, (-1.0, 1.0), atol=1e-10)
    np.testing.assert_allclose(diagonal.direction, (-1.6, 1.6), atol=1e-10)
    assert diagonal.metric_kind == "diagonal"
    assert not bool(jnp.any(diagonal.aligned_conflicts))
    assert not jnp.isclose(
        diagonal.metric_correction_norm,
        euclidean.metric_correction_norm,
    )


def test_rank_stationarity_activity_and_complex_geometry_are_distinct():
    duplicate = project_conflict_free_direction(
        jnp.asarray((-1.0, 1.0)),
        (jnp.asarray((1.0, 0.0)), jnp.asarray((1.0, 0.0))),
    )
    inactive_nonfinite = project_conflict_free_direction(
        jnp.asarray((1.0, 0.0)),
        (jnp.asarray((1.0, 0.0)), jnp.asarray((jnp.nan, 0.0))),
        active=jnp.asarray((True, False)),
    )
    stationary = project_conflict_free_direction(
        jnp.asarray((1.0, 0.0)),
        (jnp.zeros((2,)),),
    )
    complex_result = project_conflict_free_direction(
        {"z": jnp.asarray((-1.0 - 1.0j,))},
        ({"z": jnp.asarray((1.0 + 1.0j,))},),
    )

    np.testing.assert_allclose(duplicate.direction, (0.0, 1.0), atol=1e-10)
    assert bool(duplicate.successful)
    assert bool(inactive_nonfinite.successful)
    assert jnp.isfinite(inactive_nonfinite.kkt_residual_norm)
    assert int(stationary.status) == int(ConflictFreeUpdateStatus.NO_EFFECTIVE_OBJECTIVES)
    assert bool(stationary.stationary[0])
    np.testing.assert_allclose(complex_result.direction["z"], 0.0, atol=1e-10)
    assert bool(complex_result.pareto_stationary)


def test_large_objective_dual_and_failure_contracts_are_audited():
    gradients = tuple(jnp.eye(4)[index] for index in range(4))
    projected = project_conflict_free_direction(-jnp.ones((4,)), gradients)

    assert projected.solver_method == "dense-primal-dual"
    assert bool(projected.successful)
    np.testing.assert_allclose(projected.direction, 0.0, atol=1e-7)
    assert not bool(jnp.any(projected.aligned_conflicts))

    invalid_metric = project_conflict_free_direction(
        jnp.ones((2,)),
        (jnp.ones((2,)),),
        metric_diagonal=jnp.asarray((1.0, 0.0)),
    )
    assert not bool(invalid_metric.successful)
    assert int(invalid_metric.status) == int(ConflictFreeUpdateStatus.INVALID_METRIC)
    np.testing.assert_allclose(invalid_metric.direction, 0.0, atol=0.0)

    nonfinite = project_conflict_free_direction(
        jnp.ones((1,)),
        (jnp.asarray((jnp.nan,)),),
    )
    assert int(nonfinite.status) == int(ConflictFreeUpdateStatus.NONFINITE)
    with pytest.raises(eqx.EquinoxRuntimeError, match="could not be projected"):
        checked = project_conflict_free_direction(
            jnp.ones((1,)),
            (jnp.asarray((jnp.nan,)),),
            policy=ConflictFreeUpdatePolicy(failure="error"),
        )
        np.asarray(checked.direction)

    with pytest.raises(ValueError, match="structures"):
        project_conflict_free_direction(
            {"x": jnp.ones((1,))},
            ({"y": jnp.ones((1,))},),
        )
    with pytest.raises(ValueError, match="one Boolean"):
        project_conflict_free_direction(
            jnp.ones((1,)),
            (jnp.ones((1,)),),
            active=jnp.ones((2,), dtype=bool),
        )


def test_statistics_preserve_four_stage_mismatch_evidence():
    result = project_conflict_free_direction(
        jnp.asarray((-1.0, 2.0)),
        (jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))),
    )
    statistics = ConflictFreeUpdateStatistics.zeros(jnp.float64).update(
        result,
        gradient_conflict=True,
        constructed_conflict=False,
    )

    assert int(statistics.steps) == 1
    assert float(statistics.gradient_conflict_rate) == 1.0
    assert float(statistics.constructed_conflict_rate) == 0.0
    assert float(statistics.proposal_conflict_rate) == 1.0
    assert float(statistics.applied_conflict_rate) == 0.0
    assert float(statistics.projection_rate) == 1.0
    assert float(statistics.mean_correction_norm) > 0.0
