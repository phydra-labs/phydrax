"""Public static contracts; dtype/rank/typed-key/domain distinctions are runtime."""

from __future__ import annotations

from typing import assert_type

from jax import Array

from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    PreparedProjectorMonteCarlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloAnalysis,
    ProjectorMonteCarloResult,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStepResult,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
)
from phydrax.typing import PRNGKey
from phydrax.uq import CorrelatedRatioPolicy, CorrelatedRatioResult


def public_projector_types(
    prepared: PreparedProjectorMonteCarlo,
    key: PRNGKey,
    result: ProjectorMonteCarloResult,
) -> None:
    state = initialize_projector_monte_carlo(prepared, key)
    assert_type(state, ProjectorMonteCarloState)
    assert_type(state.support_keys, Array)
    assert_type(state.root_key, Array)
    assert_type(
        step_projector_monte_carlo(prepared, state), ProjectorMonteCarloStepResult
    )
    assert_type(
        solve_projector_monte_carlo(prepared, state, steps=1), ProjectorMonteCarloResult
    )
    analysis = analyze_projector_monte_carlo(
        prepared, result, policy=ProjectorEstimatorPolicy()
    )
    assert_type(analysis, ProjectorMonteCarloAnalysis)
    assert_type(analysis.projected, CorrelatedRatioResult)
    assert_type(analysis.projected.mean_covariance, Array)
    initialize_projector_monte_carlo(prepared, 1)  # ty: ignore[invalid-argument-type]
    solve_projector_monte_carlo(prepared, object(), steps=1)  # ty: ignore[invalid-argument-type]
    CorrelatedRatioPolicy(minimum_blocks="few")  # ty: ignore[invalid-argument-type]
    analyze_projector_monte_carlo(prepared, result)  # ty: ignore[missing-argument]
