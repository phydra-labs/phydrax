#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.solver._deterministic_ensemble import (
    ClassicalStatisticalScalarRecipe,
    DeterministicEnsemblePlan,
    DeterministicInitialCondition,
    execute_deterministic_ensemble,
    MMSTInitialConditionRecipe,
    WeightedEnsembleReducer,
)
from phydrax.solver._differential import DifferentialProblem


def _zero_drift(time, state, args):
    del time, args
    return jnp.zeros_like(state)


def test_qft_deterministic_ensemble_preserves_paths_and_weighted_moments():
    problem = DifferentialProblem(
        _zero_drift,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        problem_id="qft-deterministic-zero",
    )
    paths = (
        DeterministicInitialCondition([1.0], 1.0, "low-amplitude"),
        DeterministicInitialCondition([3.0], 3.0, "high-amplitude"),
    )
    prepared = DeterministicEnsemblePlan(problem, paths).prepare([0.0, 0.5, 1.0])
    result = execute_deterministic_ensemble(prepared)

    np.testing.assert_allclose(result.mean(), [[2.5], [2.5], [2.5]], atol=1e-10)
    np.testing.assert_allclose(
        result.reducer.variance(result.states[:, -1, 0]), 0.75, atol=1e-10
    )
    np.testing.assert_allclose(result.reducer.effective_sample_size, 1.6)
    assert bool(result.evidence.successful)
    assert len(set(result.path_ids)) == 2
    assert result.path_ids == prepared.plan.path_ids
    with pytest.raises(MemoryError, match="maximum_output_bytes"):
        DeterministicEnsemblePlan(problem, paths, maximum_output_bytes=1).prepare(
            [0.0, 1.0]
        )


def test_qft_weighted_reducer_masks_invalid_paths_without_changing_shape():
    reducer = WeightedEnsembleReducer([1.0, 2.0, 100.0], [True, True, False])
    values = jnp.asarray([[1.0, 2.0], [4.0, 8.0], [jnp.nan, jnp.nan]])

    np.testing.assert_allclose(reducer.mean(values), [3.0, 6.0])
    np.testing.assert_allclose(reducer.effective_sample_size, 1.8)
    assert bool(reducer.successful)


def test_qft_scalar_and_mmst_recipes_encode_declared_second_moments():
    coordinates = jnp.asarray(
        (
            ((-1.0,), (-1.0,)),
            ((-1.0,), (1.0,)),
            ((1.0,), (-1.0,)),
            ((1.0,), (1.0,)),
        )
    )
    scalar = ClassicalStatisticalScalarRecipe(
        [2.0], coordinates, jnp.ones((4,)), occupations=[0.5]
    )
    scalar_states = jnp.stack(
        tuple(path.initial_state for path in scalar.initial_conditions())
    )
    reducer = WeightedEnsembleReducer(jnp.ones((4,)))
    np.testing.assert_allclose(reducer.mean(scalar_states), 0.0, atol=1e-12)
    np.testing.assert_allclose(reducer.variance(scalar_states[:, 0, 0]), 0.5)
    np.testing.assert_allclose(reducer.variance(scalar_states[:, 1, 0]), 2.0)
    changed = ClassicalStatisticalScalarRecipe(
        [3.0], 2.0 * coordinates, 2.0 * jnp.ones((4,)), occupations=[1.5]
    )
    scalar_paths = scalar.initial_conditions()
    changed_paths = changed.initial_conditions()
    assert tuple(path.semantic_path_id for path in scalar_paths) == tuple(
        path.semantic_path_id for path in changed_paths
    )
    assert tuple(path.realization_id for path in scalar_paths) != tuple(
        path.realization_id for path in changed_paths
    )

    mmst = MMSTInitialConditionRecipe([0.25, 0.75], [1.0], [-0.5])
    mmst_paths = mmst.initial_conditions()
    mapping_actions = jnp.stack(
        tuple(
            0.5
            * (
                path.initial_state[2] ** 2
                + path.initial_state[3] ** 2
                - mmst.zero_point_parameter
            )
            for path in mmst_paths
        )
    )
    np.testing.assert_allclose(mapping_actions, jnp.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        WeightedEnsembleReducer(mmst.populations).mean(mapping_actions),
        mmst.populations,
        atol=1e-12,
    )
