#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _affine_map_problem():
    generator = np.random.default_rng(62)
    cases = 10
    capacity = 9
    states = np.empty((cases, capacity, 1))
    states[:, 0, 0] = generator.uniform(-1.0, 1.0, cases)
    for step in range(capacity - 1):
        states[:, step + 1, 0] = 0.4 + 0.7 * states[:, step, 0]
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        np.broadcast_to(np.arange(capacity), (cases, capacity)),
        states,
        state_layout=layout,
        case_axes=("initial_condition",),
        case_axis_roles=("case",),
        source_id="affine-map-selection",
    )
    return phx.dynamics.identification.SINDyProblem(
        data=data,
        library=phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2),
        formulation=phx.dynamics.identification.DiscreteSINDyFormulation(),
    )


def test_selection_splits_complete_cases_and_retains_invalid_candidates():
    problem = _affine_map_problem()
    regressors = (
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            1e-8, threshold_space="physical"
        ),
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            0.5, threshold_space="physical"
        ),
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            2.0, threshold_space="physical"
        ),
    )

    result = phx.dynamics.identification.select_sindy_model(
        problem,
        regressors,
        policy=phx.dynamics.identification.SINDySelectionPolicy(
            criterion="combined",
            validation_fraction=0.3,
            rollout_horizon=3,
            max_rollouts=6,
            complexity_weight=1e-8,
            seed=7,
        ),
    )

    assert bool(result.valid)
    assert int(result.selected_index) == 0
    assert not bool(result.candidate_valid[-1])
    train_cases = set(
        np.asarray(result.candidates[0].design.case_index[result.train_mask]).tolist()
    )
    validation_cases = set(
        np.asarray(
            result.candidates[0].design.case_index[result.validation_mask]
        ).tolist()
    )
    assert train_cases.isdisjoint(validation_cases)
    assert bool(jnp.all(jnp.isfinite(result.rollout_error[:2])))


def test_ensemble_bootstrap_is_deterministic_and_reports_inclusion():
    problem = _affine_map_problem()
    regressor = phx.dynamics.identification.SequentialThresholdedLeastSquares(
        1e-8, threshold_space="physical"
    )

    first = phx.dynamics.identification.fit_ensemble_sindy(
        problem,
        regressor,
        num_members=6,
        sample_fraction=0.7,
        feature_fraction=1.0,
        seed=13,
    )
    second = phx.dynamics.identification.fit_ensemble_sindy(
        problem,
        regressor,
        num_members=6,
        sample_fraction=0.7,
        feature_fraction=1.0,
        seed=13,
    )

    assert bool(first.valid)
    np.testing.assert_allclose(
        np.asarray(first.coefficients), np.asarray(second.coefficients), atol=0.0
    )
    np.testing.assert_allclose(np.asarray(first.inclusion_frequency), 1.0)
    names = problem.library.feature_names
    np.testing.assert_allclose(
        np.asarray(first.coefficient_mean[0, names.index("1")]), 0.4, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(first.coefficient_mean[0, names.index("state:x")]),
        0.7,
        atol=1e-10,
    )


def test_single_trajectory_selection_embargoes_overlapping_weak_windows():
    time = jnp.linspace(0.0, 4.0, 81)
    state = jnp.exp(-time)[:, None]
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    problem = phx.dynamics.identification.SINDyProblem(
        data=phx.dynamics.TrajectoryData(
            time,
            state,
            state_layout=layout,
            source_id="weak-window-selection",
        ),
        library=phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=1),
        formulation=phx.dynamics.identification.WeakSINDyFormulation(window_size=6),
    )

    result = phx.dynamics.identification.select_sindy_model(
        problem,
        (phx.dynamics.identification.SequentialThresholdedLeastSquares(1e-8),),
        policy=phx.dynamics.identification.SINDySelectionPolicy(
            criterion="equation",
            validation_fraction=0.25,
        ),
    )

    design = result.candidates[0].design
    widest_window = int(jnp.max(design.window_end - design.window_start))
    assert result.embargo >= widest_window
    assert int(jnp.max(design.window_end[result.train_mask])) < int(
        jnp.min(design.window_start[result.validation_mask])
    )


def test_sr3_l0_recovers_sparse_fourier_law_with_unbiased_refit():
    state = np.linspace(-np.pi, np.pi, 257, endpoint=False)[:, None]
    derivative = 1.2 * np.sin(state) - 0.7 * np.cos(2.0 * state)
    layout = phx.dynamics.StateLayout((1,), component_names=("angle",))
    data = phx.dynamics.TrajectoryData(
        jnp.arange(state.shape[0], dtype=float),
        state,
        state_layout=layout,
        derivatives=derivative,
        source_id="sr3-fourier",
    )
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=phx.dynamics.identification.FourierFeatureLibrary(
            layout,
            jnp.asarray([[1.0], [2.0]]),
            include_bias=True,
        ),
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )

    result = phx.dynamics.identification.fit_sindy(
        problem,
        phx.dynamics.identification.SR3Regression(
            1e-3,
            relaxation_strength=0.3,
            penalty="l0",
            max_iterations=200,
            tolerance=1e-8,
            unbiased_refit=True,
        ),
    )

    assert bool(result.valid)
    names = result.design.feature_names
    expected = np.zeros((len(names),))
    expected[names.index("sin(1*state:angle)")] = 1.2
    expected[names.index("cos(2*state:angle)")] = -0.7
    np.testing.assert_allclose(np.asarray(result.coefficients[0]), expected, atol=1e-11)
    assert result.regression.solver_diagnostics.objective.shape[0] == 201
