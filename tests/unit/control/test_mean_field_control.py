#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games._mean_field import (
    FrozenLawBestResponseProblem,
    solve_frozen_law_best_response,
)
from phydrax.control.stochastic._mean_field_control import (
    evaluate_mean_field_control_planner,
    MEAN_FIELD_CONTROL_PLANNER_STATIONARITY,
    MeanFieldControlProblem,
    MeanFieldControlStatus,
    MeanFieldExternality,
)
from phydrax.stochastic import (
    BSDEPathBatch,
    EmpiricalMeanField,
    MeanFieldBSDEControlAdapter,
    MeanFieldBSDEProblem,
)


def _paths(*, path_id="planner-paths", valid=None, particles=None):
    if particles is None:
        particles = jnp.ones((2, 2, 1))
    return BSDEPathBatch(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray(particles),
        jnp.zeros((2, 1, 1)),
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id=path_id,
        process_id="planner-process",
        valid=valid,
    )


def _base_problem(paths, flow, *, terminal_target=0.0):
    adapter = MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, control, args: control.reshape((1,)),
        lambda time, state, law, action, args: jnp.asarray([0.0]),
        lambda time, state, law, action, args: jnp.asarray([0.0]),
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="planner-physical-control",
    )
    return MeanFieldBSDEProblem(
        lambda key: paths,
        flow,
        lambda time, state, law, args: jnp.asarray([0.0]),
        lambda time, state, law, args: jnp.asarray([[1.0]]),
        lambda time, state, law, value, control, args: jnp.asarray([0.0]),
        lambda state, law, args: jnp.asarray([terminal_target]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="planner-bsde",
        process_id="planner-process",
        control_adapter=adapter,
    )


def _analytic_externality(*, running=None, terminal=None):
    if running is None:
        running = lambda time, state, law, value, control, action, args: jnp.asarray(
            [law.mean[0]]
        )
    if terminal is None:
        terminal = lambda state, law, args: jnp.asarray([0.0])
    return MeanFieldExternality(
        running,
        terminal,
        mode="analytic-lions",
        externality_id="social-cost-lions-derivative",
        running_id="analytic-running-lions-integral",
        terminal_id="analytic-terminal-lions-integral",
    )


def _problem(
    paths,
    *,
    flow=None,
    externality=None,
    terminal_target=0.0,
    stationarity=None,
):
    if flow is None:
        flow = EmpiricalMeanField.from_paths(paths, mean_field_id="planner-law")
    if externality is None:
        externality = _analytic_externality()
    if stationarity is None:
        stationarity = lambda time, state, law, value, control, action, args: (
            action + 2.0 * law.mean
        )
    return MeanFieldControlProblem(
        _base_problem(paths, flow, terminal_target=terminal_target),
        externality,
        lambda time, state, law, action, args: (
            0.5 * action[0] ** 2 + 2.0 * law.mean[0] * action[0]
        ),
        lambda state, law, args: state[0] ** 2,
        stationarity,
        welfare_running_id="social-running-objective",
        welfare_terminal_id="social-terminal-objective",
        stationarity_id="social-hamiltonian-control-gradient",
        problem_id="one-period-social-planner",
    )


def _evaluate(problem, paths, *, action=0.0, value=0.0):
    return evaluate_mean_field_control_planner(
        problem,
        paths,
        lambda time, state: jnp.asarray([value]),
        control_predictor=lambda time, state: jnp.asarray([[action]]),
    )


def test_one_period_social_planner_control_differs_from_mfg_control():
    """The population term doubles the mean coupling in the planner FOC."""
    paths = _paths()
    problem = _problem(paths)
    frozen_mfg = FrozenLawBestResponseProblem(
        problem.base_problem,
        problem.adapter,
        supplied_law_id=problem.mean_field.mean_field_id,
        problem_id="one-period-frozen-mfg",
    )

    mfg = solve_frozen_law_best_response(
        frozen_mfg,
        paths,
        lambda time, state: jnp.asarray([0.0]),
        control_predictor=lambda time, state: jnp.asarray([[-1.0]]),
    )
    mfc = _evaluate(problem, paths, action=-2.0)
    mfg_action_in_planner_contract = _evaluate(problem, paths, action=-1.0)

    assert bool(mfg.valid)
    np.testing.assert_allclose(mfg.selected_controls, -1.0)
    assert bool(mfc.valid)
    np.testing.assert_allclose(mfc.physical_controls, -2.0)
    np.testing.assert_allclose(mfc.hamiltonian_stationarity, 0.0)
    np.testing.assert_allclose(mfc.stationarity_infinity_norm, 0.0)
    np.testing.assert_allclose(
        mfg_action_in_planner_contract.hamiltonian_stationarity, 1.0
    )
    assert float(mfc.welfare) < float(mfg_action_in_planner_contract.welfare)
    assert mfc.certificate_label == MEAN_FIELD_CONTROL_PLANNER_STATIONARITY


def test_explicit_zero_externality_reduces_to_base_bsde_residuals():
    paths = _paths()
    zero = _analytic_externality(
        running=lambda time, state, law, value, control, action, args: jnp.zeros((1,)),
        terminal=lambda state, law, args: jnp.zeros((1,)),
    )
    result = _evaluate(_problem(paths, externality=zero), paths)

    np.testing.assert_array_equal(result.running_externality_contributions, 0.0)
    np.testing.assert_array_equal(result.terminal_externality_contributions, 0.0)
    np.testing.assert_allclose(
        result.measure_adjoint_residuals,
        result.bsde_evaluation.local_residuals,
    )
    np.testing.assert_allclose(
        result.terminal_residual,
        result.bsde_evaluation.terminal_residual,
    )


def test_missing_externality_derivative_is_rejected_instead_of_dropped():
    with pytest.raises(TypeError, match="running must be callable"):
        MeanFieldExternality(
            None,
            lambda state, law, args: jnp.zeros((1,)),
            mode="analytic-lions",
            externality_id="incomplete",
            running_id="missing-running",
            terminal_id="terminal",
        )

    with pytest.raises(TypeError, match="externality must be an explicit"):
        paths = _paths()
        MeanFieldControlProblem(
            _base_problem(
                paths,
                EmpiricalMeanField.from_paths(paths, mean_field_id="planner-law"),
            ),
            None,
            lambda time, state, law, action, args: jnp.asarray(0.0),
            lambda state, law, args: jnp.asarray(0.0),
            lambda time, state, law, value, control, action, args: action,
            welfare_running_id="running",
            welfare_terminal_id="terminal",
            stationarity_id="stationarity",
            problem_id="missing-externality",
        )


def test_finite_particle_adjoint_preserves_bias_audit_evidence():
    paths = _paths()
    particle_externality = MeanFieldExternality(
        lambda time, state, law, value, control, action, args: jnp.asarray([law.mean[0]]),
        lambda state, law, args: jnp.zeros((1,)),
        mode="finite-particle-adjoint",
        externality_id="leave-one-out-particle-adjoint",
        running_id="particle-running-adjoint",
        terminal_id="particle-terminal-adjoint",
        particle_count=2,
        discretization_id="two-particle-leave-one-out-dt-1",
        bias_bound=0.25,
    )
    result = _evaluate(_problem(paths, externality=particle_externality), paths)

    assert bool(result.valid)
    assert result.externality_mode == "finite-particle-adjoint"
    assert result.finite_particle_count == 2
    assert result.finite_particle_discretization_id == ("two-particle-leave-one-out-dt-1")
    np.testing.assert_allclose(result.finite_particle_bias_bound, 0.25)
    assert result.finite_particle_adjoint_evaluated
    assert result.finite_particle_bias_audited
    assert not result.analytic_lions_derivatives_evaluated

    with pytest.raises(ValueError, match="requires a finite bias_bound"):
        MeanFieldExternality(
            lambda *args: jnp.zeros((1,)),
            lambda *args: jnp.zeros((1,)),
            mode="finite-particle-adjoint",
            externality_id="unaudited",
            running_id="running",
            terminal_id="terminal",
            particle_count=2,
            discretization_id="two-particle",
        )


def test_invalid_law_low_ess_and_wrong_path_identity_have_distinct_statuses():
    invalid_paths = _paths(valid=jnp.asarray([[True, True], [False, False]]))
    invalid = _evaluate(_problem(invalid_paths), invalid_paths)
    assert not bool(invalid.valid)
    assert int(invalid.status) == int(MeanFieldControlStatus.INVALID_LAW_EVIDENCE)

    paths = _paths()
    low_ess_flow = EmpiricalMeanField.from_paths(
        paths,
        weights=jnp.asarray([[1.0, 1.0], [0.0, 0.0]]),
        mean_field_id="degenerate-law",
    )
    low_ess = _evaluate(_problem(paths, flow=low_ess_flow), paths)
    assert bool(low_ess.law_evidence_valid)
    assert not bool(low_ess.effective_sample_size_sufficient)
    assert int(low_ess.status) == int(MeanFieldControlStatus.LOW_EFFECTIVE_SAMPLE_SIZE)

    wrong_source = EmpiricalMeanField(
        paths.times,
        paths.states,
        sample_shape=paths.sample_shape,
        state_shape=paths.state_shape,
        mean_field_id="wrong-source-law",
        source_path_id="different-paths",
    )
    mismatch = _evaluate(_problem(paths, flow=wrong_source), paths)
    assert not bool(mismatch.path_identity_valid)
    assert int(mismatch.status) == int(MeanFieldControlStatus.PATH_IDENTITY_MISMATCH)


def test_terminal_residual_includes_terminal_measure_externality():
    paths = _paths()
    externality = _analytic_externality(
        running=lambda time, state, law, value, control, action, args: jnp.zeros((1,)),
        terminal=lambda state, law, args: jnp.asarray([0.5]),
    )
    result = _evaluate(
        _problem(paths, externality=externality, terminal_target=1.0),
        paths,
        value=3.0,
    )

    np.testing.assert_allclose(result.bsde_evaluation.terminal_residual, 2.0)
    np.testing.assert_allclose(result.terminal_externality_contributions, 0.5)
    np.testing.assert_allclose(result.terminal_residual, 1.5)
    np.testing.assert_allclose(result.terminal_infinity_norm, 1.5)


def test_result_makes_only_the_explicit_planner_stationarity_claim():
    paths = _paths()
    result = _evaluate(_problem(paths), paths, action=-2.0)

    assert result.candidate_evaluation_only
    assert result.planner_stationarity_evaluated
    assert result.certificate_label == "MEAN_FIELD_CONTROL_PLANNER_STATIONARITY"
    assert not result.frozen_law_best_response_claimed
    assert not result.mean_field_game_equilibrium_claimed
    assert not result.mean_field_control_optimum_claimed
    assert not result.global_optimality_claimed
    assert not result.finite_population_game_claimed
