#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_pseudo_arclength_continues_through_fold_with_hook_and_switch_seed():
    layout = phx.dynamics.StateLayout((), component_names=("x",))
    hook = phx.dynamics.analysis.CallableNormalFormHook(
        lambda state, parameter, args: jnp.asarray([state**3]),
        names=("cubic-state",),
        hook_id="fold-normal-form",
    )
    problem = phx.dynamics.analysis.ContinuationProblem(
        lambda state, parameter, args: state**2 - parameter,
        state_layout=layout,
        parameter_id="mu",
        spectrum_kind="flow",
        normal_form_hook=hook,
        problem_id="quadratic-fold",
    )

    branch = phx.dynamics.analysis.continue_branch(
        problem,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        method="pseudo_arclength",
        direction=-1,
        initial_step=0.08,
        min_step=0.01,
        max_step=0.08,
        max_points=36,
        max_newton_iterations=12,
    )

    count = int(branch.count)
    states = np.asarray(branch.states[:count])
    parameters = np.asarray(branch.parameters[:count])
    minimum = int(np.argmin(parameters))
    assert count == 36
    assert minimum > 2
    assert minimum < count - 2
    assert states[0] > 0.0
    assert states[-1] < 0.0
    assert parameters[-1] > parameters[minimum]
    assert bool(jnp.any(branch.bifurcations.fold[:count]))
    assert int(branch.stability[0]) == phx.dynamics.analysis.STABILITY_UNSTABLE
    assert int(branch.stability[count - 1]) == phx.dynamics.analysis.STABILITY_STABLE
    assert bool(jnp.all(branch.hook_valid[:count]))
    np.testing.assert_allclose(
        np.asarray(branch.hook_values[:count, 0]), states**3, atol=1e-12
    )

    switch = phx.dynamics.analysis.CallableBranchSwitchHook(
        lambda state, parameter, tangent_state, tangent_parameter, args: (
            state + 0.01,
            parameter,
            -tangent_state,
            -tangent_parameter,
        ),
        switch_id="reverse-and-perturb",
    )
    seed = phx.dynamics.analysis.branch_switch_seed(branch, minimum, switch)
    assert bool(seed.valid)
    np.testing.assert_allclose(np.asarray(seed.state), states[minimum] + 0.01, atol=0.0)


def test_natural_map_continuation_detects_flip_and_stops_at_bound():
    layout = phx.dynamics.StateLayout((), component_names=("fixed_point",))
    problem = phx.dynamics.analysis.ContinuationProblem(
        lambda state, parameter, args: (parameter - 1.0) * state,
        state_layout=layout,
        parameter_id="multiplier",
        spectrum_kind="map",
        problem_id="linear-map-fixed-point",
    )

    branch = phx.dynamics.analysis.continue_branch(
        problem,
        jnp.asarray(0.0),
        jnp.asarray(-0.5),
        method="natural",
        direction=-1,
        initial_step=0.1,
        min_step=0.02,
        max_step=0.2,
        parameter_bounds=(-1.6, -0.4),
        max_points=32,
    )

    count = int(branch.count)
    assert count > 3
    assert int(branch.termination_status) == (
        phx.dynamics.analysis.CONTINUATION_PARAMETER_BOUND_REACHED
    )
    assert bool(jnp.any(branch.bifurcations.flip[:count]))
    assert int(branch.stability[0]) == phx.dynamics.analysis.STABILITY_STABLE
    assert int(branch.stability[count - 1]) == phx.dynamics.analysis.STABILITY_UNSTABLE
    np.testing.assert_allclose(
        np.asarray(branch.spectra[:count, 0]).real,
        np.asarray(branch.parameters[:count]),
        atol=1e-12,
    )


def test_hopf_torus_and_branch_point_indicators_are_spectrum_specific():
    layout = phx.dynamics.StateLayout((2,), component_names=("x", "y"))

    def hopf_residual(state, parameter, args):
        matrix = jnp.asarray([[parameter, -1.0], [1.0, parameter]])
        return matrix @ state

    hopf_problem = phx.dynamics.analysis.ContinuationProblem(
        hopf_residual,
        state_layout=layout,
        parameter_id="real-part",
        spectrum_kind="flow",
        problem_id="hopf-normal-form-linearization",
    )
    hopf = phx.dynamics.analysis.continue_branch(
        hopf_problem,
        jnp.zeros((2,)),
        jnp.asarray(-0.2),
        method="natural",
        direction=1,
        initial_step=0.1,
        min_step=0.1,
        max_step=0.1,
        max_points=6,
    )

    angle = 0.4
    rotation = jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )
    torus_problem = phx.dynamics.analysis.ContinuationProblem(
        lambda state, parameter, args: (parameter * rotation - jnp.eye(2)) @ state,
        state_layout=layout,
        parameter_id="radius",
        spectrum_kind="map",
        problem_id="neimark-sacker-linearization",
    )
    torus = phx.dynamics.analysis.continue_branch(
        torus_problem,
        jnp.zeros((2,)),
        jnp.asarray(0.8),
        method="natural",
        direction=1,
        initial_step=0.1,
        min_step=0.1,
        max_step=0.1,
        max_points=6,
    )

    branch_problem = phx.dynamics.analysis.ContinuationProblem(
        lambda state, parameter, args: parameter * state,
        state_layout=layout,
        parameter_id="double-zero",
        spectrum_kind="flow",
        problem_id="double-zero-branch-point",
    )
    branch_point = phx.dynamics.analysis.continue_branch(
        branch_problem,
        jnp.zeros((2,)),
        jnp.asarray(-0.2),
        method="natural",
        direction=1,
        initial_step=0.1,
        min_step=0.1,
        max_step=0.1,
        max_points=5,
        bifurcation_tolerance=1e-12,
    )

    assert bool(jnp.any(hopf.bifurcations.hopf[: int(hopf.count)]))
    assert bool(jnp.any(torus.bifurcations.torus[: int(torus.count)]))
    assert bool(
        jnp.any(branch_point.bifurcations.branch_point[: int(branch_point.count)])
    )


def test_continuation_dense_dimension_guard_is_explicit():
    layout = phx.dynamics.StateLayout((2,))
    problem = phx.dynamics.analysis.ContinuationProblem(
        lambda state, parameter, args: state,
        state_layout=layout,
        parameter_id="mu",
        spectrum_kind="flow",
        problem_id="guarded-continuation",
    )

    with pytest.raises(ValueError, match="max_dense_dimension"):
        phx.dynamics.analysis.continue_branch(
            problem,
            jnp.zeros((2,)),
            jnp.asarray(0.0),
            method="pseudo_arclength",
            max_dense_dimension=2,
        )
