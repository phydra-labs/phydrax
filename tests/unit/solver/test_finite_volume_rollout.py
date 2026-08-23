#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _rollout_runtime(cells=12):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "rollout-euler",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.15, 1.0]), (cells, 3))
    initial = phx.solver.FiniteVolumeRuntimeState(
        system.primitive_to_conserved(primitive), 0.0, 0.001
    )
    return runtime, initial


def test_rollout_retention_policies_preserve_constant_state():
    runtime, initial = _rollout_runtime()
    trajectory = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 6, retention="trajectory"
    ).rollout(initial)
    checkpoints = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 6, retention="checkpoints", checkpoint_stride=2
    ).rollout(initial)
    final = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 6, retention="final"
    ).rollout(initial)

    assert trajectory.retained_states.shape[0] == 6
    assert checkpoints.retained_states.shape[0] == 3
    assert final.retained_states.shape[0] == 1
    np.testing.assert_allclose(
        trajectory.final_state.conservative_state,
        initial.conservative_state,
        atol=2e-12,
    )
    assert jnp.all(trajectory.accepted)


def test_step_rematerialization_matches_uncheckpointed_rollout():
    runtime, initial = _rollout_runtime()
    direct = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 4, rematerialization="none"
    ).rollout(initial)
    rematerialized = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 4, rematerialization="step"
    ).rollout(initial)

    np.testing.assert_allclose(
        rematerialized.final_state.conservative_state,
        direct.final_state.conservative_state,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rollout_gradient_report_matches_forward_reverse_and_finite_difference():
    runtime, initial = _rollout_runtime(8)
    plan = phx.solver.FiniteVolumeRolloutPlan(
        runtime, 3, rematerialization="step"
    )
    tangent = jnp.linspace(-0.2, 0.2, initial.conservative_state.size).reshape(
        initial.conservative_state.shape
    )
    report = plan.gradient_report(
        lambda final, args: jnp.sum(final.conservative_state[..., -1]),
        initial,
        tangent,
        epsilon=1e-5,
    )

    assert report.jvp_vjp_residual < 1e-9
    assert report.finite_difference_residual < 1e-7
