import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples.differentiable_two_body_orbit import build_workflow


def test_two_body_workflow_propagates_and_differentiates_end_to_end():
    _, initial, plan = build_workflow()
    analytic = plan.solve_analytic_two_body(initial)
    numerical = plan.solve(initial)
    assert bool(analytic.successful)
    assert bool(numerical.successful)
    np.testing.assert_allclose(
        analytic.trajectory.states[-1], initial.packed(), atol=2.0e-10
    )
    np.testing.assert_allclose(
        numerical.trajectory.states[-1], analytic.trajectory.states[-1], atol=2.0e-7
    )

    objective = lambda mu: phx.applications.astrodynamics.propagate_universal_kepler(
        initial, 1.0, mu
    ).state.position[0]
    value, tangent = jax.jvp(objective, (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    epsilon = 1.0e-5
    finite_difference = (objective(1.0 + epsilon) - objective(1.0 - epsilon)) / (
        2.0 * epsilon
    )
    assert bool(jnp.isfinite(value))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-5, atol=2.0e-7)
