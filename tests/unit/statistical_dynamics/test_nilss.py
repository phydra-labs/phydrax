#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.statistical_dynamics._nilss import NILSSPlan


def test_nilss_solves_global_shadowing_coefficients_and_directional_gradient():
    def step(state, parameter):
        return 0.5 * state + jnp.asarray([parameter])

    def objective(state, parameter):
        del parameter
        return state[0]

    prepared = NILSSPlan(
        1,
        1,
        10,
        10,
        regularization=1.0e-12,
    ).prepare(
        step,
        objective,
        jnp.asarray([0.0]),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        dynamics_id="contracting-affine-map",
        objective_id="state-average",
    )
    result = prepared.solve()

    assert bool(result.successful)
    assert result.segment_coefficients.shape == (10, 1)
    assert result.shadowing_tangent.shape == (100, 1)
    np.testing.assert_allclose(result.continuity_residual, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.directional_gradient, 2.0, atol=7e-2)


def test_nilss_resource_and_rank_preflight_refuse_unsupported_runs():
    with pytest.raises(MemoryError, match="maximum_retained_bytes"):
        NILSSPlan(
            10,
            2,
            100,
            10,
            maximum_retained_bytes=1,
        ).prepare(
            lambda state, parameter: state + parameter,
            lambda state, parameter: jnp.sum(state + parameter),
            jnp.zeros(10),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            dynamics_id="resource-refusal",
            objective_id="sum",
        )

    plan = NILSSPlan(2, 2, 2, 2)
    with pytest.raises(ValueError, match="lost numerical rank"):
        plan.prepare(
            lambda state, parameter: state + parameter,
            lambda state, parameter: jnp.sum(state + parameter),
            jnp.zeros(2),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            dynamics_id="rank-refusal",
            objective_id="sum",
            initial_basis=jnp.ones((2, 2)),
        )
