#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mesh():
    return phx.geometry.TriangleMesh(
        jnp.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ),
        jnp.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
        source_id="gl-square",
    )


def _plan():
    return phx.applications.superconductivity.GaugeCovariantGLPlan(
        _mesh(),
        alpha=-1.0,
        beta=1.0,
        kinetic_coefficient=0.5,
        magnetic_coefficient=1.0,
        gauge_coupling=1.0,
        tolerance=1.0e-8,
    )


def test_local_gauge_transform_preserves_covariant_energy_and_uniform_branch():
    plan = _plan()
    state = plan.initialize(jnp.ones((5,), dtype=jnp.complex128))
    parameter = jnp.asarray((0.1, -0.2, 0.3, -0.1, 0.0))
    transformed_gauge = plan.gauge.transform(state.gauge, parameter)
    transformed = phx.applications.superconductivity.GinzburgLandauState(
        transformed_gauge, state.time, state.accepted_step, plan.plan_id
    )
    evidence = plan.gauge.validate_transform(state.gauge, parameter)
    solved = plan.solve(state, step_size=0.05, iterations=8)

    assert bool(evidence.successful)
    np.testing.assert_allclose(
        plan.energy(transformed).total, plan.energy(state).total, atol=1e-10
    )
    assert bool(solved.evidence.successful)
    np.testing.assert_allclose(solved.state.gauge.scalar, 1.0 + 0.0j, atol=1e-10)


def test_tdgl_decreases_energy_and_rejects_invalid_step_atomically():
    plan = _plan()
    state = plan.initialize(0.5 * jnp.ones((5,), dtype=jnp.complex128))
    advanced = plan.tdgl_step(state, 0.01)
    rejected = plan.tdgl_step(state, -0.01)

    assert bool(advanced.successful)
    assert advanced.candidate_energy.total <= advanced.initial_energy.total
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted.gauge.scalar, state.gauge.scalar)
    np.testing.assert_array_equal(
        rejected.accepted.gauge.vector_potential, state.gauge.vector_potential
    )
    np.testing.assert_array_equal(rejected.accepted.time, state.time)
