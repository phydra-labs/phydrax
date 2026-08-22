#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _transport(order=5, points=64):
    spacing = 1.0 / points
    reconstruction = phx.discretization.WENOReconstructionPlan(order)
    flux = phx.discretization.RusanovFluxPlan(
        lambda state, args: state,
        lambda left, right, args: jnp.ones_like(left),
    )
    return phx.discretization.FluxDifferenceDynamics1D(
        reconstruction,
        flux,
        spacing,
    )


def test_weno_rusanov_flux_difference_preserves_constants_and_global_conservation():
    dynamics = _transport()
    constant = jnp.full((64,), 2.5)
    varying = jnp.sin(2.0 * jnp.pi * jnp.arange(64) / 64.0)

    constant_rate = dynamics(jnp.asarray(0.0), constant, None)
    varying_rate = dynamics(jnp.asarray(0.0), varying, None)

    assert jnp.allclose(constant_rate, 0.0)
    assert jnp.allclose(jnp.sum(varying_rate), 0.0, atol=1e-5)


def test_weno5_smooth_face_reconstruction_converges_faster_than_third_order():
    def error(points):
        spacing = 1.0 / points
        left_edges = jnp.arange(points) * spacing
        right_edges = left_edges + spacing
        values = (
            jnp.cos(2.0 * jnp.pi * left_edges) - jnp.cos(2.0 * jnp.pi * right_edges)
        ) / (2.0 * jnp.pi * spacing)
        left, _ = phx.discretization.WENOReconstructionPlan(5).reconstruct(values)
        exact = jnp.sin(2.0 * jnp.pi * right_edges)
        return jnp.sqrt(jnp.mean((left - exact) ** 2))

    coarse = error(40)
    fine = error(80)

    assert coarse / fine > 16.0


def test_ssprk3_step_preserves_constant_transport_state():
    dynamics = _transport()
    state = jnp.ones((64,))

    result = dynamics.ssprk3_step(jnp.asarray(0.0), state, 0.005)

    assert jnp.allclose(result, state)


def test_local_implicit_source_matches_backward_euler_for_linear_decay():
    plan = phx.solver.LocalImplicitSourcePlan(
        lambda state, args: -args["rate"] * state,
        iterations=3,
        tolerance=1e-10,
    )
    state = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    result = plan.step(state, 0.1, {"rate": 2.0})

    assert jnp.allclose(result, state / 1.2, atol=1e-7)


def test_strang_split_composes_source_transport_source_in_order():
    split = phx.solver.StrangSplitPlan(
        lambda time, state, dt, args: state + dt * args["transport"],
        lambda time, state, dt, args: state * jnp.exp(-args["rate"] * dt),
    )
    state = jnp.asarray([1.0, 2.0])

    result = split.step(
        jnp.asarray(0.0),
        state,
        0.2,
        {"transport": 3.0, "rate": 0.5},
    )
    expected = (state * jnp.exp(-0.05) + 0.6) * jnp.exp(-0.05)

    assert jnp.allclose(result, expected)
