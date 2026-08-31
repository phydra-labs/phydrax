#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _transport(order=5, points=64):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(points, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id=f"weno-transport-{order}",
    )
    problem = phx.equations.ConservationProblemIR(
        "weno-transport",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.WENOReconstructionPlan(order),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(problem, discretization, method)


def test_weno_rusanov_flux_difference_preserves_constants_and_global_conservation():
    dynamics = _transport()
    constant = jnp.full((64, 1), 2.5)
    varying = jnp.sin(2.0 * jnp.pi * jnp.arange(64) / 64.0)[:, None]

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
    dynamics = _transport().dynamics
    state = jnp.ones((64, 1))

    result = (
        phx.solver.UnsplitFiniteVolumeSSPRK3Plan(dynamics)
        .advance(jnp.asarray(0.0), state, 0.005)
        .state
    )

    assert jnp.allclose(result, state)
