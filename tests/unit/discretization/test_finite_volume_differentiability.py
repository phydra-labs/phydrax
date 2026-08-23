#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_problem(count=16, *, smooth_epsilon=0.0):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: args["speed"] * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1], jnp.abs(args["speed"])
        ),
        system_id="differentiable-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "differentiable-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(smooth_epsilon=smooth_epsilon),
        differentiability="smooth_surrogate" if smooth_epsilon else "branchwise",
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method
    ), grid


def test_state_jvp_matches_centered_directional_difference():
    compiled, grid = _periodic_problem()
    x = grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[..., None]
    tangent = jnp.cos(4.0 * jnp.pi * x)[..., None]
    args = {"speed": jnp.asarray(0.7)}
    _, jvp = jax.jvp(
        lambda value: compiled(0.0, value, args), (state,), (tangent,)
    )
    epsilon = 1e-5
    finite_difference = (
        compiled(0.0, state + epsilon * tangent, args)
        - compiled(0.0, state - epsilon * tangent, args)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(jvp, finite_difference, rtol=2e-9, atol=2e-9)


def test_smooth_wave_speed_has_finite_parameter_gradient():
    compiled, grid = _periodic_problem(smooth_epsilon=1e-3)
    state = jnp.sin(
        2.0 * jnp.pi * grid.structured_axes[0].interval_centers
    )[..., None]

    gradient = jax.grad(
        lambda speed: jnp.sum(compiled(0.0, state, {"speed": speed}) ** 2)
    )(jnp.asarray(0.0))

    assert jnp.isfinite(gradient)
    assert compiled.method.differentiability == "smooth_surrogate"


def test_boundary_control_gradient_flows_through_exterior_state():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(10),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="boundary-control",
    )
    controlled = phx.discretization.PrescribedStateBoundary(
        lambda time, interior, coordinates, normal, args: args["inflow"],
        boundary_id="controlled-inflow",
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        controlled, phx.discretization.ExtrapolationBoundary()
    )
    problem = phx.equations.ConservationProblemIR(
        "boundary-control",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
    )
    state = jnp.zeros(discretization.state_shape)

    gradient = jax.grad(
        lambda inflow: jnp.sum(
            discretization.cell_volumes[..., None]
            * compiled(0.0, state, {"inflow": inflow})
        )
    )(jnp.asarray(0.4))

    np.testing.assert_allclose(gradient, 1.0, rtol=1e-12)


def test_mapped_cell_volume_is_differentiable_at_fixed_topology():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    reference = phx.discretization.FiniteVolumePlan(grid).prepare()

    def total_volume(scale):
        geometry = phx.discretization.evaluate_mapped_finite_volume_geometry(
            reference,
            lambda point: jnp.stack((scale * point[0], point[1])),
        )
        return jnp.sum(geometry[2])

    value, tangent = jax.jvp(
        total_volume, (jnp.asarray(1.3),), (jnp.asarray(1.0),)
    )

    np.testing.assert_allclose(value, 1.3, rtol=1e-12)
    np.testing.assert_allclose(tangent, 1.0, rtol=1e-12)


def test_fixed_amr_transfer_has_exact_transpose_pairing():
    transfer = phx.discretization.ConservativeBlockTransfer(1, 2)
    coarse = jnp.asarray([[1.0], [2.0], [3.0]])
    tangent = jnp.asarray([[0.2], [-0.1], [0.4]])
    cotangent = jnp.arange(6.0)[:, None]
    _, prolonged_tangent = jax.jvp(transfer.prolong, (coarse,), (tangent,))
    _, pullback = jax.vjp(transfer.prolong, coarse)
    coarse_cotangent = pullback(cotangent)[0]

    np.testing.assert_allclose(
        jnp.vdot(prolonged_tangent, cotangent),
        jnp.vdot(tangent, coarse_cotangent),
        rtol=1e-12,
    )


def test_hard_limiter_reports_frozen_decision_semantics():
    reconstruction = phx.discretization.MUSCLReconstruction(
        phx.discretization.SuperbeeLimiter()
    )
    assert reconstruction.differentiability == "frozen_decision"
