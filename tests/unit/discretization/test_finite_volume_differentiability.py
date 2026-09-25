#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
        lambda left, right, axis, args: jnp.full(left.shape[:-1], jnp.abs(args["speed"])),
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
        differentiability=(
            phx.BranchDifferentiationPolicy.SMOOTH_SURROGATE
            if smooth_epsilon
            else phx.BranchDifferentiationPolicy.BRANCHWISE
        ),
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
    _, jvp = jax.jvp(lambda value: compiled(0.0, value, args), (state,), (tangent,))
    epsilon = 1e-5
    finite_difference = (
        compiled(0.0, state + epsilon * tangent, args)
        - compiled(0.0, state - epsilon * tangent, args)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(jvp, finite_difference, rtol=2e-9, atol=2e-9)


def test_smooth_wave_speed_has_finite_parameter_gradient():
    compiled, grid = _periodic_problem(smooth_epsilon=1e-3)
    state = jnp.sin(2.0 * jnp.pi * grid.structured_axes[0].interval_centers)[..., None]

    gradient = jax.grad(
        lambda speed: jnp.sum(compiled(0.0, state, {"speed": speed}) ** 2)
    )(jnp.asarray(0.0))

    assert jnp.isfinite(gradient)
    assert (
        compiled.method.differentiability
        is phx.BranchDifferentiationPolicy.SMOOTH_SURROGATE
    )


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

    value, tangent = jax.jvp(total_volume, (jnp.asarray(1.3),), (jnp.asarray(1.0),))

    np.testing.assert_allclose(value, 1.3, rtol=1e-12)
    np.testing.assert_allclose(tangent, 1.0, rtol=1e-12)


def test_hard_limiter_reports_frozen_decision_semantics():
    reconstruction = phx.discretization.MUSCLReconstruction(
        phx.discretization.SuperbeeLimiter()
    )
    assert (
        reconstruction.differentiability
        is phx.BranchDifferentiationPolicy.FROZEN_DECISION
    )


def test_branch_policy_owners_reject_members_outside_their_subset():
    policy = phx.BranchDifferentiationPolicy
    with pytest.raises(ValueError, match="FiniteVolumeMethodPlan supports"):
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
            differentiability=policy.FROZEN_DECISION,
        )
    with pytest.raises(ValueError, match="ExplicitStabilizationPlan supports"):
        phx.discretization.ExplicitStabilizationPlan(
            0.1, differentiability=policy.BRANCHWISE
        )
    with pytest.raises(ValueError, match="EntropyFilterPlan supports"):
        phx.equations.fem.EntropyFilterPlan(differentiability=policy.SMOOTH)
    with pytest.raises(TypeError, match="BranchDifferentiationPolicy"):
        phx.discretization.PseudospectralMethodPlan(differentiability="branchwise")


def test_explicit_stabilization_policy_selects_sensor_derivative():
    values = jnp.asarray((0.0, 1.0, 0.0, 1.0))
    measure = jnp.ones((4,))

    def sensor_gradient(differentiability):
        plan = phx.discretization.ExplicitStabilizationPlan(
            0.25, differentiability=differentiability, periodic=True
        )
        return jax.grad(
            lambda sensor: jnp.sum(plan.apply(values, sensor, measure=measure) ** 2)
        )(jnp.full((4,), 0.5))

    frozen = sensor_gradient(phx.BranchDifferentiationPolicy.FROZEN_DECISION)
    smooth = sensor_gradient(phx.BranchDifferentiationPolicy.SMOOTH)
    np.testing.assert_allclose(frozen, 0.0)
    assert jnp.all(jnp.abs(smooth) > 0.0)
