#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _dynamics(dimension=1, orientation=3, activity=0.0):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(8, periodic=True) for _ in range(dimension)
    )
    names = tuple("xyz"[:dimension])
    bounds = jnp.stack((jnp.zeros(dimension), jnp.ones(dimension)))
    grid = phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(bounds)
    finite_difference = phx.discretization.periodic_finite_difference(grid)
    basis = phx.equations.NematicTensorBasis(orientation)
    return phx.solver.PreparedNematicDynamics(
        finite_difference,
        phx.equations.LandauDeGennesClosure(basis),
        phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.05),
        phx.equations.BerisEdwardsParameters(0.5, 0.7, activity=activity),
        energy_tolerance=1.0e-8,
    )


def test_passive_nematic_relaxation_decreases_free_energy():
    dynamics = _dynamics()
    compact = jnp.zeros((8, 5)).at[:, 0].set(0.1)
    before = dynamics.evaluate(compact)
    result = dynamics.step(compact, jnp.asarray(1.0e-3))

    assert result.successful
    assert result.evaluation.total_free_energy <= before.total_free_energy + 1e-8
    np.testing.assert_allclose(
        result.evaluation.thermodynamics.trace_residual, 0.0, atol=1e-20
    )
    semi_implicit = phx.solver.PreparedNematicSemiImplicitStepPlan(dynamics, 1.0e-3).step(
        compact
    )
    assert semi_implicit.successful
    assert semi_implicit.evaluation.total_free_energy <= before.total_free_energy + 1.0e-8


def test_periodic_mac_nematic_stress_is_work_dual_and_commit_is_atomic():
    axes = (
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
        jnp.asarray(((0.0, 0.0), (1.0, 1.0)))
    )
    finite_difference = phx.discretization.periodic_finite_difference(grid)
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    basis = phx.equations.NematicTensorBasis(2)
    dynamics = phx.solver.PreparedNematicDynamics(
        finite_difference,
        phx.equations.LandauDeGennesClosure(basis),
        phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.05),
        phx.equations.BerisEdwardsParameters(0.5, 0.7),
        energy_tolerance=1.0e-8,
    )
    points = grid.points.reshape(grid.shape + (2,))
    compact = jnp.zeros(grid.shape + (2,))
    compact = compact.at[..., 0].set(0.1 * jnp.sin(2.0 * jnp.pi * points[..., 0]))
    face_velocity = (
        0.03 * jnp.sin(2.0 * jnp.pi * points[..., 1]),
        0.02 * jnp.cos(2.0 * jnp.pi * points[..., 0]),
    )
    coupling = phx.solver.MACNematicCouplingPlan(
        dynamics,
        operators,
        work_tolerance=1.0e-9,
    )
    evaluated = coupling.evaluate(compact, face_velocity)

    assert evaluated.successful
    np.testing.assert_allclose(
        evaluated.fluid_work,
        -evaluated.nematic_stress_work,
        atol=1.0e-11,
    )
    assert evaluated.work_residual <= 1.0e-11

    state = coupling.initialize_state(compact, face_velocity)
    rejected = coupling.step(state, -1.0)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted_state.compact_q, state.compact_q)
    for accepted, incoming in zip(
        rejected.accepted_state.face_velocity, state.face_velocity, strict=True
    ):
        np.testing.assert_array_equal(accepted, incoming)
    assert rejected.accepted_state.accepted_steps == state.accepted_steps
