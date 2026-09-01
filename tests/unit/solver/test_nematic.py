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


def test_homeotropic_anchoring_and_mac_stress_divergence_are_finite():
    axes = (
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
        jnp.asarray(((0.0, 0.0), (1.0, 1.0)))
    )
    finite_difference = phx.discretization.periodic_finite_difference(grid)
    basis = phx.equations.NematicTensorBasis(2)
    mask = jnp.ones(grid.shape, dtype=bool)
    normals = jnp.zeros(grid.shape + (2,)).at[..., 0].set(1.0)
    anchoring = phx.equations.NematicAnchoringPlan(
        basis,
        phx.equations.NematicAnchoringKind.HOMEOTROPIC,
        mask,
        normals=normals,
        strength=0.1,
        scalar_order=0.5,
    )
    dynamics = phx.solver.PreparedNematicDynamics(
        finite_difference,
        phx.equations.LandauDeGennesClosure(basis),
        phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.05),
        phx.equations.BerisEdwardsParameters(0.5, 0.7),
        anchoring=anchoring,
    )
    compact = anchoring.preferred_compact
    coupling = phx.solver.MACNematicCouplingPlan(dynamics)
    evaluated = coupling.evaluate(compact, jnp.zeros(grid.shape + (2, 2)))

    assert evaluated.successful
    np.testing.assert_allclose(evaluated.cell_body_force, 0.0, atol=1e-10)
