import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.compressible_flow import (
    CompressibleAerodynamicReference,
    CompressibleSurfaceObservationPlan,
    CompressibleSurfacePatchPlan,
)


def _bounded_dynamics(system, *, viscous=False):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8),
            phx.discretization.UniformCellAxisSpec(6),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    if viscous:
        extrapolation = phx.discretization.ExtrapolationBoundary()
        x_pair = phx.discretization.FiniteVolumeBoundaryPair(extrapolation, extrapolation)
        y_pair = phx.discretization.FiniteVolumeBoundaryPair(
            phx.discretization.NoSlipAdiabaticWallBoundary(jnp.asarray((0.0, 0.0))),
            phx.discretization.NoSlipAdiabaticWallBoundary(jnp.asarray((1.0, 0.0))),
        )
        pairs = (x_pair, y_pair)
    else:
        wall = phx.discretization.SlipWallBoundary()
        pair = phx.discretization.FiniteVolumeBoundaryPair(wall, wall)
        pairs = (pair, pair)
    boundaries = phx.discretization.FiniteVolumeBoundarySet(("x", "y"), pairs)
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
        viscous=phx.discretization.ViscousFluxPlan() if viscous else None,
    )
    return phx.discretization.PreparedFiniteVolumeDynamics(
        system, discretization, method, boundaries
    )


def _surface_plan(dynamics):
    reference = CompressibleAerodynamicReference(
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        jnp.asarray((0.5, 0.5)),
        jnp.eye(2),
    )
    patches = (
        CompressibleSurfacePatchPlan(0, "lower", name="left"),
        CompressibleSurfacePatchPlan(0, "upper", name="right"),
        CompressibleSurfacePatchPlan(1, "lower", name="bottom"),
        CompressibleSurfacePatchPlan(1, "upper", name="top"),
    )
    return CompressibleSurfaceObservationPlan(dynamics, patches, reference)


def test_uniform_closed_surface_has_zero_net_force_and_moment():
    system = phx.equations.EulerSystem(2)
    dynamics = _bounded_dynamics(system)
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0)), dynamics.discretization.state_shape
    )
    state = system.primitive_to_conserved(primitive)
    result = _surface_plan(dynamics).evaluate(0.0, state)

    assert bool(result.successful)
    np.testing.assert_allclose(result.integrated_total_force, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.integrated_moment, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.force_balance_defect, 0.0, atol=1.0e-12)
    for patch in result.patches:
        np.testing.assert_allclose(patch.pressure_coefficient, 0.0, atol=1.0e-12)


def test_surface_viscous_traction_matches_linear_couette_shear():
    viscosity = 0.2
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(viscosity, 0.0), 2
    )
    dynamics = _bounded_dynamics(system, viscous=True)
    centers = dynamics.discretization.cell_centers
    primitive = jnp.stack(
        (
            jnp.ones(dynamics.discretization.cell_shape),
            centers[..., 1],
            jnp.zeros(dynamics.discretization.cell_shape),
            jnp.ones(dynamics.discretization.cell_shape),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    result = _surface_plan(dynamics).evaluate(0.0, state)

    bottom = result.patches[2]
    top = result.patches[3]
    np.testing.assert_allclose(
        bottom.integrated_viscous_force[0], viscosity, atol=3.0e-12
    )
    np.testing.assert_allclose(top.integrated_viscous_force[0], -viscosity, atol=3.0e-12)
    np.testing.assert_allclose(
        bottom.integrated_viscous_force + top.integrated_viscous_force,
        0.0,
        atol=3.0e-12,
    )
