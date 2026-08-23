#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(count=12, *, periodic=False):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _system():
    return phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2)
    )


def test_slip_and_no_slip_walls_apply_distinct_velocity_parity():
    system = _system()
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.3, 1.0]]))
    coordinates = jnp.asarray([[0.0]])
    normal = jnp.asarray([-1.0])
    slip = phx.discretization.SlipWallBoundary().exterior_state(
        system, 0.0, interior, coordinates, normal, 0, None
    )
    no_slip = phx.discretization.NoSlipAdiabaticWallBoundary(
        jnp.asarray([0.1])
    ).exterior_state(system, 0.0, interior, coordinates, normal, 0, None)

    np.testing.assert_allclose(system.conserved_to_primitive(slip)[..., 1], -0.3)
    np.testing.assert_allclose(
        system.conserved_to_primitive(no_slip)[..., 1], -0.1
    )


def test_isothermal_wall_places_target_temperature_at_face_average():
    system = _system()
    primitive = jnp.asarray([[1.0, 0.2, 2.0]])
    interior = system.primitive_to_conserved(primitive)
    boundary = phx.discretization.NoSlipIsothermalWallBoundary(
        jnp.asarray([0.0]), 1.5
    )
    exterior = boundary.exterior_state(
        system,
        0.0,
        interior,
        jnp.asarray([[0.0]]),
        jnp.asarray([-1.0]),
        0,
        None,
    )

    face_temperature = 0.5 * (
        system.temperature(interior) + system.temperature(exterior)
    )
    np.testing.assert_allclose(face_temperature, 1.5, rtol=1e-12)
    np.testing.assert_allclose(
        system.conserved_to_primitive(exterior)[..., 1], -0.2
    )


def test_characteristic_boundaries_return_finite_admissible_states():
    system = phx.equations.EulerSystem()
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 1.0]]))
    def target(time, primitive, coordinates, normal, args):
        del time, primitive, coordinates, normal, args
        return jnp.asarray([1.1, 0.1, 1.05])

    def pressure_target(time, primitive, coordinates, normal, args):
        del time, primitive, coordinates, normal, args
        return 0.95

    inflow = phx.discretization.CharacteristicInflowBoundary(
        target, boundary_id="subsonic-inflow"
    )
    outflow = phx.discretization.CharacteristicOutflowBoundary(
        pressure_target,
        boundary_id="subsonic-outflow",
    )
    arguments = (
        system,
        jnp.asarray(0.0),
        interior,
        jnp.asarray([[0.0]]),
        jnp.asarray([-1.0]),
        0,
        None,
    )
    inflow_state = inflow.exterior_state(*arguments)
    outflow_state = outflow.exterior_state(*arguments)

    assert jnp.all(jnp.isfinite(inflow_state))
    assert jnp.all(jnp.isfinite(outflow_state))
    assert jnp.all(system.admissible(inflow_state))
    assert jnp.all(system.admissible(outflow_state))


def test_prepared_periodic_halo_wraps_declared_reconstruction_depth():
    grid = _grid(12, periodic=True)
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    reconstruction = phx.discretization.HighResolutionReconstructionPlan("weno_z")
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization, reconstruction, boundaries
    ).prepare()
    state = jnp.arange(12.0)[:, None]
    ghosted = halo.ghosted_axis(
        phx.equations.ScalarConservationSystem(
            1,
            lambda value, axis, args: value,
            lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
            system_id="halo-scalar",
        ),
        0.0,
        state,
        0,
    )

    assert halo.depth_by_axis == (3,)
    np.testing.assert_allclose(ghosted[:3, 0], [9.0, 10.0, 11.0])
    np.testing.assert_allclose(ghosted[-3:, 0], [0.0, 1.0, 2.0])


def test_halo_plan_rejects_insufficient_local_extent():
    grid = _grid(4, periodic=True)
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    with pytest.raises(ValueError, match="halo depth"):
        phx.discretization.FiniteVolumeHaloPlan(
            discretization,
            phx.discretization.HighResolutionReconstructionPlan("weno_z"),
            phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
        )


def test_viscous_flux_consumes_isothermal_wall_ghost_temperature():
    grid = _grid(12)
    system = _system()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    wall = phx.discretization.NoSlipIsothermalWallBoundary(
        jnp.asarray([0.0]), 2.0
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(wall, wall),),
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        boundaries,
    ).prepare()
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (12, 3))
    state = system.primitive_to_conserved(primitive)
    flux = phx.discretization.ViscousFluxPlan().face_fluxes(
        system, 0.0, state, discretization, halo
    )[0]

    assert flux[0, -1] < 0.0
    assert flux[-1, -1] > 0.0


def test_prescribed_heat_flux_wall_sets_oriented_energy_flux():
    grid = _grid(10)
    system = _system()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()

    def heat_target(time, interior, coordinates, normal, args):
        del time, interior, coordinates, normal, args
        return 3.0

    wall = phx.discretization.PrescribedHeatFluxWallBoundary(
        jnp.asarray([0.0]),
        heat_target,
        boundary_id="heat-flux-wall",
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(wall, wall),),
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        boundaries,
    ).prepare()
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (10, 3))
    state = system.primitive_to_conserved(primitive)
    flux = phx.discretization.ViscousFluxPlan().face_fluxes(
        system, 0.0, state, discretization, halo
    )[0]

    np.testing.assert_allclose(flux[0, -1], -3.0)
    np.testing.assert_allclose(flux[-1, -1], 3.0)


def test_materialized_halo_contains_mirrored_coordinates_and_layer_states():
    grid = _grid(8)
    system = _system()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    wall = phx.discretization.NoSlipAdiabaticWallBoundary(
        jnp.asarray([0.0])
    )
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.HighResolutionReconstructionPlan("weno_z"),
        phx.discretization.FiniteVolumeBoundarySet(
            ("x",),
            (phx.discretization.FiniteVolumeBoundaryPair(wall, wall),),
        ),
    ).prepare()
    velocity = jnp.linspace(0.1, 0.8, 8)
    primitive = jnp.stack(
        (jnp.ones(8), velocity, jnp.ones(8)), axis=-1
    )
    state = system.primitive_to_conserved(primitive)
    ghosted = halo.materialize_axis(system, 0.0, state, 0)
    ghost_primitive = system.conserved_to_primitive(ghosted.values)

    assert ghosted.depth == 3
    assert ghosted.values.shape[0] == 14
    assert jnp.all(jnp.diff(ghosted.axis_coordinates) > 0.0)
    np.testing.assert_allclose(
        ghost_primitive[:3, 1], [-0.3, -0.2, -0.1], atol=1e-12
    )
