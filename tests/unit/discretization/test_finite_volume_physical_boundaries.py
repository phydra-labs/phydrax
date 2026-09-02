#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
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


def _wall_velocity_provider(time, point, normal, args):
    del time, point, normal
    return args


def _heat_flux_provider(time, interior, point, normal, args):
    del time, interior, point, normal, args
    return 0.0


def _ale_identity_kwargs(
    *,
    topology_epoch_id="topology-epoch:7",
    geometry_layout_id="geometry-layout:moving",
    geometry_version=11,
    face_block_id="face-block:wall",
    motion_plan_id="motion-plan:translation",
):
    return {
        "topology_epoch_id": topology_epoch_id,
        "geometry_layout_id": geometry_layout_id,
        "geometry_version": jnp.asarray(geometry_version, dtype=jnp.int32),
        "face_block_id": face_block_id,
        "motion_plan_id": motion_plan_id,
    }


def _primitive_target(time, primitive, point, normal, args):
    del time, primitive, point, normal, args
    return jnp.asarray([1.1, 0.1, -0.05, 1.05])


def _pressure_target(time, primitive, point, normal, args):
    del time, primitive, point, normal, args
    return 0.95


def _axis_based_ale_boundary(kind):
    if kind == "reflective":
        return phx.discretization.ReflectiveBoundary()
    if kind == "characteristic-inflow":
        return phx.discretization.CharacteristicInflowBoundary(
            _primitive_target,
            boundary_id="ale-characteristic-inflow",
        )
    if kind == "characteristic-outflow":
        return phx.discretization.CharacteristicOutflowBoundary(
            _pressure_target,
            boundary_id="ale-characteristic-outflow",
        )
    if kind == "far-field":
        return phx.discretization.FarFieldBoundary(
            _primitive_target,
            boundary_id="ale-far-field",
        )
    raise AssertionError(f"Unknown boundary kind {kind}.")


def _moving_wall(
    *,
    provider_id="constant-moving-wall",
    absolute_tolerance=1.0e-12,
    relative_tolerance=1.0e-10,
):
    return phx.discretization.MovingSlipWallBoundary(
        _wall_velocity_provider,
        wall_velocity_provider_id=provider_id,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _ale_context(
    boundary,
    wall_velocity,
    normal,
    *,
    grid_velocity=None,
    identity=None,
):
    wall = jnp.asarray(wall_velocity)
    normal_ = jnp.asarray(normal)
    grid = wall if grid_velocity is None else jnp.asarray(grid_velocity)
    identity_kwargs = _ale_identity_kwargs() if identity is None else dict(identity)
    return boundary.make_context(
        jnp.asarray(0.25),
        jnp.zeros((1, wall.size)),
        normal_[None, :],
        grid[None, :],
        wall,
        **identity_kwargs,
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
    np.testing.assert_allclose(system.conserved_to_primitive(no_slip)[..., 1], -0.1)


def test_isothermal_wall_places_target_temperature_at_face_average():
    system = _system()
    primitive = jnp.asarray([[1.0, 0.2, 2.0]])
    interior = system.primitive_to_conserved(primitive)
    boundary = phx.discretization.NoSlipIsothermalWallBoundary(jnp.asarray([0.0]), 1.5)
    exterior = boundary.exterior_state(
        system,
        0.0,
        interior,
        jnp.asarray([[0.0]]),
        jnp.asarray([-1.0]),
        0,
        None,
    )

    face_temperature = 0.5 * (system.temperature(interior) + system.temperature(exterior))
    np.testing.assert_allclose(face_temperature, 1.5, rtol=1e-12)
    np.testing.assert_allclose(system.conserved_to_primitive(exterior)[..., 1], -0.2)


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


def test_characteristic_boundaries_are_axis_independent_for_oblique_normals():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, -0.1, 1.0]]))
    normal = jnp.asarray([[0.6, 0.8]])

    def target(time, primitive, coordinates, outward_normal, args):
        del time, primitive, coordinates, outward_normal, args
        return jnp.asarray([1.05, 0.1, 0.05, 0.95])

    boundary = phx.discretization.CharacteristicInflowBoundary(
        target, boundary_id="oblique-inflow"
    )
    first = boundary.exterior_state(
        system,
        jnp.asarray(0.0),
        interior,
        jnp.zeros((1, 2)),
        normal,
        0,
        None,
    )
    second = boundary.exterior_state(
        system,
        jnp.asarray(0.0),
        interior,
        jnp.zeros((1, 2)),
        normal,
        1,
        None,
    )
    np.testing.assert_allclose(first, second, rtol=2.0e-12, atol=2.0e-12)
    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(system.admissible(first))


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
    wall = phx.discretization.NoSlipIsothermalWallBoundary(jnp.asarray([0.0]), 2.0)
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
    wall = phx.discretization.NoSlipAdiabaticWallBoundary(jnp.asarray([0.0]))
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.HighResolutionReconstructionPlan("weno_z"),
        phx.discretization.FiniteVolumeBoundarySet(
            ("x",),
            (phx.discretization.FiniteVolumeBoundaryPair(wall, wall),),
        ),
    ).prepare()
    velocity = jnp.linspace(0.1, 0.8, 8)
    primitive = jnp.stack((jnp.ones(8), velocity, jnp.ones(8)), axis=-1)
    state = system.primitive_to_conserved(primitive)
    ghosted = halo.materialize_axis(system, 0.0, state, 0)
    ghost_primitive = system.conserved_to_primitive(ghosted.values)

    assert ghosted.depth == 3
    assert ghosted.values.shape[0] == 14
    assert jnp.all(jnp.diff(ghosted.axis_coordinates) > 0.0)
    np.testing.assert_allclose(ghost_primitive[:3, 1], [-0.3, -0.2, -0.1], atol=1e-12)


def test_moving_slip_wall_matches_static_slip_wall_at_zero_wall_speed():
    system = phx.equations.EulerSystem(2)
    interior_primitive = jnp.asarray([[1.3, 0.7, -0.2, 1.1]])
    interior = system.primitive_to_conserved(interior_primitive)
    normal = jnp.asarray([[1.0, 0.0]])
    static_exterior = phx.discretization.SlipWallBoundary().exterior_state(
        system,
        jnp.asarray(0.25),
        interior,
        jnp.zeros((1, 2)),
        normal,
        0,
        None,
    )
    moving = _moving_wall()
    context = _ale_context(moving, [0.0, 0.0], [1.0, 0.0])
    moving_exterior = moving.ale_exterior_state(system, interior, context, 0)

    np.testing.assert_allclose(moving_exterior, static_exterior, rtol=1e-12)
    np.testing.assert_allclose(context.kinematic_defect, 0.0)
    assert jnp.all(context.kinematics_consistent)


def test_static_slip_wall_rejects_nonzero_conforming_grid_motion():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 0.0, 1.0]]))
    context = _ale_context(_moving_wall(), [0.2, 0.0], [1.0, 0.0])

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="Static ALE boundaries require zero",
    ):
        exterior = phx.discretization.SlipWallBoundary().ale_exterior_state(
            system,
            interior,
            context,
            0,
        )
        jax.block_until_ready(exterior)


def test_moving_slip_wall_reflects_translating_relative_normal_velocity_only():
    system = phx.equations.EulerSystem(2)
    interior_primitive = jnp.asarray([[1.4, 1.1, -0.3, 1.7]])
    interior = system.primitive_to_conserved(interior_primitive)
    boundary = _moving_wall()
    context = _ale_context(boundary, [0.4, 0.2], [1.0, 0.0])
    exterior = boundary.ale_exterior_state(system, interior, context, 0)
    exterior_primitive = system.conserved_to_primitive(exterior)

    np.testing.assert_allclose(exterior_primitive[..., 0], 1.4, rtol=1e-12)
    np.testing.assert_allclose(exterior_primitive[..., 1], -0.3, rtol=1e-12)
    np.testing.assert_allclose(exterior_primitive[..., 2], -0.3, rtol=1e-12)
    np.testing.assert_allclose(exterior_primitive[..., -1], 1.7, rtol=1e-12)


def test_conforming_translating_wall_has_zero_relative_mass_flux_and_wall_work():
    system = phx.equations.EulerSystem(2)
    pressure = 1.6
    wall_velocity = jnp.asarray([-0.4, 0.25])
    normal = jnp.asarray([-1.0, 0.0])
    interior = system.primitive_to_conserved(
        jnp.asarray([[1.2, wall_velocity[0], wall_velocity[1], pressure]])
    )
    boundary = _moving_wall()
    context = _ale_context(boundary, wall_velocity, normal)
    exterior = boundary.ale_exterior_state(system, interior, context, 0)
    physical_flux = system.physical_normal_flux(
        0.5 * (interior + exterior),
        normal[None, :],
    )
    ale_flux = physical_flux - context.grid_normal_velocity[..., None] * 0.5 * (
        interior + exterior
    )

    np.testing.assert_allclose(ale_flux[..., 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        ale_flux[..., 1:3],
        pressure * normal[None, :],
        rtol=1e-12,
        atol=1e-12,
    )
    assert context.wall_normal_velocity.item() > 0.0
    np.testing.assert_allclose(
        ale_flux[..., -1],
        pressure * context.wall_normal_velocity,
        rtol=1e-12,
    )
    assert ale_flux[..., -1].item() > 0.0


def test_moving_slip_wall_preserves_relative_tangent_for_arbitrary_normal():
    system = phx.equations.EulerSystem(2)
    normal = jnp.asarray([0.6, 0.8])
    tangent = jnp.asarray([-normal[1], normal[0]])
    wall_velocity = jnp.asarray([0.2, -0.1])
    interior_velocity = jnp.asarray([1.1, 0.35])
    interior = system.primitive_to_conserved(
        jnp.asarray([[1.0, interior_velocity[0], interior_velocity[1], 0.9]])
    )
    boundary = _moving_wall()
    context = _ale_context(boundary, wall_velocity, normal)
    exterior = boundary.ale_exterior_state(system, interior, context, 0)
    exterior_velocity = system.conserved_to_primitive(exterior)[0, 1:-1]
    interior_relative = interior_velocity - wall_velocity
    exterior_relative = exterior_velocity - wall_velocity

    np.testing.assert_allclose(
        jnp.dot(exterior_relative, normal),
        -jnp.dot(interior_relative, normal),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.dot(exterior_relative, tangent),
        jnp.dot(interior_relative, tangent),
        rtol=1e-12,
        atol=1e-12,
    )


def test_moving_slip_wall_rejects_grid_wall_normal_velocity_mismatch():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.0, 0.0, 1.0]]))
    boundary = _moving_wall(absolute_tolerance=1.0e-13, relative_tolerance=0.0)
    context = _ale_context(
        boundary,
        [0.3, 0.0],
        [1.0, 0.0],
        grid_velocity=[0.31, 4.0],
    )
    assert not jnp.any(context.kinematics_consistent)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="normal velocity does not match",
    ):
        exterior = boundary.ale_exterior_state(system, interior, context, 0)
        jax.block_until_ready(exterior)


def test_ale_boundary_context_rejects_nonfinite_and_mismatched_geometry():
    boundary = _moving_wall()
    with pytest.raises(ValueError, match="same non-scalar shape"):
        phx.discretization.ALEBoundaryContext(
            face_point=jnp.zeros((1, 2)),
            outward_normal=jnp.asarray([1.0, 0.0]),
            quadrature_grid_velocity=jnp.zeros((1, 2)),
            wall_velocity=jnp.zeros((1, 2)),
            time=0.0,
            args=None,
            **_ale_identity_kwargs(),
            absolute_tolerance=boundary.absolute_tolerance,
            relative_tolerance=boundary.relative_tolerance,
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="must be finite",
    ):
        context = phx.discretization.ALEBoundaryContext(
            face_point=jnp.asarray([[jnp.nan, 0.0]]),
            outward_normal=jnp.asarray([[1.0, 0.0]]),
            quadrature_grid_velocity=jnp.zeros((1, 2)),
            wall_velocity=jnp.zeros((1, 2)),
            time=0.0,
            args=None,
            **_ale_identity_kwargs(),
            absolute_tolerance=boundary.absolute_tolerance,
            relative_tolerance=boundary.relative_tolerance,
        )
        jax.block_until_ready(context.face_point)


def test_moving_context_carries_exact_stage_route_identity():
    identity = _ale_identity_kwargs(
        topology_epoch_id="topology-epoch:accepted-3",
        geometry_layout_id="geometry-layout:stage",
        geometry_version=19,
        face_block_id="face-block:physical-2",
        motion_plan_id="motion-plan:deforming",
    )
    context = _ale_context(
        _moving_wall(),
        [0.0, 0.0],
        [1.0, 0.0],
        identity=identity,
    )

    assert context.topology_epoch_id == identity["topology_epoch_id"]
    assert context.geometry_layout_id == identity["geometry_layout_id"]
    assert int(context.geometry_version) == int(identity["geometry_version"])
    assert context.face_block_id == identity["face_block_id"]
    assert context.motion_plan_id == identity["motion_plan_id"]


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("topology_epoch_id", "topology-epoch:stale"),
        ("geometry_layout_id", "geometry-layout:stale"),
        ("geometry_version", jnp.asarray(12, dtype=jnp.int32)),
        ("face_block_id", "face-block:stale"),
        ("motion_plan_id", "motion-plan:stale"),
    ),
)
def test_ale_context_rejects_stale_consumer_stage_identity(field, stale_value):
    identity = _ale_identity_kwargs()
    context = _ale_context(
        _moving_wall(),
        [0.0, 0.0],
        [1.0, 0.0],
        identity=identity,
    )
    consumer_identity = dict(identity)
    consumer_identity[field] = stale_value

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match=field,
    ):
        consumed = context.validate_consumer_identity(
            jnp.ones((1, 4)),
            **consumer_identity,
        )
        jax.block_until_ready(consumed)


def test_ale_context_consumer_version_check_is_jittable():
    identity = _ale_identity_kwargs(geometry_version=23)
    context = _ale_context(
        _moving_wall(),
        [0.0, 0.0],
        [1.0, 0.0],
        identity=identity,
    )

    @jax.jit
    def consume(geometry_version):
        return context.validate_consumer_identity(
            jnp.asarray([[2.0, 3.0]]),
            topology_epoch_id=identity["topology_epoch_id"],
            geometry_layout_id=identity["geometry_layout_id"],
            geometry_version=geometry_version,
            face_block_id=identity["face_block_id"],
            motion_plan_id=identity["motion_plan_id"],
        )

    np.testing.assert_allclose(
        consume(jnp.asarray(23, dtype=jnp.int32)),
        [[2.0, 3.0]],
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="geometry_version",
    ):
        jax.block_until_ready(consume(jnp.asarray(24, dtype=jnp.int32)))


def test_static_slip_wall_accepts_oblique_ale_normal():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.7, -0.2, 1.1]]))
    normal = jnp.asarray([0.6, 0.8])
    context = _ale_context(_moving_wall(), [0.0, 0.0], normal)
    boundary = phx.discretization.SlipWallBoundary()

    exterior = boundary.ale_exterior_state(system, interior, context, 0)
    expected = boundary.exterior_state(
        system,
        context.time,
        interior,
        context.face_point,
        context.outward_normal,
        0,
        context.args,
    )

    np.testing.assert_allclose(exterior, expected, rtol=1e-12, atol=1e-12)


def test_axis_based_reflective_ale_boundary_rejects_oblique_normal():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, -0.1, 1.0]]))
    context = _ale_context(_moving_wall(), [0.0, 0.0], [0.6, 0.8])
    boundary = _axis_based_ale_boundary("reflective")
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="oblique",
    ):
        exterior = boundary.ale_exterior_state(system, interior, context, 0)
        jax.block_until_ready(exterior)


@pytest.mark.parametrize(
    "kind", ("characteristic-inflow", "characteristic-outflow", "far-field")
)
def test_characteristic_ale_boundaries_accept_oblique_normals(kind):
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, -0.1, 1.0]]))
    context = _ale_context(_moving_wall(), [0.0, 0.0], [0.6, 0.8])
    boundary = _axis_based_ale_boundary(kind)
    exterior = boundary.ale_exterior_state(system, interior, context, 0)
    assert jnp.all(jnp.isfinite(exterior))
    assert jnp.all(system.admissible(exterior))


@pytest.mark.parametrize(
    "kind",
    (
        "reflective",
        "characteristic-inflow",
        "characteristic-outflow",
        "far-field",
    ),
)
def test_axis_aligned_ale_boundary_dispatch_matches_static_parity(kind):
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, -0.1, 1.0]]))
    normal = jnp.asarray([0.0, -1.0])
    context = _ale_context(_moving_wall(), [0.0, 0.0], normal)
    boundary = _axis_based_ale_boundary(kind)

    exterior = boundary.ale_exterior_state(system, interior, context, 1)
    expected = boundary.exterior_state(
        system,
        context.time,
        interior,
        context.face_point,
        context.outward_normal,
        1,
        context.args,
    )

    np.testing.assert_allclose(exterior, expected, rtol=1e-12, atol=1e-12)


def test_moving_slip_wall_fingerprints_provider_identity_and_tolerances():
    baseline = _moving_wall()
    changed_provider = _moving_wall(provider_id="different-provider")
    changed_tolerance = _moving_wall(relative_tolerance=2.0e-10)

    assert baseline.boundary_id != changed_provider.boundary_id
    assert baseline.boundary_id != changed_tolerance.boundary_id


def test_moving_slip_wall_is_jittable_and_differentiable_in_wall_speed():
    system = phx.equations.EulerSystem(2)
    interior = system.primitive_to_conserved(jnp.asarray([[1.0, 0.9, 0.2, 1.0]]))
    normal = jnp.asarray([1.0, 0.0])
    boundary = _moving_wall()

    def reflected_normal_velocity(wall_speed):
        wall_velocity = jnp.asarray([wall_speed, 0.0])
        context = _ale_context(boundary, wall_velocity, normal)
        exterior = boundary.ale_exterior_state(system, interior, context, 0)
        return system.conserved_to_primitive(exterior)[0, 1]

    reflected = jax.jit(reflected_normal_velocity)(jnp.asarray(0.3))
    derivative = jax.grad(reflected_normal_velocity)(jnp.asarray(0.3))

    np.testing.assert_allclose(reflected, -0.3, rtol=1e-12)
    np.testing.assert_allclose(derivative, 2.0, rtol=1e-12)


def test_moving_slip_wall_rejects_scalar_and_ale_no_slip_thermal_systems():
    euler = phx.equations.EulerSystem(2)
    moving = _moving_wall()
    context = _ale_context(moving, [0.0, 0.0], [1.0, 0.0])
    scalar = phx.equations.ScalarConservationSystem(
        2,
        lambda value, axis, args: value,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="moving-wall-scalar",
    )
    with pytest.raises(TypeError, match="Euler-compatible"):
        moving.ale_exterior_state(scalar, jnp.ones((1, 1)), context, 0)

    interior = euler.primitive_to_conserved(jnp.asarray([[1.0, 0.0, 0.0, 1.0]]))
    unsupported = (
        phx.discretization.NoSlipAdiabaticWallBoundary(jnp.zeros(2)),
        phx.discretization.NoSlipIsothermalWallBoundary(jnp.zeros(2), 1.0),
        phx.discretization.PrescribedHeatFluxWallBoundary(
            jnp.zeros(2),
            _heat_flux_provider,
            boundary_id="unsupported-ale-heat",
        ),
    )
    for boundary in unsupported:
        with pytest.raises(ValueError, match="no-slip and thermal"):
            boundary.ale_exterior_state(euler, interior, context, 0)
