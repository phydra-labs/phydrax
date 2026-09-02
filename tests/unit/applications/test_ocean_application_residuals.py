import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


ocean_api = phx.applications.ocean


def _ocean(*, policy=None, wetting=False):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (4.0, 4.0, 0.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        finite_volume,
        jnp.full((4, 4), 10.0),
    ).prepare()
    return ocean_api.HydrostaticPrimitiveEquationPlan(
        geometry,
        external_mode="split-explicit",
        wetting_and_drying=wetting,
        subcycle_policy=(
            ocean_api.ExternalModeSubcyclePolicy.fixed(4) if policy is None else policy
        ),
    ).prepare()


def _state(prepared, eta=0.0):
    return prepared.initialize_state(
        jnp.full(prepared.geometry.horizontal_shape, eta),
        tracers={
            "absolute_salinity": jnp.full(prepared.geometry.cell_shape, 35.0),
            "conservative_temperature": jnp.full(prepared.geometry.cell_shape, 10.0),
        },
    )


def test_gsw75_named_subset_derivatives_and_funnel():
    eos = ocean_api.TEOS10GSW75EOS()
    result = eos.evaluate(jnp.asarray([35.0]), jnp.asarray([10.0]), jnp.asarray([0.0]))
    assert bool(result.successful)
    np.testing.assert_allclose(result.density, [1026.825], rtol=3.0e-4)
    assert result.alpha[0] > 0.0
    assert result.beta[0] > 0.0
    assert result.density_pressure_derivative[0] > 0.0
    outside = eos.evaluate(jnp.asarray([50.0]), jnp.asarray([10.0]), jnp.asarray([0.0]))
    assert not bool(outside.valid)
    assert not bool(outside.successful)
    prepared = ocean_api.HydrostaticPrimitiveEquationPlan(
        _ocean().geometry, eos=eos
    ).prepare()
    state = prepared.initialize_state(
        jnp.zeros(prepared.geometry.horizontal_shape),
        tracers={
            "absolute_salinity": jnp.full(prepared.geometry.cell_shape, 50.0),
            "conservative_temperature": jnp.full(prepared.geometry.cell_shape, 10.0),
        },
    )
    view = prepared.view(state)
    assert not bool(view.eos_valid)
    assert bool(view.eos_finite)
    assert not bool(view.eos_successful)
    diagnostic = ocean_api.hydrostatic_diagnostic_view(prepared, state)
    assert not bool(diagnostic.eos_valid)
    assert bool(diagnostic.eos_finite)
    assert not bool(diagnostic.eos_successful)
    assert not bool(diagnostic.successful)
    continuation = ocean_api.HydrostaticContinuationState.initialize(prepared, state)
    advanced = ocean_api.HydrostaticIMEXMidpointMethod(prepared).step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )
    assert not bool(advanced.successful)
    np.testing.assert_array_equal(
        advanced.accepted_state.state.tracer_inventory["absolute_salinity"],
        state.tracer_inventory["absolute_salinity"],
    )


def test_spherical_mosaics_cover_caps_tripolar_and_cube_with_oriented_seams():
    radius = 2.0
    vertical = jnp.asarray([-1.0, -0.5, 0.0])
    polar = ocean_api.polar_cap(
        (24, 12), vertical, 1.0, radius=radius, cap_latitude=np.deg2rad(60.0)
    ).prepare()
    polar_area = sum(jnp.sum(block.cell_area) for block in polar.blocks)
    expected_cap = 2.0 * np.pi * radius**2 * (1.0 - np.sin(np.deg2rad(60.0)))
    np.testing.assert_allclose(polar_area, expected_cap, rtol=2.0e-6)
    assert isinstance(polar.topology, phx.discretization.PreparedMultiblockGrid)

    tripolar = ocean_api.tripolar(
        (16, 10), vertical, 1.0, radius=radius, cap_latitude=np.deg2rad(50.0)
    ).prepare()
    tripolar_area = sum(jnp.sum(block.cell_area) for block in tripolar.blocks)
    np.testing.assert_allclose(tripolar_area, 4.0 * np.pi * radius**2, rtol=2.0e-6)
    assert len(tripolar.topology.plan.interfaces) == len(tripolar.seams)
    assert all(report.passed for report in tripolar.topology.interface_reports)
    for seam, interface in zip(
        tripolar.seams, tripolar.topology.plan.interfaces, strict=True
    ):
        assert (
            interface.left_axis,
            interface.left_side,
            interface.right_axis,
            interface.right_side,
        ) == (
            seam.left_axis,
            seam.left_side,
            seam.right_axis,
            seam.right_side,
        )
        assert interface.orientation.orientation_id == seam.orientation.orientation_id
    assert tripolar.northern_poles.shape == (2, 3)
    assert jnp.all(tripolar.northern_poles[:, 2] > 0.0)
    assert jnp.all(tripolar.northern_poles[:, 2] < 1.0)
    assert not bool(jnp.allclose(tripolar.northern_poles[0], tripolar.northern_poles[1]))
    assert all(
        bool(jnp.all(jnp.isfinite(block.horizontal_jacobian)))
        and bool(jnp.all(block.horizontal_jacobian > 0.0))
        for block in tripolar.blocks
    )
    assert any(seam.orientation.flips == (True,) for seam in tripolar.seams)
    left, right = tripolar.scatter_seam_flux(0, jnp.ones((16,)))
    np.testing.assert_allclose(jnp.sum(left) + jnp.sum(right), 0.0, atol=1.0e-12)
    block_values = {
        block.name: jnp.ones(block.cell_area.shape) for block in tripolar.blocks
    }
    left_trace, right_trace = tripolar.seam_traces(0, block_values)
    assert left_trace.shape == right_trace.shape
    block_states = {
        block.name: ocean_api.HydrostaticOceanState(
            jnp.zeros(block.geometry.horizontal_shape),
            (
                jnp.zeros(block.geometry.x_face_shape),
                jnp.zeros(block.geometry.y_face_shape),
            ),
            {},
            jnp.zeros(block.geometry.cell_shape),
        )
        for block in tripolar.blocks
    }
    left_transport, right_transport = tripolar.seam_transport_traces(0, block_states)
    assert left_transport.shape == right_transport.shape
    scattered = tripolar.scatter_seam_flux(0, jnp.ones(left_transport.shape))
    np.testing.assert_allclose(scattered[0] + scattered[1], 0.0)
    assert isinstance(
        tripolar.prepare_ocean("southwest-belt"),
        ocean_api.PreparedHydrostaticOcean,
    )

    cube = ocean_api.equiangular_cubed_sphere(
        (24, 24), vertical, 1.0, radius=radius
    ).prepare()
    cube_area = sum(jnp.sum(block.cell_area) for block in cube.blocks)
    np.testing.assert_allclose(cube_area, 4.0 * np.pi * radius**2, rtol=3.0e-3)
    assert len(cube.blocks) == 6
    assert len(cube.seams) == 12
    assert len(cube.topology.plan.interfaces) == len(cube.seams)
    assert all(report.passed for report in cube.topology.interface_reports)
    for seam in cube.seams:
        left_frame = cube.block(seam.left_block).interface_frame(
            seam.left_axis, seam.left_side
        )
        right_frame = seam.orientation.apply(
            cube.block(seam.right_block).interface_frame(
                seam.right_axis, seam.right_side
            ),
            trailing_axes=2,
        )
        overlap = np.einsum("...ki,...kj->...ij", left_frame, right_frame)
        right_gram = np.einsum("...ki,...kj->...ij", right_frame, right_frame)
        expected_rotation = np.einsum(
            "...ik,...kj->...ij", overlap, np.linalg.inv(right_gram)
        )
        assert seam.vector_rotation.ndim == 3
        np.testing.assert_allclose(seam.vector_rotation, expected_rotation, atol=2.0e-12)
    block_ocean = cube.prepare_ocean("+x")
    block_state = block_ocean.initialize_state(
        jnp.zeros(block_ocean.geometry.horizontal_shape)
    )
    assert bool(block_ocean.view(block_state).eos_successful)


def test_cubed_sphere_cross_metric_drives_manufactured_gradients_and_transports():
    count = 48
    grid = ocean_api.equiangular_cubed_sphere(
        (count, count), jnp.asarray([-1.0, 0.0, 1.0]), 2.0, radius=1.0
    ).prepare()
    block = grid.block("+x")
    geometry = block.geometry
    covariant = block.covariant_metric
    contravariant = block.contravariant_metric
    identity = np.einsum("...ik,...kj->...ij", covariant, contravariant)
    np.testing.assert_allclose(
        identity,
        np.broadcast_to(np.eye(2), identity.shape),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert float(jnp.max(jnp.abs(covariant[..., 0, 1]))) > 0.1

    xi = (jnp.arange(count, dtype=float) + 0.5) / count
    eta = (jnp.arange(count, dtype=float) + 0.5) / count
    potential = xi[:, None] + 2.0 * eta[None, :]
    gradient = geometry.surface_gradient(potential)
    cell_gradient = (
        0.5 * (gradient[0][:-1] + gradient[0][1:]),
        0.5 * (gradient[1][:, :-1] + gradient[1][:, 1:]),
    )
    first_scale = jnp.sqrt(covariant[..., 0, 0])
    second_scale = jnp.sqrt(covariant[..., 1, 1])
    cross = covariant[..., 0, 1] / (first_scale * second_scale)
    normal_scale = jnp.sqrt(1.0 - cross**2)
    expected = (
        (1.0 / first_scale - 2.0 * cross / second_scale) / normal_scale,
        (2.0 / second_scale - cross / first_scale) / normal_scale,
    )
    interior = (slice(3, -3), slice(3, -3))
    np.testing.assert_allclose(
        cell_gradient[0][interior], expected[0][interior], rtol=4.0e-3, atol=4.0e-3
    )
    np.testing.assert_allclose(
        cell_gradient[1][interior], expected[1][interior], rtol=4.0e-3, atol=4.0e-3
    )

    epoch = geometry.metric_epoch(jnp.zeros(geometry.horizontal_shape))
    coordinate_velocity = (
        jnp.ones(geometry.cell_shape),
        -0.5 * jnp.ones(geometry.cell_shape),
    )
    normal_velocity = geometry.contravariant_to_normal_velocity(coordinate_velocity)
    transports = geometry.contravariant_transport(coordinate_velocity, epoch)
    reconstructed = (
        transports[0] / epoch.x_face_area,
        transports[1] / epoch.y_face_area,
    )
    np.testing.assert_allclose(
        reconstructed[0][1:-1],
        0.5 * (normal_velocity[0][:-1] + normal_velocity[0][1:]),
    )
    np.testing.assert_allclose(
        reconstructed[1][:, 1:-1],
        0.5 * (normal_velocity[1][:, :-1] + normal_velocity[1][:, 1:]),
    )
    layer_potential = jnp.broadcast_to(potential[..., None], geometry.cell_shape)
    face_force = geometry.layer_potential_transport_force(layer_potential, epoch)
    np.testing.assert_allclose(face_force[0], -epoch.x_face_area * gradient[0][..., None])
    np.testing.assert_allclose(face_force[1], -epoch.y_face_area * gradient[1][..., None])
    rotated = geometry.rotate_normal_velocity(normal_velocity, geometry.coriolis)
    coriolis = geometry.coriolis[..., None]
    cross_layer = cross[..., None]
    normal_scale_layer = normal_scale[..., None]
    np.testing.assert_allclose(
        rotated[0],
        coriolis
        * (normal_velocity[1] + cross_layer * normal_velocity[0])
        / normal_scale_layer,
    )
    np.testing.assert_allclose(
        rotated[1],
        -coriolis
        * (normal_velocity[0] + cross_layer * normal_velocity[1])
        / normal_scale_layer,
    )


def test_public_mosaic_advance_reconciles_every_physical_seam_conservatively():
    grid = ocean_api.equiangular_cubed_sphere(
        (3, 3), jnp.asarray([-1.0, -0.5, 0.0]), 1.0, radius=10.0
    ).prepare()
    transports = {}
    for block in grid.blocks:
        x = jnp.zeros(block.geometry.x_face_shape)
        y = jnp.zeros(block.geometry.y_face_shape)
        if block.name == "+x":
            x = x.at[-1].set(2.0)
        transports[block.name] = (x, y)
    coupled = grid.prepare_oceans(
        external_mode="implicit",
        subcycle_policy=ocean_api.ExternalModeSubcyclePolicy.fixed(1),
    )
    assert isinstance(coupled, ocean_api.PreparedHydrostaticMosaicOcean)
    state = coupled.initialize_state(transports=transports)
    first_left, first_right = grid.seam_transport_traces(
        0, {name: value.state for name, value in state.blocks.items()}
    )
    np.testing.assert_allclose(first_left, first_right)
    initial_volume = sum(
        jnp.sum(
            grid.block(name).geometry.metric_epoch(continuation.state.eta).cell_volume
        )
        for name, continuation in state.blocks.items()
    )
    uncoupled_candidates = {}
    for block, method in zip(grid.blocks, coupled.methods, strict=True):
        uncoupled_candidates[block.name] = method.step(
            jnp.asarray(0),
            jnp.asarray(0.0),
            state.blocks[block.name],
            jnp.asarray(1.0e-4),
            None,
        ).candidate_state.state
    uncoupled_left, uncoupled_right = grid.seam_transport_traces(0, uncoupled_candidates)
    assert float(jnp.max(jnp.abs(uncoupled_left - uncoupled_right))) > 1.0e-12
    result = coupled.advance(state, jnp.asarray(0.0), jnp.asarray(1.0e-4))
    assert bool(result.successful)
    assert float(jnp.max(jnp.abs(result.seam_fluxes[0]))) > 0.0
    final_volume = sum(
        jnp.sum(
            grid.block(name).geometry.metric_epoch(continuation.state.eta).cell_volume
        )
        for name, continuation in result.state.blocks.items()
    )
    np.testing.assert_allclose(final_volume, initial_volume, rtol=5.0e-12, atol=1.0e-12)
    raw_states = {name: value.state for name, value in result.state.blocks.items()}
    for index, seam in enumerate(grid.seams):
        left, right = grid.seam_transport_traces(index, raw_states)
        left_sign = -1.0 if seam.left_side == "lower" else 1.0
        right_sign = -1.0 if seam.right_side == "lower" else 1.0
        np.testing.assert_allclose(
            left_sign * left + right_sign * right, 0.0, atol=1.0e-12
        )


def test_mosaic_advance_commits_atomically_when_one_block_fails():
    grid = ocean_api.equiangular_cubed_sphere(
        (2, 2), jnp.asarray([-1.0, -0.5, 0.0]), 1.0, radius=10.0
    ).prepare()
    coupled = grid.prepare_oceans(
        external_mode="implicit",
        subcycle_policy=ocean_api.ExternalModeSubcyclePolicy.fixed(1),
    )
    shape = grid.block("+x").geometry.cell_shape
    state = coupled.initialize_state(
        tracers={
            "+x": {
                "absolute_salinity": jnp.full(shape, jnp.nan),
                "conservative_temperature": jnp.full(shape, 10.0),
            }
        }
    )
    result = coupled.advance(state, jnp.asarray(0.0), jnp.asarray(1.0e-4))
    assert not bool(result.successful)
    actual_leaves = jax.tree.leaves(result.state)
    expected_leaves = jax.tree.leaves(state)
    assert len(actual_leaves) == len(expected_leaves)
    for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_split_external_mode_is_only_available_without_multiblock_seams():
    vertical = jnp.asarray([-1.0, -0.5, 0.0])
    cube = ocean_api.equiangular_cubed_sphere(
        (2, 2), vertical, 1.0, radius=10.0
    ).prepare()
    with pytest.raises(ValueError, match="globally coupled implicit"):
        cube.prepare_oceans(external_mode="split-explicit")

    polar = ocean_api.polar_cap((4, 3), vertical, 1.0, radius=10.0).prepare()
    boundary = ocean_api.HydrostaticOpenBoundary(
        1,
        "lower",
        "prescribed-transport",
        target_transport=1.0e-3,
    )
    coupled = polar.prepare_oceans(
        external_mode="split-explicit",
        boundaries=(boundary,),
        subcycle_policy=ocean_api.ExternalModeSubcyclePolicy.fixed(1),
    )
    state = coupled.initialize_state()
    result = coupled.advance(state, jnp.asarray(0.0), jnp.asarray(1.0e-5))
    assert bool(result.successful)


def test_mosaic_ghosts_launch_cross_seam_wave_and_conserve_upwind_tracer():
    grid = ocean_api.equiangular_cubed_sphere(
        (3, 3), jnp.asarray([-1.0, -0.5, 0.0]), 1.0, radius=10.0
    ).prepare()
    eta = {block.name: 1.0e-3 * block.cartesian_unit[..., 0] for block in grid.blocks}
    transports = {}
    tracers = {}
    for block in grid.blocks:
        x = jnp.zeros(block.geometry.x_face_shape)
        y = jnp.zeros(block.geometry.y_face_shape)
        if block.name == "+x":
            x = x.at[-1].set(1.0)
        elif block.name == "+y":
            x = x.at[0].set(1.0)
        transports[block.name] = (x, y)
        salinity = 34.0 if block.name == "+x" else 36.0
        tracers[block.name] = {
            "absolute_salinity": jnp.full(block.geometry.cell_shape, salinity),
            "conservative_temperature": jnp.full(block.geometry.cell_shape, 10.0),
            "passive": jnp.full(
                block.geometry.cell_shape,
                1.0e-20 if block.name == "+x" else 2.0e-20,
            ),
        }
    coupled = grid.prepare_oceans(
        external_mode="implicit",
        subcycle_policy=ocean_api.ExternalModeSubcyclePolicy.fixed(1),
    )
    state = coupled.initialize_state(eta, transports=transports, tracers=tracers)
    initial_salt = sum(
        jnp.sum(value.state.tracer_inventory["absolute_salinity"])
        for value in state.blocks.values()
    )
    initial_passive = sum(
        jnp.sum(value.state.tracer_inventory["passive"])
        for value in state.blocks.values()
    )
    initial_receiver = jnp.sum(
        state.blocks["+y"].state.tracer_inventory["absolute_salinity"]
    )
    result = coupled.advance(state, jnp.asarray(0.0), jnp.asarray(1.0e-4))
    assert bool(result.successful)
    assert max(float(jnp.max(jnp.abs(value))) for value in result.seam_fluxes) > 1.0e-8
    final_salt = sum(
        jnp.sum(value.state.tracer_inventory["absolute_salinity"])
        for value in result.state.blocks.values()
    )
    np.testing.assert_allclose(final_salt, initial_salt, rtol=2.0e-12)
    final_passive = sum(
        jnp.sum(value.state.tracer_inventory["passive"])
        for value in result.state.blocks.values()
    )
    np.testing.assert_allclose(final_passive, initial_passive, rtol=2.0e-12, atol=0.0)
    assert (
        jnp.abs(
            jnp.sum(result.state.blocks["+y"].state.tracer_inventory["absolute_salinity"])
            - initial_receiver
        )
        > 1.0e-6
    )


def test_mosaic_redi_gradient_uses_neighbor_trace_and_is_globally_conservative():
    grid = ocean_api.equiangular_cubed_sphere(
        (3, 3), jnp.asarray([-1.0, -0.5, 0.0]), 1.0, radius=10.0
    ).prepare()
    tracers = {
        block.name: {
            "absolute_salinity": jnp.full(
                block.geometry.cell_shape,
                34.0 if block.name == "+x" else 36.0,
            ),
            "conservative_temperature": jnp.full(block.geometry.cell_shape, 10.0),
        }
        for block in grid.blocks
    }
    coupled = grid.prepare_oceans(
        external_mode="implicit",
        subcycle_policy=ocean_api.ExternalModeSubcyclePolicy.fixed(1),
        mixing=ocean_api.HydrostaticMixingPlan(
            "redi-gm",
            background_diffusivity=0.0,
            background_viscosity=0.0,
            redi_coefficient=1.0e-3,
        ),
    )
    state = coupled.initialize_state(tracers=tracers)
    initial = sum(
        jnp.sum(value.state.tracer_inventory["absolute_salinity"])
        for value in state.blocks.values()
    )
    result = coupled.advance(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert bool(result.successful)
    final = sum(
        jnp.sum(value.state.tracer_inventory["absolute_salinity"])
        for value in result.state.blocks.values()
    )
    np.testing.assert_allclose(final, initial, rtol=2.0e-12)
    assert (
        jnp.max(
            jnp.abs(
                result.state.blocks["+x"].state.tracer_inventory["absolute_salinity"]
                - state.blocks["+x"].state.tracer_inventory["absolute_salinity"]
            )
        )
        > 1.0e-9
    )


def test_adaptive_subcycle_schedule_is_masked_and_fails_over_capacity():
    policy = ocean_api.ExternalModeSubcyclePolicy.adaptive_cfl(16, target_courant=0.4)
    prepared = _ocean(policy=policy)
    state = _state(prepared)
    barotropic = prepared.geometry.depth_integrate(state.transports)
    schedule = policy.schedule(
        prepared.geometry,
        state.eta,
        0.1,
        prepared.plan.gravity,
        barotropic_transport=barotropic,
    )
    assert bool(schedule.successful)
    assert schedule.active_mask.shape == (16,)
    assert jnp.sum(schedule.active_mask) == schedule.count
    overflow = policy.schedule(
        prepared.geometry,
        state.eta,
        100.0,
        prepared.plan.gravity,
        barotropic_transport=barotropic,
    )
    assert not bool(overflow.capacity_valid)
    assert not bool(overflow.successful)


def test_adaptive_subcycle_uses_the_controlling_directional_spacing():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (0.04, 400.0, 0.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        finite_volume, jnp.full((4, 4), 10.0)
    ).prepare()
    policy = ocean_api.ExternalModeSubcyclePolicy.adaptive_cfl(64, target_courant=0.5)
    schedule = policy.schedule(geometry, jnp.zeros((4, 4)), 0.01, 9.81)

    assert bool(schedule.successful)
    assert int(schedule.count) == 20
    epoch = geometry.metric_epoch(jnp.zeros((4, 4)))
    advective = policy.schedule(
        geometry,
        jnp.zeros((4, 4)),
        0.01,
        9.81,
        barotropic_transport=(
            10.0 * jnp.sum(epoch.x_face_area, axis=-1),
            jnp.zeros(geometry.y_face_shape[:-1]),
        ),
    )
    assert bool(advective.successful)
    assert int(advective.count) == 40


def test_passive_vertical_velocity_is_grid_area_invariant():
    def advect_on_square(length):
        grid = phx.discretization.TensorGridPlan(
            (
                phx.discretization.UniformCellAxisSpec(2),
                phx.discretization.UniformCellAxisSpec(2),
                phx.discretization.UniformCellAxisSpec(2),
            ),
            axis_names=("x", "y", "z"),
        ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (length, length, 0.0))))
        finite_volume = phx.discretization.FiniteVolumePlan(
            grid, component_names=("ocean",)
        ).prepare()
        geometry = phx.discretization.TensorZHydrostaticGridPlan(
            finite_volume, jnp.full((2, 2), 10.0)
        ).prepare()
        prepared = ocean_api.HydrostaticPrimitiveEquationPlan(geometry).prepare()
        eta = jnp.zeros((2, 2))
        epoch = geometry.metric_epoch(eta)
        x_coordinate = jnp.linspace(0.0, length, 3)[:, None, None]
        transports = (
            x_coordinate * epoch.x_face_area,
            jnp.zeros(geometry.y_face_shape),
        )
        state = prepared.initialize_state(eta, transports=transports)
        initial = jnp.asarray([[0.5 * length, 0.5 * length, -7.5]])
        return ocean_api.PassiveOceanTrajectoryPlan(prepared, 1).advect(
            initial, state, 0.01, 1
        )

    unit = advect_on_square(1.0)
    scaled = advect_on_square(4.0)
    assert bool(unit.successful)
    assert bool(scaled.successful)
    np.testing.assert_allclose(
        unit.sampled_velocity[0, 0, 2],
        scaled.sampled_velocity[0, 0, 2],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        unit.trajectory.states[0, 1, 2] - unit.trajectory.states[0, 0, 2],
        scaled.trajectory.states[0, 1, 2] - scaled.trajectory.states[0, 0, 2],
        atol=1.0e-6,
    )


def test_wet_dry_epoch_semantics_and_passive_trajectory_lowering():
    prepared = _ocean(wetting=True)
    previous = _state(prepared, eta=0.0)
    candidate = _state(prepared, eta=0.0)
    candidate = phx.applications.ocean.HydrostaticOceanState(
        candidate.eta.at[0, 0].set(-10.0),
        candidate.transports,
        candidate.tracer_inventory,
        candidate.tke_inventory,
    )
    event = ocean_api.HydrostaticWetDryEventPlan(
        ocean_api.WetDryEpochPolicy(wet_depth=1.0e-3, dry_depth=5.0e-4)
    ).transition(prepared, previous, candidate, eta_tangent=jnp.ones((4, 4)))
    assert bool(event.evidence.deactivated[0, 0])
    assert bool(event.evidence.topology_changed)
    assert event.evidence.event_count == 1
    assert bool(event.evidence.derivative_available)
    assert event.eta_tangent[0, 0] == 0.0

    still = _state(prepared)
    trajectory = ocean_api.PassiveOceanTrajectoryPlan(prepared, 4).advect(
        jnp.asarray([[1.0, 1.0, -5.0]]), still, 0.1, 3
    )
    assert bool(trajectory.successful)
    assert trajectory.trajectory.states.shape == (1, 5, 3)
    np.testing.assert_allclose(
        trajectory.trajectory.states[:, :4],
        jnp.asarray([[[1.0, 1.0, -5.0]] * 4]),
        atol=1.0e-12,
    )
    assert trajectory.active_steps[0] == 3


def test_passive_trajectory_uses_nonuniform_vertical_layer_boundaries():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.NonuniformCellAxisSpec(
                jnp.asarray([0.0, 0.1, 0.2, 0.4, 1.0])
            ),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (1.0, 1.0, 0.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        finite_volume, jnp.full((2, 2), 10.0)
    ).prepare()
    prepared = ocean_api.HydrostaticPrimitiveEquationPlan(geometry).prepare()
    eta = jnp.zeros((2, 2))
    epoch = geometry.metric_epoch(eta)
    layer_scale = jnp.asarray([1.0, 3.0, 7.0, 15.0])
    x_face_velocity = jnp.broadcast_to(
        jnp.arange(1, 4, dtype=eta.dtype)[:, None, None] * layer_scale[None, None, :],
        geometry.x_face_shape,
    )
    y_face_velocity = jnp.broadcast_to(
        (20.0 + layer_scale)[None, None, :],
        geometry.y_face_shape,
    )
    state = prepared.initialize_state(
        eta,
        transports=(
            x_face_velocity * epoch.x_face_area,
            y_face_velocity * epoch.y_face_area,
        ),
    )
    result = ocean_api.PassiveOceanTrajectoryPlan(prepared, 1).advect(
        jnp.asarray([[0.25, 0.25, -7.0]]), state, 0.0, 1
    )

    view = prepared.view(state)
    vertical_face_velocity = view.vertical_flux / geometry.cell_area[..., None]
    expected = jnp.asarray(
        [
            0.5 * (view.velocity[0][0, 0, 2] + view.velocity[0][1, 0, 2]),
            0.5 * (view.velocity[1][0, 0, 2] + view.velocity[1][0, 1, 2]),
            0.5 * (vertical_face_velocity[0, 0, 2] + vertical_face_velocity[0, 0, 3]),
        ]
    )
    uniform_grid_choice = jnp.asarray(
        [
            0.5 * (view.velocity[0][0, 0, 1] + view.velocity[0][1, 0, 1]),
            0.5 * (view.velocity[1][0, 0, 1] + view.velocity[1][0, 1, 1]),
            0.5 * (vertical_face_velocity[0, 0, 1] + vertical_face_velocity[0, 0, 2]),
        ]
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.sampled_velocity[0, 0], expected)
    assert not bool(jnp.allclose(result.sampled_velocity[0, 0], uniform_grid_choice))


def test_passive_trajectory_invalidates_boundary_crossing_final_sample():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(2) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (1.0, 1.0, 0.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        finite_volume, jnp.full((2, 2), 10.0)
    ).prepare()
    prepared = ocean_api.HydrostaticPrimitiveEquationPlan(geometry).prepare()
    eta = jnp.zeros((2, 2))
    epoch = geometry.metric_epoch(eta)
    state = prepared.initialize_state(
        eta,
        transports=(epoch.x_face_area, jnp.zeros(geometry.y_face_shape)),
    )
    result = ocean_api.PassiveOceanTrajectoryPlan(prepared, 1).advect(
        jnp.asarray([[0.7, 0.5, -5.0]]), state, 0.1, 1
    )

    np.testing.assert_allclose(result.trajectory.states[0, 1, 0], 0.8)
    np.testing.assert_array_equal(
        result.trajectory.sample_valid[0], jnp.asarray([True, False])
    )
    assert bool(result.exited[0])
    assert result.active_steps[0] == 1
    assert not bool(result.successful)
