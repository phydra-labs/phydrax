#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.lattice_boltzmann import (
    CompiledLatticeBoltzmannLinkTopology,
    D3Q27,
    DampedLocalRootSolver,
    LatticeBoltzmannBoundaryStage,
    LatticeBoltzmannLinkOwner,
    VelocityDependentAccelerationPlan,
    VelocityDependentAccelerationProblem,
)


def test_compiled_link_topology_enforces_write_once_stages():
    shape = (3, 3, 9)
    owner = np.full(shape, int(LatticeBoltzmannLinkOwner.LOCAL), dtype=np.int8)
    owner[0, 1, 1] = int(LatticeBoltzmannLinkOwner.HALFWAY)
    parameter = np.full(shape, -1, dtype=np.int32)
    axis = np.full(shape, -1, dtype=np.int8)
    sign = np.zeros(shape, dtype=np.int8)
    body = np.full(shape, -1, dtype=np.int32)
    body[0, 1, 1] = 0
    fraction = np.zeros(shape)
    fraction[0, 1, 1] = 0.5
    topology = CompiledLatticeBoltzmannLinkTopology(
        owner,
        parameter,
        axis,
        sign,
        body,
        fraction,
        np.ones(shape[:-1], dtype=bool),
        topology_id="unit-topology",
    )
    state = topology.begin(jnp.zeros(shape))
    streamed = topology.commit(
        state,
        jnp.ones(shape),
        LatticeBoltzmannBoundaryStage.STREAM,
        (
            LatticeBoltzmannLinkOwner.LOCAL,
            LatticeBoltzmannLinkOwner.PERIODIC,
            LatticeBoltzmannLinkOwner.HALO,
        ),
    )

    assert bool(streamed.written[1, 1, 0])
    assert not bool(streamed.written[0, 1, 1])


def test_velocity_dependent_force_matches_linear_drag_root():
    drag = 0.2
    gravity = jnp.asarray((0.01, -0.02))
    plan = VelocityDependentAccelerationPlan(
        lambda time, coordinates, velocity, parameters: gravity - drag * velocity,
        acceleration_id="linear-drag",
    )
    density = jnp.ones((2, 2))
    raw_momentum = jnp.zeros((2, 2, 2))
    problem = VelocityDependentAccelerationProblem(
        jnp.asarray(0.0),
        jnp.zeros((2, 2, 2)),
        density,
        raw_momentum,
    )
    result = plan.solve(problem, DampedLocalRootSolver(iterations=64, damping=1.0))
    expected = 0.5 * gravity / (1.0 + 0.5 * drag)

    assert jnp.all(result.root.converged)
    expected_field = jnp.broadcast_to(expected, result.velocity.shape)
    np.testing.assert_allclose(result.velocity, expected_field, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.force_density, gravity - drag * expected_field)


def test_d3q27_body_diagonal_has_certified_opposite():
    lattice = D3Q27()
    assert np.array_equal(
        np.asarray(lattice.velocities)[np.asarray(lattice.opposite)],
        -np.asarray(lattice.velocities),
    )


def test_staged_boundary_executes_stream_then_wall_without_overwrite():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    shape = discretization.population_shape
    owner = np.full(shape, int(LatticeBoltzmannLinkOwner.LOCAL), dtype=np.int8)
    owner[0, 1, 1] = int(LatticeBoltzmannLinkOwner.HALFWAY)
    parameter = np.full(shape, -1, dtype=np.int32)
    axis = np.full(shape, -1, dtype=np.int8)
    sign = np.zeros(shape, dtype=np.int8)
    body = np.full(shape, -1, dtype=np.int32)
    body[0, 1, 1] = 0
    fraction = np.zeros(shape)
    fraction[0, 1, 1] = 0.5
    topology = CompiledLatticeBoltzmannLinkTopology(
        owner,
        parameter,
        axis,
        sign,
        body,
        fraction,
        np.ones(shape[:-1], dtype=bool),
        topology_id="staged-wall-test",
    )
    boundary = phx.discretization.PreparedStagedLatticeBoltzmannBoundary(
        discretization,
        topology,
        body_ids=("wall",),
    )
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    parameters = phx.discretization.LatticeBoltzmannBoundaryParameters(
        body_centers=jnp.asarray(((0.0, 0.5),)),
        body_linear_velocities=jnp.zeros((1, 2)),
        body_angular_velocities=jnp.zeros((1, 1)),
        half_force_density=jnp.zeros(grid.shape + (2,)),
        time_step=jnp.asarray(0.01),
    )
    result = boundary.apply(
        populations,
        jnp.ones(grid.shape),
        boundary.initial_state(populations),
        parameters,
    )

    assert result.populations.shape == populations.shape
    assert jnp.all(jnp.isfinite(result.populations))
    np.testing.assert_allclose(
        result.ledger.fluid_impulse,
        -result.ledger.body_impulse,
        atol=1e-14,
    )


def test_typed_face_compiler_assigns_open_parameter_order_and_corner_precedence():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6),
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    faces = (
        phx.discretization.LatticeBoltzmannFaceBoundary(
            "x",
            "lower",
            LatticeBoltzmannLinkOwner.VELOCITY,
            parameter_id="inlet",
            flow_direction="inlet",
        ),
        phx.discretization.LatticeBoltzmannFaceBoundary(
            "x",
            "upper",
            LatticeBoltzmannLinkOwner.PRESSURE,
            parameter_id="outlet",
        ),
    )
    corner_rules = tuple(
        phx.discretization.LatticeBoltzmannCornerRule(
            (("x", x_side), ("y", y_side)),
            ("x", x_side),
        )
        for x_side in ("lower", "upper")
        for y_side in ("lower", "upper")
    )
    plan = phx.discretization.compile_staged_lattice_boltzmann_boundary(
        discretization,
        faces=faces,
        corner_rules=corner_rules,
    )

    assert plan.velocity_parameter_ids == ("inlet",)
    assert plan.pressure_parameter_ids == ("outlet",)
    assert plan.topology.owner_counts[int(LatticeBoltzmannLinkOwner.VELOCITY)] > 0
    assert plan.topology.owner_counts[int(LatticeBoltzmannLinkOwner.PRESSURE)] > 0
    boundary = plan.prepare(discretization)
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    result = boundary.apply(
        populations,
        jnp.ones(grid.shape),
        boundary.initial_state(populations),
        phx.discretization.LatticeBoltzmannBoundaryParameters(
            velocity_targets=jnp.asarray(((0.01, 0.0),)),
            pressure_densities=jnp.asarray((1.0,)),
            pressure_tangential_velocities=jnp.zeros((1, 2)),
            half_force_density=jnp.zeros(grid.shape + (2,)),
        ),
    )
    assert jnp.all(jnp.isfinite(result.populations))


def test_fixed_sdf_compiles_curved_bouzidi_links_and_body_identity():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    coordinates = grid.points.reshape(grid.shape + (2,))
    signed_distance = (
        jnp.sqrt((coordinates[..., 0] - 0.5) ** 2 + (coordinates[..., 1] - 0.5) ** 2)
        - 0.2
    )
    geometry = phx.discretization.FixedSDFLinkGeometry(
        discretization,
        signed_distance,
        body_names=("obstacle",),
    )
    plan = phx.discretization.compile_staged_lattice_boltzmann_boundary(
        discretization,
        geometry=geometry,
        body_boundaries=(
            phx.discretization.LatticeBoltzmannBodyBoundary(
                "obstacle", LatticeBoltzmannLinkOwner.BOUZIDI
            ),
        ),
    )
    curved = plan.topology.owner == int(LatticeBoltzmannLinkOwner.BOUZIDI)

    assert plan.body_ids == ("obstacle",)
    assert jnp.any(curved)
    assert jnp.any(jnp.abs(plan.topology.link_fraction[curved] - 0.5) > 1e-6)


def test_convective_outlet_updates_explicit_boundary_history():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6),
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    faces = (
        phx.discretization.LatticeBoltzmannFaceBoundary(
            "x",
            "lower",
            LatticeBoltzmannLinkOwner.VELOCITY,
            parameter_id="inlet",
            flow_direction="inlet",
        ),
        phx.discretization.LatticeBoltzmannFaceBoundary(
            "x",
            "upper",
            LatticeBoltzmannLinkOwner.CONVECTIVE,
            parameter_id="convective-outlet",
        ),
    )
    corner_rules = tuple(
        phx.discretization.LatticeBoltzmannCornerRule(
            (("x", x_side), ("y", y_side)),
            ("x", x_side),
        )
        for x_side in ("lower", "upper")
        for y_side in ("lower", "upper")
    )
    plan = phx.discretization.compile_staged_lattice_boltzmann_boundary(
        discretization,
        faces=faces,
        corner_rules=corner_rules,
    )
    boundary = plan.prepare(discretization)
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    result = boundary.apply(
        populations,
        jnp.ones(grid.shape),
        boundary.initial_state(),
        phx.discretization.LatticeBoltzmannBoundaryParameters(
            velocity_targets=jnp.asarray(((0.01, 0.0),)),
            convective_speeds=jnp.asarray((0.5,)),
            half_force_density=jnp.zeros(grid.shape + (2,)),
        ),
    )
    convective = plan.topology.owner == int(LatticeBoltzmannLinkOwner.CONVECTIVE)

    assert plan.convective_parameter_ids == ("convective-outlet",)
    assert jnp.all(result.state.convective_initialized[convective])
    np.testing.assert_allclose(
        result.state.convective_history[convective],
        result.populations[convective],
        atol=1e-14,
    )
