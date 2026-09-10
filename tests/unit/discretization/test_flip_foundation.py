#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mac(count=8, *, periodic=False):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    return grid, finite_volume, operators, boundaries


def test_free_surface_projection_sets_air_pressure_and_projects_liquid_divergence():
    _, finite_volume, operators, boundaries = _mac()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        operators, boundaries=boundaries, tolerance=1e-7
    )
    velocity = tuple(
        jnp.sin(jnp.arange(math.prod(layout.shape), dtype=float)).reshape(layout.shape)
        * 1e-3
        for layout in finite_volume.face_layouts
    )
    liquid = jnp.zeros(finite_volume.cell_shape, dtype=bool).at[2:6, 2:6].set(True)
    result = projection.project(velocity, liquid, 1.0e-3)
    assert result.successful
    assert result.active_divergence_norm < 1e-6
    assert result.air_pressure_defect == 0.0
    np.testing.assert_allclose(jnp.where(~liquid, result.pressure, 0.0), 0.0)


def test_free_surface_projection_rejects_empty_liquid_mask():
    _, finite_volume, operators, boundaries = _mac()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(operators, boundaries=boundaries)
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    result = projection.project(
        velocity, jnp.zeros(finite_volume.cell_shape, dtype=bool), 1.0e-3
    )
    assert not result.successful


def test_flip_transfer_preserves_cell_volume_and_face_momentum_numerators():
    _, finite_volume, operators, _ = _mac(periodic=True)
    position = jnp.asarray([[0.2, 0.2], [0.4, 0.3], [0.7, 0.65]])
    velocity = jnp.asarray([[1.0, 0.2], [-0.3, 0.7], [0.5, -0.4]])
    mass = jnp.asarray([1.0, 2.0, 1.5])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), mass, ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(operators).prepare(
        particles
    )
    routes = transfer.build(position)
    p2g = transfer.particle_to_grid(routes, velocity, 2.0)
    assert p2g.successful
    np.testing.assert_allclose(jnp.sum(p2g.particle_volume_content), jnp.sum(mass) / 2.0)
    for axis, momentum in enumerate(p2g.face_momentum):
        np.testing.assert_allclose(
            jnp.sum(momentum), jnp.sum(mass * velocity[:, axis]), atol=1e-12
        )
    g2p = transfer.grid_to_particle(routes, p2g.velocity, p2g.velocity)
    assert g2p.successful
    np.testing.assert_allclose(g2p.flip_increment, 0.0, atol=1e-14)


def test_flip_method_validates_explicit_pic_fraction():
    with pytest.raises(ValueError, match="pic_fraction"):
        phx.discretization.flip.FLIPMethodPlan(1.1)
    assert phx.discretization.flip.FLIPMethodPlan(0.0).pic_fraction == 0.0
    assert phx.discretization.flip.FLIPMethodPlan(1.0).pic_fraction == 1.0


def test_sparse_flip_transfer_uses_compact_cell_and_face_storage():
    grid_plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    index = grid_plan.prepare_index_space(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.2, 0.2], [0.25, 0.22]])
    velocity = jnp.asarray([[0.1, 0.0], [0.1, 0.0]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    cell_topology = phx.discretization.SparseBlockTopologyPlan(
        index, (2, 2), 8, layout=index.cells()
    )
    face_topologies = tuple(
        phx.discretization.SparseBlockTopologyPlan(
            index, (2, 2), 8, layout=index.faces(axis)
        )
        for axis in index.axis_names
    )
    transfer = phx.discretization.SparseFLIPParticleTransferPlan(
        index, cell_topology, face_topologies
    ).prepare(particles)
    routes = transfer.build(position)
    p2g = transfer.particle_to_grid(routes, velocity, 1.0)
    g2p = transfer.grid_to_particle(routes, p2g.velocity, p2g.velocity)

    assert bool(routes.successful)
    assert bool(p2g.successful)
    assert p2g.particle_volume_content.shape == (32,)
    assert all(value.shape == (32,) for value in p2g.velocity)
    np.testing.assert_allclose(g2p.pic_velocity, velocity, atol=1.0e-12)
    np.testing.assert_allclose(g2p.flip_increment, 0.0, atol=1.0e-14)


def test_sparse_flip_pressure_and_particle_step_commit_atomically():
    grid_plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    index = grid_plan.prepare_index_space(bounds)
    position = jnp.asarray([[0.2, 0.2], [0.25, 0.22]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    closure = tuple((row, column) for row in (-1, 0, 1) for column in (-1, 0, 1))
    cell = phx.discretization.SparseBlockTopologyPlan(
        index,
        (2, 2),
        16,
        layout=index.cells(),
        closure_offsets=closure,
    )
    faces = tuple(
        phx.discretization.SparseBlockTopologyPlan(
            index,
            (2, 2),
            16,
            layout=index.faces(axis),
            closure_offsets=closure,
        )
        for axis in index.axis_names
    )
    transfer = phx.discretization.SparseFLIPParticleTransferPlan(
        index, cell, faces
    ).prepare(particles)
    projection = phx.solver.SparseMACFreeSurfaceProjectionPlan(
        transfer, tolerance=1.0e-7, maximum_iterations=100
    )
    problem = phx.equations.FLIPProblemIR("sparse-flip-step", 1.0, jnp.zeros((2,)))
    compiled = phx.equations.compile_sparse_flip_problem(
        problem,
        transfer,
        projection,
        phx.discretization.FLIPMethodPlan(
            1.0,
            liquid_fraction_threshold=0.01,
            cfl_fraction=1.0,
        ),
    )
    state = compiled.initialize_state(
        position, jnp.broadcast_to(jnp.asarray((0.01, 0.0)), position.shape)
    )
    result = compiled.step_detailed(state, 0.001)

    assert bool(result.successful)
    assert bool(result.projection.topology_complete)
    assert result.diagnostics.divergence_norm < 1.0e-8
    assert int(result.accepted_state.accepted_step) == 1
