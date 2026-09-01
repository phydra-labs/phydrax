#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mac(count=8):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    return grid, finite_volume, mac


def test_particle_level_set_capillarity_and_ghost_projection():
    grid, finite_volume, mac = _mac()
    position = jnp.asarray([[0.4, 0.4], [0.5, 0.4], [0.4, 0.5], [0.5, 0.5]])
    geometry = phx.discretization.flip.ParticleLevelSetPlan(
        grid, 0.15
    ).evaluate(position, jnp.ones((4,), dtype=bool))
    assert geometry.successful
    capillary = phx.discretization.finite_volume.MACGhostFluidCapillaryPlan(
        0.1, interface_width=0.2
    ).evaluate(geometry)
    assert capillary.successful
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac,
        boundaries=phx.discretization.MACBoundaryPlan(mac).prepare(),
        tolerance=1e-7,
    )
    ghost = phx.solver.MACGhostFluidProjectionPlan(projection)
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    result = ghost.project(
        velocity, geometry, 1e-3, pressure_jump=capillary.pressure_jump
    )
    assert result.successful
    assert result.projection.air_pressure_defect == 0.0


def test_cut_cell_and_variational_viscosity_are_finite_and_dissipative():
    grid, finite_volume, mac = _mac()
    position = jnp.asarray([[0.35, 0.35], [0.55, 0.35], [0.35, 0.55], [0.55, 0.55]])
    interface = phx.discretization.flip.ParticleLevelSetPlan(
        grid, 0.2
    ).evaluate(position, jnp.ones((4,), dtype=bool))

    def sdf(points, time, args):
        del time, args
        return jnp.sqrt(jnp.sum((points - jnp.asarray([0.1, 0.5])) ** 2, axis=-1)) - 0.05

    def wall(points, time, args):
        del time, args
        return jnp.zeros_like(points)

    solid = phx.discretization.finite_volume.MACCutCellGeometryPlan(
        mac, sdf, wall, field_id="solid", interface_width=0.1
    ).evaluate(0.0)
    assert solid.successful
    measures = phx.discretization.finite_volume.MACFreeSurfaceViscousMeasurePlan(
        mac, 1.0
    ).evaluate(interface, 0.2, solid=solid)
    velocity = tuple(
        jnp.full(layout.shape, 1e-3 * (axis + 1))
        for axis, layout in enumerate(finite_volume.face_layouts)
    )
    result = phx.solver.MACVariationalViscosityPlan(
        mac, tolerance=1e-7
    ).solve(velocity, measures, 1e-3)
    assert result.successful
    assert result.dissipation >= 0.0
    assert result.energy_increase < 1e-8


def test_flip_reseeding_preserves_mass_and_momentum():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(6), jnp.ones((6,)), ambient_dimension=2
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles).initialize(
        active_mask=jnp.asarray([True, True, False, False, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    state = phx.discretization.flip.FLIPParticleState(
        jnp.asarray([[0.2, 0.2], [0.25, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    result = phx.discretization.flip.FLIPReseedingPlan(
        2,
        target_per_cell=3,
        minimum_per_cell=2,
        maximum_per_cell=4,
        maximum_events=4,
    ).apply(
        population,
        state,
        jnp.asarray([0, 0, -1, -1, -1, -1]),
        jnp.asarray([[0.2, 0.2], [0.8, 0.8]]),
    )
    assert result.successful
    np.testing.assert_allclose(result.mass_defect, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.momentum_defect, 0.0, atol=1e-12)
