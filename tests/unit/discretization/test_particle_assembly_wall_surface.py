#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


class IntervalGeometry:
    bounds = jnp.asarray([[0.0], [1.0]])

    @staticmethod
    def signed_distance(points):
        x = points[:, 0]
        return jnp.minimum(x, 1.0 - x)

    @staticmethod
    def boundary_normal(points):
        return jnp.where(points[:, :1] < 0.5, -1.0, 1.0)


def test_particle_assembly_and_bipartite_relations_preserve_population_identity():
    fluid = phx.discretization.ParticleSetPlan(
        [0, 1], [0.5, 0.5], ambient_dimension=1, name="fluid"
    ).prepare()
    wall = phx.discretization.ParticleSetPlan(
        [0, 1, 2], [1.0, 1.0, 1.0], ambient_dimension=1, name="wall"
    ).prepare()
    fluid_population = phx.discretization.ParticlePopulation(
        "fluid",
        fluid,
        role="dynamic-fluid",
        state_shape=(2, 3),
    )
    wall_population = phx.discretization.ParticlePopulation(
        "wall", wall, role="static-boundary"
    )
    assembly = phx.discretization.ParticleAssemblyPlan(
        (fluid_population, wall_population)
    )
    interaction = phx.discretization.ParticleInteractionKey(
        fluid_population, wall_population, "wall", reciprocal=True
    )
    prepared = phx.discretization.DenseBipartiteParticleNeighborhoodPlan(6).prepare(
        fluid,
        wall,
        target_population_id=fluid_population.population_id,
        source_population_id=wall_population.population_id,
    )
    state = prepared.build(
        jnp.asarray([[0.2], [0.8]]), jnp.asarray([[0.0], [0.5], [1.0]])
    )

    assert assembly.population("fluid").population_id == fluid_population.population_id
    assert interaction.reciprocal
    assert int(state.pair_count) == 6
    assert state.successful
    assert state.relation.target_population_id == fluid_population.population_id
    assert state.relation.source_population_id == wall_population.population_id


def test_wall_generation_volume_and_adami_reaction_are_finite_and_reciprocal():
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    wall = phx.discretization.WallParticleGenerationPlan(
        IntervalGeometry(), kernel, 0.25, 0.3, layers=1
    ).prepare()
    fluid = phx.discretization.ParticleSetPlan(
        [0, 1], [0.5, 0.5], ambient_dimension=1
    ).prepare()
    relation = (
        phx.discretization.DenseBipartiteParticleNeighborhoodPlan(
            2 * wall.particles.capacity
        )
        .prepare(
            fluid,
            wall.particles,
            target_population_id="fluid",
            source_population_id="wall",
        )
        .build(jnp.asarray([[0.1], [0.9]]), wall.positions)
    )
    material = phx.equations.TaitBarotropicMaterial(1.0, 10.0)
    result = phx.discretization.evaluate_wall_interaction(
        phx.discretization.AdamiWallBoundaryPlan(material, slip="no-slip"),
        wall,
        relation,
        jnp.asarray([[0.1], [0.9]]),
        jnp.asarray([[0.2], [-0.2]]),
        jnp.ones((2,)),
        jnp.zeros((2,)),
        jnp.asarray([0.5, 0.5]),
        kernel,
        0.3,
    )

    assert wall.quality.particle_count > 0
    assert jnp.all(wall.volumes > 0.0)
    assert jnp.all(jnp.isfinite(result.wall_pressure))
    assert jnp.allclose(result.ledger.action_reaction_defect, 0.0, atol=1e-13)
    assert jnp.allclose(
        jnp.sum(result.fluid_force, axis=0) + jnp.sum(result.wall_reaction, axis=0),
        0.0,
        atol=1e-13,
    )


def test_free_surface_detection_marks_truncated_support_and_pressure_correction():
    particles = phx.discretization.ParticleSetPlan(
        np.arange(5), np.full((5,), 0.1), ambient_dimension=1
    ).prepare()
    position = jnp.arange(5, dtype=float)[:, None] * 0.1
    prepared = phx.discretization.DenseParticleNeighborhoodPlan(10).prepare(particles)
    neighborhood = prepared.build(position)
    geometry = phx.discretization.particle_pair_geometry(
        position, neighborhood.pair_relation
    )
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    physical = geometry.valid & (geometry.distance < 0.3)
    surface = phx.discretization.detect_free_surface(
        phx.discretization.FreeSurfaceDetectionPlan(
            completeness_threshold=0.95,
            normal_threshold=0.01,
            cone_angle=1.2,
        ),
        particles,
        jnp.ones((5,)),
        neighborhood.pair_relation,
        geometry,
        physical,
        kernel,
        0.15,
        phx.discretization.ParticleExecutionPolicy(),
    )
    corrected = phx.discretization.FreeSurfacePressurePlan(0.0).apply(
        jnp.ones((5,)), surface
    )

    assert int(jnp.sum(surface.hard_mask)) >= 1
    assert jnp.all((surface.smooth_weight >= 0.0) & (surface.smooth_weight <= 1.0))
    assert jnp.all(jnp.where(surface.hard_mask, corrected == 0.0, True))
