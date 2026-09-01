#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _prepared_coarse_graining():
    axis = phx.discretization.UniformAxisSpec(5)
    grid = phx.discretization.TensorGridPlan((axis, axis), axis_names=("x", "y")).prepare(
        jnp.asarray([[-1.0, -1.0], [1.0, 1.0]])
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]),
        jnp.asarray([2.0, 2.0]),
        ambient_dimension=2,
    ).prepare()
    relation = (
        phx.discretization.DenseParticleNeighborhoodPlan(1)
        .prepare(particles)
        .build(jnp.asarray([[-0.25, 0.0], [0.25, 0.0]]))
        .pair_relation
    )
    plan = phx.discretization.ParticleCoarseGrainingPlan(
        phx.discretization.ParticleGridSplatPlan(grid),
        quadrature_order=4,
    )
    return plan.prepare(particles, 1), relation


def test_particle_coarse_graining_conserves_primary_fields_and_pair_virial():
    prepared, relation = _prepared_coarse_graining()
    position = jnp.asarray([[-0.25, 0.0], [0.25, 0.0]])
    velocity = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    displacement = jnp.asarray([[-0.5, 0.0]])
    pair_force = jnp.asarray([[-2.0, 0.0]])
    result = prepared.evaluate(
        position,
        velocity,
        jnp.asarray([2.0, 2.0]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([True, True]),
        relation,
        displacement,
        pair_force,
        jnp.asarray([True]),
        constituent_weights=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert result.successful
    assert jnp.allclose(jnp.sum(result.mass_content), 4.0)
    assert jnp.allclose(jnp.sum(result.volume_content), 0.2)
    assert jnp.allclose(
        jnp.sum(result.momentum_content, axis=(0, 1)), jnp.asarray([4.0, 0.0])
    )
    assert jnp.allclose(
        jnp.sum(result.contact_stress_content, axis=(0, 1)),
        jnp.asarray([[-1.0, 0.0], [0.0, 0.0]]),
        atol=1.0e-12,
    )
    assert jnp.max(jnp.abs(result.kinetic_stress)) < 1.0e-12
    target_measure = prepared.particle_splat.target_measure.weights.reshape((5, 5, 1))
    assert jnp.allclose(jnp.sum(result.partial_mass_density * target_measure), 4.0)
    assert result.maximum_particle_balance_defect < 1.0e-12
    assert result.contact_stress_balance_defect < 1.0e-12

    shifted = prepared.evaluate(
        position,
        velocity + jnp.asarray([3.0, -2.0]),
        jnp.asarray([2.0, 2.0]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([True, True]),
        relation,
        displacement,
        pair_force,
        jnp.asarray([True]),
    )
    assert jnp.allclose(shifted.kinetic_stress, result.kinetic_stress, atol=1.0e-12)
    assert jnp.allclose(shifted.contact_stress, result.contact_stress)


def test_segment_stress_is_differentiable_with_frozen_routes():
    prepared, relation = _prepared_coarse_graining()
    position = jnp.asarray([[-0.25, 0.0], [0.25, 0.0]])
    displacement = jnp.asarray([[-0.5, 0.0]])

    def integrated_contact_stress(force_x):
        fields = prepared.evaluate(
            position,
            jnp.zeros_like(position),
            jnp.asarray([2.0, 2.0]),
            jnp.asarray([0.1, 0.1]),
            jnp.asarray([True, True]),
            relation,
            displacement,
            jnp.asarray([[force_x, 0.0]]),
            jnp.asarray([True]),
        )
        return jnp.sum(fields.contact_stress_content[..., 0, 0])

    derivative = jax.grad(integrated_contact_stress)(jnp.asarray(-2.0))
    assert jnp.allclose(derivative, 0.5, atol=1.0e-12)
