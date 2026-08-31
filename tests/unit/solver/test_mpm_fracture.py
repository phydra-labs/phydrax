#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _fracture_case():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(10, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.3, 0.3], [0.45, 0.35], [0.35, 0.5]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(2)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("fracture", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    mechanics = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    return compiled, arguments, mechanics


def test_phase_field_material_degrades_tension_and_updates_history():
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
    )
    deformation = jnp.asarray([[[1.1, 0.0], [0.0, 1.0]]])
    intact = material.evaluate(
        deformation, jnp.asarray([[0.0, 0.0]]), jnp.asarray([1.0]), parameters, 0.0, 0.01
    )
    damaged = material.evaluate(
        deformation, jnp.asarray([[0.8, 0.0]]), jnp.asarray([1.0]), parameters, 0.0, 0.01
    )

    assert intact.trial_state[0, 1] > 0.0
    assert jnp.linalg.norm(damaged.first_piola) < jnp.linalg.norm(intact.first_piola)
    assert damaged.reference_energy_density[0] < intact.reference_energy_density[0]


def test_phase_field_step_is_irreversible_and_transactional():
    compiled, arguments, mechanics = _fracture_case()
    prepared = phx.solver.PreparedMPMPhaseFieldDynamics(
        compiled.dynamics,
        phx.solver.MPMPhaseFieldFracturePlan(
            maximum_damage_iterations=200, tolerance=1e-6
        ),
    )
    state = prepared.initialize_state(mechanics)
    detail = prepared.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    assert bool(detail.evidence.irreversibility_valid)
    assert jnp.all(detail.accepted_state.damage >= state.damage)
    assert detail.evidence.fracture_energy >= 0.0


def test_field_partition_and_cpic_are_distinct_topology_paths():
    partition = phx.discretization.MPMFieldPartitionFracturePlan(2)
    topology = partition.update(
        jnp.asarray((0.2, 0.99, 0.99)),
        jnp.asarray((-1.0, -0.2, 0.3)),
        jnp.zeros((3,), dtype=jnp.int32),
        0,
    )
    assert bool(topology.successful)
    np.testing.assert_array_equal(topology.velocity_field_slots, (0, 0, 1))
    assert int(topology.topology_generation) == 1

    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    position = jnp.asarray([[0.45, 0.5], [0.55, 0.5]])
    routes = splat.build(position)
    x_coordinates, _ = jnp.meshgrid(*splat.layout.coordinates_by_axis, indexing="ij")
    node_tags = jnp.where(x_coordinates.reshape((-1,)) < 0.5, 0, 1).astype(jnp.int32)
    cpic = phx.discretization.CPICFracturePlan(2)
    compatibility = cpic.build(
        routes, jnp.asarray((0, 1)), node_tags, topology.topology_generation
    )
    grid_velocity = jnp.zeros(splat.target_shape + (2,))
    particle_velocity = jnp.asarray(((0.1, 0.0), (-0.1, 0.0)))
    affine = jnp.zeros((2, 2, 2))
    routed = cpic.route_velocities(
        compatibility, routes, grid_velocity, particle_velocity, affine
    )

    assert bool(compatibility.successful)
    incompatible = ~compatibility.compatible
    expected_ghost = jnp.broadcast_to(particle_velocity[:, None, :], routed.shape)
    np.testing.assert_allclose(
        jnp.where(incompatible[..., None], routed, 0.0),
        jnp.where(incompatible[..., None], expected_ghost, 0.0),
    )
