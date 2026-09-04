#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _schema(phases=(None, None)):
    selected = (
        (phx.equations.ChemicalPhaseKind.SOLID, phx.equations.ChemicalPhaseKind.SOLID)
        if phases == (None, None)
        else phases
    )
    return phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        selected,
        jnp.asarray([0.01, 0.01]),
        ("X",),
        jnp.asarray([[1, 1]]),
        jnp.zeros_like(jnp.asarray([0.01, 0.01]), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )


def _batch(species_count=2, shells=3):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    plan = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, shells
        ),
        species_count,
    )
    return particles, plan, plan.prepare(particles)


def _conversion_role_value(state, scale):
    return jax.tree.map(
        lambda leaf: (
            jnp.full_like(leaf, scale)
            if eqx.is_inexact_array(leaf)
            else jnp.zeros_like(leaf)
        ),
        state,
    )


def _conversion_tree_pair(reference, covector, vector):
    products = (
        jnp.vdot(covector_leaf, vector_leaf)
        for base, covector_leaf, vector_leaf in zip(
            jax.tree.leaves(reference),
            jax.tree.leaves(covector),
            jax.tree.leaves(vector),
            strict=True,
        )
        if eqx.is_inexact_array(base)
    )
    return sum(products, start=jnp.asarray(0.0))


def test_particle_conversion_geometry_certifies_four_spaces_and_frozen_routes():
    _, _, batch = _batch(shells=2)
    batch_state = phx.discretization.initialize_particle_internal_batch(
        batch,
        jnp.ones((1, 2)),
        jnp.ones((1, 2, 2)),
        jnp.full((1, 2), 0.2),
        jnp.ones((1, 2)),
        jnp.ones((1,)),
    )
    state = phx.discretization.initialize_particle_conversion_state((batch_state,))
    geometry = phx.discretization.ParticleConversionStateGeometry(state.state_id)
    local = _conversion_role_value(state, 1.0e-4)
    direction = _conversion_role_value(state, -2.0e-4)
    cotangent = _conversion_role_value(state, 3.0e-4)

    point = geometry.retract(state, local)
    recovered = geometry.inverse_retract(state, point)
    for expected, actual in zip(
        jax.tree.leaves(local),
        jax.tree.leaves(recovered),
        strict=True,
    ):
        assert jnp.allclose(actual, expected)

    pushed = geometry.retraction_jvp(state, local, direction)
    inverse_pushed = geometry.retraction_inverse_jvp(state, point, pushed)
    pulled = geometry.retraction_vjp(state, local, cotangent)
    assert _conversion_tree_pair(state, cotangent, pushed) == pytest.approx(
        _conversion_tree_pair(state, pulled, direction)
    )
    for expected, actual in zip(
        jax.tree.leaves(direction),
        jax.tree.leaves(inverse_pushed),
        strict=True,
    ):
        assert jnp.allclose(actual, expected)

    transported = geometry.transport_tangent(state, point, pushed)
    transport_pullback = geometry.transport_cotangent_pullback(
        state,
        point,
        cotangent,
    )
    assert _conversion_tree_pair(
        state,
        cotangent,
        transported,
    ) == pytest.approx(_conversion_tree_pair(state, transport_pullback, pushed))
    assert jnp.array_equal(point.batches[0].active, state.batches[0].active)

    incompatible = eqx.tree_at(
        lambda value: value.batches[0].active,
        point,
        point.batches[0].active.at[0].set(False),
    )
    with pytest.raises(Exception, match="incompatible frozen route"):
        geometry.inverse_retract(state, incompatible)


def test_thermodynamic_inversion_and_radial_transport_are_conservative():
    schema = _schema()
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema,
            jnp.asarray([[10.0, 1.0e-3], [12.0, 5.0e-4]]),
            jnp.asarray([0.0, -100.0]),
        ),
    )
    transport = phx.equations.ParticleTransportMaterialPlan(
        schema,
        jnp.asarray([1.0, 0.5]),
        jnp.asarray([1.0e-6, 2.0e-6]),
    )
    material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermodynamics, transport
    )
    _, _, batch = _batch()
    species = jnp.asarray([[[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]])
    target_temperature = jnp.asarray([[300.0, 400.0, 500.0]])
    energy = thermodynamics.energy_from_temperature(target_temperature, species)
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        energy,
        species,
        jnp.full((1, 3), 0.2),
        jnp.ones((1, 3)),
        jnp.asarray([1.0]),
    )
    metrics = batch.mesh.metrics(state.outer_scale)
    recovered = thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    assert recovered.successful
    assert jnp.allclose(recovered.temperature, target_temperature, rtol=1.0e-10)

    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([600.0]),
        jnp.zeros((1, 2)),
        jnp.asarray([2.0]),
        jnp.zeros((1, 2)),
        jnp.zeros((1,)),
        jnp.zeros((1, 2)),
    )
    evaluation = phx.equations.evaluate_particle_transport(
        batch, state, material, boundary
    )
    assert evaluation.successful
    assert jnp.abs(evaluation.internal_energy_residual) < 1.0e-10
    assert jnp.max(jnp.abs(evaluation.internal_species_residual)) < 1.0e-12
    assert evaluation.entropy_production >= 0.0


def test_reaction_network_conserves_elements_and_reaction_energy():
    schema = _schema()
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([10.0, 10.0]), jnp.asarray([0.0, -100.0])
        )
    )
    particles, plan, batch = _batch(shells=2)
    del particles, plan
    species = jnp.asarray([[[1.0, 0.0], [1.0, 0.0]]])
    energy = thermodynamics.energy_from_temperature(jnp.full((1, 2), 400.0), species)
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        energy,
        species,
        jnp.full((1, 2), 0.2),
        jnp.ones((1, 2)),
        jnp.asarray([1.0]),
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "particle-conversion",
        schema,
        thermodynamics.species_thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(2.0),
            ),
        ),
    ).prepare()
    reaction = phx.equations.ParticleReactionProcessPlan(mechanism)
    evaluation = reaction.evaluate(batch, state, thermodynamics)
    assert evaluation.successful
    assert jnp.max(jnp.abs(evaluation.element_residual)) < 1.0e-12
    assert jnp.all(evaluation.extent_rate >= 0.0)
    assert jnp.all(evaluation.internal_energy_rate > 0.0)


def test_evaporation_and_shrinking_core_report_exhaustion_restrictions():
    schema = _schema(
        (phx.equations.ChemicalPhaseKind.LIQUID, phx.equations.ChemicalPhaseKind.GAS)
    )
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([75.0, 35.0]), jnp.asarray([0.0, 0.0])
        )
    )
    _, _, batch = _batch(shells=1)
    species = jnp.asarray([[[1.0, 0.0]]])
    energy = thermodynamics.energy_from_temperature(jnp.asarray([[373.15]]), species)
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        energy,
        species,
        jnp.asarray([[0.5]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([1.0]),
    )
    metrics = batch.mesh.metrics(state.outer_scale)
    thermo = thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    evaporation = phx.equations.EvaporationPhaseChangePlan(
        schema,
        0,
        1,
        0.1,
        40000.0,
        phx.equations.AntoineSaturationPressurePlan(8.07131, 1730.63, 233.426),
    )
    phase = evaporation.evaluate(batch, state, thermo, metrics)
    assert phase.successful
    assert jnp.max(jnp.abs(schema.element_amount(phase.species_amount_rate))) < 1.0e-12
    assert phase.explicit_step_restriction > 0.0

    shrinking = phx.equations.ShrinkingCoreConversionPlan(
        1.0, 1.0, 1000.0, 0.1, 1.0e-6, 0.01
    )
    core = shrinking.evaluate(
        phx.equations.ShrinkingCoreState(jnp.asarray([1.0])),
        jnp.asarray([0.1]),
        jnp.asarray([1.0]),
    )
    assert core.successful
    assert core.core_radius_rate[0] < 0.0
    assert core.explicit_step_restriction > 0.0


def test_continuum_exchange_deposits_exact_opposite_heat_and_species():
    schema = _schema()
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([10.0, 10.0]), jnp.asarray([0.0, 0.0])
        )
    )
    particles, plan, batch = _batch(shells=1)
    species = jnp.asarray([[[1.0, 0.0]]])
    energy = thermodynamics.energy_from_temperature(jnp.asarray([[300.0]]), species)
    batch_state = phx.discretization.initialize_particle_internal_batch(
        batch,
        energy,
        species,
        jnp.asarray([[0.2]]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
    )
    state = phx.discretization.initialize_particle_conversion_state((batch_state,))
    mesh = phx.discretization.CellMesh(
        jnp.asarray(
            (
                (-0.25, -0.25, -0.25),
                (0.75, -0.25, -0.25),
                (-0.25, 0.75, -0.25),
                (-0.25, -0.25, 0.75),
            )
        ),
        (
            phx.discretization.CellBlock(
                "cell", "tetrahedron", jnp.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    measure = phx.discretization.DiscreteMeasure(
        "cell_volume",
        mesh.support.support_id,
        mesh.topology.entities(3).entity_set_id,
        jnp.asarray((1.0,)),
    )
    transfer = phx.discretization.MeshCompactKernelSplatAssignment(0.5, 1).prepare(
        phx.discretization.MeshSplatTarget(mesh, entity_dimension=3, measure=measure),
        jnp.zeros((particles.capacity, 3)),
        particles.active_mask,
        particles.particle_ids,
    )
    exchange = phx.equations.ParticleContinuumExchangePlan(
        transfer,
        jnp.asarray([2.0]),
        jnp.asarray([[0.1, 0.2]]),
        schema_id=schema.schema_id,
    )
    evaluation = exchange.evaluate(
        jnp.asarray([[0.0, 0.0, 0.0]]),
        (batch,),
        state,
        (thermodynamics,),
        jnp.asarray([500.0]),
        jnp.asarray([[0.0, 1.0]]),
    )
    assert evaluation.successful
    assert jnp.abs(evaluation.energy_residual) < 1.0e-12
    assert jnp.max(jnp.abs(evaluation.species_residual)) < 1.0e-12
