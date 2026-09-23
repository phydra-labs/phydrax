#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.discrete_velocity import (
    AdaptiveGaugePlan,
    CompressibleKineticPrecisionPolicy,
    d3q33_filtered_rule,
    d3q39_guided_rule,
    d3q343_entropic_rule,
    entropic_d3q343_plan,
    FullRangeQuasiEquilibriumPlan,
    guided_d3q39_plan,
    IntegerKineticFramePlan,
    IntegerLatticeTransportPlan,
    KineticAMRTransferPlan,
    KineticAuxiliaryState,
    KineticRadiationAblationPlan,
    KineticSpeciesTransportPlan,
    KineticSpectralAnalysisPlan,
    KineticStoragePlan,
    KineticVelocityPartitionPlan,
    MovingKineticGeometryPlan,
    PredictiveKineticRefinementPlan,
    remap_kinetic_frame,
)
from phydrax.discretization.discrete_velocity._filtered_d3q33 import (
    FilteredD3Q33Plan,
)


def _uniform(model, shape=(8, 8, 8)):
    return model.initialize(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        jnp.ones(shape),
    )


def test_production_velocity_rules_have_canonical_population_identity():
    rules = (d3q39_guided_rule(), d3q343_entropic_rule(), d3q33_filtered_rule())
    assert tuple(rule.population_count for rule in rules) == (39, 343, 33)
    assert tuple(rule.dual_dimension for rule in rules) == (12, 4, 4)
    for rule in rules:
        np.testing.assert_array_equal(
            rule.opposite[rule.opposite], np.arange(rule.population_count)
        )
        np.testing.assert_allclose(jnp.sum(rule.base_probabilities), 1.0, atol=2e-14)


def test_guided_d3q39_collision_and_full_prandtl_closure_conserve_state():
    model = guided_d3q39_plan(gamma=1.4)
    state = _uniform(model)
    direct = model.collide(state, 1.0)
    below, below_evidence = FullRangeQuasiEquilibriumPlan(
        model, prandtl_number=0.72
    ).collide(state, 1.0)
    above, above_evidence = FullRangeQuasiEquilibriumPlan(
        model, prandtl_number=1.4
    ).collide(state, 1.0)

    assert jnp.all(direct.successful)
    assert jnp.all(below.successful)
    assert jnp.all(above.successful)
    assert below_evidence.slow_family == "heat-flux"
    assert above_evidence.slow_family == "stress"
    np.testing.assert_allclose(direct.conservation.mass_defect, 0.0, atol=1e-10)
    np.testing.assert_allclose(direct.conservation.energy_defect, 0.0, atol=1e-9)


def test_entropic_d3q343_velocity_partition_roundtrips_and_collides():
    model = entropic_d3q343_plan()
    state = _uniform(model, shape=(1,))
    partition = KineticVelocityPartitionPlan(model.rule, 7)
    shards = partition.partition(state.population("particle"))
    assembled, evidence = partition.assemble(shards)
    collision = model.collide(state, 0.5)

    assert bool(evidence.successful)
    assert jnp.all(collision.successful)
    np.testing.assert_array_equal(assembled, state.population("particle"))


def test_filtered_d3q33_filters_only_nonconserved_moments():
    plan = FilteredD3Q33Plan()
    state = _uniform(plan.model)
    result, evidence = plan.collide(state, 1.0)

    assert jnp.all(result.successful)
    assert jnp.all(evidence.conserved_moment_defect < 1e-9)
    assert plan.filter_indices.size == 15


def test_integer_frame_remap_and_adaptive_gauge_retain_supported_state():
    source = guided_d3q39_plan()
    frame = IntegerKineticFramePlan(source.rule, (1, 0, 0))
    target = type(source)(
        frame.shifted_rule,
        gamma=source.gamma,
        gas_constant=source.gas_constant,
        collision_kind=source.collision_kind,
    )
    state = source.initialize(
        jnp.ones((1,)),
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.ones((1,)),
    )
    remapped = remap_kinetic_frame(source, target, state)
    gauge = AdaptiveGaugePlan()
    _, scale, supported = gauge.gauge(jnp.zeros((1, 3)), jnp.ones((1,)))

    assert jnp.all(remapped.evidence.successful)
    assert jnp.all(supported)
    np.testing.assert_allclose(scale, 1.0)


def test_periodic_transport_amr_precision_and_moving_geometry_preserve_contracts():
    model = guided_d3q39_plan()
    state = _uniform(model)
    transport = IntegerLatticeTransportPlan(
        model.rule,
        state.spatial_shape,
        storage=KineticStoragePlan("aa"),
    )
    streamed, transport_evidence = transport.stream(state)
    transfer = KineticAMRTransferPlan(3)
    coarse = transfer.restrict(transfer.prolong(state))
    precision = CompressibleKineticPrecisionPolicy(
        storage_dtype="float16", compute_dtype="float32", accumulation_dtype="float64"
    )
    encoded = precision.encode(state.population("particle"))
    decoded = precision.decode(encoded, state.population("particle").shape)
    moving = MovingKineticGeometryPlan(state.spatial_shape)
    moved, moving_evidence = moving.update(
        state,
        jnp.ones(state.spatial_shape, dtype=jnp.bool_),
        jnp.ones(state.spatial_shape, dtype=jnp.bool_),
        state,
    )

    assert bool(transport_evidence.successful)
    assert bool(moving_evidence.successful)
    np.testing.assert_allclose(
        streamed.population("particle"), state.population("particle")
    )
    np.testing.assert_allclose(
        coarse.population("particle"), state.population("particle")
    )
    np.testing.assert_allclose(
        decoded, state.population("particle"), rtol=8e-4, atol=1e-7
    )
    np.testing.assert_array_equal(
        moved.population("particle"), state.population("particle")
    )


def test_predictive_refinement_species_transport_radiation_and_spectrum_are_audited():
    model = guided_d3q39_plan()
    state = _uniform(model)
    transport = IntegerLatticeTransportPlan(model.rule, state.spatial_shape)
    refinement = PredictiveKineticRefinementPlan(
        model.rule, refine_threshold=0.5, coarsen_threshold=0.1
    ).evaluate(jnp.zeros(state.spatial_shape))
    species_plan = KineticSpeciesTransportPlan(
        transport,
        jnp.asarray(((1.0, 1.0),)),
        jnp.asarray((0.0, 0.0)),
    )
    species = jnp.broadcast_to(jnp.asarray((0.7, 0.3)), state.spatial_shape + (2,))
    transported, species_evidence = species_plan.advect(state, species)
    auxiliary = KineticAuxiliaryState(
        species,
        jnp.zeros(state.spatial_shape + (1,)),
        jnp.zeros(state.spatial_shape),
        jnp.zeros(state.spatial_shape + (1,)),
        jnp.ones(state.spatial_shape + (1,)),
    )
    updated, exchange = KineticRadiationAblationPlan().exchange(
        auxiliary,
        jnp.zeros(state.spatial_shape + (1,)),
        jnp.zeros_like(species),
        jnp.zeros(state.spatial_shape + (3,)),
        jnp.zeros(state.spatial_shape),
        species_charges=jnp.asarray((0.0, 0.0)),
    )
    spectrum = KineticSpectralAnalysisPlan(state.spatial_shape).evaluate(
        jnp.zeros(state.spatial_shape + (3,))
    )

    assert bool(refinement.successful)
    assert jnp.all(species_evidence.successful)
    assert jnp.all(exchange.successful)
    assert bool(spectrum.finite)
    np.testing.assert_allclose(jnp.sum(transported, axis=-1), 1.0, atol=2e-12)
    np.testing.assert_array_equal(updated.species_densities, species)
