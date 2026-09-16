#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _catalogue():
    return phx.particle_physics.ParticleCatalogueReference(
        source_id="test-pdg",
        provider_release="test",
        checksum="test-checksum",
        citation_url="https://pdg.lbl.gov/",
    )


def _event_plan(event_capacity=2, particle_capacity=4):
    return phx.particle_physics.ParticleEventPlan(
        catalogue=_catalogue(),
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        event_capacity=event_capacity,
        particle_capacity=particle_capacity,
        vertex_capacity=1,
        provider_status_namespace="LHEF",
    )


def test_event_weight_accounting_preserves_signed_statistics_and_overflow():
    weights = phx.particle_physics.EventWeightSet(
        jnp.asarray([[1.0], [-0.5], [0.0]]),
        names=("nominal",),
        variation_kinds=(phx.particle_physics.WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    ledger = phx.particle_physics.summarize_event_weights(
        weights,
        attempted_count=3,
        cross_section=0.5,
        cross_section_uncertainty=0.1,
    )
    assert bool(ledger.successful)
    assert int(ledger.positive_count) == 1
    assert int(ledger.negative_count) == 1
    assert int(ledger.zero_count) == 1
    assert jnp.isclose(ledger.sum_weights, 0.5)
    assert jnp.isclose(ledger.sum_squared_weights, 1.25)

    overflow = phx.particle_physics.summarize_event_weights(
        weights,
        attempted_count=3,
        overflow=jnp.asarray([False, True, False]),
    )
    assert not bool(overflow.successful)
    assert int(overflow.overflow_count) == 1


def test_lhef_roundtrip_preserves_supported_event_semantics():
    text = """<LesHouchesEvents version=\"3.0\">
<init>0 0 0 0 0 0 0 0 3 1\n0 0 0 0</init>
<event>
4 1 2.5 10 0.007 0.118
11 -1 0 0 0 0 0 0 5 5 0 0 9
-11 -1 0 0 0 0 0 0 -5 5 0 0 9
13 1 1 2 0 0 3 0 4 5 0 0 9
-13 1 1 2 0 0 -3 0 -4 5 0 0 9
</event>
</LesHouchesEvents>
"""
    imported = phx.interchange.hep.read_lhef(text, _event_plan())
    assert imported.successful
    assert bool(imported.events.successful)
    assert jnp.array_equal(
        imported.events.pdg_ids[0, :4], jnp.asarray([11, -11, 13, -13])
    )
    assert jnp.isclose(imported.events.weights.nominal[0], 2.5)
    assert jnp.array_equal(imported.events.mother_indices[0, 2], jnp.asarray([0, 1]))

    exported = phx.interchange.hep.write_lhef(imported.events)
    reimported = phx.interchange.hep.read_lhef(exported.payload, _event_plan())
    assert reimported.successful
    assert jnp.allclose(reimported.events.momenta[0, :4], imported.events.momenta[0, :4])
    assert jnp.isclose(reimported.events.weights.nominal[0], 2.5)
    hepmc = phx.interchange.hep.write_hepmc3_ascii(imported.events)
    assert "\nU GEV MM\n" in hepmc.payload


def test_duplicate_event_identity_invalidates_every_duplicate():
    plan = _event_plan().prepare()
    weights = phx.particle_physics.EventWeightSet(
        jnp.ones((2, 1)),
        names=("nominal",),
        variation_kinds=(phx.particle_physics.WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    events = plan.admit(
        event_ids=jnp.asarray([7, 7]),
        subevent_ids=jnp.asarray([0, 0]),
        event_active=jnp.ones(2, dtype=bool),
        pdg_ids=jnp.zeros((2, 4), dtype=jnp.int32),
        roles=jnp.zeros((2, 4), dtype=jnp.int32),
        provider_status=jnp.zeros((2, 4), dtype=jnp.int32),
        momenta=jnp.zeros((2, 4, 4)),
        rest_energies=jnp.zeros((2, 4)),
        particle_active=jnp.zeros((2, 4), dtype=bool),
        mother_indices=jnp.full((2, 4, 2), -1),
        production_vertex_indices=jnp.full((2, 4), -1),
        end_vertex_indices=jnp.full((2, 4), -1),
        color_flow=jnp.zeros((2, 4, 2), dtype=jnp.int32),
        production_vertices=jnp.zeros((2, 1, 4)),
        vertex_active=jnp.zeros((2, 1), dtype=bool),
        weights=weights,
        source_id="duplicate-test",
    )
    assert not bool(events.successful)
    assert jnp.all(
        events.status == int(phx.particle_physics.ParticleEventStatus.DUPLICATE_EVENT_ID)
    )


def test_native_two_body_production_has_correct_constant_matrix_element_normalization():
    scattering = phx.applications.relativistic_scattering
    incoming = (
        scattering.Particle(
            "a",
            mass=0.0,
            charge=0.0,
            spin_twice=0,
            antiparticle="a",
            statistics="boson",
        ),
        scattering.Particle(
            "b",
            mass=0.0,
            charge=0.0,
            spin_twice=0,
            antiparticle="b",
            statistics="boson",
        ),
    )
    outgoing = (
        scattering.Particle(
            "c",
            mass=0.0,
            charge=0.0,
            spin_twice=0,
            antiparticle="c",
            statistics="boson",
        ),
        scattering.Particle(
            "d",
            mass=0.0,
            charge=0.0,
            spin_twice=0,
            antiparticle="d",
            statistics="boson",
        ),
    )
    prepared = scattering.HardProcessPlan(
        scattering.ScatteringProcess("constant", incoming, outgoing),
        scattering.BeamPlan((1, 2), 10.0),
        scattering.ScalePlan(10.0, 10.0, scheme_id="fixed"),
        outgoing_pdg_ids=(3, 4),
        matrix_element_id="constant-one",
        coupling_scheme_id="none",
    ).prepare(lambda incoming_momenta, outgoing_momenta: jnp.asarray(1.0))
    production = scattering.produce_hard_events(
        prepared,
        jnp.asarray([[0.25, 0.25], [0.75, 0.75]]),
        _event_plan(event_capacity=2),
    )
    expected = 1.0 / (1600.0 * jnp.pi)
    assert bool(production.successful)
    assert jnp.allclose(production.stream.weights, expected)
    assert jnp.isclose(production.ledger.cross_section, expected)
    assert jnp.all(production.derivative_valid)
