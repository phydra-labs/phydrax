#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.particle_physics._dark_shower import (
    certified_splitting_envelope,
    dark_splitting_kernel,
    DarkColorRule,
    DarkShowerEpochPlan,
    DarkShowerOrdering,
    DarkSplittingChannel,
    DarkSplittingKernelKind,
    evolve_dark_shower_epoch,
    running_dark_coupling,
    stage_dark_shower_continuation,
    sudakov_no_emission_probability,
)
from phydrax.particle_physics._events import ParticleEventPlan
from phydrax.particle_physics._identity import ParticleCatalogueReference, ParticleRole
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.particle_physics._weights import EventWeightSet, WeightVariationKind
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)
from phydrax.units import COULOMB


def _units_and_frame():
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(1),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        source_id="test-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="test-observer",
        orientation_id="future-right-handed",
    )
    return units, frame


def _runtime(frontier_capacity=8):
    return DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=4,
        product_capacity=8,
        radiation_capacity=1,
        work_capacity=frontier_capacity,
        frontier_capacity=frontier_capacity,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
        precision_id="float64",
    )


def _shower_plan():
    units, frame = _units_and_frame()
    catalogue = ParticleCatalogueReference(
        source_id="dark-test",
        provider_release="test",
        checksum="species-checksum",
        citation_url="https://example.test/dark-species",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((100, 101)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
        catalogue=catalogue,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    channel = DarkSplittingChannel(
        100,
        (100, 101),
        kernel_kind=DarkSplittingKernelKind.FERMION_VECTOR,
        color_rule=DarkColorRule.FUNDAMENTAL_EMISSION,
        kernel_coefficient=1.0,
        envelope_coefficient=2.0,
    )
    return DarkShowerEpochPlan(
        _runtime(),
        species,
        units,
        frame,
        (channel,),
        ordering=DarkShowerOrdering.TRANSVERSE_MOMENTUM,
        model_id="declared-dark-u1",
        model_revision_id="dark-u1-r1",
        tune_id="unit-control",
        alpha_reference=0.2,
        reference_scale=10.0,
        beta0=1.0,
        infrared_cutoff=1.0,
        maximum_scale=10.0,
        z_bounds=(0.1, 0.9),
        proposal_capacity=1,
        production_evidence_ids=("dark-shower-control",),
    )


def _event(plan, particle_capacity=5):
    event_plan = ParticleEventPlan(
        catalogue=plan.species.catalogue,
        momentum_unit=plan.units.energy_unit,
        length_unit=plan.units.scale.dimensional_scale.length_unit,
        time_unit=plan.units.scale.dimensional_scale.time_unit,
        event_capacity=1,
        particle_capacity=particle_capacity,
        vertex_capacity=2,
        provider_status_namespace="native-dark-shower",
    ).prepare()
    active = jnp.zeros((1, particle_capacity), dtype=bool).at[0, 0].set(True)
    momenta = (
        jnp.zeros((1, particle_capacity, 4))
        .at[0, 0]
        .set(jnp.asarray((10.0, 0.0, 0.0, 10.0)))
    )
    pdg_ids = jnp.zeros((1, particle_capacity), dtype=jnp.int32).at[0, 0].set(100)
    roles = (
        jnp.zeros((1, particle_capacity), dtype=jnp.int32)
        .at[0, 0]
        .set(int(ParticleRole.OUTGOING))
    )
    color = (
        jnp.zeros((1, particle_capacity, 2), dtype=jnp.int32)
        .at[0, 0]
        .set(jnp.asarray((11, 0), dtype=jnp.int32))
    )
    weights = EventWeightSet(
        jnp.ones((1, 1)),
        names=("nominal",),
        variation_kinds=(WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    return event_plan.admit(
        event_ids=jnp.asarray((7,)),
        subevent_ids=jnp.asarray((0,)),
        event_active=jnp.asarray((True,)),
        pdg_ids=pdg_ids,
        roles=roles,
        provider_status=jnp.zeros((1, particle_capacity), dtype=jnp.int32),
        momenta=momenta,
        rest_energies=jnp.zeros((1, particle_capacity)),
        particle_active=active,
        mother_indices=jnp.full((1, particle_capacity, 2), -1),
        production_vertex_indices=jnp.full((1, particle_capacity), -1),
        end_vertex_indices=jnp.full((1, particle_capacity), -1),
        color_flow=color,
        production_vertices=jnp.zeros((1, 2, 4)),
        vertex_active=jnp.zeros((1, 2), dtype=bool),
        weights=weights,
        source_id="hard-event",
    )


def test_sudakov_running_coupling_and_veto_bound_are_certified():
    plan = _shower_plan()
    channel = plan.channels[0]
    z = jnp.linspace(plan.z_bounds[0], plan.z_bounds[1], 101)
    assert jnp.all(
        dark_splitting_kernel(channel, z) <= certified_splitting_envelope(channel, z)
    )
    assert running_dark_coupling(plan, 1.0) > running_dark_coupling(plan, 10.0)
    no_emission = sudakov_no_emission_probability(plan, channel, 10.0, 2.0)
    shorter_interval = sudakov_no_emission_probability(plan, channel, 10.0, 5.0)
    assert 0.0 < no_emission < shorter_interval < 1.0


def test_splitting_preserves_ordering_recoil_charge_color_and_four_momentum():
    plan = _shower_plan()
    original = _event(plan)
    result = evolve_dark_shower_epoch(
        plan, original, jnp.asarray([[[0.0, 0.7, 0.4, 0.0]]])
    )
    assert bool(result.accepted[0, 0])
    assert float(result.proposal_scales[0, 0]) < plan.maximum_scale
    assert float(result.proposal_scales[0, 0]) > plan.infrared_cutoff
    child_slots = np.flatnonzero(np.asarray(result.frontier[0]))
    assert child_slots.tolist() == [1, 2]
    np.testing.assert_allclose(
        np.asarray(result.events.momenta[0, child_slots]).sum(axis=0),
        np.asarray(original.momenta[0, 0]),
        atol=1e-7,
    )
    assert int(result.events.pdg_ids[0, 1]) == 100
    assert int(result.events.pdg_ids[0, 2]) == 101
    assert int(result.events.mother_indices[0, 1, 0]) == 0
    assert int(result.events.mother_indices[0, 2, 0]) == 0
    assert int(result.events.color_flow[0, 1, 0]) == int(
        result.events.color_flow[0, 2, 1]
    )
    daughter_charge = plan.species.charges[0] + plan.species.charges[1]
    assert jnp.isclose(daughter_charge, plan.species.charges[0])
    durable = stage_dark_shower_continuation(
        plan,
        result,
        empty_dark_sector_epoch_state(plan.runtime_plan, epoch_sequence=0),
    )
    assert not bool(durable.rolled_back)
    assert int(durable.state.work_mask.sum()) == 2


def test_split_capacity_is_atomic_and_reports_backpressure():
    plan = _shower_plan()
    original = _event(plan, particle_capacity=2)
    result = evolve_dark_shower_epoch(
        plan, original, jnp.asarray([[[0.0, 0.7, 0.4, 0.0]]])
    )
    assert bool(result.backpressured[0])
    assert not bool(result.accepted[0, 0])
    np.testing.assert_array_equal(result.events.particle_active, original.particle_active)
    np.testing.assert_allclose(result.events.momenta, original.momenta)
