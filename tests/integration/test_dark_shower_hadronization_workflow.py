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
    DarkColorRule,
    DarkShowerEpochPlan,
    DarkSplittingChannel,
    DarkSplittingKernelKind,
    evolve_dark_shower_epoch,
)
from phydrax.particle_physics._events import ParticleEventPlan
from phydrax.particle_physics._hadronization import (
    DarkHadronPairChannel,
    DarkStringFragmentationPlan,
    fragment_dark_string_chain,
)
from phydrax.particle_physics._identity import ParticleCatalogueReference, ParticleRole
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.particle_physics._weights import EventWeightSet, WeightVariationKind
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from phydrax.units import COULOMB


def _workflow():
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(12),
        chart_id="flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros(4),
        convention=units.convention,
        source_id="observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="observer",
        orientation_id="future-right-handed",
    )
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=4,
        product_capacity=8,
        radiation_capacity=1,
        work_capacity=8,
        frontier_capacity=8,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id="9" * 64,
        topology_revision_id="a" * 64,
    )
    catalogue = ParticleCatalogueReference(
        source_id="workflow-species",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/workflow",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((100, -100, 101, 200, -200)),
        jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0)),
        jnp.asarray((1.0, -1.0, 0.0, 1.0, -1.0)),
        catalogue=catalogue,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    shower = DarkShowerEpochPlan(
        runtime,
        species,
        units,
        frame,
        (
            DarkSplittingChannel(
                100,
                (100, 101),
                kernel_kind=DarkSplittingKernelKind.FERMION_VECTOR,
                color_rule=DarkColorRule.FUNDAMENTAL_EMISSION,
                kernel_coefficient=1.0,
                envelope_coefficient=2.0,
            ),
        ),
        ordering="transverse-momentum",
        model_id="dark-u1",
        model_revision_id="dark-u1-r1",
        tune_id="workflow-tune",
        alpha_reference=0.2,
        reference_scale=10.0,
        beta0=1.0,
        infrared_cutoff=1.0,
        maximum_scale=10.0,
        z_bounds=(0.1, 0.9),
        proposal_capacity=1,
        production_evidence_ids=("workflow-shower-control",),
    )
    string = DarkStringFragmentationPlan(
        runtime,
        species,
        units,
        frame,
        (DarkHadronPairChannel((200, -200), 1.0, spectrum_label="dark-meson-pair"),),
        model_id="dark-string",
        model_revision_id="dark-string-r1",
        tune_id="workflow-string-tune",
        string_tension=10.0,
        longitudinal_shape=(0.3, 0.8),
        production_evidence_ids=("workflow-control",),
    )
    event_plan = ParticleEventPlan(
        catalogue=catalogue,
        momentum_unit=units.energy_unit,
        length_unit=units.scale.dimensional_scale.length_unit,
        time_unit=units.scale.dimensional_scale.time_unit,
        event_capacity=1,
        particle_capacity=7,
        vertex_capacity=3,
        provider_status_namespace="native-dark",
    ).prepare()
    weights = EventWeightSet(
        jnp.ones((1, 1)),
        names=("nominal",),
        variation_kinds=(WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    active = jnp.asarray(((True, True, False, False, False, False, False),))
    momenta = (
        jnp.zeros((1, 7, 4))
        .at[0, 0]
        .set((5.0, 0.0, 0.0, 5.0))
        .at[0, 1]
        .set((5.0, 0.0, 0.0, -5.0))
    )
    pdg = jnp.zeros((1, 7), dtype=jnp.int32).at[0, 0].set(100).at[0, 1].set(-100)
    roles = jnp.zeros((1, 7), dtype=jnp.int32).at[0, :2].set(int(ParticleRole.OUTGOING))
    color = (
        jnp.zeros((1, 7, 2), dtype=jnp.int32)
        .at[0, 0]
        .set(jnp.asarray((11, 0), dtype=jnp.int32))
        .at[0, 1]
        .set(jnp.asarray((0, 11), dtype=jnp.int32))
    )
    events = event_plan.admit(
        event_ids=jnp.asarray((1,)),
        subevent_ids=jnp.asarray((0,)),
        event_active=jnp.asarray((True,)),
        pdg_ids=pdg,
        roles=roles,
        provider_status=jnp.zeros((1, 7), dtype=jnp.int32),
        momenta=momenta,
        rest_energies=jnp.zeros((1, 7)),
        particle_active=active,
        mother_indices=jnp.full((1, 7, 2), -1),
        production_vertex_indices=jnp.full((1, 7), -1),
        end_vertex_indices=jnp.full((1, 7), -1),
        color_flow=color,
        production_vertices=jnp.zeros((1, 3, 4)),
        vertex_active=jnp.zeros((1, 3), dtype=bool),
        weights=weights,
        source_id="hard-dark-pair",
    )
    return shower, string, events


def test_dark_shower_color_chain_fragments_with_end_to_end_conservation_and_identity():
    shower, string, hard = _workflow()
    showered = evolve_dark_shower_epoch(
        shower, hard, jnp.asarray([[[0.0, 0.6, 0.4, 0.0]]])
    )
    assert bool(showered.accepted[0, 0])
    hadrons = fragment_dark_string_chain(
        string,
        showered.events.momenta[0],
        showered.events.pdg_ids[0],
        showered.events.color_flow[0],
        showered.frontier[0],
        jnp.asarray((0.2, 0.4, 0.8)),
        parent_entity_id="showered-color-singlet",
    )
    assert bool(hadrons.successful)
    initial_total = hard.momenta[0, 0] + hard.momenta[0, 1]
    np.testing.assert_allclose(hadrons.input_four_momentum, initial_total, atol=1e-6)
    np.testing.assert_allclose(
        hadrons.output_four_momenta.sum(axis=0), initial_total, atol=1e-5
    )
    np.testing.assert_allclose(hadrons.charge_residual, 0.0, atol=1e-12)
    assert showered.frame_realization_id == hadrons.frame_realization_id
    assert shower.runtime_plan.plan_id == string.runtime_plan.plan_id
