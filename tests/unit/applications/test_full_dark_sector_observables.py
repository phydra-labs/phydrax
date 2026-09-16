#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax.applications.cosmology._full_dark_sector_observables import (
    EventShowerHadronizationObservables,
    FullDarkSectorLedgerObservables,
    FullDarkSectorObservableBundle,
    FullDarkSectorObservationPlan,
    MetricStressObservables,
    QuantumCoherenceObservables,
    RadiationObservables,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from phydrax.observation import LinearObservationPlan


def _geometry_and_stress() -> tuple[ADMGridGeometry, StressEnergyProjection]:
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(7, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id="adm",
        scale_id="relativistic-units",
        topology_id="mesh",
        geometry_lineage_id="geometry",
    )
    stress = StressEnergyProjection(
        jnp.asarray(2.0),
        jnp.asarray((0.1, 0.2, 0.3)),
        jnp.eye(3),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id="total-stress",
    )
    return geometry, stress


def _bundle() -> FullDarkSectorObservableBundle:
    geometry, stress = _geometry_and_stress()
    metric = MetricStressObservables(
        geometry,
        stress,
        jnp.asarray((1.0e-10, 2.0e-10)),
        jnp.asarray((3.0e-10,)),
        frame_id="frame",
        source_ids=("matter", "radiation"),
        frame_realization_id="frame-realization",
    )
    events = EventShowerHadronizationObservables(
        jnp.asarray((2.0, 3.0)),
        jnp.asarray((0.8, 1.2)),
        jnp.asarray((4.0, 1.0)),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((5.0, 2.0)),
        jnp.asarray((11, 12)),
        jnp.asarray((1.0,)),
        jnp.asarray((21,)),
        jnp.asarray(1.0e-12),
        jnp.asarray((True, True)),
        event_manifest_id="epoch-manifest",
        matrix_element_revision_id="matrix-revision",
        provider_chain_id="provider-chain",
        shower_profile_id="shower-profile",
        hadronization_profile_id="hadron-profile",
        source_ids=("event-owner",),
        successful=True,
    )
    quantum = QuantumCoherenceObservables(
        jnp.asarray((0.4, 0.2)),
        jnp.asarray((0.1, 0.0)),
        jnp.asarray((0.0, -0.1)),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((0.3, 0.7)),
        jnp.asarray((0.2, 0.1)),
        jnp.asarray(0.6),
        jnp.asarray(0.2),
        jnp.asarray(1.0e-12),
        jnp.asarray(2.0e-12),
        quantum_profile_id="quantum-profile",
        coherent_profile_id="coherent-profile",
        off_shell_profile_id="off-shell-profile",
        frame_id="frame",
        frame_token=jnp.asarray(7, dtype=jnp.int32),
        frame_realization_id="frame-realization",
        unit_contract_id="relativistic-units",
        source_ids=("quantum-owner", "coherent-owner"),
        successful=True,
    )
    radiation = RadiationObservables(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((3.0, 4.0)),
        jnp.asarray(((3.0, 0.1, 0.2, 0.3), (4.0, 0.2, 0.1, 0.0))),
        jnp.asarray(0.15),
        jnp.asarray((0.2, 0.3)),
        jnp.asarray(((0.1, 0.0, 0.0), (0.0, 0.1, 0.0))),
        jnp.zeros((2, 4)),
        radiation_profile_id="radiation-profile",
        packet_profile_id="packet-profile",
        frame_id="frame",
        frame_token=jnp.asarray(7, dtype=jnp.int32),
        frame_realization_id="frame-realization",
        unit_contract_id="relativistic-units",
        source_ids=("radiation-owner",),
        successful=True,
    )
    ledgers = FullDarkSectorLedgerObservables(
        jnp.asarray((1.0e-12, 2.0e-12)),
        jnp.asarray((2.0e-12, 3.0e-12)),
        jnp.asarray((1.0e-12, 1.0e-12)),
        jnp.asarray((0.2, 0.3)),
        jnp.asarray((1.0e-13, 2.0e-13)),
        jnp.asarray(3.0e-12),
        jnp.asarray(3.0e-12),
        jnp.asarray(1.0e-12),
        jnp.asarray(0.5),
        jnp.asarray(2.0e-13),
        jnp.asarray((True, True)),
        component_names=("matter", "radiation"),
        source_evidence_ids=("evidence-matter", "evidence-radiation"),
        successful=True,
    )
    return FullDarkSectorObservableBundle(
        metric,
        events,
        quantum,
        radiation,
        ledgers,
        stage_id="stage",
        epoch_manifest_id="epoch-manifest",
    )


def test_full_dark_sector_observation_delegates_identity_maps() -> None:
    bundle = _bundle()
    sources = bundle.theory_vectors()
    owners = tuple(
        LinearObservationPlan(
            jnp.eye(source.layout.size),
            source.layout,
            source.layout,
        )
        for source in sources
    )
    plan = FullDarkSectorObservationPlan(*owners)

    observed = plan.apply(bundle)
    outputs = (
        observed.metric_stress,
        observed.event_shower_hadronization,
        observed.quantum_coherence,
        observed.radiation,
        observed.ledgers,
    )

    assert bool(bundle.successful)
    assert observed.source_output_id == bundle.output_id
    for source, output in zip(sources, outputs):
        assert output.layout.layout_id == source.layout.layout_id
        assert jnp.array_equal(output.values, source.values)


def test_full_dark_sector_observable_identity_binds_stage_and_sources() -> None:
    first = _bundle()
    second = FullDarkSectorObservableBundle(
        first.metric_stress,
        first.event_shower_hadronization,
        first.quantum_coherence,
        first.radiation,
        first.ledgers,
        stage_id="next-stage",
        epoch_manifest_id=first.epoch_manifest_id,
    )

    assert first.output_id != second.output_id
    assert first.radiation.product_id != first.quantum_coherence.product_id
    assert first.event_shower_hadronization.event_manifest_id == first.epoch_manifest_id
