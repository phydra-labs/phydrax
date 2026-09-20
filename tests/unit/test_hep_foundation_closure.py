#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _interval():
    start = phx.measurement.OperationalCoordinate("collider", {"run": 7, "lb": 0})
    end = phx.measurement.OperationalCoordinate("collider", {"run": 7, "lb": 10})
    return phx.measurement.OperationalInterval(start, end)


def _run_context():
    interval = _interval()
    coordinate = phx.measurement.OperationalCoordinate("collider", {"run": 7, "lb": 3})
    payload = phx.measurement.ConditionPayloadReference(
        semantic_name="alignment",
        authority="external-test",
        external_tag="tag-a",
        checksum="payload-checksum",
        source_uri="memory://alignment",
        interval=interval,
    )
    conditions = phx.measurement.ResolvedConditionSnapshot(
        coordinate,
        (payload,),
        resolution_time=10,
        resolver_id="test-resolver",
        required_names=("alignment",),
    )
    quality = phx.measurement.DataQualityAnnotation(
        interval,
        certified=True,
        authority="external-test",
        evidence_ids=("dq-evidence",),
    )
    exposure = phx.measurement.ExposureRecord(
        phx.measurement.ExposureKind.CERTIFIED_LUMINOSITY,
        100.0,
        1.0,
        "fb^-1",
        interval,
        authority="external-test",
        correlation_id="lumi-2026",
    )
    beam = phx.particle_physics.BeamConditionSnapshot(
        species_pdg_ids=(2212, 2212),
        beam_energies=(6800.0, 6800.0),
        bunch_intensities=(1.0e11, 1.0e11),
        crossing_angle=0.0,
        beam_spot_mean=jnp.zeros(4),
        beam_spot_covariance=jnp.eye(4),
        energy_unit_id="GeV",
        length_unit_id="mm",
        source_id="beam-test",
    )
    return phx.particle_physics.HEPRunContext(
        coordinate,
        beam,
        conditions,
        quality,
        (exposure,),
        campaign_id="campaign-test",
        stream_id="physics",
    )


def _event_plan():
    catalog = phx.particle_physics.ParticleCatalogReference(
        source_id="pdg-test",
        provider_release="test",
        checksum="catalog-checksum",
        citation_url="https://pdg.lbl.gov/",
    )
    return phx.particle_physics.ParticleEventPlan(
        catalog=catalog,
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        event_capacity=1,
        particle_capacity=3,
        vertex_capacity=1,
        provider_status_namespace="host-test",
    )


def test_host_event_pack_unpack_preserves_graph_and_reports_attribute_loss():
    pp = phx.particle_physics
    event = pp.HostEventRecord(
        9,
        0,
        (
            pp.HostParticleRecord(
                0,
                11,
                pp.ParticleRole.INCOMING,
                -1,
                (5.0, 0.0, 0.0, 5.0),
                0.0,
                end_vertex_id=0,
            ),
            pp.HostParticleRecord(
                1,
                -11,
                pp.ParticleRole.INCOMING,
                -1,
                (5.0, 0.0, 0.0, -5.0),
                0.0,
                end_vertex_id=0,
            ),
            pp.HostParticleRecord(
                2,
                22,
                pp.ParticleRole.OUTGOING,
                1,
                (10.0, 0.0, 0.0, 0.0),
                10.0,
                production_vertex_id=0,
            ),
        ),
        (pp.HostVertexRecord(0, (0.0, 0.0, 0.0, 0.0), (0, 1), (2,)),),
        (pp.HostEventWeight("nominal", 1.0, pp.WeightVariationKind.NOMINAL, "nominal"),),
        "host-test",
        "host-source",
        (pp.HostAttribute("provider", "raw", "application/octet-stream", b"abc"),),
    )
    packed = pp.pack_host_events((event,), _event_plan(), source_id="packed-source")
    assert packed.successful
    assert not packed.report.lossless
    assert packed.report.attribute_loss_count == 1
    assert int(packed.events.production_vertex_indices[0, 2]) == 0
    assert int(packed.events.end_vertex_indices[0, 0]) == 0
    unpacked = pp.unpack_particle_events(packed.events)
    assert unpacked[0].vertices[0].incoming_particle_ids == (0, 1)
    assert unpacked[0].vertices[0].outgoing_particle_ids == (2,)


def test_conditions_normalization_and_systematics_remain_distinct():
    context = _run_context()
    assert context.scientifically_admitted
    first = phx.particle_physics.ProcessNormalization(
        process_id="z-production",
        attempted_count=100,
        generated_count=90,
        accepted_count=80,
        positive_count=85,
        negative_count=5,
        sum_weights=75.0,
        sum_absolute_weights=85.0,
        sum_squared_weights=95.0,
        cross_section=2.0,
        cross_section_uncertainty=0.2,
        cross_section_unit_id="pb",
        provider_id="generator-test",
    )
    merged = phx.particle_physics.merge_process_normalizations((first, first))
    assert merged.attempted_count == 200
    assert merged.sum_weights == 150.0
    assert jnp.isclose(merged.cross_section_uncertainty, 0.2 / jnp.sqrt(2.0))
    systematic = phx.particle_physics.SystematicSource(
        "jet-energy-scale",
        phx.particle_physics.SystematicKind.CALIBRATION,
        ("down", "up"),
        affected_collections=("jets",),
        correlation_scopes=("run-2026",),
        provider_id="calibration-test",
    )
    assert (
        phx.particle_physics.SystematicConfiguration((systematic,))
        .by_name("jet-energy-scale")
        .kind
        is phx.particle_physics.SystematicKind.CALIBRATION
    )


def test_shared_binned_statistics_fit_and_governed_profile():
    stats = phx.particle_physics.statistics
    parameter = stats.StatisticalParameter(
        "signal-strength",
        initial=0.0,
        lower=-2.0,
        upper=2.0,
    )
    model = stats.BinnedStatisticalModel(
        jnp.asarray([[[10.0, 20.0]]]),
        jnp.asarray([[12.0, 24.0]]),
        jnp.asarray([[[[0.2, 0.2]]]]),
        parameters=(parameter,),
        modifier_modes=(stats.BinnedModifierMode.EXPONENTIAL,),
        channel_names=("signal",),
        sample_names=("prediction",),
    )
    nominal = stats.evaluate_binned_model(model, jnp.asarray([0.0]))
    assert bool(nominal.valid)
    fit = stats.fit_binned_model(model, maximum_steps=128)
    assert bool(fit.valid)
    assert fit.parameters[0] > 0.0

    criterion = phx.qualification.ScientificMetricCriterion(
        "mean-relative-error",
        "at_most",
        None,
        0.1,
        "1",
        "pooled",
    )
    evidence = phx.qualification.ReleaseGateEvidence(
        "contract",
        passed=True,
        evidence_ids=("contract-evidence",),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=100,
    )
    bundle = phx.particle_physics.build_hep_qualification_bundle(
        capability_name="hep.statistical-model",
        support_attributes={"model": "binned-poisson", "bins": 2},
        profile_name="hep.statistical-model.reference",
        provider="phydrax",
        provider_version="current",
        release_evidence=(evidence,),
        released=True,
        campaign_id="campaign",
        observable_ids=("fit",),
        condition_domain_ids=("synthetic",),
        required_stage_ids=("numerical-validity",),
        criteria=(criterion,),
        frozen_criteria_ids=(criterion.criterion_id,),
        abstention_policy_id="invalid-model",
        invalidation_triggers=("provider-change",),
        required_release_gates=("contract",),
    )
    assert bundle.capability.released
    assert bundle.support.capability == "hep.statistical-model"


def test_distributed_preservation_bundle_is_content_addressed_and_read_only():
    hep_io = phx.interchange.hep
    replica = hep_io.HEPFileReplica(
        "file.root",
        "adler32:1234",
        byte_size=1024,
        event_count=10,
        replica_uris=("root://example/file.root",),
    )
    dataset = hep_io.HEPDatasetSnapshot(
        "dataset",
        "scope:dataset",
        (replica,),
        closed=True,
        expected_file_count=1,
    )
    environment = hep_io.HEPSoftwareEnvironment(
        source_revision="abc",
        container_digest="sha256:container",
        provider_binding_ids=("provider-binding",),
        sbom_checksum="sha256:sbom",
    )
    workload = hep_io.HEPWorkloadExport(
        hep_io.WorkloadBackend.REANA,
        workset_plan_id="workset",
        dataset_snapshot=dataset,
        environment=environment,
        output_repository_id="artifact-repository",
    )
    run = _run_context()
    bundle = hep_io.HEPPreservationBundle(
        run,
        run.conditions,
        dataset,
        environment,
        workload,
        statistical_model_ids=("model",),
        output_artifact_ids=("output",),
        qualification_evidence_ids=("evidence",),
        rights_manifest_ids=("rights",),
    )
    assert dataset.event_count == 10
    assert workload.read_only_credentials
    assert not bundle.externally_approved
    assert bundle.bundle_id
