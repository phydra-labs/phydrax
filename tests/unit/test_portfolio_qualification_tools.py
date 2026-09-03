#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.closure_data._alignment import ConservativeAlignmentPlan
from phydrax.closure_data._analysis import (
    ClosureAnalysisDAG,
    ClosureAnalysisNode,
    ClosureQualityReport,
    ClosureTarget,
)
from phydrax.closure_data._binding import LearnedClosureBindingPlan
from phydrax.closure_data._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureDatasetChunk,
    ClosureSample,
    ClosureSampleKey,
    DatasetExtent,
    LeakageSafePartitionPlan,
    TrainOnlyNormalizer,
)
from phydrax.closure_data._filters import FilterSpec
from phydrax.discretization.lattice_boltzmann._commercial_qualification import (
    c0_guo_baseline_profiles,
    LatticeBoltzmannDeploymentRecord,
    LatticeBoltzmannQualificationClaim,
)
from phydrax.discretization.lattice_boltzmann._conjugate_thermal import (
    ConjugateThermalPlan,
)
from phydrax.discretization.lattice_boltzmann._operating_envelope import (
    LatticeBoltzmannHardwareTarget,
    LatticeBoltzmannOperatingPoint,
)
from phydrax.discretization.lattice_boltzmann._thermal import (
    ThermalLatticeBoltzmannPlan,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.statistical_dynamics._cumulants import (
    ForcingCovariance,
    SecondCumulantLayout,
)
from phydrax.statistical_dynamics._distributed import (
    DistributedBatchLayout,
    DistributedCovarianceLayout,
    DistributedRestartRelation,
    DistributedStatisticalLayout,
)
from phydrax.statistical_dynamics._interactions import (
    GeneralizedQuasilinearInteractions,
    NonlinearInteractions,
    QuasilinearInteractions,
)
from phydrax.statistical_dynamics._nilss import NILSSPlan
from phydrax.statistical_dynamics._plan import (
    QuadraticDynamics,
    StatisticalDynamicsPlan,
)
from tools.closure_data_qualification import (
    build_closure_data_candidate,
    closure_candidate_profile,
)
from tools.immersed_commercial_qualification import immersed_profile_record
from tools.lbm_commercial_qualification import (
    build_lbm_candidate,
    lbm_profile_records,
)
from tools.statistical_dynamics_qualification import (
    build_statistical_dynamics_candidate,
    statistical_candidate_profile,
    statistical_model_label,
    statistical_profile_records,
)


def _evidence(
    kind,
    subject,
    criteria=("criterion",),
    *,
    outcome="passed",
    precision="float64",
):
    return QualificationEvidence(
        kind,
        outcome,
        (subject,),
        build_id="build-a",
        environment_id="environment-a",
        backend="jax",
        topology="topology-a",
        precision=precision,
        reduction="deterministic-tree",
        replay_id="replay-a",
        criteria_ids=criteria,
        raw_artifact_ids=(f"raw-{kind}",),
        reviewer_id="reviewer-a",
        issued_at=9,
        expires_at=11,
        reason="the named bounded observation was reviewed",
    )


def _run(profile, support):
    dependency = SupportDependency(profile.profile_id, support.support_tuple_id)
    return ResolvedRunSpec(
        (dependency,),
        (),
        release_index_id="release-index-a",
        profile_ids=(profile.profile_id,),
        trust_policy_id="trust-policy-a",
        valid_at=10,
        valid_from=9,
        valid_until=11,
        prepared_configuration_id="prepared-configuration-a",
        precision_policy_id="precision-policy-a",
        resource_policy_id="resource-policy-a",
        checkpoint_policy_id="checkpoint-policy-a",
        output_policy_id="output-policy-a",
        repository_id="repository-a",
        scheduler_id="scheduler-a",
        auth_policy_id="auth-policy-a",
    )


def _lbm_point():
    return LatticeBoltzmannOperatingPoint(
        mach_number=0.05,
        knudsen_number=0.005,
        relaxation_rate=1.0,
        minimum_density=1.0,
        maximum_density=1.05,
        force_number=0.005,
        wall_resolution_cells=16.0,
        viscosity_ratio=1.05,
        relative_mass_drift=1.0e-10,
    )


def _lbm_evidence(profile, *, omit_operational=()):
    operational_claims = tuple(
        value.value
        for value in profile.required_claims
        if value
        in (
            LatticeBoltzmannQualificationClaim.FUSED_PARITY,
            LatticeBoltzmannQualificationClaim.AA_PARITY,
            LatticeBoltzmannQualificationClaim.SHARDED_PARITY,
            LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY,
            LatticeBoltzmannQualificationClaim.OUTPUT_PARITY,
        )
        and value.value not in omit_operational
    )
    scientific_claims = tuple(
        value.value
        for value in profile.required_claims
        if value.value
        not in {
            "fused-parity",
            "aa-parity",
            "sharded-parity",
            "checkpoint-parity",
            "output-parity",
        }
    )
    precision = profile.envelope.precision.policy_id
    return (
        _evidence(
            "scientific",
            profile.profile_id,
            scientific_claims,
            precision=precision,
        ),
        _evidence("performance", profile.profile_id, precision=precision),
        _evidence(
            "operational",
            profile.profile_id,
            operational_claims,
            precision=precision,
        ),
        _evidence("security", profile.profile_id, precision=precision),
    )


def _lbm_deployment(profile, evidence):
    return LatticeBoltzmannDeploymentRecord(
        "sharded",
        "array-archive",
        "kinetic-array-archive",
        host_count=profile.envelope.hardware.host_count,
        devices_per_host=profile.envelope.hardware.devices_per_host,
        execution_plan_id="execution-plan-a",
        output_plan_id="output-plan-a",
        checkpoint_plan_id="checkpoint-plan-a",
        execution_topology_id="topology-a",
        restart_topology_id="topology-b",
        topology_restart_relation_id="restart-relation-a",
        parity_evidence_ids=tuple(value.evidence_id for value in evidence),
    )


def test_immersed_and_lbm_profile_records_preserve_exact_route_distinctions():
    immersed = immersed_profile_record()
    body_methods = {
        value["regime"]: value["body_method"] for value in immersed["supports"]
    }
    assert body_methods["prescribed-marker"] == "marker-regularized"
    assert body_methods["fixed-topology-sharp"] == "sharp-interface"
    assert body_methods["lbm-body"] == "lattice-boltzmann-body"
    assert (
        len(
            {value["support_tuple"]["support_tuple_id"] for value in immersed["supports"]}
        )
        == 6
    )
    assert (
        immersed["signed"] is immersed["released"] is immersed["release_ready"] is False
    )

    thermal = ConjugateThermalPlan(
        ThermalLatticeBoltzmannPlan(2.0, 4.0, reference_temperature=300.0),
        3.0,
        6.0,
    )
    lbm = lbm_profile_records(conjugate_thermal=thermal)
    assert {value["tier"] for value in lbm} == {
        "C0",
        "C1",
        "C2",
        "C3",
        "conjugate-thermal",
    }
    assert all(value["method_class"] == "lattice-kinetic" for value in lbm)
    assert all(not value["signed"] and not value["released"] for value in lbm)
    assert lbm == lbm_profile_records(conjugate_thermal=thermal)


def test_lbm_candidate_retains_gap_report_and_resource_refusal():
    hardware = LatticeBoltzmannHardwareTarget(
        "cpu",
        "qualification-test",
        "portable-test",
        maximum_device_bytes=64,
    )
    profile = c0_guo_baseline_profiles(hardware=hardware)[0]
    complete = _lbm_evidence(profile)
    deployment = _lbm_deployment(profile, complete)
    run = _run(profile, profile.support_tuple)
    refused = build_lbm_candidate(
        profile,
        _lbm_point(),
        deployment,
        run,
        complete,
        at_time=10,
        resource_counts={"local_cell_count": 64},
    )
    assert refused["gates"]["performance"]["outcome"] == "failed"
    assert refused["resources"]["fits_budget"] is False
    assert refused["status"] == "failed"
    assert refused["release_ready"] is False
    assert refused == build_lbm_candidate(
        profile,
        _lbm_point(),
        deployment,
        run,
        complete,
        at_time=10,
        resource_counts={"local_cell_count": 64},
    )

    missing = _lbm_evidence(profile, omit_operational=("output-parity",))
    gap_candidate = build_lbm_candidate(
        profile,
        _lbm_point(),
        _lbm_deployment(profile, missing),
        run,
        missing,
        at_time=10,
        resource_counts={"local_cell_count": 1},
    )
    coverage = gap_candidate["commercial_evidence"]["coverage"]
    assert "output-parity" in coverage["inconclusive_predicate_ids"]
    assert gap_candidate["gates"]["operational"]["outcome"] == "inconclusive"


def _closure_pipeline(*, partition_salt="partition-a"):
    prepared_filter = FilterSpec.identity().prepare((2, 2))
    alignment = ConservativeAlignmentPlan().prepare((2, 2), (2, 2))
    node = ClosureAnalysisNode(
        "source_residual",
        ("fine-source", "resolved-source"),
        output_name="source",
        output_units="1/s",
    )
    dag = ClosureAnalysisDAG(("fine-source", "resolved-source"), (node,))
    target = ClosureTarget(
        jnp.ones((2, 2)), node, target_kind="source", schema_id="schema-a"
    )
    quality = ClosureQualityReport((target,), maximum_allowed=2.0)
    extent = DatasetExtent(
        case_id="case-a",
        trajectory_id="trajectory-a",
        realization_id="realization-a",
        time_block_id="block-a",
        sample_count=1,
    )
    payload = b"closure-data"
    chunk = ClosureDatasetChunk.from_payload(
        payload,
        extent_id=extent.extent_id,
        logical_name="samples",
        chunk_index=0,
        sample_start=0,
        sample_stop=1,
        byte_offset=0,
    )
    dataset = ChunkedClosureDatasetManifest(
        dataset_id="dataset-a",
        schema_id="schema-a",
        analysis_dag_id=dag.dag_id,
        extents=(extent,),
        chunks=(chunk,),
    )
    sample = ClosureSample(
        jnp.asarray((1.0, 2.0)),
        ClosureSampleKey(
            case_id="case-a",
            trajectory_id="trajectory-a",
            realization_id="realization-a",
            time_block_id="block-a",
            time_index=0,
        ),
        schema_id="schema-a",
    )
    partition = LeakageSafePartitionPlan(
        "trajectory",
        train_fraction=1.0,
        validation_fraction=0.0,
        test_fraction=0.0,
        salt=partition_salt,
    ).assign((sample,))
    normalizer = TrainOnlyNormalizer.fit((sample,), partition, feature_name="state")
    binding = LearnedClosureBindingPlan(
        lambda values, _args=None: values,
        deployment_kind="conservative_face",
        schema_id="schema-a",
        input_component_names=("rho",),
        output_component_names=("rho",),
        model_artifact_id="model-artifact-a",
        normalizer_provenance_id=normalizer.provenance.provenance_id,
    )
    return (
        prepared_filter,
        alignment,
        dag,
        dataset,
        partition,
        normalizer,
        binding,
        quality,
    )


def _denied_reference():
    return ReferenceArtifactManifest(
        "restricted-reference",
        checksum_algorithm="sha256",
        checksum="a" * 64,
        size_bytes=128,
        license_id="restricted",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="restricted",
        nondimensionalization={"length": 1.0},
        uncertainty={"state": 0.01},
        lineage_ids=("source-a",),
    )


def test_closure_candidates_isolate_offline_deployment_and_fail_leakage_rights():
    pipeline = _closure_pipeline()
    prepared_filter, alignment, dag, dataset, partition, normalizer, binding, quality = (
        pipeline
    )
    offline = closure_candidate_profile(
        prepared_filter, alignment, dag, dataset, partition
    )
    deployed = closure_candidate_profile(
        prepared_filter,
        alignment,
        dag,
        dataset,
        partition,
        normalizer=normalizer,
        binding=binding,
    )
    assert dict(offline.support_tuples[0].attributes)["route"] == "offline-closure-data"
    assert (
        dict(deployed.support_tuples[0].attributes)["route"]
        == "deployed-conservative_face"
    )
    assert (
        offline.support_tuples[0].support_tuple_id
        != deployed.support_tuples[0].support_tuple_id
    )

    mismatched_partition = _closure_pipeline(partition_salt="partition-b")[4]
    mismatched_profile = closure_candidate_profile(
        prepared_filter,
        alignment,
        dag,
        dataset,
        mismatched_partition,
        normalizer=normalizer,
        binding=binding,
    )
    subject = binding.binding_id
    evidence = tuple(
        _evidence(kind, subject)
        for kind in ("scientific", "performance", "operational", "security")
    )
    candidate = build_closure_data_candidate(
        mismatched_profile,
        prepared_filter,
        alignment,
        dag,
        dataset,
        mismatched_partition,
        quality,
        _run(mismatched_profile, mismatched_profile.support_tuples[0]),
        evidence,
        at_time=10,
        normalizer=normalizer,
        binding=binding,
        reference_manifests=(_denied_reference(),),
    )
    assert candidate["gates"]["security"]["outcome"] == "failed"
    reasons = candidate["gates"]["security"]["failed_reasons"]
    assert "normalizer-partition-mismatch" in reasons
    assert any("commercial-use-not-permitted" in value for value in reasons)
    assert any("training-use-not-permitted" in value for value in reasons)
    assert candidate["route"] == "deployed-conservative_face"
    assert candidate["release_ready"] is False


def _cumulant_plan(closure, interaction_model):
    layout = SecondCumulantLayout(2, (0,), eddy_indices=(1,))
    dynamics = QuadraticDynamics(jnp.zeros((2,)), jnp.zeros((2, 2)), jnp.zeros((2, 2, 2)))
    forcing = ForcingCovariance(jnp.eye(1))
    return StatisticalDynamicsPlan(
        layout,
        dynamics,
        forcing,
        closure=closure,
        interaction_model=interaction_model,
        time_step=0.1,
    )


def _statistical_layout(process_count):
    return DistributedStatisticalLayout(
        DistributedBatchLayout(
            4,
            process_count,
            item_bytes=8,
            maximum_local_bytes=128,
        ),
        DistributedCovarianceLayout(
            2,
            process_count,
            maximum_local_bytes=128,
        ),
    )


def test_statistical_labels_resources_and_topology_restart_are_isolated():
    nl = NonlinearInteractions()
    ql = QuasilinearInteractions()
    gql = GeneralizedQuasilinearInteractions()
    ce2 = _cumulant_plan("ce2", "ql")
    gce2 = _cumulant_plan("gce2", "gql")
    assert tuple(
        statistical_model_label(value) for value in (nl, ql, gql, ce2, gce2)
    ) == (
        "nl-dns",
        "ql-ensemble",
        "gql-ensemble",
        "ce2",
        "gce2",
    )
    records = statistical_profile_records((nl, ql, gql, ce2, gce2))
    assert len(records) == 5
    assert len({value["support_tuples"][0]["support_tuple_id"] for value in records}) == 5
    with pytest.raises(ValueError, match="cannot describe"):
        statistical_candidate_profile(nl, model_label="ce2")
    with pytest.raises(ValueError, match="only for the NL DNS route"):
        statistical_candidate_profile(ql, nilss=NILSSPlan(2, 1, 2, 2))

    source = _statistical_layout(1)
    target = _statistical_layout(2)
    restart = DistributedRestartRelation(source, target)
    nilss = NILSSPlan(2, 1, 2, 2)
    profile = statistical_candidate_profile(nl, distributed_layout=target, nilss=nilss)
    evidence = tuple(
        _evidence(kind, nl.model_id)
        for kind in ("scientific", "performance", "operational", "security")
    )
    candidate = build_statistical_dynamics_candidate(
        profile,
        nl,
        _run(profile, profile.support_tuples[0]),
        evidence,
        at_time=10,
        resource_measurements={
            "retained_bytes": 1024,
            "nilss_workspace_bytes": 2048,
        },
        distributed_layout=target,
        restart_relation=restart,
        nilss=nilss,
    )
    assert candidate["model_label"] == "nl-dns"
    assert candidate["restart"]["accepted"] is True
    assert candidate["restart"]["topology_changed"] is True
    assert candidate["gates"]["operational"]["outcome"] == "passed"
    assert candidate["status"] == "passed"
    assert candidate == build_statistical_dynamics_candidate(
        profile,
        nl,
        _run(profile, profile.support_tuples[0]),
        evidence,
        at_time=10,
        resource_measurements={
            "retained_bytes": 1024,
            "nilss_workspace_bytes": 2048,
        },
        distributed_layout=target,
        restart_relation=restart,
        nilss=nilss,
    )

    constrained = _cumulant_plan("ce2", "ql")
    constrained_profile = statistical_candidate_profile(constrained)
    constrained_evidence = tuple(
        _evidence(kind, constrained.plan_id)
        for kind in ("scientific", "performance", "operational", "security")
    )
    refused = build_statistical_dynamics_candidate(
        constrained_profile,
        constrained,
        _run(constrained_profile, constrained_profile.support_tuples[0]),
        constrained_evidence,
        at_time=10,
        resource_measurements={
            "state_bytes": constrained.maximum_state_bytes + 1,
            "workspace_bytes": 1,
        },
    )
    assert refused["gates"]["performance"]["outcome"] == "failed"
    assert refused["status"] == "failed"
