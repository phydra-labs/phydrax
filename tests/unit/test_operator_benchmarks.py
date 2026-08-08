#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from tools.operator_benchmarks import (
    assert_benchmark_thresholds,
    audit_external_candidate,
    compatible_architectures,
    ExternalOperatorCandidate,
    OperatorBenchmarkAggregate,
    OperatorBenchmarkRegressionError,
    OperatorBenchmarkThreshold,
    periodic_burgers_scenario,
    run_benchmark_matrix,
    run_operator_benchmark,
    save_benchmark_artifacts,
    scenario_checksum,
    select_benchmark_superior_external,
    split_operator_scenario,
    standard_operator_benchmarks,
    verify_external_candidate_artifact,
)


def test_standard_operator_benchmarks_cover_required_regimes():
    scenarios = standard_operator_benchmarks(quick=True)
    names = {scenario.name for scenario in scenarios}
    assert names == {
        "analytic_green_query",
        "darcy_2d",
        "euler_bernoulli_transient",
        "graph_diffusion_resolution_transfer",
        "irregular_poisson_2d",
        "multi_input_diffusion",
        "navier_stokes_vorticity_2d",
        "periodic_burgers_1d",
    }
    assert all(jnp.all(jnp.isfinite(scenario.train_target)) for scenario in scenarios)


def test_benchmark_runner_trains_and_reports_cross_resolution_metrics():
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=2,
    )
    model = phx.nn.operator.architectures.FNO(
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(0),
    )
    _, result = run_operator_benchmark(
        model,
        scenario,
        steps=1,
        repeats=1,
    )
    assert result.parameter_count > 0
    assert result.training_steps == 1
    assert jnp.isfinite(result.initial_loss)
    assert jnp.isfinite(result.final_loss)
    assert result.final_loss <= result.initial_loss
    assert {evaluation.name for evaluation in result.evaluations} == {
        "train_resolution",
        "higher_resolution",
    }
    assert all(jnp.isfinite(evaluation.relative_l2) for evaluation in result.evaluations)


def test_benchmark_runner_records_validation_plateau_early_stopping():
    scenario = split_operator_scenario(
        periodic_burgers_scenario(
            train_resolution=8,
            test_resolution=12,
            num_cases=4,
        ),
        seed=4,
    )
    model = phx.nn.operator.architectures.FNO(
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(0),
    )
    _, result = run_operator_benchmark(
        model,
        scenario,
        steps=20,
        learning_rate=0.0,
        repeats=1,
        validation_interval=1,
        patience=2,
        relative_minimum_delta=1e-3,
    )
    assert result.training_steps == 2
    assert result.validation_loss is not None
    assert result.stopped_early
    assert result.converged
    assert result.resumed_from_step == 0


def test_benchmark_runner_resumes_model_optimizer_and_curve_exactly(tmp_path):
    scenario = split_operator_scenario(
        periodic_burgers_scenario(
            train_resolution=8,
            test_resolution=12,
            num_cases=6,
        ),
        seed=17,
    )

    def construct():
        return phx.nn.operator.architectures.FNO(
            width=4,
            depth=1,
            n_modes=(3,),
            key=jr.key(23),
        )

    checkpoint = tmp_path / "trial"
    _, partial = run_operator_benchmark(
        construct(),
        scenario,
        steps=2,
        learning_rate=1e-3,
        repeats=1,
        validation_interval=1,
        checkpoint_path=checkpoint,
        checkpoint_metadata={"commit_identity": "pinned"},
        checkpoint_key=jr.key(29),
        run_evaluations=False,
    )
    resumed_model, resumed = run_operator_benchmark(
        construct(),
        scenario,
        steps=4,
        learning_rate=1e-3,
        repeats=1,
        validation_interval=1,
        checkpoint_path=checkpoint,
        resume=True,
        checkpoint_metadata={"commit_identity": "pinned"},
        checkpoint_key=jr.key(29),
        run_evaluations=False,
    )
    fresh_model, fresh = run_operator_benchmark(
        construct(),
        scenario,
        steps=4,
        learning_rate=1e-3,
        repeats=1,
        validation_interval=1,
        run_evaluations=False,
    )
    assert partial.training_steps == 2
    assert resumed.resumed_from_step == 2
    assert resumed.training_steps == 4
    assert resumed.losses == fresh.losses
    assert resumed.validation_steps == fresh.validation_steps
    assert resumed.validation_losses == fresh.validation_losses
    assert jnp.array_equal(
        resumed_model(scenario.train_batch),
        fresh_model(scenario.train_batch),
    )

    with pytest.raises(ValueError, match="checkpoint contract mismatch"):
        run_operator_benchmark(
            construct(),
            scenario,
            steps=5,
            learning_rate=2e-3,
            repeats=1,
            validation_interval=1,
            checkpoint_path=checkpoint,
            resume=True,
            checkpoint_metadata={"commit_identity": "pinned"},
            run_evaluations=False,
        )


def test_architecture_matrix_contains_baselines_and_operator_families():
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=4,
    )
    architectures = compatible_architectures(scenario, quick=True)
    names = {architecture.name for architecture in architectures}
    assert {
        "weighted_mean",
        "nearest_neighbor",
        "identity",
        "pointwise_affine",
        "deeponet",
        "local_integral",
        "fno",
        "tfno",
        "cno",
        "uno",
        "ifno",
        "transolver",
        "gnot",
        "upt",
    } <= names
    assert "pod_deeponet" not in names
    assert all(
        architecture.build(scenario, 0) is not None for architecture in architectures
    )


def test_architecture_registry_models_run_every_declared_evaluation():
    for scenario in standard_operator_benchmarks(quick=True):
        for architecture in compatible_architectures(scenario, quick=True):
            model = architecture.build(scenario, 0)
            assert jnp.all(jnp.isfinite(model(scenario.train_batch)))
            for evaluation in scenario.evaluations:
                assert jnp.all(jnp.isfinite(model(evaluation.batch)))


def test_case_splits_are_disjoint_sized_and_seed_deterministic():
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=10,
    )
    first = split_operator_scenario(scenario, seed=17)
    repeated = split_operator_scenario(scenario, seed=17)
    changed = split_operator_scenario(scenario, seed=18)
    assert first.validation is not None
    assert (
        first.train_batch.case_shape[0]
        + first.validation.batch.case_shape[0]
        + first.evaluations[0].batch.case_shape[0]
        == 10
    )
    assert scenario_checksum(first) == scenario_checksum(repeated)
    assert scenario_checksum(first) != scenario_checksum(changed)


def test_standard_benchmarks_include_controlled_distribution_shifts():
    scenarios = standard_operator_benchmarks(quick=True)
    shifts = {
        evaluation.shift for scenario in scenarios for evaluation in scenario.evaluations
    }
    assert {
        "resolution",
        "geometry",
        "input_noise",
        "sensor_dropout",
        "rollout",
    } <= shifts


def test_matrix_aggregates_seeds_persists_artifacts_and_checks_thresholds(tmp_path):
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=4,
    )
    matrix = run_benchmark_matrix(
        (scenario,),
        seeds=(3, 5),
        architecture_names=("weighted_mean",),
        steps=0,
        repeats=1,
        quick=True,
        commit_identity="unit-test",
    )
    assert len(matrix.results) == 2
    assert all(aggregate.seeds == (3, 5) for aggregate in matrix.aggregates)
    json_path, parquet_path = save_benchmark_artifacts(tmp_path, matrix)
    assert json_path.is_file() and parquet_path.is_file()
    threshold = OperatorBenchmarkThreshold(
        scenario.name,
        "weighted_mean",
        "train_resolution",
        maximum_relative_l2=1.01,
    )
    assert_benchmark_thresholds(matrix, (threshold,))
    failing = OperatorBenchmarkThreshold(
        scenario.name,
        "weighted_mean",
        "train_resolution",
        maximum_relative_l2=0.5,
    )
    with pytest.raises(OperatorBenchmarkRegressionError, match="relative_l2"):
        assert_benchmark_thresholds(matrix, (failing,))


def _aggregate(architecture, relative_l2, inference_seconds):
    return OperatorBenchmarkAggregate(
        scenario="manufactured",
        architecture=architecture,
        family="external" if architecture == "candidate" else "native",
        evaluation="resolution_transfer",
        split="test",
        shift="resolution",
        seeds=(0, 1, 2),
        parameter_count_mean=10.0,
        relative_l2_mean=relative_l2,
        relative_l2_std=0.01,
        absolute_l2_mean=relative_l2,
        h1_mean=None,
        spectral_mean=None,
        conservation_error_mean=0.0,
        maximum_absolute_error_mean=relative_l2,
        compile_seconds_mean=0.1,
        inference_seconds_mean=inference_seconds,
        training_seconds_mean=1.0,
    )


def test_external_candidate_requires_audit_and_uniform_benchmark_superiority(tmp_path):
    checkpoint = tmp_path / "candidate.bin"
    checkpoint.write_bytes(b"candidate-weights")
    checkpoint_digest = phx.nn.operator.adapters.checkpoint_sha256(checkpoint)
    candidate = ExternalOperatorCandidate(
        name="candidate",
        source_uri="https://example.test/source",
        checkpoint_uri="https://example.test/checkpoint",
        revision="abc123",
        code_license="Apache-2.0",
        weights_license="Apache-2.0",
        input_schema_declared=True,
        output_schema_declared=True,
        preprocessing_declared=True,
        normalization_declared=True,
        dataset_provenance_declared=True,
        checkpoint_sha256=checkpoint_digest,
    )
    audit = audit_external_candidate(candidate)
    assert audit.eligible
    metadata_audit = audit
    native = (_aggregate("fno", 1.0, 1.0),)
    unverified = select_benchmark_superior_external(
        candidate,
        metadata_audit,
        (_aggregate("candidate", 0.8, 1.5),),
        native,
    )
    assert not unverified.integrated
    assert "checkpoint artifact" in " ".join(unverified.reasons)
    manifest = phx.nn.operator.adapters.OperatorCheckpointManifest(
        architecture="candidate",
        model_version="1.0.0",
        source_uri=candidate.source_uri,
        checkpoint_uri=candidate.checkpoint_uri,
        revision=candidate.revision,
        input_schema={"u": {"channels": 1}},
        output_schema={"y": {"channels": 1}},
        preprocessing={"layout": "case-query-channel"},
        normalization={"u": {"mean": 0.0, "std": 1.0}},
        dataset_provenance=("synthetic",),
        code_license=candidate.code_license,
        weights_license=candidate.weights_license,
        checkpoint_sha256=checkpoint_digest,
    )
    audit = verify_external_candidate_artifact(candidate, manifest, checkpoint)
    assert audit.eligible
    assert audit.artifact_verified
    accepted = select_benchmark_superior_external(
        candidate,
        audit,
        (_aggregate("candidate", 0.8, 1.5),),
        native,
    )
    assert accepted.integrated

    rejected = select_benchmark_superior_external(
        candidate,
        audit,
        (_aggregate("candidate", 0.98, 1.0),),
        native,
    )
    assert not rejected.integrated
    assert "does not improve relative L2" in rejected.reasons[0]


def test_external_candidate_rejects_missing_weight_provenance():
    candidate = ExternalOperatorCandidate(
        name="unlicensed",
        source_uri="https://example.test/source",
        checkpoint_uri="https://example.test/checkpoint",
        revision="abc123",
        code_license=None,
        weights_license=None,
        input_schema_declared=True,
        output_schema_declared=True,
        preprocessing_declared=False,
        normalization_declared=False,
        dataset_provenance_declared=True,
        checkpoint_sha256=None,
    )
    audit = audit_external_candidate(candidate)
    assert not audit.eligible
    assert "code license is absent or not approved" in audit.reasons
    assert "weights license is absent or not approved" in audit.reasons
    assert "missing valid checkpoint SHA-256" in audit.reasons
