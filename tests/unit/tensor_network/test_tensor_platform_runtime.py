#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.backends._types import BackendAvailability, BackendCapabilities
from phydrax.qualification import ReleaseGateEvidence
from phydrax.solver._runtime_lifecycle import RuntimeCheckpointEnvelope
from phydrax.tensor_network._core import (
    LocallyPurifiedDensity,
    MatrixProductOperator,
    MatrixProductState,
)
from phydrax.tensor_network._platform_archive import (
    read_tensor_network_archive,
    TensorNetworkArchiveKind,
    TensorNetworkArchiveMismatchError,
    write_tensor_network_archive,
)
from phydrax.tensor_network._platform_qualification import (
    evaluate_tensor_network_release,
    TensorNetworkClaimEvidence,
    TensorNetworkQualificationProfile,
    TensorNetworkQualificationResult,
    TensorNetworkReleaseGate,
)
from phydrax.tensor_network._platform_runtime import (
    compare_tensor_network_replays,
    redact_tensor_network_telemetry,
    TensorNetworkAcceptedCheckpointBoundary,
    TensorNetworkReplayRecord,
    TensorNetworkRunStatus,
    TensorNetworkRunSupervisor,
    TensorNetworkTelemetryPolicy,
)
from phydrax.tensor_network._platform_support import (
    admit_tensor_network_resources,
    forecast_tensor_network_resources,
    TensorNetworkClaim,
    TensorNetworkExecutionManifest,
    TensorNetworkFailure,
    TensorNetworkResourcePolicy,
    TensorNetworkSupportTuple,
)


def _mps() -> MatrixProductState:
    return MatrixProductState(
        (
            jnp.arange(4, dtype=jnp.float32).reshape(1, 2, 2),
            jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2),
            jnp.arange(4, dtype=jnp.float32).reshape(2, 2, 1),
        )
    )


def _support(state: MatrixProductState) -> TensorNetworkSupportTuple:
    return TensorNetworkSupportTuple(
        representation="mps",
        workflow="finite-evolution",
        boundary="open",
        algorithm="two-site-tdvp",
        backend=jax.default_backend(),
        dtype=np.dtype(state.tensors[0].dtype).name,
    )


def _execution(
    state: MatrixProductState,
) -> tuple[TensorNetworkExecutionManifest, TensorNetworkSupportTuple]:
    support = _support(state)
    policy = TensorNetworkResourcePolicy(
        maximum_compile_units=1_000,
        maximum_host_bytes=1_000_000,
        maximum_device_bytes=1_000_000,
        maximum_output_queue_bytes=1_000_000,
    )
    forecast = forecast_tensor_network_resources(state, support, policy)
    admission = admit_tensor_network_resources(forecast, (support,))
    capabilities = BackendCapabilities(
        backend=jax.default_backend(),
        problem_kinds=(support.workflow,),
        execution="host",
        host_only=True,
        supports_matrix_free=True,
        supports_assembled=False,
        coordinate_dtypes=(support.dtype,),
    )
    evidence = BackendAvailability(
        capabilities=capabilities,
        available=True,
        requirement="jax runtime",
        reason="selected runtime is available",
        versions=(("jax", jax.__version__),),
    )
    return (
        TensorNetworkExecutionManifest(
            support,
            admission,
            evidence,
            structure_id=state.structure_id,
            method_id=support.algorithm,
            precision_policy_id=state.precision.policy_id,
            source_id="source-fixture",
            input_id="input-fixture",
        ),
        support,
    )


def test_pickle_free_archive_round_trips_explicit_supported_kinds(tmp_path) -> None:
    mps = _mps()
    mpo = MatrixProductOperator(
        (
            jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 2, 2),
            jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2, 1),
        )
    )
    lpdo = LocallyPurifiedDensity(
        (
            jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 2, 2),
            jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2, 1),
        )
    )
    artifacts = (
        (TensorNetworkArchiveKind.MPS, mps),
        (TensorNetworkArchiveKind.MPO, mpo),
        (TensorNetworkArchiveKind.LPDO, lpdo),
    )
    for kind, value in artifacts:
        path = tmp_path / f"{kind.value}.phx"
        written = write_tensor_network_archive(
            path, value, kind=kind, source_id="round-trip-source"
        )
        restored = read_tensor_network_archive(
            path,
            kind=kind,
            expected_structure_id=value.structure_id,
            expected_precision_policy_id=value.precision.policy_id,
            expected_source_id="round-trip-source",
        )
        assert restored.record.artifact_id == written.record.artifact_id
        assert restored.value.structure_id == value.structure_id
        for left, right in zip(restored.value.tensors, value.tensors, strict=True):
            np.testing.assert_array_equal(left, right)

    tree = {"left": jnp.arange(3, dtype=jnp.int32), "right": (jnp.ones((2,)),)}
    path = tmp_path / "tree.phx"
    write_tensor_network_archive(
        path,
        tree,
        kind=TensorNetworkArchiveKind.ARRAY_PYTREE,
        source_id="tree-source",
        structure_id="tree-structure",
        precision_policy_id="tree-policy",
    )
    restored_tree = read_tensor_network_archive(
        path,
        kind=TensorNetworkArchiveKind.ARRAY_PYTREE,
        template=tree,
        expected_structure_id="tree-structure",
        expected_precision_policy_id="tree-policy",
        expected_source_id="tree-source",
    )
    np.testing.assert_array_equal(restored_tree.value["left"], tree["left"])
    with pytest.raises(TensorNetworkArchiveMismatchError):
        read_tensor_network_archive(
            path,
            kind=TensorNetworkArchiveKind.ARRAY_PYTREE,
            template=tree,
            expected_source_id="different-source",
        )


def test_resource_refusal_is_explicit_and_typed() -> None:
    state = _mps()
    support = _support(state)
    policy = TensorNetworkResourcePolicy(
        maximum_compile_units=1,
        maximum_host_bytes=1,
        maximum_device_bytes=1,
        maximum_output_queue_bytes=1,
        maximum_array_leaves=1,
        maximum_elements=1,
    )
    forecast = forecast_tensor_network_resources(state, support, policy)
    refusal = admit_tensor_network_resources(forecast, (support,))
    assert not refusal.admitted
    assert refusal.failure == TensorNetworkFailure.RESOURCE_REFUSED
    assert refusal.reasons

    unsupported = TensorNetworkSupportTuple(
        representation="mps",
        workflow="other-workflow",
        boundary="open",
        algorithm="two-site-tdvp",
        backend=jax.default_backend(),
        dtype="float32",
    )
    unsupported_refusal = admit_tensor_network_resources(forecast, (unsupported,))
    assert unsupported_refusal.failure == TensorNetworkFailure.UNSUPPORTED_TUPLE


def test_telemetry_redacts_before_persisting_or_fingerprinting() -> None:
    policy = TensorNetworkTelemetryPolicy(
        ("iteration", "operator_norm", "access_token"),
        redacted_fields=("access_token",),
    )
    secret = "unrecoverable-secret"
    record = redact_tensor_network_telemetry(
        policy,
        {"iteration": 3, "operator_norm": 1.25, "access_token": secret},
        run_id="run-1",
        event="accepted-step",
        sequence=4,
        timestamp_ns=10,
    )
    assert dict(record.attributes)["access_token"] == "<redacted>"
    assert secret not in repr(record)
    assert record.redacted_fields == ("access_token",)


def test_accepted_checkpoint_and_replay_compatibility_are_exact(tmp_path) -> None:
    state = _mps()
    execution, _ = _execution(state)
    boundary = TensorNetworkAcceptedCheckpointBoundary(
        tmp_path / "checkpoints", execution
    )
    envelope = RuntimeCheckpointEnvelope(
        state.tensors,
        time=1.0,
        step_index=2,
        schedule_cursor=2,
        mesh_id=execution.structure_id,
        method_id=execution.method_id,
        precision_id=execution.precision_policy_id,
        topology_epoch_id=execution.source_id,
    )
    rejected = boundary.publish(0, envelope, accepted=False)
    assert not rejected.published
    assert rejected.failure == TensorNetworkFailure.CHECKPOINT_NOT_ACCEPTED
    publication = boundary.publish(0, envelope, accepted=True)
    assert publication.published and publication.record is not None
    restored = boundary.latest(state.tensors)
    assert restored.checkpoint_id == envelope.checkpoint_id

    reference = TensorNetworkReplayRecord(
        execution,
        state.tensors,
        checkpoint_id=envelope.checkpoint_id,
        route_id="route-1",
        accepted_steps=2,
    )
    matching = TensorNetworkReplayRecord(
        execution,
        state.tensors,
        checkpoint_id=envelope.checkpoint_id,
        route_id="route-1",
        accepted_steps=2,
    )
    changed = TensorNetworkReplayRecord(
        execution,
        (state.tensors[0] + 1.0,) + state.tensors[1:],
        checkpoint_id=envelope.checkpoint_id,
        route_id="route-1",
        accepted_steps=2,
    )
    assert compare_tensor_network_replays(reference, matching).compatible
    mismatch = compare_tensor_network_replays(reference, changed)
    assert not mismatch.compatible
    assert mismatch.failure == TensorNetworkFailure.REPLAY_MISMATCH

    supervisor = TensorNetworkRunSupervisor(execution)
    assert supervisor.start().status == TensorNetworkRunStatus.RUNNING
    supervisor.record_checkpoint(publication)
    assert supervisor.complete(matching).status == TensorNetworkRunStatus.COMPLETED

    cancelled = TensorNetworkRunSupervisor(execution)
    cancelled.start()
    cancelled_state = cancelled.request_cancellation("operator cancellation")
    assert cancelled_state.status == TensorNetworkRunStatus.CANCELLED
    assert cancelled_state.failure == TensorNetworkFailure.CANCELLED


def test_release_gate_decision_uses_computed_claim_evidence() -> None:
    state = _mps()
    support = _support(state)
    claims = (
        TensorNetworkClaim.FINITE_EXECUTION,
        TensorNetworkClaim.DETERMINISTIC_REPLAY,
    )
    gates = (
        TensorNetworkReleaseGate.CODE_VERIFICATION,
        TensorNetworkReleaseGate.ARCHIVE_REPLAY,
    )
    profile = TensorNetworkQualificationProfile(
        (support,),
        {claim: 1e-6 for claim in claims},
        required_claims=claims,
        required_release_gates=gates,
    )
    evidence = tuple(
        TensorNetworkClaimEvidence(
            profile,
            support,
            claim,
            jnp.asarray([0.0, 5e-7]),
            source_id=f"computed-{claim.value}",
        )
        for claim in claims
    )
    result = TensorNetworkQualificationResult(profile, support, evidence)
    assert result.passed
    passing_gates = tuple(
        ReleaseGateEvidence(
            gate.value,
            passed=True,
            evidence_ids=(f"artifact-{gate.value}",),
            reviewer_id="independent-reviewer",
            issued_at=1,
            expires_at=10,
        )
        for gate in gates
    )
    released = evaluate_tensor_network_release(result, passing_gates, evaluated_at=5)
    assert released.released
    assert released.failure == TensorNetworkFailure.NONE

    rejected_gate = ReleaseGateEvidence(
        TensorNetworkReleaseGate.ARCHIVE_REPLAY.value,
        passed=False,
        evidence_ids=("failed-replay",),
        reviewer_id="independent-reviewer",
        issued_at=1,
        expires_at=10,
    )
    refused = evaluate_tensor_network_release(
        result,
        (passing_gates[0], rejected_gate),
        evaluated_at=5,
    )
    assert not refused.released
    assert refused.failure == TensorNetworkFailure.RELEASE_GATE_FAILED
