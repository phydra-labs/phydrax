#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib

import pytest

from phydrax._external_runtime import pin_energy_executable
from phydrax.backends.homotopy_continuation import (
    execute_homotopy_continuation,
    homotopy_continuation_availability,
    HomotopyContinuationEnvironment,
    HomotopyContinuationExecutionStatus,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
    HomotopyContinuationRequest,
)


_UUID = "f213a82b-91d6-5c58-9dd6-746a5d3f2e4c"


def _provider(tmp_path, behavior="complete"):
    project = tmp_path / "julia-project"
    project.mkdir()
    project_bytes = (
        b'[deps]\nHomotopyContinuation = "f213a82b-91d6-5c58-9dd6-746a5d3f2e4c"\n'
    )
    manifest_bytes = b'manifest_format = "2.0"\n'
    (project / "Project.toml").write_bytes(project_bytes)
    (project / "Manifest.toml").write_bytes(manifest_bytes)
    executable = tmp_path / "fake-julia"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys

behavior = %r
with open("homotopy-input.json", encoding="ascii") as stream:
    request = json.load(stream)
if behavior == "process-failure":
    sys.exit(7)
path = {
    "path_index": 0,
    "return_code": "success",
    "status": "regular_endpoint",
    "endpoint": [[1.0, 0.0]],
    "provider_residual_norm": 1e-14,
    "condition_number": 1.0,
}
record = {
    "protocol_id": request["protocol_id"],
    "request_id": request["request_id"],
    "support_id": request["support_id"],
    "system_id": request["system_id"],
    "environment_id": request["environment_id"],
    "policy_id": request["policy_id"],
    "homotopy_continuation_uuid": request["homotopy_continuation_uuid"],
    "homotopy_continuation_version": request["homotopy_continuation_version"],
    "start_system": request["start_system"],
    "execution_status": "complete",
    "start_count": 1,
    "tracked_path_count": 1,
    "counts": {
        "at_infinity": 0,
        "excess_solution": 0,
        "invalid_endpoint": 0,
        "regular_endpoint": 1,
        "singular_endpoint_candidate": 0,
        "tracking_failed": 0,
    },
    "paths": [path],
}
if behavior == "identity-mismatch":
    record["request_id"] = "wrong-request"
elif behavior == "shape-mismatch":
    record["paths"][0]["endpoint"] = [[1.0, 0.0], [2.0, 0.0]]
elif behavior == "path-count-mismatch":
    record["tracked_path_count"] = 2
with open("homotopy-output.json", "w", encoding="ascii") as stream:
    json.dump(record, stream, sort_keys=True)
"""
        % behavior,
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = HomotopyContinuationEnvironment(
        project,
        hashlib.sha256(project_bytes).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
        _UUID,
        "2.15.0",
    )
    return HomotopyContinuationProvider(
        pin_energy_executable(executable, version="1.11.7", license_id="MIT"),
        environment,
    )


def _request():
    return HomotopyContinuationRequest(
        "request",
        "support",
        "system",
        1,
        1,
        (0, 0),
        ((0,), (2,)),
        (-1.0, 1.0),
    )


def test_explicit_provider_retains_protocol_identity_shape_and_raw_path_evidence(
    tmp_path,
):
    provider = _provider(tmp_path)
    result = execute_homotopy_continuation(
        provider,
        HomotopyContinuationPolicy(
            start_system="total-degree", path_capacity=2, timeout_seconds=10
        ),
        _request(),
    )

    assert result.status is HomotopyContinuationExecutionStatus.COMPLETE
    assert result.start_count == result.tracked_path_count == 1
    assert result.paths[0].return_code == "success"
    assert result.paths[0].endpoint == (1 + 0j,)
    assert dict(result.counts)["regular_endpoint"] == 1
    assert result.run is not None and result.run.returncode == 0
    assert homotopy_continuation_availability(provider).available
    assert not homotopy_continuation_availability().available


@pytest.mark.parametrize(
    ("behavior", "status"),
    (
        ("identity-mismatch", HomotopyContinuationExecutionStatus.SEMANTIC_MISMATCH),
        ("shape-mismatch", HomotopyContinuationExecutionStatus.INVALID_OUTPUT),
        ("path-count-mismatch", HomotopyContinuationExecutionStatus.INVALID_OUTPUT),
    ),
)
def test_output_identity_shape_and_path_accounting_fail_separately(
    tmp_path, behavior, status
):
    result = execute_homotopy_continuation(
        _provider(tmp_path, behavior),
        HomotopyContinuationPolicy(path_capacity=2, timeout_seconds=10),
        _request(),
    )

    assert result.status is status
    assert result.run is not None


def test_process_failure_retains_raw_return_code(tmp_path):
    result = execute_homotopy_continuation(
        _provider(tmp_path, "process-failure"),
        HomotopyContinuationPolicy(path_capacity=2, timeout_seconds=10),
        _request(),
    )

    assert result.status is HomotopyContinuationExecutionStatus.PROVIDER_FAILED
    assert result.run is not None
    assert result.run.returncode == 7


def test_project_pin_is_rechecked_before_execution(tmp_path):
    provider = _provider(tmp_path)
    (tmp_path / "julia-project" / "Project.toml").write_text("[deps]\n", encoding="utf-8")

    result = execute_homotopy_continuation(
        provider,
        HomotopyContinuationPolicy(path_capacity=2),
        _request(),
    )

    assert result.status is HomotopyContinuationExecutionStatus.PROVIDER_FAILED
    assert result.run is None
    assert "before execution" in result.error
