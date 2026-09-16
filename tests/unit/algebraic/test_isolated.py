#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib

import numpy as np
import pytest

import phydrax.algebraic._isolated as isolated_module
from phydrax._external_runtime import pin_energy_executable
from phydrax.algebraic._isolated import (
    IsolatedPolynomialRootProblem,
    plan_isolated_roots,
    PolynomialPathStatus,
    PolynomialSolveStatus,
    prepare_isolated_roots,
    refresh_isolated_roots,
    solve_prepared_isolated_roots,
)
from phydrax.algebraic._system import SparsePolynomialSystem
from phydrax.backends.homotopy_continuation import (
    HomotopyContinuationEnvironment,
    HomotopyContinuationExecution,
    HomotopyContinuationExecutionStatus,
    HomotopyContinuationPathRecord,
    HomotopyContinuationPathStatus,
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
)


_UUID = "f213a82b-91d6-5c58-9dd6-746a5d3f2e4c"
_COUNT_KEYS = (
    "at_infinity",
    "excess_solution",
    "invalid_endpoint",
    "regular_endpoint",
    "singular_endpoint_candidate",
    "tracking_failed",
)


def _provider(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    project_bytes = b"[deps]\n"
    manifest_bytes = b'manifest_format = "2.0"\n'
    (project / "Project.toml").write_bytes(project_bytes)
    (project / "Manifest.toml").write_bytes(manifest_bytes)
    executable = tmp_path / "julia"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
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


def _quadratic():
    return SparsePolynomialSystem.from_coo(
        ("x",),
        ("x-squared-minus-one",),
        (0, 0),
        ((0,), (2,)),
        (-1.0, 1.0),
    )


def _execution(prepared, paths):
    counts = {key: 0 for key in _COUNT_KEYS}
    for path in paths:
        counts[path.status.value] += 1
    plan = prepared.plan
    request = prepared.request
    return HomotopyContinuationExecution(
        HomotopyContinuationExecutionStatus.COMPLETE,
        tuple(paths),
        tuple((key, counts[key]) for key in _COUNT_KEYS),
        len(paths),
        len(paths),
        request.request_id,
        request.support_id,
        request.system_id,
        plan.provider.provider_id,
        plan.policy.policy_id,
        None,
        "",
        "fake-execution",
    )


def _path(index, endpoint, *, status=HomotopyContinuationPathStatus.REGULAR_ENDPOINT):
    return HomotopyContinuationPathRecord(
        index,
        "success" if endpoint is not None else "step_limit",
        status,
        endpoint,
        0.0 if endpoint is not None else None,
        1.0 if endpoint is not None else None,
    )


def _prepared(tmp_path, **plan_options):
    policy = HomotopyContinuationPolicy(start_system="total-degree", path_capacity=4)
    plan = plan_isolated_roots(
        IsolatedPolynomialRootProblem(_quadratic()),
        _provider(tmp_path),
        policy=policy,
        **plan_options,
    )
    return prepare_isolated_roots(plan)


def test_deterministic_clustering_retains_every_raw_path_and_near_real_evidence(
    tmp_path, monkeypatch
):
    prepared = _prepared(
        tmp_path,
        residual_tolerance=1e-3,
        cluster_tolerance=1e-6,
        near_real_absolute_tolerance=1e-12,
        near_real_relative_tolerance=0,
    )
    paths = (
        _path(0, (1 + 0j,)),
        _path(1, (-1 + 0j,)),
        _path(2, (1 + 1e-10j,)),
        _path(3, (-1 + 1e-10j,)),
    )
    monkeypatch.setattr(
        isolated_module,
        "execute_homotopy_continuation",
        lambda provider, policy, request: _execution(prepared, paths),
    )

    result = solve_prepared_isolated_roots(prepared)

    assert result.status is PolynomialSolveStatus.SUCCESS
    assert result.roots.shape == (4, 1)
    assert result.root_mask.tolist() == [True, True, False, False]
    np.testing.assert_allclose(result.roots[:2, 0].real, [-1, 1], atol=1e-12)
    assert result.cluster_path_counts[:2].tolist() == [2, 2]
    assert result.near_real_mask[:2].tolist() == [True, True]
    assert len(result.paths) == result.coverage.tracked_path_count == 4
    assert result.coverage.all_tracked_paths_accounted
    assert result.paths[3].near_real is False
    assert result.paths[3].independently_accepted
    assert result.paths[3].cluster_index == 0
    assert result.provider.raw_return_codes == ("success",) * 4


def test_original_system_residual_rejects_provider_claim_independently(
    tmp_path, monkeypatch
):
    prepared = _prepared(tmp_path, residual_tolerance=1e-8)
    paths = (_path(0, (2 + 0j,)),)
    monkeypatch.setattr(
        isolated_module,
        "execute_homotopy_continuation",
        lambda provider, policy, request: _execution(prepared, paths),
    )

    result = solve_prepared_isolated_roots(prepared)

    assert result.status is PolynomialSolveStatus.RESIDUAL_REJECTED
    assert not result.root_mask.any()
    assert result.paths[0].provider_residual_norm == 0.0
    assert result.paths[0].original_residual_norm == pytest.approx(3.0)
    assert not result.paths[0].independently_accepted


def test_tracking_failure_is_not_hidden_by_a_valid_endpoint(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    paths = (
        _path(0, (1 + 0j,)),
        _path(
            1,
            None,
            status=HomotopyContinuationPathStatus.TRACKING_FAILED,
        ),
    )
    monkeypatch.setattr(
        isolated_module,
        "execute_homotopy_continuation",
        lambda provider, policy, request: _execution(prepared, paths),
    )

    result = solve_prepared_isolated_roots(prepared)

    assert result.status is PolynomialSolveStatus.PARTIAL_PATH_FAILURE
    assert result.paths[1].status is PolynomialPathStatus.TRACKING_FAILED
    assert result.coverage.tracking_failed_count == 1
    assert result.coverage.classified_path_count == 2


def test_total_degree_capacity_fails_before_provider_execution(tmp_path, monkeypatch):
    plan = plan_isolated_roots(
        _quadratic(),
        _provider(tmp_path),
        policy=HomotopyContinuationPolicy(start_system="total-degree", path_capacity=1),
    )
    prepared = prepare_isolated_roots(plan)

    def unexpected_provider_call(provider, policy, request):
        raise AssertionError("provider must not run above the declared path capacity")

    monkeypatch.setattr(
        isolated_module,
        "execute_homotopy_continuation",
        unexpected_provider_call,
    )

    result = solve_prepared_isolated_roots(prepared)

    assert result.status is PolynomialSolveStatus.PATH_CAPACITY_EXCEEDED
    assert result.coverage.start_count == 2
    assert result.coverage.tracked_path_count == 0
    assert result.provider.start_count == 2


def test_refresh_rejects_support_structural_mismatch(tmp_path):
    prepared = _prepared(tmp_path)
    changed_support = SparsePolynomialSystem.from_coo(
        ("x",),
        ("x-cubed-minus-one",),
        (0, 0),
        ((0,), (3,)),
        (-1.0, 1.0),
    )

    with pytest.raises(ValueError, match="structural support mismatch"):
        refresh_isolated_roots(prepared, changed_support)
