#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import json
import sys
from pathlib import Path

import pytest

from phydrax._external_runtime import PinnedExecutable
from phydrax.algebraic._exact import (
    EliminateArguments,
    ExactSparsePolynomialSystem,
    ExactSymbolicOperation,
    ExactSymbolicStatus,
    NormalFormArguments,
    plan_exact_symbolic,
    QQ,
    UnivariateDiscriminantArguments,
    UnivariateResultantArguments,
)
from phydrax.backends._types import BackendUnavailableError
from phydrax.backends.macaulay2 import (
    macaulay2_availability,
    macaulay2_request_record,
    Macaulay2Environment,
    Macaulay2IdentityError,
    Macaulay2Provider,
    parse_macaulay2_result,
    prepare_macaulay2_symbolic,
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def provider():
    executable = PinnedExecutable(
        sys.executable,
        _digest(sys.executable),
        "1.26.05-test-pin",
        "GPL-3.0-only",
        "https://macaulay2.com/",
    )
    return Macaulay2Provider(Macaulay2Environment(executable))


def _univariate_system():
    return ExactSparsePolynomialSystem.from_coo(
        ("x",),
        ("f", "g"),
        [0, 0, 1, 1],
        [[0], [2], [0], [1]],
        ("-1", "1", "-1", "1"),
        QQ,
    )


def _bivariate_system():
    return ExactSparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f", "g"),
        [0, 0, 1, 1],
        [[0, 0], [1, 0], [0, 0], [0, 1]],
        ("-1", "1", "-1", "1"),
        QQ,
    )


def _prepared(provider, operation):
    if operation is ExactSymbolicOperation.ELIMINATE:
        system = _bivariate_system()
        arguments = EliminateArguments((0,))
    else:
        system = _univariate_system()
        if operation is ExactSymbolicOperation.NORMAL_FORM:
            dividend = ExactSparsePolynomialSystem.from_coo(
                ("x",), ("h",), [0, 0], [[0], [3]], ("1", "1"), QQ
            )
            arguments = NormalFormArguments(dividend)
        elif operation is ExactSymbolicOperation.RESULTANT_UNIVARIATE:
            arguments = UnivariateResultantArguments((0, 1), 0)
        elif operation is ExactSymbolicOperation.DISCRIMINANT_UNIVARIATE:
            arguments = UnivariateDiscriminantArguments(0, 0)
        else:
            arguments = None
    plan = plan_exact_symbolic(system, operation, arguments)
    return prepare_macaulay2_symbolic(plan, provider)


def _response(prepared, polynomials):
    provider = prepared.provider
    environment = provider.environment
    plan = prepared.plan
    return {
        "request_id": prepared.request_id,
        "plan_id": plan.plan_id,
        "system_id": plan.system.system_id,
        "support_id": plan.system.support.support_id,
        "domain": plan.system.domain.to_record(),
        "operation": plan.operation.value,
        "environment_id": environment.environment_id,
        "provider_id": provider.provider_id,
        "provider_version": environment.executable.version,
        "executable_sha256": environment.executable.sha256,
        "worker_sha256": environment.worker_sha256,
        "status": "success",
        "diagnostic": "",
        "claim": "exact_claimed_by_external_provider",
        "polynomials": polynomials,
        "provenance": {
            "provider": "Macaulay2",
            "provider_version": environment.executable.version,
            "executable_sha256": environment.executable.sha256,
            "worker_sha256": environment.worker_sha256,
            "environment_id": environment.environment_id,
            "external_exact_claim": True,
        },
    }


def _encoded(payload):
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize("operation", tuple(ExactSymbolicOperation))
def test_closed_operations_build_data_only_requests_and_parse_exact_fake_results(
    provider, operation
):
    prepared = _prepared(provider, operation)
    request = macaulay2_request_record(prepared)

    assert request["operation"] == operation.value
    assert request["variables"] == [
        f"x{index}" for index in range(prepared.plan.system.variable_count)
    ]
    assert "schema" not in request
    assert not {"code", "packages", "paths"}.intersection(request)
    if operation is ExactSymbolicOperation.ELIMINATE:
        variable_indices = [1]
        exponents = [[0], [1]]
    else:
        variable_indices = [0]
        exponents = [[0], [2]]
    payload = _response(
        prepared,
        {
            "variable_indices": variable_indices,
            "equation_count": 1,
            "equation_indices": [0, 0],
            "exponents": exponents,
            "coefficients": ["-1", "1"],
        },
    )
    result = parse_macaulay2_result(
        _encoded(payload), prepared, run_artifact_id="fake-worker-run"
    )

    assert result.status is ExactSymbolicStatus.SUCCESS
    assert result.output.coefficients == ("-1", "1")
    assert result.evidence.claim == "exact_claimed_by_external_provider"
    assert result.evidence.run_artifact_id == "fake-worker-run"
    assert "operation-variable-order" in result.evidence.independently_checked


def test_malformed_output_and_duplicate_fields_are_rejected(provider):
    prepared = _prepared(provider, ExactSymbolicOperation.GROEBNER_BASIS)
    with pytest.raises(ValueError, match="unexpected field inventory"):
        parse_macaulay2_result(b"{}", prepared)
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        parse_macaulay2_result(b'{"request_id":"a","request_id":"b"}', prepared)


def test_response_identity_mismatch_is_distinct_from_malformed_output(provider):
    prepared = _prepared(provider, ExactSymbolicOperation.GROEBNER_BASIS)
    payload = _response(
        prepared,
        {
            "variable_indices": [0],
            "equation_count": 1,
            "equation_indices": [0],
            "exponents": [[0]],
            "coefficients": ["1"],
        },
    )
    payload["support_id"] = "0" * 64

    with pytest.raises(Macaulay2IdentityError, match="support_id"):
        parse_macaulay2_result(_encoded(payload), prepared)


def test_unavailable_provider_requires_explicit_environment_and_has_no_fallback():
    availability = macaulay2_availability()

    assert not availability.available
    assert "discovery is disabled" in availability.reason
    with pytest.raises(BackendUnavailableError, match="macaulay2"):
        availability.require("algebraic.exact.groebner_basis")
