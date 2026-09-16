#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy

import numpy as np
import pytest

from phydrax.algebraic._positive_dimensional import WitnessSet
from phydrax.algebraic._system import SparsePolynomialSystem
from phydrax.backends.homotopy_geometry import (
    decode_homotopy_geometry_result,
    HOMOTOPY_GEOMETRY_PROTOCOL,
    HOMOTOPY_GEOMETRY_WORKER_SHA256,
    HomotopyGeometryPolicy,
    HomotopyGeometryRequest,
    HomotopyGeometryStatus,
    MembershipEvidence,
)


_PROVIDER_ID = "provider-identity"
_ENVIRONMENT_ID = "environment-identity"


def _wire(value):
    array = np.asarray(value)
    pairs = np.stack((array.real, array.imag), axis=-1)
    return {
        "shape": list(array.shape),
        "values": pairs.reshape((-1, 2)).tolist(),
    }


def _parabola_system():
    return SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("parabola",),
        np.asarray([0, 0]),
        np.asarray([[2, 0], [0, 1]]),
        np.asarray([1.0, -1.0]),
    )


def _parabola_witness(system):
    return WitnessSet(
        system.system_id,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        np.asarray([[-1.0, 1.0], [1.0, 1.0]], dtype=complex),
        np.asarray([0.0, 0.0]),
    )


def _path_records(request, *, failed=(), unattempted=(), targets=None):
    failed = set(failed)
    unattempted = set(unattempted)
    records = []
    for position, path in enumerate(request.paths):
        if position in unattempted:
            status = "not-attempted"
            target = None
            residual = None
            diagnostic = "path budget exhausted"
        elif position in failed:
            status = "tracking-failed"
            target = None
            residual = None
            diagnostic = "tracker step limit"
        else:
            status = "success"
            target = path.source_index if targets is None else targets[position]
            residual = 0.0
            diagnostic = ""
        records.append(
            {
                "path_id": path.path_id,
                "batch_id": path.batch_id,
                "source_index": path.source_index,
                "target_index": target,
                "status": status,
                "residual_norm": residual,
                "diagnostic": diagnostic,
            }
        )
    return records


def _response(request, result, paths, status, *, budget_exhausted=False):
    return {
        "protocol": HOMOTOPY_GEOMETRY_PROTOCOL,
        "request_id": request.request_id,
        "provider_id": _PROVIDER_ID,
        "environment_id": _ENVIRONMENT_ID,
        "worker_sha256": HOMOTOPY_GEOMETRY_WORKER_SHA256,
        "operation": request.operation.value,
        "system_id": request.system_id,
        "support_id": request.support_id,
        "status": status,
        "budget_exhausted": budget_exhausted,
        "paths": paths,
        "result": result,
    }


def _witness_result(points):
    return {
        "dimension": 1,
        "slice_matrix": _wire([[0.0, 1.0]]),
        "slice_offset": _wire([-1.0]),
        "points": _wire(points),
        "residual_norms": [0.0] * len(points),
    }


def _decode(request, policy, response):
    return decode_homotopy_geometry_result(
        request,
        policy,
        response,
        provider_id=_PROVIDER_ID,
        environment_id=_ENVIRONMENT_ID,
    )


def test_generic_slice_decodes_isolated_numerical_witness_data_without_execution():
    system = _parabola_system()
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    policy = HomotopyGeometryPolicy(path_capacity=2)
    response = _response(
        request,
        _witness_result([[-1.0, 1.0], [1.0, 1.0]]),
        _path_records(request, targets=(0, 1)),
        "success",
    )

    decoded = _decode(request, policy, response)

    assert decoded.status is HomotopyGeometryStatus.SUCCESS
    assert decoded.paths.successful
    assert decoded.output.degree == 2
    assert decoded.output.system_id == system.system_id
    assert decoded.run is None


def test_request_and_result_reject_malformed_slices():
    system = _parabola_system()
    with pytest.raises(ValueError, match="codimension"):
        HomotopyGeometryRequest.generic_slice(
            system,
            1,
            np.zeros((0, 2)),
            np.zeros((0,)),
            path_count=1,
        )

    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    malformed = _witness_result([[-1.0, 1.0], [1.0, 1.0]])
    malformed["slice_matrix"] = _wire([[1.0]])
    response = _response(
        request,
        malformed,
        _path_records(request, targets=(0, 1)),
        "success",
    )
    with pytest.raises(ValueError, match="width"):
        _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)


def test_monodromy_decode_rejects_malformed_permutation():
    system = _parabola_system()
    witness = _parabola_witness(system)
    request = HomotopyGeometryRequest.monodromy(
        system,
        witness,
        (("loop-0", np.asarray([[0.0, 1.0]]), np.asarray([-2.0])),),
    )
    result = {
        "witness_set_id": witness.witness_id,
        "point_count": 2,
        "attempted_loop_ids": ["loop-0"],
        "completed": [{"loop_id": "loop-0", "permutation": [1, 1]}],
    }
    response = _response(
        request,
        result,
        _path_records(request, targets=(1, 0)),
        "success",
    )

    with pytest.raises(ValueError, match="valid permutations"):
        _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)


def test_partial_path_failure_is_retained_instead_of_promoted_to_success():
    system = _parabola_system()
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    response = _response(
        request,
        _witness_result([[-1.0, 1.0]]),
        _path_records(request, failed=(1,), targets=(0, 0)),
        "partial-path-failure",
    )

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)

    assert decoded.status is HomotopyGeometryStatus.PARTIAL_PATH_FAILURE
    assert decoded.paths.successful_count == 1
    assert decoded.output.degree == 1
    assert not decoded.successful


def test_budget_exhaustion_retains_unattempted_path_inventory():
    system = _parabola_system()
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    response = _response(
        request,
        _witness_result([[-1.0, 1.0]]),
        _path_records(request, unattempted=(1,), targets=(0, 0)),
        "budget-exhausted",
        budget_exhausted=True,
    )

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)

    assert decoded.status is HomotopyGeometryStatus.BUDGET_EXHAUSTED
    assert decoded.paths.budget_exhausted
    assert decoded.paths.records[1].status.value == "not-attempted"


def test_trace_test_failure_is_an_explicit_non_success_outcome():
    system = _parabola_system()
    witness = _parabola_witness(system)
    request = HomotopyGeometryRequest.trace_test(
        system,
        witness,
        (0, 1),
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([[-0.5], [-1.0], [-1.5]]),
        tolerance=1.0e-8,
    )
    result = {
        "witness_set_id": witness.witness_id,
        "point_indices": [0, 1],
        "sample_parameters": [-1.0, 0.0, 1.0],
        "trace_values": _wire([[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0]]),
        "affine_fit_residual": 1.0e-3,
        "tolerance": 1.0e-8,
        "passed": False,
    }
    response = _response(
        request,
        result,
        _path_records(request),
        "trace-test-failed",
    )

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=6), response)

    assert decoded.status is HomotopyGeometryStatus.TRACE_TEST_FAILED
    assert not decoded.output.passed
    assert not decoded.successful


def test_provider_identity_mismatch_is_retained_as_identity_failure():
    system = _parabola_system()
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    response = _response(
        request,
        _witness_result([[-1.0, 1.0], [1.0, 1.0]]),
        _path_records(request, targets=(0, 1)),
        "success",
    )
    response["provider_id"] = "different-provider"

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)

    assert decoded.status is HomotopyGeometryStatus.IDENTITY_MISMATCH
    assert decoded.output is None
    assert decoded.paths is None
    assert "provider_id" in decoded.error


def test_image_degree_decodes_a_pseudo_witness_without_certification_claim():
    system = _parabola_system()
    map_system = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("x-coordinate",),
        np.asarray([0]),
        np.asarray([[1, 0]]),
        np.asarray([1.0]),
    )
    request = HomotopyGeometryRequest.image_degree(
        system,
        map_system,
        "x-projection",
        1,
        1,
        np.zeros((0, 2)),
        np.zeros((0,)),
        np.asarray([[1.0]]),
        np.asarray([-1.0]),
        path_count=1,
    )
    result = {
        "source_system_id": system.system_id,
        "map_id": "x-projection",
        "source_dimension": 1,
        "image_dimension": 1,
        "source_slice_matrix": _wire(np.zeros((0, 2))),
        "source_slice_offset": _wire(np.zeros((0,))),
        "image_slice_matrix": _wire([[1.0]]),
        "image_slice_offset": _wire([-1.0]),
        "source_points": _wire([[1.0, 1.0]]),
        "image_points": _wire([[1.0]]),
        "residual_norms": [0.0],
        "image_degree": 1,
    }
    response = _response(request, result, _path_records(request, targets=(0,)), "success")

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=1), response)

    assert decoded.output.image_degree == 1
    assert decoded.output.map_id == "x-projection"
    with pytest.raises(AttributeError):
        _ = decoded.output.certified


def test_membership_output_uses_qualified_witness_transport_evidence():
    system = _parabola_system()
    witness = _parabola_witness(system)
    request = HomotopyGeometryRequest.membership(
        system,
        (witness,),
        np.asarray([[0.0, 0.0], [0.0, 1.0]]),
        tolerance=1.0e-8,
    )
    result = {
        "query_points": _wire([[0.0, 0.0], [0.0, 1.0]]),
        "member_witness_set_ids": [[witness.witness_id], []],
        "residual_norms": [0.0, 1.0],
        "tolerance": 1.0e-8,
        "claim": "numerical-witness-transport-membership-not-exact-ideal-membership",
    }
    response = _response(request, result, _path_records(request), "success")

    decoded = _decode(request, HomotopyGeometryPolicy(path_capacity=4), response)

    assert isinstance(decoded.output, MembershipEvidence)
    assert decoded.output.member_witness_set_ids == ((witness.witness_id,), ())
    assert decoded.output.evidence_complete
    assert "not-exact-ideal-membership" in decoded.output.claim


def test_response_status_cannot_falsely_claim_success_over_failed_paths():
    system = _parabola_system()
    request = HomotopyGeometryRequest.generic_slice(
        system,
        1,
        np.asarray([[0.0, 1.0]]),
        np.asarray([-1.0]),
        path_count=2,
    )
    response = _response(
        request,
        _witness_result([[-1.0, 1.0]]),
        _path_records(request, failed=(1,), targets=(0, 0)),
        "success",
    )

    with pytest.raises(ValueError, match="contradicts"):
        _decode(request, HomotopyGeometryPolicy(path_capacity=2), response)

    extra_field = copy.deepcopy(response)
    extra_field["unbounded_diagnostic"] = "not admitted"
    with pytest.raises(ValueError, match="exact protocol fields"):
        _decode(request, HomotopyGeometryPolicy(path_capacity=2), extra_field)
