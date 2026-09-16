#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import numpy as np
import pytest

from phydrax.applications.conformal_bootstrap import (
    audit_pmp_samples,
    ConformalPolynomialMatrixProgram,
    DampedRationalPrefactor,
    parse_sdpb_output,
    parse_sdpb_vector,
    PolynomialMatrixBlock,
    reconstruct_pmp_functional,
)


def _program():
    block = PolynomialMatrixBlock(
        DampedRationalPrefactor("0.3678794411714423215955", "1"),
        (
            (
                (
                    ("1", "0", "0", "0", "1"),
                    ("0", "0", "1", "0", "0.0833333333333333333333"),
                ),
            ),
        ),
        sample_points=("0", "1"),
        sample_scalings=("1", "1"),
        reduced_sample_scalings=("1", "1"),
    )
    return ConformalPolynomialMatrixProgram(
        ("0", "-1"),
        ("1", "0"),
        (block,),
        frontend_id="independent-polynomial-fixture",
        frontend_precision_bits=256,
    )


def test_pmp_serialization_preserves_exact_decimal_strings():
    program = _program()
    record = json.loads(program.to_json_bytes())
    assert record["objective"] == ["0", "-1"]
    polynomial = record["PositiveMatrixWithPrefactorArray"][0]["polynomials"][0][0][1]
    assert polynomial[-1] == "0.0833333333333333333333"
    assert program.pmp_id


def test_finite_pmp_audit_reports_sampled_psd_only():
    program = _program()
    accepted = audit_pmp_samples(program, (1.0, 0.0), (0.0, 0.5, 1.0))
    assert bool(accepted.accepted)
    np.testing.assert_allclose(accepted.normalization_residual, 0.0)
    assert "not-continuum" in accepted.claim
    rejected = audit_pmp_samples(program, (1.0, -24.0), (1.0,))
    assert not bool(rejected.positive_semidefinite)
    assert not bool(rejected.accepted)


def test_sdpb_documented_output_and_dual_vector_reconstruct_functional():
    output = (
        b'terminateReason = "found primal-dual optimal solution";\n'
        b"primalObjective = 1.840265763132049246688;\n"
        b"dualObjective = 1.840265763132049246687;\n"
        b"dualityGap = 3.5e-31;\n"
        b"primalError = 8.0e-136;\n"
        b"dualError = 3.6e-131;\n"
    )
    summary = parse_sdpb_output(output)
    assert summary.terminate_reason == "found primal-dual optimal solution"
    assert summary.duality_gap == "3.5e-31"
    free = parse_sdpb_vector(b"1 1\n-1.840265763132049246688\n")
    functional = reconstruct_pmp_functional(_program(), free)
    assert functional == ("1", "-1.840265763132049246688")


def test_pmp_rejects_nonsymmetric_polynomial_matrix():
    prefactor = DampedRationalPrefactor("0.5", "1")
    with pytest.raises(ValueError, match="symmetric"):
        PolynomialMatrixBlock(
            prefactor,
            (
                ((("1",),), (("2",),)),
                ((("3",),), (("1",),)),
            ),
        )


def test_sdpb_parser_rejects_incomplete_summary():
    with pytest.raises(ValueError, match="exactly one"):
        parse_sdpb_output(b'terminateReason = "maxIterations exceeded";\n')
    with pytest.raises(ValueError, match="payload size"):
        parse_sdpb_vector(b"2 1\n1\n")
