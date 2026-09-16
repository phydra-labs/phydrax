#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite conformal-data, block, crossing, and PMP candidate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from phydrax.applications.conformal_bootstrap import (
    audit_pmp_samples,
    conformal_bootstrap_candidate_profiles,
    ConformalDataPlan,
    ConformalPolynomialMatrixProgram,
    CrossingChannel,
    CrossingSectorPlan,
    DampedRationalPrefactor,
    ExchangedOperatorSector,
    ExternalScalarOperator,
    GlobalScalarBlockPlan,
    ising_sigma_crossing_evidence,
    PolynomialMatrixBlock,
    prepare_crossing_sectors,
    prepare_global_scalar_blocks,
)


jax.config.update("jax_enable_x64", True)


def _data():
    external = tuple(
        ExternalScalarOperator(f"phi-{index}", 0.5, "scalar") for index in range(4)
    )
    return ConformalDataPlan(
        2.0,
        external,
        (
            ExchangedOperatorSector(
                "even",
                "scalar",
                (0, 2),
                parity="even",
                minimum_dimension=0.1,
            ),
        ),
        (CrossingChannel("s-t", (2, 1, 0, 3), involutive=True),),
        category=None,
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
        sample_points=("0", "0.5", "1"),
        sample_scalings=("1", "1", "1"),
        reduced_sample_scalings=("1", "1", "1"),
    )
    return ConformalPolynomialMatrixProgram(
        ("0", "-1"),
        ("1", "0"),
        (block,),
        frontend_id="phydrax-independent-polynomial-control",
        frontend_precision_bits=256,
    )


def run_qualification() -> dict[str, object]:
    data = _data()
    crossing = prepare_crossing_sectors(
        CrossingSectorPlan(
            data,
            ("identity",),
            (((1.0,),),),
            basis_gauge_id="one-component-control",
        )
    )
    blocks = prepare_global_scalar_blocks(
        GlobalScalarBlockPlan(
            data,
            ((0.2, 0.2), (0.3, 0.3)),
            (0,),
            ((0, 0), (1, 0)),
            recursion_order=4,
            hypergeometric_order=384,
        )
    )
    block_evidence = blocks.evidence(2.0, 0)
    closed = np.log1p(-np.asarray((0.2, 0.3))) ** 2
    block_residual = float(np.max(np.abs(np.asarray(block_evidence.values) - closed)))
    program = _program()
    audit = audit_pmp_samples(program, (1.0, 0.0), (0.0, 0.5, 1.0))
    virasoro = ising_sigma_crossing_evidence((0.2, 0.35, 0.65, 0.8))
    successful = bool(
        crossing.evidence.accepted
        and block_evidence.finite
        and block_residual <= 1e-11
        and audit.accepted
        and virasoro.accepted
    )
    return {
        "kind": "conformal-bootstrap-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in conformal_bootstrap_candidate_profiles()
        ],
        "case": {
            "data_plan_id": data.plan_id,
            "crossing_plan_id": crossing.plan.plan_id,
            "block_plan_id": blocks.plan.plan_id,
            "pmp_id": program.pmp_id,
            "block_method": blocks.method,
            "virasoro_crossing_id": virasoro.crossing_id,
        },
        "raw": {
            "block_values": np.asarray(block_evidence.values).tolist(),
            "block_derivatives": np.asarray(block_evidence.derivatives).tolist(),
            "block_truncation_proxy": np.asarray(
                block_evidence.truncation_proxy
            ).tolist(),
            "block_casimir_residuals": np.asarray(
                block_evidence.casimir_residuals
            ).tolist(),
            "crossing_involution_residuals": np.asarray(
                crossing.evidence.involution_residuals
            ).tolist(),
            "pmp_json": program.to_json_bytes().decode("ascii"),
            "pmp_sample_minimum_eigenvalues": np.asarray(
                audit.minimum_eigenvalues
            ).tolist(),
            "virasoro_direct_correlator": np.asarray(virasoro.direct_correlator).tolist(),
            "virasoro_crossing_residuals": np.asarray(
                virasoro.pointwise_residual
            ).tolist(),
        },
        "criteria": {
            "closed_block_residual": block_residual,
            "crossing_accepted": bool(crossing.evidence.accepted),
            "pmp_sample_audit_accepted": bool(audit.accepted),
            "virasoro_maximum_crossing_residual": float(virasoro.maximum_residual),
        },
        "successful": successful,
        "claim": "finite-global-bpz-virasoro-and-pmp-candidate-no-continuum-cft-exclusion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
