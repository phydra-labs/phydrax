#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Count-distribution and multiple-testing qualification for omics statistics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import omics
from tools.bioinformatics_common_qualification import (
    emit_report,
    external_dataset_campaign,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _nb2_log_probability(count: int, mean: float, dispersion: float) -> float:
    if dispersion == 0.0:
        return count * math.log(mean) - mean - math.lgamma(count + 1.0)
    inverse = 1.0 / dispersion
    return (
        math.lgamma(count + inverse)
        - math.lgamma(inverse)
        - math.lgamma(count + 1.0)
        + inverse * math.log(inverse / (inverse + mean))
        + count * math.log(mean / (inverse + mean))
    )


def _negative_binomial_case() -> dict[str, object]:
    counts = jnp.asarray((0, 1, 4, 7), dtype=jnp.int32)
    means = jnp.asarray((0.7, 1.3, 3.2, 6.8))
    dispersion = jnp.asarray(0.35)
    observed = omics.negative_binomial_log_probability(counts, means, dispersion)
    oracle = np.asarray(
        [
            _nb2_log_probability(int(count), float(mean), float(dispersion))
            for count, mean in zip(np.asarray(counts), np.asarray(means), strict=True)
        ]
    )
    probability_error = float(np.max(np.abs(np.asarray(observed) - oracle)))

    count = jnp.asarray(4.0)
    mean = jnp.asarray(3.2)

    def objective(value):
        return omics.negative_binomial_log_probability(count, value, dispersion)

    automatic_gradient = float(np.asarray(jax.grad(objective)(mean)))
    inverse = 1.0 / float(np.asarray(dispersion))
    analytic_gradient = float(np.asarray(count / mean)) - (
        float(np.asarray(count)) + inverse
    ) / (float(np.asarray(mean)) + inverse)
    gradient_error = abs(automatic_gradient - analytic_gradient)
    method = {
        "method_name": "negative-binomial-NB2-log-probability",
        "method_kind": "exact_model",
        "execution_kind": "floating_point_direct",
        "differentiation_kind": "exact_ad",
        "variance_law": "mean + dispersion * mean^2",
        "count_unit": "molecule_or_read_count",
        "mean_unit": "expected_count_in_the_same_experimental_unit",
    }
    method_fingerprint = fingerprint(method)
    inputs = {
        "counts": counts,
        "means": means,
        "dispersion": dispersion,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "closed-form NB2 log probability",
        "gradient_oracle": "y/mu - (y + 1/alpha)/(mu + 1/alpha)",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": method_fingerprint,
        "method": {"fingerprint": method_fingerprint, **method},
        "maximum_log_probability_error": probability_error,
        "automatic_mean_gradient": automatic_gradient,
        "analytic_mean_gradient": analytic_gradient,
        "absolute_gradient_error": gradient_error,
        "passed": probability_error <= 3.0e-6 and gradient_error <= 3.0e-6,
    }


def _multiple_testing_case() -> dict[str, object]:
    p_values = jnp.asarray((0.01, 0.04, 0.03, 0.20))
    family = jnp.asarray((True, True, True, True))
    result = omics.benjamini_hochberg(p_values, family)
    expected = np.asarray((0.04, 0.05333333333333334, 0.05333333333333334, 0.20))
    adjustment_error = float(
        np.max(np.abs(np.asarray(result.adjusted_p_values) - expected))
    )
    invalid = omics.benjamini_hochberg(
        jnp.asarray((0.01, 1.20)), jnp.asarray((True, True))
    )
    invalid_rejected = (
        not bool(np.asarray(invalid.valid)) and int(np.asarray(invalid.status)) != 0
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {"p_values": p_values, "tested_family": family}
    return {
        "scope": "unit_qualification",
        "oracle": "ranked Benjamini-Hochberg step-up adjustment",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "observed_adjusted_p_values": np.asarray(result.adjusted_p_values).tolist(),
        "expected_adjusted_p_values": expected.tolist(),
        "maximum_adjustment_error": adjustment_error,
        "family_size": int(np.asarray(result.family_size)),
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "invalid_input_status_check": {
            "status": int(np.asarray(invalid.status)),
            "rejected": invalid_rejected,
        },
        "passed": bool(
            np.asarray(result.valid) and adjustment_error <= 2.0e-7 and invalid_rejected
        ),
    }


def qualification(
    *,
    omics_standard_root: Path | None = None,
    omics_standard_sha256: str | None = None,
) -> dict[str, object]:
    campaigns = {
        "omics_standard": external_dataset_campaign(
            "omics-standard",
            omics_standard_root,
            omics_standard_sha256,
        )
    }
    return qualification_report(
        "omics_statistics",
        {
            "negative_binomial_nb2": _negative_binomial_case(),
            "benjamini_hochberg": _multiple_testing_case(),
        },
        external_campaigns=campaigns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify public omics statistical APIs; external standard roots are "
            "opt-in and never downloaded."
        )
    )
    parser.add_argument("--omics-standard-root", type=Path)
    parser.add_argument("--omics-standard-sha256")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification(
        omics_standard_root=arguments.omics_standard_root,
        omics_standard_sha256=arguments.omics_standard_sha256,
    )
    return emit_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
