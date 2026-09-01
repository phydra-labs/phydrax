#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small-variant normalization and germline genotyping qualification."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import genomics
from tools.bioinformatics_common_qualification import (
    emit_report,
    external_dataset_campaign,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _variant_normalization_case() -> dict[str, object]:
    reference = "CAAAC"
    result = genomics.normalize_small_variant(
        reference,
        2,
        "AA",
        ("A",),
        max_alleles=2,
        max_allele_length=2,
    )
    alleles = genomics.decode_variant_alleles(result.site)
    observed_position = int(np.asarray(result.site.position))

    insufficient = genomics.normalize_small_variant(
        reference,
        2,
        "AA",
        ("A",),
        max_alleles=2,
        max_allele_length=1,
    )
    capacity_rejected = (
        not bool(np.asarray(insufficient.valid))
        and int(np.asarray(insufficient.status)) != 0
        and int(np.asarray(insufficient.evidence.required_max_allele_length)) == 2
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "reference": reference,
        "position": 2,
        "reference_allele": "AA",
        "alternate_alleles": ["A"],
        "max_alleles": 2,
        "max_allele_length": 2,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "minimal left-aligned deletion in a tandem repeat",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "observed_position": observed_position,
        "expected_position": 0,
        "observed_alleles": list(alleles),
        "expected_alleles": ["CA", "C"],
        "left_shift": int(np.asarray(result.evidence.left_shift)),
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "capacity_check": {
            "configured_max_allele_length": 1,
            "required_max_allele_length": int(
                np.asarray(insufficient.evidence.required_max_allele_length)
            ),
            "status": int(np.asarray(insufficient.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and observed_position == 0
            and alleles == ("CA", "C")
            and capacity_rejected
        ),
    }


def _genotype_case() -> dict[str, object]:
    state_space = genomics.enumerate_genotype_states(2, 2, 3)
    log_allele_likelihoods = jnp.log(
        jnp.asarray(
            (
                (0.99, 0.01),
                (0.80, 0.20),
                (0.10, 0.90),
            )
        )
    )
    evidence = genomics.local_haplotype_evidence(
        log_allele_likelihoods,
        jnp.asarray((True, True, True)),
    )
    likelihoods = genomics.genotype_likelihoods_from_reads(evidence, state_space)
    states = np.asarray(state_space.states)[np.asarray(state_space.state_mask)]
    allele_probabilities = np.exp(np.asarray(log_allele_likelihoods))
    brute_values = []
    for first, second in states:
        per_read = 0.5 * (
            allele_probabilities[:, first] + allele_probabilities[:, second]
        )
        brute_values.append(float(np.sum(np.log(per_read))))
    brute = np.asarray(brute_values)
    brute -= np.max(brute)
    observed = np.asarray(likelihoods.log_likelihoods)[np.asarray(likelihoods.state_mask)]
    likelihood_error = float(np.max(np.abs(observed - brute)))

    prior = genomics.uniform_genotype_prior(state_space)
    gl = jnp.asarray((-3.0, -1.0, 0.0))

    def log_evidence(gl_values):
        supplied = genomics.genotype_likelihoods_from_gl(gl_values, state_space, depth=3)
        inferred = genomics.infer_genotype(
            supplied,
            prior,
            state_space,
            min_depth=0,
            min_posterior=0.0,
        )
        return inferred.posterior.evidence.log_evidence

    supplied = genomics.genotype_likelihoods_from_gl(gl, state_space, depth=3)
    inferred = genomics.infer_genotype(
        supplied,
        prior,
        state_space,
        min_depth=0,
        min_posterior=0.0,
    )
    automatic_gradient = jax.grad(log_evidence)(gl)
    maximum_index = int(np.argmax(np.asarray(gl)))
    maximum_basis = jax.nn.one_hot(maximum_index, gl.shape[0])
    analytic_gradient = math.log(10.0) * (
        inferred.posterior.probabilities - maximum_basis
    )
    gradient_error = float(
        np.max(np.abs(np.asarray(automatic_gradient - analytic_gradient)))
    )

    insufficient = genomics.enumerate_genotype_states(2, 2, 2)
    capacity_rejected = (
        not bool(np.asarray(insufficient.valid))
        and int(np.asarray(insufficient.status)) != 0
        and int(np.asarray(insufficient.evidence.required_state_count)) == 3
    )
    contract = method_contract_evidence(likelihoods.method_contract)
    inputs = {
        "allele_log_likelihoods": log_allele_likelihoods,
        "allele_count": 2,
        "ploidy": 2,
        "genotype_capacity": 3,
        "gradient_gl": gl,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "explicit diploid read-by-genotype likelihood enumeration",
        "gradient_oracle": (
            "gradient of max-normalized GL evidence equals ln(10) times "
            "posterior minus the unique maximizing basis vector"
        ),
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "maximum_likelihood_error": likelihood_error,
        "maximum_gradient_identity_error": gradient_error,
        "posterior_probability_sum": float(
            np.sum(np.asarray(inferred.posterior.probabilities))
        ),
        "status": int(np.asarray(likelihoods.status)),
        "valid": bool(np.asarray(likelihoods.valid)),
        "capacity_check": {
            "configured_genotype_capacity": 2,
            "required_genotype_capacity": int(
                np.asarray(insufficient.evidence.required_state_count)
            ),
            "status": int(np.asarray(insufficient.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(likelihoods.valid)
            and np.asarray(inferred.valid)
            and likelihood_error <= 2.0e-6
            and gradient_error <= 2.0e-5
            and capacity_rejected
        ),
    }


def qualification(
    *,
    giab_root: Path | None = None,
    giab_sha256: str | None = None,
    cami_root: Path | None = None,
    cami_sha256: str | None = None,
) -> dict[str, object]:
    campaigns = {
        "giab": external_dataset_campaign("GIAB", giab_root, giab_sha256),
        "cami": external_dataset_campaign("CAMI", cami_root, cami_sha256),
    }
    return qualification_report(
        "genomics_variant",
        {
            "small_variant_normalization": _variant_normalization_case(),
            "diploid_genotype_likelihood": _genotype_case(),
        },
        external_campaigns=campaigns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify public genomics and small-variant APIs; external GIAB/CAMI "
            "roots are opt-in and never downloaded."
        )
    )
    parser.add_argument("--giab-root", type=Path)
    parser.add_argument("--giab-sha256")
    parser.add_argument("--cami-root", type=Path)
    parser.add_argument("--cami-sha256")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification(
        giab_root=arguments.giab_root,
        giab_sha256=arguments.giab_sha256,
        cami_root=arguments.cami_root,
        cami_sha256=arguments.cami_sha256,
    )
    return emit_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
