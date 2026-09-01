#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run deterministic mathematical qualification checks for bioinformatics kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def run_qualification(
    *,
    external_file: Path | None = None,
    external_sha256: str | None = None,
) -> dict:
    table = phx.bioinformatics.sequence.nucleotide_substitution_table(
        match_score=2.0,
        mismatch_score=-3.0,
    )
    alignment = phx.bioinformatics.sequence.align_affine(
        jnp.asarray((0, 1, 2, 3), dtype=jnp.int32),
        jnp.asarray((0, 2, 3), dtype=jnp.int32),
        table,
        phx.bioinformatics.sequence.AffineGapPenalties(-4.0, -1.0),
        phx.bioinformatics.sequence.AlignmentExecutionPlan.full(
            4,
            3,
            traceback_capacity=7,
        ),
    )

    substitution = phx.bioinformatics.phylogenetics.jc69(dtype=jnp.float64)
    transition = substitution.transition_matrix(0.7)

    rna_model = phx.bioinformatics.rna.nussinov_energy_model(
        pair_energy=-1.0,
        minimum_hairpin_length=0,
    )
    rna = phx.bioinformatics.rna.partition_function(
        jnp.asarray((0, 3, 2, 1), dtype=jnp.int32),
        rna_model,
    )

    adjusted = phx.bioinformatics.omics.benjamini_hochberg(
        jnp.asarray((0.001, 0.02, 0.8), dtype=jnp.float64),
        jnp.ones((3,), dtype=bool),
    )

    checks = {
        "alignment_valid": bool(alignment.valid),
        "alignment_traceback_score_matches": bool(
            jnp.isclose(alignment.score, alignment.traceback_score)
        ),
        "phylogenetic_transition_nonnegative": bool(jnp.all(transition >= 0.0)),
        "phylogenetic_transition_rows_normalized": bool(
            jnp.allclose(jnp.sum(transition, axis=-1), 1.0, atol=2e-10)
        ),
        "rna_partition_valid": bool(rna.valid),
        "rna_pair_marginals_symmetric": bool(
            jnp.allclose(rna.pair_marginals, rna.pair_marginals.T, atol=1e-8)
        ),
        "rna_per_residue_pair_mass_bounded": bool(
            jnp.all(jnp.sum(rna.pair_marginals, axis=-1) <= 1.0 + 1e-7)
        ),
        "multiple_testing_valid": bool(adjusted.valid),
        "multiple_testing_adjusted_not_smaller": bool(
            jnp.all(adjusted.adjusted_p_values >= adjusted.raw_p_values)
        ),
    }

    external = None
    if external_file is not None:
        if not external_file.is_file():
            raise ValueError("external_file must name an existing regular file.")
        observed = _sha256(external_file)
        expected = None if external_sha256 is None else external_sha256.lower()
        external = {
            "path": str(external_file),
            "sha256": observed,
            "expected_sha256": expected,
            "digest_matches": expected is None or observed == expected,
        }
        checks["external_digest_matches"] = bool(external["digest_matches"])
    elif external_sha256 is not None:
        raise ValueError("external_sha256 requires external_file.")

    return {
        "checks": checks,
        "external_input": external,
        "jax_version": jax.__version__,
        "method_contract_ids": {
            "alignment": alignment.method_contract.contract_id,
            "rna": rna.method_contract.contract_id,
        },
        "passed": all(checks.values()),
        "source_sha256": _sha256(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-file", type=Path)
    parser.add_argument("--external-sha256")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_qualification(
        external_file=arguments.external_file,
        external_sha256=arguments.external_sha256,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
