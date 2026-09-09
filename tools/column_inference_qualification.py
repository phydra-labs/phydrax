# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Runnable local-Gaussian and physical-column inverse-experiment qualification.

JAX_ENABLE_X64=1 PYTHONPATH=. python tools/column_inference_qualification.py
Synthetic evidence qualifies implementation, NOT Earth prediction or coverage.
"""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

from examples.interactive_column_inference import run_twin
from phydrax.applications.geophysics._inference import column_local_information


def qualify(*, steps=12, maximum_steps=24):
    linear = column_local_information(
        lambda z: jnp.asarray([2 * z[0], 3 * z[1]]), jnp.ones(2)
    )
    np.testing.assert_allclose(linear.fisher, np.diag([4.0, 9.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        linear.covariance_on_identifiable_subspace,
        np.diag([0.25, 1 / 9]),
        rtol=1e-12,
        atol=1e-12,
    )
    confounded = column_local_information(
        lambda z: jnp.atleast_1d(z[0] + z[1]), jnp.ones(2)
    )
    if int(confounded.rank) != 1:
        raise AssertionError("Single sum must diagnose a one-dimensional nullspace.")
    np.testing.assert_allclose(
        confounded.unidentifiable_projector @ jnp.asarray([1.0, -1.0]),
        [1.0, -1.0],
        atol=1e-12,
    )
    summary = run_twin(steps=steps, maximum_steps=maximum_steps)
    required = (
        summary["successful"],
        summary["bounds_respected"],
        summary["derivative_valid"],
        summary["confounded_rank"] == 1,
        summary["complementary_rank"] == len(summary["parameters"]),
        summary["chosen_candidate"] >= 0,
        summary["continuation_max_error"] == 0.0,
    )
    if not all(required):
        raise AssertionError(json.dumps(summary, indent=2))
    audit = summary["gradient_audit"]
    if not (
        audit["center_valid"]
        and all(audit["stencil_valid"])
        and all(audit["refinement_valid"])
    ):
        raise AssertionError(
            "Gradient audit crossed a phase/admission boundary; evidence is invalid."
        )
    if min(audit["relative_errors"]) > 2e-5:
        raise AssertionError(
            f"JVP/real perturbation disagreement: {audit['relative_errors']}"
        )
    coarse, fine = audit["dt_relative_changes"]
    if fine > 0.8 * coarse + 1e-7:
        raise AssertionError(
            f"First-order gradient refinement did not improve: {coarse}, {fine}"
        )
    for name, score in summary["heldout"].items():
        if not (score["physical_successful"] and score["uncertainty_valid"]):
            raise AssertionError(
                f"Invalid withheld {name} physical/uncertainty response."
            )
        if score["noise_normalized_rmse"] > 0.05:
            raise AssertionError(
                f"Exact-mean twin failed withheld {name} response: {score['noise_normalized_rmse']}"
            )
    summary["analytic_gaussian_limit"] = {
        "fisher": np.asarray(linear.fisher).tolist(),
        "covariance": np.asarray(linear.covariance_on_identifiable_subspace).tolist(),
        "confounded_rank": int(confounded.rank),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--maximum-steps", type=int, default=24)
    args = parser.parse_args()
    print(
        json.dumps(qualify(steps=args.steps, maximum_steps=args.maximum_steps), indent=2)
    )
