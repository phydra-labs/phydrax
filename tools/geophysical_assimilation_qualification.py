# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run the geophysical linear-Gaussian and labelled heat-reservoir qualifications.

PYTHONPATH=. python tools/geophysical_assimilation_qualification.py
"""

from __future__ import annotations

import json

import jax
import numpy as np

from examples.geophysical_assimilation import make_twin_problem, run_twin
from phydrax.uq import ensemble_filter_step, initialize_ensemble_filter


def qualify_linear_gaussian_limit():
    problem, _, _, _ = make_twin_problem()
    state = initialize_ensemble_filter(jax.random.key(42), problem, ensemble_size=16)
    ensemble = np.asarray(state.ensemble)
    mean = ensemble.mean(axis=0)
    covariance = np.cov(ensemble, rowvar=False)
    error = np.eye(2) * (0.05**2 + 0.03**2)
    gain = np.linalg.solve(covariance + error, covariance).T
    target = np.asarray(problem.observations.values[0])
    expected_mean = mean + gain @ (target - mean)
    expected_covariance = covariance - gain @ covariance
    _, record = ensemble_filter_step(problem, state)
    actual = np.asarray(record.analysis_ensemble)
    mean_error = float(np.max(np.abs(actual.mean(axis=0) - expected_mean)))
    covariance_error = float(
        np.max(np.abs(np.cov(actual, rowvar=False) - expected_covariance))
    )
    return {
        "mean_max_error_K": mean_error,
        "covariance_max_error_K2": covariance_error,
        "passed": mean_error < 1.0e-10 and covariance_error < 1.0e-10,
    }


def qualify():
    linear = qualify_linear_gaussian_limit()
    twin = run_twin()
    twin_passed = (
        twin["analysis_rmse_K"] < 0.15
        and twin["analysis_rmse_K"] < 0.1 * twin["unassimilated_rmse_K"]
        and twin["two_day_forecast_rmse_K"] < 0.2
        and twin["initial_analysis_spread_K2"] < 0.1 * twin["initial_forecast_spread_K2"]
        and twin["budget_relative_residual"] < 1.0e-11
        and twin["restart_bitwise_equal"]
        and twin["restart_key_equal"]
        and twin["smoother_valid"]
        and twin["observed_counts"] == [2, 1, 2, 1, 2, 1, 2, 1]
    )
    return {
        "linear_gaussian": linear,
        "physical_twin": twin,
        "passed": bool(linear["passed"] and twin_passed),
    }


if __name__ == "__main__":
    with jax.enable_x64(True):
        report = qualify()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
