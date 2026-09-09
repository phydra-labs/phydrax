# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import numpy as np

from tools.global_feedback_qualification import (
    block_statistics,
    compare,
    deterministic_equilibrium_assessment,
    REQUIRED_SENSITIVITIES,
    RESPONSE_TOLERANCES,
)


def test_spinup_and_incomplete_tail_do_not_bias_complete_block_means():
    days = np.arange(1.0, 33.0)
    values = np.full_like(days, 2.0)
    values[:10], values[30:] = 100.0, 1000.0
    result = block_statistics(days, values, spinup_days=10.0, block_days=5.0)
    assert result["complete_blocks"] == 4
    assert result["mean"] == 2.0
    assert not result["sampling_adequate"]
    assert not result["stationarity_consistent"]


def test_long_correlated_drift_is_not_stationary_despite_many_samples():
    days = np.arange(1.0, 121.0)
    result = block_statistics(days, 280.0 + days / 100, spinup_days=20.0, block_days=10.0)
    assert result["complete_blocks"] == 10
    assert result["lag1_block_correlation"] > 0.3
    assert not result["sampling_adequate"]
    assert not result["stationarity_consistent"]


def test_two_steps_do_not_produce_a_climate_confidence_interval():
    result = block_statistics(
        [20 / 86400, 40 / 86400], [290.0, 290.001], spinup_days=0.0, block_days=30.0
    )
    assert result["mean"] is None
    assert result["ci95_half_width"] is None
    assert not result["sampling_adequate"]
    assert not result["stationarity_consistent"]


def _response_record(
    scenario,
    *,
    offset=0.0,
    response_scale=1.0,
    uncertainty_scale=0.001,
    absolute_filter_work=0.01,
):
    """Numerical inputs to the report gate, not simulated climate evidence."""
    return {
        "scenario": scenario,
        "stationary_sample_qualified": True,
        "production_filter_work_w_m2": 0.0,
        "production_absolute_filter_work_w_m2": absolute_filter_work,
        "statistics": {
            key: {
                "mean": tolerance
                * (offset + (10 * response_scale if scenario != "baseline" else 0)),
                "ci95_half_width": tolerance * uncertainty_scale,
            }
            for key, tolerance in RESPONSE_TOLERANCES.items()
        },
    }


def test_statistical_response_requires_all_paired_numerical_sensitivities():
    baseline, forced = _response_record("baseline"), _response_record("solar")
    without = compare(baseline, forced)
    assert without["sst_k"]["statistically_resolved_difference"]
    assert not without["sst_k"]["response_claim_supported"]
    # Large common shifts of the reference climate cancel in a robust paired
    # response: compare perturbed response, not the baseline field difference.
    pairs = {
        name: (
            _response_record("baseline", offset=50),
            _response_record("solar", offset=50),
        )
        for name in REQUIRED_SENSITIVITIES
    }
    complete = compare(baseline, forced, sensitivity_pairs=pairs)
    assert complete["sst_k"]["response_claim_supported"]
    missing = {key: value for key, value in pairs.items() if key != "higher_resolution"}
    assert not compare(baseline, forced, sensitivity_pairs=missing)["sst_k"][
        "response_claim_supported"
    ]


def test_changed_response_or_unresolved_sensitivity_blocks_physical_claim():
    baseline, forced = _response_record("baseline"), _response_record("solar")
    pairs = {name: (baseline, forced) for name in REQUIRED_SENSITIVITIES}
    pairs["half_dt"] = (baseline, _response_record("solar", response_scale=1.2))
    shifted = compare(baseline, forced, sensitivity_pairs=pairs)["sst_k"]
    assert shifted["statistically_resolved_difference"]
    assert not shifted["response_claim_supported"]
    # A broad interval cannot be used to excuse a large numerical uncertainty.
    pairs["half_dt"] = (baseline, _response_record("solar", uncertainty_scale=2.0))
    uncertain = compare(baseline, forced, sensitivity_pairs=pairs)["sst_k"]
    assert not uncertain["response_claim_supported"]


def test_filter_work_budget_cannot_disappear_in_zero_signed_accounting():
    baseline = _response_record("baseline")
    forced = _response_record("solar", absolute_filter_work=0.2)
    pairs = {name: (baseline, forced) for name in REQUIRED_SENSITIVITIES}
    result = compare(baseline, forced, sensitivity_pairs=pairs)["sst_k"]
    assert result["statistically_resolved_difference"]
    assert (
        result["reference_filter_work_budget"]["forced_minus_baseline_work_w_m2"] == 0.0
    )
    assert not result["reference_filter_work_budget"]["within_declared_budget"]
    assert not result["response_claim_supported"]


def test_deterministic_equilibrium_does_not_fabricate_independent_samples():
    changes = {
        "sst_k": 0.08,
        "atmosphere_temperature_k": 0.11,
        "eddy_kinetic_energy_m2_s2": 1e-12,
        "precipitation_kg_m2_s": 6e-8,
        "toa_net_in_w_m2": -0.33,
        "surface_net_in_w_m2": -0.12,
    }
    statistics = {
        key: {
            "half_record_change": value,
            "lag1_block_correlation": 0.99,
            "sampling_adequate": False,
            "stationarity_consistent": False,
        }
        for key, value in changes.items()
    }
    last = {"toa_net_in_w_m2": -0.075, "surface_net_in_w_m2": -0.014}
    result = deterministic_equilibrium_assessment(
        statistics,
        last,
        numerical_accounting=True,
        radiative_equilibrium=True,
        duration_adequate=True,
        temperature_change_k=0.2,
        flux_change_w_m2=0.5,
        endpoint_flux_w_m2=0.1,
        eke_change_m2_s2=0.01,
        precipitation_change_kg_m2_s=1e-7,
    )
    assert result["qualified"]
    assert "independent sample" in result["scope"]
    drifting = {
        **statistics,
        "sst_k": {**statistics["sst_k"], "half_record_change": 0.3},
    }
    assert not deterministic_equilibrium_assessment(
        drifting,
        last,
        numerical_accounting=True,
        radiative_equilibrium=True,
        duration_adequate=True,
        temperature_change_k=0.2,
        flux_change_w_m2=0.5,
        endpoint_flux_w_m2=0.1,
        eke_change_m2_s2=0.01,
        precipitation_change_kg_m2_s=1e-7,
    )["qualified"]
