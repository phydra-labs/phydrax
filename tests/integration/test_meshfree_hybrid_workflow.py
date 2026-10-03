# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hybrid point-cloud / finite-volume overlap solve and its calibrated bands.

One module-scoped run of ``examples/meshfree_hybrid_calibrated.py``. The oracle
is the closed-form manufactured family (sources from exact automatic
derivatives) and a monolithic point-cloud solve on the whole square; neither
depends on the overlap law under test.
"""

from __future__ import annotations

import pytest

from examples.meshfree_hybrid_calibrated import (
    finite_sample_band,
    HybridWorkflowReport,
    run_workflow,
)


@pytest.fixture(scope="module")
def workflow() -> HybridWorkflowReport:
    return run_workflow(
        levels=(16, 24, 32),
        calibration_cells=16,
        train_cases=6,
        calibration_cases=19,
        test_cases=20,
        shift_cases=10,
        alpha=0.1,
        seed=0,
    )


def test_hybrid_overlap_converges_to_the_original_problem(
    workflow: HybridWorkflowReport,
) -> None:
    refinement = workflow["refinement"]
    levels = refinement["levels"]
    for name, errors in (
        ("cloud_max_error", [level["cloud_max_error"] for level in levels]),
        ("grid_max_error", [level["grid_max_error"] for level in levels]),
        (
            "monolithic_meshfree_max_error",
            [level["monolithic_meshfree_max_error"] for level in levels],
        ),
    ):
        assert all(later < earlier for earlier, later in zip(errors, errors[1:])), name
    # The composite scheme is second order; the observed rates are pre-asymptotic.
    for name, rates in (
        ("cloud_rates", refinement["cloud_rates"]),
        ("grid_rates", refinement["grid_rates"]),
    ):
        assert min(rates) > 1.0, name
    finest = levels[-1]
    assert finest["cloud_max_error"] < 4.0e-3
    assert finest["grid_max_error"] < 5.0e-3


def test_overlap_certificates_and_transfers(workflow: HybridWorkflowReport) -> None:
    refinement = workflow["refinement"]
    mismatch = refinement["overlap_mismatch_max"]
    assert all(later < earlier for earlier, later in zip(mismatch, mismatch[1:]))
    for level in refinement["levels"]:
        defects = level["interface_defects"]
        gated = level["interface_gated"]
        assert gated == {
            "artificial-node-transfer": True,
            "overlap-mismatch-max": False,
            "overlap-mismatch-l2": False,
        }
        assert defects["artificial-node-transfer"] < 1.0e-8
        assert level["node_identity_defect"] == 0.0
        assert level["node_load_defect"] == 0.0
        assert level["overlap_gap"] > 0.1
        for transfer in level["transfers"]:
            assert transfer["refused_rows"] == 0
            assert transfer["constant_defect"] < 1.0e-9
            assert transfer["linear_defect"] < 1.0e-9
        for subdomain in level["schwarz_subdomains"]:
            assert subdomain["block_relative_error"] < 1.0e-12


def test_schwarz_preconditioning_bounds_krylov_work(
    workflow: HybridWorkflowReport,
) -> None:
    levels = workflow["refinement"]["levels"]
    multiplicative: list[float] = []
    for level in levels:
        solves = level["solves"]
        additive = solves["additive-schwarz"]
        alternating = solves["multiplicative-schwarz"]
        baseline = solves["unpreconditioned"]
        for record in (additive, alternating):
            assert record["native_successful"] is True
            assert record["accepted"] is True
        assert alternating["iterations"] < additive["iterations"]
        assert (
            baseline["accepted"] is False
            or baseline["iterations"] > 10 * alternating["iterations"]
        )
        multiplicative.append(alternating["iterations"])
    # Exact subdomain solves over a fixed physical overlap: mesh-independent work.
    assert max(multiplicative) <= 2.0 * min(multiplicative)


def test_in_distribution_coverage_on_disjoint_test_cases(
    workflow: HybridWorkflowReport,
) -> None:
    calibration = workflow["calibration"]
    split = calibration["split"]
    assert split == {"train": 6, "calibration": 19, "test": 20, "disjoint": True}
    assert calibration["all_cases_accepted"] is True
    nominal = calibration["nominal_coverage"]
    assert nominal == pytest.approx(0.9)
    measured = calibration["in_distribution"]
    mean, deviation = finite_sample_band(19, 20, 0.1)
    coverage = measured["measured_simultaneous_coverage"]
    assert coverage >= mean - 3.0 * deviation
    assert 0.0 < measured["mean_width"] < 1.0


def test_shifted_sets_are_reported_separately_without_guarantee(
    workflow: HybridWorkflowReport,
) -> None:
    calibration = workflow["calibration"]
    shifted = calibration["shifted"]
    assert set(shifted) == {"regime", "geometry"}
    for kind, entry in shifted.items():
        assert entry["distribution_free_guarantee"] is False
        assert entry["cases"] == 10
        assert entry["all_cases_accepted"] is True
        assert 0.0 <= entry["measured_simultaneous_coverage"] <= 1.0
        assert entry["label"] != calibration["in_distribution"]["label"], kind
