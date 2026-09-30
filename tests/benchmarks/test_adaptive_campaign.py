#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_adaptive_campaign import analyze_rows, load_rows


def _row(
    *,
    regime: str = "low",
    method: str = "fixed_sobol",
    seed: int = 0,
    error: float = 1.0,
    residual: float = 1.0,
    work: int = 100,
) -> dict[str, object]:
    return {
        "regime": regime,
        "problem": "helmholtz_k7p5",
        "method": method,
        "seed": seed,
        "relative_l2_u": error,
        "validation_residual_rms": residual,
        "final_training_loss": 0.1,
        "wall_time_seconds": 2.0,
        "max_abs_u": 1.5,
        "estimated_total_residual_evaluations": work,
    }


def _promotion_rows(
    *, error_ratio: float = 0.8, residual_ratio: float = 1.05
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for regime in ("low", "medium", "convergence"):
        for seed in range(10):
            rows.append(_row(regime=regime, seed=seed))
            rows.append(
                _row(
                    regime=regime,
                    method="rar_d",
                    seed=seed,
                    error=error_ratio,
                    residual=residual_ratio,
                )
            )
    return rows


def test_analyzer_combines_disjoint_campaign_shards(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "rows.jsonl").write_text(json.dumps(_row(seed=0, error=0.2)) + "\n")
    (second / "rows.jsonl").write_text(json.dumps(_row(seed=1, error=0.4)) + "\n")

    report = analyze_rows(load_rows((first, second)), expected_seeds=2)

    assert report.integrity.valid
    summary = report.problem_summary["low/helmholtz_k7p5"][0]
    assert summary.seeds == 2
    assert summary.relative_l2_mean == pytest.approx(0.3)
    assert summary.relative_l2_std == pytest.approx(0.1)


def test_analyzer_refuses_repeated_shard_paths(tmp_path: Path) -> None:
    (tmp_path / "rows.jsonl").write_text(json.dumps(_row()) + "\n")
    with pytest.raises(ValueError, match="Repeated campaign shard"):
        load_rows((tmp_path, tmp_path / "."))


def test_integrity_distinguishes_regime_and_allocation_evidence() -> None:
    rows = [_row(regime="low"), _row(regime="medium"), {"allocated_points": 256}]
    report = analyze_rows(rows, expected_seeds=1)

    assert report.integrity.valid
    assert report.integrity.scientific_rows == 2
    assert report.integrity.allocation_only_rows == 1
    assert set(report.problem_summary) == {"low/helmholtz_k7p5", "medium/helmholtz_k7p5"}
    assert report.promotion_summary["rar_d"].decision == "insufficient_evidence"


def test_allocation_only_rows_cannot_establish_scientific_integrity() -> None:
    report = analyze_rows([{"allocated_points": 256}], expected_seeds=1)
    assert not report.integrity.valid
    assert report.problem_summary == {}
    assert report.promotion_summary["rar_d"].decision == "insufficient_evidence"


def test_integrity_reports_duplicate_and_incomplete_seed_groups() -> None:
    report = analyze_rows([_row(seed=0), _row(seed=0)], expected_seeds=3)
    assert not report.integrity.valid
    assert report.integrity.duplicate_seed_groups[0].rows == 2
    assert report.integrity.incomplete_groups[0].rows == 2


@pytest.mark.parametrize(
    "error", [float("nan"), float("inf"), -0.1], ids=["nan", "infinite", "negative"]
)
def test_invalid_scientific_error_is_not_ranked(error: float) -> None:
    report = analyze_rows([_row(error=error)], expected_seeds=1)
    assert not report.integrity.valid
    assert report.integrity.invalid_rows[0].seed == 0
    assert report.problem_summary == {}
    assert report.method_summary == []


def test_work_accounting_preserves_operator_orders_and_fractional_medians() -> None:
    first = _row(seed=0, work=100)
    first.update(
        first_derivative_residual_evaluations=40,
        second_derivative_residual_evaluations=100,
    )
    second = _row(seed=1, work=101)
    second.update(
        first_derivative_residual_evaluations=41,
        second_derivative_residual_evaluations=101,
    )
    report = analyze_rows([first, second], expected_seeds=2)
    summary = report.problem_summary["low/helmholtz_k7p5"][0]
    assert summary.estimated_total_residual_evaluations == 100.5
    assert summary.first_derivative_residual_evaluations == 40.5
    assert summary.second_derivative_residual_evaluations == 100.5


def test_missing_derivative_work_remains_unknown_not_zero() -> None:
    first = _row(seed=0)
    first["first_derivative_residual_evaluations"] = 100
    report = analyze_rows([first, _row(seed=1)], expected_seeds=2)
    summary = report.problem_summary["low/helmholtz_k7p5"][0]
    assert summary.first_derivative_residual_evaluations is None
    assert summary.second_derivative_residual_evaluations is None


@pytest.mark.parametrize(
    "work", [-1, 1.5, True], ids=["negative", "fractional", "boolean"]
)
def test_work_accounting_refuses_invalid_counts(work: object) -> None:
    row = _row()
    row["estimated_total_residual_evaluations"] = work
    with pytest.raises(
        (TypeError, ValueError), match="estimated_total_residual_evaluations"
    ):
        analyze_rows([row], expected_seeds=1)


@pytest.mark.parametrize(
    ("error_ratio", "residual_ratio", "decision"),
    [
        (0.8, 1.05, "promote"),
        (0.95, 1.05, "retain_conditional"),
        (0.8, 1.2, "retain_conditional"),
    ],
    ids=[
        "improves-error-and-preserves-residual",
        "insufficient-error-gain",
        "degrades-residual",
    ],
)
def test_promotion_requires_error_gain_and_residual_quality(
    error_ratio: float, residual_ratio: float, decision: str
) -> None:
    report = analyze_rows(
        _promotion_rows(error_ratio=error_ratio, residual_ratio=residual_ratio),
        expected_seeds=10,
    )
    promotion = report.promotion_summary["rar_d"]
    assert promotion.decision == decision
    assert all(evidence.paired_seeds == 10 for evidence in promotion.regimes.values())


def test_promotion_requires_evidence_in_every_regime() -> None:
    rows = [row for row in _promotion_rows() if row["regime"] != "convergence"]
    report = analyze_rows(rows, expected_seeds=10)
    promotion = report.promotion_summary["rar_d"]
    assert promotion.decision == "insufficient_evidence"
    assert promotion.regimes["low"].passes
    assert not promotion.regimes["convergence"].passes


def test_promotion_pairs_seed_identity_not_row_counts() -> None:
    rows = _promotion_rows()
    for row in rows:
        if row["method"] == "rar_d":
            seed = row["seed"]
            assert isinstance(seed, int)
            row["seed"] = seed + 10
    report = analyze_rows(rows, expected_seeds=10)
    assert report.integrity.valid
    promotion = report.promotion_summary["rar_d"]
    assert promotion.decision == "insufficient_evidence"
    assert promotion.regimes["low"].paired_seeds == 0


def test_promotion_refuses_duplicate_seed_evidence() -> None:
    rows = _promotion_rows()
    rows.append(_row(method="rar_d", error=0.8))
    report = analyze_rows(rows, expected_seeds=10)
    assert not report.integrity.valid
    assert report.promotion_summary["rar_d"].decision != "promote"
    assert not report.promotion_summary["rar_d"].regimes["low"].passes


def test_promotion_refuses_invalid_evidence_even_with_ten_paired_seeds() -> None:
    rows = _promotion_rows()
    rows[1]["relative_l2_u"] = float("nan")
    report = analyze_rows(rows, expected_seeds=10)
    assert not report.integrity.valid
    assert report.promotion_summary["rar_d"].decision != "promote"


def test_promotion_does_not_claim_gain_against_exact_zero_error() -> None:
    rows = _promotion_rows()
    rows[0]["relative_l2_u"] = 0.0
    rows[1]["relative_l2_u"] = 0.0
    report = analyze_rows(rows, expected_seeds=10)
    assert report.promotion_summary["rar_d"].decision != "promote"


@pytest.mark.parametrize(
    ("winning_seeds", "decision"),
    [(6, "retain_conditional"), (7, "promote")],
    ids=["below-win-fraction", "at-win-fraction"],
)
def test_promotion_requires_seventy_percent_paired_wins(
    winning_seeds: int, decision: str
) -> None:
    rows = _promotion_rows(error_ratio=0.9)
    for row in rows:
        seed = row["seed"]
        assert isinstance(seed, int)
        if row["method"] == "rar_d" and seed >= winning_seeds:
            row["relative_l2_u"] = 1.1
    report = analyze_rows(rows, expected_seeds=10)
    assert report.promotion_summary["rar_d"].decision == decision
    assert (
        report.promotion_summary["rar_d"].regimes["low"].win_fraction
        == winning_seeds / 10
    )


def test_separable_promotion_uses_its_own_problem_and_baseline() -> None:
    rows = _promotion_rows()
    for row in rows:
        row["problem"] = "separable_poisson_2d"
        row["method"] = (
            "fixed_separable" if row["method"] == "fixed_sobol" else "hierarchical_axes"
        )
    report = analyze_rows(rows, expected_seeds=10)
    assert report.promotion_summary["hierarchical_axes"].decision == "promote"
    assert report.promotion_summary["rar_d"].decision == "insufficient_evidence"
