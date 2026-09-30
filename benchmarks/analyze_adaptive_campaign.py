#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


type ResultKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class ScientificRow:
    regime: str
    problem: str
    method: str
    seed: int
    relative_l2_u: float
    validation_residual_rms: float
    final_training_loss: float
    wall_time_seconds: float
    max_abs_u: float
    estimated_total_residual_evaluations: int
    relative_l2_k: float | None = None
    first_derivative_residual_evaluations: int | None = None
    second_derivative_residual_evaluations: int | None = None

    @property
    def key(self) -> ResultKey:
        return self.regime, self.problem, self.method

    @property
    def valid(self) -> bool:
        metrics = (
            self.relative_l2_u,
            self.validation_residual_rms,
            self.final_training_loss,
            self.wall_time_seconds,
            self.max_abs_u,
        )
        if self.relative_l2_k is not None:
            metrics += (self.relative_l2_k,)
        return all(math.isfinite(value) and value >= 0.0 for value in metrics)


@dataclass(frozen=True, slots=True)
class GroupEvidence:
    regime: str
    problem: str
    method: str
    rows: int


@dataclass(frozen=True, slots=True)
class InvalidRow:
    regime: str
    problem: str
    method: str
    seed: int


@dataclass(frozen=True, slots=True)
class Integrity:
    scientific_rows: int
    allocation_only_rows: int
    invalid_rows: tuple[InvalidRow, ...]
    incomplete_groups: tuple[GroupEvidence, ...]
    duplicate_seed_groups: tuple[GroupEvidence, ...]
    valid: bool


@dataclass(slots=True)
class ProblemSummary:
    regime: str
    problem: str
    method: str
    seeds: int
    relative_l2_mean: float
    relative_l2_median: float
    relative_l2_std: float
    max_abs_mean: float
    validation_residual_rms_mean: float
    wall_time_mean_seconds: float
    relative_l2_k_mean: float | None
    estimated_total_residual_evaluations: float
    first_derivative_residual_evaluations: float | None
    second_derivative_residual_evaluations: float | None
    rank: int = 0
    error_ratio_to_fixed: float | None = None


@dataclass(frozen=True, slots=True)
class MethodSummary:
    regime: str
    method: str
    problems: int
    mean_rank: float
    wins: int
    geometric_mean_error_ratio_to_fixed: float | None


@dataclass(frozen=True, slots=True)
class RegimeEvidence:
    paired_seeds: int
    passes: bool
    median_error_ratio: float | None = None
    win_fraction: float | None = None
    mean_residual_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class PromotionSummary:
    problem: str
    baseline: str
    decision: Literal["promote", "retain_conditional", "insufficient_evidence"]
    regimes: dict[str, RegimeEvidence]


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    integrity: Integrity
    problem_summary: dict[str, list[ProblemSummary]]
    method_summary: list[MethodSummary]
    promotion_summary: dict[str, PromotionSummary]


def load_rows(directories: Path | Sequence[Path]) -> list[dict[str, object]]:
    """Read every JSONL shard once in deterministic directory/file order."""
    paths = (directories,) if isinstance(directories, Path) else directories
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for directory in paths:
        for path in sorted(directory.glob("*.jsonl")):
            identity = path.resolve()
            if identity in seen:
                raise ValueError(f"Repeated campaign shard: {path}")
            seen.add(identity)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Campaign row in {path} must be a JSON object.")
                rows.append(value)
    return rows


def _text(row: Mapping[str, object], name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _number(row: Mapping[str, object], name: str) -> float:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    return float(value)


def _integer(row: Mapping[str, object], name: str) -> int:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _optional_work(row: Mapping[str, object], name: str) -> int | None:
    return None if row.get(name) is None else _integer(row, name)


def _scientific_row(row: Mapping[str, object]) -> ScientificRow:
    return ScientificRow(
        regime=_text(row, "regime"),
        problem=_text(row, "problem"),
        method=_text(row, "method"),
        seed=_integer(row, "seed"),
        relative_l2_u=_number(row, "relative_l2_u"),
        validation_residual_rms=_number(row, "validation_residual_rms"),
        final_training_loss=_number(row, "final_training_loss"),
        wall_time_seconds=_number(row, "wall_time_seconds"),
        max_abs_u=_number(row, "max_abs_u"),
        estimated_total_residual_evaluations=_integer(
            row, "estimated_total_residual_evaluations"
        ),
        relative_l2_k=(
            None if row.get("relative_l2_k") is None else _number(row, "relative_l2_k")
        ),
        first_derivative_residual_evaluations=_optional_work(
            row, "first_derivative_residual_evaluations"
        ),
        second_derivative_residual_evaluations=_optional_work(
            row, "second_derivative_residual_evaluations"
        ),
    )


def _work_median(values: Sequence[int | None]) -> float | None:
    # Unknown derivative work is not evidence of zero derivative work.
    known = [value for value in values if value is not None]
    return statistics.median(known) if len(known) == len(values) else None


def _summarize(key: ResultKey, rows: Sequence[ScientificRow]) -> ProblemSummary:
    errors = [row.relative_l2_u for row in rows]
    coefficient_errors = [
        row.relative_l2_k for row in rows if row.relative_l2_k is not None
    ]
    return ProblemSummary(
        *key,
        seeds=len(rows),
        relative_l2_mean=statistics.fmean(errors),
        relative_l2_median=statistics.median(errors),
        relative_l2_std=statistics.pstdev(errors),
        max_abs_mean=statistics.fmean(row.max_abs_u for row in rows),
        validation_residual_rms_mean=statistics.fmean(
            row.validation_residual_rms for row in rows
        ),
        wall_time_mean_seconds=statistics.fmean(row.wall_time_seconds for row in rows),
        relative_l2_k_mean=statistics.fmean(coefficient_errors)
        if coefficient_errors
        else None,
        estimated_total_residual_evaluations=statistics.median(
            row.estimated_total_residual_evaluations for row in rows
        ),
        first_derivative_residual_evaluations=_work_median(
            [row.first_derivative_residual_evaluations for row in rows]
        ),
        second_derivative_residual_evaluations=_work_median(
            [row.second_derivative_residual_evaluations for row in rows]
        ),
    )


def _method_summaries(
    problems: Mapping[str, list[ProblemSummary]],
) -> list[MethodSummary]:
    ranks: dict[tuple[str, str], list[int]] = defaultdict(list)
    wins: dict[tuple[str, str], int] = defaultdict(int)
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for summaries in problems.values():
        summaries.sort(key=lambda item: (item.relative_l2_median, item.method))
        baseline = next(
            (
                item.relative_l2_median
                for item in summaries
                if item.method in {"fixed_sobol", "fixed_separable"}
            ),
            None,
        )
        for rank, summary in enumerate(summaries, start=1):
            summary.rank = rank
            key = summary.regime, summary.method
            ranks[key].append(rank)
            wins[key] += rank == 1
            if baseline is not None and baseline > 0.0:
                summary.error_ratio_to_fixed = summary.relative_l2_median / baseline
                ratios[key].append(summary.error_ratio_to_fixed)
    result = [
        MethodSummary(
            regime=regime,
            method=method,
            problems=len(values),
            mean_rank=statistics.fmean(values),
            wins=wins[(regime, method)],
            geometric_mean_error_ratio_to_fixed=(
                0.0
                if 0.0 in ratios[(regime, method)]
                else math.exp(
                    statistics.fmean(
                        math.log(value) for value in ratios[(regime, method)]
                    )
                )
            )
            if ratios[(regime, method)]
            else None,
        )
        for (regime, method), values in sorted(ranks.items())
    ]
    return sorted(result, key=lambda item: (item.regime, item.mean_rank, item.method))


def _regime_evidence(
    method: Sequence[ScientificRow], baseline: Sequence[ScientificRow]
) -> RegimeEvidence:
    method_by_seed = {row.seed: row for row in method}
    baseline_by_seed = {row.seed: row for row in baseline}
    paired = sorted(method_by_seed.keys() & baseline_by_seed.keys())
    # Duplicate, failed, or zero-baseline evidence cannot establish promotion.
    if (
        len(method_by_seed) != len(method)
        or len(baseline_by_seed) != len(baseline)
        or any(not row.valid for row in (*method, *baseline))
        or any(
            baseline_by_seed[seed].relative_l2_u == 0.0
            or baseline_by_seed[seed].validation_residual_rms == 0.0
            for seed in paired
        )
    ):
        return RegimeEvidence(len(paired), False)
    if not paired:
        return RegimeEvidence(0, False)
    errors = [
        method_by_seed[seed].relative_l2_u / baseline_by_seed[seed].relative_l2_u
        for seed in paired
    ]
    residuals = [
        method_by_seed[seed].validation_residual_rms
        / baseline_by_seed[seed].validation_residual_rms
        for seed in paired
    ]
    median = statistics.median(errors)
    wins = statistics.fmean(ratio < 1.0 for ratio in errors)
    residual = statistics.fmean(residuals)
    return RegimeEvidence(
        len(paired),
        len(paired) >= 10 and median <= 0.9 and wins >= 0.7 and residual <= 1.1,
        median,
        wins,
        residual,
    )


def _promotions(
    groups: Mapping[ResultKey, list[ScientificRow]],
) -> dict[str, PromotionSummary]:
    result: dict[str, PromotionSummary] = {}
    for method, problem, baseline in (
        ("rar_d", "helmholtz_k7p5", "fixed_sobol"),
        ("hierarchical_axes", "separable_poisson_2d", "fixed_separable"),
    ):
        regimes = {
            regime: _regime_evidence(
                groups.get((regime, problem, method), []),
                groups.get((regime, problem, baseline), []),
            )
            for regime in ("low", "medium", "convergence")
        }
        complete = all(value.paired_seeds >= 10 for value in regimes.values())
        decision: Literal["promote", "retain_conditional", "insufficient_evidence"] = (
            "promote"
            if all(value.passes for value in regimes.values())
            else "retain_conditional"
            if complete
            else "insufficient_evidence"
        )
        result[method] = PromotionSummary(problem, baseline, decision, regimes)
    return result


def analyze_rows(
    rows: Sequence[Mapping[str, object]], *, expected_seeds: int
) -> AnalysisReport:
    if expected_seeds <= 0:
        raise ValueError("expected_seeds must be positive.")
    scientific = [_scientific_row(row) for row in rows if "relative_l2_u" in row]
    groups: dict[ResultKey, list[ScientificRow]] = defaultdict(list)
    for row in scientific:
        groups[row.key].append(row)
    invalid = tuple(InvalidRow(*row.key, row.seed) for row in scientific if not row.valid)
    incomplete = tuple(
        GroupEvidence(*key, len(values))
        for key, values in sorted(groups.items())
        if len(values) != expected_seeds
    )
    duplicates = tuple(
        GroupEvidence(*key, len(values))
        for key, values in sorted(groups.items())
        if len({row.seed for row in values}) != len(values)
    )
    integrity = Integrity(
        len(scientific),
        len(rows) - len(scientific),
        invalid,
        incomplete,
        duplicates,
        bool(scientific) and not invalid and not incomplete and not duplicates,
    )
    problems: dict[str, list[ProblemSummary]] = defaultdict(list)
    # Invalid groups retain integrity evidence but must not enter scientific rankings.
    for key, values in sorted(groups.items()):
        if all(row.valid for row in values):
            problems[f"{key[0]}/{key[1]}"].append(_summarize(key, values))
    return AnalysisReport(
        integrity, dict(problems), _method_summaries(problems), _promotions(groups)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()
    report = analyze_rows(load_rows(args.input), expected_seeds=args.expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(asdict(report.integrity), sort_keys=True))


if __name__ == "__main__":
    main()
