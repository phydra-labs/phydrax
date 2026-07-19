#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .configuration import get_configuration, PROFILES
from .report import BenchmarkReport, collect_environment, ScenarioResult, utc_now_iso
from .scenarios import SCENARIOS


def run_benchmark_matrix(
    *,
    profile: str = "smoke",
    root_seed: int = 20260718,
    scenario_names: Sequence[str] | None = None,
) -> BenchmarkReport:
    """Run selected scenarios in stable registry order and retain failures in JSON."""
    configuration = get_configuration(profile)
    selected = _selected_scenarios(scenario_names)
    started_at = utc_now_iso()
    started = time.perf_counter()
    results: list[ScenarioResult] = []
    for name, scenario in selected:
        registry_index = tuple(SCENARIOS).index(name)
        seed = int(root_seed) + 10_000 * (registry_index + 1)
        try:
            result = scenario(configuration, seed)
            if result.name != name or result.seed != seed:
                raise RuntimeError(
                    f"Scenario {name!r} returned incompatible identity metadata."
                )
        except Exception as error:
            result = ScenarioResult(
                name=name,
                description=scenario.__doc__ or name,
                seed=seed,
                metadata={"profile": configuration.profile},
                error_type=type(error).__name__,
                error_message=str(error),
            )
        results.append(result)
    duration = time.perf_counter() - started
    return BenchmarkReport(
        profile=configuration.profile,
        root_seed=int(root_seed),
        started_at_utc=started_at,
        duration_seconds=duration,
        configuration=configuration.as_dict(),
        environment=collect_environment(),
        scenarios=tuple(results),
    )


def _selected_scenarios(
    requested: Sequence[str] | None,
) -> tuple[tuple[str, Any], ...]:
    if requested is None:
        names = tuple(SCENARIOS)
    else:
        requested_names = tuple(str(name) for name in requested)
        if not requested_names or len(requested_names) != len(set(requested_names)):
            raise ValueError("scenario_names must contain distinct registered names.")
        unknown = tuple(name for name in requested_names if name not in SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown benchmark scenarios: {unknown!r}.")
        requested_set = frozenset(requested_names)
        names = tuple(name for name in SCENARIOS if name in requested_set)
    return tuple((name, SCENARIOS[name]) for name in names)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.uq_benchmarks",
        description="Run the deterministic PhydraX UQ scientific benchmark matrix.",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIOS),
        dest="scenarios",
        help="Run one scenario; repeat to select several. Defaults to the full matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path. Defaults to .tmp/uq-benchmark-<profile>.json.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Return zero even when a release gate fails; the JSON still records failure.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered scenario names without running them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list:
        print("\n".join(SCENARIOS))
        return 0
    output = args.output or Path(f".tmp/uq-benchmark-{args.profile}.json")
    report = run_benchmark_matrix(
        profile=args.profile,
        root_seed=args.seed,
        scenario_names=args.scenarios,
    )
    destination = report.write_json(output)
    for scenario in report.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        failures = "" if scenario.passed else f" ({', '.join(scenario.failures)})"
        print(f"{status:4} {scenario.name}{failures}")
    print(
        f"{report.summary['scenarios_passed']}/{report.summary['scenario_count']} "
        f"scenarios passed in {report.duration_seconds:.3f}s"
    )
    print(destination)
    return 0 if report.passed or args.no_fail else 1


__all__ = ["main", "run_benchmark_matrix"]
