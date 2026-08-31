#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks._comparison import PerformancePolicy
from benchmarks._io import write_json_atomic

from .adapters import adapter_names, load_adapter, load_adapters
from .campaign import AVAILABLE_CASES, build_cases, CampaignConfig, PRESETS
from .compare import compare_reports
from .harness import run_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare reproducible advanced-solver benchmark reports.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a selected benchmark campaign")
    run.add_argument("--preset", choices=tuple(PRESETS), default="ci")
    run.add_argument("--adapter", action="append", choices=adapter_names())
    run.add_argument("--case", action="append", choices=AVAILABLE_CASES)
    run.add_argument("--size", type=_at_least_eight)
    run.add_argument("--seed", type=int)
    run.add_argument("--warmup", type=_nonnegative_integer)
    run.add_argument("--repeats", type=_positive_integer)
    run.add_argument("--relative-tolerance", type=_nonnegative_float)
    run.add_argument("--absolute-tolerance", type=_nonnegative_float)
    run.add_argument("--max-steps", type=_positive_integer)
    run.add_argument("--output", type=Path)

    control = commands.add_parser(
        "control",
        help="run prepared sparse and shifted-warm MPC horizon campaigns",
    )
    control.add_argument("--horizon", action="append", type=_positive_integer)
    control.add_argument("--seed", type=int, default=20260816)
    control.add_argument("--warmup", type=_nonnegative_integer, default=1)
    control.add_argument("--repeats", type=_positive_integer, default=5)
    control.add_argument("--output", type=Path)
    compare = commands.add_parser("compare", help="compare two complete JSON reports")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--allow-different-environments", action="store_true")
    compare.add_argument("--output", type=Path)
    compare.add_argument("--relative-performance-tolerance", type=_nonnegative_float)
    compare.add_argument("--absolute-performance-tolerance-ms", type=_nonnegative_float)
    compare.add_argument("--performance-confidence", type=float, default=0.95)
    compare.add_argument(
        "--performance-bootstrap-resamples",
        type=_positive_integer,
        default=10_000,
    )
    compare.add_argument(
        "--performance-minimum-samples",
        type=_positive_integer,
        default=5,
    )

    capability = commands.add_parser(
        "capabilities",
        help="emit exact adapter availability for every common capability",
    )
    capability.add_argument("--adapter", action="append", choices=adapter_names())
    capability.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        output = _run(arguments)
        destination = arguments.output
    elif arguments.command == "control":
        from .control_campaign import run_control_horizon_campaign

        output = run_control_horizon_campaign(
            tuple(arguments.horizon or (8, 32, 128)),
            seed=arguments.seed,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        destination = arguments.output
    elif arguments.command == "compare":
        reference = _read_json(arguments.reference)
        candidate = _read_json(arguments.candidate)
        output = compare_reports(
            reference,
            candidate,
            require_same_environment=not arguments.allow_different_environments,
            performance_policy=_performance_policy(arguments),
        )
        destination = arguments.output
    else:
        selected = tuple(arguments.adapter or adapter_names())
        output = {
            name: {
                capability: load_adapter(name).availability(capability).as_dict()
                for capability in _all_capabilities()
            }
            for name in selected
        }
        destination = arguments.output
    _emit_json(output, destination)


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[arguments.preset]
    selected_adapters = tuple(arguments.adapter or preset.adapters)
    selected_cases = tuple(arguments.case or preset.cases)
    config = CampaignConfig(
        seed=preset.seed if arguments.seed is None else arguments.seed,
        size=preset.size if arguments.size is None else arguments.size,
        warmup=preset.warmup if arguments.warmup is None else arguments.warmup,
        repeats=preset.repeats if arguments.repeats is None else arguments.repeats,
        adapters=selected_adapters,
        cases=selected_cases,
        relative_tolerance=(
            preset.relative_tolerance
            if arguments.relative_tolerance is None
            else arguments.relative_tolerance
        ),
        absolute_tolerance=(
            preset.absolute_tolerance
            if arguments.absolute_tolerance is None
            else arguments.absolute_tolerance
        ),
        max_steps=(
            preset.max_steps if arguments.max_steps is None else arguments.max_steps
        ),
    )
    adapters = load_adapters(config.adapters)
    cases = build_cases(config)
    return run_campaign(
        adapters,
        cases,
        selected_adapters=config.adapters,
        selected_cases=config.cases,
        seed=config.seed,
        warmup=config.warmup,
        repeats=config.repeats,
    )


def _read_json(path: Path, /) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON report {path} must contain an object")
    return value


def _emit_json(value: Any, destination: Path | None, /) -> None:
    if destination is None:
        payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        sys.stdout.write(payload)
    else:
        write_json_atomic(destination, value)


def _performance_policy(arguments: argparse.Namespace, /) -> PerformancePolicy | None:
    relative = arguments.relative_performance_tolerance
    absolute_ms = arguments.absolute_performance_tolerance_ms
    if relative is None and absolute_ms is None:
        return None
    return PerformancePolicy(
        objective="minimize",
        relative_tolerance=relative,
        absolute_tolerance=None if absolute_ms is None else absolute_ms / 1_000.0,
        confidence=float(arguments.performance_confidence),
        bootstrap_resamples=int(arguments.performance_bootstrap_resamples),
        minimum_samples=int(arguments.performance_minimum_samples),
    )


def _all_capabilities() -> tuple[str, ...]:
    return (
        "linear.scalar",
        "linear.block",
        "nonlinear.root",
        "nonlinear.vi",
        "eigen.general",
        "continuation.fold",
        "optimization.unconstrained",
        "optimization.constrained",
        "optimization.proximal",
        "optimization.bounded-least-squares",
        "optimization.linear-program",
        "optimization.quadratic-program",
        "optimization.conic-program",
    )


def _positive_integer(value: str, /) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _at_least_eight(value: str, /) -> int:
    parsed = int(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("expected an integer at least eight")
    return parsed


def _nonnegative_integer(value: str, /) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _nonnegative_float(value: str, /) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return parsed


if __name__ == "__main__":
    main()
