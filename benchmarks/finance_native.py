#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


jax.config.update("jax_enable_x64", True)

_SUITES = ("valuation", "econometrics", "portfolio", "exposure", "execution", "advanced")


def _synchronize(value: Any, /) -> Any:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    return value


def _valuation() -> Callable[[], jax.Array]:
    model = phx.finance.models.BlackScholesModel(0.2)

    def solve() -> jax.Array:
        result = phx.finance.valuation.evaluate_black_scholes_european(
            model, 100.0, 100.0, 1.0, 0.05
        )
        implied = phx.finance.valuation.invert_black_scholes_implied_volatility(
            result.value, 100.0, 100.0, 1.0, 0.05
        )
        return jnp.stack((result.value, implied.volatility))

    return solve


def _econometrics() -> Callable[[], jax.Array]:
    series = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])

    def solve() -> jax.Array:
        fit = phx.uq.fit_arima(
            series,
            p=1,
            include_intercept=False,
            ridge=1.0e-12,
        )
        return fit.model.autoregressive

    return solve


def _portfolio() -> Callable[[], jax.Array]:
    evidence = phx.finance.FinanceEvidenceBinding(
        ("synthetic-data",),
        ("synthetic-law",),
        ("native-qp",),
        ("benchmark-only",),
    )
    forecast = phx.finance.portfolio.ForecastLaw(
        ("asset:a", "asset:b"),
        jnp.zeros(2),
        jnp.asarray([[1.0, 0.0], [0.0, 4.0]]),
        law=phx.finance.PhysicalLaw(
            "benchmark-forecast", "synthetic", "two-assets", "decision"
        ),
        as_of_time_ns=1,
        available_time_ns=2,
        evidence=evidence,
    )
    problem = phx.finance.portfolio.PortfolioProblem(
        forecast,
        phx.finance.portfolio.MeanVarianceObjective(1.0, return_weight=0.0),
        phx.finance.portfolio.PortfolioConstraints(
            lower_weights=jnp.zeros(2),
            upper_weights=jnp.ones(2),
        ),
        problem_id="benchmark-portfolio",
        decision_time_ns=2,
    )
    compiled = phx.finance.portfolio.compile_portfolio_problem(problem)

    def solve() -> jax.Array:
        native = phx.optim.solve_quadratic_program(compiled.program)
        result = phx.finance.portfolio.portfolio_result_from_native(
            compiled,
            native,
            feasibility_tolerance=2.0e-5,
        )
        return result.decision.weights

    return solve


def _exposure_profile() -> phx.finance.exposure.ExposureProfile:
    return phx.finance.exposure.ExposureProfile(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.asarray([0.0, 4.0, 8.0]),
        jnp.asarray([0.0, 10.0, 20.0]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.asarray(1.0),
        jnp.asarray(False),
        0.95,
        "benchmark-exposure",
        "benchmark-values",
        "benchmark-weights",
        None,
    )


def _exposure() -> Callable[[], jax.Array]:
    profile = _exposure_profile()

    def solve() -> jax.Array:
        result = phx.finance.exposure.compute_cva(
            profile,
            jnp.asarray([0.0, 0.1, 0.1]),
            0.4,
            jnp.ones(3),
            counterparty_default_law_id="counterparty-q",
            recovery_terms_id="recovery",
            discount_curve_id="discount",
        )
        return jnp.atleast_1d(result.adjustment)

    return solve


def _execution() -> Callable[[], jax.Array]:
    model = phx.finance.execution.AlmgrenChrissModel(
        volatility=0.2,
        risk_aversion=0.1,
        temporary_impact=0.5,
        permanent_impact=0.01,
        model_id="benchmark-impact",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, 9),
        time_id="benchmark-execution-grid",
    )

    def solve() -> jax.Array:
        schedule = phx.finance.execution.solve_almgren_chriss_schedule(model, grid, 12.0)
        return jnp.stack(
            (
                schedule.inventory[0],
                schedule.inventory[-1],
                schedule.conservation_residual,
            )
        )

    return solve


def _finite_target(points: Sequence[float], weights: Sequence[float], provenance: str):
    return phx.integration.discrete(
        jnp.asarray(points),
        cx.Field(jnp.asarray(weights), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


def _advanced() -> Callable[[], jax.Array]:
    source = jnp.asarray([-1.0, 1.0])
    target = jnp.asarray([-2.0, 2.0])
    transport = phx.transport.discrete_problem(
        _finite_target(source, [0.5, 0.5], "benchmark-source"),
        _finite_target(target, [0.5, 0.5], "benchmark-target"),
        cost=phx.transport.PrecomputedCost((source[:, None] - target[None, :]) ** 2),
    )
    problem = phx.transport.MartingaleTransportProblem(transport)

    def solve() -> jax.Array:
        result = phx.transport.solve_martingale_transport(problem)
        return result.coupling.reshape((-1,))

    return solve


_PREPARERS: Mapping[str, Callable[[], Callable[[], jax.Array]]] = {
    "valuation": _valuation,
    "econometrics": _econometrics,
    "portfolio": _portfolio,
    "exposure": _exposure,
    "execution": _execution,
    "advanced": _advanced,
}


def _run_suite(
    name: str,
    reference: Mapping[str, object],
    /,
    *,
    repeats: int,
) -> dict[str, object]:
    started = time.perf_counter()
    operation = _PREPARERS[name]()
    preparation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = _synchronize(operation())
    solve_seconds = time.perf_counter() - started

    started = time.perf_counter()
    replay = _synchronize(operation())
    replay_seconds = time.perf_counter() - started

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _synchronize(operation())
        samples.append(time.perf_counter() - started)

    expected = np.asarray(reference["expected"], dtype=float)
    result_host = np.asarray(result, dtype=float)
    replay_host = np.asarray(replay, dtype=float)
    if result_host.shape != expected.shape:
        raise ValueError(f"Reference shape for suite {name!r} is inconsistent.")
    tolerance = float(reference["absolute_tolerance"])
    replay_tolerance = float(reference["replay_tolerance"])
    maximum_error = float(np.max(np.abs(result_host - expected)))
    maximum_replay_error = float(np.max(np.abs(replay_host - result_host)))
    correctness_passed = bool(
        np.isfinite(result_host).all() and maximum_error <= tolerance
    )
    replay_passed = bool(
        np.isfinite(replay_host).all() and maximum_replay_error <= replay_tolerance
    )
    return {
        "suite": name,
        "correctness_passed": correctness_passed,
        "replay_passed": replay_passed,
        "passed": correctness_passed and replay_passed,
        "maximum_absolute_error": maximum_error,
        "maximum_replay_error": maximum_replay_error,
        "tolerance": tolerance,
        "replay_tolerance": replay_tolerance,
        "result": result_host.tolist(),
        "preparation": {"seconds": preparation_seconds},
        "solve": {"seconds": solve_seconds},
        "replay": {"seconds": replay_seconds},
        "timing": {
            "repeats": repeats,
            "samples_seconds": samples,
            "mean_seconds": float(np.mean(samples)),
        },
    }


def _read_references(path: Path, /) -> Mapping[str, Mapping[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != set(_SUITES):
        raise ValueError("Finance benchmark reference suites are not canonical.")
    result: dict[str, Mapping[str, object]] = {}
    for name, record in value.items():
        if not isinstance(record, Mapping) or set(record) != {
            "expected",
            "absolute_tolerance",
            "replay_tolerance",
        }:
            raise ValueError(f"Finance benchmark reference {name!r} is invalid.")
        result[str(name)] = record
    return result


def run(
    suites: Sequence[str],
    /,
    *,
    repeats: int,
    reference_path: Path,
) -> dict[str, object]:
    """Run deterministic native finance workflows; timings never decide acceptance."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    selected = tuple(suites)
    if not selected or any(name not in _SUITES for name in selected):
        raise ValueError(
            "suites must name one or more supported finance benchmark suites."
        )
    if len(set(selected)) != len(selected):
        raise ValueError("suites must not contain duplicates.")
    references = _read_references(reference_path)
    results = tuple(
        _run_suite(name, references[name], repeats=repeats) for name in selected
    )
    return {
        "benchmark": "finance-native",
        "acceptance_basis": "correctness-and-replay-only",
        "suites": list(results),
        "passed": all(item["passed"] for item in results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correctness-first native finance workflow benchmarks."
    )
    parser.add_argument("--suite", action="append", choices=_SUITES, dest="suites")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("finance_reference.json"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    selected = _SUITES if arguments.suites is None else tuple(arguments.suites)
    result = run(selected, repeats=arguments.repeats, reference_path=arguments.reference)
    payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
