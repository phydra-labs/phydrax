#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class ClosureBenchmarkCase:
    name: str
    size: int
    compile_and_first_ms: float
    steady_ms: float
    scientific_defect: float
    successful: bool


@dataclass(frozen=True)
class ClosureBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    cases: tuple[ClosureBenchmarkCase, ...]

    @property
    def passed(self):
        return bool(
            self.cases
            and all(case.successful for case in self.cases)
            and max(case.scientific_defect for case in self.cases) < 0.35
        )


def _time(function, *arguments, repetitions=3):
    started = perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    first_ms = 1.0e3 * (perf_counter() - started)
    started = perf_counter()
    value = first
    for _ in range(repetitions):
        value = function(*arguments)
    jax.block_until_ready(value)
    return first_ms, 1.0e3 * (perf_counter() - started) / repetitions, value


def _fmm_case(count, target_count):
    key = jax.random.key(count)
    position = 1.8 * jax.random.uniform(key, (count, 2)) - 0.9
    strength = jax.random.normal(jax.random.fold_in(key, 1), (count,))
    core = jnp.full((count,), 0.04)
    target_position = (
        1.8 * jax.random.uniform(jax.random.fold_in(key, 2), (target_count, 2)) - 0.9
    )
    source = phx.discretization.VortexSourceState(position, strength, core_radius=core)
    target = phx.discretization.VortexTargetState(target_position)
    fmm = phx.operators.VortexFMMPlan(
        position,
        (-1.0, -1.0),
        (1.0, 1.0),
        depth=2,
        expansion_order=1,
        leaf_capacity=max(8, count // 4),
    ).prepare(
        source_capacity=count,
        target_capacity=target_count,
        target_topology="arbitrary-targets",
    )
    direct = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=count, maximum_targets=target_count
    ).prepare(
        source_capacity=count,
        target_capacity=target_count,
        target_topology="arbitrary-targets",
    )
    evaluate = eqx.filter_jit(lambda src, tgt: fmm.evaluate(src, tgt).velocity)
    first_ms, steady_ms, value = _time(evaluate, source, target)
    reference = direct.evaluate(source, target).velocity
    defect = float(
        jnp.linalg.norm(value - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1.0e-12)
    )
    return ClosureBenchmarkCase(
        "hierarchical-fmm-2d",
        count,
        first_ms,
        steady_ms,
        defect,
        bool(jnp.all(jnp.isfinite(value)) and defect < 0.35),
    )


def _remesh_case(count, grid_count):
    key = jax.random.key(1000 + count)
    capacity = grid_count**2
    position = (
        jnp.zeros((capacity, 2)).at[:count].set(jax.random.uniform(key, (count, 2)))
    )
    strength = (
        jnp.zeros((capacity,))
        .at[:count]
        .set(jax.random.normal(jax.random.fold_in(key, 1), (count,)))
    )
    active = jnp.arange(capacity) < count
    state = phx.discretization.VortexPopulationState(
        position,
        strength,
        jnp.where(active, 0.05, 1.0),
        jnp.where(active, 1.0 / count, 1.0),
        active,
        jnp.where(active, jnp.arange(capacity), -1),
        jnp.full((capacity,), -1),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,)),
        jnp.asarray(count, dtype=jnp.int64),
    )
    plan = phx.discretization.CompleteVortexRemeshPlan(
        (0.0, 0.0),
        (1.0, 1.0),
        (grid_count, grid_count),
        degree=3,
        boundary="reject",
        periodic=(True, True),
    )
    evaluate = eqx.filter_jit(lambda current: plan.apply(current))
    first_ms, steady_ms, result = _time(evaluate, state)
    defect = float(jnp.max(jnp.abs(result.evidence.circulation_residual)))
    return ClosureBenchmarkCase(
        "vortex-remesh-2d", count, first_ms, steady_ms, defect, bool(result.successful)
    )


def run_closure_benchmarks(*, smoke=False):
    count = 8 if smoke else 64
    targets = 4 if smoke else 32
    grid = 8 if smoke else 16
    return ClosureBenchmarkReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        (_fmm_case(count, targets), _remesh_case(min(count, grid * grid // 2), grid)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark completed vortex capabilities."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_closure_benchmarks(smoke=args.smoke)
    payload = json.dumps({**asdict(report), "passed": report.passed}, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        temporary.replace(args.output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
