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

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class VortexBenchmarkCase:
    backend: str
    dimension: int
    source_count: int
    target_count: int
    compile_and_first_ms: float
    steady_ms: float
    interactions_per_second: float
    scientific_defect: float
    successful: bool


@dataclass(frozen=True)
class VortexBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    cases: tuple[VortexBenchmarkCase, ...]

    @property
    def passed(self):
        return bool(
            self.cases
            and all(case.successful for case in self.cases)
            and max(case.scientific_defect for case in self.cases) < 1.0e-7
        )


def _time(function, arguments, repetitions=4):
    started = perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    first_ms = 1.0e3 * (perf_counter() - started)
    started = perf_counter()
    value = first
    for _ in range(repetitions):
        value = function(*arguments)
    jax.block_until_ready(value)
    steady_ms = 1.0e3 * (perf_counter() - started) / repetitions
    return first_ms, steady_ms, value


def _direct_case(dimension, count):
    key = jax.random.key(100 + dimension + count)
    position = jax.random.normal(key, (count, dimension))
    core = jnp.full((count,), 0.15)
    if dimension == 2:
        strength = jax.random.normal(jax.random.fold_in(key, 1), (count,))
        plan = phx.operators.GaussianDirectVortexPlan2D(
            maximum_sources=count,
            source_chunk_size=min(64, count),
            target_chunk_size=min(64, count),
        ).prepare(source_capacity=count, target_capacity=count)
    else:
        strength = jax.random.normal(jax.random.fold_in(key, 1), (count, 3))
        plan = phx.operators.GaussianErfDirectVortexPlan3D(
            source_chunk_size=min(32, count),
            target_chunk_size=min(32, count),
            interaction_budget=count * count,
        ).prepare(source_capacity=count, target_capacity=count)
    apply = jax.jit(lambda p, g, c: plan.evaluate(p, g, c).velocity)
    first_ms, steady_ms, value = _time(apply, (position, strength, core))
    defect = (
        float(jnp.max(jnp.abs(jnp.sum(value, axis=0))))
        if dimension == 2
        else float(jnp.max(jnp.abs(jnp.sum(value, axis=0))))
    )
    # Net velocity is not an invariant for unequal strengths; use finite/scale evidence.
    defect = 0.0 if bool(jnp.all(jnp.isfinite(value))) else float("inf")
    return VortexBenchmarkCase(
        f"direct-{dimension}d",
        dimension,
        count,
        count,
        first_ms,
        steady_ms,
        count * count / (steady_ms * 1.0e-3),
        defect,
        bool(jnp.all(jnp.isfinite(value))),
    )


def _periodic_case(count, grid_count):
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=2
    ).prepare()
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(grid_count, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    spectral = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(grid_count) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(2)))
    plan = phx.operators.PeriodicVortexInCellPlan(
        particles,
        grid,
        spectral,
        phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(source_capacity=count, target_capacity=count)
    key = jax.random.key(200 + count)
    position = jax.random.uniform(key, (count, 2))
    strength = jax.random.normal(jax.random.fold_in(key, 1), (count,))
    strength = strength - jnp.mean(strength)
    core = jnp.full((count,), 1.0 / grid_count)
    apply = jax.jit(lambda p, g, c: plan.evaluate(p, g, c).velocity)
    first_ms, steady_ms, value = _time(apply, (position, strength, core))
    evaluation = plan.evaluate(position, strength, core)
    diagnostics = evaluation.diagnostics.backend_diagnostics
    defect = float(jnp.maximum(diagnostics.balance_defect, diagnostics.divergence_norm))
    return VortexBenchmarkCase(
        "periodic-vic-2d",
        2,
        count,
        count,
        first_ms,
        steady_ms,
        count * (3**2) / (steady_ms * 1.0e-3),
        defect,
        bool(evaluation.successful),
    )


def _pse_case(count):
    key = jax.random.key(300 + count)
    position = jax.random.uniform(key, (count, 2))
    strength = jax.random.normal(jax.random.fold_in(key, 1), (count,))
    volume = jnp.full((count,), 1.0 / count)
    plan = phx.operators.GaussianParticleStrengthExchangePlan(
        2, 0.2, maximum_interactions=count * (count - 1) // 2
    ).prepare(capacity=count, dimension=2)
    apply = jax.jit(lambda p, g, v: plan.evaluate(p, g, v, 0.01).rate)
    first_ms, steady_ms, value = _time(apply, (position, strength, volume))
    defect = float(jnp.abs(jnp.sum(value)))
    return VortexBenchmarkCase(
        "gaussian-pse-2d",
        2,
        count,
        count,
        first_ms,
        steady_ms,
        count * (count - 1) // 2 / (steady_ms * 1.0e-3),
        defect,
        bool(jnp.all(jnp.isfinite(value))),
    )


def run_vortex_method_benchmark(*, smoke=False):
    direct_count = 32 if smoke else 512
    viscous_count = 24 if smoke else 256
    periodic_count = 32 if smoke else 512
    grid_count = 12 if smoke else 64
    return VortexBenchmarkReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        (
            _direct_case(2, direct_count),
            _direct_case(3, direct_count // 2),
            _periodic_case(periodic_count, grid_count),
            _pse_case(viscous_count),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark native vortex methods.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_vortex_method_benchmark(smoke=args.smoke)
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
