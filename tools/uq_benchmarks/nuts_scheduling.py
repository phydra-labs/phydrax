#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx

from .report import (
    BenchmarkReport,
    collect_environment,
    metric,
    ScenarioResult,
    utc_now_iso,
)
from .scenarios import _elliptic_solution


ProfileName = Literal["smoke", "standard"]
TargetName = Literal["correlated", "funnel", "elliptic"]
_TARGETS: tuple[TargetName, ...] = ("correlated", "funnel", "elliptic")
_MAX_NUM_DOUBLINGS = 8
_SAMPLE_PARITY_ATOL = 1e-7
_ACCEPTANCE_PARITY_ATOL = 1e-9


def _correlated_problem() -> phx.uq.PosteriorProblem:
    precision = jnp.asarray([[8.0, 2.8], [2.8, 1.6]])
    initial = jnp.zeros((2,))
    space = phx.uq.ParameterSpace(
        initial,
        log_prior=lambda value: -0.5 * value @ precision @ value,
    )
    return phx.uq.PosteriorProblem(space, lambda _: jnp.asarray(0.0))


def _funnel_problem() -> phx.uq.PosteriorProblem:
    initial = jnp.zeros((7,))

    def log_density(value):
        scale = value[0]
        latent = value[1:]
        variance = jnp.exp(scale)
        return (
            -0.5 * (scale / 3.0) ** 2
            - 0.5 * jnp.sum(latent**2) / variance
            - 0.5 * latent.size * scale
        )

    space = phx.uq.ParameterSpace(initial, log_prior=log_density)
    return phx.uq.PosteriorProblem(space, lambda _: jnp.asarray(0.0))


def _elliptic_problem(seed: int) -> phx.uq.PosteriorProblem:
    true_parameters = jnp.asarray([0.2, -0.55])
    true_solution = _elliptic_solution(true_parameters)
    sensor_indices = jnp.arange(1, true_solution.size, 3)
    observation_scale = 0.0025
    observations = true_solution[sensor_indices] + observation_scale * jr.normal(
        jr.key(seed),
        sensor_indices.shape,
    )
    likelihood = phx.uq.GaussianLikelihood(observation_scale)
    space = phx.uq.ParameterSpace(
        jnp.zeros((2,)),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    return phx.uq.PosteriorProblem(
        space,
        lambda parameters: jnp.sum(
            likelihood.log_prob(
                _elliptic_solution(parameters)[sensor_indices],
                observations,
            )
        ),
    )


def _problem(target: TargetName, seed: int) -> phx.uq.PosteriorProblem:
    if target == "correlated":
        return _correlated_problem()
    if target == "funnel":
        return _funnel_problem()
    return _elliptic_problem(seed)


def _maximum_tree_difference(left, right) -> float:
    differences = jax.tree_util.tree_map(
        lambda x, y: jnp.max(jnp.abs(x - y)),
        left,
        right,
    )
    return max(float(value) for value in jax.tree_util.tree_leaves(differences))


def _interleaved_work_quanta(integration_steps, *, chunk_size: int) -> int:
    draws = int(integration_steps.shape[1])
    total = 0
    for start in range(0, draws, chunk_size):
        chunk = integration_steps[:, start : start + chunk_size]
        total += int(jnp.max(jnp.sum(chunk, axis=1)))
    return total


def _case(
    *,
    target: TargetName,
    num_chains: int,
    num_warmup: int,
    num_draws: int,
    seed: int,
) -> ScenarioResult:
    problem = _problem(target, seed)
    sample_key = jr.key(seed + 1)
    initial_step_size = 0.1 if target == "elliptic" else 0.25
    started = time.perf_counter()
    vectorized = phx.uq.sample_nuts(
        problem,
        key=sample_key,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_draws,
        target_acceptance_rate=0.9,
        initial_step_size=initial_step_size,
        max_num_doublings=_MAX_NUM_DOUBLINGS,
        chain_method="vectorized",
    )
    interleaved = phx.uq.sample_nuts(
        problem,
        key=sample_key,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_draws,
        target_acceptance_rate=0.9,
        initial_step_size=initial_step_size,
        max_num_doublings=_MAX_NUM_DOUBLINGS,
        chain_method="interleaved",
    )
    jax.block_until_ready(interleaved.log_density)
    duration = time.perf_counter() - started
    lockstep_quanta = int(jnp.sum(jnp.max(vectorized.num_integration_steps, axis=0)))
    chunk_size = min(100, num_draws)
    interleaved_quanta = _interleaved_work_quanta(
        interleaved.num_integration_steps,
        chunk_size=chunk_size,
    )
    sample_difference = _maximum_tree_difference(
        vectorized.unconstrained_samples,
        interleaved.unconstrained_samples,
    )
    acceptance_difference = float(
        jnp.max(jnp.abs(vectorized.acceptance_rate - interleaved.acceptance_rate))
    )
    integration_mismatches = int(
        jnp.sum(vectorized.num_integration_steps != interleaved.num_integration_steps)
    )
    expansion_mismatches = int(
        jnp.sum(
            vectorized.num_trajectory_expansions != interleaved.num_trajectory_expansions
        )
    )
    divergence_mismatches = int(jnp.sum(vectorized.divergent != interleaved.divergent))
    mean_integration_steps = float(jnp.mean(interleaved.num_integration_steps))
    maximum_integration_steps = int(jnp.max(interleaved.num_integration_steps))
    trajectory_saturation_count = int(
        jnp.sum(interleaved.num_trajectory_expansions == _MAX_NUM_DOUBLINGS)
    )
    return ScenarioResult(
        name=f"{target}_chains_{num_chains}",
        description=(
            "Compare equal-draw vectorized and independently progressing NUTS chains."
        ),
        seed=seed,
        metrics={
            "maximum_sample_difference": metric(
                sample_difference,
                "accuracy",
                maximum=_SAMPLE_PARITY_ATOL,
                description="Same-backend absolute compiler-order tolerance.",
            ),
            "maximum_acceptance_difference": metric(
                acceptance_difference,
                "accuracy",
                maximum=_ACCEPTANCE_PARITY_ATOL,
                description="Same-backend absolute compiler-order tolerance.",
            ),
            "integration_step_mismatches": metric(
                integration_mismatches,
                "accuracy",
                maximum=0.0,
            ),
            "trajectory_expansion_mismatches": metric(
                expansion_mismatches,
                "accuracy",
                maximum=0.0,
            ),
            "divergence_mismatches": metric(
                divergence_mismatches,
                "accuracy",
                maximum=0.0,
            ),
            "vectorized_sampling_seconds": metric(
                vectorized.sampling_duration_seconds,
                "performance",
                unit="s",
            ),
            "interleaved_sampling_seconds": metric(
                interleaved.sampling_duration_seconds,
                "performance",
                unit="s",
            ),
            "vectorized_samples_per_second": metric(
                vectorized.samples_per_second,
                "performance",
                unit="sample/s",
            ),
            "interleaved_samples_per_second": metric(
                interleaved.samples_per_second,
                "performance",
                unit="sample/s",
            ),
            "observed_sampling_speedup": metric(
                vectorized.sampling_duration_seconds
                / interleaved.sampling_duration_seconds,
                "performance",
            ),
            "lockstep_work_quanta": metric(
                lockstep_quanta,
                "diagnostic",
            ),
            "interleaved_work_quanta": metric(
                interleaved_quanta,
                "diagnostic",
            ),
            "work_quanta_ratio": metric(
                lockstep_quanta / interleaved_quanta,
                "diagnostic",
            ),
            "mean_integration_steps": metric(
                mean_integration_steps,
                "diagnostic",
            ),
            "maximum_integration_steps": metric(
                maximum_integration_steps,
                "diagnostic",
            ),
            "trajectory_saturation_count": metric(
                trajectory_saturation_count,
                "diagnostic",
            ),
            "vectorized_divergences": metric(
                jnp.sum(vectorized.divergent),
                "diagnostic",
            ),
            "interleaved_divergences": metric(
                jnp.sum(interleaved.divergent),
                "diagnostic",
            ),
            "retained_sample_memory_bytes": metric(
                interleaved.sample_memory_bytes,
                "performance",
                unit="byte",
            ),
            "case_wall_seconds": metric(duration, "performance", unit="s"),
        },
        metadata={
            "target": target,
            "num_chains": num_chains,
            "num_warmup": num_warmup,
            "num_draws": num_draws,
            "chunk_size": chunk_size,
            "max_num_doublings": _MAX_NUM_DOUBLINGS,
            "jax_backend": jax.default_backend(),
        },
    )


def run_nuts_scheduling_benchmark(
    *,
    profile: ProfileName = "smoke",
    root_seed: int = 20260807,
    targets: Sequence[TargetName] | None = None,
    chain_counts: Sequence[int] | None = None,
) -> BenchmarkReport:
    """Run deterministic vectorized-versus-interleaved NUTS comparisons."""
    if profile == "smoke":
        num_warmup = 50
        num_draws = 100
        default_chain_counts = (4, 16)
    elif profile == "standard":
        num_warmup = 500
        num_draws = 1_000
        default_chain_counts = (4, 16, 64)
    else:
        raise ValueError("profile must be 'smoke' or 'standard'.")
    selected_targets = _TARGETS if targets is None else tuple(targets)
    selected_chain_counts = (
        default_chain_counts if chain_counts is None else tuple(chain_counts)
    )
    if not selected_targets or any(target not in _TARGETS for target in selected_targets):
        raise ValueError(f"targets must be selected from {_TARGETS!r}.")
    if not selected_chain_counts or any(count < 2 for count in selected_chain_counts):
        raise ValueError("chain_counts must contain integers of at least two.")
    if len(selected_targets) != len(set(selected_targets)):
        raise ValueError("targets must be distinct.")
    if len(selected_chain_counts) != len(set(selected_chain_counts)):
        raise ValueError("chain_counts must be distinct.")
    jax.config.update("jax_enable_x64", True)
    started_at = utc_now_iso()
    started = time.perf_counter()
    cases = tuple(
        (target, chain_count)
        for target in selected_targets
        for chain_count in selected_chain_counts
    )
    scenarios = tuple(
        _case(
            target=target,
            num_chains=chain_count,
            num_warmup=num_warmup,
            num_draws=num_draws,
            seed=root_seed + index,
        )
        for index, (target, chain_count) in enumerate(cases)
    )
    return BenchmarkReport(
        profile=profile,
        root_seed=root_seed,
        started_at_utc=started_at,
        duration_seconds=time.perf_counter() - started,
        configuration={
            "profile": profile,
            "targets": list(selected_targets),
            "chain_counts": list(selected_chain_counts),
            "num_warmup": num_warmup,
            "num_draws": num_draws,
            "speed_metrics_are_release_gates": False,
        },
        environment=collect_environment(),
        scenarios=scenarios,
        suite="phydrax-uq-nuts-scheduling",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vectorized and interleaved PhydraX NUTS chains."
    )
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--root-seed", type=int, default=20260807)
    parser.add_argument("--target", action="append", choices=_TARGETS, dest="targets")
    parser.add_argument("--chains", action="append", type=int, dest="chain_counts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_nuts_scheduling_benchmark(
        profile=args.profile,
        root_seed=args.root_seed,
        targets=args.targets,
        chain_counts=args.chain_counts,
    )
    output = args.output or Path(f".tmp/uq-nuts-scheduling-{args.profile}.json")
    destination = report.write_json(output)
    for scenario in report.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        print(f"{status:4} {scenario.name}")
    print(destination)
    return 0 if report.passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_nuts_scheduling_benchmark"]
