"""Benchmark the qualified contact-free tendon-driven reduced-rod profile."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.solid_mechanics import (
    FrictionlessElasticTendonPlan,
    prepare_frictionless_elastic_tendon,
    prepare_reduced_rod,
    prepare_reduced_rod_dynamics,
    prepare_reduced_rod_plant,
    prepare_rod,
    prepare_tendon_driven_rod_plant,
    ReducedRodPlan,
    ReducedRodSemiImplicitVelocityEuler,
    RodMaterialStation,
    RodPlan,
    RodStrainBasisPlan,
    TendonRoutePlan,
)
from phydrax.dynamics import PlantStepContext


def _synchronize(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(operation: Callable[[], Any], /, *, repeats: int):
    result = operation()
    _synchronize(result)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        _synchronize(result)
        samples.append((time.perf_counter_ns() - started) * 1.0e-6)
    values = np.asarray(samples, dtype=float)
    return result, {
        "maximum_ms": round(float(np.max(values)), 6),
        "median_ms": round(float(np.median(values)), 6),
        "minimum_ms": round(float(np.min(values)), 6),
    }


def _storage_bytes(value: Any, /) -> int:
    total = 0
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        if isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
            array = np.asarray(leaf)
            total += int(array.size * array.dtype.itemsize)
    return total


def _prepare_case(case: dict[str, int], /):
    dtype = jnp.float32
    segment_count = case["segments"]
    piece_count = case["pieces"]
    rest_positions = jnp.stack(
        (
            jnp.linspace(0.0, 1.0, segment_count + 1, dtype=dtype),
            jnp.zeros((segment_count + 1,), dtype=dtype),
            jnp.zeros((segment_count + 1,), dtype=dtype),
        ),
        axis=-1,
    )
    rod = prepare_rod(
        RodPlan(
            jnp.stack(
                (
                    jnp.arange(segment_count, dtype=jnp.int32),
                    jnp.arange(1, segment_count + 1, dtype=jnp.int32),
                ),
                axis=-1,
            ),
            rest_positions,
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (segment_count, 3, 3)),
            jnp.ones((segment_count + 1,), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.02, 0.02, 0.01), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((100.0, 40.0, 40.0), dtype=dtype)),
                (segment_count, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((4.0, 4.0, 2.0), dtype=dtype)),
                (segment_count - 1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.piecewise_constant(
        jnp.linspace(0.0, 1.0, piece_count + 1, dtype=dtype),
        dimension=3,
        component_scales=jnp.ones((6,), dtype=dtype),
        quadrature_order=case["quadrature"],
    )
    if basis.coordinate_count != case["coordinates"]:
        raise ValueError("benchmark coordinates must equal six times pieces")
    reduced = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduced)
    base = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
    )
    tendon_count = case["tendons"]
    offsets = np.linspace(-0.03, 0.03, tendon_count) if tendon_count > 1 else (0.0,)
    tendons = []
    for tendon_index, offset_y in enumerate(offsets):
        route = TendonRoutePlan(
            (
                RodMaterialStation(
                    0, 0.0, jnp.asarray((0.0, offset_y, 0.0), dtype=dtype)
                ),
                RodMaterialStation(
                    segment_count - 1,
                    1.0,
                    jnp.asarray((0.0, offset_y, 0.0), dtype=dtype),
                ),
            ),
            label=f"benchmark-tendon-{tendon_index}",
        )
        tendon = FrictionlessElasticTendonPlan(
            route,
            20.0,
            free_length_bounds=(0.8, 1.2),
            payout_rate_bounds=(-0.05, 0.05),
            tendon_length_bounds=(0.9, 1.1),
            maximum_tension=20.0,
            power_tolerance=1.0e-5,
            label=f"benchmark-tendon-{tendon_index}",
        )
        tendons.append(prepare_frictionless_elastic_tendon(tendon, reduced))
    return prepare_tendon_driven_rod_plant(
        base,
        tuple(tendons),
        tuple(jnp.asarray(0.98, dtype=dtype) for _ in tendons),
    )


def _rollout(plant, case: dict[str, int], /):
    parameters = plant.bind_parameters()
    results = []
    for world in range(case["worlds"]):
        reset = plant.reset(jax.random.fold_in(jax.random.key(1729), world), parameters)
        state = reset.accepted_state
        for step_index in range(case["steps"]):
            direction = -1.0 if (world + step_index) % 2 else 1.0
            rates = tuple(
                jnp.asarray(
                    direction * (tendon_index + 1) * 0.002 / case["tendons"],
                    dtype=state.time.dtype,
                )
                for tendon_index in range(case["tendons"])
            )
            command = plant.command(rates)
            context = PlantStepContext(
                state.time,
                state.time + jnp.asarray(1.0e-4, dtype=state.time.dtype),
                state.step_index,
            )
            result = plant.step(context, state, command, parameters)
            results.append(result)
            state = result.accepted_state
    return tuple(results)


def _run_case(case: dict[str, int], /, *, repeats: int) -> dict[str, Any]:
    plant, preparation_timing = _measure(lambda: _prepare_case(case), repeats=repeats)
    results, rollout_timing = _measure(lambda: _rollout(plant, case), repeats=repeats)
    successful = all(bool(np.asarray(result.successful)) for result in results)
    statuses = tuple(int(np.asarray(result.status)) for result in results)
    tendon_residual = max(
        (
            abs(float(np.asarray(result.evidence.tendon_ledger.total_energy_residual)))
            for result in results
        ),
        default=0.0,
    )
    mechanics_residual = max(
        (
            abs(
                float(
                    np.asarray(
                        result.evidence.integration_result.evidence.ledger.balance_residual
                    )
                )
            )
            for result in results
        ),
        default=0.0,
    )
    tendon_balanced = all(
        bool(np.asarray(result.evidence.tendon_ledger.balanced)) for result in results
    )
    mechanics_balanced = all(
        bool(np.asarray(result.evidence.integration_result.evidence.ledger.balanced))
        for result in results
    )
    finite_timings = all(
        math.isfinite(value)
        for timing in (preparation_timing, rollout_timing)
        for value in timing.values()
    )
    transition_count = case["worlds"] * case["steps"]
    passed = (
        successful
        and tendon_balanced
        and mechanics_balanced
        and len(results) == transition_count
        and finite_timings
    )
    return {
        "evidence": {
            "all_mechanics_ledgers_balanced": mechanics_balanced,
            "all_tendon_ledgers_balanced": tendon_balanced,
            "all_transitions_successful": successful,
            "statuses": statuses,
        },
        "passed": passed,
        "residuals": {
            "maximum_absolute_mechanics_balance": mechanics_residual,
            "maximum_absolute_tendon_energy": tendon_residual,
        },
        "size": dict(case),
        "storage_bytes": {
            "accepted_result_sequence": _storage_bytes(results),
            "plant_arrays": _storage_bytes(plant),
            "single_accepted_state": _storage_bytes(results[-1].accepted_state),
        },
        "timings_ms": {
            "preparation": preparation_timing,
            "rollout": rollout_timing,
        },
        "work": {
            "timed_calls": repeats,
            "transition_count_per_call": transition_count,
            "warmup_calls": 1,
        },
    }


def run_benchmarks(*, smoke: bool = False) -> dict[str, Any]:
    matrix = (
        (
            {
                "coordinates": 6,
                "pieces": 1,
                "quadrature": 2,
                "segments": 2,
                "steps": 1,
                "tendons": 1,
                "worlds": 1,
            },
        )
        if smoke
        else (
            {
                "coordinates": 6,
                "pieces": 1,
                "quadrature": 2,
                "segments": 4,
                "steps": 8,
                "tendons": 1,
                "worlds": 1,
            },
            {
                "coordinates": 12,
                "pieces": 2,
                "quadrature": 4,
                "segments": 8,
                "steps": 16,
                "tendons": 2,
                "worlds": 4,
            },
            {
                "coordinates": 24,
                "pieces": 4,
                "quadrature": 6,
                "segments": 16,
                "steps": 32,
                "tendons": 4,
                "worlds": 8,
            },
        )
    )
    repeats = 1 if smoke else 5
    results = tuple(_run_case(case, repeats=repeats) for case in matrix)
    return {
        "benchmark": "contact-free-tendon-driven-spatial-reduced-rod",
        "execution": "eager-jax-with-one-untimed-warmup",
        "matrix": matrix,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark contact-free tendon-driven spatial reduced rods."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one minimal matrix case with one timed call",
    )
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run_benchmarks(smoke=arguments.smoke),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
