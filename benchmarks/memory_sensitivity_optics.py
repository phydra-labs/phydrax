"""Phase-separated selected Greek and optical interaction memory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax._physical import RelativityScaleContract
from phydrax._strict import StrictModule
from phydrax.finance.valuation._sensitivity import evaluate_aad_greeks, GreekRequest
from phydrax.optics.geometric._nonsequential import NonSequentialSurfaceTable
from phydrax.optics.transport._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    TissueOpticalMedium,
)

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


class PolynomialValuation(StrictModule):
    def __call__(self, parameters: Array) -> Array:
        return jnp.sum(parameters**4) + jnp.sum(parameters) ** 2


def greek_entry(count: int, dense_baseline: bool) -> tuple[Any, Array]:
    names = tuple(f"p{index}" for index in range(count))
    request = GreekRequest(names, second_order_names=(names[0], names[-1]))

    def entry(parameters: Array) -> tuple[Array, Array, Array]:
        if dense_baseline:
            base, gradient = jax.value_and_grad(PolynomialValuation())(parameters)
            diagonal = jnp.diag(jax.hessian(PolynomialValuation())(parameters))
            requested = jnp.asarray(
                tuple(name in request.second_order_names for name in names),
                dtype=jnp.bool_,
            )
            diagonal = jnp.where(requested, diagonal, jnp.nan)
            finite = (
                jnp.isfinite(base)
                & jnp.all(jnp.isfinite(gradient))
                & jnp.all(jnp.isfinite(jnp.where(requested, diagonal, 0.0)))
            )
            gradient = eqx.error_if(
                gradient, ~finite, "AAD Greek evaluation produced non-finite derivatives."
            )
            return gradient, diagonal, finite
        result = evaluate_aad_greeks(
            PolynomialValuation(), parameters, request, valuation_id="memory-polynomial"
        )
        return result.first_order, result.second_order_diagonal, result.evidence.finite

    return jax.jit(entry), jnp.linspace(0.1, 0.9, count, dtype=jnp.float64)


def optical_entry(count: int, batch: int) -> tuple[Any, Array]:
    surfaces = NonSequentialSurfaceTable(
        jnp.asarray(
            [
                [-100.0, -100.0, -50.0],
                [100.0, -100.0, -50.0],
                [100.0, 100.0, -50.0],
                [-100.0, 100.0, -50.0],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((1,), dtype=jnp.float64),
        surface_ids=jnp.zeros((2,), dtype=jnp.int32),
    )
    medium = TissueOpticalMedium(
        jnp.asarray([0.2], dtype=jnp.float64),
        jnp.asarray([0.8], dtype=jnp.float64),
        jnp.asarray([0.6], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    prepared = prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=RelativityScaleContract.si(),
            maximum_interactions=count,
            photon_batch_size=batch,
        )
    )
    source = ExplicitPhotonSource(
        launch_optical_photons(
            jnp.zeros((batch, 3), dtype=jnp.float64),
            jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64), (batch, 3)),
            0,
            wavelengths=500e-9,
        )
    )

    def entry(key: Array) -> tuple[Array, ...]:
        result = simulate_optical_photons(prepared, source, key)
        return (
            result.per_photon_tallies.truncated,
            result.status,
            result.terminal_state.weights,
            result.terminal_state.positions,
            result.terminal_state.directions,
            result.terminal_state.jones_vectors,
            result.interaction_counts,
            result.maximum_absolute_ledger_residual,
        )

    return jax.jit(entry), jax.random.key(2024)


def measure(
    area: str, capacity: int, repeats: int, batch: int, dense_baseline: bool
) -> dict[str, Any]:
    (entry, argument), peak = host_allocation_peak(
        lambda: (
            greek_entry(capacity, dense_baseline)
            if area == "sensitivity"
            else optical_entry(capacity, batch)
        )
    )
    compiled, timing = measure_lower_and_compile(
        lambda: entry.lower(argument), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(lambda: compiled(argument), warmup=1, repeats=repeats)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled",
        unavailable_reason="Compiler estimate unavailable.",
    )
    defect = 0.0
    if area == "sensitivity":
        x = np.asarray(argument)
        defect = float(np.max(np.abs(np.asarray(result[0]) - (4 * x**3 + 2 * x.sum()))))
        defect = max(
            defect, float(np.nanmax(np.abs(np.asarray(result[1]) - (12 * x**2 + 2))))
        )
    else:
        defect = float(np.asarray(result[-1]))
    history_digest = (
        None
        if area == "sensitivity"
        else hashlib.sha256(
            b"".join(np.asarray(value).tobytes() for value in result[1:-1])
        ).hexdigest()
    )
    return {
        "capacity": capacity,
        "batch": batch,
        "dense_baseline": dense_baseline,
        "history_digest": history_digest,
        "preparation_peak_bytes": peak,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "temporary_bytes": compiled.memory_analysis().temp_size_in_bytes,
        "retained_output_bytes": logical_array_bytes(result),
        "reference_defect": defect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dense-baseline", action="store_true")
    args = parser.parse_args()
    sizes = {
        "sensitivity": [16, 64, 256] if args.smoke else [16, 64, 256, 1024],
        "optics": [16, 64, 256],
    }
    batches = [1] if args.smoke else [1, 16, 256, 1024]
    records = {
        "sensitivity": [
            measure("sensitivity", size, args.repeats, 1, args.dense_baseline)
            for size in sizes["sensitivity"]
        ]
    }
    records.update(
        {
            f"optics_batch{batch}": [
                measure("optics", size, args.repeats, batch, False)
                for size in sizes["optics"]
            ]
            for batch in batches
        }
    )
    gates: dict[str, Any] = {}
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        for area, cases in records.items():
            capacities = [case["capacity"] for case in cases]
            gates[area] = evaluate_memory_gate(
                capacities,
                [case["temporary_bytes"] for case in baseline["cases"][area]],
                [case["temporary_bytes"] for case in cases],
                MemoryGatePolicy(),
            ).to_dict()
    output = {
        "identity": asdict(
            capture_benchmark_identity(
                Path(__file__).resolve().parents[1],
                Path(__file__).resolve(),
                ("cases", "gates"),
            )
        ),
        "environment": asdict(capture_environment()),
        "cases": records,
        "gates": gates,
        "logical_workspace": {
            "sensitivity_requested_hvp": "O(P), two requested diagonals",
            "optics_candidate": "8 bytes scalar truncation carry",
        },
        "logical_exponents": {
            "sensitivity": fitted_scaling_exponent(
                sizes["sensitivity"], [8 * n for n in sizes["sensitivity"]]
            ),
            "optics": fitted_scaling_exponent(sizes["optics"], [8] * 3),
        },
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    write_json_atomic(args.output, output)


if __name__ == "__main__":
    main()
