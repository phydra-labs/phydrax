"""Bubble-cloud coupled-acceleration scaling: dense Cholesky versus FMM conjugate gradients.

For every bubble count the benchmark builds one Rayleigh–Plesset cloud on a
jittered cubic lattice and times one coupled incompressible rate evaluation
(the implicit acceleration solve inside every ODE stage) on each route that
the resource policy admits. The auto-selected plan is the measured dense or
FMM plan, so route selection is observed rather than inferred. An explicit
dense construction beyond the bound records the actual resource refusal. The
benchmark also records preparation, lowering, compilation, warmed runtime,
compiler cost/memory analysis, logical retained bytes, FMM/dense agreement,
CG iterations and residuals.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax.bubble_dynamics as bd


_AMBIENT = 101325.0
_DENSITY = 998.0
_RADIUS = 10.0e-6


def _model() -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        "rayleigh_plesset",
        bd.PolytropicBubbleGasLaw(1.4),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        bd.CleanBubbleInterfaceLaw(0.072),
        bd.BubbleEnvironment(_AMBIENT, 293.15),
        liquid_density=_DENSITY,
        liquid_sound_speed=1481.0,
    )


def _cloud(count: int, seed: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(count ** (1.0 / 3.0)))
    spacing = 8.0 * _RADIUS
    lattice = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1)
    points = lattice.reshape(-1, 3)[:count] * spacing
    points = points + rng.uniform(-0.2, 0.2, size=points.shape) * spacing
    radii = _RADIUS * rng.uniform(0.8, 1.2, size=count)
    velocity = rng.uniform(-1.0, 1.0, size=count)
    return points, radii, velocity


def _plan(
    route: bd.BubbleCloudRoute,
    points: np.ndarray,
    radii: np.ndarray,
    velocity: np.ndarray,
    order: int,
    dense_entries: int,
    /,
) -> bd.BubbleCloudPlan:
    count = radii.shape[0]
    group = bd.BubbleSpeciesGroup(
        _model(),
        radii,
        points,
        bubble_ids=tuple(range(count)),
        initial_radii=radii * 1.05,
        initial_wall_velocities=velocity,
    )
    resources = bd.BubbleCloudResourcePolicy(
        maximum_dense_entries=dense_entries,
        maximum_dense_flops=1 << 40,
        fmm_order=order,
        coupling_tolerance=1.0e-12,
    )
    plan = bd.BubbleCloudPlan(
        (group,),
        bd.HarmonicPressureDrive(2.0e4, 2.0 * np.pi * 1.0e5),
        np.array([1.0e-6]),
        route=route,
        resources=resources,
    )
    return plan


def _dense_refusal(
    points: np.ndarray,
    radii: np.ndarray,
    velocity: np.ndarray,
    order: int,
    dense_entries: int,
    /,
) -> dict[str, str]:
    try:
        _plan("dense", points, radii, velocity, order, dense_entries)
    except ValueError as error:
        return {"exception": "ValueError", "message": str(error)}
    raise RuntimeError("The dense route was expected to exceed its resource policy.")


def _measure(
    prepared: bd.PreparedBubbleCloud, warmup: int, repeats: int, /
) -> dict[str, Any]:
    dynamic, static = eqx.partition(prepared, eqx.is_array)

    def evaluate(leaves: Any) -> Any:
        current = eqx.combine(leaves, static)
        rates = current.rates(current.initial_state, jnp.asarray(2.5e-7))
        return (
            rates.acceleration,
            rates.uncoupled_acceleration,
            rates.coupling_successful,
            rates.coupling_iterations,
            rates.coupling_residual,
        )

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(dynamic), warmup=warmup, repeats=repeats
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    acceleration, uncoupled, successful, iterations, residual = result
    return {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_seconds_dict(),
        "compiler": {
            "flops": evidence.flops,
            "temporary_bytes": evidence.temporary_bytes,
            "output_bytes": evidence.output_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
        },
        "logical_retained_bytes": logical_array_bytes(prepared),
        "coupling_successful": bool(successful),
        "coupling_iterations": int(iterations),
        "coupling_residual": float(residual),
        "correction": np.asarray(acceleration - uncoupled),
    }


def _measure_plan(
    plan: bd.BubbleCloudPlan, warmup: int, repeats: int, /
) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = plan.prepare()
    jax.block_until_ready(prepared)
    preparation_seconds = time.perf_counter() - started
    measurement = _measure(prepared, warmup, repeats)
    measurement["preparation_seconds"] = preparation_seconds
    return measurement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 64, 216, 512])
    parser.add_argument("--dense-entries", type=int, default=4096)
    parser.add_argument("--fmm-order", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(count < 2 for count in args.counts) or args.repeats < 1 or args.warmup < 0:
        raise ValueError(
            "counts >= 2, positive repeats and nonnegative warmup are required"
        )
    rows = []
    for count in args.counts:
        points, radii, velocity = _cloud(count, args.seed)
        policy = bd.BubbleCloudResourcePolicy(
            maximum_dense_entries=args.dense_entries, maximum_dense_flops=1 << 40
        )
        dense_admissible = policy.dense_admissible(count)
        auto_plan = _plan(
            "auto", points, radii, velocity, args.fmm_order, args.dense_entries
        )
        row: dict[str, Any] = {
            "bubble_count": count,
            "auto_selected_route": auto_plan.route,
            "dense_status": (
                "admitted" if dense_admissible else "refused-by-resource-policy"
            ),
            "dense_refusal": (
                None
                if dense_admissible
                else _dense_refusal(
                    points,
                    radii,
                    velocity,
                    args.fmm_order,
                    args.dense_entries,
                )
            ),
        }
        measurements = {
            auto_plan.route: _measure_plan(auto_plan, args.warmup, args.repeats)
        }
        if auto_plan.route == "dense":
            fmm_plan = _plan(
                "fmm", points, radii, velocity, args.fmm_order, args.dense_entries
            )
            measurements["fmm"] = _measure_plan(fmm_plan, args.warmup, args.repeats)
            dense = measurements["dense"]["correction"]
            fmm = measurements["fmm"]["correction"]
            row["fmm_relative_correction_error"] = float(
                np.max(np.abs(fmm - dense)) / np.max(np.abs(dense))
            )
        for measurement in measurements.values():
            measurement.pop("correction")
        row["routes"] = measurements
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = {
        "benchmark": "bubble-cloud-scaling",
        "environment": capture_environment().to_dict(),
        "fmm_order": args.fmm_order,
        "dense_entry_bound": args.dense_entries,
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
