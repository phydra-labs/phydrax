from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import geophysics as geo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 17 or args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError(
            "size >= 17, positive steps/repeats and nonnegative warmup are required"
        )

    spacing = 5.0
    time_step = 0.001
    maximum_speed = 2000.0
    grid = geo.AcousticGrid((args.size, args.size), (spacing, spacing))
    centre = spacing * (args.size - 1) / 2
    acquisition = geo.SeismicAcquisition(
        grid,
        ((centre, centre),),
        ((centre + 4 * spacing, centre), (centre, centre + 4 * spacing)),
    )
    plan = geo.ConstantDensityAcousticPlan(
        grid,
        time_step,
        args.steps,
        maximum_speed,
        1000.0,
        absorber_cells=6,
        absorber_strength=5.0,
    )
    times = time_step * jnp.arange(args.steps)
    source = 1.0e-3 * geo.ricker_wavelet(times, 25.0, delay=0.04)
    rates = source[:, None]
    wavespeed = jnp.full(grid.shape, 1500.0)
    direction = jnp.exp(
        -(
            (grid.axis_nodes()[0][:, None] - centre) ** 2
            + (grid.axis_nodes()[1][None, :] - centre) ** 2
        )
        / (2 * (4 * spacing) ** 2)
    )
    cotangent = jnp.ones((acquisition.receivers.count, args.steps + 1))

    def forward(speed):
        return plan.simulate(
            speed,
            acquisition,
            rates,
            replay="block",
            block_size=min(16, args.steps),
        ).traces.values

    def evaluate(speed):
        traces, tangent = jax.jvp(forward, (speed,), (direction,))
        _, pullback = jax.vjp(forward, speed)
        adjoint = pullback(cotangent)[0]
        pairing = jnp.vdot(cotangent, tangent) - jnp.vdot(adjoint, direction)
        return traces, tangent, adjoint, pairing

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(wavespeed), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(wavespeed), warmup=args.warmup, repeats=args.repeats
    )
    traces, tangent, adjoint, pairing = result
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "shape": grid.shape,
            "steps": args.steps,
            "receivers": acquisition.receivers.count,
            "checkpoint_block_size": min(16, args.steps),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "identity": plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "finite_traces": bool(jnp.all(jnp.isfinite(traces))),
            "maximum_pressure_Pa": float(jnp.max(jnp.abs(traces))),
            "finite_tangent": bool(jnp.all(jnp.isfinite(tangent))),
            "finite_adjoint": bool(jnp.all(jnp.isfinite(adjoint))),
            "jvp_vjp_pairing_residual": float(jnp.abs(pairing)),
            "cfl_number": plan.cfl_number,
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
