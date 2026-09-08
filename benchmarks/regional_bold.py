"""Delayed regional/BOLD forward and parameter-gradient scaling benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import neuroscience as ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.regions < 2
        or args.duration <= 0.0
        or args.dt <= 0.0
        or args.samples < 2
        or args.warmup < 0
        or args.repeats <= 0
    ):
        raise ValueError("Regions/samples/repeats and physical times must be positive.")

    weights = np.zeros((args.regions, args.regions))
    delays = np.zeros_like(weights)
    for target in range(args.regions):
        weights[target, (target - 1) % args.regions] = 0.5
        weights[target, (target + 1) % args.regions] = 0.5
        delays[target, (target - 1) % args.regions] = 2.7 * args.dt
        delays[target, (target + 1) % args.regions] = 4.3 * args.dt
    connectivity = ns.RegionalConnectivity(
        tuple(f"region-{index}" for index in range(args.regions)), weights, delays
    )

    def history(time_s, user_args):
        del time_s, user_args
        return jnp.zeros((args.regions, 2))

    def drive(time_s, neural, user_args):
        del user_args
        pulse = 0.2 * jnp.exp(-jnp.square((time_s - 0.25) / 0.08))
        return jnp.zeros_like(neural).at[0, 0].set(pulse)

    base = ns.regional_bold_problem(
        connectivity,
        ns.Hopf(a_per_s=-0.4, frequency_hz=0.1, coupling_per_s=0.6),
        history,
        ns.BalloonWindkessel(),
        ns.NeuralBOLDDrive([1.0, 0.0], [0.0, 0.0], gain=0.5),
        t0=0.0,
        t1=args.duration,
        drive=drive,
    )
    times = jnp.linspace(0.0, args.duration, args.samples)

    def evaluate(gain):
        problem = eqx.tree_at(lambda value: value.args.model.coupling_per_s, base, gain)
        result = ns.solve_regional(
            problem,
            save_times=times,
            solver=dfx.Heun(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=args.dt,
            max_steps=int(np.ceil(args.duration / args.dt)) + 16,
        )
        if result.bold is None:
            raise RuntimeError("Joint regional problem omitted BOLD samples.")
        return result.bold.values, result.bold.sample_valid

    gain = jnp.asarray(0.6)
    forward = eqx.filter_jit(evaluate)
    forward_compiled, forward_compilation = measure_lower_and_compile(
        lambda: forward.lower(gain), lambda lowered: lowered.compile()
    )
    forward_result, forward_execution = measure_repeated(
        lambda: forward_compiled(gain), warmup=args.warmup, repeats=args.repeats
    )

    objective = lambda value: jnp.sum(evaluate(value)[0][-1])
    value_and_grad = eqx.filter_jit(jax.value_and_grad(objective))
    gradient_compiled, gradient_compilation = measure_lower_and_compile(
        lambda: value_and_grad.lower(gain), lambda lowered: lowered.compile()
    )
    gradient_result, gradient_execution = measure_repeated(
        lambda: gradient_compiled(gain), warmup=args.warmup, repeats=args.repeats
    )

    bold, valid = forward_result
    objective_value, gradient = gradient_result
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "regions": args.regions,
            "directed_edges": connectivity.delayed_edge_count,
            "delay_values_s": list(connectivity.propagation_lags_s),
            "duration_s": args.duration,
            "dt_s": args.dt,
            "samples": args.samples,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "forward": {
            "lowering_seconds": forward_compilation.lowering_seconds,
            "compilation_seconds": forward_compilation.compilation_seconds,
            "execution": forward_execution.to_seconds_dict(),
        },
        "parameter_gradient": {
            "lowering_seconds": gradient_compilation.lowering_seconds,
            "compilation_seconds": gradient_compilation.compilation_seconds,
            "execution": gradient_execution.to_seconds_dict(),
            "objective": float(objective_value),
            "coupling_gradient": float(gradient),
        },
        "physics": {
            "all_samples_valid": bool(jnp.all(valid)),
            "finite_bold": bool(jnp.all(jnp.isfinite(bold))),
            "maximum_absolute_bold": float(jnp.max(jnp.abs(bold))),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
