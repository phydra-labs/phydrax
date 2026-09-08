"""Native weighted-SVD fit and physical-rate decoder cost, with held-out error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import electrophysiology as ep
from phydrax.domain import HyperRectangle
from phydrax.nn import population as pc


def target(points):
    return jnp.stack((points[..., 0] ** 2, jnp.sin(2.0 * points[..., 0])), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.neurons, args.samples, args.queries, args.repeats) < 1 or args.warmup < 0:
        raise ValueError(
            "Sizes and repeats must be positive; warmup must be nonnegative."
        )
    keys = jr.split(jr.key(42), 3)
    neuron = ep.LeakyIntegrateAndFire(0.2, 0.01, -65.0, -50.0, -62.0, refractory_ms=2.0)
    population = pc.prepare_lif_population(
        HyperRectangle([-1.0], [1.0]), neuron, args.neurons, key=keys[0]
    )
    training = pc.sample_population_points(population, args.samples, key=keys[1])
    queries = pc.sample_population_points(population, args.queries, key=keys[2])
    targets = target(training)

    fit = eqx.filter_jit(
        lambda points, values: pc.fit_population_decoder(population, points, values)
    )
    fit_compiled, fit_compilation = measure_lower_and_compile(
        lambda: fit.lower(training, targets), lambda lowered: lowered.compile()
    )
    code, fit_execution = measure_repeated(
        lambda: fit_compiled(training, targets), warmup=args.warmup, repeats=args.repeats
    )
    decode = jax.jit(lambda points: code(points))
    decode_compiled, decode_compilation = measure_lower_and_compile(
        lambda: decode.lower(queries), lambda lowered: lowered.compile()
    )
    _, decode_execution = measure_repeated(
        lambda: decode_compiled(queries), warmup=args.warmup, repeats=args.repeats
    )
    assessment = pc.assess_population_code(code, queries, target(queries))
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "neurons": args.neurons,
            "samples": args.samples,
            "queries": args.queries,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "fit": {
            "lowering_seconds": fit_compilation.lowering_seconds,
            "compilation_seconds": fit_compilation.compilation_seconds,
            "execution": fit_execution.to_seconds_dict(),
        },
        "rate_and_decode": {
            "lowering_seconds": decode_compilation.lowering_seconds,
            "compilation_seconds": decode_compilation.compilation_seconds,
            "execution": decode_execution.to_seconds_dict(),
        },
        "approximation": {
            "valid": bool(code.least_squares.valid),
            "rank": int(code.least_squares.rank),
            "silent_neurons": int(jnp.sum(code.silent_neurons)),
            "condition_number": float(code.least_squares.condition_number),
            "normal_equation_error": float(code.least_squares.normal_equation_error),
            "held_out_rmse": assessment.rmse.tolist(),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
