from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_repeated
from flowjax.bijections import Affine

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.dimension <= 0:
        raise ValueError("samples and dimension must be positive")
    shape = (args.dimension,)
    source = phx.uq.CallableReducedPotential(
        lambda value: 0.5 * jnp.sum(value * value), shape, "benchmark-source"
    )
    target = phx.uq.CallableReducedPotential(
        lambda value: 0.5 * jnp.sum(((value - 1.0) / 0.7) ** 2),
        shape,
        "benchmark-target",
    )
    adapter = phx.uq.FlowJAXBijectionAdapter(
        Affine(jnp.ones(shape), jnp.full(shape, 0.7)),
        architecture_id="benchmark-affine",
    )
    problem = phx.uq.TargetedFreeEnergyProblem(
        source,
        target,
        phx.uq.TargetedMapPlan(adapter, shape, architecture_id="benchmark-affine"),
    )
    samples = jax.random.normal(jax.random.key(0), (args.samples, args.dimension))
    compiled = jax.jit(lambda values: phx.uq.evaluate_targeted_work(problem, values))
    result, elapsed = measure_repeated(
        lambda: compiled(samples), warmup=args.warmup, repeats=args.repeats
    )
    weights = jnp.exp(-result.forward_work + jnp.min(result.forward_work))
    effective = jnp.sum(weights) ** 2 / jnp.sum(weights * weights)
    payload = {
        "environment": capture_environment().to_dict(),
        "samples": args.samples,
        "dimension": args.dimension,
        "execution_seconds": elapsed.to_seconds_dict(),
        "successful": bool(result.valid),
        "work_standard_deviation": float(jnp.std(result.forward_work)),
        "effective_samples": float(effective),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
