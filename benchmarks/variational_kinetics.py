from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_repeated

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 2 or args.features <= 0:
        raise ValueError(
            "samples and features must be positive with at least three samples"
        )
    rates = jnp.linspace(0.4, 0.99, args.features)
    forcing = jax.random.normal(jax.random.key(0), (args.samples, args.features)) * 0.02

    def step(state, noise):
        following = rates * state + noise
        return following, state

    _, states = jax.lax.scan(step, jnp.ones((args.features,)), forcing)
    layout = phx.dynamics.StateLayout((args.features,))
    data = phx.dynamics.TrajectoryData(
        jnp.arange(args.samples, dtype=states.dtype),
        states,
        state_layout=layout,
        source_id="variational-kinetics-benchmark",
    )
    library = phx.dynamics.identification.CustomFeatureLibrary(
        lambda values, inputs: values,
        state_layout=layout,
        feature_names=tuple(f"x{index}" for index in range(args.features)),
        library_id="benchmark-identity",
    )
    vamp, vamp_time = measure_repeated(
        lambda: phx.dynamics.identification.fit_vamp(
            data,
            library,
            n_modes=min(4, args.features),
            regularization=1.0e-5,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    tica, tica_time = measure_repeated(
        lambda: phx.dynamics.identification.fit_tica(
            data,
            n_modes=min(4, args.features),
            regularization=1.0e-5,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "samples": args.samples,
        "features": args.features,
        "vamp_seconds": vamp_time.to_seconds_dict(),
        "tica_seconds": tica_time.to_seconds_dict(),
        "vamp_valid": bool(vamp.valid),
        "tica_valid": bool(tica.valid),
        "vamp_score": float(vamp.diagnostics.score),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
