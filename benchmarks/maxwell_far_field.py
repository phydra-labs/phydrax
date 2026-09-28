from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx


mx = phx.solver.maxwell
_LENGTH = 2.0
_OMEGA0 = 4.0 * np.pi
_TAU = 0.15
_T0 = 3.0 * _TAU


def _envelope(time: Any, args: Any) -> Any:
    del args
    return jnp.exp(-(((time - _T0) / _TAU) ** 2)) * jnp.sin(_OMEGA0 * (time - _T0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=36)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--polar", type=int, default=16)
    parser.add_argument("--azimuthal", type=int, default=32)
    parser.add_argument("--stop-time", type=float, default=1.6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.size < 16
        or args.frequencies <= 0
        or args.polar <= 0
        or args.azimuthal <= 0
        or args.stop_time <= 0.0
        or args.warmup < 0
        or args.repeats <= 0
    ):
        raise ValueError(
            "size >= 16, positive frequencies/directions/stop time/repeats and "
            "nonnegative warmup are required"
        )

    count = args.size
    spacing = _LENGTH / count
    center = count // 2
    half = max(2, int(0.3 / spacing))
    omegas = jnp.linspace(0.8 * _OMEGA0, 1.2 * _OMEGA0, args.frequencies)

    def prepare() -> Any:
        grid = phx.discretization.TensorGridPlan(
            tuple(phx.discretization.UniformCellAxisSpec(count) for _ in range(3)),
            axis_names=("x", "y", "z"),
        ).prepare(jnp.asarray([[0.0] * 3, [_LENGTH] * 3]))
        bridge = phx.discretization.StructuredCochainBridge(grid)
        edge = bridge.orientation_offsets[1][2] + int(
            np.ravel_multi_index(
                (center, center, center), bridge.orientation_shapes[1][2]
            )
        )
        acquisition = mx.MaxwellSpectralAcquisition(
            omegas, sign="positive", measure="time-integral", stop_time=args.stop_time
        )
        box = mx.MaxwellHuygensBoxPlan(
            bridge,
            (center - half,) * 3,
            (center + half,) * 3,
            acquisition,
            mx.HomogeneousMaxwellExterior(),
        )
        return phx.solver.CompatibleMaxwellPlan(
            bridge,
            observers=(box,),
            sources=(
                mx.MaxwellElectricCurrentSourcePlan(
                    jnp.asarray([edge]), jnp.asarray([1.0]), envelope=_envelope
                ),
            ),
        ).prepare()

    runtime, preparation_seconds = measure_host(prepare)
    sampler = runtime.observers[0]
    if not isinstance(sampler, mx.PreparedMaxwellHuygensBox):
        raise TypeError("The prepared runtime must carry the Huygens box sampler.")
    step = 0.9 * float(runtime.stable_dt)
    steps = int(np.ceil(args.stop_time / step))
    initial = runtime.initialize()

    def run() -> Any:
        # solve_compatible_maxwell validates the step on the host and scans a
        # compiled step, so the first call carries tracing and compilation.
        result = mx.solve_compatible_maxwell(runtime, initial, 0.0, step, steps)
        return result.final_state.observations[0]

    _, solve_first_call_seconds = measure_synchronized(run)
    observation, solve_execution = measure_repeated(
        run, warmup=args.warmup, repeats=args.repeats
    )
    phasors = sampler.surface_phasors(observation)

    nodes, weights = np.polynomial.legendre.leggauss(args.polar)
    phi = 2.0 * np.pi * (np.arange(args.azimuthal) + 0.5) / args.azimuthal
    cos_theta, phi_grid = np.meshgrid(nodes, phi, indexing="ij")
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    directions = np.stack(
        (sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), cos_theta),
        axis=-1,
    ).reshape(-1, 3)
    quadrature = (
        weights[:, None] * np.full((1, args.azimuthal), 2.0 * np.pi / args.azimuthal)
    ).reshape(-1)
    far_field, far_field_preparation_seconds = measure_host(
        lambda: mx.MaxwellFarFieldPlan(
            directions, jnp.asarray([0.0, 0.0, 1.0]), sampler.exterior
        )
    )

    def transform(value: Any) -> Any:
        result = far_field.evaluate(value)
        return result.spectral_energy, mx.spectral_poynting_energy(value)

    evaluate = jax.jit(transform)
    compiled_far, far_compilation = measure_lower_and_compile(
        lambda: evaluate.lower(phasors), lambda lowered: lowered.compile()
    )
    (spectral_energy, surface_energy), far_execution = measure_repeated(
        lambda: compiled_far(phasors), warmup=args.warmup, repeats=args.repeats
    )

    moment = (
        np.exp(1j * np.asarray(omegas) * _T0)
        * (_TAU * np.sqrt(np.pi) / 2j)
        * (
            np.exp(-(_TAU**2) * (np.asarray(omegas) + _OMEGA0) ** 2 / 4.0)
            - np.exp(-(_TAU**2) * (np.asarray(omegas) - _OMEGA0) ** 2 / 4.0)
        )
        * spacing**2
    )
    exact = np.asarray(omegas) ** 2 * np.abs(moment) ** 2 / (6.0 * np.pi**2)
    radiated = np.asarray(spectral_energy) @ quadrature
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "shape": [count, count, count],
            "steps": steps,
            "surface_cells": sampler.surface_count,
            "frequencies": args.frequencies,
            "directions": int(directions.shape[0]),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "identity": runtime.prepared_id,
        "preparation": {
            "runtime_seconds": preparation_seconds,
            "far_field_seconds": far_field_preparation_seconds,
        },
        "compilation": {
            "solve_first_call_seconds": solve_first_call_seconds,
            "far_field_lowering_seconds": far_compilation.lowering_seconds,
            "far_field_compilation_seconds": far_compilation.compilation_seconds,
        },
        "execution": {
            "solve": solve_execution.to_seconds_dict(),
            "far_field": far_execution.to_seconds_dict(),
        },
        "physics": {
            "far_field_energy_over_exact": (radiated / exact).tolist(),
            "surface_energy_over_exact": (np.asarray(surface_energy) / exact).tolist(),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
