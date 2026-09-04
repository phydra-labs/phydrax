"""Core geometric and plane-wave optics compilation and execution benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.geometry import RigidFrame
from phydrax.optics.geometric import evaluate_refractive_interface
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    PlaneFieldSpace,
    propagate_angular_spectrum,
    ScalarPlaneField,
)


jax.config.update("jax_enable_x64", True)


def _timed(function):
    start = time.perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def benchmark() -> dict[str, object]:
    ray_count = 65_536
    angles = jnp.linspace(-0.5, 0.5, ray_count)
    directions = jnp.stack(
        (jnp.sin(angles), jnp.zeros_like(angles), jnp.cos(angles)), axis=-1
    )
    normal = jnp.asarray((0.0, 0.0, 1.0))
    interface = eqx.filter_jit(evaluate_refractive_interface)
    interface_result, interface_cold = _timed(
        lambda: interface(directions, normal, 1.0, 1.5)
    )
    _, interface_warm = _timed(lambda: interface(directions, normal, 1.0, 1.5))

    shape = (128, 128)
    grid = TensorGridPlan(
        tuple(FourierAxisSpec(size) for size in shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-jnp.pi, -jnp.pi), (jnp.pi, jnp.pi))))
    space = PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")
    coordinates = space.transverse_coordinates
    field = ScalarPlaneField(
        space,
        jnp.exp(-(coordinates[..., 0] ** 2 + coordinates[..., 1] ** 2)),
        13.0,
        0.0,
    )
    prepared = AngularSpectrumPlan().prepare(space)
    propagate = eqx.filter_jit(propagate_angular_spectrum)
    propagated, wave_cold = _timed(lambda: propagate(prepared, field, 0.5, 12.0))
    _, wave_warm = _timed(lambda: propagate(prepared, field, 0.5, 12.0))

    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "problems": {
            "ray_count": ray_count,
            "wave_shape": shape,
            "wave_complex_elements_per_component": (
                prepared.workspace_complex_elements_per_component
            ),
        },
        "timings_seconds": {
            "interface_cold": interface_cold,
            "interface_warm": interface_warm,
            "angular_spectrum_cold": wave_cold,
            "angular_spectrum_warm": wave_warm,
        },
        "evidence": {
            "interface_maximum_energy_error": float(
                jnp.max(interface_result.energy_balance_error)
            ),
            "interface_success_fraction": float(
                jnp.mean(interface_result.transmission_valid)
            ),
            "angular_spectrum_status": int(propagated.status),
            "angular_spectrum_leakage": float(propagated.leakage_fraction),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/optics.json"))
    arguments = parser.parse_args()
    payload = benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
