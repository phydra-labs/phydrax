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

from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.geometric import evaluate_refractive_interface
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    DirectFresnelPlan,
    PlaneFieldSpace,
    prepare_direct_fresnel,
    propagate_angular_spectrum,
    propagate_direct_fresnel,
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

    fresnel_input_shape = (81, 83)
    fresnel_output_shape = (101, 103)
    fresnel_input_grid = TensorGridPlan(
        tuple(UniformAxisSpec(size) for size in fresnel_input_shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-4.0, -4.0), (4.0, 4.0))))
    fresnel_output_grid = TensorGridPlan(
        tuple(UniformAxisSpec(size) for size in fresnel_output_shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-6.0, -6.0), (6.0, 6.0))))
    fresnel_input_space = PlaneFieldSpace(
        fresnel_input_grid, RigidFrame.identity(3), "finite-window"
    )
    fresnel_output_space = PlaneFieldSpace(
        fresnel_output_grid, RigidFrame.identity(3), "finite-window"
    )
    fresnel_coordinates = fresnel_input_space.transverse_coordinates
    fresnel_field = ScalarPlaneField(
        fresnel_input_space,
        jnp.exp(-jnp.sum(fresnel_coordinates**2, axis=-1)),
        15.0,
        0.0,
    )
    fresnel_prepared = prepare_direct_fresnel(
        DirectFresnelPlan(
            fresnel_input_space,
            fresnel_output_space,
            maximum_sampling_phase_step=100.0,
            maximum_paraxial_angle=1.5,
            maximum_power_error=0.1,
        )
    )
    fresnel_propagate = eqx.filter_jit(propagate_direct_fresnel)
    fresnel_result, fresnel_cold = _timed(
        lambda: fresnel_propagate(fresnel_prepared, fresnel_field, 4.0, 20.0)
    )
    _, fresnel_warm = _timed(
        lambda: fresnel_propagate(fresnel_prepared, fresnel_field, 4.0, 20.0)
    )

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
            "fresnel_input_shape": fresnel_input_shape,
            "fresnel_output_shape": fresnel_output_shape,
            "fresnel_workspace_complex_elements": (
                fresnel_prepared.workspace_complex_elements_per_component
            ),
        },
        "timings_seconds": {
            "interface_cold": interface_cold,
            "interface_warm": interface_warm,
            "angular_spectrum_cold": wave_cold,
            "angular_spectrum_warm": wave_warm,
            "direct_fresnel_cold": fresnel_cold,
            "direct_fresnel_warm": fresnel_warm,
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
            "direct_fresnel_status": int(fresnel_result.status),
            "direct_fresnel_power_error": float(
                fresnel_result.evidence.relative_power_error
            ),
            "direct_fresnel_sampling_phase_step": float(
                fresnel_result.evidence.maximum_sampling_phase_step
            ),
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
