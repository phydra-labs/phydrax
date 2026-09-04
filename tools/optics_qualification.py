"""Deterministic geometric and plane-wave optics qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.geometric import evaluate_refractive_interface
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    fraunhofer_psf,
    FraunhoferImagingPlan,
    PlaneFieldSpace,
    propagate_angular_spectrum,
    ScalarPlaneField,
)


jax.config.update("jax_enable_x64", True)


def _interface_case() -> dict[str, float | int | bool]:
    result = evaluate_refractive_interface(
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )
    expected_reflectance = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
    return {
        "status": int(result.status),
        "transmission_valid": bool(result.transmission_valid),
        "reflectance_error": float(
            jnp.max(jnp.abs(result.reflectance - expected_reflectance))
        ),
        "energy_balance_error": float(result.energy_balance_error),
    }


def _periodic_space(size: int) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (FourierAxisSpec(size), FourierAxisSpec(size)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-jnp.pi, -jnp.pi), (jnp.pi, jnp.pi))))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _angular_spectrum_case() -> dict[str, float | int | bool]:
    space = _periodic_space(32)
    transverse_wavevector = jnp.asarray((2.0, -3.0))
    values = jnp.exp(
        1j * jnp.sum(space.transverse_coordinates * transverse_wavevector, axis=-1)
    )
    field = ScalarPlaneField(space, values, 13.0, 0.0)
    distance = 0.37
    medium_wavenumber = 8.0
    result = propagate_angular_spectrum(
        AngularSpectrumPlan().prepare(space),
        field,
        distance,
        medium_wavenumber,
    )
    longitudinal = jnp.sqrt(medium_wavenumber**2 - jnp.sum(transverse_wavevector**2))
    expected = values * jnp.exp(1j * longitudinal * distance)
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "maximum_field_error": float(jnp.max(jnp.abs(result.field.values - expected))),
        "leakage_fraction": float(result.leakage_fraction),
        "cropped_energy": float(result.cropped_energy),
    }


def _finite_space(size: int, lower: float, upper: float, z: float) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (UniformAxisSpec(size), UniformAxisSpec(size)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((lower, lower), (upper, upper))))
    return PlaneFieldSpace(
        grid,
        RigidFrame(jnp.eye(3), jnp.asarray((0.0, 0.0, z))),
        "finite-window",
    )


def _fraunhofer_case() -> dict[str, float | int | bool]:
    pupil_space = _finite_space(33, -0.5, 0.5, 0.0)
    image_space = _finite_space(51, -4.0, 4.0, 1.0)
    radius = jnp.sqrt(jnp.sum(pupil_space.transverse_coordinates**2, axis=-1))
    field = ScalarPlaneField(pupil_space, (radius <= 0.5).astype(float), 1.0, 0.0)
    prepared = FraunhoferImagingPlan(
        pupil_space,
        image_space,
        1.0,
        2.0 * jnp.pi,
        1.0,
    ).prepare()
    result = fraunhofer_psf(prepared, field)
    integrated = jnp.sum(result.plane.values * image_space.area_weights)
    return {
        "status": int(result.status),
        "valid": bool(result.valid),
        "integrated_power_error": float(jnp.abs(integrated - 1.0)),
        "samples_per_airy_radius": float(result.sampling.samples_per_airy_radius),
    }


def qualify() -> dict[str, object]:
    cases = {
        "refractive_interface": _interface_case(),
        "angular_spectrum": _angular_spectrum_case(),
        "fraunhofer": _fraunhofer_case(),
    }
    accepted = (
        cases["refractive_interface"]["reflectance_error"] < 1.0e-12
        and cases["refractive_interface"]["energy_balance_error"] < 1.0e-12
        and cases["angular_spectrum"]["maximum_field_error"] < 5.0e-5
        and cases["fraunhofer"]["integrated_power_error"] < 5.0e-6
    )
    return {"accepted": bool(accepted), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optics_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
