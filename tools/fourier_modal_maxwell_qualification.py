"""Deterministic Fourier-modal Maxwell qualification artifact."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


jax.config.update("jax_enable_x64", True)


def _interface_case() -> dict[str, float | int]:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    dielectric = fm.FrequencyMaxwellMaterial(4.0, material_id="dielectric")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (),
        fm.HomogeneousMaxwellPort(dielectric, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    expected_reflection = 1.0 / 9.0
    expected_transmission = 8.0 / 9.0
    return {
        "reflection": float(result.reflected_power[0]),
        "transmission": float(result.transmitted_power[0]),
        "reflection_error": float(abs(result.reflected_power[0] - expected_reflection)),
        "transmission_error": float(
            abs(result.transmitted_power[0] - expected_transmission)
        ),
        "status": int(result.status),
    }


def _propagation_case() -> dict[str, float | int]:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    material = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.0, 0.0)),
    )
    boundary = fm.prepare_layer_boundary(
        operator,
        0.125,
        fm.BoundaryCascadePolicy(
            doublings=9,
            initializer_order=6,
            paired_error=True,
            relative_tolerance=1e-8,
        ),
    )
    modal = fm.prepare_modal_boundary(operator, 0.125)
    difference = jnp.sqrt(
        jnp.sum(jnp.abs(boundary.a - modal.boundary.a) ** 2)
        + jnp.sum(jnp.abs(boundary.b - modal.boundary.b) ** 2)
        + jnp.sum(jnp.abs(boundary.c - modal.boundary.c) ** 2)
        + jnp.sum(jnp.abs(boundary.d - modal.boundary.d) ** 2)
    )
    return {
        "boundary_modal_difference": float(difference),
        "paired_error": float(boundary.diagnostics.paired_error),
        "constitutive_residual": float(operator.diagnostics.constitutive_residual),
        "modal_status": int(modal.status),
    }


def qualification() -> dict[str, object]:
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "interface": _interface_case(),
        "propagation": _propagation_case(),
        "reference": {
            "upstream_release": "v1.7.1",
            "upstream_commit": "e13d422cbb8b77820a5e375eb9f5c415be01b81e",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fourier_modal_maxwell_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
