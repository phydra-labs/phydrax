from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

import phydrax as phx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 4
        or arguments.steps < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "size >= 4, positive steps/repeats, and nonnegative warmup are required"
        )

    cosmology = phx.applications.cosmology
    shape = (arguments.size,) * 3
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for count in shape),
        axis_names=("x", "y", "z"),
        field_name="psi",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in shape))
    background = cosmology.FLRWBackground(1.0, 1.0)
    policy = cosmology.WaveDarkMatterStepPolicy(
        maximum_phase_radians=2.0,
        minimum_de_broglie_cells=2.0,
        norm_relative_tolerance=1.0e-6,
    )
    schedule = 0.5 + 1.0e-4 * jnp.arange(arguments.steps + 1)
    prepared = cosmology.WaveDarkMatterPlan(
        1.0,
        schedule,
        gravitational_constant=0.05,
        reduced_planck_constant=0.03,
        step_policy=policy,
    ).prepare(space, background)

    coordinates = jnp.meshgrid(
        *(axis.nodes for axis in space.axes),
        indexing="ij",
    )
    radius_squared = sum(
        jnp.minimum(jnp.abs(axis - 0.5), 1.0 - jnp.abs(axis - 0.5)) ** 2
        for axis in coordinates
    )
    initial = jnp.exp(-radius_squared / (2.0 * 0.16**2)).astype(jnp.complex128)
    initial = initial / jnp.sqrt(
        jnp.sum(space.quadrature_weights * jnp.abs(initial) ** 2)
    )
    state = prepared.initialize(initial)
    function = eqx.filter_jit(prepared.solve)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(state),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(state),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )

    diagnostics = result.diagnostics
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "shape": shape,
            "steps": arguments.steps,
            "evaluation_shape": prepared.dealiasing.report.evaluation_shape,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "identity": prepared.prepared_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "completed": bool(result.successful),
            "accepted_steps": int(diagnostics.accepted_steps),
            "maximum_norm_relative_error": float(
                jnp.max(diagnostics.norm_relative_error)
            ),
            "maximum_poisson_relative_residual": float(
                jnp.max(diagnostics.poisson_relative_residual)
            ),
            "maximum_kinetic_phase": float(jnp.max(diagnostics.maximum_kinetic_phase)),
            "maximum_potential_phase": float(
                jnp.max(diagnostics.maximum_potential_phase)
            ),
            "maximum_de_broglie_nyquist_fraction": float(
                jnp.max(diagnostics.de_broglie_nyquist_fraction)
            ),
            "final_total_energy": float(diagnostics.total_energy[-1]),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
