from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

import phydrax as phx
from phydrax.applications.cosmology._wave_finite_difference import (
    PeriodicWaveFiniteDifferencePlan,
    WaveContactSelfInteractionPlan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.size < 8 or arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError(
            "size >= 8, nonnegative warmup, and positive repeats are required"
        )

    shape = (arguments.size,) * 3
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    prepared = PeriodicWaveFiniteDifferencePlan(
        1.0,
        reduced_planck_constant=0.05,
        contact=WaveContactSelfInteractionPlan(
            0.01,
            maximum_dealiasing_defect=0.2,
        ),
    ).prepare(grid)
    coordinates = grid.points.reshape(shape + (3,))
    radius_squared = jnp.sum(
        jnp.minimum(jnp.abs(coordinates - 0.5), 1.0 - jnp.abs(coordinates - 0.5)) ** 2,
        axis=-1,
    )
    psi = jnp.exp(-radius_squared / (2.0 * 0.16**2)).astype(jnp.complex128)
    psi = psi / jnp.sqrt(jnp.sum(grid.quadrature_weights * jnp.abs(psi) ** 2))
    state = prepared.initialize(psi)
    potential = jnp.zeros(shape)
    function = eqx.filter_jit(
        lambda value: prepared.step(
            value,
            potential,
            1.0e-5,
            1.0e-5,
            end_coordinate_time=1.0e-5,
            contact_action_factor=1.0e-5,
        )
    )
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
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "operator": "weighted-cell-centred-second-order-negative-laplacian",
        },
        "identity": prepared.prepared_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "successful": bool(result.successful),
            "norm_relative_error": float(diagnostics.norm_relative_error),
            "cayley_relative_residual": float(diagnostics.cayley_relative_residual),
            "self_adjoint_residual": float(diagnostics.self_adjoint_residual),
            "maximum_kinetic_phase": float(diagnostics.maximum_kinetic_phase),
            "contact_input_truncation_defect": float(
                diagnostics.contact_input_truncation_defect
            ),
            "contact_aliasing_defect": float(diagnostics.contact_aliasing_defect),
            "contact_dealiasing_defect": float(diagnostics.contact_dealiasing_defect),
            "contact_energy_relative_error": float(
                diagnostics.contact_energy_relative_error
            ),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
