from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import geophysics as geo
from phydrax.discretization import CellMesh


def _problem():
    mesh = CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    )
    exterior = np.flatnonzero(np.asarray(mesh.connectivity.boundary_faces))
    patches = tuple(
        geo.ElectrodePatch(f"electrode-{index}", [int(face)])
        for index, face in enumerate(exterior[:4])
    )
    currents = jnp.asarray(((1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -1.0)))
    receivers = jnp.asarray(
        ((1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -1.0), (1.0, 0.0, -1.0, 0.0))
    )
    survey = geo.ElectricalSurvey(patches, currents, receivers, jnp.asarray((0, 1, 0)))
    return geo.FinitePatchDCPlan(mesh, survey, batch_size=1).prepare()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")

    prepared = _problem()
    parameters = jnp.asarray((0.1, -0.2))
    direction = jnp.asarray((0.3, -0.4))
    cotangent = jnp.asarray((0.2, -0.5, 0.7))

    def evaluate(log_conductivity):
        def forward(value):
            return prepared.predict(jnp.exp(value))

        prediction, tangent = jax.jvp(forward, (log_conductivity,), (direction,))
        _, pullback = jax.vjp(forward, log_conductivity)
        adjoint = pullback(cotangent)[0]
        return prediction, tangent, adjoint

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(parameters), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(parameters), warmup=args.warmup, repeats=args.repeats
    )
    prediction, tangent, adjoint = result
    pairing = jnp.vdot(cotangent, tangent) - jnp.vdot(adjoint, direction)
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "cells": prepared.cell_count,
            "electrodes": len(prepared.plan.survey.patches),
            "sources": prepared.plan.survey.source_count,
            "measurements": prepared.plan.survey.measurement_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "identity": prepared.plan.plan_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "finite_prediction": bool(jnp.all(jnp.isfinite(prediction))),
            "minimum_voltage_V": float(jnp.min(prediction)),
            "maximum_voltage_V": float(jnp.max(prediction)),
            "jvp_vjp_pairing_residual": float(jnp.abs(pairing)),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
