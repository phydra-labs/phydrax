#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Material-scalar Marangoni migration compared with Young–Goldstein–Block.

The volume scalar is temperature content, not a VOF surface-surfactant field.
The short coarse run is a smoke comparison; the qualification tool owns the
resolved campaign and reports the discrepancy from the low-Re reference.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


CELLS = 24
RADIUS = 0.2
VISCOSITY = 1.0
SURFACE_TENSION_SLOPE = -1.0e-2
TEMPERATURE_GRADIENT = 1.0
STEP_SIZE = 1.0e-3
STEPS = 4


def _disk() -> np.ndarray:
    samples = 4
    lower = np.arange(CELLS) / CELLS
    offsets = (np.arange(samples) + 0.5) / (samples * CELLS)
    content = np.zeros((CELLS, CELLS), dtype=np.float64)
    for offset_x in offsets:
        for offset_y in offsets:
            x, y = np.meshgrid(lower + offset_x, lower + offset_y, indexing="ij")
            content += (x - 0.5) ** 2 + (y - 0.5) ** 2 < RADIUS**2
    return content / samples**2


def run() -> dict[str, Any]:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(CELLS, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1.0,
        gas_density=1.0,
        liquid_viscosity=VISCOSITY,
        gas_viscosity=VISCOSITY,
    )
    law = phx.discretization.LinearSurfaceTensionLaw(0.05, SURFACE_TENSION_SLOPE, 0.5)
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=1000,
        surface_tension_law=law,
        surface_tension_scalar="temperature",
    ).prepare()
    alpha = jnp.asarray(_disk())
    temperature = TEMPERATURE_GRADIENT * discretization.cell_centers[..., 0]
    state = two_phase.initial_state(alpha, material_scalars={"temperature": temperature})
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(state)
    initial_content = float(jnp.sum(state.material_scalar_content["temperature"]))
    for index in range(STEPS):
        result = method.step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * STEP_SIZE),
            continuation,
            jnp.asarray(STEP_SIZE),
            None,
        )
        if not bool(result.successful):
            raise RuntimeError(f"Thermocapillary step {index} was refused.")
        continuation = result.accepted_state
    velocity = two_phase.velocity(continuation.state)[0]
    cell_velocity = 0.5 * (velocity + jnp.roll(velocity, -1, axis=0))
    measured = float(jnp.sum(cell_velocity * alpha) / jnp.sum(alpha))
    reference = (
        -SURFACE_TENSION_SLOPE
        * TEMPERATURE_GRADIENT
        * RADIUS
        / (4.0 * (VISCOSITY + VISCOSITY))
    )
    final_content = float(
        jnp.sum(continuation.state.material_scalar_content["temperature"])
    )
    evidence = continuation.evidence
    return {
        "successful": True,
        "cells": CELLS,
        "steps": STEPS,
        "measured_velocity": measured,
        "young_goldstein_block_velocity": reference,
        "velocity_ratio": measured / reference,
        "direction_agrees": measured * reference > 0.0,
        "material_scalar_content_residual": final_content - initial_content,
        "reported_conservation_residual": (
            None if evidence is None else float(evidence.material_scalar_residual)
        ),
        "surface_gamma_claim": False,
    }


if __name__ == "__main__":
    print(run())
