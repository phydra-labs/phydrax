#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved capillary drop in a walled, viscous box, without and with gravity.

A liquid drop of radius 0.3 (7.2 cells per radius on a 24 x 24 grid with
no-slip walls) starts at rest from area-accurate volume fractions; densities
1000/1, viscosities 0.1/0.001, surface tension 1.  Without gravity the drop is
a static equilibrium: the pressure jump between drop and gas approaches the
Laplace value ``sigma / R`` and the residual (parasitic) velocity stays small.
The jump is measured on the absolute pressure two cells inside and outside
the interface.  With gravity ``(0, -1)`` (reduced gravity referenced to the top
of the box) the drop falls; the ledger pairs the gravity work with the
potential-energy change.  Each step is taken below the Brackbill capillary
limit; any refused step raises with its evidence.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


CELLS = 24
RADIUS = 0.3
SURFACE_TENSION = 1.0
LIQUID_DENSITY = 1000.0
GAS_DENSITY = 1.0
GRAVITY = 1.0
REFERENCE_PRESSURE = 1.0e5
STEP_SIZE = 0.01
STEPS = 10


def _drop_fraction(samples: int = 16) -> np.ndarray:
    """Liquid fraction of the centered drop from ``samples²`` points per cell."""

    offsets = (np.arange(samples) + 0.5) / (samples * CELLS)
    x = (np.arange(CELLS)[:, None] / CELLS + offsets).reshape(-1)
    inside = (x[:, None] - 0.5) ** 2 + (x[None, :] - 0.5) ** 2 < RADIUS**2
    return inside.reshape(CELLS, samples, CELLS, samples).mean(axis=(1, 3))


def _prepared(discretization: Any, *, gravity: bool) -> Any:
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=LIQUID_DENSITY,
        gas_density=GAS_DENSITY,
        liquid_viscosity=0.1,
        gas_viscosity=0.001,
        surface_tension=SURFACE_TENSION,
    )
    walls = tuple(
        phx.discretization.MACBoundarySide(axis, side, "no-slip")
        for axis in ("x", "y")
        for side in ("lower", "upper")
    )
    return phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=4000,
        gravity=(0.0, -GRAVITY) if gravity else None,
        hydrostatic_reference=(0.5, 1.0) if gravity else None,
        reference_pressure=REFERENCE_PRESSURE,
        wall_sides=walls,
    ).prepare()


def _advance(two_phase: Any, alpha: np.ndarray) -> tuple[Any, Any]:
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(alpha))
    )
    step = eqx.filter_jit(method.step)
    for index in range(STEPS):
        result = step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * STEP_SIZE),
            continuation,
            jnp.asarray(STEP_SIZE),
            None,
        )
        if not bool(result.successful):
            evidence = result.candidate_state.evidence
            refusal = {
                field.name: np.asarray(getattr(evidence, field.name)).item()
                for field in dataclasses.fields(evidence)
            }
            raise RuntimeError(f"Two-phase VOF step {index} was refused: {refusal}")
        continuation = result.accepted_state
    return result, continuation


def run() -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(CELLS),
            phx.discretization.UniformCellAxisSpec(CELLS),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    alpha = _drop_fraction()
    centers = np.asarray(discretization.cell_centers)
    radius = np.linalg.norm(centers - 0.5, axis=-1)
    inside = radius < RADIUS - 2.0 / CELLS
    outside = radius > RADIUS + 2.0 / CELLS

    static = _prepared(discretization, gravity=False)
    result, continuation = _advance(static, alpha)
    evidence = continuation.evidence
    view = static.view(continuation.state, continuation.pressure)
    pressure = np.asarray(view.absolute_pressure)
    _, accepted_inspection = phx.applications.two_phase_flow.two_phase_inspection_frames(
        static,
        result,
        time=jnp.asarray((STEPS - 1) * STEP_SIZE),
        step=STEPS - 1,
        step_size=jnp.asarray(STEP_SIZE),
        result_id="advanced-two-phase-vof-drop",
    )

    falling = _prepared(discretization, gravity=True)
    _, fallen = _advance(falling, alpha)
    fallen_view = falling.view(fallen.state, fallen.pressure)
    liquid = np.asarray(fallen_view.alpha) * np.asarray(discretization.cell_volumes)
    vertical = np.asarray(fallen_view.velocity[1])
    center_vertical = 0.5 * (vertical[:, :-1] + vertical[:, 1:])
    ledger = fallen.ledger
    return {
        "successful": True,
        "steps": STEPS,
        "capillary_step_limit": float(evidence.capillary_step_limit),
        "laplace_jump": SURFACE_TENSION / RADIUS,
        "measured_pressure_jump": float(
            pressure[inside].mean() - pressure[outside].mean()
        ),
        "curvature_pressure_jump": float(evidence.capillary_pressure_jump),
        "parasitic_velocity": float(evidence.parasitic_velocity),
        "curvature_status": {
            "valid": int(evidence.curvature_valid_count),
            "fallback": int(evidence.curvature_fallback_count),
            "underresolved": int(evidence.curvature_underresolved_count),
        },
        "liquid_volume": float(view.topology.liquid_volume),
        "liquid_volume_change": float(continuation.ledger.liquid_volume_change),
        "disk_area": np.pi * RADIUS**2,
        "interface_measure": float(view.topology.interface_measure),
        "circle_perimeter": 2.0 * np.pi * RADIUS,
        "viscous_dissipation": float(continuation.ledger.viscous_dissipation),
        "inspection_fields": tuple(
            field.name for field in accepted_inspection.frame.fields
        ),
        "falling_drop_velocity": float(np.sum(liquid * center_vertical) / np.sum(liquid)),
        "free_fall_velocity": -GRAVITY * STEPS * STEP_SIZE,
        "falling_maximum_face_speed": float(fallen.evidence.parasitic_velocity),
        "falling_liquid_volume_change": float(ledger.liquid_volume_change),
        "gravity_work": float(ledger.gravity_work),
        "gravitational_energy_change": float(ledger.gravitational_energy_change),
        "gravitational_energy_residual": float(ledger.gravitational_energy_residual),
    }


if __name__ == "__main__":
    print(run())
