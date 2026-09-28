#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved two-bubble contact with multi-marker and film-drainage gating."""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx


CELLS = 24


def _gas_disk(center: tuple[float, float], radius: float) -> np.ndarray:
    samples = 4
    lower = np.arange(CELLS) / CELLS
    offsets = (np.arange(samples) + 0.5) / (samples * CELLS)
    content = np.zeros((CELLS, CELLS), dtype=np.float64)
    for offset_x in offsets:
        for offset_y in offsets:
            x, y = np.meshgrid(lower + offset_x, lower + offset_y, indexing="ij")
            content += (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
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
        liquid_viscosity=0.0,
        gas_viscosity=0.0,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization, material, maximum_iterations=1000
    ).prepare()
    first = _gas_disk((0.37, 0.5), 0.1)
    second = _gas_disk((0.63, 0.5), 0.1)
    alpha = 1.0 - first - second
    color = np.where(second > first, 1, 0)
    identity = phx.applications.two_phase_flow.BubbleComponentPlan(
        two_phase, component_capacity=6, maximum_rounds=64, pair_capacity=16
    )
    markers = phx.applications.two_phase_flow.MultiMarkerPlan(
        two_phase,
        marker_capacity=3,
        component_capacity=6,
        pair_capacity=16,
        proximity_radius=3,
    )
    drainage = phx.applications.two_phase_flow.FilmDrainageCoalescencePlan(
        regime="immobile",
        geometry="planar",
        pair_capacity=16,
        id_upper_bound=1_000_000,
        continuous_viscosity=1.0e-3,
        surface_tension=0.05,
        initial_film_thickness=1.0e-6,
        critical_thickness=5.0e-8,
    )
    plan = phx.applications.two_phase_flow.BubblyFlowPlan(
        two_phase,
        identity,
        markers=markers,
        coalescence=drainage,
        near_contact_pressure=1.0e-3,
    )
    bubbles = plan.initial_state(jnp.asarray(alpha), color=jnp.asarray(color))
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, bubbles=plan
    )
    continuation = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(alpha)), bubbles=bubbles
    )
    run_result = phx.applications.two_phase_flow.run_bubbly_flow(
        method, continuation, steps=2, step_size=1.0e-4
    )
    if not run_result.completed:
        raise RuntimeError(
            f"Bubble-coalescence run failed with {run_result.status.name}: "
            f"{run_result.message}"
        )
    final = run_result.continuation
    evidence = final.bubble_evidence
    ledger = final.ledger
    return {
        "successful": run_result.completed,
        "status": run_result.status.name,
        "accepted_steps": run_result.steps,
        "journal_events": len(run_result.journal.records),
        "marker_sum_residual": (
            None if evidence is None else float(evidence.marker_sum_residual)
        ),
        "merge_proposals": (None if evidence is None else int(evidence.merge_proposals)),
        "near_contact_work": (
            None if evidence is None else float(evidence.near_contact_work)
        ),
        "contact_work_ledger": float(ledger.contact_work),
        "pressure_work_ledger": float(ledger.pressure_work),
    }


if __name__ == "__main__":
    print(run())
