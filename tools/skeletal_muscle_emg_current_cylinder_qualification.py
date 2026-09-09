#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implementation oracles only: cable cosine and independent radial finite volumes.

No anatomy, physiological waveform, source executable, or held-out validation
claim is made by these manufactured numerical experiments.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from phydrax.applications.skeletal_muscle.electromyography import (
    Farina2004CylindricalConductorPlan,
    PereiraBotelho2019FiberCurrentPlan,
)
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
    SkeletalFiberBundleState,
)


def make_source(nodes: int = 33, fibers: int = 1):
    """Explicit manufactured inputs, never anatomical package defaults."""
    ids = tuple(f"manufactured-fiber-{i}" for i in range(fibers))
    schedule = PrescribedFiberStimulusSchedule(
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        jnp.zeros((0, fibers, nodes), dtype=bool),
    )
    fiber = SkeletalFiberBundlePlan(
        ids, nodes, jnp.full((fibers,), 40.0), jnp.full((fibers,), 0.05), schedule
    ).prepare()
    positions = jnp.zeros((fibers, nodes, 3)).at[..., 0].set(0.015)
    positions = positions.at[..., 2].set(jnp.linspace(-0.02, 0.02, nodes))
    current = PereiraBotelho2019FiberCurrentPlan(
        ids,
        positions,
        jnp.full((fibers,), 25e-6),
        geometry_source_id="manufactured-cosine-qualification-not-anatomy",
        geometry_license="CC0-1.0 manufactured numerical input",
    ).prepare(fiber)
    state = fiber.initialize()
    voltage = 40.0 * jnp.cos(jnp.linspace(0, 2 * jnp.pi, nodes))
    snapshot = SkeletalFiberBundleState(1.0, state.values.at[..., 0].set(voltage))
    prior = current.initialize()
    candidate = current.propose(
        prior,
        snapshot,
        fiber_prepared_id=fiber.prepared_id,
        geometry_id=current.plan.geometry_id,
    )
    return fiber, current, snapshot, candidate.commit(prior, snapshot)


def make_conductor(current, *, angular=2, axial=3):
    return Farina2004CylindricalConductorPlan(
        [0.03, 0.035, 0.04],
        [0.1, 0.5, 0.05, 1.0],
        [[0.0, -0.006], [0.15, 0.009]],
        [[0.002, 0.003], [0.002, 0.003]],
        [[1.0, -1.0]],
        ("e0", "e1"),
        ("bipolar",),
        axial_period_m=0.2,
        longitudinal_modes=axial,
        angular_modes=angular,
        coordinate_frame_id="manufactured-cylinder-z",
        material_source_id="Farina-2004-Fig3-conductivities-only-no-bone-geometry-replay",
        electrode_source_id="manufactured-passive-rectangles",
    ).prepare(current, coordinate_frame_id="manufactured-cylinder-z")


def radial_finite_volume_oracle(radial_cells: int, wavenumber: float) -> float:
    """Solve -(r sigma_r phi')' + r sigma_z k² phi = delta(r-R).

    Axisymmetric mode only. Both radial fluxes vanish. Tissue interfaces
    coincide with faces, and face conductance uses the series resistance
    of the two half cells. This does not evaluate any Bessel function.
    """
    edges = np.linspace(0, 0.04, radial_cells + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    h = edges[1]
    radial_sigma = np.where(centers < 0.03, 0.1, np.where(centers < 0.035, 0.05, 1.0))
    axial_sigma = np.where(centers < 0.03, 0.5, np.where(centers < 0.035, 0.05, 1.0))
    face_sigma = 2 / (1 / radial_sigma[:-1] + 1 / radial_sigma[1:])
    conductance = edges[1:-1] * face_sigma / h
    diagonal = axial_sigma * wavenumber**2 * (edges[1:] ** 2 - edges[:-1] ** 2) / 2
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    matrix = np.zeros((3, radial_cells))
    matrix[0, 1:] = -conductance
    matrix[1] = diagonal
    matrix[2, :-1] = -conductance
    right = np.zeros(radial_cells)
    source = 0.015
    left = np.searchsorted(centers, source) - 1
    fraction = (source - centers[left]) / h
    right[left] = 1 - fraction
    right[left + 1] = fraction
    solution = solve_banded((1, 1), matrix, right)
    return float(solution[-1])


def qualify() -> dict[str, object]:
    source_rows = []
    for nodes in (17, 33, 65):
        _, prepared, _, output = make_source(nodes)
        sigma = float(prepared.intracellular_conductivity_S_per_m[0])
        area = np.pi * float(prepared.plan.radius_m[0]) ** 2
        exact = (
            -sigma
            * area
            * (2 * np.pi / 0.04) ** 2
            * np.asarray(output.membrane_voltage_V)
        )
        error = np.linalg.norm(
            np.asarray(output.transmembrane_line_current_A_per_m) - exact
        ) / np.linalg.norm(exact)
        source_rows.append(
            {
                "nodes": nodes,
                "relative_line_current_error": float(error),
                "net_current_A": float(jnp.sum(output.transmembrane_current_A)),
            }
        )
    _, source, _, accepted = make_source()
    conductor = make_conductor(source, angular=0)
    k = 2 * np.pi / conductor.plan.axial_period_m
    # Storage uses n >= 0 and kz > 0; Fourier symmetry restores the other modes.
    reference = float(conductor.radial_transfer_ohm_m[0, 0, 0])
    radial_rows = []
    for cells in (64, 128, 256, 512):
        oracle = radial_finite_volume_oracle(cells, k)
        radial_rows.append(
            {
                "radial_cells": cells,
                "surface_green_ohm_m": oracle,
                "relative_error": abs(oracle - reference) / abs(reference),
            }
        )
    prior = conductor.initialize()
    candidate = conductor.propose(prior, accepted)
    output = candidate.commit(prior, accepted)
    source_order = np.log2(
        source_rows[-2]["relative_line_current_error"]
        / source_rows[-1]["relative_line_current_error"]
    )
    radial_order = np.log2(
        radial_rows[-2]["relative_error"] / radial_rows[-1]["relative_error"]
    )
    passed = (
        bool(candidate.evidence.successful)
        and source_order > 1.8
        and radial_order > 1.5
        and radial_rows[-1]["relative_error"] < 0.002
        and all(abs(row["net_current_A"]) < 1e-14 for row in source_rows)
    )
    return {
        "qualification": "source-current-and-Farina-cylinder-numerical-oracles",
        "passed": bool(passed),
        "source_cosine_refinement": source_rows,
        "source_observed_order": float(source_order),
        "radial_finite_volume_refinement": radial_rows,
        "radial_observed_order": float(radial_order),
        "native_surface_green_ohm_m": reference,
        "interface_relative_residual": float(conductor.interface_relative_residual),
        "lead_voltage_V": np.asarray(output.lead_voltage_V).tolist(),
        "source_prepared_id": source.prepared_id,
        "conductor_prepared_id": conductor.prepared_id,
        "claim_scope": "manufactured SI current conservation and cylindrical PDE numerics only",
        "unclosed_gates": [
            "Farina published waveform digitization/external executable replay",
            "same-anatomy MRI/DTI/mesh/material/contact bundle",
            "intramuscular contact/encapsulation parameters",
            "committed registered moving geometry and conductivity transport law",
            "held-out subject/electrode recordings",
        ],
    }


def main():
    result = qualify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
