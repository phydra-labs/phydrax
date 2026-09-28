#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned surface nanobubble (Lohse & Zhang 2015) versus free-bubble dissolution."""

from __future__ import annotations

from typing import Any

import numpy as np

import phydrax.bubble_dynamics as bd


def run() -> dict[str, Any]:
    tension, ambient = 0.072, 101325.0
    oversaturated = bd.GasSolutionProperties(2.0e-9, 0.6, 2.0, tension, ambient, 293.15)
    validity = bd.BubbleValidityPolicy(molecular_diameter=3.64e-10, tolman_length=2.0e-10)
    footprint = 1.0e-6
    pinned = bd.PinnedSurfaceBubblePlan(
        oversaturated, 1.0, np.linspace(0.0, 2.0e-3, 41), contact="pinned", validity=validity
    )
    equilibrium = pinned.equilibrium(footprint)
    evolving = bd.solve_surface_bubble(pinned.prepare(footprint, np.pi - np.radians(40.0)))

    undersaturated = bd.GasSolutionProperties(2.0e-9, 0.6, 0.9, tension, ambient, 293.15)
    unpinned = bd.PinnedSurfaceBubblePlan(
        undersaturated, -0.1, np.linspace(0.0, 1.0e-2, 41), contact="unpinned", validity=validity
    )
    dissolving = bd.solve_surface_bubble(unpinned.prepare(footprint, np.pi - np.radians(20.0)))

    free_radius = 0.5e-6
    free = bd.EpsteinPlessetPlan(
        oversaturated,
        np.linspace(0.0, 1.0, 21),
        route="full_history",
        initial_radius=free_radius,
        validity=validity,
        relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-10,
        maximum_steps=65536,
    )
    free_result = bd.solve_epstein_plesset(free.prepare(free_radius))
    if not (
        bool(evolving.successful) and bool(dissolving.successful) and bool(free_result.successful)
    ):
        raise RuntimeError("Surface-nanobubble workflow failed.")
    knudsen = evolving.evidence.max_knudsen
    if knudsen is None:
        raise RuntimeError("Knudsen evidence requires a declared molecular diameter.")
    final_angle = float(np.asarray(evolving.gas_contact_angle)[-1])
    return {
        "equilibrium_liquid_angle_deg": float(np.degrees(equilibrium.liquid_contact_angle)),
        "equilibrium_gas_angle_deg": float(np.degrees(equilibrium.gas_contact_angle)),
        "equilibrium_stable": bool(equilibrium.stable),
        "stability_derivative_per_s": float(equilibrium.stability_derivative),
        "pinned_final_gas_angle_deg": float(np.degrees(final_angle)),
        "pinned_status": bd.BubbleDynamicsStatus(int(evolving.status)).name,
        "pinned_max_knudsen": float(knudsen),
        "pinned_continuum_support": bool(evolving.evidence.continuum_support),
        "unpinned_status": bd.BubbleDynamicsStatus(int(dissolving.status)).name,
        "unpinned_lifetime_s": float(dissolving.evidence.lifetime),
        "free_bubble_status": bd.BubbleDynamicsStatus(int(free_result.status)).name,
        "free_bubble_lifetime_s": float(free_result.evidence.lifetime),
        "free_bubble_laplace_ratio": float(free_result.evidence.max_laplace_ratio),
    }


if __name__ == "__main__":
    print(run())
