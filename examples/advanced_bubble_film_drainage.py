#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gravity drainage of a soap-bubble film and its black-film equilibrium.

A 1 um symmetric film with immobile interfaces covers a 1 cm sphere. Gravity
drains liquid toward the bottom through the lubrication route while capillary
pressure, a DLVO disjoining pressure and the positivity/conservation evidence
are tracked. The same DLVO law gives the common-black-film thickness in
equilibrium with the bubble Laplace suction.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _icosphere(level: int, radius: float) -> phx.geometry.TriangleMesh:
    ratio = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, ratio, 0),
            (1, ratio, 0),
            (-1, -ratio, 0),
            (1, -ratio, 0),
            (0, -1, ratio),
            (0, 1, ratio),
            (0, -1, -ratio),
            (0, 1, -ratio),
            (ratio, 0, -1),
            (ratio, 0, 1),
            (-ratio, 0, -1),
            (-ratio, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoints = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        count = faces.shape[0]
        ab, bc, ca = (
            vertices.shape[0] + inverse[k * count : (k + 1) * count] for k in range(3)
        )
        a, b, c = faces.T
        faces = np.concatenate(
            (
                np.stack((a, ab, ca), 1),
                np.stack((b, bc, ab), 1),
                np.stack((c, ca, bc), 1),
                np.stack((ab, bc, ca), 1),
            )
        )
        vertices = np.concatenate((vertices, midpoints))
    return phx.geometry.TriangleMesh(radius * vertices, faces.astype(np.int32))


_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))


def run() -> Any:
    it = phx.interfacial_transport
    radius, thickness, tension = 1e-2, 1e-6, 0.03
    dlvo = it.CompositeDisjoiningPressure(
        (
            it.VanDerWaalsDisjoiningPressure(4e-20),
            it.DoubleLayerDisjoiningPressure(1.0, 0.05, 298.15, 78.5),
        )
    )
    surface = it.prepare_film_surface(_icosphere(3, radius))
    prepared = it.SurfaceLubricationPlan(
        surface,
        mobility_law="immobile-free-film",
        surface_tension_n_m=tension,
        viscosity_pa_s=1e-3,
        density_kg_m3=1000.0,
        gravity_m_s2=(0.0, 0.0, -9.81),
        disjoining=dlvo,
        rupture_thickness_m=3e-8,
    ).prepare()
    state = prepared.initial_state(thickness)
    initial_volume = float(jnp.sum(state.liquid_volume_m3))
    energy_changes = []
    for _ in range(40):
        result = _STEP(prepared, state, 50.0)
        if not bool(result.accepted):
            raise RuntimeError(
                f"Drainage step rejected with status {int(result.status)}."
            )
        energy_changes.append(float(result.evidence.energy_change_j))
        state = result.state
    height = np.asarray(surface.coordinates[:, 2])
    film = np.asarray(prepared.thickness(state))
    black_film = it.black_film_equilibrium(dlvo, 4.0 * tension / radius, (8.5e-9, 1e-6))
    return {
        "vertices": surface.topology.num_vertices,
        "conductance_admissible": bool(surface.evidence.conductance_admissible),
        "drained_time_s": 40 * 50.0,
        "top_thickness_m": float(film[np.argmax(height)]),
        "bottom_thickness_m": float(film[np.argmin(height)]),
        "volume_residual_relative": abs(
            float(jnp.sum(state.liquid_volume_m3)) - initial_volume
        )
        / initial_volume,
        "positivity_guaranteed": bool(result.evidence.positivity_guaranteed),
        "energy_decreased_every_step": all(change < 0.0 for change in energy_changes),
        "rupture_flagged": bool(jnp.any(result.evidence.rupture_mask)),
        "black_film_thickness_m": float(black_film.thickness_m),
        "black_film_stable": bool(black_film.stable),
        "black_film_status": int(black_film.status),
    }


if __name__ == "__main__":
    print(run())
