#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.film_meshes import icosphere


_LUBRICATE = eqx.filter_jit(
    lambda prepared, state, step_size: prepared.step(state, step_size)
)


def test_draining_film_on_an_inflating_bubble_conserves_liquid() -> None:
    it = phx.interfacial_transport
    surface = it.prepare_film_surface(icosphere(2, 1e-2))
    prepared = it.SurfaceLubricationPlan(
        surface,
        mobility_law="immobile-free-film",
        surface_tension_n_m=0.03,
        viscosity_pa_s=1e-3,
        density_kg_m3=1000.0,
        gravity_m_s2=(0.0, 0.0, -9.81),
    ).prepare()
    state = prepared.initial_state(1e-6)
    total = float(jnp.sum(state.liquid_volume_m3))
    for _ in range(3):
        current = prepared.plan.surface
        motion = it.SurfaceMeshMotion(current, 1.02 * current.coordinates, 10.0)
        assert bool(motion.evidence.valid)
        # The film is carried by the inflating surface: a Lagrangian mesh.
        moved = motion.transport(state.liquid_volume_m3, motion.mesh_velocity)
        assert bool(moved.accepted)
        prepared = prepared.with_surface(motion.target)
        state = it.SurfaceLubricationState(moved.content, topology_id=state.topology_id)
        result = _LUBRICATE(prepared, state, 10.0)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        assert bool(result.evidence.positivity_guaranteed)
        state = result.state
    assert int(prepared.plan.surface.geometry_revision) == 3
    np.testing.assert_allclose(float(jnp.sum(state.liquid_volume_m3)), total, rtol=1e-13)
    height = np.asarray(prepared.plan.surface.coordinates[:, 2])
    thickness = np.asarray(prepared.thickness(state))
    # Inflation thins the film; gravity drains the top toward the bottom.
    assert thickness.max() < 1e-6
    assert thickness[np.argmax(height)] < thickness[np.argmin(height)]
