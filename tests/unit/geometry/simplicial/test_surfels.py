from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_simplicial_surfel_preserves_face_measure_and_orientation() -> None:
    surface = phx.geometry.TriangleSurface(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        jnp.asarray(((0, 1, 2),)),
        source_id="single-triangle-surfels",
    )
    prepared = phx.geometry.SimplicialSurfelPlan(
        surface, footprint_area_ratio=1.25
    ).prepare()
    assert bool(prepared.geometry.evidence.successful)
    np.testing.assert_allclose(prepared.geometry.position, [[1.0 / 3.0] * 2 + [0.0]])
    np.testing.assert_allclose(prepared.geometry.normal, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(prepared.geometry.physical_surface_weight, [0.5])
    np.testing.assert_allclose(prepared.geometry.footprint_measure, [0.625])
    assert (
        prepared.geometry.certificate.position_accuracy
        is phx.discretization.SurfelAccuracy.EXACT
    )
    assert prepared.discretization.source_entity_ids[0] == 0


def test_reversed_triangle_reverses_surfel_normal_without_changing_measure() -> None:
    vertices = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    first = phx.geometry.SimplicialSurfelPlan(
        phx.geometry.TriangleSurface(
            vertices, jnp.asarray(((0, 1, 2),)), source_id="forward"
        )
    ).prepare()
    reversed_surface = phx.geometry.SimplicialSurfelPlan(
        phx.geometry.TriangleSurface(
            vertices, jnp.asarray(((0, 2, 1),)), source_id="reverse"
        )
    ).prepare()
    np.testing.assert_allclose(first.geometry.normal, -reversed_surface.geometry.normal)
    np.testing.assert_allclose(
        first.geometry.physical_surface_weight,
        reversed_surface.geometry.physical_surface_weight,
    )
