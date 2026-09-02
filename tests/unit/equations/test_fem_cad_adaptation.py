#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_cad_projection_curvature_order_and_geometry_recovery_are_atomic():
    points = np.asarray(((0.9, 0.0), (0.0, 0.9), (-0.9, 0.0), (0.0, -0.9)))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "quadrilateral", np.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "state",
            phx.discretization.discontinuous_element("quadrilateral", 2),
            component_shape=(1,),
        ),
    ).prepare()
    coordinates = discretization.default_runtime.coordinates
    snapshot = phx.equations.fem.FiniteElementGeometrySnapshot(
        coordinates,
        jnp.zeros_like(coordinates),
        0.0,
        topology_id=mesh.topology_id,
        geometry_layout_id="cad-circle",
    )
    projection = phx.equations.fem.CADProjectionPlan(
        lambda values: values / jnp.sqrt(jnp.sum(values * values, axis=-1))[..., None],
        lambda values: values,
        selector_id="circle-face",
    )
    curvature = phx.equations.fem.CurvatureAdaptationPlan(
        target_error=0.01, minimum_degree=1, maximum_degree=6
    )
    result = phx.equations.fem.project_and_recover_cad_geometry(
        discretization,
        snapshot,
        jnp.arange(coordinates.shape[0]),
        projection,
        curvature,
        jnp.ones((coordinates.shape[0],)),
        jnp.full((coordinates.shape[0],), 0.5),
    )
    assert result.accepted
    assert result.projection.converged
    assert jnp.all(result.requested_degrees >= 1)
    assert jnp.all(result.requested_degrees <= 6)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(result.snapshot.coordinates**2, axis=-1)),
        1.0,
        atol=2.0e-8,
    )
