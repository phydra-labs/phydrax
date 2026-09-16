#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _mesh():
    return phx.geometry.TriangleMesh(
        jnp.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ),
        jnp.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
        source_id="bridge-square",
    )


def test_gl_london_bridge_uses_declared_current_projection_and_support():
    mesh = _mesh()
    gl_plan = phx.applications.superconductivity.GaugeCovariantGLPlan(
        mesh,
        alpha=-1.0,
        beta=1.0,
        kinetic_coefficient=0.5,
        magnetic_coefficient=1.0,
        gauge_coupling=1.0,
    )
    gl_state = gl_plan.initialize(jnp.ones((5,), dtype=jnp.complex128))
    london_plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        mesh, pearl_length=0.2
    )
    london = london_plan.solve(0.0)
    projection = jnp.zeros((mesh.topology.edges.shape[0], london.face_sheet_current.size))
    bridge = phx.applications.superconductivity.GLLondonBridgePlan(
        projection,
        relative_tolerance=1.0e-12,
        minimum_scale_separation=5.0,
    )

    evidence = bridge.evaluate(gl_plan, gl_state, london, scale_separation=10.0)
    unsupported = bridge.evaluate(gl_plan, gl_state, london, scale_separation=1.0)

    assert bool(evidence.successful)
    assert not bool(unsupported.supported)
    assert not bool(unsupported.successful)
