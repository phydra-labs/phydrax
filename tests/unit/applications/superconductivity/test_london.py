#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

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
        source_id="planar-square-film",
    )


def test_zero_field_london_state_is_zero_and_divergence_free():
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(), pearl_length=0.2, tolerance=1.0e-8
    )
    result = plan.solve(0.0)

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.stream_function, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.face_sheet_current, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.total_energy, 0.0, atol=1e-12)


def test_london_constraints_and_inductance_are_reciprocal():
    constraint = jnp.asarray(((1.0, -1.0, 0.0, 0.0, 0.0),))
    plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        _mesh(),
        pearl_length=0.2,
        constraint_matrix=constraint,
        constraint_labels=("terminal-current",),
        tolerance=1.0e-7,
    )
    result = plan.solve(0.0, jnp.asarray((1.0,)))
    inductance, reciprocity = plan.inductance_matrix()

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(constraint @ result.stream_function, (1.0,), atol=1e-8)
    assert result.kinetic_energy >= 0.0
    assert inductance.shape == (1, 1)
    np.testing.assert_allclose(reciprocity, 0.0, atol=1e-12)
