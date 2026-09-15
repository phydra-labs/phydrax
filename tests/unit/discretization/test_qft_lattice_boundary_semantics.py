#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization import TensorTopology
from phydrax.discretization._lattice_boundary import (
    CheckerboardEntityLayout,
    LatticeBoundaryPhasePlan,
)


def test_boundary_phase_shift_applies_each_oriented_crossing():
    topology = TensorTopology(("x",), (4,), periodic=(True,))
    boundary = LatticeBoundaryPhasePlan(
        topology, jnp.asarray([-1.0 + 0.0j]), maximum_displacement=5
    )
    values = jnp.arange(4.0)

    assert jnp.allclose(
        boundary.forward(values, 0),
        jnp.asarray([1, 2, 3, 0]) * jnp.asarray([1, 1, 1, -1]),
    )
    assert jnp.allclose(
        boundary.backward(values, 0),
        jnp.asarray([3, 0, 1, 2]) * jnp.asarray([-1, 1, 1, 1]),
    )
    assert jnp.allclose(
        boundary.shift(values, 0, 5),
        jnp.asarray([-1, -2, -3, 0]),
    )
    compiled = jax.jit(lambda field: boundary.forward(field, 0))
    assert jnp.allclose(compiled(values), boundary.forward(values, 0))


def test_open_boundary_shift_masks_wrapped_values():
    topology = TensorTopology(("x",), (3,), periodic=(False,))
    boundary = LatticeBoundaryPhasePlan(topology, jnp.ones((1,)))

    assert jnp.allclose(
        boundary.forward(jnp.asarray([2.0, 3.0, 5.0]), 0), jnp.asarray([3.0, 5.0, 0.0])
    )


def test_checkerboard_parity_split_merge_and_periodic_certification():
    topology = TensorTopology(("x", "y"), (4, 2), periodic=(True, True))
    layout = CheckerboardEntityLayout(topology)
    field = jnp.arange(16).reshape((4, 2, 2))
    even, odd = layout.split(field)

    assert jnp.array_equal(layout.site_parity, jnp.asarray([0, 1, 1, 0, 0, 1, 1, 0]))
    assert jnp.array_equal(layout.merge(even, odd), field)
    assert jnp.array_equal(
        layout.parity_at(jnp.asarray([[0, 0], [1, 0], [3, 1]])), jnp.asarray([0, 1, 0])
    )

    with pytest.raises(ValueError, match="even extent"):
        CheckerboardEntityLayout(TensorTopology(("x",), (3,), periodic=(True,)))
