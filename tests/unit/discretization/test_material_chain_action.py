from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.finite_volume._material_chain_action import (
    material_chain_apply,
    material_chain_transpose,
)
from phydrax.discretization.finite_volume._material_schedule import MaterialChainSchedule
from phydrax.discretization.finite_volume._remap_csr_view import material_csr_rows
from phydrax.linalg import SparseAssemblyPolicy


def _two_edit_schedule() -> MaterialChainSchedule:
    return MaterialChainSchedule(
        np.asarray(((0, 1, 0, 0), (2, 3, 4, 5)), dtype=np.int64),
        np.asarray(((2, 3, 4, 5), (0, 1, 0, 0)), dtype=np.int64),
        np.asarray(((True, True, False, False), (True, True, True, True))),
        np.asarray(((True, True, True, True), (True, True, False, False))),
        np.asarray(((0, 0, 1, 1), (2, 3, 4, 5)), dtype=np.int64),
        np.asarray(((2, 3, 4, 5), (0, 0, 1, 1)), dtype=np.int64),
        np.asarray(((0.5, 0.75, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))),
        np.asarray(((-1, 0, 0, 0), (1, 1, 1, 1)), dtype=np.int64),
        np.ones((2, 4), dtype=np.bool_),
        np.asarray((0, 1), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int64),
        source_count=2,
        target_count=2,
        slot_count=6,
        schedule_id="two-edit-material-content-test",
    )


def test_two_edit_exponent_safe_content_action_and_transpose() -> None:
    schedule = _two_edit_schedule()
    source = jnp.asarray(((2.0**900, 2.0**899), (2.0**899, 2.0**898)), dtype=jnp.float64)
    target = material_chain_apply(schedule, source)
    np.testing.assert_array_equal(target, source)

    cotangents = jnp.asarray(((0.5, -0.25), (0.25, 0.5)), dtype=jnp.float64)
    transposed = material_chain_transpose(schedule, cotangents)
    np.testing.assert_allclose(
        np.vdot(np.asarray(target), np.asarray(cotangents)),
        np.vdot(np.asarray(source), np.asarray(transposed)),
        rtol=1e-15,
        atol=0.0,
    )
    for material in range(source.shape[1]):
        before = sum(
            (
                Fraction.from_float(float(value))
                for value in np.asarray(source)[:, material]
            ),
            Fraction(0),
        )
        after = sum(
            (
                Fraction.from_float(float(value))
                for value in np.asarray(target)[:, material]
            ),
            Fraction(0),
        )
        assert after == before


def test_two_edit_csr_view_matches_the_sparse_action() -> None:
    schedule = _two_edit_schedule()
    offsets, sources, weights = material_csr_rows(schedule, SparseAssemblyPolicy())
    np.testing.assert_array_equal(offsets, (0, 1, 2))
    np.testing.assert_array_equal(sources, (0, 1))
    np.testing.assert_array_equal(weights, (1.0, 1.0))
