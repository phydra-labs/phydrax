# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Restore interval owners by validating their retained source context."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.implicit._enclosure import IntervalFieldBounds
from phydrax.units import METER
from tests._support.assertions import assert_tree_equal


def _bounds() -> IntervalFieldBounds:
    geometry = phx.geometry.Sphere(
        (0.013, -0.021, 0.007), 0.4, feature_id="restored-interval"
    ).compile(coordinate_unit=METER)
    return IntervalFieldBounds(geometry)


def test_interval_restore_validates_the_same_retained_owner() -> None:
    value = _bounds()
    rebuilt = IntervalFieldBounds.from_stored_fields(value)
    assert rebuilt is value
    assert_tree_equal(rebuilt, value)
    lower = np.asarray(((0.5, 0.0, 0.0),), dtype=np.float64)
    upper = np.asarray(((0.6, 0.1, 0.1),), dtype=np.float64)
    before = value.boxes(lower, upper)
    after = rebuilt.boxes(lower, upper)
    np.testing.assert_array_equal(after.value_lower, before.value_lower)
    np.testing.assert_array_equal(after.value_upper, before.value_upper)
    np.testing.assert_array_equal(after.directional_lower, before.directional_lower)
    np.testing.assert_array_equal(after.directional_upper, before.directional_upper)


def test_interval_restore_refuses_invalid_retained_geometry_state() -> None:
    value = _bounds()
    radius = value.state.schema.index(
        phx.geometry.ParameterId("restored-interval", "radius")
    )
    state = value.state.replace_at(radius, jnp.asarray(-0.4, dtype=jnp.float64))
    corrupted = eqx.tree_at(lambda bounds: bounds.state, value, state)
    with pytest.raises(ValueError, match=r"IntervalFieldBounds\.state"):
        IntervalFieldBounds.from_stored_fields(corrupted)


def test_interval_restore_refuses_a_changed_source_program_shape() -> None:
    value = _bounds()
    center = value.state.schema.index(
        phx.geometry.ParameterId("restored-interval", "center")
    )
    leaves = list(value.state.values)
    leaves[center] = jnp.asarray((0.0, 0.0), dtype=jnp.float64)
    state = eqx.tree_at(lambda state: state.values, value.state, tuple(leaves))
    corrupted = eqx.tree_at(lambda bounds: bounds.state, value, state)
    with pytest.raises(ValueError, match="shape"):
        IntervalFieldBounds.from_stored_fields(corrupted)
