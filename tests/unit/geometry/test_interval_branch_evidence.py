#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Per-box branch decisions of the host interval program.

Each oracle decides smoothness or continuity of the point map on a box from
the box corners alone, by exact comparisons of the closed-form branch sets.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry._interval_enclosure import prepare_interval_function


_BOXES = np.asarray(
    (
        ((0.25, -0.5, 0.0), (0.5, -0.25, 0.25)),  # x > y strictly, x > 0
        ((-0.25, -0.25, 0.0), (0.25, 0.25, 0.25)),  # meets x = y and x = 0
        ((-0.5, 0.25, 0.0), (-0.25, 0.5, 0.25)),  # x < y strictly, x < 0
        ((0.0, -0.5, 0.0), (0.5, -0.25, 0.25)),  # touches x = 0 only
    ),
    dtype=np.float64,
)


def _evaluate(
    function: Callable[[Array], Array],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    program = prepare_interval_function(function, 3, batch_capacity=3)
    return program.evaluate_branches(_BOXES[:, 0], _BOXES[:, 1])


def _meets_diagonal() -> np.ndarray:
    # The closed box meets x = y iff x - y ranges over an interval holding 0.
    low = _BOXES[:, 0, 0] - _BOXES[:, 1, 1]
    high = _BOXES[:, 1, 0] - _BOXES[:, 0, 1]
    return (low <= 0.0) & (high >= 0.0)


def _meets_plane() -> np.ndarray:
    return (_BOXES[:, 0, 0] <= 0.0) & (_BOXES[:, 1, 0] >= 0.0)


def test_minimum_kinks_only_where_the_box_meets_its_tie() -> None:
    def field(point: Array) -> Array:
        return jnp.minimum(point[0], point[1]) + point[2] ** 2

    lower, upper, smooth, continuous = _evaluate(field)
    plain = prepare_interval_function(field, 3, batch_capacity=3).evaluate(
        _BOXES[:, 0], _BOXES[:, 1]
    )
    np.testing.assert_array_equal(lower, plain[0])
    np.testing.assert_array_equal(upper, plain[1])
    np.testing.assert_array_equal(smooth, ~_meets_diagonal())
    assert np.all(continuous)


def test_equal_operands_decide_their_extremum() -> None:
    def field(point: Array) -> Array:
        first = jnp.sqrt(point[0] ** 2 + 1.0)
        second = jnp.sqrt(point[0] ** 2 + 1.0)
        return jnp.min(jnp.stack((first, second))) + jnp.max(jnp.asarray((2.0, 2.0, 1.0)))

    _, _, smooth, continuous = _evaluate(field)
    assert np.all(smooth) and np.all(continuous)


def test_absolute_value_kinks_and_selection_jumps_on_the_plane() -> None:
    def kinked(point: Array) -> Array:
        return jnp.abs(point[0]) + point[1]

    def stepped(point: Array) -> Array:
        return jnp.where(point[0] > 0.0, 1.0 + point[1], -1.0 + point[1])

    _, _, smooth, continuous = _evaluate(kinked)
    np.testing.assert_array_equal(smooth, ~_meets_plane())
    assert np.all(continuous)
    _, _, smooth, continuous = _evaluate(stepped)
    np.testing.assert_array_equal(smooth, ~_meets_plane())
    np.testing.assert_array_equal(continuous, ~_meets_plane())


def test_square_root_kinks_where_its_argument_reaches_zero() -> None:
    def field(point: Array) -> Array:
        return jnp.sqrt(point[0] * point[0] + point[2] * point[2])

    _, _, smooth, continuous = _evaluate(field)
    # x^2 + z^2 vanishes on the box iff it contains a point with x = z = 0.
    reaches = _meets_plane() & (_BOXES[:, 0, 2] <= 0.0) & (_BOXES[:, 1, 2] >= 0.0)
    np.testing.assert_array_equal(smooth, ~reaches)
    assert np.all(continuous)


def test_decided_selections_keep_derivative_enclosures_finite() -> None:
    centers = jnp.asarray(((-0.6, 0.0, 0.0), (0.6, 0.0, 0.0)))

    def union(point: Array) -> Array:
        distances = jnp.linalg.norm(point[None, :] - centers, axis=1) - 0.3
        return jnp.min(distances) + jnp.minimum(point[2], point[2])

    lower = np.asarray(((0.85, -0.05, -0.05), (-0.05, -0.05, -0.05)), dtype=np.float64)
    upper = lower + 0.1
    gradient = prepare_interval_function(jax.grad(union), 3)
    gradient_lower, gradient_upper = gradient.evaluate(lower, upper)
    # Near the right sphere its distance wins strictly, so the gradient is
    # that sphere's unit normal plus e_z. On the bisector box both distances
    # tie; the enclosure stays finite (the attaining count is at least one)
    # and holds both branch gradients. Every corner's exact gradient lies
    # inside each finite enclosure.
    assert np.all(np.isfinite(gradient_lower)) and np.all(np.isfinite(gradient_upper))
    for row, branches in (
        (0, ((0.6, 0.0, 0.0),)),
        (1, ((-0.6, 0.0, 0.0), (0.6, 0.0, 0.0))),
    ):
        for corner in product(*zip(lower[row], upper[row], strict=True)):
            for center in branches:
                offset = np.asarray(corner) - np.asarray(center)
                exact = offset / np.linalg.norm(offset) + np.asarray((0.0, 0.0, 1.0))
                assert np.all(gradient_lower[row] <= exact)
                assert np.all(exact <= gradient_upper[row])
