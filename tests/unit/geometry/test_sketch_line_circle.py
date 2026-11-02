#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.design._sketch import (
    FixedPoint,
    Horizontal,
    PointDistance,
    Sketch,
    Vertical,
)
from phydrax.geometry.design._sketch_certificate import (
    LineCircleConstruction,
    certify_line_circle_construction,
)
from phydrax.geometry.design._solve import (
    prepare_sketch_fixed_coordinates,
    seed_sketch_circle_intersections,
    solve_prepared_sketch_residual,
)


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_line_circle_reference_branch(axis: int, sign: int) -> None:
    anchor = [1.0, 1.0]
    center = [0.0, 0.0]
    point = [0.0, 0.0]
    point[axis] = 1.0
    point[1 - axis] = sign * 0.75
    sketch = Sketch(
        jnp.asarray([anchor, center, point, [4.0, 5.0]]),
        lines=jnp.asarray([[0, 2]]),
        constraints=(
            FixedPoint(0, jnp.asarray(anchor)),
            FixedPoint(1, jnp.asarray(center)),
            Vertical(0) if axis == 0 else Horizontal(0),
            PointDistance(1, 2, 2.0),
        ),
    )
    initial = sketch.points.reshape((-1,))
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    seed, constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, initial, indices, values
    )
    point_index, witness = constructions[0]
    assert point_index == 2 and isinstance(point_index, int)
    assert isinstance(witness, LineCircleConstruction)
    assert witness.fixed_axis == axis
    assert witness.discriminant == Fraction(3)
    assert witness.branch_sign == sign
    expected = np.asarray(point, dtype=np.float64)
    expected[1 - axis] = sign * np.sqrt(3.0)
    np.testing.assert_allclose(np.asarray(seed, dtype=np.float64)[4:6], expected)
    result, used = solve_prepared_sketch_residual(
        lambda x: sketch.residual(x.reshape((-1, 2)), sketch.circle_radii),
        initial,
        seed,
        indices,
        values,
        constructions,
        max_iterations=100,
        tolerance=1e-8,
        damping=1e-3,
        problem_id="line-circle-test",
        bounds=None,
    )
    assert used and bool(result.converged)
    np.testing.assert_allclose(
        np.asarray(result.coordinates, dtype=np.float64)[4:6], expected, atol=1e-8
    )


def test_line_circle_predicates_are_exact_and_reject_ambiguous_branches() -> None:
    center = np.array([0.0, 0.0], dtype=np.float64)
    assert (
        certify_line_circle_construction(
            center, 1, 1, 1, np.array([1.0, 1.0], dtype=np.float64)
        )
        is None
    )
    assert (
        certify_line_circle_construction(
            center, 1, 1, 2, np.array([1.0, 2.0], dtype=np.float64)
        )
        is None
    )
    assert (
        certify_line_circle_construction(
            center, 2, 1, 1, np.array([0.0, 1.0], dtype=np.float64)
        )
        is None
    )
    near = np.nextafter(1.0, 0.0)
    witness = certify_line_circle_construction(
        center, 1, 1, near, np.array([1.0, near], dtype=np.float64)
    )
    assert witness is not None
    assert witness.discriminant == 1 - Fraction.from_float(near) ** 2
    assert witness.discriminant > 0


@pytest.mark.parametrize("reference_x,radius", [(0.0, 2.0), (1.0, 1.0)])
def test_unresolved_or_tangent_line_circle_is_only_a_seed(
    reference_x: float,
    radius: float,
) -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 1.0], [0.0, 0.0], [reference_x, 1.0]]),
        lines=jnp.asarray([[0, 2]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 1.0])),
            FixedPoint(1, jnp.asarray([0.0, 0.0])),
            Horizontal(0),
            PointDistance(1, 2, radius),
        ),
    )
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    seed, constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, sketch.points.reshape((-1,)), indices, values
    )
    assert not constructions
    np.testing.assert_allclose(
        np.asarray(seed, dtype=np.float64)[4:6], [np.sqrt(radius**2 - 1), 1.0]
    )


def test_incompatible_line_circle_constraints_fall_back_to_full_residual() -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        lines=jnp.asarray([[0, 2]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 1.0])),
            FixedPoint(1, jnp.asarray([0.0, 0.0])),
            Horizontal(0),
            PointDistance(1, 2, 2.0),
            PointDistance(1, 2, 3.0),
        ),
    )
    initial = sketch.points.reshape((-1,))
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    seed, constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, initial, indices, values
    )
    assert len(constructions) == 1
    result, used = solve_prepared_sketch_residual(
        lambda x: sketch.residual(x.reshape((-1, 2)), sketch.circle_radii),
        initial,
        seed,
        indices,
        values,
        constructions,
        max_iterations=100,
        tolerance=1e-8,
        damping=1e-3,
        problem_id="incompatible-line-circle-test",
        bounds=None,
    )
    assert not used
    assert not bool(result.converged)
    np.testing.assert_allclose(
        np.asarray(result.residual, dtype=np.float64),
        np.asarray(
            sketch.residual(result.coordinates.reshape((-1, 2)), sketch.circle_radii),
            dtype=np.float64,
        ),
    )


def test_fully_constructed_line_circle_evaluates_original_residual() -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]),
        lines=jnp.asarray([[0, 2]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 1.0])),
            FixedPoint(1, jnp.asarray([0.0, 0.0])),
            Horizontal(0),
            PointDistance(1, 2, 2.0),
        ),
    )
    initial = sketch.points.reshape((-1,))
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    seed, constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, initial, indices, values
    )
    result, used = solve_prepared_sketch_residual(
        lambda x: sketch.residual(x.reshape((-1, 2)), sketch.circle_radii),
        initial,
        seed,
        indices,
        values,
        constructions,
        max_iterations=100,
        tolerance=1e-8,
        damping=1e-3,
        problem_id="fully-constructed-line-circle-test",
        bounds=None,
    )
    assert used and bool(result.converged)
    assert int(result.iterations) == 0
    np.testing.assert_allclose(
        np.asarray(result.coordinates, dtype=np.float64)[4:6], [np.sqrt(3.0), 1.0]
    )
    np.testing.assert_allclose(
        np.asarray(result.residual, dtype=np.float64), 0.0, atol=1e-8
    )
