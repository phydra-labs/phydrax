#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.geometry.design._sketch import (
    Coincident,
    FixedPoint,
    Horizontal,
    Midpoint,
    PointDistance,
    Sketch,
)
from phydrax.geometry.design._sketch_certificate import (
    certify_two_circle_construction,
    TwoCircleConstruction,
)
from phydrax.geometry.design._solve import (
    prepare_sketch_fixed_coordinates,
    seed_sketch_circle_intersections,
    solve_design_residual,
)


def test_fixed_coordinate_propagation_preserves_full_residual() -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 0.0], [2.0, 0.3], [1.0, 0.4], [4.0, 0.0]]),
        lines=jnp.asarray([[0, 1]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 0.0])),
            FixedPoint(1, jnp.asarray([2.0, 0.0])),
            Midpoint(2, 0),
            Coincident(2, 3),
            Horizontal(0),
        ),
    )
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    coordinates = np.concatenate(
        (np.asarray(sketch.points).ravel(), np.asarray(sketch.circle_radii))
    )
    coordinates[np.asarray(indices)] = np.asarray(values)
    points = coordinates[: sketch.points.size].reshape(sketch.points.shape)
    np.testing.assert_array_equal(
        np.asarray(sketch.residual(points, sketch.circle_radii)), 0.0
    )
    solution = sketch.solve()
    assert bool(solution.converged)
    np.testing.assert_allclose(
        np.asarray(solution.points), [[0, 0], [2, 0], [1, 0], [1, 0]], atol=1e-9
    )


def test_inconsistent_fixed_targets_retain_unreduced_objective() -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 0.0]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 0.0])),
            FixedPoint(0, jnp.asarray([2.0, 0.0])),
        ),
    )
    solution = sketch.solve()
    assert not bool(solution.converged)
    np.testing.assert_allclose(np.asarray(solution.points), [[1.0, 0.0]], atol=1e-7)
    unreduced = solve_design_residual(
        lambda vector: sketch.residual(vector.reshape((1, 2)), sketch.circle_radii),
        sketch.points.reshape((-1,)),
        max_iterations=32,
        tolerance=1e-9,
        damping=1e-8,
        problem_id=sketch._problem_id(),
    )
    np.testing.assert_allclose(
        np.asarray(solution.residual), np.asarray(unreduced.residual), atol=1e-8
    )


def test_two_circle_constructions_choose_reference_branch() -> None:
    sketch = Sketch(
        jnp.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 0.8]]),
        constraints=(
            FixedPoint(0, jnp.asarray([0.0, 0.0])),
            FixedPoint(1, jnp.asarray([2.0, 0.0])),
            PointDistance(0, 2, 2.0),
            PointDistance(1, 2, 2.0),
        ),
    )
    indices, values = prepare_sketch_fixed_coordinates(sketch, sketch.dimensions)
    assert 4 not in indices and 5 not in indices
    upper = jnp.asarray([0.0, 0.0, 2.0, 0.0, 1.0, 0.8])
    lower = upper.at[5].set(-0.8)
    up_seed, up_constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, upper, indices, values
    )
    down_seed, down_constructions = seed_sketch_circle_intersections(
        sketch, sketch.dimensions, lower, indices, values
    )
    assert isinstance(up_constructions[0][1], TwoCircleConstruction)
    assert isinstance(down_constructions[0][1], TwoCircleConstruction)
    assert up_constructions[0][1].branch_sign == 1
    assert up_constructions[0][1].squared_height_numerator == 12
    assert down_constructions[0][1].squared_height_numerator == 12
    assert down_constructions[0][1].branch_sign == -1
    np.testing.assert_allclose(np.asarray(up_seed)[4:], [1.0, np.sqrt(3.0)], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(down_seed)[4:], [1.0, -np.sqrt(3.0)], atol=1e-12
    )


def test_circle_certificate_rejects_degenerate_loci_and_unresolved_branch() -> None:
    origin = np.asarray([0, 0], dtype=np.float64)
    second = np.asarray([2, 0], dtype=np.float64)
    before = np.asarray([1, 1], dtype=np.float64)
    assert certify_two_circle_construction(origin, second, 1, 1, before) is None
    assert certify_two_circle_construction(origin, origin, 1, 1, before) is None
    assert (
        certify_two_circle_construction(
            origin, np.asarray([3, 0], dtype=np.float64), 1, 1, before
        )
        is None
    )
    assert (
        certify_two_circle_construction(
            origin, second, 2, 2, np.asarray([1, 0], dtype=np.float64)
        )
        is None
    )
