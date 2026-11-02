#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax.geometry.brep._constructors import ProfileArc, ProfileLine
from phydrax.geometry.design._sketch import (
    FixedPoint,
    Sketch,
    SketchPreparationEvidence,
    SketchSolution,
)
from phydrax.geometry.design._sketch_certificate import RootDOFEvidence
from phydrax.geometry.design._sketch_priority import solve_priority_sketch


_POINTS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float64)


def _sketch(
    points: npt.NDArray[np.float64] = _POINTS, *, reverse: bool = False
) -> Sketch:
    return Sketch(
        points,
        lines=np.array([[1, 0]], dtype=np.int64),
        arc_endpoints=np.array([[1, 0] if reverse else [0, 1]], dtype=np.int64),
        arc_centers=np.array([2], dtype=np.int64),
        arc_counterclockwise=np.array([not reverse], dtype=np.bool_),
        constraints=tuple(FixedPoint(i, point) for i, point in enumerate(points)),
        feature_id="directed-arc",
    )


def _root_at_initial_points(sketch: Sketch) -> SketchSolution:
    coordinates = jnp.concatenate((sketch.points.reshape(-1), sketch.circle_radii))
    point_size = sketch.points.size

    def residual(vector: jax.Array) -> jax.Array:
        return sketch.residual(
            vector[:point_size].reshape(sketch.points.shape), vector[point_size:]
        )

    values = residual(coordinates)
    evidence = RootDOFEvidence(
        sketch._problem_id(),
        coordinates,
        values,
        jax.jacfwd(residual)(coordinates),
        residual_tolerance=1e-9,
    )
    indices, fixed, seed, constructions, clusters, reductions = (
        sketch._prepare_components(coordinates, sketch.dimensions, 1e-9)
    )
    return SketchSolution(
        points=sketch.points,
        circle_radii=sketch.circle_radii,
        residual=values,
        residual_norm=jnp.linalg.norm(values),
        converged=evidence.root_converged,
        iterations=0,
        status=0,
        root_evidence=evidence,
        preparation=SketchPreparationEvidence(
            sketch._problem_id(),
            coordinates,
            seed,
            indices,
            fixed,
            constructions,
            clusters,
            tuple(item.point_indices for item in reductions),
            32,
            1e-8,
            False,
        ),
        dimensions=sketch.dimensions,
    )


def test_arc_incidence_is_part_of_residual_without_new_coordinates() -> None:
    sketch = _sketch()
    points = np.array(_POINTS, dtype=np.float64)
    points[1, 0] = -2.0
    residual = np.asarray(sketch.residual(points, sketch.circle_radii), dtype=np.float64)
    np.testing.assert_allclose(residual[-1], -1.0)
    assert residual.shape == (7,)
    assert sketch.circle_radii.size == 0
    assert len(sketch._residual_components(jnp.asarray(points), sketch.circle_radii)) == 3


def test_exact_arc_profile_and_reversed_directed_use() -> None:
    forward = _sketch()
    reverse = _sketch(reverse=True)
    first = forward.to_profile(_root_at_initial_points(forward), name="half-disk")
    second = reverse.to_profile(_root_at_initial_points(reverse), name="half-disk")
    assert first.profile.outer == second.profile.outer
    loop = first.profile.outer
    arc_index = next(
        i for i, segment in enumerate(loop.segments) if isinstance(segment, ProfileArc)
    )
    arc = loop.segments[arc_index]
    assert isinstance(arc, ProfileArc)
    assert arc.center == (0.0, 0.0)
    assert arc.counterclockwise
    assert loop.vertices[arc_index] == (1.0, 0.0)
    assert loop.vertices[(arc_index + 1) % len(loop.vertices)] == (-1.0, 0.0)
    assert sum(isinstance(segment, ProfileLine) for segment in loop.segments) == 1
    assert forward._problem_id() != reverse._problem_id()
    assert first.lineage_id != second.lineage_id
    assert first.source_id != second.source_id
    with pytest.raises(NotImplementedError, match="Arc profiles"):
        forward.to_source()


def test_profile_rejects_numerically_converged_but_inexact_arc_incidence() -> None:
    points = np.array(_POINTS, dtype=np.float64)
    points[1, 0] -= 2.0**-40
    sketch = _sketch(points)
    solution = _root_at_initial_points(sketch)
    assert solution.root_evidence.root_converged
    with pytest.raises(ValueError, match="represented coordinates"):
        sketch.to_profile(solution, name="inexact")


def test_arc_topology_validation_and_problem_identity() -> None:
    with pytest.raises(ValueError, match="distinct endpoints"):
        Sketch(
            _POINTS,
            arc_endpoints=np.array([[0, 0]], dtype=np.int64),
            arc_centers=np.array([2], dtype=np.int64),
            arc_counterclockwise=np.array([True], dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="absent point"):
        Sketch(
            _POINTS,
            arc_endpoints=np.array([[0, 3]], dtype=np.int64),
            arc_centers=np.array([2], dtype=np.int64),
            arc_counterclockwise=np.array([True], dtype=np.bool_),
        )
    sketch = _sketch()
    # Reordering or changing incidence must change identity independently of coordinates.
    changed = Sketch(
        np.vstack((_POINTS, [[0.0, 1.0]])),
        lines=np.array([[1, 0]], dtype=np.int64),
        arc_endpoints=np.array([[0, 1]], dtype=np.int64),
        arc_centers=np.array([3], dtype=np.int64),
        arc_counterclockwise=np.array([True], dtype=np.bool_),
        constraints=sketch.constraints,
        feature_id="directed-arc",
    )
    assert sketch._problem_id() != changed._problem_id()


def test_priority_solve_reconstructs_arc_topology() -> None:
    sketch = _sketch()
    result = solve_priority_sketch(sketch)
    solved = result.solved_sketch
    np.testing.assert_array_equal(solved.arc_endpoints, sketch.arc_endpoints)
    np.testing.assert_array_equal(solved.arc_centers, sketch.arc_centers)
    np.testing.assert_array_equal(
        solved.arc_counterclockwise, sketch.arc_counterclockwise
    )
    assert solved.residual(
        result.solution.points, result.solution.circle_radii
    ).shape == (7,)
