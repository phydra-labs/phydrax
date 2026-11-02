#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.geometry.design import FixedPoint, PointDistance, Sketch
from phydrax.optim import (
    LevenbergMarquardt,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
    least_squares,
)


def _unreduced(sketch: Sketch, *, max_iterations: int = 100) -> LeastSquaresResult:
    """Solve the original public residual without sketch coordinate preparation."""
    initial = sketch.points.reshape((-1,))
    problem = NonlinearLeastSquaresProblem(
        lambda vector, _args: sketch.residual(
            vector.reshape(sketch.points.shape), sketch.circle_radii
        ),
        problem_id="acceptance-unreduced",
    )
    return least_squares(
        problem,
        initial,
        method=LevenbergMarquardt(initial_damping=1e-8),
        termination=OptimizationTermination(
            absolute_optimality=1e-9,
            relative_optimality=0.0,
            maximum_steps=max_iterations,
        ),
    )


def _circle_intersection(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_radius: float,
    second_radius: float,
    sign: int,
) -> NDArray[np.float64]:
    """Independent Euclidean construction, not the sketch residual or seed code."""
    first = np.asarray(first, dtype=np.float64)
    delta = np.asarray(second, dtype=np.float64) - first
    distance = np.linalg.norm(delta)
    direction = delta / distance
    along = (first_radius**2 - second_radius**2 + distance**2) / (2 * distance)
    height = np.sqrt(max(0.0, first_radius**2 - along**2))
    return (
        first
        + along * direction
        + sign * height * np.array([-direction[1], direction[0]], dtype=np.float64)
    )


def _assert_distances(
    points: NDArray[np.float64], edges: tuple[tuple[int, int, float], ...]
) -> None:
    for first, second, distance in edges:
        np.testing.assert_allclose(
            np.linalg.norm(points[second] - points[first]), distance, atol=1e-8
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_anchored_rigid_triangle_preserves_mirrored_branch(sign: int) -> None:
    first = np.array([1.0, -2.0], dtype=np.float64)
    second = np.array([4.0, -2.0], dtype=np.float64)
    expected = _circle_intersection(first, second, 2.5, 2.0, sign)
    points = np.array([first, second, expected + [0.05, sign * 0.1]], dtype=np.float64)
    edges = ((0, 1, 3.0), (0, 2, 2.5), (1, 2, 2.0))
    sketch = Sketch(
        points,
        constraints=(
            FixedPoint(0, first),
            FixedPoint(1, second),
            *(PointDistance(a, b, distance) for a, b, distance in edges),
        ),
    )

    result = sketch.solve(max_iterations=100)
    original = _unreduced(sketch)
    assert bool(result.converged)
    assert bool(original.successful)
    solved = np.asarray(result.points)
    np.testing.assert_allclose(solved, np.vstack((first, second, expected)), atol=1e-8)
    _assert_distances(solved, edges)
    assert sign * (solved[2, 1] - first[1]) > 0
    np.testing.assert_allclose(
        np.asarray(original.parameters).reshape((-1, 2)), solved, atol=1e-7
    )
    np.testing.assert_allclose(result.residual, original.residual, atol=1e-8)
    assert result.root_evidence.jacobian_rank == 6
    assert result.root_evidence.degrees_of_freedom == 0


def test_floating_rigid_cluster_and_underconstrained_chain_have_distinct_dof() -> None:
    triangle = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    edges = ((0, 1, 3.0), (0, 2, 4.0), (1, 2, 5.0))
    rigid = Sketch(triangle, constraints=tuple(PointDistance(*edge) for edge in edges))
    chain = Sketch(
        triangle, constraints=tuple(PointDistance(*edge) for edge in edges[:2])
    )
    rigid_result = rigid.solve()
    chain_result = chain.solve()
    assert bool(rigid_result.converged) and bool(chain_result.converged)
    _assert_distances(np.asarray(rigid_result.points), edges)
    _assert_distances(np.asarray(chain_result.points), edges[:2])
    assert rigid_result.root_evidence.jacobian_rank == 3
    assert rigid_result.root_evidence.degrees_of_freedom == 3
    assert chain_result.root_evidence.jacobian_rank == 2
    assert chain_result.root_evidence.degrees_of_freedom == 4


def test_tangent_distance_loci_are_singular_but_valid_root() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    sketch = Sketch(
        points,
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 2, 1.0),
            PointDistance(1, 2, 1.0),
        ),
    )
    result = sketch.solve(max_iterations=100)
    original = _unreduced(sketch)
    assert bool(result.converged) and bool(original.successful)
    np.testing.assert_allclose(
        result.points[2],
        _circle_intersection(points[0], points[1], 1.0, 1.0, 1),
        atol=1e-9,
    )
    np.testing.assert_allclose(result.residual, original.residual, atol=1e-9)
    assert result.root_evidence.jacobian_rank == 5
    assert result.root_evidence.degrees_of_freedom == 1
    assert not result.root_evidence.regular_root_indicated


def test_redundant_constraints_remain_in_full_residual() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    sketch = Sketch(
        points,
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 2, np.sqrt(2.0)),
            PointDistance(1, 2, np.sqrt(2.0)),
            PointDistance(0, 2, np.sqrt(2.0)),
        ),
    )
    result = sketch.solve(max_iterations=100)
    original = _unreduced(sketch)
    assert bool(result.converged) and bool(original.successful)
    np.testing.assert_allclose(result.points, points, atol=1e-8)
    np.testing.assert_allclose(result.residual, original.residual, atol=1e-8)
    assert len(result.residual) == 7
    assert result.root_evidence.jacobian_rank == 6
    assert result.root_evidence.degrees_of_freedom == 0


def test_inconsistent_fixed_constraints_preserve_unreduced_weighted_optimum() -> None:
    sketch = Sketch(
        jnp.array([[0.0, 0.0]]),
        constraints=(
            FixedPoint(0, jnp.array([0.0, 0.0]), weight=1.0),
            FixedPoint(0, jnp.array([2.0, 0.0]), weight=3.0),
        ),
    )
    result = sketch.solve(max_iterations=100)
    original = _unreduced(sketch)
    assert not bool(result.converged)
    np.testing.assert_allclose(result.points, [[1.5, 0.0]], atol=1e-7)
    np.testing.assert_allclose(
        result.points, np.asarray(original.parameters).reshape((1, 2)), atol=1e-7
    )
    np.testing.assert_allclose(result.residual, original.residual, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(result.residual), np.sqrt(3.0), atol=1e-7)
    assert not result.root_evidence.root_converged


def test_continuation_reference_restores_in_fresh_process(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    from phydrax.geometry.design import (
        ParameterId,
        SketchBranchWitness,
        SketchContinuationReference,
    )
    from phydrax.geometry.design._continuation import (
        SketchContinuationReferenceArchiveRecipe,
    )
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import write_model_artifact

    distance = ParameterId("triangle", "radius")
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], dtype=np.float64)
    sketch = Sketch(
        points,
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 1, 2.0),
            PointDistance(0, 2, 2.0, parameter_id=distance),
            PointDistance(1, 2, 2.0),
        ),
        feature_id="triangle",
    )
    solution = sketch.solve(max_iterations=100)
    assert bool(solution.converged)
    assert solution.preparation.constructions
    assert solution.preparation.rigid_clusters
    reference = SketchContinuationReference(
        sketch, solution, (SketchBranchWitness((0, 1, 2)),)
    )
    recipe = SketchContinuationReferenceArchiveRecipe.from_reference(reference)
    with pytest.raises(ValueError, match="branch evidence"):
        recipe._replace(witness_signs=(-1,)).restore()
    with pytest.raises(ValueError, match="Preparation witnesses"):
        recipe._replace(
            preparation_pinned_values=np.ones_like(recipe.preparation_pinned_values)
        ).restore()
    composition = Composition(
        (
            CompositionEntry(
                reference,
                entry_id="sketch",
                role="model-state",
                owner_id="phydrax.geometry",
                structure_id="triangle",
                revision_id="triangle-root",
                semantics_id="continuation",
            ),
        ),
        boundary_id="sketch-archive",
    )
    path = write_model_artifact(tmp_path / "sketch.phx", composition, limits=None)
    child = """
import sys
import numpy as np
from phydrax.geometry.design import ParameterId, SketchContinuationReference, continue_sketch
from phydrax.lifecycle import Composition
from phydrax.meshing._model_codec import read_model_artifact, restore_model_archive_value
composition = read_model_artifact(sys.argv[1], expected_type=Composition,
                                  restore=restore_model_archive_value, limits=None)
reference = composition.value("sketch")
assert type(reference) is SketchContinuationReference
assert reference.witness_signs == (1,)
assert reference.solution.preparation.constructions
assert reference.solution.preparation.rigid_clusters
sketch = reference.sketch
np.testing.assert_allclose(reference.solution.points[2], [1., np.sqrt(3.)], atol=1e-8)
updated = reference.solution.dimensions.updated({ParameterId("triangle", "radius"): 2.1})
result = continue_sketch(sketch, reference, updated, solve_options={"max_iterations": 100})
assert result.accepted
points = np.asarray(result.reference.solution.points)
assert points[2, 1] > 0
np.testing.assert_allclose(np.linalg.norm(points[2] - points[0]), 2.1, atol=1e-8)
np.testing.assert_allclose(np.linalg.norm(points[2] - points[1]), 2., atol=1e-8)
"""
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "-c", child, str(path)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )
