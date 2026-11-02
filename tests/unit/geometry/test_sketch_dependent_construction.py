#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import jax
import jax.numpy as jnp

import numpy as np
import pytest

from phydrax.geometry.design import (
    FixedPoint,
    LineAngle,
    ParameterId,
    PointDistance,
    Sketch,
    SketchBranchWitness,
    SketchContinuationReference,
    continue_sketch,
)
from phydrax.geometry.design import _solve
from phydrax.geometry.design._sketch_certificate import TwoCircleConstruction


_RADIUS = ParameterId("dependent", "first-radius")


def _sketch(
    sign: int = 1,
    *,
    second_radii: tuple[float, float] = (4.0, 3.0),
    ambiguous: bool = False,
    overconstrained: bool = False,
    free_point: bool = False,
) -> Sketch:
    # The first intersection is (9/5, +/-12/5), not a binary float.
    first = np.array([1.8, sign * 2.4], dtype=np.float64)
    delta = np.array([-first[1], first[0]], dtype=np.float64)
    second = (8.0 / 9.0) * first + sign * (np.sqrt(80.0) / 9.0) * delta
    if ambiguous:
        second = first / 2.0
    points = [[0.0, 0.0], [5.0, 0.0], first.tolist(), second.tolist()]
    if free_point:
        points.append([7.0, 8.0])
    constraints = [
        FixedPoint(0, np.array([0.0, 0.0], dtype=np.float64)),
        FixedPoint(1, np.array([5.0, 0.0], dtype=np.float64)),
        PointDistance(0, 2, 3.0, parameter_id=_RADIUS),
        PointDistance(1, 2, 4.0),
        PointDistance(0, 3, second_radii[0]),
        PointDistance(2, 3, second_radii[1]),
    ]
    if overconstrained:
        constraints.append(PointDistance(0, 3, 5.0))
    return Sketch(
        np.asarray(points, dtype=np.float64),
        constraints=tuple(constraints),
        feature_id="dependent",
    )


def _assert_full_residual(sketch: Sketch, solution: Any) -> None:
    residual = sketch.residual(
        solution.points, solution.circle_radii, dimensions=solution.dimensions
    )
    np.testing.assert_allclose(solution.residual, residual, atol=1e-12)
    assert np.asarray(solution.residual).size == np.asarray(residual).size


@pytest.mark.parametrize("sign", [1, -1])
def test_dependent_exact_witness_and_reduced_root(
    sign: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    sketch = _sketch(sign, free_point=True)
    original = _solve.solve_design_residual
    extents: list[int] = []

    def record(residual: Any, initial: Any, *args: Any, **kwargs: Any) -> Any:
        extents.append(int(initial.size))
        return original(residual, initial, *args, **kwargs)

    monkeypatch.setattr(_solve, "solve_design_residual", record)
    solution = sketch.solve(max_iterations=100)
    assert bool(solution.converged)
    assert solution.preparation.reduced_root_accepted
    assert extents and extents[0] < sketch.points.size
    witnesses = dict(solution.preparation.constructions)
    assert 2 in witnesses and 3 in witnesses
    assert isinstance(witnesses[2], TwoCircleConstruction)
    assert isinstance(witnesses[3], TwoCircleConstruction)
    assert witnesses[2].branch_sign == sign
    assert witnesses[3].branch_sign == sign
    exact_center = (Fraction(9, 5), Fraction(sign * 12, 5))
    assert exact_center in (witnesses[3].first, witnesses[3].second)
    np.testing.assert_allclose(
        solution.points[2], np.asarray(exact_center, dtype=np.float64), atol=1e-9
    )
    assert sign * float(solution.points[3, 1]) > 0
    _assert_full_residual(sketch, solution)
    np.testing.assert_allclose(solution.residual, 0.0, atol=1e-9)


def test_grounded_rigid_triangle_keeps_pose_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array(
        [[0.0, 0.0], [0.0, 3.0], [-4.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
        dtype=np.float64,
    )
    sketch = Sketch(
        points,
        lines=np.array([[0, 1], [3, 4]], dtype=np.int32),
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(3, points[3]),
            FixedPoint(4, points[4]),
            PointDistance(0, 1, 3.0),
            PointDistance(0, 2, 4.0),
            PointDistance(1, 2, 5.0),
            LineAngle(0, 1, 0.0),
        ),
    )
    original = _solve.solve_design_residual
    extents: list[int] = []

    def record(residual: Any, initial: Any, *args: Any, **kwargs: Any) -> Any:
        extents.append(int(initial.size))
        return original(residual, initial, *args, **kwargs)

    monkeypatch.setattr(_solve, "solve_design_residual", record)
    solution = sketch.solve(max_iterations=100)
    assert bool(solution.converged)
    assert solution.preparation.reduced_root_accepted
    assert (0, 1, 2) in solution.preparation.rigid_reduction_points
    assert extents and extents[0] == 1
    np.testing.assert_allclose(
        solution.points,
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 0.0], [11.0, 0.0]],
        atol=1e-8,
    )
    _assert_full_residual(sketch, solution)
    np.testing.assert_allclose(solution.residual, 0.0, atol=1e-9)


def test_regular_branch_sensitivity_and_continuation() -> None:
    sketch = _sketch()
    solution = sketch.solve(max_iterations=100)
    assert bool(solution.converged)
    reference = SketchContinuationReference(
        sketch, solution, (SketchBranchWitness((0, 1, 2)), SketchBranchWitness((0, 2, 3)))
    )
    result = continue_sketch(sketch, reference, sketch.dimensions)
    assert result.accepted
    assert result.attempt.root_evidence.regular_root_indicated
    sensitivity = np.asarray(result.dimension_sensitivity(sketch, _RADIUS).coordinates)
    np.testing.assert_allclose(sensitivity[4:6], [0.6, 0.8], atol=1e-7)
    jvp = result.dimension_jvp(sketch, {_RADIUS: jnp.asarray(1.0)})
    coordinate_cotangent = jnp.arange(jvp.size, dtype=jvp.dtype) + 1
    vjp = result.dimension_vjp(sketch, coordinate_cotangent, (_RADIUS,))
    np.testing.assert_allclose(jvp, sensitivity, atol=1e-9)
    np.testing.assert_allclose(
        vjp[_RADIUS], jnp.dot(coordinate_cotangent, jvp), atol=1e-9
    )
    coordinates = jnp.asarray(result.attempt.points).reshape((-1,))
    dimension_index = sketch.dimension_schema.index(_RADIUS)
    dimension_value = result.attempt.dimensions.values[dimension_index]

    def residual(vector: jax.Array, value: jax.Array) -> jax.Array:
        dimensions = result.attempt.dimensions.replace_at(dimension_index, value)
        return sketch.residual(
            vector.reshape(sketch.points.shape),
            result.attempt.circle_radii,
            dimensions=dimensions,
        )

    _, tangent = jax.jvp(
        residual,
        (coordinates, dimension_value),
        (jnp.asarray(sensitivity), jnp.ones_like(dimension_value)),
    )
    np.testing.assert_allclose(tangent, 0.0, atol=1e-8)
    cotangent = jnp.arange(tangent.size, dtype=coordinates.dtype) + 1
    _, pullback = jax.vjp(residual, coordinates, dimension_value)
    coordinate_bar, dimension_bar = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.dot(coordinate_bar, jnp.asarray(sensitivity)) + dimension_bar,
        jnp.dot(cotangent, tangent),
        atol=1e-10,
    )
    step = 1e-4
    plus = continue_sketch(
        sketch, reference, sketch.dimensions.updated({_RADIUS: 3.0 + step})
    )
    minus = continue_sketch(
        sketch, reference, sketch.dimensions.updated({_RADIUS: 3.0 - step})
    )
    assert plus.accepted and minus.accepted
    difference = (
        np.asarray(plus.attempt.points) - np.asarray(minus.attempt.points)
    ).reshape(-1) / (2 * step)
    np.testing.assert_allclose(sensitivity, difference, rtol=2e-5, atol=2e-6)
    _assert_full_residual(sketch, result.attempt)


def test_angle_branch_cut_refuses_sensitivity() -> None:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    parameter = ParameterId("angle-cut", "angle")
    sketch = Sketch(
        points,
        lines=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        feature_id="angle-cut",
        constraints=(
            *(FixedPoint(index, point) for index, point in enumerate(points)),
            LineAngle(0, 1, 0.0, parameter_id=parameter, weight=1e-24),
        ),
    )
    solution = sketch.solve()
    assert bool(solution.converged)
    with pytest.raises(NotImplementedError, match="angle branch cut"):
        sketch.dimension_sensitivity(solution, parameter)


@pytest.mark.parametrize(
    ("radii", "ambiguous"), [((1.0, 1.0), False), ((1.5, 1.5), False), ((4.0, 3.0), True)]
)
def test_disjoint_tangent_and_ambiguous_loci_are_not_certified(
    radii: tuple[float, float], ambiguous: bool
) -> None:
    sketch = _sketch(second_radii=radii, ambiguous=ambiguous)
    solution = sketch.solve(max_iterations=100)
    witnesses = dict(solution.preparation.constructions)
    assert 2 in witnesses
    assert 3 not in witnesses
    _assert_full_residual(sketch, solution)
    if radii == (1.0, 1.0):
        assert not bool(solution.converged)
        assert not solution.preparation.reduced_root_accepted


def test_overconstrained_construction_falls_back_to_full_objective() -> None:
    sketch = _sketch(overconstrained=True)
    solution = sketch.solve(max_iterations=100)
    native = _solve.solve_design_residual(
        lambda vector: sketch.residual(
            vector.reshape(sketch.points.shape), sketch.circle_radii
        ),
        sketch.points.reshape(-1),
        max_iterations=100,
        tolerance=1e-9,
        damping=1e-8,
        problem_id=sketch._problem_id(),
    )
    assert not bool(solution.converged)
    assert not solution.preparation.reduced_root_accepted
    _assert_full_residual(sketch, solution)
    np.testing.assert_allclose(solution.residual, native.residual, atol=1e-8)
    np.testing.assert_allclose(solution.points.reshape(-1), native.coordinates, atol=1e-8)


def test_dependent_reference_restores_in_fresh_process(tmp_path: Path) -> None:
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import write_model_artifact

    sketch = _sketch()
    solution = sketch.solve(max_iterations=100)
    assert bool(solution.converged)
    assert {2, 3} <= set(dict(solution.preparation.constructions))
    reference = SketchContinuationReference(
        sketch, solution, (SketchBranchWitness((0, 1, 2)), SketchBranchWitness((0, 2, 3)))
    )
    composition = Composition(
        (
            CompositionEntry(
                reference,
                entry_id="sketch",
                role="model-state",
                owner_id="phydrax.geometry",
                structure_id="dependent",
                revision_id="root",
                semantics_id="continuation",
            ),
        ),
        boundary_id="dependent-archive",
    )
    path = write_model_artifact(tmp_path / "dependent.phx", composition, limits=None)
    child = """
from fractions import Fraction
import sys
import numpy as np
from phydrax.geometry.design import ParameterId, SketchContinuationReference, continue_sketch
from phydrax.lifecycle import Composition
from phydrax.meshing._model_codec import read_model_artifact, restore_model_archive_value
composition = read_model_artifact(sys.argv[1], expected_type=Composition,
                                  restore=restore_model_archive_value, limits=None)
reference = composition.value('sketch')
assert type(reference) is SketchContinuationReference
witnesses = dict(reference.solution.preparation.constructions)
assert {2, 3} <= witnesses.keys()
assert (Fraction(9, 5), Fraction(12, 5)) in (witnesses[3].first, witnesses[3].second)
assert reference.solution.preparation.reduced_root_accepted
result = continue_sketch(reference.sketch, reference,
    reference.solution.dimensions.updated({ParameterId('dependent', 'first-radius'): 3.1}),
    solve_options={'max_iterations': 100})
assert result.accepted
assert result.reference.witness_signs == reference.witness_signs
points = np.asarray(result.attempt.points)
np.testing.assert_allclose(np.linalg.norm(points[2] - points[0]), 3.1, atol=1e-8)
np.testing.assert_allclose(np.linalg.norm(points[3] - points[2]), 3.0, atol=1e-8)
np.testing.assert_allclose(result.attempt.residual, reference.sketch.residual(
    result.attempt.points, result.attempt.circle_radii, dimensions=result.attempt.dimensions), atol=1e-12)
"""
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "-c", child, str(path)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )
