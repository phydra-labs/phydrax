#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax.geometry.design._continuation import (
    SketchBranchWitness,
    SketchContinuationReference,
    SketchContinuationResult,
    continue_sketch,
)
from phydrax.geometry.design._schema import ParameterId
from phydrax.geometry.design._sketch import (
    FixedPoint,
    PointDistance,
    Sketch,
    SketchPreparationEvidence,
    SketchSolution,
)
from phydrax.geometry.design._sketch_certificate import RootDOFEvidence


_LAYOUT = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


def _solution(
    sketch: Sketch, points: ArrayLike, *, rank: int = 0, converged: bool = True
) -> SketchSolution:
    coordinates = np.asarray(points, dtype=np.float64).reshape(-1)
    residual = (
        np.empty(0, dtype=np.float64)
        if converged and rank == 0
        else np.zeros(6, dtype=np.float64)
    )
    if not converged:
        residual = np.ones(6, dtype=np.float64)
    jacobian = (
        np.empty((0, 6), dtype=np.float64)
        if residual.size == 0
        else np.diag(np.array([1.0] * rank + [0.0] * (6 - rank), dtype=np.float64))
    )
    evidence = RootDOFEvidence(
        sketch._problem_id(), coordinates, residual, jacobian, residual_tolerance=1e-9
    )
    return SketchSolution(
        points=points,
        circle_radii=np.empty(0, dtype=np.float64),
        residual=residual,
        residual_norm=np.linalg.norm(residual),
        converged=converged,
        iterations=1,
        status=0,
        root_evidence=evidence,
        preparation=SketchPreparationEvidence(
            sketch._problem_id(),
            coordinates,
            coordinates,
            (),
            np.empty(0, dtype=np.float64),
            (),
            (),
            False,
        ),
        dimensions=sketch.dimensions,
    )


def _reference(sketch: Sketch) -> SketchContinuationReference:
    return SketchContinuationReference(
        sketch, _solution(sketch, _LAYOUT), (SketchBranchWitness((0, 1, 2)),)
    )


def test_mirrored_and_degenerate_witnesses_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sketch = Sketch(_LAYOUT)
    reference = _reference(sketch)
    for points, reason in (
        (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
            "witness_changed",
        ),
        (
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
            "witness_unresolved",
        ),
    ):
        attempt = _solution(sketch, points)
        monkeypatch.setattr(Sketch, "solve", lambda self, **kwargs: attempt)
        result = continue_sketch(sketch, reference, sketch.dimensions)
        assert not result.accepted
        assert result.reason == reason
        assert result.reference is reference
        assert result.attempt is attempt
        with pytest.raises(ValueError, match="accepted transition"):
            result.dimension_sensitivity(sketch, ParameterId("test", "distance"))


def test_rank_change_and_failed_root_roll_back(monkeypatch: pytest.MonkeyPatch) -> None:
    sketch = Sketch(_LAYOUT)
    reference = _reference(sketch)
    for attempt, reason in (
        (_solution(sketch, _LAYOUT, rank=5), "rank_changed"),
        (_solution(sketch, _LAYOUT, converged=False), "not_converged"),
    ):
        monkeypatch.setattr(Sketch, "solve", lambda self, **kwargs: attempt)
        result = continue_sketch(sketch, reference, sketch.dimensions)
        assert not result.accepted
        assert result.reason == reason
        assert result.reference is reference
        assert result.attempt is attempt


def test_restoration_uses_saved_reference_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sketch = Sketch(_LAYOUT)
    reference = _reference(sketch)
    moved = _LAYOUT + np.array([2.0, 3.0], dtype=np.float64)
    attempt = _solution(sketch, moved)
    seeds: list[NDArray[np.float64]] = []

    def solve(self: Sketch, **kwargs: Any) -> SketchSolution:
        seeds.append(np.asarray(kwargs["initial_coordinates"], dtype=np.float64))
        return attempt

    monkeypatch.setattr(Sketch, "solve", solve)
    first = continue_sketch(sketch, reference, sketch.dimensions)
    assert first.accepted
    assert first.reference is not reference
    reconstructed = SketchContinuationReference(
        sketch, reference.solution, reference.witnesses
    )
    restored = continue_sketch(sketch, reconstructed, sketch.dimensions)
    assert restored.accepted
    np.testing.assert_array_equal(seeds[0], _LAYOUT.reshape(-1))
    np.testing.assert_array_equal(seeds[1], seeds[0])
    np.testing.assert_array_equal(reference.solution.points, _LAYOUT)
    np.testing.assert_array_equal(restored.reference.solution.points, moved)


def test_reference_rejects_degenerate_witness_and_other_sketch() -> None:
    sketch = Sketch(_LAYOUT, feature_id="first")
    other = Sketch(_LAYOUT, feature_id="second")
    with pytest.raises(ValueError, match="different sketch"):
        SketchContinuationReference(other, _solution(sketch, _LAYOUT), ())
    collinear = _solution(
        sketch, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="nonzero"):
        SketchContinuationReference(sketch, collinear, (SketchBranchWitness((0, 1, 2)),))


def test_dimension_edit_and_restoration_with_native_solver() -> None:
    distance = ParameterId("triangle", "distance")
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=np.float64)
    sketch = Sketch(
        points,
        feature_id="triangle",
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 2, 1.0, parameter_id=distance),
            PointDistance(1, 2, 1.0),
        ),
    )
    initial = sketch.solve()
    reference = SketchContinuationReference(
        sketch, initial, (SketchBranchWitness((0, 1, 2)),)
    )
    edited = sketch.dimensions.updated({distance: 1.1})
    result = continue_sketch(sketch, reference, edited)
    assert result.accepted
    assert result.reference.witness_signs == reference.witness_signs
    sensitivity = result.dimension_sensitivity(sketch, distance)
    assert sensitivity.parameter_id == distance
    assert sensitivity.root_evidence_id == result.attempt.root_evidence.evidence_id
    x = 1.1**2 / 2.0
    y = np.sqrt(1.1**2 - x**2)
    oracle = np.array([0.0, 0.0, 0.0, 0.0, 1.1, (1.1 - x * 1.1) / y], dtype=np.float64)
    np.testing.assert_allclose(sensitivity.coordinates, oracle, atol=1e-7)
    coordinates = jnp.asarray(result.attempt.points).reshape((-1,))

    def residual(vector: jax.Array) -> jax.Array:
        return sketch.residual(
            vector.reshape(sketch.points.shape),
            result.attempt.circle_radii,
            dimensions=result.attempt.dimensions,
        )

    tangent = jnp.asarray(oracle)
    _, jvp = jax.jvp(residual, (coordinates,), (tangent,))
    cotangent = jnp.arange(jvp.size, dtype=coordinates.dtype) + 1
    _, pullback = jax.vjp(residual, coordinates)
    vjp = pullback(cotangent)[0]
    np.testing.assert_allclose(jnp.dot(jvp, cotangent), jnp.dot(tangent, vjp), atol=1e-10)
    copied_attempt = eqx.tree_at(
        lambda value: value.points, result.attempt, result.attempt.points.copy()
    )
    reconstructed = SketchContinuationResult(
        SketchContinuationReference(
            sketch, result.reference.solution, reference.witnesses
        ),
        copied_attempt,
        SketchContinuationReference(sketch, reference.solution, reference.witnesses),
        True,
        "accepted",
    )
    assert reconstructed.attempt is not reconstructed.reference.solution
    replayed = reconstructed.dimension_sensitivity(sketch, distance)
    np.testing.assert_allclose(replayed.coordinates, sensitivity.coordinates)

    mirrored_points = points.copy()
    mirrored_points[2, 1] *= -1
    mirrored = sketch.solve(initial_coordinates=mirrored_points.reshape(-1))
    mirrored_reference = SketchContinuationReference(
        sketch, mirrored, reference.witnesses
    )
    forged = SketchContinuationResult(
        mirrored_reference, mirrored, reference, True, "accepted"
    )
    with pytest.raises(ValueError, match="witness"):
        forged.dimension_sensitivity(sketch, distance)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.attempt.points[2] - result.attempt.points[0])),
        1.1,
        atol=1e-8,
    )
    restored = continue_sketch(sketch, result.reference, sketch.dimensions)
    assert restored.accepted
    np.testing.assert_allclose(restored.attempt.points, initial.points, atol=1e-8)
    independent = continue_sketch(sketch, reference, edited)
    assert independent.accepted
    np.testing.assert_allclose(
        independent.attempt.points, result.attempt.points, atol=1e-8
    )


def test_reference_replays_root_and_rank_evidence() -> None:
    sketch = Sketch(_LAYOUT, constraints=(FixedPoint(0, _LAYOUT[0]),))
    solution = sketch.solve()
    witnesses = (SketchBranchWitness((0, 1, 2)),)
    moved = eqx.tree_at(
        lambda value: value.points,
        solution,
        solution.points.at[0, 0].set(0.5),
    )
    with pytest.raises(ValueError, match="not a converged"):
        SketchContinuationReference(sketch, moved, witnesses)

    unconstrained = Sketch(_LAYOUT)
    forged = _solution(unconstrained, _LAYOUT, rank=5)
    with pytest.raises(ValueError, match="root evidence"):
        SketchContinuationReference(unconstrained, forged, witnesses)

    changed = eqx.tree_at(
        lambda value: value.points,
        _solution(unconstrained, _LAYOUT),
        _LAYOUT + 1.0,
    )
    with pytest.raises(ValueError, match="root evidence"):
        SketchContinuationReference(unconstrained, changed, witnesses)
    forged_seed = eqx.tree_at(
        lambda value: value.preparation.seed_coordinates,
        _solution(unconstrained, _LAYOUT),
        jnp.ones((6,), dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="Preparation witnesses"):
        SketchContinuationReference(unconstrained, forged_seed, witnesses)


def test_reference_rejects_nonpositive_radius() -> None:
    sketch = Sketch(
        _LAYOUT,
        circle_centers=np.array([0], dtype=np.int32),
        circle_radii=np.array([1.0], dtype=np.float64),
    )
    solution = sketch.solve()
    invalid = eqx.tree_at(
        lambda value: value.circle_radii, solution, np.array([-1.0], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="radii"):
        SketchContinuationReference(sketch, invalid, ())


def test_changed_constraint_incidence_refuses_saved_reference() -> None:
    original = Sketch(
        _LAYOUT,
        feature_id="binding",
        constraint_ids=("fixed",),
        constraints=(FixedPoint(0, _LAYOUT[0]),),
    )
    changed = Sketch(
        _LAYOUT,
        feature_id="binding",
        constraint_ids=("fixed",),
        constraints=(FixedPoint(1, _LAYOUT[0]),),
    )
    reference = SketchContinuationReference(original, original.solve(), ())
    assert original._problem_id() != changed._problem_id()
    with pytest.raises(ValueError, match="different sketch"):
        continue_sketch(changed, reference, changed.dimensions)
