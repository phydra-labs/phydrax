#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import numpy as np
import pytest

from phydrax.geometry.design import (
    continue_sketch,
    FixedPoint,
    ParameterId,
    PointDistance,
    Radius,
    Sketch,
    SketchBranchWitness,
    SketchContinuationReference,
    SketchContinuationResult,
    SketchSolution,
)
from phydrax.geometry.design._continuation import (
    _same_array_bytes,
    SketchContinuationReferenceArchiveRecipe,
)
from phydrax.geometry.design._sketch_certificate import RootDOFEvidence


def test_source_bytes_compare_exactly_across_storage_layouts() -> None:
    source = np.arange(12, dtype=np.float64).reshape((3, 4)).T
    contiguous = np.ascontiguousarray(source)
    assert _same_array_bytes(source, contiguous)
    plus_zero = np.asarray([0.0], dtype=np.float64)
    minus_zero = np.asarray([-0.0], dtype=np.float64)
    assert not _same_array_bytes(plus_zero, minus_zero)


def _sketch() -> Sketch:
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], dtype=np.float64)
    return Sketch(
        points,
        circle_centers=np.array([0], dtype=np.int32),
        circle_radii=np.array([1.0], dtype=np.float64),
        constraints=(
            FixedPoint(0, points[0]),
            FixedPoint(1, points[1]),
            PointDistance(0, 1, 2.0),
            PointDistance(0, 2, 2.0, parameter_id=ParameterId("triangle", "distance")),
            PointDistance(1, 2, 2.0),
            Radius(0, 1.0),
        ),
        feature_id="triangle",
    )


def _reference(sketch: Sketch, sign: int) -> SketchContinuationReference:
    initial = np.concatenate(
        (np.asarray(sketch.points).reshape(-1), np.asarray(sketch.circle_radii))
    )
    initial[5] *= sign
    solution = sketch.solve(initial_coordinates=initial, max_iterations=100)
    assert bool(solution.converged)
    return SketchContinuationReference(
        sketch, solution, (SketchBranchWitness((0, 1, 2)),)
    )


def test_consistent_mirrored_output_cannot_replace_checkpoint_branch() -> None:
    sketch = _sketch()
    upper = SketchContinuationReferenceArchiveRecipe.from_reference(_reference(sketch, 1))
    lower = SketchContinuationReferenceArchiveRecipe.from_reference(
        _reference(sketch, -1)
    )
    assert upper.restore().witness_signs == (1,)
    assert lower.restore().witness_signs == (-1,)
    forged = lower._replace(
        checkpoint_coordinates=upper.checkpoint_coordinates,
        checkpoint_dimensions=upper.checkpoint_dimensions,
        checkpoint_max_iterations=upper.checkpoint_max_iterations,
        checkpoint_tolerance=upper.checkpoint_tolerance,
        checkpoint_damping=upper.checkpoint_damping,
    )
    with pytest.raises(ValueError, match="checkpoint replay"):
        forged.restore()


@pytest.mark.parametrize("changed_field", ["points", "radii", "constraint"])
def test_changed_source_is_rejected_by_continuation_and_derivatives(
    changed_field: str,
) -> None:
    sketch = _sketch()
    reference = _reference(sketch, 1)
    result = continue_sketch(sketch, reference, reference.solution.dimensions)
    assert result.accepted
    if changed_field == "points":
        changed = eqx.tree_at(
            lambda value: value.points, sketch, sketch.points.at[2, 0].add(0.25)
        )
    elif changed_field == "radii":
        changed = eqx.tree_at(
            lambda value: value.circle_radii, sketch, sketch.circle_radii.at[0].set(1.25)
        )
    else:
        changed = eqx.tree_at(
            lambda value: value.constraints[0].target,
            sketch,
            sketch.constraints[0].target.at[0].set(0.25),
        )
    with pytest.raises(ValueError, match="different sketch source"):
        continue_sketch(changed, reference, reference.solution.dimensions)
    parameter = ParameterId("triangle", "distance")
    with pytest.raises(ValueError, match="different sketch source"):
        result.dimension_sensitivity(changed, parameter)
    with pytest.raises(ValueError, match="different sketch source"):
        result.dimension_jvp(changed, {parameter: 1.0})
    with pytest.raises(ValueError, match="different sketch source"):
        result.dimension_vjp(changed, np.zeros(7, dtype=np.float64), (parameter,))


def test_derivative_rejects_unrelated_same_branch_prior() -> None:
    sketch = _sketch()
    reference = _reference(sketch, 1)
    parameter = ParameterId("triangle", "distance")
    result = continue_sketch(sketch, reference, reference.solution.dimensions)
    altered = continue_sketch(
        sketch, reference, reference.solution.dimensions.updated({parameter: 2.1})
    )
    assert result.accepted and altered.accepted
    forged = eqx.tree_at(lambda value: value.prior_reference, result, altered.reference)
    with pytest.raises(ValueError, match="checkpoint"):
        forged.dimension_sensitivity(sketch, parameter)


def test_derivative_replays_checkpoint_for_forged_regular_endpoint() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    sketch = Sketch(points, feature_id="free-source")
    prior_solution = sketch.solve()
    prior = SketchContinuationReference(sketch, prior_solution, ())
    moved = points + np.asarray([3.0, 4.0], dtype=np.float64)
    residual = np.empty((0,), dtype=np.float64)
    evidence = RootDOFEvidence(
        sketch._problem_id(),
        moved.reshape((-1,)),
        residual,
        np.empty((0, moved.size), dtype=np.float64),
        residual_tolerance=1e-9,
    )
    forged_solution = SketchSolution(
        points=moved,
        circle_radii=np.empty((0,), dtype=np.float64),
        residual=residual,
        residual_norm=0.0,
        converged=True,
        iterations=0,
        status=0,
        root_evidence=evidence,
        preparation=prior_solution.preparation,
        dimensions=sketch.dimensions,
    )
    forged_reference = SketchContinuationReference(sketch, forged_solution, ())
    forged = SketchContinuationResult(
        forged_reference, forged_solution, prior, True, "accepted"
    )
    with pytest.raises(ValueError, match="checkpoint replay"):
        forged.dimension_jvp(sketch, {})
