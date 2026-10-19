# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Dimension derivatives through the native implicit sketch-root attachment."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from ._sketch import _SketchResidual, LineAngle, PointDistance, Radius
from ._sketch_certificate import SketchCertifiedSolution
from ._solve import attach_sketch_sensitivity


@final
class SketchDimensionDerivative(StrictModule):
    """Rows follow flattened points then radii; columns follow dimension shape."""

    constraint_id: str = eqx.field(static=True)
    dimension_shape: tuple[int, ...] = eqx.field(static=True)
    derivative: Array


@checked
def sketch_dimension_derivatives(
    certificate: SketchCertifiedSolution,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SketchDimensionDerivative, ...]:
    """Refuse nonregular sketches; differentiate distance, radius and angle data.

    Native ``implicit_root_output`` supplies the derivative, rather than the LM
    iteration path. Rank certification is mandatory; enclosure support is an
    independent capability and is reported separately by the certificate.
    """
    solution = certificate.solution
    if (
        not certificate.dof.fully_constrained
        or not bool(solution.converged)
        or not bool(solution.derivative_evidence.successful)
    ):
        raise ValueError(
            f"SketchDimensionDerivative.solution must be fully constrained and regular with certified rank; got certificate={certificate.status}, derivative_status={solution.derivative_evidence.status}"
        )
    sketch = solution.sketch
    size = solution.points.size + solution.circle_radii.size
    residual_bound = float(solution.residual_bound)
    derivatives = []
    for index, (identity, constraint) in enumerate(
        zip(sketch.constraint_ids, sketch.constraints, strict=True)
    ):
        match constraint:
            case PointDistance():
                dimension, select = constraint.distance, lambda value: value.distance
            case Radius():
                dimension, select = constraint.radius, lambda value: value.radius
            case LineAngle():
                dimension, select = constraint.angle, lambda value: value.angle
            case _:
                continue
        budget.reserve(64 * max(1, size) ** 3, 512 * max(1, size) ** 2)

        def root(value: Array) -> Array:
            revised_constraint = eqx.tree_at(select, constraint, value)
            revised_sketch = eqx.tree_at(
                lambda owner: owner.constraints[index], sketch, revised_constraint
            )
            attached, evidence, _ = attach_sketch_sensitivity(
                _SketchResidual(revised_sketch),
                solution.solver_result,
                residual_bound,
            )
            return eqx.error_if(
                attached,
                ~evidence.successful,
                "SketchDimensionDerivative.evidence must be successful for a native implicit-root action; got failed sensitivity evidence",
            )

        derivative = jax.jacfwd(root)(dimension)
        derivatives.append(
            SketchDimensionDerivative(identity, dimension.shape, derivative)
        )
    return tuple(derivatives)


__all__ = ["SketchDimensionDerivative", "sketch_dimension_derivatives"]
