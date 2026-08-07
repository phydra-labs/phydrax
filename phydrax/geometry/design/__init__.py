#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._schema import (
    DesignState,
    ParameterBinding,
    ParameterId,
    ParameterSchema,
    ParameterSpec,
)


_CONSTRAINT_EXPORTS = frozenset(
    {
        "AbstractDesignConstraint",
        "BRepSeamCompatibility",
        "BoundaryMeasureTarget",
        "BoundaryPoints",
        "ConstraintSolveResult",
        "DesignConstraintSystem",
        "ExteriorClearance",
        "InteriorClearance",
        "MeasureTarget",
        "ParameterEquality",
        "ParameterTarget",
    }
)
_SKETCH_EXPORTS = frozenset(
    {
        "AbstractSketchConstraint",
        "Coincident",
        "EqualLength",
        "FixedPoint",
        "Horizontal",
        "LineAngle",
        "Midpoint",
        "Parallel",
        "Perpendicular",
        "PointDistance",
        "PointOnLine",
        "Radius",
        "Sketch",
        "SketchConstraint",
        "SketchSolution",
        "TangentCircles",
        "TangentLineCircle",
        "Vertical",
    }
)


def __getattr__(name: str):
    """Load construction layers lazily to keep the kernel/schema import acyclic."""

    if name in _CONSTRAINT_EXPORTS:
        from . import _constraints as module
    elif name in _SKETCH_EXPORTS:
        from . import _sketch as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = vars(module)[name]
    globals()[name] = value
    return value


__all__ = [
    "AbstractDesignConstraint",
    "AbstractSketchConstraint",
    "BRepSeamCompatibility",
    "BoundaryMeasureTarget",
    "BoundaryPoints",
    "Coincident",
    "ConstraintSolveResult",
    "DesignConstraintSystem",
    "DesignState",
    "EqualLength",
    "ExteriorClearance",
    "FixedPoint",
    "Horizontal",
    "InteriorClearance",
    "LineAngle",
    "MeasureTarget",
    "Midpoint",
    "Parallel",
    "ParameterEquality",
    "ParameterTarget",
    "Perpendicular",
    "PointDistance",
    "PointOnLine",
    "Radius",
    "Sketch",
    "SketchConstraint",
    "SketchSolution",
    "TangentCircles",
    "TangentLineCircle",
    "Vertical",
    "ParameterBinding",
    "ParameterId",
    "ParameterSchema",
    "ParameterSpec",
]
