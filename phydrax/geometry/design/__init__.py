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
_CONTINUATION_EXPORTS = frozenset(
    {
        "CSGContinuationPolicy",
        "CSGContinuationResult",
        "PreparedCSGContinuation",
        "prepare_csg_continuation",
        "solve_csg_continuation",
    }
)
_SEARCH_EXPORTS = frozenset(
    {
        "DesignSearchResult",
    }
)
_REDUCED_EXPORTS = frozenset(
    {
        "DesignBindingGraph",
        "DesignEvaluation",
        "DesignParameterization",
        "ReducedDesignProblem",
    }
)
_QUALIFICATION_EXPORTS = frozenset(
    {
        "DerivativeTier",
        "DesignQualificationEvidence",
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
    elif name in _CONTINUATION_EXPORTS:
        from . import _continuation as module
    elif name in _SEARCH_EXPORTS:
        from . import _search as module
    elif name in _QUALIFICATION_EXPORTS:
        from . import _qualification as module
    elif name in _REDUCED_EXPORTS:
        from . import _reduced as module
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
    "CSGContinuationPolicy",
    "CSGContinuationResult",
    "DesignConstraintSystem",
    "DerivativeTier",
    "DesignBindingGraph",
    "DesignEvaluation",
    "DesignParameterization",
    "DesignQualificationEvidence",
    "DesignSearchResult",
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
    "PreparedCSGContinuation",
    "prepare_csg_continuation",
    "solve_csg_continuation",
    "Perpendicular",
    "PointDistance",
    "PointOnLine",
    "Radius",
    "Sketch",
    "SketchConstraint",
    "SketchSolution",
    "ReducedDesignProblem",
    "TangentCircles",
    "TangentLineCircle",
    "Vertical",
    "ParameterBinding",
    "ParameterId",
    "ParameterSchema",
    "ParameterSpec",
]
