#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

from ..._jax_config import process_numerics
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
        "SketchSearchResult",
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
        "SketchDimensionBinding",
        "SketchDimensionSensitivity",
        "SketchModelResult",
        "SketchProfileResult",
    }
)

_NAMING_EXPORTS = frozenset(
    {
        "RecipeEntityNames",
        "RecipeEntityKind",
        "RecipeEntityName",
        "RecipeNamingError",
        "name_partition_entities",
        "transport_recipe_names",
    }
)

_SKETCH_PRIORITY_EXPORTS = frozenset(
    {"SketchPrioritySolution", "solve_priority_sketch"}
)
_ROOT_EVIDENCE_EXPORTS = frozenset({"RootDOFEvidence"})


def __getattr__(name: str) -> Any:
    """Load construction layers lazily to keep the kernel/schema import acyclic."""

    with process_numerics():
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
        elif name in _SKETCH_PRIORITY_EXPORTS:
            from . import _sketch_priority as module
        elif name in _NAMING_EXPORTS:
            from . import _naming as module
        elif name in _ROOT_EVIDENCE_EXPORTS:
            from . import _sketch_certificate as module
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
    "RecipeEntityNames",
    "RecipeEntityKind",
    "RecipeEntityName",
    "RecipeNamingError",
    "RootDOFEvidence",
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
    "SketchDimensionBinding",
    "SketchDimensionSensitivity",
    "SketchModelResult",
    "SketchProfileResult",
    "SketchSearchResult",
    "SketchPrioritySolution",
    "SketchSolution",
    "solve_priority_sketch",
    "name_partition_entities",
    "transport_recipe_names",
    "ReducedDesignProblem",
    "TangentCircles",
    "TangentLineCircle",
    "Vertical",
    "ParameterBinding",
    "ParameterId",
    "ParameterSchema",
    "ParameterSpec",
]
