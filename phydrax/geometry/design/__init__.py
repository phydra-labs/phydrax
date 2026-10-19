#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from importlib import import_module
from typing import Any

from ..._jax_config import process_numerics
from ._schema import (
    DesignState,
    ParameterBinding,
    ParameterId,
    ParameterSchema,
    ParameterSpec,
)


_NAMING_EXPORTS = frozenset(
    {
        "RecipeEntityKind",
        "RecipeEntityName",
        "RecipeEntityNames",
        "RecipeNamingError",
        "name_partition_entities",
        "transport_recipe_names",
    }
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
_SKETCH_CERTIFICATE_EXPORTS = frozenset(
    {
        "SketchCertificateStatus",
        "SketchComponentDof",
        "SketchDofCertificate",
        "SketchCertifiedSolution",
        "SketchCertifiedProfile",
        "certify_sketch_solution",
    }
)
_SKETCH_PRIORITY_EXPORTS = frozenset({"SketchPrioritySolution", "solve_priority_sketch"})
_SKETCH_DIMENSION_EXPORTS = frozenset(
    {"SketchDimensionDerivative", "sketch_dimension_derivatives"}
)


def __getattr__(name: str) -> Any:
    """Load construction layers lazily to keep the kernel/schema import acyclic."""

    with process_numerics():
        if name in _CONSTRAINT_EXPORTS:
            module = import_module("phydrax.geometry.design._constraints")
        elif name in _CONTINUATION_EXPORTS:
            module = import_module("phydrax.geometry.design._continuation")
        elif name in _SEARCH_EXPORTS:
            module = import_module("phydrax.geometry.design._search_result")
        elif name in _QUALIFICATION_EXPORTS:
            module = import_module("phydrax.geometry.design._qualification")
        elif name in _REDUCED_EXPORTS:
            module = import_module("phydrax.geometry.design._reduced")
        elif name in _SKETCH_EXPORTS:
            module = import_module("phydrax.geometry.design._sketch")
        elif name in _NAMING_EXPORTS:
            module = import_module("phydrax.geometry.design._naming")
        elif name in _SKETCH_CERTIFICATE_EXPORTS:
            module = import_module("phydrax.geometry.design._sketch_certificate")
        elif name in _SKETCH_PRIORITY_EXPORTS:
            module = import_module("phydrax.geometry.design._sketch_priority")
        elif name in _SKETCH_DIMENSION_EXPORTS:
            module = import_module("phydrax.geometry.design._sketch_dimensions")
        else:
            raise AttributeError(
                f"geometry.design.exports.name must identify a declared export; got {name!r}"
            )
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
    "RecipeEntityKind",
    "RecipeEntityName",
    "RecipeEntityNames",
    "RecipeNamingError",
    "name_partition_entities",
    "transport_recipe_names",
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
__all__ += sorted(
    _SKETCH_CERTIFICATE_EXPORTS | _SKETCH_PRIORITY_EXPORTS | _SKETCH_DIMENSION_EXPORTS
)
