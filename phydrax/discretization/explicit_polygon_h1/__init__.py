#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit lowest-order H1 bases on conforming star-shaped polygon meshes."""

from ._basis import ExplicitPolygonH1BasisEvidence, ExplicitPolygonH1BlockData
from ._constraints import (
    explicit_polygon_h1_dirichlet_constraint,
    ExplicitPolygonH1DirichletConstraint,
)
from ._dofs import ExplicitPolygonH1DofMap
from ._precision import ExplicitPolygonH1PrecisionPolicy
from ._reconstruction import (
    evaluate_explicit_polygon_h1_reconstruction,
    evaluate_explicit_polygon_h1_trace,
    ExplicitPolygonH1Reconstruction,
    prepare_explicit_polygon_h1_reconstruction,
)
from ._space import (
    ExplicitPolygonH1Discretization,
    ExplicitPolygonH1Plan,
    ExplicitPolygonH1RuntimeData,
)
from ._spec import (
    ExplicitPolygonH1FieldSpec,
    ExplicitPolygonH1QuadraturePolicy,
    ExplicitPolygonH1QualificationPolicy,
    ExplicitPolygonH1ResourceBudget,
)


__all__ = [
    "ExplicitPolygonH1BasisEvidence",
    "ExplicitPolygonH1BlockData",
    "ExplicitPolygonH1DirichletConstraint",
    "ExplicitPolygonH1Discretization",
    "ExplicitPolygonH1DofMap",
    "ExplicitPolygonH1FieldSpec",
    "ExplicitPolygonH1Plan",
    "ExplicitPolygonH1PrecisionPolicy",
    "ExplicitPolygonH1QuadraturePolicy",
    "ExplicitPolygonH1QualificationPolicy",
    "ExplicitPolygonH1Reconstruction",
    "ExplicitPolygonH1ResourceBudget",
    "ExplicitPolygonH1RuntimeData",
    "evaluate_explicit_polygon_h1_reconstruction",
    "evaluate_explicit_polygon_h1_trace",
    "explicit_polygon_h1_dirichlet_constraint",
    "prepare_explicit_polygon_h1_reconstruction",
]
