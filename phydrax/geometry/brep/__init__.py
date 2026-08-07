#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._differentiable import (
    BRepParameterLink,
    evaluate_fixed_topology_mesh,
    FixedTopologyBRepRealization,
    FixedTopologyBRepSource,
)
from ._model import (
    BRepBoundaryMap,
    BRepEntityId,
    BRepImportReport,
    BRepModel,
    BRepTopology,
)
from ._occt import import_brep, model_from_occt_shape, read_occt_shape
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    ConePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
    surface_differential,
    surface_jacobian,
    surface_normal,
    SurfacePatch,
    TorusPatch,
)
from ._source import BRep, BRepSource


__all__ = [
    "AbstractSurfacePatch",
    "BRep",
    "BRepBoundaryMap",
    "BRepParameterLink",
    "BRepEntityId",
    "BRepImportReport",
    "BRepModel",
    "BRepTopology",
    "BRepSource",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "ConePatch",
    "CylinderPatch",
    "PlanePatch",
    "SpherePatch",
    "SurfacePatch",
    "TorusPatch",
    "FixedTopologyBRepRealization",
    "FixedTopologyBRepSource",
    "evaluate_fixed_topology_mesh",
    "import_brep",
    "model_from_occt_shape",
    "read_occt_shape",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
