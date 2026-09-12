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
from ._occt import (
    import_brep,
    model_from_occt_shape,
    persist_occt_shape,
    read_occt_shape,
)
from ._partition import (
    all_brep_faces,
    all_brep_solids,
    BRepPartitionHistoryError,
    BRepPartitionOperand,
    BRepPartitionPatch,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionReport,
    BRepPartitionResult,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
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
from ._planar import (
    partition_planar,
    PlanarEmbedding,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from ._source import BRep, BRepSource


__all__ = [
    "AbstractSurfacePatch",
    "BRep",
    "BRepBoundaryMap",
    "BRepEntityId",
    "BRepImportReport",
    "BRepModel",
    "BRepParameterLink",
    "BRepPartitionHistoryError",
    "BRepPartitionOperand",
    "BRepPartitionPatch",
    "BRepPartitionPlan",
    "BRepPartitionPolicy",
    "BRepPartitionRegion",
    "BRepPartitionReport",
    "BRepPartitionResult",
    "BRepPartitionRole",
    "BRepSource",
    "BRepTopology",
    "BSplineCurve",
    "BSplineSurfacePatch",
    "ConePatch",
    "CylinderPatch",
    "FixedTopologyBRepRealization",
    "FixedTopologyBRepSource",
    "PlanarEmbedding",
    "PlanarPartitionOperand",
    "PlanarPartitionPlan",
    "PlanePatch",
    "SpherePatch",
    "SurfacePatch",
    "TorusPatch",
    "all_brep_faces",
    "all_brep_solids",
    "cad_revision_from_brep_model",
    "evaluate_fixed_topology_mesh",
    "import_brep",
    "model_from_occt_shape",
    "partition_brep",
    "partition_planar",
    "persist_occt_shape",
    "read_occt_shape",
    "surface_differential",
    "surface_jacobian",
    "surface_normal",
]
