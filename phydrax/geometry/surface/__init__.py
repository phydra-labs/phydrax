#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authoritative surface geometry, triangle arrangements, Booleans, and audits."""

from importlib import import_module
from typing import Any

from ..._jax_config import process_numerics
from ._contracts import (
    InterfaceSide,
    SurfaceAuditPolicy,
    SurfaceAuditReport,
    SurfaceChartMappingEvidence,
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceOrientationRepair,
    SurfacePreparationError,
    SurfacePreparationStatus,
    SurfaceSelection,
    SurfaceValidityCertificate,
)


_EXPORT_GROUPS = (
    (
        "._arrangement",
        (
            "arrange_triangle_surfaces",
            "SurfaceArrangement",
            "SurfaceArrangementError",
            "SurfaceArrangementEvidence",
            "SurfaceArrangementStatus",
        ),
    ),
    (
        "._boolean",
        (
            "surface_boolean",
            "SurfaceBooleanError",
            "SurfaceBooleanOperation",
            "SurfaceBooleanResult",
            "SurfaceBooleanStatus",
            "SurfaceClosureEvidence",
        ),
    ),
    (
        "._g1_multipatch",
        (
            "BiquinticGluingBasis",
            "BiquinticGluingConstraints",
            "G1PatchStatus",
            "GluingContinuity",
            "PatchEdge",
            "PatchTopologyError",
            "PatchVertexKind",
            "PreparedHalfEdgePatchTopology",
            "ProjectedBiquinticCoefficients",
            "SeamContinuityEvidence",
            "UnsupportedG1TopologyError",
        ),
    ),
    (
        "._high_order",
        (
            "HighOrderDifferentiationError",
            "HighOrderGeometryMismatchError",
            "HighOrderSurfaceFrameEvidence",
            "HighOrderSurfacePolicy",
            "HighOrderSurfaceRealization",
            "HighOrderSurfaceReport",
            "HighOrderSurfaceSource",
            "isoparametric_triangle_reference_nodes",
            "realize_boundary_atlas",
            "realize_bspline_surface",
            "realize_isoparametric_triangles",
        ),
    ),
    (
        "._interop",
        (
            "PortableSurfaceField",
            "SurfaceDataCorruptionError",
            "SurfaceExportPolicy",
            "SurfaceExportResult",
            "SurfaceFieldAssociation",
            "SurfaceFieldRecord",
            "SurfaceFileFormat",
            "SurfaceImportPolicy",
            "SurfaceImportResult",
            "SurfaceInteropError",
            "SurfaceInteropReport",
            "SurfaceLossyOperationError",
            "SurfaceOrientationPolicy",
            "SurfaceProviderUnavailableError",
            "SurfaceResourceLimitError",
            "SurfaceUnsupportedFormatError",
            "export_surface",
            "import_surface",
        ),
    ),
)
from ._intersection import (
    intersect_plane_surface,
    PlaneSectionEvidence,
    PlaneSectionLoop,
    PlaneSectionStatus,
    PlaneSurfaceSection,
)
from ._model import SurfaceModel, SurfaceRealization


_EXPORT_MODULE_BY_NAME = {
    name: module_name for module_name, names in _EXPORT_GROUPS for name in names
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with process_numerics():
        module = import_module(module_name, __package__)
        value = getattr(module, name)
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "InterfaceSide",
    "PlaneSectionEvidence",
    "PlaneSectionLoop",
    "PlaneSectionStatus",
    "PlaneSurfaceSection",
    "SurfaceAuditPolicy",
    "SurfaceAuditReport",
    "SurfaceChartMappingEvidence",
    "SurfaceInterface",
    "SurfaceMetadata",
    "SurfaceModel",
    "SurfaceOrientationRepair",
    "SurfacePreparationError",
    "SurfacePreparationStatus",
    "SurfaceRealization",
    "SurfaceSelection",
    "SurfaceValidityCertificate",
    "intersect_plane_surface",
]
__all__ += [name for _, names in _EXPORT_GROUPS for name in names if name not in __all__]
