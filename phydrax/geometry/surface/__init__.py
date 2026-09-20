#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authoritative surface geometry, audits, selections, and intersections."""

from importlib import import_module

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
from ._g1_multipatch import __all__ as _g1_multipatch_all
from ._high_order import __all__ as _high_order_all
from ._interop import __all__ as _interop_all
from ._intersection import (
    intersect_plane_surface,
    PlaneSectionEvidence,
    PlaneSectionLoop,
    PlaneSectionStatus,
    PlaneSurfaceSection,
)
from ._model import SurfaceModel, SurfaceRealization


_FACADE_EXPORT_MODULES = ("._g1_multipatch", "._high_order", "._interop")


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
__all__ += [
    name
    for name in (*_g1_multipatch_all, *_high_order_all, *_interop_all)
    if name not in __all__
]
