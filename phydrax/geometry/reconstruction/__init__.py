from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "reconstruct_dem_region": "._core",
    "reconstruct_lidar_region": "._core",
    "reconstruct_planar_region": "._core",
    "reconstruct_point_region": "._core",
    "reconstruct_surface_region": "._core",
    "ReconstructedGeometrySource": "._core",
    "ReconstructionFailure": "._core",
    "ReconstructionReport": "._core",
    "ReconstructionReportProvider": "._core",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._core import (
        reconstruct_dem_region,
        reconstruct_lidar_region,
        reconstruct_planar_region,
        reconstruct_point_region,
        reconstruct_surface_region,
        ReconstructedGeometrySource,
        ReconstructionFailure,
        ReconstructionReport,
        ReconstructionReportProvider,
    )


__all__ = [
    "ReconstructedGeometrySource",
    "ReconstructionFailure",
    "ReconstructionReport",
    "ReconstructionReportProvider",
    "reconstruct_dem_region",
    "reconstruct_lidar_region",
    "reconstruct_point_region",
    "reconstruct_planar_region",
    "reconstruct_surface_region",
]
