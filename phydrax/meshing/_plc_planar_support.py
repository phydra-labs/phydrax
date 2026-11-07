#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent oriented source domains for planar PLC routes."""

from __future__ import annotations

import numpy as np

from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..typing import checked
from .providers._native_sources import NativePlanarSource


@checked
def planar_region_domain(
    source: NativePlanarSource, /
) -> PiecewiseLinearDomain:
    """Declare the full outer boundary and holes of one planar source region."""
    region = source.region
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != edges.shape[0]
        or np.any(offsets[1:] - offsets[:-1] < 3)
    ):
        raise ValueError("Planar source loop offsets must partition every oriented edge cycle.")
    # PlanarMeshRegion requires a CCW exterior and CW holes, so the fluid is
    # on the same side of every directed edge.
    pairs = np.tile(np.asarray((0, -1), dtype=np.int64), (edges.shape[0], 1))
    return PiecewiseLinearDomain(
        vertices,
        edges,
        pairs,
        (source.region_entity_id,),
        source_id=source.source_id,
    )


__all__ = ["planar_region_domain"]
