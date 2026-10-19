"""Small source-bound spline charts for generation accounting contracts."""

from __future__ import annotations

import numpy as np

from phydrax.geometry import (
    BSplineSurfacePatch,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)


def accounting_spline_domain() -> MeshingDomain:
    """Build the bilinear rational source used by the surface accounting regression.

    The controls and weights preserve the source fixture from
    test_native_surface_generation's source-budget regression.
    """
    corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    loop = tuple(
        PatchCurveUse(
            row,
            LineCurve(head, corners[(row + 1) % 4] - head),
            0.0,
            1.0,
        )
        for row, head in enumerate(corners)
    )
    patch = BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.05]], [[1.0, 0.0, 0.1], [1.0, 1.0, 0.2]]],
        [[1.0, 0.9], [1.1, 1.0]],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="native-spline-normal-owner",
        source_revision="authored",
    )
