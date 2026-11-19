# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent source-coefficient checks for native CAD fidelity."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import CylinderPatch, LineCurve
from phydrax.meshing.providers._native_cad_bundle import (
    cad_geometry_bundle,
    coefficient_bounds,
)


def test_lazy_cylinder_coefficient_bounds() -> None:
    """Each exact cylinder coefficient, not its rounded shadow, is enclosed."""
    surface = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), float(Fraction(1, 3)), radius_exact=Fraction(1, 3),
    )
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))

    def loop(sense: int) -> tuple[PatchCurveUse, ...]:
        uses = tuple(
            PatchCurveUse(
                curve,
                LineCurve(
                    np.asarray(corners[first], dtype=np.float64),
                    np.asarray(corners[last], dtype=np.float64)
                    - np.asarray(corners[first], dtype=np.float64),
                ),
                0.0,
                1.0,
                sense=sense,
            )
            for curve, (first, last) in enumerate(pairs)
        )
        return uses if sense > 0 else tuple(reversed(uses))

    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (loop(1),)),
         MeshingSurfacePatch(surface, (loop(-1),), orientation=-1)),
        tuple(MeshingDomainCurve(first, last) for first, last in pairs),
        4,
        source_id="qualification:native-fidelity-cylinder",
        source_revision="r1",
    )
    bundle = cad_geometry_bundle(domain)
    before = bundle.model.node_reals.copy()
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.array_equal(bundle.model.node_reals, before)
    assert not lower.flags.writeable and not upper.flags.writeable
    cylinders = 0
    for patch, node in zip(domain.patches, bundle.model.surface_nodes, strict=True):
        if not isinstance(patch.surface, CylinderPatch):
            continue
        cylinders += 1
        surface = patch.surface
        exact = (
            *surface.origin_exact,
            *surface.first_axis_exact,
            *surface.second_axis_exact,
            *surface.axis_exact,
            surface.radius_exact,
        )
        offset = bundle.model.node_real_offsets[node]
        for index, value in enumerate(exact):
            assert Fraction.from_float(lower[offset + index]) <= value
            assert value <= Fraction.from_float(upper[offset + index])
    assert cylinders > 0
