#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small independent line-source and grouped certificate cases."""

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry import Sphere
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.geometry._source_fidelity import (
    ImplicitBoundarySource,
    ParametricCurveBoundarySource,
    SourceFidelityBounds,
    SourceFidelityCertificate,
)
from phydrax.geometry.brep._patches import LineCurve
from phydrax.units import METER


def fidelity_triangle() -> tuple[
    CellMesh, CellGeometrySpec, ParametricCurveBoundarySource
]:
    """Build a triangle and three independent oriented line carriers."""
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    mesh = CellMesh.from_triangles(points, np.asarray(((0, 1, 2),), dtype=np.int64))
    source = ParametricCurveBoundarySource(
        (
            LineCurve((0.0, 0.0), (1.0, 0.0)),
            LineCurve((1.0, 0.0), (-1.0, 1.0)),
            LineCurve((0.0, 1.0), (0.0, -1.0)),
        ),
        ((0.0, 1.0),) * 3,
        source_id="independent-triangle-lines",
        source_revision="triangle-line-revision",
        covering_radius=0.125,
        maximum_samples=1024,
    )
    return mesh, CellGeometrySpec.affine(mesh), source


def grouped_fidelity_certificate(
    target_facet_ids: tuple[int, ...] | None = None,
) -> SourceFidelityCertificate:
    """Build directed evidence with independently inspectable scalar payloads."""
    mesh, geometry, source = fidelity_triangle()
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "affine",
        MeshCertificateLimits(),
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        (),
        tolerance=0.5,
        bounds=SourceFidelityBounds(
            semantics=("certified", "certified"),
            mesh_to_source=(0.25, 0.125),
            source_to_mesh=(0.375, 0.0625),
            sample_counts=(4, 5),
        ),
        sample_order=4,
        target_facet_ids=target_facet_ids,
        source_scope_id=None if target_facet_ids is None else "triangle-selected-scope",
        boundary_degree_id="triangle-boundary-degree",
    )


def implicit_boundary_source() -> ImplicitBoundarySource:
    """Build a sphere query whose design state supports validity refusal tests."""
    geometry = Sphere(
        (0.0, 0.0, 0.0), 1.0, feature_id="source-reconstruction-sphere"
    ).compile(coordinate_unit=METER)
    return ImplicitBoundarySource(
        geometry, source_id="source-reconstruction-sphere", spacing=0.25
    )
