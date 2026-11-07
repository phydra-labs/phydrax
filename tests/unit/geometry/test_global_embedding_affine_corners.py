from __future__ import annotations

import numpy as np

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.geometry._global_embedding import certify_global_embedding


def _corner_mesh(*, overlapping: bool) -> CellMesh:
    # Exact binary64 operands from the original closed-box FAN layer source.
    # The pyramid is exactly affine; rounding makes the prism genuinely mapped.
    points = np.asarray(
        (
            (-0.32199999999999834, -0.3, -0.3),
            (-0.31905255888325934, -0.3, -0.31100000000000094),
            (-0.31270170592217006, -0.31270170592217006, -0.31270170592217006),
            (-0.33639999999999726, -0.3, -0.3),
            (-0.33639999999999726, -0.3, 0.0),
            (-0.33152332469775636, -0.3, -0.3182000000000016),
            (-0.33152332469775636, -0.3182000000000016, -0.3),
            (-0.33152332469775636, -0.3182000000000016, 0.0),
            (-0.32101554979849956, -0.32101554979849956, -0.32101554979849956),
            (-0.41095609932056326, -0.3297305972082279, -0.15),
        ),
        dtype=np.float64,
    )
    prism = np.asarray((2, 0, 1, 8, 3, 5), dtype=np.int32)
    if overlapping:
        # A proper half-turn about the shared corner preserves prism orientation.
        for axis in (0, 2):
            points[prism, axis] = 2.0 * points[3, axis] - points[prism, axis]
    return CellMesh(
        points,
        (
            CellBlock(
                "pyramid",
                "pyramid",
                np.asarray(((3, 6, 7, 4, 9),), dtype=np.int32),
                global_ids=np.asarray((75,), dtype=np.int64),
            ),
            CellBlock(
                "prism",
                "prism",
                prism[None, :],
                global_ids=np.asarray((432,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(
            (160, 214, 286, 294, 300, 348, 352, 354, 420, 431), dtype=np.int64
        ),
    )


def test_affine_pyramid_and_mapped_prism_certify_shared_corner() -> None:
    mesh = _corner_mesh(overlapping=False)
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings


def test_shared_corner_does_not_admit_overlapping_positive_maps() -> None:
    mesh = _corner_mesh(overlapping=True)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    barycentric = np.asarray((0.05, 0.75, 0.2), dtype=np.float64)
    prism_point = 0.01 * (barycentric @ points[[2, 0, 1]]) + 0.99 * (
        barycentric @ points[[8, 3, 5]]
    )
    # Independent physical interior witness: the affine parallelogram-base
    # pyramid is p0 + a*edge1 + b*edge2 + c*apex, with a,b,c>0 and a+c,b+c<1.
    pyramid_coordinates = np.linalg.solve(
        np.column_stack(
            (points[6] - points[3], points[4] - points[3], points[9] - points[3])
        ),
        prism_point - points[3],
    )
    assert np.all(pyramid_coordinates > 0.0)
    assert np.all(pyramid_coordinates[:2] + pyramid_coordinates[2] < 1.0)
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status != "certified"
    assert any(finding.check == "mapped_cell_contact" for finding in certificate.findings)


def test_affine_pyramid_and_mapped_prism_certify_shared_edge() -> None:
    # Original FAN cells 75/440 share vertices 294 and 352.
    points = np.asarray(
        (
            (-0.32199999999999834, -0.3, -0.3),
            (-0.31905255888325934, -0.31100000000000094, -0.3),
            (-0.31270170592217006, -0.31270170592217006, -0.31270170592217006),
            (-0.33639999999999726, -0.3, -0.3),
            (-0.33639999999999726, -0.3, 0.0),
            (-0.33152332469775636, -0.3182000000000016, -0.3),
            (-0.33152332469775636, -0.3182000000000016, 0.0),
            (-0.32101554979849956, -0.32101554979849956, -0.32101554979849956),
            (-0.41095609932056326, -0.3297305972082279, -0.15),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "pyramid",
                "pyramid",
                np.asarray(((3, 5, 6, 4, 8),), dtype=np.int32),
                global_ids=np.asarray((75,), dtype=np.int64),
            ),
            CellBlock(
                "prism",
                "prism",
                np.asarray(((2, 1, 0, 7, 5, 3),), dtype=np.int32),
                global_ids=np.asarray((440,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(
            (160, 218, 286, 294, 300, 352, 354, 420, 431), dtype=np.int64
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified", certificate.findings
