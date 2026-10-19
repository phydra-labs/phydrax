#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts for binding a proved support map to an overlay's source faces."""

import pytest

from phydrax import geometry
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._boolean_coincidence import (
    CoincidentFaceOverlay,
    CoincidentSurfaceCorrespondence,
    prove_surface_correspondence,
)
from phydrax.units import MILLIMETER


@pytest.fixture(scope="module")
def box() -> geometry.BRepModel:
    return geometry.brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract(MILLIMETER),
    )


def _identity(model: geometry.BRepModel, face: int) -> CoincidentSurfaceCorrespondence:
    proof = prove_surface_correspondence(model.patches[face], model.patches[face])
    if proof is None:
        raise ValueError(
            "test_coincident_face_overlay.proof must prove self-correspondence; got None"
        )
    return proof


def test_overlay_retains_the_supplied_proof_and_source_entities(
    box: geometry.BRepModel,
) -> None:
    proof = _identity(box, 0)
    overlay = CoincidentFaceOverlay(box, 0, box, 0, proof, geometry.BRepBooleanPolicy())
    assert overlay.correspondence is proof
    assert overlay.first_face_id == box.face_ids[0]
    assert overlay.second_face_id == box.face_ids[0]
    source = box.require_geometry()
    assert tuple(arc.coedge for loop in overlay.first_loops for arc in loop) == tuple(
        coedge for loop in source.face_loops[0] for coedge in loop
    )
    assert tuple(arc.coedge for loop in overlay.second_loops for arc in loop) == tuple(
        coedge for loop in source.face_loops[0] for coedge in loop
    )


def test_overlay_refuses_a_proof_for_different_faces(box: geometry.BRepModel) -> None:
    proof = _identity(box, 1)
    with pytest.raises(ValueError, match="CoincidentFaceOverlay.correspondence"):
        CoincidentFaceOverlay(box, 0, box, 0, proof, geometry.BRepBooleanPolicy())
