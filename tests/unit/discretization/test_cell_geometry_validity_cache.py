# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Validity reuse identity at serial and owner-local host boundaries."""

from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    _validity_request_key,
    CellValidityPolicy,
)
from tests._support.cell_meshes import unit_square


def test_serial_validity_request_key_preserves_full_content_identity() -> None:
    mesh = unit_square()
    geometry = CellGeometrySpec.affine(mesh)
    # Observed full-content digest before the distributed-storage key cutover;
    # lf19_serial_key_identity.log records its byte-identical current value.
    assert _validity_request_key(geometry, mesh, CellValidityPolicy()) == (
        "dd204a6872746f48b9b3bd30b6fecee1240d47585ef85bca54e875d1f44f4b55"
    )
