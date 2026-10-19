# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""White-box polynomial reference equality and retained-source admission contracts."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._cell_geometry import CellGeometryRestrictionSource
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._nested_reference import (
    _require_polynomial_reference_action,
    rooted_nested_reference_pairs,
)
from phydrax.discretization._nested_reference_types import PolynomialReferencePair
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.rational_mapped_cells import cylinder_source, six_quad_source


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


def test_nonlinear_pairs_certify_zero_physical_map_bound(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    pairs, bound = rooted_nested_reference_pairs(
        mesh,
        geometry,
        fine_mesh,
        fine_geometry,
        parent_cells=np.zeros(6, dtype=np.int64),
        budget=budget,
    )
    assert len(pairs) == 6
    assert bound == 0.0
    for pair in pairs:
        match pair:
            case PolynomialReferencePair():
                assert pair.jacobian_bounds[0] > 0
            case _:
                raise AssertionError(
                    "The fixture must exercise nonlinear reference actions."
                )
    np.testing.assert_array_equal(fine_geometry.coordinates, controls)


def test_nonlinear_pairs_refuse_changed_source_coefficients(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    changed = np.asarray(fine_geometry.coordinates).copy()
    changed[1, 2] += 0.125
    stale = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(
                fine_mesh.blocks, fine_geometry.elements, strict=True
            )
        },
        {
            block.name: route
            for block, route in zip(
                fine_mesh.blocks, fine_geometry.geometry_dofs, strict=True
            )
        },
        changed,
        restriction_source=fine_geometry.restriction_source,
    )
    with pytest.raises(ValueError, match="nested_reference.source_root"):
        rooted_nested_reference_pairs(
            mesh,
            geometry,
            fine_mesh,
            stale,
            parent_cells=np.zeros(6, dtype=np.int64),
            budget=budget,
        )
    np.testing.assert_array_equal(fine_geometry.coordinates, controls)


def test_nonlinear_pairs_refuse_changed_retained_root_identity(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    ancestry = fine_geometry.restriction_source
    if ancestry is None:
        raise AssertionError("The fixture must retain its root coordinate identity.")
    foreign = CellGeometryRestrictionSource(
        "foreign-coordinate-source",
        ancestry.source_topology_id,
        ancestry.block_parent_cell_ids,
        ancestry.block_parent_vertex_ids,
    )
    stale = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(
                fine_mesh.blocks, fine_geometry.elements, strict=True
            )
        },
        {
            block.name: route
            for block, route in zip(
                fine_mesh.blocks, fine_geometry.geometry_dofs, strict=True
            )
        },
        controls,
        restriction_source=foreign,
    )
    with pytest.raises(ValueError, match="nested_reference.source_root"):
        rooted_nested_reference_pairs(
            mesh,
            geometry,
            fine_mesh,
            stale,
            parent_cells=np.zeros(6, dtype=np.int64),
            budget=budget,
        )


def test_reference_equality_refuses_interior_defect_with_equal_corners(
    budget: CoordinateEnclosureBudget,
) -> None:
    # u(1-u)v(1-v)/4 vanishes at all four corners but equals 1/64 at the center.
    # Coefficient equality, unlike matching corner samples, detects this defect.
    first: Polynomial = {(1, 0): Fraction(1)}
    second: Polynomial = {(0, 1): Fraction(1)}
    defective: Polynomial = {
        (1, 0): Fraction(1),
        (1, 1): Fraction(1, 4),
        (2, 1): Fraction(-1, 4),
        (1, 2): Fraction(-1, 4),
        (2, 2): Fraction(1, 4),
    }
    with pytest.raises(ValueError, match="nested_reference.arguments"):
        _require_polynomial_reference_action(
            (first, second),
            (defective, second),
            (first, second),
            budget=budget,
        )
