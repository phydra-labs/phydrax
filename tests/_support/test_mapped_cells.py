"""Coordinate-map and ancestry invariants of shared mapped fixtures."""

import math

import numpy as np
import pytest

from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from phydrax.discretization.fem import FiniteElementSpec
from tests._support.mapped_cells import (
    curved_surface_chart_fv,
    curved_volume,
    dg_content,
    dg_space,
    rational_cylinder,
    six_quads,
)


# A single tabulation/matrix product uses at most 27 source nodes; 128 eps
# covers basis evaluation and the three-coordinate weighted sum at these points.
_TABULATION_ROUNDOFF = 128 * np.finfo(np.float64).eps
# pi/2 and sqrt(1/2) each introduce one rounded transcendental constant;
# 8 eps covers those reference values and their bounded sums/products.
_ARC_REFERENCE_ROUNDOFF = 8 * np.finfo(np.float64).eps
# Empirical absolute content allowance retained from HEAD's
# test_mapped_nested_dg::test_actual_scalar_dg_identity_and_component_transpose:
# degree-14 mapped physical quadrature of order-one component integrals.
_DG_CONTENT_ATOL = 1e-12


@pytest.mark.parametrize("kind", ("tetrahedron", "hexahedron", "prism", "pyramid"))
def test_curved_volume_encodes_the_shear_and_explicit_cell_identity(kind: str) -> None:
    mesh, geometry = curved_volume(kind)
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, (17,))
    np.testing.assert_array_equal(
        mesh.vertex_global_ids, np.arange(100, 100 + len(mesh.coordinates))
    )
    element = geometry.elements[0]
    assert isinstance(element, FiniteElementSpec)
    assert element.degree == 2
    point = np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64)
    basis, _ = element.tabulate(point)
    represented = np.asarray(basis) @ np.asarray(geometry.coordinates)
    expected = np.asarray(((0.25, 0.25 + 0.05 / 16, 0.25),), dtype=np.float64)
    np.testing.assert_allclose(represented, expected, rtol=0, atol=_TABULATION_ROUNDOFF)
    assert cell_geometry_id(geometry) != cell_geometry_id(CellGeometrySpec.affine(mesh))


def test_dg_space_uses_retained_coordinates_and_declared_component_shape() -> None:
    mesh, geometry = curved_volume("tetrahedron")
    space = dg_space(mesh, geometry, 1, component_shape=(3,))
    assert space.coordinate_spec is geometry
    assert space.mesh.topology_id == mesh.topology_id
    assert space.construction_plan.fields[0].component_shape == (3,)
    assert space.elements[0][0].conformity == "L2"
    assert space.elements[0][0].degree == 1


def test_dg_content_integrates_constants_and_detects_corrupted_coefficients() -> None:
    mesh, geometry = curved_volume("tetrahedron")
    space = dg_space(mesh, geometry, 0)
    coefficients = np.asarray(((3.0, 6.0),), dtype=np.float64)
    content = dg_content(space, geometry, coefficients)
    # Unit tetrahedron volume is 1/6; the shear has determinant one.
    # Reuse the mapped-DG consumer's empirical physical-content allowance.
    np.testing.assert_allclose(content, (0.5, 1.0), rtol=0, atol=_DG_CONTENT_ATOL)
    corrupted = coefficients.copy()
    corrupted[0, 0] += 6.0
    altered = dg_content(space, geometry, corrupted)
    np.testing.assert_allclose(
        altered - content, (1.0, 0.0), rtol=0, atol=_DG_CONTENT_ATOL
    )
    assert abs(altered[0] - 0.5) > _DG_CONTENT_ATOL
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(altered, (0.5, 1.0), rtol=0, atol=_DG_CONTENT_ATOL)


def test_rational_cylinder_represents_the_circular_arc_not_its_chord() -> None:
    mesh, geometry, root, controls = rational_cylinder()
    basis, _ = root.tabulate(np.asarray(((0.5, 0.5),), dtype=np.float64))
    point = np.asarray(basis) @ controls
    expected = np.asarray(((0.5, math.sqrt(0.5), math.sqrt(0.5)),), dtype=np.float64)
    np.testing.assert_allclose(point, expected, rtol=0, atol=_ARC_REFERENCE_ROUNDOFF)
    np.testing.assert_array_equal(geometry.coordinates, controls)
    assert root.source_id == "spline" and root.source_revision == "original-r1"
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        values, errors, exact = certified_cell_measures(
            mesh, geometry, relative_tolerance=0, budget=budget
        )
    assert not exact
    assert abs(values.sum() - math.pi / 2) <= errors.sum() + _ARC_REFERENCE_ROUNDOFF
    assert abs(values.sum() - math.sqrt(2)) > 0.1


def test_six_quads_retains_rational_source_ancestry_and_total_area() -> None:
    mesh, geometry, root, controls = rational_cylinder()
    quads, restricted = six_quads(mesh, geometry, root, controls)
    assert sum(block.cell_count for block in quads.blocks) == 6
    assert all(block.cell_kind == "quadrilateral" for block in quads.blocks)
    np.testing.assert_array_equal(restricted.coordinates, controls)
    assert restricted.restriction_source is not None
    assert restricted.restriction_source.source_geometry_id == cell_geometry_id(geometry)
    assert restricted.restriction_source.source_topology_id == mesh.topology_id
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        values, errors, exact = certified_cell_measures(
            quads, restricted, relative_tolerance=0, budget=budget
        )
    assert not exact
    assert abs(values.sum() - math.pi / 2) <= errors.sum() + _ARC_REFERENCE_ROUNDOFF


def test_curved_surface_chart_fv_retains_occurrences_geometry_and_distinct_inventory() -> (
    None
):
    source, target, core, carrier = curved_surface_chart_fv()
    assert source.cell_geometry is not None and target.cell_geometry is not None
    assert core.source_geometry_id == cell_geometry_id(source.cell_geometry)
    assert core.target_geometry_id == cell_geometry_id(target.cell_geometry)
    assert core.source_witness.occurrence_paths == (("assembly", "face-7"),) * 2
    assert core.target_witness.occurrence_paths == core.source_witness.occurrence_paths
    # SurfaceChartWitness canonicalizes rows by stable cell identity.
    np.testing.assert_array_equal(core.source_witness.cell_global_ids, (13, 107))
    np.testing.assert_array_equal(core.target_witness.cell_global_ids, (59, 211))
    assert np.sum(source.cell_volumes) < np.sum(target.cell_volumes)
    assert carrier.chart_deformation is core
    assert carrier.source_geometry_id == core.source_geometry_id
    assert carrier.target_geometry_id == core.target_geometry_id
    assert source.mesh.topology_id != target.mesh.topology_id
