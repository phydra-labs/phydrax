# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Affine reference restriction and recorded-parent restoration of bilinear maps."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellGeometryTransitionError,
    CellMesh,
    ExactPlcCellGeometrySource,
    NestedReferenceWitnesses,
    transition_nested_cell_geometry,
)
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import (
    certify_cell_mesh,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingLimits,
    prepare_mesh_adaptation,
)
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


@pytest.fixture
def quad_case() -> tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses]:
    # F(u,v)=(u,(1+u/4)v) is bilinear, not an affine physical coordinate map.
    source = CellMesh(
        np.asarray(((0, 0), (1, 0), (1, 1.25), (0, 1)), dtype=np.float64),
        (
            CellBlock(
                "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    fine = CellMesh(
        np.asarray(
            ((0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1.125), (1, 1.25)), dtype=np.float64
        ),
        (
            CellBlock(
                "left",
                "quadrilateral",
                np.asarray(((0, 1, 4, 3),), dtype=np.int64),
                global_ids=np.asarray((10,), dtype=np.int64),
            ),
            CellBlock(
                "right",
                "quadrilateral",
                np.asarray(((1, 2, 5, 4),), dtype=np.int64),
                global_ids=np.asarray((11,), dtype=np.int64),
            ),
        ),
    )
    corners = np.asarray(
        (((0, 0), (0.5, 0), (0.5, 1), (0, 1)), ((0.5, 0), (1, 0), (1, 1), (0.5, 1))),
        dtype=np.float64,
    )
    records = NestedReferenceWitnesses(
        np.asarray((10, 11), dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        corners,
    )
    return source, CellGeometrySpec.affine(source), fine, records


def test_bilinear_quad_restriction_retains_source_law_and_measure(
    quad_case: tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses],
    budget: CoordinateEnclosureBudget,
) -> None:
    source, geometry, fine, records = quad_case
    transition = transition_nested_cell_geometry(
        source,
        geometry,
        fine,
        CellGeometrySpec.affine(fine),
        refinement=records,
        budget=budget,
    )
    np.testing.assert_array_equal(transition.vertex_coordinates, fine.coordinates)
    np.testing.assert_array_equal(transition.geometry.coordinates, geometry.coordinates)
    assert transition.evidence.kind == "nested_restriction"
    assert transition.evidence.exact
    assert transition.evidence.approximation_bound == 0.0
    assert transition.evidence.containment_defect == 0.0
    assert transition.evidence.continuity_residual == 0.0
    # Integral of det(DF)=1+u/4 over [0,1]^2 is 9/8.
    assert transition.evidence.source_measure == 1.125
    assert transition.evidence.target_measure == 1.125
    assert transition.evidence.coverage_defect == 0.0


def test_recorded_quad_siblings_restore_the_parent_coordinate_law(
    quad_case: tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses],
    budget: CoordinateEnclosureBudget,
) -> None:
    source, geometry, fine, records = quad_case
    refined = transition_nested_cell_geometry(
        source,
        geometry,
        fine,
        CellGeometrySpec.affine(fine),
        refinement=records,
        budget=budget,
    )
    restored = transition_nested_cell_geometry(
        fine,
        refined.geometry,
        source,
        geometry,
        refinement=None,
        coarsening=records,
        budget=budget,
    )
    np.testing.assert_array_equal(restored.vertex_coordinates, source.coordinates)
    np.testing.assert_array_equal(restored.geometry.coordinates, geometry.coordinates)
    assert restored.geometry.elements[0].element_id == geometry.elements[0].element_id
    assert restored.evidence.kind == "parent_restoration"
    assert restored.evidence.exact and restored.evidence.coverage_defect == 0.0


def test_quad_restoration_refuses_unrecorded_child_reference_action(
    quad_case: tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses],
    budget: CoordinateEnclosureBudget,
) -> None:
    source, geometry, fine, records = quad_case
    refined = transition_nested_cell_geometry(
        source,
        geometry,
        fine,
        CellGeometrySpec.affine(fine),
        refinement=records,
        budget=budget,
    )
    wrong: NDArray[np.float64] = records.fine_reference_vertices.copy()
    wrong[0, :, 0] += 0.125
    stale = NestedReferenceWitnesses(
        records.fine_cell_ids, records.coarse_cell_ids, wrong
    )
    before = np.asarray(refined.geometry.coordinates).copy()
    with pytest.raises(
        CellGeometryTransitionError, match="nested_geometry.child_action"
    ) as refusal:
        transition_nested_cell_geometry(
            fine,
            refined.geometry,
            source,
            geometry,
            refinement=None,
            coarsening=stale,
            budget=budget,
        )
    assert refusal.value.reason == "approximation_bound"
    np.testing.assert_array_equal(refined.geometry.coordinates, before)


def test_identity_quad_restriction_preserves_the_coordinate_element(
    quad_case: tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses],
    budget: CoordinateEnclosureBudget,
) -> None:
    source, geometry, _fine, _records = quad_case
    identity = NestedReferenceWitnesses(
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        np.asarray((((0, 0), (1, 0), (1, 1), (0, 1)),), dtype=np.float64),
    )
    transition = transition_nested_cell_geometry(
        source,
        geometry,
        source,
        geometry,
        refinement=identity,
        budget=budget,
    )
    assert transition.geometry.elements[0].element_id == geometry.elements[0].element_id
    np.testing.assert_array_equal(transition.vertex_coordinates, source.coordinates)
    np.testing.assert_array_equal(transition.geometry.coordinates, geometry.coordinates)


def test_nested_transition_refuses_wrong_nominal_mesh_at_the_boundary(
    quad_case: tuple[CellMesh, CellGeometrySpec, CellMesh, NestedReferenceWitnesses],
    budget: CoordinateEnclosureBudget,
) -> None:
    _source, geometry, fine, records = quad_case
    with pytest.raises(TypeError, match="source_mesh"):
        transition_nested_cell_geometry(
            None,  # ty: ignore[invalid-argument-type]
            geometry,
            fine,
            CellGeometrySpec.affine(fine),
            refinement=records,
            budget=budget,
        )


# Half an ulp encloses nearest-binary64 publication of the rational 1/3 source
# coefficient; it does not admit the neighboring binary64 carrier used below.
PLC_COEFFICIENT_ROUNDING_BOUND = np.spacing(np.float64(1.0 / 3.0)) / 2


@pytest.fixture
def plc_quad_case() -> tuple[CellMesh, CellGeometrySpec]:
    """One Q1 chart on a rational P1 triangle with every coefficient referenced."""
    coefficients = np.asarray(((0, 0), (1.0 / 3.0, 0), (0, 1)), dtype=np.float64)
    with coordinate_reference_budget(MeshingLimits()) as ledger:
        source = ExactPlcCellGeometrySource(
            np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray((0, 1, 0), dtype=np.int64),
            np.asarray((-1, 0, -1), dtype=np.int64),
            np.asarray(((0, 0), (1.0 / 3.0, 0), (0, 0)), dtype=np.float64),
            domain_source_id="rational-triangle-quad",
            domain_source_revision="third-coefficient",
            source_triangle_ids=np.empty(0, dtype=np.int64),
            source_triangle_bounds=np.empty(0, dtype=np.float64),
            source_segment_ids=np.asarray((7,), dtype=np.int64),
            source_segment_bounds=np.asarray(
                (PLC_COEFFICIENT_ROUNDING_BOUND,), dtype=np.float64
            ),
            vertex_parameter_ratios=(
                ((0, 1), (0, 1)),
                ((1, 3), (0, 1)),
                ((0, 1), (0, 1)),
            ),
            budget=ledger,
        )
        parent = CellMesh(
            coefficients,
            (
                CellBlock(
                    "parent",
                    "triangle",
                    np.asarray(((0, 1, 2),), dtype=np.int64),
                    global_ids=np.asarray((71,), dtype=np.int64),
                ),
            ),
            vertex_global_ids=np.asarray((37, 41, 43), dtype=np.int64),
        )
        triangle = coordinate_lagrange_element("triangle", 1)
        parent_geometry = CellGeometrySpec(
            {"parent": triangle},
            {"parent": np.asarray(((0, 1, 2),), dtype=np.int64)},
            coefficients,
            exact_source=source,
            budget=ledger,
        )
        chart = PolynomialComposedCellGeometryElement(
            triangle,
            coordinate_lagrange_element("quadrilateral", 1),
            np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int64),
            np.asarray(((1, 1), (2, 1), (3, 3), (1, 2)), dtype=np.int64),
            budget=ledger,
        )
        # Averaging the published top-edge y values gives 0.41666666666666663,
        # but its rational source image 5/12 rounds to 0.4166666666666667.
        # The mixed transition must bind this source-law carrier before measuring.
        mesh = CellMesh(
            np.asarray(
                ((0, 0), (1.0 / 6.0, 0), (1.0 / 9.0, 1.0 / 3.0), (0, 0.5)),
                dtype=np.float64,
            ),
            (
                CellBlock(
                    "cells",
                    "quadrilateral",
                    np.asarray(((0, 1, 2, 3),), dtype=np.int64),
                    global_ids=np.asarray((101,), dtype=np.int64),
                ),
            ),
            vertex_global_ids=np.asarray((37, 44, 45, 46), dtype=np.int64),
        )
        geometry = CellGeometrySpec(
            {"cells": chart},
            {"cells": np.asarray(((0, 1, 2),), dtype=np.int64)},
            coefficients,
            restriction_source=CellGeometryRestrictionSource(
                cell_geometry_id(parent_geometry),
                parent.topology_id,
                {"cells": np.asarray((71,), dtype=np.int64)},
                {"cells": np.asarray(((37, 41, 43),), dtype=np.int64)},
            ),
            exact_source=source,
            budget=ledger,
        )
    return mesh, geometry


def test_plc_quad_native_mixed_refinement_and_restoration_preserve_exact_source(
    plc_quad_case: tuple[CellMesh, CellGeometrySpec],
) -> None:
    mesh, geometry = plc_quad_case
    identity = cell_geometry_id(geometry)
    original_exact = geometry.exact_source
    assert isinstance(original_exact, ExactPlcCellGeometrySource)
    assert original_exact.vertex_parameter_ratios is not None
    source = certify_cell_mesh(mesh, SpatialCoordinateContract.si(), geometry=geometry)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_MIXED)
    refined = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, MarkedMeshAdaptation((101,)), policy=policy)
    )
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert refined.target.audit.passed
    fine = refined.target
    children = np.concatenate(
        [np.asarray(block.global_ids) for block in fine.mesh.blocks]
    )
    assert children.size == 4
    restored = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            fine,
            MarkedMeshAdaptation((), children, hierarchy=refined.hierarchy),
            policy=policy,
        )
    )
    assert restored.status is MeshAdaptationStatus.COMPLETE
    assert restored.target.audit.passed
    assert (
        restored.target.geometry.elements[0].element_id == geometry.elements[0].element_id
    )
    np.testing.assert_array_equal(restored.target.mesh.coordinates, mesh.coordinates)
    for adaptation in (refined, restored):
        transition = adaptation.geometry_transition
        assert transition is not None and transition.evidence.exact
        assert transition.evidence.coverage_defect == 0.0
        assert transition.evidence.approximation_bound == 0.0
        # Shoelace area of the authored rational corners is exactly 1/18.
        expected_measure = Fraction(1, 18)
        for measure, error in (
            (
                transition.evidence.source_measure,
                transition.evidence.source_measure_error_bound,
            ),
            (
                transition.evidence.target_measure,
                transition.evidence.target_measure_error_bound,
            ),
        ):
            assert error is not None
            assert abs(Fraction(measure) - expected_measure) <= Fraction(error)
        target_geometry = adaptation.target.geometry
        assert isinstance(target_geometry.exact_source, ExactPlcCellGeometrySource)
        assert (
            target_geometry.exact_source.domain_source_id
            == original_exact.domain_source_id
        )
        assert (
            target_geometry.exact_source.domain_source_revision
            == original_exact.domain_source_revision
        )
        assert (
            target_geometry.exact_source.vertex_parameter_ratios
            == original_exact.vertex_parameter_ratios
        )
        with coordinate_reference_budget(MeshingLimits()) as ledger:
            bank = target_geometry.source_coordinates(budget=ledger)
        assert bank == (
            (Fraction(0), Fraction(0)),
            (Fraction(1, 3), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        np.testing.assert_array_equal(target_geometry.coordinates, geometry.coordinates)
        ancestry = target_geometry.restriction_source
        assert ancestry is not None and geometry.restriction_source is not None
        assert (
            ancestry.source_geometry_id == geometry.restriction_source.source_geometry_id
        )
        assert (
            ancestry.source_topology_id == geometry.restriction_source.source_topology_id
        )
        for parent_ids in ancestry.block_parent_cell_ids.values():
            np.testing.assert_array_equal(
                parent_ids, np.full(parent_ids.shape, 71, dtype=np.int64)
            )
    assert cell_geometry_id(geometry) == identity


@pytest.mark.parametrize("tampering", ("outside-reference", "rounded-coefficient"))
def test_plc_quad_restriction_refuses_incompatible_map_without_mutating_source(
    plc_quad_case: tuple[CellMesh, CellGeometrySpec], tampering: str
) -> None:
    mesh, geometry = plc_quad_case
    identity = cell_geometry_id(geometry)
    before = np.asarray(geometry.coordinates).copy()
    coefficients = before.copy()
    element = geometry.elements[0]
    assert isinstance(element, PolynomialComposedCellGeometryElement)
    if tampering == "outside-reference":
        element = RestrictedCellGeometryElement(
            element,
            "quadrilateral",
            np.eye(2, dtype=np.float64),
            np.asarray((0.75, 0), dtype=np.float64),
        )
        message = r"require_plc_source_layout\.restriction"
    else:
        coefficients[1, 0] = np.nextafter(coefficients[1, 0], np.inf)
        message = r"ExactPlcCellGeometrySource\.source_row_bounds"
    with coordinate_reference_budget(MeshingLimits()) as ledger:
        with pytest.raises(ValueError, match=message):
            incompatible = CellGeometrySpec(
                {"cells": element},
                {"cells": geometry.geometry_dofs[0]},
                coefficients,
                exact_source=geometry.exact_source,
                restriction_source=geometry.restriction_source,
                budget=ledger,
            )
            incompatible.resolve(mesh, budget=ledger)
    assert cell_geometry_id(geometry) == identity
    np.testing.assert_array_equal(geometry.coordinates, before)
