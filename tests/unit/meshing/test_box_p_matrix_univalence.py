# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Whole-box univalence beyond the fixed symmetric-Jacobian criterion."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import (
    axes,
    constant,
    coordinate_expressions,
    expression_add,
    expression_bernstein_coefficients,
    expression_determinant,
    expression_evaluate,
    expression_multiply,
    expression_scale,
    Polynomial,
)
from phydrax.geometry._embedding_cells import Cell, vertices
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits
from phydrax.geometry._mapped_embedding import (
    _box_p_matrix_univalent,
    _constant_frame_nonsingular,
    _midpoint_projection,
    _multi_affine_monotone,
    injective,
    jacobian_midpoint,
    projected_jacobian,
    strict_spd,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.cut_corner_p_matrix import cut_corner_p_matrix_cases, RationalPoint
from tests._support.mesh_certificate_fixtures import tapered_hexes


_CUT_CASES = cut_corner_p_matrix_cases()


@pytest.mark.parametrize(
    ("sci", "corners"),
    _CUT_CASES,
    ids=tuple(f"cut-sci-{sci}" for sci, _ in _CUT_CASES),
)
def test_actual_cut_q1_maps_prove_univalence_after_symmetric_criterion_fails(
    sci: int, corners: tuple[RationalPoint, ...]
) -> None:
    # Full cube proofs close without any subdivision. These are the unchanged
    # source-corner laws of the 21 recorded symmetric-criterion failures.
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    element = coordinate_lagrange_element("hexahedron", 1)
    reference = vertices("hexahedron")
    images = dict(zip(reference, corners, strict=True))
    local = tuple(
        images[tuple(Fraction(float(value)) for value in point)]
        for point in np.asarray(element.reference_nodes, dtype=np.float64)
    )
    with coordinate_reference_budget(limits) as budget:
        coordinates = coordinate_expressions(element, local, budget=budget)
        assert coordinates is not None
        cell = Cell(
            coordinates, "box", "hexahedron", 3, tuple(range(8)), reference, corners
        )
        assert not _multi_affine_monotone(cell, budget=budget)
        projection = _midpoint_projection(jacobian_midpoint(cell, budget=budget), 3)
        assert projection is not None
        projected = projected_jacobian(cell, projection, budget=budget)
        assert projected is not None
        assert not strict_spd(
            projected,
            "box",
            3,
            np.zeros((3,), dtype=np.float64),
            np.eye(3, dtype=np.float64),
            {},
            budget=budget,
        )
        state = EmbeddingState([], [])
        assert injective(cell, limits, state, budget=budget) is None, f"cut SCI {sci}"
    assert state.clean
    assert state.subdivision_pieces == 1
    assert "mapped_box_p_matrix_univalence" in state.checks


def test_constructed_q1_shear_uses_positive_principal_minors_not_symmetric_parts() -> (
    None
):
    # F(u,v,w)=(u+4vw,v,w) has inverse u=x-4yz. With the midpoint inverse
    # frame its Jacobian is triangular with unit diagonal, so all seven
    # principal minors equal one. Its symmetric part is indefinite at v=w=0.
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    reference = vertices("hexahedron")
    with coordinate_reference_budget(limits) as budget:
        u, v, w = axes(3, budget=budget)
        x = expression_add(
            u,
            expression_scale(expression_multiply(v, w, budget=budget), 4, budget=budget),
            budget=budget,
        )
        coordinates = (x, v, w)
        corners = tuple(
            tuple(
                expression_evaluate(value, point, budget=budget) for value in coordinates
            )
            for point in reference
        )
        cell = Cell(
            coordinates, "box", "hexahedron", 3, tuple(range(8)), reference, corners
        )
        assert not _multi_affine_monotone(cell, budget=budget)
        state = EmbeddingState([], [])
        assert injective(cell, limits, state, budget=budget) is None
    assert state.subdivision_pieces == 1
    assert "mapped_box_p_matrix_univalence" in state.checks


@pytest.mark.parametrize("cell_index", (0, 1), ids=("left-cell", "right-cell"))
def test_strong_taper_has_positive_exact_minors_and_an_independent_inverse(
    cell_index: int,
) -> None:
    mesh = tapered_hexes(0.05)
    geometry = CellGeometrySpec.affine(mesh)
    element = geometry.elements[0]
    route = np.asarray(geometry.geometry_dofs[0], dtype=np.int64)[cell_index]
    local = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(geometry.coordinates, dtype=np.float64)[route]
    )
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    reference = vertices("hexahedron")
    with coordinate_reference_budget(limits) as budget:
        coordinates = coordinate_expressions(element, local, budget=budget)
        assert coordinates is not None
        u, v, w = axes(3, budget=budget)
        origin = expression_evaluate(coordinates[0], (Fraction(0),) * 3, budget=budget)
        top = expression_evaluate(
            coordinates[1], (Fraction(0), Fraction(1), Fraction(1)), budget=budget
        ) - expression_evaluate(
            coordinates[1], (Fraction(0), Fraction(0), Fraction(1)), budget=budget
        )
        assert 0 < top < 1
        expected_y = expression_add(
            expression_add(
                v,
                expression_scale(
                    expression_multiply(v, w, budget=budget), top - 1, budget=budget
                ),
                budget=budget,
            ),
            expression_scale(w, (1 - top) / 2, budget=budget),
            budget=budget,
        )
        # Coefficient identity proves F=(origin+u, .5+s(w)*(v-.5), w),
        # s(w)=1+(top-1)*w >= top > 0 throughout the closed cube. Independently
        # of Gale--Nikaido, its unique inverse is
        # (x-origin, .5+(y-.5)/s(z), z). No samples establish this identity.
        assert coordinates == (
            expression_add(u, constant(origin, 3, budget=budget), budget=budget),
            expected_y,
            w,
        )
        corners = tuple(
            tuple(
                expression_evaluate(value, point, budget=budget) for value in coordinates
            )
            for point in reference
        )
        cell = Cell(
            coordinates, "box", "hexahedron", 3, tuple(range(8)), reference, corners
        )
        assert not _multi_affine_monotone(cell, budget=budget)
        projection = _midpoint_projection(jacobian_midpoint(cell, budget=budget), 3)
        assert projection is not None
        np.testing.assert_array_equal(projection, np.diag((1, projection[1, 1], 1)))
        projected = projected_jacobian(cell, projection, budget=budget)
        assert projected is not None
        assert not strict_spd(
            projected, "box", 3, np.zeros(3), np.eye(3), {}, budget=budget
        )
        lower_bounds = []
        for order in (1, 2, 3):
            for indices in combinations(range(3), order):
                minor = (
                    projected[indices[0]][indices[0]]
                    if order == 1
                    else expression_determinant(
                        tuple(tuple(projected[i][j] for j in indices) for i in indices),
                        variable_dimension=3,
                        budget=budget,
                    )
                )
                controls = expression_bernstein_coefficients(
                    minor, "box", 3, budget=budget
                )
                lower_bounds.append(min(controls))
        tapered_lower = Fraction(float(projection[1, 1])) * top
        assert tuple(lower_bounds) == (
            1,
            tapered_lower,
            1,
            tapered_lower,
            1,
            tapered_lower,
            tapered_lower,
        )
        assert all(value > 0 for value in lower_bounds)
        state = EmbeddingState([], [])
        assert injective(cell, limits, state, budget=budget) is None
    assert state.subdivision_pieces == 1
    assert "mapped_box_p_matrix_univalence" in state.checks


@pytest.mark.parametrize("factor", (1, 2), ids=("zero-boundary-minor", "negative-minor"))
def test_nonpositive_box_principal_minor_preserves_unresolved_refusal(
    factor: int,
) -> None:
    # F(u,v,w)=(u,v,w(1-factor*uv)) collapses a w fiber when factor*uv=1.
    # Its midpoint frame exists, but one whole-box principal-minor lower
    # bound is zero or negative and must not be admitted.
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    reference = vertices("hexahedron")
    with coordinate_reference_budget(limits) as budget:
        u, v, w = axes(3, budget=budget)
        product = expression_multiply(
            expression_multiply(u, v, budget=budget), w, budget=budget
        )
        z = expression_add(
            w, expression_scale(product, -factor, budget=budget), budget=budget
        )
        cell = Cell((u, v, z), "box", "hexahedron", 3, tuple(range(8)), reference)
        projection = _midpoint_projection(jacobian_midpoint(cell, budget=budget), 3)
        assert projection is not None
        projected = projected_jacobian(cell, projection, budget=budget)
        assert projected is not None
        assert not strict_spd(
            projected, "box", 3, np.zeros(3), np.eye(3), {}, budget=budget
        )
        assert not _box_p_matrix_univalent(cell, projection, projected, budget=budget)
        state = EmbeddingState([], [])
        assert (
            injective(cell, limits, state, budget=budget)
            == "local_injectivity_subdivision_depth"
        )
    assert "mapped_box_p_matrix_univalence" not in state.checks
    assert state.subdivision_pieces == 1


def test_nonsingular_frame_is_required_for_box_univalence() -> None:
    with coordinate_reference_budget() as budget:
        assert not _constant_frame_nonsingular(
            np.zeros((3, 3), dtype=np.float64), 3, budget=budget
        )


def test_box_p_matrix_proof_does_not_apply_to_a_simplex() -> None:
    with coordinate_reference_budget() as budget:
        coordinates = axes(3, budget=budget)
        cell = Cell(
            coordinates,
            "simplex",
            "tetrahedron",
            3,
            tuple(range(4)),
            vertices("tetrahedron"),
        )
        projected = projected_jacobian(cell, np.eye(3, dtype=np.float64), budget=budget)
        assert projected is not None
        assert not _box_p_matrix_univalent(
            cell, np.eye(3, dtype=np.float64), projected, budget=budget
        )


def test_existing_symmetric_box_proof_keeps_its_success_path() -> None:
    limits = MeshCertificateLimits(maximum_subdivision_depth=0)
    reference = vertices("hexahedron")
    with coordinate_reference_budget(limits) as budget:
        coordinates = axes(3, budget=budget)
        cell = Cell(
            coordinates, "box", "hexahedron", 3, tuple(range(8)), reference, reference
        )
        state = EmbeddingState([], [])
        assert injective(cell, limits, state, budget=budget) is None
    assert state.clean
    assert state.subdivision_pieces == 1
    assert "mapped_box_p_matrix_univalence" not in state.checks


def test_box_p_matrix_expansion_refusal_keeps_measured_resource_evidence() -> None:
    # Polynomial identity data needs no preparation arithmetic here.
    coordinates: tuple[Polynomial, ...] = tuple(
        {tuple(int(axis == selected) for axis in range(3)): Fraction(1)}
        for selected in range(3)
    )
    jacobian = tuple(
        tuple({(0, 0, 0): Fraction(1)} if i == j else {} for j in range(3))
        for i in range(3)
    )
    cell = Cell(
        coordinates, "box", "hexahedron", 3, tuple(range(8)), vertices("hexahedron")
    )
    limits = MeshCertificateLimits(maximum_work_units=1)
    with coordinate_reference_budget(limits) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            with budget.measurement():
                _box_p_matrix_univalent(
                    cell, np.eye(3, dtype=np.float64), jacobian, budget=budget
                )
        assert caught.value.resource == "coefficient_work"
        assert caught.value.phase_evidence is not None
        assert caught.value.phase_evidence.work_units == 0
