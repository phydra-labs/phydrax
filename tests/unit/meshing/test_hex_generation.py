#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, cast, Never

import equinox as eqx
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import (
    certify_domain_coverage,
    DomainCoverageCertificate,
)
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._contracts import CellFamilyPolicy, MeshingLimits
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._hex_dominant import extract_hex_dominant
from phydrax.meshing._hex_generation import (
    extract_volume_hexes,
    generate_integer_grid_hexes,
    NativeHexGridRoute,
    NativeHexGridSchedule,
    realize_mapped_grid_hexes,
)
from phydrax.meshing._host_banks import entities
from phydrax.meshing._organization_records import (
    MeshPatch,
    MeshPatchBank,
    MeshZone,
    MeshZoneRole,
)
from phydrax.meshing._quad_generation import remap_dual_metadata
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.hex_fields import (
    allfield_inventory,
    allfield_state,
    allfield_transfer,
    quadratic_source,
    quadratic_value,
    TRANSFER_TOLERANCE,
    TransferEvidence,
)
from tests._support.hex_sources import (
    hex_volumes,
    mapped_occupied_source,
    nonunit_material_publication,
    occupied_cells,
    two_tetrahedra,
)


# For the small positive-volume fixtures, sequential quadrature and contraction
# roundoff is bounded by gamma_4096 < 1e-12 in binary64 (Higham, Accuracy and
# Stability of Numerical Algorithms, second edition, section 3.1). Scaled
# coverage measures use rational integration with one directed endpoint rounding.
# Preserve the existing absolute/relative comparison limit.
VOLUME_TOLERANCE = 1.0e-12
# Fixed mapped-FV qualification bounds retained from
# tools/meshing_qualification/scenarios/family.py:_family_mapped_fv.
FV_VOLUME_RTOL = 1.0e-10
STATIONARY_EULER_ATOL = 1.0e-10


def test_all_hex_template_has_positive_maps_and_exact_volume() -> None:
    source = two_tetrahedra()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert result.validity.certified_valid_count == 8
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        volumes = hex_volumes(result.mesh)
        np.testing.assert_allclose(
            volumes, np.full((8,), 1.0 / 24.0), rtol=VOLUME_TOLERANCE
        )
        assert np.sum(volumes) == pytest.approx(1.0 / 3.0)
        assert result.mesh.topology.entities(2).count == 33
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        np.testing.assert_array_equal(
            np.diff(result.mesh.connectivity.face_vertex_offsets), 4
        )
        np.testing.assert_array_equal(
            np.diff(result.mesh.connectivity.cell_face_offsets), 6
        )
        assert isinstance(source.connectivity, TetrahedralConnectivity)
        assert np.max(np.asarray(result.mesh.connectivity.face_cell_counts)) == 2
        source_edges = np.asarray(source.coordinates)[
            np.asarray(source.connectivity.edges)
        ]
        target_edges = np.asarray(result.mesh.coordinates)[
            np.asarray(result.mesh.connectivity.edges)
        ]
        source_maximum = np.max(
            np.linalg.norm(source_edges[:, 1] - source_edges[:, 0], axis=1)
        )
        target_maximum = np.max(
            np.linalg.norm(target_edges[:, 1] - target_edges[:, 0], axis=1)
        )
        assert target_maximum <= 0.5 * source_maximum


def test_dual_hex_maps_preserve_original_source_bank_and_exact_fractional_corners() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        prepared_coordinate_source_bank,
        rounded_point,
    )

    source = two_tetrahedra()
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert result.geometry is not None and result.source_geometry is not None
        np.testing.assert_array_equal(
            np.asarray(result.geometry.coordinates).view(np.uint64),
            np.asarray(result.source_geometry.coordinates).view(np.uint64),
        )
        bank = prepared_coordinate_source_bank(result.geometry, budget=reference_budget)
        elements, routes, _ = result.geometry.resolve(result.mesh)
        source_points = np.asarray(source.coordinates)
        for block, element, block_routes in zip(
            result.mesh.blocks, elements, routes, strict=True
        ):
            for cell, route in zip(
                np.asarray(block.vertices), np.asarray(block_routes), strict=True
            ):
                images = coordinate_corner_images(
                    element,
                    tuple(bank[int(index)] for index in route),
                    budget=reference_budget,
                )
                assert images is not None
                for vertex, actual in zip(cell, images, strict=True):
                    support = result.vertex_supports[vertex]
                    support = support[support >= 0]
                    expected = tuple(
                        sum(
                            (
                                Fraction(float(source_points[index, axis]))
                                for index in support
                            ),
                            Fraction(0),
                        )
                        / support.size
                        for axis in range(3)
                    )
                    assert actual == expected
                    np.testing.assert_array_equal(
                        result.mesh.coordinates[vertex], rounded_point(expected)
                    )


def test_exact_hex_quality_accepts_authoritative_unit_map_at_quality_boundary() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality

    points = np.asarray(reference_cell_topology("hexahedron").vertices)
    mesh = CellMesh(points, (CellBlock("authored", "hexahedron", np.arange(8)[None, :]),))
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0,
        minimum_mean_ratio=1.0,
        maximum_aspect_ratio=1.0,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        scaled, mean, aspect = exact_mapped_volume_quality(
            mesh,
            CellGeometrySpec.affine(mesh),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(scaled >= schedule.minimum_scaled_jacobian)
        assert np.all(mean >= schedule.minimum_mean_ratio)
        assert np.all(aspect <= schedule.maximum_aspect_ratio)


@pytest.mark.parametrize("count", (1, 8))
def test_exact_hex_quality_retires_cell_polynomial_preparations(count: int) -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality

    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    points = np.concatenate(
        [reference + (2 * index, 0, 0) for index in range(count)]
    )
    mesh = CellMesh(
        points,
        (CellBlock("authored", "hexahedron", np.arange(8 * count).reshape(count, 8)),),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        before_temporary = coordinate_budget.temporary_bytes_upper
        before_work = coordinate_budget.work_units
        before_bernstein = dict(coordinate_budget.bernstein_coefficient_cache)
        before_affine = dict(coordinate_budget.axis_affine_polynomial_cache)
        scaled, mean, aspect = exact_mapped_volume_quality(
            mesh,
            CellGeometrySpec.affine(mesh),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        np.testing.assert_array_equal(scaled, np.ones(count))
        np.testing.assert_array_equal(mean, np.ones(count))
        np.testing.assert_array_equal(aspect, np.ones(count))
        assert coordinate_budget.work_units > before_work
        assert coordinate_budget.temporary_bytes_upper == before_temporary
        assert dict(coordinate_budget.bernstein_coefficient_cache) == before_bernstein
        assert dict(coordinate_budget.axis_affine_polynomial_cache) == before_affine


@pytest.mark.parametrize("kind", ("hexahedron", "tetrahedron", "pyramid"))
def test_exact_hex_quality_reused_minors_match_reference_determinant(kind: str) -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._coordinate_enclosure import (
        bernstein_coefficients,
        coordinate_polynomials,
        determinant_polynomial,
        prepared_coordinate_source_bank,
    )
    from phydrax.meshing._hex_generation import _prepare_mapped_quality_source

    reference = np.asarray(reference_cell_topology(kind).vertices)
    matrix = np.asarray(((1.0, 0.25, 0.0), (0.0, 1.0, 0.125), (0.25, 0.0, 1.0)))
    points = reference @ matrix.T
    mesh = CellMesh(
        points,
        (CellBlock("authored", kind, np.arange(points.shape[0])[None, :]),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    with coordinate_reference_budget() as coordinate_budget:
        elements, routes, _ = geometry.resolve(
            mesh, budget=coordinate_budget, exact_source_prepared=False
        )
        coordinates = prepared_coordinate_source_bank(geometry, budget=coordinate_budget)
        route = np.asarray(routes[0][0])
        domain = "simplex" if kind == "tetrahedron" else "box"
        source = _prepare_mapped_quality_source(
            elements[0], coordinates, route, kind, domain, budget=coordinate_budget
        )
        polynomials = coordinate_polynomials(
            elements[0],
            tuple(coordinates[int(index)] for index in route),
            budget=coordinate_budget,
        )
        assert polynomials is not None
        determinant = determinant_polynomial(
            polynomials, kind, 3, budget=coordinate_budget
        )
        assert determinant is not None
        expected = min(
            bernstein_coefficients(determinant, domain, 3, budget=coordinate_budget)
        )
        assert source.determinant_lower == expected


@pytest.mark.parametrize("degree", (1, 2))
def test_exact_hex_quality_bernstein_nets_match_or_tighten_monomial_route(
    degree: int,
) -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_polynomials,
        prepared_coordinate_source_bank,
    )
    from phydrax.meshing._hex_generation import (
        _box_bernstein_quality_source,
        _monomial_quality_source,
    )

    points = np.asarray(reference_cell_topology("hexahedron").vertices)
    mesh = CellMesh(points, (CellBlock("authored", "hexahedron", np.arange(8)[None, :]),))
    element = coordinate_lagrange_element("hexahedron", degree)
    nodes = np.asarray(element.reference_nodes)
    # A dual-hex-like map coupling every coordinate trilinearly, plus a curved
    # interior at degree two; dyadic coefficients keep the controls exact.
    u, v, w = nodes.T
    controls = np.stack(
        (
            u + v / 8 + u * v / 16 + v * w / 8 + u * v * w / 32,
            v + w / 16 - u * w / 8 + u * v / 16 + u * v * w / 32,
            w + u / 8 + u * v / 8 - v * w / 16 + u * v * w / 16
            + (degree - 1) * u * (1.0 - u) / 4,
        ),
        axis=1,
    )
    geometry = CellGeometrySpec(
        {"authored": element},
        {"authored": np.arange(controls.shape[0])[None, :]},
        controls,
    )
    with coordinate_reference_budget() as coordinate_budget:
        elements, routes, _ = geometry.resolve(
            mesh, budget=coordinate_budget, exact_source_prepared=False
        )
        coordinates = prepared_coordinate_source_bank(geometry, budget=coordinate_budget)
        polynomials = coordinate_polynomials(
            elements[0],
            tuple(coordinates[int(index)] for index in np.asarray(routes[0][0])),
            budget=coordinate_budget,
        )
        assert polynomials is not None
        assert any(sum(index) > 1 for polynomial in polynomials for index in polynomial)
        before = coordinate_budget.work_units
        bernstein = _box_bernstein_quality_source(polynomials, budget=coordinate_budget)
        bernstein_work = coordinate_budget.work_units - before
        before = coordinate_budget.work_units
        monomial = _monomial_quality_source(
            polynomials, "hexahedron", "box", budget=coordinate_budget
        )
        monomial_work = coordinate_budget.work_units - before
    assert bernstein is not None
    assert bernstein.determinant_lower > 0
    assert bernstein.determinant_lower >= monomial.determinant_lower
    for value, bound in zip(
        bernstein.column_squared, monomial.column_squared, strict=True
    ):
        assert value <= bound
    assert bernstein.aspect_squared <= monomial.aspect_squared
    if degree == 1:
        # Without degree cancellation the nominal nets are the monomial nets.
        assert bernstein == monomial
    assert bernstein_work < monomial_work


def test_exact_hex_aspect_detects_curved_interior_despite_regular_corners() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality
    from phydrax.meshing._quality import evaluate_cell_quality

    points = np.asarray(reference_cell_topology("hexahedron").vertices)
    mesh = CellMesh(points, (CellBlock("authored", "hexahedron", np.arange(8)[None, :]),))
    element = coordinate_lagrange_element("hexahedron", 2)
    controls = np.asarray(element.reference_nodes).copy()
    controls[:, 2] *= 1.0 + 8.0 * controls[:, 0] * (1.0 - controls[:, 0])
    geometry = CellGeometrySpec(
        {"authored": element},
        {"authored": np.arange(controls.shape[0])[None, :]},
        controls,
    )
    schedule = NativeHexGridSchedule("frame_grid", maximum_aspect_ratio=4.0)
    assert np.all(
        np.asarray(evaluate_cell_quality(mesh).aspect_ratios)
        <= schedule.maximum_aspect_ratio
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        _, _, aspect = exact_mapped_volume_quality(
            mesh, geometry, scope=execution_scope, budget=coordinate_budget
        )
        jacobian = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (4.0, 0.0, 2.5)))
        actual_interior_condition = np.linalg.cond(jacobian)
        assert actual_interior_condition > schedule.maximum_aspect_ratio
        assert np.all(aspect >= actual_interior_condition)


def _dual_hex_fixture(
    tets: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[CellMesh, phx.discretization.CellGeometrySpec]:
    """Every tet's four corner hexes as P1-tet ∘ Q1-hex charts, one block per corner."""
    from fractions import Fraction

    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )

    reference = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    simplex = np.vstack((np.zeros(3), np.eye(3)))
    elements, routes, blocks, points = {}, {}, [], []
    coordinates = np.asarray([vertex for tet in tets for vertex in tet], dtype=np.float64)
    for corner in range(4):
        others = [index for index in range(4) if index != corner]
        chart = np.asarray(
            [
                np.mean(
                    simplex[[corner, *(o for o, on in zip(others, row) if on)]], axis=0
                )
                for row in reference
            ]
        )
        edges = chart[[1, 3, 4]] - chart[0]
        if np.linalg.det(edges) < 0:
            others[0], others[1] = others[1], others[0]
            chart = np.asarray(
                [
                    np.mean(
                        simplex[[corner, *(o for o, on in zip(others, row) if on)]],
                        axis=0,
                    )
                    for row in reference
                ]
            )
        exact = [[Fraction(value).limit_denominator(12) for value in row]
                 for row in cast(list[list[float]], chart.tolist())]
        with coordinate_reference_budget() as budget:
            element = PolynomialComposedCellGeometryElement(
                coordinate_lagrange_element("tetrahedron", 1),
                coordinate_lagrange_element("hexahedron", 1),
                np.asarray([[value.numerator for value in row] for row in exact]),
                np.asarray([[value.denominator for value in row] for row in exact]),
                budget=budget,
            )
        name = f"corner{corner}"
        elements[name] = element
        routes[name] = np.arange(4 * len(tets)).reshape(len(tets), 4)
        rows = []
        for tet in tets:
            vertices = np.asarray(tet, dtype=np.float64)
            frame = vertices[1:] - vertices[0]
            rows.append(np.arange(len(points), len(points) + 8))
            points.extend(vertices[0] + chart @ frame)
        blocks.append(
            CellBlock(
                name,
                "hexahedron",
                np.asarray(rows),
                global_ids=np.arange(len(tets)) + corner * len(tets),
            )
        )
    mesh = CellMesh(np.asarray(points), tuple(blocks))
    return mesh, CellGeometrySpec(elements, routes, coordinates)


_DUAL_HEX_TETS = (
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.5, 0.25, 0.0), (2.0, 0.5, 0.25), (0.75, 1.5, 0.125), (0.875, 0.625, 1.25)),
    # A flat sliver-like source tetrahedron.
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0625)),
)
# The same sliver with reversed orientation inverts its four hexahedra.
_INVERTED_TET = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.0625))


def test_multilinear_chart_validity_equals_polynomial_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry_validity as validity

    mesh, geometry = _dual_hex_fixture((*_DUAL_HEX_TETS, _INVERTED_TET))
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        fast = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        fast_work = budget.work_units - before
    monkeypatch.setattr(validity, "_multilinear_chart", lambda *args, **kwargs: None)
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        generic = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        generic_work = budget.work_units - before
    for name in ("status", "determinant_lower", "determinant_upper", "depth"):
        np.testing.assert_array_equal(
            np.asarray(getattr(fast, name)), np.asarray(getattr(generic, name))
        )
    assert fast.unresolved_reasons == generic.unresolved_reasons
    assert fast.invalid_count == 4 and fast.unresolved_count == 0
    assert fast_work < generic_work


def test_multilinear_chart_quality_encloses_actual_dual_hex_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry_validity as validity
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality

    # Each block's chart factors and regular-frame condition are prepared once
    # per request; repeated source cells amortize them as a mesh block does.
    mesh, geometry = _dual_hex_fixture(_DUAL_HEX_TETS * 4)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        before = budget.work_units
        scaled, mean, aspect = (
            np.asarray(value).copy()
            for value in exact_mapped_volume_quality(
                mesh, geometry, scope=scope, budget=budget
            )
        )
        fast_work = budget.work_units - before
    monkeypatch.setattr(
        validity, "multilinear_chart_factors", lambda *args, **kwargs: None
    )
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        before = budget.work_units
        generic_scaled, _, _ = exact_mapped_volume_quality(
            mesh, geometry, scope=scope, budget=budget
        )
        generic_work = budget.work_units - before
    assert fast_work < generic_work
    assert np.all(np.isfinite(generic_scaled))
    samples = np.asarray(list(product(np.linspace(0.0, 1.0, 9), repeat=3)))
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    factors = np.where(corners[None] > 0, samples[:, None, :], 1.0 - samples[:, None, :])
    signs = np.where(corners > 0, 1.0, -1.0)
    gradients = np.empty((samples.shape[0], corners.shape[0], 3))
    for derivative in range(3):
        others = [axis for axis in range(3) if axis != derivative]
        gradients[..., derivative] = signs[None, :, derivative] * np.prod(
            factors[..., others], axis=-1
        )
    points = np.asarray(mesh.coordinates)
    row = 0
    for block in mesh.blocks:
        for cell in np.asarray(block.vertices):
            jacobian = np.einsum("nd,snk->sdk", points[cell], gradients)
            determinant = np.linalg.det(jacobian)
            columns = np.linalg.norm(jacobian, axis=1)
            frobenius = np.sum(jacobian**2, axis=(1, 2))
            assert np.all(determinant > 0)
            tolerance = 1.0 + 1.0e-12
            assert scaled[row] <= tolerance * np.min(
                determinant / np.prod(columns, axis=1)
            )
            assert mean[row] <= tolerance * np.min(
                3.0 * determinant ** (2.0 / 3.0) / frobenius
            )
            assert aspect[row] * tolerance >= np.max(np.linalg.cond(jacobian))
            row += 1


def _transition_fixture(
    tets: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[CellMesh, phx.discretization.CellGeometrySpec]:
    """Hex-dominant transition pyramids and tetrahedra over P1 source tetrahedra.

    Every face corner of a source tetrahedron cones its quarter quadrilateral
    (corner, edge midpoint, face centroid, edge midpoint) and one fan triangle
    to the cell centroid, one block per chart, as ``extract_hex_dominant`` does.
    """
    from fractions import Fraction

    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
    )

    simplex = np.vstack((np.zeros(3), np.eye(3)))
    source = coordinate_lagrange_element("tetrahedron", 1)
    charts = []
    for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        for offset in range(3):
            a, b, c = (face[(offset + step) % 3] for step in range(3))
            charts.append(("pyramid", ((a,), (a, b), face, (a, c), (0, 1, 2, 3))))
            charts.append(("tetrahedron", ((a,), (a, b), face, (0, 1, 2, 3))))
    coordinates = np.asarray([vertex for tet in tets for vertex in tet], dtype=np.float64)
    elements, routes, blocks, points = {}, {}, [], []
    for index, (kind, supports) in enumerate(charts):
        nodes = np.asarray([simplex[list(support)].mean(axis=0) for support in supports])
        if np.linalg.det(nodes[[1, 2, -1]] - nodes[0]) < 0:
            nodes = nodes[[0, 3, 2, 1, 4]] if kind == "pyramid" else nodes[[0, 2, 1, 3]]
        exact = [[Fraction(value).limit_denominator(12) for value in row]
                 for row in cast(list[list[float]], nodes.tolist())]
        numerators = np.asarray([[value.numerator for value in row] for row in exact])
        denominators = np.asarray([[value.denominator for value in row] for row in exact])
        chart = coordinate_lagrange_element(kind, 1)
        if kind == "pyramid":
            element = RationalComposedCellGeometryElement(
                source, chart, numerators, denominators
            )
        else:
            with coordinate_reference_budget() as budget:
                element = PolynomialComposedCellGeometryElement(
                    source, chart, numerators, denominators, budget=budget
                )
        name = f"{kind}{index}"
        elements[name] = element
        routes[name] = np.arange(4 * len(tets)).reshape(len(tets), 4)
        rows = []
        for tet in tets:
            vertices = np.asarray(tet, dtype=np.float64)
            rows.append(np.arange(len(points), len(points) + nodes.shape[0]))
            points.extend(vertices[0] + nodes @ (vertices[1:] - vertices[0]))
        blocks.append(
            CellBlock(
                name,
                kind,
                np.asarray(rows),
                global_ids=np.arange(len(tets)) + index * len(tets),
            )
        )
    mesh = CellMesh(np.asarray(points), tuple(blocks))
    return mesh, CellGeometrySpec(elements, routes, coordinates)


def test_rational_pyramid_chart_validity_equals_polynomial_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry_validity as validity

    mesh, geometry = _transition_fixture((*_DUAL_HEX_TETS, _INVERTED_TET))
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        fast = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        fast_work = budget.work_units - before
    monkeypatch.setattr(validity, "_rational_pyramid_chart", lambda *args, **kwargs: None)
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        generic = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        generic_work = budget.work_units - before
    for name in ("status", "determinant_lower", "determinant_upper", "depth"):
        np.testing.assert_array_equal(
            np.asarray(getattr(fast, name)), np.asarray(getattr(generic, name))
        )
    assert fast.unresolved_reasons == generic.unresolved_reasons
    # Every pyramid and tetrahedron of the inverted source tetrahedron.
    assert fast.invalid_count == 24
    assert fast_work < generic_work


def test_transition_quality_routes_enclose_actual_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry_validity as validity
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality

    mesh, geometry = _transition_fixture(_DUAL_HEX_TETS)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        before = budget.work_units
        scaled, mean, aspect = (
            np.asarray(value).copy()
            for value in exact_mapped_volume_quality(
                mesh, geometry, scope=scope, budget=budget
            )
        )
        fast_work = budget.work_units - before
    monkeypatch.setattr(validity, "pyramid_chart_factors", lambda *args, **kwargs: None)
    monkeypatch.setattr(validity, "affine_chart_frame", lambda *args, **kwargs: None)
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        before = budget.work_units
        generic = tuple(
            np.asarray(value).copy()
            for value in exact_mapped_volume_quality(
                mesh, geometry, scope=scope, budget=budget
            )
        )
        generic_work = budget.work_units - before
    assert fast_work < generic_work
    # Collapsed-box samples of the reference pyramid, short of the apex.
    collapsed = np.asarray(
        list(product(np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 7), np.linspace(0.0, 0.95, 7)))
    )
    pyramid_points = np.stack(
        (
            (1.0 - collapsed[:, 2]) * collapsed[:, 0] + 0.5 * collapsed[:, 2],
            (1.0 - collapsed[:, 2]) * collapsed[:, 1] + 0.5 * collapsed[:, 2],
            collapsed[:, 2],
        ),
        axis=1,
    )
    elements, routes, _ = geometry.resolve(mesh)
    bank = np.asarray(geometry.coordinates)
    row = 0
    pyramid_rows = []
    for block, element, block_routes in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind == "tetrahedron":
            for values, reference in zip(
                (scaled, mean, aspect), generic, strict=True
            ):
                np.testing.assert_array_equal(
                    values[row : row + block.cell_count],
                    reference[row : row + block.cell_count],
                )
            row += block.cell_count
            continue
        assert isinstance(element, phx.discretization.RationalComposedCellGeometryElement)
        _, gradients = element.tabulate(pyramid_points)
        for route in np.asarray(block_routes):
            jacobian = np.einsum("ir,sik->srk", bank[route], np.asarray(gradients))
            determinant = np.linalg.det(jacobian)
            columns = np.linalg.norm(jacobian, axis=1)
            frobenius = np.sum(jacobian**2, axis=(1, 2))
            assert np.all(determinant > 0)
            tolerance = 1.0 + 1.0e-12
            assert scaled[row] <= tolerance * np.min(
                determinant / np.prod(columns, axis=1)
            )
            assert mean[row] <= tolerance * np.min(
                3.0 * determinant ** (2.0 / 3.0) / frobenius
            )
            assert aspect[row] * tolerance >= np.max(np.linalg.cond(jacobian))
            pyramid_rows.append(row)
            row += 1
    # The regular-frame chart condition tightens the generic one/infinity bound.
    assert np.max(aspect[pyramid_rows]) < np.max(generic[2][pyramid_rows])


@pytest.mark.parametrize("compact_patches", (False, True))
def test_material_face_subdivision_and_region_identity_are_exact(
    compact_patches: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = two_tetrahedra()
    cells = source.topology.entities(3)
    zones = tuple(
        MeshZone(
            name,
            MeshZoneRole.REGION,
            MeshingScope(
                source.mesh_id,
                source.numeric_version,
                MeshingEntityKind.MESH,
                3,
                cells.entity_set_id,
                np.asarray(cells.entity_ids)[[row]],
            ),
        )
        for row, name in enumerate(("left", "right"))
    )
    faces = source.topology.entities(2)
    assert isinstance(source.connectivity, TetrahedralConnectivity)
    face_vertices = np.asarray(source.connectivity.faces)
    interface = np.flatnonzero(
        np.all(np.sort(face_vertices, axis=1) == np.asarray((0, 1, 2)), axis=1)
    )
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "authoritative-material-solid",
        "material-source",
        faces.entity_set_id,
        np.asarray(faces.entity_ids)[interface],
        ("material-source:interface:0",),
        np.zeros((1,), dtype=np.float64),
        exact=True,
    )
    patch = MeshPatch(
        "material-interface",
        MeshingScope(
            source.mesh_id,
            source.numeric_version,
            MeshingEntityKind.MESH,
            2,
            faces.entity_set_id,
            np.asarray(faces.entity_ids)[interface],
        ),
        connected=True,
        adjacent_zone_ids=tuple(zone.zone_id for zone in zones),
        source_adjacent_region_ids=("right", "left"),
    )
    patches = MeshPatchBank((patch,)) if compact_patches else (patch,)
    from phydrax.meshing import _quad_generation as metadata

    original_completion = metadata._complete_vertex_associations
    completion_bounds = []

    def complete(*args: Any, **kwargs: Any) -> Any:
        retained = kwargs["retained_bytes"]
        completed = original_completion(*args, **kwargs)
        completion_bounds.append((retained, completed))
        return completed

    monkeypatch.setattr(metadata, "_complete_vertex_associations", complete)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        new_zones, new_patches, _, new_associations = remap_dual_metadata(
            result,
            zones=zones,
            patches=patches,
            associations=(association,),
            scope=execution_scope,
        )
        # A partial face association does not activate all-cell vertex
        # completion; its consumed incidence workspace must not stay retained.
        assert completion_bounds and all(before == after for before, after in completion_bounds)
        assert tuple(zone.name for zone in new_zones) == ("left", "right")
        assert tuple(zone.scope.entity_ids.size for zone in new_zones) == (4, 4)
        assert len(new_patches) == 1
        assert new_patches[0].name == patch.name
        assert new_patches[0].connected
        assert new_patches[0].source_adjacent_region_ids == ("right", "left")
        assert new_patches[0].adjacent_zone_ids == tuple(
            sorted(zone.zone_id for zone in new_zones)
        )
        assert new_patches[0].scope.entity_ids.size == 3
        assert new_patches[0].scope.source_id == result.mesh.mesh_id
        face_association = next(
            value
            for value in new_associations
            if value.target_entity_set_id
            == result.mesh.topology.entities(2).entity_set_id
        )
        assert (
            face_association.source_entities.entity_ids()
            == ("material-source:interface:0",) * 3
        )
        assert face_association.exact
        np.testing.assert_array_equal(
            face_association.residuals, np.zeros((3,), dtype=np.float64)
        )
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        assert np.all(
            np.asarray(result.mesh.connectivity.face_cell_counts)[
                (result.entity_parent_dimensions[2] == 2)
                & (result.entity_parent_rows[2] == interface[0])
            ]
            == 2
        )


def test_exact_association_does_not_survive_rounded_off_parent_face() -> None:
    source = two_tetrahedra()
    assert isinstance(source.connectivity, TetrahedralConnectivity)
    faces = source.topology.entities(2)
    interface = np.flatnonzero(
        np.all(
            np.sort(np.asarray(source.connectivity.faces), axis=1)
            == np.asarray((0, 1, 2), dtype=np.int64),
            axis=1,
        )
    )
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "material-solid",
        "material-source",
        faces.entity_set_id,
        np.asarray(faces.entity_ids)[interface],
        ("material-source:interface:0",),
        np.zeros((1,), dtype=np.float64),
        exact=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        extraction = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        center = np.flatnonzero(
            (extraction.entity_parent_dimensions[0] == 2)
            & (extraction.entity_parent_rows[0] == interface[0])
        )[0]
        coordinates = np.asarray(extraction.mesh.coordinates, dtype=np.float64).copy()
        coordinates[center, 2] = np.spacing(np.float64(1.0))
        perturbed = CellMesh(
            coordinates,
            extraction.mesh.blocks,
            vertex_global_ids=extraction.mesh.vertex_global_ids,
            numeric_version="off-parent-face",
        )
        candidate = eqx.tree_at(lambda value: value.mesh, extraction, perturbed)
        _, _, _, associations = remap_dual_metadata(
            candidate, associations=(association,), scope=execution_scope
        )
        face_association = next(
            value
            for value in associations
            if value.target_entity_set_id == perturbed.topology.entities(2).entity_set_id
        )
        assert not face_association.exact
        assert np.all(
            np.asarray(face_association.residuals) >= np.spacing(np.float64(1.0))
        )


@pytest.mark.parametrize("domain", ("cavity", "nonsweepable"))
def test_general_all_hex_positive_domains(domain: str) -> None:
    if domain == "cavity":
        occupied = set(product(range(3), range(3), range(3))) - {(1, 1, 1)}
    else:
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    source = occupied_cells(occupied)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert result.validity.certified_valid_count == 24 * len(occupied)
        assert np.sum(hex_volumes(result.mesh)) == pytest.approx(
            len(occupied), rel=VOLUME_TOLERANCE
        )
        points = np.asarray(result.mesh.coordinates)
        corners = points[entities(result.mesh, 3)]
        if domain == "cavity":
            assert not np.any(
                np.all(
                    (np.mean(corners, axis=1) > 1.0) & (np.mean(corners, axis=1) < 2.0),
                    axis=1,
                )
            )


def test_hex_dominant_pyramid_transition_closes_interface() -> None:
    source = two_tetrahedra()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        counts: dict[str, int] = {}
        for block in result.mesh.blocks:
            counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
        assert counts["hexahedron"] == 4
        assert counts["pyramid"] == 3
        assert counts["tetrahedron"] == 12
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        assert np.max(np.asarray(result.mesh.connectivity.face_cell_counts)) == 2
        assert result.geometry is not None and result.source_geometry is not None
        np.testing.assert_array_equal(
            np.asarray(result.geometry.coordinates).view(np.uint64),
            np.asarray(result.source_geometry.coordinates).view(np.uint64),
        )
        from phydrax.discretization._cell_geometry import (
            RationalComposedCellGeometryElement,
        )
        from phydrax.discretization._coordinate_enclosure import (
            coordinate_corner_images,
            prepared_coordinate_source_bank,
        )

        bank = prepared_coordinate_source_bank(result.geometry, budget=reference_budget)
        elements, routes, _ = result.geometry.resolve(result.mesh)
        for block, element, block_routes in zip(
            result.mesh.blocks, elements, routes, strict=True
        ):
            if block.cell_kind != "pyramid":
                continue
            assert isinstance(element, RationalComposedCellGeometryElement)
            points = np.asarray(
                ((0.5, 0.5, 0.5), reference_cell_topology("pyramid").vertices[4])
            )
            np.testing.assert_array_equal(
                element.reference_derivative_status(points), (True, False)
            )
            for route in np.asarray(block_routes):
                corners = coordinate_corner_images(
                    element,
                    tuple(bank[int(index)] for index in route),
                    budget=reference_budget,
                )
                assert corners is not None
                assert all(
                    np.isfinite(float(value)) for corner in corners for value in corner
                )


def test_mixed_source_maps_join_actual_family_fem_and_fv_consumers() -> None:
    import jax.numpy as jnp

    source = two_tetrahedra()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        source_corners = np.asarray(source.coordinates)[entities(source, 3)]
        volume = float(
            np.sum(np.linalg.det(source_corners[:, 1:] - source_corners[:, :1])) / 6.0
        )
        elements = {
            kind: phx.discretization.lagrange_element(kind, 1)
            for kind in {block.cell_kind for block in result.mesh.blocks}
        }
        field = phx.discretization.FiniteElementFieldSpec(
            "u", {block.name: elements[block.cell_kind] for block in result.mesh.blocks}
        )
        space = phx.discretization.FiniteElementPlan(
            result.mesh, field, coordinate_spec=result.geometry
        ).prepare()
        form = phx.equations.FiniteElementForm(
            "exact-family-manufactured-diffusion",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        quadratic_source, coefficient_id="quadratic-source"
                    ),
                ),
            ),
        )
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        phx.equations.compile_finite_element_problem(
            form, space, constraint=constraint, dirichlet_values=quadratic_value
        )
        values = jnp.ones(
            (space.dof_maps[0].global_dof_count,), dtype=result.mesh.coordinates.dtype
        )
        line = phx.integration.GaussLegendreRule(6)
        quadratures = {
            "hexahedron": phx.integration.reference_rule_data(
                phx.integration.ReferenceHexahedronRule(line)
            ),
            "tetrahedron": phx.integration.reference_rule_data(
                phx.integration.ReferenceTetrahedronRule(line)
            ),
            "pyramid": phx.integration.reference_rule_data(
                phx.integration.ReferencePyramidRule(line)
            ),
        }
        inventory = 0.0
        for index, block in enumerate(space.mesh.blocks):
            quadrature = quadratures[block.cell_kind]
            geometry = space.evaluate_block_geometry(
                "u",
                index,
                space.default_runtime.coordinates,
                quadrature.points,
                quadrature.weights,
            )
            dofs = space.dof_maps[0].cell_dofs[index]
            discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
            inventory += float(
                np.sum(np.asarray(geometry.physical_weights) * np.asarray(discrete))
            )
        assert inventory == pytest.approx(volume, rel=0.0, abs=VOLUME_TOLERANCE)
        system = phx.equations.EulerSystem(result.mesh.ambient_dimension)
        fv = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            result.mesh, component_names=system.component_names
        ).prepare(cell_geometry=result.geometry)
        volumes = np.asarray(fv.cell_volumes)
        assert np.all(np.isfinite(volumes))
        assert np.all(volumes > 0.0)
        assert abs(float(np.sum(volumes)) - volume) <= FV_VOLUME_RTOL * volume
        assert float(np.sum(volumes)) == pytest.approx(
            volume, rel=0.0, abs=VOLUME_TOLERANCE
        )
        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            fv.boundary_patch_names,
            {
                name: phx.discretization.SlipWallBoundary()
                for name in fv.boundary_patch_names
            },
        )
        method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        )
        dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
            system, fv, method, boundaries
        )
        primitive = jnp.zeros(
            (fv.cell_count, result.mesh.ambient_dimension + 2),
            dtype=result.mesh.coordinates.dtype,
        )
        primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
        state = system.primitive_to_conserved(primitive)
        rate = dynamics(jnp.asarray(0.0, dtype=state.dtype), state)
        defect = float(jnp.max(jnp.abs(rate)))
        assert np.isfinite(defect)
        assert defect <= STATIONARY_EULER_ATOL


@pytest.mark.parametrize("row_order", tuple(permutations(range(3))))
def test_hex_dominant_partial_face_fans_close_across_source_cell_order(
    row_order: tuple[int, ...],
) -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, -1.0, 0.0),
        )
    )
    tetrahedra = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4), (0, 4, 1, 5)), dtype=np.int64)
    source = CellMesh.from_tetrahedra(coordinates, tetrahedra[list(row_order)])
    core = row_order.index(0)
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[core]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {
            "hexahedron",
            "pyramid",
            "tetrahedron",
        }
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        if not isinstance(source.connectivity, TetrahedralConnectivity):
            raise TypeError(
                "fixture.source.connectivity must be tetrahedral; got incompatible connectivity"
            )
        if not isinstance(result.mesh.connectivity, PolyhedralConnectivity):
            raise TypeError(
                "fixture.result.connectivity must be polyhedral; got incompatible connectivity"
            )
        source_incidence = np.asarray(source.connectivity.face_cell_counts)
        dimensions, rows = (
            result.entity_parent_dimensions[2],
            result.entity_parent_rows[2],
        )
        expected = np.full(dimensions.shape, 2, dtype=np.int64)
        inherited = dimensions == 2
        expected[inherited] = source_incidence[rows[inherited]]
        np.testing.assert_array_equal(result.mesh.connectivity.face_cell_counts, expected)
        # The complement's shared face has only one marked edge. Its full fan
        # must agree from both sides, not merely the selected core's quad interface.
        source_faces = entities(source, 2)
        partial_face = np.flatnonzero(
            np.all(np.sort(source_faces, axis=1) == (0, 1, 4), axis=1)
        )[0]
        partial_children = inherited & (rows == partial_face)
        assert np.count_nonzero(partial_children) > 1
        assert np.all(
            np.asarray(result.mesh.connectivity.face_cell_counts)[partial_children] == 2
        )
        # Independently integrate each cell family, then group by retained parent
        # ancestry. This catches a closed but duplicated/missing transition cone.
        family_volumes = [hex_volumes(result.mesh)]
        target_points = np.asarray(result.mesh.coordinates)
        for block in result.mesh.blocks:
            if block.cell_kind == "hexahedron":
                continue
            vertices = target_points[np.asarray(block.vertices)]
            simplices = (
                ((0, 1, 2, 4), (0, 2, 3, 4))
                if block.cell_kind == "pyramid"
                else ((0, 1, 2, 3),)
            )
            volumes = np.zeros(block.cell_count)
            for simplex in simplices:
                corners = vertices[:, simplex, :]
                edges = corners[:, 1:] - corners[:, :1]
                determinants = np.linalg.det(edges)
                assert np.all(determinants > 0.0)
                volumes += determinants / 6.0
            family_volumes.append(volumes)
        source_corners = coordinates[tetrahedra[list(row_order)]]
        source_volumes = (
            np.linalg.det(source_corners[:, 1:] - source_corners[:, :1]) / 6.0
        )
        transferred_volumes = np.bincount(
            result.parent_cells, weights=np.concatenate(family_volumes), minlength=3
        )
        np.testing.assert_allclose(
            transferred_volumes,
            source_volumes,
            rtol=0.0,
            # The unit source-volume determinant comparison budgets 32 edge/
            # determinant roundoffs per side plus accumulation: 128 * eps.
            atol=128 * np.finfo(np.float64).eps,
        )


def test_hex_dominant_never_weakens_required_pure_family() -> None:
    source = two_tetrahedra()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_hex_dominant(
                source,
                MeshingLimits(),
                CellFamilyPolicy(required=("hexahedron",)),
                hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED


@pytest.mark.parametrize("dimension,limit", ((1, "maximum_edges"), (2, "maximum_faces")))
def test_hex_dominant_actual_topology_capacity_refuses_without_mutation(
    dimension: int,
    limit: str,
) -> None:
    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    core = np.asarray(source.topology.entities(3).entity_ids)[[0]]
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        accepted = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=core,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        cap = accepted.mesh.topology.entities(dimension).count - 1
        with pytest.raises(MeshingFailure) as failed:
            extract_hex_dominant(
                source,
                MeshingLimits(**{limit: cap}),
                policy,
                hex_core_cells=core,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(source.coordinates, original)


def test_hex_budget_and_immutable_boundary_refuse_without_source_change() -> None:
    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    with native_meshing_request(MeshingLimits(maximum_cells=7)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                source,
                MeshingLimits(maximum_cells=7),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                source,
                MeshingLimits(),
                boundary_subdivision=False,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
        np.testing.assert_array_equal(source.coordinates, original)


def test_invalid_native_topology_preserves_the_constructor_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._meshcore as meshcore

    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    extract = meshcore.tetrahedron_dual_hexes

    def faulty(
        nodes: np.ndarray, vertex_count: int, /, *, maximum_cells: int
    ) -> np.ndarray:
        cells = extract(nodes, vertex_count, maximum_cells=maximum_cells)
        cells[0] = cells[0, (0, 3, 2, 1, 4, 7, 6, 5)]
        return cells

    monkeypatch.setattr(meshcore, "tetrahedron_dual_hexes", faulty)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(ValueError, match="orientation|face"):
            extract_volume_hexes(
                source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
            )
    np.testing.assert_array_equal(source.coordinates, original)


def test_misoriented_native_template_fails_independent_map_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._meshcore as meshcore

    extract = meshcore.tetrahedron_dual_hexes

    def faulty(
        nodes: np.ndarray, vertex_count: int, /, *, maximum_cells: int
    ) -> np.ndarray:
        cells = extract(nodes, vertex_count, maximum_cells=maximum_cells)
        cells[:] = cells[:, (0, 3, 2, 1, 4, 7, 6, 5)]
        return cells

    monkeypatch.setattr(meshcore, "tetrahedron_dual_hexes", faulty)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                two_tetrahedra(),
                MeshingLimits(),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED


def test_source_lipschitz_consumes_nine_jets_and_retires_root_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.meshing import _hex_generation as generation

    polynomials: tuple[enclosure.Polynomial, ...] = (
        {(1, 0, 0): Fraction(1), (2, 1, 0): Fraction(1, 8)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    inverse = [
        [Fraction(1), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]
    original_derivative = enclosure.derivative
    derivatives: list[tuple[enclosure.Polynomial, int]] = []

    def capture_derivative(
        polynomial: enclosure.Polynomial,
        axis: int,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> enclosure.Polynomial:
        derivatives.append((polynomial, axis))
        return original_derivative(polynomial, axis, budget=budget)

    monkeypatch.setattr(enclosure, "derivative", capture_derivative)
    with coordinate_reference_budget() as budget:
        # A source preparation already owned by the request must survive each
        # root's local Bernstein memo, including when its exact content is reused.
        enclosure.bernstein_coefficients(polynomials[0], "box", 3, budget=budget)
        before_temporary = budget.temporary_bytes_upper
        before_retained = budget.retained_basis_bytes
        before_bernstein = dict(budget.bernstein_coefficient_cache)
        expected = generation._sqrt_upper(Fraction(399, 64))
        for _ in range(2):
            before_work = budget.work_units
            actual = generation._source_root_lipschitz_bound(
                polynomials, inverse, budget=budget
            )
            assert actual == expected
            assert budget.work_units > before_work
            assert budget.temporary_bytes_upper == before_temporary
            assert budget.retained_basis_bytes == before_retained
            assert dict(budget.bernstein_coefficient_cache) == before_bernstein
    assert derivatives == [
        (polynomial, axis)
        for _ in range(2)
        for polynomial in polynomials
        for axis in range(3)
    ]


@pytest.mark.parametrize(
    "matrix",
    (
        ((2, 0, 0), (0, -3, 0), (0, 0, 1)),
        ((0, 2, -3), (1, 0, 0), (0, 0, 0)),
        ((1, 2, 0), (-1, 1, 0), (0, 0, -2)),
        ((1, 2, -1), (-1, 1, 3), (2, -3, 1)),
    ),
)
@pytest.mark.parametrize("local_bounds", (False, True))
def test_source_lipschitz_sparse_columns_match_independent_transformation(
    matrix: tuple[tuple[int, ...], ...], local_bounds: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing import _hex_generation as generation

    polynomials: tuple[enclosure.Polynomial, ...] = (
        {(0, 0, 0): Fraction(7, 13), (1, 0, 0): Fraction(1), (2, 1, 0): Fraction(-1, 8)},
        {(0, 1, 0): Fraction(2, 3), (1, 1, 2): Fraction(1, 7)},
        {(0, 0, 1): Fraction(5, 3), (2, 0, 2): Fraction(-1, 11)},
    )
    inverse = [[Fraction(value, 3) for value in row] for row in matrix]
    original = tuple(dict(polynomial) for polynomial in polynomials)
    # Differentiate and transform coefficients directly; do not use the
    # production derivative, sparse-column path, or polynomial combinations.
    with coordinate_reference_budget() as budget:
        expected_rows = []
        local_squared = [Fraction(0)] * 3
        for polynomial in polynomials:
            jets: list[enclosure.Polynomial] = [{}, {}, {}]
            for powers, coefficient in polynomial.items():
                for axis, degree in enumerate(powers):
                    if degree:
                        index = tuple(power - int(column == axis) for column, power in enumerate(powers))
                        jets[axis][index] = coefficient * degree
            for axis, jet in enumerate(jets):
                magnitude = generation._polynomial_magnitude(jet, budget=budget)
                local_squared[axis] += magnitude * magnitude
            row = []
            for column in range(3):
                transformed: enclosure.Polynomial = {}
                for axis, jet in enumerate(jets):
                    for index, coefficient in jet.items():
                        transformed[index] = transformed.get(index, Fraction(0)) + coefficient * inverse[axis][column]
                transformed = {index: coefficient for index, coefficient in transformed.items() if coefficient}
                row.append(generation._polynomial_magnitude(transformed, budget=budget))
            expected_rows.append(tuple(row))
        expected = generation._sqrt_upper(generation._matrix_norm_product(tuple(expected_rows)))
        expected_local = np.asarray([generation._sqrt_upper(value) for value in local_squared])
        expected_reference = np.asarray([
            max(0.5, generation._sqrt_upper(sum((row[axis] ** 2 for row in expected_rows), Fraction(0))))
            for axis in range(3)
        ])

    magnitudes = []
    original_magnitude = generation._polynomial_magnitude

    def capture_magnitude(polynomial: enclosure.Polynomial, /, **kwargs: Any) -> Fraction:
        magnitudes.append(polynomial)
        return original_magnitude(polynomial, **kwargs)

    monkeypatch.setattr(generation, "_polynomial_magnitude", capture_magnitude)

    def evaluate(limit: int) -> tuple[float, int]:
        magnitudes.clear()
        with coordinate_reference_budget(MeshingLimits(maximum_work_units=limit)) as budget:
            local = np.full(3, np.nan) if local_bounds else None
            reference = np.full(3, 0.5)
            result = generation._source_root_lipschitz_bound(
                polynomials, inverse, budget=budget,
                root_axis_bounds=local, reference_axis_bounds=reference,
            )
            assert result == expected
            np.testing.assert_array_equal(reference.view(np.uint64), expected_reference.view(np.uint64))
            if local is not None:
                np.testing.assert_array_equal(local.view(np.uint64), expected_local.view(np.uint64))
            assert budget.temporary_bytes_upper == budget.retained_basis_bytes == 0
            columns = [tuple(axis for axis in range(3) if matrix[axis][column]) for column in range(3)]
            singleton_axes = {column[0] for column in columns if len(column) == 1}
            expected_calls = 3 * (
                (3 if local_bounds else len(singleton_axes))
                + sum(len(column) > 1 for column in columns)
            )
            assert len(magnitudes) == expected_calls
            return result, budget.work_units

    _, required = evaluate(2_000_000)
    assert evaluate(required) == (expected, required)
    with pytest.raises(CoordinateEnclosureResourceError):
        evaluate(required - 1)
    assert polynomials == original


def test_balanced_grid_capacity_counts_active_templates_not_dyadic_cube() -> None:
    from phydrax.meshing._hex_generation import (
        _extract_balanced_cells,
        _prepare_grid_layout,
    )

    # Six active intervals per axis select 1 + 8 + 27 refined nodes.
    # Their genuine tree has 1 + 7 * 36 leaves, not 512 uniform leaves.
    limits = MeshingLimits(maximum_cells=216, maximum_connectivity_entries=1728)
    knots = tuple(np.arange(4, dtype=np.float64) for _ in range(3))
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            knots, NativeHexGridSchedule("balanced_grid"), limits, 0.5,
            scope=scope, budget=budget
        )
        assert layout.required == 216
        assert layout.refined_shapes == ((1, 1, 1), (2, 2, 2), (3, 3, 3))
        np.testing.assert_array_equal(layout.sizes, (6, 6, 6))
        extracted = _extract_balanced_cells(layout, limits, scope=scope)
        assert extracted.cells.shape == (216, 8)
        assert extracted.integer_cells.shape == (216, 3)
        assert extracted.octree_levels.size == 253
        assert np.count_nonzero(extracted.octree_levels == 3) == 216
        assert np.count_nonzero(extracted.octree_levels == 2) == 37
        assert np.all(extracted.integer_cells >= 0)
        assert np.all(extracted.integer_cells < 6)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_aimed_grid_preserves_admitted_target_before_considering_maximum(
    route: NativeHexGridRoute,
) -> None:
    from fractions import Fraction

    from phydrax.meshing._hex_generation import _prepare_grid_layout

    knots = tuple(np.asarray((0.0, 1.0)) for _ in range(3))
    originals = tuple(axis.copy() for axis in knots)
    limits = MeshingLimits(maximum_cells=64)
    identity = limits.limits_id
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            knots, NativeHexGridSchedule(route), limits, 1.0 / 3.0,
            maximum_size=1.0, scope=scope, budget=budget,
        )
        # Preserve the original floating target count decision; exact knots
        # still describe thirds, not rounded copies of a new maximum-size grid.
        np.testing.assert_array_equal(layout.sizes, (3, 3, 3))
        assert layout.required == 27
        assert all(axis == (Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(1)) for axis in layout.exact_axis_knots)
    assert limits.limits_id == identity
    for original, axis in zip(originals, knots, strict=True):
        np.testing.assert_array_equal(axis, original)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_aimed_grid_coarsens_to_finest_admitted_counts_inside_original_band(
    route: NativeHexGridRoute,
) -> None:
    from fractions import Fraction

    from phydrax.meshing._hex_generation import _prepare_grid_layout

    limits = MeshingLimits(maximum_cells=64, maximum_connectivity_entries=512)
    knots = tuple(np.asarray((0.0, 1.0)) for _ in range(3))
    with native_meshing_request(limits) as (scope, budget, _):
        before = scope.remaining().remaining_work_units
        layout = _prepare_grid_layout(
            knots, NativeHexGridSchedule(route), limits, 0.2,
            maximum_size=0.3, scope=scope, budget=budget,
        )
        assert scope.remaining().remaining_work_units < before
        np.testing.assert_array_equal(layout.sizes, (4, 4, 4))
        assert layout.required == 64
        assert all(axis == tuple(Fraction(index, 4) for index in range(5)) for axis in layout.exact_axis_knots)
        # The next finer count bank is 125 cells; choosing the coarsest allowed
        # bank, changing limits, or spending a replacement allowance is wrong.
        assert 5**3 > limits.maximum_cells
        assert limits.maximum_cells == 64
        assert limits.maximum_connectivity_entries == 512


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_aimed_grid_finishes_directional_capacity_instead_of_stopping_at_cube(
    route: NativeHexGridRoute,
) -> None:
    from phydrax.meshing._hex_generation import _prepare_grid_layout

    limits = MeshingLimits(maximum_cells=100)
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            tuple(np.asarray((0.0, 1.0)) for _ in range(3)),
            NativeHexGridSchedule(route), limits, 0.2,
            maximum_size=0.3, scope=scope, budget=budget,
        )
        np.testing.assert_array_equal(layout.sizes, (5, 5, 4))
        assert layout.required == 100
        # Any componentwise refinement toward the desired (5,5,5) now
        # exceeds the unchanged capacity; the admitted 4-cube was not maximal.
        assert 5**3 > limits.maximum_cells


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_aimed_grid_refuses_when_no_count_bank_in_authored_band_fits(
    route: NativeHexGridRoute,
) -> None:
    from phydrax.meshing._hex_generation import _prepare_grid_layout

    limits = MeshingLimits(maximum_cells=63)
    knots = tuple(np.asarray((0.0, 1.0)) for _ in range(3))
    originals = tuple(axis.copy() for axis in knots)
    with native_meshing_request(limits) as (scope, budget, _):
        with pytest.raises(MeshingFailure, match="authored size band") as failed:
            _prepare_grid_layout(
                knots, NativeHexGridSchedule(route), limits, 0.2,
                maximum_size=0.3, scope=scope, budget=budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    for original, axis in zip(originals, knots, strict=True):
        np.testing.assert_array_equal(axis, original)


def test_balanced_grid_streams_real_halo_templates_under_original_cell_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._meshcore as native
    from phydrax.meshing._hex_generation import (
        _extract_balanced_cells,
        _prepare_grid_layout,
    )

    calls: list[tuple[int, int]] = []
    original = native.subdivide_hex_grid

    def capture(nodes: np.ndarray, vertex_count: int, /, *, maximum_cells: int) -> np.ndarray:
        calls.append((len(nodes), maximum_cells))
        return original(nodes, vertex_count, maximum_cells=maximum_cells)

    monkeypatch.setattr(native, "subdivide_hex_grid", capture)
    limits = MeshingLimits(maximum_cells=27, maximum_connectivity_entries=216)
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            tuple(np.asarray((0.0, 1.0)) for _ in range(3)),
            NativeHexGridSchedule("balanced_grid"), limits, 1.0 / 3.0,
            maximum_size=1.0, scope=scope, budget=budget,
        )
        assert layout.required == 27
        extracted = _extract_balanced_cells(layout, limits, scope=scope)
        assert extracted.cells.shape == (27, 8)
        np.testing.assert_array_equal(
            np.unique(extracted.integer_cells, axis=0),
            np.asarray(list(product(range(3), repeat=3))),
        )
        assert sum(8 * count for count, _ in calls) == 64
        assert len(calls) == 3
        assert all(8 * count <= maximum == 27 for count, maximum in calls)
        assert layout.extraction_work > 64
        assert scope.remaining().remaining_work_units < limits.maximum_work_units - layout.extraction_work


@pytest.mark.parametrize("resource", ("scratch", "work"))
def test_aimed_grid_coarsening_charges_original_storage_and_total_work(
    resource: str,
) -> None:
    from phydrax.meshing._hex_generation import (
        _extract_balanced_cells,
        _prepare_grid_layout,
    )

    limits = (
        MeshingLimits(maximum_cells=125, maximum_scratch_bytes=600_000)
        if resource == "scratch"
        else MeshingLimits(maximum_cells=125, maximum_work_units=8_000)
    )
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            tuple(np.asarray((0.0, 1.0)) for _ in range(3)),
            NativeHexGridSchedule("balanced_grid"), limits, 0.2,
            maximum_size=0.3, scope=scope, budget=budget,
        )
        selected = (5, 4, 4) if resource == "scratch" else (4, 4, 4)
        np.testing.assert_array_equal(layout.sizes, selected)
        assert layout.extraction_bytes + layout.retained_bytes <= limits.maximum_scratch_bytes
        before = scope.remaining().remaining_work_units
        extracted = _extract_balanced_cells(layout, limits, scope=scope)
        assert extracted.cells.shape == (int(np.prod(selected)), 8)
        assert before - scope.remaining().remaining_work_units == layout.extraction_work
        if resource == "scratch":
            from phydrax.meshing._hex_generation import _grid_vertex_coordinates

            # Shared axis atoms now fit the original cap. Exercise actual
            # placement, not merely the improved metadata admission.
            physical, exact = _grid_vertex_coordinates(
                extracted.integer_vertices, layout.exact_axis_knots, 1, np.eye(3),
                scope=scope, budget=budget,
            )
            assert len(exact) == len(physical)
            np.testing.assert_array_equal(
                physical, np.asarray(extracted.integer_vertices) / np.asarray(selected)
            )
            assert limits.maximum_scratch_bytes == 600_000


def test_source_grid_uses_tighter_directional_bound_even_when_global_grid_fits() -> None:
    from phydrax.meshing._hex_frame_map import (
        extract_source_block_integer_grid,
        prepare_source_hex_frame,
    )
    from phydrax.meshing._hex_generation import mapped_source_lipschitz_bound
    from phydrax.meshing.qualification import curved_occupied_source

    source = curved_occupied_source(
        {(0, 0, 0)}, curvature=0.0, chart_scale=(2.0, 0.5, 1.0)
    )
    domain = source.domain
    limits = MeshingLimits(maximum_cells=1000)
    with native_meshing_request(limits) as (scope, budget, _):
        local, reference = np.empty((1, 3)), np.empty((1, 3))
        bound = mapped_source_lipschitz_bound(
            domain.reference_mesh, domain.source_geometry,
            root_axis_bounds=local, root_reference_axis_bounds=reference,
            scope=scope, budget=budget,
        )
        prepared = prepare_source_hex_frame(
            domain.reference_mesh, domain.source_geometry, scope=scope, budget=budget
        )
        grid = extract_source_block_integer_grid(
            prepared, None, domain.cell_regions, limits, 6,
            axis_bounds=local, desired_axis_bounds=reference,
            physical_lipschitz_bound=bound, target_size=0.25, maximum_size=0.5,
            scope=scope, budget=budget,
        )
        np.testing.assert_array_equal(grid.root_intervals, ((8, 2, 4),))
        assert sum(block.cell_count for block in grid.mesh.blocks) == 64
        assert np.all(local / grid.root_intervals <= 0.25)


@pytest.mark.parametrize("reversed_reference", (False, True))
def test_aimed_source_counts_use_anisotropic_nonunit_reversed_compatible_roots(
    reversed_reference: bool,
) -> None:
    from phydrax.meshing._hex_frame_map import (
        extract_source_block_integer_grid,
        prepare_source_hex_frame,
    )
    from phydrax.meshing._hex_generation import mapped_source_lipschitz_bound
    from phydrax.meshing.qualification import curved_occupied_source

    source = curved_occupied_source(
        {(0, 0, 0), (1, 0, 0)}, curvature=0.0, chart_scale=(2.0, 0.5, 1.0),
    )
    domain = source.domain
    signs = (-1.0, 1.0, -1.0) if reversed_reference else (1.0, 1.0, 1.0)
    transform = np.diag(np.asarray(signs) * (1.0 / 16.0, 16.0, 1.0))
    reference = CellMesh(
        np.asarray(domain.reference_mesh.coordinates) @ transform.T,
        domain.reference_mesh.blocks,
        vertex_global_ids=domain.reference_mesh.vertex_global_ids,
    )
    original_points = np.asarray(reference.coordinates).copy()
    original_values = np.asarray(domain.source_geometry.coordinates).copy()
    limits = MeshingLimits(maximum_cells=64)
    with native_meshing_request(limits) as (scope, budget, workspace):
        workspace.set_bound(4096)
        local = np.empty((2, 3))
        reference_lengths = np.empty((2, 3))
        bound = mapped_source_lipschitz_bound(
            reference, domain.source_geometry, root_axis_bounds=local,
            root_reference_axis_bounds=reference_lengths, scope=scope, budget=budget,
        )
        np.testing.assert_array_equal(local, ((2.0, 0.5, 1.0), (2.0, 0.5, 1.0)))
        # A second valid conservative upper bound on the shared root axis
        # exercises class maxima, rather than equal-root special casing.
        local[1, 1] = 0.75
        before_local, before_reference = local.copy(), reference_lengths.copy()
        prepared = prepare_source_hex_frame(reference, domain.source_geometry, scope=scope, budget=budget)
        grid = extract_source_block_integer_grid(
            prepared, None, domain.cell_regions, limits, 6,
            axis_bounds=local, desired_axis_bounds=reference_lengths,
            physical_lipschitz_bound=bound, target_size=0.25, maximum_size=0.5,
            scope=scope, budget=budget,
        )
        np.testing.assert_array_equal(grid.root_intervals, ((5, 2, 3), (5, 2, 3)))
        assert sum(block.cell_count for block in grid.mesh.blocks) == 60
        np.testing.assert_array_equal(np.bincount(grid.source_parent_rows), (30, 30))
        assert set(np.unique(grid.chart_denominators)) == {2, 3, 5}
        np.testing.assert_array_equal(local, before_local)
        np.testing.assert_array_equal(reference_lengths, before_reference)
    np.testing.assert_array_equal(reference.coordinates, original_points)
    np.testing.assert_array_equal(domain.source_geometry.coordinates, original_values)
    assert limits.maximum_cells == 64


@pytest.mark.parametrize("route", ("mapped_balanced_grid_hex", "mapped_frame_grid_hex"))
@pytest.mark.parametrize("capacity", (63, 64))
def test_aimed_provider_preserves_original_controls_and_types_impossible_band(
    route: phx.meshing.NativeMeshingRoute, capacity: int,
) -> None:
    from phydrax.meshing.qualification import curved_occupied_source

    source = curved_occupied_source({(0, 0, 0)}, curvature=0.0)
    root = source.domain.reference_mesh
    scope = MeshingScope(
        source.source_id, source.source_revision, MeshingEntityKind.GEOMETRY,
        2, source.domain.entity_set_id(2), root.topology.entities(2).entity_ids,
    )
    control = phx.meshing.UniformSizeControl(
        scope, 0.2, minimum_size=0.18, maximum_size=0.27,
        maximum_growth_rate=1.0,
    )
    policy = phx.meshing.SizeCompliancePolicy(relative_tolerance=0.5)
    limits = MeshingLimits(maximum_cells=capacity)
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, CellFamilyPolicy(required=("hexahedron",)), geometry_order=2,
        ),
        scope, phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(control,), size_compliance=policy, limits=limits,
    )
    before = (request.specification_id, control.control_id, policy.policy_id, limits.limits_id)
    original_points = np.asarray(root.coordinates).copy()
    original_values = np.asarray(source.domain.source_geometry.coordinates).copy()
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(route),
    ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
    if capacity == 63:
        with pytest.raises(MeshingFailure) as failed:
            plan.execute()
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    else:
        result = plan.execute()
        assert result.compliance.passed
        assert sum(block.cell_count for block in result.mesh.blocks) == 64
        achieved = dict(result.compliance.achieved)
        requested = dict(result.compliance.requested)
        key = f"size:{control.control_id}"
        assert requested[f"{key}:target_size"] == 0.2
        assert requested[f"{key}:minimum_size"] == 0.18
        assert requested[f"{key}:maximum_size"] == 0.27
        assert achieved[f"{key}:minimum_edge"] == 0.25
        assert achieved[f"{key}:maximum_edge"] == 0.25
    assert (request.specification_id, control.control_id, policy.policy_id, limits.limits_id) == before
    np.testing.assert_array_equal(root.coordinates, original_points)
    np.testing.assert_array_equal(source.domain.source_geometry.coordinates, original_values)


def test_source_count_planner_preserves_original_admitted_counts_before_directional_fallback() -> None:
    from fractions import Fraction
    from math import prod

    from phydrax.meshing._hex_generation import source_block_interval_counts

    limits = MeshingLimits(maximum_cells=64)
    with native_meshing_request(limits) as (scope, _, _):
        counts = source_block_interval_counts(
            (Fraction(1),) * 3, 0.5, 1.0,
            lambda candidate: prod(candidate) <= limits.maximum_cells,
            desired_lengths=(Fraction(2),) * 3, scope=scope,
        )
        assert counts == (4, 4, 4)


def test_source_lipschitz_releases_root_workspace_after_resource_refusal() -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing._hex_generation import _source_root_lipschitz_bound

    with coordinate_reference_budget(MeshingLimits(maximum_work_units=1)) as budget:
        before_temporary = budget.temporary_bytes_upper
        before_retained = budget.retained_basis_bytes
        with pytest.raises(CoordinateEnclosureResourceError):
            _source_root_lipschitz_bound(
                ({(1, 0, 0): Fraction(1), (2, 1, 0): Fraction(1, 8)},),
                [[Fraction(int(row == column)) for column in range(3)] for row in range(3)],
                budget=budget,
            )
        assert budget.temporary_bytes_upper == before_temporary
        assert budget.retained_basis_bytes == before_retained
        assert not budget.bernstein_coefficient_cache


def test_balanced_grid_preserves_exact_subdivision_before_reference_rounding() -> None:
    from fractions import Fraction

    from phydrax.meshing._family_execution import coordinate_phase

    complex_, domain = mapped_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        before_work = scope.remaining().remaining_work_units
        grid = generate_integer_grid_hexes(
            complex_, domain, NativeHexGridSchedule("balanced_grid"), limits,
            1.0 / 3.0, scope=scope, budget=budget,
        )
        assert grid.work_units == before_work - scope.remaining().remaining_work_units
        # Knot axes use the issued signed-permutation frame, not physical axes.
        assert any(abs(value) == Fraction(4, 3) for axis in grid.exact_axis_knots for value in axis)
        assert any(point[0] == Fraction(4, 3) for point in grid.exact_vertex_coordinates)
        basis = np.asarray(grid.frame.basis, dtype=np.int64)
        np.testing.assert_array_equal(np.abs(basis).sum(axis=0), (1, 1, 1))
        np.testing.assert_array_equal(np.abs(basis).sum(axis=1), (1, 1, 1))
        assert np.any(basis < 0)
        for address, point, rounded in zip(
            grid.logical_vertices, grid.exact_vertex_coordinates,
            np.asarray(grid.mesh.coordinates), strict=True,
        ):
            logical = tuple(
                grid.exact_axis_knots[axis][int(address[axis])]
                for axis in range(3)
            )
            expected = tuple(
                sum(
                    (logical[axis] * int(basis[axis, component]) for axis in range(3)),
                    Fraction(0),
                )
                for component in range(3)
            )
            assert point == expected
            np.testing.assert_array_equal(rounded, tuple(map(float, expected)))
        mapped = realize_mapped_grid_hexes(
            grid, domain.reference_mesh, domain.source_geometry, domain.cell_regions,
            limits, scope=scope, budget=budget,
        )
        assert sum(block.cell_count for block in mapped.mesh.blocks) == 54
        assert len(mapped.mesh.blocks) == 27
        with coordinate_phase(scope=scope, budget=budget):
            elements, _, _ = mapped.geometry.resolve(mapped.mesh, budget=budget)
        for element in elements:
            assert isinstance(element, phx.discretization.PolynomialComposedCellGeometryElement)
            assert all(
                denominator in (1, 3)
                for row in element.chart_coefficients for _, denominator in row
            )
        moved = np.asarray(grid.mesh.coordinates).copy()
        moved[0, 0] = np.nextafter(moved[0, 0], np.inf)
        foreign = eqx.tree_at(
            lambda value: value.mesh, grid,
            grid.mesh.with_coordinates(moved, numeric_version="mutated-grid"),
        )
        with pytest.raises(MeshingFailure, match="exact_placement must round"):
            realize_mapped_grid_hexes(
                foreign, domain.reference_mesh, domain.source_geometry, domain.cell_regions,
                limits, scope=scope, budget=budget,
            )


@pytest.mark.parametrize(
    "limits",
    (
        MeshingLimits(maximum_cells=215),
        MeshingLimits(maximum_connectivity_entries=1727),
    ),
)
def test_balanced_grid_actual_template_capacity_still_refuses(
    limits: MeshingLimits,
) -> None:
    from phydrax.meshing._hex_generation import _prepare_grid_layout

    knots = tuple(np.arange(4, dtype=np.float64) for _ in range(3))
    with native_meshing_request(limits) as (scope, budget, _):
        with pytest.raises(MeshingFailure, match="check_family_limits.capacity"):
            _prepare_grid_layout(
                knots, NativeHexGridSchedule("balanced_grid"), limits, 0.5,
                scope=scope, budget=budget
            )


@pytest.mark.parametrize("sizes", ((62, 4, 2), (2, 62, 4), (4, 2, 62)))
def test_balanced_grid_sparse_box_preserves_cells_and_coordinates_under_low_memory(
    sizes: tuple[int, int, int],
) -> None:
    from fractions import Fraction
    from math import prod

    from phydrax.discretization.spatial._level_octree import refined_octree_leaves
    from phydrax.discretization.spatial._morton import (
        morton_decode_integer_host,
        morton_encode_integer_host,
        MortonAddressPlan,
    )
    from phydrax.meshing._hex_generation import (
        _extract_balanced_cells,
        _grid_vertex_coordinates,
        _prepare_grid_layout,
    )

    count = prod(sizes)
    limits = MeshingLimits(
        maximum_cells=count,
        maximum_connectivity_entries=8 * count,
        maximum_scratch_bytes=8 * 1024**2,
    )
    starts = (-7, -5, -3)
    knots = tuple(
        np.asarray((start, start + size), dtype=np.float64)
        for start, size in zip(starts, sizes, strict=True)
    )
    originals = tuple(axis.copy() for axis in knots)
    with native_meshing_request(limits) as (scope, budget, _):
        layout = _prepare_grid_layout(
            knots, NativeHexGridSchedule("balanced_grid"), limits, 1.0,
            scope=scope, budget=budget,
        )
        assert layout.depth == 6
        assert layout.required == count == 496
        assert layout.extraction_bytes < limits.maximum_scratch_bytes
        assert 1024 * (1 << layout.depth) ** 3 > limits.maximum_scratch_bytes
        before_work = scope.remaining().remaining_work_units
        extracted = _extract_balanced_cells(layout, limits, scope=scope)
        assert before_work - scope.remaining().remaining_work_units == layout.extraction_work
        selected = sum(prod(shape) for shape in layout.refined_shapes)
        assert extracted.octree_levels.size == 1 + 7 * selected
        assert np.unique(extracted.octree_levels).size > 1
        assert extracted.closure_rounds == layout.depth

        address = MortonAddressPlan((0.0,) * 3, (64.0,) * 3, layout.depth)
        refined = []
        for level, shape in enumerate(layout.refined_shapes):
            corners = np.asarray(list(product(*(range(value) for value in shape))))
            refined.append(np.sort(
                morton_encode_integer_host(corners << (layout.depth - level), layout.depth)
                >> np.uint64(3 * (layout.depth - level))
            ))
            if level:
                assert np.all(np.isin(refined[level] >> np.uint64(3), refined[level - 1]))
        # Independently exercise closure: the halo rectangles are already a
        # fixed point, so balanced=True must not invent additional prefixes.
        unbalanced = refined_octree_leaves(address, refined, balanced=False)
        balanced = refined_octree_leaves(address, refined, balanced=True)
        for actual, expected in zip(balanced, unbalanced, strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(extracted.octree_prefixes, balanced[0])
        np.testing.assert_array_equal(extracted.octree_levels, balanced[1])
        lower = morton_decode_integer_host(
            extracted.octree_prefixes
            << (3 * (layout.depth - extracted.octree_levels)).astype(np.uint64),
            3, layout.depth,
        )
        widths = 1 << (layout.depth - extracted.octree_levels)
        assert sum(int(width) ** 3 for width in widths) == 64**3
        # Check genuine face/edge/corner 2:1 balance, not merely level extrema.
        for row in range(lower.shape[0]):
            touching = np.all(
                (lower <= lower[row] + widths[row])
                & (lower + widths[:, None] >= lower[row]), axis=1,
            )
            assert np.all(
                np.abs(extracted.octree_levels[touching] - extracted.octree_levels[row]) <= 1
            )

        expected_cells = np.asarray([
            tuple(2 * parent[axis] + child[axis] for axis in range(3))
            for parent in product(*(range(size // 2) for size in sizes))
            for child in product(range(2), repeat=3)
        ], dtype=np.int64)
        np.testing.assert_array_equal(extracted.integer_cells, expected_cells)
        np.testing.assert_array_equal(
            extracted.integer_vertices[extracted.cells],
            expected_cells[:, None, :] + extracted.offsets[None, :, :],
        )
        assert extracted.cells.shape == (count, 8)
        basis = np.asarray(((0, -1, 0), (0, 0, 1), (-1, 0, 0)), dtype=np.float64)
        coordinates, exact = _grid_vertex_coordinates(
            extracted.integer_vertices, layout.exact_axis_knots, layout.subdivisions,
            basis, scope=scope, budget=budget,
        )
        expected_coordinates = (extracted.integer_vertices + np.asarray(starts)) @ basis
        np.testing.assert_array_equal(coordinates, expected_coordinates)
        assert exact == tuple(
            tuple(Fraction(int(value)) for value in row) for row in expected_coordinates
        )
    # All real 1:8 children are active here: one below the actual requirement
    # refuses before prefix/template allocation, without changing source knots.
    refused = MeshingLimits(
        maximum_cells=count - 1,
        maximum_connectivity_entries=8 * count,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
    )
    with native_meshing_request(refused) as (scope, budget, _):
        with pytest.raises(MeshingFailure, match="check_family_limits.capacity"):
            _prepare_grid_layout(
                knots, NativeHexGridSchedule("balanced_grid"), refused, 1.0,
                scope=scope, budget=budget,
            )
    for original, axis in zip(originals, knots, strict=True):
        np.testing.assert_array_equal(axis, original)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_independent_curved_source_restrictions_have_zero_fidelity(
    route: NativeHexGridRoute,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    schedule = NativeHexGridSchedule(route)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            schedule,
            MeshingLimits(),
            0.5,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
        from phydrax.discretization._coordinate_enclosure import CoordinateSourceBank
        from phydrax.discretization._reference_cell_types import CellGeometryElement
        from phydrax.meshing import _hex_generation as generation

        placements: list[generation._MappedRootPlacement | None] = []
        original_prepare = generation._prepare_mapped_root_placement

        def capture_placement(
            element: CellGeometryElement,
            coefficients: CoordinateSourceBank,
            /,
            *,
            budget: CoordinateEnclosureBudget,
        ) -> generation._MappedRootPlacement | None:
            placement = original_prepare(element, coefficients, budget=budget)
            placements.append(placement)
            return placement

        monkeypatch.setattr(generation, "_prepare_mapped_root_placement", capture_placement)
        result = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert len(placements) == 1
        assert placements[0] is not None
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert sum(block.cell_count for block in result.mesh.blocks) == 8
        assert np.min(result.scaled_jacobian_lower) > 0.9
        assert np.min(result.mean_ratio_lower) > 0.9
        embedding = certify_global_embedding(
            result.mesh,
            result.geometry,
            certify_cell_geometry_validity(result.geometry, mesh=result.mesh),
        )
        coverage = certify_domain_coverage(
            result.mesh, result.geometry, domain, result.cell_regions, embedding=embedding
        )
        assert embedding.status == "certified"
        assert coverage.status == "certified"
        source = MappedDomainBoundarySource(domain, result.cell_regions)
        fidelity = certify_source_fidelity(
            result.mesh, result.geometry, source, tolerance=0.0
        )
        assert fidelity.status == "certified"
        assert fidelity.mesh_to_source_lower == fidelity.mesh_to_source_upper == 0.0
        assert fidelity.source_to_mesh_lower == fidelity.source_to_mesh_upper == 0.0
        assert (
            fidelity.domain_coverage is not None
            and fidelity.domain_coverage.status == "certified"
        )


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
@pytest.mark.parametrize("kind", ("cavity", "nonsweepable", "material"))
def test_grid_portfolio_closes_nontrivial_curved_domains(
    route: NativeHexGridRoute, kind: str
) -> None:
    if kind == "cavity":
        occupied = set(product(range(3), range(3), range(3))) - {(1, 1, 1)}
    elif kind == "nonsweepable":
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    else:
        occupied = {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    complex_, domain = mapped_occupied_source(occupied, materials=kind == "material")
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule(route),
            MeshingLimits(),
            1.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert {block.cell_kind for block in mapped.mesh.blocks} == {"hexahedron"}
        assert sum(block.cell_count for block in mapped.mesh.blocks) == len(occupied)
        assert np.min(mapped.scaled_jacobian_lower) >= 0.85
        assert np.min(mapped.mean_ratio_lower) >= 0.85
        assert domain.exact_region_measures() == tuple(
            np.sum(domain.cell_regions == region)
            for region in range(len(domain.region_ids))
        )
        embedding = certify_global_embedding(
            mapped.mesh,
            mapped.geometry,
            certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh),
        )
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        assert embedding.status == coverage.status == "certified"
        if route == "balanced_grid":
            assert np.max(grid.octree_levels) == grid.closure_rounds
            assert np.all(grid.octree_levels > 0)
            finest_parents = np.prod([
                (axis.size // 2) for axis in grid.axis_knots
            ])
            assert np.count_nonzero(grid.octree_levels == grid.closure_rounds) == 8 * finest_parents
            assert sum(
                8 ** (grid.closure_rounds - int(level)) for level in grid.octree_levels
            ) == 8 ** grid.closure_rounds


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
@pytest.mark.parametrize(
    "limit", ("maximum_edges", "maximum_faces", "maximum_data_bytes")
)
def test_integer_grid_rejects_actual_entity_and_publication_capacity(
    route: NativeHexGridRoute,
    limit: str,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    source_coordinates = np.asarray(domain.reference_mesh.coordinates).copy()
    limits = MeshingLimits(**{limit: 1})
    with native_meshing_request(limits) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            generate_integer_grid_hexes(
                complex_,
                domain,
                NativeHexGridSchedule(route),
                limits,
                1.0,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(
            domain.reference_mesh.coordinates, source_coordinates
        )


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_fine_grid_capacity_refusal_preserves_original_source(
    route: NativeHexGridRoute,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    original = np.asarray(domain.reference_mesh.coordinates).copy()
    with native_meshing_request(MeshingLimits(maximum_cells=8)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            generate_integer_grid_hexes(
                complex_,
                domain,
                NativeHexGridSchedule(route, maximum_depth=20),
                MeshingLimits(maximum_cells=8),
                1.0e-6,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(domain.reference_mesh.coordinates, original)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_integer_grid_entity_capacity_counts_retained_cavity_not_bounding_box(
    route: NativeHexGridRoute,
) -> None:
    occupied = {(0, 0, 0), (2, 0, 0)}
    complex_, domain = mapped_occupied_source(occupied, curvature=0.125)
    limits = MeshingLimits(maximum_edges=24, maximum_faces=12)
    with native_meshing_request(limits) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule(route),
            limits,
            1.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert sum(block.cell_count for block in grid.mesh.blocks) == len(occupied)
        assert entities(grid.mesh, 1).shape[0] == 24
        assert entities(grid.mesh, 2).shape[0] == 12


@pytest.mark.parametrize("route", ("mapped_balanced_grid_hex", "mapped_frame_grid_hex"))
def test_original_zero_tolerance_curved_size_request_remains_unmet(
    route: phx.meshing.NativeMeshingRoute, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aim at the target, without disguising an unmet stricter compliance policy."""
    from phydrax.meshing.providers import _native_dual as provider

    original = provider._mapped_grid_construction
    placement_sizes = []

    def capture_size(*args: Any, **kwargs: Any) -> Any:
        placement_sizes.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(provider, "_mapped_grid_construction", capture_size)

    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    reference = phx.meshing.NativePlcSource(
        complex_, domain.reference_domain.source_id, "reference-state"
    )
    source = phx.meshing.NativeMappedHexSource(reference, domain)
    faces = domain.reference_mesh.topology.entities(2)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        faces.entity_ids,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.5,
                maximum_size=0.6,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(),
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(route)
    ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
    with pytest.raises(MeshingFailure) as failed:
        plan.execute()
    control = request.size_controls[0]
    assert isinstance(control, phx.meshing.UniformSizeControl)
    assert placement_sizes == [control.target_size]
    assert failed.value.category is phx.meshing.MeshingFailureCategory.COMPLIANCE_FAILED
    achieved = dict(failed.value.evidence.achieved)
    key = f"size:{request.size_controls[0].control_id}"
    assert achieved[f"{key}:p50_edge"] < 0.5
    if route == "mapped_frame_grid_hex":
        assert achieved[f"{key}:p95_edge"] == 0.5
    else:
        assert achieved[f"{key}:p95_edge"] < 0.5
    assert dict(failed.value.evidence.requested)[f"{key}:target_size"] == 0.5
    assert (
        failed.value.stage == phx.meshing.MeshingStageKind.SPECIFICATION_COMPLIANCE.value
    )


@dataclass(frozen=True, slots=True)
class _MaterialPublication:
    result: CellMeshingResult
    domain: MappedReferenceDomain
    expected_measures: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _MaterialRoundTrip:
    coarsening: MeshAdaptationResult
    fields: dict[str, Array]
    restored_fields: dict[str, Array]
    refined_evidence: TransferEvidence
    restored_evidence: TransferEvidence


@pytest.fixture(
    scope="module",
    params=(
        pytest.param((2.0, 2.0, (2.0, 1.0)), id="binary-inverse"),
        pytest.param((3.0, 2.5, (3.0, 1.5)), id="nonbinary-inverse"),
    ),
)
def material_publication(request: pytest.FixtureRequest) -> _MaterialPublication:
    axis_scale, maximum_size, expected_measures = request.param
    result, domain = nonunit_material_publication(axis_scale, maximum_size)
    return _MaterialPublication(result, domain, expected_measures)


@pytest.fixture(scope="module")
def material_refinement(
    material_publication: _MaterialPublication,
) -> MeshAdaptationResult:
    result, domain = material_publication.result, material_publication.domain
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in result.mesh.blocks]
    )
    refinement = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MarkedMeshAdaptation(cell_ids),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=transfer,
                audit_policy=phx.meshing.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    assert refinement.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    return refinement


@pytest.fixture(scope="module")
def material_round_trip(
    material_publication: _MaterialPublication,
    material_refinement: MeshAdaptationResult,
) -> _MaterialRoundTrip:
    result, domain = material_publication.result, material_publication.domain
    refinement = material_refinement
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    fields = allfield_state(result)
    refined_fields, refined_evidence = allfield_transfer(refinement, fields, refine=True)
    assert set(refined_evidence) == set(fields)
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in refinement.target.mesh.blocks]
    )
    coarsening = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            refinement.target,
            phx.meshing.MarkedMeshAdaptation(
                np.empty((0,), dtype=np.int64), fine_ids, hierarchy=refinement.hierarchy
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=transfer,
                audit_policy=phx.meshing.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    assert coarsening.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    restored_fields, restored_evidence = allfield_transfer(
        coarsening, refined_fields, refine=False
    )
    return _MaterialRoundTrip(
        coarsening, fields, restored_fields, refined_evidence, restored_evidence
    )


def test_nonunit_curved_roots_publish_materials_and_preserve_source_strata(
    material_publication: _MaterialPublication,
) -> None:
    result, domain, expected_measures = (
        material_publication.result,
        material_publication.domain,
        material_publication.expected_measures,
    )
    assert result.audit.passed and result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert isinstance(result.certification.coverage, DomainCoverageCertificate)
    for actual, expected in zip(
        result.certification.coverage.achieved_region_measures,
        expected_measures,
        strict=True,
    ):
        assert actual is not None
        assert actual == pytest.approx(expected, rel=0.0, abs=VOLUME_TOLERANCE)
    assert {zone.material_id for zone in result.zones} == {
        "material-left",
        "material-right",
    }
    assert "material-interface" in {patch.name for patch in result.patches}
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    vertex, dimensions = transfer.source_associations(result)
    assert dimensions == (1, 2, 3)
    assert vertex.exact


def test_nonunit_curved_roots_refine_with_positive_pure_hex_maps(
    material_publication: _MaterialPublication,
    material_refinement: MeshAdaptationResult,
) -> None:
    refinement = material_refinement
    assert refinement.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    transfer = phx.meshing.MappedReferenceAssociationTransfer(material_publication.domain)
    transfer.source_associations(refinement.target)
    assert (
        refinement.target.certification is not None
        and refinement.target.certification.passed
    )
    assert {block.cell_kind for block in refinement.target.mesh.blocks} == {"hexahedron"}


def test_nonunit_curved_roots_continuation_retains_all_declared_fields(
    material_round_trip: _MaterialRoundTrip,
) -> None:
    cycle = material_round_trip
    assert cycle.coarsening.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert set(cycle.refined_evidence) == set(cycle.fields)
    assert set(cycle.restored_evidence) == set(cycle.fields)
    assert set(allfield_inventory(cycle.coarsening.target, cycle.restored_fields)) == set(
        cycle.fields
    )


@pytest.mark.parametrize("kind", ("H1", "DG", "Hcurl", "Hdiv"))
def test_nonunit_curved_roots_form_history_round_trip(
    material_round_trip: _MaterialRoundTrip,
    kind: str,
) -> None:
    cycle = material_round_trip
    np.testing.assert_allclose(
        cycle.restored_fields[kind],
        cycle.fields[kind],
        rtol=TRANSFER_TOLERANCE,
        atol=TRANSFER_TOLERANCE,
    )


def test_allfield_inventory_refuses_a_missing_role(
    material_publication: _MaterialPublication,
) -> None:
    fields = allfield_state(material_publication.result)
    del fields["DG"]
    with pytest.raises(AssertionError):
        allfield_inventory(material_publication.result, fields)


def test_allfield_inventory_refuses_truncated_history(
    material_publication: _MaterialPublication,
) -> None:
    fields = allfield_state(material_publication.result)
    fields["H1"] = fields["H1"][:, :1]
    with pytest.raises(AssertionError, match="H1"):
        allfield_inventory(material_publication.result, fields)


def test_mapped_corners_use_correct_rounding_of_the_original_polynomial() -> None:

    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule("frame_grid"),
            MeshingLimits(),
            1.0 / 3.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        expected = []
        for x, y, z in grid.exact_vertex_coordinates:
            expected.append((float(x), float(y), float(z + x * x * y / 8)))
        np.testing.assert_array_equal(
            mapped.mesh.coordinates, np.asarray(expected, dtype=np.float64)
        )


def test_private_root_placement_copies_atoms_and_evaluates_actual_source() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.meshing import _hex_generation as generation

    source = coordinate_lagrange_element("hexahedron", 2)
    bank = tuple(
        (x, y, z + x * x * y / 8)
        for x, y, z in (
            tuple(Fraction(float(value)) for value in row)
            for row in source.reference_nodes
        )
    )
    point = (Fraction(1, 3), Fraction(2, 5), Fraction(3, 7))
    with coordinate_reference_budget() as budget:
        original = enclosure.coordinate_polynomials(source, bank, budget=budget)
        assert original is not None
        placement = generation._prepare_mapped_root_placement(source, bank, budget=budget)
        assert placement is not None
        for actual, public in zip(placement.polynomials, original, strict=True):
            assert actual == public
            assert actual is not public
            for index, value in actual.items():
                assert type(value) is Fraction
                assert value is not public[index]
        before_retained = budget.retained_basis_bytes
        with budget.live_storage() as storage:
            image = generation._mapped_root_point_image(
                placement, point, storage=storage, budget=budget
            )
            assert image == (
                point[0], point[1], point[2] + point[0] ** 2 * point[1] / 8
            )
            assert storage.bound > 0
            assert budget.retained_basis_bytes == before_retained


@pytest.mark.parametrize("radical", (False, True))
def test_private_root_placement_reauthenticates_changed_public_bank_or_falls_back(
    radical: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.meshing import _hex_generation as generation

    source = coordinate_lagrange_element("hexahedron", 2)
    original = tuple(
        tuple(Fraction(float(value)) for value in row) for row in source.reference_nodes
    )
    changed = tuple((x + Fraction(1, 13), y, z) for x, y, z in original)
    banks = (
        tuple(tuple(RadicalElement(value) for value in row) for row in bank)
        for bank in (original, changed)
    ) if radical else (original, changed)
    with coordinate_reference_budget() as budget:
        placements = [
            generation._prepare_mapped_root_placement(source, bank, budget=budget)
            for bank in banks
        ]
        if radical:
            assert placements == [None, None]
        else:
            first, second = placements
            assert first is not None and second is not None
            with budget.live_storage() as storage:
                zero = (Fraction(0), Fraction(0), Fraction(0))
                assert generation._mapped_root_point_image(
                    first, zero, storage=storage, budget=budget
                ) == zero
                assert generation._mapped_root_point_image(
                    second, zero, storage=storage, budget=budget
                ) == (Fraction(1, 13), Fraction(0), Fraction(0))


def _streamed_placement_fixture(budget: Any, roots: int = 3) -> tuple[Any, ...]:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import (
        _cardinal_corner_nodes,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.meshing import _hex_generation as generation

    source = coordinate_lagrange_element("hexahedron", 2)
    chart = coordinate_lagrange_element("hexahedron", 1)
    nodes = _cardinal_corner_nodes(chart, budget=budget)
    elements, groups, cells, parents, vertices = {}, [], [], [], {}
    for half, name in enumerate(("left", "right")):
        controls = np.asarray(chart.reference_nodes, dtype=np.int64)
        controls[:, 0] += half
        denominators = np.ones_like(controls)
        denominators[:, 0] = 2
        elements[name] = PolynomialComposedCellGeometryElement(
            source, chart, controls, denominators, budget=budget
        )
        rows = []
        # Deliberately put the largest root first in the publication order.
        for parent in reversed(range(roots)):
            rows.append(len(cells))
            parents.append(parent)
            cell = []
            for x, y, z in reference_cell_topology("hexahedron").vertices:
                point = (Fraction(parent) + Fraction(int(x) + half, 2), Fraction(y), Fraction(z))
                cell.append(vertices.setdefault(point, len(vertices)))
            cells.append(cell)
        groups.append((name, np.asarray(rows, dtype=np.int64)))
    banks = {
        parent: tuple(
            (x + parent, y, z + (x + parent) ** 2 * y / 8)
            for x, y, z in (
                tuple(Fraction(float(value)) for value in row)
                for row in source.reference_nodes
            )
        )
        for parent in range(roots)
    }
    placements = {
        parent: generation._prepare_mapped_root_placement(source, bank, budget=budget)
        for parent, bank in banks.items()
    }
    points = np.full((len(vertices), 3), np.nan)
    return (
        np.asarray(cells, dtype=np.int64), np.asarray(parents, dtype=np.int64),
        groups, elements, placements, banks, nodes, points,
    )


def _old_chart_major_placement(*args: Any, budget: Any, **kwargs: Any) -> None:
    """Independent expanded composed-law reference, without the private memo."""
    from phydrax.discretization import _coordinate_enclosure as enclosure

    cells, parents, groups, elements, _, banks, _, points = args
    witnesses = {}
    for name, rows in groups:
        for row in rows:
            images = enclosure.coordinate_corner_images(
                elements[name], banks[int(parents[row])], budget=budget
            )
            assert images is not None
            for vertex, image in zip(cells[row], images, strict=True):
                assert vertex not in witnesses or witnesses[vertex] == image
                witnesses[vertex] = image
                points[vertex] = enclosure.rounded_point(image)


@pytest.mark.parametrize("roots", (3, 9))
def test_streamed_placement_matches_expanded_law_and_retires_images(
    roots: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _hex_generation as generation

    with coordinate_reference_budget() as budget:
        fixture = _streamed_placement_fixture(budget, roots)
        expected = fixture[-1].copy()
        _old_chart_major_placement(*fixture[:-1], expected, budget=budget)
        before_cells, before_parents = fixture[0].copy(), fixture[1].copy()
        placements = fixture[4]
        calls, live = [], []
        original = generation._mapped_root_point_image

        def capture(placement: Any, point: Any, **kwargs: Any) -> Any:
            parent = next(parent for parent, owner in placements.items() if owner is placement)
            calls.append((parent, point))
            image = original(placement, point, **kwargs)
            live.append(kwargs["storage"].bound)
            return image

        monkeypatch.setattr(generation, "_mapped_root_point_image", capture)
        with budget.live_storage() as storage:
            generation._place_mapped_geometry(*fixture, storage=storage, budget=budget)
            assert storage.bound == 0
        np.testing.assert_array_equal(fixture[-1].view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(fixture[0], before_cells)
        np.testing.assert_array_equal(fixture[1], before_parents)
        assert len(calls) == len(set(calls)) == 12 * roots
        assert [parent for parent, _ in calls] == sorted(parent for parent, _ in calls)
        # At most one root's corners and the preceding interface can be live,
        # independent of whether the complete source has three or nine roots.
        assert max(live) <= 16 * max(
            live[index] - (live[index - 1] if index else 0)
            for index in range(len(live))
        )


@pytest.mark.parametrize("late", (False, True))
def test_streamed_placement_keeps_cross_root_witness_until_last_use(
    late: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction

    from phydrax.meshing import _hex_generation as generation

    with coordinate_reference_budget() as budget:
        fixture = _streamed_placement_fixture(budget)
        target = fixture[4][2 if late else 1]
        original = generation._mapped_root_point_image

        def conflicting(placement: Any, point: Any, **kwargs: Any) -> Any:
            image = original(placement, point, **kwargs)
            if placement is target and point == (Fraction(0), Fraction(1), Fraction(1)):
                return (image[0], image[1], image[2] + Fraction(1, 17))
            return image

        monkeypatch.setattr(generation, "_mapped_root_point_image", conflicting)
        with budget.live_storage() as storage:
            with pytest.raises(MeshingFailure, match="restriction_traces must agree"):
                generation._place_mapped_geometry(*fixture, storage=storage, budget=budget)


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_streamed_placement_exact_resource_boundary(resource: str) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing import _hex_generation as generation

    def run(limits: MeshingLimits) -> tuple[int, int]:
        with coordinate_reference_budget(limits) as budget:
            fixture = _streamed_placement_fixture(budget)
            with budget.live_storage() as storage:
                generation._place_mapped_geometry(*fixture, storage=storage, budget=budget)
                assert storage.bound == 0
            return budget.work_units, budget.peak_bytes_upper

    work, storage = run(MeshingLimits())
    required = work if resource == "work" else storage
    field = "maximum_work_units" if resource == "work" else "maximum_scratch_bytes"
    assert run(MeshingLimits(**{field: required})) == (work, storage)
    with pytest.raises(CoordinateEnclosureResourceError):
        run(MeshingLimits(**{field: required - 1}))


def test_streamed_placement_preserves_published_ids_actions_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _hex_generation as generation

    complex_, domain = mapped_occupied_source({(0, 0, 0), (1, 0, 0)}, curvature=0.125)
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        grid = generate_integer_grid_hexes(
            complex_, domain, NativeHexGridSchedule("frame_grid"), limits, 0.5,
            scope=scope, budget=budget,
        )
        streamed = realize_mapped_grid_hexes(
            grid, domain.reference_mesh, domain.source_geometry, domain.cell_regions,
            limits, scope=scope, budget=budget,
        )
        monkeypatch.setattr(generation, "_place_mapped_geometry", _old_chart_major_placement)
        old = realize_mapped_grid_hexes(
            grid, domain.reference_mesh, domain.source_geometry, domain.cell_regions,
            limits, scope=scope, budget=budget,
        )
        assert streamed.mapped_id == old.mapped_id
        assert streamed.mesh.mesh_id == old.mesh.mesh_id
        assert streamed.reference_mesh.mesh_id == old.reference_mesh.mesh_id
        for attribute in (
            "cell_regions", "source_parent_rows", "target_reference_rows",
            "scaled_jacobian_lower", "mean_ratio_lower", "aspect_ratio_upper",
        ):
            np.testing.assert_array_equal(getattr(streamed, attribute), getattr(old, attribute))
        np.testing.assert_array_equal(streamed.mesh.coordinates, old.mesh.coordinates)
        streamed_elements, streamed_routes, _ = streamed.geometry.resolve(streamed.mesh, budget=budget)
        old_elements, old_routes, _ = old.geometry.resolve(old.mesh, budget=budget)
        for actual, expected, route, old_route in zip(
            streamed_elements, old_elements, streamed_routes, old_routes, strict=True,
        ):
            assert actual.element_id == expected.element_id
            np.testing.assert_array_equal(route, old_route)


def _expanded_nominal_box_quality_reference(element: Any, bank: Any) -> Any:
    """Independent pre-reuse expanded-child convolution, using Fraction controls."""
    from fractions import Fraction
    from math import comb, prod

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.meshing._hex_generation import _MappedQualitySource

    def weights(shape: tuple[int, ...]) -> np.ndarray:
        result = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            result[index] = prod(
                [comb(length - 1, value) for length, value in zip(shape, index, strict=True)]
            )
        return result

    def multiply(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
        if first is None or second is None:
            return None
        shape = tuple(a + b - 1 for a, b in zip(first.shape, second.shape, strict=True))
        result = np.full(shape, Fraction(0), dtype=object)
        for index in np.ndindex(first.shape):
            slices = tuple(slice(value, value + length)
                           for value, length in zip(index, second.shape, strict=True))
            result[slices] += first[index] * second
        return result

    def combine(
        first: np.ndarray | None, second: np.ndarray | None, subtract: bool = False
    ) -> np.ndarray | None:
        if second is None:
            return first
        if first is None:
            return -second if subtract else second
        return first - second if subtract else first + second

    def bound(net: np.ndarray | None, lower: bool = False) -> Fraction:
        if net is None:
            return Fraction(0)
        controls = net / weights(net.shape)
        return min(controls.flat) if lower else max(abs(value) for value in controls.flat)

    with coordinate_reference_budget() as budget:
        polynomials = enclosure.coordinate_polynomials(element, bank, budget=budget)
        assert polynomials is not None
        jacobian = []
        for polynomial in polynomials:
            degrees = tuple(max((index[axis] for index in polynomial), default=0)
                            for axis in range(3))
            controls = np.asarray(
                enclosure.bernstein_coefficients(polynomial, "box", 3, budget=budget),
                dtype=object,
            ).reshape(tuple(degree + 1 for degree in degrees))
            row = []
            for axis, degree in enumerate(degrees):
                derivative = None if degree == 0 else np.diff(controls, axis=axis) * degree
                row.append(None if derivative is None
                           else derivative * weights(derivative.shape))
            jacobian.append(row)
    minors = []
    for removed_row in range(3):
        a, b = [row for row in range(3) if row != removed_row]
        row = []
        for removed_column in range(3):
            c, d = [column for column in range(3) if column != removed_column]
            row.append(combine(multiply(jacobian[a][c], jacobian[b][d]),
                               multiply(jacobian[a][d], jacobian[b][c]), True))
        minors.append(row)
    determinant = None
    for column in range(3):
        determinant = combine(determinant, multiply(jacobian[0][column], minors[0][column]),
                              column == 1)
    magnitudes = tuple(tuple(bound(net) for net in row) for row in jacobian)
    cofactor_magnitudes = tuple(tuple(bound(net) for net in row) for row in minors)

    def norm_product(rows: tuple[tuple[Fraction, ...], ...]) -> Fraction:
        return max(sum(row, Fraction(0)) for row in rows) * max(
            sum((row[column] for row in rows), Fraction(0)) for column in range(3)
        )

    return _MappedQualitySource(
        tuple(sum((row[column] ** 2 for row in magnitudes), Fraction(0)) for column in range(3)),
        norm_product(magnitudes) * norm_product(cofactor_magnitudes),
        bound(determinant, True),
    )


def _quality_root_box_fixture(case: str, *, cancellation: bool = False) -> Any:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )

    root = coordinate_lagrange_element("hexahedron", 2)
    chart = coordinate_lagrange_element("hexahedron", 1)
    nodes = tuple(tuple(Fraction(float(value)) for value in row) for row in root.reference_nodes)
    bank = tuple(
        (u, v, w + u * u * v / 8) if cancellation else
        (u + u * v / 32, v + u * v / 32,
         w + u * u * v / 64 + u * v * w / 64)
        for u, v, w in nodes
    )
    maps = {
        "anisotropic": (((0, 1, 2), (1, 1, 1), (3, 5, 7), (0, 0, 0)),),
        "reversed": (((0, 1, 2), (-1, -1, 1), (3, 5, 7), (1, 1, 0)),),
        "permuted": (((1, 2, 0), (1, 1, 1), (3, 5, 7), (0, 0, 0)),),
        "tiny": (((0, 1, 2), (1, 1, 1), (2**30, 2**31, 2**32), (0, 0, 0)),),
        "nested": (
            ((1, 2, 0), (-1, -1, 1), (3, 5, 7), (1, 1, 0)),
            ((2, 0, 1), (1, 1, 1), (11, 13, 17), (0, 0, 0)),
        ),
    }
    element = root
    with coordinate_reference_budget() as budget:
        for axes, signs, denominators, offsets in maps[case]:
            controls = tuple(
                tuple(Fraction(offset) + Fraction(sign, denominator) * Fraction(float(point[axis]))
                      for axis, sign, denominator, offset in
                      zip(axes, signs, denominators, offsets, strict=True))
                for point in chart.reference_nodes
            )
            element = PolynomialComposedCellGeometryElement(
                element, chart,
                np.asarray([[value.numerator for value in row] for row in controls], dtype=object),
                np.asarray([[value.denominator for value in row] for row in controls], dtype=object),
                budget=budget,
            )
    return element, bank


@pytest.mark.parametrize("case", ("anisotropic", "reversed", "permuted", "tiny", "nested"))
@pytest.mark.parametrize("cancellation", (False, True))
def test_root_box_quality_matches_independent_expanded_nominal_nets(
    case: str, cancellation: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.meshing import _hex_generation as generation

    element, bank = _quality_root_box_fixture(case, cancellation=cancellation)
    expected = _expanded_nominal_box_quality_reference(element, bank)
    prepare = generation._prepare_box_quality_operands
    calls = []

    def counted(polynomials: Any, *, budget: Any) -> Any:
        calls.append(polynomials)
        return prepare(polynomials, budget=budget)

    coordinate_polynomials = enclosure.coordinate_polynomials

    def root_or_chart_only(actual_element: Any, local: Any, *, budget: Any) -> Any:
        assert not isinstance(actual_element, PolynomialComposedCellGeometryElement)
        return coordinate_polynomials(actual_element, local, budget=budget)

    monkeypatch.setattr(enclosure, "coordinate_polynomials", root_or_chart_only)
    monkeypatch.setattr(generation, "_prepare_box_quality_operands", counted)
    with coordinate_reference_budget() as budget:
        actual = generation._prepare_mapped_quality_source(
            element, bank, np.arange(len(bank)), "hexahedron", "box", budget=budget
        )
        retained = budget.retained_basis_bytes
        second = generation._prepare_mapped_quality_source(
            element, bank, np.arange(len(bank)), "hexahedron", "box", budget=budget
        )
        assert actual == second == expected
        assert len(calls) == len(budget.root_box_quality_operands_cache) == 1
        assert budget.retained_basis_bytes == retained


def test_root_box_quality_reauthenticates_mutated_public_fraction() -> None:
    from fractions import Fraction

    from phydrax.meshing import _hex_generation as generation

    element, bank = _quality_root_box_fixture("anisotropic")
    atom = bank[0][0]
    numerator, denominator = atom.numerator, atom.denominator
    with coordinate_reference_budget() as budget:
        first = generation._root_box_quality_source(element, bank, budget=budget)
        try:
            object.__setattr__(atom, "_numerator", 1)
            object.__setattr__(atom, "_denominator", 4096)
            expected = _expanded_nominal_box_quality_reference(element, bank)
            second = generation._root_box_quality_source(element, bank, budget=budget)
            assert second == expected
            assert second != first
            assert len(budget.root_box_quality_operands_cache) == 2
            # The old request-private key and controls did not alias this atom.
            first_key = next(iter(budget.root_box_quality_operands_cache))
            stored_atom = first_key[2][0][0]
            assert isinstance(stored_atom, Fraction)
            assert stored_atom.numerator == numerator
            assert stored_atom.denominator == denominator
            assert stored_atom is not atom
        finally:
            object.__setattr__(atom, "_numerator", numerator)
            object.__setattr__(atom, "_denominator", denominator)


def test_root_box_quality_authenticates_complete_child_and_source_tree() -> None:
    import equinox as eqx

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.meshing import _hex_generation as generation

    element, bank = _quality_root_box_fixture("anisotropic", cancellation=True)
    with coordinate_reference_budget() as budget:
        first = generation._root_box_quality_source(element, bank, budget=budget)
        old_identity = enclosure.element_identity(element, budget=budget)
        moved_chart = eqx.tree_at(
            lambda value: value.chart_coordinates, element,
            element.chart_coordinates + 1 / 1024,
        )
        assert enclosure.element_identity(moved_chart, budget=budget) != old_identity
        assert generation._root_box_quality_source(moved_chart, bank, budget=budget) == first
        assert len(budget.root_box_quality_operands_cache) == 1
        nodes = np.asarray(element.source_element.reference_nodes).copy()
        nodes[:, :2] = 1 - nodes[:, :2]
        moved_source = eqx.tree_at(
            lambda value: value.source_element.reference_nodes, element, nodes
        )
        expected = _expanded_nominal_box_quality_reference(moved_source, bank)
        actual = generation._root_box_quality_source(moved_source, bank, budget=budget)
        assert actual == expected
        assert len(budget.root_box_quality_operands_cache) == 2


@pytest.mark.parametrize("unsupported", ("mixed", "nonaffine", "outside", "radical"))
def test_root_box_quality_declines_unsupported_maps_without_root_cache(unsupported: str) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry import (
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.meshing import _hex_generation as generation

    element, bank = _quality_root_box_fixture("anisotropic")
    if unsupported == "radical":
        bank = tuple(tuple(RadicalElement(value) for value in row) for row in bank)
    else:
        chart = coordinate_lagrange_element("hexahedron", 1)
        controls = []
        for point in chart.reference_nodes:
            u, v, w = (Fraction(float(value)) for value in point)
            x = Fraction(1, 8) + u / 2
            if unsupported == "mixed":
                x += v / 8
            elif unsupported == "nonaffine":
                x += u * v / 8
            else:
                x -= Fraction(1, 4)
            controls.append((x, v / 2, w / 2))
        with coordinate_reference_budget() as budget:
            element = PolynomialComposedCellGeometryElement(
                element.source_element, chart,
                np.asarray([[value.numerator for value in row] for row in controls], dtype=object),
                np.asarray([[value.denominator for value in row] for row in controls], dtype=object),
                budget=budget,
            )
    with coordinate_reference_budget() as budget:
        assert generation._root_box_quality_source(element, bank, budget=budget) is None
        assert not budget.root_box_quality_operands_cache
        if unsupported != "radical":
            expected = _expanded_nominal_box_quality_reference(element, bank)
            actual = generation._prepare_mapped_quality_source(
                element, bank, np.arange(len(bank)), "hexahedron", "box", budget=budget
            )
            assert actual == expected


def test_root_box_quality_casteljau_admits_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    from fractions import Fraction

    from phydrax.meshing import _hex_generation as generation

    values = np.asarray([[[1]], [[3]], [[7]]], dtype=object)
    admissions = []

    def refuse(work: int, terms: int, dimension: int, bits: int, /, *, budget: Any) -> Never:
        admissions.append((work, terms, dimension, bits))
        raise ValueError("deliberate interpolation refusal")

    def unexpected_allocation(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("de Casteljau allocated before admission")

    monkeypatch.setattr(
        generation._dependency_phydrax_discretization__coordinate_budget,
        "reserve_polynomial", refuse,
    )
    monkeypatch.setattr(generation.np, "empty_like", unexpected_allocation)
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError, match="deliberate interpolation refusal"):
            generation._split_box_quality_axis(
                (values, 13, 3), 0, Fraction(1, 2**64 + 1), right=True, budget=budget
            )
    assert len(admissions) == 1
    assert admissions[0][0] > 0
    assert admissions[0][3] >= 128


def test_root_box_quality_reuses_actual_root_across_distinct_children(monkeypatch: pytest.MonkeyPatch) -> None:
    from phydrax.meshing import _hex_generation as generation

    children = [_quality_root_box_fixture(case) for case in
                ("anisotropic", "reversed", "permuted", "tiny", "nested")]
    expected = [_expanded_nominal_box_quality_reference(element, bank)
                for element, bank in children]
    prepare = generation._prepare_box_quality_operands
    calls = []

    def counted(polynomials: Any, *, budget: Any) -> Any:
        calls.append(polynomials)
        return prepare(polynomials, budget=budget)

    monkeypatch.setattr(generation, "_prepare_box_quality_operands", counted)
    with coordinate_reference_budget() as budget:
        actual = [generation._root_box_quality_source(element, bank, budget=budget)
                  for element, bank in children]
        assert actual == expected
        assert len(calls) == len(budget.root_box_quality_operands_cache) == 1


@pytest.mark.parametrize("mixed_basis", (False, True), ids=("shared-signed-axes", "mixed-exact-sums"))
def test_generated_coordinate_storage_follows_actual_shared_axis_operands(mixed_basis: bool) -> None:
    from fractions import Fraction

    from phydrax.meshing._hex_generation import _grid_vertex_coordinates

    side = 6 if mixed_basis else 20
    vertices = np.indices((side, side, side), dtype=np.int64).reshape(3, -1).T
    knots = tuple(tuple(Fraction(index, side - 1) for index in range(side)) for _ in range(3))
    basis = (
        np.asarray(((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.0625, 0.0, 1.0)))
        if mixed_basis else np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    )
    limits = MeshingLimits(maximum_scratch_bytes=20_000_000)
    with native_meshing_request(limits) as (scope, budget, _):
        physical, exact = _grid_vertex_coordinates(
            vertices, knots, 1, basis, scope=scope, budget=budget
        )
        assert budget.peak_bytes_upper <= limits.maximum_scratch_bytes
    expected = tuple(
        tuple(
            sum(
                (
                    Fraction(int(address[axis]), side - 1)
                    * Fraction(float(basis[axis, component]))
                    for axis in range(3)
                ),
                Fraction(0),
            )
            for component in range(3)
        )
        for address in vertices
    )
    assert exact == expected
    np.testing.assert_array_equal(
        physical, np.asarray(tuple(tuple(map(float, row)) for row in expected))
    )
    if not mixed_basis:
        assert len({id(value) for row in exact for value in row}) <= 3 * side
