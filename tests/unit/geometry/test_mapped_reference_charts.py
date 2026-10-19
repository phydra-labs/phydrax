# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Coordinate laws and shared-vertex refusal of retained reference charts."""

from fractions import Fraction

import jax
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_polynomials,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    MeshCertificateLimits,
)
from phydrax.geometry._mapped_reference_coverage import (
    _pyramid_reference_source,
    AffineReferenceChart,
    ComposedReferenceChart,
    exact_chart_mesh,
)


@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid")
)
def test_affine_reference_chart_retains_nonbinary_translation(kind: str) -> None:
    topology = reference_cell_topology(kind)
    dimension = len(topology.vertices[0])
    matrix = tuple(
        tuple(Fraction(axis == column) for column in range(dimension))
        for axis in range(dimension)
    )
    offset = (Fraction(1, 3), *(Fraction(0) for _ in range(dimension - 1)))
    chart = AffineReferenceChart(
        "root", 17, kind, tuple(range(len(topology.vertices))), matrix, offset
    )
    expected = tuple(
        tuple(
            Fraction(float(value)) + shift
            for value, shift in zip(corner, offset, strict=True)
        )
        for corner in topology.vertices
    )
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        prepared = exact_chart_mesh([chart], dimension, budget=budget)
        assert prepared is not None
        mesh, geometry = prepared
        elements, routes, controls = geometry.resolve(mesh)
        host_controls, host_routes = jax.device_get((controls, routes[0]))
        images = coordinate_corner_images(
            elements[0],
            np.asarray(host_controls)[np.asarray(host_routes)[0]],
            budget=budget,
        )
        assert images == expected
        np.testing.assert_array_equal(
            mesh.coordinates, np.asarray(expected, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            mesh.blocks[0].global_ids, np.asarray((17,), dtype=np.int64)
        )


def test_composed_reference_chart_retains_its_coordinate_law() -> None:
    element = coordinate_lagrange_element("triangle", 1)
    controls = np.asarray(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)), dtype=np.float64)
    chart = ComposedReferenceChart("root", 23, "triangle", (3, 7, 11), element, controls)
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        prepared = exact_chart_mesh([chart], 2, budget=budget)
        assert prepared is not None
        mesh, geometry = prepared
        elements, _, _ = geometry.resolve(mesh)
        assert elements[0].element_id == element.element_id
        np.testing.assert_array_equal(geometry.coordinates, controls)
        np.testing.assert_array_equal(
            mesh.vertex_global_ids, np.asarray((3, 7, 11), dtype=np.int64)
        )


def test_reference_charts_refuse_conflicting_shared_vertex_images() -> None:
    matrix = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    first = AffineReferenceChart(
        "first", 1, "triangle", (0, 1, 2), matrix, (Fraction(0), Fraction(0))
    )
    second = AffineReferenceChart(
        "second", 2, "triangle", (0, 3, 4), matrix, (Fraction(1, 3), Fraction(0))
    )
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        assert exact_chart_mesh([first, second], 2, budget=budget) is None


def test_pyramid_reference_frame_retains_whole_collapsed_coordinate_law() -> None:
    matrix = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    chart = AffineReferenceChart(
        "pyramid",
        31,
        "pyramid",
        (0, 1, 2, 3, 4),
        matrix,
        (Fraction(1, 3), Fraction(0), Fraction(0)),
    )
    # On collapsed coordinates (u, v, w), x = u(1-w) + w/2 + 1/3,
    # y = v(1-w) + w/2, z = w. These coefficient laws are independent
    # of the enclosing source frame chosen by chart preparation.
    expected = (
        {
            (0, 0, 0): Fraction(1, 3),
            (1, 0, 0): Fraction(1),
            (1, 0, 1): Fraction(-1),
            (0, 0, 1): Fraction(1, 2),
        },
        {(0, 1, 0): Fraction(1), (0, 1, 1): Fraction(-1), (0, 0, 1): Fraction(1, 2)},
        {(0, 0, 1): Fraction(1)},
    )
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        prepared = exact_chart_mesh([chart], 3, budget=budget)
        assert prepared is not None
        mesh, geometry = prepared
        elements, routes, controls = geometry.resolve(mesh)
        host_controls, host_routes = jax.device_get((controls, routes[0]))
        actual = coordinate_polynomials(
            elements[0],
            np.asarray(host_controls)[np.asarray(host_routes)[0]],
            budget=budget,
        )
        assert actual == expected


def test_pyramid_reference_frame_charges_each_rational_normalization() -> None:
    identity = coordinate_lagrange_element("tetrahedron", 1)
    nodal = coordinate_lagrange_element("pyramid", 1)
    coefficients = tuple(
        (
            Fraction(float(point[0])) + Fraction(1, 3),
            Fraction(float(point[1])),
            Fraction(float(point[2])),
        )
        for point in reference_cell_topology("pyramid").vertices
    )
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        with budget.measurement() as measurement:
            _pyramid_reference_source(identity, nodal, coefficients, budget=budget)
        # Five source-reference triples each require an addition and a division.
        assert measurement.work_units == 5 * 3 * 2
