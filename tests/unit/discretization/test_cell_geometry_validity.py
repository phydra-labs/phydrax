#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


D = phx.discretization
G = phx.geometry
Status = D.CellValidityStatus


def _polygon_status(points: Any, policy: Any = None) -> Any:
    points = np.asarray(points, dtype=np.float64)
    loop = np.arange(points.shape[0], dtype=np.int32)[None]
    mesh = D.CellMesh(points, (D.CellBlock("cells", "polygon", loop),))
    certificate = D.certify_cell_geometry_validity(mesh, policy=policy)
    return Status(int(certificate.status[0]))


def _pentagram() -> Any:
    angles = np.pi / 2.0 + 4.0 * np.pi * np.arange(5) / 5.0
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _embedded(points: Any, *, bend: Any = 0.0) -> Any:
    planar = np.column_stack((np.asarray(points, dtype=np.float64), np.zeros(5)))
    planar[3, 2] = bend
    angle = 0.4
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        )
    )
    return planar @ rotation.T


CONVEX = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0))
# Notched square whose vertex-zero fan leaves the cell.
CONCAVE = ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (1.5, 1.0), (0.0, 3.0))


def test_cell_geometry_validity_scenario_1() -> None:
    for points, expected in (
        (CONVEX, Status.CERTIFIED_VALID),
        (CONCAVE, Status.CERTIFIED_VALID),
        (CONVEX[::-1], Status.INVALID),
        # Bow tie with a crossing between non-adjacent edges.
        (((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)), Status.INVALID),
        (_pentagram(), Status.INVALID),
        # Repeated vertex coordinates behind distinct vertex ids.
        (((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 0.0), (0.0, 2.0)), Status.INVALID),
        # A vertex touching a non-adjacent edge.
        (((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)), Status.INVALID),
        # Adjacent edges doubling back along one line.
        (((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)), Status.INVALID),
    ):
        assert _polygon_status(points) == expected
    points = np.asarray(((0.0, 0.0), (1.0e-10, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    policy = D.CellValidityPolicy(relative_determinant_floor=1.0e-8)

    assert _polygon_status(points, policy) == Status.INVALID
    result = G.polygon_simplicity_2d(
        np.asarray(CONVEX),
        mode=G.PredicateMode.EXACT,
        maximum_candidate_pairs=1,
    )

    assert int(result.status) == G.PolygonSimplicityStatus.UNCERTAIN
    assert result.candidate_pair_count == 1
    assert result.candidate_capacity_exceeded
    assert _polygon_status(_embedded(CONCAVE)) == Status.CERTIFIED_VALID
    bent = _embedded(CONCAVE, bend=0.1)
    loose = D.CellValidityPolicy(relative_planarity_tolerance=0.5)

    assert _polygon_status(bent) == Status.INVALID
    assert _polygon_status(bent, loose) == Status.CERTIFIED_VALID


def _spike(offset: Any) -> Any:
    # Vertex 3 approaches edge 0 (on the line y = x) from above.
    tip = (0.3, 0.3 if offset == 0 else np.nextafter(0.3, 1.0))
    return np.asarray(((0.1, 0.1), (0.7, 0.7), (0.1, 0.9), tip, (0.0, 0.5)))


@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_filtered_simplicity_is_uncertain_where_exact_decides() -> None:
    for offset, exact in (
        (0, G.PolygonSimplicityStatus.SELF_INTERSECTING),
        (1, G.PolygonSimplicityStatus.SIMPLE),
    ):
        points = _spike(offset)
        filtered = G.polygon_simplicity_2d(points, mode=G.PredicateMode.FILTERED)
        resolved = G.polygon_simplicity_2d(points, mode=G.PredicateMode.EXACT)

        assert int(filtered.status) == G.PolygonSimplicityStatus.UNCERTAIN
        assert int(filtered.uncertain_pairs) > 0
        assert int(resolved.status) == exact
        assert int(resolved.uncertain_pairs) == 0
        expected = Status.INVALID if offset == 0 else Status.CERTIFIED_VALID
        assert _polygon_status(points) == expected


def test_cell_geometry_validity_scenario_2() -> None:
    a = np.asarray(((0.0, 0.0),) * 5)
    b = np.asarray(((2.0, 2.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0)))
    c = np.asarray(((0.0, 2.0), (1.0, 0.0), (1.0, 0.0), (2.0, 2.0), (2.0, 0.0)))
    d = np.asarray(((2.0, 0.0), (1.0, 1.0), (3.0, 0.0), (3.0, 3.0), (3.0, 0.0)))
    S = G.SegmentIntersectionStatus

    result = G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED)

    assert tuple(int(value) for value in result.status) == (
        S.PROPER_CROSSING,
        S.ENDPOINT_CONTACT,
        S.COLLINEAR_OVERLAP,
        S.DISJOINT,
        S.DISJOINT,
    )
    with pytest.raises(ValueError):
        G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED_DEVICE)
    points = np.asarray(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)))
    mesh = D.CellMesh(
        points, (D.CellBlock("cells", "polygon", np.arange(5, dtype=np.int32)[None]),)
    )

    with pytest.raises(phx.meshing.MeshingFailure, match="self_intersection"):
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _reference_mesh(kind: str) -> Any:
    vertices = np.asarray(D.reference_cell_topology(kind).vertices, dtype=np.float64)
    return D.CellMesh(
        vertices,
        (D.CellBlock("cells", kind, np.arange(vertices.shape[0], dtype=np.int32)[None]),),
    )


def _lagrange_geometry(kind: str, degree: int, *, shear: float = 0.0) -> Any:
    element = D.fem.lagrange_element(kind, degree)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[:, 0] += shear * nodes[:, -1]
    return D.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(nodes.shape[0])[None]}, nodes
    )


@pytest.mark.parametrize(
    ("kind", "degree"),
    [
        ("triangle", 2),
        ("triangle", 4),
        ("tetrahedron", 2),
        ("tetrahedron", 3),
        ("quadrilateral", 3),
        ("hexahedron", 2),
        ("prism", 2),
        ("pyramid", 2),
    ],
)
def test_source_rounding_enclosure_certifies_supported_degrees(
    kind: str, degree: int
) -> None:
    mesh = _reference_mesh(kind)
    geometry = _lagrange_geometry(kind, degree, shear=0.25)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)

    assert Status(int(certificate.status[0])) == Status.CERTIFIED_VALID
    assert certificate.unresolved_reasons == ()
    assert float(certificate.determinant_lower[0]) > 0.0


def test_bernstein_table_budget_is_explicit_unresolved_evidence() -> None:
    mesh = _reference_mesh("hexahedron")
    geometry = _lagrange_geometry("hexahedron", 2)
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64).copy()
    x, y, z = np.asarray(geometry.elements[0].reference_nodes).T
    amplitude = 1.0 / 64.0
    coordinates[:, 0] += amplitude * x * (1.0 - x) * y * z
    coordinates[:, 1] += amplitude * y * (1.0 - y) * z * x
    coordinates[:, 2] += amplitude * z * (1.0 - z) * x * y
    geometry = D.CellGeometrySpec(
        {"cells": geometry.elements[0]},
        {"cells": np.arange(coordinates.shape[0])[None]},
        coordinates,
    )
    policy = D.CellValidityPolicy(maximum_bernstein_nodes=32)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)

    assert Status(int(certificate.status[0])) == Status.UNRESOLVED
    assert certificate.unresolved_reasons == (("cells", "bernstein_node_budget"),)


def test_subdivision_exhaustion_is_reported_with_its_budget() -> None:
    mesh = D.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (D.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )
    element = D.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    # Valid everywhere, but a Bernstein edge coefficient is negative (the same
    # cell as the meshing audit tests): depth zero cannot decide it.
    nodes[3:] = ((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51))
    geometry = D.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(6)[None]}, nodes
    )
    shallow = D.certify_cell_geometry_validity(
        geometry, mesh=mesh, policy=D.CellValidityPolicy(maximum_subdivision_depth=0)
    )

    assert Status(int(shallow.status[0])) == Status.UNRESOLVED
    assert shallow.unresolved_reasons == (("cells", "subdivision_depth"),)


def test_certificate_is_bound_to_the_certified_coordinates_and_topology() -> None:
    mesh = _reference_mesh("tetrahedron")
    geometry = _lagrange_geometry("tetrahedron", 2)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    moved = _lagrange_geometry("tetrahedron", 2, shear=0.5)

    certificate.require_bound(geometry, mesh=mesh)
    assert certificate.topology_id == mesh.topology_id
    with pytest.raises(ValueError, match="coordinate arrays"):
        certificate.require_bound(moved, mesh=mesh)


@pytest.mark.parametrize(
    "kind",
    (
        "interval",
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
    ),
)
@pytest.mark.parametrize("degree", (2, 3, 4))
def test_source_enclosure_certifies_standard_curved_lattice_maps(
    kind: str, degree: int
) -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element(kind, degree)
    points = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    if points.shape[1] == 1:
        points[:, 0] += 0.03125 * points[:, 0] ** 2
    else:
        points[:, 0] += 0.03125 * points[:, -1] ** 2
    geometry = D.CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )
    certificate = D.certify_cell_geometry_validity(geometry)

    assert certificate.all_certified
    assert float(certificate.determinant_lower[0]) > 0.9
    assert float(certificate.determinant_upper[0]) < 1.1


def test_partition_of_unity_is_not_a_pointwise_source_error_bound() -> None:
    import jax.numpy as jnp
    from jax import Array

    from phydrax.discretization.fem._reference import FiniteElementSpec

    def fabricated_basis(points: Array) -> tuple[Array, Array]:
        values = jnp.stack(
            (1.0 - points[:, 0] - points[:, 1], points[:, 0], points[:, 1]), axis=-1
        )
        # A factor-two error preserves sum(grad N)==0 identically.
        gradients = jnp.broadcast_to(
            jnp.asarray(((-2.0, -2.0), (2.0, 0.0), (0.0, 2.0)), dtype=jnp.float64),
            (points.shape[0], 3, 2),
        )
        return values, gradients

    element = FiniteElementSpec(
        "Lagrange",
        "triangle",
        1,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (((0,), (1,), (2,)), ((), (), ()), ((),)),
        value_spec=D.fem.lagrange_element("triangle", 1).value_spec,
        tabulator=fabricated_basis,
        tabulator_id="deliberately-biased-pointwise-source",
    )
    geometry = D.CellGeometrySpec(
        {"cells": element},
        {"cells": np.asarray(((0, 1, 2),), dtype=np.int32)},
        np.asarray(element.reference_nodes, dtype=np.float64),
    )
    certificate = D.certify_cell_geometry_validity(geometry)

    assert int(certificate.status[0]) == Status.UNRESOLVED
    assert certificate.unresolved_reasons == (("cells", "unsupported_element"),)


def test_source_bernstein_subdivision_finds_interior_fold() -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("interval", 3)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = (reference - 0.5) ** 3 / 3.0 - reference / 16.0
    geometry = D.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(4, dtype=np.int32)[None]}, coordinates
    )
    bounded = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(maximum_subdivision_depth=0)
    )
    resolved = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(maximum_subdivision_depth=1)
    )

    assert int(bounded.status[0]) == Status.UNRESOLVED
    assert bounded.unresolved_reasons == (("cells", "subdivision_depth"),)
    assert int(resolved.status[0]) == Status.INVALID
    assert float(resolved.determinant_lower[0]) < -0.05


def test_affine_prism_source_encloses_analytic_constant_determinant() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.0, 0.1),
            (0.5, 0.0, 0.1),
            (0.5, 0.5, 0.1),
        ),
        dtype=np.float64,
    )
    mesh = D.CellMesh(
        points,
        (
            D.CellBlock(
                "prisms", "prism", np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32)
            ),
        ),
    )
    geometry = D.CellGeometrySpec.affine(mesh)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)

    # Independent affine formula det([(.5,0,0),(.5,.5,0),(0,0,.1)])=.025.
    assert certificate.all_certified
    assert float(certificate.determinant_lower[0]) <= 0.025
    assert float(certificate.determinant_upper[0]) >= 0.025
    assert (
        float(certificate.determinant_upper[0]) - float(certificate.determinant_lower[0])
        < 1.0e-16
    )


def test_validity_binding_rejects_changed_routes_despite_cached_layout_id() -> None:
    import equinox as eqx
    import jax.numpy as jnp

    mesh = _reference_mesh("tetrahedron")
    geometry = D.CellGeometrySpec.affine(mesh)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    changed = eqx.tree_at(
        lambda value: value.geometry_dofs,
        geometry,
        (jnp.asarray(((1, 0, 2, 3),), dtype=jnp.int32),),
    )

    with pytest.raises(ValueError, match="coordinate arrays"):
        certificate.require_bound(changed, mesh=mesh)


def test_validity_binding_rejects_changed_basis_coefficients_without_moving_nodes() -> (
    None
):
    import equinox as eqx
    import jax.numpy as jnp

    from phydrax.discretization.fem._high_order import SimplexNodalFamily

    family = SimplexNodalFamily("tetrahedron", 3)
    element = family.finite_element()
    points = np.asarray(element.reference_nodes, dtype=np.float64)
    geometry = D.CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )
    certificate = D.certify_cell_geometry_validity(geometry)
    modified_family = eqx.tree_at(
        lambda value: value.coefficients, family, jnp.zeros_like(family.coefficients)
    )
    changed = D.CellGeometrySpec(
        {"cells": modified_family.finite_element()},
        {"cells": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )

    assert certificate.all_certified
    with pytest.raises(ValueError, match="coordinate arrays"):
        certificate.require_bound(changed)


@pytest.mark.parametrize("degree", (0, 1), ids=("constant-field", "linear-field"))
def test_portable_discontinuous_sources_preserve_exact_field_integrals(
    degree: int,
) -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import coordinate_polynomials
    from phydrax.discretization.fem._reference import discontinuous_element
    from phydrax.geometry._mapped_coverage import integrate

    element = discontinuous_element("tetrahedron", degree)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coefficients = (
        np.full((1, 1), 2.75, dtype=np.float64)
        if degree == 0
        else (1.0 + 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 4.0 * nodes[:, 2])[:, None]
    )
    expression = coordinate_polynomials(element, coefficients)
    if expression is None:
        pytest.fail("Canonical immutable DG source expression was unavailable.")

    expected = Fraction(11, 24) if degree == 0 else Fraction(7, 24)
    assert integrate(expression[0], "simplex", 3) == expected
