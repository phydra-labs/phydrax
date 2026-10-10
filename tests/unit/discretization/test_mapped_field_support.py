#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Whole mapped-domain membership, inverse statuses and field evaluation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax import discretization as D
from phydrax.discretization._cell_complex import (
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._mapped_locator import PreparedMappedCellLocator
from phydrax.discretization._simplicial_locator import (
    CellLocationStatus,
    SimplicialLocationPolicy,
)
from phydrax.discretization._view_support import (
    mapped_mesh_support_geometry,
    mapped_mesh_support_query,
    MappedSupportQueryStatus,
    simplicial_mesh_support_geometry,
    verify_mesh_support_geometry,
)
from phydrax.discretization.fem import (
    prepare_finite_element_cell_map,
    prepare_finite_element_field_reconstruction,
)
from phydrax.discretization.fem._generic import FiniteElementDiscretization
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.geometry import GeometryCapability, Orthotope


jax.config.update("jax_enable_x64", True)


def _prepare(
    kind: str,
    element: FiniteElementSpec
    | RestrictedCellGeometryElement
    | SplineCellGeometryElement,
    coordinates: NDArray[np.float64],
    *,
    copies: int = 1,
) -> FiniteElementDiscretization:
    vertices = np.asarray(D.reference_cell_topology(kind).vertices)
    weights, _ = element.tabulate(vertices)
    corners = np.asarray(weights) @ coordinates
    vertex_count = corners.shape[0]
    mesh = D.CellMesh(
        np.tile(corners, (copies, 1)),
        (
            D.CellBlock(
                "cells",
                kind,
                np.arange(copies * vertex_count, dtype=np.int32).reshape(
                    (copies, vertex_count)
                ),
            ),
        ),
    )
    geometry = D.CellGeometrySpec(
        {"cells": element},
        {
            "cells": np.arange(copies * coordinates.shape[0], dtype=np.int32).reshape(
                (copies, coordinates.shape[0])
            )
        },
        np.tile(coordinates, (copies, 1)),
    )
    prepared = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec("u", D.lagrange_element(kind, 1)),
        coordinate_spec=geometry,
    ).prepare()
    if not isinstance(prepared, FiniteElementDiscretization):
        raise TypeError(
            "Mapped support fixtures require a finite-element discretization."
        )
    return prepared


def _locator(
    prepared: FiniteElementDiscretization, policy: SimplicialLocationPolicy | None = None
) -> PreparedMappedCellLocator:
    return PreparedMappedCellLocator(
        prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        SimplicialLocationPolicy(16, 48, 8) if policy is None else policy,
    )


def _image(locator: PreparedMappedCellLocator, reference: ArrayLike) -> Array:
    reference = jnp.asarray(reference)
    return locator.cell_map.evaluate(
        locator.coordinates,
        jnp.zeros((reference.shape[0],), dtype=jnp.int32),
        reference,
    ).physical_points


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_curved_simplex_support_keeps_image_beyond_all_coordinate_nodes(
    kind: str,
) -> None:
    element = coordinate_lagrange_element(kind, 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    coordinates[:, -1] += 0.4 * coordinates[:, 0] * (coordinates[:, 0] - 0.75)
    prepared = _prepare(kind, element, coordinates)
    locator = _locator(prepared)
    reference = np.full((1, locator.dimension), 0.002)
    reference[0, 0] = 0.375
    physical = _image(locator, reference)
    assert float(physical[0, -1]) < float(coordinates[:, -1].min())
    result = eqx.filter_jit(lambda owner, points: owner.locate(points))(locator, physical)
    assert result.status.tolist() == [int(CellLocationStatus.LOCATED)]
    assert result.candidates_complete.tolist() == [True]
    np.testing.assert_allclose(result.reference_coordinates, reference, atol=2e-10)
    assert float(result.geometry_residual[0]) <= locator.policy.residual_tolerance
    support = simplicial_mesh_support_geometry(locator, "curved-support")
    query = eqx.filter_jit(mapped_mesh_support_query)(support, physical)
    assert query.inside.tolist() == [True]
    assert query.successful.tolist() == [True]
    assert not support.has_capability(GeometryCapability.SIGNED_DISTANCE)
    assert not support.has_capability(GeometryCapability.INTERIOR_MEASURE)
    assert not support.field_certificate.is_signed_distance
    with pytest.raises(NotImplementedError):
        support.signed_distance(physical)
    outside = mapped_mesh_support_query(support, jnp.full((1, locator.dimension), 3.0))
    assert outside.status.tolist() == [int(CellLocationStatus.OUTSIDE)]
    assert outside.successful.tolist() == [True]
    assert outside.inside.tolist() == [False]


@pytest.mark.parametrize(
    ("kind", "reference"),
    (
        ("quadrilateral", (0.8, 0.9)),
        ("hexahedron", (0.8, 0.7, 0.9)),
        ("prism", (0.7, 0.2, 0.8)),
        ("pyramid", (0.5, 0.5, 0.8)),
    ),
)
def test_hybrid_and_tensor_queries_use_actual_reference_domain_and_coordinate_map(
    kind: str, reference: tuple[float, ...]
) -> None:
    element = coordinate_lagrange_element(kind, 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    if len(reference) == 2:
        coordinates[:, 0] += 0.25 * coordinates[:, 0] * coordinates[:, 1]
    else:
        coordinates[:, 0] += 0.25 * coordinates[:, 0] * coordinates[:, 2]
    prepared = _prepare(kind, element, coordinates)
    locator = _locator(prepared)
    physical = _image(locator, jnp.asarray((reference,)))
    location = locator.locate(physical)
    assert location.inside.tolist() == [True]
    np.testing.assert_allclose(location.reference_coordinates, (reference,), atol=2e-10)
    assert np.isfinite(np.asarray(location.jacobian_condition)).all()
    support = mapped_mesh_support_geometry(locator, "hybrid-support")
    assert eqx.filter_jit(lambda geometry, points: geometry.contains(points))(
        support, physical
    ).tolist() == [True]
    # A reference-linear field must be evaluated on the inverse of the actual
    # map, not on a tetrahedron made from a few physical corners.
    field = prepared.elements[0][0]
    routes = np.asarray(prepared.dof_maps[0].cell_dofs[0])[0]
    coefficients = np.zeros((prepared.dof_maps[0].global_dof_count,))
    coefficients[routes] = 1.0 + np.asarray(field.reference_nodes) @ np.arange(
        1.0, len(reference) + 1.0
    )
    reconstruction = prepare_finite_element_field_reconstruction(prepared, "u")
    query = reconstruction.prepare_query(physical)
    values = eqx.filter_jit(lambda route, state: route.apply(state))(
        query, jnp.asarray(coefficients)
    )
    np.testing.assert_allclose(
        values,
        (1.0 + np.asarray(reference) @ np.arange(1.0, len(reference) + 1.0),),
        atol=2e-10,
    )
    if kind == "pyramid":
        cube_only = _image(locator, jnp.asarray(((0.05, 0.5, 0.8),)))
        assert locator.locate(cube_only).status.tolist() == [
            int(CellLocationStatus.OUTSIDE)
        ]
    if kind == "prism":
        cube_only = _image(locator, jnp.asarray(((0.8, 0.8, 0.5),)))
        assert locator.locate(cube_only).status.tolist() == [
            int(CellLocationStatus.OUTSIDE)
        ]


def test_exact_restricted_child_keeps_parent_coefficients_and_child_membership() -> None:
    source = coordinate_lagrange_element("triangle", 2)
    coordinates = np.asarray(source.reference_nodes).copy()
    coordinates[:, 1] += 0.4 * coordinates[:, 0] * (coordinates[:, 0] - 0.75)
    child = RestrictedCellGeometryElement(
        source, "triangle", 0.5 * np.eye(2), np.zeros(2)
    )
    prepared = _prepare("triangle", child, coordinates)
    locator = _locator(prepared)
    reference = np.asarray(((0.75, 0.1),))
    source_reference = reference @ np.asarray(child.matrix).T + np.asarray(child.offset)
    expected = source_reference.copy()
    expected[:, 1] += 0.4 * source_reference[:, 0] * (source_reference[:, 0] - 0.75)
    np.testing.assert_allclose(_image(locator, reference), expected, atol=1e-13)
    location = locator.locate(expected)
    np.testing.assert_allclose(location.reference_coordinates, reference, atol=2e-10)
    assert location.status.tolist() == [int(CellLocationStatus.LOCATED)]
    # Inside the source triangle, outside this exact restricted child.
    parent_only = np.asarray(((0.7, 0.05 + 0.4 * 0.7 * (0.7 - 0.75)),))
    assert locator.locate(parent_only).status.tolist() == [
        int(CellLocationStatus.OUTSIDE)
    ]
    support = mapped_mesh_support_geometry(locator, "restricted-child")
    verify_mesh_support_geometry(support, locator, tolerance=0.0)
    assert support.contains(jnp.asarray(expected)).tolist() == [True]
    field = prepared.elements[0][0]
    routes = np.asarray(prepared.dof_maps[0].cell_dofs[0])[0]
    coefficients = np.zeros((prepared.dof_maps[0].global_dof_count,))
    coefficients[routes] = 1.0 + np.asarray(field.reference_nodes) @ (1.0, 2.0)
    reconstruction = prepare_finite_element_field_reconstruction(prepared, "u")
    query = reconstruction.prepare_query(expected)
    np.testing.assert_allclose(query.apply(coefficients), (1.95,), atol=2e-10)
    derivative = reconstruction.prepare_query(expected, derivative=(1, 0))
    # source x=.375 makes the source shear derivative zero; the child
    # horizontal chart scale is .5, so the physical derivative is 2.
    np.testing.assert_allclose(derivative.apply(coefficients), (2.0,), atol=2e-10)


def test_rational_pyramid_child_uses_parent_enclosure_but_actual_restricted_inverse() -> (
    None
):
    source = coordinate_lagrange_element("pyramid", 1)
    coordinates = np.asarray(source.reference_nodes).copy()
    coordinates[:, 0] += 0.2 * coordinates[:, 0] * coordinates[:, 1]
    child = RestrictedCellGeometryElement(
        source, "pyramid", 0.5 * np.eye(3), np.asarray((0.25, 0.25, 0.0))
    )
    locator = _locator(_prepare("pyramid", child, coordinates))
    reference = np.asarray(((0.7, 0.6, 0.2),))
    parent = 0.5 * reference + (0.25, 0.25, 0.0)
    expected = parent.copy()
    expected[:, 0] += (
        0.2
        * (parent[:, 0] - 0.5 * parent[:, 2])
        * (parent[:, 1] - 0.5 * parent[:, 2])
        / (1.0 - parent[:, 2])
    )
    # The perturbed apex is also an authoritative source coefficient: its
    # contribution is 0.2 * 0.5 * 0.5 times the apex weight z.
    expected[:, 0] += 0.05 * parent[:, 2]
    np.testing.assert_allclose(_image(locator, reference), expected, atol=2e-12)
    location = locator.locate(expected)
    assert location.status.tolist() == [int(CellLocationStatus.LOCATED)]
    np.testing.assert_allclose(location.reference_coordinates, reference, atol=2e-10)
    support = mapped_mesh_support_geometry(locator, "rational-child")
    assert support.contains(jnp.asarray(expected)).tolist() == [True]
    parent_only = np.asarray(((0.9, 0.05, 0.05),))
    parent_weights, _ = source.tabulate(parent_only)
    parent_physical = np.asarray(parent_weights) @ coordinates
    assert mapped_mesh_support_query(
        support, jnp.asarray(parent_physical)
    ).status.tolist() == [int(CellLocationStatus.OUTSIDE)]


def test_bounded_inverse_failure_singular_source_and_nonfinite_are_not_outside() -> None:
    element = coordinate_lagrange_element("hexahedron", 1)
    coordinates = np.asarray(element.reference_nodes).copy()
    prepared = _prepare("hexahedron", element, coordinates)
    exhausted = _locator(prepared, SimplicialLocationPolicy(1, 1, 1, trust_radius=1e-6))
    point = jnp.asarray(((0.85, 0.8, 0.9),))
    result = exhausted.locate(point)
    assert result.status.tolist() == [int(CellLocationStatus.INVERSE_MAP_EXHAUSTED)]
    assert result.candidates_complete.tolist() == [False]
    assert np.isfinite(np.asarray(result.geometry_residual)).all()
    assert float(result.geometry_residual[0]) > exhausted.policy.residual_tolerance
    support = mapped_mesh_support_geometry(exhausted, "bounded-failure")
    query = mapped_mesh_support_query(support, point)
    assert query.successful.tolist() == [False]
    assert np.isnan(np.asarray(support.boundary_field(point))).all()
    collapsed = coordinates.copy()
    collapsed[:, 0] = 0.0
    singular = PreparedMappedCellLocator(
        exhausted.cell_map, collapsed, SimplicialLocationPolicy(1, 8, 1)
    )
    assert singular.locate(jnp.asarray(((0.0, 0.5, 0.5),))).status.tolist() == [
        int(CellLocationStatus.DEGENERATE_CELL)
    ]
    assert exhausted.locate(jnp.asarray(((np.nan, 0.5, 0.5),))).status.tolist() == [
        int(CellLocationStatus.NONFINITE)
    ]


def test_candidate_capacity_is_observable_and_all_containing_cells_are_retained() -> None:
    element = coordinate_lagrange_element("triangle", 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    prepared = _prepare("triangle", element, coordinates, copies=2)
    point = np.asarray(((0.2, 0.3),))
    small = _locator(prepared, SimplicialLocationPolicy(1, 16, 1))
    incomplete = small.locate(point)
    assert incomplete.status.tolist() == [int(CellLocationStatus.RESOURCE_EXCEEDED)]
    assert incomplete.successful.tolist() == [False]
    assert incomplete.candidates_complete.tolist() == [False]
    complete = _locator(prepared, SimplicialLocationPolicy(2, 16, 1)).locate(point)
    assert complete.status.tolist() == [int(CellLocationStatus.LOCATED)]
    assert complete.candidate_count.tolist() == [2]
    assert set(np.asarray(complete.candidate_cells)[0]) == {0, 1}
    assert complete.candidates_complete.tolist() == [True]
    masked = _locator(prepared, SimplicialLocationPolicy(2, 16, 1)).locate(
        point, cell_mask=np.asarray((False, True))
    )
    assert masked.cell_ids.tolist() == [1]
    assert masked.candidate_count.tolist() == [1]


def test_explicit_support_requires_whole_source_binding_and_rejects_stale_boxes() -> None:
    element = coordinate_lagrange_element("hexahedron", 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    coordinates[:, 0] += 0.2 * coordinates[:, 2] ** 2
    locator = _locator(_prepare("hexahedron", element, coordinates))
    support = mapped_mesh_support_geometry(locator, "bound-source")
    verify_mesh_support_geometry(support, locator, tolerance=0.0)
    moved = coordinates.copy()
    moved[:, 0] += 0.1
    revised = PreparedMappedCellLocator(locator.cell_map, moved, locator.policy)
    with pytest.raises(ValueError, match="source revision"):
        verify_mesh_support_geometry(support, revised, tolerance=0.0)
    stale = eqx.tree_at(
        lambda geometry: geometry.state, support, support.state.replace_at(0, moved)
    )
    query = eqx.filter_jit(mapped_mesh_support_query)(
        stale, _image(locator, jnp.asarray(((0.2, 0.3, 0.4),)))
    )
    assert query.status.tolist() == [
        int(MappedSupportQueryStatus.SOURCE_REVISION_MISMATCH)
    ]
    assert query.successful.tolist() == [False]
    assert np.isnan(np.asarray(stale.bounds)).all()
    with pytest.raises(ValueError, match="whole mapped source"):
        verify_mesh_support_geometry(
            Orthotope((0.6, 0.5, 0.5), (1.2, 1.0, 1.0)).compile(),
            locator,
            tolerance=1e-10,
        )


def test_source_enclosure_capacity_and_unknown_tabulator_refuse_preparation() -> None:
    element = coordinate_lagrange_element("triangle", 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    locator = _locator(_prepare("triangle", element, coordinates))
    with pytest.raises(ValueError, match="source-bound coefficient capacity"):
        PreparedMappedCellLocator(
            locator.cell_map, coordinates, locator.policy, maximum_bound_coefficients=1
        )

    def unknown(points: ArrayLike) -> tuple[Array, Array]:
        return element.tabulate(points)

    unsupported = FiniteElementSpec(
        element.family,
        element.cell_kind,
        element.degree,
        element.reference_nodes,
        element.entity_dofs,
        value_spec=element.value_spec,
        tabulator=unknown,
        tabulator_id="opaque-coordinate-source",
    )
    opaque = eqx.tree_at(
        lambda cell_map: cell_map.coordinate_element, locator.cell_map, unsupported
    )
    with pytest.raises(ValueError, match="source is unsupported"):
        PreparedMappedCellLocator(opaque, coordinates, locator.policy)


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron"))
def test_boundary_atlas_keeps_curved_source_image_jacobian_and_outward_orientation(
    kind: str,
) -> None:
    element = coordinate_lagrange_element(kind, 2)
    coordinates = np.asarray(element.reference_nodes).copy()
    coordinates[:, -1] += 0.4 * coordinates[:, 0] * (coordinates[:, 0] - 0.75)
    locator = _locator(_prepare(kind, element, coordinates))
    geometry = mapped_mesh_support_geometry(locator, "source-boundary")
    assert geometry.has_capability(GeometryCapability.BOUNDARY_ATLAS)
    atlas = geometry.boundary_atlas
    mesh = locator.cell_map.mesh
    ids = np.asarray(mesh.entity_set(locator.dimension - 1).entity_ids)
    connectivity = mesh.connectivity
    if kind == "triangle":
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("Triangle boundary fixtures require polygonal connectivity.")
        facets = np.asarray(connectivity.edges)
        slot = int(np.flatnonzero(np.all(np.sort(facets, axis=1) == (0, 1), axis=1))[0])
        parameters = jnp.asarray(((0.375,),))
        expected = np.asarray(((0.375, -0.05625),))
        normal = np.asarray(((0.0, -1.0),))
        density = (1.0,)
    else:
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError(
                "Tetrahedron boundary fixtures require tetrahedral connectivity."
            )
        facets = np.asarray(connectivity.faces)
        slot = int(
            np.flatnonzero(np.all(np.sort(facets, axis=1) == (0, 1, 2), axis=1))[0]
        )
        parameters = jnp.asarray(((0.25, 0.5),))
        expected = np.asarray(((0.375, 0.25, -0.05625),))
        normal = np.asarray(((0.0, 0.0, -1.0),))
        density = (0.75,)
    chart = jnp.asarray(np.flatnonzero(np.asarray(atlas.source_entity_ids) == ids[slot]))
    assert set(np.asarray(atlas.source_entity_ids)) == set(ids)
    mapped = eqx.filter_jit(
        lambda source, index, reference: source.map(index, reference)
    )(
        atlas,
        jnp.asarray(chart),
        jnp.asarray(parameters),
    )
    np.testing.assert_allclose(mapped, expected, atol=2e-12)
    np.testing.assert_allclose(atlas.jacobian(chart, parameters), density, atol=2e-12)
    frame = atlas.frame(chart, parameters)
    assert frame.regular.tolist() == [True]
    assert frame.jacobian_consistent.tolist() == [True]
    np.testing.assert_allclose(frame.normal, normal, atol=2e-12)


def test_restricted_boundary_atlas_evaluates_full_parent_source_and_child_chain() -> None:
    source = coordinate_lagrange_element("triangle", 2)
    coordinates = np.asarray(source.reference_nodes).copy()
    coordinates[:, 1] += 0.4 * coordinates[:, 0] * (coordinates[:, 0] - 0.75)
    child = RestrictedCellGeometryElement(
        source, "triangle", 0.5 * np.eye(2), np.zeros(2)
    )
    locator = _locator(_prepare("triangle", child, coordinates))
    atlas = mapped_mesh_support_geometry(locator, "restricted-boundary").boundary_atlas
    mesh = locator.cell_map.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Triangle boundary fixtures require polygonal connectivity.")
    facets = np.asarray(connectivity.edges)
    slot = int(np.flatnonzero(np.all(np.sort(facets, axis=1) == (0, 1), axis=1))[0])
    entity_id = int(mesh.entity_set(1).entity_ids[slot])
    chart = jnp.asarray(np.flatnonzero(np.asarray(atlas.source_entity_ids) == entity_id))
    parameters = jnp.asarray(((0.75,),))
    np.testing.assert_allclose(
        atlas.map(chart, parameters), ((0.375, -0.05625),), atol=2e-12
    )
    np.testing.assert_allclose(atlas.jacobian(chart, parameters), (0.5,), atol=2e-12)
    np.testing.assert_allclose(
        atlas.frame(chart, parameters).normal, ((0.0, -1.0),), atol=2e-12
    )


def test_atlas_omits_only_exact_matching_shared_traces_not_curved_coordinate_gaps() -> (
    None
):
    element = coordinate_lagrange_element("triangle", 2)
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    mesh = D.CellMesh(vertices, (D.CellBlock("cells", "triangle", cells),))
    reference = np.asarray(element.reference_nodes)
    local = []
    for cell in cells:
        corners = vertices[cell]
        points = corners[0] + reference @ (corners[1:] - corners[0])
        points[:, 1] += 0.1 * points[:, 0] * (1.0 - points[:, 0])
        local.append(points)
    coordinates = np.concatenate(local)
    prepared = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1)),
        coordinate_spec=D.CellGeometrySpec(
            {"cells": element},
            {"cells": np.arange(12, dtype=np.int32).reshape((2, 6))},
            coordinates,
        ),
    ).prepare()
    locator = _locator(prepared)
    geometry = mapped_mesh_support_geometry(locator, "continuous-union")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Triangle boundary fixtures require polygonal connectivity.")
    boundary_ids = np.asarray(mesh.entity_set(1).entity_ids)[
        np.asarray(connectivity.boundary_edges)
    ]
    assert set(np.asarray(geometry.boundary_atlas.source_entity_ids)) == set(boundary_ids)
    assert geometry.contains(jnp.asarray(((0.5, 1.01),))).tolist() == [True]
    # The mapped diagonal is a shared interior seam, not the solid boundary.
    points = jnp.asarray(((0.5, 0.525), (0.5, 1.025), (0.5, 1.1)))
    support_query = mapped_mesh_support_query(geometry, points)
    assert support_query.successful.tolist() == [True, True, True]
    assert support_query.inside.tolist() == [True, True, False]
    assert support_query.location.candidate_count[0] == 2
    field = eqx.filter_jit(lambda source, sites: source.boundary_field(sites))(
        geometry, points
    )
    assert float(field[0]) < -0.25
    np.testing.assert_allclose(field[1], 0.0, atol=2e-10)
    assert float(field[2]) > 0.0
    # Same topological/corner mesh, but the complete shared diagonal sources
    # disagree at their midpoint. Erasing that face would lose real support.
    moved = coordinates.copy()
    midpoint = int(np.flatnonzero(np.all(reference == (0.5, 0.0), axis=1))[0])
    moved[6 + midpoint, 1] += 0.1
    gap = PreparedMappedCellLocator(locator.cell_map, moved, locator.policy)
    with pytest.raises(
        ValueError, match="nonmatching full coordinate-source facet traces"
    ):
        mapped_mesh_support_geometry(gap, "source-gap")


def test_rational_spline_locator_and_field_preserve_original_control_map() -> None:
    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    ).reshape((-1, 3))
    weight = np.sqrt(0.5)
    element = SplineCellGeometryElement(
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
        np.tile(np.asarray([1.0, weight, 1.0], dtype=np.float64), (2, 1)),
        1,
        2,
        (1, 2),
        "original-rational-cylinder",
        "accepted-source",
    )
    prepared = _prepare("quadrilateral", element, controls)
    locator = _locator(prepared)
    reference = np.asarray([[0.2, 0.25], [0.7, 0.6]], dtype=np.float64)
    v = reference[:, 1]
    cross = 2 * weight * v * (1 - v)
    denominator = (1 - v) ** 2 + cross + v**2
    points = jnp.asarray(
        np.stack(
            (
                reference[:, 0],
                ((1 - v) ** 2 + cross) / denominator,
                (cross + v**2) / denominator,
            ),
            axis=1,
        )
    )
    location = locator.locate(points)
    np.testing.assert_array_equal(location.status, [int(CellLocationStatus.LOCATED)] * 2)
    np.testing.assert_allclose(
        location.reference_coordinates, reference, rtol=0.0, atol=2e-10
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(points[:, 1:]) ** 2, axis=1), 1.0, rtol=0.0, atol=2e-15
    )
    reconstruction = prepare_finite_element_field_reconstruction(
        prepared, "u", locator=locator
    )
    query = reconstruction.prepare_query(points)
    values = query.apply(prepared.dof_maps[0].dof_coordinates[:, 0])
    np.testing.assert_allclose(values, reference[:, 0], rtol=0.0, atol=2e-10)
