#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import masked_simplex_facet_neighbors, MaskedSimplexMesh


_FACETS = {
    3: ((1, 2), (2, 0), (0, 1)),
    4: ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
}


def _square(count, generator):
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2).reshape((count + 1, count + 1))
    cells = []
    for i, j in itertools.product(range(count), repeat=2):
        a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
        cells.extend(((a, b, c), (a, d, c)))
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += generator.uniform(-0.2, 0.2, points[interior].shape) / count
    return points, np.asarray(cells, dtype=np.int32)


def _cube(count, generator):
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    index = np.arange((count + 1) ** 3).reshape((count + 1,) * 3)
    cells = []
    for corner in itertools.product(range(count), repeat=3):
        for order in itertools.permutations(range(3)):
            vertex = np.asarray(corner)
            path = [index[tuple(vertex)]]
            for axis_index in order:
                vertex = vertex + np.eye(3, dtype=np.int64)[axis_index]
                path.append(index[tuple(vertex)])
            cells.append(path)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += generator.uniform(-0.1, 0.1, points[interior].shape) / count
    return points, np.asarray(cells, dtype=np.int32)


def _compact(points, cells):
    """Compact FV reference; its plan also fixes positive cell orientation."""
    generator = np.random.default_rng(3)
    order = generator.permutation(cells.shape[0])
    key = "triangles" if cells.shape[1] == 3 else "tetrahedra"
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        points,
        **{key: cells[order]},
        vertex_global_ids=np.arange(points.shape[0]),
        cell_global_ids=np.arange(cells.shape[0]),
    )
    oriented = np.asarray(plan.triangles if key == "triangles" else plan.tetrahedra)
    return plan.prepare(), oriented


def _masked_layout(points, cells, vertex_capacity, cell_capacity, seed):
    """Pad a compact mesh into capacity slots with interleaved inactive lanes."""
    generator = np.random.default_rng(seed)
    vertex_slots = np.sort(
        generator.choice(vertex_capacity, points.shape[0], replace=False)
    )
    cell_slots = np.sort(generator.choice(cell_capacity, cells.shape[0], replace=False))
    coordinates = np.full((vertex_capacity, points.shape[1]), np.nan)
    coordinates[vertex_slots] = points
    vertex_ids = np.full((vertex_capacity,), -1, dtype=np.int64)
    vertex_ids[vertex_slots] = np.arange(points.shape[0])
    slot_cells = np.zeros((cell_capacity, cells.shape[1]), dtype=np.int32)
    # Inactive rows mix all-zero padding with rows through unallocated NaN slots.
    unallocated = np.setdiff1d(np.arange(vertex_capacity), vertex_slots)
    slot_cells[1::2] = unallocated[: cells.shape[1]]
    slot_cells[cell_slots] = vertex_slots[cells]
    cell_ids = np.full((cell_capacity,), -1, dtype=np.int64)
    cell_ids[cell_slots] = np.arange(cells.shape[0])
    slot_cells = jnp.asarray(slot_cells)
    cell_active = jnp.asarray(cell_ids >= 0)
    return MaskedSimplexMesh(
        jnp.asarray(coordinates),
        jnp.asarray(vertex_ids),
        jnp.asarray(vertex_ids >= 0),
        slot_cells,
        jnp.asarray(cell_ids),
        cell_active,
        masked_simplex_facet_neighbors(slot_cells, cell_active),
    )


_CASES = {
    "triangle": (lambda generator: _square(4, generator), 37, 45),
    "tetrahedron": (lambda generator: _cube(2, generator), 40, 64),
}


@pytest.fixture(params=sorted(_CASES))
def case(request):
    build, vertex_capacity, cell_capacity = _CASES[request.param]
    points, cells = build(np.random.default_rng(11))
    compact, oriented = _compact(points, cells)
    mesh = _masked_layout(points, oriented, vertex_capacity, cell_capacity, 5)
    geometry = phx.discretization.evaluate_masked_fv_geometry(mesh)
    return compact, mesh, geometry


def _route_keys(mesh, geometry):
    """Sorted vertex global IDs of every half-facet route, shaped (faces, d)."""
    cells = np.asarray(mesh.cells)
    facets = np.asarray(_FACETS[cells.shape[1]])
    vertex_ids = np.asarray(mesh.vertex_ids)
    keys = np.sort(vertex_ids[cells[:, facets]], axis=-1)
    return keys.reshape((geometry.face_capacity, -1))


def _compact_face_index(compact):
    connectivity = compact.connectivity
    faces = np.asarray(
        connectivity.edges if compact.cell_dimension == 2 else connectivity.faces
    )
    vertex_ids = np.asarray(compact.vertex_global_ids)
    return {tuple(np.sort(vertex_ids[face])): index for index, face in enumerate(faces)}


def _route_correspondence(compact, mesh, geometry):
    """Compact face and owner-orientation sign of every active masked route."""
    routes = np.flatnonzero(np.asarray(geometry.face_active))
    lookup = _compact_face_index(compact)
    keys = _route_keys(mesh, geometry)
    faces = np.asarray([lookup[tuple(keys[route])] for route in routes])
    cell_ids = np.asarray(mesh.cell_ids)
    compact_cell_ids = np.asarray(compact.cell_global_ids)
    owner_ids = cell_ids[np.asarray(geometry.owner_cells)[routes]]
    compact_owner_ids = compact_cell_ids[np.asarray(compact.owner_cells)[faces]]
    signs = np.where(owner_ids == compact_owner_ids, 1.0, -1.0)
    return routes, faces, signs


def test_masked_geometry_matches_compact_geometry_on_active_lanes(case):
    compact, mesh, geometry = case
    cell_active = np.asarray(mesh.cell_active)
    cell_slots = np.flatnonzero(cell_active)
    compact_cells = np.asarray(mesh.cell_ids)[cell_slots]
    np.testing.assert_allclose(
        geometry.cell_volumes[cell_slots],
        np.asarray(compact.cell_volumes)[compact_cells],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        geometry.cell_centers[cell_slots],
        np.asarray(compact.cell_centers)[compact_cells],
        rtol=1e-13,
        atol=1e-15,
    )

    routes, faces, signs = _route_correspondence(compact, mesh, geometry)
    assert routes.size == compact.face_measures.size
    assert np.unique(faces).size == faces.size
    np.testing.assert_allclose(
        geometry.area_vectors[routes],
        signs[:, None] * np.asarray(compact.area_vectors)[faces],
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        geometry.face_measures[routes],
        np.asarray(compact.face_measures)[faces],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        geometry.face_centers[routes],
        np.asarray(compact.face_centers)[faces],
        rtol=1e-13,
        atol=1e-15,
    )
    cell_ids = np.asarray(mesh.cell_ids)
    neighbor = np.asarray(geometry.neighbor_cells)[routes]
    compact_owner = np.asarray(compact.cell_global_ids)[
        np.asarray(compact.owner_cells)[faces]
    ]
    compact_neighbor_slot = np.asarray(compact.neighbor_cells)[faces]
    compact_neighbor = np.where(
        compact_neighbor_slot >= 0,
        np.asarray(compact.cell_global_ids)[np.maximum(compact_neighbor_slot, 0)],
        -1,
    )
    masked_pair = np.sort(
        np.stack(
            (
                cell_ids[np.asarray(geometry.owner_cells)[routes]],
                np.where(neighbor >= 0, cell_ids[np.maximum(neighbor, 0)], -1),
            ),
            axis=1,
        ),
        axis=1,
    )
    compact_pair = np.sort(np.stack((compact_owner, compact_neighbor), axis=1), axis=1)
    np.testing.assert_array_equal(masked_pair, compact_pair)
    np.testing.assert_array_equal(
        np.asarray(geometry.boundary_faces)[routes], compact_neighbor < 0
    )


def test_masked_geometry_padding_lanes_are_exact_zero_and_finite(case):
    _, mesh, geometry = case
    cell_inactive = ~np.asarray(mesh.cell_active)
    route_inactive = ~np.asarray(geometry.face_active)
    assert cell_inactive.any() and route_inactive.any()
    for value in (
        geometry.cell_volumes,
        geometry.cell_centers,
        geometry.area_vectors,
        geometry.face_measures,
        geometry.face_centers,
    ):
        assert np.all(np.isfinite(np.asarray(value)))
    assert np.all(np.asarray(geometry.cell_volumes)[cell_inactive] == 0.0)
    assert np.all(np.asarray(geometry.cell_centers)[cell_inactive] == 0.0)
    assert np.all(np.asarray(geometry.area_vectors)[route_inactive] == 0.0)
    assert np.all(np.asarray(geometry.face_measures)[route_inactive] == 0.0)
    assert np.all(np.asarray(geometry.neighbor_cells)[route_inactive] == -1)


def _compact_divergence(compact, face_flux):
    owner = np.asarray(compact.owner_cells)
    neighbor = np.asarray(compact.neighbor_cells)
    content = np.zeros((compact.cell_count,) + face_flux.shape[1:])
    np.add.at(content, owner, -face_flux)
    interior = neighbor >= 0
    np.add.at(content, neighbor[interior], face_flux[interior])
    return content / np.asarray(compact.cell_volumes)[:, None]


def test_masked_divergence_matches_compact_and_ignores_padding(case):
    compact, mesh, geometry = case
    generator = np.random.default_rng(17)
    face_flux = generator.normal(size=(geometry.face_capacity, 2))
    residual = np.asarray(
        phx.discretization.masked_fv_flux_divergence(geometry, face_flux)
    )

    routes, faces, signs = _route_correspondence(compact, mesh, geometry)
    compact_flux = np.zeros((compact.face_measures.size, 2))
    compact_flux[faces] = signs[:, None] * face_flux[routes]
    expected = _compact_divergence(compact, compact_flux)
    cell_slots = np.flatnonzero(np.asarray(mesh.cell_active))
    np.testing.assert_allclose(
        residual[cell_slots],
        expected[np.asarray(mesh.cell_ids)[cell_slots]],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(residual[~np.asarray(mesh.cell_active)] == 0.0)

    inactive = ~np.asarray(geometry.face_active)
    perturbed = face_flux.copy()
    perturbed[inactive] = generator.normal(size=perturbed[inactive].shape) * 1e6
    np.testing.assert_array_equal(
        phx.discretization.masked_fv_flux_divergence(geometry, perturbed), residual
    )


def test_masked_conservation_ledger_balances_boundary_flux(case):
    _, mesh, geometry = case
    generator = np.random.default_rng(23)
    face_flux = generator.normal(size=(geometry.face_capacity, 3))
    cell_active = np.asarray(mesh.cell_active)
    source = np.where(
        cell_active[:, None],
        generator.normal(size=(geometry.cell_capacity, 3)),
        0.0,
    )
    boundary = np.asarray(geometry.boundary_faces)
    boundary_total = np.sum(face_flux[boundary], axis=0)

    evidence = phx.discretization.evaluate_masked_fv_conservation(
        geometry, face_flux, source
    )
    scale = np.sum(np.abs(face_flux)) + np.sum(np.abs(source))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    np.testing.assert_allclose(
        evidence.source_sum, np.sum(source, axis=0), atol=tolerance
    )
    np.testing.assert_allclose(
        evidence.boundary_outward_sum, boundary_total, atol=tolerance
    )
    np.testing.assert_allclose(
        evidence.net_cell_sum, np.sum(source, axis=0) - boundary_total, atol=tolerance
    )
    assert np.all(np.abs(np.asarray(evidence.residual)) <= tolerance)
    content = np.asarray(evidence.ledger.scatter_content_rate())
    assert np.all(content[~cell_active] == 0.0)

    residual = np.asarray(
        phx.discretization.masked_fv_flux_divergence(geometry, face_flux)
    )
    volume_weighted = np.sum(
        np.asarray(geometry.cell_volumes)[:, None] * residual, axis=0
    )
    np.testing.assert_allclose(volume_weighted, -boundary_total, atol=tolerance)

    interior_only = np.where(boundary[:, None], 0.0, face_flux)
    interior_residual = phx.discretization.masked_fv_flux_divergence(
        geometry, interior_only
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(geometry.cell_volumes)[:, None] * interior_residual, axis=0),
        0.0,
        atol=tolerance,
    )
    interior_evidence = phx.discretization.evaluate_masked_fv_conservation(
        geometry, interior_only, np.zeros_like(source)
    )
    np.testing.assert_allclose(interior_evidence.net_cell_sum, 0.0, atol=tolerance)
    np.testing.assert_array_equal(interior_evidence.boundary_outward_sum, 0.0)


def test_masked_conservation_rejects_source_on_inactive_cells(case):
    _, mesh, geometry = case
    source = np.zeros((geometry.cell_capacity, 1))
    source[np.flatnonzero(~np.asarray(mesh.cell_active))[0]] = 1.0
    with pytest.raises(eqx.EquinoxRuntimeError, match="exactly zero on inactive cells"):
        phx.discretization.evaluate_masked_fv_conservation(
            geometry, np.zeros((geometry.face_capacity, 1)), source
        )


def test_masked_geometry_rejects_negatively_oriented_active_cell(case):
    _, mesh, _ = case
    cells = np.asarray(mesh.cells).copy()
    slot = np.flatnonzero(np.asarray(mesh.cell_active))[0]
    cells[slot, [0, 1]] = cells[slot, [1, 0]]
    cells = jnp.asarray(cells)
    flipped = MaskedSimplexMesh(
        mesh.coordinates,
        mesh.vertex_ids,
        mesh.vertex_active,
        cells,
        mesh.cell_ids,
        mesh.cell_active,
        masked_simplex_facet_neighbors(cells, mesh.cell_active),
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="positively oriented|owner-outward"
    ):
        phx.discretization.evaluate_masked_fv_geometry(flipped)
