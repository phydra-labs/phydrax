from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_power_geometry import ExactPowerCellGeometrySource
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.meshing import (
    MeshingEntityKind,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePeriodicSource,
    PeriodicPointOrbits,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._periodic import periodic_entity_orbit_map
from phydrax.meshing.providers._native_periodic_power import periodic_power_mesh


def test_periodic_power_constructs_source_bound_polyhedral_quotient_cells() -> None:
    vectors = np.asarray(
        ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
        dtype=np.float64,
    )
    cell = PeriodicCell(vectors)
    fractional = np.asarray(
        (
            (0.125, 0.1875, 0.25),
            (0.625, 0.125, 0.125),
            (0.1875, 0.625, 0.3125),
            (0.3125, 0.3125, 0.75),
            (0.75, 0.8125, 0.5625),
            (0.875, 0.4375, 0.9375),
        ),
        dtype=np.float64,
    )
    source_points = PeriodicPointOrbits(
        cell,
        np.asarray(cell.cartesian(fractional), dtype=np.float64),
        np.asarray((91, 4, 53, 17, 62, 31), dtype=np.int64),
    )
    source_id = "periodic-power-sites"
    source_revision = "periodic-power-sites-revision-1"
    authored_weights = np.asarray(
        (0.0, 0.015, -0.01, 0.02, 0.005, -0.015), dtype=np.float64
    )
    source = NativePeriodicSource(
        source_points,
        source_id,
        source_revision,
        site_weights=authored_weights,
    )
    np.testing.assert_array_equal(source.site_weights, authored_weights)
    assert source.site_weights is not None and not source.site_weights.flags.writeable
    scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("polyhedron",)),
        ),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )

    construction = NativeMeshingProvider(
        NativeMeshingOptions("periodic_power")
    ).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()
    mesh = construction.mesh

    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    unweighted = periodic_power_mesh(source_points).mesh
    assert not np.array_equal(mesh.coordinates, unweighted.coordinates)
    topology = mesh.periodic_topology
    assert topology is not None
    geometry = construction.geometry
    assert geometry is not None
    source_map = geometry.source_map
    assert isinstance(source_map, ExactPowerCellGeometrySource)
    assert topology.coordinate_source_map_id == source_map.source_map_id
    source_map.require_periodic_lift(
        mesh.coordinates,
        cell,
        topology.vertex_representatives,
        topology.vertex_shifts,
    )
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    assert np.any(np.asarray(connectivity.boundary_faces, dtype=np.bool_))
    assert not np.any(
        np.asarray(topology.quotient.entity_sets[2].subset("boundary").mask, dtype=np.bool_)
    )
    assert all(block.cell_kind == "polyhedron" for block in mesh.blocks)
    quotient_cells = np.asarray(topology.quotient.entity_sets[3].entity_ids, dtype=np.int64)
    assert np.array_equal(
        np.sort(quotient_cells), np.sort(np.asarray(source_points.point_ids, dtype=np.int64))
    )
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_cells = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    begin, end = map(int, lifted_offsets[3:5])
    quotient_begin, quotient_end = map(int, quotient_offsets[3:5])
    cell_orbits = np.asarray(topology.orbit_indices, dtype=np.int64)[begin:end] - quotient_begin
    assert lifted_cells.size == source_points.point_ids.size
    assert np.unique(cell_orbits).size == quotient_end - quotient_begin
    assert np.all(np.bincount(cell_orbits, minlength=quotient_end - quotient_begin) == 1)

    cell_map = periodic_entity_orbit_map(mesh, 3)
    cell_corner_counts = np.diff(
        np.asarray(cell_map.corner_permutation_offsets, dtype=np.int64)
    )
    assert np.array_equal(
        cell_corner_counts,
        np.diff(np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)),
    )
    assert np.unique(cell_corner_counts).size > 1
    face_map = periodic_entity_orbit_map(mesh, 2)
    face_corner_counts = np.diff(
        np.asarray(face_map.corner_permutation_offsets, dtype=np.int64)
    )
    assert np.array_equal(
        face_corner_counts,
        np.diff(np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)),
    )
    vertex_map = periodic_entity_orbit_map(mesh, 0)
    assert np.any(np.asarray(vertex_map.winding_shifts, dtype=np.int64))
    assert construction.attributes == ()
    assert construction.zones == ()
    assert construction.compliance.issues == ()


def _polyhedral_cell_volumes(mesh: CellMesh, /) -> dict[int, float]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Cell-volume oracle requires polyhedral connectivity.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    volumes = {}
    for cell, cell_id in enumerate(cell_ids):
        signed_volume = 0.0
        start, stop = map(int, cell_offsets[cell : cell + 2])
        for face, sign in zip(
            cell_faces[start:stop], cell_signs[start:stop], strict=True
        ):
            face_start, face_stop = map(
                int, face_offsets[int(face) : int(face) + 2]
            )
            points = coordinates[face_vertices[face_start:face_stop]]
            for index in range(1, points.shape[0] - 1):
                signed_volume += float(
                    sign
                    * np.dot(points[0], np.cross(points[index], points[index + 1]))
                    / 6.0
                )
        volumes[int(cell_id)] = abs(signed_volume)
    return volumes


def test_periodic_power_matches_exact_unequal_weight_cube_oracle() -> None:
    cell = PeriodicCell(np.eye(3, dtype=np.float64))
    point_ids = np.asarray((101, 202), dtype=np.int64)
    points = PeriodicPointOrbits(
        cell,
        np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), dtype=np.float64),
        point_ids,
    )

    mesh = periodic_power_mesh(
        points, np.asarray((1.0 / 16.0, 0.0), dtype=np.float64)
    ).mesh
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    topology = mesh.periodic_topology
    assert topology is not None
    assert all(block.cell_kind == "polyhedron" for block in mesh.blocks)
    assert np.array_equal(
        np.sort(np.asarray(topology.quotient.entity_sets[3].entity_ids, dtype=np.int64)),
        np.sort(point_ids),
    )
    assert not np.any(
        np.asarray(
            topology.quotient.entity_sets[2].subset("boundary").mask,
            dtype=np.bool_,
        )
    )
    assert np.any(np.asarray(connectivity.boundary_faces, dtype=np.bool_))

    x_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)[:, 0]
    assert np.any(
        np.isclose(x_coordinates, -1.0 / 16.0, rtol=0.0, atol=1.0e-14)
    )
    assert np.any(np.isclose(x_coordinates, 9.0 / 16.0, rtol=0.0, atol=1.0e-14))
    volumes = _polyhedral_cell_volumes(mesh)
    assert set(volumes) == {101, 202}
    np.testing.assert_allclose(
        (volumes[101], volumes[202]),
        (5.0 / 8.0, 3.0 / 8.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        sum(volumes.values()), 1.0, rtol=0.0, atol=1.0e-12
    )


def test_periodic_power_source_identity_is_stable_under_site_row_permutation() -> None:
    from phydrax.discretization._exact_power_geometry import (
        ExactPeriodicPowerSourceContext,
    )

    cell = PeriodicCell(np.eye(3, dtype=np.float64))
    point_ids = np.asarray((41, 7, 93, 18), dtype=np.int64)
    points = np.asarray(
        (
            (0.125, 0.25, 0.375),
            (0.375, 0.625, 0.125),
            (0.625, 0.125, 0.75),
            (0.875, 0.75, 0.5),
        ),
        dtype=np.float64,
    )
    weights = np.asarray((0.125, -0.25, 0.0625, -0.0), dtype=np.float64)
    owners = np.asarray((1, 0, 1, 0), dtype=np.int32)
    permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
    orbits = PeriodicPointOrbits(cell, points, point_ids)
    source = NativePeriodicSource(
        orbits,
        "row-order-periodic-power",
        "row-order-periodic-power-revision-1",
        site_weights=weights,
        point_owner_ranks=owners,
    )
    permuted_orbits = PeriodicPointOrbits(
        cell,
        points[permutation],
        point_ids[permutation],
    )
    permuted_source = NativePeriodicSource(
        permuted_orbits,
        "row-order-periodic-power",
        "row-order-periodic-power-revision-1",
        site_weights=weights[permutation],
        point_owner_ranks=owners[permutation],
    )
    np.testing.assert_array_equal(
        np.sort(np.asarray(source.scope_entity_ids(0), dtype=np.int64)),
        np.sort(np.asarray(permuted_source.scope_entity_ids(0), dtype=np.int64)),
    )
    assert source.binding_id == permuted_source.binding_id

    image_owners = np.arange(point_ids.size, dtype=np.int64)
    image_windings = np.zeros((point_ids.size, 3), dtype=np.int64)
    context = ExactPeriodicPowerSourceContext(
        orbits,
        weights,
        image_owners,
        image_windings,
        weights_present=True,
    )
    np.testing.assert_array_equal(np.asarray(context.source_weights), weights)
    permuted_context = ExactPeriodicPowerSourceContext(
        permuted_orbits,
        weights[permutation],
        image_owners,
        image_windings,
        weights_present=True,
    )
    np.testing.assert_array_equal(
        np.asarray(permuted_context.source_weights), weights[permutation]
    )
    assert context.context_id == permuted_context.context_id
    positive_zero_weights = weights.copy()
    positive_zero_weights[-1] = 0.0
    positive_zero_context = ExactPeriodicPowerSourceContext(
        orbits,
        positive_zero_weights,
        image_owners,
        image_windings,
        weights_present=True,
    )
    assert context.context_id != positive_zero_context.context_id
    zero_weights = np.zeros_like(weights)
    absent_weight_context = ExactPeriodicPowerSourceContext(
        orbits,
        zero_weights,
        image_owners,
        image_windings,
        weights_present=False,
    )
    explicit_zero_weight_context = ExactPeriodicPowerSourceContext(
        orbits,
        zero_weights,
        image_owners,
        image_windings,
        weights_present=True,
    )
    assert not absent_weight_context.weights_present
    assert explicit_zero_weight_context.weights_present
    assert absent_weight_context.context_id != explicit_zero_weight_context.context_id
    renewed = ExactPeriodicPowerSourceContext.from_stored_fields(context)
    assert renewed.context_id == context.context_id


def test_owner_local_site_weight_support_preserves_signed_zero_bits() -> None:
    import jax
    import pytest

    from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
    from phydrax.meshing.providers._native_periodic_power import (
        _owner_local_periodic_site_weights_for_support,
    )

    if jax.process_count() != 1:
        pytest.skip("This source-weight identity contract uses a one-owner group.")
    process = jax.process_index()
    device = next(
        device for device in jax.devices() if device.process_index == process
    )
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "periodic-site-weight-bits",
            (process,),
            ((process, device.id),),
            mesh_axes=(("site_weights", 1),),
        ),
        (device,),
    )
    point_ids = np.asarray((4, 9, 17), dtype=np.int64)
    points = PeriodicPointOrbits(
        PeriodicCell(np.eye(3, dtype=np.float64)),
        np.asarray(
            ((0.125, 0.25, 0.375), (0.625, 0.75, 0.875), (0.1875, 0.3125, 0.4375)),
            dtype=np.float64,
        ),
        point_ids,
    )
    weights = np.asarray((-0.0, 0.0, -0.25), dtype=np.float64)
    source = NativePeriodicSource(
        points,
        "signed-zero-periodic-sites",
        "signed-zero-periodic-sites-revision-1",
        site_weights=weights,
        point_owner_ranks=np.zeros((point_ids.size,), dtype=np.int32),
    )
    support_ids = np.asarray((17, 4, 9), dtype=np.int64)
    support_owners = np.zeros(support_ids.shape, dtype=np.int32)
    support_weights = _owner_local_periodic_site_weights_for_support(
        source,
        support_ids,
        support_owners,
        0,
        group,
        axis_name="signed_zero_site_weight_support",
    )
    assert support_weights is not None
    np.testing.assert_array_equal(
        np.asarray(support_weights).view(np.uint64),
        weights[[2, 0, 1]].view(np.uint64),
    )
