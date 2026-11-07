from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

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

    construction = (
        NativeMeshingProvider(NativeMeshingOptions("periodic_power"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
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
    assert isinstance(topology.exact_coordinate_source, ExactPowerCellGeometrySource)
    topology.require_exact_geometry(mesh, geometry)
    restored = CellMesh.from_stored_fields(mesh, actual_geometry=geometry)
    assert restored.periodic_topology is not None
    assert isinstance(
        restored.periodic_topology.exact_coordinate_source, ExactPowerCellGeometrySource
    )
    assert (
        restored.periodic_topology.exact_coordinate_source.source_map_id
        == source_map.source_map_id
    )
    # A lifted vertex-row permutation changes only the output selection, not
    # the original power sites, carrier, cell metadata, or deck action.
    from phydrax.meshing._topology_edit import _reorder_periodic_coordinate_source

    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    permutation = np.arange(vertex_ids.size - 1, -1, -1, dtype=np.int64)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(permutation.size, dtype=np.int64)
    reordered = _reorder_periodic_coordinate_source(
        source_map, vertex_ids, vertex_ids[permutation], permutation
    )
    assert isinstance(reordered, ExactPowerCellGeometrySource)
    assert reordered.source_map_id != source_map.source_map_id
    assert reordered.source_coordinates() == tuple(
        source_map.source_coordinates()[int(row)] for row in permutation
    )
    np.testing.assert_array_equal(reordered.site_points, source_map.site_points)
    np.testing.assert_array_equal(reordered.site_weights, source_map.site_weights)
    np.testing.assert_array_equal(reordered.cell_sites, source_map.cell_sites)
    np.testing.assert_array_equal(reordered.cell_regions, source_map.cell_regions)
    np.testing.assert_array_equal(
        reordered.output_vertex_indices,
        np.asarray(source_map.output_vertex_indices)[permutation],
    )
    assert reordered.carrier_mesh_id == source_map.carrier_mesh_id
    assert reordered.periodic_source_context is not None
    assert source_map.periodic_source_context is not None
    assert (
        reordered.periodic_source_context.context_id
        == source_map.periodic_source_context.context_id
    )
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    assert isinstance(topology.cell, PeriodicCell)
    reordered.require_periodic_lift(
        np.asarray(mesh.coordinates)[permutation],
        topology.cell,
        inverse_permutation[representatives[permutation]],
        shifts[permutation],
    )
    renewed_reordered = ExactPowerCellGeometrySource.from_stored_fields(
        reordered, target_periodic_cell=cell
    )
    assert renewed_reordered.source_map_id == reordered.source_map_id
    assert renewed_reordered.source_coordinates() == reordered.source_coordinates()
    # A forged power carrier must not create a recursive witness/PyTree graph.
    from copy import copy

    from phydrax.discretization._periodic_topology import (
        _require_acyclic_periodic_witness,
    )

    cyclic_source = copy(topology.exact_coordinate_source)
    cyclic_carrier = copy(cyclic_source.carrier_mesh)
    cyclic_topology = copy(topology)
    object.__setattr__(cyclic_source, "carrier_mesh", cyclic_carrier)
    object.__setattr__(cyclic_carrier, "periodic_topology", cyclic_topology)
    object.__setattr__(cyclic_topology, "exact_coordinate_source", cyclic_source)
    with pytest.raises(ValueError, match="carrier cycle"):
        _require_acyclic_periodic_witness(cyclic_source)
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
        np.asarray(
            topology.quotient.entity_sets[2].subset("boundary").mask, dtype=np.bool_
        )
    )
    assert all(block.cell_kind == "polyhedron" for block in mesh.blocks)
    quotient_cells = np.asarray(
        topology.quotient.entity_sets[3].entity_ids, dtype=np.int64
    )
    assert np.array_equal(
        np.sort(quotient_cells),
        np.sort(np.asarray(source_points.point_ids, dtype=np.int64)),
    )
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_cells = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    begin, end = map(int, lifted_offsets[3:5])
    quotient_begin, quotient_end = map(int, quotient_offsets[3:5])
    cell_orbits = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[begin:end] - quotient_begin
    )
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
            face_start, face_stop = map(int, face_offsets[int(face) : int(face) + 2])
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
    assert np.any(np.isclose(x_coordinates, -1.0 / 16.0, rtol=0.0, atol=1.0e-14))
    assert np.any(np.isclose(x_coordinates, 9.0 / 16.0, rtol=0.0, atol=1.0e-14))
    volumes = _polyhedral_cell_volumes(mesh)
    assert set(volumes) == {101, 202}
    np.testing.assert_allclose(
        (volumes[101], volumes[202]),
        (5.0 / 8.0, 3.0 / 8.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(sum(volumes.values()), 1.0, rtol=0.0, atol=1.0e-12)


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
    device = next(device for device in jax.devices() if device.process_index == process)
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
    with pytest.raises(ValueError, match="lacks its authored site weight"):
        _owner_local_periodic_site_weights_for_support(
            source,
            np.asarray((999,), dtype=np.int64),
            np.asarray((0,), dtype=np.int32),
            0,
            group,
            axis_name="missing_site_weight_support",
        )


@pytest.mark.parametrize(
    "corruption", ("neighbor", "owner", "action", "face", "orientation", "missing")
)
def test_periodic_power_face_evidence_rejects_mismatch(corruption: str) -> None:
    import jax

    from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
    from phydrax.meshing.providers._native_periodic import (
        _owner_local_require_periodic_face_coverage,
    )

    if jax.process_count() != 1:
        pytest.skip("This evidence refusal test uses a one-owner group.")
    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "power-face-evidence",
            (process,),
            ((process, device.id),),
            mesh_axes=(("power_face_evidence", 1),),
        ),
        (device,),
    )
    key = ((1, (0, 0, 0)), (2, (0, 0, 0)), (3, (0, 0, 0)))
    records = ((key, 1, 10, 0), (key, -1, 20, 0))
    first = (key, 1, 10, 20, 0, (1, 0, 0), 0)
    second = (key, -1, 20, 10, 0, (-1, 0, 0), 0)
    if corruption == "neighbor":
        first = (key, 1, 10, 99, 0, (1, 0, 0), 0)
    elif corruption == "owner":
        first = (key, 1, 10, 20, 1, (1, 0, 0), 0)
    elif corruption == "action":
        first = (key, 1, 10, 20, 0, (2, 0, 0), 0)
    elif corruption == "face":
        other_key = ((1, (0, 0, 0)), (2, (0, 0, 0)), (4, (0, 0, 0)))
        first = (other_key, 1, 10, 20, 0, (1, 0, 0), 0)
    elif corruption == "orientation":
        first = (key, -1, 10, 20, 0, (1, 0, 0), 0)
    evidence = (second,) if corruption == "missing" else (first, second)
    with pytest.raises(ValueError, match="Native periodic"):
        _owner_local_require_periodic_face_coverage(
            records,
            np.asarray((10, 20), dtype=np.int64),
            3,
            0,
            group,
            "power-evidence",
            "r1",
            "content",
            "cell",
            evidence,
            np.zeros(3, dtype=np.int64),
            "power_evidence",
        )


def test_owner_local_power_context_is_authored_over_the_support_bank() -> None:
    import jax
    import pytest

    from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
    from phydrax.meshing.providers._native_periodic_power import (
        _owner_local_periodic_power_candidate,
    )

    if jax.process_count() != 1:
        pytest.skip("This support-bank context contract uses a one-owner group.")
    process = jax.process_index()
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "periodic-power-support-context",
            (process,),
            ((process, device.id),),
            mesh_axes=(("periodic_power_support", 1),),
        ),
        (device,),
    )
    cell = PeriodicCell(
        np.asarray(
            ((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)),
            dtype=np.float64,
        )
    )
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
    # Authored rows are not in ID order and the weights are unequal, so a
    # context bound to the authored rows instead of the sorted support bank
    # permutes its image weights.
    point_ids = np.asarray((91, 4, 53, 17, 62, 31), dtype=np.int64)
    weights = np.asarray((0.0, 0.015, -0.01, 0.02, 0.005, -0.015), dtype=np.float64)
    source = NativePeriodicSource(
        PeriodicPointOrbits(
            cell,
            np.asarray(cell.cartesian(fractional), dtype=np.float64),
            point_ids,
        ),
        "periodic-power-support-context",
        "periodic-power-support-context-revision-1",
        site_weights=weights,
        point_owner_ranks=np.zeros((point_ids.size,), dtype=np.int32),
    )
    power = _owner_local_periodic_power_candidate(
        source, 0, group, "periodic-power-support-context-manifest"
    )
    source_map = power["source_map"]
    assert isinstance(source_map, ExactPowerCellGeometrySource)
    mesh = power["mesh"]
    assert mesh is not None
    assert source_map.runtime_vertices.shape[0] == mesh.coordinates.shape[0]
    np.testing.assert_array_equal(
        np.asarray(source_map.output_vertex_indices),
        np.arange(mesh.coordinates.shape[0]),
    )
    assert source_map.cell_sites.shape[0] == mesh.entity_set(3).entity_ids.shape[0]
    np.testing.assert_array_equal(source_map.rounded_vertices, mesh.coordinates)
    context = source_map.periodic_source_context
    assert context is not None
    assert context.periodic_source is power["source"].domain
    order = np.argsort(point_ids)
    np.testing.assert_array_equal(
        np.asarray(context.periodic_source.point_ids), point_ids[order]
    )
    np.testing.assert_array_equal(np.asarray(context.source_weights), weights[order])
    owners = np.asarray(context.image_owners, dtype=np.int64)
    assert context.exact_image_weights == tuple(
        Fraction.from_float(float(value)) for value in weights[order][owners]
    )


def test_power_image_neighbor_csr_matches_exact_dyadic_pair_oracle() -> None:
    from phydrax.meshing.providers._native_periodic_power import (
        _float_expansion,
        _image_radius_neighbor_csr,
    )

    tiny = Fraction(1, 2**53)
    points = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0), Fraction(0)),
        (Fraction(2) + tiny, Fraction(0), Fraction(0)),
        (-Fraction(2) + tiny, Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    )
    offsets = [0]
    components: list[float] = []
    for point in points:
        for coordinate in point:
            components.extend(_float_expansion(coordinate))
            offsets.append(len(components))
    csr_offsets, neighbors = _image_radius_neighbor_csr(
        np.arange(len(points), dtype=np.int32),
        np.zeros((len(points), 3), dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(components, dtype=np.float64),
        1.0,
    )
    expected = tuple(
        tuple(
            second
            for second, other in enumerate(points)
            if second != first
            and sum((point[axis] - other[axis]) ** 2 for axis in range(3)) <= 4
        )
        for first, point in enumerate(points)
    )
    actual = tuple(
        tuple(int(value) for value in neighbors[csr_offsets[row] : csr_offsets[row + 1]])
        for row in range(len(points))
    )
    assert actual == expected
    assert 1 in actual[0] and 2 not in actual[0]


@pytest.mark.parametrize("magnitude", (0, 420))
def test_power_image_neighbor_csr_decides_boundary_band_exactly(
    magnitude: int,
) -> None:
    from phydrax.meshing.providers._native_periodic_power import (
        _float_expansion,
        _image_radius_neighbor_csr,
    )

    # Integer grid points sit exactly on the diameter, while two-component
    # expansions displace copies by far less than a binary64 coordinate ulp.
    # ``magnitude`` 420 exceeds the certified binary64 range, so every pair is
    # then decided by the exact integer predicate alone.
    scale = Fraction(2) ** magnitude
    rng = np.random.default_rng(1729)
    tiny = Fraction(1, 2**80)
    points: list[tuple[Fraction, Fraction, Fraction]] = [
        (Fraction(x), Fraction(y), Fraction(z))
        for x in range(5)
        for y in range(3)
        for z in range(3)
    ]
    for shift in (tiny, -tiny):
        points.extend((Fraction(2) + shift, Fraction(y), Fraction(0)) for y in range(3))
        points.append((Fraction(0), Fraction(0), Fraction(2) + shift))
    points.extend(
        (Fraction(int(x), 64), Fraction(int(y), 64), Fraction(int(z), 64))
        for x, y, z in rng.integers(-64, 5 * 64, size=(160, 3))
    )
    points = [(scale * x, scale * y, scale * z) for x, y, z in points]
    offsets = [0]
    components: list[float] = []
    for point in points:
        for coordinate in point:
            components.extend(_float_expansion(coordinate))
            offsets.append(len(components))
    assert max(np.diff(offsets)) == 2
    csr_offsets, neighbors = _image_radius_neighbor_csr(
        np.arange(len(points), dtype=np.int32),
        np.zeros((len(points), 3), dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(components, dtype=np.float64),
        float(scale),
    )
    limit = 4 * scale**2
    expected = tuple(
        tuple(
            second
            for second, other in enumerate(points)
            if second != first
            and sum((point[axis] - other[axis]) ** 2 for axis in range(3)) <= limit
        )
        for first, point in enumerate(points)
    )
    actual = tuple(
        tuple(int(value) for value in neighbors[csr_offsets[row] : csr_offsets[row + 1]])
        for row in range(len(points))
    )
    assert csr_offsets.dtype == np.int64 and neighbors.dtype == np.int32
    assert actual == expected
    # Origin row: (2, 0, 0) lies exactly on the diameter, the +tiny copies at
    # rows 45 and 48 lie outside, and the -tiny copies at rows 49 and 52 inside.
    assert 18 in actual[0]
    assert 45 not in actual[0] and 49 in actual[0]
    assert 48 not in actual[0] and 52 in actual[0]


@pytest.mark.parametrize("case", ("skew", "ties", "dominated"))
def test_owner_power_covering_image_bound_matches_larger_exact_reference(
    case: str,
) -> None:
    import jax

    import phydrax._meshcore as meshcore
    from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
    from phydrax.meshing.providers._native_periodic_power import (
        _box_tetrahedra,
        _exact_image_coordinates,
        _exact_lattice_inverse,
        _image_radius_neighbor_csr,
        _image_row,
        _lattice_covering_radius,
        _outward_bound,
        _outward_scalar,
        _owner_local_periodic_power_candidate,
        _polyhedral_quotient_from_power,
        _power_images,
        _upper_sqrt,
    )

    if jax.process_count() != 1:
        pytest.skip("Exact reference compares one-owner image banks.")
    process = int(jax.process_index())
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            f"power-image-reference-{case}",
            (process,),
            ((process, device.id),),
            mesh_axes=(("power_reference", 1),),
        ),
        (device,),
    )
    if case == "skew":
        lattice = np.asarray(((1.0, 0.125, 0.0), (0.0, 1.0, 0.25), (0.125, 0.0, 1.0)))
        fractional = np.asarray(
            (
                (0.125, 0.1875, 0.25),
                (0.625, 0.125, 0.125),
                (0.1875, 0.625, 0.3125),
                (0.3125, 0.3125, 0.75),
                (0.75, 0.8125, 0.5625),
                (0.875, 0.4375, 0.9375),
            )
        )
        weights = np.asarray((0.0, 0.015, -0.01, 0.02, 0.005, -0.015))
    else:
        lattice = np.eye(3)
        fractional = np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
        weights = np.asarray((0.5, 0.0) if case == "dominated" else (0.0, 0.0))
    cell = PeriodicCell(lattice)
    point_ids = np.arange(fractional.shape[0], dtype=np.int64) + 100
    source = NativePeriodicSource(
        PeriodicPointOrbits(cell, np.asarray(cell.cartesian(fractional)), point_ids),
        f"power-image-reference-{case}",
        "power-image-reference-r1",
        site_weights=weights,
        point_owner_ranks=np.zeros(point_ids.shape, dtype=np.int32),
    )
    power = _owner_local_periodic_power_candidate(
        source, 0, group, f"power-image-reference-{case}-manifest"
    )
    mesh = power["mesh"]
    assert mesh is not None
    source_map = power["source_map"]
    assert isinstance(source_map, ExactPowerCellGeometrySource)
    source_map.require_periodic_mesh(mesh)
    assert isinstance(source.domain, PeriodicPointOrbits)
    points = np.asarray(source.domain.points, dtype=np.float64)
    radius = _lattice_covering_radius(lattice)
    old_radius = _upper_sqrt(
        Fraction.from_float(radius) ** 2
        + Fraction.from_float(float(np.max(weights)))
        - Fraction.from_float(float(np.min(weights)))
    )
    reference_radius = _outward_scalar(1.125 * old_radius)
    margin = power["candidate"]["certified_periodic_margin"]
    carrier_lower = _outward_bound(np.min(points, axis=0) - margin, -np.inf)
    carrier_upper = _outward_bound(np.max(points, axis=0) + margin, np.inf)
    image_owners, image_windings = _power_images(
        points,
        point_ids,
        _exact_lattice_inverse(lattice),
        carrier_lower,
        carrier_upper,
        reference_radius,
        int(np.argmax(weights)),
    )
    offsets, components = _exact_image_coordinates(
        points, lattice, image_owners, image_windings
    )
    neighbor_offsets, neighbors = _image_radius_neighbor_csr(
        image_owners, image_windings, offsets, components, reference_radius
    )
    carrier_points, tets, facets = _box_tetrahedra(carrier_lower, carrier_upper)
    restricted = meshcore.restricted_power_cells_exact(
        points,
        weights,
        image_owners,
        offsets,
        components,
        neighbor_offsets,
        neighbors,
        np.zeros(image_owners.size, dtype=np.int8),
        carrier_points,
        tets,
        np.zeros(tets.shape[0], dtype=np.int32),
        facets,
    )
    cell_sites = np.asarray(restricted.cell_sites, dtype=np.int64)
    cell_rows = {}
    for row in range(point_ids.size):
        image = _image_row(image_owners, image_windings, row)
        cells = np.flatnonzero(cell_sites == image)
        assert cells.size <= 1
        if cells.size:
            cell_rows[row] = int(cells[0])
    reference, exact_vertices, *_ = _polyhedral_quotient_from_power(
        source.domain,
        source.source_id,
        source.source_revision,
        point_ids,
        np.zeros(point_ids.shape, dtype=np.int32),
        weights,
        image_owners,
        image_windings,
        offsets,
        components,
        carrier_points,
        restricted,
        cell_rows,
    )
    if case == "dominated":
        assert set(np.asarray(mesh.entity_set(3).entity_ids)) == {100}
    assert set(np.asarray(mesh.entity_set(3).entity_ids)) == set(
        np.asarray(reference.entity_set(3).entity_ids)
    )
    actual_exact = dict(
        zip(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            power["exact_vertex_bank"],
            strict=True,
        )
    )
    reference_exact = dict(
        zip(
            np.asarray(reference.vertex_global_ids, dtype=np.int64),
            exact_vertices,
            strict=True,
        )
    )
    assert actual_exact == reference_exact
    # The native execution shadow must be the correctly rounded binary64 of
    # the exact scientific point, including the sign of exact zero.
    for tested_mesh, exact in (
        (mesh, power["exact_vertex_bank"]),
        (reference, exact_vertices),
    ):
        expected = np.asarray(
            [[float(coordinate) for coordinate in point] for point in exact],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            np.asarray(tested_mesh.coordinates, dtype=np.float64).view(np.uint64),
            expected.view(np.uint64),
        )

    def incidence(value: CellMesh) -> dict[int, set[tuple[int, ...]]]:
        connectivity = value.connectivity
        assert isinstance(connectivity, PolyhedralConnectivity)
        ids = np.asarray(value.entity_set(3).entity_ids, dtype=np.int64)
        vertices = np.asarray(value.vertex_global_ids, dtype=np.int64)
        fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        fv = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        return {
            int(identifier): {
                tuple(sorted(int(vertices[v]) for v in fv[fo[face] : fo[face + 1]]))
                for face in cf[co[row] : co[row + 1]]
            }
            for row, identifier in enumerate(ids)
        }

    assert incidence(mesh) == incidence(reference)
