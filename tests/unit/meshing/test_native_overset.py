#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._distributed_field import DistributedHaloPlan
from phydrax.discretization._mapped_locator import PreparedMappedCellLocator
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._polyhedral_locator import PreparedPolyhedralCellLocator
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.discretization.finite_volume import (
    PiecewiseConstantReconstruction,
    prepare_finite_volume_field_reconstruction,
    UnstructuredFiniteVolumePlan,
)
from phydrax.lifecycle import Composition
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._overset import (
    OversetCellStatus,
    OversetConnectivityError,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
    prepare_overset_conservative_remap,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)
from phydrax.meshing._periodic import PeriodicPointOrbits
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_periodic import (
    NativePeriodicSource,
    prepare_native_periodic,
)
from tests._support.volume_material_map_fixtures import make_curved_material_refinement


def test_owner_local_finite_volume_reconstruction_factory_is_exported() -> None:
    name = "prepare_owner_local_finite_volume_field_reconstruction"
    assert name in phx.discretization.__all__
    assert name in phx.discretization.finite_volume.__all__
    assert getattr(phx.discretization, name) is getattr(
        phx.discretization.finite_volume, name
    )


def _part(name: str, points: Any, triangles: Any) -> Any:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int32)
    )
    return MeshPart(
        name,
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
    )


def _boundary_facets(part: Any, dimension: int = 1) -> Any:
    mesh = part.carrier.mesh
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        identifiers = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        boundary = np.asarray(mesh.connectivity.boundary_faces, dtype=np.bool_)
        return part.scope(2, identifiers[boundary])
    identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    boundary = np.asarray(mesh.connectivity.boundary_facets, dtype=np.bool_)
    return part.scope(dimension, identifiers[boundary])


def _native_pair() -> tuple[Any, Any, Any]:
    donor = _part(
        "donor",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
        ((0, 1, 2),),
    )
    receptor = _part(
        "receptor",
        ((0.2, 0.2), (0.4, 0.2), (0.2, 0.4)),
        ((0, 1, 2),),
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("donor"),
            OversetPartSpec("receptor", boundary=_boundary_facets(receptor)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    return donor, receptor, connectivity


def _poly_part(name: str, lower: Any, upper: Any) -> Any:
    unit = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    coordinates = lower_ + unit * (upper_ - lower_)
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    mesh = phx.discretization.CellMesh.from_polyhedra(coordinates, (faces,))
    return MeshPart(
        name,
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
    )


def test_mapped_locator_contains_whole_curved_source_map() -> None:
    def curve(xi: Any, eta: Any) -> Any:
        return np.stack((xi, eta + xi**2 - 0.4 * xi), axis=-1)

    reference_nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    )
    coordinates = curve(reference_nodes[:, 0], reference_nodes[:, 1])
    mesh = phx.discretization.CellMesh.from_triangles(
        coordinates[:3], np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": phx.discretization.lagrange_element("triangle", 2)},
        {"triangles": np.arange(6, dtype=np.int64)[None, :]},
        coordinates,
    )
    locator = PreparedMappedCellLocator.from_cell_geometry(mesh, geometry, 0)
    reference = np.asarray(((0.2, 0.01),), dtype=np.float64)
    point = curve(reference[:, 0], reference[:, 1])
    assert point[0, 1] < np.min(coordinates[:, 1])

    located = locator.locate(point)

    assert np.asarray(located.candidates_complete).tolist() == [True]
    assert np.asarray(located.status).tolist() == [
        int(phx.discretization.CellLocationStatus.LOCATED)
    ]
    assert np.asarray(located.candidate_cells)[0, 0] == 0
    np.testing.assert_allclose(
        np.asarray(located.candidate_reference)[0, 0], reference[0], atol=1e-10
    )


def test_mapped_locator_reports_complete_multiple_inverse_roots_as_ambiguous() -> None:
    def folded_curve(xi: Any, eta: Any) -> Any:
        return np.stack((xi, eta**2 - 0.9 * eta + 0.25), axis=-1)

    reference_nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    )
    coordinates = folded_curve(reference_nodes[:, 0], reference_nodes[:, 1])
    mesh = phx.discretization.CellMesh.from_triangles(
        coordinates[:3], np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": phx.discretization.lagrange_element("triangle", 2)},
        {"triangles": np.arange(6, dtype=np.int64)[None, :]},
        coordinates,
    )
    locator = PreparedMappedCellLocator.from_cell_geometry(mesh, geometry, 0)
    point = folded_curve(np.asarray((0.1,)), np.asarray((0.3,)))

    result = locator.locate(point)

    assert not result.successful.tolist()[0]
    assert result.candidates_complete.tolist() == [True]
    assert np.asarray(result.status).tolist() == [
        int(phx.discretization.CellLocationStatus.AMBIGUOUS)
    ]
    assert np.asarray(result.candidate_count).tolist() == [2]
    np.testing.assert_allclose(
        np.asarray(result.candidate_reference)[0, :2],
        ((0.1, 0.3), (0.1, 0.6)),
        atol=1.0e-8,
    )


def test_mapped_locator_keeps_tangent_root_finitely_unresolved() -> None:
    def folded_curve(xi: Any, eta: Any) -> Any:
        return np.stack((xi, eta**2 - 0.9 * eta + 0.25), axis=-1)

    reference_nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    )
    coordinates = folded_curve(reference_nodes[:, 0], reference_nodes[:, 1])
    mesh = phx.discretization.CellMesh.from_triangles(
        coordinates[:3], np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": phx.discretization.lagrange_element("triangle", 2)},
        {"triangles": np.arange(6, dtype=np.int64)[None, :]},
        coordinates,
    )
    locator = PreparedMappedCellLocator.from_cell_geometry(mesh, geometry, 0)
    point = folded_curve(np.asarray((0.2,)), np.asarray((0.45,)))
    result = locator.locate(point)

    assert not result.successful.tolist()[0]
    assert not result.candidates_complete.tolist()[0]
    assert np.asarray(result.status).tolist() == [
        int(phx.discretization.CellLocationStatus.INVERSE_MAP_EXHAUSTED)
    ]


def test_polyhedral_locator_uses_authored_planar_faces() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    mesh = phx.discretization.CellMesh.from_polyhedra(coordinates, (faces,))
    locator = PreparedPolyhedralCellLocator(mesh)

    located = locator.locate(np.asarray(((0.5, 0.5, 0.5), (0.0, 0.5, 0.5))))
    outside = locator.locate(np.asarray(((1.5, 0.5, 0.5),)))

    assert located.candidates_complete.tolist() == [True, True]
    assert located.status.tolist() == [
        int(phx.discretization.CellLocationStatus.LOCATED),
        int(phx.discretization.CellLocationStatus.LOCATED),
    ]
    assert located.candidate_cells.tolist() == [[0], [0]]
    assert outside.status.tolist() == [int(phx.discretization.CellLocationStatus.OUTSIDE)]
    assert outside.candidates_complete.tolist() == [True]


def test_polyhedral_locator_classifies_nonconvex_authored_cell() -> None:
    polygon = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ),
        dtype=np.float64,
    )
    coordinates = np.concatenate(
        (
            np.column_stack((polygon, np.zeros((polygon.shape[0],)))),
            np.column_stack((polygon, np.ones((polygon.shape[0],)))),
        ),
        axis=0,
    )
    face_loops = [
        tuple(reversed(range(polygon.shape[0]))),
        tuple(range(polygon.shape[0], 2 * polygon.shape[0])),
    ]
    face_loops.extend(
        (
            index,
            (index + 1) % polygon.shape[0],
            (index + 1) % polygon.shape[0] + polygon.shape[0],
            index + polygon.shape[0],
        )
        for index in range(polygon.shape[0])
    )
    mesh = phx.discretization.CellMesh.from_polyhedra(coordinates, (tuple(face_loops),))
    located = PreparedPolyhedralCellLocator(mesh).locate(
        np.asarray(
            (
                (0.5, 1.5, 0.5),
                (1.5, 0.5, 0.5),
                (1.5, 1.5, 0.5),
                (1.0, 1.5, 0.5),
            )
        )
    )

    assert located.candidates_complete.tolist() == [True, True, True, True]
    assert located.status.tolist() == [
        int(phx.discretization.CellLocationStatus.LOCATED),
        int(phx.discretization.CellLocationStatus.LOCATED),
        int(phx.discretization.CellLocationStatus.OUTSIDE),
        int(phx.discretization.CellLocationStatus.LOCATED),
    ]


def test_polyhedral_overset_transfer_uses_native_owner_geometry() -> None:
    donor = _poly_part("poly-donor", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    receptor = _poly_part("poly-receptor", (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor)),
        (
            OversetPartSpec("poly-donor"),
            OversetPartSpec("poly-receptor", boundary=_boundary_facets(receptor, 2)),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    discretization = UnstructuredFiniteVolumePlan.from_cell_mesh(
        donor.carrier.mesh
    ).prepare()
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, PiecewiseConstantReconstruction()
    )
    transfer = prepare_overset_field_transfer(
        connectivity, {"poly-donor": reconstruction}, "state"
    )
    values = jnp.asarray(((6.25,),))

    assert connectivity.complete
    assert connectivity.donor_search_complete
    assert all(route.query.complete for route in transfer.routes)
    output = transfer.apply({"poly-donor": values})["poly-receptor"]
    np.testing.assert_allclose(output, 6.25)


def test_distributed_halo_plan_keeps_search_only_partitions_empty() -> None:
    plan = DistributedHaloPlan(
        np.asarray((0, 2, 0, 2), dtype=np.int32),
        np.asarray(((0, 1), (2, 3)), dtype=np.int32),
        3,
    )

    assert plan.part_count == 3
    np.testing.assert_array_equal(np.asarray(plan.local_global_ids)[0], (0, 2, 1, 3))
    np.testing.assert_array_equal(np.asarray(plan.local_global_ids)[2], (1, 3, 0, 2))
    assert not np.any(np.asarray(plan.local_valid)[1])
    assert not np.any(np.asarray(plan.local_owned)[1])
    assert not np.any(np.asarray(plan.phase_send_valid)[:, 1])
    assert not np.any(np.asarray(plan.phase_receive_valid)[:, 1])
    values = jnp.asarray((2.0, 3.0, 5.0, 7.0))
    np.testing.assert_array_equal(plan.unpack_owned(plan.pack_owned(values)), values)
    with pytest.raises(ValueError, match="ownership"):
        DistributedHaloPlan(
            np.asarray((0, 2, 0, 2), dtype=np.int32),
            np.asarray(((0, 1), (2, 3)), dtype=np.int32),
            0,
        )


def test_owner_local_exchange_supports_search_only_owner_and_syncs_failure() -> None:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or tuple(device.process_index for device in devices) != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip("This regression needs three owner processes with one device each.")
    donor, receptor, _ = _native_pair()
    search_only = _part(
        "search-only",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
        ((0, 1, 2),),
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((donor, receptor, search_only)),
        (
            OversetPartSpec("donor", priority=1, owner_rank=0),
            OversetPartSpec(
                "receptor", boundary=_boundary_facets(receptor), owner_rank=1
            ),
            OversetPartSpec("search-only", priority=0, owner_rank=2),
        ),
        policy=OversetPolicy(fringe_layers=1, donor_priority="part-priority"),
    )
    discretization = phx.discretization.FiniteElementPlan(
        donor.carrier.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    transfer = prepare_overset_field_transfer(
        connectivity, {"donor": reconstruction}, "u"
    )
    field = jnp.asarray(
        1.0
        + np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 0]
        + 2.0 * np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 1]
    )
    stable_ids = np.arange(discretization.dof_maps[0].global_dof_count, dtype=np.int64)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-overset-search-only-owner-test",
            (0, 1, 2),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("overset", 3),),
        ),
        devices,
    )
    mismatched_ids = stable_ids.copy()
    if jax.process_index() == 2:
        mismatched_ids[0] += 100
    with pytest.raises(ValueError, match="metadata differ"):
        transfer.prepare_owner_local_exchange({"donor": mismatched_ids}, group)

    exchange = transfer.prepare_owner_local_exchange({"donor": stable_ids}, group)
    assert not np.any(np.asarray(exchange.halo.local_valid)[2])
    assert not np.any(np.asarray(exchange.halo.local_owned)[2])
    assert not np.any(np.asarray(exchange.halo.phase_send_valid)[:, 2])
    assert not np.any(np.asarray(exchange.halo.phase_receive_valid)[:, 2])
    local_donors = {"donor": field} if jax.process_index() == 0 else {}
    output = exchange.apply(local_donors)
    if jax.process_index() == 1:
        expected = np.asarray(receptor.carrier.mesh.coordinates)
        np.testing.assert_allclose(
            output["receptor"], 1.0 + expected[:, 0] + 2.0 * expected[:, 1]
        )
    else:
        assert not output

    local_duals = (
        {"receptor": jnp.ones((receptor.carrier.mesh.coordinates.shape[0],))}
        if jax.process_index() == 1
        else {}
    )
    transposed = exchange.transpose(local_duals)
    if jax.process_index() == 0:
        assert set(transposed) == {"donor"}
    else:
        assert not transposed


def _owner_local_periodic_assembly(
    name: str,
    fractional_points: np.ndarray,
    point_owners: np.ndarray,
    execution_group: ExecutionGroup,
) -> MeshAssembly:
    revision = f"{name}-revision-1"
    cell = PeriodicCell(np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64))
    point_ids = np.asarray((91, 4, 53, 17), dtype=np.int64)
    local_rows = point_owners == jax.process_index()
    source = NativePeriodicSource(
        PeriodicPointOrbits(
            cell,
            np.asarray(cell.cartesian(fractional_points[local_rows]), dtype=np.float64),
            point_ids[local_rows],
        ),
        name,
        revision,
        point_owner_ranks=point_owners[local_rows],
    )
    domain_scope = MeshingScope(
        name,
        revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        domain_scope,
        size_controls=(UniformSizeControl(domain_scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    return prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute(execution_group=execution_group)


def test_native_owner_local_overset_routes_accepted_remote_fields_and_transpose() -> None:
    """Exercise real three-process owner shards, not a forced-device rank simulation."""
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    group_ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or group_ranks != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip(
            "This smoke requires three native process owners, one device per process."
        )
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-owner-local-overset-public-smoke",
            group_ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("overset", 3),),
        ),
        devices,
    )
    partition = group_ranks.index(jax.process_index())
    donor_name, shadow_name, receptor_name = (
        "owner-donor",
        "owner-donor-shadow",
        "owner-receptor",
    )
    donor_source = _owner_local_periodic_assembly(
        donor_name,
        np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77))),
        np.zeros((4,), dtype=np.int32),
        group,
    )
    shadow_source = _owner_local_periodic_assembly(
        shadow_name,
        np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77))),
        np.zeros((4,), dtype=np.int32),
        group,
    )
    receptor_source = _owner_local_periodic_assembly(
        receptor_name,
        np.asarray(((0.19, 0.21), (0.63, 0.11), (0.23, 0.70), (0.77, 0.83))),
        np.asarray((1, 1, 0, 0), dtype=np.int32),
        group,
    )
    local_parts = tuple(
        part
        for source_assembly in (donor_source, shadow_source, receptor_source)
        for part in source_assembly.parts
    )
    manifests = tuple(
        manifest
        for source_assembly in (donor_source, shadow_source, receptor_source)
        for manifest in source_assembly.part_manifest
    )
    assembly = MeshAssembly(
        local_parts,
        global_parts=manifests,
        local_rank=partition,
    )
    parts_by_name = {part.name: part for part in assembly.parts}
    views = {name: part.carrier for name, part in parts_by_name.items()}
    empty_rank_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    int(
                        partition == 2
                        and not any(
                            np.asarray(part.carrier.cell_ids).size
                            or np.asarray(part.carrier.vertex_ids).size
                            for part in local_parts
                        )
                    ),
                ),
                dtype=np.int32,
            )
        )
    ).reshape((-1,))
    assert empty_rank_flags[2] == 1

    receptor_view = views.get(receptor_name)
    source_halo_valid = True
    remote_source_count = 0
    if receptor_view is not None:
        expected_source_owners = {91: 1, 4: 1, 53: 0, 17: 0}
        source_ids = np.asarray(receptor_view.source_point_ids, dtype=np.int64)
        source_owners = np.asarray(receptor_view.source_owner_ranks, dtype=np.int32)
        source_owned = np.asarray(receptor_view.source_owned, dtype=np.bool_)
        source_halo_valid = all(
            expected_source_owners.get(int(identifier)) == int(owner)
            and bool(owned) == (int(owner) == partition)
            for identifier, owner, owned in zip(
                source_ids, source_owners, source_owned, strict=True
            )
        )
        # Adapted vertices are their own quotient keys; their exact source-site
        # roots are the active rows of the cumulative coordinate ancestry.
        vertex_keys = np.asarray(receptor_view.vertex_source_ids, dtype=np.int64)
        stencil = receptor_view.source_coordinate_stencil
        roots = (
            vertex_keys
            if stencil is None
            else np.asarray(stencil.source_global_ids, dtype=np.int64)[
                np.asarray(stencil.valid, dtype=np.bool_)
            ]
        )
        source_halo_valid = source_halo_valid and bool(
            np.all(np.isin(roots, source_ids))
            and np.all(
                np.isin(
                    np.asarray(receptor_view.cell_key_source_ids, dtype=np.int64),
                    np.union1d(vertex_keys, source_ids),
                )
            )
        )
        remote_source_count = np.count_nonzero(source_owners != partition)
    source_halo_evidence = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(source_halo_valid), remote_source_count), dtype=np.int64)
        )
    ).reshape((3, 2))
    assert np.all(source_halo_evidence[:, 0] == 1)
    assert source_halo_evidence[1, 1] > 0
    boundary = None
    boundary_facets: list[int] = []
    if partition == 1 and receptor_view is not None:
        facet_dimension = receptor_view.topological_dimension - 1
        _, facet_ids = receptor_view.entity_binding(facet_dimension)
        vertex_ids = np.asarray(receptor_view.vertex_ids, dtype=np.int64)
        vertex_owners = np.asarray(receptor_view.vertex_owner_ranks, dtype=np.int32)
        owner_by_vertex = {
            int(identifier): int(owner)
            for identifier, owner in zip(vertex_ids, vertex_owners, strict=True)
        }
        for identifier, loop in zip(
            facet_ids,
            receptor_view.entity_vertex_loops(facet_dimension, facet_ids),
            strict=True,
        ):
            facet_vertices = vertex_ids[np.asarray(loop, dtype=np.int64)]
            if any(owner_by_vertex[int(vertex)] == 1 for vertex in facet_vertices):
                boundary_facets.append(int(identifier))
        if boundary_facets:
            boundary = parts_by_name[receptor_name].scope(
                facet_dimension, np.asarray(boundary_facets[:1], dtype=np.int64)
            )
    boundary_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((len(boundary_facets),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert boundary_counts[1] > 0
    role_specs = (
        OversetPartSpec(donor_name, priority=2, owner_rank=0),
        OversetPartSpec(shadow_name, priority=1, owner_rank=0),
        OversetPartSpec(receptor_name, boundary=boundary, owner_rank=1),
    )
    candidate = prepare_overset_connectivity(
        assembly,
        role_specs,
        policy=OversetPolicy(fringe_layers=1, donor_priority="part-priority"),
        execution_group=group,
    )
    assert candidate.owner_local
    assert not candidate.donor_search_complete
    assert not candidate.complete
    assert candidate.receptors_of(receptor_name).receptor_ids.size > 0
    with pytest.raises(
        OversetConnectivityError, match="donor coverage has not been proved"
    ) as unproved:
        candidate.require_complete()
    assert unproved.value.connectivity is candidate

    local_reconstructions = {}
    coefficient_ids = {}
    local_donor_layout_valid = True
    for name, view in views.items():
        if name != donor_name:
            continue
        blocks = [block for block in view.blocks if np.asarray(block.cell_ids).size]
        if not blocks:
            continue
        if len(blocks) != 1:
            local_donor_layout_valid = False
            continue
        block = blocks[0]
        local_reconstructions[name] = (
            phx.discretization.prepare_owner_local_finite_volume_field_reconstruction(
                view, block.name, "u"
            )
        )
        coefficient_ids[name] = np.asarray(block.cell_ids, dtype=np.int64)
    block_layout_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(not local_donor_layout_valid),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert not np.any(block_layout_flags)
    stale_pair_available = partition != 0 or (
        donor_name in views
        and receptor_name in views
        and any(np.asarray(block.cell_ids).size for block in views[receptor_name].blocks)
    )
    stale_pair_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(stale_pair_available),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert stale_pair_flags[0] == 1

    stale_binding_refused = True
    if partition == 0:
        stale_view = views[receptor_name]
        stale_blocks = [
            block for block in stale_view.blocks if np.asarray(block.cell_ids).size
        ]
        stale_binding_refused = False
        if stale_blocks:
            stale = (
                phx.discretization.prepare_owner_local_finite_volume_field_reconstruction(
                    stale_view, stale_blocks[0].name, "u"
                )
            )
            try:
                prepare_overset_field_transfer(candidate, {donor_name: stale}, "u")
            except ValueError as error:
                stale_binding_refused = "stale source/map/context identity" in str(error)
    stale_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(stale_binding_refused),), dtype=np.int32)
        )
    ).reshape((-1,))
    assert stale_flags[0] == 1

    transfer = prepare_overset_field_transfer(candidate, local_reconstructions, "u")
    exchange = transfer.prepare_owner_local_exchange(coefficient_ids, group, views)
    resolved = exchange.connectivity
    receptor_evidence = resolved.receptors_of(receptor_name)
    assert resolved.donor_search_complete
    assert resolved.complete
    assert resolved.orphan_count == 0
    assert np.all(np.asarray(receptor_evidence.found))
    assert np.all(np.asarray(receptor_evidence.candidate_parts) == 2)
    assert np.all(
        np.asarray(receptor_evidence.donor_parts) == resolved.part_index(donor_name)
    )
    digest = np.frombuffer(bytes.fromhex(resolved.connectivity_id), dtype=np.uint8)
    gathered_digests = np.asarray(multihost_utils.process_allgather(digest))
    assert np.all(gathered_digests == gathered_digests[0])

    def field_value(ids: np.ndarray) -> np.ndarray:
        return 1.0 + (np.asarray(ids, dtype=np.int64) % 17) * 0.125

    local_donors = {}
    if donor_name in local_reconstructions:
        donor_view = views[donor_name]
        id_to_owner = dict(
            zip(
                np.asarray(donor_view.cell_ids, dtype=np.int64).tolist(),
                np.asarray(donor_view.cell_owner_ranks, dtype=np.int32).tolist(),
                strict=True,
            )
        )
        ids = coefficient_ids[donor_name]
        owned = np.asarray(
            [id_to_owner[int(identifier)] == partition for identifier in ids],
            dtype=np.bool_,
        )
        if np.any(owned):
            local_donors[donor_name] = jnp.asarray(field_value(ids[owned]))
    output = exchange.apply(local_donors)
    duals = {
        name: jnp.asarray(np.linspace(0.25, 1.25, num=shape[0]))
        for name, shape, _ in exchange.receptor_layouts
    }
    transposed = exchange.transpose(duals)
    local_lhs = sum(
        float(np.vdot(np.asarray(output[name]), np.asarray(duals[name])))
        for name in output
    )
    local_rhs = sum(
        float(np.vdot(np.asarray(local_donors[name]), np.asarray(transposed[name])))
        for name in transposed
    )
    lhs = float(
        np.sum(np.asarray(multihost_utils.process_allgather(np.asarray((local_lhs,)))))
    )
    rhs = float(
        np.sum(np.asarray(multihost_utils.process_allgather(np.asarray((local_rhs,)))))
    )

    expected_by_receptor = {
        int(identifier): float(value)
        for identifier, value in zip(
            np.asarray(receptor_evidence.receptor_ids, dtype=np.int64),
            field_value(np.asarray(receptor_evidence.donor_cells, dtype=np.int64)),
            strict=True,
        )
    }
    output_matches_winners = True
    local_receptor_ids = dict(exchange.receptor_global_ids)
    for name, values in output.items():
        output_matches_winners &= np.allclose(
            np.asarray(values).reshape(-1),
            np.asarray(
                [
                    expected_by_receptor[identifier]
                    for identifier in local_receptor_ids[name]
                ]
            ),
            rtol=0,
            atol=0,
        )
    halo_ids = np.asarray(exchange.halo.local_global_ids, dtype=np.int64)
    halo_valid = np.asarray(exchange.halo.local_valid, dtype=np.bool_)
    halo_owned = np.asarray(exchange.halo.local_owned, dtype=np.bool_)
    halo_owners = np.asarray(exchange.halo.entity_owner, dtype=np.int32)
    remote_donor_halo = bool(
        partition == 1
        and np.any(
            halo_valid[partition]
            & ~halo_owned[partition]
            & (halo_owners[halo_ids[partition]] == 0)
        )
    )
    empty_rank_clean = bool(
        partition != 2
        or (
            not np.any(halo_valid[partition])
            and not np.any(halo_owned[partition])
            and not np.any(np.asarray(exchange.halo.phase_send_valid)[:, partition])
            and not np.any(np.asarray(exchange.halo.phase_receive_valid)[:, partition])
        )
    )
    locally_owned_winners_are_active = True
    if donor_name in views:
        donor_view = views[donor_name]
        owner_by_cell = dict(
            zip(
                np.asarray(donor_view.cell_ids, dtype=np.int64).tolist(),
                np.asarray(donor_view.cell_owner_ranks, dtype=np.int32).tolist(),
                strict=True,
            )
        )
        status_by_cell = dict(
            zip(
                np.asarray(
                    resolved.blanking_of(donor_name).cell_ids, dtype=np.int64
                ).tolist(),
                np.asarray(
                    resolved.blanking_of(donor_name).cell_status, dtype=np.int32
                ).tolist(),
                strict=True,
            )
        )
        ambiguous_by_cell = dict(
            zip(
                np.asarray(
                    resolved.blanking_of(donor_name).cell_ids, dtype=np.int64
                ).tolist(),
                np.asarray(
                    resolved.blanking_of(donor_name).ambiguous_cells, dtype=np.bool_
                ).tolist(),
                strict=True,
            )
        )
        locally_owned_winners_are_active = all(
            status_by_cell[int(identifier)] == int(OversetCellStatus.ACTIVE)
            and not ambiguous_by_cell[int(identifier)]
            for identifier in np.asarray(receptor_evidence.donor_cells, dtype=np.int64)
            if owner_by_cell.get(int(identifier)) == partition
        )
    route_flags = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    int(remote_donor_halo),
                    int(empty_rank_clean),
                    int(receptor_name in output),
                    int(output_matches_winners),
                    int(locally_owned_winners_are_active),
                ),
                dtype=np.int32,
            )
        )
    )
    assert route_flags[1, 0] == 1
    assert route_flags[2, 1] == 1
    assert route_flags[1, 2] == 1
    assert np.all(route_flags[:, 3:] == 1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

    wrong_ranks = tuple(reversed(group_ranks))
    wrong_devices = tuple(reversed(devices))
    wrong_group = ExecutionGroup(
        ExecutionGroupSpec(
            "native-owner-local-overset-stale-owner-order",
            wrong_ranks,
            tuple((device.process_index, device.id) for device in wrong_devices),
            mesh_axes=(("overset", 3),),
        ),
        wrong_devices,
    )
    with pytest.raises(OversetConnectivityError, match="group shard") as stale_owner:
        transfer.prepare_owner_local_exchange(coefficient_ids, wrong_group, views)
    assert stale_owner.value.connectivity.connectivity_id == candidate.connectivity_id

    orphan_specs = (
        OversetPartSpec(
            donor_name,
            priority=2,
            owner_rank=0,
            image_rotation=np.eye(2),
            image_translation=np.asarray((100.0, 0.0)),
        ),
        OversetPartSpec(
            shadow_name,
            priority=1,
            owner_rank=0,
            image_rotation=np.eye(2),
            image_translation=np.asarray((100.0, 0.0)),
        ),
        OversetPartSpec(receptor_name, boundary=boundary, owner_rank=1),
    )
    orphan_candidate = prepare_overset_connectivity(
        assembly,
        orphan_specs,
        policy=OversetPolicy(fringe_layers=1, donor_priority="part-priority"),
        execution_group=group,
    )
    assert not orphan_candidate.complete
    orphan_transfer = prepare_overset_field_transfer(
        orphan_candidate,
        local_reconstructions,
        "u",
        value_action="invariant",
    )
    with pytest.raises(OversetConnectivityError, match="orphan") as orphan_failure:
        orphan_transfer.prepare_owner_local_exchange(coefficient_ids, group, views)
    assert orphan_failure.value.connectivity.donor_search_complete
    assert not orphan_failure.value.connectivity.complete
    assert orphan_failure.value.connectivity.orphan_count > 0


def test_native_field_action_and_transpose_are_exact() -> None:
    donor, receptor, connectivity = _native_pair()
    discretization = phx.discretization.FiniteElementPlan(
        donor.carrier.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    transfer = prepare_overset_field_transfer(
        connectivity, {"donor": reconstruction}, "u"
    )
    assert not transfer.conservative
    field = jnp.asarray(
        1.0
        + np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 0]
        + 2.0 * np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 1]
    )
    output = transfer.apply({"donor": field})["receptor"]
    expected = np.asarray(receptor.carrier.mesh.coordinates)
    np.testing.assert_allclose(output, 1.0 + expected[:, 0] + 2.0 * expected[:, 1])
    dual = jnp.asarray((1.5, -0.25, 2.0))
    transpose = transfer.transpose({"receptor": dual})["donor"]
    np.testing.assert_allclose(
        jnp.vdot(output, dual), jnp.vdot(field, transpose), atol=1e-12
    )
    invalid_cotangents = {
        evidence.part_name: jnp.zeros((evidence.receptor_ids.size, *transfer.value_shape))
        for evidence in transfer.receptors
    }
    invalid_cotangents["receptor"] = jnp.zeros((2,))
    with pytest.raises(ValueError, match="prepared receptor layouts"):
        transfer.transpose(invalid_cotangents)


def test_native_conservative_common_refinement_conserves_extensive_content() -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    source = _part("source", points, ((0, 1, 2), (0, 2, 3)))
    target = _part("target", points, ((0, 1, 3), (1, 2, 3)))
    remap = prepare_overset_conservative_remap(source, target)

    assert remap.complete
    assert remap.evidence.source_gap_count == 0
    assert remap.evidence.target_gap_count == 0
    source_average = jnp.asarray((2.0, 5.0))
    source_content = source_average * remap.refinement.source_measures
    target_content = remap.apply_extensive(source_content)
    action_evidence = remap.apply_extensive_with_evidence(source_content)
    np.testing.assert_allclose(action_evidence.values, target_content, atol=1e-12)
    np.testing.assert_allclose(
        action_evidence.absolute_conservation_defect, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(target_content.sum(), source_content.sum(), atol=1e-12)
    np.testing.assert_allclose(remap.apply_intensive(jnp.full((2,), 3.5)), 3.5)
    target_dual = jnp.asarray((1.25, -0.75))
    np.testing.assert_allclose(
        jnp.vdot(target_content, target_dual),
        jnp.vdot(source_content, remap.apply_extensive_transpose(target_dual)),
        atol=1e-12,
    )


def test_curved_common_root_overset_remap_conserves_extensive_content() -> None:
    (
        root_mesh,
        root_geometry,
        source_geometry,
        target_mesh,
        target_geometry,
        _,
        witness,
    ) = make_curved_material_refinement()
    assert witness is not None
    source_result = phx.meshing.certify_cell_mesh(
        root_mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=source_geometry,
    )
    target_result = phx.meshing.certify_cell_mesh(
        target_mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=target_geometry,
    )
    source = MeshPart("curved-source", source_result)
    target = MeshPart("curved-target", target_result)
    remap = prepare_overset_conservative_remap(source, target, common_root=witness)

    assert remap.complete
    assert remap.integration_evidence is not None
    assert remap.integration_evidence.exact_polynomial_integral_count > 0
    source_content = jnp.asarray((6.0,))
    target_content = remap.apply_extensive(source_content)
    np.testing.assert_allclose(target_content, (3.0, 3.0), atol=1e-12)
    source_average = jnp.asarray((2.5,))
    target_average = remap.apply_intensive(source_average)
    np.testing.assert_allclose(target_average, (2.5, 2.5))
    np.testing.assert_allclose(target_content.sum(), source_content.sum(), atol=1e-12)
    target_dual = jnp.asarray((1.0, 3.0))
    np.testing.assert_allclose(
        jnp.vdot(target_content, target_dual),
        jnp.vdot(source_content, remap.apply_extensive_transpose(target_dual)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.vdot(target_average, target_dual),
        jnp.vdot(source_average, remap.apply_intensive_transpose(target_dual)),
        atol=1e-12,
    )


def test_orphaned_moving_overset_rebind_preserves_accepted_composition() -> None:
    _, receptor, connectivity = _native_pair()
    composition = Composition(
        connectivity.composition_entries(), boundary_id="accepted-0"
    )
    monitor = phx.meshing.MeshMotionMonitor(receptor.carrier.mesh)
    moved_coordinates = np.asarray(receptor.carrier.mesh.coordinates) + np.asarray(
        (4.0, 0.0)
    )
    advance = phx.meshing.advance_mesh_motion(
        monitor, receptor.carrier, moved_coordinates, boundary_residual=0.0
    )
    assert advance.accepted
    candidate = connectivity.moved_from_advances({"receptor": advance})
    assert not candidate.complete
    staged = prepare_overset_motion_rebind(composition, connectivity, candidate)

    receipt = staged.commit(accepted_boundary=True)

    assert not receipt.published
    assert receipt.composition is composition
    assert connectivity.epoch == 0
    assert candidate.epoch == 1


def test_overset_motion_requires_accepted_source_bound_advance() -> None:
    _, receptor, connectivity = _native_pair()
    monitor = phx.meshing.MeshMotionMonitor(receptor.carrier.mesh)
    moved_coordinates = np.asarray(receptor.carrier.mesh.coordinates) + np.asarray(
        (0.02, 0.0)
    )
    advance = phx.meshing.advance_mesh_motion(
        monitor, receptor.carrier, moved_coordinates, boundary_residual=0.0
    )
    assert advance.accepted
    with pytest.raises(ValueError, match="current overset source"):
        connectivity.moved_from_advances({"donor": advance})

    rejected = phx.meshing.advance_mesh_motion(
        monitor,
        receptor.carrier,
        moved_coordinates,
        boundary_residual=1.0,
    )
    assert not rejected.accepted
    with pytest.raises(ValueError, match="accepted mesh-motion"):
        connectivity.moved_from_advances({"receptor": rejected})


def test_accepted_moving_overset_rebind_updates_interpolated_fields() -> None:
    donor, receptor, connectivity = _native_pair()
    composition = Composition(
        connectivity.composition_entries(), boundary_id="accepted-0"
    )
    discretization = phx.discretization.FiniteElementPlan(
        donor.carrier.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    coefficients = jnp.asarray(
        1.0
        + np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 0]
        + 2.0 * np.asarray(discretization.dof_maps[0].dof_coordinates)[:, 1]
    )
    initial_transfer = prepare_overset_field_transfer(
        connectivity, {"donor": reconstruction}, "u"
    )
    initial = initial_transfer.apply({"donor": coefficients})["receptor"]

    moved_coordinates = np.asarray(receptor.carrier.mesh.coordinates) + np.asarray(
        (0.02, 0.0)
    )
    monitor = phx.meshing.MeshMotionMonitor(receptor.carrier.mesh)
    advance = phx.meshing.advance_mesh_motion(
        monitor, receptor.carrier, moved_coordinates, boundary_residual=0.0
    )
    assert advance.accepted
    candidate = connectivity.moved_from_advances({"receptor": advance})
    assert candidate.complete
    staged = prepare_overset_motion_rebind(composition, connectivity, candidate)
    refused = staged.commit(accepted_boundary=False)
    assert not refused.published
    assert refused.composition is composition
    receipt = staged.commit(accepted_boundary=True)

    assert receipt.published
    assert receipt.composition is not composition
    assert receipt.composition.value("overset:connectivity") is candidate
    moved_transfer = prepare_overset_field_transfer(
        candidate, {"donor": reconstruction}, "u"
    )
    moved = moved_transfer.apply({"donor": coefficients})["receptor"]
    np.testing.assert_allclose(moved, initial + 0.02, rtol=0.0, atol=1.0e-11)


def test_uncovered_conservative_remap_fails_closed() -> None:
    source = _part(
        "source",
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
        ((0, 1, 2),),
    )
    target = _part(
        "target",
        ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)),
        ((0, 1, 2),),
    )
    remap = prepare_overset_conservative_remap(source, target)

    assert not remap.complete
    with pytest.raises(ValueError, match="incomplete"):
        remap.apply_extensive(jnp.asarray((1.0,)))
