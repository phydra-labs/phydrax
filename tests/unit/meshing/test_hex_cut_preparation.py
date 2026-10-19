from fractions import Fraction

import numpy as np
import pytest

from phydrax._host_storage import retained_host_storage_profile
from phydrax._meshcore import is_native_host_buffer
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._mapped_geometry_measure import certified_cell_measures
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._hex_cut_preparation import (
    certify_cut_corner_coverage,
    CutCornerLinkError,
    prepare_cut_corner_hexes,
    realize_cut_corner_hexes,
)
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.hex_sources import pentagonal_cut_prisms


@pytest.mark.parametrize("layers", (1, 2))
def test_nonpolycube_corner_maps_keep_source_and_exact_volume(layers: int) -> None:
    mesh, geometry, original_domain = pentagonal_cut_prisms(layers)
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget(
            queries_upper=cell_geometry_preparation_query_upper(geometry),
        ) as reference_budget,
    ):
        prepared = prepare_cut_corner_hexes(
            mesh,
            geometry,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
            workspace=host_workspace,
        )
        assert prepared.hexes.shape == (10 * layers, 8)
        assert prepared.exact_vertices[
            : mesh.coordinates.shape[0]
        ] == geometry.source_coordinates(budget=reference_budget)
        assert any(
            value.denominator == 5 for point in prepared.exact_vertices for value in point
        )
        construction = realize_cut_corner_hexes(
            prepared,
            MeshingLimits(),
            certificate_limits=MeshCertificateLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        coverage, regions = certify_cut_corner_coverage(
            construction,
            original_domain,
            np.arange(layers),
            certificate_limits=MeshCertificateLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert coverage.status == "certified"
        assert np.array_equal(regions, prepared.parent_cells)
        assert construction.validity.all_certified
        assert construction.embedding.status == "certified"
        assert all(block.cell_kind == "hexahedron" for block in construction.mesh.blocks)
        with coordinate_reference_budget(
            queries_upper=(
                cell_geometry_preparation_query_upper(construction.geometry)
                + cell_geometry_preparation_query_upper(construction.geometry)
            ),
        ) as reference_budget:
            assert (
                construction.geometry.source_coordinates(budget=reference_budget)
                == prepared.exact_vertices
            )
            volumes, errors, exact = certified_cell_measures(
                construction.mesh, construction.geometry, budget=reference_budget
            )
        assert exact
        # Independent shoelace sum 2+8+2 gives area 6, through unit layers.
        expected = Fraction(6 * layers)
        measured = sum((Fraction(float(value)) for value in volumes), Fraction(0))
        bound = sum((Fraction(float(value)) for value in errors), Fraction(0))
        assert abs(measured - expected) <= bound


def test_nonsimple_link_reports_original_scientific_incidence() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    cells = [
        [
            np.asarray(face, dtype=np.int64)
            for face in ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
        ]
    ]
    mesh = CellMesh.from_polyhedra(
        points,
        cells,
        vertex_global_ids=np.asarray([11, 17, 23, 29, 31]),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(CutCornerLinkError) as error:
            prepare_cut_corner_hexes(
                mesh,
                CellGeometrySpec.affine(mesh),
                MeshingLimits(),
                scope=execution_scope,
                budget=coordinate_budget,
                workspace=host_workspace,
            )
        witness = error.value.incidence
        assert witness.vertex_id == 31
        assert len(witness.edge_ids) == len(witness.face_ids) == 4
        assert len(set(witness.edge_ids)) == len(set(witness.face_ids)) == 4


@pytest.mark.parametrize(
    "face_arity", (5, 4), ids=("pentagonal-face", "quadrilateral-face")
)
def test_immutable_original_face_is_not_retagged_as_hex_boundary(face_arity: int) -> None:
    mesh, geometry, _ = pentagonal_cut_prisms()
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "fixture.connectivity must be polyhedral; got incompatible connectivity"
        )
    arities = np.diff(np.asarray(connectivity.face_vertex_offsets))
    face_row = np.flatnonzero(arities == face_arity)[0]
    face = int(np.asarray(mesh.entity_set(2).entity_ids)[face_row])
    identity = mesh.mesh_id
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure, match=r"cut_preparation\.immutable_face_ids"):
            prepare_cut_corner_hexes(
                mesh,
                geometry,
                MeshingLimits(),
                immutable_face_ids=(face,),
                scope=execution_scope,
                budget=coordinate_budget,
                workspace=host_workspace,
            )
        assert mesh.mesh_id == identity
        assert host_workspace.bound == 0


def test_cut_capacity_refuses_before_corner_construction() -> None:
    mesh, geometry, _ = pentagonal_cut_prisms()
    with native_meshing_request(MeshingLimits(maximum_cells=9)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure, match="cells"):
            prepare_cut_corner_hexes(
                mesh,
                geometry,
                MeshingLimits(maximum_cells=9),
                scope=execution_scope,
                budget=coordinate_budget,
                workspace=host_workspace,
            )


def test_immutable_cut_face_ids_must_name_declared_scientific_faces() -> None:
    mesh, geometry, _ = pentagonal_cut_prisms()
    unknown = int(np.max(np.asarray(mesh.entity_set(2).entity_ids))) + 1
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        workspace,
    ):
        with pytest.raises(ValueError, match=r"cut_preparation\.immutable_face_ids"):
            prepare_cut_corner_hexes(
                mesh,
                geometry,
                MeshingLimits(),
                immutable_face_ids=(unknown,),
                scope=execution_scope,
                budget=coordinate_budget,
                workspace=workspace,
            )


def test_cut_corner_banks_use_the_admitted_host_ticket_allowance() -> None:
    """A full ticket reservation must still permit its own small output banks."""
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = (((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),)
    mesh = CellMesh.from_polyhedra(points, faces)
    geometry = CellGeometrySpec.affine(mesh)
    limits = MeshingLimits(maximum_scratch_bytes=1024**2, maximum_work_units=100_000)
    with native_meshing_request(limits) as (
        execution_scope,
        coordinate_budget,
        workspace,
    ):
        prepared = prepare_cut_corner_hexes(
            mesh,
            geometry,
            limits,
            scope=execution_scope,
            budget=coordinate_budget,
            workspace=workspace,
        )
        assert prepared.hexes.shape == (4, 8)
        assert prepared.corner_orientations.shape == (4,)
        assert np.all(np.abs(prepared.corner_orientations) == 1)
        assert np.array_equal(prepared.parent_cells, np.zeros(4, dtype=np.int64))
        assert prepared.exact_vertices[:4] == tuple(
            tuple(Fraction(value) for value in point) for point in points
        )
        assert prepared.work_units <= limits.maximum_work_units
        _, retained_bytes, logical_bytes, _ = retained_host_storage_profile(
            prepared, {}, is_native_host_buffer
        )
        assert workspace.bound == retained_bytes
        assert workspace.logical_unmanaged_bytes_upper == logical_bytes
    assert workspace.bound == 0


def test_cut_corner_interface_uses_one_shared_pentagonal_face_node() -> None:
    mesh, geometry, _ = pentagonal_cut_prisms(2)
    with native_meshing_request(MeshingLimits()) as (scope, budget, workspace):
        prepared = prepare_cut_corner_hexes(
            mesh,
            geometry,
            MeshingLimits(),
            scope=scope,
            budget=budget,
            workspace=workspace,
        )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "fixture.connectivity must be polyhedral; got incompatible connectivity"
        )
    incidence = np.bincount(
        np.asarray(connectivity.cell_face_values), minlength=connectivity.face_count
    )
    shared = int(np.flatnonzero(incidence == 2)[0])
    face_node = connectivity.vertex_count + connectivity.edge_count + shared
    assert np.count_nonzero(prepared.hexes == face_node) == 10
    assert set(prepared.parent_cells.tolist()) == {0, 1}
