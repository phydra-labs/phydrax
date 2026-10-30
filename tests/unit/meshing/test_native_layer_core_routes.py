import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing._layer_core_mapped import mapped_source_lipschitz_bound
from phydrax.meshing._layer_core_regions import prepare_layer_region_identity
from phydrax.meshing._layer_core_topology import cycle, faces, triangles
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_layer import _implicit_layer_surface_chart_witness
from tests._support.layer_curving_fixtures import _layer_mesh, _mixed_construction
from tests.unit.geometry.test_implicit_source_charts import _adaptive_source


def _tetrahedron_plc(region_id: str = "core") -> PiecewiseLinearComplex:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    polygons = (
        np.asarray((1, 2, 3), dtype=np.int64),
        np.asarray((0, 3, 2), dtype=np.int64),
        np.asarray((0, 1, 3), dtype=np.int64),
        np.asarray((0, 2, 1), dtype=np.int64),
    )
    return PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(4, dtype=np.int64),
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        (region_id,),
        vertex_ids=("v0", "v1", "v2", "v3"),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=("f0", "f1", "f2", "f3"),
        facet_occurrence_paths=((),) * 4,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )


def test_fixed_plc_region_identity_retains_explicit_composite_materials() -> None:
    complex_ = _tetrahedron_plc("layer")

    names, core_map = prepare_layer_region_identity(
        complex_, np.asarray((0,), dtype=np.int64), ("core", "layer"), np.asarray((1,))
    )

    assert names == ("core", "layer")
    np.testing.assert_array_equal(core_map, np.asarray((1,), dtype=np.int64))
    assert not core_map.flags.writeable


def test_layer_core_region_identity_rejects_unrealized_composite_region() -> None:
    complex_ = _tetrahedron_plc()

    with pytest.raises(ValueError, match="realized layer or bounded core material incidence"):
        prepare_layer_region_identity(
            complex_, np.asarray((0,), dtype=np.int64), ("core", "layer"), np.asarray((0,))
        )


def test_quad_split_is_cyclically_stable_and_orientation_preserving() -> None:
    assert cycle((2, 3, 0, 1)) == (0, 1, 2, 3)
    assert triangles((0, 1, 2, 3)) == ((0, 1, 2), (0, 2, 3))
    assert triangles((0, 3, 2, 1)) == ((0, 3, 2), (0, 2, 1))


def test_combined_face_table_retains_opposite_transition_sides() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "transition-tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
                global_ids=np.asarray((10, 20), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((41, 11, 31, 21, 51), dtype=np.int64),
    )

    table = faces(mesh, np.asarray((3, 7), dtype=np.int64))

    shared = table[(0, 1, 2)]
    assert len(shared) == 2
    assert tuple(face.region for face in shared) == (3, 7)
    assert cycle(shared[0].vertices) == cycle(tuple(reversed(shared[1].vertices)))


def test_combined_face_table_rejects_nonmanifold_core_transition() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 2.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "nonmanifold-tetrahedra",
                "tetrahedron",
                np.asarray(
                    ((0, 1, 2, 3), (0, 2, 1, 4), (0, 1, 2, 5)),
                    dtype=np.int64,
                ),
                global_ids=np.asarray((10, 20, 30), dtype=np.int64),
            ),
        ),
    )

    with pytest.raises(ValueError, match="at most two incident cells"):
        faces(mesh, np.asarray((0, 0, 0), dtype=np.int64))


def test_mapped_layer_core_sizing_uses_physical_root_jacobian_bound() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    facets = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)
    reference_domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("solid",),
        source_id="mapped-layer-root",
    )
    root_mesh = CellMesh(
        vertices,
        (CellBlock("root", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)),),
        vertex_global_ids=np.asarray((11, 13, 17, 19), dtype=np.int64),
    )
    affine = CellGeometrySpec.affine(root_mesh)
    coordinates = 2.0 * vertices + np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    source_geometry = CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(affine.block_names, affine.geometry_dofs, strict=True)
        },
        coordinates,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-layer-root",
        source_revision="revision-1",
        coordinate_contract=SpatialCoordinateContract.si(),
    )

    assert mapped_source_lipschitz_bound(domain) == pytest.approx(2.0)

def test_implicit_layer_witness_projects_to_output_volume_facets() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    _surface, query, contract, wall, _wall_geometry, root_witness = _adaptive_source()
    layers = _layer_mesh(wall, None)
    source, construction = _mixed_construction(
        layers,
        query.source_id,
        query.source_revision,
        wall,
        contract.spatial_id,
        fidelity_source=query,
    )
    mesh = construction.mesh
    witness = _implicit_layer_surface_chart_witness(source, construction)

    assert witness.mesh_id == mesh.mesh_id
    assert witness.reference_dimension == 3
    root_row_by_chart = {
        chart_id: row for row, chart_id in enumerate(root_witness.source_chart_ids)
    }
    assert set(witness.source_chart_ids) == set(root_row_by_chart)
    for row, chart_id in enumerate(witness.source_chart_ids):
        root_row = root_row_by_chart[chart_id]
        np.testing.assert_array_equal(
            witness.implicit_cell_keys[row], root_witness.implicit_cell_keys[root_row]
        )
        np.testing.assert_array_equal(
            witness.implicit_root_edge_keys[row],
            root_witness.implicit_root_edge_keys[root_row],
        )
    assert np.unique(witness.entity_global_ids).size == len(root_witness.source_chart_ids)


def test_mapped_scoped_chart_witness_allows_unrelated_target_exterior_facets() -> None:
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from tests.unit.meshing.test_native_mapped_hex_routes import (
        _mapped_cube_source,
        _mapped_surface_chart_witness,
        _slice_mapped_chart_witness,
    )

    source, domain = _mapped_cube_source()
    geometry = CellGeometrySpec.affine(domain.reference_mesh)
    full_query, full_witness = _mapped_surface_chart_witness(
        source, domain, geometry
    )
    selected_query = source.boundary_query(
        face_scope_keys=(full_query.face_scope_keys[0],)
    )
    selected_chart_ids = set(selected_query.face_chart_ids)
    rows = tuple(
        row
        for row, chart_id in enumerate(full_witness.source_chart_ids)
        if chart_id in selected_chart_ids
    )
    witness = _slice_mapped_chart_witness(full_witness, rows)

    validate_source_fidelity_witness(
        domain.reference_mesh, geometry, selected_query, witness
    )
