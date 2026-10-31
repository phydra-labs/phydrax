import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing._layer_core_mapped import mapped_source_lipschitz_bound
from phydrax.meshing._layer_core_regions import prepare_layer_region_identity
from phydrax.meshing._layer_core_topology import cycle, faces, triangles
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_layer import _implicit_layer_surface_chart_witness
from tests._support.implicit_layer_curving_fixtures import certified_implicit_layer_wall
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

    with pytest.raises(
        ValueError, match="realized layer or bounded core material incidence"
    ):
        prepare_layer_region_identity(
            complex_,
            np.asarray((0,), dtype=np.int64),
            ("core", "layer"),
            np.asarray((0,)),
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


def test_layer_core_assembly_rejects_nonmanifold_core_transition() -> None:
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
    # Three cells on one face never reach the combined face table: assembling
    # the composed mesh already refuses the non-manifold transition.
    with pytest.raises(ValueError, match="face-manifold"):
        CellMesh(
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


def _assert_remeshed_implicit_layer_witness(*, smooth: bool) -> tuple[str, ...]:
    _surface, query, contract, _discovery_wall, _geometry, _ = _adaptive_source(
        smooth=smooth
    )
    # The discovery scaffold has sliver triangles, so the layer wall is the
    # certified zero-set remeshed carrier; its cells may share scaffold charts.
    layer_wall, root_witness = certified_implicit_layer_wall(query)
    assert dict(layer_wall.evidence)["carrier_remeshed"] == 1.0
    wall = layer_wall.wall
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
    assert witness.source_chart_ids == root_witness.source_chart_ids
    np.testing.assert_array_equal(
        witness.implicit_cell_keys, root_witness.implicit_cell_keys
    )
    np.testing.assert_array_equal(
        witness.implicit_root_edge_keys, root_witness.implicit_root_edge_keys
    )
    assert np.unique(witness.entity_global_ids).size == wall.blocks[0].cell_count
    return witness.source_chart_ids


def test_implicit_layer_witness_projects_to_output_volume_facets() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    _assert_remeshed_implicit_layer_witness(smooth=True)


def test_remeshed_sphere_layer_witness_repeats_seeding_scaffold_charts() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    chart_ids = _assert_remeshed_implicit_layer_witness(smooth=False)
    # Remeshed cells name their nearest scaffold triangle, so charts repeat.
    assert len(set(chart_ids)) < len(chart_ids)


def test_mapped_scoped_chart_witness_allows_unrelated_target_exterior_facets() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from tests.unit.meshing.test_native_mapped_hex_routes import (
        _mapped_cube_source,
        _mapped_surface_chart_witness,
        _slice_mapped_chart_witness,
    )

    source, domain = _mapped_cube_source()
    geometry = CellGeometrySpec.affine(domain.reference_mesh)
    full_query, full_witness = _mapped_surface_chart_witness(source, domain, geometry)

    def scoped_chart_ids(query: MappedReferenceBoundarySource) -> set[str]:
        # A scoped query keeps the full chart table and selects its scope rows.
        return {query.face_chart_ids[row] for row in query._selected_rows(2)}

    # Scope every source triangle that charts one complete target hex facet;
    # the other five facets stay unrelated exterior facets.
    facets = tuple(
        zip(
            np.asarray(full_witness.cell_global_ids).tolist(),
            np.asarray(full_witness.local_entity_indices).tolist(),
            strict=True,
        )
    )
    rows = tuple(row for row, facet in enumerate(facets) if facet == facets[0])
    facet_chart_ids = {full_witness.source_chart_ids[row] for row in rows}
    selected_query = source.boundary_query(
        face_scope_keys=tuple(
            key
            for key in full_query.face_scope_keys
            if scoped_chart_ids(source.boundary_query(face_scope_keys=(key,)))
            & facet_chart_ids
        )
    )
    assert scoped_chart_ids(selected_query) == facet_chart_ids
    assert 1 < len(rows) < len(facets)
    sliced = _slice_mapped_chart_witness(full_witness, rows)
    # A scoped witness names the scoped query it was derived for.
    witness = SourceBoundaryChartWitness(
        mesh_id=sliced.mesh_id,
        geometry_id=sliced.geometry_id,
        topology_id=sliced.topology_id,
        source_id=sliced.source_id,
        source_revision=sliced.source_revision,
        source_query_id=selected_query.query_id,
        chart_dimension=sliced.chart_dimension,
        reference_dimension=sliced.reference_dimension,
        cell_global_ids=sliced.cell_global_ids,
        entity_global_ids=sliced.entity_global_ids,
        local_entity_indices=sliced.local_entity_indices,
        target_reference_vertices=sliced.target_reference_vertices,
        target_reference_rational_tokens=sliced.target_reference_rational_tokens,
        target_face_vertex_global_ids=sliced.target_face_vertex_global_ids,
        source_chart_ids=sliced.source_chart_ids,
        source_entity_ids=sliced.source_entity_ids,
        occurrence_paths=sliced.occurrence_paths,
        source_parameters=sliced.source_parameters,
        source_parameter_rational_tokens=sliced.source_parameter_rational_tokens,
        orientations=sliced.orientations,
    )

    validate_source_fidelity_witness(
        domain.reference_mesh, geometry, selected_query, witness
    )
