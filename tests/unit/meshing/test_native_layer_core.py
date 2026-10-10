from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing._layer_core import (
    _prepare_identity,
    layer_interval_classes,
    remap_layer_index_attribute,
    validate_layer_index_attributes,
)
from phydrax.meshing._lineage import identity_lineage


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native fixed PLC recovery requires meshcore"
)
M = phx.meshing


@dataclass(frozen=True)
class _Case:
    layers: M.BoundaryLayerMesh
    core: M.PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    layer_regions: np.ndarray
    region_ids: tuple[str, ...]
    core_region_map: np.ndarray


def _case(*, material_interface: bool = False) -> _Case:
    wall = phx.discretization.CellMesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "wall", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
            ),
        ),
    )
    entities = wall.entity_set(2)
    scope = M.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        entities.entity_ids,
    )
    control = M.BoundaryLayerControl(
        scope,
        M.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=M.BoundaryLayerRoute.ADVANCING,
    )
    layers = M.prepare_boundary_layers(wall, control)
    if layers.cap is None:
        raise RuntimeError("The square wall must produce a cap.")
    cap = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    )
    cap_points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    upper = cap_points.copy()
    upper[:, 2] = 1.0
    outer = np.asarray(
        (
            (4, 5, 6),
            (4, 6, 7),
            (0, 5, 4),
            (0, 1, 5),
            (1, 6, 5),
            (1, 2, 6),
            (2, 7, 6),
            (2, 3, 7),
            (3, 4, 7),
            (3, 0, 4),
        ),
        dtype=np.int64,
    )
    polygons = np.concatenate((cap, outer))
    core_region = 0
    region_ids = ("wall-material", "core-material") if material_interface else ("fluid",)
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(polygons),
        np.concatenate(
            (
                np.zeros(cap.shape[0], dtype=np.int64),
                np.ones(outer.shape[0], dtype=np.int64),
            )
        ),
        np.asarray(((core_region, -1), (-1, core_region)), dtype=np.int64),
        (region_ids[-1],),
        boundary="fixed",
    )
    return _Case(
        layers,
        complex_,
        np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(4, -1, dtype=np.int64),
            )
        ),
        np.arange(cap.shape[0], dtype=np.int64),
        np.zeros(sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64),
        region_ids,
        np.asarray([1 if material_interface else 0], dtype=np.int64),
    )


def _result(case: _Case) -> M.CellMeshingResult:
    source = M.NativeLayerCoreSource(
        case.layers,
        case.core,
        "layered-square",
        "reference",
        vertex_layer_ids=case.vertex_layer_ids,
        cap_polygon_ids=case.cap_polygon_ids,
        layer_regions=case.layer_regions,
        region_ids=case.region_ids,
        core_region_map=case.core_region_map,
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "layered-core-facets",
        np.arange(case.core.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def test_native_open_layer_cap_and_complete_core_publish_one_volume() -> None:
    case = _case()
    result = _result(case)
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    np.testing.assert_array_equal(
        points[: case.layers.mesh.coordinates.shape[0]], case.layers.mesh.coordinates
    )
    assert result.audit.passed and result.compliance.passed
    if result.certification is None or result.certification.coverage is None:
        raise AssertionError("The combined layer/core needs independent coverage.")
    assert result.certification.passed
    np.testing.assert_allclose(
        np.asarray(result.certification.coverage.achieved_region_measures),
        (1.0,),
        rtol=1e-12,
    )
    assert {block.cell_kind for block in result.mesh.blocks} == {"prism", "tetrahedron"}
    assert {zone.name for zone in result.zones} == {"fluid"}
    assert {label.name for label in result.labels} >= {"boundary-layer", "core"}
    interface = next(
        patch for patch in result.patches if patch.name == "layer-core-interface"
    )
    assert interface.scope.entity_ids.shape == (2,)
    ancestry = next(
        attribute for attribute in result.attributes if attribute.name == "layer_index"
    )
    values = np.asarray(ancestry.values)
    assert np.count_nonzero(values == 0) == 2 and np.count_nonzero(values == 1) == 2
    assert np.count_nonzero(values == -1) == sum(
        block.cell_count for block in result.mesh.blocks if block.name == "core"
    )
    columns = np.asarray(
        next(
            attribute
            for attribute in result.attributes
            if attribute.name == "layer_column"
        ).values
    )
    np.testing.assert_array_equal(columns[values == -1], -1)
    np.testing.assert_array_equal(
        np.sort(columns[values == 0]), np.sort(columns[values == 1])
    )
    assert np.unique(columns[values == 0]).size == 2


def test_native_layer_core_preserves_material_interface_and_region_measures() -> None:
    result = _result(_case(material_interface=True))
    if result.certification is None or result.certification.coverage is None:
        raise AssertionError("Material coverage must be independently certified.")
    assert result.certification.passed
    np.testing.assert_allclose(
        np.asarray(result.certification.coverage.achieved_region_measures),
        (0.2, 0.8),
        rtol=1e-12,
    )
    assert {zone.name for zone in result.zones} == {"wall-material", "core-material"}
    interface = next(label for label in result.labels if label.name == "interface")
    cap = next(patch for patch in result.patches if patch.name == "layer-core-interface")
    np.testing.assert_array_equal(interface.scope.entity_ids, cap.scope.entity_ids)


def test_cap_identity_does_not_weld_numerically_equal_unrelated_vertices() -> None:
    case = _case()
    mapping = case.vertex_layer_ids.copy()
    mapping[0] = -1
    with pytest.raises(ValueError, match="exact layer vertex IDs"):
        _prepare_identity(case.layers, case.core, mapping, case.cap_polygon_ids)


def test_cap_reversed_orientation_is_refused_before_recovery() -> None:
    case = _case()
    polygons = case.core.polygon_vertices.reshape(-1, 3).copy()
    polygons[0] = polygons[0, ::-1]
    invalid = M.PiecewiseLinearComplex(
        case.core.vertices,
        tuple(polygons),
        case.core.polygon_facets,
        case.core.facet_regions,
        case.core.region_ids,
        boundary="fixed",
    )
    with pytest.raises(ValueError, match="orientation"):
        _prepare_identity(
            case.layers, invalid, case.vertex_layer_ids, case.cap_polygon_ids
        )


def test_measured_active_columns_retain_each_physical_interval() -> None:
    case = _case()
    evidence = case.layers.evidence
    np.testing.assert_array_equal(evidence.active_column_counts, (4, 4))
    np.testing.assert_array_equal(evidence.column_active, np.ones((2, 4), dtype=np.bool_))
    np.testing.assert_allclose(evidence.column_thicknesses, 0.1, rtol=1e-12)


def test_unrelated_planar_marker_has_no_reserved_layer_semantics() -> None:
    mesh = M.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
            np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        )
    )
    cells = mesh.entity_set(2)
    scope = M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.sort(np.asarray(cells.entity_ids)),
    )
    marker = M.MeshAttribute(
        "marker",
        M.MeshAttributeRole.MARKER,
        scope,
        np.asarray([19, 23], dtype=np.int64),
    )
    source = M.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        attributes=(marker,),
    )
    validate_layer_index_attributes(source)
    np.testing.assert_array_equal(layer_interval_classes(source), [0, 0])
    assert remap_layer_index_attribute(source, identity_lineage(mesh, mesh), mesh) == ()
    np.testing.assert_array_equal(source.attributes[0].values, [19, 23])


@pytest.mark.parametrize(
    "complete_pair", (True, False), ids=("complete", "missing-column")
)
def test_reserved_layer_pair_validation_is_independent_of_unrelated_marker(
    complete_pair: bool,
) -> None:
    mesh = M.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_tetrahedra(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        )
    )
    cells = mesh.entity_set(3)
    scope = M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        np.sort(np.asarray(cells.entity_ids)),
    )
    entries = (("marker", 19), ("layer_index", 0), ("layer_column", 41))
    attributes = tuple(
        M.MeshAttribute(
            name, M.MeshAttributeRole.MARKER, scope, np.asarray([value], dtype=np.int64)
        )
        for name, value in (entries if complete_pair else entries[:2])
    )
    source = M.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        attributes=attributes,
    )
    if not complete_pair:
        with pytest.raises(ValueError):
            validate_layer_index_attributes(source)
        return
    validate_layer_index_attributes(source)
    reserved = remap_layer_index_attribute(source, identity_lineage(mesh, mesh), mesh)
    assert {attribute.name for attribute in reserved} == {"layer_index", "layer_column"}
    np.testing.assert_array_equal(
        next(
            attribute.values for attribute in reserved if attribute.name == "layer_index"
        ),
        [0],
    )
    np.testing.assert_array_equal(
        next(
            attribute.values for attribute in reserved if attribute.name == "layer_column"
        ),
        [41],
    )
    np.testing.assert_array_equal(
        next(
            attribute.values
            for attribute in source.attributes
            if attribute.name == "marker"
        ),
        [19],
    )


def _periodic_material_source(
    *, ridge: bool = False, material_interface: bool = True
) -> M.NativeLayerCoreSource:
    from phydrax.discretization import (
        CellBlock,
        CellMesh,
        PeriodicIsometryGroup,
        PeriodicMeshTopology,
    )

    if ridge:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.5, 0.1),
                (1.0, 0.5, 0.1),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
            )
        )
        rows = np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64)
    else:
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        )
        rows = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
    wall_block = CellBlock("wall", "triangle", rows)
    plain = CellMesh(points, (wall_block,))
    generator = np.eye(4, dtype=np.float64)[None]
    generator[0, 0, 3] = 1.0
    wall_roots = np.repeat(np.arange(0, points.shape[0], 2, dtype=np.int64), 2)
    wall_shifts = np.tile(
        np.asarray(((0,), (1,)), dtype=np.int64), (points.shape[0] // 2, 1)
    )
    topology = PeriodicMeshTopology(
        plain, PeriodicIsometryGroup(generator), wall_roots, wall_shifts
    )
    wall = CellMesh(points, (wall_block,), periodic_topology=topology)
    entities = wall.entity_set(2)
    scope = M.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        entities.entity_ids,
    )
    control = M.BoundaryLayerControl(
        scope,
        M.LayerSchedule.geometric(2, 0.01, growth_rate=1.0),
        route=M.BoundaryLayerRoute.ADVANCING,
        feature_angle=0.1,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
    )
    layers = M.prepare_boundary_layers(wall, control)
    cap = layers.cap
    if cap is None or cap.periodic_topology is None:
        raise RuntimeError("Periodic wall must retain its authored cap.")
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_points = np.asarray(cap.coordinates, dtype=np.float64)
    n = cap_points.shape[0]
    upper = cap_points + np.asarray((0.0, 0.0, 0.03))
    cap_roots = np.asarray(cap.periodic_topology.vertex_representatives, dtype=np.int64)
    cap_shifts = np.asarray(cap.periodic_topology.vertex_shifts, dtype=np.int64)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in cap_rows:
        for a, b in zip(row, np.roll(row, -1), strict=True):
            first, second = int(a), int(b)
            edges.setdefault((min(first, second), max(first, second)), []).append(
                (first, second)
            )
    polygons = cap_rows.tolist() + (cap_rows + n).tolist()
    seam_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for incidents in edges.values():
        if len(incidents) != 1:
            continue
        a, b = incidents[0]
        forward = (int(cap_roots[a]), a) <= (int(cap_roots[b]), b)
        first, second = (a, b) if forward else (b, a)
        triangles = ((first, second, second + n), (first, second + n, first + n))
        start = len(polygons)
        polygons.extend(
            triangles if forward else tuple(tuple(reversed(row)) for row in triangles)
        )
        if cap_roots[a] != cap_roots[b] and cap_shifts[a, 0] == cap_shifts[b, 0]:
            first_root, second_root = int(cap_roots[a]), int(cap_roots[b])
            seam_edges.setdefault(
                (min(first_root, second_root), max(first_root, second_root)), []
            ).append((start, start + 1))
    pairs = np.asarray(
        [
            (copies[0][triangle], copies[1][triangle])
            for copies in seam_edges.values()
            for triangle in range(2)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    polygon_rows = np.asarray(polygons, dtype=np.int64)
    cap_count = cap_rows.shape[0]
    core_region = 0
    incidence = np.tile(
        np.asarray((-1, core_region), dtype=np.int64), (polygon_rows.shape[0], 1)
    )
    incidence[:cap_count] = np.asarray((core_region, -1))
    core = M.PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(polygon_rows),
        np.arange(polygon_rows.shape[0], dtype=np.int64),
        incidence,
        ("core-material",) if material_interface else ("fluid",),
        boundary="fixed",
    )
    return M.NativeLayerCoreSource(
        layers,
        core,
        "periodic-layered-gap",
        "source-authored",
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(n, -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=np.arange(cap_count, dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        region_ids=("wall-material", "core-material")
        if material_interface
        else ("fluid",),
        core_region_map=np.asarray([1 if material_interface else 0], dtype=np.int64),
        core_vertex_representatives=np.concatenate((cap_roots, cap_roots + n)),
        core_vertex_shifts=np.concatenate((cap_shifts, cap_shifts)),
        core_seam_polygon_pairs=pairs,
    )


def _periodic_material_specification(
    source: M.NativeLayerCoreSource,
) -> M.VolumeMeshingSpec:
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "authored-core-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    pairs = source.core_seam_polygon_pairs
    if pairs is None:
        raise RuntimeError("The periodic fixture requires source-authored seam pairs.")
    first = pairs[:, 0]
    second = pairs[:, 1]
    transform = np.eye(4, dtype=np.float64)
    roots = source.core_vertex_representatives
    shifts = source.core_vertex_shifts
    if roots is None or shifts is None:
        raise RuntimeError("The periodic fixture requires source-authored seam roots.")
    rows = source.complex.polygon_vertices.reshape(-1, 3)
    reverse = shifts[rows[first, 0], 0] > shifts[rows[second, 0], 0]
    first, second = np.where(reverse, second, first), np.where(reverse, first, second)
    order = np.argsort(second, kind="stable")
    first, second = first[order], second[order]
    transform[0, 3] = 1.0
    source_scope = M.MeshingScope(
        scope.source_id,
        scope.source_revision,
        scope.entity_kind,
        2,
        scope.entity_set_id,
        first,
    )
    target_scope = M.MeshingScope(
        scope.source_id,
        scope.source_revision,
        scope.entity_kind,
        2,
        scope.entity_set_id,
        second,
    )
    constraint = M.PeriodicConstraint(
        source_scope,
        target_scope,
        transform,
        source_entity_ids=first,
        orientations=np.full(first.size, -1, dtype=np.int64),
    )
    regions = tuple(
        M.RegionControl(
            M.MeshingScope(
                scope.source_id,
                scope.source_revision,
                scope.entity_kind,
                3,
                "authored-regions",
                np.asarray([index], dtype=np.int64),
            ),
            name,
            f"material:{index}",
            M.RegionRole.FLUID,
        )
        for index, name in enumerate(source.region_ids)
    )
    cap_scope = M.MeshingScope(
        scope.source_id,
        scope.source_revision,
        scope.entity_kind,
        2,
        scope.entity_set_id,
        source.cap_polygon_ids,
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3,
            3,
            M.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("hexahedron", "pyramid"),
                allow_mixed=True,
            ),
        ),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=regions,
        patch_controls=(
            M.PatchControl("material-contact", cap_scope, source.region_ids),
        ),
        periodic_constraints=(constraint,),
    )


def test_periodic_fixed_core_retains_material_seams_cap_and_region_controls(
    tmp_path: Path,
) -> None:
    source = _periodic_material_source()
    specification = _periodic_material_specification(source)
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert (
        result.compliance.passed
        and result.certification is not None
        and result.certification.passed
    )
    periodic = result.mesh.periodic_topology
    assert periodic is not None
    np.testing.assert_array_equal(
        np.asarray(result.mesh.coordinates)[
            : source.layers.mesh.coordinates.shape[0]
        ].view(np.uint64),
        np.asarray(source.layers.mesh.coordinates).view(np.uint64),
    )
    incidences = [
        incidence.scipy_boundary().toarray() for incidence in periodic.quotient.incidences
    ]
    for lower, upper in zip(incidences, incidences[1:]):
        np.testing.assert_array_equal(lower @ upper, 0.0)
    assert {zone.material_id for zone in result.zones} == {"material:0", "material:1"}
    interface = next(
        patch for patch in result.patches if patch.name == "material-contact"
    )
    cap = next(patch for patch in result.patches if patch.name == "layer-core-interface")
    np.testing.assert_array_equal(interface.scope.entity_ids, cap.scope.entity_ids)
    coverage = result.certification.coverage
    if coverage is None or any(
        value is None for value in coverage.achieved_region_measures
    ):
        raise AssertionError("Every requested physical material measure must be decided.")
    np.testing.assert_allclose(
        np.asarray(coverage.achieved_region_measures, dtype=np.float64),
        (0.02, 0.03),
        rtol=1e-12,
    )
    from phydrax.lifecycle._meshing_sources import (
        _validate_layer_core_association,
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    records = {
        "certification_inputs": result.certification.request,
        "report": result.certification,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": M.NativeMeshingOptions("layer_core"),
        "generation_part": M.MeshPart("periodic-layer-core", result),
    }
    # Default untrusted limits: the node-table recipe keeps constant JSON nesting
    # and dtype banks keep the member count independent of the source strata.
    receipt = write_meshing_source_closure(tmp_path / "layer-core", records)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    restored_result = restored["generation_part"].carrier
    assert restored_result.mesh.mesh_id == result.mesh.mesh_id
    assert restored["generation_source"].layers.result_id == source.layers.result_id
    np.testing.assert_array_equal(
        np.asarray(
            restored_result.mesh.periodic_topology.quotient.entity_sets[2].entity_ids
        ),
        np.asarray(periodic.quotient.entity_sets[2].entity_ids),
    )
    cap_association = next(
        value
        for value in result.associations
        if value.source_id == source.layers.result_id
        and value.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
    )
    wrong_indices = np.roll(np.asarray(cap_association.source_indices), 1)
    wrong = M.GeometryAssociation(
        cap_association.association_kind,
        cap_association.source_id,
        cap_association.source_revision,
        cap_association.target_entity_set_id,
        cap_association.target_global_ids,
        tuple(f"{source.layers.result_id}:facet:{int(index)}" for index in wrong_indices),
        cap_association.residuals,
        exact=True,
        source_dimensions=cap_association.source_dimensions,
        source_indices=wrong_indices,
        source_entity_roles=cap_association.source_entity_roles,
        orientations=cap_association.orientations,
    )
    with pytest.raises(ValueError):
        _validate_layer_core_association(
            result.certification.request, wrong, source, mesh=result.mesh
        )
    wrong_revision = M.GeometryAssociation(
        cap_association.association_kind,
        cap_association.source_id,
        "unretained",
        cap_association.target_entity_set_id,
        cap_association.target_global_ids,
        tuple(
            f"unretained:facet:{int(index)}"
            for index in np.asarray(cap_association.source_indices)
        ),
        cap_association.residuals,
        exact=True,
        source_dimensions=cap_association.source_dimensions,
        source_indices=cap_association.source_indices,
        source_entity_roles=cap_association.source_entity_roles,
        orientations=cap_association.orientations,
    )
    with pytest.raises(ValueError):
        _validate_layer_core_association(
            result.certification.request, wrong_revision, source, mesh=result.mesh
        )


@pytest.mark.parametrize("failure", ("phase", "seam", "transform", "allocation", "work"))
def test_periodic_core_refusal_preserves_accepted_layers(failure: str) -> None:
    import equinox as eqx

    source = _periodic_material_source()
    specification = _periodic_material_specification(source)
    before = np.asarray(source.layers.mesh.coordinates).copy()
    original_shifts = source.core_vertex_shifts
    original_pairs = source.core_seam_polygon_pairs
    if original_shifts is None or original_pairs is None:
        raise RuntimeError(
            "The authored fixture requires its complete periodic ancestry."
        )
    if failure == "phase":
        shifts = original_shifts.copy()
        shifts[0, 0] = 1
        source = eqx.tree_at(lambda value: value.core_vertex_shifts, source, shifts)
    elif failure == "seam":
        source = eqx.tree_at(
            lambda value: value.core_seam_polygon_pairs, source, original_pairs[:-1]
        )
    elif failure == "transform":
        constraint = specification.periodic_constraints[0]
        transform = np.asarray(constraint.transform).copy()
        transform[0, 3] *= -1.0
        wrong = M.PeriodicConstraint(
            constraint.source_scope,
            constraint.target_scope,
            transform,
            source_entity_ids=constraint.source_entity_ids,
            orientations=constraint.orientations,
        )
        specification = eqx.tree_at(
            lambda value: value.periodic_constraints, specification, (wrong,)
        )
    elif failure == "work":
        specification = eqx.tree_at(
            lambda value: value.limits,
            specification,
            M.MeshingLimits(maximum_work_units=1),
        )
    else:
        specification = eqx.tree_at(
            lambda value: value.limits, specification, M.MeshingLimits(maximum_cells=1)
        )
    with pytest.raises((ValueError, M.MeshingFailure)):
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core")).plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        ).execute()
    np.testing.assert_array_equal(np.asarray(source.layers.mesh.coordinates), before)


def test_composite_material_map_cannot_swap_the_core_physical_identity() -> None:
    case = _case(material_interface=True)
    before = np.asarray(case.layers.mesh.coordinates).copy()
    with pytest.raises(ValueError, match="exact authoritative material names"):
        M.NativeLayerCoreSource(
            case.layers,
            case.core,
            "layered-square",
            "reference",
            vertex_layer_ids=case.vertex_layer_ids,
            cap_polygon_ids=case.cap_polygon_ids,
            layer_regions=case.layer_regions,
            region_ids=case.region_ids,
            core_region_map=np.asarray([0], dtype=np.int64),
        )
    np.testing.assert_array_equal(case.layers.mesh.coordinates, before)


def test_nonplanar_layer_source_transfer_rejects_changed_original_controls() -> None:
    import equinox as eqx

    from phydrax.meshing._association import (
        AssociationPropagationError,
        MappedReferenceAssociationTransfer,
    )
    from phydrax.meshing._association_composition import ComposedAssociationTransfer

    source = _periodic_material_source(ridge=True)
    specification = _periodic_material_specification(source)
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    transfer = M.prepare_native_layer_association_transfer(source, specification, result)
    before = np.asarray(result.mesh.coordinates).copy()
    classes = transfer.classes(result)
    boundary = np.asarray(
        next(
            subset.mask
            for subset in result.mesh.topology.entity_sets[2].subsets
            if subset.name == "boundary"
        ),
        dtype=np.bool_,
    )
    assert np.all(classes[2].dimensions[boundary] == 2)
    wrong: ComposedAssociationTransfer | MappedReferenceAssociationTransfer
    match transfer:
        case ComposedAssociationTransfer():
            layer_index = next(
                index
                for index, plan in enumerate(transfer.plans)
                if plan.domain.source_id == source.layers.result_id
            )
            original = transfer.reference_geometries[layer_index]
            if original is None:
                raise AssertionError(
                    "The nonplanar layer needs its actual mapped source expression."
                )
            coordinates = np.asarray(original.coordinates).copy()
            coordinates[:, 2] += 0.001
            changed = original.with_coordinates(coordinates)
            geometries = tuple(
                changed if index == layer_index else geometry
                for index, geometry in enumerate(transfer.reference_geometries)
            )
            wrong = eqx.tree_at(
                lambda value: value.reference_geometries, transfer, geometries
            )
        case MappedReferenceAssociationTransfer():
            coordinates = np.asarray(transfer.domain.source_geometry.coordinates).copy()
            coordinates[:, 2] += 0.001
            wrong = eqx.tree_at(
                lambda value: value.domain.source_geometry.coordinates,
                transfer,
                coordinates,
            )
    with pytest.raises((ValueError, AssociationPropagationError)):
        wrong.classes(result)
    np.testing.assert_array_equal(np.asarray(result.mesh.coordinates), before)


def _authored_columns(result: M.CellMeshingResult) -> M.MixedLayerColumns:
    validate_layer_index_attributes(result)
    attributes = {
        attribute.name: attribute
        for attribute in result.attributes
        if attribute.name in ("layer_index", "layer_column")
    }
    ids = np.asarray(attributes["layer_index"].scope.entity_ids, dtype=np.int64)
    intervals = np.asarray(attributes["layer_index"].values, dtype=np.int64)
    columns = np.asarray(attributes["layer_column"].values, dtype=np.int64)
    layered = intervals >= 0
    return M.MixedLayerColumns(
        ids[layered],
        columns[layered],
        intervals[layered],
        hard_first_thickness=True,
        axial_refinement=False,
        allow_schedule_change=False,
    )


def test_reopened_layer_core_lineage_continues_fields_volumes_history_and_pde(
    tmp_path: Path,
) -> None:
    import equinox as eqx

    from phydrax.lifecycle._meshing_source_families import native_family_audit_policy
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from tools.layer_core_lifecycle_continuation import (
        _LifecyclePlanCache,
        _require_restored_result_identity,
        accepted_lineage,
        continue_fields,
        continue_finite_volume,
        continue_history,
        material_declaration,
    )

    source = _periodic_material_source(ridge=True)
    specification = _periodic_material_specification(source)
    plan = M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core")).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    result = plan.execute()
    if result.certification is None:
        raise AssertionError("The accepted layer/core source must be certified.")
    records = {
        "certification_inputs": result.certification.request,
        "report": result.certification,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": plan.options,
        "generation_part": M.MeshPart("periodic-layer-core", result),
    }
    columns = _authored_columns(result)
    policy = M.MeshAdaptationPolicy(
        M.MeshAdaptationRoute.NATIVE_MIXED,
        limits=specification.limits,
        audit_policy=native_family_audit_policy(source, specification, result.mesh),
        association_transfer=M.prepare_native_layer_association_transfer(
            source, specification, result
        ),
    )
    refinement = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            result,
            M.MarkedMeshAdaptation(
                np.asarray((0,), dtype=np.int64), layer_columns=columns
            ),
            policy=policy,
        )
    )
    assert refinement.status is M.MeshAdaptationStatus.COMPLETE
    staged = write_meshing_source_closure(
        tmp_path / "refined",
        {**records, "accepted_data": {"adaptation_results": (refinement,)}},
    )
    (reopened,) = read_meshing_source_closure(
        staged.path, expected_content_id=staged.content_id
    )["accepted_data"]["adaptation_results"]
    assert reopened.result_id == refinement.result_id
    active = set(
        np.concatenate(
            [np.asarray(block.global_ids) for block in reopened.target.mesh.blocks]
        ).tolist()
    )
    children = sorted(
        {
            child
            for record in reopened.hierarchy.records
            for child in record.child_ids
            if child in active
        }
    )
    coarsening = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            reopened.target,
            M.MarkedMeshAdaptation(
                (), np.asarray(children, dtype=np.int64), hierarchy=reopened.hierarchy
            ),
            policy=policy,
        )
    )
    assert coarsening.status is M.MeshAdaptationStatus.COMPLETE
    durable = write_meshing_source_closure(
        tmp_path / "lifecycle",
        {**records, "accepted_data": {"adaptation_results": (reopened, coarsening)}},
        retained_layer_core_plan=plan,
    )
    restored, refined, coarsened = accepted_lineage(
        read_meshing_source_closure(durable.path, expected_content_id=durable.content_id)
    )
    counts = [
        sum(block.cell_count for block in current.mesh.blocks)
        for current in (restored, refined.target, coarsened.target)
    ]
    root_count = sum(block.cell_count for block in result.mesh.blocks)
    # Lifecycle counts are relative to the accepted native root representation,
    # not the superseded pre-cutover cell presentation.
    assert counts == [root_count, root_count + 4, root_count]
    assert restored.mesh.mesh_id == result.mesh.mesh_id
    assert _authored_columns(coarsened.target).columns_id == columns.columns_id
    np.testing.assert_array_equal(
        coarsened.target.mesh.vertex_global_ids, restored.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(
        np.asarray(coarsened.target.mesh.coordinates).view(np.uint64),
        np.asarray(restored.mesh.coordinates).view(np.uint64),
    )
    # Coarsening retains its actual restriction provenance. Equal restored
    # coordinate maps are not an excuse to force historical geometry IDs.
    np.testing.assert_array_equal(
        np.asarray(coarsened.target.geometry.coordinates).view(np.uint64),
        np.asarray(restored.geometry.coordinates).view(np.uint64),
    )
    for original, current in zip(
        restored.geometry.geometry_dofs,
        coarsened.target.geometry.geometry_dofs,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    assert (
        coarsened.target.geometry.source_coordinates()
        == restored.geometry.source_coordinates()
    )
    _require_restored_result_identity(restored, coarsened.target)
    restored_report = coarsened.target.certification
    if restored_report is None:
        raise AssertionError("The restored result must retain source certification.")
    from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain
    from phydrax.meshing._certification_inputs import MeshCertificationInputs

    request = restored_report.request
    domain = request.domain
    if not isinstance(domain, PiecewiseLinearDomain):
        raise AssertionError("The direct layer/core fixture requires its PLC domain.")
    changed_vertices = np.asarray(domain.vertices).copy()
    nonzero = np.argwhere(changed_vertices != 0.0)[0]
    row, column = int(nonzero[0]), int(nonzero[1])
    changed_vertices[row, column] = np.nextafter(changed_vertices[row, column], np.inf)
    changed_domain = PiecewiseLinearDomain(
        changed_vertices,
        domain.facets,
        domain.facet_regions,
        domain.region_ids,
        source_id=domain.source_id,
    )
    changed_request = MeshCertificationInputs(
        coarsened.target.mesh,
        coarsened.target.geometry,
        request.schedule,
        domain=changed_domain,
        cell_regions=request.cell_regions,
        source=request.source,
        fidelity_tolerance=request.fidelity_tolerance,
        fidelity_sample_order=request.fidelity_sample_order,
        limits=request.limits,
        junction_vertices=request.junction_vertices,
        scoped_fidelity=request.scoped_fidelity,
    )
    changed_report = eqx.tree_at(
        lambda value: value.request, restored_report, changed_request
    )
    changed_result = eqx.tree_at(
        lambda value: value.certification, coarsened.target, changed_report
    )
    with pytest.raises(ValueError, match="source revision"):
        _require_restored_result_identity(restored, changed_result)
    plans = _LifecyclePlanCache()
    fields = continue_fields(restored, refined, coarsened, plans=plans)
    assert [summary["field"] for summary in fields] == ["H1", "DG", "Hcurl", "Hdiv"]
    material = material_declaration(restored)
    volumes = continue_finite_volume(
        restored,
        refined,
        coarsened,
        material=material,
        plans=plans,
    )
    coverage = result.certification.coverage
    if coverage is None or any(
        value is None for value in coverage.achieved_region_measures
    ):
        raise AssertionError(
            "Continued inventories need independently certified material measures."
        )
    np.testing.assert_allclose(
        volumes["region_inventories"],
        np.asarray(coverage.achieved_region_measures, dtype=np.float64) * (2.0, 3.0),
        rtol=1e-12,
    )
    history = continue_history(
        restored,
        refined,
        coarsened,
        material=material,
        plans=plans,
    )
    assert [stage["phase"] for stage in history["stages"]] == ["refine", "coarsen"]
    # Two prism blocks with distinct declared discontinuous elements are a
    # block-specific field; regrouping them onto one restored prism block must
    # be refused, never aliased to one per-family field identity.
    prism_blocks = [
        block.name for block in refined.target.mesh.blocks if block.cell_kind == "prism"
    ]
    assert len(prism_blocks) > 1
    heterogeneous = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.discontinuous_element(
                block.cell_kind, 1 if block.name in prism_blocks[1:] else 2
            )
            for block in refined.target.mesh.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(
        refined.target.mesh, heterogeneous, coordinate_spec=refined.target.geometry
    ).prepare()
    accepted = phx.solver.FiniteElementAcceptedState(
        (space.project("u", lambda points, args: points[..., 1]),),
        0.0,
        0,
        refined.target.mesh.topology_id,
        space.prepared_id,
        "block-specific-declaration",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda *arguments: True, fields=(heterogeneous,)
    )
    with pytest.raises(ValueError, match="unambiguous field-family"):
        transaction.execute(accepted, refined.target.mesh, coarsened)


def test_original_open_wall_layers_cover_every_declared_cube_side() -> None:
    from tests.unit.meshing.test_meshing_source_authority import _cube_domain, _cube_query

    case = _case()
    domain = _cube_domain()
    query = _cube_query(domain)
    boundary = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(6, dtype=np.int64),
    )
    wall_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray([0], dtype=np.int64),
    )
    volume = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        domain.entity_set_id(3),
        np.asarray([0], dtype=np.int64),
    )
    wall = case.layers.source_wall
    wall_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    association = M.GeometryAssociation(
        M.GeometryAssociationKind.SURFACE,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        wall_ids,
        (f"{domain.source_revision}:surface:0",) * wall_ids.size,
        np.zeros(wall_ids.size, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(wall_ids.size, 2, dtype=np.int8),
        source_indices=np.zeros(wall_ids.size, dtype=np.int64),
        orientations=-np.ones(wall_ids.size, dtype=np.int8),
    )
    control = M.BoundaryLayerControl(
        wall_scope,
        case.layers.control.schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume,
    )
    layers = M.prepare_boundary_layers(
        wall, control, wall_association=association, source_domain=domain
    )
    core = M.PiecewiseLinearComplex(
        case.core.vertices,
        tuple(case.core.polygon_vertices.reshape(-1, 3)),
        np.repeat(np.arange(6, dtype=np.int64), 2),
        np.asarray(((0, -1), *[(-1, 0)] * 5), dtype=np.int64),
        case.core.region_ids,
        boundary="fixed",
    )
    source = M.NativeLayerCoreSource(
        layers,
        core,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=case.vertex_layer_ids,
        cap_polygon_ids=case.cap_polygon_ids,
        layer_regions=case.layer_regions,
        region_ids=case.region_ids,
        core_region_map=case.core_region_map,
        source_boundary_scope=boundary,
        source_domain=domain,
        fidelity_source=query,
        core_facet_source_ids=np.asarray((-1, 1, 2, 3, 4, 5), dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        layer_controls=(control,),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    report = result.certification
    if report is None or report.coverage is None:
        raise AssertionError(
            "The original open-wall layer source requires complete volume and scoped original-face evidence."
        )
    assert report.passed and len(report.scoped_fidelity) == 6
    assert all(
        certificate.status == "certified" for certificate in report.scoped_fidelity
    )
    assert all(
        certificate.mesh_to_source_upper <= 1e-12
        and certificate.source_to_mesh_upper <= 1e-12
        for certificate in report.scoped_fidelity
    )
    measures = report.coverage.achieved_region_measures
    if any(value is None for value in measures):
        raise AssertionError(
            "Original volume coverage must retain every independently measured material volume."
        )
    np.testing.assert_allclose(np.asarray(measures, dtype=np.float64), (1.0,), rtol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(result.mesh.coordinates)[: layers.mesh.coordinates.shape[0]],
        layers.mesh.coordinates,
    )


def test_layer_source_work_refusal_cannot_refund_consumed_budget() -> None:
    from phydrax.meshing._layer_core_resources import LayerCoreSourceWork
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    with native_volume_execution_budget(
        M.MeshingLimits(maximum_work_units=5)
    ) as execution:
        work = LayerCoreSourceWork(5)
        work.charge(3)
        with pytest.raises(ValueError):
            work.charge(-1)
        assert work.work_units == 3
        with pytest.raises(M.MeshingFailure):
            work.charge(3)
        assert work.work_units == 3
        work.charge(2)
        assert work.work_units == 5
    assert execution.evidence is not None
    assert execution.evidence.externally_charged_work == 5
    assert int(execution.evidence.work_evidence[1]) == 0


def test_standalone_prepared_layer_route_retains_actual_preparation_receipt() -> None:
    from phydrax.meshing.providers._native_layer import (
        execute_layer_route,
        PreparedLayerCore,
    )

    source = _periodic_material_source()
    specification = _periodic_material_specification(source)
    contract = phx.SpatialCoordinateContract.si()
    plan = M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core")).plan(
        source,
        specification,
        coordinate_contract=contract,
    )
    if not isinstance(plan.prepared, PreparedLayerCore):
        raise AssertionError("Layer/core planning must publish PreparedLayerCore.")
    prepared = plan.prepared
    original = np.asarray(source.layers.mesh.coordinates).view(np.uint64).copy()
    with pytest.raises(ValueError, match="actual preparation receipt"):
        execute_layer_route(
            source,
            specification,
            prepared,
            contract,
            M.NativeMeshingProvider.info(),
            plan.plan_id,
        )
    np.testing.assert_array_equal(
        np.asarray(source.layers.mesh.coordinates).view(np.uint64),
        original,
    )
    result = execute_layer_route(
        source,
        specification,
        prepared,
        contract,
        M.NativeMeshingProvider.info(),
        plan.plan_id,
        preparation_evidence=plan.preparation_evidence,
    )
    receipt = result.execution_evidence
    assert receipt is not None and receipt.preparation_evidence is not None
    receipt.require_valid()
    assert receipt.owner_id == plan.plan_id
    assert plan.preparation_evidence is not None
    assert (
        receipt.preparation_evidence.to_record() == plan.preparation_evidence.to_record()
    )
    assert result.certification is not None and result.certification.passed


@pytest.mark.parametrize("material_interface", (False, True))
def test_material_continuation_uses_only_declared_source_regions(
    material_interface: bool,
) -> None:
    from tools.layer_core_lifecycle_continuation import (
        _LifecyclePlanCache,
        _material_values,
        _measures,
        _region_inventory,
        _region_labels,
        material_declaration,
    )

    result = _result(_case(material_interface=material_interface))
    densities = (2.0, 3.0) if material_interface else (2.0,)
    material = material_declaration(
        result, densities=densities, site_id="declared-fluid-density"
    )
    actual = _material_values(result, material)
    np.testing.assert_array_equal(actual, densities)
    labels = _region_labels(result)
    inventory = _region_inventory(
        jnp.asarray(_measures(result, plans=_LifecyclePlanCache()) * actual[labels]),
        labels,
        len(densities),
    )
    expected = (0.4, 2.4) if material_interface else (2.0,)
    np.testing.assert_allclose(inventory, expected, rtol=1.0e-12, atol=1.0e-12)
    with pytest.raises(ValueError, match="original source or region namespace"):
        _material_values(result, {**material, "source_id": "foreign-original-source"})
    with pytest.raises(ValueError, match="exactly the original regions"):
        material_declaration(
            result, densities=(*densities, 4.0), site_id="declared-fluid-density"
        )


def test_live_layer_preparation_requires_exact_same_root_handle() -> None:
    from phydrax.meshing._volume_generation import (
        _native_live_preparation_is_active,
        native_volume_execution_budget,
    )
    from phydrax.meshing.providers._native_layer import (
        execute_layer_route,
        PreparedLayerCore,
    )

    source = _periodic_material_source()
    specification = _periodic_material_specification(source)
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
    contract = phx.SpatialCoordinateContract.si()
    historical = provider.plan(source, specification, coordinate_contract=contract)
    if not isinstance(historical.prepared, PreparedLayerCore):
        raise AssertionError("Layer/core planning must publish PreparedLayerCore.")
    historical_prepared = historical.prepared
    with native_volume_execution_budget(specification.limits):
        current = provider.plan(source, specification, coordinate_contract=contract)
        if not isinstance(current.prepared, PreparedLayerCore):
            raise AssertionError("Layer/core planning must publish PreparedLayerCore.")
        current_prepared = current.prepared
        assert _native_live_preparation_is_active(current_prepared)
        assert not _native_live_preparation_is_active(historical_prepared)
        with native_volume_execution_budget(specification.limits, borrow_active=False):
            assert _native_live_preparation_is_active(current_prepared)
            with pytest.raises(ValueError, match="exact live prepared owner"):
                execute_layer_route(
                    source,
                    specification,
                    historical_prepared,
                    contract,
                    provider.info(),
                    historical.plan_id,
                )
    assert not _native_live_preparation_is_active(current_prepared)


def test_closed_layer_preparation_scope_releases_its_strong_owner() -> None:
    import gc
    import weakref

    from phydrax.meshing._volume_generation import native_volume_execution_budget
    from phydrax.meshing.providers._native_layer import PreparedLayerCore

    source = _periodic_material_source()
    specification = _periodic_material_specification(source)

    def prepare() -> weakref.ReferenceType[PreparedLayerCore]:
        with native_volume_execution_budget(specification.limits):
            candidate = (
                M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
                .plan(
                    source,
                    specification,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                )
                .prepared
            )
            if not isinstance(candidate, PreparedLayerCore):
                raise AssertionError(
                    "Layer/core planning must publish PreparedLayerCore."
                )
            reference = weakref.ref(candidate)
        return reference

    reference = prepare()
    gc.collect()
    assert reference() is None


def test_refused_layer_preparation_scope_releases_its_strong_owner() -> None:
    import gc
    import weakref

    from phydrax.meshing._volume_generation import native_volume_execution_budget
    from phydrax.meshing.providers._native_layer import PreparedLayerCore

    source = _periodic_material_source()
    specification = _periodic_material_specification(source)

    def prepare() -> weakref.ReferenceType[PreparedLayerCore]:
        with pytest.raises(M.MeshingFailure):
            with native_volume_execution_budget(specification.limits) as execution:
                candidate = (
                    M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
                    .plan(
                        source,
                        specification,
                        coordinate_contract=phx.SpatialCoordinateContract.si(),
                    )
                    .prepared
                )
                if not isinstance(candidate, PreparedLayerCore):
                    raise AssertionError(
                        "Layer/core planning must publish PreparedLayerCore."
                    )
                reference = weakref.ref(candidate)
                execution.charge(work=specification.limits.maximum_work_units + 1)
        return reference

    reference = prepare()
    gc.collect()
    assert reference() is None
