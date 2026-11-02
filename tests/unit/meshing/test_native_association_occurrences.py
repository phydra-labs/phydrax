from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    EntitySelection,
    SimplicialConnectivity,
)
from phydrax.geometry._cad_revision import AssociationStatus, CADSelectionSet, CADSelector
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.brep import brep_box, prepare_brep_projection
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._model import BRepEntityId, BRepGeometry, BRepModel
from phydrax.geometry.brep._partition import (
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
from phydrax.meshing import CellMeshingResult, PatchControl, RegionControl, RegionRole
from phydrax.meshing._association import (
    associate_mesh_entities,
    associate_mesh_vertices,
    AssociationPropagationPolicy,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._scope import MeshingScope


def test_surface_association_keeps_large_source_indices_and_occurrence_identity() -> None:
    source_index = 2**40
    paths = (("assembly", "left"), ("assembly", "right"))
    keys = tuple(
        source_entity_id(
            "surface", "revision", "surface", source_index, occurrence_path=path
        )
        for path in paths
    )
    association = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        "surface",
        "revision",
        "mesh-faces",
        np.asarray((7, 19), dtype=np.int64),
        keys,
        np.zeros(2, dtype=np.float64),
        source_dimensions=np.asarray((2, 2), dtype=np.int8),
        source_indices=np.asarray((source_index, source_index), dtype=np.int64),
        source_roles=(GeometrySourceEntityRole.FACET, GeometrySourceEntityRole.FACET),
        source_occurrence_paths=paths,
    )

    assert keys[0] != keys[1]
    rows = association.target_rows(np.asarray((19, 7), dtype=np.int64))
    assert tuple(association.source_entity_ids[int(row)] for row in rows) == keys[::-1]
    assert (
        tuple(association.source_occurrence_paths[int(row)] for row in rows)
        == paths[::-1]
    )
    np.testing.assert_array_equal(
        association.source_indices, (source_index, source_index)
    )


@pytest.mark.parametrize(
    "dimensions, indices",
    (
        (np.asarray((-1,), dtype=np.int8), np.asarray((0,), dtype=np.int64)),
        (np.asarray((2,), dtype=np.int8), np.asarray((2**63,), dtype=np.uint64)),
    ),
    ids=("unpaired-stratum", "unrepresentable-index"),
)
def test_surface_association_refuses_invalid_source_strata(
    dimensions: np.ndarray, indices: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            "surface",
            "revision",
            "mesh-faces",
            np.asarray((7,), dtype=np.int64),
            ("source-face",),
            np.zeros(1, dtype=np.float64),
            source_dimensions=dimensions,
            source_indices=indices,
        )


def _two_box_occurrences() -> BRepModel:
    source = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    placement = BRepAffinePlacement.from_matrix(
        (
            (1, 0, 0, Fraction(3, 2)),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    assembly = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", placement=placement, solid_index=0),
                ),
            ),
        ),
        solid_count=1,
    )
    original = source.geometry
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        vertex_exact_points=original.vertex_exact_points,
        vertex_roots=original.vertex_roots,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        edge_endpoint_roots=original.edge_endpoint_roots,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        coedge_endpoint_roots=original.coedge_endpoint_roots,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        assembly_graph=assembly,
    )
    return BRepModel(
        geometry=geometry,
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        trim_domains=source.trim_domains,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
    )


def test_native_associations_do_not_merge_reused_definition_occurrences() -> None:
    model = _two_box_occurrences()
    left = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64
    )
    points = np.concatenate((left, left + np.asarray((1.5, 0.0, 0.0), dtype=np.float64)))
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "faces", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
            ),
        ),
    )
    projection = prepare_brep_projection(model)
    policy = AssociationPropagationPolicy()
    vertices = associate_mesh_vertices(mesh, projection, policy=policy)
    faces = associate_mesh_entities(mesh, vertices, projection, 2, policy=policy)

    assert vertices.complete and faces.complete
    assert vertices.source_occurrence_paths == (
        (("pair", "left"),) * 3 + (("pair", "right"),) * 3
    )
    assert faces.source_occurrence_paths == (("pair", "left"), ("pair", "right"))
    np.testing.assert_array_equal(faces.source_indices, (0, 0))
    assert faces.source_entity_ids[0] != faces.source_entity_ids[1]


@pytest.mark.parametrize(
    ("foreign_source", "foreign_path"),
    (("different-source", ("left",)), ("source", ("right",))),
    ids=("source-owner", "occurrence"),
)
def test_unclassified_brep_rows_remain_bound_to_source_occurrence(
    foreign_source: str, foreign_path: tuple[str, ...]
) -> None:
    key = source_entity_id(
        "source",
        "revision",
        "unclassified",
        "unclassified",
        occurrence_path=("left",),
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        "source",
        "revision",
        "vertices",
        np.asarray((7,), dtype=np.int64),
        (key,),
        np.asarray((np.inf,), dtype=np.float64),
        resolved=np.asarray((False,), dtype=np.bool_),
        source_dimensions=np.asarray((-1,), dtype=np.int8),
        source_indices=np.asarray((-1,), dtype=np.int64),
        source_occurrence_paths=(("left",),),
    )
    assert not association.complete
    assert association.source_entity_ids == (key,)
    foreign = source_entity_id(
        foreign_source,
        "revision",
        "unclassified",
        "unclassified",
        occurrence_path=foreign_path,
    )
    with pytest.raises(ValueError):
        GeometryAssociation(
            GeometryAssociationKind.BREP,
            "source",
            "revision",
            "vertices",
            np.asarray((7,), dtype=np.int64),
            (foreign,),
            np.asarray((np.inf,), dtype=np.float64),
            resolved=np.asarray((False,), dtype=np.bool_),
            source_dimensions=np.asarray((-1,), dtype=np.int8),
            source_indices=np.asarray((-1,), dtype=np.int64),
            source_occurrence_paths=(("left",),),
        )


def test_cad_scopes_preserve_occurrences_and_refuse_ambiguous_definitions() -> None:
    model = _two_box_occurrences()
    domain = MeshingDomain.from_brep(model)
    from phydrax.meshing.providers._native_cad_volume import _domain_scope_rows

    left_entity = BRepEntityId(
        model.source_id, model.source_revision, "face", 0, ("pair", "left")
    )
    right_entity = BRepEntityId(
        model.source_id, model.source_revision, "face", 0, ("pair", "right")
    )
    left = MeshingScope.from_geometry(model, left_entity)
    right = MeshingScope.from_geometry(model, right_entity)
    assert not np.intersect1d(
        np.asarray(left.entity_ids), np.asarray(right.entity_ids)
    ).size
    for scope, entity in ((left, left_entity), (right, right_entity)):
        keys = tuple(
            domain.source_entity_key_for_scope(2, int(identifier))
            for identifier in np.asarray(scope.entity_ids)
        )
        assert keys == (
            (
                source_entity_id(
                    entity.source_id,
                    entity.source_revision,
                    entity.kind,
                    entity.index,
                    occurrence_path=entity.occurrence_path,
                ),
                entity.occurrence_path,
            ),
        )
        rows = _domain_scope_rows(domain, scope, 2)
        assert len(rows) == 1
        assert domain.source_indices[2][int(rows[0])] == entity.index
        assert domain.source_occurrences[2][int(rows[0])] == entity.occurrence_path
    whole = MeshingScope.whole(model, 2)
    np.testing.assert_array_equal(whole.entity_ids, domain.scope_entity_ids(2))
    assert whole.entity_set_id == domain.scope_entity_set_id(2)
    assert set(domain.source_occurrences[2]) == {("pair", "left"), ("pair", "right")}
    with pytest.raises(ValueError):
        MeshingScope.from_geometry(model, model.face_ids[0])


def test_cad_scope_rejects_a_foreign_owner_with_the_same_revision() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    selected = MeshingScope.from_geometry(model, model.face_ids[0])
    assert selected.source_id == model.source_id
    foreign = BRepEntityId("foreign-owner", model.source_revision, "face", 0)
    with pytest.raises(ValueError):
        MeshingScope.from_geometry(model, foreign)


def test_cad_selection_scope_authenticates_occurrences_and_revision() -> None:
    model = _two_box_occurrences()
    revision = cad_revision_from_brep_model(model)
    face_entities = (
        BRepEntityId(model.source_id, model.source_revision, "face", 0, ("pair", "left")),
        BRepEntityId(
            model.source_id, model.source_revision, "face", 0, ("pair", "right")
        ),
    )
    from phydrax.geometry.brep._partition import _entity_occurrence_id

    selections = tuple(
        CADSelectionSet.from_revision(revision, (_entity_occurrence_id(entity),))
        for entity in face_entities
    )
    scopes = tuple(MeshingScope.from_geometry(model, value) for value in selections)
    for scope, entity in zip(scopes, face_entities, strict=True):
        expected = MeshingScope.from_geometry(model, entity)
        assert scope.scope_id == expected.scope_id
    assert scopes[0].scope_id != scopes[1].scope_id
    patch = PatchControl("wall", scopes[0], ("region",))
    assert patch.scope.scope_id == scopes[0].scope_id
    combined = CADSelectionSet(
        revision.revision_id, tuple(s.selectors[0] for s in selections)
    )
    np.testing.assert_array_equal(
        MeshingScope.from_geometry(model, combined).entity_ids,
        scopes[0].union(scopes[1]).entity_ids,
    )
    selector = selections[0].selectors[0]
    forged = CADSelector(
        selector.revision_id,
        selector.occurrence_id,
        selector.kind,
        selector.path,
        "foreign-entity",
    )
    with pytest.raises(ValueError, match="inventory"):
        MeshingScope.from_geometry(
            model, CADSelectionSet(revision.revision_id, (forged,))
        )
    with pytest.raises(ValueError, match="revision"):
        MeshingScope.from_geometry(model, CADSelectionSet("foreign-revision", ()))
    with pytest.raises(ValueError, match="at least one"):
        MeshingScope.from_geometry(model, CADSelectionSet(revision.revision_id, ()))
    with pytest.raises(ValueError, match="dimension"):
        MeshingScope.from_geometry(model, selections[0], dimension=3)


def test_shared_face_cad_sides_require_complete_representable_scope() -> None:
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from phydrax.geometry.brep._partition import _entity_occurrence_id
    from phydrax.interchange import import_occt_shape

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        )
    splitter.Perform()
    assert not splitter.HasErrors()
    model = import_occt_shape(
        splitter.Shape(), coordinate_contract=SpatialCoordinateContract.si()
    ).model
    shared = tuple(
        index
        for index, owners in enumerate(model.topology.face_solids)
        if len(owners) == 2
    )
    assert len(shared) == 1
    revision = cad_revision_from_brep_model(model)
    members = {
        record.member
        for record in model.geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
        if record.member.kind == "face" and record.member.index == shared[0]
    }
    assert len(members) == 2
    selectors = tuple(
        revision.select(_entity_occurrence_id(member)) for member in members
    )
    assert len({selector.path for selector in selectors}) == 2
    for selector in selectors:
        with pytest.raises(ValueError, match="cannot represent only one occurrence side"):
            MeshingScope.from_geometry(
                model, CADSelectionSet(revision.revision_id, (selector,))
            )

    selection = CADSelectionSet(revision.revision_id, selectors)
    scope = MeshingScope.from_geometry(model, selection)
    domain = MeshingDomain.from_brep(model)
    rows = tuple(
        row for row, index in enumerate(domain.source_indices[2]) if index == shared[0]
    )
    assert len(rows) == 1
    from phydrax.meshing.providers._native_cad_volume import _domain_scope_rows

    np.testing.assert_array_equal(_domain_scope_rows(domain, scope, 2), rows)
    assert domain.source_occurrences[2][rows[0]] in {
        member.occurrence_path for member in members
    }
    assert (
        PatchControl("interface", scope, ("left", "right")).scope.scope_id
        == scope.scope_id
    )

    original = model.geometry
    assembly = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("first-0", solid_index=0),
            BRepAssemblyUse("second-0", solid_index=0),
            BRepAssemblyUse("second-1", solid_index=1),
            BRepAssemblyUse("first-1", solid_index=1),
        ),
        definitions=(),
        solid_count=2,
    )
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        vertex_exact_points=original.vertex_exact_points,
        vertex_roots=original.vertex_roots,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        edge_endpoint_roots=original.edge_endpoint_roots,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        coedge_endpoint_roots=original.coedge_endpoint_roots,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        assembly_graph=assembly,
    )
    repeated = BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        trim_domains=model.trim_domains,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
    )
    repeated_revision = cad_revision_from_brep_model(repeated)
    repeated_members = {
        record.member
        for record in geometry.qualified_entity_incidence(
            repeated.source_id, repeated.source_revision
        )
        if record.member.kind == "face" and record.member.index == shared[0]
    }
    assert len(repeated_members) == 4
    repeated_selection = CADSelectionSet.from_revision(
        repeated_revision,
        tuple(_entity_occurrence_id(member) for member in repeated_members),
    )
    with pytest.raises(ValueError, match="ambiguous coincident assembly copies"):
        MeshingScope.from_geometry(repeated, repeated_selection)


def test_partition_descendants_bind_region_control_exactly() -> None:
    coordinates = SpatialCoordinateContract.si()
    region = brep_box((0.0, 0.0, 0.0), (3.0, 1.0, 1.0), coordinate_contract=coordinates)
    void = brep_box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0), coordinate_contract=coordinates)

    def operand(
        name: str,
        model: BRepModel,
        role: BRepPartitionRole,
        targets: tuple[str, ...] = (),
    ) -> BRepPartitionOperand:
        revision = cad_revision_from_brep_model(model)
        selection = CADSelectionSet.from_revision(
            revision,
            tuple(o.occurrence_id for o in revision.occurrences if o.kind == "solid"),
        )
        return BRepPartitionOperand(name, model, role, selection, targets)

    result = partition_brep(
        BRepPartitionPlan(
            coordinates,
            (
                operand("region", region, BRepPartitionRole.REGION),
                operand("void", void, BRepPartitionRole.VOID, ("region",)),
            ),
            BRepPartitionPolicy(("region",)),
        )
    )
    graph = result.association_graph
    source = CADSelectionSet(
        graph.source_revision.revision_id,
        (graph.source_revision.select("operand:region:solid:0"),),
    )
    descendants = graph.resolve_target_selection(source)
    scope = MeshingScope.from_geometry(result.model, descendants)
    expected = MeshingScope.from_geometry(
        result.model, result.region("region").entity_ids
    )
    assert scope.scope_id == expected.scope_id
    control = RegionControl(scope, "region", "material", RegionRole.SOLID)
    assert control.scope.scope_id == expected.scope_id
    face_resolution = next(
        resolution
        for occurrence in graph.source_revision.occurrences
        if occurrence.kind == "face"
        and occurrence.occurrence_id.startswith("operand:region:")
        for resolution in (
            graph.resolve_target(graph.source_revision.select(occurrence.occurrence_id)),
        )
        if resolution.status in (AssociationStatus.UNIQUE, AssociationStatus.MULTIPLE)
        and any(candidate.kind == "face" for candidate in resolution.candidate_selectors)
        and all(
            {
                selector.occurrence_id
                for selector in graph.resolve_preimage(candidate).candidate_selectors
            }
            == {occurrence.occurrence_id}
            for candidate in resolution.candidate_selectors
            if candidate.kind == "face"
        )
    )
    faces = graph.resolve_target_selection(
        CADSelectionSet(
            graph.source_revision.revision_id, (face_resolution.queried_selector,)
        )
    )
    boundary_scope = MeshingScope.from_geometry(result.model, faces)
    boundary = PatchControl("descendant-wall", boundary_scope, ("region",))
    assert boundary.scope.entity_dimension == 2
    assert boundary.scope.entity_ids.size == len(faces.selectors)
    with pytest.raises(ValueError, match="revision"):
        MeshingScope.from_geometry(region, descendants)


def test_transported_face_drives_native_mesh_and_fe_dirichlet_constraint() -> None:
    contract = SpatialCoordinateContract.si()
    model = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    revision = cad_revision_from_brep_model(model)
    solid = CADSelectionSet.from_revision(
        revision,
        tuple(o.occurrence_id for o in revision.occurrences if o.kind == "solid"),
    )
    partition = partition_brep(
        BRepPartitionPlan(
            contract,
            (BRepPartitionOperand("region", model, BRepPartitionRole.REGION, solid),),
            BRepPartitionPolicy(("region",)),
        )
    )
    graph = partition.association_graph
    source_face = next(o for o in graph.source_revision.occurrences if o.kind == "face")
    descendants = graph.resolve_target_selection(
        CADSelectionSet.from_revision(graph.source_revision, (source_face.occurrence_id,))
    )
    wall = MeshingScope.from_geometry(partition.model, descendants)
    boundary = MeshingScope.whole(partition.model, 2)
    region_scope = MeshingScope.from_geometry(
        partition.model, partition.region("region").entity_ids
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(boundary, 0.5),),
        region_controls=(
            RegionControl(region_scope, "region", "material", RegionRole.SOLID),
        ),
        patch_controls=(PatchControl("wall", wall, ("region",)),),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider().plan(partition.model, specification).execute()
    )
    patch = next(value for value in result.patches if value.name == "wall")
    selection = phx.meshing.resolve_mesh_scope(result.mesh, patch.scope)
    _assert_fe_fv_boundary_conditions(result, selection)


def _assert_fe_fv_boundary_conditions(
    result: CellMeshingResult, selection: EntitySelection
) -> None:
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    selected_faces = np.flatnonzero(np.asarray(selection.mask))
    assert selected_faces.size > 0
    expected_vertices = np.unique(np.asarray(connectivity.entities[2])[selected_faces])
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("simplex:3", 1)
    )
    prepared = phx.discretization.FiniteElementPlan(result.mesh, field).prepare()
    constraint = phx.discretization.dirichlet_constraint(
        prepared, "u", boundary_selection=selection
    )
    np.testing.assert_array_equal(constraint.constrained_dofs, expected_vertices)
    assert constraint.free_dofs.size > 0

    # FV reconstructs its face ordering; transfer the exact selected mesh faces
    # by their vertex incidence, not by assuming the two orderings coincide.
    mesh_faces = np.asarray(connectivity.entities[2])
    exterior = np.asarray(connectivity.boundary_masks[2], dtype=np.bool_)
    selected_mask = np.asarray(selection.mask, dtype=np.bool_)
    assert np.all(exterior[selected_mask])
    system = phx.equations.TwoMaterialVOFSystem(
        3,
        eos=phx.equations.TwoMaterialEOSClosure(
            phx.equations.IdealGasMaterial(1.4),
            phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
        ),
    )
    fv = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(result.mesh.coordinates),
        tetrahedra=np.asarray(result.mesh.blocks[0].vertices),
        vertex_global_ids=np.asarray(result.mesh.vertex_global_ids),
        cell_global_ids=np.asarray(result.mesh.blocks[0].global_ids),
        boundary_patches={
            "wall": mesh_faces[selected_mask],
            "other": mesh_faces[exterior & ~selected_mask],
        },
        component_names=system.component_names,
    ).prepare()
    selected_keys = {tuple(sorted(face)) for face in mesh_faces[selected_mask].tolist()}
    fv_faces = np.asarray(fv.connectivity.faces)
    wall_id = fv.boundary_patch_names.index("wall")
    fv_wall = np.asarray(fv.boundary_patch_ids) == wall_id
    assert {tuple(sorted(face)) for face in fv_faces[fv_wall].tolist()} == selected_keys
    primitive = np.tile(
        np.asarray((1.2, 0.7, 0.0, 0.0, 0.0, 2.5, 0.6)), (fv.cell_count, 1)
    )
    state = system.primitive_to_conserved(jnp.asarray(primitive))
    thermal = phx.discretization.UnstructuredTwoMaterialThermalDiffusionPlan(
        fv,
        2.0,
        0.5,
        boundaries={
            "wall": phx.discretization.UnstructuredThermalBoundaryCondition(
                "heat_flux", 5.0
            )
        },
    ).evaluate(system, state)
    assert thermal.successful
    np.testing.assert_allclose(
        thermal.boundary_energy_rate,
        -5.0 * np.sum(np.asarray(fv.face_measures)[fv_wall]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(thermal.conservation_defect, 0.0, atol=1e-10)


def test_split_face_descendants_drive_fe_and_fv_boundary_conditions() -> None:
    from phydrax.meshing.providers._native_cad_volume import _domain_scope_rows

    contract = SpatialCoordinateContract.si()
    region = brep_box((0.0, 0.0, 0.0), (3.0, 1.0, 1.0), coordinate_contract=contract)
    void = brep_box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0), coordinate_contract=contract)

    def operand(
        name: str,
        model: BRepModel,
        role: BRepPartitionRole,
        targets: tuple[str, ...] = (),
    ) -> BRepPartitionOperand:
        revision = cad_revision_from_brep_model(model)
        solids = CADSelectionSet.from_revision(
            revision,
            tuple(o.occurrence_id for o in revision.occurrences if o.kind == "solid"),
        )
        return BRepPartitionOperand(name, model, role, solids, targets)

    partition = partition_brep(
        BRepPartitionPlan(
            contract,
            (
                operand("region", region, BRepPartitionRole.REGION),
                operand("void", void, BRepPartitionRole.VOID, ("region",)),
            ),
            BRepPartitionPolicy(("region",)),
        )
    )
    graph = partition.association_graph
    resolution = next(
        value
        for occurrence in graph.source_revision.occurrences
        if occurrence.kind == "face"
        and occurrence.occurrence_id.startswith("operand:region:")
        for value in (
            graph.resolve_target(graph.source_revision.select(occurrence.occurrence_id)),
        )
        if sum(candidate.kind == "face" for candidate in value.candidate_selectors) == 2
        and all(
            {
                item.occurrence_id
                for item in graph.resolve_preimage(candidate).candidate_selectors
            }
            == {occurrence.occurrence_id}
            for candidate in value.candidate_selectors
            if candidate.kind == "face"
        )
    )
    descendants = graph.resolve_target_selection(
        CADSelectionSet(graph.source_revision.revision_id, (resolution.queried_selector,))
    )
    assert len(descendants.selectors) == 2
    domain = MeshingDomain.from_brep(partition.model)
    solid_entities = partition.region("region").entity_ids
    assert len(solid_entities) == 2
    region_controls = []
    names_by_row = {}
    for index, entity in enumerate(solid_entities):
        scope = MeshingScope.from_geometry(partition.model, entity)
        row = int(_domain_scope_rows(domain, scope, 3)[0])
        name = f"region-{index}"
        names_by_row[row] = name
        region_controls.append(RegionControl(scope, name, "material", RegionRole.SOLID))
    patch_controls = []
    for index, selector in enumerate(descendants.selectors):
        selection = CADSelectionSet(descendants.revision_id, (selector,))
        scope = MeshingScope.from_geometry(partition.model, selection)
        face_row = int(_domain_scope_rows(domain, scope, 2)[0])
        owners = [
            row
            for row, value in enumerate(domain.regions)
            if any(face == face_row for face, _ in value.boundary)
        ]
        assert len(owners) == 1
        patch_controls.append(
            PatchControl(f"wall-{index}", scope, (names_by_row[owners[0]],))
        )
    assert {control.adjacent_region_names for control in patch_controls} == {
        (name,) for name in names_by_row.values()
    }
    boundary = MeshingScope.whole(partition.model, 2)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        boundary,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(boundary, 0.5),),
        region_controls=tuple(region_controls),
        patch_controls=tuple(patch_controls),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider().plan(partition.model, specification).execute()
    )
    scopes = tuple(
        next(patch.scope for patch in result.patches if patch.name == control.name)
        for control in patch_controls
    )
    for scope in scopes:
        assert np.asarray(scope.entity_ids).size > 0
    assert not np.intersect1d(
        np.asarray(scopes[0].entity_ids), np.asarray(scopes[1].entity_ids)
    ).size
    selection = phx.meshing.resolve_mesh_scope(result.mesh, scopes[0].union(scopes[1]))
    _assert_fe_fv_boundary_conditions(result, selection)
