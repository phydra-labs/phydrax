from copy import copy

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry._planar_embedding import PlanarEmbedding
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._organization import MeshLabel, MeshPatch, RegionBoundaryEvidence
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope


def test_native_refinement_preserves_world_frame_and_oriented_source_boundary() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "triangle",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((41,), dtype=np.int64),
            ),
        ),
    )
    boundary_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        mesh.entity_set(1).entity_set_id,
        mesh.entity_set(1).entity_ids,
    )
    source_scope = MeshingScope(
        "authored-sheet",
        "sheet-revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "authored-regions",
        np.asarray((7,), dtype=np.int64),
    )
    patch = MeshPatch(
        "wall",
        boundary_scope,
        source_adjacent_region_ids=("sheet-region", None),
        source_adjacent_region_occurrence_paths=(("sheet-occurrence",), None),
    )
    label = MeshLabel("region-boundary", boundary_scope)
    boundary = RegionBoundaryEvidence(
        mesh,
        source_scope,
        "sheet-region",
        "region-control",
        None,
        None,
        label,
        ((patch.patch_id, 1),),
        source_occurrence_path=("sheet-occurrence",),
    )
    embedding = PlanarEmbedding(
        (3.0, 4.0, 5.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    source = certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        patches=(patch,),
        labels=(label,),
        region_boundary_evidence=(boundary,),
        planar_embedding=embedding,
    )
    prepared = prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(np.asarray((41,), dtype=np.int64)),
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
    )

    target = execute_mesh_adaptation(prepared, prove=True).target

    assert target.planar_embedding == embedding
    assert target.planar_embedding is not None
    np.testing.assert_allclose(
        target.planar_embedding.to_world(target.mesh.coordinates)[:, 2], 5.0
    )
    assert target.patches[0].source_adjacent_region_ids == ("sheet-region", None)
    assert target.patches[0].source_adjacent_region_occurrence_paths == (
        ("sheet-occurrence",),
        None,
    )
    (evidence,) = target.region_boundary_evidence
    assert evidence.source_scope.scope_id == source_scope.scope_id
    assert evidence.source_region_id == "sheet-region"
    assert evidence.source_occurrence_path == ("sheet-occurrence",)
    assert evidence.patch_sides == ((target.patches[0].patch_id, 1),)
    evidence.require_current(target.mesh, target.labels, target.patches)
    edges = target.mesh.entity_set(1)
    np.testing.assert_array_equal(
        target.labels[0].scope.entity_ids,
        np.sort(np.asarray(edges.entity_ids[edges.subset("boundary").mask])),
    )
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_bisection_preserves_shared_and_independent_scoped_fidelity() -> None:
    from phydrax.discretization._surface_chart_records import SurfaceChartWitness
    from phydrax.geometry._source_fidelity import certify_source_fidelity
    from phydrax.meshing._bisection import BisectionCompatibility
    from phydrax.meshing._certification_inputs import (
        MeshCertificationInputs,
        ScopedFidelityInput,
    )
    from phydrax.meshing._surface_association_transfer import SurfaceAssociationTransfer
    from tests.unit.meshing.test_transition_acceptance import tiny_chart_source

    domain, initial = tiny_chart_source()
    assert initial.certification is not None
    original = initial.certification.inputs
    query, witness = original.source, original.fidelity_witness
    assert query is not None and isinstance(witness, SurfaceChartWitness)
    distinct = copy(witness)

    def inputs(scoped: tuple[ScopedFidelityInput, ...]) -> MeshCertificationInputs:
        return MeshCertificationInputs(
            initial.mesh,
            initial.geometry,
            original.schedule,
            source_id=original.source_id,
            source_revision=original.source_revision,
            domain=original.domain,
            cell_regions=original.cell_regions,
            source=query,
            fidelity_tolerance=original.fidelity_tolerance,
            maximum_normal_deviation=original.maximum_normal_deviation,
            junction_vertices=original.junction_vertices,
            protected_features=original.protected_features,
            scoped_fidelity=scoped,
            fidelity_witness=witness,
            feature_source=original.feature_source,
            source_coverage_witness=original.source_coverage_witness,
        )

    forged = copy(witness)
    object.__setattr__(forged, "domain_id", "forged-domain")
    with pytest.raises(ValueError):
        inputs(((query, forged, 0.2),))

    source = certify_cell_mesh(
        initial.mesh,
        initial.coordinate_contract,
        geometry=initial.geometry,
        certification_inputs=inputs(((query, witness, 0.2), (query, distinct, 0.19))),
        boundary=initial.boundary,
        patches=initial.patches,
        zones=initial.zones,
        labels=initial.labels,
        attributes=initial.attributes,
        associations=initial.associations,
        surface_source=initial.surface_source,
        region_boundary_evidence=initial.region_boundary_evidence,
        planar_embedding=initial.planar_embedding,
    )
    result = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)
            ),
            policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=SurfaceAssociationTransfer(domain),
            ),
        ),
        prove=True,
    )
    target = result.target
    assert target.certification is not None
    target_inputs = target.certification.inputs
    assert len(target_inputs.scoped_fidelity) == 2
    assert len(target.certification.scoped_fidelity) == 2
    for (scoped_query, scoped_witness, tolerance), certificate in zip(
        target_inputs.scoped_fidelity, target.certification.scoped_fidelity, strict=True
    ):
        independent = certify_source_fidelity(
            target.mesh,
            target.geometry,
            scoped_query,
            tolerance=tolerance,
            witness=scoped_witness,
        )
        assert certificate.status == independent.status
        assert certificate.findings == independent.findings
        assert certificate.certificate_id == independent.certificate_id
