from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry._compartments import CompartmentMeshingSource
from phydrax.geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionPlan,
    LabelFieldSurfaceExtractionResult,
    MultiRegionSurfaceValidationPolicy,
)
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_compartment import (
    _normalize_label_field_source,
    prepare_native_compartment,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.threshold_dynamics import LabelFieldState


pytestmark = pytest.mark.meshcore


def _image_source() -> tuple[CompartmentMeshingSource, phx.SpatialCoordinateContract]:
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-lps",
        reference_frame="patient",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    asset = phx.imaging.MedicalImageAsset(
        "one-voxel-labels",
        "segmentation",
        np.ones((1, 1, 1), dtype=np.int16),
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "deid", "subject", "protocol", True, True, True
        ),
        (
            phx.qualification.ReferenceArtifactManifest(
                "synthetic-one-voxel",
                checksum_algorithm="sha256",
                checksum="2" * 64,
                size_bytes=1,
                license_id="synthetic",
                commercial_use_permitted=True,
                redistribution_permitted=True,
                training_use_permitted=True,
                export_permitted=True,
                export_classification="public",
                nondimensionalization={"length": 1.0},
                uncertainty={"value": 0.0},
                lineage_ids=("synthetic-one-voxel",),
            ),
        ),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="native-image-meshing-test",
        ),
    )
    labels = phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(
            "one-voxel-ontology",
            "synthetic",
            "1",
            (phx.imaging.LabelDefinition(1, "solid-label", "Solid"),),
        ),
    )
    compartments = (
        phx.geometry.CompartmentDefinition("solid", ("solid-label",), "solid"),
    )
    complex_ = phx.imaging.build_compartment_complex(labels, compartments, ())
    interfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)

    vertices = np.asarray(
        (
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
        ),
        dtype=np.float64,
    )
    oriented_quads = (
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
    )
    faces = np.asarray(
        tuple(
            triangle
            for quad in oriented_quads
            for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
        ),
        dtype=np.int64,
    )
    outer = phx.geometry.SurfaceModel.from_triangles(
        vertices,
        faces,
        phx.geometry.SurfaceMetadata(
            source_id="one-voxel-outer-boundary",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("native-image-route-test",),
        ),
    )
    return CompartmentMeshingSource(labels, complex_, outer, interfaces), contract


def test_native_image_route_constructs_labeled_tetrahedra_and_region() -> None:
    source, contract = _image_source()
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 1.0),),
        region_seeds=(
            RegionSeed(
                np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
                "solid",
                "solid-material",
                RegionRole.SOLID,
            ),
        ),
    )
    construction = prepare_native_compartment(
        source,
        specification,
        NativeMeshingOptions(
            "image_material_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        ),
        contract,
    ).execute()

    assert construction.mesh.blocks[0].cell_kind == "simplex:3"
    assert construction.mesh.blocks[0].arity == 4
    assert construction.certification.cell_regions is not None
    assert construction.certification.domain is not None
    boundary = construction.organization.boundary
    assert boundary is not None
    assert boundary.metadata.source_id == source.outer_surface.metadata.source_id
    assert source.outer_surface.model_id in boundary.metadata.provenance
    connectivity = construction.mesh.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    exterior = np.asarray(connectivity.entities[2])[
        np.asarray(connectivity.boundary_faces)
    ]
    assert np.array_equal(
        np.asarray(boundary.mesh.vertex_global_ids),
        np.unique(np.asarray(construction.mesh.vertex_global_ids)[exterior]),
    )
    assert {zone.name for zone in construction.organization.zones} == {"solid"}
    assert all(
        zone.region_role is RegionRole.SOLID for zone in construction.organization.zones
    )
    assert construction.compliance.passed, (
        f"{construction.compliance.issues}; requested={construction.compliance.requested}; "
        f"achieved={construction.compliance.achieved}"
    )

    edge_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(1).entity_set_id
    )
    assert edge_association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
    assert GeometrySourceEntityRole.EDGE in edge_association.source_roles
    assert GeometrySourceEntityRole.FACET in edge_association.source_roles


def _two_material_surface() -> tuple[
    LabelFieldSurfaceExtractionResult,
    phx.SpatialCoordinateContract,
]:
    contract = phx.SpatialCoordinateContract.si()
    resolution = 4
    spacing = 1.0 / resolution
    labels = (np.indices((resolution,) * 3)[0] >= resolution // 2).astype(np.int32)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] *= spacing
    affine[:3, 3] = 0.5 * spacing
    label_ids = ("material-a", "material-b")
    plan = LabelFieldSurfaceExtractionPlan(
        label_ids,
        index_to_world=affine,
        boundary_region_id="ambient",
        validation_policy=MultiRegionSurfaceValidationPolicy(profile="dry_foam"),
    )
    state = LabelFieldState(
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=label_ids,
        route_id="native-material-test-route",
        prepared_id="native-material-test-prepared",
        site_id="native-material-test-sites",
    )
    return plan.prepare(labels.shape).extract(state), contract


def test_native_label_surface_route_preserves_multimaterial_plc_identity() -> None:
    extraction, contract = _two_material_surface()
    assert extraction.accepted
    assert extraction.topology is not None
    normalization = _normalize_label_field_source(extraction)
    source = normalization.source
    scope = MeshingScope.whole(extraction, 2)
    selected_scope = MeshingScope.from_geometry(
        extraction, np.asarray(scope.entity_ids), dimension=2
    )
    assert selected_scope.source_id == scope.source_id
    assert selected_scope.source_revision == scope.source_revision
    assert selected_scope.entity_set_id == scope.entity_set_id
    np.testing.assert_array_equal(selected_scope.entity_ids, scope.entity_ids)
    scope = selected_scope
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.4),),
        region_seeds=(
            RegionSeed(
                np.asarray((0.25, 0.5, 0.5), dtype=np.float64),
                "material-a",
                "material-a",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.75, 0.5, 0.5), dtype=np.float64),
                "material-b",
                "material-b",
                RegionRole.SOLID,
            ),
        ),
    )
    construction = prepare_native_compartment(
        extraction,
        specification,
        NativeMeshingOptions(
            "plc_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        ),
        contract,
    ).execute()

    assert construction.certification.source_id == source.source_id
    assert construction.certification.source_revision == source.source_revision
    assert {zone.name for zone in construction.organization.zones} == {
        "material-a",
        "material-b",
    }
    source_region_keys = {
        region_id: source.source_entity_key_for_scope(3, int(entity_id))
        for region_id, entity_id in zip(
            source.complex.region_ids,
            np.asarray(source.complex.region_entity_ids, dtype=np.int64),
            strict=True,
        )
    }

    def unordered_pair(
        sides: tuple[tuple[str, tuple[str, ...]] | None, ...], /
    ) -> tuple[tuple[str, tuple[str, ...]] | None, ...]:
        return tuple(
            sorted(
                sides,
                key=lambda side: (
                    side is not None,
                    "" if side is None else side[0],
                    () if side is None else side[1],
                ),
            )
        )

    expected_pairs = {
        unordered_pair(
            tuple(
                None
                if region_id == normalization.boundary_region_id
                else source_region_keys[region_id]
                for region_id in pair
            )
        )
        for pair in normalization.facet_source_region_ids
    }
    actual_pairs = set()
    for patch in construction.organization.patches:
        region_ids = patch.source_adjacent_region_ids
        occurrence_paths = patch.source_adjacent_region_occurrence_paths
        if region_ids is None or occurrence_paths is None:
            raise AssertionError(
                "A source PLC patch lost its region identity or occurrence."
            )
        sides = []
        for region_id, occurrence_path in zip(region_ids, occurrence_paths, strict=True):
            if region_id is None:
                if occurrence_path is not None:
                    raise AssertionError(
                        "An exterior PLC side cannot carry a source occurrence."
                    )
                sides.append(None)
            elif occurrence_path is None:
                raise AssertionError(
                    "An authored PLC region side lost its occurrence path."
                )
            else:
                sides.append((region_id, occurrence_path))
        actual_pairs.add(unordered_pair(tuple(sides)))
    assert actual_pairs == expected_pairs
    facet_source_keys = {
        source.source_entity_key_for_scope(2, int(identifier))[0]
        for identifier in source.scope_entity_ids(2)
    }
    face_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(2).entity_set_id
    )
    assert facet_source_keys <= set(face_association.source_entity_ids)
    assert face_association.exact
