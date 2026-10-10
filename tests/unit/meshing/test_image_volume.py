import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._compartments import (
    prepare_compartment_complex,
    revalidate_region_evidence,
)
from tests._support.image_sources import _execute, _region_measure_array, _source


@pytest.mark.parametrize(
    "matrix",
    (
        np.asarray(
            (
                (1.0, 0.5, 0.25, 2.0),
                (0.0, 2.0, 0.5, -1.0),
                (0.0, 0.0, 0.5, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        np.asarray(
            (
                (-1.0, 0.0, 0.0, 2.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 2.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
    ),
)
def test_oblique_anisotropic_and_reflected_images_preserve_interface_and_measure(
    matrix: np.ndarray,
) -> None:
    values = np.ones((2, 2, 1), dtype=np.int16)
    values[1] = 2
    source = _source(values, matrix)
    result = _execute(source)
    evidence = result.region_evidence
    assert evidence is not None
    evidence.require_source(source.compartments)
    voxel_measure = abs(np.linalg.det(matrix[:3, :3]))
    np.testing.assert_allclose(
        _region_measure_array(evidence),
        np.asarray((2 * voxel_measure, 2 * voxel_measure), dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise TypeError("The image fixture must have tetrahedral incidence.")
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    ids = np.asarray(result.mesh.entity_set(2).entity_ids, dtype=np.int64)
    interface = np.asarray(
        [
            row
            for row, facet in enumerate(ids.tolist())
            if any(value[1] == facet for value in evidence.interface_facets)
        ],
        dtype=np.int64,
    )
    image_points = source.labels.asset.spatial_affine.world_to_index(
        np.asarray(result.mesh.coordinates)[np.unique(faces[interface])]
    )
    np.testing.assert_allclose(image_points[:, 0], 0.5, rtol=0.0, atol=1e-12)
    assert evidence.adjacency_pairs == (("region:1", "region:2"),)


def test_explicit_changed_image_source_revalidates_legal_material_translation() -> None:
    values = np.asarray((((1,),), ((2,),)), dtype=np.int16)
    source = _source(values)
    result = _execute(source)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (0.125, -0.25, 0.5)
    updated_source = _source(values, matrix)
    moved = result.mesh.with_coordinates(
        np.asarray(result.mesh.coordinates) + matrix[:3, 3],
        numeric_version="translated-image",
    )
    renewal = revalidate_region_evidence(
        result,
        moved,
        phx.discretization.CellGeometrySpec.affine(moved),
        (),
        (),
        compartment_source=updated_source,
    )
    renewal.region_evidence.require_source(updated_source.compartments)
    assert result.region_evidence is not None
    assert renewal.region_evidence.coverage_id != result.region_evidence.coverage_id
    assert (
        renewal.region_evidence.cell_region_ids == result.region_evidence.cell_region_ids
    )
    assert (
        renewal.region_evidence.adjacency_pairs == result.region_evidence.adjacency_pairs
    )


def test_image_outer_surface_cannot_silently_override_occupied_cell_extent() -> None:
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    outer = source.outer_surface
    changed = phx.geometry.SurfaceModel.from_triangles(
        np.asarray(outer.mesh.coordinates) * 2.0,
        np.asarray(outer.mesh.blocks[0].vertices, dtype=np.int64),
        outer.metadata,
    )
    with pytest.raises(phx.meshing.MeshingFailure, match="outer-source coverage"):
        _execute(replace(source, outer_surface=changed))


def test_finite_image_background_is_exterior_not_centroid_relabeling() -> None:
    source = _source(np.asarray((((0,),), ((1,),)), dtype=np.int16))
    definition = phx.geometry.CompartmentDefinition("region:1", ("label:1",), "material")
    complex_ = phx.imaging.build_compartment_complex(source.labels, (definition,), ())
    box = _source(np.ones((1, 1, 1), dtype=np.int16)).outer_surface
    outer = phx.geometry.SurfaceModel.from_triangles(
        np.asarray(box.mesh.coordinates) + np.asarray((1.0, 0.0, 0.0)),
        np.asarray(box.mesh.blocks[0].vertices, dtype=np.int64),
        box.metadata,
    )
    occupied = phx.geometry.CompartmentMeshingSource(
        source.labels,
        complex_,
        outer,
        phx.imaging.extract_compartment_surfaces(source.labels, complex_),
    )
    result = _execute(occupied)
    evidence = result.region_evidence
    assert evidence is not None
    assert set(evidence.cell_region_ids) == {"region:1"}
    assert evidence.adjacency_pairs == ()
    np.testing.assert_allclose(
        _region_measure_array(evidence),
        np.asarray((1.0,), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    points = np.asarray(result.mesh.coordinates)
    assert np.min(points[:, 0]) == 0.5
    assert np.max(points[:, 0]) == 1.5


def _categorical_extraction() -> (
    phx.geometry.multiregion_surface.LabelFieldSurfaceExtractionResult
):
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[1] = 1
    state = phx.threshold_dynamics.LabelFieldState(
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray((True, True), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=("left", "right"),
        route_id="categorical-image-test",
        prepared_id="categorical-image-prepared",
        site_id="categorical-image-sites",
    )
    capacity = phx.geometry.multiregion_surface.MultiRegionSurfaceCapacityPlan(
        vertex_capacity=4096,
        edge_capacity=8192,
        face_capacity=8192,
        region_capacity=4,
        region_pair_capacity=6,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=6,
        resource_id="categorical-image-volume",
    )
    extraction = (
        phx.geometry.multiregion_surface.LabelFieldSurfaceExtractionPlan(
            ("left", "right"), capacity, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )
        .prepare(labels.shape)
        .extract(state)
    )
    return extraction


def _image_source_cold_identity(source: object) -> dict[str, object]:
    if not isinstance(
        source, (phx.geometry.CompartmentMeshingSource, LabelFieldVolumeBinding)
    ):
        raise TypeError(
            "A cold image oracle requires the actual interpreted source owner."
        )
    source.validate_source_integrity()
    identity: dict[str, object] = {
        "type": f"{type(source).__module__}.{type(source).__qualname__}",
        "source": source.source_id,
        "revision": source.source_revision,
        "interpretation": source.interpretation,
        "coordinates": source.coordinate_contract.spatial_id,
        "interfaces": source.interface_definitions,
    }
    if isinstance(source, phx.geometry.CompartmentMeshingSource):
        identity.update(
            {
                "values": array_tree_fingerprint(source.labels.asset.values),
                "valid_mask": array_tree_fingerprint(source.labels.asset.valid_mask),
                "affine": array_tree_fingerprint(
                    source.labels.asset.spatial_affine.matrix
                ),
                "labels": source.labels.label_volume_id,
                "ontology": source.labels.ontology.ontology_content_id,
                "complex": source.compartments.complex_id,
                "extraction": source.interfaces.extraction_id,
                "outer": source.outer_surface.model_id,
                "complex_alias": source.interfaces.complex is source.compartments,
            }
        )
    else:
        identity.update(
            {
                "binding": source.binding_id,
                "extraction": array_tree_fingerprint(source.extraction),
                "domain": array_tree_fingerprint(source.domain),
                "lineage_alias": source.lineage is source.extraction.lineage,
                "evidence_alias": source.extraction_evidence
                is source.extraction.evidence,
            }
        )
    return identity


@pytest.mark.parametrize("interpretation", ("occupied", "reconstructed"))
def test_image_source_cold_restore_preserves_actual_interpretation_and_authority(
    tmp_path: Path,
    interpretation: str,
) -> None:
    if interpretation == "occupied":
        values = np.asarray((((1,), (2,)), ((3,), (3,))), dtype=np.int16)
        matrix = np.asarray(
            (
                (-1.0, 0.5, 0.25, 2.0),
                (0.0, 2.0, 0.5, -1.0),
                (0.0, 0.0, 0.5, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        source = _source(values, matrix)
    else:
        source = LabelFieldVolumeBinding(
            _categorical_extraction(), phx.SpatialCoordinateContract.si()
        )
    receipt = write_meshing_source_closure(tmp_path / "image-source", source)
    restored = read_meshing_source_closure(
        tmp_path / "image-source",
        expected_content_id=receipt.content_id,
    )
    assert type(restored) is type(source)
    restored.validate_source_integrity()
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.interpretation == source.interpretation
    assert restored.interface_definitions == source.interface_definitions
    assert (
        restored.coordinate_contract.spatial_id == source.coordinate_contract.spatial_id
    )
    if isinstance(source, phx.geometry.CompartmentMeshingSource):
        assert isinstance(restored, phx.geometry.CompartmentMeshingSource)
        assert restored.labels.asset.values.dtype == source.labels.asset.values.dtype
        assert restored.labels.asset.values.shape == source.labels.asset.values.shape
        assert (
            restored.labels.asset.values.tobytes() == source.labels.asset.values.tobytes()
        )
        assert (
            restored.labels.asset.spatial_affine.matrix.tobytes()
            == source.labels.asset.spatial_affine.matrix.tobytes()
        )
        assert restored.compartments.complex_id == source.compartments.complex_id
        assert restored.interfaces.extraction_id == source.interfaces.extraction_id
        assert restored.outer_surface.model_id == source.outer_surface.model_id
    else:
        assert isinstance(restored, LabelFieldVolumeBinding)
        assert restored.binding_id == source.binding_id
        assert array_tree_fingerprint(restored.extraction) == array_tree_fingerprint(
            source.extraction
        )
        assert array_tree_fingerprint(restored.domain) == array_tree_fingerprint(
            source.domain
        )
    script = """
import json
import sys
from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
from tests.unit.meshing.test_image_volume import _image_source_cold_identity
source = read_meshing_source_closure(sys.argv[1], expected_content_id=sys.argv[2])
print(json.dumps(_image_source_cold_identity(source), sort_keys=True))
"""
    cold = subprocess.run(
        (sys.executable, "-c", script, str(receipt.path), receipt.content_id),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads(cold.stdout) == json.loads(
        json.dumps(_image_source_cold_identity(source))
    )


def test_categorical_sample_reconstruction_fills_its_declared_interface_not_voxel_cells() -> (
    None
):
    extraction = _categorical_extraction()
    assert extraction.accepted
    source = LabelFieldVolumeBinding(extraction, phx.SpatialCoordinateContract.si())
    topology, geometry_state = extraction.topology, extraction.state
    assert topology is not None and geometry_state is not None
    corners = np.asarray(geometry_state.positions)[topology.host_faces()]
    signed_volumes = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    )
    sides = topology.host_face_labels()
    expected_volumes = np.asarray(
        [
            np.sum(signed_volumes[sides[:, 0] == topology.region_index(region)])
            - np.sum(signed_volumes[sides[:, 1] == topology.region_index(region)])
            for region in source.domain.region_ids
        ],
        dtype=np.float64,
    )
    result = _execute(source)
    evidence = result.region_evidence
    assert evidence is not None
    assert source.interpretation == "freudenthal-categorical-interface-reconstruction"
    assert evidence.source_revision == source.source_revision
    np.testing.assert_allclose(
        _region_measure_array(evidence),
        expected_volumes,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.sum(expected_volumes) != pytest.approx(8.0)
    assert set(evidence.cell_region_ids) == {"left", "right"}


@pytest.mark.parametrize(
    "limits",
    (
        phx.meshing.MeshingLimits(maximum_edges=17),
        phx.meshing.MeshingLimits(maximum_faces=11),
        phx.meshing.MeshingLimits(maximum_connectivity_entries=35),
        phx.meshing.MeshingLimits(maximum_data_bytes=583),
        phx.meshing.MeshingLimits(maximum_geometry_queries=7),
        phx.meshing.MeshingLimits(maximum_scratch_bytes=1000),
        phx.meshing.MeshingLimits(maximum_work_units=1),
        phx.meshing.MeshingLimits(maximum_wall_seconds=1e-12),
    ),
)
def test_occupied_image_source_preparation_refuses_real_budget_boundaries(
    limits: phx.meshing.MeshingLimits,
) -> None:
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    original = np.asarray(source.labels.asset.values).copy()
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        prepare_compartment_complex(source, limits=limits)
    expected = (
        phx.meshing.MeshingFailureCategory.TIMED_OUT
        if limits.maximum_wall_seconds == 1e-12
        else phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    )
    assert refused.value.category is expected
    assert refused.value.evidence.stage == "source_inspection"
    np.testing.assert_array_equal(source.labels.asset.values, original)


def test_occupied_image_publication_counts_canonical_incidence_connectivity() -> None:
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    original = np.asarray(source.labels.asset.values).copy()
    limits = phx.meshing.MeshingLimits(maximum_connectivity_entries=36)
    prepare_compartment_complex(source, limits=limits)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        f"{source.source_id}:boundary",
        np.asarray((0,), dtype=np.int64),
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                3.0,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("image_material_tetrahedral"),
    ).plan(source, request, coordinate_contract=source.coordinate_contract)
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        plan.execute()
    evidence = refused.value.evidence
    assert evidence.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert evidence.stage == phx.meshing.MeshingStageKind.CANONICALIZATION.value
    assert dict(evidence.requested)["maximum_connectivity_entries"] == 36
    achieved = dict(evidence.achieved)
    assert achieved["connectivity_entries"] > 36
    assert achieved["native_scope:work_units"] > 0
    np.testing.assert_array_equal(source.labels.asset.values, original)


@pytest.mark.parametrize(
    "change", ("inactive-vertex", "inactive-region", "moved-geometry")
)
def test_categorical_volume_rejects_stale_or_noncanonical_active_authority(
    change: str,
) -> None:
    extraction = _categorical_extraction()
    state = extraction.state
    topology = extraction.topology
    assert state is not None and topology is not None
    match change:
        case "inactive-vertex":
            changed = eqx.tree_at(
                lambda value: value.vertex_active,
                state,
                state.vertex_active.at[0].set(False),
            )
        case "inactive-region":
            changed = eqx.tree_at(
                lambda value: value.region_active,
                state,
                state.region_active.at[0].set(False),
            )
        case "moved-geometry":
            changed = state.with_positions(state.positions.at[0, 0].add(0.125))
        case _:
            raise ValueError(change)
    candidate = eqx.tree_at(lambda value: value.state, extraction, changed)
    with pytest.raises(ValueError):
        LabelFieldVolumeBinding(candidate, phx.SpatialCoordinateContract.si())


def test_changed_image_renewal_rejects_incompatible_outer_source_without_mutating_result() -> (
    None
):
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    result = _execute(source)
    outer = source.outer_surface.mesh.with_coordinates(
        np.asarray(source.outer_surface.mesh.coordinates) + (0.125, 0.0, 0.0),
        numeric_version="incompatible-renewed-envelope",
    )
    changed = phx.geometry.SurfaceModel(outer, source.outer_surface.metadata)
    updated = replace(source, outer_surface=changed)
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        revalidate_region_evidence(
            result,
            result.mesh,
            result.geometry,
            result.zones,
            result.patches,
            compartment_source=updated,
            limits=phx.meshing.MeshingLimits(),
        )
    assert refused.value.category is phx.meshing.MeshingFailureCategory.COMPLIANCE_FAILED
    assert result.region_evidence is not None
    result.region_evidence.require_current(
        result.mesh,
        result.zones,
        result.patches,
        geometry=result.geometry,
    )


def test_image_preparation_reserves_against_original_live_native_storage() -> None:
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    original = np.asarray(source.labels.asset.values).copy()
    limits = phx.meshing.MeshingLimits(maximum_scratch_bytes=1024**2)
    with pytest.raises(phx.meshing.MeshingFailure) as refused:
        with native_volume_execution_budget(
            limits,
            stage=phx.meshing.MeshingStageKind.SOURCE_INSPECTION,
        ) as execution:
            retained = execution.allocate_host_array((1024**2 - 32768,), np.uint8)
            retained.fill(17)
            prepare_compartment_complex(source, limits=limits)
    assert refused.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert execution.evidence is not None
    assert execution.evidence.externally_charged_geometry_queries == 0
    assert execution.evidence.host_storage_live_bytes_upper == 0
    assert retained[0] == retained[-1] == 17
    np.testing.assert_array_equal(source.labels.asset.values, original)
