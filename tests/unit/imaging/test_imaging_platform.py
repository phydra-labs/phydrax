from hashlib import sha256

import numpy as np
import pytest

import phydrax as phx


def _manifest(
    artifact_name="synthetic-medical-image",
    *,
    export_permitted=True,
):
    return phx.qualification.ReferenceArtifactManifest(
        artifact_name,
        checksum_algorithm="sha256",
        checksum="0" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=export_permitted,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )


def _deid():
    return phx.imaging.DeidentificationEvidence(
        "synthetic-deid", "subject-0", "synthetic", True, True, True
    )


def _derivation():
    return phx.measurement.DerivationRecord(
        phx.measurement.DataOrigin.SYNTHETIC,
        phx.measurement.DataStage.RECONSTRUCTED,
        transformation_id="synthetic-test-generator",
    )


def _affine(matrix=None):
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="synthetic-patient",
    )
    return phx.imaging.ImageIndexAffine(
        np.eye(4) if matrix is None else matrix,
        "voxel-index",
        contract,
        phx.imaging.ImageAxisConvention.LPS,
    )


def _asset(
    values,
    layout,
    *,
    mask=None,
    affine=None,
    asset_id="image",
    references=None,
    uncertainty=None,
    quality_flags=(),
):
    return phx.imaging.MedicalImageAsset(
        asset_id,
        "synthetic-mri",
        np.asarray(values),
        _affine() if affine is None else affine,
        layout,
        _deid(),
        (_manifest(),) if references is None else references,
        _derivation(),
        valid_mask=mask,
        uncertainty=uncertainty,
        quality_flags=quality_flags,
    )


def test_medical_image_asset_preserves_references_uncertainty_and_quality_flags():
    values = np.arange(8.0).reshape((2, 2, 2))
    layout = phx.imaging.ImageFieldSpec.named(
        "signal", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
    )
    references = (_manifest("source-a"), _manifest("source-b"))
    uncertainty = phx.measurement.IndependentStandardUncertainty(
        np.full(values.shape, 0.25), phx.units.ONE
    )
    flag = phx.measurement.QualityFlag(
        "motion",
        np.broadcast_to(np.eye(2, dtype="bool")[:, :, None], values.shape),
        "Potential motion artifact",
    )
    asset = _asset(
        values,
        layout,
        references=references,
        uncertainty=uncertainty,
        quality_flags=(flag,),
    )

    assert asset.references == references
    assert asset.measurement.references == references
    assert asset.uncertainty is uncertainty
    assert asset.measurement.field.uncertainty is uncertainty
    assert asset.quality_flags[0] is flag
    assert asset.measurement.field.quality_flags[0] is flag

    with pytest.raises(TypeError, match="at least one"):
        _asset(values, layout, references=())
    with pytest.raises(ValueError, match="references must be unique"):
        _asset(values, layout, references=(references[0], references[0]))


def test_nifti_export_requires_rights_from_every_reference(tmp_path):
    values = np.zeros((2, 2, 2), dtype="float64")
    layout = phx.imaging.ImageFieldSpec.named(
        "signal", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
    )
    asset = _asset(
        values,
        layout,
        references=(
            _manifest("exportable"),
            _manifest("controlled", export_permitted=False),
        ),
    )

    with pytest.raises(PermissionError, match="export-not-permitted"):
        phx.imaging.NibabelImageProvider().write(asset, tmp_path / "blocked.nii")


def test_nifti_export_requires_explicit_semantic_loss_admission(tmp_path):
    asset = _asset(
        np.zeros((2, 2, 2), dtype=np.float64),
        phx.imaging.ImageFieldSpec.named(
            "signal", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
        ),
    )
    with pytest.raises(ValueError, match="governed image semantics"):
        phx.imaging.NibabelImageProvider().write(asset, tmp_path / "lossy.nii")


def test_image_affine_units_frames_and_qform_conflict():
    matrix = np.asarray(
        (
            (0.0, -2.0, 0.0, 10.0),
            (1.5, 0.0, 0.0, -4.0),
            (0.0, 0.0, 3.0, 7.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    affine = _affine(matrix)
    index = np.asarray(((0.25, 1.5, 0.75),))
    world = affine.index_to_world(index)
    np.testing.assert_allclose(affine.world_to_index(world), index)
    ras = affine.to_convention(phx.imaging.ImageAxisConvention.RAS)
    np.testing.assert_allclose(ras.index_to_world(index), world * (-1.0, -1.0, 1.0))
    meters = affine.to_unit(phx.units.METER)
    np.testing.assert_allclose(meters.index_to_world(index), world * 1.0e-3)
    with pytest.raises(ValueError, match="qform/sform conflict"):
        phx.imaging.ImageIndexAffine.from_qform_sform(
            qform=np.eye(4),
            sform=np.diag((2.0, 1.0, 1.0, 1.0)),
            source_frame_id="voxel-index",
            coordinate_contract=affine.coordinate_contract,
            axis_convention=phx.imaging.ImageAxisConvention.LPS,
        )


def test_scalar_projection_and_voxel_sampling_reproduce_affine_field():
    with pytest.raises(ValueError, match="coordinate system disagree"):
        phx.imaging.ImageIndexAffine(
            np.eye(4),
            "voxel-index",
            _affine().coordinate_contract,
            phx.imaging.ImageAxisConvention.RAS,
        )
    values = np.fromfunction(lambda i, j, k: i + 2.0 * j + 3.0 * k, (2, 2, 2))
    layout = phx.imaging.ImageFieldSpec.named(
        "tracer", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
    )
    asset = _asset(values, layout)
    query = np.asarray(((0.25, 0.5, 0.75),))
    sampled = (
        phx.spatial_sampling.VoxelObservationPlan(
            values.shape, asset.spatial_affine, query, require_complete_coverage=True
        )
        .prepare()
        .apply(values)
    )
    np.testing.assert_allclose(sampled.values, (3.5,))
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),)),
    )
    prepared = phx.imaging.ImageToP1ProjectionPlan(
        asset, mesh, asset.spatial_affine.coordinate_contract
    ).prepare()
    projected = prepared.apply(values)
    np.testing.assert_allclose(projected.values, (0.0, 1.0, 2.0, 3.0), atol=1e-12)
    assert bool(projected.evidence.successful)


def test_tensor_transfer_reorients_spd_field():
    tensor = np.broadcast_to(np.diag((3.0, 2.0, 1.0)), (2, 2, 2, 3, 3)).copy()
    layout = phx.imaging.ImageFieldSpec.named(
        "diffusion",
        phx.units.ONE,
        phx.measurement.ValueKind.SYMMETRIC_TENSOR,
        (3, 3),
        "scanner-basis",
    )
    image = phx.imaging.DiffusionTensorImage(_asset(tensor, layout), 0.1)
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    result = phx.imaging.TensorImageTransferPlan(
        image, np.asarray(((0.4, 0.4, 0.4),)), "mesh-basis", rotation
    ).execute()
    np.testing.assert_allclose(result.values[0], np.diag((2.0, 3.0, 1.0)), atol=1e-12)
    assert bool(result.evidence.successful)
    with pytest.raises(ValueError, match="orthogonal"):
        phx.imaging.TensorImageTransferPlan(
            image,
            np.asarray(((0.4, 0.4, 0.4),)),
            "mesh-basis",
            2.0 * np.eye(3),
        )


def test_registration_refuses_empty_support_and_non_millimeter_policy():
    with pytest.raises(ValueError, match="at least one point"):
        phx.imaging.RegistrationEvaluationPlan(
            np.zeros((0, 3)),
            "reference",
            "target",
        )
    with pytest.raises(ValueError, match="canonical millimeter"):
        phx.imaging.RegistrationEvaluationPlan(
            np.zeros((1, 3)),
            "reference",
            "target",
            length_unit=phx.units.METER,
        )


def test_conservative_overlap_has_exact_mass_and_dual_pairing():
    transfer = phx.imaging.ConservativeVoxelCellTransfer(
        np.asarray((0, 0, 1, 1)),
        np.asarray((0, 1, 0, 1)),
        np.asarray((0.25, 0.75, 0.75, 0.25)),
        np.asarray((1.0, 1.0)),
        np.asarray((1.0, 1.0)),
        source_id="voxels",
        target_id="cells",
    )
    result = transfer.apply(np.asarray((2.0, 4.0)))
    np.testing.assert_allclose(result.values, (3.5, 2.5))
    assert bool(result.evidence.successful)


def test_label_compartments_and_single_oriented_interface():
    values = np.ones((2, 2, 2), dtype=np.int16)
    values[1] = 2
    layout = phx.imaging.ImageFieldSpec.named(
        "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
    )
    ontology = phx.imaging.LabelOntology(
        "synthetic-labels",
        "synthetic",
        "1",
        (
            phx.imaging.LabelDefinition(1, "left-label", "Left"),
            phx.imaging.LabelDefinition(2, "right-label", "Right"),
        ),
    )
    labels = phx.imaging.LabelVolume(_asset(values, layout, asset_id="labels"), ontology)
    compartments = (
        phx.geometry.CompartmentDefinition(
            "left", ("left-label",), "material", allowed_neighbor_ids=("right",)
        ),
        phx.geometry.CompartmentDefinition(
            "right", ("right-label",), "material", allowed_neighbor_ids=("left",)
        ),
    )
    interface = phx.geometry.CompartmentInterfaceDefinition(
        "left-right", "left", "right", "exchange"
    )
    complex_ = phx.imaging.build_compartment_complex(labels, compartments, (interface,))
    surfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)
    assert complex_.adjacency.successful
    assert len(surfaces.surfaces) == 1
    assert surfaces.surfaces[0].surface.mesh.entity_set(2).count == 8
    mesh = surfaces.surfaces[0].surface.mesh
    triangles = np.asarray(mesh.blocks[0].vertices)
    points = np.asarray(mesh.coordinates)[triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    assert np.all(normals[:, 0] > 0.0)

    reflected_matrix = np.diag((-1.0, 1.0, 1.0, 1.0))
    reflected_labels = phx.imaging.LabelVolume(
        _asset(
            values,
            layout,
            asset_id="reflected-labels",
            affine=_affine(reflected_matrix),
        ),
        ontology,
    )
    reflected_complex = phx.imaging.build_compartment_complex(
        reflected_labels, compartments, (interface,)
    )
    reflected = (
        phx.imaging.extract_compartment_surfaces(reflected_labels, reflected_complex)
        .surfaces[0]
        .surface.mesh
    )
    reflected_triangles = np.asarray(reflected.blocks[0].vertices)
    reflected_points = np.asarray(reflected.coordinates)[reflected_triangles]
    reflected_normals = np.cross(
        reflected_points[:, 1] - reflected_points[:, 0],
        reflected_points[:, 2] - reflected_points[:, 0],
    )
    assert np.all(reflected_normals[:, 0] < 0.0)


def test_probability_transfer_preserves_simplex_and_segmentation_transition():
    probabilities = np.zeros((2, 2, 2, 2), dtype="float64")
    probabilities[..., 0] = np.fromfunction(lambda i, j, k: (i + j + k) / 3.0, (2, 2, 2))
    probabilities[..., 1] = 1.0 - probabilities[..., 0]
    probability_layout = phx.imaging.ImageFieldSpec.named(
        "tissue-probability",
        phx.units.ONE,
        phx.measurement.ValueKind.PROBABILITY,
        (2,),
    )
    probability_asset = _asset(probabilities, probability_layout)
    transferred = phx.imaging.ProbabilityImageTransferPlan(
        probability_asset, np.asarray(((0.25, 0.5, 0.75),))
    ).execute()
    np.testing.assert_allclose(np.sum(transferred.values, axis=-1), 1.0)
    assert bool(transferred.evidence.successful)

    labels_array = np.zeros((4, 4, 4), dtype=np.int16)
    labels_array[0, 0, 0] = 1
    labels_array[2:, 2:, 2:] = 1
    label_layout = phx.imaging.ImageFieldSpec.named(
        "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
    )
    ontology = phx.imaging.LabelOntology(
        "binary",
        "synthetic",
        "1",
        (
            phx.imaging.LabelDefinition(0, "background", "Background"),
            phx.imaging.LabelDefinition(1, "tissue", "Tissue"),
        ),
    )
    source_uncertainty = phx.measurement.IndependentStandardUncertainty(
        np.full(labels_array.shape, 0.5), phx.units.ONE
    )
    source_flag = phx.measurement.QualityFlag(
        "reviewed", np.ones(labels_array.shape, dtype="bool"), "Reviewed source voxel"
    )
    labels = phx.imaging.LabelVolume(
        _asset(
            labels_array,
            label_layout,
            asset_id="binary-labels",
            uncertainty=source_uncertainty,
            quality_flags=(source_flag,),
        ),
        ontology,
    )
    uncovered = phx.imaging.LabelImageTransferPlan(
        labels, np.asarray(((10.0, 10.0, 10.0),))
    ).execute()
    assert not bool(uncovered.valid[0])
    assert uncovered.values[0] == 0
    transition = phx.imaging.SegmentationProcessingPlan(
        labels,
        (
            phx.imaging.SegmentationOperation(
                phx.imaging.SegmentationOperationKind.KEEP_LARGEST_COMPONENT,
                "tissue",
                "background",
            ),
        ),
    ).execute()
    assert transition.target.asset.values[0, 0, 0] == 0
    assert transition.reports[0].changed_voxels == 1
    assert transition.target.asset.references == labels.asset.references
    assert transition.target.asset.uncertainty is None
    assert transition.target.asset.measurement.field.uncertainty is None
    assert transition.target.asset.quality_flags[0] is source_flag


def test_real_nifti_roundtrip_preserves_values_affine_and_rights(tmp_path):
    nib = pytest.importorskip("nibabel")

    source = tmp_path / "source.nii.gz"
    values = np.arange(8.0).reshape((2, 2, 2))
    matrix = np.asarray(
        (
            (2.0, 0.0, 0.0, 10.0),
            (0.0, 3.0, 0.0, -4.0),
            (0.0, 0.0, 5.0, 7.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    image = nib.Nifti1Image(values, matrix)
    image.header.set_xyzt_units("mm")
    nib.save(image, source)
    reference = phx.qualification.ReferenceArtifactManifest(
        "synthetic-nifti",
        checksum_algorithm="sha256",
        checksum=sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )
    layout = phx.imaging.ImageFieldSpec.named(
        "signal", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
    )
    provider = phx.imaging.NibabelImageProvider()
    asset = provider.read(
        source,
        layout,
        _deid(),
        reference,
        _derivation(),
        asset_id="nifti",
        modality="synthetic-mri",
        reference_frame="patient",
    )
    assert asset.references == (reference,)
    export = provider.write(asset, tmp_path / "restored.nii.gz", allow_semantic_loss=True)
    assert {"references", "derivation", "deidentification"} <= set(export.lost_semantics)
    restored_image = nib.load(export.path)
    np.testing.assert_array_equal(np.asanyarray(restored_image.dataobj), values)
    np.testing.assert_allclose(restored_image.affine, matrix)


def test_real_timed_nifti_roundtrip_preserves_temporal_axis(tmp_path):
    nib = pytest.importorskip("nibabel")

    source = tmp_path / "timed.nii.gz"
    values = np.arange(24.0).reshape((2, 2, 2, 3))
    image = nib.Nifti1Image(values, np.eye(4))
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_zooms((1.0, 1.0, 1.0, 0.5))
    nib.save(image, source)
    reference = phx.qualification.ReferenceArtifactManifest(
        "timed-nifti",
        checksum_algorithm="sha256",
        checksum=sha256(source.read_bytes()).hexdigest(),
        size_bytes=source.stat().st_size,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )
    time_axis = phx.measurement.SampleTimeAxis(
        "dynamic-series", np.asarray((0.0, 0.5, 1.0)), phx.units.SECOND
    )
    provider = phx.imaging.NibabelImageProvider()
    asset = provider.read(
        source,
        phx.imaging.ImageFieldSpec.named(
            "dynamic-signal", phx.units.ONE, phx.measurement.ValueKind.REAL_SCALAR
        ),
        _deid(),
        reference,
        _derivation(),
        asset_id="timed",
        modality="dynamic-mri",
        reference_frame="patient",
        time_axis=time_axis,
    )
    export = provider.write(
        asset, tmp_path / "timed-restored.nii.gz", allow_semantic_loss=True
    )
    restored = nib.load(export.path)
    assert restored.header.get_xyzt_units() == ("mm", "sec")
    assert restored.header.get_zooms()[3] == pytest.approx(0.5)
