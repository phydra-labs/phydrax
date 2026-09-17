# Scientific imaging

::: phydrax.imaging
    options:
      members:
        - ImagePlaneSupport
        - ImageAsset
        - ImageFieldSpec
        - ImageSample2D
        - image_coordinates
        - bilinear_sample
        - backward_warp
        - ImageAxisConvention
        - VoxelReference
        - ImageIndexAffine
        - MedicalImageSupport
        - DeidentificationEvidence
        - MedicalImageAsset
        - LabelDefinition
        - LabelOntology
        - LabelVolume
        - RegistrationDirection
        - RegistrationEvaluationPlan
        - PreparedRegistrationEvaluation
        - RegistrationCandidate
        - RegistrationEvidence
        - RegistrationCheckpoint
        - DiffusionTensorImage
        - NibabelImageProvider
        - ConservativeVoxelCellTransfer
        - ImageToP1ProjectionPlan
        - LabelImageTransferPlan
        - ProbabilityImageTransferPlan
        - TensorImageTransferPlan
        - SegmentationOperation
        - SegmentationProcessingPlan
        - build_compartment_complex
        - extract_compartment_surfaces
        - SchlierenImagePair
        - GladstoneDaleRelation
        - SchlierenDeflectionPlan
        - KnifeEdgeSchlierenPlan
        - BackgroundOrientedSchlierenPlan

## Bounded tomography routes

`TetrahedralXRayTransformPlan` uses the shared affine-simplex map and packed
BVH traversal. `maximum_segments_per_ray` bounds retained ray--cell routes;
query traversal uses a depth-sized stack and bounded query chunks. Insufficient
candidate or traversal capacity fails preparation rather than dropping
attenuation. Voxel and tetrahedral forward/transpose pairs retain matched
route weights.

::: phydrax.imaging.camera

::: phydrax.spatial_sampling

## CT material calibration

::: phydrax.imaging
    options:
      members:
        - HUCalibrationAnchor
        - HUToMaterialCalibration
        - HUToMaterialResult
        - apply_hu_calibration
        - imaging_candidate_profile
        - imaging_candidate_profiles

## Diagnostic tomography

::: phydrax.imaging.tomography
    options:
      show_root_heading: true
      show_source: false
      members:
        - TubeSpectrum
        - FilterStack
        - BowtieTransmission
        - AECSetting
        - DetectorResponse
        - ScatterLabel
        - CTViewAcquisition
        - CTAcquisitionProtocol
        - MaterialBasisProjectionPlan
        - PolychromaticDetectorPlan

## DICOM interchange

::: phydrax.imaging.interchange
    options:
      show_root_heading: true
      show_source: false
