"""Lazy imaging facade."""

from importlib import import_module
from typing import Any, TYPE_CHECKING


_SYMBOL_MODULES: dict[str, tuple[str, str | None]] = {
    "ANTsRegistrationProvider": ("._providers", "ANTsRegistrationProvider"),
    "BackgroundOrientedSchlierenPlan": (".schlieren", "BackgroundOrientedSchlierenPlan"),
    "CompartmentSurface": ("._segmentation", "CompartmentSurface"),
    "CompartmentSurfaceResult": ("._segmentation", "CompartmentSurfaceResult"),
    "ConservativeVoxelCellTransfer": ("._transfer", "ConservativeVoxelCellTransfer"),
    "CoveragePolicy": ("._transfer", "CoveragePolicy"),
    "CurvedSchlierenPlan": (".schlieren", "CurvedSchlierenPlan"),
    "CurvedSchlierenResult": (".schlieren", "CurvedSchlierenResult"),
    "Dcm2NiixProvider": ("._providers", "Dcm2NiixProvider"),
    "DeidentificationEvidence": ("._core", "DeidentificationEvidence"),
    "DiffusionTensorImage": ("._core", "DiffusionTensorImage"),
    "FastSurferProvider": ("._providers", "FastSurferProvider"),
    "FreeSurferProvider": ("._providers", "FreeSurferProvider"),
    "GladstoneDaleRelation": (".schlieren", "GladstoneDaleRelation"),
    "GreedyRegistrationProvider": ("._providers", "GreedyRegistrationProvider"),
    "HUCalibrationAnchor": ("._ct", "HUCalibrationAnchor"),
    "HUToMaterialCalibration": ("._ct", "HUToMaterialCalibration"),
    "HUToMaterialResult": ("._ct", "HUToMaterialResult"),
    "HelmholtzContinuationEvidence": (".schlieren", "HelmholtzContinuationEvidence"),
    "HelmholtzContinuationResult": (".schlieren", "HelmholtzContinuationResult"),
    "HostMetadataValue": ("._core", "HostMetadataValue"),
    "ImageAsset": ("._asset", "ImageAsset"),
    "ImageAxisConvention": ("._core", "ImageAxisConvention"),
    "ImageFieldSpec": ("._asset", "ImageFieldSpec"),
    "ImageIndexAffine": ("._core", "ImageIndexAffine"),
    "ImagePlaneSupport": ("._plane", "ImagePlaneSupport"),
    "ImageSample2D": ("._sampling", "ImageSample2D"),
    "ImageToP1ProjectionPlan": ("._transfer", "ImageToP1ProjectionPlan"),
    "ImageTransferResult": ("._transfer", "ImageTransferResult"),
    "KnifeEdgeSchlierenPlan": (".schlieren", "KnifeEdgeSchlierenPlan"),
    "LabelDefinition": ("._core", "LabelDefinition"),
    "LabelImageTransferPlan": ("._transfer", "LabelImageTransferPlan"),
    "LabelOntology": ("._core", "LabelOntology"),
    "LabelVolume": ("._core", "LabelVolume"),
    "MedicalImageAsset": ("._core", "MedicalImageAsset"),
    "MedicalImageSupport": ("._core", "MedicalImageSupport"),
    "MedicalToolProvider": ("._providers", "MedicalToolProvider"),
    "MedicalToolResult": ("._providers", "MedicalToolResult"),
    "MultisliceEvidence": (".schlieren", "MultisliceEvidence"),
    "MultisliceRefractivePlan": (".schlieren", "MultisliceRefractivePlan"),
    "MultisliceResult": (".schlieren", "MultisliceResult"),
    "NibabelImageProvider": ("._nibabel", "NibabelImageProvider"),
    "NiftiExportResult": ("._nibabel", "NiftiExportResult"),
    "PreparedImageToP1Projection": ("._transfer", "PreparedImageToP1Projection"),
    "PreparedRegistrationEvaluation": (
        "._registration",
        "PreparedRegistrationEvaluation",
    ),
    "ProbabilityImageTransferPlan": ("._transfer", "ProbabilityImageTransferPlan"),
    "RefractivePhaseEvidence": (".schlieren", "RefractivePhaseEvidence"),
    "RefractivePhaseScreenPlan": (".schlieren", "RefractivePhaseScreenPlan"),
    "RegistrationCandidate": ("._registration", "RegistrationCandidate"),
    "RegistrationCheckpoint": ("._registration", "RegistrationCheckpoint"),
    "RegistrationDirection": ("._registration", "RegistrationDirection"),
    "RegistrationEvaluationPlan": ("._registration", "RegistrationEvaluationPlan"),
    "RegistrationEvidence": ("._registration", "RegistrationEvidence"),
    "ScalarHelmholtzContinuationPlan": (".schlieren", "ScalarHelmholtzContinuationPlan"),
    "SchlierenDeflectionPlan": (".schlieren", "SchlierenDeflectionPlan"),
    "SchlierenDeflectionResult": (".schlieren", "SchlierenDeflectionResult"),
    "SchlierenImageFormationResult": (".schlieren", "SchlierenImageFormationResult"),
    "SchlierenImagePair": (".schlieren", "SchlierenImagePair"),
    "SchlierenMethod": (".schlieren", "SchlierenMethod"),
    "SegmentationOperation": ("._segmentation", "SegmentationOperation"),
    "SegmentationOperationKind": ("._segmentation", "SegmentationOperationKind"),
    "SegmentationOperationReport": ("._segmentation", "SegmentationOperationReport"),
    "SegmentationProcessingPlan": ("._segmentation", "SegmentationProcessingPlan"),
    "SegmentationTransition": ("._segmentation", "SegmentationTransition"),
    "SynthSegProvider": ("._providers", "SynthSegProvider"),
    "TensorImageTransferPlan": ("._transfer", "TensorImageTransferPlan"),
    "TensorInterpolationPolicy": ("._transfer", "TensorInterpolationPolicy"),
    "TransferEvidence": ("._transfer", "TransferEvidence"),
    "VoxelReference": ("._core", "VoxelReference"),
    "WaveSchlierenEvidence": (".schlieren", "WaveSchlierenEvidence"),
    "WaveSchlierenPlan": (".schlieren", "WaveSchlierenPlan"),
    "WaveSchlierenResult": (".schlieren", "WaveSchlierenResult"),
    "apply_hu_calibration": ("._ct", "apply_hu_calibration"),
    "backward_warp": ("._sampling", "backward_warp"),
    "bilinear_sample": ("._sampling", "bilinear_sample"),
    "build_compartment_complex": ("._segmentation", "build_compartment_complex"),
    "camera": (".camera", None),
    "extract_compartment_surfaces": ("._segmentation", "extract_compartment_surfaces"),
    "image_coordinates": ("._sampling", "image_coordinates"),
    "imaging_candidate_profile": ("._qualification", "imaging_candidate_profile"),
    "imaging_candidate_profiles": ("._qualification", "imaging_candidate_profiles"),
    "interchange": (".interchange", None),
    "mri": (".mri", None),
    "tomography": (".tomography", None),
}

_FACADE_EXPORT_MODULES = ()


if TYPE_CHECKING:
    from . import camera, interchange, mri, tomography
    from ._asset import (
        ImageAsset,
        ImageFieldSpec,
    )
    from ._core import (
        DeidentificationEvidence,
        DiffusionTensorImage,
        HostMetadataValue,
        ImageAxisConvention,
        ImageIndexAffine,
        LabelDefinition,
        LabelOntology,
        LabelVolume,
        MedicalImageAsset,
        MedicalImageSupport,
        VoxelReference,
    )
    from ._ct import (
        apply_hu_calibration,
        HUCalibrationAnchor,
        HUToMaterialCalibration,
        HUToMaterialResult,
    )
    from ._nibabel import (
        NibabelImageProvider,
        NiftiExportResult,
    )
    from ._plane import (
        ImagePlaneSupport,
    )
    from ._providers import (
        ANTsRegistrationProvider,
        Dcm2NiixProvider,
        FastSurferProvider,
        FreeSurferProvider,
        GreedyRegistrationProvider,
        MedicalToolProvider,
        MedicalToolResult,
        SynthSegProvider,
    )
    from ._qualification import (
        imaging_candidate_profile,
        imaging_candidate_profiles,
    )
    from ._registration import (
        PreparedRegistrationEvaluation,
        RegistrationCandidate,
        RegistrationCheckpoint,
        RegistrationDirection,
        RegistrationEvaluationPlan,
        RegistrationEvidence,
    )
    from ._sampling import (
        backward_warp,
        bilinear_sample,
        image_coordinates,
        ImageSample2D,
    )
    from ._segmentation import (
        build_compartment_complex,
        CompartmentSurface,
        CompartmentSurfaceResult,
        extract_compartment_surfaces,
        SegmentationOperation,
        SegmentationOperationKind,
        SegmentationOperationReport,
        SegmentationProcessingPlan,
        SegmentationTransition,
    )
    from ._transfer import (
        ConservativeVoxelCellTransfer,
        CoveragePolicy,
        ImageToP1ProjectionPlan,
        ImageTransferResult,
        LabelImageTransferPlan,
        PreparedImageToP1Projection,
        ProbabilityImageTransferPlan,
        TensorImageTransferPlan,
        TensorInterpolationPolicy,
        TransferEvidence,
    )
    from .schlieren import (
        BackgroundOrientedSchlierenPlan,
        CurvedSchlierenPlan,
        CurvedSchlierenResult,
        GladstoneDaleRelation,
        HelmholtzContinuationEvidence,
        HelmholtzContinuationResult,
        KnifeEdgeSchlierenPlan,
        MultisliceEvidence,
        MultisliceRefractivePlan,
        MultisliceResult,
        RefractivePhaseEvidence,
        RefractivePhaseScreenPlan,
        ScalarHelmholtzContinuationPlan,
        SchlierenDeflectionPlan,
        SchlierenDeflectionResult,
        SchlierenImageFormationResult,
        SchlierenImagePair,
        SchlierenMethod,
        WaveSchlierenEvidence,
        WaveSchlierenPlan,
        WaveSchlierenResult,
    )


def __getattr__(name: str) -> Any:
    owner = _SYMBOL_MODULES.get(name)
    if owner is not None:
        module_name, symbol = owner
        module = import_module(module_name, __package__)
        value = module if symbol is None else getattr(module, symbol)
        globals()[name] = value
        return value
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "interchange",
    "HUCalibrationAnchor",
    "HUToMaterialCalibration",
    "HUToMaterialResult",
    "apply_hu_calibration",
    "imaging_candidate_profile",
    "imaging_candidate_profiles",
    "mri",
    "tomography",
    "ANTsRegistrationProvider",
    "camera",
    "backward_warp",
    "BackgroundOrientedSchlierenPlan",
    "bilinear_sample",
    "build_compartment_complex",
    "CompartmentSurface",
    "CompartmentSurfaceResult",
    "ConservativeVoxelCellTransfer",
    "Dcm2NiixProvider",
    "CoveragePolicy",
    "DeidentificationEvidence",
    "CurvedSchlierenPlan",
    "CurvedSchlierenResult",
    "DiffusionTensorImage",
    "HostMetadataValue",
    "ImageToP1ProjectionPlan",
    "ImageTransferResult",
    "LabelImageTransferPlan",
    "FastSurferProvider",
    "FreeSurferProvider",
    "GladstoneDaleRelation",
    "HelmholtzContinuationEvidence",
    "HelmholtzContinuationResult",
    "GreedyRegistrationProvider",
    "ImageAsset",
    "ImageFieldSpec",
    "ImageAxisConvention",
    "image_coordinates",
    "ImageSample2D",
    "ImageIndexAffine",
    "ImagePlaneSupport",
    "KnifeEdgeSchlierenPlan",
    "LabelDefinition",
    "LabelOntology",
    "LabelVolume",
    "extract_compartment_surfaces",
    "MedicalImageAsset",
    "MedicalImageSupport",
    "MedicalToolProvider",
    "MedicalToolResult",
    "NibabelImageProvider",
    "NiftiExportResult",
    "MultisliceEvidence",
    "MultisliceRefractivePlan",
    "MultisliceResult",
    "ProbabilityImageTransferPlan",
    "PreparedRegistrationEvaluation",
    "RegistrationCandidate",
    "RegistrationCheckpoint",
    "RegistrationDirection",
    "RegistrationEvaluationPlan",
    "RegistrationEvidence",
    "PreparedImageToP1Projection",
    "RefractivePhaseEvidence",
    "RefractivePhaseScreenPlan",
    "ScalarHelmholtzContinuationPlan",
    "TensorImageTransferPlan",
    "SchlierenDeflectionPlan",
    "SchlierenDeflectionResult",
    "SchlierenImageFormationResult",
    "SchlierenImagePair",
    "SchlierenMethod",
    "TensorInterpolationPolicy",
    "TransferEvidence",
    "SegmentationOperation",
    "SegmentationOperationKind",
    "SegmentationOperationReport",
    "SegmentationProcessingPlan",
    "SegmentationTransition",
    "SynthSegProvider",
    "WaveSchlierenEvidence",
    "WaveSchlierenPlan",
    "WaveSchlierenResult",
    "VoxelReference",
]
