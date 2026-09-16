from __future__ import annotations

from io import BytesIO

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian


STUDY_UID = "1.2.826.0.1.3680043.10.999.1"
SERIES_UID = "1.2.826.0.1.3680043.10.999.2"
FRAME_UID = "1.2.826.0.1.3680043.10.999.3"
CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
ENHANCED_CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2.1"
NM_STORAGE = "1.2.840.10008.5.1.4.1.1.20"
PET_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
RTPLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"
RTDOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"


def item(**values):
    result = Dataset()
    for keyword, value in values.items():
        setattr(result, keyword, value)
    return result


def sequence(*items):
    return Sequence(items)


def _base(sop_class_uid: str, sop_instance_uid: str, modality: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999.99"
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\x00" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = STUDY_UID
    dataset.SeriesInstanceUID = SERIES_UID
    dataset.FrameOfReferenceUID = FRAME_UID
    dataset.Modality = modality
    dataset.PatientIdentityRemoved = "YES"
    dataset.BurnedInAnnotation = "NO"
    return dataset


def _pixels(dataset: Dataset, pixels: np.ndarray) -> None:
    array = np.asarray(pixels, dtype="<i2")
    frames, rows, columns = (1, *array.shape) if array.ndim == 2 else array.shape
    dataset.Rows = rows
    dataset.Columns = columns
    if frames > 1:
        dataset.NumberOfFrames = frames
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelData = array.tobytes(order="C")


def encode(dataset: FileDataset) -> bytes:
    stream = BytesIO()
    dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


def legacy_ct(
    *,
    sop_instance_uid: str,
    z_mm: float,
    pixels: np.ndarray,
    slope: float = 2.0,
    intercept: float = -1000.0,
) -> bytes:
    dataset = _base(CT_STORAGE, sop_instance_uid, "CT")
    _pixels(dataset, pixels)
    dataset.ImagePositionPatient = [0.0, 0.0, z_mm]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [2.0, 3.0]
    dataset.SliceThickness = 4.0
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.RescaleType = "HU"
    return encode(dataset)


def enhanced_ct(*, sop_instance_uid: str) -> bytes:
    dataset = _base(ENHANCED_CT_STORAGE, sop_instance_uid, "CT")
    pixels = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    _pixels(dataset, pixels)
    shared = item(
        PlaneOrientationSequence=sequence(
            item(ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        ),
        PixelMeasuresSequence=sequence(
            item(PixelSpacing=[2.0, 3.0], SpacingBetweenSlices=4.0, SliceThickness=4.0)
        ),
    )
    dataset.SharedFunctionalGroupsSequence = sequence(shared)
    dataset.PerFrameFunctionalGroupsSequence = sequence(
        item(
            PlanePositionSequence=sequence(item(ImagePositionPatient=[0.0, 0.0, 4.0])),
            PixelValueTransformationSequence=sequence(
                item(RescaleSlope=2.0, RescaleIntercept=-1000.0, RescaleType="HU")
            ),
        ),
        item(
            PlanePositionSequence=sequence(item(ImagePositionPatient=[0.0, 0.0, 0.0])),
            PixelValueTransformationSequence=sequence(
                item(RescaleSlope=3.0, RescaleIntercept=-900.0, RescaleType="HU")
            ),
        ),
    )
    return encode(dataset)


def nm_counts(*, sop_instance_uid: str, slope: float = 1.0) -> bytes:
    dataset = _base(NM_STORAGE, sop_instance_uid, "NM")
    _pixels(dataset, np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    dataset.Units = "CNTS"
    dataset.NumberOfTimeSlices = 1
    dataset.NumberOfSlices = 2
    dataset.ActualFrameDuration = 1000
    dataset.FrameReferenceTime = 500.0
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = 0.0
    dataset.DetectorInformationSequence = sequence(
        item(
            ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ImagePositionPatient=[0.0, 0.0, 0.0],
            PixelSpacing=[2.0, 3.0],
            SpacingBetweenSlices=4.0,
        )
    )
    return encode(dataset)


def pet_slice(*, sop_instance_uid: str, z_mm: float, pixels: np.ndarray) -> bytes:
    dataset = _base(PET_STORAGE, sop_instance_uid, "PT")
    _pixels(dataset, pixels)
    dataset.ImagePositionPatient = [0.0, 0.0, z_mm]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [2.0, 3.0]
    dataset.SliceThickness = 4.0
    dataset.Units = "BQML"
    dataset.DecayCorrection = "START"
    dataset.CorrectedImage = ["ATTN", "DECY"]
    dataset.DecayFactor = 0.9
    dataset.SeriesDate = "20200101"
    dataset.SeriesTime = "120000.000000"
    dataset.RescaleSlope = 0.5
    dataset.RescaleIntercept = 0.0
    dataset.RescaleType = "BQML"
    return encode(dataset)


def rtstruct(*, sop_instance_uid: str, image_sop_uid: str) -> bytes:
    dataset = _base(RTSTRUCT_STORAGE, sop_instance_uid, "RTSTRUCT")
    dataset.StructureSetLabel = "SYNTHETIC"
    dataset.StructureSetROISequence = sequence(
        item(
            ROINumber=1,
            ROIName="TARGET",
            ReferencedFrameOfReferenceUID=FRAME_UID,
            ROIGenerationAlgorithm="MANUAL",
        )
    )
    dataset.RTROIObservationsSequence = sequence(
        item(ObservationNumber=1, ReferencedROINumber=1, RTROIInterpretedType="CTV")
    )
    contour_image = item(
        ReferencedSOPClassUID=CT_STORAGE,
        ReferencedSOPInstanceUID=image_sop_uid,
    )
    contour = item(
        ContourImageSequence=sequence(contour_image),
        ContourGeometricType="CLOSED_PLANAR",
        NumberOfContourPoints=4,
        ContourData=[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0, 0.0],
    )
    dataset.ROIContourSequence = sequence(
        item(ReferencedROINumber=1, ContourSequence=sequence(contour))
    )
    return encode(dataset)


def rtplan(*, sop_instance_uid: str, structure_sop_uid: str) -> bytes:
    dataset = _base(RTPLAN_STORAGE, sop_instance_uid, "RTPLAN")
    dataset.RTPlanLabel = "SYNTHETIC"
    dataset.RTPlanName = "RESEARCH"
    dataset.ApprovalStatus = "UNAPPROVED"
    dataset.ReferencedStructureSetSequence = sequence(
        item(
            ReferencedSOPClassUID=RTSTRUCT_STORAGE,
            ReferencedSOPInstanceUID=structure_sop_uid,
        )
    )
    dataset.FractionGroupSequence = sequence(item(FractionGroupNumber=1))
    dataset.DoseReferenceSequence = sequence(item(DoseReferenceNumber=1))
    return encode(dataset)


def rtdose(
    *,
    sop_instance_uid: str,
    plan_sop_uid: str | None,
    dose_units: str = "GY",
) -> bytes:
    dataset = _base(RTDOSE_STORAGE, sop_instance_uid, "RTDOSE")
    _pixels(dataset, np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [2.0, 3.0]
    dataset.GridFrameOffsetVector = [0.0, 4.0]
    dataset.DoseUnits = dose_units
    dataset.DoseType = "PHYSICAL"
    dataset.DoseSummationType = "PLAN" if plan_sop_uid is not None else "RECORD"
    dataset.DoseGridScaling = 0.01
    if plan_sop_uid is not None:
        dataset.ReferencedRTPlanSequence = sequence(
            item(
                ReferencedSOPClassUID=RTPLAN_STORAGE,
                ReferencedSOPInstanceUID=plan_sop_uid,
            )
        )
    return encode(dataset)
