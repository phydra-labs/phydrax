#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed external measurements and predicted scientific quantities."""

from typing import TYPE_CHECKING

from .._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "AcquisitionIdentity": "._asset",
    "DataOrigin": "._asset",
    "DataStage": "._asset",
    "DerivationRecord": "._asset",
    "MeasurementAsset": "._asset",
    "AffineClockMap": "._clock",
    "ClockIdentity": "._clock",
    "ClockMappingEvidence": "._clock",
    "PiecewiseClockMap": "._clock",
    "PreparedClockMap": "._clock",
    "MeasurementCollection": "._collection",
    "MeasurementRelation": "._collection",
    "MeasurementRelationKind": "._collection",
    "MeasurementRole": "._collection",
    "MeasurementRoleAssignment": "._collection",
    "IndependentStandardUncertainty": "._field",
    "PreparedQuantityField": "._field",
    "QualityFlag": "._field",
    "QuantityField": "._field",
    "SamplingSemantics": "._field",
    "SpatialSamplingKind": "._field",
    "LasPointProvider": "._las",
    "ConditionPayloadReference": "._operations",
    "DataQualityAnnotation": "._operations",
    "ExposureKind": "._operations",
    "ExposureRecord": "._operations",
    "OperationalCoordinate": "._operations",
    "OperationalInterval": "._operations",
    "ResolvedConditionSnapshot": "._operations",
    "QuantitySpec": "._quantity",
    "resolve_quantity": "._quantity",
    "ValueKind": "._quantity",
    "ValueLayout": "._quantity",
    "RadiationQuantityKind": "._radiation",
    "resolve_radiation_quantity": "._radiation",
    "MeasurementSelectionPlan": "._selection",
    "IndexSampleSupport": "._support",
    "PointSampleSupport": "._support",
    "prepare_point_support": "._support",
    "prepare_ray_support": "._support",
    "PreparedPointSampleSupport": "._support",
    "PreparedRaySampleSupport": "._support",
    "RaySampleSupport": "._support",
    "SampleSupport": "._support",
    "SampleTimeAxis": "._time",
    "TemporalSampling": "._time",
    "TemporalSamplingKind": "._time",
    "TimeBasis": "._time",
    "PulseResponse": "._waveform",
    "WaveformSupport": "._waveform",
    "cartesianize_lidar_scan": ".lidar",
    "LidarPointProduct": ".lidar",
    "LidarScan": ".lidar",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._asset import (
        AcquisitionIdentity,
        DataOrigin,
        DataStage,
        DerivationRecord,
        MeasurementAsset,
    )
    from ._clock import (
        AffineClockMap,
        ClockIdentity,
        ClockMappingEvidence,
        PiecewiseClockMap,
        PreparedClockMap,
    )
    from ._collection import (
        MeasurementCollection,
        MeasurementRelation,
        MeasurementRelationKind,
        MeasurementRole,
        MeasurementRoleAssignment,
    )
    from ._field import (
        IndependentStandardUncertainty,
        PreparedQuantityField,
        QualityFlag,
        QuantityField,
        SamplingSemantics,
        SpatialSamplingKind,
    )
    from ._las import LasPointProvider
    from ._operations import (
        ConditionPayloadReference,
        DataQualityAnnotation,
        ExposureKind,
        ExposureRecord,
        OperationalCoordinate,
        OperationalInterval,
        ResolvedConditionSnapshot,
    )
    from ._quantity import QuantitySpec, resolve_quantity, ValueKind, ValueLayout
    from ._radiation import RadiationQuantityKind, resolve_radiation_quantity
    from ._selection import MeasurementSelectionPlan
    from ._support import (
        IndexSampleSupport,
        PointSampleSupport,
        prepare_point_support,
        prepare_ray_support,
        PreparedPointSampleSupport,
        PreparedRaySampleSupport,
        RaySampleSupport,
        SampleSupport,
    )
    from ._time import SampleTimeAxis, TemporalSampling, TemporalSamplingKind, TimeBasis
    from ._waveform import PulseResponse, WaveformSupport
    from .lidar import cartesianize_lidar_scan, LidarPointProduct, LidarScan


__all__ = [
    "AcquisitionIdentity",
    "AffineClockMap",
    "ClockIdentity",
    "ClockMappingEvidence",
    "ConditionPayloadReference",
    "DataQualityAnnotation",
    "DataOrigin",
    "DataStage",
    "DerivationRecord",
    "IndependentStandardUncertainty",
    "IndexSampleSupport",
    "LidarPointProduct",
    "LidarScan",
    "ExposureKind",
    "ExposureRecord",
    "LasPointProvider",
    "MeasurementAsset",
    "MeasurementCollection",
    "MeasurementRelation",
    "MeasurementRelationKind",
    "MeasurementRole",
    "MeasurementRoleAssignment",
    "MeasurementSelectionPlan",
    "OperationalCoordinate",
    "OperationalInterval",
    "PointSampleSupport",
    "PreparedQuantityField",
    "PreparedPointSampleSupport",
    "PreparedRaySampleSupport",
    "PiecewiseClockMap",
    "PreparedClockMap",
    "PulseResponse",
    "QualityFlag",
    "QuantityField",
    "QuantitySpec",
    "RadiationQuantityKind",
    "RaySampleSupport",
    "SampleSupport",
    "ResolvedConditionSnapshot",
    "SampleTimeAxis",
    "SamplingSemantics",
    "SpatialSamplingKind",
    "TemporalSampling",
    "TemporalSamplingKind",
    "TimeBasis",
    "ValueKind",
    "ValueLayout",
    "cartesianize_lidar_scan",
    "prepare_point_support",
    "prepare_ray_support",
    "WaveformSupport",
    "resolve_quantity",
    "resolve_radiation_quantity",
]
