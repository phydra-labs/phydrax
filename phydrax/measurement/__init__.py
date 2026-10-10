#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed external measurements and predicted scientific quantities."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

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


if TYPE_CHECKING:
    from ._las import LasPointProvider


def __getattr__(name: str) -> Any:
    if name != "LasPointProvider":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("._las", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
