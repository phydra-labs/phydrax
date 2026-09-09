#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""LiDAR ray/range acquisitions and explicitly derived Cartesian points."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..qualification import ReferenceArtifactManifest
from ..units import conversion_factor, LENGTH
from ._asset import DataStage, DerivationRecord, MeasurementAsset
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import ValueKind
from ._support import PointSampleSupport, RaySampleSupport


@dataclass(frozen=True, slots=True)
class LidarScan:
    """Range measurements and optional return channels on one ray support."""

    range_asset: MeasurementAsset
    intensity_asset: MeasurementAsset | None = None
    return_index_asset: MeasurementAsset | None = None
    scan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.range_asset, MeasurementAsset):
            raise TypeError("range_asset must be MeasurementAsset.")
        range_field = self.range_asset.field
        if not isinstance(range_field.support, RaySampleSupport):
            raise TypeError("LiDAR range measurements require RaySampleSupport.")
        if range_field.quantity.unit.dimension != LENGTH:
            raise ValueError("LiDAR range measurements require a length unit.")
        if range_field.layout.kind is not ValueKind.REAL_SCALAR:
            raise ValueError("LiDAR ranges require real scalar values.")
        assets = tuple(
            asset
            for asset in (self.range_asset, self.intensity_asset, self.return_index_asset)
            if asset is not None
        )
        if any(not isinstance(asset, MeasurementAsset) for asset in assets):
            raise TypeError("LiDAR optional channels must be MeasurementAsset values.")
        if any(
            asset.field.support.support_id != range_field.support.support_id
            for asset in assets
        ):
            raise ValueError("Every LiDAR channel must share the range ray support.")
        acquisition_ids = {
            None if asset.acquisition is None else asset.acquisition.identity_id
            for asset in assets
        }
        if len(acquisition_ids) != 1:
            raise ValueError("Every LiDAR channel must share one acquisition identity.")
        if self.return_index_asset is not None:
            index_field = self.return_index_asset.field
            if index_field.layout.kind not in {ValueKind.CATEGORICAL, ValueKind.COUNT}:
                raise ValueError(
                    "LiDAR return indices require categorical or count values."
                )
        object.__setattr__(
            self,
            "scan_id",
            canonical_fingerprint(
                {
                    "kind": "lidar-scan",
                    "range": self.range_asset.content_id,
                    "intensity": None
                    if self.intensity_asset is None
                    else self.intensity_asset.content_id,
                    "return_index": None
                    if self.return_index_asset is None
                    else self.return_index_asset.content_id,
                }
            ),
        )

    @property
    def support(self) -> RaySampleSupport:
        support = self.range_asset.field.support
        if not isinstance(support, RaySampleSupport):
            raise TypeError("LiDAR range support is not RaySampleSupport.")
        return support


@dataclass(frozen=True, slots=True)
class LidarPointProduct:
    """Derived Cartesian scan points with lineage-preserving attributes."""

    support: PointSampleSupport
    attributes: tuple[QuantityField, ...]
    derivation: DerivationRecord
    source_scan_id: str
    references: tuple[ReferenceArtifactManifest, ...] = ()
    point_product_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.support, PointSampleSupport):
            raise TypeError("support must be PointSampleSupport.")
        attributes = tuple(self.attributes)
        if any(not isinstance(value, QuantityField) for value in attributes):
            raise TypeError("attributes must contain QuantityField values.")
        if any(
            value.support.support_id != self.support.support_id for value in attributes
        ):
            raise ValueError("Every point attribute must share the point support.")
        if len({value.field_id for value in attributes}) != len(attributes):
            raise ValueError("Point attribute field IDs must be unique.")
        if not isinstance(self.derivation, DerivationRecord):
            raise TypeError("derivation must be DerivationRecord.")
        if self.derivation.stage is not DataStage.DERIVED:
            raise ValueError("Cartesian LiDAR points must be a derived product.")
        if not isinstance(self.source_scan_id, str) or not self.source_scan_id:
            raise ValueError("source_scan_id must be non-empty.")
        references = tuple(self.references)
        if any(not isinstance(value, ReferenceArtifactManifest) for value in references):
            raise TypeError("references must contain ReferenceArtifactManifest values.")
        if len({value.manifest_id for value in references}) != len(references):
            raise ValueError("references must be unique.")
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(
            self,
            "point_product_id",
            canonical_fingerprint(
                {
                    "kind": "lidar-point-product",
                    "support": self.support.support_id,
                    "attributes": [value.content_id for value in attributes],
                    "derivation": self.derivation.derivation_id,
                    "scan": self.source_scan_id,
                    "references": [value.manifest_id for value in references],
                }
            ),
        )


def cartesianize_lidar_scan(scan: LidarScan, /) -> LidarPointProduct:
    """Convert calibrated ranges on declared rays to frame-preserving points."""
    if not isinstance(scan, LidarScan):
        raise TypeError("scan must be LidarScan.")
    support = scan.support
    range_field = scan.range_asset.field
    factor = float(
        conversion_factor(
            range_field.quantity.unit,
            support.coordinate_contract.length_unit,
        )
    )
    ranges = np.asarray(range_field.values, dtype=float) * factor
    valid = (
        np.asarray(range_field.valid_mask, dtype=bool)
        & np.asarray(support.active_mask, dtype=bool)
        & np.isfinite(ranges)
        & (ranges >= np.asarray(support.near))
        & (ranges <= np.asarray(support.far))
    )
    origins = np.asarray(support.origins)
    directions = np.asarray(support.directions)
    points = origins + np.where(valid, ranges, 0.0)[:, None] * directions
    point_support = PointSampleSupport(
        points,
        support.sample_ids,
        support.coordinate_contract,
        sample_times=support.sample_times,
        time_unit_id=support.time_unit_id,
        active_mask=valid,
    )
    source_assets = tuple(
        asset
        for asset in (scan.range_asset, scan.intensity_asset, scan.return_index_asset)
        if asset is not None
    )
    transformation_id = canonical_fingerprint(
        {
            "kind": "lidar-cartesian-conversion",
            "scan": scan.scan_id,
            "target_frame": support.coordinate_contract.reference_frame,
            "target_unit": support.coordinate_contract.length_unit.unit_id,
        }
    )
    derivation = DerivationRecord(
        scan.range_asset.derivation.origin,
        DataStage.DERIVED,
        tuple(asset.content_id for asset in source_assets),
        transformation_id,
        calibration_ids=()
        if scan.range_asset.acquisition is None
        else scan.range_asset.acquisition.calibration_ids,
        differentiation=scan.range_asset.derivation.differentiation,
    )
    point_sampling = SamplingSemantics(SpatialSamplingKind.POINT)
    attributes = tuple(
        QuantityField(
            f"{asset.field.field_id}.at-point",
            asset.field.quantity,
            asset.field.layout,
            point_support,
            point_sampling,
            asset.field.values,
            np.asarray(asset.field.valid_mask, dtype=bool) & valid,
            asset.field.uncertainty,
            asset.field.quality_flags,
        )
        for asset in source_assets
    )
    references_by_id = {
        reference.manifest_id: reference
        for asset in source_assets
        for reference in asset.references
    }
    return LidarPointProduct(
        point_support,
        attributes,
        derivation,
        scan.scan_id,
        tuple(references_by_id.values()),
    )


__all__ = ["LidarPointProduct", "LidarScan", "cartesianize_lidar_scan"]
