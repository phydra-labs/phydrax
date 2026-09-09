#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded selection of resident measurement collections."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._asset import DataStage, DerivationRecord, MeasurementAsset
from ._collection import MeasurementCollection, MeasurementRoleAssignment
from ._field import IndependentStandardUncertainty, QualityFlag, QuantityField
from ._quantity import canonical_quantity_text
from ._support import IndexSampleSupport, PointSampleSupport, RaySampleSupport
from ._time import SampleTimeAxis


@dataclass(frozen=True, slots=True)
class MeasurementSelectionPlan:
    asset_ids: tuple[str, ...]
    start: int = 0
    stop: int | None = None
    maximum_samples: int = 1_000_000
    maximum_decoded_bytes: int = 1_073_741_824
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        assets = tuple(
            canonical_quantity_text(value, "asset_id") for value in self.asset_ids
        )
        if not assets or len(assets) != len(set(assets)):
            raise ValueError("asset_ids must be nonempty and unique.")
        start = int(self.start)
        stop = None if self.stop is None else int(self.stop)
        if start < 0 or (stop is not None and stop <= start):
            raise ValueError("Selection requires 0 <= start < stop.")
        if self.maximum_samples < 1 or self.maximum_decoded_bytes < 1:
            raise ValueError("Selection resource limits must be positive.")
        object.__setattr__(self, "asset_ids", assets)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-selection-plan",
                    "assets": list(assets),
                    "start": start,
                    "stop": stop,
                    "maximum_samples": self.maximum_samples,
                    "maximum_decoded_bytes": self.maximum_decoded_bytes,
                }
            ),
        )

    def apply(self, collection: MeasurementCollection, /) -> MeasurementCollection:
        if not isinstance(collection, MeasurementCollection):
            raise TypeError("collection must be MeasurementCollection.")
        selected = tuple(collection.asset(value) for value in self.asset_ids)
        result_assets = tuple(
            _slice_asset(value, self.start, self.stop, self.plan_id) for value in selected
        )
        masks = tuple(value.field.valid_mask for value in result_assets)
        assert all(value is not None for value in masks)
        sample_count = sum(value.size for value in masks if value is not None)
        decoded_bytes = sum(
            asset.field.values.nbytes + mask.nbytes
            for asset, mask in zip(result_assets, masks, strict=True)
            if mask is not None
        )
        if sample_count > self.maximum_samples:
            raise MemoryError("Selection exceeds maximum_samples.")
        if decoded_bytes > self.maximum_decoded_bytes:
            raise MemoryError("Selection exceeds maximum_decoded_bytes.")
        roles_by_id = {value.asset_id: value.role for value in collection.roles}
        roles = tuple(
            MeasurementRoleAssignment(value.asset_id, roles_by_id[source.asset_id])
            for value, source in zip(result_assets, selected, strict=True)
        )
        return MeasurementCollection(
            f"{collection.collection_id}:selection:{self.plan_id[:12]}",
            collection.campaign_id,
            result_assets,
            roles,
            parent_collection_ids=(collection.content_id,),
            platform_id=collection.platform_id,
            metadata={"selection_plan_id": self.plan_id},
        )


def _slice_support(support, start: int, stop: int | None):
    selection = slice(start, stop)
    if isinstance(support, IndexSampleSupport):
        shape = (
            len(range(*selection.indices(support.sample_shape[0]))),
        ) + support.sample_shape[1:]
        time_axis = support.time_axis
        if time_axis is not None and support.time_axis_dimension == 0:
            time_axis = SampleTimeAxis(
                time_axis.label,
                time_axis.sample_times[selection],
                time_axis.time_unit,
                time_axis.basis,
                time_axis.origin,
            )
        return IndexSampleSupport(
            shape,
            support.axis_labels,
            time_axis,
            support.time_axis_dimension,
            support.frame_id,
        )
    if isinstance(support, PointSampleSupport):
        return PointSampleSupport(
            support.points[selection],
            support.sample_ids[selection],
            support.coordinate_contract,
            sample_times=None
            if support.sample_times is None
            else support.sample_times[selection],
            time_unit_id=support.time_unit_id,
            active_mask=support.active_mask[selection],
        )
    if isinstance(support, RaySampleSupport):
        return RaySampleSupport(
            support.origins[selection],
            support.directions[selection],
            support.sample_ids[selection],
            support.coordinate_contract,
            sample_times=None
            if support.sample_times is None
            else support.sample_times[selection],
            time_unit_id=support.time_unit_id,
            active_mask=support.active_mask[selection],
            near=support.near[selection],
            far=support.far[selection],
        )
    raise TypeError("Selection supports index, point, or ray measurements.")


def _slice_asset(
    asset: MeasurementAsset, start: int, stop: int | None, plan_id: str
) -> MeasurementAsset:
    selection = slice(start, stop)
    source = asset.field
    support = _slice_support(source.support, start, stop)
    uncertainty = None
    if source.uncertainty is not None:
        uncertainty = IndependentStandardUncertainty(
            source.uncertainty.values[selection], source.uncertainty.unit
        )
    flags = tuple(
        QualityFlag(value.name, value.mask[selection], value.meaning)
        for value in source.quality_flags
    )
    field = QuantityField(
        f"{source.field_id}:selection:{plan_id[:12]}",
        source.quantity,
        source.layout,
        support,
        source.sampling,
        source.values[selection],
        np.asarray(source.valid_mask)[selection],
        uncertainty,
        flags,
    )
    derivation = DerivationRecord(
        asset.derivation.origin,
        DataStage.DERIVED,
        (asset.content_id,),
        plan_id,
        differentiation=asset.derivation.differentiation,
    )
    return MeasurementAsset(
        f"{asset.asset_id}:selection:{plan_id[:12]}",
        field,
        asset.acquisition,
        asset.references,
        derivation,
        asset.intended_use,
        {"selection_plan_id": plan_id},
    )


__all__ = ["MeasurementSelectionPlan"]
