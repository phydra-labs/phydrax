#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded E57 multi-scan point-product admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..geometry.analytic import RigidFrame
from ..interchange import AdapterLoss, AdapterReport, AdapterStatus
from ..qualification import ReferenceArtifactManifest
from ..units import ONE
from ._asset import DataOrigin, DataStage, DerivationRecord, MeasurementAsset
from ._collection import (
    MeasurementCollection,
    MeasurementRole,
    MeasurementRoleAssignment,
)
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import QuantitySpec, ValueKind, ValueLayout
from ._support import PointSampleSupport
from .lidar import LidarPointProduct


@dataclass(frozen=True, slots=True)
class E57ScanRecord:
    scan_guid: str
    points: LidarPointProduct
    sensor_to_collection: RigidFrame
    row_indices: np.ndarray | None = None
    column_indices: np.ndarray | None = None
    image_asset_ids: tuple[str, ...] = ()
    scan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.points, LidarPointProduct):
            raise TypeError("points must be LidarPointProduct.")
        if (
            not isinstance(self.sensor_to_collection, RigidFrame)
            or self.sensor_to_collection.dimension != 3
        ):
            raise TypeError(
                "sensor_to_collection must be a three-dimensional RigidFrame."
            )
        row = (
            None
            if self.row_indices is None
            else np.array(self.row_indices, dtype=np.int64, copy=True)
        )
        column = (
            None
            if self.column_indices is None
            else np.array(self.column_indices, dtype=np.int64, copy=True)
        )
        for value, name in ((row, "row_indices"), (column, "column_indices")):
            if value is not None and value.shape != self.points.support.sample_shape:
                raise ValueError(f"{name} must match the point sample shape.")
            if value is not None:
                value.setflags(write=False)
        images = tuple(str(value) for value in self.image_asset_ids)
        if any(not value for value in images) or len(images) != len(set(images)):
            raise ValueError("image_asset_ids must be nonempty and unique.")
        object.__setattr__(self, "row_indices", row)
        object.__setattr__(self, "column_indices", column)
        object.__setattr__(self, "image_asset_ids", images)
        object.__setattr__(
            self,
            "scan_id",
            canonical_fingerprint(
                {
                    "kind": "e57-scan-record",
                    "guid": self.scan_guid,
                    "points": self.points.point_product_id,
                    "rotation": np.asarray(self.sensor_to_collection.rotation).tolist(),
                    "translation": np.asarray(
                        self.sensor_to_collection.translation
                    ).tolist(),
                    "row": None if row is None else row.tolist(),
                    "column": None if column is None else column.tolist(),
                    "images": list(images),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class E57ScanCollection:
    measurement_collection: MeasurementCollection
    scans: tuple[E57ScanRecord, ...]
    adapter_report: AdapterReport
    collection_id: str = field(init=False)

    def __post_init__(self) -> None:
        scans = tuple(self.scans)
        if not scans or any(not isinstance(value, E57ScanRecord) for value in scans):
            raise TypeError("scans must contain at least one E57ScanRecord.")
        if len({value.scan_guid for value in scans}) != len(scans):
            raise ValueError("E57 scan GUIDs must be unique.")
        asset_ids = {value.asset_id for value in self.measurement_collection.assets}
        if any(
            any(image not in asset_ids for image in scan.image_asset_ids)
            for scan in scans
        ):
            raise ValueError(
                "Every embedded E57 image must belong to the measurement collection."
            )
        object.__setattr__(self, "scans", scans)
        object.__setattr__(
            self,
            "collection_id",
            canonical_fingerprint(
                {
                    "kind": "e57-scan-collection",
                    "measurement_collection": self.measurement_collection.content_id,
                    "scans": [value.scan_id for value in scans],
                    "report": self.adapter_report.report_id,
                }
            ),
        )


def _verify(path: Path, reference: ReferenceArtifactManifest) -> None:
    if path.stat().st_size != reference.size_bytes:
        raise ValueError("E57 size disagrees with the reference manifest.")
    digest = new_digest(reference.checksum_algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("E57 checksum disagrees with the reference manifest.")


def _attribute(
    name: str, values: np.ndarray, support: PointSampleSupport, kind: ValueKind
) -> QuantityField:
    return QuantityField(
        name,
        QuantitySpec(
            "e57",
            name,
            name,
            ONE,
            f"e57.{name}",
            support_association="point",
            reference_configuration="scanner-product",
        ),
        ValueLayout(kind),
        support,
        SamplingSemantics(SpatialSamplingKind.POINT),
        values,
        np.asarray(support.active_mask),
    )


class E57Provider:
    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        campaign_id: str,
        maximum_source_bytes: int = 4_000_000_000,
        maximum_points_per_scan: int = 20_000_000,
        image_assets: tuple = (),
    ) -> E57ScanCollection:
        if util.find_spec("pye57") is None:
            raise ImportError("E57 admission requires the optional 'e57' extra.")
        if reference.size_bytes > maximum_source_bytes:
            raise MemoryError("E57 source exceeds maximum_source_bytes.")
        reference.require_rights()
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        _verify(source, reference)
        backend = import_module("pye57")
        document = backend.E57(str(source))
        records = []
        losses = []
        for scan_index in range(document.scan_count):
            header = document.get_header(scan_index)
            raw = document.read_scan_raw(scan_index)
            required = ("cartesianX", "cartesianY", "cartesianZ")
            if any(name not in raw for name in required):
                raise ValueError("E57 scan lacks Cartesian coordinates.")
            count = len(raw["cartesianX"])
            if count < 1 or count > maximum_points_per_scan:
                raise MemoryError("E57 scan exceeds maximum_points_per_scan.")
            points = np.column_stack(tuple(np.asarray(raw[name]) for name in required))
            invalid = np.zeros((count,), dtype=bool)
            if "cartesianInvalidState" in raw:
                invalid |= np.asarray(raw["cartesianInvalidState"]) != 0
            identifiers = tuple(
                f"{campaign_id}.scan-{scan_index}.point-{index}" for index in range(count)
            )
            support = PointSampleSupport(
                points, identifiers, coordinate_contract, active_mask=~invalid
            )
            attributes = []
            if "intensity" in raw:
                attributes.append(
                    _attribute(
                        f"{campaign_id}.scan-{scan_index}.intensity",
                        np.asarray(raw["intensity"], dtype=float),
                        support,
                        ValueKind.REAL_SCALAR,
                    )
                )
            if all(name in raw for name in ("colorRed", "colorGreen", "colorBlue")):
                color = np.stack(
                    tuple(
                        np.asarray(raw[name], dtype=float)
                        for name in ("colorRed", "colorGreen", "colorBlue")
                    ),
                    axis=-1,
                )
                attributes.append(
                    QuantityField(
                        f"{campaign_id}.scan-{scan_index}.color",
                        QuantitySpec(
                            "e57",
                            "color",
                            "detector-color",
                            ONE,
                            "sensor.color",
                            support_association="point",
                        ),
                        ValueLayout(
                            ValueKind.VECTOR,
                            (3,),
                            ("red", "green", "blue"),
                            "detector-color-basis",
                        ),
                        support,
                        SamplingSemantics(SpatialSamplingKind.POINT),
                        color,
                        ~invalid,
                    )
                )
            row = (
                None
                if "rowIndex" not in raw
                else np.asarray(raw["rowIndex"], dtype=np.int64)
            )
            column = (
                None
                if "columnIndex" not in raw
                else np.asarray(raw["columnIndex"], dtype=np.int64)
            )
            report_id = canonical_fingerprint(
                {
                    "kind": "e57-scan-import",
                    "source": reference.manifest_id,
                    "scan": scan_index,
                }
            )
            product = LidarPointProduct(
                support,
                tuple(attributes),
                DerivationRecord(
                    DataOrigin.EXTERNAL,
                    DataStage.DERIVED,
                    transformation_id=report_id,
                    adapter_report_ids=(report_id,),
                ),
                f"e57:{reference.manifest_id}:{scan_index}",
                (reference,),
            )
            rotation = np.asarray(header.rotation_matrix)
            translation = np.asarray(header.translation)
            records.append(
                E57ScanRecord(
                    str(header.guid),
                    product,
                    RigidFrame(rotation, translation),
                    row,
                    column,
                    tuple(value.asset_id for value in image_assets),
                )
            )
        document.close()
        if not image_assets:
            losses.append(
                AdapterLoss(
                    "images2D",
                    "import",
                    "unsupported",
                    "The selected backend did not expose embedded E57 images; none were claimed.",
                    changes_interpretation=False,
                    affected_capability_ids=("e57.embedded-images",),
                )
            )
        status = AdapterStatus.LOSSLESS if not losses else AdapterStatus.DECLARED_LOSS
        report = AdapterReport(
            status,
            "E57",
            "phydrax-e57-scan-collection",
            source_id=reference.manifest_id,
            target_id=campaign_id,
            coordinate_mapping=(
                "E57 Cartesian points -> declared spatial coordinate contract",
            ),
            preserved_fields=(
                "scan-guid",
                "pose",
                "invalid-state",
                "row-column",
                "intensity",
                "color",
            ),
            losses=tuple(losses),
            stage="e57-admission",
        )
        point_assets = tuple(
            _point_anchor(campaign_id, index, record, reference)
            for index, record in enumerate(records)
        )
        assets = point_assets + tuple(image_assets)
        roles = tuple(
            MeasurementRoleAssignment(value.asset_id, MeasurementRole.DERIVED_PRODUCT)
            for value in point_assets
        ) + tuple(
            MeasurementRoleAssignment(value.asset_id, MeasurementRole.REFERENCE)
            for value in image_assets
        )
        collection = MeasurementCollection(
            campaign_id,
            campaign_id,
            assets,
            roles,
        )
        return E57ScanCollection(collection, tuple(records), report)


def _point_anchor(
    campaign_id: str,
    scan_index: int,
    record: E57ScanRecord,
    reference: ReferenceArtifactManifest,
) -> MeasurementAsset:
    first = record.points
    support = first.support
    field = QuantityField(
        f"{campaign_id}.point-count",
        QuantitySpec("e57", "active-point", "active-point", ONE, "e57.active-point"),
        ValueLayout(ValueKind.COUNT),
        support,
        SamplingSemantics(SpatialSamplingKind.POINT),
        np.asarray(support.active_mask, dtype=np.int64),
        np.asarray(support.active_mask),
    )

    asset = MeasurementAsset.from_single_reference(
        f"{campaign_id}.scan-{scan_index}.anchor",
        field,
        reference,
        DerivationRecord(
            DataOrigin.EXTERNAL,
            DataStage.DERIVED,
            transformation_id=first.derivation.derivation_id,
        ),
    )
    return asset


__all__ = ["E57Provider", "E57ScanCollection", "E57ScanRecord"]
