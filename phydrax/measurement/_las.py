#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded optional LAS point-product admission."""

from __future__ import annotations

from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path

import numpy as np

from .._physical import SpatialCoordinateContract
from ..interchange import AdapterLoss, AdapterReport, AdapterStatus
from ..qualification import ReferenceArtifactManifest
from ..units import ONE, SECOND
from ._asset import DataOrigin, DataStage, DerivationRecord
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import QuantitySpec, ValueKind, ValueLayout
from ._support import PointSampleSupport
from .lidar import LidarPointProduct


def _verify_reference(path: Path, reference: ReferenceArtifactManifest, /) -> None:
    if path.stat().st_size != reference.size_bytes:
        raise ValueError("LAS source size does not match its reference manifest.")
    digest = new_digest(reference.checksum_algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("LAS source checksum does not match its reference manifest.")


def _quantity(name: str, kind: str, unit=ONE) -> QuantitySpec:
    return QuantitySpec(
        "lidar",
        name,
        kind,
        unit,
        f"lidar.{kind}",
        support_association="point",
        reference_configuration="sensor-product",
    )


def _field(
    field_id: str,
    values: np.ndarray,
    support: PointSampleSupport,
    quantity: QuantitySpec,
    kind: ValueKind,
) -> QuantityField:
    return QuantityField(
        field_id,
        quantity,
        ValueLayout(kind),
        support,
        SamplingSemantics(SpatialSamplingKind.POINT),
        values,
        np.asarray(support.active_mask),
    )


class LasPointProvider:
    """Read bounded LAS/LAZ Cartesian point products through optional laspy."""

    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        product_id: str,
        intended_use: str = "research",
        maximum_points: int = 10_000_000,
        maximum_source_bytes: int = 2_000_000_000,
    ) -> tuple[LidarPointProduct, AdapterReport]:
        if util.find_spec("laspy") is None:
            raise ImportError(
                "LAS point admission requires the optional 'lidar-las' extra."
            )
        if not isinstance(reference, ReferenceArtifactManifest):
            raise TypeError("reference must be ReferenceArtifactManifest.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if isinstance(maximum_points, bool) or int(maximum_points) < 1:
            raise ValueError("maximum_points must be a positive integer.")
        if isinstance(maximum_source_bytes, bool) or int(maximum_source_bytes) < 1:
            raise ValueError("maximum_source_bytes must be a positive integer.")
        if reference.size_bytes > int(maximum_source_bytes):
            raise MemoryError(
                "LAS source exceeds maximum_source_bytes before provider admission."
            )
        requested = {
            "research": {},
            "commercial": {"commercial_use": True},
            "training": {"training_use": True},
            "redistribution": {"redistribution": True},
            "export": {"export": True},
        }
        if intended_use not in requested:
            raise ValueError("Unsupported intended_use.")
        reference.require_rights(**requested[intended_use])
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        _verify_reference(source, reference)
        backend = import_module("laspy")
        las = backend.read(str(source))
        count = len(las.points)
        if count < 1 or count > int(maximum_points):
            raise MemoryError(
                f"LAS point count {count} exceeds bounded maximum_points={maximum_points}."
            )
        points = np.column_stack(
            (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
        )
        identifiers = tuple(f"{product_id}.point.{index}" for index in range(count))
        support = PointSampleSupport(points, identifiers, coordinate_contract)
        dimensions = frozenset(str(name) for name in las.point_format.dimension_names)
        attributes: list[QuantityField] = []
        if "intensity" in dimensions:
            attributes.append(
                _field(
                    f"{product_id}.intensity",
                    np.asarray(las["intensity"]),
                    support,
                    _quantity("return-intensity", "detector-count"),
                    ValueKind.COUNT,
                )
            )
        if "return_number" in dimensions:
            attributes.append(
                _field(
                    f"{product_id}.return-number",
                    np.asarray(las["return_number"]),
                    support,
                    _quantity("return-number", "return-number"),
                    ValueKind.COUNT,
                )
            )
        if "number_of_returns" in dimensions:
            attributes.append(
                _field(
                    f"{product_id}.return-count",
                    np.asarray(las["number_of_returns"]),
                    support,
                    _quantity("return-count", "return-count"),
                    ValueKind.COUNT,
                )
            )
        if "classification" in dimensions:
            attributes.append(
                _field(
                    f"{product_id}.classification",
                    np.asarray(las["classification"]),
                    support,
                    _quantity("classification", "classification"),
                    ValueKind.CATEGORICAL,
                )
            )
        if "gps_time" in dimensions:
            attributes.append(
                _field(
                    f"{product_id}.gps-time",
                    np.asarray(las["gps_time"], dtype=float),
                    support,
                    _quantity("gps-time", "sample-time", SECOND),
                    ValueKind.REAL_SCALAR,
                )
            )
        loss = AdapterLoss(
            "acquisition.beam_geometry",
            "import",
            "unsupported",
            "LAS stores Cartesian point products and cannot recover authoritative emission rays or time-of-flight.",
            changes_interpretation=False,
            affected_capability_ids=("lidar.raw-ray-acquisition",),
        )
        report = AdapterReport(
            AdapterStatus.DECLARED_LOSS,
            "LAS",
            "phydrax-lidar-point-product",
            source_id=reference.manifest_id,
            target_id=support.support_id,
            coordinate_mapping=("scaled LAS XYZ -> declared point coordinate contract",),
            preserved_fields=tuple(field.quantity.name for field in attributes),
            assumptions=(
                "Declared coordinate contract matches the LAS scaled coordinate frame.",
            ),
            losses=(loss,),
            stage="lidar-point-admission",
        )
        derivation = DerivationRecord(
            DataOrigin.EXTERNAL,
            DataStage.DERIVED,
            transformation_id=report.report_id,
            adapter_report_ids=(report.report_id,),
        )
        product = LidarPointProduct(
            support,
            tuple(attributes),
            derivation,
            f"las:{reference.manifest_id}",
            (reference,),
        )
        return product, report


__all__ = ["LasPointProvider"]
