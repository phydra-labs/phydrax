#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, provenance-preserving GDSII and OASIS layout import."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast, Never

import numpy as np

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
)
from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..geometry._polygon import signed_area2, validate_simple_polygon
from ..geometry.simplicial import PlanarMeshRegion
from ..units import METER
from ._report import (
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    negotiate_adapter,
)


class LayoutFormat(str, Enum):
    """Supported neutral planar layout encodings."""

    GDSII = "gdsii"
    OASIS = "oasis"


_PROPERTY_CAPABILITY_ID = AdapterCapability("layout.source-properties").capability_id
_LABEL_CAPABILITY_ID = AdapterCapability("layout.source-labels").capability_id
_PATH_EXACTNESS_CAPABILITY_ID = AdapterCapability(
    "layout.exact-path-geometry"
).capability_id
_OPTIONAL_SEMANTICS = (
    (_PROPERTY_CAPABILITY_ID, "layout.source-properties"),
    (_LABEL_CAPABILITY_ID, "layout.source-labels"),
    (_PATH_EXACTNESS_CAPABILITY_ID, "layout.exact-path-geometry"),
)


@dataclass(frozen=True, slots=True, order=True)
class LayoutLayerKey:
    """Exact GDSII/OASIS layer and datatype pair."""

    layer: int
    datatype: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, Integral)
            or int(self.layer) < 0
        ):
            raise ValueError("Layout layers must be nonnegative integers.")
        if (
            isinstance(self.datatype, bool)
            or not isinstance(self.datatype, Integral)
            or int(self.datatype) < 0
        ):
            raise ValueError("Layout datatypes must be nonnegative integers.")
        object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "datatype", int(self.datatype))


@dataclass(frozen=True, slots=True, eq=False)
class LayoutRegion:
    """One physical planar region with source-occurrence provenance."""

    layer: LayoutLayerKey
    geometry: PlanarMeshRegion
    coordinate_contract: SpatialCoordinateContract
    occurrence_path: tuple[str, ...]
    source_format: LayoutFormat
    source_digest: str
    region_id: str
    provenance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer, LayoutLayerKey):
            raise TypeError("layer must be a LayoutLayerKey.")
        if not isinstance(self.geometry, PlanarMeshRegion):
            raise TypeError("geometry must be a PlanarMeshRegion.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        path = tuple(str(value) for value in self.occurrence_path)
        if not path or any(not value for value in path):
            raise ValueError("Layout occurrence paths must be nonempty.")
        if not isinstance(self.source_format, LayoutFormat):
            raise TypeError("source_format must be a LayoutFormat.")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise ValueError("source_digest must be nonempty.")
        if not isinstance(self.region_id, str) or not self.region_id:
            raise ValueError("region_id must be nonempty.")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be nonempty.")
        object.__setattr__(self, "occurrence_path", path)

    def __eq__(self, other: object) -> bool:
        """Compare physical geometry, deliberately excluding file provenance."""

        return isinstance(other, LayoutRegion) and self.region_id == other.region_id

    def __hash__(self) -> int:
        return hash(self.region_id)


@dataclass(frozen=True, slots=True, eq=False)
class LayoutModel:
    """Material-neutral physical layout in one explicit coordinate contract."""

    coordinate_contract: SpatialCoordinateContract
    regions: tuple[LayoutRegion, ...]
    model_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        regions = tuple(self.regions)
        if not regions or not all(isinstance(value, LayoutRegion) for value in regions):
            raise ValueError("A layout model requires at least one LayoutRegion.")
        if any(
            value.coordinate_contract.spatial_id != self.coordinate_contract.spatial_id
            for value in regions
        ):
            raise ValueError(
                "Every layout region must use the model coordinate contract."
            )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "physical-layout-model",
                    "coordinate_contract": self.coordinate_contract.spatial_id,
                    "regions": sorted(value.region_id for value in regions),
                }
            ),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LayoutModel) and self.model_id == other.model_id

    def __hash__(self) -> int:
        return hash(self.model_id)


@dataclass(frozen=True, slots=True)
class LayoutImportPolicy:
    """Explicit format, coordinate, hierarchy, approximation, and resource policy."""

    format: LayoutFormat
    coordinate_contract: SpatialCoordinateContract
    limits: ResourceLimits
    top_cell: str | None = None
    path_tolerance: float | None = None
    waivers: tuple[AdapterWaiver, ...] = ()
    waived_loss_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.format, LayoutFormat):
            raise TypeError("format must be a LayoutFormat.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "Layout import requires an explicit Cartesian coordinate contract."
            )
        if not isinstance(self.limits, ResourceLimits):
            raise TypeError("limits must be ResourceLimits.")
        top = None if self.top_cell is None else str(self.top_cell).strip()
        if self.top_cell is not None and not top:
            raise ValueError("top_cell must be nonempty when supplied.")
        tolerance = self.path_tolerance
        if tolerance is not None:
            if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
                raise TypeError("path_tolerance must be a real number or None.")
            tolerance = float(tolerance)
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError("path_tolerance must be finite and positive.")
        waivers = tuple(self.waivers)
        if not all(isinstance(value, AdapterWaiver) for value in waivers):
            raise TypeError("waivers must contain AdapterWaiver values.")
        paths = tuple(str(value) for value in self.waived_loss_paths)
        if any(not value for value in paths) or len(set(paths)) != len(paths):
            raise ValueError("waived_loss_paths must contain unique nonempty paths.")
        object.__setattr__(self, "top_cell", top)
        object.__setattr__(self, "path_tolerance", tolerance)
        object.__setattr__(self, "waivers", waivers)
        object.__setattr__(self, "waived_loss_paths", paths)


@dataclass(frozen=True, slots=True)
class LayoutImportResult:
    """Native layout plus exact source and semantic-accounting evidence."""

    model: LayoutModel
    format: LayoutFormat
    top_cell: str
    source_digest: str
    source_user_unit_meters: float
    source_database_unit_meters: float
    resource_manifest: ResourceManifest
    report: AdapterReport
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, LayoutModel):
            raise TypeError("model must be a LayoutModel.")
        if not isinstance(self.format, LayoutFormat):
            raise TypeError("format must be a LayoutFormat.")
        if not isinstance(self.top_cell, str) or not self.top_cell:
            raise ValueError("top_cell must be nonempty.")
        if not isinstance(self.resource_manifest, ResourceManifest):
            raise TypeError("resource_manifest must be a ResourceManifest.")
        if self.source_digest != self.resource_manifest.content_sha256:
            raise ValueError("source_digest must match the exact resource manifest.")
        units = (
            float(self.source_user_unit_meters),
            float(self.source_database_unit_meters),
        )
        if not all(math.isfinite(value) and value > 0.0 for value in units):
            raise ValueError("Layout source units must be finite and positive.")
        if not isinstance(self.report, AdapterReport) or not self.report.valid:
            raise ValueError("A layout import result requires a valid AdapterReport.")
        object.__setattr__(
            self,
            "result_id",
            canonical_fingerprint(
                {
                    "kind": "layout-import-result",
                    "model": self.model.model_id,
                    "format": self.format.value,
                    "top_cell": self.top_cell,
                    "source_digest": self.source_digest,
                    "source_user_unit_meters": units[0],
                    "source_database_unit_meters": units[1],
                    "resource_manifest": self.resource_manifest.manifest_id,
                    "report": self.report.report_id,
                }
            ),
        )


class LayoutAdapterError(AdapterError):
    """Typed, fail-closed layout import error."""

    report: AdapterReport | None
    resource_manifest: ResourceManifest | None

    def __init__(
        self,
        status: AdapterStatus,
        message: str,
        /,
        *,
        report: AdapterReport | None = None,
        resource_manifest: ResourceManifest | None = None,
    ):
        self.report = report
        self.resource_manifest = resource_manifest
        super().__init__(status, message)


@dataclass(slots=True)
class _DecodeCounter:
    maximum: int
    nodes: int

    def bump(self, count: int = 1) -> None:
        self.nodes += int(count)
        if self.nodes > self.maximum:
            raise ResourceReadError(
                "limit", "Expanded layout node count exceeds its resource limit."
            )


class _LayoutDecoder:
    def __init__(
        self,
        gdstk: Any,
        library: Any,
        resource: BoundedResource,
        policy: LayoutImportPolicy,
        scale: float,
    ):
        self.gdstk = gdstk
        self.library = library
        self.resource = resource
        self.policy = policy
        self.scale = scale
        self.cells = self._cells_by_name()
        self.losses: list[AdapterLoss] = []
        self.regions: list[LayoutRegion] = []
        self.maximum_depth = 0
        base_nodes, attributes = self._base_counts()
        if attributes > policy.limits.max_attributes:
            raise ResourceReadError(
                "limit", "Layout attribute count exceeds its resource limit."
            )
        self.attributes = attributes
        self.counter = _DecodeCounter(policy.limits.max_nodes, 0)
        self.counter.bump(base_nodes)

    def _cells_by_name(self) -> dict[str, Any]:
        cells: dict[str, Any] = {}
        for cell in self.library.cells:
            name = str(cell.name)
            if not name or name in cells:
                _fail(
                    AdapterStatus.MALFORMED_SOURCE,
                    "Layout cells must have unique nonempty names.",
                    resource_manifest=self.resource.manifest,
                )
            cells[name] = cell
        if not cells:
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                "The layout library contains no cells.",
                resource_manifest=self.resource.manifest,
            )
        return cells

    @staticmethod
    def _property_count(value: Any) -> int:
        return sum(max(1, len(entry) - 1) for entry in value.properties)

    def _base_counts(self) -> tuple[int, int]:
        nodes = 0
        attributes = 0
        for cell in self.cells.values():
            elements = (
                tuple(cell.polygons)
                + tuple(cell.paths)
                + tuple(cell.references)
                + tuple(cell.labels)
            )
            nodes += 1 + len(elements)
            attributes += self._property_count(cell)
            attributes += sum(self._property_count(value) for value in elements)
        return nodes, attributes

    def _resolved_cell(self, reference: Any, path: str) -> Any:
        cell = reference.cell
        if not isinstance(cell, self.gdstk.Cell):
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"Layout reference {path} does not resolve to a decoded Cell.",
                resource_manifest=self.resource.manifest,
            )
        name = str(cell.name)
        if name not in self.cells or self.cells[name] is not cell:
            _fail(
                AdapterStatus.INCONSISTENT_SOURCE,
                f"Layout reference {path} resolves outside the decoded library.",
                resource_manifest=self.resource.manifest,
            )
        return cell

    def _validate_graph(self) -> None:
        visited: set[str] = set()

        def visit(cell: Any, active: tuple[str, ...], depth: int) -> None:
            name = str(cell.name)
            if name in active:
                cycle = " -> ".join((*active, name))
                _fail(
                    AdapterStatus.MALFORMED_SOURCE,
                    f"Cyclic layout hierarchy is forbidden: {cycle}.",
                    resource_manifest=self.resource.manifest,
                )
            if depth > self.policy.limits.max_depth:
                raise ResourceReadError(
                    "limit", "Layout hierarchy exceeds its resource depth limit."
                )
            if name in visited:
                return
            branch = (*active, name)
            for index, reference in enumerate(cell.references):
                path = f"{name}/reference:{index}"
                child = self._resolved_cell(reference, path)
                _reference_transform(reference, path, self.resource.manifest)
                visit(child, branch, depth + 1)
            visited.add(name)

        for cell in self.cells.values():
            visit(cell, (), 1)

    def _top_cell(self) -> Any:
        referenced = {
            str(self._resolved_cell(reference, f"{cell.name}/reference:{index}").name)
            for cell in self.cells.values()
            for index, reference in enumerate(cell.references)
        }
        top_names = tuple(sorted(set(self.cells) - referenced))
        selected = self.policy.top_cell
        if selected is not None:
            if selected not in self.cells:
                _fail(
                    AdapterStatus.INCONSISTENT_SOURCE,
                    f"Explicit top cell {selected!r} is absent from the layout.",
                    resource_manifest=self.resource.manifest,
                )
            return self.cells[selected]
        if len(top_names) != 1:
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Layout top cell is ambiguous; LayoutImportPolicy.top_cell is required.",
                resource_manifest=self.resource.manifest,
            )
        return self.cells[top_names[0]]

    @staticmethod
    def _repetition_count(repetition: Any) -> int:
        if repetition.columns is not None:
            return int(repetition.columns) * int(repetition.rows)
        if repetition.offsets is not None:
            return np.asarray(repetition.offsets).shape[0] + 1
        if repetition.x_offsets is not None:
            return np.asarray(repetition.x_offsets).size + 1
        if repetition.y_offsets is not None:
            return np.asarray(repetition.y_offsets).size + 1
        return 1

    def _offsets(self, repetition: Any, path: str) -> np.ndarray:
        if repetition is None:
            return np.zeros((1, 2), dtype=np.float64)
        count = self._repetition_count(repetition)
        if count == 1 and repetition.get_offsets().size == 0:
            return np.zeros((1, 2), dtype=np.float64)
        if count <= 0 or count > self.policy.limits.max_nodes - self.counter.nodes:
            raise ResourceReadError(
                "limit", f"Layout repetition {path} exceeds its resource node limit."
            )
        offsets = np.asarray(repetition.get_offsets(), dtype=np.float64)
        if offsets.shape != (count, 2) or not np.all(np.isfinite(offsets)):
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"Layout repetition {path} has invalid offsets.",
                resource_manifest=self.resource.manifest,
            )
        return offsets

    def _add_loss(self, loss: AdapterLoss) -> None:
        if len(self.losses) >= self.policy.limits.max_losses:
            raise ResourceReadError(
                "limit", "Layout semantic-loss count exceeds its resource limit."
            )
        self.losses.append(loss)

    def _property_loss(self, value: Any, path: str) -> None:
        if value.properties:
            self._add_loss(
                AdapterLoss(
                    f"{path}/properties",
                    "import",
                    "dropped",
                    "Layout element properties have no declared native region semantic.",
                    changes_interpretation=True,
                    affected_capability_ids=(_PROPERTY_CAPABILITY_ID,),
                )
            )

    def _label_losses(self, cell: Any, occurrence: tuple[str, ...]) -> None:
        for index, label in enumerate(cell.labels):
            path = "/".join((*occurrence, f"label:{index}"))
            offsets = self._offsets(label.repetition, path)
            self.counter.bump(offsets.shape[0])
            self._add_loss(
                AdapterLoss(
                    path,
                    "import",
                    "dropped",
                    "Layout labels have no declared native planar-region semantic.",
                    changes_interpretation=True,
                    affected_capability_ids=(_LABEL_CAPABILITY_ID,),
                )
            )
            self._property_loss(label, path)

    def _region(
        self,
        points: np.ndarray,
        layer: LayoutLayerKey,
        occurrence: tuple[str, ...],
    ) -> None:
        canonical = _canonical_polygon(points)
        self.counter.bump(canonical.shape[0])
        physical_id = canonical_fingerprint(
            {
                "kind": "layout-region",
                "coordinate_contract": self.policy.coordinate_contract.spatial_id,
                "layer": [layer.layer, layer.datatype],
                "vertices": canonical.tolist(),
            }
        )
        provenance_id = canonical_fingerprint(
            {
                "kind": "layout-region-provenance",
                "format": self.policy.format.value,
                "source_digest": self.resource.manifest.content_sha256,
                "occurrence_path": list(occurrence),
            }
        )
        geometry = PlanarMeshRegion(
            canonical,
            (tuple(range(canonical.shape[0])),),
            feature_id=f"layout-region-{physical_id}",
        )
        self.regions.append(
            LayoutRegion(
                layer,
                geometry,
                self.policy.coordinate_contract,
                occurrence,
                self.policy.format,
                self.resource.manifest.content_sha256,
                physical_id,
                provenance_id,
            )
        )

    def _polygon(
        self,
        polygon: Any,
        transform: np.ndarray,
        occurrence: tuple[str, ...],
        path: str,
    ) -> None:
        self._property_loss(polygon, path)
        offsets = self._offsets(polygon.repetition, path)
        points = np.asarray(polygon.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"Layout polygon {path} has nonfinite or malformed coordinates.",
                resource_manifest=self.resource.manifest,
            )
        for repetition_index, offset in enumerate(offsets):
            self.counter.bump()
            instance = transform @ _translation(offset)
            transformed = _transform_points(points, instance)
            self._region(
                transformed,
                LayoutLayerKey(polygon.layer, polygon.datatype),
                (
                    *occurrence,
                    f"polygon:{path.rsplit(':', 1)[-1]}:rep:{repetition_index}",
                ),
            )

    def _path(
        self,
        path_value: Any,
        transform: np.ndarray,
        occurrence: tuple[str, ...],
        index: int,
    ) -> None:
        path = "/".join((*occurrence, f"path:{index}"))
        if self.policy.path_tolerance is None:
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"Layout path {path} requires an explicit positive path_tolerance.",
                resource_manifest=self.resource.manifest,
            )
        path_value.tolerance = self.policy.path_tolerance / self.scale
        polygons = tuple(path_value.to_polygons())
        if not polygons:
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"Layout path {path} has no supported polygonal representation.",
                resource_manifest=self.resource.manifest,
            )
        offsets = self._offsets(path_value.repetition, path)
        if (
            len(polygons) * offsets.shape[0]
            > self.policy.limits.max_nodes - self.counter.nodes
        ):
            raise ResourceReadError(
                "limit", f"Polygonized layout path {path} exceeds its node limit."
            )
        self._add_loss(
            AdapterLoss(
                path,
                "import",
                "transformed",
                "A supported layout path was polygonized under the caller-declared "
                f"{self.policy.path_tolerance:g} target-unit tolerance.",
                changes_interpretation=True,
                affected_capability_ids=(_PATH_EXACTNESS_CAPABILITY_ID,),
            )
        )
        self._property_loss(path_value, path)
        for repetition_index, offset in enumerate(offsets):
            instance = transform @ _translation(offset)
            for polygon_index, polygon in enumerate(polygons):
                self.counter.bump()
                points = _transform_points(
                    np.asarray(polygon.points, dtype=np.float64), instance
                )
                self._region(
                    points,
                    LayoutLayerKey(polygon.layer, polygon.datatype),
                    (
                        *occurrence,
                        f"path:{index}:polygon:{polygon_index}:rep:{repetition_index}",
                    ),
                )

    def _visit(
        self,
        cell: Any,
        transform: np.ndarray,
        occurrence: tuple[str, ...],
        depth: int,
    ) -> None:
        self.maximum_depth = max(self.maximum_depth, depth)
        if depth > self.policy.limits.max_depth:
            raise ResourceReadError(
                "limit", "Expanded layout hierarchy exceeds its depth limit."
            )
        self._property_loss(cell, "/".join(occurrence))
        self._label_losses(cell, occurrence)
        for index, polygon in enumerate(cell.polygons):
            path = "/".join((*occurrence, f"polygon:{index}"))
            self._polygon(polygon, transform, occurrence, path)
        for index, path_value in enumerate(cell.paths):
            self._path(path_value, transform, occurrence, index)
        for index, reference in enumerate(cell.references):
            path = "/".join((*occurrence, f"reference:{index}"))
            self._property_loss(reference, path)
            child = self._resolved_cell(reference, path)
            reference_transform = _reference_transform(
                reference, path, self.resource.manifest
            )
            offsets = self._offsets(reference.repetition, path)
            for repetition_index, offset in enumerate(offsets):
                self.counter.bump()
                instance = transform @ _translation(offset) @ reference_transform
                if (
                    not np.all(np.isfinite(instance))
                    or abs(np.linalg.det(instance[:2, :2])) <= 0.0
                ):
                    _fail(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"Layout reference {path} has a nonfinite or singular transform.",
                        resource_manifest=self.resource.manifest,
                    )
                child_occurrence = (
                    *occurrence,
                    f"reference:{index}:{child.name}:rep:{repetition_index}",
                )
                self._visit(child, instance, child_occurrence, depth + 1)

    def decode(self) -> tuple[LayoutModel, str, int, int, tuple[AdapterLoss, ...]]:
        self._validate_graph()
        top = self._top_cell()
        root_transform = np.asarray(
            ((self.scale, 0.0, 0.0), (0.0, self.scale, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        self._visit(top, root_transform, (str(top.name),), 1)
        if not self.regions:
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Selected layout top cell expands to no planar regions.",
                resource_manifest=self.resource.manifest,
            )
        model = LayoutModel(self.policy.coordinate_contract, tuple(self.regions))
        return (
            model,
            str(top.name),
            self.maximum_depth,
            self.counter.nodes,
            tuple(self.losses),
        )


def _fail(
    status: AdapterStatus,
    message: str,
    /,
    *,
    report: AdapterReport | None = None,
    resource_manifest: ResourceManifest | None = None,
) -> Never:
    raise LayoutAdapterError(
        status,
        message,
        report=report,
        resource_manifest=resource_manifest,
    )


def _resource_failure(error: ResourceReadError, /) -> Never:
    status = (
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        if error.reason == "policy"
        else AdapterStatus.INCONSISTENT_SOURCE
        if error.reason == "inconsistent"
        else AdapterStatus.MALFORMED_SOURCE
    )
    raise LayoutAdapterError(status, str(error)) from error


def _gdstk(resource: BoundedResource) -> Any:
    try:
        return cast(Any, import_module("gdstk"))
    except ImportError as error:
        raise LayoutAdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "GDSII/OASIS import requires the optional gdstk runtime.",
            resource_manifest=resource.manifest,
        ) from error


def _parse_library(
    gdstk: Any, resource: BoundedResource, policy: LayoutImportPolicy
) -> Any:
    suffix = ".gds" if policy.format is LayoutFormat.GDSII else ".oas"
    descriptor, name = tempfile.mkstemp(prefix="phydrax-layout-", suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(resource.data)
        if policy.format is LayoutFormat.GDSII:
            return gdstk.read_gds(name, unit=0, tolerance=0)
        return gdstk.read_oas(name, unit=0, tolerance=0)
    except (OSError, RuntimeError, ValueError) as error:
        raise LayoutAdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            f"gdstk could not decode the bounded {policy.format.value} resource.",
            resource_manifest=resource.manifest,
        ) from error
    finally:
        Path(name).unlink(missing_ok=True)


def _source_units(
    library: Any, policy: LayoutImportPolicy, resource: BoundedResource
) -> tuple[float, float, float]:
    user_unit = float(library.unit)
    database_unit = float(library.precision)
    if not all(
        math.isfinite(value) and value > 0.0 for value in (user_unit, database_unit)
    ):
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            "Layout user and database units must be finite and positive.",
            resource_manifest=resource.manifest,
        )
    target_unit = policy.coordinate_contract.length_unit
    if target_unit.reference_system_id != METER.reference_system_id:
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Layout metric units require a target length unit in the SI reference system.",
            resource_manifest=resource.manifest,
        )
    target_unit_meters = float(target_unit.scale_to_reference)
    scale = user_unit / target_unit_meters
    if not math.isfinite(scale) or scale <= 0.0:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            "Layout coordinate unit conversion is nonfinite or nonpositive.",
            resource_manifest=resource.manifest,
        )
    return user_unit, database_unit, scale


def _translation(offset: np.ndarray) -> np.ndarray:
    return np.asarray(
        ((1.0, 0.0, float(offset[0])), (0.0, 1.0, float(offset[1])), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _reference_transform(
    reference: Any,
    path: str,
    manifest: ResourceManifest,
) -> np.ndarray:
    origin = np.asarray(reference.origin, dtype=np.float64)
    rotation = reference.rotation
    magnification = reference.magnification
    if (
        origin.shape != (2,)
        or not np.all(np.isfinite(origin))
        or isinstance(rotation, bool)
        or not isinstance(rotation, Real)
        or isinstance(magnification, bool)
        or not isinstance(magnification, Real)
    ):
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"Layout reference {path} has an invalid transform.",
            resource_manifest=manifest,
        )
    angle = float(rotation)
    scale = float(magnification)
    if not math.isfinite(angle) or not math.isfinite(scale) or scale <= 0.0:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"Layout reference {path} has a nonfinite or nonpositive transform.",
            resource_manifest=manifest,
        )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    reflection = -1.0 if reference.x_reflection else 1.0
    linear = scale * np.asarray(
        ((cosine, -sine * reflection), (sine, cosine * reflection)), dtype=np.float64
    )
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = linear
    result[:2, 2] = origin
    return result


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        _fail(AdapterStatus.MALFORMED_SOURCE, "Layout polygon coordinates are invalid.")
    homogeneous = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :2]


def _canonical_polygon(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        _fail(AdapterStatus.MALFORMED_SOURCE, "Layout polygon coordinates are invalid.")
    if values.shape[0] > 1 and np.array_equal(values[0], values[-1]):
        values = values[:-1]
    if values.shape[0] < 3:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            "Layout polygons require at least three vertices.",
        )
    keep = np.ones((values.shape[0],), dtype=np.bool_)
    keep[1:] = np.any(values[1:] != values[:-1], axis=1)
    values = values[keep]
    if values.shape[0] < 3 or len({tuple(row) for row in values}) < 3:
        _fail(AdapterStatus.MALFORMED_SOURCE, "Layout polygons are degenerate.")
    try:
        validate_simple_polygon(values, require_counter_clockwise=False)
    except ValueError:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            "Layout polygon is invalid or self-intersecting.",
        )
    signed_area = 0.5 * signed_area2(values)
    if not math.isfinite(signed_area) or signed_area == 0.0:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            "Layout polygon has zero or nonfinite area.",
        )
    if signed_area < 0.0:
        values = values[::-1]
    values = np.array(values, copy=True)
    values[values == 0.0] = 0.0
    start = min(range(values.shape[0]), key=lambda index: tuple(values[index]))
    return np.ascontiguousarray(np.roll(values, -start, axis=0))


def _applied_waivers(
    losses: tuple[AdapterLoss, ...], policy: LayoutImportPolicy
) -> tuple[AdapterWaiver, ...]:
    by_path = {loss.path: loss for loss in losses}
    unknown = tuple(path for path in policy.waived_loss_paths if path not in by_path)
    if unknown:
        _fail(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"Layout loss waiver paths do not name declared losses: {unknown!r}.",
        )
    generated = tuple(
        AdapterWaiver(
            by_path[path],
            f"Caller explicitly accepted the interpretation change at {path}.",
        )
        for path in policy.waived_loss_paths
    )
    combined = policy.waivers + generated
    if len({value.loss_id for value in combined}) != len(combined):
        raise ValueError("Each layout loss may be waived at most once.")
    return combined


def decode_layout_resource(
    resource: BoundedResource, policy: LayoutImportPolicy, /
) -> LayoutImportResult:
    """Decode one already bounded layout resource under an explicit policy."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, LayoutImportPolicy):
        raise TypeError("policy must be a LayoutImportPolicy.")
    if resource.manifest.limits != policy.limits:
        raise ValueError("The bounded resource and layout policy limits must match.")
    gdstk = _gdstk(resource)
    library = _parse_library(gdstk, resource, policy)
    user_unit, database_unit, scale = _source_units(library, policy, resource)
    try:
        decoder = _LayoutDecoder(gdstk, library, resource, policy, scale)
        model, top_cell, depth, nodes, losses = decoder.decode()
        accounted = account_bounded_resource(
            resource,
            depth=depth,
            nodes=nodes,
            attributes=decoder.attributes,
            losses=len(losses),
        )
    except ResourceReadError as error:
        _resource_failure(error)
    waivers = _applied_waivers(losses, policy)
    required = (
        AdapterRequirement("layout.finite-valid-planar-regions"),
        AdapterRequirement("layout.layer-datatype-identity"),
        AdapterRequirement("layout.explicit-spatial-contract"),
        AdapterRequirement("layout.expanded-occurrence-provenance"),
    )
    affected_capabilities = frozenset(
        capability_id for loss in losses for capability_id in loss.affected_capability_ids
    )
    optional = tuple(
        AdapterRequirement(semantic_id, required=False)
        for capability_id, semantic_id in _OPTIONAL_SEMANTICS
        if capability_id in affected_capabilities
    )
    requirements = required + optional
    capabilities = tuple(
        AdapterCapability(requirement.semantic_id) for requirement in required
    )
    negotiation = negotiate_adapter(
        requirements, capabilities, losses=losses, waivers=waivers
    )
    source_format = "GDSII" if policy.format is LayoutFormat.GDSII else "OASIS"
    source_id = canonical_fingerprint(
        {
            "kind": "layout-source",
            "format": policy.format.value,
            "sha256": accounted.manifest.content_sha256,
        }
    )
    report = AdapterReport(
        negotiation.status,
        source_format,
        "phydrax-planar-layout",
        source_id=source_id,
        target_id=model.model_id,
        stage="host-import",
        source_profile=AdapterFormatProfile(
            source_format,
            qualifiers={
                "user-unit-m": format(user_unit, ".17g"),
                "database-unit-m": format(database_unit, ".17g"),
                "top-cell": top_cell,
            },
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-planar-layout",
            qualifiers={
                "coordinate-contract": policy.coordinate_contract.spatial_id,
                "material-semantics": "none",
            },
        ),
        coordinate_mapping=(
            (
                f"source user unit {user_unit:.17g} m -> target unit {policy.coordinate_contract.length_unit.symbol}"
            ),
            "hierarchical affine occurrences -> explicit planar region occurrences",
        ),
        preserved_fields=(
            "finite polygon geometry",
            "layer and datatype",
            "hierarchy occurrence path",
            "source content digest",
        ),
        assumptions=(
            "layout coordinates are Cartesian x-y coordinates",
            "no layer is interpreted as a material or process step",
        ),
        losses=losses,
        requirements=requirements,
        capabilities=capabilities,
        waivers=waivers,
    )
    if not negotiation.valid or not report.valid:
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Layout input contains an unwaived semantic loss or stale waiver.",
            report=report,
            resource_manifest=accounted.manifest,
        )
    return LayoutImportResult(
        model,
        policy.format,
        top_cell,
        accounted.manifest.content_sha256,
        user_unit,
        database_unit,
        accounted.manifest,
        report,
    )


def decode_layout_bytes(
    data: bytes,
    policy: LayoutImportPolicy,
    /,
    *,
    source_path: str | None = None,
) -> LayoutImportResult:
    """Bound and decode exact in-memory GDSII/OASIS bytes."""

    if not isinstance(policy, LayoutImportPolicy):
        raise TypeError("policy must be a LayoutImportPolicy.")
    try:
        resource = bounded_resource_from_bytes(
            data, limits=policy.limits, source_path=source_path
        )
    except ResourceReadError as error:
        _resource_failure(error)
    return decode_layout_resource(resource, policy)


def read_layout(
    path: str | os.PathLike[str],
    policy: LayoutImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
) -> LayoutImportResult:
    """Descriptor-read and decode one bounded local GDSII/OASIS resource."""

    if not isinstance(policy, LayoutImportPolicy):
        raise TypeError("policy must be a LayoutImportPolicy.")
    try:
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=policy.limits
        )
    except ResourceReadError as error:
        _resource_failure(error)
    return decode_layout_resource(resource, policy)


__all__ = [
    "LayoutAdapterError",
    "LayoutFormat",
    "LayoutImportPolicy",
    "LayoutImportResult",
    "LayoutLayerKey",
    "LayoutModel",
    "LayoutRegion",
    "decode_layout_bytes",
    "decode_layout_resource",
    "read_layout",
]
