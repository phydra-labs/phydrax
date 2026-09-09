#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit host-side geospatial qualification; no CRS inference or reprojection."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import equinox as eqx
import numpy as np
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import ANGLE, conversion_factor, DEGREE, LENGTH, UnitDefinition
from ._inspection import _readonly_array
from ._report import AdapterReport, AdapterStatus
from ._resource import bounded_resource_from_bytes, ResourceLimits, ResourceManifest


def _label(value: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _optional_label(value: str | None, owner: str) -> str | None:
    return None if value is None else _label(value, owner)


def _known(value: str | None) -> bool:
    return value is not None and value.casefold() not in (
        "unknown",
        "unspecified",
        "undefined",
    )


def _manifests(resources: Sequence[ResourceManifest]) -> tuple[ResourceManifest, ...]:
    result = tuple(resources)
    if not all(isinstance(item, ResourceManifest) for item in result):
        raise TypeError("resources must contain ResourceManifest values.")
    return result


def _metadata_json(metadata: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> str:
    items = list(metadata.items()) if isinstance(metadata, Mapping) else list(metadata)
    if any(not isinstance(key, str) for key, _ in items):
        raise TypeError("Metadata keys must be strings.")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("Metadata keys must be unique.")
    return canonical_json(dict(items))


class GeospatialTransform(StrictModule, NonTrainableState):
    """Provenance of an explicitly performed external coordinate transformation.

    This record does not execute or certify the named operation. Endpoint IDs are
    ``GeospatialContract.coordinate_id`` values, which exclude provenance itself.
    Parameters are independent immutable JSON; resource manifests identify exact
    caller-supplied transformation inputs (for example a local geoid model).
    """

    operation: str = eqx.field(static=True)
    source_coordinate_id: str = eqx.field(static=True)
    target_coordinate_id: str = eqx.field(static=True)
    parameters_json: str = eqx.field(static=True)
    resources: tuple[ResourceManifest, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        operation: str,
        source_coordinate_id: str,
        target_coordinate_id: str,
        /,
        *,
        parameters: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        resources: Sequence[ResourceManifest] = (),
    ):
        self.operation = _label(operation, "Transformation operation")
        self.source_coordinate_id = _label(source_coordinate_id, "Source coordinate ID")
        self.target_coordinate_id = _label(target_coordinate_id, "Target coordinate ID")
        self.parameters_json = _metadata_json(parameters)
        self.resources = _manifests(resources)
        self.transform_id = canonical_fingerprint(
            {
                "kind": "geospatial-transform",
                "operation": self.operation,
                "source_coordinate_id": self.source_coordinate_id,
                "target_coordinate_id": self.target_coordinate_id,
                "parameters": json.loads(self.parameters_json),
                "resources": [item.manifest_id for item in self.resources],
            }
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self.parameters_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "source_coordinate_id": self.source_coordinate_id,
            "target_coordinate_id": self.target_coordinate_id,
            "parameters": self.parameters,
            "resources": [asdict(item) for item in self.resources],
            "transform_id": self.transform_id,
        }


class GeospatialContract(StrictModule, NonTrainableState):
    """Immutable qualification composed with, not added to, native spatial identity.

    Unknown labels and JSON metadata can be retained for interchange. They never
    imply permission to use geographic degrees as Cartesian lengths or to compose
    different datums. ``horizontal_axes`` gives the ordered axis directions:
    ``('east', 'north')`` for Cartesian coordinates, ``('longitude', 'latitude')``
    for geographic grids. ``epoch_required=False`` explicitly declares a static
    coordinate frame; ``None`` leaves that fact unknown. Epochs are decimal years.

    Vertical kinds qualified here are ``local_height``, ``ellipsoidal_height``,
    ``orthometric_height``, and ``depth``. A solver's third Cartesian axis is
    positive up; depth/down coordinates need an explicit external transformation.
    ``longitude_domain`` is an unwrapped full-turn interval in horizontal units;
    ``longitude_seam`` is ``none`` for a regional nonwrapping grid or ``periodic``
    for a full-turn grid. No seam is silently unwrapped, joined, or duplicated.
    """

    spatial: SpatialCoordinateContract = eqx.field(static=True)
    horizontal_crs: str | None = eqx.field(static=True)
    horizontal_kind: str = eqx.field(static=True)
    horizontal_axes: tuple[str, str] = eqx.field(static=True)
    horizontal_units: tuple[UnitDefinition | None, UnitDefinition | None] = eqx.field(
        static=True
    )
    horizontal_datum: str | None = eqx.field(static=True)
    coordinate_epoch: float | None = eqx.field(static=True)
    epoch_required: bool | None = eqx.field(static=True)
    vertical_kind: str = eqx.field(static=True)
    vertical_positive: str = eqx.field(static=True)
    vertical_datum: str | None = eqx.field(static=True)
    vertical_unit: UnitDefinition | None = eqx.field(static=True)
    registration: str = eqx.field(static=True)
    longitude_seam: str = eqx.field(static=True)
    longitude_domain: tuple[float, float] | None = eqx.field(static=True)
    mask_semantics: str = eqx.field(static=True)
    metadata_json: str = eqx.field(static=True)
    transformations: tuple[GeospatialTransform, ...] = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    geospatial_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial: SpatialCoordinateContract,
        /,
        *,
        horizontal_crs: str | None = None,
        horizontal_kind: str = "unknown",
        horizontal_axes: tuple[str, str] = ("unknown", "unknown"),
        horizontal_units: tuple[UnitDefinition | None, UnitDefinition | None] = (
            None,
            None,
        ),
        horizontal_datum: str | None = None,
        coordinate_epoch: float | None = None,
        epoch_required: bool | None = None,
        vertical_kind: str = "unknown",
        vertical_positive: str = "unknown",
        vertical_datum: str | None = None,
        vertical_unit: UnitDefinition | None = None,
        registration: str = "unknown",
        longitude_seam: str = "unknown",
        longitude_domain: tuple[float, float] | None = None,
        mask_semantics: str = "unknown",
        metadata: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        transformations: Sequence[GeospatialTransform] = (),
    ):
        if not isinstance(spatial, SpatialCoordinateContract):
            raise TypeError("spatial must be SpatialCoordinateContract.")
        axes = tuple(_label(item, "Horizontal axis") for item in horizontal_axes)
        units = tuple(horizontal_units)
        if len(axes) != 2 or len(units) != 2:
            raise ValueError("Exactly two horizontal axes and units are required.")
        if any(
            item is not None and not isinstance(item, UnitDefinition) for item in units
        ):
            raise TypeError("Horizontal units must be UnitDefinition or None.")
        if vertical_unit is not None and not isinstance(vertical_unit, UnitDefinition):
            raise TypeError("vertical_unit must be UnitDefinition or None.")
        if epoch_required is not None and not isinstance(epoch_required, bool):
            raise TypeError("epoch_required must be bool or None.")
        epoch = None if coordinate_epoch is None else float(coordinate_epoch)
        if epoch is not None and not np.isfinite(epoch):
            raise ValueError("Coordinate epoch must be finite.")
        domain = None
        if longitude_domain is not None:
            domain = tuple(float(value) for value in longitude_domain)
            if (
                len(domain) != 2
                or not np.all(np.isfinite(domain))
                or domain[0] >= domain[1]
            ):
                raise ValueError("Longitude domain must be a finite increasing interval.")
        transforms = tuple(transformations)
        if not all(isinstance(item, GeospatialTransform) for item in transforms):
            raise TypeError("transformations must contain GeospatialTransform records.")
        self.spatial = spatial
        self.horizontal_crs = _optional_label(horizontal_crs, "Horizontal CRS")
        self.horizontal_kind = _label(horizontal_kind, "Horizontal kind")
        self.horizontal_axes = axes
        self.horizontal_units = units
        self.horizontal_datum = _optional_label(horizontal_datum, "Horizontal datum")
        self.coordinate_epoch = epoch
        self.epoch_required = epoch_required
        self.vertical_kind = _label(vertical_kind, "Vertical kind")
        self.vertical_positive = _label(vertical_positive, "Vertical sign")
        self.vertical_datum = _optional_label(vertical_datum, "Vertical datum")
        self.vertical_unit = vertical_unit
        self.registration = _label(registration, "Registration")
        self.longitude_seam = _label(longitude_seam, "Longitude seam")
        self.longitude_domain = domain
        self.mask_semantics = _label(mask_semantics, "Mask semantics")
        self.metadata_json = _metadata_json(metadata)
        self.transformations = transforms
        self.coordinate_id = canonical_fingerprint(self._coordinate_payload())
        for before, after in itertools.pairwise(transforms):
            if before.target_coordinate_id != after.source_coordinate_id:
                raise ValueError(
                    "Transformation provenance must form one connected chain."
                )
        if transforms and transforms[-1].target_coordinate_id != self.coordinate_id:
            raise ValueError("Transformation provenance does not end at this contract.")
        self.geospatial_id = canonical_fingerprint(
            {
                "kind": "geospatial-contract",
                "coordinate_id": self.coordinate_id,
                "registration": self.registration,
                "longitude_seam": self.longitude_seam,
                "longitude_domain": self.longitude_domain,
                "mask_semantics": self.mask_semantics,
                "metadata": self.metadata,
                "transformations": [item.transform_id for item in transforms],
            }
        )

    @classmethod
    def local_cartesian(
        cls,
        spatial: SpatialCoordinateContract,
        /,
        *,
        vertical_datum: str,
        registration: str = "unknown",
        mask_semantics: str = "unknown",
    ) -> GeospatialContract:
        """Declare a static local east/north/up frame; no datum is inferred."""
        return cls(
            spatial,
            horizontal_crs=f"local:{spatial.reference_frame}",
            horizontal_kind="local_cartesian",
            horizontal_axes=("east", "north"),
            horizontal_units=(spatial.length_unit, spatial.length_unit),
            horizontal_datum=f"local:{spatial.reference_frame}",
            epoch_required=False,
            vertical_kind="local_height",
            vertical_positive="up",
            vertical_datum=vertical_datum,
            vertical_unit=spatial.length_unit,
            registration=registration,
            longitude_seam="none",
            mask_semantics=mask_semantics,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """An independent JSON copy; mutating it cannot alter the contract."""
        return json.loads(self.metadata_json)

    def to_dict(self) -> dict[str, Any]:
        """Serialize geospatial metadata without changing native spatial fields."""
        payload = self._coordinate_payload()
        payload.pop("kind")
        payload.pop("spatial_id")
        payload.update(
            spatial=self.spatial.to_dict(),
            horizontal_axes=list(self.horizontal_axes),
            horizontal_units=[
                None if unit is None else unit.to_dict() for unit in self.horizontal_units
            ],
            vertical_unit=None
            if self.vertical_unit is None
            else self.vertical_unit.to_dict(),
            registration=self.registration,
            longitude_seam=self.longitude_seam,
            longitude_domain=None
            if self.longitude_domain is None
            else list(self.longitude_domain),
            mask_semantics=self.mask_semantics,
            metadata=self.metadata,
            transformations=[item.to_dict() for item in self.transformations],
            coordinate_id=self.coordinate_id,
            geospatial_id=self.geospatial_id,
        )
        return payload

    def _coordinate_payload(self) -> dict[str, Any]:
        return {
            "kind": "geospatial-coordinates",
            "spatial_id": self.spatial.spatial_id,
            "horizontal_crs": self.horizontal_crs,
            "horizontal_kind": self.horizontal_kind,
            "horizontal_axes": self.horizontal_axes,
            "horizontal_units": [
                None if item is None else item.unit_id for item in self.horizontal_units
            ],
            "horizontal_datum": self.horizontal_datum,
            "coordinate_epoch": self.coordinate_epoch,
            "epoch_required": self.epoch_required,
            "vertical_kind": self.vertical_kind,
            "vertical_positive": self.vertical_positive,
            "vertical_datum": self.vertical_datum,
            "vertical_unit": None
            if self.vertical_unit is None
            else self.vertical_unit.unit_id,
        }

    def _require_horizontal(self, *, geographic: bool = False) -> None:
        if self.spatial.length_coordinate_kind != "physical":
            raise ValueError("Geospatial numerical coordinates must be physical.")
        if not _known(self.horizontal_crs) or not _known(self.horizontal_datum):
            raise ValueError(
                "Numerical coordinates require explicit horizontal CRS and datum."
            )
        if self.epoch_required is None or (
            self.epoch_required and self.coordinate_epoch is None
        ):
            raise ValueError(
                "Coordinate epoch semantics are ambiguous or the required epoch is missing."
            )
        if geographic:
            if self.horizontal_kind != "geographic" or self.horizontal_axes != (
                "longitude",
                "latitude",
            ):
                raise ValueError(
                    "Geographic grids require ordered longitude/latitude axes."
                )
            if any(
                unit is None or unit.dimension != ANGLE for unit in self.horizontal_units
            ):
                raise ValueError(
                    "Geographic horizontal axes require explicit angular units."
                )
            if self.horizontal_units[0] != self.horizontal_units[1]:
                raise ValueError("Geographic grid axes must use the same angular unit.")
        else:
            if self.horizontal_kind not in ("local_cartesian", "projected"):
                raise ValueError(
                    "Cartesian numerical composition rejects geographic or unknown CRS kinds."
                )
            if self.spatial.coordinate_system != "cartesian":
                raise ValueError(
                    "Native numerical coordinates must use the Cartesian spatial contract."
                )
            if self.horizontal_axes != ("east", "north"):
                raise ValueError(
                    "Cartesian numerical axes must be explicitly ordered east/north."
                )
            if any(unit != self.spatial.length_unit for unit in self.horizontal_units):
                raise ValueError(
                    "Horizontal units must exactly match the native spatial length unit."
                )
            if self.longitude_seam != "none" or self.longitude_domain is not None:
                raise ValueError(
                    "Cartesian coordinates cannot carry an unresolved longitude seam."
                )

    def _require_vertical(self, *, positive_up: bool) -> None:
        if self.vertical_kind not in (
            "local_height",
            "ellipsoidal_height",
            "orthometric_height",
            "depth",
        ):
            raise ValueError("Vertical coordinate kind is unknown or unsupported.")
        if self.vertical_positive not in ("up", "down") or not _known(
            self.vertical_datum
        ):
            raise ValueError("Vertical coordinates require an explicit sign and datum.")
        if self.vertical_unit is None or self.vertical_unit.dimension != LENGTH:
            raise ValueError("Vertical coordinates require an explicit length unit.")
        if positive_up and (
            self.vertical_positive != "up"
            or self.vertical_kind == "depth"
            or self.vertical_unit != self.spatial.length_unit
        ):
            raise ValueError(
                "Native Cartesian z requires positive-up height in the spatial length unit."
            )

    def require_cartesian(
        self,
        spatial: SpatialCoordinateContract | None = None,
        /,
        *,
        dimensions: int = 3,
    ) -> SpatialCoordinateContract:
        """Fail closed before a 2D/3D Cartesian solver consumes these coordinates."""
        if dimensions not in (2, 3) or isinstance(dimensions, bool):
            raise ValueError(
                "Cartesian qualification supports exactly two or three dimensions."
            )
        self._require_horizontal()
        if spatial is not None:
            if not isinstance(spatial, SpatialCoordinateContract):
                raise TypeError("spatial must be SpatialCoordinateContract or None.")
            if spatial.spatial_id != self.spatial.spatial_id:
                raise ValueError(
                    "Native spatial unit, frame, or coordinate identity differs."
                )
        if dimensions == 3:
            self._require_vertical(positive_up=True)
        return self.spatial

    def require_compatible(
        self,
        other: GeospatialContract,
        /,
        *,
        dimensions: int = 3,
        require_grid: bool = False,
    ) -> None:
        """Require exact qualified Cartesian semantics, never transform implicitly."""
        if not isinstance(other, GeospatialContract):
            raise TypeError("other must be GeospatialContract.")
        self.require_cartesian(dimensions=dimensions)
        other.require_cartesian(self.spatial, dimensions=dimensions)
        left, right = self._coordinate_payload(), other._coordinate_payload()
        if dimensions == 2:
            for key in (
                "vertical_kind",
                "vertical_positive",
                "vertical_datum",
                "vertical_unit",
            ):
                left.pop(key)
                right.pop(key)
        if left != right or self.metadata_json != other.metadata_json:
            raise ValueError(
                "Geospatial coordinate semantics differ; an explicit transformation is required."
            )
        if require_grid:
            self.require_grid()
            other.require_grid()
            if (
                self.registration,
                self.longitude_seam,
                self.longitude_domain,
                self.mask_semantics,
            ) != (
                other.registration,
                other.longitude_seam,
                other.longitude_domain,
                other.mask_semantics,
            ):
                raise ValueError("Grid registration, seam, or mask semantics differ.")

    def require_grid(self) -> None:
        """Qualify a regular horizontal grid independently of Cartesian solvers."""
        geographic = self.horizontal_kind == "geographic"
        self._require_horizontal(geographic=geographic)
        if self.registration not in ("pixel", "gridline"):
            raise ValueError("Grid registration must be explicitly pixel or gridline.")
        if self.mask_semantics not in ("none", "valid_true"):
            raise ValueError("Grid mask semantics must be none or valid_true.")
        if geographic:
            horizontal_unit = self.horizontal_units[0]
            longitude_domain = self.longitude_domain
            if (
                self.longitude_seam not in ("none", "periodic")
                or longitude_domain is None
                or horizontal_unit is None
            ):
                raise ValueError(
                    "Geographic grids require explicit longitude domain, angular unit, "
                    "and seam semantics."
                )
            period = 360.0 / float(conversion_factor(horizontal_unit, DEGREE))
            if not np.isclose(
                longitude_domain[1] - longitude_domain[0],
                period,
                rtol=1e-12,
                atol=0,
            ):
                raise ValueError(
                    "Longitude domain must span exactly one full angular turn."
                )


class QualifiedGeospatialGrid(StrictModule, NonTrainableState):
    """Regular, immutable native host grid, never an xarray/GMT object in a PyTree.

    Values and Boolean validity have shape ``(len(y), len(x))``. Coordinate vectors
    are the exact sample locations: pixel centers or gridline nodes. Increasing
    and decreasing vectors are both preserved; no transpose, interpolation, datum
    conversion, or seam repair is performed. ``manifest`` identifies the canonical
    qualification descriptor (including array hashes), not an invented source
    file; ``resources`` records any actual caller-supplied input resources.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    contract: GeospatialContract = eqx.field(static=True)
    value_unit: UnitDefinition = eqx.field(static=True)
    name: str = eqx.field(static=True)
    value_role: str = eqx.field(static=True)
    region: tuple[float, float, float, float] = eqx.field(static=True)
    spacing: tuple[float, float] = eqx.field(static=True)
    orientation: tuple[str, str] = eqx.field(static=True)
    resources: tuple[ResourceManifest, ...] = eqx.field(static=True)
    manifest: ResourceManifest = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        values: ArrayLike,
        contract: GeospatialContract,
        /,
        *,
        value_unit: UnitDefinition,
        name: str = "z",
        valid: ArrayLike | None = None,
        value_role: str = "scalar",
        resources: Sequence[ResourceManifest] = (),
    ):
        if not isinstance(contract, GeospatialContract):
            raise TypeError("contract must be GeospatialContract.")
        contract.require_grid()
        if not isinstance(value_unit, UnitDefinition):
            raise TypeError("value_unit must be UnitDefinition.")
        if value_role not in ("scalar", "vertical"):
            raise ValueError("value_role must be scalar or vertical.")
        if value_role == "vertical":
            contract._require_vertical(positive_up=False)
            if value_unit != contract.vertical_unit:
                raise ValueError(
                    "Vertical grid values must use the declared vertical unit."
                )
        x_, dx = _regular_axis(x, "x")
        y_, dy = _regular_axis(y, "y")
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or raw.shape != (y_.size, x_.size):
            raise ValueError(
                "Grid values must be a real matrix in explicit (y, x) order."
            )
        if valid is None:
            mask = np.ones(raw.shape, dtype=bool)
        else:
            mask = np.asarray(valid)
            if mask.dtype.kind != "b" or mask.shape != raw.shape:
                raise ValueError(
                    "Grid validity must be a Boolean matrix matching values."
                )
        if contract.mask_semantics == "none" and not np.all(mask):
            raise ValueError("A no-mask contract cannot contain invalid grid samples.")
        if not np.all(np.isfinite(raw[mask])):
            raise ValueError(
                "Valid grid samples must be finite; missing data needs an explicit mask."
            )
        margin = 0.5 if contract.registration == "pixel" else 0.0
        region = (
            float(np.min(x_) - margin * abs(dx)),
            float(np.max(x_) + margin * abs(dx)),
            float(np.min(y_) - margin * abs(dy)),
            float(np.max(y_) + margin * abs(dy)),
        )
        if contract.horizontal_kind == "geographic":
            _qualify_longitude_grid(contract, region, x_, raw, mask)
        self.x = _readonly_array(x_)
        self.y = _readonly_array(y_)
        self.values = _readonly_array(raw)
        self.valid = _readonly_array(mask)
        self.contract = contract
        self.value_unit = value_unit
        self.name = _label(name, "Grid name")
        if self.name in ("x", "y", "valid"):
            raise ValueError(
                "Grid name cannot collide with coordinate names x, y, or valid."
            )
        self.value_role = value_role
        self.region = region
        self.spacing = (abs(dx), abs(dy))
        self.orientation = (
            "increasing" if dx > 0 else "decreasing",
            "increasing" if dy > 0 else "decreasing",
        )
        self.resources = _manifests(resources)
        descriptor = {
            "kind": "qualified-geospatial-grid",
            "contract": contract.geospatial_id,
            "name": self.name,
            "value_unit": value_unit.unit_id,
            "value_role": value_role,
            "arrays": array_tree_fingerprint((self.x, self.y, self.values, self.valid)),
            "region": region,
            "resources": [item.manifest_id for item in self.resources],
        }
        self.grid_id = canonical_fingerprint(descriptor)
        payload = canonical_json(descriptor).encode("utf-8")
        self.manifest = bounded_resource_from_bytes(
            payload,
            limits=ResourceLimits(len(payload), 16, 128, 128, 0),
        ).manifest
        self.report = AdapterReport(
            AdapterStatus.LOSSLESS,
            "native-geospatial-arrays",
            "qualified-geospatial-grid",
            source_id=self.manifest.manifest_id,
            target_id=self.grid_id,
            coordinate_mapping=(
                "values[j,i] is sampled at (x[i],y[j]); input orientations retained",
            ),
            preserved_fields=(
                "coordinates",
                "registration",
                "vertical_reference",
                "longitude_seam",
                "mask",
                "metadata",
                "resources",
            ),
            assumptions=(),
        )


def _regular_axis(values: ArrayLike, name: str) -> tuple[np.ndarray, float]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"Grid {name} must be a real vector with at least two samples.")
    axis = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"Grid {name} coordinates must be finite.")
    if not np.array_equal(raw.astype(np.longdouble), axis.astype(np.longdouble)):
        raise ValueError(
            f"Grid {name} coordinates cannot be represented exactly as float64."
        )
    delta = np.diff(axis)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(
            f"Grid {name} must be strictly monotone; wrapped seams need an explicit transformation."
        )
    step = float((axis[-1] - axis[0]) / (axis.size - 1))
    # Bound cancellation tolerance by relative spacing: large origins must never
    # make materially nonuniform coordinates appear to be a regular lattice.
    tolerance = min(
        8 * np.finfo(np.float64).eps * max(float(np.max(np.abs(axis))), abs(step)),
        abs(step) * 1e-10,
    )
    if np.any(np.abs(delta - step) > tolerance):
        raise ValueError(
            f"Grid {name} is nonuniform; this boundary never resamples coordinates."
        )
    return axis, step


def _qualify_longitude_grid(
    contract: GeospatialContract,
    region: tuple[float, float, float, float],
    x: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> None:
    horizontal_unit = contract.horizontal_units[0]
    domain = contract.longitude_domain
    if horizontal_unit is None or domain is None:
        raise ValueError("Geographic grid requires an angular unit and longitude domain.")
    factor = float(conversion_factor(horizontal_unit, DEGREE))
    tolerance = 1e-10 / factor
    if region[0] < domain[0] - tolerance or region[1] > domain[1] + tolerance:
        raise ValueError(
            "Grid crosses its declared longitude domain; no implicit seam wrapping is permitted."
        )
    if region[2] < -90 / factor - tolerance or region[3] > 90 / factor + tolerance:
        raise ValueError("Geographic grid latitude support lies outside the poles.")
    if contract.longitude_seam == "periodic":
        if not np.allclose(region[:2], domain, rtol=0, atol=tolerance):
            raise ValueError(
                "Periodic longitude grids must cover the entire declared domain."
            )
        if contract.registration == "gridline":
            if not np.array_equal(valid[:, 0], valid[:, -1]):
                raise ValueError("Periodic gridline seam endpoint masks disagree.")
            if not np.array_equal(values[valid[:, 0], 0], values[valid[:, -1], -1]):
                raise ValueError("Periodic gridline seam endpoint values disagree.")
    elif np.isclose(abs(x[-1] - x[0]), 360 / factor, rtol=0, atol=tolerance):
        raise ValueError(
            "Duplicated full-turn gridline endpoints require a periodic seam declaration."
        )


__all__ = ["GeospatialContract", "GeospatialTransform", "QualifiedGeospatialGrid"]
