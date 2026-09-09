# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded, eager CF host interchange; no external execution state enters a model."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any

import numpy as np

from ..._array_archive import array_collection_digest
from ..._fingerprint import canonical_fingerprint
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ...interchange._resource import (
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from ...units import (
    conversion_factor,
    DEGREE,
    derived_unit,
    JOULE,
    KELVIN,
    KILOGRAM,
    KILOMETER,
    KILOPASCAL,
    METER,
    MILLIMETER,
    ONE,
    PASCAL,
    SECOND,
    UnitDefinition,
)
from ._quantities import GeophysicalQuantity
from ._time import GeophysicalTimeSpec, TemporalSupport
from ._vertical import HybridPressureCoordinate


DEFAULT_GEOPHYSICAL_DATA_LIMITS = ResourceLimits(
    max_bytes=256 * 1024 * 1024,
    max_depth=32,
    max_nodes=10000,
    max_attributes=100000,
    max_losses=10000,
)
# This is an adapter vocabulary, not a second physical-quantity/unit registry.
_CF_KINDS = {
    "air_temperature": "temperature",
    "surface_temperature": "temperature",
    "sea_surface_temperature": "temperature",
    "air_pressure": "pressure",
    "surface_air_pressure": "surface_pressure",
    "air_density": "mass_density",
    "eastward_wind": "velocity",
    "northward_wind": "velocity",
    "upward_air_velocity": "velocity",
    "geopotential": "geopotential",
    "geopotential_height": "height",
    "specific_humidity": "specific_humidity",
    "humidity_mixing_ratio": "vapor_mixing_ratio",
    "mass_fraction_of_cloud_liquid_water_in_air": "liquid_water_fraction",
    "mass_fraction_of_cloud_ice_in_air": "ice_water_fraction",
    "precipitation_flux": "water_mass_flux",
    "rainfall_flux": "water_mass_flux",
    "precipitation_amount": "precipitation_amount",
    "rainfall_amount": "precipitation_amount",
    "lwe_precipitation_rate": "water_volume_flux",
    "surface_downwelling_shortwave_flux_in_air": "radiative_flux",
    "surface_upwelling_shortwave_flux_in_air": "radiative_flux",
    "surface_downwelling_longwave_flux_in_air": "radiative_flux",
    "surface_upwelling_longwave_flux_in_air": "radiative_flux",
    "toa_outgoing_longwave_flux": "radiative_flux",
    "surface_upward_sensible_heat_flux": "heat_flux",
    "surface_upward_latent_heat_flux": "heat_flux",
    "atmosphere_mass_content_of_water_vapor": "column_mass",
}
_CF_EXPLICIT_REFERENCE = frozenset(
    {
        "eastward_wind",
        "northward_wind",
        "upward_air_velocity",
        "geopotential_height",
        "surface_downwelling_shortwave_flux_in_air",
        "surface_upwelling_shortwave_flux_in_air",
        "surface_downwelling_longwave_flux_in_air",
        "surface_upwelling_longwave_flux_in_air",
        "toa_outgoing_longwave_flux",
        "surface_upward_sensible_heat_flux",
        "surface_upward_latent_heat_flux",
    }
)
_UNSUPPORTED = frozenset(
    {
        "climatology",
        "compress",
        "sample_dimension",
        "instance_dimension",
        "mesh",
        "location",
        "geometry",
        "node_coordinates",
        "cell_measures",
        "ancillary_variables",
        "_Unsigned",
    }
)
_PACKING = frozenset(
    {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "valid_min",
        "valid_max",
        "valid_range",
    }
)
_REQUIRED = frozenset(
    {
        "standard_name",
        "units",
        "cell_methods",
        "bounds",
        "calendar",
        "formula_terms",
        "grid_mapping",
        "coordinates",
        "masks",
        "packing",
        "provenance",
    }
)


def _fail(
    message: str, status: AdapterStatus = AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
):
    raise AdapterError(status, message)


def _require(module: str):
    if importlib.util.find_spec(module) is None:
        _fail(
            f"Geophysical interchange requires optional dependency '{module}'.",
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
        )
    return importlib.import_module(module)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        # CF missing-value attributes may be NaN, unlike canonical JSON.
        return {
            "nonfinite": "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    _fail(
        f"Unsupported host metadata type {type(value).__name__}; explicit conversion is required."
    )


def _json(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _restore_attrs(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"nonfinite"}:
            return float(value["nonfinite"])
        return {key: _restore_attrs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_attrs(item) for item in value]
    return value


def _cf_unit(text: str) -> tuple[UnitDefinition, float]:
    """Only documented CF spellings; affine temperature is local to this boundary."""
    normalized = str(text).strip().replace("**", "^")
    plain = normalized.replace(" ", "").replace("^", "")
    simple = {
        "1": ONE,
        "K": KELVIN,
        "kelvin": KELVIN,
        "Pa": PASCAL,
        "kPa": KILOPASCAL,
        "km": KILOMETER,
        "mm": MILLIMETER,
        "m": METER,
        "kg": KILOGRAM,
        "s": SECOND,
        "seconds": SECOND,
        "degrees_east": DEGREE,
        "degree_east": DEGREE,
        "degrees_north": DEGREE,
        "degree_north": DEGREE,
        "degrees": DEGREE,
    }
    if normalized in simple:
        return simple[normalized], 0.0
    if normalized in {"degC", "degree_Celsius", "degrees_Celsius", "Celsius"}:
        return KELVIN, 273.15
    if normalized in {"hPa", "mbar", "millibar"}:
        return UnitDefinition(
            normalized, PASCAL.dimension, PASCAL.reference_system_id, 100
        ), 0.0
    components = {
        "ms-1": ((METER, 1), (SECOND, -1)),
        "m/s": ((METER, 1), (SECOND, -1)),
        "m2s-2": ((METER, 2), (SECOND, -2)),
        "m2/s2": ((METER, 2), (SECOND, -2)),
        "kgm-3": ((KILOGRAM, 1), (METER, -3)),
        "kg/m3": ((KILOGRAM, 1), (METER, -3)),
        "kgm-2": ((KILOGRAM, 1), (METER, -2)),
        "kg/m2": ((KILOGRAM, 1), (METER, -2)),
        "kgm-2s-1": ((KILOGRAM, 1), (METER, -2), (SECOND, -1)),
        "kg/m2/s": ((KILOGRAM, 1), (METER, -2), (SECOND, -1)),
        "Wm-2": ((JOULE, 1), (SECOND, -1), (METER, -2)),
        "W/m2": ((JOULE, 1), (SECOND, -1), (METER, -2)),
        "Jkg-1": ((JOULE, 1), (KILOGRAM, -1)),
        "J/kg": ((JOULE, 1), (KILOGRAM, -1)),
        "s-1": ((SECOND, -1),),
        "1/s": ((SECOND, -1),),
    }
    if plain in {"kgkg-1", "kg/kg"}:
        return ONE, 0.0
    if plain in components:
        return derived_unit(normalized, components[plain]), 0.0
    _fail(f"Unsupported required CF units {text!r}.")


def _terms(text: str) -> dict[str, str]:
    tokens = re.findall(r"([A-Za-z][\w]*):\s*([A-Za-z_][\w]*)", str(text))
    if (
        not tokens
        or len(dict(tokens)) != len(tokens)
        or re.sub(r"([A-Za-z][\w]*):\s*([A-Za-z_][\w]*)", "", str(text)).strip()
    ):
        _fail("Malformed CF formula_terms.", AdapterStatus.MALFORMED_SOURCE)
    return dict(tokens)


@dataclass(frozen=True, slots=True, init=False)
class GeophysicalData:
    """Immutable host initialization or sampled result, never a simulator checkpoint.

    ``descriptor`` is self-contained, JSON-safe scientific meaning; ``arrays`` owns
    values and explicit validity under canonical payload references. Bindings refer
    to existing native storage identities, not a duplicate state/layout owner.
    """

    descriptor_json: str
    arrays: Mapping[str, np.ndarray]
    data_id: str

    def __init__(self, descriptor: Mapping[str, Any], arrays: Mapping[str, Any]):
        text = _json(descriptor)
        desc = json.loads(text)
        if desc.get("kind") != "geophysical-data" or desc.get("purpose") not in {
            "initialization",
            "sampled-result",
        }:
            raise ValueError(
                "Geophysical data must declare initialization or sampled-result purpose."
            )
        immutable = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise ValueError(
                    "Geophysical payloads must be real numeric or Boolean arrays."
                )
            immutable[str(name)] = np.frombuffer(
                array.tobytes(), dtype=array.dtype
            ).reshape(array.shape)
        _validate_descriptor(desc, immutable)
        object.__setattr__(self, "descriptor_json", text)
        object.__setattr__(self, "arrays", MappingProxyType(immutable))
        object.__setattr__(
            self,
            "data_id",
            canonical_fingerprint(
                {
                    "descriptor": desc,
                    "payloads": array_collection_digest(immutable),
                }
            ),
        )

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(self.descriptor_json)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.descriptor["fields"])

    def values(self, name: str) -> np.ndarray:
        return self.arrays[self.descriptor["variables"][name]["payload"]]

    def valid(self, name: str) -> np.ndarray:
        return self.arrays[self.descriptor["variables"][name]["valid_payload"]]

    def require_bindings(self, bindings: Mapping[str, Any]) -> None:
        """Verify an exact native storage binding without claiming solver restart."""
        records = self.descriptor["fields"]
        if set(bindings) != set(records):
            raise ValueError("Binding inventory does not match the native product.")
        for name, binding in bindings.items():
            if binding.binding_id != records[name]["binding_id"]:
                raise ValueError(f"Native binding identity mismatch for {name!r}.")


def _validate_descriptor(
    desc: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    expected = {
        "kind",
        "purpose",
        "variables",
        "fields",
        "coordinates",
        "dimensions",
        "attributes",
        "time",
        "vertical",
        "provenance",
    }
    if set(desc) != expected or not desc["fields"]:
        raise ValueError("Incomplete or unexpected geophysical data descriptor.")
    refs = set()
    dimensions = desc["dimensions"]
    for name, record in desc["variables"].items():
        if set(record) != {"dims", "attrs", "source_attrs", "payload", "valid_payload"}:
            raise ValueError("Invalid geophysical variable descriptor.")
        payload, valid_ref = record["payload"], record["valid_payload"]
        if payload in refs or valid_ref in refs or payload == valid_ref:
            raise ValueError("Geophysical payload references must be unique.")
        refs.update((payload, valid_ref))
        if payload not in arrays or valid_ref not in arrays:
            raise ValueError("Missing geophysical payload reference.")
        shape = tuple(dimensions[dim] for dim in record["dims"])
        if (
            arrays[payload].shape != shape
            or arrays[valid_ref].shape != shape
            or arrays[valid_ref].dtype != np.dtype(bool)
        ):
            raise ValueError(f"Payload shape or mask mismatch for {name!r}.")
        if not np.all(np.isfinite(arrays[payload][arrays[valid_ref]])):
            raise ValueError("Valid geophysical values must be finite.")
    if refs != set(arrays) or not set(desc["coordinates"]).issubset(desc["variables"]):
        raise ValueError("Geophysical payload/coordinate inventory mismatch.")
    for name, field in desc["fields"].items():
        if name not in desc["variables"] or not field["binding_id"]:
            raise ValueError("Geophysical field reference is invalid.")
        quantity = GeophysicalQuantity.from_dict(field["quantity"])
        if quantity.quantity_id != field["quantity_id"]:
            raise ValueError("Geophysical quantity identity mismatch.")
        if canonical_fingerprint(field["binding"]) != field["binding_descriptor_id"]:
            raise ValueError("Geophysical binding descriptor checksum mismatch.")
        binding = field["binding"]
        if (
            binding["binding_id"] != field["binding_id"]
            or canonical_fingerprint(
                {key: value for key, value in binding.items() if key != "binding_id"}
            )
            != field["binding_id"]
        ):
            raise ValueError("Geophysical binding identity mismatch.")
        if binding["quantity"] != field["quantity"]:
            raise ValueError("Field quantity disagrees with its native binding.")
        if desc["variables"][name]["attrs"]["units"] != quantity.unit.symbol:
            raise ValueError("Field units disagree with the bound quantity.")
        TemporalSupport.from_dict(field["temporal"])
        if (
            _temporal_support(name, desc, arrays).support_id
            != field["temporal"]["support_id"]
        ):
            raise ValueError("Field temporal descriptor disagrees with its CF support.")
        if (
            binding["temporal"] is not None
            and TemporalSupport.from_dict(binding["temporal"]).support_id
            != field["temporal"]["support_id"]
        ):
            raise ValueError("Bound temporal support disagrees with the sampled field.")
        if binding["vertical"] is not None:
            vertical = HybridPressureCoordinate.from_dict(binding["vertical"])
            ids = {
                item["coordinate_id"]
                for coord, item in desc["vertical"].items()
                if coord in desc["variables"][name]["dims"]
                and item["kind"] == "hybrid-pressure"
            }
            if vertical.coordinate_id not in ids:
                raise ValueError(
                    "Bound hybrid coordinate disagrees with the sampled field."
                )
    for record in desc["time"].values():
        spec = GeophysicalTimeSpec(
            calendar=record["calendar"], epoch=record["epoch"], unit=record["unit"]
        )
        if spec.time_id != record["time_id"]:
            raise ValueError("Geophysical calendar/time identity mismatch.")
    for record in desc["vertical"].values():
        if record["kind"] == "hybrid-pressure":
            HybridPressureCoordinate.from_dict(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"kind", "formula_terms"}
                }
            )
        elif record["kind"] != "pressure":
            raise ValueError("Unknown geophysical vertical descriptor.")
    scientific = {**desc, "time": {}, "vertical": {}}
    _validate_cf_coordinates(
        scientific,
        arrays,
        {
            name: record["dtype"]
            for name, record in desc["vertical"].items()
            if record["kind"] == "hybrid-pressure"
        },
    )
    if scientific["time"] != desc["time"] or scientific["vertical"] != desc["vertical"]:
        raise ValueError("Time/vertical descriptors disagree with coordinate payloads.")


def _decoded(variable: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asanyarray(variable.values)
    if raw.dtype.kind not in "biuf":
        _fail(
            "CF boundary requires undecoded real numeric arrays; reopen with decode_cf=False."
        )
    attrs = variable.attrs
    valid = np.isfinite(raw) & ~np.ma.getmaskarray(raw)
    raw = np.asarray(np.ma.getdata(raw))
    for key in ("_FillValue", "missing_value"):
        for fill in np.asarray(attrs.get(key, [])).reshape(-1):
            valid &= raw != fill
    if "valid_range" in attrs:
        bounds = np.asarray(attrs["valid_range"])
        if bounds.shape != (2,) or bounds[0] > bounds[1]:
            _fail("Malformed CF valid_range.", AdapterStatus.MALFORMED_SOURCE)
        valid &= (raw >= bounds[0]) & (raw <= bounds[1])
    if "valid_min" in attrs:
        valid &= raw >= attrs["valid_min"]
    if "valid_max" in attrs:
        valid &= raw <= attrs["valid_max"]
    scale, offset = (
        float(attrs.get("scale_factor", 1.0)),
        float(attrs.get("add_offset", 0.0)),
    )
    if not np.isfinite(scale) or scale == 0 or not np.isfinite(offset):
        _fail(
            "CF packing scale/offset must be finite with nonzero scale.",
            AdapterStatus.MALFORMED_SOURCE,
        )
    decoded = raw.astype(np.float64) * scale + offset
    valid &= np.isfinite(decoded)
    return np.where(valid, decoded, np.nan), valid


def _time_descriptor(
    attrs: Mapping[str, Any],
) -> tuple[GeophysicalTimeSpec, dict[str, Any]]:
    match = re.fullmatch(
        r"(seconds?|s|minutes?|min|hours?|h|days?|d) since (.+)",
        str(attrs.get("units", "")),
    )
    if match is None:
        _fail("CF time requires supported numeric '<unit> since <epoch>' units.")
    unit = {
        "s": "s",
        "second": "s",
        "seconds": "s",
        "min": "min",
        "minute": "min",
        "minutes": "min",
        "h": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
    }[match[1]]
    spec = GeophysicalTimeSpec(
        calendar=attrs.get("calendar", "standard"), epoch=match[2], unit=unit
    )
    return spec, {
        "calendar": spec.calendar,
        "epoch": spec.epoch,
        "unit": spec.unit,
        "time_id": spec.time_id,
    }


def from_cf_dataset(
    dataset: Any,
    /,
    *,
    bindings: Mapping[str, Any],
    limits: ResourceLimits = DEFAULT_GEOPHYSICAL_DATA_LIMITS,
    materialize_lazy: bool = False,
    purpose: str = "initialization",
    required_semantics: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> tuple[GeophysicalData, AdapterReport]:
    """Import selected bound variables and their complete supported CF dependency closure.

    External formats are transformed initialization/samples, never exact restarts.
    Missing values stay masked; sums remain accumulations, never implicit rates.
    """
    xr = _require("xarray")
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset.")
    unknown = set(required_semantics) - _REQUIRED
    if unknown:
        _fail(f"Unknown required semantics: {sorted(unknown)}.")
    if not bindings or not set(bindings).issubset(dataset.variables):
        _fail(
            "Every imported field requires an explicit existing native binding.",
            AdapterStatus.INCONSISTENT_SOURCE,
        )
    if purpose not in {"initialization", "sampled-result"}:
        _fail(
            "CF imports are initialization or sampled-result products, not exact restarts."
        )
    conventions = (
        str(dataset.attrs.get("Conventions", "CF-1.11")).replace(",", " ").split()
    )
    if any(not (item.startswith("CF-") or item == "COARDS") for item in conventions):
        _fail("Dataset declares unsupported required conventions.")
    selected = set(bindings)
    pending = list(selected)
    while pending:
        name = pending.pop()
        var = dataset[name]
        if _UNSUPPORTED.intersection(var.attrs):
            _fail(
                f"Unsupported required semantics on {name!r}: {sorted(_UNSUPPORTED.intersection(var.attrs))}."
            )
        if (
            var.attrs.get("GRIB_stepType", "instant") != "instant"
            and "cell_methods" not in var.attrs
        ):
            _fail(
                "GRIB interval statistics require explicit CF time bounds and cell_methods."
            )
        if var.attrs.get("GRIB_uvRelativeToGrid", 0) and var.attrs.get(
            "standard_name"
        ) in {"eastward_wind", "northward_wind"}:
            _fail("Grid-relative GRIB winds require an explicit vector-frame conversion.")
        if (
            var.attrs.get("GRIB_typeOfLevel") in {"hybrid", "hybridLayer"}
            and "formula_terms" not in var.attrs
        ):
            _fail(
                "GRIB hybrid levels require decoded CF formula terms and interface bounds."
            )
        refs = set(var.dims) & set(dataset.variables)
        refs.update(str(var.attrs.get("coordinates", "")).split())
        for key in ("bounds", "grid_mapping"):
            if key in var.attrs:
                refs.add(str(var.attrs[key]))
        if "formula_terms" in var.attrs:
            refs.update(_terms(var.attrs["formula_terms"]).values())
        refs.update(
            coord
            for coord in dataset.coords
            if set(dataset[coord].dims).issubset(var.dims)
        )
        for ref in refs:
            if ref not in dataset.variables:
                _fail(
                    f"Missing CF dependency {ref!r} referenced by {name!r}.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
            if ref not in selected:
                selected.add(ref)
                pending.append(ref)
        if len(selected) > limits.max_nodes:
            raise ResourceReadError(
                "limit", "CF variable dependency count exceeds max_nodes."
            )
    attribute_count = len(dataset.attrs) + sum(
        len(dataset[name].attrs) for name in selected
    )
    if attribute_count > limits.max_attributes:
        raise ResourceReadError("limit", "CF metadata exceeds max_attributes.")
    metadata = {
        "attributes": dict(dataset.attrs),
        "variables": {name: dict(dataset[name].attrs) for name in selected},
    }
    _bound_metadata(metadata, limits)
    # Check shape/dtype metadata BEFORE triggering any lazy reads or decompression.
    eager_bytes = sum(
        int(dataset[name].size) * (max(8, dataset[name].dtype.itemsize) + 1)
        for name in selected
    )
    if eager_bytes > limits.max_bytes:
        raise ResourceReadError("limit", "Decoded CF values and masks exceed max_bytes.")
    if (
        any(dataset[name].chunks is not None for name in selected)
        and not materialize_lazy
    ):
        _fail("Lazy CF data requires explicit materialize_lazy=True and finite limits.")
    variables, arrays, losses = {}, {}, []
    for name in sorted(selected):
        var = dataset[name]
        values, valid = _decoded(var)
        attrs = {
            key: _json_value(value)
            for key, value in var.attrs.items()
            if key not in _PACKING
        }
        dims = tuple(var.dims)
        if name in bindings:
            binding = bindings[name]
            quantity = binding.quantity
            standard = attrs.get("standard_name")
            if standard not in _CF_KINDS:
                _fail(
                    f"Unsupported or missing required standard_name {standard!r} on {name!r}."
                )
            if _CF_KINDS[standard] != quantity.quantity_kind:
                _fail(
                    f"CF standard_name {standard!r} is incompatible with quantity kind {quantity.quantity_kind!r}.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
            reference = standard if standard in _CF_EXPLICIT_REFERENCE else "absolute"
            if (
                quantity.reference_configuration != reference
                or quantity.sign_convention != "positive"
            ):
                _fail(
                    f"{standard!r} requires reference_configuration={reference!r} with positive sign.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
            source_unit, offset = _cf_unit(attrs.get("units", ""))
            if offset and quantity.quantity_kind != "temperature":
                _fail("Affine Celsius units are only supported for absolute temperature.")
            factor = float(conversion_factor(source_unit, quantity.unit))
            values = (values + offset) * factor
            if quantity.axes:
                if set(quantity.axes) != set(dims) or len(quantity.axes) != len(dims):
                    _fail(
                        f"Bound axes for {name!r} must be an exact permutation of CF dimensions.",
                        AdapterStatus.INCONSISTENT_SOURCE,
                    )
                order = tuple(dims.index(axis) for axis in quantity.axes)
                values, valid, dims = (
                    values.transpose(order),
                    valid.transpose(order),
                    quantity.axes,
                )
            attrs["units"] = quantity.unit.symbol
            if offset or factor != 1:
                losses.append(
                    AdapterLoss(
                        name + ".units",
                        "import",
                        "transformed",
                        "CF values converted to bound native units; affine offset precedes scaling.",
                        changes_interpretation=False,
                    )
                )
        if _PACKING.intersection(var.attrs):
            losses.append(
                AdapterLoss(
                    name + ".packing",
                    "import",
                    "transformed",
                    "Decoded float64 values and Boolean mask; original packing retained as provenance.",
                    changes_interpretation=False,
                )
            )
        payload, mask = f"values/{name}", f"valid/{name}"
        variables[name] = {
            "dims": list(dims),
            "attrs": attrs,
            "source_attrs": _json_value(dict(var.attrs)),
            "payload": payload,
            "valid_payload": mask,
        }
        arrays[payload], arrays[mask] = values, valid
    converted_bounds = set()
    for name, binding in bindings.items():
        parent = variables[name]
        bounds_name = parent["attrs"].get("bounds")
        if bounds_name is not None:
            if bounds_name in converted_bounds or bounds_name in bindings:
                _fail(
                    "A bound coordinate requires an unambiguous, separately owned bounds variable."
                )
            converted_bounds.add(bounds_name)
            bound = variables[bounds_name]
            source_unit, offset = _cf_unit(
                bound["source_attrs"].get("units", parent["source_attrs"]["units"])
            )
            factor = float(conversion_factor(source_unit, binding.quantity.unit))
            arrays[bound["payload"]] = (arrays[bound["payload"]] + offset) * factor
            order = tuple(bound["dims"].index(dim) for dim in parent["dims"]) + (
                len(bound["dims"]) - 1,
            )
            arrays[bound["payload"]] = arrays[bound["payload"]].transpose(order)
            arrays[bound["valid_payload"]] = arrays[bound["valid_payload"]].transpose(
                order
            )
            bound["dims"] = parent["dims"] + [bound["dims"][-1]]
            bound["attrs"]["units"] = binding.quantity.unit.symbol
    desc = {
        "kind": "geophysical-data",
        "purpose": purpose,
        "variables": variables,
        "fields": {
            name: {"quantity": binding.quantity.to_dict()}
            for name, binding in bindings.items()
        },
        "coordinates": sorted(set(dataset.coords) & selected),
        "dimensions": {
            dim: int(dataset.sizes[dim])
            for name in selected
            for dim in dataset[name].dims
        },
        "attributes": _json_value(dict(dataset.attrs)),
        "time": {},
        "vertical": {},
        "provenance": _json_value(dict(provenance or {})),
    }
    vertical_dtypes = {}
    for name, binding in bindings.items():
        if binding.vertical_id is not None:
            dtype = binding.to_dict()["vertical"]["dtype"]
            for coord in variables[name]["dims"]:
                if (
                    coord in variables
                    and variables[coord]["attrs"].get("standard_name")
                    == "atmosphere_hybrid_sigma_pressure_coordinate"
                ):
                    if coord in vertical_dtypes and vertical_dtypes[coord] != dtype:
                        _fail(
                            "A shared CF vertical coordinate cannot bind conflicting native precisions."
                        )
                    vertical_dtypes[coord] = dtype
    _validate_cf_coordinates(desc, arrays, vertical_dtypes)
    for name, binding in bindings.items():
        var = variables[name]
        temporal = _temporal_support(name, desc, arrays)
        if (
            binding.temporal is not None
            and binding.temporal.support_id != temporal.support_id
        ):
            _fail(
                f"Temporal support mismatch for {name!r}.",
                AdapterStatus.INCONSISTENT_SOURCE,
            )
        if binding.vertical_id is not None:
            ids = {
                record["coordinate_id"]
                for coord, record in desc["vertical"].items()
                if coord in var["dims"] and record["kind"] == "hybrid-pressure"
            }
            if binding.vertical_id not in ids:
                _fail(
                    f"Hybrid vertical binding mismatch for {name!r}.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
        binding_desc = binding.to_dict()
        desc["fields"][name] = {
            "binding_id": binding.binding_id,
            "binding": binding_desc,
            "binding_descriptor_id": canonical_fingerprint(binding_desc),
            "quantity": binding.quantity.to_dict(),
            "quantity_id": binding.quantity.quantity_id,
            "temporal": {
                "kind": temporal.kind,
                "position": temporal.position,
                "bounds": temporal.bounds,
                "support_id": temporal.support_id,
            },
        }
    if len(losses) > limits.max_losses:
        raise ResourceReadError("limit", "CF conversion loss count exceeds max_losses.")
    data = GeophysicalData(desc, arrays)
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "CF-dataset",
        "phydrax-geophysical-data",
        source_id=canonical_fingerprint(
            desc["provenance"]
            or {
                "variables": {
                    name: record["source_attrs"] for name, record in variables.items()
                },
                "payloads": array_collection_digest(arrays),
            }
        ),
        target_id=data.data_id,
        preserved_fields=tuple(sorted(selected)),
        assumptions=(
            "Selected fields only; no remapping, rotation, calendar/rate conversion, or simulator restart.",
        ),
        losses=losses,
    )
    return data, report


def _validate_cf_coordinates(
    desc: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    vertical_dtypes: Mapping[str, str] | None = None,
) -> None:
    variables = desc["variables"]
    for name, var in variables.items():
        attrs, dims = var["attrs"], var["dims"]
        values = arrays[var["payload"]]
        if "grid_mapping_name" in attrs:
            if attrs["grid_mapping_name"] not in {
                "latitude_longitude",
                "rotated_latitude_longitude",
            }:
                _fail(
                    f"Unsupported required grid mapping {attrs['grid_mapping_name']!r}."
                )
            if dims:
                _fail("CF grid_mapping variables must be scalar.")
            if attrs["grid_mapping_name"] == "rotated_latitude_longitude" and not {
                "grid_north_pole_latitude",
                "grid_north_pole_longitude",
            }.issubset(attrs):
                _fail("Rotated-pole grid mapping requires its complete pole definition.")
        if "bounds" in attrs:
            bound = variables[attrs["bounds"]]
            bv = arrays[bound["payload"]]
            if (
                bv.ndim != values.ndim + 1
                or bound["dims"][:-1] != dims
                or bv.shape[:-1] != values.shape
                or bv.shape[-1] < 2
                or not np.all(arrays[bound["valid_payload"]])
            ):
                _fail(
                    f"Invalid coordinate bounds for {name!r}.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
        is_time = (
            attrs.get("standard_name") == "time"
            or attrs.get("axis") == "T"
            or (name in desc["coordinates"] and " since " in str(attrs.get("units", "")))
        )
        is_pressure = (
            attrs.get("standard_name") == "air_pressure" and name in desc["coordinates"]
        )
        if name in desc["coordinates"] and len(dims) == 1 and dims[0] == name:
            if not np.all(arrays[var["valid_payload"]]) or (
                values.size > 1
                and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))
            ):
                _fail(
                    f"Coordinate {name!r} must be finite, unmasked, and strictly monotone.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
        if attrs.get("standard_name") in {"latitude", "longitude"}:
            unit, offset = _cf_unit(attrs.get("units", ""))
            if offset or unit.dimension != DEGREE.dimension:
                _fail("Latitude/longitude require angular units.")
            if attrs["standard_name"] == "latitude" and np.any(
                np.abs(values * float(conversion_factor(unit, DEGREE))) > 90
            ):
                _fail(
                    "Latitude lies outside [-90, 90] degrees.",
                    AdapterStatus.INCONSISTENT_SOURCE,
                )
        if is_time:
            if (
                len(dims) > 1
                or not np.all(arrays[var["valid_payload"]])
                or np.any(np.diff(values.reshape(-1)) <= 0)
            ):
                _fail(
                    "Time coordinate must be scalar or one-dimensional, unmasked, and increasing."
                )
            spec, record = _time_descriptor(attrs)
            spec.decode(values.reshape(-1))  # Validate instants in the declared calendar.
            attrs["calendar"] = spec.calendar
            desc["time"][name] = record
            if "bounds" in attrs:
                bound_attrs = variables[attrs["bounds"]]["attrs"]
                bound_spec, _ = _time_descriptor({**attrs, **bound_attrs})
                if bound_spec.time_id != spec.time_id:
                    _fail(
                        "Time coordinate and bounds must use the same calendar, epoch, and unit."
                    )
                spec.decode(arrays[variables[attrs["bounds"]]["payload"]].reshape(-1))
        if is_pressure:
            if name in desc["fields"]:
                unit, offset = (
                    GeophysicalQuantity.from_dict(desc["fields"][name]["quantity"]).unit,
                    0.0,
                )
            else:
                unit, offset = _cf_unit(attrs.get("units", ""))
            pressure = values * float(conversion_factor(unit, PASCAL))
            if offset or np.any(pressure <= 0) or attrs.get("positive", "down") != "down":
                _fail(
                    "Pressure coordinates require positive pressures and positive='down'."
                )
            desc["vertical"][name] = {
                "kind": "pressure",
                "unit": unit.to_dict(),
                "payload": var["payload"],
                "positive": "down",
            }
        if attrs.get("standard_name") == "atmosphere_hybrid_sigma_pressure_coordinate":
            _hybrid(
                name,
                desc,
                arrays,
                dtype=None if vertical_dtypes is None else vertical_dtypes.get(name),
            )
        elif "formula_terms" in attrs:
            _fail(f"Unsupported required parametric vertical coordinate on {name!r}.")


def _hybrid(
    name: str,
    desc: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    dtype: str | None = None,
) -> None:
    variables = desc["variables"]
    terms = _terms(variables[name]["attrs"].get("formula_terms", ""))
    pressure_a = "ap" in terms
    if set(terms) != ({"ap", "b", "ps"} if pressure_a else {"a", "b", "ps", "p0"}):
        _fail("Hybrid formula must be p=ap+b*ps or p=a*p0+b*ps.")

    def term_variable(
        variable_name: str, target: UnitDefinition, inherited_unit: str = ""
    ) -> np.ndarray:
        record = variables[variable_name]
        if not np.all(arrays[record["valid_payload"]]):
            _fail("Hybrid formula coefficients and surface pressure cannot be masked.")
        unit, offset = _cf_unit(
            record["attrs"].get("units", inherited_unit or ("1" if target == ONE else ""))
        )
        if offset:
            _fail("Hybrid formula terms cannot use affine units.")
        return arrays[record["payload"]] * float(conversion_factor(unit, target))

    def term(key: str, target: UnitDefinition) -> np.ndarray:
        return term_variable(terms[key], target)

    a, b = (
        term("ap" if pressure_a else "a", PASCAL if pressure_a else ONE),
        term("b", ONE),
    )
    if (
        a.ndim != 1
        or b.shape != a.shape
        or variables[terms["b"]]["dims"]
        != variables[terms["ap" if pressure_a else "a"]]["dims"]
    ):
        _fail("Hybrid coefficients must share one vertical dimension.")
    p0 = 100000.0 if pressure_a else term("p0", PASCAL)
    if np.ndim(p0) != 0:
        _fail("Hybrid p0 must be scalar.")
    ps = term("ps", PASCAL)
    if name in variables[terms["ps"]]["dims"] or np.any(ps <= 0):
        _fail("Hybrid surface pressure must be positive and omit the vertical dimension.")
    # CF coefficients describe sample levels. Native hybrid coordinates describe
    # interfaces; only explicit, contiguous coefficient bounds establish them.
    if "bounds" not in variables[name]["attrs"]:
        _fail(
            "Hybrid layers need explicit coordinate/coefficient bounds; midpoints are not interfaces."
        )
    centers_a, centers_b = a, b
    interfaces = []
    for key, target in (
        ("ap" if pressure_a else "a", PASCAL if pressure_a else ONE),
        ("b", ONE),
    ):
        bounds_name = variables[terms[key]]["attrs"].get("bounds")
        if bounds_name is None:
            _fail("Hybrid layer coefficients require explicit interface bounds.")
        bounds = term_variable(
            bounds_name, target, variables[terms[key]]["attrs"].get("units", "")
        )
        if bounds.shape != (a.size, 2) or not np.array_equal(
            bounds[:-1, 1], bounds[1:, 0]
        ):
            _fail(
                "Hybrid coefficient bounds must define contiguous interfaces.",
                AdapterStatus.INCONSISTENT_SOURCE,
            )
        interfaces.append(np.concatenate((bounds[:, 0], bounds[-1:, 1])))
    a, b = interfaces
    coordinate = HybridPressureCoordinate(
        a.tolist(),
        b.tolist(),
        reference_pressure=float(p0),
        a_is_pressure=pressure_a,
        dtype=dtype,
    )
    # Host validation avoids device allocations and proves ordering for every column.
    pressure = (a if pressure_a else a * float(p0)) + ps[..., None] * b
    if (
        not np.all(np.isfinite(pressure))
        or np.any(pressure < 0)
        or np.any(np.diff(pressure, axis=-1) <= 0)
    ):
        _fail(
            "Hybrid pressure must increase strictly top-to-bottom for every source column.",
            AdapterStatus.INCONSISTENT_SOURCE,
        )
    centers = (centers_a if pressure_a else centers_a * float(p0)) + ps[
        ..., None
    ] * centers_b
    if np.any(centers < pressure[..., :-1]) or np.any(centers > pressure[..., 1:]):
        _fail(
            "Hybrid sample pressures must lie within their explicit interfaces.",
            AdapterStatus.INCONSISTENT_SOURCE,
        )
    native = coordinate.to_dict()
    native_a, native_b = np.asarray(native["a"]), np.asarray(native["b"])
    native_pressure = (native_a if pressure_a else native_a * float(p0)) + ps[
        ..., None
    ] * native_b
    if np.any(native_pressure < 0) or np.any(np.diff(native_pressure, axis=-1) <= 0):
        _fail(
            "Requested native coefficient precision invalidates a source pressure column.",
            AdapterStatus.INCONSISTENT_SOURCE,
        )
    desc["vertical"][name] = {"kind": "hybrid-pressure", **native, "formula_terms": terms}


def _temporal_support(
    name: str, desc: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> TemporalSupport:
    variable = desc["variables"][name]
    method = str(variable["attrs"].get("cell_methods", "")).strip()
    times = [
        key
        for key in desc["time"]
        if desc["variables"][key]["attrs"].get("standard_name")
        != "forecast_reference_time"
        and set(desc["variables"][key]["dims"]).issubset(variable["dims"])
        and (
            key in desc["coordinates"]
            or key in str(variable["attrs"].get("coordinates", "")).split()
            or key in variable["dims"]
        )
    ]
    if len(times) > 1:
        _fail("A field cannot have multiple independent time coordinates.")
    if not method:
        return TemporalSupport()
    if not times:
        _fail("CF cell_methods requires an identified time coordinate.")
    match = re.fullmatch(
        r"(?:time|" + re.escape(times[0]) + r"): (point|mean|sum)", method
    )
    if match is None:
        _fail(
            f"Unsupported required cell_methods {method!r}; only a single time point/mean/sum is supported."
        )
    if match[1] == "point":
        return TemporalSupport()
    coordinate = desc["variables"][times[0]]
    bounds_name = coordinate["attrs"].get("bounds")
    if bounds_name is None:
        _fail("Time means and accumulations require explicit time bounds.")
    bounds = arrays[desc["variables"][bounds_name]["payload"]]
    points = arrays[coordinate["payload"]].reshape(-1)
    if not coordinate["dims"] and bounds.shape == (2,):
        bounds = bounds.reshape(1, 2)
    if (
        bounds.shape != (points.size, 2)
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(points < bounds[:, 0])
        or np.any(points > bounds[:, 1])
    ):
        _fail(
            "Time bounds must bracket each sample with positive interval width.",
            AdapterStatus.INCONSISTENT_SOURCE,
        )
    position = "point"
    for label, representative in (
        ("start", bounds[:, 0]),
        ("midpoint", bounds.mean(axis=1)),
        ("end", bounds[:, 1]),
    ):
        if np.array_equal(points, representative):
            position = label
            break
    return TemporalSupport(
        kind="mean" if match[1] == "mean" else "accumulation",
        bounds=tuple(map(tuple, bounds.tolist())),
        position=position,
    )


def to_cf_dataset(data: GeophysicalData, /) -> tuple[Any, AdapterReport]:
    """Export scientific reference units, never arbitrary native unit symbols."""
    xr = _require("xarray")
    desc = data.descriptor
    quantities = {}
    for name, field in desc["fields"].items():
        quantity = GeophysicalQuantity.from_dict(field["quantity"])
        quantities[name] = quantity
        bounds = desc["variables"][name]["attrs"].get("bounds")
        if bounds is not None:
            quantities[bounds] = quantity
    variables, losses = {}, []
    for name, record in desc["variables"].items():
        values = np.where(
            data.arrays[record["valid_payload"]], data.arrays[record["payload"]], np.nan
        )
        attrs = _restore_attrs(record["attrs"])
        if name in quantities:
            quantity = quantities[name]
            values = quantity.to_si(values)
            attrs["units"] = quantity.reference_unit.symbol
            if (
                quantity.si_factor != 1
                or quantity.unit.symbol != quantity.reference_unit.symbol
            ):
                losses.append(
                    AdapterLoss(
                        name + ".units",
                        "export",
                        "transformed",
                        "Converted to CF reference units; original native units remain in the archive.",
                        changes_interpretation=False,
                    )
                )
        variables[name] = xr.Variable(tuple(record["dims"]), values, attrs=attrs)
    coords = {name: variables.pop(name) for name in desc["coordinates"]}
    dataset = xr.Dataset(
        variables, coords=coords, attrs=_restore_attrs(desc["attributes"])
    )
    dataset.attrs["Conventions"] = "CF-1.11"
    dataset.attrs["phydrax_data_id"] = data.data_id
    dataset.attrs["phydrax_provenance"] = _json(desc["provenance"])
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "phydrax-geophysical-data",
        "CF-dataset",
        source_id=data.data_id,
        target_id=data.data_id,
        preserved_fields=tuple(desc["variables"]),
        assumptions=(
            "Decoded values, masks and metadata only; no original packed bytes or solver restart.",
        ),
        losses=losses,
    )
    return dataset, report


def _bound_metadata(value: Any, limits: ResourceLimits) -> None:
    stack, nodes, bytes_ = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > limits.max_depth or nodes > limits.max_attributes + limits.max_nodes:
            raise ResourceReadError("limit", "CF metadata exceeds structural limits.")
        if isinstance(item, Mapping):
            stack.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif isinstance(item, (tuple, list)):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, np.ndarray):
            bytes_ += item.nbytes
        elif isinstance(item, str):
            bytes_ += len(item.encode("utf-8"))
        else:
            bytes_ += 16
        if bytes_ > limits.max_bytes:
            raise ResourceReadError("limit", "CF metadata exceeds max_bytes.")


def _grib_to_cf(dataset: Any, bindings: Mapping[str, Any]) -> Any:
    """Resolve one forecast reference/lead pair, without guessing interval support."""
    if not bindings or not set(bindings).issubset(dataset.data_vars):
        _fail("GRIB fields require an explicit binding to an available decoded variable.")
    temporal_names = {}
    for standard in ("forecast_reference_time", "forecast_period", "time"):
        names = [
            name
            for name, var in dataset.variables.items()
            if var.attrs.get("standard_name") == standard
        ]
        if len(names) > 1 or (not names and standard != "time"):
            _fail("GRIB requires an unambiguous forecast reference time and lead time.")
        if names:
            name = names[0]
            if dataset[name].dims or dataset[name].size != 1:
                _fail("Select scalar GRIB forecast coordinates before import.")
            temporal_names[name] = standard
    source_names = {standard: name for name, standard in temporal_names.items()}
    reference_var = dataset[source_names["forecast_reference_time"]]
    period_var = dataset[source_names["forecast_period"]]
    spec, _ = _time_descriptor(reference_var.attrs)
    step_unit = str(period_var.attrs.get("units", ""))
    seconds = {
        "s": 1.0,
        "seconds": 1.0,
        "h": 3600.0,
        "hours": 3600.0,
        "d": 86400.0,
        "days": 86400.0,
    }
    if step_unit not in seconds:
        _fail(f"Unsupported GRIB lead-time unit {step_unit!r}.")
    step_values, step_valid = _decoded(period_var)
    reference_values, reference_valid = _decoded(reference_var)
    if not np.all(step_valid) or not np.all(reference_valid):
        _fail("GRIB forecast reference time and lead time must be finite and unmasked.")
    step = float(step_values)
    reference = float(reference_values)
    endpoint = reference + step * seconds[step_unit] / spec.seconds_per_unit
    if "time" in source_names:
        valid_var = dataset[source_names["time"]]
        valid_spec, _ = _time_descriptor(valid_var.attrs)
        valid_values, valid_mask = _decoded(valid_var)
        if (
            not np.all(valid_mask)
            or valid_spec.calendar != spec.calendar
            or valid_spec.decode(valid_values.reshape(-1)) != spec.decode([endpoint])
        ):
            _fail(
                "GRIB valid time disagrees with its forecast reference and lead time.",
                AdapterStatus.INCONSISTENT_SOURCE,
            )
    records = [dataset[name].attrs for name in bindings]
    step_types = {attrs.get("GRIB_stepType", "instant") for attrs in records}
    if len(step_types) != 1 or not step_types.issubset({"instant", "accum", "avg"}):
        _fail("GRIB fields must share supported instantaneous/accumulation/mean support.")
    step_type = step_types.pop()
    normalized = dataset.rename(temporal_names).set_coords(list(temporal_names.values()))
    # xarray renames variables and dimensions, not CF references stored in attrs.
    # Translate every reference; unknown names remain for dependency validation.
    for var in normalized.variables.values():
        attrs = dict(var.attrs)
        if "coordinates" in attrs:
            attrs["coordinates"] = " ".join(
                temporal_names.get(ref, ref) for ref in str(attrs["coordinates"]).split()
            )
        for key in ("bounds", "grid_mapping"):
            if key in attrs:
                attrs[key] = temporal_names.get(str(attrs[key]), attrs[key])
        if "formula_terms" in attrs:
            attrs["formula_terms"] = " ".join(
                f"{term}: {temporal_names.get(ref, ref)}"
                for term, ref in _terms(attrs["formula_terms"]).items()
            )
        var.attrs = attrs
    time_attrs = normalized["time"].attrs if "time" in source_names else {}
    normalized = normalized.assign_coords(
        time=(
            (),
            endpoint,
            {
                **{
                    key: value for key, value in time_attrs.items() if key not in _PACKING
                },
                "standard_name": "time",
                "units": f"{spec.unit} since {spec.epoch}",
                "calendar": spec.calendar,
            },
        )
    ).expand_dims("time")
    if step_type != "instant":
        supports = []
        for attrs in records:
            if not {"GRIB_startStep", "GRIB_endStep", "GRIB_stepUnits"}.issubset(attrs):
                _fail(
                    "GRIB interval support requires decoded startStep/endStep/stepUnits."
                )
            unit_seconds = {
                0: 60.0,
                1: 3600.0,
                2: 86400.0,
                10: 10800.0,
                11: 21600.0,
                12: 43200.0,
                13: 1.0,
            }.get(int(attrs["GRIB_stepUnits"]))
            if unit_seconds is None:
                _fail("Unsupported GRIB interval time unit.")
            supports.append(
                (
                    reference
                    + float(attrs["GRIB_startStep"])
                    * unit_seconds
                    / spec.seconds_per_unit,
                    reference
                    + float(attrs["GRIB_endStep"]) * unit_seconds / spec.seconds_per_unit,
                )
            )
        if len(set(supports)) != 1 or supports[0][1] != endpoint:
            _fail("GRIB interval bounds disagree across fields or with valid time.")
        normalized["time_bounds"] = (("time", "bounds"), np.asarray(supports[:1]))
        normalized["time"].attrs["bounds"] = "time_bounds"
        for name in bindings:
            normalized[name].attrs = {
                **normalized[name].attrs,
                "cell_methods": "time: sum" if step_type == "accum" else "time: mean",
            }
    return normalized


def read_cf(
    path: str | Path,
    /,
    *,
    bindings: Mapping[str, Any],
    format: str = "netcdf",
    trusted_root: str | Path | None = None,
    engine: str | None = None,
    limits: ResourceLimits = DEFAULT_GEOPHYSICAL_DATA_LIMITS,
    **conversion: Any,
) -> tuple[GeophysicalData, AdapterReport]:
    """Snapshot bounded local NetCDF, Zarr, or GRIB resources before decoding.

    NetCDF defaults to the scipy NetCDF3 engine; select h5netcdf for NetCDF4.
    GRIB uses real cfgrib/ecCodes, with sidecar index writes disabled.
    """
    xr = _require("xarray")
    source = Path(path)
    root = source.parent if trusted_root is None else Path(trusted_root)
    relative = (
        source.name
        if trusted_root is None
        else str(source.relative_to(root) if source.is_absolute() else source)
    )
    if format not in {"netcdf", "zarr", "grib"}:
        raise ValueError("format must be netcdf, zarr, or grib.")
    manifests = []
    with TemporaryDirectory(prefix="phydrax-cf-") as temporary:
        snapshot = Path(temporary) / ("store" if format == "zarr" else "source")
        if format == "zarr":
            _require("zarr")
            source_dir = root / relative
            if not source_dir.is_dir() or source_dir.is_symlink():
                raise ResourceReadError(
                    "policy", "Zarr source must be a local non-symlink directory."
                )
            total = 0
            for directory, subdirs, files in os.walk(source_dir, followlinks=False):
                depth = len(Path(directory).relative_to(source_dir).parts)
                if depth > limits.max_depth or any(
                    (Path(directory) / item).is_symlink() for item in subdirs
                ):
                    raise ResourceReadError(
                        "policy",
                        "Zarr directory links or excessive depth are prohibited.",
                    )
                for filename in sorted(files):
                    member = Path(directory) / filename
                    resource = read_bounded_resource(
                        member.relative_to(root), trusted_root=root, limits=limits
                    )
                    total += len(resource.data)
                    if total > limits.max_bytes or len(manifests) >= limits.max_nodes:
                        raise ResourceReadError(
                            "limit", "Zarr store exceeds aggregate resource limits."
                        )
                    destination = snapshot / member.relative_to(source_dir)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(resource.data)
                    manifests.append(asdict(resource.manifest))
            dataset = xr.open_zarr(
                snapshot, decode_cf=False, chunks=None, consolidated=None
            )
        else:
            resource = read_bounded_resource(relative, trusted_root=root, limits=limits)
            manifests.append(asdict(resource.manifest))
            snapshot.write_bytes(resource.data)
            if format == "grib":
                _require("eccodes")
                _require("cfgrib")
                dataset = xr.open_dataset(
                    snapshot,
                    engine="cfgrib",
                    decode_cf=False,
                    chunks=None,
                    backend_kwargs={
                        "indexpath": "",
                        "read_keys": [
                            "stepType",
                            "startStep",
                            "endStep",
                            "stepUnits",
                            "uvRelativeToGrid",
                        ],
                    },
                )
            else:
                selected_engine = "scipy" if engine is None else engine
                if selected_engine not in {"scipy", "h5netcdf", "netcdf4"}:
                    raise ValueError("NetCDF engine must be scipy, h5netcdf, or netcdf4.")
                _require("netCDF4" if selected_engine == "netcdf4" else selected_engine)
                dataset = xr.open_dataset(
                    snapshot, engine=selected_engine, decode_cf=False, chunks=None
                )
        with dataset:
            if format == "grib":
                normalized = _grib_to_cf(dataset, bindings)
            else:
                normalized = dataset
            return from_cf_dataset(
                normalized,
                bindings=bindings,
                limits=limits,
                provenance={"format": format, "resources": manifests},
                **conversion,
            )


def write_cf(
    data: GeophysicalData,
    path: str | Path,
    /,
    *,
    format: str = "netcdf",
    engine: str = "scipy",
) -> Path:
    """Write actual NetCDF or Zarr via optional host dependencies; GRIB is decode-only."""
    dataset, _ = to_cf_dataset(data)
    destination = Path(path)
    if format == "netcdf":
        if engine not in {"scipy", "h5netcdf", "netcdf4"}:
            raise ValueError("NetCDF engine must be scipy, h5netcdf, or netcdf4.")
        _require("netCDF4" if engine == "netcdf4" else engine)
        dataset.to_netcdf(destination, engine=engine)
    elif format == "zarr":
        _require("zarr")
        dataset.to_zarr(destination, mode="w")
    else:
        raise ValueError("CF export supports netcdf or zarr; GRIB is decode-only.")
    return destination


__all__ = [
    "DEFAULT_GEOPHYSICAL_DATA_LIMITS",
    "GeophysicalData",
    "from_cf_dataset",
    "to_cf_dataset",
    "read_cf",
    "write_cf",
]
