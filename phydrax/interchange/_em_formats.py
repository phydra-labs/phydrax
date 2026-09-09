#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._geospatial import GeospatialContract
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits


_COMPONENTS = {"ZXX": (0, 0), "ZXY": (0, 1), "ZYX": (1, 0), "ZYY": (1, 1)}


class MTImpedanceData(StrictModule, NonTrainableState):
    frequencies_Hz: Array
    impedance_ohm: Array
    standard_deviation_ohm: Array
    rotation_degrees: Array
    longitude_latitude_elevation: tuple[float, float, float] = eqx.field(static=True)
    coordinates: GeospatialContract
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    data_id: str = eqx.field(static=True)


def _dms(value: str) -> float:
    fields = re.split(r"[: ]+", value.strip())
    if len(fields) == 1:
        result = float(fields[0])
    elif len(fields) == 3:
        degree, minute, second = (float(item) for item in fields)
        sign = -1.0 if degree < 0 else 1.0
        result = degree + sign * (abs(minute) / 60.0 + abs(second) / 3600.0)
    else:
        raise ValueError(
            "EDI latitude/longitude must be decimal or degrees:minutes:seconds."
        )
    if not np.isfinite(result):
        raise ValueError("EDI coordinate must be finite.")
    return result


def _edi_blocks(
    text: str, limits: ResourceLimits, /
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    header: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    decoded_nodes = 0
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith(">"):
            name = line[1:].split()[0].upper()
            current = name
            if name in blocks:
                raise ValueError("EDI numeric block is duplicated.")
            blocks[name] = []
            continue
        if "=" in line and current in (None, "HEAD", "INFO", "DEFINEMEAS"):
            key, value = line.split("=", 1)
            key = key.strip().upper()
            if key in header:
                raise ValueError("EDI header key is duplicated.")
            header[key] = value.strip().strip('"')
        elif current is not None:
            tokens = line.replace("D", "E").split()
            blocks[current].extend(tokens)
            decoded_nodes += len(tokens)
        if len(header) > limits.max_attributes or decoded_nodes > limits.max_nodes:
            raise ValueError("EDI metadata or numeric values exceed resource limits.")
    arrays = {
        name: np.asarray([float(value) for value in values], dtype=float)
        for name, values in blocks.items()
        if values
    }
    return header, arrays


def _mt_result(
    *,
    frequencies,
    impedance,
    deviation,
    rotation,
    position,
    coordinates,
    resource,
    profile,
    assumptions=(),
):
    frequencies = np.asarray(frequencies, dtype=float)
    impedance = np.asarray(impedance, dtype=np.complex128)
    deviation = np.asarray(deviation, dtype=float)
    position = tuple(float(value) for value in position)
    rotation = np.asarray(rotation, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or impedance.shape != (frequencies.size, 2, 2)
    ):
        raise ValueError("MT frequency and impedance shapes are invalid.")
    if deviation.shape != impedance.shape or rotation.shape != frequencies.shape:
        raise ValueError(
            "MT deviation and rotation shapes must match impedance frequencies."
        )
    if (
        np.any(~np.isfinite(frequencies))
        or np.any(frequencies <= 0)
        or np.any(~np.isfinite(impedance))
        or np.any(~np.isfinite(deviation))
        or np.any(deviation <= 0)
        or np.any(~np.isfinite(rotation))
        or len(position) != 3
        or any(not np.isfinite(value) for value in position)
        or not -180 <= position[0] <= 180
        or not -90 <= position[1] <= 90
    ):
        raise ValueError(
            "MT frequencies, impedances, deviations, and rotations must be physical."
        )
    order = np.argsort(frequencies)
    frequencies, impedance, deviation, rotation = (
        frequencies[order],
        impedance[order],
        deviation[order],
        rotation[order],
    )
    identity = canonical_fingerprint(
        {
            "kind": "mt-impedance-data",
            "profile": profile,
            "resource": resource.manifest.content_sha256,
            "coordinates": coordinates.coordinate_id,
            "frequencies_Hz": frequencies,
            "impedance_ohm": impedance,
            "standard_deviation_ohm": deviation,
            "rotation_degrees": rotation,
            "position": position,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        profile,
        "MTImpedanceData",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(profile),
        coordinate_mapping=("longitude latitude and positive-up elevation retained",),
        preserved_fields=(
            "frequency",
            "complex impedance tensor",
            "standard deviations",
            "rotation",
            "station position",
            "exact source bytes",
        ),
        assumptions=assumptions,
        losses=(
            AdapterLoss(
                "unmodeled-source-metadata",
                "import",
                "dropped",
                "The normalized MT view retains impedance, uncertainty, rotation, "
                "and station position; other metadata remain only in the exact source bytes.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("magnetotelluric-impedance"),
            AdapterCapability("complex-uncertainty"),
        ),
    )
    return MTImpedanceData(
        jnp.asarray(frequencies),
        jnp.asarray(impedance),
        jnp.asarray(deviation),
        jnp.asarray(rotation),
        position,
        coordinates,
        resource,
        report,
        identity,
    )


def read_edi_impedance(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    coordinates: GeospatialContract,
    impedance_scale_ohm: float,
) -> MTImpedanceData:
    if coordinates.horizontal_kind != "geographic":
        raise ValueError("EDI station coordinates require a geographic contract.")
    scale = float(impedance_scale_ohm)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("EDI impedance scale to ohms must be positive and explicit.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    try:
        text = resource.data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("EDI must be ASCII.") from error
    header, blocks = _edi_blocks(text, limits)
    if "FREQ" not in blocks or not {"LAT", "LONG", "ELEV"}.issubset(header):
        raise ValueError("EDI profile lacks frequency or station coordinates.")
    frequency = blocks["FREQ"]
    count = frequency.size
    impedance = np.empty((count, 2, 2), dtype=np.complex128)
    deviation = np.empty((count, 2, 2), dtype=float)
    for component, (row, column) in _COMPONENTS.items():
        required = (component + "R", component + "I", component + ".VAR")
        if any(name not in blocks for name in required):
            raise ValueError(
                f"EDI profile lacks required {component} real/imaginary/variance blocks."
            )
        real, imaginary, variance = (blocks[name] for name in required)
        if real.size != count or imaginary.size != count or variance.size != count:
            raise ValueError("EDI impedance blocks must match frequency count.")
        if np.any(variance <= 0):
            raise ValueError("EDI impedance variances must be positive.")
        impedance[:, row, column] = scale * (real + 1j * imaginary)
        deviation[:, row, column] = scale * np.sqrt(variance)
    rotation = blocks.get("ZROT", np.zeros(count))
    if rotation.size == 1:
        rotation = np.full(count, rotation[0])
    position = (_dms(header["LONG"]), _dms(header["LAT"]), float(header["ELEV"]))
    return _mt_result(
        frequencies=frequency,
        impedance=impedance,
        deviation=deviation,
        rotation=rotation,
        position=position,
        coordinates=coordinates,
        resource=resource,
        profile="edi-impedance",
        assumptions=("Caller-supplied impedance scale converts stored values to ohms.",),
    )


def _local_name(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def read_emtf_xml_impedance(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    coordinates: GeospatialContract,
) -> MTImpedanceData:
    if coordinates.horizontal_kind != "geographic":
        raise ValueError("EMTF station coordinates require a geographic contract.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    root = ElementTree.fromstring(resource.data)
    elements = tuple(root.iter())
    if (
        len(elements) > limits.max_nodes
        or sum(len(element.attrib) for element in elements) > limits.max_attributes
    ):
        raise ValueError("EMTF XML exceeds decoded resource limits.")
    location: dict[str, float] = {}
    for element in elements:
        name = _local_name(element).lower()
        if name in ("longitude", "latitude", "elevation") and element.text:
            if name in location:
                raise ValueError("EMTF XML station coordinate is duplicated.")
            location[name] = float(element.text.strip())
    if set(location) != {"longitude", "latitude", "elevation"}:
        raise ValueError("EMTF XML lacks station longitude, latitude, or elevation.")
    periods: list[float] = []
    impedance_rows: list[np.ndarray] = []
    deviation_rows: list[np.ndarray] = []
    rotations: list[float] = []
    for period_element in elements:
        if (
            _local_name(period_element).lower() != "period"
            or "value" not in period_element.attrib
        ):
            continue
        period = float(period_element.attrib["value"])
        tensor = np.full((2, 2), np.nan + 1j * np.nan)
        error = np.full((2, 2), np.nan)
        rotation = float(period_element.attrib.get("rotation", 0.0))
        for element in period_element.iter():
            if _local_name(element).lower() not in ("z", "impedance"):
                continue
            output = element.attrib.get("output", "").lower()
            input_ = element.attrib.get("input", "").lower()
            key = {
                ("ex", "hx"): (0, 0),
                ("ex", "hy"): (0, 1),
                ("ey", "hx"): (1, 0),
                ("ey", "hy"): (1, 1),
            }.get((output, input_))
            if key is None:
                continue
            if np.isfinite(tensor[key]) or np.isfinite(error[key]):
                raise ValueError("EMTF XML impedance component is duplicated.")
            real = imaginary = sigma = None
            for child in element.iter():
                child_name = _local_name(child).lower()
                if child.text is None:
                    continue
                if child_name == "real":
                    real = float(child.text)
                elif child_name in ("imag", "imaginary"):
                    imaginary = float(child.text)
                elif child_name in ("error", "standarddeviation", "sigma"):
                    sigma = float(child.text)
            if real is not None and imaginary is not None and sigma is not None:
                tensor[key], error[key] = real + 1j * imaginary, sigma
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(error)):
            raise ValueError(
                "Every EMTF XML period must contain a complete impedance tensor."
            )
        periods.append(period)
        impedance_rows.append(tensor)
        deviation_rows.append(error)
        rotations.append(rotation)
    if not periods or np.any(np.asarray(periods) <= 0):
        raise ValueError(
            "EMTF XML contains no complete positive-period impedance tensors."
        )
    return _mt_result(
        frequencies=1.0 / np.asarray(periods),
        impedance=np.asarray(impedance_rows),
        deviation=np.asarray(deviation_rows),
        rotation=np.asarray(rotations),
        position=(location["longitude"], location["latitude"], location["elevation"]),
        coordinates=coordinates,
        resource=resource,
        profile="emtf-xml-impedance",
    )


__all__ = ["MTImpedanceData", "read_edi_impedance", "read_emtf_xml_impedance"]
