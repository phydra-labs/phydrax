#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


TouchstoneFormat = Literal["RI", "MA", "DB"]


class TouchstonePolicy(StrictModule):
    """Strict native subset policy; unsupported grammar is rejected."""

    allow_version_1: bool = eqx.field(static=True)
    allow_version_2: bool = eqx.field(static=True)
    require_monotone_frequency: bool = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        allow_version_1: bool = True,
        allow_version_2: bool = True,
        require_monotone_frequency: bool = True,
    ):
        self.allow_version_1 = bool(allow_version_1)
        self.allow_version_2 = bool(allow_version_2)
        self.require_monotone_frequency = bool(require_monotone_frequency)


class TouchstoneData(StrictModule):
    """Host-parsed float64/complex128 full scattering data and provenance."""

    frequencies_hz: Array
    scattering: Array
    reference_impedance: Array
    port_names: tuple[str, ...] = eqx.field(static=True)
    data_format: TouchstoneFormat = eqx.field(static=True)
    frequency_unit: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    file_convention: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies_hz: ArrayLike,
        scattering: ArrayLike,
        reference_impedance: ArrayLike,
        /,
        *,
        port_names: Sequence[str] = (),
        data_format: TouchstoneFormat = "RI",
        frequency_unit: str = "HZ",
        version: str = "1.0",
        source_hash: str = "",
        file_convention: str = "touchstone-column-major",
    ):
        frequencies = jnp.asarray(frequencies_hz, dtype=jnp.float64)
        matrix = jnp.asarray(scattering, dtype=jnp.complex128)
        if (
            frequencies.ndim != 1
            or matrix.ndim != 3
            or matrix.shape[0] != frequencies.size
            or matrix.shape[1] != matrix.shape[2]
        ):
            raise ValueError(
                "Touchstone data requires frequencies (F,) and scattering (F,N,N)."
            )
        if frequencies.size == 0 or not bool(jnp.all(jnp.isfinite(frequencies))):
            raise ValueError("Touchstone frequencies must be non-empty and finite.")
        reference = jnp.asarray(reference_impedance, dtype=jnp.float64)
        count = int(matrix.shape[-1])
        if reference.ndim == 0:
            reference = jnp.broadcast_to(reference, (count,))
        if reference.shape != (count,) or not bool(
            jnp.all(jnp.isfinite(reference) & (reference > 0.0))
        ):
            raise ValueError(
                "Touchstone references must be real positive scalar or (N,)."
            )
        names = tuple(str(name) for name in port_names) or tuple(
            f"port{index + 1}" for index in range(count)
        )
        if len(names) != count or any(not name for name in names):
            raise ValueError("port_names must contain one non-empty name per port.")
        if data_format not in ("RI", "MA", "DB"):
            raise ValueError("data_format must be RI, MA, or DB.")
        unit = str(frequency_unit).upper()
        if unit not in ("HZ", "KHZ", "MHZ", "GHZ"):
            raise ValueError("Unsupported Touchstone frequency unit.")
        self.frequencies_hz = frequencies
        self.scattering = matrix
        self.reference_impedance = reference
        self.port_names = names
        self.data_format = data_format
        self.frequency_unit = unit
        self.version = str(version)
        self.source_hash = str(source_hash)
        self.file_convention = str(file_convention)

    @property
    def port_count(self) -> int:
        return int(self.scattering.shape[-1])


_UNIT_SCALE = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def _port_count_from_suffix(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if (
        len(suffix) >= 4
        and suffix.startswith(".s")
        and suffix.endswith("p")
        and suffix[2:-1].isdigit()
    ):
        return int(suffix[2:-1])
    return None


def _option(tokens: list[str]) -> tuple[str, TouchstoneFormat, float]:
    upper = [token.upper() for token in tokens]
    if len(upper) != 5 or upper[1] != "S" or upper[3] != "R":
        raise ValueError("Supported option grammar is '# <unit> S <RI|MA|DB> R <real>'.")
    unit, data_format = upper[0], upper[2]
    if unit not in _UNIT_SCALE or data_format not in ("RI", "MA", "DB"):
        raise ValueError("Unsupported Touchstone unit, parameter, or data format.")
    reference = float(tokens[4])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("Touchstone reference resistance must be finite and positive.")
    return unit, data_format, reference


def _complex(first: float, second: float, data_format: TouchstoneFormat) -> complex:
    if data_format == "RI":
        return complex(first, second)
    angle = np.deg2rad(second)
    magnitude = first if data_format == "MA" else 10.0 ** (first / 20.0)
    return complex(magnitude * np.cos(angle), magnitude * np.sin(angle))


def read_touchstone(
    path: str | Path,
    policy: TouchstonePolicy | None = None,
    /,
) -> TouchstoneData:
    """Parse the strict native Touchstone 1.0/2.0 full-S subset."""
    selected = TouchstonePolicy() if policy is None else policy
    if not isinstance(selected, TouchstonePolicy):
        raise TypeError("policy must be TouchstonePolicy or None.")
    source = Path(path)
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    version = "1.0"
    port_count = _port_count_from_suffix(source)
    option_value: tuple[str, TouchstoneFormat, float] | None = None
    network_tokens: list[str] = []
    references: list[float] | None = None
    port_names: dict[int, str] = {}
    in_network_data = False
    matrix_format = "FULL"
    number_of_frequencies: int | None = None
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        content, _, comment = raw_line.partition("!")
        stripped_comment = comment.strip()
        if (
            stripped_comment.lower().startswith("port[")
            and "]" in stripped_comment
            and "=" in stripped_comment
        ):
            left, name = stripped_comment.split("=", 1)
            number = int(left[left.index("[") + 1 : left.index("]")])
            port_names[number - 1] = name.strip()
        line = content.strip()
        if not line:
            index += 1
            continue
        if line.startswith("#"):
            option_value = _option(line[1:].split())
            index += 1
            continue
        if line.startswith("["):
            close = line.index("]")
            keyword = line[1:close].strip().upper()
            value = line[close + 1 :].strip()
            if keyword == "VERSION":
                version = value
            elif keyword == "NUMBER OF PORTS":
                port_count = int(value)
            elif keyword == "NUMBER OF FREQUENCIES":
                number_of_frequencies = int(value)
            elif keyword == "MATRIX FORMAT":
                matrix_format = value.upper()
            elif keyword == "REFERENCE":
                references = [float(token) for token in value.split()]
            elif keyword == "NETWORK DATA":
                in_network_data = True
            elif keyword == "END":
                in_network_data = False
            elif keyword in ("TWO-PORT DATA ORDER",):
                if value.upper() != "21_12":
                    raise ValueError(
                        "Only canonical two-port 21_12 Touchstone ordering is supported."
                    )
            elif keyword in ("MIXED-MODE ORDER", "NOISE DATA", "BEGIN INFORMATION"):
                raise ValueError(f"Unsupported Touchstone section [{keyword}].")
            else:
                raise ValueError(f"Unsupported Touchstone keyword [{keyword}].")
            index += 1
            continue
        if version.startswith("2") and not in_network_data:
            raise ValueError("Numeric Touchstone 2.0 data must follow [Network Data].")
        network_tokens.extend(line.split())
        index += 1
    if option_value is None or port_count is None or port_count <= 0:
        raise ValueError("Touchstone option line and positive port count are required.")
    if matrix_format != "FULL":
        raise ValueError("Only full Touchstone matrices are supported.")
    if version.startswith("1") and not selected.allow_version_1:
        raise ValueError("Touchstone 1.0 is disabled by policy.")
    if version.startswith("2") and not selected.allow_version_2:
        raise ValueError("Touchstone 2.0 is disabled by policy.")
    if not (version.startswith("1") or version.startswith("2")):
        raise ValueError("Only Touchstone versions 1.x and 2.x are supported.")
    unit, data_format, global_reference = option_value
    record_size = 1 + 2 * port_count * port_count
    if len(network_tokens) == 0 or len(network_tokens) % record_size != 0:
        raise ValueError("Malformed Touchstone continuation or matrix column count.")
    records = len(network_tokens) // record_size
    if number_of_frequencies is not None and number_of_frequencies != records:
        raise ValueError("[Number of Frequencies] does not match network data.")
    frequencies = np.empty((records,), dtype=np.float64)
    scattering = np.empty((records, port_count, port_count), dtype=np.complex128)
    for record in range(records):
        tokens = network_tokens[record * record_size : (record + 1) * record_size]
        frequencies[record] = float(tokens[0]) * _UNIT_SCALE[unit]
        values = [
            _complex(
                float(tokens[1 + 2 * entry]), float(tokens[2 + 2 * entry]), data_format
            )
            for entry in range(port_count * port_count)
        ]
        scattering[record] = np.asarray(values, dtype=np.complex128).reshape(
            (port_count, port_count), order="F"
        )
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(scattering)):
        raise ValueError("Touchstone numeric data must be finite.")
    if selected.require_monotone_frequency and not np.all(np.diff(frequencies) > 0.0):
        raise ValueError("Touchstone frequencies must be strictly increasing and unique.")
    reference_values = (
        [global_reference] * port_count if references is None else references
    )
    if len(reference_values) == 1:
        reference_values *= port_count
    if len(reference_values) != port_count or not np.all(
        np.asarray(reference_values) > 0.0
    ):
        raise ValueError("[Reference] must contain one positive real value per port.")
    names = tuple(port_names.get(port, f"port{port + 1}") for port in range(port_count))
    return TouchstoneData(
        frequencies,
        scattering,
        np.asarray(reference_values),
        port_names=names,
        data_format=data_format,
        frequency_unit=unit,
        version=version,
        source_hash=hashlib.sha256(raw).hexdigest(),
    )


def _pair(value: complex, data_format: TouchstoneFormat) -> tuple[float, float]:
    if data_format == "RI":
        return float(np.real(value)), float(np.imag(value))
    magnitude = float(np.abs(value))
    angle = float(np.rad2deg(np.angle(value)))
    if data_format == "MA":
        return magnitude, angle
    return (
        float(-np.inf) if magnitude == 0.0 else float(20.0 * np.log10(magnitude))
    ), angle


def write_touchstone(
    path: str | Path,
    data: TouchstoneData,
    /,
    *,
    version: str | None = None,
    data_format: TouchstoneFormat | None = None,
    frequency_unit: str | None = None,
) -> None:
    """Write the supported full-matrix subset without changing port order or references."""
    if not isinstance(data, TouchstoneData):
        raise TypeError("data must be TouchstoneData.")
    output = Path(path)
    target_version = data.version if version is None else str(version)
    target_format = data.data_format if data_format is None else data_format
    unit = data.frequency_unit if frequency_unit is None else str(frequency_unit).upper()
    if target_format not in ("RI", "MA", "DB") or unit not in _UNIT_SCALE:
        raise ValueError("Unsupported output format or frequency unit.")
    if not (target_version.startswith("1") or target_version.startswith("2")):
        raise ValueError("Only Touchstone versions 1.x and 2.x can be written.")
    references = np.asarray(data.reference_impedance)
    if target_version.startswith("1") and not np.all(references == references[0]):
        raise ValueError("Per-port reference impedances require Touchstone 2.0.")
    lines: list[str] = []
    if target_version.startswith("2"):
        lines.extend(
            (
                f"[Version] {target_version}",
                f"[Number of Ports] {data.port_count}",
                f"[Number of Frequencies] {data.frequencies_hz.size}",
                "[Matrix Format] Full",
                "[Reference] " + " ".join(f"{value:.17g}" for value in references),
            )
        )
    for index, name in enumerate(data.port_names):
        lines.append(f"! Port[{index + 1}] = {name}")
    lines.append(f"# {unit} S {target_format} R {references[0]:.17g}")
    if target_version.startswith("2"):
        lines.append("[Network Data]")
    scale = _UNIT_SCALE[unit]
    matrices = np.asarray(data.scattering)
    frequencies = np.asarray(data.frequencies_hz)
    for frequency, matrix in zip(frequencies, matrices, strict=True):
        tokens = [f"{frequency / scale:.17g}"]
        for value in matrix.reshape((-1,), order="F"):
            first, second = _pair(value, target_format)
            tokens.extend((f"{first:.17g}", f"{second:.17g}"))
        lines.append(" ".join(tokens))
    if target_version.startswith("2"):
        lines.append("[End]")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "TouchstoneData",
    "TouchstoneFormat",
    "TouchstonePolicy",
    "read_touchstone",
    "write_touchstone",
]
