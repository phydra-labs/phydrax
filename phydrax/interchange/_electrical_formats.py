#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import AMPERE, convert_value, METER, UnitDefinition, VOLT
from ._geospatial import GeospatialContract
from ._report import AdapterCapability, AdapterFormatProfile, AdapterReport, AdapterStatus
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits


_REQUIRED_COLUMNS = (
    "source_a_x",
    "source_a_y",
    "source_a_z",
    "source_b_x",
    "source_b_y",
    "source_b_z",
    "receiver_m_x",
    "receiver_m_y",
    "receiver_m_z",
    "receiver_n_x",
    "receiver_n_y",
    "receiver_n_z",
    "current",
    "voltage",
    "standard_deviation",
)


class ElectricalTabularSurvey(StrictModule, NonTrainableState):
    source_a_m: Array
    source_b_m: Array
    receiver_m_m: Array
    receiver_n_m: Array
    current_A: Array
    voltage_V: Array
    standard_deviation_V: Array
    valid: Array
    coordinates: GeospatialContract
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    survey_id: str = eqx.field(static=True)


def read_electrical_survey_csv(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    coordinates: GeospatialContract,
    length_unit: UnitDefinition = METER,
    current_unit: UnitDefinition = AMPERE,
    voltage_unit: UnitDefinition = VOLT,
    delimiter: str = ",",
) -> ElectricalTabularSurvey:
    if not isinstance(coordinates, GeospatialContract):
        raise TypeError("Electrical survey coordinates require GeospatialContract.")
    spatial = coordinates.require_cartesian(dimensions=3)
    if spatial.length_unit != length_unit:
        raise ValueError("Electrical CSV length unit and coordinate contract disagree.")
    if len(delimiter) != 1 or "\n" in delimiter or "\r" in delimiter:
        raise ValueError("Electrical CSV delimiter must be exactly one character.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    try:
        text = resource.data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Electrical CSV must be UTF-8.") from error
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Electrical CSV requires a header and at least one observation.")
    columns = tuple(value.strip() for value in lines[0].split(delimiter))
    if columns != _REQUIRED_COLUMNS:
        raise ValueError(
            "Electrical CSV columns must exactly match the qualified profile."
        )
    if len(lines) - 1 > limits.max_nodes or len(columns) > limits.max_attributes:
        raise ValueError("Electrical CSV exceeds decoded resource limits.")
    values = np.genfromtxt(lines[1:], delimiter=delimiter, dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] != len(columns):
        raise ValueError("Electrical CSV rows do not match the declared columns.")
    valid = np.all(np.isfinite(values[:, :14]), axis=1) & np.isfinite(values[:, 14])
    if np.any(~np.isfinite(values[:, :14])) or np.any(values[:, 12] == 0):
        raise ValueError(
            "Electrical geometry/current/voltage must be finite and current nonzero."
        )
    if np.any(valid & (values[:, 14] <= 0)):
        raise ValueError("Valid electrical standard deviations must be positive.")
    length_factor = float(convert_value(1.0, source=length_unit, target=METER))
    source_a = values[:, 0:3] * length_factor
    source_b = values[:, 3:6] * length_factor
    receiver_m = values[:, 6:9] * length_factor
    receiver_n = values[:, 9:12] * length_factor
    current = np.asarray(convert_value(values[:, 12], source=current_unit, target=AMPERE))
    voltage = np.asarray(convert_value(values[:, 13], source=voltage_unit, target=VOLT))
    deviation = np.asarray(convert_value(values[:, 14], source=voltage_unit, target=VOLT))
    identity = canonical_fingerprint(
        {
            "kind": "electrical-tabular-survey",
            "resource": resource.manifest.content_sha256,
            "coordinates": coordinates.coordinate_id,
            "source_units": (
                length_unit.unit_id,
                current_unit.unit_id,
                voltage_unit.unit_id,
            ),
            "source_a_m": source_a,
            "source_b_m": source_b,
            "receiver_m_m": receiver_m,
            "receiver_n_m": receiver_n,
            "current_A": current,
            "voltage_V": voltage,
            "standard_deviation_V": deviation,
            "valid": valid,
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "electrical-survey-csv",
        "ElectricalTabularSurvey",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "electrical-survey-csv",
            qualifiers={
                "columns": ",".join(_REQUIRED_COLUMNS),
                "delimiter": delimiter,
                "length_unit": length_unit.unit_id,
                "current_unit": current_unit.unit_id,
                "voltage_unit": voltage_unit.unit_id,
            },
        ),
        coordinate_mapping=("A/B/M/N coordinates normalized to Cartesian metres",),
        assumptions=(
            "The caller explicitly binds the profile's bare coordinate/current/voltage columns to the supplied units.",
        ),
        preserved_fields=(
            "source and receiver positions",
            "signed current and voltage",
            "standard deviation and validity",
            "exact source bytes",
        ),
        capabilities=(
            AdapterCapability("electrical-quadrupole-observations"),
            AdapterCapability("measurement-uncertainty"),
        ),
    )
    return ElectricalTabularSurvey(
        jnp.asarray(source_a),
        jnp.asarray(source_b),
        jnp.asarray(receiver_m),
        jnp.asarray(receiver_n),
        jnp.asarray(current),
        jnp.asarray(voltage),
        jnp.asarray(deviation),
        jnp.asarray(valid),
        coordinates,
        resource,
        report,
        identity,
    )


__all__ = ["ElectricalTabularSurvey", "read_electrical_survey_csv"]
