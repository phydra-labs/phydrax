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
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits


class ICGEMGravityModel(StrictModule, NonTrainableState):
    model_name: str = eqx.field(static=True)
    gravitational_constant_m3_s2: float = eqx.field(static=True)
    reference_radius_m: float = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    tide_system: str = eqx.field(static=True)
    cosine: Array
    sine: Array
    cosine_sigma: Array
    sine_sigma: Array
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


class GeomagneticHarmonicModel(StrictModule, NonTrainableState):
    model_name: str = eqx.field(static=True)
    epoch_decimal_year: float = eqx.field(static=True)
    reference_radius_m: float = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    g_nT: Array
    h_nT: Array
    secular_g_nT_year: Array
    secular_h_nT_year: Array
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    model_id: str = eqx.field(static=True)


def _tokens(resource: BoundedResource) -> list[str]:
    try:
        text = resource.data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Potential-field coefficient file must be ASCII.") from error
    return text.splitlines()


def read_icgem_gfc(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
) -> ICGEMGravityModel:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    lines = _tokens(resource)
    header: dict[str, str] = {}
    coefficients: list[tuple[int, int, float, float, float, float]] = []
    in_coefficients = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_coefficients:
            if stripped == "end_of_head":
                in_coefficients = True
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2 or parts[0] in header:
                raise ValueError("ICGEM header must contain unique key/value records.")
            header[parts[0]] = parts[1]
            if len(header) > limits.max_attributes:
                raise ValueError("ICGEM header exceeds the attribute limit.")
            continue
        parts = stripped.replace("D", "E").split()
        if parts[0] != "gfc" or len(parts) not in (5, 7):
            raise ValueError("Static ICGEM profile accepts only gfc coefficient records.")
        if len(coefficients) >= limits.max_nodes:
            raise ValueError("ICGEM coefficients exceed the decoded-node limit.")
        degree, order = int(parts[1]), int(parts[2])
        sigma_c = float(parts[5]) if len(parts) == 7 else np.nan
        sigma_s = float(parts[6]) if len(parts) == 7 else np.nan
        coefficients.append(
            (degree, order, float(parts[3]), float(parts[4]), sigma_c, sigma_s)
        )
    required = {
        "modelname",
        "earth_gravity_constant",
        "radius",
        "max_degree",
        "norm",
        "tide_system",
    }
    if not in_coefficients or not required.issubset(header) or not coefficients:
        raise ValueError(
            "ICGEM file lacks required static model metadata or coefficients."
        )
    maximum = int(header["max_degree"])
    if maximum < 0:
        raise ValueError("ICGEM maximum degree must be nonnegative.")
    if 4 * (maximum + 1) ** 2 > limits.max_nodes:
        raise ValueError("ICGEM dense coefficient arrays exceed the decoded-node limit.")
    shape = (maximum + 1, maximum + 1)
    cosine = np.zeros(shape)
    sine = np.zeros(shape)
    sigma_c = np.full(shape, np.nan)
    sigma_s = np.full(shape, np.nan)
    seen: set[tuple[int, int]] = set()
    for degree, order, c_value, s_value, c_sigma, s_sigma in coefficients:
        if not 0 <= order <= degree <= maximum or (degree, order) in seen:
            raise ValueError("ICGEM coefficient degree/order is invalid or duplicated.")
        if not np.all(np.isfinite((c_value, s_value))):
            raise ValueError("ICGEM coefficients must be finite.")
        seen.add((degree, order))
        cosine[degree, order], sine[degree, order] = c_value, s_value
        sigma_c[degree, order], sigma_s[degree, order] = c_sigma, s_sigma
    gm = float(header["earth_gravity_constant"].replace("D", "E"))
    radius = float(header["radius"].replace("D", "E"))
    if not np.isfinite(gm) or gm <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("ICGEM GM and radius must be positive finite SI values.")
    identity = canonical_fingerprint(
        {
            "kind": "icgem-gravity-model",
            "resource": resource.manifest.content_sha256,
            "header": header,
            "cosine": cosine,
            "sine": sine,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "ICGEM-gfc",
        "ICGEMGravityModel",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "ICGEM-gfc", qualifiers={"profile": "static-gfc"}
        ),
        preserved_fields=(
            "normalization and tide system",
            "GM and reference radius",
            "static coefficients and supplied sigmas",
            "exact source bytes",
        ),
        losses=(
            AdapterLoss(
                "unmodeled-header-fields",
                "import",
                "dropped",
                "The normalized model retains required static gravity fields; "
                "additional header fields remain only in the exact source bytes.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(AdapterCapability("spherical-harmonic-gravity"),),
    )
    return ICGEMGravityModel(
        header["modelname"],
        gm,
        radius,
        maximum,
        header["norm"],
        header["tide_system"],
        jnp.asarray(cosine),
        jnp.asarray(sine),
        jnp.asarray(sigma_c),
        jnp.asarray(sigma_s),
        resource,
        report,
        identity,
    )


def read_geomagnetic_coefficients(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    model_name: str,
    epoch_decimal_year: float,
    reference_radius_m: float,
) -> GeomagneticHarmonicModel:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    records: list[tuple[int, int, float, float, float, float]] = []
    for line in _tokens(resource):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace("D", "E").split()
        if len(parts) != 6:
            raise ValueError("Geomagnetic profile requires n m g h dg/dt dh/dt records.")
        if len(records) >= limits.max_nodes:
            raise ValueError("Geomagnetic coefficients exceed the decoded-node limit.")
        records.append(
            (
                int(parts[0]),
                int(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    if not records:
        raise ValueError("Geomagnetic coefficient file is empty.")
    maximum = max(record[0] for record in records)
    if 4 * (maximum + 1) ** 2 > limits.max_nodes:
        raise ValueError(
            "Geomagnetic dense coefficient arrays exceed the decoded-node limit."
        )
    shape = (maximum + 1, maximum + 1)
    g, h, dg, dh = (np.zeros(shape) for _ in range(4))
    seen: set[tuple[int, int]] = set()
    for degree, order, g_value, h_value, dg_value, dh_value in records:
        if (
            not 1 <= degree <= maximum
            or not 0 <= order <= degree
            or (degree, order) in seen
        ):
            raise ValueError("Geomagnetic degree/order is invalid or duplicated.")
        if not np.all(np.isfinite((g_value, h_value, dg_value, dh_value))):
            raise ValueError("Geomagnetic coefficients must be finite.")
        seen.add((degree, order))
        g[degree, order], h[degree, order] = g_value, h_value
        dg[degree, order], dh[degree, order] = dg_value, dh_value
    epoch, radius = float(epoch_decimal_year), float(reference_radius_m)
    name = str(model_name).strip()
    if not name or not np.isfinite(epoch) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Geomagnetic model identity, epoch, and radius are required.")
    identity = canonical_fingerprint(
        {
            "kind": "geomagnetic-harmonic-model",
            "resource": resource.manifest.content_sha256,
            "name": name,
            "epoch": epoch,
            "radius_m": radius,
            "coefficients": (g, h, dg, dh),
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "geomagnetic-coefficients",
        "GeomagneticHarmonicModel",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "geomagnetic-coefficients",
            qualifiers={
                "profile": "n-m-g-h-dg-dh",
                "model_name": name,
                "epoch_decimal_year": str(epoch),
                "reference_radius_m": str(radius),
            },
        ),
        preserved_fields=(
            "degree/order Gauss coefficients and secular variation",
            "exact source bytes",
        ),
        assumptions=(
            "Model name, epoch, reference radius, and Schmidt normalization are caller-bound profile semantics.",
        ),
        capabilities=(AdapterCapability("spherical-harmonic-magnetics"),),
    )
    return GeomagneticHarmonicModel(
        name,
        epoch,
        radius,
        maximum,
        jnp.asarray(g),
        jnp.asarray(h),
        jnp.asarray(dg),
        jnp.asarray(dh),
        resource,
        report,
        identity,
    )


__all__ = [
    "GeomagneticHarmonicModel",
    "ICGEMGravityModel",
    "read_geomagnetic_coefficients",
    "read_icgem_gfc",
]
