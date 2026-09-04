#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest
from ...units import (
    ANGLE,
    convert_value,
    DEGREE,
    derived_unit,
    KILOMETER,
    SECOND,
    UnitDefinition,
)
from ._bodies import CelestialBodyCatalog
from ._chebyshev_ephemeris import ChebyshevEphemeris
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._eop import EarthOrientationRecordSet, PreparedEarthOrientation
from ._gravity_field import SphericalHarmonicGravityField
from ._time import LeapSecondTable


_ARCSECOND = UnitDefinition(
    "arcsecond",
    ANGLE,
    DEGREE.reference_system_id,
    DEGREE.scale_to_reference / 3600,
)
_ASTRONOMY_ANGLE_UNITS = {"arcsecond": _ARCSECOND}
_ASTRONOMY_LENGTH_UNITS = {"km": KILOMETER}
_ASTRONOMY_TIME_UNITS = {"s": SECOND}


def _source_unit(
    payload: dict[str, object],
    field: str,
    aliases: dict[str, UnitDefinition],
    /,
) -> UnitDefinition:
    token = payload.get(field)
    if not isinstance(token, str) or token not in aliases:
        raise ValueError(f"Unknown astronomy source unit {field}={token!r}.")
    return aliases[token]


def _require_payload_frame(
    payload: dict[str, object],
    context: AstrodynamicsContext,
    /,
) -> None:
    if (
        payload.get("origin_id") != context.frame.origin_id
        or payload.get("orientation_id") != context.frame.orientation_id
        or payload.get("pseudo_inertial") is not context.frame.pseudo_inertial
    ):
        raise ValueError("Astronomy source frame does not match the target context.")


def _require_payload_epoch(
    payload: dict[str, object],
    context: AstrodynamicsContext,
    /,
) -> None:
    high = payload.get("reference_julian_date_high")
    low = payload.get("reference_julian_date_low")
    if (
        isinstance(high, bool)
        or not isinstance(high, (int, float))
        or isinstance(low, bool)
        or not isinstance(low, (int, float))
        or float(high) != context.epoch.instant.julian_date.high
        or float(low) != context.epoch.instant.julian_date.low
        or payload.get("time_scale") != context.epoch.time_scale
        or payload.get("continuous") is not context.epoch.continuous
    ):
        raise ValueError("Astronomy source epoch does not match the target context.")


class PinnedArtifact(StrictModule, NonTrainableState):
    path: str = eqx.field(static=True)
    manifest: ArtifactManifest


class AstrodynamicsDataStore(StrictModule, NonTrainableState):
    root: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)

    def __init__(self, root: str | Path, /):
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                "Astrodynamics data-store root must exist and be a directory."
            )
        self.root = str(path)
        self.store_id = canonical_fingerprint(
            {"kind": "astrodynamics-data-store", "root": str(path)}
        )

    def resolve(
        self, relative_path: str, manifest: ArtifactManifest, /
    ) -> PinnedArtifact:
        if not isinstance(manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        path = (Path(self.root) / relative_path).resolve()
        if Path(self.root) not in path.parents:
            raise ValueError("Artifact path escapes the configured store.")
        if not path.is_file():
            raise ValueError("Pinned artifact is absent from the configured store.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.sha256 or len(payload) != manifest.byte_size:
            raise ValueError(
                "Pinned artifact checksum or byte size does not match manifest."
            )
        return PinnedArtifact(str(path), manifest)


class AstronomyCoefficientTable(StrictModule, NonTrainableState):
    model: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    angle_unit: UnitDefinition = eqx.field(static=True)
    coefficient_names: tuple[str, ...] = eqx.field(static=True)
    coefficients: tuple[Array, ...]
    provenance: AstrodynamicsDataProvenance
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str,
        frame_id: str,
        angle_unit: UnitDefinition,
        coefficients: dict[str, object],
        provenance: AstrodynamicsDataProvenance,
        /,
    ):
        names = tuple(sorted(str(name) for name in coefficients))
        values = tuple(
            jax.lax.stop_gradient(jnp.asarray(coefficients[name], dtype=float))
            for name in names
        )
        if not isinstance(angle_unit, UnitDefinition):
            raise TypeError("angle_unit must be a UnitDefinition.")
        if (
            not model
            or not frame_id
            or angle_unit.dimension != ANGLE
            or not names
            or any(bool(jnp.any(~jnp.isfinite(value))) for value in values)
        ):
            raise ValueError("Astronomy coefficient table is invalid.")
        self.model = str(model)
        self.frame_id = str(frame_id)
        self.angle_unit = angle_unit
        self.coefficient_names = names
        self.coefficients = values
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "astronomy-coefficient-table",
                "model": self.model,
                "frame": self.frame_id,
                "angle_unit": self.angle_unit.unit_id,
                "coefficient_names": list(names),
                "provenance": provenance.provenance_id,
            }
        )

    def coefficient(self, name: str, /) -> Array:
        key = str(name)
        if key not in self.coefficient_names:
            raise KeyError(f"Unknown astronomy coefficient {key!r}.")
        return self.coefficients[self.coefficient_names.index(key)]


ASTRONOMY_ASSET_MANIFESTS = {
    "leap_seconds.json": ArtifactManifest(
        artifact_id="phydrax-bundled-leap-seconds",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="b7d5f4e80f8d15e6373adb98b14d537b130a39e72a7ee8058e8175192c221edf",
        byte_size=473,
        source_uri="package:phydrax.astrodynamics/leap_seconds.json",
        license_id="CC0-1.0",
        model="UTC-TAI-step-table",
        coverage="1972-01-01/2030-01-01",
    ),
    "eop_cip_2024.json": ArtifactManifest(
        artifact_id="phydrax-bundled-eop-cip-2024",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="ea26e3a540fee4bed37660b7ca4f18a50e5b75fa9560d760bd251668f2236b1d",
        byte_size=653,
        source_uri="package:phydrax.astrodynamics/eop_cip_2024.json",
        license_id="CC0-1.0",
        model="bounded-IERS-EOP-CIP-example",
        coverage="2024-01-01/2024-01-05",
    ),
    "earth_gravity_degree4.json": ArtifactManifest(
        artifact_id="phydrax-bundled-earth-gravity-degree4",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="c6e0ed18129de020c288f79cafad921e7d20753520bf6d0ef81c7487df00e20c",
        byte_size=742,
        source_uri="package:phydrax.astrodynamics/earth_gravity_degree4.json",
        license_id="CC0-1.0",
        model="bounded-low-order-earth-gravity",
        coverage="degree/order 0:4",
    ),
    "sun_earth_moon_chebyshev.json": ArtifactManifest(
        artifact_id="phydrax-bundled-sun-earth-moon-chebyshev",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="fb8ad29923c9035d77bc7cda917ebc6baf4bc01400c5dc0868f90d7d5845b2a4",
        byte_size=710,
        source_uri="package:phydrax.astrodynamics/sun_earth_moon_chebyshev.json",
        license_id="CC0-1.0",
        model="bounded-Sun-Earth-Moon-Chebyshev-example",
        coverage="TDB seconds 0/86400 from J2000+bounded-demo",
    ),
    "iau_precession_nutation.json": ArtifactManifest(
        artifact_id="phydrax-bundled-iau-precession-nutation",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="20ac3332ef08c10415cc41fa782aec19d2fc2941f3f084124e992a882e955b2e",
        byte_size=448,
        source_uri="package:phydrax.astrodynamics/iau_precession_nutation.json",
        license_id="CC0-1.0",
        model="IAU-2006-precession-bounded-coefficients",
        coverage="published polynomial coefficient support",
    ),
}


def bundled_astronomy_data_store() -> AstrodynamicsDataStore:
    return AstrodynamicsDataStore(Path(__file__).with_name("data"))


def _store(store: AstrodynamicsDataStore | None, /) -> AstrodynamicsDataStore:
    resolved = bundled_astronomy_data_store() if store is None else store
    if not isinstance(resolved, AstrodynamicsDataStore):
        raise TypeError("store must be an AstrodynamicsDataStore or None.")
    return resolved


def _payload(
    name: str, store: AstrodynamicsDataStore | None, /
) -> tuple[dict[str, object], ArtifactManifest]:
    manifest = ASTRONOMY_ASSET_MANIFESTS[name]
    pinned = _store(store).resolve(name, manifest)
    return json.loads(Path(pinned.path).read_text(encoding="utf-8")), manifest


def _provenance(
    manifest: ArtifactManifest,
    /,
    *,
    frame_id: str,
    epoch_id: str,
    scale_id: str,
) -> AstrodynamicsDataProvenance:
    return AstrodynamicsDataProvenance(
        producer=manifest.producer,
        producer_version=manifest.version,
        source_id=manifest.artifact_id,
        checksum=manifest.sha256,
        license_id=manifest.license_id,
        frame_id=frame_id,
        epoch_id=epoch_id,
        scale_id=scale_id,
        differentiability="constant",
    )


def load_bundled_leap_seconds(
    store: AstrodynamicsDataStore | None = None, /
) -> LeapSecondTable:
    payload, manifest = _payload("leap_seconds.json", store)
    provenance = _provenance(
        manifest,
        frame_id="UTC",
        epoch_id=str(payload["epoch_utc"]),
        scale_id="SI-second",
    )
    return LeapSecondTable(
        payload["transition_seconds"], payload["tai_minus_utc"], provenance
    )


def load_bundled_earth_orientation(
    store: AstrodynamicsDataStore | None = None, /
) -> PreparedEarthOrientation:
    payload, manifest = _payload("eop_cip_2024.json", store)
    provenance = _provenance(
        manifest,
        frame_id="GCRS/ITRS",
        epoch_id=str(payload["epoch_utc"]),
        scale_id="UTC-second",
    )
    records = EarthOrientationRecordSet(
        payload["relative_utc_seconds"],
        payload["xp_radians"],
        payload["yp_radians"],
        payload["dut1_seconds"],
        payload["lod_seconds"],
        payload["cip_x_radians"],
        payload["cip_y_radians"],
        payload["cio_s_radians"],
        payload["predicted"],
        provenance,
    )
    return PreparedEarthOrientation(records, float(payload["reference_jd_utc"]))


def load_bundled_earth_gravity(
    context: AstrodynamicsContext,
    store: AstrodynamicsDataStore | None = None,
    /,
) -> SphericalHarmonicGravityField:
    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    payload, manifest = _payload("earth_gravity_degree4.json", store)
    if context.scale.length_coordinate_kind != "physical":
        raise ValueError("Bundled gravity requires physical length coordinates.")
    _require_payload_frame(payload, context)
    provenance = _provenance(
        manifest,
        frame_id=context.frame.frame_id,
        epoch_id="static-low-order",
        scale_id=context.scale.scale_id,
    )
    source_length = _source_unit(payload, "length_unit", _ASTRONOMY_LENGTH_UNITS)
    source_time = _source_unit(payload, "time_unit", _ASTRONOMY_TIME_UNITS)
    source_mu = derived_unit(
        f"{source_length.symbol}^3/{source_time.symbol}^2",
        ((source_length, 3), (source_time, -2)),
    )
    mu = convert_value(
        payload["gravitational_parameter"],
        source=source_mu,
        target=context.scale.gravitational_parameter_unit,
    )
    reference_radius = convert_value(
        payload["reference_radius"],
        source=source_length,
        target=context.scale.length_unit,
    )
    return SphericalHarmonicGravityField(
        payload["cosine"],
        payload["sine"],
        mu,
        reference_radius,
        context,
        provenance,
        maximum_degree=int(payload["maximum_degree"]),
        maximum_order=int(payload["maximum_order"]),
        tide_system=str(payload["tide_system"]),
    )


def load_bundled_sun_earth_moon_ephemeris(
    context: AstrodynamicsContext,
    store: AstrodynamicsDataStore | None = None,
    /,
) -> ChebyshevEphemeris:
    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    payload, manifest = _payload("sun_earth_moon_chebyshev.json", store)
    if context.scale.length_coordinate_kind != "physical":
        raise ValueError("Bundled ephemeris requires physical length coordinates.")
    _require_payload_frame(payload, context)
    _require_payload_epoch(payload, context)
    provenance = _provenance(
        manifest,
        frame_id=context.frame.frame_id,
        epoch_id=context.epoch.epoch_id,
        scale_id=context.scale.scale_id,
    )
    source_length = _source_unit(payload, "length_unit", _ASTRONOMY_LENGTH_UNITS)
    source_time = _source_unit(payload, "time_unit", _ASTRONOMY_TIME_UNITS)
    source_mu = derived_unit(
        f"{source_length.symbol}^3/{source_time.symbol}^2",
        ((source_length, 3), (source_time, -2)),
    )
    catalog = CelestialBodyCatalog(
        tuple(str(value) for value in payload["body_ids"]),
        convert_value(
            payload["gravitational_parameters"],
            source=source_mu,
            target=context.scale.gravitational_parameter_unit,
        ),
        convert_value(
            payload["reference_radii"],
            source=source_length,
            target=context.scale.length_unit,
        ),
        context,
    )
    bounds = convert_value(
        payload["segment_bounds"],
        source=source_time,
        target=context.scale.time_unit,
    )
    coefficients = convert_value(
        payload["position_coefficients"],
        source=source_length,
        target=context.scale.length_unit,
    )
    return ChebyshevEphemeris(bounds, coefficients, catalog, provenance)


def load_bundled_iau_coefficients(
    store: AstrodynamicsDataStore | None = None, /
) -> AstronomyCoefficientTable:
    payload, manifest = _payload("iau_precession_nutation.json", store)
    provenance = _provenance(
        manifest,
        frame_id=str(payload["frame_id"]),
        epoch_id="J2000",
        scale_id=str(payload["time_argument"]),
    )
    coefficients = dict(payload["iau_2006_precession"])
    return AstronomyCoefficientTable(
        str(payload["model"]),
        str(payload["frame_id"]),
        _source_unit(payload, "angle_unit", _ASTRONOMY_ANGLE_UNITS),
        coefficients,
        provenance,
    )


__all__ = [
    "ASTRONOMY_ASSET_MANIFESTS",
    "ArtifactManifest",
    "AstronomyCoefficientTable",
    "AstrodynamicsDataStore",
    "PinnedArtifact",
    "bundled_astronomy_data_store",
    "load_bundled_earth_gravity",
    "load_bundled_earth_orientation",
    "load_bundled_iau_coefficients",
    "load_bundled_leap_seconds",
    "load_bundled_sun_earth_moon_ephemeris",
]
