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
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest
from ._bodies import CelestialBodyCatalog
from ._chebyshev_ephemeris import ChebyshevEphemeris
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._eop import EarthOrientationRecordSet, PreparedEarthOrientation
from ._gravity_field import SphericalHarmonicGravityField
from ._time import LeapSecondTable


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
    angle_unit: str = eqx.field(static=True)
    coefficient_names: tuple[str, ...] = eqx.field(static=True)
    coefficients: tuple[Array, ...]
    provenance: AstrodynamicsDataProvenance
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str,
        frame_id: str,
        angle_unit: str,
        coefficients: dict[str, object],
        provenance: AstrodynamicsDataProvenance,
        /,
    ):
        names = tuple(sorted(str(name) for name in coefficients))
        values = tuple(
            jax.lax.stop_gradient(jnp.asarray(coefficients[name], dtype=float))
            for name in names
        )
        if (
            not model
            or not frame_id
            or not angle_unit
            or not names
            or any(bool(jnp.any(~jnp.isfinite(value))) for value in values)
        ):
            raise ValueError("Astronomy coefficient table is invalid.")
        self.model = str(model)
        self.frame_id = str(frame_id)
        self.angle_unit = str(angle_unit)
        self.coefficient_names = names
        self.coefficients = values
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "astronomy-coefficient-table",
                "model": self.model,
                "frame": self.frame_id,
                "angle_unit": self.angle_unit,
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
        sha256="773349a1039483ff740ac0f885b51917553a43d845e2d55d5b40bfdb6900d1af",
        byte_size=646,
        source_uri="package:phydrax.astrodynamics/earth_gravity_degree4.json",
        license_id="CC0-1.0",
        model="bounded-low-order-earth-gravity",
        coverage="degree/order 0:4",
    ),
    "sun_earth_moon_chebyshev.json": ArtifactManifest(
        artifact_id="phydrax-bundled-sun-earth-moon-chebyshev",
        producer="phydrax-curated-astronomy",
        version="2026-09-01",
        sha256="64e7acfa0734ecef0e8b0015aa79501a416f3bb7c0ef56a3ae5f81ed3cfe325c",
        byte_size=581,
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
    provenance = _provenance(
        manifest,
        frame_id=str(payload["frame_id"]),
        epoch_id="static-low-order",
        scale_id=context.scale.scale_id,
    )
    length_factor = 1000.0 / context.scale.length_to_reference
    mu_factor = length_factor**3 * context.scale.time_to_reference**2
    return SphericalHarmonicGravityField(
        payload["cosine"],
        payload["sine"],
        float(payload["mu_km3_s2"]) * mu_factor,
        float(payload["reference_radius_km"]) * length_factor,
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
    provenance = _provenance(
        manifest,
        frame_id=str(payload["frame_id"]),
        epoch_id=str(payload["epoch_id"]),
        scale_id=context.scale.scale_id,
    )
    length_factor = 1000.0 / context.scale.length_to_reference
    mu_factor = length_factor**3 * context.scale.time_to_reference**2
    catalog = CelestialBodyCatalog(
        tuple(str(value) for value in payload["body_ids"]),
        np.asarray(payload["gravitational_parameters_km3_s2"]) * mu_factor,
        np.asarray(payload["reference_radii_km"]) * length_factor,
        context,
    )
    bounds = np.asarray(payload["segment_bounds_tdb_seconds"]) / (
        context.scale.time_to_reference
    )
    coefficients = np.asarray(payload["position_coefficients_km"]) * length_factor
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
        str(payload["angle_unit"]),
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
