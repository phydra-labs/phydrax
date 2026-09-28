"""Thin-film interference and colorimetry qualification against pinned CIE data.

The CIE 1931 color-matching and D65 datasets are licensed CC BY-SA 4.0 and are
never vendored. This tool reads them from a local cache directory; with
``--fetch`` it downloads them from the CIE only when the bytes match the pinned
SHA-256 published in the CIE dataset metadata. Every table then enters Phydrax
through the admitted-artifact policy, exactly as a user resource would.

Checks:

- analytic Wyman-Sloan-Shirley observer error against the CIE 1931 table,
  compared with the published Table 2 values;
- unit reflector under CIE D65 in linear sRGB for the analytic and tabulated
  observers on two spectral grids;
- equal-energy white chromaticity;
- Airy summation against an independent characteristic-matrix (Abeles/Macleod
  admittance) reference for dielectric and absorbing substrates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import urllib.request
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.optics.wave import ThinFilmInterferencePlan
from phydrax.rendering import (
    AnalyticColorMatchingFunctions,
    COLOR_MATCHING_TABLE_MODEL,
    read_spectral_illuminant,
    SpectralColorimetryPlan,
    SpectralIlluminant,
    TabulatedColorMatchingFunctions,
)


jax.config.update("jax_enable_x64", True)

_CIE_BASE_URL = "https://files.cie.co.at/Publications-datasets/"
_LICENSE = "CC-BY-SA-4.0"
_RESOURCES = {
    "d65": {
        "file": "CIE_std_illum_D65.csv",
        "sha256": "e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb",
        "doi": "10.25039/CIE.DS.hjfjmt59",
        "model": "cie-standard-illuminant-d65",
    },
    "cie1931": {
        "file": "CIE_xyz_1931_2deg.csv",
        "sha256": "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1",
        "doi": "10.25039/CIE.DS.xvudnb9b",
        "model": COLOR_MATCHING_TABLE_MODEL,
    },
}
_PUBLISHED_MAXIMUM_SQUARED = (2.0e-4, 6.4e-5, 4.9e-4)
_PUBLISHED_MEAN_SQUARED = (3.1e-5, 7.1e-6, 1.6e-5)


def _resource(cache: Path, key: str, fetch: bool, /) -> tuple[Path, bytes]:
    record = _RESOURCES[key]
    path = cache / record["file"]
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path} is missing; rerun with --fetch.")
        with urllib.request.urlopen(_CIE_BASE_URL + record["file"], timeout=120) as reply:
            payload = reply.read()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Fetched {record['file']} does not match its SHA-256 pin.")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"{path} does not match its pinned SHA-256.")
    return path, payload


def _manifest(key: str, payload: bytes, /) -> ArtifactManifest:
    record = _RESOURCES[key]
    return ArtifactManifest(
        artifact_id=record["file"],
        producer="CIE",
        version=record["doi"],
        sha256=record["sha256"],
        byte_size=len(payload),
        source_uri=_CIE_BASE_URL + record["file"],
        license_id=_LICENSE,
        model=record["model"],
        coverage="1 nm CIE dataset",
    )


def _characteristic_matrix(
    wavelengths: np.ndarray,
    ambient: float,
    film: float,
    substrate: complex,
    thickness: float,
    cosine: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Macleod-admittance reference in the exp(+i omega t) convention."""
    index2 = np.conj(np.complex128(substrate))
    tangential = (ambient * np.sqrt(1.0 - cosine * cosine)) ** 2
    normal0 = np.complex128(ambient * cosine)
    normal1 = np.sqrt(np.complex128(film * film - tangential))
    normal1 = np.where(np.imag(normal1) > 0.0, -normal1, normal1)
    normal2 = np.sqrt(index2 * index2 - tangential)
    normal2 = np.where(np.imag(normal2) > 0.0, -normal2, normal2)
    delta = 2.0 * np.pi * normal1 * thickness / wavelengths
    reflectance = []
    transmittance = []
    for admittance0, admittance1, admittance2 in (
        (normal0, normal1, normal2),
        (
            ambient**2 / normal0,
            film**2 / normal1,
            index2 * index2 / normal2,
        ),
    ):
        b = np.cos(delta) + 1j * np.sin(delta) * admittance2 / admittance1
        c = 1j * admittance1 * np.sin(delta) + np.cos(delta) * admittance2
        denominator = admittance0 * b + c
        reflectance.append(np.abs((admittance0 * b - c) / denominator) ** 2)
        transmittance.append(
            4.0 * np.real(admittance0) * np.real(admittance2) / np.abs(denominator) ** 2
        )
    return np.stack(reflectance, axis=-1), np.stack(transmittance, axis=-1)


def _airy_campaign() -> dict[str, Any]:
    wavelengths = np.arange(380.0, 781.0, 5.0) / 1.0e9
    thicknesses = np.linspace(0.0, 2.0e-6, 21)
    cosines = np.cos(np.deg2rad(np.linspace(0.0, 80.0, 9)))
    cases: dict[str, Any] = {}
    for name, ambient, film, substrate in (
        ("air-soap-air", 1.0, 1.33, 1.0),
        ("air-soap-glass", 1.0, 1.33, 1.52 + 0.0j),
        ("air-soap-silver-like", 1.0, 1.33, 0.05 + 3.3j),
        ("air-soap-silicon-like", 1.0, 1.33, 4.0 + 0.03j),
    ):
        plan = ThinFilmInterferencePlan(wavelengths, ambient, film, substrate)
        result = plan.evaluate(thicknesses[:, None], cosines[None, :])
        reflectance = np.asarray(result.reflectance)
        transmittance = np.asarray(result.transmittance)
        reference_r = np.empty_like(reflectance)
        reference_t = np.empty_like(transmittance)
        for row, thickness in enumerate(thicknesses):
            for column, cosine in enumerate(cosines):
                reference_r[row, column], reference_t[row, column] = (
                    _characteristic_matrix(
                        wavelengths, ambient, film, substrate, thickness, cosine
                    )
                )
        reflectance_error = float(np.max(np.abs(reflectance - reference_r)))
        transmittance_error = float(np.max(np.abs(transmittance - reference_t)))
        successful = bool(jnp.all(result.accepted))
        cases[name] = {
            "maximum_reflectance_difference": reflectance_error,
            "maximum_transmittance_difference": transmittance_error,
            "maximum_energy_residual": float(jnp.max(result.energy_residual)),
            "accepted_fraction": int(jnp.sum(result.accepted)) / result.accepted.size,
            "successful": successful,
            "passed": successful
            and reflectance_error <= 1.0e-12
            and transmittance_error <= 1.0e-12,
        }
    return cases


def _observer_campaign(
    observer: TabulatedColorMatchingFunctions,
) -> dict[str, Any]:
    analytic = AnalyticColorMatchingFunctions().sample(observer.table_wavelengths)
    squared = (analytic - observer.table_values) ** 2
    maximum = squared.max(axis=0)
    mean = squared.mean(axis=0)
    return {
        "maximum_squared_error": maximum.tolist(),
        "mean_squared_error": mean.tolist(),
        "published_maximum_squared_error": list(_PUBLISHED_MAXIMUM_SQUARED),
        "published_mean_squared_error": list(_PUBLISHED_MEAN_SQUARED),
        "passed": bool(
            np.all(np.abs(maximum - _PUBLISHED_MAXIMUM_SQUARED) <= 0.05 * maximum)
            and np.all(np.abs(mean - _PUBLISHED_MEAN_SQUARED) <= 0.05 * mean)
        ),
    }


def _white_campaign(
    d65: SpectralIlluminant, observer: TabulatedColorMatchingFunctions
) -> dict[str, Any]:
    grids = {
        "360-830nm-1nm": np.arange(360.0, 831.0, 1.0) / 1.0e9,
        "380-780nm-5nm": np.arange(380.0, 781.0, 5.0) / 1.0e9,
    }
    bounds = {"analytic": 3.0e-3, "tabulated": 1.0e-3}
    records: dict[str, Any] = {}
    for grid_name, grid in grids.items():
        for observer_name, functions in (
            ("analytic", AnalyticColorMatchingFunctions()),
            ("tabulated", observer),
        ):
            plan = SpectralColorimetryPlan(grid, d65, color_matching=functions)
            white = np.asarray(plan.white_xyz)
            records[f"{observer_name}:{grid_name}"] = {
                "white_linear_srgb": np.asarray(plan.white_linear_srgb).tolist(),
                "white_neutral_error": plan.white_neutral_error,
                "white_chromaticity": (white[:2] / white.sum()).tolist(),
                "bound": bounds[observer_name],
                "passed": plan.white_neutral_error <= bounds[observer_name],
            }
    full = grids["360-830nm-1nm"]
    equal_energy = SpectralIlluminant(full, np.ones_like(full), illuminant_id="cie-e")
    for observer_name, functions, bound in (
        ("analytic", AnalyticColorMatchingFunctions(), 1.0e-3),
        ("tabulated", observer, 1.0e-4),
    ):
        plan = SpectralColorimetryPlan(full, equal_energy, color_matching=functions)
        white = np.asarray(plan.white_xyz)
        deviation = float(np.max(np.abs(white[:2] / white.sum() - 1.0 / 3.0)))
        records[f"equal-energy:{observer_name}"] = {
            "chromaticity_deviation": deviation,
            "bound": bound,
            "passed": deviation <= bound,
        }
    return records


def _qualification_successful(record: dict[str, Any], /) -> bool:
    return bool(
        record["observer_fit"]["passed"]
        and all(entry["passed"] for entry in record["white_balance"].values())
        and all(
            entry["successful"] is True and entry["passed"] is True
            for entry in record["airy_versus_characteristic_matrix"].values()
        )
    )


def qualify(cache: Path, fetch: bool, /) -> dict[str, Any]:
    d65_path, d65_bytes = _resource(cache, "d65", fetch)
    observer_path, observer_bytes = _resource(cache, "cie1931", fetch)
    if d65_path.parent != observer_path.parent:
        raise RuntimeError("CIE resources must share one cache directory.")
    policy = ExternalArtifactPolicy(
        d65_path.parent,
        maximum_bytes=1 << 20,
        allowed_license_ids=[_LICENSE],
        allowed_suffixes=[".csv"],
    )
    d65_manifest = _manifest("d65", d65_bytes)
    observer_manifest = _manifest("cie1931", observer_bytes)
    d65 = read_spectral_illuminant(
        admit_external_artifact(d65_path.name, d65_manifest, policy=policy),
        d65_manifest,
        policy=policy,
        illuminant_id="cie-standard-illuminant-d65",
    )
    observer = TabulatedColorMatchingFunctions(
        admit_external_artifact(observer_path.name, observer_manifest, policy=policy),
        observer_manifest,
        policy=policy,
    )
    record = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
        },
        "sources": {
            key: {
                "doi": value["doi"],
                "sha256": value["sha256"],
                "license": _LICENSE,
                "vendored": False,
            }
            for key, value in _RESOURCES.items()
        },
        "observer_fit": _observer_campaign(observer),
        "white_balance": _white_campaign(d65, observer),
        "airy_versus_characteristic_matrix": _airy_campaign(),
    }
    record["successful"] = _qualification_successful(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/phydrax/cie"))
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download missing CIE tables; bytes must match the pinned SHA-256.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    record = qualify(arguments.cache_dir.resolve(), arguments.fetch)
    text = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(text)
    return 0 if record["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
