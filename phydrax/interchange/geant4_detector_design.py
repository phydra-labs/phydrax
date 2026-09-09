#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only pinned GYM4DetectorDesign Fun4All/Geant4 tracking adapter.

The adapter runs the upstream pi-minus generator, Geant4 geometry and fast
Kalman tracking, then the upstream ROOT double-Gaussian reconstruction. The
known upstream C++ return type and silicon pitch/thickness unit defects are
corrected only in detached staged source bytes, with exact before/after evidence.
The caller supplies a pinned ROOT executable, exact field map, and an already
initialized ECCE/Fun4All/Geant4 runtime identified by ``runtime_build_id``.
Source: https://github.com/wmdataphys/GYM4DetectorDesign/tree/bbfd1b8dd10dc36b28ab118aac8af5dbace79296
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .._fingerprint import canonical_json
from ..artifacts import ScientificArtifactEnvelope
from ._device_design import device_artifact, DeviceQualificationError, DeviceSource
from .energy_runtime import (
    _host_only,
    EnergyRunResult,
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)


GYM4DETECTOR_COMMIT = "bbfd1b8dd10dc36b28ab118aac8af5dbace79296"
GYM4DETECTOR_SOURCE_URL = (
    "https://github.com/wmdataphys/GYM4DetectorDesign/tree/" + GYM4DETECTOR_COMMIT
)
_ETA = (-3.4, -2.0, -1.0, 1.0, 2.0, 3.4)
_MOMENTUM = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
_PARAMETER_COLUMNS = (
    "etarange",
    "prange",
    "dp_p_p",
    "error_dp_p_p",
    "dth_th_p",
    "error_dth_th_p",
    "dph_ph_p",
    "error_dph_ph_p",
    "dca2d",
    "error_dca2d",
    "dca2d_v_pT",
    "error_dca2d_v_pT",
    "InEfficiency",
    "error_InEfficiency",
    "KF_InEfficiency",
    "error_KF_InEfficiency",
    "GlobalKFInEff",
    "error_GlobalKFInEff",
    "EventsInBin",
    "A1",
    "A2",
    "A1A2",
    "Integral",
    "PseudoGlobalKFInEff",
    "error_PseudoGlobalKFInEff",
    "OverFlowUnderFlowFlag",
)
_FIT_COLUMNS = (
    "etarange",
    "prange",
    "Chi2_dpp",
    "NDF_dpp",
    "Chi2_dth",
    "NDF_dth",
    "Chi2_dph",
    "NDF_dph",
    "Chi2_dca2d",
    "NDF_dca2d",
)
_CORRECTIONS = (
    (
        "G4_Barrel_EIC.C",
        "void BarrelSetup(PHG4Reco* g4Reco)",
        "double BarrelSetup(PHG4Reco* g4Reco)",
        1,
        "Match the radius value returned by the upstream implementation.",
    ),
    (
        "G4_Barrel_EIC.C",
        "pitch / 10000. / sqrt(12.)",
        "pitch / sqrt(12.)",
        2,
        "Do not convert pitch from micrometres to centimetres twice.",
    ),
    (
        "G4_FST_EIC.C",
        'Form("SI_L%i_THICKNESS", j + 1)]*Units::um',
        'Form("SI_L%i_THICKNESS", j + 1)] * 9.37 / 100.',
        1,
        "Use the barrel percent-X0 to centimetre conversion in disk interpolation.",
    ),
    (
        "G4_FST_EIC.C",
        'Form("SI_L%i_THICKNESS", j)]*Units::um',
        'Form("SI_L%i_THICKNESS", j)] * 9.37 / 100.',
        1,
        "Use the barrel percent-X0 to centimetre conversion in disk interpolation.",
    ),
    (
        "G4_TrackingSupport.C",
        'Form("SI_L%i_THICKNESS", ilyr)] * Units::um',
        'Form("SI_L%i_THICKNESS", ilyr)] * 9.37 / 100.',
        1,
        "Place support outside the actual barrel silicon thickness.",
    ),
)


def read_gym_detector_config(data: bytes) -> tuple[tuple[str, float], ...]:
    """Read upstream ``NAME : number`` files without evaluating source text."""
    _host_only()
    if not isinstance(data, bytes):
        raise TypeError("Detector configuration must be exact bytes.")
    values: dict[str, float] = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*([^#]+)", line.strip())
        if match is None or match[1] in values:
            raise ValueError("Malformed or duplicate detector configuration entry.")
        value = float(match[2])
        if not math.isfinite(value):
            raise ValueError("Detector configuration values must be finite.")
        values[match[1]] = value
    if not values:
        raise ValueError("Detector configuration is empty.")
    return tuple(values.items())


def _configuration(values: tuple[tuple[str, float], ...]) -> bytes:
    if not isinstance(values, tuple) or len(dict(values)) != len(values):
        raise ValueError("Detector configuration must contain unique ordered keys.")
    data = "".join(f"{key} : {value:.17g}\n" for key, value in values).encode()
    read_gym_detector_config(data)
    return data


@dataclass(frozen=True, slots=True)
class GYMDetectorProfile:
    """Pinned pi-minus generation and reconstruction qualification profile.

    The upstream analysis fixes eta bins to [-3.4, 3.4] and momentum bins to
    [1, 20] GeV/c. ``events`` is a Fun4All event count; each event produces the
    settings' ``N_PER_EVENT`` tracks. Fit errors are upstream chi-square-scaled
    fit diagnostics, while Kalman errors are binomial estimates.
    """

    settings: tuple[tuple[str, float], ...]
    events: int
    seed: int
    runtime_build_id: str
    geometry_profile: Literal["corrected-silicon-units"]
    minimum_tracks_per_bin: int = 20
    maximum_reduced_chi_squared: float = 10.0

    def __post_init__(self) -> None:
        _host_only()
        if self.geometry_profile != "corrected-silicon-units":
            raise ValueError(
                "Select the explicit corrected-silicon-units geometry profile."
            )
        _configuration(self.settings)
        values = dict(self.settings)
        required = {
            "NLAYERS_SI_BAR",
            "NLAYERS_SI_EDISK",
            "NLAYERS_SI_HDISK",
            "MIN_RADIUS",
            "MAX_RADIUS",
            "MIN_HZ",
            "MAX_HZ",
            "MIN_EZ",
            "MAX_EZ",
            "MAX_VTX_SPRT_RADIUS",
            "MAX_SAG_SPRT_RADIUS",
            "AVG_SUPRT_THICKNESS",
            "PDG",
            "N_PER_EVENT",
            "VERTEX_DISTRIBUTION_WIDTH",
        }
        if not required <= values.keys():
            raise ValueError(
                f"Missing detector settings: {sorted(required - values.keys())}"
            )
        if values["PDG"] != -211:
            raise ValueError(
                "The pinned analysis_resolution profile is pi-minus (PDG -211)."
            )
        for key in (
            "NLAYERS_SI_BAR",
            "NLAYERS_SI_EDISK",
            "NLAYERS_SI_HDISK",
            "N_PER_EVENT",
        ):
            if values[key] < 1 or not float(values[key]).is_integer():
                raise ValueError(f"{key} must be a positive integer.")
        if values["NLAYERS_SI_BAR"] < 2:
            raise ValueError(
                "The pinned tracking-support interpolation requires two barrel layers."
            )
        if (
            type(self.events) is not int
            or self.events < 1
            or type(self.seed) is not int
            or not 0 < self.seed < 2**31
        ):
            raise ValueError("Positive event count and a 31-bit seed are required.")
        if (
            type(self.minimum_tracks_per_bin) is not int
            or self.minimum_tracks_per_bin < 2
        ):
            raise ValueError("At least two reconstructed tracks per bin are required.")
        if (
            not math.isfinite(self.maximum_reduced_chi_squared)
            or self.maximum_reduced_chi_squared <= 0
        ):
            raise ValueError("maximum_reduced_chi_squared must be positive finite.")
        if (
            not isinstance(self.runtime_build_id, str)
            or not self.runtime_build_id.strip()
        ):
            raise ValueError(
                "Declare the Fun4All/Geant4 build or container content identity."
            )
        if (
            values["VERTEX_DISTRIBUTION_WIDTH"] < 0
            or not float(values["VERTEX_DISTRIBUTION_WIDTH"]).is_integer()
        ):
            raise ValueError(
                "Upstream truncates vertex width to integer cm; supply an integer."
            )
        if not (
            0 < values["MIN_RADIUS"] < values["MAX_RADIUS"]
            and values["MAX_EZ"] < values["MIN_EZ"] < 0
            and 0 < values["MIN_HZ"] < values["MAX_HZ"]
            and 0
            < values["MAX_VTX_SPRT_RADIUS"]
            < values["MAX_SAG_SPRT_RADIUS"]
            < values["MAX_RADIUS"]
            and values["AVG_SUPRT_THICKNESS"] > 0
        ):
            raise ValueError(
                "Detector setting extents/support radii are physically inconsistent."
            )
        total_generated = self.events * int(values["N_PER_EVENT"])
        if total_generated < self.minimum_tracks_per_bin * 50:
            raise ValueError(
                "Requested event population cannot cover all reconstruction bins."
            )


def _source_identity(files: Mapping[str, bytes]) -> tuple[dict[str, str], str]:
    identities = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    identity = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identities, identity


def _correct_gym_sources(
    inputs: dict[str, bytes],
) -> tuple[dict[str, bytes], tuple[dict, ...]]:
    """Apply exact-count corrections to detached bytes and retain every stage."""
    staged, records = dict(inputs), []
    for stage, (path, before, after, count, reason) in enumerate(_CORRECTIONS, start=1):
        if path not in staged:
            raise ValueError(f"Corrected geometry source is absent from the pin: {path}")
        original = staged[path]
        text = original.decode("utf-8")
        if text.count(before) != count:
            raise ValueError(
                f"Corrected geometry source context disagrees with its pin: {path}"
            )
        corrected = text.replace(before, after).encode()
        records.append(
            {
                "stage": stage,
                "path": path,
                "before": before,
                "after": after,
                "count": count,
                "reason": reason,
                "before_sha256": hashlib.sha256(original).hexdigest(),
                "after_sha256": hashlib.sha256(corrected).hexdigest(),
            }
        )
        staged[path] = corrected
    for path, before, _, _, _ in _CORRECTIONS:
        if before.encode() in staged[path]:
            raise ValueError(
                f"Known-bad detector geometry remained after correction: {path}"
            )
    return staged, tuple(records)


@dataclass(frozen=True, slots=True)
class DetectorResolutionBin:
    eta_range: tuple[float, float]
    momentum_range_gev: tuple[float, float]
    momentum_resolution_percent: float
    momentum_fit_error_percent: float
    kalman_inefficiency: float
    kalman_binomial_error: float
    generated_tracks: int
    estimated_reconstructed_tracks: float
    fit_chi_squared: float
    fit_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class GYMDetectorDesignResult:
    bins: tuple[DetectorResolutionBin, ...]
    momentum_resolution_percent: float
    propagated_fit_error_percent: float
    kalman_inefficiency: float
    kalman_binomial_error: float
    upstream_inverse_thickness_diagnostic: float
    elapsed_seconds: float
    artifact: ScientificArtifactEnvelope
    simulation: EnergyRunResult
    reconstruction: EnergyRunResult
    accepted: bool
    qualification_failures: tuple[str, ...]


def _range(text: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([-+\d.eE]+)\s+-\s+([-+\d.eE]+)\s*", text)
    if match is None:
        raise ValueError("Malformed upstream reconstruction bin label.")
    lower, upper = float(match[1]), float(match[2])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Malformed upstream reconstruction bin range.")
    return lower, upper


def _rows(data: bytes, columns: tuple[str, ...]) -> dict:
    if not isinstance(data, bytes):
        raise TypeError("Reconstruction CSV must be exact bytes.")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    if reader.fieldnames != list(columns):
        raise ValueError(
            "Reconstruction columns disagree with the pinned upstream macro."
        )
    result = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Ragged reconstruction CSV row.")
        key = (_range(row["etarange"]), _range(row["prange"]))
        if key in result:
            raise ValueError("Duplicate reconstruction bin.")
        result[key] = row
    return result


def read_gym_detector_metrics(
    parameters_csv: bytes,
    fit_csv: bytes,
    *,
    minimum_tracks_per_bin: int = 20,
    maximum_reduced_chi_squared: float = 10.0,
) -> tuple[DetectorResolutionBin, ...]:
    """Read and qualify the actual pinned ``analysis_resolution.C`` outputs.

    Missing bins, overflow, failed/nonfinite fits, insufficient reconstructed
    tracks, inconsistent binomial errors, and excessive reduced chi-square are
    failures. Nothing is clipped, imputed, or converted to an optimizer penalty.
    """
    _host_only()
    if type(minimum_tracks_per_bin) is not int or minimum_tracks_per_bin < 2:
        raise ValueError("minimum_tracks_per_bin must be at least two.")
    if not math.isfinite(maximum_reduced_chi_squared) or maximum_reduced_chi_squared <= 0:
        raise ValueError("maximum_reduced_chi_squared must be positive finite.")
    rows = _rows(parameters_csv, _PARAMETER_COLUMNS)
    fits = _rows(fit_csv, _FIT_COLUMNS)
    expected = {
        ((eta_lower, eta_upper), (p_lower, p_upper))
        for eta_lower, eta_upper in zip(_ETA[:-1], _ETA[1:], strict=True)
        for p_lower, p_upper in zip(_MOMENTUM[:-1], _MOMENTUM[1:], strict=True)
    }
    if rows.keys() != expected or fits.keys() != expected:
        raise ValueError(
            "Reconstruction must contain every declared eta/momentum bin exactly once."
        )
    bins = []
    for key in sorted(expected):
        row, fit = rows[key], fits[key]
        names = (
            "dp_p_p",
            "error_dp_p_p",
            "KF_InEfficiency",
            "error_KF_InEfficiency",
            "EventsInBin",
            "OverFlowUnderFlowFlag",
        )
        measured = [float(row[name]) for name in names]
        chi, ndf = float(fit["Chi2_dpp"]), float(fit["NDF_dpp"])
        if not all(math.isfinite(value) for value in (*measured, chi, ndf)):
            raise ValueError("Nonfinite measured detector result or fit error.")
        width, fit_error, inefficiency, inefficiency_error, generated, overflow = measured
        if overflow != 0:
            raise ValueError("Upstream reported momentum histogram overflow/underflow.")
        if width <= 0 or fit_error <= 0 or chi < 0 or ndf <= 0 or not ndf.is_integer():
            raise ValueError("Momentum reconstruction fit is not qualified.")
        if chi / ndf > maximum_reduced_chi_squared:
            raise ValueError(
                "Momentum reconstruction reduced chi-square exceeds its qualification limit."
            )
        if generated < 1 or not float(generated).is_integer():
            raise ValueError("Invalid generated-track count in a reconstruction bin.")
        if not 0 <= inefficiency <= 1 or inefficiency_error < 0:
            raise ValueError("Invalid Kalman reconstruction inefficiency/error.")
        expected_error = math.sqrt(inefficiency * (1 - inefficiency) / generated)
        if not math.isclose(
            inefficiency_error, expected_error, rel_tol=5e-5, abs_tol=5e-8
        ):
            raise ValueError("Kalman binomial error disagrees with its measured counts.")
        reconstructed = generated * (1 - inefficiency)
        if reconstructed < minimum_tracks_per_bin:
            raise ValueError("Insufficient reconstructed tracks in a reconstruction bin.")
        bins.append(
            DetectorResolutionBin(
                key[0],
                key[1],
                width,
                fit_error,
                inefficiency,
                inefficiency_error,
                int(generated),
                reconstructed,
                chi,
                int(ndf),
            )
        )
    return tuple(bins)


def _design_parameters(
    design: tuple[tuple[str, float], ...], profile: GYMDetectorProfile
) -> bytes:
    data = _configuration(design)
    values, settings = dict(design), dict(profile.settings)
    barrel_count = int(settings["NLAYERS_SI_BAR"])
    electron_count = int(settings["NLAYERS_SI_EDISK"])
    hadron_count = int(settings["NLAYERS_SI_HDISK"])
    expected = {"B_FIELD"}
    for prefix, count, suffixes in (
        ("SI_L", barrel_count, ("RADIUS", "PITCH", "THICKNESS", "E_LENGTH", "H_LENGTH")),
        ("SI_ED", electron_count, ("Z", "THICKNESS", "PITCH")),
        ("SI_HD", hadron_count, ("Z", "THICKNESS", "PITCH")),
    ):
        for index in range(1, count + 1):
            for suffix in suffixes:
                expected.add(f"{prefix}{index}_{suffix}")
    if values.keys() != expected:
        raise ValueError(
            "Detector geometry keys disagree with layer counts: "
            f"missing {sorted(expected - values.keys())}; extra {sorted(values.keys() - expected)}"
        )
    for key, value in values.items():
        if key.startswith("SI_ED") and key.endswith("_Z"):
            if value >= 0:
                raise ValueError("Electron disk positions must be negative cm.")
        elif value <= 0:
            raise ValueError(f"Positive native-unit detector parameter required: {key}")

    radii = [values[f"SI_L{index}_RADIUS"] for index in range(1, barrel_count + 1)]
    electron_lengths = [
        values[f"SI_L{index}_E_LENGTH"] for index in range(1, barrel_count + 1)
    ]
    hadron_lengths = [
        values[f"SI_L{index}_H_LENGTH"] for index in range(1, barrel_count + 1)
    ]
    if any(lower >= upper for lower, upper in zip(radii[:-1], radii[1:], strict=True)):
        raise ValueError("Barrel radii must increase strictly.")
    for lengths in (electron_lengths, hadron_lengths):
        if any(
            lower > upper for lower, upper in zip(lengths[:-1], lengths[1:], strict=True)
        ):
            raise ValueError(
                "Barrel side lengths must be nondecreasing for support interpolation."
            )
    if radii[0] < settings["MIN_RADIUS"] or radii[-1] > settings["MAX_RADIUS"]:
        raise ValueError("Barrel radii exceed the declared detector extent.")
    if electron_lengths[0] < abs(settings["MIN_EZ"]) or electron_lengths[-1] > abs(
        settings["MAX_EZ"]
    ):
        raise ValueError(
            "Electron-side barrel lengths exceed the declared detector extent."
        )
    if hadron_lengths[0] < settings["MIN_HZ"] or hadron_lengths[-1] > settings["MAX_HZ"]:
        raise ValueError(
            "Hadron-side barrel lengths exceed the declared detector extent."
        )
    for index, radius in enumerate(radii, start=1):
        thickness_cm = values[f"SI_L{index}_THICKNESS"] * 9.37 / 100
        if radius + thickness_cm > settings["MAX_RADIUS"]:
            raise ValueError("Barrel silicon outer radius exceeds the detector extent.")
    nearest_support_layer = min(
        range(barrel_count),
        key=lambda index: abs(radii[index] - settings["MAX_VTX_SPRT_RADIUS"]),
    )
    if nearest_support_layer == barrel_count - 1:
        raise ValueError(
            "Tracking support interpolation requires a barrel layer outside the vertex support."
        )

    electron_z = [values[f"SI_ED{index}_Z"] for index in range(1, electron_count + 1)]
    hadron_z = [values[f"SI_HD{index}_Z"] for index in range(1, hadron_count + 1)]
    if any(
        lower <= upper
        for lower, upper in zip(electron_z[:-1], electron_z[1:], strict=True)
    ):
        raise ValueError(
            "Electron disk positions must move strictly away from the interaction point."
        )
    if any(
        lower >= upper for lower, upper in zip(hadron_z[:-1], hadron_z[1:], strict=True)
    ):
        raise ValueError(
            "Hadron disk positions must move strictly away from the interaction point."
        )
    if not all(settings["MAX_EZ"] <= value <= settings["MIN_EZ"] for value in electron_z):
        raise ValueError("Electron disk position exceeds the declared detector extent.")
    if not all(settings["MIN_HZ"] <= value <= settings["MAX_HZ"] for value in hadron_z):
        raise ValueError("Hadron disk position exceeds the declared detector extent.")
    for positions, lengths in (
        (electron_z, electron_lengths),
        (hadron_z, hadron_lengths),
    ):
        for position in positions:
            distance = abs(position)
            if distance <= lengths[-1] and not any(
                lower <= distance <= upper and lower < upper
                for lower, upper in zip(lengths[:-1], lengths[1:], strict=True)
            ):
                raise ValueError(
                    "Disk position falls outside upstream support interpolation intervals."
                )
    return data


def _engine_evidence(data: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError("Malformed or duplicate Geant4 engine evidence.")
        values[key] = value
    required = {"ROOT", "GEANT4", "RANDOMSEED", "LOADED_LIBRARIES"}
    if values.keys() != required or not values["ROOT"] or not values["GEANT4"]:
        raise ValueError("Incomplete Geant4 engine evidence.")
    return values


def run_gym_detector_design(
    root: PinnedExecutable,
    source: DeviceSource,
    profile: GYMDetectorProfile,
    design: tuple[tuple[str, float], ...],
    *,
    field_map: bytes,
    timeout: float = 3600,
    max_output_bytes: int = 256 * 1024 * 1024,
    environment: Mapping[str, str] | None = None,
) -> GYMDetectorDesignResult:
    """Run real Fun4All/Geant4 tracking and pinned ROOT reconstruction.

    The field map and all selected upstream source bytes are detached and
    digested. Runtime DLLs, Geant4 data, and calibration resources are not
    transitively pinned by the ROOT executable; ``runtime_build_id`` is the
    caller's exact environment/container identity and loaded-library paths are
    retained as observed evidence. The upstream inverse-thickness expression is
    reported only as a mixed-unit diagnostic and is never an accepted cost.
    """
    _host_only(design, field_map)
    if source.commit != GYM4DETECTOR_COMMIT:
        raise ValueError("Unsupported GYM4DetectorDesign source commit.")
    if not isinstance(root, PinnedExecutable):
        raise TypeError("root must be a PinnedExecutable.")
    if not isinstance(field_map, bytes) or not field_map.startswith(b"root"):
        raise ValueError("Supply exact sPHENIX ROOT field-map bytes.")
    if (
        type(max_output_bytes) is not int
        or max_output_bytes < 1
        or len(field_map) >= max_output_bytes
    ):
        raise ValueError("Field map and artifacts require a positive finite byte bound.")
    geometry = _design_parameters(design, profile)
    snapshot = source.snapshot(["AllMacros"], max_bytes=max_output_bytes - len(field_map))
    original_inputs = {
        name.removeprefix("AllMacros/"): data for name, data in snapshot.items()
    }
    original_identities, original_source_identity = _source_identity(original_inputs)
    corrected_inputs, corrections = _correct_gym_sources(original_inputs)
    corrected_identities, corrected_source_identity = _source_identity(corrected_inputs)
    metadata: dict[str, Any] = {
        "profile": asdict(profile),
        "source_commit": source.commit,
        "source_license_id": source.license_id,
        "source_repository_url": GYM4DETECTOR_SOURCE_URL,
        "original_source_files_sha256": original_identities,
        "original_source_tree_identity": original_source_identity,
        "corrected_source_files_sha256": corrected_identities,
        "corrected_source_tree_identity": corrected_source_identity,
        "source_corrections": corrections,
        "root_executable": asdict(root),
        "runtime_build_id": profile.runtime_build_id,
        "geometry_sha256": hashlib.sha256(geometry).hexdigest(),
        "field_map_sha256": hashlib.sha256(field_map).hexdigest(),
        "geometry_semantics": (
            "position/radius/barrel lengths = cm",
            "pitch = micrometres; tracking sigma = pitch_cm / sqrt(12)",
            "barrel thickness = percent radiation length; silicon X0 = 9.37 cm",
            "disk silicon thickness = micrometres",
        ),
        "accepted_objectives": ("momentum_resolution_percent", "kalman_inefficiency"),
        "excluded_costs": ("upstream_inverse_thickness_diagnostic",),
        "upstream_caveats": (
            "inverse-thickness diagnostic adds reciprocal values with unlike native units",
            "fit error is the upstream chi-square-scaled fitted-width error",
            "runtime shared libraries/data are identified by caller build id and observed paths",
        ),
    }
    inputs = dict(corrected_inputs)
    inputs.update(
        {
            "design.config": geometry,
            "settings.config": _configuration(profile.settings),
            "sPHENIX.2d.root": field_map,
            "request.json": canonical_json(metadata).encode(),
        }
    )
    driver = f"""#include "Fun4All_G4_EICDetector.C"
#include <G4Version.hh>
#include <TROOT.h>
#include <TSystem.h>
#include <fstream>
void phydrax_detector() {{
  std::ofstream evidence("engine-evidence.txt");
  evidence << "ROOT=" << gROOT->GetVersion() << "\\n";
  evidence << "GEANT4=" << G4Version << "\\n";
  evidence << "RANDOMSEED={profile.seed}\\n";
  evidence << "LOADED_LIBRARIES=" << gSystem->GetLibraries() << "\\n";
  evidence.close();
  recoConsts::instance()->set_IntFlag("RANDOMSEED", {profile.seed});
  Enable::OVERLAPCHECK = true;
  G4MAGNET::magfield = "sPHENIX.2d.root";
  Fun4All_G4_EICDetector("design.config", "settings.config", {profile.events},
                        -3.4, 3.4, 1.0, 20.0, "detector.root", false);
}}
"""
    inputs["phydrax_detector.C"] = driver.encode()
    env = dict(environment or {})
    env["ROOT_INCLUDE_PATH"] = "." + (
        os.pathsep + env["ROOT_INCLUDE_PATH"] if "ROOT_INCLUDE_PATH" in env else ""
    )
    try:
        simulation = run_energy_command(
            root,
            ("-l", "-b", "-q", "phydrax_detector.C"),
            inputs=inputs,
            outputs=("detector_g4tracking_eval.root", "engine-evidence.txt"),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            environment=env,
        )
    except EnergyRuntimeError as failure:
        runs = () if failure.result is None else (failure.result,)
        raise DeviceQualificationError(str(failure), runs=runs) from failure
    try:
        evidence = _engine_evidence(simulation.output("engine-evidence.txt"))
        if evidence["ROOT"] != root.version or evidence["RANDOMSEED"] != str(
            profile.seed
        ):
            raise ValueError(
                "Observed ROOT release/seed differs from the run declaration."
            )
        tracking = simulation.output("detector_g4tracking_eval.root")
        if not tracking.startswith(b"root"):
            raise ValueError("Fun4All tracking output is not a ROOT file.")
        logs = (simulation.stdout + simulation.stderr).lower()
        overlap_markers = (
            b"geomvol1002",
            b"overlap is detected",
            b"overlapping daughter",
        )
        if any(marker in logs for marker in overlap_markers):
            raise ValueError("Geant4 reported overlapping detector geometry.")
        if b"all done" not in logs:
            raise ValueError("Fun4All did not report normal event-loop completion.")
    except (ValueError, KeyError, UnicodeError) as failure:
        raise DeviceQualificationError(str(failure), runs=(simulation,)) from failure

    analysis_inputs = {
        "analysis_resolution.C": corrected_inputs["analysis_resolution.C"],
        "tracks.root": tracking,
        "request.json": inputs["request.json"],
        "simulation-evidence.txt": simulation.output("engine-evidence.txt"),
    }
    field_strength = dict(design)["B_FIELD"]
    try:
        reconstruction = run_energy_command(
            root,
            (
                "-l",
                "-b",
                "-q",
                f'analysis_resolution.C("tracks.root",false,{field_strength:.17g})',
            ),
            inputs=analysis_inputs,
            outputs=("params.csv", "Chi2NDF_DoubleGaus.csv"),
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            environment=env,
        )
    except EnergyRuntimeError as failure:
        runs = (simulation,) if failure.result is None else (simulation, failure.result)
        raise DeviceQualificationError(str(failure), runs=runs) from failure
    try:
        bins = read_gym_detector_metrics(
            reconstruction.output("params.csv"),
            reconstruction.output("Chi2NDF_DoubleGaus.csv"),
            minimum_tracks_per_bin=profile.minimum_tracks_per_bin,
            maximum_reduced_chi_squared=profile.maximum_reduced_chi_squared,
        )
    except (ValueError, KeyError, TypeError, UnicodeError) as failure:
        raise DeviceQualificationError(
            str(failure), runs=(simulation, reconstruction)
        ) from failure

    eta_means, eta_variances = [], []
    for eta_range in sorted({item.eta_range for item in bins}):
        selected = [item for item in bins if item.eta_range == eta_range]
        weights = [1 / item.momentum_fit_error_percent**2 for item in selected]
        eta_means.append(
            sum(
                weight * item.momentum_resolution_percent
                for weight, item in zip(weights, selected, strict=True)
            )
            / sum(weights)
        )
        eta_variances.append(1 / sum(weights))
    momentum_mean = sum(eta_means) / len(eta_means)
    momentum_error = math.sqrt(sum(eta_variances)) / len(eta_variances)
    generated_tracks = sum(item.generated_tracks for item in bins)
    kalman_inefficiency = (
        sum(item.kalman_inefficiency * item.generated_tracks for item in bins)
        / generated_tracks
    )
    kalman_error = math.sqrt(
        kalman_inefficiency * (1 - kalman_inefficiency) / generated_tracks
    )
    inverse_thickness = sum(
        1 / value for key, value in design if key.endswith("_THICKNESS")
    )
    failures: tuple[str, ...] = ()
    metadata.update(
        {
            "engine_evidence": evidence,
            "bins": [asdict(item) for item in bins],
            "momentum_resolution_percent": momentum_mean,
            "propagated_fit_error_percent": momentum_error,
            "kalman_inefficiency": kalman_inefficiency,
            "kalman_binomial_error": kalman_error,
            "upstream_inverse_thickness_diagnostic": inverse_thickness,
            "fit_error_assumption": "independent disjoint-bin upstream fitted-width errors",
            "kalman_error_assumption": "aggregate independent Bernoulli reconstruction outcomes",
            "accepted": True,
            "qualification_failures": failures,
        }
    )
    return GYMDetectorDesignResult(
        bins,
        momentum_mean,
        momentum_error,
        kalman_inefficiency,
        kalman_error,
        inverse_thickness,
        simulation.elapsed_seconds + reconstruction.elapsed_seconds,
        device_artifact(
            "geant4-detector-design", metadata, source, (simulation, reconstruction)
        ),
        simulation,
        reconstruction,
        True,
        failures,
    )
