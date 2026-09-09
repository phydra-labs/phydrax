#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only HFSS eigenmode/EPR and Q3D targets for one pinned tutorial.

The supported runtime is Windows, Python 3.11/3.12, AEDT 2021 R2 or 2022 R2,
and the exact dependency versions in :data:`HFSS_DEPENDENCY_PROFILE`. The
adapter runs QDesignOptimizer's coupled-transmon geometry through HFSS
Eigenmode, pyEPR, and Q3D. It never substitutes a driven S-parameter model.
Source: https://github.com/202Q-lab/QDesignOptimizer/tree/d4f6ada5ada59b786df1006d53f8f148b364364e
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Literal

from .._fingerprint import canonical_json
from ..artifacts import ScientificArtifactEnvelope
from ._device_design import device_artifact, DeviceQualificationError, DeviceSource
from .energy_runtime import (
    _host_only,
    EnergyRunResult,
    PinnedExecutable,
    run_energy_command,
)


QDESIGNOPTIMIZER_COMMIT = "d4f6ada5ada59b786df1006d53f8f148b364364e"
HFSS_DEPENDENCY_PROFILE = (
    ("gdstk", "0.9.62"),
    ("geopandas", "1.1.2"),
    ("gmsh", "4.11.1"),
    ("ipykernel", "7.1.0"),
    ("ipython", "9.10.0"),
    ("matplotlib", "3.10.8"),
    ("numpy", "1.26.4"),
    ("pandas", "2.3.3"),
    ("pint", "0.24.4"),
    ("psutil", "6.1.1"),
    ("pyaedt", "0.23.0"),
    ("pyepr-quantum", "1.0.0"),
    ("pygments", "2.14.0"),
    ("pyside6", "6.10.2"),
    ("pywin32", "308"),
    ("pyyaml", "6.0.3"),
    ("qdarkstyle", "3.1"),
    ("quantum-metal", "0.7.4"),
    ("qutip", "5.2.3"),
    ("scipy", "1.15.3"),
    ("scqubits", "4.3.1"),
    ("shapely", "2.0.7"),
)
_TUTORIAL = "tutorials/examples_coupled_transmon_chip"
_GROUP_MODES = {1: ("qubit_1", "resonator_1"), 2: ("qubit_2", "resonator_2")}
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class HFSSDesignTarget:
    """Named physical target; capacitance values retain their signed fF value."""

    name: str
    quantity: Literal[
        "frequency_hz", "kappa_hz", "chi_hz", "participation", "capacitance_ff"
    ]
    labels: tuple[str, ...]
    value: float
    scale: float

    def __post_init__(self) -> None:
        _host_only(self.value, self.scale)
        arity = {
            "frequency_hz": 1,
            "kappa_hz": 1,
            "chi_hz": 2,
            "participation": 2,
            "capacitance_ff": 2,
        }
        if self.quantity not in arity or len(self.labels) != arity[self.quantity]:
            raise ValueError("Target quantity and label arity disagree.")
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(not isinstance(label, str) or not label for label in self.labels)
        ):
            raise ValueError("Targets require nonempty names and physical labels.")
        if (
            not math.isfinite(self.value)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Target value must be finite and scale positive finite.")


@dataclass(frozen=True, slots=True)
class HFSSDesignProfile:
    """One pinned qubit/readout-resonator group and Q3D extraction.

    ``mode_windows_hz`` qualifies the upstream frequency-rank labels. This is
    not field-overlap tracking: a mode that crosses or leaves its disjoint owner
    window is rejected. Mesh identity covers exported adaptive histories and
    mesh-statistics reports, not element connectivity.
    """

    group: int
    mode_windows_hz: tuple[tuple[str, float, float], ...]
    targets: tuple[HFSSDesignTarget, ...]
    max_passes: int = 12
    frequency_tolerance_percent: float = 0.03
    capacitance_tolerance_percent: float = 0.5

    def __post_init__(self) -> None:
        _host_only()
        if (
            self.group not in _GROUP_MODES
            or type(self.max_passes) is not int
            or self.max_passes < 2
        ):
            raise ValueError("Choose group 1/2 and at least two adaptive passes.")
        if (
            len(self.mode_windows_hz) != 2
            or len({window[0] for window in self.mode_windows_hz}) != 2
        ):
            raise ValueError("Supply exactly two distinct physical mode windows.")
        if {window[0] for window in self.mode_windows_hz} != set(
            _GROUP_MODES[self.group]
        ):
            raise ValueError("Mode windows must name the pinned tutorial group modes.")
        ordered = sorted(self.mode_windows_hz, key=lambda window: window[1])
        for name, lower, upper in ordered:
            if (
                not isinstance(name, str)
                or not name
                or not (
                    math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper
                )
            ):
                raise ValueError("Mode windows must have positive finite ordered bounds.")
        if ordered[0][2] >= ordered[1][1]:
            raise ValueError(
                "Mode windows must be disjoint; ambiguous identities are rejected."
            )
        if not self.targets or len({target.name for target in self.targets}) != len(
            self.targets
        ):
            raise ValueError("Supply uniquely named targets.")
        modes = set(_GROUP_MODES[self.group])
        for target in self.targets:
            if not isinstance(target, HFSSDesignTarget):
                raise TypeError("targets must contain HFSSDesignTarget values.")
            mode_labels = (
                target.labels if target.quantity == "chi_hz" else target.labels[:1]
            )
            if (
                target.quantity in ("frequency_hz", "kappa_hz", "chi_hz", "participation")
                and not set(mode_labels) <= modes
            ):
                raise ValueError(
                    "Target mode labels must belong to the selected tutorial group."
                )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.frequency_tolerance_percent,
                self.capacitance_tolerance_percent,
            )
        ):
            raise ValueError(
                "Convergence tolerances must be positive finite percentages."
            )


@dataclass(frozen=True, slots=True)
class HFSSDesignResult:
    values: tuple[float, ...]
    normalized_residuals: tuple[float, ...]
    mode_labels: tuple[str, ...]
    mesh_identity: str
    artifact: ScientificArtifactEnvelope
    run: EnergyRunResult
    # Adaptive convergence is numerical evidence, not statistical uncertainty.
    uncertainty: None = None
    accepted: bool = True


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {label}.")
    return number


def _table(table: object, label: str) -> tuple[list[str], list[str], list[list[object]]]:
    if not isinstance(table, dict) or set(table) != {"index", "columns", "data"}:
        raise ValueError(f"Malformed {label} table.")
    index, columns, data = table["index"], table["columns"], table["data"]
    if (
        not isinstance(index, list)
        or not isinstance(columns, list)
        or not isinstance(data, list)
    ):
        raise ValueError(f"Malformed {label} table.")
    if (
        len(index) != len(data)
        or len(index) != len(set(index))
        or len(columns) != len(set(columns))
    ):
        raise ValueError(f"Duplicate or inconsistent {label} table labels.")
    if any(not isinstance(item, str) or not item for item in (*index, *columns)):
        raise ValueError(f"Empty {label} table label.")
    if any(not isinstance(row, list) or len(row) != len(columns) for row in data):
        raise ValueError(f"Ragged {label} table.")
    return index, columns, data


def _matrix_value(table: dict, row: str, column: str) -> float:
    index, columns, data = _table(table, "capacitance")
    return _finite(data[index.index(row)][columns.index(column)], "capacitance target")


def _convergence(table: object, tolerance: float) -> None:
    _, columns, data = _table(table, "adaptive convergence")
    if len(data) < 2:
        raise ValueError("At least two observed adaptive passes are required.")
    indices = [
        index
        for index, name in enumerate(columns)
        if "delta" in name.lower() and "%" in name
    ]
    if not indices:
        raise ValueError("No recognized measured percentage convergence column.")
    for index in indices:
        value = data[-1][index]
        if (
            value is None
            or abs(_finite(value, "adaptive convergence result")) > tolerance
        ):
            raise ValueError("Adaptive mesh convergence tolerance was not achieved.")


def _validate_capacitance_table(table: object) -> None:
    index, columns, data = _table(table, "capacitance")
    if len(index) != len(columns) or set(index) != set(columns):
        raise ValueError(
            "Capacitance matrix must be square with matching physical labels."
        )
    for row, name in enumerate(index):
        for column, other in enumerate(columns):
            value = _finite(data[row][column], "capacitance matrix value")
            reverse = _finite(
                data[index.index(other)][columns.index(name)], "capacitance matrix value"
            )
            if not math.isclose(value, reverse, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Capacitance matrix is not symmetric.")


def read_hfss_design_result(
    data: bytes, profile: HFSSDesignProfile
) -> tuple[tuple[float, ...], dict]:
    """Parse and qualify the real worker payload; fixture bytes are not a solve."""
    _host_only()
    if not isinstance(data, bytes):
        raise TypeError("HFSS result must be exact bytes.")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("HFSS worker output must be a JSON object.")
    if result["source_commit"] != QDESIGNOPTIMIZER_COMMIT:
        raise ValueError("HFSS result source identity mismatch.")
    if result["profile"] != json.loads(canonical_json(asdict(profile))):
        raise ValueError("HFSS result belongs to a different requested profile.")
    expected_repository = (
        "https://github.com/202Q-lab/QDesignOptimizer/tree/" + QDESIGNOPTIMIZER_COMMIT
    )
    if (
        not isinstance(result["source_license_id"], str)
        or not result["source_license_id"].strip()
        or result["source_repository_url"] != expected_repository
        or result["qdesignoptimizer_version"] != "0.2.0"
    ):
        raise ValueError("Incomplete QDesignOptimizer source provenance.")
    source_files = result["source_files_sha256"]
    if (
        not isinstance(source_files, dict)
        or not {"LICENSE.txt", "pyproject.toml", "poetry.lock"} <= source_files.keys()
        or not any(path.startswith("src/qdesignoptimizer/") for path in source_files)
        or not any(path.startswith(_TUTORIAL + "/") for path in source_files)
    ):
        raise ValueError("Missing exact QDesignOptimizer source identities.")
    for path, digest in source_files.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith(("/", "\\", ":"))
            or "\\" in path
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError("Unsafe source identity path in HFSS output.")
        _sha256(digest, "source file identity")
    expected_source_identity = hashlib.sha256(
        json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if result["source_tree_identity"] != expected_source_identity:
        raise ValueError("HFSS source tree identity mismatch.")
    if result["dependency_lock_sha256"] != source_files.get("poetry.lock"):
        raise ValueError("HFSS dependency lock identity mismatch.")
    _sha256(result["worker_sha256"], "worker identity")
    _sha256(result["engine_sha256"], "AEDT executable identity")
    _sha256(result["python_sha256"], "Python executable identity")
    if not isinstance(result["engine_version"], str) or not result["engine_version"]:
        raise ValueError("Missing observed AEDT release.")
    if not isinstance(result["python_version"], str) or not result["python_version"]:
        raise ValueError("Missing observed Python release.")
    for key in ("python_license_id", "engine_license_id"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError("Missing executable license provenance.")
    for key in ("python_source_url", "engine_source_url"):
        if not isinstance(result[key], str):
            raise ValueError("Malformed executable source provenance.")
    packages = result["packages"]
    if (
        not isinstance(packages, dict)
        or set(packages) != dict(HFSS_DEPENDENCY_PROFILE).keys()
    ):
        raise ValueError("HFSS dependency evidence does not match the pinned profile.")
    for package, version in HFSS_DEPENDENCY_PROFILE:
        evidence = packages[package]
        if not isinstance(evidence, dict) or evidence.get("version") != version:
            raise ValueError(f"HFSS dependency version mismatch: {package}.")
        _sha256(evidence.get("metadata_sha256"), f"{package} metadata identity")
        _sha256(evidence.get("record_sha256"), f"{package} installation identity")
        _sha256(evidence.get("content_sha256"), f"{package} installed-content identity")
        if (
            type(evidence.get("file_count")) is not int
            or evidence["file_count"] < 1
            or type(evidence.get("total_bytes")) is not int
            or evidence["total_bytes"] < 1
        ):
            raise ValueError(f"Incomplete installed-content evidence: {package}.")
        direct = evidence.get("direct_url_sha256")
        if direct is not None:
            _sha256(direct, f"{package} direct-url identity")
    if (
        result["mode_identity_scope"]
        != "upstream-target-frequency-rank-with-disjoint-owner-windows"
    ):
        raise ValueError("Unsupported HFSS mode identity evidence.")
    if (
        result["mesh_identity_scope"]
        != "exported-statistics-and-adaptive-history-not-connectivity"
    ):
        raise ValueError("Unsupported HFSS mesh identity evidence.")
    _convergence(result["eigenmode_convergence"], profile.frequency_tolerance_percent)
    _convergence(result["capacitance_convergence"], profile.capacitance_tolerance_percent)
    _validate_capacitance_table(result["capacitance_ff"])

    windows = {name: (lower, upper) for name, lower, upper in profile.mode_windows_hz}
    modes = result["mode_labels"]
    if (
        not isinstance(modes, list)
        or len(modes) != 2
        or len(set(modes)) != 2
        or set(modes) != set(windows)
    ):
        raise ValueError("Solved mode labels do not match requested identities.")
    eigenmode = result["eigenmode_frequency_hz"]
    epr = result["epr_frequency_hz"]
    kappa = result["eigenmode_kappa_hz"]
    if not all(
        isinstance(values, list) and len(values) == len(modes)
        for values in (eigenmode, epr, kappa)
    ):
        raise ValueError("Incomplete HFSS mode results.")
    for index, name in enumerate(modes):
        eigen_frequency = _finite(eigenmode[index], "eigenmode frequency")
        epr_frequency = _finite(epr[index], "EPR frequency")
        decay = _finite(kappa[index], "eigenmode linewidth")
        lower, upper = windows[name]
        if not lower <= eigen_frequency <= upper or not lower <= epr_frequency <= upper:
            raise ValueError(
                "Frequency-rank mode identity left its qualification window."
            )
        if decay < 0:
            raise ValueError("Negative eigenmode linewidth.")

    junctions = result["junction_labels"]
    chi, participation = result["chi_hz"], result["participation"]
    if (
        not isinstance(junctions, list)
        or not junctions
        or len(junctions) != len(set(junctions))
    ):
        raise ValueError("Missing or duplicate EPR junction labels.")
    if (
        not isinstance(chi, list)
        or len(chi) != len(modes)
        or any(not isinstance(row, list) or len(row) != len(modes) for row in chi)
    ):
        raise ValueError("Malformed EPR Kerr matrix.")
    if (
        not isinstance(participation, list)
        or len(participation) != len(modes)
        or any(
            not isinstance(row, list) or len(row) != len(junctions)
            for row in participation
        )
    ):
        raise ValueError("Malformed EPR participation matrix.")
    for row in range(len(modes)):
        for column in range(len(modes)):
            value = _finite(chi[row][column], "EPR Kerr value")
            reverse = _finite(chi[column][row], "EPR Kerr value")
            if not math.isclose(value, reverse, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("EPR Kerr matrix is not symmetric.")
        for value in participation[row]:
            ratio = _finite(value, "EPR participation ratio")
            if ratio < 0 or ratio > 1 + 1e-9:
                raise ValueError("EPR participation ratio is outside [0, 1].")

    values = []
    for target in profile.targets:
        if target.quantity == "frequency_hz":
            value = epr[modes.index(target.labels[0])]
        elif target.quantity == "kappa_hz":
            value = kappa[modes.index(target.labels[0])]
        elif target.quantity == "chi_hz":
            value = chi[modes.index(target.labels[0])][modes.index(target.labels[1])]
        elif target.quantity == "participation":
            value = participation[modes.index(target.labels[0])][
                junctions.index(target.labels[1])
            ]
        else:
            value = _matrix_value(result["capacitance_ff"], *target.labels)
        values.append(_finite(value, f"HFSS target {target.name}"))

    required_exports = {
        "request.json",
        "eigenmode-mesh.txt",
        "eigenmode-convergence.txt",
        "capacitance-mesh.txt",
        "capacitance-convergence.txt",
    }
    exports = result["mesh_exports"]
    if not isinstance(exports, dict) or set(exports) != required_exports:
        raise ValueError("Incomplete HFSS mesh/convergence artifact identity.")
    for digest in exports.values():
        _sha256(digest, "mesh evidence identity")
    expected_mesh_identity = hashlib.sha256(
        json.dumps(exports, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if result["mesh_identity"] != expected_mesh_identity:
        raise ValueError("Mesh evidence identity does not match its exported artifacts.")
    return tuple(values), result


def run_hfss_design(
    python: PinnedExecutable,
    aedt: PinnedExecutable,
    source: DeviceSource,
    profile: HFSSDesignProfile,
    design_variables: Mapping[str, str],
    *,
    timeout: float = 3600,
    max_output_bytes: int = 64 * 1024 * 1024,
    environment: Mapping[str, str] | None = None,
) -> HFSSDesignResult:
    """Run one real target evaluation on an exclusive licensed Windows host.

    Exact Python, AEDT, source, lockfile, worker, design, and installed package
    identities are retained. Dynamic AEDT DLLs remain part of the declared AEDT
    installation and are not implied to be content-pinned by its executable hash.
    Failures raise with engine evidence and never become numerical penalties.
    """
    _host_only(design_variables)
    if source.commit != QDESIGNOPTIMIZER_COMMIT:
        raise ValueError(
            "This profile requires the supported QDesignOptimizer source pin."
        )
    if not isinstance(python, PinnedExecutable) or not isinstance(aedt, PinnedExecutable):
        raise TypeError("python and aedt must be pinned executables.")
    if not isinstance(design_variables, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) or not key or not value
        for key, value in design_variables.items()
    ):
        raise ValueError(
            "Design variables require explicit nonempty unit-bearing strings."
        )
    sources = source.snapshot(
        [
            "LICENSE.txt",
            "pyproject.toml",
            "poetry.lock",
            "src/qdesignoptimizer",
            *[
                f"{_TUTORIAL}/{name}"
                for name in (
                    "design.py",
                    "names.py",
                    "mini_studies.py",
                    "parameter_targets.py",
                    "design_variables.json",
                )
            ],
        ],
        max_bytes=max_output_bytes,
    )
    source_identities = {
        name: hashlib.sha256(value).hexdigest() for name, value in sources.items()
    }
    worker = Path(__file__).with_name("_hfss_design_worker.py").read_bytes()
    request = {
        "profile": asdict(profile),
        "design_variables": dict(design_variables),
        "source_commit": source.commit,
        "source_license_id": source.license_id,
        "source_repository_url": (
            "https://github.com/202Q-lab/QDesignOptimizer/tree/" + QDESIGNOPTIMIZER_COMMIT
        ),
        "source_files_sha256": source_identities,
        "source_tree_identity": hashlib.sha256(
            json.dumps(source_identities, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "dependency_profile": dict(HFSS_DEPENDENCY_PROFILE),
        "python": asdict(python),
        "aedt": asdict(aedt),
    }
    inputs = {f"source/{name}": value for name, value in sources.items()}
    inputs["request.json"] = canonical_json(request).encode()
    inputs["worker.py"] = worker
    run = run_energy_command(
        python,
        ("worker.py",),
        inputs=inputs,
        outputs=(
            "result.json",
            "eigenmode-convergence.txt",
            "eigenmode-mesh.txt",
            "capacitance-convergence.txt",
            "capacitance-mesh.txt",
        ),
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        environment=environment,
    )
    try:
        values, output = read_hfss_design_result(run.output("result.json"), profile)
        for name, digest in output["mesh_exports"].items():
            detached = (
                inputs["request.json"] if name == "request.json" else run.output(name)
            )
            if hashlib.sha256(detached).hexdigest() != digest:
                raise ValueError(
                    "Detached HFSS artifact differs from its worker evidence."
                )
        if (
            output["source_files_sha256"] != source_identities
            or output["source_license_id"] != source.license_id
        ):
            raise ValueError("HFSS output does not match the detached source snapshot.")
        if output["worker_sha256"] != hashlib.sha256(worker).hexdigest():
            raise ValueError("HFSS output does not match the detached worker.")
        if (
            output["engine_sha256"] != aedt.sha256
            or output["engine_version"] != aedt.version
            or output["engine_license_id"] != aedt.license_id
            or output["engine_source_url"] != aedt.source_url
        ):
            raise ValueError("HFSS output does not match the requested AEDT pin.")
        if (
            output["python_sha256"] != python.sha256
            or output["python_version"] != python.version
            or output["python_license_id"] != python.license_id
            or output["python_source_url"] != python.source_url
        ):
            raise ValueError("HFSS output does not match the requested Python pin.")
    except (ValueError, KeyError, TypeError, IndexError) as failure:
        raise DeviceQualificationError(str(failure), runs=(run,)) from failure
    residuals = tuple(
        (value - target.value) / target.scale
        for value, target in zip(values, profile.targets, strict=True)
    )
    return HFSSDesignResult(
        values,
        residuals,
        tuple(output["mode_labels"]),
        output["mesh_identity"],
        device_artifact("hfss-superconducting-design", output, source, (run,)),
        run,
    )
