#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Acquire and freeze external Flexodeal outputs, without importing Phydrax physics.

This tool does not admit a continuum fidelity. The frozen records are numerical
cross-implementation references, not biological validation or convergence proof.
Build/run are deliberately separate from acquisition and offline asset checking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "tests/fixtures/flexodeal_0698e3d"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest() -> dict:
    return json.loads((ASSETS / "manifest.json").read_text())


def verify_files(directory: Path, entries: dict) -> None:
    for name, entry in entries.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symbolic reference asset: {name}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Reference asset content mismatch: {name}")


def verify_inputs() -> dict:
    spec = manifest()
    verify_files(
        ASSETS, {name: spec["upstream_files"][name] for name in spec["vendored_inputs"]}
    )
    return {
        "source_id": spec["source_id"],
        "input_assets_verified": True,
        "continuum_implementation_admitted": spec["gate"][
            "idealized_ma_implementation_admitted"
        ],
        "gate": spec["gate"],
    }


def acquire(destination: Path) -> dict:
    spec = manifest()
    with urllib.request.urlopen(spec["archive"]["url"], timeout=120) as response:
        payload = response.read()
    if (
        len(payload) != spec["archive"]["bytes"]
        or hashlib.sha256(payload).hexdigest() != spec["archive"]["sha256"]
    ):
        raise ValueError("Pinned upstream archive content mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Read only explicitly pinned members; never extract untrusted paths.
        members = {}
        for name, entry in spec["upstream_files"].items():
            content = archive.read(spec["archive"]["prefix"] + name)
            if (
                len(content) != entry["bytes"]
                or hashlib.sha256(content).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"Pinned upstream member content mismatch: {name}")
            members[name] = content
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in members.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return {
        "source_id": spec["source_id"],
        "source_directory": str(destination.resolve()),
        "archive_sha256": spec["archive"]["sha256"],
    }


def _write_record(path: Path, record: dict) -> dict:
    content = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    record = {**record, "content_id": hashlib.sha256(content.encode()).hexdigest()}
    # The record is the publication point. Failed commands leave diagnostics only.
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return record


def _run(command: list[str], cwd: Path, log: Path, timeout: float) -> float:
    start = time.monotonic()
    with log.open("xb") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    return time.monotonic() - start


def build(
    source: Path,
    destination: Path,
    deal_ii: Path,
    runtime_evidence: Path,
    jobs: int,
    timeout: float,
) -> dict:
    spec = manifest()
    source, destination, deal_ii = (
        source.resolve(),
        destination.resolve(),
        deal_ii.resolve(),
    )
    verify_files(source, spec["upstream_files"])
    # Caller supplies observed compiler, deal.II build, and transitive runtime-library
    # identities; no dependency version is guessed from a directory name.
    environment = runtime_evidence.read_bytes()
    if not environment.strip():
        raise ValueError("Observed runtime dependency evidence is required")
    if not deal_ii.is_dir() or jobs < 1 or timeout <= 0:
        raise ValueError(
            "An installed deal.II directory and positive jobs/timeout are required"
        )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "runtime-evidence.txt").write_bytes(environment)
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(destination),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DDEAL_II_DIR={deal_ii}",
    ]
    compile_command = ["cmake", "--build", str(destination), "--parallel", str(jobs)]
    configure_seconds = _run(
        configure, destination, destination / "configure.log", timeout
    )
    compile_seconds = _run(
        compile_command, destination, destination / "compile.log", timeout
    )
    verify_files(source, spec["upstream_files"])
    names = (
        "flexodeal",
        "CMakeCache.txt",
        "runtime-evidence.txt",
        "configure.log",
        "compile.log",
    )
    artifacts = {
        name: {
            "sha256": sha256(destination / name),
            "bytes": (destination / name).stat().st_size,
        }
        for name in names
    }
    return _write_record(
        destination / "build-record.json",
        {
            "source_id": spec["source_id"],
            "source_manifest_sha256": sha256(ASSETS / "manifest.json"),
            "source_directory": str(source),
            "platform": platform.platform(),
            "commands": [configure, compile_command],
            "configure_seconds": configure_seconds,
            "compile_seconds": compile_seconds,
            "artifacts": artifacts,
            "runtime_evidence_reviewed": False,
        },
    )


def _parameter(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^(\s*set {re.escape(key)}\s*=\s*)[^\n]+$"
    updated, count = re.subn(pattern, lambda match: match[1] + value, text)
    if count != 1:
        raise ValueError(f"Expected one source parameter: {key}")
    return updated


def case_inputs(case: str) -> dict[str, bytes]:
    spec = manifest()
    if case not in spec["campaign_cases"]:
        raise ValueError(f"Unknown reference case: {case}")
    verify_inputs()
    names = (
        "parameters.prm",
        "markers.dat",
        "control_points_activation.dat",
        "control_points_strain.dat",
    )
    inputs = {name: (ASSETS / name).read_bytes() for name in names}
    if case == "repository-default-qp":
        inputs["qp_data_2.qp.csv"] = (ASSETS / "qp_data_2.qp.csv").read_bytes()
    if case in ("repository-default-fields", "passive-cyclic", "fixed-end-activation"):
        parameters = inputs["parameters.prm"].decode()
        for key in ("Output binary files main variables", "Output binary files tensors"):
            parameters = _parameter(parameters, key, "true")
        inputs["parameters.prm"] = parameters.encode()
    zeroed = {
        "passive-cyclic": "control_points_activation.dat",
        "fixed-end-activation": "control_points_strain.dat",
    }
    if case in zeroed:
        name = zeroed[case]
        rows = [line.split() for line in inputs[name].decode().splitlines()]
        if any(len(row) != 2 for row in rows):
            raise ValueError("Expected the source two-column time history")
        inputs[name] = "".join(f"{row[0]}\t0.000000\n" for row in rows).encode()
    return inputs


def validate_history(path: Path, expected_times: list[float]) -> dict:
    with path.open(newline="") as stream:
        table = csv.DictReader(stream)
        if (
            not table.fieldnames
            or "Time [s]" not in table.fieldnames
            or "Volume [m^3]" not in table.fieldnames
        ):
            raise ValueError(f"Missing source time/volume columns: {path.name}")
        rows = list(table)
    if len(rows) != len(expected_times):
        raise ValueError(f"Incomplete reference history: {path.name}")
    for row, expected in zip(rows, expected_times, strict=True):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed reference row: {path.name}")
        values = {key: float(value) for key, value in row.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite reference history: {path.name}")
        if not math.isclose(values["Time [s]"], expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Wrong reference time grid: {path.name}")
        if values["Volume [m^3]"] <= 0:
            raise ValueError(f"Nonpositive reference volume: {path.name}")
    return {
        "samples": len(rows),
        "start_s": expected_times[0],
        "end_s": expected_times[-1],
    }


def run_case(build_directory: Path, destination: Path, case: str, timeout: float) -> dict:
    spec = manifest()
    build_directory, destination = build_directory.resolve(), destination.resolve()
    build_record = json.loads((build_directory / "build-record.json").read_text())
    if build_record["source_id"] != spec["source_id"] or build_record[
        "source_manifest_sha256"
    ] != sha256(ASSETS / "manifest.json"):
        raise ValueError("Foreign reference build")
    verify_files(build_directory, build_record["artifacts"])
    inputs = case_inputs(case)
    if timeout <= 0:
        raise ValueError("Reference timeout must be positive")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in inputs.items():
        (destination / name).write_bytes(content)
    # Flexodeal uses an 80-byte output-directory C buffer; keep this argument
    # short and relative even when the containing campaign path is long.
    command = [str(build_directory / "flexodeal"), "-OUTPUT_DIR=outputs"]
    if case == "repository-default-qp":
        command.append("-QP_FILE=qp_data_2.qp.csv")
    elapsed = _run(command, destination, destination / "run.log", timeout)
    verify_files(build_directory, build_record["artifacts"])
    for name, content in inputs.items():
        if (destination / name).read_bytes() != content:
            raise ValueError(f"Reference input changed during execution: {name}")
    parameters = inputs["parameters.prm"].decode()
    dt = float(re.search(r"set Time step size\s*=\s*(\S+)", parameters)[1])
    end = float(re.search(r"set End time\s*=\s*(\S+)", parameters)[1])
    expected_times = [i * dt for i in range(math.ceil(end / dt))]
    outputs = destination / "outputs"
    histories = {
        name: validate_history(outputs / name, expected_times)
        for name in ("force_data-3d.csv", "energy_data-3d.csv")
    }
    if not (outputs / "grid-3d.msh").is_file():
        raise ValueError("Missing generated reference mesh")
    for step in range(len(expected_times)):
        if not (outputs / f"force_data-3d-{step:03}.csv").is_file():
            raise ValueError(f"Missing boundary-force field at step {step}")
        if case in (
            "repository-default-fields",
            "passive-cyclic",
            "fixed-end-activation",
        ):
            for kind, columns in (("main", 21), ("tensors", 69)):
                path = outputs / f"cell_data_{kind}-3d-{step:03}.data"
                if (
                    not path.is_file()
                    or path.stat().st_size == 0
                    or path.stat().st_size % (4 * columns)
                ):
                    raise ValueError(f"Missing or truncated binary QP field: {path.name}")
    for name in ("build-record.json", "runtime-evidence.txt"):
        shutil.copyfile(build_directory / name, destination / name)
    shutil.copyfile(ASSETS / "manifest.json", destination / "source-manifest.json")
    artifacts = {
        str(path.relative_to(destination)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    return _write_record(
        destination / "frozen-reference.json",
        {
            "status": "frozen-executable-output",
            "source_id": spec["source_id"],
            "case": case,
            "case_description": spec["campaign_cases"][case],
            "build_record_sha256": sha256(build_directory / "build-record.json"),
            "command": command,
            "elapsed_seconds": elapsed,
            "histories": histories,
            "artifacts": artifacts,
            "continuum_implementation_admitted": False,
            "limitations": [
                "This case ran; other source campaign cases and dependency evidence still require review.",
                *spec["gate"]["blockers"],
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-inputs")
    acquisition = commands.add_parser("acquire")
    acquisition.add_argument("--destination", type=Path, required=True)
    compilation = commands.add_parser("build")
    compilation.add_argument("--source", type=Path, required=True)
    compilation.add_argument("--destination", type=Path, required=True)
    compilation.add_argument("--deal-ii", type=Path, required=True)
    compilation.add_argument(
        "--runtime-evidence",
        type=Path,
        required=True,
        help="Observed compiler, deal.II, transitive library identities; review remains required",
    )
    compilation.add_argument("--jobs", type=int, default=1)
    compilation.add_argument("--timeout", type=float, default=3600)
    run = commands.add_parser("run")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument("--case", choices=tuple(manifest()["campaign_cases"]), required=True)
    run.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.command == "verify-inputs":
        report = verify_inputs()
    elif args.command == "acquire":
        report = acquire(args.destination)
    elif args.command == "build":
        report = build(
            args.source,
            args.destination,
            args.deal_ii,
            args.runtime_evidence,
            args.jobs,
            args.timeout,
        )
    else:
        report = run_case(args.build, args.destination, args.case, args.timeout)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
