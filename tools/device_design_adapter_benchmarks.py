#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Live device-engine qualification; parser fixtures never count as benchmarks.

Run from the repository with its Python environment:
  python tools/device_design_adapter_benchmarks.py \\
    --engine hfss --config hfss-run.json --output hfss-evidence
  python tools/device_design_adapter_benchmarks.py \\
    --engine geant4 --config detector-run.json \\
    --output detector-evidence --repeats 3

Both JSON configs require ``source={directory,commit,license_id}``. Executable
objects use the PinnedExecutable fields ``path``, ``sha256``, ``version``,
``license_id``, and ``source_url``. Relative executable/source paths resolve
from the config directory without traversal; detector input resources must
remain beneath that directory.

HFSS config: ``python`` and ``aedt`` executable objects;
``profile={group,mode_windows_hz,targets,max_passes,
frequency_tolerance_percent,capacitance_tolerance_percent}``; and
``design_variables`` mapping exact upstream names to unit-bearing strings.
Targets have ``name,quantity,labels,value,scale``. The live profile requires
an exclusive licensed Windows AEDT 2021 R2/2022 R2 host, Python 3.11/3.12, and
the exact QDesignOptimizer Poetry-lock dependency profile.

Geant4 config: a ``root`` executable object plus ``settings_file``,
``design_file``, and ``field_map_file``; the profile has ``events``, ``seed``,
``runtime_build_id``, ``geometry_profile='corrected-silicon-units'``,
``minimum_tracks_per_bin``, and ``maximum_reduced_chi_squared``. It requires an
initialized ECCE/Fun4All/Geant4/ROOT host or container and its runtime data.
Missing engines/dependencies are reported as failed/unavailable qualification
and always produce a nonzero exit.

Optional common keys are ``timeout``, ``max_output_bytes``, and ``environment``.
Dependencies are never installed or downloaded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path, PurePath

from phydrax.interchange._device_design import DeviceQualificationError, DeviceSource
from phydrax.interchange._resource import read_bounded_resource, ResourceLimits
from phydrax.interchange.energy_runtime import EnergyRuntimeError, PinnedExecutable
from phydrax.interchange.geant4_detector_design import (
    GYMDetectorProfile,
    read_gym_detector_config,
    run_gym_detector_design,
)
from phydrax.interchange.hfss_design import (
    HFSSDesignProfile,
    HFSSDesignTarget,
    run_hfss_design,
)


_CONFIG_BYTES = 1024 * 1024


def _resource(path, base, max_bytes):
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ValueError("Input resource paths must be nonempty POSIX text paths.")
    return read_bounded_resource(
        path,
        trusted_root=base,
        limits=ResourceLimits(
            max_bytes=max_bytes,
            max_depth=32,
            max_nodes=1,
            max_attributes=0,
            max_losses=0,
        ),
    ).data


def _external_path(value, base):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("External runtime paths must be nonempty text paths.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if ".." in PurePath(value).parts:
            raise ValueError("Relative external runtime path traversal is disabled.")
        path = base / path
    return str(path)


def _pinned_executable(data, base):
    fields = dict(data)
    fields["path"] = _external_path(fields["path"], base)
    return PinnedExecutable(**fields)


def _source(data, base):
    fields = dict(data)
    fields["directory"] = _external_path(fields["directory"], base)
    return DeviceSource(**fields)


def _save_run(run, directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "stdout.txt").write_bytes(run.stdout)
    (directory / "stderr.txt").write_bytes(run.stderr)
    for output in run.outputs:
        path = directory / output.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(output.data)
    return {
        "artifact_id": run.artifact.artifact_id,
        "command": run.command,
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "elapsed_seconds": run.elapsed_seconds,
        "error": run.error,
        "outputs": {
            output.path: output.artifact.content_digest for output in run.outputs
        },
    }


def _failure_status(failure):
    if isinstance(failure, DeviceQualificationError):
        if isinstance(failure.__cause__, EnergyRuntimeError):
            return "engine-unavailable-or-failed"
        return "qualification-failed"
    if isinstance(failure, EnergyRuntimeError):
        return "engine-unavailable-or-failed"
    if isinstance(failure, (OSError, subprocess.SubprocessError)):
        return "runtime-or-source-unavailable"
    return "configuration-failed"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engine", choices=("hfss", "geant4"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    config_path = args.config.expanduser().resolve(strict=True)
    base = config_path.parent
    config = json.loads(_resource(config_path.name, base, _CONFIG_BYTES))
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a JSON object.")
    environment = config.get("environment")
    if environment is not None and (
        not isinstance(environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
    ):
        raise ValueError("environment must map text names to text values.")
    options = {
        key: config[key]
        for key in ("timeout", "max_output_bytes", "environment")
        if key in config
    }
    input_limit = config.get("max_output_bytes", 256 * 1024 * 1024)
    if type(input_limit) is not int or input_limit < 1:
        raise ValueError("max_output_bytes must be a positive integer.")

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "engine": args.engine,
        "evidence_kind": "live-engine-qualification",
        "accepted": False,
        "runs": [],
        "dependencies_installed_by_benchmark": False,
    }
    accepted = True
    source = None
    for repeat in range(args.repeats):
        directory = args.output / f"repeat-{repeat}"
        try:
            if source is None:
                source = _source(config["source"], base)
            if args.engine == "hfss":
                profile_data = dict(config["profile"])
                profile_data["mode_windows_hz"] = tuple(
                    tuple(window) for window in profile_data["mode_windows_hz"]
                )
                profile_data["targets"] = tuple(
                    HFSSDesignTarget(**{**target, "labels": tuple(target["labels"])})
                    for target in profile_data["targets"]
                )
                profile = HFSSDesignProfile(**profile_data)
                result = run_hfss_design(
                    _pinned_executable(config["python"], base),
                    _pinned_executable(config["aedt"], base),
                    source,
                    profile,
                    config["design_variables"],
                    **options,
                )
                row = {
                    "values": result.values,
                    "normalized_residuals": result.normalized_residuals,
                    "mesh_identity": result.mesh_identity,
                    "uncertainty": result.uncertainty,
                    "execution": _save_run(result.run, directory / "hfss"),
                }
            else:
                settings = read_gym_detector_config(
                    _resource(config["settings_file"], base, _CONFIG_BYTES)
                )
                profile = GYMDetectorProfile(settings=settings, **config["profile"])
                profile = replace(profile, seed=profile.seed + repeat)
                result = run_gym_detector_design(
                    _pinned_executable(config["root"], base),
                    source,
                    profile,
                    read_gym_detector_config(
                        _resource(config["design_file"], base, _CONFIG_BYTES)
                    ),
                    field_map=_resource(config["field_map_file"], base, input_limit),
                    **options,
                )
                row = {
                    "momentum_resolution_percent": result.momentum_resolution_percent,
                    "propagated_fit_error_percent": result.propagated_fit_error_percent,
                    "kalman_inefficiency": result.kalman_inefficiency,
                    "kalman_binomial_error": result.kalman_binomial_error,
                    "fit_errors_are_calibrated_observation_noise": False,
                    "upstream_inverse_thickness_diagnostic": (
                        result.upstream_inverse_thickness_diagnostic
                    ),
                    "cost_objective": None,
                    "seed": profile.seed,
                    "qualification_failures": result.qualification_failures,
                    "simulation": _save_run(result.simulation, directory / "simulation"),
                    "reconstruction": _save_run(
                        result.reconstruction, directory / "reconstruction"
                    ),
                }
            row.update(
                {
                    "accepted": result.accepted,
                    "status": "accepted" if result.accepted else "qualification-failed",
                    "artifact_id": result.artifact.artifact_id,
                    "artifact_status": result.artifact.status,
                }
            )
            accepted = accepted and result.accepted
        except (
            EnergyRuntimeError,
            DeviceQualificationError,
            ValueError,
            TypeError,
            KeyError,
            OSError,
            subprocess.SubprocessError,
        ) as failure:
            accepted = False
            row = {
                "accepted": False,
                "status": _failure_status(failure),
                "error_type": type(failure).__name__,
                "error": str(failure),
            }
            if isinstance(failure, EnergyRuntimeError) and failure.result is not None:
                row["execution"] = _save_run(failure.result, directory / "failed-engine")
            if isinstance(failure, DeviceQualificationError):
                row["artifact_id"] = failure.artifact.artifact_id
                row["executions"] = [
                    _save_run(run, directory / f"failed-{index}")
                    for index, run in enumerate(failure.runs)
                ]
        report["runs"].append(row)
        if not accepted:
            break

    report["accepted"] = accepted
    if args.engine == "geant4" and accepted and len(report["runs"]) > 1:
        report["independent_seed_repeatability_sample_sd_percent"] = statistics.stdev(
            row["momentum_resolution_percent"] for row in report["runs"]
        )
        report["independent_seed_kalman_inefficiency_sample_sd"] = statistics.stdev(
            row["kalman_inefficiency"] for row in report["runs"]
        )
    payload = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "qualification.json").write_text(payload)
    print(payload)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
