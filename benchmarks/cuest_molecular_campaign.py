# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fresh-process cuEST capacity qualification, not a single-H2 speedup claim.

Run ``python -m benchmarks.cuest_molecular_campaign --output campaign.json``.
``--angulars 0 1 2`` and ``--molecule-counts 1 2 4`` define the default Cartesian
capacity grid. Angular momentum may be zero; molecule counts and repeats must
be positive. Each case runs sequentially in a new Python process with the exact
parent environment and the actual worktree root as its working directory.

The controller performs metadata-only source identity capture, never environment
or device discovery. Children own GPU state and all scientific measurements.
Every child JSON and stdout/stderr log survives, including unsuccessful cases.
Completion requires every requested child to qualify against the matched DF
oracle, and at least two observed sizes for BOTH orbital and auxiliary capacity.
Qualification applies only to these measured capacities, policies, and devices;
no unmeasured speedup, scaling law, or whole-SCF JIT performance is inferred.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_SCIENTIFIC_CHECKS = frozenset(
    {
        "reference_converged",
        "native_jk_successful",
        "energies_finite",
        "forces_finite",
        "coulomb_finite",
        "exchange_finite",
        "matched_df_energy",
        "matched_df_forces",
        "matched_df_coulomb",
        "matched_df_exchange",
    }
)
_EVIDENCE_FIELDS = (
    "cases",
    "controlling_capacities",
    "expected_child_identity",
    "qualification_checks",
    "qualification_scope",
    "requested_cases",
    "status",
)


def capacity_grid(
    angulars: Sequence[int], molecule_counts: Sequence[int], *, repeats: int
) -> tuple[dict[str, int], ...]:
    """Normalize the requested Cartesian grid to angular-major numeric order."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not angulars or any(
        type(value) is not int or value not in (0, 1, 2) for value in angulars
    ):
        raise ValueError("angulars must be a nonempty list drawn from 0, 1, 2")
    if not molecule_counts or any(
        type(value) is not int or value < 1 for value in molecule_counts
    ):
        raise ValueError("molecule counts must be a nonempty list of positive integers")
    return tuple(
        {"angular": angular, "molecules": molecules, "repeats": repeats}
        for angular in sorted(set(angulars))
        for molecules in sorted(set(molecule_counts))
    )


def _nonnegative_seconds(value: object, /) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _sample_qualified(value: object, expected_checks: set[str], /) -> bool:
    if not isinstance(value, Mapping):
        return False
    checks = value.get("scientific_checks")
    return (
        isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(passed is True for passed in checks.values())
    )


def _provenance_errors(record: Mapping[str, Any], /) -> list[str]:
    """Admit software, binary, device, and execution-profile provenance."""
    errors: list[str] = []
    environment = record.get("environment")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(environment.get("fingerprint"), str)
        or not environment["fingerprint"]
    ):
        errors.append("Child shared runtime environment identity is missing.")
    dependencies = record.get("dependencies")
    required = ("phydrax", "phydrax-cuda13", "phydrax-cuest", "jax", "jaxlib", "pyscf")
    if not isinstance(dependencies, Mapping) or any(
        not isinstance(dependencies.get(name), str) or not dependencies[name]
        for name in required
    ):
        errors.append("Child required package/vendor dependency versions are missing.")
    if not isinstance(record.get("execution_flags"), Mapping):
        errors.append("Child execution/compiler-cache environment flags are missing.")
    native = record.get("native_component")
    if (
        not isinstance(native, Mapping)
        or not isinstance(native.get("build_hash"), str)
        or not native["build_hash"]
    ):
        errors.append("Child native build identity is missing.")
    else:
        for name in ("companion_binary", "vendor_binary"):
            binary = native.get(name)
            if not isinstance(binary, Mapping):
                errors.append(f"Child {name} identity is missing.")
                continue
            digest = binary.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(binary.get("bytes")) is not int
                or binary["bytes"] <= 0
            ):
                errors.append(f"Child {name} content identity is malformed.")
    devices = record.get("selected_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(
            not isinstance(device, Mapping)
            or device.get("platform") != "cuda"
            or type(device.get("id")) is not int
            or not isinstance(device.get("kind"), str)
            or device["id"] < 0
            or not device["kind"]
            for device in devices
        )
    ):
        errors.append("Child actual selected CUDA-device identities are missing.")
    return errors


def _measurement_errors(record: Mapping[str, Any], repeats: int, /) -> list[str]:
    """Admit compiler studies only after retaining provenance errors in order."""
    errors = _provenance_errors(record)
    study = record.get("compiled_kernels")
    if not isinstance(study, Mapping):
        return errors + ["Child real raw-FFI compiler evidence is missing."]
    checks = study.get("scientific_checks")
    if (
        not isinstance(checks, Mapping)
        or checks.get("all_first_and_warm_samples_qualified") is not True
    ):
        errors.append("Child compiled-kernel scientific qualification failed.")
    kernels = study.get("kernels")
    if not isinstance(kernels, Mapping) or set(kernels) != {
        "one_electron",
        "jk",
        "gradient",
    }:
        return errors + [
            "Child compiler study lacks the required three numerical kernels."
        ]
    for name, kernel in kernels.items():
        errors.extend(_kernel_measurement_errors(name, kernel, repeats))
    return errors


def _kernel_measurement_errors(name: str, kernel: object, repeats: int, /) -> list[str]:
    """Validate one executable's phase clocks, samples, and compiler memory."""
    errors: list[str] = []
    if not isinstance(kernel, Mapping):
        errors.append(f"Child {name} compiler record is malformed.")
        return errors
    timing = kernel.get("timings_seconds")
    if not isinstance(timing, Mapping) or any(
        not _nonnegative_seconds(timing.get(phase))
        for phase in ("lowering", "compile", "first_execution")
    ):
        errors.append(
            f"Child {name} lowering/compile/first timings are missing or invalid."
        )
        return errors
    warm = timing.get("warm")
    if (
        not isinstance(warm, Mapping)
        or type(warm.get("count")) is not int
        or warm["count"] != repeats
        or not isinstance(warm.get("samples_seconds"), list)
        or len(warm["samples_seconds"]) != repeats
        or any(not _nonnegative_seconds(value) for value in warm["samples_seconds"])
    ):
        errors.append(f"Child {name} warm compiler-executable timings are incomplete.")
    observable_names = {
        "one_electron": ("overlap", "kinetic", "nuclear_attraction"),
        "jk": ("coulomb", "exchange"),
        "gradient": ("one_electron", "two_electron", "pulay", "total_gradient", "forces"),
    }[name]
    expected_checks = {"status_zero"} | {
        f"{observable}_{suffix}"
        for observable in observable_names
        for suffix in ("finite", "matches_host")
    }
    if not _sample_qualified(kernel.get("first_validation"), expected_checks):
        errors.append(f"Child {name} first executable result is unqualified.")
    validations = kernel.get("warm_validation")
    if (
        not isinstance(validations, list)
        or len(validations) != repeats
        or any(not _sample_qualified(value, expected_checks) for value in validations)
    ):
        errors.append(f"Child {name} warm executable scientific results are incomplete.")
    compiler = kernel.get("compiler")
    if (
        not isinstance(compiler, Mapping)
        or not isinstance(compiler.get("source"), str)
        or not compiler["source"]
    ):
        errors.append(
            f"Child {name} compiler cost/memory analysis provenance is missing."
        )
    else:
        for field in ("argument_bytes", "output_bytes", "temporary_bytes"):
            value = compiler.get(field)
            if value is None:
                if (
                    not isinstance(compiler.get("unavailable_reason"), str)
                    or not compiler["unavailable_reason"]
                ):
                    errors.append(
                        f"Child {name} missing compiler memory estimate has no actual API reason."
                    )
            elif type(value) is not int or value < 0:
                errors.append(
                    f"Child {name} compiler memory estimate is not a measured nonnegative byte count."
                )
    return errors


def _case_errors(
    row: Mapping[str, Any],
    requested: Mapping[str, int],
    expected_child_identity: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if row.get("requested_capacity") != requested:
        errors.append("Controller row does not match its requested capacity/order.")
    if type(row.get("returncode")) is not int or row["returncode"] != 0:
        errors.append("Child process did not exit successfully.")
    if row.get("record_error"):
        errors.append(str(row["record_error"]))
    record = row.get("record")
    if not isinstance(record, Mapping):
        errors.append("Child JSON record is missing or is not an object.")
        return errors
    if record.get("provider") != "cuEST":
        errors.append("Child provider is not cuEST.")
    if record.get("identity") != expected_child_identity:
        errors.append("Child source/driver/evidence identity is missing or stale.")
    child_capacity = record.get("requested_capacity")
    if (
        not isinstance(child_capacity, Mapping)
        or child_capacity != requested
        or any(type(child_capacity.get(name)) is not int for name in requested)
    ):
        errors.append("Child requested capacity does not match the command.")
    if record.get("status") != "completed":
        errors.append("Child benchmark is not completed.")
    checks = record.get("scientific_checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _REQUIRED_SCIENTIFIC_CHECKS
        or any(passed is not True for passed in checks.values())
    ):
        errors.append("Child scientific checks are missing, malformed, or not all true.")
    reference = record.get("independent_reference")
    if not isinstance(reference, Mapping) or reference.get("status") != "qualified":
        errors.append("Child matched DF independent reference is not qualified.")
    for name in ("orbital_count", "auxiliary_count"):
        if type(record.get(name)) is not int or record[name] < 1:
            errors.append(f"Child {name} must be an observed positive integer.")
    errors.extend(_measurement_errors(record, requested.get("repeats", 0)))
    return errors


def aggregate_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    angulars: Sequence[int],
    molecule_counts: Sequence[int],
    repeats: int,
    expected_child_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure qualification of retained child records; never execute a campaign."""
    requested_cases = capacity_grid(angulars, molecule_counts, repeats=repeats)
    qualified_rows: list[dict[str, Any]] = []
    observed_orbitals: set[int] = set()
    observed_auxiliaries: set[int] = set()
    for index, row in enumerate(rows):
        requested = requested_cases[index] if index < len(requested_cases) else {}
        errors = _case_errors(row, requested, expected_child_identity)
        retained = dict(row)
        retained["qualification_errors"] = errors
        retained["status"] = "qualified" if not errors else "failed"
        qualified_rows.append(retained)
        if not errors:
            observed_orbitals.add(row["record"]["orbital_count"])
            observed_auxiliaries.add(row["record"]["auxiliary_count"])
    profiles = [
        {
            "environment": row["record"]["environment"]["fingerprint"],
            "dependencies": row["record"]["dependencies"],
            "native_companion": row["record"]["native_component"]["companion_binary"][
                "sha256"
            ],
            "native_vendor": row["record"]["native_component"]["vendor_binary"]["sha256"],
            "devices": row["record"]["selected_devices"],
            "flags": row["record"]["execution_flags"],
        }
        for row in qualified_rows
        if row["status"] == "qualified"
    ]
    checks = {
        "all_requested_cases_retained": len(rows) == len(requested_cases),
        "all_children_scientifically_qualified": bool(rows)
        and all(row["status"] == "qualified" for row in qualified_rows),
        "orbital_capacity_varies": len(observed_orbitals) >= 2,
        "auxiliary_capacity_varies": len(observed_auxiliaries) >= 2,
        "comparison_environment_consistent": bool(profiles)
        and all(profile == profiles[0] for profile in profiles),
    }
    if (
        not checks["all_requested_cases_retained"]
        or not checks["all_children_scientifically_qualified"]
    ):
        status = "failed"
    else:
        status = "completed" if all(checks.values()) else "unqualified"
    return {
        "provider": "cuEST",
        "status": status,
        "requested_cases": list(requested_cases),
        "expected_child_identity": dict(expected_child_identity),
        "cases": qualified_rows,
        "qualification_checks": checks,
        "controlling_capacities": {
            "angulars": sorted(set(angulars)),
            "molecule_counts": sorted(set(molecule_counts)),
            "repeats_per_case": repeats,
            "qualified_observed_orbital_counts": sorted(observed_orbitals),
            "qualified_observed_auxiliary_counts": sorted(observed_auxiliaries),
        },
        "qualification_scope": (
            "Only the retained, independently matched-DF-qualified capacities, policies, "
            "source, flags, and child devices are measured. Both orbital and auxiliary "
            "capacities must vary. No single-H2 speedup extrapolation, unmeasured scaling "
            "claim, or whole-SCF JIT claim is made."
        ),
    }


def _driver_evidence_fields(driver: Path) -> tuple[str, ...]:
    """Read the driver's literal schema without importing its numerical modules."""
    module = ast.parse(driver.read_text(encoding="utf-8"), filename=str(driver))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EVIDENCE_FIELDS"
            for target in node.targets
        ):
            fields = ast.literal_eval(node.value)
            if (
                isinstance(fields, tuple)
                and fields
                and all(isinstance(field, str) for field in fields)
            ):
                return fields
            break
    raise ValueError("Singlecase driver must declare its literal _EVIDENCE_FIELDS tuple.")


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def _load_child_record(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
        ), None
    except (OSError, UnicodeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def run_campaign(
    *, angulars: Sequence[int], molecule_counts: Sequence[int], repeats: int, output: Path
) -> dict[str, Any]:
    """Run sequential fresh children and retain every result, without GPU discovery."""
    requested_cases = capacity_grid(angulars, molecule_counts, repeats=repeats)
    # Snapshot before importing metadata helpers; never override any parent flag.
    child_environment = dict(os.environ)
    from benchmarks._io import write_json_atomic
    from benchmarks._runtime import capture_benchmark_identity

    driver = PROJECT_ROOT / "benchmarks" / "cuest_molecular.py"
    expected_child_identity = capture_benchmark_identity(
        PROJECT_ROOT, driver, _driver_evidence_fields(driver)
    ).to_dict()
    identity = capture_benchmark_identity(
        PROJECT_ROOT, Path(__file__), _EVIDENCE_FIELDS
    ).to_dict()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Persistent unique directories prevent old child JSON from qualifying a new run.
    artifacts = Path(tempfile.mkdtemp(prefix=f"{output.stem}.cases-", dir=output.parent))
    rows: list[dict[str, Any]] = []
    for capacity in requested_cases:
        stem = f"angular-{capacity['angular']}_molecules-{capacity['molecules']}"
        child_json = artifacts / f"{stem}.json"
        stdout_path = artifacts / f"{stem}.stdout.log"
        stderr_path = artifacts / f"{stem}.stderr.log"
        command = [
            sys.executable,
            "-m",
            "benchmarks.cuest_molecular",
            "--angular",
            str(capacity["angular"]),
            "--molecules",
            str(capacity["molecules"]),
            "--repeats",
            str(repeats),
            "--output",
            str(child_json),
        ]
        row: dict[str, Any] = {
            "requested_capacity": capacity,
            "command": command,
            "cwd": str(PROJECT_ROOT),
            "output_path": str(child_json),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "returncode": None,
        }
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                process = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=child_environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            row["returncode"] = process.returncode
        except OSError as error:
            row["process_error"] = f"{type(error).__name__}: {error}"
        row["record"], row["record_error"] = _load_child_record(child_json)
        rows.append(row)
        # Preserve every completed/failed row even if a later child is interrupted.
        partial = aggregate_cases(
            rows,
            angulars=angulars,
            molecule_counts=molecule_counts,
            repeats=repeats,
            expected_child_identity=expected_child_identity,
        )
        partial.update(
            identity=identity,
            artifacts_path=str(artifacts),
            environment_policy="Exact parent environment snapshot; no child overrides.",
            execution_policy="Sequential fresh Python processes; no controller GPU discovery.",
        )
        write_json_atomic(output, partial)
    return partial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angulars", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--molecule-counts", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        capacity_grid(args.angulars, args.molecule_counts, repeats=args.repeats)
    except ValueError as error:
        parser.error(str(error))
    result = run_campaign(
        angulars=args.angulars,
        molecule_counts=args.molecule_counts,
        repeats=args.repeats,
        output=args.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False, sort_keys=True))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
