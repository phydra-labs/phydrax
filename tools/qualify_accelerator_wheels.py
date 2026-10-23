# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Qualify selected upstream accelerator wheels on actual Linux hardware.

Creates a fresh isolated venv, installs the supplied core wheel and selected
upstream extra, binds installed core bytes to the wheel and hashes upstream
RECORD files, then runs the exact scientific cases and advertised example.
No provisioning, CPU fallback, skips-as-passes, or owned upstream build IDs.
Hardware qualification remains pending unless every stage passes here.
Each explicitly selected local device executes the full suite in its own process.
Observed device counts do not qualify unselected devices or architectures.
ROCm also initializes CPU for JAX host callbacks; scientific arrays must stay
exclusively on the explicitly selected AMD device.
Full provenance uses a validated, 64-MiB-limited JSON sidecar, never log tails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from tools._accelerator_process import read_json_evidence, run
from tools.check_accelerator_extras import Accelerator, metadata_errors, sha256
from tools.check_cuda_extras import wheel_metadata
from tools.qualify_cuda_wheels import test_requirements


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = "tests/integration/test_accelerator_scientific_execution.py"
REQUIRED_CASES = (
    "test_mlp_pde_derivatives_and_update",
    "test_polynomial_coordinate_gradient_and_hessian",
    *(f"test_wide_integer_sort_and_arithmetic[{dtype}]" for dtype in ("int64", "uint64")),
    "test_point_gather_scatter_full_width_addresses",
    *(f"test_full_width_window_gather[{dtype}]" for dtype in ("int64", "uint64")),
    *(
        f"test_scatter_update_cotangent_drops_out_of_bounds[{dtype}]"
        for dtype in ("int64", "uint64")
    ),
    "test_compiled_loops_and_ordered_callbacks",
    "test_compiled_callback_failure",
    "test_bounded_linear_solve",
    "test_linear_iteration_exhaustion",
    "test_real_fourier_resampling",
)
EXAMPLE = "examples/accelerator_scientific_execution.py"
RUNTIME_PROPERTY = "accelerator_runtime"
ROCM_CASES = tuple(
    f"{name}[{kind}]"
    for name in (
        "test_rocm_native_linear_precision",
        "test_rocm_linear_precision_exhaustion",
        "test_rocm_fourier_precision",
    )
    for kind in ("float64", "complex128")
)
ROCM_ENVIRONMENT = (
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_DEVICE_LIB_PATH",
    "LLVM_PATH",
    "LD_LIBRARY_PATH",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def host_errors(system: str, machine: str) -> list[str]:
    return (
        []
        if system == "Linux" and machine == "x86_64"
        else [
            "Qualification requires native Linux x86_64 with the selected accelerator; emulation and CPU fallback are forbidden"
        ]
    )


def clean_environment(venv: Path, accelerator: Accelerator | None) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR")
        if key in os.environ
    }
    environment["PATH"] = os.pathsep.join((str(venv / "bin"), "/usr/bin", "/bin"))
    environment["PYTHONNOUSERSITE"] = "1"
    if accelerator is not None:
        environment.update(
            JAX_PLATFORMS=",".join(accelerator.qualification_platforms),
            JAX_ENABLE_X64="1",
            PHYDRAX_ACCELERATOR=accelerator.value,
        )
    return environment


def wheel_files(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        # Installers rewrite RECORD; bind every immutable wheel member instead.
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if not name.endswith(("/", "/RECORD")) and ".data/" not in name
        }


def runtime_errors(record: Mapping[str, object], accelerator: Accelerator) -> list[str]:
    errors: list[str] = []
    if "frontend_index_patch" not in record or not _frontend_patch_valid(
        record["frontend_index_patch"]
    ):
        errors.append("Missing or invalid compiler frontend patch identity")
    if record.get("selected") != accelerator.value:
        errors.append("Runtime selection differs from requested accelerator")
    runtime = record.get("platform_version")
    kind = record.get("device_kind")
    if (
        not isinstance(runtime, str)
        or not runtime
        or not isinstance(kind, str)
        or not kind
    ):
        errors.append("Missing actual upstream runtime/device identity")
    match accelerator:
        case Accelerator.TPU:
            if record.get("platform") != "tpu" or record.get("client_platform") != "tpu":
                errors.append("Actual TPU platform is required")
            if record.get("matmul_precision") != "highest":
                errors.append(
                    "TPU scientific qualification requires highest matmul precision (six BF16 passes)"
                )
        case Accelerator.ONEAPI:
            platform_name = record.get("platform")
            client = str(record.get("client_platform", "")).lower()
            if not (
                platform_name == "oneapi"
                or (
                    platform_name == "gpu"
                    and (
                        client in {"oneapi", "sycl"}
                        or str(runtime).lower().startswith(("oneapi", "sycl"))
                    )
                )
            ):
                errors.append(
                    "Actual Intel oneAPI runtime is required, not CPU or another GPU"
                )
        case Accelerator.ROCM10:
            if (
                not accelerator.matches_rocm_runtime(
                    str(record.get("platform", "")),
                    str(record.get("client_platform", "")),
                    str(runtime),
                )
                or record.get("rocm_major") != accelerator.rocm_major
            ):
                errors.append(
                    f"Actual ROCm identity and selected installation family {accelerator.rocm_major} are required"
                )
    if record.get("x64_enabled") is not True:
        errors.append("Exact wide integer addressing requires x64 enabled")
    if (
        any(
            type(record.get(field)) is not int
            for field in ("process_count", "process_index", "execution_device_count")
        )
        or record.get("process_count") != 1
        or record.get("process_index") != 0
        or record.get("execution_device_count") != 1
    ):
        errors.append(
            "Bounded qualification requires one process executing on one selected device"
        )
    for field in ("global_device_count", "local_device_count"):
        count = record.get(field)
        if type(count) is not int or count < 1:
            errors.append(f"Missing truthful observed {field}")
    if type(record.get("qualified_device_id")) is not int:
        errors.append("Missing actual qualified device ID")
    return errors


def case_report_errors(
    path: Path, expected_runtime: Mapping[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    """Require every exact case once, passed with actual matching runtime evidence."""
    if not path.is_file():
        return ["Scientific case report is missing"], []
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        return [f"Malformed scientific case report: {error}"], []
    errors: list[str] = []
    nodes = list(root.iter("testcase"))
    counts = Counter(node.get("name") for node in nodes)
    selected = Accelerator(str(expected_runtime["selected"]))
    required = REQUIRED_CASES + (ROCM_CASES if selected.rocm_major is not None else ())
    if counts != Counter(required):
        errors.append(f"Exact case coverage differs: {dict(counts)}")
    cases: list[dict[str, object]] = []
    for node in nodes:
        name = node.get("name")
        if (
            node.get("classname")
            != "tests.integration.test_accelerator_scientific_execution"
        ):
            errors.append(f"{name}: unexpected scientific test owner")
        failures = [
            child.tag for child in node if child.tag in {"failure", "error", "skipped"}
        ]
        properties = [
            prop.get("value", "")
            for prop in node.findall("./properties/property")
            if prop.get("name") == RUNTIME_PROPERTY
        ]
        if failures:
            errors.append(f"{name}: nonpassing stages {failures}")
        if not properties:
            errors.append(f"{name}: actual result placement evidence missing")
        records: list[object] = []
        for text in properties:
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                errors.append(f"{name}: malformed runtime evidence")
                continue
            records.append(record)
            if record != dict(expected_runtime):
                errors.append(f"{name}: runtime differs from probed installed runtime")
        cases.append(
            {
                "name": name,
                "outcome": "passed" if not failures else "failed",
                "runtime": records,
                "duration_seconds": node.get("time"),
            }
        )
    return errors, cases


def last_json(result: Mapping[str, Any]) -> dict[str, Any] | None:
    if result["returncode"] != 0 or result.get("timed_out", False):
        return None
    lines = str(result["stdout"]).splitlines()
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frontend_patch_valid(record: object) -> bool:
    if record is None:
        return True
    return (
        isinstance(record, dict)
        and set(record)
        == {
            "algorithm",
            "jax_version",
            "upstream_source_sha256",
            "upstream_function_sha256",
            "upstream_dynamic_slice_sha256",
            "patch_source_sha256",
        }
        and all(
            isinstance(record[key], str) and bool(record[key])
            for key in ("algorithm", "jax_version")
        )
        and all(
            _sha256_text(record[key])
            for key in (
                "upstream_source_sha256",
                "upstream_function_sha256",
                "upstream_dynamic_slice_sha256",
                "patch_source_sha256",
            )
        )
    )


def _distribution_record_valid(record: object) -> bool:
    if not isinstance(record, dict) or set(record) != {
        "version",
        "files",
        "installed_sha256",
    }:
        return False
    files = record["files"]
    if (
        not isinstance(record["version"], str)
        or not record["version"]
        or not isinstance(files, dict)
        or not files
    ):
        return False
    if any(
        not isinstance(path, str) or not path or not _sha256_text(digest)
        for path, digest in files.items()
    ):
        return False
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return record["installed_sha256"] == hashlib.sha256(canonical).hexdigest()


def _bundle_record_valid(provenance: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    """Validate Linux-produced ROCm paths independently of the inspecting host."""
    try:
        bundle = provenance["rocm_bundle"]
        manifest_text = bundle["manifest"]
        manifest = json.loads(manifest_text)
        distributions = provenance["distributions"]
        facts = diagnostics["facts"]
        manifest_relative = "phydrax_rocm_runtime/runtime.json"
        if (
            manifest["sdk_version"] != "10.1.0"
            or manifest["architectures"] != ["gfx1250", "gfx942", "gfx950"]
            or facts["bundle_version"] != "10.1.0"
            or facts["bundle_architectures"] != ["gfx1250", "gfx942", "gfx950"]
            or facts["bundle_error"] is not None
            or facts["bundle_loaded"] is not True
            or distributions["phydrax-rocm-runtime"]["files"].get(manifest_relative)
            != hashlib.sha256(manifest_text.encode()).hexdigest()
        ):
            return False
        venv = PurePosixPath(provenance["venv"])
        manifest_path = PurePosixPath(facts["bundle_manifest"])
        if (
            not venv.is_absolute()
            or not manifest_path.is_absolute()
            or ".." in manifest_path.parts
            or not manifest_path.is_relative_to(venv)
            or not str(manifest_path).endswith("/" + manifest_relative)
        ):
            return False
        site = manifest_path.parent.parent
        expected_components = {
            "core",
            "libraries",
            "device_gfx1250",
            "device_gfx942",
            "device_gfx950",
        }
        if set(manifest["components"]) != expected_components:
            return False
        if set(bundle["payloads"]) != {
            "phydrax-rocm-" + key.replace("_", "-") for key in expected_components
        }:
            return False
        expected_facts = []
        owned_files: dict[str, str] = {}
        for key, component in manifest["components"].items():
            name = "phydrax-rocm-" + key.replace("_", "-")
            if component["distribution"] != name or component["version"] != "10.1.0":
                return False
            upstream = component["upstream"]
            if not upstream["url"].startswith("https://") or not _sha256_text(
                upstream["sha256"]
            ):
                return False
            payload = bundle["payloads"][name]
            files = distributions[name]["files"]
            digest = hashlib.sha256(payload["text"].encode()).hexdigest()
            if (
                not payload["path"].endswith(".dist-info/phydrax-payload.json")
                or files.get(payload["path"]) != digest
                or component["payload_sha256"] != digest
            ):
                return False
            inventory = json.loads(payload["text"])
            if not isinstance(inventory, list) or not inventory:
                return False
            inventory_paths = set()
            for item in inventory:
                relative = item["path"]
                path = PurePosixPath(relative)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or relative in inventory_paths
                    or files.get(relative) != item["sha256"]
                    or not _sha256_text(item["sha256"])
                    or type(item["size"]) is not int
                    or item["size"] < 0
                    or type(item["mode"]) is not int
                ):
                    return False
                inventory_paths.add(relative)
            owned_files.update(files)
            expected_facts.append([name, "10.1.0", digest])
        if sorted(facts["bundle_components"]) != sorted(expected_facts):
            return False
        loader = manifest["loader"]
        if set(loader["kpack"]) != {"gfx1250", "gfx942", "gfx950"}:
            return False
        for architecture, paths in loader["kpack"].items():
            device_files = distributions[f"phydrax-rocm-device-{architecture}"]["files"]
            if not paths or any(
                architecture not in path or path not in device_files for path in paths
            ):
                return False
        libraries = loader["libraries"]
        if not libraries or not loader["preload"]:
            return False
        if any(owned_files.get(item["path"]) != item["sha256"] for item in libraries):
            return False
        if not set(loader["preload"]) <= {item["path"] for item in libraries}:
            return False
        compiler = loader["compiler"]
        environment = diagnostics["automatic_runtime_environment"]
        if environment.get("LD_LIBRARY_PATH") is not None:
            return False
        for variable, key in (
            ("ROCM_PATH", "rocm_path"),
            ("LLVM_PATH", "llvm_path"),
            ("HIP_DEVICE_LIB_PATH", "bitcode_path"),
        ):
            relative = PurePosixPath(compiler[key])
            if relative.is_absolute() or ".." in relative.parts:
                return False
            if environment.get(variable) != str(site / relative):
                return False
        if compiler["lld_path"] not in owned_files:
            return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def probe_record(path: Path, accelerator: Accelerator) -> dict[str, Any] | None:
    """Load complete size-bounded evidence and validate its producer data contract."""
    try:
        record = read_json_evidence(path)
    except (OSError, ValueError):
        return None
    if set(record) != {"provenance", "runtime", "diagnostics"}:
        return None
    provenance, runtime, diagnostics = (
        record["provenance"],
        record["runtime"],
        record["diagnostics"],
    )
    if (
        not isinstance(provenance, dict)
        or not isinstance(runtime, dict)
        or not isinstance(diagnostics, dict)
    ):
        return None
    if set(provenance) != {
        "package",
        "venv",
        "distributions",
        "frontend_index_patch",
        "rocm_bundle",
    }:
        return None
    if any(
        not isinstance(provenance[key], str) or not provenance[key]
        for key in ("package", "venv")
    ):
        return None
    distributions = provenance["distributions"]
    if (
        not isinstance(distributions, dict)
        or not {"phydrax", *accelerator.distributions} <= distributions.keys()
    ):
        return None
    if any(
        not isinstance(name, str) or not _distribution_record_valid(item)
        for name, item in distributions.items()
    ):
        return None
    if any(
        distributions[name]["version"] != version
        for name, version in accelerator.distributions.items()
    ):
        return None
    patch = provenance["frontend_index_patch"]
    if not _frontend_patch_valid(patch) or patch != runtime.get("frontend_index_patch"):
        return None
    if patch is not None:
        jax_distribution = distributions.get("jax", {})
        if (
            patch["jax_version"] != jax_distribution.get("version")
            or patch["upstream_source_sha256"]
            != jax_distribution.get("files", {}).get("jax/_src/numpy/indexing.py")
            or patch["patch_source_sha256"]
            != distributions["phydrax"]["files"].get("phydrax/_jax_indexing_frontend.py")
        ):
            return None
    if accelerator is Accelerator.ROCM10 and any(
        name.startswith("jax-rocm") and name not in accelerator.distributions
        for name in distributions
    ):
        return None
    facts = diagnostics.get("facts")
    if (
        not isinstance(facts, dict)
        or facts.get("device_probed") is not True
        or not isinstance(diagnostics.get("issues"), list)
    ):
        return None
    if runtime_errors(runtime, accelerator):
        return None
    if (
        accelerator.rocm_major is not None
        and facts.get("selected_major") != accelerator.rocm_major
    ):
        return None
    if accelerator is Accelerator.ROCM10:
        if not _bundle_record_valid(provenance, diagnostics):
            return None
    elif provenance["rocm_bundle"] is not None:
        return None
    return record


def _copy_harness(destination: Path) -> None:
    for relative in (
        TEST_FILE,
        "tests/_support/accelerator_qualification.py",
        "tools/_accelerator_process.py",
        "tools/check_accelerator_extras.py",
        "tools/check_cuda_extras.py",
        EXAMPLE,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for package in ("tests", "tests/_support", "tests/integration", "tools", "examples"):
        (destination / package / "__init__.py").touch()


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    accelerator = Accelerator(args.accelerator)
    wheel = args.phydrax_wheel.resolve()
    report: dict[str, Any] = {
        "accelerator": accelerator.value,
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "stages": {},
        "static_architecture_coverage": [],
        "static_assets_are_hardware_qualification": False,
        "errors": [],
        "successful": False,
        "hardware_qualified": False,
    }
    errors: list[str] = report["errors"]
    errors.extend(host_errors(platform.system(), platform.machine()))
    metadata = wheel_metadata(wheel)
    errors.extend([metadata] if isinstance(metadata, str) else metadata_errors(metadata))
    work = args.work_dir.resolve()
    if work.is_relative_to(ROOT):
        errors.append("Qualification work directory must be outside the source checkout")
    if args.tpu_autodetect and accelerator is not Accelerator.TPU:
        errors.append("--tpu-autodetect requires the TPU selection")
    device_ids = [0] if args.device_id is None else args.device_id
    if any(device < 0 for device in device_ids) or len(set(device_ids)) != len(
        device_ids
    ):
        errors.append("Device IDs must be distinct nonnegative integers")
    report["selected_device_ids"] = device_ids
    if errors:
        return report
    work.mkdir(parents=True, exist_ok=False)
    venv = work / "venv"
    harness = work / "harness"
    harness.mkdir()
    _copy_harness(harness)
    report["harness_sha256"] = {
        str(path.relative_to(harness)): sha256(path)
        for path in sorted(harness.rglob("*.py"))
    }
    host_environment = clean_environment(venv, None)
    # Install-time index/cache configuration is not inherited by device children.
    install_environment = {
        **host_environment,
        "PATH": os.environ.get("PATH", host_environment["PATH"]),
    }
    for key, value in os.environ.items():
        if key.startswith("UV_CACHE"):
            install_environment[key] = value
    stages: dict[str, Any] = report["stages"]
    commands = (
        ("venv", [args.uv, "venv", "--python", str(args.python.resolve()), str(venv)]),
        (
            "install",
            [
                args.uv,
                "--no-config",
                "pip",
                "install",
                "--refresh",
                "--python",
                str(venv / "bin/python"),
                f"phydrax[{accelerator}] @ {wheel.as_uri()}",
                *(
                    value
                    for path in args.find_links
                    for value in ("--find-links", str(path.resolve()))
                ),
                *test_requirements(wheel),
            ],
        ),
    )
    for name, command in commands:
        stages[name] = run(
            command, cwd=harness, environment=install_environment, timeout=args.timeout
        )
        if stages[name]["returncode"] or stages[name]["timed_out"]:
            errors.append(f"{name} failed")
            return report
    expected = harness / "expected-core.json"
    expected.write_text(json.dumps(wheel_files(wheel), sort_keys=True))
    report["runtime_environment_overrides"] = {
        key: value
        for key, value in clean_environment(venv, accelerator).items()
        if key in ROCM_ENVIRONMENT
    }
    report["devices"] = [
        _qualify_device(args, accelerator, harness, venv, expected, device_id)
        for device_id in device_ids
    ]
    for device_report in report["devices"]:
        errors.extend(
            f"Device {device_report['selected_device_id']}: {error}"
            for error in device_report["errors"]
        )
        if accelerator is Accelerator.ROCM10 and "installed" in device_report:
            report["static_architecture_coverage"] = ["gfx1250", "gfx942", "gfx950"]
    report["successful"] = not errors
    report["hardware_qualified"] = not errors
    return report


def _qualify_device(
    args: argparse.Namespace,
    accelerator: Accelerator,
    harness: Path,
    venv: Path,
    expected: Path,
    device_id: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "selected_device_id": device_id,
        "stages": {},
        "errors": [],
        "successful": False,
        "hardware_qualified": False,
    }
    errors: list[str] = report["errors"]
    stages: dict[str, Any] = report["stages"]
    environment = clean_environment(venv, accelerator)
    environment["PHYDRAX_ACCELERATOR_DEVICE_ID"] = str(device_id)
    bootstrap = (
        "; phx.config.configure(phx.config.RuntimeConfig(bootstrap=phx.config.RuntimeBootstrap(autodetect=True)))"
        if args.tpu_autodetect
        else ""
    )
    probe_path = harness / f"installed-runtime-device-{device_id}.json"
    source = (
        "import sys,json\n"
        "from pathlib import Path\n"
        "if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3,13): raise RuntimeError('Qualification targets CPython 3.13')\n"
        f"sys.path.insert(0,{str(harness)!r})\n"
        f"import phydrax as phx{bootstrap}\n"
        "from tests._support.accelerator_qualification import accelerator_device,installed_provenance,runtime_evidence,diagnostic_evidence\n"
        "from tools._accelerator_process import write_json_evidence\n"
        "import jax,jax.numpy as jnp\n"
        f"provenance=installed_provenance(Path({str(venv)!r}),json.loads(Path({str(expected)!r}).read_text()))\n"
        "with phx.execution.default_device(accelerator_device()): runtime=runtime_evidence(jnp.asarray([1],dtype=jnp.int64))\n"
        "diagnostics=diagnostic_evidence()\n"
        f"write_json_evidence(Path({str(probe_path)!r}),dict(provenance=provenance,runtime=runtime,diagnostics=diagnostics))\n"
        "print(json.dumps(dict(status='observed')))"
    )
    python = str(venv / "bin/python")
    stages["installed-runtime"] = run(
        [python, "-I", "-c", source],
        cwd=harness,
        environment=environment,
        timeout=args.timeout,
    )
    probe = (
        probe_record(probe_path, accelerator)
        if stages["installed-runtime"]["returncode"] == 0
        and not stages["installed-runtime"]["timed_out"]
        else None
    )
    if (
        probe is None
        or not isinstance(probe.get("runtime"), dict)
        or not isinstance(probe.get("provenance"), dict)
    ):
        errors.append(
            "Installed wheel identity/device probe failed; driver or hardware unavailable is a refusal"
        )
        return report
    errors.extend(runtime_errors(probe["runtime"], accelerator))
    if probe["runtime"].get("qualified_device_id") != device_id:
        errors.append("Observed device differs from explicit device selection")
    report["installed"] = probe
    report["probe_sidecar"] = {
        "path": str(probe_path),
        "sha256": sha256(probe_path),
        "bytes": probe_path.stat().st_size,
    }
    if errors:
        return report
    # Bootstrap belongs to each fresh process, never to diagnostic collection.
    prefix = f"import phydrax as phx{bootstrap}; "
    cases_path = harness / f"cases-device-{device_id}.xml"
    case_selection = (
        [] if accelerator.rocm_major is not None else ["-k", "not test_rocm_"]
    )
    test_source = (
        prefix
        + f"import sys; sys.path.insert(0,{str(harness)!r}); import pytest; raise SystemExit(pytest.main([{TEST_FILE!r},'-q','-o','addopts=','-o','junit_family=xunit2','--junitxml={cases_path}',*{case_selection!r}]))"
    )
    stages["scientific-cases"] = run(
        [python, "-I", "-c", test_source],
        cwd=harness,
        environment=environment,
        timeout=args.timeout,
    )
    case_errors, cases = case_report_errors(cases_path, probe["runtime"])
    report["cases"] = cases
    errors.extend(case_errors)
    if (
        stages["scientific-cases"]["returncode"]
        or stages["scientific-cases"]["timed_out"]
    ):
        errors.append("Scientific cases failed or timed out")
    command = [
        python,
        "-I",
        str(harness / EXAMPLE),
        "--accelerator",
        accelerator.value,
        "--device-id",
        str(device_id),
    ]
    if args.tpu_autodetect:
        command.append("--tpu-autodetect")
    stages["example"] = run(
        command, cwd=harness, environment=environment, timeout=args.timeout
    )
    example = last_json(stages["example"])
    report["example"] = example
    if (
        example is None
        or example.get("status") != "passed"
        or example.get("results_on_device") is not True
        or "frontend_index_patch" not in example
        or any(
            example.get(key) != probe["runtime"].get(key)
            for key in (
                "platform",
                "platform_version",
                "device_kind",
                "qualified_device_id",
                "execution_device_count",
                "global_device_count",
                "local_device_count",
                "frontend_index_patch",
            )
        )
        or example.get("accelerator") != accelerator.value
    ):
        errors.append(
            "Advertised example failed or did not execute on the probed accelerator"
        )
    report["successful"] = not errors
    report["hardware_qualified"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", required=True, choices=tuple(Accelerator))
    parser.add_argument("--phydrax-wheel", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--device-id",
        type=int,
        action="append",
        help="Repeat for every local device ID to qualify; defaults to ID 0 only",
    )
    parser.add_argument(
        "--tpu-autodetect",
        action="store_true",
        help="Explicitly request existing JAX TPU launcher discovery in each child",
    )
    args = parser.parse_args()
    report = qualify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
