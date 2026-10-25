#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build and audit CUDA companion wheels using an already provisioned Python.

Example (development, not release architecture qualification)::

    python tools/build_cuda_wheels.py --architectures native \
        --components cudss --python /path/to/build-env/bin/python \
        --build-dir /path/to/build/cuda13 --wheelhouse /path/to/wheelhouse

Container builds use the host's manylinux_2_28_x86_64 or manylinux_2_28_aarch64
image. ``--python`` must name a provisioned interpreter usable *inside* that
image; this tool never installs dependencies or creates an environment. Host
environments are bind-mounted at their original
paths, but their interpreter and binary dependencies must be image-compatible.
The host repair tag reflects the host glibc, not a manylinux_2_28 release claim.
A JSON manifest is written beside the wheel, including on audit failure.

Native Windows x64 builds default to cuDSS/AmgX/QUDA/cuTENSOR and emit an
unrepaired ``win_amd64`` wheel. Set explicit ``PHX_WINDOWS_CC``, ``CXX``,
``CUDA_HOST_COMPILER``, ``NVCC``, ``CMAKE``, ``NINJA``, ``LIB``, ``MSVC_ROOT``,
``SDK_ROOT``, and ``SDK_VERSION`` variables (each with the PHX_WINDOWS_ prefix),
plus ``PHX_PATCH_EXECUTABLE``. The host compiler is clang-cl; nvcc uses MSVC cl.
``PHX_WINDOWS_CUOBJDUMP`` optionally selects the inspection executable.
All paths and temporary/cache outputs must remain under the user profile.
``--audit-wheel`` skips compilation and audits an existing wheel against its
original ``--build-dir`` metadata and link commands. Static audits do not load
the DLLs, initialize CUDA, or establish GPU/runtime qualification.
"""

from __future__ import annotations

import argparse
import base64
import email.parser
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from phydrax._pe_dependencies import PeImage


type Json = None | bool | int | float | str | Sequence[Json] | Mapping[str, Json]

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("cudss", "amgx", "quda", "cutensor", "cutensornet", "cuest")
WINDOWS_COMPONENTS = ("cudss", "amgx", "quda", "cutensor")
HOST_ONLY_COMPONENTS = frozenset(("cudss", "cutensor", "cutensornet", "cuest"))
DRIVER_LIBRARIES = frozenset(("libcuda.so.1", "libnvidia-ml.so.1"))
CLEAN_ENVIRONMENT = ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT")
RELEASE_ARCHITECTURES = (
    "80-real",
    "86-real",
    "89-real",
    "90-real",
    "100-real",
    "120-real",
    "120-virtual",
)


class Options(argparse.Namespace):
    architectures: str | None
    components: str
    python: Path
    wheelhouse: Path
    build_dir: Path
    download_dir: Path | None
    jobs: int
    container: str
    audit_wheel: Path | None


@dataclass(frozen=True)
class Configuration:
    architectures: tuple[str, ...]
    components: tuple[str, ...]
    python: Path
    wheelhouse: Path
    build_dir: Path
    download_dir: Path | None
    jobs: int
    container: str
    audit_wheel: Path | None = None


@dataclass(frozen=True)
class NvidiaInventory:
    versions: dict[str, str]
    libraries: dict[str, Path]
    executables: dict[str, Path]


@dataclass(frozen=True)
class Elf:
    needed: tuple[str, ...]
    runpath: tuple[str, ...]
    soname: str | None


class AuditFailure(RuntimeError):
    """A build artifact failed a packaging invariant."""


@dataclass(frozen=True)
class HostPlatform:
    architecture: str
    container_platform: str
    loader: str
    elf_machine: str

    @property
    def image(self) -> str:
        return f"quay.io/pypa/manylinux_2_28_{self.architecture}:latest"


def host_platform() -> HostPlatform:
    architecture = platform.machine()
    if sys.platform == "win32" and architecture.lower() in ("amd64", "x86_64"):
        return HostPlatform("x86_64", "", "", "")
    if sys.platform == "linux":
        match architecture:
            case "x86_64":
                return HostPlatform(
                    architecture,
                    "linux/amd64",
                    "ld-linux-x86-64.so.2",
                    "Advanced Micro Devices X86-64",
                )
            case "aarch64":
                return HostPlatform(
                    architecture, "linux/arm64", "ld-linux-aarch64.so.1", "AArch64"
                )
    raise AuditFailure(
        "wheel builds require Linux x86_64/aarch64 or Windows x86_64: "
        f"{sys.platform}/{architecture}"
    )


def parse_configuration(argv: Sequence[str] | None = None) -> Configuration:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        help="Semicolon-separated CMake targets, or native (development only). "
        "Default: 80-real;86-real;89-real;90-real;100-real;120-real;120-virtual.",
    )
    parser.add_argument(
        "--components",
        default=";".join(WINDOWS_COMPONENTS if sys.platform == "win32" else COMPONENTS),
        help="Semicolon- or comma-separated subset of "
        + ", ".join(COMPONENTS)
        + ". Windows default: "
        + ", ".join(WINDOWS_COMPONENTS)
        + ".",
    )
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Existing build-environment Python; no build isolation or installation.",
    )
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="CUDA 13 CMake build tree, reused incrementally.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Shared archive cache; defaults to <build-dir>/downloads.",
    )
    parser.add_argument(
        "--jobs", type=int, default=2, help="Maximum parallel compile jobs."
    )
    parser.add_argument(
        "--container", choices=("none", "podman", "docker"), default="none"
    )
    parser.add_argument(
        "--audit-wheel",
        type=Path,
        help="Audit an existing wheel using its original --build-dir; do not build.",
    )
    options = parser.parse_args(argv, namespace=Options())
    architectures = (
        tuple(options.architectures.split(";"))
        if options.architectures is not None
        else RELEASE_ARCHITECTURES
    )
    if architectures != ("native",) and any(
        re.fullmatch(r"[1-9][0-9]*[af]?(-(real|virtual))?", entry) is None
        for entry in architectures
    ):
        parser.error("--architectures must be native or a semicolon-separated CMake list")
    if len(set(architectures)) != len(architectures):
        parser.error("--architectures contains duplicate targets")
    selected = options.components.replace(",", ";").split(";")
    if not selected or any(entry not in COMPONENTS for entry in selected):
        parser.error("--components must select from " + ", ".join(COMPONENTS))
    if len(set(selected)) != len(selected):
        parser.error("--components contains duplicates")
    if sys.platform == "win32":
        unavailable = sorted(set(selected).difference(WINDOWS_COMPONENTS))
        if unavailable:
            parser.error("Windows components unavailable: " + ", ".join(unavailable))
        if options.container != "none":
            parser.error("Windows native builds do not support Linux containers")
    if options.audit_wheel is not None and options.container != "none":
        parser.error("--audit-wheel requires --container none")
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    return Configuration(
        architectures,
        tuple(component for component in COMPONENTS if component in selected),
        options.python.absolute(),
        options.wheelhouse.resolve(),
        options.build_dir.resolve(),
        options.download_dir.resolve() if options.download_dir is not None else None,
        options.jobs,
        options.container,
        options.audit_wheel.resolve() if options.audit_wheel is not None else None,
    )


def clean_environment(python: Path, jobs: int) -> dict[str, str]:
    environment = dict(os.environ)
    for name in CLEAN_ENVIRONMENT:
        environment.pop(name, None)
    # Never discover a system nvcc through PATH; CMake owns the wheel toolkit.
    paths = environment.get("PATH", os.defpath).split(os.pathsep)
    environment["PATH"] = os.pathsep.join(
        [str(python.parent)] + [entry for entry in paths if "cuda" not in entry.lower()]
    )
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    return environment


def profile_path(path: Path) -> Path:
    """Reject profile escapes, including existing junctions, before tool I/O."""
    resolved = path.resolve()
    if not resolved.is_relative_to(Path.home().resolve()):
        raise AuditFailure(f"Windows build path escapes the user profile: {path}")
    return resolved


def windows_environment(
    config: Configuration, environment: dict[str, str], evidence: dict[str, Json]
) -> None:
    for path in (
        ROOT,
        config.python,
        config.build_dir,
        config.wheelhouse,
        config.download_dir or config.build_dir / "downloads",
    ):
        profile_path(path)
    if config.audit_wheel is not None:
        profile_path(config.audit_wheel)
    temporary = profile_path(config.build_dir / "tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "UV_CACHE_DIR": str(config.build_dir / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(config.build_dir / "uv-python"),
            "CMAKE_GENERATOR": "Ninja",
        }
    )
    if config.audit_wheel is not None:
        return
    tool_paths: dict[str, Json] = {}
    for name in (
        "PHX_WINDOWS_CC",
        "PHX_WINDOWS_CXX",
        "PHX_WINDOWS_CUDA_HOST_COMPILER",
        "PHX_WINDOWS_NVCC",
        "PHX_WINDOWS_CMAKE",
        "PHX_WINDOWS_NINJA",
        "PHX_WINDOWS_LIB",
        "PHX_PATCH_EXECUTABLE",
    ):
        value = environment.get(name)
        if not value or not Path(value).is_absolute():
            raise AuditFailure(f"{name} must name an explicit absolute executable")
        path = profile_path(Path(value))
        if not path.is_file():
            raise AuditFailure(f"{name} executable does not exist: {path}")
        tool_paths[name] = {"path": str(path), "sha256": file_hash(path)}
        environment[name] = str(path)
    for name in ("PHX_WINDOWS_MSVC_ROOT", "PHX_WINDOWS_SDK_ROOT"):
        value = environment.get(name)
        if not value or not Path(value).is_absolute():
            raise AuditFailure(f"{name} must name an explicit absolute directory")
        path = profile_path(Path(value))
        if not path.is_dir():
            raise AuditFailure(f"{name} directory does not exist: {path}")
        environment[name] = str(path)
    if not environment.get("PHX_WINDOWS_SDK_VERSION"):
        raise AuditFailure("PHX_WINDOWS_SDK_VERSION must select the Windows SDK")
    sdk_version = environment["PHX_WINDOWS_SDK_VERSION"]
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", sdk_version) is None:
        raise AuditFailure("PHX_WINDOWS_SDK_VERSION must be a numeric SDK version")
    msvc = Path(environment["PHX_WINDOWS_MSVC_ROOT"])
    sdk = Path(environment["PHX_WINDOWS_SDK_ROOT"])
    includes = [msvc / "include"] + [
        sdk / "Include" / sdk_version / name for name in ("ucrt", "shared", "um", "winrt")
    ]
    libraries = [
        msvc / "lib/x64",
        sdk / "Lib" / sdk_version / "ucrt/x64",
        sdk / "Lib" / sdk_version / "um/x64",
    ]
    tools = [
        Path(environment[name]).parent
        for name in (
            "PHX_WINDOWS_CC",
            "PHX_WINDOWS_CXX",
            "PHX_WINDOWS_CUDA_HOST_COMPILER",
            "PHX_WINDOWS_NINJA",
            "PHX_WINDOWS_LIB",
            "PHX_PATCH_EXECUTABLE",
        )
    ] + [sdk / "bin" / sdk_version / "x64"]
    for path in (*includes, *libraries, *tools):
        if not profile_path(path).is_dir():
            raise AuditFailure(f"missing explicit Windows toolchain directory: {path}")
    environment["INCLUDE"] = ";".join(str(path) for path in includes)
    environment["LIB"] = ";".join(str(path) for path in libraries)
    environment["PATH"] = os.pathsep.join(
        [str(path) for path in dict.fromkeys(tools)] + [environment["PATH"]]
    )
    environment["CMAKE_EXECUTABLE"] = environment["PHX_WINDOWS_CMAKE"]
    evidence["windows_toolchain"] = tool_paths


def pe_reader() -> ModuleType:
    """Import the canonical static PE reader without initializing Phydrax."""
    name = "_phydrax_cuda_wheel_pe"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "phydrax/_pe_dependencies.py"
        )
        if spec is None or spec.loader is None:
            raise AuditFailure("canonical PE reader is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def run(
    command: Sequence[str],
    environment: Mapping[str, str],
    evidence: dict[str, Json],
    *,
    cwd: Path = ROOT,
    save_output: bool = True,
    expected_failure: tuple[int, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    record: dict[str, Json] = {
        "command": list(command),
        "returncode": result.returncode,
        "stderr": result.stderr,
    }
    if save_output:
        record["stdout"] = result.stdout
    else:
        encoded = result.stdout.encode("utf-8")
        record["stdout_bytes"] = len(encoded)
        record["stdout_sha256"] = hashlib.sha256(encoded).hexdigest()
    commands = evidence.setdefault("commands", [])
    if not isinstance(commands, list):
        raise TypeError("manifest commands must be a list")
    commands.append(record)
    if result.stdout and save_output:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        diagnostic = (result.stdout + result.stderr).strip()
        if expected_failure is not None and (
            result.returncode == expected_failure[0]
            and diagnostic.split() == expected_failure[1].split()
        ):
            record["accepted_diagnostic"] = diagnostic
            return expected_failure[1]
        raise AuditFailure(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout


def container_command(config: Configuration) -> list[str]:
    if os.name != "posix":
        raise AuditFailure("Container wheel builds require a POSIX host.")
    host = host_platform()
    command = [config.container, "run", "--rm", "--platform", host.container_platform]
    if config.container == "podman":
        command.append("--userns=keep-id")
    else:
        command.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
    config.build_dir.mkdir(parents=True, exist_ok=True)
    config.wheelhouse.mkdir(parents=True, exist_ok=True)
    mounts = {ROOT, config.build_dir, config.wheelhouse}
    if config.download_dir is not None:
        config.download_dir.mkdir(parents=True, exist_ok=True)
        mounts.add(config.download_dir)
    prefix = config.python.parent.parent
    if prefix.is_dir() and not any(prefix.is_relative_to(path) for path in mounts):
        mounts.add(prefix)
    for path in sorted(mounts):
        command.extend(("--volume", f"{path}:{path}"))
    command.extend(("--workdir", str(ROOT)))
    if config.architectures == ("native",):
        if config.container == "docker":
            command.extend(("--gpus", "all"))
        else:
            command.extend(("--device", "nvidia.com/gpu=all"))
    command.extend(
        (
            host.image,
            str(config.python),
            str(Path(__file__).resolve()),
            "--architectures",
            ";".join(config.architectures),
            "--components",
            ";".join(config.components),
            "--python",
            str(config.python),
            "--wheelhouse",
            str(config.wheelhouse),
            "--build-dir",
            str(config.build_dir),
            "--jobs",
            str(config.jobs),
            "--container",
            "none",
        )
    )
    if config.download_dir is not None:
        command.extend(("--download-dir", str(config.download_dir)))
    return command


def json_value(value: object) -> Json:
    """Validate the dynamic JSON interchange boundary once."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): json_value(item) for key, item in value.items()}
    raise TypeError("manifest input is not JSON")


def read_json(path: Path) -> dict[str, Json]:
    value = json_value(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def inventory() -> NvidiaInventory:
    versions: dict[str, str] = {}
    libraries: dict[str, Path] = {}
    executables: dict[str, Path] = {}
    for distribution in sorted(
        importlib.metadata.distributions(),
        key=lambda item: item.metadata["Name"] or "",
    ):
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"] or "").lower()
        if not (
            name.startswith("nvidia-")
            or name == "phydrax-cuest"
            or re.fullmatch(r"(?:cutensor|cutensornet)-cu\d+", name)
        ):
            continue
        suffix = re.search(r"-cu(\d+)$", name)
        if suffix is not None and suffix.group(1) != "13":
            raise AuditFailure(
                f"foreign NVIDIA distribution in build environment: {name}"
            )
        versions[name] = distribution.version
        files = distribution.files
        if files is None:
            raise AuditFailure(f"NVIDIA distribution has no RECORD: {name}")
        for member in files:
            path = Path(str(distribution.locate_file(member))).absolute()
            if sys.platform == "win32":
                basename = path.name.lower()
                if not (
                    basename.endswith(".dll") or basename in ("nvcc.exe", "cuobjdump.exe")
                ):
                    continue
                path = profile_path(path)
                digest = member.hash
                if digest is None or digest.mode != "sha256" or member.size is None:
                    raise AuditFailure(f"{name}: unhashed NVIDIA RECORD asset {member}")
                actual = (
                    base64.urlsafe_b64encode(bytes.fromhex(file_hash(path)))
                    .rstrip(b"=")
                    .decode("ascii")
                )
                if actual != digest.value or path.stat().st_size != member.size:
                    raise AuditFailure(f"{name}: NVIDIA RECORD mismatch: {member}")
                if basename.endswith(".dll"):
                    previous = libraries.get(basename)
                    if previous is not None and previous != path:
                        raise AuditFailure(f"ambiguous NVIDIA library {basename}")
                    libraries[basename] = path
                else:
                    tool = basename.removesuffix(".exe")
                    if tool in executables and executables[tool] != path:
                        raise AuditFailure(f"ambiguous NVIDIA executable {basename}")
                    executables[tool] = path
                continue
            if re.search(r"\.so(?:\.|$)", path.name) and path.is_file():
                previous = libraries.get(path.name)
                if previous is not None and previous.resolve() != path.resolve():
                    raise AuditFailure(f"ambiguous NVIDIA library {path.name}")
                libraries[path.name] = path
            if path.name in ("nvcc", "cuobjdump") and path.is_file():
                executables[path.name] = path
    if not libraries:
        raise AuditFailure("build environment contains no NVIDIA wheel libraries")
    return NvidiaInventory(versions, libraries, executables)


def manylinux_policy() -> tuple[str, frozenset[str]]:
    distribution = importlib.metadata.distribution("auditwheel")
    path = Path(str(distribution.locate_file("auditwheel/policy/manylinux-policy.json")))
    policies = json_value(json.loads(path.read_text(encoding="utf-8")))
    architecture = host_platform()
    libc, version = platform.libc_ver()
    if libc != "glibc":
        raise AuditFailure("wheel builds require glibc Linux x86_64 or aarch64")
    host = tuple(int(part) for part in version.split(".")[:2])
    candidates: list[tuple[tuple[int, int], str, frozenset[str]]] = []
    if not isinstance(policies, list):
        raise ValueError("auditwheel policy must be a list")
    for policy in policies:
        if not isinstance(policy, dict):
            raise ValueError("invalid auditwheel policy entry")
        name = policy.get("name")
        whitelist = policy.get("lib_whitelist")
        if not isinstance(name, str) or not isinstance(whitelist, list):
            raise ValueError("auditwheel policy lacks name/lib_whitelist")
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)", name)
        if match is None:
            continue
        minimum = (int(match.group(1)), int(match.group(2)))
        symbols = policy.get("symbol_versions")
        if not isinstance(symbols, dict):
            raise ValueError("auditwheel policy lacks architecture symbol versions")
        if architecture.architecture not in symbols:
            continue
        if (2, 28) <= minimum <= host:
            if not all(isinstance(item, str) for item in whitelist):
                raise ValueError("invalid auditwheel system-library whitelist")
            candidates.append((minimum, name, frozenset(str(item) for item in whitelist)))
    if not candidates:
        raise AuditFailure(f"no supported manylinux policy for glibc {version}")
    _, name, allowed = max(candidates)
    # auditwheel's policy additionally exempts the ELF loader, outside its JSON
    # whitelist (WheelPolicies.lddtree_external_references).
    return f"{name}_{architecture.architecture}", allowed.union((architecture.loader,))


def inspect_elf(
    path: Path,
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> Elf:
    output = run(
        ("readelf", "--file-header", "--dynamic", "--wide", str(path)),
        environment,
        evidence,
    )
    machine = re.search(r"^\s*Machine:\s*(.+?)\s*$", output, re.MULTILINE)
    elf_class = re.search(r"^\s*Class:\s*(\S+)", output, re.MULTILINE)
    data = re.search(r"^\s*Data:\s*(.+?)\s*$", output, re.MULTILINE)
    expected_machine = host_platform().elf_machine
    if (
        machine is None
        or machine[1] != expected_machine
        or elf_class is None
        or elf_class[1] != "ELF64"
        or data is None
        or data[1] != "2's complement, little endian"
    ):
        raise AuditFailure(
            f"{path.name}: expected little-endian ELF64 {expected_machine}"
        )
    needed = tuple(re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", output))
    paths = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^\]]*)\]", output)
    runpath = tuple(entry for value in paths for entry in value.split(":"))
    sonames = re.findall(r"\(SONAME\).*?\[([^\]]+)\]", output)
    return Elf(needed, runpath, sonames[0] if sonames else None)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_metadata(config: Configuration) -> dict[str, Json]:
    metadata = read_json(config.build_dir / "phx_cuda_build.json")
    build_hash = metadata.get("build_hash")
    if (
        not isinstance(build_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", build_hash) is None
    ):
        raise AuditFailure("CMake build metadata lacks a SHA256 build_hash")
    for key in ("toolkit_wheels", "upstream_pins", "patch_hashes"):
        if not isinstance(metadata.get(key), dict):
            raise AuditFailure(f"CMake build metadata lacks {key}")
    if sys.platform == "win32":
        source = ROOT / "native/cuda"
        files = sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file()
                and (
                    path.suffix
                    in (
                        ".cpp",
                        ".cu",
                        ".h",
                        ".hpp",
                        ".cuh",
                        ".cmake",
                        ".patch",
                        ".map",
                        ".py",
                    )
                    or path.name.endswith("CMakeLists.txt")
                )
            ),
            key=lambda path: path.relative_to(source).as_posix(),
        )
        identity = "".join(
            f"{path.relative_to(source).as_posix()}:{file_hash(path)}\n" for path in files
        )
        if hashlib.sha256(identity.encode("utf-8")).hexdigest() != build_hash:
            raise AuditFailure(
                "CMake build hash does not match the current native sources"
            )
    patch_hashes: dict[str, Json] = {
        path.relative_to(ROOT).as_posix(): file_hash(path)
        for path in sorted((ROOT / "native/cuda").rglob("*.patch"))
    }
    metadata["source_patch_hashes"] = patch_hashes
    recipes: dict[str, Json] = {}
    for path in sorted((ROOT / "native/cuda/cmake").glob("*.cmake")):
        source = path.read_text(encoding="utf-8")
        urls = sorted(set(re.findall(r'https://[^\s")]+', source)))
        hashes = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", source)))
        if urls or hashes:
            recipes[path.relative_to(ROOT).as_posix()] = {
                "sha256": file_hash(path),
                "archive_urls": urls,
                "archive_sha256": hashes,
            }
    metadata["upstream_pin_recipes"] = recipes
    return metadata


def locate_cuda_tool(name: str, config: Configuration, wheels: NvidiaInventory) -> Path:
    if sys.platform == "win32":
        discovery = read_json(config.build_dir / "pip-cuda/discovery.json")
        executables = discovery.get("executables")
        toolchain = discovery.get("toolchain")
        if not isinstance(executables, dict) or not isinstance(toolchain, dict):
            raise AuditFailure("Windows toolkit discovery lacks executable identity")
        recorded = toolchain.get(
            "cuobjdump" if name == "cuobjdump" else "PHX_WINDOWS_NVCC"
        )
        selected = os.environ.get("PHX_WINDOWS_" + name.upper()) or executables.get(name)
        if not isinstance(recorded, dict) or not isinstance(selected, str):
            raise AuditFailure(f"Windows toolkit discovery lacks {name}")
        expected_hash = recorded.get("sha256")
        candidate = Path(selected)
        if not candidate.is_absolute():
            raise AuditFailure(f"Windows {name} must have an absolute path")
        candidate = profile_path(candidate)
        if not candidate.is_file() or file_hash(candidate) != expected_hash:
            raise AuditFailure(
                f"Windows {name} differs from the recorded toolkit: {candidate}"
            )
        return candidate
    recorded = wheels.executables.get(name)
    if recorded is not None:
        return recorded
    candidates = sorted(config.build_dir.glob(f"**/bin/{name}"))
    if name == "cuobjdump":
        # System CUDA is permitted ONLY for binary inspection, never compilation.
        found = shutil.which(name)
        if found is not None:
            candidates.append(Path(found))
        candidates.extend(sorted(Path("/usr/local").glob("cuda*/bin/cuobjdump")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise AuditFailure(
        f"missing {name}: provide it in the NVIDIA wheel RECORD or assembled toolkit"
        + (
            " (a system cuobjdump is also permitted for inspection)"
            if name == "cuobjdump"
            else ""
        )
    )


def verify_shared_cudart(
    path: Path,
    elf: Elf,
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    runtime = "libcudart.so.13"
    cudarts = [name for name in elf.needed if name.startswith("libcudart.so")]
    if not cudarts or any(name != runtime for name in cudarts):
        raise AuditFailure(f"{path.name}: expected NEEDED {runtime}, found {cudarts}")
    symbols = run(
        ("readelf", "--symbols", "--wide", str(path)),
        environment,
        evidence,
        save_output=False,
    )
    if "Symbol table '.symtab'" not in symbols:
        raise AuditFailure(
            f"{path.name}: stripped symbol table prevents static-cudart audit"
        )
    static_definitions: list[str] = []
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":") or fields[6] == "UND":
            continue
        symbol = fields[7].split("@")[0]
        if re.fullmatch(
            r"cuda[A-Z]\w*|__cudaRegisterFatBinary|__cudaUnregisterFatBinary", symbol
        ):
            static_definitions.append(symbol)
        if "__cudart" in symbol:
            static_definitions.append(symbol)
    if static_definitions:
        raise AuditFailure(
            f"{path.name}: static cudart definitions: {static_definitions}"
        )
    exported = re.search(
        r"GLOBAL\s+DEFAULT\s+\d+\s+phx_cuda_build_hash(?:@@\S+)?(?:\s|$)",
        symbols,
    )
    if exported is None:
        raise AuditFailure(f"{path.name}: missing exported phx_cuda_build_hash")
    return {"needed": runtime, "defined_runtime_symbols": static_definitions}


def archive_link_evidence(config: Configuration) -> dict[str, Json]:
    paths = sorted(config.build_dir.glob("**/link.txt"))
    paths.extend(sorted(config.build_dir.glob("**/build.ninja")))
    if not paths:
        raise AuditFailure("no link-command evidence (build.ninja or link.txt)")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "cudart_static" in line.lower() and (
                path.name == "link.txt"
                or "LINK_LIBRARIES =" in line
                or "LINK_FLAGS =" in line
            ):
                raise AuditFailure(
                    f"static cudart archive in link command: {path}: {line}"
                )
            if sys.platform == "win32" and re.search(r"(?:^|\s)[/-]MTd?(?:\s|$)", line):
                raise AuditFailure(f"static MSVC CRT in build command: {path}: {line}")
    return {
        "files": [str(path) for path in paths],
        "sha256": {str(path): file_hash(path) for path in paths},
        "static_cudart_archive": False,
    }


def inspect_device_code(
    library: Path,
    cuobjdump: Path,
    config: Configuration,
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    host_only_allowed = library.name in {
        filename
        for component in HOST_ONLY_COMPONENTS
        for filename in (
            f"libphydrax_cuda_{component}.so",
            f"phydrax_cuda_{component}.dll",
        )
    }
    no_code_diagnostic = f"cuobjdump info : File '{library}' does not contain device code"
    no_code_exit = 0xFFFFFFFF if sys.platform == "win32" else 255
    expected_failure = (no_code_exit, no_code_diagnostic) if host_only_allowed else None
    sass = run(
        (str(cuobjdump), "--list-elf", str(library)),
        environment,
        evidence,
        expected_failure=expected_failure,
    )
    ptx = run(
        (str(cuobjdump), "--list-ptx", str(library)),
        environment,
        evidence,
        expected_failure=expected_failure,
    )
    sass_targets = sorted(set(re.findall(r"sm_(\d+[af]?)", sass)))
    ptx_targets = sorted(set(re.findall(r"(?:sm|compute)_(\d+[af]?)", ptx)))
    device_code_absent = not sass_targets and not ptx_targets
    if device_code_absent and not host_only_allowed:
        raise AuditFailure(
            f"{library.name}: component unexpectedly contains no device code"
        )
    if (
        sass == no_code_diagnostic or ptx == no_code_diagnostic
    ) and not device_code_absent:
        raise AuditFailure(f"{library.name}: inconsistent cuobjdump device-code evidence")
    requested_sass: set[str] = set()
    requested_ptx: set[str] = set()
    if config.architectures != ("native",):
        for entry in config.architectures:
            number, _, kind = entry.partition("-")
            if kind != "virtual":
                requested_sass.add(number)
            if kind != "real":
                requested_ptx.add(number)
        # A host-only component has no code targets to qualify. Its NVIDIA
        # provider's architecture support is not established by this wheel.
        if sass_targets or ptx_targets:
            missing_sass = requested_sass.difference(sass_targets)
            missing_ptx = requested_ptx.difference(ptx_targets)
            if missing_sass or missing_ptx:
                raise AuditFailure(
                    f"{library.name}: missing SASS {sorted(missing_sass)} / "
                    f"PTX {sorted(missing_ptx)}"
                )
    release_sass = {
        entry.removesuffix("-real")
        for entry in RELEASE_ARCHITECTURES
        if entry.endswith("-real")
    }
    release_ptx = {
        entry.removesuffix("-virtual")
        for entry in RELEASE_ARCHITECTURES
        if entry.endswith("-virtual")
    }
    return {
        "sass": sass_targets,
        "ptx": ptx_targets,
        "list_elf": sass,
        "list_ptx": ptx,
        "device_code_absent": device_code_absent,
        "host_only_component": device_code_absent and host_only_allowed,
        "release_architecture_coverage": release_sass.issubset(sass_targets)
        and release_ptx.issubset(ptx_targets),
    }


def verify_dependencies(
    path: Path,
    elf: Elf,
    wheel_root: Path,
    wheel_libraries: Mapping[str, Path],
    nvidia: NvidiaInventory,
    system_libraries: frozenset[str],
) -> dict[str, Json]:
    for entry in elf.runpath:
        if not (entry == "$ORIGIN" or entry.startswith("$ORIGIN/")):
            raise AuditFailure(f"{path.name}: non-$ORIGIN RUNPATH/RPATH {entry!r}")
        if "$" in entry.removeprefix("$ORIGIN"):
            raise AuditFailure(f"{path.name}: unsupported RUNPATH token {entry!r}")
    resolutions: dict[str, Json] = {}
    for needed in elf.needed:
        if "/" in needed:
            raise AuditFailure(f"{path.name}: absolute/path-bearing NEEDED {needed}")
        bundled = wheel_libraries.get(needed)
        provider = nvidia.libraries.get(needed)
        if bundled is not None:
            candidates = [
                (path.parent / entry.removeprefix("$ORIGIN/")).resolve()
                if entry != "$ORIGIN"
                else path.parent.resolve()
                for entry in elf.runpath
            ]
            if bundled.parent.resolve() not in candidates:
                raise AuditFailure(
                    f"{path.name}: bundled {needed} is not RUNPATH-reachable"
                )
            resolutions[needed] = bundled.relative_to(wheel_root).as_posix()
        elif provider is not None:
            # Evaluate NVIDIA paths in the eventual site-packages layout, not
            # against the temporary extraction directory. Loader preloading is
            # an additional guarantee, not a substitute for correct RUNPATH.
            relative = path.relative_to(wheel_root)
            site = Path(sysconfig.get_path("platlib"))
            origin = (site / relative).parent
            candidates = [
                origin
                if entry == "$ORIGIN"
                else (origin / entry.removeprefix("$ORIGIN/")).resolve()
                for entry in elf.runpath
            ]
            if provider.parent.resolve() not in candidates:
                raise AuditFailure(
                    f"{path.name}: NVIDIA {needed} is not RUNPATH-reachable"
                )
            resolutions[needed] = str(provider)
        elif needed in system_libraries or needed in DRIVER_LIBRARIES:
            resolutions[needed] = "platform" if needed in system_libraries else "driver"
        else:
            raise AuditFailure(f"{path.name}: unresolved/non-policy NEEDED {needed}")
    return resolutions


def verify_nvidia_closure(
    roots: Sequence[Elf],
    nvidia: NvidiaInventory,
    wheel_libraries: Mapping[str, Path],
    system_libraries: frozenset[str],
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    """Audit transitive providers without loading CUDA or consulting system CUDA."""
    pending = sorted({name for elf in roots for name in elf.needed})
    visited: set[str] = set()
    closure: dict[str, Json] = {}
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if (
            name in wheel_libraries
            or name in system_libraries
            or name in DRIVER_LIBRARIES
        ):
            continue
        provider = nvidia.libraries.get(name)
        if provider is None or "/" in name:
            raise AuditFailure(
                f"unresolved/non-policy transitive NVIDIA dependency: {name}"
            )
        elf = inspect_elf(provider, environment, evidence)
        for entry in elf.runpath:
            if not (entry == "$ORIGIN" or entry.startswith("$ORIGIN/")):
                raise AuditFailure(f"{name}: non-$ORIGIN NVIDIA RUNPATH/RPATH {entry!r}")
            if "$" in entry.removeprefix("$ORIGIN"):
                raise AuditFailure(f"{name}: unsupported NVIDIA RUNPATH token {entry!r}")
        closure[name] = {"path": str(provider), "needed": list(elf.needed)}
        pending.extend(elf.needed)
    return closure


def verify_wheel(
    wheel: Path,
    extraction: Path,
    config: Configuration,
    nvidia: NvidiaInventory,
    system_libraries: frozenset[str],
    metadata: dict[str, Json],
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    sizes: dict[str, Json] = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise AuditFailure(f"unsafe wheel member: {member.filename}")
            sizes[member.filename] = {
                "compressed": member.compress_size,
                "uncompressed": member.file_size,
            }
        archive.extractall(extraction)
    libraries: dict[str, Path] = {}
    elfs: dict[Path, Elf] = {}
    for path in sorted(extraction.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            is_elf = stream.read(4) == b"\x7fELF"
        if not is_elf:
            continue
        elf = inspect_elf(path, environment, evidence)
        names = (path.name, elf.soname or "")
        forbidden = set(nvidia.libraries).union(DRIVER_LIBRARIES)
        if any(
            name == original or name.startswith(original.partition(".so")[0] + "-")
            for name in names
            for original in forbidden
        ):
            raise AuditFailure(f"NVIDIA/driver library bundled in wheel: {path.name}")
        for name in (path.name, elf.soname):
            if name is not None:
                if name in libraries and libraries[name] != path:
                    raise AuditFailure(f"ambiguous wheel library {name}")
                libraries[name] = path
        elfs[path] = elf
    dependencies: dict[str, Json] = {}
    for path, elf in elfs.items():
        dependencies[path.relative_to(extraction).as_posix()] = verify_dependencies(
            path,
            elf,
            extraction,
            libraries,
            nvidia,
            system_libraries,
        )
    nvidia_closure = verify_nvidia_closure(
        list(elfs.values()),
        nvidia,
        libraries,
        system_libraries,
        environment,
        evidence,
    )
    cuobjdump = locate_cuda_tool("cuobjdump", config, nvidia)
    components: dict[str, Json] = {}
    build_hash = metadata["build_hash"]
    if not isinstance(build_hash, str):
        raise TypeError("validated build hash must be a string")
    for component in config.components:
        path = extraction / f"phydrax_cuda13/lib/libphydrax_cuda_{component}.so"
        elf = elfs.get(path)
        if elf is None:
            raise AuditFailure(f"wheel lacks component library: {path.name}")
        if build_hash.encode("ascii") + b"\0" not in path.read_bytes():
            raise AuditFailure(f"{path.name}: CMake build hash absent from library")
        components[component] = {
            "sha256": file_hash(path),
            "build_hash": build_hash,
            "shared_cudart": verify_shared_cudart(path, elf, environment, evidence),
            "device_code": inspect_device_code(
                path, cuobjdump, config, environment, evidence
            ),
        }
    expected = {f"libphydrax_cuda_{component}.so" for component in config.components}
    actual = {path.name for path in elfs if path.name.startswith("libphydrax_cuda_")}
    if actual != expected:
        raise AuditFailure(
            f"wheel component set mismatch: {sorted(actual)} != {sorted(expected)}"
        )
    if config.components == ("cudss",) and wheel.stat().st_size >= 100_000_000:
        raise AuditFailure("cuDSS-only wheel must be smaller than 100 MB")
    return {
        "filename": wheel.name,
        "sha256": file_hash(wheel),
        "compressed_bytes": wheel.stat().st_size,
        "members": sizes,
        "dependencies": dependencies,
        "components": components,
        "nvidia_dependency_closure": nvidia_closure,
        "archive_link_evidence": archive_link_evidence(config),
        "cuobjdump": str(cuobjdump),
        "cuobjdump_version": run((str(cuobjdump), "--version"), environment, evidence),
    }


def expected_windows_exports(component: str) -> frozenset[str]:
    """Use the public header and component's declared handlers as ABI owners."""
    header = (ROOT / "native/cuda/include/phydrax_cuda.h").read_text(encoding="utf-8")
    public = set(re.findall(r"\b(phx_cuda_\w+)\s*\(", header))
    recipe = (ROOT / f"native/cuda/src/{component}/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    handlers = re.search(r"\bHANDLERS\s+([^)]+)", recipe)
    if not public or handlers is None:
        raise AuditFailure(f"{component}: public export declarations are missing")
    names = handlers[1].split()
    if not names or any(
        re.fullmatch(rf"phydrax_{component}_\w+", name) is None for name in names
    ):
        raise AuditFailure(f"{component}: invalid declared handler list")
    return frozenset(public.union(names))


def verify_pe_exports(path: Path, image: PeImage, component: str) -> list[str]:
    expected = expected_windows_exports(component)
    names: list[str] = []
    for export in image.exports:
        if export.name is None or export.forwarder is not None or not export.rva:
            raise AuditFailure(f"{path.name}: ordinal-only, forwarded, or null export")
        names.append(export.name)
    if len(set(names)) != len(names) or set(names) != expected:
        raise AuditFailure(
            f"{path.name}: export mismatch: missing {sorted(expected.difference(names))}; "
            f"unexpected {sorted(set(names).difference(expected))}"
        )
    return sorted(names)


def pe_imports(image: PeImage) -> tuple[str, ...]:
    names = set(image.imports).union(image.delay_imports)
    for export in image.exports:
        if export.forwarder is not None:
            library, separator, symbol = export.forwarder.rpartition(".")
            if not separator or not library or not symbol:
                raise AuditFailure(f"invalid PE export forwarder: {export.forwarder}")
            names.add(library if library.lower().endswith(".dll") else library + ".dll")
    normalized: set[str] = set()
    for name in names:
        if (
            not name
            or "/" in name
            or "\\" in name
            or ":" in name
            or not name.lower().endswith(".dll")
        ):
            raise AuditFailure(f"path-bearing or invalid PE import: {name!r}")
        normalized.add(name.lower())
    return tuple(sorted(normalized))


def verify_pe_shared_cudart(
    path: Path, image: PeImage, metadata: dict[str, Json], config: Configuration
) -> str:
    cudarts = [name for name in pe_imports(image) if name.startswith("cudart")]
    linkage = metadata.get("runtime_linkage")
    if linkage == {
        "kind": "driver-hybrid",
        "target_dll": "nvcudart_hybrid64.dll",
        "owner": "display-driver",
    }:
        if cudarts:
            raise AuditFailure(f"{path.name}: mixed standalone and hybrid CUDA runtimes")
        toolchain = metadata.get("toolchain")
        loader = (
            toolchain.get("cudart_link_library") if isinstance(toolchain, dict) else None
        )
        if not isinstance(loader, dict) or loader.get("kind") != "driver-hybrid":
            raise AuditFailure(
                f"{path.name}: missing authenticated hybrid loader identity"
            )
        link_map = path.with_suffix(".map")
        if file_hash(link_map) != file_hash(config.build_dir / link_map.name):
            raise AuditFailure(
                f"{path.name}: installed link map differs from the actual build"
            )
        pe_reader().inspect_cuda_runtime_linkage(path, link_map, linkage)
        return "nvcudart_hybrid64.dll"
    if linkage != {
        "kind": "dll-import",
        "target_dll": "cudart64_13.dll",
        "owner": "nvidia-cuda-runtime",
    } or cudarts != ["cudart64_13.dll"]:
        raise AuditFailure(
            f"{path.name}: expected recorded shared CUDA runtime linkage, found {linkage}, {cudarts}"
        )
    return cudarts[0]


def verify_pe_closure(
    roots: Mapping[str, PeImage], nvidia: NvidiaInventory
) -> dict[str, Json]:
    reader = pe_reader()
    pending = sorted({name for image in roots.values() for name in pe_imports(image)})
    closure: dict[str, Json] = {}
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        if name in roots:
            closure[name] = {"owner": "companion"}
            continue
        if reader.is_system_import(name):
            closure[name] = {
                "owner": "driver" if name in ("nvcuda.dll", "nvml.dll") else "platform",
                "runtime_availability_observed": False,
            }
            continue
        path = nvidia.libraries.get(name)
        if path is None:
            raise AuditFailure(f"unresolved/non-policy PE import: {name}")
        image = reader.inspect_pe(path)
        if image.machine != 0x8664:
            raise AuditFailure(f"{name}: expected PE x86-64")
        imports = pe_imports(image)
        closure[name] = {
            "owner": "NVIDIA wheel",
            "path": str(path),
            "sha256": file_hash(path),
            "imports": list(image.imports),
            "delay_imports": list(image.delay_imports),
            "closure_imports": list(imports),
        }
        pending.extend(imports)
    return dict(sorted(closure.items()))


def verify_windows_identity(
    extraction: Path, config: Configuration, nvidia: NvidiaInventory
) -> dict[str, Json]:
    original = read_json(config.build_dir / "phx_cuda_build.json")
    installed = read_json(extraction / "phydrax_cuda13/phx_cuda_build.json")
    if installed != original:
        raise AuditFailure("installed build identity differs from the CMake build")
    expected: dict[str, Json] = {
        "components": list(config.components),
        "architectures": list(config.architectures),
        "host_system": "Windows",
        "cuda_runtime": "Shared",
        "msvc_runtime": "MultiThreadedDLL",
    }
    for key, value in expected.items():
        if installed.get(key) != value:
            raise AuditFailure(f"Windows build identity {key}: expected {value!r}")
    architecture = installed.get("host_architecture")
    if not isinstance(architecture, str) or architecture.lower() not in (
        "amd64",
        "x86_64",
    ):
        raise AuditFailure("Windows build identity must declare x86-64")
    for key, prefix in (("cxx_compiler", "Clang "), ("cuda_compiler", "NVIDIA ")):
        compiler = installed.get(key)
        if not isinstance(compiler, str) or not compiler.startswith(prefix):
            raise AuditFailure(
                f"Windows build identity requires {prefix.strip()} for {key}"
            )
    cuda_host = installed.get("cuda_host_compiler")
    if not isinstance(cuda_host, str) or Path(cuda_host).name.lower() != "cl.exe":
        raise AuditFailure("Windows nvcc must use the recorded MSVC cl.exe host compiler")
    if (
        installed.get("redistribution")
        != "component-only; NVIDIA numerical libraries supplied by owning wheels; CUDA runtime owner recorded in runtime_linkage"
    ):
        raise AuditFailure("Windows build identity has an invalid redistribution policy")
    toolkit_wheels = installed.get("toolkit_wheels")
    if (
        not isinstance(toolkit_wheels, dict)
        or not toolkit_wheels
        or any(
            nvidia.versions.get(name) != version
            for name, version in toolkit_wheels.items()
        )
    ):
        raise AuditFailure(
            "installed NVIDIA wheel versions differ from the build identity"
        )
    discovery = read_json(config.build_dir / "pip-cuda/discovery.json")
    if installed.get("runtime_linkage") != discovery.get("runtime_linkage"):
        raise AuditFailure(
            "installed CUDA runtime linkage differs from toolkit discovery"
        )
    toolchain = installed.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or not toolchain
        or toolchain != discovery.get("toolchain")
        or toolkit_wheels != discovery.get("wheels")
    ):
        raise AuditFailure("installed toolchain identity differs from toolkit discovery")
    for name, record in toolchain.items():
        if not isinstance(record, dict):
            raise AuditFailure(f"invalid Windows toolchain identity: {name}")
        if name == "sdk":
            for key in ("root", "msvc_root"):
                root = record.get(key)
                if not isinstance(root, str) or not profile_path(Path(root)).is_dir():
                    raise AuditFailure(f"missing recorded Windows {key}")
            continue
        path = record.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise AuditFailure(f"missing Windows toolchain path: {name}")
        if file_hash(profile_path(Path(path))) != record.get("sha256"):
            raise AuditFailure(f"Windows toolchain hash mismatch: {name}")
    records = discovery.get("records")
    if not isinstance(records, list) or not records:
        raise AuditFailure("Windows toolkit discovery lacks wheel input records")
    for record in records:
        if not isinstance(record, dict):
            raise AuditFailure("invalid Windows toolkit input record")
        path = record.get("path")
        view = record.get("view")
        if not isinstance(path, str) or not isinstance(view, str):
            raise AuditFailure("Windows toolkit input record lacks original/view paths")
        relative = PurePosixPath(view)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in view
            or ":" in view
        ):
            raise AuditFailure(f"invalid Windows toolkit view path: {view}")
        for asset in (Path(path), config.build_dir / "pip-cuda" / view):
            if file_hash(profile_path(asset)) != record.get("sha256"):
                raise AuditFailure(f"Windows toolkit input hash mismatch: {asset}")
    manifest = read_json(extraction / "phydrax_cuda13/components.json")
    expected_components = {
        component: f"lib/phydrax_cuda_{component}.dll" for component in config.components
    }
    if manifest != expected_components:
        raise AuditFailure(
            "installed component manifest differs from the requested DLL set"
        )
    expected_maps = {
        component: f"lib/phydrax_cuda_{component}.map" for component in config.components
    }
    if installed.get("link_maps") != expected_maps:
        raise AuditFailure(
            "installed link-map manifest differs from the requested component set"
        )
    return installed


def verify_windows_wheel(
    wheel: Path,
    extraction: Path,
    config: Configuration,
    nvidia: NvidiaInventory,
    metadata: dict[str, Json],
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    if not wheel.name.endswith("-win_amd64.whl"):
        raise AuditFailure(f"Windows companion must have a win_amd64 tag: {wheel.name}")
    sizes: dict[str, Json] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or ":" in member.filename
                or any(part.rstrip(" .") != part for part in relative.parts)
                or member.filename.lower() in seen
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise AuditFailure(
                    f"unsafe/duplicate Windows wheel member: {member.filename}"
                )
            seen.add(member.filename.lower())
            sizes[member.filename] = {
                "compressed": member.compress_size,
                "uncompressed": member.file_size,
            }
        wheel_metadata = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata) != 1:
            raise AuditFailure("wheel must have exactly one WHEEL metadata file")
        tags = re.findall(
            r"^Tag:\s*(\S+)\s*$",
            archive.read(wheel_metadata[0]).decode("utf-8"),
            re.MULTILINE,
        )
        if not tags or any(tag.rsplit("-", 1)[-1] != "win_amd64" for tag in tags):
            raise AuditFailure(f"non-Windows platform in wheel metadata tags: {tags}")
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise AuditFailure("wheel must have exactly one METADATA file")
        package = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
        project = tomllib.loads(
            (ROOT / "native/cuda13/pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        for key in ("name", "version"):
            if package.get_all(key) != [project[key]]:
                raise AuditFailure(f"wheel {key} differs from the companion project")
        archive.extractall(extraction)
    identity = verify_windows_identity(extraction, config, nvidia)
    reader = pe_reader()
    expected = {
        f"phydrax_cuda13/lib/phydrax_cuda_{component}.dll": component
        for component in config.components
    }
    images: dict[str, PeImage] = {}
    components: dict[str, Json] = {}
    cuobjdump = locate_cuda_tool("cuobjdump", config, nvidia)
    for path in sorted(extraction.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(extraction).as_posix()
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic == b"\x7fELF":
            raise AuditFailure(f"ELF asset bundled in Windows wheel: {relative}")
        if magic[:2] != b"MZ" and path.suffix.lower() not in (
            ".dll",
            ".pyd",
            ".exe",
            ".lib",
            ".a",
            ".cubin",
            ".fatbin",
        ):
            continue
        component = expected.get(relative)
        if component is None:
            raise AuditFailure(
                f"unexpected/NVIDIA native asset bundled in wheel: {relative}"
            )
        image = reader.inspect_pe(path)
        if image.machine != 0x8664:
            raise AuditFailure(f"{relative}: expected PE x86-64")
        images[path.name.lower()] = image
        exports = verify_pe_exports(path, image, component)
        cudart = verify_pe_shared_cudart(path, image, metadata, config)
        build_hash = metadata["build_hash"]
        if not isinstance(build_hash, str):
            raise TypeError("validated build hash must be a string")
        if build_hash.encode("ascii") + b"\0" not in path.read_bytes():
            raise AuditFailure(f"{path.name}: CMake build hash absent from library")
        components[component] = {
            "sha256": file_hash(path),
            "build_hash": build_hash,
            "exports": exports,
            "imports": list(image.imports),
            "delay_imports": list(image.delay_imports),
            "shared_cudart": cudart,
            "device_code": inspect_device_code(
                path, cuobjdump, config, environment, evidence
            ),
        }
    if set(components) != set(config.components):
        raise AuditFailure(
            f"wheel component set mismatch: {sorted(components)} != {sorted(config.components)}"
        )
    if config.components == ("cudss",) and wheel.stat().st_size >= 100_000_000:
        raise AuditFailure("cuDSS-only wheel must be smaller than 100 MB")
    return {
        "filename": wheel.name,
        "sha256": file_hash(wheel),
        "compressed_bytes": wheel.stat().st_size,
        "members": sizes,
        "components": components,
        "build_identity": identity,
        "nvidia_dependency_closure": verify_pe_closure(images, nvidia),
        "archive_link_evidence": archive_link_evidence(config),
        "link_maps": {
            component: {
                "path": str(config.build_dir / f"phydrax_cuda_{component}.map"),
                "sha256": file_hash(config.build_dir / f"phydrax_cuda_{component}.map"),
            }
            for component in config.components
        },
        "cuobjdump": str(cuobjdump),
        "cuobjdump_sha256": file_hash(cuobjdump),
        "cuobjdump_version": run((str(cuobjdump), "--version"), environment, evidence),
        "runtime_qualification": False,
    }


def build_and_repair(
    config: Configuration,
    staging: Path,
    nvidia: NvidiaInventory,
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> tuple[Path, frozenset[str]]:
    raw = staging / "raw"
    repaired = staging / "repaired"
    raw.mkdir()
    repaired.mkdir()
    settings = {
        "build-dir": str(config.build_dir),
        "cmake.build-type": "Release",
        "cmake.define.PHX_CUDA_ARCHITECTURES": ";".join(config.architectures),
        "cmake.define.PHX_CUDA_COMPONENTS": ";".join(config.components),
        "cmake.define.Python_EXECUTABLE": str(config.python),
        "build.tool-args": f"-j{config.jobs}",
        "install.strip": "false",
    }
    if config.download_dir is not None:
        settings["cmake.define.PHX_CUDA_DOWNLOAD_DIR"] = str(config.download_dir)
    if sys.platform == "win32":
        settings["cmake.define.CMAKE_MAKE_PROGRAM"] = environment["PHX_WINDOWS_NINJA"]
        settings["cmake.define.CMAKE_MSVC_RUNTIME_LIBRARY"] = "MultiThreadedDLL"
        settings["cmake.define.CMAKE_CUDA_RUNTIME_LIBRARY"] = "Shared"
    command = [
        str(config.python),
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(raw),
    ]
    command.extend(f"--config-setting={key}={value}" for key, value in settings.items())
    command.append(str(ROOT / "native/cuda13"))
    run(command, environment, evidence)
    wheels = sorted(raw.glob("*.whl"))
    if len(wheels) != 1:
        raise AuditFailure(f"expected one built wheel, found {len(wheels)}")
    if sys.platform == "win32":
        evidence["wheel_platform"] = "win_amd64"
        return wheels[0], frozenset()
    plat, allowed = manylinux_policy()
    evidence["repair_platform"] = plat
    excludes = sorted(set(nvidia.libraries).union(DRIVER_LIBRARIES))
    evidence["auditwheel_excludes"] = excludes
    command = [
        str(config.python),
        "-m",
        "auditwheel",
        "repair",
        "--plat",
        plat,
        "--only-plat",
        "--wheel-dir",
        str(repaired),
    ]
    for name in excludes:
        command.extend(("--exclude", name))
    command.append(str(wheels[0]))
    run(command, environment, evidence)
    wheels = sorted(repaired.glob("*.whl"))
    if len(wheels) != 1:
        raise AuditFailure(f"expected one repaired wheel, found {len(wheels)}")
    return wheels[0], allowed


def execute(config: Configuration) -> int:
    if sys.platform == "win32":
        profile_path(config.wheelhouse)
    config.wheelhouse.mkdir(parents=True, exist_ok=True)
    manifest_path = config.wheelhouse / "phydrax-cuda13-build.json"
    evidence: dict[str, Json] = {
        "major": 13,
        "architectures": list(config.architectures),
        "components": list(config.components),
        "python": str(config.python),
        "build_dir": str(config.build_dir),
        "audit_wheel": str(config.audit_wheel)
        if config.audit_wheel is not None
        else None,
        "successful": False,
        "commands": [],
        "container_image": os.environ.get("PHX_CUDA_BUILD_CONTAINER_IMAGE"),
    }
    environment = clean_environment(config.python, config.jobs)
    try:
        evidence["host_architecture"] = host_platform().architecture
        if sys.platform == "win32":
            windows_environment(config, environment, evidence)
        nvidia = inventory()
        evidence["nvidia_wheels"] = dict(nvidia.versions)
        tools: dict[str, Json] = {"python": sys.version}
        packages = (
            ("build", "scikit-build-core")
            if sys.platform == "win32"
            else (
                "build",
                "scikit-build-core",
                "auditwheel",
                "cmake",
                "ninja",
                "patchelf",
            )
        )
        if config.audit_wheel is None or sys.platform != "win32":
            for name in packages:
                tools[name] = importlib.metadata.version(name)
        if sys.platform != "win32":
            tools["readelf"] = run(("readelf", "--version"), environment, evidence)
        evidence["tool_versions"] = tools
        with tempfile.TemporaryDirectory(
            prefix=".cuda-wheel-", dir=config.wheelhouse
        ) as temporary:
            staging = Path(temporary)
            if config.audit_wheel is None:
                wheel, allowed = build_and_repair(
                    config, staging, nvidia, environment, evidence
                )
            else:
                wheel = config.audit_wheel
                allowed = (
                    frozenset() if sys.platform == "win32" else manylinux_policy()[1]
                )
            metadata = build_metadata(config)
            evidence["build"] = metadata
            if config.audit_wheel is None:
                nvcc = locate_cuda_tool("nvcc", config, nvidia)
                tools["nvcc"] = run((str(nvcc), "--version"), environment, evidence)
            if sys.platform == "win32":
                evidence["wheel"] = verify_windows_wheel(
                    wheel,
                    staging / "extracted",
                    config,
                    nvidia,
                    metadata,
                    environment,
                    evidence,
                )
            else:
                evidence["wheel"] = verify_wheel(
                    wheel,
                    staging / "extracted",
                    config,
                    nvidia,
                    allowed,
                    metadata,
                    environment,
                    evidence,
                )
            destination = config.wheelhouse / wheel.name
            if wheel.resolve() != destination.resolve():
                shutil.copy2(wheel, destination)
        evidence["successful"] = True
        # Coverage is component-specific, and never asserted for native builds.
        evidence["architecture_mode"] = (
            "native-development"
            if config.architectures == ("native",)
            else "explicit-targets"
        )
    except (
        AuditFailure,
        OSError,
        TypeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
        zipfile.BadZipFile,
        ImportError,
    ) as error:
        evidence["error"] = str(error)
        print(f"CUDA wheel build refused: {error}", file=sys.stderr)
        return 1
    finally:
        manifest_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Build manifest: {manifest_path}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_configuration(argv)
    environment = clean_environment(config.python, config.jobs)
    if sys.platform == "win32":
        profile_path(config.python)
    if config.container != "none":
        command = container_command(config)
        # Explicit image identity survives the inner no-container invocation.
        image = host_platform().image
        insertion = command.index(image)
        command[insertion:insertion] = [
            "--env",
            f"PHX_CUDA_BUILD_CONTAINER_IMAGE={image}",
        ]
        return subprocess.run(command, env=environment, check=False).returncode
    if not config.python.is_file():
        raise FileNotFoundError(
            f"build-environment Python does not exist: {config.python}"
        )
    if Path(sys.executable).absolute() != config.python:
        command = [str(config.python), str(Path(__file__).resolve())]
        command.extend(sys.argv[1:] if argv is None else argv)
        return subprocess.run(command, env=environment, check=False).returncode
    return execute(config)


if __name__ == "__main__":
    raise SystemExit(main())
