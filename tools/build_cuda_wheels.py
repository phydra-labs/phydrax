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
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


type Json = None | bool | int | float | str | Sequence[Json] | Mapping[str, Json]

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("cudss", "amgx", "quda")
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
        f"wheel builds require Linux x86_64 or aarch64: {sys.platform}/{architecture}"
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
        default=";".join(COMPONENTS),
        help="Semicolon- or comma-separated subset of cudss, amgx, quda.",
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
        parser.error("--components must select cudss, amgx, and/or quda")
    if len(set(selected)) != len(selected):
        parser.error("--components contains duplicates")
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
        if not (name.startswith("nvidia-") or name.startswith("cudss-")):
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
            if "cudart_static" in line and (
                path.name == "link.txt"
                or "LINK_LIBRARIES =" in line
                or "LINK_FLAGS =" in line
            ):
                raise AuditFailure(
                    f"static cudart archive in link command: {path}: {line}"
                )
    return {"files": [str(path) for path in paths], "static_cudart_archive": False}


def inspect_device_code(
    library: Path,
    cuobjdump: Path,
    config: Configuration,
    environment: Mapping[str, str],
    evidence: dict[str, Json],
) -> dict[str, Json]:
    host_only_allowed = library.name == "libphydrax_cuda_cudss.so"
    no_code_diagnostic = f"cuobjdump info : File '{library}' does not contain device code"
    expected_failure = (255, no_code_diagnostic) if host_only_allowed else None
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
    config.wheelhouse.mkdir(parents=True, exist_ok=True)
    manifest_path = config.wheelhouse / "phydrax-cuda13-build.json"
    evidence: dict[str, Json] = {
        "major": 13,
        "architectures": list(config.architectures),
        "components": list(config.components),
        "python": str(config.python),
        "build_dir": str(config.build_dir),
        "successful": False,
        "commands": [],
        "container_image": os.environ.get("PHX_CUDA_BUILD_CONTAINER_IMAGE"),
    }
    environment = clean_environment(config.python, config.jobs)
    try:
        evidence["host_architecture"] = host_platform().architecture
        nvidia = inventory()
        evidence["nvidia_wheels"] = dict(nvidia.versions)
        tools: dict[str, Json] = {"python": sys.version}
        for name in (
            "build",
            "scikit-build-core",
            "auditwheel",
            "cmake",
            "ninja",
            "patchelf",
        ):
            tools[name] = importlib.metadata.version(name)
        tools["readelf"] = run(("readelf", "--version"), environment, evidence)
        evidence["tool_versions"] = tools
        with tempfile.TemporaryDirectory(
            prefix=".cuda-wheel-", dir=config.wheelhouse
        ) as temporary:
            staging = Path(temporary)
            wheel, allowed = build_and_repair(
                config, staging, nvidia, environment, evidence
            )
            metadata = build_metadata(config)
            evidence["build"] = metadata
            nvcc = locate_cuda_tool("nvcc", config, nvidia)
            tools["nvcc"] = run((str(nvcc), "--version"), environment, evidence)
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
            shutil.copy2(wheel, config.wheelhouse / wheel.name)
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
