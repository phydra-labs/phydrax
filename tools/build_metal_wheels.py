#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build and audit the phydrax-metal companion wheel from the canonical source lock.

Example::

    python tools/build_metal_wheels.py --python /path/to/python3.13 \
        --cmake /path/to/cmake --ninja /path/to/ninja \
        --build-dir /path/to/build/metal --wheelhouse /path/to/wheelhouse \
        --output /path/to/metal-build.json

Every source is acquired from ``native/metal/sources.lock.json`` and verified by
SHA-256 before extraction; every patch is verified before it is applied. The
tool never installs Python packages or searches Homebrew/``/usr/local``: commands
run with the system ``PATH`` and explicit tool paths, CMake ignores those
prefixes, and MLX's network ``FetchContent`` steps are replaced by locked
archives. ``--python`` is only used to run the companion's PEP 517 backend.

Dependency builds (LLVM/MLIR, MLX) are reused only when their recorded input key
is identical. ``--dependencies-only`` stops after them; it never produces a
wheel. Evidence is written to ``--output`` on success and on failure.

The canonical ``phydrax/_jax_indexing.py`` is a build input and is packaged as
``phydrax_metal/_jax_indexing.py`` with its own asset hash. The companion remains
standalone and never installs files into the core distribution's module paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


type Json = None | bool | int | float | str | Sequence[Json] | Mapping[str, Json]

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "metal"
LOCK = NATIVE / "sources.lock.json"
PLATFORM = "mps"
LIBRARY = "lib/libpjrt_plugin_mps.dylib"
METALLIB = "lib/mlx.metallib"
EXPORTED_SYMBOLS = frozenset(("_GetPjrtApi", "_PhydraxMetal_SetHostCallbackTrampoline"))
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
IGNORED_PREFIXES = ("/opt/homebrew", "/usr/local")
SYSTEM_LIBRARY_PREFIXES = ("/usr/lib/", "/System/Library/")
LOG_TAIL_BYTES = 8192
LLVM_ARGUMENTS = (
    "-DLLVM_ENABLE_PROJECTS=mlir",
    "-DLLVM_TARGETS_TO_BUILD=AArch64",
    "-DLLVM_ENABLE_ASSERTIONS=OFF",
    "-DLLVM_ENABLE_ZLIB=OFF",
    "-DLLVM_ENABLE_ZSTD=OFF",
    "-DLLVM_ENABLE_LIBXML2=OFF",
    "-DLLVM_ENABLE_LIBEDIT=OFF",
    "-DLLVM_ENABLE_LIBPFM=OFF",
    "-DLLVM_ENABLE_TERMINFO=OFF",
    "-DLLVM_ENABLE_BACKTRACES=OFF",
    "-DLLVM_ENABLE_CRASH_OVERRIDES=OFF",
    "-DLLVM_BUILD_TOOLS=OFF",
    "-DLLVM_INSTALL_UTILS=ON",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DLLVM_INCLUDE_BENCHMARKS=OFF",
    "-DLLVM_INCLUDE_EXAMPLES=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
    "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF",
    "-DMLIR_ENABLE_EXECUTION_ENGINE=OFF",
    "-DMLIR_INCLUDE_TESTS=OFF",
)
MLX_ARGUMENTS = (
    "-DBUILD_SHARED_LIBS=OFF",
    "-DMLX_BUILD_TESTS=OFF",
    "-DMLX_BUILD_EXAMPLES=OFF",
    "-DMLX_BUILD_BENCHMARKS=OFF",
    "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
    "-DMLX_BUILD_PYTHON_STUBS=OFF",
    "-DMLX_BUILD_GGUF=OFF",
    "-DMLX_BUILD_SAFETENSORS=OFF",
    "-DMLX_USE_CCACHE=OFF",
    "-DMLX_METAL_JIT=OFF",
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
)


class BuildFailure(RuntimeError):
    """A required build or audit obligation failed."""


class Options(argparse.Namespace):
    python: Path
    cmake: Path
    ninja: Path
    build_dir: Path
    wheelhouse: Path
    output: Path
    download_dir: Path | None
    jobs: int
    dependencies_only: bool


@dataclass(frozen=True)
class Configuration:
    python: Path
    cmake: Path
    ninja: Path
    build_dir: Path
    wheelhouse: Path
    output: Path
    download_dir: Path
    jobs: int
    dependencies_only: bool


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    strip_prefix: str
    license_files: tuple[str, ...]


@dataclass(frozen=True)
class Patch:
    target: str
    origin: str
    path: str
    sha256: str
    strip: int


@dataclass(frozen=True)
class Lock:
    sha256: str
    deployment_target: str
    jax_version: str
    jaxlib_version: str
    sources: Mapping[str, Source]
    patches: tuple[Patch, ...]
    entries: tuple[Json, ...]


@dataclass(frozen=True)
class Toolchain:
    cc: Path
    cxx: Path
    sysroot: Path
    identity: Mapping[str, Json]


def parse_configuration(argv: Sequence[str] | None = None) -> Configuration:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--ninja", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Immutable, hash-verified archive cache (default: BUILD_DIR/downloads).",
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--dependencies-only", action="store_true")
    options = parser.parse_args(argv, namespace=Options())
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    for name in ("python", "cmake", "ninja"):
        path: Path = getattr(options, name)
        if not path.is_absolute() or not os.access(path, os.X_OK):
            parser.error(f"--{name} must be an absolute executable path")
    build_dir = options.build_dir.resolve()
    return Configuration(
        python=options.python,
        cmake=options.cmake,
        ninja=options.ninja,
        build_dir=build_dir,
        wheelhouse=options.wheelhouse.resolve(),
        output=options.output.resolve(),
        download_dir=(
            options.download_dir.resolve()
            if options.download_dir is not None
            else build_dir / "downloads"
        ),
        jobs=options.jobs,
        dependencies_only=options.dependencies_only,
    )


def json_value(value: object) -> Json:
    """Validate the dynamic JSON interchange boundary once."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): json_value(item) for key, item in value.items()}
    raise TypeError("lock input is not JSON")


def canonical(value: Json) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def text(mapping: Mapping[str, Json], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise BuildFailure(f"source lock requires a non-empty string {key!r}")
    return value


def read_lock() -> Lock:
    data = LOCK.read_bytes()
    document = json_value(json.loads(data))
    if not isinstance(document, dict):
        raise BuildFailure("source lock must be an object")
    jax = document.get("jax")
    sources = document.get("sources")
    patches = document.get("patches")
    if (
        not isinstance(jax, dict)
        or not isinstance(sources, list)
        or not isinstance(patches, list)
    ):
        raise BuildFailure("source lock requires jax, sources and patches")
    parsed: dict[str, Source] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            raise BuildFailure("source entries must be objects")
        sha = text(entry, "sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise BuildFailure(f"source {entry.get('name')!r} has no SHA-256")
        licenses = entry.get("license_files")
        if (
            not isinstance(licenses, list)
            or not licenses
            or not all(isinstance(item, str) for item in licenses)
        ):
            raise BuildFailure(f"source {entry.get('name')!r} lists no license files")
        name = text(entry, "name")
        if name in parsed:
            raise BuildFailure(f"duplicate source {name!r}")
        parsed[name] = Source(
            name,
            text(entry, "url"),
            sha,
            text(entry, "strip_prefix"),
            tuple(str(item) for item in licenses),
        )
    ordered: list[Patch] = []
    for entry in patches:
        if not isinstance(entry, dict):
            raise BuildFailure("patch entries must be objects")
        strip = entry.get("strip")
        sha = text(entry, "sha256")
        if not isinstance(strip, int) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise BuildFailure(f"patch {entry.get('path')!r} has no strip/SHA-256")
        patch = Patch(
            text(entry, "target"), text(entry, "origin"), text(entry, "path"), sha, strip
        )
        if patch.target not in parsed or (
            patch.origin != "native/metal" and patch.origin not in parsed
        ):
            raise BuildFailure(f"patch {patch.path!r} names an unknown source")
        ordered.append(patch)
    return Lock(
        sha256=digest(data),
        deployment_target=text(document, "deployment_target"),
        jax_version=text(jax, "jax_version"),
        jaxlib_version=text(jax, "jaxlib_version"),
        sources=parsed,
        patches=tuple(ordered),
        entries=tuple(sources),
    )


class Evidence:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.record: dict[str, Json] = {
            "tool": "tools/build_metal_wheels.py",
            "arguments": sys.argv[1:],
            "status": "incomplete",
            "commands": [],
            "steps": [],
        }

    def add(self, key: str, value: Json) -> None:
        items = self.record[key]
        if not isinstance(items, list):
            raise TypeError(f"evidence {key} must be a list")
        items.append(value)

    def write(self) -> None:
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.output.write_text(
            json.dumps(self.record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def clean_environment(config: Configuration, lock: Lock) -> dict[str, str]:
    environment = {
        "PATH": SYSTEM_PATH,
        "HOME": os.environ.get("HOME", str(config.build_dir)),
        "TMPDIR": str(config.build_dir / "tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "MACOSX_DEPLOYMENT_TARGET": lock.deployment_target,
        "CMAKE_BUILD_PARALLEL_LEVEL": str(config.jobs),
    }
    return environment


def run(
    evidence: Evidence,
    name: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    cwd: Path,
) -> str:
    log_dir = evidence.config.build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{name}.log"
    started = time.monotonic()
    with log.open("wb") as stream:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output = log.read_bytes()
    record: dict[str, Json] = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "returncode": result.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "log": str(log),
    }
    if result.returncode:
        record["log_tail"] = output[-LOG_TAIL_BYTES:].decode("utf-8", "replace")
    evidence.add("commands", record)
    if result.returncode:
        raise BuildFailure(f"{name} failed ({result.returncode}); see {log}")
    return output.decode("utf-8", "replace")


def capture(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        env={"PATH": SYSTEM_PATH, "LANG": "C"},
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise BuildFailure(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def toolchain(config: Configuration) -> Toolchain:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise BuildFailure("phydrax-metal builds only on macOS arm64 hosts")
    cc = Path(capture(("/usr/bin/xcrun", "--find", "clang")))
    cxx = Path(capture(("/usr/bin/xcrun", "--find", "clang++")))
    sysroot = Path(capture(("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path")))
    developer = capture(("/usr/bin/xcode-select", "-p"))
    for compiler in (cc, cxx):
        if not str(compiler.resolve()).startswith(str(Path(developer).resolve())):
            raise BuildFailure(
                f"compiler {compiler} is not from the selected Xcode {developer}"
            )
    identity: dict[str, Json] = {
        "developer_dir": capture(("/usr/bin/xcode-select", "-p")),
        "xcode": capture(("/usr/bin/xcodebuild", "-version")).splitlines(),
        "sdk_version": capture(
            ("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version")
        ),
        "sdk_build": capture(
            ("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-build-version")
        ),
        "clang": capture((str(cxx), "--version")).splitlines()[0],
        "metal": capture(
            ("/usr/bin/xcrun", "--sdk", "macosx", "metal", "--version")
        ).splitlines()[0],
        "cmake": capture((str(config.cmake), "--version")).splitlines()[0],
        "ninja": capture((str(config.ninja), "--version")),
        "build_host": f"{platform.system()} {platform.mac_ver()[0]} {platform.machine()}",
    }
    return Toolchain(cc, cxx, sysroot, identity)


def cmake_common(config: Configuration, lock: Lock, tools: Toolchain) -> list[str]:
    prefix_map = f"-ffile-prefix-map={config.build_dir}=/phydrax-metal-build"
    return [
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={config.ninja}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={tools.cc}",
        f"-DCMAKE_CXX_COMPILER={tools.cxx}",
        f"-DCMAKE_OSX_SYSROOT={tools.sysroot}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={lock.deployment_target}",
        "-DCMAKE_OSX_ARCHITECTURES=arm64",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DCMAKE_IGNORE_PREFIX_PATH={';'.join(IGNORED_PREFIXES)}",
        f"-DCMAKE_SYSTEM_IGNORE_PREFIX_PATH={';'.join(IGNORED_PREFIXES)}",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
        f"-DCMAKE_C_FLAGS={prefix_map}",
        f"-DCMAKE_CXX_FLAGS={prefix_map}",
    ]


def acquire(config: Configuration, source: Source, evidence: Evidence) -> Path:
    config.download_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        ".zip"
        if source.url.endswith(".zip")
        else ".tar.xz"
        if source.url.endswith(".tar.xz")
        else ".tar.gz"
    )
    archive = config.download_dir / f"{source.name}-{source.sha256[:16]}{suffix}"
    if not archive.exists():
        partial = archive.with_name(archive.name + ".partial")
        with (
            urllib.request.urlopen(source.url, timeout=120) as response,
            partial.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        partial.replace(archive)
    actual = file_hash(archive)
    evidence.add(
        "steps",
        {
            "step": "verify-source",
            "name": source.name,
            "sha256": actual,
            "url": source.url,
        },
    )
    if actual != source.sha256:
        raise BuildFailure(
            f"{source.name} archive hash {actual} does not match the lock {source.sha256}"
        )
    return archive


def extract(archive: Path, source: Source, destination: Path) -> Path:
    target = destination / source.name
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        root = Path(scratch)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    resolved = (root / member).resolve()
                    if not resolved.is_relative_to(root.resolve()):
                        raise BuildFailure(f"{archive} contains unsafe member {member}")
                bundle.extractall(root)
        else:
            with tarfile.open(archive) as bundle:
                bundle.extractall(root, filter="data")
        extracted = root / source.strip_prefix
        if not extracted.is_dir():
            raise BuildFailure(f"{archive} lacks top-level {source.strip_prefix}")
        extracted.replace(target)
    return target


def patch_file(patch: Patch, trees: Mapping[str, Path]) -> Path:
    base = NATIVE if patch.origin == "native/metal" else trees[patch.origin]
    path = (base / patch.path).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise BuildFailure(f"patch {patch.origin}:{patch.path} is missing")
    actual = file_hash(path)
    if actual != patch.sha256:
        raise BuildFailure(
            f"patch {patch.origin}:{patch.path} hash {actual} is not locked"
        )
    return path


def prepare_sources(
    config: Configuration, lock: Lock, environment: Mapping[str, str], evidence: Evidence
) -> dict[str, Path]:
    trees: dict[str, Path] = {}
    source_root = config.build_dir / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    for source in lock.sources.values():
        trees[source.name] = extract(
            acquire(config, source, evidence), source, source_root
        )
    # Source trees are plain directories (no .git). Without a ceiling, git would
    # discover an enclosing repository (e.g. a checkout holding the build dir)
    # and silently skip every path outside it; --verbose records each file.
    patch_environment = {**environment, "GIT_CEILING_DIRECTORIES": str(source_root)}
    for index, patch in enumerate(lock.patches):
        path = patch_file(patch, trees)
        output = run(
            evidence,
            f"patch-{index:02d}-{patch.target}",
            (
                "/usr/bin/git",
                "apply",
                "--verbose",
                f"-p{patch.strip}",
                "--whitespace=nowarn",
                str(path),
            ),
            patch_environment,
            cwd=trees[patch.target],
        )
        if "Applied patch" not in output:
            raise BuildFailure(f"patch {patch.origin}:{patch.path} applied no changes")
    for source in lock.sources.values():
        for relative in source.license_files:
            if not (trees[source.name] / relative).is_file():
                raise BuildFailure(f"{source.name} lacks license file {relative}")
    return trees


def dependency(
    config: Configuration,
    evidence: Evidence,
    environment: Mapping[str, str],
    name: str,
    source_dir: Path,
    arguments: Sequence[str],
    key_material: Json,
) -> Path:
    prefix = config.build_dir / "prefix" / name
    stamp = prefix / ".phydrax-metal-input"
    key = digest(canonical({"arguments": list(arguments), "inputs": key_material}))
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == key:
        evidence.add(
            "steps", {"step": "reuse-dependency", "name": name, "input_key": key}
        )
        return prefix
    work = config.build_dir / "work" / name
    for stale in (work, prefix):
        if stale.exists():
            shutil.rmtree(stale)
    run(
        evidence,
        f"{name}-configure",
        (
            str(config.cmake),
            "-S",
            str(source_dir),
            "-B",
            str(work),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            *arguments,
        ),
        environment,
        cwd=config.build_dir,
    )
    run(
        evidence,
        f"{name}-build",
        (str(config.cmake), "--build", str(work), "-j", str(config.jobs)),
        environment,
        cwd=config.build_dir,
    )
    run(
        evidence,
        f"{name}-install",
        (str(config.cmake), "--install", str(work)),
        environment,
        cwd=config.build_dir,
    )
    stamp.write_text(key + "\n", encoding="utf-8")
    evidence.add("steps", {"step": "built-dependency", "name": name, "input_key": key})
    return prefix


def source_identity(lock: Lock) -> tuple[str, dict[str, str]]:
    if file_hash(LOCK) != lock.sha256:
        raise BuildFailure("Native source lock changed during the build.")
    files = [
        LOCK,
        NATIVE / "CMakeLists.txt",
        NATIVE / "cmake" / "sdy" / "CMakeLists.txt",
        NATIVE / "pyproject.toml",
        NATIVE / "_build_backend.py",
        ROOT / "phydrax" / "_jax_indexing.py",
        Path(__file__),
        *sorted((NATIVE / "python").rglob("*.py")),
    ]
    for patch in lock.patches:
        if patch.origin == "native/metal":
            path = NATIVE / patch.path
            if file_hash(path) != patch.sha256:
                raise BuildFailure(f"Native patch changed after locking: {patch.path}")
            files.append(path)
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in files}
    return digest(canonical(hashes)), hashes


def companion_version() -> str:
    project = tomllib.loads((NATIVE / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    if not isinstance(version, str):
        raise BuildFailure("native/metal/pyproject.toml has no version")
    return version


def header_version(header: Path, pattern: str) -> tuple[int, ...]:
    match = re.search(pattern, header.read_text(encoding="utf-8"), re.S)
    if match is None:
        raise BuildFailure(f"cannot read version from {header}")
    return tuple(int(group) for group in match.groups())


def macho_audit(
    library: Path, lock: Lock, build_dir: Path, evidence: Evidence
) -> dict[str, Json]:
    archs = capture(("/usr/bin/lipo", "-archs", str(library)))
    loads = capture(("/usr/bin/otool", "-l", str(library)))
    linked = capture(("/usr/bin/otool", "-L", str(library))).splitlines()[1:]
    exports = capture(("/usr/bin/nm", "-gUj", str(library))).split()
    failures: list[str] = []
    if archs != "arm64":
        failures.append(f"architectures {archs!r} are not exactly arm64")
    builds = re.findall(
        r"cmd LC_BUILD_VERSION.*?platform (\d+).*?minos ([\d.]+)", loads, re.S
    )
    if len(builds) != 1 or builds[0][0] != "1":
        failures.append(f"expected one macOS LC_BUILD_VERSION, found {builds}")
    minos = builds[0][1] if builds else ""
    if minos != lock.deployment_target:
        failures.append(
            f"minos {minos} differs from deployment target {lock.deployment_target}"
        )
    rpaths = re.findall(r"cmd LC_RPATH.*?path (\S+)", loads, re.S)
    if rpaths:
        failures.append(f"unexpected LC_RPATH entries {rpaths}")
    dependencies = [line.strip().split(" (")[0] for line in linked]
    own = str(library.name)
    for dependency_path in dependencies:
        if dependency_path.endswith(own) or dependency_path.startswith("@rpath/" + own):
            continue
        if not dependency_path.startswith(SYSTEM_LIBRARY_PREFIXES):
            failures.append(f"non-system dependency {dependency_path}")
    if set(exports) != EXPORTED_SYMBOLS:
        failures.append(
            f"exported symbols {sorted(exports)} are not exactly {sorted(EXPORTED_SYMBOLS)}"
        )
    contents = library.read_bytes()
    leaked = [
        prefix
        for prefix in (str(build_dir), *IGNORED_PREFIXES)
        if prefix.encode() in contents
    ]
    if leaked:
        failures.append(f"embedded nonportable build prefixes: {leaked}")
    signature = subprocess.run(
        ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(library)),
        text=True,
        capture_output=True,
        check=False,
    )
    if signature.returncode:
        failures.append(f"code signature invalid: {signature.stderr.strip()}")
    audit: dict[str, Json] = {
        "architectures": archs,
        "minos": minos,
        "dependencies": dependencies,
        "exports": sorted(exports),
        "embedded_path_strings": leaked,
        "codesign": signature.stderr.strip(),
        "failures": failures,
    }
    evidence.record["macho_audit"] = audit
    if failures:
        raise BuildFailure("Mach-O audit failed: " + "; ".join(failures))
    return audit


def build_plugin(
    config: Configuration,
    lock: Lock,
    tools: Toolchain,
    trees: Mapping[str, Path],
    prefixes: Mapping[str, Path],
    environment: Mapping[str, str],
    evidence: Evidence,
    version: str,
    source_id: str,
) -> Path:
    work = config.build_dir / "work" / "plugin"
    stage = config.build_dir / "stage"
    for stale in (work, stage):
        if stale.exists():
            shutil.rmtree(stale)
    run(
        evidence,
        "plugin-configure",
        (
            str(config.cmake),
            "-S",
            str(NATIVE),
            "-B",
            str(work),
            f"-DCMAKE_INSTALL_PREFIX={stage / 'phydrax_metal'}",
            *cmake_common(config, lock, tools),
            f"-DMLIR_DIR={prefixes['llvm-project'] / 'lib' / 'cmake' / 'mlir'}",
            f"-DLLVM_DIR={prefixes['llvm-project'] / 'lib' / 'cmake' / 'llvm'}",
            f"-DMLX_DIR={prefixes['mlx'] / 'share' / 'cmake' / 'MLX'}",
            f"-DLLVM_EXTERNAL_LIT={trees['llvm-project'] / 'llvm' / 'utils' / 'lit' / 'lit.py'}",
            f"-DPHYDRAX_METAL_JAX_MPS_SOURCE_DIR={trees['jax-mps']}",
            f"-DPHYDRAX_METAL_STABLEHLO_SOURCE_DIR={trees['stablehlo']}",
            f"-DPHYDRAX_METAL_SHARDY_SOURCE_DIR={trees['shardy']}",
            f"-DPHYDRAX_METAL_XLA_SOURCE_DIR={trees['xla']}",
            f"-DPHYDRAX_METAL_VERSION={version}",
            f"-DPHYDRAX_METAL_SOURCE_ID={source_id}",
        ),
        environment,
        cwd=config.build_dir,
    )
    run(
        evidence,
        "plugin-build",
        (str(config.cmake), "--build", str(work), "-j", str(config.jobs)),
        environment,
        cwd=config.build_dir,
    )
    run(
        evidence,
        "plugin-install",
        (str(config.cmake), "--install", str(work)),
        environment,
        cwd=config.build_dir,
    )
    package = stage / "phydrax_metal"
    for relative in (LIBRARY, METALLIB):
        if not (package / relative).is_file() or (package / relative).stat().st_size == 0:
            raise BuildFailure(f"native install did not produce {relative}")
    if (package / METALLIB).read_bytes()[:4] != b"MTLB":
        raise BuildFailure(f"{METALLIB} is not a Metal library")
    licenses = stage / "licenses"
    for source in lock.sources.values():
        for relative in source.license_files:
            destination = licenses / source.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trees[source.name] / relative, destination)
    return stage


def manifest(
    config: Configuration,
    lock: Lock,
    tools: Toolchain,
    trees: Mapping[str, Path],
    stage: Path,
    audit: Mapping[str, Json],
    version: str,
    source_id: str,
) -> dict[str, Json]:
    package = stage / "phydrax_metal"
    indexing_source = ROOT / "phydrax" / "_jax_indexing.py"
    shutil.copyfile(indexing_source, package / "_jax_indexing.py")
    pjrt = header_version(
        trees["xla"] / "xla" / "pjrt" / "c" / "pjrt_c_api.h",
        r"#define PJRT_API_MAJOR (\d+).*?#define PJRT_API_MINOR (\d+)",
    )
    stablehlo = header_version(
        trees["stablehlo"] / "stablehlo" / "dialect" / "Version.h",
        r"getCurrentVersion\(\)\s*\{\s*return Version\((\d+),\s*(\d+),\s*(\d+)\)",
    )
    assets = {
        path.relative_to(package).as_posix(): file_hash(path)
        for path in sorted([*(package / "lib").rglob("*"), package / "_jax_indexing.py"])
        if path.is_file()
    }
    record: dict[str, Json] = {
        "version": version,
        "platform": PLATFORM,
        "jax_version": lock.jax_version,
        "jaxlib_version": lock.jaxlib_version,
        "pjrt_c_api": {"major": pjrt[0], "minor": pjrt[1]},
        "stablehlo_version": ".".join(str(part) for part in stablehlo),
        "minimum_macos": audit["minos"],
        "library": LIBRARY,
        "assets": assets,
        "sources": {
            "lock_sha256": lock.sha256,
            "entries": list(lock.entries),
            "patches": [
                {
                    "target": patch.target,
                    "origin": patch.origin,
                    "path": patch.path,
                    "sha256": patch.sha256,
                }
                for patch in lock.patches
            ],
        },
        "source_id": source_id,
        "toolchain": dict(tools.identity),
    }
    record["build_id"] = digest(canonical(record))
    (package / "build.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def build_wheel(config: Configuration, stage: Path, evidence: Evidence) -> Path:
    config.wheelhouse.mkdir(parents=True, exist_ok=True)
    script = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import _build_backend\n"
        "name = _build_backend.build_wheel(sys.argv[2], {'staging': sys.argv[3]})\n"
        "print(json.dumps(name))\n"
    )
    output = run(
        evidence,
        "wheel",
        (
            str(config.python),
            "-I",
            "-c",
            script,
            str(NATIVE),
            str(config.wheelhouse),
            str(stage),
        ),
        {"PATH": SYSTEM_PATH, "LANG": "C"},
        cwd=NATIVE,
    )
    name = json.loads(output.strip().splitlines()[-1])
    wheel = config.wheelhouse / name
    if not wheel.is_file():
        raise BuildFailure(f"backend did not write {wheel}")
    return wheel


def execute(config: Configuration) -> int:
    evidence = Evidence(config)
    try:
        lock = read_lock()
        tools = toolchain(config)
        evidence.record["toolchain"] = dict(tools.identity)
        evidence.record["lock_sha256"] = lock.sha256
        (config.build_dir / "tmp").mkdir(parents=True, exist_ok=True)
        environment = clean_environment(config, lock)
        environment["DEVELOPER_DIR"] = str(tools.identity["developer_dir"])
        trees = prepare_sources(config, lock, environment, evidence)
        common = cmake_common(config, lock, tools)
        prefixes = {
            "llvm-project": dependency(
                config,
                evidence,
                environment,
                "llvm-project",
                trees["llvm-project"] / "llvm",
                [*common, *LLVM_ARGUMENTS, f"-DPython3_EXECUTABLE={config.python}"],
                {
                    "llvm": lock.sources["llvm-project"].sha256,
                    "toolchain": dict(tools.identity),
                },
            ),
        }
        mlx_patches = [patch.sha256 for patch in lock.patches if patch.target == "mlx"]
        prefixes["mlx"] = dependency(
            config,
            evidence,
            environment,
            "mlx",
            trees["mlx"],
            [
                *common,
                *MLX_ARGUMENTS,
                f"-DFETCHCONTENT_SOURCE_DIR_METAL_CPP={trees['metal-cpp']}",
                f"-DFETCHCONTENT_SOURCE_DIR_JSON={trees['nlohmann-json']}",
                f"-DFETCHCONTENT_SOURCE_DIR_FMT={trees['fmt']}",
            ],
            {
                "sources": [
                    lock.sources[name].sha256
                    for name in ("mlx", "metal-cpp", "nlohmann-json", "fmt")
                ],
                "patches": mlx_patches,
                "toolchain": dict(tools.identity),
            },
        )
        if config.dependencies_only:
            evidence.record["status"] = "dependencies-built"
            return 0
        version = companion_version()
        source_id, native_files = source_identity(lock)
        evidence.record["source_id"] = source_id
        evidence.record["native_files"] = native_files
        stage = build_plugin(
            config,
            lock,
            tools,
            trees,
            prefixes,
            environment,
            evidence,
            version,
            source_id,
        )
        audit = macho_audit(
            stage / "phydrax_metal" / LIBRARY, lock, config.build_dir, evidence
        )
        evidence.record["build"] = manifest(
            config, lock, tools, trees, stage, audit, version, source_id
        )
        wheel = build_wheel(config, stage, evidence)
        observed_source, observed_files = source_identity(lock)
        if observed_source != source_id or observed_files != native_files:
            raise BuildFailure("Native inputs changed while compiling or packaging.")
        evidence.record["wheel"] = {"path": str(wheel), "sha256": file_hash(wheel)}
        evidence.record["status"] = "passed"
        return 0
    except (
        BuildFailure,
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
    ) as error:
        evidence.record["status"] = "failed"
        evidence.record["error"] = f"{type(error).__name__}: {error}"
        print(f"build_metal_wheels: {error}", file=sys.stderr)
        return 1
    finally:
        evidence.write()


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_configuration(argv))


if __name__ == "__main__":
    raise SystemExit(main())
