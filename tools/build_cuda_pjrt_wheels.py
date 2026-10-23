# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Build the locked Windows CUDA PJRT/compiler kernel stack, then package staging.

Run only in a standard CPython 3.13 x64 Windows environment. All mutable state,
including Bazel's output base, repositories, downloads and logs, is build-owned.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
import platform
import shlex
import shutil
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._accelerator_process import run, write_json_evidence


PACKAGE = ROOT / "native/cuda13_pjrt"
PROFILE = Path.home()
PJRT_TARGET = "@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
KERNEL_TARGET = "//jaxlib/tools:jax_cuda13_plugin_wheel"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def owned(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(PROFILE.resolve()):
        raise ValueError(f"Path is outside the authorized profile: {path}")
    return path


def module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load build module: {path}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def download(url: str, sha256: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        partial = destination.with_suffix(destination.suffix + ".partial")
        with (
            urllib.request.urlopen(url, timeout=120) as source,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if digest(partial) != sha256:
            raise ValueError(f"Archive hash mismatch: {url}")
        partial.replace(destination)
    if digest(destination) != sha256:
        raise ValueError(f"Cached input hash mismatch: {destination}")
    return destination


def member_path(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    result = directory.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return result


def archive_name(name: str, parent: str = "") -> str:
    """Normalize a tar namespace path without consulting the host filesystem."""
    if not name or name.startswith("/") or any(char in name for char in "\\:\0"):
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = parent.split("/") if parent else []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"Archive link escapes its namespace: {name!r}")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def tar_index(
    source: tarfile.TarFile,
) -> tuple[dict[str, tarfile.TarInfo], dict[str, set[str]]]:
    entries: dict[str, tarfile.TarInfo] = {}
    children: dict[str, set[str]] = {"": set()}
    case_names: dict[str, str] = {}
    for entry in source.getmembers():
        name = archive_name(entry.name)
        if not name:
            if entry.isdir():
                continue
            raise ValueError("Non-directory tar member names the archive root")
        if entry.name.startswith("/") or ".." in PurePosixPath(entry.name).parts:
            raise ValueError(f"Unsafe tar member name: {entry.name!r}")
        if not (entry.isfile() or entry.isdir() or entry.issym() or entry.islnk()):
            raise ValueError(f"Unsupported archive member: {entry.name}")
        key = name.casefold()
        if key in case_names and case_names[key] != name:
            raise ValueError(f"Case-ambiguous archive member: {name}")
        if name in entries and not (entries[name].isdir() and entry.isdir()):
            raise ValueError(f"Duplicate archive member: {name}")
        entries[name] = entry
        parts = name.split("/")
        for index, part in enumerate(parts):
            parent = "/".join(parts[:index])
            prefix = "/".join(parts[: index + 1])
            key = prefix.casefold()
            if key in case_names and case_names[key] != prefix:
                raise ValueError(f"Case-ambiguous archive directory: {prefix}")
            case_names[key] = prefix
            children.setdefault(parent, set()).add(part)
        if entry.isdir():
            children.setdefault(name, set())
    for name, entry in entries.items():
        if name in children and not entry.isdir():
            raise ValueError(f"Archive children traverse a non-directory member: {name}")
    return entries, children


def tar_target(
    name: str, entries: dict[str, tarfile.TarInfo], children: dict[str, set[str]]
) -> tuple[str, tarfile.TarInfo | None]:
    visited: set[str] = set()
    traversed_hardlink = False
    while True:
        if name in visited or len(visited) >= 256:
            raise ValueError(f"Cyclic or excessive archive link chain: {name}")
        visited.add(name)
        parts = name.split("/") if name else []
        for index in range(len(parts)):
            prefix = "/".join(parts[: index + 1])
            entry = entries.get(prefix)
            if entry is not None and (entry.issym() or entry.islnk()):
                traversed_hardlink |= entry.islnk()
                parent = prefix.rpartition("/")[0] if entry.issym() else ""
                target = archive_name(entry.linkname, parent)
                remainder = "/".join(parts[index + 1 :])
                name = archive_name(remainder, target) if remainder else target
                break
            if entry is not None and not entry.isdir() and index < len(parts) - 1:
                raise ValueError(f"Archive link traverses a regular file: {name}")
        else:
            entry = entries.get(name)
            if entry is None and name not in children and traversed_hardlink:
                raise ValueError(f"Archive hardlink target is missing: {name}")
            return name, entry


def extract_tar(source: tarfile.TarFile, directory: Path) -> list[dict[str, str]]:
    entries, children = tar_index(source)
    links: list[dict[str, str]] = []
    # Validate every link even when it is also reached through a directory alias.
    for name, entry in entries.items():
        if entry.issym() or entry.islnk():
            tar_target(name, entries, children)

    def materialize(name: str, destination: Path, ancestors: tuple[str, ...]) -> None:
        target, entry = tar_target(name, entries, children)
        original = entries.get(name)
        dangling = entry is None and target not in children
        if original is not None and (original.issym() or original.islnk()):
            link_bytes = original.linkname.encode(source.encoding, source.errors)
            links.append(
                {
                    "path": destination.relative_to(directory).as_posix(),
                    "archive_member": name,
                    "kind": "symlink" if original.issym() else "hardlink",
                    "link_target": original.linkname,
                    "link_target_sha256": hashlib.sha256(link_bytes).hexdigest(),
                    "resolved_target": target,
                    "representation": (
                        "git-core.symlinks=false"
                        if dangling
                        else "materialized-directory"
                        if target in children
                        else "materialized-file"
                    ),
                }
            )
        if dangling:
            if original is None or not original.issym():
                raise ValueError(f"Archive link target is missing: {name}")
            # Match Git core.symlinks=false: preserve the exact link payload,
            # never invent a target file or read a similarly named host file.
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(original.linkname.encode(source.encoding, source.errors))
            return
        if target in ancestors or len(ancestors) >= 256:
            raise ValueError(f"Cyclic or excessive archive directory link: {name}")
        if entry is None or entry.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(children[target]):
                materialize(
                    f"{target}/{child}" if target else child,
                    member_path(destination, child),
                    (*ancestors, target),
                )
            return
        # Only a resolved regular member reaches extractfile: tarfile's link
        # resolver does not normalize relative '..' targets consistently.
        data = source.extractfile(entry)
        if data is None:
            raise ValueError(f"Unreadable source member: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with data, destination.open("xb") as output:
            shutil.copyfileobj(data, output)

    for name in sorted(children[""]):
        materialize(name, member_path(directory, name), ())
    return sorted(links, key=lambda record: record["path"])


def extract(archive: Path, directory: Path) -> list[dict[str, str]]:
    """Publish a complete extraction atomically; never adopt a partial tree."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() or directory.is_symlink() or directory.is_junction():
        raise FileExistsError(f"Extraction destination already exists: {directory}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.extract-", dir=directory.parent)
    )
    links: list[dict[str, str]] = []
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for entry in source.infolist():
                    destination = member_path(temporary, entry.filename)
                    if entry.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.open(entry) as data, destination.open("xb") as output:
                            shutil.copyfileobj(data, output)
        else:
            with tarfile.open(archive) as source:
                if isinstance(
                    source.fileobj, (gzip.GzipFile, bz2.BZ2File, lzma.LZMAFile)
                ):
                    # Link resolution visits members out of archive order.
                    # Seeking compressed streams repeatedly reinflates their
                    # prefixes; inflate once, without changing archive semantics.
                    with tempfile.TemporaryFile(dir=directory.parent) as seekable:
                        source.fileobj.seek(0)
                        shutil.copyfileobj(source.fileobj, seekable, 1024 * 1024)
                        seekable.seek(0)
                        with tarfile.open(fileobj=seekable, mode="r:") as unpacked:
                            links = extract_tar(unpacked, temporary)
                else:
                    links = extract_tar(source, temporary)
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return links


class Builder:
    def __init__(self, args: argparse.Namespace, lock: dict[str, Any]) -> None:
        self.args, self.lock = args, lock
        self.work = owned(args.work_root)
        self.work.mkdir(parents=True, exist_ok=True)
        self.cache = self.work / "downloads"
        self.build = Path(tempfile.mkdtemp(prefix="build-", dir=self.work))
        self.sources: dict[str, Path] = {"phydrax": ROOT}
        self.source_extractions: dict[str, dict[str, Any]] = {}
        # Bazel batch mode serializes the client environment into the Windows
        # JVM command line. Unrelated inherited shell/agent settings can exceed
        # CreateProcess's 32768-character limit alongside the upstream bazelrc.
        inherited = {
            "SYSTEMROOT",
            "WINDIR",
            "SYSTEMDRIVE",
            "COMSPEC",
            "PATHEXT",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "OS",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
        }
        self.environment = {
            name.upper(): value
            for name, value in os.environ.items()
            if name.upper() in inherited
        }
        self.environment["PYTHONUTF8"] = "1"
        self.environment["UV_NO_CONFIG"] = "true"
        for name, relative in (
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("TMPDIR", "tmp"),
            ("XDG_CACHE_HOME", "cache"),
            ("PIP_CACHE_DIR", "pip-cache"),
            ("UV_CACHE_DIR", "uv-cache"),
            ("HOME", "home"),
            ("USERPROFILE", "home"),
        ):
            path = self.build / relative
            path.mkdir(exist_ok=True)
            self.environment[name] = str(path)
        self.commands: list[dict[str, Any]] = []
        git = Path(args.git).resolve(strict=True)
        if not git.is_file():
            raise ValueError(f"Explicit Git executable is not a file: {git}")
        self.git = str(git)
        self.python = owned(args.python)
        self.environment["PATH"] = os.pathsep.join(
            (
                str(self.python.parent),
                str(git.parent),
                str(Path(self.environment["SYSTEMROOT"]) / "System32"),
                str(
                    Path(self.environment["SYSTEMROOT"])
                    / "System32/WindowsPowerShell/v1.0"
                ),
            )
        )
        self.pe = module("_cuda_build_pe", ROOT / "phydrax/_pe_dependencies.py")

    def execute(self, command: list[str], cwd: Path) -> dict[str, Any]:
        if command[0] == "git":
            command = [self.git, *command[1:]]
        evidence = run(
            command, cwd=cwd, environment=self.environment, timeout=self.args.timeout
        )
        self.commands.append(dict(evidence))
        write_json_evidence(self.build / "commands.json", {"commands": self.commands})
        if evidence["returncode"] or evidence["timed_out"]:
            raise RuntimeError(
                f"Build command failed; full evidence: {evidence['stderr_path']}"
            )
        return dict(evidence)

    def tool_command(self, command: list[str], cwd: Path) -> None:
        self.execute(command, cwd)

    def acquire_sources(self) -> Path:
        entries = [
            *self.lock["sources"],
            {
                "name": "winjax",
                **self.lock["port_lineage"],
                "strip_prefix": "winjax-" + self.lock["port_lineage"]["revision"],
            },
        ]
        for entry in entries:
            name = entry["name"]
            archive = download(
                entry["url"],
                entry["sha256"],
                self.cache / f"{name}-{entry['sha256']}.tar.gz",
            )
            directory = self.build / "sources" / name
            links = extract(archive, directory)
            self.source_extractions[name] = {
                "archive_sha256": entry["sha256"],
                "link_policy": "materialize-resolved-preserve-dangling-symlink",
                "links": links,
            }
            self.sources[name] = directory / entry["strip_prefix"]
        for name in ("jax", "xla"):
            expected = self.lock["upstream_module_graph"][f"{name}_module_sha256"]
            if digest(self.sources[name] / "MODULE.bazel") != expected:
                raise ValueError(f"Unexpected {name} module graph")
        return self.sources["winjax"]

    def prepare_sources(self) -> Path:
        evidence = self.build / "sources.json"
        phase = "acquisition"
        try:
            lineage = self.acquire_sources()
            phase = "patching"
            self.patch_sources()
        except Exception as error:
            write_json_evidence(
                evidence,
                {
                    "status": "failed",
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sources_lock_sha256": digest(PACKAGE / "sources.lock.json"),
                    "sources": {
                        name: str(path) for name, path in sorted(self.sources.items())
                    },
                    "source_extractions": self.source_extractions,
                    "commands_path": str(self.build / "commands.json")
                    if self.commands
                    else None,
                },
            )
            print(f"Source phase failed; evidence: {evidence}", file=sys.stderr)
            raise
        write_json_evidence(
            evidence,
            {
                "status": "completed",
                "phase": "patching",
                "sources_lock_sha256": digest(PACKAGE / "sources.lock.json"),
                "sources": {
                    name: str(path) for name, path in sorted(self.sources.items())
                },
                "source_extractions": self.source_extractions,
                "commands_path": str(self.build / "commands.json"),
            },
        )
        return lineage

    def patch_sources(self) -> None:
        jax, xla = self.sources["jax"], self.sources["xla"]
        # --override_module bypasses archive_override's patch application. Apply
        # BOTH upstream JAX-owned XLA patches first; never replace their series.
        for name in ("xpu_triton_hook.patch", "nvptx_compiler_target.patch"):
            self.execute(
                [
                    self.git,
                    "apply",
                    "--unidiff-zero",
                    str(jax / "third_party/xla" / name),
                ],
                xla,
            )
        for patch in self.lock["patches"]:
            path = PACKAGE / patch["path"]
            if digest(path) != patch["sha256"]:
                raise ValueError(f"Port patch changed without lock update: {path}")
            if patch.get("phase") == "toolchain":
                continue
            if patch["source"] in ("jax", "xla"):
                self.execute(
                    [self.git, "apply", "--unidiff-zero", str(path)],
                    self.sources[patch["source"]],
                )
            else:
                owner = (
                    jax
                    if patch["source"] in ("abseil", "protobuf", "rules_python")
                    else xla
                )
                destination = owner / "third_party/windows_port"
                destination.mkdir(exist_ok=True)
                shutil.copyfile(path, destination / (path.parent.name + ".patch"))
        for owner in (jax, xla):
            destination = owner / "third_party/windows_port"
            (destination / "BUILD.bazel").write_text(
                'exports_files(glob(["*.patch"]))\n', encoding="utf-8"
            )

    def acquire_cuda(self, config: dict[str, Any]) -> None:
        cuda = owned(config["cuda"])
        cuda.mkdir(parents=True, exist_ok=False)
        for entry in self.lock["toolchain"]["cuda_redistributables"]:
            archive = download(
                "https://developer.download.nvidia.com/compute/cuda/redist/"
                + entry["relative_path"],
                entry["sha256"],
                self.cache / Path(entry["relative_path"]).name,
            )
            directory = self.build / "cuda-redist" / entry["name"]
            extract(archive, directory)
            roots = list(directory.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError(f"Unexpected CUDA archive layout: {archive}")
            source = roots[0]
            for child in source.iterdir():
                if child.is_dir():
                    target = cuda / child.name
                    if entry["name"] == "cuda_cupti":
                        target = (
                            cuda
                            / "extras/CUPTI"
                            / ("lib64" if child.name == "lib" else child.name)
                        )
                    shutil.copytree(child, target, dirs_exist_ok=True)
        cudnn_entry = self.lock["toolchain"]["cudnn_wheel"]
        archive = download(
            cudnn_entry["url"],
            cudnn_entry["sha256"],
            self.cache / cudnn_entry["filename"],
        )
        directory = self.build / "cudnn-wheel"
        extract(archive, directory)
        shutil.copytree(directory / "nvidia/cudnn", owned(config["cudnn"]))

    def windows_bazelrc(self, toolchain_rc: Path) -> Path:
        upstream = self.sources["jax"] / ".bazelrc"
        entries: list[tuple[str, list[str]]] = []
        for path in (upstream, toolchain_rc):
            for line in path.read_text(encoding="utf-8").splitlines():
                tokens = shlex.split(line, comments=True)
                if not tokens:
                    continue
                if line.rstrip().endswith("\\"):
                    raise ValueError(f"Review multiline Bazel configuration: {path}")
                if tokens[0] in ("import", "try-import"):
                    # The pinned source's optional local overrides are absent in
                    # fresh source trees. Never admit machine/user configuration.
                    if tokens[0] != "try-import" or len(tokens) != 2:
                        raise ValueError(f"Unexpected Bazel configuration import: {line}")
                    if ".." in PurePosixPath(tokens[1].replace("\\", "/")).parts:
                        raise ValueError(f"Escaping Bazel configuration import: {line}")
                    imported = Path(
                        tokens[1].replace("%workspace%", str(upstream.parent))
                    )
                    if not imported.is_relative_to(upstream.parent) or imported.exists():
                        raise ValueError(
                            f"Unexpected local Bazel configuration: {imported}"
                        )
                    continue
                entries.append((line, tokens))
        selected = {"windows", "winjax_cuda"}
        while True:
            previous = selected.copy()
            for _, tokens in entries:
                _, separator, name = tokens[0].partition(":")
                if separator and name not in selected:
                    continue
                for index, token in enumerate(tokens[1:], 1):
                    if token.startswith("--config="):
                        config_name = token.partition("=")[2]
                        if not config_name:
                            raise ValueError("Missing Bazel --config value")
                        selected.add(config_name)
                    elif token == "--config":
                        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                            raise ValueError("Missing Bazel --config value")
                        selected.add(tokens[index + 1])
            if selected == previous:
                break
        result = self.build / "windows.bazelrc"
        result.write_text(
            "\n".join(
                line
                for line, tokens in entries
                if ":" not in tokens[0] or tokens[0].partition(":")[2] in selected
            )
            + "\n",
            encoding="utf-8",
        )
        write_json_evidence(
            self.build / "bazelrc.json",
            {
                "upstream_sha256": digest(upstream),
                "toolchain_sha256": digest(toolchain_rc),
                "selected_configurations": sorted(selected),
                "selected_sha256": digest(result),
            },
        )
        return result

    def bazel(self, rc: Path, config: dict[str, Any]) -> list[str]:
        toolchain = self.lock["toolchain"]
        executable = download(
            toolchain["bazel_url"],
            toolchain["bazel_sha256"],
            self.cache / "bazel-9.2.0-windows-x86_64.exe",
        )
        self.environment["BAZEL_SH"] = str(owned(config["msys"]) / "usr/bin/bash.exe")
        self.environment["BAZEL_LLVM"] = str(owned(config["llvm"]))
        self.environment["BAZEL_VC"] = str(owned(config["msvc"]))
        self.environment["BAZEL_WINSDK_FULL_VERSION"] = config["windows_sdk_version"]
        self.environment["TF_NEED_CUDA"] = "1"
        self.environment["HERMETIC_CUDA_VERSION"] = "13.3.0"
        self.environment["PATH"] = os.pathsep.join(
            (
                str(owned(config["msys"]) / "usr/bin"),
                str(owned(config["llvm"]) / "bin"),
                str(owned(config["msvc"]) / "bin/Hostx64/x64"),
                str(
                    owned(config["windows_sdk"])
                    / "bin"
                    / config["windows_sdk_version"]
                    / "x64"
                ),
                self.environment["PATH"],
            )
        )
        return [
            str(executable),
            "--batch",
            f"--host_jvm_args=-Xmx{self.args.memory_mb // 3}m",
            "--nosystem_rc",
            "--nohome_rc",
            "--noworkspace_rc",
            f"--install_base={self.work / 'bazel-9.2.0'}",
            f"--output_user_root={self.build / 'bazel-user'}",
            f"--output_base={self.build / 'bazel-output'}",
            f"--bazelrc={self.windows_bazelrc(rc)}",
        ]

    def compile(self, bazel: list[str]) -> tuple[Path, Path]:
        common = [
            "--config=winjax_cuda",
            "--enable_bzlmod",
            f"--repository_cache={self.cache / 'bazel-repositories'}",
            "--lockfile_mode=update",
        ]
        # Reserve a third for the bounded Java heap and another twelfth for
        # JVM/native build overhead; the remainder is Bazel's action estimate.
        action_memory_mb = (
            self.args.memory_mb
            - self.args.memory_mb // 3
            - max(1, self.args.memory_mb // 12)
        )
        self.execute(
            [
                *bazel,
                "build",
                *common,
                f"--jobs={self.args.jobs}",
                f"--local_resources=memory={action_memory_mb}",
                PJRT_TARGET,
                KERNEL_TARGET,
            ],
            self.sources["jax"],
        )
        outputs: list[Path] = []
        for target, suffix in ((PJRT_TARGET, ".so"), (KERNEL_TARGET, ".whl")):
            evidence = self.execute(
                [*bazel, "cquery", *common, target, "--output=files"], self.sources["jax"]
            )
            paths = [
                self.sources["jax"] / line.strip()
                for line in Path(evidence["stdout_path"]).read_text().splitlines()
                if line.strip().endswith(suffix)
            ]
            if len(paths) != 1 or not paths[0].is_file():
                raise ValueError(f"Expected one real output for {target}: {paths}")
            outputs.append(paths[0])
        return outputs[0], outputs[1]

    def stage(self, pjrt: Path, wheel: Path, config: dict[str, Any]) -> Path:
        staging = self.build / "staging"
        staging.mkdir()
        shutil.copytree(
            PACKAGE / "python",
            staging,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        package = staging / "phydrax_cuda13_pjrt"
        shutil.copyfile(
            ROOT / "phydrax/_pe_dependencies.py", package / "_pe_dependencies.py"
        )
        shutil.copyfile(pjrt, package / "xla_cuda_plugin.dll")
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if (
                    name.startswith("jax_cuda13_plugin/")
                    and name.endswith((".pyd", ".py"))
                    and name != "jax_cuda13_plugin/__init__.py"
                ):
                    destination = member_path(staging, name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
        cuda = owned(config["cuda"])
        for basename, relative in (
            ("ptxas.exe", "cuda/bin/ptxas.exe"),
            ("nvlink.exe", "cuda/bin/nvlink.exe"),
            ("libdevice.10.bc", "cuda/nvvm/libdevice/libdevice.10.bc"),
        ):
            matches = list(cuda.rglob(basename))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one build-owned {basename}: {matches}"
                )
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(matches[0], destination)
        for entry in self.lock["license_sources"]:
            destination = staging / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = self.lock["licenses"][entry["path"]]
            data = self.license_bytes(entry)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"License identity mismatch: {entry['path']}")
            destination.write_bytes(data)
        self.write_manifest(staging, wheel, config)
        return staging

    def license_bytes(self, entry: dict[str, Any]) -> bytes:
        """Read exact locked notices, including stripped archive-root licenses."""
        if "archive_sha256" in entry:
            expected = entry["archive_sha256"]
            archive = member_path(
                self.cache / "bazel-repositories/content_addressable/sha256",
                expected + "/file",
            )
            if digest(archive) != expected:
                raise ValueError(f"License source archive identity mismatch: {archive}")
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as source:
                    data = source.read(entry["archive_member"])
            else:
                with tarfile.open(archive) as source:
                    member = source.getmember(entry["archive_member"])
                    if not member.isfile():
                        raise ValueError("License archive member must be a regular file")
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError("Unreadable license archive member")
                    with stream:
                        data = stream.read()
        else:
            if "url" in entry:
                expected = self.lock["licenses"][entry["path"]]
                path = download(
                    entry["url"], expected, self.cache / (expected + ".license")
                )
            elif "build_path" in entry:
                path = member_path(
                    self.work,
                    (self.build.relative_to(self.work) / entry["build_path"]).as_posix(),
                )
            else:
                path = self.sources[entry["source"]] / entry["source_path"]
            data = path.read_bytes()
        if "line_range" in entry:
            first, last = entry["line_range"]
            lines = data.splitlines(keepends=True)
            if not 1 <= first <= last <= len(lines):
                raise ValueError(f"Invalid complete-notice range: {entry['path']}")
            data = b"".join(lines[first - 1 : last])
        return data

    def write_manifest(self, staging: Path, wheel: Path, config: dict[str, Any]) -> None:
        assets, licenses = [], []
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(staging).as_posix()
            record = {"path": name, "sha256": digest(path), "size": path.stat().st_size}
            if name.startswith("licenses/"):
                licenses.append(record)
                continue
            role = "runtime"
            if path.suffix == ".pyd":
                role = "kernel"
            elif path.name == "xla_cuda_plugin.dll":
                role = "pjrt"
            elif "/cuda/" in name:
                role = "tool"
            elif path.name == "_pe_dependencies.py":
                role = "parser"
            record["role"] = role
            if path.suffix in (".dll", ".pyd", ".exe"):
                image = self.pe.inspect_pe(path)
                exports = {
                    item.name
                    for item in image.exports
                    if item.rva and item.forwarder is None
                }
                required = (
                    "GetPjrtApi"
                    if role == "pjrt"
                    else "PyInit_" + path.name.split(".")[0]
                    if role == "kernel"
                    else None
                )
                if image.machine != 0x8664 or (
                    required is not None and required not in exports
                ):
                    raise ValueError(f"AMD64/export contract failed: {path}")
            assets.append(record)
        project = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"]
        module_lock = self.sources["jax"] / "MODULE.bazel.lock"
        resume = self.build / "resume.json"
        native_lock = (
            json.loads(resume.read_text())["sources_lock_sha256"]
            if resume.is_file()
            else digest(PACKAGE / "sources.lock.json")
        )
        manifest = {
            "distribution": "phydrax-cuda13-pjrt",
            "version": project["version"],
            "cuda_major": 13,
            "jax_version": "0.12.0",
            "jaxlib_version": "0.12.0",
            "runtime_requirements": project["dependencies"],
            "python_tag": "cp313",
            "abi_tag": "cp313",
            "platform_tag": "win_amd64",
            "sources_lock_sha256": digest(PACKAGE / "sources.lock.json"),
            "sources": self.lock["sources"],
            "source_extractions": self.source_extractions,
            "patches": self.lock["patches"],
            "toolchain": self.lock["toolchain"],
            "capabilities": self.lock["capabilities"],
            "unsupported_capabilities": self.lock["unsupported_capabilities"],
            "assets": assets,
            "licenses": licenses,
            "license_sources": self.lock["license_sources"],
            "native_build": {
                "sources_lock_sha256": native_lock,
                "commands_sha256": digest(self.build / "commands.json"),
                "source_preparation_sha256": digest(self.build / "sources.json"),
                "resume_sha256": digest(resume) if resume.is_file() else None,
            },
            "kernel_wheel": {"filename": wheel.name, "sha256": digest(wheel)},
            "compiler_inputs": self.compiler_inputs(config),
            "bazel_module_lock_sha256": digest(module_lock),
            "qualification": "Native assets built; installed-wheel GPU qualification is separate and not implied.",
        }
        manifest["build_id"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "phydrax_cuda13_pjrt/manifest.json").write_bytes(canonical(manifest))

    def compiler_inputs(self, config: dict[str, Any]) -> list[dict[str, str]]:
        records = []
        for name in ("msvc", "windows_sdk", "llvm", "msys"):
            root = owned(config[name])
            tree = hashlib.sha256()
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    tree.update(
                        path.relative_to(root).as_posix().encode()
                        + b"\0"
                        + bytes.fromhex(digest(path))
                    )
            records.append({"name": name, "tree_sha256": tree.hexdigest()})
        return records


def refresh_recipe_lock() -> None:
    """Explicitly admit maintainer-reviewed recipe edits; never refresh source pins."""
    path = PACKAGE / "sources.lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    for relative in lock["recipe_inputs"]:
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError(f"Recipe input escapes the worktree: {relative}")
        lock["recipe_inputs"][relative] = digest(source)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python")
    parser.add_argument("--git")
    parser.add_argument("--work-root")
    parser.add_argument("--toolchain-inputs")
    parser.add_argument("--output-dir")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=14400)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--sources-only", action="store_true")
    phase.add_argument("--refresh-recipe-lock", action="store_true")
    args = parser.parse_args()
    if args.refresh_recipe_lock:
        refresh_recipe_lock()
        return
    if args.python is None or args.git is None or args.work_root is None:
        parser.error(
            "--python, --git, and --work-root are required for acquisition/build phases"
        )
    if not args.sources_only and (
        args.toolchain_inputs is None or args.output_dir is None
    ):
        parser.error(
            "--toolchain-inputs and --output-dir are required unless --sources-only is selected"
        )
    if (
        sys.platform != "win32"
        or sys.version_info[:2] != (3, 13)
        or platform.machine().lower() not in ("amd64", "x86_64")
    ):
        raise RuntimeError("Use standard CPython 3.13 win_amd64")
    if owned(args.python) != Path(sys.executable).resolve():
        raise ValueError("Invoke this script with the exact --python interpreter")
    if args.jobs < 1 or args.memory_mb < 3 or args.timeout <= 0:
        raise ValueError("Build limits require jobs >=1, memory >=3 MiB, and timeout >0")
    lock = json.loads((PACKAGE / "sources.lock.json").read_text())
    for relative, expected in lock["recipe_inputs"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Build recipe changed without lock update: {relative}")
    builder = Builder(args, lock)
    lineage = builder.prepare_sources()
    if args.sources_only:
        print(builder.build / "sources.json")
        return
    config = json.loads(owned(args.toolchain_inputs).read_text())
    builder.acquire_cuda(config)
    configure = module("_cuda_explicit_toolchain", PACKAGE / "toolchains/configure.py")
    # The renderer uses only this call's environment, including profile-local TEMP.
    os.environ.update(builder.environment)
    rc = configure.configure(
        config,
        lineage=lineage,
        work=builder.build,
        xla=builder.sources["xla"],
        python=builder.python,
        execute=builder.tool_command,
        pe=builder.pe,
    )
    bazel = builder.bazel(rc, config)
    if args.prepare_only:
        write_json_evidence(
            builder.build / "prepared.json",
            {
                "sources_lock_sha256": digest(PACKAGE / "sources.lock.json"),
                "source_extractions": builder.source_extractions,
                "bazel": bazel,
                "targets": [PJRT_TARGET, KERNEL_TARGET],
            },
        )
        return
    pjrt, wheel = builder.compile(bazel)
    staging = builder.stage(pjrt, wheel, config)
    output = owned(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backend = module("_cuda_owned_wheel_backend", PACKAGE / "_build_backend.py")
    filename = backend.build_wheel(
        str(output), {"staging": str(staging), "kernel-wheel": str(wheel)}
    )
    print(output / filename)


if __name__ == "__main__":
    main()
