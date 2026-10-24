# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Render the pinned port's toolchain in a private build tree, without probing."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any


PROFILE = Path.home()


def owned_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(PROFILE.resolve()):
        raise ValueError(f"Build inputs and outputs must be profile-local: {path}")
    return path


def configure(
    config: dict[str, Any],
    *,
    lineage: Path,
    work: Path,
    xla: Path,
    python: Path,
    execute: Callable[[list[str], Path], None],
    pe: Any,
) -> Path:
    """Use explicit MSVC/SDK/LLVM paths; never call lineage's discovery/main."""
    required = (
        "msvc",
        "windows_sdk",
        "windows_sdk_version",
        "llvm",
        "llvm_version",
        "msys",
        "cuda",
        "cudnn",
    )
    if set(config) != set(required):
        raise ValueError(f"Toolchain JSON requires exactly {required}")
    paths = {
        name: owned_path(config[name])
        for name in ("msvc", "windows_sdk", "llvm", "msys", "cuda", "cudnn")
    }
    msvc, sdk, llvm = paths["msvc"], paths["windows_sdk"], paths["llvm"]
    sdk_version, llvm_version = config["windows_sdk_version"], config["llvm_version"]
    if int(llvm_version.split(".")[0]) < 20:
        raise ValueError("The supported MSVC 14.51 STL requires LLVM >=20")
    required_files = [
        msvc / "bin/Hostx64/x64/cl.exe",
        msvc / "lib/x64/msvcrt.lib",
        sdk / f"Lib/{sdk_version}/um/x64/kernel32.lib",
        sdk / f"bin/{sdk_version}/x64/rc.exe",
        llvm / "bin/clang-cl.exe",
        llvm / "bin/clang++.exe",
        llvm / "bin/llvm-lib.exe",
        paths["msys"] / "usr/bin/bash.exe",
        paths["cuda"] / "include/cuda.h",
        paths["cudnn"] / "include/cudnn.h",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing explicit toolchain prerequisite: {path}")
    toolchains = work / "toolchains"
    shutil.copytree(lineage / "toolchains", toolchains)
    for patch in sorted(
        (Path(__file__).resolve().parents[1] / "patches/winjax").glob("*.patch")
    ):
        execute(["git", "apply", "--unidiff-zero", str(patch)], toolchains)
    spec = importlib.util.spec_from_file_location(
        "_cuda_port_configuration", lineage / "configure.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned build-only configuration")
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    for name, value in (
        ("ROOT", work),
        ("TOOLCHAINS", toolchains),
        ("TEMPLATES", toolchains / "templates"),
        ("PATCHES", lineage / "patches"),
    ):
        setattr(port, name, str(value))
    includes = [msvc / "include"] + [
        sdk / f"Include/{sdk_version}/{name}"
        for name in ("ucrt", "shared", "um", "winrt")
    ]
    libraries = [msvc / "lib/x64"] + [
        sdk / f"Lib/{sdk_version}/{name}/x64" for name in ("ucrt", "um")
    ]
    bins = [
        llvm / "bin",
        msvc / "bin/Hostx64/x64",
        sdk / f"bin/{sdk_version}/x64",
        paths["msys"] / "usr/bin",
        python.parent,
    ]
    cap = {
        "INCLUDE": ";".join(map(str, includes)),
        "LIB": ";".join(map(str, libraries)),
        "PATH": ";".join(map(str, bins)),
    }
    # Unused cross-architecture definitions are rendered, but never selected.
    vs = {
        "msvc_dir": str(msvc),
        "vc_root": str(msvc),
        "vc_ver": msvc.name,
        "vcvarsall": str(work / "unused-vcvarsall.bat"),
    }
    ll = {
        "bin": str(llvm / "bin"),
        "resource_dir": str(llvm / "lib/clang" / llvm_version.split(".")[0]),
    }
    cuda = {"root": str(paths["cuda"]), "version": "13.3"}
    shutil.copytree(paths["cuda"] / "include/cccl", toolchains / "cccl_patched")
    for patch in sorted(
        (Path(__file__).resolve().parents[1] / "patches/cccl").glob("*.patch")
    ):
        execute(
            ["git", "apply", "--unidiff-zero", str(patch)], toolchains / "cccl_patched"
        )
    port.gen_local_config_cc(
        vs, cap, cap, ll, cuda, {"root": str(paths["msys"])}, str(python)
    )
    port.gen_llvm_bin_path(ll, cuda, ["sm_120"])
    port.gen_local_config_cuda_win(cuda, "9", ["sm_120"])
    # Bazel 9 removes the native C++ globals. Keep the existing toolchain
    # features, importing their owners from the selected rules_cc module.
    for name in ("windows_cc_toolchain_config.bzl", "armeabi_cc_toolchain_config.bzl"):
        path = toolchains / "local_config_cc" / name
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "@bazel_tools//tools/build_defs/cc:action_names.bzl",
            "@rules_cc//cc:action_names.bzl",
        )
        text = text.replace(
            "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
            "@rules_cc//cc:cc_toolchain_config_lib.bzl",
        )
        text = (
            'load("@rules_cc//cc/common:cc_common.bzl", "cc_common")\nload("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")\n'
            + text
        )
        path.write_text(text, encoding="utf-8")
    cudnn = toolchains / "cudnn/unpacked/nvidia/cudnn"
    shutil.copytree(paths["cudnn"], cudnn)
    (cudnn / "lib").mkdir(exist_ok=True)
    for dll_name, lib_name in port.CUDNN_LIB_PAIRS:
        dll = cudnn / "bin" / dll_name
        image = pe.inspect_pe(dll)
        if image.machine != 0x8664:
            raise ValueError(f"cuDNN must be AMD64: {dll}")
        exports = [item for item in image.exports if item.rva]
        if not exports:
            raise ValueError(f"cuDNN has no exports: {dll}")
        definition = cudnn / "lib" / (Path(lib_name).stem + ".def")
        declarations = [
            f"{item.name} @{item.ordinal}"
            if item.name
            else f"ordinal_{item.ordinal} @{item.ordinal} NONAME"
            for item in exports
        ]
        definition.write_text(
            f"LIBRARY {dll_name}\nEXPORTS\n" + "\n".join(declarations) + "\n",
            encoding="ascii",
        )
        execute(
            [
                str(llvm / "bin/llvm-lib.exe"),
                f"/def:{definition}",
                "/machine:x64",
                f"/out:{cudnn / 'lib' / lib_name}",
            ],
            work,
        )
    # Copy, rather than junction, into the private repository overrides. This
    # works without Developer Mode and never mutates installed SDK/toolkit files.
    for repository, mappings in port.junction_manifest(cuda).items():
        for relative, source in mappings.items():
            shutil.copytree(
                source,
                toolchains / "cuda_repos" / repository / relative,
                dirs_exist_ok=True,
            )
    # CUDA 13 ships static dispatch-loader thunks under these names, not
    # import libraries or the full static CUDA runtime. Preserve NVIDIA's
    # versioned dispatch ABI; cudart_static.lib remains a distinct target.
    runtime_build = toolchains / "cuda_repos/cuda_cudart/BUILD"
    text = runtime_build.read_text(encoding="utf-8")
    for library in ("cuda.lib", "cudart.lib"):
        before = f'interface_library = "lib/{library}",\n    system_provided = 1,'
        if text.count(before) != 1:
            raise ValueError(f"Unexpected CUDA loader target for {library}")
        text = text.replace(before, f'static_library = "lib/{library}",')
    # The private repository deliberately contains the merged toolkit include
    # tree. Declare that actual tree, including CRT/curand/NVTX transitive
    # headers, rather than hiding undeclared inclusions with compiler flags.
    header_glob = "    srcs = glob([\n"
    if text.count(header_glob) != 1:
        raise ValueError("Unexpected CUDA runtime header declaration")
    text = text.replace(header_glob, header_glob + '        "include/**",\n')
    runtime_build.write_text(text, encoding="utf-8")
    # Bazel 9 separately discovers its Windows resource compiler through
    # vcvarsall. Supply the real SDK compiler through the same toolchain API.
    resource = toolchains / "local_config_winsdk"
    resource.mkdir()
    (resource / "MODULE.bazel").write_text(
        'module(name = "local_config_winsdk")\n', encoding="utf-8"
    )
    (resource / "rc_x64.bat").write_text(
        '@echo off\n'
        f'set "INCLUDE={cap["INCLUDE"]}"\n'
        f'@"{sdk / f"bin/{sdk_version}/x64/rc.exe"}" %*\n',
        encoding="utf-8", newline="\r\n",
    )
    (resource / "BUILD.bazel").write_text(
        'load("@bazel_tools//tools/res:winsdk_toolchain.bzl", '
        '"WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE", '
        '"windows_resource_compiler_toolchain")\n'
        'windows_resource_compiler_toolchain(\n'
        '    name = "local_x64_tc",\n'
        '    rc_exe = "rc_x64.bat",\n'
        ')\n'
        'toolchain(\n'
        '    name = "local_x64",\n'
        '    exec_compatible_with = ["@platforms//os:windows", "@platforms//cpu:x86_64"],\n'
        '    target_compatible_with = ["@platforms//os:windows"],\n'
        '    toolchain = ":local_x64_tc",\n'
        '    toolchain_type = WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE,\n'
        '    visibility = ["//visibility:public"],\n'
        ')\n',
        encoding="utf-8",
    )
    port.gen_bazelrc(str(xla))
    rc = toolchains / "winjax_cuda.bazelrc"
    text = rc.read_text(encoding="utf-8")
    text += "\ncommon:winjax_cuda --repo_env=TF_NEED_CUDA=1\ncommon:winjax_cuda --repo_env=HERMETIC_CUDA_VERSION=13.3.0\ncommon:winjax_cuda --repo_env=HERMETIC_PYTHON_VERSION=3.13\ncommon:winjax_cuda --repo_env=ML_WHEEL_TYPE=release\n"
    text += (
        "common:winjax_cuda --override_repository="
        f"bazel_tools+winsdk_configure+local_config_winsdk={resource.as_posix()}\n"
    )
    rc.write_text(text, encoding="utf-8")
    (work / "explicit-toolchain.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return rc
