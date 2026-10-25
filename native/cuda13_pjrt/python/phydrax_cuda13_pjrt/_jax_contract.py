# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact source admission for the stock JAX 0.12 Windows integration hooks.

Digests are SHA-256 of complete AST-delimited upstream function source (including
multiline signatures and bodies), with trailing whitespace stripped and one
final newline, at JAX commit
8dcb7aeced770bd22531b42d0b925c7342787c27. Installed frontend files are never edited.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType


HOOK_HASHES = {
    "jax._src.xla_bridge.register_backend_factory": "8bfb2cb61fe144a995e3f64401d85f9f523ca7d3a939db2c78a72791d3a3af23",
    "jax._src.xla_bridge.register_plugin": "47fef9bba50948662a29e1b00233c19e8147ae79a6af11c5cc42c3c404d6b6f1",
    "jax._src.xla_bridge.backends": "fdc8ccefa62049010a075479c76727eb8bc25c543e65b7054592bd8c5ba90ac7",
    "jax._src.xla_bridge.discover_pjrt_plugins": "b205806a0c361bb73ccf38defc85738044f4b0d60bd13ef8f07db7bd293eccb5",
    "jax._src.xla_bridge._discover_and_register_pjrt_plugins": "fb62e04a984ec2a7be0203f3ffd74a518695ee6a81157ab5372ef980624ceb72",
    "jax._src.xla_bridge.make_pjrt_c_api_client": "c0bb42cd1768bb883eea493c96fc0dfe84227d120d2e0e6fe72344f5643faea3",
    "jax._src.hardware_utils.has_visible_nvidia_gpu": "83383112ccb33b5338adfa343ca3a0efab86843fee5ee12a3708466802841a80",
    "jax._src.lib._cuda_path": "ca1931a9020cc4fa5287eaf30a99a24cd6f954f2ffbc549da069f65714c4a4a5",
    "jaxlib.xla_client.generate_pjrt_gpu_plugin_options": "28f292a5ba549e13181e046190eef2ed026d85dd40bf8d6f0f5a5980b4bc02fb",
}


def _digest(source: str) -> str:
    return hashlib.sha256((source.rstrip() + "\n").encode()).hexdigest()


def validate_installed_sources() -> None:
    """Admit source files using only metadata, including during partial JAX import."""
    for distribution in ("jax", "jaxlib"):
        if metadata.version(distribution) != "0.12.0":
            raise RuntimeError(
                f"Windows CUDA runtime requires stock {distribution}==0.12.0"
            )
    by_module: dict[str, dict[str, str]] = {}
    for qualified, digest in HOOK_HASHES.items():
        module, name = qualified.rsplit(".", 1)
        by_module.setdefault(module, {})[name] = digest
    for module, expected in by_module.items():
        owner = metadata.distribution(module.split(".", 1)[0])
        relative = module.replace(".", "/") + (
            "/__init__.py" if module == "jax._src.lib" else ".py"
        )
        located = owner.locate_file(relative)
        if not isinstance(located, os.PathLike):
            raise RuntimeError("Stock JAX must be installed as filesystem files")
        path = Path(located)
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        actual = {}
        for node in ast.parse(source, filename=str(path)).body:
            if isinstance(node, ast.FunctionDef) and node.name in expected:
                if node.name in actual or node.decorator_list:
                    raise RuntimeError(
                        f"Unexpected wrapped/duplicate JAX hook: {module}.{node.name}"
                    )
                actual[node.name] = _digest(
                    "\n".join(lines[node.lineno - 1 : node.end_lineno])
                )
        if actual != expected:
            raise RuntimeError(
                f"Unsupported or modified stock JAX integration source: {path}"
            )


def validate_loaded_hook(module: ModuleType, name: str, function: object) -> None:
    """Authenticate the executing code, never an arbitrary ``__wrapped__`` chain."""
    owner = module.__name__
    if sys.modules.get(owner) is not module:
        raise RuntimeError(f"Noncanonical loaded JAX module: {owner}")
    qualified = f"{owner}.{name}"
    if (
        not inspect.isfunction(function)
        or function.__module__ != owner
        or function.__qualname__ != name
        or _digest(inspect.getsource(function.__code__)) != HOOK_HASHES[qualified]
    ):
        raise RuntimeError(
            f"Unsupported or replaced JAX integration hook: {qualified}"
        )


def validate_loaded_hooks(*modules: ModuleType) -> None:
    """Refuse replaced in-memory hooks before installing the narrow adaptation."""
    for module in modules:
        if sys.modules.get(module.__name__) is not module:
            raise RuntimeError(f"Noncanonical loaded JAX module: {module.__name__}")
        for qualified in HOOK_HASHES:
            owner, name = qualified.rsplit(".", 1)
            if owner == module.__name__:
                validate_loaded_hook(module, name, vars(module).get(name))
