#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
from collections.abc import Sequence

from .base import BenchmarkAdapter


_ADAPTERS = {
    "phydrax": ("benchmarks.advanced_solvers.adapters.phydrax", "PhydraxAdapter"),
    "jax": ("benchmarks.advanced_solvers.adapters.jax", "JaxAdapter"),
    "lineax": ("benchmarks.advanced_solvers.adapters.lineax", "LineaxAdapter"),
    "optimistix": (
        "benchmarks.advanced_solvers.adapters.optimistix",
        "OptimistixAdapter",
    ),
    "scipy": ("benchmarks.advanced_solvers.adapters.scipy", "ScipyAdapter"),
    "pyamg": ("benchmarks.advanced_solvers.adapters.pyamg", "PyamgAdapter"),
    "amgcl": ("benchmarks.advanced_solvers.adapters.amgcl", "AmgclAdapter"),
    "amgx": ("benchmarks.advanced_solvers.adapters.amgx", "AmgxAdapter"),
    "petsc": ("benchmarks.advanced_solvers.adapters.petsc", "PetscAdapter"),
    "slepc": ("benchmarks.advanced_solvers.adapters.slepc", "SlepcAdapter"),
}


def adapter_names() -> tuple[str, ...]:
    return tuple(_ADAPTERS)


def load_adapter(name: str, /) -> BenchmarkAdapter:
    """Load one adapter module without importing any other optional integration."""
    if name not in _ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; choose from {', '.join(_ADAPTERS)}")
    module_name, class_name = _ADAPTERS[name]
    module = importlib.import_module(module_name)
    adapter_class = module.__dict__[class_name]
    return adapter_class()


def load_adapters(names: Sequence[str], /) -> dict[str, BenchmarkAdapter]:
    if len(set(names)) != len(names):
        raise ValueError("adapter names must not contain duplicates")
    return {name: load_adapter(name) for name in names}


__all__ = ["adapter_names", "load_adapter", "load_adapters"]
