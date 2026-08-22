#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .adapters.base import CaseSpec, Tolerances
from .problems import default_problems


DEFAULT_ADAPTERS = (
    "phydrax",
    "jax",
    "lineax",
    "optimistix",
    "scipy",
    "pyamg",
    "amgcl",
    "amgx",
    "petsc",
    "slepc",
)
DEFAULT_CASES = (
    "linear-scalar",
    "linear-block",
    "nonlinear-root",
    "nonlinear-vi",
    "general-eigen",
    "continuation-fold",
    "optimization-unconstrained",
    "optimization-constrained",
    "optimization-proximal",
)
MATCHED_ROOT_CASES = (
    "nonlinear-root-dense",
    "nonlinear-root-matrix-free",
    "nonlinear-root-sparse-pde",
)
PROGRAM_CASES = (
    "optimization-linear-program",
    "optimization-quadratic-program",
    "optimization-conic-program",
)
AVAILABLE_CASES = DEFAULT_CASES + MATCHED_ROOT_CASES + PROGRAM_CASES


@dataclass(frozen=True)
class CampaignConfig:
    seed: int
    size: int
    warmup: int
    repeats: int
    adapters: tuple[str, ...] = DEFAULT_ADAPTERS
    cases: tuple[str, ...] = DEFAULT_CASES
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    max_steps: int = 500

    def __post_init__(self) -> None:
        if self.size < 8:
            raise ValueError("campaign size must be at least eight")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup must be nonnegative and repeats must be positive")
        tolerances = (self.relative_tolerance, self.absolute_tolerance)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("tolerances must be finite and nonnegative")
        if max(tolerances) == 0.0:
            raise ValueError("at least one convergence tolerance must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.adapters or not self.cases:
            raise ValueError("campaign adapters and cases must be non-empty")


PRESETS = {
    "ci": CampaignConfig(seed=20260816, size=16, warmup=1, repeats=3),
    "local": CampaignConfig(seed=20260816, size=64, warmup=2, repeats=10),
    "convex": CampaignConfig(
        seed=20260816,
        size=32,
        warmup=1,
        repeats=5,
        adapters=("phydrax", "mpax", "clarabel"),
        cases=PROGRAM_CASES,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-7,
        max_steps=2_000,
    ),
}


def build_cases(config: CampaignConfig, /) -> dict[str, CaseSpec]:
    problems = default_problems(size=config.size, seed=config.seed)
    tolerance = Tolerances(
        relative=config.relative_tolerance,
        absolute=config.absolute_tolerance,
        max_steps=config.max_steps,
    )
    solver_modes: dict[str, Literal["default", "dense", "matrix-free", "sparse"]] = {
        "nonlinear-root-dense": "dense",
        "nonlinear-root-matrix-free": "matrix-free",
        "nonlinear-root-sparse-pde": "sparse",
    }
    return {
        name: CaseSpec(
            name=name,
            problem=problem,
            tolerances=tolerance,
            solver_mode=solver_modes.get(name, "default"),
        )
        for name, problem in problems.items()
    }


__all__ = [
    "CampaignConfig",
    "AVAILABLE_CASES",
    "DEFAULT_ADAPTERS",
    "DEFAULT_CASES",
    "MATCHED_ROOT_CASES",
    "PRESETS",
    "PROGRAM_CASES",
    "build_cases",
]
