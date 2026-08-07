#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class KFAC:
    """Configuration for Phydrax-native type-II GGN KFAC."""

    learning_rate: float = 1.0
    damping: float = 1e-3
    factor_decay: float = 0.95
    factor_update_period: int = 1
    factor_chunk_size: int = 32
    approximation: Literal["expand", "reduce"] = "expand"
    cg_max_steps: int = 50
    cg_relative_tolerance: float = 1e-6
    exact_block_max_size: int = 64
    uncovered: Literal["error", "diagonal"] = "error"
    max_update_norm: float | None = None
    line_search: bool = True
    line_search_shrink: float = 0.5
    line_search_c1: float = 1e-4
    line_search_max_steps: int = 10

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.damping <= 0.0:
            raise ValueError("damping must be positive.")
        if not 0.0 <= self.factor_decay < 1.0:
            raise ValueError("factor_decay must lie in [0, 1).")
        if self.factor_update_period <= 0:
            raise ValueError("factor_update_period must be positive.")
        if self.factor_chunk_size <= 0:
            raise ValueError("factor_chunk_size must be positive.")
        if self.approximation not in ("expand", "reduce"):
            raise ValueError("approximation must be either 'expand' or 'reduce'.")
        if self.cg_max_steps <= 0:
            raise ValueError("cg_max_steps must be positive.")
        if self.cg_relative_tolerance <= 0.0:
            raise ValueError("cg_relative_tolerance must be positive.")
        if self.exact_block_max_size < 0:
            raise ValueError("exact_block_max_size must be nonnegative.")
        if self.uncovered not in ("error", "diagonal"):
            raise ValueError("uncovered must be either 'error' or 'diagonal'.")
        if self.max_update_norm is not None and self.max_update_norm <= 0.0:
            raise ValueError("max_update_norm must be positive when provided.")
        if not 0.0 < self.line_search_shrink < 1.0:
            raise ValueError("line_search_shrink must lie in (0, 1).")
        if not 0.0 < self.line_search_c1 < 1.0:
            raise ValueError("line_search_c1 must lie in (0, 1).")
        if self.line_search_max_steps <= 0:
            raise ValueError("line_search_max_steps must be positive.")


def kfac(
    *,
    learning_rate: float = 1.0,
    damping: float = 1e-3,
    factor_decay: float = 0.95,
    factor_update_period: int = 1,
    factor_chunk_size: int = 32,
    approximation: Literal["expand", "reduce"] = "expand",
    cg_max_steps: int = 50,
    cg_relative_tolerance: float = 1e-6,
    exact_block_max_size: int = 64,
    uncovered: Literal["error", "diagonal"] = "error",
    max_update_norm: float | None = None,
    line_search: bool = True,
    line_search_shrink: float = 0.5,
    line_search_c1: float = 1e-4,
    line_search_max_steps: int = 10,
) -> KFAC:
    """Construct a structured KFAC optimizer for `FunctionalSolver`."""

    return KFAC(
        learning_rate=learning_rate,
        damping=damping,
        factor_decay=factor_decay,
        factor_update_period=factor_update_period,
        factor_chunk_size=factor_chunk_size,
        approximation=approximation,
        cg_max_steps=cg_max_steps,
        cg_relative_tolerance=cg_relative_tolerance,
        exact_block_max_size=exact_block_max_size,
        uncovered=uncovered,
        max_update_norm=max_update_norm,
        line_search=line_search,
        line_search_shrink=line_search_shrink,
        line_search_c1=line_search_c1,
        line_search_max_steps=line_search_max_steps,
    )


__all__ = ["KFAC", "kfac"]
