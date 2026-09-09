#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class JointDecompositionTraining(StrictModule, NonTrainableState):
    """Monolithic optimization of every decomposition parameter."""

    num_iterations: int = eqx.field(static=True)

    def __init__(self, num_iterations: int, /):
        value = int(num_iterations)
        if value < 0:
            raise ValueError("num_iterations must be non-negative.")
        self.num_iterations = value


class BlockDecompositionTraining(StrictModule, NonTrainableState):
    """Block-coordinate local-field training on one coupled objective."""

    sweeps: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    sweep: Literal["jacobi", "gauss-seidel", "colored"] = eqx.field(static=True)
    active_patch_ids: tuple[str, ...] | None = eqx.field(static=True)
    fixed_patch_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        sweeps: int,
        inner_iterations: int,
        /,
        *,
        sweep: Literal["jacobi", "gauss-seidel", "colored"] = "jacobi",
        active_patch_ids: tuple[str, ...] | None = None,
        fixed_patch_ids: tuple[str, ...] = (),
    ):
        sweeps_ = int(sweeps)
        inner_ = int(inner_iterations)
        if sweeps_ < 0 or inner_ < 0:
            raise ValueError("sweeps and inner_iterations must be non-negative.")
        if sweep not in ("jacobi", "gauss-seidel", "colored"):
            raise ValueError("sweep must be 'jacobi', 'gauss-seidel', or 'colored'.")
        active = (
            None
            if active_patch_ids is None
            else tuple(str(value) for value in active_patch_ids)
        )
        fixed = tuple(str(value) for value in fixed_patch_ids)
        if active is not None and (not active or len(set(active)) != len(active)):
            raise ValueError("active_patch_ids must be distinct and non-empty.")
        if len(set(fixed)) != len(fixed):
            raise ValueError("fixed_patch_ids must be distinct.")
        if active is not None and set(active) & set(fixed):
            raise ValueError("A patch cannot be both active and fixed.")
        self.sweeps = sweeps_
        self.inner_iterations = inner_
        self.sweep = sweep
        self.active_patch_ids = active
        self.fixed_patch_ids = fixed


class SchwarzDecompositionTraining(StrictModule, NonTrainableState):
    """Inexact neural Schwarz iteration with frozen neighboring local fields."""

    sweeps: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    sweep: Literal["jacobi", "gauss-seidel", "colored"] = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    interface_tolerance: float | None = eqx.field(static=True)

    def __init__(
        self,
        sweeps: int,
        inner_iterations: int,
        /,
        *,
        sweep: Literal["jacobi", "gauss-seidel", "colored"] = "jacobi",
        relaxation: float = 1.0,
        interface_tolerance: float | None = None,
    ):
        sweeps_ = int(sweeps)
        inner_ = int(inner_iterations)
        relaxation_ = float(relaxation)
        if sweeps_ < 0 or inner_ < 0:
            raise ValueError("sweeps and inner_iterations must be non-negative.")
        if sweep not in ("jacobi", "gauss-seidel", "colored"):
            raise ValueError("sweep must be 'jacobi', 'gauss-seidel', or 'colored'.")
        if not math.isfinite(relaxation_) or not 0.0 < relaxation_ <= 1.0:
            raise ValueError("relaxation must be finite and in (0, 1].")
        if interface_tolerance is None:
            tolerance = None
        else:
            tolerance = float(interface_tolerance)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("interface_tolerance must be finite and non-negative.")
        self.sweeps = sweeps_
        self.inner_iterations = inner_
        self.sweep = sweep
        self.relaxation = relaxation_
        self.interface_tolerance = tolerance


DecompositionTraining = (
    JointDecompositionTraining | BlockDecompositionTraining | SchwarzDecompositionTraining
)


__all__ = [
    "BlockDecompositionTraining",
    "JointDecompositionTraining",
    "SchwarzDecompositionTraining",
]
