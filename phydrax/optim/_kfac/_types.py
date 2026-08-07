#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from jaxtyping import Array


@dataclass(frozen=True, slots=True)
class AffineBlockSpec:
    """Static mapping from one affine weight/bias block to flat parameters."""

    name: str
    indices: tuple[int, ...]
    output_size: int
    input_size: int
    has_bias: bool

    @property
    def parameter_count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True, slots=True)
class UncoveredBlockSpec:
    """A joint exact or diagonal block for supported non-affine parameters."""

    name: str
    indices: tuple[int, ...]
    approximation: Literal["exact", "diagonal"]

    @property
    def parameter_count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True, slots=True)
class ParameterLayout:
    """Static KFAC block layout for one trainable parameter PyTree."""

    affine_blocks: tuple[AffineBlockSpec, ...]
    uncovered_block: UncoveredBlockSpec | None
    parameter_count: int


class KronFactorState(NamedTuple):
    activation: Array
    sensitivity: Array
    initialized: Array


class DenseFactorState(NamedTuple):
    value: Array
    initialized: Array


class BlockCurvatureState(NamedTuple):
    affine: tuple[tuple[KronFactorState, ...], ...]
    uncovered: tuple[DenseFactorState, ...]


class AffineFactorObservation(NamedTuple):
    activation: Array
    sensitivity: Array


class BlockCurvatureObservation(NamedTuple):
    affine: tuple[AffineFactorObservation, ...]
    uncovered: Array | None


class KFACState(NamedTuple):
    step: Array
    curvature: BlockCurvatureState
    factor_updates: Array


class KFACMetrics(NamedTuple):
    """KFAC-native metrics emitted by one frozen functional update."""

    factor_updates: Array
    cg_iterations_max: Array
    cg_relative_residual_max: Array
    quadratic_update_norm: Array
    accepted_step_size: Array
    line_search_steps: Array


__all__ = [
    "AffineBlockSpec",
    "AffineFactorObservation",
    "BlockCurvatureObservation",
    "BlockCurvatureState",
    "DenseFactorState",
    "KFACMetrics",
    "KFACState",
    "KronFactorState",
    "ParameterLayout",
    "UncoveredBlockSpec",
]
