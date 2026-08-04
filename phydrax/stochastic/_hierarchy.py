#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import isfinite
from typing import Literal, overload, TypeAlias

import equinox as eqx

from .._frozendict import frozendict
from .._strict import StrictModule


RefinementAxis: TypeAlias = Literal["time", "space", "noise_rank", "surrogate", "other"]
NoiseCoupling: TypeAlias = Literal["shared", "nested", "independent"]


@overload
def _identifier(value: str, name: str, /, *, required: Literal[True]) -> str: ...


@overload
def _identifier(
    value: str | None, name: str, /, *, required: Literal[False]
) -> str | None: ...


def _identifier(value: str | None, name: str, /, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} must be a non-empty string.")
        return None
    if not isinstance(value, str) or not value:
        suffix = "" if required else " or None"
        raise ValueError(f"{name} must be a non-empty string{suffix}.")
    return value


def _fingerprint(parts: Sequence[object], /) -> str:
    digest = hashlib.sha256(b"phydrax-stochastic-hierarchy\0")
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class StochasticLevelSpec(StrictModule):
    """Static identity and compatibility contract for one approximation level."""

    metadata: frozendict[str, str] = eqx.field(static=True)
    level_id: str = eqx.field(static=True)
    refinement_index: int = eqx.field(static=True)
    refinement_axes: tuple[RefinementAxis, ...] = eqx.field(static=True)
    resolutions: tuple[float, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    parent_level_id: str | None = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    state_transfer_id: str | None = eqx.field(static=True)
    noise_coupling: NoiseCoupling = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        level_id: str,
        refinement_index: int,
        /,
        *,
        refinement_axes: Sequence[RefinementAxis],
        resolutions: Sequence[float],
        state_shape: Sequence[int],
        problem_id: str,
        observable_id: str,
        solver_id: str,
        approximation_id: str,
        parent_level_id: str | None = None,
        discretization_id: str | None = None,
        basis_id: str | None = None,
        state_transfer_id: str | None = None,
        noise_coupling: NoiseCoupling = "shared",
        metadata: Mapping[str, str] | None = None,
    ):
        identifier = _identifier(level_id, "level_id", required=True)
        assert identifier is not None
        index = int(refinement_index)
        if index < 0:
            raise ValueError("refinement_index must be non-negative.")
        axes = tuple(refinement_axes)
        allowed = ("time", "space", "noise_rank", "surrogate", "other")
        if not axes or any(axis not in allowed for axis in axes):
            raise ValueError(f"refinement_axes must contain values from {allowed!r}.")
        if len(set(axes)) != len(axes):
            raise ValueError("refinement_axes must be unique.")
        scales = tuple(float(value) for value in resolutions)
        if len(scales) != len(axes) or any(
            not isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError(
                "resolutions must contain one finite positive value per refinement axis."
            )
        shape = tuple(int(size) for size in state_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        if noise_coupling not in ("shared", "nested", "independent"):
            raise ValueError(
                "noise_coupling must be 'shared', 'nested', or 'independent'."
            )
        identities = frozendict({} if metadata is None else metadata)
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in identities.items()
        ):
            raise ValueError("metadata keys and values must be non-empty strings.")
        problem = _identifier(problem_id, "problem_id", required=True)
        observable = _identifier(observable_id, "observable_id", required=True)
        solver = _identifier(solver_id, "solver_id", required=True)
        approximation = _identifier(approximation_id, "approximation_id", required=True)
        parent = _identifier(parent_level_id, "parent_level_id", required=False)
        discretization = _identifier(
            discretization_id, "discretization_id", required=False
        )
        basis = _identifier(basis_id, "basis_id", required=False)
        transfer = _identifier(state_transfer_id, "state_transfer_id", required=False)
        self.level_id = identifier
        self.refinement_index = index
        self.refinement_axes = axes
        self.resolutions = scales
        self.state_shape = shape
        self.problem_id = problem
        self.observable_id = observable
        self.solver_id = solver
        self.approximation_id = approximation
        self.parent_level_id = parent
        self.discretization_id = discretization
        self.basis_id = basis
        self.state_transfer_id = transfer
        self.noise_coupling = noise_coupling
        self.metadata = identities
        self.fingerprint = _fingerprint(
            (
                identifier,
                index,
                axes,
                scales,
                shape,
                problem,
                observable,
                solver,
                approximation,
                parent,
                discretization,
                basis,
                transfer,
                noise_coupling,
                tuple(identities.items()),
            )
        )

    @property
    def resolution_by_axis(self) -> frozendict[str, float]:
        return frozendict(zip(self.refinement_axes, self.resolutions, strict=True))


class StochasticHierarchy(StrictModule):
    """Ordered one- or multi-axis approximation hierarchy with explicit coupling."""

    levels: tuple[StochasticLevelSpec, ...]
    hierarchy_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    refinement_axes: tuple[RefinementAxis, ...] = eqx.field(static=True)
    allow_multi_axis: bool = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[StochasticLevelSpec],
        /,
        *,
        hierarchy_id: str,
        allow_multi_axis: bool = False,
    ):
        values = tuple(levels)
        if not values or any(
            not isinstance(level, StochasticLevelSpec) for level in values
        ):
            raise ValueError("StochasticHierarchy requires valid non-empty levels.")
        identifier = _identifier(hierarchy_id, "hierarchy_id", required=True)
        assert identifier is not None
        expected_indices = tuple(range(len(values)))
        indices = tuple(level.refinement_index for level in values)
        if indices != expected_indices:
            raise ValueError(
                "Hierarchy refinement indices must be consecutive and start at zero."
            )
        level_ids = tuple(level.level_id for level in values)
        if len(set(level_ids)) != len(level_ids):
            raise ValueError("Hierarchy level IDs must be unique.")
        if values[0].parent_level_id is not None:
            raise ValueError("The base level cannot declare a parent level.")
        for position, level in enumerate(values[1:], start=1):
            parent = values[position - 1]
            if level.parent_level_id != parent.level_id:
                raise ValueError(
                    f"Level {level.level_id!r} must name {parent.level_id!r} as its parent."
                )
            if (
                level.state_shape != parent.state_shape
                and level.state_transfer_id is None
            ):
                raise ValueError(
                    f"Level {level.level_id!r} changes state_shape and requires "
                    "state_transfer_id."
                )
        problem_ids = {level.problem_id for level in values}
        observable_ids = {level.observable_id for level in values}
        if len(problem_ids) != 1 or len(observable_ids) != 1:
            raise ValueError(
                "Every hierarchy level must share one problem_id and observable_id."
            )
        axes = values[0].refinement_axes
        if any(level.refinement_axes != axes for level in values[1:]):
            raise ValueError("Every hierarchy level must refine the same declared axes.")
        multi_axis = bool(allow_multi_axis)
        if len(axes) > 1 and not multi_axis:
            raise ValueError(
                "Multi-axis refinement requires allow_multi_axis=True explicitly."
            )
        for coarse, fine in pairwise(values):
            for axis, coarse_scale, fine_scale in zip(
                axes, coarse.resolutions, fine.resolutions, strict=True
            ):
                if not fine_scale < coarse_scale:
                    raise ValueError(
                        f"Resolution for axis {axis!r} must strictly decrease from "
                        f"{coarse.level_id!r} to {fine.level_id!r}."
                    )
            if fine.noise_coupling == "nested":
                if coarse.basis_id is None or fine.basis_id is None:
                    raise ValueError("Nested noise levels require explicit basis IDs.")
                coarse_family = coarse.metadata.get("noise_family_id")
                fine_family = fine.metadata.get("noise_family_id")
                if coarse_family is None or coarse_family != fine_family:
                    raise ValueError(
                        "Nested noise levels require one shared metadata noise_family_id."
                    )
        self.levels = values
        self.hierarchy_id = identifier
        self.problem_id = values[0].problem_id
        self.observable_id = values[0].observable_id
        self.refinement_axes = axes
        self.allow_multi_axis = multi_axis
        self.fingerprint = _fingerprint(
            (identifier, tuple(level.fingerprint for level in values), multi_axis)
        )

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def coupled(self) -> bool:
        return all(level.noise_coupling != "independent" for level in self.levels[1:])

    def level(self, value: int | str, /) -> StochasticLevelSpec:
        if isinstance(value, int):
            return self.levels[value]
        matches = tuple(level for level in self.levels if level.level_id == value)
        if len(matches) != 1:
            raise KeyError(f"Unknown stochastic hierarchy level {value!r}.")
        return matches[0]


__all__ = [
    "NoiseCoupling",
    "RefinementAxis",
    "StochasticHierarchy",
    "StochasticLevelSpec",
]
