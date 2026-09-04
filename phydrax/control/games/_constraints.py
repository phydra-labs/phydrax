#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constraint ownership, layouts, and sampled open-loop game feasibility."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite, prod
from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._trajectory_optimization import (
    BoundedPathConstraint,
    BoundedTrajectoryConstraint,
    TrajectoryOptimizationView,
)
from ._layout import PlayerControlPartition


class GameConstraintScope(IntEnum):
    """Stable ownership classes for open-loop game constraints."""

    PLAYER_LOCAL = 0
    PLAYER_OWNED_COUPLED = 1
    SHARED = 2


class GameConstraintSite(IntEnum):
    """Stable sites at which a game constraint is evaluated."""

    PATH = 0
    TERMINAL = 1
    TRAJECTORY = 2


class GameFeasibilityStatus(IntEnum):
    """Stable case-local statuses for game constraint evaluation."""

    FEASIBLE = 0
    INFEASIBLE = 1
    NONFINITE_RESIDUAL = 2


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _identifiers(value: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    resolved = tuple(value)
    if any(not isinstance(item, str) or not item for item in resolved):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates.")
    return resolved


def _residual_shape(value: Sequence[int], /) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("residual_shape must be a sequence of positive integers.")
    raw = tuple(value)
    if any(isinstance(size, bool) for size in raw):
        raise TypeError("residual_shape dimensions must be integers, not booleans.")
    shape = tuple(index(size) for size in raw)
    if any(size <= 0 for size in shape):
        raise ValueError("residual_shape dimensions must be positive.")
    return shape


def _standard_bound(value: Any, shape: tuple[int, ...], name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    broadcast_shape = jnp.broadcast_shapes(array.shape, shape)
    if broadcast_shape != shape:
        raise ValueError(f"{name} must broadcast exactly to residual_shape {shape}.")
    return jnp.broadcast_to(array, shape)


class GameConstraintBlock(StrictModule):
    """One explicitly owned fixed-shape equality or ``residual <= 0`` block.

    The wrapped bounded constraint is the sole callback convention. Equality blocks
    must use zero lower and upper bounds; inequality blocks must use a negative
    infinite lower bound and zero upper bound. The callback value is consequently
    the raw residual reported by feasibility evaluation.
    """

    constraint: BoundedPathConstraint | BoundedTrajectoryConstraint
    scope: GameConstraintScope = eqx.field(static=True)
    participants: tuple[str, ...] = eqx.field(static=True)
    owner: str | None = eqx.field(static=True)
    site: GameConstraintSite = eqx.field(static=True)
    equality: bool = eqx.field(static=True)
    residual_shape: tuple[int, ...] = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)
    state_dependent: bool = eqx.field(static=True)
    control_dependencies: tuple[str, ...] = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraint: BoundedPathConstraint | BoundedTrajectoryConstraint,
        /,
        *,
        scope: GameConstraintScope,
        participants: Sequence[str],
        owner: str | None,
        site: GameConstraintSite,
        equality: bool,
        residual_shape: Sequence[int],
        time_dependent: bool,
        state_dependent: bool,
        control_dependencies: Sequence[str],
    ):
        if not isinstance(scope, GameConstraintScope):
            raise TypeError("scope must be a GameConstraintScope.")
        if not isinstance(site, GameConstraintSite):
            raise TypeError("site must be a GameConstraintSite.")
        if not isinstance(equality, bool):
            raise TypeError("equality must be a bool.")
        if not isinstance(time_dependent, bool):
            raise TypeError("time_dependent must be a bool.")
        if not isinstance(state_dependent, bool):
            raise TypeError("state_dependent must be a bool.")
        if site is GameConstraintSite.PATH:
            if not isinstance(constraint, BoundedPathConstraint):
                raise TypeError("PATH blocks require a BoundedPathConstraint.")
        elif not isinstance(constraint, BoundedTrajectoryConstraint):
            raise TypeError(
                "TERMINAL and TRAJECTORY blocks require a BoundedTrajectoryConstraint."
            )

        participants_ = _identifiers(participants, "participants")
        if not participants_:
            raise ValueError("GameConstraintBlock requires at least one participant.")
        dependencies = _identifiers(control_dependencies, "control_dependencies")
        if any(player_id not in participants_ for player_id in dependencies):
            raise ValueError("control_dependencies must be a subset of participants.")

        if scope is GameConstraintScope.PLAYER_LOCAL:
            owner_ = _identifier(owner, "owner") if owner is not None else None
            if owner_ is None or participants_ != (owner_,):
                raise ValueError(
                    "PLAYER_LOCAL blocks require participants to contain only owner."
                )
        elif scope is GameConstraintScope.PLAYER_OWNED_COUPLED:
            owner_ = _identifier(owner, "owner") if owner is not None else None
            if owner_ is None or owner_ not in participants_ or len(participants_) < 2:
                raise ValueError(
                    "PLAYER_OWNED_COUPLED blocks require an owner and at least "
                    "two participants including that owner."
                )
        else:
            if owner is not None:
                raise ValueError("SHARED blocks must not declare an owner.")
            owner_ = None

        shape = _residual_shape(residual_shape)
        lower = _standard_bound(constraint.lower, shape, "constraint lower bound")
        upper = _standard_bound(constraint.upper, shape, "constraint upper bound")
        _standard_bound(constraint.scale, shape, "constraint scale")
        if equality:
            standard = jnp.all(jnp.isfinite(lower)) & jnp.all(lower == 0.0)
            standard = standard & jnp.all(upper == 0.0)
            if not bool(standard):
                raise ValueError(
                    "Equality blocks require zero lower and upper residual bounds."
                )
        else:
            standard = jnp.all(jnp.isneginf(lower)) & jnp.all(upper == 0.0)
            if not bool(standard):
                raise ValueError(
                    "Inequality blocks require residual <= 0 bounds "
                    "(lower=-inf, upper=0)."
                )

        self.constraint = constraint
        self.scope = scope
        self.participants = participants_
        self.owner = owner_
        self.site = site
        self.equality = equality
        self.residual_shape = shape
        self.time_dependent = time_dependent
        self.state_dependent = state_dependent
        self.control_dependencies = dependencies
        self.constraint_id = _identifier(constraint.constraint_id, "constraint_id")

    @property
    def residual_size(self) -> int:
        return prod(self.residual_shape)


class OpenLoopGameConstraints(StrictModule):
    """An ordered, ownership-checked collection of open-loop game constraints."""

    partition: PlayerControlPartition
    blocks: tuple[GameConstraintBlock, ...]
    constraints_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: PlayerControlPartition,
        blocks: Sequence[GameConstraintBlock] = (),
        /,
    ):
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        if isinstance(blocks, (str, bytes)):
            raise TypeError("blocks must be a sequence of GameConstraintBlock values.")
        blocks_ = tuple(blocks)
        if any(not isinstance(block, GameConstraintBlock) for block in blocks_):
            raise TypeError("blocks must contain only GameConstraintBlock values.")
        block_ids = tuple(block.constraint_id for block in blocks_)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Game constraint identifiers must be unique.")

        positions = {player_id: i for i, player_id in enumerate(partition.player_ids)}
        for block in blocks_:
            if any(player_id not in positions for player_id in block.participants):
                raise ValueError(
                    f"Constraint {block.constraint_id!r} contains an unknown participant."
                )
            participant_positions = tuple(
                positions[player_id] for player_id in block.participants
            )
            if participant_positions != tuple(sorted(participant_positions)):
                raise ValueError(
                    f"Constraint {block.constraint_id!r} participants must follow "
                    "partition order."
                )
            dependency_positions = tuple(
                positions[player_id] for player_id in block.control_dependencies
            )
            if dependency_positions != tuple(sorted(dependency_positions)):
                raise ValueError(
                    f"Constraint {block.constraint_id!r} control_dependencies must "
                    "follow partition order."
                )

        payload = {
            "partition_id": partition.partition_id,
            "blocks": [
                {
                    "constraint_id": block.constraint_id,
                    "scope": int(block.scope),
                    "participants": block.participants,
                    "owner": block.owner,
                    "site": int(block.site),
                    "equality": block.equality,
                    "residual_shape": block.residual_shape,
                    "time_dependent": block.time_dependent,
                    "state_dependent": block.state_dependent,
                    "control_dependencies": block.control_dependencies,
                }
                for block in blocks_
            ],
        }
        self.partition = partition
        self.blocks = blocks_
        self.constraints_id = (
            f"open-loop-game-constraints:{canonical_fingerprint(payload)}"
        )

    def layout(self, /, *, num_path_sites: int) -> GameConstraintLayout:
        """Build the deterministic physical-residual layout for a horizon."""
        return GameConstraintLayout(self, num_path_sites=num_path_sites)


class GameConstraintLayout(StrictModule):
    """Deterministic layout of physical residuals, with no multiplier copies."""

    constraints: OpenLoopGameConstraints
    feasibility_incidence: Array
    block_ids: tuple[str, ...] = eqx.field(static=True)
    block_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    block_output_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    block_sizes: tuple[int, ...] = eqx.field(static=True)
    scopes: tuple[GameConstraintScope, ...] = eqx.field(static=True)
    sites: tuple[GameConstraintSite, ...] = eqx.field(static=True)
    equalities: tuple[bool, ...] = eqx.field(static=True)
    num_path_sites: int = eqx.field(static=True)
    num_residuals: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraints: OpenLoopGameConstraints,
        /,
        *,
        num_path_sites: int,
    ):
        if not isinstance(constraints, OpenLoopGameConstraints):
            raise TypeError("constraints must be OpenLoopGameConstraints.")
        if isinstance(num_path_sites, bool):
            raise TypeError("num_path_sites must be a positive integer.")
        sites_ = index(num_path_sites)
        if sites_ <= 0:
            raise ValueError("num_path_sites must be positive.")

        cursor = 0
        slices: list[tuple[int, int]] = []
        output_shapes: list[tuple[int, ...]] = []
        sizes: list[int] = []
        incidence_rows: list[tuple[bool, ...]] = []
        players = constraints.partition.player_ids
        for block in constraints.blocks:
            output_shape = (
                (sites_,) + block.residual_shape
                if block.site is GameConstraintSite.PATH
                else block.residual_shape
            )
            size = prod(output_shape)
            slices.append((cursor, cursor + size))
            output_shapes.append(output_shape)
            sizes.append(size)
            cursor += size
            relevant = (
                block.participants
                if block.scope is GameConstraintScope.SHARED
                else (block.owner,)
            )
            incidence_rows.append(tuple(player_id in relevant for player_id in players))

        incidence = (
            jnp.asarray(incidence_rows, dtype=bool)
            if incidence_rows
            else jnp.zeros((0, constraints.partition.num_players), dtype=bool)
        )
        payload = {
            "constraints_id": constraints.constraints_id,
            "num_path_sites": sites_,
            "block_slices": slices,
        }
        self.constraints = constraints
        self.feasibility_incidence = incidence
        self.block_ids = tuple(block.constraint_id for block in constraints.blocks)
        self.block_slices = tuple(slices)
        self.block_output_shapes = tuple(output_shapes)
        self.block_sizes = tuple(sizes)
        self.scopes = tuple(block.scope for block in constraints.blocks)
        self.sites = tuple(block.site for block in constraints.blocks)
        self.equalities = tuple(block.equality for block in constraints.blocks)
        self.num_path_sites = sites_
        self.num_residuals = cursor
        self.layout_id = f"game-constraint-layout:{canonical_fingerprint(payload)}"

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def multiplier_layout(self, /, *, variational: bool) -> GameMultiplierLayout:
        """Allocate GNE copies or explicitly requested variational multipliers."""
        return GameMultiplierLayout(self, variational=variational)


class GameMultiplierLayout(StrictModule):
    """Player-private and optional common variational multiplier allocation."""

    constraint_layout: GameConstraintLayout
    player_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    player_block_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    player_multiplier_slices: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(
        static=True
    )
    player_residual_slices: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(
        static=True
    )
    shared_slice: tuple[int, int] = eqx.field(static=True)
    shared_block_indices: tuple[int, ...] = eqx.field(static=True)
    shared_multiplier_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    shared_residual_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    num_multipliers: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraint_layout: GameConstraintLayout,
        /,
        *,
        variational: bool,
    ):
        if not isinstance(constraint_layout, GameConstraintLayout):
            raise TypeError("constraint_layout must be a GameConstraintLayout.")
        if not isinstance(variational, bool):
            raise TypeError("variational must be a bool.")

        cursor = 0
        player_slices: list[tuple[int, int]] = []
        player_blocks: list[tuple[int, ...]] = []
        player_multiplier_slices: list[tuple[tuple[int, int], ...]] = []
        player_residual_slices: list[tuple[tuple[int, int], ...]] = []
        blocks = constraint_layout.constraints.blocks
        for player_id in constraint_layout.constraints.partition.player_ids:
            start = cursor
            indices: list[int] = []
            destinations: list[tuple[int, int]] = []
            sources: list[tuple[int, int]] = []
            for block_index, block in enumerate(blocks):
                owned = (
                    block.scope is not GameConstraintScope.SHARED
                    and block.owner == player_id
                )
                shared_copy = (
                    block.scope is GameConstraintScope.SHARED
                    and not variational
                    and player_id in block.participants
                )
                if owned or shared_copy:
                    size = constraint_layout.block_sizes[block_index]
                    indices.append(block_index)
                    destinations.append((cursor, cursor + size))
                    sources.append(constraint_layout.block_slices[block_index])
                    cursor += size
            player_slices.append((start, cursor))
            player_blocks.append(tuple(indices))
            player_multiplier_slices.append(tuple(destinations))
            player_residual_slices.append(tuple(sources))

        shared_start = cursor
        shared_blocks: list[int] = []
        shared_destinations: list[tuple[int, int]] = []
        shared_sources: list[tuple[int, int]] = []
        if variational:
            for block_index, block in enumerate(blocks):
                if block.scope is GameConstraintScope.SHARED:
                    size = constraint_layout.block_sizes[block_index]
                    shared_blocks.append(block_index)
                    shared_destinations.append((cursor, cursor + size))
                    shared_sources.append(constraint_layout.block_slices[block_index])
                    cursor += size

        payload = {
            "constraint_layout_id": constraint_layout.layout_id,
            "variational": variational,
            "player_block_indices": player_blocks,
            "shared_block_indices": shared_blocks,
        }
        self.constraint_layout = constraint_layout
        self.player_slices = tuple(player_slices)
        self.player_block_indices = tuple(player_blocks)
        self.player_multiplier_slices = tuple(player_multiplier_slices)
        self.player_residual_slices = tuple(player_residual_slices)
        self.shared_slice = (shared_start, cursor)
        self.shared_block_indices = tuple(shared_blocks)
        self.shared_multiplier_slices = tuple(shared_destinations)
        self.shared_residual_slices = tuple(shared_sources)
        self.variational = variational
        self.num_multipliers = cursor
        self.layout_id = f"game-multiplier-layout:{canonical_fingerprint(payload)}"


class GameFeasibilityEvidence(StrictModule):
    """Case-local evidence for declared constraints at supplied trajectory sites."""

    layout: GameConstraintLayout
    raw_residuals: tuple[Array, ...]
    violations: tuple[Array, ...]
    block_maximum_violation: Array
    block_finite: Array
    block_feasible: Array
    player_valid: Array
    player_feasible: Array
    maximum_violation: Array
    finite: Array
    feasible: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    feasibility_scope: str = eqx.field(static=True)
    sampled_only: bool = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        layout: GameConstraintLayout,
        raw_residuals: Sequence[ArrayLike],
        violations: Sequence[ArrayLike],
        block_maximum_violation: ArrayLike,
        block_finite: ArrayLike,
        block_feasible: ArrayLike,
        player_valid: ArrayLike,
        player_feasible: ArrayLike,
        maximum_violation: ArrayLike,
        finite: ArrayLike,
        feasible: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        case_shape: Sequence[int],
        tolerance: float,
    ):
        if not isinstance(layout, GameConstraintLayout):
            raise TypeError("layout must be a GameConstraintLayout.")
        cases = tuple(index(size) for size in case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("case_shape dimensions must be positive.")
        raw = tuple(jnp.asarray(value) for value in raw_residuals)
        violation = tuple(jnp.asarray(value) for value in violations)
        if len(raw) != layout.num_blocks or len(violation) != layout.num_blocks:
            raise ValueError(
                "raw_residuals and violations must provide one array per block."
            )
        for block_id, output_shape, raw_value, violation_value in zip(
            layout.block_ids,
            layout.block_output_shapes,
            raw,
            violation,
            strict=True,
        ):
            expected = cases + output_shape
            if raw_value.shape != expected or violation_value.shape != expected:
                raise ValueError(
                    f"Constraint {block_id!r} evidence must have shape {expected}."
                )

        block_shape = cases + (layout.num_blocks,)
        player_shape = cases + (layout.constraints.partition.num_players,)
        case_arrays = (
            jnp.asarray(maximum_violation),
            jnp.asarray(finite, dtype=bool),
            jnp.asarray(feasible, dtype=bool),
            jnp.asarray(valid, dtype=bool),
            jnp.asarray(status),
        )
        block_arrays = (
            jnp.asarray(block_maximum_violation),
            jnp.asarray(block_finite, dtype=bool),
            jnp.asarray(block_feasible, dtype=bool),
        )
        player_arrays = (
            jnp.asarray(player_valid, dtype=bool),
            jnp.asarray(player_feasible, dtype=bool),
        )
        if any(value.shape != cases for value in case_arrays):
            raise ValueError("Case-level feasibility evidence must have case_shape.")
        if any(value.shape != block_shape for value in block_arrays):
            raise ValueError(
                "Block-level feasibility evidence must have case_shape + (num_blocks,)."
            )
        if any(value.shape != player_shape for value in player_arrays):
            raise ValueError(
                "Player-level feasibility evidence must have case_shape + (num_players,)."
            )

        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("feasibility tolerance must be finite and nonnegative.")
        self.layout = layout
        self.raw_residuals = raw
        self.violations = violation
        self.block_maximum_violation = block_arrays[0]
        self.block_finite = block_arrays[1]
        self.block_feasible = block_arrays[2]
        self.player_valid = player_arrays[0]
        self.player_feasible = player_arrays[1]
        self.maximum_violation = case_arrays[0]
        self.finite = case_arrays[1]
        self.feasible = case_arrays[2]
        self.valid = case_arrays[3]
        self.status = case_arrays[4]
        self.case_shape = cases
        self.tolerance = tolerance_
        self.feasibility_scope = "declared-open-loop-blocks-at-supplied-trajectory-sites"
        self.sampled_only = True
        self.certified = False
        self.method_id = "game-constraint:sampled-open-loop-evaluation:v1"

    @property
    def equality_violations(self) -> tuple[Array, ...]:
        """Absolute residuals for equality blocks, in declared block order."""
        return tuple(
            violation
            for violation, equality in zip(
                self.violations,
                self.layout.equalities,
                strict=True,
            )
            if equality
        )

    @property
    def positive_inequality_violations(self) -> tuple[Array, ...]:
        """Positive residual parts for inequality blocks, in declared block order."""
        return tuple(
            violation
            for violation, equality in zip(
                self.violations,
                self.layout.equalities,
                strict=True,
            )
            if not equality
        )


def _dependency_finite(
    block: GameConstraintBlock,
    constraints: OpenLoopGameConstraints,
    trajectory: TrajectoryOptimizationView,
    /,
) -> Array:
    cases = trajectory.case_shape
    finite = jnp.ones(cases, dtype=bool)
    if block.time_dependent:
        finite = finite & jnp.all(jnp.isfinite(trajectory.times))
    if block.state_dependent:
        if block.site is GameConstraintSite.PATH:
            values = trajectory.states[(slice(None),) * len(cases) + (slice(None, -1),)]
        elif block.site is GameConstraintSite.TERMINAL:
            values = trajectory.final_state
        else:
            values = trajectory.states
        axes = tuple(range(len(cases), values.ndim))
        finite = finite & jnp.all(jnp.isfinite(values), axis=axes)
    if block.control_dependencies:
        controls = tuple(
            trajectory.controls[..., start:stop]
            for player_id, (start, stop) in zip(
                constraints.partition.player_ids,
                constraints.partition.control_slices,
                strict=True,
            )
            if player_id in block.control_dependencies
        )
        for values in controls:
            axes = tuple(range(len(cases), values.ndim))
            finite = finite & jnp.all(jnp.isfinite(values), axis=axes)
    return finite


def _evaluate_path_block(
    block: GameConstraintBlock,
    trajectory: TrajectoryOptimizationView,
    args: Any,
    /,
) -> Array:
    count = prod(trajectory.case_shape) if trajectory.case_shape else 1
    states = trajectory.states.reshape(
        (count, trajectory.num_nodes) + trajectory.state_shape
    )
    controls = trajectory.controls.reshape(
        (count, trajectory.num_intervals) + trajectory.control_shape
    )

    def evaluate_case(case_states: Array, case_controls: Array) -> Array:
        return jax.vmap(
            lambda time, state, control: block.constraint(
                time,
                state,
                control,
                args,
            )
        )(trajectory.times[:-1], case_states[:-1], case_controls)

    values = jax.vmap(evaluate_case)(states, controls)
    expected_flat = (count, trajectory.num_intervals) + block.residual_shape
    if values.shape != expected_flat:
        raise ValueError(
            f"Constraint {block.constraint_id!r} callback must return residual_shape "
            f"{block.residual_shape} at each path site; got {values.shape[2:]}."
        )
    return values.reshape(
        trajectory.case_shape + (trajectory.num_intervals,) + block.residual_shape
    )


def _evaluate_trajectory_block(
    block: GameConstraintBlock,
    trajectory: TrajectoryOptimizationView,
    args: Any,
    /,
) -> Array:
    values = jnp.asarray(block.constraint(trajectory, args))
    expected = trajectory.case_shape + block.residual_shape
    if values.shape != expected:
        raise ValueError(
            f"Constraint {block.constraint_id!r} callback must return shape "
            f"{expected}; got {values.shape}."
        )
    return values


def _payload_all(value: Array, case_rank: int, /) -> Array:
    axes = tuple(range(case_rank, value.ndim))
    return jnp.all(value, axis=axes) if axes else value


def _payload_max(value: Array, case_rank: int, /) -> Array:
    axes = tuple(range(case_rank, value.ndim))
    return jnp.max(value, axis=axes) if axes else value


def evaluate_game_feasibility(
    constraints: OpenLoopGameConstraints,
    trajectory: TrajectoryOptimizationView,
    args: Any = None,
    /,
    *,
    tolerance: float = 0.0,
) -> GameFeasibilityEvidence:
    """Evaluate declared residuals without solving or making a safety claim."""
    if not isinstance(constraints, OpenLoopGameConstraints):
        raise TypeError("constraints must be OpenLoopGameConstraints.")
    if not isinstance(trajectory, TrajectoryOptimizationView):
        raise TypeError("trajectory must be a TrajectoryOptimizationView.")
    expected_control_shape = (constraints.partition.joint_control_size,)
    if trajectory.control_shape != expected_control_shape:
        raise ValueError(
            "trajectory control_shape must match the flattened joint-control "
            f"partition {expected_control_shape}; got {trajectory.control_shape}."
        )
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("feasibility tolerance must be finite and nonnegative.")

    layout = constraints.layout(num_path_sites=trajectory.num_intervals)
    raw_residuals: list[Array] = []
    violations: list[Array] = []
    block_maximum: list[Array] = []
    block_finite: list[Array] = []
    block_feasible: list[Array] = []
    case_rank = len(trajectory.case_shape)
    for block in constraints.blocks:
        raw = (
            _evaluate_path_block(block, trajectory, args)
            if block.site is GameConstraintSite.PATH
            else _evaluate_trajectory_block(block, trajectory, args)
        )
        element_finite = jnp.isfinite(raw)
        safe_raw = jnp.where(element_finite, raw, 0.0)
        violation = jnp.abs(safe_raw) if block.equality else jnp.maximum(safe_raw, 0.0)
        violation = jnp.where(element_finite, violation, jnp.inf)
        finite = _payload_all(element_finite, case_rank)
        finite = finite & _dependency_finite(block, constraints, trajectory)
        maximum = _payload_max(violation, case_rank)
        maximum = jnp.where(finite, maximum, jnp.inf)
        feasible = finite & (maximum <= tolerance_)
        raw_residuals.append(raw)
        violations.append(violation)
        block_maximum.append(maximum)
        block_finite.append(finite)
        block_feasible.append(feasible)

    cases = trajectory.case_shape
    num_blocks = len(constraints.blocks)
    if num_blocks:
        block_maximum_array = jnp.stack(block_maximum, axis=-1)
        block_finite_array = jnp.stack(block_finite, axis=-1)
        block_feasible_array = jnp.stack(block_feasible, axis=-1)
        finite = jnp.all(block_finite_array, axis=-1)
        feasible = jnp.all(block_feasible_array, axis=-1)
        maximum = jnp.max(block_maximum_array, axis=-1)
    else:
        dtype = trajectory.states.real.dtype
        block_maximum_array = jnp.zeros(cases + (0,), dtype=dtype)
        block_finite_array = jnp.ones(cases + (0,), dtype=bool)
        block_feasible_array = jnp.ones(cases + (0,), dtype=bool)
        finite = jnp.ones(cases, dtype=bool)
        feasible = jnp.ones(cases, dtype=bool)
        maximum = jnp.zeros(cases, dtype=dtype)

    incidence = layout.feasibility_incidence.astype(jnp.int32)
    invalid_counts = ein.contract(
        "...b,bp->...p",
        (~block_finite_array).astype(jnp.int32),
        incidence,
    )
    failed_counts = ein.contract(
        "...b,bp->...p",
        (~block_feasible_array).astype(jnp.int32),
        incidence,
    )
    player_valid = invalid_counts == 0
    player_feasible = failed_counts == 0
    valid = finite
    status = jnp.where(
        ~finite,
        int(GameFeasibilityStatus.NONFINITE_RESIDUAL),
        jnp.where(
            feasible,
            int(GameFeasibilityStatus.FEASIBLE),
            int(GameFeasibilityStatus.INFEASIBLE),
        ),
    ).astype(jnp.int32)
    return GameFeasibilityEvidence(
        layout=layout,
        raw_residuals=raw_residuals,
        violations=violations,
        block_maximum_violation=block_maximum_array,
        block_finite=block_finite_array,
        block_feasible=block_feasible_array,
        player_valid=player_valid,
        player_feasible=player_feasible,
        maximum_violation=maximum,
        finite=finite,
        feasible=feasible,
        valid=valid,
        status=status,
        case_shape=cases,
        tolerance=tolerance_,
    )


__all__ = [
    "GameConstraintBlock",
    "GameConstraintLayout",
    "GameConstraintScope",
    "GameConstraintSite",
    "GameFeasibilityEvidence",
    "GameFeasibilityStatus",
    "GameMultiplierLayout",
    "OpenLoopGameConstraints",
    "evaluate_game_feasibility",
]
