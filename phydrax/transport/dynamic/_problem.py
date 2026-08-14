#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, prod
from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import TimeGrid
from ...integration import IntegrationRealization
from ...integration._targets import (
    DensityTarget,
    DiscreteMeasureTarget,
    WeightedSampleTarget,
)
from ...stochastic._state_space import AbstractTransitionKernel, StateSpaceStepContext


FiniteBridgeTarget = (
    DiscreteMeasureTarget | WeightedSampleTarget | DensityTarget | IntegrationRealization
)


class _FiniteEndpoint(StrictModule):
    support: Array
    probabilities: Array
    mass: Array
    mask: Array
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    provenance: str = eqx.field(static=True)


class BridgeProblemProvenance(StrictModule):
    """Static endpoint and reference-process identity for a bridge problem."""

    initial: str = eqx.field(static=True)
    terminal: str = eqx.field(static=True)
    reference_process: str = eqx.field(static=True)
    time_grid: str = eqx.field(static=True)


def _product_weights(weights: Mapping[str, cx.Field], axes: tuple[str, ...]) -> cx.Field:
    values = weights
    total = cx.Field(jnp.asarray(1.0), dims=())
    for axis in axes:
        total = total * values[axis]
    return total


def _canonical_support(
    value: Any,
    /,
    *,
    atom_axes: tuple[str, ...] | tuple[int, ...],
    atom_shape: tuple[int, ...],
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    weight_rank: int,
    weight_order: tuple[int, ...],
    weight_shape: tuple[int, ...],
    name: str,
) -> tuple[Array, tuple[int, ...]]:
    leaf = value
    if isinstance(leaf, cx.Field):
        if not all(isinstance(axis, str) for axis in atom_axes):
            raise TypeError(f"{name} cannot pair a named field with integer atom axes.")
        named_atoms = cast(tuple[str, ...], atom_axes)
        missing = tuple(axis for axis in named_atoms if axis not in leaf.named_dims)
        if missing:
            raise ValueError(f"{name} is missing state axes {missing!r}.")
        present_cases = tuple(axis for axis in case_axes if axis in leaf.named_dims)
        if present_cases not in ((), case_axes):
            raise ValueError(f"{name} must contain either all case axes or none of them.")
        leading = present_cases + named_atoms
        positions = tuple(leaf.dims.index(axis) for axis in leading)
        remaining = tuple(index for index in range(leaf.ndim) if index not in positions)
        canonical = jnp.transpose(jnp.asarray(leaf.data), positions + remaining)
        observed = tuple(int(size) for size in canonical.shape[: len(leading)])
        expected = (case_shape if present_cases else ()) + atom_shape
        if observed != expected:
            raise ValueError(f"{name} leading shape must be {expected}; got {observed}.")
        event_shape = tuple(int(size) for size in canonical.shape[len(leading) :])
        support = canonical.reshape(
            expected[: len(present_cases)] + (prod(atom_shape),) + event_shape
        )
        if not present_cases:
            support = jnp.broadcast_to(support, case_shape + support.shape)
        return support, event_shape

    data = jnp.asarray(leaf)
    if (
        data.ndim >= weight_rank
        and tuple(data.shape[:weight_rank]) == weight_shape
        and weight_order != tuple(range(weight_rank))
    ):
        outputs = tuple(range(weight_rank, data.ndim))
        canonical = jnp.transpose(data, weight_order + outputs)
        event_shape = tuple(
            int(size) for size in canonical.shape[len(case_shape) + len(atom_shape) :]
        )
        return (
            canonical.reshape(case_shape + (prod(atom_shape),) + event_shape),
            event_shape,
        )
    shared_prefix = atom_shape
    case_prefix = case_shape + atom_shape
    if tuple(data.shape[: len(case_prefix)]) == case_prefix:
        event_shape = tuple(int(size) for size in data.shape[len(case_prefix) :])
        return data.reshape(case_shape + (prod(atom_shape),) + event_shape), event_shape
    if tuple(data.shape[: len(shared_prefix)]) == shared_prefix:
        event_shape = tuple(int(size) for size in data.shape[len(shared_prefix) :])
        shared = data.reshape((prod(atom_shape),) + event_shape)
        return jnp.broadcast_to(shared, case_shape + shared.shape), event_shape
    if data.ndim >= weight_rank and tuple(data.shape[:weight_rank]) == weight_shape:
        outputs = tuple(range(weight_rank, data.ndim))
        canonical = jnp.transpose(data, weight_order + outputs)
        event_shape = tuple(
            int(size) for size in canonical.shape[len(case_shape) + len(atom_shape) :]
        )
        return (
            canonical.reshape(case_shape + (prod(atom_shape),) + event_shape),
            event_shape,
        )
    raise ValueError(
        f"{name} must begin with state shape {atom_shape} or case/state shape "
        f"{case_shape + atom_shape}."
    )


def _finish_endpoint(
    support: Array,
    values: Array,
    included: Array,
    /,
    *,
    logarithmic: bool,
    normalized: bool,
    target_mass: Array | None,
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
    provenance: str,
    name: str,
) -> _FiniteEndpoint:
    included = jnp.asarray(included, dtype=bool)
    values = jnp.asarray(values, dtype=float)
    if logarithmic:
        admissible = jnp.isfinite(values) | jnp.isneginf(values)
        values = eqx.error_if(
            values,
            jnp.any(included & ~admissible),
            f"{name} active log weights must be finite or negative infinity.",
        )
        safe = jnp.where(included, values, -jnp.inf)
        maximum = jnp.max(safe, axis=-1, keepdims=True)
        shifted = jnp.where(jnp.isfinite(maximum), safe - maximum, -jnp.inf)
        positive = jnp.where(included, jnp.exp(shifted), 0.0)
        scaled_mass = jnp.sum(positive, axis=-1)
        log_raw_mass = jnp.squeeze(maximum, axis=-1) + jnp.log(scaled_mass)
        raw_mass = jnp.exp(log_raw_mass)
    else:
        values = eqx.error_if(
            values,
            jnp.any(included & (~jnp.isfinite(values) | (values < 0.0))),
            f"{name} active weights must be finite and nonnegative.",
        )
        positive = jnp.where(included, values, 0.0)
        raw_mass = jnp.sum(positive, axis=-1)
    raw_mass = eqx.error_if(
        raw_mass,
        jnp.any(~jnp.isfinite(raw_mass) | (raw_mass <= 0.0)),
        f"{name} must contain positive finite mass in every case.",
    )
    probabilities = positive / jnp.sum(positive, axis=-1, keepdims=True)
    if target_mass is not None:
        mass = jnp.broadcast_to(jnp.asarray(target_mass, dtype=float), case_shape)
    elif normalized:
        mass = jnp.ones(case_shape, dtype=probabilities.dtype)
    else:
        mass = raw_mass
    mass = eqx.error_if(
        mass,
        jnp.any(~jnp.isfinite(mass) | (mass <= 0.0)),
        f"{name} target mass must be finite and positive in every case.",
    )
    event_axes = tuple(range(support.ndim - len(event_shape), support.ndim))
    support_finite = jnp.isfinite(support)
    if event_axes:
        support_finite = jnp.all(support_finite, axis=event_axes)
    support = eqx.error_if(
        support,
        jnp.any(~support_finite),
        f"{name} finite state support must contain only finite values.",
    )
    return _FiniteEndpoint(
        support=support,
        probabilities=probabilities,
        mass=mass,
        mask=included,
        case_axes=case_axes,
        case_shape=case_shape,
        event_shape=event_shape,
        normalized=normalized,
        provenance=provenance,
    )


def _lower_discrete(target: DiscreteMeasureTarget, name: str, /) -> _FiniteEndpoint:
    weights = (
        target.weights
        if isinstance(target.weights, cx.Field)
        else _product_weights(target.weights, target.axes)
    )
    atom_axes = target.axes
    case_axes = tuple(cast(str, dim) for dim in weights.dims if dim not in atom_axes)
    case_positions = tuple(weights.dims.index(axis) for axis in case_axes)
    atom_positions = tuple(weights.dims.index(axis) for axis in atom_axes)
    order = case_positions + atom_positions
    canonical = jnp.transpose(jnp.asarray(weights.data, dtype=float), order)
    case_shape = tuple(int(canonical.shape[index]) for index in range(len(case_axes)))
    atom_shape = tuple(
        int(canonical.shape[len(case_axes) + index]) for index in range(len(atom_axes))
    )
    values = canonical.reshape(case_shape + (prod(atom_shape),))
    if target.mask is None:
        included = jnp.ones_like(values, dtype=bool)
    else:
        mask = target.mask.broadcast_like(weights)
        included = jnp.transpose(jnp.asarray(mask.data, dtype=bool), order).reshape(
            values.shape
        )
    support, event_shape = _canonical_support(
        target.points,
        atom_axes=atom_axes,
        atom_shape=atom_shape,
        case_axes=case_axes,
        case_shape=case_shape,
        weight_rank=weights.ndim,
        weight_order=order,
        weight_shape=tuple(int(size) for size in weights.shape),
        name=f"{name} points",
    )
    return _finish_endpoint(
        support,
        values,
        included,
        logarithmic=False,
        normalized=target.normalized,
        target_mass=target.target_mass,
        case_axes=case_axes,
        case_shape=case_shape,
        event_shape=event_shape,
        provenance=target.provenance,
        name=name,
    )


def _lower_weighted(target: WeightedSampleTarget, name: str, /) -> _FiniteEndpoint:
    if isinstance(target.log_weights, cx.Field):
        weights = target.log_weights
        atom_axes = cast(tuple[str, ...], target.sample_axes)
        case_axes = tuple(cast(str, dim) for dim in weights.dims if dim not in atom_axes)
        case_positions = tuple(weights.dims.index(axis) for axis in case_axes)
        atom_positions = tuple(weights.dims.index(axis) for axis in atom_axes)
        order = case_positions + atom_positions
        canonical = jnp.transpose(jnp.asarray(weights.data, dtype=float), order)
        case_shape = tuple(int(canonical.shape[index]) for index in range(len(case_axes)))
        atom_shape = tuple(
            int(canonical.shape[len(case_axes) + index])
            for index in range(len(atom_axes))
        )
        values = canonical.reshape(case_shape + (prod(atom_shape),))
        if target.mask is None:
            included = jnp.ones_like(values, dtype=bool)
        else:
            mask = cast(cx.Field, target.mask).broadcast_like(weights)
            included = jnp.transpose(jnp.asarray(mask.data, dtype=bool), order).reshape(
                values.shape
            )
        weight_rank = weights.ndim
    else:
        raw = jnp.asarray(target.log_weights, dtype=float)
        atom_positions = cast(tuple[int, ...], target.sample_axes)
        case_positions = tuple(
            index for index in range(raw.ndim) if index not in atom_positions
        )
        order = case_positions + atom_positions
        canonical = jnp.transpose(raw, order)
        case_shape = tuple(int(raw.shape[index]) for index in case_positions)
        atom_shape = tuple(int(raw.shape[index]) for index in atom_positions)
        case_axes = tuple(f"case_{index}" for index in range(len(case_shape)))
        atom_axes = atom_positions
        values = canonical.reshape(case_shape + (prod(atom_shape),))
        included_raw = (
            jnp.ones_like(raw, dtype=bool)
            if target.mask is None
            else jnp.broadcast_to(jnp.asarray(target.mask, dtype=bool), raw.shape)
        )
        included = jnp.transpose(included_raw, order).reshape(values.shape)
        weight_rank = raw.ndim
    if target.support_valid is not None:
        support_valid = jnp.asarray(target.support_valid, dtype=bool)
        if support_valid.size != prod(atom_shape):
            raise ValueError(
                f"{name} support_valid must contain one value per finite state."
            )
        included = included & support_valid.reshape(
            (1,) * len(case_shape) + (prod(atom_shape),)
        )
    support, event_shape = _canonical_support(
        target.samples,
        atom_axes=atom_axes,
        atom_shape=atom_shape,
        case_axes=case_axes,
        case_shape=case_shape,
        weight_rank=weight_rank,
        weight_order=order,
        weight_shape=(
            tuple(int(size) for size in weights.shape)
            if isinstance(target.log_weights, cx.Field)
            else tuple(int(size) for size in raw.shape)
        ),
        name=f"{name} samples",
    )
    return _finish_endpoint(
        support,
        values,
        included,
        logarithmic=True,
        normalized=target.normalized,
        target_mass=target.target_mass,
        case_axes=case_axes,
        case_shape=case_shape,
        event_shape=event_shape,
        provenance=target.provenance,
        name=name,
    )


def _lower_endpoint(target: FiniteBridgeTarget, name: str, /) -> _FiniteEndpoint:
    if isinstance(target, IntegrationRealization):
        return _lower_endpoint(target.target, name)
    if isinstance(target, DiscreteMeasureTarget):
        return _lower_discrete(target, name)
    if isinstance(target, WeightedSampleTarget):
        return _lower_weighted(target, name)
    if isinstance(target, DensityTarget):
        base = _lower_endpoint(target.base, name)
        log_density = (
            target.log_density(base.support)
            if callable(target.log_density)
            else target.log_density
        )
        density = jnp.broadcast_to(
            jnp.asarray(log_density, dtype=float), base.probabilities.shape
        )
        base_log = jnp.where(
            base.probabilities > 0.0, jnp.log(base.probabilities), -jnp.inf
        )
        combined = base_log + density
        finite = jnp.isfinite(combined)
        maximum = jnp.max(jnp.where(finite, combined, -jnp.inf), axis=-1, keepdims=True)
        unnormalized = jnp.where(finite, jnp.exp(combined - maximum), 0.0)
        partition = jnp.sum(unnormalized, axis=-1, keepdims=True)
        probabilities = unnormalized / partition
        log_partition = jnp.squeeze(maximum, -1) + jnp.log(jnp.squeeze(partition, -1))
        mass = base.mass if target.normalized else base.mass * jnp.exp(log_partition)
        probabilities = eqx.error_if(
            probabilities,
            jnp.any(~jnp.isfinite(probabilities)),
            f"{name} density must leave positive finite endpoint mass.",
        )
        return _FiniteEndpoint(
            support=base.support,
            probabilities=probabilities,
            mass=mass,
            mask=base.mask & finite,
            case_axes=base.case_axes,
            case_shape=base.case_shape,
            event_shape=base.event_shape,
            normalized=target.normalized,
            provenance=f"density:{base.provenance}",
        )
    raise TypeError(
        f"{name} must be a DiscreteMeasureTarget, WeightedSampleTarget, "
        "finite DensityTarget, or IntegrationRealization."
    )


class SchrodingerBridgeProblem(StrictModule):
    """Exact finite-state Schrödinger bridge with explicit endpoint measures."""

    initial: _FiniteEndpoint
    terminal: _FiniteEndpoint
    times: Array
    reference: AbstractTransitionKernel
    context: StateSpaceStepContext
    mass: Array
    provenance: BridgeProblemProvenance
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_states: int = eqx.field(static=True)
    transition_tolerance: float = eqx.field(static=True)
    mass_tolerance: float = eqx.field(static=True)
    time_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial: FiniteBridgeTarget,
        terminal: FiniteBridgeTarget,
        times: ArrayLike | TimeGrid,
        reference: AbstractTransitionKernel,
        context: StateSpaceStepContext,
        /,
        *,
        transition_tolerance: float = 1e-7,
        mass_tolerance: float = 1e-8,
    ):
        if not isinstance(reference, AbstractTransitionKernel):
            raise TypeError("reference must implement AbstractTransitionKernel.")
        if not reference.has_log_density:
            raise ValueError(
                "Exact finite-state Schrödinger bridges require a reference transition "
                "with a normalized log density; sampler-only kernels are unsupported."
            )
        if not isinstance(context, StateSpaceStepContext):
            raise TypeError("context must be an explicit StateSpaceStepContext.")
        transition_tolerance = float(transition_tolerance)
        mass_tolerance = float(mass_tolerance)
        if (
            not isfinite(transition_tolerance)
            or transition_tolerance < 0.0
            or not isfinite(mass_tolerance)
            or mass_tolerance < 0.0
        ):
            raise ValueError("Bridge tolerances must be finite and nonnegative.")
        if isinstance(times, TimeGrid):
            grid = times.times
            time_id = times.time_id
        else:
            grid = jnp.asarray(times, dtype=float)
            time_id = "schrodinger-bridge-grid"
        if grid.ndim != 1 or grid.shape[0] < 2:
            raise ValueError(
                "times must be a one-dimensional grid with at least two entries."
            )
        if bool(jnp.any(~jnp.isfinite(grid))) or bool(jnp.any(jnp.diff(grid) <= 0.0)):
            raise ValueError("times must be finite and strictly increasing.")
        initial_endpoint = _lower_endpoint(initial, "initial")
        terminal_endpoint = _lower_endpoint(terminal, "terminal")
        if initial_endpoint.case_axes != terminal_endpoint.case_axes:
            raise ValueError(
                "Initial and terminal endpoint case axes must agree exactly."
            )
        if initial_endpoint.case_shape != terminal_endpoint.case_shape:
            raise ValueError(
                "Initial and terminal endpoint case shapes must agree exactly."
            )
        if initial_endpoint.event_shape != terminal_endpoint.event_shape:
            raise ValueError(
                "Initial and terminal endpoint event shapes must agree exactly."
            )
        if initial_endpoint.support.shape != terminal_endpoint.support.shape:
            raise ValueError(
                "Initial and terminal finite state support shapes must agree."
            )
        support_difference = jnp.max(
            jnp.abs(initial_endpoint.support - terminal_endpoint.support)
        )
        initial_support = eqx.error_if(
            initial_endpoint.support,
            support_difference > transition_tolerance,
            "Initial and terminal finite state supports must agree in the same order.",
        )
        initial_endpoint = eqx.tree_at(
            lambda endpoint: endpoint.support, initial_endpoint, initial_support
        )
        mass = eqx.error_if(
            initial_endpoint.mass,
            jnp.any(
                ~jnp.isclose(
                    initial_endpoint.mass,
                    terminal_endpoint.mass,
                    rtol=mass_tolerance,
                    atol=mass_tolerance,
                )
            ),
            "A Schrödinger bridge requires equal physical endpoint mass in every case.",
        )
        state_shape = initial_endpoint.event_shape
        if reference.state_shape != state_shape:
            raise ValueError(
                "reference.state_shape must equal the endpoint event shape "
                f"{state_shape}; got {reference.state_shape}."
            )
        self.initial = initial_endpoint
        self.terminal = terminal_endpoint
        self.times = grid
        self.reference = reference
        self.context = context
        self.mass = mass
        self.provenance = BridgeProblemProvenance(
            initial_endpoint.provenance,
            terminal_endpoint.provenance,
            reference.process_id,
            time_id,
        )
        self.case_axes = initial_endpoint.case_axes
        self.case_shape = initial_endpoint.case_shape
        self.state_shape = state_shape
        self.num_states = int(initial_endpoint.probabilities.shape[-1])
        self.transition_tolerance = transition_tolerance
        self.mass_tolerance = mass_tolerance
        self.time_id = time_id

    @property
    def state_support(self) -> Array:
        """Finite state values, with shape ``case_shape + (state,) + state_shape``."""
        return self.initial.support

    @property
    def initial_probabilities(self) -> Array:
        return self.initial.probabilities

    @property
    def terminal_probabilities(self) -> Array:
        return self.terminal.probabilities

    @property
    def initial_weights(self) -> Array:
        return self.mass[..., None] * self.initial.probabilities

    @property
    def terminal_weights(self) -> Array:
        return self.mass[..., None] * self.terminal.probabilities

    @property
    def num_steps(self) -> int:
        return int(self.times.shape[0] - 1)

    @property
    def num_cases(self) -> int:
        return prod(self.case_shape) if self.case_shape else 1


__all__ = [
    "BridgeProblemProvenance",
    "FiniteBridgeTarget",
    "SchrodingerBridgeProblem",
]
