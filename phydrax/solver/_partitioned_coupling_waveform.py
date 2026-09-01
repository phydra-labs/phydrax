#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..dynamics import TimeGrid
from ..linalg import AbstractVectorSpace
from ._partitioned_coupling_types import (
    AbstractCouplingSubsystem,
    CouplingPort,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingWindow,
)


CouplingTemporalInterpolation: TypeAlias = Literal["held", "linear"]


class CouplingWaveform(StrictModule):
    """One immutable field or vector-space signal on a fixed relative time grid."""

    grid: TimeGrid
    values: Any
    space_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)

    def __init__(
        self,
        grid: TimeGrid,
        values: Any,
        space: AbstractVectorSpace,
        /,
    ):
        if not isinstance(grid, TimeGrid):
            raise TypeError("Coupling waveform grid must be a TimeGrid.")
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("Coupling waveform space must be an AbstractVectorSpace.")
        leaves, treedef = jax.tree.flatten(values)
        structure_leaves, structure_def = jax.tree.flatten(space.structure())
        if treedef != structure_def or len(leaves) != len(structure_leaves):
            raise ValueError("Coupling waveform values must match their vector space.")
        arrays = tuple(jnp.asarray(value) for value in leaves)
        for value, spec in zip(arrays, structure_leaves, strict=True):
            expected = (grid.num_points, *spec.shape)
            if value.shape != expected:
                raise ValueError(
                    f"Coupling waveform leaf must have shape {expected}; got {value.shape}."
                )
            if value.dtype != spec.dtype:
                raise TypeError(
                    f"Coupling waveform leaf must have dtype {spec.dtype}; got {value.dtype}."
                )
        self.grid = grid
        self.values = jax.tree.unflatten(treedef, arrays)
        self.space_id = space.space_id
        self.sample_count = grid.num_points

    def sample(self, index: Any, space: AbstractVectorSpace, /) -> Any:
        if space.space_id != self.space_id:
            raise ValueError("Coupling waveform space identity mismatch.")
        return space.validate(jax.tree.map(lambda value: value[index], self.values))

    @classmethod
    def constant(
        cls,
        grid: TimeGrid,
        value: Any,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        validated = space.validate(value)
        values = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (grid.num_points, *leaf.shape)),
            validated,
        )
        return cls(grid, values, space)


class AbstractCouplingTemporalTransfer(StrictModule, NonTrainableState):
    transfer_id: AbstractAttribute[str]

    @abc.abstractmethod
    def interpolate(
        self,
        waveform: CouplingWaveform,
        target_grid: TimeGrid,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        raise NotImplementedError


class HeldCouplingTemporalTransfer(AbstractCouplingTemporalTransfer):
    """Left-held interpolation with explicit no-extrapolation checks."""

    transfer_id: str = eqx.field(static=True, default="coupling-temporal:held-left")

    def interpolate(
        self,
        waveform: CouplingWaveform,
        target_grid: TimeGrid,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        return interpolate_coupling_waveform(waveform, target_grid, space, kind="held")


class LinearCouplingTemporalTransfer(AbstractCouplingTemporalTransfer):
    """Piecewise-linear interpolation with explicit no-extrapolation checks."""

    transfer_id: str = eqx.field(static=True, default="coupling-temporal:linear")

    def interpolate(
        self,
        waveform: CouplingWaveform,
        target_grid: TimeGrid,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        return interpolate_coupling_waveform(waveform, target_grid, space, kind="linear")


def coupling_signal_structure(port, /) -> Any:
    structure = port.space.structure()
    if port.sample_grid is None:
        return structure
    return jax.tree.map(
        lambda spec: jax.ShapeDtypeStruct(
            (port.sample_grid.num_points, *spec.shape), spec.dtype
        ),
        structure,
    )


def validate_coupling_signal(port, value: Any, /) -> Any:
    if port.sample_grid is None:
        return port.space.validate(value)
    if not isinstance(value, CouplingWaveform):
        raise TypeError(f"Waveform port {port.port_id!r} requires CouplingWaveform data.")
    if value.space_id != port.space.space_id:
        raise ValueError(f"Waveform port {port.port_id!r} space identity mismatch.")
    if value.grid.time_id != port.sample_grid.time_id:
        raise ValueError(f"Waveform port {port.port_id!r} grid identity mismatch.")
    return value


def coupling_signal_finite(value: Any, /) -> Array:
    if isinstance(value, CouplingWaveform):
        finite = jnp.asarray(True)
        for leaf in jax.tree.leaves(value.values):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite & jnp.all(jnp.isfinite(value.grid.times))
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(value):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def flatten_coupling_signal(port, value: Any, /) -> Array:
    validated = validate_coupling_signal(port, value)
    if port.sample_grid is None:
        return port.space.flatten(validated)
    samples = tuple(
        port.space.flatten(validated.sample(index, port.space))
        for index in range(port.sample_grid.num_points)
    )
    return samples[0] if len(samples) == 1 else jnp.concatenate(samples)


def unflatten_coupling_signal(port, coordinates: Array, /) -> Any:
    value = jnp.asarray(coordinates)
    if port.sample_grid is None:
        return port.space.unflatten(value)
    expected = port.sample_grid.num_points * port.space.size
    if value.shape != (expected,):
        raise ValueError(
            f"Waveform coordinates must have shape {(expected,)}; got {value.shape}."
        )
    samples = tuple(
        port.space.unflatten(
            value[index * port.space.size : (index + 1) * port.space.size]
        )
        for index in range(port.sample_grid.num_points)
    )
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)
    return CouplingWaveform(port.sample_grid, values, port.space)


def subtract_coupling_signals(port, left: Any, right: Any, /) -> Any:
    left_ = validate_coupling_signal(port, left)
    right_ = validate_coupling_signal(port, right)
    if port.sample_grid is None:
        return jax.tree.map(lambda x, y: x - y, left_, right_)
    return CouplingWaveform(
        port.sample_grid,
        jax.tree.map(lambda x, y: x - y, left_.values, right_.values),
        port.space,
    )


def coupling_signal_norm(port, value: Any, /) -> Array:
    validated = validate_coupling_signal(port, value)
    if port.sample_grid is None:
        squared = jnp.real(port.space.inner(validated, validated))
        return jnp.sqrt(jnp.maximum(squared, 0.0))
    times = port.sample_grid.times
    intervals = jnp.diff(times)
    duration = times[-1] - times[0]
    weights = jnp.zeros_like(times)
    weights = weights.at[0].set(0.5 * intervals[0])
    weights = weights.at[-1].set(0.5 * intervals[-1])
    if port.sample_grid.num_points > 2:
        weights = weights.at[1:-1].set(0.5 * (intervals[:-1] + intervals[1:]))
    weights = weights / duration
    squared = jnp.asarray(0.0, dtype=times.dtype)
    for index in range(port.sample_grid.num_points):
        sample = validated.sample(index, port.space)
        squared = squared + weights[index] * jnp.real(port.space.inner(sample, sample))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def interpolate_coupling_waveform(
    waveform: CouplingWaveform,
    target_grid: TimeGrid,
    space: AbstractVectorSpace,
    /,
    *,
    kind: CouplingTemporalInterpolation,
) -> CouplingWaveform:
    if not isinstance(waveform, CouplingWaveform):
        raise TypeError("waveform must be CouplingWaveform.")
    if not isinstance(target_grid, TimeGrid):
        raise TypeError("target_grid must be TimeGrid.")
    if waveform.space_id != space.space_id:
        raise ValueError("Waveform interpolation space identity mismatch.")
    if kind not in ("held", "linear"):
        raise ValueError("Temporal interpolation kind must be 'held' or 'linear'.")
    source_times = waveform.grid.times
    target_times = target_grid.times.astype(source_times.dtype)
    target_times = eqx.error_if(
        target_times,
        jnp.any(target_times < source_times[0])
        | jnp.any(target_times > source_times[-1]),
        "Coupling temporal transfer does not extrapolate.",
    )
    if kind == "held":
        indices = jnp.searchsorted(source_times, target_times, side="right") - 1
        indices = jnp.clip(indices, 0, waveform.sample_count - 1)
        values = jax.tree.map(lambda value: value[indices], waveform.values)
        return CouplingWaveform(target_grid, values, space)

    upper = jnp.searchsorted(source_times, target_times, side="right")
    upper = jnp.clip(upper, 1, waveform.sample_count - 1)
    lower = upper - 1
    left_time = source_times[lower]
    right_time = source_times[upper]
    fraction = (target_times - left_time) / (right_time - left_time)
    fraction = jnp.where(target_times == source_times[-1], 1.0, fraction)

    def interpolate_values(value):
        weight = fraction
        for _ in range(value.ndim - 1):
            weight = weight[..., None]
        return (1.0 - weight) * value[lower] + weight * value[upper]

    values = jax.tree.map(interpolate_values, waveform.values)
    return CouplingWaveform(target_grid, values, space)


def transfer_coupling_signal(
    source_port,
    target_port,
    source_value: Any,
    spatial_action,
    /,
) -> Any:
    """Apply temporal conversion and one supplied linear spatial action."""

    source = validate_coupling_signal(source_port, source_value)
    if source_port.sample_grid is None and target_port.sample_grid is None:
        return target_port.space.validate(spatial_action(source))
    if source_port.sample_grid is None:
        source_waveform = CouplingWaveform.constant(
            target_port.sample_grid, source, source_port.space
        )
    else:
        source_waveform = source
    if target_port.sample_grid is None:
        source_sample = source_waveform.sample(-1, source_port.space)
        return target_port.space.validate(spatial_action(source_sample))
    if source_waveform.grid.time_id != target_port.sample_grid.time_id:
        transfer = (
            HeldCouplingTemporalTransfer()
            if target_port.temporal_interpolation == "held"
            else LinearCouplingTemporalTransfer()
        )
        source_waveform = transfer.interpolate(
            source_waveform, target_port.sample_grid, source_port.space
        )
    samples = tuple(
        target_port.space.validate(
            spatial_action(source_waveform.sample(index, source_port.space))
        )
        for index in range(target_port.sample_grid.num_points)
    )
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)
    return CouplingWaveform(target_port.sample_grid, values, target_port.space)


class FixedGridSubcyclingSubsystem(AbstractCouplingSubsystem):
    """Adapt a fixed substep callback to one sampled coupling waveform."""

    advance_substep: Any
    observe: Any
    input_ports: tuple[CouplingPort, ...]
    output_ports: tuple[CouplingPort, ...]
    capabilities: CouplingSubsystemCapabilities
    subsystem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        advance_substep: Any,
        observe: Any,
        /,
        *,
        subsystem_id: str,
        input_ports: tuple[CouplingPort, ...],
        output_ports: tuple[CouplingPort, ...],
        differentiable: bool,
        discretization_bundle_id: str | None = None,
        counts_complete: bool = True,
    ):
        if not callable(advance_substep) or not callable(observe):
            raise TypeError("Subcycling advance_substep and observe must be callable.")
        inputs = tuple(input_ports)
        outputs = tuple(output_ports)
        if not inputs or not outputs:
            raise ValueError("Fixed-grid subcycling requires input and output ports.")
        if any(port.direction != "input" for port in inputs) or any(
            port.direction != "output" for port in outputs
        ):
            raise ValueError("Subcycling ports have inconsistent directions.")
        if any(port.sample_grid is None for port in (*inputs, *outputs)):
            raise ValueError("Fixed-grid subcycling requires waveform-valued ports.")
        grid_ids = {
            port.sample_grid.time_id
            for port in (*inputs, *outputs)
            if port.sample_grid is not None
        }
        if len(grid_ids) != 1:
            raise ValueError("Subcycling participant ports must share one sample grid.")
        identifier = str(subsystem_id)
        if not identifier:
            raise ValueError("Subcycling subsystem_id must be non-empty.")
        bundle_id = (
            None if discretization_bundle_id is None else str(discretization_bundle_id)
        )
        if bundle_id == "":
            raise ValueError("discretization_bundle_id must be non-empty or None.")
        self.advance_substep = advance_substep
        self.observe = observe
        self.input_ports = inputs
        self.output_ports = outputs
        self.capabilities = CouplingSubsystemCapabilities(
            jit=True,
            differentiable=differentiable,
            deterministic_replay=True,
            fixed_topology=True,
            supports_endpoint=False,
            supports_waveform=True,
            counts_complete=counts_complete,
        )
        self.subsystem_id = identifier
        self.discretization_bundle_id = bundle_id

    def advance_window(
        self,
        window: CouplingWindow,
        start_state: Any,
        inputs: tuple[Any, ...],
        args: Any,
        /,
    ) -> CouplingSubsystemResult:
        waveforms = tuple(
            validate_coupling_signal(port, value)
            for port, value in zip(self.input_ports, inputs, strict=True)
        )
        grid = self.input_ports[0].sample_grid
        if grid is None:
            raise RuntimeError("Prepared subcycling grid is unavailable.")
        initial_inputs = tuple(
            waveform.sample(0, port.space)
            for waveform, port in zip(waveforms, self.input_ports, strict=True)
        )
        initial_outputs = tuple(self.observe(start_state, initial_inputs, args))
        if len(initial_outputs) != len(self.output_ports):
            raise ValueError("Subcycling observe returned the wrong output count.")
        output_samples = [
            [port.space.validate(value)]
            for port, value in zip(self.output_ports, initial_outputs, strict=True)
        ]
        state = start_state
        successful = jnp.asarray(True)
        status = jnp.asarray(0, dtype=jnp.int32)
        residual_norm = jnp.asarray(0.0, dtype=window.start.dtype)
        iterations = jnp.asarray(0, dtype=jnp.int32)
        work = jnp.asarray(0, dtype=jnp.int32)
        auxiliary: list[Any] = []
        for step_index in range(grid.num_steps):
            subwindow = CouplingWindow(
                step_index,
                window.start + grid.times[step_index],
                window.start + grid.times[step_index + 1],
            )
            subinputs = tuple(
                waveform.sample(step_index, port.space)
                for waveform, port in zip(waveforms, self.input_ports, strict=True)
            )
            previous_outputs = tuple(samples[-1] for samples in output_samples)

            def execute(
                _,
                current_window=subwindow,
                current_state=state,
                current_inputs=subinputs,
            ):
                result = self.advance_substep(
                    current_window, current_state, current_inputs, args
                )
                if not isinstance(result, CouplingSubsystemResult):
                    raise TypeError(
                        "Subcycling advance_substep must return CouplingSubsystemResult."
                    )
                if len(result.outputs) != len(self.output_ports):
                    raise ValueError(
                        "Subcycling advance_substep returned the wrong output count."
                    )
                if result.auxiliary is not None:
                    raise ValueError(
                        "Fixed-grid subcycling requires auxiliary=None from each substep."
                    )
                validated_outputs = tuple(
                    port.space.validate(value)
                    for port, value in zip(self.output_ports, result.outputs, strict=True)
                )
                return (
                    result.candidate_state,
                    validated_outputs,
                    result.successful,
                    result.status,
                    result.residual_norm,
                    result.iterations,
                    result.work,
                )

            def skip(
                _,
                current_state=state,
                current_outputs=previous_outputs,
                current_status=status,
                current_residual=residual_norm,
            ):
                return (
                    current_state,
                    current_outputs,
                    jnp.asarray(False),
                    current_status,
                    jnp.asarray(0.0, dtype=current_residual.dtype),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )

            (
                state,
                step_outputs,
                step_successful,
                step_status,
                step_residual,
                step_iterations,
                step_work,
            ) = jax.lax.cond(successful, execute, skip, operand=None)
            status = jnp.where(successful, step_status, status)
            successful = successful & step_successful
            residual_norm = jnp.maximum(residual_norm, step_residual)
            iterations = iterations + step_iterations
            work = work + step_work
            auxiliary.append(step_status)
            for samples, value in zip(output_samples, step_outputs, strict=True):
                samples.append(value)
        outputs = tuple(
            CouplingWaveform(
                port.sample_grid,
                jax.tree.map(lambda *values: jnp.stack(values), *samples),
                port.space,
            )
            for port, samples in zip(self.output_ports, output_samples, strict=True)
        )
        return CouplingSubsystemResult(
            state,
            outputs,
            successful=successful,
            status=status,
            residual_norm=residual_norm,
            iterations=iterations,
            work=work,
            auxiliary=tuple(auxiliary),
        )


__all__ = [
    "AbstractCouplingTemporalTransfer",
    "CouplingTemporalInterpolation",
    "CouplingWaveform",
    "FixedGridSubcyclingSubsystem",
    "HeldCouplingTemporalTransfer",
    "LinearCouplingTemporalTransfer",
    "coupling_signal_finite",
    "coupling_signal_norm",
    "coupling_signal_structure",
    "flatten_coupling_signal",
    "interpolate_coupling_waveform",
    "subtract_coupling_signals",
    "transfer_coupling_signal",
    "unflatten_coupling_signal",
    "validate_coupling_signal",
]
