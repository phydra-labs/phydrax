#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractVectorSpace
from ._partitioned_coupling_types import (
    AbstractCouplingSubsystem,
    CouplingPort,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingWindow,
    CouplingWindowErrorEstimate,
)


class CouplingWaveformAdaptationPolicy(StrictModule, NonTrainableState):
    """Finite candidate reservoir for deterministic masked node refinement."""

    candidate_nodes: Array
    observable_tolerance: float = eqx.field(static=True)
    maximum_additions_per_attempt: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_nodes: tuple[float, ...],
        /,
        *,
        observable_tolerance: float,
        maximum_additions_per_attempt: int = 1,
    ):
        candidates = np.asarray(candidate_nodes, dtype=float)
        tolerance = float(observable_tolerance)
        maximum = int(maximum_additions_per_attempt)
        if (
            candidates.ndim != 1
            or np.any(~np.isfinite(candidates))
            or np.any(candidates <= 0.0)
            or np.any(candidates >= 1.0)
            or len(np.unique(candidates)) != candidates.size
        ):
            raise ValueError("Waveform candidate nodes must be unique and inside (0, 1).")
        if not math.isfinite(tolerance) or tolerance <= 0.0 or maximum < 1:
            raise ValueError("Waveform adaptation tolerances/capacities are invalid.")
        self.candidate_nodes = jnp.asarray(candidates)
        self.observable_tolerance = tolerance
        self.maximum_additions_per_attempt = maximum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "coupling-waveform-adaptation",
                "candidate_nodes": candidates.tolist(),
                "observable_tolerance": tolerance,
                "maximum_additions_per_attempt": maximum,
            }
        )


class CouplingWaveformGrid(StrictModule):
    """Sorted normalized nodes with a fixed-capacity active prefix."""

    nodes: Array
    active: Array
    sample_count: Array
    capacity_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: Array,
        active: Array,
        sample_count: Array,
        /,
        *,
        capacity_id: str,
    ):
        nodes_ = jnp.asarray(nodes)
        active_ = jnp.asarray(active, dtype=bool)
        count = jnp.asarray(sample_count, dtype=jnp.int32).reshape(())
        if nodes_.ndim != 1 or active_.shape != nodes_.shape:
            raise ValueError("Waveform grid nodes/activity must be equal-length vectors.")
        expected = jnp.arange(nodes_.size, dtype=jnp.int32) < count
        adjacent = jnp.arange(nodes_.size - 1, dtype=jnp.int32) < count - 1
        nodes_ = eqx.error_if(
            nodes_,
            (count < 2)
            | (count > nodes_.size)
            | jnp.any(active_ != expected)
            | jnp.any(~jnp.isfinite(nodes_))
            | (nodes_[0] != 0.0)
            | (nodes_[count - 1] != 1.0)
            | jnp.any(adjacent & (jnp.diff(nodes_) <= 0.0)),
            "Waveform grid must have a sorted active prefix with exact endpoints 0/1.",
        )
        self.nodes = jnp.where(active_, nodes_, jnp.asarray(1.0, dtype=nodes_.dtype))
        self.active = active_
        self.sample_count = count
        self.capacity_id = str(capacity_id)

    @property
    def sample_capacity(self) -> int:
        return int(self.nodes.size)

    @property
    def num_steps(self) -> int:
        return self.sample_capacity - 1


class CouplingWaveformPlan(StrictModule, NonTrainableState):
    """Static sample capacity, polynomial degree, metric, and initial grid."""

    sample_capacity: int = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    initial_nodes: Array
    metric_order: int = eqx.field(static=True)
    adaptation: CouplingWaveformAdaptationPolicy | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_capacity: int,
        polynomial_degree: int,
        initial_nodes: tuple[float, ...],
        /,
        *,
        metric_order: int | None = None,
        adaptation: CouplingWaveformAdaptationPolicy | None = None,
        plan_id: str | None = None,
    ):
        capacity = int(sample_capacity)
        degree = int(polynomial_degree)
        nodes = np.asarray(initial_nodes, dtype=float)
        order = max(degree + 1, 1) if metric_order is None else int(metric_order)
        if capacity < 2 or degree not in (0, 1, 2, 3):
            raise ValueError("Waveform capacity must be >=2 and degree must be 0..3.")
        if (
            nodes.ndim != 1
            or nodes.size < max(2, degree + 1)
            or nodes.size > capacity
            or np.any(~np.isfinite(nodes))
            or nodes[0] != 0.0
            or nodes[-1] != 1.0
            or np.any(np.diff(nodes) <= 0.0)
            or order < 1
        ):
            raise ValueError("Initial waveform nodes/metric order are invalid.")
        if adaptation is not None and not isinstance(
            adaptation, CouplingWaveformAdaptationPolicy
        ):
            raise TypeError(
                "adaptation must be CouplingWaveformAdaptationPolicy or None."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "coupling-waveform-plan",
                    "sample_capacity": capacity,
                    "polynomial_degree": degree,
                    "initial_nodes": nodes.tolist(),
                    "metric_order": order,
                    "adaptation": None if adaptation is None else adaptation.policy_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("Coupling waveform plan_id must be non-empty.")
        self.sample_capacity = capacity
        self.polynomial_degree = degree
        self.initial_nodes = jnp.asarray(nodes)
        self.metric_order = order
        self.adaptation = adaptation
        self.plan_id = identifier

    def initial_grid(self) -> CouplingWaveformGrid:
        count = int(self.initial_nodes.size)
        nodes = jnp.ones((self.sample_capacity,), dtype=self.initial_nodes.dtype)
        nodes = nodes.at[:count].set(self.initial_nodes)
        active = jnp.arange(self.sample_capacity) < count
        return CouplingWaveformGrid(
            nodes,
            active,
            jnp.asarray(count, dtype=jnp.int32),
            capacity_id=self.plan_id,
        )


class CouplingWaveform(StrictModule):
    """Capacity-padded immutable signal; inactive rows are canonical zero."""

    grid: CouplingWaveformGrid
    values: Any
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: CouplingWaveformGrid,
        values: Any,
        space: AbstractVectorSpace,
        /,
    ):
        if not isinstance(grid, CouplingWaveformGrid):
            raise TypeError("Coupling waveform grid must be CouplingWaveformGrid.")
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("Coupling waveform space must be AbstractVectorSpace.")
        leaves, treedef = jax.tree.flatten(values)
        structure_leaves, structure_def = jax.tree.flatten(space.structure())
        if treedef != structure_def or len(leaves) != len(structure_leaves):
            raise ValueError("Coupling waveform values must match their vector space.")
        arrays = tuple(jnp.asarray(value) for value in leaves)
        masked: list[Array] = []
        for value, spec in zip(arrays, structure_leaves, strict=True):
            expected = (grid.sample_capacity, *spec.shape)
            if value.shape != expected or value.dtype != spec.dtype:
                raise ValueError(
                    f"Coupling waveform leaf must have shape/dtype {expected}/{spec.dtype}."
                )
            mask = grid.active.reshape((grid.sample_capacity,) + (1,) * len(spec.shape))
            masked.append(jnp.where(mask, value, jnp.zeros((), dtype=value.dtype)))
        self.grid = grid
        self.values = jax.tree.unflatten(treedef, tuple(masked))
        self.space_id = space.space_id

    def sample(self, index: Any, space: AbstractVectorSpace, /) -> Any:
        if space.space_id != self.space_id:
            raise ValueError("Coupling waveform space identity mismatch.")
        return space.validate(jax.tree.map(lambda value: value[index], self.values))

    @classmethod
    def constant(
        cls,
        grid: CouplingWaveformGrid,
        value: Any,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        validated = space.validate(value)
        values = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (grid.sample_capacity, *leaf.shape)),
            validated,
        )
        return cls(grid, values, space)


class AbstractCouplingTemporalTransfer(StrictModule, NonTrainableState):
    transfer_id: AbstractAttribute[str]

    @abc.abstractmethod
    def interpolate(
        self,
        waveform: CouplingWaveform,
        target_grid: CouplingWaveformGrid,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        raise NotImplementedError


def _barycentric_values(
    waveform: CouplingWaveform,
    target_nodes: Array,
    target_active: Array,
    space: AbstractVectorSpace,
    degree: int,
    /,
) -> Any:
    source_nodes = waveform.grid.nodes
    count = waveform.grid.sample_count
    stencil_size = degree + 1
    count = eqx.error_if(
        count,
        count < stencil_size,
        "Waveform interpolation has insufficient active source nodes.",
    )
    if degree == 0:
        indices = jnp.searchsorted(source_nodes, target_nodes, side="right") - 1
        indices = jnp.clip(indices, 0, count - 1)
        values = jax.tree.map(lambda value: value[indices], waveform.values)
    else:
        upper = jnp.searchsorted(source_nodes, target_nodes, side="right")
        start = jnp.clip(
            upper - (stencil_size + 1) // 2,
            0,
            count - stencil_size,
        )
        indices = start[:, None] + jnp.arange(stencil_size)[None, :]
        stencil_nodes = source_nodes[indices]
        weights = jnp.ones_like(stencil_nodes)
        for left in range(stencil_size):
            for right in range(stencil_size):
                if left != right:
                    weights = weights.at[:, left].multiply(
                        (target_nodes - stencil_nodes[:, right])
                        / (stencil_nodes[:, left] - stencil_nodes[:, right])
                    )
        values = jax.tree.map(
            lambda value: jnp.sum(
                weights.reshape(weights.shape + (1,) * (value.ndim - 1)) * value[indices],
                axis=1,
            ),
            waveform.values,
        )
    return jax.tree.map(
        lambda value: jnp.where(
            target_active.reshape((target_active.size,) + (1,) * (value.ndim - 1)),
            value,
            jnp.zeros((), dtype=value.dtype),
        ),
        values,
    )


class BarycentricCouplingTemporalTransfer(AbstractCouplingTemporalTransfer):
    """Degree-zero through cubic nonuniform local Lagrange transfer."""

    degree: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(self, degree: int = 1, /):
        degree_ = int(degree)
        if degree_ not in (0, 1, 2, 3):
            raise ValueError("Coupling temporal transfer degree must be 0..3.")
        self.degree = degree_
        self.transfer_id = f"coupling-temporal:barycentric:{degree_}"

    def interpolate(
        self,
        waveform: CouplingWaveform,
        target_grid: CouplingWaveformGrid,
        space: AbstractVectorSpace,
        /,
    ) -> CouplingWaveform:
        if waveform.space_id != space.space_id:
            raise ValueError("Waveform interpolation space identity mismatch.")
        target_nodes = eqx.error_if(
            target_grid.nodes,
            jnp.any(
                target_grid.active
                & (
                    (target_grid.nodes < waveform.grid.nodes[0])
                    | (
                        target_grid.nodes
                        > waveform.grid.nodes[waveform.grid.sample_count - 1]
                    )
                )
            ),
            "Coupling temporal transfer does not extrapolate.",
        )
        values = _barycentric_values(
            waveform, target_nodes, target_grid.active, space, self.degree
        )
        return CouplingWaveform(target_grid, values, space)


class CouplingWaveformAdaptationEvidence(StrictModule):
    previous_sample_count: Array
    candidate_index: Array
    candidate_node: Array
    maximum_normalized_defect: Array
    activated: Array
    capacity_exhausted: Array
    reliable: Array


class CouplingWaveformCapacityRequest(StrictModule, NonTrainableState):
    required_samples: Array
    port_id: str = eqx.field(static=True)


def adapt_coupling_waveform_grid(
    plan: CouplingWaveformPlan,
    grid: CouplingWaveformGrid,
    normalized_defects: Array,
    port_id: str,
    /,
) -> tuple[
    CouplingWaveformGrid,
    CouplingWaveformAdaptationEvidence,
    CouplingWaveformCapacityRequest,
]:
    """Activate the largest deterministic candidate or request a host epoch."""

    if plan.adaptation is None:
        raise ValueError("Waveform plan has no adaptation policy.")
    defects = jnp.asarray(normalized_defects, dtype=grid.nodes.dtype)
    candidates = plan.adaptation.candidate_nodes.astype(grid.nodes.dtype)
    if defects.shape != candidates.shape:
        raise ValueError("Waveform defects must match the candidate reservoir.")
    duplicate = jnp.any(
        jnp.abs(candidates[:, None] - grid.nodes[None, :])
        <= 32.0 * jnp.finfo(grid.nodes.dtype).eps,
        axis=1,
    )
    eligible = ~duplicate & jnp.isfinite(defects)
    scored = jnp.where(eligible, defects, -jnp.inf)
    candidate_index = jnp.argmax(scored).astype(jnp.int32)
    maximum = scored[candidate_index]
    needs_refinement = jnp.any(eligible) & (
        maximum > plan.adaptation.observable_tolerance
    )
    capacity_exhausted = needs_refinement & (grid.sample_count >= plan.sample_capacity)
    activated = needs_refinement & ~capacity_exhausted
    candidate_node = candidates[candidate_index]
    insertion = jnp.where(activated, candidate_node, jnp.inf)
    nodes = grid.nodes.at[grid.sample_count].set(insertion)
    nodes = jnp.sort(nodes)
    next_count = grid.sample_count + activated.astype(jnp.int32)
    active = jnp.arange(plan.sample_capacity, dtype=jnp.int32) < next_count
    nodes = jnp.where(active, nodes, jnp.asarray(1.0, dtype=nodes.dtype))
    next_grid = CouplingWaveformGrid(
        nodes, active, next_count, capacity_id=grid.capacity_id
    )
    evidence = CouplingWaveformAdaptationEvidence(
        grid.sample_count,
        jnp.where(needs_refinement, candidate_index, -1),
        jnp.where(needs_refinement, candidate_node, jnp.nan),
        maximum,
        activated,
        capacity_exhausted,
        jnp.all(jnp.isfinite(jnp.where(eligible, defects, 0.0))),
    )
    request = CouplingWaveformCapacityRequest(
        jnp.where(capacity_exhausted, grid.sample_count + 1, grid.sample_count),
        str(port_id),
    )
    return next_grid, evidence, request


def coupling_signal_structure(port, /) -> Any:
    structure = port.space.structure()
    if port.waveform_plan is None:
        return structure
    return jax.tree.map(
        lambda spec: jax.ShapeDtypeStruct(
            (port.waveform_plan.sample_capacity, *spec.shape), spec.dtype
        ),
        structure,
    )


def validate_coupling_signal(port, value: Any, /) -> Any:
    if port.waveform_plan is None:
        return port.space.validate(value)
    if not isinstance(value, CouplingWaveform):
        raise TypeError(f"Waveform port {port.port_id!r} requires CouplingWaveform data.")
    if value.space_id != port.space.space_id:
        raise ValueError(f"Waveform port {port.port_id!r} space identity mismatch.")
    if value.grid.capacity_id != port.waveform_plan.plan_id:
        raise ValueError(f"Waveform port {port.port_id!r} capacity identity mismatch.")
    return value


def coupling_signal_finite(value: Any, /) -> Array:
    if isinstance(value, CouplingWaveform):
        finite = jnp.asarray(True)
        for leaf in jax.tree.leaves(value.values):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite & jnp.all(
            jnp.where(value.grid.active, jnp.isfinite(value.grid.nodes), True)
        )
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(value):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def flatten_coupling_signal(port, value: Any, /) -> Array:
    validated = validate_coupling_signal(port, value)
    if port.waveform_plan is None:
        return port.space.flatten(validated)
    samples = tuple(
        port.space.flatten(validated.sample(index, port.space))
        for index in range(port.waveform_plan.sample_capacity)
    )
    return jnp.concatenate(samples)


def unflatten_coupling_signal(port, coordinates: Array, /) -> Any:
    value = jnp.asarray(coordinates)
    if port.waveform_plan is None:
        return port.space.unflatten(value)
    expected = port.waveform_plan.sample_capacity * port.space.size
    if value.shape != (expected,):
        raise ValueError(f"Waveform coordinates must have shape {(expected,)}.")
    samples = tuple(
        port.space.unflatten(
            value[index * port.space.size : (index + 1) * port.space.size]
        )
        for index in range(port.waveform_plan.sample_capacity)
    )
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)
    return CouplingWaveform(port.waveform_plan.initial_grid(), values, port.space)


def subtract_coupling_signals(port, left: Any, right: Any, /) -> Any:
    left_ = validate_coupling_signal(port, left)
    right_ = validate_coupling_signal(port, right)
    if port.waveform_plan is None:
        return jax.tree.map(lambda x, y: x - y, left_, right_)
    if left_.grid.capacity_id != right_.grid.capacity_id:
        raise ValueError("Coupling residual waveform grids do not match.")
    return CouplingWaveform(
        left_.grid,
        jax.tree.map(lambda x, y: x - y, left_.values, right_.values),
        port.space,
    )


def coupling_signal_norm(port, value: Any, /) -> Array:
    validated = validate_coupling_signal(port, value)
    if port.waveform_plan is None:
        squared = jnp.real(port.space.inner(validated, validated))
        return jnp.sqrt(jnp.maximum(squared, 0.0))
    order = port.waveform_plan.metric_order
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    gauss_nodes_ = jnp.asarray(gauss_nodes, dtype=validated.grid.nodes.dtype)
    gauss_weights_ = jnp.asarray(gauss_weights, dtype=validated.grid.nodes.dtype)
    degree = port.temporal_transfer.degree
    squared = jnp.asarray(0.0, dtype=validated.grid.nodes.dtype)
    for interval_index in range(port.waveform_plan.sample_capacity - 1):
        left = validated.grid.nodes[interval_index]
        right = validated.grid.nodes[interval_index + 1]
        active_interval = interval_index < validated.grid.sample_count - 1
        half = 0.5 * (right - left)
        center = 0.5 * (right + left)
        query = center + half * gauss_nodes_
        query_values = _barycentric_values(
            validated,
            query,
            jnp.ones((order,), dtype=bool),
            port.space,
            degree,
        )
        interval_value = jnp.asarray(0.0, dtype=squared.dtype)
        for quadrature_index in range(order):
            sample = port.space.validate(
                jax.tree.map(lambda leaf: leaf[quadrature_index], query_values)
            )
            interval_value = interval_value + gauss_weights_[quadrature_index] * jnp.real(
                port.space.inner(sample, sample)
            )
        squared = squared + jnp.where(active_interval, half * interval_value, 0.0)
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def transfer_coupling_signal(
    source_port,
    target_port,
    source_value: Any,
    spatial_action,
    /,
) -> Any:
    """Apply one explicit temporal transfer and one supplied spatial action."""

    source = validate_coupling_signal(source_port, source_value)
    if source_port.waveform_plan is None and target_port.waveform_plan is None:
        return target_port.space.validate(spatial_action(source))
    if source_port.waveform_plan is None:
        source_waveform = CouplingWaveform.constant(
            target_port.waveform_plan.initial_grid(), source, source_port.space
        )
    else:
        source_waveform = source
    if target_port.waveform_plan is None:
        source_sample = source_waveform.sample(
            source_waveform.grid.sample_count - 1, source_port.space
        )
        return target_port.space.validate(spatial_action(source_sample))
    target_grid = target_port.waveform_plan.initial_grid()
    source_waveform = target_port.temporal_transfer.interpolate(
        source_waveform, target_grid, source_port.space
    )
    samples = tuple(
        target_port.space.validate(
            spatial_action(source_waveform.sample(index, source_port.space))
        )
        for index in range(target_port.waveform_plan.sample_capacity)
    )
    values = jax.tree.map(lambda *leaves: jnp.stack(leaves), *samples)
    return CouplingWaveform(target_grid, values, target_port.space)


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
        if any(port.waveform_plan is None for port in (*inputs, *outputs)):
            raise ValueError("Fixed-grid subcycling requires waveform-valued ports.")
        plan_ids = {
            port.waveform_plan.plan_id
            for port in (*inputs, *outputs)
            if port.waveform_plan is not None
        }
        if len(plan_ids) != 1:
            raise ValueError("Subcycling participant ports must share one waveform plan.")
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
        grid = waveforms[0].grid
        if any(value.grid.capacity_id != grid.capacity_id for value in waveforms):
            raise ValueError("Subcycling input waveform grids must share one epoch.")
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
        error_norm = jnp.asarray(0.0, dtype=window.start.dtype)
        error_reference = jnp.asarray(1.0, dtype=window.start.dtype)
        error_order = jnp.asarray(1, dtype=jnp.int32)
        error_reliable = jnp.asarray(True)
        auxiliary: list[Any] = []
        for step_index in range(grid.num_steps):
            subwindow = CouplingWindow(
                step_index,
                window.start + window.size * grid.nodes[step_index],
                window.start + window.size * grid.nodes[step_index + 1],
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
                    result.error_estimate.error_norm,
                    result.error_estimate.reference_norm,
                    result.error_estimate.order,
                    result.error_estimate.reliable,
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
                    jnp.asarray(True),
                    current_status,
                    jnp.asarray(0.0, dtype=current_residual.dtype),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0.0, dtype=current_residual.dtype),
                    jnp.asarray(1.0, dtype=current_residual.dtype),
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(True),
                )

            active_step = successful & (step_index < grid.sample_count - 1)
            (
                state,
                step_outputs,
                step_successful,
                step_status,
                step_residual,
                step_iterations,
                step_work,
                step_error,
                step_reference,
                step_order,
                step_reliable,
            ) = jax.lax.cond(active_step, execute, skip, operand=None)
            status = jnp.where(active_step, step_status, status)
            successful = successful & step_successful
            residual_norm = jnp.maximum(residual_norm, step_residual)
            iterations = iterations + step_iterations
            work = work + step_work
            error_norm = jnp.maximum(error_norm, step_error)
            error_reference = jnp.maximum(error_reference, step_reference)
            error_order = jnp.minimum(error_order, step_order)
            error_reliable = error_reliable & step_reliable
            auxiliary.append(step_status)
            for samples, value in zip(output_samples, step_outputs, strict=True):
                samples.append(value)
        outputs = tuple(
            CouplingWaveform(
                port.waveform_plan.initial_grid(),
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
            error_estimate=CouplingWindowErrorEstimate(
                error_norm, error_reference, error_order, error_reliable
            ),
            work=work,
            auxiliary=tuple(auxiliary),
        )


__all__ = [
    "AbstractCouplingTemporalTransfer",
    "BarycentricCouplingTemporalTransfer",
    "CouplingWaveform",
    "CouplingWaveformAdaptationEvidence",
    "CouplingWaveformAdaptationPolicy",
    "CouplingWaveformCapacityRequest",
    "CouplingWaveformGrid",
    "CouplingWaveformPlan",
    "FixedGridSubcyclingSubsystem",
    "adapt_coupling_waveform_grid",
    "coupling_signal_finite",
    "coupling_signal_norm",
    "coupling_signal_structure",
    "flatten_coupling_signal",
    "subtract_coupling_signals",
    "transfer_coupling_signal",
    "unflatten_coupling_signal",
    "validate_coupling_signal",
]
