#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity-bounded tetrahedral metric bisection and complete-family coarsening.

This route reuses Maubach conformity closure and the persistent sibling forest;
it is not general tetrahedral edge collapse, cavity swapping or relocation.
Compiled edits remain provisional until the process-local exact barrier returns.
The canonical adaptation owner retains geometry/lineage/transfer publication.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._meshcore import exact_orient3d
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    AdaptiveSimplexLayout,
    AdaptiveSimplexReport,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex,
    refine_adaptive_simplex,
)
from ._metric import interpolate_mesh_metric, metric_edge_lengths


@final
class DeviceTetraMetricLayout(StrictModule, NonTrainableState):
    """One fixed bucket and a separated split/coarsen hysteresis interval."""

    simplex: AdaptiveSimplexLayout
    lower_length: float = eqx.field(static=True)
    upper_length: float = eqx.field(static=True)

    def __init__(
        self,
        simplex: AdaptiveSimplexLayout,
        /,
        *,
        lower_length: float = 1.0 / np.sqrt(2.0),
        upper_length: float = np.sqrt(2.0),
    ) -> None:
        if not isinstance(simplex, AdaptiveSimplexLayout):
            raise TypeError("simplex must be AdaptiveSimplexLayout.")
        if simplex.dimension != 3 or simplex.ambient_dimension != 3:
            raise ValueError(
                "Device tetra metric adaptation requires physical tetrahedra."
            )
        if (
            not np.isfinite(lower_length)
            or not np.isfinite(upper_length)
            or not 0 < lower_length < upper_length
        ):
            raise ValueError(
                "Metric thresholds must satisfy 0 < lower_length < upper_length."
            )
        self.simplex = simplex
        self.lower_length = float(lower_length)
        self.upper_length = float(upper_length)


@final
class DeviceTetraMetricState(StrictModule, NonTrainableState):
    simplex: AdaptiveSimplexState
    metric: Array


@final
class DeviceTetraMetricReport(StrictModule):
    refinement: AdaptiveSimplexReport
    coarsening: AdaptiveSimplexReport
    maximum_edge_length: Array
    unresolved_long_cells: Array


@final
class DeviceTetraMetricEvidence(StrictModule, NonTrainableState):
    """Actual process-local host barrier work; no all-device claim."""

    status: int = eqx.field(static=True)
    exact_resolution_count: int = eqx.field(static=True)
    host_barriers: int = eqx.field(static=True)
    transferred_bytes: int = eqx.field(static=True)
    committed: bool = eqx.field(static=True)


@final
class DeviceTetraMetricUpdate(StrictModule, NonTrainableState):
    state: DeviceTetraMetricState
    report: DeviceTetraMetricReport
    evidence: DeviceTetraMetricEvidence


@final
class _Provisional(StrictModule):
    state: DeviceTetraMetricState
    report: DeviceTetraMetricReport


def prepare_device_tetra_metric(
    layout: DeviceTetraMetricLayout,
    state: AdaptiveSimplexState,
    metric: NDArray[np.float64],
    /,
) -> DeviceTetraMetricState:
    """Bind already prepared tetrahedral ancestry to owner-local vertex metrics.

    Supply a full-capacity SPD metric, including inactive slots. No geometry is
    downgraded here: the caller must use the canonical adaptation preparation
    and commit its accepted simplex state through that owner.
    """
    if not isinstance(layout, DeviceTetraMetricLayout) or not isinstance(
        state, AdaptiveSimplexState
    ):
        raise TypeError("Expected a tetra metric layout and adaptive simplex state.")
    if state.mesh.signature_id != layout.simplex.mesh_signature_id:
        raise ValueError("The state does not belong to this tetra metric bucket.")
    if (
        not isinstance(metric, np.ndarray)
        or metric.dtype != np.float64
        or metric.shape != (layout.simplex.vertex_capacity, 3, 3)
    ):
        raise TypeError("metric must be a float64 (vertex_capacity, 3, 3) host array.")
    if (
        not np.all(np.isfinite(metric))
        or not np.array_equal(metric, np.swapaxes(metric, -1, -2))
        or np.any(np.linalg.eigvalsh(metric) <= 0)
    ):
        raise ValueError(
            "Every vertex-slot metric must be finite, symmetric and positive definite."
        )
    if state.mesh.coordinates.dtype != jnp.float64:
        raise TypeError(
            "Tetra metric worksets use explicit float64 coordinates and metrics."
        )
    return DeviceTetraMetricState(state, jnp.asarray(metric))


def _lengths(state: DeviceTetraMetricState, /) -> Array:
    mesh = state.simplex.mesh
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32)
    edges = mesh.cells[:, pairs].reshape((-1, 2))
    return metric_edge_lengths(state.metric, mesh.coordinates, edges).reshape((-1, 6))


def _issued_metrics(
    source: DeviceTetraMetricState,
    target: AdaptiveSimplexState,
    /,
) -> Array:
    """Midpoint parents precede children in semantic-ID order, including closure."""
    created = (source.simplex.mesh.vertex_ids < 0) & (target.mesh.vertex_ids >= 0)

    def step(slot: Array, values: Array) -> Array:
        def interpolate(current: Array) -> Array:
            parents = target.vertex_parents[slot]
            tensors = current[jnp.maximum(parents, 0)]
            value = interpolate_mesh_metric(
                tensors, jnp.full((2,), 0.5, dtype=jnp.float64)
            )
            return current.at[slot].set(value)

        return jax.lax.cond(created[slot], interpolate, lambda current: current, values)

    return jax.lax.fori_loop(0, target.mesh.vertex_capacity, step, source.metric)


def _adapt(
    layout: DeviceTetraMetricLayout, source: DeviceTetraMetricState, /
) -> _Provisional:
    lengths = _lengths(source)
    marks = source.simplex.mesh.cell_active & (
        jnp.max(lengths, axis=1) > layout.upper_length
    )
    refined = refine_adaptive_simplex(layout.simplex, source.simplex, marks)
    middle = DeviceTetraMetricState(refined.state, _issued_metrics(source, refined.state))
    lengths = _lengths(middle)
    # Only complete ancestral families are restored by the substrate. A parent
    # edge between the siblings must ALSO be short; child edges alone can hide it.
    parents = jnp.maximum(middle.simplex.parents, 0)
    parent_rows = middle.simplex.mesh.cells[parents]
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32)
    parent_lengths = metric_edge_lengths(
        middle.metric,
        middle.simplex.mesh.coordinates,
        parent_rows[:, pairs].reshape((-1, 2)),
    ).reshape((-1, 6))
    small = (
        jnp.maximum(jnp.max(lengths, axis=1), jnp.max(parent_lengths, axis=1))
        < layout.lower_length
    )
    marks = middle.simplex.mesh.cell_active & (middle.simplex.parents >= 0) & small
    coarsened = coarsen_adaptive_simplex(layout.simplex, middle.simplex, marks)
    target = DeviceTetraMetricState(coarsened.state, middle.metric)
    final_lengths = jnp.where(
        target.simplex.mesh.cell_active[:, None], _lengths(target), 0.0
    )
    report = DeviceTetraMetricReport(
        refined.report,
        coarsened.report,
        jnp.max(final_lengths),
        jnp.sum(
            target.simplex.mesh.cell_active
            & (jnp.max(final_lengths, axis=1) > layout.upper_length),
            dtype=jnp.int32,
        ),
    )
    return _Provisional(target, report)


_compiled_adapt = eqx.filter_jit(_adapt)
_TERMINAL = int(
    AdaptiveSimplexStatus.CAPACITY_EXCEEDED
    | AdaptiveSimplexStatus.CLOSURE_LIMIT
    | AdaptiveSimplexStatus.PROTECTED_CONFLICT
    | AdaptiveSimplexStatus.INVALID_GEOMETRY
)


def adapt_device_tetra_metric(
    layout: DeviceTetraMetricLayout,
    state: DeviceTetraMetricState,
    /,
) -> DeviceTetraMetricUpdate:
    """One compiled refine/coarsen cycle followed by one explicit exact barrier.

    Failure of either phase rolls the ENTIRE cycle back, including newly issued
    metrics, and retains terminal status on the rejected epoch. This function is
    process-local; distributed cavity resolution/collective commit is not implied.
    """
    if not isinstance(layout, DeviceTetraMetricLayout) or not isinstance(
        state, DeviceTetraMetricState
    ):
        raise TypeError("Expected tetra metric layout and state.")
    if (
        state.simplex.mesh.signature_id != layout.simplex.mesh_signature_id
        or state.metric.shape != (layout.simplex.vertex_capacity, 3, 3)
    ):
        raise ValueError("The tetra metric state does not belong to this layout.")
    if any(
        isinstance(leaf, Array) and not leaf.is_fully_addressable
        for leaf in jax.tree_util.tree_leaves(state)
    ):
        raise ValueError("Tetra metric exact barriers require process-local arrays.")
    pending = _compiled_adapt(layout, state)
    # One explicit process-local transaction barrier, never one transfer per cell.
    host = jax.device_get(pending)
    status = int(np.asarray(host.state.simplex.status_flags))
    exact = 0
    if not status & _TERMINAL and status & int(
        AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION
    ):
        mesh = host.state.simplex.mesh
        active = np.asarray(mesh.cell_active)
        corners = np.asarray(mesh.coordinates)[np.asarray(mesh.cells)[active]]
        exact = corners.shape[0]
        signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
        if np.any(signs != 1):
            status |= int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    committed = not bool(status & _TERMINAL)
    target = pending.state if committed else state
    # Preserve the poison flag on rollback, preventing a later accidental commit.
    if not committed:
        simplex = eqx.tree_at(
            lambda value: value.clocks,
            state.simplex,
            state.simplex.clocks.at[2].set(jnp.asarray(status, dtype=jnp.int32)),
        )
        target = DeviceTetraMetricState(simplex, state.metric)
    transferred = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree_util.tree_leaves(host)
        if isinstance(leaf, np.ndarray)
    )
    evidence = DeviceTetraMetricEvidence(status, exact, 1, transferred, committed)
    return DeviceTetraMetricUpdate(target, pending.report, evidence)


__all__ = [
    "DeviceTetraMetricLayout",
    "DeviceTetraMetricState",
    "DeviceTetraMetricReport",
    "DeviceTetraMetricEvidence",
    "DeviceTetraMetricUpdate",
    "prepare_device_tetra_metric",
    "adapt_device_tetra_metric",
]
