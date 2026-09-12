#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static-bucket JAX FillPatch execution for variable logical patch boxes."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._variable import (
    VariablePatchFieldState,
    VariablePatchHierarchyTopology,
)


class VariablePatchFillSource(IntEnum):
    INACTIVE = 0
    INTERIOR = 1
    SAME_LEVEL = 2
    PERIODIC = 3
    COARSE_TIME_INTERPOLATED = 4
    PHYSICAL_BOUNDARY = 5
    UNRESOLVED = 6


class VariablePatchHierarchyState(StrictModule):
    """One cell-field payload per variable-patch level and exact topology epoch."""

    topology: VariablePatchHierarchyTopology
    levels: tuple[VariablePatchFieldState, ...]
    component_shape: tuple[int, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        levels: Sequence[VariablePatchFieldState],
        /,
    ):
        values = tuple(levels)
        if (
            not isinstance(topology, VariablePatchHierarchyTopology)
            or len(values) != len(topology.levels)
            or any(
                not isinstance(state, VariablePatchFieldState)
                or state.metadata.metadata_id != metadata.metadata_id
                for state, metadata in zip(values, topology.levels, strict=True)
            )
        ):
            raise ValueError("Variable patch hierarchy state does not match topology.")
        component_shapes = tuple(
            tuple(value.shape[1 + len(bucket.signature.envelope_shape) :])
            for level, state in zip(topology.plan.levels, values, strict=True)
            for bucket, value in zip(level.buckets, state.values, strict=True)
        )
        if len(set(component_shapes)) != 1:
            raise ValueError(
                "Variable patch hierarchy levels require one field component shape."
            )
        self.topology = topology
        self.levels = values
        self.component_shape = component_shapes[0]
        self.state_id = canonical_fingerprint(
            {
                "kind": "variable-patch-hierarchy-state",
                "epoch": topology.epoch.epoch_id,
                "field_shapes": [
                    [list(value.shape) for value in state.values] for state in values
                ],
                "dtypes": [
                    [str(value.dtype) for value in state.values] for state in values
                ],
            }
        )


class VariablePatchFillPatchWorkspace(StrictModule):
    values: tuple[Array, ...]
    valid: tuple[Array, ...]
    source_class: tuple[Array, ...]
    workspace_id: str = eqx.field(static=True)


class VariablePatchPhysicalBoundaryRequest(StrictModule, NonTrainableState):
    level: int = eqx.field(static=True)
    masks: tuple[Array, ...]
    request_id: str = eqx.field(static=True)


class VariablePatchFillPatchResult(StrictModule):
    workspaces: tuple[VariablePatchFillPatchWorkspace, ...]
    physical_boundary_requests: tuple[VariablePatchPhysicalBoundaryRequest, ...]
    complete: Array
    result_id: str = eqx.field(static=True)

    def require_complete(self, /) -> None:
        if isinstance(self.complete, jax.core.Tracer):
            raise RuntimeError("FillPatch completeness must be checked outside tracing.")
        if not bool(self.complete):
            raise ValueError("Variable patch FillPatch has unresolved active values.")


class VariablePatchFillPatchPlan(StrictModule, NonTrainableState):
    """Per-level static route plan over finite variable-patch bucket envelopes."""

    topology: VariablePatchHierarchyTopology
    level: int = eqx.field(static=True)
    source_class: tuple[Array, ...]
    same_indices: tuple[Array, ...]
    coarse_indices: tuple[Array, ...]
    active_padding: tuple[Array, ...]
    physical_masks: tuple[Array, ...]
    bucket_offsets: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        level: int,
        /,
        *,
        component_shape: Sequence[int] = (),
    ):
        if not isinstance(topology, VariablePatchHierarchyTopology):
            raise TypeError("Variable patch FillPatch requires realized topology.")
        level_ = int(level)
        if level_ < 0 or level_ >= len(topology.levels):
            raise ValueError("Variable patch FillPatch level is out of range.")
        components = tuple(int(value) for value in component_shape)
        if any(value <= 0 for value in components):
            raise ValueError("Variable patch FillPatch component shape is invalid.")
        metadata = topology.levels[level_]
        level_plan = topology.plan.levels[level_]
        offsets: list[int] = []
        next_offset = 0
        for bucket in level_plan.buckets:
            offsets.append(next_offset)
            next_offset += bucket.lane_capacity * prod(bucket.signature.envelope_shape)
        coarse_offsets: tuple[int, ...] = ()
        if level_ > 0:
            previous = topology.plan.levels[level_ - 1]
            values = []
            next_coarse = 0
            for bucket in previous.buckets:
                values.append(next_coarse)
                next_coarse += bucket.lane_capacity * prod(
                    bucket.signature.envelope_shape
                )
            coarse_offsets = tuple(values)
        source_classes: list[Array] = []
        same_routes: list[Array] = []
        coarse_routes: list[Array] = []
        active_padding: list[Array] = []
        physical_masks: list[Array] = []
        global_shape = topology.plan.global_cell_shapes[level_]
        ratio = None if level_ == 0 else topology.plan.levels[level_ - 1].refinement_ratio
        for bucket_index, bucket in enumerate(level_plan.buckets):
            signature = bucket.signature
            padded_shape = tuple(
                extent + 2 * halo
                for extent, halo in zip(
                    signature.envelope_shape, signature.halo_width, strict=True
                )
            )
            route_shape = (bucket.lane_capacity,) + padded_shape
            source = np.full(
                route_shape, int(VariablePatchFillSource.INACTIVE), dtype=np.int8
            )
            same = np.full(route_shape, -1, dtype=np.int32)
            coarse = np.full(route_shape, -1, dtype=np.int32)
            active = np.zeros(route_shape, dtype=bool)
            physical = np.zeros(route_shape, dtype=bool)
            lower = np.asarray(metadata.lower[bucket_index], dtype=np.int32)
            extent = np.asarray(metadata.extent[bucket_index], dtype=np.int32)
            lane_active = np.asarray(metadata.active[bucket_index], dtype=bool)
            for lane in range(bucket.lane_capacity):
                if not lane_active[lane]:
                    continue
                for padded in np.ndindex(padded_shape):
                    local = tuple(
                        index - halo
                        for index, halo in zip(padded, signature.halo_width, strict=True)
                    )
                    route_index = (lane,) + padded
                    interior = all(
                        0 <= value < extent[lane, axis]
                        for axis, value in enumerate(local)
                    )
                    if any(
                        value < -signature.halo_width[axis]
                        or value >= int(extent[lane, axis]) + signature.halo_width[axis]
                        for axis, value in enumerate(local)
                    ):
                        continue
                    if interior:
                        source[route_index] = int(VariablePatchFillSource.INTERIOR)
                        active[route_index] = True
                        same[route_index] = (
                            offsets[bucket_index]
                            + lane * prod(signature.envelope_shape)
                            + int(np.ravel_multi_index(local, signature.envelope_shape))
                        )
                        continue
                    raw = tuple(
                        int(lower[lane, axis]) + value for axis, value in enumerate(local)
                    )
                    mapped = list(raw)
                    periodic = False
                    external = False
                    for axis, size in enumerate(global_shape):
                        if 0 <= mapped[axis] < size:
                            continue
                        if topology.plan.periodic_axes[axis]:
                            mapped[axis] %= size
                            periodic = True
                        else:
                            external = True
                    if external:
                        source[route_index] = int(
                            VariablePatchFillSource.PHYSICAL_BOUNDARY
                        )
                        active[route_index] = True
                        physical[route_index] = True
                        continue
                    located = topology.locate(level_, tuple(mapped))
                    if located is not None:
                        donor_bucket, donor_lane, donor_box = located
                        donor_signature = level_plan.buckets[donor_bucket].signature
                        donor_local = tuple(
                            value - start
                            for value, start in zip(mapped, donor_box.lower, strict=True)
                        )
                        source[route_index] = int(
                            VariablePatchFillSource.PERIODIC
                            if periodic
                            else VariablePatchFillSource.SAME_LEVEL
                        )
                        active[route_index] = True
                        same[route_index] = (
                            offsets[donor_bucket]
                            + donor_lane * prod(donor_signature.envelope_shape)
                            + int(
                                np.ravel_multi_index(
                                    donor_local,
                                    donor_signature.envelope_shape,
                                )
                            )
                        )
                        continue
                    if level_ > 0 and ratio is not None:
                        coarse_coordinate = tuple(value // ratio for value in mapped)
                        coarse_located = topology.locate(level_ - 1, coarse_coordinate)
                        if coarse_located is not None:
                            donor_bucket, donor_lane, donor_box = coarse_located
                            donor_signature = (
                                topology.plan.levels[level_ - 1]
                                .buckets[donor_bucket]
                                .signature
                            )
                            donor_local = tuple(
                                value - start
                                for value, start in zip(
                                    coarse_coordinate,
                                    donor_box.lower,
                                    strict=True,
                                )
                            )
                            source[route_index] = int(
                                VariablePatchFillSource.COARSE_TIME_INTERPOLATED
                            )
                            active[route_index] = True
                            coarse[route_index] = (
                                coarse_offsets[donor_bucket]
                                + donor_lane * prod(donor_signature.envelope_shape)
                                + int(
                                    np.ravel_multi_index(
                                        donor_local,
                                        donor_signature.envelope_shape,
                                    )
                                )
                            )
                            continue
                    source[route_index] = int(VariablePatchFillSource.UNRESOLVED)
                    active[route_index] = True
            source_classes.append(jnp.asarray(source))
            same_routes.append(jnp.asarray(same))
            coarse_routes.append(jnp.asarray(coarse))
            active_padding.append(jnp.asarray(active))
            physical_masks.append(jnp.asarray(physical))
        if any(
            np.any(np.asarray(source) == int(VariablePatchFillSource.UNRESOLVED))
            for source in source_classes
        ):
            raise ValueError("Variable patch FillPatch has an unresolved route.")
        self.topology = topology
        self.level = level_
        self.source_class = tuple(source_classes)
        self.same_indices = tuple(same_routes)
        self.coarse_indices = tuple(coarse_routes)
        self.active_padding = tuple(active_padding)
        self.physical_masks = tuple(physical_masks)
        self.bucket_offsets = tuple(offsets)
        self.component_shape = components
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-fill-patch-plan",
                "epoch": topology.epoch.epoch_id,
                "level": level_,
                "source_classes": [
                    array_tree_fingerprint(value) for value in source_classes
                ],
                "same_routes": [array_tree_fingerprint(value) for value in same_routes],
                "coarse_routes": [
                    array_tree_fingerprint(value) for value in coarse_routes
                ],
            }
        )

    @staticmethod
    def _flatten(level: VariablePatchFieldState, /) -> Array:
        component_count = prod(level.component_shape) if level.component_shape else 1
        return jnp.concatenate(
            tuple(value.reshape((-1, component_count)) for value in level.safe_values()),
            axis=0,
        )

    def execute(
        self,
        state: VariablePatchHierarchyState,
        coarse_old: VariablePatchHierarchyState | None = None,
        coarse_new: VariablePatchHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_values: Sequence[ArrayLike | None] | None = None,
        /,
    ) -> tuple[VariablePatchFillPatchWorkspace, VariablePatchPhysicalBoundaryRequest]:
        if (
            not isinstance(state, VariablePatchHierarchyState)
            or state.topology.epoch.epoch_id != self.topology.epoch.epoch_id
        ):
            raise ValueError("Variable patch FillPatch state has a stale topology epoch.")
        old = state if coarse_old is None else coarse_old
        new = state if coarse_new is None else coarse_new
        if (
            old.topology.epoch.epoch_id != self.topology.epoch.epoch_id
            or new.topology.epoch.epoch_id != self.topology.epoch.epoch_id
        ):
            raise ValueError("Variable patch FillPatch coarse states have stale epochs.")
        supplied = (
            (None,) * len(self.source_class)
            if physical_values is None
            else tuple(physical_values)
        )
        if len(supplied) != len(self.source_class):
            raise ValueError(
                "Variable patch physical values require one array per bucket."
            )
        if state.component_shape != self.component_shape:
            raise ValueError(
                "Variable patch FillPatch field components do not match plan."
            )
        direct = self._flatten(state.levels[self.level])
        old_flat = (
            direct if self.level == 0 else self._flatten(old.levels[self.level - 1])
        )
        new_flat = (
            direct if self.level == 0 else self._flatten(new.levels[self.level - 1])
        )
        dtype = direct.dtype
        old_time = jnp.asarray(coarse_old_time, dtype=dtype)
        new_time = jnp.asarray(coarse_new_time, dtype=dtype)
        target_time = jnp.asarray(fill_time, dtype=dtype)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(old_time),
                jnp.maximum(jnp.abs(new_time), jnp.abs(target_time)),
            ),
        )
        tolerance = 16.0 * jnp.finfo(dtype).eps * scale
        span = new_time - old_time
        same_time = jnp.abs(span) <= tolerance
        invalid_time = (
            ~jnp.isfinite(old_time)
            | ~jnp.isfinite(new_time)
            | ~jnp.isfinite(target_time)
            | (span < -tolerance)
            | (target_time < old_time - tolerance)
            | (target_time > new_time + tolerance)
            | (same_time & (jnp.abs(target_time - old_time) > tolerance))
        )
        alpha = jnp.where(
            same_time, 0.0, (target_time - old_time) / jnp.where(same_time, 1.0, span)
        )
        alpha = eqx.error_if(
            alpha,
            invalid_time,
            "Variable patch FillPatch time must lie within coarse endpoints.",
        )
        alpha = jnp.clip(alpha, 0.0, 1.0)
        values: list[Array] = []
        valid: list[Array] = []
        for bucket, source, same, coarse, active, physical, boundary in zip(
            self.topology.plan.levels[self.level].buckets,
            self.source_class,
            self.same_indices,
            self.coarse_indices,
            self.active_padding,
            self.physical_masks,
            supplied,
            strict=True,
        ):
            safe_same = jnp.maximum(same, 0)
            direct_values = direct[safe_same].reshape(source.shape + self.component_shape)
            if self.level == 0:
                coarse_values = jnp.zeros_like(direct_values)
            else:
                safe_coarse = jnp.maximum(coarse, 0)
                coarse_values = (
                    (1.0 - alpha) * old_flat[safe_coarse] + alpha * new_flat[safe_coarse]
                ).reshape(source.shape + self.component_shape)
            trailing = (1,) * len(self.component_shape)
            value = jnp.where(
                (source == int(VariablePatchFillSource.COARSE_TIME_INTERPOLATED)).reshape(
                    source.shape + trailing
                ),
                coarse_values,
                direct_values,
            )
            if boundary is not None:
                boundary_array = jnp.asarray(boundary)
                expected = source.shape + self.component_shape
                if boundary_array.shape != expected:
                    raise ValueError(
                        "Variable patch boundary values do not match padded bucket shape."
                    )
                value = jnp.where(
                    physical.reshape(physical.shape + trailing),
                    boundary_array,
                    value,
                )
            is_resolved = (source != int(VariablePatchFillSource.UNRESOLVED)) & ~physical
            if boundary is not None:
                is_resolved = is_resolved | physical
            valid.append(active & is_resolved)
            values.append(
                jnp.where(
                    active.reshape(active.shape + trailing),
                    value,
                    0.0,
                )
            )
        workspace = VariablePatchFillPatchWorkspace(
            tuple(values),
            tuple(valid),
            self.source_class,
            canonical_fingerprint(
                {
                    "kind": "variable-patch-fill-patch-workspace",
                    "plan": self.plan_id,
                    "shapes": [list(value.shape) for value in values],
                }
            ),
        )
        request = VariablePatchPhysicalBoundaryRequest(
            self.level,
            self.physical_masks,
            canonical_fingerprint(
                {
                    "kind": "variable-patch-physical-boundary-request",
                    "plan": self.plan_id,
                    "masks": [
                        array_tree_fingerprint(value) for value in self.physical_masks
                    ],
                }
            ),
        )
        return workspace, request


__all__ = [
    "VariablePatchFillPatchPlan",
    "VariablePatchFillPatchResult",
    "VariablePatchFillPatchWorkspace",
    "VariablePatchFillSource",
    "VariablePatchHierarchyState",
    "VariablePatchPhysicalBoundaryRequest",
]
