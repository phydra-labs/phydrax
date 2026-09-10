#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..._execution_runtime import ExecutionGroup
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._trainable import place_array_leaves
from .data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorTargetBatch,
)


class OperatorShardingPolicy(StrictModule):
    """Named case-axis sharding with replicated sample geometry and parameters."""

    mesh: Mesh
    mesh_axis: str
    case_axis: int
    execution_group_id: str | None = eqx.field(static=True)
    coordinator_process: int = eqx.field(static=True)

    def __init__(
        self,
        mesh: Mesh | None = None,
        /,
        *,
        mesh_axis: str = "data",
        case_axis: int = 0,
        execution_group_id: str | None = None,
        coordinator_process: int = 0,
    ):
        axis = str(mesh_axis)
        if mesh is None:
            mesh = Mesh(np.asarray(jax.devices()), (axis,))
        if axis not in mesh.axis_names:
            raise ValueError(f"Mesh has no axis named {axis!r}.")
        if coordinator_process < 0:
            raise ValueError("coordinator_process must be non-negative.")
        self.mesh = mesh
        self.mesh_axis = axis
        self.case_axis = int(case_axis)
        self.execution_group_id = execution_group_id
        self.coordinator_process = int(coordinator_process)

    @classmethod
    def from_execution_group(
        cls,
        execution_group: ExecutionGroup,
        /,
        *,
        mesh_axis: str | None = None,
        case_axis: int = 0,
    ) -> OperatorShardingPolicy:
        axis = execution_group.mesh.axis_names[0] if mesh_axis is None else mesh_axis
        return cls(
            execution_group.mesh,
            mesh_axis=axis,
            case_axis=case_axis,
            execution_group_id=execution_group.spec.group_id,
            coordinator_process=min(execution_group.spec.process_indices),
        )

    @property
    def replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    @property
    def data_axis_size(self) -> int:
        return int(self.mesh.shape[self.mesh_axis])

    @property
    def is_primary_process(self) -> bool:
        return jax.process_index() == self.coordinator_process

    def validate_case_shape(self, case_shape: tuple[int, ...], /) -> None:
        if not case_shape:
            raise ValueError("Case-axis sharding requires a non-empty case shape.")
        axis = self.case_axis
        if axis < 0:
            axis += len(case_shape)
        if axis < 0 or axis >= len(case_shape):
            raise ValueError("Sharding case_axis does not name a case dimension.")
        if int(case_shape[axis]) % self.data_axis_size:
            raise ValueError(
                f"Sharded case dimension {case_shape[axis]} must be divisible by "
                f"mesh axis size {self.data_axis_size}."
            )

    def synchronize(self, name: str, /) -> None:
        """Barrier all JAX processes at one named training lifecycle boundary."""
        multihost_utils.sync_global_devices(str(name))

    def for_array(self, ndim: int, /, *, per_case: bool) -> NamedSharding:
        if not per_case:
            return self.replicated
        axis = self.case_axis
        if axis < 0:
            axis += int(ndim)
        if axis < 0 or axis >= int(ndim):
            raise ValueError("Configured case sharding axis is out of range.")
        partitions: list[Any] = [None] * int(ndim)
        partitions[axis] = self.mesh_axis
        return NamedSharding(self.mesh, PartitionSpec(*partitions))


def _put_array(
    value,
    policy: OperatorShardingPolicy,
    /,
    *,
    per_case: bool,
    process_local: bool = False,
    global_case_count: int | None = None,
):
    array = jax.numpy.asarray(value)
    sharding = policy.for_array(array.ndim, per_case=per_case)
    if not process_local or not per_case:
        return jax.device_put(array, sharding)
    if global_case_count is None or global_case_count <= 0:
        raise ValueError(
            "process-local case arrays require a positive global_case_count."
        )
    axis = policy.case_axis if policy.case_axis >= 0 else array.ndim + policy.case_axis
    global_shape = list(array.shape)
    global_shape[axis] = int(global_case_count)
    policy.validate_case_shape(tuple(global_shape))
    return jax.make_array_from_process_local_data(
        sharding,
        np.asarray(jax.device_get(array)),
        tuple(global_shape),
    )


def _shard_axis(axis: OperatorAxis, policy: OperatorShardingPolicy, /) -> OperatorAxis:
    weights = (
        None
        if axis.quadrature_weights is None
        else _put_array(axis.quadrature_weights, policy, per_case=False)
    )
    return OperatorAxis(
        axis.name,
        _put_array(axis.nodes, policy, per_case=False),
        quadrature_weights=weights,
        basis=axis.basis,
        periodic=axis.periodic,
    )


def _shard_samples(
    samples: FunctionSamples,
    policy: OperatorShardingPolicy,
    /,
    *,
    has_cases: bool,
    process_local: bool = False,
    global_case_count: int | None = None,
) -> FunctionSamples:
    if samples.values is None:
        values = None
    else:
        values = jax.tree_util.tree_map(
            lambda leaf: _put_array(
                leaf,
                policy,
                per_case=has_cases,
                process_local=process_local,
                global_case_count=global_case_count,
            ),
            samples.values,
        )
    geometry_cases = bool(samples.geometry_case_shape)
    coordinates = (
        None
        if samples.coordinates is None
        else _put_array(
            samples.coordinates,
            policy,
            per_case=geometry_cases,
            process_local=process_local,
            global_case_count=global_case_count,
        )
    )
    quadrature = (
        None
        if samples.quadrature_weights is None
        else _put_array(
            samples.quadrature_weights,
            policy,
            per_case=geometry_cases,
            process_local=process_local,
            global_case_count=global_case_count,
        )
    )
    mask = (
        None
        if samples.mask is None
        else _put_array(
            samples.mask,
            policy,
            per_case=geometry_cases,
            process_local=process_local,
            global_case_count=global_case_count,
        )
    )
    topology = (
        None
        if samples.topology is None
        else jax.tree_util.tree_map(
            lambda leaf: (
                jax.device_put(leaf, policy.replicated) if eqx.is_array(leaf) else leaf
            ),
            samples.topology,
        )
    )
    return FunctionSamples(
        values=values,
        axes=tuple(_shard_axis(axis, policy) for axis in samples.axes),
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
        topology=topology,
    )


def shard_operator_batch(
    batch: OperatorBatch,
    policy: OperatorShardingPolicy,
    /,
    *,
    process_local: bool = False,
    global_case_count: int | None = None,
) -> OperatorBatch:
    """Place case dimensions on a named mesh while replicating shared geometry."""
    case_shape = list(batch.case_shape)
    if process_local:
        if global_case_count is None:
            raise ValueError("process-local operator batches require global_case_count.")
        axis = (
            policy.case_axis
            if policy.case_axis >= 0
            else len(case_shape) + policy.case_axis
        )
        case_shape[axis] = int(global_case_count)
    policy.validate_case_shape(tuple(case_shape))
    inputs = frozendict(
        {
            name: _shard_samples(
                samples,
                policy,
                has_cases=True,
                process_local=process_local,
                global_case_count=global_case_count,
            )
            for name, samples in batch.inputs.items()
        }
    )
    queries = {
        name: _shard_samples(
            samples,
            policy,
            has_cases=samples.values is not None,
            process_local=process_local,
            global_case_count=global_case_count,
        )
        for name, samples in batch.queries.items()
    }
    return OperatorBatch(
        inputs=inputs,
        queries=queries,
        case_axes=batch.case_axes,
        case_shape=tuple(case_shape),
    )


def shard_operator_targets(
    targets: OperatorTargetBatch,
    policy: OperatorShardingPolicy,
    /,
    *,
    process_local: bool = False,
    global_case_count: int | None = None,
) -> OperatorTargetBatch:
    """Shard every supervised field along the configured case dimension."""
    case_shape = list(targets.case_shape)
    if process_local:
        if global_case_count is None:
            raise ValueError("process-local operator targets require global_case_count.")
        axis = (
            policy.case_axis
            if policy.case_axis >= 0
            else len(case_shape) + policy.case_axis
        )
        case_shape[axis] = int(global_case_count)
    policy.validate_case_shape(tuple(case_shape))
    return targets.map_values(
        lambda value: _put_array(
            value,
            policy,
            per_case=True,
            process_local=process_local,
            global_case_count=global_case_count,
        )
    )


def shard_operator_case_array(
    value,
    policy: OperatorShardingPolicy,
    /,
    *,
    process_local: bool = False,
    global_case_count: int | None = None,
):
    """Shard one array whose leading dimensions are operator case dimensions."""

    array = jax.numpy.asarray(value)
    shape = list(array.shape)
    if process_local:
        if global_case_count is None:
            raise ValueError("process-local case arrays require global_case_count.")
        axis = (
            policy.case_axis if policy.case_axis >= 0 else array.ndim + policy.case_axis
        )
        shape[axis] = int(global_case_count)
    policy.validate_case_shape(tuple(shape))
    return _put_array(
        array,
        policy,
        per_case=True,
        process_local=process_local,
        global_case_count=global_case_count,
    )


def replicate_operator_model(model, policy: OperatorShardingPolicy, /):
    """Replicate every array leaf of a model on the policy mesh."""
    return place_array_leaves(model, policy.replicated)


__all__ = [
    "OperatorShardingPolicy",
    "replicate_operator_model",
    "shard_operator_batch",
    "shard_operator_case_array",
    "shard_operator_targets",
]
