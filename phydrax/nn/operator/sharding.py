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

from ..._frozendict import frozendict
from ..._strict import StrictModule
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

    def __init__(
        self,
        mesh: Mesh | None = None,
        /,
        *,
        mesh_axis: str = "data",
        case_axis: int = 0,
    ):
        axis = str(mesh_axis)
        if mesh is None:
            mesh = Mesh(np.asarray(jax.devices()), (axis,))
        if axis not in mesh.axis_names:
            raise ValueError(f"Mesh has no axis named {axis!r}.")
        self.mesh = mesh
        self.mesh_axis = axis
        self.case_axis = int(case_axis)

    @property
    def replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    @property
    def data_axis_size(self) -> int:
        return int(self.mesh.shape[self.mesh_axis])

    @property
    def is_primary_process(self) -> bool:
        return jax.process_index() == 0

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


def _put_array(value, policy: OperatorShardingPolicy, /, *, per_case: bool):
    array = jax.numpy.asarray(value)
    return jax.device_put(
        array,
        policy.for_array(array.ndim, per_case=per_case),
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
) -> FunctionSamples:
    if samples.values is None:
        values = None
    else:
        values = jax.tree_util.tree_map(
            lambda leaf: _put_array(leaf, policy, per_case=has_cases),
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
        )
    )
    quadrature = (
        None
        if samples.quadrature_weights is None
        else _put_array(
            samples.quadrature_weights,
            policy,
            per_case=geometry_cases,
        )
    )
    mask = (
        None
        if samples.mask is None
        else _put_array(
            samples.mask,
            policy,
            per_case=geometry_cases,
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
) -> OperatorBatch:
    """Place case dimensions on a named mesh while replicating shared geometry."""
    policy.validate_case_shape(batch.case_shape)
    inputs = frozendict(
        {
            name: _shard_samples(samples, policy, has_cases=True)
            for name, samples in batch.inputs.items()
        }
    )
    queries = {
        name: _shard_samples(
            samples,
            policy,
            has_cases=samples.values is not None,
        )
        for name, samples in batch.queries.items()
    }
    return OperatorBatch(
        inputs=inputs,
        queries=queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def shard_operator_targets(
    targets: OperatorTargetBatch,
    policy: OperatorShardingPolicy,
    /,
) -> OperatorTargetBatch:
    """Shard every supervised field along the configured case dimension."""
    policy.validate_case_shape(targets.case_shape)
    return targets.map_values(lambda value: _put_array(value, policy, per_case=True))


def replicate_operator_model(model, policy: OperatorShardingPolicy, /):
    """Replicate every array leaf of a model on the policy mesh."""
    return eqx.filter_shard(model, policy.replicated)


__all__ = [
    "OperatorShardingPolicy",
    "replicate_operator_model",
    "shard_operator_batch",
    "shard_operator_targets",
]
