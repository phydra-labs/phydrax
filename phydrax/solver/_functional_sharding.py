#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._execution_runtime import ExecutionGroup
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState, place_array_leaves
from ._functional_objective import _PreparedObjective


class FunctionalShardingPolicy(StrictModule, NonTrainableState):
    """Named sample-axis sharding with replicated parameters and shared state."""

    mesh: Mesh
    axis_mapping: frozendict[str, str]
    policy_id: str = eqx.field(static=True)
    execution_group_id: str | None = eqx.field(static=True)
    coordinator_process: int = eqx.field(static=True)

    def __init__(
        self,
        axis_mapping: Mapping[str, str],
        /,
        *,
        mesh: Mesh | None = None,
        policy_id: str = "functional-data-parallel",
        execution_group_id: str | None = None,
        coordinator_process: int = 0,
    ):
        mapping = frozendict(
            {str(sample): str(device) for sample, device in axis_mapping.items()}
        )
        if not mapping or any(not key or not value for key, value in mapping.items()):
            raise ValueError("Functional sharding requires named sample-to-mesh axes.")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Each sample axis must use a distinct mesh axis.")
        mesh_ = (
            Mesh(np.asarray(jax.devices()), tuple(mapping.values()))
            if mesh is None and len(mapping) == 1
            else mesh
        )
        if mesh_ is None:
            raise ValueError("Multi-axis functional sharding requires an explicit Mesh.")
        missing = tuple(axis for axis in mapping.values() if axis not in mesh_.axis_names)
        if missing:
            raise ValueError(f"Mesh is missing functional axes {missing!r}.")
        identifier = str(policy_id)
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        if coordinator_process < 0:
            raise ValueError("coordinator_process must be non-negative.")
        self.mesh = mesh_
        self.axis_mapping = mapping
        self.policy_id = identifier
        self.execution_group_id = execution_group_id
        self.coordinator_process = int(coordinator_process)

    @classmethod
    def from_execution_group(
        cls,
        axis_mapping: Mapping[str, str],
        execution_group: ExecutionGroup,
        /,
        *,
        policy_id: str = "functional-data-parallel",
    ) -> FunctionalShardingPolicy:
        return cls(
            axis_mapping,
            mesh=execution_group.mesh,
            policy_id=policy_id,
            execution_group_id=execution_group.spec.group_id,
            coordinator_process=min(execution_group.spec.process_indices),
        )

    @property
    def replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    @property
    def is_primary_process(self) -> bool:
        return jax.process_index() == self.coordinator_process

    def synchronize(self, name: str, /) -> None:
        multihost_utils.sync_global_devices(str(name))

    def field_sharding(self, field: cx.Field, /) -> NamedSharding:
        if not isinstance(field, cx.Field):
            raise TypeError("field must be a coordax.Field.")
        entries: list[str | None] = []
        for axis, size in zip(field.dims, field.data.shape, strict=True):
            device_axis = None if axis is None else self.axis_mapping.get(axis)
            if device_axis is not None:
                device_count = int(self.mesh.shape[device_axis])
                if int(size) % device_count:
                    raise ValueError(
                        f"Sample axis {axis!r} size {size} is not divisible by "
                        f"mesh axis {device_axis!r} size {device_count}."
                    )
            entries.append(device_axis)
        return NamedSharding(self.mesh, PartitionSpec(*entries))

    def place_field(self, field: cx.Field, /) -> cx.Field:
        return cx.Field(
            jax.device_put(field.data, self.field_sharding(field)),
            dims=field.dims,
        )

    def place_tree(self, tree: Any, /, *, replicate_other_arrays: bool = True):
        def place(value):
            if isinstance(value, cx.Field):
                return self.place_field(value)
            if replicate_other_arrays and eqx.is_array(value):
                return jax.device_put(value, self.replicated)
            return value

        return jax.tree.map(
            place,
            tree,
            is_leaf=lambda value: isinstance(value, cx.Field),
        )

    def place_parameters(self, parameters: Any, /):
        return place_array_leaves(parameters, self.replicated)

    def place_prepared(self, prepared: _PreparedObjective, /) -> _PreparedObjective:
        if not isinstance(prepared, _PreparedObjective):
            raise TypeError("prepared must be a _PreparedObjective.")
        return self.place_tree(prepared)


__all__ = ["FunctionalShardingPolicy"]
