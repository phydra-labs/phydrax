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

from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._functional_objective import _PreparedObjective


class FunctionalShardingPolicy(StrictModule, NonTrainableState):
    """Named sample-axis sharding with replicated parameters and shared state."""

    mesh: Mesh
    axis_mapping: frozendict[str, str]
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_mapping: Mapping[str, str],
        /,
        *,
        mesh: Mesh | None = None,
        policy_id: str = "functional-data-parallel",
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
        self.mesh = mesh_
        self.axis_mapping = mapping
        self.policy_id = identifier

    @property
    def replicated(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    @property
    def is_primary_process(self) -> bool:
        return jax.process_index() == 0

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
        return eqx.filter_shard(parameters, self.replicated)

    def place_prepared(self, prepared: _PreparedObjective, /) -> _PreparedObjective:
        if not isinstance(prepared, _PreparedObjective):
            raise TypeError("prepared must be a _PreparedObjective.")
        return self.place_tree(prepared)


__all__ = ["FunctionalShardingPolicy"]
