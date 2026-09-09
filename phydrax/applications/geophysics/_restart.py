#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    ArrayArchiveLimits,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._evidence import GeophysicalResourceEstimate


class GeophysicalResourcePolicy(StrictModule, NonTrainableState):
    maximum_device_bytes: int = eqx.field(static=True)
    maximum_checkpoint_bytes: int = eqx.field(static=True)
    maximum_sources: int = eqx.field(static=True)
    maximum_observations: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_device_bytes: int,
        maximum_checkpoint_bytes: int,
        maximum_sources: int,
        maximum_observations: int,
        maximum_steps: int,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_device_bytes,
                maximum_checkpoint_bytes,
                maximum_sources,
                maximum_observations,
                maximum_steps,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Geophysical resource limits must be positive integers.")
        (
            self.maximum_device_bytes,
            self.maximum_checkpoint_bytes,
            self.maximum_sources,
            self.maximum_observations,
            self.maximum_steps,
        ) = values
        self.policy_id = canonical_fingerprint(
            {"kind": "geophysical-resource-policy", "limits": values}
        )

    def admit(
        self,
        estimate: GeophysicalResourceEstimate,
        /,
        *,
        source_count: int,
        observation_count: int,
        step_count: int,
    ) -> None:
        if not isinstance(estimate, GeophysicalResourceEstimate):
            raise TypeError("Resource admission requires GeophysicalResourceEstimate.")
        if estimate.total_bytes > self.maximum_device_bytes:
            raise MemoryError("Estimated device allocation exceeds geophysical policy.")
        if estimate.checkpoint_bytes > self.maximum_checkpoint_bytes:
            raise MemoryError(
                "Estimated checkpoint allocation exceeds geophysical policy."
            )
        if int(source_count) > self.maximum_sources:
            raise ValueError("Source count exceeds geophysical resource policy.")
        if int(observation_count) > self.maximum_observations:
            raise ValueError("Observation count exceeds geophysical resource policy.")
        if int(step_count) > self.maximum_steps:
            raise ValueError("Step count exceeds geophysical resource policy.")


class GeophysicalContinuationState(StrictModule):
    physical_state: Any
    auxiliary_state: Any
    inference_state: Any
    time: Array
    accepted_step: Array
    source_position: Array
    random_key: Array
    topology_epoch: Array

    def __init__(
        self,
        physical_state: Any,
        /,
        *,
        auxiliary_state: Any = (),
        inference_state: Any = (),
        time: ArrayLike,
        accepted_step: ArrayLike,
        source_position: ArrayLike,
        random_key: ArrayLike,
        topology_epoch: ArrayLike,
    ):
        time_ = jnp.asarray(time)
        step = jnp.asarray(accepted_step, dtype=jnp.int64)
        source = jnp.asarray(source_position, dtype=jnp.int64)
        key = jnp.asarray(random_key, dtype=jnp.uint32)
        epoch = jnp.asarray(topology_epoch, dtype=jnp.int64)
        if (
            time_.shape != ()
            or step.shape != ()
            or source.shape != ()
            or epoch.shape != ()
        ):
            raise ValueError(
                "Continuation time, step, source position, and epoch must be scalar."
            )
        if key.shape != (2,):
            raise ValueError(
                "Continuation random key must be an explicit JAX key-data pair."
            )
        if (
            bool(~jnp.isfinite(time_))
            or bool(step < 0)
            or bool(source < 0)
            or bool(epoch < 0)
        ):
            raise ValueError("Continuation coordinates must be finite and nonnegative.")
        self.physical_state = physical_state
        self.auxiliary_state = auxiliary_state
        self.inference_state = inference_state
        self.time, self.accepted_step = time_, step
        self.source_position, self.random_key, self.topology_epoch = source, key, epoch


class GeophysicalCheckpointPlan(StrictModule, NonTrainableState):
    plan_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    resource_policy: GeophysicalResourcePolicy
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan_id: str,
        geometry_id: str,
        observation_id: str,
        partition_id: str,
        resource_policy: GeophysicalResourcePolicy,
        /,
    ):
        values = tuple(
            str(value).strip()
            for value in (plan_id, geometry_id, observation_id, partition_id)
        )
        if any(not value for value in values) or not isinstance(
            resource_policy, GeophysicalResourcePolicy
        ):
            raise ValueError("Checkpoint identities and resource policy are required.")
        self.plan_id, self.geometry_id, self.observation_id, self.partition_id = values
        self.resource_policy = resource_policy
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "geophysical-checkpoint-plan",
                "identities": values,
                "resource_policy": resource_policy.policy_id,
            }
        )

    def write(self, path: str | Path, state: GeophysicalContinuationState, /) -> Path:
        if not isinstance(state, GeophysicalContinuationState):
            raise TypeError("Checkpoint state must be GeophysicalContinuationState.")
        arrays: dict[str, object] = {}
        state_specification = pack_array_tree("continuation", state, arrays)
        estimated_bytes = sum(np.asarray(value).nbytes for value in arrays.values())
        if estimated_bytes > self.resource_policy.maximum_checkpoint_bytes:
            raise MemoryError("Checkpoint payload exceeds declared resource policy.")
        destination = write_array_archive(
            path,
            manifest={
                "kind": "geophysical-continuation-checkpoint",
                "checkpoint_plan_id": self.checkpoint_id,
                "plan_id": self.plan_id,
                "geometry_id": self.geometry_id,
                "observation_id": self.observation_id,
                "partition_id": self.partition_id,
                "continuation": state_specification,
            },
            arrays=arrays,
        )
        if destination.stat().st_size > self.resource_policy.maximum_checkpoint_bytes:
            destination.unlink()
            raise MemoryError("Serialized checkpoint exceeds declared resource policy.")
        return destination

    def read(
        self,
        path: str | Path,
        template: GeophysicalContinuationState,
        /,
    ) -> GeophysicalContinuationState:
        maximum = self.resource_policy.maximum_checkpoint_bytes
        manifest, arrays = read_array_archive(
            path,
            limits=ArrayArchiveLimits(
                max_container_bytes=maximum,
                max_aggregate_bytes=maximum,
                max_member_bytes=maximum,
                max_manifest_bytes=min(maximum, 1_048_576),
                max_central_directory_bytes=min(maximum, 1_048_576),
            ),
        )
        expected = {
            "kind": "geophysical-continuation-checkpoint",
            "checkpoint_plan_id": self.checkpoint_id,
            "plan_id": self.plan_id,
            "geometry_id": self.geometry_id,
            "observation_id": self.observation_id,
            "partition_id": self.partition_id,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Checkpoint identities do not match the prepared geophysical run."
            )
        state = unpack_array_tree(manifest["continuation"], arrays, template)
        if not isinstance(state, GeophysicalContinuationState):
            raise TypeError("Checkpoint continuation tree has the wrong type.")
        return state


__all__ = [
    "GeophysicalCheckpointPlan",
    "GeophysicalContinuationState",
    "GeophysicalResourcePolicy",
]
