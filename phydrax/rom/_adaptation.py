#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ROMGeneration(StrictModule, NonTrainableState):
    generation: int = eqx.field(static=True)
    parent_generation_id: str | None = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    hyperreduction_id: str | None = eqx.field(static=True)
    certification_id: str | None = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    qualification_ids: tuple[str, ...] = eqx.field(static=True)
    generation_id: str = eqx.field(static=True)

    def __init__(
        self,
        generation: int,
        /,
        *,
        parent_generation_id: str | None,
        representation_id: str,
        dynamics_id: str,
        partition_id: str,
        support_id: str,
        qualification_ids: Sequence[str],
        hyperreduction_id: str | None = None,
        certification_id: str | None = None,
    ):
        number = int(generation)
        identifiers = tuple(
            str(value)
            for value in (representation_id, dynamics_id, partition_id, support_id)
        )
        qualifications = tuple(str(value) for value in qualification_ids)
        parent = None if parent_generation_id is None else str(parent_generation_id)
        hyper = None if hyperreduction_id is None else str(hyperreduction_id)
        certificate = None if certification_id is None else str(certification_id)
        if number < 0 or any(not value for value in (*identifiers, *qualifications)):
            raise ValueError("ROM generation identities are invalid.")
        if number == 0 and parent is not None or number > 0 and not parent:
            raise ValueError("Only generation zero may omit a parent generation.")
        self.generation = number
        self.parent_generation_id = parent
        self.representation_id = identifiers[0]
        self.dynamics_id = identifiers[1]
        self.hyperreduction_id = hyper
        self.certification_id = certificate
        self.partition_id = identifiers[2]
        self.support_id = identifiers[3]
        self.qualification_ids = qualifications
        self.generation_id = canonical_fingerprint(
            {
                "kind": "rom-generation",
                "generation": number,
                "parent": parent,
                "representation": identifiers[0],
                "dynamics": identifiers[1],
                "hyperreduction": hyper,
                "certification": certificate,
                "partition": identifiers[2],
                "support": identifiers[3],
                "qualification": list(qualifications),
            }
        )


class ActiveLearningPlan(StrictModule, NonTrainableState):
    maximum_acquisitions: int = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    cost_weight: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_acquisitions: int,
        /,
        *,
        minimum_separation: float = 0.0,
        cost_weight: float = 0.0,
    ):
        maximum = int(maximum_acquisitions)
        separation = float(minimum_separation)
        cost = float(cost_weight)
        if (
            maximum <= 0
            or separation < 0.0
            or cost < 0.0
            or not np.isfinite(separation + cost)
        ):
            raise ValueError("Active-learning plan values are invalid.")
        self.maximum_acquisitions = maximum
        self.minimum_separation = separation
        self.cost_weight = cost
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rom-active-learning-plan",
                "maximum": maximum,
                "separation": separation,
                "cost_weight": cost,
            }
        )

    def select(
        self,
        candidates: ArrayLike,
        scores: ArrayLike,
        costs: ArrayLike,
        /,
    ) -> Array:
        values = np.asarray(candidates)
        score = np.asarray(scores)
        cost = np.asarray(costs)
        if (
            values.ndim != 2
            or score.shape != (values.shape[0],)
            or cost.shape != score.shape
        ):
            raise ValueError(
                "Active-learning candidates, scores, and costs are misaligned."
            )
        if (
            np.any(~np.isfinite(values))
            or np.any(~np.isfinite(score))
            or np.any(~np.isfinite(cost))
            or np.any(cost < 0.0)
        ):
            raise ValueError(
                "Active-learning inputs must be finite with nonnegative cost."
            )
        utility = score - self.cost_weight * cost
        order = np.argsort(utility)[::-1]
        selected: list[int] = []
        for index in order:
            if len(selected) >= self.maximum_acquisitions:
                break
            if all(
                np.linalg.norm(values[index] - values[other]) >= self.minimum_separation
                for other in selected
            ):
                selected.append(int(index))
        return jnp.asarray(selected, dtype=jnp.int32)


class EnrichmentTransaction(StrictModule, NonTrainableState):
    parent: ROMGeneration
    child: ROMGeneration
    truth_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    replay_artifact_id: str = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent: ROMGeneration,
        child: ROMGeneration,
        /,
        *,
        truth_artifact_ids: Sequence[str],
        replay_artifact_id: str,
    ):
        if not isinstance(parent, ROMGeneration) or not isinstance(child, ROMGeneration):
            raise TypeError("parent and child must be ROMGeneration values.")
        truth = tuple(str(value) for value in truth_artifact_ids)
        replay = str(replay_artifact_id)
        if (
            child.parent_generation_id != parent.generation_id
            or child.generation != parent.generation + 1
        ):
            raise ValueError("Child generation does not directly descend from parent.")
        if not truth or any(not value for value in truth) or not replay:
            raise ValueError("Enrichment truth and replay artifacts must be non-empty.")
        self.parent = parent
        self.child = child
        self.truth_artifact_ids = truth
        self.replay_artifact_id = replay
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "rom-enrichment-transaction",
                "parent": parent.generation_id,
                "child": child.generation_id,
                "truth": list(truth),
                "replay": replay,
            }
        )


class DistributedBasisArtifact(StrictModule, NonTrainableState):
    local_basis: Array
    global_dimension: int = eqx.field(static=True)
    local_offset: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_basis: ArrayLike,
        /,
        *,
        global_dimension: int,
        local_offset: int,
        axis_name: str,
        partition_id: str,
    ):
        basis = jnp.asarray(local_basis)
        global_size = int(global_dimension)
        offset = int(local_offset)
        axis = str(axis_name)
        partition = str(partition_id)
        if (
            basis.ndim != 2
            or global_size <= 0
            or offset < 0
            or offset + basis.shape[0] > global_size
            or not axis
            or not partition
        ):
            raise ValueError("Distributed basis partition is invalid.")
        self.local_basis = basis
        self.global_dimension = global_size
        self.local_offset = offset
        self.axis_name = axis
        self.partition_id = partition
        self.basis_id = canonical_fingerprint(
            {
                "kind": "distributed-basis",
                "global_dimension": global_size,
                "offset": offset,
                "axis": axis,
                "partition": partition,
                "content": array_tree_fingerprint(basis)["sha256"],
            }
        )

    def project(self, local_state: ArrayLike, /) -> Array:
        state = jnp.asarray(local_state)
        if state.shape != (self.local_basis.shape[0],):
            raise ValueError(
                "Local state shape does not match the distributed basis partition."
            )
        local = jnp.conj(self.local_basis.T) @ state
        return jax.lax.psum(local, self.axis_name)

    def reconstruct_local(self, reduced_state: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced_state)
        if value.shape != (self.local_basis.shape[1],):
            raise ValueError("Reduced state shape does not match distributed basis rank.")
        return self.local_basis @ value


class StreamingCorrelationAccumulator(StrictModule, NonTrainableState):
    correlation: Array
    sample_count: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, correlation: ArrayLike, sample_count: int, /, *, source_id: str):
        value = jnp.asarray(correlation)
        count = int(sample_count)
        source = str(source_id)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or count < 0 or not source:
            raise ValueError("Streaming correlation state is invalid.")
        self.correlation = value
        self.sample_count = jnp.asarray(count, dtype=jnp.int32)
        self.source_id = source

    def merge(
        self, other: StreamingCorrelationAccumulator, /
    ) -> StreamingCorrelationAccumulator:
        if (
            not isinstance(other, StreamingCorrelationAccumulator)
            or other.source_id != self.source_id
            or other.correlation.shape != self.correlation.shape
        ):
            raise ValueError("Streaming correlation accumulators are incompatible.")
        return StreamingCorrelationAccumulator(
            self.correlation + other.correlation,
            int(np.asarray(self.sample_count + other.sample_count)),
            source_id=self.source_id,
        )


__all__ = [
    "ActiveLearningPlan",
    "DistributedBasisArtifact",
    "EnrichmentTransaction",
    "ROMGeneration",
    "StreamingCorrelationAccumulator",
]
