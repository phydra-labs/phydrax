#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._validation import (
    boolean_array,
    content_id,
    integer_array,
    string_tuple,
)


class BiologicalGrouping(StrictModule, NonTrainableState):
    """Coarse-to-fine transitive biological groups for observation isolation."""

    observation_ids: Array
    group_ids: Array
    active: Array
    group_names: tuple[str, ...] = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    level_count: int = eqx.field(static=True)
    grouping_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation_ids: ArrayLike,
        group_ids: ArrayLike,
        /,
        *,
        group_names: Sequence[str],
        active: ArrayLike | None = None,
    ):
        observations = integer_array("observation_ids", observation_ids, ndim=1)
        groups = integer_array("group_ids", group_ids, ndim=2)
        observation_count = int(observations.shape[0])
        if groups.shape[0] != observation_count:
            raise ValueError("group_ids rows must match the number of observation IDs.")
        names = string_tuple("group_names", group_names, allow_empty=False)
        level_count = int(groups.shape[1])
        if level_count != len(names):
            raise ValueError(
                f"group_names must have length {level_count}; got {len(names)}."
            )
        active_ = boolean_array("active", active, (observation_count,), default=True)
        observation_host = np.asarray(observations)
        group_host = np.asarray(groups)
        active_host = np.asarray(active_)
        active_observations = observation_host[active_host]
        if np.any(active_observations < 0):
            raise ValueError("Active observation IDs must be non-negative.")
        if np.unique(active_observations).size != active_observations.size:
            raise ValueError("Active observation IDs must be unique.")
        if np.any(group_host[active_host] < 0):
            raise ValueError("Active observations require non-negative group IDs.")
        _validate_transitive_groups(group_host[active_host])

        grouping_id = content_id(
            "biological_grouping",
            {"group_names": names},
            (observations, groups, active_),
        )
        self.observation_ids = observations
        self.group_ids = groups
        self.active = active_
        self.group_names = names
        self.observation_count = observation_count
        self.level_count = level_count
        self.grouping_id = grouping_id


class BiologicalSplit(StrictModule, NonTrainableState):
    """Disjoint, exhaustive train/validation/test observation index partitions."""

    grouping: BiologicalGrouping
    train_indices: Array
    validation_indices: Array
    test_indices: Array
    split_id: str = eqx.field(static=True)

    def __init__(
        self,
        grouping: BiologicalGrouping,
        train_indices: ArrayLike,
        validation_indices: ArrayLike,
        test_indices: ArrayLike,
        /,
    ):
        if not isinstance(grouping, BiologicalGrouping):
            raise TypeError("grouping must be a BiologicalGrouping.")
        train = integer_array("train_indices", train_indices, ndim=1)
        validation = integer_array("validation_indices", validation_indices, ndim=1)
        test = integer_array("test_indices", test_indices, ndim=1)
        partitions = {
            "train_indices": np.asarray(train),
            "validation_indices": np.asarray(validation),
            "test_indices": np.asarray(test),
        }
        for name, indices in partitions.items():
            if indices.size and (
                np.any(indices < 0) or np.any(indices >= grouping.observation_count)
            ):
                raise ValueError(f"{name} lie outside the observation space.")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"{name} must not contain duplicates.")
        train_set = set(partitions["train_indices"].tolist())
        validation_set = set(partitions["validation_indices"].tolist())
        test_set = set(partitions["test_indices"].tolist())
        if (
            train_set & validation_set
            or train_set & test_set
            or validation_set & test_set
        ):
            raise ValueError("Biological split partitions must be disjoint.")
        assigned = train_set | validation_set | test_set
        active = set(np.flatnonzero(np.asarray(grouping.active)).tolist())
        if assigned != active:
            raise ValueError(
                "Biological split partitions must assign every active observation "
                "exactly once and no inactive observations."
            )

        split_id = content_id(
            "biological_split",
            {"grouping_id": grouping.grouping_id},
            (train, validation, test),
        )
        self.grouping = grouping
        self.train_indices = train
        self.validation_indices = validation
        self.test_indices = test
        self.split_id = split_id


class LeakageAudit(StrictModule, NonTrainableState):
    """Level-wise detection of biological groups crossing split partitions."""

    split: BiologicalSplit
    leaking_group_counts: Array
    leaking_observation_mask: Array
    has_leakage: Array
    passed: Array
    audit_id: str = eqx.field(static=True)

    def __init__(self, split: BiologicalSplit, /):
        if not isinstance(split, BiologicalSplit):
            raise TypeError("split must be a BiologicalSplit.")
        grouping = split.grouping
        partition = np.full(grouping.observation_count, -1, dtype=np.int32)
        partition[np.asarray(split.train_indices)] = 0
        partition[np.asarray(split.validation_indices)] = 1
        partition[np.asarray(split.test_indices)] = 2
        groups = np.asarray(grouping.group_ids)
        active = np.asarray(grouping.active)
        leaking_mask = np.zeros(groups.shape, dtype=bool)
        leaking_counts = np.zeros((grouping.level_count,), dtype=np.int32)
        for level in range(grouping.level_count):
            level_groups = groups[active, level]
            active_rows = np.flatnonzero(active)
            for group_id in np.unique(level_groups).tolist():
                members = active_rows[level_groups == group_id]
                if np.unique(partition[members]).size > 1:
                    leaking_mask[members, level] = True
                    leaking_counts[level] += 1
        counts = jnp.asarray(leaking_counts, dtype=jnp.int32)
        observation_mask = jnp.asarray(leaking_mask, dtype=bool)
        has_leakage = jnp.asarray(np.any(leaking_mask), dtype=bool)
        passed = jnp.logical_not(has_leakage)
        audit_id = content_id(
            "leakage_audit",
            {"split_id": split.split_id},
            (counts, observation_mask, has_leakage),
        )
        self.split = split
        self.leaking_group_counts = counts
        self.leaking_observation_mask = observation_mask
        self.has_leakage = has_leakage
        self.passed = passed
        self.audit_id = audit_id


def _validate_transitive_groups(groups: np.ndarray, /) -> None:
    for level in range(1, groups.shape[1]):
        child_parent: dict[int, int] = {}
        for parent, child in zip(
            groups[:, level - 1].tolist(),
            groups[:, level].tolist(),
            strict=True,
        ):
            prior = child_parent.setdefault(int(child), int(parent))
            if prior != int(parent):
                raise ValueError(
                    "Biological group levels must be transitively nested from "
                    "coarse to fine."
                )
