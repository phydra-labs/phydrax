#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ._piv import PIVSyntheticCase
    from ._ptv import PTVSyntheticCase

    SyntheticCase: TypeAlias = PIVSyntheticCase | PTVSyntheticCase
else:
    SyntheticCase: TypeAlias = object


class ScenarioSplitPolicy(StrictModule, NonTrainableState):
    """Deterministic family-exclusive train/validation/test allocation policy."""

    train_fraction: float = eqx.field(static=True)
    validation_fraction: float = eqx.field(static=True)
    test_fraction: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        train_fraction: float = 0.6,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ):
        fractions = np.asarray(
            (train_fraction, validation_fraction, test_fraction), dtype=float
        )
        if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
            raise ValueError("All scenario split fractions must be finite and positive.")
        if not np.isclose(float(np.sum(fractions)), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("Scenario split fractions must sum to one.")
        seed_ = int(seed)
        self.train_fraction = float(fractions[0])
        self.validation_fraction = float(fractions[1])
        self.test_fraction = float(fractions[2])
        self.seed = seed_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "synthetic-scenario-split-policy",
                "fractions": fractions.tolist(),
                "seed": seed_,
            }
        )


class SyntheticScenarioSplit(StrictModule, NonTrainableState):
    """Leakage-safe sample indices with disjoint scenario-family evidence."""

    train_indices: Array
    validation_indices: Array
    test_indices: Array
    train_families: tuple[str, ...] = eqx.field(static=True)
    validation_families: tuple[str, ...] = eqx.field(static=True)
    test_families: tuple[str, ...] = eqx.field(static=True)
    scenario_ids: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    split_id: str = eqx.field(static=True)

    def __init__(
        self,
        train_indices: ArrayLike,
        validation_indices: ArrayLike,
        test_indices: ArrayLike,
        /,
        *,
        train_families: Sequence[str],
        validation_families: Sequence[str],
        test_families: Sequence[str],
        scenario_ids: Sequence[str],
        policy_id: str,
    ):
        indices = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (train_indices, validation_indices, test_indices)
        )
        if any(value.ndim != 1 or value.size == 0 for value in indices):
            raise ValueError("Every scenario split must be a non-empty index vector.")
        if any(bool(jnp.any(value < 0)) for value in indices):
            raise ValueError("Scenario split indices must be non-negative.")
        if any(int(jnp.unique(value).size) != int(value.size) for value in indices):
            raise ValueError("A scenario split must not contain duplicate indices.")
        if any(
            bool(jnp.any(jnp.isin(indices[left], indices[right])))
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("Train, validation, and test indices must not overlap.")

        families = tuple(
            tuple(str(item) for item in values)
            for values in (train_families, validation_families, test_families)
        )
        if any(not values or any(not item for item in values) for values in families):
            raise ValueError(
                "Every scenario split requires non-empty family identifiers."
            )
        if any(len(set(values)) != len(values) for values in families):
            raise ValueError("A scenario split must not repeat family identifiers.")
        family_sets = tuple(set(values) for values in families)
        if any(
            family_sets[left].intersection(family_sets[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("Scenario families must not overlap across splits.")

        ids = tuple(str(value) for value in scenario_ids)
        if not ids or any(not value for value in ids):
            raise ValueError("scenario_ids must be non-empty identifiers.")
        if len(set(ids)) != len(ids):
            raise ValueError("scenario_ids must be unique to prevent split leakage.")
        total = sum(int(value.size) for value in indices)
        if total != len(ids):
            raise ValueError("Split indices must cover every scenario exactly once.")
        combined = jnp.concatenate(indices)
        if int(jnp.unique(combined).size) != len(ids) or bool(
            jnp.any(combined >= len(ids))
        ):
            raise ValueError("Split indices must form the complete scenario index set.")
        policy = str(policy_id)
        if not policy:
            raise ValueError("policy_id must be non-empty.")

        self.train_indices, self.validation_indices, self.test_indices = indices
        self.train_families, self.validation_families, self.test_families = families
        self.scenario_ids = ids
        self.policy_id = policy
        self.split_id = canonical_fingerprint(
            {
                "kind": "synthetic-scenario-split",
                "policy": policy,
                "scenario_ids": list(ids),
                "families": [list(values) for values in families],
                "indices": [np.asarray(value).tolist() for value in indices],
            }
        )


def _group_counts(
    group_count: int, fractions: tuple[float, float, float], /
) -> list[int]:
    counts = [1, 1, 1]
    targets = [group_count * value for value in fractions]
    for _ in range(group_count - 3):
        split_index = max(
            range(3),
            key=lambda index: (targets[index] - counts[index], -index),
        )
        counts[split_index] += 1
    return counts


def split_synthetic_scenarios(
    scenarios: Sequence[SyntheticCase],
    policy: ScenarioSplitPolicy,
    /,
) -> SyntheticScenarioSplit:
    """Split cases without placing one scenario family in multiple partitions."""
    if not isinstance(policy, ScenarioSplitPolicy):
        raise TypeError("policy must be a ScenarioSplitPolicy.")
    cases = tuple(scenarios)
    if not cases:
        raise ValueError("At least one synthetic scenario is required.")
    scenario_ids = tuple(case.scenario_id for case in cases)
    if any(not value for value in scenario_ids) or len(set(scenario_ids)) != len(
        scenario_ids
    ):
        raise ValueError("Synthetic scenario IDs must be non-empty and unique.")

    family_to_indices: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        family = str(case.family_id)
        if not family:
            raise ValueError("Synthetic scenario families must be non-empty.")
        family_to_indices.setdefault(family, []).append(index)
    if len(family_to_indices) < 3:
        raise ValueError(
            "Leakage-safe train/validation/test splitting requires at least three families."
        )

    ordered_families = sorted(
        family_to_indices,
        key=lambda family: canonical_fingerprint(
            {
                "kind": "synthetic-scenario-family-order",
                "policy": policy.policy_id,
                "family": family,
            }
        ),
    )
    counts = _group_counts(
        len(ordered_families),
        (policy.train_fraction, policy.validation_fraction, policy.test_fraction),
    )
    family_partitions: list[tuple[str, ...]] = []
    start = 0
    for count in counts:
        family_partitions.append(tuple(ordered_families[start : start + count]))
        start += count
    index_partitions = tuple(
        jnp.asarray(
            sorted(index for family in families for index in family_to_indices[family]),
            dtype=jnp.int32,
        )
        for families in family_partitions
    )
    return SyntheticScenarioSplit(
        *index_partitions,
        train_families=family_partitions[0],
        validation_families=family_partitions[1],
        test_families=family_partitions[2],
        scenario_ids=scenario_ids,
        policy_id=policy.policy_id,
    )


__all__ = [
    "ScenarioSplitPolicy",
    "SyntheticScenarioSplit",
    "split_synthetic_scenarios",
]
