#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


PartitionName: TypeAlias = Literal["train", "validation", "test"]


def _identifiers(name: str, values: Sequence[str], /) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)
    if not identifiers or any(not value for value in identifiers):
        raise ValueError(f"{name} must contain non-empty identifiers.")
    return identifiers


class CasePartitionManifest(StrictModule, NonTrainableState):
    """Content-addressed train, validation, and test case membership."""

    case_ids: tuple[str, ...] = eqx.field(static=True)
    group_ids: tuple[str, ...] = eqx.field(static=True)
    train_ids: tuple[str, ...] = eqx.field(static=True)
    validation_ids: tuple[str, ...] = eqx.field(static=True)
    test_ids: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        case_ids: Sequence[str],
        group_ids: Sequence[str],
        /,
        *,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        test_ids: Sequence[str],
        policy_id: str,
        source_id: str,
    ):
        cases = _identifiers("case_ids", case_ids)
        groups = _identifiers("group_ids", group_ids)
        if len(cases) != len(groups):
            raise ValueError("group_ids must assign one group to every case.")
        if len(set(cases)) != len(cases):
            raise ValueError("case_ids must be unique.")
        train = _identifiers("train_ids", train_ids)
        validation = _identifiers("validation_ids", validation_ids)
        test = _identifiers("test_ids", test_ids)
        for name, values in (
            ("train_ids", train),
            ("validation_ids", validation),
            ("test_ids", test),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")
        partitions = (set(train), set(validation), set(test))
        if (
            partitions[0] & partitions[1]
            or partitions[0] & partitions[2]
            or partitions[1] & partitions[2]
        ):
            raise ValueError("Train, validation, and test case IDs must be disjoint.")
        known = set(cases)
        assigned = partitions[0] | partitions[1] | partitions[2]
        if assigned != known:
            missing = tuple(sorted(known - assigned))
            unknown = tuple(sorted(assigned - known))
            raise ValueError(
                "Partitions must assign every case exactly once; "
                f"missing={missing}, unknown={unknown}."
            )
        membership = {
            case_id: partition
            for partition, values in zip(
                ("train", "validation", "test"), (train, validation, test), strict=True
            )
            for case_id in values
        }
        group_partitions: dict[str, str] = {}
        for case_id, group_id in zip(cases, groups, strict=True):
            partition = membership[case_id]
            previous = group_partitions.setdefault(group_id, partition)
            if previous != partition:
                raise ValueError(
                    f"Split group {group_id!r} crosses {previous!r} and {partition!r}."
                )
        policy = str(policy_id)
        source = str(source_id)
        if not policy or not source:
            raise ValueError("policy_id and source_id must be non-empty strings.")
        self.case_ids = cases
        self.group_ids = groups
        self.train_ids = train
        self.validation_ids = validation
        self.test_ids = test
        self.policy_id = policy
        self.source_id = source
        self.partition_id = f"case-partition:{canonical_fingerprint(self.to_dict())}"

    @classmethod
    def from_indices(
        cls,
        case_ids: Sequence[str],
        group_ids: Sequence[str],
        /,
        *,
        train_indices: Sequence[int],
        validation_indices: Sequence[int],
        test_indices: Sequence[int],
        policy_id: str,
        source_id: str,
    ) -> CasePartitionManifest:
        cases = tuple(str(case_id) for case_id in case_ids)

        def select(indices: Sequence[int], name: str) -> tuple[str, ...]:
            positions = tuple(int(index) for index in indices)
            if any(index < 0 or index >= len(cases) for index in positions):
                raise IndexError(f"{name} contains an out-of-range case index.")
            return tuple(cases[index] for index in positions)

        return cls(
            cases,
            group_ids,
            train_ids=select(train_indices, "train_indices"),
            validation_ids=select(validation_indices, "validation_indices"),
            test_ids=select(test_indices, "test_indices"),
            policy_id=policy_id,
            source_id=source_id,
        )

    def subset_ids(self, partition: PartitionName, /) -> tuple[str, ...]:
        if partition == "train":
            return self.train_ids
        if partition == "validation":
            return self.validation_ids
        if partition == "test":
            return self.test_ids
        raise ValueError("partition must be 'train', 'validation', or 'test'.")

    def partition(self, case_id: str, /) -> PartitionName:
        identifier = str(case_id)
        if identifier in self.train_ids:
            return "train"
        if identifier in self.validation_ids:
            return "validation"
        if identifier in self.test_ids:
            return "test"
        raise KeyError(f"Unknown partition case {identifier!r}.")

    def indices(
        self,
        ordered_case_ids: Sequence[str],
        partition: PartitionName,
        /,
    ) -> tuple[int, ...]:
        ordered = tuple(str(case_id) for case_id in ordered_case_ids)
        if len(set(ordered)) != len(ordered):
            raise ValueError("ordered_case_ids must be unique.")
        if set(ordered) != set(self.case_ids):
            raise ValueError("ordered_case_ids must match the manifest case membership.")
        positions = {case_id: index for index, case_id in enumerate(ordered)}
        return tuple(positions[case_id] for case_id in self.subset_ids(partition))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_ids": list(self.case_ids),
            "group_ids": list(self.group_ids),
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "policy_id": self.policy_id,
            "source_id": self.source_id,
        }


__all__ = ["CasePartitionManifest", "PartitionName"]
