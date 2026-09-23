#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState


class PrecedenceOperation(StrictModule, NonTrainableState):
    operation_id: str = eqx.field(static=True)
    predecessors: tuple[str, ...] = eqx.field(static=True)
    exclusive_group: str | None = eqx.field(static=True)
    simultaneous_group: str | None = eqx.field(static=True)
    resource_demand: tuple[tuple[str, int], ...] = eqx.field(static=True)
    mandatory: bool = eqx.field(static=True)

    def __init__(
        self,
        operation_id: str,
        /,
        *,
        predecessors: Sequence[str] = (),
        exclusive_group: str | None = None,
        simultaneous_group: str | None = None,
        resource_demand: Mapping[str, int] | None = None,
        mandatory: bool = True,
    ):
        identifier = str(operation_id)
        if not identifier:
            raise ValueError("operation_id must be nonempty.")
        demand = tuple(
            sorted(
                (str(key), int(value)) for key, value in (resource_demand or {}).items()
            )
        )
        if any(value < 0 for _, value in demand):
            raise ValueError("Resource demand must be nonnegative.")
        self.operation_id = identifier
        self.predecessors = tuple(str(value) for value in predecessors)
        self.exclusive_group = None if exclusive_group is None else str(exclusive_group)
        self.simultaneous_group = (
            None if simultaneous_group is None else str(simultaneous_group)
        )
        self.resource_demand = demand
        self.mandatory = bool(mandatory)


class PrecedenceNode(StrictModule, NonTrainableState):
    completed: tuple[str, ...] = eqx.field(static=True)
    skipped: tuple[str, ...] = eqx.field(static=True)

    @property
    def node_id(self) -> str:
        return "|".join(self.completed) + "::" + "|".join(self.skipped)


class PrecedenceSpace(StrictModule, NonTrainableState):
    operations: tuple[PrecedenceOperation, ...]
    resource_limits: tuple[tuple[str, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        operations: Sequence[PrecedenceOperation],
        /,
        *,
        resource_limits: Mapping[str, int] | None = None,
    ):
        operations_ = tuple(operations)
        identifiers = {value.operation_id for value in operations_}
        if not operations_ or len(identifiers) != len(operations_):
            raise ValueError("Operations must be nonempty with unique IDs.")
        for operation in operations_:
            if any(value not in identifiers for value in operation.predecessors):
                raise ValueError("Every predecessor must identify a declared operation.")
            if operation.operation_id in operation.predecessors:
                raise ValueError("An operation may not precede itself.")
        simultaneous_groups: dict[str, list[PrecedenceOperation]] = {}
        for operation in operations_:
            if operation.simultaneous_group is not None:
                simultaneous_groups.setdefault(operation.simultaneous_group, []).append(
                    operation
                )
        for group in simultaneous_groups.values():
            members = {operation.operation_id for operation in group}
            if any(members.intersection(operation.predecessors) for operation in group):
                raise ValueError(
                    "Simultaneous operations may not depend on another group member."
                )
            if len({operation.mandatory for operation in group}) != 1:
                raise ValueError(
                    "A simultaneous group must have one shared mandatory policy."
                )
            exclusive = [
                operation.exclusive_group
                for operation in group
                if operation.exclusive_group is not None
            ]
            if len(exclusive) != len(set(exclusive)):
                raise ValueError(
                    "Simultaneous operations may not share an exclusive group."
                )
        limits = tuple(
            sorted(
                (str(key), int(value)) for key, value in (resource_limits or {}).items()
            )
        )
        if any(value < 0 for _, value in limits):
            raise ValueError("Resource limits must be nonnegative.")
        self.operations = operations_
        self.resource_limits = limits
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        completed: set[str] = set()
        while len(completed) < len(self.operations):
            available = [
                value.operation_id
                for value in self.operations
                if value.operation_id not in completed
                and set(value.predecessors).issubset(completed)
            ]
            if not available:
                raise ValueError("Precedence operations contain a cycle.")
            completed.update(available)

    def root(self, /) -> PrecedenceNode:
        return PrecedenceNode((), ())

    def available(self, node: PrecedenceNode, /) -> tuple[PrecedenceOperation, ...]:
        completed = set(node.completed)
        skipped = set(node.skipped)
        selected_groups = {
            value.exclusive_group
            for value in self.operations
            if value.operation_id in completed and value.exclusive_group is not None
        }
        undecided = tuple(
            operation
            for operation in self.operations
            if operation.operation_id not in completed
            and operation.operation_id not in skipped
        )
        ready = tuple(
            operation
            for operation in undecided
            if set(operation.predecessors).issubset(completed)
            and operation.exclusive_group not in selected_groups
        )
        ready_ids = {operation.operation_id for operation in ready}
        limits = dict(self.resource_limits)
        available = []
        for operation in ready:
            group = (
                (operation,)
                if operation.simultaneous_group is None
                else tuple(
                    candidate
                    for candidate in undecided
                    if candidate.simultaneous_group == operation.simultaneous_group
                )
            )
            if any(candidate.operation_id not in ready_ids for candidate in group):
                continue
            demand: dict[str, int] = {}
            for candidate in group:
                for name, value in candidate.resource_demand:
                    demand[name] = demand.get(name, 0) + value
            if any(demand.get(name, 0) > limit for name, limit in limits.items()):
                continue
            available.append(operation)
        return tuple(sorted(available, key=lambda value: value.operation_id))

    def branch(self, node: PrecedenceNode, /) -> tuple[PrecedenceNode, ...]:
        available = self.available(node)
        children = []
        emitted: set[str] = set()
        for operation in available:
            key = (
                operation.operation_id
                if operation.simultaneous_group is None
                else operation.simultaneous_group
            )
            if key in emitted:
                continue
            emitted.add(key)
            group = (
                (operation,)
                if operation.simultaneous_group is None
                else tuple(
                    candidate
                    for candidate in available
                    if candidate.simultaneous_group == operation.simultaneous_group
                )
            )
            completed = node.completed + tuple(
                candidate.operation_id
                for candidate in sorted(group, key=lambda value: value.operation_id)
            )
            children.append(PrecedenceNode(completed, node.skipped))
        optional = [value for value in available if not value.mandatory]
        emitted.clear()
        for operation in optional:
            key = (
                operation.operation_id
                if operation.simultaneous_group is None
                else operation.simultaneous_group
            )
            if key in emitted:
                continue
            emitted.add(key)
            group = (
                (operation,)
                if operation.simultaneous_group is None
                else tuple(
                    candidate
                    for candidate in optional
                    if candidate.simultaneous_group == operation.simultaneous_group
                )
            )
            skipped = node.skipped + tuple(
                candidate.operation_id
                for candidate in sorted(group, key=lambda value: value.operation_id)
            )
            children.append(PrecedenceNode(node.completed, skipped))
        return tuple(children)

    def complete(self, node: PrecedenceNode, /) -> bool:
        decided = set(node.completed) | set(node.skipped)
        return len(decided) == len(self.operations) and all(
            not value.mandatory or value.operation_id in node.completed
            for value in self.operations
        )


__all__ = ["PrecedenceNode", "PrecedenceOperation", "PrecedenceSpace"]
