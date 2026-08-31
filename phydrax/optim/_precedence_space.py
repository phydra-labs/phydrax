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
        available = []
        for operation in self.operations:
            if operation.operation_id in completed or operation.operation_id in skipped:
                continue
            if not set(operation.predecessors).issubset(completed):
                continue
            if operation.exclusive_group in selected_groups:
                continue
            demand = dict(operation.resource_demand)
            if any(demand.get(name, 0) > limit for name, limit in self.resource_limits):
                continue
            available.append(operation)
        return tuple(sorted(available, key=lambda value: value.operation_id))

    def branch(self, node: PrecedenceNode, /) -> tuple[PrecedenceNode, ...]:
        children = [
            PrecedenceNode(node.completed + (operation.operation_id,), node.skipped)
            for operation in self.available(node)
        ]
        optional = [
            value
            for value in self.operations
            if not value.mandatory
            and value.operation_id not in node.completed
            and value.operation_id not in node.skipped
            and set(value.predecessors).issubset(node.completed)
        ]
        children.extend(
            PrecedenceNode(node.completed, node.skipped + (operation.operation_id,))
            for operation in optional
        )
        return tuple(children)

    def complete(self, node: PrecedenceNode, /) -> bool:
        decided = set(node.completed) | set(node.skipped)
        return len(decided) == len(self.operations) and all(
            not value.mandatory or value.operation_id in node.completed
            for value in self.operations
        )


__all__ = ["PrecedenceNode", "PrecedenceOperation", "PrecedenceSpace"]
