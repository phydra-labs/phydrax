#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle
from ..discretization._core import nonempty_identifier, resolved_identifier


FidelityCoupling = Literal["shared", "nested", "independent"]


def _metadata_items(metadata: Mapping[str, str], /) -> tuple[tuple[str, str], ...]:
    items = tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
    if any(not key or not value for key, value in items):
        raise ValueError("Fidelity metadata keys and values must be non-empty strings.")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("Fidelity metadata keys must be unique.")
    return items


def _optional_identifier(name: str, value: str | None, /) -> str | None:
    return None if value is None else nonempty_identifier(name, value)


class FidelityLevelSpec(StrictModule, NonTrainableState):
    """One model or approximation level participating in a fidelity workflow."""

    discretization_bundle: DiscretizationBundle | None
    level_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    observable_contract_id: str = eqx.field(static=True)
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        level_id: str,
        /,
        *,
        problem_id: str,
        observable_id: str,
        model_id: str,
        approximation_id: str,
        observable_contract_id: str,
        discretization_bundle: DiscretizationBundle | None = None,
        metadata: Mapping[str, str] | None = None,
    ):
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        self.discretization_bundle = discretization_bundle
        self.level_id = nonempty_identifier("level_id", level_id)
        self.problem_id = nonempty_identifier("problem_id", problem_id)
        self.observable_id = nonempty_identifier("observable_id", observable_id)
        self.model_id = nonempty_identifier("model_id", model_id)
        self.approximation_id = nonempty_identifier("approximation_id", approximation_id)
        self.observable_contract_id = nonempty_identifier(
            "observable_contract_id", observable_contract_id
        )
        self.metadata = _metadata_items({} if metadata is None else metadata)

    @property
    def level_fingerprint(self) -> str:
        return resolved_identifier(
            "level_fingerprint",
            None,
            {
                "kind": "fidelity-level",
                "level": self.level_id,
                "problem": self.problem_id,
                "observable": self.observable_id,
                "model": self.model_id,
                "approximation": self.approximation_id,
                "observable_contract": self.observable_contract_id,
                "discretization_bundle": (
                    None
                    if self.discretization_bundle is None
                    else self.discretization_bundle.bundle_id
                ),
                "metadata": list(self.metadata),
            },
        )


class FidelityRelation(StrictModule, NonTrainableState):
    """Directed information or refinement relation between two fidelity levels."""

    source_level_id: str = eqx.field(static=True)
    target_level_id: str = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)
    state_transfer_id: str | None = eqx.field(static=True)
    observable_transfer_id: str | None = eqx.field(static=True)
    coupling_id: str | None = eqx.field(static=True)
    coupling: FidelityCoupling = eqx.field(static=True)
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        source_level_id: str,
        target_level_id: str,
        /,
        *,
        relation_id: str | None = None,
        state_transfer_id: str | None = None,
        observable_transfer_id: str | None = None,
        coupling_id: str | None = None,
        coupling: FidelityCoupling = "independent",
        metadata: Mapping[str, str] | None = None,
    ):
        source = nonempty_identifier("source_level_id", source_level_id)
        target = nonempty_identifier("target_level_id", target_level_id)
        if source == target:
            raise ValueError("A fidelity relation cannot connect a level to itself.")
        if coupling not in ("shared", "nested", "independent"):
            raise ValueError("coupling must be 'shared', 'nested', or 'independent'.")
        coupling_identifier = _optional_identifier("coupling_id", coupling_id)
        if coupling != "independent" and coupling_identifier is None:
            raise ValueError("Shared or nested fidelity coupling requires coupling_id.")
        self.source_level_id = source
        self.target_level_id = target
        self.state_transfer_id = _optional_identifier(
            "state_transfer_id", state_transfer_id
        )
        self.observable_transfer_id = _optional_identifier(
            "observable_transfer_id", observable_transfer_id
        )
        self.coupling_id = coupling_identifier
        self.coupling = coupling
        self.metadata = _metadata_items({} if metadata is None else metadata)
        self.relation_id = resolved_identifier(
            "relation_id",
            relation_id,
            {
                "kind": "fidelity-relation",
                "source": source,
                "target": target,
                "state_transfer": self.state_transfer_id,
                "observable_transfer": self.observable_transfer_id,
                "coupling": coupling,
                "coupling_id": coupling_identifier,
                "metadata": list(self.metadata),
            },
        )


class FidelityPath(StrictModule, NonTrainableState):
    """One unambiguous ordered path through a fidelity hierarchy."""

    levels: tuple[FidelityLevelSpec, ...]
    relations: tuple[FidelityRelation, ...]
    hierarchy_id: str = eqx.field(static=True)
    hierarchy_fingerprint: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[FidelityLevelSpec],
        relations: Sequence[FidelityRelation],
        /,
        *,
        hierarchy_id: str,
        hierarchy_fingerprint: str,
    ):
        levels_ = tuple(levels)
        relations_ = tuple(relations)
        if not levels_ or not all(
            isinstance(level, FidelityLevelSpec) for level in levels_
        ):
            raise TypeError("levels must contain one or more FidelityLevelSpec values.")
        if not all(isinstance(relation, FidelityRelation) for relation in relations_):
            raise TypeError("relations must contain FidelityRelation values.")
        if len(relations_) != len(levels_) - 1:
            raise ValueError("A fidelity path requires one relation between each level.")
        for index, relation in enumerate(relations_):
            if (
                relation.source_level_id != levels_[index].level_id
                or relation.target_level_id != levels_[index + 1].level_id
            ):
                raise ValueError("Fidelity path relations must connect adjacent levels.")
        self.levels = levels_
        self.relations = relations_
        self.hierarchy_id = nonempty_identifier("hierarchy_id", hierarchy_id)
        self.hierarchy_fingerprint = nonempty_identifier(
            "hierarchy_fingerprint", hierarchy_fingerprint
        )
        self.path_id = resolved_identifier(
            "path_id",
            None,
            {
                "kind": "fidelity-path",
                "hierarchy": self.hierarchy_fingerprint,
                "levels": [level.level_fingerprint for level in levels_],
                "relations": [relation.relation_id for relation in relations_],
            },
        )

    @property
    def fingerprint(self) -> str:
        return self.path_id

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def target(self) -> FidelityLevelSpec:
        return self.levels[-1]

    @property
    def level_ids(self) -> tuple[str, ...]:
        return tuple(level.level_id for level in self.levels)


class FidelityHierarchy(StrictModule, NonTrainableState):
    """Acyclic model-fidelity graph with one explicit authoritative target."""

    levels: tuple[FidelityLevelSpec, ...]
    relations: tuple[FidelityRelation, ...]
    target_level_id: str = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[FidelityLevelSpec],
        relations: Sequence[FidelityRelation] = (),
        /,
        *,
        target_level_id: str,
        hierarchy_id: str | None = None,
    ):
        levels_ = tuple(sorted(levels, key=lambda level: level.level_id))
        if not levels_ or not all(
            isinstance(level, FidelityLevelSpec) for level in levels_
        ):
            raise TypeError("levels must contain one or more FidelityLevelSpec values.")
        level_ids = tuple(level.level_id for level in levels_)
        if len(set(level_ids)) != len(level_ids):
            raise ValueError("Fidelity level IDs must be unique.")
        target = nonempty_identifier("target_level_id", target_level_id)
        if target not in set(level_ids):
            raise ValueError("target_level_id must identify a fidelity level.")
        reference = levels_[0]
        for level in levels_[1:]:
            if (
                level.problem_id != reference.problem_id
                or level.observable_id != reference.observable_id
                or level.observable_contract_id != reference.observable_contract_id
            ):
                raise ValueError(
                    "All fidelity levels must share problem, observable, and observable contract IDs."
                )
        relations_ = tuple(
            sorted(
                relations,
                key=lambda relation: (
                    relation.source_level_id,
                    relation.target_level_id,
                    relation.relation_id,
                ),
            )
        )
        if not all(isinstance(relation, FidelityRelation) for relation in relations_):
            raise TypeError("relations must contain FidelityRelation values.")
        relation_ids = tuple(relation.relation_id for relation in relations_)
        pairs = tuple(
            (relation.source_level_id, relation.target_level_id)
            for relation in relations_
        )
        if len(set(relation_ids)) != len(relation_ids) or len(set(pairs)) != len(pairs):
            raise ValueError(
                "Fidelity relations and directed level pairs must be unique."
            )
        known = set(level_ids)
        if any(
            relation.source_level_id not in known or relation.target_level_id not in known
            for relation in relations_
        ):
            raise ValueError("Every fidelity relation must reference known levels.")
        if any(relation.source_level_id == target for relation in relations_):
            raise ValueError(
                "The target fidelity level must not have outgoing relations."
            )
        self._validate_acyclic(level_ids, relations_)
        self._validate_target_reachability(level_ids, relations_, target)
        self.levels = levels_
        self.relations = relations_
        self.target_level_id = target
        payload = {
            "kind": "fidelity-hierarchy",
            "levels": [level.level_fingerprint for level in levels_],
            "relations": [relation.relation_id for relation in relations_],
            "target": target,
        }
        self.fingerprint = resolved_identifier("fingerprint", None, payload)
        self.hierarchy_id = resolved_identifier(
            "hierarchy_id",
            hierarchy_id,
            payload,
        )

    @staticmethod
    def _validate_acyclic(
        level_ids: tuple[str, ...],
        relations: tuple[FidelityRelation, ...],
        /,
    ) -> None:
        indegree = {level_id: 0 for level_id in level_ids}
        outgoing = {level_id: [] for level_id in level_ids}
        for relation in relations:
            indegree[relation.target_level_id] += 1
            outgoing[relation.source_level_id].append(relation.target_level_id)
        ready = [level_id for level_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            level_id = ready.pop()
            visited += 1
            for child in outgoing[level_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(level_ids):
            raise ValueError("Fidelity relations must form an acyclic graph.")

    @staticmethod
    def _validate_target_reachability(
        level_ids: tuple[str, ...],
        relations: tuple[FidelityRelation, ...],
        target: str,
        /,
    ) -> None:
        reverse = {level_id: [] for level_id in level_ids}
        for relation in relations:
            reverse[relation.target_level_id].append(relation.source_level_id)
        reachable = {target}
        pending = [target]
        while pending:
            child = pending.pop()
            for parent in reverse[child]:
                if parent not in reachable:
                    reachable.add(parent)
                    pending.append(parent)
        if reachable != set(level_ids):
            missing = tuple(sorted(set(level_ids) - reachable))
            raise ValueError(
                f"Every fidelity level must reach target {target!r}; disconnected levels: {missing}."
            )

    @property
    def target(self) -> FidelityLevelSpec:
        return self.level(self.target_level_id)

    @property
    def is_linear(self) -> bool:
        if len(self.relations) != len(self.levels) - 1:
            return False
        indegree = {level.level_id: 0 for level in self.levels}
        outdegree = {level.level_id: 0 for level in self.levels}
        for relation in self.relations:
            outdegree[relation.source_level_id] += 1
            indegree[relation.target_level_id] += 1
        return (
            sum(degree == 0 for degree in indegree.values()) == 1
            and all(degree <= 1 for degree in indegree.values())
            and all(degree <= 1 for degree in outdegree.values())
        )

    def level(self, level_id: str, /) -> FidelityLevelSpec:
        identifier = str(level_id)
        for level in self.levels:
            if level.level_id == identifier:
                return level
        raise KeyError(f"Unknown fidelity level {identifier!r}.")

    def parents(self, level_id: str, /) -> tuple[FidelityLevelSpec, ...]:
        identifier = self.level(level_id).level_id
        parent_ids = {
            relation.source_level_id
            for relation in self.relations
            if relation.target_level_id == identifier
        }
        return tuple(level for level in self.levels if level.level_id in parent_ids)

    def children(self, level_id: str, /) -> tuple[FidelityLevelSpec, ...]:
        identifier = self.level(level_id).level_id
        child_ids = {
            relation.target_level_id
            for relation in self.relations
            if relation.source_level_id == identifier
        }
        return tuple(level for level in self.levels if level.level_id in child_ids)

    def path(
        self,
        source_level_id: str,
        target_level_id: str | None = None,
        /,
    ) -> FidelityPath:
        source = self.level(source_level_id).level_id
        target = (
            self.target_level_id
            if target_level_id is None
            else self.level(target_level_id).level_id
        )
        outgoing: dict[str, tuple[FidelityRelation, ...]] = {
            level.level_id: tuple(
                relation
                for relation in self.relations
                if relation.source_level_id == level.level_id
            )
            for level in self.levels
        }
        matches: list[tuple[tuple[str, ...], tuple[FidelityRelation, ...]]] = []
        pending: list[tuple[str, tuple[str, ...], tuple[FidelityRelation, ...]]] = [
            (source, (source,), ())
        ]
        while pending and len(matches) < 2:
            current, level_ids, relations = pending.pop()
            if current == target:
                matches.append((level_ids, relations))
                continue
            for relation in outgoing[current]:
                pending.append(
                    (
                        relation.target_level_id,
                        (*level_ids, relation.target_level_id),
                        (*relations, relation),
                    )
                )
        if not matches:
            raise ValueError(f"No fidelity path connects {source!r} to {target!r}.")
        if len(matches) != 1:
            raise ValueError(f"Fidelity path from {source!r} to {target!r} is ambiguous.")
        level_ids, relations = matches[0]
        return FidelityPath(
            tuple(self.level(level_id) for level_id in level_ids),
            relations,
            hierarchy_id=self.hierarchy_id,
            hierarchy_fingerprint=self.fingerprint,
        )

    def linear_path(self) -> FidelityPath:
        if not self.is_linear:
            raise ValueError("Fidelity hierarchy is not linear.")
        child_ids = {relation.target_level_id for relation in self.relations}
        roots = tuple(
            level.level_id for level in self.levels if level.level_id not in child_ids
        )
        return self.path(roots[0])


__all__ = [
    "FidelityCoupling",
    "FidelityHierarchy",
    "FidelityLevelSpec",
    "FidelityPath",
    "FidelityRelation",
]
