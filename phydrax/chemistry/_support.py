#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact dependency graph over provider-neutral chemistry support tuples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import CapabilityProfile, SupportDependency, SupportTuple


class ChemistrySupportEdge(StrictModule, NonTrainableState):
    support_tuple_id: str = eqx.field(static=True)
    dependency_support_tuple_id: str = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(self, support_tuple_id: str, dependency_support_tuple_id: str, /):
        source = str(support_tuple_id).strip()
        dependency = str(dependency_support_tuple_id).strip()
        if not source or not dependency or source == dependency:
            raise ValueError(
                "Chemistry support edges require distinct non-empty tuple IDs."
            )
        self.support_tuple_id = source
        self.dependency_support_tuple_id = dependency
        self.edge_id = canonical_fingerprint(
            {
                "kind": "chemistry-support-edge",
                "support": source,
                "dependency": dependency,
            }
        )


class ChemistrySupportRegistry(StrictModule, NonTrainableState):
    support_tuples: tuple[SupportTuple, ...] = eqx.field(static=True)
    edges: tuple[ChemistrySupportEdge, ...] = eqx.field(static=True)
    registry_id: str = eqx.field(static=True)

    def __init__(
        self,
        support_tuples: Sequence[SupportTuple],
        edges: Sequence[ChemistrySupportEdge] = (),
        /,
    ):
        tuples = tuple(sorted(support_tuples, key=lambda value: value.support_tuple_id))
        edges_ = tuple(sorted(edges, key=lambda value: value.edge_id))
        if not tuples or any(not isinstance(value, SupportTuple) for value in tuples):
            raise TypeError("support_tuples must contain typed non-empty support.")
        if any(not isinstance(value, ChemistrySupportEdge) for value in edges_):
            raise TypeError("edges must contain ChemistrySupportEdge values.")
        identifiers = {value.support_tuple_id for value in tuples}
        if len(identifiers) != len(tuples):
            raise ValueError("Chemistry support tuples must be unique.")
        if any(
            edge.support_tuple_id not in identifiers
            or edge.dependency_support_tuple_id not in identifiers
            for edge in edges_
        ):
            raise ValueError("Chemistry support edges must reference registry tuples.")
        dependencies = {
            identifier: {
                edge.dependency_support_tuple_id
                for edge in edges_
                if edge.support_tuple_id == identifier
            }
            for identifier in identifiers
        }
        completed: set[str] = set()
        while len(completed) < len(identifiers):
            ready = {
                identifier
                for identifier in identifiers - completed
                if dependencies[identifier] <= completed
            }
            if not ready:
                raise ValueError("Chemistry support dependencies must be acyclic.")
            completed.update(ready)
        self.support_tuples = tuples
        self.edges = edges_
        self.registry_id = canonical_fingerprint(
            {
                "kind": "chemistry-support-registry",
                "support_tuples": [value.support_tuple_id for value in tuples],
                "edges": [value.edge_id for value in edges_],
            }
        )

    def dependencies(self, support: SupportTuple, /) -> tuple[SupportTuple, ...]:
        if not isinstance(support, SupportTuple):
            raise TypeError("support must be SupportTuple.")
        by_id = {value.support_tuple_id: value for value in self.support_tuples}
        if support.support_tuple_id not in by_id:
            raise ValueError("Support tuple is not registered.")
        identifiers = {
            edge.dependency_support_tuple_id
            for edge in self.edges
            if edge.support_tuple_id == support.support_tuple_id
        }
        return tuple(by_id[value] for value in sorted(identifiers))

    def dependency_records(
        self,
        support: SupportTuple,
        profiles_by_tuple: Mapping[str, CapabilityProfile],
        /,
    ) -> tuple[SupportDependency, ...]:
        records = []
        for dependency in self.dependencies(support):
            profile = profiles_by_tuple.get(dependency.support_tuple_id)
            if (
                profile is None
                or not profile.released
                or not profile.supports(dependency)
            ):
                raise ValueError(
                    "Every chemistry dependency requires an exact released capability profile."
                )
            records.append(
                SupportDependency(profile.profile_id, dependency.support_tuple_id)
            )
        return tuple(records)


__all__ = ["ChemistrySupportEdge", "ChemistrySupportRegistry"]
