#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from ._connector import Connector


@dataclass(frozen=True, slots=True)
class ConnectionSet:
    connector_ids: tuple[str, ...]

    @classmethod
    def create(cls, ids):
        return cls(tuple(sorted(str(x) for x in ids)))

    def __post_init__(self):
        if len(self.connector_ids) < 2 or len(set(self.connector_ids)) != len(
            self.connector_ids
        ):
            raise ValueError("Connection set requires unique connectors.")


@dataclass(frozen=True, slots=True)
class AcausalSystem:
    connectors: tuple[Connector, ...]
    connections: tuple[ConnectionSet, ...]

    @classmethod
    def create(cls, connectors, connections):
        return cls(tuple(connectors), tuple(connections))

    @property
    def system_id(self):
        return canonical_fingerprint(
            {
                "kind": "acausal-system",
                "connectors": [x.connector_id for x in self.connectors],
                "connections": [x.connector_ids for x in self.connections],
            }
        )

    def connection_residual(self, values: ArrayLike, /):
        data = jnp.asarray(values)
        index = {x.connector_id: i for i, x in enumerate(self.connectors)}
        residual = []
        for group in self.connections:
            members = [self.connectors[index[x]] for x in group.connector_ids]
            reference = data[index[group.connector_ids[0]]]
            for variable_index, variable in enumerate(
                members[0].connector_type.variables
            ):
                if variable.kind == "across":
                    residual.extend(
                        data[index[x], variable_index] - reference[variable_index]
                        for x in group.connector_ids[1:]
                    )
                else:
                    residual.append(
                        sum(
                            member.orientation
                            * data[index[member.connector_id], variable_index]
                            for member in members
                        )
                    )
        return jnp.stack(residual) if residual else jnp.zeros((0,))


__all__ = ["AcausalSystem", "ConnectionSet"]
