#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import equinox as eqx
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
        if (
            len(self.connector_ids) < 2
            or any(
                not isinstance(value, str) or not value for value in self.connector_ids
            )
            or len(set(self.connector_ids)) != len(self.connector_ids)
            or self.connector_ids != tuple(sorted(self.connector_ids))
        ):
            raise ValueError(
                "Connection set requires unique nonempty connectors in canonical order."
            )


@dataclass(frozen=True, slots=True)
class AcausalSystem:
    connectors: tuple[Connector, ...]
    connections: tuple[ConnectionSet, ...]

    @classmethod
    def create(cls, connectors, connections):
        connector_values = tuple(connectors)
        connection_values = tuple(connections)
        return cls(connector_values, connection_values)

    def __post_init__(self):
        if any(not isinstance(item, Connector) for item in self.connectors):
            raise TypeError("Acausal systems require Connector entries.")
        if any(not isinstance(item, ConnectionSet) for item in self.connections):
            raise TypeError("Acausal systems require ConnectionSet entries.")
        connector_ids = tuple(item.connector_id for item in self.connectors)
        if not connector_ids or len(set(connector_ids)) != len(connector_ids):
            raise ValueError("Acausal connectors must have unique nonempty IDs.")
        known = set(connector_ids)
        connected: set[str] = set()
        lookup = {item.connector_id: item for item in self.connectors}
        for connection in self.connections:
            members = connection.connector_ids
            if any(member not in known for member in members):
                raise ValueError("Connection set references an unknown connector.")
            if any(member in connected for member in members):
                raise ValueError("A connector may belong to only one connection set.")
            connected.update(members)
            signatures = {
                tuple(
                    (variable.name, variable.kind, variable.unit)
                    for variable in lookup[member].connector_type.variables
                )
                for member in members
            }
            if len(signatures) != 1:
                raise ValueError("Connected connector types are not compatible.")

    @property
    def system_id(self):
        return canonical_fingerprint(
            {
                "kind": "acausal-system",
                "connectors": sorted(
                    (
                        {
                            "connector_id": connector.connector_id,
                            "connector_type_id": connector.connector_type.connector_type_id,
                            "variables": [
                                {
                                    "name": variable.name,
                                    "kind": variable.kind,
                                    "unit": variable.unit,
                                }
                                for variable in connector.connector_type.variables
                            ],
                            "orientation": connector.orientation,
                        }
                        for connector in self.connectors
                    ),
                    key=lambda item: item["connector_id"],
                ),
                "connections": sorted(item.connector_ids for item in self.connections),
            }
        )

    def connection_residual(self, values: ArrayLike, /):
        data = jnp.asarray(values)
        index = {x.connector_id: i for i, x in enumerate(self.connectors)}
        variable_counts = {
            len(connector.connector_type.variables) for connector in self.connectors
        }
        if len(variable_counts) != 1:
            raise ValueError(
                "Dense connection residuals require a common connector variable count."
            )
        expected_shape = (len(self.connectors), next(iter(variable_counts)))
        if data.shape != expected_shape:
            raise ValueError(
                f"Connection values must have shape {expected_shape}; got {data.shape}."
            )
        data = eqx.error_if(
            data,
            jnp.any(~jnp.isfinite(data)),
            "Connection values must be finite.",
        )
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
