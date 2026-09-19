#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed acausal connectors, components, flattening, and structural analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..qualification import CapabilityProfile, SupportTuple


VariableKind: TypeAlias = Literal["across", "through"]


@dataclass(frozen=True, slots=True)
class ConnectorVariable:
    name: str
    kind: VariableKind
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit or self.kind not in ("across", "through"):
            raise ValueError(
                "Connector variable requires canonical name, kind, and unit."
            )


@dataclass(frozen=True, slots=True)
class ConnectorType:
    connector_type_id: str
    variables: tuple[ConnectorVariable, ...]

    @classmethod
    def create(cls, connector_type_id: str, variables):
        return cls(str(connector_type_id), tuple(variables))

    def __post_init__(self) -> None:
        if not self.connector_type_id or not self.variables:
            raise ValueError("Connector type identity and variables are required.")
        names = tuple(value.name for value in self.variables)
        if len(set(names)) != len(names):
            raise ValueError("Connector variable names must be unique.")


@dataclass(frozen=True, slots=True)
class Connector:
    connector_id: str
    connector_type: ConnectorType


@dataclass(frozen=True, slots=True)
class ConnectionSet:
    connector_ids: tuple[str, ...]

    @classmethod
    def create(cls, connector_ids):
        return cls(tuple(sorted(str(value) for value in connector_ids)))

    def __post_init__(self) -> None:
        if len(self.connector_ids) < 2 or len(set(self.connector_ids)) != len(
            self.connector_ids
        ):
            raise ValueError("Connection sets require at least two unique connectors.")


@dataclass(frozen=True, slots=True)
class AcausalSystem:
    connectors: tuple[Connector, ...]
    connections: tuple[ConnectionSet, ...]

    @classmethod
    def create(cls, connectors, connections):
        return cls(tuple(connectors), tuple(connections))

    def __post_init__(self) -> None:
        identities = tuple(value.connector_id for value in self.connectors)
        if len(set(identities)) != len(identities):
            raise ValueError("Connector identities must be unique.")
        known = set(identities)
        if any(set(group.connector_ids).difference(known) for group in self.connections):
            raise ValueError("Connection set references an unknown connector.")

    @property
    def system_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "acausal-system",
                "connectors": [
                    (value.connector_id, value.connector_type.connector_type_id)
                    for value in self.connectors
                ],
                "connections": [value.connector_ids for value in self.connections],
            }
        )

    def connection_residual(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != len(self.connectors):
            raise ValueError("Connector values must begin with the connector axis.")
        index = {
            value.connector_id: position for position, value in enumerate(self.connectors)
        }
        residuals = []
        for group in self.connections:
            positions = tuple(index[value] for value in group.connector_ids)
            reference = array[positions[0]]
            for position in positions[1:]:
                residuals.append(array[position] - reference)
        return jnp.stack(residuals) if residuals else jnp.zeros((0, *array.shape[1:]))


def structural_incidence(equation_variables: ArrayLike, /) -> tuple[Array, int]:
    incidence = np.asarray(equation_variables, dtype=bool)
    if incidence.ndim != 2:
        raise ValueError("Structural incidence must be a matrix.")
    rank = int(np.linalg.matrix_rank(incidence.astype(float)))
    return jnp.asarray(incidence), rank


def system_modeling_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "system-modeling.acausal-connectors",
            {"variables": "across-through", "connections": "equality"},
        ),
        ("system-modeling.structural-incidence", {"analysis": "boolean-rank"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=(
                "connection-conservation",
                "structural-analysis",
                "public-workflow",
            ),
        )
        for name, attrs in specs
    )


__all__ = [
    "AcausalSystem",
    "ConnectionSet",
    "Connector",
    "ConnectorType",
    "ConnectorVariable",
    "VariableKind",
    "structural_incidence",
    "system_modeling_candidate_profiles",
]
