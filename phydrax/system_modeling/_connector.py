#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias


VariableKind: TypeAlias = Literal["across", "through"]


@dataclass(frozen=True, slots=True)
class ConnectorVariable:
    name: str
    kind: VariableKind
    unit: str

    def __post_init__(self):
        if not self.name or not self.unit or self.kind not in ("across", "through"):
            raise ValueError("Connector variable invalid.")


@dataclass(frozen=True, slots=True)
class ConnectorType:
    connector_type_id: str
    variables: tuple[ConnectorVariable, ...]

    @classmethod
    def create(cls, identifier, variables):
        return cls(str(identifier), tuple(variables))

    def __post_init__(self):
        if (
            not self.connector_type_id
            or not self.variables
            or len({x.name for x in self.variables}) != len(self.variables)
        ):
            raise ValueError("Connector type invalid.")


@dataclass(frozen=True, slots=True)
class Connector:
    connector_id: str
    connector_type: ConnectorType
    orientation: float = 1.0

    def __post_init__(self):
        if not isinstance(self.connector_id, str) or not self.connector_id:
            raise ValueError("Connector identifier must be a non-empty string.")
        if not isinstance(self.connector_type, ConnectorType):
            raise TypeError("connector_type must be ConnectorType.")
        if not isfinite(self.orientation) or self.orientation not in (-1.0, 1.0):
            raise ValueError("Connector orientation must be exactly -1 or +1.")


__all__ = ["Connector", "ConnectorType", "ConnectorVariable", "VariableKind"]
