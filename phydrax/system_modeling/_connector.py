#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
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


__all__ = ["Connector", "ConnectorType", "ConnectorVariable", "VariableKind"]
