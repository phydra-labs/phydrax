#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Structural compilation of linear acausal connector systems."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._connection import AcausalSystem
from ._structural import maximum_structural_matching


@dataclass(frozen=True, slots=True)
class AcausalSolveResult:
    values: Array
    residual_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CompiledAcausalSystem:
    variable_keys: tuple[tuple[str, str], ...]
    system_matrix: Array
    right_hand_side: Array
    connection_equation_count: int
    structural_rank: int
    equation_matching: tuple[int, ...]

    @property
    def square(self) -> bool:
        return self.system_matrix.shape[0] == self.system_matrix.shape[1]

    @property
    def structurally_nonsingular(self) -> bool:
        return self.square and self.structural_rank == self.system_matrix.shape[1]

    def solve(self, /, *, relative_tolerance: float = 1e-9) -> AcausalSolveResult:
        if not self.structurally_nonsingular:
            raise ValueError("Acausal system is not structurally square and nonsingular.")
        native = solve(
            LinearSystem(DenseLinearOperator(self.system_matrix)),
            self.right_hand_side,
            policy=LinearSolvePolicy(DenseLU()),
        )
        residual = self.system_matrix @ native.value - self.right_hand_side
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        right_norm = jnp.sqrt(
            jnp.real(
                contract("i,i->", jnp.conj(self.right_hand_side), self.right_hand_side)
            )
        )
        successful = native.successful & (
            residual_norm <= float(relative_tolerance) * jnp.maximum(right_norm, 1.0)
        )
        return AcausalSolveResult(native.value, residual_norm, successful)

    def connector_values(self, values: ArrayLike, /) -> dict[str, dict[str, Array]]:
        data = jnp.asarray(values)
        if data.shape != (len(self.variable_keys),):
            raise ValueError("Compiled acausal values do not match the variable basis.")
        grouped: dict[str, dict[str, Array]] = {}
        for index, (connector_id, variable_name) in enumerate(self.variable_keys):
            grouped.setdefault(connector_id, {})[variable_name] = data[index]
        return grouped


def compile_linear_acausal_system(
    system: AcausalSystem,
    component_matrix: ArrayLike,
    component_right_hand_side: ArrayLike,
    /,
) -> CompiledAcausalSystem:
    """Compile constitutive equations plus across/through connection equations."""

    connector_ids = tuple(connector.connector_id for connector in system.connectors)
    if len(connector_ids) != len(set(connector_ids)):
        raise ValueError("Acausal connector IDs must be unique.")
    connectors = {connector.connector_id: connector for connector in system.connectors}
    variable_keys = tuple(
        (connector.connector_id, variable.name)
        for connector in system.connectors
        for variable in connector.connector_type.variables
    )
    variable_index = {key: index for index, key in enumerate(variable_keys)}

    connection_rows: list[np.ndarray] = []
    for connection in system.connections:
        if any(identifier not in connectors for identifier in connection.connector_ids):
            raise ValueError("Connection set references an unknown connector.")
        members = tuple(connectors[identifier] for identifier in connection.connector_ids)
        signature = tuple(
            (variable.name, variable.kind, variable.unit)
            for variable in members[0].connector_type.variables
        )
        if any(
            tuple(
                (variable.name, variable.kind, variable.unit)
                for variable in member.connector_type.variables
            )
            != signature
            for member in members[1:]
        ):
            raise ValueError("Connected connector types are not compatible.")
        reference = members[0]
        for variable in reference.connector_type.variables:
            if variable.kind == "across":
                for member in members[1:]:
                    row = np.zeros((len(variable_keys),), dtype=np.float64)
                    row[variable_index[(member.connector_id, variable.name)]] = 1.0
                    row[variable_index[(reference.connector_id, variable.name)]] = -1.0
                    connection_rows.append(row)
            else:
                row = np.zeros((len(variable_keys),), dtype=np.float64)
                for member in members:
                    row[variable_index[(member.connector_id, variable.name)]] = (
                        member.orientation
                    )
                connection_rows.append(row)

    component = np.asarray(component_matrix)
    right = np.asarray(component_right_hand_side)
    if component.ndim != 2 or component.shape[1] != len(variable_keys):
        raise ValueError("Component equations do not match the connector variable basis.")
    if right.shape != (component.shape[0],):
        raise ValueError("Component right-hand side does not match component equations.")
    connection_matrix = (
        np.stack(connection_rows)
        if connection_rows
        else np.zeros((0, len(variable_keys)), dtype=component.dtype)
    )
    dtype = np.result_type(component, right, connection_matrix)
    matrix = np.concatenate(
        (component.astype(dtype), connection_matrix.astype(dtype)), axis=0
    )
    rhs = np.concatenate(
        (right.astype(dtype), np.zeros((len(connection_rows),), dtype=dtype))
    )
    incidence = np.abs(matrix) > 0
    matching, rank = maximum_structural_matching(incidence)
    return CompiledAcausalSystem(
        variable_keys,
        jnp.asarray(matrix),
        jnp.asarray(rhs),
        len(connection_rows),
        rank,
        matching,
    )


__all__ = [
    "AcausalSolveResult",
    "CompiledAcausalSystem",
    "compile_linear_acausal_system",
]
