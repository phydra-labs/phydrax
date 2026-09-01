#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._register import HilbertRegisterLayout


QuantumStateKind: TypeAlias = Literal["state-vector", "density-matrix"]


def _targets(value: Sequence[str], /) -> tuple[str, ...]:
    targets = tuple(str(wire_id) for wire_id in value)
    if not targets or any(not wire_id for wire_id in targets):
        raise ValueError("Local quantum-operation targets must be non-empty.")
    if len(set(targets)) != len(targets):
        raise ValueError("Local quantum-operation targets must be unique.")
    return targets


def _complex_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must use complex floating-point coordinates.")
    return array


class LocalUnitaryOperation(StrictModule):
    """One ordered local unitary matrix and its target Hilbert factors."""

    unitary: Array
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        unitary: ArrayLike,
        target_wire_ids: Sequence[str],
        /,
    ):
        matrix = _complex_array(unitary, "unitary")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("unitary must have exact square shape (dT, dT).")
        targets = _targets(target_wire_ids)
        self.unitary = matrix
        self.target_wire_ids = targets
        self.schema_id = canonical_fingerprint(
            {
                "kind": "local-unitary-operation",
                "targets": targets,
                "shape": matrix.shape,
                "dtype": str(matrix.dtype),
            }
        )


class LocalKrausChannelOperation(StrictModule):
    """One local completely-positive map represented by Kraus matrices."""

    kraus: Array
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        kraus: ArrayLike,
        target_wire_ids: Sequence[str],
        /,
    ):
        operators = _complex_array(kraus, "kraus")
        if (
            operators.ndim != 3
            or operators.shape[0] < 1
            or operators.shape[1] != operators.shape[2]
        ):
            raise ValueError("kraus must have exact shape (K, dT, dT) with K >= 1.")
        targets = _targets(target_wire_ids)
        self.kraus = operators
        self.target_wire_ids = targets
        self.schema_id = canonical_fingerprint(
            {
                "kind": "local-kraus-channel-operation",
                "targets": targets,
                "shape": operators.shape,
                "dtype": str(operators.dtype),
            }
        )


QuantumOperation: TypeAlias = LocalUnitaryOperation | LocalKrausChannelOperation


class QuantumProgram(StrictModule):
    """Immutable ordered local-operation program on one explicit Hilbert layout."""

    layout: HilbertRegisterLayout
    operations: tuple[QuantumOperation, ...]
    state_kind: QuantumStateKind = eqx.field(static=True)
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: HilbertRegisterLayout,
        operations: Sequence[QuantumOperation],
        /,
        *,
        state_kind: QuantumStateKind,
    ):
        if not isinstance(layout, HilbertRegisterLayout):
            raise TypeError("layout must be a HilbertRegisterLayout.")
        if state_kind not in ("state-vector", "density-matrix"):
            raise ValueError("Unknown quantum-program state kind.")
        selected = tuple(operations)
        for operation in selected:
            if not isinstance(
                operation, (LocalUnitaryOperation, LocalKrausChannelOperation)
            ):
                raise TypeError(
                    "Quantum programs contain only supported local operations."
                )
            dimension = layout.target_dimension(operation.target_wire_ids)
            matrix_dimension = (
                operation.unitary.shape[-1]
                if isinstance(operation, LocalUnitaryOperation)
                else operation.kraus.shape[-1]
            )
            if matrix_dimension != dimension:
                raise ValueError(
                    "Local operation matrix dimension does not match its ordered targets."
                )
            if state_kind == "state-vector" and isinstance(
                operation, LocalKrausChannelOperation
            ):
                raise ValueError("Kraus channels require a density-matrix program.")
        self.layout = layout
        self.operations = selected
        self.state_kind = state_kind
        self.program_id = canonical_fingerprint(
            {
                "kind": "quantum-program",
                "layout": layout.layout_id,
                "state_kind": state_kind,
                "operations": [operation.schema_id for operation in selected],
            }
        )


__all__ = [
    "LocalKrausChannelOperation",
    "LocalUnitaryOperation",
    "QuantumOperation",
    "QuantumProgram",
    "QuantumStateKind",
]
