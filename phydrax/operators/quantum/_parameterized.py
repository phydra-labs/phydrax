#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._operations import (
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumOperation,
    QuantumProgram,
    QuantumStateKind,
)
from ._register import _target_wire_ids, HilbertRegisterLayout


_PAULI_AXES = ("X", "Y", "Z")


class PauliRotationInstruction(StrictModule):
    """One angle-bound one- or two-qubit Pauli rotation instruction."""

    pauli: Array
    axes: tuple[str, ...] = eqx.field(static=True)
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    angle_index: int = eqx.field(static=True)
    instruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[str],
        target_wire_ids: Sequence[str],
        angle_index: int,
        /,
        *,
        dtype: DTypeLike = jnp.complex128,
    ):
        selected_axes = tuple(str(axis).upper() for axis in axes)
        targets = _target_wire_ids(target_wire_ids)
        if len(selected_axes) != len(targets) or len(targets) not in (1, 2):
            raise ValueError(
                "Pauli rotations require one axis per target and one or two targets."
            )
        if any(axis not in _PAULI_AXES for axis in selected_axes):
            raise ValueError("Pauli-rotation axes must be X, Y, or Z.")
        selected_index = int(angle_index)
        if selected_index < 0:
            raise ValueError("angle_index must be nonnegative.")
        selected_dtype = jnp.dtype(dtype)
        if selected_dtype not in (jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)):
            raise TypeError(
                "Pauli rotations require complex64 or complex128 coordinates."
            )
        one = jnp.asarray(1.0, dtype=selected_dtype)
        zero = jnp.asarray(0.0, dtype=selected_dtype)
        imaginary = jnp.asarray(1.0j, dtype=selected_dtype)
        matrices = {
            "X": jnp.asarray([[zero, one], [one, zero]], dtype=selected_dtype),
            "Y": jnp.asarray(
                [[zero, -imaginary], [imaginary, zero]], dtype=selected_dtype
            ),
            "Z": jnp.asarray([[one, zero], [zero, -one]], dtype=selected_dtype),
        }
        pauli = matrices[selected_axes[0]]
        for axis in selected_axes[1:]:
            pauli = jnp.kron(pauli, matrices[axis])
        self.pauli = pauli
        self.axes = selected_axes
        self.target_wire_ids = targets
        self.angle_index = selected_index
        self.instruction_id = canonical_fingerprint(
            {
                "kind": "pauli-rotation-instruction",
                "axes": selected_axes,
                "targets": targets,
                "angle_index": selected_index,
                "dtype": str(selected_dtype),
            }
        )


QuantumProgramInstruction: TypeAlias = QuantumOperation | PauliRotationInstruction


class QuantumProgramTemplate(StrictModule):
    """Host-constructed angle template that lowers to one canonical program schema."""

    layout: HilbertRegisterLayout
    instructions: tuple[QuantumProgramInstruction, ...]
    state_kind: QuantumStateKind = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    angle_count: int = eqx.field(static=True)
    parameterized_operation_count: int = eqx.field(static=True)
    occurrence_angle_indices: tuple[int, ...] = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: HilbertRegisterLayout,
        instructions: Sequence[QuantumProgramInstruction],
        /,
        *,
        state_kind: QuantumStateKind,
        dtype: DTypeLike = jnp.complex128,
    ):
        if not isinstance(layout, HilbertRegisterLayout):
            raise TypeError("layout must be a HilbertRegisterLayout.")
        if state_kind not in ("state-vector", "density-matrix"):
            raise ValueError("Unknown quantum-program state kind.")
        selected_dtype = jnp.dtype(dtype)
        if selected_dtype not in (jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)):
            raise TypeError("Quantum program templates require complex64 or complex128.")
        selected = tuple(instructions)
        angle_indices: list[int] = []
        identities: list[dict[str, object]] = []
        for instruction in selected:
            if isinstance(instruction, PauliRotationInstruction):
                if instruction.pauli.dtype != selected_dtype:
                    raise TypeError(
                        "Parameterized and template dtypes must match exactly."
                    )
                if any(
                    layout.local_dimensions[layout.wire_index(wire_id)] != 2
                    for wire_id in instruction.target_wire_ids
                ):
                    raise ValueError("Pauli rotations require two-dimensional targets.")
                angle_indices.append(instruction.angle_index)
                identities.append(
                    {"kind": "parameterized", "id": instruction.instruction_id}
                )
                continue
            if not isinstance(
                instruction, (LocalUnitaryOperation, LocalKrausChannelOperation)
            ):
                raise TypeError("Unknown quantum-program template instruction.")
            matrix = (
                instruction.unitary
                if isinstance(instruction, LocalUnitaryOperation)
                else instruction.kraus
            )
            if matrix.dtype != selected_dtype:
                raise TypeError("Fixed-operation and template dtypes must match exactly.")
            if matrix.shape[-1] != layout.target_dimension(instruction.target_wire_ids):
                raise ValueError(
                    "Template operation dimension does not match its ordered targets."
                )
            if state_kind == "state-vector" and isinstance(
                instruction, LocalKrausChannelOperation
            ):
                raise ValueError("Kraus channels require a density-matrix template.")
            identities.append(
                {
                    "kind": "fixed",
                    "schema": instruction.schema_id,
                    "content": array_tree_fingerprint(matrix)["sha256"],
                }
            )
        angle_count = max(angle_indices, default=-1) + 1
        if set(angle_indices) != set(range(angle_count)):
            raise ValueError("Template angle indices must be contiguous and all used.")
        self.layout = layout
        self.instructions = selected
        self.state_kind = state_kind
        self.dtype = str(selected_dtype)
        self.angle_count = angle_count
        self.parameterized_operation_count = len(angle_indices)
        self.occurrence_angle_indices = tuple(angle_indices)
        self.template_id = canonical_fingerprint(
            {
                "kind": "quantum-program-template",
                "layout": layout.layout_id,
                "state_kind": state_kind,
                "dtype": str(selected_dtype),
                "instructions": identities,
            }
        )

    @property
    def complex_dtype(self) -> jnp.dtype:
        return jnp.dtype(self.dtype)

    @property
    def angle_dtype(self) -> jnp.dtype:
        if self.complex_dtype == jnp.dtype(jnp.complex64):
            return jnp.dtype(jnp.float32)
        return jnp.dtype(jnp.float64)


def _materialize_quantum_program(
    template: QuantumProgramTemplate,
    angles: ArrayLike,
    /,
    *,
    shifted_occurrence: int | Array = -1,
    shift: ArrayLike = 0.0,
) -> QuantumProgram:
    if not isinstance(template, QuantumProgramTemplate):
        raise TypeError("template must be a QuantumProgramTemplate.")
    values = jnp.asarray(angles)
    if values.shape != (template.angle_count,):
        raise ValueError("angles must have exact shape (template.angle_count,).")
    if values.dtype != template.angle_dtype:
        raise TypeError("Angle and template precisions must match exactly.")
    selected_occurrence = jnp.asarray(shifted_occurrence, dtype=jnp.int32)
    selected_shift = jnp.asarray(shift, dtype=template.angle_dtype)
    operations: list[QuantumOperation] = []
    occurrence = 0
    complex_unit = jnp.asarray(1.0j, dtype=template.complex_dtype)
    for instruction in template.instructions:
        if not isinstance(instruction, PauliRotationInstruction):
            operations.append(instruction)
            continue
        angle = values[instruction.angle_index] + jnp.where(
            selected_occurrence == occurrence,
            selected_shift,
            jnp.asarray(0.0, dtype=template.angle_dtype),
        )
        identity = jnp.eye(instruction.pauli.shape[0], dtype=template.complex_dtype)
        unitary = (
            jnp.cos(0.5 * angle) * identity
            - complex_unit * jnp.sin(0.5 * angle) * instruction.pauli
        )
        operations.append(LocalUnitaryOperation(unitary, instruction.target_wire_ids))
        occurrence += 1
    return QuantumProgram(template.layout, operations, state_kind=template.state_kind)


def materialize_quantum_program(
    template: QuantumProgramTemplate,
    angles: ArrayLike,
    /,
) -> QuantumProgram:
    """Lower one angle vector to the canonical numeric quantum-program IR."""
    return _materialize_quantum_program(template, angles)


__all__ = [
    "PauliRotationInstruction",
    "QuantumProgramInstruction",
    "QuantumProgramTemplate",
    "materialize_quantum_program",
]
