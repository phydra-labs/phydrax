#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._trainable import NonTrainableState
from ...operators.quantum._observables import LocalObservable
from ...operators.quantum._operations import LocalUnitaryOperation
from ...operators.quantum._parameterized import (
    PauliRotationInstruction,
    QuantumProgramInstruction,
    QuantumProgramTemplate,
)
from ...operators.quantum._register import HilbertRegisterLayout
from ...solver._quantum_expectation import DenseQuantumObservablePolicy
from ...solver._quantum_program import DenseQuantumProgramPolicy
from ._models import (
    CircuitGradientMethod,
    DenseCircuitExpectationModel,
    DenseCircuitStateModel,
)


def _qubit_layout(layout: HilbertRegisterLayout, /) -> None:
    if not isinstance(layout, HilbertRegisterLayout):
        raise TypeError("layout must be a HilbertRegisterLayout.")
    if any(dimension != 2 for dimension in layout.local_dimensions):
        raise ValueError("Standard quantum feature maps require qubit layouts.")


def _edges(
    layout: HilbertRegisterLayout,
    edges: Sequence[tuple[str, str]],
    /,
) -> tuple[tuple[str, str], ...]:
    selected = tuple((str(left), str(right)) for left, right in edges)
    if len(set(selected)) != len(selected):
        raise ValueError("Entanglement edges must be unique.")
    for left, right in selected:
        if left == right:
            raise ValueError("Entanglement edges require distinct wires.")
        layout.target_indices((left, right))
    return selected


def _complex_dtype(dtype: DTypeLike, /) -> jnp.dtype:
    selected = jnp.dtype(dtype)
    if selected not in (jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)):
        raise TypeError("Quantum feature maps require complex64 or complex128.")
    return selected


def _angle_dtype(dtype: jnp.dtype, /) -> jnp.dtype:
    return (
        jnp.dtype(jnp.float32)
        if dtype == jnp.dtype(jnp.complex64)
        else jnp.dtype(jnp.float64)
    )


def _hadamard(wire_id: str, dtype: jnp.dtype, /) -> LocalUnitaryOperation:
    value = jnp.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=dtype) / jnp.sqrt(
        jnp.asarray(2.0, dtype=_angle_dtype(dtype))
    )
    return LocalUnitaryOperation(value, (wire_id,))


def _controlled_z(edge: tuple[str, str], dtype: jnp.dtype, /) -> LocalUnitaryOperation:
    return LocalUnitaryOperation(
        jnp.diag(jnp.asarray([1.0, 1.0, 1.0, -1.0], dtype=dtype)),
        edge,
    )


def _pauli_matrix(axis: str, dtype: jnp.dtype, /) -> Array:
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    imaginary = jnp.asarray(1.0j, dtype=dtype)
    if axis == "X":
        return jnp.asarray([[zero, one], [one, zero]], dtype=dtype)
    if axis == "Y":
        return jnp.asarray([[zero, -imaginary], [imaginary, zero]], dtype=dtype)
    return jnp.asarray([[one, zero], [zero, -one]], dtype=dtype)


class IQPAngleMap(AbstractArrayModel, NonTrainableState):
    """Fixed IQP phase map with explicit single and pair feature monomials."""

    pair_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    repetitions: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        pair_indices: Sequence[tuple[int, int]],
        /,
        *,
        repetitions: int = 1,
        dtype: DTypeLike = jnp.float64,
    ):
        size = int(input_size)
        repeats = int(repetitions)
        selected_dtype = jnp.dtype(dtype)
        pairs = tuple((int(left), int(right)) for left, right in pair_indices)
        if size <= 0 or repeats <= 0:
            raise ValueError("IQP input size and repetitions must be positive.")
        if selected_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("IQP angles require float32 or float64 coordinates.")
        if len(set(pairs)) != len(pairs):
            raise ValueError("IQP feature pairs must be unique.")
        if any(
            left == right or not 0 <= left < size or not 0 <= right < size
            for left, right in pairs
        ):
            raise ValueError("IQP feature-pair indices are invalid.")
        self.pair_indices = pairs
        self.repetitions = repeats
        self.dtype = str(selected_dtype)
        self.in_size = size
        self.out_size = repeats * (size + len(pairs))
        self.map_id = canonical_fingerprint(
            {
                "kind": "iqp-angle-map",
                "input_size": size,
                "pair_indices": pairs,
                "repetitions": repeats,
                "dtype": str(selected_dtype),
            }
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        value = jnp.asarray(x)
        if value.shape != (self.in_size,):
            raise ValueError("IQP input must have exact shape (in_size,).")
        if str(value.dtype) != self.dtype:
            raise TypeError("IQP input and angle-map dtypes must match exactly.")
        value = eqx.error_if(
            value, ~jnp.all(jnp.isfinite(value)), "IQP input is nonfinite."
        )
        pair_values = (
            jnp.stack([value[left] * value[right] for left, right in self.pair_indices])
            if self.pair_indices
            else jnp.empty((0,), dtype=value.dtype)
        )
        one_layer = jnp.concatenate((value, pair_values))
        return jnp.tile(one_layer, self.repetitions)


class ReuploadingAngleMap(AbstractArrayModel):
    """Trainable per-occurrence affine re-uploading of selected input features."""

    scale: Array
    bias: Array
    feature_indices: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        feature_indices: Sequence[int],
        key: PRNGKeyArray,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ):
        size = int(input_size)
        indices = tuple(int(index) for index in feature_indices)
        selected_dtype = jnp.dtype(dtype)
        if size <= 0 or not indices:
            raise ValueError("Re-uploading input and angle counts must be positive.")
        if selected_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Re-uploading angles require float32 or float64 coordinates.")
        if any(not 0 <= index < size for index in indices):
            raise ValueError("Re-uploading feature index is outside the input vector.")
        scale_key, bias_key = jax.random.split(key)
        self.scale = jnp.ones(
            (len(indices),), dtype=selected_dtype
        ) + 0.05 * jax.random.normal(
            scale_key,
            (len(indices),),
            dtype=selected_dtype,
        )
        self.bias = 0.05 * jax.random.normal(
            bias_key,
            (len(indices),),
            dtype=selected_dtype,
        )
        self.feature_indices = indices
        self.in_size = size
        self.out_size = len(indices)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        value = jnp.asarray(x)
        if value.shape != (self.in_size,):
            raise ValueError("Re-uploading input must have exact shape (in_size,).")
        if value.dtype != self.scale.dtype:
            raise TypeError("Re-uploading input and parameter dtypes must match exactly.")
        value = eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value)),
            "Re-uploading input is nonfinite.",
        )
        return self.scale * value[jnp.asarray(self.feature_indices)] + self.bias


def _iqp_components(
    layout: HilbertRegisterLayout,
    repetitions: int,
    entanglement_edges: Sequence[tuple[str, str]],
    dtype: DTypeLike,
    /,
) -> tuple[QuantumProgramTemplate, IQPAngleMap]:
    _qubit_layout(layout)
    repeats = int(repetitions)
    if repeats <= 0:
        raise ValueError("IQP repetitions must be positive.")
    selected_edges = _edges(layout, entanglement_edges)
    complex_dtype = _complex_dtype(dtype)
    pair_indices = tuple(
        (layout.wire_index(left), layout.wire_index(right))
        for left, right in selected_edges
    )
    angle_map = IQPAngleMap(
        layout.wire_count,
        pair_indices,
        repetitions=repeats,
        dtype=_angle_dtype(complex_dtype),
    )
    instructions: list[QuantumProgramInstruction] = []
    angle_index = 0
    for _ in range(repeats):
        instructions.extend(
            _hadamard(wire_id, complex_dtype) for wire_id in layout.wire_ids
        )
        for wire_id in layout.wire_ids:
            instructions.append(
                PauliRotationInstruction(
                    ("Z",),
                    (wire_id,),
                    angle_index,
                    dtype=complex_dtype,
                )
            )
            angle_index += 1
        for edge in selected_edges:
            instructions.append(
                PauliRotationInstruction(
                    ("Z", "Z"),
                    edge,
                    angle_index,
                    dtype=complex_dtype,
                )
            )
            angle_index += 1
    template = QuantumProgramTemplate(
        layout,
        instructions,
        state_kind="state-vector",
        dtype=complex_dtype,
    )
    return template, angle_map


def iqp_state_feature_map(
    layout: HilbertRegisterLayout,
    /,
    *,
    repetitions: int = 1,
    entanglement_edges: Sequence[tuple[str, str]] = (),
    dtype: DTypeLike = jnp.complex128,
    initial_state: ArrayLike | None = None,
    policy: DenseQuantumProgramPolicy | None = None,
) -> DenseCircuitStateModel:
    """Build an exact dense IQP state feature map."""
    template, angle_map = _iqp_components(
        layout,
        repetitions,
        entanglement_edges,
        dtype,
    )
    return DenseCircuitStateModel(
        angle_map,
        template,
        initial_state=initial_state,
        policy=policy,
    )


def projected_iqp_feature_map(
    layout: HilbertRegisterLayout,
    /,
    *,
    repetitions: int = 1,
    entanglement_edges: Sequence[tuple[str, str]] = (),
    axes: Sequence[str] = ("X", "Y", "Z"),
    dtype: DTypeLike = jnp.complex128,
    initial_state: ArrayLike | None = None,
    program_policy: DenseQuantumProgramPolicy | None = None,
    observable_policy: DenseQuantumObservablePolicy | None = None,
) -> DenseCircuitExpectationModel:
    """Build exact local Pauli features from an IQP state map."""
    template, angle_map = _iqp_components(
        layout,
        repetitions,
        entanglement_edges,
        dtype,
    )
    complex_dtype = _complex_dtype(dtype)
    selected_axes = tuple(str(axis).upper() for axis in axes)
    if not selected_axes or any(axis not in ("X", "Y", "Z") for axis in selected_axes):
        raise ValueError("Projected IQP axes must be a non-empty X/Y/Z sequence.")
    observables = tuple(
        LocalObservable(
            _pauli_matrix(axis, complex_dtype),
            (wire_id,),
        )
        for wire_id in layout.wire_ids
        for axis in selected_axes
    )
    return DenseCircuitExpectationModel(
        angle_map,
        template,
        observables,
        initial_state=initial_state,
        program_policy=program_policy,
        observable_policy=observable_policy,
    )


def data_reuploading_feature_map(
    input_size: int,
    layout: HilbertRegisterLayout,
    layers: int,
    key: PRNGKeyArray,
    /,
    *,
    entanglement_edges: Sequence[tuple[str, str]] = (),
    readout_wire_ids: Sequence[str] | None = None,
    gradient_method: CircuitGradientMethod = "autodiff",
    dtype: DTypeLike = jnp.complex128,
    initial_state: ArrayLike | None = None,
    program_policy: DenseQuantumProgramPolicy | None = None,
    observable_policy: DenseQuantumObservablePolicy | None = None,
) -> DenseCircuitExpectationModel:
    """Build a trainable exact affine data-reuploading feature model."""
    _qubit_layout(layout)
    size = int(input_size)
    depth = int(layers)
    if size <= 0 or depth <= 0:
        raise ValueError("Data-reuploading input size and layers must be positive.")
    selected_edges = _edges(layout, entanglement_edges)
    complex_dtype = _complex_dtype(dtype)
    instructions: list[QuantumProgramInstruction] = []
    feature_indices: list[int] = []
    angle_index = 0
    for _ in range(depth):
        for wire_index, wire_id in enumerate(layout.wire_ids):
            for axis_index, axis in enumerate(("X", "Y", "Z")):
                instructions.append(
                    PauliRotationInstruction(
                        (axis,),
                        (wire_id,),
                        angle_index,
                        dtype=complex_dtype,
                    )
                )
                feature_indices.append((3 * wire_index + axis_index) % size)
                angle_index += 1
        instructions.extend(_controlled_z(edge, complex_dtype) for edge in selected_edges)
    template = QuantumProgramTemplate(
        layout,
        instructions,
        state_kind="state-vector",
        dtype=complex_dtype,
    )
    angle_map = ReuploadingAngleMap(
        size,
        feature_indices,
        key,
        dtype=_angle_dtype(complex_dtype),
    )
    selected_readouts = (
        layout.wire_ids if readout_wire_ids is None else tuple(readout_wire_ids)
    )
    if not selected_readouts:
        raise ValueError("At least one data-reuploading readout wire is required.")
    observables = tuple(
        LocalObservable(
            _pauli_matrix("Z", complex_dtype),
            (wire_id,),
        )
        for wire_id in selected_readouts
    )
    return DenseCircuitExpectationModel(
        angle_map,
        template,
        observables,
        gradient_method=gradient_method,
        initial_state=initial_state,
        program_policy=program_policy,
        observable_policy=observable_policy,
    )


__all__ = [
    "IQPAngleMap",
    "ReuploadingAngleMap",
    "data_reuploading_feature_map",
    "iqp_state_feature_map",
    "projected_iqp_feature_map",
]
