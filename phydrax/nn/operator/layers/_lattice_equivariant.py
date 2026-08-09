#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.nn.layers._measure_convolution import _broadcast_sample_field
from phydrax.nn.operator.representations import (
    FiniteOrthogonalGroup,
    TensorFieldLayout,
)


def _kernel_shape(value: int | Sequence[int], dimension: int, /) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (int(value),) * dimension
    else:
        result = tuple(int(size) for size in value)
    if len(result) != dimension or any(size <= 0 or size % 2 == 0 for size in result):
        raise ValueError("kernel_shape must contain one positive odd size per dimension.")
    return result


def _centered_spatial_action_numpy(
    values: np.ndarray,
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
    /,
) -> np.ndarray:
    dimension = len(permutation)
    axes = (
        (0,)
        + tuple(1 + axis for axis in permutation)
        + tuple(range(1 + dimension, values.ndim))
    )
    transformed = np.transpose(values, axes)
    for axis, sign in enumerate(signs, start=1):
        if sign < 0:
            transformed = np.flip(transformed, axis=axis)
    return transformed


def _kernel_group_action_numpy(
    kernels: np.ndarray,
    group: FiniteOrthogonalGroup,
    input_actions: np.ndarray,
    output_actions: np.ndarray,
    element: int,
    /,
) -> np.ndarray:
    if group.lattice_permutations is None or group.lattice_signs is None:
        raise ValueError("Invariant lattice filters require signed-permutation groups.")
    spatial = _centered_spatial_action_numpy(
        kernels,
        group.lattice_permutations[element],
        group.lattice_signs[element],
    )
    return np.einsum(
        "oa,n...ab,bi->n...oi",
        output_actions[element],
        spatial,
        np.linalg.inv(input_actions[element]),
        optimize=True,
    )


def _ordinary_scalar_indices(layout: TensorFieldLayout, /) -> tuple[int, ...]:
    indices = []
    start = 0
    for block in layout.blocks:
        stop = start + block.channel_count
        if block.tensor_type.is_scalar:
            indices.extend(range(start, stop))
        start = stop
    return tuple(indices)


class InvariantFilterBasis(StrictModule, NonTrainableState):
    """Deterministic orthonormal basis of finite-group equivariant kernels."""

    group: FiniteOrthogonalGroup
    input_layout: TensorFieldLayout
    output_layout: TensorFieldLayout
    kernel_shape: tuple[int, ...]
    basis: Array
    rank: int
    construction_size: int
    equivariance_tolerance: float
    fingerprint: str

    def __init__(
        self,
        group: FiniteOrthogonalGroup,
        input_layout: TensorFieldLayout,
        output_layout: TensorFieldLayout,
        /,
        *,
        kernel_shape: int | Sequence[int] = 3,
        rank_tolerance: float = 1e-10,
        equivariance_tolerance: float = 1e-10,
        max_construction_bytes: int = 256 * 1024**2,
    ):
        if not isinstance(group, FiniteOrthogonalGroup):
            raise TypeError("group must be a FiniteOrthogonalGroup.")
        if not isinstance(input_layout, TensorFieldLayout) or not isinstance(
            output_layout, TensorFieldLayout
        ):
            raise TypeError(
                "input_layout and output_layout must be TensorFieldLayout values."
            )
        if not group.supports_lattice_action:
            raise ValueError(
                "Invariant lattice filters require a signed-permutation group."
            )
        if (
            input_layout.dimension != group.dimension
            or output_layout.dimension != group.dimension
        ):
            raise ValueError("Group and tensor layout dimensions must agree.")
        shape = _kernel_shape(kernel_shape, group.dimension)
        for permutation in group.lattice_permutations or ():
            if any(
                shape[axis] != shape[permutation[axis]] for axis in range(group.dimension)
            ):
                raise ValueError("Axis-permuting groups require matching kernel sizes.")
        rank_tol = float(rank_tolerance)
        equivariance_tol = float(equivariance_tolerance)
        if not np.isfinite(rank_tol) or rank_tol <= 0.0:
            raise ValueError("rank_tolerance must be positive and finite.")
        if not np.isfinite(equivariance_tol) or equivariance_tol <= 0.0:
            raise ValueError("equivariance_tolerance must be positive and finite.")
        construction_size = (
            math.prod(shape) * input_layout.channel_count * output_layout.channel_count
        )
        estimated_bytes = 3 * construction_size * construction_size * 8
        if estimated_bytes > int(max_construction_bytes):
            raise ValueError(
                "Invariant filter construction exceeds max_construction_bytes; "
                f"estimated {estimated_bytes} bytes."
            )

        input_actions = np.asarray(
            input_layout.channel_actions(group.matrices), dtype=float
        )
        output_actions = np.asarray(
            output_layout.channel_actions(group.matrices), dtype=float
        )
        candidates = np.eye(construction_size, dtype=float).reshape(
            (construction_size,)
            + shape
            + (output_layout.channel_count, input_layout.channel_count)
        )
        projected = np.zeros_like(candidates)
        for element in range(group.order):
            projected += _kernel_group_action_numpy(
                candidates,
                group,
                input_actions,
                output_actions,
                element,
            )
        projected /= float(group.order)
        projector = projected.reshape(construction_size, construction_size)
        projector = 0.5 * (projector + projector.T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        retained = eigenvalues > rank_tol
        basis_flat = eigenvectors[:, retained].T
        for index in range(basis_flat.shape[0]):
            pivot = int(np.argmax(np.abs(basis_flat[index])))
            if basis_flat[index, pivot] < 0.0:
                basis_flat[index] *= -1.0
        basis = basis_flat.reshape(
            (basis_flat.shape[0],)
            + shape
            + (output_layout.channel_count, input_layout.channel_count)
        )
        defect = 0.0
        for element in range(group.order):
            transformed = _kernel_group_action_numpy(
                basis,
                group,
                input_actions,
                output_actions,
                element,
            )
            if basis.size:
                defect = max(defect, float(np.max(np.abs(transformed - basis))))
        if defect > equivariance_tol:
            raise ValueError(
                "Constructed invariant basis does not meet the equivariance tolerance: "
                f"{defect} > {equivariance_tol}."
            )
        digest = hashlib.sha256()
        digest.update(group.fingerprint.encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "input": input_layout.to_dict(),
                    "output": output_layout.to_dict(),
                    "kernel_shape": shape,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(np.round(basis, decimals=12).tobytes())

        self.group = group
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.kernel_shape = shape
        self.basis = jnp.asarray(basis)
        self.rank = int(basis.shape[0])
        self.construction_size = int(construction_size)
        self.equivariance_tolerance = equivariance_tol
        self.fingerprint = digest.hexdigest()

    def synthesize(self, coefficients: Array, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.rank,):
            raise ValueError(f"coefficients must have shape ({self.rank},).")
        return jnp.einsum("r,r...oi->...oi", values, self.basis.astype(values.dtype))

    def project(self, kernel: Array, /) -> Array:
        values = jnp.asarray(kernel)
        expected = self.kernel_shape + (
            self.output_layout.channel_count,
            self.input_layout.channel_count,
        )
        if values.shape != expected:
            raise ValueError(f"kernel must have shape {expected}; got {values.shape}.")
        flat_basis = self.basis.astype(values.dtype).reshape(self.rank, -1)
        coefficients = jnp.einsum("ri,i->r", flat_basis, values.reshape(-1))
        return self.synthesize(coefficients)

    def equivariance_defect(self, kernel: Array, /) -> Array:
        values = jnp.asarray(kernel)
        projected = self.project(values)
        return jnp.max(jnp.abs(values - projected), initial=0.0)


class LatticeEquivariantConvND(StrictModule):
    """Periodic measure-aware convolution parameterized only in an invariant basis."""

    invariant_basis: InvariantFilterBasis
    coefficients: Array
    bias: Array | None
    bias_indices: tuple[int, ...]
    epsilon: float

    def __init__(
        self,
        invariant_basis: InvariantFilterBasis,
        /,
        *,
        use_bias: bool = True,
        epsilon: float = 1e-12,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(invariant_basis, InvariantFilterBasis):
            raise TypeError("invariant_basis must be an InvariantFilterBasis.")
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        rank = invariant_basis.rank
        scale = 1.0 / math.sqrt(max(rank, 1))
        self.coefficients = scale * jr.normal(key, (rank,), dtype=resolved_dtype)
        scalar_indices = _ordinary_scalar_indices(invariant_basis.output_layout)
        if use_bias and not scalar_indices:
            raise ValueError(
                "Equivariant biases require an ordinary scalar output block."
            )
        self.bias = (
            jnp.zeros((len(scalar_indices),), dtype=resolved_dtype) if use_bias else None
        )
        self.bias_indices = scalar_indices if use_bias else ()
        self.invariant_basis = invariant_basis
        self.epsilon = epsilon_value

    @property
    def spatial_ndim(self) -> int:
        return self.invariant_basis.group.dimension

    @property
    def in_channels(self) -> int:
        return self.invariant_basis.input_layout.channel_count

    @property
    def out_channels(self) -> int:
        return self.invariant_basis.output_layout.channel_count

    def kernel(self, /) -> Array:
        return self.invariant_basis.synthesize(self.coefficients)

    def _convolve(self, values: Array, kernel: Array, /) -> Array:
        spatial_axes = tuple(range(values.ndim - self.spatial_ndim - 1, values.ndim - 1))
        output = jnp.zeros(values.shape[:-1] + (self.out_channels,), dtype=values.dtype)
        centers = tuple(size // 2 for size in self.invariant_basis.kernel_shape)
        for index in np.ndindex(self.invariant_basis.kernel_shape):
            offsets = tuple(
                position - center for position, center in zip(index, centers, strict=True)
            )
            shifted = jnp.roll(
                values,
                shift=tuple(-offset for offset in offsets),
                axis=spatial_axes,
            )
            output = output + jnp.einsum("...i,oi->...o", shifted, kernel[index])
        return output

    def __call__(
        self,
        values: Array,
        /,
        *,
        source_mask: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        quadrature: ArrayLike | None = None,
    ) -> Array:
        inputs = jnp.asarray(values)
        if (
            inputs.ndim < self.spatial_ndim + 1
            or int(inputs.shape[-1]) != self.in_channels
        ):
            raise ValueError(
                "values must have case axes followed by compatible spatial axes and channels."
            )
        sample_shape = tuple(
            int(size) for size in inputs.shape[-self.spatial_ndim - 1 : -1]
        )
        case_shape = tuple(int(size) for size in inputs.shape[: -self.spatial_ndim - 1])
        compute_dtype = jnp.result_type(inputs.dtype, self.coefficients.dtype)
        inputs = inputs.astype(compute_dtype)
        source_valid = (
            jnp.ones(case_shape + sample_shape, dtype=bool)
            if source_mask is None
            else _broadcast_sample_field(
                source_mask,
                case_shape,
                sample_shape,
                owner="source_mask",
                dtype=bool,
            )
        )
        measure = (
            jnp.ones(case_shape + sample_shape, dtype=compute_dtype)
            if quadrature is None
            else _broadcast_sample_field(
                quadrature,
                case_shape,
                sample_shape,
                owner="quadrature",
                dtype=compute_dtype,
            )
        )
        measure = eqx.error_if(
            measure,
            jnp.any(~jnp.isfinite(measure) | (measure < 0.0)),
            "quadrature must be finite and non-negative.",
        )
        measured = (
            jnp.where(source_valid[..., None], inputs, jnp.zeros_like(inputs))
            * measure[..., None]
        )
        numerator = self._convolve(measured, self.kernel().astype(compute_dtype))
        support = jnp.where(source_valid, measure, jnp.zeros_like(measure))
        support_kernel = jnp.ones(
            self.invariant_basis.kernel_shape + (1, 1), dtype=compute_dtype
        )
        support_sum = self._convolve(support[..., None], support_kernel)[..., 0]
        mean_measure = support_sum / float(math.prod(self.invariant_basis.kernel_shape))
        has_support = support_sum > 0.0
        output = numerator / jnp.maximum(mean_measure[..., None], self.epsilon)
        if self.bias is not None:
            full_bias = jnp.zeros((self.out_channels,), dtype=compute_dtype)
            full_bias = full_bias.at[jnp.asarray(self.bias_indices)].set(
                self.bias.astype(compute_dtype)
            )
            output = output + full_bias
        output = jnp.where(has_support[..., None], output, jnp.zeros_like(output))
        if target_mask is not None:
            target_valid = _broadcast_sample_field(
                target_mask,
                case_shape,
                sample_shape,
                owner="target_mask",
                dtype=bool,
            )
            output = jnp.where(target_valid[..., None], output, jnp.zeros_like(output))
        return output


class TensorPointwiseLinear(StrictModule):
    """Pointwise tensor intertwiner parameterized in an exact invariant basis."""

    basis: InvariantFilterBasis
    coefficients: Array
    bias: Array | None
    bias_indices: tuple[int, ...]

    def __init__(
        self,
        group: FiniteOrthogonalGroup,
        input_layout: TensorFieldLayout,
        output_layout: TensorFieldLayout,
        /,
        *,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        basis = InvariantFilterBasis(
            group,
            input_layout,
            output_layout,
            kernel_shape=(1,) * group.dimension,
        )
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        scale = 1.0 / math.sqrt(max(basis.rank, 1))
        self.coefficients = scale * jr.normal(key, (basis.rank,), dtype=resolved_dtype)
        scalar_indices = _ordinary_scalar_indices(output_layout)
        if use_bias and not scalar_indices:
            raise ValueError(
                "Equivariant biases require an ordinary scalar output block."
            )
        self.bias = (
            jnp.zeros((len(scalar_indices),), dtype=resolved_dtype) if use_bias else None
        )
        self.bias_indices = scalar_indices if use_bias else ()
        self.basis = basis

    def __call__(self, values: Array, /) -> Array:
        inputs = jnp.asarray(values)
        if (
            inputs.ndim < 1
            or int(inputs.shape[-1]) != self.basis.input_layout.channel_count
        ):
            raise ValueError("values have an incompatible tensor channel width.")
        matrix = self.basis.synthesize(self.coefficients)
        matrix = matrix.reshape(
            self.basis.output_layout.channel_count,
            self.basis.input_layout.channel_count,
        )
        output = jnp.einsum("...i,oi->...o", inputs, matrix.astype(inputs.dtype))
        if self.bias is not None:
            full_bias = jnp.zeros((matrix.shape[0],), dtype=output.dtype)
            full_bias = full_bias.at[jnp.asarray(self.bias_indices)].set(
                self.bias.astype(output.dtype)
            )
            output = output + full_bias
        return output


class TensorNormActivation(StrictModule):
    """Equivariant scalar activation and invariant-norm tensor rescaling."""

    layout: TensorFieldLayout
    activation: Callable
    epsilon: float

    def __init__(
        self,
        layout: TensorFieldLayout,
        activation: Callable = jax.nn.gelu,
        /,
        *,
        epsilon: float = 1e-12,
    ):
        if not isinstance(layout, TensorFieldLayout):
            raise TypeError("layout must be a TensorFieldLayout.")
        if not callable(activation):
            raise TypeError("activation must be callable.")
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        self.layout = layout
        self.activation = activation
        self.epsilon = epsilon_value

    def __call__(self, values: Array, /) -> Array:
        activated = []
        for block, array in zip(
            self.layout.blocks, self.layout.unpack(values), strict=True
        ):
            if block.tensor_type.is_scalar:
                activated.append(self.activation(array))
                continue
            component_axes = tuple(range(array.ndim - block.tensor_type.rank, array.ndim))
            if component_axes:
                norm = jnp.sqrt(
                    jnp.sum(array * array, axis=component_axes, keepdims=True)
                )
            else:
                norm = jnp.abs(array)
            scale = self.activation(norm) / jnp.maximum(norm, self.epsilon)
            activated.append(array * scale)
        return self.layout.pack(activated)


class TensorRMSNorm(StrictModule):
    """Per-block RMS normalization with one invariant gain per tensor copy."""

    layout: TensorFieldLayout
    gains: tuple[Array, ...]
    epsilon: float

    def __init__(
        self,
        layout: TensorFieldLayout,
        /,
        *,
        epsilon: float = 1e-6,
        dtype: Any = jnp.float32,
    ):
        if not isinstance(layout, TensorFieldLayout):
            raise TypeError("layout must be a TensorFieldLayout.")
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("dtype must be a real floating dtype.")
        self.layout = layout
        self.gains = tuple(
            jnp.ones((block.multiplicity,), dtype=resolved_dtype)
            for block in layout.blocks
        )
        self.epsilon = epsilon_value

    def __call__(self, values: Array, /) -> Array:
        normalized = []
        for block, gain, array in zip(
            self.layout.blocks,
            self.gains,
            self.layout.unpack(values),
            strict=True,
        ):
            block_axes = tuple(range(array.ndim - len(block.value_shape), array.ndim))
            rms = jnp.sqrt(
                jnp.mean(array * array, axis=block_axes, keepdims=True) + self.epsilon
            )
            gain_shape = (
                (1,) * (array.ndim - len(block.value_shape))
                + (block.multiplicity,)
                + (1,) * block.tensor_type.rank
            )
            normalized.append(array / rms * gain.reshape(gain_shape))
        return self.layout.pack(normalized)


__all__ = [
    "InvariantFilterBasis",
    "LatticeEquivariantConvND",
    "TensorNormActivation",
    "TensorPointwiseLinear",
    "TensorRMSNorm",
]
