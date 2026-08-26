#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._coordinates import HermitianSpectralCoordinates
from ._space import TensorSpectralDiscretization


class TensorSpectralSymmetry(StrictModule, NonTrainableState):
    """One orthogonal translation/reflection action on tensor spectral fields."""

    discretization: TensorSpectralDiscretization
    translations: Array
    component_matrix: Array
    axis_signs: tuple[int, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    symmetry_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        axis_signs: Sequence[int] | None = None,
        translations: Sequence[float] | None = None,
        component_matrix: ArrayLike | None = None,
        component_count: int | None = None,
        symmetry_id: str | None = None,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        rank = len(discretization.axes)
        signs = (
            (1,) * rank
            if axis_signs is None
            else tuple(int(value) for value in axis_signs)
        )
        if len(signs) != rank or any(value not in (-1, 1) for value in signs):
            raise ValueError("axis_signs must contain one +1 or -1 per spectral axis.")
        shifts = np.zeros((rank,), dtype=float)
        if translations is not None:
            shifts = np.asarray(tuple(translations), dtype=float)
        if shifts.shape != (rank,) or np.any(~np.isfinite(shifts)):
            raise ValueError("translations must contain one finite value per axis.")
        for axis, shift, sign in zip(discretization.axes, shifts, signs, strict=True):
            if axis.family != "fourier" and shift != 0.0:
                raise ValueError("Translations are supported only on Fourier axes.")
            if sign == -1 and axis.family not in ("fourier", "chebyshev", "legendre"):
                raise ValueError(
                    "Reflections require Fourier, Chebyshev, or Legendre axes."
                )
        count = (
            int(component_count)
            if component_count is not None
            else 1
            if component_matrix is None
            else int(np.asarray(component_matrix).shape[0])
        )
        if count < 1:
            raise ValueError("component_count must be positive.")
        matrix = (
            np.eye(count, dtype=float)
            if component_matrix is None
            else np.asarray(component_matrix)
        )
        if matrix.shape != (count, count) or np.any(~np.isfinite(matrix)):
            raise ValueError("component_matrix must be one finite square matrix.")
        if np.iscomplexobj(matrix) or not np.allclose(
            matrix.T @ matrix,
            np.eye(count),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("component_matrix must be real orthogonal.")
        normalized = np.mod(shifts, 1.0)
        state_shape = discretization.modal_shape + (count,)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "tensor-spectral-symmetry-v1",
                    "discretization": discretization.prepared_id,
                    "axis_signs": list(signs),
                    "translations": array_tree_fingerprint(normalized),
                    "component_matrix": array_tree_fingerprint(matrix),
                }
            )
            if symmetry_id is None
            else str(symmetry_id)
        )
        if not identifier:
            raise ValueError("symmetry_id must be non-empty.")
        dtype = jnp.empty(
            (), dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype)
        ).real.dtype
        self.discretization = discretization
        self.translations = jnp.asarray(normalized, dtype=dtype)
        self.component_matrix = jnp.asarray(matrix, dtype=dtype)
        self.axis_signs = signs
        self.component_count = count
        self.state_shape = state_shape
        self.symmetry_id = identifier

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"Spectral symmetry state must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Spectral symmetry state must be complex-valued.")
        return value

    def apply(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        result = value
        for axis_index, (axis, sign) in enumerate(
            zip(self.discretization.axes, self.axis_signs, strict=True)
        ):
            if sign == -1:
                if axis.family == "fourier":
                    result = jnp.take(
                        result,
                        axis.modes.conjugate_indices,
                        axis=axis_index,
                    )
                else:
                    parity = (-1.0) ** axis.modes.mode_numbers
                    shape = [1] * result.ndim
                    shape[axis_index] = axis.mode_count
                    result = result * parity.reshape(tuple(shape)).astype(result.dtype)
            if axis.family == "fourier":
                numbers = axis.modes.mode_numbers.astype(self.translations.dtype)
                phase = jnp.exp(
                    2j
                    * jnp.asarray(jnp.pi, dtype=self.translations.dtype)
                    * sign
                    * numbers
                    * self.translations[axis_index]
                )
                shape = [1] * result.ndim
                shape[axis_index] = axis.mode_count
                result = result * phase.reshape(tuple(shape)).astype(result.dtype)
        return oe.contract(
            "ij,...j->...i",
            self.component_matrix.astype(result.dtype),
            result,
            backend="jax",
        )

    def compose(self, other: TensorSpectralSymmetry, /) -> TensorSpectralSymmetry:
        """Return the action applying ``other`` first and then ``self``."""
        self._validate_compatible(other)
        signs = tuple(
            left * right
            for left, right in zip(self.axis_signs, other.axis_signs, strict=True)
        )
        translations = np.mod(
            np.asarray(self.axis_signs) * np.asarray(other.translations)
            + np.asarray(self.translations),
            1.0,
        )
        matrix = np.asarray(self.component_matrix) @ np.asarray(other.component_matrix)
        return TensorSpectralSymmetry(
            self.discretization,
            axis_signs=signs,
            translations=translations,
            component_matrix=matrix,
            component_count=self.component_count,
        )

    def inverse(self, /) -> TensorSpectralSymmetry:
        translations = np.mod(
            -np.asarray(self.axis_signs) * np.asarray(self.translations), 1.0
        )
        return TensorSpectralSymmetry(
            self.discretization,
            axis_signs=self.axis_signs,
            translations=translations,
            component_matrix=np.asarray(self.component_matrix).T,
            component_count=self.component_count,
        )

    def translation_generator(self, state: ArrayLike, axis: int, /) -> Array:
        value = self.validate_state(state)
        axis_index = int(axis)
        if axis_index < 0 or axis_index >= len(self.discretization.axes):
            raise ValueError("axis is outside the tensor spectral rank.")
        prepared = self.discretization.axes[axis_index]
        if prepared.family != "fourier":
            raise ValueError("Translation generators require a Fourier axis.")
        numbers = prepared.modes.mode_numbers.astype(value.real.dtype)
        shape = [1] * value.ndim
        shape[axis_index] = prepared.mode_count
        multiplier = 2j * jnp.asarray(jnp.pi, dtype=value.real.dtype) * numbers
        return value * multiplier.reshape(tuple(shape)).astype(value.dtype)

    def fixed_subspace_defect(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        return jnp.linalg.norm((self.apply(value) - value).reshape((-1,)))

    def apply_real_coordinates(
        self,
        coordinates: ArrayLike,
        chart: HermitianSpectralCoordinates,
        /,
    ) -> Array:
        if chart.state_shape != self.state_shape:
            raise ValueError("Symmetry and Hermitian coordinate state shapes must match.")
        state = chart.from_real_coordinates(coordinates)
        return chart.to_real_coordinates(self.apply(state))

    def _validate_compatible(self, other: Any, /) -> None:
        if not isinstance(other, TensorSpectralSymmetry):
            raise TypeError("other must be a TensorSpectralSymmetry.")
        if (
            other.discretization.prepared_id != self.discretization.prepared_id
            or other.component_count != self.component_count
        ):
            raise ValueError("Tensor spectral symmetries are not compatible.")


def project_tensor_spectral_symmetries(
    state: ArrayLike,
    symmetries: Sequence[TensorSpectralSymmetry],
    /,
) -> Array:
    """Apply the Reynolds average over an explicitly enumerated finite group."""
    elements = tuple(symmetries)
    if not elements:
        raise ValueError("symmetries must contain at least one group element.")
    first = elements[0]
    value = first.validate_state(state)
    for element in elements[1:]:
        first._validate_compatible(element)
    return sum(
        (element.apply(value) for element in elements), jnp.zeros_like(value)
    ) / len(elements)


__all__ = ["TensorSpectralSymmetry", "project_tensor_spectral_symmetries"]
