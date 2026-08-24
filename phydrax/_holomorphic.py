#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


HolomorphicParameterCoverage = Literal[
    "finite-subspace",
    "finite-parametric-family",
]


class ComplexAffineNormalization(StrictModule, NonTrainableState):
    """Fixed invertible complex-affine coordinate normalization."""

    center: Array
    matrix: Array
    normalization_id: str = eqx.field(static=True)

    def __init__(self, center: ArrayLike, matrix: ArrayLike, /):
        center_ = np.asarray(center, dtype=np.complex128).reshape((-1,))
        matrix_ = np.asarray(matrix, dtype=np.complex128)
        if center_.size == 0 or matrix_.shape != (center_.size, center_.size):
            raise ValueError("Complex normalization requires a nonempty square map.")
        if not np.all(np.isfinite(center_)) or not np.all(np.isfinite(matrix_)):
            raise ValueError("Complex normalization data must be finite.")
        singular_values = np.linalg.svd(matrix_, compute_uv=False)
        tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * max(float(singular_values[0]), 1.0)
            * center_.size
        )
        if singular_values[-1] <= tolerance:
            raise ValueError("Complex normalization matrix must be invertible.")
        self.center = jnp.asarray(center_)
        self.matrix = jnp.asarray(matrix_)
        self.normalization_id = canonical_fingerprint(
            {
                "kind": "complex-affine-normalization-v1",
                "center": array_tree_fingerprint(self.center),
                "matrix": array_tree_fingerprint(self.matrix),
            }
        )

    @classmethod
    def identity(cls, dimension: int, /) -> "ComplexAffineNormalization":
        dimension_ = int(dimension)
        if dimension_ <= 0:
            raise ValueError("Complex normalization dimension must be positive.")
        return cls(
            np.zeros((dimension_,), dtype=np.complex128),
            np.eye(dimension_, dtype=np.complex128),
        )

    @classmethod
    def scalar(
        cls,
        *,
        center: complex = 0.0j,
        scale: complex = 1.0 + 0.0j,
    ) -> "ComplexAffineNormalization":
        scale_ = complex(scale)
        if not math.isfinite(abs(scale_)) or scale_ == 0.0j:
            raise ValueError(
                "Complex scalar normalization scale must be finite and nonzero."
            )
        return cls([complex(center)], [[1.0 / scale_]])

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"Complex normalization expected shape ({self.dimension},); got {values.shape}."
            )
        dtype = jnp.result_type(values, self.center, self.matrix)
        return self.matrix.astype(dtype) @ (
            values.astype(dtype) - self.center.astype(dtype)
        )


class HolomorphicMapCertificate(StrictModule, NonTrainableState):
    """Construction evidence that a complex map is holomorphic."""

    complex_input_size: int = eqx.field(static=True)
    complex_output_size: int = eqx.field(static=True)
    construction: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    operations: tuple[str, ...] = eqx.field(static=True)
    parameter_mode: str = eqx.field(static=True)
    parameter_coverage: HolomorphicParameterCoverage = eqx.field(static=True)
    linear_in_parameters: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        complex_input_size: int,
        complex_output_size: int,
        construction: str,
        normalization_id: str,
        maximum_derivative_order: int,
        operations: Sequence[str],
        parameter_coverage: HolomorphicParameterCoverage,
        linear_in_parameters: bool,
        parameter_mode: str = "real-cartesian",
    ):
        input_size = int(complex_input_size)
        output_size = int(complex_output_size)
        derivative_order = int(maximum_derivative_order)
        operations_ = tuple(str(value) for value in operations)
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Holomorphic map dimensions must be positive.")
        if derivative_order < 0:
            raise ValueError("maximum_derivative_order must be nonnegative.")
        if not construction or not normalization_id or not parameter_mode:
            raise ValueError("Holomorphic certificate identifiers must be nonempty.")
        if not operations_ or any(not value for value in operations_):
            raise ValueError("Holomorphic certificates require declared operations.")
        if parameter_coverage not in (
            "finite-subspace",
            "finite-parametric-family",
        ):
            raise ValueError("Unknown holomorphic parameter coverage.")
        if parameter_coverage == "finite-subspace" and not linear_in_parameters:
            raise ValueError(
                "Finite-subspace holomorphic maps must be linear in their parameters."
            )
        self.complex_input_size = input_size
        self.complex_output_size = output_size
        self.construction = str(construction)
        self.normalization_id = str(normalization_id)
        self.maximum_derivative_order = derivative_order
        self.operations = operations_
        self.parameter_mode = str(parameter_mode)
        self.parameter_coverage = parameter_coverage
        self.linear_in_parameters = bool(linear_in_parameters)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "holomorphic-map-certificate-v2",
                "complex_input_size": input_size,
                "complex_output_size": output_size,
                "construction": construction,
                "normalization_id": normalization_id,
                "maximum_derivative_order": derivative_order,
                "operations": list(operations_),
                "parameter_mode": parameter_mode,
                "parameter_coverage": parameter_coverage,
                "linear_in_parameters": bool(linear_in_parameters),
            }
        )


class HolomorphicJet(StrictModule):
    """Complex potential value and scalar-input derivatives by increasing order."""

    value: Array
    derivatives: tuple[Array, ...]

    def __init__(self, value: ArrayLike, derivatives: Sequence[ArrayLike], /):
        value_ = jnp.asarray(value)
        derivatives_ = tuple(jnp.asarray(item) for item in derivatives)
        if any(item.shape != value_.shape for item in derivatives_):
            raise ValueError("Holomorphic jet derivatives must match the value shape.")
        self.value = value_
        self.derivatives = derivatives_

    def derivative(self, order: int, /) -> Array:
        order_ = int(order)
        if order_ == 0:
            return self.value
        if order_ < 0 or order_ > len(self.derivatives):
            raise ValueError("Requested holomorphic derivative order is unavailable.")
        return self.derivatives[order_ - 1]


@runtime_checkable
class HolomorphicPotentialProvider(Protocol):
    def __call__(self, coordinates: Array, /) -> Array: ...

    def holomorphic_certificate(self) -> HolomorphicMapCertificate: ...

    def jet(self, coordinates: Array, order: int, /) -> HolomorphicJet: ...


__all__ = [
    "ComplexAffineNormalization",
    "HolomorphicJet",
    "HolomorphicMapCertificate",
    "HolomorphicParameterCoverage",
    "HolomorphicPotentialProvider",
]
