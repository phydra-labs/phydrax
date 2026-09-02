#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._symmetric_coordinates import smat, svec, symmetric_packed_dimension
from ..._trainable import NonTrainableState
from ...linalg import HermitianPrecisionPolicy
from .._base import _AbstractBaseModel
from .._initializers import _initializer_dict
from .._keys import EvalKey
from .._utils import _canonical_size, _get_size, _get_value_shape, SizeLike
from ..parameters import PositiveSemidefiniteTransform


_Monotonicity = Literal["free", "increasing", "decreasing"]
_MONOTONICITY_MODES = frozenset(("free", "increasing", "decreasing"))
_PSD_TRANSFORM = PositiveSemidefiniteTransform()


def _fingerprint_float(value: float, /) -> float | str:
    return value if math.isfinite(value) else "infinite"


def _diagonal_matrix(diagonal: Array, /) -> Array:
    dimension = int(diagonal.shape[-1])
    identity = jnp.eye(dimension, dtype=diagonal.dtype)
    return diagonal[..., :, None] * identity


def _pack_diagonal_factor(diagonal: Array, /) -> Array:
    dimension = int(diagonal.shape[-1])
    factor = _diagonal_matrix(jnp.sqrt(diagonal))
    rows, columns = jnp.tril_indices(dimension)
    return factor[..., rows, columns]


class SpectralNeuronInitializationReport(StrictModule, NonTrainableState):
    """Fresh-state eigengap evidence for a spectral-neuron initializer."""

    feature_count: int = eqx.field(static=True)
    matrix_size: int = eqx.field(static=True)
    eigen_index: int = eqx.field(static=True)
    initialization_radius: float = eqx.field(static=True)
    nominal_origin_gap: float = eqx.field(static=True)
    origin_gap: float = eqx.field(static=True)
    jitter_bound: float = eqx.field(static=True)
    perturbation_bound: float = eqx.field(static=True)
    certified_minimum_gap: float = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        feature_count: int,
        matrix_size: int,
        eigen_index: int,
        initialization_radius: float,
        nominal_origin_gap: float,
        origin_gap: float,
        jitter_bound: float,
        perturbation_bound: float,
        certified_minimum_gap: float,
    ):
        features = int(feature_count)
        dimension = int(matrix_size)
        index = int(eigen_index)
        radius = float(initialization_radius)
        nominal = float(nominal_origin_gap)
        origin = float(origin_gap)
        jitter = float(jitter_bound)
        perturbation = float(perturbation_bound)
        certified = float(certified_minimum_gap)
        if features <= 0 or dimension <= 0 or not 0 <= index < dimension:
            raise ValueError("Invalid spectral-neuron initialization dimensions.")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("initialization_radius must be finite and positive.")
        if math.isnan(nominal) or nominal <= 0.0:
            raise ValueError("nominal_origin_gap must be positive.")
        if math.isnan(origin) or origin <= 0.0:
            raise ValueError("origin_gap must be positive.")
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("jitter_bound must be finite and nonnegative.")
        if not math.isfinite(perturbation) or perturbation < 0.0:
            raise ValueError("perturbation_bound must be finite and nonnegative.")
        if math.isnan(certified) or certified < 0.0:
            raise ValueError("certified_minimum_gap must be nonnegative.")
        self.feature_count = features
        self.matrix_size = dimension
        self.eigen_index = index
        self.initialization_radius = radius
        self.nominal_origin_gap = nominal
        self.origin_gap = origin
        self.jitter_bound = jitter
        self.perturbation_bound = perturbation
        self.certified_minimum_gap = certified
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-neuron-initialization",
                "feature_count": features,
                "matrix_size": dimension,
                "eigen_index": index,
                "initialization_radius": radius,
                "nominal_origin_gap": _fingerprint_float(nominal),
                "origin_gap": _fingerprint_float(origin),
                "jitter_bound": jitter,
                "perturbation_bound": perturbation,
                "certified_minimum_gap": _fingerprint_float(certified),
            }
        )


class SpectralNeuron(_AbstractBaseModel):
    """Selected eigenvalue of a trainable affine real-symmetric matrix pencil.

    The ascending eigenvalue at ``eigen_index`` is evaluated from
    ``A0 + sum_i x_i Ai``. Per-feature monotonicity is enforced by constructing
    increasing coefficients as positive semidefinite matrices and decreasing
    coefficients as negative semidefinite matrices.

    The initialization report applies only to the fresh parameters and to the
    declared infinity-norm input box. Optimizer updates invalidate that report.
    """

    free_coordinates: Array
    increasing_factor_coordinates: Array
    decreasing_factor_coordinates: Array
    in_size: SizeLike
    out_size: SizeLike
    matrix_size: int = eqx.field(static=True)
    eigen_index: int = eqx.field(static=True)
    monotonicity: tuple[_Monotonicity, ...] = eqx.field(static=True)
    initialization: SpectralNeuronInitializationReport
    precision: HermitianPrecisionPolicy
    _input_count: int = eqx.field(static=True)
    _in_value_shape: tuple[int, ...] = eqx.field(static=True)
    _free_indices: tuple[int, ...] = eqx.field(static=True)
    _increasing_indices: tuple[int, ...] = eqx.field(static=True)
    _decreasing_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_size: SizeLike,
        matrix_size: int,
        eigen_index: int,
        monotonicity: Sequence[_Monotonicity] | _Monotonicity | None = None,
        initialization_radius: float = 5.0,
        dtype: Any = jnp.float32,
        precision: HermitianPrecisionPolicy | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if isinstance(matrix_size, bool) or not isinstance(matrix_size, Integral):
            raise TypeError("SpectralNeuron matrix_size must be an integer.")
        if isinstance(eigen_index, bool) or not isinstance(eigen_index, Integral):
            raise TypeError("SpectralNeuron eigen_index must be an integer.")
        dimension = int(matrix_size)
        index = int(eigen_index)
        if dimension <= 0:
            raise ValueError("SpectralNeuron matrix_size must be positive.")
        if not 0 <= index < dimension:
            raise ValueError("SpectralNeuron eigen_index must lie in [0, matrix_size).")
        parameter_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(parameter_dtype, jnp.floating):
            raise TypeError("SpectralNeuron parameters require a real floating dtype.")
        radius = float(initialization_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("initialization_radius must be finite and positive.")
        if precision is not None and not isinstance(precision, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")

        in_size_ = _canonical_size(in_size)
        in_shape = _get_value_shape(in_size_)
        input_count = _get_size(in_size_)
        if monotonicity is None:
            modes = ("free",) * input_count
        elif isinstance(monotonicity, str):
            modes = (monotonicity,)
        else:
            modes = tuple(monotonicity)
        if len(modes) != input_count:
            raise ValueError(
                "SpectralNeuron monotonicity must have one entry per input feature."
            )
        if any(mode not in _MONOTONICITY_MODES for mode in modes):
            raise ValueError(
                "SpectralNeuron monotonicity entries must be 'free', "
                "'increasing', or 'decreasing'."
            )
        canonical_modes = tuple(modes)
        free_indices = tuple(
            i for i, mode in enumerate(canonical_modes) if mode == "free"
        )
        increasing_indices = tuple(
            i for i, mode in enumerate(canonical_modes) if mode == "increasing"
        )
        decreasing_indices = tuple(
            i for i, mode in enumerate(canonical_modes) if mode == "decreasing"
        )

        base_key, slope_key, jitter_key = jr.split(key, 3)
        orthogonal = _initializer_dict["orthogonal"](
            base_key, (dimension, dimension), parameter_dtype
        )
        positions = jnp.arange(dimension)
        base_spectrum = jnp.where(
            positions < index,
            jnp.asarray(-1.0, dtype=parameter_dtype),
            jnp.where(
                positions > index,
                jnp.asarray(1.0, dtype=parameter_dtype),
                jnp.asarray(0.0, dtype=parameter_dtype),
            ),
        )
        base_matrix = contract("ai,i,bi->ab", orthogonal, base_spectrum, orthogonal)

        scale = 1.0 / math.sqrt(float(input_count))
        slope_samples = jr.uniform(
            slope_key,
            (input_count,),
            dtype=parameter_dtype,
            minval=-1.0,
            maxval=1.0,
        )
        free_slopes = jnp.asarray(scale, dtype=parameter_dtype) * slope_samples
        constrained_slopes = jnp.asarray(scale, dtype=parameter_dtype) * (
            0.5 + 0.5 * jnp.abs(slope_samples)
        )
        delta = 1.0 / (4.0 * float(input_count) * radius)
        raw_jitter = jr.uniform(
            jitter_key,
            (input_count, dimension),
            dtype=parameter_dtype,
            minval=-delta,
            maxval=delta,
        )
        free_diagonal = free_slopes[:, None] + raw_jitter
        constrained_jitter = jnp.abs(raw_jitter)
        constrained_diagonal = constrained_slopes[:, None] + constrained_jitter

        packed_size = symmetric_packed_dimension(dimension)
        free_feature_coordinates = svec(_diagonal_matrix(free_diagonal))
        factor_coordinates = _pack_diagonal_factor(constrained_diagonal)
        free_take = jnp.asarray(free_indices, dtype=jnp.int32)
        increasing_take = jnp.asarray(increasing_indices, dtype=jnp.int32)
        decreasing_take = jnp.asarray(decreasing_indices, dtype=jnp.int32)
        base_coordinates = svec(base_matrix)
        self.free_coordinates = jnp.concatenate(
            (base_coordinates[None, :], free_feature_coordinates[free_take]), axis=0
        )
        self.increasing_factor_coordinates = factor_coordinates[increasing_take].reshape(
            (len(increasing_indices), packed_size)
        )
        self.decreasing_factor_coordinates = factor_coordinates[decreasing_take].reshape(
            (len(decreasing_indices), packed_size)
        )

        if dimension == 1:
            nominal_gap = math.inf
            origin_gap = math.inf
            certified_gap = math.inf
        else:
            origin_values = np.asarray(
                jax.device_get(jnp.linalg.eigvalsh(base_matrix)), dtype=np.float64
            )
            adjacent = []
            if index > 0:
                adjacent.append(float(origin_values[index] - origin_values[index - 1]))
            if index + 1 < dimension:
                adjacent.append(float(origin_values[index + 1] - origin_values[index]))
            nominal_gap = 1.0
            origin_gap = min(adjacent)
            jitter_host = np.asarray(jax.device_get(raw_jitter), dtype=np.float64)
            per_feature_jitter = np.max(np.abs(jitter_host), axis=-1)
            perturbation_bound = radius * float(np.sum(per_feature_jitter))
            certified_gap = max(origin_gap - 2.0 * perturbation_bound, 0.0)
        if dimension == 1:
            perturbation_bound = radius * float(
                np.sum(
                    np.max(
                        np.abs(np.asarray(jax.device_get(raw_jitter), dtype=np.float64)),
                        axis=-1,
                    )
                )
            )
        initialization = SpectralNeuronInitializationReport(
            feature_count=input_count,
            matrix_size=dimension,
            eigen_index=index,
            initialization_radius=radius,
            nominal_origin_gap=nominal_gap,
            origin_gap=origin_gap,
            jitter_bound=delta,
            perturbation_bound=perturbation_bound,
            certified_minimum_gap=certified_gap,
        )

        self.in_size = in_size_
        self.out_size = "scalar"
        self.matrix_size = dimension
        self.eigen_index = index
        self.monotonicity = canonical_modes
        self.initialization = initialization
        self.precision = HermitianPrecisionPolicy() if precision is None else precision
        self._input_count = input_count
        self._in_value_shape = in_shape
        self._free_indices = free_indices
        self._increasing_indices = increasing_indices
        self._decreasing_indices = decreasing_indices

    @property
    def is_convex(self) -> bool:
        """Whether the selected eigenvalue is globally convex."""
        return self.eigen_index == self.matrix_size - 1

    @property
    def is_concave(self) -> bool:
        """Whether the selected eigenvalue is globally concave."""
        return self.eigen_index == 0

    def materialize_coefficients(self) -> tuple[Array, Array]:
        """Return the effective intercept and ordered feature matrices."""
        base = smat(self.free_coordinates[0], matrix_dimension=self.matrix_size)
        feature_dtype = jnp.result_type(
            self.free_coordinates,
            self.increasing_factor_coordinates,
            self.decreasing_factor_coordinates,
        )
        features = jnp.zeros(
            (self._input_count, self.matrix_size, self.matrix_size),
            dtype=feature_dtype,
        )
        if self._free_indices:
            free = smat(self.free_coordinates[1:], matrix_dimension=self.matrix_size)
            features = features.at[jnp.asarray(self._free_indices)].set(free)
        if self._increasing_indices:
            increasing = _PSD_TRANSFORM(self.increasing_factor_coordinates)
            features = features.at[jnp.asarray(self._increasing_indices)].set(increasing)
        if self._decreasing_indices:
            decreasing = -_PSD_TRANSFORM(self.decreasing_factor_coordinates)
            features = features.at[jnp.asarray(self._decreasing_indices)].set(decreasing)
        return base, features

    def _flatten_input(self, value: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError("SpectralNeuron requires real-valued inputs.")
        in_shape = self._in_value_shape
        if in_shape:
            if array.ndim < len(in_shape) or array.shape[-len(in_shape) :] != in_shape:
                raise ValueError(
                    f"SpectralNeuron expected trailing shape {in_shape}; got {array.shape}."
                )
            leading = array.shape[: -len(in_shape)]
            flattened = array.reshape(leading + (self._input_count,))
        else:
            if array.shape == () or array.shape == (1,):
                leading = ()
                flattened = array.reshape((1,))
            else:
                leading = array.shape
                flattened = array.reshape(leading + (1,))
        dtype = jnp.result_type(flattened, self.free_coordinates)
        return flattened.astype(dtype), leading

    def matrix_pencil(self, value: ArrayLike, /) -> Array:
        """Evaluate the affine real-symmetric matrix pencil."""
        flattened, _ = self._flatten_input(value)
        packed = self.free_coordinates[0]
        if self._free_indices:
            free_input = flattened[..., jnp.asarray(self._free_indices)]
            packed = packed + contract(
                "...i,ip->...p", free_input, self.free_coordinates[1:]
            )
        matrix = smat(packed, matrix_dimension=self.matrix_size)
        if self._increasing_indices:
            increasing_input = flattened[..., jnp.asarray(self._increasing_indices)]
            increasing = _PSD_TRANSFORM(self.increasing_factor_coordinates)
            matrix = matrix + contract("...i,ijk->...jk", increasing_input, increasing)
        if self._decreasing_indices:
            decreasing_input = flattened[..., jnp.asarray(self._decreasing_indices)]
            decreasing = _PSD_TRANSFORM(self.decreasing_factor_coordinates)
            matrix = matrix - contract("...i,ijk->...jk", decreasing_input, decreasing)
        return matrix

    def eigenvalues(self, value: ArrayLike, /) -> Array:
        """Return the full ascending eigenvalue spectrum of the evaluated pencil."""
        matrix = self.precision.compute(self.matrix_pencil(value))
        matrix = self.precision.factorization(matrix)
        return self.precision.output(jnp.linalg.eigvalsh(matrix))

    def __call__(
        self,
        value: ArrayLike,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        return self.eigenvalues(value)[..., self.eigen_index]


__all__ = ["SpectralNeuron", "SpectralNeuronInitializationReport"]
