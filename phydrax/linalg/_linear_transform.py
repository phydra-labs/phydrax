#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.fft as jfft
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._spaces import ArraySpace


TrigonometricTransformKind: TypeAlias = Literal["dct", "dst"]


class AbstractLinearTransform(StrictModule, NonTrainableState):
    """Invertible linear map between declared physical and modal spaces."""

    physical_space: AbstractAttribute[ArraySpace]
    modal_space: AbstractAttribute[ArraySpace]
    transform_id: AbstractAttribute[str]

    @abc.abstractmethod
    def analyze(self, values: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        raise NotImplementedError


class DenseLinearTransform(AbstractLinearTransform):
    """Explicit analysis/synthesis matrices for small or irregular bases."""

    physical_space: ArraySpace
    modal_space: ArraySpace
    analysis: Array
    synthesis: Array
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        analysis: ArrayLike,
        synthesis: ArrayLike,
        /,
        *,
        transform_id: str | None = None,
    ):
        analysis_ = jnp.asarray(analysis)
        synthesis_ = jnp.asarray(synthesis)
        if analysis_.ndim != 2 or synthesis_.shape != (
            analysis_.shape[1],
            analysis_.shape[0],
        ):
            raise ValueError("Dense transform matrices must have transposed shapes.")
        self.physical_space = ArraySpace(
            (int(analysis_.shape[1]),), dtype=synthesis_.dtype
        )
        self.modal_space = ArraySpace((int(analysis_.shape[0]),), dtype=analysis_.dtype)
        self.analysis = analysis_
        self.synthesis = synthesis_
        self.transform_id = (
            canonical_fingerprint(
                {
                    "kind": "dense-linear-transform",
                    "analysis": array_tree_fingerprint(analysis_),
                    "synthesis": array_tree_fingerprint(synthesis_),
                }
            )
            if transform_id is None
            else str(transform_id)
        )
        if not self.transform_id:
            raise ValueError("transform_id must be non-empty.")

    def analyze(self, values: ArrayLike, /) -> Array:
        return self.analysis @ self.physical_space.validate(jnp.asarray(values))

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        return self.synthesis @ self.modal_space.validate(jnp.asarray(coefficients))


class FFTLinearTransform(AbstractLinearTransform):
    """Orthonormal complex FFT without dense transform storage."""

    physical_space: ArraySpace
    modal_space: ArraySpace
    transform_id: str = eqx.field(static=True)

    def __init__(self, count: int, /, *, dtype: Any = complex):
        size = int(count)
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if size <= 0:
            raise ValueError("FFT count must be positive.")
        if not jnp.issubdtype(dtype_, jnp.complexfloating):
            raise TypeError("FFT transforms require a complex dtype.")
        self.physical_space = ArraySpace((size,), dtype=dtype_)
        self.modal_space = ArraySpace((size,), dtype=dtype_)
        self.transform_id = canonical_fingerprint(
            {
                "kind": "fft-linear-transform",
                "count": size,
                "dtype": dtype_.str,
                "normalization": "ortho",
            }
        )

    def analyze(self, values: ArrayLike, /) -> Array:
        value = self.physical_space.validate(
            jnp.asarray(values, dtype=self.physical_space.dtype)
        )
        return jnp.fft.fft(value, norm="ortho")

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        value = self.modal_space.validate(
            jnp.asarray(coefficients, dtype=self.modal_space.dtype)
        )
        return jnp.fft.ifft(value, norm="ortho")


class RealTrigonometricTransform(AbstractLinearTransform):
    """Orthonormal DCT/DST-I–IV using JAX FFT/DCT primitives."""

    physical_space: ArraySpace
    modal_space: ArraySpace
    kind: TrigonometricTransformKind = eqx.field(static=True)
    transform_type: int = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    _dtype: np.dtype = eqx.field(static=True)

    def __init__(
        self,
        kind: TrigonometricTransformKind,
        transform_type: Literal[1, 2, 3, 4],
        count: int,
        /,
        *,
        dtype: Any = float,
    ):
        size = int(count)
        type_ = int(transform_type)
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        invalid_size = size < 1 or (kind == "dct" and type_ == 1 and size < 2)
        if kind not in ("dct", "dst") or type_ not in (1, 2, 3, 4) or invalid_size:
            raise ValueError("Trigonometric transform kind/type/count is invalid.")
        if not jnp.issubdtype(dtype_, jnp.floating):
            raise TypeError("Real trigonometric transforms require a floating dtype.")
        space = ArraySpace((size,), dtype=dtype_)
        self.physical_space = space
        self.modal_space = space
        self.kind = kind
        self.transform_type = type_
        self._dtype = dtype_
        self.transform_id = canonical_fingerprint(
            {
                "kind": "real-trigonometric-transform",
                "family": kind,
                "type": type_,
                "count": size,
                "dtype": dtype_.str,
                "normalization": "ortho",
            }
        )

    def _dct1(self, values: Array) -> Array:
        size = values.size
        scaled = values.at[0].multiply(jnp.sqrt(2.0))
        scaled = scaled.at[-1].multiply(jnp.sqrt(2.0))
        extended = jnp.concatenate((scaled, scaled[-2:0:-1]))
        transformed = jnp.real(jnp.fft.fft(extended))[:size]
        transformed = transformed / jnp.sqrt(2.0 * (size - 1.0))
        transformed = transformed.at[0].divide(jnp.sqrt(2.0))
        return transformed.at[-1].divide(jnp.sqrt(2.0))

    def _dct4(self, values: Array) -> Array:
        size = values.size
        indices = jnp.arange(size, dtype=self._dtype)
        phase = jnp.exp(1j * jnp.pi * indices / (2.0 * size))
        padded = jnp.concatenate((values * phase, jnp.zeros((size,), dtype=phase.dtype)))
        sums = jnp.fft.ifft(padded)[:size] * (2.0 * size)
        output_phase = jnp.exp(
            1j * jnp.pi * (indices / (2.0 * size) + 1.0 / (4.0 * size))
        )
        return jnp.sqrt(2.0 / size) * jnp.real(output_phase * sums)

    def _alternating_sign(self) -> Array:
        return (-1.0) ** jnp.arange(self.physical_space.size, dtype=self._dtype)

    def _dst2(self, values: Array) -> Array:
        transformed = jfft.dct(
            self._alternating_sign() * values,
            type=2,
            norm="ortho",
        )
        return transformed[::-1]

    def _idst2(self, values: Array) -> Array:
        return self._alternating_sign() * jfft.idct(
            values[::-1],
            type=2,
            norm="ortho",
        )

    def _dst1(self, values: Array) -> Array:
        size = values.size
        zero = jnp.zeros((1,), dtype=self._dtype)
        extended = jnp.concatenate((zero, values, zero, -values[::-1]))
        unnormalized = -jnp.imag(jnp.fft.fft(extended))[1 : size + 1]
        return unnormalized / jnp.sqrt(2.0 * (size + 1.0))

    def _dct(self, values: Array, *, inverse: bool) -> Array:
        if self.transform_type == 1:
            return self._dct1(values)
        if self.transform_type == 2:
            function = jfft.idct if inverse else jfft.dct
            return function(values, type=2, norm="ortho")
        if self.transform_type == 3:
            function = jfft.dct if inverse else jfft.idct
            return function(values, type=2, norm="ortho")
        return self._dct4(values)

    def _apply(self, values: Array, *, inverse: bool) -> Array:
        if self.kind == "dct":
            return self._dct(values, inverse=inverse)
        if self.transform_type == 1:
            return self._dst1(values)
        if self.transform_type == 2:
            return self._idst2(values) if inverse else self._dst2(values)
        if self.transform_type == 3:
            return self._dst2(values) if inverse else self._idst2(values)
        return self._dct4(self._alternating_sign() * values)[::-1]

    def _validated_apply(self, values: ArrayLike, *, inverse: bool) -> Array:
        value = jnp.asarray(values)
        if value.shape != self.physical_space.shape:
            raise ValueError(
                f"Trigonometric values must have shape {self.physical_space.shape}."
            )
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            real = self._apply(
                jnp.asarray(jnp.real(value), dtype=self._dtype), inverse=inverse
            )
            imag = self._apply(
                jnp.asarray(jnp.imag(value), dtype=self._dtype), inverse=inverse
            )
            return real + 1j * imag
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(
                "Trigonometric values must have real or complex inexact dtype."
            )
        return self._apply(jnp.asarray(value, dtype=self._dtype), inverse=inverse)

    def analyze(self, values: ArrayLike, /) -> Array:
        return self._validated_apply(values, inverse=False)

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        return self._validated_apply(coefficients, inverse=True)


class SimilarityScaledLinearTransform(AbstractLinearTransform):
    """Fast similarity transform with one nonzero physical-coordinate scaling."""

    base: AbstractLinearTransform
    scaling: Array
    physical_space: ArraySpace
    modal_space: ArraySpace
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractLinearTransform,
        scaling: ArrayLike,
        /,
    ):
        if not isinstance(base, AbstractLinearTransform):
            raise TypeError("base must be an AbstractLinearTransform.")
        if not isinstance(base.physical_space, ArraySpace):
            raise TypeError("Similarity scaling currently requires an ArraySpace.")
        scaling_ = jnp.asarray(scaling, dtype=base.physical_space.dtype)
        if scaling_.shape != base.physical_space.shape:
            raise ValueError(
                "Similarity scaling must match the physical transform shape."
            )
        if not bool(np.all(np.isfinite(np.asarray(scaling_)))) or bool(
            np.any(np.asarray(scaling_) == 0)
        ):
            raise ValueError("Similarity scaling must be finite and nonzero.")
        self.base = base
        self.scaling = scaling_
        self.physical_space = base.physical_space
        self.modal_space = base.modal_space
        self.transform_id = canonical_fingerprint(
            {
                "kind": "similarity-scaled-linear-transform",
                "base": base.transform_id,
                "scaling": array_tree_fingerprint(scaling_),
            }
        )

    def analyze(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape != self.physical_space.shape:
            raise ValueError("Similarity-scaled physical values have wrong shape.")
        return self.base.analyze(self.scaling * value)

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        return self.base.synthesize(coefficients) / self.scaling


class TensorLinearTransform(AbstractLinearTransform):
    """Separable per-axis composition of fast or dense one-dimensional transforms."""

    transforms: tuple[AbstractLinearTransform, ...]
    physical_space: ArraySpace
    modal_space: ArraySpace
    transform_id: str = eqx.field(static=True)

    def __init__(self, transforms: Sequence[AbstractLinearTransform], /):
        values = tuple(transforms)
        if not values or not all(
            isinstance(value, AbstractLinearTransform) for value in values
        ):
            raise TypeError("transforms must contain AbstractLinearTransform values.")
        if not all(
            isinstance(value.physical_space, ArraySpace)
            and isinstance(value.modal_space, ArraySpace)
            and len(value.physical_space.shape) == 1
            and len(value.modal_space.shape) == 1
            for value in values
        ):
            raise ValueError(
                "Tensor transforms require one-dimensional ArraySpace factors."
            )
        self.transforms = values
        self.physical_space = ArraySpace(
            tuple(value.physical_space.shape[0] for value in values),
            dtype=jnp.result_type(*[value.physical_space.dtype for value in values]),
        )
        self.modal_space = ArraySpace(
            tuple(value.modal_space.shape[0] for value in values),
            dtype=jnp.result_type(*[value.modal_space.dtype for value in values]),
        )
        self.transform_id = canonical_fingerprint(
            {
                "kind": "tensor-linear-transform",
                "transforms": [value.transform_id for value in values],
            }
        )

    def analyze(self, values: ArrayLike, /) -> Array:
        result = self.physical_space.validate(
            jnp.asarray(values, dtype=self.physical_space.dtype)
        )
        for axis, transform in enumerate(self.transforms):
            moved = jnp.moveaxis(result, axis, -1)
            flattened = moved.reshape((-1, moved.shape[-1]))
            transformed = jax.vmap(transform.analyze)(flattened)
            moved = transformed.reshape(moved.shape[:-1] + (transformed.shape[-1],))
            result = jnp.moveaxis(moved, -1, axis)
        return result

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        result = self.modal_space.validate(
            jnp.asarray(coefficients, dtype=self.modal_space.dtype)
        )
        for axis in reversed(range(len(self.transforms))):
            transform = self.transforms[axis]
            moved = jnp.moveaxis(result, axis, -1)
            flattened = moved.reshape((-1, moved.shape[-1]))
            transformed = jax.vmap(transform.synthesize)(flattened)
            moved = transformed.reshape(moved.shape[:-1] + (transformed.shape[-1],))
            result = jnp.moveaxis(moved, -1, axis)
        return result


__all__ = [
    "AbstractLinearTransform",
    "DenseLinearTransform",
    "FFTLinearTransform",
    "RealTrigonometricTransform",
    "SimilarityScaledLinearTransform",
    "TensorLinearTransform",
    "TrigonometricTransformKind",
]
