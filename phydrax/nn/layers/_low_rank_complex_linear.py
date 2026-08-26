#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from .._utils import _canonical_size, _get_size, _get_value_shape, SizeLike


class LowRankComplexLinearInitializationReport(StrictModule, NonTrainableState):
    """Spectral evidence for one truncated dense complex initializer."""

    input_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    requested_rank: int = eqx.field(static=True)
    realized_rank: int = eqx.field(static=True)
    retained_energy: float = eqx.field(static=True)
    relative_truncation_residual: float = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_count: int,
        output_count: int,
        requested_rank: int,
        realized_rank: int,
        retained_energy: float,
        relative_truncation_residual: float,
    ):
        counts = (
            int(input_count),
            int(output_count),
            int(requested_rank),
            int(realized_rank),
        )
        energy = float(retained_energy)
        residual = float(relative_truncation_residual)
        if any(value <= 0 for value in counts):
            raise ValueError("Low-rank complex initializer counts must be positive.")
        if counts[3] > counts[2] or counts[2] > min(counts[0], counts[1]):
            raise ValueError("Invalid low-rank complex initializer rank evidence.")
        if not 0.0 <= energy <= 1.0 or not 0.0 <= residual <= 1.0:
            raise ValueError("Low-rank initializer energy evidence must lie in [0, 1].")
        self.input_count = counts[0]
        self.output_count = counts[1]
        self.requested_rank = counts[2]
        self.realized_rank = counts[3]
        self.retained_energy = energy
        self.relative_truncation_residual = residual
        self.report_id = canonical_fingerprint(
            {
                "kind": "low-rank-complex-linear-initialization",
                "input_count": counts[0],
                "output_count": counts[1],
                "requested_rank": counts[2],
                "realized_rank": counts[3],
                "retained_energy": energy,
                "relative_truncation_residual": residual,
            }
        )


class LowRankComplexLinear(_AbstractBaseModel):
    """Complex affine map factored through one explicit low-rank complex space."""

    input_factor_real: Array
    input_factor_imag: Array
    output_factor_real: Array
    output_factor_imag: Array
    bias_real: Array | None
    bias_imag: Array | None
    in_size: SizeLike
    out_size: SizeLike
    rank: int = eqx.field(static=True)
    _in_value_shape: tuple[int, ...] = eqx.field(static=True)
    _out_value_shape: tuple[int, ...] = eqx.field(static=True)
    initialization: LowRankComplexLinearInitializationReport
    factorization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: SizeLike,
        rank: int,
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        in_size_ = _canonical_size(in_size)
        out_size_ = _canonical_size(out_size)
        in_shape = _get_value_shape(in_size_)
        out_shape = _get_value_shape(out_size_)
        input_count = _get_size(in_size_)
        output_count = _get_size(out_size_)
        if isinstance(rank, bool) or not isinstance(rank, Integral):
            raise TypeError("LowRankComplexLinear rank must be an integer.")
        rank_ = int(rank)
        if rank_ <= 0 or rank_ > min(input_count, output_count):
            raise ValueError(
                "LowRankComplexLinear rank must lie in [1, min(in_size, out_size)]."
            )
        real_key, imaginary_key = jr.split(key)
        component_scale = 1.0 / math.sqrt(float(input_count + output_count))
        dense = component_scale * (
            jr.normal(real_key, (output_count, input_count))
            + 1j * jr.normal(imaginary_key, (output_count, input_count))
        )
        dense_host = np.asarray(jax.device_get(dense), dtype=np.complex128)
        left, singular_values, right = np.linalg.svd(dense_host, full_matrices=False)
        retained = singular_values[:rank_]
        root = np.sqrt(retained)
        output_factor = left[:, :rank_] * root[None, :]
        input_factor = root[:, None] * right[:rank_, :]
        total_energy = float(np.sum(singular_values**2))
        retained_energy = (
            1.0 if total_energy == 0.0 else float(np.sum(retained**2) / total_energy)
        )
        residual = math.sqrt(max(1.0 - retained_energy, 0.0))
        tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * max(float(singular_values[0]), 1.0)
            * max(input_count, output_count)
        )
        realized_rank = int(np.count_nonzero(retained > tolerance))
        if realized_rank <= 0:
            raise RuntimeError(
                "Low-rank complex initializer is numerically rank deficient."
            )
        initialization = LowRankComplexLinearInitializationReport(
            input_count=input_count,
            output_count=output_count,
            requested_rank=rank_,
            realized_rank=realized_rank,
            retained_energy=retained_energy,
            relative_truncation_residual=residual,
        )
        real_dtype = dense.real.dtype
        self.input_factor_real = jnp.asarray(input_factor.real, dtype=real_dtype)
        self.input_factor_imag = jnp.asarray(input_factor.imag, dtype=real_dtype)
        self.output_factor_real = jnp.asarray(output_factor.real, dtype=real_dtype)
        self.output_factor_imag = jnp.asarray(output_factor.imag, dtype=real_dtype)
        self.bias_real = (
            jnp.zeros((output_count,), dtype=real_dtype) if use_bias else None
        )
        self.bias_imag = (
            jnp.zeros((output_count,), dtype=real_dtype) if use_bias else None
        )
        self.in_size = in_size_
        self.out_size = out_size_
        self.rank = rank_
        self._in_value_shape = in_shape
        self._out_value_shape = out_shape
        self.initialization = initialization
        self.factorization_id = canonical_fingerprint(
            {
                "kind": "low-rank-complex-linear",
                "in_shape": list(in_shape),
                "out_shape": list(out_shape),
                "rank": rank_,
                "use_bias": bool(use_bias),
                "initialization": initialization.report_id,
            }
        )

    @property
    def input_factor(self) -> Array:
        return self.input_factor_real + 1j * self.input_factor_imag

    @property
    def output_factor(self) -> Array:
        return self.output_factor_real + 1j * self.output_factor_imag

    @property
    def bias(self) -> Array | None:
        if self.bias_real is None or self.bias_imag is None:
            return None
        return self.bias_real + 1j * self.bias_imag

    def materialize_weight(self) -> Array:
        """Materialize the dense effective weight for diagnostics and export."""
        return contract("or,ri->oi", self.output_factor, self.input_factor)

    def __call__(
        self,
        value: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        array = jnp.asarray(value)
        in_shape = self._in_value_shape
        if in_shape:
            if array.ndim < len(in_shape) or array.shape[-len(in_shape) :] != in_shape:
                raise ValueError(
                    f"LowRankComplexLinear expected trailing shape {in_shape}; "
                    f"got {array.shape}."
                )
            leading = array.shape[: -len(in_shape)]
            flattened = array.reshape(leading + (_get_size(in_shape),))
        else:
            if array.shape == () or array.shape == (1,):
                leading = ()
                flattened = array.reshape((1,))
            else:
                leading = array.shape
                flattened = array.reshape(leading + (1,))
        latent = contract("ri,...i->...r", self.input_factor, flattened)
        output = contract("or,...r->...o", self.output_factor, latent)
        bias = self.bias
        if bias is not None:
            output = output + bias
        out_shape = self._out_value_shape
        if out_shape:
            return output.reshape(leading + out_shape)
        if int(output.shape[-1]) != 1:
            raise ValueError("Scalar LowRankComplexLinear output requires one feature.")
        return jnp.squeeze(output, axis=-1)


__all__ = [
    "LowRankComplexLinear",
    "LowRankComplexLinearInitializationReport",
]
