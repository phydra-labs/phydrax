#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import FunctionSamples, OperatorBatch


class LaplaceTemporalOperator(_AbstractOperatorModel):
    """Stable causal pole-residue operator for nonperiodic transient dynamics.

    Stored poles represent one member of a complex-conjugate pair. The response is
    reconstructed as twice the real part of the paired contribution, which gives
    real outputs without allowing optimization to break conjugacy. Negative real
    pole parts are enforced by a softplus decay parameterization.
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    num_poles: int
    log_decay: Array
    frequency: Array
    residue: Array
    direct_weight: Array
    bias: Array
    min_decay: float
    source_key: str | None

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        num_poles: int = 16,
        min_decay: float = 1e-4,
        max_initial_frequency: float = 8.0,
        source_key: str | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.num_poles = int(num_poles)
        self.min_decay = float(min_decay)
        self.source_key = source_key
        if self.num_poles <= 0 or self.min_decay <= 0.0:
            raise ValueError("num_poles and min_decay must be positive.")
        if float(max_initial_frequency) < 0.0:
            raise ValueError("max_initial_frequency must be non-negative.")

        in_count = _get_size(in_channels)
        out_count = _get_size(out_channels)
        decay_key, real_key, imag_key, direct_key = jr.split(key, 4)
        initial_decay = jr.uniform(
            decay_key,
            shape=(self.num_poles,),
            minval=0.1,
            maxval=2.0,
        )
        self.log_decay = jnp.log(jnp.expm1(initial_decay))
        self.frequency = jnp.linspace(
            0.0,
            float(max_initial_frequency),
            self.num_poles,
        )
        scale = 1.0 / jnp.sqrt(float(self.num_poles * in_count))
        self.residue = scale * (
            jr.normal(real_key, shape=(self.num_poles, in_count, out_count))
            + 1j * jr.normal(imag_key, shape=(self.num_poles, in_count, out_count))
        )
        self.direct_weight = jr.normal(
            direct_key, shape=(in_count, out_count)
        ) / jnp.sqrt(float(in_count))
        self.bias = jnp.zeros((out_count,), dtype=float)

    def poles(self, /) -> Array:
        """Return the constrained poles; every real part is strictly negative."""
        decay = jax.nn.softplus(self.log_decay) + self.min_decay
        return -decay + 1j * self.frequency

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("source_key is required for multiple operator inputs.")
        return next(iter(batch.inputs.values()))

    @staticmethod
    def _times(
        samples: FunctionSamples,
        name: str,
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        if samples.axes and len(samples.axes) != 1:
            raise ValueError(f"{name} must have exactly one temporal axis.")
        if not samples.axes and (
            samples.coordinates is None or int(samples.coordinates.shape[-1]) != 1
        ):
            raise ValueError(f"{name} must provide one-dimensional time coordinates.")
        coordinates = samples.coordinates_array(
            case_shape=case_shape,
            flatten=True,
        )
        cases = prod(case_shape) if case_shape else 1
        return coordinates.reshape((cases, samples.sample_shape[0], -1))[..., 0]

    def _values(
        self,
        source: FunctionSamples,
        /,
        *,
        case_ndim: int,
    ) -> tuple[Array, tuple[int, ...]]:
        if source.values is None:
            raise ValueError("Laplace operator source values cannot be None.")
        values = jnp.asarray(source.values)
        sample_shape = source.sample_shape
        if len(sample_shape) != 1:
            raise ValueError("Laplace operator source must have one sample axis.")
        if int(values.shape[case_ndim]) != sample_shape[0]:
            raise ValueError("Source values do not align with the temporal sample axis.")
        case_shape = tuple(int(size) for size in values.shape[:case_ndim])
        trailing = tuple(int(size) for size in values.shape[case_ndim + 1 :])
        if not trailing:
            values = values[..., None]
        elif trailing != (_get_size(self.in_size),):
            raise ValueError(
                f"Expected {_get_size(self.in_size)} input channels, got {trailing}."
            )
        return values.reshape(
            (
                prod(case_shape) if case_shape else 1,
                sample_shape[0],
                _get_size(self.in_size),
            )
        ), case_shape

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        source = self._source(batch)
        source_time = self._times(source, "source", batch.case_shape)
        query_time = self._times(batch.require_single_query(), "query", batch.case_shape)
        if int(source_time.shape[1]) < 2:
            raise ValueError("Laplace operator requires at least two source times.")
        values, case_shape = self._values(source, case_ndim=len(batch.case_axes))
        source_mask = source.mask_array(case_shape=case_shape).reshape(source_time.shape)
        valid_interval = source_mask[:, :-1] & source_mask[:, 1:]
        invalid_order = jnp.any((jnp.diff(source_time, axis=1) <= 0.0) & valid_interval)
        source_time = eqx.error_if(
            source_time,
            invalid_order,
            "Valid source times must be strictly increasing.",
        )

        left_time = source_time[:, :-1]
        right_time = source_time[:, 1:]
        query = query_time[:, :, None]
        clipped_right = jnp.minimum(query, right_time[:, None, :])
        partial_width = jnp.maximum(clipped_right - left_time[:, None, :], 0.0)
        active = query > left_time[:, None, :]
        partial_width = partial_width * active * valid_interval[:, None, :]

        full_interval = query >= right_time[:, None, :]
        left_values = values[:, :-1, :]
        right_values = jnp.where(
            full_interval[..., None],
            values[:, None, 1:, :],
            values[:, None, :-1, :],
        )
        poles = self.poles()
        left_kernel = jnp.exp(
            (query - left_time[:, None, :])[..., None] * poles[None, None, None, :]
        )
        right_kernel = jnp.exp(
            (query - clipped_right)[..., None] * poles[None, None, None, :]
        )
        left_weight = 0.5 * partial_width[..., None] * left_kernel
        right_weight = 0.5 * partial_width[..., None] * right_kernel
        left_response = jnp.einsum(
            "cqnp,cni,pio->cqo",
            left_weight,
            left_values,
            self.residue,
        )
        right_response = jnp.einsum(
            "cqnp,cqni,pio->cqo",
            right_weight,
            right_values,
            self.residue,
        )
        response = 2.0 * jnp.real(left_response + right_response)

        # Strictly causal zero-order-hold feedthrough: no future sample is consulted.
        past = (source_time[:, None, :] <= query_time[:, :, None]) & source_mask[
            :, None, :
        ]
        indices = jnp.maximum(jnp.sum(past, axis=2) - 1, 0)
        has_past = jnp.any(past, axis=2)
        held = jnp.take_along_axis(
            values,
            indices[..., None],
            axis=1,
        )
        held = held * has_past[..., None]
        response = response + jnp.einsum("cqi,io->cqo", held, self.direct_weight)
        response = response + self.bias
        query_mask = batch.require_single_query().mask_array(case_shape=case_shape).reshape(
            query_time.shape
        )
        response = response * query_mask[..., None]
        output = response.reshape(
            case_shape + batch.require_single_query().sample_shape + (_get_size(self.out_size),)
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def recurrent(
        self,
        batch: OperatorBatch,
        /,
    ) -> Array:
        """Evaluate aligned source/query times with a linear-memory recurrence."""
        source = self._source(batch)
        source_time = self._times(source, "source", batch.case_shape)
        query_time = self._times(batch.require_single_query(), "query", batch.case_shape)
        if source_time.shape != query_time.shape:
            raise ValueError(
                "Recurrent Laplace execution requires query times aligned with source times."
            )
        source_time = eqx.error_if(
            source_time,
            jnp.any(~jnp.isclose(source_time, query_time, rtol=1e-10, atol=1e-12)),
            "Recurrent Laplace execution requires query times aligned with source times.",
        )
        values, case_shape = self._values(
            source,
            case_ndim=len(batch.case_axes),
        )
        source_mask = source.mask_array(case_shape=case_shape).reshape(source_time.shape)
        query_mask = batch.require_single_query().mask_array(case_shape=case_shape).reshape(
            query_time.shape
        )
        valid_interval = source_mask[:, :-1] & source_mask[:, 1:]
        delta = jnp.diff(source_time, axis=1)
        source_time = eqx.error_if(
            source_time,
            jnp.any((delta <= 0.0) & valid_interval),
            "Valid source times must be strictly increasing.",
        )
        poles = self.poles()
        state = jnp.zeros(
            (
                values.shape[0],
                self.num_poles,
                _get_size(self.in_size),
            ),
            dtype=self.residue.dtype,
        )

        def step(current_state, inputs):
            width, left, right, valid = inputs
            transition = jnp.exp(width[:, None] * poles[None, :])
            increment = (
                0.5
                * width[:, None, None]
                * (transition[..., None] * left[:, None, :] + right[:, None, :])
            )
            candidate = transition[..., None] * current_state + increment
            next_state = jnp.where(valid[:, None, None], candidate, current_state)
            response = 2.0 * jnp.real(jnp.einsum("cpi,pio->co", next_state, self.residue))
            held = right * valid[:, None]
            response = response + jnp.einsum(
                "ci,io->co",
                held,
                self.direct_weight,
            )
            return next_state, response + self.bias

        _, scanned = jax.lax.scan(
            step,
            state,
            (
                jnp.moveaxis(delta, 1, 0),
                jnp.moveaxis(values[:, :-1, :], 1, 0),
                jnp.moveaxis(values[:, 1:, :], 1, 0),
                jnp.moveaxis(valid_interval, 1, 0),
            ),
        )
        initial = (
            jnp.einsum(
                "ci,io->co",
                values[:, 0, :] * source_mask[:, :1],
                self.direct_weight,
            )
            + self.bias
        )
        response = jnp.concatenate(
            (initial[:, None, :], jnp.moveaxis(scanned, 0, 1)),
            axis=1,
        )
        response = response * query_mask[..., None]
        output = response.reshape(
            case_shape + batch.require_single_query().sample_shape + (_get_size(self.out_size),)
        )
        if self.out_size == "scalar":
            return output[..., 0]
        return output

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("LaplaceTemporalOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["LaplaceTemporalOperator"]
