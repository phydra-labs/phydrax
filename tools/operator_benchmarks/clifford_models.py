#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from phydrax._fingerprint import canonical_fingerprint
from phydrax._trainable import NonTrainableState
from phydrax.nn.operator.layers import (
    clifford_gated_activation,
    CliffordGeometricProductLayer,
    CliffordGradeLinear,
)
from phydrax.nn.operator.representations import CliffordGradeRepresentation


class PeriodicCliffordLaplacian(eqx.Module, NonTrainableState):
    """Exact Fourier replay of the periodic componentwise grid Laplacian."""

    __hash__ = object.__hash__

    grid_shape: tuple[int, ...] = eqx.field(static=True)
    periods: tuple[float, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: Sequence[int],
        /,
        *,
        periods: Sequence[float] | None = None,
    ):
        shape = tuple(int(value) for value in grid_shape)
        if not shape or any(value <= 1 for value in shape):
            raise ValueError(
                "Periodic Clifford Laplacian requires grid axes larger than one."
            )
        period_values = (
            (2.0 * math.pi,) * len(shape)
            if periods is None
            else tuple(float(value) for value in periods)
        )
        if len(period_values) != len(shape) or any(
            not math.isfinite(value) or value <= 0.0 for value in period_values
        ):
            raise ValueError("Periodic Clifford periods must be finite and positive.")
        self.grid_shape = shape
        self.periods = period_values
        self.operator_id = canonical_fingerprint(
            {
                "kind": "periodic-clifford-fourier-laplacian-v1",
                "grid_shape": list(shape),
                "periods": list(period_values),
            }
        )

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        spatial_rank = len(self.grid_shape)
        if (
            array.ndim < spatial_rank + 1
            or array.shape[-spatial_rank - 1 : -1] != self.grid_shape
        ):
            raise ValueError(
                "Clifford Laplacian input does not match its declared periodic grid."
            )
        axes = tuple(range(array.ndim - spatial_rank - 1, array.ndim - 1))
        multiplier = None
        for local_axis, (count, period) in enumerate(zip(self.grid_shape, self.periods)):
            spacing = period / count
            frequency = 2.0 * jnp.pi * jnp.fft.fftfreq(count, d=spacing)
            shape = [1] * array.ndim
            shape[axes[local_axis]] = count
            square = frequency.reshape(shape) ** 2
            multiplier = square if multiplier is None else multiplier + square
        transformed = jnp.fft.fftn(array, axes=axes)
        return jnp.fft.ifftn(-multiplier * transformed, axes=axes).real.astype(
            array.dtype
        )


class DifferentialCliffordOperatorBlock(eqx.Module):
    """Benchmark candidate coupling a declared Laplacian to Clifford products."""

    __hash__ = object.__hash__

    representation: CliffordGradeRepresentation
    latent_representation: CliffordGradeRepresentation
    context_operator: PeriodicCliffordLaplacian
    state_lift: CliffordGradeLinear
    context_lift: CliffordGradeLinear
    interaction: CliffordGeometricProductLayer
    projection: CliffordGradeLinear
    residual_scale: float = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: CliffordGradeRepresentation,
        context_operator: PeriodicCliffordLaplacian,
        /,
        *,
        latent_channels: int = 2,
        residual_scale: float = 0.05,
        key: Array = jr.key(0),
    ):
        if not isinstance(representation, CliffordGradeRepresentation):
            raise TypeError("representation must be CliffordGradeRepresentation.")
        if not isinstance(context_operator, PeriodicCliffordLaplacian):
            raise TypeError("context_operator must be PeriodicCliffordLaplacian.")
        channels = int(latent_channels)
        scale = float(residual_scale)
        if channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("residual_scale must be finite and nonnegative.")
        latent = CliffordGradeRepresentation(
            representation.algebra,
            (channels,) * (representation.algebra.dimension + 1),
        )
        state_key, context_key, interaction_key, projection_key = jr.split(key, 4)
        self.representation = representation
        self.latent_representation = latent
        self.context_operator = context_operator
        self.state_lift = CliffordGradeLinear(
            representation,
            latent,
            use_scalar_bias=False,
            key=state_key,
        )
        self.context_lift = CliffordGradeLinear(
            representation,
            latent,
            use_scalar_bias=False,
            key=context_key,
        )
        self.interaction = CliffordGeometricProductLayer(
            latent,
            key=interaction_key,
        )
        self.projection = CliffordGradeLinear(
            latent,
            representation,
            use_scalar_bias=False,
            key=projection_key,
        )
        self.residual_scale = scale
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "differential-clifford-operator-candidate-v1",
                "representation": representation.representation_id,
                "latent": latent.representation_id,
                "context_operator": context_operator.operator_id,
                "residual_scale": scale,
            }
        )

    def __call__(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[-1] != self.representation.packed_size:
            raise ValueError(
                "Differential Clifford candidate received wrong field width."
            )
        state = self.state_lift(array)
        context = self.context_lift(self.context_operator(array))
        activated = clifford_gated_activation(
            self.interaction(state + context),
            self.latent_representation,
        )
        return array + self.residual_scale * self.projection(activated)


__all__ = ["DifferentialCliffordOperatorBlock", "PeriodicCliffordLaplacian"]
