#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._event_layout import ComplexEventLayout
from .._probability import AbstractProbabilityLaw, DiagonalNormalLaw
from .._strict import StrictModule
from ..domain._measure import MeasureKind
from ._gaussian_diffusion import VariancePreservingDiffusion


ComplexScoreConvention: TypeAlias = Literal["real-packed", "wirtinger"]


class ComplexNormalLaw(AbstractProbabilityLaw):
    """Proper circular complex Normal law with real-coordinate Lebesgue density."""

    location: Array
    variance: Array
    layout: ComplexEventLayout
    real_law: DiagonalNormalLaw

    def __init__(self, location: ArrayLike, variance: ArrayLike, /, *, event_shape):
        mean = jnp.asarray(location)
        if not jnp.iscomplexobj(mean):
            raise TypeError("ComplexNormalLaw location must be complex-valued.")
        shape = tuple(int(size) for size in event_shape)
        if mean.shape != shape:
            raise ValueError("Complex Normal location must match event_shape exactly.")
        value = jnp.broadcast_to(jnp.asarray(variance, dtype=mean.real.dtype), shape)
        if bool(jnp.any(~jnp.isfinite(value) | (value <= 0.0))):
            raise ValueError("Complex variance must be finite and positive.")
        layout = ComplexEventLayout(shape)
        packed_mean = layout.to_real_coordinates(mean)
        packed_scale = jnp.concatenate(
            (jnp.sqrt(value.reshape(-1) / 2.0), jnp.sqrt(value.reshape(-1) / 2.0))
        )
        self.location = mean
        self.variance = value
        self.layout = layout
        self.real_law = DiagonalNormalLaw(
            packed_mean,
            packed_scale,
            event_shape=(layout.coordinate_size,),
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.layout.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> MeasureKind:
        return "lebesgue"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.layout.from_real_coordinates(self.real_law.sample(key, sample_shape))

    def contains(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if not jnp.iscomplexobj(array):
            raise TypeError("Complex Normal values must be complex-valued.")
        return self.real_law.contains(self.layout.to_real_coordinates(array))

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.real_law.log_prob(self.layout.to_real_coordinates(value))

    def score(self, value: ArrayLike, /, *, convention: ComplexScoreConvention = "real-packed"):
        packed = self.real_law.score(self.layout.to_real_coordinates(value))
        if convention == "real-packed":
            return packed
        if convention == "wirtinger":
            return self.layout.from_real_coordinates(packed)
        raise ValueError("Unknown complex score convention.")


class ComplexVariancePreservingDiffusion(StrictModule):
    """Proper complex VP diffusion implemented through explicit real coordinates."""

    layout: ComplexEventLayout
    real_process: VariancePreservingDiffusion
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_shape,
        /,
        *,
        beta_minimum: float = 0.1,
        beta_maximum: float = 20.0,
        terminal_time: float = 1.0,
        process_id: str | None = None,
    ):
        layout = ComplexEventLayout(event_shape)
        real = VariancePreservingDiffusion(
            layout.coordinate_size,
            beta_minimum=beta_minimum,
            beta_maximum=beta_maximum,
            terminal_time=terminal_time,
            process_id=process_id,
        )
        self.layout = layout
        self.real_process = real
        self.process_id = real.process_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.layout.event_shape

    @property
    def terminal_time(self) -> float:
        return self.real_process.terminal_time

    def perturb(self, key: Key[Array, ""], clean: ArrayLike, /, *, time: ArrayLike) -> Array:
        packed = self.layout.to_real_coordinates(clean)
        perturbed = self.real_process.perturb(key, packed, t1=time)
        return self.layout.from_real_coordinates(perturbed)

    def conditional_score(
        self,
        perturbed: ArrayLike,
        clean: ArrayLike,
        /,
        *,
        time: ArrayLike,
        convention: ComplexScoreConvention = "real-packed",
    ):
        noisy = self.layout.to_real_coordinates(perturbed)
        source = self.layout.to_real_coordinates(clean)
        score = self.real_process.conditional_score(noisy, source, t1=time)
        if convention == "real-packed":
            return score
        if convention == "wirtinger":
            return self.layout.from_real_coordinates(score)
        raise ValueError("Unknown complex score convention.")

    def real_terminal_reference(self):
        return self.real_process.asymptotic_terminal_reference()


__all__ = [
    "ComplexNormalLaw",
    "ComplexScoreConvention",
    "ComplexVariancePreservingDiffusion",
]
