#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule


class InjectiveDensityResult(StrictModule):
    latent_state: Array
    reconstructed_state: Array
    gram_log_determinant: Array
    image_residual: Array
    rank_margin: Array
    log_prob: Array
    valid: Array
    status: Array
    reference_measure: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class InjectiveContinuousFlowLaw(AbstractProbabilityLaw):
    """Full-column-rank map of a latent law with Hausdorff density."""

    latent_law: AbstractProbabilityLaw
    map: Any
    left_inverse: Any
    _event_shape: tuple[int, ...] = eqx.field(static=True)
    image_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        latent_law: AbstractProbabilityLaw,
        map: Any,
        left_inverse: Any,
        /,
        *,
        event_shape: tuple[int, ...],
        image_tolerance: float = 1.0e-7,
        rank_tolerance: float = 1.0e-8,
        maximum_dimension: int = 64,
        law_id: str | None = None,
    ):
        if not isinstance(latent_law, AbstractProbabilityLaw):
            raise TypeError("latent_law must be an AbstractProbabilityLaw.")
        if not callable(map) or not callable(left_inverse):
            raise TypeError("map and left_inverse must be callable.")
        target_shape = tuple(int(size) for size in event_shape)
        if not target_shape or any(size <= 0 for size in target_shape):
            raise ValueError("event_shape must contain positive dimensions.")
        latent_dimension = prod(latent_law.event_shape)
        target_dimension = prod(target_shape)
        cap = int(maximum_dimension)
        if target_dimension < latent_dimension:
            raise ValueError(
                "Injective density requires target dimension >= latent dimension."
            )
        if cap <= 0 or max(latent_dimension, target_dimension) > cap:
            raise ValueError("injective density dimension exceeds maximum_dimension.")
        image_threshold = float(image_tolerance)
        rank_threshold = float(rank_tolerance)
        if image_threshold <= 0.0 or rank_threshold <= 0.0:
            raise ValueError("injective density tolerances must be positive.")
        test = jnp.zeros(latent_law.event_shape)
        mapped = jnp.asarray(map(test))
        if mapped.shape != target_shape:
            raise ValueError("map output shape must equal event_shape.")
        recovered = jnp.asarray(left_inverse(mapped))
        if recovered.shape != latent_law.event_shape:
            raise ValueError("left_inverse output shape must equal latent event_shape.")
        resolved_id = law_id or canonical_fingerprint(
            {
                "kind": "injective-continuous-flow-law-v1",
                "latent_shape": latent_law.event_shape,
                "target_shape": target_shape,
                "image_tolerance": image_threshold,
                "rank_tolerance": rank_threshold,
            }
        )
        self.latent_law = latent_law
        self.map = map
        self.left_inverse = left_inverse
        self._event_shape = target_shape
        self.image_tolerance = image_threshold
        self.rank_tolerance = rank_threshold
        self.maximum_dimension = cap
        self.law_id = resolved_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.latent_law.batch_shape

    @property
    def density_measure_kind(self) -> str:
        return "hausdorff"

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        latent = jnp.asarray(self.latent_law.sample(key, sample_shape))
        leading = latent.shape[: -len(self.latent_law.event_shape)]
        flat = latent.reshape((-1,) + self.latent_law.event_shape)
        mapped = jax.vmap(self.map)(flat)
        return mapped.reshape(leading + self.event_shape)

    def log_prob_with_diagnostics(self, value: ArrayLike, /) -> InjectiveDensityResult:
        values = jnp.asarray(value)
        if values.shape[-len(self.event_shape) :] != self.event_shape:
            raise ValueError("value does not end in the injective law event_shape.")
        leading = values.shape[: -len(self.event_shape)]
        flat = values.reshape((-1,) + self.event_shape)
        latent_size = prod(self.latent_law.event_shape)
        target_size = prod(self.event_shape)

        def one(target):
            latent = jnp.asarray(self.left_inverse(target))
            reconstructed = jnp.asarray(self.map(latent))
            residual = jnp.sqrt(jnp.sum((reconstructed - target) ** 2))

            def flattened_map(flat_latent):
                return jnp.asarray(
                    self.map(flat_latent.reshape(self.latent_law.event_shape))
                ).reshape((target_size,))

            jacobian = jax.jacfwd(flattened_map)(latent.reshape((latent_size,)))
            gram = jacobian.T @ jacobian
            eigenvalues = jnp.linalg.eigvalsh(0.5 * (gram + gram.T))
            margin = jnp.min(eigenvalues)
            sign, log_determinant = jnp.linalg.slogdet(gram)
            base = self.latent_law.log_prob(latent)
            valid = (
                (residual <= self.image_tolerance)
                & (margin > self.rank_tolerance)
                & (sign > 0.0)
                & jnp.isfinite(log_determinant)
                & jnp.isfinite(base)
            )
            density = jnp.where(valid, base - 0.5 * log_determinant, -jnp.inf)
            status = jnp.where(
                valid, 0, jnp.where(residual > self.image_tolerance, 1, 2)
            ).astype(jnp.int32)
            return (
                latent,
                reconstructed,
                log_determinant,
                residual,
                margin,
                density,
                valid,
                status,
            )

        latent, reconstructed, logdet, residual, margin, density, valid, status = (
            jax.vmap(one)(flat)
        )
        return InjectiveDensityResult(
            latent_state=latent.reshape(leading + self.latent_law.event_shape),
            reconstructed_state=reconstructed.reshape(leading + self.event_shape),
            gram_log_determinant=logdet.reshape(leading),
            image_residual=residual.reshape(leading),
            rank_margin=margin.reshape(leading),
            log_prob=density.reshape(leading),
            valid=valid.reshape(leading),
            status=status.reshape(leading),
            reference_measure="hausdorff",
            law_id=self.law_id,
        )

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).log_prob

    def contains(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).valid


__all__ = ["InjectiveContinuousFlowLaw", "InjectiveDensityResult"]
