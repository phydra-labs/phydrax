#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import AbstractAttribute, StrictModule


class LatentPosterior(StrictModule):
    law: AbstractProbabilityLaw
    valid: Array
    encoder_id: str = eqx.field(static=True)


class DecodedDistribution(StrictModule):
    law: AbstractProbabilityLaw | None
    location: Any
    valid: Array
    decoder_id: str = eqx.field(static=True)
    density_kind: str = eqx.field(static=True)


class AbstractLatentRepresentation(StrictModule):
    data_event_shape: AbstractAttribute[tuple[int, ...]]
    latent_event_shape: AbstractAttribute[tuple[int, ...]]
    representation_id: AbstractAttribute[str]
    density_capability: AbstractAttribute[str]

    @abstractmethod
    def encode(self, value: ArrayLike, /, *, key: Key[Array, ""]) -> LatentPosterior:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: ArrayLike, /, *, key: Key[Array, ""]) -> DecodedDistribution:
        raise NotImplementedError


class CallableLatentRepresentation(AbstractLatentRepresentation):
    """Typed adapter around explicit encoder and decoder callables."""

    encoder: Any
    decoder: Any
    data_event_shape: tuple[int, ...] = eqx.field(static=True)
    latent_event_shape: tuple[int, ...] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    density_capability: str = eqx.field(static=True)

    def __init__(
        self,
        encoder,
        decoder,
        /,
        *,
        data_event_shape,
        latent_event_shape,
        representation_id: str,
        density_capability: str = "decoder-likelihood",
    ):
        if not callable(encoder) or not callable(decoder):
            raise TypeError("encoder and decoder must be callable.")
        data_shape = tuple(int(size) for size in data_event_shape)
        latent_shape = tuple(int(size) for size in latent_event_shape)
        if not data_shape or not latent_shape:
            raise ValueError("Data and latent event shapes must be non-empty.")
        if density_capability not in ("decoder-likelihood", "sample-only"):
            raise ValueError("Unknown latent density capability.")
        if not representation_id:
            raise ValueError("representation_id must be non-empty.")
        self.encoder = encoder
        self.decoder = decoder
        self.data_event_shape = data_shape
        self.latent_event_shape = latent_shape
        self.representation_id = representation_id
        self.density_capability = density_capability

    def encode(self, value, /, *, key):
        result = self.encoder(value, key=key)
        if not isinstance(result, LatentPosterior):
            raise TypeError("encoder must return LatentPosterior.")
        if result.law.event_shape != self.latent_event_shape:
            raise ValueError("Encoder posterior latent event shape is incompatible.")
        return result

    def decode(self, latent, /, *, key):
        value = jnp.asarray(latent)
        if value.shape[-len(self.latent_event_shape) :] != self.latent_event_shape:
            raise ValueError("Latent value has an incompatible event shape.")
        result = self.decoder(value, key=key)
        if not isinstance(result, DecodedDistribution):
            raise TypeError("decoder must return DecodedDistribution.")
        if result.density_kind != self.density_capability:
            raise ValueError("Decoder density kind contradicts the representation capability.")
        if self.density_capability == "decoder-likelihood" and result.law is None:
            raise ValueError("Decoder-likelihood capability requires a probability law.")
        if result.law is not None and result.law.event_shape != self.data_event_shape:
            raise ValueError("Decoder law data event shape is incompatible.")
        if result.location is None:
            if result.law is None:
                raise ValueError("Decoder must provide a law or sample/location.")
        else:
            location = jnp.asarray(result.location)
            data_rank = len(self.data_event_shape)
            if (
                location.ndim < data_rank
                or tuple(location.shape[-data_rank:]) != self.data_event_shape
            ):
                raise ValueError("Decoder location has an incompatible data event shape.")
        return result


class LatentDiffusionSample(StrictModule):
    latent: Array
    decoded: Any
    valid: Array
    representation_id: str = eqx.field(static=True)
    latent_sampler_id: str = eqx.field(static=True)


class LatentDiffusion(StrictModule):
    """Compose a frozen latent sampler with one explicit representation decoder."""

    representation: AbstractLatentRepresentation
    latent_sampler: Any
    latent_sampler_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: AbstractLatentRepresentation,
        latent_sampler,
        /,
        *,
        latent_sampler_id: str,
        model_id: str | None = None,
    ):
        if not isinstance(representation, AbstractLatentRepresentation):
            raise TypeError("representation must implement AbstractLatentRepresentation.")
        if not callable(latent_sampler) or not latent_sampler_id:
            raise TypeError("latent_sampler must be callable with a non-empty ID.")
        identifier = model_id or canonical_fingerprint(
            {
                "kind": "latent-diffusion-composition",
                "representation_id": representation.representation_id,
                "latent_sampler_id": latent_sampler_id,
            }
        )
        self.representation = representation
        self.latent_sampler = latent_sampler
        self.latent_sampler_id = latent_sampler_id
        self.model_id = identifier

    def sample(self, key: Key[Array, ""], sample_shape, /) -> LatentDiffusionSample:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        latent_key, decode_key = jax.random.split(key)
        latent = jnp.asarray(self.latent_sampler(latent_key, samples))
        expected = samples + self.representation.latent_event_shape
        if latent.shape != expected:
            raise ValueError(f"Latent sampler must return shape {expected}; got {latent.shape}.")
        decoded = self.representation.decode(latent, key=decode_key)
        event_axes = tuple(range(len(samples), latent.ndim))
        finite = jnp.all(jnp.isfinite(latent), axis=event_axes)
        decoded_valid = jnp.broadcast_to(jnp.asarray(decoded.valid, dtype=bool), samples)
        valid = decoded_valid & finite
        return LatentDiffusionSample(
            latent,
            decoded,
            valid,
            self.representation.representation_id,
            self.latent_sampler_id,
        )


def latent_reconstruction_loss(
    representation: AbstractLatentRepresentation,
    value: ArrayLike,
    key: Key[Array, ""],
    /,
) -> Array:
    encode_key, latent_key, decode_key = jax.random.split(key, 3)
    posterior = representation.encode(value, key=encode_key)
    latent = posterior.law.sample(latent_key)
    decoded = representation.decode(latent, key=decode_key)
    if decoded.law is None:
        reconstruction = jnp.asarray(decoded.location)
        return jnp.mean((reconstruction - jnp.asarray(value)) ** 2)
    return -jnp.mean(decoded.law.log_prob(value))


__all__ = [
    "AbstractLatentRepresentation",
    "CallableLatentRepresentation",
    "DecodedDistribution",
    "LatentDiffusion",
    "LatentDiffusionSample",
    "LatentPosterior",
    "latent_reconstruction_loss",
]
