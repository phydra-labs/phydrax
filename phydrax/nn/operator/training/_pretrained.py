# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ..catalog import operator_pretrained_artifacts, PretrainedOperatorArtifact


class BundledPretrainedOperator(StrictModule):
    """Locally materialized first-party weights with pure JAX prediction."""

    weights: tuple[Array, ...]
    descriptor: PretrainedOperatorArtifact = eqx.field(static=True)
    family: Literal["fno", "deeponet"] = eqx.field(static=True)

    # JAX hashes externally jitted bound methods; loaded model state is immutable.
    __hash__ = object.__hash__

    def predict(
        self, values: ArrayLike, query: ArrayLike | int | None = None, /
    ) -> Array:
        source = jnp.asarray(values, dtype=jnp.float32)
        if self.family == "fno":
            spectral_weight, lift_weight, projection_weight, mean, scale = self.weights
            if source.shape[-1:] != (1,):
                source = source[..., None]
            normalized = (source - mean) / scale
            lifted = contract("...ni,wi->...nw", normalized, lift_weight)
            coefficients = jnp.fft.rfft(lifted, axis=-2)
            modes = min(int(coefficients.shape[-2]), int(spectral_weight.shape[-1]))
            mixed = contract(
                "...mr,orm->...mo",
                coefficients[..., :modes, :],
                spectral_weight[..., :modes],
            )
            source_size = int(source.shape[-2])
            target_size = source_size if query is None else int(query)
            if target_size != source_size:
                mixed = mixed * (target_size / source_size)
            physical = jnp.fft.irfft(mixed, n=target_size, axis=-2)
            return contract("...nw,ow->...no", physical, projection_weight) * scale + mean
        branch_weight, trunk_weight, output_scale, mean, scale = self.weights
        flattened = (
            source.reshape(source.shape[:-2] + (-1,)) if source.ndim >= 2 else source
        )
        if flattened.shape[-1] != branch_weight.shape[-1]:
            raise ValueError("DeepONet pretrained source requires 32 sensor values.")
        if query is None:
            raise ValueError("DeepONet pretrained prediction requires query coordinates.")
        coordinates = jnp.asarray(query, dtype=source.dtype)
        if coordinates.shape[-1:] != (1,):
            coordinates = coordinates[..., None]
        branch = contract(
            "...s,ls->...l", (flattened - mean[0]) / scale[0], branch_weight
        )
        trunk = coordinates * trunk_weight[:, 0]
        return output_scale * contract("...l,...ql->...q", branch, trunk)

    def __call__(
        self, values: ArrayLike, query: ArrayLike | int | None = None, /
    ) -> Array:
        return self.predict(values, query)


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_operator(
    name: str, *, cache_dir: str | Path | None = None
) -> BundledPretrainedOperator:
    """Verify and host-load one catalog-derived packaged pretrained operator."""
    del cache_dir
    matches = tuple(item for item in operator_pretrained_artifacts() if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"Unknown pretrained operator artifact {name!r}.")
    descriptor = matches[0]
    source = Path(__file__).resolve().parents[1] / descriptor.resource
    if not source.is_file() or _sha256(source) != descriptor.sha256:
        raise ValueError("Packaged pretrained operator is missing or checksum-invalid.")
    with np.load(source, allow_pickle=False) as archive:
        if name == "fno-diffusion-1d":
            keys = (
                "spectral_weight",
                "lift_weight",
                "projection_weight",
                "normalization_mean",
                "normalization_scale",
            )
            family: Literal["fno", "deeponet"] = "fno"
        else:
            keys = (
                "branch_weight",
                "trunk_weight",
                "output_scale",
                "normalization_mean",
                "normalization_scale",
            )
            family = "deeponet"
        weights = tuple(jnp.asarray(archive[key]) for key in keys)
    return BundledPretrainedOperator(weights, descriptor, family)


__all__ = ["BundledPretrainedOperator", "load_pretrained_operator"]
