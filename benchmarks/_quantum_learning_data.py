#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint


@dataclass(frozen=True, slots=True)
class QuantumLearningDataset:
    features: Array
    targets: Array
    dataset_id: str
    parameters: dict[str, int | float | str]


def _dataset(
    kind: str,
    features: Array,
    targets: Array,
    parameters: dict[str, int | float | str],
    /,
) -> QuantumLearningDataset:
    fingerprint = array_tree_fingerprint((features, targets))["sha256"]
    dataset_id = canonical_fingerprint(
        {
            "kind": kind,
            "parameters": parameters,
            "content": fingerprint,
        }
    )
    return QuantumLearningDataset(features, targets, dataset_id, parameters)


def linearly_separable(
    key: PRNGKeyArray,
    sample_count: int,
    feature_count: int,
    /,
) -> QuantumLearningDataset:
    count = int(sample_count)
    dimension = int(feature_count)
    if count < 4 or dimension <= 0:
        raise ValueError(
            "Linear benchmark data require at least four samples and one feature."
        )
    point_key, direction_key = jax.random.split(key)
    features = jax.random.normal(
        point_key,
        (count, dimension),
        dtype=jnp.float64,
    )
    direction = jax.random.normal(
        direction_key,
        (dimension,),
        dtype=jnp.float64,
    )
    targets = (features @ direction >= 0.0).astype(jnp.float64)
    return _dataset(
        "linearly-separable",
        features,
        targets,
        {"sample_count": count, "feature_count": dimension},
    )


def two_curves(
    key: PRNGKeyArray,
    sample_count: int,
    /,
    *,
    noise: float = 0.05,
) -> QuantumLearningDataset:
    count = int(sample_count)
    scale = float(noise)
    if count < 4 or count % 2 != 0:
        raise ValueError("Two-curves data require an even sample count of at least four.")
    if not 0.0 <= scale < 1.0:
        raise ValueError("Two-curves noise must lie in [0, 1).")
    parameter_key, noise_key, permutation_key = jax.random.split(key, 3)
    half = count // 2
    parameter = jax.random.uniform(
        parameter_key,
        (half,),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float64,
    )
    perturbation = scale * jax.random.normal(
        noise_key,
        (count, 2),
        dtype=jnp.float64,
    )
    lower = jnp.stack(
        (parameter, 0.35 * jnp.sin(jnp.pi * parameter) - 0.45),
        axis=-1,
    )
    upper = jnp.stack(
        (parameter, 0.35 * jnp.sin(jnp.pi * parameter) + 0.45),
        axis=-1,
    )
    features = jnp.concatenate((lower, upper), axis=0) + perturbation
    targets = jnp.concatenate((jnp.zeros((half,)), jnp.ones((half,))))
    permutation = jax.random.permutation(permutation_key, count)
    features = features[permutation]
    targets = targets[permutation]
    return _dataset(
        "two-curves",
        features,
        targets,
        {"sample_count": count, "feature_count": 2, "noise": scale},
    )


def hyperplane_parity(
    key: PRNGKeyArray,
    sample_count: int,
    feature_count: int,
    /,
) -> QuantumLearningDataset:
    count = int(sample_count)
    dimension = int(feature_count)
    if count < 4 or dimension < 2:
        raise ValueError("Parity data require at least four samples and two features.")
    features = jax.random.normal(
        key,
        (count, dimension),
        dtype=jnp.float64,
    )
    targets = (jnp.prod(jnp.where(features >= 0.0, 1.0, -1.0), axis=-1) > 0.0).astype(
        jnp.float64
    )
    return _dataset(
        "hyperplane-parity",
        features,
        targets,
        {"sample_count": count, "feature_count": dimension},
    )


def hidden_manifold(
    key: PRNGKeyArray,
    sample_count: int,
    ambient_dimension: int,
    /,
) -> QuantumLearningDataset:
    count = int(sample_count)
    ambient = int(ambient_dimension)
    if count < 4 or ambient < 2:
        raise ValueError(
            "Hidden-manifold data require at least four samples and two dimensions."
        )
    latent_key, projection_key = jax.random.split(key)
    latent = jax.random.normal(latent_key, (count, 2), dtype=jnp.float64)
    lifted = jnp.stack(
        (latent[:, 0], latent[:, 1], latent[:, 0] * latent[:, 1]),
        axis=-1,
    )
    projection = jax.random.normal(
        projection_key,
        (3, ambient),
        dtype=jnp.float64,
    )
    features = lifted @ projection / jnp.sqrt(3.0)
    targets = (latent[:, 0] * latent[:, 1] >= 0.0).astype(jnp.float64)
    return _dataset(
        "hidden-manifold",
        features,
        targets,
        {
            "sample_count": count,
            "feature_count": ambient,
            "intrinsic_dimension": 2,
        },
    )


__all__ = [
    "QuantumLearningDataset",
    "hidden_manifold",
    "hyperplane_parity",
    "linearly_separable",
    "two_curves",
]
