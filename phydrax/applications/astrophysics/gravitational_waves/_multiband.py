#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from phydrax.ein import contract

from ...._fingerprint import canonical_fingerprint
from ._approximation import (
    LikelihoodApproximationPolicy,
    LinearQuadraticCompressedLikelihood,
    QualifiedGravitationalWaveLikelihood,
    qualify_likelihood,
)
from ._likelihood import GravitationalWaveLikelihoodPlan


def _multiband_nodes(
    frequency: np.ndarray,
    active: np.ndarray,
    band_edges: tuple[float, ...],
    strides: tuple[int, ...],
) -> np.ndarray:
    selected: set[int] = set()
    for band_index, stride in enumerate(strides):
        lower, upper = band_edges[band_index : band_index + 2]
        indices = np.flatnonzero(active & (frequency >= lower) & (frequency <= upper))
        if indices.size == 0:
            raise ValueError("Every multiband interval must contain active frequencies.")
        selected.update(int(value) for value in indices[::stride])
        selected.add(int(indices[0]))
        selected.add(int(indices[-1]))
    result = np.asarray(sorted(selected), dtype=np.int32)
    if result.size < 2:
        raise ValueError("Multiband preparation requires at least two frequency nodes.")
    return result


def _linear_reconstruction(
    frequency: np.ndarray,
    active: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    node_frequency = frequency[nodes]
    matrix = np.zeros((frequency.size, nodes.size), dtype=float)
    for index in np.flatnonzero(active):
        location = int(np.searchsorted(node_frequency, frequency[index], side="right"))
        if location == 0:
            matrix[index, 0] = 1.0
        elif location == nodes.size:
            matrix[index, -1] = 1.0
        else:
            left, right = location - 1, location
            width = node_frequency[right] - node_frequency[left]
            fraction = (frequency[index] - node_frequency[left]) / width
            matrix[index, left] = 1.0 - fraction
            matrix[index, right] = fraction
    return matrix


def prepare_multiband_likelihood(
    exact: GravitationalWaveLikelihoodPlan,
    band_edges: Sequence[float],
    strides: Sequence[int],
    validation_parameters: Sequence[PyTree[Any]],
    validation_ids: Sequence[str],
    /,
    *,
    policy: LikelihoodApproximationPolicy | None = None,
) -> QualifiedGravitationalWaveLikelihood:
    if not isinstance(exact, GravitationalWaveLikelihoodPlan):
        raise TypeError("exact must be GravitationalWaveLikelihoodPlan.")
    edges = tuple(float(value) for value in band_edges)
    stride_values = tuple(int(value) for value in strides)
    if (
        len(edges) < 2
        or len(stride_values) != len(edges) - 1
        or any(not np.isfinite(value) for value in edges)
        or any(right <= left for left, right in itertools.pairwise(edges))
        or any(value <= 0 for value in stride_values)
    ):
        raise ValueError("Multiband edges and strides are invalid.")
    frequency = np.asarray(exact.network.frequency)
    active = np.any(np.asarray(exact.network.active), axis=0)
    active_frequency = frequency[active]
    if edges[0] > active_frequency[0] or edges[-1] < active_frequency[-1]:
        raise ValueError(
            "Multiband edges must cover the complete active frequency support."
        )
    nodes = _multiband_nodes(frequency, active, edges, stride_values)
    reconstruction = jnp.asarray(_linear_reconstruction(frequency, active, nodes))
    inverse_variance = exact.network.inverse_variance
    linear_functional = 2.0 * jnp.conj(exact.network.strain) * inverse_variance
    quadratic_functional = 2.0 * inverse_variance
    linear_weights = contract("df,fn->dn", linear_functional, reconstruction)
    quadratic_weights = contract("df,fn->dn", quadratic_functional, reconstruction)
    approximation_id = canonical_fingerprint(
        {
            "kind": "gravitational-wave-multiband-policy",
            "edges": list(edges),
            "strides": list(stride_values),
            "nodes": nodes.tolist(),
        }
    )
    candidate = LinearQuadraticCompressedLikelihood(
        exact,
        exact.network.frequency[jnp.asarray(nodes)],
        exact.network.frequency[jnp.asarray(nodes)],
        linear_weights,
        quadratic_weights,
        approximation_id=f"multiband:{approximation_id}",
    )
    return qualify_likelihood(
        exact,
        candidate,
        validation_parameters,
        validation_ids,
        policy=policy,
    )


__all__ = ["prepare_multiband_likelihood"]
