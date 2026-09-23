#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Non-Abelian twist-manifold topology for finite many-body eigenspaces."""

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ManyBodyTwistChernResult(StrictModule, NonTrainableState):
    link_determinants: Array
    plaquette_phases: Array
    raw_chern: Array
    nearest_integer: Array
    quantization_residual: Array
    minimum_link_singular_value: Array
    minimum_direct_gap: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def many_body_twist_chern(
    states: ArrayLike,
    direct_gaps: ArrayLike,
    /,
    *,
    gap_tolerance: float = 1.0e-8,
    link_tolerance: float = 1.0e-10,
    quantization_tolerance: float = 1.0e-6,
) -> ManyBodyTwistChernResult:
    """Evaluate one periodic two-dimensional twist grid of degenerate manifolds."""

    values = np.asarray(states)
    gaps = np.asarray(direct_gaps, dtype=np.float64)
    gap_tol = float(gap_tolerance)
    link_tol = float(link_tolerance)
    quant_tol = float(quantization_tolerance)
    if (
        values.ndim != 4
        or values.shape[0] < 2
        or values.shape[1] < 2
        or values.shape[2] < 1
        or values.shape[3] < 1
        or gaps.shape != values.shape[:2]
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(gaps))
        or any(
            not isfinite(value) or value <= 0.0
            for value in (gap_tol, link_tol, quant_tol)
        )
    ):
        raise ValueError("Many-body twist states, gaps, or tolerances are invalid.")
    gram = np.einsum("xyda,xydb->xyab", np.conj(values), values)
    normalization = float(np.max(np.abs(gram - np.eye(values.shape[-1]))))
    links = np.empty(values.shape[:2] + (2,), dtype=np.complex128)
    minimum = np.inf
    for first in range(values.shape[0]):
        for second in range(values.shape[1]):
            for axis, target in enumerate(
                (
                    ((first + 1) % values.shape[0], second),
                    (first, (second + 1) % values.shape[1]),
                )
            ):
                overlap = np.conj(values[first, second]).T @ values[target]
                left, singular, right_h = np.linalg.svd(overlap)
                minimum = min(minimum, float(np.min(singular)))
                unitary = left @ right_h
                determinant = np.linalg.det(unitary)
                links[first, second, axis] = determinant / abs(determinant)
    phases = np.empty(values.shape[:2], dtype=np.float64)
    for first in range(values.shape[0]):
        for second in range(values.shape[1]):
            phases[first, second] = np.angle(
                links[first, second, 0]
                * links[(first + 1) % values.shape[0], second, 1]
                * np.conj(links[first, (second + 1) % values.shape[1], 0])
                * np.conj(links[first, second, 1])
            )
    raw = float(np.sum(phases) / (2.0 * pi))
    nearest = int(np.rint(raw))
    quantization = abs(raw - nearest)
    successful = bool(
        normalization <= link_tol
        and minimum > link_tol
        and float(np.min(gaps)) > gap_tol
        and quantization <= quant_tol
    )
    return ManyBodyTwistChernResult(
        jnp.asarray(links),
        jnp.asarray(phases),
        jnp.asarray(raw),
        jnp.asarray(nearest, dtype=jnp.int32),
        jnp.asarray(quantization),
        jnp.asarray(minimum),
        jnp.asarray(np.min(gaps)),
        jnp.asarray(successful),
        canonical_fingerprint(
            {
                "kind": "many-body-twist-chern-result",
                "arrays": array_tree_fingerprint(
                    {"links": links, "phases": phases, "gaps": gaps}
                ),
                "raw_chern": raw,
                "nearest_integer": nearest,
            }
        ),
    )


__all__ = ["ManyBodyTwistChernResult", "many_body_twist_chern"]
