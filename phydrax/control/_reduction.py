#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._gramian import (
    continuous_controllability_gramian,
    continuous_observability_gramian,
    discrete_controllability_gramian,
    discrete_observability_gramian,
)


class BalancedTruncationResult(StrictModule, NonTrainableState):
    matrix: Array
    input_matrix: Array
    output_matrix: Array
    feedthrough: Array
    trial_transform: Array
    test_transform: Array
    hankel_singular_values: Array
    hinfinity_error_bound: Array
    stable: Array
    model_id: str = eqx.field(static=True)


def balanced_truncation(
    matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    rank: int,
    /,
    *,
    feedthrough: ArrayLike | None = None,
    discrete: bool = False,
    stability_tolerance: float = 1.0e-7,
) -> BalancedTruncationResult:
    """Square-root balancing for one stable standard LTI realization."""
    a = jnp.asarray(matrix)
    b = jnp.asarray(input_matrix)
    c = jnp.asarray(output_matrix)
    n = a.shape[0]
    retained = int(rank)
    if (
        a.shape != (n, n)
        or b.ndim != 2
        or b.shape[0] != n
        or c.ndim != 2
        or c.shape[1] != n
    ):
        raise ValueError("LTI matrices have incompatible shape.")
    if retained <= 0 or retained >= n:
        raise ValueError(
            "Balanced truncation rank must satisfy 0 < rank < state dimension."
        )
    d = (
        jnp.zeros((c.shape[0], b.shape[1]), dtype=a.dtype)
        if feedthrough is None
        else jnp.asarray(feedthrough)
    )
    if d.shape != (c.shape[0], b.shape[1]):
        raise ValueError("feedthrough has invalid shape.")
    if discrete:
        controllability = discrete_controllability_gramian(
            a, b, stability_tolerance=stability_tolerance, max_dimension=n
        )
        observability = discrete_observability_gramian(
            a, c, stability_tolerance=stability_tolerance, max_dimension=n
        )
    else:
        controllability = continuous_controllability_gramian(
            a, b, stability_tolerance=stability_tolerance, max_dimension=n
        )
        observability = continuous_observability_gramian(
            a, c, stability_tolerance=stability_tolerance, max_dimension=n
        )
    valid = (
        controllability.diagnostics.converged
        & observability.diagnostics.converged
        & controllability.diagnostics.stable
        & observability.diagnostics.stable
        & controllability.diagnostics.positive_semidefinite
        & observability.diagnostics.positive_semidefinite
    )
    if not bool(np.asarray(valid)):
        raise ValueError("Balanced truncation requires stable converged PSD Gramians.")

    def factor(value):
        eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (value + jnp.conj(value.T)))
        order = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = jnp.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        threshold = (
            jnp.finfo(eigenvalues.dtype).eps * n * jnp.maximum(eigenvalues[0], 1.0)
        )
        keep = eigenvalues > threshold
        count = int(np.sum(np.asarray(keep)))
        return eigenvectors[:, :count] * jnp.sqrt(eigenvalues[:count])[None, :]

    controllability_factor = factor(controllability.value)
    observability_factor = factor(observability.value)
    overlap = jnp.conj(observability_factor.T) @ controllability_factor
    left, singular, right_h = jnp.linalg.svd(overlap, full_matrices=False)
    if singular.size < retained or float(np.asarray(singular[retained - 1])) <= 0.0:
        raise ValueError("Requested balanced rank exceeds the minimal realization rank.")
    inverse_root = jnp.diag(1.0 / jnp.sqrt(singular[:retained]))
    trial = controllability_factor @ jnp.conj(right_h[:retained, :].T) @ inverse_root
    test = observability_factor @ left[:, :retained] @ inverse_root
    reduced_a = jnp.conj(test.T) @ a @ trial
    reduced_b = jnp.conj(test.T) @ b
    reduced_c = c @ trial
    bound = 2.0 * jnp.sum(singular[retained:])
    eigenvalues = jnp.linalg.eigvals(reduced_a)
    reduced_stable = (
        jnp.max(jnp.abs(eigenvalues)) < 1.0 - stability_tolerance
        if discrete
        else jnp.max(jnp.real(eigenvalues)) < -stability_tolerance
    )
    model_id = canonical_fingerprint(
        {
            "kind": "balanced-truncation-result",
            "discrete": discrete,
            "rank": retained,
            "content": array_tree_fingerprint({"a": a, "b": b, "c": c, "d": d})["sha256"],
        }
    )
    return BalancedTruncationResult(
        reduced_a,
        reduced_b,
        reduced_c,
        d,
        trial,
        test,
        singular,
        bound,
        reduced_stable,
        model_id,
    )


class RationalKrylovReduction(StrictModule, NonTrainableState):
    matrix: Array
    input_matrix: Array
    output_matrix: Array
    basis: Array
    shifts: Array
    model_id: str = eqx.field(static=True)


def rational_krylov_reduction(
    matrix: ArrayLike,
    input_matrix: ArrayLike,
    output_matrix: ArrayLike,
    shifts: Sequence[complex],
    /,
) -> RationalKrylovReduction:
    a = jnp.asarray(matrix)
    b = jnp.asarray(input_matrix)
    c = jnp.asarray(output_matrix)
    shifts_ = jnp.asarray(tuple(shifts), dtype=jnp.result_type(a.dtype, jnp.complex64))
    n = a.shape[0]
    if (
        a.shape != (n, n)
        or b.ndim != 2
        or b.shape[0] != n
        or c.ndim != 2
        or c.shape[1] != n
        or shifts_.ndim != 1
        or shifts_.size == 0
    ):
        raise ValueError("Rational Krylov inputs have incompatible shape.")
    identity = jnp.eye(n, dtype=shifts_.dtype)
    blocks = tuple(jnp.linalg.solve(shift * identity - a, b) for shift in shifts_)
    raw = jnp.concatenate(blocks, axis=1)
    basis, _ = jnp.linalg.qr(raw, mode="reduced")
    reduced_a = jnp.conj(basis.T) @ a @ basis
    reduced_b = jnp.conj(basis.T) @ b
    reduced_c = c @ basis
    model_id = canonical_fingerprint(
        {
            "kind": "rational-krylov-reduction",
            "shifts": [str(value) for value in tuple(shifts)],
            "content": array_tree_fingerprint({"a": a, "b": b, "c": c})["sha256"],
        }
    )
    return RationalKrylovReduction(
        reduced_a,
        reduced_b,
        reduced_c,
        basis,
        shifts_,
        model_id,
    )


__all__ = [
    "BalancedTruncationResult",
    "RationalKrylovReduction",
    "balanced_truncation",
    "rational_krylov_reduction",
]
