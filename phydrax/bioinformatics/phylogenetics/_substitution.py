#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.solver._jump import FiniteStateGenerator


class SubstitutionModelEvidence(StrictModule):
    """Generator and equilibrium diagnostics for a substitution model."""

    finite: Array
    off_diagonal_nonnegative: Array
    rows_conservative: Array
    root_distribution_nonnegative: Array
    root_distribution_normalized: Array
    root_stationary: Array
    expected_rate: Array


class FiniteStateSubstitutionModel(StrictModule):
    """General finite-state CTMC substitution model.

    Rows of ``generator.matrix`` index ancestral states and columns index
    descendant states. Transition probabilities are delegated to Phydrax's
    native exact finite-state generator matrix-exponential machinery.
    """

    generator: FiniteStateGenerator
    root_distribution: Array
    valid: Array
    evidence: SubstitutionModelEvidence
    state_count: int = eqx.field(static=True)
    model_name: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)

    @property
    def rate_matrix(self) -> Array:
        return self.generator.matrix

    def transition_matrix(self, duration: ArrayLike, /) -> Array:
        return self.generator.transition_matrix(duration)

    def transition_matrices(self, durations: ArrayLike, /) -> Array:
        values = jnp.asarray(durations, dtype=self.generator.matrix.dtype)
        flat = values.reshape((-1,))
        matrices = jax.vmap(self.generator.transition_matrix)(flat)
        return matrices.reshape(values.shape + (self.state_count, self.state_count))


def _stationary_distribution(matrix: Array, /) -> Array:
    values, vectors = jnp.linalg.eig(matrix.T)
    index = jnp.argmin(jnp.abs(values))
    vector = jnp.real(vectors[:, index])
    vector = jnp.where(jnp.sum(vector) < 0.0, -vector, vector)
    vector = jnp.maximum(vector, 0.0)
    total = jnp.sum(vector)
    return vector / jnp.where(total > 0.0, total, 1.0)


def general_substitution_model(
    rate_matrix: ArrayLike,
    /,
    *,
    root_distribution: ArrayLike | None = None,
    normalize: bool = False,
    model_name: str = "general-finite-state",
) -> FiniteStateSubstitutionModel:
    """Construct a closed finite-state generator with explicit diagnostics.

    ``normalize=True`` rescales the generator to one expected substitution per
    unit branch length under ``root_distribution``. If no root distribution is
    supplied, a stationary distribution is obtained from the generator.
    """

    matrix = jnp.asarray(rate_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("rate_matrix must be a non-empty square matrix.")
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        matrix = matrix.astype(jnp.float32)
    state_count = int(matrix.shape[0])
    root = (
        _stationary_distribution(matrix)
        if root_distribution is None
        else jnp.asarray(root_distribution, dtype=matrix.dtype)
    )
    if root.shape != (state_count,):
        raise ValueError("root_distribution must have one probability per state.")

    expected_rate = -jnp.sum(root * jnp.diag(matrix))
    safe_expected_rate = jnp.where(
        jnp.isfinite(expected_rate) & (expected_rate > 0.0), expected_rate, 1.0
    )
    normalized_matrix = matrix / safe_expected_rate if normalize else matrix
    off_diagonal = normalized_matrix - jnp.diag(jnp.diag(normalized_matrix))
    tolerance = jnp.asarray(64.0 * jnp.finfo(normalized_matrix.dtype).eps)
    finite = jnp.all(jnp.isfinite(normalized_matrix)) & jnp.all(jnp.isfinite(root))
    off_diagonal_nonnegative = jnp.all(off_diagonal >= -tolerance)
    rows_conservative = jnp.all(jnp.abs(jnp.sum(normalized_matrix, axis=-1)) <= tolerance)
    root_nonnegative = jnp.all(root >= 0.0)
    root_normalized = jnp.abs(jnp.sum(root) - 1.0) <= tolerance
    stationary_residual = root @ normalized_matrix
    root_stationary = jnp.all(jnp.abs(stationary_residual) <= 128.0 * tolerance)
    positive_rate = expected_rate > 0.0
    valid = (
        finite
        & off_diagonal_nonnegative
        & rows_conservative
        & root_nonnegative
        & root_normalized
        & root_stationary
        & (~jnp.asarray(normalize) | positive_rate)
    )
    native_generator = FiniteStateGenerator(
        states=jnp.arange(state_count, dtype=jnp.int32),
        matrix=normalized_matrix,
        escaped_rates=jnp.zeros((state_count,), dtype=normalized_matrix.dtype),
        process_id=f"phylogenetic-substitution:{model_name}",
        boundary_policy="error",
    )
    evidence = SubstitutionModelEvidence(
        finite=finite,
        off_diagonal_nonnegative=off_diagonal_nonnegative,
        rows_conservative=rows_conservative,
        root_distribution_nonnegative=root_nonnegative,
        root_distribution_normalized=root_normalized,
        root_stationary=root_stationary,
        expected_rate=expected_rate,
    )
    return FiniteStateSubstitutionModel(
        generator=native_generator,
        root_distribution=root,
        valid=valid,
        evidence=evidence,
        state_count=state_count,
        model_name=str(model_name),
        normalized=bool(normalize),
    )


def _dna_reversible_matrix(frequencies: Array, exchangeabilities: Array) -> Array:
    matrix = exchangeabilities * frequencies[None, :]
    matrix = matrix.at[jnp.diag_indices(4)].set(0.0)
    return matrix.at[jnp.diag_indices(4)].set(-jnp.sum(matrix, axis=-1))


def jc69(*, dtype: jnp.dtype = jnp.float32) -> FiniteStateSubstitutionModel:
    """Jukes--Cantor 1969 model normalized to unit expected rate."""

    matrix = jnp.full((4, 4), jnp.asarray(1.0 / 3.0, dtype=dtype))
    matrix = matrix.at[jnp.diag_indices(4)].set(jnp.asarray(-1.0, dtype=dtype))
    return general_substitution_model(
        matrix,
        root_distribution=jnp.full((4,), jnp.asarray(0.25, dtype=dtype)),
        model_name="JC69",
    )


def k80(
    transition_transversion_ratio: ArrayLike,
    /,
    *,
    dtype: jnp.dtype | None = None,
) -> FiniteStateSubstitutionModel:
    """Kimura 1980 model with A,G and C,T transitions."""

    kappa = jnp.asarray(transition_transversion_ratio, dtype=dtype)
    if kappa.shape != ():
        raise ValueError("transition_transversion_ratio must be scalar.")
    resolved_dtype = jnp.result_type(kappa, jnp.float32)
    exchangeabilities = jnp.ones((4, 4), dtype=resolved_dtype)
    exchangeabilities = exchangeabilities.at[0, 2].set(kappa)
    exchangeabilities = exchangeabilities.at[2, 0].set(kappa)
    exchangeabilities = exchangeabilities.at[1, 3].set(kappa)
    exchangeabilities = exchangeabilities.at[3, 1].set(kappa)
    matrix = _dna_reversible_matrix(
        jnp.full((4,), jnp.asarray(0.25, dtype=resolved_dtype)), exchangeabilities
    )
    return general_substitution_model(
        matrix,
        root_distribution=jnp.full((4,), jnp.asarray(0.25, dtype=resolved_dtype)),
        normalize=True,
        model_name="K80",
    )


def hky85(
    frequencies: ArrayLike,
    transition_transversion_ratio: ArrayLike,
    /,
) -> FiniteStateSubstitutionModel:
    """HKY85 model in canonical state order A,C,G,T."""

    root = jnp.asarray(frequencies)
    kappa = jnp.asarray(transition_transversion_ratio, dtype=root.dtype)
    if root.shape != (4,) or kappa.shape != ():
        raise ValueError("HKY85 requires four frequencies and one scalar ratio.")
    exchangeabilities = jnp.ones((4, 4), dtype=root.dtype)
    exchangeabilities = exchangeabilities.at[0, 2].set(kappa)
    exchangeabilities = exchangeabilities.at[2, 0].set(kappa)
    exchangeabilities = exchangeabilities.at[1, 3].set(kappa)
    exchangeabilities = exchangeabilities.at[3, 1].set(kappa)
    return general_substitution_model(
        _dna_reversible_matrix(root, exchangeabilities),
        root_distribution=root,
        normalize=True,
        model_name="HKY85",
    )


def gtr(
    frequencies: ArrayLike,
    exchangeabilities: ArrayLike,
    /,
) -> FiniteStateSubstitutionModel:
    """General time-reversible DNA model.

    Exchangeabilities use the conventional order AC, AG, AT, CG, CT, GT.
    """

    root = jnp.asarray(frequencies)
    rates = jnp.asarray(exchangeabilities, dtype=root.dtype)
    if root.shape != (4,) or rates.shape != (6,):
        raise ValueError("GTR requires four frequencies and six exchangeabilities.")
    symmetric = jnp.zeros((4, 4), dtype=root.dtype)
    row = jnp.asarray([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
    column = jnp.asarray([1, 2, 3, 2, 3, 3], dtype=jnp.int32)
    symmetric = symmetric.at[row, column].set(rates)
    symmetric = symmetric.at[column, row].set(rates)
    return general_substitution_model(
        _dna_reversible_matrix(root, symmetric),
        root_distribution=root,
        normalize=True,
        model_name="GTR",
    )


JC69 = jc69
K80 = k80
HKY85 = hky85
GTR = gtr


__all__ = [
    "FiniteStateSubstitutionModel",
    "GTR",
    "HKY85",
    "JC69",
    "K80",
    "SubstitutionModelEvidence",
    "general_substitution_model",
    "gtr",
    "hky85",
    "jc69",
    "k80",
]
