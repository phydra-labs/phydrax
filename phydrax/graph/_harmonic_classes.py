#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..topology import CellSubcomplex, RationalClassBasis
from ._cochain import CochainComplexIR, HarmonicSubspace
from ._cochain_homology import validate_hodge_homology


class HarmonicClassFrame(StrictModule, NonTrainableState):
    """Metric harmonic cochains labelled by exact rational homology classes."""

    exact_basis: RationalClassBasis
    harmonic_subspace: HarmonicSubspace
    harmonic_basis: Array
    period_matrix: Array
    period_inverse: Array
    degree: int = eqx.field(static=True)
    complex_fingerprint: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        exact_basis: RationalClassBasis,
        harmonic_subspace: HarmonicSubspace,
        harmonic_basis: Array,
        period_matrix: Array,
        period_inverse: Array,
        /,
        *,
        degree: int,
        complex_fingerprint: str,
    ):
        self.exact_basis = exact_basis
        self.harmonic_subspace = harmonic_subspace
        self.harmonic_basis = jnp.asarray(harmonic_basis)
        self.period_matrix = jnp.asarray(period_matrix)
        self.period_inverse = jnp.asarray(period_inverse)
        self.degree = int(degree)
        self.complex_fingerprint = str(complex_fingerprint)
        self.frame_id = canonical_fingerprint(
            {
                "kind": "harmonic-class-frame",
                "exact_basis": exact_basis.basis_id,
                "harmonic": harmonic_subspace.complex_fingerprint,
                "degree": int(degree),
                "complex": self.complex_fingerprint,
                "periods": array_tree_fingerprint(self.period_matrix),
            }
        )

    def periods(self, cochain: Array, /) -> Array:
        cycles = _rational_dense(self.exact_basis, int(cochain.shape[-1]))
        return jnp.asarray(cycles.T) @ jnp.asarray(cochain)

    def with_periods(self, cochain: Array, target: Array, /) -> Array:
        values = jnp.asarray(cochain)
        desired = jnp.asarray(target)
        current = self.periods(values)
        correction = self.period_inverse @ (desired - current)
        return values + self.harmonic_basis @ correction


def _rational_dense(basis: RationalClassBasis, cell_count: int, /) -> np.ndarray:
    matrix = np.zeros((cell_count, basis.generator_count), dtype=float)
    for cell, generator, numerator, denominator in zip(
        np.asarray(basis.cell_indices),
        np.asarray(basis.generator_indices),
        basis.numerators,
        basis.denominators,
        strict=True,
    ):
        matrix[int(cell), int(generator)] = numerator / denominator
    return matrix


def _dense_inverse(matrix: np.ndarray, tolerance: float, /) -> np.ndarray:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError("Harmonic period matrix must be square.")
    augmented = np.concatenate(
        (matrix.astype(complex), np.eye(size, dtype=complex)),
        axis=1,
    )
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) <= float(tolerance):
            raise ValueError("Harmonic period matrix is singular.")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row != column:
                augmented[row] -= augmented[row, column] * augmented[column]
    inverse = augmented[:, size:]
    return np.real_if_close(inverse)


def prepare_harmonic_class_frame(
    complex_ir: CochainComplexIR,
    exact_basis: RationalClassBasis,
    /,
    *,
    harmonic_subspace: HarmonicSubspace | None = None,
    tolerance: float = 1e-9,
) -> HarmonicClassFrame:
    """Prepare period-normalized harmonic coordinates for one exact free-class basis."""
    degree = exact_basis.degree
    full = CellSubcomplex.full(complex_ir.discretization.topology)
    if exact_basis.source_id != full.subcomplex_id:
        raise ValueError("Rational class basis belongs to a different cell complex.")
    resolved, report = validate_hodge_homology(
        complex_ir,
        degree,
        harmonic_subspace=harmonic_subspace,
        tolerance=tolerance,
    )
    if not bool(report.complete):
        raise ValueError(
            "Harmonic class frame requires complete numerical kernel evidence."
        )
    rank = resolved.ranks[degree]
    if rank != exact_basis.generator_count:
        raise ValueError("Exact class count and harmonic rank differ.")
    harmonic = np.asarray(resolved.bases[degree][:, :rank])
    cycles = _rational_dense(exact_basis, harmonic.shape[0])
    periods = cycles.T @ harmonic
    inverse = _dense_inverse(periods, tolerance)
    normalized = harmonic @ inverse
    return HarmonicClassFrame(
        exact_basis,
        resolved,
        jnp.asarray(normalized),
        jnp.eye(rank, dtype=jnp.asarray(normalized).dtype),
        jnp.eye(rank, dtype=jnp.asarray(normalized).dtype),
        degree=degree,
        complex_fingerprint=complex_ir.fingerprint,
    )


class CochainTransferCertificate(StrictModule, NonTrainableState):
    """Period and coboundary evidence for one numeric cochain transfer."""

    commutator_residual: Array
    period_residual: Array
    valid: Array
    source_frame_id: str = eqx.field(static=True)
    target_frame_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        commutator_residual: Array,
        period_residual: Array,
        /,
        *,
        source_frame_id: str,
        target_frame_id: str,
        tolerance: float,
    ):
        commutator = jnp.asarray(commutator_residual)
        period = jnp.asarray(period_residual)
        self.commutator_residual = commutator
        self.period_residual = period
        self.valid = (commutator <= float(tolerance)) & (period <= float(tolerance))
        self.source_frame_id = str(source_frame_id)
        self.target_frame_id = str(target_frame_id)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "cochain-transfer-certificate",
                "source": self.source_frame_id,
                "target": self.target_frame_id,
                "tolerance": float(tolerance),
            }
        )


__all__ = [
    "CochainTransferCertificate",
    "HarmonicClassFrame",
    "prepare_harmonic_class_frame",
]
