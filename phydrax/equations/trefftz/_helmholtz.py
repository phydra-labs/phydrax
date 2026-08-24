#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._core import (
    AbstractTrefftzBasis,
    SimilarityNormalization,
    TrefftzResourceBudget,
    TrefftzResourceEvidence,
    TrialSpaceCertificate,
)


def sample_unit_directions(
    count: int,
    dimension: int,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    """Sample fixed unit directions from normalized Gaussian vectors."""

    count_ = int(count)
    dimension_ = int(dimension)
    if count_ <= 0 or dimension_ < 2:
        raise ValueError("Direction count must be positive and dimension at least two.")
    values = jr.normal(key, (count_, dimension_), dtype=float)
    norms = jnp.linalg.norm(values, axis=-1, keepdims=True)
    return values / norms


def _canonical_directions(directions: ArrayLike, dimension: int) -> tuple[np.ndarray, float]:
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != dimension or values.shape[0] == 0:
        raise ValueError(
            f"directions must have shape (num_directions, {dimension}) with positive count."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Helmholtz directions must be finite.")
    norms = np.linalg.norm(values, axis=-1)
    if np.any(norms == 0.0):
        raise ValueError("Helmholtz directions must be nonzero.")
    residual = float(np.max(np.abs(norms - 1.0)))
    tolerance = 256.0 * np.finfo(np.float64).eps * max(dimension, 1)
    if residual > tolerance:
        raise ValueError(
            "Helmholtz directions must already be unit length; "
            f"maximum norm residual is {residual:.3e}."
        )
    values = values / norms[:, None]

    sign_tolerance = 32.0 * np.finfo(np.float64).eps
    canonical = []
    for row in values:
        significant = np.flatnonzero(np.abs(row) > sign_tolerance)
        if significant.size == 0:
            raise ValueError("Helmholtz direction is numerically zero.")
        oriented = -row if row[int(significant[0])] < 0.0 else row
        canonical.append(oriented)
    canonical_array = np.asarray(canonical)
    order = sorted(range(canonical_array.shape[0]), key=lambda index: tuple(canonical_array[index]))
    canonical_array = canonical_array[np.asarray(order, dtype=np.int32)]
    for index in range(1, canonical_array.shape[0]):
        if np.allclose(
            canonical_array[index - 1],
            canonical_array[index],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Helmholtz directions contain duplicate or antipodal modes.")
    return canonical_array, residual


class HelmholtzPlaneWaveBasis(AbstractTrefftzBasis):
    """Real sine/cosine plane-wave basis for the homogeneous Helmholtz equation."""

    normalization: SimilarityNormalization
    physical_wavenumber: Array
    normalized_wavenumber: Array
    directions: Array
    direction_norm_residual: Array
    _basis_id: str = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _resource_evidence: TrefftzResourceEvidence

    def __init__(
        self,
        dimension: int,
        wavenumber: float,
        directions: ArrayLike,
        /,
        *,
        normalization: SimilarityNormalization | None = None,
        resources: TrefftzResourceBudget | None = None,
    ):
        dimension_ = int(dimension)
        wavenumber_ = float(wavenumber)
        if dimension_ < 2:
            raise ValueError("HelmholtzPlaneWaveBasis requires dimension >= 2.")
        if not math.isfinite(wavenumber_) or wavenumber_ <= 0.0:
            raise ValueError("Helmholtz wavenumber must be finite and positive.")
        normalization_ = (
            SimilarityNormalization(np.zeros((dimension_,), dtype=float), 1.0)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, SimilarityNormalization):
            raise TypeError("normalization must be a SimilarityNormalization or None.")
        if normalization_.dimension != dimension_:
            raise ValueError("Normalization dimension must match the plane-wave basis.")
        directions_host, direction_residual = _canonical_directions(directions, dimension_)
        budget = TrefftzResourceBudget() if resources is None else resources
        if not isinstance(budget, TrefftzResourceBudget):
            raise TypeError("resources must be a TrefftzResourceBudget or None.")
        direction_count = int(directions_host.shape[0])
        rank = 2 * direction_count
        evidence = budget.check(
            rank=rank,
            monomials=direction_count,
            basis_entries=int(directions_host.size),
            basis_bytes=int(directions_host.nbytes),
        )
        directions_ = jnp.asarray(directions_host, dtype=float)
        basis_id = canonical_fingerprint(
            {
                "kind": "helmholtz-real-plane-wave-basis-v1",
                "dimension": dimension_,
                "physical_wavenumber": wavenumber_,
                "normalization_id": normalization_.normalization_id,
                "directions": array_tree_fingerprint(directions_),
            }
        )
        certificate = TrialSpaceCertificate(
            equation_family="helmholtz",
            ambient_dimension=dimension_,
            construction="real-sine-cosine-plane-wave-basis",
            equation_parameters={"wavenumber": wavenumber_},
            normalization_id=normalization_.normalization_id,
            basis_id=basis_id,
            rank=rank,
            assumptions=(
                "Euclidean homogeneous Helmholtz operator",
                "real positive constant wavenumber",
                "fixed unit propagation directions",
                "finite plane-wave subspace",
            ),
            construction_residual=direction_residual,
            construction_tolerance=(256.0 * np.finfo(np.float64).eps * dimension_),
        )
        self.normalization = normalization_
        self.physical_wavenumber = jnp.asarray(wavenumber_, dtype=float).reshape(())
        self.normalized_wavenumber = (
            self.physical_wavenumber * normalization_.scale
        )
        self.directions = directions_
        self.direction_norm_residual = jnp.asarray(direction_residual, dtype=float).reshape(())
        self._basis_id = basis_id
        self._certificate = certificate
        self._resource_evidence = evidence

    @property
    def dimension(self) -> int:
        return self.normalization.dimension

    @property
    def rank(self) -> int:
        return self._resource_evidence.rank

    @property
    def dtype(self) -> jnp.dtype:
        return self.directions.dtype

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def certificate(self) -> TrialSpaceCertificate:
        return self._certificate

    @property
    def resource_evidence(self) -> TrefftzResourceEvidence:
        return self._resource_evidence

    def evaluate(self, point: ArrayLike, /) -> Array:
        normalized = self.normalization(point)
        phases = self.normalized_wavenumber.astype(normalized.dtype) * (
            self.directions.astype(normalized.dtype) @ normalized
        )
        return jnp.concatenate((jnp.cos(phases), jnp.sin(phases)), axis=0)


__all__ = ["HelmholtzPlaneWaveBasis", "sample_unit_directions"]
