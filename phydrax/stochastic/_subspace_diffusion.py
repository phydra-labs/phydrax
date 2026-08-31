#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._probability import AbstractProbabilityLaw, DiagonalNormalLaw
from .._strict import StrictModule
from ..domain._measure import MeasureKind
from ._gaussian_diffusion import AbstractGaussianDiffusion


class AffineSubspaceLayout(StrictModule):
    """Full-column-rank affine event subspace with an explicit diagonal metric."""

    origin: Array
    basis: Array
    quadrature_weights: Array
    gram: Array
    gram_cholesky: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    event_size: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        origin: ArrayLike,
        basis: ArrayLike,
        /,
        *,
        event_shape,
        quadrature_weights: ArrayLike | None = None,
        layout_id: str | None = None,
    ):
        events = tuple(int(size) for size in event_shape)
        if not events or any(size <= 0 for size in events):
            raise ValueError("event_shape must contain positive dimensions.")
        size = prod(events)
        center = jnp.asarray(origin)
        raw_vectors = jnp.asarray(basis)
        if jnp.iscomplexobj(center) or jnp.iscomplexobj(raw_vectors):
            raise TypeError("Affine subspace layouts require real coordinates.")
        dtype = jnp.result_type(center.dtype, raw_vectors.dtype)
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(float)
        center = center.astype(dtype)
        vectors = raw_vectors.astype(dtype)
        if center.shape != events or vectors.ndim != 2 or vectors.shape[0] != size:
            raise ValueError("origin/basis do not match the declared ambient event.")
        rank = int(vectors.shape[1])
        if rank <= 0 or rank > size:
            raise ValueError("Subspace rank must lie in [1, event_size].")
        weights = (
            jnp.ones((size,), dtype=center.real.dtype)
            if quadrature_weights is None
            else jnp.asarray(quadrature_weights, dtype=center.real.dtype).reshape((size,))
        )
        if bool(jnp.any(~jnp.isfinite(weights) | (weights <= 0.0))):
            raise ValueError("quadrature_weights must be finite and positive.")
        gram = oe.contract("ir,i,is->rs", vectors, weights, vectors)
        eigenvalues = jnp.linalg.eigvalsh(gram)
        if bool(jnp.any(~jnp.isfinite(eigenvalues) | (eigenvalues <= 0.0))):
            raise ValueError("Subspace basis columns must be metric-linearly independent.")
        identifier = layout_id or canonical_fingerprint(
            {
                "kind": "affine-subspace-layout",
                "event_shape": list(events),
                "rank": rank,
            }
        )
        self.origin = center
        self.basis = vectors
        self.quadrature_weights = weights
        self.gram = gram
        self.gram_cholesky = jnp.linalg.cholesky(gram)
        self.event_shape = events
        self.event_size = size
        self.rank = rank
        self.layout_id = identifier

    def solve_gram(self, rhs: ArrayLike, /) -> Array:
        value = jnp.asarray(rhs, dtype=self.origin.dtype)
        if value.shape[-1:] != (self.rank,):
            raise ValueError("Gram right-hand side must end in the subspace rank.")
        leading = value.shape[:-1]
        flat = value.reshape((-1, self.rank))
        solved = jax.vmap(
            lambda row: jsp.linalg.cho_solve((self.gram_cholesky, True), row)
        )(flat)
        return solved.reshape(leading + (self.rank,))

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients, dtype=self.origin.dtype)
        if value.shape[-1:] != (self.rank,):
            raise ValueError("Subspace coefficients must end in the retained rank.")
        flat = self.origin.reshape((self.event_size,)) + oe.contract(
            "ir,...r->...i", self.basis, value
        )
        return flat.reshape(value.shape[:-1] + self.event_shape)

    def project(self, value: ArrayLike, /) -> tuple[Array, Array]:
        array = jnp.asarray(value, dtype=self.origin.dtype)
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError("Ambient value does not match the subspace event shape.")
        leading = array.shape[:-rank]
        residual = array.reshape(leading + (self.event_size,)) - self.origin.reshape(
            (self.event_size,)
        )
        rhs = oe.contract("ir,i,...i->...r", self.basis, self.quadrature_weights, residual)
        coefficients = self.solve_gram(rhs)
        reconstruction = oe.contract("ir,...r->...i", self.basis, coefficients)
        orthogonal = residual - reconstruction
        norm = jnp.sqrt(
            oe.contract("...i,i,...i->...", orthogonal, self.quadrature_weights, orthogonal)
        )
        return coefficients, norm

    @property
    def log_volume(self) -> Array:
        return jnp.sum(jnp.log(jnp.diag(self.gram_cholesky)))


class SubspaceGaussianLaw(AbstractProbabilityLaw):
    """Pushforward of a coefficient law to an affine Hausdorff event measure."""

    layout: AffineSubspaceLayout
    coefficient_law: AbstractProbabilityLaw
    support_tolerance: Array

    def __init__(self, layout, coefficient_law, /, *, support_tolerance=1e-8):
        from ..uq._factor_law import GaussianFactorLaw

        if not isinstance(layout, AffineSubspaceLayout):
            raise TypeError("layout must be an AffineSubspaceLayout.")
        if not isinstance(coefficient_law, (DiagonalNormalLaw, GaussianFactorLaw)):
            raise TypeError("SubspaceGaussianLaw requires a Gaussian coefficient law.")
        if coefficient_law.density_measure_kind != "lebesgue":
            raise ValueError("Coefficient Gaussian must be full rank in subspace coordinates.")
        if coefficient_law.batch_shape or coefficient_law.event_shape != (layout.rank,):
            raise ValueError("Coefficient law must be unbatched with event shape (rank,).")
        tolerance = jnp.asarray(support_tolerance, dtype=float).reshape(())
        if bool(~jnp.isfinite(tolerance)) or float(tolerance) < 0.0:
            raise ValueError("support_tolerance must be finite and nonnegative.")
        self.layout = layout
        self.coefficient_law = coefficient_law
        self.support_tolerance = tolerance

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.layout.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> MeasureKind:
        return "hausdorff"

    def sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.layout.synthesize(self.coefficient_law.sample(key, sample_shape))

    def contains(self, value: ArrayLike, /) -> Array:
        coefficients, residual = self.layout.project(value)
        finite = jnp.all(jnp.isfinite(coefficients), axis=-1)
        return finite & (residual <= self.support_tolerance)

    def log_prob(self, value: ArrayLike, /) -> Array:
        coefficients, residual = self.layout.project(value)
        density = self.coefficient_law.log_prob(coefficients) - self.layout.log_volume
        return jnp.where(
            (residual <= self.support_tolerance) & jnp.isfinite(density),
            density,
            -jnp.inf,
        )

    def coefficient_score(self, value: ArrayLike, /) -> Array:
        coefficients, residual = self.layout.project(value)
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(residual > self.support_tolerance),
            "Ambient value lies outside the affine subspace.",
        )
        return self.coefficient_law.score(coefficients)

    def tangent_score(self, value: ArrayLike, /) -> Array:
        coefficient_score = self.coefficient_score(value)
        tangent_coefficients = self.layout.solve_gram(coefficient_score)
        tangent = oe.contract("ir,...r->...i", self.layout.basis, tangent_coefficients)
        return tangent.reshape(coefficient_score.shape[:-1] + self.event_shape)


class SubspaceGaussianDiffusion(StrictModule):
    """Run one Gaussian score diffusion in coefficients and synthesize ambient events."""

    layout: AffineSubspaceLayout
    coefficient_process: AbstractGaussianDiffusion
    process_id: str = eqx.field(static=True)

    def __init__(self, layout, coefficient_process, /, *, process_id: str | None = None):
        if not isinstance(layout, AffineSubspaceLayout):
            raise TypeError("layout must be an AffineSubspaceLayout.")
        if not isinstance(coefficient_process, AbstractGaussianDiffusion):
            raise TypeError("coefficient_process must implement AbstractGaussianDiffusion.")
        if coefficient_process.state_shape != (layout.rank,):
            raise ValueError("Coefficient diffusion dimension must equal subspace rank.")
        self.layout = layout
        self.coefficient_process = coefficient_process
        self.process_id = process_id or canonical_fingerprint(
            {
                "kind": "subspace-gaussian-diffusion",
                "layout_id": layout.layout_id,
                "coefficient_process_id": coefficient_process.process_id,
            }
        )

    def perturb(self, key: Key[Array, ""], value: ArrayLike, /, *, time: ArrayLike):
        coefficients, residual = self.layout.project(value)
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(residual > 1e-8),
            "Subspace diffusion input lies outside the represented support.",
        )
        perturbed = self.coefficient_process.perturb(key, coefficients, t1=time)
        return self.layout.synthesize(perturbed)

    def conditional_coefficient_score(
        self,
        perturbed: ArrayLike,
        clean: ArrayLike,
        /,
        *,
        time: ArrayLike,
    ) -> Array:
        noisy_coefficients, noisy_residual = self.layout.project(perturbed)
        clean_coefficients, clean_residual = self.layout.project(clean)
        score = self.coefficient_process.conditional_score(
            noisy_coefficients,
            clean_coefficients,
            t1=time,
        )
        return eqx.error_if(
            score,
            jnp.any(noisy_residual > 1e-8) | jnp.any(clean_residual > 1e-8),
            "Subspace score inputs lie outside the represented support.",
        )


__all__ = [
    "AffineSubspaceLayout",
    "SubspaceGaussianDiffusion",
    "SubspaceGaussianLaw",
]
