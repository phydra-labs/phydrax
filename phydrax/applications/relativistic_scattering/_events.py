#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polarization density algebra and fixed-capacity weighted event streams."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PolarizationDensity(StrictModule):
    """Finite-dimensional polarization density operator and validity evidence."""

    matrix: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    basis: str = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, /, *, basis: str = "helicity"):
        matrix_ = jnp.asarray(matrix)
        if (
            matrix_.ndim != 2
            or matrix_.shape[0] != matrix_.shape[1]
            or matrix_.shape[0] == 0
        ):
            raise ValueError("Polarization density matrices must be nonempty and square.")
        if not basis:
            raise ValueError("Polarization basis identity must be nonempty.")
        hermitian_residual = jnp.max(jnp.abs(matrix_ - jnp.conj(matrix_.T)))
        trace = jnp.trace(matrix_)
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (matrix_ + jnp.conj(matrix_.T)))
        tolerance = 64.0 * jnp.finfo(jnp.real(matrix_).dtype).eps
        self.matrix = matrix_
        self.valid = (
            jnp.all(jnp.isfinite(matrix_))
            & (hermitian_residual <= tolerance)
            & (jnp.abs(trace - 1.0) <= tolerance)
            & (jnp.min(eigenvalues) >= -tolerance)
        )
        self.dimension = matrix_.shape[0]
        self.basis = str(basis)


def pure_polarization_density(
    state: ArrayLike, /, *, basis: str = "helicity"
) -> PolarizationDensity:
    """Construct a normalized rank-one density operator."""
    state_ = jnp.asarray(state)
    if state_.ndim != 1 or state_.size == 0:
        raise ValueError("Pure polarization states must be nonempty vectors.")
    norm = jnp.real(jnp.vdot(state_, state_))
    matrix = state_[:, None] * jnp.conj(state_[None, :]) / norm
    return PolarizationDensity(matrix, basis=basis)


def mixed_polarization_density(
    densities: Sequence[PolarizationDensity], probabilities: ArrayLike, /
) -> PolarizationDensity:
    """Return a classical convex mixture in one common basis."""
    densities_ = tuple(densities)
    probabilities_ = jnp.asarray(probabilities)
    if not densities_ or probabilities_.shape != (len(densities_),):
        raise ValueError("Density mixture probabilities must align with inputs.")
    dimension = densities_[0].dimension
    basis = densities_[0].basis
    if any(
        density.dimension != dimension or density.basis != basis for density in densities_
    ):
        raise ValueError("Mixed polarization densities require one dimension and basis.")
    normalized = probabilities_ / jnp.sum(probabilities_)
    matrices = jnp.stack(tuple(density.matrix for density in densities_))
    return PolarizationDensity(
        ein.contract("n,nab->ab", normalized, matrices), basis=basis
    )


def rotate_polarization_density(
    density: PolarizationDensity, unitary: ArrayLike, /, *, basis: str | None = None
) -> PolarizationDensity:
    """Apply ``rho -> U rho U dagger`` without diagonalizing the state."""
    matrix = jnp.asarray(unitary)
    if matrix.shape != (density.dimension, density.dimension):
        raise ValueError("Polarization rotation has incompatible shape.")
    rotated = matrix @ density.matrix @ jnp.conj(matrix.T)
    return PolarizationDensity(rotated, basis=density.basis if basis is None else basis)


def polarization_expectation(
    density: PolarizationDensity, observable: ArrayLike, /
) -> Array:
    """Return ``Tr(rho observable)``."""
    observable_ = jnp.asarray(observable)
    if observable_.shape != density.matrix.shape:
        raise ValueError("Polarization observable has incompatible shape.")
    return jnp.trace(density.matrix @ observable_)


def tensor_polarization_density(
    left: PolarizationDensity, right: PolarizationDensity, /
) -> PolarizationDensity:
    """Tensor two independent polarization states."""
    product = ein.contract("ab,cd->acbd", left.matrix, right.matrix).reshape(
        (left.dimension * right.dimension, left.dimension * right.dimension)
    )
    return PolarizationDensity(product, basis=f"{left.basis}*{right.basis}")


def partial_trace_polarization(
    density: PolarizationDensity,
    dimensions: tuple[int, int],
    /,
    *,
    trace_right: bool,
) -> PolarizationDensity:
    """Partial trace of a bipartite polarization state."""
    left, right = map(int, dimensions)
    if left < 1 or right < 1 or left * right != density.dimension:
        raise ValueError("Bipartite dimensions do not factor the density matrix.")
    tensor = density.matrix.reshape((left, right, left, right))
    if trace_right:
        reduced = ein.contract("arbr->ab", tensor)
        basis = f"left({density.basis})"
    else:
        reduced = ein.contract("albl->ab", jnp.transpose(tensor, (1, 0, 3, 2)))
        basis = f"right({density.basis})"
    return PolarizationDensity(reduced, basis=basis)


def stokes_density(stokes: ArrayLike, /) -> PolarizationDensity:
    """Construct a photon helicity density from normalized Stokes parameters."""
    values = jnp.asarray(stokes)
    if values.shape != (3,):
        raise ValueError("Stokes polarization requires (Q, U, V).")
    q, u, v = values
    matrix = 0.5 * jnp.asarray([[1.0 + v, q - 1.0j * u], [q + 1.0j * u, 1.0 - v]])
    return PolarizationDensity(matrix, basis="photon-helicity")


def stokes_parameters(density: PolarizationDensity, /) -> Array:
    """Recover ``(Q, U, V)`` from a two-helicity density."""
    if density.dimension != 2:
        raise ValueError(
            "Stokes parameters require a two-dimensional polarization state."
        )
    q = jnp.real(density.matrix[0, 1] + density.matrix[1, 0])
    u = jnp.real(1.0j * (density.matrix[0, 1] - density.matrix[1, 0]))
    v = jnp.real(density.matrix[0, 0] - density.matrix[1, 1])
    return jnp.asarray([q, u, v])


class EventStatus(IntEnum):
    """Terminal status for weighted-to-unweighted conversion."""

    SUCCESS = 0
    SUPPORT_BOUND_VIOLATED = 1
    NEGATIVE_WEIGHT_NOT_ENABLED = 2
    NONFINITE_WEIGHT = 3


class WeightedEventStream(StrictModule):
    """Fixed-capacity momenta, signed weights, and active event slots."""

    momenta: Array
    weights: Array
    active: Array
    finite: Array
    capacity: int = eqx.field(static=True)
    multiplicity: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)

    def __init__(
        self,
        momenta: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        provenance: str,
    ):
        momenta_ = jnp.asarray(momenta)
        weights_ = jnp.asarray(weights)
        if momenta_.ndim != 3 or momenta_.shape[-1] != 4:
            raise ValueError("Event momenta must have shape (capacity, multiplicity, 4).")
        if weights_.shape != (momenta_.shape[0],):
            raise ValueError("Event weights must align with stream capacity.")
        active_ = (
            jnp.ones(weights_.shape, dtype=bool)
            if active is None
            else jnp.asarray(active, dtype=bool)
        )
        if active_.shape != weights_.shape or not provenance:
            raise ValueError("Event activity/provenance declaration is invalid.")
        self.momenta = momenta_
        self.weights = weights_
        self.active = active_
        self.finite = jnp.all(jnp.isfinite(momenta_), axis=(1, 2)) & jnp.isfinite(
            weights_
        )
        self.capacity = momenta_.shape[0]
        self.multiplicity = momenta_.shape[1]
        self.provenance = str(provenance)

    @property
    def signs(self) -> Array:
        """Per-event signs, including zero for exactly zero weights."""
        return jnp.sign(self.weights)

    @property
    def absolute_weights(self) -> Array:
        """Per-event nonnegative absolute weights."""
        return jnp.abs(self.weights)


class RejectionUnweightingPlan(StrictModule, NonTrainableState):
    """Certified absolute-weight support bound for exact rejection sampling."""

    support_bound: Array
    capacity: int = eqx.field(static=True)
    signed: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, support_bound: float, capacity: int, /, *, signed: bool = False):
        bound = float(support_bound)
        capacity_ = int(capacity)
        if not math.isfinite(bound) or bound <= 0.0 or capacity_ < 1:
            raise ValueError("Unweighting support bound and capacity must be positive.")
        self.support_bound = jnp.asarray(bound)
        self.capacity = capacity_
        self.signed = bool(signed)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-rejection-unweighting",
                "support_bound": bound,
                "capacity": capacity_,
                "signed": bool(signed),
            }
        )


class UnweightedEventStream(StrictModule):
    """Fixed-capacity accepted events, with signs retained when requested."""

    momenta: Array
    signs: Array
    active: Array
    acceptance_probability: Array
    accepted_count: Array
    status: Array
    support_excess: Array
    capacity: int = eqx.field(static=True)
    multiplicity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def rejection_unweight(
    stream: WeightedEventStream,
    plan: RejectionUnweightingPlan,
    key: Key[Array, ""],
    /,
) -> UnweightedEventStream:
    """Perform exact rejection unweighting, never clipping a violated bound."""
    if stream.capacity != plan.capacity:
        raise ValueError("Unweighting plan capacity must equal event stream capacity.")
    magnitudes = jnp.abs(stream.weights)
    active_magnitudes = jnp.where(stream.active, magnitudes, 0.0)
    support_excess = jnp.max(active_magnitudes - plan.support_bound)
    support_valid = support_excess <= 0.0
    sign_valid = plan.signed | jnp.all(
        jnp.where(stream.active, stream.weights >= 0.0, True)
    )
    finite = jnp.all(jnp.where(stream.active, stream.finite, True))
    probability = magnitudes / plan.support_bound
    draws = jr.uniform(key, (stream.capacity,))
    globally_valid = support_valid & sign_valid & finite
    accepted = stream.active & globally_valid & (draws < probability)
    signs = jnp.where(
        plan.signed, jnp.sign(stream.weights), jnp.ones_like(stream.weights)
    )
    status = jnp.where(
        ~finite,
        int(EventStatus.NONFINITE_WEIGHT),
        jnp.where(
            ~sign_valid,
            int(EventStatus.NEGATIVE_WEIGHT_NOT_ENABLED),
            jnp.where(
                ~support_valid,
                int(EventStatus.SUPPORT_BOUND_VIOLATED),
                int(EventStatus.SUCCESS),
            ),
        ),
    )
    return UnweightedEventStream(
        stream.momenta,
        signs,
        accepted,
        probability,
        jnp.sum(accepted, dtype=jnp.int32),
        status.astype(jnp.int32),
        jnp.maximum(support_excess, 0.0),
        stream.capacity,
        stream.multiplicity,
        plan.plan_id,
    )


__all__ = [
    "EventStatus",
    "PolarizationDensity",
    "RejectionUnweightingPlan",
    "UnweightedEventStream",
    "WeightedEventStream",
    "mixed_polarization_density",
    "partial_trace_polarization",
    "polarization_expectation",
    "pure_polarization_density",
    "rejection_unweight",
    "rotate_polarization_density",
    "stokes_density",
    "stokes_parameters",
    "tensor_polarization_density",
]
