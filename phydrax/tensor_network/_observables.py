#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._canonical import canonicalize_mps
from ._core import MatrixProductState
from ._environments import mps_norm_squared


class FiniteCorrelationResult(StrictModule):
    expectation: Array
    connected: Array
    normalization: Array
    valid: Array


class FiniteReducedDensityResult(StrictModule):
    density: Array
    trace: Array
    hermiticity_residual: Array
    positivity_floor: Array
    valid: Array
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)


class FiniteEntanglementResult(StrictModule):
    spectrum: Array
    active: Array
    entropy: Array
    normalization: Array
    bond: int = eqx.field(static=True)


class FiniteTransferStatus(IntEnum):
    SUCCESS = 0
    NONSQUARE_TRANSFER = 1
    NONFINITE = 2


class FiniteTransferSpectrum(StrictModule):
    eigenvalues: Array
    active: Array
    spectral_radius: Array
    status: Array
    site: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteTransferStatus.SUCCESS)


def _identity_environments(
    state: MatrixProductState, /
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    precision = state.precision
    tensors = precision.accumulation(state.tensors)
    dtype = tensors[0].dtype
    left = [jnp.ones((1, 1), dtype=dtype)]
    for tensor in tensors:
        left.append(oe.contract("ab,api,bpj->ij", left[-1], jnp.conj(tensor), tensor))
    right: list[Array] = [jnp.zeros((0, 0), dtype=dtype) for _ in tensors] + [
        jnp.ones((1, 1), dtype=dtype)
    ]
    for index in range(state.site_count - 1, -1, -1):
        tensor = tensors[index]
        right[index] = oe.contract(
            "api,bpj,ij->ab", jnp.conj(tensor), tensor, right[index + 1]
        )
    return tuple(left), tuple(right)


def finite_correlation_matrix(
    state: MatrixProductState,
    operator: ArrayLike,
    /,
    *,
    second_operator: ArrayLike | None = None,
) -> FiniteCorrelationResult:
    """Compute all ordered two-point and connected correlators on a finite chain."""
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    if len(set(state.physical_dimensions)) != 1:
        raise ValueError("Correlation matrices require a uniform local dimension.")
    dimension = state.physical_dimensions[0]
    first = state.precision.accumulation(jnp.asarray(operator))
    second = (
        first
        if second_operator is None
        else state.precision.accumulation(jnp.asarray(second_operator))
    )
    if first.shape != (dimension, dimension) or second.shape != (dimension, dimension):
        raise ValueError("Correlation operators must match the local dimension.")
    tensors = state.precision.accumulation(state.tensors)
    norm = mps_norm_squared(state)
    norm = eqx.error_if(
        norm, ~jnp.isfinite(norm) | (norm <= 0.0), "MPS norm must be finite and positive."
    )
    one_first = jnp.zeros((state.site_count,), dtype=jnp.result_type(tensors[0], first))
    one_second = jnp.zeros((state.site_count,), dtype=jnp.result_type(tensors[0], second))
    values = jnp.zeros(
        (state.site_count, state.site_count),
        dtype=jnp.result_type(tensors[0], first, second),
    )
    for row in range(state.site_count):
        for column in range(state.site_count):
            environment = jnp.ones((1, 1), dtype=values.dtype)
            for site, tensor in enumerate(tensors):
                if site == row == column:
                    insertion = first @ second
                elif site == row:
                    insertion = first
                elif site == column:
                    insertion = second
                else:
                    insertion = jnp.eye(dimension, dtype=values.dtype)
                environment = oe.contract(
                    "ab,api,pq,bqj->ij",
                    environment,
                    jnp.conj(tensor),
                    insertion,
                    tensor,
                )
            values = values.at[row, column].set(environment.reshape(()) / norm)
    for site in range(state.site_count):
        environment_first = jnp.ones((1, 1), dtype=values.dtype)
        environment_second = jnp.ones((1, 1), dtype=values.dtype)
        for index, tensor in enumerate(tensors):
            insertion_first = (
                first if index == site else jnp.eye(dimension, dtype=values.dtype)
            )
            insertion_second = (
                second if index == site else jnp.eye(dimension, dtype=values.dtype)
            )
            environment_first = oe.contract(
                "ab,api,pq,bqj->ij",
                environment_first,
                jnp.conj(tensor),
                insertion_first,
                tensor,
            )
            environment_second = oe.contract(
                "ab,api,pq,bqj->ij",
                environment_second,
                jnp.conj(tensor),
                insertion_second,
                tensor,
            )
        one_first = one_first.at[site].set(environment_first.reshape(()) / norm)
        one_second = one_second.at[site].set(environment_second.reshape(()) / norm)
    connected = values - one_first[:, None] * one_second[None, :]
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(connected))
    return FiniteCorrelationResult(
        state.precision.output(values),
        state.precision.output(connected),
        norm,
        valid,
    )


def finite_reduced_density(
    state: MatrixProductState,
    start: int,
    stop: int,
    /,
    *,
    maximum_elements: int = 1_000_000,
) -> FiniteReducedDensityResult:
    """Contract a normalized contiguous reduced density matrix without global densification."""
    start_ = int(start)
    stop_ = int(stop)
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    if not 0 <= start_ < stop_ <= state.site_count:
        raise ValueError("Reduced-density range must be a nonempty subchain.")
    block_dimension = 1
    for dimension in state.physical_dimensions[start_:stop_]:
        block_dimension *= dimension
    required = block_dimension * block_dimension
    if int(maximum_elements) < required:
        raise MemoryError("Reduced density exceeds maximum_elements.")
    left, right = _identity_environments(state)
    tensors = state.precision.accumulation(state.tensors)
    block = tensors[start_]
    for tensor in tensors[start_ + 1 : stop_]:
        joined = oe.contract("apr,rsb->apsb", block, tensor)
        block = joined.reshape(
            (joined.shape[0], joined.shape[1] * joined.shape[2], joined.shape[3])
        )
    density = oe.contract(
        "ab,apc,bqd,cd->pq",
        left[start_],
        jnp.conj(block),
        block,
        right[stop_],
    )
    trace = jnp.real(jnp.trace(density))
    trace = eqx.error_if(
        trace, ~jnp.isfinite(trace) | (trace <= 0.0), "Reduced density has invalid trace."
    )
    density = density / trace
    hermiticity = jnp.linalg.norm(density - jnp.conj(density.T)) / jnp.maximum(
        jnp.linalg.norm(density), 1.0
    )
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (density + jnp.conj(density.T)))
    floor = jnp.min(eigenvalues)
    valid = jnp.all(jnp.isfinite(density)) & (hermiticity <= 1e-7) & (floor >= -1e-7)
    return FiniteReducedDensityResult(
        state.precision.output(density),
        trace,
        hermiticity,
        floor,
        valid,
        start_,
        stop_,
    )


def finite_entanglement_spectrum(
    state: MatrixProductState,
    bond: int,
    /,
    *,
    maximum_rank: int,
    entropy_floor: float = 1e-15,
) -> FiniteEntanglementResult:
    bond_ = int(bond)
    capacity = int(maximum_rank)
    floor = float(entropy_floor)
    if not 0 <= bond_ < state.site_count - 1:
        raise ValueError("Entanglement bond is outside the finite chain.")
    if capacity < 1 or not isfinite(floor) or floor <= 0.0:
        raise ValueError("maximum_rank and entropy_floor must be positive.")
    canonical, _ = canonicalize_mps(state, center=bond_, normalize=True)
    center = canonical.precision.factorization(canonical.tensors[bond_])
    singular = jnp.linalg.svd(center.reshape((-1, center.shape[-1])), compute_uv=False)
    probabilities = jnp.square(jnp.abs(singular))
    normalization = jnp.sum(probabilities)
    probabilities = probabilities / jnp.maximum(normalization, floor)
    count = min(capacity, int(probabilities.shape[0]))
    spectrum = (
        jnp.zeros((capacity,), dtype=probabilities.dtype)
        .at[:count]
        .set(probabilities[:count])
    )
    active = jnp.arange(capacity) < count
    entropy = -jnp.sum(
        jnp.where(active & (spectrum > floor), spectrum * jnp.log(spectrum), 0.0)
    )
    return FiniteEntanglementResult(spectrum, active, entropy, normalization, bond_)


def finite_transfer_spectrum(
    state: MatrixProductState,
    site: int,
    /,
    *,
    maximum_modes: int,
) -> FiniteTransferSpectrum:
    site_ = int(site)
    capacity = int(maximum_modes)
    if not 0 <= site_ < state.site_count or capacity < 1:
        raise ValueError("site must be valid and maximum_modes positive.")
    tensor = state.precision.accumulation(state.tensors[site_])
    left = int(tensor.shape[0])
    right = int(tensor.shape[-1])
    dtype = jnp.result_type(tensor)
    eigenvalues = jnp.full(
        (capacity,), jnp.nan + 0j, dtype=jnp.result_type(dtype, jnp.complex64)
    )
    active = jnp.zeros((capacity,), dtype=bool)
    if left != right:
        return FiniteTransferSpectrum(
            eigenvalues,
            active,
            jnp.asarray(jnp.nan, dtype=tensor.real.dtype),
            jnp.asarray(int(FiniteTransferStatus.NONSQUARE_TRANSFER), dtype=jnp.int32),
            site_,
        )
    transfer = oe.contract("apr,bps->abrs", jnp.conj(tensor), tensor).reshape(
        (left * left, right * right)
    )
    computed = jnp.linalg.eigvals(transfer)
    order = jnp.argsort(jnp.abs(computed))[::-1]
    computed = computed[order]
    count = min(capacity, int(computed.shape[0]))
    eigenvalues = eigenvalues.at[:count].set(computed[:count])
    active = active.at[:count].set(True)
    finite = jnp.all(jnp.isfinite(computed))
    status = jnp.where(
        finite, int(FiniteTransferStatus.SUCCESS), int(FiniteTransferStatus.NONFINITE)
    )
    radius = jnp.where(finite, jnp.max(jnp.abs(computed)), jnp.nan)
    return FiniteTransferSpectrum(
        eigenvalues, active, radius, status.astype(jnp.int32), site_
    )


__all__ = [
    "FiniteCorrelationResult",
    "FiniteEntanglementResult",
    "FiniteReducedDensityResult",
    "FiniteTransferSpectrum",
    "FiniteTransferStatus",
    "finite_correlation_matrix",
    "finite_entanglement_spectrum",
    "finite_reduced_density",
    "finite_transfer_spectrum",
]
