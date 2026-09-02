#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PSD-by-construction LPDO compression with finite error evidence."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ._core import LocallyPurifiedDensity


class LPDOCompressionPlan(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_purification_dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_dense_amplitude_elements: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_bond_dimension: int,
        maximum_purification_dimension: int,
        tolerance: float = 1e-10,
        maximum_dense_amplitude_elements: int = 1_000_000,
    ):
        bond, purification = (
            int(maximum_bond_dimension),
            int(maximum_purification_dimension),
        )
        dense = int(maximum_dense_amplitude_elements)
        if bond <= 0 or purification <= 0 or tolerance < 0.0 or dense <= 0:
            raise ValueError("LPDO compression capacities/tolerance are invalid.")
        self.maximum_bond_dimension = bond
        self.maximum_purification_dimension = purification
        self.tolerance = float(tolerance)
        self.maximum_dense_amplitude_elements = dense


class LPDOCompressionCertificate(StrictModule):
    state: LocallyPurifiedDensity
    original_trace: Array
    compressed_trace: Array
    trace_loss: Array
    discarded_purification_norm: Array
    discarded_bond_norm: Array
    purification_overlap: Array
    trace_distance_upper_bound: Array
    valid: Array
    positive_by_construction: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def compress_lpdo(
    state: LocallyPurifiedDensity,
    plan: LPDOCompressionPlan,
    /,
) -> LPDOCompressionCertificate:
    """Compress only a purification factor; arbitrary MPO input is rejected by type."""
    if not isinstance(state, LocallyPurifiedDensity):
        raise TypeError(
            "state must be LocallyPurifiedDensity; arbitrary MPO compression is not physicality preserving."
        )
    if not isinstance(plan, LPDOCompressionPlan):
        raise TypeError("plan must be LPDOCompressionPlan.")
    physical_size = 1
    purification_size = 1
    for physical, purification in zip(
        state.physical_dimensions, state.purification_dimensions, strict=True
    ):
        physical_size *= physical
        purification_size *= purification
    if physical_size * purification_size > plan.maximum_dense_amplitude_elements:
        raise MemoryError(
            "LPDO compression certificate exceeds dense amplitude evidence capacity."
        )
    original_amplitude = state._amplitude()
    tensors = [
        tensor[:, :, : min(tensor.shape[2], plan.maximum_purification_dimension), :]
        for tensor in state.tensors
    ]
    discarded_purification = jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.abs(original[:, :, compressed.shape[2] :, :]) ** 2)
                for original, compressed in zip(state.tensors, tensors, strict=True)
            ),
            jnp.asarray(0.0),
        )
    )
    discarded_bond_squared = jnp.asarray(0.0)
    for site in range(len(tensors) - 1):
        tensor = tensors[site]
        left, physical, purification, right = tensor.shape
        matrix = tensor.reshape((left * physical * purification, right))
        decomposition = factorize(DenseLinearOperator(matrix), FactorizationPolicy("svd"))
        svd = decomposition.prepared_solve.state
        retained = min(int(svd.singular_values.shape[0]), plan.maximum_bond_dimension)
        discarded_bond_squared = discarded_bond_squared + jnp.sum(
            jnp.abs(svd.singular_values[retained:]) ** 2
        )
        left_tensor = (svd.u[:, :retained] * svd.singular_values[:retained]).reshape(
            (left, physical, purification, retained)
        )
        transfer = svd.vh[:retained, :]
        next_tensor = jnp.tensordot(transfer, tensors[site + 1], axes=(1, 0))
        tensors[site] = left_tensor
        tensors[site + 1] = next_tensor
    compressed = LocallyPurifiedDensity(tensors, precision=state.precision)
    compressed_amplitude = compressed._amplitude()
    # Embed retained local auxiliary indices in the original tensor-product
    # purification layout before overlap/error comparison.
    original_tensor = original_amplitude.reshape(
        (physical_size,) + state.purification_dimensions
    )
    compressed_tensor = compressed_amplitude.reshape(
        (physical_size,) + compressed.purification_dimensions
    )
    retained_slices = (slice(None),) + tuple(
        slice(0, dimension) for dimension in compressed.purification_dimensions
    )
    padded_tensor = (
        jnp.zeros_like(original_tensor).at[retained_slices].set(compressed_tensor)
    )
    padded = padded_tensor.reshape(original_amplitude.shape)
    difference = original_amplitude - padded
    difference_norm = jnp.sqrt(jnp.sum(jnp.abs(difference) ** 2))
    original_norm = jnp.sqrt(jnp.sum(jnp.abs(original_amplitude) ** 2))
    compressed_norm = jnp.sqrt(jnp.sum(jnp.abs(compressed_amplitude) ** 2))
    original_trace = original_norm**2
    compressed_trace = compressed_norm**2
    overlap = jnp.sum(jnp.conj(original_amplitude) * padded)
    trace_distance_bound = difference_norm * (original_norm + compressed_norm)
    valid = (
        jnp.all(jnp.isfinite(original_amplitude))
        & jnp.all(jnp.isfinite(compressed_amplitude))
        & jnp.isfinite(trace_distance_bound)
        & (compressed_trace >= 0.0)
        & (trace_distance_bound >= 0.0)
    )
    return LPDOCompressionCertificate(
        state=compressed,
        original_trace=original_trace,
        compressed_trace=compressed_trace,
        trace_loss=original_trace - compressed_trace,
        discarded_purification_norm=discarded_purification,
        discarded_bond_norm=jnp.sqrt(discarded_bond_squared),
        purification_overlap=overlap,
        trace_distance_upper_bound=trace_distance_bound,
        valid=valid,
        positive_by_construction=True,
        claim="finite-lpdo-purification-factor-compression-not-arbitrary-mpo",
    )


__all__ = ["LPDOCompressionCertificate", "LPDOCompressionPlan", "compress_lpdo"]
