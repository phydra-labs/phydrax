#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified finite-dimensional completely-positive trace-preserving maps.

The canonical Choi tensor has axes ``(output, input, output, input)`` with
``C[a,i,b,j] = sum_k K[k,a,i] conj(K[k,b,j])``. Consequently application is
exactly ``out[a,b] = contract('aibj,ij->ab', choi4, rho)``. Dimensions and the
source representation are explicit; no orientation is inferred and invalid maps are
never repaired during execution.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._strict import StrictModule
from ...linalg import HermitianSpectrum


class FiniteChannelPhysicalityEvidence(StrictModule):
    """Finite Choi CP/TP evidence for one declared input/output space."""

    choi_hermiticity_residual: Array
    minimum_choi_eigenvalue: Array
    choi_scale: Array
    trace_preservation_residual: Array
    representation_reconstruction_residual: Array
    active_kraus_count: Array
    valid: Array
    tolerance: float = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    source_representation: str = eqx.field(static=True)
    vectorization: str = eqx.field(static=True)


class FiniteCPTPMap(StrictModule):
    """Canonical finite map with physicality evidence and explicit dimensions."""

    choi_matrix: Array
    superoperator: Array
    evidence: FiniteChannelPhysicalityEvidence
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    source_representation: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.evidence.valid

    def apply(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        expected = (self.input_dimension, self.input_dimension)
        if value.shape != expected:
            raise ValueError(f"density must have shape {expected}; got {value.shape}.")
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            value = value.astype(jnp.result_type(value.dtype, self.choi_matrix.dtype))
        choi4 = self.choi_matrix.reshape(
            self.output_dimension,
            self.input_dimension,
            self.output_dimension,
            self.input_dimension,
        )
        return contract("aibj,ij->ab", choi4, value)


class FiniteChannelFactorizationPolicy(StrictModule):
    """Preparation-only Choi factorization and numerical-cleanup policy."""

    cleanup: Literal["reject", "bounded-numerical"] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_cleanup_norm: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        cleanup: Literal["reject", "bounded-numerical"] = "reject",
        tolerance: float = 1e-8,
        maximum_cleanup_norm: float = 1e-8,
    ):
        if cleanup not in ("reject", "bounded-numerical"):
            raise ValueError("cleanup must be 'reject' or 'bounded-numerical'.")
        tolerance_ = float(tolerance)
        cleanup_limit = float(maximum_cleanup_norm)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        if not np.isfinite(cleanup_limit) or cleanup_limit < 0.0:
            raise ValueError("maximum_cleanup_norm must be finite and non-negative.")
        self.cleanup = cleanup
        self.tolerance = tolerance_
        self.maximum_cleanup_norm = cleanup_limit


class FiniteKrausFactorization(StrictModule):
    """Fixed-capacity Kraus preparation result; inactive rows are zero."""

    kraus: Array
    active: Array
    cleanup_norm: Array
    reconstruction_residual: Array
    valid: Array
    policy: FiniteChannelFactorizationPolicy
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)


def _dimensions(input_dimension: int, output_dimension: int) -> tuple[int, int]:
    input_ = int(input_dimension)
    output_ = int(output_dimension)
    if input_ <= 0 or output_ <= 0:
        raise ValueError("Finite channel dimensions must be positive.")
    return input_, output_


def _complex_matrix(value: ArrayLike, shape: tuple[int, int], role: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{role} must have shape {shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array.astype(jnp.result_type(array.dtype, 1j))


def _superoperator_from_choi(
    choi: Array, input_dimension: int, output_dimension: int
) -> Array:
    choi4 = choi.reshape(
        output_dimension, input_dimension, output_dimension, input_dimension
    )
    return jnp.transpose(choi4, (0, 2, 1, 3)).reshape(
        output_dimension * output_dimension,
        input_dimension * input_dimension,
    )


def _choi_from_superoperator(
    superoperator: Array, input_dimension: int, output_dimension: int
) -> Array:
    super4 = superoperator.reshape(
        output_dimension, output_dimension, input_dimension, input_dimension
    )
    return jnp.transpose(super4, (0, 2, 1, 3)).reshape(
        output_dimension * input_dimension,
        output_dimension * input_dimension,
    )


def _physicality(
    choi: Array,
    input_dimension: int,
    output_dimension: int,
    *,
    source_representation: str,
    reconstruction_residual: ArrayLike,
    active_kraus_count: ArrayLike,
    tolerance: float,
) -> FiniteChannelPhysicalityEvidence:
    roundoff = np.finfo(np.dtype(choi.real.dtype)).eps * choi.shape[0]
    effective_tolerance = max(float(tolerance), float(roundoff))
    hermitian = 0.5 * (choi + jnp.conj(choi.T))
    spectrum = HermitianSpectrum(hermitian, tolerance=effective_tolerance)
    scale = jnp.maximum(jnp.max(jnp.abs(spectrum.eigenvalues)), 1.0)
    choi4 = choi.reshape(
        output_dimension, input_dimension, output_dimension, input_dimension
    )
    partial = jnp.trace(choi4, axis1=0, axis2=2)
    tp_residual = jnp.max(jnp.abs(partial - jnp.eye(input_dimension, dtype=choi.dtype)))
    hermiticity = jnp.max(jnp.abs(choi - jnp.conj(choi.T)))
    reconstruction = jnp.asarray(reconstruction_residual, dtype=scale.dtype)
    count = jnp.asarray(active_kraus_count, dtype=jnp.int32)
    threshold = jnp.asarray(effective_tolerance, dtype=scale.dtype) * scale
    valid = (
        jnp.all(jnp.isfinite(choi))
        & spectrum.valid
        & (hermiticity <= threshold)
        & (spectrum.minimum_eigenvalue >= -threshold)
        & (tp_residual <= threshold)
        & jnp.isfinite(reconstruction)
        & (reconstruction <= threshold)
    )
    return FiniteChannelPhysicalityEvidence(
        choi_hermiticity_residual=hermiticity,
        minimum_choi_eigenvalue=spectrum.minimum_eigenvalue,
        choi_scale=scale,
        trace_preservation_residual=tp_residual,
        representation_reconstruction_residual=reconstruction,
        active_kraus_count=count,
        valid=valid,
        tolerance=effective_tolerance,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        source_representation=source_representation,
        vectorization="row-major/output-input-choi",
    )


def _map_from_choi(
    choi: Array,
    input_dimension: int,
    output_dimension: int,
    *,
    source_representation: str,
    superoperator: Array | None = None,
    reconstruction_residual: ArrayLike = 0.0,
    active_kraus_count: ArrayLike = 0,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    canonical_superoperator = (
        _superoperator_from_choi(choi, input_dimension, output_dimension)
        if superoperator is None
        else superoperator
    )
    evidence = _physicality(
        choi,
        input_dimension,
        output_dimension,
        source_representation=source_representation,
        reconstruction_residual=reconstruction_residual,
        active_kraus_count=active_kraus_count,
        tolerance=tolerance,
    )
    return FiniteCPTPMap(
        choi_matrix=choi,
        superoperator=canonical_superoperator,
        evidence=evidence,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        source_representation=source_representation,
    )


def _map_from_superoperator(
    superoperator: Array,
    input_dimension: int,
    output_dimension: int,
    *,
    source_representation: str,
    tolerance: float,
) -> FiniteCPTPMap:
    choi = _choi_from_superoperator(superoperator, input_dimension, output_dimension)
    reconstructed = _superoperator_from_choi(choi, input_dimension, output_dimension)
    return _map_from_choi(
        choi,
        input_dimension,
        output_dimension,
        source_representation=source_representation,
        superoperator=superoperator,
        reconstruction_residual=jnp.max(jnp.abs(reconstructed - superoperator)),
        tolerance=tolerance,
    )


def finite_cptp_from_kraus(
    kraus: ArrayLike,
    /,
    *,
    active: ArrayLike | None = None,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    """Construct a canonical map from fixed-capacity rectangular Kraus operators."""
    operators = jnp.asarray(kraus)
    if operators.ndim != 3 or min(operators.shape) < 1:
        raise ValueError("kraus must have shape (capacity, output, input).")
    operators = operators.astype(jnp.result_type(operators.dtype, 1j))
    capacity, output_dimension, input_dimension = map(int, operators.shape)
    mask = (
        jnp.ones((capacity,), dtype=bool)
        if active is None
        else jnp.asarray(active, dtype=bool)
    )
    if mask.shape != (capacity,):
        raise ValueError(f"active must have shape ({capacity},); got {mask.shape}.")
    masked = jnp.where(mask[:, None, None], operators, 0.0)
    vectors = jnp.reshape(masked, (capacity, -1))
    choi = contract("ki,kj->ij", vectors, jnp.conj(vectors))
    superoperator = contract("kai,kbj->abij", masked, jnp.conj(masked)).reshape(
        output_dimension * output_dimension, input_dimension * input_dimension
    )
    reconstructed = _superoperator_from_choi(choi, input_dimension, output_dimension)
    residual = jnp.max(jnp.abs(reconstructed - superoperator))
    return _map_from_choi(
        choi,
        input_dimension,
        output_dimension,
        source_representation="kraus",
        superoperator=superoperator,
        reconstruction_residual=residual,
        active_kraus_count=jnp.sum(mask),
        tolerance=tolerance,
    )


def finite_cptp_from_choi(
    choi: ArrayLike,
    input_dimension: int,
    output_dimension: int,
    /,
    *,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    input_, output_ = _dimensions(input_dimension, output_dimension)
    matrix = _complex_matrix(choi, (input_ * output_, input_ * output_), "choi")
    return _map_from_choi(
        matrix, input_, output_, source_representation="choi", tolerance=tolerance
    )


def finite_cptp_from_superoperator(
    superoperator: ArrayLike,
    input_dimension: int,
    output_dimension: int,
    /,
    *,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    input_, output_ = _dimensions(input_dimension, output_dimension)
    matrix = _complex_matrix(
        superoperator, (output_ * output_, input_ * input_), "superoperator"
    )
    return _map_from_superoperator(
        matrix,
        input_,
        output_,
        source_representation="superoperator",
        tolerance=tolerance,
    )


def finite_cptp_from_unitary(
    unitary: ArrayLike, /, *, tolerance: float = 1e-8
) -> FiniteCPTPMap:
    value = jnp.asarray(unitary)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError("unitary must be a nonempty square matrix.")
    value = value.astype(jnp.result_type(value.dtype, 1j))
    return finite_cptp_from_kraus(value[None, ...], tolerance=tolerance)


def apply_finite_cptp(channel: FiniteCPTPMap, density: ArrayLike, /) -> Array:
    if not isinstance(channel, FiniteCPTPMap):
        raise TypeError("channel must be a FiniteCPTPMap.")
    return channel.apply(density)


def compose_finite_cptp(
    after: FiniteCPTPMap,
    before: FiniteCPTPMap,
    /,
    *,
    tolerance: float | None = None,
) -> FiniteCPTPMap:
    """Return ``after ∘ before`` while retaining canonical physicality evidence."""
    if not isinstance(after, FiniteCPTPMap) or not isinstance(before, FiniteCPTPMap):
        raise TypeError("after and before must be FiniteCPTPMap values.")
    if before.output_dimension != after.input_dimension:
        raise ValueError("Finite channel composition dimensions do not agree.")
    threshold = (
        max(after.evidence.tolerance, before.evidence.tolerance)
        if tolerance is None
        else float(tolerance)
    )
    superoperator = after.superoperator @ before.superoperator
    return _map_from_superoperator(
        superoperator,
        before.input_dimension,
        after.output_dimension,
        source_representation="composition",
        tolerance=threshold,
    )


def tensor_finite_cptp(
    left: FiniteCPTPMap,
    right: FiniteCPTPMap,
    /,
    *,
    tolerance: float | None = None,
) -> FiniteCPTPMap:
    """Tensor two channels in left-major Hilbert order."""
    if not isinstance(left, FiniteCPTPMap) or not isinstance(right, FiniteCPTPMap):
        raise TypeError("left and right must be FiniteCPTPMap values.")
    lo, li = left.output_dimension, left.input_dimension
    ro, ri = right.output_dimension, right.input_dimension
    left4 = left.superoperator.reshape(lo, lo, li, li)
    right4 = right.superoperator.reshape(ro, ro, ri, ri)
    combined = contract("abij,cdkl->acbdikjl", left4, right4).reshape(
        (lo * ro) ** 2, (li * ri) ** 2
    )
    threshold = (
        max(left.evidence.tolerance, right.evidence.tolerance)
        if tolerance is None
        else float(tolerance)
    )
    return _map_from_superoperator(
        combined,
        li * ri,
        lo * ro,
        source_representation="tensor",
        tolerance=threshold,
    )


def factor_finite_cptp(
    channel: FiniteCPTPMap,
    /,
    *,
    policy: FiniteChannelFactorizationPolicy | None = None,
) -> FiniteKrausFactorization:
    """Prepare fixed-capacity Kraus factors; cleanup is explicit and bounded."""
    if not isinstance(channel, FiniteCPTPMap):
        raise TypeError("channel must be a FiniteCPTPMap.")
    policy_ = FiniteChannelFactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, FiniteChannelFactorizationPolicy):
        raise TypeError("policy must be FiniteChannelFactorizationPolicy or None.")
    spectrum = HermitianSpectrum(
        0.5 * (channel.choi_matrix + jnp.conj(channel.choi_matrix.T)),
        tolerance=policy_.tolerance,
    )
    negative = jnp.minimum(spectrum.eigenvalues, 0.0)
    cleanup_norm = jnp.sqrt(jnp.sum(negative * negative))
    if policy_.cleanup == "reject":
        cleaned = spectrum.eigenvalues
        cleanup_valid = spectrum.minimum_eigenvalue >= 0.0
    else:
        cleaned = jnp.maximum(spectrum.eigenvalues, 0.0)
        cleanup_valid = cleanup_norm <= policy_.maximum_cleanup_norm
    active = cleaned > policy_.tolerance
    magnitudes = jnp.sqrt(jnp.where(active, cleaned, 0.0))
    vectors = jnp.swapaxes(spectrum.eigenvectors, 0, 1) * magnitudes[:, None]
    kraus = vectors.reshape((-1, channel.output_dimension, channel.input_dimension))
    rebuilt = finite_cptp_from_kraus(
        kraus,
        active=active,
        tolerance=max(policy_.tolerance, channel.evidence.tolerance),
    )
    residual = jnp.max(jnp.abs(rebuilt.choi_matrix - channel.choi_matrix))
    valid = (
        channel.valid
        & spectrum.valid
        & cleanup_valid
        & (residual <= jnp.maximum(policy_.maximum_cleanup_norm, policy_.tolerance))
    )
    return FiniteKrausFactorization(
        kraus=kraus,
        active=active,
        cleanup_norm=cleanup_norm,
        reconstruction_residual=residual,
        valid=valid,
        policy=policy_,
        input_dimension=channel.input_dimension,
        output_dimension=channel.output_dimension,
    )


def finite_cptp_from_local_kraus_operation(
    operation,
    /,
    *,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    """Adapt the canonical local Kraus IR operation without replacing its role."""
    from ._operations import LocalKrausChannelOperation

    if not isinstance(operation, LocalKrausChannelOperation):
        raise TypeError("operation must be LocalKrausChannelOperation.")
    return finite_cptp_from_kraus(
        operation.kraus,
        tolerance=tolerance,
    )


def finite_cptp_from_prepared_local_kraus_channel(
    prepared,
    /,
    *,
    tolerance: float = 1e-8,
) -> FiniteCPTPMap:
    """Adapt the specialized tensor-network local channel with retained evidence."""
    from ...tensor_network._local_lindblad import PreparedLocalKrausChannel

    if not isinstance(prepared, PreparedLocalKrausChannel):
        raise TypeError("prepared must be PreparedLocalKrausChannel.")
    return finite_cptp_from_superoperator(
        prepared.superoperator,
        prepared.dimension,
        prepared.dimension,
        tolerance=tolerance,
    )


def finite_cptp_to_local_kraus_operation(
    channel: FiniteCPTPMap,
    target_wire_ids,
    /,
    *,
    policy: FiniteChannelFactorizationPolicy | None = None,
):
    """Prepare an explicit canonical-IR Kraus operation; never factor at runtime."""
    from ._operations import LocalKrausChannelOperation

    factor = factor_finite_cptp(channel, policy=policy)
    if not bool(np.asarray(factor.valid)):
        raise ValueError("Finite CPTP map could not be factored within policy.")
    return LocalKrausChannelOperation(
        factor.kraus,
        target_wire_ids,
    )


__all__ = [
    "FiniteCPTPMap",
    "FiniteChannelFactorizationPolicy",
    "FiniteChannelPhysicalityEvidence",
    "FiniteKrausFactorization",
    "apply_finite_cptp",
    "compose_finite_cptp",
    "factor_finite_cptp",
    "finite_cptp_from_local_kraus_operation",
    "finite_cptp_from_prepared_local_kraus_channel",
    "finite_cptp_from_choi",
    "finite_cptp_from_kraus",
    "finite_cptp_from_superoperator",
    "finite_cptp_from_unitary",
    "finite_cptp_to_local_kraus_operation",
    "tensor_finite_cptp",
]
