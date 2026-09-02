#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from opt_einsum import contract

from .._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from .._strict import StrictModule
from ._factorizations import PreparedFactorization
from ._materialization import MaterializationPolicy, materialize
from ._prepared import PreparedLinearSolve


class InertiaPolicy(StrictModule):
    """Tolerance-defined numerical inertia and bounded evidence source."""

    absolute_zero_tolerance: float = eqx.field(static=True)
    relative_zero_tolerance: float = eqx.field(static=True)
    certification_margin: float = eqx.field(static=True)
    source: Literal["provider", "bounded-dense"] = eqx.field(static=True)
    maximum_dense_dimension: int = eqx.field(static=True)
    materialization: MaterializationPolicy
    precision: PrecisionRequest = eqx.field(static=True)

    def __init__(
        self,
        absolute_zero_tolerance: float = 0.0,
        relative_zero_tolerance: float = 0.0,
        certification_margin: float = 8.0,
        source: Literal["provider", "bounded-dense"] = "bounded-dense",
        maximum_dense_dimension: int = 512,
        materialization: MaterializationPolicy | None = None,
        precision: PrecisionRequest | None = None,
    ):
        absolute = float(absolute_zero_tolerance)
        relative = float(relative_zero_tolerance)
        margin = float(certification_margin)
        dimension = int(maximum_dense_dimension)
        if (
            not all(isfinite(value) for value in (absolute, relative, margin))
            or absolute < 0
            or relative < 0
            or margin < 1
        ):
            raise ValueError("Inertia tolerances/margin are invalid.")
        if source not in ("provider", "bounded-dense"):
            raise ValueError("Inertia source must be provider or bounded-dense.")
        if dimension < 1:
            raise ValueError("maximum_dense_dimension must be positive.")
        precision_ = (
            PrecisionRequest("linalg-inertia", {"certification": "float32"})
            if precision is None
            else precision
        )
        if not isinstance(precision_, PrecisionRequest):
            raise TypeError("precision must be a PrecisionRequest or None.")
        requested = dict(precision_.requested)
        if set(requested) != {"certification"} or requested["certification"] is None:
            raise ValueError("Inertia precision must declare certification dtype only.")
        if not isinstance(requested["certification"], str) or requested[
            "certification"
        ].startswith("float8_"):
            raise ValueError("Inertia certification requires scalar float32 or wider.")
        self.absolute_zero_tolerance = absolute
        self.relative_zero_tolerance = relative
        self.certification_margin = margin
        self.source = source
        self.maximum_dense_dimension = dimension
        self.materialization = (
            MaterializationPolicy(
                max_entries=dimension * dimension,
                max_bytes=dimension * dimension * 16,
            )
            if materialization is None
            else materialization
        )
        self.precision = precision_


class InertiaEvidence(StrictModule):
    """Checkable evidence for one tolerance-defined numerical inertia count."""

    positive: Array
    negative: Array
    zero: Array
    threshold: Array
    spectral_error_bound: Array
    certified: Array
    provider_zero_capable: Array
    zero_count_reliable: Array
    source: str = eqx.field(static=True)
    factorization_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    non_claim: str = eqx.field(static=True)


def _owner(prepared: Any, /):
    if isinstance(prepared, PreparedFactorization):
        return prepared.operator, prepared.factorization_id, prepared.prepared_solve.state
    if isinstance(prepared, PreparedLinearSolve):
        return (
            prepared.problem.operator,
            f"{prepared.plan.plan_id}/numeric/{int(prepared.numeric_version)}",
            prepared.state,
        )
    raise TypeError("prepared must be a PreparedFactorization or PreparedLinearSolve.")


def _precision_evidence(policy: InertiaPolicy, dtype: Any, provider: str, /):
    name = jnp.dtype(dtype).name
    resolution = PrecisionResolution(
        policy.precision,
        provider,
        {"certification": name},
    )
    return PrecisionEvidenceEnvelope(resolution, {"certification": name})


def factorization_inertia(
    prepared: PreparedFactorization | PreparedLinearSolve,
    policy: InertiaPolicy,
    /,
) -> InertiaEvidence:
    """Return provider evidence or an independently bounded dense certificate."""
    if not isinstance(policy, InertiaPolicy):
        raise TypeError("policy must be an InertiaPolicy.")
    operator, factorization_id, state = _owner(prepared)
    if operator.source.size != operator.target.size:
        raise ValueError("Inertia requires a square endomorphism.")
    if policy.source == "provider":
        from .backends._spineax import SpineaxFactorState

        if not isinstance(state, SpineaxFactorState):
            raise ValueError("Prepared provider does not expose inertia evidence.")
        batch_shape = state.batch_shape
        unavailable = jnp.full(batch_shape, -1, dtype=jnp.int32)
        false = jnp.zeros(batch_shape, dtype=bool)
        nan = jnp.full(batch_shape, jnp.nan, dtype=state.storage.values.real.dtype)
        return InertiaEvidence(
            positive=state.positive_inertia,
            negative=state.negative_inertia,
            zero=unavailable,
            threshold=nan,
            spectral_error_bound=nan,
            certified=false,
            provider_zero_capable=false,
            zero_count_reliable=false,
            source="provider:spineax-cudss-partial",
            factorization_id=factorization_id,
            precision_evidence=_precision_evidence(
                policy, state.storage.values.real.dtype, "spineax-cudss"
            ),
            non_claim="provider positive/negative counts do not certify zero inertia",
        )
    if operator.source.size > policy.maximum_dense_dimension:
        raise ValueError("Bounded dense inertia dimension exceeds the declared limit.")
    matrix = materialize(operator, policy.materialization)
    requested = dict(policy.precision.requested)["certification"]
    certification_dtype = jnp.dtype(requested)
    if certification_dtype.itemsize < 4:
        raise ValueError("Inertia certification dtype must be float32 or wider.")
    dtype = (
        jnp.complex64
        if jnp.issubdtype(matrix.dtype, jnp.complexfloating)
        and certification_dtype.itemsize <= 4
        else jnp.complex128
        if jnp.issubdtype(matrix.dtype, jnp.complexfloating)
        else certification_dtype
    )
    matrix_ = matrix.astype(dtype)
    adjoint = jnp.conj(jnp.swapaxes(matrix_, -1, -2))
    hermitian_defect = jnp.max(jnp.abs(matrix_ - adjoint), axis=(-2, -1))
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix_)
    reconstruction = contract(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        eigenvalues,
        jnp.conj(eigenvectors),
        backend="jax",
    )
    reconstruction_defect = jnp.max(jnp.abs(matrix_ - reconstruction), axis=(-2, -1))
    identity = jnp.eye(operator.source.size, dtype=dtype)
    orthogonality = contract(
        "...ki,...kj->...ij",
        jnp.conj(eigenvectors),
        eigenvectors,
        backend="jax",
    )
    orthogonality_defect = jnp.max(jnp.abs(orthogonality - identity), axis=(-2, -1))
    scale = jnp.max(jnp.abs(eigenvalues), axis=-1)
    epsilon = jnp.finfo(certification_dtype).eps
    spectral_error = policy.certification_margin * (
        reconstruction_defect
        + hermitian_defect
        + scale * orthogonality_defect
        + epsilon * jnp.maximum(scale, 1)
    )
    threshold = policy.absolute_zero_tolerance + policy.relative_zero_tolerance * scale
    lower = eigenvalues - spectral_error[..., None]
    upper = eigenvalues + spectral_error[..., None]
    positive_mask = lower > threshold[..., None]
    negative_mask = upper < -threshold[..., None]
    zero_mask = (lower >= -threshold[..., None]) & (upper <= threshold[..., None])
    classified = positive_mask | negative_mask | zero_mask
    finite = (
        jnp.all(jnp.isfinite(matrix_), axis=(-2, -1))
        & jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        & jnp.isfinite(spectral_error)
    )
    hermitian = hermitian_defect <= jnp.maximum(
        threshold,
        policy.certification_margin * epsilon * jnp.maximum(scale, 1),
    )
    certified = finite & hermitian & jnp.all(classified, axis=-1)
    return InertiaEvidence(
        positive=jnp.sum(positive_mask, axis=-1, dtype=jnp.int32),
        negative=jnp.sum(negative_mask, axis=-1, dtype=jnp.int32),
        zero=jnp.sum(zero_mask, axis=-1, dtype=jnp.int32),
        threshold=threshold,
        spectral_error_bound=spectral_error,
        certified=certified,
        provider_zero_capable=jnp.zeros_like(certified),
        zero_count_reliable=certified,
        source="bounded-dense",
        factorization_id=factorization_id,
        precision_evidence=_precision_evidence(
            policy, certification_dtype, "phydrax-bounded-dense"
        ),
        non_claim="certifies numerical inertia at the declared threshold, not exact nullity",
    )


__all__ = ["InertiaEvidence", "InertiaPolicy", "factorization_inertia"]
