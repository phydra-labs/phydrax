#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator, adjoint
from ._spaces import _coordinate_dtype


CertificateEvidence: TypeAlias = Literal["construction", "verified", "asserted"]
CertificateScope: TypeAlias = Literal["structural", "numerical"]


def _operator_numeric_fingerprint(operator: AbstractLinearOperator, /) -> str:
    return canonical_fingerprint(
        {
            "operator": operator.operator_id,
            "arrays": array_tree_fingerprint(operator),
        }
    )


def _validate_evidence(value: CertificateEvidence, /) -> CertificateEvidence:
    if value not in ("construction", "verified", "asserted"):
        raise ValueError(
            "certificate evidence must be 'construction', 'verified', or 'asserted'."
        )
    return value


def _validate_scope(value: CertificateScope, /) -> CertificateScope:
    if value not in ("structural", "numerical"):
        raise ValueError("certificate scope must be 'structural' or 'numerical'.")
    return value


def _subspace_residuals(operator: AbstractLinearOperator, subspace: Any, /) -> Array:
    capacity = subspace.capacity
    if capacity == 0:
        return jnp.zeros((0,), dtype=jnp.asarray(0.0).dtype)
    mask = jnp.arange(capacity) < subspace.dimension
    columns = jnp.where(mask[None, :], subspace.basis, 0)
    real_dtype = columns.real.dtype
    floor = jnp.finfo(real_dtype).tiny

    def residual(column):
        vector = operator.source.unflatten(column)
        image = operator.mv(vector)
        input_norm = jnp.sqrt(
            jnp.maximum(jnp.real(operator.source.inner(vector, vector)), 0.0)
        )
        output_norm = jnp.sqrt(
            jnp.maximum(jnp.real(operator.target.inner(image, image)), 0.0)
        )
        return output_norm / jnp.maximum(input_norm, floor)

    values = jax.vmap(residual, in_axes=1)(columns)
    return jnp.where(mask, values, 0.0)


class KernelCertificate(StrictModule):
    """Auditable right/left-kernel evidence scoped to one operator structure or value."""

    right: Any
    left: Any
    right_residual_norms: Array
    left_residual_norms: Array
    valid: Array
    operator_id: str = eqx.field(static=True)
    evidence: CertificateEvidence = eqx.field(static=True)
    scope: CertificateScope = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    numeric_fingerprint: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        right: Any,
        /,
        *,
        left: Any | None = None,
        evidence: CertificateEvidence = "verified",
        scope: CertificateScope = "numerical",
        complete: bool = False,
        tolerance: float = 1e-10,
    ):
        from ._subspaces import LinearSubspace

        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("Kernel certificates require an unbatched operator.")
        if not isinstance(right, LinearSubspace):
            raise TypeError("right must be a LinearSubspace.")
        if (
            left is None
            and operator.properties.certifies("self_adjoint")
            and operator.source.compatible(operator.target)
        ):
            left = right
        if left is not None and not isinstance(left, LinearSubspace):
            raise TypeError("left must be a LinearSubspace or None.")
        if not right.space.compatible(operator.source):
            raise ValueError("Right kernel space must match the operator source.")
        if left is not None and not left.space.compatible(operator.target):
            raise ValueError("Left kernel space must match the operator target.")
        evidence_ = _validate_evidence(evidence)
        scope_ = _validate_scope(scope)
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError(
                "Kernel certificate tolerance must be finite and non-negative."
            )
        right_residuals = _subspace_residuals(operator, right)
        left_residuals = (
            jnp.zeros((0,), dtype=right_residuals.dtype)
            if left is None
            else _subspace_residuals(adjoint(operator), left)
        )
        right_mask = jnp.arange(right.capacity) < right.dimension
        left_mask = (
            jnp.zeros((0,), dtype=bool)
            if left is None
            else jnp.arange(left.capacity) < left.dimension
        )
        valid = (
            jnp.all(jnp.isfinite(right_residuals))
            & jnp.all(jnp.where(right_mask, right_residuals <= tolerance_, True))
            & jnp.all(jnp.isfinite(left_residuals))
            & jnp.all(jnp.where(left_mask, left_residuals <= tolerance_, True))
        )
        numeric_fingerprint = _operator_numeric_fingerprint(operator)
        structure_id = canonical_fingerprint(
            {
                "kind": "kernel-certificate-structure",
                "operator": operator.operator_id,
                "right": right.subspace_id,
                "left": None if left is None else left.subspace_id,
                "evidence": evidence_,
                "scope": scope_,
                "complete": bool(complete),
            }
        )
        certificate_id = canonical_fingerprint(
            {
                "kind": "kernel-certificate",
                "structure": structure_id,
                "numeric": numeric_fingerprint,
                "tolerance": tolerance_,
            }
        )
        self.right = right
        self.left = left
        self.right_residual_norms = right_residuals
        self.left_residual_norms = left_residuals
        self.valid = valid
        self.operator_id = operator.operator_id
        self.evidence = evidence_
        self.scope = scope_
        self.complete = bool(complete)
        self.tolerance = tolerance_
        self.numeric_fingerprint = numeric_fingerprint
        self.structure_id = structure_id
        self.certificate_id = certificate_id

    def matches(self, operator: AbstractLinearOperator, /) -> bool:
        """Return whether this certificate applies to the supplied operator value."""
        if not isinstance(operator, AbstractLinearOperator):
            return False
        if operator.operator_id != self.operator_id:
            return False
        return self.scope == "structural" or (
            _operator_numeric_fingerprint(operator) == self.numeric_fingerprint
        )


class SpectralInterval(StrictModule):
    """Explicit real spectral enclosure with provenance and validity scope."""

    lower: Array
    upper: Array
    operator_id: str = eqx.field(static=True)
    evidence: CertificateEvidence = eqx.field(static=True)
    scope: CertificateScope = eqx.field(static=True)
    numeric_fingerprint: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        evidence: CertificateEvidence = "asserted",
        scope: CertificateScope = "numerical",
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        coordinate_dtype = _coordinate_dtype(operator.source)
        real_dtype = jnp.empty((), dtype=coordinate_dtype).real.dtype
        lower_ = jnp.asarray(lower, dtype=real_dtype)
        upper_ = jnp.asarray(upper, dtype=real_dtype)
        if lower_.shape != () or upper_.shape != ():
            raise ValueError("Spectral interval endpoints must be scalar.")
        endpoints = jnp.stack((lower_, upper_))
        endpoints = eqx.error_if(
            endpoints,
            jnp.any(~jnp.isfinite(endpoints)) | (lower_ > upper_),
            "Spectral interval endpoints must be finite and ordered.",
        )
        evidence_ = _validate_evidence(evidence)
        scope_ = _validate_scope(scope)
        numeric_fingerprint = _operator_numeric_fingerprint(operator)
        certificate_id = canonical_fingerprint(
            {
                "kind": "spectral-interval",
                "operator": operator.operator_id,
                "numeric": numeric_fingerprint,
                "evidence": evidence_,
                "scope": scope_,
            }
        )
        self.lower = endpoints[0]
        self.upper = endpoints[1]
        self.operator_id = operator.operator_id
        self.evidence = evidence_
        self.scope = scope_
        self.numeric_fingerprint = numeric_fingerprint
        self.certificate_id = certificate_id

    def matches(self, operator: AbstractLinearOperator, /) -> bool:
        if not isinstance(operator, AbstractLinearOperator):
            return False
        if operator.operator_id != self.operator_id:
            return False
        return self.scope == "structural" or (
            _operator_numeric_fingerprint(operator) == self.numeric_fingerprint
        )


__all__ = [
    "CertificateEvidence",
    "CertificateScope",
    "KernelCertificate",
    "SpectralInterval",
]
