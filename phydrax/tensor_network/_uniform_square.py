#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .. import ein
from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import hermitian_sqrt, HermitianPrecisionPolicy
from ._precision import TensorNetworkPrecisionPolicy


class UniformSquareTensor(StrictModule):
    """One rank-four tensor repeated on an infinite square lattice.

    Axes are ordered ``(up, right, down, left)``. Opposite bond dimensions
    match, while vertical and horizontal bond dimensions may differ.
    """

    value: Array
    vertical_bond_dimension: int = eqx.field(static=True)
    horizontal_bond_dimension: int = eqx.field(static=True)
    precision: TensorNetworkPrecisionPolicy
    tensor_id: str = eqx.field(static=True)
    numeric_version: Array

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        precision: TensorNetworkPrecisionPolicy | None = None,
        numeric_version: ArrayLike = 0,
    ):
        array = jnp.asarray(value)
        if array.ndim != 4:
            raise ValueError(
                "Uniform square tensors require axes (up, right, down, left)."
            )
        if any(dimension < 1 for dimension in array.shape):
            raise ValueError("Uniform square tensor dimensions must be positive.")
        if array.shape[0] != array.shape[2]:
            raise ValueError("Uniform square up/down bond dimensions must match.")
        if array.shape[1] != array.shape[3]:
            raise ValueError("Uniform square right/left bond dimensions must match.")
        precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, TensorNetworkPrecisionPolicy):
            raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
        precision_.validate_storage(array)
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.value = array
        self.vertical_bond_dimension = int(array.shape[0])
        self.horizontal_bond_dimension = int(array.shape[1])
        self.precision = precision_
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "uniform-square-tensor",
                "shape": tuple(int(dimension) for dimension in array.shape),
                "dtype": str(array.dtype),
                "precision": precision_.policy_id,
            }
        )
        self.numeric_version = version


class UniformSquareTensorBuildEvidence(StrictModule):
    vertical_hermiticity_residual: Array
    horizontal_hermiticity_residual: Array
    vertical_minimum_eigenvalue: Array
    horizontal_minimum_eigenvalue: Array
    vertical_reconstruction_residual: Array
    horizontal_reconstruction_residual: Array
    finite: Array
    positive_semidefinite: Array
    accepted: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    builder_id: str = eqx.field(static=True)


class UniformSquareTensorBuildResult(StrictModule):
    tensor: UniformSquareTensor
    evidence: UniformSquareTensorBuildEvidence


def _hermitian_precision(
    precision: TensorNetworkPrecisionPolicy,
    /,
) -> HermitianPrecisionPolicy:
    return HermitianPrecisionPolicy(
        compute_dtype=precision.contraction_dtype,
        factorization_dtype=precision.factorization_dtype,
        accumulation_dtype=precision.accumulation_dtype,
        decision_dtype=precision.decision_dtype,
        output_dtype=precision.storage_dtype,
    )


def _matrix_residual(value: Array, reference: Array, precision, /) -> Array:
    numerator = precision.norm(value - reference)
    denominator = jnp.maximum(precision.norm(reference), precision.decision(1.0))
    return precision.decision(numerator / denominator)


def build_uniform_pair_partition_tensor(
    vertical_weight: ArrayLike,
    horizontal_weight: ArrayLike | None = None,
    /,
    *,
    site_weight: ArrayLike | None = None,
    positivity_tolerance: float = 1e-10,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> UniformSquareTensorBuildResult:
    """Factor real pair weights into a uniform square-lattice partition tensor."""

    tolerance = float(positivity_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("positivity_tolerance must be finite and nonnegative.")
    vertical = jnp.asarray(vertical_weight)
    horizontal = vertical if horizontal_weight is None else jnp.asarray(horizontal_weight)
    if (
        vertical.ndim != 2
        or horizontal.ndim != 2
        or vertical.shape[0] != vertical.shape[1]
        or horizontal.shape[0] != horizontal.shape[1]
        or vertical.shape != horizontal.shape
        or vertical.shape[0] < 1
    ):
        raise ValueError(
            "Pair weights must be nonempty square matrices with one shared state size."
        )
    if jnp.issubdtype(vertical.dtype, jnp.complexfloating) or jnp.issubdtype(
        horizontal.dtype, jnp.complexfloating
    ):
        raise TypeError("Pair partition tensor construction requires real weights.")
    states = int(vertical.shape[0])
    pair_dtype = jnp.result_type(vertical, horizontal)
    if not jnp.issubdtype(pair_dtype, jnp.inexact):
        pair_dtype = jnp.dtype(float)
    onsite = (
        jnp.ones((states,), dtype=pair_dtype)
        if site_weight is None
        else jnp.asarray(site_weight)
    )
    if onsite.shape != (states,):
        raise ValueError("site_weight must have one entry per local state.")
    if jnp.issubdtype(onsite.dtype, jnp.complexfloating):
        raise TypeError("Pair partition tensor construction requires real site weights.")

    dtype = jnp.result_type(vertical, horizontal, onsite)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.dtype(float)
    vertical = vertical.astype(dtype)
    horizontal = horizontal.astype(dtype)
    onsite = onsite.astype(dtype)
    precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TensorNetworkPrecisionPolicy):
        raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
    vertical = precision_.storage(vertical)
    horizontal = precision_.storage(horizontal)
    onsite = precision_.storage(onsite)
    reconstruction_tolerance = max(
        tolerance,
        float(128 * states * jnp.finfo(vertical.dtype).eps),
    )
    hermitian_precision = _hermitian_precision(precision_)
    vertical_root = hermitian_sqrt(
        vertical,
        tolerance=tolerance,
        precision=hermitian_precision,
    )
    horizontal_root = hermitian_sqrt(
        horizontal,
        tolerance=tolerance,
        precision=hermitian_precision,
    )
    vertical_reconstructed = ein.contract(
        "su,tu->st",
        vertical_root.value,
        vertical_root.value,
        backend="jax",
    )
    horizontal_reconstructed = ein.contract(
        "sr,tr->st",
        horizontal_root.value,
        horizontal_root.value,
        backend="jax",
    )
    vertical_residual = _matrix_residual(
        vertical_reconstructed,
        vertical,
        precision_,
    )
    horizontal_residual = _matrix_residual(
        horizontal_reconstructed,
        horizontal,
        precision_,
    )
    finite = (
        jnp.all(jnp.isfinite(vertical))
        & jnp.all(jnp.isfinite(horizontal))
        & jnp.all(jnp.isfinite(onsite))
        & jnp.all(jnp.isfinite(vertical_root.value))
        & jnp.all(jnp.isfinite(horizontal_root.value))
    )
    positive_semidefinite = (vertical_root.spectrum.minimum_eigenvalue >= -tolerance) & (
        horizontal_root.spectrum.minimum_eigenvalue >= -tolerance
    )
    nonnegative_onsite = jnp.all(onsite >= 0.0)
    accepted = (
        finite
        & vertical_root.valid
        & horizontal_root.valid
        & positive_semidefinite
        & nonnegative_onsite
        & (vertical_residual <= reconstruction_tolerance)
        & (horizontal_residual <= reconstruction_tolerance)
    )
    factors = (
        vertical_root.value,
        horizontal_root.value,
        onsite,
    )
    tensor_value = ein.contract(
        "su,sr,sd,sl,s->urdl",
        factors[0],
        factors[1],
        factors[0],
        factors[1],
        factors[2],
        backend="jax",
    )
    tensor_value = precision_.storage(tensor_value)
    tensor_value = eqx.error_if(
        tensor_value,
        ~accepted,
        "Pair partition tensor inputs are not finite positive-semidefinite weights.",
    )
    tensor = UniformSquareTensor(tensor_value, precision=precision_)
    builder_id = canonical_fingerprint(
        {
            "kind": "uniform-pair-partition-tensor",
            "states": states,
            "dtype": str(tensor_value.dtype),
            "precision": precision_.policy_id,
            "positivity_tolerance": tolerance,
        }
    )
    evidence = UniformSquareTensorBuildEvidence(
        vertical_root.spectrum.hermiticity_residual,
        horizontal_root.spectrum.hermiticity_residual,
        vertical_root.spectrum.minimum_eigenvalue,
        horizontal_root.spectrum.minimum_eigenvalue,
        vertical_residual,
        horizontal_residual,
        finite,
        positive_semidefinite & nonnegative_onsite,
        accepted,
        precision_.evidence_for(
            (vertical, horizontal, onsite),
            children={
                "vertical_sqrt": vertical_root.spectrum.precision_evidence,
                "horizontal_sqrt": horizontal_root.spectrum.precision_evidence,
            },
            output_value=tensor_value,
        ),
        builder_id,
    )
    return UniformSquareTensorBuildResult(tensor, evidence)


__all__ = [
    "UniformSquareTensor",
    "UniformSquareTensorBuildEvidence",
    "UniformSquareTensorBuildResult",
    "build_uniform_pair_partition_tensor",
]
