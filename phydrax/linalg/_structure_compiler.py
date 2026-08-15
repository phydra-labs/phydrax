#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._materialization import MaterializationPolicy, materialize
from ._operators import (
    AbstractLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
)
from ._spaces import ArraySpace
from ._structured_operators import (
    BandedLinearOperator,
    PermutationLinearOperator,
    TriangularLinearOperator,
    TridiagonalLinearOperator,
)
from ._transform_operators import (
    _forward_transform,
    OrthogonalTransformKind,
    TransformDiagonalLinearOperator,
)


CompiledStructure: TypeAlias = Literal[
    "diagonal",
    "permutation",
    "tridiagonal",
    "triangular",
    "banded",
    "dct-diagonal",
    "fft-diagonal",
    "dense",
]
StructureCandidate: TypeAlias = Literal[
    "diagonal",
    "permutation",
    "tridiagonal",
    "triangular",
    "banded",
    "dct-diagonal",
    "fft-diagonal",
]
CompilerFallback: TypeAlias = Literal["dense", "error"]


class StructureCompilationPolicy(StrictModule):
    """Candidate order, approximation consent, and bounded setup policy."""

    candidates: tuple[StructureCandidate, ...] = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    allow_approximation: bool = eqx.field(static=True)
    max_bandwidth: int = eqx.field(static=True)
    fallback: CompilerFallback = eqx.field(static=True)
    materialization: MaterializationPolicy

    def __init__(
        self,
        *,
        candidates: Sequence[StructureCandidate] = (
            "diagonal",
            "permutation",
            "tridiagonal",
            "triangular",
            "banded",
            "dct-diagonal",
            "fft-diagonal",
        ),
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        allow_approximation: bool = False,
        max_bandwidth: int = 4,
        fallback: CompilerFallback = "dense",
        materialization: MaterializationPolicy | None = None,
    ):
        candidates_ = tuple(candidates)
        valid = {
            "diagonal",
            "permutation",
            "tridiagonal",
            "triangular",
            "banded",
            "dct-diagonal",
            "fft-diagonal",
        }
        if not candidates_ or len(set(candidates_)) != len(candidates_):
            raise ValueError(
                "candidates must be a non-empty sequence without duplicates."
            )
        if any(candidate not in valid for candidate in candidates_):
            raise ValueError("Unknown structure compiler candidate.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or absolute < 0
            or relative < 0
        ):
            raise ValueError("Compiler tolerances must be finite and non-negative.")
        if not allow_approximation and (absolute != 0 or relative != 0):
            raise ValueError(
                "Nonzero compiler tolerances require allow_approximation=True."
            )
        bandwidth = int(max_bandwidth)
        if bandwidth < 0:
            raise ValueError("max_bandwidth must be non-negative.")
        if fallback not in ("dense", "error"):
            raise ValueError("fallback must be 'dense' or 'error'.")
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        self.candidates = candidates_
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.allow_approximation = bool(allow_approximation)
        self.max_bandwidth = bandwidth
        self.fallback = fallback
        self.materialization = materialization_


class StructureCompilationResult(StrictModule):
    """Compiled operator plus explicit projection error and refresh identity."""

    operator: AbstractLinearOperator
    policy: StructureCompilationPolicy
    discarded_norm: Array
    relative_discarded_norm: Array
    numeric_version: Array
    structure: CompiledStructure = eqx.field(static=True)
    variant: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    original_operator_id: str = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator: AbstractLinearOperator,
        policy: StructureCompilationPolicy,
        discarded_norm: Any,
        relative_discarded_norm: Any,
        numeric_version: Any,
        structure: CompiledStructure,
        variant: str,
        exact: bool,
        original_operator_id: str,
        compiler_id: str,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        self.operator = operator
        self.policy = policy
        self.discarded_norm = jnp.asarray(discarded_norm)
        self.relative_discarded_norm = jnp.asarray(relative_discarded_norm)
        self.numeric_version = version
        self.structure = structure
        self.variant = str(variant)
        self.exact = bool(exact)
        self.original_operator_id = str(original_operator_id)
        self.compiler_id = str(compiler_id)


def compile_linear_structure(
    operator_or_matrix: AbstractLinearOperator | ArrayLike,
    policy: StructureCompilationPolicy | None = None,
    /,
) -> StructureCompilationResult:
    """Compile verified matrix structure; approximation requires explicit consent."""
    policy_ = StructureCompilationPolicy() if policy is None else policy
    if not isinstance(policy_, StructureCompilationPolicy):
        raise TypeError("policy must be a StructureCompilationPolicy or None.")
    original, matrix = _coerce_operator(operator_or_matrix, policy_.materialization)
    compiler_id = _compiler_id(original, policy_)
    return _compile_matrix(
        original,
        matrix,
        policy_,
        compiler_id=compiler_id,
        numeric_version=0,
    )


def refresh_linear_structure(
    compiled: StructureCompilationResult,
    operator_or_matrix: AbstractLinearOperator | ArrayLike,
    /,
    *,
    recompile: bool = False,
) -> StructureCompilationResult:
    """Refresh coefficients while preserving structure, or explicitly recompile."""
    if not isinstance(compiled, StructureCompilationResult):
        raise TypeError("compiled must be a StructureCompilationResult.")
    original, matrix = _coerce_operator(
        operator_or_matrix,
        compiled.policy.materialization,
    )
    if original.operator_id != compiled.original_operator_id:
        raise ValueError("Structure refresh changed the original operator identity.")
    version = compiled.numeric_version + jnp.asarray(1, dtype=jnp.int32)
    candidate = _compile_candidate(
        original,
        matrix,
        compiled.structure,
        compiled.policy,
        compiled.operator.operator_id,
    )
    if candidate is not None and candidate[1] == compiled.variant:
        operator, variant, projected = candidate
        return _result(
            original,
            matrix,
            projected,
            operator,
            compiled.policy,
            compiled.structure,
            variant,
            compiled.compiler_id,
            version,
        )
    if not recompile:
        raise ValueError(
            "Refreshed coefficients no longer satisfy the compiled structure."
        )
    return _compile_matrix(
        original,
        matrix,
        compiled.policy,
        compiler_id=compiled.compiler_id,
        numeric_version=version,
    )


def _compile_matrix(
    original: AbstractLinearOperator,
    matrix: np.ndarray,
    policy: StructureCompilationPolicy,
    *,
    compiler_id: str,
    numeric_version: Any,
) -> StructureCompilationResult:
    for structure in policy.candidates:
        candidate = _compile_candidate(
            original,
            matrix,
            structure,
            policy,
            f"{compiler_id}:{structure}",
        )
        if candidate is not None:
            operator, variant, projected = candidate
            return _result(
                original,
                matrix,
                projected,
                operator,
                policy,
                structure,
                variant,
                compiler_id,
                numeric_version,
            )
    if policy.fallback == "error":
        raise ValueError("No declared structure candidate matches the operator.")
    dense = DenseLinearOperator(
        jnp.asarray(matrix),
        source=original.source,
        target=original.target,
        operator_id=f"{compiler_id}:dense",
    )
    return _result(
        original,
        matrix,
        matrix,
        dense,
        policy,
        "dense",
        "dense-fallback",
        compiler_id,
        numeric_version,
    )


def _compile_candidate(
    original: AbstractLinearOperator,
    matrix: np.ndarray,
    structure: CompiledStructure,
    policy: StructureCompilationPolicy,
    operator_id: str,
) -> tuple[AbstractLinearOperator, str, np.ndarray] | None:
    n = matrix.shape[0]
    space = original.source
    if structure == "dense":
        projected = matrix
        return (
            DenseLinearOperator(
                jnp.asarray(projected),
                source=space,
                target=original.target,
                operator_id=operator_id,
            ),
            "dense-fallback",
            projected,
        )
    if structure == "diagonal":
        projected = np.diag(np.diag(matrix))
        if not _accepted(matrix, projected, policy):
            return None
        return (
            DiagonalLinearOperator(
                jnp.asarray(np.diag(projected)),
                space=space,
                operator_id=operator_id,
            ),
            "main-diagonal",
            projected,
        )
    if structure == "permutation":
        permutation = np.argmax(np.abs(matrix), axis=1)
        if len(set(int(value) for value in permutation)) != n:
            return None
        projected = np.eye(n, dtype=matrix.dtype)[permutation]
        if not _accepted(matrix, projected, policy):
            return None
        return (
            PermutationLinearOperator(
                jnp.asarray(permutation, dtype=jnp.int32),
                space=space,
                operator_id=operator_id,
            ),
            "unit-permutation",
            projected,
        )
    if structure == "tridiagonal":
        projected = np.diag(np.diag(matrix))
        projected += np.diag(np.diag(matrix, -1), -1)
        projected += np.diag(np.diag(matrix, 1), 1)
        if not _accepted(matrix, projected, policy):
            return None
        return (
            TridiagonalLinearOperator(
                jnp.asarray(np.diag(projected, -1)),
                jnp.asarray(np.diag(projected)),
                jnp.asarray(np.diag(projected, 1)),
                space=space,
                operator_id=operator_id,
            ),
            "three-diagonal",
            projected,
        )
    if structure == "triangular":
        lower = np.tril(matrix)
        upper = np.triu(matrix)
        lower_error = np.linalg.norm(matrix - lower)
        upper_error = np.linalg.norm(matrix - upper)
        projected = lower if lower_error <= upper_error else upper
        is_lower = lower_error <= upper_error
        if not _accepted(matrix, projected, policy):
            return None
        variant = "lower" if is_lower else "upper"
        return (
            TriangularLinearOperator(
                jnp.asarray(projected),
                lower=is_lower,
                space=space,
                operator_id=operator_id,
            ),
            variant,
            projected,
        )
    if structure == "banded":
        lower, upper = _detected_bandwidth(matrix, policy)
        if lower + upper > policy.max_bandwidth:
            return None
        projected = _band_projection(matrix, lower, upper)
        if not _accepted(matrix, projected, policy):
            return None
        return (
            BandedLinearOperator(
                jnp.asarray(_band_storage(projected, lower, upper)),
                lower_bandwidth=lower,
                upper_bandwidth=upper,
                space=space,
                operator_id=operator_id,
            ),
            f"lower={lower},upper={upper}",
            projected,
        )
    if structure in ("dct-diagonal", "fft-diagonal"):
        transform: OrthogonalTransformKind = (
            "dct" if structure == "dct-diagonal" else "fft"
        )
        if not isinstance(space, ArraySpace):
            return None
        if transform == "dct" and np.issubdtype(matrix.dtype, np.complexfloating):
            return None
        if transform == "fft" and not np.issubdtype(matrix.dtype, np.complexfloating):
            return None
        transform_matrix = _orthogonal_transform_matrix(
            space.shape,
            matrix.dtype,
            transform,
        )
        spectral_matrix = transform_matrix @ matrix @ np.conj(transform_matrix.T)
        spectrum = np.diag(spectral_matrix)
        threshold = _threshold(matrix, policy)
        if transform == "dct":
            spectrum = np.real(spectrum)
        property_ = "general"
        if np.max(np.abs(np.imag(spectrum))) <= threshold:
            real_spectrum = np.real(spectrum)
            spectrum = real_spectrum.astype(matrix.dtype)
            if np.min(real_spectrum) > threshold:
                property_ = "positive-definite"
            elif np.min(real_spectrum) >= -threshold:
                spectrum = np.maximum(real_spectrum, 0).astype(matrix.dtype)
                property_ = "positive-semidefinite"
            else:
                property_ = "self-adjoint"
        projected = np.conj(transform_matrix.T) @ np.diag(spectrum) @ transform_matrix
        projected = np.asarray(projected, dtype=matrix.dtype)
        if not _accepted(matrix, projected, policy):
            return None
        nonsingular = bool(np.all(np.abs(spectrum) > threshold))
        return (
            TransformDiagonalLinearOperator(
                jnp.asarray(spectrum.reshape(space.shape)),
                space=space,
                transform=transform,
                spectral_property=property_,
                nonsingular=nonsingular,
                operator_id=operator_id,
            ),
            f"{transform}-orthonormal",
            np.asarray(projected, dtype=matrix.dtype),
        )
    raise ValueError(f"Unknown compiled structure {structure!r}.")


def _coerce_operator(
    operator_or_matrix: AbstractLinearOperator | ArrayLike,
    materialization: MaterializationPolicy,
    /,
) -> tuple[AbstractLinearOperator, np.ndarray]:
    if isinstance(operator_or_matrix, AbstractLinearOperator):
        operator = operator_or_matrix
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Structure compilation requires an unbatched endomorphism.")
        matrix = np.asarray(materialize(operator, materialization))
    else:
        value = jnp.asarray(operator_or_matrix)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Structure compilation requires one square matrix.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        operator = DenseLinearOperator(value)
        matrix = np.asarray(value)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Structure compilation requires finite coefficients.")
    return operator, matrix


def _result(
    original: AbstractLinearOperator,
    matrix: np.ndarray,
    projected: np.ndarray,
    operator: AbstractLinearOperator,
    policy: StructureCompilationPolicy,
    structure: CompiledStructure,
    variant: str,
    compiler_id: str,
    numeric_version: Any,
) -> StructureCompilationResult:
    discarded = float(np.linalg.norm(matrix - projected))
    baseline = float(np.linalg.norm(matrix))
    relative = discarded / max(baseline, np.finfo(matrix.real.dtype).tiny)
    exact = bool(np.array_equal(matrix, projected))
    return StructureCompilationResult(
        operator=operator,
        policy=policy,
        discarded_norm=jnp.asarray(discarded, dtype=matrix.real.dtype),
        relative_discarded_norm=jnp.asarray(relative, dtype=matrix.real.dtype),
        numeric_version=numeric_version,
        structure=structure,
        variant=variant,
        exact=exact,
        original_operator_id=original.operator_id,
        compiler_id=compiler_id,
    )


def _accepted(
    matrix: np.ndarray,
    projected: np.ndarray,
    policy: StructureCompilationPolicy,
    /,
) -> bool:
    error = float(np.max(np.abs(matrix - projected)))
    return error <= _threshold(matrix, policy)


def _threshold(matrix: np.ndarray, policy: StructureCompilationPolicy, /) -> float:
    if not policy.allow_approximation:
        return 0.0
    scale = float(np.max(np.abs(matrix)))
    return policy.absolute_tolerance + policy.relative_tolerance * scale


def _detected_bandwidth(
    matrix: np.ndarray,
    policy: StructureCompilationPolicy,
    /,
) -> tuple[int, int]:
    significant = np.abs(matrix) > _threshold(matrix, policy)
    rows, columns = np.nonzero(significant)
    if rows.size == 0:
        return 0, 0
    lower = int(np.max(np.maximum(rows - columns, 0)))
    upper = int(np.max(np.maximum(columns - rows, 0)))
    return lower, upper


def _band_projection(matrix: np.ndarray, lower: int, upper: int, /) -> np.ndarray:
    n = matrix.shape[0]
    rows = np.arange(n)[:, None]
    columns = np.arange(n)[None, :]
    mask = (rows - columns <= lower) & (columns - rows <= upper)
    return np.where(mask, matrix, 0)


def _band_storage(
    matrix: np.ndarray,
    lower: int,
    upper: int,
    /,
) -> np.ndarray:
    n = matrix.shape[0]
    bands = np.zeros((lower + upper + 1, n), dtype=matrix.dtype)
    for offset in range(-upper, lower + 1):
        column_start = max(0, -offset)
        column_stop = min(n, n - offset)
        rows = np.arange(column_start, column_stop) + offset
        columns = np.arange(column_start, column_stop)
        bands[upper + offset, columns] = matrix[rows, columns]
    return bands


def _orthogonal_transform_matrix(
    shape: tuple[int, ...],
    dtype: np.dtype,
    transform: OrthogonalTransformKind,
    /,
) -> np.ndarray:
    size = int(np.prod(shape))
    axes = tuple(range(len(shape)))
    basis = jnp.eye(size, dtype=dtype)

    def column(coordinates):
        value = coordinates.reshape(shape)
        return _forward_transform(value, transform, axes).reshape((-1,))

    return np.asarray(jax.vmap(column)(basis).T)


def _compiler_id(
    operator: AbstractLinearOperator,
    policy: StructureCompilationPolicy,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "linear-structure-compiler",
            "operator": operator.operator_id,
            "source": operator.source.space_id,
            "target": operator.target.space_id,
            "candidates": list(policy.candidates),
            "absolute_tolerance": policy.absolute_tolerance,
            "relative_tolerance": policy.relative_tolerance,
            "allow_approximation": policy.allow_approximation,
            "max_bandwidth": policy.max_bandwidth,
            "fallback": policy.fallback,
        }
    )


__all__ = [
    "CompiledStructure",
    "CompilerFallback",
    "StructureCandidate",
    "StructureCompilationPolicy",
    "StructureCompilationResult",
    "compile_linear_structure",
    "refresh_linear_structure",
]
