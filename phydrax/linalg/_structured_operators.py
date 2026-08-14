#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ._operators import (
    _generic_adjoint,
    _id,
    _validate_action_dtype,
    AbstractLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
)
from ._properties import OperatorCapabilities, OperatorProperties, PropertyEvidence
from ._space_extensions import TensorProductSpace
from ._spaces import (
    _has_euclidean_pairing,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
)


def _same(left: AbstractVectorSpace, right: AbstractVectorSpace, /) -> None:
    if not left.compatible(right):
        raise ValueError("Operator spaces are incompatible.")


def _space(size: int, dtype: Any, supplied: AbstractVectorSpace | None, /):
    result = ArraySpace((size,), dtype=dtype) if supplied is None else supplied
    if not isinstance(result, AbstractVectorSpace) or result.size != size:
        raise ValueError("space must have the operator's coordinate dimension.")
    return result


def _apply_factor_axis(
    value: Array,
    factor: AbstractLinearOperator,
    axis: int,
    mode: Literal["forward", "transpose", "adjoint"],
    /,
) -> Array:
    if mode == "forward":
        input_space, output_space, action = factor.source, factor.target, factor.mv
    elif mode == "transpose":
        input_space, output_space, action = (
            factor.target,
            factor.source,
            factor.transpose_mv,
        )
    else:
        input_space, output_space, action = (
            factor.target,
            factor.source,
            factor.adjoint_mv,
        )
    moved = jnp.moveaxis(value, axis, 0)
    trailing_shape = moved.shape[1:]
    columns = moved.reshape((input_space.size, -1)).T

    def apply_column(coordinates):
        vector = input_space.unflatten(coordinates)
        return output_space.flatten(action(vector))

    applied = jax.vmap(apply_column)(columns).T
    return jnp.moveaxis(
        applied.reshape((output_space.size,) + trailing_shape),
        0,
        axis,
    )


def _apply_matrix(
    matrix: Array,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    vector: PyTree[Any],
    /,
) -> PyTree[Array]:
    coordinates = source.flatten(vector)
    return target.unflatten(matrix @ coordinates)


def _construction_evidence(
    **claims: bool | int | None,
) -> dict[str, PropertyEvidence]:
    return {
        name: "construction"
        for name, value in claims.items()
        if value is not False and value is not None
    }


def _derived_rank_evidence(
    operators: Sequence[AbstractLinearOperator],
    rank: int | None,
    /,
) -> dict[str, PropertyEvidence]:
    if rank is not None and all(
        operator.properties.certifies("rank") for operator in operators
    ):
        return {"rank": "transformed"}
    return {}


def _validated_permutation(permutation: Array, size: int, /) -> Array:
    invalid = jnp.any(jnp.sort(permutation) != jnp.arange(size))
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(
            permutation,
            invalid,
            "permutation must contain each coordinate exactly once.",
        )
    if bool(invalid):
        raise ValueError("permutation must contain each coordinate exactly once.")
    return permutation


class PermutationLinearOperator(AbstractLinearOperator):
    """Exact coordinate permutation without materializing a permutation matrix."""

    permutation: Array
    inverse_permutation: Array

    def __init__(
        self,
        permutation: ArrayLike,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        dtype: Any = np.float64,
        operator_id: str | None = None,
    ):
        permutation_ = jnp.asarray(permutation)
        if permutation_.ndim != 1 or not jnp.issubdtype(permutation_.dtype, jnp.integer):
            raise TypeError("permutation must be one integer vector.")
        size = int(permutation_.size)
        permutation_ = _validated_permutation(permutation_, size)
        space_ = _space(size, dtype, space)
        self.source = space_
        self.target = space_
        self.permutation = permutation_.astype(jnp.int32)
        self.inverse_permutation = jnp.argsort(self.permutation)
        self.properties = OperatorProperties(
            rank=size,
            evidence={"rank": "construction"},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id, {"kind": "permutation", "space": space_.space_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.source.unflatten(self.source.flatten(vector)[self.permutation])

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.source.unflatten(
            self.source.flatten(vector)[self.inverse_permutation]
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        dtype = self.source.structure()
        coordinate_dtype = jax.tree.leaves(dtype)[0].dtype
        return jnp.eye(self.source.size, dtype=coordinate_dtype)[self.permutation]


class TriangularLinearOperator(AbstractLinearOperator):
    """Explicit lower or upper triangular endomorphism."""

    matrix: Array
    lower: bool
    unit_diagonal: bool

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        lower: bool,
        unit_diagonal: bool = False,
        space: AbstractVectorSpace | None = None,
        operator_id: str | None = None,
    ):
        matrix_ = jnp.asarray(matrix)
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1]:
            raise ValueError("matrix must be one square matrix.")
        if not jnp.issubdtype(matrix_.dtype, jnp.inexact):
            matrix_ = matrix_.astype(float)
        lower_ = bool(lower)
        expected = jnp.tril(matrix_) if lower_ else jnp.triu(matrix_)
        expected = eqx.error_if(
            expected,
            jnp.any(matrix_ != expected),
            "matrix has entries outside its declared triangle.",
        )
        if unit_diagonal:
            expected = expected.at[jnp.diag_indices(matrix_.shape[0])].set(1)
        size = int(matrix_.shape[0])
        space_ = _space(size, matrix_.dtype, space)
        _validate_action_dtype(matrix_.dtype, space_, space_, "triangular matrix")
        rank = size if unit_diagonal else None
        self.source = space_
        self.target = space_
        self.matrix = expected
        self.lower = lower_
        self.unit_diagonal = bool(unit_diagonal)
        self.properties = OperatorProperties(
            triangular=True,
            rank=rank,
            evidence=_construction_evidence(triangular=True, rank=rank),
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "triangular",
                "space": space_.space_id,
                "lower": lower_,
                "unit_diagonal": bool(unit_diagonal),
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _apply_matrix(self.matrix, self.source, self.target, vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _apply_matrix(self.matrix.T, self.target, self.source, vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return self.matrix


class TridiagonalLinearOperator(AbstractLinearOperator):
    """Exact tridiagonal endomorphism stored in three vectors."""

    lower: Array
    diagonal: Array
    upper: Array

    def __init__(
        self,
        lower: ArrayLike,
        diagonal: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        operator_id: str | None = None,
    ):
        lower_, diagonal_, upper_ = map(jnp.asarray, (lower, diagonal, upper))
        if (
            diagonal_.ndim != 1
            or lower_.shape != (diagonal_.size - 1,)
            or upper_.shape != (diagonal_.size - 1,)
        ):
            raise ValueError("Tridiagonal storage must have lengths n-1, n, n-1.")
        dtype = jnp.result_type(lower_, diagonal_, upper_, float)
        lower_, diagonal_, upper_ = (
            value.astype(dtype) for value in (lower_, diagonal_, upper_)
        )
        size = int(diagonal_.size)
        space_ = _space(size, dtype, space)
        self.source = space_
        self.target = space_
        self.lower = lower_
        self.diagonal = diagonal_
        self.upper = upper_
        self.properties = OperatorProperties(
            triangular=False,
            evidence={},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id, {"kind": "tridiagonal", "space": space_.space_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.source.flatten(vector)
        y = self.diagonal * x
        y = y.at[1:].add(self.lower * x[:-1])
        y = y.at[:-1].add(self.upper * x[1:])
        return self.target.unflatten(y)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.target.flatten(vector)
        y = self.diagonal * x
        y = y.at[1:].add(self.upper * x[:-1])
        y = y.at[:-1].add(self.lower * x[1:])
        return self.source.unflatten(y)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        matrix = jnp.diag(self.diagonal)
        matrix = matrix + jnp.diag(self.lower, -1)
        return matrix + jnp.diag(self.upper, 1)


class BandedLinearOperator(AbstractLinearOperator):
    """General fixed-bandwidth matrix in SciPy diagonal-ordered storage."""

    bands: Array
    lower_bandwidth: int
    upper_bandwidth: int

    def __init__(
        self,
        bands: ArrayLike,
        /,
        *,
        lower_bandwidth: int,
        upper_bandwidth: int,
        space: AbstractVectorSpace | None = None,
        operator_id: str | None = None,
    ):
        bands_ = jnp.asarray(bands)
        lower_ = int(lower_bandwidth)
        upper_ = int(upper_bandwidth)
        if lower_ < 0 or upper_ < 0:
            raise ValueError("bandwidths must be non-negative.")
        if bands_.ndim != 2 or bands_.shape[0] != lower_ + upper_ + 1:
            raise ValueError("bands must have shape (lower + upper + 1, n).")
        if not jnp.issubdtype(bands_.dtype, jnp.inexact):
            bands_ = bands_.astype(float)
        size = int(bands_.shape[1])
        space_ = _space(size, bands_.dtype, space)
        self.source = space_
        self.target = space_
        self.bands = bands_
        self.lower_bandwidth = lower_
        self.upper_bandwidth = upper_
        diagonal = lower_ == 0 and upper_ == 0
        triangular = lower_ == 0 or upper_ == 0
        self.properties = OperatorProperties(
            diagonal=diagonal,
            triangular=triangular,
            evidence=_construction_evidence(diagonal=diagonal, triangular=triangular),
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "banded",
                "space": space_.space_id,
                "lower": lower_,
                "upper": upper_,
            },
        )

    def _apply_bands(
        self,
        coordinates: Array,
        /,
        *,
        transpose: bool,
        conjugate: bool,
    ) -> Array:
        size = self.source.size
        result = jnp.zeros((size,), dtype=coordinates.dtype)
        for offset in range(-self.upper_bandwidth, self.lower_bandwidth + 1):
            column_start = max(0, -offset)
            column_stop = min(size, size - offset)
            row_start = column_start + offset
            row_stop = column_stop + offset
            values = self.bands[
                self.upper_bandwidth + offset,
                column_start:column_stop,
            ]
            if conjugate:
                values = jnp.conj(values)
            if transpose:
                result = result.at[column_start:column_stop].add(
                    values * coordinates[row_start:row_stop]
                )
            else:
                result = result.at[row_start:row_stop].add(
                    values * coordinates[column_start:column_stop]
                )
        return result

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.source.flatten(vector)
        return self.target.unflatten(
            self._apply_bands(coordinates, transpose=False, conjugate=False)
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.target.flatten(vector)
        return self.source.unflatten(
            self._apply_bands(coordinates, transpose=True, conjugate=False)
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        n = self.source.size
        rows = jnp.arange(n)[:, None]
        columns = jnp.arange(n)[None, :]
        band_rows = self.upper_bandwidth + rows - columns
        valid = (band_rows >= 0) & (band_rows < self.bands.shape[0])
        clipped = jnp.clip(band_rows, 0, self.bands.shape[0] - 1)
        values = self.bands[clipped, columns]
        return jnp.where(valid, values, 0)


class BlockDiagonalLinearOperator(AbstractLinearOperator):
    """Exact block diagonal operator preserving block-vector structure."""

    blocks: tuple[AbstractLinearOperator, ...]
    source: BlockSpace
    target: BlockSpace

    def __init__(
        self,
        blocks: Sequence[AbstractLinearOperator],
        /,
        *,
        operator_id: str | None = None,
    ):
        blocks_ = tuple(blocks)
        if not blocks_ or not all(
            isinstance(block, AbstractLinearOperator) for block in blocks_
        ):
            raise TypeError("blocks must contain AbstractLinearOperator values.")
        if any(block.batch_shape for block in blocks_):
            raise ValueError(
                "BlockDiagonalLinearOperator does not accept batched blocks."
            )
        source = BlockSpace(tuple(block.source for block in blocks_))
        target = BlockSpace(tuple(block.target for block in blocks_))
        rank: int | None = 0
        for block in blocks_:
            block_rank = block.properties.rank
            if block_rank is None:
                rank = None
                break
            rank += block_rank
        self.blocks = blocks_
        self.source = source
        self.target = target
        self.properties = OperatorProperties(
            block_diagonal=True,
            rank=rank,
            evidence={
                "block_diagonal": "construction",
                **_derived_rank_evidence(blocks_, rank),
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=all(block.capabilities.transpose for block in blocks_),
            adjoint=all(block.capabilities.adjoint for block in blocks_),
            materialize=all(block.capabilities.materialize for block in blocks_),
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {"kind": "block-diagonal", "blocks": [b.operator_id for b in blocks_]},
        )

    def mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.source.validate(vector)
        return tuple(
            block.mv(value) for block, value in zip(self.blocks, values, strict=True)
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.target.validate(vector)
        return tuple(
            block.transpose_mv(value)
            for block, value in zip(self.blocks, values, strict=True)
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        rows = []
        dtype = jnp.result_type(*[block._materialize().dtype for block in self.blocks])
        for row_index, row_block in enumerate(self.blocks):
            rows.append(
                jnp.concatenate(
                    tuple(
                        row_block._materialize()
                        if row_index == column_index
                        else jnp.zeros(
                            (row_block.target.size, column_block.source.size),
                            dtype=dtype,
                        )
                        for column_index, column_block in enumerate(self.blocks)
                    ),
                    axis=1,
                )
            )
        return jnp.concatenate(tuple(rows), axis=0)


class LowRankLinearOperator(AbstractLinearOperator):
    """Rectangular operator ``U Vᴴ`` with explicit rank factors."""

    left_factor: Array
    right_factor: Array

    def __init__(
        self,
        left_factor: ArrayLike,
        right_factor: ArrayLike,
        /,
        *,
        source: AbstractVectorSpace | None = None,
        target: AbstractVectorSpace | None = None,
        exact_rank: bool = False,
        operator_id: str | None = None,
    ):
        left_, right_ = jnp.asarray(left_factor), jnp.asarray(right_factor)
        if left_.ndim != 2 or right_.ndim != 2 or left_.shape[1] != right_.shape[1]:
            raise ValueError("factors must have shapes (m, r) and (n, r).")
        dtype = jnp.result_type(left_, right_, float)
        left_, right_ = left_.astype(dtype), right_.astype(dtype)
        target_ = _space(int(left_.shape[0]), dtype, target)
        source_ = _space(int(right_.shape[0]), dtype, source)
        rank = int(left_.shape[1]) if exact_rank else None
        self.source = source_
        self.target = target_
        self.left_factor = left_
        self.right_factor = right_
        self.properties = OperatorProperties(
            rank=rank,
            evidence={"rank": "asserted"} if rank is not None else {},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {"kind": "low-rank", "source": source_.space_id, "target": target_.space_id},
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.source.flatten(vector)
        return self.target.unflatten(
            self.left_factor @ (jnp.conj(self.right_factor.T) @ x)
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        y = self.target.flatten(vector)
        return self.source.unflatten(
            jnp.conj(self.right_factor) @ (self.left_factor.T @ y)
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return self.left_factor @ jnp.conj(self.right_factor.T)


class SymmetricLowRankLinearOperator(AbstractLinearOperator):
    """Self-adjoint low-rank operator ``U diag(weights) Uᴴ``."""

    factor: Array
    weights: Array

    def __init__(
        self,
        factor: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        space: AbstractVectorSpace | None = None,
        positive_semidefinite: bool = False,
        operator_id: str | None = None,
    ):
        factor_ = jnp.asarray(factor)
        if factor_.ndim != 2:
            raise ValueError("factor must have shape (n, r).")
        if not jnp.issubdtype(factor_.dtype, jnp.inexact):
            factor_ = factor_.astype(float)
        weights_ = (
            jnp.ones((factor_.shape[1],), dtype=factor_.real.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=factor_.real.dtype)
        )
        if weights_.shape != (factor_.shape[1],):
            raise ValueError("weights must have one entry per factor column.")
        factor_ = eqx.error_if(
            factor_,
            jnp.any(~jnp.isfinite(factor_)),
            "factor entries must be finite.",
        )
        invalid_weights = ~jnp.isfinite(weights_)
        if positive_semidefinite:
            invalid_weights = invalid_weights | (weights_ < 0.0)
        weights_ = eqx.error_if(
            weights_,
            jnp.any(invalid_weights),
            "weights must be finite and satisfy the declared semidefiniteness.",
        )
        space_ = _space(int(factor_.shape[0]), factor_.dtype, space)
        self_adjoint = _has_euclidean_pairing(space_)
        if positive_semidefinite and not self_adjoint:
            raise ValueError(
                "positive_semidefinite requires a Euclidean pairing for "
                "SymmetricLowRankLinearOperator."
            )
        rank = None
        self.source = space_
        self.target = space_
        self.factor = factor_
        self.weights = weights_
        self.properties = OperatorProperties(
            self_adjoint=self_adjoint,
            positive_semidefinite=positive_semidefinite,
            rank=rank,
            evidence=_construction_evidence(
                self_adjoint=self_adjoint,
                positive_semidefinite=positive_semidefinite,
                rank=rank,
            ),
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id, {"kind": "symmetric-low-rank", "space": space_.space_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.source.flatten(vector)
        return self.target.unflatten(
            self.factor @ (self.weights * (jnp.conj(self.factor.T) @ x))
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.target.flatten(vector)
        return self.source.unflatten(
            jnp.conj(self.factor) @ (self.weights * (self.factor.T @ coordinates))
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return (self.factor * self.weights) @ jnp.conj(self.factor.T)


class DiagonalPlusLowRankLinearOperator(AbstractLinearOperator):
    """Structured endomorphism ``diag(d) + U Vᴴ`` supporting Woodbury solves."""

    diagonal: Array
    left_factor: Array
    right_factor: Array
    nonsingular_diagonal: bool = eqx.field(static=True)

    def __init__(
        self,
        diagonal: ArrayLike,
        left_factor: ArrayLike,
        right_factor: ArrayLike | None = None,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        nonsingular_diagonal: bool = False,
        operator_id: str | None = None,
    ):
        diagonal_ = jnp.asarray(diagonal)
        left_ = jnp.asarray(left_factor)
        right_ = left_ if right_factor is None else jnp.asarray(right_factor)
        if diagonal_.ndim != 1 or left_.ndim != 2 or right_.shape != left_.shape:
            raise ValueError("Expected diagonal (n,) and factors (n, r).")
        if left_.shape[0] != diagonal_.size:
            raise ValueError("Factor rows must match diagonal length.")
        dtype = jnp.result_type(diagonal_, left_, right_, float)
        diagonal_, left_, right_ = (
            value.astype(dtype) for value in (diagonal_, left_, right_)
        )
        nonsingular = bool(nonsingular_diagonal)
        if nonsingular:
            diagonal_ = eqx.error_if(
                diagonal_,
                jnp.any(~jnp.isfinite(diagonal_) | (diagonal_ == 0)),
                "A certified nonsingular diagonal must be finite and nonzero.",
            )
        space_ = _space(int(diagonal_.size), dtype, space)
        self.source = space_
        self.target = space_
        self.diagonal = diagonal_
        self.left_factor = left_
        self.right_factor = right_
        self.nonsingular_diagonal = nonsingular
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "diagonal-plus-low-rank",
                "space": space_.space_id,
                "nonsingular_diagonal": nonsingular,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.source.flatten(vector)
        y = self.diagonal * x + self.left_factor @ (jnp.conj(self.right_factor.T) @ x)
        return self.target.unflatten(y)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        x = self.target.flatten(vector)
        y = self.diagonal * x + jnp.conj(self.right_factor) @ (self.left_factor.T @ x)
        return self.source.unflatten(y)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return jnp.diag(self.diagonal) + self.left_factor @ jnp.conj(self.right_factor.T)


class KroneckerLinearOperator(AbstractLinearOperator):
    """Kronecker product whose action contracts one factor axis at a time."""

    factors: tuple[AbstractLinearOperator, ...]
    source: TensorProductSpace
    target: TensorProductSpace

    def __init__(
        self,
        factors: Sequence[AbstractLinearOperator],
        /,
        *,
        operator_id: str | None = None,
    ):
        factors_ = tuple(factors)
        if not factors_ or any(factor.batch_shape for factor in factors_):
            raise ValueError("Kronecker factors must be nonempty and unbatched.")
        source = TensorProductSpace(tuple(factor.source for factor in factors_))
        target = TensorProductSpace(tuple(factor.target for factor in factors_))
        factor_ranks = tuple(factor.properties.rank for factor in factors_)
        rank = (
            prod(value for value in factor_ranks if value is not None)
            if all(value is not None for value in factor_ranks)
            else None
        )
        self.factors = factors_
        self.source = source
        self.target = target
        self.properties = OperatorProperties(
            rank=rank,
            evidence=_derived_rank_evidence(factors_, rank),
        )
        self.capabilities = OperatorCapabilities(
            transpose=all(factor.capabilities.transpose for factor in factors_),
            adjoint=all(factor.capabilities.adjoint for factor in factors_),
            materialize=all(factor.capabilities.materialize for factor in factors_),
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {"kind": "kronecker", "factors": [factor.operator_id for factor in factors_]},
        )

    def _apply(self, vector: Any, mode: Literal["forward", "transpose", "adjoint"]):
        value = (
            self.source.validate(vector)
            if mode == "forward"
            else self.target.validate(vector)
        )
        for axis, factor in enumerate(self.factors):
            value = _apply_factor_axis(value, factor, axis, mode)
        return value

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._apply(vector, "forward")

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._apply(vector, "transpose")

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        result = jnp.asarray([[1.0]], dtype=self.source.structure().dtype)
        for factor in self.factors:
            result = jnp.kron(result, factor._materialize())
        return result


class KroneckerSumLinearOperator(AbstractLinearOperator):
    """Kronecker sum of square factors without materializing the full matrix."""

    factors: tuple[AbstractLinearOperator, ...]
    source: TensorProductSpace
    target: TensorProductSpace

    def __init__(
        self,
        factors: Sequence[AbstractLinearOperator],
        /,
        *,
        operator_id: str | None = None,
    ):
        factors_ = tuple(factors)
        if not factors_ or any(
            factor.batch_shape or not factor.source.compatible(factor.target)
            for factor in factors_
        ):
            raise ValueError("Kronecker-sum factors must be square and unbatched.")
        space = TensorProductSpace(tuple(factor.source for factor in factors_))
        self.factors = factors_
        self.source = space
        self.target = space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=all(factor.capabilities.transpose for factor in factors_),
            adjoint=all(factor.capabilities.adjoint for factor in factors_),
            materialize=all(factor.capabilities.materialize for factor in factors_),
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "kronecker-sum",
                "factors": [factor.operator_id for factor in factors_],
            },
        )

    def _apply(self, vector: Any, mode: Literal["forward", "transpose", "adjoint"]):
        value = self.source.validate(vector)
        output = jnp.zeros_like(value)
        for axis, factor in enumerate(self.factors):
            output = output + _apply_factor_axis(value, factor, axis, mode)
        return output

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._apply(vector, "forward")

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self._apply(vector, "transpose")

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        sizes = tuple(factor.source.size for factor in self.factors)
        dtype = self.source.structure().dtype
        result = jnp.zeros((self.source.size, self.source.size), dtype=dtype)
        for index, factor in enumerate(self.factors):
            term = jnp.asarray([[1.0]], dtype=dtype)
            for axis, size in enumerate(sizes):
                matrix = (
                    factor._materialize() if axis == index else jnp.eye(size, dtype=dtype)
                )
                term = jnp.kron(term, matrix)
            result = result + term
        return result


class StackedLinearOperator(AbstractLinearOperator):
    """Vertical or horizontal stack retaining explicit block-vector structure."""

    operators: tuple[AbstractLinearOperator, ...]
    axis: Literal["vertical", "horizontal"]

    def __init__(
        self,
        operators: Sequence[AbstractLinearOperator],
        /,
        *,
        axis: Literal["vertical", "horizontal"] = "vertical",
        operator_id: str | None = None,
    ):
        operators_ = tuple(operators)
        if not operators_ or any(operator.batch_shape for operator in operators_):
            raise ValueError("Stacked operators must be nonempty and unbatched.")
        if axis not in ("vertical", "horizontal"):
            raise ValueError("axis must be 'vertical' or 'horizontal'.")
        if axis == "vertical":
            source = operators_[0].source
            if any(not source.compatible(operator.source) for operator in operators_):
                raise ValueError("Vertical stack operators must share a source space.")
            target = BlockSpace(tuple(operator.target for operator in operators_))
        else:
            target = operators_[0].target
            if any(not target.compatible(operator.target) for operator in operators_):
                raise ValueError("Horizontal stack operators must share a target space.")
            source = BlockSpace(tuple(operator.source for operator in operators_))
        self.operators = operators_
        self.axis = axis
        self.source = source
        self.target = target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=all(operator.capabilities.transpose for operator in operators_),
            adjoint=all(operator.capabilities.adjoint for operator in operators_),
            materialize=all(operator.capabilities.materialize for operator in operators_),
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "stacked",
                "axis": axis,
                "operators": [o.operator_id for o in operators_],
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if self.axis == "vertical":
            return tuple(operator.mv(vector) for operator in self.operators)
        values = self.source.validate(vector)
        images = tuple(
            operator.mv(value)
            for operator, value in zip(self.operators, values, strict=True)
        )
        result = images[0]
        for image in images[1:]:
            result = jax.tree.map(lambda left, right: left + right, result, image)
        return result

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if self.axis == "horizontal":
            return tuple(operator.transpose_mv(vector) for operator in self.operators)
        values = self.target.validate(vector)
        images = tuple(
            operator.transpose_mv(value)
            for operator, value in zip(self.operators, values, strict=True)
        )
        result = images[0]
        for image in images[1:]:
            result = jax.tree.map(lambda left, right: left + right, result, image)
        return result

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        matrices = tuple(operator._materialize() for operator in self.operators)
        return jnp.concatenate(matrices, axis=0 if self.axis == "vertical" else 1)


class SchurComplementLinearOperator(AbstractLinearOperator):
    """Matrix-free Schur complement ``D - C A⁻¹ B`` with an explicit inverse action."""

    diagonal_block: AbstractLinearOperator
    lower_block: AbstractLinearOperator
    upper_block: AbstractLinearOperator
    inverse_action: Callable[[PyTree[Any]], PyTree[Array]]

    def __init__(
        self,
        diagonal_block: AbstractLinearOperator,
        lower_block: AbstractLinearOperator,
        inverse_action: Callable[[PyTree[Any]], PyTree[Array]],
        upper_block: AbstractLinearOperator,
        /,
        *,
        operator_id: str | None = None,
    ):
        if not callable(inverse_action):
            raise TypeError("inverse_action must be callable.")
        blocks = (diagonal_block, lower_block, upper_block)
        if any(not isinstance(block, AbstractLinearOperator) for block in blocks):
            raise TypeError("Schur complement blocks must be linear operators.")
        if any(block.batch_shape for block in blocks):
            raise ValueError("SchurComplementLinearOperator requires unbatched blocks.")
        _same(upper_block.target, lower_block.source)
        _same(diagonal_block.source, upper_block.source)
        _same(diagonal_block.target, lower_block.target)
        self.source = diagonal_block.source
        self.target = diagonal_block.target
        self.diagonal_block = diagonal_block
        self.lower_block = lower_block
        self.upper_block = upper_block
        self.inverse_action = inverse_action
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=False, adjoint=False, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "schur-complement",
                "diagonal": diagonal_block.operator_id,
                "lower": lower_block.operator_id,
                "upper": upper_block.operator_id,
            },
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        direct = self.diagonal_block.mv(vector)
        correction = self.lower_block.mv(self.inverse_action(self.upper_block.mv(vector)))
        return jax.tree.map(lambda left, right: left - right, direct, correction)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise ValueError(
            "SchurComplementLinearOperator has no declared transpose action."
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise ValueError("SchurComplementLinearOperator has no declared adjoint action.")

    def _materialize(self, /) -> Array:
        raise ValueError("SchurComplementLinearOperator is intentionally matrix-free.")


def _is_structured_exact(operator: AbstractLinearOperator, /) -> bool:
    """Return whether the native structured backend can solve the operator exactly."""
    if isinstance(operator, KroneckerLinearOperator):
        return all(
            factor.source.size == factor.target.size
            and (isinstance(factor, DenseLinearOperator) or _is_structured_exact(factor))
            for factor in operator.factors
        )
    if isinstance(
        operator,
        (
            IdentityLinearOperator,
            DiagonalLinearOperator,
            PermutationLinearOperator,
            TriangularLinearOperator,
            TridiagonalLinearOperator,
            BandedLinearOperator,
            DiagonalPlusLowRankLinearOperator,
        ),
    ):
        return not operator.batch_shape
    return isinstance(operator, BlockDiagonalLinearOperator) and all(
        block.source.size == block.target.size
        and (isinstance(block, DenseLinearOperator) or _is_structured_exact(block))
        for block in operator.blocks
    )


__all__ = [
    "BandedLinearOperator",
    "BlockDiagonalLinearOperator",
    "DiagonalPlusLowRankLinearOperator",
    "KroneckerLinearOperator",
    "KroneckerSumLinearOperator",
    "LowRankLinearOperator",
    "PermutationLinearOperator",
    "SchurComplementLinearOperator",
    "StackedLinearOperator",
    "SymmetricLowRankLinearOperator",
    "TriangularLinearOperator",
    "TridiagonalLinearOperator",
]
