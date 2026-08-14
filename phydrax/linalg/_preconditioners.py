#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._spaces import (
    _coordinate_dtype,
    _has_diagonal_pairing,
    AbstractVectorSpace,
    ArraySpace,
)


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _validated(value: Array, invalid: Array, message: str, /) -> Array:
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(value, invalid, message)
    if bool(invalid):
        raise ValueError(message)
    return value


def _coerce_coefficients(
    value: Array,
    space: AbstractVectorSpace,
    name: str,
    /,
) -> Array:
    space_dtype = _coordinate_dtype(space)
    result_dtype = np.dtype(
        jax.dtypes.canonicalize_dtype(jnp.result_type(value.dtype, space_dtype))
    )
    if result_dtype != space_dtype:
        raise TypeError(
            f"{name} acting on {space_dtype} coordinates would produce "
            f"{result_dtype} coordinates."
        )
    return value.astype(space_dtype)


class AbstractPreconditioner(StrictModule):
    """Prepared approximate inverse with explicit source-space semantics."""

    space: AbstractVectorSpace
    positive_definite: bool = eqx.field(static=True)
    preconditioner_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError


class IdentityPreconditioner(AbstractPreconditioner):
    def __init__(self, space: AbstractVectorSpace, /):
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        self.space = space
        self.positive_definite = True
        self.preconditioner_id = canonical_fingerprint(
            {"kind": "identity", "space": space.space_id}
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.space.validate(residual)


class DiagonalPreconditioner(AbstractPreconditioner):
    """Jacobi inverse prepared from a nonzero canonical diagonal."""

    inverse_diagonal: Array

    def __init__(
        self,
        diagonal: ArrayLike,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        positive_definite: bool | None = None,
        preconditioner_id: str | None = None,
    ):
        values = _inexact(diagonal)
        if values.ndim != 1:
            raise ValueError("diagonal must be one-dimensional.")
        values = _validated(
            values,
            jnp.any(~jnp.isfinite(values)) | jnp.any(values == 0),
            "diagonal entries must be finite and nonzero.",
        )
        space_ = ArraySpace(values.shape, dtype=values.dtype) if space is None else space
        if not isinstance(space_, AbstractVectorSpace) or space_.size != values.size:
            raise ValueError("space size must match the diagonal length.")
        values = _coerce_coefficients(values, space_, "diagonal")
        if positive_definite is None:
            positive_test = (
                not jnp.issubdtype(values.dtype, jnp.complexfloating)
                and _has_diagonal_pairing(space_)
                and jnp.all(values > 0.0)
            )
            positive = (
                False
                if isinstance(positive_test, jax_core.Tracer)
                else bool(positive_test)
            )
        else:
            positive = bool(positive_definite)
        self.space = space_
        self.inverse_diagonal = jnp.reciprocal(values)
        self.positive_definite = positive
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "diagonal",
            space_,
            extra={"size": values.size},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.space.flatten(residual)
        return self.space.unflatten(self.inverse_diagonal * coordinates)


class BlockDiagonalPreconditioner(AbstractPreconditioner):
    """Prepared inverse of independent dense canonical-coordinate blocks."""

    inverse_blocks: tuple[Array, ...]
    offsets: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[ArrayLike],
        /,
        *,
        space: AbstractVectorSpace,
        positive_definite: bool = False,
        preconditioner_id: str | None = None,
    ):
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        matrices = tuple(jnp.asarray(block) for block in blocks)
        if not matrices:
            raise ValueError("blocks must contain at least one matrix.")
        sizes: list[int] = []
        inverses: list[Array] = []
        for original in matrices:
            if original.ndim != 2 or original.shape[0] != original.shape[1]:
                raise ValueError("Each block must be a square matrix.")
            matrix = _coerce_coefficients(original, space, "block")
            matrix = _validated(
                matrix,
                jnp.any(~jnp.isfinite(matrix)),
                "Preconditioner blocks must contain only finite values.",
            )
            inverse = jnp.linalg.inv(matrix)
            inverse = _validated(
                inverse,
                jnp.any(~jnp.isfinite(inverse)),
                "Every preconditioner block must be nonsingular.",
            )
            sizes.append(int(matrix.shape[0]))
            inverses.append(inverse)
        if sum(sizes) != space.size:
            raise ValueError("Block dimensions must partition the preconditioner space.")
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        self.space = space
        self.inverse_blocks = tuple(inverses)
        self.offsets = tuple(offsets)
        self.positive_definite = bool(positive_definite)
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "block-diagonal",
            space,
            extra={"sizes": sizes},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.space.flatten(residual)
        pieces = tuple(
            inverse @ coordinates[start:stop]
            for inverse, start, stop in zip(
                self.inverse_blocks,
                self.offsets[:-1],
                self.offsets[1:],
                strict=True,
            )
        )
        return self.space.unflatten(jnp.concatenate(pieces))


class IncompleteFactorizationPreconditioner(AbstractPreconditioner):
    """Triangular-factor approximate inverse, suitable for ILU/IC factors."""

    lower: Array
    upper: Array
    unit_lower: bool = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        space: AbstractVectorSpace | None = None,
        unit_lower: bool = False,
        positive_definite: bool = False,
        preconditioner_id: str | None = None,
    ):
        lower_ = _inexact(lower)
        upper_ = _inexact(upper)
        if (
            lower_.ndim != 2
            or upper_.shape != lower_.shape
            or lower_.shape[0] != lower_.shape[1]
        ):
            raise ValueError("lower and upper must be equal-sized square matrices.")
        size = int(lower_.shape[0])
        space_ = (
            ArraySpace((size,), dtype=jnp.result_type(lower_, upper_))
            if space is None
            else space
        )
        if not isinstance(space_, AbstractVectorSpace) or space_.size != size:
            raise ValueError("space size must match factor dimensions.")
        lower_ = _coerce_coefficients(lower_, space_, "lower factor")
        upper_ = _coerce_coefficients(upper_, space_, "upper factor")
        lower_ = _validated(
            lower_,
            jnp.any(~jnp.isfinite(lower_))
            | ((not unit_lower) & jnp.any(jnp.diag(lower_) == 0)),
            "Incomplete lower factors must be finite and nonsingular.",
        )
        upper_ = _validated(
            upper_,
            jnp.any(~jnp.isfinite(upper_)) | jnp.any(jnp.diag(upper_) == 0),
            "Incomplete upper factors must be finite and nonsingular.",
        )
        self.space = space_
        self.lower = lower_
        self.upper = upper_
        self.unit_lower = bool(unit_lower)
        self.positive_definite = bool(positive_definite)
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "incomplete-factorization",
            space_,
            extra={"unit_lower": self.unit_lower},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.space.flatten(residual)
        intermediate = jsp.linalg.solve_triangular(
            self.lower,
            coordinates,
            lower=True,
            unit_diagonal=self.unit_lower,
        )
        solution = jsp.linalg.solve_triangular(self.upper, intermediate, lower=False)
        return self.space.unflatten(solution)


class LowRankWoodburyPreconditioner(AbstractPreconditioner):
    """Prepared inverse of ``diag(d) + U C Vᴴ`` via Woodbury."""

    inverse_diagonal: Array
    left: Array
    right: Array
    middle_inverse: Array

    def __init__(
        self,
        diagonal: ArrayLike,
        left: ArrayLike,
        core: ArrayLike,
        /,
        *,
        right: ArrayLike | None = None,
        space: AbstractVectorSpace | None = None,
        positive_definite: bool = False,
        preconditioner_id: str | None = None,
    ):
        diagonal_ = _inexact(diagonal)
        left_ = _inexact(left)
        core_ = _inexact(core)
        right_ = left_ if right is None else _inexact(right)
        if diagonal_.ndim != 1:
            raise ValueError("diagonal must be one-dimensional.")
        size = int(diagonal_.size)
        if (
            left_.ndim != 2
            or right_.ndim != 2
            or left_.shape[0] != size
            or right_.shape[0] != size
        ):
            raise ValueError(
                "Low-rank factors must have leading dimension len(diagonal)."
            )
        rank = int(left_.shape[1])
        if right_.shape[1] != rank or core_.shape != (rank, rank):
            raise ValueError("Low-rank factors and core dimensions must agree.")
        dtype = jnp.result_type(diagonal_, left_, right_, core_)
        space_ = ArraySpace((size,), dtype=dtype) if space is None else space
        if not isinstance(space_, AbstractVectorSpace) or space_.size != size:
            raise ValueError("space size must match the diagonal length.")
        diagonal_ = _coerce_coefficients(diagonal_, space_, "diagonal")
        left_ = _coerce_coefficients(left_, space_, "left factor")
        right_ = _coerce_coefficients(right_, space_, "right factor")
        core_ = _coerce_coefficients(core_, space_, "core")
        invalid_coefficients = (
            jnp.any(~jnp.isfinite(diagonal_))
            | jnp.any(~jnp.isfinite(left_))
            | jnp.any(~jnp.isfinite(right_))
            | jnp.any(~jnp.isfinite(core_))
            | jnp.any(diagonal_ == 0)
        )
        diagonal_ = _validated(
            diagonal_,
            invalid_coefficients,
            "Woodbury coefficients must be finite and the diagonal nonzero.",
        )
        inverse_diagonal = jnp.reciprocal(diagonal_)
        weighted_left = inverse_diagonal[:, None] * left_
        core_inverse = jnp.linalg.inv(core_)
        core_inverse = _validated(
            core_inverse,
            jnp.any(~jnp.isfinite(core_inverse)),
            "Woodbury core must be nonsingular.",
        )
        middle = core_inverse + jnp.conj(right_.T) @ weighted_left
        middle_inverse = jnp.linalg.inv(middle)
        middle_inverse = _validated(
            middle_inverse,
            jnp.any(~jnp.isfinite(middle_inverse)),
            "Woodbury correction system must be nonsingular.",
        )
        self.space = space_
        self.inverse_diagonal = inverse_diagonal
        self.left = left_
        self.right = right_
        self.middle_inverse = middle_inverse
        self.positive_definite = bool(positive_definite)
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "low-rank-woodbury",
            space_,
            extra={"rank": rank},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.space.flatten(residual)
        diagonal_solution = self.inverse_diagonal * coordinates
        correction_coordinates = jnp.conj(self.right.T) @ diagonal_solution
        correction = (
            self.inverse_diagonal[:, None]
            * self.left
            @ (self.middle_inverse @ correction_coordinates)
        )
        return self.space.unflatten(diagonal_solution - correction)


class OperatorPreconditioner(AbstractPreconditioner):
    """Adapter for an already-prepared inverse action."""

    operator: AbstractLinearOperator

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        positive_definite: bool = False,
        preconditioner_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not operator.source.compatible(operator.target) or operator.batch_shape:
            raise ValueError(
                "A preconditioning operator must be an unbatched endomorphism."
            )
        self.space = operator.source
        self.operator = operator
        self.positive_definite = bool(positive_definite)
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "operator",
            operator.source,
            extra={"operator": operator.operator_id},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(residual)


class MultigridPreconditioner(AbstractPreconditioner):
    """One recursive geometric/algebraic V-cycle over explicit transfer operators."""

    operators: tuple[AbstractLinearOperator, ...]
    smoothers: tuple[AbstractPreconditioner, ...]
    restrictions: tuple[AbstractLinearOperator, ...]
    prolongations: tuple[AbstractLinearOperator, ...]
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)

    def __init__(
        self,
        operators: Sequence[AbstractLinearOperator],
        smoothers: Sequence[AbstractPreconditioner],
        restrictions: Sequence[AbstractLinearOperator],
        prolongations: Sequence[AbstractLinearOperator],
        /,
        *,
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
        positive_definite: bool = False,
        preconditioner_id: str | None = None,
    ):
        operators_ = tuple(operators)
        smoothers_ = tuple(smoothers)
        restrictions_ = tuple(restrictions)
        prolongations_ = tuple(prolongations)
        if not all(
            isinstance(operator, AbstractLinearOperator) for operator in operators_
        ):
            raise TypeError("operators must contain AbstractLinearOperator values.")
        if not all(
            isinstance(smoother, AbstractPreconditioner) for smoother in smoothers_
        ):
            raise TypeError("smoothers must contain AbstractPreconditioner values.")
        if not all(
            isinstance(transfer, AbstractLinearOperator)
            for transfer in restrictions_ + prolongations_
        ):
            raise TypeError(
                "restrictions and prolongations must contain linear operators."
            )
        levels = len(operators_)
        if levels < 2 or len(smoothers_) != levels:
            raise ValueError(
                "Multigrid requires at least two operators and one smoother per level."
            )
        if len(restrictions_) != levels - 1 or len(prolongations_) != levels - 1:
            raise ValueError(
                "Multigrid requires one restriction/prolongation pair per transition."
            )
        for level, (operator, smoother) in enumerate(
            zip(operators_, smoothers_, strict=True)
        ):
            if operator.batch_shape or not operator.source.compatible(operator.target):
                raise ValueError(
                    "Every multigrid level operator must be an unbatched endomorphism."
                )
            if not smoother.space.compatible(operator.source):
                raise ValueError(f"Smoother space mismatch at level {level}.")
        for level, (restriction, prolongation) in enumerate(
            zip(restrictions_, prolongations_, strict=True)
        ):
            if restriction.batch_shape or prolongation.batch_shape:
                raise ValueError("Multigrid transfer operators must be unbatched.")
            fine = operators_[level].source
            coarse = operators_[level + 1].source
            if not restriction.source.compatible(
                fine
            ) or not restriction.target.compatible(coarse):
                raise ValueError(f"Restriction space mismatch at transition {level}.")
            if not prolongation.source.compatible(
                coarse
            ) or not prolongation.target.compatible(fine):
                raise ValueError(f"Prolongation space mismatch at transition {level}.")
        pre = int(pre_smoothing)
        post = int(post_smoothing)
        if pre < 0 or post < 0:
            raise ValueError("Smoothing counts must be non-negative.")
        self.space = operators_[0].source
        self.operators = operators_
        self.smoothers = smoothers_
        self.restrictions = restrictions_
        self.prolongations = prolongations_
        self.pre_smoothing = pre
        self.post_smoothing = post
        self.positive_definite = bool(positive_definite)
        self.preconditioner_id = _identifier(
            preconditioner_id,
            "multigrid",
            self.space,
            extra={"levels": levels, "pre": pre, "post": post},
        )

    def apply(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self._cycle(0, self.space.validate(residual))

    def _cycle(self, level: int, residual: PyTree[Array], /) -> PyTree[Array]:
        if level == len(self.operators) - 1:
            return self.smoothers[level].apply(residual)
        estimate = jax.tree.map(jnp.zeros_like, residual)
        operator = self.operators[level]
        smoother = self.smoothers[level]
        for _ in range(self.pre_smoothing):
            defect = _subtract(residual, operator.mv(estimate))
            estimate = _add(estimate, smoother.apply(defect))
        defect = _subtract(residual, operator.mv(estimate))
        coarse_residual = self.restrictions[level].mv(defect)
        coarse_correction = self._cycle(level + 1, coarse_residual)
        estimate = _add(estimate, self.prolongations[level].mv(coarse_correction))
        for _ in range(self.post_smoothing):
            defect = _subtract(residual, operator.mv(estimate))
            estimate = _add(estimate, smoother.apply(defect))
        return estimate


def _add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _subtract(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _identifier(
    value: str | None,
    kind: str,
    space: AbstractVectorSpace,
    /,
    *,
    extra: dict[str, Any],
) -> str:
    if value is None:
        return canonical_fingerprint(
            {"kind": kind, "space": space.space_id, "structure": extra}
        )
    identifier = str(value)
    if not identifier:
        raise ValueError("preconditioner_id must be non-empty.")
    return identifier


__all__ = [
    "AbstractPreconditioner",
    "BlockDiagonalPreconditioner",
    "DiagonalPreconditioner",
    "IdentityPreconditioner",
    "IncompleteFactorizationPreconditioner",
    "LowRankWoodburyPreconditioner",
    "MultigridPreconditioner",
    "OperatorPreconditioner",
]
