#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from ._costs import PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._sparse_triangular import (
    analyze_sparse_triangular,
    SparseTriangularFactor,
    SparseTriangularStatus,
)


GaussSeidelDirection: TypeAlias = Literal["forward", "backward", "symmetric"]


def _explicit_csr(operator: AbstractLinearOperator, /) -> sp.csr_matrix:
    if isinstance(operator, AbstractSparseLinearOperator):
        storage = operator.sparse_storage()
        if not storage.canonical or not storage.sorted_indices:
            raise ValueError("Gauss-Seidel requires canonical sorted CSR storage.")
        matrix = sp.csr_matrix(
            (
                np.asarray(storage.values),
                np.asarray(storage.indices),
                np.asarray(storage.indptr),
            ),
            shape=storage.shape,
        )
    elif isinstance(operator, DenseLinearOperator):
        matrix = sp.csr_matrix(np.asarray(operator.matrix))
    else:
        raise TypeError("Gauss-Seidel requires an explicit dense or sparse operator.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gauss-Seidel requires a square operator.")
    diagonal = matrix.diagonal()
    if np.any(~np.isfinite(matrix.data)) or np.any(~np.isfinite(diagonal)):
        raise ValueError("Gauss-Seidel requires finite operator entries.")
    if np.any(diagonal == 0):
        raise ValueError("Gauss-Seidel requires a nonzero diagonal.")
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _triangular_storage(
    matrix: sp.csr_matrix,
    triangle: Literal["lower", "upper"],
    relaxation: float,
    /,
) -> SparseStorage:
    triangular = (
        sp.tril(matrix, format="csr")
        if triangle == "lower"
        else sp.triu(matrix, format="csr")
    )
    triangular.setdiag(matrix.diagonal() / relaxation)
    triangular.sum_duplicates()
    triangular.sort_indices()
    index_dtype = np.int32 if matrix.shape[0] <= np.iinfo(np.int32).max else np.int64
    return SparseStorage(
        jnp.asarray(triangular.data),
        jnp.asarray(triangular.indices, dtype=index_dtype),
        jnp.asarray(triangular.indptr, dtype=index_dtype),
        shape=triangular.shape,
    )


def _factor(
    storage: SparseStorage,
    triangle: Literal["lower", "upper"],
    /,
    *,
    previous: SparseTriangularFactor | None = None,
) -> SparseTriangularFactor:
    analysis = analyze_sparse_triangular(storage, triangle=triangle)
    if previous is not None:
        if previous.analysis.pattern_id != analysis.pattern_id:
            raise ValueError(
                "Gauss-Seidel numeric refresh requires an unchanged triangular pattern."
            )
        analysis = previous.analysis
    return SparseTriangularFactor(analysis, storage.values)


class GaussSeidelPreconditioner(AbstractPreconditioner):
    """Prepared forward, backward, or multiplicative symmetric sweep."""

    operator: AbstractLinearOperator
    forward_factor: SparseTriangularFactor | None
    backward_factor: SparseTriangularFactor | None
    direction: GaussSeidelDirection = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        direction: GaussSeidelDirection = "symmetric",
        relaxation: float = 1.0,
        previous: "GaussSeidelPreconditioner | None" = None,
    ):
        if direction not in ("forward", "backward", "symmetric"):
            raise ValueError(f"Unknown Gauss-Seidel direction {direction!r}.")
        omega = float(relaxation)
        if not isfinite(omega) or omega <= 0.0 or omega >= 2.0:
            raise ValueError("Gauss-Seidel relaxation must lie strictly between 0 and 2.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Gauss-Seidel requires an unbatched endomorphism.")
        matrix = _explicit_csr(operator)
        old_forward = None if previous is None else previous.forward_factor
        old_backward = None if previous is None else previous.backward_factor
        forward = (
            _factor(
                _triangular_storage(matrix, "lower", omega),
                "lower",
                previous=old_forward,
            )
            if direction in ("forward", "symmetric")
            else None
        )
        backward = (
            _factor(
                _triangular_storage(matrix, "upper", omega),
                "upper",
                previous=old_backward,
            )
            if direction in ("backward", "symmetric")
            else None
        )
        symmetric = direction == "symmetric" and operator.properties.certifies(
            "self_adjoint"
        )
        positive = symmetric and operator.properties.certifies("positive_definite")
        evidence = {
            "linear": "construction",
            "stationary": "construction",
            **({"self_adjoint": "transformed"} if symmetric else {}),
            **({"positive_definite": "transformed"} if positive else {}),
        }
        self.operator = operator
        self.forward_factor = forward
        self.backward_factor = backward
        self.direction = direction
        self.relaxation = omega
        self.space = operator.source
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=symmetric,
            positive_definite=positive,
            evidence=evidence,
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "gauss-seidel",
                "operator": operator.operator_id,
                "direction": direction,
                "relaxation": omega,
                "forward_pattern": (
                    None if forward is None else forward.analysis.pattern_id
                ),
                "backward_pattern": (
                    None if backward is None else backward.analysis.pattern_id
                ),
            }
        )

    def _solve(
        self,
        factor: SparseTriangularFactor,
        residual: PyTree[Any],
        /,
    ) -> PyTree[Array]:
        coordinates = self.space.flatten(self.space.validate(residual))
        result = factor.solve(coordinates)
        value = eqx.error_if(
            result.value,
            result.status != int(SparseTriangularStatus.SUCCESS),
            "Gauss-Seidel triangular sweep failed.",
        )
        return self.space.unflatten(value)

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        residual_ = self.space.validate(residual)
        if self.direction == "forward":
            if self.forward_factor is None:
                raise RuntimeError("Prepared forward factor is missing.")
            return self._solve(self.forward_factor, residual_)
        if self.direction == "backward":
            if self.backward_factor is None:
                raise RuntimeError("Prepared backward factor is missing.")
            return self._solve(self.backward_factor, residual_)
        if self.forward_factor is None or self.backward_factor is None:
            raise RuntimeError("Prepared symmetric factors are missing.")
        first = self._solve(self.forward_factor, residual_)
        defect = jax.tree.map(
            lambda rhs, image: rhs - image,
            residual_,
            self.operator.mv(first),
        )
        second = self._solve(self.backward_factor, defect)
        return jax.tree.map(lambda left, right: left + right, first, second)

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        if not self.space.compatible(setup_operator.source):
            raise ValueError("Gauss-Seidel and setup operator spaces must match.")
        factors = tuple(
            factor
            for factor in (self.forward_factor, self.backward_factor)
            if factor is not None
        )
        storage = sum(
            int(
                factor.values.nbytes
                + factor.analysis.indices.nbytes
                + factor.analysis.indptr.nbytes
                + factor.analysis.row_indices.nbytes
                + factor.analysis.row_levels.nbytes
            )
            for factor in factors
        )
        itemsize = self.space.flatten(
            self.space.unflatten(jnp.zeros(self.space.size))
        ).dtype.itemsize
        multiplier = 4 if self.direction == "symmetric" else 2
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=storage,
            apply_workspace_bytes_per_rhs=multiplier * self.space.size * itemsize,
            reason="prepared level-scheduled Gauss-Seidel sweep",
        )


class GaussSeidelPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Build a reusable Gauss-Seidel smoother from explicit operator storage."""

    direction: GaussSeidelDirection = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        direction: GaussSeidelDirection = "symmetric",
        relaxation: float = 1.0,
    ):
        if direction not in ("forward", "backward", "symmetric"):
            raise ValueError(f"Unknown Gauss-Seidel direction {direction!r}.")
        omega = float(relaxation)
        if not isfinite(omega) or omega <= 0.0 or omega >= 2.0:
            raise ValueError("Gauss-Seidel relaxation must lie strictly between 0 and 2.")
        self.direction = direction
        self.relaxation = omega
        self._builder_id = canonical_fingerprint(
            {
                "kind": "gauss-seidel-builder",
                "direction": direction,
                "relaxation": omega,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if not isinstance(setup_operator, AbstractLinearOperator):
            raise TypeError("setup_operator must be an AbstractLinearOperator.")
        if setup_operator.batch_shape or not setup_operator.source.compatible(
            setup_operator.target
        ):
            raise ValueError("Gauss-Seidel requires an unbatched endomorphism.")
        symmetric = self.direction == "symmetric" and setup_operator.properties.certifies(
            "self_adjoint"
        )
        positive = symmetric and setup_operator.properties.certifies("positive_definite")
        evidence = {
            "linear": "construction",
            "stationary": "construction",
            **({"self_adjoint": "transformed"} if symmetric else {}),
            **({"positive_definite": "transformed"} if positive else {}),
        }
        return PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=symmetric,
            positive_definite=positive,
            evidence=evidence,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        if not isinstance(
            setup_operator,
            (DenseLinearOperator, AbstractSparseLinearOperator),
        ):
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason="Gauss-Seidel requires explicit dense or sparse operator storage",
            )
        matrix = _explicit_csr(setup_operator)
        factors = 2 if self.direction == "symmetric" else 1
        itemsize = matrix.data.dtype.itemsize
        index_size = matrix.indices.dtype.itemsize
        storage = factors * (
            matrix.nnz * (itemsize + index_size)
            + (matrix.shape[0] + 1) * matrix.indptr.dtype.itemsize
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=int(storage),
            preparation_workspace_bytes=int(storage),
            apply_workspace_bytes_per_rhs=int(
                (4 if self.direction == "symmetric" else 2) * matrix.shape[0] * itemsize
            ),
            accepted=True,
            reason="explicit triangular sweep preparation",
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> GaussSeidelPreconditioner:
        del materialization
        return GaussSeidelPreconditioner(
            setup_operator,
            direction=self.direction,
            relaxation=self.relaxation,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> GaussSeidelPreconditioner:
        del materialization
        if not isinstance(preconditioner, GaussSeidelPreconditioner):
            raise TypeError("Gauss-Seidel refresh requires GaussSeidelPreconditioner.")
        if preconditioner.direction != self.direction:
            raise ValueError("Gauss-Seidel refresh cannot change sweep direction.")
        return GaussSeidelPreconditioner(
            setup_operator,
            direction=self.direction,
            relaxation=self.relaxation,
            previous=preconditioner,
        )


__all__ = [
    "GaussSeidelDirection",
    "GaussSeidelPreconditioner",
    "GaussSeidelPreconditionerBuilder",
]
