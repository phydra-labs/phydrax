#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._ops import linear_adjoint_apply, linear_apply, linear_transpose_apply
from ._relation import EdgeRelation, RowRelation, SparseRelation


class LinearAction(Protocol):
    """Minimal structured linear action accepted by matrix-free consumers."""

    @property
    def input_shape(self) -> tuple[int, ...]: ...

    @property
    def output_shape(self) -> tuple[int, ...]: ...

    def mv(self, vector: Any, /) -> Any: ...

    def transpose_mv(self, vector: Any, /) -> Any: ...

    def adjoint_mv(self, vector: Any, /) -> Any: ...


class SparseLinearMap(StrictModule):
    """Scalar-coefficient linear action over one immutable sparse relation."""

    relation: SparseRelation
    coefficients: Array

    def __init__(
        self,
        relation: SparseRelation,
        coefficients: ArrayLike,
        /,
    ):
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        values = jnp.asarray(coefficients)
        if values.shape != relation.route_shape:
            raise ValueError(
                f"Sparse coefficients must have route shape {relation.route_shape}; "
                f"got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        self.relation = relation
        self.coefficients = values

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.relation.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.relation.output_shape

    @property
    def input_size(self) -> int:
        return prod(self.input_shape) if self.input_shape else 1

    @property
    def output_size(self) -> int:
        return prod(self.output_shape) if self.output_shape else 1

    def mv(self, vector: Any, /) -> Any:
        """Apply the sparse map while preserving trailing payload dimensions."""
        return linear_apply(self.relation, self.coefficients, vector)

    def transpose_mv(self, vector: Any, /) -> Any:
        """Apply the algebraic transpose without conjugating coefficients."""
        return linear_transpose_apply(self.relation, self.coefficients, vector)

    def adjoint_mv(self, vector: Any, /) -> Any:
        """Apply the conjugate adjoint."""
        return linear_adjoint_apply(self.relation, self.coefficients, vector)

    def __call__(self, vector: Any, /) -> Any:
        return self.mv(vector)

    def _edge_form(self) -> tuple[EdgeRelation, Array]:
        if isinstance(self.relation, EdgeRelation):
            return self.relation, self.coefficients
        return self.relation.as_edge_relation(), self.coefficients.reshape((-1,))

    def as_dense(self) -> Array:
        """Materialize a two-dimensional matrix over flattened source and target spaces."""
        relation, coefficients = self._edge_form()
        safe_source = jnp.where(relation.valid, relation.source_indices, 0)
        safe_target = jnp.where(relation.valid, relation.target_indices, 0)
        values = jnp.where(
            relation.valid,
            coefficients,
            jnp.zeros((), dtype=coefficients.dtype),
        )
        matrix = jnp.zeros(
            (relation.target_size, relation.source_size),
            dtype=coefficients.dtype,
        )
        return matrix.at[safe_target, safe_source].add(values)

    def to_scipy(self):
        """Return a host-side CSR matrix, coalescing duplicate linear routes."""
        import scipy.sparse as sp

        relation, coefficients = self._edge_form()
        valid = np.asarray(relation.valid, dtype=bool)
        source = np.asarray(relation.source_indices)[valid]
        target = np.asarray(relation.target_indices)[valid]
        values = np.asarray(coefficients)[valid]
        return sp.coo_matrix(
            (values, (target, source)),
            shape=(relation.target_size, relation.source_size),
        ).tocsr()


__all__ = ["LinearAction", "SparseLinearMap"]
