#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
)
from ..linalg._operators import _validate_properties
from ..linalg._spaces import _coordinate_dtype
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
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


class SparseLinearMap(AbstractSparseLinearOperator):
    """Scalar-coefficient linear action over one immutable sparse relation."""

    relation: SparseRelation
    coefficients: Array

    def __init__(
        self,
        relation: SparseRelation,
        coefficients: ArrayLike,
        /,
        *,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        values = jnp.asarray(coefficients)
        route_ndim = len(relation.route_shape)
        if (
            values.ndim < route_ndim
            or tuple(values.shape[-route_ndim:]) != relation.route_shape
        ):
            raise ValueError(
                f"Sparse coefficients must end in route shape {relation.route_shape}; "
                f"got {values.shape}."
            )
        batch = tuple(int(size) for size in values.shape[:-route_ndim])
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        self.relation = relation
        self.coefficients = values
        source = ArraySpace(relation.input_shape, dtype=values.dtype)
        target = ArraySpace(relation.output_shape, dtype=values.dtype)
        self.source = source
        self.target = target
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, source, target)
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=source.compatible(target),
        )
        self.batch_shape = batch
        self.operator_id = (
            canonical_fingerprint(
                {
                    "kind": "sparse-linear-map",
                    "source": source.space_id,
                    "target": target.space_id,
                    "relation": _relation_payload(relation),
                    "properties": _properties_payload(properties_),
                    "batch_shape": list(batch),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty.")

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

    @property
    def supports_fused_block_action(self) -> bool:
        return True

    def mv(self, vector: Any, /) -> Any:
        """Apply the sparse map while preserving trailing payload dimensions."""
        return linear_apply(self.relation, self.coefficients, vector)

    def transpose_mv(self, vector: Any, /) -> Any:
        """Apply the algebraic transpose without conjugating coefficients."""
        return linear_transpose_apply(self.relation, self.coefficients, vector)

    def adjoint_mv(self, vector: Any, /) -> Any:
        """Apply the conjugate adjoint."""
        return linear_adjoint_apply(self.relation, self.coefficients, vector)

    def _edge_form(self) -> tuple[EdgeRelation, Array]:
        if isinstance(self.relation, EdgeRelation):
            return self.relation, self.coefficients
        return (
            self.relation.as_edge_relation(),
            self.coefficients.reshape(self.batch_shape + (-1,)),
        )

    def as_dense(self) -> Array:
        """Materialize a dense matrix for every shared-pattern batch member."""
        relation, coefficients = self._edge_form()
        safe_source = jnp.where(relation.valid, relation.source_indices, 0)
        safe_target = jnp.where(relation.valid, relation.target_indices, 0)

        def materialize_one(values):
            values = jnp.where(
                relation.valid,
                values,
                jnp.zeros((), dtype=values.dtype),
            )
            return (
                jnp.zeros(
                    (relation.target_size, relation.source_size),
                    dtype=values.dtype,
                )
                .at[safe_target, safe_source]
                .add(values)
            )

        if not self.batch_shape:
            return materialize_one(coefficients)
        flattened = coefficients.reshape((-1,) + relation.route_shape)
        return jax.vmap(materialize_one)(flattened).reshape(
            self.batch_shape + (relation.target_size, relation.source_size)
        )

    def _assemble_diagonal(self, /) -> Array:
        relation, coefficients = self._edge_form()
        return _assemble_relation_diagonal(relation, coefficients)

    def _materialize(self, /) -> Array:
        return self.as_dense()

    def sparse_storage(self, /) -> SparseStorage:
        return _canonical_sparse_storage(self.relation, self.coefficients)

    def to_scipy(self):
        """Return a host-side CSR matrix, coalescing duplicate linear routes."""
        import scipy.sparse as sp

        if self.batch_shape:
            raise ValueError("to_scipy requires an unbatched sparse operator.")

        relation, coefficients = self._edge_form()
        valid = np.asarray(relation.valid, dtype=bool)
        source = np.asarray(relation.source_indices)[valid]
        target = np.asarray(relation.target_indices)[valid]
        values = np.asarray(coefficients)[valid]
        return sp.coo_matrix(
            (values, (target, source)),
            shape=(relation.target_size, relation.source_size),
        ).tocsr()


class SparseCoordinateOperator(AbstractSparseLinearOperator):
    """Sparse canonical-coordinate map between structured vector spaces."""

    relation: SparseRelation
    coefficients: Array
    accumulation_dtype: Any = eqx.field(static=True)

    def __init__(
        self,
        relation: SparseRelation,
        coefficients: ArrayLike,
        /,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        properties: OperatorProperties | None = None,
        operator_id: str | None = None,
        accumulation_dtype: Any | None = None,
    ):
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        edge_relation = (
            relation
            if isinstance(relation, EdgeRelation)
            else relation.as_edge_relation()
        )
        if (
            source.size != edge_relation.source_size
            or target.size != edge_relation.target_size
        ):
            raise ValueError(
                "Vector-space sizes must match the sparse relation source and target."
            )
        values = jnp.asarray(coefficients)
        if values.shape != relation.route_shape:
            raise ValueError(
                f"Sparse coefficients must have route shape {relation.route_shape}; "
                f"got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        accumulation_dtype_ = jnp.dtype(
            values.dtype if accumulation_dtype is None else accumulation_dtype
        )
        if not jnp.issubdtype(accumulation_dtype_, jnp.inexact):
            raise TypeError("Sparse accumulation dtype must be inexact.")
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, source, target)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "sparse-coordinate-operator",
                    "source": source.space_id,
                    "target": target.space_id,
                    "relation": _relation_payload(relation),
                    "properties": _properties_payload(properties_),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.relation = relation
        self.coefficients = values
        self.accumulation_dtype = accumulation_dtype_
        self.source = source
        self.target = target
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=source.compatible(target),
        )
        self.batch_shape = ()
        self.operator_id = identifier

    def mv(self, vector: Any, /) -> Any:
        coordinates = self.source.flatten(vector).astype(self.accumulation_dtype)
        relation, coefficients = self._edge_form()
        output = linear_apply(
            relation,
            coefficients.astype(self.accumulation_dtype),
            coordinates,
        )
        return self.target.unflatten(output.astype(_coordinate_dtype(self.target)))

    def transpose_mv(self, vector: Any, /) -> Any:
        coordinates = self.target.flatten(vector).astype(self.accumulation_dtype)
        relation, coefficients = self._edge_form()
        output = linear_transpose_apply(
            relation,
            coefficients.astype(self.accumulation_dtype),
            coordinates,
        )
        return self.source.unflatten(output.astype(_coordinate_dtype(self.source)))

    def adjoint_mv(self, vector: Any, /) -> Any:
        target_covector = self.target.flatten(self.target.riesz(vector)).astype(
            self.accumulation_dtype
        )
        relation, coefficients = self._edge_form()
        source_covector = self.source.unflatten(
            linear_adjoint_apply(
                relation,
                coefficients.astype(self.accumulation_dtype),
                target_covector,
            ).astype(_coordinate_dtype(self.source))
        )
        return self.source.inverse_riesz(source_covector)

    def _edge_form(self) -> tuple[EdgeRelation, Array]:
        if isinstance(self.relation, EdgeRelation):
            return self.relation, self.coefficients
        return self.relation.as_edge_relation(), self.coefficients.reshape((-1,))

    def as_dense(self) -> Array:
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

    def _assemble_diagonal(self, /) -> Array:
        relation, coefficients = self._edge_form()
        return _assemble_relation_diagonal(relation, coefficients)

    def _materialize(self, /) -> Array:
        return self.as_dense()

    def sparse_storage(self, /) -> SparseStorage:
        return _canonical_sparse_storage(self.relation, self.coefficients)


def _assemble_relation_diagonal(
    relation: EdgeRelation,
    coefficients: Array,
    /,
) -> Array:
    diagonal_entry = relation.valid & (relation.source_indices == relation.target_indices)
    safe_target = jnp.where(diagonal_entry, relation.target_indices, 0)

    def assemble_one(values):
        values = jnp.where(
            diagonal_entry,
            values,
            jnp.zeros((), dtype=values.dtype),
        )
        return (
            jnp.zeros((relation.target_size,), dtype=values.dtype)
            .at[safe_target]
            .add(values)
        )

    batch_shape = coefficients.shape[: -len(relation.route_shape)]
    if not batch_shape:
        return assemble_one(coefficients)
    flattened = coefficients.reshape((-1,) + relation.route_shape)
    return jax.vmap(assemble_one)(flattened).reshape(
        batch_shape + (relation.target_size,)
    )


def _canonical_sparse_storage(
    relation: SparseRelation,
    coefficients: Array,
    /,
) -> SparseStorage:
    edge_relation = (
        relation if isinstance(relation, EdgeRelation) else relation.as_edge_relation()
    )
    valid = np.asarray(edge_relation.valid, dtype=bool).reshape((-1,))
    source = np.asarray(edge_relation.source_indices).reshape((-1,))[valid]
    target = np.asarray(edge_relation.target_indices).reshape((-1,))[valid]
    positions = np.flatnonzero(valid)
    order = np.lexsort((source, target))
    source = source[order]
    target = target[order]
    positions = positions[order]
    if positions.size:
        starts = np.concatenate(
            (
                np.asarray([True]),
                (source[1:] != source[:-1]) | (target[1:] != target[:-1]),
            )
        )
        groups = np.cumsum(starts, dtype=np.int64) - 1
        canonical_source = source[starts]
        canonical_target = target[starts]
        number_groups = int(groups[-1]) + 1
    else:
        groups = np.zeros((0,), dtype=np.int64)
        canonical_source = source
        canonical_target = target
        number_groups = 0
    largest = max(
        edge_relation.source_size,
        edge_relation.target_size,
        number_groups,
    )
    index_dtype = jnp.int32 if largest < np.iinfo(np.int32).max else jnp.int64
    batch_shape = coefficients.shape[: -len(relation.route_shape)]
    flattened_coefficients = coefficients.reshape(batch_shape + (-1,))
    route_values = jnp.take(
        flattened_coefficients,
        jnp.asarray(positions),
        axis=-1,
    )
    values = (
        jnp.zeros(batch_shape + (number_groups,), dtype=coefficients.dtype)
        .at[..., jnp.asarray(groups)]
        .add(route_values)
    )
    counts = np.bincount(
        canonical_target,
        minlength=edge_relation.target_size,
    )
    indptr = np.concatenate((np.asarray([0]), np.cumsum(counts)))
    return SparseStorage(
        values,
        jnp.asarray(canonical_source, dtype=index_dtype),
        jnp.asarray(indptr, dtype=index_dtype),
        shape=(edge_relation.target_size, edge_relation.source_size),
        sorted_indices=True,
        canonical=True,
    )


def _relation_payload(relation: SparseRelation, /) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": type(relation).__name__,
        "source_indices": np.asarray(relation.source_indices).tolist(),
        "valid": np.asarray(relation.valid).tolist(),
        "source_size": relation.source_size,
    }
    if isinstance(relation, EdgeRelation):
        payload["target_size"] = relation.target_size
        payload["target_indices"] = np.asarray(relation.target_indices).tolist()
    else:
        payload["target_shape"] = list(relation.target_shape)
        payload["case_shape"] = list(relation.case_shape)
    return payload


def _properties_payload(properties: OperatorProperties, /) -> dict[str, object]:
    return {
        "diagonal": properties.diagonal,
        "triangular": properties.triangular,
        "self_adjoint": properties.self_adjoint,
        "positive_definite": properties.positive_definite,
        "positive_semidefinite": properties.positive_semidefinite,
        "block_diagonal": properties.block_diagonal,
        "rank": properties.rank,
        "evidence": properties.evidence,
    }


__all__ = ["LinearAction", "SparseCoordinateOperator", "SparseLinearMap"]
