#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, ClassVar, final, Literal, Protocol, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
)
from ..linalg._operators import _validate_properties
from ..linalg._spaces import _coordinate_dtype
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._ops import (
    block_linear_adjoint_apply,
    block_linear_apply,
    block_linear_transpose_apply,
    linear_adjoint_apply,
    linear_apply,
    linear_transpose_apply,
)
from ._relation import EdgeRelation, RowRelation, SparseRelation


if TYPE_CHECKING:
    import scipy.sparse as sp


class LinearAction(Protocol):
    """Minimal structured linear action accepted by matrix-free consumers."""

    @property
    def input_shape(self) -> tuple[int, ...]: ...

    @property
    def output_shape(self) -> tuple[int, ...]: ...

    def mv(self, vector: Any, /) -> Any: ...

    def transpose_mv(self, vector: Any, /) -> Any: ...

    def adjoint_mv(self, vector: Any, /) -> Any: ...


@final
class SparseLinearMap(AbstractSparseLinearOperator):
    """Scalar-coefficient linear action over one immutable sparse relation."""

    _fused_block_action_kind: ClassVar[Literal["fused"]] = "fused"

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
    ) -> None:
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("relation must be an EdgeRelation or RowRelation.")
        values = jnp.asarray(coefficients)
        route_ndim = len(relation.route_shape)
        if (
            values.ndim < route_ndim
            or tuple(values.shape[-route_ndim:]) != relation.route_shape
        ):
            raise ValueError(
                f"Sparse coefficients must end in route shape {relation.route_shape}; got {values.shape}."
            )
        batch = tuple(values.shape[:-route_ndim])
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(jnp.float64)
        source = ArraySpace(relation.input_shape, dtype=values.dtype)
        target = ArraySpace(relation.output_shape, dtype=values.dtype)
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        _validate_properties(properties_, source, target)
        identifier = (
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
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.relation = relation
        self.coefficients = values
        self.source = source
        self.target = target
        self.properties = properties_
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=source.compatible(target),
        )
        self.batch_shape = batch
        self.operator_id = identifier

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

        def materialize_one(values: Array) -> Array:
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

    def to_scipy(self) -> sp.csr_matrix:
        """Return a host-side CSR matrix, coalescing duplicate linear routes."""
        import scipy.sparse as sp

        if self.batch_shape:
            raise ValueError("to_scipy requires an unbatched sparse operator.")

        relation, coefficients = self._edge_form()
        valid = np.asarray(relation.valid, dtype=np.bool_)
        source = np.asarray(relation.source_indices)[valid]
        target = np.asarray(relation.target_indices)[valid]
        values = np.asarray(coefficients)[valid]
        return sp.coo_matrix(
            (values, (target, source)),
            shape=(relation.target_size, relation.source_size),
        ).tocsr()


@final
class SparseCoordinateOperator(AbstractSparseLinearOperator):
    """Sparse canonical-coordinate map, optionally with explicit matrix fibers.

    ``block_shape=(r_t, r_s)`` admits coefficients shaped
    ``(*relation.route_shape, r_t, r_s)``. Canonical coordinates are ordered
    cell-major, fiber-minor; the source and target sizes include their fibers.
    Without ``block_shape`` coefficients remain scalar and unbatched.
    """

    relation: SparseRelation
    coefficients: Array
    accumulation_dtype: np.dtype = eqx.field(static=True)
    block_shape: tuple[int, int] | None = eqx.field(static=True)
    _storage_plan: _SparseStoragePlan | None

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
        accumulation_dtype: DTypeLike | None = None,
        block_shape: tuple[int, int] | None = None,
        storage_plan: _SparseStoragePlan | None = None,
    ) -> None:
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
        block_shape = _validated_block_shape(block_shape)
        target_fiber, source_fiber = (1, 1) if block_shape is None else block_shape
        if (
            source.size != edge_relation.source_size * source_fiber
            or target.size != edge_relation.target_size * target_fiber
        ):
            raise ValueError(
                "Vector-space sizes must match sparse relation sizes including fibers."
            )
        values = jnp.asarray(coefficients)
        coefficient_shape = relation.route_shape + (
            () if block_shape is None else block_shape
        )
        if values.shape != coefficient_shape:
            raise ValueError(
                f"Sparse coefficients must have shape {coefficient_shape}; got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(jnp.float64)
        accumulation_dtype_ = jnp.dtype(
            jnp.result_type(
                values.dtype, _coordinate_dtype(source), _coordinate_dtype(target)
            )
            if accumulation_dtype is None
            else accumulation_dtype
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
                    **({} if block_shape is None else {"block_shape": block_shape}),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        if storage_plan is not None:
            if not isinstance(storage_plan, _SparseStoragePlan):
                raise TypeError("storage_plan must be a native sparse storage plan.")
            if (
                storage_plan.block_shape != block_shape
                or storage_plan.route_shape != coefficient_shape
                or storage_plan.relation_sizes
                != (edge_relation.target_size, edge_relation.source_size)
            ):
                raise ValueError(
                    "Sparse storage plan must match the operator route and fiber layout."
                )
        self.relation = relation
        self.coefficients = values
        self.accumulation_dtype = accumulation_dtype_
        self.block_shape = block_shape
        self._storage_plan = storage_plan
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
        if self.block_shape is not None:
            output = block_linear_apply(
                self.relation,
                self.coefficients.astype(self.accumulation_dtype),
                coordinates.reshape(self.relation.input_shape + (self.block_shape[1],)),
            )
            return self.target.unflatten(
                output.reshape((-1,)).astype(_coordinate_dtype(self.target))
            )
        relation, coefficients = self._edge_form()
        output = linear_apply(
            relation,
            coefficients.astype(self.accumulation_dtype),
            coordinates,
        )
        return self.target.unflatten(output.astype(_coordinate_dtype(self.target)))

    def transpose_mv(self, vector: Any, /) -> Any:
        coordinates = self.target.flatten(vector).astype(self.accumulation_dtype)
        if self.block_shape is not None:
            output = block_linear_transpose_apply(
                self.relation,
                self.coefficients.astype(self.accumulation_dtype),
                coordinates.reshape(self.relation.output_shape + (self.block_shape[0],)),
            )
            return self.source.unflatten(
                output.reshape((-1,)).astype(_coordinate_dtype(self.source))
            )
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
        if self.block_shape is not None:
            output = block_linear_adjoint_apply(
                self.relation,
                self.coefficients.astype(self.accumulation_dtype),
                target_covector.reshape(
                    self.relation.output_shape + (self.block_shape[0],)
                ),
            )
            source_covector = self.source.unflatten(
                output.reshape((-1,)).astype(_coordinate_dtype(self.source))
            )
            return self.source.inverse_riesz(source_covector)
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
        if self.block_shape is not None:
            edge = (
                self.relation
                if isinstance(self.relation, EdgeRelation)
                else self.relation.as_edge_relation()
            )
            return _block_edge_relation(
                edge, self.block_shape
            ), self.coefficients.reshape((-1,))
        if isinstance(self.relation, EdgeRelation):
            return self.relation, self.coefficients
        return self.relation.as_edge_relation(), self.coefficients.reshape((-1,))

    def as_dense(self) -> Array:
        relation, coefficients = self._edge_form()
        coefficients = coefficients.astype(self.accumulation_dtype)
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
        return (
            matrix.at[safe_target, safe_source]
            .add(values)
            .astype(_coordinate_dtype(self.target))
        )

    def _assemble_diagonal(self, /) -> Array:
        relation, coefficients = self._edge_form()
        return _assemble_relation_diagonal(
            relation, coefficients.astype(self.accumulation_dtype)
        ).astype(_coordinate_dtype(self.target))

    def _materialize(self, /) -> Array:
        return self.as_dense()

    def sparse_storage(self, /) -> SparseStorage:
        if self._storage_plan is not None:
            return self._storage_plan.apply(self.coefficients, relation=self.relation)
        return _canonical_sparse_storage(
            self.relation, self.coefficients, block_shape=self.block_shape
        )


def _assemble_relation_diagonal(
    relation: EdgeRelation,
    coefficients: Array,
    /,
) -> Array:
    diagonal_entry = relation.valid & (relation.source_indices == relation.target_indices)
    safe_target = jnp.where(diagonal_entry, relation.target_indices, 0)

    def assemble_one(values: Array) -> Array:
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


def _validated_block_shape(
    block_shape: tuple[int, int] | None, /
) -> tuple[int, int] | None:
    if block_shape is not None and (
        not isinstance(block_shape, tuple)
        or len(block_shape) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in block_shape
        )
    ):
        raise ValueError(
            "block_shape must contain positive target and source fiber sizes."
        )
    return block_shape


def _block_edge_relation(
    relation: EdgeRelation, block_shape: tuple[int, int], /
) -> EdgeRelation:
    """Expand matrix-fiber entries in route/target-fiber/source-fiber order."""
    target_fiber, source_fiber = block_shape
    shape = relation.route_shape + block_shape
    source = (
        jnp.where(relation.valid, relation.source_indices, 0)[:, None, None]
        * source_fiber
        + jnp.arange(source_fiber, dtype=relation.source_indices.dtype)[None, None, :]
    )
    target = (
        jnp.where(relation.valid, relation.target_indices, 0)[:, None, None]
        * target_fiber
        + jnp.arange(target_fiber, dtype=relation.target_indices.dtype)[None, :, None]
    )
    return EdgeRelation(
        jnp.broadcast_to(source, shape).reshape((-1,)),
        jnp.broadcast_to(target, shape).reshape((-1,)),
        source_size=relation.source_size * source_fiber,
        target_size=relation.target_size * target_fiber,
        valid=jnp.broadcast_to(relation.valid[:, None, None], shape).reshape((-1,)),
    )


@final
class _SparseStoragePlan(StrictModule):
    """Host-planned route coalescing, reusable inside numeric JAX refresh."""

    positions: Array
    groups: Array
    indices: Array
    indptr: Array
    source_indices: Array
    target_indices: Array
    valid: Array
    shape: tuple[int, int] = eqx.field(static=True)
    route_shape: tuple[int, ...] = eqx.field(static=True)
    relation_shape: tuple[int, ...] = eqx.field(static=True)
    relation_sizes: tuple[int, int] = eqx.field(static=True)
    block_shape: tuple[int, int] | None = eqx.field(static=True)
    nnz: int = eqx.field(static=True)

    def __init__(
        self,
        relation: SparseRelation,
        /,
        *,
        block_shape: tuple[int, int] | None = None,
    ) -> None:
        block_shape = _validated_block_shape(block_shape)
        edge = (
            relation
            if isinstance(relation, EdgeRelation)
            else relation.as_edge_relation()
        )
        valid = np.asarray(edge.valid, dtype=np.bool_).reshape(-1)
        source = np.asarray(edge.source_indices, dtype=np.int64).reshape(-1)
        target = np.asarray(edge.target_indices, dtype=np.int64).reshape(-1)
        target_fiber, source_fiber = (1, 1) if block_shape is None else block_shape
        if block_shape is not None:
            block_route_shape = (edge.capacity,) + block_shape
            source = np.broadcast_to(
                source[:, None, None] * source_fiber
                + np.arange(source_fiber, dtype=np.int64)[None, None, :],
                block_route_shape,
            ).reshape(-1)
            target = np.broadcast_to(
                target[:, None, None] * target_fiber
                + np.arange(target_fiber, dtype=np.int64)[None, :, None],
                block_route_shape,
            ).reshape(-1)
            valid = np.broadcast_to(valid[:, None, None], block_route_shape).reshape(-1)
        positions = np.flatnonzero(valid)
        source, target = source[valid], target[valid]
        shape = (edge.target_size * target_fiber, edge.source_size * source_fiber)
        order = np.lexsort((source, target))
        source, target, positions = source[order], target[order], positions[order]
        if positions.size:
            starts = np.concatenate(
                (
                    np.asarray([True]),
                    (source[1:] != source[:-1]) | (target[1:] != target[:-1]),
                )
            )
            groups = np.cumsum(starts, dtype=np.int64) - 1
            canonical_source, canonical_target = source[starts], target[starts]
            number_groups = int(groups[-1]) + 1
        else:
            groups = np.zeros((0,), dtype=np.int64)
            canonical_source, canonical_target = source, target
            number_groups = 0
        largest = max(*shape, number_groups)
        index_dtype = jnp.int32 if largest < np.iinfo(np.int32).max else jnp.int64
        counts = np.bincount(canonical_target, minlength=shape[0])
        # Host pattern data stays concrete even when planned under a trace, so
        # pattern validation can read it while coefficients remain traced.
        with jax.ensure_compile_time_eval():
            self.positions = jnp.asarray(positions)
            self.groups = jnp.asarray(groups)
            self.indices = jnp.asarray(canonical_source, dtype=index_dtype)
            self.indptr = jnp.asarray(
                np.concatenate((np.asarray([0]), np.cumsum(counts))),
                dtype=index_dtype,
            )
        self.source_indices = edge.source_indices
        self.target_indices = edge.target_indices
        self.valid = edge.valid
        self.shape = shape
        self.route_shape = relation.route_shape + (
            () if block_shape is None else block_shape
        )
        self.relation_shape = relation.route_shape
        self.relation_sizes = (edge.target_size, edge.source_size)
        self.block_shape = block_shape
        self.nnz = number_groups

    def apply(
        self, coefficients: Array, /, *, relation: SparseRelation | None = None
    ) -> SparseStorage:
        if tuple(coefficients.shape[-len(self.route_shape) :]) != self.route_shape:
            raise ValueError("Sparse storage refresh requires unchanged route shape.")
        if relation is not None:
            edge = (
                relation
                if isinstance(relation, EdgeRelation)
                else relation.as_edge_relation()
            )
            if (
                relation.route_shape != self.relation_shape
                or (edge.target_size, edge.source_size) != self.relation_sizes
            ):
                raise ValueError("Sparse storage refresh requires unchanged topology.")
            changed = (
                jnp.any(edge.valid != self.valid)
                | jnp.any(
                    jnp.where(
                        self.valid, edge.source_indices != self.source_indices, False
                    )
                )
                | jnp.any(
                    jnp.where(
                        self.valid, edge.target_indices != self.target_indices, False
                    )
                )
            )
            message = "Sparse storage refresh requires unchanged relation routes."
            if isinstance(changed, jax_core.Tracer):
                coefficients = eqx.error_if(coefficients, changed, message)
            elif bool(changed):
                raise ValueError(message)
        batch_shape = coefficients.shape[: -len(self.route_shape)]
        values = jnp.take(
            coefficients.reshape(batch_shape + (-1,)), self.positions, axis=-1
        )
        canonical = jnp.zeros(batch_shape + (self.nnz,), dtype=coefficients.dtype)
        canonical = canonical.at[..., self.groups].add(values)
        return SparseStorage(canonical, self.indices, self.indptr, shape=self.shape)


def _canonical_sparse_storage(
    relation: SparseRelation,
    coefficients: Array,
    /,
    *,
    block_shape: tuple[int, int] | None = None,
) -> SparseStorage:
    return _SparseStoragePlan(relation, block_shape=block_shape).apply(coefficients)


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
