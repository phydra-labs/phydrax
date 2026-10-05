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

from .. import ein
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractVectorSpace,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
)
from ..linalg._costs import _array_tree_storage_bytes
from ..linalg._operators import _validate_properties
from ..linalg._spaces import _coordinate_dtype
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._ops import (
    block_linear_apply,
    block_linear_transpose_apply,
    linear_adjoint_apply,
    linear_apply,
    linear_transpose_apply,
)
from ._relation import (
    _coalesced_route_count,
    _relation_traced,
    EdgeRelation,
    RowRelation,
    SparseRelation,
)


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
    _canonical_nnz: int | None = eqx.field(static=True)

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
        values = _coefficient_values(coefficients)
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
        self._canonical_nnz = _coalesced_route_count(relation)
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

    def _resident_storage_bytes(self, /) -> int:
        return _array_tree_storage_bytes(self) + _canonical_storage_bytes(
            self._canonical_nnz,
            self.relation,
            self.coefficients.dtype,
            self.batch_shape,
            (1, 1),
            indices_resident=False,
        )

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

    Host-prepared topology also fixes one output-major row-gather layout per
    apply direction (by target for ``mv``, by source for the transpose and
    adjoint). Each output then accumulates its routes sequentially in route
    order. A direction whose widest output would pad beyond twice its routes
    plus outputs, or a traced relation, keeps the route scatter. Routes are
    construction-time structure: coefficients may be replaced in place, but new
    routes require a new operator.
    """

    relation: SparseRelation
    coefficients: Array
    accumulation_dtype: np.dtype = eqx.field(static=True)
    block_shape: tuple[int, int] | None = eqx.field(static=True)
    _storage_plan: _SparseStoragePlan | None
    _canonical_nnz: int | None = eqx.field(static=True)
    _target_gather: _RowGatherLayout | None
    _source_gather: _RowGatherLayout | None

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
        values = _coefficient_values(coefficients)
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
        if storage_plan is not None:
            self._canonical_nnz = storage_plan.nnz
        else:
            route_count = _coalesced_route_count(relation)
            self._canonical_nnz = (
                None if route_count is None else route_count * target_fiber * source_fiber
            )
        self._target_gather, self._source_gather = (
            (None, None)
            if _relation_traced(edge_relation)
            else _row_gather_layouts(edge_relation)
        )
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

    def _coordinate_apply(
        self, coordinates: Array, /, *, reverse: bool, conjugate: bool
    ) -> Array:
        """Apply to flat accumulation-dtype coordinates, forward or reversed."""
        coefficients = self.coefficients.astype(self.accumulation_dtype)
        if conjugate:
            coefficients = jnp.conj(coefficients)
        layout = self._source_gather if reverse else self._target_gather
        fibers = self.block_shape
        if layout is not None:
            if fibers is None:
                return layout.apply(
                    coefficients.reshape((-1,)), coordinates, reverse=reverse
                )
            cells = coordinates.reshape((-1, fibers[0] if reverse else fibers[1]))
            return layout.apply(
                coefficients.reshape((-1,) + fibers), cells, reverse=reverse
            ).reshape((-1,))
        if fibers is not None:
            if reverse:
                return block_linear_transpose_apply(
                    self.relation,
                    coefficients,
                    coordinates.reshape(self.relation.output_shape + (fibers[0],)),
                ).reshape((-1,))
            return block_linear_apply(
                self.relation,
                coefficients,
                coordinates.reshape(self.relation.input_shape + (fibers[1],)),
            ).reshape((-1,))
        relation = (
            self.relation
            if isinstance(self.relation, EdgeRelation)
            else self.relation.as_edge_relation()
        )
        apply = linear_transpose_apply if reverse else linear_apply
        return apply(relation, coefficients.reshape((-1,)), coordinates)

    def mv(self, vector: Any, /) -> Any:
        coordinates = self.source.flatten(vector).astype(self.accumulation_dtype)
        output = self._coordinate_apply(coordinates, reverse=False, conjugate=False)
        return self.target.unflatten(output.astype(_coordinate_dtype(self.target)))

    def transpose_mv(self, vector: Any, /) -> Any:
        coordinates = self.target.flatten(vector).astype(self.accumulation_dtype)
        output = self._coordinate_apply(coordinates, reverse=True, conjugate=False)
        return self.source.unflatten(output.astype(_coordinate_dtype(self.source)))

    def adjoint_mv(self, vector: Any, /) -> Any:
        target_covector = self.target.flatten(self.target.riesz(vector)).astype(
            self.accumulation_dtype
        )
        output = self._coordinate_apply(target_covector, reverse=True, conjugate=True)
        source_covector = self.source.unflatten(
            output.astype(_coordinate_dtype(self.source))
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

    def _resident_storage_bytes(self, /) -> int:
        return _array_tree_storage_bytes(self) + _canonical_storage_bytes(
            self._canonical_nnz,
            self.relation,
            self.coefficients.dtype,
            (),
            (1, 1) if self.block_shape is None else self.block_shape,
            indices_resident=self._storage_plan is not None,
        )


def _canonical_storage_bytes(
    nnz: int | None,
    relation: SparseRelation,
    value_dtype: np.dtype,
    batch_shape: tuple[int, ...],
    fibers: tuple[int, int],
    /,
    *,
    indices_resident: bool,
) -> int:
    """Bytes of the canonical CSR arrays ``sparse_storage`` adds to the operator.

    Uses the entry count read from host topology at construction, so traced
    executions never synchronize topology. Values are always new; a prepared
    storage plan already holds the shared indices and row pointers.
    """
    if nnz is None:
        raise ValueError(
            "Sparse cost estimation requires host-prepared topology; this operator "
            "was constructed from a traced relation without a storage plan."
        )
    values = prod(batch_shape) * nnz * value_dtype.itemsize
    if indices_resident:
        return values
    shape = (
        prod(relation.output_shape) * fibers[0],
        prod(relation.input_shape) * fibers[1],
    )
    index_dtype = jax.dtypes.canonicalize_dtype(_storage_index_dtype(shape, nnz))
    return values + (nnz + shape[0] + 1) * index_dtype.itemsize


def _storage_index_dtype(shape: tuple[int, int], nnz: int, /) -> np.dtype:
    """Narrowest admitted CSR index dtype for one canonical storage pattern."""
    largest = max(*shape, nnz)
    return np.dtype(np.int32 if largest < np.iinfo(np.int32).max else np.int64)


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


# A fixed-width gather pays for ``width * outputs`` slots where the route
# scatter pays for its routes and the zero-filled outputs; beyond this ratio a
# skewed direction keeps the scatter.
_MAX_ROW_GATHER_PADDING = 2


@final
class _RowGatherLayout(StrictModule):
    """Output-major fixed-width routes of one apply direction (ELL form).

    ``routes[slot, output]`` is the ``slot``-th valid route into ``output`` in
    route order, or ``route_count`` for padding; ``inputs`` holds the input
    cell that route reads (zero for padding).
    """

    routes: Array
    inputs: Array
    route_count: int = eqx.field(static=True)

    def apply(self, coefficients: Array, values: Array, /, *, reverse: bool) -> Array:
        """Sum route-major coefficients times gathered input cells per output.

        Scalar coefficients are ``(routes,)``; matrix fibers are
        ``(routes, r_t, r_s)``, contracted on ``r_s`` forward or ``r_t`` when
        ``reverse``.
        """
        valid = self.routes < self.route_count
        gathered = jnp.take(coefficients, self.routes, axis=0, mode="fill", fill_value=0)
        sources = values[self.inputs]
        if coefficients.ndim == 1:
            products = gathered * sources
        elif reverse:
            products = ein.contract("wnoi,wno->wni", gathered, sources)
        else:
            products = ein.contract("wnoi,wni->wno", gathered, sources)
        products = jnp.where(
            valid.reshape(valid.shape + (1,) * (products.ndim - 2)),
            products,
            jnp.zeros((), dtype=products.dtype),
        )
        if products.shape[0] == 0:
            # No valid routes: the loop body could not index an empty slot axis.
            return jnp.zeros(products.shape[1:], dtype=products.dtype)
        # The loop-carried sum fixes route order per output; unrolled and
        # reduce-based sums were not bitwise route-ordered on CPU. A
        # module-level body keeps eager applies on one cached executable.
        total, _ = jax.lax.fori_loop(
            0,
            products.shape[0],
            _accumulate_slot,
            (jnp.zeros(products.shape[1:], dtype=products.dtype), products),
        )
        return total


def _accumulate_slot(slot: Array, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
    total, products = carry
    return total + products[slot], products


def _row_gather_layout(
    outputs: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    output_size: int,
    /,
) -> _RowGatherLayout | None:
    routes = np.flatnonzero(valid)
    targets = outputs[routes]
    counts = np.bincount(targets, minlength=output_size)
    width = int(counts.max(initial=0))
    if width * output_size > _MAX_ROW_GATHER_PADDING * (routes.size + output_size):
        return None
    if np.any(targets[1:] < targets[:-1]):
        order = np.argsort(targets, kind="stable")
        routes, targets = routes[order], targets[order]
    slots = np.arange(routes.size) - (np.cumsum(counts) - counts)[targets]
    largest = max(valid.size, output_size, int(inputs.max(initial=0)))
    index_dtype = np.int32 if largest < np.iinfo(np.int32).max else np.int64
    gathered_routes = np.full((width, output_size), valid.size, dtype=index_dtype)
    gathered_routes[slots, targets] = routes
    gathered_inputs = np.zeros((width, output_size), dtype=index_dtype)
    gathered_inputs[slots, targets] = inputs[routes]
    with jax.ensure_compile_time_eval():
        return _RowGatherLayout(
            jnp.asarray(gathered_routes),
            jnp.asarray(gathered_inputs),
            route_count=valid.size,
        )


def _row_gather_layouts(
    relation: EdgeRelation, /
) -> tuple[_RowGatherLayout | None, _RowGatherLayout | None]:
    """Target-major (forward) and source-major (reverse) layouts of host routes."""
    source, target, valid = (
        np.asarray(array).reshape(-1)
        for array in jax.device_get(
            (relation.source_indices, relation.target_indices, relation.valid)
        )
    )
    valid = valid.astype(np.bool_)
    source = np.where(valid, source, 0).astype(np.int64)
    target = np.where(valid, target, 0).astype(np.int64)
    return (
        _row_gather_layout(target, source, valid, relation.target_size),
        _row_gather_layout(source, target, valid, relation.source_size),
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
        if _relation_traced(relation):
            raise ValueError(
                "Canonical sparse storage requires host-prepared topology; prepare "
                "the storage plan from concrete relation indices before tracing."
            )
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
        index_dtype = _storage_index_dtype(shape, number_groups)
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


def _coefficient_values(coefficients: ArrayLike, /) -> Array:
    """Device coefficients; host data is transferred, never staged per shape."""
    if isinstance(coefficients, jax.Array):
        return coefficients
    return jax.device_put(np.asarray(coefficients))


def _relation_payload(relation: SparseRelation, /) -> dict[str, object]:
    # Index and mask arrays enter the identity as content digests (dtype, shape
    # and bytes) rather than JSON integer lists: equal identity, linear hashing.
    if _relation_traced(relation):
        raise ValueError(
            "A sparse operator identity requires host-prepared topology; pass "
            "operator_id, or replace the coefficients of a prepared operator."
        )
    payload: dict[str, object] = {
        "kind": type(relation).__name__,
        "source_indices": np.asarray(relation.source_indices),
        "valid": np.asarray(relation.valid),
        "source_size": relation.source_size,
    }
    if isinstance(relation, EdgeRelation):
        payload["target_size"] = relation.target_size
        payload["target_indices"] = np.asarray(relation.target_indices)
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
