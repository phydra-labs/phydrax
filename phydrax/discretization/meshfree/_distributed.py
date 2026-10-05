# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Owned-row distributed meshfree operators and global reductions.

A distributed operator owns every route exactly once: by the owner of its
target row. Sources referenced on other owners become deduplicated halo
columns (:class:`~phydrax.discretization.spatial.DistributedHaloPlan`). The
forward action gathers halo columns and reduces owned routes; the transpose
reduces route cotangents into local columns and returns halo contributions to
their owners; the Hilbert adjoint composes the transpose with the diagonal
pairings of the bound spaces. Reductions are owner-local in slot order and then
combined across owners by the collective all-reduce, so results are
reproducible for a fixed partition and backend, not across partitions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends.distributed import JaxCollectiveProvider
from ...linalg import ArraySpace, DiagonalPairing, EuclideanPairing
from ...sparse import (
    block_linear_adjoint_apply,
    block_linear_apply,
    block_linear_transpose_apply,
    EdgeRelation,
    linear_adjoint_apply,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseCoordinateOperator,
)
from ..spatial import (
    DistributedHaloEvidence,
    DistributedHaloPlan,
    DistributedNeighborResult,
    DistributedPointLayout,
)


def _spec(axis: str, ndim: int) -> PartitionSpec:
    return PartitionSpec(axis, *([None] * (ndim - 1)))


class DistributedMeshfreeEvidence(NonTrainableState, StrictModule):
    """Halo, route-capacity, and partition-epoch evidence of one binding."""

    halo: DistributedHaloEvidence
    routes: Array
    route_capacity: Array
    source_epochs: Array
    target_epochs: Array


def _pairing_weights(
    space: ArraySpace | None,
    layout: DistributedPointLayout,
    fiber: int,
    block: bool,
) -> Array | None:
    """Distributed diagonal Riesz weights; ``None`` for the Euclidean pairing."""
    if space is None:
        return None
    if not isinstance(space, ArraySpace):
        raise TypeError("Distributed adjoints require native ArraySpace spaces.")
    pairing = space.pairing
    if isinstance(pairing, EuclideanPairing):
        return None
    if not isinstance(pairing, DiagonalPairing):
        raise TypeError(
            "Distributed adjoints support only Euclidean or diagonal pairings."
        )
    weights = jnp.asarray(pairing.weights).reshape(
        (layout.logical_count, fiber) if block else (layout.logical_count,)
    )
    blocked = layout.distribute(weights)
    mask = layout.active.reshape(layout.active.shape + (1,) * (blocked.ndim - 1))
    return jnp.where(mask, blocked, jnp.ones((), dtype=blocked.dtype))


@final
class DistributedMeshfreeOperator(StrictModule):
    """A sparse meshfree operator bound to owned target rows and halo columns.

    Values live in the owner-blocked slot layouts of ``source_layout`` and
    ``target_layout``. A binding is valid for the layout epochs recorded in
    ``evidence``; after an ownership migration the operator must be rebound.
    """

    source_layout: DistributedPointLayout
    target_layout: DistributedPointLayout
    halo: DistributedHaloPlan
    route_targets: Array
    coefficients: Array
    source_weights: Array | None
    target_weights: Array | None
    evidence: DistributedMeshfreeEvidence
    block_shape: tuple[int, int] | None = eqx.field(static=True)
    accumulation_dtype: np.dtype = eqx.field(static=True)
    output_dtype: np.dtype = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_layout: DistributedPointLayout,
        target_layout: DistributedPointLayout,
        route_targets: ArrayLike,
        route_source_owners: ArrayLike,
        route_source_slots: ArrayLike,
        route_valid: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        halo_capacity: int,
        operator_id: str,
        block_shape: tuple[int, int] | None = None,
        accumulation_dtype: DTypeLike | None = None,
        output_dtype: DTypeLike | None = None,
        source_space: ArraySpace | None = None,
        target_space: ArraySpace | None = None,
    ) -> None:
        """Bind owner-blocked routes; refuse incomplete halos.

        Route arrays are owner-blocked by target owner with ``route_capacity``
        routes per owner. ``route_targets`` are owned target slots and the
        source owner/slot pairs address ``source_layout``.
        """
        if not isinstance(source_layout, DistributedPointLayout) or not isinstance(
            target_layout, DistributedPointLayout
        ):
            raise TypeError("Layouts must be DistributedPointLayout values.")
        source_plan = source_layout.plan
        if not source_plan.compatible(target_layout.plan):
            raise ValueError("Source and target layouts must share one owner mesh.")
        targets = jnp.asarray(route_targets)
        valid = jnp.asarray(route_valid, dtype=jnp.bool_)
        if targets.ndim != 1 or targets.shape[0] % source_plan.owner_count:
            raise ValueError("route_targets must be an owner-blocked route vector.")
        if valid.shape != targets.shape:
            raise ValueError("route_valid must match route_targets.")
        block = None if block_shape is None else tuple(int(size) for size in block_shape)
        if block is not None and (len(block) != 2 or min(block) < 1):
            raise ValueError("block_shape must hold positive target and source fibers.")
        values = jnp.asarray(coefficients)
        expected = targets.shape + (() if block is None else block)
        if values.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError("coefficients must be inexact.")
        identifier = str(operator_id)
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        halo = DistributedHaloPlan(
            source_plan,
            route_source_owners,
            route_source_slots,
            valid,
            halo_capacity=halo_capacity,
        )
        # Preparation boundary: an incomplete halo would silently drop routes.
        if not bool(halo.evidence.successful):
            raise ValueError(
                "Distributed meshfree halo is incomplete: "
                f"{int(halo.evidence.refused_routes)} routes refused with "
                f"halo_capacity={halo.halo_capacity} and maximum halo load "
                f"{int(halo.evidence.maximum_halo_load)}."
            )
        target_plan = target_layout.plan
        accumulation = np.dtype(
            values.dtype if accumulation_dtype is None else accumulation_dtype
        )
        if not np.issubdtype(accumulation, np.inexact):
            raise TypeError("accumulation_dtype must be inexact.")
        source_fiber, target_fiber = (1, 1) if block is None else (block[1], block[0])
        self.source_layout = source_layout
        self.target_layout = target_layout
        self.halo = halo
        self.route_targets = target_plan.place(
            jnp.where(halo.route_valid, targets, 0).astype(jnp.int32)
        )
        self.coefficients = target_plan.place(values)
        self.source_weights = _pairing_weights(
            source_space, source_layout, source_fiber, block is not None
        )
        self.target_weights = _pairing_weights(
            target_space, target_layout, target_fiber, block is not None
        )
        self.evidence = DistributedMeshfreeEvidence(
            halo=halo.evidence,
            routes=jnp.sum(halo.route_valid, dtype=jnp.int32),
            route_capacity=jnp.asarray(halo.route_capacity, dtype=jnp.int32),
            source_epochs=source_layout.owner_epochs,
            target_epochs=target_layout.owner_epochs,
        )
        self.block_shape = block
        self.accumulation_dtype = accumulation
        self.output_dtype = np.dtype(
            accumulation if output_dtype is None else output_dtype
        )
        self.operator_id = identifier
        self.binding_id = canonical_fingerprint(
            {
                "kind": "distributed-meshfree-operator",
                "operator_id": identifier,
                "source_ownership": source_plan.plan_id,
                "target_ownership": target_plan.plan_id,
                "route_capacity": halo.route_capacity,
                "halo_capacity": halo.halo_capacity,
                "block_shape": None if block is None else list(block),
            }
        )

    @classmethod
    def bind(
        cls,
        operator: SparseCoordinateOperator,
        source_layout: DistributedPointLayout,
        target_layout: DistributedPointLayout,
        /,
        *,
        halo_capacity: int,
        route_capacity: int | None = None,
    ) -> DistributedMeshfreeOperator:
        """Partition a prepared sparse operator over owned target rows.

        Host preparation reads the relation once; coefficients stay dynamic
        and are gathered on device into the owner-blocked route order.
        """
        if not isinstance(operator, SparseCoordinateOperator):
            raise TypeError("operator must be a native SparseCoordinateOperator.")
        relation = operator.relation
        block = operator.block_shape
        if isinstance(relation, RowRelation):
            if relation.case_shape or len(relation.target_shape) != 1:
                raise ValueError(
                    "Distributed row relations must be one unbatched target vector."
                )
            edge = relation.as_edge_relation()
            flat = operator.coefficients.reshape((-1,) + (() if block is None else block))
        elif isinstance(relation, EdgeRelation):
            edge = relation
            flat = operator.coefficients
        else:
            raise TypeError("operator relation must be a native sparse relation.")
        if edge.source_size != source_layout.logical_count:
            raise ValueError("Operator source size must match the source layout.")
        if edge.target_size != target_layout.logical_count:
            raise ValueError("Operator target size must match the target layout.")
        routes = _owner_blocked_routes(
            np.asarray(jax.device_get(edge.source_indices)),
            np.asarray(jax.device_get(edge.target_indices)),
            np.asarray(jax.device_get(edge.valid)),
            source_layout,
            target_layout,
            route_capacity,
        )
        route_index, target_slots, source_owners, source_slots, valid = routes
        mask = jnp.asarray(valid).reshape(valid.shape + (1,) * (flat.ndim - 1))
        coefficients = jnp.where(
            mask, flat[jnp.asarray(route_index)], jnp.zeros((), dtype=flat.dtype)
        )
        source_space = operator.source
        target_space = operator.target
        if not isinstance(source_space, ArraySpace) or not isinstance(
            target_space, ArraySpace
        ):
            raise TypeError("Distributed binding requires native ArraySpace spaces.")
        return cls(
            source_layout,
            target_layout,
            target_slots,
            source_owners,
            source_slots,
            valid,
            coefficients,
            halo_capacity=halo_capacity,
            operator_id=operator.operator_id,
            block_shape=block,
            accumulation_dtype=operator.accumulation_dtype,
            output_dtype=target_space.dtype,
            source_space=source_space,
            target_space=target_space,
        )

    @classmethod
    def from_neighbor_rows(
        cls,
        rows: DistributedNeighborResult,
        coefficients: ArrayLike,
        source_layout: DistributedPointLayout,
        target_layout: DistributedPointLayout,
        /,
        *,
        halo_capacity: int,
        operator_id: str,
        block_shape: tuple[int, int] | None = None,
    ) -> DistributedMeshfreeOperator:
        """Bind distributed neighbor rows and per-row coefficients without a host step.

        ``coefficients`` follow the rows' owner-blocked shape, optionally with
        trailing ``block_shape`` fibers. Only ``COMPLETE`` rows carry routes.
        """
        if not isinstance(rows, DistributedNeighborResult):
            raise TypeError("rows must be a DistributedNeighborResult.")
        shape = rows.valid.shape
        if shape[0] != target_layout.plan.total_capacity:
            raise ValueError("rows must follow the target layout's owner blocks.")
        width = shape[1]
        local = target_layout.plan.local_capacity
        slots = jnp.repeat(jnp.arange(shape[0], dtype=jnp.int32) % local, width)
        values = jnp.asarray(coefficients)
        fibers = () if block_shape is None else tuple(block_shape)
        if values.shape != shape + fibers:
            raise ValueError(f"coefficients must have shape {shape + fibers}.")
        return cls(
            source_layout,
            target_layout,
            slots,
            rows.source_owners.reshape((-1,)),
            rows.source_slots.reshape((-1,)),
            rows.valid.reshape((-1,)),
            values.reshape((-1,) + fibers),
            halo_capacity=halo_capacity,
            operator_id=operator_id,
            block_shape=block_shape,
        )

    def _relation(self, columns: Array, targets: Array, valid: Array) -> EdgeRelation:
        return EdgeRelation(
            columns,
            targets,
            source_size=self.halo.column_count,
            target_size=self.target_layout.plan.local_capacity,
            valid=valid,
        )

    def _validated(
        self, values: ArrayLike, layout: DistributedPointLayout, *, source: bool
    ) -> Array:
        array = jnp.asarray(values)
        fiber = (
            ()
            if self.block_shape is None
            else ((self.block_shape[1],) if source else (self.block_shape[0],))
        )
        if (
            array.ndim < 1 + len(fiber)
            or array.shape[0] != layout.plan.total_capacity
            or tuple(array.shape[1 : 1 + len(fiber)]) != fiber
        ):
            raise ValueError(
                "Values must begin with the owner-blocked slot axis"
                + (f" and fiber {fiber}." if fiber else ".")
            )
        return layout.plan.place(array.astype(self.accumulation_dtype))

    def _map(
        self, kernel: Callable[..., Array], values: Array, output_ndim: int
    ) -> Array:
        axis = self.source_layout.plan.axis_name
        row = _spec(axis, 1)
        coefficient_spec = _spec(axis, self.coefficients.ndim)
        return self.source_layout.plan.map(
            kernel,
            (
                _spec(axis, values.ndim),
                _spec(axis, 2),
                _spec(axis, 2),
                row,
                row,
                row,
                coefficient_spec,
            ),
            _spec(axis, output_ndim),
        )(
            values,
            self.halo.send_slots,
            self.halo.send_valid,
            self.halo.route_columns,
            self.route_targets,
            self.halo.route_valid,
            self.coefficients.astype(self.accumulation_dtype),
        )

    def apply(self, values: ArrayLike, /) -> Array:
        """Owner-blocked target values ``A x`` from owner-blocked sources ``x``."""
        return _operator_apply(
            self, self._validated(values, self.source_layout, source=True)
        )

    def _forward(self, array: Array) -> Array:
        def kernel(
            local: Array,
            send_slots: Array,
            send_valid: Array,
            columns: Array,
            targets: Array,
            valid: Array,
            coefficients: Array,
        ) -> Array:
            gathered = self.halo.local_gather(local, send_slots, send_valid)
            relation = self._relation(columns, targets, valid)
            output = (
                linear_apply(relation, coefficients, gathered)
                if self.block_shape is None
                else block_linear_apply(relation, coefficients, gathered)
            )
            return output.astype(self.output_dtype)

        return self._map(kernel, array, array.ndim)

    def _reverse(self, array: Array, adjoint: bool) -> Array:
        if adjoint and self.target_weights is not None:
            weights = self.target_weights.reshape(
                self.target_weights.shape + (1,) * (array.ndim - self.target_weights.ndim)
            )
            array = array * weights.astype(array.dtype)

        def kernel(
            local: Array,
            send_slots: Array,
            send_valid: Array,
            columns: Array,
            targets: Array,
            valid: Array,
            coefficients: Array,
        ) -> Array:
            relation = self._relation(columns, targets, valid)
            if self.block_shape is None:
                routed = (
                    linear_adjoint_apply(relation, coefficients, local)
                    if adjoint
                    else linear_transpose_apply(relation, coefficients, local)
                )
            else:
                routed = (
                    block_linear_adjoint_apply(relation, coefficients, local)
                    if adjoint
                    else block_linear_transpose_apply(relation, coefficients, local)
                )
            return self.halo.local_transpose(routed, send_slots, send_valid)

        output = self._map(kernel, array, array.ndim)
        if adjoint and self.source_weights is not None:
            weights = self.source_weights.reshape(
                self.source_weights.shape
                + (1,) * (output.ndim - self.source_weights.ndim)
            )
            output = output / weights.astype(output.dtype)
        return output.astype(self.output_dtype)

    def transpose_apply(self, values: ArrayLike, /) -> Array:
        """Algebraic transpose ``A^T y`` summed exactly once onto source owners."""
        return _operator_reverse(
            self, self._validated(values, self.target_layout, source=False), False
        )

    def adjoint_apply(self, values: ArrayLike, /) -> Array:
        """Hilbert adjoint ``M_s^{-1} A^H M_t y`` under the bound diagonal pairings."""
        return _operator_reverse(
            self, self._validated(values, self.target_layout, source=False), True
        )


@eqx.filter_jit
def _operator_apply(operator: DistributedMeshfreeOperator, values: Array) -> Array:
    return operator._forward(values)


@eqx.filter_jit
def _operator_reverse(
    operator: DistributedMeshfreeOperator, values: Array, adjoint: bool
) -> Array:
    return operator._reverse(values, adjoint)


def _owner_blocked_routes(
    sources: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    source_layout: DistributedPointLayout,
    target_layout: DistributedPointLayout,
    route_capacity: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-order routes by (target owner, target slot, route) into owner blocks."""
    source_owner, source_slot = source_layout.host_addresses()
    target_owner, target_slot = target_layout.host_addresses()
    routes = np.flatnonzero(valid)
    if np.any(source_owner[sources[routes]] < 0) or np.any(
        target_owner[targets[routes]] < 0
    ):
        raise ValueError("Operator routes reference points absent from the layouts.")
    owner = target_owner[targets[routes]]
    slot = target_slot[targets[routes]]
    order = np.lexsort((routes, slot, owner))
    routes, owner, slot = routes[order], owner[order], slot[order]
    owners = target_layout.plan.owner_count
    loads = np.bincount(owner, minlength=owners)
    capacity = max(1, int(np.max(loads, initial=0)))
    if route_capacity is not None:
        if int(route_capacity) < capacity:
            raise ValueError(
                f"route_capacity={int(route_capacity)} is below the owner route "
                f"load {capacity}."
            )
        capacity = int(route_capacity)
    starts = np.concatenate(([0], np.cumsum(loads)[:-1]))
    position = owner * capacity + (np.arange(routes.size) - starts[owner])
    total = owners * capacity
    route_index = np.zeros((total,), dtype=np.int64)
    target_slots = np.zeros((total,), dtype=np.int32)
    source_owners = np.zeros((total,), dtype=np.int32)
    source_slots = np.zeros((total,), dtype=np.int32)
    route_valid = np.zeros((total,), dtype=np.bool_)
    route_index[position] = routes
    target_slots[position] = slot
    source_owners[position] = source_owner[sources[routes]]
    source_slots[position] = source_slot[sources[routes]]
    route_valid[position] = True
    return route_index, target_slots, source_owners, source_slots, route_valid


def _owned_weights(
    layout: DistributedPointLayout, values: Array, weights: ArrayLike | None
) -> Array:
    mask = layout.active.reshape(layout.active.shape + (1,) * (values.ndim - 1))
    if weights is None:
        return mask.astype(values.real.dtype)
    weight = jnp.asarray(weights)
    if weight.shape[:1] != values.shape[:1]:
        raise ValueError("weights must begin with the owner-blocked slot axis.")
    weight = weight.reshape(weight.shape + (1,) * (values.ndim - weight.ndim))
    return jnp.where(mask, weight, jnp.zeros((), dtype=weight.dtype))


@eqx.filter_jit
def _global_reduce(
    layout: DistributedPointLayout, product: Array, keep_payload: bool
) -> Array:
    plan = layout.plan
    axis = plan.axis_name
    placed = plan.place(product)

    def reduce(local: Array) -> Array:
        total = jnp.sum(local, axis=0) if keep_payload else jnp.sum(local)
        return JaxCollectiveProvider(axis).sum(total)

    return plan.map(reduce, (_spec(axis, placed.ndim),), PartitionSpec())(placed)


def distributed_sum(
    layout: DistributedPointLayout,
    values: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Weighted sum over owned active slots, e.g. a compatibility integral."""
    array = jnp.asarray(values)
    if array.ndim == 0 or array.shape[0] != layout.plan.total_capacity:
        raise ValueError("values must begin with the owner-blocked slot axis.")
    weight = _owned_weights(layout, array, weights)
    return _global_reduce(layout, weight * array, True)


def distributed_inner(
    layout: DistributedPointLayout,
    left: ArrayLike,
    right: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Weighted Hermitian inner product over owned active slots."""
    first = jnp.asarray(left)
    second = jnp.asarray(right)
    if first.shape != second.shape:
        raise ValueError("Inner-product arguments must have equal shapes.")
    if first.ndim == 0 or first.shape[0] != layout.plan.total_capacity:
        raise ValueError("values must begin with the owner-blocked slot axis.")
    weight = _owned_weights(layout, first, weights)
    return _global_reduce(layout, jnp.conj(first) * weight * second, False)


def distributed_norm(
    layout: DistributedPointLayout,
    values: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Weighted Euclidean norm over owned active slots."""
    return jnp.sqrt(jnp.real(distributed_inner(layout, values, values, weights=weights)))


__all__ = [
    "DistributedMeshfreeEvidence",
    "DistributedMeshfreeOperator",
    "distributed_inner",
    "distributed_norm",
    "distributed_sum",
]
