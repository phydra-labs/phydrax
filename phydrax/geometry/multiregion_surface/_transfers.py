#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse conservative field transfers between multiregion entity layouts.

Topology changes (remeshing, T1, pinch, burst) move data between a source and
a target layout of vertices, sheet slots, faces or regions. Two contracts are
distinguished:

- `ConservativeFieldTransfer` moves **extensive** content (liquid volume,
  surfactant amount, gas amount): ``target_t = sum_s w_ts source_s`` with
  ``w_ts >= 0`` and ``sum_t w_ts = 1`` for every active source, so totals are
  conserved exactly up to roundoff and nonnegative content stays nonnegative;
- `BoundedFieldReconstruction` reconstructs **intensive** quantities
  (thickness, concentration, velocity) as convex combinations
  ``target_t = sum_s w_ts source_s / sum_s w_ts``, so every target lies inside
  the range of its supporting sources.

Both are fixed sparse relations executed with `phydrax.sparse`; construction
refuses nonconservative or unsupported weights, and every application reports
conservation, positivity/boundedness and support evidence. A conservative
transfer between consecutive topology epochs is also exposed as the
nondifferentiable `TopologyEpochTransition` of the discretization substrate.

Sheet-slot content across local topology events uses corner pooling: the
content of a ``(vertex, region-pair)`` slot is attributed to the slot's face
corners in proportion to face area (uniform density over the barycentric dual
cell); corners of faces untouched by an event keep their content at the same
``(vertex, pair)`` slot, and the corners of the faces an event replaces are
pooled per region pair and redistributed over the replacement faces' corners
in proportion to their barycentric areas. A local edit pair without replacement
faces is redistributed over all replacement faces of that edit. Whole-sheet
burst is different: its vanished content leaves the surviving-sheet transfer
and is returned explicitly for the foam rim ledger. Uniform thickness is
reproduced exactly when an event preserves area (midpoint split), and the total
content is conserved exactly otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_integer
from ...discretization._spaces import DiscreteFieldSpace, EntityDofLayout
from ...discretization._topology_epoch import TopologyEpoch, TopologyEpochTransition
from ...discretization._transfer import (
    FieldTransfer,
    TransferGeometryBinding,
    TransferProperties,
)
from ...linalg import adjoint, ArraySpace, DiagonalPairing, transpose
from ...sparse import (
    EdgeRelation,
    gather_routes,
    linear_apply,
    route_reduce,
    SparseCoordinateOperator,
)
from ...typing import Bool, Dim, Float, Identifier, Scalar, Size


# Roundoff admission: dtype epsilon scaled by the route fan-in.
_RELATIVE_TOLERANCE = 64.0


class _TransferSourceDim(Dim, minimum=1):
    """Source entity slots."""


class _TransferTargetDim(Dim, minimum=1):
    """Target entity slots."""


class _TransferRouteDim(Dim, minimum=1):
    """Sparse transfer routes."""


class _FieldDims(Dim):
    """Trailing field components of transferred values."""


@final
class ExtensiveTransferEvidence(StrictModule):
    """Conservation, positivity and support evidence of one extensive transfer."""

    __strict_contract__ = True

    source_total: Float[_FieldDims]
    target_total: Float[_FieldDims]
    absolute_defect: Float[_FieldDims]
    relative_defect: Float[Scalar]
    tolerance: Float[Scalar]
    conservative: Bool[Scalar]
    source_nonnegative: Bool[Scalar]
    target_nonnegative: Bool[Scalar]
    positivity_preserved: Bool[Scalar]
    support_complete: Bool[Scalar]
    finite: Bool[Scalar]
    successful: Bool[Scalar]


@final
class IntensiveReconstructionEvidence(StrictModule):
    """Boundedness and support evidence of one intensive reconstruction."""

    __strict_contract__ = True

    lower_excess: Float[Scalar]
    upper_excess: Float[Scalar]
    tolerance: Float[Scalar]
    bounded: Bool[Scalar]
    support_complete: Bool[Scalar]
    finite: Bool[Scalar]
    successful: Bool[Scalar]


def _routes(
    source_indices: ArrayLike,
    target_indices: ArrayLike,
    weights: ArrayLike,
    source_active: ArrayLike,
    target_active: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = np.asarray(source_indices)
    targets = np.asarray(target_indices)
    values = np.asarray(weights, dtype=np.float64)
    source_mask = np.asarray(source_active)
    target_mask = np.asarray(target_active)
    if (
        sources.ndim != 1
        or targets.shape != sources.shape
        or values.shape != sources.shape
    ):
        raise ValueError("Transfer routes must be aligned rank-1 arrays.")
    if sources.size == 0:
        raise ValueError("A transfer needs at least one route.")
    if not (
        np.issubdtype(sources.dtype, np.integer)
        and np.issubdtype(targets.dtype, np.integer)
    ):
        raise TypeError("Transfer route indices must be integers.")
    if source_mask.ndim != 1 or source_mask.dtype != np.bool_:
        raise TypeError("source_active must be a boolean vector.")
    if target_mask.ndim != 1 or target_mask.dtype != np.bool_:
        raise TypeError("target_active must be a boolean vector.")
    positive_integer(source_mask.size, "source size")
    positive_integer(target_mask.size, "target size")
    if np.any(sources < 0) or np.any(sources >= source_mask.size):
        raise ValueError("Transfer source indices are out of range.")
    if np.any(targets < 0) or np.any(targets >= target_mask.size):
        raise ValueError("Transfer target indices are out of range.")
    if not (np.all(source_mask[sources]) and np.all(target_mask[targets])):
        raise ValueError("Transfer routes must join active sources to active targets.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Transfer weights must be finite and nonnegative.")
    if np.unique(np.stack((sources, targets), axis=1), axis=0).shape[0] != sources.size:
        raise ValueError("Transfer routes must be unique (source, target) pairs.")
    return (
        sources.astype(np.int64),
        targets.astype(np.int64),
        values,
        source_mask,
        target_mask,
    )


def _tolerance(fan_in: int, /) -> float:
    return _RELATIVE_TOLERANCE * float(np.finfo(np.float64).eps) * max(1, fan_in)


def _padded_routes(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    source_size: int,
    target_size: int,
    /,
) -> tuple[EdgeRelation, Array]:
    """Routes padded to a power-of-two capacity with invalid zero-weight rows.

    Route counts change with every topology event; bucketing the route axis
    keeps the device kernels of successive epochs on a few shapes instead of
    compiling one per event pass. Padding rows are invalid and weightless.
    """
    capacity = max(8, 1 << (sources.size - 1).bit_length())
    padding = capacity - sources.size
    relation = EdgeRelation(
        np.concatenate((sources, np.zeros((padding,), dtype=np.int64))),
        np.concatenate((targets, np.zeros((padding,), dtype=np.int64))),
        source_size=source_size,
        target_size=target_size,
        valid=np.arange(capacity) < sources.size,
    )
    padded = np.concatenate((weights, np.zeros((padding,), dtype=np.float64)))
    return relation, jnp.asarray(padded, dtype=jnp.float64)


@final
class ConservativeFieldTransfer(StrictModule, NonTrainableState):
    """Exactly conservative, positivity-preserving sparse extensive transfer.

    Every active source distributes its content over its routes with
    nonnegative weights summing to one; every active target must receive from
    at least one route (``support_complete``). `apply` maps capacity-shaped
    source values ``(source_size, ...)`` to target values ``(target_size, ...)``.
    """

    __strict_contract__ = True

    relation: EdgeRelation
    weights: Float[_TransferRouteDim]
    source_active: Bool[_TransferSourceDim]
    target_active: Bool[_TransferTargetDim]
    source_size: Size[_TransferSourceDim] = eqx.field(static=True)
    target_size: Size[_TransferTargetDim] = eqx.field(static=True)
    route_capacity: Size[_TransferRouteDim] = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    maximum_partition_defect: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    transfer_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_active: ArrayLike,
        target_active: ArrayLike,
    ) -> None:
        sources, targets, values, source_mask, target_mask = _routes(
            source_indices, target_indices, weights, source_active, target_active
        )
        partition = np.bincount(sources, weights=values, minlength=source_mask.size)
        fan_out = np.bincount(sources, minlength=source_mask.size)
        tolerance = _tolerance(int(np.max(fan_out)))
        defect = float(np.max(np.abs(partition[source_mask] - 1.0)))
        if defect > tolerance:
            raise ValueError(
                "Extensive transfer weights must sum to one over the routes of every "
                f"active source (defect {defect:.3e} > {tolerance:.3e})."
            )
        received = np.bincount(targets, weights=values, minlength=target_mask.size)
        if np.any(received[target_mask] <= 0.0):
            raise ValueError("Every active target must receive positive transfer weight.")
        self.relation, self.weights = _padded_routes(
            sources, targets, values, source_mask.size, target_mask.size
        )
        self.source_active = jnp.asarray(source_mask)
        self.target_active = jnp.asarray(target_mask)
        self.source_size = source_mask.size
        self.target_size = target_mask.size
        self.route_capacity = self.weights.shape[0]
        self.route_count = sources.size
        self.maximum_partition_defect = defect
        self.tolerance = tolerance
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "conservative-field-transfer",
                "sources": array_tree_fingerprint(sources),
                "targets": array_tree_fingerprint(targets),
                "weights": array_tree_fingerprint(values),
                "source_active": array_tree_fingerprint(source_mask),
                "target_active": array_tree_fingerprint(target_mask),
            }
        )

    def _masked(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.source_size:
            raise ValueError("Transfer values must lead with the source capacity axis.")
        mask = self.source_active.reshape((-1,) + (1,) * (array.ndim - 1))
        return jnp.where(mask, array, jnp.zeros((), dtype=array.dtype))

    def apply(self, values: ArrayLike, /) -> Array:
        """Transfer extensive source content onto the target layout."""
        return linear_apply(self.relation, self.weights, self._masked(values))

    def evidence(
        self, source_values: ArrayLike, target_values: ArrayLike, /
    ) -> ExtensiveTransferEvidence:
        """Conservation/positivity/support evidence of one transfer application."""
        source = self._masked(source_values)
        target = jnp.asarray(target_values)
        if target.shape != (self.target_size,) + source.shape[1:]:
            raise ValueError("target_values must match the target layout.")
        target_mask = self.target_active.reshape((-1,) + (1,) * (target.ndim - 1))
        target = jnp.where(target_mask, target, jnp.zeros((), dtype=target.dtype))
        source_total = jnp.sum(source, axis=0)
        target_total = jnp.sum(target, axis=0)
        defect = jnp.abs(target_total - source_total)
        scale = jnp.maximum(
            jnp.max(jnp.sum(jnp.abs(source), axis=0), initial=0.0), 1e-300
        )
        relative = jnp.max(defect, initial=0.0) / scale
        tolerance = jnp.asarray(self.tolerance * max(1, self.route_count) ** 0.5)
        source_nonnegative = jnp.all(source >= 0.0)
        target_nonnegative = jnp.all(target >= 0.0)
        received = linear_apply(
            self.relation,
            self.weights,
            self.source_active.astype(self.weights.dtype),
        )
        support = jnp.all(jnp.where(self.target_active, received > 0.0, True))
        finite = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
        conservative = relative <= tolerance
        positivity = ~source_nonnegative | target_nonnegative
        return ExtensiveTransferEvidence(
            source_total=source_total.reshape((-1,)),
            target_total=target_total.reshape((-1,)),
            absolute_defect=defect.reshape((-1,)),
            relative_defect=relative,
            tolerance=tolerance,
            conservative=conservative,
            source_nonnegative=source_nonnegative,
            target_nonnegative=target_nonnegative,
            positivity_preserved=positivity,
            support_complete=support,
            finite=finite,
            successful=finite & conservative & positivity & support,
        )

    def epoch_transition(
        self,
        source: TopologyEpoch,
        target: TopologyEpoch,
        /,
        *,
        field_name: str,
        geometry: TransferGeometryBinding,
    ) -> TopologyEpochTransition:
        """This content transfer as a nondifferentiable topology-epoch transition.

        Source and target are ``cell_integral`` content spaces with unit
        measures (inactive padding slots carry zero content), so the epoch
        transition's conservation check is the total-content identity.

        ``geometry`` must bind the owning source and target surface records.
        The extensive route map alone has no coordinate witness; the epochs
        are checked independently against this binding.
        """
        name = canonical_identifier(field_name, "field_name")
        if not isinstance(geometry, TransferGeometryBinding):
            raise TypeError("geometry must be TransferGeometryBinding.")
        source_space = _content_space(name, source, self.source_size)
        target_space = _content_space(name, target, self.target_size)
        primal = SparseCoordinateOperator(
            self.relation,
            self.weights,
            source=source_space.vector_space,
            target=target_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "multiregion-content-transfer-operator",
                    "transfer": self.transfer_id,
                    "source": source.epoch_id,
                    "target": target.epoch_id,
                    "field": name,
                }
            ),
        )
        transfer = FieldTransfer(
            source_space,
            target_space,
            primal,
            dual_pullback_operator=transpose(primal),
            hilbert_adjoint_operator=adjoint(primal),
            geometry=geometry,
            properties=TransferProperties(
                conservative=True,
                positivity_preserving=True,
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=("extensive-content",),
            ),
        )
        return TopologyEpochTransition(
            source,
            target,
            transfer,
            np.ones((self.source_size,), dtype=np.float64),
            np.ones((self.target_size,), dtype=np.float64),
        )


def _content_space(name: str, epoch: TopologyEpoch, size: int, /) -> DiscreteFieldSpace:
    layout = EntityDofLayout(
        canonical_fingerprint(
            {"kind": "multiregion-content-slots", "epoch": epoch.epoch_id, "size": size}
        ),
        size,
        size,
    )
    return DiscreteFieldSpace(
        name,
        epoch.epoch_id,
        layout,
        ArraySpace(
            (size,),
            dtype=jnp.float64,
            pairing=DiagonalPairing(jnp.ones((size,), dtype=jnp.float64)),
        ),
        representation="cell_integral",
        conformity="discontinuous",
    )


@final
class BoundedFieldReconstruction(StrictModule, NonTrainableState):
    """Convex (bounded) sparse reconstruction of intensive fields.

    Weights are normalized per active target, so each reconstructed value lies
    in the closed range of its supporting source values.
    """

    __strict_contract__ = True

    relation: EdgeRelation
    weights: Float[_TransferRouteDim]
    source_active: Bool[_TransferSourceDim]
    target_active: Bool[_TransferTargetDim]
    source_size: Size[_TransferSourceDim] = eqx.field(static=True)
    target_size: Size[_TransferTargetDim] = eqx.field(static=True)
    route_capacity: Size[_TransferRouteDim] = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    reconstruction_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_active: ArrayLike,
        target_active: ArrayLike,
    ) -> None:
        sources, targets, values, source_mask, target_mask = _routes(
            source_indices, target_indices, weights, source_active, target_active
        )
        totals = np.bincount(targets, weights=values, minlength=target_mask.size)
        if np.any(totals[target_mask] <= 0.0):
            raise ValueError("Every active target needs positive reconstruction support.")
        normalized = values / totals[targets]
        self.relation, self.weights = _padded_routes(
            sources, targets, normalized, source_mask.size, target_mask.size
        )
        self.source_active = jnp.asarray(source_mask)
        self.target_active = jnp.asarray(target_mask)
        self.source_size = source_mask.size
        self.target_size = target_mask.size
        self.route_capacity = self.weights.shape[0]
        self.route_count = sources.size
        self.reconstruction_id = canonical_fingerprint(
            {
                "kind": "bounded-field-reconstruction",
                "sources": array_tree_fingerprint(sources),
                "targets": array_tree_fingerprint(targets),
                "weights": array_tree_fingerprint(normalized),
                "source_active": array_tree_fingerprint(source_mask),
                "target_active": array_tree_fingerprint(target_mask),
            }
        )

    def apply(self, values: ArrayLike, /) -> Array:
        """Reconstruct intensive target values as convex source combinations."""
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[0] != self.source_size:
            raise ValueError("Reconstruction values must lead with the source axis.")
        return linear_apply(self.relation, self.weights, array)

    def evidence(
        self, source_values: ArrayLike, target_values: ArrayLike, /
    ) -> IntensiveReconstructionEvidence:
        """Local-bound and support evidence of one reconstruction."""
        source = jnp.asarray(source_values)
        target = jnp.asarray(target_values)
        if target.shape != (self.target_size,) + source.shape[1:]:
            raise ValueError("target_values must match the target layout.")
        routed = gather_routes(self.relation, source)
        lower = route_reduce(self.relation, routed, reduction="min")
        upper = route_reduce(self.relation, routed, reduction="max")
        mask = self.target_active.reshape((-1,) + (1,) * (target.ndim - 1))
        lower_excess = jnp.max(jnp.where(mask, lower - target, 0.0), initial=0.0)
        upper_excess = jnp.max(jnp.where(mask, target - upper, 0.0), initial=0.0)
        scale = jnp.maximum(jnp.max(jnp.abs(source), initial=0.0), 1e-300)
        tolerance = jnp.asarray(_tolerance(self.route_count)) * scale
        counts = route_reduce(
            self.relation, jnp.ones((self.route_capacity,), dtype=jnp.float64)
        )
        support = jnp.all(jnp.where(self.target_active, counts > 0.0, True))
        finite = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
        bounded = (lower_excess <= tolerance) & (upper_excess <= tolerance)
        return IntensiveReconstructionEvidence(
            lower_excess=lower_excess,
            upper_excess=upper_excess,
            tolerance=tolerance,
            bounded=bounded,
            support_complete=support,
            finite=finite,
            successful=finite & bounded & support,
        )


def _combined_routes(
    sources: np.ndarray, targets: np.ndarray, weights: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge repeated ``(source, target)`` routes by summing their weights."""
    keys, inverse = np.unique(
        np.stack((sources, targets), axis=1), axis=0, return_inverse=True
    )
    summed = np.bincount(inverse.reshape(-1), weights=weights, minlength=keys.shape[0])
    return keys[:, 0], keys[:, 1], summed


def _pooled_slot_routes(
    source_slots: np.ndarray,
    source_areas: np.ndarray,
    source_pairs: np.ndarray,
    target_slots: np.ndarray,
    target_areas: np.ndarray,
    target_pairs: np.ndarray,
    corner_targets: np.ndarray,
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner-pooling routes of sheet-slot content (see the module docstring).

    ``source_slots``/``target_slots`` hold the flattened ``vertex * width +
    slot`` index of every face corner; ``corner_targets`` holds the target slot
    of every corner of an identity-kept face and ``-1`` on pooled faces.
    ``source_pairs``/``target_pairs`` are region-pair keys in one shared
    (parent-region) numbering. ``groups`` lists the replaced source faces and
    replacement target faces of every event.
    """
    slot_area = np.bincount(
        source_slots.reshape(-1),
        weights=np.repeat(source_areas, 3),
        minlength=int(np.max(source_slots)) + 1,
    )
    fraction = source_areas[:, None] / slot_area[source_slots]
    kept = corner_targets >= 0
    sources = [source_slots[kept]]
    targets = [corner_targets[kept]]
    weights = [fraction[kept]]
    for replaced, replacement in groups:
        if replacement.size == 0:
            raise RuntimeError("An event group lost every replacement face.")
        for pair in np.unique(source_pairs[replaced]):
            members = replaced[source_pairs[replaced] == pair]
            receivers = replacement[target_pairs[replacement] == pair]
            receivers = replacement if receivers.size == 0 else receivers
            share = np.repeat(
                target_areas[receivers] / np.sum(target_areas[receivers]), 3
            )
            corner_source = source_slots[members].reshape(-1)
            corner_fraction = fraction[members].reshape(-1)
            corner_target = target_slots[receivers].reshape(-1)
            sources.append(np.repeat(corner_source, corner_target.size))
            targets.append(np.tile(corner_target, corner_source.size))
            weights.append(np.outer(corner_fraction, share / 3.0).reshape(-1))
    return _combined_routes(
        np.concatenate(sources), np.concatenate(targets), np.concatenate(weights)
    )


__all__ = [
    "BoundedFieldReconstruction",
    "ConservativeFieldTransfer",
    "ExtensiveTransferEvidence",
    "IntensiveReconstructionEvidence",
]
