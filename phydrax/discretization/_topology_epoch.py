#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._transfer import FieldTransfer


if TYPE_CHECKING:
    from ..lifecycle import CompositionEntry, CompositionTransport


class TopologyEpoch(StrictModule, NonTrainableState):
    """Canonical identity of one realized geometry/topology/partition epoch."""

    index: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self, index: int, geometry_id: str, topology_id: str, partition_id: str, /
    ) -> None:
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or int(index) < 0
            or int(index) > np.iinfo(np.int32).max
        ):
            raise ValueError("Topology epoch index must be a nonnegative int32 value.")
        identities = (geometry_id, topology_id, partition_id)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in identities
        ):
            raise ValueError("Topology epoch identities must be canonical identifiers.")
        index_ = int(index)
        self.index = index_
        self.geometry_id = geometry_id
        self.topology_id = topology_id
        self.partition_id = partition_id
        self.epoch_id = canonical_fingerprint(
            {"kind": "topology-epoch", "index": index_, "identities": identities}
        )

    def to_archive_record(self) -> dict[str, Any]:
        """Return the complete canonical JSON record for this epoch."""

        return {
            "index": self.index,
            "geometry_id": self.geometry_id,
            "topology_id": self.topology_id,
            "partition_id": self.partition_id,
            "epoch_id": self.epoch_id,
        }

    @classmethod
    def from_archive_record(cls, record: dict[str, Any], /) -> TopologyEpoch:
        """Strictly reconstruct an epoch and verify its canonical identity."""

        fields = frozenset(
            (
                "index",
                "geometry_id",
                "topology_id",
                "partition_id",
                "epoch_id",
            )
        )
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("Topology epoch archive fields changed.")
        expected = record["epoch_id"]
        if not isinstance(expected, str) or not expected or expected != expected.strip():
            raise ValueError("epoch_id must be a nonempty canonical identifier.")
        epoch = cls(
            record["index"],
            record["geometry_id"],
            record["topology_id"],
            record["partition_id"],
        )
        if epoch.epoch_id != expected:
            raise ValueError("Topology epoch archive identity changed.")
        return epoch


class TopologyEpochTransitionResult(StrictModule):
    """Transferred values with their content ledger.

    ``content_tolerance`` is the admissible ``|conservation_residual|``: the
    roundoff of both content sums plus the transfer's certified measure defect
    acting on this field.
    """

    values: Array
    source_content: Array
    target_content: Array
    conservation_residual: Array
    content_tolerance: Array
    successful: Array
    differentiation_available: Array


class TopologyEpochTransition(StrictModule, NonTrainableState):
    """Explicit fixed transfer between two nondifferentiable topology epochs.

    ``measure_defect_bound`` is the owner-certified bound, per source DOF and in
    measure units, on ``|P^T target_measures - source_measures|`` of the
    transfer ``P``; the content of a field ``v`` then changes by at most
    ``sum(bound * |v|)`` beyond roundoff. ``None`` certifies exact conservation
    (only roundoff remains), as for nested or exactly normalized transfers.
    """

    source: TopologyEpoch
    target: TopologyEpoch
    transfer: FieldTransfer
    source_measures: Array
    target_measures: Array
    measure_defect_bound: Array
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TopologyEpoch,
        target: TopologyEpoch,
        transfer: FieldTransfer,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        measure_defect_bound: ArrayLike | None = None,
    ) -> None:
        if not isinstance(source, TopologyEpoch) or not isinstance(target, TopologyEpoch):
            raise TypeError("Topology transition endpoints must be TopologyEpoch values.")
        if target.index != source.index + 1 or source.epoch_id == target.epoch_id:
            raise ValueError(
                "Topology transitions must connect consecutive distinct epochs."
            )
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("Topology transition requires FieldTransfer.")
        if (
            not transfer.properties.conservative
            or not transfer.properties.adjoint_paired
            or transfer.properties.differentiable_geometry
            or transfer.dual_pullback_operator is None
            or transfer.hilbert_adjoint_operator is None
        ):
            raise ValueError(
                "Topology transfer needs conservative dual/adjoint pairs and nondifferentiable geometry."
            )
        source_measure = np.asarray(source_measures, dtype=np.float64)
        target_measure = np.asarray(target_measures, dtype=np.float64)
        if (
            source_measure.shape != (transfer.primal_operator.source.size,)
            or target_measure.shape != (transfer.primal_operator.target.size,)
            or np.any(~np.isfinite(source_measure))
            or np.any(source_measure <= 0)
            or np.any(~np.isfinite(target_measure))
            or np.any(target_measure <= 0)
        ):
            raise ValueError(
                "Topology transfer measures must match positive scalar spaces."
            )
        defect_bound = (
            np.zeros_like(source_measure)
            if measure_defect_bound is None
            else np.broadcast_to(
                np.asarray(measure_defect_bound, dtype=np.float64),
                source_measure.shape,
            )
        )
        if np.any(~np.isfinite(defect_bound)) or np.any(defect_bound < 0.0):
            raise ValueError(
                "measure_defect_bound must be finite, nonnegative, and one per source "
                "DOF."
            )
        self.source, self.target, self.transfer = source, target, transfer
        self.source_measures, self.target_measures = (
            jnp.asarray(source_measure),
            jnp.asarray(target_measure),
        )
        self.measure_defect_bound = jnp.asarray(defect_bound)
        self.transition_id = canonical_fingerprint(
            {
                "kind": "topology-epoch-transition",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "transfer": transfer.transfer_id,
                "source_measures": source_measure,
                "target_measures": target_measure,
                "measure_defect_bound": np.ascontiguousarray(defect_bound),
            }
        )

    def apply(self, values: ArrayLike, /) -> TopologyEpochTransitionResult:
        flat = jnp.asarray(values).reshape(-1)
        source_space = self.transfer.primal_operator.source
        target_space = self.transfer.primal_operator.target
        if flat.shape != (source_space.size,):
            raise ValueError("Topology transition field does not match source space.")
        result = target_space.flatten(
            self.transfer.primal_operator.mv(source_space.unflatten(flat))
        )
        source_content = jnp.vdot(self.source_measures, flat)
        target_content = jnp.vdot(self.target_measures, result)
        residual = target_content - source_content
        # Roundoff of both content sums scales with the transported magnitudes
        # themselves (a unit floor would admit relative errors of 1e-5 for SI
        # content such as liquid volumes of 1e-9 m^3); the certified measure
        # defect adds its exact action bound on this field.
        finfo = jnp.finfo(jnp.real(result).dtype)
        magnitude = jnp.vdot(self.source_measures, jnp.abs(flat)) + jnp.vdot(
            self.target_measures, jnp.abs(result)
        )
        tolerance = 100 * finfo.eps * jnp.maximum(magnitude, finfo.tiny) + jnp.vdot(
            self.measure_defect_bound, jnp.abs(flat)
        )
        successful = jnp.all(jnp.isfinite(result)) & (jnp.abs(residual) <= tolerance)
        return TopologyEpochTransitionResult(
            result,
            source_content,
            target_content,
            residual,
            tolerance,
            successful,
            jnp.asarray(False),
        )

    def transpose(self, target_cotangent: ArrayLike, /) -> Array:
        flat = jnp.asarray(target_cotangent).reshape(-1)
        adjoint = self.transfer.hilbert_adjoint_operator
        if adjoint is None:
            raise RuntimeError("Topology transition lost its required Hilbert adjoint.")
        target_space = adjoint.source
        source_space = adjoint.target
        if flat.shape != (target_space.size,):
            raise ValueError("Topology transition cotangent does not match target space.")
        return source_space.flatten(adjoint.mv(target_space.unflatten(flat)))

    def require_differentiable_topology(self) -> None:
        raise ValueError(
            "Topology selection is nondifferentiable; differentiate only within one fixed epoch."
        )

    def composition_transport(
        self, source: CompositionEntry, target: CompositionEntry, /
    ) -> CompositionTransport:
        """Physical-remap evidence of this transition for one composition state entry.

        `source` holds state on this transition's source epoch and `target` the
        staged state on its target epoch (entry structure identities are the epoch
        IDs). The transport carries this transition's own content evidence and
        succeeds only when `target` is the transition image of `source` within
        roundoff, so a staged value cannot borrow another route's evidence.
        """

        # Lazy: the lifecycle package sits above the discretization owners.
        from ..lifecycle import CompositionEntry, CompositionTransport

        if not isinstance(source, CompositionEntry) or not isinstance(
            target, CompositionEntry
        ):
            raise TypeError("Composition transports bind CompositionEntry values.")
        if (
            source.structure_id != self.source.epoch_id
            or target.structure_id != self.target.epoch_id
        ):
            raise ValueError(
                "Composition entries do not live on this transition's topology epochs."
            )
        result = self.apply(source.value)
        staged = jnp.asarray(target.value).reshape(-1)
        if staged.shape != result.values.shape:
            raise ValueError("Staged target does not match the target field space.")
        # The staged image must match within roundoff of the transported values.
        finfo = jnp.finfo(jnp.real(result.values).dtype)
        scale = jnp.maximum(jnp.max(jnp.abs(result.values)), finfo.tiny)
        image = jnp.max(jnp.abs(staged - result.values)) <= 100 * finfo.eps * scale
        return CompositionTransport(
            "physical-remap",
            (source.entry_id,),
            (target,),
            source_structure_ids=(self.source.epoch_id,),
            route_id=self.transition_id,
            successful=result.successful & image,
            source_content=result.source_content[None],
            target_content=result.target_content[None],
            content_tolerance=result.content_tolerance[None],
        )


__all__ = [
    "TopologyEpoch",
    "TopologyEpochTransition",
    "TopologyEpochTransitionResult",
]
