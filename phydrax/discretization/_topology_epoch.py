#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._transfer import FieldTransfer


class TopologyEpoch(StrictModule, NonTrainableState):
    """Canonical identity of one realized geometry/topology/partition epoch."""

    index: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self, index: int, geometry_id: str, topology_id: str, partition_id: str, /
    ):
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
    values: Array
    source_content: Array
    target_content: Array
    conservation_residual: Array
    successful: Array
    differentiation_available: Array


class TopologyEpochTransition(StrictModule, NonTrainableState):
    """Explicit fixed transfer between two nondifferentiable topology epochs."""

    source: TopologyEpoch
    target: TopologyEpoch
    transfer: FieldTransfer
    source_measures: Array
    target_measures: Array
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TopologyEpoch,
        target: TopologyEpoch,
        transfer: FieldTransfer,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
    ):
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
        self.source, self.target, self.transfer = source, target, transfer
        self.source_measures, self.target_measures = (
            jnp.asarray(source_measure),
            jnp.asarray(target_measure),
        )
        self.transition_id = canonical_fingerprint(
            {
                "kind": "topology-epoch-transition",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "transfer": transfer.transfer_id,
                "source_measures": source_measure,
                "target_measures": target_measure,
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
        scale = jnp.maximum(jnp.abs(source_content), 1.0)
        tolerance = 100 * jnp.finfo(jnp.real(result).dtype).eps * scale
        successful = jnp.all(jnp.isfinite(result)) & (jnp.abs(residual) <= tolerance)
        return TopologyEpochTransitionResult(
            result,
            source_content,
            target_content,
            residual,
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


__all__ = [
    "TopologyEpoch",
    "TopologyEpochTransition",
    "TopologyEpochTransitionResult",
]
