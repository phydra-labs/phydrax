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
from .._validation import canonical_identifier
from ..linalg import AbstractLinearOperator
from ..typing import checked
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


def _require_epoch_binding(
    source: TopologyEpoch, target: TopologyEpoch, transfer: FieldTransfer, /
) -> None:
    """Admit only the scientific endpoints owned by the prepared field action."""
    binding = transfer.geometry
    if binding is None:
        raise ValueError("An epoch transition requires actual topology/geometry binding.")
    if (
        source.topology_id != binding.source_topology_id
        or target.topology_id != binding.target_topology_id
    ):
        raise ValueError(
            "Epoch topology identities do not match the prepared field transfer."
        )
    if (
        source.geometry_id != binding.source_geometry_id
        or target.geometry_id != binding.target_geometry_id
    ):
        raise ValueError(
            "Epoch geometry identities do not match the prepared field transfer."
        )


def _staged_image_matches(staged: Array, values: Array, /) -> Array:
    """Whether a staged target is the transition image within storage roundoff.

    The admissible difference is the roundoff of the coarser of the staged
    storage dtype and the transfer dtype, so state stored in lower precision than
    the transfer space is compared at the precision it can represent.
    """

    eps = max(
        float(jnp.finfo(jnp.real(staged).dtype).eps),
        float(jnp.finfo(jnp.real(values).dtype).eps),
    )
    tiny = float(jnp.finfo(jnp.real(values).dtype).tiny)
    scale = jnp.maximum(jnp.max(jnp.abs(values)), tiny)
    return jnp.max(jnp.abs(staged - values)) <= 100 * eps * scale


class TopologyEpochTransitionResult(StrictModule):
    """Transferred values with their content ledger.

    ``content_tolerance`` is the admissible ``|conservation_residual|``: the
    roundoff of both content sums plus the transfer's certified measure defect
    acting on this field. ``differentiation_available`` refers to the epoch
    selection (geometry and topology), which is never differentiable;
    ``value_derivative_available`` states that the values map through the
    frozen linear transfer, whose JVP is the primal action and whose VJP is the
    coordinate dual pullback.
    """

    values: Array
    source_content: Array
    target_content: Array
    conservation_residual: Array
    content_tolerance: Array
    successful: Array
    differentiation_available: Array
    value_derivative_available: Array


class TopologyEpochTransition(StrictModule, NonTrainableState):
    """Explicit fixed transfer between two nondifferentiable topology epochs.

    ``measure_defect_bound`` is the owner-certified bound, per source DOF and in
    measure units, on ``|P^T target_measures - source_measures|`` of the
    transfer ``P``; the content of a field ``v`` then changes by at most
    ``sum(bound * |v|)`` beyond roundoff. ``None`` certifies exact conservation
    (only roundoff remains), as for nested or exactly normalized transfers.

    Values remain differentiable across the frozen transfer: ``pullback`` is
    the coordinate dual ``P^T`` (the VJP of ``apply``) and ``adjoint`` the
    Hilbert adjoint in the source/target field-space pairings. Selection of
    the epoch itself has no derivative.
    """

    source: TopologyEpoch
    target: TopologyEpoch
    transfer: FieldTransfer
    source_measures: Array
    target_measures: Array
    measure_defect_bound: Array
    transition_id: str = eqx.field(static=True)

    @checked
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
        if target.index != source.index + 1 or source.epoch_id == target.epoch_id:
            raise ValueError(
                "Topology transitions must connect consecutive distinct epochs."
            )
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
        _require_epoch_binding(source, target, transfer)
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
            jnp.asarray(True),
        )

    def _reverse(
        self, operator: AbstractLinearOperator | None, value: ArrayLike
    ) -> Array:
        if operator is None:
            raise RuntimeError("Topology transition lost a required reverse operator.")
        flat = jnp.asarray(value).reshape(-1)
        if flat.shape != (operator.source.size,):
            raise ValueError("Topology transition cotangent does not match target space.")
        return operator.target.flatten(operator.mv(operator.source.unflatten(flat)))

    def pullback(self, target_cotangent: ArrayLike, /) -> Array:
        """Coordinate dual ``P^T w``: the VJP of ``apply`` for a target covector."""
        return self._reverse(self.transfer.dual_pullback_operator, target_cotangent)

    def adjoint(self, target_value: ArrayLike, /) -> Array:
        """Hilbert adjoint ``P^*`` in the field-space pairings of both epochs."""
        return self._reverse(self.transfer.hilbert_adjoint_operator, target_value)

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
        image = _staged_image_matches(staged, result.values)
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


class FieldEpochTransitionResult(StrictModule):
    """Transferred values of one non-conservative epoch transition.

    ``successful`` combines the owner's certified transfer evidence with the
    finiteness of the transferred values.
    """

    values: Array
    successful: Array
    differentiation_available: Array


class FieldEpochTransition(StrictModule, NonTrainableState):
    """Explicit fixed non-conservative field transfer between two topology epochs.

    The transfer carries checked semantics (interpolation, compatible Piola
    transfer, projection) rather than a content ledger: its evidence is the
    owner's certificate ``evidence_passed`` (reproduction, continuity, and
    commuting defects within tolerance). Conservative transfers form a
    :class:`TopologyEpochTransition`, whose content ledger this class never
    replaces.
    """

    source: TopologyEpoch
    target: TopologyEpoch
    transfer: FieldTransfer
    evidence_passed: bool = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TopologyEpoch,
        target: TopologyEpoch,
        transfer: FieldTransfer,
        /,
        *,
        evidence_passed: bool,
        evidence_id: str,
    ) -> None:
        if not isinstance(source, TopologyEpoch) or not isinstance(target, TopologyEpoch):
            raise TypeError("Field transition endpoints must be TopologyEpoch values.")
        if target.index != source.index + 1 or source.epoch_id == target.epoch_id:
            raise ValueError(
                "Field transitions must connect consecutive distinct epochs."
            )
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("Field transition requires FieldTransfer.")
        if not isinstance(evidence_passed, bool):
            raise TypeError("evidence_passed must be an explicit host bool.")
        properties = transfer.properties
        if (
            properties.semantics == "unspecified"
            or properties.conservative
            or properties.differentiable_geometry
            or transfer.hilbert_adjoint_operator is None
            or transfer.dual_pullback_operator is None
        ):
            raise ValueError(
                "A field epoch transition needs declared non-conservative semantics, "
                "dual/adjoint operators, and nondifferentiable geometry; conservative "
                "transfers form a TopologyEpochTransition."
            )
        evidence = canonical_identifier(evidence_id, "evidence_id")
        _require_epoch_binding(source, target, transfer)
        self.source, self.target, self.transfer = source, target, transfer
        self.evidence_passed = evidence_passed
        self.transition_id = canonical_fingerprint(
            {
                "kind": "field-epoch-transition",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "transfer": transfer.transfer_id,
                "evidence": evidence,
                "evidence_passed": evidence_passed,
            }
        )

    def apply(self, values: ArrayLike, /) -> FieldEpochTransitionResult:
        flat = jnp.asarray(values).reshape(-1)
        source_space = self.transfer.primal_operator.source
        target_space = self.transfer.primal_operator.target
        if flat.shape != (source_space.size,):
            raise ValueError("Field transition values do not match the source space.")
        result = target_space.flatten(
            self.transfer.primal_operator.mv(source_space.unflatten(flat))
        )
        successful = jnp.asarray(self.evidence_passed) & jnp.all(jnp.isfinite(result))
        return FieldEpochTransitionResult(result, successful, jnp.asarray(False))

    def transpose(self, target_cotangent: ArrayLike, /) -> Array:
        """Pull a target cotangent back through the declared Hilbert adjoint."""

        flat = jnp.asarray(target_cotangent).reshape(-1)
        adjoint = self.transfer.hilbert_adjoint_operator
        if adjoint is None:
            raise RuntimeError("Field transition lost its required Hilbert adjoint.")
        if flat.shape != (adjoint.source.size,):
            raise ValueError("Field transition cotangent does not match target space.")
        return adjoint.target.flatten(adjoint.mv(adjoint.source.unflatten(flat)))

    def require_differentiable_topology(self) -> None:
        raise ValueError(
            "Topology selection is nondifferentiable; differentiate only within one fixed epoch."
        )

    def composition_transport(
        self, source: CompositionEntry, target: CompositionEntry, /
    ) -> CompositionTransport:
        """Physical-remap evidence of this transition for one composition state entry.

        Entry structure identities are the epoch IDs. The transport reports no
        content (the transfer is not conservative) and succeeds only when the
        owner evidence passed and `target` is the transition image of `source`
        within roundoff.
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
        image = _staged_image_matches(staged, result.values)
        return CompositionTransport(
            "physical-remap",
            (source.entry_id,),
            (target,),
            source_structure_ids=(self.source.epoch_id,),
            route_id=self.transition_id,
            successful=result.successful & image,
        )


__all__ = [
    "FieldEpochTransition",
    "FieldEpochTransitionResult",
    "TopologyEpoch",
    "TopologyEpochTransition",
    "TopologyEpochTransitionResult",
]
