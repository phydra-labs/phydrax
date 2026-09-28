#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-only exchange between resolved and reduced bubble models.

The records in this module are host-side, immutable evidence.  Converters read
committed result objects and never modify either solver.  In particular, an
accepted request is not a solver handoff: flow-field initialization/removal is
always left as an explicitly missing invariant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum, IntFlag
from numbers import Integral
from typing import Literal

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import canonical_identifier, nonnegative_integer
from ...bubble_dynamics import (
    BubbleCloudResult,
    BubbleDynamicsStatus,
    BubbleEnvironment,
    BubbleEventKind,
    SingleBubbleResult,
)
from ...typing import parse
from ._bubble_components import (
    ATMOSPHERE_ID,
    BubbleTransitionKind,
    BubbleTransitionRecord,
)
from ._bubble_thermodynamics import (
    BubbleCompartmentEvaluation,
    BubbleCompartmentState,
)


type ReducedBubbleRequestReason = Literal["vanish", "entrain", "support_loss"]
type ReducedBubbleFailureReason = Literal["overlap", "support_loss"]


class BubbleExchangeInvariant(IntFlag):
    """Invariant evidence absent from an exchange record or request."""

    NONE = 0
    GAS_AMOUNT = 1 << 0
    INTERNAL_ENERGY = 1 << 1
    LAW_STATE = 1 << 2
    LAW_SUPPORT = 1 << 3
    TRANSLATIONAL_MOMENTUM = 1 << 4
    LIQUID_IMPULSE = 1 << 5
    AMBIENT_STATE = 1 << 6
    FLOW_FIELD_INITIALIZATION = 1 << 7
    FLOW_FIELD_REMOVAL = 1 << 8


class BubbleExchangeStatus(IntEnum):
    """Completeness of evidence, never readiness for an automatic handoff."""

    EVIDENCE_ONLY = 0
    HANDOFF_INCOMPLETE = 1


_ALL_INVARIANTS = (
    BubbleExchangeInvariant.GAS_AMOUNT
    | BubbleExchangeInvariant.INTERNAL_ENERGY
    | BubbleExchangeInvariant.LAW_STATE
    | BubbleExchangeInvariant.LAW_SUPPORT
    | BubbleExchangeInvariant.TRANSLATIONAL_MOMENTUM
    | BubbleExchangeInvariant.LIQUID_IMPULSE
    | BubbleExchangeInvariant.AMBIENT_STATE
    | BubbleExchangeInvariant.FLOW_FIELD_INITIALIZATION
    | BubbleExchangeInvariant.FLOW_FIELD_REMOVAL
)


def _finite_scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> float:
    host = np.asarray(value)
    if host.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    result = float(host)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _canonical_ids(
    values: Sequence[int], name: str, /, *, positive: bool
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    checked: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integers.")
        identifier = int(value)
        if identifier < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must contain {qualifier} identifiers.")
        checked.append(identifier)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(checked))


def _finite_tuple(
    values: Sequence[float], name: str, /, *, sizes: tuple[int, ...] | None = None
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real values.")
    host = np.asarray(tuple(values), dtype=np.float64)
    if host.ndim != 1 or (sizes is not None and host.shape[0] not in sizes):
        expected = "one-dimensional" if sizes is None else f"of size {sizes}"
        raise ValueError(f"{name} must be {expected}.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return tuple(float(value) for value in host)


def _optional_vector(
    value: Sequence[float] | None, name: str, dimension: int, /
) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _finite_tuple(value, name, sizes=(dimension,))


def _ambient_state(
    environment: BubbleEnvironment | None, /
) -> tuple[float, float, float] | None:
    if environment is None:
        return None
    if not isinstance(environment, BubbleEnvironment):
        raise TypeError("environment must be a BubbleEnvironment or None.")
    pressure = _finite_scalar(environment.ambient_pressure, "ambient pressure")
    temperature = _finite_scalar(
        environment.ambient_temperature, "ambient temperature", positive=True
    )
    vapor = _finite_scalar(environment.vapor_pressure, "vapor pressure")
    if pressure < 0.0 or vapor < 0.0:
        raise ValueError("Ambient and vapor pressures must be nonnegative.")
    return pressure, temperature, vapor


def _missing_mask(value: BubbleExchangeInvariant, /) -> BubbleExchangeInvariant:
    if not isinstance(value, BubbleExchangeInvariant):
        raise TypeError("missing_invariants must be a BubbleExchangeInvariant mask.")
    if int(value) & ~int(_ALL_INVARIANTS):
        raise ValueError("missing_invariants contains an unknown invariant bit.")
    return value


class ResolvedBubbleEvidenceRecord(StrictModule):
    """One resolved-compartment state attached to a canonical lineage event.

    ``translational_momentum`` and ``liquid_impulse`` are optional evidence,
    not inferred values.  A literal zero vector is available evidence only when
    the caller explicitly supplies it.  ``FLOW_FIELD_REMOVAL`` remains missing
    because this record cannot remove or initialize physical flow state.
    """

    bubble_id: int = eqx.field(static=True)
    parent_ids: tuple[int, ...] = eqx.field(static=True)
    child_ids: tuple[int, ...] = eqx.field(static=True)
    event_kind: BubbleTransitionKind = eqx.field(static=True)
    gas_amount: float = eqx.field(static=True)
    internal_energy: float = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    law_state: tuple[float, ...] = eqx.field(static=True)
    law_supported: bool | None = eqx.field(static=True)
    equivalent_volume_radius: float = eqx.field(static=True)
    centroid: tuple[float, ...] = eqx.field(static=True)
    translational_momentum: tuple[float, ...] | None = eqx.field(static=True)
    liquid_impulse: tuple[float, ...] | None = eqx.field(static=True)
    ambient_state: tuple[float, float, float] | None = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    missing_invariants: BubbleExchangeInvariant = eqx.field(static=True)
    status: BubbleExchangeStatus = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        bubble_id: int,
        event_kind: BubbleTransitionKind,
        /,
        *,
        parent_ids: Sequence[int],
        child_ids: Sequence[int],
        gas_amount: ArrayLike,
        internal_energy: ArrayLike,
        law_id: str,
        law_state: Sequence[float],
        law_supported: bool | None,
        equivalent_volume_radius: ArrayLike,
        centroid: Sequence[float],
        translational_momentum: Sequence[float] | None,
        liquid_impulse: Sequence[float] | None,
        ambient_state: tuple[float, float, float] | None,
        epoch: int,
        time: ArrayLike,
        source_realization_id: str,
    ) -> None:
        if isinstance(bubble_id, bool) or not isinstance(bubble_id, Integral):
            raise TypeError("bubble_id must be an integer.")
        identifier = int(bubble_id)
        if identifier <= ATMOSPHERE_ID:
            raise ValueError("bubble_id must identify a closed bubble.")
        kind = parse(event_kind, BubbleTransitionKind, "event_kind")
        parents = _canonical_ids(parent_ids, "parent_ids", positive=False)
        children = _canonical_ids(child_ids, "child_ids", positive=False)
        if identifier not in parents and identifier not in children:
            raise ValueError("bubble_id must occur in the event lineage.")
        amount = _finite_scalar(gas_amount, "gas_amount", positive=True)
        energy = _finite_scalar(internal_energy, "internal_energy")
        if energy < 0.0:
            raise ValueError("internal_energy must be nonnegative.")
        law = canonical_identifier(law_id, "law_id")
        internal = _finite_tuple(law_state, "law_state")
        if law_supported is not None and not isinstance(law_supported, bool):
            raise TypeError("law_supported must be bool or None.")
        radius = _finite_scalar(
            equivalent_volume_radius, "equivalent_volume_radius", positive=True
        )
        center = _finite_tuple(centroid, "centroid", sizes=(2, 3))
        momentum = _optional_vector(
            translational_momentum, "translational_momentum", len(center)
        )
        impulse = _optional_vector(liquid_impulse, "liquid_impulse", len(center))
        if ambient_state is None:
            ambient = None
        else:
            ambient_values = _finite_tuple(ambient_state, "ambient_state", sizes=(3,))
            ambient = (ambient_values[0], ambient_values[1], ambient_values[2])
        if ambient is not None and (
            ambient[0] < 0.0 or ambient[1] <= 0.0 or ambient[2] < 0.0
        ):
            raise ValueError(
                "ambient_state must contain nonnegative pressures and positive temperature."
            )
        event_epoch = nonnegative_integer(epoch, "epoch")
        event_time = _finite_scalar(time, "time")
        if event_time < 0.0:
            raise ValueError("time must be nonnegative.")
        realization = canonical_identifier(source_realization_id, "source_realization_id")
        missing = BubbleExchangeInvariant.FLOW_FIELD_REMOVAL
        if law_supported is not True:
            missing |= BubbleExchangeInvariant.LAW_SUPPORT
        if momentum is None:
            missing |= BubbleExchangeInvariant.TRANSLATIONAL_MOMENTUM
        if impulse is None:
            missing |= BubbleExchangeInvariant.LIQUID_IMPULSE
        if ambient is None:
            missing |= BubbleExchangeInvariant.AMBIENT_STATE
        status = (
            BubbleExchangeStatus.EVIDENCE_ONLY
            if missing == BubbleExchangeInvariant.FLOW_FIELD_REMOVAL
            else BubbleExchangeStatus.HANDOFF_INCOMPLETE
        )
        self.bubble_id = identifier
        self.parent_ids = parents
        self.child_ids = children
        self.event_kind = kind
        self.gas_amount = amount
        self.internal_energy = energy
        self.law_id = law
        self.law_state = internal
        self.law_supported = law_supported
        self.equivalent_volume_radius = radius
        self.centroid = center
        self.translational_momentum = momentum
        self.liquid_impulse = impulse
        self.ambient_state = ambient
        self.epoch = event_epoch
        self.time = event_time
        self.source_realization_id = realization
        self.missing_invariants = missing
        self.status = status
        self.record_id = canonical_fingerprint(
            {
                "kind": "resolved-bubble-evidence",
                "bubble_id": identifier,
                "parents": parents,
                "children": children,
                "event": kind,
                "gas_amount": amount,
                "internal_energy": energy,
                "law_id": law,
                "law_state": internal,
                "law_supported": law_supported,
                "equivalent_volume_radius": radius,
                "centroid": center,
                "translational_momentum": momentum,
                "liquid_impulse": impulse,
                "ambient_state": ambient,
                "epoch": event_epoch,
                "time": event_time,
                "source_realization_id": realization,
                "missing_invariants": int(missing),
            }
        )

    @property
    def handoff_ready(self) -> bool:
        """Always false: evidence does not implement a coupled-state transaction."""

        return False


class ReducedBubbleRequestRecord(StrictModule):
    """Evidence-backed request for a reduced model; no model is constructed."""

    reason: ReducedBubbleRequestReason = eqx.field(static=True)
    desired_reduced_model: str = eqx.field(static=True)
    desired_equation: str = eqx.field(static=True)
    bubble_ids: tuple[int, ...] = eqx.field(static=True)
    parent_ids: tuple[int, ...] = eqx.field(static=True)
    child_ids: tuple[int, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    missing_invariants: BubbleExchangeInvariant = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        reason: ReducedBubbleRequestReason,
        desired_reduced_model: str,
        desired_equation: str,
        /,
        *,
        bubble_ids: Sequence[int],
        parent_ids: Sequence[int],
        child_ids: Sequence[int],
        evidence_ids: Sequence[str],
        missing_invariants: BubbleExchangeInvariant,
        accepted: bool = False,
        epoch: int,
        time: ArrayLike,
        source_realization_id: str,
    ) -> None:
        selected = parse(reason, ReducedBubbleRequestReason, "reason")
        model = canonical_identifier(desired_reduced_model, "desired_reduced_model")
        equation = canonical_identifier(desired_equation, "desired_equation")
        bubbles = _canonical_ids(bubble_ids, "bubble_ids", positive=True)
        if not bubbles:
            raise ValueError("bubble_ids must not be empty.")
        parents = _canonical_ids(parent_ids, "parent_ids", positive=False)
        children = _canonical_ids(child_ids, "child_ids", positive=False)
        ids = tuple(
            sorted(canonical_identifier(value, "evidence_id") for value in evidence_ids)
        )
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("evidence_ids must be non-empty and unique.")
        missing = _missing_mask(missing_invariants)
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be bool.")
        event_epoch = nonnegative_integer(epoch, "epoch")
        event_time = _finite_scalar(time, "time")
        if event_time < 0.0:
            raise ValueError("time must be nonnegative.")
        realization = canonical_identifier(source_realization_id, "source_realization_id")
        self.reason = selected
        self.desired_reduced_model = model
        self.desired_equation = equation
        self.bubble_ids = bubbles
        self.parent_ids = parents
        self.child_ids = children
        self.evidence_ids = ids
        self.missing_invariants = missing
        self.accepted = accepted
        self.epoch = event_epoch
        self.time = event_time
        self.source_realization_id = realization
        self.request_id = canonical_fingerprint(
            {
                "kind": "reduced-bubble-request",
                "reason": selected,
                "desired_reduced_model": model,
                "desired_equation": equation,
                "bubble_ids": bubbles,
                "parents": parents,
                "children": children,
                "evidence_ids": ids,
                "missing_invariants": int(missing),
                "accepted": accepted,
                "epoch": event_epoch,
                "time": event_time,
                "source_realization_id": realization,
            }
        )

    @property
    def handoff_ready(self) -> bool:
        """Always false: acceptance is not a coupled-state transaction."""

        return False


class ReducedBubbleFailureRecord(StrictModule):
    """Reduced-model failure that requests, but never creates, a resolved model."""

    reason: ReducedBubbleFailureReason = eqx.field(static=True)
    desired_resolved_model: str = eqx.field(static=True)
    bubble_ids: tuple[int, ...] = eqx.field(static=True)
    source_status: BubbleDynamicsStatus = eqx.field(static=True)
    source_event: BubbleEventKind = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    missing_invariants: BubbleExchangeInvariant = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    failure_id: str = eqx.field(static=True)

    def __init__(
        self,
        reason: ReducedBubbleFailureReason,
        desired_resolved_model: str,
        /,
        *,
        bubble_ids: Sequence[int],
        source_status: BubbleDynamicsStatus,
        source_event: BubbleEventKind,
        source_plan_id: str,
        source_realization_id: str,
        missing_invariants: BubbleExchangeInvariant,
        accepted: bool = False,
        epoch: int = 0,
        time: ArrayLike,
    ) -> None:
        selected = parse(reason, ReducedBubbleFailureReason, "reason")
        model = canonical_identifier(desired_resolved_model, "desired_resolved_model")
        bubbles = _canonical_ids(bubble_ids, "bubble_ids", positive=False)
        if not bubbles:
            raise ValueError("bubble_ids must not be empty.")
        if not isinstance(source_status, BubbleDynamicsStatus):
            raise TypeError("source_status must be a BubbleDynamicsStatus.")
        if not isinstance(source_event, BubbleEventKind):
            raise TypeError("source_event must be a BubbleEventKind.")
        if selected == "overlap" and source_status is not BubbleDynamicsStatus.OVERLAP:
            raise ValueError("An overlap failure requires OVERLAP source status.")
        if selected == "support_loss" and source_status not in (
            BubbleDynamicsStatus.SUPPORT_EXIT,
            BubbleDynamicsStatus.VALIDITY_EXCEEDED,
        ):
            raise ValueError("A support-loss failure requires a support source status.")
        plan = canonical_identifier(source_plan_id, "source_plan_id")
        realization = canonical_identifier(source_realization_id, "source_realization_id")
        missing = _missing_mask(missing_invariants)
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be bool.")
        event_epoch = nonnegative_integer(epoch, "epoch")
        event_time = _finite_scalar(time, "time")
        if event_time < 0.0:
            raise ValueError("time must be nonnegative.")
        self.reason = selected
        self.desired_resolved_model = model
        self.bubble_ids = bubbles
        self.source_status = source_status
        self.source_event = source_event
        self.source_plan_id = plan
        self.source_realization_id = realization
        self.missing_invariants = missing
        self.accepted = accepted
        self.epoch = event_epoch
        self.time = event_time
        self.failure_id = canonical_fingerprint(
            {
                "kind": "reduced-bubble-failure",
                "reason": selected,
                "desired_resolved_model": model,
                "bubble_ids": bubbles,
                "source_status": int(source_status),
                "source_event": int(source_event),
                "source_plan_id": plan,
                "source_realization_id": realization,
                "missing_invariants": int(missing),
                "accepted": accepted,
                "epoch": event_epoch,
                "time": event_time,
            }
        )

    @property
    def handoff_ready(self) -> bool:
        """Always false: the resolved flow field has not been initialized."""

        return False


def _compartment_arrays(state: BubbleCompartmentState, /) -> tuple[np.ndarray, ...]:
    if not isinstance(state, BubbleCompartmentState):
        raise TypeError("compartments must be a BubbleCompartmentState.")
    arrays = tuple(
        np.asarray(value)
        for value in (
            state.bubble_id,
            state.amount,
            state.internal_energy,
            state.internal,
            state.volume,
            state.pressure,
            state.centroid,
        )
    )
    ids, amount, energy, internal, volume, pressure, centroid = arrays
    capacity = ids.shape[0] if ids.ndim == 1 else -1
    if (
        capacity < 1
        or amount.shape != (capacity,)
        or energy.shape != (capacity,)
        or internal.ndim != 2
        or internal.shape[0] != capacity
        or volume.shape != (capacity,)
        or pressure.shape != (capacity,)
        or centroid.ndim != 2
        or centroid.shape[0] != capacity
        or centroid.shape[1] not in (2, 3)
    ):
        raise ValueError("The compartment state has inconsistent array shapes.")
    active = ids > ATMOSPHERE_ID
    active_ids = ids[active]
    if len(set(int(value) for value in active_ids)) != active_ids.shape[0]:
        raise ValueError("Active compartment bubble identifiers must be unique.")
    return arrays


def _selected_event_ids(
    transition: BubbleTransitionRecord, ids: np.ndarray, state_epoch: int, /
) -> tuple[int, ...]:
    active = {int(value) for value in ids if value > ATMOSPHERE_ID}
    parents = tuple(value for value in transition.parent_ids if value > ATMOSPHERE_ID)
    children = tuple(value for value in transition.child_ids if value > ATMOSPHERE_ID)
    active_parents = tuple(value for value in parents if value in active)
    active_children = tuple(value for value in children if value in active)
    if active_parents and active_children and set(active_parents) != set(active_children):
        raise ValueError("compartments must represent only one side of the transition.")
    if active_children:
        if state_epoch != transition.epoch:
            raise ValueError("Child-side compartments must have the transition epoch.")
        return tuple(sorted(active_children))
    if active_parents:
        if state_epoch + 1 != transition.epoch:
            raise ValueError(
                "Parent-side compartments must precede the transition epoch."
            )
        return tuple(sorted(active_parents))
    raise ValueError("No transition bubble is present in compartments.")


def _mapping_vector(
    values: Mapping[int, ArrayLike | None] | None,
    bubble_id: int,
    dimension: int,
    name: str,
    /,
) -> tuple[float, ...] | None:
    if values is None or bubble_id not in values or values[bubble_id] is None:
        return None
    host = np.asarray(values[bubble_id], dtype=np.float64)
    if host.shape != (dimension,) or not np.all(np.isfinite(host)):
        raise ValueError(
            f"{name}[{bubble_id}] must be a finite vector of size {dimension}."
        )
    return tuple(float(value) for value in host)


def resolved_bubble_evidence_records(
    transition: BubbleTransitionRecord,
    compartments: BubbleCompartmentState,
    /,
    *,
    time: ArrayLike,
    source_realization_id: str,
    law_id: str,
    evaluation: BubbleCompartmentEvaluation | None = None,
    environment: BubbleEnvironment | None = None,
    translational_momentum: Mapping[int, ArrayLike | None] | None = None,
    liquid_impulse: Mapping[int, ArrayLike | None] | None = None,
) -> tuple[ResolvedBubbleEvidenceRecord, ...]:
    """Export one canonical record per event-side compartment.

    The state may be the parent side (epoch ``record.epoch - 1``) or child side
    (epoch ``record.epoch``), but never a mixture.  Momentum and liquid impulse
    are included only when explicitly supplied by bubble id.
    """

    if not isinstance(transition, BubbleTransitionRecord):
        raise TypeError("transition must be a BubbleTransitionRecord.")
    ids, amount, energy, internal, volume, _, centroid = _compartment_arrays(compartments)
    state_epoch = nonnegative_integer(int(np.asarray(compartments.epoch)), "state epoch")
    selected = _selected_event_ids(transition, ids, state_epoch)
    if evaluation is not None:
        if not isinstance(evaluation, BubbleCompartmentEvaluation):
            raise TypeError("evaluation must be a BubbleCompartmentEvaluation or None.")
        support = np.asarray(evaluation.admissible)
        if support.shape != ids.shape:
            raise ValueError("evaluation must align with compartments.")
    else:
        support = None
    ambient = _ambient_state(environment)
    law = canonical_identifier(law_id, "law_id")
    realization = canonical_identifier(source_realization_id, "source_realization_id")
    event_time = _finite_scalar(time, "time")
    if event_time < 0.0:
        raise ValueError("time must be nonnegative.")
    records = []
    for bubble_id in selected:
        matches = np.flatnonzero(ids == bubble_id)
        if matches.shape != (1,):
            raise ValueError("Each selected bubble must own exactly one compartment.")
        slot = int(matches[0])
        gas_amount = _finite_scalar(amount[slot], "gas amount", positive=True)
        internal_energy = _finite_scalar(energy[slot], "internal energy")
        gas_volume = _finite_scalar(volume[slot], "gas volume", positive=True)
        law_state = _finite_tuple(internal[slot], "law state")
        center = _finite_tuple(centroid[slot], "centroid", sizes=(2, 3))
        records.append(
            ResolvedBubbleEvidenceRecord(
                bubble_id,
                transition.kind,
                parent_ids=transition.parent_ids,
                child_ids=transition.child_ids,
                gas_amount=gas_amount,
                internal_energy=internal_energy,
                law_id=law,
                law_state=law_state,
                law_supported=None if support is None else bool(support[slot]),
                equivalent_volume_radius=(3.0 * gas_volume / (4.0 * np.pi))
                ** (1.0 / 3.0),
                centroid=center,
                translational_momentum=_mapping_vector(
                    translational_momentum,
                    bubble_id,
                    len(center),
                    "translational_momentum",
                ),
                liquid_impulse=_mapping_vector(
                    liquid_impulse, bubble_id, len(center), "liquid_impulse"
                ),
                ambient_state=ambient,
                epoch=transition.epoch,
                time=event_time,
                source_realization_id=realization,
            )
        )
    return tuple(records)


def reduced_bubble_request(
    evidence: ResolvedBubbleEvidenceRecord,
    desired_reduced_model: str,
    desired_equation: str,
    /,
    *,
    accepted: bool = False,
) -> ReducedBubbleRequestRecord:
    """Turn resolved evidence into an unexecuted reduced-model request."""

    if not isinstance(evidence, ResolvedBubbleEvidenceRecord):
        raise TypeError("evidence must be a ResolvedBubbleEvidenceRecord.")
    match evidence.event_kind:
        case "vanish":
            reason: ReducedBubbleRequestReason = "vanish"
        case "entrain":
            reason = "entrain"
        case _ if evidence.law_supported is False:
            reason = "support_loss"
        case _:
            raise ValueError(
                "Reduced requests require vanish, entrain, or failed law-support evidence."
            )
    return ReducedBubbleRequestRecord(
        reason,
        desired_reduced_model,
        desired_equation,
        bubble_ids=(evidence.bubble_id,),
        parent_ids=evidence.parent_ids,
        child_ids=evidence.child_ids,
        evidence_ids=(evidence.record_id,),
        missing_invariants=evidence.missing_invariants,
        accepted=accepted,
        epoch=evidence.epoch,
        time=evidence.time,
        source_realization_id=evidence.source_realization_id,
    )


def _gas_evidence_missing(
    result: BubbleCloudResult | SingleBubbleResult, /
) -> BubbleExchangeInvariant:
    missing = BubbleExchangeInvariant.NONE
    if isinstance(result, SingleBubbleResult):
        gases = (result.terminal_state.gas,)
    else:
        gases = tuple(group.gas for group in result.terminal_state.groups)
    amounts = [np.asarray(gas.amount) for gas in gases]
    if any(np.any(~np.isfinite(value)) or np.any(value <= 0.0) for value in amounts):
        missing |= BubbleExchangeInvariant.GAS_AMOUNT
    energies = [gas.internal_energy for gas in gases]
    if any(
        value is None or np.any(~np.isfinite(np.asarray(value))) for value in energies
    ):
        missing |= BubbleExchangeInvariant.INTERNAL_ENERGY
    internals = [np.asarray(gas.internal) for gas in gases]
    if any(np.any(~np.isfinite(value)) for value in internals):
        missing |= BubbleExchangeInvariant.LAW_STATE
    return missing


def reduced_bubble_failure_from_result(
    result: BubbleCloudResult | SingleBubbleResult,
    desired_resolved_model: str,
    /,
    *,
    source_realization_id: str,
    bubble_id: int | None = None,
    epoch: int = 0,
    accepted: bool = False,
) -> ReducedBubbleFailureRecord:
    """Convert an A overlap/support result into a resolved-model request only."""

    if not isinstance(result, (BubbleCloudResult, SingleBubbleResult)):
        raise TypeError("result must be a BubbleCloudResult or SingleBubbleResult.")
    try:
        status = BubbleDynamicsStatus(int(np.asarray(result.status)))
        event = BubbleEventKind(int(np.asarray(result.evidence.event_kind)))
    except ValueError as error:
        raise ValueError("The reduced result has an unknown status or event.") from error
    match status:
        case BubbleDynamicsStatus.OVERLAP:
            reason: ReducedBubbleFailureReason = "overlap"
        case BubbleDynamicsStatus.SUPPORT_EXIT | BubbleDynamicsStatus.VALIDITY_EXCEEDED:
            reason = "support_loss"
        case _:
            raise ValueError("Only overlap or support failures create resolved requests.")
    if isinstance(result, BubbleCloudResult):
        if bubble_id is not None:
            raise ValueError("bubble_id is inferred from a BubbleCloudResult.")
        bubble_ids = result.bubble_ids
    else:
        if bubble_id is None:
            raise ValueError("bubble_id is required for a SingleBubbleResult.")
        bubble_ids = (bubble_id,)
    missing = (
        _gas_evidence_missing(result)
        | BubbleExchangeInvariant.TRANSLATIONAL_MOMENTUM
        | BubbleExchangeInvariant.LIQUID_IMPULSE
        | BubbleExchangeInvariant.AMBIENT_STATE
        | BubbleExchangeInvariant.FLOW_FIELD_INITIALIZATION
    )
    if reason == "support_loss":
        missing |= BubbleExchangeInvariant.LAW_SUPPORT
    return ReducedBubbleFailureRecord(
        reason,
        desired_resolved_model,
        bubble_ids=bubble_ids,
        source_status=status,
        source_event=event,
        source_plan_id=result.plan_id,
        source_realization_id=source_realization_id,
        missing_invariants=missing,
        accepted=accepted,
        epoch=epoch,
        time=result.terminal_time,
    )


__all__ = [
    "BubbleExchangeInvariant",
    "BubbleExchangeStatus",
    "ReducedBubbleFailureReason",
    "ReducedBubbleFailureRecord",
    "ReducedBubbleRequestReason",
    "ReducedBubbleRequestRecord",
    "ResolvedBubbleEvidenceRecord",
    "reduced_bubble_failure_from_result",
    "reduced_bubble_request",
    "resolved_bubble_evidence_records",
]
