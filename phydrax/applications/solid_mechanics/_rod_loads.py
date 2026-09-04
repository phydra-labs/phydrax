#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ._rod_dynamics import PreparedRod, RodState


RodForceFrame: TypeAlias = Literal["world"]
RodMomentFrame: TypeAlias = Literal["material"]


def _identifier(value: str, owner: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{owner} must be nonempty.")
    return identifier


def _real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class RodLoad(StrictModule, NonTrainableState):
    """One source-resolved native effort in world-force/material-moment frames."""

    forces: Array
    moments: Array
    source_id: str = eqx.field(static=True)
    power_channel: str = eqx.field(static=True)
    force_frame: RodForceFrame = eqx.field(static=True)
    moment_frame: RodMomentFrame = eqx.field(static=True)
    force_unit: str = eqx.field(static=True)
    moment_unit: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    load_id: str = eqx.field(static=True)

    def __init__(
        self,
        forces: ArrayLike,
        moments: ArrayLike,
        /,
        *,
        source_id: str,
        power_channel: str,
        force_frame: RodForceFrame = "world",
        moment_frame: RodMomentFrame = "material",
        force_unit: str = "N",
        moment_unit: str = "N*m",
    ):
        forces_ = _real_array("forces", forces, 2)
        if forces_.shape[1] not in (2, 3) or forces_.shape[0] < 2:
            raise ValueError("Rod forces must have shape (nodes, 2|3).")
        dimension = int(forces_.shape[1])
        segment_count = int(forces_.shape[0] - 1)
        moment_rank = 1 if dimension == 2 else 2
        moments_ = _real_array("moments", moments, moment_rank)
        expected_moments = (segment_count,) if dimension == 2 else (segment_count, 3)
        if moments_.shape != expected_moments:
            raise ValueError(
                "Rod moments must have shape (segments,) in 2-D or (segments, 3) in 3-D."
            )
        if forces_.dtype != moments_.dtype:
            raise TypeError("Rod forces and moments must share a dtype.")
        if force_frame != "world":
            raise ValueError("Native rod nodal forces must use the world frame.")
        if moment_frame != "material":
            raise ValueError("Native rod segment moments must use the material frame.")
        source = _identifier(source_id, "source_id")
        channel = _identifier(power_channel, "power_channel")
        force_unit_ = _identifier(force_unit, "force_unit")
        moment_unit_ = _identifier(moment_unit, "moment_unit")
        self.forces = jnp.asarray(forces_)
        self.moments = jnp.asarray(moments_)
        self.source_id = source
        self.power_channel = channel
        self.force_frame = force_frame
        self.moment_frame = moment_frame
        self.force_unit = force_unit_
        self.moment_unit = moment_unit_
        self.dimension = dimension
        self.node_count = int(forces_.shape[0])
        self.segment_count = segment_count
        self.load_id = canonical_fingerprint(
            {
                "kind": "native-rod-load",
                "source_id": source,
                "power_channel": channel,
                "frames": {"force": force_frame, "moment": moment_frame},
                "units": {"force": force_unit_, "moment": moment_unit_},
                "content": array_tree_fingerprint(
                    {"forces": forces_, "moments": moments_}
                ),
            }
        )


class RodLoadPowerEvidence(StrictModule):
    """Source/channel power ledger certified by the native algebraic dual pairing."""

    source_power: Array
    channel_power: Array
    total_power: Array
    paired_power: Array
    absolute_pairing_error: Array
    finite: Array
    valid: Array
    source_ids: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def power_for_source(self, source_id: str, /) -> Array:
        source = _identifier(source_id, "source_id")
        try:
            index = self.source_ids.index(source)
        except ValueError as error:
            raise KeyError(f"Unknown rod load source {source!r}.") from error
        return self.source_power[index]

    def power_for_channel(self, channel: str, /) -> Array:
        name = _identifier(channel, "power channel")
        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown rod power channel {name!r}.") from error
        return self.channel_power[index]


class RodLoadLedger(StrictModule, NonTrainableState):
    """Ordered source ledger preserving frames, units, and named power channels."""

    loads: tuple[RodLoad, ...]
    source_ids: tuple[str, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    force_frame: RodForceFrame = eqx.field(static=True)
    moment_frame: RodMomentFrame = eqx.field(static=True)
    force_unit: str = eqx.field(static=True)
    moment_unit: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def __init__(self, loads: Sequence[RodLoad], /):
        loads_ = tuple(loads)
        if not loads_ or any(not isinstance(load, RodLoad) for load in loads_):
            raise TypeError("loads must contain one or more RodLoad values.")
        source_ids = tuple(load.source_id for load in loads_)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Rod load source IDs must be unique within a ledger.")
        contract = (
            loads_[0].dimension,
            loads_[0].node_count,
            loads_[0].segment_count,
            loads_[0].force_frame,
            loads_[0].moment_frame,
            loads_[0].force_unit,
            loads_[0].moment_unit,
            np.dtype(loads_[0].forces.dtype),
        )
        for load in loads_[1:]:
            candidate = (
                load.dimension,
                load.node_count,
                load.segment_count,
                load.force_frame,
                load.moment_frame,
                load.force_unit,
                load.moment_unit,
                np.dtype(load.forces.dtype),
            )
            if candidate != contract:
                raise ValueError(
                    "Every rod load in a ledger must share shape, dtype, frame, and units."
                )
        channels = tuple(dict.fromkeys(load.power_channel for load in loads_))
        self.loads = loads_
        self.source_ids = source_ids
        self.channel_names = channels
        self.dimension = contract[0]
        self.node_count = contract[1]
        self.segment_count = contract[2]
        self.force_frame = contract[3]
        self.moment_frame = contract[4]
        self.force_unit = contract[5]
        self.moment_unit = contract[6]
        self.ledger_id = canonical_fingerprint(
            {
                "kind": "native-rod-load-ledger",
                "loads": [load.load_id for load in loads_],
                "sources": list(source_ids),
                "channels": list(channels),
                "frames": {"force": contract[3], "moment": contract[4]},
                "units": {"force": contract[5], "moment": contract[6]},
            }
        )

    def _validate_rod(self, prepared: PreparedRod, /) -> None:
        from ._rod_dynamics import PreparedRod

        if not isinstance(prepared, PreparedRod):
            raise TypeError("prepared must be a PreparedRod.")
        plan = prepared.plan
        if (
            plan.dimension != self.dimension
            or plan.node_count != self.node_count
            or plan.segment_count != self.segment_count
            or np.dtype(plan.rest_positions.dtype) != np.dtype(self.loads[0].forces.dtype)
        ):
            raise ValueError("Rod load ledger does not match the prepared rod.")

    def source_efforts(self, prepared: PreparedRod, /) -> tuple[tuple[Array, Array], ...]:
        self._validate_rod(prepared)
        return tuple(
            prepared.effort_from_load(load.forces, load.moments) for load in self.loads
        )

    def total_effort(self, prepared: PreparedRod, /) -> tuple[Array, Array]:
        efforts = self.source_efforts(prepared)
        total_forces = efforts[0][0]
        total_moments = efforts[0][1]
        for forces, moments in efforts[1:]:
            total_forces = total_forces + forces
            total_moments = total_moments + moments
        return prepared.effort_space.validate((total_forces, total_moments))

    def power_evidence(
        self, prepared: PreparedRod, velocity: tuple[ArrayLike, ArrayLike], /
    ) -> RodLoadPowerEvidence:
        self._validate_rod(prepared)
        velocity_ = prepared.velocity_space.validate(velocity)
        efforts = self.source_efforts(prepared)
        direct = jnp.stack(
            tuple(
                jnp.sum(load.forces * velocity_[0]) + jnp.sum(load.moments * velocity_[1])
                for load in self.loads
            )
        )
        channel_power = jnp.stack(
            tuple(
                jnp.sum(
                    jnp.stack(
                        tuple(
                            direct[index]
                            for index, load in enumerate(self.loads)
                            if load.power_channel == channel
                        )
                    )
                )
                for channel in self.channel_names
            )
        )
        total_power = jnp.sum(direct)
        paired_power = prepared.effort_space.pair(self.total_effort(prepared), velocity_)
        error = jnp.abs(total_power - paired_power)
        finite = (
            jnp.all(jnp.isfinite(direct))
            & jnp.all(jnp.isfinite(channel_power))
            & jnp.isfinite(total_power)
            & jnp.isfinite(paired_power)
            & jnp.isfinite(error)
        )
        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=error.dtype),
            jnp.maximum(jnp.abs(total_power), jnp.abs(paired_power)),
        )
        valid = finite & (error <= 64.0 * jnp.finfo(error.dtype).eps * scale)
        return RodLoadPowerEvidence(
            direct,
            channel_power,
            total_power,
            paired_power,
            error,
            finite,
            valid,
            self.source_ids,
            self.channel_names,
            self.ledger_id,
        )

    def power_from_state(
        self, prepared: PreparedRod, state: RodState, /
    ) -> RodLoadPowerEvidence:
        return self.power_evidence(prepared, prepared.velocity_from_state(state))


class ReducedRodLoadBundle(StrictModule, NonTrainableState):
    """Source-aligned reduced efforts produced by a reduction-specific pullback."""

    ledger: RodLoadLedger
    source_efforts: Array
    reduction_id: str = eqx.field(static=True)
    reduced_size: int = eqx.field(static=True)
    effort_unit: str = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        ledger: RodLoadLedger,
        source_efforts: ArrayLike,
        reduction_id: str,
        /,
        *,
        effort_unit: str = "dual-to-reduced-coordinate-rate",
    ):
        if not isinstance(ledger, RodLoadLedger):
            raise TypeError("ledger must be a RodLoadLedger.")
        efforts = _real_array("source_efforts", source_efforts, 2)
        if efforts.shape[0] != len(ledger.loads) or efforts.shape[1] < 1:
            raise ValueError(
                "Reduced source efforts must have one nonempty row per native source."
            )
        if efforts.dtype != np.asarray(ledger.loads[0].forces).dtype:
            raise TypeError("Reduced and native rod efforts must share a dtype.")
        reduction = _identifier(reduction_id, "reduction_id")
        unit = _identifier(effort_unit, "effort_unit")
        self.ledger = ledger
        self.source_efforts = jnp.asarray(efforts)
        self.reduction_id = reduction
        self.reduced_size = int(efforts.shape[1])
        self.effort_unit = unit
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-load-bundle",
                "ledger": ledger.ledger_id,
                "reduction": reduction,
                "effort_unit": unit,
                "source_efforts": array_tree_fingerprint(efforts),
            }
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.ledger.source_ids

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.ledger.channel_names

    def total_effort(self, /) -> Array:
        return jnp.sum(self.source_efforts, axis=0)

    def effort_for_source(self, source_id: str, /) -> Array:
        source = _identifier(source_id, "source_id")
        try:
            index = self.source_ids.index(source)
        except ValueError as error:
            raise KeyError(f"Unknown rod load source {source!r}.") from error
        return self.source_efforts[index]

    def effort_for_channel(self, channel: str, /) -> Array:
        name = _identifier(channel, "power channel")
        if name not in self.channel_names:
            raise KeyError(f"Unknown rod power channel {name!r}.")
        indices = tuple(
            index
            for index, load in enumerate(self.ledger.loads)
            if load.power_channel == name
        )
        return jnp.sum(self.source_efforts[jnp.asarray(indices, dtype=jnp.int32)], axis=0)


__all__ = [
    "ReducedRodLoadBundle",
    "RodForceFrame",
    "RodLoad",
    "RodLoadLedger",
    "RodLoadPowerEvidence",
    "RodMomentFrame",
]
