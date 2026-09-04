#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse motor-unit territories and explicit endplate stimulus routing."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class MotorUnitTerritoryEvidence(StrictModule, NonTrainableState):
    fiber_count_per_unit: Array
    every_fiber_assigned: Array
    every_unit_represented: Array
    endplates_in_bounds: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class MotorUnitEndplateStimulus(StrictModule, NonTrainableState):
    """Fixed-capacity event block routed sparsely to one node per fiber."""

    event_times_ms: Array
    event_mask: Array
    fiber_motor_unit_index: Array
    fiber_endplate_node: Array
    amplitude_uA_per_cm2: Array
    duration_ms: Array
    node_count: int = eqx.field(static=True)
    event_source_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    @property
    def fiber_count(self) -> int:
        return self.fiber_motor_unit_index.shape[0]

    @property
    def unit_count(self) -> int:
        return self.event_times_ms.shape[0]

    def current(self, time_ms: ArrayLike, /) -> Array:
        time = jnp.asarray(time_ms, dtype=self.event_times_ms.dtype)
        if time.shape != ():
            raise ValueError("time_ms must be scalar.")
        active = (
            self.event_mask
            & (time >= self.event_times_ms)
            & (
                time
                < self.event_times_ms + self.duration_ms[:, None]
            )
        )
        unit_current = jnp.sum(
            active * self.amplitude_uA_per_cm2[:, None], axis=1
        )
        fiber_current = unit_current[self.fiber_motor_unit_index]
        return jnp.zeros(
            (self.fiber_count, self.node_count), dtype=unit_current.dtype
        ).at[jnp.arange(self.fiber_count), self.fiber_endplate_node].set(fiber_current)

    def event_boundaries_ms(self, /) -> Array:
        masked_start = jnp.where(self.event_mask, self.event_times_ms, jnp.inf)
        masked_end = jnp.where(
            self.event_mask,
            self.event_times_ms + self.duration_ms[:, None],
            jnp.inf,
        )
        return jnp.sort(
            jnp.concatenate((masked_start.reshape(-1), masked_end.reshape(-1)))
        )


class MotorUnitTerritoryPlan(StrictModule, NonTrainableState):
    """Static sparse map from motor units to fibers and endplate nodes."""

    unit_ids: tuple[str, ...] = eqx.field(static=True)
    fiber_ids: tuple[str, ...] = eqx.field(static=True)
    fiber_motor_unit_index: Array
    fiber_endplate_node: Array
    node_count: int = eqx.field(static=True)
    amplitude_uA_per_cm2: Array
    duration_ms: Array
    stimulus_source_id: str = eqx.field(static=True)
    evidence: MotorUnitTerritoryEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        unit_ids: tuple[str, ...],
        fiber_ids: tuple[str, ...],
        fiber_motor_unit_index: ArrayLike,
        fiber_endplate_node: ArrayLike,
        node_count: int,
        amplitude_uA_per_cm2: ArrayLike,
        duration_ms: ArrayLike,
        /,
        *,
        stimulus_source_id: str,
    ):
        units = tuple(str(value).strip() for value in unit_ids)
        fibers = tuple(str(value).strip() for value in fiber_ids)
        if (
            not units
            or not fibers
            or any(not value for value in units + fibers)
            or len(set(units)) != len(units)
            or len(set(fibers)) != len(fibers)
        ):
            raise ValueError("Motor-unit and fiber IDs must be nonempty and unique.")
        if isinstance(node_count, bool) or not isinstance(node_count, Integral):
            raise TypeError("node_count must be an integer.")
        nodes = int(node_count)
        if nodes < 1:
            raise ValueError("node_count must be positive.")
        unit_index = jnp.asarray(fiber_motor_unit_index, dtype=jnp.int32)
        endplate = jnp.asarray(fiber_endplate_node, dtype=jnp.int32)
        amplitude = jnp.asarray(amplitude_uA_per_cm2)
        duration = jnp.asarray(duration_ms)
        if unit_index.shape != (len(fibers),) or endplate.shape != (len(fibers),):
            raise ValueError("Territory maps must contain one entry per fiber.")
        if amplitude.shape != (len(units),) or duration.shape != (len(units),):
            raise ValueError("Stimulus parameters must contain one value per motor unit.")
        source = str(stimulus_source_id).strip()
        if not source:
            raise ValueError("stimulus_source_id must be nonempty.")
        host_units = np.asarray(unit_index)
        host_endplates = np.asarray(endplate)
        if np.any(host_units < 0) or np.any(host_units >= len(units)):
            raise ValueError("fiber_motor_unit_index contains an out-of-range unit.")
        if np.any(host_endplates < 0) or np.any(host_endplates >= nodes):
            raise ValueError("fiber_endplate_node contains an out-of-range node.")
        if (
            not np.all(np.isfinite(np.asarray(amplitude)))
            or np.any(np.asarray(amplitude) <= 0.0)
            or not np.all(np.isfinite(np.asarray(duration)))
            or np.any(np.asarray(duration) <= 0.0)
        ):
            raise ValueError("Endplate amplitudes and durations must be positive and finite.")
        counts = jnp.bincount(unit_index, length=len(units))
        every_fiber = jnp.asarray(unit_index.shape[0] == len(fibers))
        every_unit = jnp.all(counts > 0)
        endplates_valid = jnp.all((endplate >= 0) & (endplate < nodes))
        plan_id = canonical_fingerprint(
            {
                "kind": "sparse-motor-unit-endplate-territory",
                "unit_ids": units,
                "fiber_ids": fibers,
                "unit_index": array_tree_fingerprint(unit_index),
                "endplate_node": array_tree_fingerprint(endplate),
                "node_count": nodes,
                "amplitude": array_tree_fingerprint(amplitude),
                "duration": array_tree_fingerprint(duration),
                "stimulus_source_id": source,
            }
        )
        self.unit_ids = units
        self.fiber_ids = fibers
        self.fiber_motor_unit_index = unit_index
        self.fiber_endplate_node = endplate
        self.node_count = nodes
        self.amplitude_uA_per_cm2 = amplitude
        self.duration_ms = duration
        self.stimulus_source_id = source
        self.evidence = MotorUnitTerritoryEvidence(
            counts,
            every_fiber,
            every_unit,
            endplates_valid,
            every_fiber & every_unit & endplates_valid,
            plan_id,
        )
        self.plan_id = plan_id

    def bind_events(
        self,
        event_times_ms: ArrayLike,
        event_mask: ArrayLike,
        /,
        *,
        event_source_id: str,
    ) -> MotorUnitEndplateStimulus:
        times = jnp.asarray(event_times_ms)
        mask = jnp.asarray(event_mask, dtype=bool)
        if times.ndim != 2 or times.shape[0] != len(self.unit_ids):
            raise ValueError("event_times_ms must have shape (motor_unit, event_slot).")
        if mask.shape != times.shape:
            raise ValueError("event_mask must match event_times_ms.")
        source = str(event_source_id).strip()
        if not source:
            raise ValueError("event_source_id must be nonempty.")
        active_times = np.asarray(times)[np.asarray(mask)]
        if active_times.size and not np.all(np.isfinite(active_times)):
            raise ValueError("Active motor-unit event times must be finite.")
        return MotorUnitEndplateStimulus(
            times,
            mask,
            self.fiber_motor_unit_index,
            self.fiber_endplate_node,
            self.amplitude_uA_per_cm2,
            self.duration_ms,
            self.node_count,
            source,
            canonical_fingerprint(
                {
                    "kind": "bound-motor-unit-endplate-stimulus",
                    "territory": self.plan_id,
                    "event_source_id": source,
                    "event_capacity": times.shape[1],
                }
            ),
        )


__all__ = [
    "MotorUnitEndplateStimulus",
    "MotorUnitTerritoryEvidence",
    "MotorUnitTerritoryPlan",
]
