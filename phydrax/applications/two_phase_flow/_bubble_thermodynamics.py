#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermodynamic compartment registry of resolved bubbles.

Every non-atmosphere bubble identity owns one compartment slot. A slot stores
the gas amount, the internal energy and the law-specific internal state; these
are the extensive quantities of `phydrax.bubble_dynamics`. It also stores the
committed volume target, the compartment pressure and the centroid. The
compartment gas law (`AbstractBubbleCompartmentGasLaw`) is the only
thermodynamic authority.

Fixed topology (device, every accepted step):

- `BubbleCompartmentPlan.evaluate` returns the law pressure and the process
  compliance ``C = -dV/dp`` at the geometric volume. The compliance is a JVP
  of the law pressure along the law's own process direction
  ``(dV, dU) = (1, dU/dV)``.
- `BubbleCompartmentPlan.commit_projection` applies the projection's volume
  rate and new pressure. An adiabatic law takes the exact backward-Euler work
  ``dU = -p^{n+1} Q dt``. A heat-exchanging law keeps its own energy rate and
  books the heat ``dU + p^{n+1} Q dt``.

Topology events (host, epoch boundary): `BubbleCompartmentPlan.transact`
applies the journal records of `_bubble_components` through the law:

- a merge uses ``law.merge`` (conserves amount and energy, reports the mixing
  entropy);
- a split uses ``law.split`` (uniform-intensive policy, zero entropy
  production);
- a reconnect is a merge followed by a split;
- a created or entrained bubble is initialized at the declared local pressure
  and the ambient temperature;
- a vanishing or vented bubble is removed, and its amount and energy go to
  the ledger.

Tiny bubbles are never frozen: they keep evolving until the identity
labeling no longer finds them. Their removal is then a journaled vanish event.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import positive_integer
from ...bubble_dynamics import (
    AbstractBubbleCompartmentGasLaw,
    BubbleEnvironment,
    BubbleGasState,
)
from ._bubble_components import (
    ATMOSPHERE_ID,
    BubbleComponentState,
    BubbleTransitionRecord,
)


class BubbleCompartmentStatus(IntEnum):
    """Outcome of a compartment transaction or projection commit."""

    COMMITTED = 0
    CAPACITY_EXCEEDED = 1
    MISSING_COMPARTMENT = 2
    INADMISSIBLE_STATE = 3


class BubbleCompartmentState(StrictModule):
    """Fixed-capacity registry of gas compartments keyed by bubble identity.

    ``bubble_id`` is ``-1`` for empty slots. ``internal`` stacks the
    law-specific internal state (shape ``(capacity, S)``).
    """

    bubble_id: Array
    amount: Array
    internal_energy: Array
    internal: Array
    volume: Array
    pressure: Array
    centroid: Array
    epoch: Array

    @property
    def active(self) -> Array:
        return self.bubble_id > ATMOSPHERE_ID


class BubbleCompartmentEvaluation(StrictModule):
    """Law pressure, temperature, compliance and admissibility of every slot."""

    pressure: Array
    temperature: Array
    compliance: Array
    admissible: Array


class BubbleCompartmentWork(StrictModule):
    """Work, heat and equation-of-state evidence of one projection commit.

    ``work`` is the pressure work done by each gas compartment on the liquid
    (``p^{n+1} Q dt``, positive on expansion); ``energy_change`` and ``heat``
    close the first law ``dU = heat - work`` exactly. ``eos_residual`` is the
    relative mismatch between the law pressure at the committed volume target
    and the projection pressure, i.e. the linearization error of the implicit
    thermodynamic closure.
    """

    work: Array
    heat: Array
    energy_change: Array
    eos_residual: Array
    first_law_residual: Array
    admissible: Array


class BubbleCompartmentLedger(StrictModule):
    """Cumulative gas amount and energy accounting across topology events."""

    created_amount: Array
    created_energy: Array
    vanished_amount: Array
    vanished_energy: Array
    vented_amount: Array
    vented_energy: Array
    entrained_amount: Array
    entrained_energy: Array
    merge_entropy_production: Array
    split_entropy_production: Array
    amount_residual: Array
    energy_residual: Array

    @classmethod
    def zeros(cls) -> BubbleCompartmentLedger:
        zero = jnp.zeros((), dtype=jnp.float64)
        return cls(*((zero,) * 12))


class BubbleCompartmentTransaction(StrictModule):
    """Result of one host topology transaction on the registry.

    On refusal (``status`` not ``COMMITTED``) ``state`` is the unchanged
    source state and the ledger increment is zero.
    """

    state: BubbleCompartmentState
    ledger: BubbleCompartmentLedger
    status: BubbleCompartmentStatus = eqx.field(static=True)
    records: tuple[BubbleTransitionRecord, ...] = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)

    @property
    def committed(self) -> bool:
        return self.status is BubbleCompartmentStatus.COMMITTED


def _slot_state(state: BubbleCompartmentState, slot: int, /) -> BubbleGasState:
    return BubbleGasState(
        state.amount[slot], state.internal_energy[slot], state.internal[slot]
    )


def _stack_states(states: list[BubbleGasState], /) -> BubbleGasState:
    energies = [state.internal_energy for state in states]
    if any(energy is None for energy in energies):
        raise ValueError("Compartment gas states require internal energy.")
    return BubbleGasState(
        jnp.stack([state.amount for state in states]),
        jnp.stack([jnp.asarray(energy) for energy in energies]),
        jnp.stack([state.internal for state in states]),
    )


def _split_state(states: BubbleGasState, index: int, /) -> BubbleGasState:
    energy = states.internal_energy
    if energy is None:
        raise ValueError("Compartment gas states require internal energy.")
    return BubbleGasState(states.amount[index], energy[index], states.internal[index])


def _energy(state: BubbleGasState, /) -> Array:
    if state.internal_energy is None:
        raise ValueError("Compartment gas states require internal energy.")
    return state.internal_energy


class BubbleCompartmentPlan(StrictModule):
    """Registry of bubble gas compartments driven by one compartment gas law."""

    law: AbstractBubbleCompartmentGasLaw
    environment: BubbleEnvironment
    capacity: int = eqx.field(static=True)
    internal_size: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        law: AbstractBubbleCompartmentGasLaw,
        environment: BubbleEnvironment,
        /,
        *,
        capacity: int,
        dimension: int,
    ) -> None:
        if not isinstance(law, AbstractBubbleCompartmentGasLaw):
            raise TypeError("law must be an AbstractBubbleCompartmentGasLaw.")
        if not isinstance(environment, BubbleEnvironment):
            raise TypeError("environment must be a BubbleEnvironment.")
        slots = positive_integer(capacity, "capacity")
        space = positive_integer(dimension, "dimension")
        if space not in (2, 3):
            raise ValueError("dimension must be 2 or 3.")
        probe = law.initialize(
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            environment,
        )
        if probe.internal_energy is None:
            raise ValueError("Compartment laws must carry internal energy.")
        self.law = law
        self.environment = environment
        self.capacity = slots
        self.internal_size = probe.internal.shape[0]
        self.dimension = space
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubble-compartment-plan",
                "law": law.law_id,
                "capacity": slots,
                "dimension": space,
            }
        )

    def empty_state(self) -> BubbleCompartmentState:
        capacity = self.capacity
        zero = jnp.zeros((capacity,), dtype=jnp.float64)
        return BubbleCompartmentState(
            bubble_id=jnp.full((capacity,), -1, dtype=jnp.int32),
            amount=zero,
            internal_energy=zero,
            internal=jnp.zeros((capacity, self.internal_size), dtype=jnp.float64),
            volume=zero,
            pressure=zero,
            centroid=jnp.zeros((capacity, self.dimension), dtype=jnp.float64),
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def initial_state(
        self, identity: BubbleComponentState, pressure: ArrayLike, /
    ) -> BubbleCompartmentState:
        """Compartments of every initial bubble at ``pressure`` and ambient T.

        ``pressure`` is one absolute pressure per identity component slot (or a
        scalar). This is a host preparation step.
        """

        ids = np.asarray(identity.slot_ids)
        volumes = np.asarray(identity.labels.volume)
        centroids = np.asarray(identity.labels.centroid)
        slot_pressure = np.broadcast_to(np.asarray(pressure, dtype=np.float64), ids.shape)
        bubbles = np.flatnonzero(ids > ATMOSPHERE_ID)
        if bubbles.size > self.capacity:
            raise ValueError("The initial bubble count exceeds the registry capacity.")
        state = self.empty_state()
        for slot, component in enumerate(bubbles):
            gas = self.law.initialize(
                jnp.asarray(volumes[component], dtype=jnp.float64),
                jnp.asarray(slot_pressure[component], dtype=jnp.float64),
                self.environment,
            )
            state = self._store(
                state,
                slot,
                int(ids[component]),
                gas,
                float(volumes[component]),
                float(slot_pressure[component]),
                centroids[component],
            )
        return state

    def _store(
        self,
        state: BubbleCompartmentState,
        slot: int,
        bubble_id: int,
        gas: BubbleGasState,
        volume: float,
        pressure: float,
        centroid: np.ndarray,
        /,
    ) -> BubbleCompartmentState:
        return BubbleCompartmentState(
            bubble_id=state.bubble_id.at[slot].set(bubble_id),
            amount=state.amount.at[slot].set(gas.amount),
            internal_energy=state.internal_energy.at[slot].set(_energy(gas)),
            internal=state.internal.at[slot].set(gas.internal),
            volume=state.volume.at[slot].set(volume),
            pressure=state.pressure.at[slot].set(pressure),
            centroid=state.centroid.at[slot].set(jnp.asarray(centroid)),
            epoch=state.epoch,
        )

    def _gas_states(self, state: BubbleCompartmentState, /) -> BubbleGasState:
        active = state.active
        return BubbleGasState(
            jnp.where(active, state.amount, 1.0),
            jnp.where(active, state.internal_energy, 1.0),
            state.internal,
        )

    def evaluate(
        self, state: BubbleCompartmentState, volume: ArrayLike, /
    ) -> BubbleCompartmentEvaluation:
        """Law pressure, temperature and process compliance at ``volume``.

        Inactive slots report zero pressure and compliance and are admissible.
        """

        active = state.active
        volume_ = jnp.asarray(volume, dtype=jnp.float64)
        if volume_.shape != (self.capacity,):
            raise ValueError("volume must list one value per compartment slot.")
        safe_volume = jnp.where(active & (volume_ > 0.0), volume_, 1.0)
        gases = self._gas_states(state)
        law = self.law
        environment = self.environment

        def one(volume_value: Array, gas: BubbleGasState) -> tuple[Array, ...]:
            energy = _energy(gas)

            def pressure_of(value: Array, internal_energy: Array) -> Array:
                return law.evaluate(
                    value,
                    jnp.zeros_like(value),
                    BubbleGasState(gas.amount, internal_energy, gas.internal),
                    environment,
                ).pressure

            unit_rate = law.evaluate(
                volume_value, jnp.ones_like(volume_value), gas, environment
            )
            direction = (
                jnp.zeros_like(volume_value)
                if unit_rate.energy_rate is None
                else unit_rate.energy_rate
            )
            pressure, slope = jax.jvp(
                pressure_of,
                (volume_value, energy),
                (jnp.ones_like(volume_value), direction),
            )
            return pressure, unit_rate.temperature, -1.0 / slope, unit_rate.admissible

        pressure, temperature, compliance, admissible = jax.vmap(one)(safe_volume, gases)
        usable = active & (volume_ > 0.0)
        return BubbleCompartmentEvaluation(
            pressure=jnp.where(usable, pressure, 0.0),
            temperature=jnp.where(usable, temperature, 0.0),
            compliance=jnp.where(usable, compliance, 0.0),
            admissible=jnp.where(
                active,
                usable
                & admissible
                & jnp.isfinite(compliance)
                & (compliance > 0.0)
                & jnp.isfinite(pressure),
                True,
            ),
        )

    def slot_map(
        self, state: BubbleCompartmentState, component_ids: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Registry slot of every identity component slot, and missing bubbles.

        The tiny ``components x capacity`` identity comparison is the bounded
        lookup of this registry; ``-1`` marks atmosphere and empty components.
        """

        ids = jnp.asarray(component_ids, dtype=jnp.int32)
        matches = (ids[:, None] == state.bubble_id[None, :]) & (
            ids[:, None] > ATMOSPHERE_ID
        )
        found = jnp.any(matches, axis=1)
        slot = jnp.where(found, jnp.argmax(matches, axis=1), -1).astype(jnp.int32)
        return slot, (ids > ATMOSPHERE_ID) & ~found

    def commit_projection(
        self,
        state: BubbleCompartmentState,
        volume: ArrayLike,
        centroid: ArrayLike,
        pressure: ArrayLike,
        volume_rate: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> tuple[BubbleCompartmentState, BubbleCompartmentWork]:
        """Apply the projection's ``p^{n+1}`` and ``Q`` to every compartment.

        ``volume`` is the geometric volume at the projection. The committed
        slot volume is the target ``V + Q dt`` that the next transport
        realizes, so the stored amount, energy, volume and pressure are
        consistent with the law to the linearization error ``eos_residual``.
        """

        active = state.active
        volume_ = jnp.asarray(volume, dtype=jnp.float64)
        pressure_ = jnp.asarray(pressure, dtype=jnp.float64)
        rate = jnp.asarray(volume_rate, dtype=jnp.float64)
        dt = jnp.asarray(step_size, dtype=jnp.float64)
        work = jnp.where(active, pressure_ * rate * dt, 0.0)
        gases = self._gas_states(state)
        safe_volume = jnp.where(active & (volume_ > 0.0), volume_, 1.0)
        if self.law.capabilities.heat_transfer:
            law = self.law
            environment = self.environment

            def law_energy_rate(
                volume_value: Array, rate_value: Array, gas: BubbleGasState
            ) -> Array:
                evaluation = law.evaluate(volume_value, rate_value, gas, environment)
                if evaluation.energy_rate is None:
                    return jnp.zeros_like(volume_value)
                return evaluation.energy_rate

            energy_change = jnp.where(
                active,
                jax.vmap(law_energy_rate)(safe_volume, rate, gases) * dt,
                0.0,
            )
        else:
            energy_change = -work
        heat = energy_change + work
        target = jnp.where(active, volume_ + rate * dt, 0.0)
        updated = BubbleCompartmentState(
            bubble_id=state.bubble_id,
            amount=state.amount,
            internal_energy=state.internal_energy + energy_change,
            internal=state.internal,
            volume=target,
            pressure=jnp.where(active, pressure_, 0.0),
            centroid=jnp.where(
                active[:, None], jnp.asarray(centroid, dtype=jnp.float64), 0.0
            ),
            epoch=state.epoch,
        )
        law_pressure = self.evaluate(updated, target).pressure
        scale = jnp.where(active & (pressure_ > 0.0), pressure_, 1.0)
        eos = jnp.where(active, jnp.abs(law_pressure - pressure_) / scale, 0.0)
        admissible = jnp.all(
            jnp.where(
                active, jnp.isfinite(updated.internal_energy) & (target > 0.0), True
            )
        )
        return updated, BubbleCompartmentWork(
            work=work,
            heat=heat,
            energy_change=energy_change,
            eos_residual=eos,
            first_law_residual=jnp.max(jnp.abs(energy_change - heat + work)),
            admissible=admissible,
        )

    def transact(
        self,
        state: BubbleCompartmentState,
        records: tuple[BubbleTransitionRecord, ...],
        creation_pressure: ArrayLike,
        atmosphere_pressure: ArrayLike,
        centroid: ArrayLike,
        /,
    ) -> BubbleCompartmentTransaction:
        """Apply journaled topology events to the registry through the law.

        ``creation_pressure`` and ``centroid`` are indexed by the proposal's
        identity component slot (``record.child_slots``). This is a host
        epoch transition; refusal returns the unchanged ``state``.
        """

        pressures = np.asarray(creation_pressure, dtype=np.float64)
        centroids = np.asarray(centroid, dtype=np.float64)
        atmosphere = jnp.asarray(atmosphere_pressure, dtype=jnp.float64)
        ids = [int(value) for value in np.asarray(state.bubble_id)]
        slot_of = {bubble: slot for slot, bubble in enumerate(ids) if bubble > 0}
        totals = {
            name: jnp.zeros((), dtype=jnp.float64)
            for name in (
                "created_amount",
                "created_energy",
                "vanished_amount",
                "vanished_energy",
                "vented_amount",
                "vented_energy",
                "entrained_amount",
                "entrained_energy",
                "merge_entropy_production",
                "split_entropy_production",
                "amount_residual",
                "energy_residual",
            )
        }
        working = state
        for record in records:
            parents = [bubble for bubble in record.parent_ids if bubble > ATMOSPHERE_ID]
            if any(bubble not in slot_of for bubble in parents):
                return self._refuse(
                    state, records, BubbleCompartmentStatus.MISSING_COMPARTMENT
                )
            parent_states = [_slot_state(working, slot_of[bubble]) for bubble in parents]
            parent_volumes = [
                float(working.volume[slot_of[bubble]]) for bubble in parents
            ]
            children = [
                (bubble, volume, slot)
                for bubble, volume, slot in zip(
                    record.child_ids,
                    record.child_volumes,
                    record.child_slots,
                    strict=True,
                )
                if bubble > ATMOSPHERE_ID
            ]
            child_states, admissible = self._event_children(
                record,
                parent_states,
                parent_volumes,
                children,
                pressures,
                atmosphere,
                totals,
            )
            if not admissible:
                return self._refuse(
                    state, records, BubbleCompartmentStatus.INADMISSIBLE_STATE
                )
            for bubble in parents:
                slot = slot_of.pop(bubble)
                ids[slot] = -1
                working = self._clear(working, slot)
            for (bubble, volume, component), gas in zip(
                children, child_states, strict=True
            ):
                free = [slot for slot, value in enumerate(ids) if value < 0]
                if not free:
                    return self._refuse(
                        state, records, BubbleCompartmentStatus.CAPACITY_EXCEEDED
                    )
                slot = free[0]
                ids[slot] = bubble
                slot_of[bubble] = slot
                evaluation = self.law.evaluate(
                    jnp.asarray(volume, dtype=jnp.float64),
                    jnp.zeros((), dtype=jnp.float64),
                    gas,
                    self.environment,
                )
                working = self._store(
                    working,
                    slot,
                    bubble,
                    gas,
                    volume,
                    float(evaluation.pressure),
                    centroids[component],
                )
        committed = BubbleCompartmentState(
            bubble_id=working.bubble_id,
            amount=working.amount,
            internal_energy=working.internal_energy,
            internal=working.internal,
            volume=working.volume,
            pressure=working.pressure,
            centroid=working.centroid,
            epoch=state.epoch + (1 if records else 0),
        )
        return BubbleCompartmentTransaction(
            state=committed,
            ledger=BubbleCompartmentLedger(**totals),
            status=BubbleCompartmentStatus.COMMITTED,
            records=records,
            transaction_id=self._transaction_id(state, records, "committed"),
        )

    def _event_children(
        self,
        record: BubbleTransitionRecord,
        parents: list[BubbleGasState],
        parent_volumes: list[float],
        children: list[tuple[int, float, int]],
        pressures: np.ndarray,
        atmosphere: Array,
        totals: dict[str, Array],
        /,
    ) -> tuple[list[BubbleGasState], bool]:
        """Child gas states of one record and its ledger contributions."""

        law = self.law
        environment = self.environment
        parent_amount = sum((state.amount for state in parents), start=jnp.zeros(()))
        parent_energy = sum((_energy(state) for state in parents), start=jnp.zeros(()))
        vented = ATMOSPHERE_ID in record.child_ids or ATMOSPHERE_ID in record.parent_ids
        if vented or not parents:
            key = "vented" if parents and vented else "vanished"
            totals[f"{key}_amount"] = totals[f"{key}_amount"] + parent_amount
            totals[f"{key}_energy"] = totals[f"{key}_energy"] + parent_energy
            source = atmosphere if vented else None
            created = []
            for _, volume, component in children:
                reference = (
                    source
                    if source is not None
                    else jnp.asarray(pressures[component], dtype=jnp.float64)
                )
                gas = law.initialize(
                    jnp.asarray(volume, dtype=jnp.float64), reference, environment
                )
                name = "entrained" if vented else "created"
                totals[f"{name}_amount"] = totals[f"{name}_amount"] + gas.amount
                totals[f"{name}_energy"] = totals[f"{name}_energy"] + _energy(gas)
                created.append(gas)
            return created, True
        if not children:
            totals["vanished_amount"] = totals["vanished_amount"] + parent_amount
            totals["vanished_energy"] = totals["vanished_energy"] + parent_energy
            return [], True
        if len(parents) == 1:
            merged = parents[0]
            merged_volume = jnp.asarray(parent_volumes[0], dtype=jnp.float64)
            merge_ok = jnp.asarray(True)
        else:
            merged_volume = jnp.asarray(
                sum(volume for _, volume, _ in children)
                if len(children) == 1
                else sum(parent_volumes),
                dtype=jnp.float64,
            )
            merge = law.merge(
                _stack_states(parents),
                jnp.asarray(parent_volumes, dtype=jnp.float64),
                merged_volume,
                environment,
            )
            merged = merge.state
            merge_ok = merge.admissible
            totals["merge_entropy_production"] = (
                totals["merge_entropy_production"] + merge.entropy_production
            )
        if len(children) == 1:
            outcome = [merged]
            split_ok = jnp.asarray(True)
        else:
            split = law.split(
                merged,
                merged_volume,
                jnp.asarray([volume for _, volume, _ in children], dtype=jnp.float64),
                environment,
            )
            outcome = [
                _split_state(split.states, index) for index in range(len(children))
            ]
            split_ok = split.admissible
            totals["split_entropy_production"] = (
                totals["split_entropy_production"] + split.entropy_production
            )
        child_amount = sum((state.amount for state in outcome), start=jnp.zeros(()))
        child_energy = sum((_energy(state) for state in outcome), start=jnp.zeros(()))
        totals["amount_residual"] = totals["amount_residual"] + jnp.abs(
            child_amount - parent_amount
        )
        totals["energy_residual"] = totals["energy_residual"] + jnp.abs(
            child_energy - parent_energy
        )
        return outcome, bool(merge_ok & split_ok)

    def _clear(
        self, state: BubbleCompartmentState, slot: int, /
    ) -> BubbleCompartmentState:
        return BubbleCompartmentState(
            bubble_id=state.bubble_id.at[slot].set(-1),
            amount=state.amount.at[slot].set(0.0),
            internal_energy=state.internal_energy.at[slot].set(0.0),
            internal=state.internal.at[slot].set(0.0),
            volume=state.volume.at[slot].set(0.0),
            pressure=state.pressure.at[slot].set(0.0),
            centroid=state.centroid.at[slot].set(0.0),
            epoch=state.epoch,
        )

    def _refuse(
        self,
        state: BubbleCompartmentState,
        records: tuple[BubbleTransitionRecord, ...],
        status: BubbleCompartmentStatus,
        /,
    ) -> BubbleCompartmentTransaction:
        return BubbleCompartmentTransaction(
            state=state,
            ledger=BubbleCompartmentLedger.zeros(),
            status=status,
            records=records,
            transaction_id=self._transaction_id(state, records, status.name.lower()),
        )

    def _transaction_id(
        self,
        state: BubbleCompartmentState,
        records: tuple[BubbleTransitionRecord, ...],
        outcome: str,
        /,
    ) -> str:
        return canonical_fingerprint(
            {
                "kind": "bubble-compartment-transaction",
                "plan": self.plan_id,
                "epoch": int(state.epoch),
                "records": [record.record_id for record in records],
                "outcome": outcome,
            }
        )

    def totals(self, state: BubbleCompartmentState, /) -> tuple[Array, Array]:
        """Registry gas amount and internal energy over active compartments."""

        active = state.active
        return (
            jnp.sum(jnp.where(active, state.amount, 0.0)),
            jnp.sum(jnp.where(active, state.internal_energy, 0.0)),
        )


__all__ = [
    "BubbleCompartmentEvaluation",
    "BubbleCompartmentLedger",
    "BubbleCompartmentPlan",
    "BubbleCompartmentState",
    "BubbleCompartmentStatus",
    "BubbleCompartmentTransaction",
    "BubbleCompartmentWork",
]
