#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical blood-compartment networks compiled to finite-state CTMCs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ....solver import finite_state_generator, FiniteStateGenerator
from ....stochastic import AbstractJumpProcess


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _positive_capacity(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class BloodCompartment:
    """One fixed-volume, well-mixed blood compartment in SI units."""

    compartment_id: str
    volume_m3: float
    absorbing: bool = False

    def __post_init__(self) -> None:
        identifier = _identifier(self.compartment_id, "compartment_id")
        volume = float(self.volume_m3)
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("volume_m3 must be finite and positive.")
        if not isinstance(self.absorbing, bool):
            raise TypeError("absorbing must be a boolean.")
        object.__setattr__(self, "compartment_id", identifier)
        object.__setattr__(self, "volume_m3", volume)


@dataclass(frozen=True, slots=True)
class BloodFlow:
    """Directed physical volume flow between two declared compartments."""

    source_compartment_id: str
    target_compartment_id: str
    volume_flow_m3_per_s: float

    def __post_init__(self) -> None:
        source = _identifier(self.source_compartment_id, "source_compartment_id")
        target = _identifier(self.target_compartment_id, "target_compartment_id")
        flow = float(self.volume_flow_m3_per_s)
        if source == target:
            raise ValueError("Blood flows must connect distinct compartments.")
        if not math.isfinite(flow) or flow <= 0.0:
            raise ValueError("volume_flow_m3_per_s must be finite and positive.")
        object.__setattr__(self, "source_compartment_id", source)
        object.__setattr__(self, "target_compartment_id", target)
        object.__setattr__(self, "volume_flow_m3_per_s", flow)


@dataclass(frozen=True, slots=True)
class CirculatingBloodModel:
    """A bounded fixed-volume circulation network.

    Each flow contributes the tracer transition intensity ``flow / source volume``.
    Compartments without an outgoing flow must be declared absorbing explicitly.
    The model does not infer missing return paths, source terms, or age structure.
    """

    compartments: tuple[BloodCompartment, ...]
    flows: tuple[BloodFlow, ...]
    model_id: str = field(init=False)

    def __post_init__(self) -> None:
        compartments = tuple(self.compartments)
        flows = tuple(self.flows)
        if not compartments or any(
            not isinstance(value, BloodCompartment) for value in compartments
        ):
            raise ValueError("compartments must contain BloodCompartment values.")
        if not flows or any(not isinstance(value, BloodFlow) for value in flows):
            raise ValueError("flows must contain BloodFlow values.")
        identifiers = tuple(value.compartment_id for value in compartments)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Compartment identifiers must be unique.")
        known = set(identifiers)
        if any(
            flow.source_compartment_id not in known
            or flow.target_compartment_id not in known
            for flow in flows
        ):
            raise ValueError("Every blood flow endpoint must identify a compartment.")
        edges = tuple(
            (flow.source_compartment_id, flow.target_compartment_id) for flow in flows
        )
        if len(set(edges)) != len(edges):
            raise ValueError("Directed blood-flow edges must be unique.")
        outgoing = {identifier: 0 for identifier in identifiers}
        for flow in flows:
            outgoing[flow.source_compartment_id] += 1
        for compartment in compartments:
            count = outgoing[compartment.compartment_id]
            if compartment.absorbing and count:
                raise ValueError("Absorbing compartments cannot have outgoing flows.")
            if not compartment.absorbing and count == 0:
                raise ValueError(
                    "A compartment without outgoing flow must be declared absorbing."
                )
        object.__setattr__(self, "compartments", compartments)
        object.__setattr__(self, "flows", flows)
        object.__setattr__(
            self,
            "model_id",
            canonical_fingerprint(
                {
                    "kind": "circulating-blood-model",
                    "compartments": [
                        {
                            "id": value.compartment_id,
                            "volume_m3": value.volume_m3,
                            "absorbing": value.absorbing,
                        }
                        for value in compartments
                    ],
                    "flows": [
                        {
                            "source": value.source_compartment_id,
                            "target": value.target_compartment_id,
                            "volume_flow_m3_per_s": value.volume_flow_m3_per_s,
                        }
                        for value in flows
                    ],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CirculationCapacityEvidence:
    compartment_count: int
    flow_count: int
    maximum_compartments: int
    maximum_flows: int

    @property
    def successful(self) -> bool:
        return (
            self.compartment_count <= self.maximum_compartments
            and self.flow_count <= self.maximum_flows
        )


class BloodTransitJumpProcess(AbstractJumpProcess):
    """Unmarked compartment-index jump process compiled from physical flows."""

    source_indices: Array
    target_indices: Array
    transition_rates_per_s: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    compartment_count: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        transition_rates_per_s: ArrayLike,
        /,
        *,
        compartment_count: int,
        process_id: str,
    ) -> None:
        source = jnp.asarray(source_indices, dtype=jnp.int32)
        target = jnp.asarray(target_indices, dtype=jnp.int32)
        rates = jnp.asarray(transition_rates_per_s, dtype=float)
        if (
            source.ndim != 1
            or source.shape != target.shape
            or source.shape != rates.shape
        ):
            raise ValueError(
                "Blood transition arrays must be matching non-empty vectors."
            )
        if source.shape[0] == 0:
            raise ValueError("A blood transit process requires at least one flow.")
        count = int(compartment_count)
        if count <= 0:
            raise ValueError("compartment_count must be positive.")
        if bool(jnp.any((source < 0) | (source >= count))):
            raise ValueError("Blood-flow source indices are outside compartment support.")
        if bool(jnp.any((target < 0) | (target >= count))):
            raise ValueError("Blood-flow target indices are outside compartment support.")
        if bool(jnp.any(~jnp.isfinite(rates) | (rates <= 0.0))):
            raise ValueError("Blood transition rates must be finite and positive.")
        self.source_indices = source
        self.target_indices = target
        self.transition_rates_per_s = rates
        self.state_shape = (1,)
        self.mark_shape = ()
        self.num_channels = int(source.shape[0])
        self.compartment_count = count
        self.process_id = _identifier(process_id, "process_id")

    def _state_index(self, state: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(state)[0]
        valid = (
            jnp.isfinite(value)
            & (value == jnp.floor(value))
            & (value >= 0)
            & (value < self.compartment_count)
        )
        index = jnp.clip(value, 0, self.compartment_count - 1).astype(jnp.int32)
        return index, valid

    def intensities(self, t: ArrayLike, state: ArrayLike, args=None, /) -> Array:
        del t, args
        index, valid = self._state_index(state)
        rates = jnp.where(self.source_indices == index, self.transition_rates_per_s, 0.0)
        return jnp.where(valid, rates, jnp.full_like(rates, jnp.nan))

    def jump(
        self, state: ArrayLike, channel: ArrayLike, mark: ArrayLike, args=None, /
    ) -> Array:
        del mark, args
        values = jnp.asarray(state)
        index, valid_state = self._state_index(values)
        raw_channel = jnp.asarray(channel)
        valid_channel = (
            (raw_channel >= 0)
            & (raw_channel < self.num_channels)
            & (raw_channel == jnp.floor(raw_channel))
        )
        selected = jnp.clip(raw_channel, 0, self.num_channels - 1).astype(jnp.int32)
        active = valid_state & valid_channel & (self.source_indices[selected] == index)
        destination = self.target_indices[selected].astype(values.dtype)
        return jnp.where(active, destination[None], values)

    def sample_mark(self, key, t, state, channel, args=None, /) -> Array:
        del key, t, channel, args
        return jnp.asarray(0, dtype=jnp.asarray(state).dtype)


@dataclass(frozen=True, slots=True)
class PreparedCirculatingBloodModel:
    model: CirculatingBloodModel
    process: BloodTransitJumpProcess
    generator: FiniteStateGenerator
    capacity: CirculationCapacityEvidence
    compartment_ids: tuple[str, ...]

    def encode(self, compartment_id: str, /) -> Array:
        identifier = _identifier(compartment_id, "compartment_id")
        if identifier not in self.compartment_ids:
            raise ValueError("Unknown circulating-blood compartment identifier.")
        return jnp.asarray([self.compartment_ids.index(identifier)], dtype=jnp.int32)

    def point_distribution(self, compartment_id: str, /) -> Array:
        index = int(self.encode(compartment_id)[0])
        return jnp.zeros((len(self.compartment_ids),), dtype=float).at[index].set(1.0)

    def stationary_distribution(self) -> Array:
        """Return a stationary law only for an irreducible circulation graph."""

        adjacency = np.asarray(jax.device_get(self.generator.matrix)) > 0.0
        reachable = adjacency | np.eye(adjacency.shape[0], dtype=bool)
        for intermediate in range(reachable.shape[0]):
            reachable |= (
                reachable[:, intermediate, None] & reachable[None, intermediate, :]
            )
        if not bool(np.all(reachable)):
            raise ValueError(
                "A stationary circulating-blood distribution requires an irreducible "
                "network; absorbing or disconnected models are refused."
            )
        return self.generator.stationary_distribution()


def prepare_circulating_blood_model(
    model: CirculatingBloodModel,
    /,
    *,
    maximum_compartments: int = 256,
    maximum_flows: int = 4096,
) -> PreparedCirculatingBloodModel:
    """Compile physical flow/volume ratios into a closed finite-state generator."""

    if not isinstance(model, CirculatingBloodModel):
        raise TypeError("model must be a CirculatingBloodModel.")
    state_capacity = _positive_capacity(maximum_compartments, "maximum_compartments")
    flow_capacity = _positive_capacity(maximum_flows, "maximum_flows")
    compartment_count = len(model.compartments)
    flow_count = len(model.flows)
    if compartment_count > state_capacity:
        raise ValueError(
            f"Circulating-blood compartment capacity exceeded: {compartment_count} > "
            f"{state_capacity}."
        )
    if flow_count > flow_capacity:
        raise ValueError(
            f"Circulating-blood flow capacity exceeded: {flow_count} > {flow_capacity}."
        )
    identifiers = tuple(value.compartment_id for value in model.compartments)
    indices = {identifier: index for index, identifier in enumerate(identifiers)}
    volumes = {value.compartment_id: value.volume_m3 for value in model.compartments}
    source = tuple(indices[value.source_compartment_id] for value in model.flows)
    target = tuple(indices[value.target_compartment_id] for value in model.flows)
    rates = tuple(
        value.volume_flow_m3_per_s / volumes[value.source_compartment_id]
        for value in model.flows
    )
    process = BloodTransitJumpProcess(
        source,
        target,
        rates,
        compartment_count=compartment_count,
        process_id=canonical_fingerprint(
            {"kind": "circulating-blood-jump-process", "model": model.model_id}
        ),
    )
    states = jnp.arange(compartment_count, dtype=jnp.int32)[:, None]
    generator = finite_state_generator(process, states, boundary_policy="error")
    evidence = CirculationCapacityEvidence(
        compartment_count,
        flow_count,
        state_capacity,
        flow_capacity,
    )
    return PreparedCirculatingBloodModel(
        model,
        process,
        generator,
        evidence,
        identifiers,
    )


__all__ = [
    "BloodCompartment",
    "BloodFlow",
    "BloodTransitJumpProcess",
    "CirculatingBloodModel",
    "CirculationCapacityEvidence",
    "PreparedCirculatingBloodModel",
    "prepare_circulating_blood_model",
]
