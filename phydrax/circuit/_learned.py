#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._elements import (
    AbstractCircuitEnergyLaw,
    AbstractImplicitCircuitLaw,
    CircuitElement,
    CircuitElementEvaluation,
    CircuitElementStateLayout,
)


class LearnedCircuitLawEvidence(StrictModule):
    passive_by_construction: bool = eqx.field(static=True)
    causal_by_construction: bool = eqx.field(static=True)
    finite_probe: Array
    minimum_conductance: Array


class MonotoneLearnedConductanceLaw(AbstractImplicitCircuitLaw):
    """Learned nonnegative conductance; power g(v, u) v² is nonnegative."""

    model: Any
    minimum_conductance: Array
    evidence: LearnedCircuitLawEvidence

    def __init__(
        self,
        model: Callable[[Array], ArrayLike],
        /,
        *,
        minimum_conductance: ArrayLike = 0.0,
        probe_voltages: ArrayLike = (-1.0, 0.0, 1.0),
        law_id: str | None = None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        minimum = jnp.asarray(minimum_conductance, dtype=float)
        probes = jnp.asarray(probe_voltages, dtype=float)
        if (
            minimum.shape != ()
            or bool(~jnp.isfinite(minimum))
            or bool(minimum < 0.0)
            or probes.ndim != 1
            or probes.size == 0
            or bool(jnp.any(~jnp.isfinite(probes)))
        ):
            raise ValueError("Learned conductance policy values are invalid.")
        raw = jnp.stack(tuple(jnp.asarray(model(value)) for value in probes))
        if raw.shape != probes.shape:
            raise ValueError("Learned conductance model must map scalars to scalars.")
        conductance = minimum + jax.nn.softplus(raw)
        evidence = LearnedCircuitLawEvidence(
            True,
            True,
            jnp.all(jnp.isfinite(conductance)),
            jnp.min(conductance),
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "monotone-learned-conductance",
                    "minimum": float(minimum),
                    "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
                }
            )
            if law_id is None
            else str(law_id)
        )
        if not identifier:
            raise ValueError("law_id must be non-empty.")
        self.model = model
        self.minimum_conductance = minimum
        self.evidence = evidence
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = identifier

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, state, state_rate, inputs, args
        voltage = terminal_voltages[0] - terminal_voltages[1]
        raw = jnp.asarray(self.model(voltage))
        if raw.shape != ():
            raise ValueError("Learned conductance model must return one scalar.")
        conductance = self.minimum_conductance + jax.nn.softplus(raw)
        current = conductance * voltage
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


class LearnedConductanceEnergyLaw(AbstractCircuitEnergyLaw):
    def stored_energy(
        self, terminal_voltages: Array, state: Array, /, *, args: Any = None
    ) -> Array:
        del terminal_voltages, state, args
        return jnp.asarray(0.0)

    def dissipated_power(
        self,
        terminal_voltages: Array,
        terminal_currents: Array,
        state: Array,
        /,
        *,
        args: Any = None,
    ) -> Array:
        del state, args
        return jnp.real(jnp.vdot(terminal_voltages, terminal_currents))


def learned_conductance_element(
    model: Callable[[Array], ArrayLike],
    /,
    *,
    minimum_conductance: ArrayLike = 0.0,
    probe_voltages: ArrayLike = (-1.0, 0.0, 1.0),
    element_id: str = "learned-conductance",
) -> CircuitElement:
    law = MonotoneLearnedConductanceLaw(
        model,
        minimum_conductance=minimum_conductance,
        probe_voltages=probe_voltages,
        law_id=f"{element_id}/law",
    )
    return CircuitElement(
        law,
        energy_law=LearnedConductanceEnergyLaw(),
        element_id=element_id,
    )


__all__ = [
    "LearnedCircuitLawEvidence",
    "LearnedConductanceEnergyLaw",
    "MonotoneLearnedConductanceLaw",
    "learned_conductance_element",
]
