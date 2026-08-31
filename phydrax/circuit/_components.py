#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._mna import AbstractMNAComponent, MNAStamp


def _batched_parameter(value: Array, omega: Array, name: str, /) -> Array:
    if value.ndim == 0:
        return jnp.broadcast_to(value, omega.shape)
    if value.shape != omega.shape:
        raise ValueError(f"{name} must be scalar or match angular_frequency shape.")
    return value


def _two_terminal_admittance(admittance: Array) -> MNAStamp:
    pattern = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=admittance.dtype)
    y = admittance[..., None, None] * pattern
    batch = admittance.shape
    empty_b = jnp.zeros(batch + (2, 0), dtype=y.dtype)
    empty_c = jnp.zeros(batch + (0, 2), dtype=y.dtype)
    empty_d = jnp.zeros(batch + (0, 0), dtype=y.dtype)
    return MNAStamp(y, empty_b, empty_c, empty_d)


class Resistor(AbstractMNAComponent):
    """Ideal two-terminal resistor."""

    resistance: Array
    component_id: str = eqx.field(static=True)

    def __init__(self, resistance: ArrayLike, /, *, component_id: str = "resistor"):
        value = jnp.asarray(resistance)
        value = eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value))
            | jnp.any(jnp.real(value) <= 0.0)
            | jnp.any(jnp.imag(value) != 0.0),
            "Resistance must be finite, real, and positive.",
        )
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.resistance = value
        self.component_id = identifier

    @property
    def terminal_count(self) -> int:
        return 2

    @property
    def auxiliary_count(self) -> int:
        return 0

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        resistance = _batched_parameter(self.resistance, omega, "resistance")
        return _two_terminal_admittance(1.0 / resistance)


class Capacitor(AbstractMNAComponent):
    """Ideal capacitor under the exp(-i omega t) convention."""

    capacitance: Array
    component_id: str = eqx.field(static=True)

    def __init__(self, capacitance: ArrayLike, /, *, component_id: str = "capacitor"):
        value = jnp.asarray(capacitance)
        value = eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value))
            | jnp.any(jnp.real(value) <= 0.0)
            | jnp.any(jnp.imag(value) != 0.0),
            "Capacitance must be finite, real, and positive.",
        )
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.capacitance = value
        self.component_id = identifier

    @property
    def terminal_count(self) -> int:
        return 2

    @property
    def auxiliary_count(self) -> int:
        return 0

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        capacitance = _batched_parameter(self.capacitance, omega, "capacitance")
        return _two_terminal_admittance(-1j * omega * capacitance)


class Inductor(AbstractMNAComponent):
    """Ideal inductor stamped by current auxiliary, including the exact DC limit."""

    inductance: Array
    component_id: str = eqx.field(static=True)

    def __init__(self, inductance: ArrayLike, /, *, component_id: str = "inductor"):
        value = jnp.asarray(inductance)
        value = eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value))
            | jnp.any(jnp.real(value) <= 0.0)
            | jnp.any(jnp.imag(value) != 0.0),
            "Inductance must be finite, real, and positive.",
        )
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.inductance = value
        self.component_id = identifier

    @property
    def terminal_count(self) -> int:
        return 2

    @property
    def auxiliary_count(self) -> int:
        return 1

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        inductance = _batched_parameter(self.inductance, omega, "inductance")
        impedance = -1j * omega * inductance
        batch = omega.shape
        y = jnp.zeros(batch + (2, 2), dtype=jnp.result_type(impedance, jnp.complex128))
        b = jnp.broadcast_to(jnp.asarray([[1.0], [-1.0]], dtype=y.dtype), batch + (2, 1))
        c = jnp.broadcast_to(jnp.asarray([[1.0, -1.0]], dtype=y.dtype), batch + (1, 2))
        d = -impedance[..., None, None]
        return MNAStamp(y, b, c, d)


class AdmittanceComponent(AbstractMNAComponent):
    """Linear N-terminal component storing one native admittance representation."""

    admittance: Array
    component_id: str = eqx.field(static=True)
    _terminal_count: int = eqx.field(static=True)

    def __init__(
        self, admittance: ArrayLike, /, *, component_id: str = "admittance-component"
    ):
        value = jnp.asarray(admittance)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError("admittance must end in one square terminal matrix.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.admittance = value.astype(jnp.result_type(value, jnp.complex128))
        self.component_id = identifier
        self._terminal_count = int(value.shape[-1])

    @property
    def terminal_count(self) -> int:
        return self._terminal_count

    @property
    def auxiliary_count(self) -> int:
        return 0

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        if self.admittance.ndim == 2:
            y = jnp.broadcast_to(self.admittance, omega.shape + self.admittance.shape)
        elif self.admittance.shape[:-2] == omega.shape:
            y = self.admittance
        else:
            raise ValueError("Batched admittance must match angular_frequency shape.")
        batch = omega.shape
        count = self.terminal_count
        return MNAStamp(
            y,
            jnp.zeros(batch + (count, 0), dtype=y.dtype),
            jnp.zeros(batch + (0, count), dtype=y.dtype),
            jnp.zeros(batch + (0, 0), dtype=y.dtype),
        )


class ImpedanceComponent(AbstractMNAComponent):
    """N-port impedance relation lowered without inverting the impedance matrix."""

    impedance: Array
    component_id: str = eqx.field(static=True)
    _terminal_count: int = eqx.field(static=True)

    def __init__(
        self, impedance: ArrayLike, /, *, component_id: str = "impedance-component"
    ):
        value = jnp.asarray(impedance)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError("impedance must end in one square port matrix.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.impedance = value.astype(jnp.result_type(value, jnp.complex128))
        self.component_id = identifier
        self._terminal_count = int(value.shape[-1])

    @property
    def terminal_count(self) -> int:
        return self._terminal_count

    @property
    def auxiliary_count(self) -> int:
        return self._terminal_count

    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        omega = jnp.asarray(angular_frequency)
        if self.impedance.ndim == 2:
            impedance = jnp.broadcast_to(
                self.impedance, omega.shape + self.impedance.shape
            )
        elif self.impedance.shape[:-2] == omega.shape:
            impedance = self.impedance
        else:
            raise ValueError("Batched impedance must match angular_frequency shape.")
        count = self.terminal_count
        identity = jnp.broadcast_to(
            jnp.eye(count, dtype=impedance.dtype), omega.shape + (count, count)
        )
        zeros = jnp.zeros_like(identity)
        return MNAStamp(zeros, identity, identity, -impedance)


__all__ = [
    "AdmittanceComponent",
    "Capacitor",
    "ImpedanceComponent",
    "Inductor",
    "Resistor",
]
