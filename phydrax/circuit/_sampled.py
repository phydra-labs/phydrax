#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._models import AbstractScatteringComponent, ScatteringResponse
from ._ports import ElectricalWaveReference, WavePort
from .io import TouchstoneData


class ScatteringInterpolationPolicy(StrictModule):
    """Cartesian linear interpolation with an explicit out-of-band policy."""

    method: Literal["cartesian-linear"] = eqx.field(static=True)
    out_of_band: Literal["error", "constant"] = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        method: Literal["cartesian-linear"] = "cartesian-linear",
        out_of_band: Literal["error", "constant"] = "error",
    ):
        if method != "cartesian-linear":
            raise ValueError("Only Cartesian linear interpolation is supported.")
        if out_of_band not in ("error", "constant"):
            raise ValueError("out_of_band must be 'error' or explicit 'constant'.")
        self.method = method
        self.out_of_band = out_of_band


class SampledScatteringModel(AbstractScatteringComponent):
    """In-band Cartesian interpolation of imported scattering samples."""

    frequencies_hz: Array
    scattering: Array
    _ports: tuple[WavePort, ...]
    policy: ScatteringInterpolationPolicy
    numeric_version: Array
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies_hz: ArrayLike,
        scattering: ArrayLike,
        ports: Sequence[WavePort],
        /,
        *,
        policy: ScatteringInterpolationPolicy | None = None,
        numeric_version: ArrayLike = 0,
        component_id: str = "sampled-scattering",
    ):
        frequencies = jnp.asarray(frequencies_hz, dtype=jnp.float64)
        matrix = jnp.asarray(scattering, dtype=jnp.complex128)
        port_tuple = tuple(ports)
        if frequencies.ndim != 1 or frequencies.size < 2:
            raise ValueError("Sampled scattering requires at least two 1-D frequencies.")
        if matrix.shape != (frequencies.size, len(port_tuple), len(port_tuple)):
            raise ValueError(
                "Sampled scattering must have shape (frequency, port, port)."
            )
        if not bool(jnp.all(jnp.diff(frequencies) > 0.0)):
            raise ValueError("Sample frequencies must be strictly increasing and unique.")
        if len({port.port_id for port in port_tuple}) != len(port_tuple):
            raise ValueError("Sampled scattering port IDs must be unique.")
        selected = ScatteringInterpolationPolicy() if policy is None else policy
        if not isinstance(selected, ScatteringInterpolationPolicy):
            raise TypeError("policy must be ScatteringInterpolationPolicy or None.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        self.frequencies_hz = frequencies
        self.scattering = matrix
        self._ports = port_tuple
        self.policy = selected
        self.numeric_version = version
        self.component_id = identifier

    @classmethod
    def from_touchstone(
        cls,
        data: TouchstoneData,
        /,
        *,
        policy: ScatteringInterpolationPolicy | None = None,
        component_id: str = "sampled-touchstone",
    ) -> "SampledScatteringModel":
        if not isinstance(data, TouchstoneData):
            raise TypeError("data must be TouchstoneData.")
        ports = tuple(
            WavePort(name, ElectricalWaveReference(reference))
            for name, reference in zip(
                data.port_names, data.reference_impedance, strict=True
            )
        )
        return cls(
            data.frequencies_hz,
            data.scattering,
            ports,
            policy=policy,
            component_id=component_id,
        )

    @property
    def ports(self) -> tuple[WavePort, ...]:
        return self._ports

    def evaluate(self, angular_frequency: ArrayLike, /) -> ScatteringResponse:
        omega = jnp.asarray(angular_frequency)
        frequencies = omega / (2.0 * jnp.pi)
        if self.policy.out_of_band == "error":
            frequencies = eqx.error_if(
                frequencies,
                jnp.any(frequencies < self.frequencies_hz[0])
                | jnp.any(frequencies > self.frequencies_hz[-1]),
                "Sampled scattering evaluation lies outside the imported frequency band.",
            )
        count = len(self._ports)
        entries = []
        for output in range(count):
            for input_ in range(count):
                values = self.scattering[:, output, input_]
                real = jnp.interp(frequencies, self.frequencies_hz, jnp.real(values))
                imaginary = jnp.interp(frequencies, self.frequencies_hz, jnp.imag(values))
                entries.append(real + 1j * imaginary)
        matrix = jnp.stack(entries, axis=-1).reshape(frequencies.shape + (count, count))
        return ScatteringResponse(
            matrix,
            tuple(reference for port in self._ports for reference in port.references),
            self.numeric_version,
        )


__all__ = ["SampledScatteringModel", "ScatteringInterpolationPolicy"]
