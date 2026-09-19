#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency axes, harmonic conventions, ports, and network evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import CapabilityProfile, SupportTuple


AmplitudeConvention: TypeAlias = Literal["peak", "rms"]
PhasorConvention: TypeAlias = Literal["exp-positive-iwt", "exp-negative-iwt"]


@dataclass(frozen=True, slots=True)
class HarmonicConvention:
    amplitude: AmplitudeConvention = "peak"
    phasor: PhasorConvention = "exp-positive-iwt"

    @property
    def convention_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "harmonic-convention",
                "amplitude": self.amplitude,
                "phasor": self.phasor,
            }
        )


class FrequencyAxis(StrictModule, NonTrainableState):
    frequency_hz: Array
    axis_id: str = eqx.field(static=True)

    def __init__(self, frequency_hz: ArrayLike, /):
        values = np.asarray(frequency_hz, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(values < 0.0)
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(
                "Frequencies must be finite, non-negative, and strictly increasing."
            )
        self.frequency_hz = jnp.asarray(values)
        self.axis_id = canonical_fingerprint(
            {"kind": "frequency-axis", "frequency_hz": values.tolist()}
        )

    @property
    def angular_frequency_rad_s(self) -> Array:
        return 2.0 * jnp.pi * self.frequency_hz


@dataclass(frozen=True, slots=True)
class Port:
    port_id: str
    reference_impedance_ohm: complex
    orientation: tuple[float, ...]
    frame_id: str

    def __post_init__(self) -> None:
        if not self.port_id or not self.frame_id or not self.orientation:
            raise ValueError("Port identity, frame, and orientation are required.")
        if self.reference_impedance_ohm.real <= 0.0 or not np.isfinite(
            self.reference_impedance_ohm
        ):
            raise ValueError(
                "Port reference impedance must be finite with positive real part."
            )


class ScatteringMatrix(StrictModule, NonTrainableState):
    values: Array
    reference_impedances_ohm: Array
    network_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, reference_impedances_ohm: ArrayLike, /):
        matrix = jnp.asarray(values)
        impedances = jnp.asarray(reference_impedances_ohm)
        if (
            matrix.ndim != 3
            or matrix.shape[-1] != matrix.shape[-2]
            or impedances.shape != (matrix.shape[-1],)
        ):
            raise ValueError(
                "Scattering data require shape (frequency, port, port) and one impedance per port."
            )
        self.values = matrix
        self.reference_impedances_ohm = impedances
        self.network_id = canonical_fingerprint(
            {
                "kind": "scattering-matrix",
                "shape": matrix.shape,
                "impedances": np.asarray(impedances).tolist(),
            }
        )

    @property
    def reciprocal_error(self) -> Array:
        return jnp.max(jnp.abs(self.values - jnp.swapaxes(self.values, -1, -2)))

    @property
    def maximum_power_gain(self) -> Array:
        return jnp.max(jnp.linalg.svd(self.values, compute_uv=False) ** 2)

    @property
    def passive(self) -> Array:
        return self.maximum_power_gain <= 1.0 + 1.0e-10


def frequency_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("frequency.axis-convention", {"axis": "strict-hz", "phasor": "explicit"}),
        (
            "frequency.scattering-network",
            {"matrix": "frequency-port-port", "evidence": "passivity-reciprocity"},
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("convention", "analytic-control", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = [
    "AmplitudeConvention",
    "FrequencyAxis",
    "HarmonicConvention",
    "PhasorConvention",
    "Port",
    "ScatteringMatrix",
    "frequency_candidate_profiles",
]
