#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical and numerical conventions for finite-spin magnetic resonance."""

from __future__ import annotations

import math

import equinox as eqx

from ..._strict import StrictModule


HBAR_J_S = 1.054_571_817e-34
MU0_OVER_4PI_N_A2 = 1.0e-7
TWO_PI = 2.0 * math.pi


class MagneticResonanceConvention(StrictModule):
    """One non-configurable convention shared by all resonance profiles.

    Spin matrices are dimensionless and obey ``[Ix, Iy] = i Iz``. Hamiltonian
    generators are stored as ``H / hbar`` in rad/s. A signed gyromagnetic ratio
    gives ``H_Z / hbar = -gamma B.I``. Chemical-shift tensors are dimensionless
    ppm increments and give ``-gamma B.(delta 1e-6).I`` in addition to Zeeman.
    Scalar J and supplied hyperfine/quadrupole tensors are in cycles/s and are
    multiplied by ``2 pi`` exactly once. Molecular-frame vectors and tensors are
    actively rotated into the laboratory frame by active ZYZ Euler rotations.
    Exact evolution is laboratory-frame, pulses are piecewise-constant lab-frame
    fields, and the receiver is the lowering operator ``Ix - i Iy``. The FFT is
    ``dt * FFT(fid)`` with the negative exponential and shifted signed-Hz axis.
    """

    hbar_j_s: float = eqx.field(static=True)
    hamiltonian_unit: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    euler_action: str = eqx.field(static=True)
    pulse_convention: str = eqx.field(static=True)
    receiver_convention: str = eqx.field(static=True)
    fft_convention: str = eqx.field(static=True)

    def __init__(self):
        self.hbar_j_s = HBAR_J_S
        self.hamiltonian_unit = "angular-frequency-rad-s^-1"
        self.frame = "laboratory"
        self.euler_action = "active-ZYZ-molecular-to-laboratory"
        self.pulse_convention = "piecewise-constant-laboratory-field-tesla"
        self.receiver_convention = "sum-weighted-Ix-minus-iIy"
        self.fft_convention = "dt-fft-negative-exponent-shifted-signed-hz"


class MagneticResonanceResourcePolicy(StrictModule):
    """Conservative execution guard, not a statement of released support."""

    maximum_hilbert_dimension: int = eqx.field(static=True)
    maximum_density_elements: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_hilbert_dimension: int = 256,
        maximum_density_elements: int = 65_536,
    ):
        dimension = int(maximum_hilbert_dimension)
        density = int(maximum_density_elements)
        if dimension < 1 or density < 1:
            raise ValueError("Magnetic-resonance resource limits must be positive.")
        self.maximum_hilbert_dimension = dimension
        self.maximum_density_elements = density

    def guard(self, dimension: int, /) -> None:
        value = int(dimension)
        if value > self.maximum_hilbert_dimension:
            raise ValueError(
                f"Hilbert dimension D={value} exceeds maximum_hilbert_dimension="
                f"{self.maximum_hilbert_dimension}."
            )
        entries = value * value
        if entries > self.maximum_density_elements:
            raise ValueError(
                f"Density storage D^2={entries} exceeds maximum_density_elements="
                f"{self.maximum_density_elements}."
            )


def hz_to_angular_rate(frequency_hz: float, /) -> float:
    """Convert cycles/s to signed rad/s."""

    value = float(frequency_hz)
    if not math.isfinite(value):
        raise ValueError("frequency_hz must be finite.")
    return TWO_PI * value


def angular_rate_to_hz(angular_rate_rad_s: float, /) -> float:
    """Convert signed rad/s to cycles/s."""

    value = float(angular_rate_rad_s)
    if not math.isfinite(value):
        raise ValueError("angular_rate_rad_s must be finite.")
    return value / TWO_PI


def chemical_shift_angular_rate(
    gamma_rad_s_t: float,
    field_t: float,
    shift_ppm: float,
    /,
) -> float:
    """Return the signed Hamiltonian coefficient ``-gamma B delta 1e-6``."""

    gamma = float(gamma_rad_s_t)
    field = float(field_t)
    shift = float(shift_ppm)
    if not all(math.isfinite(value) for value in (gamma, field, shift)):
        raise ValueError("gamma_rad_s_t, field_t, and shift_ppm must be finite.")
    return -gamma * field * shift * 1.0e-6


__all__ = [
    "HBAR_J_S",
    "MU0_OVER_4PI_N_A2",
    "TWO_PI",
    "MagneticResonanceConvention",
    "MagneticResonanceResourcePolicy",
    "angular_rate_to_hz",
    "chemical_shift_angular_rate",
    "hz_to_angular_rate",
]
