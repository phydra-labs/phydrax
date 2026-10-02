#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Undulator lattices for the period-averaged free-electron laser.

A lattice is an ordered sequence of :class:`FELUndulatorSegment` entries. Each
segment wraps one X1a :class:`InsertionDeviceField` (its period, period count,
peak field, and polarization), followed by a field-free break that can carry a
thin quadrupole and a phase shifter. Stepwise taper is expressed through the
segments' peak fields. Inside a segment the averaged model uses the flat-top
length ``N λ_u``; the matched smooth terminations of the X1a device are
absorbed into the adjacent break.

Undulator quantities follow the planar/helical conventions of the insertion
device: ``K = e B₀ λ_u / (2π mₑ c)`` with rms strength ``a_w² = K²/2``
(planar) or ``a_w² = K²`` (helical). The harmonic-``h`` coupling of a planar
device to a linearly polarized field is
``[JJ]_h = (−1)^((h−1)/2) [J_((h−1)/2)(hξ) − J_((h+1)/2)(hξ)]`` with
``ξ = K² / (4 + 2K²)``; a helical device couples only its fundamental on axis
(``[JJ]_1 = 1``). Even planar harmonics and helical harmonics above one vanish
on axis and are refused rather than silently zeroed.
"""

from __future__ import annotations

import math
from typing import assert_never, NamedTuple

import equinox as eqx
import jax
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ...._physical import ElectromagneticScaleContract
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar, positive_finite_float
from ....special import jv
from ....typing import checked, ConvertibleToArray
from .._field_map import InsertionDeviceField, InsertionDevicePolarization


# Host preparation evaluates the native Bessel function through one stable
# compiled entry point instead of op-by-op dispatch.
_BESSEL_J = jax.jit(jv)


def undulator_rms_strength_squared(
    deflection: float, polarization: InsertionDevicePolarization, /
) -> float:
    """``a_w²``: ``K²/2`` for a planar device, ``K²`` for a helical one."""
    match polarization:
        case "planar":
            return 0.5 * deflection * deflection
        case "helical":
            return deflection * deflection
        case _:
            assert_never(polarization)


def undulator_coupling_factors(
    deflections: ConvertibleToArray,
    harmonics: tuple[int, ...],
    polarization: InsertionDevicePolarization,
    /,
) -> np.ndarray:
    """Return the on-axis Bessel couplings ``[JJ]_h`` as ``[deflection, harmonic]``.

    Refuses harmonics without on-axis coupling: even planar harmonics and every
    helical harmonic above the fundamental.
    """
    strength = np.asarray(deflections, dtype=np.float64).reshape((-1,))
    if (
        strength.shape[0] < 1
        or not np.all(np.isfinite(strength))
        or np.any(strength <= 0.0)
    ):
        raise ValueError("deflections must be finite and positive.")
    orders_ = tuple(harmonics)
    if not orders_:
        raise ValueError("At least one harmonic is required.")
    for harmonic in orders_:
        if isinstance(harmonic, bool) or not isinstance(harmonic, int) or harmonic < 1:
            raise ValueError("harmonics must be positive integers.")
    match polarization:
        case "planar":
            if any(harmonic % 2 == 0 for harmonic in orders_):
                raise ValueError(
                    "Even planar-undulator harmonics have no on-axis coupling in the "
                    "period-averaged model."
                )
            orders = np.asarray(orders_, dtype=np.float64)
            xi = strength * strength / (4.0 + 2.0 * strength * strength)
            argument = orders[None, :] * xi[:, None]
            bessel_order = np.broadcast_to((orders[None, :] - 1.0) / 2.0, argument.shape)
            # One vectorized native Bessel evaluation for both orders.
            values = np.asarray(
                _BESSEL_J(
                    np.concatenate((bessel_order, bessel_order + 1.0)),
                    np.concatenate((argument, argument)),
                ),
                dtype=np.float64,
            )
            lower, upper = values[: strength.shape[0]], values[strength.shape[0] :]
            sign = np.where(((orders - 1.0) / 2.0) % 2.0 == 1.0, -1.0, 1.0)
            return sign[None, :] * (lower - upper)
        case "helical":
            if orders_ != (1,):
                raise ValueError(
                    "A helical undulator couples only its fundamental on axis."
                )
            return np.ones((strength.shape[0], 1), dtype=np.float64)
        case _:
            assert_never(polarization)


def _deflection(
    scale: ElectromagneticScaleContract, device: InsertionDeviceField, /
) -> float:
    return (
        float(scale.elementary_charge)
        * abs(device.peak_field)
        * device.period
        / (2.0 * math.pi * float(scale.electron_mass) * float(scale.speed_of_light))
    )


class FELUndulatorSegment(StrictModule, NonTrainableState):
    """One undulator module followed by a field-free break.

    ``smooth_focusing_gradient`` ``G_s`` (field per length, scale units) adds a
    symmetric FODO-averaged focusing ``k² = e G_s / (γ mₑ c)`` in both planes
    inside the module on top of the device's natural focusing.
    ``quadrupole_integrated_gradient`` ``∫G dl`` is a thin quadrupole at the
    break center (``Δx′ = −e∫G dl · x/(γ mₑ c)``, opposite sign in ``y``);
    ``phase_shift`` (radians) advances every particle's ponderomotive phase at
    the break center.
    """

    device: InsertionDeviceField
    drift_length: float = eqx.field(static=True)
    smooth_focusing_gradient: float = eqx.field(static=True)
    quadrupole_integrated_gradient: float = eqx.field(static=True)
    phase_shift: float = eqx.field(static=True)
    segment_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        device: InsertionDeviceField,
        /,
        *,
        drift_length: float = 0.0,
        smooth_focusing_gradient: float = 0.0,
        quadrupole_integrated_gradient: float = 0.0,
        phase_shift: float = 0.0,
    ) -> None:
        drift = finite_real_scalar(drift_length, "drift_length")
        if drift < 0.0:
            raise ValueError("drift_length must be nonnegative.")
        smooth = finite_real_scalar(smooth_focusing_gradient, "smooth_focusing_gradient")
        if smooth < 0.0:
            raise ValueError("smooth_focusing_gradient must be nonnegative.")
        quadrupole = finite_real_scalar(
            quadrupole_integrated_gradient, "quadrupole_integrated_gradient"
        )
        shift = finite_real_scalar(phase_shift, "phase_shift")
        if drift == 0.0 and (quadrupole != 0.0 or shift != 0.0):
            raise ValueError(
                "A thin quadrupole or phase shifter needs a break with positive drift_length."
            )
        self.device = device
        self.drift_length = drift
        self.smooth_focusing_gradient = smooth
        self.quadrupole_integrated_gradient = quadrupole
        self.phase_shift = shift
        self.segment_id = canonical_fingerprint(
            {
                "kind": "fel-undulator-segment",
                "device": device.source_id,
                "drift_length": drift,
                "smooth_focusing_gradient": smooth,
                "quadrupole_integrated_gradient": quadrupole,
                "phase_shift": shift,
            }
        )


class FELStepTable(NamedTuple):
    """Host step schedule of a lattice (one row per integration step).

    ``natural_focusing`` holds the lattice's per-plane multipliers ``c`` of
    ``k² = c a_w² k_u² / γ²`` (zero in breaks because ``a_w = 0``);
    ``smooth_focusing`` and ``quadrupole_kick`` are ``e G_s/(mₑ c)`` and
    ``e∫G dl/(mₑ c)`` (``k² = value/γ``).
    """

    lengths: np.ndarray
    end_positions: np.ndarray
    undulator_wavenumbers: np.ndarray
    deflections: np.ndarray
    rms_strength_squared: np.ndarray
    natural_focusing: np.ndarray
    smooth_focusing: np.ndarray
    quadrupole_kick: np.ndarray
    phase_shift: np.ndarray
    segment_index: np.ndarray


class FELUndulatorLattice(StrictModule, NonTrainableState):
    """Ordered undulator segments bound to one electromagnetic scale.

    Every segment shares one polarization (the averaged field is one scalar
    polarization state). Each module is divided into
    ``ceil(N λ_u / step_length)`` equal steps; each break is one exact step.
    """

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    segments: tuple[FELUndulatorSegment, ...]
    polarization: InsertionDevicePolarization = eqx.field(static=True)
    step_length: float = eqx.field(static=True)
    deflections: tuple[float, ...] = eqx.field(static=True)
    fundamental_couplings: tuple[float, ...] = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        segments: tuple[FELUndulatorSegment, ...],
        /,
        *,
        step_length: float,
    ) -> None:
        segments_ = tuple(segments)
        if not segments_:
            raise ValueError("An FEL lattice needs at least one undulator segment.")
        for segment in segments_:
            if not isinstance(segment, FELUndulatorSegment):
                raise TypeError("Every lattice entry must be an FELUndulatorSegment.")
        polarizations = {segment.device.polarization for segment in segments_}
        if len(polarizations) != 1:
            raise ValueError(
                "Every FEL undulator segment must share one polarization; the "
                "averaged field carries one polarization state."
            )
        self.scale = scale
        self.segments = segments_
        self.polarization = segments_[0].device.polarization
        self.step_length = positive_finite_float(step_length, "step_length")
        self.deflections = tuple(
            _deflection(scale, segment.device) for segment in segments_
        )
        self.fundamental_couplings = tuple(
            float(value) for value in self.coupling_table((1,))[:, 0]
        )
        self.lattice_id = canonical_fingerprint(
            {
                "kind": "fel-undulator-lattice",
                "scale": scale.scale_id,
                "segments": [segment.segment_id for segment in segments_],
                "step_length": self.step_length,
            }
        )

    @property
    def length(self) -> float:
        """Total lattice length: module flat tops plus breaks."""
        return sum(
            segment.device.period * segment.device.period_count + segment.drift_length
            for segment in self.segments
        )

    def resonant_wavelength(self, lorentz_factor: float, /, *, segment: int = 0) -> float:
        """Fundamental ``λ_u (1 + a_w²) / (2γ²)`` of one segment."""
        gamma = positive_finite_float(lorentz_factor, "lorentz_factor")
        if gamma <= 1.0:
            raise ValueError("lorentz_factor must exceed one.")
        device = self.segments[segment].device
        strength = undulator_rms_strength_squared(
            self.deflections[segment], self.polarization
        )
        return device.period * (1.0 + strength) / (2.0 * gamma * gamma)

    def coupling_table(self, harmonics: tuple[int, ...], /) -> np.ndarray:
        """``κ_h = a_w [JJ]_h / √2`` as ``[segment, harmonic]``."""
        deflections = np.asarray(self.deflections, dtype=np.float64)
        strengths = np.sqrt(
            np.asarray(
                [
                    undulator_rms_strength_squared(value, self.polarization)
                    for value in self.deflections
                ],
                dtype=np.float64,
            )
        )
        factors = undulator_coupling_factors(deflections, harmonics, self.polarization)
        return strengths[:, None] * factors / math.sqrt(2.0)

    def step_table(self) -> FELStepTable:
        """Build the host step schedule used by the averaged solver."""
        charge_over_momentum = float(self.scale.elementary_charge) / (
            float(self.scale.electron_mass) * float(self.scale.speed_of_light)
        )
        match self.polarization:
            case "planar":
                natural = (0.0, 1.0)
            case "helical":
                natural = (0.5, 0.5)
            case _:
                assert_never(self.polarization)
        # Row: length, k_u, K, e G_s/(m c), e∫G dl/(m c), phase shift, segment.
        rows: list[tuple[float, float, float, float, float, float, int]] = []
        for index, (segment, deflection) in enumerate(
            zip(self.segments, self.deflections, strict=True)
        ):
            device = segment.device
            module = device.period * device.period_count
            count = max(1, math.ceil(module / self.step_length - 1.0e-12))
            wavenumber = 2.0 * math.pi / device.period
            smooth = charge_over_momentum * segment.smooth_focusing_gradient
            rows.extend(
                (module / count, wavenumber, deflection, smooth, 0.0, 0.0, index)
                for _ in range(count)
            )
            if segment.drift_length > 0.0:
                quadrupole = charge_over_momentum * segment.quadrupole_integrated_gradient
                rows.append(
                    (
                        segment.drift_length,
                        0.0,
                        0.0,
                        0.0,
                        quadrupole,
                        segment.phase_shift,
                        index,
                    )
                )
        lengths = np.asarray([row[0] for row in rows], dtype=np.float64)
        deflections = np.asarray([row[2] for row in rows], dtype=np.float64)
        return FELStepTable(
            lengths=lengths,
            end_positions=np.cumsum(lengths),
            undulator_wavenumbers=np.asarray([row[1] for row in rows], dtype=np.float64),
            deflections=deflections,
            rms_strength_squared=np.asarray(
                [
                    undulator_rms_strength_squared(row[2], self.polarization)
                    for row in rows
                ],
                dtype=np.float64,
            ),
            natural_focusing=np.asarray(natural, dtype=np.float64),
            smooth_focusing=np.asarray([row[3] for row in rows], dtype=np.float64),
            quadrupole_kick=np.asarray([row[4] for row in rows], dtype=np.float64),
            phase_shift=np.asarray([row[5] for row in rows], dtype=np.float64),
            segment_index=np.asarray([row[6] for row in rows], dtype=np.int32),
        )


__all__ = [
    "FELStepTable",
    "FELUndulatorLattice",
    "FELUndulatorSegment",
    "undulator_coupling_factors",
    "undulator_rms_strength_squared",
]
