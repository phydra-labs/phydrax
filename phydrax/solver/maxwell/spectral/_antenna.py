#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spectral (PSATD) preparation of the sampled plane current antenna.

`PreparedSpectralPlaneAntenna` drives `PreparedSpectralMaxwell` with the B5
`SampledPlaneCurrentAntennaPlan`: the same rest-frame samples, emission
direction, sheet world line, and Lorentz transform, lowered onto the periodic
spectral grid instead of the Maxwell cochain lattice.

The antenna is a moving total-field/scattered-field boundary. With the
simulation-frame incident wave ``(E, H)`` (the Lorentz transform of the
rest-frame wave), ``Θ`` the step onto the emission side ``s(x_n − x_s(t)) > 0``
of the sheet ``x_s(t) = x₀ + v t`` (``v = βc``), and ``(E, H)`` on the sheet at
the rest-frame time ``τ = t/γ``, the field ``Θ·(E, H)`` is radiated exactly by

    J = s δ(x_n − x_s)(n̂ × H + v εE),   M = s δ(x_n − x_s)(−n̂ × E + v μH)

(``∂ₜB = −∇×E − M``): the TFSF curl commutator plus the convective currents
``−D ∂ₜΘ`` and ``−B ∂ₜΘ``. As on the cochain lattice only the sampled
tangential components enter. The delta is band-limited along the normal,

    δ_w(u) = L⁻¹ Σ_k w(k) e^{iku},   w = 1 for |k| ≤ k_N/4,
    w = cos²(2π(|k| − k_N/4)/k_N) for k_N/4 < |k| < k_N/2, w = 0 above,

(``k_N = π/h``) sampled at each current component's own location (Yee offsets
on staggered grids). A sheet source radiates at normal wavenumbers
``k = ±ω/c`` only, where ``w = 1`` for every admitted emission (at least eight
cells per emitted wavelength, refused otherwise): the forward wave is exact
and the Huygens pair cancels exactly behind the sheet. The smooth roll-off
removes the algebraic Gibbs ringing a sharp truncation would leave as a bound
near field, and the empty upper half of the normal spectrum, where numerical
Cherenkov growth lives, stays free of source energy for the NCI monitor. The
tangential profile is bilinear in the samples at the components' tangential
positions; the aperture is the sampled window.

The sheet's divergences are declared charges: ``∂ₜρ_a = −∇⁻·J`` and
``∂ₜρ_m = −∇⁺·M`` are advanced by the solver with the same exact propagator as
the fields, so the Gauss laws hold with them to roundoff.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._fingerprint import canonical_fingerprint
from ...._interpolation import cubic_hermite_interpolate, local_cubic_slopes
from ...._lorentz import boost_wavevector
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....typing import checked
from ..._maxwell_antenna import (
    _transverse_samples,
    SampledPlaneAntennaEvidence,
    SampledPlaneCurrentAntennaPlan,
)


class PreparedSpectralPlaneAntenna(StrictModule, NonTrainableState):
    """A `SampledPlaneCurrentAntennaPlan` prepared on one spectral PSATD grid.

    ``first_samples[t, b, c, (E′_b, H′_c)]`` sample the rest-frame envelopes at
    the ``J_b``/``M_c`` tangential locations and ``second_samples[t, b, c,
    (E′_c, H′_b)]`` at the ``J_c``/``M_b`` ones. ``wavenumbers`` are the normal
    Fourier wavenumbers and ``electric_phase``/``magnetic_phase`` the sheet
    spectra ``w(k)e^{ik(x₀ + o)}`` at the tangential ``J``/``M`` normal offsets
    ``o``; ``evidence`` is the shared B5 antenna evidence.
    """

    times: Array
    first_samples: Array
    first_slopes: Array
    second_samples: Array
    second_slopes: Array
    wavenumbers: Array
    electric_phase: Array
    magnetic_phase: Array
    evidence: SampledPlaneAntennaEvidence
    normal_axis: int = eqx.field(static=True)
    emission_sign: float = eqx.field(static=True)
    orientation: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    lorentz_factor: float = eqx.field(static=True)
    boost_speed: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)
    plane_coordinate: float = eqx.field(static=True)
    grid_velocity: float = eqx.field(static=True)
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    counts: tuple[int, int, int] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: SampledPlaneCurrentAntennaPlan,
        counts: tuple[int, int, int],
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        offsets: tuple[np.ndarray, np.ndarray],
        magnetic_symbols: tuple[np.ndarray, np.ndarray],
        grid_velocity: float,
        /,
    ) -> None:
        """Lower ``plan`` onto the grid ``counts``/``spacing``/``origin``.

        ``offsets`` are the half-cell ``[component, axis]`` offsets of ``E`` and
        ``B``; ``magnetic_symbols`` the solver's ``∇⁺`` symbols along the two
        tangential axes (for the magnetic-sheet closure evidence);
        ``grid_velocity`` the Galilean grid velocity along the normal.
        """
        a = plan.normal_axis
        b, c = plan.tangential_axes
        electric_offsets, magnetic_offsets = offsets

        def coordinates(axis: int, offset: float) -> np.ndarray:
            return origin[axis] + (np.arange(counts[axis]) + offset) * spacing[axis]

        # J_b and M_c share the tangential location of E_b; J_c and M_b that of E_c.
        first_b = coordinates(b, electric_offsets[b, b])
        first_c = coordinates(c, electric_offsets[b, c])
        second_b = coordinates(b, electric_offsets[c, b])
        second_c = coordinates(c, electric_offsets[c, c])
        shape_first = (plan.times.shape[0], first_b.size, first_c.size)
        shape_second = (plan.times.shape[0], second_b.size, second_c.size)
        e_b, support_first = _transverse_samples(plan, first_b, first_c, 0, plan.electric)
        h_c, _ = _transverse_samples(plan, first_b, first_c, 1, plan.magnetic)
        e_c, support_second = _transverse_samples(
            plan, second_b, second_c, 1, plan.electric
        )
        h_b, _ = _transverse_samples(plan, second_b, second_c, 0, plan.magnetic)
        first = jnp.stack((e_b.reshape(shape_first), h_c.reshape(shape_first)), axis=-1)
        second = jnp.stack(
            (e_c.reshape(shape_second), h_b.reshape(shape_second)), axis=-1
        )
        sign = plan.emission_sign
        orientation = 1.0 if (a, b, c) != (1, 0, 2) else -1.0
        gamma = plan.lorentz_factor
        speed = (
            plan.medium.wave_speed
            if plan.scale is None
            else float(plan.scale.speed_of_light)
        )
        times = np.asarray(plan.times, dtype=np.float64)
        window = (gamma * times[0], gamma * times[-1])
        # Sheet grid coordinate (in cells) at the window ends; it moves linearly.
        velocity = plan.beta * speed - grid_velocity
        sheet = (
            plan.plane_coordinate + velocity * np.asarray(window) - origin[a]
        ) / spacing[a]
        node_range = (int(np.floor(sheet.min())), int(np.ceil(sheet.max())))
        interval_range = (
            int(np.floor(sheet.min() - 0.5)),
            int(np.ceil(sheet.max() - 0.5)),
        )
        normal = np.zeros(3)
        normal[a] = 1.0
        # The simulation frame moves with −β relative to the antenna rest frame.
        emitted = boost_wavevector(
            jnp.asarray(-plan.beta * normal),
            jnp.asarray(plan.carrier_angular_frequency),
            jnp.asarray(sign * normal),
        )[0]
        wavelength_cells = jnp.where(
            emitted > 0.0,
            2.0
            * jnp.pi
            * plan.medium.wave_speed
            / jnp.maximum(emitted, 1e-300)
            / spacing[a],
            jnp.inf,
        )
        if float(wavelength_cells) < 8.0:
            raise ValueError(
                "The spectral antenna sheet is exact up to a quarter of the Nyquist "
                "wavenumber; the emitted carrier needs at least eight cells per "
                f"wavelength along the normal (has {float(wavelength_cells):.3g})."
            )
        support = jnp.concatenate((support_first, support_second))
        evidence = SampledPlaneAntennaEvidence(
            jnp.mean(support.astype(jnp.float64)),
            jnp.asarray(emitted, dtype=jnp.float64),
            jnp.asarray(wavelength_cells, dtype=jnp.float64),
            jnp.asarray(gamma, dtype=jnp.float64),
            jnp.asarray(
                _closure_defect(
                    np.asarray(first[..., 0]),
                    np.asarray(second[..., 0]),
                    magnetic_symbols,
                    min(spacing[b], spacing[c]),
                ),
                dtype=jnp.float64,
            ),
            jnp.asarray(window, dtype=jnp.float64),
            jnp.asarray(node_range, dtype=jnp.int32),
            jnp.asarray(interval_range, dtype=jnp.int32),
            jnp.all(support),
        )
        self.times = plan.times
        self.first_samples = first
        self.first_slopes = local_cubic_slopes(plan.times, first, axis=0)
        self.second_samples = second
        self.second_slopes = local_cubic_slopes(plan.times, second, axis=0)
        # Tangential J_b, J_c share one normal offset, as do M_b, M_c.
        wavenumbers, electric_phase, magnetic_phase = _sheet_spectrum(
            counts[a],
            spacing[a],
            origin[a],
            (electric_offsets[b, a], magnetic_offsets[b, a]),
        )
        self.evidence = evidence
        self.normal_axis = a
        self.emission_sign = sign
        self.orientation = orientation
        self.beta = plan.beta
        self.lorentz_factor = gamma
        self.boost_speed = speed
        self.carrier_angular_frequency = plan.carrier_angular_frequency
        self.plane_coordinate = plan.plane_coordinate
        self.grid_velocity = float(grid_velocity)
        self.permittivity = plan.medium.permittivity
        self.permeability = plan.medium.permeability
        self.length = counts[a] * spacing[a]
        self.wavenumbers = jnp.asarray(wavenumbers)
        self.electric_phase = jnp.asarray(electric_phase)
        self.magnetic_phase = jnp.asarray(magnetic_phase)
        self.counts = counts
        self.source_id = plan.source_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-plane-antenna",
                "source": plan.source_id,
                "counts": list(counts),
                "spacing": [float(value).hex() for value in spacing],
                "origin": [float(value).hex() for value in origin],
                "offsets": [np.asarray(value).tolist() for value in offsets],
                "grid_velocity": float(grid_velocity).hex(),
            }
        )

    @property
    def tangential_axes(self) -> tuple[int, int]:
        first, second = (axis for axis in range(3) if axis != self.normal_axis)
        return (first, second)

    def sheet_position(self, time: Array, /) -> Array:
        """Grid coordinate of the sheet along the normal at simulation time ``time``."""
        return (
            self.plane_coordinate
            + (self.beta * self.boost_speed - self.grid_velocity) * time
        )

    def _delta(self, phase: Array, time: Array, /) -> Array:
        """Band-limited periodic sheet ``δ_w(x_n − x_s(t))`` on the grid points.

        ``δ_w(x_j) = L⁻¹ Σ_m ŝ_m e^{ik_m x_j}`` with ``ŝ = phase·e^{−ik x_s}`` is
        ``(N/L)`` times the inverse DFT of ``ŝ``; the taper is symmetric with an
        empty Nyquist plane, so the sheet is real.
        """
        spectrum = phase * jnp.exp(-1j * self.wavenumbers * self.sheet_position(time))
        return jnp.real(jnp.fft.ifft(spectrum)) * (
            self.wavenumbers.shape[0] / self.length
        )

    def sources(self, time: Array, /) -> tuple[Array, Array]:
        """Real-space ``J`` and ``M`` ``[N₀, N₁, N₂, 3]`` at simulation time ``time``."""
        retarded = time / self.lorentz_factor
        carrier = jnp.exp(-1j * self.carrier_angular_frequency * retarded)

        def rest(samples: Array, slopes: Array) -> Array:
            envelope = cubic_hermite_interpolate(
                self.times, samples, retarded, slopes=slopes, bounds="fill"
            ).values
            return jnp.real(envelope * carrier)

        first = rest(self.first_samples, self.first_slopes)
        second = rest(self.second_samples, self.second_slopes)
        impedance = math.sqrt(self.permeability / self.permittivity)
        tilt = self.beta * self.orientation
        gamma = self.lorentz_factor
        # Tangential boost E = γ(E′ − v × B′), H = γ(H′ + v × D′), v = βc n̂.
        e_b = gamma * (first[..., 0] + tilt * impedance * first[..., 1])
        h_c = gamma * (first[..., 1] + tilt * first[..., 0] / impedance)
        e_c = gamma * (second[..., 0] - tilt * impedance * second[..., 1])
        h_b = gamma * (second[..., 1] - tilt * second[..., 0] / impedance)
        sign, epsilon = self.emission_sign, self.orientation
        speed = self.beta * self.boost_speed
        # (n̂ × F)_b = −ε F_c and (n̂ × F)_c = ε F_b with ε the (n, b, c) orientation.
        j_b = sign * (-epsilon * h_c + speed * self.permittivity * e_b)
        j_c = sign * (epsilon * h_b + speed * self.permittivity * e_c)
        m_b = sign * (epsilon * e_c + speed * self.permeability * h_b)
        m_c = sign * (-epsilon * e_b + speed * self.permeability * h_c)
        electric = self._delta(self.electric_phase, time)
        magnetic = self._delta(self.magnetic_phase, time)
        return (
            self._place(electric, j_b, j_c),
            self._place(magnetic, m_b, m_c),
        )

    def _place(self, normal: Array, first: Array, second: Array, /) -> Array:
        """Outer product of the normal delta with the tangential components."""
        b, c = self.tangential_axes
        zero = jnp.zeros((normal.size, *first.shape), dtype=first.dtype)
        components = [zero, zero, zero]
        components[b] = normal[:, None, None] * first[None]
        components[c] = normal[:, None, None] * second[None]
        # Arrays are ordered (n, b, c); b < c, so moving n into place restores
        # the grid axis order.
        return jnp.moveaxis(jnp.stack(components, axis=-1), 0, self.normal_axis)


def _sheet_spectrum(
    count: int, spacing: float, origin: float, offsets: tuple[float, float], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal wavenumbers and the tapered sheet spectra at two component offsets.

    ``w(k)`` is one up to a quarter of the Nyquist wavenumber ``k_N = π/h``,
    rolls off as ``cos²`` to zero at ``k_N/2``, and vanishes above.
    """
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    quarter = 0.25 * np.pi / spacing
    fraction = np.clip((np.abs(wavenumbers) - quarter) / quarter, 0.0, 1.0)
    taper = np.cos(0.5 * np.pi * fraction) ** 2
    electric, magnetic = (
        taper * np.exp(1j * wavenumbers * (origin + offset * spacing))
        for offset in offsets
    )
    return wavenumbers, electric, magnetic


def _closure_defect(
    electric_first: np.ndarray,
    electric_second: np.ndarray,
    symbols: tuple[np.ndarray, np.ndarray],
    spacing: float,
    /,
) -> float:
    """Relative ``∇⁺`` tangential divergence of the rest-frame magnetic sheet.

    ``K_m = −s n̂ × E′`` has components ``(ε E′_c, −ε E′_b)`` up to the common
    sign, whose magnitude is irrelevant to the relative defect.
    """
    sheet_b = np.fft.fft2(electric_second, axes=(1, 2))
    sheet_c = -np.fft.fft2(electric_first, axes=(1, 2))
    divergence = symbols[0][None, :, None] * sheet_b + symbols[1][None, None, :] * sheet_c
    scale = max(np.max(np.abs(electric_first)), np.max(np.abs(electric_second)))
    if scale == 0.0:
        return 0.0
    values = np.fft.ifft2(divergence, axes=(1, 2))
    return float(np.max(np.abs(values)) * spacing / scale)


__all__ = ["PreparedSpectralPlaneAntenna"]
