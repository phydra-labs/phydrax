#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamic space charge of the time-dependent averaged FEL.

Every integration step evaluates the collective field from the current
particles and applies it as the kick ``K(Δz)`` at the center of the Strang
step (``S(Δz/2) F(Δz) K(Δz) S(Δz/2)``; ``K`` depends only on phases and
positions, so the composition stays symmetric).

Intra-slice (wavelength-scale) longitudinal space charge
    Harmonic ``l = 1 … L`` of the slice's ponderomotive-phase density drives
    ``dγ/dz = Σ_l 2 Re(E_l e^{ilθ})``. In the mean-motion frame (Lorentz factor
    ``γ_z² = γ_r²/(1 + a_w²)``) the field solves
    ``[1 − γ_z²/(l k)² ∇⊥²] E_l = −i (I/(ε₀ c l k)) (e/mₑc²) ρ̂_l`` with
    ``ρ̂_l = N⁻¹ Σ_j δ²(x⊥ − x_j) e^{−ilθ_j}`` (Genesis 1.3 version 4 short-range
    model). The ``"angular-spectrum"`` model solves it on a radial grid of
    ``radial_cells`` annuli of width ``R/(G − 1)`` centered on the slice
    centroid (finite-volume, azimuthal modes ``|m| ≤ azimuthal_modes``, zero
    field one cell beyond the grid), with ``R = max(radial_extent, 1.5 r_max)``;
    the ``"one-dimensional"`` model treats the slice as a uniform disk of the
    slice area ``A = 2π σ_x σ_y`` (current rms sizes) and applies the
    transversely averaged reduction ``F_l = 1 − 2 I₁(ξ_l) K₁(ξ_l)``,
    ``ξ_l = l k √(A/π)/γ_z``: ``E_l = −i (I/(ε₀ c l k)) (e/mₑc²) F_l b_l/A``.
    For a wide cold beam this gives plasma oscillations at
    ``k_p² = e² n F/(ε₀ mₑ c² γ γ_z²)``.

Bunch-scale (inter-slice) field
    The X3 :class:`SpaceChargeIGFPlan` deposits every macroparticle at its
    slice center in the mean-motion frame ``γ_z`` of the current step (particle
    Lorentz factor ``γ/√(1 + a_w²)``, transverse velocity ``u⊥/γ``) and
    returns ``Δ(p c) = q (E + v × B) Δz/β_z``. The energy change is
    ``Δγ = β · Δ(p c)/(mₑc²)``; the transverse increment ``Δu⊥ = Δ(p⊥c)/(mₑc²)``
    (``∝ 1/γ_z²``) is applied only for ``transverse="applied"``. Either way the
    accumulated rms transverse kick relative to the entrance rms ``u⊥`` is
    reported per slice; an ``"omitted"`` transverse kick above
    ``transverse_tolerance`` raises
    ``FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED``. Particles stay in their
    slice, so the bunch-scale field acts on the fixed current profile with the
    current transverse sizes.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar, nonnegative_integer
from ....linalg import solve_tridiagonal_lines
from ....special import ive, kve
from ....typing import parse
from .._beam import AcceleratorBunch
from .._space_charge import SpaceChargeIGFPlan


FELTransverseSpaceCharge: TypeAlias = Literal["applied", "omitted"]


class FELSpaceCharge(StrictModule, NonTrainableState):
    """Space charge of the time-dependent FEL, evaluated every step.

    ``bunch`` is the X3 plan of the bunch-scale field (``None`` omits it; it
    needs an open window, and its capacity must equal slices × particles per
    slice); ``harmonics`` is the number ``L`` of intra-slice longitudinal
    harmonics (``0`` omits them; loading needs
    ``particles_per_beamlet ≥ 2L``). ``radial_extent``, ``radial_cells``,
    and ``azimuthal_modes`` configure the radial solve of the grid model.
    ``transverse`` declares whether the X3 transverse kick is applied; it can
    be ``"applied"`` only with ``bunch``.
    """

    bunch: SpaceChargeIGFPlan | None
    harmonics: int = eqx.field(static=True)
    radial_extent: float = eqx.field(static=True)
    radial_cells: int = eqx.field(static=True)
    azimuthal_modes: int = eqx.field(static=True)
    transverse: FELTransverseSpaceCharge = eqx.field(static=True)
    transverse_tolerance: float = eqx.field(static=True)
    space_charge_id: str = eqx.field(static=True)

    def __init__(
        self,
        bunch: SpaceChargeIGFPlan | None = None,
        /,
        *,
        transverse: FELTransverseSpaceCharge,
        harmonics: int = 0,
        radial_extent: float = 0.0,
        radial_cells: int = 100,
        azimuthal_modes: int = 0,
        transverse_tolerance: float = 1.0e-2,
    ) -> None:
        if bunch is not None and not isinstance(bunch, SpaceChargeIGFPlan):
            raise TypeError("bunch must be a SpaceChargeIGFPlan or None.")
        transverse_ = parse(transverse, FELTransverseSpaceCharge, "transverse")
        orders = nonnegative_integer(harmonics, "harmonics")
        extent = finite_real_scalar(radial_extent, "radial_extent")
        cells = nonnegative_integer(radial_cells, "radial_cells")
        modes = nonnegative_integer(azimuthal_modes, "azimuthal_modes")
        tolerance = finite_real_scalar(transverse_tolerance, "transverse_tolerance")
        if bunch is None and orders == 0:
            raise ValueError(
                "FELSpaceCharge needs the bunch-scale plan or at least one harmonic."
            )
        if bunch is None and transverse_ == "applied":
            raise ValueError("The transverse kick comes from the bunch-scale plan.")
        if extent < 0.0 or cells < 2 or tolerance <= 0.0:
            raise ValueError(
                "radial_extent must be nonnegative, radial_cells at least two, and "
                "transverse_tolerance positive."
            )
        self.bunch = bunch
        self.harmonics = orders
        self.radial_extent = extent
        self.radial_cells = cells
        self.azimuthal_modes = modes
        self.transverse = transverse_
        self.transverse_tolerance = tolerance
        self.space_charge_id = canonical_fingerprint(
            {
                "kind": "fel-space-charge",
                "bunch": None if bunch is None else bunch.plan_id,
                "harmonics": orders,
                "radial_extent": extent,
                "radial_cells": cells,
                "azimuthal_modes": modes,
                "transverse": transverse_,
                "transverse_tolerance": tolerance,
            }
        )


def _centroid_offsets(positions: Array, /) -> Array:
    return positions - jnp.mean(positions, axis=1, keepdims=True)


def uniform_harmonic_rates(
    phases: Array,
    positions: Array,
    source: Array,
    orders: Array,
    wavenumber: float,
    frame_squared: Array,
    /,
) -> tuple[Array, Array]:
    """One-dimensional harmonic LSC ``dγ/dz [S, N]`` and the finite-area flag.

    ``source[s] = I_s e/(ε₀ c k mₑc²)``; ``orders`` are ``1 … L``.
    """
    offsets = _centroid_offsets(positions)
    variance = jnp.mean(offsets * offsets, axis=1)
    area = 2.0 * jnp.pi * jnp.sqrt(variance[:, 0] * variance[:, 1])
    finite_area = area > 0.0
    safe_area = jnp.where(finite_area, area, 1.0)
    argument = (
        orders[None, :]
        * wavenumber
        * jnp.sqrt(safe_area / jnp.pi)[:, None]
        / jnp.sqrt(frame_squared)
    )
    reduction = 1.0 - 2.0 * ive(1.0, argument) * kve(1.0, argument)
    rotation = jnp.exp(1j * orders[None, None, :] * phases[:, :, None])
    bunching = jnp.mean(jnp.conj(rotation), axis=1)
    field = (
        -1j
        * source[:, None]
        / orders[None, :]
        * reduction
        * bunching
        / safe_area[:, None]
    )
    rates = 2.0 * jnp.sum(jnp.real(field[:, None, :] * rotation), axis=-1)
    return jnp.where(finite_area[:, None], rates, 0.0), jnp.all(finite_area)


def radial_harmonic_rates(
    phases: Array,
    positions: Array,
    source: Array,
    orders: Array,
    wavenumber: float,
    frame_squared: Array,
    config: FELSpaceCharge,
    /,
) -> tuple[Array, Array]:
    """Radial-grid harmonic LSC ``dγ/dz [S, N]`` and the solve's success flag."""
    slices, particles = phases.shape
    cells = config.radial_cells
    offsets = _centroid_offsets(positions)
    radius = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1))
    extent = jnp.maximum(config.radial_extent, 1.5 * jnp.max(radius, axis=1))
    finite_extent = extent > 0.0
    width = jnp.where(finite_extent, extent, 1.0) / (cells - 1)
    index = jnp.clip(jnp.floor(radius / width[:, None]), 0, cells - 1).astype(jnp.int32)
    azimuth = jnp.arctan2(offsets[..., 1], offsets[..., 0])
    modes = jnp.arange(
        -config.azimuthal_modes, config.azimuthal_modes + 1, dtype=jnp.float64
    )
    rotation = jnp.exp(
        1j
        * (
            modes[None, None, :, None] * azimuth[:, :, None, None]
            + orders[None, None, None, :] * phases[:, :, None, None]
        )
    )
    segments = (jnp.arange(slices, dtype=jnp.int32)[:, None] * cells + index).reshape(-1)
    deposited = jax.ops.segment_sum(
        jnp.conj(rotation).reshape(slices * particles, -1),
        segments,
        num_segments=slices * cells,
    ).reshape(slices, cells, modes.shape[0], orders.shape[0])
    deposited = jnp.moveaxis(deposited, 1, -1)
    ring = jnp.arange(cells, dtype=jnp.float64)
    area = jnp.pi * width[:, None] ** 2 * (2.0 * ring[None, :] + 1.0)
    # ∫ 2π dr/r over annulus j is 2π ln((j+1)/j); the axis disk weight
    # (|m| + 2)/(2|m|) is exact for the regular solution E ∝ r^{|m|}.
    safe_ring = jnp.maximum(ring, 1.0)
    logarithm = jnp.log((safe_ring + 1.0) / safe_ring)
    axis_weight = (jnp.abs(modes) + 2.0) / (2.0 * jnp.maximum(jnp.abs(modes), 1.0))
    weight = jnp.where(ring[None, :] == 0.0, axis_weight[:, None], logarithm[None, :])
    screening = frame_squared / (orders * wavenumber) ** 2
    coefficient = 2.0 * jnp.pi * screening[None, None, :, None] / area[:, None, None, :]
    shape = (slices, modes.shape[0], orders.shape[0], cells)
    lower = jnp.broadcast_to(-coefficient * ring, shape)
    upper = jnp.broadcast_to(-coefficient * (ring + 1.0), shape)
    diagonal = 1.0 + coefficient * (
        2.0 * ring + 1.0 + (modes * modes)[None, :, None, None] * weight[None, :, None, :]
    )
    rhs = (
        -1j
        * (source[:, None] / orders[None, :])[:, None, :, None]
        * deposited
        / (particles * area[:, None, None, :])
    )
    solved = solve_tridiagonal_lines(
        lower.astype(jnp.complex128),
        diagonal.astype(jnp.complex128),
        upper.astype(jnp.complex128),
        rhs,
    )
    field = jnp.take_along_axis(
        jnp.moveaxis(solved.value, -1, 1), index[:, :, None, None], axis=1
    )
    rates = 2.0 * jnp.sum(jnp.real(field * rotation), axis=(-2, -1))
    return rates, solved.successful & jnp.all(finite_extent)


def bunch_kick(
    plan: SpaceChargeIGFPlan,
    lorentz_factors: Array,
    positions: Array,
    momenta: Array,
    slice_positions: Array,
    electrons: Array,
    strength_squared: Array,
    reference_lorentz_factor: float,
    rest_energy: float,
    step_length: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """X3 ``Δγ [S, N]``, ``Δu⊥ [S, N, 2]``, acceptance, and cells per σ."""
    slices, particles = lorentz_factors.shape
    stretch = jnp.sqrt(1.0 + strength_squared)
    # Mean longitudinal motion: γ_p = γ/√(1 + a_w²) and u⊥,p = u⊥/√(1 + a_w²)
    # keep the real transverse velocity u⊥/γ and 1 − β_z² = (1 + a_w² + u⊥²)/γ².
    gamma = lorentz_factors / stretch
    transverse = momenta / stretch
    total = jnp.sqrt(gamma * gamma - 1.0)
    reference = math.sqrt(reference_lorentz_factor**2 - 1.0)
    coordinates = jnp.stack(
        (
            positions[..., 0],
            transverse[..., 0] / reference,
            positions[..., 1],
            transverse[..., 1] / reference,
            jnp.broadcast_to(slice_positions[:, None], (slices, particles)),
            total / reference - 1.0,
        ),
        axis=-1,
    ).reshape(slices * particles, 6)
    bunch = AcceleratorBunch(
        coordinates,
        jnp.repeat(electrons / particles, particles),
        jnp.arange(slices * particles, dtype=jnp.int32),
        reference_rest_energy=rest_energy,
        reference_momentum=reference * rest_energy,
        reference_charge=-1.0,
        bunch_id="fel-time-dependent-particles",
    )
    kick = plan.evaluate(
        bunch,
        step_length,
        frame_lorentz_factor=reference_lorentz_factor / stretch,
    )
    increment = (kick.momentum_kick * (reference * rest_energy)).reshape(
        slices, particles, 3
    )
    longitudinal = jnp.sqrt(total * total - jnp.sum(transverse * transverse, axis=-1))
    velocity = (
        jnp.concatenate((transverse, longitudinal[..., None]), axis=-1) / gamma[..., None]
    )
    change = jnp.sum(velocity * increment, axis=-1) / rest_energy
    return (
        change,
        increment[..., :2] / rest_energy,
        kick.accepted,
        kick.cells_per_sigma,
    )


__all__ = [
    "FELSpaceCharge",
    "FELTransverseSpaceCharge",
    "bunch_kick",
    "radial_harmonic_rates",
    "uniform_harmonic_rates",
]
