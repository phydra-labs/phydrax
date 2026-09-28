#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Laser modulators and dispersive chicanes ahead of an FEL radiator.

High-gain harmonic generation (HGHG) and echo-enabled harmonic generation
(EEHG) prepare the beam at a base wavenumber ``k_b`` (the seed-laser
wavenumber) and radiate at ``h k_b``. The time-dependent solver loads every
slice at the base wavelength ``λ_b = h λ`` with quiet-start phases ``ψ``, runs
the ordered stages, and hands the radiator the phases ``θ = h ψ``.

A :class:`FELModulator` is the thin period-averaged energy kick of a resonant
modulator undulator driven by a laser at ``p k_b``:
``γ ← γ + Δγ sin(p ψ + φ)``. A chicane (or any linear element) is an existing
:class:`~phydrax.applications.accelerator.SymplecticMapPlan` acting on
``(x, p_x/p₀, y, p_y/p₀, ζ, δ)`` with ``p₀ = mₑc √(γ₀² − 1)``,
``p_x/p₀ = u_x/√(γ₀² − 1)``, ``δ = √(γ² − 1)/√(γ₀² − 1) − 1``, and the
longitudinal coordinate ``ζ = σ (ζ_s − ψ/k_b)`` of a particle in the slice
centered at ``ζ_s`` (``σ = +1`` for a ``"positive-late"`` map convention and
``−1`` for ``"positive-early"``). After the map the particle's phase is
re-read from its new position, so a chicane with ``R₅₆`` applies
``ψ ← ψ − σ k_b R₅₆ δ``. Particles stay in their slice: the slice is the
periodic representative of the locally uniform beam, and the largest
longitudinal displacement is reported as evidence.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar, positive_integer
from .._advanced import SymplecticMapPlan
from .._beam import _late_sign


class FELModulator(StrictModule, NonTrainableState):
    """Sinusoidal laser energy modulation ``Δγ sin(p ψ + φ)``.

    ``energy_amplitude`` is ``Δγ = ΔE/(mₑc²)``; ``harmonic`` ``p`` is the laser
    wavenumber in units of the base wavenumber; ``phase`` ``φ`` is in radians.
    """

    energy_amplitude: float = eqx.field(static=True)
    harmonic: int = eqx.field(static=True)
    phase: float = eqx.field(static=True)
    modulator_id: str = eqx.field(static=True)

    def __init__(
        self, energy_amplitude: float, /, *, harmonic: int = 1, phase: float = 0.0
    ) -> None:
        amplitude = finite_real_scalar(energy_amplitude, "energy_amplitude")
        if amplitude < 0.0:
            raise ValueError("energy_amplitude must be nonnegative.")
        self.energy_amplitude = amplitude
        self.harmonic = positive_integer(harmonic, "harmonic")
        self.phase = finite_real_scalar(phase, "phase")
        self.modulator_id = canonical_fingerprint(
            {
                "kind": "fel-modulator",
                "energy_amplitude": amplitude,
                "harmonic": self.harmonic,
                "phase": self.phase,
            }
        )


type FELPrebunchingStage = FELModulator | SymplecticMapPlan


class FELPrebunchedParticles(StrictModule):
    """Radiator-frame particles of one slice after the prebunching stages."""

    phases: Array
    lorentz_factors: Array
    positions: Array
    transverse_momenta: Array
    maximum_displacement: Array


class FELPrebunching(StrictModule, NonTrainableState):
    """Ordered modulators and linear maps ahead of a harmonic radiator.

    ``harmonic`` ``h`` is the radiator wavenumber in units of the base
    wavenumber; ``reference_lorentz_factor`` ``γ₀`` defines ``p₀`` and ``δ`` for
    the maps. Map lengths use the length unit of the FEL scale.
    """

    stages: tuple[FELPrebunchingStage, ...]
    harmonic: int = eqx.field(static=True)
    reference_lorentz_factor: float = eqx.field(static=True)
    prebunching_id: str = eqx.field(static=True)

    def __init__(
        self,
        stages: tuple[FELPrebunchingStage, ...],
        /,
        *,
        harmonic: int,
        reference_lorentz_factor: float,
    ) -> None:
        stages_ = tuple(stages)
        if not stages_:
            raise ValueError("Prebunching needs at least one stage.")
        identities: list[str] = []
        for stage in stages_:
            match stage:
                case FELModulator():
                    identities.append(stage.modulator_id)
                case SymplecticMapPlan():
                    # Validates the canonical momentum normalization and sign.
                    _late_sign(stage.convention)
                    identities.append(stage.plan_id)
                case _:
                    raise TypeError(
                        "Prebunching stages must be FELModulator or SymplecticMapPlan."
                    )
        gamma = finite_real_scalar(reference_lorentz_factor, "reference_lorentz_factor")
        if gamma <= 1.0:
            raise ValueError("reference_lorentz_factor must exceed one.")
        self.stages = stages_
        self.harmonic = positive_integer(harmonic, "harmonic")
        self.reference_lorentz_factor = gamma
        self.prebunching_id = canonical_fingerprint(
            {
                "kind": "fel-prebunching",
                "stages": identities,
                "harmonic": self.harmonic,
                "reference_lorentz_factor": gamma,
            }
        )

    def apply(
        self,
        phases: Array,
        lorentz_factors: Array,
        positions: Array,
        transverse_momenta: Array,
        slice_position: Array,
        base_wavenumber: float,
        /,
    ) -> FELPrebunchedParticles:
        """Run every stage on one slice and convert ``ψ`` to ``θ = h ψ``."""
        reference = math.sqrt(self.reference_lorentz_factor**2 - 1.0)
        psi = phases
        gamma = lorentz_factors
        transverse = positions
        momenta = transverse_momenta
        displacement = jnp.zeros((), dtype=jnp.float64)
        for stage in self.stages:
            match stage:
                case FELModulator():
                    gamma = gamma + stage.energy_amplitude * jnp.sin(
                        stage.harmonic * psi + stage.phase
                    )
                case SymplecticMapPlan():
                    sign = _late_sign(stage.convention)
                    late = slice_position - psi / base_wavenumber
                    delta = jnp.sqrt(gamma * gamma - 1.0) / reference - 1.0
                    coordinates = jnp.stack(
                        (
                            transverse[:, 0],
                            momenta[:, 0] / reference,
                            transverse[:, 1],
                            momenta[:, 1] / reference,
                            sign * late,
                            delta,
                        ),
                        axis=-1,
                    )
                    mapped = coordinates @ stage.matrix.T + stage.offset
                    moved = sign * mapped[:, 4]
                    displacement = jnp.maximum(
                        displacement, jnp.max(jnp.abs(moved - late))
                    )
                    psi = base_wavenumber * (slice_position - moved)
                    transverse = jnp.stack((mapped[:, 0], mapped[:, 2]), axis=-1)
                    momenta = reference * jnp.stack((mapped[:, 1], mapped[:, 3]), axis=-1)
                    momentum = reference * (1.0 + mapped[:, 5])
                    gamma = jnp.sqrt(1.0 + momentum * momentum)
                case _:
                    raise TypeError(
                        "Prebunching stages must be FELModulator or SymplecticMapPlan."
                    )
        return FELPrebunchedParticles(
            phases=self.harmonic * psi,
            lorentz_factors=gamma,
            positions=transverse,
            transverse_momenta=momenta,
            maximum_displacement=displacement,
        )


__all__ = ["FELModulator", "FELPrebunchedParticles", "FELPrebunching"]
