#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Beam slices, quiet-start loading, and Fawley shot noise.

A time-independent slice is one radiation wavelength ``λ`` of a beam with
current ``I``: it holds ``N_λ = I λ / (e c)`` electrons. The slice is
represented by ``B`` beamlets of ``M`` macroparticles each. Every beamlet
shares one energy and one transverse phase-space point drawn from the slice's
Gaussian (Twiss) distribution; its particles sit at equally spaced
ponderomotive phases ``θ_j = θ₀ + 2π j / M``, so the beamlet bunching
``M⁻¹ Σ_j e^{−ihθ_j}`` vanishes for every harmonic ``h`` that is not a multiple
of ``M`` (quiet start). ``M ≥ 2h_max`` is required for every modeled harmonic.

Fawley shot noise (Fawley, PRST-AB 5, 070701, 2002) perturbs the phases of a
beamlet representing ``N_e = N_λ / B`` electrons by
``δθ_j = Σ_{n=1}^{⌊M/2⌋} (a_n cos nθ_j + b_n sin nθ_j)`` with independent
``a_n, b_n ~ 𝒩(0, 2/(N_e n²))`` (half that variance at the Nyquist harmonic
``n = M/2``, whose aliased mode feeds ``e^{±inθ}``); to first order this gives
``⟨|b_n|²⟩ = 1/N_e``
per beamlet and ``⟨|b_n|²⟩ = 1/N_λ`` per slice, the Poisson statistics of
``N_λ`` independent electrons.

Randomness is identity-addressed: beamlet ``b`` of the slice with identity
``s`` draws from ``derive_key(key, address, 0, s, b, event)``, so loading is
invariant to slice order and batching and stores no RNG state.
"""

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._sampling import derive_key, SampleAddress
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import positive_integer
from ....typing import (
    as_host_array,
    ConvertibleToArray,
    Float64,
    HostFloat64,
    HostInt64,
    Identifier,
    parse,
    PRNGKey,
    Scope,
    Size,
    UInt32,
)
from ._types import FELSliceDim


FELShotNoise: TypeAlias = Literal["quiet", "fawley"]

_COORDINATE_ADDRESS = SampleAddress(
    "phydrax.accelerator.fel", "beamlet-loading", role="coordinates"
)
_PHASE_ADDRESS = SampleAddress(
    "phydrax.accelerator.fel", "beamlet-loading", role="reference-phase"
)
_NOISE_ADDRESS = SampleAddress("phydrax.accelerator.fel", "shot-noise", role="fawley")


class FELBeamSlices(StrictModule):
    """Longitudinal slices of an electron beam for the averaged FEL.

    ``positions`` are slice centers along the bunch in the scale's length unit
    with the ``AcceleratorConvention`` ``"positive-late"`` sense (larger values
    trail). ``currents`` are in the scale's current unit; ``lorentz_factors``
    are slice mean energies ``γ``; ``relative_energy_spreads`` are rms
    ``σ_γ/γ``. Transverse phase space per plane ``(x, y)`` is Gaussian with
    normalized emittance ``ε_n`` (length unit) and Twiss ``β`` (length unit),
    ``α``. ``identities`` address each slice's random streams.
    """

    __strict_contract__ = True

    positions: Float64[FELSliceDim]
    currents: Float64[FELSliceDim]
    lorentz_factors: Float64[FELSliceDim]
    relative_energy_spreads: Float64[FELSliceDim]
    normalized_emittances: Float64[FELSliceDim, Literal[2]]
    beta_functions: Float64[FELSliceDim, Literal[2]]
    alpha_functions: Float64[FELSliceDim, Literal[2]]
    identities: UInt32[FELSliceDim]
    slice_count: Size[FELSliceDim] = eqx.field(static=True)
    position_spacing: float | None = eqx.field(static=True)
    slices_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        positions: ConvertibleToArray,
        currents: ConvertibleToArray,
        lorentz_factors: ConvertibleToArray,
        relative_energy_spreads: ConvertibleToArray,
        normalized_emittances: ConvertibleToArray,
        beta_functions: ConvertibleToArray,
        alpha_functions: ConvertibleToArray,
        /,
        *,
        identities: ConvertibleToArray | None = None,
    ) -> None:
        scope = Scope()
        position = as_host_array(
            positions, HostFloat64[FELSliceDim], "positions", scope=scope
        )
        current = as_host_array(
            currents, HostFloat64[FELSliceDim], "currents", scope=scope
        )
        gamma = as_host_array(
            lorentz_factors, HostFloat64[FELSliceDim], "lorentz_factors", scope=scope
        )
        spread = as_host_array(
            relative_energy_spreads,
            HostFloat64[FELSliceDim],
            "relative_energy_spreads",
            scope=scope,
        )
        emittance = as_host_array(
            normalized_emittances,
            HostFloat64[FELSliceDim, Literal[2]],
            "normalized_emittances",
            scope=scope,
        )
        beta = as_host_array(
            beta_functions,
            HostFloat64[FELSliceDim, Literal[2]],
            "beta_functions",
            scope=scope,
        )
        alpha = as_host_array(
            alpha_functions,
            HostFloat64[FELSliceDim, Literal[2]],
            "alpha_functions",
            scope=scope,
        )
        count = position.shape[0]
        if identities is None:
            identity = np.arange(count, dtype=np.int64)
        else:
            identity = as_host_array(
                identities, HostInt64[FELSliceDim], "identities", scope=scope
            )
        arrays = (position, current, gamma, spread, emittance, beta, alpha)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Every slice parameter must be finite.")
        if np.any(current <= 0.0):
            raise ValueError("Slice currents must be positive.")
        if np.any(gamma <= 1.0):
            raise ValueError("Slice Lorentz factors must exceed one.")
        if np.any(spread < 0.0):
            raise ValueError("Relative energy spreads must be nonnegative.")
        if np.any(emittance <= 0.0) or np.any(beta <= 0.0):
            raise ValueError("Emittances and beta functions must be positive.")
        if np.any(identity < 0) or np.any(identity >= 2**32):
            raise ValueError("Slice identities must be unsigned 32-bit integers.")
        if np.unique(identity).shape[0] != count:
            raise ValueError("Slice identities must be unique.")
        spacing: float | None = None
        if count >= 2:
            steps = np.diff(position)
            if np.all(steps > 0.0) and np.allclose(
                steps, steps[0], rtol=1.0e-9, atol=0.0
            ):
                spacing = float(steps[0])
        self.positions = jnp.asarray(position)
        self.currents = jnp.asarray(current)
        self.lorentz_factors = jnp.asarray(gamma)
        self.relative_energy_spreads = jnp.asarray(spread)
        self.normalized_emittances = jnp.asarray(emittance)
        self.beta_functions = jnp.asarray(beta)
        self.alpha_functions = jnp.asarray(alpha)
        self.identities = jnp.asarray(identity.astype(np.uint32))
        self.slice_count = parse(count, Size[FELSliceDim], "slice_count")
        self.position_spacing = spacing
        self.slices_id = canonical_fingerprint(
            {
                "kind": "fel-beam-slices",
                "arrays": array_tree_fingerprint(arrays + (identity,)),
            }
        )


class FELLoading(StrictModule, NonTrainableState):
    """Quiet-start beamlet loading with optional Fawley shot noise."""

    beamlet_count: int = eqx.field(static=True)
    particles_per_beamlet: int = eqx.field(static=True)
    shot_noise: FELShotNoise = eqx.field(static=True)
    loading_id: str = eqx.field(static=True)

    def __init__(
        self,
        beamlet_count: int,
        particles_per_beamlet: int,
        /,
        *,
        shot_noise: FELShotNoise,
    ) -> None:
        beamlets = positive_integer(beamlet_count, "beamlet_count")
        particles = positive_integer(particles_per_beamlet, "particles_per_beamlet")
        if particles < 2:
            raise ValueError("A quiet-start beamlet needs at least two particles.")
        noise = parse(shot_noise, FELShotNoise, "shot_noise")
        self.beamlet_count = beamlets
        self.particles_per_beamlet = particles
        self.shot_noise = noise
        self.loading_id = canonical_fingerprint(
            {
                "kind": "fel-quiet-start-loading",
                "beamlet_count": beamlets,
                "particles_per_beamlet": particles,
                "shot_noise": noise,
            }
        )

    @property
    def particle_count(self) -> int:
        return self.beamlet_count * self.particles_per_beamlet


class FELParticles(StrictModule):
    """Macroparticle state per slice: ``[slice, particle]`` arrays.

    ``phases`` are ponderomotive phases ``θ``; ``transverse_momenta`` are
    ``(γx′, γy′)``. Particles of one beamlet are contiguous.
    """

    phases: Array
    lorentz_factors: Array
    positions: Array
    transverse_momenta: Array


class LoadedSlice(StrictModule):
    """One slice loaded in the solver's working layout (no leading slice axis)."""

    phases: Array
    lorentz_factors: Array
    positions: Array
    transverse_momenta: Array
    noise_amplitude: Array


def load_slice(
    loading: FELLoading,
    key: PRNGKey,
    identity: Array,
    lorentz_factor: Array,
    relative_energy_spread: Array,
    normalized_emittance: Array,
    beta_function: Array,
    alpha_function: Array,
    electrons_per_beamlet: Array,
    /,
) -> LoadedSlice:
    """Load one slice (traceable; ``identity`` is the slice's ``uint32`` word)."""
    beamlets = loading.beamlet_count
    particles = loading.particles_per_beamlet
    indices = jnp.arange(beamlets, dtype=jnp.uint32)

    def beamlet(index: Array) -> tuple[Array, Array, Array]:
        coordinates = jax.random.normal(
            derive_key(key, _COORDINATE_ADDRESS, 0, identity, index, 0),
            (5,),
            dtype=jnp.float64,
        )
        reference = jax.random.uniform(
            derive_key(key, _PHASE_ADDRESS, 0, identity, index, 1),
            (),
            dtype=jnp.float64,
            maxval=2.0 * jnp.pi,
        )
        noise = jax.random.normal(
            derive_key(key, _NOISE_ADDRESS, 0, identity, index, 2),
            (2, particles // 2),
            dtype=jnp.float64,
        )
        return coordinates, reference, noise

    coordinates, reference, noise = jax.vmap(beamlet)(indices)
    gamma = lorentz_factor * (1.0 + relative_energy_spread * coordinates[:, 0])
    geometric = normalized_emittance / lorentz_factor
    size = jnp.sqrt(geometric * beta_function)
    divergence = jnp.sqrt(geometric / beta_function)
    normal_position = coordinates[:, 1:3]
    normal_angle = coordinates[:, 3:5]
    position = size[None, :] * normal_position
    angle = divergence[None, :] * (
        normal_angle - alpha_function[None, :] * normal_position
    )
    momentum = gamma[:, None] * angle

    offsets = 2.0 * jnp.pi * jnp.arange(particles, dtype=jnp.float64) / particles
    phases = reference[:, None] + offsets[None, :]
    orders = jnp.arange(1, particles // 2 + 1, dtype=jnp.float64)
    # σ_n = √(2/N_e)/n gives ⟨|b_n|²⟩ = 1/N_e per beamlet to first order.
    amplitude = jnp.sqrt(2.0 / electrons_per_beamlet)
    match loading.shot_noise:
        case "quiet":
            perturbed = phases
        case "fawley":
            # At the Nyquist harmonic n = M/2 the cosine and sine modes alias onto
            # one real mode that feeds both e^{±inθ}; halving its variance keeps
            # ⟨|b_n|²⟩ = 1/N_e.
            nyquist = jnp.where(2.0 * orders == particles, jnp.sqrt(0.5), 1.0)
            scaled = amplitude * nyquist * noise / orders[None, None, :]
            angles = orders[None, None, :] * phases[:, :, None]
            perturbation = jnp.sum(
                scaled[:, 0, None, :] * jnp.cos(angles)
                + scaled[:, 1, None, :] * jnp.sin(angles),
                axis=-1,
            )
            perturbed = phases + perturbation
        case _:
            assert_never(loading.shot_noise)

    def expand(values: Array) -> Array:
        return jnp.repeat(values, particles, axis=0)

    return LoadedSlice(
        phases=perturbed.reshape((-1,)),
        lorentz_factors=expand(gamma),
        positions=expand(position),
        transverse_momenta=expand(momentum),
        noise_amplitude=amplitude,
    )


__all__ = [
    "FELBeamSlices",
    "FELLoading",
    "FELParticles",
    "FELShotNoise",
    "LoadedSlice",
    "load_slice",
]
