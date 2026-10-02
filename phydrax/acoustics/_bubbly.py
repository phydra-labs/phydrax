#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Linear acoustics of a dilute bubbly liquid (Commander & Prosperetti 1989).

The mixture pressure obeys the van Wijngaarden–Papanicolaou equations
`(1/(ρc²)) ∂P/∂t + ∇·u = ∂β/∂t`, `ρ ∂u/∂t + ∇P = 0`, with the gas volume
fraction `β = Σ_b (4/3)π R_b³ n_b`. For a plane wave `exp(iωt − ikx)` and the
bin radius response `R̂_b(ω)` per unit excess far-field pressure, this gives

    k² = ω²/c² − 4πρω² Σ_b n_b R_b² R̂_b(ω),

which reduces to Commander & Prosperetti's eq. (41),
`k² = ω²/c² + 4πω² Σ_b n_b R_b/(ω0_b² − ω² + 2i b_b ω)`, for the equivalent
oscillator `R̂ = −1/(ρR(ω0² − ω² + 2ibω))`. `R̂_b` comes from
`bubble_dynamics.linear_bubble_response` of one composed radial model, so
thermal, viscous, shell and radiation damping all come from the composed laws.
"""

from __future__ import annotations

from enum import IntEnum
from math import e, log10

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field
from ..bubble_dynamics import linear_bubble_response, RadialBubbleModel
from ..qualification import CapabilityProfile, SupportTuple
from ..typing import checked


_DECIBEL_PER_NEPER = 20.0 * log10(e)


class BubblyMediumDispersionStatus(IntEnum):
    """Mutually exclusive outcome of one bubbly-medium dispersion evaluation.

    Precedence is the member order: a failed bin response outranks a
    non-finite wavenumber, which outranks a void fraction above the declared
    dilute bound. Only `SUCCESS` is inside the model's declared support.
    """

    SUCCESS = 0
    RESPONSE_FAILURE = 1
    NONFINITE = 2
    OUTSIDE_DILUTE_LIMIT = 3


def _host_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    host = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if host.ndim != 1 or host.shape[0] == 0:
        raise ValueError(f"{name} must be a scalar or a non-empty rank-1 array.")
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    return host


def _positive_host(value: ArrayLike, name: str, /) -> np.ndarray:
    host = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(host)) or not np.all(host > 0.0):
        raise ValueError(f"{name} must be positive and finite.")
    return host


class BubblyMediumDispersionPlan(StrictModule):
    """Composed bubble model, discrete size bins and angular frequencies.

    `bin_radii` (m) are equilibrium radii and `number_densities` (m⁻³) the
    number of bubbles of each bin per unit mixture volume; a continuous size
    distribution enters as quadrature nodes and weighted densities. The liquid
    density and sound speed are the far-field values of `model`. Bins are
    evaluated with one static model structure and a dynamic equilibrium radius,
    `bin_batch_size` bins at a time (working set: `bin_batch_size` × frequency
    count dense harmonic solves of the model state dimension).
    `maximum_void_fraction` is the declared dilute bound of the
    `O(β)`-accurate model.
    """

    model: RadialBubbleModel
    bin_radii: Array = parameter_field()
    number_densities: Array = parameter_field()
    angular_frequencies: Array = fixed_field()
    maximum_void_fraction: float = eqx.field(static=True)
    bin_batch_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: RadialBubbleModel,
        bin_radii: ArrayLike,
        number_densities: ArrayLike,
        angular_frequencies: ArrayLike,
        /,
        *,
        maximum_void_fraction: float = 1.0e-2,
        bin_batch_size: int = 32,
    ) -> None:
        _positive_host(model.far_field_density(), "liquid density")
        _positive_host(model.far_field_sound_speed(), "liquid sound speed")
        radii = _host_vector(bin_radii, "bin_radii")
        if not np.all(radii > 0.0):
            raise ValueError("bin_radii must be positive.")
        densities = _host_vector(number_densities, "number_densities")
        if densities.shape != radii.shape:
            raise ValueError("number_densities must have one entry per bin radius.")
        if not np.all(densities >= 0.0) or not np.any(densities > 0.0):
            raise ValueError(
                "number_densities must be nonnegative with a positive entry."
            )
        frequencies = _host_vector(angular_frequencies, "angular_frequencies")
        if not np.all(frequencies > 0.0):
            raise ValueError("angular_frequencies must be positive.")
        bound = float(maximum_void_fraction)
        if not (0.0 < bound < 1.0):
            raise ValueError("maximum_void_fraction must lie in (0, 1).")
        batch = int(bin_batch_size)
        if batch <= 0:
            raise ValueError("bin_batch_size must be positive.")
        self.model = model
        self.bin_radii = jnp.asarray(radii, dtype=jnp.float64)
        self.number_densities = jnp.asarray(densities, dtype=jnp.float64)
        self.angular_frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
        self.maximum_void_fraction = bound
        self.bin_batch_size = batch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubbly-medium-dispersion-plan",
                "model": model.model_id,
                "bin_count": radii.shape[0],
                "angular_frequencies": frequencies,
                "maximum_void_fraction": bound,
                "bin_batch_size": batch,
            }
        )

    def parameter_id(self) -> str:
        """Host fingerprint of the bins and every dynamic model coefficient.

        This is an explicit host boundary; call it outside traced code.
        """
        leaves = jax.tree.leaves(
            eqx.filter((self.model, self.bin_radii, self.number_densities), eqx.is_array)
        )
        return canonical_fingerprint(
            {
                "kind": "bubbly-medium-dispersion-parameters",
                "plan": self.plan_id,
                "leaves": [np.asarray(leaf) for leaf in leaves],
            }
        )


class BubblyMediumDispersionEvidence(StrictModule):
    """Bin-response, support and effective-medium evidence.

    `spacing_wavelength_ratio` is the mean inter-bubble spacing `(Σn)^(-1/3)`
    over the mixture wavelength `2π/Re k`; `maximum_size_parameter` is the
    largest `|k| R_b`. Both must be small for the continuum model to apply.
    """

    response_successful: Array
    maximum_relative_residual: Array
    dilute: Array
    finite: Array
    spacing_wavelength_ratio: Array
    maximum_size_parameter: Array
    maximum_void_fraction: float = eqx.field(static=True)


class BubblyMediumDispersionResult(StrictModule):
    """Complex mixture wavenumber and derived plane-wave properties.

    The branch is the decaying one of `exp(iωt − ikx)`: `Im k ≤ 0`. Phase
    speed is `ω/Re k`, attenuation `−Im k` in Np m⁻¹ and `20 log₁₀(e)` times
    that in dB m⁻¹. `radius_response` is `R̂_b(ω)` (m Pa⁻¹) per bin and
    frequency; `resonance_frequency` is each bin's self-consistent angular
    resonance `ω = ω0(ω)`.
    """

    angular_frequency: Array
    wavenumber: Array
    phase_speed: Array
    attenuation_neper_per_meter: Array
    attenuation_decibel_per_meter: Array
    void_fraction: Array
    resonance_frequency: Array
    radius_response: Array
    status: Array
    evidence: BubblyMediumDispersionEvidence
    plan_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether every bin response succeeded inside the dilute support."""
        return self.status == int(BubblyMediumDispersionStatus.SUCCESS)

    def realization_id(self) -> str:
        """Host fingerprint of the computed realization (explicit host boundary)."""
        return canonical_fingerprint(
            {
                "kind": "bubbly-medium-dispersion-realization",
                "plan": self.plan_id,
                "model": self.model_id,
                "status": int(self.status),
                "wavenumber": np.asarray(self.wavenumber),
            }
        )


def _bin_responses(
    plan: BubblyMediumDispersionPlan, /
) -> tuple[Array, Array, Array, Array]:
    """Radius response, resonance, success and residual of every bin, batched."""
    frequencies = plan.angular_frequencies

    def respond(radius: Array) -> tuple[Array, Array, Array, Array]:
        response = linear_bubble_response(plan.model, radius, frequencies)
        return (
            response.radius_response,
            response.resonance_frequency,
            response.successful,
            jnp.max(response.evidence.relative_residual),
        )

    batch = min(plan.bin_batch_size, plan.bin_radii.shape[0])
    return jax.lax.map(respond, plan.bin_radii, batch_size=batch)


def _status(responses_ok: Array, finite: Array, dilute: Array, /) -> Array:
    return jnp.where(
        ~responses_ok,
        int(BubblyMediumDispersionStatus.RESPONSE_FAILURE),
        jnp.where(
            ~finite,
            int(BubblyMediumDispersionStatus.NONFINITE),
            jnp.where(
                ~dilute,
                int(BubblyMediumDispersionStatus.OUTSIDE_DILUTE_LIMIT),
                int(BubblyMediumDispersionStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)


def solve_bubbly_medium_dispersion(
    plan: BubblyMediumDispersionPlan, /
) -> BubblyMediumDispersionResult:
    """Commander–Prosperetti dispersion of `plan` from the composed bin responses."""
    if not isinstance(plan, BubblyMediumDispersionPlan):
        raise TypeError("plan must be a BubblyMediumDispersionPlan.")
    model = plan.model
    density = model.far_field_density()
    sound_speed = model.far_field_sound_speed()
    radii = plan.bin_radii
    numbers = plan.number_densities
    omega = plan.angular_frequencies
    response, resonance, response_ok, residual = _bin_responses(plan)
    squared = omega**2 / sound_speed**2 - 4.0 * jnp.pi * density * omega**2 * (
        (numbers * radii**2) @ response
    )
    principal = jnp.sqrt(squared)
    wavenumber = jnp.where(jnp.imag(principal) > 0.0, -principal, principal)
    attenuation = -jnp.imag(wavenumber)
    void_fraction = jnp.sum(numbers * 4.0 * jnp.pi * radii**3 / 3.0)
    spacing = jnp.sum(numbers) ** (-1.0 / 3.0)
    responses_ok = jnp.all(response_ok)
    finite = jnp.all(jnp.isfinite(wavenumber))
    dilute = void_fraction <= plan.maximum_void_fraction
    evidence = BubblyMediumDispersionEvidence(
        response_ok,
        jnp.max(residual),
        dilute,
        finite,
        spacing * jnp.real(wavenumber) / (2.0 * jnp.pi),
        jnp.max(jnp.abs(wavenumber)) * jnp.max(radii),
        maximum_void_fraction=plan.maximum_void_fraction,
    )
    return BubblyMediumDispersionResult(
        omega,
        wavenumber,
        omega / jnp.real(wavenumber),
        attenuation,
        _DECIBEL_PER_NEPER * attenuation,
        void_fraction,
        resonance,
        response,
        _status(responses_ok, finite, dilute),
        evidence,
        plan_id=plan.plan_id,
        model_id=model.model_id,
    )


def wood_sound_speed(
    void_fraction: ArrayLike,
    liquid_density: ArrayLike,
    liquid_sound_speed: ArrayLike,
    gas_density: ArrayLike,
    gas_sound_speed: ArrayLike,
    /,
) -> Array:
    """Wood's (1930) equilibrium sound speed of a two-phase mixture.

    `1/(ρ_m c_m²) = β/(ρ_g c_g²) + (1 − β)/(ρ_l c_l²)` with
    `ρ_m = βρ_g + (1 − β)ρ_l`. Inputs are validated on the host (void fraction
    in `[0, 1]`, positive densities and sound speeds) and broadcast together.
    """
    fraction = np.asarray(void_fraction, dtype=np.float64)
    if not np.all(np.isfinite(fraction)) or not np.all(
        (fraction >= 0.0) & (fraction <= 1.0)
    ):
        raise ValueError("void_fraction must lie in [0, 1].")
    rho_l = _positive_host(liquid_density, "liquid_density")
    c_l = _positive_host(liquid_sound_speed, "liquid_sound_speed")
    rho_g = _positive_host(gas_density, "gas_density")
    c_g = _positive_host(gas_sound_speed, "gas_sound_speed")
    beta = jnp.asarray(fraction, dtype=jnp.float64)
    liquid_stiffness = jnp.asarray(rho_l * c_l**2, dtype=jnp.float64)
    gas_stiffness = jnp.asarray(rho_g * c_g**2, dtype=jnp.float64)
    compliance = beta / gas_stiffness + (1.0 - beta) / liquid_stiffness
    density = beta * jnp.asarray(rho_g, dtype=jnp.float64) + (1.0 - beta) * jnp.asarray(
        rho_l, dtype=jnp.float64
    )
    return jnp.sqrt(1.0 / (density * compliance))


def bubbly_medium_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the unreleased bubbly-medium acoustics candidate."""
    return (
        CapabilityProfile(
            "acoustics.bubbly-medium.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "acoustics.bubbly-medium",
                    {
                        "formulation": "commander-prosperetti-linear",
                        "bubble_response": "composed-linear-bubble-response",
                        "size_distribution": "discrete-bins",
                        "regime": "dilute",
                    },
                ),
            ),
            required_gates=(
                "wood-low-frequency-limit",
                "dilute-analytic-limit",
                "resonance-attenuation",
                "commander-prosperetti-reference",
                "public-workflow",
            ),
        ),
    )


__all__ = [
    "BubblyMediumDispersionEvidence",
    "BubblyMediumDispersionPlan",
    "BubblyMediumDispersionResult",
    "BubblyMediumDispersionStatus",
    "bubbly_medium_candidate_profiles",
    "solve_bubbly_medium_dispersion",
    "wood_sound_speed",
]
