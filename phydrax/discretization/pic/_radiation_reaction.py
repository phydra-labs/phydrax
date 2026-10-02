#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Classical and quantum-corrected radiation reaction on charged particles.

`RadiationReactionPlan` applies one explicit radiation-reaction step to proper
velocities ``u = γv`` of one physical species (charge ``q``, mass ``m``) in the
units of one `ElectromagneticScaleContract`. With ``τ = q²/(6π ε₀ m c³)``,
``L = E + v×B`` and ``Q² = L² − (v·E)²/c²`` the models are

- ``"landau-lifshitz-reduced"`` (Tamburini et al. 2010): the Landau–Lifshitz
  force without its field-derivative term,
  ``F = τ (q²/m) [L×B + (v·E) E/c²] − τ (q²/m) (γ²/c²) Q² v``;
- ``"landau-lifshitz"`` (Landau & Lifshitz §76): adds
  ``τ q γ [D E + v × D B]`` with the convective derivative
  ``D = ∂_t + v·∇``, which needs field gradients and time derivatives;
- ``"quantum-corrected-landau-lifshitz"``: ``g(χ)`` times the reduced force,
  ``g`` the ratio of the quantum to the classical synchrotron power;
- ``"stochastic-fokker-planck"`` (Niel et al., PRE 97, 043209, 2018): the
  quantum-corrected drift plus an Euler–Maruyama diffusion of ``γ`` along
  ``u`` with coefficient ``B = γ (P_cl/mc²) (55/(16√3)) χ h(χ)``, driven by
  standard normal Wiener increments addressed by persistent particle identity.

``χ = γ √Q² / E_S`` with the species critical field ``E_S = m²c³/(|q|ħ)``;
``P_cl = (2/3) α_q m²c⁴ χ²/ħ`` with ``α_q = q²/(4π ε₀ ħ c)``; the critical
angular frequency is ``ω_c = 3 χ γ m c²/(2ħ)``. One step is the first-order
operator split ``u ← u + Δt F/m`` (plus the diffusion increment), so the
radiated energy is exactly the kinetic energy removed,
``W = m c² (γ_before − γ_after)``.

``g`` and ``h`` come from `RadiationReactionTables`: host quadrature of the
Baier–Katkov photon spectrum written with the synchrotron kernels
``F(x) = x∫ₓ^∞K_{5/3}`` and ``G(x) = xK_{2/3}(x)`` of `phydrax.special`,
tabulated in ``log χ`` with exact slopes and cubic Hermite interpolation; both
tend to one as ``χ → 0``, where the quantum models reduce to the classical one.

`RadiationReactionProcess` binds a plan to one PIC species as a momentum-stage
process that claims ``"subgrid-reaction"`` radiation ownership; its ledger
reports the radiated energy and the scale separation between each emitting
particle's critical frequency and the field grid's cutoff frequency.
"""

from __future__ import annotations

import math
from enum import IntFlag
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import cubic_hermite_interpolate
from ..._numerics import gauss_legendre_data
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...ein import contract
from ...special import synchrotron_f, synchrotron_g
from ...typing import checked, Dim, Float64, parse, PRNGKey, Size
from ._charge_state import PICSpeciesPlan, PICSpeciesState
from ._process import (
    AbstractPICProcess,
    PICProcessContext,
    PICProcessLedger,
    PICProcessRadiation,
    PICProcessResult,
    PICProcessStage,
    RadiationOwnership,
)
from ._types import PICParticleState


RadiationReactionModel: TypeAlias = Literal[
    "landau-lifshitz-reduced",
    "landau-lifshitz",
    "quantum-corrected-landau-lifshitz",
    "stochastic-fokker-planck",
]


class RadiationReactionFlag(IntFlag):
    """Per-particle radiation-reaction support flags.

    ``BELOW_MINIMUM_GAMMA`` marks a declared exemption (no reaction applied),
    ``SCALE_UNSEPARATED`` a radiation-ownership conflict with the field grid;
    the remaining flags are support failures.
    """

    NONE = 0
    BELOW_MINIMUM_GAMMA = 1
    CHI_EXCEEDED = 2
    STEP_LOSS_EXCEEDED = 4
    SCALE_UNSEPARATED = 8
    NONFINITE = 16


_SUPPORT_FAILURE = (
    RadiationReactionFlag.CHI_EXCEEDED
    | RadiationReactionFlag.STEP_LOSS_EXCEEDED
    | RadiationReactionFlag.NONFINITE
)
# Normalizations fixed by the classical limits g(0) = h(0) = 1: ∫F = 8π/(9√3)
# and ∫xF / ∫F = 55/(24√3) (Sands).
_POWER_PREFACTOR = 9.0 * math.sqrt(3.0) / (2.0 * math.pi)
_DIFFUSION_PREFACTOR = 648.0 / (55.0 * math.pi)
_DIFFUSION_SCALE = 55.0 / (16.0 * math.sqrt(3.0))
# Below this χ the tables interpolate linearly to g(0) = h(0) = 1; the
# neglected curvature is O(50 χ²) ≤ 5e-11.
_MINIMUM_TABLE_CHI = 1.0e-6
# F, G ≤ √(πν/2) e^{-ν}(1 + O(1/ν)): the omitted tail beyond ν = 80 is < 1e-33.
_MAXIMUM_NU = 80.0
_PANEL_ORDER = 16
_INTERPOLATION_TOLERANCE = 1.0e-6


class _ChiNodeDim(Dim, minimum=4):
    """Tabulation nodes in ``log χ``."""


def _spectral_moments(chi: Array, panels: int, /) -> tuple[Array, Array]:
    """``g(χ)`` and ``h(χ)`` by composite Gauss–Legendre in ``t = ν^{1/3}``.

    With ``a = 3χν`` and ``d = 2 + a``:
    ``g = (9√3/2π) ∫ [2F/d³ + a²G/d⁴] dν`` and
    ``h = (648/55π) ∫ ν [2F/d⁴ + a²G/d⁵] dν``; in ``t`` the kernels are
    polynomial-smooth at the origin (``F, G ~ ν^{1/3}``).
    """
    rule = gauss_legendre_data(_PANEL_ORDER)
    edges = jnp.linspace(0.0, _MAXIMUM_NU ** (1.0 / 3.0), panels + 1)
    widths = jnp.diff(edges)
    points = edges[:-1, None] + 0.5 * widths[:, None] * (rule.nodes[None, :] + 1.0)
    weights = 0.5 * widths[:, None] * rule.weights[None, :]
    t = points.reshape(-1)
    nu = t**3
    jacobian = 3.0 * t**2 * weights.reshape(-1)
    f = synchrotron_f(nu)
    g = synchrotron_g(nu)
    a = 3.0 * chi[:, None] * nu[None, :]
    d = 2.0 + a
    power = (2.0 * f / d**3 + a**2 * g / d**4) @ jacobian
    diffusion = (nu * (2.0 * f / d**4 + a**2 * g / d**5)) @ jacobian
    return _POWER_PREFACTOR * power, _DIFFUSION_PREFACTOR * diffusion


class RadiationReactionTables(StrictModule, NonTrainableState):
    """Quantum synchrotron corrections ``g(χ)`` and ``h(χ)`` for ``χ ≤ maximum_chi``.

    ``g = P_quantum/P_classical`` is the power correction (Niel et al. 2018,
    eq. for ``g``); ``h`` is the second photon-energy moment normalized to its
    classical limit, related to Niel's ``h_N`` by ``h_N = (55/(16√3)) χ³ h``.
    Values and exact ``d/d log χ`` slopes (forward-mode through the quadrature)
    are tabulated on ``nodes_per_decade`` log-spaced nodes per decade over
    ``[1e-6, maximum_chi]`` and interpolated by cubic Hermite segments; below
    ``1e-6`` both interpolate linearly to one at ``χ = 0``.
    ``quadrature_error`` is the largest relative change from halving the
    quadrature panels and ``interpolation_error`` the largest relative
    interpolation error at the segment midpoints; construction refuses tables
    whose interpolation error exceeds ``1e-6``.
    """

    __strict_contract__ = True

    log_chi: Float64[_ChiNodeDim]
    power_values: Float64[_ChiNodeDim]
    power_slopes: Float64[_ChiNodeDim]
    diffusion_values: Float64[_ChiNodeDim]
    diffusion_slopes: Float64[_ChiNodeDim]
    maximum_chi: float = eqx.field(static=True)
    quadrature_error: float = eqx.field(static=True)
    interpolation_error: float = eqx.field(static=True)
    node_count: Size[_ChiNodeDim] = eqx.field(static=True)
    tables_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_chi: float,
        nodes_per_decade: int = 48,
        quadrature_panels: int = 64,
    ) -> None:
        maximum = positive_finite_float(maximum_chi, "maximum_chi")
        if maximum <= 10.0 * _MINIMUM_TABLE_CHI:
            raise ValueError("maximum_chi must exceed 1e-5.")
        density = positive_integer(nodes_per_decade, "nodes_per_decade")
        panels = positive_integer(quadrature_panels, "quadrature_panels")
        if density < 4 or panels < 8:
            raise ValueError(
                "nodes_per_decade ≥ 4 and quadrature_panels ≥ 8 are required."
            )
        count = math.ceil(density * math.log10(maximum / _MINIMUM_TABLE_CHI)) + 1
        log_chi = jnp.linspace(math.log(_MINIMUM_TABLE_CHI), math.log(maximum), count)

        def moments(value: Array) -> tuple[Array, Array]:
            return _spectral_moments(jnp.exp(value), panels)

        (power, diffusion), (power_slope, diffusion_slope) = jax.jvp(
            moments, (log_chi,), (jnp.ones_like(log_chi),)
        )
        coarse_power, coarse_diffusion = _spectral_moments(jnp.exp(log_chi), panels // 2)
        quadrature = max(
            float(jnp.max(jnp.abs(coarse_power / power - 1.0))),
            float(jnp.max(jnp.abs(coarse_diffusion / diffusion - 1.0))),
        )
        midpoints = 0.5 * (log_chi[:-1] + log_chi[1:])
        direct = _spectral_moments(jnp.exp(midpoints), panels)
        interpolation = max(
            float(
                jnp.max(
                    jnp.abs(
                        cubic_hermite_interpolate(
                            log_chi, values, midpoints, slopes=slopes
                        ).values
                        / reference
                        - 1.0
                    )
                )
            )
            for values, slopes, reference in (
                (power, power_slope, direct[0]),
                (diffusion, diffusion_slope, direct[1]),
            )
        )
        if not interpolation <= _INTERPOLATION_TOLERANCE:
            raise ValueError(
                f"Radiation-reaction table interpolation error {interpolation:.2e} "
                f"exceeds {_INTERPOLATION_TOLERANCE:.0e}; increase nodes_per_decade."
            )
        self.log_chi = log_chi
        self.power_values = power
        self.power_slopes = power_slope
        self.diffusion_values = diffusion
        self.diffusion_slopes = diffusion_slope
        self.maximum_chi = maximum
        self.quadrature_error = quadrature
        self.interpolation_error = interpolation
        self.node_count = count
        self.tables_id = canonical_fingerprint(
            {
                "kind": "radiation-reaction-tables",
                "maximum_chi": maximum,
                "nodes_per_decade": density,
                "quadrature_panels": panels,
                "minimum_chi": _MINIMUM_TABLE_CHI,
            }
        )

    def _evaluate(self, values: Array, slopes: Array, chi: Array, /) -> Array:
        clipped = jnp.clip(chi, _MINIMUM_TABLE_CHI, self.maximum_chi)
        tabulated = cubic_hermite_interpolate(
            self.log_chi, values, jnp.log(clipped), slopes=slopes
        ).values
        linear = 1.0 + (values[0] - 1.0) * chi / _MINIMUM_TABLE_CHI
        return jnp.where(chi < _MINIMUM_TABLE_CHI, linear, tabulated)

    def power_correction(self, chi: ArrayLike, /) -> Array:
        """``g(χ) = P_quantum/P_classical``; ``χ`` above the table is clipped."""
        return self._evaluate(
            self.power_values, self.power_slopes, jnp.asarray(chi, dtype=jnp.float64)
        )

    def diffusion_correction(self, chi: ArrayLike, /) -> Array:
        """``h(χ)``, the photon-energy second moment over its classical limit."""
        return self._evaluate(
            self.diffusion_values,
            self.diffusion_slopes,
            jnp.asarray(chi, dtype=jnp.float64),
        )


class RadiationReactionResult(StrictModule):
    """One radiation-reaction step of ``N`` particles.

    ``radiated_energy[N]`` is per physical particle, ``m c² (γ_before −
    γ_after)``, zero where the reaction is not applied (inactive or exempt).
    ``drift_rate`` is the deterministic ``dγ/dt`` (the Fokker–Planck ``A``) and
    ``diffusion`` the Fokker–Planck ``B`` (zero for deterministic models).
    ``scale_separation`` is ``ω_c`` over the grid cutoff frequency (``inf``
    without a grid). ``supported`` excludes support failures on active
    particles; ``scale_separated`` excludes ownership conflicts of emitting
    particles; ``successful`` is ``all(supported)``. Proper velocities are the
    candidate update; consumers commit them only when successful.
    """

    proper_velocity: Array
    radiated_energy: Array
    quantum_parameter: Array
    critical_frequency: Array
    scale_separation: Array
    drift_rate: Array
    diffusion: Array
    flags: Array
    applied: Array
    emitting: Array
    supported: Array
    scale_separated: Array
    successful: Array


def _cross(left: Array, right: Array, /) -> Array:
    return jnp.cross(left, right, axis=-1)


class RadiationReactionPlan(StrictModule, NonTrainableState):
    """Radiation reaction of one physical species in one electromagnetic scale.

    ``physical_charge``/``physical_mass`` are the single-particle charge and
    mass in ``scale`` units. ``maximum_chi`` bounds the supported quantum
    parameter (classical models are valid for ``χ ≪ 1``); quantum models need
    ``tables`` covering it. Particles with ``γ < minimum_gamma`` are exempt
    (flagged, no reaction). A step fails where the per-step kinetic-energy
    change exceeds ``maximum_relative_step_loss`` of the kinetic energy (for
    the Fokker–Planck model ``|A|Δt + √(BΔt)``) or would leave ``γ < 1``.
    ``minimum_scale_separation`` is the smallest admissible ratio of an
    emitting particle's critical frequency to the field-grid cutoff.
    """

    tables: RadiationReactionTables | None
    model: RadiationReactionModel = eqx.field(static=True)
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    physical_charge: float = eqx.field(static=True)
    physical_mass: float = eqx.field(static=True)
    maximum_chi: float = eqx.field(static=True)
    minimum_gamma: float = eqx.field(static=True)
    maximum_relative_step_loss: float = eqx.field(static=True)
    minimum_scale_separation: float = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    time_constant: float = eqx.field(static=True)
    critical_field: float = eqx.field(static=True)
    critical_frequency_scale: float = eqx.field(static=True)
    classical_power_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: RadiationReactionModel,
        scale: ElectromagneticScaleContract,
        physical_charge: float,
        physical_mass: float,
        /,
        *,
        tables: RadiationReactionTables | None = None,
        maximum_chi: float,
        minimum_gamma: float,
        maximum_relative_step_loss: float = 0.1,
        minimum_scale_separation: float = 10.0,
    ) -> None:
        model_ = parse(model, RadiationReactionModel, "model")
        charge = float(physical_charge)
        if not math.isfinite(charge) or charge == 0.0:
            raise ValueError("physical_charge must be finite and nonzero.")
        mass = positive_finite_float(physical_mass, "physical_mass")
        chi = positive_finite_float(maximum_chi, "maximum_chi")
        gamma = float(minimum_gamma)
        if not math.isfinite(gamma) or gamma < 1.0:
            raise ValueError("minimum_gamma must be finite and at least one.")
        loss = positive_finite_float(
            maximum_relative_step_loss, "maximum_relative_step_loss"
        )
        if loss >= 1.0:
            raise ValueError("maximum_relative_step_loss must be below one.")
        separation = positive_finite_float(
            minimum_scale_separation, "minimum_scale_separation"
        )
        match model_:
            case "landau-lifshitz-reduced" | "landau-lifshitz":
                if tables is not None:
                    raise ValueError(f"The classical {model_!r} model takes no tables.")
            case "quantum-corrected-landau-lifshitz" | "stochastic-fokker-planck":
                if not isinstance(tables, RadiationReactionTables):
                    raise TypeError(f"The {model_!r} model requires tables.")
                if tables.maximum_chi < chi:
                    raise ValueError("tables must cover maximum_chi.")
            case _:
                assert_never(model_)
        light = float(scale.speed_of_light)
        hbar = float(scale.reduced_planck_constant)
        permittivity = float(scale.vacuum_permittivity)
        self.tables = tables
        self.model = model_
        self.scale = scale
        self.physical_charge = charge
        self.physical_mass = mass
        self.maximum_chi = chi
        self.minimum_gamma = gamma
        self.maximum_relative_step_loss = loss
        self.minimum_scale_separation = separation
        self.speed_of_light = light
        self.time_constant = charge**2 / (6.0 * math.pi * permittivity * mass * light**3)
        self.critical_field = mass**2 * light**3 / (abs(charge) * hbar)
        self.critical_frequency_scale = 1.5 * mass * light**2 / hbar
        self.classical_power_scale = (
            (2.0 / 3.0)
            * charge**2
            / (4.0 * math.pi * permittivity * hbar * light)
            * mass**2
            * light**4
            / hbar
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "radiation-reaction-plan",
                "model": model_,
                "scale": scale.scale_id,
                "physical_charge": charge,
                "physical_mass": mass,
                "tables": None if tables is None else tables.tables_id,
                "maximum_chi": chi,
                "minimum_gamma": gamma,
                "maximum_relative_step_loss": loss,
                "minimum_scale_separation": separation,
            }
        )

    @property
    def requires_field_derivatives(self) -> bool:
        return self.model == "landau-lifshitz"

    @property
    def stochastic(self) -> bool:
        return self.model == "stochastic-fokker-planck"

    @checked
    def validate_relativity(self, relativity: RelativityScaleContract, /) -> None:
        """Refuse a pusher whose units or speed of light differ from ``scale``."""
        own = self.scale.relativity
        if relativity.dimensional_scale.scale_id != own.dimensional_scale.scale_id:
            raise ValueError(
                "The pusher and the radiation-reaction scale use different length, "
                "mass, or time units."
            )
        if relativity.speed_of_light != own.speed_of_light:
            raise ValueError(
                "The pusher speed of light differs from the radiation-reaction scale."
            )

    def matches_species(self, charge: ArrayLike, mass: ArrayLike, /) -> Array:
        """Whether particles have this plan's charge and mass (relative 1e-12)."""
        charge_ = jnp.asarray(charge, dtype=jnp.float64)
        mass_ = jnp.asarray(mass, dtype=jnp.float64)
        return (
            jnp.abs(charge_ - self.physical_charge) <= 1.0e-12 * abs(self.physical_charge)
        ) & (jnp.abs(mass_ - self.physical_mass) <= 1.0e-12 * self.physical_mass)

    def wiener_increments(
        self, key: PRNGKey, identity_high: ArrayLike, identity_low: ArrayLike, /
    ) -> Array:
        """Standard normal Wiener increments addressed by persistent identity.

        ``key`` must already be specific to the step (and consumer); each
        particle draws from ``derive_key(key, address, id_hi, id_lo)``, so the
        increment follows the particle, not its storage slot.
        """
        high = jnp.asarray(identity_high)
        low = jnp.asarray(identity_low)
        if high.dtype != jnp.uint32 or low.dtype != jnp.uint32:
            raise TypeError("Identity words must be uint32.")
        if high.shape != low.shape or high.ndim != 1:
            raise ValueError("Identity words must be matching one-dimensional arrays.")
        address = SampleAddress(
            "phydrax.radiation-reaction", "fokker-planck", role="wiener"
        )
        return jax.vmap(
            lambda hi, lo: jr.normal(derive_key(key, address, hi, lo), dtype=jnp.float64)
        )(high, low)

    def _reduced_force(
        self, velocity: Array, gamma: Array, electric: Array, magnetic: Array, /
    ) -> tuple[Array, Array]:
        """Reduced Landau–Lifshitz force and ``Q²``."""
        c2 = self.speed_of_light**2
        lorentz = electric + _cross(velocity, magnetic)
        work = jnp.sum(velocity * electric, axis=-1)
        invariant = jnp.maximum(jnp.sum(lorentz**2, axis=-1) - work**2 / c2, 0.0)
        coefficient = self.time_constant * self.physical_charge**2 / self.physical_mass
        force = coefficient * (
            _cross(lorentz, magnetic)
            + work[:, None] * electric / c2
            - (gamma**2 * invariant / c2)[:, None] * velocity
        )
        return force, invariant

    def _derivative_force(
        self,
        velocity: Array,
        gamma: Array,
        electric_gradient: Array,
        magnetic_gradient: Array,
        electric_rate: Array,
        magnetic_rate: Array,
        /,
    ) -> Array:
        """``τ q γ [D E + v × D B]`` with ``D = ∂_t + v·∇``."""
        convective_electric = electric_rate + contract(
            "nij,nj->ni", electric_gradient, velocity
        )
        convective_magnetic = magnetic_rate + contract(
            "nij,nj->ni", magnetic_gradient, velocity
        )
        return (
            (self.time_constant * self.physical_charge)
            * gamma[:, None]
            * (convective_electric + _cross(velocity, convective_magnetic))
        )

    def _require_tables(self) -> RadiationReactionTables:
        tables = self.tables
        if tables is None:
            raise ValueError(f"The {self.model!r} model lost its tables.")
        return tables

    def apply(
        self,
        proper_velocity: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        step_size: ArrayLike,
        active: ArrayLike,
        /,
        *,
        electric_gradient: ArrayLike | None = None,
        magnetic_gradient: ArrayLike | None = None,
        electric_rate: ArrayLike | None = None,
        magnetic_rate: ArrayLike | None = None,
        wiener: ArrayLike | None = None,
        grid_cutoff_frequency: ArrayLike | None = None,
    ) -> RadiationReactionResult:
        """One radiation-reaction step of ``u[N, 3]`` in fields ``E, B[N, 3]``.

        The ``"landau-lifshitz"`` model requires the gradients ``∂F_i/∂x_j``
        (``[N, 3, 3]``) and time derivatives ``∂_t F`` (``[N, 3]``) of both
        fields; the ``"stochastic-fokker-planck"`` model requires standard normal
        ``wiener[N]`` increments (`wiener_increments`). Other models refuse
        these inputs. ``grid_cutoff_frequency`` enables the scale-separation
        check against a field grid.
        """
        u = jnp.asarray(proper_velocity, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        mask = jnp.asarray(active, dtype=jnp.bool_)
        if u.ndim != 2 or u.shape[1] != 3 or e.shape != u.shape or b.shape != u.shape:
            raise ValueError("Proper velocities and fields must have shape [N, 3].")
        if mask.shape != u.shape[:1]:
            raise ValueError("active must have shape [N].")
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        derivatives = (electric_gradient, magnetic_gradient, electric_rate, magnetic_rate)
        if self.requires_field_derivatives != all(v is not None for v in derivatives):
            raise ValueError(
                "Field gradients and time derivatives are required by, and only by, "
                "the 'landau-lifshitz' model."
            )
        if self.stochastic != (wiener is not None):
            raise ValueError(
                "Wiener increments are required by, and only by, the "
                "'stochastic-fokker-planck' model."
            )
        c2 = self.speed_of_light**2
        rest_energy = self.physical_mass * c2
        gamma = jnp.sqrt(1.0 + jnp.sum(u**2, axis=-1) / c2)
        velocity = u / gamma[:, None]
        reduced, invariant = self._reduced_force(velocity, gamma, e, b)
        chi = gamma * jnp.sqrt(invariant) / self.critical_field
        diffusion = jnp.zeros_like(gamma)
        match self.model:
            case "landau-lifshitz-reduced":
                force = reduced
            case "landau-lifshitz":
                if (
                    electric_gradient is None
                    or magnetic_gradient is None
                    or electric_rate is None
                    or magnetic_rate is None
                ):
                    raise ValueError("The 'landau-lifshitz' model needs derivatives.")
                gradient_e = jnp.asarray(electric_gradient, dtype=jnp.float64)
                gradient_b = jnp.asarray(magnetic_gradient, dtype=jnp.float64)
                rate_e = jnp.asarray(electric_rate, dtype=jnp.float64)
                rate_b = jnp.asarray(magnetic_rate, dtype=jnp.float64)
                if gradient_e.shape != u.shape + (3,) or gradient_b.shape != u.shape + (
                    3,
                ):
                    raise ValueError("Field gradients must have shape [N, 3, 3].")
                if rate_e.shape != u.shape or rate_b.shape != u.shape:
                    raise ValueError("Field time derivatives must have shape [N, 3].")
                force = reduced + self._derivative_force(
                    velocity, gamma, gradient_e, gradient_b, rate_e, rate_b
                )
            case "quantum-corrected-landau-lifshitz":
                force = self._require_tables().power_correction(chi)[:, None] * reduced
            case "stochastic-fokker-planck":
                tables = self._require_tables()
                force = tables.power_correction(chi)[:, None] * reduced
                power = self.classical_power_scale * chi**2
                diffusion = (
                    gamma
                    * power
                    / rest_energy
                    * _DIFFUSION_SCALE
                    * chi
                    * tables.diffusion_correction(chi)
                )
            case _:
                assert_never(self.model)
        drift_rate = jnp.sum(force * velocity, axis=-1) / rest_energy
        drifted = u + dt * force / self.physical_mass
        drifted_gamma = jnp.sqrt(1.0 + jnp.sum(drifted**2, axis=-1) / c2)
        change = jnp.abs(gamma - drifted_gamma)
        if wiener is None:
            candidate = drifted
            nonphysical = jnp.zeros_like(mask)
        else:
            increments = jnp.asarray(wiener, dtype=jnp.float64)
            if increments.shape != mask.shape:
                raise ValueError("wiener must have shape [N].")
            deviation = jnp.sqrt(diffusion * dt)
            target = drifted_gamma + deviation * increments
            norm = jnp.sqrt(jnp.sum(drifted**2, axis=-1))
            moving = norm > 0.0
            ratio = jnp.where(
                moving,
                self.speed_of_light
                * jnp.sqrt(jnp.maximum(target**2 - 1.0, 0.0))
                / jnp.where(moving, norm, 1.0),
                1.0,
            )
            candidate = drifted * ratio[:, None]
            change = change + deviation
            nonphysical = target < 1.0
        kinetic = gamma - 1.0
        positive = kinetic > 0.0
        relative = jnp.where(
            positive,
            change / jnp.where(positive, kinetic, 1.0),
            jnp.where(change > 0.0, jnp.inf, 0.0),
        )
        applied = mask & (gamma >= self.minimum_gamma)
        updated = jnp.where(applied[:, None], candidate, u)
        updated_gamma = jnp.sqrt(1.0 + jnp.sum(updated**2, axis=-1) / c2)
        radiated = jnp.where(applied, rest_energy * (gamma - updated_gamma), 0.0)
        frequency = self.critical_frequency_scale * chi * gamma
        if grid_cutoff_frequency is None:
            separation = jnp.full_like(gamma, jnp.inf)
        else:
            separation = frequency / jnp.asarray(
                grid_cutoff_frequency, dtype=jnp.float64
            ).reshape(())
        emitting = applied & (chi > 0.0)
        finite = (
            jnp.all(jnp.isfinite(updated), axis=-1)
            & jnp.isfinite(chi)
            & jnp.isfinite(radiated)
        )
        unseparated = emitting & (separation < self.minimum_scale_separation)
        flags = jnp.zeros(mask.shape, dtype=jnp.int32)
        for condition, flag in (
            (mask & ~applied, RadiationReactionFlag.BELOW_MINIMUM_GAMMA),
            (applied & (chi > self.maximum_chi), RadiationReactionFlag.CHI_EXCEEDED),
            (
                applied & ((relative > self.maximum_relative_step_loss) | nonphysical),
                RadiationReactionFlag.STEP_LOSS_EXCEEDED,
            ),
            (unseparated, RadiationReactionFlag.SCALE_UNSEPARATED),
            (mask & ~finite, RadiationReactionFlag.NONFINITE),
        ):
            flags = jnp.where(condition, flags | int(flag), flags)
        supported = ~mask | ((flags & int(_SUPPORT_FAILURE)) == 0)
        return RadiationReactionResult(
            updated,
            radiated,
            chi,
            frequency,
            separation,
            drift_rate,
            diffusion,
            flags,
            applied,
            emitting,
            supported,
            ~unseparated,
            jnp.all(supported),
        )


class RadiationReactionProcess(AbstractPICProcess, NonTrainableState):
    """`RadiationReactionPlan` on one PIC species as a momentum-stage process.

    The species must carry the plan's physical charge-to-mass ratio with a fixed
    charge number, and the PIC pusher must use the plan's units and speed of
    light (both refused at run construction). Each macroparticle represents
    ``mass / physical_mass`` physical particles. The process claims
    ``"subgrid-reaction"`` radiation ownership; its ledger radiation reports
    the radiated energy (equal to the kinetic energy removed, whose difference
    is ``energy_defect``) and the scale separation to the grid cutoff. The
    Fokker–Planck model draws Wiener increments from the step's process key
    addressed by each particle's persistent identity. Per-particle
    `RadiationReactionResult` evidence is the process evidence.
    """

    plan: RadiationReactionPlan
    species_index: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(self, plan: RadiationReactionPlan, species: int, /) -> None:
        if isinstance(species, bool) or not isinstance(species, int) or species < 0:
            raise ValueError("species must be a nonnegative species index.")
        self.plan = plan
        self.species_index = species
        self.stage = "momentum"
        self.stochastic = plan.stochastic
        self.radiation_ownership = "subgrid-reaction"
        self.species_indices = (species,)
        self.process_id = canonical_fingerprint(
            {
                "kind": "pic-radiation-reaction-process",
                "plan": plan.plan_id,
                "species": species,
            }
        )

    @property
    def requires_field_derivatives(self) -> bool:
        return self.plan.requires_field_derivatives

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        self.plan.validate_relativity(relativity)
        model = species[self.species_index].charge_model
        if model.minimum_charge_number != model.maximum_charge_number:
            raise ValueError(
                "Radiation reaction requires a species with a fixed charge number."
            )
        specific = model.base_specific_charge * model.initial_charge_number
        expected = self.plan.physical_charge / self.plan.physical_mass
        if abs(specific - expected) > 1.0e-12 * abs(expected):
            raise ValueError(
                f"Species {model.species_id!r} charge-to-mass ratio {specific!r} "
                f"differs from the radiation-reaction species ratio {expected!r}."
            )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        del species
        index = self.species_index
        state = context.species[index]
        population = state.population
        proper = state.particles.proper_velocity
        wiener = None
        if self.stochastic:
            if context.key is None:
                raise ValueError("Stochastic radiation reaction requires a process key.")
            wiener = self.plan.wiener_increments(
                context.key, population.id_hi, population.id_lo
            )
        gradients_e, gradients_b = context.electric_gradient, context.magnetic_gradient
        rates_e, rates_b = context.electric_rate, context.magnetic_rate
        result = self.plan.apply(
            proper,
            context.electric[index],
            context.magnetic[index],
            context.step_size,
            population.active,
            electric_gradient=None if gradients_e is None else gradients_e[index],
            magnetic_gradient=None if gradients_b is None else gradients_b[index],
            electric_rate=None if rates_e is None else rates_e[index],
            magnetic_rate=None if rates_b is None else rates_b[index],
            wiener=wiener,
            grid_cutoff_frequency=context.grid_cutoff_frequency,
        )
        successful = result.successful
        mass = population.mass.astype(jnp.float64)
        c2 = self.plan.speed_of_light**2
        before = jnp.sqrt(1.0 + jnp.sum(proper.astype(jnp.float64) ** 2, axis=-1) / c2)
        after = jnp.sqrt(1.0 + jnp.sum(result.proper_velocity**2, axis=-1) / c2)
        radiated = jnp.sum(
            jnp.where(
                result.applied,
                mass / self.plan.physical_mass * result.radiated_energy,
                0.0,
            )
        )
        kinetic_loss = jnp.sum(
            jnp.where(population.active, mass * c2 * (before - after), 0.0)
        )
        zero = jnp.zeros((), dtype=jnp.float64)
        values = list(context.species)
        values[index] = PICSpeciesState(
            PICParticleState(
                state.particles.position,
                jnp.where(successful, result.proper_velocity, proper).astype(
                    proper.dtype
                ),
            ),
            population,
            state.charge,
        )
        return PICProcessResult(
            tuple(values),
            PICProcessLedger(
                jnp.sum(result.applied, dtype=jnp.int32),
                zero,
                zero,
                jnp.where(successful, radiated - kinetic_loss, zero),
                successful,
                self.process_id,
                PICProcessRadiation(
                    jnp.where(successful, radiated, zero),
                    jnp.max(
                        jnp.where(result.emitting, result.critical_frequency, 0.0),
                        initial=0.0,
                    ),
                    jnp.min(
                        jnp.where(result.emitting, result.scale_separation, jnp.inf),
                        initial=jnp.inf,
                    ),
                    jnp.all(result.scale_separated),
                ),
            ),
            result,
        )


__all__ = [
    "RadiationReactionFlag",
    "RadiationReactionModel",
    "RadiationReactionPlan",
    "RadiationReactionProcess",
    "RadiationReactionResult",
    "RadiationReactionTables",
]
