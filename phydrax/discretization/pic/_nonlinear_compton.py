#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Monte Carlo nonlinear Compton emission in the locally constant field.

`NonlinearComptonPlan` emits photons from leptons of one physical species
(charge ``q``, mass ``m``) in the units of one `ElectromagneticScaleContract`.
With ``α_q = q²/(4π ε₀ ħ c)``, ``τ_C = ħ/(mc²)`` and the quantum parameter
``χ = γ √((E + v×B)² − (v·E)²/c²) / E_S`` (``E_S = m²c³/(|q|ħ)``) the photon
rate and spectrum are

    dW/dξ = (α_q/(τ_C γ)) s(χ, ξ),   W = (α_q/(τ_C γ)) K(χ),

with ``s`` and ``K`` from a Compton `QEDTable`. Models:

- ``"lcfa"``: the locally constant field spectrum;
- ``"improved-lcfa"`` (Di Piazza, Tamburini, Meuren & Keitel, PRA 99,
  022125, 2019): below the photon energy ``ω_th = ζ ω_LCFA`` the spectrum is
  held at its value at ``ω_th``, removing the unphysical ``ξ^{−2/3}`` infrared
  divergence. ``ω_LCFA = ε / (1 + (4/(3χ)) (3S + 4S³))`` with
  ``S = χ τ/(8 γ τ_C)`` equates the formation time
  ``t_f(ω) = τ_C (8γ/χ) sinh(arsinh((3/4) χ ε'/ω)/3)`` to the local field
  variation time ``τ = 2 √(F⊥² / (Ḟ⊥² + |F⊥·F̈⊥|))`` along the trajectory
  (``F⊥`` the Lorentz force transverse to the momentum); ``ζ = 0.7`` is the
  paper's tuning. Without a finite ``τ`` (constant fields, or a particle with
  fewer than two recorded steps) the model is the LCFA; when
  ``ε − ω_LCFA < 10⁻³ ε`` no emission occurs.

Emission uses the optical-depth method: each lepton carries an optical depth
``τ_opt`` drawn as ``−log U``; it decreases by ``W h`` per subcycle ``h`` and a
photon is emitted when it reaches zero. The next target is a fresh draw minus
the overshoot, so the emissions are the renewal points of the cumulative
optical depth (an unbiased Poisson process at fixed rate; the overshoot is
charged at the pre-emission rate). A second crossing within one subcycle stays
pending as a nonpositive depth and is emitted in the next subcycle (or the
next step), never discarded. The step
is split into ``n = ceil(W Δt / p_max)`` subcycles (at most
``maximum_subcycles``) so that each subcycle's event probability stays below
``maximum_event_probability``; a particle that would need more subcycles is a
support failure. Photons are emitted along the lepton momentum ``p̂``. The
declared ``conservation`` fixes the recoil: ``"momentum"`` keeps
``p' = p − (ξε/c) p̂`` exactly and hands the energy defect
``ε' + ξε − ε = 2ξε m²c⁴/((ε + |p|c)(ε' + (1 − ξ)ε)) = O(m²c⁴/ε)`` to the
field; ``"energy"`` keeps ``ε' = (1 − ξ)ε`` exactly and hands the momentum
defect along ``p̂`` to the field. Random numbers are addressed by the step key,
the particle's persistent identity, and the subcycle event index. Emission is
a discrete event: results are not differentiable with respect to it.

``polarization`` (`QEDPolarizationModel`) resolves particle polarizations
with the Seipt–King LCFA rates (Seipt & King, PRA 102, 052805, 2020) in the
spin quantization axis (SQA) scheme of Li et al. (PRL 122, 154801, 2019):

- ``"unpolarized"``: the spin- and polarization-averaged rates above;
- ``"photon-polarized"``: unpolarized leptons; each emitted photon's linear
  polarization is sampled along ``e₁ = F̂⊥`` (the transverse Lorentz force) or
  ``e₂ = p̂ × e₁`` from the channel rates at its energy, whose mean Stokes
  parameter is ``ξ₃ = G(δ)/(F(δ) + ξ²/(1 − ξ) G(δ))``;
- ``"spin-and-photon-polarized"``: leptons carry a rest-frame polarization
  vector ``S`` (``|S| ≤ 1``). With ``ê = v̂ × F̂⊥`` (the rest-frame magnetic
  field direction for an electron) and ``P = S·ê`` the emission rate is the
  mixture ``W(P) = ((1 + P)/2) W₊ + ((1 − P)/2) W₋`` of the ``"positive"`` and
  ``"negative"`` spin tables. An emission samples the photon energy from that
  mixture and then the final spin ``P_f = ±1`` and the photon polarization
  jointly from the four Seipt–King channels at that energy; the lepton spin
  collapses to ``P_f ê`` (spin flips included). Without emission the spin
  follows the no-emission (mass-operator) evolution
  ``ρ ← e^{−Γh/2} ρ e^{−Γh/2}/Tr`` with ``Γ = W₀ + W₁ σ·ê``:
  ``P ← (P − t)/(1 − P t)``, ``S⊥ ← S⊥/(cosh(W₁h) − P sinh(W₁h))``,
  ``t = tanh(W₁h)``, and the optical depth decreases by the exact survival
  exponent ``W₀h − log(cosh(W₁h) − P sinh(W₁h))``. The ensemble then
  polarizes antiparallel to ``ê`` (Sokolov–Ternov: electron spins
  antiparallel, positron spins parallel to a magnetic field).
  Thomas–Bargmann–Michel–Telegdi precession is owned by the pusher
  (`RelativisticPushPlan.precess`).

The polarized models draw one more uniform per subcycle (the channel); the
unpolarized model's arithmetic and random numbers are unchanged.
"""

from __future__ import annotations

import math
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...special import synchrotron_f, synchrotron_g, synchrotron_h
from ...typing import parse, PRNGKey
from ._qed_tables import (
    _INVERSE_NORMALIZATION,
    QED_SUPPORT_FAILURE,
    QEDConservation,
    QEDEventFlag,
    QEDPolarizationModel,
    QEDTable,
)


QEDEmissionModel: TypeAlias = Literal["lcfa", "improved-lcfa"]

_EVENT_ADDRESS = SampleAddress("phydrax.qed", "nonlinear-compton", role="event")
_DEPTH_ADDRESS = SampleAddress("phydrax.qed", "nonlinear-compton", role="optical-depth")
# Di Piazza et al. 2019: no emission when ε − ω_LCFA < 10⁻³ ε.
_MINIMUM_LCFA_WINDOW = 1.0e-3


def _cross(left: Array, right: Array, /) -> Array:
    return jnp.cross(left, right, axis=-1)


_SIGNS = (1.0, -1.0)


def _unit(vector: Array, /) -> Array:
    """``v/|v|`` along the last axis (zero for zero vectors)."""
    norm = jnp.sqrt(jnp.sum(vector**2, axis=-1, keepdims=True))
    return jnp.where(norm > 0.0, vector / jnp.where(norm > 0.0, norm, 1.0), 0.0)


def _survive(polarization: Array, axis: Array, exponent: Array, /) -> tuple[Array, Array]:
    """Polarization after surviving ``Γ = W₀ + W₁ σ·axis`` for ``W₁h = exponent``.

    Returns the normalized polarization vector and ``log(cosh x − P sinh x)``
    (the spin-dependent part of the survival exponent).
    """
    projection = jnp.sum(polarization * axis, axis=-1)
    transverse = polarization - projection[:, None] * axis
    cosh = jnp.cosh(exponent)
    sinh = jnp.sinh(exponent)
    norm = cosh - projection * sinh
    updated = (projection * cosh - sinh) / norm
    return updated[:, None] * axis + transverse / norm[:, None], jnp.log(norm)


def _pick(weights: Array, uniform: Array, /) -> Array:
    """Index sampled from nonnegative ``weights[..., K]`` by one uniform."""
    cumulative = jnp.cumsum(weights, axis=-1)
    target = uniform * cumulative[..., -1]
    return jnp.sum(cumulative[..., :-1] <= target[..., None], axis=-1)


def _formation_time(
    chi: Array, gamma: Array, energy_ratio: Array, compton_time: float, /
) -> Array:
    """``τ_C (8γ/χ) sinh(arsinh((3/4) χ ρ)/3)`` with ``ρ = ε'/ω`` (``2γτ_Cρ`` as ``χ→0``)."""
    argument = 0.75 * chi * energy_ratio
    small = argument < 1.0e-6
    safe = jnp.where(small, 1.0, argument)
    safe_chi = jnp.where(small, 1.0, chi)
    exact = 8.0 * gamma * jnp.sinh(jnp.arcsinh(safe) / 3.0) / safe_chi
    return compton_time * jnp.where(small, 2.0 * gamma * energy_ratio, exact)


class NonlinearComptonResult(StrictModule):
    """One nonlinear Compton step of ``N`` leptons with ``S`` subcycle slots.

    Per particle: the candidate ``proper_velocity`` and ``optical_depth``,
    step-start ``quantum_parameter``, ``lorentz_factor``, ``rate``, and
    ``critical_frequency`` (``3χγmc²/(2ħ)``), the ``subcycles`` used, the largest
    subcycle ``event_probability``, the field-exchange ``field_energy`` and
    ``field_momentum`` (per physical particle, energy and momentum the field
    supplied to close the declared kinematics), ``scale_separation`` (critical
    over grid cutoff frequency), and `QEDEventFlag` ``flags``. Per subcycle
    slot: ``emitted`` photons with ``photon_momentum`` (per physical photon),
    ``photon_energy``, energy ``photon_fraction``, ``formation_ratio`` (formation
    over field-variation time; zero in constant fields) and ``event_flags``.
    ``supported`` excludes support failures of active particles; ``successful``
    is ``all(supported)``.

    Polarized models also report, per subcycle slot, each emitted photon's
    linear Stokes parameter ``photon_stokes`` (``±1``) relative to its
    polarization axis ``photon_axis`` (``e₁ = F̂⊥``); the spin model reports
    the candidate polarization vectors ``spin[N, 3]`` and, per slot,
    ``spin_projection`` (initial ``P = S·ê`` and sampled final ``P_f``) of
    each emission. Fields a model does not resolve are ``None``.
    """

    proper_velocity: Array
    optical_depth: Array
    quantum_parameter: Array
    lorentz_factor: Array
    rate: Array
    critical_frequency: Array
    scale_separation: Array
    subcycles: Array
    event_probability: Array
    field_energy: Array
    field_momentum: Array
    flags: Array
    emitted: Array
    photon_momentum: Array
    photon_energy: Array
    photon_fraction: Array
    formation_ratio: Array
    event_flags: Array
    applied: Array
    supported: Array
    scale_separated: Array
    successful: Array
    spin: Array | None
    photon_stokes: Array | None
    photon_axis: Array | None
    spin_projection: Array | None


class NonlinearComptonPlan(StrictModule, NonTrainableState):
    """Photon emission by leptons of one physical species in one electromagnetic scale.

    ``physical_charge``/``physical_mass`` are the single-particle charge and
    mass in ``scale`` units; ``table`` is a ``"nonlinear-compton"`` `QEDTable`
    covering ``maximum_chi``. Leptons with ``γ < minimum_gamma`` are exempt.
    ``minimum_scale_separation`` is the smallest admissible ratio of an
    emitting particle's critical frequency to the field-grid cutoff
    (``subgrid-reaction`` ownership). ``polarization`` selects the resolved
    polarizations; ``"spin-and-photon-polarized"`` requires ``spin_tables``,
    the ``"positive"`` and ``"negative"`` Compton tables covering
    ``maximum_chi``.
    """

    table: QEDTable
    spin_tables: tuple[QEDTable, QEDTable] | None
    model: QEDEmissionModel = eqx.field(static=True)
    polarization: QEDPolarizationModel = eqx.field(static=True)
    conservation: QEDConservation = eqx.field(static=True)
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    physical_charge: float = eqx.field(static=True)
    physical_mass: float = eqx.field(static=True)
    maximum_chi: float = eqx.field(static=True)
    minimum_gamma: float = eqx.field(static=True)
    maximum_event_probability: float = eqx.field(static=True)
    maximum_subcycles: int = eqx.field(static=True)
    infrared_tuning: float = eqx.field(static=True)
    minimum_scale_separation: float = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    rest_energy: float = eqx.field(static=True)
    critical_field: float = eqx.field(static=True)
    compton_time: float = eqx.field(static=True)
    rate_scale: float = eqx.field(static=True)
    critical_frequency_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: QEDEmissionModel,
        scale: ElectromagneticScaleContract,
        physical_charge: float,
        physical_mass: float,
        table: QEDTable,
        /,
        *,
        maximum_chi: float,
        minimum_gamma: float,
        maximum_event_probability: float = 0.1,
        maximum_subcycles: int = 8,
        conservation: QEDConservation = "momentum",
        infrared_tuning: float = 0.7,
        minimum_scale_separation: float = 10.0,
        polarization: QEDPolarizationModel = "unpolarized",
        spin_tables: tuple[QEDTable, QEDTable] | None = None,
    ) -> None:
        model_ = parse(model, QEDEmissionModel, "model")
        polarization_ = parse(polarization, QEDPolarizationModel, "polarization")
        conservation_ = parse(conservation, QEDConservation, "conservation")
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        if not isinstance(table, QEDTable):
            raise TypeError("table must be a QEDTable.")
        if table.process != "nonlinear-compton":
            raise ValueError("NonlinearComptonPlan requires a 'nonlinear-compton' table.")
        if table.polarization != "averaged":
            raise ValueError("table must be the 'averaged' Compton table.")
        charge = float(physical_charge)
        if not math.isfinite(charge) or charge == 0.0:
            raise ValueError("physical_charge must be finite and nonzero.")
        mass = positive_finite_float(physical_mass, "physical_mass")
        chi = positive_finite_float(maximum_chi, "maximum_chi")
        if table.maximum_chi < chi:
            raise ValueError("table must cover maximum_chi.")
        spin_tables_ = _validated_spin_tables(polarization_, spin_tables, chi)
        gamma = float(minimum_gamma)
        if not math.isfinite(gamma) or gamma < 1.0:
            raise ValueError("minimum_gamma must be finite and at least one.")
        probability = positive_finite_float(
            maximum_event_probability, "maximum_event_probability"
        )
        if probability >= 1.0:
            raise ValueError("maximum_event_probability must be below one.")
        subcycles = positive_integer(maximum_subcycles, "maximum_subcycles")
        tuning = positive_finite_float(infrared_tuning, "infrared_tuning")
        if tuning > 1.0:
            raise ValueError("infrared_tuning must lie in (0, 1].")
        separation = positive_finite_float(
            minimum_scale_separation, "minimum_scale_separation"
        )
        light = float(scale.speed_of_light)
        hbar = float(scale.reduced_planck_constant)
        permittivity = float(scale.vacuum_permittivity)
        rest = mass * light**2
        alpha = charge**2 / (4.0 * math.pi * permittivity * hbar * light)
        self.table = table
        self.spin_tables = spin_tables_
        self.model = model_
        self.polarization = polarization_
        self.conservation = conservation_
        self.scale = scale
        self.physical_charge = charge
        self.physical_mass = mass
        self.maximum_chi = chi
        self.minimum_gamma = gamma
        self.maximum_event_probability = probability
        self.maximum_subcycles = subcycles
        self.infrared_tuning = tuning
        self.minimum_scale_separation = separation
        self.speed_of_light = light
        self.rest_energy = rest
        self.critical_field = mass**2 * light**3 / (abs(charge) * hbar)
        self.compton_time = hbar / rest
        self.rate_scale = alpha * rest / hbar
        self.critical_frequency_scale = 1.5 * rest / hbar
        identity: dict[str, object] = {
            "kind": "nonlinear-compton-plan",
            "model": model_,
            "conservation": conservation_,
            "scale": scale.scale_id,
            "physical_charge": charge,
            "physical_mass": mass,
            "table": table.table_id,
            "maximum_chi": chi,
            "minimum_gamma": gamma,
            "maximum_event_probability": probability,
            "maximum_subcycles": subcycles,
            "infrared_tuning": tuning,
            "minimum_scale_separation": separation,
        }
        # Unpolarized plans keep the identity (and so the random streams keyed
        # by it) of the unpolarized model.
        if self.resolves_photon_polarization:
            identity["polarization"] = polarization_
            identity["spin_tables"] = (
                None
                if spin_tables_ is None
                else [value.table_id for value in spin_tables_]
            )
        self.plan_id = canonical_fingerprint(identity)

    def validate_relativity(self, relativity: RelativityScaleContract, /) -> None:
        """Refuse a pusher whose units or speed of light differ from ``scale``."""
        if not isinstance(relativity, RelativityScaleContract):
            raise TypeError("relativity must be a RelativityScaleContract.")
        own = self.scale.relativity
        if relativity.dimensional_scale.scale_id != own.dimensional_scale.scale_id:
            raise ValueError(
                "The pusher and the QED scale use different length, mass, or time units."
            )
        if relativity.speed_of_light != own.speed_of_light:
            raise ValueError("The pusher speed of light differs from the QED scale.")

    # -- kinematics --------------------------------------------------------------

    def quantum_parameter(
        self, proper_velocity: ArrayLike, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Lorentz factor ``γ`` and quantum parameter ``χ`` of ``u[N, 3]`` in ``E, B``."""
        u = jnp.asarray(proper_velocity, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        gamma = jnp.sqrt(1.0 + jnp.sum(u**2, axis=-1) / self.speed_of_light**2)
        velocity = u / gamma[..., None]
        lorentz = e + _cross(velocity, b)
        work = jnp.sum(velocity * e, axis=-1)
        invariant = jnp.maximum(
            jnp.sum(lorentz**2, axis=-1) - work**2 / self.speed_of_light**2, 0.0
        )
        return gamma, gamma * jnp.sqrt(invariant) / self.critical_field

    def transverse_force(
        self, proper_velocity: ArrayLike, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> Array:
        """Lorentz force ``q(E + v×B)`` without its component along the momentum."""
        u = jnp.asarray(proper_velocity, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        gamma = jnp.sqrt(1.0 + jnp.sum(u**2, axis=-1) / self.speed_of_light**2)
        force = self.physical_charge * (e + _cross(u / gamma[..., None], b))
        norm = jnp.sqrt(jnp.sum(u**2, axis=-1, keepdims=True))
        direction = jnp.where(norm > 0.0, u / jnp.where(norm > 0.0, norm, 1.0), 0.0)
        return force - jnp.sum(force * direction, axis=-1, keepdims=True) * direction

    def infrared_threshold(
        self, chi: ArrayLike, gamma: ArrayLike, variation_time: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Improved-LCFA threshold fraction ``ζ ω_LCFA/ε`` and emission admissibility."""
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        gamma_ = jnp.asarray(gamma, dtype=jnp.float64)
        tau = jnp.asarray(variation_time, dtype=jnp.float64)
        local = jnp.isfinite(tau) & (chi_ > 0.0)
        s = jnp.where(local, chi_ * tau / (8.0 * gamma_ * self.compton_time), 0.0)
        denominator = 1.0 + 4.0 * (3.0 * s + 4.0 * s**3) / (
            3.0 * jnp.where(local, chi_, 1.0)
        )
        fraction = jnp.where(local, 1.0 / denominator, 0.0)
        return self.infrared_tuning * fraction, ~(
            local & (1.0 - fraction < _MINIMUM_LCFA_WINDOW)
        )

    # -- polarization ------------------------------------------------------------

    @property
    def resolves_photon_polarization(self) -> bool:
        match self.polarization:
            case "unpolarized":
                return False
            case "photon-polarized" | "spin-and-photon-polarized":
                return True
            case _:
                assert_never(self.polarization)

    @property
    def resolves_spin(self) -> bool:
        match self.polarization:
            case "unpolarized" | "photon-polarized":
                return False
            case "spin-and-photon-polarized":
                return True
            case _:
                assert_never(self.polarization)

    @property
    def uniform_count(self) -> int:
        """Uniforms per subcycle: fraction, optical depth (and polarization channel)."""
        return 3 if self.resolves_photon_polarization else 2

    def _require_spin_tables(self) -> tuple[QEDTable, QEDTable]:
        if self.spin_tables is None:
            raise ValueError("This NonlinearComptonPlan does not resolve lepton spin.")
        return self.spin_tables

    def spin_axis(
        self, proper_velocity: ArrayLike, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> Array:
        """Spin quantization axis ``ê = v̂ × F̂⊥`` of ``u[N, 3]`` (zero where undefined)."""
        u = jnp.asarray(proper_velocity, dtype=jnp.float64)
        force = self.transverse_force(u, electric, magnetic)
        return _unit(_cross(_unit(u), force))

    def channel_spectrum(
        self, chi: ArrayLike, fraction: ArrayLike, spin_projection: ArrayLike, /
    ) -> Array:
        """Seipt–King channel densities ``s(χ, ξ; P → P_f, τ)`` with shape ``[..., 2, 2]``.

        ``spin_projection`` is the initial ``P = S·ê`` (``|P| ≤ 1``); the last
        two axes are the final spin ``P_f = (+1, −1)`` and the photon linear
        Stokes parameter ``τ = (+1, −1)`` along ``e₁ = F̂⊥``. Their sum is the
        spin-resolved spectrum ``s(χ, ξ) − P ξH(δ)/(√3πδ)``, whose average over
        ``P = ±1`` is the unpolarized ``s`` of `QEDTable.spectrum`.
        """
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        xi = jnp.asarray(fraction, dtype=jnp.float64)
        projection = jnp.asarray(spin_projection, dtype=jnp.float64)
        chi_, xi, projection = jnp.broadcast_arrays(chi_, xi, projection)
        interior = (xi > 0.0) & (xi < 1.0) & (chi_ > 0.0)
        safe_xi = jnp.where(interior, xi, 0.5)[..., None, None]
        safe_chi = jnp.where(interior, chi_, 1.0)[..., None, None]
        initial = projection[..., None, None]
        final = jnp.asarray(_SIGNS)[:, None]
        tau = jnp.asarray(_SIGNS)[None, :]
        delta = 2.0 * safe_xi / (3.0 * safe_chi * (1.0 - safe_xi))
        f = synchrotron_f(delta)
        g = synchrotron_g(delta)
        h = synchrotron_h(delta)
        ratio = safe_xi / (1.0 - safe_xi)
        kibble = 1.0 + 0.5 * safe_xi * ratio
        product = initial * final
        a = 1.0 + product + tau * product * (1.0 - kibble)
        b = safe_xi * initial + ratio * final + tau * (ratio * initial + safe_xi * final)
        c = kibble + product + 0.5 * tau * (1.0 + kibble * product)
        # δ·[−A∫K_{1/3} − B K_{1/3} + 2C K_{2/3}] with ∫K_{1/3} = (2G − F)/δ.
        value = (-a * (2.0 * g - f) - b * h + 2.0 * c * g) / delta
        return jnp.where(
            interior[..., None, None], 0.25 * _INVERSE_NORMALIZATION * value, 0.0
        )

    def photon_stokes_parameter(self, chi: ArrayLike, fraction: ArrayLike, /) -> Array:
        """Mean linear Stokes ``ξ₃`` along ``F̂⊥`` of photons from unpolarized leptons.

        ``ξ₃ = G(δ)/(F(δ) + ξ²/(1 − ξ) G(δ))``: ``(G ± …)`` weight the photons
        polarized along and across the transverse force.
        """
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        xi = jnp.asarray(fraction, dtype=jnp.float64)
        chi_, xi = jnp.broadcast_arrays(chi_, xi)
        interior = (xi > 0.0) & (xi < 1.0) & (chi_ > 0.0)
        safe_xi = jnp.where(interior, xi, 0.5)
        safe_chi = jnp.where(interior, chi_, 1.0)
        delta = 2.0 * safe_xi / (3.0 * safe_chi * (1.0 - safe_xi))
        g = synchrotron_g(delta)
        total = synchrotron_f(delta) + safe_xi**2 / (1.0 - safe_xi) * g
        return jnp.where(interior, g / total, 0.0)

    # -- rates and spectra -------------------------------------------------------

    def _spectral_weights(
        self, table: QEDTable, chi: Array, gamma: Array, variation_time: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        """Dimensionless total, threshold, threshold CDF and flat mass of ``table``."""
        total = table.rate_function(chi)
        match self.model:
            case "lcfa":
                zero = jnp.zeros_like(chi)
                return total, zero, zero, zero
            case "improved-lcfa":
                threshold, allowed = self.infrared_threshold(chi, gamma, variation_time)
                below = table.cdf(chi, threshold)
                flat = threshold * table.spectrum(chi, threshold)
                weight = jnp.where(allowed, total * (1.0 - below) + flat, 0.0)
                return weight, threshold, below, jnp.where(allowed, flat, 0.0)
            case _:
                assert_never(self.model)

    def _spin_weights(
        self, chi: Array, gamma: Array, variation_time: Array, /
    ) -> tuple[Array, Array]:
        """Dimensionless totals of spin projections ``P = +1`` and ``P = −1``."""
        positive, negative = self._require_spin_tables()
        return (
            self._spectral_weights(positive, chi, gamma, variation_time)[0],
            self._spectral_weights(negative, chi, gamma, variation_time)[0],
        )

    def rate(
        self,
        proper_velocity: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        /,
        *,
        variation_time: ArrayLike | None = None,
        spin: ArrayLike | None = None,
    ) -> Array:
        """Photon emission rate ``W`` of each lepton (per unit time).

        Without ``spin`` the spin-averaged rate; with polarization vectors
        ``spin[N, 3]`` (spin model only) the rate ``W(S·ê)``.
        """
        gamma, chi = self.quantum_parameter(proper_velocity, electric, magnetic)
        tau = (
            jnp.full_like(chi, jnp.inf)
            if variation_time is None
            else jnp.asarray(variation_time, dtype=jnp.float64)
        )
        if spin is None:
            weight, _, _, _ = self._spectral_weights(self.table, chi, gamma, tau)
            return self.rate_scale * weight / gamma
        projection = jnp.sum(
            jnp.asarray(spin, dtype=jnp.float64)
            * self.spin_axis(proper_velocity, electric, magnetic),
            axis=-1,
        )
        positive, negative = self._spin_weights(chi, gamma, tau)
        weight = 0.5 * (1.0 + projection) * positive + 0.5 * (1.0 - projection) * negative
        return self.rate_scale * weight / gamma

    def sample_fraction(
        self,
        chi: ArrayLike,
        gamma: ArrayLike,
        uniform: ArrayLike,
        /,
        *,
        variation_time: ArrayLike | None = None,
        spin_projection: ArrayLike | None = None,
    ) -> Array:
        """Photon energy fraction ``ξ`` from one uniform per lepton (inverse CDF).

        ``spin_projection`` (spin model only) samples the spectrum of a lepton
        with ``P = S·ê``.
        """
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        gamma_ = jnp.asarray(gamma, dtype=jnp.float64)
        tau = (
            jnp.full_like(chi_, jnp.inf)
            if variation_time is None
            else jnp.asarray(variation_time, dtype=jnp.float64)
        )
        draw = jnp.asarray(uniform, dtype=jnp.float64)
        if spin_projection is None:
            return self._sample(self.table, chi_, gamma_, tau, draw)[0]
        projection = jnp.asarray(spin_projection, dtype=jnp.float64)
        return self._sample_spin(chi_, gamma_, tau, projection, draw)[0]

    def _sample(
        self,
        table: QEDTable,
        chi: Array,
        gamma: Array,
        variation_time: Array,
        uniform: Array,
        /,
    ) -> tuple[Array, Array]:
        """Fraction and whether it came from the flat infrared part."""
        weight, threshold, below, flat = self._spectral_weights(
            table, chi, gamma, variation_time
        )
        flat_probability = jnp.where(
            weight > 0.0, flat / jnp.where(weight > 0.0, weight, 1.0), 0.0
        )
        infrared = uniform < flat_probability
        rescaled = jnp.where(
            infrared,
            0.0,
            (uniform - flat_probability)
            / jnp.where(flat_probability < 1.0, 1.0 - flat_probability, 1.0),
        )
        fraction = table.quantile(chi, below + rescaled * (1.0 - below))
        uniform_below = (
            threshold * uniform / jnp.where(flat_probability > 0.0, flat_probability, 1.0)
        )
        return jnp.where(infrared, uniform_below, fraction), infrared

    def _sample_spin(
        self,
        chi: Array,
        gamma: Array,
        variation_time: Array,
        projection: Array,
        uniform: Array,
        /,
    ) -> tuple[Array, Array]:
        """Fraction from the spin mixture ``((1 ± P)/2) s_±``: one uniform picks the
        component and, rescaled, samples it."""
        positive, negative = self._require_spin_tables()
        upper, lower = self._spin_weights(chi, gamma, variation_time)
        upper = 0.5 * (1.0 + projection) * upper
        total = upper + 0.5 * (1.0 - projection) * lower
        share = jnp.where(total > 0.0, upper / jnp.where(total > 0.0, total, 1.0), 1.0)
        chosen = uniform < share
        rescaled = jnp.where(
            chosen,
            uniform / jnp.where(share > 0.0, share, 1.0),
            (uniform - share) / jnp.where(share < 1.0, 1.0 - share, 1.0),
        )
        rescaled = jnp.clip(rescaled, 0.0, 1.0)
        first, first_infrared = self._sample(
            positive, chi, gamma, variation_time, rescaled
        )
        second, second_infrared = self._sample(
            negative, chi, gamma, variation_time, rescaled
        )
        return (
            jnp.where(chosen, first, second),
            jnp.where(chosen, first_infrared, second_infrared),
        )

    # -- random numbers ----------------------------------------------------------

    def uniforms(
        self, key: PRNGKey, identity_high: ArrayLike, identity_low: ArrayLike, /
    ) -> Array:
        """``[N, S, uniform_count]`` uniforms in ``[0, 1)`` by identity and subcycle.

        ``key`` must be specific to the step (and consumer); particle ``n`` and
        subcycle ``k`` draw from ``derive_key(key, address, id_hi, id_lo, k)``:
        the first uniform samples the photon fraction, the second the new
        optical depth, and the third (polarized models) the polarization
        channel.
        """
        high, low = _identity_words(identity_high, identity_low)
        events = jnp.arange(self.maximum_subcycles, dtype=jnp.uint32)
        width = self.uniform_count
        return jax.vmap(
            lambda hi, lo: jax.vmap(
                lambda event: jr.uniform(
                    derive_key(key, _EVENT_ADDRESS, hi, lo, event),
                    (width,),
                    dtype=jnp.float64,
                )
            )(events)
        )(high, low)

    def initial_optical_depth(
        self, key: PRNGKey, identity_high: ArrayLike, identity_low: ArrayLike, /
    ) -> Array:
        """Fresh optical depths ``−log(1 − U)`` addressed by identity."""
        high, low = _identity_words(identity_high, identity_low)
        return jax.vmap(
            lambda hi, lo: (
                -jnp.log1p(
                    -jr.uniform(
                        derive_key(key, _DEPTH_ADDRESS, hi, lo), dtype=jnp.float64
                    )
                )
            )
        )(high, low)

    # -- step ------------------------------------------------------------------

    def _recoil(
        self, u: Array, gamma: Array, fraction: Array, /
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        """Recoiled ``u``, photon momentum and energy, field energy and momentum.

        The second value marks draws the declared conservation can realize.
        """
        c = self.speed_of_light
        mass = self.physical_mass
        energy = gamma * self.rest_energy
        norm = jnp.sqrt(jnp.sum(u**2, axis=-1))
        moving = norm > 0.0
        direction = jnp.where(
            moving[:, None], u / jnp.where(moving, norm, 1.0)[:, None], 0.0
        )
        momentum = mass * norm
        photon_energy = fraction * energy
        photon = (photon_energy / c)[:, None] * direction
        rest2 = self.rest_energy**2
        match self.conservation:
            case "momentum":
                remaining = momentum - photon_energy / c
                feasible = moving & (remaining >= 0.0)
                after_energy = jnp.sqrt(rest2 + (c * remaining) ** 2)
                field_energy = (
                    2.0
                    * photon_energy
                    * rest2
                    / ((energy + c * momentum) * (after_energy + energy - photon_energy))
                )
                field_momentum = jnp.zeros_like(u)
            case "energy":
                after_energy = energy - photon_energy
                feasible = moving & (after_energy >= self.rest_energy)
                safe = jnp.maximum(after_energy, self.rest_energy)
                remaining = jnp.sqrt(jnp.maximum(safe**2 - rest2, 0.0)) / c
                field_energy = jnp.zeros_like(gamma)
                defect = (
                    -rest2 / (safe + c * remaining) + rest2 / (energy + c * momentum)
                ) / c
                field_momentum = defect[:, None] * direction
            case _:
                assert_never(self.conservation)
        recoiled = (remaining / mass)[:, None] * direction
        return (
            jnp.where(feasible[:, None], recoiled, u),
            photon,
            photon_energy,
            jnp.where(feasible, field_energy, 0.0),
            jnp.where(feasible[:, None], field_momentum, 0.0),
        ), feasible

    def _spin_rates(
        self,
        u: Array,
        spin: Array,
        electric: Array,
        magnetic: Array,
        gamma: Array,
        chi: Array,
        variation_time: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        """Mean rate ``W₀``, half difference ``W₁``, axis ``ê`` and ``P = S·ê``."""
        axis = self.spin_axis(u, electric, magnetic)
        projection = jnp.sum(spin * axis, axis=-1)
        positive, negative = self._spin_weights(chi, gamma, variation_time)
        upper = self.rate_scale * positive / gamma
        lower = self.rate_scale * negative / gamma
        return 0.5 * (upper + lower), 0.5 * (upper - lower), axis, projection

    def _channel(
        self,
        u: Array,
        electric: Array,
        magnetic: Array,
        chi: Array,
        fraction: Array,
        projection: Array | None,
        uniform: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        """Sampled final spin ``P_f``, photon Stokes ``τ`` and photon axis ``F̂⊥``."""
        axis = _unit(self.transverse_force(u, electric, magnetic))
        if projection is None:
            stokes = self.photon_stokes_parameter(chi, fraction)
            polarization = jnp.where(uniform < 0.5 * (1.0 + stokes), 1.0, -1.0)
            return jnp.zeros_like(chi), polarization, axis
        weights = self.channel_spectrum(chi, fraction, projection).reshape(-1, 4)
        index = _pick(weights, uniform)
        final = jnp.where(index < 2, 1.0, -1.0)
        polarization = jnp.where(index % 2 == 0, 1.0, -1.0)
        return final, polarization, axis

    def apply(
        self,
        proper_velocity: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        step_size: ArrayLike,
        active: ArrayLike,
        optical_depth: ArrayLike,
        uniforms: ArrayLike,
        /,
        *,
        variation_time: ArrayLike | None = None,
        grid_cutoff_frequency: ArrayLike | None = None,
        spin: ArrayLike | None = None,
    ) -> NonlinearComptonResult:
        """One emission step of ``u[N, 3]`` in fields ``E, B[N, 3]``.

        ``optical_depth[N]`` is each lepton's remaining optical depth and
        ``uniforms[N, S, uniform_count]`` its random numbers (`uniforms`);
        ``variation_time[N]`` is the local field-variation time (``inf`` for
        constant fields, the default); ``grid_cutoff_frequency`` enables the
        scale-separation check against a field grid. The spin model requires
        the polarization vectors ``spin[N, 3]`` (``|S| ≤ 1``).
        """
        u0 = jnp.asarray(proper_velocity, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        mask = jnp.asarray(active, dtype=jnp.bool_)
        depth0 = jnp.asarray(optical_depth, dtype=jnp.float64)
        draws = jnp.asarray(uniforms, dtype=jnp.float64)
        count = u0.shape[0]
        slots = self.maximum_subcycles
        width = self.uniform_count
        resolves_spin = self.resolves_spin
        resolves_photons = self.resolves_photon_polarization
        # Carried polarization arrays: (spin, stokes, axes, projections) for the
        # spin model, (stokes, axes) for photon polarization only.
        offset = 1 if resolves_spin else 0
        if u0.ndim != 2 or u0.shape[1] != 3 or e.shape != u0.shape or b.shape != u0.shape:
            raise ValueError("Proper velocities and fields must have shape [N, 3].")
        if mask.shape != (count,) or depth0.shape != (count,):
            raise ValueError("active and optical_depth must have shape [N].")
        if draws.shape != (count, slots, width):
            raise ValueError(f"uniforms must have shape [N, maximum_subcycles, {width}].")
        if resolves_spin != (spin is not None):
            raise ValueError("spin is required by, and only by, the spin model.")
        spin0 = jnp.zeros_like(u0) if spin is None else jnp.asarray(spin, jnp.float64)
        if spin0.shape != u0.shape:
            raise ValueError("spin must have shape [N, 3].")
        tau = (
            jnp.full((count,), jnp.inf)
            if variation_time is None
            else jnp.asarray(variation_time, dtype=jnp.float64)
        )
        if tau.shape != (count,):
            raise ValueError("variation_time must have shape [N].")
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        gamma0, chi0 = self.quantum_parameter(u0, e, b)
        if resolves_spin:
            mean0, half0, _, projection0 = self._spin_rates(
                u0, spin0, e, b, gamma0, chi0, tau
            )
            rate0 = mean0 + projection0 * half0
        else:
            weight0, _, _, _ = self._spectral_weights(self.table, chi0, gamma0, tau)
            rate0 = self.rate_scale * weight0 / gamma0
        applied = mask & (gamma0 >= self.minimum_gamma)
        expected = rate0 * dt
        subcycles = jnp.clip(
            jnp.ceil(expected / self.maximum_event_probability), 1, slots
        ).astype(jnp.int32)
        subcycles = jnp.where(applied, subcycles, 0)
        step = dt / jnp.maximum(subcycles, 1)

        def subcycle(
            index: int, carry: tuple[tuple[Array, ...], tuple[Array, ...]]
        ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
            (
                (
                    u,
                    depth,
                    largest,
                    field_energy,
                    field_momentum,
                    emitted,
                    photons,
                    energies,
                    fractions,
                    ratios,
                    flags,
                ),
                polarized,
            ) = carry
            live = applied & (index < subcycles)
            gamma, chi = self.quantum_parameter(u, e, b)
            draw = draws[:, index]
            if resolves_spin:
                current = polarized[0]
                mean, half, spin_axis, projection = self._spin_rates(
                    u, current, e, b, gamma, chi, tau
                )
                survivor, logarithm = _survive(current, spin_axis, half * step)
                rate = mean + projection * half
                probability = mean * step - logarithm
                fraction, infrared = self._sample_spin(
                    chi, gamma, tau, projection, draw[:, 0]
                )
            else:
                weight, _, _, _ = self._spectral_weights(self.table, chi, gamma, tau)
                rate = self.rate_scale * weight / gamma
                probability = rate * step
                fraction, infrared = self._sample(self.table, chi, gamma, tau, draw[:, 0])
            largest = jnp.maximum(largest, jnp.where(live, probability, 0.0))
            remaining = depth - probability
            crossed = live & (remaining <= 0.0) & (rate > 0.0)
            (
                (recoiled, photon, photon_energy, energy_defect, momentum_defect),
                feasible,
            ) = self._recoil(u, gamma, fraction)
            event = crossed & feasible
            ratio = jnp.where(
                jnp.isfinite(tau),
                _formation_time(
                    chi,
                    gamma,
                    (1.0 - fraction) / jnp.maximum(fraction, jnp.finfo(jnp.float64).tiny),
                    self.compton_time,
                )
                / jnp.where(jnp.isfinite(tau), tau, 1.0),
                0.0,
            )
            event_flag = jnp.zeros((count,), dtype=jnp.int32)
            for condition, flag in (
                (crossed & ~feasible, QEDEventFlag.KINEMATICS_REFUSED),
                (event & (ratio > 1.0), QEDEventFlag.OUTSIDE_LCFA_VALIDITY),
                (event & infrared, QEDEventFlag.INFRARED_CORRECTED),
            ):
                event_flag = jnp.where(condition, event_flag | int(flag), event_flag)
            fresh = -jnp.log1p(-draw[:, 1])
            if resolves_photons:
                # Infrared-corrected emissions hold the spectrum, and its channel
                # ratios, at the threshold fraction.
                threshold = self._spectral_weights(self.table, chi, gamma, tau)[1]
                evaluated = jnp.where(infrared, threshold, fraction)
                final, stokes, photon_axis = self._channel(
                    u,
                    e,
                    b,
                    chi,
                    evaluated,
                    projection if resolves_spin else None,
                    draw[:, 2],
                )
                updated = (
                    polarized[offset].at[:, index].set(jnp.where(event, stokes, 0.0)),
                    polarized[offset + 1]
                    .at[:, index]
                    .set(jnp.where(event[:, None], photon_axis, 0.0)),
                )
                if resolves_spin:
                    updated = (
                        jnp.where(
                            event[:, None],
                            final[:, None] * spin_axis,
                            jnp.where(live[:, None], survivor, polarized[0]),
                        ),
                        *updated,
                        polarized[3]
                        .at[:, index]
                        .set(
                            jnp.where(
                                event[:, None],
                                jnp.stack((projection, final), axis=-1),
                                0.0,
                            )
                        ),
                    )
                polarized = updated
            return (
                (
                    jnp.where(event[:, None], recoiled, u),
                    # Renewal: the overshoot −remaining carries into the next target.
                    jnp.where(
                        crossed, fresh + remaining, jnp.where(live, remaining, depth)
                    ),
                    largest,
                    field_energy + jnp.where(event, energy_defect, 0.0),
                    field_momentum + jnp.where(event[:, None], momentum_defect, 0.0),
                    emitted.at[:, index].set(event),
                    photons.at[:, index].set(jnp.where(event[:, None], photon, 0.0)),
                    energies.at[:, index].set(jnp.where(event, photon_energy, 0.0)),
                    fractions.at[:, index].set(jnp.where(event, fraction, 0.0)),
                    ratios.at[:, index].set(jnp.where(event, ratio, 0.0)),
                    flags.at[:, index].set(event_flag),
                ),
                polarized,
            )

        zeros = jnp.zeros((count,))
        slot_zeros = jnp.zeros((count, slots))
        polarized0: tuple[Array, ...] = ()
        if resolves_photons:
            polarized0 = (slot_zeros, jnp.zeros((count, slots, 3)))
            if resolves_spin:
                polarized0 = (spin0, *polarized0, jnp.zeros((count, slots, 2)))
        (
            (
                u,
                depth,
                largest,
                field_energy,
                field_momentum,
                emitted,
                photons,
                energies,
                fractions,
                ratios,
                event_flags,
            ),
            polarized,
        ) = jax.lax.fori_loop(
            0,
            slots,
            subcycle,
            (
                (
                    u0,
                    depth0,
                    zeros,
                    zeros,
                    jnp.zeros_like(u0),
                    jnp.zeros((count, slots), dtype=jnp.bool_),
                    jnp.zeros((count, slots, 3)),
                    slot_zeros,
                    slot_zeros,
                    slot_zeros,
                    jnp.zeros((count, slots), dtype=jnp.int32),
                ),
                polarized0,
            ),
        )
        frequency = self.critical_frequency_scale * chi0 * gamma0
        if grid_cutoff_frequency is None:
            separation = jnp.full_like(gamma0, jnp.inf)
        else:
            separation = frequency / jnp.asarray(
                grid_cutoff_frequency, dtype=jnp.float64
            ).reshape(())
        emitting = applied & (chi0 > 0.0)
        unseparated = emitting & (separation < self.minimum_scale_separation)
        finite = (
            jnp.all(jnp.isfinite(u), axis=-1)
            & jnp.isfinite(chi0)
            & jnp.isfinite(rate0)
            & jnp.isfinite(field_energy)
            & jnp.all(jnp.isfinite(photons), axis=(-2, -1))
        )
        if resolves_spin:
            finite = finite & jnp.all(jnp.isfinite(polarized[0]), axis=-1)
        trident = _formation_time(
            chi0, gamma0, jnp.ones_like(chi0), self.compton_time
        ) >= jnp.where(jnp.isfinite(tau), tau, jnp.inf)
        flags = jnp.zeros((count,), dtype=jnp.int32)
        for condition, flag in (
            (mask & ~applied, QEDEventFlag.BELOW_THRESHOLD),
            (applied & (chi0 > self.maximum_chi), QEDEventFlag.CHI_EXCEEDED),
            (
                applied & (expected / slots > self.maximum_event_probability),
                QEDEventFlag.EVENT_PROBABILITY_EXCEEDED,
            ),
            (unseparated, QEDEventFlag.SCALE_UNSEPARATED),
            (mask & ~finite, QEDEventFlag.NONFINITE),
            (emitting & trident, QEDEventFlag.ONE_STEP_TRIDENT),
        ):
            flags = jnp.where(condition, flags | int(flag), flags)
        flags = flags | jax.lax.reduce(
            event_flags, jnp.int32(0), jax.lax.bitwise_or, (1,)
        )
        supported = ~mask | ((flags & int(QED_SUPPORT_FAILURE)) == 0)
        return NonlinearComptonResult(
            jnp.where(applied[:, None], u, u0),
            jnp.where(applied, depth, depth0),
            chi0,
            gamma0,
            rate0,
            frequency,
            separation,
            subcycles,
            largest,
            field_energy,
            field_momentum,
            flags,
            emitted,
            photons,
            energies,
            fractions,
            ratios,
            event_flags,
            applied,
            supported,
            ~unseparated,
            jnp.all(supported),
            jnp.where(applied[:, None], polarized[0], spin0) if resolves_spin else None,
            polarized[offset] if resolves_photons else None,
            polarized[offset + 1] if resolves_photons else None,
            polarized[3] if resolves_spin else None,
        )


def _validated_spin_tables(
    polarization: QEDPolarizationModel,
    tables: tuple[QEDTable, QEDTable] | None,
    maximum_chi: float,
    /,
) -> tuple[QEDTable, QEDTable] | None:
    match polarization:
        case "unpolarized" | "photon-polarized":
            if tables is not None:
                raise ValueError(
                    "spin_tables are used only by the 'spin-and-photon-polarized' model."
                )
            return None
        case "spin-and-photon-polarized":
            if tables is None:
                raise ValueError("The spin model requires spin_tables.")
            positive, negative = tables
            if not all(isinstance(value, QEDTable) for value in (positive, negative)):
                raise TypeError("spin_tables must hold two QEDTable instances.")
            if (
                positive.process != "nonlinear-compton"
                or negative.process != "nonlinear-compton"
                or positive.polarization != "positive"
                or negative.polarization != "negative"
            ):
                raise ValueError(
                    "spin_tables must be the 'positive' and 'negative' "
                    "'nonlinear-compton' tables."
                )
            if min(positive.maximum_chi, negative.maximum_chi) < maximum_chi:
                raise ValueError("spin_tables must cover maximum_chi.")
            return positive, negative
        case _:
            assert_never(polarization)


def _identity_words(high: ArrayLike, low: ArrayLike, /) -> tuple[Array, Array]:
    high_ = jnp.asarray(high)
    low_ = jnp.asarray(low)
    if high_.dtype != jnp.uint32 or low_.dtype != jnp.uint32:
        raise TypeError("Identity words must be uint32.")
    if high_.shape != low_.shape or high_.ndim != 1:
        raise ValueError("Identity words must be matching one-dimensional arrays.")
    return high_, low_


__all__ = [
    "NonlinearComptonPlan",
    "NonlinearComptonResult",
    "QEDEmissionModel",
]
