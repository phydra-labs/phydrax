#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Monte Carlo nonlinear Breit–Wheeler pair creation in the locally constant field.

`NonlinearBreitWheelerPlan` decays photons into electron–positron pairs of one
lepton species (charge magnitude ``e``, mass ``m``) in the units of one
`ElectromagneticScaleContract`. A photon of momentum ``k`` (energy
``ε_γ = c|k|``, direction ``k̂``) has the quantum parameter
``χ_γ = (ε_γ/mc²) √((E + c k̂×B)² − (k̂·E)²) / E_S`` and decays at

    dW/dξ = (α m²c⁴/(ħ ε_γ)) s(χ_γ, ξ),   W = (α m²c⁴/(ħ ε_γ)) R(χ_γ),

with the electron energy fraction ``ξ`` and ``s``, ``R = χ_γ T(χ_γ)`` from a
Breit–Wheeler `QEDTable`. Photons below the threshold ``ε_γ = 2mc²`` do not
decay. Decay uses the optical-depth method: the photon's optical depth
decreases by ``W Δt`` and the photon decays when it reaches zero. The rate is
constant over a step (photon energy and direction are fixed and the fields are
frozen), so the decrement is exact; ``maximum_event_probability`` bounds
``W Δt`` so that the frozen fields remain representative, and a photon beyond
it is a support failure. Pairs are collinear with ``k̂``. The declared
``conservation`` fixes the split: ``"momentum"`` gives ``p₋ = ξk`` and
``p₊ = (1 − ξ)k`` exactly and hands the energy defect
``ε₋ + ε₊ − ε_γ = m²c⁴/(ε₋ + ξε_γ) + m²c⁴/(ε₊ + (1 − ξ)ε_γ) = O(m²c⁴/ε_γ)``
to the field; ``"energy"`` gives ``ε₋ = ξε_γ`` and ``ε₊ = (1 − ξ)ε_γ`` exactly
and hands the momentum defect along ``k̂`` to the field (a draw leaving a
product below ``mc²`` is refused and counted). The formation time
``τ_C (8γ_γ/χ_γ) sinh(arsinh((3/4) χ_γ ξ(1 − ξ))/3)`` is the crossing-symmetric
continuation of the Compton formation time; its ratio to the local
field-variation time along the photon path is the per-event LCFA validity
evidence. Random numbers are addressed by the step key and the photon's
persistent identity.

``polarization`` (`QEDPolarizationModel`) resolves polarizations with the
Seipt–King LCFA rates (PRA 102, 052805, 2020). A polarized photon carries the
linear Stokes parameters ``(Q, U)`` relative to its polarization axis ``e₁``
(``⊥ k̂``); each step they are rotated to the local basis
``e₁' = (E + c k̂×B)⊥/|…|``, where ``τ = Q'`` sets the rate
``W(τ) = ((1 + τ)/2) W_∥ + ((1 − τ)/2) W_⊥`` from the ``"positive"``
(``∥``) and ``"negative"`` (``⊥``) Breit–Wheeler tables (``W_⊥ → 2W_∥`` as
``χ_γ → 0``). A surviving photon's Stokes vector follows the no-decay
evolution (vacuum dichroism) ``Q ← (Q − t)/(1 − Qt)``,
``U ← U/(cosh(W₁Δt) − Q sinh(W₁Δt))`` with ``t = tanh(W₁Δt)``,
``W₁ = (W_∥ − W_⊥)/2``, and the optical depth decreases by the exact survival
exponent ``W₀Δt − log(cosh(W₁Δt) − Q sinh(W₁Δt))``. The electron fraction is
sampled from the mixture; with ``"spin-and-photon-polarized"`` the pair's
spin projections ``P_± = ±1`` on their quantization axes
``ê_± = ±(k̂ × e₁')`` (``v̂ × F̂`` of each lepton) are then sampled jointly from
the four Seipt–King channels and the leptons are created in those states.
"""

from __future__ import annotations

import math
from typing import assert_never

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import ElectromagneticScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float
from ...special import synchrotron_f, synchrotron_g, synchrotron_h
from ...typing import checked, parse, PRNGKey
from ._nonlinear_compton import (
    _formation_time,
    _identity_words,
    _pick,
    _SIGNS,
    _survive,
    _unit,
)
from ._qed_tables import (
    _INVERSE_NORMALIZATION,
    QED_SUPPORT_FAILURE,
    QEDConservation,
    QEDEventFlag,
    QEDPolarizationModel,
    QEDTable,
)


_EVENT_ADDRESS = SampleAddress("phydrax.qed", "nonlinear-breit-wheeler", role="event")
_DEPTH_ADDRESS = SampleAddress(
    "phydrax.qed", "nonlinear-breit-wheeler", role="optical-depth"
)


def _rotate_stokes(
    stokes: Array, axis: Array, target: Array, direction: Array, /
) -> Array:
    """Linear Stokes ``(Q, U)`` relative to unit ``axis`` re-expressed along ``target``.

    Both axes are unit vectors perpendicular to ``direction`` (the photon
    momentum); ``U`` is positive along ``(e₁ + k̂×e₁)/√2``. A rotation by ``ψ``
    about ``k̂`` maps ``Q' = Q cos 2ψ + U sin 2ψ``, ``U' = U cos 2ψ − Q sin 2ψ``.
    """
    cosine = jnp.sum(axis * target, axis=-1)
    sine = jnp.sum(jnp.cross(axis, target, axis=-1) * _unit(direction), axis=-1)
    double_cosine = cosine**2 - sine**2
    double_sine = 2.0 * cosine * sine
    return jnp.stack(
        (
            stokes[..., 0] * double_cosine + stokes[..., 1] * double_sine,
            stokes[..., 1] * double_cosine - stokes[..., 0] * double_sine,
        ),
        axis=-1,
    )


class NonlinearBreitWheelerResult(StrictModule):
    """One pair-creation step of ``M`` photons.

    ``decayed`` photons create an electron with ``electron_momentum`` and a
    positron with ``positron_momentum`` (per physical particle) carrying the
    energy fraction ``electron_fraction``/``1 − electron_fraction``.
    ``optical_depth`` is the candidate remaining optical depth (after a refused
    draw, a fresh draw minus the overshoot, as for a renewal process).
    ``field_energy``/``field_momentum`` are what the field
    supplied to close the declared kinematics (per physical pair).
    ``formation_ratio`` is the formation over the field-variation time (zero
    in constant fields). ``supported`` excludes support failures of active
    photons; ``successful`` is ``all(supported)``.

    Polarized models report every photon's candidate linear Stokes parameters
    ``stokes[M, 2]`` relative to ``polarization_axis[M, 3]`` (the local basis
    after the no-decay evolution); the spin model reports the created
    ``electron_spin``/``positron_spin`` vectors. Unresolved fields are ``None``.
    """

    decayed: Array
    electron_momentum: Array
    positron_momentum: Array
    electron_fraction: Array
    optical_depth: Array
    quantum_parameter: Array
    rate: Array
    event_probability: Array
    formation_ratio: Array
    field_energy: Array
    field_momentum: Array
    flags: Array
    supported: Array
    successful: Array
    stokes: Array | None
    polarization_axis: Array | None
    electron_spin: Array | None
    positron_spin: Array | None


class NonlinearBreitWheelerPlan(StrictModule, NonTrainableState):
    """Pair creation by photons into one lepton species in one electromagnetic scale.

    ``lepton_charge`` (its magnitude is used) and ``lepton_mass`` are the
    single-particle charge and mass of the created leptons in ``scale`` units;
    ``table`` is a ``"nonlinear-breit-wheeler"`` `QEDTable` covering
    ``maximum_chi``. Polarized models require ``polarized_tables``, the
    ``"positive"`` (``∥``) and ``"negative"`` (``⊥``) Breit–Wheeler tables.
    """

    table: QEDTable
    polarized_tables: tuple[QEDTable, QEDTable] | None
    conservation: QEDConservation = eqx.field(static=True)
    polarization: QEDPolarizationModel = eqx.field(static=True)
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    lepton_charge: float = eqx.field(static=True)
    lepton_mass: float = eqx.field(static=True)
    maximum_chi: float = eqx.field(static=True)
    maximum_event_probability: float = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    rest_energy: float = eqx.field(static=True)
    critical_field: float = eqx.field(static=True)
    compton_time: float = eqx.field(static=True)
    rate_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        lepton_charge: float,
        lepton_mass: float,
        table: QEDTable,
        /,
        *,
        maximum_chi: float,
        maximum_event_probability: float = 0.1,
        conservation: QEDConservation = "momentum",
        polarization: QEDPolarizationModel = "unpolarized",
        polarized_tables: tuple[QEDTable, QEDTable] | None = None,
    ) -> None:
        conservation_ = parse(conservation, QEDConservation, "conservation")
        polarization_ = parse(polarization, QEDPolarizationModel, "polarization")
        if table.process != "nonlinear-breit-wheeler":
            raise ValueError(
                "NonlinearBreitWheelerPlan requires a 'nonlinear-breit-wheeler' table."
            )
        if table.polarization != "averaged":
            raise ValueError("table must be the 'averaged' Breit–Wheeler table.")
        charge = abs(float(lepton_charge))
        if not math.isfinite(charge) or charge == 0.0:
            raise ValueError("lepton_charge must be finite and nonzero.")
        mass = positive_finite_float(lepton_mass, "lepton_mass")
        chi = positive_finite_float(maximum_chi, "maximum_chi")
        if table.maximum_chi < chi:
            raise ValueError("table must cover maximum_chi.")
        polarized = _validated_polarized_tables(polarization_, polarized_tables, chi)
        probability = positive_finite_float(
            maximum_event_probability, "maximum_event_probability"
        )
        if probability >= 1.0:
            raise ValueError("maximum_event_probability must be below one.")
        light = float(scale.speed_of_light)
        hbar = float(scale.reduced_planck_constant)
        permittivity = float(scale.vacuum_permittivity)
        rest = mass * light**2
        alpha = charge**2 / (4.0 * math.pi * permittivity * hbar * light)
        self.table = table
        self.polarized_tables = polarized
        self.conservation = conservation_
        self.polarization = polarization_
        self.scale = scale
        self.lepton_charge = charge
        self.lepton_mass = mass
        self.maximum_chi = chi
        self.maximum_event_probability = probability
        self.speed_of_light = light
        self.rest_energy = rest
        self.critical_field = mass**2 * light**3 / (charge * hbar)
        self.compton_time = hbar / rest
        # W = (α mc²/ħ) (mc²/ε_γ) R(χ_γ).
        self.rate_scale = alpha * rest**2 / hbar
        identity: dict[str, object] = {
            "kind": "nonlinear-breit-wheeler-plan",
            "conservation": conservation_,
            "scale": scale.scale_id,
            "lepton_charge": charge,
            "lepton_mass": mass,
            "table": table.table_id,
            "maximum_chi": chi,
            "maximum_event_probability": probability,
        }
        # Unpolarized plans keep the identity of the unpolarized model.
        if polarized is not None:
            identity["polarization"] = polarization_
            identity["polarized_tables"] = [value.table_id for value in polarized]
        self.plan_id = canonical_fingerprint(identity)

    def quantum_parameter(
        self, momentum: ArrayLike, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Photon energy ``ε_γ`` and ``χ_γ`` of photon momenta ``k[M, 3]`` in ``E, B``."""
        k = jnp.asarray(momentum, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        c = self.speed_of_light
        norm = jnp.sqrt(jnp.sum(k**2, axis=-1))
        direction = jnp.where(
            (norm > 0.0)[..., None], k / jnp.where(norm > 0.0, norm, 1.0)[..., None], 0.0
        )
        effective = e + c * jnp.cross(direction, b, axis=-1)
        along = jnp.sum(direction * e, axis=-1)
        invariant = jnp.maximum(jnp.sum(effective**2, axis=-1) - along**2, 0.0)
        energy = c * norm
        return energy, energy / self.rest_energy * jnp.sqrt(
            invariant
        ) / self.critical_field

    def transverse_field(
        self, momentum: ArrayLike, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> Array:
        """``E + c k̂×B`` without its component along ``k̂`` (enters ``χ_γ``)."""
        k = jnp.asarray(momentum, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        norm = jnp.sqrt(jnp.sum(k**2, axis=-1, keepdims=True))
        direction = jnp.where(norm > 0.0, k / jnp.where(norm > 0.0, norm, 1.0), 0.0)
        effective = e + self.speed_of_light * jnp.cross(direction, b, axis=-1)
        return (
            effective - jnp.sum(effective * direction, axis=-1, keepdims=True) * direction
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
        """Uniforms per photon: fraction, optical depth (and pair spin channel)."""
        return 3 if self.resolves_spin else 2

    def _require_polarized_tables(self) -> tuple[QEDTable, QEDTable]:
        if self.polarized_tables is None:
            raise ValueError(
                "This NonlinearBreitWheelerPlan does not resolve photon polarization."
            )
        return self.polarized_tables

    def local_stokes(
        self,
        momentum: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        stokes: ArrayLike,
        polarization_axis: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Linear Stokes ``(Q, U)[M, 2]`` rotated from ``polarization_axis`` to the
        local axis ``e₁' = (E + c k̂×B)⊥/|…|``, and that axis.

        A basis rotation by ``ψ`` about ``k̂`` maps ``Q' = Q cos 2ψ + U sin 2ψ``,
        ``U' = U cos 2ψ − Q sin 2ψ``. Where the transverse field vanishes the
        stored axis is kept.
        """
        k = jnp.asarray(momentum, dtype=jnp.float64)
        axis = jnp.asarray(polarization_axis, dtype=jnp.float64)
        local = _unit(self.transverse_field(k, electric, magnetic))
        local = jnp.where(jnp.any(local != 0.0, axis=-1, keepdims=True), local, axis)
        return (
            _rotate_stokes(jnp.asarray(stokes, dtype=jnp.float64), axis, local, k),
            local,
        )

    def channel_spectrum(
        self, chi: ArrayLike, fraction: ArrayLike, stokes_parameter: ArrayLike, /
    ) -> Array:
        """Seipt–King pair channels ``s(χ_γ, ξ; τ → P₊, P₋)`` with shape ``[..., 2, 2]``.

        ``fraction`` is the electron energy fraction ``ξ`` and
        ``stokes_parameter`` the photon's ``τ`` along the local field axis; the
        last two axes are the positron and electron spin projections
        ``P₊, P₋ = (+1, −1)`` on their quantization axes ``ê_± = ±(k̂ × e₁')``.
        Their sum is ``s(χ_γ, ξ) − τ G(δ)/(√3πδ)``.
        """
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        xi = jnp.asarray(fraction, dtype=jnp.float64)
        tau_ = jnp.asarray(stokes_parameter, dtype=jnp.float64)
        chi_, xi, tau_ = jnp.broadcast_arrays(chi_, xi, tau_)
        interior = (xi > 0.0) & (xi < 1.0) & (chi_ > 0.0)
        electron = jnp.where(interior, xi, 0.5)[..., None, None]
        safe_chi = jnp.where(interior, chi_, 1.0)[..., None, None]
        tau = tau_[..., None, None]
        positron = 1.0 - electron
        up = jnp.asarray(_SIGNS)[:, None]
        down = jnp.asarray(_SIGNS)[None, :]
        product = electron * positron
        delta = 2.0 / (3.0 * safe_chi * product)
        f = synchrotron_f(delta)
        g = synchrotron_g(delta)
        h = synchrotron_h(delta)
        kibble = 1.0 - 0.5 / product
        spins = up * down
        a = 1.0 - spins - tau * spins * (1.0 - kibble)
        b = -up / positron - down / electron + tau * (down / positron + up / electron)
        c = kibble - spins + 0.5 * tau * (1.0 - kibble * spins)
        # δ·[A∫K_{1/3} + B K_{1/3} − 2C K_{2/3}] with ∫K_{1/3} = (2G − F)/δ.
        value = (a * (2.0 * g - f) + b * h - 2.0 * c * g) / delta
        return jnp.where(
            interior[..., None, None], 0.25 * _INVERSE_NORMALIZATION * value, 0.0
        )

    # -- rates -------------------------------------------------------------------

    def _polarized_rates(self, energy: Array, chi: Array, /) -> tuple[Array, Array]:
        """Mean rate ``W₀`` and half difference ``W₁`` of ``∥`` and ``⊥`` photons."""
        parallel, perpendicular = self._require_polarized_tables()
        above = energy >= 2.0 * self.rest_energy
        scale = jnp.where(above, self.rate_scale / jnp.where(above, energy, 1.0), 0.0)
        upper = scale * parallel.rate_function(chi)
        lower = scale * perpendicular.rate_function(chi)
        return 0.5 * (upper + lower), 0.5 * (upper - lower)

    def rate(
        self,
        momentum: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        /,
        *,
        stokes: ArrayLike | None = None,
        polarization_axis: ArrayLike | None = None,
    ) -> Array:
        """Pair-creation rate ``W`` of each photon (zero below ``2mc²``).

        Without ``stokes`` the polarization-averaged rate; with linear Stokes
        parameters relative to ``polarization_axis`` (polarized models) the
        rate of those photons.
        """
        energy, chi = self.quantum_parameter(momentum, electric, magnetic)
        if stokes is None or polarization_axis is None:
            if stokes is not None or polarization_axis is not None:
                raise ValueError("stokes and polarization_axis go together.")
            above = energy >= 2.0 * self.rest_energy
            return jnp.where(
                above,
                self.rate_scale
                * self.table.rate_function(chi)
                / jnp.where(above, energy, 1.0),
                0.0,
            )
        local, _ = self.local_stokes(
            momentum, electric, magnetic, stokes, polarization_axis
        )
        mean, half = self._polarized_rates(energy, chi)
        return mean + local[..., 0] * half

    def uniforms(
        self, key: PRNGKey, identity_high: ArrayLike, identity_low: ArrayLike, /
    ) -> Array:
        """``[M, uniform_count]`` uniforms in ``[0, 1)`` addressed by photon identity.

        The first samples the electron fraction, the second the optical depth
        redrawn after a refused draw, the third (spin model) the pair spins.
        """
        high, low = _identity_words(identity_high, identity_low)
        width = self.uniform_count
        return jax.vmap(
            lambda hi, lo: jr.uniform(
                derive_key(key, _EVENT_ADDRESS, hi, lo), (width,), dtype=jnp.float64
            )
        )(high, low)

    def initial_optical_depth(
        self, key: PRNGKey, identity_high: ArrayLike, identity_low: ArrayLike, /
    ) -> Array:
        """Fresh optical depths ``−log(1 − U)`` addressed by photon identity."""
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

    def _split(
        self, momentum: Array, energy: Array, fraction: Array, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Electron and positron momenta, field energy and momentum, feasibility."""
        c = self.speed_of_light
        rest = self.rest_energy
        rest2 = rest**2
        direction = momentum / jnp.where(energy > 0.0, energy / c, 1.0)[:, None]
        electron_energy = fraction * energy
        positron_energy = energy - electron_energy
        match self.conservation:
            case "momentum":
                electron = fraction[:, None] * momentum
                positron = momentum - electron
                electron_total = jnp.sqrt(rest2 + electron_energy**2)
                positron_total = jnp.sqrt(rest2 + positron_energy**2)
                field_energy = rest2 / (electron_total + electron_energy) + rest2 / (
                    positron_total + positron_energy
                )
                field_momentum = jnp.zeros_like(momentum)
                feasible = energy > 0.0
            case "energy":
                feasible = (electron_energy >= rest) & (positron_energy >= rest)
                safe_electron = jnp.maximum(electron_energy, rest)
                safe_positron = jnp.maximum(positron_energy, rest)
                electron_size = jnp.sqrt(safe_electron**2 - rest2) / c
                positron_size = jnp.sqrt(safe_positron**2 - rest2) / c
                electron = electron_size[:, None] * direction
                positron = positron_size[:, None] * direction
                defect = (
                    -(
                        rest2 / (safe_electron + c * electron_size)
                        + rest2 / (safe_positron + c * positron_size)
                    )
                    / c
                )
                field_energy = jnp.zeros_like(energy)
                field_momentum = defect[:, None] * direction
            case _:
                assert_never(self.conservation)
        return electron, positron, field_energy, field_momentum, feasible

    def apply(
        self,
        momentum: ArrayLike,
        electric: ArrayLike,
        magnetic: ArrayLike,
        step_size: ArrayLike,
        active: ArrayLike,
        optical_depth: ArrayLike,
        uniforms: ArrayLike,
        /,
        *,
        variation_time: ArrayLike | None = None,
        stokes: ArrayLike | None = None,
        polarization_axis: ArrayLike | None = None,
    ) -> NonlinearBreitWheelerResult:
        """One pair-creation step of photon momenta ``k[M, 3]`` in fields ``E, B[M, 3]``.

        ``optical_depth[M]`` is each photon's remaining optical depth,
        ``uniforms[M, uniform_count]`` its random numbers (`uniforms`), and
        ``variation_time[M]`` the local field-variation time along its path
        (``inf`` by default). Polarized models require the photons' linear
        Stokes parameters ``stokes[M, 2]`` relative to ``polarization_axis[M, 3]``.
        """
        k = jnp.asarray(momentum, dtype=jnp.float64)
        e = jnp.asarray(electric, dtype=jnp.float64)
        b = jnp.asarray(magnetic, dtype=jnp.float64)
        mask = jnp.asarray(active, dtype=jnp.bool_)
        depth = jnp.asarray(optical_depth, dtype=jnp.float64)
        draws = jnp.asarray(uniforms, dtype=jnp.float64)
        count = k.shape[0]
        width = self.uniform_count
        polarized = self.resolves_photon_polarization
        if k.ndim != 2 or k.shape[1] != 3 or e.shape != k.shape or b.shape != k.shape:
            raise ValueError("Photon momenta and fields must have shape [M, 3].")
        if mask.shape != (count,) or depth.shape != (count,):
            raise ValueError("active and optical_depth must have shape [M].")
        if draws.shape != (count, width):
            raise ValueError(f"uniforms must have shape [M, {width}].")
        if polarized != (stokes is not None) or polarized != (
            polarization_axis is not None
        ):
            raise ValueError(
                "stokes and polarization_axis are required by, and only by, "
                "polarized models."
            )
        tau = (
            jnp.full((count,), jnp.inf)
            if variation_time is None
            else jnp.asarray(variation_time, dtype=jnp.float64)
        )
        if tau.shape != (count,):
            raise ValueError("variation_time must have shape [M].")
        dt = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
        energy, chi = self.quantum_parameter(k, e, b)
        above = mask & (energy >= 2.0 * self.rest_energy)
        if stokes is None or polarization_axis is None:
            rate = jnp.where(above, self.rate(k, e, b), 0.0)
            probability = rate * dt
            fraction = self.table.quantile(chi, draws[:, 0])
            local = survivor = axis = None
        else:
            local, axis = self.local_stokes(k, e, b, stokes, polarization_axis)
            if local.shape != (count, 2) or axis.shape != (count, 3):
                raise ValueError("stokes and polarization_axis must be [M, 2], [M, 3].")
            mean, half = self._polarized_rates(energy, chi)
            rate = jnp.where(above, mean + local[:, 0] * half, 0.0)
            survivor, logarithm = _survive(
                local, jnp.zeros_like(local).at[:, 0].set(1.0), half * dt
            )
            probability = jnp.where(above, mean * dt - logarithm, 0.0)
            fraction = self._sample_polarized(chi, local[:, 0], draws[:, 0])
        remaining = depth - probability
        crossed = above & (remaining <= 0.0) & (rate > 0.0)
        electron, positron, field_energy, field_momentum, feasible = self._split(
            k, energy, fraction
        )
        decayed = crossed & feasible
        variation = jnp.isfinite(tau)
        ratio = jnp.where(
            variation & decayed,
            _formation_time(
                chi,
                energy / self.rest_energy,
                fraction * (1.0 - fraction),
                self.compton_time,
            )
            / jnp.where(variation, tau, 1.0),
            0.0,
        )
        splitting = _formation_time(
            chi, energy / self.rest_energy, jnp.full_like(chi, 0.25), self.compton_time
        ) >= jnp.where(variation, tau, jnp.inf)
        finite = (
            jnp.isfinite(chi)
            & jnp.isfinite(rate)
            & jnp.all(jnp.isfinite(electron) & jnp.isfinite(positron), axis=-1)
        )
        flags = jnp.zeros((count,), dtype=jnp.int32)
        for condition, flag in (
            (mask & ~above, QEDEventFlag.BELOW_THRESHOLD),
            (above & (chi > self.maximum_chi), QEDEventFlag.CHI_EXCEEDED),
            (
                above & (probability > self.maximum_event_probability),
                QEDEventFlag.EVENT_PROBABILITY_EXCEEDED,
            ),
            (crossed & ~feasible, QEDEventFlag.KINEMATICS_REFUSED),
            (decayed & (ratio > 1.0), QEDEventFlag.OUTSIDE_LCFA_VALIDITY),
            (above & splitting, QEDEventFlag.PHOTON_SPLITTING),
            (mask & ~finite, QEDEventFlag.NONFINITE),
        ):
            flags = jnp.where(condition, flags | int(flag), flags)
        supported = ~mask | ((flags & int(QED_SUPPORT_FAILURE)) == 0)
        fresh = -jnp.log1p(-draws[:, 1])
        electron_spin = positron_spin = None
        if local is not None and survivor is not None and axis is not None:
            local = jnp.where((above & ~decayed)[:, None], survivor, local)
            if self.resolves_spin:
                electron_spin, positron_spin = self._pair_spins(
                    k, e, b, chi, fraction, local[:, 0], draws[:, 2], decayed
                )
        return NonlinearBreitWheelerResult(
            decayed,
            jnp.where(decayed[:, None], electron, 0.0),
            jnp.where(decayed[:, None], positron, 0.0),
            jnp.where(decayed, fraction, 0.0),
            jnp.where(crossed, fresh + remaining, jnp.where(above, remaining, depth)),
            chi,
            rate,
            probability,
            ratio,
            jnp.where(decayed, field_energy, 0.0),
            jnp.where(decayed[:, None], field_momentum, 0.0),
            flags,
            supported,
            jnp.all(supported),
            local,
            axis,
            electron_spin,
            positron_spin,
        )

    def _sample_polarized(
        self, chi: Array, stokes_parameter: Array, uniform: Array, /
    ) -> Array:
        """Electron fraction from the mixture ``((1 ± τ)/2) s_{∥,⊥}``."""
        parallel, perpendicular = self._require_polarized_tables()
        upper = 0.5 * (1.0 + stokes_parameter) * parallel.rate_function(chi)
        total = upper + 0.5 * (1.0 - stokes_parameter) * perpendicular.rate_function(chi)
        share = jnp.where(total > 0.0, upper / jnp.where(total > 0.0, total, 1.0), 1.0)
        chosen = uniform < share
        rescaled = jnp.clip(
            jnp.where(
                chosen,
                uniform / jnp.where(share > 0.0, share, 1.0),
                (uniform - share) / jnp.where(share < 1.0, 1.0 - share, 1.0),
            ),
            0.0,
            1.0,
        )
        return jnp.where(
            chosen,
            parallel.quantile(chi, rescaled),
            perpendicular.quantile(chi, rescaled),
        )

    def _pair_spins(
        self,
        momentum: Array,
        electric: Array,
        magnetic: Array,
        chi: Array,
        fraction: Array,
        stokes_parameter: Array,
        uniform: Array,
        decayed: Array,
        /,
    ) -> tuple[Array, Array]:
        """Electron and positron polarization vectors sampled from the channels."""
        weights = self.channel_spectrum(chi, fraction, stokes_parameter).reshape(-1, 4)
        index = _pick(weights, uniform)
        positron = jnp.where(index < 2, 1.0, -1.0)
        electron = jnp.where(index % 2 == 0, 1.0, -1.0)
        axis = _unit(
            jnp.cross(
                _unit(momentum),
                self.transverse_field(momentum, electric, magnetic),
                axis=-1,
            )
        )
        return (
            jnp.where(decayed[:, None], -electron[:, None] * axis, 0.0),
            jnp.where(decayed[:, None], positron[:, None] * axis, 0.0),
        )


def _validated_polarized_tables(
    polarization: QEDPolarizationModel,
    tables: tuple[QEDTable, QEDTable] | None,
    maximum_chi: float,
    /,
) -> tuple[QEDTable, QEDTable] | None:
    match polarization:
        case "unpolarized":
            if tables is not None:
                raise ValueError("polarized_tables are used only by polarized models.")
            return None
        case "photon-polarized" | "spin-and-photon-polarized":
            if tables is None:
                raise ValueError("Polarized models require polarized_tables.")
            parallel, perpendicular = tables
            if not all(
                isinstance(value, QEDTable) for value in (parallel, perpendicular)
            ):
                raise TypeError("polarized_tables must hold two QEDTable instances.")
            if (
                parallel.process != "nonlinear-breit-wheeler"
                or perpendicular.process != "nonlinear-breit-wheeler"
                or parallel.polarization != "positive"
                or perpendicular.polarization != "negative"
            ):
                raise ValueError(
                    "polarized_tables must be the 'positive' and 'negative' "
                    "'nonlinear-breit-wheeler' tables."
                )
            if min(parallel.maximum_chi, perpendicular.maximum_chi) < maximum_chi:
                raise ValueError("polarized_tables must cover maximum_chi.")
            return parallel, perpendicular
        case _:
            assert_never(polarization)


__all__ = ["NonlinearBreitWheelerPlan", "NonlinearBreitWheelerResult"]
