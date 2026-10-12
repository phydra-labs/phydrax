#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""TUBRNP radial power and plutonium profile of UO2 pellets under LWR conditions.

Lassmann, O'Carroll, van de Laar and Walker, "The radial distribution of
plutonium in high burnup UO2 fuels", J. Nucl. Mater. 208 (1994) 223-231, as
adopted in FRAPCON-3 (Lanning et al., NUREG/CR-6534 Vol. 1, 1997, Section 6.3,
Eqs. 6.4-6.10) and FRAPCON-4.0 (Geelhood et al., PNNL-19418 Vol. 1 Rev. 2, 2015,
Section 2.3.5.5, Eqs. 2.43-2.51, Table 2.1).

Pellet-average concentrations obey the one-group depletion chain in the flux-time
``tau`` (barn^-1)::

    dN235/dtau = -sa235 N235
    dN238/dtau = -sa238 N238
    dN239/dtau = -sa239 N239 + sc238 N238
    dNj/dtau   = -saj Nj + sc(j-1) N(j-1)          (j = Pu240, Pu241, Pu242)

and local concentrations follow Eqs. 6.7-6.10: U-235 is radially uniform, U-238
capture and Pu-239 production carry the normalized form function
``f(r) = 1 + p1 exp(-p2 ((r_out - r)/length_unit)^p3)`` (volume average one), so
``N238(r) = N238(0) - f(r) [N238(0) - N238_avg]`` and ``NPu(r) = f(r) NPu_avg``.
The burnup increment fixes the flux-time increment through Eq. 2.46,
``d bu = E_f sum_k sf_k N_k d tau / rho_HM``, solved exactly for each step.

Thermal-flux depression follows the solid-cylinder diffusion solution
``phi(r) = C I0(kappa r)`` of Eq. 2.41. The effective scattering cross section
and its data source are required model inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import i0e
from jax.typing import ArrayLike

from ... import linalg as la
from ..._dtype_names import inexact_result_type
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ...nonlinear import LocalRootPlan
from ...nuclear import FISSIONING_NUCLIDES, HeavyMetalReference, NuclideKey
from ._constants import JOULE_PER_MEGAWATT_DAY
from ._host_validation import finite_scalar, finite_vector, host_list


# PNNL-19418 Vol. 1 Rev. 2 Table 2.1, LWR one-group cross sections [barn].
_LWR_FISSION_BARN: tuple[float, ...] = (41.5, 0.0, 105.0, 0.584, 120.0, 0.458)
_LWR_CAPTURE_BARN: tuple[float, ...] = (9.7, 0.78, 58.6, 100.0, 50.0, 80.0)
_SQUARE_METER_PER_BARN = 1.0e-28
_NUCLIDE_COUNT = len(FISSIONING_NUCLIDES)
# Rows of the augmented depletion state after the nuclide concentrations:
# cumulative fissions per initial heavy-metal atom, then cumulative fission
# energy in units of the U-235 energy per fission.
_FIMA_INDEX = _NUCLIDE_COUNT
_ENERGY_INDEX = _FIMA_INDEX + 1
# PNNL-19418 Vol. 1 Rev. 2 Section 2.3.5.5 form-function defaults.
_FORM_AMPLITUDE = 3.45
_FORM_DECAY = 3.0
_FORM_EXPONENT = 0.45
# Solver controls of the per-step flux-time root (residual in flux-time units,
# barn^-1).
_FLUX_TIME_ROOT_MAXIMUM_STEPS = 40
_FLUX_TIME_ROOT_TOLERANCE = 1.0e-13
# NUREG/CR-6534 Vol. 1 Section 6.5 range of application.
_MAXIMUM_BURNUP_MWD_PER_KGHM = 80.0


def _normalized_to_unit_area_mean(values: Array, weights: Array, /) -> Array:
    """Scale ``values (..., R)`` so that ``sum(weights * values) == sum(weights)``."""

    mean = jnp.sum(weights * values, axis=-1) / jnp.sum(weights, axis=-1)
    return values / mean[..., None]


@final
class RadialPowerState(StrictModule):
    """TUBRNP state of pellets on a fixed reference radial quadrature.

    ``radii`` (m) and ``weights`` (m^2, quadrature of ``2 pi R dR`` over the pellet
    reference area) have shape ``(..., R)``; ``outer_radius`` (m) is the reference
    pellet radius. ``atom_fractions (..., 6)`` are pellet-average concentrations
    per initial heavy-metal atom in ``FISSIONING_NUCLIDES`` order, ``fima`` the
    pellet-average fissions per initial heavy-metal atom, and ``flux_time`` the
    one-group fluence (n/m^2). ``successful`` accumulates the exponential-action
    and flux-time root status of every step.
    """

    radii: Array
    weights: Array
    outer_radius: Array
    atom_fractions: Array
    initial_atom_fractions: Array
    heavy_metal_atom_density: Array
    heavy_metal_mass_per_atom: Array
    fima: Array
    burnup_mwd_per_kghm: Array
    flux_time: Array
    successful: Array


@final
class RadialPowerProfileModel(StrictModule):
    """TUBRNP radial power shape and fissioning-nuclide fractions of UO2 (LWR).

    Defaults are the FRAPCON-4.0 LWR values: cross sections of PNNL-19418 Vol. 1
    Rev. 2 Table 2.1 and ``p1 = 3.45``, ``p2 = 3.0``, ``p3 = 0.45``
    (Section 2.3.5.5). ``form_depth_unit_m`` is the length unit [m] of the
    depth in the empirical form, which the report does not specify.
    ``energy_per_fission`` holds the six recoverable energies per fission [J]
    in ``FISSIONING_NUCLIDES`` order. The scattering cross section is an
    effective spectrum-averaged UO2 value per initial heavy-metal atom, with
    its data source named by ``scattering_source_id``. The validity interval is
    the FRAPCON-3 range of application (NUREG/CR-6534 Vol. 1 Section 6.5).
    """

    fission_cross_section_barn: Array
    capture_cross_section_barn: Array
    form_amplitude: Array
    form_decay: Array
    form_exponent: Array
    form_depth_unit_m: Array
    energy_per_fission: Array
    scattering_cross_section_barn: Array
    scattering_source_id: str = eqx.field(static=True)
    root: LocalRootPlan
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        energy_per_fission: ArrayLike,
        scattering_cross_section_barn: ArrayLike,
        scattering_source_id: str,
        form_depth_unit_m: ArrayLike,
        fission_cross_section_barn: ArrayLike | Sequence[float] = _LWR_FISSION_BARN,
        capture_cross_section_barn: ArrayLike | Sequence[float] = _LWR_CAPTURE_BARN,
        form_amplitude: ArrayLike = _FORM_AMPLITUDE,
        form_decay: ArrayLike = _FORM_DECAY,
        form_exponent: ArrayLike = _FORM_EXPONENT,
    ) -> None:
        fission = finite_vector(
            fission_cross_section_barn,
            "fission_cross_section_barn",
            _NUCLIDE_COUNT,
            sign="nonnegative",
        )
        capture = finite_vector(
            capture_cross_section_barn,
            "capture_cross_section_barn",
            _NUCLIDE_COUNT,
            sign="nonnegative",
        )
        if not float(np.asarray(fission)[0]) > 0.0:
            raise ValueError("The U-235 fission cross section must be positive.")
        amplitude = finite_scalar(form_amplitude, "form_amplitude", positive=True)
        decay = finite_scalar(form_decay, "form_decay", positive=True)
        exponent = finite_scalar(form_exponent, "form_exponent", positive=True)
        depth_unit = finite_scalar(form_depth_unit_m, "form_depth_unit_m", positive=True)
        energy = finite_vector(
            energy_per_fission, "energy_per_fission", _NUCLIDE_COUNT, sign="positive"
        )
        scattering = finite_scalar(
            scattering_cross_section_barn, "scattering_cross_section_barn", positive=True
        )
        if not isinstance(scattering_source_id, str):
            raise TypeError("scattering_source_id must be a string.")
        if not scattering_source_id.strip():
            raise ValueError("scattering_source_id must name the scattering data source.")
        source = "pnnl-19418-vol1-rev2:TUBRNP"
        self.fission_cross_section_barn = fission
        self.capture_cross_section_barn = capture
        self.form_amplitude = amplitude
        self.form_decay = decay
        self.form_exponent = exponent
        self.form_depth_unit_m = depth_unit
        self.energy_per_fission = energy
        self.scattering_cross_section_barn = scattering
        self.scattering_source_id = scattering_source_id
        self.root = LocalRootPlan(
            maximum_steps=_FLUX_TIME_ROOT_MAXIMUM_STEPS,
            tolerance=_FLUX_TIME_ROOT_TOLERANCE,
            plan_id="tubrnp-flux-time",
        )
        self.validity = PropertyValidityRange(
            {"burnup_mwd_per_kghm": (0.0, _MAXIMUM_BURNUP_MWD_PER_KGHM, "MWd/kgHM")},
            source_id="nureg-cr-6534-vol1:TUBRNP",
        )
        self.source_id = source
        self.law_id = canonical_fingerprint(
            {
                "kind": "tubrnp-radial-power",
                "nuclides": list(FISSIONING_NUCLIDES),
                "fission_cross_section_barn": host_list(fission),
                "capture_cross_section_barn": host_list(capture),
                "form": host_list(jnp.stack([amplitude, decay, exponent])),
                "form_depth_unit_m": host_list(depth_unit)[0],
                "energy_per_fission": host_list(energy),
                "scattering_cross_section_barn": host_list(scattering)[0],
                "scattering_source_id": scattering_source_id,
                "flux": "solid-cylinder-one-group-diffusion",
                "source": source,
            }
        )

    def initial_state(
        self,
        radii: ArrayLike,
        /,
        *,
        weights: ArrayLike,
        outer_radius: ArrayLike,
        heavy_metal: HeavyMetalReference,
    ) -> RadialPowerState:
        """Fresh-fuel state on reference ``radii (..., R)`` with area ``weights``.

        The fresh fuel must have a positive one-group fission rate (for the
        default cross sections, a positive U-235 content); the power shape and
        fission fractions are normalized by it.
        """

        if not isinstance(heavy_metal, HeavyMetalReference):
            raise TypeError("heavy_metal must be a HeavyMetalReference.")
        radii_ = jnp.asarray(radii)
        weights_ = jnp.asarray(weights)
        outer = jnp.asarray(outer_radius)
        if radii_.ndim == 0 or radii_.shape != weights_.shape:
            raise ValueError("radii and weights must share one (..., R) shape.")
        dtype = inexact_result_type(radii_, weights_, outer, self.form_amplitude)
        leading = jnp.broadcast_shapes(radii_.shape[:-1], outer.shape)
        radial_point_count = radii_.shape[-1]
        radii_ = jnp.broadcast_to(radii_.astype(dtype), (*leading, radial_point_count))
        weights_ = jnp.broadcast_to(
            weights_.astype(dtype), (*leading, radial_point_count)
        )
        outer = jnp.broadcast_to(outer.astype(dtype), leading)
        outer = eqx.error_if(
            outer,
            ~jnp.isfinite(outer) | (outer <= 0.0),
            "TUBRNP outer_radius must be finite and positive.",
        )
        radii_ = eqx.error_if(
            radii_,
            ~jnp.isfinite(radii_) | (radii_ < 0.0) | (radii_ > outer[..., None]),
            "TUBRNP radii must lie in [0, outer_radius].",
        )
        weights_ = eqx.error_if(
            weights_,
            ~jnp.isfinite(weights_)
            | (weights_ < 0.0)
            | (jnp.sum(weights_, axis=-1, keepdims=True) <= 0.0),
            "TUBRNP area weights must be nonnegative with a positive sum.",
        )
        uranium_index = {
            key: index for index, key in enumerate(heavy_metal.uranium_isotopes)
        }
        uranium_densities = heavy_metal.uranium_number_densities.astype(dtype)
        heavy_metal_atoms = heavy_metal.heavy_metal_atom_density.astype(dtype)
        zero = jnp.zeros((), dtype=dtype)
        # FISSIONING_NUCLIDES order: U-235, U-238, then plutonium (absent).
        fractions = jnp.stack(
            [
                uranium_densities[uranium_index[NuclideKey(92, 235)]] / heavy_metal_atoms,
                uranium_densities[uranium_index[NuclideKey(92, 238)]] / heavy_metal_atoms,
                zero,
                zero,
                zero,
                zero,
            ]
        )
        fractions = eqx.error_if(
            fractions,
            ~(jnp.sum(fractions * self.fission_cross_section_barn.astype(dtype)) > 0.0),
            "TUBRNP fresh fuel must have a positive fission rate.",
        )
        fractions = jnp.broadcast_to(fractions, (*leading, _NUCLIDE_COUNT))
        return RadialPowerState(
            radii=radii_,
            weights=weights_,
            outer_radius=outer,
            atom_fractions=fractions,
            initial_atom_fractions=fractions,
            heavy_metal_atom_density=jnp.broadcast_to(heavy_metal_atoms, leading),
            heavy_metal_mass_per_atom=jnp.broadcast_to(
                heavy_metal.heavy_metal_mass_density.astype(dtype) / heavy_metal_atoms,
                leading,
            ),
            fima=jnp.zeros(leading, dtype=dtype),
            burnup_mwd_per_kghm=jnp.zeros(leading, dtype=dtype),
            flux_time=jnp.zeros(leading, dtype=dtype),
            successful=jnp.ones(leading, dtype=jnp.bool_),
        )

    def _generator(self, dtype: np.dtype, /) -> Array:
        """Depletion chain plus cumulative fissions per initial heavy-metal atom."""

        fission = self.fission_cross_section_barn.astype(dtype)
        capture = self.capture_cross_section_barn.astype(dtype)
        absorption = fission + capture
        size = _NUCLIDE_COUNT + 2
        chain = jnp.zeros((size, size), dtype=dtype)
        diagonal = jnp.arange(_NUCLIDE_COUNT)
        chain = chain.at[diagonal, diagonal].set(-absorption)
        # U-238 -> Pu-239 -> Pu-240 -> Pu-241 -> Pu-242 by capture.
        parents = jnp.arange(1, _NUCLIDE_COUNT - 1)
        chain = chain.at[parents + 1, parents].set(capture[1:-1])
        chain = chain.at[_FIMA_INDEX, :_NUCLIDE_COUNT].set(fission)
        energy = self.energy_per_fission.astype(dtype)
        return chain.at[_ENERGY_INDEX, :_NUCLIDE_COUNT].set(fission * energy / energy[0])

    def advance(
        self, state: RadialPowerState, /, *, burnup_increment_mwd_per_kghm: ArrayLike
    ) -> RadialPowerState:
        """Advance by a pellet-average burnup increment (MWd/kgHM), exactly in burnup."""

        if not isinstance(state, RadialPowerState):
            raise TypeError("state must be a RadialPowerState.")
        leading = state.fima.shape
        dtype = state.fima.dtype
        increment = jnp.broadcast_to(
            jnp.asarray(burnup_increment_mwd_per_kghm).astype(dtype), leading
        )
        increment = eqx.error_if(
            increment,
            ~jnp.isfinite(increment) | (increment < 0.0),
            "TUBRNP burnup increments must be finite and nonnegative.",
        )
        reference_energy = self.energy_per_fission.astype(dtype)[0]
        energy_increment = (
            increment
            * JOULE_PER_MEGAWATT_DAY
            * state.heavy_metal_mass_per_atom
            / reference_energy
        )
        generator = self._generator(dtype)
        operator = la.DenseLinearOperator(generator, operator_id="tubrnp-depletion")
        accumulated_energy = (
            state.burnup_mwd_per_kghm
            * JOULE_PER_MEGAWATT_DAY
            * state.heavy_metal_mass_per_atom
            / reference_energy
        )
        initial = jnp.concatenate(
            [state.atom_fractions, state.fima[..., None], accumulated_energy[..., None]],
            axis=-1,
        )
        fission = self.fission_cross_section_barn.astype(dtype)
        energy_rate = fission * self.energy_per_fission.astype(dtype) / reference_energy

        def lane(start: Array, target: Array) -> tuple[Array, Array, Array]:
            rate = jnp.dot(energy_rate, start[:_FIMA_INDEX])

            def evolve(flux_time: Array) -> tuple[Array, Array]:
                action = la.matrix_exponential_action(operator, start, flux_time)
                return jnp.asarray(action.value), action.successful

            def residual(flux_time: Array) -> Array:
                value, _ = evolve(flux_time)
                return (value[_ENERGY_INDEX] - start[_ENERGY_INDEX] - target) / rate

            step, diagnostics = self.root.solve_with_diagnostics(residual, target / rate)
            value, evolved = evolve(step)
            return value, step, diagnostics.converged & evolved

        flat_initial = initial.reshape(-1, _ENERGY_INDEX + 1)
        values, steps, converged = jax.vmap(lane)(
            flat_initial, energy_increment.reshape(-1)
        )
        values = values.reshape(*leading, _ENERGY_INDEX + 1)
        return RadialPowerState(
            radii=state.radii,
            weights=state.weights,
            outer_radius=state.outer_radius,
            atom_fractions=values[..., :_FIMA_INDEX],
            initial_atom_fractions=state.initial_atom_fractions,
            heavy_metal_atom_density=state.heavy_metal_atom_density,
            heavy_metal_mass_per_atom=state.heavy_metal_mass_per_atom,
            fima=values[..., _FIMA_INDEX],
            burnup_mwd_per_kghm=state.burnup_mwd_per_kghm + increment,
            flux_time=state.flux_time + steps.reshape(leading) / _SQUARE_METER_PER_BARN,
            successful=state.successful & converged.reshape(leading),
        )

    def form_function(self, state: RadialPowerState, /) -> Array:
        """Plutonium-production form function ``(..., R)`` with unit area average."""

        dtype = state.radii.dtype
        depth = jnp.maximum(state.outer_radius[..., None] - state.radii, 0.0)
        scaled_depth = depth / self.form_depth_unit_m.astype(dtype)
        inside = scaled_depth > 0.0
        # The inner selection keeps the power and its derivative finite at the
        # surface, where ``depth**p3`` with ``p3 < 1`` has an infinite slope.
        depth_power = jnp.where(
            inside,
            jnp.where(inside, scaled_depth, 1.0) ** self.form_exponent.astype(dtype),
            0.0,
        )
        raw = 1.0 + self.form_amplitude.astype(dtype) * jnp.exp(
            -self.form_decay.astype(dtype) * depth_power
        )
        return _normalized_to_unit_area_mean(raw, state.weights)

    def local_atom_fractions(self, state: RadialPowerState, /) -> Array:
        """Local concentrations ``(..., R, 6)`` per initial heavy-metal atom."""

        form = self.form_function(state)[..., None]
        average = state.atom_fractions[..., None, :]
        initial = state.initial_atom_fractions[..., None, :]
        uranium_238 = FISSIONING_NUCLIDES.index("U238")
        depleted = form * (initial - average)
        nuclide_indices = jnp.arange(_NUCLIDE_COUNT).reshape(
            (1,) * (average.ndim - 1) + (_NUCLIDE_COUNT,)
        )
        is_plutonium = nuclide_indices >= uranium_238 + 1
        shaped = jnp.where(is_plutonium, form * average, average)
        return jnp.where(nuclide_indices == uranium_238, initial - depleted, shaped)

    def _fission_rates(self, state: RadialPowerState, /) -> Array:
        """Local fission cross sections ``(..., R, 6)`` [barn per initial HM atom]."""

        local = self.local_atom_fractions(state)
        fission = self.fission_cross_section_barn.astype(state.radii.dtype).reshape(
            (1,) * (local.ndim - 1) + (_NUCLIDE_COUNT,)
        )
        return local * fission

    def thermal_flux_shape(self, state: RadialPowerState, /) -> Array:
        """Solid-cylinder flux ``(..., R)`` with unit reference-area average.

        ``kappa^2 = 3 Sigma_a Sigma_s`` (PNNL-19418 Eq. 2.41).
        Scaled I0 avoids overflow; the common surface exponential cancels during
        normalization. The scattering cross section is the declared model input.
        """

        dtype = state.radii.dtype
        density = state.heavy_metal_atom_density
        absorption = (
            self.fission_cross_section_barn + self.capture_cross_section_barn
        ).reshape((1,) * (state.atom_fractions.ndim - 1) + (_NUCLIDE_COUNT,))
        sigma_a = (
            density
            * jnp.sum(state.atom_fractions * absorption, axis=-1)
            * _SQUARE_METER_PER_BARN
        )
        sigma_s = density * self.scattering_cross_section_barn * _SQUARE_METER_PER_BARN
        kappa = jnp.sqrt(3.0 * sigma_a * sigma_s).astype(dtype)
        argument = kappa[..., None] * state.radii
        raw = i0e(argument) * jnp.exp(
            argument - kappa[..., None] * state.outer_radius[..., None]
        )
        return _normalized_to_unit_area_mean(raw, state.weights)

    def power_shape(self, state: RadialPowerState, /) -> Array:
        """Local power density ``(..., R)`` normalized to unit reference-area average.

        ``sum(weights * shape) == sum(weights)``, so the integral of the shape
        over ``2 pi R dR`` equals the pellet reference area.
        """

        rates = self._fission_rates(state)
        energy = self.energy_per_fission.reshape(
            (1,) * (rates.ndim - 1) + (_NUCLIDE_COUNT,)
        )
        density = jnp.sum(rates * energy, axis=-1) * self.thermal_flux_shape(state)
        return _normalized_to_unit_area_mean(density, state.weights)

    def fission_fractions(self, state: RadialPowerState, /) -> Array:
        """Local fission fractions ``(..., R, 6)`` in ``FISSIONING_NUCLIDES`` order.

        Requires a positive local fission rate at every radius; `initial_state`
        refuses fresh fuel without one.
        """

        rates = self._fission_rates(state)
        return rates / jnp.sum(rates, axis=-1, keepdims=True)


__all__ = [
    "RadialPowerProfileModel",
    "RadialPowerState",
]
