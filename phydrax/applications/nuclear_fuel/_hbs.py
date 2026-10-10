#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""High-burnup structure of UO2: restructuring and HBS pores.

Barani et al. (J. Nucl. Mater. 539 (2020) 152296) high-burnup-structure
formation on the effective burnup, and the Barani et al. (2022) source cluster
generator and pore-volume kinetics of the HBS pores that collect the depleted
intragranular gas.
"""

from __future__ import annotations

from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...nonlinear import (
    Brent,
    NonlinearTermination,
    ScalarRootProblem,
    VectorLocalRootDiagnostics,
    VectorLocalRootPlan,
)
from ...nonlinear._domain_cond import domain_cond
from ._constants import BOLTZMANN_J_PER_K
from ._gas_math import (
    carnahan_starling_numerator,
    host_parameters,
    positive_square_root,
)
from ._gas_state import FissionGasSiteState


# HBS pores nucleate as gas dimers (Barani et al., 2022).
DIMER_ATOMS = 2.0
# HBS grain-boundary tilt angle [degrees]: it rises linearly from the
# fabricated value with the restructured fraction, and its sine ratio scales
# the HBS gas and vacancy diffusivities.
_FABRICATED_TILT_DEGREES = 4.0
_RESTRUCTURED_TILT_INCREASE_DEGREES = 36.0


def _sphere_radius(volume: Array, /) -> Array:
    """Radius [m] of a sphere of ``volume`` [m^3]."""

    return jnp.cbrt(3.0 * volume / (4.0 * jnp.pi))


@final
class HighBurnupStructureFormation(StrictModule):
    """Barani et al. (2020) KJMA restructuring on the effective burnup.

    ``alpha = 1 - exp(-K bu_eff^gamma)`` with ``bu_eff`` in MWd/kgU accumulated
    only while ``T <= 1273.15`` K (Barani, PhD thesis 2020, Eqs. (3.1)-(3.3);
    BISON HighBurnupStructureFormation). ``K = 2.77e-7`` and ``gamma = 3.54``
    as published with Barani et al. (2020) and used by its authors in SCIANTIX;
    the thesis pre-print fit reports ``gamma = 3.35``.
    """

    rate_constant: Array
    avrami_exponent: Array
    threshold_temperature: Array
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        rate_constant: ArrayLike = 2.77e-7,
        avrami_exponent: ArrayLike = 3.54,
        threshold_temperature: ArrayLike = 1273.15,
    ) -> None:
        parameters = host_parameters(
            {
                "rate_constant": rate_constant,
                "avrami_exponent": avrami_exponent,
                "threshold_temperature": threshold_temperature,
            },
            positive=("avrami_exponent", "threshold_temperature"),
        )
        self.rate_constant = jnp.asarray(parameters["rate_constant"])
        self.avrami_exponent = jnp.asarray(parameters["avrami_exponent"])
        self.threshold_temperature = jnp.asarray(parameters["threshold_temperature"])
        self.source_id = "barani-2020-jnm-539-152296:kjma-effective-burnup"
        self.law_id = canonical_fingerprint(
            {"kind": "hbs-formation", "parameters": parameters}
        )

    def restructured_fraction(self, effective_burnup_mwd_per_kghm: ArrayLike, /) -> Array:
        """Restructured volume fraction ``alpha`` in ``[0, 1)``."""

        burnup = jnp.asarray(effective_burnup_mwd_per_kghm)
        positive = burnup > 0.0
        # Zero exposure is exactly unrestructured; the guard keeps derivatives
        # with respect to the Avrami exponent finite there.
        power = jnp.where(
            positive,
            jnp.power(jnp.where(positive, burnup, 1.0), self.avrami_exponent),
            0.0,
        )
        return -jnp.expm1(-self.rate_constant * power)


@final
class HbsPoreUpdate(StrictModule):
    gas: Array
    boundary_gas: Array
    density: Array
    variance: Array
    vacancies: Array
    porosity: Array
    nucleation: Array
    trapping: Array
    resolution: Array
    converged: Array
    residual_norm: Array
    condition_estimate: Array
    linear_status: Array
    linear_rank: Array
    conditioning_available: Array


class _HbsRates(NamedTuple):
    """Temperature- and restructuring-dependent HBS coefficients of one site."""

    gas_atomic_volume: Array  # Xe hard-sphere volume [m^3/atom]
    thermal_energy: Array  # k T [J]
    gas_diffusivity: Array  # [m^2/s]
    vacancy_diffusivity: Array  # [m^2/s]


class _HbsSite(NamedTuple):
    """Accepted HBS inventories of one site and its endpoint inputs."""

    pore_gas: Array  # [atoms/m^3]
    boundary_gas: Array  # [atoms/m^3]
    pore_number_density: Array  # [1/m^3]
    variance: Array  # integrated central second moment [atoms^2/m^3]
    vacancies: Array  # [vacancies/m^3]
    arrival: Array  # gas entering HBS storage during the step [atoms/m^3]
    total_gas: Array  # pore_gas + boundary_gas + arrival [atoms/m^3]
    nucleated: Array  # dimer pores born during the step [1/m^3]
    fission_rate_density: Array  # [fissions/(m^3 s)]
    stress: Array  # hydrostatic stress [Pa], compression negative
    dt: Array  # [s]
    rates: _HbsRates


class _HbsPoreMoments(NamedTuple):
    number: Array  # pore number density [1/m^3]
    atoms: Array  # pore gas [atoms/m^3]
    variance: Array  # integrated central second moment [atoms^2/m^3]
    vacancies: Array  # [vacancies/m^3]


class _HbsRootScaling(NamedTuple):
    """Scaled coordinates of the four HBS root unknowns.

    ``value[0] = log(N / N_s) / L``, ``value[1] = (m - 2) / X``,
    ``value[2] = M2 / V`` and ``value[3] = log(n_v N_s / (N W))`` for the pore
    number density ``N``, mean atoms per pore ``m``, integrated central second
    moment ``M2`` and vacancy inventory ``n_v``.
    """

    number: Array  # N_s, pore number density scale [1/m^3]
    log_number: Array  # L, scale of log(N / N_s)
    excess: Array  # X, scale of the mean atoms per pore beyond the dimer
    variance: Array  # V [atoms^2/m^3]
    vacancies: Array  # W [vacancies/m^3]
    dimer_margin: Array  # boundary gas left after filling N_s dimers [atoms/m^3]

    def encode(
        self, number_loss: Array, excess: Array, variance: Array, vacancies: Array
    ) -> Array:
        """Seed coordinates for the pore number ``N_s / (1 + number_loss)``."""

        return jnp.stack(
            [
                self.number_coordinate(number_loss),
                excess / self.excess,
                variance / self.variance,
                jnp.log(vacancies / self.vacancies),
            ]
        )

    def number_coordinate(self, number_loss: Array) -> Array:
        return -jnp.log1p(number_loss) / self.log_number

    def excess_coordinate(self, excess_atoms: Array) -> Array:
        return excess_atoms / (self.number * self.excess)

    def decode(self, value: Array) -> _HbsPoreMoments:
        number = jnp.exp(value[0] * self.log_number) * self.number
        mean = DIMER_ATOMS + value[1] * self.excess
        return _HbsPoreMoments(
            number=number,
            atoms=number * mean,
            variance=value[2] * self.variance,
            vacancies=jnp.exp(value[3]) * self.vacancies * (number / self.number),
        )

    def remaining_gas(self, value: Array) -> Array:
        """Boundary gas left at ``value`` [atoms/m^3]."""

        return _hbs_remaining_gas(
            self.number,
            self.dimer_margin,
            value[0] * self.log_number,
            value[1] * self.excess,
        )


class _HbsRootTerms(NamedTuple):
    """Residual rows and step transfers of the HBS root at one trial value."""

    variance_residual: Array
    vacancy_residual: Array
    trapped: Array  # boundary gas captured by pores [atoms/m^3]
    resolved: Array  # pore gas re-solved to the boundary [atoms/m^3]
    pore_volume: Array  # mean volume per pore [m^3]
    porosity: Array
    number_loss: Array  # dt times the mean pore destruction rate
    excess_atoms: Array  # pore gas beyond the accepted dimers [atoms/m^3]


class _HbsRootSetup(NamedTuple):
    """Scaling, frozen rate parameters and seed of one active HBS root."""

    scaling: _HbsRootScaling
    rate_parameters: Array  # stacked parameters of ``_hbs_rate_moments``
    seed: Array
    seed_excess: Array
    seeded_vacancies: Array  # [vacancies/m^3]
    baseline_porosity: Array


def _porosity_capture_factor(porosity: Array, /) -> Array:
    """Porosity enhancement ``1 + 1.8 p^1.3`` of the diffusion-limited gas capture."""

    return 1.0 + 1.8 * porosity**1.3


def _diffusion_limited_capture_rate(
    gas_diffusivity: Array, dissolved_gas: Array, radius: Array, porosity: Array, /
) -> Array:
    """Capture rate ``4 pi D c R (1 + 1.8 p^1.3)`` [1/s] of one pore of radius ``R``."""

    return (
        4.0
        * jnp.pi
        * gas_diffusivity
        * dissolved_gas
        * radius
        * _porosity_capture_factor(porosity)
    )


def _hbs_rate_moments_impl(
    excess: Array,
    central_variance: Array,
    per_pore_vacancies: Array,
    dissolved: Array,
    porosity: Array,
    parameters: Array,
    *,
    tangent_form: bool,
) -> Array:
    """Two-node Gauss–Radau moments of the size-dependent source rates.

    The pore size ``n`` has mean ``2 + excess`` and ``central_variance``; the
    nodes are the dimer and ``2 + excess + central_variance / excess``. With
    the re-solution rate ``alpha(n)`` and capture rate ``beta(n)`` [1/s] the
    result stacks ``[E alpha, E beta, E n alpha, Cov(n, alpha), Cov(n, beta),
    E (n - E n)^2 alpha]``. ``tangent_form`` evaluates the same moments with
    divided differences that stay accurate as the distribution degenerates.
    """

    omega, vacancy_volume, resolution_scale, distance, layer, gas_diffusion = parameters
    beta_scale = (
        4.0 * jnp.pi * gas_diffusion * dissolved * _porosity_capture_factor(porosity)
    )

    def rates_at_radius(radius: Array) -> tuple[Array, Array]:
        alpha = (
            resolution_scale
            * (3.0 * distance / (3.0 * distance + radius))
            * (layer / (layer + radius))
        )
        return alpha, beta_scale * radius

    def rates(size: Array) -> tuple[Array, Array]:
        radius = _sphere_radius(omega * size + vacancy_volume * per_pore_vacancies)
        return rates_at_radius(radius)

    def quadrature(_: None) -> Array:
        radius_low = _sphere_radius(
            DIMER_ATOMS * omega + vacancy_volume * per_pore_vacancies
        )
        size_high = DIMER_ATOMS + excess + central_variance / excess
        radius_high = _sphere_radius(
            omega * size_high + vacancy_volume * per_pore_vacancies
        )
        alpha_low, beta_low = rates_at_radius(radius_low)
        alpha_high, beta_high = rates_at_radius(radius_high)
        radius_slope = (3.0 * omega / (4.0 * jnp.pi)) / (
            radius_high**2 + radius_high * radius_low + radius_low**2
        )
        alpha_slope = (
            -resolution_scale
            * 3.0
            * distance
            * layer
            * radius_slope
            * (3.0 * distance + layer + radius_low + radius_high)
            / (
                (3.0 * distance + radius_low)
                * (layer + radius_low)
                * (3.0 * distance + radius_high)
                * (layer + radius_high)
            )
        )
        beta_slope = beta_scale * radius_slope
        denominator = excess**2 + central_variance
        high_weight = excess**2 / denominator
        low_weight = central_variance / denominator
        if tangent_form:
            # Algebraically identical divided differences avoid cancellation
            # in the degenerate-distribution derivative.
            mean_alpha = alpha_low + excess * alpha_slope
            mean_beta = beta_low + excess * beta_slope
        else:
            mean_alpha = low_weight * alpha_low + high_weight * alpha_high
            mean_beta = low_weight * beta_low + high_weight * beta_high
        return jnp.stack(
            [
                mean_alpha,
                mean_beta,
                DIMER_ATOMS * mean_alpha + excess * alpha_high,
                central_variance * alpha_slope,
                central_variance * beta_slope,
                central_variance * (high_weight * alpha_low + low_weight * alpha_high),
            ]
        )

    def degenerate(_: None) -> Array:
        mean = DIMER_ATOMS + excess
        alpha, beta = rates(mean)
        if tangent_form:
            alpha_first, beta_first = jax.jacfwd(rates)(mean)
            alpha_second, beta_second = jax.jacfwd(jax.jacfwd(rates))(mean)
            return jnp.stack(
                [
                    alpha + 0.5 * central_variance * alpha_second,
                    beta + 0.5 * central_variance * beta_second,
                    mean * alpha
                    + central_variance * (alpha_first + 0.5 * mean * alpha_second),
                    central_variance * alpha_first,
                    central_variance * beta_first,
                    central_variance * alpha,
                ]
            )
        zero = jnp.zeros_like(mean)
        return jnp.stack([alpha, beta, mean * alpha, zero, zero, zero])

    return domain_cond(excess > 0.0, quadrature, degenerate)


@jax.custom_jvp
def _hbs_rate_moments(
    excess: Array,
    central_variance: Array,
    per_pore_vacancies: Array,
    dissolved: Array,
    porosity: Array,
    parameters: Array,
) -> Array:
    """Positive source rate moments with exact moment-preserving tangents."""

    return _hbs_rate_moments_impl(
        excess,
        central_variance,
        per_pore_vacancies,
        dissolved,
        porosity,
        parameters,
        tangent_form=False,
    )


@_hbs_rate_moments.defjvp
def _hbs_rate_moments_jvp(
    primals: tuple[Array, ...], tangents: tuple[Array, ...]
) -> tuple[Array, Array]:
    primal = _hbs_rate_moments(*primals)
    _, tangent = jax.jvp(
        lambda *values: _hbs_rate_moments_impl(*values, tangent_form=True),
        primals,
        tangents,
    )
    return primal, tangent


def _hbs_remaining_gas(
    number_scale: Array,
    dimer_margin: Array,
    log_number_ratio: Array,
    mean_excess: Array,
    /,
) -> Array:
    """Boundary gas left without subtracting nearly equal dimer totals [atoms/m^3]."""

    number = number_scale * jnp.exp(log_number_ratio)
    return (
        dimer_margin
        - DIMER_ATOMS * number_scale * jnp.expm1(log_number_ratio)
        - number * mean_excess
    )


@final
class HighBurnupStructurePores(StrictModule):
    """Barani et al. (2022), source cluster generator and pore-volume kinetics.

    Couples pore number, gas first moment, central second moment and total
    vacancies in one backward-Euler root. Source size-dependent rates are
    averaged with a positive two-node Gauss–Radau moment reconstruction anchored
    at the source dimer size. Broad distributions are not Taylor-truncated.
    Impingement uses the source monodisperse closure: gas and vacancy volume
    are conserved and the integrated central second moment is held fixed.
    """

    nucleation_coefficient: Array
    gas_diffusivity_prefactor: Array
    gas_activation_energy: Array
    vacancy_diffusivity_prefactor: Array
    vacancy_activation_energy: Array
    vacancy_irradiation_coefficient: Array
    resolution_coefficient: Array
    resolution_distance: Array
    resolution_layer: Array
    surface_energy: Array
    vacancy_volume: Array
    root: VectorLocalRootPlan
    mechanical_seed_method: Brent
    mechanical_seed_termination: NonlinearTermination
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        nucleation_coefficient: ArrayLike = 5.0e17,
        gas_diffusivity_prefactor: ArrayLike = 1.3e-7,
        gas_activation_energy: ArrayLike = 4.52e-19,
        vacancy_diffusivity_prefactor: ArrayLike = 8.86e-6,
        vacancy_activation_energy: ArrayLike = 5.75e-19,
        vacancy_irradiation_coefficient: ArrayLike = 1.0e-39,
        resolution_coefficient: ArrayLike = 2.0e-23,
        resolution_distance: ArrayLike = 1.0e-9,
        resolution_layer: ArrayLike = 1.0e-9,
        surface_energy: ArrayLike = 1.0,
        vacancy_volume: ArrayLike = 4.09e-29,
    ) -> None:
        parameters = host_parameters(
            {
                "nucleation_coefficient": nucleation_coefficient,
                "gas_diffusivity_prefactor": gas_diffusivity_prefactor,
                "gas_activation_energy": gas_activation_energy,
                "vacancy_diffusivity_prefactor": vacancy_diffusivity_prefactor,
                "vacancy_activation_energy": vacancy_activation_energy,
                "vacancy_irradiation_coefficient": vacancy_irradiation_coefficient,
                "resolution_coefficient": resolution_coefficient,
                "resolution_distance": resolution_distance,
                "resolution_layer": resolution_layer,
                "surface_energy": surface_energy,
                "vacancy_volume": vacancy_volume,
            },
            positive=("resolution_distance", "resolution_layer", "vacancy_volume"),
        )
        self.nucleation_coefficient = jnp.asarray(parameters["nucleation_coefficient"])
        self.gas_diffusivity_prefactor = jnp.asarray(
            parameters["gas_diffusivity_prefactor"]
        )
        self.gas_activation_energy = jnp.asarray(parameters["gas_activation_energy"])
        self.vacancy_diffusivity_prefactor = jnp.asarray(
            parameters["vacancy_diffusivity_prefactor"]
        )
        self.vacancy_activation_energy = jnp.asarray(
            parameters["vacancy_activation_energy"]
        )
        self.vacancy_irradiation_coefficient = jnp.asarray(
            parameters["vacancy_irradiation_coefficient"]
        )
        self.resolution_coefficient = jnp.asarray(parameters["resolution_coefficient"])
        self.resolution_distance = jnp.asarray(parameters["resolution_distance"])
        self.resolution_layer = jnp.asarray(parameters["resolution_layer"])
        self.surface_energy = jnp.asarray(parameters["surface_energy"])
        self.vacancy_volume = jnp.asarray(parameters["vacancy_volume"])
        self.root = VectorLocalRootPlan(
            4, maximum_steps=60, tolerance=1.0e-10, plan_id="hbs-pore-moments-vacancies"
        )
        self.mechanical_seed_method = Brent()
        self.mechanical_seed_termination = NonlinearTermination(
            absolute_residual=1.0e-9,
            relative_residual=0.0,
            maximum_steps=80,
            maximum_evaluations=180,
        )
        self.source_id = "barani-2022-part-II:eq1-18,table1"
        self.law_id = canonical_fingerprint(
            {
                "kind": "hbs-pore-moments",
                "parameters": parameters,
                "root": self.root.plan_id,
                "birth_extent": "boundary-monomer-dimer-budget",
                "gas_atomic_volume": "xe-temperature-dependent-hard-sphere",
                "pore_shape": "positive-dimer-radau",
                "pore_quadrature": 2,
                "state_domain": "positive-moments-conserved-gas-volume",
            }
        )

    def _gas_atomic_volume(self, temperature: Array, /) -> Array:
        """Xe hard-sphere volume [m^3/atom], Barani et al. (2022), Eq. 17."""

        diameter = 4.45e-10 * (0.8542 - 0.03996 * jnp.log(temperature / 231.2))
        return jnp.pi / 6.0 * diameter**3

    def _rates(
        self, temperature: Array, fraction: Array, fission_rate_density: Array
    ) -> _HbsRates:
        thermal_energy = BOLTZMANN_J_PER_K * temperature
        tilt = jnp.sin(
            jnp.deg2rad(
                _FABRICATED_TILT_DEGREES + _RESTRUCTURED_TILT_INCREASE_DEGREES * fraction
            )
        ) / jnp.sin(jnp.deg2rad(_FABRICATED_TILT_DEGREES))
        return _HbsRates(
            gas_atomic_volume=self._gas_atomic_volume(temperature),
            thermal_energy=thermal_energy,
            gas_diffusivity=tilt
            * self.gas_diffusivity_prefactor
            * jnp.exp(-self.gas_activation_energy / thermal_energy),
            vacancy_diffusivity=tilt
            * (
                self.vacancy_diffusivity_prefactor
                * jnp.exp(-self.vacancy_activation_energy / thermal_energy)
                + self.vacancy_irradiation_coefficient * fission_rate_density
            ),
        )

    def _resolution_loss(self, site: _HbsSite, radius: Array) -> Array:
        """Fraction ``dt alpha(R)`` of pores of radius ``R`` destroyed by re-solution."""

        return (
            site.dt
            * self.resolution_coefficient
            * site.fission_rate_density
            * (3.0 * self.resolution_distance / (3.0 * self.resolution_distance + radius))
            * (self.resolution_layer / (self.resolution_layer + radius))
        )

    def _mechanical_vacancy_seed(
        self,
        mean_atoms: Array,
        pore_number_density: Array,
        rates: _HbsRates,
        stress: Array,
        previous_vacancies: Array,
    ) -> Array:
        """Vacancies per pore at the gas-pressure/capillary balance; a root seed only.

        The pore volume ratio ``V / (Omega m)`` balances the Carnahan-Starling
        gas pressure against ``2 gamma / R - sigma``; the seed carries no
        derivative.
        """

        (
            mean_atoms,
            pore_number_density,
            thermal_energy,
            stress,
            surface_energy,
            gas_atomic_volume,
            vacancy_volume,
            previous_vacancies,
        ) = jax.lax.stop_gradient(
            jnp.stack(
                [
                    mean_atoms,
                    pore_number_density,
                    rates.thermal_energy,
                    stress,
                    self.surface_energy,
                    rates.gas_atomic_volume,
                    self.vacancy_volume,
                    previous_vacancies,
                ]
            )
        )
        gas_radius = jnp.cbrt(3.0 * gas_atomic_volume * mean_atoms / (4.0 * jnp.pi))
        pressure_scale = thermal_energy / gas_atomic_volume + jnp.abs(
            2.0 * surface_energy / gas_radius - stress
        )

        def equation(log_volume_ratio: Array, _: None) -> Array:
            packing = jnp.exp(-log_volume_ratio)
            free_fraction = -jnp.expm1(-log_volume_ratio)
            pressure = (
                thermal_energy
                / gas_atomic_volume
                * packing
                * carnahan_starling_numerator(packing)
                / free_fraction**3
            )
            capillary = (
                2.0 * surface_energy / gas_radius * jnp.exp(-log_volume_ratio / 3.0)
                - stress
            )
            return (pressure - capillary) / pressure_scale

        # At volume ratio >= 2 the packing is <= 1/2 and the exact
        # Carnahan-Starling bound is Z <= 13. Compressive-pressure and capillary
        # upper bounds then bracket the stable mechanical root.
        compressed = stress < 0.0
        compression_bound = jnp.log(
            jnp.maximum(
                2.0,
                13.0
                * thermal_energy
                / (gas_atomic_volume * jnp.where(compressed, -stress, 1.0)),
            )
        )
        capillary_bound = jnp.maximum(
            jnp.log(2.0),
            1.5
            * jnp.log(
                39.0
                * mean_atoms
                * thermal_energy
                / (
                    4.0
                    * jnp.pi
                    * jnp.where(surface_energy > 0.0, surface_energy, 1.0)
                    * gas_radius**2
                )
            ),
        )
        lower = jnp.asarray(np.finfo(np.float64).eps, dtype=jnp.float64)
        upper = jnp.minimum(
            jnp.where(compressed, compression_bound, capillary_bound),
            -jnp.log(gas_atomic_volume * mean_atoms * pore_number_density),
        )
        bracketed = (upper > lower) & jnp.isfinite(upper) & (equation(upper, None) < 0.0)

        def solve(_: None) -> Array:
            problem = ScalarRootProblem(
                equation,
                bracket=(lower, upper),
                problem_id="hbs-mechanical-initialization",
            )
            result = self.mechanical_seed_method.solve(
                problem, termination=self.mechanical_seed_termination
            )
            return jax.lax.stop_gradient(
                mean_atoms * gas_atomic_volume / vacancy_volume * jnp.expm1(result.root)
            )

        def unbracketed(_: None) -> Array:
            return previous_vacancies

        return domain_cond(bracketed, solve, unbracketed)

    def _vacancy_seed(
        self, site: _HbsSite, number_scale: Array, seeded_atoms: Array
    ) -> Array:
        """Vacancy inventory seed [vacancies/m^3] of the active root.

        Mechanical equilibrium is capped by the packed-gas kinetic limit
        ``n_v^4 = 2 dt M k T / Omega (Omega m / Omega_v)^3``, which governs when
        vacancy diffusion is very slow.
        """

        rates = site.rates

        def mechanical(_: None) -> Array:
            mean_atoms = seeded_atoms / number_scale
            equilibrium_vacancies = self._mechanical_vacancy_seed(
                mean_atoms,
                number_scale,
                rates,
                site.stress,
                site.vacancies / number_scale,
            )
            gas_radius = jnp.cbrt(
                3.0 * rates.gas_atomic_volume * mean_atoms / (4.0 * jnp.pi)
            )
            cell_radius = jnp.cbrt(3.0 / (4.0 * jnp.pi * number_scale))
            radius_ratio = gas_radius / cell_radius
            mobility_factor = (
                10.0
                * radius_ratio
                * (1.0 + radius_ratio**3)
                / (-(radius_ratio**6) + 5.0 * radius_ratio**2 - 9.0 * radius_ratio + 5.0)
            )
            mobility = (
                2.0
                * jnp.pi
                * rates.vacancy_diffusivity
                * cell_radius
                / (rates.thermal_energy * mobility_factor)
            )
            kinetic_vacancies = jnp.exp(
                0.25
                * (
                    jnp.log(
                        2.0
                        * site.dt
                        * mobility
                        * rates.thermal_energy
                        / rates.gas_atomic_volume
                    )
                    + 3.0
                    * jnp.log(rates.gas_atomic_volume * mean_atoms / self.vacancy_volume)
                )
            )
            return number_scale * jnp.minimum(
                equilibrium_vacancies,
                jnp.maximum(site.vacancies / number_scale, kinetic_vacancies),
            )

        def immobile(_: None) -> Array:
            return site.vacancies

        return domain_cond(rates.vacancy_diffusivity > 0.0, mechanical, immobile)

    def _kinetically_resolved(self, site: _HbsSite) -> Array:
        """Whether HBS rates can change scaled inventories above the root tolerance."""

        rates = site.rates
        number = site.pore_number_density + site.nucleated
        safe_number = jnp.where(number > 0.0, number, 1.0)
        mean_atoms = site.total_gas / safe_number
        volume = (
            rates.gas_atomic_volume * mean_atoms
            + self.vacancy_volume * site.vacancies / safe_number
        )
        radius = _sphere_radius(volume)
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        porosity = number * volume
        resolution = (
            site.dt
            * self.resolution_coefficient
            * site.fission_rate_density
            * (3.0 * self.resolution_distance)
            / (3.0 * self.resolution_distance + safe_radius)
            * self.resolution_layer
            / (self.resolution_layer + safe_radius)
        )
        capture = (
            site.dt
            * number
            * 4.0
            * jnp.pi
            * rates.gas_diffusivity
            * safe_radius
            * _porosity_capture_factor(porosity)
        )
        vacancy = site.dt * rates.vacancy_diffusivity / safe_radius**2
        scale = jnp.maximum(resolution, jnp.maximum(capture, vacancy))
        return (site.nucleated > 0.0) | (scale > self.root.tolerance)

    def _site_step(
        self,
        pore_gas: Array,
        boundary_gas: Array,
        density: Array,
        variance: Array,
        vacancies: Array,
        arrival: Array,
        fraction_increment: Array,
        fraction: Array,
        temperature: Array,
        fission_rate_density: Array,
        stress: Array,
        dt: Array,
    ) -> HbsPoreUpdate:
        rates = self._rates(temperature, fraction, fission_rate_density)
        site = _HbsSite(
            pore_gas=pore_gas,
            boundary_gas=boundary_gas,
            pore_number_density=density,
            variance=variance,
            vacancies=vacancies,
            arrival=arrival,
            total_gas=pore_gas + boundary_gas + arrival,
            # Pores nucleate as dimers, so the restructuring birth extent cannot
            # consume more than half of the available boundary atoms.
            nucleated=jnp.minimum(
                self.nucleation_coefficient * fraction_increment,
                0.5 * (boundary_gas + arrival),
            ),
            fission_rate_density=fission_rate_density,
            stress=stress,
            dt=dt,
            rates=rates,
        )
        active = (
            (site.total_gas > 0.0)
            & (density + site.nucleated > 0.0)
            & (dt > 0.0)
            & self._kinetically_resolved(site)
        )

        def evolve(_: None) -> HbsPoreUpdate:
            return self._active_site_step(site)

        def inactive(_: None) -> HbsPoreUpdate:
            zero = jnp.zeros_like(pore_gas)
            return HbsPoreUpdate(
                gas=pore_gas,
                boundary_gas=boundary_gas + arrival,
                density=density,
                variance=variance,
                vacancies=vacancies,
                porosity=rates.gas_atomic_volume * pore_gas
                + self.vacancy_volume * vacancies,
                nucleation=zero,
                trapping=zero,
                resolution=zero,
                converged=jnp.asarray(True, dtype=jnp.bool_),
                residual_norm=zero,
                condition_estimate=jnp.ones_like(pore_gas),
                linear_status=jnp.asarray(0, dtype=jnp.int32),
                linear_rank=jnp.asarray(self.root.dimension, dtype=jnp.int32),
                conditioning_available=jnp.asarray(True, dtype=jnp.bool_),
            )

        return domain_cond(active, evolve, inactive)

    def _active_site_step(self, site: _HbsSite) -> HbsPoreUpdate:
        """Solve the coupled pore-moment/vacancy root of one active site and commit it.

        A failed primary solve under tension with mobile vacancies is retried
        from a growing-void seed.
        """

        setup = self._root_setup(site)

        def residual(value: Array) -> Array:
            return self._root_residual(site, setup, value)

        def admissible(value: Array) -> Array:
            return self._root_admissible(site, setup, value)

        primary_solution, primary_diagnostics = self.root.solve_with_diagnostics(
            residual, setup.seed, admissible=admissible
        )

        def growing_void(_: None) -> tuple[Array, VectorLocalRootDiagnostics]:
            return self.root.solve_with_diagnostics(
                residual, self._growing_void_seed(site, setup), admissible=admissible
            )

        def keep_primary(_: None) -> tuple[Array, VectorLocalRootDiagnostics]:
            return primary_solution, primary_diagnostics

        solution, diagnostics = domain_cond(
            (~primary_diagnostics.converged)
            & (site.stress > 0.0)
            & (site.rates.vacancy_diffusivity > 0.0),
            growing_void,
            keep_primary,
        )
        return self._commit_root(site, setup, solution, diagnostics)

    def _root_setup(self, site: _HbsSite) -> _HbsRootSetup:
        """Scales and primary seed of the active root."""

        rates = site.rates
        gas_atomic_volume = rates.gas_atomic_volume
        number_scale = site.pore_number_density + site.nucleated
        dimer_margin = (
            site.pore_gas
            - DIMER_ATOMS * site.pore_number_density
            + site.boundary_gas
            + site.arrival
            - DIMER_ATOMS * site.nucleated
        )
        baseline_atoms = jnp.minimum(
            site.total_gas,
            jnp.maximum(
                site.pore_gas + DIMER_ATOMS * site.nucleated, DIMER_ATOMS * number_scale
            ),
        )
        baseline_porosity = (
            gas_atomic_volume * baseline_atoms + self.vacancy_volume * site.vacancies
        )
        # The source beta(n) ~ k n^(1/3) gives n ~ (dt k)^(3/2) for
        # capture-dominated backward-Euler steps. Use that balance only to
        # scale and seed the full root.
        growth_balance = (
            site.dt
            * 4.0
            * jnp.pi
            * rates.gas_diffusivity
            * site.total_gas
            * _sphere_radius(gas_atomic_volume)
            * _porosity_capture_factor(baseline_porosity)
        )
        predicted_mean = growth_balance * positive_square_root(growth_balance)
        seeded_atoms = jnp.minimum(
            site.total_gas, jnp.maximum(baseline_atoms, number_scale * predicted_mean)
        )
        seeded_vacancies = self._vacancy_seed(site, number_scale, seeded_atoms)
        vacancy_scale = jnp.maximum(
            seeded_atoms, jnp.maximum(site.vacancies, seeded_vacancies)
        )
        seed_radius = jnp.cbrt(
            3.0
            * (gas_atomic_volume * seeded_atoms + self.vacancy_volume * seeded_vacancies)
            / (4.0 * jnp.pi * number_scale)
        )
        seed_loss = self._resolution_loss(site, seed_radius)
        # The exact source loss is the relative number-change scale. Without
        # it, near-dimer birth creates artificial O(1/loss) Jacobian entries.
        log_number_scale = jnp.where(seed_loss > 0.0, jnp.log1p(seed_loss), 1.0)
        seed_number = number_scale / (1.0 + seed_loss)
        seed_mean = seeded_atoms / number_scale
        seed_capture = _diffusion_limited_capture_rate(
            rates.gas_diffusivity,
            _hbs_remaining_gas(
                number_scale,
                dimer_margin,
                -jnp.log1p(seed_loss),
                seed_mean - DIMER_ATOMS,
            ),
            seed_radius,
            baseline_porosity,
        )
        predicted_variance = (
            site.variance
            + site.nucleated * (seed_mean - DIMER_ATOMS) ** 2
            + site.dt * seed_number * seed_capture
        )
        seed_resolution_rate = seed_loss / site.dt
        seed_resolution_slope = (
            -seed_resolution_rate
            * gas_atomic_volume
            / (4.0 * jnp.pi * seed_radius**2)
            * (
                1.0 / (3.0 * self.resolution_distance + seed_radius)
                + 1.0 / (self.resolution_layer + seed_radius)
            )
        )
        seed_excess = jnp.minimum(
            site.total_gas / seed_number - DIMER_ATOMS,
            jnp.maximum(
                seed_mean - DIMER_ATOMS,
                site.dt
                * (seed_capture - seed_resolution_slope * site.variance / number_scale)
                / (1.0 + seed_loss),
            ),
        )
        scaling = _HbsRootScaling(
            number=number_scale,
            log_number=log_number_scale,
            excess=jnp.where(seed_excess > 0.0, seed_excess, 1.0),
            variance=jnp.where(predicted_variance > 0.0, predicted_variance, 1.0),
            vacancies=vacancy_scale,
            dimer_margin=dimer_margin,
        )
        return _HbsRootSetup(
            scaling=scaling,
            rate_parameters=jnp.stack(
                [
                    gas_atomic_volume,
                    self.vacancy_volume,
                    self.resolution_coefficient * site.fission_rate_density,
                    self.resolution_distance,
                    self.resolution_layer,
                    rates.gas_diffusivity,
                ]
            ),
            seed=scaling.encode(
                seed_loss,
                seed_excess,
                predicted_variance / (1.0 + seed_loss),
                seeded_vacancies,
            ),
            seed_excess=seed_excess,
            seeded_vacancies=seeded_vacancies,
            baseline_porosity=baseline_porosity,
        )

    def _growing_void_seed(self, site: _HbsSite, setup: _HbsRootSetup) -> Array:
        """Seed above the source capillary spinodal, where no mechanical seed exists.

        The large-void source balance is ``n_v^(4/3) = dt pi D_v cell^2 sigma /
        (k T) (4 pi / (3 Omega_v))^(1/3)``.
        """

        rates = site.rates
        scaling = setup.scaling
        cell_radius = jnp.cbrt(3.0 / (4.0 * jnp.pi * scaling.number))
        kinetic_vacancies = jnp.exp(
            0.75
            * (
                jnp.log(
                    site.dt
                    * jnp.pi
                    * rates.vacancy_diffusivity
                    * cell_radius**2
                    * site.stress
                    / rates.thermal_energy
                )
                + jnp.log(4.0 * jnp.pi / (3.0 * self.vacancy_volume)) / 3.0
            )
        )
        per_pore_vacancies = jnp.maximum(
            kinetic_vacancies,
            jnp.maximum(site.vacancies, setup.seeded_vacancies) / scaling.number,
        )
        radius = _sphere_radius(
            rates.gas_atomic_volume * (DIMER_ATOMS + setup.seed_excess)
            + self.vacancy_volume * per_pore_vacancies
        )
        loss = self._resolution_loss(site, radius)
        number = scaling.number / (1.0 + loss)
        excess = jnp.minimum(setup.seed_excess, site.total_gas / number - DIMER_ATOMS)
        capture = _diffusion_limited_capture_rate(
            rates.gas_diffusivity,
            _hbs_remaining_gas(
                scaling.number, scaling.dimer_margin, -jnp.log1p(loss), excess
            ),
            radius,
            setup.baseline_porosity,
        )
        variance = (
            site.variance + site.nucleated * excess**2 + site.dt * number * capture
        ) / (1.0 + loss)
        return scaling.encode(loss, excess, variance, per_pore_vacancies * scaling.number)

    def _root_terms(
        self, site: _HbsSite, setup: _HbsRootSetup, value: Array
    ) -> _HbsRootTerms:
        rates = site.rates
        gas_atomic_volume = rates.gas_atomic_volume
        scaling = setup.scaling
        moments = scaling.decode(value)
        number = moments.number
        excess = value[1] * scaling.excess
        mean_atoms = DIMER_ATOMS + excess
        per_pore_vacancies = moments.vacancies / number
        volume = gas_atomic_volume * mean_atoms + self.vacancy_volume * per_pore_vacancies
        radius = _sphere_radius(volume)
        porosity = number * volume
        (
            mean_resolution_rate,
            mean_capture_rate,
            resolved_atom_rate,
            resolution_covariance,
            capture_covariance,
            resolution_variance_rate,
        ) = _hbs_rate_moments(
            excess,
            moments.variance / number,
            per_pore_vacancies,
            scaling.remaining_gas(value),
            porosity,
            setup.rate_parameters,
        )
        destruction = number * mean_resolution_rate
        variance_change = site.nucleated * (
            mean_atoms - DIMER_ATOMS
        ) ** 2 + site.dt * number * (
            mean_capture_rate + 2.0 * capture_covariance - resolution_variance_rate
        )
        # Vacancy absorption of a pore in its Wigner-Seitz cell.
        cell_radius = jnp.cbrt(3.0 / (4.0 * jnp.pi * number))
        radius_ratio = jnp.cbrt(porosity)
        cell_free_fraction = -jnp.expm1(jnp.log(porosity) / 3.0)
        mobility_factor = (
            10.0
            * radius_ratio
            * (1.0 + radius_ratio**3)
            / (
                cell_free_fraction
                * (
                    radius_ratio**5
                    + radius_ratio**4
                    + radius_ratio**3
                    + radius_ratio**2
                    - 4.0 * radius_ratio
                    + 5.0
                )
            )
        )
        packing = gas_atomic_volume * mean_atoms / volume
        free_fraction = self.vacancy_volume * per_pore_vacancies / volume
        pressure = (
            rates.thermal_energy
            * mean_atoms
            / volume
            * carnahan_starling_numerator(packing)
            / free_fraction**3
        )
        equilibrium = 2.0 * self.surface_energy / radius - site.stress
        mobility = (
            2.0
            * jnp.pi
            * rates.vacancy_diffusivity
            * cell_radius
            / (rates.thermal_energy * mobility_factor)
        )
        kinetic_capacity = site.dt * number * mobility
        row_scale = (
            moments.vacancies
            + site.vacancies
            + site.dt * destruction * per_pore_vacancies
            + kinetic_capacity * (jnp.abs(pressure) + jnp.abs(equilibrium))
        )
        vacancy_residual = (
            moments.vacancies
            - site.vacancies
            + site.dt * destruction * per_pore_vacancies
        ) / row_scale - (kinetic_capacity / row_scale) * (pressure - equilibrium)
        return _HbsRootTerms(
            variance_residual=(moments.variance - site.variance - variance_change)
            / scaling.variance,
            vacancy_residual=vacancy_residual,
            trapped=site.dt * (number * mean_capture_rate),
            resolved=site.dt * (number * resolved_atom_rate),
            pore_volume=volume,
            porosity=porosity,
            number_loss=site.dt * mean_resolution_rate,
            excess_atoms=site.pore_gas
            - DIMER_ATOMS * site.pore_number_density
            + site.dt * number * (mean_capture_rate - resolution_covariance),
        )

    def _root_residual(self, site: _HbsSite, setup: _HbsRootSetup, value: Array) -> Array:
        terms = self._root_terms(site, setup, value)
        scaling = setup.scaling
        return jnp.stack(
            [
                value[0] + jnp.log1p(terms.number_loss) / scaling.log_number,
                value[1] - scaling.excess_coordinate(terms.excess_atoms),
                terms.variance_residual,
                terms.vacancy_residual,
            ]
        )

    def _root_admissible(
        self, site: _HbsSite, setup: _HbsRootSetup, value: Array
    ) -> Array:
        """Positive finite moments, conserved gas, open volume and a valid pore shape."""

        scaling = setup.scaling
        moments = scaling.decode(value)
        basic = (
            (value[1] >= 0.0)
            & (value[2] >= 0.0)
            & jnp.all(jnp.isfinite(value))
            & (moments.number > 0.0)
            & jnp.all(jnp.isfinite(jnp.stack(moments)))
            & (moments.atoms >= 0.0)
            & (moments.atoms <= site.total_gas)
            & (scaling.remaining_gas(value) >= 0.0)
            & (moments.vacancies > 0.0)
            & (
                site.rates.gas_atomic_volume * moments.atoms
                + self.vacancy_volume * moments.vacancies
                < 1.0
            )
        )

        def refused(_: None) -> Array:
            return jnp.asarray(False, dtype=jnp.bool_)

        def pore_shape(_: None) -> Array:
            width = moments.variance / moments.number
            shape_valid = jnp.isfinite(width) & (
                (value[1] > 0.0) | (moments.variance == 0.0)
            )

            def valid_shape(_: None) -> Array:
                terms = self._root_terms(site, setup, value)
                return jnp.all(jnp.isfinite(jnp.stack(terms))) & (
                    terms.excess_atoms >= 0.0
                )

            return domain_cond(shape_valid, valid_shape, refused)

        return domain_cond(basic, pore_shape, refused)

    def _commit_root(
        self,
        site: _HbsSite,
        setup: _HbsRootSetup,
        solution: Array,
        diagnostics: VectorLocalRootDiagnostics,
    ) -> HbsPoreUpdate:
        """Project a converged root onto its explicit number/mean update, then coalesce.

        A failed root stays failed: its bounded trial, not an unbounded flux
        extrapolation, remains the rejected candidate state.
        """

        scaling = setup.scaling
        gas_atomic_volume = site.rates.gas_atomic_volume
        endpoint = self._root_terms(site, setup, solution)
        committed = (
            solution.at[0]
            .set(scaling.number_coordinate(endpoint.number_loss))
            .at[1]
            .set(scaling.excess_coordinate(endpoint.excess_atoms))
        )
        commit_admissible = self._root_admissible(site, setup, committed)
        output = jnp.where(diagnostics.converged & commit_admissible, committed, solution)
        moments = scaling.decode(output)
        committed_norm = jnp.sqrt(jnp.sum(self._root_residual(site, setup, output) ** 2))
        terms = self._root_terms(site, setup, output)
        previous_pore_volume = (
            gas_atomic_volume * site.pore_gas + self.vacancy_volume * site.vacancies
        ) / jnp.where(site.pore_number_density > 0.0, site.pore_number_density, 1.0)
        growth = jnp.maximum(terms.pore_volume - previous_pore_volume, 0.0)
        # Source impingement: growth at fixed gas and vacancy volume merges pores.
        interaction = 4.0 * (2.0 - terms.porosity) / (2.0 * (1.0 - terms.porosity) ** 3)
        coalesced = moments.number / (1.0 + interaction * moments.number * growth)
        return HbsPoreUpdate(
            gas=moments.atoms,
            boundary_gas=scaling.remaining_gas(output),
            density=coalesced,
            variance=moments.variance,
            vacancies=moments.vacancies,
            porosity=gas_atomic_volume * moments.atoms
            + self.vacancy_volume * moments.vacancies,
            nucleation=DIMER_ATOMS * site.nucleated,
            trapping=terms.trapped,
            resolution=terms.resolved,
            converged=diagnostics.converged
            & commit_admissible
            & (committed_norm <= self.root.tolerance),
            residual_norm=jnp.maximum(diagnostics.residual_norm, committed_norm),
            condition_estimate=diagnostics.condition_estimate,
            linear_status=diagnostics.linear_solve_status,
            linear_rank=diagnostics.linear_solve_rank,
            conditioning_available=diagnostics.conditioning_available,
        )

    def advance(
        self,
        state: FissionGasSiteState,
        /,
        *,
        arrival: Array,
        fraction: Array,
        temperature: Array,
        fission_rate_density: Array,
        stress: Array,
        time_step: Array,
    ) -> HbsPoreUpdate:
        """Advance every site's HBS pores; inputs carry the site shape of ``state``."""

        shape = state.grain_radius.shape
        site_inputs = (
            state.hbs_pore_gas,
            state.hbs_boundary_gas,
            state.hbs_pore_number_density,
            state.hbs_pore_variance,
            state.hbs_pore_vacancies,
            arrival,
            fraction - state.hbs_fraction,
            fraction,
            temperature,
            fission_rate_density,
            stress,
            time_step,
        )
        result = jax.vmap(self._site_step)(*(value.reshape(-1) for value in site_inputs))
        return jax.tree.map(lambda value: value.reshape(shape), result)


__all__ = [
    "DIMER_ATOMS",
    "HbsPoreUpdate",
    "HighBurnupStructureFormation",
    "HighBurnupStructurePores",
]
