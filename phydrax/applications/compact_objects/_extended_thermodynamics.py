#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_ENSEMBLES = ("microcanonical", "canonical-charge", "grand-canonical")
_THEORY = "four-dimensional Einstein--Maxwell theory with Einstein--Hilbert gravity"
_ENTROPY_THEORY = "four-dimensional Einstein--Hilbert gravity"


class EinsteinWaldEntropyResult(StrictModule):
    horizon_area: Array
    geometric_entropy: Array
    entropy: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    theory: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    generalized_entropy_included: bool = eqx.field(static=True)
    island_prescription_included: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class EinsteinWaldEntropyPlan(StrictModule, NonTrainableState):
    """Einstein--Hilbert Wald entropy, exactly equal to the area entropy.

    This contract is classical and stationary.  It deliberately has no
    higher-curvature, matter-entanglement, generalized-entropy, or island term.
    """

    scale: RelativityScaleContract
    plan_id: str = eqx.field(static=True)

    def __init__(self, scale: RelativityScaleContract, /):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError("Wald-equivalent entropy requires explicit hbar and k_B.")
        self.scale = scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "einstein-hilbert-wald-equivalent-entropy",
                "scale": scale.scale_id,
                "scope": "classical-stationary-bifurcate-killing-horizon",
                "higher_curvature": False,
                "generalized_entropy": False,
                "island_prescription": False,
            }
        )

    def evaluate(self, horizon_area: ArrayLike, /) -> EinsteinWaldEntropyResult:
        area = jnp.asarray(horizon_area)
        geometric_entropy = 0.25 * area
        entropy = self.scale.area_to_entropy(area)
        finite = jnp.all(jnp.isfinite(area)) & jnp.all(jnp.isfinite(entropy))
        converged = finite
        physically_valid = jnp.all(area >= 0.0)
        qualified = finite & converged & physically_valid
        derivative_valid = qualified
        status = jnp.where(~finite, 2, jnp.where(~physically_valid, 1, 0)).astype(
            jnp.int32
        )
        return EinsteinWaldEntropyResult(
            area,
            geometric_entropy,
            entropy,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            _ENTROPY_THEORY,
            "Einstein-Hilbert classical stationary bifurcate Killing horizons; minimally coupled matter",
            False,
            False,
            self.plan_id,
        )


class KerrNewmanThermodynamics(StrictModule):
    geometric_mass: Array
    geometric_angular_momentum: Array
    geometric_charge: Array
    horizon_radii: Array
    horizon_area: Array
    geometric_entropy: Array
    entropy: Array
    surface_gravity: Array
    hawking_temperature: Array
    angular_velocity: Array
    electric_potential: Array
    smarr_residual: Array
    finite: Array
    converged: Array
    extremal: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    branch_status: Array
    status: Array
    ensemble: str = eqx.field(static=True)
    asymptotics: str = eqx.field(static=True)
    theory: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class KerrNewmanThermodynamicsPlan(StrictModule, NonTrainableState):
    """Asymptotically flat Kerr--Newman equilibrium thermodynamics."""

    geometric_mass: Array
    specific_angular_momentum: Array
    geometric_charge: Array
    scale: RelativityScaleContract
    ensemble: str = eqx.field(static=True)
    extremality_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometric_mass,
        specific_angular_momentum,
        geometric_charge,
        scale: RelativityScaleContract,
        /,
        *,
        ensemble="microcanonical",
        extremality_tolerance=1.0e-8,
        residual_tolerance=1.0e-6,
    ):
        mass = float(np.asarray(geometric_mass))
        spin = float(np.asarray(specific_angular_momentum))
        charge = float(np.asarray(geometric_charge))
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError(
                "Thermal observables require explicitly declared hbar and k_B."
            )
        if not np.all(np.isfinite((mass, spin, charge))) or mass <= 0.0:
            raise ValueError(
                "Kerr--Newman geometric parameters must be finite with M > 0."
            )
        ensemble_value = _validate_ensemble(ensemble)
        _validate_tolerances(extremality_tolerance, residual_tolerance)
        self.geometric_mass = jnp.asarray(mass)
        self.specific_angular_momentum = jnp.asarray(spin)
        self.geometric_charge = jnp.asarray(charge)
        self.scale = scale
        self.ensemble = ensemble_value
        self.extremality_tolerance = float(extremality_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "asymptotically-flat-kerr-newman-thermodynamics",
                "geometric_mass": mass,
                "specific_angular_momentum": spin,
                "geometric_charge": charge,
                "scale": scale.scale_id,
                "ensemble": ensemble_value,
                "extremality_tolerance": float(extremality_tolerance),
                "residual_tolerance": float(residual_tolerance),
            }
        )

    def evaluate(self) -> KerrNewmanThermodynamics:
        discriminant = (
            self.geometric_mass**2
            - self.specific_angular_momentum**2
            - self.geometric_charge**2
        )
        root = jnp.sqrt(discriminant)
        inner = self.geometric_mass - root
        outer = self.geometric_mass + root
        area_factor = outer**2 + self.specific_angular_momentum**2
        area = 4.0 * jnp.pi * area_factor
        geometric_entropy = 0.25 * area
        entropy = self.scale.area_to_entropy(area)
        surface_gravity = root / area_factor
        physical_surface_gravity = surface_gravity * float(self.scale.speed_of_light**2)
        hawking_temperature = self.scale.surface_gravity_to_temperature(
            physical_surface_gravity
        )
        angular_velocity = self.specific_angular_momentum / area_factor
        electric_potential = self.geometric_charge * outer / area_factor
        angular_momentum = self.geometric_mass * self.specific_angular_momentum
        smarr_residual = self.geometric_mass - (
            2.0 * surface_gravity / (2.0 * jnp.pi) * geometric_entropy
            + 2.0 * angular_velocity * angular_momentum
            + electric_potential * self.geometric_charge
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (
                        outer,
                        area,
                        entropy,
                        surface_gravity,
                        hawking_temperature,
                        angular_velocity,
                        electric_potential,
                        smarr_residual,
                    )
                )
            )
        )
        physically_valid = discriminant >= 0.0
        extremal = physically_valid & (discriminant <= self.extremality_tolerance)
        residual_ok = jnp.abs(smarr_residual) <= self.residual_tolerance * jnp.maximum(
            self.geometric_mass, 1.0
        )
        converged = finite
        qualified = finite & converged & physically_valid & residual_ok
        derivative_valid = qualified & ~extremal
        branch_status = jnp.where(~physically_valid, 2, jnp.where(extremal, 1, 0)).astype(
            jnp.int32
        )
        status = jnp.where(
            ~finite,
            3,
            jnp.where(~physically_valid, 2, jnp.where(~residual_ok, 1, 0)),
        ).astype(jnp.int32)
        return KerrNewmanThermodynamics(
            self.geometric_mass,
            angular_momentum,
            self.geometric_charge,
            jnp.asarray((inner, outer)),
            area,
            geometric_entropy,
            entropy,
            surface_gravity,
            hawking_temperature,
            angular_velocity,
            electric_potential,
            smarr_residual,
            finite,
            converged,
            extremal,
            physically_valid,
            qualified,
            derivative_valid,
            branch_status,
            status,
            self.ensemble,
            "asymptotically flat",
            _THEORY,
            self.plan_id,
        )


class KerrNewmanAdSThermodynamics(StrictModule):
    horizon_radius: Array
    geometric_mass_enthalpy: Array
    geometric_angular_momentum: Array
    geometric_charge: Array
    horizon_area: Array
    geometric_entropy: Array
    entropy: Array
    pressure: Array
    thermodynamic_volume: Array
    temperature: Array
    hawking_temperature: Array
    surface_gravity: Array
    angular_velocity: Array
    electric_potential: Array
    enthalpy_constraint_residual: Array
    temperature_constraint_residual: Array
    smarr_residual: Array
    finite: Array
    converged: Array
    extremal: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    branch_status: Array
    status: Array
    ensemble: str = eqx.field(static=True)
    asymptotics: str = eqx.field(static=True)
    theory: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class KerrNewmanAdSThermodynamicsPlan(StrictModule, NonTrainableState):
    """Four-dimensional Kerr--Newman--AdS extended thermodynamics.

    Pressure is ``3/(8 pi L^2)`` in geometrized variables, mass is enthalpy,
    and volume is its exact pressure derivative at fixed entropy, angular
    momentum, and charge.  No higher-curvature or holographic entropy term is
    present.
    """

    horizon_radius: Array
    specific_angular_momentum: Array
    charge_parameter: Array
    ads_radius: Array
    scale: RelativityScaleContract
    ensemble: str = eqx.field(static=True)
    extremality_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        horizon_radius,
        specific_angular_momentum,
        charge_parameter,
        ads_radius,
        scale: RelativityScaleContract,
        /,
        *,
        ensemble="canonical-charge",
        extremality_tolerance=1.0e-8,
        residual_tolerance=1.0e-6,
    ):
        radius = float(np.asarray(horizon_radius))
        spin = float(np.asarray(specific_angular_momentum))
        charge = float(np.asarray(charge_parameter))
        ads = float(np.asarray(ads_radius))
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError(
                "Thermal observables require explicitly declared hbar and k_B."
            )
        if (
            not np.all(np.isfinite((radius, spin, charge, ads)))
            or radius <= 0.0
            or ads <= 0.0
            or abs(spin) >= ads
        ):
            raise ValueError("Kerr--Newman--AdS requires r_+ > 0, L > 0, and |a| < L.")
        ensemble_value = _validate_ensemble(ensemble)
        _validate_tolerances(extremality_tolerance, residual_tolerance)
        self.horizon_radius = jnp.asarray(radius)
        self.specific_angular_momentum = jnp.asarray(spin)
        self.charge_parameter = jnp.asarray(charge)
        self.ads_radius = jnp.asarray(ads)
        self.scale = scale
        self.ensemble = ensemble_value
        self.extremality_tolerance = float(extremality_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kerr-newman-ads-extended-thermodynamics",
                "horizon_radius": radius,
                "specific_angular_momentum": spin,
                "charge_parameter": charge,
                "ads_radius": ads,
                "scale": scale.scale_id,
                "ensemble": ensemble_value,
                "extremality_tolerance": float(extremality_tolerance),
                "residual_tolerance": float(residual_tolerance),
            }
        )

    def evaluate(self) -> KerrNewmanAdSThermodynamics:
        radius = self.horizon_radius
        spin = self.specific_angular_momentum
        charge_parameter = self.charge_parameter
        ads = self.ads_radius
        xi = 1.0 - spin**2 / ads**2
        mass_parameter = (
            (radius**2 + spin**2) * (1.0 + radius**2 / ads**2) + charge_parameter**2
        ) / (2.0 * radius)
        parameter_mass = mass_parameter / xi**2
        angular_momentum = spin * mass_parameter / xi**2
        charge = charge_parameter / xi
        area = 4.0 * jnp.pi * (radius**2 + spin**2) / xi
        geometric_entropy = 0.25 * area
        entropy = self.scale.area_to_entropy(area)
        pressure = 3.0 / (8.0 * jnp.pi * ads**2)

        enthalpy, temperature, angular_velocity, electric_potential, volume = (
            _ads_enthalpy_derivatives(
                geometric_entropy, pressure, angular_momentum, charge
            )
        )
        horizon_temperature = (
            radius**2
            - spin**2
            - charge_parameter**2
            + (3.0 * radius**4 + spin**2 * radius**2) / ads**2
        ) / (4.0 * jnp.pi * radius * (radius**2 + spin**2))
        surface_gravity = 2.0 * jnp.pi * temperature
        physical_surface_gravity = surface_gravity * float(self.scale.speed_of_light**2)
        hawking_temperature = self.scale.surface_gravity_to_temperature(
            physical_surface_gravity
        )
        enthalpy_residual = enthalpy - parameter_mass
        temperature_residual = temperature - horizon_temperature
        smarr_residual = enthalpy - (
            2.0 * temperature * geometric_entropy
            + 2.0 * angular_velocity * angular_momentum
            + electric_potential * charge
            - 2.0 * pressure * volume
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (
                        enthalpy,
                        angular_momentum,
                        charge,
                        area,
                        entropy,
                        pressure,
                        volume,
                        temperature,
                        hawking_temperature,
                        angular_velocity,
                        electric_potential,
                        enthalpy_residual,
                        temperature_residual,
                        smarr_residual,
                    )
                )
            )
        )
        extremal = jnp.abs(temperature) <= self.extremality_tolerance
        physically_valid = (xi > 0.0) & (temperature >= -self.extremality_tolerance)
        scale = jnp.maximum(jnp.abs(enthalpy), 1.0)
        residual_ok = (
            (jnp.abs(enthalpy_residual) <= self.residual_tolerance * scale)
            & (jnp.abs(temperature_residual) <= self.residual_tolerance)
            & (jnp.abs(smarr_residual) <= self.residual_tolerance * scale)
        )
        converged = residual_ok
        qualified = finite & converged & physically_valid
        derivative_valid = qualified & ~extremal
        branch_status = jnp.where(~physically_valid, 2, jnp.where(extremal, 1, 0)).astype(
            jnp.int32
        )
        status = jnp.where(
            ~finite,
            3,
            jnp.where(~physically_valid, 2, jnp.where(~residual_ok, 1, 0)),
        ).astype(jnp.int32)
        return KerrNewmanAdSThermodynamics(
            radius,
            enthalpy,
            angular_momentum,
            charge,
            area,
            geometric_entropy,
            entropy,
            pressure,
            volume,
            temperature,
            hawking_temperature,
            surface_gravity,
            angular_velocity,
            electric_potential,
            enthalpy_residual,
            temperature_residual,
            smarr_residual,
            finite,
            converged,
            extremal,
            physically_valid,
            qualified,
            derivative_valid,
            branch_status,
            status,
            self.ensemble,
            "asymptotically anti-de Sitter",
            _THEORY,
            self.plan_id,
        )


class CavityThermodynamics(StrictModule):
    horizon_radius: Array
    cavity_radius: Array
    geometric_mass: Array
    geometric_charge: Array
    horizon_area: Array
    geometric_entropy: Array
    entropy: Array
    wall_redshift: Array
    geometric_quasilocal_energy: Array
    local_temperature: Array
    local_hawking_temperature: Array
    electric_potential: Array
    wall_pressure: Array
    fixed_charge_heat_capacity: Array
    radial_first_law_residual: Array
    charge_first_law_residual: Array
    finite: Array
    converged: Array
    stable_branch: Array
    singular_response: Array
    extremal: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    branch_status: Array
    status: Array
    ensemble: str = eqx.field(static=True)
    boundary_condition: str = eqx.field(static=True)
    theory: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ReissnerNordstromCavityPlan(StrictModule, NonTrainableState):
    """Charged static black hole in a finite spherical Dirichlet cavity.

    Energy is the flat-subtracted Brown--York quasilocal energy and the local
    temperature is the Tolman-redshifted Hawking temperature at the wall.
    The result classifies the fixed-charge heat-capacity branch in the
    canonical-charge ensemble; it is not an asymptotic-AdS pressure ensemble.
    """

    horizon_radius: Array
    geometric_charge: Array
    cavity_radius: Array
    scale: RelativityScaleContract
    ensemble: str = eqx.field(static=True)
    extremality_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        horizon_radius,
        geometric_charge,
        cavity_radius,
        scale: RelativityScaleContract,
        /,
        *,
        ensemble="canonical-charge",
        extremality_tolerance=1.0e-8,
        residual_tolerance=1.0e-6,
    ):
        radius = float(np.asarray(horizon_radius))
        charge = float(np.asarray(geometric_charge))
        wall = float(np.asarray(cavity_radius))
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not scale.quantum_constants_explicit:
            raise ValueError(
                "Thermal observables require explicitly declared hbar and k_B."
            )
        if (
            not np.all(np.isfinite((radius, charge, wall)))
            or radius <= 0.0
            or wall <= 0.0
        ):
            raise ValueError("Cavity radii must be positive and all inputs finite.")
        ensemble_value = _validate_ensemble(ensemble)
        if ensemble_value != "canonical-charge":
            raise ValueError(
                "The implemented finite-wall response is the canonical-charge ensemble."
            )
        _validate_tolerances(extremality_tolerance, residual_tolerance)
        self.horizon_radius = jnp.asarray(radius)
        self.geometric_charge = jnp.asarray(charge)
        self.cavity_radius = jnp.asarray(wall)
        self.scale = scale
        self.ensemble = ensemble_value
        self.extremality_tolerance = float(extremality_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reissner-nordstrom-dirichlet-cavity-thermodynamics",
                "horizon_radius": radius,
                "geometric_charge": charge,
                "cavity_radius": wall,
                "scale": scale.scale_id,
                "ensemble": ensemble_value,
                "extremality_tolerance": float(extremality_tolerance),
                "residual_tolerance": float(residual_tolerance),
            }
        )

    def evaluate(self) -> CavityThermodynamics:
        radius = self.horizon_radius
        charge = self.geometric_charge
        wall = self.cavity_radius
        mass = (radius**2 + charge**2) / (2.0 * radius)
        wall_lapse_squared = 1.0 - 2.0 * mass / wall + charge**2 / wall**2
        wall_redshift = jnp.sqrt(wall_lapse_squared)
        area = 4.0 * jnp.pi * radius**2
        geometric_entropy = 0.25 * area
        entropy = self.scale.area_to_entropy(area)
        horizon_temperature = (radius**2 - charge**2) / (4.0 * jnp.pi * radius**3)
        local_temperature = horizon_temperature / wall_redshift
        local_surface_gravity = 2.0 * jnp.pi * local_temperature
        local_hawking_temperature = self.scale.surface_gravity_to_temperature(
            local_surface_gravity * float(self.scale.speed_of_light**2)
        )
        quasilocal_energy = wall * (1.0 - wall_redshift)
        electric_potential = (charge / radius - charge / wall) / wall_redshift
        lapse_wall_derivative = 2.0 * mass / wall**2 - 2.0 * charge**2 / wall**3
        energy_wall_derivative = (
            1.0 - wall_redshift - wall * lapse_wall_derivative / (2.0 * wall_redshift)
        )
        wall_pressure = -energy_wall_derivative / (8.0 * jnp.pi * wall)

        mass_radius_derivative = (radius**2 - charge**2) / (2.0 * radius**2)
        lapse_radius_derivative = -2.0 * mass_radius_derivative / wall
        horizon_temperature_derivative = (-(radius**2) + 3.0 * charge**2) / (
            4.0 * jnp.pi * radius**4
        )
        local_temperature_derivative = (
            horizon_temperature_derivative / wall_redshift
            - 0.5 * horizon_temperature * lapse_radius_derivative / wall_redshift**3
        )
        energy_radius_derivative = mass_radius_derivative / wall_redshift
        heat_capacity = energy_radius_derivative / local_temperature_derivative
        entropy_radius_derivative = 2.0 * jnp.pi * radius
        radial_first_law_residual = energy_radius_derivative - (
            local_temperature * entropy_radius_derivative
        )
        mass_charge_derivative = charge / radius
        lapse_charge_derivative = (
            -2.0 * mass_charge_derivative / wall + 2.0 * charge / wall**2
        )
        energy_charge_derivative = -0.5 * wall * lapse_charge_derivative / wall_redshift
        charge_first_law_residual = energy_charge_derivative - electric_potential

        finite = jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (
                        mass,
                        entropy,
                        wall_redshift,
                        quasilocal_energy,
                        local_temperature,
                        local_hawking_temperature,
                        electric_potential,
                        wall_pressure,
                        heat_capacity,
                        radial_first_law_residual,
                        charge_first_law_residual,
                    )
                )
            )
        )
        extremal = jnp.abs(radius**2 - charge**2) <= self.extremality_tolerance
        physically_valid = (
            (wall > radius)
            & (jnp.abs(charge) <= radius + self.extremality_tolerance)
            & (wall_lapse_squared > 0.0)
        )
        singular_response = jnp.abs(local_temperature_derivative) <= 1.0e-30
        stable_branch = (
            physically_valid
            & finite
            & (heat_capacity > 0.0)
            & ~extremal
            & ~singular_response
        )
        scale = jnp.maximum(jnp.abs(quasilocal_energy), 1.0)
        residual_ok = (
            jnp.abs(radial_first_law_residual) <= self.residual_tolerance * scale
        ) & (jnp.abs(charge_first_law_residual) <= self.residual_tolerance * scale)
        converged = residual_ok
        qualified = finite & converged & physically_valid
        derivative_valid = (
            qualified & ~extremal & (jnp.abs(local_temperature_derivative) > 1.0e-30)
        )
        branch_status = jnp.where(
            ~physically_valid,
            3,
            jnp.where(
                extremal,
                2,
                jnp.where(singular_response, 4, jnp.where(stable_branch, 0, 1)),
            ),
        ).astype(jnp.int32)
        status = jnp.where(
            ~finite,
            3,
            jnp.where(~physically_valid, 2, jnp.where(~residual_ok, 1, 0)),
        ).astype(jnp.int32)
        return CavityThermodynamics(
            radius,
            wall,
            mass,
            charge,
            area,
            geometric_entropy,
            entropy,
            wall_redshift,
            quasilocal_energy,
            local_temperature,
            local_hawking_temperature,
            electric_potential,
            wall_pressure,
            heat_capacity,
            radial_first_law_residual,
            charge_first_law_residual,
            finite,
            converged,
            stable_branch,
            singular_response,
            extremal,
            physically_valid,
            qualified,
            derivative_valid,
            branch_status,
            status,
            self.ensemble,
            "finite spherical Dirichlet wall with flat Brown--York subtraction",
            _THEORY,
            self.plan_id,
        )


def _ads_enthalpy_derivatives(entropy, pressure, angular_momentum, charge):
    pi = jnp.pi
    enthalpy_term = entropy + pi * charge**2 + 8.0 * pressure * entropy**2 / 3.0
    rotation_term = 1.0 + 8.0 * pressure * entropy / 3.0
    numerator = enthalpy_term**2 + 4.0 * pi**2 * rotation_term * angular_momentum**2
    mass_squared = numerator / (4.0 * pi * entropy)
    mass = jnp.sqrt(mass_squared)

    enthalpy_entropy_derivative = 1.0 + 16.0 * pressure * entropy / 3.0
    rotation_entropy_derivative = 8.0 * pressure / 3.0
    numerator_entropy_derivative = (
        2.0 * enthalpy_term * enthalpy_entropy_derivative
        + 4.0 * pi**2 * angular_momentum**2 * rotation_entropy_derivative
    )
    mass_squared_entropy_derivative = (
        numerator_entropy_derivative * entropy - numerator
    ) / (4.0 * pi * entropy**2)
    temperature = mass_squared_entropy_derivative / (2.0 * mass)
    angular_velocity = pi * rotation_term * angular_momentum / (mass * entropy)
    electric_potential = enthalpy_term * charge / (2.0 * mass * entropy)

    enthalpy_pressure_derivative = 8.0 * entropy**2 / 3.0
    rotation_pressure_derivative = 8.0 * entropy / 3.0
    numerator_pressure_derivative = (
        2.0 * enthalpy_term * enthalpy_pressure_derivative
        + 4.0 * pi**2 * angular_momentum**2 * rotation_pressure_derivative
    )
    mass_squared_pressure_derivative = numerator_pressure_derivative / (
        4.0 * pi * entropy
    )
    volume = mass_squared_pressure_derivative / (2.0 * mass)
    return mass, temperature, angular_velocity, electric_potential, volume


def _validate_ensemble(ensemble):
    value = str(ensemble).strip().lower()
    if value not in _ENSEMBLES:
        raise ValueError(
            "ensemble must be microcanonical, canonical-charge, or grand-canonical."
        )
    return value


def _validate_tolerances(extremality_tolerance, residual_tolerance):
    if (
        not np.isfinite(extremality_tolerance)
        or extremality_tolerance <= 0.0
        or not np.isfinite(residual_tolerance)
        or residual_tolerance <= 0.0
    ):
        raise ValueError("Thermodynamic tolerances must be finite and positive.")


__all__ = [
    "CavityThermodynamics",
    "EinsteinWaldEntropyPlan",
    "EinsteinWaldEntropyResult",
    "KerrNewmanAdSThermodynamics",
    "KerrNewmanAdSThermodynamicsPlan",
    "KerrNewmanThermodynamics",
    "KerrNewmanThermodynamicsPlan",
    "ReissnerNordstromCavityPlan",
]
