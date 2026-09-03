#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.lattice_boltzmann import LatticeBoltzmannScaling
from .._quantities import cardiovascular_quantity


class HemodynamicsStatus(IntEnum):
    """Fail-closed outcome of one fixed-wall candidate audit."""

    SUCCESS = 0
    NONFINITE = 1
    LOW_MACH_VIOLATION = 2
    MASS_BALANCE_VIOLATION = 3
    MOMENTUM_VIOLATION = 4
    POPULATION_INADMISSIBLE = 5
    DENSITY_INADMISSIBLE = 6
    RHEOLOGY_INVALID = 7
    TERMINAL_BALANCE_VIOLATION = 8
    COLLISION_FAILURE = 9
    PORT_ITERATE_INVALID = 10


class FixedWallScope(StrictModule, NonTrainableState):
    """Machine-readable scope of the claims made by this workflow."""

    wall_motion_supported: bool = eqx.field(static=True)
    fluid_structure_interaction_supported: bool = eqx.field(static=True)
    curved_wall_accuracy_supported: bool = eqx.field(static=True)
    clinical_use_supported: bool = eqx.field(static=True)
    statement: str = eqx.field(static=True)
    scope_id: str = eqx.field(static=True)

    def __init__(self):
        statement = (
            "fixed voxel lumen with stationary halfway bounce-back walls; "
            "numerical hemodynamics qualification only"
        )
        self.wall_motion_supported = False
        self.fluid_structure_interaction_supported = False
        self.curved_wall_accuracy_supported = False
        self.clinical_use_supported = False
        self.statement = statement
        self.scope_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fixed-wall-scope",
                "statement": statement,
                "wall_motion_supported": False,
                "fluid_structure_interaction_supported": False,
                "curved_wall_accuracy_supported": False,
                "clinical_use_supported": False,
            }
        )


class HemodynamicsScaling(StrictModule, NonTrainableState):
    """Explicit ``mm/ms/mg/kPa`` to D3Q19 lattice conversion.

    In the cardiovascular kernel, ``mg/(mm*ms^2)`` is numerically one ``kPa``.
    Consequently the generic LBM pressure scale, ``rho * (dx/dt)^2``, converts
    directly to kPa without a hidden factor.
    """

    cell_size_mm: Array
    time_step_ms: Array
    reference_density_mg_per_mm3: Array
    reference_velocity_mm_per_ms: Array
    maximum_lattice_mach: float = eqx.field(static=True)
    lattice: LatticeBoltzmannScaling
    quantity_spec_ids: tuple[str, ...] = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_size_mm: float,
        time_step_ms: float,
        reference_density_mg_per_mm3: float,
        /,
        *,
        reference_velocity_mm_per_ms: float,
        maximum_lattice_mach: float = 0.1,
    ):
        values = tuple(
            float(value)
            for value in (
                cell_size_mm,
                time_step_ms,
                reference_density_mg_per_mm3,
                reference_velocity_mm_per_ms,
                maximum_lattice_mach,
            )
        )
        dx, dt, density, velocity, mach_limit = values
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Hemodynamics scaling values must be finite and positive.")
        if mach_limit >= 1.0:
            raise ValueError("maximum_lattice_mach must lie in (0, 1).")
        lattice = LatticeBoltzmannScaling(dx, dt, density)
        reference_mach = velocity * dt / dx / np.sqrt(1.0 / 3.0)
        if reference_mach > mach_limit:
            raise ValueError(
                "Reference velocity exceeds the declared weakly-compressible lattice Mach limit."
            )
        quantity_names = (
            "length",
            "time",
            "mass_density",
            "velocity",
            "pressure",
            "volumetric_flow_rate",
            "dynamic_viscosity",
            "strain_rate",
            "mass",
            "power",
        )
        quantity_ids = tuple(
            cardiovascular_quantity(name).quantity_id for name in quantity_names
        )
        self.cell_size_mm = jnp.asarray(dx, dtype=jnp.float64)
        self.time_step_ms = jnp.asarray(dt, dtype=jnp.float64)
        self.reference_density_mg_per_mm3 = jnp.asarray(density, dtype=jnp.float64)
        self.reference_velocity_mm_per_ms = jnp.asarray(velocity, dtype=jnp.float64)
        self.maximum_lattice_mach = mach_limit
        self.lattice = lattice
        self.quantity_spec_ids = quantity_ids
        self.scaling_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-hemodynamics-scaling",
                "cell_size_mm": dx,
                "time_step_ms": dt,
                "reference_density_mg_per_mm3": density,
                "reference_velocity_mm_per_ms": velocity,
                "maximum_lattice_mach": mach_limit,
                "generic_lattice_scaling": lattice.scaling_id,
                "quantities": quantity_ids,
            }
        )

    @property
    def reference_lattice_mach(self) -> float:
        return float(
            self.reference_velocity_mm_per_ms
            * self.time_step_ms
            / self.cell_size_mm
            / jnp.sqrt(jnp.asarray(1.0 / 3.0))
        )

    @property
    def pressure_scale_kpa(self) -> Array:
        speed_scale = self.cell_size_mm / self.time_step_ms
        return self.reference_density_mg_per_mm3 * speed_scale**2

    @property
    def flow_scale_mm3_per_ms(self) -> Array:
        return self.cell_size_mm**3 / self.time_step_ms

    @property
    def mass_scale_mg(self) -> Array:
        return self.reference_density_mg_per_mm3 * self.cell_size_mm**3

    @property
    def momentum_scale_mg_mm_per_ms(self) -> Array:
        return self.mass_scale_mg * self.cell_size_mm / self.time_step_ms

    @property
    def power_scale_mg_mm2_per_ms3(self) -> Array:
        return self.pressure_scale_kpa * self.flow_scale_mm3_per_ms

    def lattice_velocity(self, velocity_mm_per_ms: ArrayLike, /) -> Array:
        return self.lattice.lattice_velocity(velocity_mm_per_ms)

    def physical_velocity(self, velocity_lattice: ArrayLike, /) -> Array:
        return self.lattice.physical_velocity(velocity_lattice)

    def lattice_kinematic_viscosity(self, viscosity_mm2_per_ms: ArrayLike, /) -> Array:
        return self.lattice.lattice_viscosity(viscosity_mm2_per_ms)

    def physical_kinematic_viscosity(self, viscosity_lattice: ArrayLike, /) -> Array:
        return self.lattice.physical_viscosity(viscosity_lattice)

    def lattice_density(self, density_mg_per_mm3: ArrayLike, /) -> Array:
        return self.lattice.lattice_density(density_mg_per_mm3)

    def physical_density(self, density_lattice: ArrayLike, /) -> Array:
        return self.lattice.physical_density(density_lattice)

    def lattice_gauge_pressure(self, pressure_kpa: ArrayLike, /) -> Array:
        pressure = jnp.asarray(pressure_kpa)
        return pressure / self.pressure_scale_kpa.astype(pressure.dtype)

    def physical_gauge_pressure(self, pressure_lattice: ArrayLike, /) -> Array:
        pressure = jnp.asarray(pressure_lattice)
        return pressure * self.pressure_scale_kpa.astype(pressure.dtype)

    def pressure_density(self, pressure_kpa: ArrayLike, /) -> Array:
        pressure_lattice = self.lattice_gauge_pressure(pressure_kpa)
        cs2 = self.lattice.sound_speed_squared.astype(pressure_lattice.dtype)
        return 1.0 + pressure_lattice / cs2

    def density_gauge_pressure(self, density_lattice: ArrayLike, /) -> Array:
        density = jnp.asarray(density_lattice)
        cs2 = self.lattice.sound_speed_squared.astype(density.dtype)
        return self.physical_gauge_pressure(cs2 * (density - 1.0))

    def lattice_flow_rate(self, flow_mm3_per_ms: ArrayLike, /) -> Array:
        flow = jnp.asarray(flow_mm3_per_ms)
        return flow / self.flow_scale_mm3_per_ms.astype(flow.dtype)

    def physical_flow_rate(self, flow_lattice: ArrayLike, /) -> Array:
        flow = jnp.asarray(flow_lattice)
        return flow * self.flow_scale_mm3_per_ms.astype(flow.dtype)

    def lattice_shear_rate(self, shear_rate_per_ms: ArrayLike, /) -> Array:
        rate = jnp.asarray(shear_rate_per_ms)
        return rate * self.time_step_ms.astype(rate.dtype)

    def physical_shear_rate(self, shear_rate_lattice: ArrayLike, /) -> Array:
        rate = jnp.asarray(shear_rate_lattice)
        return rate / self.time_step_ms.astype(rate.dtype)

    def lattice_mass(self, mass_mg: ArrayLike, /) -> Array:
        mass = jnp.asarray(mass_mg)
        return mass / self.mass_scale_mg.astype(mass.dtype)

    def physical_mass(self, mass_lattice: ArrayLike, /) -> Array:
        mass = jnp.asarray(mass_lattice)
        return mass * self.mass_scale_mg.astype(mass.dtype)

    def lattice_momentum(self, momentum_mg_mm_per_ms: ArrayLike, /) -> Array:
        momentum = jnp.asarray(momentum_mg_mm_per_ms)
        return momentum / self.momentum_scale_mg_mm_per_ms.astype(momentum.dtype)

    def physical_momentum(self, momentum_lattice: ArrayLike, /) -> Array:
        momentum = jnp.asarray(momentum_lattice)
        return momentum * self.momentum_scale_mg_mm_per_ms.astype(momentum.dtype)

    def lattice_power(self, power_mg_mm2_per_ms3: ArrayLike, /) -> Array:
        power = jnp.asarray(power_mg_mm2_per_ms3)
        return power / self.power_scale_mg_mm2_per_ms3.astype(power.dtype)

    def physical_power(self, power_lattice: ArrayLike, /) -> Array:
        power = jnp.asarray(power_lattice)
        return power * self.power_scale_mg_mm2_per_ms3.astype(power.dtype)


class HemodynamicsValidityLimits(StrictModule, NonTrainableState):
    """Numerical acceptance envelope for fixed-wall weakly-compressible LBM."""

    maximum_lattice_mach: float = eqx.field(static=True)
    maximum_relative_mass_balance_defect: float = eqx.field(static=True)
    maximum_relative_momentum_change: float = eqx.field(static=True)
    maximum_relative_density_deviation: float = eqx.field(static=True)
    minimum_population: float = eqx.field(static=True)
    minimum_relaxation_rate: float = eqx.field(static=True)
    maximum_relaxation_rate: float = eqx.field(static=True)
    maximum_terminal_flow_relative_defect: float = eqx.field(static=True)
    maximum_terminal_pressure_absolute_defect_kpa: float = eqx.field(static=True)
    maximum_terminal_power_relative_defect: float = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_lattice_mach: float = 0.1,
        maximum_relative_mass_balance_defect: float = 5.0e-3,
        maximum_relative_momentum_change: float = 0.25,
        maximum_relative_density_deviation: float = 0.03,
        minimum_population: float = -1.0e-12,
        minimum_relaxation_rate: float = 0.02,
        maximum_relaxation_rate: float = 1.98,
        maximum_terminal_flow_relative_defect: float = 5.0e-2,
        maximum_terminal_pressure_absolute_defect_kpa: float = 1.0e-3,
        maximum_terminal_power_relative_defect: float = 5.0e-2,
    ):
        values = tuple(
            float(value)
            for value in (
                maximum_lattice_mach,
                maximum_relative_mass_balance_defect,
                maximum_relative_momentum_change,
                maximum_relative_density_deviation,
                minimum_population,
                minimum_relaxation_rate,
                maximum_relaxation_rate,
                maximum_terminal_flow_relative_defect,
                maximum_terminal_pressure_absolute_defect_kpa,
                maximum_terminal_power_relative_defect,
            )
        )
        (
            mach,
            mass,
            momentum,
            density,
            population,
            relaxation_minimum,
            relaxation_maximum,
            terminal_flow,
            terminal_pressure,
            terminal_power,
        ) = values
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Hemodynamics validity limits must be finite.")
        if not 0.0 < mach < 1.0:
            raise ValueError("maximum_lattice_mach must lie in (0, 1).")
        if any(
            value < 0.0
            for value in (
                mass,
                momentum,
                density,
                terminal_flow,
                terminal_pressure,
                terminal_power,
            )
        ):
            raise ValueError("Relative defect limits must be nonnegative.")
        if not 0.0 < relaxation_minimum < relaxation_maximum < 2.0:
            raise ValueError("Relaxation-rate limits must form a subinterval of (0, 2).")
        self.maximum_lattice_mach = mach
        self.maximum_relative_mass_balance_defect = mass
        self.maximum_relative_momentum_change = momentum
        self.maximum_relative_density_deviation = density
        self.minimum_population = population
        self.minimum_relaxation_rate = relaxation_minimum
        self.maximum_relaxation_rate = relaxation_maximum
        self.maximum_terminal_flow_relative_defect = terminal_flow
        self.maximum_terminal_pressure_absolute_defect_kpa = terminal_pressure
        self.maximum_terminal_power_relative_defect = terminal_power
        self.limits_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-hemodynamics-validity-limits",
                "maximum_lattice_mach": mach,
                "maximum_relative_mass_balance_defect": mass,
                "maximum_relative_momentum_change": momentum,
                "maximum_relative_density_deviation": density,
                "minimum_population": population,
                "minimum_relaxation_rate": relaxation_minimum,
                "maximum_relaxation_rate": relaxation_maximum,
                "maximum_terminal_flow_relative_defect": terminal_flow,
                "maximum_terminal_pressure_absolute_defect_kpa": terminal_pressure,
                "maximum_terminal_power_relative_defect": terminal_power,
            }
        )


class FixedWallLumenRegion(StrictModule, NonTrainableState):
    """Immutable voxel classification of a stationary three-dimensional lumen."""

    fluid_mask: Array
    solid_mask: Array
    lumen_name: str = eqx.field(static=True)
    lumen_id: str = eqx.field(static=True)

    def __init__(self, fluid_mask: ArrayLike, /, *, lumen_name: str = "lumen"):
        mask = np.asarray(fluid_mask, dtype=bool)
        name = str(lumen_name)
        if mask.ndim != 3 or not np.any(mask):
            raise ValueError(
                "A fixed-wall lumen requires a nonempty three-dimensional mask."
            )
        if not name:
            raise ValueError("lumen_name must be nonempty.")
        self.fluid_mask = jnp.asarray(mask, dtype=bool)
        self.solid_mask = jnp.asarray(~mask, dtype=bool)
        self.lumen_name = name
        self.lumen_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-fixed-wall-lumen",
                "name": name,
                "fluid_mask": array_tree_fingerprint(mask),
                "wall_motion": "stationary",
                "voxel_wall_rule": "halfway-bounce-back",
            }
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.fluid_mask.shape)


class HemodynamicsEvidence(StrictModule):
    """Complete fail-closed audit attached to a candidate state."""

    status: Array
    successful: Array
    finite: Array
    collision_successful: Array
    port_iterate_admissible: Array
    low_mach: Array
    mass_conservative: Array
    momentum_admissible: Array
    populations_admissible: Array
    density_admissible: Array
    rheology_admissible: Array
    terminal_balance_admissible: Array
    maximum_lattice_mach: Array
    relative_mass_balance_defect: Array
    relative_momentum_change: Array
    minimum_population: Array
    minimum_density_lattice: Array
    maximum_relative_density_deviation: Array
    minimum_relaxation_rate: Array
    maximum_relaxation_rate: Array
    terminal_flow_relative_defect: Array
    terminal_pressure_maximum_absolute_defect_kpa: Array
    terminal_power_relative_defect: Array
    wall_impulse_lattice: Array
    scope_id: str = eqx.field(static=True)


class PoiseuillePipeReference(StrictModule, NonTrainableState):
    """Steady circular-pipe reference in cardiovascular kernel units."""

    radius_mm: Array
    length_mm: Array
    pressure_drop_kpa: Array
    dynamic_viscosity_kpa_ms: Array
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius_mm: float,
        length_mm: float,
        pressure_drop_kpa: float,
        dynamic_viscosity_kpa_ms: float,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                radius_mm,
                length_mm,
                pressure_drop_kpa,
                dynamic_viscosity_kpa_ms,
            )
        )
        radius, length, pressure, viscosity = values
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Poiseuille parameters must be finite and positive.")
        self.radius_mm = jnp.asarray(radius, dtype=jnp.float64)
        self.length_mm = jnp.asarray(length, dtype=jnp.float64)
        self.pressure_drop_kpa = jnp.asarray(pressure, dtype=jnp.float64)
        self.dynamic_viscosity_kpa_ms = jnp.asarray(viscosity, dtype=jnp.float64)
        self.reference_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-poiseuille-pipe-reference",
                "radius_mm": radius,
                "length_mm": length,
                "pressure_drop_kpa": pressure,
                "dynamic_viscosity_kpa_ms": viscosity,
            }
        )

    @property
    def flow_rate_mm3_per_ms(self) -> Array:
        return (
            jnp.pi
            * self.radius_mm**4
            * self.pressure_drop_kpa
            / (8.0 * self.dynamic_viscosity_kpa_ms * self.length_mm)
        )

    @property
    def centerline_velocity_mm_per_ms(self) -> Array:
        return (
            self.pressure_drop_kpa
            * self.radius_mm**2
            / (4.0 * self.dynamic_viscosity_kpa_ms * self.length_mm)
        )

    def axial_velocity(self, radius_mm: ArrayLike, /) -> Array:
        radius = jnp.asarray(radius_mm)
        normalized = radius / self.radius_mm.astype(radius.dtype)
        velocity = self.centerline_velocity_mm_per_ms.astype(radius.dtype) * (
            1.0 - normalized**2
        )
        invalid = ~jnp.all(
            jnp.isfinite(radius) & (radius >= 0.0) & (radius <= self.radius_mm)
        )
        return eqx.error_if(
            velocity,
            invalid,
            "Poiseuille sample radii must lie inside the pipe.",
        )


def _complex_bessel_j0(value: np.ndarray, /, *, terms: int = 96) -> np.ndarray:
    """Evaluate complex J0 with a deterministic power series for references."""

    values = np.asarray(value, dtype=np.complex128)
    total = np.ones_like(values)
    term = np.ones_like(values)
    quarter_square = -0.25 * values * values
    for order in range(1, terms + 1):
        term = term * quarter_square / float(order * order)
        total = total + term
    return total


class WomersleyPipeReference(StrictModule, NonTrainableState):
    """Harmonic circular-pipe reference with an ``exp(i*omega*t)`` convention."""

    radius_mm: Array
    pressure_gradient_amplitude_kpa_per_mm: Array
    dynamic_viscosity_kpa_ms: Array
    density_mg_per_mm3: Array
    angular_frequency_per_ms: Array
    phase_radians: Array
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius_mm: float,
        pressure_gradient_amplitude_kpa_per_mm: float,
        dynamic_viscosity_kpa_ms: float,
        density_mg_per_mm3: float,
        angular_frequency_per_ms: float,
        /,
        *,
        phase_radians: float = 0.0,
    ):
        values = tuple(
            float(value)
            for value in (
                radius_mm,
                pressure_gradient_amplitude_kpa_per_mm,
                dynamic_viscosity_kpa_ms,
                density_mg_per_mm3,
                angular_frequency_per_ms,
            )
        )
        radius, gradient, viscosity, density, frequency = values
        phase = float(phase_radians)
        if any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ) or not np.isfinite(phase):
            raise ValueError("Womersley parameters must be finite and positive.")
        dtype = jnp.float64
        self.radius_mm = jnp.asarray(radius, dtype=dtype)
        self.pressure_gradient_amplitude_kpa_per_mm = jnp.asarray(gradient, dtype=dtype)
        self.dynamic_viscosity_kpa_ms = jnp.asarray(viscosity, dtype=dtype)
        self.density_mg_per_mm3 = jnp.asarray(density, dtype=dtype)
        self.angular_frequency_per_ms = jnp.asarray(frequency, dtype=dtype)
        self.phase_radians = jnp.asarray(phase, dtype=dtype)
        self.reference_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-womersley-pipe-reference",
                "radius_mm": radius,
                "pressure_gradient_amplitude_kpa_per_mm": gradient,
                "dynamic_viscosity_kpa_ms": viscosity,
                "density_mg_per_mm3": density,
                "angular_frequency_per_ms": frequency,
                "phase_radians": phase,
                "phasor_convention": "exp(+i*omega*t)",
            }
        )

    @property
    def womersley_number(self) -> float:
        return float(
            self.radius_mm
            * jnp.sqrt(
                self.angular_frequency_per_ms
                * self.density_mg_per_mm3
                / self.dynamic_viscosity_kpa_ms
            )
        )

    def axial_velocity(self, radius_mm: ArrayLike, time_ms: ArrayLike, /) -> Array:
        radius = np.asarray(radius_mm, dtype=float)
        time = np.asarray(time_ms, dtype=float)
        pipe_radius = float(self.radius_mm)
        if (
            np.any(~np.isfinite(radius))
            or np.any(radius < 0.0)
            or np.any(radius > pipe_radius)
            or np.any(~np.isfinite(time))
        ):
            raise ValueError("Womersley samples require finite in-lumen radii and times.")
        omega = float(self.angular_frequency_per_ms)
        density = float(self.density_mg_per_mm3)
        viscosity = float(self.dynamic_viscosity_kpa_ms)
        gradient = float(self.pressure_gradient_amplitude_kpa_per_mm)
        beta = pipe_radius * np.sqrt(-1j * omega * density / viscosity)
        radial_ratio = _complex_bessel_j0(
            beta * radius / pipe_radius
        ) / _complex_bessel_j0(np.asarray(beta))
        amplitude = gradient / (1j * omega * density) * (1.0 - radial_ratio)
        phase = np.exp(1j * (omega * time + float(self.phase_radians)))
        return jnp.asarray(np.real(amplitude * phase), dtype=jnp.float64)


class LBMMACComparisonEvidence(StrictModule):
    """Co-registered comparison of independent fixed-wall LBM and MAC fields."""

    velocity_relative_l2: Array
    pressure_relative_l2: Array
    velocity_maximum_absolute: Array
    pressure_maximum_absolute: Array
    finite: Array
    passed: Array
    velocity_relative_tolerance: Array
    pressure_relative_tolerance: Array
    lbm_route_id: str = eqx.field(static=True)
    mac_route_id: str = eqx.field(static=True)


def compare_lbm_mac(
    lbm_velocity_mm_per_ms: ArrayLike,
    mac_velocity_mm_per_ms: ArrayLike,
    lbm_pressure_kpa: ArrayLike,
    mac_pressure_kpa: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    velocity_relative_tolerance: float,
    pressure_relative_tolerance: float,
) -> LBMMACComparisonEvidence:
    """Compare already co-registered results without conflating solver routes."""

    lbm_velocity = jnp.asarray(lbm_velocity_mm_per_ms)
    mac_velocity = jnp.asarray(mac_velocity_mm_per_ms, dtype=lbm_velocity.dtype)
    lbm_pressure = jnp.asarray(lbm_pressure_kpa, dtype=lbm_velocity.dtype)
    mac_pressure = jnp.asarray(mac_pressure_kpa, dtype=lbm_velocity.dtype)
    weight = jnp.asarray(weights, dtype=lbm_velocity.dtype)
    if lbm_velocity.shape != mac_velocity.shape or lbm_velocity.ndim < 2:
        raise ValueError("LBM and MAC velocity arrays must share (..., component) shape.")
    sample_shape = lbm_velocity.shape[:-1]
    if lbm_pressure.shape != sample_shape or mac_pressure.shape != sample_shape:
        raise ValueError("Pressure arrays must match the velocity sample axes.")
    if weight.shape != sample_shape:
        raise ValueError("Comparison weights must match the sample axes.")
    velocity_tolerance = float(velocity_relative_tolerance)
    pressure_tolerance = float(pressure_relative_tolerance)
    if (
        not np.isfinite(velocity_tolerance)
        or not np.isfinite(pressure_tolerance)
        or velocity_tolerance < 0.0
        or pressure_tolerance < 0.0
    ):
        raise ValueError("Comparison tolerances must be finite and nonnegative.")
    positive_weight = jnp.all(jnp.isfinite(weight) & (weight >= 0.0)) & (
        jnp.sum(weight) > 0.0
    )
    velocity_error_squared = jnp.sum((lbm_velocity - mac_velocity) ** 2, axis=-1)
    velocity_reference_squared = jnp.sum(mac_velocity**2, axis=-1)
    pressure_error_squared = (lbm_pressure - mac_pressure) ** 2
    pressure_reference_squared = mac_pressure**2
    norm_floor = jnp.sqrt(jnp.finfo(lbm_velocity.dtype).eps * jnp.sum(weight))
    velocity_relative = jnp.sqrt(jnp.sum(weight * velocity_error_squared)) / jnp.maximum(
        jnp.sqrt(jnp.sum(weight * velocity_reference_squared)), norm_floor
    )
    pressure_relative = jnp.sqrt(jnp.sum(weight * pressure_error_squared)) / jnp.maximum(
        jnp.sqrt(jnp.sum(weight * pressure_reference_squared)), norm_floor
    )
    finite = (
        positive_weight
        & jnp.all(jnp.isfinite(lbm_velocity))
        & jnp.all(jnp.isfinite(mac_velocity))
        & jnp.all(jnp.isfinite(lbm_pressure))
        & jnp.all(jnp.isfinite(mac_pressure))
        & jnp.isfinite(velocity_relative)
        & jnp.isfinite(pressure_relative)
    )
    passed = (
        finite
        & (velocity_relative <= velocity_tolerance)
        & (pressure_relative <= pressure_tolerance)
    )
    return LBMMACComparisonEvidence(
        velocity_relative_l2=velocity_relative,
        pressure_relative_l2=pressure_relative,
        velocity_maximum_absolute=jnp.max(jnp.sqrt(velocity_error_squared)),
        pressure_maximum_absolute=jnp.max(jnp.abs(lbm_pressure - mac_pressure)),
        finite=finite,
        passed=passed,
        velocity_relative_tolerance=jnp.asarray(
            velocity_tolerance, dtype=lbm_velocity.dtype
        ),
        pressure_relative_tolerance=jnp.asarray(
            pressure_tolerance, dtype=lbm_velocity.dtype
        ),
        lbm_route_id="fixed-wall-d3q19-lbm",
        mac_route_id="fixed-wall-staggered-mac",
    )


__all__ = [
    "compare_lbm_mac",
    "FixedWallLumenRegion",
    "FixedWallScope",
    "HemodynamicsEvidence",
    "HemodynamicsScaling",
    "HemodynamicsStatus",
    "HemodynamicsValidityLimits",
    "LBMMACComparisonEvidence",
    "PoiseuillePipeReference",
    "WomersleyPipeReference",
]
