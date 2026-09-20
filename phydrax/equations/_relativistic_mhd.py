#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._spacetime_conventions import RelativityConvention
from ._relativistic_eos import AbstractRelativisticEOS, GammaLawEOS
from ._relativistic_hydrodynamics import (
    valencia_geometric_source_from_projection,
    ValenciaGeometrySource,
)


class ValenciaRecoveryStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_CONSERVED = 1
    GEOMETRY_INVALID = 2
    ROOT_NOT_BRACKETED = 3
    ROOT_NOT_CONVERGED = 4
    SUPERLUMINAL = 5
    DENSITY_FLOOR = 6
    PRESSURE_FLOOR = 7
    EOS_INVALID = 8
    MAGNETIZATION_LIMIT = 9


class ValenciaPrimitiveRecovery(StrictModule):
    """Bounded conserved-to-primitive result with separate scientific evidence."""

    primitive: Array
    residual: Array
    enthalpy_residual: Array
    bracket_width: Array
    magnetization: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    bound_active: Array
    status: Array
    iteration_count: Array
    system_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)
    snapshot_token: Array


class ValenciaHLLEBounds(StrictModule):
    lower: Array
    upper: Array
    left_fast_speed: Array
    right_fast_speed: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class ValenciaHLLEFlux(StrictModule):
    normal_flux: Array
    maximum_speed: Array
    bounds: ValenciaHLLEBounds
    left_recovery: ValenciaPrimitiveRecovery
    right_recovery: ValenciaPrimitiveRecovery
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class IdealValenciaGRMHDSystem(StrictModule, NonTrainableState):
    """Ideal GRMHD in the curvature-aware densitized Valencia formulation.

    Primitive components are ``(rho, v^x, v^y, v^z, p, B^x, B^y, B^z)``.
    Conserved components are ``sqrt(gamma) (D, S_x, S_y, S_z, tau, B^x,
    B^y, B^z)``.  Velocities and magnetic fields are Eulerian spatial vectors;
    momentum is covariant.  These conventions are intentionally distinct from
    the package's Newtonian flat-space ideal-MHD system.
    """

    eos: AbstractRelativisticEOS
    scale: RelativityScaleContract
    convention: RelativityConvention
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    pressure_ceiling: float = eqx.field(static=True)
    maximum_magnetization: float = eqx.field(static=True)
    recovery_iterations: int = eqx.field(static=True)
    enthalpy_iterations: int = eqx.field(static=True)
    bracket_iterations: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    primitive_names: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        eos: AbstractRelativisticEOS,
        scale: RelativityScaleContract,
        /,
        *,
        convention: RelativityConvention | None = None,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-14,
        pressure_ceiling: float = 1.0e12,
        maximum_magnetization: float = 1.0e6,
        recovery_iterations: int = 48,
        enthalpy_iterations: int = 32,
        bracket_iterations: int = 16,
        absolute_tolerance: float = 1.0e-11,
        relative_tolerance: float = 1.0e-9,
    ):
        if not isinstance(eos, AbstractRelativisticEOS):
            raise TypeError("eos must be AbstractRelativisticEOS.")
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if eos.scale.scale_id != scale.scale_id:
            raise ValueError("Relativistic EOS and GRMHD scale identities differ.")
        if float(scale.speed_of_light) != 1.0:
            raise ValueError("Valencia GRMHD requires a geometric c=1 scale.")
        convention_ = (
            RelativityConvention.canonical() if convention is None else convention
        )
        if not isinstance(convention_, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        if (
            convention_.metric_signature != "mostly_plus"
            or convention_.extrinsic_curvature_sign != -1
        ):
            raise ValueError(
                "Valencia GRMHD requires mostly-plus signature and K_ij = -1/2 L_n gamma_ij."
            )
        density = float(density_floor)
        pressure = float(pressure_floor)
        ceiling = float(pressure_ceiling)
        magnetization = float(maximum_magnetization)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        iterations = int(recovery_iterations)
        enthalpy_count = int(enthalpy_iterations)
        bracket_count = int(bracket_iterations)
        if (
            any(
                not np.isfinite(value)
                for value in (
                    density,
                    pressure,
                    ceiling,
                    magnetization,
                    absolute,
                    relative,
                )
            )
            or density <= 0.0
            or pressure <= 0.0
            or ceiling <= pressure
            or magnetization <= 0.0
            or absolute <= 0.0
            or relative <= 0.0
            or iterations <= 0
            or enthalpy_count <= 0
            or bracket_count <= 0
        ):
            raise ValueError("Valencia GRMHD recovery controls are invalid.")
        eos_id = eos.eos_id
        if not isinstance(eos_id, str) or not eos_id:
            raise ValueError("Relativistic EOS must provide a non-empty eos_id.")
        self.eos = eos
        self.scale = scale
        self.convention = convention_
        self.density_floor = density
        self.pressure_floor = pressure
        self.pressure_ceiling = ceiling
        self.maximum_magnetization = magnetization
        self.recovery_iterations = iterations
        self.enthalpy_iterations = enthalpy_count
        self.bracket_iterations = bracket_count
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.component_names = (
            "densitized_rest_mass",
            "densitized_momentum_x",
            "densitized_momentum_y",
            "densitized_momentum_z",
            "densitized_energy_minus_rest_mass",
            "densitized_magnetic_x",
            "densitized_magnetic_y",
            "densitized_magnetic_z",
        )
        self.primitive_names = (
            "rest_mass_density",
            "eulerian_velocity_x",
            "eulerian_velocity_y",
            "eulerian_velocity_z",
            "pressure",
            "eulerian_magnetic_x",
            "eulerian_magnetic_y",
            "eulerian_magnetic_z",
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "ideal-valencia-grmhd",
                "eos": eos_id,
                "scale": scale.scale_id,
                "convention": convention_.convention_id,
                "density_floor": density,
                "pressure_floor": pressure,
                "pressure_ceiling": ceiling,
                "maximum_magnetization": magnetization,
                "recovery_iterations": iterations,
                "enthalpy_iterations": enthalpy_count,
                "bracket_iterations": bracket_count,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    @staticmethod
    def _state(value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape[-1:] != (8,):
            raise ValueError(f"{name} must have trailing shape (8,).")
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"{name} must have a real floating-point dtype.")
        return array

    def _geometry(
        self,
        geometry: ADMGridGeometry,
        leading_shape: tuple[int, ...],
        /,
    ) -> ADMGridGeometry:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if geometry.leading_shape != leading_shape:
            raise ValueError(
                "ADM geometry leading shape must exactly match the Valencia state."
            )
        if geometry.scale_id != self.scale.scale_id:
            raise ValueError("ADM geometry and GRMHD system scale identities differ.")
        if geometry.convention_id != self.convention.convention_id:
            raise ValueError("ADM geometry and GRMHD convention identities differ.")
        return geometry

    def _primitive_kinematics(
        self,
        primitive: Array,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None,
        /,
    ):
        density = primitive[..., 0]
        velocity = primitive[..., 1:4]
        pressure = primitive[..., 4]
        magnetic = primitive[..., 5:8]
        velocity_covector = ein.contract(
            "...ij,...j->...i", geometry.spatial_metric, velocity
        )
        magnetic_covector = ein.contract(
            "...ij,...j->...i", geometry.spatial_metric, magnetic
        )
        velocity_squared = ein.contract("...i,...i->...", velocity_covector, velocity)
        magnetic_squared = ein.contract("...i,...i->...", magnetic_covector, magnetic)
        magnetic_velocity = ein.contract("...i,...i->...", magnetic_covector, velocity)
        epsilon = jnp.finfo(primitive.dtype).eps
        lorentz_factor = 1.0 / jnp.sqrt(jnp.maximum(1.0 - velocity_squared, epsilon))
        eos_state = self.eos.evaluate_pressure(density, pressure, composition)
        enthalpy_density = density * eos_state.specific_enthalpy
        comoving_magnetic_squared = (
            magnetic_squared / lorentz_factor**2 + magnetic_velocity**2
        )
        alpha_b_zero = lorentz_factor * magnetic_velocity
        comoving_magnetic_covector = (
            magnetic_covector / lorentz_factor[..., None]
            + alpha_b_zero[..., None] * lorentz_factor[..., None] * velocity_covector
        )
        return (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            eos_state,
            enthalpy_density,
            comoving_magnetic_squared,
            alpha_b_zero,
            comoving_magnetic_covector,
        )

    def primitive_to_conserved(
        self,
        primitive: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        value = self._state(primitive, "Valencia primitive state")
        geometry_ = self._geometry(geometry, value.shape[:-1])
        (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            eos_state,
            enthalpy_density,
            comoving_magnetic_squared,
            alpha_b_zero,
            comoving_magnetic_covector,
        ) = self._primitive_kinematics(value, geometry_, composition)
        del (
            velocity,
            velocity_squared,
            eos_state,
            comoving_magnetic_covector,
        )
        momentum = (enthalpy_density * lorentz_factor**2 + magnetic_squared)[
            ..., None
        ] * velocity_covector - magnetic_velocity[..., None] * magnetic_covector
        energy = (
            (enthalpy_density + comoving_magnetic_squared) * lorentz_factor**2
            - (pressure + 0.5 * comoving_magnetic_squared)
            - alpha_b_zero**2
        )
        rest_mass = density * lorentz_factor
        volume = geometry_.sqrt_det_spatial_metric
        return jnp.concatenate(
            (
                (volume * rest_mass)[..., None],
                volume[..., None] * momentum,
                (volume * (energy - rest_mass))[..., None],
                volume[..., None] * magnetic,
            ),
            axis=-1,
        )

    def stress_energy(
        self,
        primitive: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
        *,
        conserved: ArrayLike | None = None,
    ) -> StressEnergyProjection:
        value = self._state(primitive, "Valencia primitive state")
        geometry_ = self._geometry(geometry, value.shape[:-1])
        (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            eos_state,
            enthalpy_density,
            comoving_magnetic_squared,
            alpha_b_zero,
            comoving_magnetic_covector,
        ) = self._primitive_kinematics(value, geometry_, composition)
        del velocity, magnetic
        momentum = (enthalpy_density * lorentz_factor**2 + magnetic_squared)[
            ..., None
        ] * velocity_covector - magnetic_velocity[..., None] * magnetic_covector
        momentum_from_b = (
            (enthalpy_density + comoving_magnetic_squared) * lorentz_factor**2
        )[..., None] * velocity_covector - alpha_b_zero[
            ..., None
        ] * comoving_magnetic_covector
        stress = (
            ((enthalpy_density + comoving_magnetic_squared) * lorentz_factor**2)[
                ..., None, None
            ]
            * ein.contract("...i,...j->...ij", velocity_covector, velocity_covector)
            + (pressure + 0.5 * comoving_magnetic_squared)[..., None, None]
            * geometry_.spatial_metric
            - ein.contract(
                "...i,...j->...ij",
                comoving_magnetic_covector,
                comoving_magnetic_covector,
            )
        )
        energy = (
            (enthalpy_density + comoving_magnetic_squared) * lorentz_factor**2
            - (pressure + 0.5 * comoving_magnetic_squared)
            - alpha_b_zero**2
        )
        projection_defect = jnp.max(jnp.abs(momentum - momentum_from_b), axis=-1)
        if conserved is None:
            conservation_defect = jnp.zeros_like(energy)
        else:
            conserved_ = self._state(conserved, "Valencia conserved state")
            undensitized = conserved_ / geometry_.sqrt_det_spatial_metric[..., None]
            conservation_defect = jnp.abs(
                energy - (undensitized[..., 0] + undensitized[..., 4])
            )
        primitive_finite = jnp.all(jnp.isfinite(value), axis=-1)
        physically_valid = (
            primitive_finite
            & (density >= self.density_floor)
            & (pressure >= self.pressure_floor)
            & (velocity_squared < 1.0)
            & eos_state.physically_valid
            & geometry_.physically_valid
        )
        return StressEnergyProjection(
            energy,
            momentum,
            stress,
            geometry_.active,
            physically_valid,
            projection_defect,
            conservation_defect,
            snapshot_token=geometry_.snapshot_token,
            geometry_lineage_id=geometry_.geometry_lineage_id,
            convention_id=geometry_.convention_id,
            scale_id=geometry_.scale_id,
            topology_id=geometry_.topology_id,
            projection_id=canonical_fingerprint(
                {
                    "kind": "ideal-valencia-grmhd-stress-energy",
                    "system": self.system_id,
                    "geometry_lineage": geometry_.geometry_lineage_id,
                }
            ),
        )

    def _pressure_at_enthalpy(
        self,
        density: Array,
        target_enthalpy: Array,
        composition: ArrayLike | None,
        /,
    ):
        pressure_lower = jnp.full_like(density, self.pressure_floor)
        pressure_upper = jnp.full_like(density, self.pressure_ceiling)
        if isinstance(self.eos, GammaLawEOS):
            gamma = jnp.asarray(self.eos.adiabatic_index, dtype=density.dtype)
            unconstrained = density * (target_enthalpy - 1.0) * (gamma - 1.0) / gamma
            pressure = jnp.clip(
                unconstrained,
                self.pressure_floor,
                self.pressure_ceiling,
            )
            state = self.eos.evaluate_pressure(density, pressure, composition)
            bracketed = (
                jnp.isfinite(unconstrained)
                & (unconstrained >= pressure_lower)
                & (unconstrained <= pressure_upper)
                & state.finite
            )
            return (
                pressure,
                state,
                state.specific_enthalpy - target_enthalpy,
                bracketed,
                jnp.zeros_like(pressure),
            )
        log_pressure_lower = jnp.log(pressure_lower)
        log_pressure_upper = jnp.log(pressure_upper)
        lower_state = self.eos.evaluate_pressure(density, pressure_lower, composition)
        upper_state = self.eos.evaluate_pressure(density, pressure_upper, composition)
        bracketed = (
            lower_state.finite
            & upper_state.finite
            & (lower_state.specific_enthalpy <= target_enthalpy)
            & (upper_state.specific_enthalpy >= target_enthalpy)
        )

        def body(_, bounds):
            lower, upper = bounds
            middle = 0.5 * (lower + upper)
            state = self.eos.evaluate_pressure(density, jnp.exp(middle), composition)
            below = state.specific_enthalpy <= target_enthalpy
            return jnp.where(below, middle, lower), jnp.where(below, upper, middle)

        lower, upper = jax.lax.fori_loop(
            0,
            self.enthalpy_iterations,
            body,
            (log_pressure_lower, log_pressure_upper),
        )
        pressure_lower_final = jnp.exp(lower)
        pressure_upper_final = jnp.exp(upper)
        pressure = jnp.sqrt(pressure_lower_final * pressure_upper_final)
        state = self.eos.evaluate_pressure(density, pressure, composition)
        residual = state.specific_enthalpy - target_enthalpy
        return (
            pressure,
            state,
            residual,
            bracketed,
            pressure_upper_final - pressure_lower_final,
        )

    def _recovery_residual(
        self,
        x: Array,
        rest_mass: Array,
        momentum_squared: Array,
        magnetic_squared: Array,
        magnetic_momentum: Array,
        total_energy: Array,
        composition: ArrayLike | None,
        /,
    ):
        safe_x = jnp.maximum(x, jnp.finfo(x.dtype).tiny)
        velocity_squared = (
            momentum_squared
            + magnetic_momentum**2 * (2.0 * safe_x + magnetic_squared) / safe_x**2
        ) / (safe_x + magnetic_squared) ** 2
        one_minus_velocity_squared = jnp.maximum(
            1.0 - velocity_squared,
            jnp.finfo(x.dtype).eps,
        )
        sqrt_one_minus = jnp.sqrt(one_minus_velocity_squared)
        density = rest_mass * sqrt_one_minus
        target_enthalpy = (
            safe_x * sqrt_one_minus / jnp.maximum(rest_mass, jnp.finfo(x.dtype).tiny)
        )
        pressure, eos_state, enthalpy_residual, enthalpy_bracketed, pressure_width = (
            self._pressure_at_enthalpy(density, target_enthalpy, composition)
        )
        residual = (
            safe_x
            - pressure
            + 0.5 * magnetic_squared * (1.0 + velocity_squared)
            - 0.5 * (magnetic_momentum / safe_x) ** 2
            - total_energy
        )
        return (
            residual,
            pressure,
            eos_state,
            enthalpy_residual,
            enthalpy_bracketed,
            pressure_width,
            density,
            velocity_squared,
        )

    def recover(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
    ) -> ValenciaPrimitiveRecovery:
        value = self._state(conserved, "Valencia conserved state")
        geometry_ = self._geometry(geometry, value.shape[:-1])
        volume = geometry_.sqrt_det_spatial_metric
        safe_volume = jnp.maximum(volume, jnp.finfo(value.dtype).tiny)
        undensitized = value / safe_volume[..., None]
        rest_mass = undensitized[..., 0]
        momentum_covector = undensitized[..., 1:4]
        tau = undensitized[..., 4]
        magnetic = undensitized[..., 5:8]
        momentum_squared = ein.contract(
            "...i,...ij,...j->...",
            momentum_covector,
            geometry_.inverse_spatial_metric,
            momentum_covector,
        )
        magnetic_covector = ein.contract(
            "...ij,...j->...i", geometry_.spatial_metric, magnetic
        )
        magnetic_squared = ein.contract("...i,...i->...", magnetic_covector, magnetic)
        magnetic_momentum = ein.contract("...i,...i->...", momentum_covector, magnetic)
        total_energy = tau + rest_mass
        epsilon = jnp.finfo(value.dtype).eps
        lower = jnp.maximum(
            jnp.maximum(rest_mass, jnp.sqrt(jnp.maximum(momentum_squared, 0.0))),
            self.density_floor,
        ) * (1.0 + 64.0 * epsilon)
        lower_evaluation = self._recovery_residual(
            lower,
            rest_mass,
            momentum_squared,
            magnetic_squared,
            magnetic_momentum,
            total_energy,
            composition,
        )
        upper = jnp.maximum(
            2.0 * lower,
            jnp.abs(total_energy) + rest_mass + magnetic_squared + 1.0,
        )

        def expand(_, current):
            evaluation = self._recovery_residual(
                current,
                rest_mass,
                momentum_squared,
                magnetic_squared,
                magnetic_momentum,
                total_energy,
                composition,
            )
            return jnp.where(evaluation[0] < 0.0, 2.0 * current, current)

        upper = jax.lax.fori_loop(0, self.bracket_iterations, expand, upper)
        upper_evaluation = self._recovery_residual(
            upper,
            rest_mass,
            momentum_squared,
            magnetic_squared,
            magnetic_momentum,
            total_energy,
            composition,
        )
        root_bracketed = (lower_evaluation[0] <= 0.0) & (upper_evaluation[0] >= 0.0)

        def bisect(_, bounds):
            low, high = bounds
            middle = 0.5 * (low + high)
            evaluation = self._recovery_residual(
                middle,
                rest_mass,
                momentum_squared,
                magnetic_squared,
                magnetic_momentum,
                total_energy,
                composition,
            )
            below = evaluation[0] <= 0.0
            return jnp.where(below, middle, low), jnp.where(below, high, middle)

        final_lower, final_upper = jax.lax.fori_loop(
            0,
            self.recovery_iterations,
            bisect,
            (lower, upper),
        )
        x = 0.5 * (final_lower + final_upper)
        (
            residual,
            pressure,
            eos_state,
            enthalpy_residual,
            enthalpy_bracketed,
            pressure_width,
            density,
            velocity_squared,
        ) = self._recovery_residual(
            x,
            rest_mass,
            momentum_squared,
            magnetic_squared,
            magnetic_momentum,
            total_energy,
            composition,
        )
        magnetic_velocity = magnetic_momentum / jnp.maximum(x, jnp.finfo(x.dtype).tiny)
        velocity_covector = (
            momentum_covector + magnetic_velocity[..., None] * magnetic_covector
        ) / (x + magnetic_squared)[..., None]
        velocity = ein.contract(
            "...ij,...j->...i", geometry_.inverse_spatial_metric, velocity_covector
        )
        primitive = jnp.concatenate(
            (
                density[..., None],
                velocity,
                pressure[..., None],
                magnetic,
            ),
            axis=-1,
        )
        lorentz_factor = 1.0 / jnp.sqrt(
            jnp.maximum(1.0 - velocity_squared, jnp.finfo(x.dtype).eps)
        )
        comoving_magnetic_squared = (
            magnetic_squared / lorentz_factor**2 + magnetic_velocity**2
        )
        magnetization = comoving_magnetic_squared / jnp.maximum(
            density * eos_state.specific_enthalpy,
            jnp.finfo(x.dtype).tiny,
        )
        input_finite = jnp.all(jnp.isfinite(value), axis=-1)
        result_finite = (
            jnp.all(jnp.isfinite(primitive), axis=-1)
            & jnp.isfinite(residual)
            & jnp.isfinite(enthalpy_residual)
            & jnp.isfinite(magnetization)
        )
        finite = input_finite & result_finite & geometry_.finite
        residual_scale = jnp.maximum(jnp.abs(total_energy), 1.0)
        tolerance = jnp.maximum(
            self.absolute_tolerance + self.relative_tolerance * residual_scale,
            128.0 * epsilon * residual_scale,
        )
        enthalpy_scale = jnp.maximum(jnp.abs(eos_state.specific_enthalpy), 1.0)
        enthalpy_tolerance = jnp.maximum(
            self.absolute_tolerance + self.relative_tolerance * enthalpy_scale,
            128.0 * epsilon * enthalpy_scale,
        )
        converged = (
            root_bracketed
            & enthalpy_bracketed
            & (jnp.abs(residual) <= tolerance)
            & (jnp.abs(enthalpy_residual) <= enthalpy_tolerance)
        )
        subluminal = velocity_squared < 1.0
        density_valid = density >= self.density_floor
        pressure_valid = pressure >= self.pressure_floor
        physically_valid = (
            finite
            & geometry_.physically_valid
            & subluminal
            & density_valid
            & pressure_valid
            & eos_state.physically_valid
        )
        qualified = (
            converged
            & physically_valid
            & eos_state.qualified
            & (magnetization <= self.maximum_magnetization)
        )
        bound_active = (
            (
                (pressure_width <= 4.0 * epsilon * jnp.maximum(pressure, 1.0))
                & (
                    (pressure <= self.pressure_floor * (1.0 + 128.0 * epsilon))
                    | (pressure >= self.pressure_ceiling * (1.0 - 128.0 * epsilon))
                )
            )
            | ~root_bracketed
            | ~enthalpy_bracketed
        )
        residual_step = jnp.sqrt(epsilon) * jnp.maximum(jnp.abs(x), 1.0)
        shifted_residual = self._recovery_residual(
            x + residual_step,
            rest_mass,
            momentum_squared,
            magnetic_squared,
            magnetic_momentum,
            total_energy,
            composition,
        )[0]
        residual_slope = (shifted_residual - residual) / residual_step
        derivative_valid = (
            qualified
            & eos_state.derivative_valid
            & ~bound_active
            & jnp.isfinite(residual_slope)
            & (jnp.abs(residual_slope) > jnp.sqrt(epsilon))
        )
        status = jnp.where(
            ~input_finite,
            int(ValenciaRecoveryStatus.NONFINITE_CONSERVED),
            jnp.where(
                ~geometry_.physically_valid,
                int(ValenciaRecoveryStatus.GEOMETRY_INVALID),
                jnp.where(
                    ~root_bracketed | ~enthalpy_bracketed,
                    int(ValenciaRecoveryStatus.ROOT_NOT_BRACKETED),
                    jnp.where(
                        ~converged,
                        int(ValenciaRecoveryStatus.ROOT_NOT_CONVERGED),
                        jnp.where(
                            ~subluminal,
                            int(ValenciaRecoveryStatus.SUPERLUMINAL),
                            jnp.where(
                                ~density_valid,
                                int(ValenciaRecoveryStatus.DENSITY_FLOOR),
                                jnp.where(
                                    ~pressure_valid,
                                    int(ValenciaRecoveryStatus.PRESSURE_FLOOR),
                                    jnp.where(
                                        ~eos_state.physically_valid,
                                        int(ValenciaRecoveryStatus.EOS_INVALID),
                                        jnp.where(
                                            magnetization > self.maximum_magnetization,
                                            int(
                                                ValenciaRecoveryStatus.MAGNETIZATION_LIMIT
                                            ),
                                            int(ValenciaRecoveryStatus.SUCCESS),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return ValenciaPrimitiveRecovery(
            primitive=primitive,
            residual=residual,
            enthalpy_residual=enthalpy_residual,
            bracket_width=final_upper - final_lower,
            magnetization=magnetization,
            finite=finite,
            converged=converged,
            physically_valid=physically_valid,
            qualified=qualified,
            derivative_valid=derivative_valid,
            bound_active=bound_active,
            status=status,
            iteration_count=jnp.full(
                value.shape[:-1], self.recovery_iterations, dtype=jnp.int32
            ),
            system_id=self.system_id,
            geometry_lineage_id=geometry_.geometry_lineage_id,
            snapshot_token=geometry_.snapshot_token,
        )

    def conserved_to_primitive(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.recover(conserved, geometry, composition).primitive

    def admissible(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.recover(conserved, geometry, composition).physically_valid

    def _flux_from_recovery(
        self,
        conserved: Array,
        recovery: ValenciaPrimitiveRecovery,
        geometry: ADMGridGeometry,
        axis: int,
        composition: ArrayLike | None,
        /,
    ) -> Array:
        primitive = recovery.primitive
        (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            eos_state,
            enthalpy_density,
            comoving_magnetic_squared,
            alpha_b_zero,
            comoving_magnetic_covector,
        ) = self._primitive_kinematics(primitive, geometry, composition)
        del (
            density,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            eos_state,
            enthalpy_density,
            alpha_b_zero,
        )
        transport = (
            geometry.alpha * velocity[..., axis] - geometry.beta_contravariant[..., axis]
        )
        volume = geometry.sqrt_det_spatial_metric
        rest_mass = conserved[..., 0]
        momentum = conserved[..., 1:4]
        tau = conserved[..., 4]
        magnetic_densitized = conserved[..., 5:8]
        pressure_star = pressure + 0.5 * comoving_magnetic_squared
        momentum_flux = momentum * transport[..., None]
        momentum_flux = momentum_flux.at[..., axis].add(
            geometry.alpha * volume * pressure_star
        )
        momentum_flux = (
            momentum_flux
            - (geometry.alpha * volume * magnetic[..., axis] / lorentz_factor)[..., None]
            * comoving_magnetic_covector
        )
        momentum_contravariant_densitized = ein.contract(
            "...ij,...j->...i", geometry.inverse_spatial_metric, momentum
        )
        energy_flux = (
            geometry.alpha
            * (
                momentum_contravariant_densitized[..., axis]
                - rest_mass * velocity[..., axis]
            )
            - geometry.beta_contravariant[..., axis] * tau
        )
        magnetic_flux = transport[..., None] * magnetic_densitized - magnetic_densitized[
            ..., axis, None
        ] * (geometry.alpha[..., None] * velocity - geometry.beta_contravariant)
        magnetic_flux = magnetic_flux.at[..., axis].set(0.0)
        return jnp.concatenate(
            (
                (rest_mass * transport)[..., None],
                momentum_flux,
                energy_flux[..., None],
                magnetic_flux,
            ),
            axis=-1,
        )

    def physical_flux(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        axis: int,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        value = self._state(conserved, "Valencia conserved state")
        geometry_ = self._geometry(geometry, value.shape[:-1])
        recovery = self.recover(value, geometry_, composition)
        return self._flux_from_recovery(
            value, recovery, geometry_, int(axis), composition
        )

    def _fast_speed(
        self,
        primitive: Array,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None,
        /,
    ) -> Array:
        (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            eos_state,
            enthalpy_density,
            comoving_magnetic_squared,
            alpha_b_zero,
            comoving_magnetic_covector,
        ) = self._primitive_kinematics(primitive, geometry, composition)
        del (
            density,
            velocity,
            pressure,
            magnetic,
            velocity_covector,
            magnetic_covector,
            velocity_squared,
            magnetic_squared,
            magnetic_velocity,
            lorentz_factor,
            alpha_b_zero,
            comoving_magnetic_covector,
        )
        alfven_squared = comoving_magnetic_squared / jnp.maximum(
            enthalpy_density + comoving_magnetic_squared,
            jnp.finfo(primitive.dtype).tiny,
        )
        sound_squared = eos_state.sound_speed_squared
        fast_squared = sound_squared + alfven_squared - sound_squared * alfven_squared
        return jnp.sqrt(jnp.clip(fast_squared, 0.0, 1.0 - jnp.finfo(primitive.dtype).eps))

    def _coordinate_characteristics(
        self,
        primitive: Array,
        fast_speed: Array,
        geometry: ADMGridGeometry,
        axis: int,
        /,
    ) -> tuple[Array, Array]:
        velocity = primitive[..., 1:4]
        velocity_covector = ein.contract(
            "...ij,...j->...i", geometry.spatial_metric, velocity
        )
        velocity_squared = ein.contract("...i,...i->...", velocity_covector, velocity)
        normal_velocity = velocity[..., axis]
        speed_squared = fast_speed**2
        denominator = 1.0 - velocity_squared * speed_squared
        radical = (1.0 - velocity_squared) * (
            geometry.inverse_spatial_metric[..., axis, axis]
            * (1.0 - velocity_squared * speed_squared)
            - normal_velocity**2 * (1.0 - speed_squared)
        )
        root = fast_speed * jnp.sqrt(jnp.maximum(radical, 0.0))
        common = normal_velocity * (1.0 - speed_squared)
        lower = (
            geometry.alpha * (common - root) / denominator
            - geometry.beta_contravariant[..., axis]
        )
        upper = (
            geometry.alpha * (common + root) / denominator
            - geometry.beta_contravariant[..., axis]
        )
        return lower, upper

    def _bounds_from_recoveries(
        self,
        left_recovery: ValenciaPrimitiveRecovery,
        right_recovery: ValenciaPrimitiveRecovery,
        geometry: ADMGridGeometry,
        axis: int,
        composition: ArrayLike | None,
        /,
    ) -> ValenciaHLLEBounds:
        left_fast = self._fast_speed(left_recovery.primitive, geometry, composition)
        right_fast = self._fast_speed(right_recovery.primitive, geometry, composition)
        left_lower, left_upper = self._coordinate_characteristics(
            left_recovery.primitive, left_fast, geometry, axis
        )
        right_lower, right_upper = self._coordinate_characteristics(
            right_recovery.primitive, right_fast, geometry, axis
        )
        lower = jnp.minimum(left_lower, right_lower)
        upper = jnp.maximum(left_upper, right_upper)
        finite = (
            left_recovery.finite
            & right_recovery.finite
            & jnp.isfinite(lower)
            & jnp.isfinite(upper)
        )
        converged = left_recovery.converged & right_recovery.converged
        physically_valid = (
            left_recovery.physically_valid
            & right_recovery.physically_valid
            & (lower <= upper)
        )
        qualified = physically_valid & left_recovery.qualified & right_recovery.qualified
        derivative_valid = (
            qualified & left_recovery.derivative_valid & right_recovery.derivative_valid
        )
        return ValenciaHLLEBounds(
            lower,
            upper,
            left_fast,
            right_fast,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
        )

    def signal_bounds(
        self,
        left: ArrayLike,
        right: ArrayLike,
        geometry: ADMGridGeometry,
        axis: int,
        composition: ArrayLike | None = None,
        /,
    ) -> ValenciaHLLEBounds:
        left_ = self._state(left, "Left Valencia conserved state")
        right_ = self._state(right, "Right Valencia conserved state")
        if left_.shape != right_.shape:
            raise ValueError("Left and right Valencia states must have identical shapes.")
        geometry_ = self._geometry(geometry, left_.shape[:-1])
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Valencia flux axis must be zero, one, or two.")
        return self._bounds_from_recoveries(
            self.recover(left_, geometry_, composition),
            self.recover(right_, geometry_, composition),
            geometry_,
            axis_,
            composition,
        )

    def hlle_flux(
        self,
        left: ArrayLike,
        right: ArrayLike,
        geometry: ADMGridGeometry,
        axis: int,
        composition: ArrayLike | None = None,
        /,
    ) -> ValenciaHLLEFlux:
        left_ = self._state(left, "Left Valencia conserved state")
        right_ = self._state(right, "Right Valencia conserved state")
        if left_.shape != right_.shape:
            raise ValueError("Left and right Valencia states must have identical shapes.")
        geometry_ = self._geometry(geometry, left_.shape[:-1])
        axis_ = int(axis)
        if axis_ not in (0, 1, 2):
            raise ValueError("Valencia flux axis must be zero, one, or two.")
        left_recovery = self.recover(left_, geometry_, composition)
        right_recovery = self.recover(right_, geometry_, composition)
        bounds = self._bounds_from_recoveries(
            left_recovery,
            right_recovery,
            geometry_,
            axis_,
            composition,
        )
        lower = jnp.minimum(bounds.lower, 0.0)
        upper = jnp.maximum(bounds.upper, 0.0)
        left_flux = self._flux_from_recovery(
            left_, left_recovery, geometry_, axis_, composition
        )
        right_flux = self._flux_from_recovery(
            right_, right_recovery, geometry_, axis_, composition
        )
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (right_ - left_)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        finite = bounds.finite & jnp.all(jnp.isfinite(flux), axis=-1)
        converged = bounds.converged
        physically_valid = bounds.physically_valid & finite
        qualified = bounds.qualified & physically_valid
        derivative_valid = bounds.derivative_valid & qualified
        return ValenciaHLLEFlux(
            flux,
            jnp.maximum(jnp.abs(lower), jnp.abs(upper)),
            bounds,
            left_recovery,
            right_recovery,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
        )

    def geometric_source(
        self,
        conserved: ArrayLike,
        source_geometry: ValenciaGeometrySource,
        composition: ArrayLike | None = None,
        /,
    ) -> Array:
        value = self._state(conserved, "Valencia conserved state")
        if not isinstance(source_geometry, ValenciaGeometrySource):
            raise TypeError("source_geometry must be ValenciaGeometrySource.")
        geometry = self._geometry(source_geometry.geometry, value.shape[:-1])
        recovery = self.recover(value, geometry, composition)
        projection = self.stress_energy(
            recovery.primitive,
            geometry,
            composition,
            conserved=value,
        )
        return self.geometric_source_from_projection(projection, source_geometry)

    def geometric_source_from_projection(
        self,
        projection: StressEnergyProjection,
        source_geometry: ValenciaGeometrySource,
        /,
    ) -> Array:
        """Evaluate curvature sources from an already-recovered stage projection."""

        if not isinstance(source_geometry, ValenciaGeometrySource):
            raise TypeError("source_geometry must be ValenciaGeometrySource.")
        self._geometry(source_geometry.geometry, projection.leading_shape)
        momentum_source, energy_source = valencia_geometric_source_from_projection(
            projection, source_geometry, self.convention
        )
        source = jnp.zeros(
            projection.leading_shape + (8,), dtype=projection.energy_density.dtype
        )
        source = source.at[..., 1:4].set(momentum_source)
        return source.at[..., 4].set(energy_source)


__all__ = [
    "IdealValenciaGRMHDSystem",
    "ValenciaHLLEBounds",
    "ValenciaHLLEFlux",
    "ValenciaPrimitiveRecovery",
    "ValenciaRecoveryStatus",
]
