#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded conservative-to-primitive recovery and explicit GRHD atmosphere."""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_hydrodynamics import ValenciaGRHDSystem
from ..metrix._adm_exchange import ADMGridGeometry
from ..nonlinear import NonlinearStatus, SmallRootKernel


class GRHDC2PStatus(IntEnum):
    PRIMARY_SUCCESS = 0
    BRACKET_SUCCESS = 1
    ATMOSPHERE_APPLIED = 2
    INACTIVE = 3
    NONFINITE_CONSERVED = 4
    INVALID_GEOMETRY = 5
    ROOT_BUDGET_EXHAUSTED = 6
    RECOMPOSITION_FAILED = 7
    ATMOSPHERE_BUDGET_EXCEEDED = 8


class AtmosphereFloorStatus(IntEnum):
    NOT_APPLIED = 0
    NEAR_VACUUM = 1
    RECOVERY_FAILURE = 2
    BUDGET_EXCEEDED = 3


class AtmosphereFloorPolicy(StrictModule, NonTrainableState):
    """Explicit post-candidate atmosphere replacement and hard defect budgets.

    The atmosphere is not an EOS branch and is never used in face fluxes without
    first appearing in the returned conservative correction. Budgets are per
    recovery call in unweighted conserved coordinates; a finite-volume runtime
    additionally enforces volume-integrated budgets.
    """

    rest_mass_density: float = eqx.field(static=True)
    specific_internal_energy: float = eqx.field(static=True)
    activation_density: float = eqx.field(static=True)
    maximum_mass_addition: float = eqx.field(static=True)
    maximum_energy_addition: float = eqx.field(static=True)
    replace_failed_recovery: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        rest_mass_density: float = 1.0e-10,
        specific_internal_energy: float = 1.0e-8,
        activation_density: float | None = None,
        maximum_mass_addition: float = 1.0e30,
        maximum_energy_addition: float = 1.0e30,
        replace_failed_recovery: bool = True,
    ):
        density = float(rest_mass_density)
        energy = float(specific_internal_energy)
        activation = density if activation_density is None else float(activation_density)
        mass_budget = float(maximum_mass_addition)
        energy_budget = float(maximum_energy_addition)
        if (
            not all(
                np.isfinite(value)
                for value in (density, energy, activation, mass_budget, energy_budget)
            )
            or density <= 0.0
            or energy < 0.0
            or activation < density
            or mass_budget < 0.0
            or energy_budget < 0.0
        ):
            raise ValueError(
                "Atmosphere density/energy and activation/budget values are invalid."
            )
        self.rest_mass_density = density
        self.specific_internal_energy = energy
        self.activation_density = activation
        self.maximum_mass_addition = mass_budget
        self.maximum_energy_addition = energy_budget
        self.replace_failed_recovery = bool(replace_failed_recovery)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "grhd-atmosphere-floor",
                "rest_mass_density": density,
                "specific_internal_energy": energy,
                "activation_density": activation,
                "maximum_mass_addition": mass_budget,
                "maximum_energy_addition": energy_budget,
                "replace_failed_recovery": self.replace_failed_recovery,
            }
        )

    def primitive(self, leading_shape: tuple[int, ...], dtype, /) -> Array:
        density = jnp.full(leading_shape + (1,), self.rest_mass_density, dtype=dtype)
        energy = jnp.full(
            leading_shape + (1,), self.specific_internal_energy, dtype=dtype
        )
        velocity = jnp.zeros(leading_shape + (3,), dtype=dtype)
        return jnp.concatenate((density, energy, velocity), axis=-1)


class AtmosphereCorrectionLedger(StrictModule):
    """Exact conservative replacement increment and branch evidence per lane."""

    conservative_increment: Array
    applied: Array
    status: Array
    rest_mass_change: Array
    momentum_change: Array
    energy_change: Array
    total_positive_mass_addition: Array
    total_positive_energy_addition: Array
    within_budget: Array
    policy_id: str = eqx.field(static=True)


class GRHDC2PCandidateRecord(StrictModule):
    """Fixed three-slot primary/bracket/atmosphere candidate journal."""

    attempted: Array
    nonlinear_status: Array
    iterations: Array
    normalized_root_residual: Array
    recomposition_defect: Array
    implicit_derivative: Array
    selected_branch: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRHDC2PResult(StrictModule):
    """Certified primitives plus the exact conservative state they represent."""

    primitive: Array
    conservative_state: Array
    pressure: Array
    status: Array
    candidates: GRHDC2PCandidateRecord
    atmosphere: AtmosphereCorrectionLedger
    successful: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    snapshot_token: Array
    policy_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)

    def compatible_with(self, geometry: ADMGridGeometry, /) -> Array:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        static_compatible = (
            self.geometry_lineage_id == geometry.geometry_lineage_id
            and self.primitive.shape[:-1] == geometry.leading_shape
        )
        return jnp.asarray(static_compatible) & (
            self.snapshot_token == geometry.snapshot_token
        )


class _C2PRootArguments(StrictModule):
    mass: Array
    total_energy: Array
    momentum_squared: Array
    pressure_lower: Array
    residual_scale: Array


class _PressureLogResidual(StrictModule):
    system: ValenciaGRHDSystem

    def __call__(self, state: Array, arguments: _C2PRootArguments, /) -> Array:
        pressure = arguments.pressure_lower + jnp.exp(jnp.clip(state[0], -80.0, 80.0))
        mass = jnp.maximum(arguments.mass, jnp.finfo(state.dtype).tiny)
        q = arguments.total_energy + pressure
        velocity_squared = arguments.momentum_squared / q**2
        lorentz = 1.0 / jnp.sqrt(jnp.maximum(1.0 - velocity_squared, 1.0e-14))
        density = mass / lorentz
        enthalpy = q / (mass * lorentz)
        specific_internal_energy = enthalpy - 1.0 - pressure / density
        eos_state = self.system.eos.evaluate(density, specific_internal_energy)
        return jnp.asarray([(eos_state.pressure - pressure) / arguments.residual_scale])


class GRHDC2PPolicy(StrictModule, NonTrainableState):
    """Static warm-root -> bracket -> atmosphere -> rejection C2P ladder."""

    system: ValenciaGRHDSystem
    atmosphere: AtmosphereFloorPolicy
    primary: SmallRootKernel
    maximum_primary_iterations: int = eqx.field(static=True)
    maximum_bracket_iterations: int = eqx.field(static=True)
    maximum_bracket_expansions: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    recomposition_tolerance: float = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: ValenciaGRHDSystem,
        /,
        *,
        atmosphere: AtmosphereFloorPolicy | None = None,
        maximum_primary_iterations: int = 16,
        maximum_bracket_iterations: int = 64,
        maximum_bracket_expansions: int = 12,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-9,
        recomposition_tolerance: float = 1.0e-8,
        implicit_differentiation: bool = False,
    ):
        if not isinstance(system, ValenciaGRHDSystem):
            raise TypeError("system must be a ValenciaGRHDSystem.")
        atmosphere_ = AtmosphereFloorPolicy() if atmosphere is None else atmosphere
        if not isinstance(atmosphere_, AtmosphereFloorPolicy):
            raise TypeError("atmosphere must be an AtmosphereFloorPolicy or None.")
        primary_iterations = int(maximum_primary_iterations)
        bracket_iterations = int(maximum_bracket_iterations)
        expansions = int(maximum_bracket_expansions)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        recompose = float(recomposition_tolerance)
        if (
            primary_iterations <= 0
            or bracket_iterations <= 0
            or expansions <= 0
            or not all(np.isfinite(value) for value in (absolute, relative, recompose))
            or absolute <= 0.0
            or relative < 0.0
            or recompose <= 0.0
        ):
            raise ValueError("GRHD C2P iteration and tolerance controls are invalid.")
        self.system = system
        self.atmosphere = atmosphere_
        self.maximum_primary_iterations = primary_iterations
        self.maximum_bracket_iterations = bracket_iterations
        self.maximum_bracket_expansions = expansions
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.recomposition_tolerance = recompose
        self.implicit_differentiation = bool(implicit_differentiation)
        self.primary = SmallRootKernel(
            _PressureLogResidual(system),
            maximum_dimension=1,
            maximum_steps=primary_iterations,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )
        self.policy_id = canonical_fingerprint(
            {
                "kind": "grhd-c2p-ladder",
                "system": system.system_id,
                "atmosphere": atmosphere_.policy_id,
                "maximum_primary_iterations": primary_iterations,
                "maximum_bracket_iterations": bracket_iterations,
                "maximum_bracket_expansions": expansions,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "recomposition_tolerance": recompose,
                "implicit_differentiation": self.implicit_differentiation,
                "branches": ["small-root-log-pressure", "bounded-pressure", "atmosphere"],
            }
        )

    def _check_inputs(
        self, conserved: ArrayLike, geometry: ADMGridGeometry
    ) -> tuple[Array, tuple[int, ...]]:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        value = jnp.asarray(conserved)
        if value.shape != geometry.leading_shape + (self.system.component_count,):
            raise ValueError(
                "GRHD conserved state must match the ADM lane shape and component count."
            )
        if value.dtype != geometry.alpha.dtype:
            raise TypeError("GRHD conserved state and ADM geometry must share one dtype.")
        return value, geometry.leading_shape

    def _pressure_data(
        self, conserved: Array, geometry: ADMGridGeometry, /
    ) -> tuple[Array, _C2PRootArguments, Array, Array]:
        flat = conserved.reshape((-1, self.system.component_count))
        sqrt_det = geometry.sqrt_det_spatial_metric.reshape((-1,))
        inverse = geometry.inverse_spatial_metric.reshape((-1, 3, 3))
        undensitized = flat / sqrt_det[:, None]
        mass = undensitized[:, 0]
        momentum = undensitized[:, 1:4]
        total_energy = undensitized[:, -1] + mass
        momentum_squared = ein.contract("ni,nij,nj->n", momentum, inverse, momentum)
        momentum_norm = jnp.sqrt(jnp.maximum(momentum_squared, 0.0))
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(total_energy), momentum_norm),
            jnp.asarray(1.0, dtype=flat.dtype),
        )
        guard = 32.0 * jnp.finfo(flat.dtype).eps * scale
        lower = jnp.maximum(
            jnp.asarray(self.system.pressure_floor, dtype=flat.dtype),
            momentum_norm - total_energy + guard,
        )
        arguments = _C2PRootArguments(
            mass,
            total_energy,
            momentum_squared,
            lower,
            scale,
        )
        return flat, arguments, undensitized, inverse

    def _raw_residual(self, pressure: Array, arguments: _C2PRootArguments, /) -> Array:
        mass = jnp.maximum(arguments.mass, jnp.finfo(pressure.dtype).tiny)
        q = arguments.total_energy + pressure
        velocity_squared = arguments.momentum_squared / q**2
        lorentz = 1.0 / jnp.sqrt(jnp.maximum(1.0 - velocity_squared, 1.0e-14))
        density = mass / lorentz
        enthalpy = q / (mass * lorentz)
        internal = enthalpy - 1.0 - pressure / density
        return (
            self.system.eos.evaluate(density, internal).pressure - pressure
        ) / arguments.residual_scale

    def _bracketed_pressure(
        self, arguments: _C2PRootArguments, /
    ) -> tuple[Array, Array, Array, Array]:
        lower = arguments.pressure_lower
        lower_value = self._raw_residual(lower, arguments)
        upper = lower + jnp.maximum(
            jnp.abs(arguments.total_energy)
            + jnp.sqrt(jnp.maximum(arguments.momentum_squared, 0.0))
            + jnp.abs(arguments.mass),
            1.0,
        )
        upper_value = self._raw_residual(upper, arguments)

        def expand(_, carry):
            bound, value = carry
            same_sign = jnp.signbit(value) == jnp.signbit(lower_value)
            proposed = 2.0 * bound + 1.0
            proposed_value = self._raw_residual(proposed, arguments)
            return jnp.where(same_sign, proposed, bound), jnp.where(
                same_sign, proposed_value, value
            )

        upper, upper_value = jax.lax.fori_loop(
            0, self.maximum_bracket_expansions, expand, (upper, upper_value)
        )
        bracket_valid = (
            jnp.isfinite(lower_value)
            & jnp.isfinite(upper_value)
            & (jnp.signbit(lower_value) != jnp.signbit(upper_value))
        )

        def bisect(_, carry):
            left, right, left_value, right_value = carry
            middle = 0.5 * (left + right)
            middle_value = self._raw_residual(middle, arguments)
            replace_right = jnp.signbit(left_value) != jnp.signbit(middle_value)
            usable = bracket_valid & jnp.isfinite(middle_value)
            return (
                jnp.where(usable & ~replace_right, middle, left),
                jnp.where(usable & replace_right, middle, right),
                jnp.where(usable & ~replace_right, middle_value, left_value),
                jnp.where(usable & replace_right, middle_value, right_value),
            )

        left, right, left_value, right_value = jax.lax.fori_loop(
            0,
            self.maximum_bracket_iterations,
            bisect,
            (lower, upper, lower_value, upper_value),
        )
        choose_left = jnp.abs(left_value) <= jnp.abs(right_value)
        pressure = jnp.where(choose_left, left, right)
        residual = jnp.where(choose_left, left_value, right_value)
        threshold = self.absolute_tolerance + self.relative_tolerance * jnp.maximum(
            jnp.abs(lower_value), jnp.finfo(pressure.dtype).tiny
        )
        successful = (
            bracket_valid & jnp.isfinite(residual) & (jnp.abs(residual) <= threshold)
        )
        return pressure, residual, bracket_valid, successful

    def _candidate(
        self, pressure: Array, conserved: Array, geometry: ADMGridGeometry, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        pressure_shaped = pressure.reshape(geometry.leading_shape)
        primitive, _, _ = self.system.primitive_from_pressure(
            conserved, geometry, pressure_shaped
        )
        evaluation = self.system.primitive_evaluation(primitive, geometry)
        recomposed = self.system.primitive_to_conserved(primitive, geometry)
        scale = jnp.maximum(jnp.abs(conserved), 1.0)
        defect = jnp.max(jnp.abs(recomposed - conserved) / scale, axis=-1)
        finite = evaluation.finite & jnp.all(jnp.isfinite(recomposed), axis=-1)
        physical = (
            evaluation.physically_valid
            & (evaluation.rest_mass_density >= self.system.density_floor)
            & (evaluation.pressure >= self.system.pressure_floor)
        )
        qualified = (
            evaluation.qualified & physical & (defect <= self.recomposition_tolerance)
        )
        return primitive, evaluation.pressure, defect, finite, physical, qualified

    def recover(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        warm_pressure: ArrayLike | None = None,
    ) -> GRHDC2PResult:
        value, leading_shape = self._check_inputs(conserved, geometry)
        flat, arguments, _, _ = self._pressure_data(value, geometry)
        if warm_pressure is None:
            warm = arguments.pressure_lower + jnp.maximum(
                0.1 * jnp.abs(flat[:, -1]),
                jnp.asarray(self.system.pressure_floor, dtype=value.dtype),
            )
        else:
            warm_array = jnp.asarray(warm_pressure, dtype=value.dtype)
            if warm_array.shape != leading_shape:
                raise ValueError("warm_pressure must match the ADM lane shape.")
            warm = warm_array.reshape((-1,))
        offset = jnp.maximum(warm - arguments.pressure_lower, jnp.finfo(value.dtype).tiny)
        primary_root = self.primary.solve(jnp.log(offset)[:, None], arguments)
        primary_pressure = arguments.pressure_lower + jnp.exp(
            jnp.clip(primary_root.state[:, 0], -80.0, 80.0)
        )
        primary_derivative = jax.vmap(
            lambda root_state, root_arguments: jax.grad(
                lambda candidate: self.primary.residual(candidate, root_arguments)[0]
            )(root_state)[0]
        )(primary_root.state, arguments)
        primary = self._candidate(primary_pressure, value, geometry)
        primary_success = (
            primary_root.successful.reshape(leading_shape)
            & primary[3]
            & primary[4]
            & primary[5]
        )

        bracket_pressure, bracket_residual, bracket_valid, bracket_root_success = (
            self._bracketed_pressure(arguments)
        )
        bracket_derivative = jax.vmap(
            lambda pressure, root_arguments: jax.grad(
                lambda candidate: self._raw_residual(candidate, root_arguments)
            )(pressure)
        )(bracket_pressure, arguments)
        bracket = self._candidate(bracket_pressure, value, geometry)
        bracket_success = (
            bracket_root_success.reshape(leading_shape)
            & bracket[3]
            & bracket[4]
            & bracket[5]
        )
        use_bracket = ~primary_success & bracket_success
        recovered_success = primary_success | use_bracket
        selected_primitive = jnp.where(primary_success[..., None], primary[0], bracket[0])
        selected_pressure = jnp.where(primary_success, primary[1], bracket[1])
        selected_qualified = jnp.where(primary_success, primary[5], bracket[5])

        atmosphere_primitive = self.atmosphere.primitive(leading_shape, value.dtype)
        atmosphere_conserved = self.system.primitive_to_conserved(
            atmosphere_primitive, geometry
        )
        atmosphere_valid = self.system.primitive_evaluation(
            atmosphere_primitive, geometry
        ).physically_valid
        undensitized_mass = value[..., 0] / geometry.sqrt_det_spatial_metric
        near_vacuum = geometry.active & (
            undensitized_mass <= self.atmosphere.activation_density
        )
        failed_replacement = (
            geometry.active & ~recovered_success & self.atmosphere.replace_failed_recovery
        )
        proposed_atmosphere = (near_vacuum | failed_replacement) & atmosphere_valid
        raw_increment = jnp.where(
            proposed_atmosphere[..., None], atmosphere_conserved - value, 0.0
        )
        positive_mass = jnp.sum(jnp.maximum(raw_increment[..., 0], 0.0))
        positive_energy = jnp.sum(jnp.maximum(raw_increment[..., -1], 0.0))
        within_budget = (positive_mass <= self.atmosphere.maximum_mass_addition) & (
            positive_energy <= self.atmosphere.maximum_energy_addition
        )
        apply_atmosphere = proposed_atmosphere & within_budget
        correction = jnp.where(
            apply_atmosphere[..., None], atmosphere_conserved - value, 0.0
        )
        corrected = value + correction
        selected_primitive = jnp.where(
            apply_atmosphere[..., None], atmosphere_primitive, selected_primitive
        )
        selected_pressure = jnp.where(
            apply_atmosphere,
            self.system.eos.evaluate(
                atmosphere_primitive[..., 0], atmosphere_primitive[..., 1]
            ).pressure,
            selected_pressure,
        )
        inactive = ~geometry.active
        selected_primitive = jnp.where(
            inactive[..., None], atmosphere_primitive, selected_primitive
        )
        selected_pressure = jnp.where(
            inactive,
            self.system.eos.evaluate(
                atmosphere_primitive[..., 0], atmosphere_primitive[..., 1]
            ).pressure,
            selected_pressure,
        )
        successful = inactive | recovered_success | apply_atmosphere
        finite_conserved = jnp.all(jnp.isfinite(value), axis=-1)
        geometry_valid = geometry.physically_valid
        successful = successful & (inactive | (finite_conserved & geometry_valid))

        budget_exceeded = proposed_atmosphere & ~within_budget
        recomposition_failed = (
            ~recovered_success & (primary[3] | bracket[3]) & (primary[4] | bracket[4])
        )
        status = jnp.where(
            inactive,
            int(GRHDC2PStatus.INACTIVE),
            jnp.where(
                ~finite_conserved,
                int(GRHDC2PStatus.NONFINITE_CONSERVED),
                jnp.where(
                    ~geometry_valid,
                    int(GRHDC2PStatus.INVALID_GEOMETRY),
                    jnp.where(
                        budget_exceeded,
                        int(GRHDC2PStatus.ATMOSPHERE_BUDGET_EXCEEDED),
                        jnp.where(
                            apply_atmosphere,
                            int(GRHDC2PStatus.ATMOSPHERE_APPLIED),
                            jnp.where(
                                primary_success,
                                int(GRHDC2PStatus.PRIMARY_SUCCESS),
                                jnp.where(
                                    use_bracket,
                                    int(GRHDC2PStatus.BRACKET_SUCCESS),
                                    jnp.where(
                                        recomposition_failed,
                                        int(GRHDC2PStatus.RECOMPOSITION_FAILED),
                                        int(GRHDC2PStatus.ROOT_BUDGET_EXHAUSTED),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        atmosphere_status = jnp.where(
            budget_exceeded,
            int(AtmosphereFloorStatus.BUDGET_EXCEEDED),
            jnp.where(
                apply_atmosphere & near_vacuum,
                int(AtmosphereFloorStatus.NEAR_VACUUM),
                jnp.where(
                    apply_atmosphere & failed_replacement,
                    int(AtmosphereFloorStatus.RECOVERY_FAILURE),
                    int(AtmosphereFloorStatus.NOT_APPLIED),
                ),
            ),
        ).astype(jnp.int32)
        atmosphere_ledger = AtmosphereCorrectionLedger(
            correction,
            apply_atmosphere,
            atmosphere_status,
            correction[..., 0],
            correction[..., 1:4],
            correction[..., -1],
            jnp.sum(jnp.maximum(correction[..., 0], 0.0)),
            jnp.sum(jnp.maximum(correction[..., -1], 0.0)),
            within_budget,
            self.atmosphere.policy_id,
        )

        primary_status = primary_root.status.reshape(leading_shape)
        bracket_status = jnp.where(
            bracket_root_success.reshape(leading_shape),
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                bracket_valid.reshape(leading_shape),
                int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
                int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            ),
        ).astype(jnp.int32)
        atmosphere_nonlinear_status = jnp.where(
            apply_atmosphere,
            int(NonlinearStatus.SUCCESS),
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
        ).astype(jnp.int32)
        candidate_status = jnp.stack(
            (primary_status, bracket_status, atmosphere_nonlinear_status), axis=-1
        )
        iterations = jnp.stack(
            (
                primary_root.iterations.reshape(leading_shape),
                jnp.full(
                    leading_shape,
                    self.maximum_bracket_iterations,
                    dtype=jnp.int32,
                ),
                jnp.zeros(leading_shape, dtype=jnp.int32),
            ),
            axis=-1,
        )
        normalized_residual = jnp.stack(
            (
                primary_root.residual_norm.reshape(leading_shape),
                jnp.abs(bracket_residual).reshape(leading_shape),
                jnp.zeros(leading_shape, dtype=value.dtype),
            ),
            axis=-1,
        )
        recomposition = jnp.stack(
            (
                primary[2],
                bracket[2],
                jnp.max(
                    jnp.abs(atmosphere_conserved - corrected)
                    / jnp.maximum(jnp.abs(corrected), 1.0),
                    axis=-1,
                ),
            ),
            axis=-1,
        )
        implicit_derivative = jnp.stack(
            (
                primary_derivative.reshape(leading_shape),
                bracket_derivative.reshape(leading_shape),
                jnp.zeros(leading_shape, dtype=value.dtype),
            ),
            axis=-1,
        )
        attempted = jnp.stack(
            (
                geometry.active,
                geometry.active & ~primary_success,
                proposed_atmosphere,
            ),
            axis=-1,
        )
        selected_branch = jnp.where(
            inactive,
            -1,
            jnp.where(apply_atmosphere, 2, jnp.where(use_bracket, 1, 0)),
        ).astype(jnp.int32)
        final_evaluation = self.system.primitive_evaluation(selected_primitive, geometry)
        final_finite = inactive | (
            final_evaluation.finite & jnp.all(jnp.isfinite(corrected), axis=-1)
        )
        final_physical = inactive | final_evaluation.physically_valid
        final_qualified = inactive | (
            final_evaluation.qualified
            & jnp.where(apply_atmosphere, True, selected_qualified)
        )
        final_converged = inactive | recovered_success | apply_atmosphere
        primary_derivative_valid = jnp.isfinite(
            primary_derivative.reshape(leading_shape)
        ) & (
            jnp.abs(primary_derivative.reshape(leading_shape))
            > jnp.sqrt(jnp.finfo(value.dtype).eps)
        )
        derivative_valid = (
            primary_success
            & ~apply_atmosphere
            & self.implicit_differentiation
            & final_evaluation.derivative_valid
            & primary_derivative_valid
        )
        record = GRHDC2PCandidateRecord(
            attempted,
            candidate_status,
            iterations,
            normalized_residual,
            recomposition,
            implicit_derivative,
            selected_branch,
            final_finite,
            final_converged,
            final_physical,
            final_qualified,
            derivative_valid,
        )
        return GRHDC2PResult(
            selected_primitive,
            corrected,
            selected_pressure,
            status,
            record,
            atmosphere_ledger,
            successful,
            final_finite,
            final_converged,
            final_physical,
            final_qualified,
            derivative_valid,
            geometry.snapshot_token,
            self.policy_id,
            self.system.system_id,
            geometry.geometry_lineage_id,
        )


__all__ = [
    "AtmosphereCorrectionLedger",
    "AtmosphereFloorPolicy",
    "AtmosphereFloorStatus",
    "GRHDC2PCandidateRecord",
    "GRHDC2PPolicy",
    "GRHDC2PResult",
    "GRHDC2PStatus",
]
