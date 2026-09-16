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

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_mhd import (
    IdealValenciaGRMHDSystem,
    ValenciaPrimitiveRecovery,
)
from ..equations._relativistic_radiation_interaction import (
    GRGreyRadiationInteractionPlan,
    GRRadiationMatterExchange,
)
from ..metrix._adm_exchange import ADMGridGeometry
from ..nonlinear import SmallRootKernel
from ..nonlinear._batched import BatchedRootResult


class GRRMHDSourceStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    TEMPERATURE_UNAVAILABLE = 2
    PRIMARY_NONCONVERGENCE = 3
    SECONDARY_NONCONVERGENCE = 4
    MATERIAL_RECOVERY_FAILED = 5
    RADIATION_UNREALIZABLE = 6
    OPACITY_UNQUALIFIED = 7
    NONFINITE = 8
    CONSERVATION_DEFECT = 9


class GRRadiationExchangeLedger(StrictModule):
    material_energy_change: Array
    radiation_energy_change: Array
    material_momentum_change: Array
    radiation_momentum_change: Array
    energy_defect: Array
    momentum_defect: Array
    maximum_residual: Array
    maximum_iterations: Array
    primary_selected: Array
    jacobian_fallback: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRRMHDSourceResult(StrictModule):
    material_candidate: Array
    radiation_candidate: Array
    material_state: Array
    radiation_state: Array
    material_recovery: ValenciaPrimitiveRecovery
    exchange: GRRadiationMatterExchange
    primary: BatchedRootResult
    secondary: BatchedRootResult
    ledger: GRRadiationExchangeLedger
    accepted: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class _SourceArguments(StrictModule):
    material_initial: Array
    radiation_initial: Array
    coefficient: Array
    composition: Array
    alpha: Array
    beta: Array
    metric: Array
    inverse_metric: Array
    sqrt_metric: Array
    extrinsic: Array
    active: Array
    valid: Array
    snapshot_token: Array


def _cell_geometry(
    arguments: _SourceArguments,
    chart_id: str,
    convention_id: str,
    scale_id: str,
    topology_id: str,
    geometry_lineage_id: str,
    /,
) -> ADMGridGeometry:
    return ADMGridGeometry(
        arguments.alpha,
        arguments.beta,
        arguments.metric,
        arguments.inverse_metric,
        arguments.sqrt_metric,
        arguments.extrinsic,
        arguments.active,
        arguments.valid,
        snapshot_token=arguments.snapshot_token,
        chart_id=chart_id,
        convention_id=convention_id,
        scale_id=scale_id,
        topology_id=topology_id,
        geometry_lineage_id=geometry_lineage_id,
    )


def _bounded_vector_from_raw(
    raw: Array, inverse_metric: Array, maximum_norm: Array, /
) -> Array:
    norm = jnp.sqrt(
        jnp.maximum(
            ein.contract("i,ij,j->", raw, inverse_metric, raw),
            jnp.finfo(raw.dtype).tiny,
        )
    )
    return raw * (maximum_norm * jnp.tanh(norm) / norm)


def _raw_from_bounded_vector(
    vector: Array, inverse_metric: Array, maximum_norm: Array, /
) -> Array:
    norm = jnp.sqrt(
        jnp.maximum(ein.contract("i,ij,j->", vector, inverse_metric, vector), 0.0)
    )
    fraction = jnp.clip(
        norm / jnp.maximum(maximum_norm, jnp.finfo(vector.dtype).tiny),
        0.0,
        1.0 - 64.0 * jnp.finfo(vector.dtype).eps,
    )
    scale = jnp.where(norm > 0.0, jnp.arctanh(fraction) / norm, 1.0)
    return vector * scale


class _SourceResidualBase(StrictModule):
    __strict_abstract__ = True

    material: IdealValenciaGRMHDSystem
    interaction: GRGreyRadiationInteractionPlan
    caloric_temperature_scale: float | None = eqx.field(static=True)
    uses_composition: bool = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)

    def geometry(self, arguments: _SourceArguments, /) -> ADMGridGeometry:
        return _cell_geometry(
            arguments,
            self.chart_id,
            self.convention_id,
            self.scale_id,
            self.topology_id,
            self.geometry_lineage_id,
        )

    def composition(self, arguments: _SourceArguments, /) -> Array | None:
        return arguments.composition if self.uses_composition else None

    def matter_temperature(
        self,
        recovery: ValenciaPrimitiveRecovery,
        arguments: _SourceArguments,
        /,
    ) -> tuple[Array, Array]:
        primitive = recovery.primitive
        eos = self.material.eos.evaluate_pressure(
            primitive[0], primitive[4], self.composition(arguments)
        )
        defined = eos.evidence.temperature_defined & jnp.isfinite(eos.temperature)
        if self.caloric_temperature_scale is None:
            fallback = jnp.asarray(jnp.nan, dtype=primitive.dtype)
        else:
            fallback = (
                jnp.asarray(self.caloric_temperature_scale, dtype=primitive.dtype)
                * primitive[4]
                / jnp.maximum(primitive[0], jnp.finfo(primitive.dtype).tiny)
            )
        temperature = jnp.where(defined, eos.temperature, fallback)
        available = jnp.isfinite(temperature) & (temperature >= 0.0)
        return temperature, available

    def exchange(
        self,
        material_state: Array,
        radiation_state: Array,
        arguments: _SourceArguments,
        /,
    ) -> tuple[Array, ValenciaPrimitiveRecovery, GRRadiationMatterExchange, Array]:
        geometry = self.geometry(arguments)
        recovery = self.material.recover(
            material_state, geometry, self.composition(arguments)
        )
        temperature, temperature_available = self.matter_temperature(recovery, arguments)
        moments = radiation_state / arguments.sqrt_metric
        primitive = recovery.primitive
        magnetic_covector = ein.contract(
            "ij,j->i", geometry.spatial_metric, primitive[5:8]
        )
        magnetic_squared = ein.contract("i,i->", magnetic_covector, primitive[5:8])
        exchange = self.interaction.matter_exchange(
            moments[0],
            moments[1:],
            primitive[0],
            primitive[1:4],
            temperature,
            geometry,
            magnetic_squared=magnetic_squared,
            composition=self.composition(arguments),
        )
        source = (
            arguments.coefficient
            * geometry.alpha
            * geometry.sqrt_det_spatial_metric
            * jnp.concatenate(
                (
                    exchange.radiation_energy_source[None],
                    exchange.radiation_flux_source,
                )
            )
        )
        source = jnp.where(arguments.active, source, jnp.zeros_like(source))
        return source, recovery, exchange, temperature_available


class _RadiationVariableResidual(_SourceResidualBase):
    def radiation_state(self, raw: Array, arguments: _SourceArguments, /) -> Array:
        energy = jnp.exp(raw[0])
        maximum = (
            jnp.asarray(self.interaction.radiation.physical_light_speed, dtype=raw.dtype)
            * energy
        )
        flux = _bounded_vector_from_raw(raw[1:], arguments.inverse_metric, maximum)
        moments = jnp.concatenate((energy[None], flux))
        return arguments.sqrt_metric * moments

    @staticmethod
    def material_state(radiation_state: Array, arguments: _SourceArguments, /) -> Array:
        change = radiation_state - arguments.radiation_initial
        material = arguments.material_initial.at[1:4].add(-change[1:])
        return material.at[4].add(-change[0])

    def __call__(self, raw: Array, arguments: _SourceArguments, /) -> Array:
        radiation = self.radiation_state(raw, arguments)
        material = self.material_state(radiation, arguments)
        source, _recovery, _exchange, _temperature = self.exchange(
            material, radiation, arguments
        )
        scale = jnp.maximum(jnp.abs(arguments.radiation_initial), 1.0)
        return (radiation - arguments.radiation_initial - source) / scale


class _MaterialVariableResidual(_SourceResidualBase):
    def primitive(self, raw: Array, arguments: _SourceArguments, /) -> Array:
        pressure = jnp.exp(raw[0])
        velocity_covector = _bounded_vector_from_raw(
            raw[1:], arguments.inverse_metric, jnp.asarray(1.0, dtype=raw.dtype)
        )
        velocity = ein.contract("ij,j->i", arguments.inverse_metric, velocity_covector)
        velocity_squared = ein.contract("i,i->", velocity_covector, velocity)
        lorentz = 1.0 / jnp.sqrt(
            jnp.maximum(1.0 - velocity_squared, jnp.finfo(raw.dtype).tiny)
        )
        density = arguments.material_initial[0] / (arguments.sqrt_metric * lorentz)
        magnetic = arguments.material_initial[5:8] / arguments.sqrt_metric
        return jnp.concatenate((density[None], velocity, pressure[None], magnetic))

    def material_state(self, raw: Array, arguments: _SourceArguments, /) -> Array:
        return self.material.primitive_to_conserved(
            self.primitive(raw, arguments),
            self.geometry(arguments),
            self.composition(arguments),
        )

    @staticmethod
    def radiation_state(material_state: Array, arguments: _SourceArguments, /) -> Array:
        radiation = arguments.radiation_initial.at[1:].add(
            -(material_state[1:4] - arguments.material_initial[1:4])
        )
        return radiation.at[0].add(-(material_state[4] - arguments.material_initial[4]))

    def __call__(self, raw: Array, arguments: _SourceArguments, /) -> Array:
        material = self.material_state(raw, arguments)
        radiation = self.radiation_state(material, arguments)
        source, _recovery, _exchange, _temperature = self.exchange(
            material, radiation, arguments
        )
        scale = jnp.maximum(jnp.abs(arguments.radiation_initial), 1.0)
        return (radiation - arguments.radiation_initial - source) / scale


class GRRMHDImplicitSourcePlan(StrictModule, NonTrainableState):
    """Cell-local conservative implicit radiation-GRMHD source solve."""

    material: IdealValenciaGRMHDSystem
    interaction: GRGreyRadiationInteractionPlan
    maximum_iterations: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    caloric_temperature_scale: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: IdealValenciaGRMHDSystem,
        interaction: GRGreyRadiationInteractionPlan,
        /,
        *,
        maximum_iterations: int = 24,
        absolute_tolerance: float = 1.0e-9,
        relative_tolerance: float = 1.0e-8,
        minimum_damping: float = 1.0e-4,
        balance_tolerance: float = 1.0e-9,
        caloric_temperature_scale: float | None = None,
    ) -> None:
        if not isinstance(material, IdealValenciaGRMHDSystem):
            raise TypeError("material must be IdealValenciaGRMHDSystem.")
        if not isinstance(interaction, GRGreyRadiationInteractionPlan):
            raise TypeError("interaction must be GRGreyRadiationInteractionPlan.")
        if (
            material.scale.scale_id != interaction.radiation.scale.scale_id
            or material.convention.convention_id
            != interaction.radiation.convention.convention_id
        ):
            raise ValueError("GRMHD and radiation source contracts differ.")
        if float(material.scale.speed_of_light) != 1.0 or (
            interaction.radiation.reduced_light_speed
            != interaction.radiation.physical_light_speed
        ):
            raise ValueError("Coupled GRRMHD requires physical geometric light speed.")
        iterations = int(maximum_iterations)
        controls = (
            float(absolute_tolerance),
            float(relative_tolerance),
            float(minimum_damping),
            float(balance_tolerance),
        )
        caloric = (
            None
            if caloric_temperature_scale is None
            else float(caloric_temperature_scale)
        )
        if (
            iterations <= 0
            or any(not np.isfinite(value) or value <= 0.0 for value in controls)
            or (caloric is not None and (not np.isfinite(caloric) or caloric <= 0.0))
        ):
            raise ValueError("GRRMHD implicit source controls are invalid.")
        self.material = material
        self.interaction = interaction
        self.maximum_iterations = iterations
        self.absolute_tolerance = controls[0]
        self.relative_tolerance = controls[1]
        self.minimum_damping = controls[2]
        self.balance_tolerance = controls[3]
        self.caloric_temperature_scale = caloric
        self.plan_id = canonical_fingerprint(
            {
                "kind": "grrmhd-implicit-source",
                "material": material.system_id,
                "interaction": interaction.interaction_id,
                "maximum_iterations": iterations,
                "absolute_tolerance": controls[0],
                "relative_tolerance": controls[1],
                "minimum_damping": controls[2],
                "balance_tolerance": controls[3],
                "caloric_temperature_scale": caloric,
            }
        )

    def _residuals(
        self,
        geometry: ADMGridGeometry,
        uses_composition: bool,
        /,
    ) -> tuple[_RadiationVariableResidual, _MaterialVariableResidual]:
        common = {
            "material": self.material,
            "interaction": self.interaction,
            "caloric_temperature_scale": self.caloric_temperature_scale,
            "uses_composition": uses_composition,
            "chart_id": geometry.chart_id,
            "convention_id": geometry.convention_id,
            "scale_id": geometry.scale_id,
            "topology_id": geometry.topology_id,
            "geometry_lineage_id": geometry.geometry_lineage_id,
        }
        return _RadiationVariableResidual(**common), _MaterialVariableResidual(**common)

    def _kernel(self, residual) -> SmallRootKernel:
        return SmallRootKernel(
            residual,
            maximum_dimension=4,
            maximum_steps=self.maximum_iterations,
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
            minimum_damping=self.minimum_damping,
        )

    def advance(
        self,
        material_state: ArrayLike,
        radiation_state: ArrayLike,
        coefficient: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
    ) -> GRRMHDSourceResult:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        material = self.material._state(material_state, "GRRMHD material source state")
        radiation = jnp.asarray(radiation_state, dtype=material.dtype)
        if material.shape[:-1] != geometry.leading_shape or radiation.shape != (
            geometry.leading_shape + (4,)
        ):
            raise ValueError("GRRMHD source fields must match ADM geometry.")
        coefficient_ = jnp.asarray(coefficient, dtype=material.dtype).reshape(())
        composition_ = (
            jnp.zeros(geometry.leading_shape, dtype=material.dtype)
            if composition is None
            else jnp.broadcast_to(
                jnp.asarray(composition, dtype=material.dtype), geometry.leading_shape
            )
        )
        shape = geometry.leading_shape
        lane_count = int(np.prod(shape)) if shape else 1
        initial_recovery = self.material.recover(material, geometry, composition)
        initial_moments = radiation / geometry.sqrt_det_spatial_metric[..., None]
        energy = initial_moments[..., 0]
        flux = initial_moments[..., 1:]
        maximum = self.interaction.radiation.physical_light_speed * energy
        raw_flux = jax.vmap(_raw_from_bounded_vector)(
            flux.reshape((lane_count, 3)),
            geometry.inverse_spatial_metric.reshape((lane_count, 3, 3)),
            maximum.reshape((lane_count,)),
        )
        primary_initial = jnp.concatenate(
            (jnp.log(energy.reshape((lane_count, 1))), raw_flux), axis=-1
        )
        primitive = initial_recovery.primitive.reshape((lane_count, 8))
        velocity_covector = ein.contract(
            "...ij,...j->...i",
            geometry.spatial_metric.reshape((lane_count, 3, 3)),
            primitive[:, 1:4],
        )
        velocity_raw = jax.vmap(_raw_from_bounded_vector)(
            velocity_covector,
            geometry.inverse_spatial_metric.reshape((lane_count, 3, 3)),
            jnp.ones((lane_count,), dtype=material.dtype),
        )
        secondary_initial = jnp.concatenate(
            (jnp.log(primitive[:, 4:5]), velocity_raw), axis=-1
        )
        arguments = _SourceArguments(
            material.reshape((lane_count, 8)),
            radiation.reshape((lane_count, 4)),
            jnp.broadcast_to(coefficient_, (lane_count,)),
            composition_.reshape((lane_count,)),
            geometry.alpha.reshape((lane_count,)),
            geometry.beta_contravariant.reshape((lane_count, 3)),
            geometry.spatial_metric.reshape((lane_count, 3, 3)),
            geometry.inverse_spatial_metric.reshape((lane_count, 3, 3)),
            geometry.sqrt_det_spatial_metric.reshape((lane_count,)),
            geometry.extrinsic_curvature.reshape((lane_count, 3, 3)),
            geometry.active.reshape((lane_count,)),
            geometry.valid.reshape((lane_count,)),
            jnp.broadcast_to(geometry.snapshot_token, (lane_count,)),
        )
        primary_residual, secondary_residual = self._residuals(
            geometry, composition is not None
        )
        primary = self._kernel(primary_residual).solve(primary_initial, arguments)
        secondary = self._kernel(secondary_residual).solve(secondary_initial, arguments)
        primary_radiation = jax.vmap(primary_residual.radiation_state)(
            primary.state, arguments
        )
        primary_material = jax.vmap(primary_residual.material_state)(
            primary_radiation, arguments
        )
        secondary_material = jax.vmap(secondary_residual.material_state)(
            secondary.state, arguments
        )
        secondary_radiation = jax.vmap(secondary_residual.radiation_state)(
            secondary_material, arguments
        )
        primary_recovery = self.material.recover(
            primary_material.reshape(material.shape), geometry, composition
        )
        secondary_recovery = self.material.recover(
            secondary_material.reshape(material.shape), geometry, composition
        )
        primary_moments = (
            primary_radiation.reshape(radiation.shape)
            / geometry.sqrt_det_spatial_metric[..., None]
        )
        secondary_moments = (
            secondary_radiation.reshape(radiation.shape)
            / geometry.sqrt_det_spatial_metric[..., None]
        )
        primary_closure = self.interaction.radiation.closure(
            primary_moments[..., 0], primary_moments[..., 1:], geometry
        )
        secondary_closure = self.interaction.radiation.closure(
            secondary_moments[..., 0], secondary_moments[..., 1:], geometry
        )
        primary_ok = (
            primary.successful.reshape(shape)
            & primary_recovery.qualified
            & primary_closure.qualified
        )
        secondary_ok = (
            secondary.successful.reshape(shape)
            & secondary_recovery.qualified
            & secondary_closure.qualified
        )
        use_primary = primary_ok
        selected_material = jnp.where(
            use_primary[..., None],
            primary_material.reshape(material.shape),
            secondary_material.reshape(material.shape),
        )
        selected_radiation = jnp.where(
            use_primary[..., None],
            primary_radiation.reshape(radiation.shape),
            secondary_radiation.reshape(radiation.shape),
        )
        selected_recovery = self.material.recover(
            selected_material, geometry, composition
        )
        selected_raw = jnp.where(
            use_primary.reshape((lane_count, 1)), primary.state, secondary.state
        )
        selected_arguments = arguments
        selected_exchange_values = jax.vmap(primary_residual.exchange)(
            selected_material.reshape((lane_count, 8)),
            selected_radiation.reshape((lane_count, 4)),
            selected_arguments,
        )
        exchange = jax.tree.map(
            lambda value: (
                value.reshape(shape + value.shape[1:])
                if isinstance(value, jax.Array) and value.shape[:1] == (lane_count,)
                else value
            ),
            selected_exchange_values[2],
        )
        temperature_available = selected_exchange_values[3].reshape(shape)
        lane_success = (
            jnp.where(use_primary, primary_ok, secondary_ok)
            & temperature_available
            & exchange.qualified
        )
        accepted = jnp.all(lane_success | ~geometry.active)
        material_candidate = selected_material
        radiation_candidate = selected_radiation
        material_accepted = jnp.where(accepted, material_candidate, material)
        radiation_accepted = jnp.where(accepted, radiation_candidate, radiation)
        material_energy_change = material_candidate[..., 4] - material[..., 4]
        radiation_energy_change = radiation_candidate[..., 0] - radiation[..., 0]
        material_momentum_change = material_candidate[..., 1:4] - material[..., 1:4]
        radiation_momentum_change = radiation_candidate[..., 1:] - radiation[..., 1:]
        energy_defect = material_energy_change + radiation_energy_change
        momentum_defect = material_momentum_change + radiation_momentum_change
        finite = (
            jnp.all(jnp.isfinite(material_candidate))
            & jnp.all(jnp.isfinite(radiation_candidate))
            & jnp.all(jnp.isfinite(energy_defect))
            & jnp.all(jnp.isfinite(momentum_defect))
        )
        scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(material_energy_change), initial=0.0),
                jnp.max(jnp.abs(radiation_energy_change), initial=0.0),
            ),
            1.0,
        )
        tolerance = jnp.maximum(
            jnp.asarray(self.balance_tolerance, dtype=material.dtype),
            256.0 * jnp.finfo(material.dtype).eps * scale,
        )
        balance = (jnp.max(jnp.abs(energy_defect), initial=0.0) <= tolerance) & (
            jnp.max(jnp.abs(momentum_defect), initial=0.0) <= tolerance
        )
        qualified = accepted & finite & balance
        status = jnp.where(
            qualified,
            int(GRRMHDSourceStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(GRRMHDSourceStatus.NONFINITE),
                jnp.where(
                    ~jnp.all(temperature_available | ~geometry.active),
                    int(GRRMHDSourceStatus.TEMPERATURE_UNAVAILABLE),
                    jnp.where(
                        ~jnp.all(primary_ok | secondary_ok | ~geometry.active),
                        int(GRRMHDSourceStatus.SECONDARY_NONCONVERGENCE),
                        jnp.where(
                            ~balance,
                            int(GRRMHDSourceStatus.CONSERVATION_DEFECT),
                            int(GRRMHDSourceStatus.OPACITY_UNQUALIFIED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        maximum_residual = jnp.maximum(
            jnp.max(primary.residual_norm, initial=0.0),
            jnp.max(secondary.residual_norm, initial=0.0),
        )
        maximum_iterations = jnp.maximum(
            jnp.max(primary.iterations, initial=0),
            jnp.max(secondary.iterations, initial=0),
        )
        selected_fallback = jnp.where(
            use_primary.reshape((lane_count,)),
            primary.jacobian_fallback,
            secondary.jacobian_fallback,
        ).reshape(shape)
        ledger = GRRadiationExchangeLedger(
            material_energy_change,
            radiation_energy_change,
            material_momentum_change,
            radiation_momentum_change,
            energy_defect,
            momentum_defect,
            maximum_residual,
            maximum_iterations,
            jnp.all(use_primary | ~geometry.active),
            selected_fallback,
            accepted,
            finite,
            qualified,
            self.plan_id,
        )
        derivative_valid = (
            qualified
            & jnp.all(selected_recovery.derivative_valid | ~geometry.active)
            & jnp.all(exchange.derivative_valid | ~geometry.active)
            & jnp.all(~selected_fallback | ~geometry.active)
        )
        del selected_raw
        return GRRMHDSourceResult(
            material_candidate,
            radiation_candidate,
            material_accepted,
            radiation_accepted,
            selected_recovery,
            exchange,
            primary,
            secondary,
            ledger,
            accepted,
            status,
            finite,
            jnp.all(primary.successful | secondary.successful),
            jnp.all(selected_recovery.physically_valid | ~geometry.active)
            & jnp.all(exchange.physically_valid | ~geometry.active),
            qualified,
            derivative_valid,
            self.plan_id,
        )


__all__ = [
    "GRRMHDImplicitSourcePlan",
    "GRRMHDSourceResult",
    "GRRMHDSourceStatus",
    "GRRadiationExchangeLedger",
]
