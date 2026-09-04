#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._lagrangian_marker import LagrangianMarkerKinematics
from ...discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    MACMarkerRelation,
    MACVariationalViscosityResult,
    PreparedMACMomentumOperators,
)
from ...equations._incompressible import IncompressibleFlowProblem
from ...equations._les_closures import (
    AlgebraicLESInputs,
    AlgebraicLESResult,
    LESFilterScale,
)
from ...equations._mac_incompressible import (
    compile_mac_incompressible_flow,
    CompiledMACIncompressibleDynamics,
)
from ...equations._mac_les import MACAlgebraicLESPlan, PreparedMACAlgebraicLES
from ...solver._mac_immersed_boundary import MACImmersedBoundaryProjectionPlan
from ...solver._mac_immersed_step import (
    MACImmersedBoundaryIMEXEulerMethod,
    MACImmersedBoundaryIMEXEulerResult,
    MACImmersedBoundarySBDF2Method,
    MACImmersedBoundarySBDF2Result,
    MACImmersedBoundarySBDF2State,
)
from ...solver._structured_incompressible import MACPressureProjectionPlan
from ._boundary_turbulence import (
    PreparedVectorEquilibriumWallStress,
    VectorEquilibriumWallStressResult,
)
from ._immersed_support import ImmersedBodyRegimePlan


ImmersedLESMotion = Literal["fixed", "moving", "deforming"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return normalized


class FixedImmersedMarkerMotion(StrictModule, NonTrainableState):
    """Stationary, fixed-route marker state bound to one geometry identity."""

    kinematics: LagrangianMarkerKinematics
    motion_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        kinematics: LagrangianMarkerKinematics,
        /,
        *,
        geometry_id: str,
    ):
        if not isinstance(kinematics, LagrangianMarkerKinematics):
            raise TypeError("kinematics must be LagrangianMarkerKinematics.")
        geometry = _identifier(geometry_id, "geometry_id")
        self.kinematics = kinematics
        self.geometry_id = geometry
        self.motion_id = canonical_fingerprint(
            {
                "kind": "fixed-immersed-marker-motion",
                "geometry": geometry,
                "markers": kinematics.markers_id,
                "position": array_tree_fingerprint(kinematics.position),
                "velocity": array_tree_fingerprint(kinematics.velocity),
            }
        )

    def __call__(self, _time: Array, _args: Any) -> LagrangianMarkerKinematics:
        return self.kinematics


class ImmersedMACLESStageResult(StrictModule):
    """Masked algebraic SGS action plus optional marker wall-stress action."""

    velocity_gradient: Array
    strain: Array
    filter_scale: LESFilterScale
    model_result: AlgebraicLESResult
    viscosity_result: MACVariationalViscosityResult
    physical_rate: FaceVelocity
    sgs_rate: FaceVelocity
    wall_rate: FaceVelocity
    fluid_volume_fraction: Array
    relation: MACMarkerRelation
    wall_stress: VectorEquilibriumWallStressResult | None
    wall_traction_density: Array
    integrated_work: Array
    sgs_integrated_work: Array
    boundary_power: Array
    modeled_wall_power: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    boundary_stage_id: str = eqx.field(static=True)


class ImmersedLESBalanceLedger(StrictModule):
    """Action/reaction, impulse, and stress-work evidence for one accepted step."""

    constraint_fluid_traction_density: Array
    modeled_wall_fluid_traction_density: Array
    total_fluid_traction_density: Array
    body_traction_density: Array
    fluid_impulse: Array
    body_impulse: Array
    impulse_balance_residual: Array
    fluid_stress_work: Array
    marker_stress_work: Array
    transfer_work_residual: Array
    body_mechanical_work: Array
    slip_dissipation_work: Array
    sgs_bulk_work: Array
    modeled_wall_work: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class FixedImmersedMACLESPlan(StrictModule, NonTrainableState):
    """Exact fixed-topology marker/MAC LES coupling.

    The admitted route is three-dimensional, unit-density, single-device MAC LES
    with a stationary fixed marker relation and periodic/free-slip/symmetry outer
    boundaries. ``cell_fluid_fraction`` is caller-owned geometry evidence: it
    weights SGS stress and defines the active-fluid-volume filter width. Moving,
    deforming, distributed, and open-boundary requests fail during preparation.
    """

    algebraic_les: MACAlgebraicLESPlan
    projection: MACImmersedBoundaryProjectionPlan
    marker_motion: FixedImmersedMarkerMotion
    cell_fluid_fraction: Array
    wall_stress: PreparedVectorEquilibriumWallStress | None
    marker_wall_normal: Array
    marker_sample_distance: Array
    marker_roughness_height: Array
    motion: ImmersedLESMotion = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebraic_les: MACAlgebraicLESPlan,
        projection: MACImmersedBoundaryProjectionPlan,
        marker_kinematics: LagrangianMarkerKinematics,
        cell_fluid_fraction: ArrayLike,
        /,
        *,
        geometry_id: str,
        motion: ImmersedLESMotion = "fixed",
        distributed: bool = False,
        wall_stress: PreparedVectorEquilibriumWallStress | None = None,
        marker_wall_normal: ArrayLike | None = None,
        marker_sample_distance: ArrayLike | None = None,
        marker_roughness_height: ArrayLike = 0.0,
    ):
        if not isinstance(algebraic_les, MACAlgebraicLESPlan):
            raise TypeError("algebraic_les must be MACAlgebraicLESPlan.")
        if not isinstance(projection, MACImmersedBoundaryProjectionPlan):
            raise TypeError("projection must be MACImmersedBoundaryProjectionPlan.")
        if motion not in ("fixed", "moving", "deforming"):
            raise ValueError("Unknown immersed LES motion regime.")
        geometry = _identifier(geometry_id, "geometry_id")
        kinematics = projection.transfer.markers.validate_kinematics(marker_kinematics)
        fraction = np.asarray(cell_fluid_fraction)
        expected_cells = projection.operators.discretization.cell_shape
        if fraction.shape != expected_cells:
            raise ValueError(
                f"cell_fluid_fraction must have shape {expected_cells}; got {fraction.shape}."
            )
        if (
            not np.issubdtype(fraction.dtype, np.inexact)
            or np.any(~np.isfinite(fraction))
            or np.any((fraction < 0.0) | (fraction > 1.0))
            or not np.any(fraction > 0.0)
        ):
            raise ValueError(
                "cell_fluid_fraction must be finite, floating, lie in [0, 1], and "
                "contain active fluid."
            )
        capacity = projection.transfer.markers.capacity
        dimension = projection.transfer.dimension
        if wall_stress is None:
            if marker_wall_normal is not None or marker_sample_distance is not None:
                raise ValueError(
                    "Marker wall geometry requires a prepared vector wall-stress model."
                )
            normal = np.zeros((capacity, dimension), dtype=fraction.dtype)
            distance = np.zeros((capacity,), dtype=fraction.dtype)
            roughness = np.zeros((capacity,), dtype=fraction.dtype)
        else:
            if not isinstance(wall_stress, PreparedVectorEquilibriumWallStress):
                raise TypeError(
                    "wall_stress must be PreparedVectorEquilibriumWallStress or None."
                )
            if marker_wall_normal is None or marker_sample_distance is None:
                raise ValueError(
                    "Vector wall stress requires marker_wall_normal and "
                    "marker_sample_distance."
                )
            normal = np.asarray(marker_wall_normal)
            distance = np.asarray(marker_sample_distance)
            roughness_raw = np.asarray(marker_roughness_height)
            if normal.shape != (capacity, dimension):
                raise ValueError("marker_wall_normal has an incompatible shape.")
            if distance.shape != (capacity,):
                raise ValueError("marker_sample_distance has an incompatible shape.")
            if roughness_raw.shape not in ((), (capacity,)):
                raise ValueError(
                    "marker_roughness_height must be scalar or have marker capacity."
                )
            roughness = np.broadcast_to(roughness_raw, (capacity,)).copy()
            active = np.asarray(projection.transfer.markers.active_mask)
            normal_norm = np.linalg.norm(normal[active], axis=-1)
            if (
                np.any(~np.isfinite(normal[active]))
                or np.any(~np.isfinite(distance[active]))
                or np.any(~np.isfinite(roughness[active]))
                or np.any(normal_norm <= 0.0)
                or np.any(distance[active] <= 0.0)
                or np.any(roughness[active] < 0.0)
            ):
                raise ValueError("Active marker wall geometry is invalid.")
        dtype = projection.operators.pressure_space.dtype
        self.algebraic_les = algebraic_les
        self.projection = projection
        self.marker_motion = FixedImmersedMarkerMotion(kinematics, geometry_id=geometry)
        self.cell_fluid_fraction = jnp.asarray(fraction, dtype=dtype)
        self.wall_stress = wall_stress
        self.marker_wall_normal = jnp.asarray(normal, dtype=dtype)
        self.marker_sample_distance = jnp.asarray(distance, dtype=dtype)
        self.marker_roughness_height = jnp.asarray(roughness, dtype=dtype)
        self.motion = motion
        self.distributed = bool(distributed)
        self.geometry_id = geometry
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-immersed-mac-les-plan",
                "algebraic_les": algebraic_les.plan_id,
                "filter": algebraic_les.prepared_model.provenance.resolved_filter.filter_id,
                "model": algebraic_les.prepared_model.prepared_id,
                "geometry": geometry,
                "fluid_fraction": array_tree_fingerprint(self.cell_fluid_fraction),
                "markers": projection.transfer.markers.prepared_id,
                "marker_transfer": projection.transfer.prepared_id,
                "marker_motion": self.marker_motion.motion_id,
                "boundaries": projection.boundaries.prepared_id,
                "solver": projection.plan_id,
                "motion": motion,
                "distributed": bool(distributed),
                "wall_stress": None if wall_stress is None else wall_stress.prepared_id,
                "wall_normal": None
                if wall_stress is None
                else array_tree_fingerprint(self.marker_wall_normal),
                "wall_distance": None
                if wall_stress is None
                else array_tree_fingerprint(self.marker_sample_distance),
                "wall_roughness": None
                if wall_stress is None
                else array_tree_fingerprint(self.marker_roughness_height),
            }
        )

    def admission_regime(self, /) -> ImmersedBodyRegimePlan:
        """Bind this route to the existing prescribed-marker admission contract."""
        markers = self.projection.transfer.markers
        return ImmersedBodyRegimePlan(
            self.projection,
            marker_set_id=markers.prepared_id,
            geometry_id=self.geometry_id,
            route_id=self.projection.transfer.prepared_id,
            topology_epoch_id=markers.geometry_layout_id,
            geometry_epoch=0,
            moving=False,
            fixed_topology=True,
            marker_constraint_count=(
                markers.active_count if self.wall_stress is not None else None
            ),
        )

    def imex_euler_method(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        fixed_step_size: float,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        maximum_resource_bytes: int = 512 * 1024**2,
    ) -> MACImmersedBoundaryIMEXEulerMethod:
        """Build the bound full-no-slip or normal/wall-stress IMEX route."""
        if (
            not isinstance(dynamics, CompiledMACIncompressibleDynamics)
            or not isinstance(dynamics.algebraic_les, PreparedFixedImmersedMACLES)
            or dynamics.algebraic_les.plan.plan_id != self.plan_id
        ):
            raise ValueError("Dynamics does not contain this immersed LES plan.")
        normals = self.marker_wall_normal if self.wall_stress is not None else None
        return MACImmersedBoundaryIMEXEulerMethod(
            dynamics,
            self.projection,
            self.marker_motion,
            motion_id=self.marker_motion.motion_id,
            fixed_step_size=fixed_step_size,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            marker_constraint_normals=normals,
            maximum_resource_bytes=maximum_resource_bytes,
        )

    def sbdf2_method(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        step_size: float,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        maximum_resource_bytes: int = 512 * 1024**2,
    ) -> MACImmersedBoundarySBDF2Method:
        """Build the bound full-no-slip or normal/wall-stress SBDF2 route."""
        if (
            not isinstance(dynamics, CompiledMACIncompressibleDynamics)
            or not isinstance(dynamics.algebraic_les, PreparedFixedImmersedMACLES)
            or dynamics.algebraic_les.plan.plan_id != self.plan_id
        ):
            raise ValueError("Dynamics does not contain this immersed LES plan.")
        normals = self.marker_wall_normal if self.wall_stress is not None else None
        return MACImmersedBoundarySBDF2Method(
            dynamics,
            self.projection,
            self.marker_motion,
            step_size,
            motion_id=self.marker_motion.motion_id,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            marker_constraint_normals=normals,
            maximum_resource_bytes=maximum_resource_bytes,
        )

    def prepare(
        self,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        molecular_viscosity: ArrayLike,
    ) -> PreparedFixedImmersedMACLES:
        if self.motion != "fixed":
            raise ValueError(
                "Immersed MAC LES admits only stationary fixed geometry; moving and "
                "deforming geometry require a separately qualified filter/metric route."
            )
        if self.distributed:
            raise ValueError(
                "Distributed immersed MAC LES is not admitted by the single-device "
                "marker-transfer and work-ledger route."
            )
        return PreparedFixedImmersedMACLES(
            self, momentum, molecular_viscosity=molecular_viscosity
        )


class PreparedFixedImmersedMACLES(StrictModule, NonTrainableState):
    """Prepared masked SGS and fixed marker wall-stress action."""

    plan: FixedImmersedMACLESPlan
    base: PreparedMACAlgebraicLES
    relation: MACMarkerRelation
    molecular_viscosity: Array
    prepared_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    marker_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    support_limits: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        plan: FixedImmersedMACLESPlan,
        momentum: PreparedMACMomentumOperators,
        /,
        *,
        molecular_viscosity: ArrayLike,
    ):
        if not isinstance(plan, FixedImmersedMACLESPlan):
            raise TypeError("plan must be FixedImmersedMACLESPlan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        projection = plan.projection
        if projection.operators.prepared_id != momentum.operators.prepared_id:
            raise ValueError(
                "Immersed projection and LES momentum must share MAC operators."
            )
        if projection.boundaries.prepared_id != momentum.boundaries.prepared_id:
            raise ValueError(
                "Immersed projection and LES momentum must share boundaries."
            )
        if momentum.dimension != 3:
            raise ValueError("Immersed MAC LES requires a three-dimensional MAC route.")
        viscosity = np.asarray(molecular_viscosity)
        if (
            viscosity.shape != ()
            or not np.isfinite(viscosity)
            or viscosity < 0.0
            or (plan.wall_stress is not None and viscosity <= 0.0)
        ):
            raise ValueError(
                "Molecular viscosity must be finite and nonnegative, and positive "
                "when vector equilibrium wall stress is active."
            )
        kinematics = projection.transfer.markers.validate_kinematics(
            plan.marker_motion.kinematics
        )
        active_velocity = np.asarray(
            projection.transfer.markers.active_values(kinematics.velocity)
        )
        if np.any(active_velocity != 0.0):
            raise ValueError("Fixed immersed LES requires stationary active markers.")
        relation = projection.transfer.relation(kinematics.position)
        if not bool(np.asarray(relation.successful)):
            raise ValueError(
                "Fixed immersed LES marker support is truncated or transfer-incompatible."
            )
        if plan.wall_stress is not None and plan.wall_stress.spatial_dimension != 3:
            raise ValueError("Immersed MAC LES wall stress must be prepared in 3D.")
        base = plan.algebraic_les.prepare(momentum)
        resolved_filter = base.model.provenance.resolved_filter
        self.plan = plan
        self.base = base
        self.relation = relation
        self.molecular_viscosity = jnp.asarray(
            viscosity, dtype=momentum.operators.pressure_space.dtype
        )
        self.filter_id = resolved_filter.filter_id
        self.model_id = base.model.prepared_id
        self.geometry_id = plan.geometry_id
        self.marker_id = projection.transfer.markers.prepared_id
        self.boundary_id = projection.boundaries.prepared_id
        self.solver_id = projection.plan_id
        self.support_limits = (
            "three-dimensional unit-density incompressible MAC",
            "stationary fixed marker identities and fixed transfer routes",
            "single-device deterministic marker transfer",
            "periodic, free-slip, or symmetry outer boundaries",
            "caller-owned fixed cell fluid fractions",
            "optional attached zero-pressure-gradient vector equilibrium wall stress",
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-immersed-mac-les",
                "plan": plan.plan_id,
                "base": base.prepared_id,
                "momentum": momentum.prepared_id,
                "filter": self.filter_id,
                "model": self.model_id,
                "geometry": self.geometry_id,
                "markers": self.marker_id,
                "relation": relation.relation_id,
                "boundaries": self.boundary_id,
                "solver": self.solver_id,
                "fluid_volume_rule": "alpha-times-full-cell-volume",
                "filter_scale_rule": "alpha^(1/3)-scaled-directional-widths-on-active-cells",
                "solid_mask_rule": "zero-SGS-stress",
                "wall_stress": None
                if plan.wall_stress is None
                else plan.wall_stress.prepared_id,
            }
        )

    def filter_scale(self, /) -> LESFilterScale:
        base_scale = self.base.filter_scale().directional_widths
        fraction = self.plan.cell_fluid_fraction
        factor = jnp.where(fraction > 0.0, jnp.cbrt(fraction), 1.0)
        return LESFilterScale(base_scale * factor[..., None])

    def velocity_gradient(self, velocity: FaceVelocity, /) -> Array:
        return self.base.velocity_gradient(velocity)

    def _masked_model(
        self, velocity: FaceVelocity, /
    ) -> tuple[Array, LESFilterScale, AlgebraicLESResult]:
        gradient = self.velocity_gradient(velocity)
        scale = self.filter_scale()
        raw = self.base.model.evaluate(AlgebraicLESInputs(gradient, scale))
        fraction = self.plan.cell_fluid_fraction.astype(raw.kinematic_viscosity.dtype)
        result = AlgebraicLESResult(
            fraction * raw.kinematic_viscosity,
            fraction[..., None, None] * raw.specific_deviatoric_stress,
            fraction * raw.energy_transfer,
        )
        return gradient, scale, result

    def _wall_action(
        self, velocity: FaceVelocity, /
    ) -> tuple[
        FaceVelocity,
        Array,
        VectorEquilibriumWallStressResult | None,
        Array,
        Array,
    ]:
        transfer = self.plan.projection.transfer
        marker_space = transfer.markers.active_velocity_space
        zero_traction = marker_space.zeros()
        zero_rate = tuple(jnp.zeros_like(value) for value in velocity)
        zero_power = jnp.asarray(
            0.0, dtype=self.base.momentum.operators.pressure_space.dtype
        )
        if self.plan.wall_stress is None:
            return zero_rate, zero_power, None, zero_traction, jnp.asarray(True)
        sampled = transfer.gather(self.relation, velocity)
        wall_velocity = transfer.markers.active_values(
            self.plan.marker_motion.kinematics.velocity
        )
        normal = transfer.markers.active_values(self.plan.marker_wall_normal)
        norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
        unit_normal = normal / norm[:, None]
        relative = sampled - wall_velocity
        tangential = (
            relative - jnp.sum(relative * unit_normal, axis=-1)[:, None] * unit_normal
        )
        distance = self.plan.marker_sample_distance[transfer.markers.active_indices]
        roughness = self.plan.marker_roughness_height[transfer.markers.active_indices]
        result = self.plan.wall_stress.evaluate(
            tangential,
            unit_normal,
            distance,
            jnp.asarray(1.0, dtype=tangential.dtype),
            self.molecular_viscosity,
            roughness_height=roughness,
        )
        traction = result.traction
        rate = transfer.spread(self.relation, traction)
        power = jnp.real(marker_space.inner(sampled, traction))
        successful = jnp.all(result.successful) & self.relation.successful
        return rate, power, result, traction, successful

    def step_restriction(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> tuple[Array, bool]:
        stage = self.base.momentum.boundaries.validate_stage(boundary_stage)
        values = self.base.momentum.boundaries.enforce(
            self.base.momentum.operators.validate_velocity(velocity), stage
        )
        _, _, result = self._masked_model(values)
        bound = self.base.viscosity_action.explicit_step_bound(result.kinematic_viscosity)
        supported = (
            self.base.viscosity_action.restriction_supported
            and self.plan.wall_stress is None
        )
        return bound, supported

    def evaluate(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
    ) -> ImmersedMACLESStageResult:
        stage = self.base.momentum.boundaries.validate_stage(boundary_stage)
        values = self.base.momentum.boundaries.enforce(
            self.base.momentum.operators.validate_velocity(velocity), stage
        )
        gradient, scale, model_result = self._masked_model(values)
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        viscosity_result = self.base.viscosity_action.evaluate(
            values, model_result.kinematic_viscosity, stage
        )
        wall_rate, wall_power, wall_result, wall_traction, wall_successful = (
            self._wall_action(values)
        )
        physical_rate = tuple(
            sgs + wall
            for sgs, wall in zip(
                viscosity_result.physical_diffusive_rate, wall_rate, strict=True
            )
        )
        model_finite = (
            jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(strain))
            & jnp.all(jnp.isfinite(scale.directional_widths))
            & jnp.all(jnp.isfinite(model_result.kinematic_viscosity))
            & jnp.all(jnp.isfinite(model_result.specific_deviatoric_stress))
            & jnp.all(jnp.isfinite(model_result.energy_transfer))
            & jnp.all(jnp.isfinite(wall_traction))
            & jnp.isfinite(wall_power)
        )
        finite = stage.finite & viscosity_result.finite & model_finite
        successful = (
            stage.successful & viscosity_result.successful & wall_successful & finite
        )
        return ImmersedMACLESStageResult(
            velocity_gradient=gradient,
            strain=strain,
            filter_scale=scale,
            model_result=model_result,
            viscosity_result=viscosity_result,
            physical_rate=physical_rate,
            sgs_rate=viscosity_result.physical_diffusive_rate,
            wall_rate=wall_rate,
            fluid_volume_fraction=self.plan.cell_fluid_fraction,
            relation=self.relation,
            wall_stress=wall_result,
            wall_traction_density=wall_traction,
            integrated_work=viscosity_result.integrated_work + wall_power,
            sgs_integrated_work=viscosity_result.integrated_work,
            boundary_power=viscosity_result.boundary_power + wall_power,
            modeled_wall_power=wall_power,
            finite=finite,
            successful=successful,
            prepared_id=self.prepared_id,
            boundary_stage_id=stage.stage_id,
        )

    def balance_ledger(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        step: MACImmersedBoundaryIMEXEulerResult | MACImmersedBoundarySBDF2Result,
        /,
        *,
        history: MACImmersedBoundarySBDF2State | None = None,
        args: Any = None,
    ) -> ImmersedLESBalanceLedger:
        """Close the applied IMEX-Euler or SBDF2 marker/stress ledger.

        A non-startup SBDF2 result requires the input ``history`` because its
        extrapolated SGS and modeled-wall actions use both retained states.
        """
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if (
            dynamics.compilation_id == ""
            or not isinstance(dynamics.algebraic_les, PreparedFixedImmersedMACLES)
            or dynamics.algebraic_les.prepared_id != self.prepared_id
        ):
            raise ValueError("Dynamics does not contain this immersed LES preparation.")
        supported_step = isinstance(
            step,
            (
                MACImmersedBoundaryIMEXEulerResult,
                MACImmersedBoundarySBDF2Result,
            ),
        )
        if not supported_step:
            raise TypeError("step must be an immersed IMEX-Euler or SBDF2 result.")
        if step.projection.projection_id != self.solver_id:
            raise ValueError("Step projection does not match the immersed LES solver.")

        def evaluated_stage(time, state):
            value = dynamics.rate_components(time, state, args).les_stage
            if not isinstance(value, ImmersedMACLESStageResult):
                raise TypeError("Dynamics did not produce an immersed LES stage.")
            return value

        transfer = self.plan.projection.transfer
        marker_space = transfer.markers.active_velocity_space
        dt = step.step_size
        if isinstance(step, MACImmersedBoundaryIMEXEulerResult):
            previous_time = step.attempted_time - dt
            current_stage = evaluated_stage(previous_time, step.previous_state)
            modeled = current_stage.wall_traction_density
            sgs_bulk_work = dt * current_stage.sgs_integrated_work
            modeled_wall_work = dt * current_stage.modeled_wall_power
            stage_finite = current_stage.finite
            stage_successful = current_stage.successful
        elif step.startup:
            previous_time = step.attempted_time - dt
            current_stage = evaluated_stage(previous_time, step.history.previous_state)
            modeled = current_stage.wall_traction_density
            sgs_bulk_work = dt * current_stage.sgs_integrated_work
            modeled_wall_work = dt * current_stage.modeled_wall_power
            stage_finite = current_stage.finite
            stage_successful = current_stage.successful
        else:
            if not isinstance(history, MACImmersedBoundarySBDF2State):
                raise ValueError(
                    "Non-startup SBDF2 ledger evaluation requires its input history."
                )
            if history.method_id != step.method_id:
                raise ValueError("SBDF2 input history belongs to another method.")
            current_stage = evaluated_stage(history.time, history.state)
            previous_stage = evaluated_stage(history.time - dt, history.previous_state)
            modeled = (
                2.0 * current_stage.wall_traction_density
                - previous_stage.wall_traction_density
            )
            extrapolated_sgs_rate = tuple(
                2.0 * current - previous
                for current, previous in zip(
                    current_stage.sgs_rate,
                    previous_stage.sgs_rate,
                    strict=True,
                )
            )
            sgs_bulk_work = dt * jnp.real(
                dynamics.momentum.operators.velocity_space.inner(
                    step.velocity, extrapolated_sgs_rate
                )
            )
            sampled_velocity = transfer.gather(step.projection.relation, step.velocity)
            modeled_wall_work = dt * jnp.real(
                marker_space.inner(sampled_velocity, modeled)
            )
            stage_finite = current_stage.finite & previous_stage.finite
            stage_successful = current_stage.successful & previous_stage.successful

        constraint = step.projection.marker_force_density
        total = constraint + modeled
        diagnostics = transfer.diagnostics(step.projection.relation, step.velocity, total)
        wall_velocity = transfer.markers.active_values(
            self.plan.marker_motion.kinematics.velocity
        )
        wall_velocity_power = jnp.real(marker_space.inner(wall_velocity, total))
        fluid_impulse = dt * diagnostics.face_resultant
        body_impulse = -dt * diagnostics.marker_resultant
        fluid_work = dt * diagnostics.spreading_work
        marker_work = dt * diagnostics.interpolation_work
        body_work = -dt * wall_velocity_power
        slip_dissipation = -(fluid_work + body_work)
        impulse_residual = fluid_impulse + body_impulse
        work_residual = fluid_work - marker_work
        finite = (
            diagnostics.finite
            & stage_finite
            & jnp.all(jnp.isfinite(impulse_residual))
            & jnp.isfinite(work_residual)
            & jnp.isfinite(slip_dissipation)
            & jnp.isfinite(sgs_bulk_work)
            & jnp.isfinite(modeled_wall_work)
        )
        successful = step.accepted & diagnostics.successful & stage_successful & finite
        return ImmersedLESBalanceLedger(
            constraint_fluid_traction_density=constraint,
            modeled_wall_fluid_traction_density=modeled,
            total_fluid_traction_density=total,
            body_traction_density=-total,
            fluid_impulse=fluid_impulse,
            body_impulse=body_impulse,
            impulse_balance_residual=impulse_residual,
            fluid_stress_work=fluid_work,
            marker_stress_work=marker_work,
            transfer_work_residual=work_residual,
            body_mechanical_work=body_work,
            slip_dissipation_work=slip_dissipation,
            sgs_bulk_work=sgs_bulk_work,
            modeled_wall_work=modeled_wall_work,
            finite=finite,
            successful=successful,
            prepared_id=self.prepared_id,
            projection_id=self.solver_id,
            geometry_id=self.geometry_id,
        )


def compile_fixed_immersed_mac_les_flow(
    problem: IncompressibleFlowProblem,
    momentum: PreparedMACMomentumOperators,
    pressure_projection: MACPressureProjectionPlan,
    immersed_les: FixedImmersedMACLESPlan,
    /,
) -> CompiledMACIncompressibleDynamics:
    """Compile the standard MAC dynamics with the fixed immersed LES preparation."""
    if not isinstance(immersed_les, FixedImmersedMACLESPlan):
        raise TypeError("immersed_les must be FixedImmersedMACLESPlan.")
    base = compile_mac_incompressible_flow(
        problem,
        momentum,
        pressure_projection,
        algebraic_les=immersed_les.algebraic_les,
    )
    prepared = immersed_les.prepare(momentum, molecular_viscosity=problem.viscosity)
    compilation_id = canonical_fingerprint(
        {
            "kind": "compiled-fixed-immersed-mac-les-flow",
            "base": base.compilation_id,
            "immersed_les": prepared.prepared_id,
            "geometry": prepared.geometry_id,
            "markers": prepared.marker_id,
            "filter": prepared.filter_id,
            "model": prepared.model_id,
            "boundaries": prepared.boundary_id,
            "solver": prepared.solver_id,
        }
    )
    return CompiledMACIncompressibleDynamics(
        problem,
        momentum,
        pressure_projection,
        prepared,
        None,
        compilation_id=compilation_id,
    )


__all__ = [
    "FixedImmersedMACLESPlan",
    "FixedImmersedMarkerMotion",
    "ImmersedLESBalanceLedger",
    "ImmersedLESMotion",
    "ImmersedMACLESStageResult",
    "PreparedFixedImmersedMACLES",
    "compile_fixed_immersed_mac_les_flow",
]
