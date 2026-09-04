#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.finite_volume import (
    FaceVelocity,
    PreparedMACMomentumOperators,
)
from ..discretization.finite_volume._mac_boundary import MACBoundaryStageData
from ._dynamic_les import (
    LagrangianDynamicLESState,
    NonnegativeBackscatterClip,
)
from ._incompressible import IncompressibleFlowProblem
from ._mac_dynamic_les import (
    MACDynamicLESPlan,
    MACDynamicLESStage,
    PreparedMACDynamicLES,
)
from ._mac_les import (
    MACAlgebraicLESPlan,
    MACLESStageResult,
    PreparedMACAlgebraicLES,
)


if TYPE_CHECKING:
    from ..solver._structured_incompressible import (
        MACPressureProjectionPlan,
        MACRateProjectionResult,
    )


class MACLESStepRestriction(StrictModule):
    """Current-state explicit advective, molecular, and SGS step diagnostics."""

    advective: Array
    molecular: Array
    sgs: Array
    combined: Array
    sgs_supported: bool = eqx.field(static=True)


class MACIncompressibleRateComponents(StrictModule):
    """Typed pre-projection momentum-rate decomposition."""

    convection: FaceVelocity
    molecular: FaceVelocity
    sgs: FaceVelocity
    forcing: FaceVelocity
    unconstrained: FaceVelocity
    les_stage: MACLESStageResult | None
    dynamic_les_stage: MACDynamicLESStage | None


class MACIncompressibleDiagnostics(StrictModule):
    """Constraint and energy evidence for one bounded MAC velocity state."""

    kinetic_energy: Array
    nonlinear_energy_rate: Array
    forcing_power: Array
    viscous_energy_rate: Array
    sgs_energy_rate: Array
    sgs_dissipation: Array
    sgs_boundary_power: Array
    sgs_energy_transfer: Array
    dissipation: Array
    boundary_power: Array
    open_backflow_dissipation: Array
    integrated_mass_flux: Array
    semidiscrete_energy_rate: Array
    energy_balance_defect: Array
    divergence_norm: Array
    boundary_defect: Array
    pressure_residual_norm: Array
    pressure_gauge_residual: Array
    finite: Array
    successful: Array
    projection_converged: Array
    pressure_closure: str = eqx.field(static=True)
    pressure_gauge: str = eqx.field(static=True)
    momentum_id: str = eqx.field(static=True)
    dynamic_coefficient_minimum: Array
    dynamic_coefficient_mean: Array
    dynamic_coefficient_maximum: Array
    dynamic_regularization_activity_count: Array
    dynamic_backscatter_activity_count: Array
    dynamic_backscatter_limit_count: Array
    dynamic_accepted_update_count: Array
    dynamic_rejected_update_count: Array
    dynamic_les_available: Array
    dynamic_evidence_finite: Array
    dynamic_les_id: str | None = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)


class CompiledMACIncompressibleDynamics(StrictModule):
    """Velocity-only bounded MAC dynamics in canonical flat coordinates."""

    problem: IncompressibleFlowProblem
    momentum: PreparedMACMomentumOperators
    projection: MACPressureProjectionPlan
    algebraic_les: PreparedMACAlgebraicLES | None
    dynamic_les: PreparedMACDynamicLES | None
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        momentum: PreparedMACMomentumOperators,
        projection: MACPressureProjectionPlan,
        algebraic_les: PreparedMACAlgebraicLES | None,
        dynamic_les: PreparedMACDynamicLES | None,
        /,
        *,
        compilation_id: str,
    ):
        discretization = momentum.operators.discretization
        residual_key = DiscretizationKey(
            "mac_incompressible_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        records = [
            DiscretizationRecord(
                discretization.key,
                type(discretization).__name__,
                discretization.prepared_id,
                numeric_version=discretization.numeric_version,
            )
        ]
        dependencies = [discretization.key.key_id]
        if algebraic_les is not None or dynamic_les is not None:
            dynamic = dynamic_les is not None
            les_key = DiscretizationKey(
                "mac_dynamic_les" if dynamic else "mac_algebraic_les",
                DiscretizationRole.AUXILIARY,
                domain_labels=discretization.key.domain_labels,
            )
            les_action = dynamic_les if dynamic else algebraic_les
            records.append(
                DiscretizationRecord(
                    les_key,
                    (
                        "prepared-mac-dynamic-les"
                        if dynamic
                        else "prepared-mac-algebraic-les"
                    ),
                    les_action.prepared_id,
                    dependency_key_ids=(discretization.key.key_id,),
                )
            )
            dependencies.append(les_key.key_id)
        records.append(
            DiscretizationRecord(
                residual_key,
                "compiled-mac-incompressible-form",
                compilation_id,
                dependency_key_ids=tuple(dependencies),
            )
        )
        bundle = DiscretizationBundle(tuple(records))
        self.problem = problem
        self.momentum = momentum
        self.algebraic_les = algebraic_les
        self.dynamic_les = dynamic_les
        self.projection = projection
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = problem.problem_id
        if algebraic_les is not None:
            self.resolved_method = "mac-symmetry-preserving-les-projected"
        elif dynamic_les is not None:
            self.resolved_method = "mac-symmetry-preserving-dynamic-les-projected"
        else:
            self.resolved_method = "mac-symmetry-preserving-projected"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.momentum.operators.velocity_space.size,)

    def validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"MAC velocity coordinates must have shape {self.state_shape}; "
                f"got {value.shape}."
            )
        dtype = self.momentum.operators.pressure_space.dtype
        if value.dtype != dtype:
            raise TypeError(f"MAC velocity coordinates must have dtype {dtype}.")
        return value

    def boundary_stage(
        self, time: ArrayLike, args: Any = None, /
    ) -> MACBoundaryStageData:
        return self.momentum.boundaries.evaluate(time, args)

    def pack_velocity(
        self,
        velocity: FaceVelocity,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> Array:
        stage = self.boundary_stage(time, args)
        value = self.momentum.boundaries.enforce(velocity, stage)
        coordinates = self.momentum.operators.velocity_space.flatten(value)
        return eqx.error_if(
            coordinates,
            ~stage.successful,
            "MAC boundary provider evaluation failed.",
        )

    def unpack_velocity(self, state: ArrayLike, /) -> FaceVelocity:
        value = self.validate_state(state)
        return tuple(self.momentum.operators.velocity_space.unflatten(value))

    def project_state(
        self,
        velocity: FaceVelocity,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> Array:
        stage = self.boundary_stage(time, args)
        value = self.momentum.boundaries.enforce(velocity, stage)
        projected = self.projection.project(value, 1.0, boundary_stage=stage)
        coordinates = self.momentum.operators.velocity_space.flatten(projected.velocity)
        return eqx.error_if(
            coordinates,
            ~stage.successful | ~projected.converged,
            "Initial MAC pressure projection failed.",
        )

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> FaceVelocity:
        stage = self.boundary_stage(time, args)
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        coordinates = self.momentum.operators.velocity_space.flatten(velocity)
        checked = eqx.error_if(
            coordinates,
            ~stage.successful,
            "MAC boundary provider evaluation failed.",
        )
        return tuple(self.momentum.operators.velocity_space.unflatten(checked))

    def _forcing(
        self,
        time: Array,
        velocity: FaceVelocity,
        args: Any,
        /,
    ) -> FaceVelocity:
        if self.problem.forcing is None:
            return tuple(jnp.zeros_like(value) for value in velocity)
        forcing = self.momentum.operators.validate_velocity(
            self.problem.forcing(time, velocity, args)
        )
        return self.momentum.boundaries.homogeneous_rate(forcing)

    def _rate_components(
        self,
        time: Array,
        state: ArrayLike,
        args: Any,
        stage: MACBoundaryStageData,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> MACIncompressibleRateComponents:
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        convection = self.momentum.convection(velocity, stage=stage)
        diffusion = self.momentum.laplacian(velocity, stage=stage)
        forcing = self._forcing(time, velocity, args)
        viscosity = self.problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        molecular = tuple(viscosity * value for value in diffusion)
        if self.algebraic_les is not None:
            les_stage = self.algebraic_les.evaluate(velocity, stage)
            dynamic_stage = None
            sgs = les_stage.physical_rate
            les_successful = les_stage.successful
        elif self.dynamic_les is not None:
            les_stage = None
            dynamic_stage = self.dynamic_les.evaluate(
                velocity,
                stage,
                continuation_state,
                accepted_update_mask=accepted_update_mask,
            )
            sgs = dynamic_stage.physical_rate
            les_successful = (
                dynamic_stage.mac_stage.successful
                & dynamic_stage.dynamic_result.evidence.finite
            )
        else:
            les_stage = None
            dynamic_stage = None
            sgs = tuple(jnp.zeros_like(value) for value in velocity)
            les_successful = jnp.asarray(True)
        raw_unconstrained = tuple(
            -advective + molecular_rate + sgs_rate + source
            for advective, molecular_rate, sgs_rate, source in zip(
                convection, molecular, sgs, forcing, strict=True
            )
        )
        unconstrained = (
            raw_unconstrained
            if dynamic_stage is not None
            else tuple(
                eqx.error_if(
                    value,
                    ~les_successful,
                    "MAC algebraic LES stage evaluation failed.",
                )
                for value in raw_unconstrained
            )
        )
        return MACIncompressibleRateComponents(
            convection=convection,
            molecular=molecular,
            sgs=sgs,
            forcing=forcing,
            unconstrained=self.momentum.boundaries.enforce_rate(unconstrained, stage),
            les_stage=les_stage,
            dynamic_les_stage=dynamic_stage,
        )

    def rate_components(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> MACIncompressibleRateComponents:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        return self._rate_components(
            time_,
            state,
            args,
            stage,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        )

    def unconstrained_rate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> FaceVelocity:
        return self.rate_components(
            time,
            state,
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        ).unconstrained

    def rate_projection(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> MACRateProjectionResult:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        unconstrained = self._rate_components(
            time_,
            state,
            args,
            stage,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        ).unconstrained
        return self.projection.project_rate(unconstrained, boundary_stage=stage)

    def pressure_field(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> Array:
        projected = self.rate_projection(
            time,
            state,
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        )
        return eqx.error_if(
            projected.pressure,
            ~projected.converged,
            "MAC pressure recovery failed.",
        )

    def step_restriction(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        dynamic_les_stage: MACDynamicLESStage | None = None,
    ) -> MACLESStepRestriction:
        velocity = self.unpack_velocity(state)
        grid = self.momentum.operators.discretization.grid
        inverse_advective = jnp.zeros(
            self.momentum.operators.discretization.cell_shape,
            dtype=jnp.dtype(self.momentum.precision.reduction_dtype),
        )
        inverse_diffusive = jnp.zeros_like(inverse_advective)
        for axis_index, axis in enumerate(grid.structured_axes):
            component = velocity[axis_index]
            moved = jnp.moveaxis(component, axis_index, 0)
            cell_velocity = (
                0.5 * (moved + jnp.roll(moved, -1, axis=0))
                if axis.periodic
                else 0.5 * (moved[:-1] + moved[1:])
            )
            cell_velocity = jnp.moveaxis(cell_velocity, 0, axis_index)
            shape = [1] * inverse_advective.ndim
            shape[axis_index] = int(axis.interval_widths.size)
            widths = axis.interval_widths.reshape(tuple(shape))
            inverse_advective = inverse_advective + jnp.abs(cell_velocity) / widths
            inverse_diffusive = inverse_diffusive + 2.0 / widths**2
        advective_rate = jnp.max(inverse_advective)
        viscosity = self.problem.viscosity.astype(inverse_diffusive.dtype)
        molecular_rate = viscosity * jnp.max(inverse_diffusive)
        safe_advective = jnp.where(advective_rate > 0.0, advective_rate, 1.0)
        safe_molecular = jnp.where(molecular_rate > 0.0, molecular_rate, 1.0)
        advective = jnp.where(advective_rate > 0.0, 1.0 / safe_advective, jnp.inf)
        molecular = jnp.where(molecular_rate > 0.0, 1.0 / safe_molecular, jnp.inf)
        if self.algebraic_les is not None:
            if dynamic_les_stage is not None:
                raise ValueError("Static MAC LES cannot consume a dynamic stage.")
            boundary = self.boundary_stage(time, args)
            enforced = self.momentum.boundaries.enforce(velocity, boundary)
            sgs, sgs_supported = self.algebraic_les.step_restriction(enforced, boundary)
        elif self.dynamic_les is not None:
            if dynamic_les_stage is None:
                raise ValueError(
                    "Dynamic MAC step restriction requires an evaluated dynamic stage."
                )
            sgs, sgs_supported = self.dynamic_les.step_restriction(dynamic_les_stage)
        else:
            if dynamic_les_stage is not None:
                raise ValueError("No-LES MAC dynamics cannot consume a dynamic stage.")
            sgs = jnp.asarray(jnp.inf, dtype=inverse_diffusive.dtype)
            sgs_supported = True
        combined = jnp.minimum(jnp.minimum(advective, molecular), sgs)
        return MACLESStepRestriction(
            advective=self.momentum.precision.reduction(advective),
            molecular=self.momentum.precision.reduction(molecular),
            sgs=self.momentum.precision.reduction(sgs),
            combined=self.momentum.precision.reduction(combined),
            sgs_supported=sgs_supported,
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> MACIncompressibleDiagnostics:
        time_ = jnp.asarray(time)
        stage = self.boundary_stage(time_, args)
        velocity = self.momentum.boundaries.enforce(self.unpack_velocity(state), stage)
        components = self._rate_components(
            time_,
            state,
            args,
            stage,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        )
        projected = self.projection.project_rate(
            components.unconstrained, boundary_stage=stage
        )
        space = self.momentum.operators.velocity_space
        viscosity = self.problem.viscosity.astype(
            self.momentum.operators.pressure_space.dtype
        )
        nonlinear_rate = tuple(-value for value in components.convection)
        nonlinear_energy_rate = jnp.real(space.inner(velocity, nonlinear_rate))
        forcing_power = jnp.real(space.inner(velocity, components.forcing))
        viscous_energy_rate = jnp.real(space.inner(velocity, components.molecular))
        sgs_energy_rate = jnp.real(space.inner(velocity, components.sgs))
        semidiscrete_energy_rate = jnp.real(space.inner(velocity, projected.rate))
        momentum_diagnostics = self.momentum.diagnostics(velocity, stage=stage)
        molecular_dissipation = viscosity * momentum_diagnostics.dissipation
        traction_power = self.momentum._boundary_traction_power(velocity, stage)
        diffusive_boundary_power = momentum_diagnostics.boundary_power - traction_power
        molecular_boundary_power = viscosity * diffusive_boundary_power + traction_power
        dynamic_stage = components.dynamic_les_stage
        selected_stage = (
            components.les_stage
            if components.les_stage is not None
            else None
            if dynamic_stage is None
            else dynamic_stage.mac_stage
        )
        zero = jnp.asarray(0.0, dtype=semidiscrete_energy_rate.dtype)
        zero_count = jnp.asarray(0, dtype=jnp.int32)
        if selected_stage is None:
            sgs_dissipation = zero
            sgs_boundary_power = zero
            sgs_energy_transfer = zero
            les_finite = jnp.asarray(True)
            les_successful = jnp.asarray(True)
        else:
            sgs_dissipation = selected_stage.viscosity_result.integrated_dissipation
            sgs_boundary_power = selected_stage.boundary_power
            sgs_energy_transfer = jnp.sum(
                self.momentum.operators.discretization.cell_volumes
                * selected_stage.model_result.energy_transfer
            )
            les_finite = selected_stage.finite
            les_successful = selected_stage.successful
        if dynamic_stage is None:
            coefficient_minimum = zero
            coefficient_mean = zero
            coefficient_maximum = zero
            regularization_count = zero_count
            backscatter_count = zero_count
            backscatter_limit_count = zero_count
            accepted_count = zero_count
            rejected_count = zero_count
            dynamic_evidence_finite = jnp.asarray(True)
        else:
            coefficient = dynamic_stage.dynamic_result.coefficient
            coefficient_minimum = jnp.min(coefficient)
            coefficient_mean = jnp.mean(coefficient)
            coefficient_maximum = jnp.max(coefficient)
            evidence = dynamic_stage.dynamic_result.evidence
            regularization_count = evidence.regularization_activity_count
            backscatter_count = evidence.backscatter_activity_count
            backscatter_limit_count = evidence.backscatter_limit_count
            accepted_count = evidence.accepted_update_count
            rejected_count = evidence.rejected_update_count
            dynamic_evidence_finite = evidence.finite
            les_finite = les_finite & evidence.finite
        dissipation = molecular_dissipation + sgs_dissipation
        boundary_power = molecular_boundary_power + sgs_boundary_power
        backflow_dissipation = momentum_diagnostics.open_backflow_dissipation
        expected_rate = (
            forcing_power - dissipation + boundary_power - backflow_dissipation
        )
        volumes = self.momentum.operators.discretization.cell_volumes
        pressure_residual_norm = jnp.sqrt(
            jnp.sum(volumes * projected.pressure_residual**2)
        )
        divergence_norm = GeometryPrecisionPolicy().norm(
            projected.divergence_after.reshape((-1,))
        )
        finite = (
            momentum_diagnostics.finite
            & projected.finite
            & les_finite
            & jnp.isfinite(pressure_residual_norm)
            & jnp.isfinite(divergence_norm)
            & jnp.isfinite(semidiscrete_energy_rate)
        )
        successful = (
            finite
            & stage.successful
            & momentum_diagnostics.successful
            & les_successful
            & projected.converged
        )
        return MACIncompressibleDiagnostics(
            kinetic_energy=momentum_diagnostics.kinetic_energy,
            nonlinear_energy_rate=self.momentum.precision.reduction(
                nonlinear_energy_rate
            ),
            forcing_power=self.momentum.precision.reduction(forcing_power),
            viscous_energy_rate=self.momentum.precision.reduction(viscous_energy_rate),
            sgs_energy_rate=self.momentum.precision.reduction(sgs_energy_rate),
            sgs_dissipation=self.momentum.precision.reduction(sgs_dissipation),
            sgs_boundary_power=self.momentum.precision.reduction(sgs_boundary_power),
            sgs_energy_transfer=self.momentum.precision.reduction(sgs_energy_transfer),
            dissipation=self.momentum.precision.reduction(dissipation),
            boundary_power=self.momentum.precision.reduction(boundary_power),
            open_backflow_dissipation=self.momentum.precision.reduction(
                backflow_dissipation
            ),
            integrated_mass_flux=momentum_diagnostics.integrated_mass_flux,
            semidiscrete_energy_rate=self.momentum.precision.reduction(
                semidiscrete_energy_rate
            ),
            energy_balance_defect=self.momentum.precision.reduction(
                semidiscrete_energy_rate - expected_rate
            ),
            divergence_norm=self.momentum.precision.reduction(divergence_norm),
            boundary_defect=momentum_diagnostics.boundary_defect,
            pressure_residual_norm=self.momentum.precision.reduction(
                pressure_residual_norm
            ),
            pressure_gauge_residual=self.momentum.precision.reduction(
                projected.gauge_defect
            ),
            finite=finite,
            successful=successful,
            projection_converged=projected.converged,
            pressure_closure=projected.closure.kind,
            pressure_gauge=projected.closure.gauge,
            momentum_id=self.momentum.prepared_id,
            dynamic_coefficient_minimum=coefficient_minimum,
            dynamic_coefficient_mean=coefficient_mean,
            dynamic_coefficient_maximum=coefficient_maximum,
            dynamic_regularization_activity_count=regularization_count,
            dynamic_backscatter_activity_count=backscatter_count,
            dynamic_backscatter_limit_count=backscatter_limit_count,
            dynamic_accepted_update_count=accepted_count,
            dynamic_rejected_update_count=rejected_count,
            dynamic_les_available=dynamic_stage is not None,
            dynamic_evidence_finite=dynamic_evidence_finite,
            dynamic_les_id=(
                None if self.dynamic_les is None else self.dynamic_les.prepared_id
            ),
            projection_id=self.projection.plan_id,
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        projected = self.rate_projection(time, state, args)
        coordinates = self.momentum.operators.velocity_space.flatten(projected.rate)
        return eqx.error_if(
            coordinates,
            ~projected.converged,
            "MAC momentum-rate projection failed.",
        )


def compile_mac_incompressible_flow(
    problem: IncompressibleFlowProblem,
    momentum: PreparedMACMomentumOperators,
    projection: MACPressureProjectionPlan,
    /,
    *,
    algebraic_les: MACAlgebraicLESPlan | None = None,
    dynamic_les: MACDynamicLESPlan | None = None,
) -> CompiledMACIncompressibleDynamics:
    """Compile projected MAC flow with no, static, or periodic-uniform dynamic LES."""
    from ..solver._structured_incompressible import MACPressureProjectionPlan

    if not isinstance(problem, IncompressibleFlowProblem):
        raise TypeError("problem must be IncompressibleFlowProblem.")
    if not isinstance(momentum, PreparedMACMomentumOperators):
        raise TypeError("momentum must be PreparedMACMomentumOperators.")
    if not isinstance(projection, MACPressureProjectionPlan):
        raise TypeError("projection must be MACPressureProjectionPlan.")
    if problem.spatial_dimension != momentum.dimension:
        raise ValueError("Incompressible problem and MAC momentum dimensions differ.")
    if projection.operators.prepared_id != momentum.operators.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share operators.")
    if not np.isclose(projection.density, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("MAC compiled dynamics use unit density and reduced pressure.")
    if projection.boundaries.prepared_id != momentum.boundaries.prepared_id:
        raise ValueError("MAC momentum and pressure projection must share boundaries.")
    if algebraic_les is not None and not isinstance(algebraic_les, MACAlgebraicLESPlan):
        raise TypeError("algebraic_les must be MACAlgebraicLESPlan or None.")
    if dynamic_les is not None and not isinstance(dynamic_les, MACDynamicLESPlan):
        raise TypeError("dynamic_les must be MACDynamicLESPlan or None.")
    if algebraic_les is not None and dynamic_les is not None:
        raise ValueError("Static algebraic LES and dynamic LES are alternatives.")
    if dynamic_les is not None and not isinstance(
        dynamic_les.dynamic_model.backscatter, NonnegativeBackscatterClip
    ):
        raise ValueError(
            "Compiled MAC dynamic LES requires nonnegative backscatter clipping "
            "because the variational viscosity action admits only nonnegative "
            "cell viscosity."
        )
    prepared_les = None if algebraic_les is None else algebraic_les.prepare(momentum)
    prepared_dynamic = None if dynamic_les is None else dynamic_les.prepare(momentum)
    identity = {
        "kind": "compiled-mac-incompressible-flow",
        "problem": problem.problem_id,
        "momentum": momentum.prepared_id,
        "projection": projection.plan_id,
        "algebraic_les": None if prepared_les is None else prepared_les.prepared_id,
        "dynamic_les": (
            None if prepared_dynamic is None else prepared_dynamic.prepared_id
        ),
    }
    identifier = canonical_fingerprint(identity)
    return CompiledMACIncompressibleDynamics(
        problem,
        momentum,
        projection,
        prepared_les,
        prepared_dynamic,
        compilation_id=identifier,
    )


__all__ = [
    "CompiledMACIncompressibleDynamics",
    "MACIncompressibleDiagnostics",
    "MACIncompressibleRateComponents",
    "MACLESStepRestriction",
    "compile_mac_incompressible_flow",
]
