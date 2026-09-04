#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
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
from ..discretization.spectral import (
    IncompressibleSpectralDiagnostics,
    PeriodicLerayProjector,
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    TensorSpectralDiscretization,
)
from ..linalg import ArraySpace, DiagonalLinearOperator
from ._dynamic_les import LagrangianDynamicLESState
from ._ir import PDEField
from ._periodic_dynamic_les import (
    PeriodicDynamicLESPlan,
    PeriodicDynamicLESStage,
    PreparedPeriodicDynamicLES,
)
from ._periodic_les import (
    PeriodicAlgebraicLESPlan,
    PeriodicAlgebraicLESStage,
    PeriodicIncompressibleRateComponents,
    PeriodicIncompressibleStage,
    PeriodicLESStepRestriction,
    PreparedPeriodicAlgebraicLES,
)
from ._spectral_compile import SpectralStateLayout


class IncompressibleFlowProblem(StrictModule):
    """Newtonian incompressible velocity dynamics with compiler-space forcing."""

    viscosity: Array
    forcing: Any
    spatial_dimension: int = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_dimension: int,
        viscosity: ArrayLike,
        /,
        *,
        forcing: Any = None,
        forcing_id: str | None = None,
        problem_id: str | None = None,
    ):
        if isinstance(spatial_dimension, bool):
            raise TypeError("spatial_dimension must be an integer.")
        dimension = index(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Incompressible flow requires spatial dimension two or three."
            )
        raw_viscosity = jnp.asarray(viscosity)
        if jnp.iscomplexobj(raw_viscosity):
            raise TypeError("viscosity must be real.")
        viscosity_ = raw_viscosity.astype(float)
        if viscosity_.shape != () or not bool(
            jnp.isfinite(viscosity_) & (viscosity_ >= 0.0)
        ):
            raise ValueError("viscosity must be one finite nonnegative scalar.")
        if forcing is not None and not callable(forcing):
            raise TypeError("forcing must be callable or None.")
        if forcing is None:
            source_id = "none"
            if forcing_id is not None:
                raise ValueError("forcing_id must be omitted when forcing is None.")
        else:
            source_id = "" if forcing_id is None else str(forcing_id)
            if not source_id:
                raise ValueError("A forcing callable requires a non-empty forcing_id.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "incompressible-flow-problem-v1",
                    "dimension": dimension,
                    "viscosity": float(viscosity_),
                    "forcing": source_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.viscosity = viscosity_
        self.forcing = forcing
        self.spatial_dimension = dimension
        self.forcing_id = source_id
        self.problem_id = identifier


class _PeriodicRotationalDrift(StrictModule):
    problem: IncompressibleFlowProblem
    discretization: TensorSpectralDiscretization
    method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector
    algebraic_les: PreparedPeriodicAlgebraicLES | None
    dynamic_les: PreparedPeriodicDynamicLES | None
    nonlinear_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        discretization: TensorSpectralDiscretization,
        method: PreparedPseudospectralMethod,
        projector: PeriodicLerayProjector,
        algebraic_les: PreparedPeriodicAlgebraicLES | None,
        dynamic_les: PreparedPeriodicDynamicLES | None,
        /,
    ):
        self.problem = problem
        self.discretization = discretization
        self.method = method
        self.projector = projector
        self.algebraic_les = algebraic_les
        self.dynamic_les = dynamic_les
        if algebraic_les is None and dynamic_les is None:
            payload = {
                "kind": "periodic-incompressible-rotational-drift-v1",
                "dimension": problem.spatial_dimension,
                "forcing": problem.forcing_id,
                "discretization": discretization.prepared_id,
                "spatial_method": method.prepared_id,
                "projector": projector.projector_id,
            }
        else:
            payload = {
                "kind": "periodic-incompressible-les-drift",
                "dimension": problem.spatial_dimension,
                "forcing": problem.forcing_id,
                "discretization": discretization.prepared_id,
                "spatial_method": method.prepared_id,
                "projector": projector.projector_id,
                "algebraic_les": (
                    None if algebraic_les is None else algebraic_les.prepared_id
                ),
                "dynamic_les": (None if dynamic_les is None else dynamic_les.prepared_id),
            }
        self.nonlinear_id = canonical_fingerprint(payload)

    def _rotational_product(self, state: Array, /) -> Array:
        dealiasing = self.method.dealiasing
        evaluation = dealiasing.evaluation
        padded = dealiasing.embed(self.projector.zero_forbidden_modes(state))
        velocity = evaluation.reconstruct(padded)
        derivatives = tuple(
            evaluation.modal_derivative(padded, axis=axis)
            for axis in range(self.problem.spatial_dimension)
        )
        if self.problem.spatial_dimension == 2:
            vorticity_modal = derivatives[0][..., 1] - derivatives[1][..., 0]
            vorticity = evaluation.reconstruct(vorticity_modal)
            product = jnp.stack(
                (-vorticity * velocity[..., 1], vorticity * velocity[..., 0]),
                axis=-1,
            )
        else:
            vorticity_modal = jnp.stack(
                (
                    derivatives[1][..., 2] - derivatives[2][..., 1],
                    derivatives[2][..., 0] - derivatives[0][..., 2],
                    derivatives[0][..., 1] - derivatives[1][..., 0],
                ),
                axis=-1,
            )
            vorticity = evaluation.reconstruct(vorticity_modal)
            product = jnp.cross(vorticity, velocity, axis=-1)
        return dealiasing.project(product)

    def _advective_unprojected_rhs(self, state: Array, /) -> Array:
        return -self._rotational_product(state)

    def _forcing_unprojected_rhs(self, time: Array, state: Array, args: Any, /) -> Array:
        if self.problem.forcing is None:
            return jnp.zeros_like(state)
        return self.projector.validate_state(
            self.problem.forcing(time, state, args), owner="Modal forcing"
        )

    def nonlinear_rhs(self, state: ArrayLike, /) -> Array:
        """Return only the resolved advective contribution."""
        value = self.projector.validate_state(state)
        return self.projector.project(self._advective_unprojected_rhs(value))

    def forcing_rhs(self, time: Array, state: ArrayLike, args: Any, /) -> Array:
        value = self.projector.validate_state(state)
        return self.projector.project(self._forcing_unprojected_rhs(time, value, args))

    def algebraic_les_stage(
        self, state: ArrayLike, /
    ) -> PeriodicAlgebraicLESStage | None:
        value = self.projector.validate_state(state)
        if self.algebraic_les is None:
            return None
        return self.algebraic_les.evaluate(value)

    def dynamic_les_stage(
        self,
        state: ArrayLike,
        continuation_state: LagrangianDynamicLESState | None = None,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicDynamicLESStage | None:
        value = self.projector.validate_state(state)
        if self.dynamic_les is None:
            return None
        return self.dynamic_les.evaluate(
            value,
            continuation_state,
            accepted_update_mask=accepted_update_mask,
        )

    def stage(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicIncompressibleStage:
        value = self.projector.validate_state(state)
        advective_unprojected = self._advective_unprojected_rhs(value)
        advective = self.projector.project(advective_unprojected)
        algebraic_stage = self.algebraic_les_stage(value)
        dynamic_stage = self.dynamic_les_stage(
            value,
            continuation_state,
            accepted_update_mask=accepted_update_mask,
        )
        algebraic_unprojected = (
            jnp.zeros_like(value)
            if algebraic_stage is None
            else algebraic_stage.unprojected_rate
        )
        dynamic_unprojected = (
            jnp.zeros_like(value)
            if dynamic_stage is None
            else dynamic_stage.algebraic_stage.unprojected_rate
        )
        algebraic_rate = (
            jnp.zeros_like(value)
            if algebraic_stage is None
            else algebraic_stage.projected_rate
        )
        dynamic_rate = (
            jnp.zeros_like(value)
            if dynamic_stage is None
            else dynamic_stage.projected_rate
        )
        forcing_unprojected = self._forcing_unprojected_rhs(time, value, args)
        forcing = self.projector.project(forcing_unprojected)
        pressure_driving = self.projector.zero_forbidden_modes(
            advective_unprojected
            + algebraic_unprojected
            + dynamic_unprojected
            + forcing_unprojected
        )
        nonlinear = self.projector.project(pressure_driving)
        molecular = (
            -self.problem.viscosity.astype(self.projector.wavenumber_squared.dtype)
            * self.projector.wavenumber_squared[..., None]
            * value
        )
        rates = PeriodicIncompressibleRateComponents(
            advective_rate=advective,
            molecular_rate=molecular,
            algebraic_les_rate=algebraic_rate,
            dynamic_les_rate=dynamic_rate,
            forcing_rate=forcing,
            nonlinear_rate=nonlinear,
            total_rate=molecular + nonlinear,
        )
        return PeriodicIncompressibleStage(
            rates=rates,
            pressure_driving_unprojected_rate=pressure_driving,
            algebraic_les=algebraic_stage,
            dynamic_les=dynamic_stage,
        )

    def unconstrained_rhs(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> Array:
        if self.dynamic_les is not None:
            return self.stage(
                time,
                state,
                args,
                continuation_state=continuation_state,
                accepted_update_mask=accepted_update_mask,
            ).pressure_driving_unprojected_rate
        value = self.projector.validate_state(state)
        result = self._advective_unprojected_rhs(value)
        if self.algebraic_les is not None:
            result = result + self.algebraic_les.evaluate(value).unprojected_rate
        if self.problem.forcing is not None:
            result = result + self._forcing_unprojected_rhs(time, value, args)
        return self.projector.zero_forbidden_modes(result)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.projector.project(self.unconstrained_rhs(time, state, args))


class CompiledIncompressibleSpectralDynamics(StrictModule):
    """Velocity-only periodic incompressible dynamics in full complex coefficients."""

    problem: IncompressibleFlowProblem
    discretization: TensorSpectralDiscretization
    spatial_method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector
    layout: SpectralStateLayout
    nonlinear_drift: _PeriodicRotationalDrift
    algebraic_les: PreparedPeriodicAlgebraicLES | None
    dynamic_les: PreparedPeriodicDynamicLES | None
    semilinear_drift: Any
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        discretization: TensorSpectralDiscretization,
        spatial_method: PreparedPseudospectralMethod,
        projector: PeriodicLerayProjector,
        layout: SpectralStateLayout,
        nonlinear_drift: _PeriodicRotationalDrift,
        semilinear_drift: Any,
        /,
        *,
        compilation_id: str,
    ):
        residual_key = DiscretizationKey(
            "periodic_incompressible_form",
            DiscretizationRole.RESIDUAL,
            domain_labels=discretization.key.domain_labels,
        )
        records = [
            DiscretizationRecord(
                discretization.key,
                type(discretization).__name__,
                discretization.prepared_id,
                numeric_version=discretization.numeric_version,
                precision_evidence_id=discretization.precision_evidence_id,
                resource_evidence_id=discretization.resource_evidence_id,
            )
        ]
        residual_dependencies = [discretization.key.key_id]
        if nonlinear_drift.algebraic_les is not None:
            les_key = DiscretizationKey(
                "periodic_algebraic_les",
                DiscretizationRole.AUXILIARY,
                domain_labels=discretization.key.domain_labels,
            )
            les_kind = "prepared-periodic-algebraic-les"
            les_action = nonlinear_drift.algebraic_les
        elif nonlinear_drift.dynamic_les is not None:
            les_key = DiscretizationKey(
                "periodic_dynamic_les",
                DiscretizationRole.AUXILIARY,
                domain_labels=discretization.key.domain_labels,
            )
            les_kind = "prepared-periodic-dynamic-les"
            les_action = nonlinear_drift.dynamic_les
        else:
            les_key = None
            les_kind = ""
            les_action = None
        if les_action is not None:
            records.append(
                DiscretizationRecord(
                    les_key,
                    les_kind,
                    les_action.prepared_id,
                    dependency_key_ids=(discretization.key.key_id,),
                    precision_evidence_id=discretization.precision_evidence_id,
                    resource_evidence_id=(
                        les_action.closure_method.dealiasing.report.report_id
                    ),
                )
            )
            residual_dependencies.append(les_key.key_id)
        records.append(
            DiscretizationRecord(
                residual_key,
                "compiled-periodic-incompressible-form",
                compilation_id,
                dependency_key_ids=tuple(residual_dependencies),
            )
        )
        bundle = DiscretizationBundle(tuple(records))
        self.problem = problem
        self.discretization = discretization
        self.spatial_method = spatial_method
        self.projector = projector
        self.layout = layout
        self.nonlinear_drift = nonlinear_drift
        self.algebraic_les = nonlinear_drift.algebraic_les
        self.dynamic_les = nonlinear_drift.dynamic_les
        self.semilinear_drift = semilinear_drift
        self.discretization_bundle = bundle
        self.compilation_id = str(compilation_id)
        self.source_hash = problem.problem_id
        if self.algebraic_les is not None:
            self.resolved_method = (
                "periodic-incompressible-rotational-algebraic-les-diagonal"
            )
        elif self.dynamic_les is not None:
            self.resolved_method = (
                "periodic-incompressible-rotational-dynamic-les-diagonal"
            )
        else:
            self.resolved_method = "periodic-incompressible-rotational-diagonal"

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.projector.state_shape

    def project_state(self, values: ArrayLike, /) -> Array:
        physical = jnp.asarray(values)
        expected = self.discretization.physical_shape + (self.problem.spatial_dimension,)
        if physical.shape != expected:
            raise ValueError(
                f"Physical velocity must have shape {expected}; got {physical.shape}."
            )
        return self.projector.project(self.discretization.project(physical))

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        return self.discretization.reconstruct(self.projector.project(state))

    def physical_state(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> Array:
        del time, args
        return self.reconstruct_state(state)

    def stage(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicIncompressibleStage:
        """Evaluate every named equation rate and the selected LES stress once."""
        return self.nonlinear_drift.stage(
            jnp.asarray(time),
            state,
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
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
    ) -> PeriodicIncompressibleRateComponents:
        return self.stage(
            time,
            state,
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        ).rates

    def algebraic_les_stage(
        self, state: ArrayLike, /
    ) -> PeriodicAlgebraicLESStage | None:
        return self.nonlinear_drift.algebraic_les_stage(state)

    def dynamic_les_stage(
        self,
        state: ArrayLike,
        continuation_state: LagrangianDynamicLESState | None = None,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicDynamicLESStage | None:
        return self.nonlinear_drift.dynamic_les_stage(
            state,
            continuation_state,
            accepted_update_mask=accepted_update_mask,
        )

    def step_restriction(
        self,
        state: ArrayLike,
        /,
        *,
        algebraic_les_stage: PeriodicAlgebraicLESStage | None = None,
        dynamic_les_stage: PeriodicDynamicLESStage | None = None,
    ) -> PeriodicLESStepRestriction:
        """Return current-state ETDRK and fully explicit LES reference bounds."""
        if self.algebraic_les is not None:
            if dynamic_les_stage is not None:
                raise ValueError("Static LES cannot consume a dynamic LES stage.")
            return self.algebraic_les.step_restriction(
                state,
                self.problem.viscosity,
                stage=algebraic_les_stage,
            )
        if self.dynamic_les is not None:
            if algebraic_les_stage is not None:
                raise ValueError("Dynamic LES cannot consume a static LES stage.")
            if dynamic_les_stage is None:
                raise ValueError(
                    "Dynamic LES step restriction requires an explicit dynamic stage."
                )
            return self.dynamic_les.step_restriction(
                state,
                self.problem.viscosity,
                dynamic_les_stage,
            )
        raise ValueError("Periodic LES step restriction requires compiled LES.")

    def pressure_coefficients(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> Array:
        raw = self.nonlinear_drift.unconstrained_rhs(
            jnp.asarray(time),
            self.projector.validate_state(state),
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        )
        return self.projector.pressure_from_unconstrained_rhs(raw)

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        continuation_state: LagrangianDynamicLESState | None = None,
        accepted_update_mask: ArrayLike = True,
    ) -> IncompressibleSpectralDiagnostics:
        time_ = jnp.asarray(time)
        value = self.projector.validate_state(state)
        admissible = self.projector.project(value)
        physical = self.discretization.reconstruct(admissible)
        weights = self.discretization.quadrature_weights
        speed_squared = jnp.sum(jnp.real(physical * jnp.conj(physical)), axis=-1)
        kinetic_energy = 0.5 * jnp.sum(weights * speed_squared)
        gradient_squared = jnp.zeros(
            self.discretization.physical_shape, dtype=physical.real.dtype
        )
        for axis in range(self.problem.spatial_dimension):
            derivative = self.discretization.reconstruct(
                self.discretization.modal_derivative(admissible, axis=axis)
            )
            gradient_squared = gradient_squared + jnp.sum(
                jnp.real(derivative * jnp.conj(derivative)), axis=-1
            )
        molecular_dissipation = self.problem.viscosity * jnp.sum(
            weights * gradient_squared
        )
        equation_stage = self.stage(
            time_,
            admissible,
            args,
            continuation_state=continuation_state,
            accepted_update_mask=accepted_update_mask,
        )
        rates = equation_stage.rates

        def energy_rate(rate: Array, /) -> Array:
            physical_rate = self.discretization.reconstruct(rate)
            density = jnp.sum(jnp.real(jnp.conj(physical) * physical_rate), axis=-1)
            return jnp.sum(weights * density)

        advective_energy_rate = energy_rate(rates.advective_rate)
        forcing_power = energy_rate(rates.forcing_rate)
        molecular_energy_rate = energy_rate(rates.molecular_rate)
        algebraic_energy_rate = energy_rate(rates.algebraic_les_rate)
        dynamic_energy_rate = energy_rate(rates.dynamic_les_rate)
        semidiscrete_energy_rate = energy_rate(rates.total_rate)
        zero = jnp.zeros_like(kinetic_energy)
        algebraic_stage = equation_stage.algebraic_les
        if algebraic_stage is None:
            algebraic_dissipation = zero
            algebraic_identity = zero
            algebraic_projection_defect = zero
            algebraic_maximum_viscosity = zero
            algebraic_finite = jnp.asarray(True)
        else:
            algebraic_dissipation = algebraic_stage.modeled_dissipation
            algebraic_identity = algebraic_energy_rate + algebraic_dissipation
            algebraic_projection_defect = algebraic_stage.projection_energy_defect
            algebraic_maximum_viscosity = algebraic_stage.maximum_kinematic_viscosity
            algebraic_finite = (
                algebraic_stage.finite
                & algebraic_stage.dissipative
                & algebraic_stage.energy_consistent
            )
        dynamic_stage = equation_stage.dynamic_les
        if dynamic_stage is None:
            dynamic_dissipation = zero
            dynamic_identity = zero
            dynamic_projection_defect = zero
            dynamic_maximum_viscosity = zero
            coefficient_minimum = zero
            coefficient_mean = zero
            coefficient_maximum = zero
            regularization_count = jnp.asarray(0, dtype=jnp.int32)
            backscatter_count = jnp.asarray(0, dtype=jnp.int32)
            backscatter_limit_count = jnp.asarray(0, dtype=jnp.int32)
            accepted_count = jnp.asarray(0, dtype=jnp.int32)
            rejected_count = jnp.asarray(0, dtype=jnp.int32)
            dynamic_evidence_finite = jnp.asarray(True)
        else:
            dynamic_algebraic = dynamic_stage.algebraic_stage
            dynamic_dissipation = dynamic_algebraic.modeled_dissipation
            dynamic_identity = dynamic_energy_rate + dynamic_dissipation
            dynamic_projection_defect = dynamic_algebraic.projection_energy_defect
            dynamic_maximum_viscosity = dynamic_algebraic.maximum_kinematic_viscosity
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
            dynamic_evidence_finite = evidence.finite & dynamic_algebraic.finite
        projection_defect = algebraic_projection_defect + dynamic_projection_defect
        maximum_eddy_viscosity = jnp.maximum(
            algebraic_maximum_viscosity, dynamic_maximum_viscosity
        )
        energy_balance_defect = semidiscrete_energy_rate - (
            forcing_power
            - molecular_dissipation
            - algebraic_dissipation
            - dynamic_dissipation
        )
        forbidden = value - self.projector.zero_forbidden_modes(value)
        pressure = self.projector.pressure_from_unconstrained_rhs(
            equation_stage.pressure_driving_unprojected_rate
        )
        zero_mode = self.projector.wavenumber_squared == 0.0
        pressure_gauge = jnp.max(jnp.where(zero_mode, jnp.abs(pressure), 0.0))
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(rates.total_rate))
            & jnp.isfinite(kinetic_energy)
            & jnp.isfinite(advective_energy_rate)
            & jnp.isfinite(forcing_power)
            & jnp.isfinite(molecular_energy_rate)
            & jnp.isfinite(molecular_dissipation)
            & jnp.isfinite(algebraic_energy_rate)
            & jnp.isfinite(algebraic_dissipation)
            & jnp.isfinite(algebraic_identity)
            & jnp.isfinite(dynamic_energy_rate)
            & jnp.isfinite(dynamic_dissipation)
            & jnp.isfinite(dynamic_identity)
            & jnp.isfinite(projection_defect)
            & jnp.isfinite(maximum_eddy_viscosity)
            & jnp.isfinite(semidiscrete_energy_rate)
            & jnp.isfinite(energy_balance_defect)
            & algebraic_finite
            & dynamic_evidence_finite
        )
        return IncompressibleSpectralDiagnostics(
            kinetic_energy=kinetic_energy,
            advective_energy_rate=advective_energy_rate,
            forcing_power=forcing_power,
            molecular_energy_rate=molecular_energy_rate,
            molecular_dissipation=molecular_dissipation,
            algebraic_les_energy_rate=algebraic_energy_rate,
            algebraic_les_dissipation=algebraic_dissipation,
            algebraic_les_energy_identity_defect=algebraic_identity,
            projection_energy_defect=projection_defect,
            maximum_eddy_viscosity=maximum_eddy_viscosity,
            algebraic_les_available=algebraic_stage is not None,
            dynamic_les_energy_rate=dynamic_energy_rate,
            dynamic_les_dissipation=dynamic_dissipation,
            dynamic_les_energy_identity_defect=dynamic_identity,
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
            semidiscrete_energy_rate=semidiscrete_energy_rate,
            energy_balance_defect=energy_balance_defect,
            divergence_norm=self.projector.divergence_norm(value),
            imaginary_leakage=self.discretization.imaginary_leakage(admissible),
            forbidden_mode_norm=GeometryPrecisionPolicy().norm(forbidden.reshape((-1,))),
            pressure_gauge_residual=pressure_gauge,
            finite=finite,
            projector_id=self.projector.projector_id,
            dynamic_les_id=(
                None if self.dynamic_les is None else self.dynamic_les.prepared_id
            ),
        )

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        return self.semilinear_drift(time, state, args)


def compile_periodic_incompressible_flow(
    problem: IncompressibleFlowProblem,
    discretization: TensorSpectralDiscretization,
    method: PseudospectralMethodPlan,
    /,
    *,
    algebraic_les: PeriodicAlgebraicLESPlan | None = None,
    dynamic_les: PeriodicDynamicLESPlan | None = None,
    dynamic_test_discretization: TensorSpectralDiscretization | None = None,
) -> CompiledIncompressibleSpectralDynamics:
    """Compile rotational incompressible flow with no, static, or dynamic LES."""
    if not isinstance(problem, IncompressibleFlowProblem):
        raise TypeError("problem must be an IncompressibleFlowProblem.")
    if not isinstance(discretization, TensorSpectralDiscretization):
        raise TypeError("discretization must be a TensorSpectralDiscretization.")
    if len(discretization.axes) != problem.spatial_dimension:
        raise ValueError("Problem dimension must match the tensor spectral rank.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    if algebraic_les is not None and not isinstance(
        algebraic_les, PeriodicAlgebraicLESPlan
    ):
        raise TypeError("algebraic_les must be a PeriodicAlgebraicLESPlan or None.")
    if dynamic_les is not None and not isinstance(dynamic_les, PeriodicDynamicLESPlan):
        raise TypeError("dynamic_les must be a PeriodicDynamicLESPlan or None.")
    if algebraic_les is not None and dynamic_les is not None:
        raise ValueError("Static algebraic LES and dynamic LES are alternatives.")
    if dynamic_les is None and dynamic_test_discretization is not None:
        raise ValueError("dynamic_test_discretization is valid only with dynamic_les.")
    if dynamic_les is not None and not isinstance(
        dynamic_test_discretization, TensorSpectralDiscretization
    ):
        raise TypeError("dynamic_test_discretization must be supplied for dynamic LES.")
    prepared_method = method.prepare(
        discretization,
        required_polynomial_degree=2,
        nonlinear=True,
    )
    projector = PeriodicLerayProjector(discretization)
    prepared_les = (
        None
        if algebraic_les is None
        else algebraic_les.prepare(discretization, projector)
    )
    prepared_dynamic = (
        None
        if dynamic_les is None
        else dynamic_les.prepare(
            discretization,
            dynamic_test_discretization,
            projector,
        )
    )
    field = PDEField(
        "velocity",
        representation="vector",
        components=problem.spatial_dimension,
        coordinates=discretization.plan.axis_names,
        component_names=discretization.plan.axis_names,
    )
    layout = SpectralStateLayout((field,), discretization)
    if layout.state_shape != projector.state_shape:
        raise RuntimeError("Incompressible state and spectral field layouts disagree.")
    nonlinear = _PeriodicRotationalDrift(
        problem,
        discretization,
        prepared_method,
        projector,
        prepared_les,
        prepared_dynamic,
    )
    state_space = ArraySpace(
        projector.state_shape,
        dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
    )
    diagonal = jnp.broadcast_to(
        (
            -problem.viscosity.astype(projector.wavenumber_squared.dtype)
            * projector.wavenumber_squared
        )[..., None],
        projector.state_shape,
    ).reshape((-1,))
    linear = DiagonalLinearOperator(
        diagonal,
        space=state_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "periodic-incompressible-viscosity",
                "problem": problem.problem_id,
                "projector": projector.projector_id,
            }
        ),
    )
    from ..solver._semilinear_drift import SemilinearDrift

    semilinear = SemilinearDrift(
        linear,
        nonlinear,
        state_shape=projector.state_shape,
        operator_id=linear.operator_id,
        nonlinear_id=nonlinear.nonlinear_id,
    )
    if prepared_les is None and prepared_dynamic is None:
        compilation_payload = {
            "kind": "periodic-incompressible-compiler-v1",
            "problem": problem.problem_id,
            "nonlinear": nonlinear.nonlinear_id,
            "discretization": discretization.prepared_id,
            "method": prepared_method.prepared_id,
            "projector": projector.projector_id,
        }
    else:
        compilation_payload = {
            "kind": "periodic-incompressible-les-compiler",
            "problem": problem.problem_id,
            "nonlinear": nonlinear.nonlinear_id,
            "discretization": discretization.prepared_id,
            "method": prepared_method.prepared_id,
            "projector": projector.projector_id,
            "algebraic_les": (None if prepared_les is None else prepared_les.prepared_id),
            "dynamic_les": (
                None if prepared_dynamic is None else prepared_dynamic.prepared_id
            ),
        }
    compilation_id = canonical_fingerprint(compilation_payload)
    return CompiledIncompressibleSpectralDynamics(
        problem,
        discretization,
        prepared_method,
        projector,
        layout,
        nonlinear,
        semilinear,
        compilation_id=compilation_id,
    )


__all__ = [
    "CompiledIncompressibleSpectralDynamics",
    "IncompressibleFlowProblem",
    "compile_periodic_incompressible_flow",
]
