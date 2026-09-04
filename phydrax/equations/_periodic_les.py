#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral import (
    OversamplingDealiasingPlan,
    PeriodicLerayProjector,
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    TensorSpectralDiscretization,
)
from ._les_closures import (
    AlgebraicLESInputs,
    AlgebraicLESResult,
    LESFilterScale,
    PreparedAlgebraicLESModel,
    ResolvedLESFilter,
)


if TYPE_CHECKING:
    from ._periodic_dynamic_les import PeriodicDynamicLESStage


class PeriodicFourierGridFilterPlan(StrictModule, NonTrainableState):
    """Declare the sharp retained Fourier projection that defines a LES state."""

    resolved_filter: ResolvedLESFilter
    plan_id: str = eqx.field(static=True)

    def __init__(self, resolved_filter: ResolvedLESFilter, /):
        if not isinstance(resolved_filter, ResolvedLESFilter):
            raise TypeError("resolved_filter must be a ResolvedLESFilter.")
        if (
            resolved_filter.family != "sharp-fourier-projection"
            or resolved_filter.topology != "tensor-product"
            or resolved_filter.boundary_class != "periodic"
            or resolved_filter.scale_rule != "cutoff-equivalent"
            or resolved_filter.commutation_status != "commuting"
            or resolved_filter.repeated_filter_semantics != "idempotent"
        ):
            raise ValueError(
                "Periodic Fourier LES requires sharp Fourier projection with "
                "periodic tensor-product, cutoff-equivalent, commuting, "
                "idempotent semantics."
            )
        self.resolved_filter = resolved_filter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fourier-grid-filter-plan",
                "resolved_filter": resolved_filter.filter_id,
                "width_rule": "domain-length-over-retained-count",
                "live_modes": "retained-excluding-any-even-nyquist-plane",
            }
        )

    def prepare(
        self, discretization: TensorSpectralDiscretization, /
    ) -> PreparedPeriodicFourierGridFilter:
        """Bind exact L/N widths and the live retained-mode mask."""
        return PreparedPeriodicFourierGridFilter(self, discretization)


class PreparedPeriodicFourierGridFilter(StrictModule, NonTrainableState):
    """Sharp grid-filter binding for one three-dimensional Fourier space."""

    plan: PeriodicFourierGridFilterPlan
    discretization: TensorSpectralDiscretization
    filter_scale: LESFilterScale
    live_mask: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicFourierGridFilterPlan,
        discretization: TensorSpectralDiscretization,
        /,
    ):
        if not isinstance(plan, PeriodicFourierGridFilterPlan):
            raise TypeError("plan must be a PeriodicFourierGridFilterPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if len(discretization.axes) != 3 or any(
            axis.family != "fourier" or not axis.periodic for axis in discretization.axes
        ):
            raise ValueError(
                "Periodic Fourier grid filtering requires exactly three periodic "
                "Fourier axes."
            )
        if tuple(discretization.plan.axis_names) != plan.resolved_filter.axis_names:
            raise ValueError(
                "Resolved LES filter axis names must match the spectral discretization."
            )
        physical_dtype = jnp.dtype(discretization.plan.precision.physical_dtype)
        widths = jnp.stack(
            tuple(
                axis.length.astype(physical_dtype) / axis.physical_count
                for axis in discretization.axes
            )
        )
        masks = []
        for axis_index, axis in enumerate(discretization.axes):
            shape = [1, 1, 1]
            shape[axis_index] = axis.mode_count
            masks.append(
                jnp.broadcast_to(
                    (~axis.modes.nyquist_mask).reshape(tuple(shape)),
                    discretization.modal_shape,
                )
            )
        live = masks[0] & masks[1] & masks[2]
        self.plan = plan
        self.discretization = discretization
        self.filter_scale = LESFilterScale(widths)
        self.live_mask = live
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-fourier-grid-filter",
                "plan": plan.plan_id,
                "resolved_filter": plan.resolved_filter.filter_id,
                "discretization": discretization.prepared_id,
                "axis_names": list(discretization.plan.axis_names),
                "retained_counts": list(discretization.modal_shape),
                "domain_lengths": [float(axis.length) for axis in discretization.axes],
                "directional_widths": [float(value) for value in np.asarray(widths)],
                "live_modes": "retained-excluding-any-even-nyquist-plane",
            }
        )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Apply the exact resolved projection on arbitrary trailing components."""
        value = self.discretization._validate_leading(
            coefficients,
            self.discretization.modal_shape,
            "Periodic grid-filter coefficients",
        )
        trailing = (1,) * (value.ndim - 3)
        return value * self.live_mask.reshape(self.live_mask.shape + trailing)


class PeriodicAlgebraicLESPlan(StrictModule, NonTrainableState):
    """Bind one frozen algebraic model to periodic spectral stress realization."""

    prepared_model: PreparedAlgebraicLESModel
    grid_filter: PeriodicFourierGridFilterPlan
    closure_method: PseudospectralMethodPlan
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_model: PreparedAlgebraicLESModel,
        grid_filter: PeriodicFourierGridFilterPlan,
        closure_method: PseudospectralMethodPlan,
        /,
        *,
        energy_tolerance: float = 1e-10,
    ):
        if not isinstance(prepared_model, PreparedAlgebraicLESModel):
            raise TypeError("prepared_model must be a PreparedAlgebraicLESModel.")
        if not isinstance(grid_filter, PeriodicFourierGridFilterPlan):
            raise TypeError("grid_filter must be a PeriodicFourierGridFilterPlan.")
        if not isinstance(closure_method, PseudospectralMethodPlan):
            raise TypeError("closure_method must be a PseudospectralMethodPlan.")
        if not isinstance(closure_method.dealiasing, OversamplingDealiasingPlan):
            raise ValueError(
                "Periodic algebraic LES requires OversamplingDealiasingPlan for its "
                "nonpolynomial stress evaluation; resolved filtering is separate."
            )
        if closure_method.dealiasing.factor < 1.5:
            raise ValueError(
                "Periodic algebraic LES requires an oversampling factor of at least 1.5."
            )
        tolerance = float(energy_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and nonnegative.")
        if (
            prepared_model.provenance.resolved_filter.filter_id
            != grid_filter.resolved_filter.filter_id
        ):
            raise ValueError(
                "Prepared LES model provenance and periodic grid filter disagree."
            )
        self.prepared_model = prepared_model
        self.grid_filter = grid_filter
        self.closure_method = closure_method
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-algebraic-les-plan",
                "prepared_model": prepared_model.prepared_id,
                "grid_filter": grid_filter.plan_id,
                "closure_method": closure_method.method_id,
                "energy_tolerance": tolerance,
                "stress_sign": "rhs-minus-divergence-of-specific-sgs-stress",
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ) -> PreparedPeriodicAlgebraicLES:
        return PreparedPeriodicAlgebraicLES(self, discretization, projector)


class PeriodicAlgebraicLESStage(StrictModule):
    """One physical stress evaluation and its retained modal momentum action."""

    velocity_gradient: Array
    filter_scale: LESFilterScale
    model_result: AlgebraicLESResult
    modal_deviatoric_specific_stress: Array
    unprojected_rate: Array
    projected_rate: Array
    modeled_dissipation: Array
    unprojected_energy_rate: Array
    modal_energy_rate: Array
    energy_identity_defect: Array
    projection_energy_defect: Array
    maximum_kinematic_viscosity: Array
    divergence_norm: Array
    imaginary_leakage: Array
    finite: Array
    dissipative: Array
    energy_consistent: Array
    prepared_id: str = eqx.field(static=True)


class PeriodicLESStepRestriction(StrictModule):
    """Conservative state-dependent advective and diffusive step evidence."""

    advective: Array
    molecular_diffusive: Array
    algebraic_les_diffusive: Array
    combined_diffusive: Array
    etdrk_selected: Array
    fully_explicit_selected: Array
    maximum_kinematic_viscosity: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class PeriodicIncompressibleRateComponents(StrictModule):
    """Named projected rates for the periodic incompressible equation."""

    advective_rate: Array
    molecular_rate: Array
    algebraic_les_rate: Array
    dynamic_les_rate: Array
    forcing_rate: Array
    nonlinear_rate: Array
    total_rate: Array

    @property
    def sgs_rate(self) -> Array:
        """Return the mutually exclusive static or dynamic SGS contribution."""
        return self.algebraic_les_rate + self.dynamic_les_rate


class PeriodicIncompressibleStage(StrictModule):
    """Complete equation stage with pressure-driving and LES evidence."""

    rates: PeriodicIncompressibleRateComponents
    pressure_driving_unprojected_rate: Array
    algebraic_les: PeriodicAlgebraicLESStage | None
    dynamic_les: PeriodicDynamicLESStage | None


class PreparedPeriodicAlgebraicLES(StrictModule, NonTrainableState):
    """Prepared single-device three-dimensional periodic algebraic LES action."""

    plan: PeriodicAlgebraicLESPlan
    model: PreparedAlgebraicLESModel
    grid_filter: PreparedPeriodicFourierGridFilter
    closure_method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicAlgebraicLESPlan,
        discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ):
        if not isinstance(plan, PeriodicAlgebraicLESPlan):
            raise TypeError("plan must be a PeriodicAlgebraicLESPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        if projector.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("LES projector and spectral discretization disagree.")
        if projector.spatial_dimension != 3:
            raise ValueError("Periodic algebraic LES is implemented only in 3-D.")
        provenance = plan.prepared_model.provenance
        if provenance.discretization_id != discretization.prepared_id:
            raise ValueError(
                "Prepared LES model provenance must name the retained discretization."
            )
        grid_filter = plan.grid_filter.prepare(discretization)
        if (
            provenance.resolved_filter.filter_id
            != grid_filter.plan.resolved_filter.filter_id
        ):
            raise ValueError("Prepared LES model and grid-filter identities disagree.")
        closure = plan.closure_method.prepare(
            discretization,
            required_polynomial_degree=None,
            nonlinear=True,
        )
        self.plan = plan
        self.model = plan.prepared_model
        self.grid_filter = grid_filter
        self.closure_method = closure
        self.projector = projector
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-algebraic-les",
                "plan": plan.plan_id,
                "model": plan.prepared_model.prepared_id,
                "filter": grid_filter.prepared_id,
                "closure_aliasing": closure.prepared_id,
                "projector": projector.projector_id,
                "discretization": discretization.prepared_id,
                "runtime_scope": "single-device-3d-unit-density-full-complex-fourier",
            }
        )

    def evaluate(self, state: ArrayLike, /) -> PeriodicAlgebraicLESStage:
        """Evaluate gradient, constitutive stress, divergence, and work once."""
        retained = self.projector.validate_state(state)
        live = self.grid_filter.apply(retained)
        dealiasing = self.closure_method.dealiasing
        evaluation = dealiasing.evaluation
        embedded = dealiasing.embed(live)
        gradient_modal = jnp.stack(
            tuple(evaluation.modal_derivative(embedded, axis=axis) for axis in range(3)),
            axis=-1,
        )
        velocity_gradient = evaluation.reconstruct(gradient_modal)
        model_result = self.model.evaluate(
            AlgebraicLESInputs(velocity_gradient, self.grid_filter.filter_scale)
        )
        return _realize_periodic_les_stage(
            live,
            velocity_gradient,
            model_result,
            self.grid_filter,
            self.closure_method,
            self.projector,
            energy_tolerance=self.plan.energy_tolerance,
            prepared_id=self.prepared_id,
        )

    def step_restriction(
        self,
        state: ArrayLike,
        molecular_viscosity: ArrayLike,
        /,
        *,
        stage: PeriodicAlgebraicLESStage | None = None,
    ) -> PeriodicLESStepRestriction:
        """Return conservative explicit reference bounds for the current state."""
        value = self.projector.validate_state(state)
        stage_ = self.evaluate(value) if stage is None else stage
        if not isinstance(stage_, PeriodicAlgebraicLESStage):
            raise TypeError("stage must be a PeriodicAlgebraicLESStage or None.")
        if stage_.prepared_id != self.prepared_id:
            raise ValueError("LES stage was produced by a different prepared action.")
        live = self.grid_filter.apply(value)
        velocity = self.grid_filter.discretization.reconstruct(live)
        widths = self.grid_filter.filter_scale.directional_widths.astype(
            velocity.real.dtype
        )
        advective_frequency = jnp.max(
            jnp.sum(jnp.abs(velocity) / widths, axis=-1),
            initial=jnp.asarray(0.0, dtype=velocity.real.dtype),
        )
        infinity = jnp.asarray(jnp.inf, dtype=advective_frequency.dtype)
        advective = jnp.where(
            advective_frequency > 0.0,
            1.0
            / jnp.where(
                advective_frequency > 0.0,
                advective_frequency,
                jnp.ones_like(advective_frequency),
            ),
            infinity,
        )
        admissible_k2 = jnp.where(
            self.projector.admissibility_mask,
            self.projector.wavenumber_squared,
            jnp.zeros_like(self.projector.wavenumber_squared),
        )
        maximum_k2 = jnp.max(admissible_k2)
        maximum_viscosity = stage_.maximum_kinematic_viscosity.astype(
            advective_frequency.dtype
        )
        molecular = jnp.asarray(molecular_viscosity, dtype=advective_frequency.dtype)
        molecular = molecular.reshape(())
        positive_sgs = (maximum_viscosity > 0.0) & (maximum_k2 > 0.0)
        algebraic_diffusive = jnp.where(
            positive_sgs,
            1.0
            / jnp.where(
                positive_sgs,
                maximum_viscosity * maximum_k2,
                jnp.ones_like(maximum_viscosity),
            ),
            infinity,
        )
        positive_molecular = (molecular > 0.0) & (maximum_k2 > 0.0)
        molecular_diffusive = jnp.where(
            positive_molecular,
            2.0
            / jnp.where(
                positive_molecular,
                molecular * maximum_k2,
                jnp.ones_like(molecular),
            ),
            infinity,
        )
        combined_coefficient = molecular + 2.0 * maximum_viscosity
        positive_combined = (combined_coefficient > 0.0) & (maximum_k2 > 0.0)
        combined_diffusive = jnp.where(
            positive_combined,
            2.0
            / jnp.where(
                positive_combined,
                combined_coefficient * maximum_k2,
                jnp.ones_like(combined_coefficient),
            ),
            infinity,
        )
        finite = (
            stage_.finite
            & stage_.dissipative
            & stage_.energy_consistent
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(advective_frequency)
            & jnp.isfinite(maximum_viscosity)
            & jnp.isfinite(molecular)
            & (molecular >= 0.0)
        )
        return PeriodicLESStepRestriction(
            advective=advective,
            molecular_diffusive=molecular_diffusive,
            algebraic_les_diffusive=algebraic_diffusive,
            combined_diffusive=combined_diffusive,
            etdrk_selected=jnp.minimum(advective, algebraic_diffusive),
            fully_explicit_selected=jnp.minimum(advective, combined_diffusive),
            maximum_kinematic_viscosity=maximum_viscosity,
            finite=finite,
            prepared_id=self.prepared_id,
        )


def _periodic_fourier_stress_rate(
    modal_stress: Array,
    grid_filter: PreparedPeriodicFourierGridFilter,
    projector: PeriodicLerayProjector,
    /,
) -> tuple[Array, Array]:
    """Apply the retained conservative ``-div(tau)`` action and projection."""
    stress = grid_filter.discretization._validate_leading(
        modal_stress,
        grid_filter.discretization.modal_shape,
        "Periodic modal LES stress",
    )
    if stress.shape[3:] != (3, 3):
        raise ValueError("Periodic LES stress must have modal_shape + (3, 3).")
    rate_components = []
    for component in range(3):
        divergence = jnp.zeros(
            grid_filter.discretization.modal_shape,
            dtype=stress.dtype,
        )
        for axis in range(3):
            divergence = divergence + grid_filter.discretization.modal_derivative(
                stress[..., component, axis], axis=axis
            )
        rate_components.append(-divergence)
    unprojected = grid_filter.apply(jnp.stack(tuple(rate_components), axis=-1))
    return unprojected, projector.project(unprojected)


def _realize_periodic_les_stage(
    live: Array,
    velocity_gradient: Array,
    model_result: AlgebraicLESResult,
    grid_filter: PreparedPeriodicFourierGridFilter,
    closure_method: PreparedPseudospectralMethod,
    projector: PeriodicLerayProjector,
    /,
    *,
    energy_tolerance: float,
    prepared_id: str,
) -> PeriodicAlgebraicLESStage:
    """Realize one already evaluated LES stress on the retained Fourier space."""
    dealiasing = closure_method.dealiasing
    evaluation = dealiasing.evaluation
    modal_stress = grid_filter.apply(
        dealiasing.project(model_result.specific_deviatoric_stress)
    )
    unprojected_rate, projected_rate = _periodic_fourier_stress_rate(
        modal_stress, grid_filter, projector
    )
    weights = evaluation.quadrature_weights
    modeled_dissipation = jnp.sum(weights * model_result.energy_transfer)
    unprojected_energy_rate = jnp.real(jnp.vdot(live, unprojected_rate))
    modal_energy_rate = jnp.real(jnp.vdot(live, projected_rate))
    identity_defect = modal_energy_rate + modeled_dissipation
    projection_defect = modal_energy_rate - unprojected_energy_rate
    maximum_viscosity = jnp.max(
        model_result.kinematic_viscosity,
        initial=jnp.asarray(0.0, dtype=velocity_gradient.real.dtype),
    )
    finite = (
        jnp.all(jnp.isfinite(velocity_gradient))
        & jnp.all(jnp.isfinite(model_result.kinematic_viscosity))
        & jnp.all(jnp.isfinite(model_result.specific_deviatoric_stress))
        & jnp.all(jnp.isfinite(model_result.energy_transfer))
        & jnp.all(jnp.isfinite(modal_stress))
        & jnp.all(jnp.isfinite(unprojected_rate))
        & jnp.all(jnp.isfinite(projected_rate))
        & jnp.isfinite(modeled_dissipation)
        & jnp.isfinite(identity_defect)
        & jnp.isfinite(projection_defect)
    )
    tolerance = jnp.asarray(energy_tolerance, dtype=modeled_dissipation.dtype)
    energy_scale = jnp.maximum(
        jnp.asarray(1.0, dtype=modeled_dissipation.dtype),
        jnp.maximum(jnp.abs(modal_energy_rate), jnp.abs(modeled_dissipation)),
    )
    dissipative = (
        jnp.all(model_result.kinematic_viscosity >= 0.0)
        & jnp.all(model_result.energy_transfer >= -tolerance)
        & (modal_energy_rate <= tolerance * energy_scale)
    )
    energy_consistent = jnp.abs(identity_defect) <= tolerance * energy_scale
    return PeriodicAlgebraicLESStage(
        velocity_gradient=velocity_gradient,
        filter_scale=grid_filter.filter_scale,
        model_result=model_result,
        modal_deviatoric_specific_stress=modal_stress,
        unprojected_rate=unprojected_rate,
        projected_rate=projected_rate,
        modeled_dissipation=modeled_dissipation,
        unprojected_energy_rate=unprojected_energy_rate,
        modal_energy_rate=modal_energy_rate,
        energy_identity_defect=identity_defect,
        projection_energy_defect=projection_defect,
        maximum_kinematic_viscosity=maximum_viscosity,
        divergence_norm=projector.divergence_norm(projected_rate),
        imaginary_leakage=grid_filter.discretization.imaginary_leakage(projected_rate),
        finite=finite,
        dissipative=dissipative,
        energy_consistent=energy_consistent,
        prepared_id=prepared_id,
    )


__all__ = [
    "PeriodicAlgebraicLESPlan",
    "PeriodicAlgebraicLESStage",
    "PeriodicFourierGridFilterPlan",
    "PeriodicIncompressibleRateComponents",
    "PeriodicIncompressibleStage",
    "PeriodicLESStepRestriction",
    "PreparedPeriodicAlgebraicLES",
    "PreparedPeriodicFourierGridFilter",
]
