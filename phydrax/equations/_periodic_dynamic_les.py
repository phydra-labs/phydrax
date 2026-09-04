#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

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
    prepare_spectral_modal_transfer,
    PreparedPseudospectralMethod,
    PreparedSpectralModalTransfer,
    PseudospectralMethodPlan,
    TensorSpectralDiscretization,
)
from ._dynamic_les import (
    DynamicLESInputs,
    DynamicLESResult,
    LagrangianDynamicLESAveraging,
    LagrangianDynamicLESState,
    PreparedDynamicSmagorinskyPlan,
)
from ._les_closures import (
    AlgebraicLESInputs,
    AlgebraicLESResult,
    LESFilterScale,
    ResolvedLESFilter,
)
from ._periodic_les import (
    _realize_periodic_les_stage,
    PeriodicAlgebraicLESStage,
    PeriodicFourierGridFilterPlan,
    PeriodicLESStepRestriction,
    PreparedPeriodicFourierGridFilter,
)


class PeriodicFourierTestFilterPlan(StrictModule, NonTrainableState):
    """Declare an exact coarse retained-space Fourier test projection."""

    test_filter: ResolvedLESFilter
    grid_filter_plan: PeriodicFourierGridFilterPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, test_filter: ResolvedLESFilter, /):
        grid_filter = PeriodicFourierGridFilterPlan(test_filter)
        self.test_filter = test_filter
        self.grid_filter_plan = grid_filter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-fourier-test-filter-plan",
                "test_filter": test_filter.filter_id,
                "projection": "exact-coarse-retained-modal-mask",
                "nyquist": "exclude-any-coarse-even-nyquist-plane",
            }
        )

    def prepare(
        self,
        resolved_filter: PreparedPeriodicFourierGridFilter,
        test_discretization: TensorSpectralDiscretization,
        /,
    ) -> PreparedPeriodicFourierTestFilter:
        """Prepare exact resolved-to-test transfers and their equivalent mask."""
        return PreparedPeriodicFourierTestFilter(
            self, resolved_filter, test_discretization
        )


class PreparedPeriodicFourierTestFilter(StrictModule, NonTrainableState):
    """Exact sharp test filter represented as a retained-space modal mask."""

    plan: PeriodicFourierTestFilterPlan
    resolved_filter: PreparedPeriodicFourierGridFilter
    test_grid_filter: PreparedPeriodicFourierGridFilter
    restriction: PreparedSpectralModalTransfer
    embedding: PreparedSpectralModalTransfer
    retained_mask: Array
    filter_scale: LESFilterScale
    test_filter_ratio: tuple[float, float, float] = eqx.field(static=True)
    commutation_status: str = eqx.field(static=True)
    boundary_support: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicFourierTestFilterPlan,
        resolved_filter: PreparedPeriodicFourierGridFilter,
        test_discretization: TensorSpectralDiscretization,
        /,
    ):
        if not isinstance(plan, PeriodicFourierTestFilterPlan):
            raise TypeError("plan must be a PeriodicFourierTestFilterPlan.")
        if not isinstance(resolved_filter, PreparedPeriodicFourierGridFilter):
            raise TypeError(
                "resolved_filter must be a PreparedPeriodicFourierGridFilter."
            )
        if not isinstance(test_discretization, TensorSpectralDiscretization):
            raise TypeError("test_discretization must be a TensorSpectralDiscretization.")
        resolved = resolved_filter.discretization
        if len(resolved.axes) != 3 or len(test_discretization.axes) != 3:
            raise ValueError("Periodic dynamic LES requires exactly three Fourier axes.")
        if tuple(test_discretization.plan.axis_names) != tuple(resolved.plan.axis_names):
            raise ValueError("Resolved and test Fourier axis names must match.")
        for resolved_axis, test_axis in zip(
            resolved.axes, test_discretization.axes, strict=True
        ):
            if (
                resolved_axis.family != "fourier"
                or test_axis.family != "fourier"
                or not resolved_axis.periodic
                or not test_axis.periodic
            ):
                raise ValueError(
                    "Periodic dynamic LES test filtering requires periodic Fourier axes."
                )
            if not np.isclose(
                float(resolved_axis.length),
                float(test_axis.length),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "Resolved and test Fourier domains must have identical lengths."
                )
            if test_axis.physical_count >= resolved_axis.physical_count:
                raise ValueError(
                    "Every test-filter Fourier axis must be strictly coarser than "
                    "the resolved axis."
                )

        test_grid_filter = plan.grid_filter_plan.prepare(test_discretization)
        restriction = prepare_spectral_modal_transfer(resolved, test_discretization)
        embedding = prepare_spectral_modal_transfer(test_discretization, resolved)
        candidate = embedding(
            test_grid_filter.apply(
                restriction(
                    jnp.ones(
                        resolved.modal_shape,
                        dtype=jnp.dtype(resolved.plan.precision.coefficient_dtype),
                    )
                )
            )
        )
        retained_mask = (jnp.abs(candidate) > 0.0) & resolved_filter.live_mask
        resolved_widths = np.asarray(
            resolved_filter.filter_scale.directional_widths, dtype=float
        )
        test_widths = np.asarray(
            test_grid_filter.filter_scale.directional_widths, dtype=float
        )
        ratio = tuple(float(value) for value in test_widths / resolved_widths)

        self.plan = plan
        self.resolved_filter = resolved_filter
        self.test_grid_filter = test_grid_filter
        self.restriction = restriction
        self.embedding = embedding
        self.retained_mask = retained_mask
        self.filter_scale = test_grid_filter.filter_scale
        self.test_filter_ratio = ratio
        self.commutation_status = "commuting"
        self.boundary_support = "periodic"
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-fourier-test-filter",
                "plan": plan.plan_id,
                "resolved_filter": resolved_filter.prepared_id,
                "test_grid_filter": test_grid_filter.prepared_id,
                "restriction": restriction.prepared_id,
                "embedding": embedding.prepared_id,
                "resolved_shape": list(resolved.modal_shape),
                "test_shape": list(test_discretization.modal_shape),
                "test_filter_ratio": list(ratio),
                "commutation_status": self.commutation_status,
                "boundary_support": self.boundary_support,
            }
        )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Apply the prepared exact test projection to retained coefficients."""
        value = self.resolved_filter.discretization._validate_leading(
            coefficients,
            self.resolved_filter.discretization.modal_shape,
            "Periodic test-filter coefficients",
        )
        trailing = (1,) * (value.ndim - 3)
        return value * self.retained_mask.reshape(self.retained_mask.shape + trailing)

    def apply_physical(
        self,
        values: ArrayLike,
        closure_method: PreparedPseudospectralMethod,
        /,
    ) -> Array:
        """Project, test-filter, and reconstruct a field on the closure grid."""
        if not isinstance(closure_method, PreparedPseudospectralMethod):
            raise TypeError("closure_method must be a PreparedPseudospectralMethod.")
        dealiasing = closure_method.dealiasing
        if (
            dealiasing.retained.prepared_id
            != self.resolved_filter.discretization.prepared_id
        ):
            raise ValueError("Test filter and closure method retain different spaces.")
        filtered = self.apply(dealiasing.project(values))
        return dealiasing.evaluation.reconstruct(dealiasing.embed(filtered))


class PeriodicDynamicLESPlan(StrictModule, NonTrainableState):
    """Bind a dynamic model to exact periodic resolved and test projections."""

    dynamic_model: PreparedDynamicSmagorinskyPlan
    grid_filter: PeriodicFourierGridFilterPlan
    test_filter: PeriodicFourierTestFilterPlan
    closure_method: PseudospectralMethodPlan
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_model: PreparedDynamicSmagorinskyPlan,
        grid_filter: PeriodicFourierGridFilterPlan,
        test_filter: PeriodicFourierTestFilterPlan,
        closure_method: PseudospectralMethodPlan,
        /,
        *,
        energy_tolerance: float = 1e-10,
    ):
        if not isinstance(dynamic_model, PreparedDynamicSmagorinskyPlan):
            raise TypeError("dynamic_model must be PreparedDynamicSmagorinskyPlan.")
        if not isinstance(grid_filter, PeriodicFourierGridFilterPlan):
            raise TypeError("grid_filter must be PeriodicFourierGridFilterPlan.")
        if not isinstance(test_filter, PeriodicFourierTestFilterPlan):
            raise TypeError("test_filter must be PeriodicFourierTestFilterPlan.")
        if not isinstance(closure_method, PseudospectralMethodPlan):
            raise TypeError("closure_method must be PseudospectralMethodPlan.")
        if not isinstance(closure_method.dealiasing, OversamplingDealiasingPlan):
            raise ValueError(
                "Periodic dynamic LES requires oversampling for Germano products; "
                "the distinct resolved/test filters are retained-space projections."
            )
        if closure_method.dealiasing.factor < 1.5:
            raise ValueError(
                "Periodic dynamic LES requires an oversampling factor of at least 1.5."
            )
        tolerance = float(energy_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_tolerance must be finite and nonnegative.")
        provenance = dynamic_model.provenance
        if provenance.resolved_filter.filter_id != grid_filter.resolved_filter.filter_id:
            raise ValueError("Dynamic and periodic resolved-filter identities disagree.")
        if provenance.test_filter.filter_id != test_filter.test_filter.filter_id:
            raise ValueError("Dynamic and periodic test-filter identities disagree.")

        self.dynamic_model = dynamic_model
        self.grid_filter = grid_filter
        self.test_filter = test_filter
        self.closure_method = closure_method
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-dynamic-les-plan",
                "dynamic_model": dynamic_model.prepared_id,
                "grid_filter": grid_filter.plan_id,
                "test_filter": test_filter.plan_id,
                "closure_method": closure_method.method_id,
                "energy_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        test_discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ) -> PreparedPeriodicDynamicLES:
        """Prepare the exact-route periodic dynamic LES runtime adapter."""
        return PreparedPeriodicDynamicLES(
            self, discretization, test_discretization, projector
        )


class PeriodicDynamicLESStage(StrictModule):
    """Germano evidence, dynamic result, and realized periodic stress action."""

    leonard_tensor: Array
    modeled_tensor: Array
    test_filtered_velocity_gradient: Array
    accepted_update_mask: Array
    dynamic_result: DynamicLESResult
    algebraic_stage: PeriodicAlgebraicLESStage
    prepared_id: str = eqx.field(static=True)

    @property
    def continuation_state(self) -> LagrangianDynamicLESState | None:
        """Return the explicit continuation state produced by this evaluation."""
        return self.dynamic_result.continuation_state

    @property
    def model_result(self) -> AlgebraicLESResult:
        """Return the dynamic stress supplied to the periodic realization."""
        return self.algebraic_stage.model_result

    @property
    def projected_rate(self) -> Array:
        """Return the projected retained modal SGS rate."""
        return self.algebraic_stage.projected_rate

    @property
    def modeled_dissipation(self) -> Array:
        """Return the integrated physical modeled energy transfer."""
        return self.algebraic_stage.modeled_dissipation


class PreparedPeriodicDynamicLES(StrictModule, NonTrainableState):
    """Exact retained-space dynamic LES evaluation for periodic 3-D Fourier flow."""

    plan: PeriodicDynamicLESPlan
    dynamic_model: PreparedDynamicSmagorinskyPlan
    grid_filter: PreparedPeriodicFourierGridFilter
    test_filter: PreparedPeriodicFourierTestFilter
    closure_method: PreparedPseudospectralMethod
    projector: PeriodicLerayProjector
    continuation_required: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicDynamicLESPlan,
        discretization: TensorSpectralDiscretization,
        test_discretization: TensorSpectralDiscretization,
        projector: PeriodicLerayProjector,
        /,
    ):
        if not isinstance(plan, PeriodicDynamicLESPlan):
            raise TypeError("plan must be PeriodicDynamicLESPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be TensorSpectralDiscretization.")
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be PeriodicLerayProjector.")
        if projector.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Dynamic LES projector and discretization disagree.")
        if projector.spatial_dimension != 3:
            raise ValueError("Periodic dynamic LES is implemented only in 3-D.")
        provenance = plan.dynamic_model.provenance
        if (
            provenance.parameter_provenance.discretization_id
            != discretization.prepared_id
        ):
            raise ValueError(
                "Dynamic LES parameter provenance must name the retained discretization."
            )

        grid_filter = plan.grid_filter.prepare(discretization)
        test_filter = plan.test_filter.prepare(grid_filter, test_discretization)
        if not np.allclose(
            np.asarray(test_filter.test_filter_ratio),
            np.asarray(provenance.test_filter_ratio),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Prepared Fourier resolution ratio and dynamic test-filter ratio differ."
            )
        closure_method = plan.closure_method.prepare(
            discretization,
            required_polynomial_degree=None,
            nonlinear=True,
        )
        self.plan = plan
        self.dynamic_model = plan.dynamic_model
        self.grid_filter = grid_filter
        self.test_filter = test_filter
        self.closure_method = closure_method
        self.projector = projector
        self.continuation_required = isinstance(
            plan.dynamic_model.averaging, LagrangianDynamicLESAveraging
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-dynamic-les",
                "plan": plan.plan_id,
                "dynamic_model": plan.dynamic_model.prepared_id,
                "grid_filter": grid_filter.prepared_id,
                "test_filter": test_filter.prepared_id,
                "closure_method": closure_method.prepared_id,
                "projector": projector.projector_id,
                "discretization": discretization.prepared_id,
                "continuation_required": self.continuation_required,
                "runtime_scope": "single-device-3d-periodic-full-complex-fourier",
            }
        )

    def initial_state(
        self,
        velocity: ArrayLike,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> LagrangianDynamicLESState:
        """Create explicit zero history for this velocity field and adapter."""
        retained = self.projector.validate_state(velocity)
        physical_shape = self.closure_method.dealiasing.evaluation.physical_shape
        dtype = jnp.empty((), dtype=retained.dtype).real.dtype
        zero = jnp.zeros(physical_shape + (3, 3), dtype=dtype)
        inputs = DynamicLESInputs(
            zero,
            zero,
            AlgebraicLESInputs(zero, self.grid_filter.filter_scale),
            self.dynamic_model.provenance,
            accepted_update_mask=accepted_update_mask,
        )
        return self.dynamic_model.initial_state(inputs)

    def evaluate(
        self,
        velocity: ArrayLike,
        continuation_state: LagrangianDynamicLESState | None = None,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> PeriodicDynamicLESStage:
        """Evaluate Germano tensors, coefficient, stress, rate, and next state."""
        inputs, live, test_gradient, leonard = self._germano_inputs(
            velocity, accepted_update_mask=accepted_update_mask
        )
        dynamic_result = self.dynamic_model.evaluate(inputs, continuation_state)
        algebraic_stage = _realize_periodic_les_stage(
            live,
            inputs.algebraic_inputs.velocity_gradient,
            dynamic_result.prepared_algebraic_stress,
            self.grid_filter,
            self.closure_method,
            self.projector,
            energy_tolerance=self.plan.energy_tolerance,
            prepared_id=self.prepared_id,
        )
        return PeriodicDynamicLESStage(
            leonard_tensor=leonard,
            modeled_tensor=inputs.modeled_tensor,
            test_filtered_velocity_gradient=test_gradient,
            accepted_update_mask=inputs.accepted_update_mask,
            dynamic_result=dynamic_result,
            algebraic_stage=algebraic_stage,
            prepared_id=self.prepared_id,
        )

    def step_restriction(
        self,
        velocity: ArrayLike,
        molecular_viscosity: ArrayLike,
        stage: PeriodicDynamicLESStage,
        /,
    ) -> PeriodicLESStepRestriction:
        """Return conservative explicit bounds from one already evaluated stage."""
        value = self.projector.validate_state(velocity)
        if not isinstance(stage, PeriodicDynamicLESStage):
            raise TypeError("stage must be PeriodicDynamicLESStage.")
        if stage.prepared_id != self.prepared_id:
            raise ValueError("Dynamic LES stage belongs to another prepared action.")
        live = self.grid_filter.apply(value)
        physical_velocity = self.grid_filter.discretization.reconstruct(live)
        widths = self.grid_filter.filter_scale.directional_widths.astype(
            physical_velocity.real.dtype
        )
        advective_frequency = jnp.max(
            jnp.sum(jnp.abs(physical_velocity) / widths, axis=-1),
            initial=jnp.asarray(0.0, dtype=physical_velocity.real.dtype),
        )
        infinity = jnp.asarray(jnp.inf, dtype=advective_frequency.dtype)
        safe_advective = jnp.where(
            advective_frequency > 0.0,
            advective_frequency,
            jnp.ones_like(advective_frequency),
        )
        advective = jnp.where(advective_frequency > 0.0, 1.0 / safe_advective, infinity)
        maximum_k2 = jnp.max(
            jnp.where(
                self.projector.admissibility_mask,
                self.projector.wavenumber_squared,
                jnp.zeros_like(self.projector.wavenumber_squared),
            )
        )
        maximum_viscosity = stage.algebraic_stage.maximum_kinematic_viscosity.astype(
            advective_frequency.dtype
        )
        molecular = jnp.asarray(
            molecular_viscosity, dtype=advective_frequency.dtype
        ).reshape(())
        positive_sgs = (maximum_viscosity > 0.0) & (maximum_k2 > 0.0)
        sgs_frequency = maximum_viscosity * maximum_k2
        algebraic_diffusive = jnp.where(
            positive_sgs,
            1.0 / jnp.where(positive_sgs, sgs_frequency, jnp.ones_like(sgs_frequency)),
            infinity,
        )
        positive_molecular = (molecular > 0.0) & (maximum_k2 > 0.0)
        molecular_frequency = molecular * maximum_k2
        molecular_diffusive = jnp.where(
            positive_molecular,
            2.0
            / jnp.where(
                positive_molecular,
                molecular_frequency,
                jnp.ones_like(molecular_frequency),
            ),
            infinity,
        )
        combined_frequency = (molecular + 2.0 * maximum_viscosity) * maximum_k2
        positive_combined = combined_frequency > 0.0
        combined_diffusive = jnp.where(
            positive_combined,
            2.0
            / jnp.where(
                positive_combined,
                combined_frequency,
                jnp.ones_like(combined_frequency),
            ),
            infinity,
        )
        finite = (
            stage.algebraic_stage.finite
            & stage.dynamic_result.evidence.finite
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(physical_velocity))
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

    def _germano_inputs(
        self,
        velocity: ArrayLike,
        /,
        *,
        accepted_update_mask: ArrayLike,
    ) -> tuple[DynamicLESInputs, Array, Array, Array]:
        retained = self.projector.validate_state(velocity)
        live = self.grid_filter.apply(retained)
        dealiasing = self.closure_method.dealiasing
        evaluation = dealiasing.evaluation
        embedded = dealiasing.embed(live)
        physical_velocity = evaluation.reconstruct(embedded)
        resolved_gradient = jnp.stack(
            tuple(
                evaluation.reconstruct(evaluation.modal_derivative(embedded, axis=axis))
                for axis in range(3)
            ),
            axis=-1,
        )

        test_modal = self.test_filter.apply(live)
        embedded_test = dealiasing.embed(test_modal)
        test_velocity = evaluation.reconstruct(embedded_test)
        test_gradient = jnp.stack(
            tuple(
                evaluation.reconstruct(
                    evaluation.modal_derivative(embedded_test, axis=axis)
                )
                for axis in range(3)
            ),
            axis=-1,
        )
        resolved_product = (
            physical_velocity[..., :, None] * physical_velocity[..., None, :]
        )
        filtered_product = self.test_filter.apply_physical(
            resolved_product, self.closure_method
        )
        leonard = filtered_product - (
            test_velocity[..., :, None] * test_velocity[..., None, :]
        )

        resolved_model_tensor = _coefficient_free_smagorinsky_tensor(
            resolved_gradient, self.grid_filter.filter_scale
        )
        filtered_resolved_model = self.test_filter.apply_physical(
            resolved_model_tensor, self.closure_method
        )
        test_model_tensor = _coefficient_free_smagorinsky_tensor(
            test_gradient, self.test_filter.filter_scale
        )
        modeled = test_model_tensor - filtered_resolved_model
        inputs = DynamicLESInputs(
            leonard,
            modeled,
            AlgebraicLESInputs(resolved_gradient, self.grid_filter.filter_scale),
            self.dynamic_model.provenance,
            accepted_update_mask=accepted_update_mask,
        )
        return inputs, live, test_gradient, leonard


def _coefficient_free_smagorinsky_tensor(
    velocity_gradient: Array,
    filter_scale: LESFilterScale,
    /,
) -> Array:
    strain = 0.5 * (velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=velocity_gradient.dtype)
    deviatoric = strain - trace[..., None, None] * identity / 3.0
    squared_magnitude = 2.0 * jnp.sum(strain * strain, axis=(-2, -1))
    active = squared_magnitude > 0.0
    safe = jnp.where(active, squared_magnitude, jnp.ones_like(squared_magnitude))
    magnitude = jnp.where(active, jnp.sqrt(safe), jnp.zeros_like(safe))
    width = filter_scale.equivalent_width
    return -2.0 * width[..., None, None] ** 2 * magnitude[..., None, None] * deviatoric


__all__ = [
    "PeriodicDynamicLESPlan",
    "PeriodicDynamicLESStage",
    "PeriodicFourierTestFilterPlan",
    "PreparedPeriodicDynamicLES",
    "PreparedPeriodicFourierTestFilter",
]
