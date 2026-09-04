#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    PreparedMACMomentumOperators,
    PreparedMACVariationalViscosityAction,
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
from ._mac_les import (
    _axis_values,
    _cell_centered_component,
    _MAC_LES_REGIME,
    _mac_velocity_gradient,
    _realize_mac_les_stage,
    MACLESStageResult,
)


_BINOMIAL_TEST_KERNEL = (0.25, 0.5, 0.25)
_TEST_FILTER_RATIO = (2.0, 2.0, 2.0)


class MACExplicitTestFilterPlan(StrictModule, NonTrainableState):
    """Declare the fixed normalized periodic three-point MAC test filter."""

    test_filter: ResolvedLESFilter
    kernel_weights: tuple[float, float, float] = eqx.field(static=True)
    test_filter_ratio: tuple[float, float, float] = eqx.field(static=True)
    commutation_status: str = eqx.field(static=True)
    boundary_support: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, test_filter: ResolvedLESFilter, /):
        if not isinstance(test_filter, ResolvedLESFilter):
            raise TypeError("test_filter must be a ResolvedLESFilter.")
        if (
            test_filter.family != "explicit-filter"
            or test_filter.topology != "tensor-product"
            or test_filter.boundary_class != "periodic"
            or test_filter.scale_rule != "kernel-equivalent"
            or test_filter.commutation_status != "commuting"
            or test_filter.repeated_filter_semantics != "composed"
        ):
            raise ValueError(
                "The MAC test filter requires explicit-filter, periodic "
                "tensor-product, kernel-equivalent, commuting, composed semantics."
            )
        self.test_filter = test_filter
        self.kernel_weights = _BINOMIAL_TEST_KERNEL
        self.test_filter_ratio = _TEST_FILTER_RATIO
        self.commutation_status = "commuting"
        self.boundary_support = "periodic-wrap-only"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-explicit-test-filter-plan",
                "test_filter": test_filter.filter_id,
                "kernel_weights": list(_BINOMIAL_TEST_KERNEL),
                "test_filter_ratio": list(_TEST_FILTER_RATIO),
                "commutation_status": self.commutation_status,
                "boundary_support": self.boundary_support,
            }
        )

    def prepare(
        self, momentum: PreparedMACMomentumOperators, /
    ) -> PreparedMACExplicitTestFilter:
        """Prepare only on the exact periodic uniform three-dimensional subset."""
        return PreparedMACExplicitTestFilter(self, momentum)


class PreparedMACExplicitTestFilter(StrictModule, NonTrainableState):
    """Separable periodic binomial test filter with explicit support evidence."""

    plan: MACExplicitTestFilterPlan
    momentum: PreparedMACMomentumOperators
    resolved_axis_widths: tuple[Array, Array, Array]
    kernel_weights: tuple[float, float, float] = eqx.field(static=True)
    test_filter_ratio: tuple[float, float, float] = eqx.field(static=True)
    commutation_status: str = eqx.field(static=True)
    boundary_support: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACExplicitTestFilterPlan,
        momentum: PreparedMACMomentumOperators,
        /,
    ):
        if not isinstance(plan, MACExplicitTestFilterPlan):
            raise TypeError("plan must be MACExplicitTestFilterPlan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        if momentum.dimension != 3:
            raise ValueError("MAC dynamic LES requires a three-dimensional grid.")
        grid = momentum.operators.discretization.grid
        for axis in grid.structured_axes:
            widths = np.asarray(axis.interval_widths, dtype=float)
            if not axis.periodic or axis.primary_entity != "interval":
                raise ValueError(
                    "MAC dynamic LES is prepared only for periodic uniform cell axes."
                )
            if widths.size < 3 or not np.allclose(
                widths,
                widths[0],
                rtol=1e-12,
                atol=np.finfo(float).eps * max(1.0, abs(float(widths[0]))),
            ):
                raise ValueError(
                    "MAC dynamic LES is prepared only for periodic uniform grids "
                    "with at least three cells per axis."
                )
        if momentum.boundaries.sides:
            raise ValueError(
                "MAC dynamic LES has periodic-wrap support only; physical boundary "
                "stages are unsupported."
            )
        widths = tuple(axis.interval_widths for axis in grid.structured_axes)
        self.plan = plan
        self.momentum = momentum
        self.resolved_axis_widths = (widths[0], widths[1], widths[2])
        self.kernel_weights = plan.kernel_weights
        self.test_filter_ratio = plan.test_filter_ratio
        self.commutation_status = plan.commutation_status
        self.boundary_support = plan.boundary_support
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-explicit-test-filter",
                "plan": plan.plan_id,
                "momentum": momentum.prepared_id,
                "discretization": momentum.operators.discretization.prepared_id,
                "cell_shape": list(momentum.operators.discretization.cell_shape),
                "kernel_weights": list(plan.kernel_weights),
                "test_filter_ratio": list(plan.test_filter_ratio),
                "commutation_status": self.commutation_status,
                "boundary_support": self.boundary_support,
            }
        )

    def resolved_filter_scale(self, /) -> LESFilterScale:
        """Construct local directional implicit grid-volume widths."""
        return self._filter_scale((1.0, 1.0, 1.0))

    def test_filter_scale(self, /) -> LESFilterScale:
        """Construct local directional kernel-equivalent test widths."""
        return self._filter_scale(self.test_filter_ratio)

    def apply(self, cell_field: ArrayLike, /) -> Array:
        """Apply the fixed normalized separable kernel with periodic wrapping."""
        value = jnp.asarray(cell_field)
        cell_shape = self.momentum.operators.discretization.cell_shape
        if value.ndim < 3 or value.shape[:3] != cell_shape:
            raise ValueError(
                "MAC test-filter fields must begin with the prepared cell shape."
            )
        lower, center, upper = self.kernel_weights
        result = value
        for axis in range(3):
            result = (
                lower * jnp.roll(result, 1, axis=axis)
                + center * result
                + upper * jnp.roll(result, -1, axis=axis)
            )
        return result

    def _filter_scale(self, ratios: tuple[float, float, float], /) -> LESFilterScale:
        cell_shape = self.momentum.operators.discretization.cell_shape
        directional = jnp.stack(
            tuple(
                jnp.broadcast_to(
                    ratio * _axis_values(width, 3, axis),
                    cell_shape,
                )
                for axis, (width, ratio) in enumerate(
                    zip(self.resolved_axis_widths, ratios, strict=True)
                )
            ),
            axis=-1,
        )
        return LESFilterScale(directional)


class MACDynamicLESPlan(StrictModule, NonTrainableState):
    """Bind a dynamic Smagorinsky plan to the periodic-uniform MAC test filter."""

    dynamic_model: PreparedDynamicSmagorinskyPlan
    test_filter: MACExplicitTestFilterPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_model: PreparedDynamicSmagorinskyPlan,
        test_filter: MACExplicitTestFilterPlan,
        /,
    ):
        if not isinstance(dynamic_model, PreparedDynamicSmagorinskyPlan):
            raise TypeError("dynamic_model must be PreparedDynamicSmagorinskyPlan.")
        if not isinstance(test_filter, MACExplicitTestFilterPlan):
            raise TypeError("test_filter must be MACExplicitTestFilterPlan.")
        provenance = dynamic_model.provenance
        resolved = provenance.resolved_filter
        if (
            resolved.family != "implicit-grid-volume"
            or resolved.topology != "tensor-product"
            or resolved.boundary_class != "periodic"
            or resolved.scale_rule != "volume-equivalent"
            or resolved.commutation_status != "commuting"
            or resolved.repeated_filter_semantics != "unmodeled"
        ):
            raise ValueError(
                "Periodic-uniform MAC dynamic LES requires an implicit-grid-volume "
                "resolved filter with periodic tensor-product, volume-equivalent, "
                "commuting, unmodeled-repeat semantics."
            )
        if provenance.test_filter.filter_id != test_filter.test_filter.filter_id:
            raise ValueError("Dynamic and MAC test-filter identities disagree.")
        if not np.allclose(
            np.asarray(provenance.test_filter_ratio),
            np.asarray(test_filter.test_filter_ratio),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "MAC binomial test filtering requires directional width ratio 2."
            )
        self.dynamic_model = dynamic_model
        self.test_filter = test_filter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-dynamic-les-plan",
                "dynamic_model": dynamic_model.prepared_id,
                "test_filter": test_filter.plan_id,
                "runtime_scope": "3d-periodic-uniform-grid",
            }
        )

    def prepare(self, momentum: PreparedMACMomentumOperators, /) -> PreparedMACDynamicLES:
        """Prepare the exact-route periodic-uniform MAC dynamic adapter."""
        return PreparedMACDynamicLES(self, momentum)


class MACDynamicLESStage(StrictModule):
    """Cell Germano tensors, dynamic evidence, and MAC variational action."""

    leonard_tensor: Array
    modeled_tensor: Array
    test_filtered_velocity_gradient: Array
    accepted_update_mask: Array
    dynamic_result: DynamicLESResult
    mac_stage: MACLESStageResult
    commutation_status: str = eqx.field(static=True)
    boundary_support: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def continuation_state(self) -> LagrangianDynamicLESState | None:
        """Return the explicit continuation state produced by this evaluation."""
        return self.dynamic_result.continuation_state

    @property
    def model_result(self) -> AlgebraicLESResult:
        """Return the dynamic cell stress and energy transfer."""
        return self.mac_stage.model_result

    @property
    def physical_rate(self) -> FaceVelocity:
        """Return the unprojected variational MAC SGS rate."""
        return self.mac_stage.physical_rate

    @property
    def integrated_work(self) -> Array:
        """Return the variational kinetic-energy rate from the SGS action."""
        return self.mac_stage.integrated_work


class PreparedMACDynamicLES(StrictModule, NonTrainableState):
    """State-explicit dynamic LES runtime on periodic uniform 3-D MAC grids."""

    plan: MACDynamicLESPlan
    dynamic_model: PreparedDynamicSmagorinskyPlan
    momentum: PreparedMACMomentumOperators
    test_filter: PreparedMACExplicitTestFilter
    viscosity_action: PreparedMACVariationalViscosityAction
    continuation_required: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MACDynamicLESPlan,
        momentum: PreparedMACMomentumOperators,
        /,
    ):
        if not isinstance(plan, MACDynamicLESPlan):
            raise TypeError("plan must be MACDynamicLESPlan.")
        if not isinstance(momentum, PreparedMACMomentumOperators):
            raise TypeError("momentum must be PreparedMACMomentumOperators.")
        discretization = momentum.operators.discretization
        provenance = plan.dynamic_model.provenance
        if (
            provenance.parameter_provenance.discretization_id
            != discretization.prepared_id
        ):
            raise ValueError(
                "Dynamic LES parameter provenance must name the MAC discretization."
            )
        if provenance.parameter_provenance.regime != _MAC_LES_REGIME:
            raise ValueError(
                "MAC dynamic LES requires the 'incompressible-unit-density' regime."
            )
        if provenance.resolved_filter.axis_names != discretization.grid.axis_names:
            raise ValueError(
                "Dynamic resolved-filter axes must match the MAC grid axis order."
            )
        test_filter = plan.test_filter.prepare(momentum)
        action = PreparedMACVariationalViscosityAction(momentum)
        self.plan = plan
        self.dynamic_model = plan.dynamic_model
        self.momentum = momentum
        self.test_filter = test_filter
        self.viscosity_action = action
        self.continuation_required = isinstance(
            plan.dynamic_model.averaging, LagrangianDynamicLESAveraging
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-dynamic-les",
                "plan": plan.plan_id,
                "dynamic_model": plan.dynamic_model.prepared_id,
                "test_filter": test_filter.prepared_id,
                "momentum": momentum.prepared_id,
                "viscosity_action": action.action_id,
                "continuation_required": self.continuation_required,
                "runtime_scope": "3d-periodic-uniform-grid",
            }
        )

    def initial_state(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> LagrangianDynamicLESState:
        """Create explicit zero history for this prepared MAC cell field."""
        self.momentum.boundaries.validate_stage(boundary_stage)
        values = self.momentum.operators.validate_velocity(velocity)
        dtype = jnp.empty((), dtype=values[0].dtype).real.dtype
        zero = jnp.zeros(
            self.momentum.operators.discretization.cell_shape + (3, 3),
            dtype=dtype,
        )
        scale = self.test_filter.resolved_filter_scale()
        inputs = DynamicLESInputs(
            zero,
            zero,
            AlgebraicLESInputs(zero, scale),
            self.dynamic_model.provenance,
            accepted_update_mask=accepted_update_mask,
        )
        return self.dynamic_model.initial_state(inputs)

    def evaluate(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        continuation_state: LagrangianDynamicLESState | None = None,
        /,
        *,
        accepted_update_mask: ArrayLike = True,
    ) -> MACDynamicLESStage:
        """Evaluate cell Germano tensors, stress, MAC rate, and explicit next state."""
        inputs, values, stage, test_gradient, leonard = self._germano_inputs(
            velocity,
            boundary_stage,
            accepted_update_mask=accepted_update_mask,
        )
        dynamic_result = self.dynamic_model.evaluate(inputs, continuation_state)
        mac_stage = _realize_mac_les_stage(
            self.viscosity_action,
            values,
            stage,
            inputs.algebraic_inputs.velocity_gradient,
            inputs.algebraic_inputs.filter_scale,
            dynamic_result.prepared_algebraic_stress,
            prepared_id=self.prepared_id,
        )
        return MACDynamicLESStage(
            leonard_tensor=leonard,
            modeled_tensor=inputs.modeled_tensor,
            test_filtered_velocity_gradient=test_gradient,
            accepted_update_mask=inputs.accepted_update_mask,
            dynamic_result=dynamic_result,
            mac_stage=mac_stage,
            commutation_status=self.test_filter.commutation_status,
            boundary_support=self.test_filter.boundary_support,
            prepared_id=self.prepared_id,
        )

    def step_restriction(
        self,
        stage: MACDynamicLESStage,
        /,
    ) -> tuple[Array, bool]:
        """Return the explicit SGS bound from one already evaluated dynamic stage."""
        if not isinstance(stage, MACDynamicLESStage):
            raise TypeError("stage must be MACDynamicLESStage.")
        if stage.prepared_id != self.prepared_id:
            raise ValueError("Dynamic LES stage belongs to another prepared action.")
        return (
            self.viscosity_action.explicit_step_bound(
                stage.model_result.kinematic_viscosity
            ),
            self.viscosity_action.restriction_supported,
        )

    def _germano_inputs(
        self,
        velocity: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        accepted_update_mask: ArrayLike,
    ) -> tuple[
        DynamicLESInputs,
        FaceVelocity,
        MACBoundaryStageData,
        Array,
        Array,
    ]:
        stage = self.momentum.boundaries.validate_stage(boundary_stage)
        values = self.momentum.boundaries.enforce(
            self.momentum.operators.validate_velocity(velocity), stage
        )
        gradient = _mac_velocity_gradient(self.momentum, values)
        resolved_scale = self.test_filter.resolved_filter_scale()
        test_scale = self.test_filter.test_filter_scale()
        cell_velocity = jnp.stack(
            tuple(
                _cell_centered_component(value, axis, True)
                for axis, value in enumerate(values)
            ),
            axis=-1,
        )
        test_velocity = self.test_filter.apply(cell_velocity)
        filtered_product = self.test_filter.apply(
            cell_velocity[..., :, None] * cell_velocity[..., None, :]
        )
        leonard = filtered_product - (
            test_velocity[..., :, None] * test_velocity[..., None, :]
        )
        test_gradient = self.test_filter.apply(gradient)
        resolved_model_tensor = _coefficient_free_smagorinsky_tensor(
            gradient, resolved_scale
        )
        modeled = _coefficient_free_smagorinsky_tensor(
            test_gradient, test_scale
        ) - self.test_filter.apply(resolved_model_tensor)
        inputs = DynamicLESInputs(
            leonard,
            modeled,
            AlgebraicLESInputs(gradient, resolved_scale),
            self.dynamic_model.provenance,
            accepted_update_mask=accepted_update_mask,
        )
        return inputs, values, stage, test_gradient, leonard


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
    "MACDynamicLESPlan",
    "MACDynamicLESStage",
    "MACExplicitTestFilterPlan",
    "PreparedMACDynamicLES",
    "PreparedMACExplicitTestFilter",
]
