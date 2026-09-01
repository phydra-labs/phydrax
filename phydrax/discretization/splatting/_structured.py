#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint
from ..._interpolation import apply_gather_stencil, GatherStencil, InterpolationResult
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import DiscretizationCapability, PreparationReport, resolved_identifier
from .._measure import DiscreteMeasure
from .._tensor_entities import TensorEntityLayout
from .._tensor_support import GridLocation, PreparedTensorGrid
from .._transfer import TransferProperties
from ..particle._core import ParticleDiscretization
from ..particle._precision import ParticlePrecisionPolicy
from ._assignment import (
    AbstractStructuredSplatAssignment,
    MultilinearSplatAssignment,
    SplatAssignmentState,
)
from ._reduction import (
    _scatter_route_payload,
    cast_stage,
    certified_sum,
    deposit_routes,
)
from ._types import (
    ParticleGridSplatBudget,
    SplatBalanceEvidence,
    SplatBoundaryPolicy,
    SplatDepositResult,
    SplatExecutionPolicy,
    SplatReconstructionResult,
    SplatRouteScatterResult,
)


if TYPE_CHECKING:
    from ..._precision import PrecisionEvidenceEnvelope


class ParticleGridSplatState(StrictModule):
    """Fixed-shape assignment routes from one particle state to a structured grid."""

    stencil: GatherStencil
    assignment_state: SplatAssignmentState
    source_active_mask: Array
    supported_mask: Array
    truncated_support_mask: Array
    out_of_domain_mask: Array
    invalid_geometry_mask: Array
    partition_sums: Array
    minimum_route_weight: Array
    minimum_domain_margin: Array
    valid_route_count: Array
    dropped_source_count: Array
    invalid_geometry_count: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        stencil: GatherStencil,
        assignment_state: SplatAssignmentState,
        source_active_mask: ArrayLike,
        supported_mask: ArrayLike,
        truncated_support_mask: ArrayLike,
        out_of_domain_mask: ArrayLike,
        invalid_geometry_mask: ArrayLike,
        partition_sums: ArrayLike,
        minimum_route_weight: ArrayLike,
        minimum_domain_margin: ArrayLike,
        valid_route_count: ArrayLike,
        dropped_source_count: ArrayLike,
        invalid_geometry_count: ArrayLike,
        successful: ArrayLike,
        prepared_id: str,
    ):
        if not isinstance(stencil, GatherStencil):
            raise TypeError("stencil must be GatherStencil.")
        if not isinstance(assignment_state, SplatAssignmentState):
            raise TypeError("assignment_state must be SplatAssignmentState.")
        source_shape = stencil.support.shape
        supported = jnp.asarray(supported_mask, dtype=bool)
        truncated = jnp.asarray(truncated_support_mask, dtype=bool)
        outside = jnp.asarray(out_of_domain_mask, dtype=bool)
        invalid = jnp.asarray(invalid_geometry_mask, dtype=bool)
        partitions = jnp.asarray(partition_sums)
        if any(
            value.shape != source_shape
            for value in (supported, truncated, outside, invalid, partitions)
        ):
            raise ValueError(
                "Splat state source arrays must match the particle capacity."
            )
        source_active = jnp.asarray(source_active_mask, dtype=bool)
        if source_active.shape != supported.shape:
            raise ValueError("source_active_mask must match particle capacity.")
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be non-empty.")
        self.stencil = stencil
        self.assignment_state = assignment_state
        self.supported_mask = supported
        self.source_active_mask = source_active
        self.truncated_support_mask = truncated
        self.out_of_domain_mask = outside
        self.invalid_geometry_mask = invalid
        self.partition_sums = partitions
        self.minimum_route_weight = jnp.asarray(minimum_route_weight)
        self.minimum_domain_margin = jnp.asarray(minimum_domain_margin)
        self.valid_route_count = jnp.asarray(valid_route_count, dtype=jnp.int32)
        self.dropped_source_count = jnp.asarray(dropped_source_count, dtype=jnp.int32)
        self.invalid_geometry_count = jnp.asarray(invalid_geometry_count, dtype=jnp.int32)
        self.successful = jnp.asarray(successful, dtype=bool)
        self.prepared_id = identifier

    @property
    def captured_fractions(self) -> Array:
        return self.assignment_state.captured_fractions

    @property
    def weight_gradients(self) -> Array:
        return self.assignment_state.weight_gradients

    @property
    def route_offsets(self) -> Array:
        return self.assignment_state.route_offsets

    @property
    def first_moments(self) -> Array:
        return self.assignment_state.first_moments

    @property
    def second_moments(self) -> Array:
        return self.assignment_state.second_moments

    @property
    def gradient_sums(self) -> Array:
        return self.assignment_state.gradient_sums

    def require_success(self, value: ArrayLike, /) -> Array:
        """Return ``value`` or fail unless geometry and boundary checks passed."""
        failed = ~self.successful
        if not isinstance(failed, jax.core.Tracer):
            failed = bool(failed)
        return eqx.error_if(
            jnp.asarray(value),
            failed,
            "Particle-grid splat state failed geometry, support, or boundary checks.",
        )


class ParticleGridSplatPlan(StrictModule, NonTrainableState):
    """Particle transfer onto one structured tensor-grid layout."""

    target: PreparedTensorGrid
    location: GridLocation
    assignment: AbstractStructuredSplatAssignment
    boundary: SplatBoundaryPolicy = eqx.field(static=True)
    execution: SplatExecutionPolicy
    precision: ParticlePrecisionPolicy
    budget: ParticleGridSplatBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: PreparedTensorGrid,
        /,
        *,
        location: GridLocation | None = None,
        assignment: AbstractStructuredSplatAssignment | None = None,
        boundary: SplatBoundaryPolicy = "reject",
        execution: SplatExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        budget: ParticleGridSplatBudget | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(target, PreparedTensorGrid):
            raise TypeError("target must be PreparedTensorGrid.")
        selected_location = (
            target.location((0,) * len(target.axis_names))
            if location is None
            else location
        )
        target.layout_at(selected_location)
        assignment_ = MultilinearSplatAssignment() if assignment is None else assignment
        execution_ = SplatExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        budget_ = ParticleGridSplatBudget() if budget is None else budget
        if not isinstance(assignment_, AbstractStructuredSplatAssignment):
            raise TypeError("assignment must be AbstractStructuredSplatAssignment.")
        if boundary not in ("reject", "drop"):
            raise ValueError("boundary must be 'reject' or 'drop'.")
        if not isinstance(execution_, SplatExecutionPolicy):
            raise TypeError("execution must be SplatExecutionPolicy.")
        if not isinstance(precision_, ParticlePrecisionPolicy):
            raise TypeError("precision must be ParticlePrecisionPolicy.")
        if not isinstance(budget_, ParticleGridSplatBudget):
            raise TypeError("budget must be ParticleGridSplatBudget.")
        identifier = resolved_identifier(
            "plan_id",
            plan_id,
            {
                "kind": "particle-grid-splat-plan",
                "target": target.prepared_id,
                "location": selected_location.location_id,
                "assignment": assignment_.assignment_id,
                "boundary": boundary,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "budget": budget_.budget_id,
            },
        )
        self.target = target
        self.location = selected_location
        self.assignment = assignment_
        self.boundary = boundary
        self.execution = execution_
        self.precision = precision_
        self.budget = budget_
        self.plan_id = identifier

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedParticleGridSplat:
        """Bind this target plan to one stable material-particle support."""
        return PreparedParticleGridSplat(self, particles)


class PreparedParticleGridSplat(StrictModule, NonTrainableState):
    """Prepared particle-grid transfer with auditable resources and assignment."""

    plan: ParticleGridSplatPlan
    particles: ParticleDiscretization
    layout: TensorEntityLayout
    target_measure: DiscreteMeasure
    stable_source_order: Array
    axis_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    route_width: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    properties: TransferProperties
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ParticleGridSplatPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, ParticleGridSplatPlan):
            raise TypeError("plan must be ParticleGridSplatPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if particles.ambient_dimension != len(plan.target.axis_names):
            raise ValueError("Particle and target-grid dimensions must match.")
        layout = plan.target.layout_at(plan.location)
        axes = plan.target.structured_axes
        plan.assignment.validate(layout, axes)
        target_measure = plan.target.measure_for(layout)
        target_weights = np.asarray(target_measure.weights)
        if np.any(~np.isfinite(target_weights)) or np.any(target_weights <= 0.0):
            raise ValueError("Splat target measure must be finite and strictly positive.")
        source_count = particles.capacity
        target_size = prod(layout.shape)
        if target_size > np.iinfo(np.int32).max:
            raise ValueError("Splat target is too large for int32 route indices.")
        dimension = particles.ambient_dimension
        width = plan.assignment.route_width(dimension)
        route_count = source_count * width
        index_itemsize = np.dtype(np.int32).itemsize
        evaluation_itemsize = np.dtype(plan.precision.evaluation_dtype).itemsize
        accumulation_itemsize = np.dtype(plan.precision.accumulation_dtype).itemsize
        route_values = index_itemsize + 2 * dimension * evaluation_itemsize
        route_values += evaluation_itemsize + 1
        source_values = (dimension * dimension + 2 * dimension + 2) * evaluation_itemsize
        source_values += 4
        relation_bytes = route_count * route_values + source_count * source_values
        scalar_workspace = target_size * accumulation_itemsize
        if plan.execution.accumulation == "compensated":
            scalar_workspace *= 2
        plan.budget.admit(
            sources=source_count,
            routes=route_count,
            relation_bytes=relation_bytes,
            scalar_workspace_bytes=scalar_workspace,
        )
        particle_ids = np.asarray(particles.particle_ids, dtype=np.int64)
        stable_order = np.argsort(particle_ids, kind="stable").astype(np.int32)
        bounds = tuple(
            (float(np.asarray(axis.bounds)[0]), float(np.asarray(axis.bounds)[1]))
            for axis in axes
        )
        capabilities = (
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.FIELD_TRANSFER,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            DiscretizationCapability.MATRIX_FREE,
        )
        preparation = PreparationReport(
            capabilities=capabilities,
            diagnostics=(
                f"assignment: {type(plan.assignment).__name__}",
                f"target entities: {layout.axis_entities}",
                "fixed assignment route width",
                "piecewise geometry differentiation with frozen route indices",
                f"nonperiodic boundary policy: {plan.boundary}",
                "workspace bytes are reported per scalar payload",
            ),
            resource_counts={
                "source_capacity": source_count,
                "target_size": target_size,
                "ambient_dimension": dimension,
                "route_width": width,
                "route_count": route_count,
                "relation_bytes": relation_bytes,
                "scalar_workspace_bytes": scalar_workspace,
            },
        )
        assignment_properties = plan.assignment.capabilities
        properties = TransferProperties(
            constant_preserving=(
                assignment_properties.partition_of_unity and plan.boundary == "reject"
            ),
            conservative=(
                assignment_properties.partition_of_unity and plan.boundary == "reject"
            ),
            positivity_preserving=assignment_properties.nonnegative_weights,
            nested=False,
            adjoint_paired=True,
            differentiable_geometry=plan.execution.geometry_ad == "piecewise",
            exact_on=(
                ("constants", "coordinate-affine")
                if assignment_properties.polynomial_reproduction_order >= 1
                else ("constants",)
            ),
        )
        prepared_id = resolved_identifier(
            "prepared_id",
            None,
            {
                "kind": "prepared-particle-grid-splat",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "source_ids": array_tree_fingerprint(particle_ids),
                "assignment": plan.assignment.assignment_id,
                "target_layout": layout.layout_id,
                "target_measure": target_measure.measure_id,
                "preparation": preparation.report_id,
            },
        )
        self.plan = plan
        self.particles = particles
        self.layout = layout
        self.target_measure = target_measure
        self.stable_source_order = jnp.asarray(stable_order)
        self.axis_bounds = bounds
        self.target_shape = layout.shape
        self.target_size = target_size
        self.route_width = width
        self.route_count = route_count
        self.properties = properties
        self.preparation = preparation
        self.prepared_id = prepared_id
        self.artifact_kind = "particle-grid-splat"

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope:
        return self.plan.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        assignment_input: object = None,
    ) -> ParticleGridSplatState:
        """Build fixed assignment routes for one runtime particle configuration."""
        raw = jnp.asarray(position)
        expected = (self.particles.capacity, self.particles.ambient_dimension)
        if raw.shape != expected:
            raise ValueError(f"Particle positions must have shape {expected}.")
        if jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError("Particle positions must be real.")
        geometry = self.plan.precision.geometry(raw)
        if self.plan.execution.geometry_ad == "frozen":
            geometry = jax.lax.stop_gradient(geometry)
            assignment_input = jax.tree.map(
                lambda value: (
                    jax.lax.stop_gradient(value) if eqx.is_array(value) else value
                ),
                assignment_input,
                is_leaf=lambda value: value is None,
            )
        active = self.particles.active_mask
        if active_mask is not None:
            runtime_active = jnp.asarray(active_mask, dtype=bool)
            if runtime_active.shape != active.shape:
                raise ValueError("active_mask must have particle-capacity shape.")
            active = active & runtime_active
        finite = jnp.all(jnp.isfinite(geometry), axis=-1)
        invalid = active & ~finite
        fallback = jnp.asarray(
            tuple(bound[0] for bound in self.axis_bounds), dtype=geometry.dtype
        )
        safe = jnp.where((active & finite)[:, None], geometry, fallback)
        assignment_state = self.plan.assignment.build(
            self.layout,
            self.plan.target.structured_axes,
            self.axis_bounds,
            safe,
            active & finite,
            assignment_input=assignment_input,
        )
        captured = assignment_state.captured_fractions.astype(
            self.plan.precision.evaluation_dtype
        )
        tolerance = jnp.finfo(captured.dtype).eps * max(16, self.route_width)
        supported = active & finite & (captured > tolerance)
        truncated = active & finite & (jnp.abs(captured - 1.0) > tolerance)
        outside = active & finite & ~assignment_state.source_in_domain
        valid = assignment_state.valid & supported[:, None]
        stencil = GatherStencil(
            indices=jax.lax.stop_gradient(assignment_state.indices),
            weights=assignment_state.weights.astype(self.plan.precision.evaluation_dtype),
            source_size=self.target_size,
            valid=jax.lax.stop_gradient(valid),
            support=jax.lax.stop_gradient(supported),
        )
        any_route = jnp.any(valid)
        minimum_weight = jnp.where(
            any_route,
            jnp.min(jnp.where(valid, stencil.weights, jnp.inf)),
            jnp.asarray(0.0, dtype=stencil.weights.dtype),
        )
        margins = []
        for axis_index, (axis, (lower, upper)) in enumerate(
            zip(self.plan.target.structured_axes, self.axis_bounds, strict=True)
        ):
            if axis.periodic:
                margins.append(
                    jnp.full((self.particles.capacity,), jnp.inf, dtype=geometry.dtype)
                )
            else:
                coordinate = safe[:, axis_index]
                margins.append(jnp.minimum(coordinate - lower, upper - coordinate))
        source_margin = jnp.min(jnp.stack(margins, axis=-1), axis=-1)
        any_supported = jnp.any(supported)
        minimum_domain_margin = jnp.where(
            any_supported,
            jnp.min(jnp.where(supported, source_margin, jnp.inf)),
            jnp.asarray(0.0, dtype=geometry.dtype),
        )
        valid_routes = jnp.sum(valid, dtype=jnp.int32)
        dropped_count = jnp.sum(truncated, dtype=jnp.int32)
        invalid_count = jnp.sum(invalid, dtype=jnp.int32)
        boundary_ok = (self.plan.boundary == "drop") | ~jnp.any(truncated)
        successful = ~jnp.any(invalid) & boundary_ok
        return ParticleGridSplatState(
            source_active_mask=active,
            stencil=stencil,
            assignment_state=assignment_state,
            supported_mask=supported,
            truncated_support_mask=truncated,
            out_of_domain_mask=outside,
            invalid_geometry_mask=invalid,
            partition_sums=captured,
            minimum_route_weight=minimum_weight,
            minimum_domain_margin=minimum_domain_margin,
            valid_route_count=valid_routes,
            dropped_source_count=dropped_count,
            invalid_geometry_count=invalid_count,
            successful=successful,
            prepared_id=self.prepared_id,
        )

    def _require_state(self, state: ParticleGridSplatState, /) -> None:
        if not isinstance(state, ParticleGridSplatState):
            raise TypeError("state must be ParticleGridSplatState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Splat state was built by a different prepared transfer.")

    def _source_payload(
        self,
        state: ParticleGridSplatState,
        value: ArrayLike,
        name: str,
        /,
    ) -> Array:
        array = jnp.asarray(value)
        if array.ndim < 1 or int(array.shape[0]) != self.particles.capacity:
            raise ValueError(
                f"{name} must begin with particle capacity {self.particles.capacity}."
            )
        evaluated = cast_stage(array, self.plan.precision.evaluation_dtype)
        active = state.source_active_mask
        mask = active.reshape(active.shape + (1,) * (evaluated.ndim - 1))
        evaluated = eqx.error_if(
            evaluated,
            jnp.any(jnp.where(mask, ~jnp.isfinite(evaluated), False)),
            f"Active {name} values must be finite.",
        )
        return jnp.where(mask, evaluated, jnp.zeros((), dtype=evaluated.dtype))

    def _deposit_flat(
        self,
        state: ParticleGridSplatState,
        source_values: Array,
        /,
    ) -> Array:
        accumulated = cast_stage(source_values, self.plan.precision.accumulation_dtype)
        target = deposit_routes(
            state.stencil,
            accumulated,
            self.stable_source_order,
            self.target_size,
            self.plan.execution.accumulation,
        )
        return state.require_success(target)

    def scatter_route_payload(
        self,
        state: ParticleGridSplatState,
        payload: ArrayLike,
        /,
    ) -> SplatRouteScatterResult:
        """Reduce an already weighted payload for every particle-grid route."""
        self._require_state(state)
        values = jnp.asarray(payload)
        route_shape = state.stencil.indices.shape
        if (
            values.ndim < 2
            or tuple(int(size) for size in values.shape[:2]) != route_shape
        ):
            raise ValueError(
                f"Route payload must begin with route shape {route_shape}; "
                f"got {values.shape}."
            )
        evaluated = cast_stage(values, self.plan.precision.evaluation_dtype)
        payload_shape = evaluated.shape[2:]
        active = self.particles.active_mask.reshape(
            (self.particles.capacity, 1) + (1,) * len(payload_shape)
        )
        evaluated = eqx.error_if(
            evaluated,
            jnp.any(jnp.where(active, ~jnp.isfinite(evaluated), False)),
            "Active route payload values must be finite.",
        )
        evaluated = jnp.where(active, evaluated, jnp.zeros((), dtype=evaluated.dtype))
        accumulated = cast_stage(
            evaluated,
            self.plan.precision.accumulation_dtype,
        )
        target_flat = _scatter_route_payload(
            state.stencil,
            accumulated,
            self.stable_source_order,
            self.target_size,
            self.plan.execution.accumulation,
        )
        target = target_flat.reshape(self.target_shape + payload_shape)
        target = cast_stage(target, self.plan.precision.output_dtype)
        successful = state.successful & jnp.all(jnp.isfinite(target))
        target = state.require_success(target)
        return SplatRouteScatterResult(
            target,
            state.valid_route_count,
            successful,
            execution_policy_id=self.plan.execution.policy_id,
            precision_policy_id=self.plan.precision.policy_id,
        )

    def deposit_content(
        self,
        state: ParticleGridSplatState,
        content: ArrayLike,
        /,
    ) -> SplatDepositResult:
        """Deposit extensive particle content and derive target density."""
        self._require_state(state)
        source = self._source_payload(state, content, "content")
        target_flat = self._deposit_flat(state, source)
        payload_shape = source.shape[1:]
        target_content = target_flat.reshape(self.target_shape + payload_shape)
        target_content = cast_stage(target_content, self.plan.precision.output_dtype)
        measure = self.target_measure.weights.reshape(
            self.target_shape + (1,) * len(payload_shape)
        ).astype(target_content.real.dtype)
        density = target_content / measure
        balance = self._balance_evidence(state, source, target_flat)
        successful = (
            state.successful
            & jnp.all(jnp.isfinite(target_content))
            & jnp.all(jnp.isfinite(density))
            & jnp.isfinite(balance.maximum_absolute_balance_defect)
        )
        return SplatDepositResult(target_content, density, balance, successful)

    def _balance_evidence(
        self,
        state: ParticleGridSplatState,
        source: Array,
        target_flat: Array,
        /,
    ) -> SplatBalanceEvidence:
        active = self.particles.active_mask
        active_mask = active.reshape(active.shape + (1,) * (source.ndim - 1))
        fractions = state.captured_fractions.reshape(
            state.captured_fractions.shape + (1,) * (source.ndim - 1)
        ).astype(source.real.dtype)
        active_values = jnp.where(active_mask, source, 0)
        supported_values = active_values * fractions
        dropped_values = active_values - supported_values
        certification = self.plan.precision.certification_dtype
        active_total = certified_sum(active_values, certification, axis=0)
        supported_total = certified_sum(supported_values, certification, axis=0)
        dropped_total = certified_sum(dropped_values, certification, axis=0)
        dropped_abs = certified_sum(jnp.abs(dropped_values), certification, axis=0)
        active_abs = certified_sum(jnp.abs(active_values), certification, axis=0)
        target_total = certified_sum(target_flat, certification, axis=0)
        target_abs = certified_sum(jnp.abs(target_flat), certification, axis=0)
        defect = active_total - target_total - dropped_total
        maximum_defect = jnp.max(jnp.abs(defect), initial=0.0)
        partition_defect = jnp.max(
            jnp.where(active, jnp.abs(state.partition_sums - 1.0), 0.0),
            initial=0.0,
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.stack(
                    (
                        jnp.max(active_abs, initial=0.0),
                        jnp.max(target_abs, initial=0.0),
                        jnp.max(dropped_abs, initial=0.0),
                    )
                )
            ),
        )
        real_dtype = jnp.asarray(active_total).real.dtype
        epsilon = jnp.finfo(real_dtype).eps
        operation_count = max(
            2, self.route_count + self.particles.capacity + self.target_size
        )
        accumulated_epsilon = operation_count * epsilon
        roundoff_model_valid = accumulated_epsilon < 0.5
        gamma = accumulated_epsilon / jnp.maximum(1.0 - accumulated_epsilon, epsilon)
        tolerance = (gamma + 8.0 * epsilon) * scale
        no_drop = ~jnp.any(state.truncated_support_mask)
        closed_valid = (
            state.successful
            & no_drop
            & roundoff_model_valid
            & (maximum_defect <= tolerance)
            & (partition_defect <= tolerance)
        )
        return SplatBalanceEvidence(
            active_source_total=active_total,
            supported_source_total=supported_total,
            dropped_source_total=dropped_total,
            dropped_source_absolute_total=dropped_abs,
            target_total=target_total,
            balance_defect=defect,
            maximum_absolute_balance_defect=maximum_defect,
            maximum_partition_defect=partition_defect,
            minimum_route_weight=state.minimum_route_weight,
            valid_route_count=state.valid_route_count,
            tolerance=tolerance,
            closed_domain_conservation_valid=closed_valid,
            source_support_id=self.particles.support.support_id,
            target_measure_id=self.target_measure.measure_id,
            execution_policy_id=self.plan.execution.policy_id,
            precision_policy_id=self.plan.precision.policy_id,
        )

    def reconstruct(
        self,
        state: ParticleGridSplatState,
        values: ArrayLike,
        sample_weights: ArrayLike,
        /,
    ) -> SplatReconstructionResult:
        """Reconstruct one intensive field with explicit nonnegative sample weights."""
        self._require_state(state)
        source_values = self._source_payload(state, values, "sample")
        weights = jnp.asarray(sample_weights)
        if weights.shape != (self.particles.capacity,):
            raise ValueError("sample_weights must have the particle-capacity shape.")
        if jnp.issubdtype(weights.dtype, jnp.complexfloating):
            raise TypeError("sample_weights must be real.")
        weights = self.plan.precision.evaluation(weights)
        active = self.particles.active_mask
        weights = eqx.error_if(
            weights,
            jnp.any(active & (~jnp.isfinite(weights) | (weights < 0.0))),
            "Active sample weights must be finite and nonnegative.",
        )
        weights = jnp.where(active, weights, 0.0)
        expanded = weights.reshape(weights.shape + (1,) * (source_values.ndim - 1))
        numerator_flat = self._deposit_flat(state, source_values * expanded)
        denominator_flat = self._deposit_flat(state, weights)
        denominator = denominator_flat.reshape(self.target_shape)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(denominator), initial=0.0))
        tolerance = jnp.finfo(denominator.dtype).eps * max(16, self.route_count) * scale
        support = denominator > tolerance
        numerator = numerator_flat.reshape(self.target_shape + source_values.shape[1:])
        expanded_denominator = denominator.reshape(
            denominator.shape + (1,) * (numerator.ndim - denominator.ndim)
        )
        expanded_support = support.reshape(
            support.shape + (1,) * (numerator.ndim - support.ndim)
        )
        reconstructed = jnp.where(
            expanded_support,
            numerator / jnp.where(expanded_support, expanded_denominator, 1.0),
            0.0,
        )
        reconstructed = cast_stage(reconstructed, self.plan.precision.output_dtype)
        numerator = cast_stage(numerator, self.plan.precision.output_dtype)
        successful = (
            state.successful
            & jnp.all(jnp.isfinite(reconstructed))
            & jnp.all(jnp.isfinite(numerator))
            & jnp.all(jnp.isfinite(denominator))
        )
        return SplatReconstructionResult(
            values=reconstructed,
            numerator=numerator,
            denominator=denominator,
            support=support,
            denominator_tolerance=tolerance,
            zero_coverage_count=jnp.sum(~support, dtype=jnp.int32),
            successful=successful,
        )

    def gather(
        self,
        state: ParticleGridSplatState,
        target_values: ArrayLike,
        /,
    ) -> InterpolationResult:
        """Interpolate target-grid values onto the runtime particle positions."""
        self._require_state(state)
        array = jnp.asarray(target_values)
        if (
            array.ndim < len(self.target_shape)
            or tuple(int(size) for size in array.shape[: len(self.target_shape)])
            != self.target_shape
        ):
            raise ValueError(
                f"target_values must begin with target shape {self.target_shape}."
            )
        payload_shape = array.shape[len(self.target_shape) :]
        flat = array.reshape((self.target_size,) + payload_shape)
        flat = cast_stage(flat, self.plan.precision.evaluation_dtype)
        result = apply_gather_stencil(flat, state.stencil)
        values = state.require_success(result.values)
        return InterpolationResult(
            cast_stage(values, self.plan.precision.output_dtype), result.support
        )


__all__ = [
    "ParticleGridSplatPlan",
    "ParticleGridSplatState",
    "PreparedParticleGridSplat",
]
