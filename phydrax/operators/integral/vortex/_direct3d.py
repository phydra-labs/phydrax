#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.vortex._capabilities import VortexVelocityCapabilities
from ....discretization.vortex._compatibility import (
    request_fields,
    validate_vortex_velocity_evaluation,
    VortexVelocityCompatibility,
)
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexVelocity,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ....discretization.vortex._precision import VortexPrecisionPolicy
from ....discretization.vortex._source import VortexSourceState, VortexTargetState
from ._gaussian3d import GaussianErfVortexKernel3D


class DirectVortexResourceEvidence3D(StrictModule):
    """Auditable fixed-shape resource and identity policy for one evaluation."""

    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    source_chunk_count: int = eqx.field(static=True)
    target_chunk_count: int = eqx.field(static=True)
    estimated_working_set_bytes: int = eqx.field(static=True)
    memory_budget_bytes: int | None = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    interaction_budget: int | None = eqx.field(static=True)
    self_mapping_mode: str = eqx.field(static=True)
    free_space: bool = eqx.field(static=True)
    precision_evidence: object


class GaussianErfDirectVortexPlan3D(AbstractVortexVelocityPlan):
    """Chunked free-space Gaussian Biot--Savart evaluation with hard budgets."""

    kernel: GaussianErfVortexKernel3D
    precision: VortexPrecisionPolicy
    capabilities: VortexVelocityCapabilities
    maximum_sources: int = eqx.field(static=True)
    maximum_targets: int = eqx.field(static=True)
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    maximum_interactions: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    core_radius_convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_sources: int,
        maximum_targets: int | None = None,
        kernel: GaussianErfVortexKernel3D | None = None,
        precision: VortexPrecisionPolicy | None = None,
        source_chunk_size: int = 128,
        target_chunk_size: int = 128,
        maximum_interactions: int | None = None,
        maximum_workspace_bytes: int = 64 * 1024 * 1024,
        plan_id: str | None = None,
    ):
        sources = int(maximum_sources)
        targets = sources if maximum_targets is None else int(maximum_targets)
        kernel_ = GaussianErfVortexKernel3D() if kernel is None else kernel
        if not isinstance(kernel_, GaussianErfVortexKernel3D):
            raise TypeError("kernel must be GaussianErfVortexKernel3D or None.")
        precision_ = VortexPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, VortexPrecisionPolicy):
            raise TypeError("precision must be VortexPrecisionPolicy or None.")
        source_chunk = int(source_chunk_size)
        target_chunk = int(target_chunk_size)
        interactions = (
            sources * targets
            if maximum_interactions is None
            else int(maximum_interactions)
        )
        workspace = int(maximum_workspace_bytes)
        if sources <= 0 or targets <= 0:
            raise ValueError("Direct vortex source and target budgets must be positive.")
        if source_chunk <= 0 or target_chunk <= 0:
            raise ValueError("Direct vortex chunk sizes must be positive.")
        if interactions <= 0 or workspace <= 0:
            raise ValueError(
                "Direct vortex interaction and workspace budgets must be positive."
            )
        largest_source_chunk = min(source_chunk, sources)
        largest_target_chunk = min(target_chunk, targets)
        estimated = (
            8
            * (
                48 * largest_source_chunk * largest_target_chunk
                + 7 * largest_source_chunk
                + 15 * largest_target_chunk
            )
            + 3 * largest_source_chunk * largest_target_chunk
        )
        if estimated > workspace:
            raise ValueError(
                "Configured direct-vortex chunk block exceeds maximum_workspace_bytes."
            )
        capabilities = VortexVelocityCapabilities(
            3,
            required_source_fields=(
                "positions",
                "strength",
                "active_mask",
                "core_radius",
            ),
            supported_fields=("velocity", "velocity_gradient", "vorticity"),
            domain="free-space",
            precision=precision_,
            derivatives=(
                "source-position",
                "source-strength",
                "source-core-radius",
                "target-position",
            ),
            acceleration="direct",
        )
        self.kernel = kernel_
        self.precision = precision_
        self.capabilities = capabilities
        self.maximum_sources = sources
        self.maximum_targets = targets
        self.source_chunk_size = source_chunk
        self.target_chunk_size = target_chunk
        self.maximum_interactions = interactions
        self.maximum_workspace_bytes = workspace
        self.dimension = 3
        self.core_radius_convention = kernel_.core_radius_convention
        generated = canonical_fingerprint(
            {
                "kind": "gaussian-erf-direct-vortex-plan-3d",
                "kernel": kernel_.kernel_id,
                "precision": precision_.policy_id,
                "maximum_sources": sources,
                "maximum_targets": targets,
                "source_chunk_size": source_chunk,
                "target_chunk_size": target_chunk,
                "maximum_interactions": interactions,
                "maximum_workspace_bytes": workspace,
            }
        )
        identifier = generated if plan_id is None else str(plan_id).strip()
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.plan_id = identifier

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
        source_kind: str = "particle",
        target_topology: str = "same-support",
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> PreparedGaussianErfDirectVortex3D:
        source_count = int(source_capacity)
        target_count = source_count if target_capacity is None else int(target_capacity)
        if source_count <= 0 or target_count <= 0:
            raise ValueError("Direct vortex capacities must be positive.")
        if source_count > self.maximum_sources or target_count > self.maximum_targets:
            raise ValueError("Prepared direct-vortex capacity exceeds its plan budget.")
        interaction_count = source_count * target_count
        if interaction_count > self.maximum_interactions:
            raise ValueError("Prepared direct-vortex interactions exceed their budget.")
        source_chunk = min(self.source_chunk_size, source_count)
        target_chunk = min(self.target_chunk_size, target_count)
        source_chunks = (source_count + source_chunk - 1) // source_chunk
        target_chunks = (target_count + target_chunk - 1) // target_chunk
        estimated_bytes = (
            8 * (48 * source_chunk * target_chunk + 7 * source_chunk + 15 * target_chunk)
            + 3 * source_chunk * target_chunk
        )
        if estimated_bytes > self.maximum_workspace_bytes:
            raise ValueError("Prepared direct-vortex workspace exceeds its budget.")
        compatibility = VortexVelocityCompatibility(
            self.capabilities,
            source_capacity=source_count,
            target_capacity=target_count,
            source_kind=source_kind,
            target_topology=target_topology,
            requested_fields=request_fields(request),
        )
        return PreparedGaussianErfDirectVortex3D(
            self,
            compatibility,
            source_capacity=source_count,
            target_capacity=target_count,
            source_chunk_size=source_chunk,
            target_chunk_size=target_chunk,
            source_chunk_count=source_chunks,
            target_chunk_count=target_chunks,
            estimated_working_set_bytes=estimated_bytes,
        )


class PreparedGaussianErfDirectVortex3D(AbstractPreparedVortexVelocity):
    """Prepared fixed-capacity direct evaluator with identity-only self removal."""

    plan: GaussianErfDirectVortexPlan3D
    precision: VortexPrecisionPolicy
    capabilities: VortexVelocityCapabilities
    compatibility: VortexVelocityCompatibility
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    source_chunk_count: int = eqx.field(static=True)
    target_chunk_count: int = eqx.field(static=True)
    estimated_working_set_bytes: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    core_radius_convention: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GaussianErfDirectVortexPlan3D,
        compatibility: VortexVelocityCompatibility,
        /,
        *,
        source_capacity: int,
        target_capacity: int,
        source_chunk_size: int,
        target_chunk_size: int,
        source_chunk_count: int,
        target_chunk_count: int,
        estimated_working_set_bytes: int,
    ):
        if not isinstance(compatibility, VortexVelocityCompatibility):
            raise TypeError("compatibility must be VortexVelocityCompatibility.")
        self.plan = plan
        self.precision = plan.precision
        self.capabilities = plan.capabilities
        self.compatibility = compatibility
        self.source_capacity = source_capacity
        self.target_capacity = target_capacity
        self.source_chunk_size = source_chunk_size
        self.target_chunk_size = target_chunk_size
        self.source_chunk_count = source_chunk_count
        self.target_chunk_count = target_chunk_count
        self.estimated_working_set_bytes = estimated_working_set_bytes
        self.dimension = 3

        self.core_radius_convention = plan.core_radius_convention
        self.backend_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-erf-direct-vortex-3d-v1",
                "plan": plan.plan_id,
                "source_capacity": source_capacity,
                "target_capacity": target_capacity,
                "source_chunk_size": source_chunk_size,
                "target_chunk_size": target_chunk_size,
                "estimated_working_set_bytes": estimated_working_set_bytes,
                "compatibility": compatibility.compatibility_id,
            }
        )

    def evaluate(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        source, target = validate_vortex_velocity_evaluation(
            self.capabilities,
            self.compatibility,
            source,
            target,
            request,
        )
        safe_position = self.precision.compute(source.safe_positions())
        safe_strength = self.precision.compute(source.safe_strength())
        safe_core = self.precision.compute(source.safe_core_radius())
        safe_targets = self.precision.compute(target.positions)
        self_indices = (
            -jnp.ones((self.target_capacity,), dtype=jnp.int32)
            if target.source_indices is None
            else target.source_indices
        )
        mapping_mode = "none" if target.source_indices is None else "explicit"
        compute_dtype = jnp.result_type(
            safe_position.dtype, safe_strength.dtype, safe_core.dtype, safe_targets.dtype
        )
        safe_position = safe_position.astype(compute_dtype)
        safe_strength = safe_strength.astype(compute_dtype)
        safe_core = safe_core.astype(compute_dtype)
        safe_targets = safe_targets.astype(compute_dtype)
        inputs_finite = (
            jnp.all(jnp.isfinite(safe_position))
            & jnp.all(jnp.isfinite(safe_strength))
            & jnp.all(jnp.isfinite(safe_core))
            & jnp.all(safe_core > 0.0)
            & jnp.all(jnp.isfinite(safe_targets))
        )

        padded_source_count = self.source_chunk_count * self.source_chunk_size
        source_padding = padded_source_count - self.source_capacity
        padded_target_count = self.target_chunk_count * self.target_chunk_size
        target_padding = padded_target_count - self.target_capacity
        source_position_padded = jnp.pad(safe_position, ((0, source_padding), (0, 0)))
        source_strength_padded = jnp.pad(safe_strength, ((0, source_padding), (0, 0)))
        source_core_padded = jnp.pad(
            safe_core, ((0, source_padding),), constant_values=1.0
        )
        source_valid = jnp.pad(
            source.active_mask,
            ((0, source_padding),),
            constant_values=False,
        )
        source_indices = jnp.arange(padded_source_count, dtype=jnp.int32)
        target_position_padded = jnp.pad(safe_targets, ((0, target_padding), (0, 0)))
        target_valid = jnp.arange(padded_target_count) < self.target_capacity
        target_indices_padded = jnp.pad(
            self_indices, ((0, target_padding),), constant_values=-1
        )

        source_chunks = (
            source_position_padded.reshape(
                self.source_chunk_count, self.source_chunk_size, 3
            ),
            source_strength_padded.reshape(
                self.source_chunk_count, self.source_chunk_size, 3
            ),
            source_core_padded.reshape(self.source_chunk_count, self.source_chunk_size),
            source_valid.reshape(self.source_chunk_count, self.source_chunk_size),
            source_indices.reshape(self.source_chunk_count, self.source_chunk_size),
        )
        target_chunks = (
            target_position_padded.reshape(
                self.target_chunk_count, self.target_chunk_size, 3
            ),
            target_valid.reshape(self.target_chunk_count, self.target_chunk_size),
            target_indices_padded.reshape(
                self.target_chunk_count, self.target_chunk_size
            ),
        )
        accumulation_dtype = self.precision.accumulation(
            jnp.zeros((), dtype=compute_dtype)
        ).dtype

        def evaluate_target_chunk(target_chunk):
            chunk_targets, chunk_target_valid, chunk_self_indices = target_chunk
            initial = (
                jnp.zeros((self.target_chunk_size, 3), dtype=accumulation_dtype),
                jnp.zeros((self.target_chunk_size, 3, 3), dtype=accumulation_dtype),
                jnp.zeros((self.target_chunk_size, 3), dtype=accumulation_dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )

            def accumulate_source_chunk(carry, source_chunk):
                (
                    velocity_sum,
                    gradient_sum,
                    vorticity_sum,
                    active_count,
                    excluded_count,
                    coincident_count,
                ) = carry
                (
                    chunk_sources,
                    chunk_strength,
                    chunk_core,
                    chunk_source_valid,
                    chunk_source_indices,
                ) = source_chunk
                displacement = chunk_targets[:, None, :] - chunk_sources[None, :, :]
                valid_pair = chunk_target_valid[:, None] & chunk_source_valid[None, :]
                self_pair = valid_pair & (
                    chunk_self_indices[:, None] == chunk_source_indices[None, :]
                )
                active_pair = valid_pair & ~self_pair
                pair = self.plan.kernel.evaluate(
                    displacement,
                    chunk_strength[None, :, :],
                    chunk_core[None, :],
                )
                mask_vector = active_pair[..., None]
                mask_matrix = active_pair[..., None, None]
                if request.velocity:
                    velocity_sum = velocity_sum + self.precision.sum(
                        jnp.where(mask_vector, pair.velocity, 0.0),
                        axis=1,
                    )
                if request.velocity_gradient:
                    gradient_sum = gradient_sum + self.precision.sum(
                        jnp.where(mask_matrix, pair.velocity_gradient, 0.0),
                        axis=1,
                    )
                if request.vorticity:
                    vorticity_sum = vorticity_sum + self.precision.sum(
                        jnp.where(mask_vector, pair.vorticity, 0.0),
                        axis=1,
                    )
                active_count = active_count + jnp.sum(active_pair, dtype=jnp.int32)
                excluded_count = excluded_count + jnp.sum(self_pair, dtype=jnp.int32)
                coincident_count = coincident_count + jnp.sum(
                    active_pair & pair.coincident, dtype=jnp.int32
                )
                return (
                    velocity_sum,
                    gradient_sum,
                    vorticity_sum,
                    active_count,
                    excluded_count,
                    coincident_count,
                ), None

            result, _ = jax.lax.scan(accumulate_source_chunk, initial, source_chunks)
            return result

        (
            velocity_chunks,
            gradient_chunks,
            vorticity_chunks,
            active_counts,
            excluded_counts,
            coincident_counts,
        ) = jax.lax.map(evaluate_target_chunk, target_chunks)
        velocity_raw = self.precision.output(
            velocity_chunks.reshape(padded_target_count, 3)[: self.target_capacity]
        )
        gradient_raw = self.precision.output(
            gradient_chunks.reshape(padded_target_count, 3, 3)[: self.target_capacity]
        )
        vorticity_raw = self.precision.output(
            vorticity_chunks.reshape(padded_target_count, 3)[: self.target_capacity]
        )
        outputs_finite = jnp.asarray(True)
        if request.velocity:
            outputs_finite = outputs_finite & jnp.all(jnp.isfinite(velocity_raw))
        if request.velocity_gradient:
            outputs_finite = outputs_finite & jnp.all(jnp.isfinite(gradient_raw))
        if request.vorticity:
            outputs_finite = outputs_finite & jnp.all(jnp.isfinite(vorticity_raw))
        resource_budget_satisfied = jnp.asarray(True)
        successful = inputs_finite & outputs_finite & resource_budget_satisfied
        velocity_result = jnp.where(successful, velocity_raw, jnp.nan)
        gradient_result = jnp.where(successful, gradient_raw, jnp.nan)
        vorticity_result = jnp.where(successful, vorticity_raw, jnp.nan)
        minimum_core = jnp.min(jnp.where(source.active_mask, safe_core, jnp.inf))
        backend_diagnostics = DirectVortexResourceEvidence3D(
            self.source_chunk_size,
            self.target_chunk_size,
            self.source_chunk_count,
            self.target_chunk_count,
            self.estimated_working_set_bytes,
            self.plan.maximum_workspace_bytes,
            self.source_capacity * self.target_capacity,
            self.plan.maximum_interactions,
            mapping_mode,
            True,
            self.precision.policy_id,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(self.source_capacity, dtype=jnp.int32),
            jnp.asarray(self.target_capacity, dtype=jnp.int32),
            jnp.sum(active_counts, dtype=jnp.int32),
            jnp.sum(excluded_counts, dtype=jnp.int32),
            jnp.sum(coincident_counts, dtype=jnp.int32),
            minimum_core,
            inputs_finite,
            outputs_finite,
            resource_budget_satisfied,
            successful,
            backend_diagnostics,
        )
        evaluation_id = canonical_fingerprint(
            {
                "kind": "gaussian-erf-direct-vortex-evaluation-3d",
                "prepared": self.prepared_id,
                "request": request.request_id,
                "source": source.source_id,
                "target": target.target_id,
                "target_topology": self.compatibility.target_topology,
                "self_mapping": mapping_mode,
            }
        )
        return VortexVelocityEvaluation(
            velocity_result if request.velocity else None,
            gradient_result if request.velocity_gradient else None,
            vorticity_result if request.vorticity else None,
            successful,
            self.backend_id,
            evaluation_id,
            diagnostics,
        )


__all__ = [
    "DirectVortexResourceEvidence3D",
    "GaussianErfDirectVortexPlan3D",
    "PreparedGaussianErfDirectVortex3D",
]
