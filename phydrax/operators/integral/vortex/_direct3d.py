#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._geometry_precision import GeometryPrecisionPolicy
from ...._strict import StrictModule
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexVelocity,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
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
    """Chunked all-pairs free-space plan for Gaussian 3-D vortex blobs."""

    kernel: GaussianErfVortexKernel3D
    precision: GeometryPrecisionPolicy
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    memory_budget_bytes: int | None = eqx.field(static=True)
    interaction_budget: int | None = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    supports_velocity: bool = eqx.field(static=True)
    supports_velocity_gradient: bool = eqx.field(static=True)
    supports_vorticity: bool = eqx.field(static=True)
    supports_vector_strength: bool = eqx.field(static=True)
    supports_variable_core_radius: bool = eqx.field(static=True)
    core_radius_convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kernel: GaussianErfVortexKernel3D | None = None,
        precision: GeometryPrecisionPolicy | None = None,
        source_chunk_size: int = 128,
        target_chunk_size: int = 128,
        memory_budget_bytes: int | None = None,
        interaction_budget: int | None = None,
    ):
        kernel_ = GaussianErfVortexKernel3D() if kernel is None else kernel
        if not isinstance(kernel_, GaussianErfVortexKernel3D):
            raise TypeError("kernel must be GaussianErfVortexKernel3D or None.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be GeometryPrecisionPolicy or None.")
        requested_dtypes = (
            precision_.coordinate_dtype,
            precision_.compute_dtype,
            precision_.accumulation_dtype,
            precision_.decision_dtype,
            precision_.output_dtype,
        )
        if any(
            value is not None and not value.startswith("float")
            for value in requested_dtypes
        ):
            raise ValueError("The 3-D direct vortex backend requires real precision.")
        source_chunk = int(source_chunk_size)
        target_chunk = int(target_chunk_size)
        if source_chunk <= 0 or target_chunk <= 0:
            raise ValueError("Direct vortex chunk sizes must be positive.")
        memory_budget = None if memory_budget_bytes is None else int(memory_budget_bytes)
        interactions = None if interaction_budget is None else int(interaction_budget)
        if memory_budget is not None and memory_budget <= 0:
            raise ValueError("memory_budget_bytes must be positive or None.")
        if interactions is not None and interactions <= 0:
            raise ValueError("interaction_budget must be positive or None.")

        self.kernel = kernel_
        self.precision = precision_
        self.source_chunk_size = source_chunk
        self.target_chunk_size = target_chunk
        self.memory_budget_bytes = memory_budget
        self.interaction_budget = interactions
        self.dimension = 3
        self.supports_velocity = True
        self.supports_velocity_gradient = True
        self.supports_vorticity = True
        self.supports_vector_strength = True
        self.supports_variable_core_radius = True
        self.core_radius_convention = kernel_.core_radius_convention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-erf-direct-vortex-plan-3d-v1",
                "kernel": kernel_.kernel_id,
                "precision": precision_.policy_id,
                "source_chunk_size": source_chunk,
                "target_chunk_size": target_chunk,
                "memory_budget_bytes": memory_budget,
                "interaction_budget": interactions,
            }
        )

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
    ) -> PreparedGaussianErfDirectVortex3D:
        source_count = int(source_capacity)
        target_count = source_count if target_capacity is None else int(target_capacity)
        if source_count <= 0 or target_count <= 0:
            raise ValueError("Direct vortex capacities must be positive.")
        source_chunk = min(self.source_chunk_size, source_count)
        target_chunk = min(self.target_chunk_size, target_count)
        source_chunks = (source_count + source_chunk - 1) // source_chunk
        target_chunks = (target_count + target_chunk - 1) // target_chunk
        interaction_count = source_count * target_count
        if (
            self.interaction_budget is not None
            and interaction_count > self.interaction_budget
        ):
            raise ValueError(
                f"Direct vortex evaluation requires {interaction_count} interactions, "
                f"exceeding budget {self.interaction_budget}."
            )

        # Conservative upper bound for the pair kernel's fused scalar workspace,
        # its masks, chunk inputs, and all three requested target accumulators.
        scalar_bytes = 8
        estimated_bytes = (
            scalar_bytes
            * (48 * source_chunk * target_chunk + 7 * source_chunk + 15 * target_chunk)
            + 3 * source_chunk * target_chunk
        )
        if (
            self.memory_budget_bytes is not None
            and estimated_bytes > self.memory_budget_bytes
        ):
            raise ValueError(
                f"Direct vortex chunks require an estimated {estimated_bytes} bytes, "
                f"exceeding budget {self.memory_budget_bytes}."
            )
        return PreparedGaussianErfDirectVortex3D(
            self,
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
    precision: GeometryPrecisionPolicy
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    source_chunk_count: int = eqx.field(static=True)
    target_chunk_count: int = eqx.field(static=True)
    estimated_working_set_bytes: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    supports_velocity: bool = eqx.field(static=True)
    supports_velocity_gradient: bool = eqx.field(static=True)
    supports_vorticity: bool = eqx.field(static=True)
    supports_vector_strength: bool = eqx.field(static=True)
    supports_variable_core_radius: bool = eqx.field(static=True)
    core_radius_convention: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GaussianErfDirectVortexPlan3D,
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
        self.plan = plan
        self.precision = plan.precision
        self.source_capacity = source_capacity
        self.target_capacity = target_capacity
        self.source_chunk_size = source_chunk_size
        self.target_chunk_size = target_chunk_size
        self.source_chunk_count = source_chunk_count
        self.target_chunk_count = target_chunk_count
        self.estimated_working_set_bytes = estimated_working_set_bytes
        self.dimension = 3
        self.supports_velocity = plan.supports_velocity
        self.supports_velocity_gradient = plan.supports_velocity_gradient
        self.supports_vorticity = plan.supports_vorticity
        self.supports_vector_strength = plan.supports_vector_strength
        self.supports_variable_core_radius = plan.supports_variable_core_radius
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
            }
        )

    def evaluate(
        self,
        position: ArrayLike,
        strength: ArrayLike,
        core_radius: ArrayLike,
        /,
        *,
        targets: ArrayLike | None = None,
        target_source_indices: ArrayLike | None = None,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        if not isinstance(request, VortexFieldRequest):
            raise TypeError("request must be a VortexFieldRequest.")
        source_position = jnp.asarray(position)
        source_strength = jnp.asarray(strength)
        source_core = jnp.asarray(core_radius)
        expected_source_shape = (self.source_capacity, 3)
        if source_position.shape != expected_source_shape:
            raise ValueError(
                f"position must have shape {expected_source_shape}; got "
                f"{source_position.shape}."
            )
        if source_strength.shape != expected_source_shape:
            raise ValueError(
                f"strength must have shape {expected_source_shape}; got "
                f"{source_strength.shape}."
            )
        if source_core.shape == ():
            source_core = jnp.broadcast_to(source_core, (self.source_capacity,))
        elif source_core.shape != (self.source_capacity,):
            raise ValueError(
                f"core_radius must be scalar or shape ({self.source_capacity},); "
                f"got {source_core.shape}."
            )
        if not jnp.issubdtype(source_position.dtype, jnp.floating):
            raise TypeError("position must use a real floating dtype.")
        if not jnp.issubdtype(source_strength.dtype, jnp.floating):
            raise TypeError("strength must use a real floating dtype.")
        if not jnp.issubdtype(source_core.dtype, jnp.floating):
            raise TypeError("core_radius must use a real floating dtype.")

        source_targets = targets is None
        if source_targets:
            if self.target_capacity != self.source_capacity:
                raise ValueError(
                    "targets=None requires target_capacity == source_capacity."
                )
            query_position = source_position
        else:
            query_position = jnp.asarray(targets)
            expected_target_shape = (self.target_capacity, 3)
            if query_position.shape != expected_target_shape:
                raise ValueError(
                    f"targets must have shape {expected_target_shape}; got "
                    f"{query_position.shape}."
                )
            if not jnp.issubdtype(query_position.dtype, jnp.floating):
                raise TypeError("targets must use a real floating dtype.")

        if target_source_indices is None:
            if source_targets:
                self_indices = jnp.arange(self.target_capacity, dtype=jnp.int32)
                mapping_mode = "implicit-source-identity"
            else:
                self_indices = -jnp.ones((self.target_capacity,), dtype=jnp.int32)
                mapping_mode = "none"
        else:
            self_indices = jnp.asarray(target_source_indices)
            if self_indices.shape != (self.target_capacity,):
                raise ValueError(
                    "target_source_indices must have shape "
                    f"({self.target_capacity},); got {self_indices.shape}."
                )
            if not jnp.issubdtype(self_indices.dtype, jnp.integer):
                raise TypeError("target_source_indices must use an integer dtype.")
            self_indices = self_indices.astype(jnp.int32)
            self_indices = eqx.error_if(
                self_indices,
                jnp.any((self_indices < -1) | (self_indices >= self.source_capacity)),
                "target_source_indices entries must be -1 or valid source indices.",
            )
            mapping_mode = "explicit"

        self.precision.validate_coordinates(source_position)
        self.precision.validate_coordinates(query_position)
        precision_evidence = self.precision.evidence_for(source_position)
        inputs_finite = (
            jnp.all(jnp.isfinite(source_position))
            & jnp.all(jnp.isfinite(source_strength))
            & jnp.all(jnp.isfinite(source_core))
            & jnp.all(source_core > 0.0)
            & jnp.all(jnp.isfinite(query_position))
        )
        safe_position = jnp.where(jnp.isfinite(source_position), source_position, 0.0)
        safe_strength = jnp.where(jnp.isfinite(source_strength), source_strength, 0.0)
        safe_core = jnp.where(
            jnp.isfinite(source_core) & (source_core > 0.0), source_core, 1.0
        )
        safe_targets = jnp.where(jnp.isfinite(query_position), query_position, 0.0)
        safe_position = self.precision.compute(safe_position)
        safe_strength = self.precision.compute(safe_strength)
        safe_core = self.precision.compute(safe_core)
        safe_targets = self.precision.compute(safe_targets)
        compute_dtype = jnp.result_type(
            safe_position.dtype, safe_strength.dtype, safe_core.dtype, safe_targets.dtype
        )
        safe_position = safe_position.astype(compute_dtype)
        safe_strength = safe_strength.astype(compute_dtype)
        safe_core = safe_core.astype(compute_dtype)
        safe_targets = safe_targets.astype(compute_dtype)

        padded_source_count = self.source_chunk_count * self.source_chunk_size
        source_padding = padded_source_count - self.source_capacity
        padded_target_count = self.target_chunk_count * self.target_chunk_size
        target_padding = padded_target_count - self.target_capacity
        source_position_padded = jnp.pad(safe_position, ((0, source_padding), (0, 0)))
        source_strength_padded = jnp.pad(safe_strength, ((0, source_padding), (0, 0)))
        source_core_padded = jnp.pad(
            safe_core, ((0, source_padding),), constant_values=1.0
        )
        source_valid = jnp.arange(padded_source_count) < self.source_capacity
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
                velocity_sum = velocity_sum + jnp.sum(
                    jnp.where(mask_vector, pair.velocity, 0.0).astype(accumulation_dtype),
                    axis=1,
                )
                gradient_sum = gradient_sum + jnp.sum(
                    jnp.where(mask_matrix, pair.velocity_gradient, 0.0).astype(
                        accumulation_dtype
                    ),
                    axis=1,
                )
                vorticity_sum = vorticity_sum + jnp.sum(
                    jnp.where(mask_vector, pair.vorticity, 0.0).astype(
                        accumulation_dtype
                    ),
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
        velocity_raw = velocity_chunks.reshape(padded_target_count, 3)[
            : self.target_capacity
        ]
        gradient_raw = gradient_chunks.reshape(padded_target_count, 3, 3)[
            : self.target_capacity
        ]
        vorticity_raw = vorticity_chunks.reshape(padded_target_count, 3)[
            : self.target_capacity
        ]
        velocity_raw = self.precision.output(velocity_raw)
        gradient_raw = self.precision.output(gradient_raw)
        vorticity_raw = self.precision.output(vorticity_raw)
        outputs_finite = (
            jnp.all(jnp.isfinite(velocity_raw))
            & jnp.all(jnp.isfinite(gradient_raw))
            & jnp.all(jnp.isfinite(vorticity_raw))
        )
        resource_budget_satisfied = jnp.asarray(True)
        successful = inputs_finite & outputs_finite & resource_budget_satisfied
        velocity_result = jnp.where(successful, velocity_raw, jnp.nan)
        gradient_result = jnp.where(successful, gradient_raw, jnp.nan)
        vorticity_result = jnp.where(successful, vorticity_raw, jnp.nan)
        minimum_core = jnp.where(
            jnp.all(jnp.isfinite(source_core) & (source_core > 0.0)),
            jnp.min(source_core),
            jnp.asarray(jnp.nan, dtype=source_core.dtype),
        )
        backend_diagnostics = DirectVortexResourceEvidence3D(
            self.source_chunk_size,
            self.target_chunk_size,
            self.source_chunk_count,
            self.target_chunk_count,
            self.estimated_working_set_bytes,
            self.plan.memory_budget_bytes,
            self.source_capacity * self.target_capacity,
            self.plan.interaction_budget,
            mapping_mode,
            True,
            precision_evidence,
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
                "kind": "gaussian-erf-direct-vortex-evaluation-3d-v1",
                "prepared": self.prepared_id,
                "request": request.request_id,
                "target_mode": "sources" if source_targets else "arbitrary",
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
