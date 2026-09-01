#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
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
from ._gaussian2d import (
    gaussian_vortex_velocity_2d,
    gaussian_vortex_velocity_gradient_2d,
    gaussian_vortex_vorticity_2d,
)


class DirectVortexResourceEvidence(StrictModule, NonTrainableState):
    """Static interaction and temporary-storage certificate."""

    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    source_chunk_count: int = eqx.field(static=True)
    target_chunk_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    maximum_interactions: int = eqx.field(static=True)
    estimated_workspace_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class GaussianDirectDiagnostics2D(StrictModule):
    """Direct-backend details retained inside common velocity diagnostics."""

    resource_evidence: DirectVortexResourceEvidence
    source_chunk_count: Array
    target_chunk_count: Array
    pair_block_capacity: Array


def _workspace_bytes(source_chunk_size: int, target_chunk_size: int, /) -> int:
    # Conservative float64 accounting for geometry, three requested kernels,
    # pair masks, reductions, and one output target block.
    pair_capacity = source_chunk_size * target_chunk_size
    return 8 * (12 * pair_capacity + 7 * target_chunk_size + 4 * source_chunk_size)


def _resolved_id(name: str, value: str | None, payload: object, /) -> str:
    identifier = canonical_fingerprint(payload) if value is None else str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class GaussianDirectVortexPlan2D(AbstractVortexVelocityPlan):
    """Chunked free-space Gaussian Biot--Savart evaluation with hard budgets."""

    precision: VortexPrecisionPolicy
    capabilities: VortexVelocityCapabilities
    maximum_sources: int = eqx.field(static=True)
    maximum_targets: int = eqx.field(static=True)
    source_chunk_size: int = eqx.field(static=True)
    target_chunk_size: int = eqx.field(static=True)
    maximum_interactions: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_sources: int,
        maximum_targets: int | None = None,
        source_chunk_size: int = 128,
        target_chunk_size: int = 128,
        maximum_interactions: int | None = None,
        maximum_workspace_bytes: int = 64 * 1024 * 1024,
        precision: VortexPrecisionPolicy | None = None,
        plan_id: str | None = None,
    ):
        sources = int(maximum_sources)
        targets = sources if maximum_targets is None else int(maximum_targets)
        source_chunk = int(source_chunk_size)
        target_chunk = int(target_chunk_size)
        workspace = int(maximum_workspace_bytes)
        if sources <= 0 or targets <= 0:
            raise ValueError("Direct vortex source and target budgets must be positive.")
        if source_chunk <= 0 or target_chunk <= 0:
            raise ValueError("Direct vortex chunk sizes must be positive.")
        if workspace <= 0:
            raise ValueError("maximum_workspace_bytes must be positive.")
        interactions = (
            sources * targets
            if maximum_interactions is None
            else int(maximum_interactions)
        )
        if interactions <= 0:
            raise ValueError("maximum_interactions must be positive.")
        largest_workspace = _workspace_bytes(
            min(source_chunk, sources),
            min(target_chunk, targets),
        )
        if largest_workspace > workspace:
            raise ValueError(
                "Configured direct-vortex chunk block exceeds maximum_workspace_bytes."
            )
        precision_ = VortexPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, VortexPrecisionPolicy):
            raise TypeError("precision must be VortexPrecisionPolicy or None.")
        capabilities = VortexVelocityCapabilities(
            2,
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
        self.precision = precision_
        self.capabilities = capabilities
        self.maximum_sources = sources
        self.maximum_targets = targets
        self.source_chunk_size = source_chunk
        self.target_chunk_size = target_chunk
        self.maximum_interactions = interactions
        self.maximum_workspace_bytes = workspace
        self.dimension = 2
        self.plan_id = _resolved_id(
            "plan_id",
            plan_id,
            {
                "kind": "gaussian-direct-vortex-plan-2d",
                "maximum_sources": sources,
                "maximum_targets": targets,
                "source_chunk_size": source_chunk,
                "target_chunk_size": target_chunk,
                "maximum_interactions": interactions,
                "maximum_workspace_bytes": workspace,
                "precision": precision_.policy_id,
            },
        )

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
        source_kind: str = "particle",
        target_topology: str = "same-support",
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> PreparedGaussianDirectVortex2D:
        sources = int(source_capacity)
        targets = sources if target_capacity is None else int(target_capacity)
        if sources <= 0 or targets <= 0:
            raise ValueError("Prepared direct-vortex capacities must be positive.")
        if sources > self.maximum_sources or targets > self.maximum_targets:
            raise ValueError("Prepared direct-vortex capacity exceeds its plan budget.")
        if sources * targets > self.maximum_interactions:
            raise ValueError("Prepared direct-vortex interactions exceed their budget.")
        source_chunk = min(self.source_chunk_size, sources)
        target_chunk = min(self.target_chunk_size, targets)
        estimated = _workspace_bytes(source_chunk, target_chunk)
        if estimated > self.maximum_workspace_bytes:
            raise ValueError("Prepared direct-vortex workspace exceeds its budget.")
        source_chunks = (sources + source_chunk - 1) // source_chunk
        target_chunks = (targets + target_chunk - 1) // target_chunk
        evidence_id = canonical_fingerprint(
            {
                "kind": "direct-vortex-resource-evidence",
                "plan": self.plan_id,
                "source_capacity": sources,
                "target_capacity": targets,
                "source_chunk_size": source_chunk,
                "target_chunk_size": target_chunk,
                "source_chunk_count": source_chunks,
                "target_chunk_count": target_chunks,
                "estimated_workspace_bytes": estimated,
            }
        )
        evidence = DirectVortexResourceEvidence(
            sources,
            targets,
            source_chunk,
            target_chunk,
            source_chunks,
            target_chunks,
            sources * targets,
            self.maximum_interactions,
            estimated,
            self.maximum_workspace_bytes,
            True,
            evidence_id,
        )
        compatibility = VortexVelocityCompatibility(
            self.capabilities,
            source_capacity=sources,
            target_capacity=targets,
            source_kind=source_kind,
            target_topology=target_topology,
            requested_fields=request_fields(request),
        )
        return PreparedGaussianDirectVortex2D(self, evidence, compatibility)


class PreparedGaussianDirectVortex2D(AbstractPreparedVortexVelocity):
    """Prepared exact-capacity free-space Gaussian vortex summation."""

    plan: GaussianDirectVortexPlan2D
    resources: DirectVortexResourceEvidence
    capabilities: VortexVelocityCapabilities
    compatibility: VortexVelocityCompatibility
    dimension: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: GaussianDirectVortexPlan2D,
        resources: DirectVortexResourceEvidence,
        compatibility: VortexVelocityCompatibility,
        /,
    ):
        if not isinstance(plan, GaussianDirectVortexPlan2D):
            raise TypeError("plan must be a GaussianDirectVortexPlan2D.")
        if not isinstance(resources, DirectVortexResourceEvidence):
            raise TypeError("resources must be DirectVortexResourceEvidence.")
        if not isinstance(compatibility, VortexVelocityCompatibility):
            raise TypeError("compatibility must be VortexVelocityCompatibility.")
        self.plan = plan
        self.resources = resources
        self.capabilities = plan.capabilities
        self.compatibility = compatibility
        self.dimension = 2
        self.source_capacity = resources.source_capacity
        self.target_capacity = resources.target_capacity
        self.backend_id = canonical_fingerprint(
            {"kind": "gaussian-direct-vortex-backend-2d", "plan": plan.plan_id}
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-gaussian-direct-vortex-2d",
                "backend": self.backend_id,
                "resources": resources.evidence_id,
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
        source_position = self.plan.precision.compute(source.safe_positions())
        source_strength = self.plan.precision.compute(source.safe_strength())
        source_core = self.plan.precision.compute(source.safe_core_radius())
        target_position = self.plan.precision.compute(target.positions)
        target_identity = (
            -jnp.ones((self.target_capacity,), dtype=jnp.int32)
            if target.source_indices is None
            else target.source_indices
        )
        dtype = source_position.dtype

        source_chunk = self.resources.source_chunk_size
        target_chunk = self.resources.target_chunk_size
        padded_sources = self.resources.source_chunk_count * source_chunk
        padded_targets = self.resources.target_chunk_count * target_chunk
        source_padding = padded_sources - self.source_capacity
        target_padding = padded_targets - self.target_capacity
        source_position_padded = jnp.pad(source_position, ((0, source_padding), (0, 0)))
        source_strength_padded = jnp.pad(source_strength, ((0, source_padding),))
        source_core_padded = jnp.pad(
            source_core,
            ((0, source_padding),),
            constant_values=1.0,
        )
        source_active = jnp.pad(
            source.active_mask,
            ((0, source_padding),),
            constant_values=False,
        )
        source_capacity_mask = jnp.arange(padded_sources) < self.source_capacity
        source_indices = jnp.where(
            source_capacity_mask,
            jnp.arange(padded_sources, dtype=jnp.int32),
            jnp.asarray(-2, dtype=jnp.int32),
        )
        target_position_padded = jnp.pad(target_position, ((0, target_padding), (0, 0)))
        target_identity_padded = jnp.pad(
            target_identity,
            ((0, target_padding),),
            constant_values=-1,
        )
        target_valid = jnp.arange(padded_targets) < self.target_capacity

        source_xs = (
            source_position_padded.reshape((-1, source_chunk, 2)),
            source_strength_padded.reshape((-1, source_chunk)),
            source_core_padded.reshape((-1, source_chunk)),
            source_indices.reshape((-1, source_chunk)),
            source_active.reshape((-1, source_chunk)),
        )
        target_xs = (
            target_position_padded.reshape((-1, target_chunk, 2)),
            target_identity_padded.reshape((-1, target_chunk)),
            target_valid.reshape((-1, target_chunk)),
        )

        def target_body(
            carry: None,
            target_values: tuple[Array, Array, Array],
        ) -> tuple[None, tuple[Array, Array, Array, Array, Array]]:
            del carry
            target_block, identity_block, target_block_valid = target_values
            initial = (
                self.plan.precision.accumulation(
                    jnp.zeros((target_chunk, 2), dtype=dtype)
                ),
                self.plan.precision.accumulation(
                    jnp.zeros((target_chunk, 2, 2), dtype=dtype)
                ),
                self.plan.precision.accumulation(jnp.zeros((target_chunk,), dtype=dtype)),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )

            def source_body(
                sums: tuple[Array, Array, Array, Array, Array],
                source_values: tuple[Array, Array, Array, Array, Array],
            ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
                velocity_sum, gradient_sum, vorticity_sum, interactions, coincident = sums
                (
                    source_block,
                    strength_block,
                    core_block,
                    source_index_block,
                    source_block_valid,
                ) = source_values
                displacement = target_block[:, None, :] - source_block[None, :, :]
                pair_valid = (
                    target_block_valid[:, None]
                    & source_block_valid[None, :]
                    & (identity_block[:, None] != source_index_block[None, :])
                )
                pair_strength = jnp.broadcast_to(
                    strength_block[None, :], pair_valid.shape
                )
                pair_core = jnp.broadcast_to(core_block[None, :], pair_valid.shape)
                if request.velocity:
                    pair_velocity = gaussian_vortex_velocity_2d(
                        displacement,
                        pair_strength,
                        pair_core,
                    )
                    velocity_sum = velocity_sum + self.plan.precision.sum(
                        jnp.where(pair_valid[..., None], pair_velocity, 0.0),
                        axis=1,
                    )
                if request.velocity_gradient:
                    pair_gradient = gaussian_vortex_velocity_gradient_2d(
                        displacement,
                        pair_strength,
                        pair_core,
                    )
                    gradient_sum = gradient_sum + self.plan.precision.sum(
                        jnp.where(pair_valid[..., None, None], pair_gradient, 0.0),
                        axis=1,
                    )
                if request.vorticity:
                    pair_vorticity = gaussian_vortex_vorticity_2d(
                        displacement,
                        pair_strength,
                        pair_core,
                    )
                    vorticity_sum = vorticity_sum + self.plan.precision.sum(
                        jnp.where(pair_valid, pair_vorticity, 0.0),
                        axis=1,
                    )
                squared = jnp.sum(displacement * displacement, axis=-1)
                interactions = interactions + jnp.sum(pair_valid, dtype=jnp.int32)
                coincident = coincident + jnp.sum(
                    pair_valid & (squared == 0.0), dtype=jnp.int32
                )
                return (
                    velocity_sum,
                    gradient_sum,
                    vorticity_sum,
                    interactions,
                    coincident,
                ), None

            totals, _ = jax.lax.scan(source_body, initial, source_xs)
            return None, totals

        _, chunk_outputs = jax.lax.scan(target_body, None, target_xs)
        velocity_all = chunk_outputs[0].reshape((padded_targets, 2))[
            : self.target_capacity
        ]
        gradient_all = chunk_outputs[1].reshape((padded_targets, 2, 2))[
            : self.target_capacity
        ]
        vorticity_all = chunk_outputs[2].reshape((padded_targets,))[
            : self.target_capacity
        ]
        active_interactions = jnp.sum(chunk_outputs[3], dtype=jnp.int32)
        coincident_distinct = jnp.sum(chunk_outputs[4], dtype=jnp.int32)

        velocity = self.plan.precision.output(velocity_all) if request.velocity else None
        gradient = (
            self.plan.precision.output(gradient_all)
            if request.velocity_gradient
            else None
        )
        vorticity = (
            self.plan.precision.output(vorticity_all) if request.vorticity else None
        )
        outputs_finite = jnp.asarray(True)
        if velocity is not None:
            outputs_finite = outputs_finite & jnp.all(jnp.isfinite(velocity))
            velocity = eqx.error_if(
                velocity, ~outputs_finite, "Direct-vortex velocity is non-finite."
            )
        if gradient is not None:
            gradient_finite = jnp.all(jnp.isfinite(gradient))
            outputs_finite = outputs_finite & gradient_finite
            gradient = eqx.error_if(
                gradient, ~gradient_finite, "Direct-vortex gradient is non-finite."
            )
        if vorticity is not None:
            vorticity_finite = jnp.all(jnp.isfinite(vorticity))
            outputs_finite = outputs_finite & vorticity_finite
            vorticity = eqx.error_if(
                vorticity,
                ~vorticity_finite,
                "Direct-vortex vorticity is non-finite.",
            )

        excluded = jnp.sum(target_identity >= 0, dtype=jnp.int32)
        inputs_finite = (
            jnp.all(jnp.isfinite(source_position))
            & jnp.all(jnp.isfinite(source_strength))
            & jnp.all(jnp.isfinite(source_core))
            & jnp.all(source_core > 0.0)
            & jnp.all(jnp.isfinite(target_position))
        )
        resource_successful = jnp.asarray(self.resources.successful)
        successful = inputs_finite & outputs_finite & resource_successful
        backend_diagnostics = GaussianDirectDiagnostics2D(
            self.resources,
            jnp.asarray(self.resources.source_chunk_count, dtype=jnp.int32),
            jnp.asarray(self.resources.target_chunk_count, dtype=jnp.int32),
            jnp.asarray(source_chunk * target_chunk, dtype=jnp.int32),
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(self.source_capacity, dtype=jnp.int32),
            jnp.asarray(self.target_capacity, dtype=jnp.int32),
            active_interactions,
            excluded,
            coincident_distinct,
            jnp.min(jnp.where(source.active_mask, source_core, jnp.inf)),
            inputs_finite,
            outputs_finite,
            resource_successful,
            successful,
            backend_diagnostics,
        )
        evaluation_id = canonical_fingerprint(
            {
                "kind": "gaussian-direct-vortex-evaluation-2d",
                "prepared": self.prepared_id,
                "request": request.request_id,
                "source": source.source_id,
                "target": target.target_id,
                "target_topology": self.compatibility.target_topology,
                "identity_mode": (
                    "none" if target.source_indices is None else "explicit"
                ),
            }
        )
        return VortexVelocityEvaluation(
            velocity,
            gradient,
            vorticity,
            successful,
            self.backend_id,
            evaluation_id,
            diagnostics,
        )


__all__ = [
    "DirectVortexResourceEvidence",
    "GaussianDirectDiagnostics2D",
    "GaussianDirectVortexPlan2D",
    "PreparedGaussianDirectVortex2D",
]
