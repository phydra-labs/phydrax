#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization import (
    AbstractStructuredSplatAssignment,
    ParticleDiscretization,
    ParticleGridSplatPlan,
    PeriodicLerayProjector,
    PreparedParticleGridSplat,
    PreparedTensorGrid,
    SplatExecutionPolicy,
    TensorSpectralDiscretization,
)
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexVelocity,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)


class PeriodicVortexInCellDiagnostics(StrictModule):
    deposited_strength: Array
    compatibility_residual: Array
    compatibility_tolerance: Array
    balance_defect: Array
    partition_defect: Array
    divergence_norm: Array
    imaginary_leakage: Array
    transfer_successful: Array
    assignment_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    spectral_id: str = eqx.field(static=True)


class PeriodicVortexInCellPlan(AbstractVortexVelocityPlan):
    """Periodic particle-to-grid vorticity inversion with explicit filter identity."""

    particles: ParticleDiscretization
    grid: PreparedTensorGrid
    spectral: TensorSpectralDiscretization
    assignment: AbstractStructuredSplatAssignment
    execution: SplatExecutionPolicy
    compatibility_tolerance: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        grid: PreparedTensorGrid,
        spectral: TensorSpectralDiscretization,
        assignment: AbstractStructuredSplatAssignment,
        /,
        *,
        execution: SplatExecutionPolicy | None = None,
        compatibility_tolerance: float = 1.0e-12,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be PreparedTensorGrid.")
        if not isinstance(spectral, TensorSpectralDiscretization):
            raise TypeError("spectral must be TensorSpectralDiscretization.")
        if not isinstance(assignment, AbstractStructuredSplatAssignment):
            raise TypeError("assignment must be an AbstractStructuredSplatAssignment.")
        dimension = particles.ambient_dimension
        if dimension not in (2, 3) or len(grid.structured_axes) != dimension:
            raise ValueError("Periodic VIC requires matching dimension 2 or 3.")
        if len(spectral.axes) != dimension or any(
            axis.family != "fourier" for axis in spectral.axes
        ):
            raise ValueError("Periodic VIC requires an all-Fourier spectral space.")
        if any(not axis.periodic for axis in grid.structured_axes):
            raise ValueError("Periodic VIC requires periodic grid axes.")
        if tuple(grid.shape) != tuple(spectral.physical_shape):
            raise ValueError("VIC grid and spectral physical shapes differ.")
        for grid_axis, spectral_axis in zip(
            grid.structured_axes, spectral.axes, strict=True
        ):
            grid_nodes = np.asarray(grid_axis.point_coordinates)
            spectral_nodes = np.asarray(spectral_axis.nodes)
            if grid_nodes.shape != spectral_nodes.shape or not np.allclose(
                grid_nodes, spectral_nodes, rtol=0.0, atol=64 * np.finfo(float).eps
            ):
                raise ValueError("VIC grid and Fourier nodes must coincide exactly.")
        tolerance = float(compatibility_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("compatibility_tolerance must be finite and positive.")
        execution_ = SplatExecutionPolicy() if execution is None else execution
        if not isinstance(execution_, SplatExecutionPolicy):
            raise TypeError("execution must be SplatExecutionPolicy or None.")
        self.particles = particles
        self.grid = grid
        self.spectral = spectral
        self.assignment = assignment
        self.execution = execution_
        self.compatibility_tolerance = tolerance
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-vortex-in-cell-plan",
                "particles": particles.prepared_id,
                "grid": grid.prepared_id,
                "spectral": spectral.prepared_id,
                "assignment": assignment.assignment_id,
                "execution": execution_.policy_id,
                "compatibility_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
    ) -> PreparedPeriodicVortexInCell:
        targets = (
            int(source_capacity) if target_capacity is None else int(target_capacity)
        )
        if (
            int(source_capacity) != self.particles.capacity
            or targets != self.particles.capacity
        ):
            raise ValueError(
                "Periodic VIC currently evaluates its bound particle support."
            )
        transfer = ParticleGridSplatPlan(
            self.grid,
            assignment=self.assignment,
            execution=self.execution,
        ).prepare(self.particles)
        return PreparedPeriodicVortexInCell(self, transfer)


class PreparedPeriodicVortexInCell(AbstractPreparedVortexVelocity):
    plan: PeriodicVortexInCellPlan
    transfer: PreparedParticleGridSplat
    projector: PeriodicLerayProjector
    dimension: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: PeriodicVortexInCellPlan, transfer: PreparedParticleGridSplat, /
    ):
        self.plan = plan
        self.transfer = transfer
        self.projector = PeriodicLerayProjector(plan.spectral)
        self.dimension = plan.dimension
        self.source_capacity = plan.particles.capacity
        self.target_capacity = plan.particles.capacity
        self.backend_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-vortex-in-cell",
                "plan": plan.plan_id,
                "transfer": transfer.prepared_id,
                "projector": self.projector.projector_id,
            }
        )

    def _velocity_coefficients(self, vorticity_coefficients: Array, /) -> Array:
        inverse = self.projector.inverse_wavenumber_squared.astype(
            vorticity_coefficients.real.dtype
        )
        waves = tuple(
            wave.astype(vorticity_coefficients.dtype)
            for wave in self.projector.wavenumbers
        )
        if self.dimension == 2:
            return (
                jnp.stack(
                    (
                        1j * waves[1] * inverse * vorticity_coefficients,
                        -1j * waves[0] * inverse * vorticity_coefficients,
                    ),
                    axis=-1,
                )
                * self.projector.admissibility_mask[..., None]
            )
        wavevector = jnp.stack(waves, axis=-1)
        return (
            1j
            * jnp.cross(wavevector, vorticity_coefficients, axis=-1)
            * inverse[..., None]
            * self.projector.admissibility_mask[..., None]
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
        if targets is not None or target_source_indices is not None:
            raise ValueError("Periodic VIC evaluates only its bound particle support.")
        positions = jnp.asarray(position)
        strengths = jnp.asarray(strength)
        cores = jnp.asarray(core_radius)
        expected_strength = (
            (self.source_capacity,) if self.dimension == 2 else (self.source_capacity, 3)
        )
        if positions.shape != (self.source_capacity, self.dimension):
            raise ValueError("VIC positions do not match prepared support.")
        if strengths.shape != expected_strength:
            raise ValueError("VIC strengths do not match prepared dimension.")
        if cores.shape != (self.source_capacity,):
            raise ValueError("VIC core_radius must have particle-capacity shape.")
        active = self.plan.particles.active_mask
        strength_mask = active if self.dimension == 2 else active[:, None]
        safe_positions = jnp.where(active[:, None], positions, 0.0)
        safe_strengths = jnp.where(strength_mask, strengths, 0.0)
        transfer_state = self.transfer.build(safe_positions)
        deposited = self.transfer.deposit_content(transfer_state, safe_strengths)
        total = jnp.sum(safe_strengths, axis=0)
        scale = jnp.maximum(jnp.sum(jnp.abs(safe_strengths), axis=0), 1.0)
        residual = jnp.max(jnp.abs(total) / scale)
        compatible = residual <= self.plan.compatibility_tolerance
        safe_density = eqx.error_if(
            deposited.density,
            ~compatible,
            "Periodic vortex velocity requires zero total integrated vorticity.",
        )
        vorticity_coefficients = self.plan.spectral.project(safe_density)
        velocity_coefficients = self._velocity_coefficients(vorticity_coefficients)
        velocity_grid = self.plan.spectral.reconstruct(velocity_coefficients).real
        gathered_velocity = self.transfer.gather(transfer_state, velocity_grid)
        velocity = gathered_velocity.values if request.velocity else None
        gradient = None
        if request.velocity_gradient:
            derivatives = tuple(
                self.plan.spectral.reconstruct(
                    self.plan.spectral.modal_derivative(velocity_coefficients, axis=axis)
                ).real
                for axis in range(self.dimension)
            )
            gradient = jnp.stack(
                tuple(
                    self.transfer.gather(transfer_state, derivative).values
                    for derivative in derivatives
                ),
                axis=-1,
            )
        vorticity = (
            self.transfer.gather(transfer_state, safe_density).values
            if request.vorticity
            else None
        )
        divergence_coefficients = jnp.zeros(
            self.plan.spectral.modal_shape, dtype=velocity_coefficients.dtype
        )
        for axis, wave in enumerate(self.projector.wavenumbers):
            divergence_coefficients = (
                divergence_coefficients + 1j * wave * velocity_coefficients[..., axis]
            )
        divergence_norm = jnp.sqrt(jnp.sum(jnp.abs(divergence_coefficients) ** 2))
        outputs_finite = jnp.asarray(True)
        for value in (velocity, gradient, vorticity):
            if value is not None:
                outputs_finite = outputs_finite & jnp.all(jnp.isfinite(value))
        transfer_successful = (
            transfer_state.successful
            & deposited.successful
            & jnp.all(gathered_velocity.support)
        )
        successful = compatible & transfer_successful & outputs_finite
        backend = PeriodicVortexInCellDiagnostics(
            total,
            residual,
            jnp.asarray(self.plan.compatibility_tolerance, dtype=positions.dtype),
            deposited.balance.maximum_absolute_balance_defect,
            deposited.balance.maximum_partition_defect,
            divergence_norm,
            self.plan.spectral.imaginary_leakage(velocity_coefficients),
            transfer_successful,
            self.plan.assignment.assignment_id,
            self.plan.grid.prepared_id,
            self.plan.spectral.prepared_id,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(self.source_capacity, dtype=jnp.int32),
            jnp.asarray(self.target_capacity, dtype=jnp.int32),
            jnp.asarray(self.transfer.route_count, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.min(jnp.where(active, cores, jnp.inf)),
            jnp.all(jnp.isfinite(safe_positions)) & jnp.all(jnp.isfinite(safe_strengths)),
            outputs_finite,
            transfer_successful,
            successful,
            backend,
        )
        return VortexVelocityEvaluation(
            velocity,
            gradient,
            vorticity,
            successful,
            self.backend_id,
            canonical_fingerprint(
                {
                    "kind": "periodic-vortex-in-cell-evaluation",
                    "prepared": self.prepared_id,
                    "request": request.request_id,
                }
            ),
            diagnostics,
        )


__all__ = [
    "PeriodicVortexInCellDiagnostics",
    "PeriodicVortexInCellPlan",
    "PreparedPeriodicVortexInCell",
]
