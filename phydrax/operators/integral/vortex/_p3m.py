#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._interpolation import fourier_interpolate
from ...._strict import StrictModule
from ....discretization.vortex._interfaces import (
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ....discretization.vortex._source import VortexSourceState, VortexTargetState
from ._gaussian2d import gaussian_vortex_velocity_2d
from ._gaussian3d import GaussianErfVortexKernel3D
from ._particle_mesh import PreparedPeriodicVortexInCell


class CorrectedP3MEvidence(StrictModule):
    near_pair_count: Array
    mesh_route_count: Array
    splitting_parameter: Array
    cutoff_radius: Array
    assignment_defect: Array
    spectral_defect: Array
    core_correction_norm: Array
    cutoff_tail_bound: Array
    finite: Array


class CorrectedP3MPlan(StrictModule):
    mesh: PreparedPeriodicVortexInCell
    splitting_parameter: float = eqx.field(static=True)
    cutoff_radius: float = eqx.field(static=True)
    interpolation_method: str = eqx.field(static=True)
    interpolation_tolerance: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: PreparedPeriodicVortexInCell,
        splitting_parameter: float,
        cutoff_radius: float,
        /,
        *,
        interpolation_method: str = "direct",
        interpolation_tolerance: float | None = None,
    ):
        if (
            not isinstance(mesh, PreparedPeriodicVortexInCell)
            or float(splitting_parameter) <= 0.0
            or float(cutoff_radius) <= 0.0
            or interpolation_method not in ("direct", "nufft")
        ):
            raise ValueError(
                "P3M mesh/splitting/cutoff/interpolation controls are invalid."
            )
        periods = jnp.asarray(
            tuple(
                axis.bounds[1] - axis.bounds[0] for axis in mesh.plan.grid.structured_axes
            )
        )
        if float(cutoff_radius) >= 0.5 * float(jnp.min(periods)):
            raise ValueError("P3M cutoff must be less than half every period.")
        if interpolation_method == "nufft" and interpolation_tolerance is None:
            raise ValueError("P3M NUFFT interpolation requires tolerance.")
        self.mesh, self.splitting_parameter, self.cutoff_radius = (
            mesh,
            float(splitting_parameter),
            float(cutoff_radius),
        )
        self.interpolation_method, self.interpolation_tolerance = (
            interpolation_method,
            interpolation_tolerance,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "corrected-p3m-plan",
                "mesh": mesh.prepared_id,
                "splitting_parameter": self.splitting_parameter,
                "cutoff_radius": self.cutoff_radius,
                "interpolation_method": interpolation_method,
                "interpolation_tolerance": interpolation_tolerance,
            }
        )

    def _far_grid(self, source: VortexSourceState, /):
        transfer_state = self.mesh.transfer.build(source.safe_positions())
        deposited = self.mesh.transfer.deposit_content(
            transfer_state, source.safe_strength()
        )
        coefficients = self.mesh.plan.spectral.project(deposited.density)
        squared_wave = jnp.zeros(
            self.mesh.plan.spectral.modal_shape, dtype=coefficients.real.dtype
        )
        for wave in self.mesh.projector.wavenumbers:
            squared_wave = squared_wave + wave**2
        filter_ = jnp.exp(-squared_wave / (4.0 * self.splitting_parameter**2))
        velocity_coefficients = self.mesh._velocity_coefficients(
            coefficients
            * filter_.reshape(
                filter_.shape + (1,) * (coefficients.ndim - self.mesh.dimension)
            )
        )
        velocity_grid = self.mesh.plan.spectral.reconstruct(velocity_coefficients).real
        return transfer_state, deposited, velocity_coefficients, velocity_grid

    def _far_targets(
        self, transfer_state, velocity_grid: Array, target: VortexTargetState, /
    ) -> Array:
        same = (
            target.source_indices is not None
            and target.capacity == self.mesh.source_capacity
        )
        if same:
            return self.mesh.transfer.gather(transfer_state, velocity_grid).values
        result = fourier_interpolate(
            velocity_grid,
            target.positions,
            spatial_ndim=self.mesh.dimension,
            payload_ndim=1,
            axis_nodes=tuple(axis.nodes for axis in self.mesh.plan.spectral.axes),
            periods=tuple(
                axis.bounds[1] - axis.bounds[0]
                for axis in self.mesh.plan.grid.structured_axes
            ),
            method=self.interpolation_method,
            tolerance=self.interpolation_tolerance,
        )
        return result.values

    def _near_correction(
        self, source: VortexSourceState, target: VortexTargetState, /
    ) -> tuple[Array, Array, Array]:
        if source.core_radius is None:
            raise ValueError("Corrected P3M requires source core radii.")
        displacement = target.positions[:, None, :] - source.safe_positions()[None, :, :]
        periods = jnp.asarray(
            tuple(
                axis.bounds[1] - axis.bounds[0]
                for axis in self.mesh.plan.grid.structured_axes
            ),
            dtype=displacement.dtype,
        )
        displacement = displacement - periods * jnp.round(displacement / periods)
        squared = jnp.sum(displacement**2, axis=-1)
        active = source.active_mask[None, :] & (squared < self.cutoff_radius**2)
        if target.source_indices is not None:
            active = active & (
                target.source_indices[:, None]
                != jnp.arange(source.capacity, dtype=jnp.int32)[None, :]
            )
        safe_squared = jnp.where(
            active, jnp.maximum(squared, jnp.finfo(displacement.dtype).tiny), 1.0
        )
        alpha = jnp.asarray(self.splitting_parameter, dtype=displacement.dtype)
        if source.dimension == 2:
            pair_shape = squared.shape
            singular = source.safe_strength()[None, :] / (2.0 * jnp.pi * safe_squared)
            singular_velocity = singular[..., None] * jnp.stack(
                (-displacement[..., 1], displacement[..., 0]), axis=-1
            )
            screened = jnp.exp(-(alpha**2) * safe_squared)[..., None] * singular_velocity
            core_velocity = gaussian_vortex_velocity_2d(
                displacement,
                jnp.broadcast_to(source.safe_strength()[None, :], pair_shape),
                jnp.broadcast_to(source.safe_core_radius()[None, :], pair_shape),
            )
        else:
            radius = jnp.sqrt(safe_squared)
            singular_velocity = jnp.cross(
                source.safe_strength()[None, :, :], displacement
            ) / (4.0 * jnp.pi * safe_squared[..., None] * radius[..., None])
            screen = jax.scipy.special.erfc(
                alpha * radius
            ) + 2.0 * alpha * radius / jnp.sqrt(jnp.pi) * jnp.exp(
                -(alpha**2) * safe_squared
            )
            screened = screen[..., None] * singular_velocity
            kernel = GaussianErfVortexKernel3D()
            core_velocity = kernel.evaluate(
                displacement,
                jnp.broadcast_to(source.safe_strength()[None, :, :], displacement.shape),
                jnp.broadcast_to(source.safe_core_radius()[None, :], squared.shape),
            ).velocity
        correction = screened + core_velocity - singular_velocity
        correction = jnp.where(active[..., None], correction, 0.0)
        return (
            jnp.sum(correction, axis=1),
            jnp.sum(active, dtype=jnp.int32),
            jnp.linalg.norm(correction),
        )

    def evaluate(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        if (
            source.dimension != self.mesh.dimension
            or source.capacity != self.mesh.source_capacity
            or target.dimension != source.dimension
        ):
            raise ValueError(
                "P3M source/target dimensions or source capacity are incompatible."
            )
        transfer_state, deposited, _, velocity_grid = self._far_grid(source)
        far = self._far_targets(transfer_state, velocity_grid, target)
        correction, pair_count, correction_norm = self._near_correction(source, target)
        velocity_all = far + correction
        gradient_all = None
        if request.velocity_gradient or request.vorticity:
            gradient_all = jax.vmap(
                jax.jacfwd(
                    lambda point: self.evaluate(
                        source,
                        VortexTargetState(point[None, :]),
                        request=VortexFieldRequest(velocity=True),
                    ).velocity[0]
                )
            )(target.positions)
        if request.vorticity:
            if source.dimension == 2:
                vorticity = gradient_all[:, 1, 0] - gradient_all[:, 0, 1]
            else:
                vorticity = jnp.stack(
                    (
                        gradient_all[:, 2, 1] - gradient_all[:, 1, 2],
                        gradient_all[:, 0, 2] - gradient_all[:, 2, 0],
                        gradient_all[:, 1, 0] - gradient_all[:, 0, 1],
                    ),
                    axis=-1,
                )
        else:
            vorticity = None
        finite = jnp.all(jnp.isfinite(velocity_all))
        cutoff_tail = jnp.exp(-((self.splitting_parameter * self.cutoff_radius) ** 2))
        backend = CorrectedP3MEvidence(
            pair_count,
            jnp.asarray(self.mesh.transfer.route_count, dtype=jnp.int32),
            jnp.asarray(self.splitting_parameter),
            jnp.asarray(self.cutoff_radius),
            deposited.balance.maximum_partition_defect,
            self.mesh.plan.spectral.imaginary_leakage(
                self.mesh.plan.spectral.project(velocity_grid)
            ),
            correction_norm,
            cutoff_tail,
            finite,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
            pair_count + self.mesh.transfer.route_count,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.min(source.safe_core_radius()),
            jnp.asarray(True),
            finite,
            transfer_state.successful,
            finite & transfer_state.successful,
            backend,
        )
        return VortexVelocityEvaluation(
            velocity_all if request.velocity else None,
            gradient_all if request.velocity_gradient else None,
            vorticity,
            finite & transfer_state.successful,
            self.plan_id,
            canonical_fingerprint(
                {
                    "kind": "corrected-p3m-evaluation",
                    "plan": self.plan_id,
                    "request": request.request_id,
                    "target_count": target.capacity,
                }
            ),
            diagnostics,
        )


__all__ = ["CorrectedP3MEvidence", "CorrectedP3MPlan"]
