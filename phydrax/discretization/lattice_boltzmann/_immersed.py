#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._discretization import LatticeBoltzmannDiscretization


class ImmersedBoundaryForceLedger(StrictModule):
    """Equal-and-opposite fluid/body load, torque, and work accounting."""

    fluid_force: Array
    body_force: Array
    body_torque: Array
    body_work: Array
    force_balance_residual: Array


class ImmersedBoundaryForcingEvidence(StrictModule):
    """Regularized interpolation, partition, and direct-forcing residuals."""

    interpolated_velocity: Array
    target_velocity: Array
    velocity_residual: Array
    maximum_velocity_residual: Array
    partition_of_unity_residual: Array
    force_balance_residual: Array
    iteration_count: Array
    converged: Array
    successful: Array


class ImmersedBoundaryForcingResult(StrictModule):
    force_density: Array
    marker_force: Array
    marker_acceleration: Array
    ledger: ImmersedBoundaryForceLedger
    evidence: ImmersedBoundaryForcingEvidence


class ImmersedBoundaryForcingPlan(StrictModule, NonTrainableState):
    """Fixed-iteration regularized direct forcing for Lagrangian markers.

    Marker measures convert the interpolated acceleration correction into a total
    marker force. Spreading divides by Cartesian cell measure, so the integrated
    Eulerian force is exactly equal and opposite to the reported body load.
    """

    discretization: LatticeBoltzmannDiscretization
    iteration_count: int = eqx.field(static=True)
    kernel_radius: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        /,
        *,
        iteration_count: int = 4,
        kernel_radius: float = 2.0,
        convergence_tolerance: float = 1.0e-6,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
        iterations = int(iteration_count)
        radius = float(kernel_radius)
        tolerance = float(convergence_tolerance)
        if iterations < 1:
            raise ValueError("iteration_count must be positive.")
        if not np.isfinite(radius) or radius <= 1.0:
            raise ValueError("kernel_radius must be finite and greater than one.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be finite and positive.")
        self.discretization = discretization
        self.iteration_count = iterations
        self.kernel_radius = radius
        self.convergence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "immersed-boundary-lattice-boltzmann-forcing",
                "discretization": discretization.prepared_id,
                "iteration_count": iterations,
                "kernel_radius": radius,
                "convergence_tolerance": tolerance,
            }
        )

    def _weights(
        self,
        marker_positions: Array,
        fluid_mask: Array,
        /,
    ) -> tuple[Array, Array]:
        coordinates = self.discretization.grid.points.astype(marker_positions.dtype)
        difference = coordinates[:, None, :] - marker_positions[None, :, :]
        cell_size = jnp.asarray(
            self.discretization.cell_size, dtype=marker_positions.dtype
        )
        for axis, periodic in enumerate(self.discretization.periodic):
            if periodic:
                length = cell_size * self.discretization.grid.shape[axis]
                component = difference[..., axis]
                difference = difference.at[..., axis].set(
                    component - jnp.round(component / length) * length
                )
        scaled = jnp.abs(difference / cell_size)
        kernel = jnp.where(
            scaled < self.kernel_radius,
            0.5
            / self.kernel_radius
            * (1.0 + jnp.cos(jnp.pi * scaled / self.kernel_radius)),
            0.0,
        )
        raw = jnp.prod(kernel, axis=-1) * fluid_mask.reshape((-1, 1))
        partition = jnp.sum(raw, axis=0)
        partition = eqx.error_if(
            partition,
            jnp.any(partition <= 0.0),
            "Every immersed marker must overlap at least one active fluid cell.",
        )
        weights = raw / partition[None, :]
        return weights, jnp.max(jnp.abs(jnp.sum(weights, axis=0) - 1.0))

    def apply(
        self,
        fluid_velocity: ArrayLike,
        density: ArrayLike,
        marker_positions: ArrayLike,
        target_velocity: ArrayLike,
        marker_measures: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        fluid_mask: ArrayLike | None = None,
        body_indices: ArrayLike | None = None,
        body_centers: ArrayLike | None = None,
    ) -> ImmersedBoundaryForcingResult:
        dimension = self.discretization.velocity_set.dimension
        grid_shape = self.discretization.grid.shape
        velocity = jnp.asarray(fluid_velocity)
        rho = jnp.asarray(density, dtype=velocity.dtype)
        positions = jnp.asarray(marker_positions, dtype=velocity.dtype)
        target = jnp.asarray(target_velocity, dtype=velocity.dtype)
        measures = jnp.asarray(marker_measures, dtype=velocity.dtype)
        dt = jnp.asarray(time_step, dtype=velocity.dtype)
        if velocity.shape != grid_shape + (dimension,):
            raise ValueError("fluid_velocity must match grid shape and dimension.")
        if rho.shape != grid_shape:
            raise ValueError("density must match grid shape.")
        if positions.ndim != 2 or positions.shape[1] != dimension:
            raise ValueError("marker_positions must have shape (marker, dimension).")
        marker_count = positions.shape[0]
        if marker_count == 0:
            raise ValueError("At least one immersed marker is required.")
        if target.shape != positions.shape or measures.shape != (marker_count,):
            raise ValueError(
                "Marker target velocities and measures have incompatible shapes."
            )
        if fluid_mask is None:
            mask = jnp.ones(grid_shape, dtype=bool)
        else:
            mask = jnp.asarray(fluid_mask, dtype=bool)
            if mask.shape != grid_shape:
                raise ValueError("fluid_mask must match grid shape.")
        velocity = eqx.error_if(
            velocity,
            jnp.any(~jnp.isfinite(velocity)),
            "fluid_velocity must be finite.",
        )
        rho = eqx.error_if(
            rho,
            jnp.any(mask & (~jnp.isfinite(rho) | (rho <= 0.0))),
            "density must be finite and positive on active fluid cells.",
        )
        positions = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)) | jnp.any(~jnp.isfinite(target)),
            "Immersed marker positions and target velocities must be finite.",
        )
        if (body_indices is None) != (body_centers is None):
            raise ValueError("body_indices and body_centers must be supplied together.")
        if body_indices is None:
            indices = jnp.zeros((marker_count,), dtype=jnp.int32)
            centers = jnp.zeros((1, dimension), dtype=velocity.dtype)
        else:
            indices = jnp.asarray(body_indices, dtype=jnp.int32)
            centers = jnp.asarray(body_centers, dtype=velocity.dtype)
            if indices.shape != (marker_count,) or centers.ndim != 2:
                raise ValueError("Body indices or centers have incompatible shapes.")
            if centers.shape[1] != dimension or centers.shape[0] == 0:
                raise ValueError("body_centers must have shape (body, dimension).")
            indices = eqx.error_if(
                indices,
                jnp.any(indices < 0) | jnp.any(indices >= centers.shape[0]),
                "Every marker body index must name a prepared body.",
            )
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0),
            "Immersed-boundary time_step must be finite and positive.",
        )
        measures = eqx.error_if(
            measures,
            jnp.any(~jnp.isfinite(measures)) | jnp.any(measures <= 0.0),
            "marker_measures must be finite and positive.",
        )

        weights, partition_residual = self._weights(positions, mask)
        flat_velocity = velocity.reshape((-1, dimension))
        flat_density = rho.reshape((-1,))
        marker_density = oe.contract("nm,n->m", weights, flat_density)
        cell_measure = jnp.asarray(
            self.discretization.cell_size**dimension, dtype=velocity.dtype
        )
        force_density = jnp.zeros_like(flat_velocity)
        marker_force = jnp.zeros_like(target)
        marker_acceleration = jnp.zeros_like(target)
        corrected_velocity = flat_velocity
        for _ in range(self.iteration_count):
            interpolated = oe.contract("nm,nd->md", weights, corrected_velocity)
            marker_acceleration = (target - interpolated) / dt
            increment = marker_density[:, None] * measures[:, None] * marker_acceleration
            marker_force = marker_force + increment
            spread = oe.contract("nm,md->nd", weights, increment) / cell_measure
            force_density = force_density + spread
            corrected_velocity = corrected_velocity + dt * spread / jnp.maximum(
                flat_density[:, None], jnp.finfo(velocity.dtype).tiny
            )

        interpolated = oe.contract("nm,nd->md", weights, corrected_velocity)
        velocity_residual = interpolated - target
        maximum_residual = jnp.max(jnp.abs(velocity_residual))
        body_count = centers.shape[0]
        membership = indices[:, None] == jnp.arange(body_count)[None, :]
        body_force = -oe.contract(
            "mb,md->bd", membership.astype(velocity.dtype), marker_force
        )
        radius = positions - centers[indices]
        if dimension == 2:
            marker_torque = (
                radius[:, 0] * (-marker_force[:, 1])
                - radius[:, 1] * (-marker_force[:, 0])
            )[:, None]
        else:
            negative_force = -marker_force
            marker_torque = jnp.stack(
                (
                    radius[:, 1] * negative_force[:, 2]
                    - radius[:, 2] * negative_force[:, 1],
                    radius[:, 2] * negative_force[:, 0]
                    - radius[:, 0] * negative_force[:, 2],
                    radius[:, 0] * negative_force[:, 1]
                    - radius[:, 1] * negative_force[:, 0],
                ),
                axis=-1,
            )
        body_torque = oe.contract(
            "mb,ma->ba", membership.astype(velocity.dtype), marker_torque
        )
        marker_work = -oe.contract("md,md->m", marker_force, target)
        body_work = oe.contract("mb,m->b", membership.astype(velocity.dtype), marker_work)
        grid_force = jnp.sum(force_density, axis=0) * cell_measure
        force_balance = grid_force + jnp.sum(body_force, axis=0)
        force_balance_residual = jnp.sqrt(jnp.sum(force_balance**2))
        finite = (
            jnp.all(jnp.isfinite(force_density))
            & jnp.all(jnp.isfinite(marker_force))
            & jnp.all(jnp.isfinite(velocity_residual))
        )
        convergence = maximum_residual <= self.convergence_tolerance
        balance_tolerance = (
            128.0
            * jnp.finfo(velocity.dtype).eps
            * jnp.maximum(jnp.sqrt(jnp.sum(grid_force**2)), 1.0)
        )
        successful = finite & (force_balance_residual <= balance_tolerance)
        ledger = ImmersedBoundaryForceLedger(
            grid_force,
            body_force,
            body_torque,
            body_work,
            force_balance_residual,
        )
        evidence = ImmersedBoundaryForcingEvidence(
            interpolated,
            target,
            velocity_residual,
            maximum_residual,
            partition_residual,
            force_balance_residual,
            jnp.asarray(self.iteration_count, dtype=jnp.int32),
            convergence,
            successful,
        )
        return ImmersedBoundaryForcingResult(
            force_density.reshape(grid_shape + (dimension,)),
            marker_force,
            marker_acceleration,
            ledger,
            evidence,
        )


__all__ = [
    "ImmersedBoundaryForceLedger",
    "ImmersedBoundaryForcingEvidence",
    "ImmersedBoundaryForcingPlan",
    "ImmersedBoundaryForcingResult",
]
