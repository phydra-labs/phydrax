#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.lattice_boltzmann._discretization import (
    LatticeBoltzmannDiscretization,
)
from ....linalg import ArraySpace, SmallLinearSolvePlan, solve_small_linear
from ....nonlinear import FixedPointIteration, NonlinearTermination
from ....solver._partitioned_coupling_graph import (
    CouplingGraph,
    prepare_coupling,
    PreparedCoupling,
)
from ....solver._partitioned_coupling_types import (
    CallableCouplingSubsystem,
    CouplingDifferentiationPolicy,
    CouplingExchange,
    CouplingPort,
    CouplingSubsystemCapabilities,
    CouplingSubsystemResult,
    CouplingSweep,
    CouplingTolerance,
    ImplicitCouplingPolicy,
)


class SparseMarkerRelationEvidence(StrictModule):
    """Coverage and reproducing-moment evidence for one frozen sparse relation."""

    coverage_fraction: Array
    active_count: Array
    valid_route_count: Array
    partition_residual: Array
    first_moment_residual: Array
    minimum_route_weight: Array
    covered: Array
    local_solve_successful: Array
    finite: Array
    successful: Array


class SparseMarkerRelation(StrictModule):
    """Runtime weights on host-prepared, fixed-width marker-to-cell routes."""

    cell_indices: Array
    route_positions: Array
    weights: Array
    valid: Array
    active: Array
    marker_position: Array
    evidence: SparseMarkerRelationEvidence
    relation_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class SparseMarkerTransposeEvidence(StrictModule):
    """Integrated force, torque, and power identities of a sparse transpose."""

    marker_resultant: Array
    grid_resultant: Array
    force_residual: Array
    marker_torque: Array
    grid_torque: Array
    torque_residual: Array
    torque_origin: Array
    interpolation_power: Array
    spreading_power: Array
    transpose_power_residual: Array
    body_force: Array
    body_torque: Array
    body_power: Array
    interface_power_residual: Array
    tolerance: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class SparseMarkerTransferPlan(StrictModule, NonTrainableState):
    """Plan a bounded-memory sparse LBM marker transfer.

    Preparation performs the discrete cell search on the host. Runtime actions only
    gather or scatter ``capacity * stencil_width**dimension`` entries; no cell-by-
    marker matrix is formed or retained.
    """

    discretization: LatticeBoltzmannDiscretization
    marker_ids: Array
    local_solve: SmallLinearSolvePlan
    capacity: int = eqx.field(static=True)
    stencil_width: int = eqx.field(static=True)
    route_width: int = eqx.field(static=True)
    minimum_coverage: float = eqx.field(static=True)
    relation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        marker_ids: ArrayLike,
        /,
        *,
        stencil_width: int = 4,
        minimum_coverage: float = 0.5,
        maximum_resource_bytes: int = 1024**3,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
        identifiers = np.asarray(marker_ids)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("marker_ids must be one non-empty vector.")
        if not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("marker_ids must contain integers.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("marker_ids must be unique non-negative stable IDs.")
        width = int(stencil_width)
        if width < 2 or width % 2 != 0:
            raise ValueError("stencil_width must be an even integer of at least two.")
        coverage = float(minimum_coverage)
        if not np.isfinite(coverage) or not 0.0 < coverage <= 1.0:
            raise ValueError("minimum_coverage must lie in (0, 1].")
        dimension = discretization.velocity_set.dimension
        capacity = int(identifiers.size)
        route_width = width**dimension
        itemsize = np.dtype(discretization.velocity_space.vector_space.dtype).itemsize
        relation_bytes = (
            capacity
            * route_width
            * (
                np.dtype(np.int32).itemsize
                + np.dtype(np.bool_).itemsize
                + (dimension + 1) * itemsize
            )
        )
        workspace_bytes = int(np.prod(discretization.grid.shape)) * dimension * itemsize
        resource_limit = int(maximum_resource_bytes)
        if resource_limit <= 0 or relation_bytes + workspace_bytes > resource_limit:
            raise ValueError("Sparse marker transfer exceeds its resource budget.")
        self.discretization = discretization
        self.marker_ids = jnp.asarray(identifiers, dtype=jnp.int32)
        self.local_solve = SmallLinearSolvePlan(dimension)
        self.capacity = capacity
        self.stencil_width = width
        self.route_width = route_width
        self.minimum_coverage = coverage
        self.relation_bytes = relation_bytes
        self.workspace_bytes = workspace_bytes
        self.maximum_resource_bytes = resource_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-sparse-lbm-marker-transfer",
                "discretization": discretization.prepared_id,
                "marker_ids": identifiers.astype(np.int64).tolist(),
                "stencil_width": width,
                "minimum_coverage": coverage,
                "resource_limit": resource_limit,
            }
        )

    def prepare(
        self,
        marker_position: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
    ) -> PreparedSparseMarkerTransfer:
        position = np.asarray(marker_position, dtype=float)
        dimension = self.discretization.velocity_set.dimension
        if position.shape != (self.capacity, dimension):
            raise ValueError(
                f"marker_position must have shape {(self.capacity, dimension)}."
            )
        if np.any(~np.isfinite(position)):
            raise ValueError("Prepared marker positions must be finite.")
        active_mask = (
            np.ones((self.capacity,), dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if active_mask.shape != (self.capacity,) or not np.any(active_mask):
            raise ValueError("active must select at least one marker within capacity.")

        grid_shape = tuple(int(value) for value in self.discretization.grid.shape)
        spacing = float(np.asarray(self.discretization.cell_size))
        axes = self.discretization.grid.structured_axes
        lower = np.asarray([float(axis.bounds[0]) for axis in axes])
        periodic = tuple(bool(axis.periodic) for axis in axes)
        offsets = np.stack(
            np.meshgrid(
                *(np.arange(self.stencil_width) for _ in range(dimension)),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, dimension))
        indices = np.zeros((self.capacity, self.route_width), dtype=np.int32)
        route_positions = np.zeros(
            (self.capacity, self.route_width, dimension), dtype=position.dtype
        )
        valid = np.zeros((self.capacity, self.route_width), dtype=bool)
        host_coverage = np.zeros((self.capacity,), dtype=position.dtype)
        for marker in range(self.capacity):
            coordinate = (position[marker] - lower) / spacing - 0.5
            start = np.floor(coordinate).astype(np.int64) - self.stencil_width // 2 + 1
            raw = start[None, :] + offsets
            wrapped = raw.copy()
            route_valid = np.ones((self.route_width,), dtype=bool)
            for axis, count in enumerate(grid_shape):
                if periodic[axis]:
                    wrapped[:, axis] = np.mod(wrapped[:, axis], count)
                else:
                    route_valid &= (raw[:, axis] >= 0) & (raw[:, axis] < count)
                    wrapped[:, axis] = np.clip(wrapped[:, axis], 0, count - 1)
            indices[marker] = np.ravel_multi_index(tuple(wrapped.T), grid_shape)
            route_positions[marker] = lower + (raw + 0.5) * spacing
            valid[marker] = route_valid & active_mask[marker]
            host_coverage[marker] = np.count_nonzero(valid[marker]) / self.route_width
        return PreparedSparseMarkerTransfer(
            self,
            jnp.asarray(position),
            jnp.asarray(active_mask),
            jnp.asarray(indices),
            jnp.asarray(route_positions),
            jnp.asarray(valid),
            jnp.asarray(host_coverage),
        )


class PreparedSparseMarkerTransfer(StrictModule, NonTrainableState):
    """Prepared cell routes with differentiable weights and fixed topology."""

    plan: SparseMarkerTransferPlan
    initial_marker_position: Array
    active: Array
    cell_indices: Array
    route_positions: Array
    valid: Array
    host_coverage_fraction: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SparseMarkerTransferPlan,
        initial_marker_position: Array,
        active: Array,
        cell_indices: Array,
        route_positions: Array,
        valid: Array,
        host_coverage_fraction: Array,
        /,
    ):
        self.plan = plan
        self.initial_marker_position = initial_marker_position
        self.active = active
        self.cell_indices = jax.lax.stop_gradient(cell_indices)
        self.route_positions = route_positions
        self.valid = jax.lax.stop_gradient(valid)
        self.host_coverage_fraction = host_coverage_fraction
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardio-sparse-marker-transfer",
                "plan": plan.plan_id,
                "indices": array_tree_fingerprint(np.asarray(cell_indices)),
                "valid": array_tree_fingerprint(np.asarray(valid)),
            }
        )

    @property
    def capacity(self) -> int:
        return self.plan.capacity

    @property
    def dimension(self) -> int:
        return self.plan.discretization.velocity_set.dimension

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self.plan.discretization.grid.shape)

    def relation(self, marker_position: ArrayLike, /) -> SparseMarkerRelation:
        position = jnp.asarray(marker_position, dtype=self.initial_marker_position.dtype)
        expected = (self.capacity, self.dimension)
        if position.shape != expected:
            raise ValueError(f"marker_position must have shape {expected}.")
        finite_marker = jnp.all(jnp.isfinite(position), axis=-1)
        safe_position = jnp.where(
            (self.active & finite_marker)[:, None],
            position,
            self.initial_marker_position,
        )
        spacing = jnp.asarray(self.plan.discretization.cell_size, dtype=position.dtype)
        offset = self.route_positions - safe_position[:, None, :]
        scaled = jnp.abs(offset / spacing)
        radius = 0.5 * self.plan.stencil_width
        one_dimensional = jnp.where(
            scaled < radius,
            0.5 / radius * (1.0 + jnp.cos(jnp.pi * scaled / radius)),
            0.0,
        )
        raw = (
            jnp.prod(one_dimensional, axis=-1)
            * self.valid.astype(position.dtype)
            * self.active[:, None].astype(position.dtype)
        )
        raw_sum = jnp.sum(raw, axis=-1)
        supported_routes = self.valid & (raw > 0.0)
        coverage = jnp.sum(supported_routes, axis=-1) / self.plan.route_width
        covered = (~self.active) | (
            (raw_sum > jnp.finfo(position.dtype).tiny)
            & (coverage >= self.plan.minimum_coverage)
        )
        normalized = raw / jnp.where(raw_sum > 0.0, raw_sum, 1.0)[:, None]
        mean = oe.contract("mr,mrd->md", normalized, offset)
        centered = offset - mean[:, None, :]
        covariance = oe.contract("mr,mri,mrj->mij", normalized, centered, centered)
        solve_required = self.active & covered
        identity = jnp.broadcast_to(
            jnp.eye(self.dimension, dtype=position.dtype), covariance.shape
        )
        safe_covariance = jnp.where(solve_required[:, None, None], covariance, identity)
        correction = solve_small_linear(self.plan.local_solve, safe_covariance, -mean)
        affine = 1.0 + oe.contract("mrd,md->mr", centered, correction.value)
        weights = jnp.where(
            self.active[:, None], normalized * affine, jnp.zeros_like(normalized)
        )
        partition = jnp.abs(
            jnp.sum(weights, axis=-1) - self.active.astype(position.dtype)
        )
        first_moment = jnp.sqrt(
            jnp.sum(oe.contract("mr,mrd->md", weights, offset) ** 2, axis=-1)
        )
        minimum_weight = jnp.min(jnp.where(supported_routes, weights, jnp.inf), axis=-1)
        minimum_weight = jnp.where(self.active, minimum_weight, 0.0)
        tolerance = (
            4096.0
            * jnp.finfo(position.dtype).eps
            * jnp.maximum(1.0, jnp.max(jnp.abs(safe_position)))
        )
        local_success = (~solve_required) | correction.successful
        active_success = (
            finite_marker
            & covered
            & (coverage >= self.plan.minimum_coverage)
            & local_success
            & (partition <= tolerance)
            & (first_moment <= tolerance)
        )
        finite = (
            jnp.all(jnp.isfinite(weights), axis=-1)
            & jnp.isfinite(partition)
            & jnp.isfinite(first_moment)
        )
        successful = jnp.all((~self.active) | (active_success & finite))
        evidence = SparseMarkerRelationEvidence(
            coverage,
            jnp.sum(self.active, dtype=jnp.int32),
            jnp.sum(supported_routes, dtype=jnp.int32),
            partition,
            first_moment,
            minimum_weight,
            covered,
            local_success,
            finite,
            successful,
        )
        return SparseMarkerRelation(
            self.cell_indices,
            self.route_positions,
            weights,
            self.valid,
            self.active,
            safe_position,
            evidence,
            canonical_fingerprint(
                {
                    "kind": "cardio-fixed-route-marker-relation",
                    "transfer": self.prepared_id,
                }
            ),
            self.prepared_id,
        )

    def interpolate(
        self, relation: SparseMarkerRelation, grid_vector: ArrayLike, /
    ) -> Array:
        self._validate_relation(relation)
        values = jnp.asarray(grid_vector, dtype=relation.weights.dtype)
        expected = self.grid_shape + (self.dimension,)
        if values.shape != expected:
            raise ValueError(f"grid_vector must have shape {expected}.")
        gathered = values.reshape((-1, self.dimension))[relation.cell_indices]
        result = oe.contract("mr,mrd->md", relation.weights, gathered)
        return jnp.where(relation.active[:, None], result, 0.0)

    def spread(self, relation: SparseMarkerRelation, marker_force: ArrayLike, /) -> Array:
        """Spread total marker force as cell force density via the exact transpose."""
        self._validate_relation(relation)
        force = jnp.asarray(marker_force, dtype=relation.weights.dtype)
        expected = (self.capacity, self.dimension)
        if force.shape != expected:
            raise ValueError(f"marker_force must have shape {expected}.")
        force = jnp.where(relation.active[:, None], force, 0.0)
        payload = relation.weights[..., None] * force[:, None, :]
        cell_count = int(np.prod(self.grid_shape))
        flat = jnp.zeros((cell_count, self.dimension), dtype=force.dtype)
        flat = flat.at[relation.cell_indices.reshape((-1,))].add(
            payload.reshape((-1, self.dimension))
        )
        cell_measure = jnp.asarray(
            self.plan.discretization.cell_size**self.dimension, dtype=force.dtype
        )
        return (flat / cell_measure).reshape(self.grid_shape + (self.dimension,))

    def diagnostics(
        self,
        relation: SparseMarkerRelation,
        grid_velocity: ArrayLike,
        marker_force: ArrayLike,
        /,
        *,
        marker_velocity: ArrayLike | None = None,
        torque_origin: ArrayLike | None = None,
        body_indices: ArrayLike | None = None,
        body_centers: ArrayLike | None = None,
    ) -> SparseMarkerTransposeEvidence:
        self._validate_relation(relation)
        velocity = jnp.asarray(grid_velocity, dtype=relation.weights.dtype)
        force = jnp.asarray(marker_force, dtype=relation.weights.dtype)
        expected_grid = self.grid_shape + (self.dimension,)
        expected_marker = (self.capacity, self.dimension)
        if velocity.shape != expected_grid or force.shape != expected_marker:
            raise ValueError("Grid velocity or marker force has an incompatible shape.")
        target_velocity = (
            self.interpolate(relation, velocity)
            if marker_velocity is None
            else jnp.asarray(marker_velocity, dtype=force.dtype)
        )
        if target_velocity.shape != expected_marker:
            raise ValueError("marker_velocity has an incompatible shape.")
        if (body_indices is None) != (body_centers is None):
            raise ValueError("body_indices and body_centers must be supplied together.")
        if body_indices is None:
            indices = jnp.zeros((self.capacity,), dtype=jnp.int32)
            centers = jnp.zeros((1, self.dimension), dtype=force.dtype)
        else:
            indices = jnp.asarray(body_indices, dtype=jnp.int32)
            centers = jnp.asarray(body_centers, dtype=force.dtype)
            if indices.shape != (self.capacity,) or centers.ndim != 2:
                raise ValueError("Body indices or centers have incompatible shapes.")
            if centers.shape[1] != self.dimension or centers.shape[0] == 0:
                raise ValueError("body_centers must have shape (body, dimension).")
            indices = eqx.error_if(
                indices,
                jnp.any(indices < 0) | jnp.any(indices >= centers.shape[0]),
                "Every marker body index must name a prepared body.",
            )
        active_force = jnp.where(relation.active[:, None], force, 0.0)
        spread = self.spread(relation, active_force)
        interpolated = self.interpolate(relation, velocity)
        cell_measure = jnp.asarray(
            self.plan.discretization.cell_size**self.dimension, dtype=force.dtype
        )
        marker_resultant = jnp.sum(active_force, axis=0)
        grid_resultant = cell_measure * jnp.sum(
            spread.reshape((-1, self.dimension)), axis=0
        )
        force_residual = grid_resultant - marker_resultant
        origin = (
            jnp.zeros((self.dimension,), dtype=force.dtype)
            if torque_origin is None
            else jnp.asarray(torque_origin, dtype=force.dtype)
        )
        if origin.shape != (self.dimension,):
            raise ValueError("torque_origin must contain one coordinate per dimension.")
        marker_arm = relation.marker_position - origin
        route_force = relation.weights[..., None] * active_force[:, None, :]
        route_arm = relation.route_positions - origin
        if self.dimension == 2:
            marker_torque = jnp.sum(
                marker_arm[:, 0] * active_force[:, 1]
                - marker_arm[:, 1] * active_force[:, 0]
            ).reshape((1,))
            grid_torque = jnp.sum(
                route_arm[..., 0] * route_force[..., 1]
                - route_arm[..., 1] * route_force[..., 0]
            ).reshape((1,))
        else:
            marker_torque = jnp.sum(jnp.cross(marker_arm, active_force), axis=0)
            grid_torque = jnp.sum(jnp.cross(route_arm, route_force), axis=(0, 1))
        torque_residual = grid_torque - marker_torque
        interpolation_power = oe.contract("md,md->", interpolated, active_force)
        spreading_power = cell_measure * oe.contract("...d,...d->", velocity, spread)
        transpose_power_residual = spreading_power - interpolation_power
        body_count = centers.shape[0]
        membership = (
            indices[:, None] == jnp.arange(body_count)[None, :]
        ) & relation.active[:, None]
        body_force = -oe.contract(
            "mb,md->bd", membership.astype(force.dtype), active_force
        )
        body_route_arm = relation.marker_position - centers[indices]
        if self.dimension == 2:
            marker_body_torque = -(
                body_route_arm[:, 0] * active_force[:, 1]
                - body_route_arm[:, 1] * active_force[:, 0]
            )[:, None]
        else:
            marker_body_torque = -jnp.cross(body_route_arm, active_force)
        body_torque = oe.contract(
            "mb,ma->ba", membership.astype(force.dtype), marker_body_torque
        )
        marker_body_power = -oe.contract("md,md->m", target_velocity, active_force)
        body_power = oe.contract(
            "mb,m->b", membership.astype(force.dtype), marker_body_power
        )
        interface_power_residual = spreading_power + jnp.sum(body_power)
        finite = (
            jnp.all(jnp.isfinite(spread))
            & jnp.all(jnp.isfinite(force_residual))
            & jnp.all(jnp.isfinite(torque_residual))
            & jnp.isfinite(transpose_power_residual)
            & jnp.isfinite(interface_power_residual)
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.stack(
                    (
                        jnp.max(jnp.abs(marker_resultant)),
                        jnp.max(jnp.abs(grid_resultant)),
                        jnp.max(jnp.abs(marker_torque)),
                        jnp.max(jnp.abs(grid_torque)),
                        jnp.abs(interpolation_power),
                        jnp.abs(spreading_power),
                    )
                )
            ),
        )
        tolerance = 8192.0 * jnp.finfo(force.dtype).eps * scale
        successful = (
            relation.evidence.successful
            & finite
            & (jnp.max(jnp.abs(force_residual)) <= tolerance)
            & (jnp.max(jnp.abs(torque_residual)) <= tolerance)
            & (jnp.abs(transpose_power_residual) <= tolerance)
        )
        return SparseMarkerTransposeEvidence(
            marker_resultant,
            grid_resultant,
            force_residual,
            marker_torque,
            grid_torque,
            torque_residual,
            origin,
            interpolation_power,
            spreading_power,
            transpose_power_residual,
            body_force,
            body_torque,
            body_power,
            interface_power_residual,
            tolerance,
            finite,
            successful,
            self.prepared_id,
        )

    def _validate_relation(self, relation: SparseMarkerRelation, /) -> None:
        if not isinstance(relation, SparseMarkerRelation):
            raise TypeError("relation must be SparseMarkerRelation.")
        if relation.transfer_id != self.prepared_id:
            raise ValueError("Sparse marker relation belongs to another transfer.")


class ImmersedDirectForcingEvidence(StrictModule):
    interpolated_velocity: Array
    target_velocity: Array
    velocity_residual: Array
    maximum_velocity_residual: Array
    transpose: SparseMarkerTransposeEvidence
    iteration_count: Array
    converged: Array
    finite: Array
    successful: Array


class ImmersedDirectForcingResult(StrictModule):
    force_density: Array
    marker_fluid_force: Array
    marker_acceleration: Array
    corrected_velocity: Array
    evidence: ImmersedDirectForcingEvidence


class ImmersedDirectForcingPlan(StrictModule, NonTrainableState):
    """Fixed-work sparse direct forcing for an immersed compliant wall."""

    transfer: PreparedSparseMarkerTransfer
    iteration_count: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedSparseMarkerTransfer,
        /,
        *,
        iteration_count: int = 8,
        convergence_tolerance: float = 1.0e-6,
    ):
        if not isinstance(transfer, PreparedSparseMarkerTransfer):
            raise TypeError("transfer must be PreparedSparseMarkerTransfer.")
        iterations = int(iteration_count)
        tolerance = float(convergence_tolerance)
        if iterations <= 0:
            raise ValueError("iteration_count must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be finite and positive.")
        self.transfer = transfer
        self.iteration_count = iterations
        self.convergence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardio-sparse-immersed-direct-forcing",
                "transfer": transfer.prepared_id,
                "iterations": iterations,
                "tolerance": tolerance,
            }
        )

    def apply(
        self,
        fluid_velocity: ArrayLike,
        density: ArrayLike,
        marker_position: ArrayLike,
        target_velocity: ArrayLike,
        marker_measure: ArrayLike,
        time_step: ArrayLike,
        /,
        *,
        body_indices: ArrayLike | None = None,
        body_centers: ArrayLike | None = None,
    ) -> ImmersedDirectForcingResult:
        relation = self.transfer.relation(marker_position)
        velocity = jnp.asarray(fluid_velocity, dtype=relation.weights.dtype)
        rho = jnp.asarray(density, dtype=velocity.dtype)
        target = jnp.asarray(target_velocity, dtype=velocity.dtype)
        measures = jnp.asarray(marker_measure, dtype=velocity.dtype)
        dt = jnp.asarray(time_step, dtype=velocity.dtype).reshape(())
        expected_grid = self.transfer.grid_shape + (self.transfer.dimension,)
        expected_marker = (self.transfer.capacity, self.transfer.dimension)
        if velocity.shape != expected_grid or rho.shape != self.transfer.grid_shape:
            raise ValueError("Fluid velocity or density has an incompatible shape.")
        if target.shape != expected_marker or measures.shape != (self.transfer.capacity,):
            raise ValueError(
                "Marker target velocity or measure has an incompatible shape."
            )
        active = relation.active
        valid_input = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(rho))
            & jnp.all(jnp.where(active, jnp.isfinite(measures) & (measures > 0.0), True))
            & jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(rho > 0.0)
        )
        flat_density = rho.reshape((-1,))[relation.cell_indices]
        marker_density = jnp.sum(relation.weights * flat_density, axis=-1)
        valid_input = valid_input & jnp.all(
            jnp.where(active, jnp.isfinite(marker_density) & (marker_density > 0.0), True)
        )
        force_density = jnp.zeros_like(velocity)
        marker_force = jnp.zeros_like(target)
        marker_acceleration = jnp.zeros_like(target)
        corrected = velocity
        for _ in range(self.iteration_count):
            interpolated = self.transfer.interpolate(relation, corrected)
            marker_acceleration = jnp.where(
                active[:, None], (target - interpolated) / dt, 0.0
            )
            increment = marker_density[:, None] * measures[:, None] * marker_acceleration
            marker_force = marker_force + increment
            spread = self.transfer.spread(relation, increment)
            force_density = force_density + spread
            corrected = corrected + dt * spread / rho[..., None]
        interpolated = self.transfer.interpolate(relation, corrected)
        residual = jnp.where(active[:, None], interpolated - target, 0.0)
        maximum_residual = jnp.max(jnp.abs(residual))
        transpose = self.transfer.diagnostics(
            relation,
            velocity,
            marker_force,
            marker_velocity=target,
            body_indices=body_indices,
            body_centers=body_centers,
        )
        converged = maximum_residual <= self.convergence_tolerance
        finite = (
            valid_input
            & jnp.all(jnp.isfinite(force_density))
            & jnp.all(jnp.isfinite(marker_force))
            & jnp.all(jnp.isfinite(corrected))
        )
        successful = (
            relation.evidence.successful & transpose.successful & converged & finite
        )
        evidence = ImmersedDirectForcingEvidence(
            interpolated,
            target,
            residual,
            maximum_residual,
            transpose,
            jnp.asarray(self.iteration_count, dtype=jnp.int32),
            converged,
            finite,
            successful,
        )
        return ImmersedDirectForcingResult(
            force_density,
            marker_force,
            marker_acceleration,
            corrected,
            evidence,
        )


class ImmersedPostLBMEvidence(StrictModule):
    interpolated_velocity: Array
    target_velocity: Array
    velocity_residual: Array
    maximum_velocity_residual: Array
    finite: Array
    successful: Array


class ImmersedLBMAdvanceResult(StrictModule):
    candidate_state: Any
    velocity: Array
    density: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array


class ImmersedFEMAdvanceResult(StrictModule):
    candidate_state: Any
    marker_position: Array
    marker_velocity: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array
    work: Array


FluidFieldProvider = Callable[[Any, Any], tuple[ArrayLike, ArrayLike]]
ImmersedLBMAdvance = Callable[[Any, Any, Array, Any], ImmersedLBMAdvanceResult]
ImmersedFEMAdvance = Callable[[Any, Any, Array, Any], ImmersedFEMAdvanceResult]


class ImmersedFSIParticipantBundle(StrictModule, NonTrainableState):
    """A two-participant graph and bounded added-mass fixed-point policy."""

    graph: CouplingGraph
    policy: ImplicitCouplingPolicy
    differentiation: CouplingDifferentiationPolicy
    position_space: ArraySpace
    velocity_space: ArraySpace
    load_space: ArraySpace
    bundle_id: str = eqx.field(static=True)

    def prepare(
        self,
        fluid_state: Any,
        solid_state: Any,
        marker_position: ArrayLike,
        marker_velocity: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
        problem_id: str = "cardiovascular-immersed-fsi",
    ) -> PreparedCoupling:
        position = self.position_space.validate(
            jnp.asarray(marker_position, dtype=self.position_space.dtype)
        )
        velocity = self.velocity_space.validate(
            jnp.asarray(marker_velocity, dtype=self.velocity_space.dtype)
        )
        zero_load = self.load_space.zeros()
        return prepare_coupling(
            self.graph,
            (fluid_state, solid_state),
            (position, velocity, zero_load),
            policy=self.policy,
            differentiation=self.differentiation,
            time=time,
            args=args,
            problem_id=problem_id,
        )


def build_immersed_lbm_participant(
    forcing: ImmersedDirectForcingPlan,
    fluid_fields: FluidFieldProvider,
    advance: ImmersedLBMAdvance,
    marker_measure: ArrayLike,
    /,
    *,
    coupling_id: str,
    position_reference: float,
    velocity_reference: float,
    force_reference: float,
) -> CallableCouplingSubsystem:
    """Build the fluid participant using sparse forcing inside each window call."""
    if not isinstance(forcing, ImmersedDirectForcingPlan):
        raise TypeError("forcing must be ImmersedDirectForcingPlan.")
    if not callable(fluid_fields) or not callable(advance):
        raise TypeError("fluid_fields and advance must be callable.")
    identifier = str(coupling_id)
    if not identifier:
        raise ValueError("coupling_id must be non-empty.")
    transfer = forcing.transfer
    dtype = transfer.initial_marker_position.dtype
    interface_shape = (transfer.capacity, transfer.dimension)
    position_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-position-space",
    )
    velocity_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-velocity-space",
    )
    load_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-load-space",
    )
    measures = jnp.asarray(marker_measure, dtype=dtype)
    if measures.shape != (transfer.capacity,):
        raise ValueError("marker_measure must have one value per marker capacity slot.")
    position_port = CouplingPort(
        f"{identifier}/position-into-fluid",
        "input",
        position_space,
        reference_scale=position_reference,
    )
    velocity_port = CouplingPort(
        f"{identifier}/velocity-into-fluid",
        "input",
        velocity_space,
        reference_scale=velocity_reference,
    )
    load_port = CouplingPort(
        f"{identifier}/load-from-fluid",
        "output",
        load_space,
        reference_scale=force_reference,
    )

    def advance_window(window, start_state, inputs, args):
        position = position_space.validate(inputs[0])
        target_velocity = velocity_space.validate(inputs[1])
        fluid_velocity, density = fluid_fields(start_state, args)
        direct = forcing.apply(
            fluid_velocity,
            density,
            position,
            target_velocity,
            measures,
            window.size,
        )
        step = advance(window, start_state, direct.force_density, args)
        if not isinstance(step, ImmersedLBMAdvanceResult):
            raise TypeError("Immersed LBM advance must return ImmersedLBMAdvanceResult.")
        candidate_velocity = jnp.asarray(step.velocity, dtype=dtype)
        candidate_density = jnp.asarray(step.density, dtype=dtype)
        if candidate_velocity.shape != transfer.grid_shape + (transfer.dimension,):
            raise ValueError("Immersed LBM candidate velocity has an incompatible shape.")
        if candidate_density.shape != transfer.grid_shape:
            raise ValueError("Immersed LBM candidate density has an incompatible shape.")
        relation = transfer.relation(position)
        actual_velocity = transfer.interpolate(relation, candidate_velocity)
        actual_residual = jnp.where(
            relation.active[:, None],
            actual_velocity - target_velocity,
            0.0,
        )
        actual_maximum = jnp.max(jnp.abs(actual_residual))
        actual_finite = jnp.all(jnp.isfinite(actual_velocity)) & jnp.all(
            jnp.isfinite(actual_residual)
        )
        actual_successful = (
            relation.evidence.successful
            & actual_finite
            & (actual_maximum <= forcing.convergence_tolerance)
        )
        post_lbm = ImmersedPostLBMEvidence(
            actual_velocity,
            target_velocity,
            actual_residual,
            actual_maximum,
            actual_finite,
            actual_successful,
        )
        body_load = load_space.validate(-direct.marker_fluid_force)
        successful = (
            jnp.asarray(step.successful)
            & direct.evidence.successful
            & post_lbm.successful
        )
        residual = jnp.maximum(
            jnp.maximum(
                jnp.asarray(step.residual_norm),
                direct.evidence.maximum_velocity_residual,
            ),
            post_lbm.maximum_velocity_residual,
        )
        return CouplingSubsystemResult(
            step.candidate_state,
            (body_load,),
            successful=successful,
            status=step.status,
            residual_norm=residual,
            iterations=step.iterations + direct.evidence.iteration_count,
            work=step.work + direct.evidence.iteration_count,
            auxiliary=(step, direct, post_lbm),
        )

    return CallableCouplingSubsystem(
        advance_window,
        subsystem_id=f"{identifier}/lbm-fluid",
        input_ports=(position_port, velocity_port),
        output_ports=(load_port,),
        capabilities=CouplingSubsystemCapabilities(
            jit=True,
            differentiable=True,
            deterministic_replay=True,
            fixed_topology=True,
        ),
        discretization_bundle_id=transfer.plan.discretization.prepared_id,
    )


def build_immersed_fem_participant(
    transfer: PreparedSparseMarkerTransfer,
    advance: ImmersedFEMAdvance,
    /,
    *,
    coupling_id: str,
    position_reference: float,
    velocity_reference: float,
    force_reference: float,
    discretization_bundle_id: str | None = None,
) -> CallableCouplingSubsystem:
    """Build the compliant FEM wall participant over marker loads and kinematics."""
    if not isinstance(transfer, PreparedSparseMarkerTransfer):
        raise TypeError("transfer must be PreparedSparseMarkerTransfer.")
    if not callable(advance):
        raise TypeError("advance must be callable.")
    identifier = str(coupling_id)
    if not identifier:
        raise ValueError("coupling_id must be non-empty.")
    dtype = transfer.initial_marker_position.dtype
    interface_shape = (transfer.capacity, transfer.dimension)
    position_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-position-space",
    )
    velocity_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-velocity-space",
    )
    load_space = ArraySpace(
        interface_shape,
        dtype=dtype,
        space_id=f"{identifier}/interface-load-space",
    )
    load_port = CouplingPort(
        f"{identifier}/load-into-solid",
        "input",
        load_space,
        reference_scale=force_reference,
    )
    position_port = CouplingPort(
        f"{identifier}/position-from-solid",
        "output",
        position_space,
        reference_scale=position_reference,
    )
    velocity_port = CouplingPort(
        f"{identifier}/velocity-from-solid",
        "output",
        velocity_space,
        reference_scale=velocity_reference,
    )

    def advance_window(window, start_state, inputs, args):
        load = load_space.validate(inputs[0])
        step = advance(window, start_state, load, args)
        if not isinstance(step, ImmersedFEMAdvanceResult):
            raise TypeError("Immersed FEM advance must return ImmersedFEMAdvanceResult.")
        position = position_space.validate(jnp.asarray(step.marker_position, dtype=dtype))
        velocity = velocity_space.validate(jnp.asarray(step.marker_velocity, dtype=dtype))
        finite = jnp.all(jnp.isfinite(position)) & jnp.all(jnp.isfinite(velocity))
        return CouplingSubsystemResult(
            step.candidate_state,
            (position, velocity),
            successful=jnp.asarray(step.successful) & finite,
            status=step.status,
            residual_norm=step.residual_norm,
            iterations=step.iterations,
            work=step.work,
            auxiliary=step,
        )

    return CallableCouplingSubsystem(
        advance_window,
        subsystem_id=f"{identifier}/fem-wall",
        input_ports=(load_port,),
        output_ports=(position_port, velocity_port),
        capabilities=CouplingSubsystemCapabilities(
            jit=True,
            differentiable=True,
            deterministic_replay=True,
            fixed_topology=True,
        ),
        discretization_bundle_id=discretization_bundle_id,
    )


def build_immersed_fsi_participants(
    forcing: ImmersedDirectForcingPlan,
    fluid_fields: FluidFieldProvider,
    advance_fluid: ImmersedLBMAdvance,
    advance_solid: ImmersedFEMAdvance,
    marker_measure: ArrayLike,
    /,
    *,
    coupling_id: str = "cardiovascular-immersed-fsi",
    position_reference: float = 1.0,
    velocity_reference: float = 1.0,
    force_reference: float = 1.0,
    position_tolerance: float = 1.0e-6,
    velocity_tolerance: float = 1.0e-6,
    force_tolerance: float = 1.0e-6,
    damping: float = 0.5,
    maximum_iterations: int = 25,
    solid_discretization_bundle_id: str | None = None,
) -> ImmersedFSIParticipantBundle:
    """Build an implicit, rollback-capable LBM--FEM added-mass coupling graph."""
    fluid = build_immersed_lbm_participant(
        forcing,
        fluid_fields,
        advance_fluid,
        marker_measure,
        coupling_id=coupling_id,
        position_reference=position_reference,
        velocity_reference=velocity_reference,
        force_reference=force_reference,
    )
    solid = build_immersed_fem_participant(
        forcing.transfer,
        advance_solid,
        coupling_id=coupling_id,
        position_reference=position_reference,
        velocity_reference=velocity_reference,
        force_reference=force_reference,
        discretization_bundle_id=solid_discretization_bundle_id,
    )
    position_exchange = CouplingExchange(
        f"{coupling_id}/solid-position",
        solid.output_ports[0].port_id,
        fluid.input_ports[0].port_id,
    )
    velocity_exchange = CouplingExchange(
        f"{coupling_id}/solid-velocity",
        solid.output_ports[1].port_id,
        fluid.input_ports[1].port_id,
    )
    load_exchange = CouplingExchange(
        f"{coupling_id}/fluid-load",
        fluid.output_ports[0].port_id,
        solid.input_ports[0].port_id,
    )
    graph = CouplingGraph(
        (fluid, solid),
        (position_exchange, velocity_exchange, load_exchange),
    )
    sweep = CouplingSweep(
        "gauss-seidel", subsystem_order=(fluid.subsystem_id, solid.subsystem_id)
    )
    termination = NonlinearTermination(
        absolute_residual=min(
            float(position_tolerance),
            float(velocity_tolerance),
            float(force_tolerance),
        ),
        relative_residual=0.0,
        maximum_steps=maximum_iterations,
    )
    policy = ImplicitCouplingPolicy(
        FixedPointIteration(damping=damping),
        termination,
        (
            CouplingTolerance(
                fluid.input_ports[0].port_id,
                absolute=position_tolerance,
            ),
            CouplingTolerance(
                fluid.input_ports[1].port_id,
                absolute=velocity_tolerance,
            ),
            CouplingTolerance(
                solid.input_ports[0].port_id,
                absolute=force_tolerance,
            ),
        ),
        fixed_point_sweep=sweep,
    )
    differentiation = CouplingDifferentiationPolicy("none")
    return ImmersedFSIParticipantBundle(
        graph,
        policy,
        differentiation,
        fluid.input_ports[0].space,
        fluid.input_ports[1].space,
        fluid.output_ports[0].space,
        canonical_fingerprint(
            {
                "kind": "cardiovascular-immersed-fsi-participants",
                "graph": graph.graph_id,
                "policy": policy.policy_id,
                "fixed_route_transfer": forcing.transfer.prepared_id,
            }
        ),
    )


__all__ = [
    "FluidFieldProvider",
    "ImmersedDirectForcingEvidence",
    "ImmersedDirectForcingPlan",
    "ImmersedDirectForcingResult",
    "ImmersedFEMAdvance",
    "ImmersedFEMAdvanceResult",
    "ImmersedFSIParticipantBundle",
    "ImmersedLBMAdvance",
    "ImmersedLBMAdvanceResult",
    "ImmersedPostLBMEvidence",
    "PreparedSparseMarkerTransfer",
    "SparseMarkerRelation",
    "SparseMarkerRelationEvidence",
    "SparseMarkerTransferPlan",
    "SparseMarkerTransposeEvidence",
    "build_immersed_fem_participant",
    "build_immersed_fsi_participants",
    "build_immersed_lbm_participant",
]
