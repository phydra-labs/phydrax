#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_distributed import (
    MACDistributedPlanStatus,
    MACDistributedState,
    PreparedMACDistributedTopology,
)


def _finite_tree(pressure: Array, velocity: FaceVelocity, /) -> Array:
    finite = jnp.all(jnp.isfinite(pressure))
    for component in velocity:
        finite = finite & jnp.all(jnp.isfinite(component))
    return finite


class MACCollectiveAdapter(StrictModule, NonTrainableState):
    """Named-mesh collective algebra used only from a shard_map context."""

    mesh_axis_names: tuple[str, ...] = eqx.field(static=True)
    collective_size: int = eqx.field(static=True)
    process_index: int = eqx.field(static=True)
    process_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(self, topology: PreparedMACDistributedTopology, /):
        if not isinstance(topology, PreparedMACDistributedTopology):
            raise TypeError("topology must be PreparedMACDistributedTopology.")
        names = tuple(str(name) for name in topology.plan.mesh.axis_names)
        self.mesh_axis_names = names
        self.collective_size = topology.status.device_count
        self.process_index = topology.status.process_index
        self.process_count = topology.status.process_count
        self.topology_id = topology.plan.topology_id
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "mac-collective-adapter",
                "topology": topology.plan.topology_id,
                "mesh_axis_names": list(names),
                "collective_size": topology.status.device_count,
                "process_index": topology.status.process_index,
                "process_count": topology.status.process_count,
            }
        )

    def sum(self, value: Array, /) -> Array:
        return jax.lax.psum(value, axis_name=self.mesh_axis_names)

    def minimum(self, value: Array, /) -> Array:
        return jax.lax.pmin(value, axis_name=self.mesh_axis_names)

    def maximum(self, value: Array, /) -> Array:
        return jax.lax.pmax(value, axis_name=self.mesh_axis_names)

    def all(self, predicate: Array, /) -> Array:
        encoded = jnp.asarray(predicate, dtype=jnp.int32)
        return self.minimum(encoded) == 1

    def agreement(self, predicate: Array, /) -> Array:
        encoded = jnp.asarray(predicate, dtype=jnp.int32)
        return self.minimum(encoded) == self.maximum(encoded)

    def weighted_inner(
        self,
        left: Array,
        right: Array,
        weights: Array,
        /,
    ) -> Array:
        local = jnp.sum(weights * jnp.conj(left) * right)
        return self.sum(local)

    def weighted_mean(self, value: Array, weights: Array, /) -> Array:
        numerator = self.sum(jnp.sum(weights * value))
        denominator = self.sum(jnp.sum(weights))
        return numerator / denominator


class MACDistributedProjectionResult(StrictModule):
    """Globally agreed atomic projection result and matrix-free solve evidence."""

    state: MACDistributedState
    velocity: FaceVelocity
    pressure: Array
    pressure_increment: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    face_inverse_momentum: FaceVelocity
    iterations: Array
    residual_norm: Array
    rhs_norm: Array
    divergence_norm: Array
    compatibility_defect: Array
    gauge_defect: Array
    pressure_action_adjoint_defect: Array
    all_finite: Array
    rank_agreement: Array
    linear_converged: Array
    converged: Array
    committed: Array
    topology_id: str = eqx.field(static=True)
    projection_plan_id: str = eqx.field(static=True)
    collective_adapter_id: str = eqx.field(static=True)
    lifecycle: str = eqx.field(static=True)
    result_contract_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.committed


class MACDistributedProjectionPlan(StrictModule, NonTrainableState):
    """Collective, globally gauged matrix-free CG projection on a MAC mesh."""

    topology: PreparedMACDistributedTopology
    collectives: MACCollectiveAdapter
    density: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    status: MACDistributedPlanStatus
    plan_id: str = eqx.field(static=True)
    result_contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: PreparedMACDistributedTopology,
        /,
        *,
        density: float = 1.0,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-9,
        maximum_iterations: int = 500,
    ):
        if not isinstance(topology, PreparedMACDistributedTopology):
            raise TypeError("topology must be PreparedMACDistributedTopology.")
        density_ = float(density)
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(relative)
            or relative <= 0.0
            or not np.isfinite(absolute)
            or absolute <= 0.0
            or iterations <= 0
        ):
            raise ValueError(
                "Distributed projection density, tolerances, and iteration limit are invalid."
            )
        collectives = MACCollectiveAdapter(topology)
        plan_id = canonical_fingerprint(
            {
                "kind": "mac-distributed-projection-plan",
                "topology": topology.prepared_id,
                "collectives": collectives.adapter_id,
                "density": density_,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "maximum_iterations": iterations,
                "pressure_solver": "collective-matrix-free-cg",
                "compatibility": "global-volume-mean",
                "gauge": "global-volume-zero-mean",
                "commit": "all-rank-atomic",
            }
        )
        result_contract_id = canonical_fingerprint(
            {
                "kind": "mac-distributed-projection-result-contract",
                "plan": plan_id,
                "stage_data": "dynamic-not-fingerprinted",
                "lifecycle": "atomic-global-commit-or-rollback",
                "adjoint_evidence": "weighted-pressure-action",
            }
        )
        self.topology = topology
        self.collectives = collectives
        self.density = density_
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.maximum_iterations = iterations
        self.status = topology.status
        self.plan_id = plan_id
        self.result_contract_id = result_contract_id

    def _require_ready(self, /) -> None:
        if not self.status.ready:
            raise RuntimeError(
                f"Distributed MAC projection is unavailable: {self.status.reason}."
            )

    def project(
        self,
        state: MACDistributedState,
        step_size: ArrayLike,
        /,
        *,
        inverse_momentum_diagonal: ArrayLike | None = None,
    ) -> MACDistributedProjectionResult:
        """Project once and commit on every rank only when every rank accepts."""

        self._require_ready()
        incoming = self.topology.validate(state)
        dtype = self.topology.operators.pressure_space.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        if inverse_momentum_diagonal is None:
            inverse_host = jnp.full(
                self.topology.operators.discretization.cell_shape,
                step / self.density,
                dtype=dtype,
            )
            inverse = self.topology.distribute_pressure(inverse_host)
        elif isinstance(inverse_momentum_diagonal, jax.Array):
            inverse = inverse_momentum_diagonal
            self.topology._validate_pressure_sharding(inverse)
        else:
            inverse = self.topology.distribute_pressure(inverse_momentum_diagonal)

        stencils = self.topology.local_stencils
        collectives = self.collectives
        relative_tolerance = self.relative_tolerance
        absolute_tolerance = self.absolute_tolerance
        maximum_iterations = self.maximum_iterations
        local_shape = stencils.local_pressure_shape
        dimension = stencils.dimension

        def local_project(
            velocity,
            incoming_pressure,
            inverse_cell,
            cell_volumes,
            face_measures,
            face_distances,
        ):
            def compatibility_project(value):
                return value - collectives.weighted_mean(value, cell_volumes)

            def gauge_project(value):
                return value - collectives.weighted_mean(value, cell_volumes)

            def gradient(value):
                return stencils.gradient(value, face_distances)

            def divergence(value):
                return stencils.divergence(value, face_measures, cell_volumes)

            face_inverse = stencils.interpolate_inverse_momentum(inverse_cell)

            def action(value):
                mean = collectives.weighted_mean(value, cell_volumes)
                projected = value - mean
                derivative = gradient(projected)
                flux = tuple(
                    coefficient * component
                    for coefficient, component in zip(
                        face_inverse, derivative, strict=True
                    )
                )
                return -divergence(flux) + mean

            divergence_before = divergence(velocity)
            right_hand_side = -compatibility_project(divergence_before)
            initial = gauge_project(incoming_pressure)
            residual = right_hand_side - action(initial)
            direction = residual
            residual_squared = jnp.real(
                collectives.weighted_inner(residual, residual, cell_volumes)
            )
            rhs_squared = jnp.real(
                collectives.weighted_inner(right_hand_side, right_hand_side, cell_volumes)
            )
            tolerance = jnp.maximum(
                absolute_tolerance,
                relative_tolerance * jnp.sqrt(jnp.maximum(rhs_squared, 0.0)),
            )
            tolerance_squared = tolerance * tolerance
            initial_finite = collectives.all(
                jnp.all(jnp.isfinite(initial))
                & jnp.all(jnp.isfinite(residual))
                & jnp.isfinite(residual_squared)
                & jnp.isfinite(rhs_squared)
                & jnp.all(jnp.isfinite(inverse_cell))
                & jnp.all(inverse_cell > 0.0)
                & jnp.isfinite(step)
                & (step > 0.0)
            )

            def condition(carry):
                iteration, _, _, _, squared, valid = carry
                return (
                    (iteration < maximum_iterations)
                    & valid
                    & (squared > tolerance_squared)
                )

            def body(carry):
                iteration, solution, residual_, direction_, squared, valid = carry
                action_direction = action(direction_)
                denominator = jnp.real(
                    collectives.weighted_inner(direction_, action_direction, cell_volumes)
                )
                step_valid = collectives.all(
                    jnp.isfinite(denominator)
                    & (denominator > 0.0)
                    & jnp.isfinite(squared)
                    & (squared >= 0.0)
                )
                alpha = jnp.where(step_valid, squared / denominator, 0.0)
                solution_candidate = gauge_project(solution + alpha * direction_)
                residual_candidate = compatibility_project(
                    residual_ - alpha * action_direction
                )
                squared_candidate = jnp.real(
                    collectives.weighted_inner(
                        residual_candidate, residual_candidate, cell_volumes
                    )
                )
                candidate_finite = collectives.all(
                    jnp.all(jnp.isfinite(solution_candidate))
                    & jnp.all(jnp.isfinite(residual_candidate))
                    & jnp.isfinite(squared_candidate)
                    & (squared_candidate >= 0.0)
                )
                valid_candidate = valid & step_valid & candidate_finite
                beta = jnp.where(
                    valid_candidate & (squared > 0.0),
                    squared_candidate / squared,
                    0.0,
                )
                direction_candidate = residual_candidate + beta * direction_
                return (
                    iteration + 1,
                    jnp.where(valid_candidate, solution_candidate, solution),
                    jnp.where(valid_candidate, residual_candidate, residual_),
                    jnp.where(valid_candidate, direction_candidate, direction_),
                    jnp.where(valid_candidate, squared_candidate, squared),
                    valid_candidate,
                )

            iteration, solution, _, _, _, linear_valid = jax.lax.while_loop(
                condition,
                body,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    initial,
                    residual,
                    direction,
                    residual_squared,
                    initial_finite,
                ),
            )
            increment_candidate = gauge_project(solution)
            pressure_residual = action(increment_candidate) - right_hand_side
            residual_norm = jnp.sqrt(
                jnp.maximum(
                    jnp.real(
                        collectives.weighted_inner(
                            pressure_residual, pressure_residual, cell_volumes
                        )
                    ),
                    0.0,
                )
            )
            rhs_norm = jnp.sqrt(jnp.maximum(rhs_squared, 0.0))
            correction_gradient = gradient(increment_candidate)
            velocity_candidate = tuple(
                component - coefficient * derivative
                for component, coefficient, derivative in zip(
                    velocity,
                    face_inverse,
                    correction_gradient,
                    strict=True,
                )
            )
            pressure_candidate = gauge_project(incoming_pressure + increment_candidate)
            divergence_candidate = divergence(velocity_candidate)
            divergence_norm = jnp.sqrt(
                jnp.maximum(
                    jnp.real(
                        collectives.weighted_inner(
                            divergence_candidate,
                            divergence_candidate,
                            cell_volumes,
                        )
                    ),
                    0.0,
                )
            )
            compatibility_defect = jnp.abs(
                collectives.weighted_mean(right_hand_side, cell_volumes)
            )
            gauge_defect = jnp.abs(
                collectives.weighted_mean(pressure_candidate, cell_volumes)
            )

            global_linear_index = jnp.zeros(local_shape, dtype=dtype)
            stride = 1
            for axis in range(dimension - 1, -1, -1):
                mesh_axis = stencils.axis_mesh_names[axis]
                partition_offset = (
                    0
                    if mesh_axis is None
                    else jax.lax.axis_index(mesh_axis) * local_shape[axis]
                )
                coordinate = jnp.arange(local_shape[axis], dtype=dtype)
                coordinate = coordinate + jnp.asarray(partition_offset, dtype=dtype)
                reshape = [1] * dimension
                reshape[axis] = local_shape[axis]
                global_linear_index = global_linear_index + stride * coordinate.reshape(
                    tuple(reshape)
                )
                stride *= stencils.pressure_shape[axis]
            probe = compatibility_project(jnp.sin(0.37 * (global_linear_index + 1.0)))
            action_probe = action(probe)
            action_increment = action(increment_candidate)
            adjoint_defect = jnp.abs(
                collectives.weighted_inner(
                    increment_candidate, action_probe, cell_volumes
                )
                - collectives.weighted_inner(action_increment, probe, cell_volumes)
            )
            candidate_finite_local = (
                _finite_tree(pressure_candidate, velocity_candidate)
                & jnp.all(jnp.isfinite(pressure_residual))
                & jnp.all(jnp.isfinite(divergence_candidate))
                & jnp.isfinite(residual_norm)
                & jnp.isfinite(rhs_norm)
                & jnp.isfinite(divergence_norm)
                & jnp.isfinite(compatibility_defect)
                & jnp.isfinite(gauge_defect)
                & jnp.isfinite(adjoint_defect)
            )
            all_finite = collectives.all(candidate_finite_local)
            scale = jnp.maximum(rhs_norm, 1.0)
            adjoint_scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.abs(
                        collectives.weighted_inner(
                            increment_candidate, action_probe, cell_volumes
                        )
                    ),
                    jnp.abs(
                        collectives.weighted_inner(action_increment, probe, cell_volumes)
                    ),
                ),
            )
            adjoint_tolerance = (
                4096.0 * jnp.finfo(incoming_pressure.dtype).eps * adjoint_scale
            )
            linear_converged = linear_valid & (
                residual_norm
                <= jnp.maximum(absolute_tolerance, relative_tolerance * rhs_norm)
            )
            local_accept = (
                linear_converged
                & all_finite
                & (
                    divergence_norm
                    <= jnp.maximum(absolute_tolerance, relative_tolerance * scale)
                )
                & (compatibility_defect <= absolute_tolerance)
                & (gauge_defect <= absolute_tolerance)
                & (adjoint_defect <= adjoint_tolerance)
            )
            rank_agreement = collectives.agreement(local_accept)
            committed = collectives.all(local_accept) & rank_agreement
            velocity_result = tuple(
                jnp.where(committed, candidate, original)
                for candidate, original in zip(velocity_candidate, velocity, strict=True)
            )
            pressure_result = jnp.where(committed, pressure_candidate, incoming_pressure)
            increment_result = jnp.where(
                committed,
                increment_candidate,
                jnp.zeros_like(increment_candidate),
            )
            divergence_result = jnp.where(
                committed, divergence_candidate, divergence_before
            )
            return (
                velocity_result,
                pressure_result,
                increment_result,
                divergence_before,
                divergence_result,
                pressure_residual,
                right_hand_side,
                face_inverse,
                iteration,
                residual_norm,
                rhs_norm,
                divergence_norm,
                compatibility_defect,
                gauge_defect,
                adjoint_defect,
                all_finite,
                rank_agreement,
                linear_converged,
                committed,
            )

        scalar_spec = PartitionSpec()
        outputs = jax.shard_map(
            local_project,
            mesh=self.topology.plan.mesh,
            in_specs=(
                self.topology.plan.face_specs,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.face_specs,
                self.topology.plan.face_specs,
            ),
            out_specs=(
                self.topology.plan.face_specs,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.pressure_spec,
                self.topology.plan.face_specs,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
                scalar_spec,
            ),
        )(
            incoming.velocity,
            incoming.pressure,
            inverse,
            self.topology.cell_volumes,
            self.topology.face_measures,
            self.topology.face_distances,
        )
        (
            velocity,
            pressure,
            pressure_increment,
            divergence_before,
            divergence_after,
            pressure_residual,
            compatible_rhs,
            face_inverse,
            iterations,
            residual_norm,
            rhs_norm,
            divergence_norm,
            compatibility_defect,
            gauge_defect,
            adjoint_defect,
            all_finite,
            rank_agreement,
            linear_converged,
            committed,
        ) = outputs
        result_state = MACDistributedState(
            pressure,
            velocity,
            self.topology.plan.topology_id,
            self.topology.plan.layout_id,
        )
        return MACDistributedProjectionResult(
            state=result_state,
            velocity=velocity,
            pressure=pressure,
            pressure_increment=pressure_increment,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=pressure_residual,
            compatible_rhs=compatible_rhs,
            face_inverse_momentum=face_inverse,
            iterations=iterations,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            divergence_norm=divergence_norm,
            compatibility_defect=compatibility_defect,
            gauge_defect=gauge_defect,
            pressure_action_adjoint_defect=adjoint_defect,
            all_finite=all_finite,
            rank_agreement=rank_agreement,
            linear_converged=linear_converged,
            converged=committed,
            committed=committed,
            topology_id=self.topology.plan.topology_id,
            projection_plan_id=self.plan_id,
            collective_adapter_id=self.collectives.adapter_id,
            lifecycle="atomic-global-commit-or-rollback",
            result_contract_id=self.result_contract_id,
        )


__all__ = [
    "MACCollectiveAdapter",
    "MACDistributedProjectionPlan",
    "MACDistributedProjectionResult",
]
