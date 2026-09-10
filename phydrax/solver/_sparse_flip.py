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
from ..discretization.flip import (
    PreparedSparseFLIPParticleTransfer,
    SparseFLIPTransferState,
)
from ..linalg import (
    ArraySpace,
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
    TolerancePolicy,
)


class _SparseMACRelations(StrictModule, NonTrainableState):
    cell_lower_faces: tuple[Array, ...]
    cell_upper_faces: tuple[Array, ...]
    face_lower_cells: tuple[Array, ...]
    face_upper_cells: tuple[Array, ...]
    face_lower_supported: tuple[Array, ...]
    face_upper_supported: tuple[Array, ...]
    cell_valid: Array
    face_valid: tuple[Array, ...]
    complete: Array


def _gradient(
    pressure: Array,
    liquid: Array,
    relations: _SparseMACRelations,
    spacing: tuple[float, ...],
) -> tuple[Array, ...]:
    values = []
    for axis, width in enumerate(spacing):
        lower = relations.face_lower_cells[axis]
        upper = relations.face_upper_cells[axis]
        lower_value = jnp.where(
            relations.face_lower_supported[axis] & liquid[lower],
            pressure[lower],
            0.0,
        )
        upper_value = jnp.where(
            relations.face_upper_supported[axis] & liquid[upper],
            pressure[upper],
            0.0,
        )
        values.append(
            jnp.where(
                relations.face_valid[axis],
                (upper_value - lower_value) / width,
                0.0,
            )
        )
    return tuple(values)


def _divergence(
    velocity: tuple[Array, ...],
    relations: _SparseMACRelations,
    spacing: tuple[float, ...],
) -> Array:
    result = jnp.zeros(relations.cell_valid.shape, dtype=velocity[0].dtype)
    for axis, width in enumerate(spacing):
        lower = velocity[axis][relations.cell_lower_faces[axis]]
        upper = velocity[axis][relations.cell_upper_faces[axis]]
        result = result + (upper - lower) / width
    return jnp.where(relations.cell_valid, result, 0.0)


class _SparseMACPressureAction(StrictModule, NonTrainableState):
    relations: _SparseMACRelations
    liquid: Array
    coefficient: Array
    cell_measure: Array
    spacing: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, pressure: Array, /) -> Array:
        all_liquid = jnp.all(self.liquid | ~self.relations.cell_valid)
        denominator = jnp.sum(jnp.where(self.liquid, self.cell_measure, 0.0))
        mean = jnp.where(
            denominator > 0.0,
            jnp.sum(self.cell_measure * jnp.where(self.liquid, pressure, 0.0))
            / jnp.where(denominator > 0.0, denominator, 1.0),
            0.0,
        )
        restricted = jnp.where(
            all_liquid,
            jnp.where(self.liquid, pressure - mean, 0.0),
            jnp.where(self.liquid, pressure, 0.0),
        )
        gradient = _gradient(
            restricted,
            self.liquid,
            self.relations,
            self.spacing,
        )
        weighted = tuple(self.coefficient * value for value in gradient)
        core = -_divergence(weighted, self.relations, self.spacing)
        return jnp.where(
            all_liquid,
            jnp.where(self.liquid, core + mean, pressure),
            jnp.where(self.liquid, core, pressure),
        )


class SparseMACFreeSurfaceProjectionResult(StrictModule):
    velocity: tuple[Array, ...]
    pressure: Array
    liquid_mask: Array
    divergence_before: Array
    divergence_after: Array
    residual_norm: Array
    active_divergence_norm: Array
    energy_before: Array
    energy_after: Array
    liquid_count: Array
    air_count: Array
    linear: LinearSolveResult
    topology_complete: Array
    finite: Array
    successful: Array
    projection_id: str = eqx.field(static=True)


class SparseMACFreeSurfaceProjectionPlan(StrictModule, NonTrainableState):
    """Compact atmospheric pressure projection over sparse cell/face layouts."""

    transfer: PreparedSparseFLIPParticleTransfer
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    prepared_linear: PreparedLinearSolve
    operator_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedSparseFLIPParticleTransfer,
        /,
        *,
        density: float = 1.0,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ) -> None:
        if not isinstance(transfer, PreparedSparseFLIPParticleTransfer):
            raise TypeError("transfer must be PreparedSparseFLIPParticleTransfer.")
        density_ = float(density)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Projection density, tolerance, and iterations are invalid.")
        spacing = tuple(
            float(np.min(np.asarray(axis.interval_widths)))
            for axis in transfer.plan.index_space.structured_axes
        )
        cell_capacity = transfer.plan.cell_topology.storage_capacity
        face_capacities = tuple(
            value.storage_capacity for value in transfer.plan.face_topologies
        )
        dummy_relations = _SparseMACRelations(
            cell_lower_faces=tuple(
                jnp.zeros((cell_capacity,), dtype=jnp.int32) for _ in spacing
            ),
            cell_upper_faces=tuple(
                jnp.zeros((cell_capacity,), dtype=jnp.int32) for _ in spacing
            ),
            face_lower_cells=tuple(
                jnp.zeros((capacity,), dtype=jnp.int32) for capacity in face_capacities
            ),
            face_upper_cells=tuple(
                jnp.zeros((capacity,), dtype=jnp.int32) for capacity in face_capacities
            ),
            cell_valid=jnp.ones((cell_capacity,), dtype=bool),
            face_lower_supported=tuple(
                jnp.ones((capacity,), dtype=bool) for capacity in face_capacities
            ),
            face_upper_supported=tuple(
                jnp.ones((capacity,), dtype=bool) for capacity in face_capacities
            ),
            face_valid=tuple(
                jnp.ones((capacity,), dtype=bool) for capacity in face_capacities
            ),
            complete=jnp.ones((cell_capacity,), dtype=bool),
        )
        dtype = jnp.dtype(transfer.plan.precision.output_dtype)
        dummy_action = _SparseMACPressureAction(
            dummy_relations,
            jnp.ones((cell_capacity,), dtype=bool),
            jnp.asarray(1.0, dtype=dtype),
            jnp.ones((cell_capacity,), dtype=dtype),
            spacing,
        )
        operator_id = canonical_fingerprint(
            {
                "kind": "sparse-mac-free-surface-pressure-operator",
                "transfer": transfer.prepared_id,
                "cell_capacity": cell_capacity,
                "face_capacities": list(face_capacities),
            }
        )
        space = ArraySpace((cell_capacity,), dtype=dtype)
        operator = FunctionLinearOperator(
            dummy_action,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=operator_id,
        )
        problem_id = canonical_fingerprint(
            {"kind": "sparse-mac-free-surface-system", "operator": operator_id}
        )
        policy = (
            LinearSolvePolicy(
                ConjugateGradient(),
                tolerance=TolerancePolicy(
                    relative=0.1 * tolerance_,
                    absolute=0.1 * tolerance_,
                    max_steps=iterations,
                ),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.transfer = transfer
        self.density = density_
        self.tolerance = tolerance_
        self.spacing = spacing
        self.linear_policy = policy
        self.prepared_linear = prepare(
            LinearSystem(operator, problem_id=problem_id), policy
        )
        self.operator_id = operator_id
        self.problem_id = problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-mac-free-surface-projection",
                "transfer": transfer.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "linear": self.prepared_linear.plan.plan_id,
            }
        )

    def _relations(self, state: SparseFLIPTransferState) -> _SparseMACRelations:
        cell_topology = state.cell_topology
        cell_layout = self.transfer.plan.index_space.cells()
        cell_ids = cell_topology.logical_node_ids.reshape((-1,))
        cell_valid = cell_topology.node_valid.reshape((-1,))
        cell_coordinates, _ = cell_layout.integer_coordinates(cell_ids)
        cell_lower_faces = []
        cell_upper_faces = []
        face_lower_cells = []
        face_upper_cells = []
        face_lower_supported = []
        face_upper_supported = []
        face_valid = []
        complete = cell_valid
        for axis, (name, face_topology) in enumerate(
            zip(
                self.transfer.plan.index_space.axis_names,
                state.face_topologies,
                strict=True,
            )
        ):
            face_layout = self.transfer.plan.index_space.faces(name)
            lower_face_coordinates = cell_coordinates
            upper_face_coordinates = cell_coordinates.at[:, axis].add(1)
            if self.transfer.plan.index_space.periodic_axes[axis]:
                upper_face_coordinates = upper_face_coordinates.at[:, axis].set(
                    jnp.mod(upper_face_coordinates[:, axis], face_layout.shape[axis])
                )
            else:
                upper_face_coordinates = upper_face_coordinates.at[:, axis].set(
                    jnp.clip(
                        upper_face_coordinates[:, axis], 0, face_layout.shape[axis] - 1
                    )
                )
            lower_face_ids, lower_face_domain = face_layout.flat_indices(
                lower_face_coordinates
            )
            upper_face_ids, upper_face_domain = face_layout.flat_indices(
                upper_face_coordinates
            )
            lower_face_lookup = face_topology.lookup(
                lower_face_ids, cell_valid & lower_face_domain
            )
            upper_face_lookup = face_topology.lookup(
                upper_face_ids, cell_valid & upper_face_domain
            )
            cell_lower_faces.append(lower_face_lookup.storage_slots)
            cell_upper_faces.append(upper_face_lookup.storage_slots)
            complete = complete & (
                lower_face_lookup.supported & upper_face_lookup.supported
            )

            face_ids = face_topology.logical_node_ids.reshape((-1,))
            current_face_valid = face_topology.node_valid.reshape((-1,))
            face_coordinates, _ = face_layout.integer_coordinates(face_ids)
            lower_cell_coordinates = face_coordinates.at[:, axis].add(-1)
            upper_cell_coordinates = face_coordinates
            if self.transfer.plan.index_space.periodic_axes[axis]:
                lower_cell_coordinates = lower_cell_coordinates.at[:, axis].set(
                    jnp.mod(lower_cell_coordinates[:, axis], cell_layout.shape[axis])
                )
                upper_cell_coordinates = upper_cell_coordinates.at[:, axis].set(
                    jnp.mod(upper_cell_coordinates[:, axis], cell_layout.shape[axis])
                )
            else:
                lower_cell_coordinates = lower_cell_coordinates.at[:, axis].set(
                    jnp.clip(
                        lower_cell_coordinates[:, axis], 0, cell_layout.shape[axis] - 1
                    )
                )
                upper_cell_coordinates = upper_cell_coordinates.at[:, axis].set(
                    jnp.clip(
                        upper_cell_coordinates[:, axis], 0, cell_layout.shape[axis] - 1
                    )
                )
            lower_cell_ids, lower_cell_domain = cell_layout.flat_indices(
                lower_cell_coordinates
            )
            upper_cell_ids, upper_cell_domain = cell_layout.flat_indices(
                upper_cell_coordinates
            )
            lower_cell_lookup = cell_topology.lookup(
                lower_cell_ids, current_face_valid & lower_cell_domain
            )
            upper_cell_lookup = cell_topology.lookup(
                upper_cell_ids, current_face_valid & upper_cell_domain
            )
            face_lower_cells.append(lower_cell_lookup.storage_slots)
            face_upper_cells.append(upper_cell_lookup.storage_slots)
            face_lower_supported.append(lower_cell_lookup.supported)
            face_upper_supported.append(upper_cell_lookup.supported)
            face_valid.append(current_face_valid)
        return _SparseMACRelations(
            cell_lower_faces=tuple(cell_lower_faces),
            cell_upper_faces=tuple(cell_upper_faces),
            face_lower_cells=tuple(face_lower_cells),
            face_upper_cells=tuple(face_upper_cells),
            face_lower_supported=tuple(face_lower_supported),
            face_upper_supported=tuple(face_upper_supported),
            cell_valid=cell_valid,
            face_valid=tuple(face_valid),
            complete=complete,
        )

    def project(
        self,
        state: SparseFLIPTransferState,
        velocity: tuple[ArrayLike, ...],
        liquid_mask: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> SparseMACFreeSurfaceProjectionResult:
        if not isinstance(state, SparseFLIPTransferState):
            raise TypeError("state must be SparseFLIPTransferState.")
        values = tuple(jnp.asarray(value) for value in velocity)
        expected = tuple(
            topology.plan.storage_capacity for topology in state.face_topologies
        )
        if len(values) != len(expected) or any(
            value.shape != (size,) for value, size in zip(values, expected, strict=True)
        ):
            raise ValueError("velocity must contain one compact field per face axis.")
        relations = self._relations(state)
        liquid = jnp.asarray(liquid_mask, dtype=bool)
        if liquid.shape != relations.cell_valid.shape:
            raise ValueError("liquid_mask must match compact cell storage.")
        liquid = liquid & relations.cell_valid
        dtype = values[0].dtype
        dt = jnp.asarray(step_size, dtype=dtype).reshape(())
        incoming = (
            jnp.zeros(liquid.shape, dtype=dtype)
            if pressure is None
            else jnp.asarray(pressure, dtype=dtype)
        )
        if incoming.shape != liquid.shape:
            raise ValueError("pressure must match compact cell storage.")
        cell_measure, measure_valid = self.transfer.plan.cell_topology.layout.measure_at(
            state.cell_topology.logical_node_ids.reshape((-1,))
        )
        cell_measure = jnp.where(
            relations.cell_valid & measure_valid, cell_measure.astype(dtype), 1.0
        )
        coefficient = dt / self.density
        divergence_before = _divergence(values, relations, self.spacing)
        all_liquid = jnp.all(liquid | ~relations.cell_valid)
        any_liquid = jnp.any(liquid)
        topology_complete = jnp.all(~liquid | relations.complete)

        def gauge(field):
            denominator = jnp.sum(jnp.where(liquid, cell_measure, 0.0))
            mean = jnp.where(
                denominator > 0.0,
                jnp.sum(cell_measure * jnp.where(liquid, field, 0.0))
                / jnp.where(denominator > 0.0, denominator, 1.0),
                0.0,
            )
            return jnp.where(liquid, field - mean, 0.0)

        rhs = jnp.where(liquid, -divergence_before, 0.0)
        rhs = jnp.where(all_liquid, gauge(rhs), rhs)
        initial = jnp.where(all_liquid, gauge(incoming), jnp.where(liquid, incoming, 0.0))
        action = _SparseMACPressureAction(
            relations,
            liquid,
            coefficient,
            cell_measure,
            self.spacing,
        )
        space = self.prepared_linear.problem.operator.source
        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=self.operator_id,
        )
        prepared = refresh(
            self.prepared_linear,
            LinearSystem(operator, problem_id=self.problem_id),
        )
        linear = solve(prepared, rhs, initial_guess=initial)
        pressure_candidate = jnp.where(
            all_liquid,
            gauge(linear.value),
            jnp.where(liquid, linear.value, 0.0),
        )
        gradient = _gradient(
            pressure_candidate,
            liquid,
            relations,
            self.spacing,
        )
        corrected = tuple(
            jnp.where(valid, value - coefficient * derivative, 0.0)
            for value, derivative, valid in zip(
                values, gradient, relations.face_valid, strict=True
            )
        )
        divergence_after = _divergence(corrected, relations, self.spacing)
        residual = action(pressure_candidate) - rhs
        residual_norm = jnp.sqrt(jnp.sum(cell_measure * residual**2))
        active_divergence_norm = jnp.sqrt(
            jnp.sum(cell_measure * jnp.where(liquid, divergence_after, 0.0) ** 2)
        )
        energy_before = (
            0.5
            * self.density
            * sum(
                jnp.sum(jnp.where(valid, value * value, 0.0))
                for value, valid in zip(values, relations.face_valid, strict=True)
            )
        )
        energy_after = (
            0.5
            * self.density
            * sum(
                jnp.sum(jnp.where(valid, value * value, 0.0))
                for value, valid in zip(corrected, relations.face_valid, strict=True)
            )
        )
        rhs_norm = jnp.sqrt(jnp.sum(cell_measure * rhs**2))
        finite = (
            jnp.isfinite(dt)
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(pressure_candidate))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in corrected))
            )
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(active_divergence_norm)
        )
        converged = linear.successful & (
            residual_norm <= self.tolerance * jnp.maximum(1.0, rhs_norm)
        )
        successful = (
            state.successful & topology_complete & any_liquid & finite & converged
        )
        return SparseMACFreeSurfaceProjectionResult(
            velocity=corrected,
            pressure=pressure_candidate,
            liquid_mask=liquid,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            residual_norm=residual_norm,
            active_divergence_norm=active_divergence_norm,
            energy_before=energy_before,
            energy_after=energy_after,
            liquid_count=jnp.sum(liquid, dtype=jnp.int32),
            air_count=jnp.sum(relations.cell_valid & ~liquid, dtype=jnp.int32),
            linear=linear,
            topology_complete=topology_complete,
            finite=finite,
            successful=successful,
            projection_id=self.plan_id,
        )


__all__ = [
    "SparseMACFreeSurfaceProjectionPlan",
    "SparseMACFreeSurfaceProjectionResult",
]
