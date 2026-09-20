#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...linalg import DenseLinearOperator, matrix_exponential_action


class EvolvingFlowCellState(StrictModule):
    vectors: Array
    deformation_gradient: Array
    lattice_transform: Array
    time: Array
    step_index: Array
    remap_count: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _with_plan_id(state: EvolvingFlowCellState, plan_id: str, /) -> EvolvingFlowCellState:
    return EvolvingFlowCellState(
        state.vectors,
        state.deformation_gradient,
        state.lattice_transform,
        state.time,
        state.step_index,
        state.remap_count,
        state.successful,
        plan_id,
    )


class EvolvingFlowCellStepResult(StrictModule):
    candidate_state: EvolvingFlowCellState
    accepted_state: EvolvingFlowCellState
    volume: Array
    condition_number: Array
    incompressibility_residual: Array
    exponential_converged: Array
    accepted: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class EvolvingFlowCellPlan(StrictModule, NonTrainableState):
    cell: PeriodicCell
    maximum_condition_number: float = eqx.field(static=True)
    maximum_relative_volume_error: float = eqx.field(static=True)
    incompressibility_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        /,
        *,
        maximum_condition_number: float = 1.0e4,
        maximum_relative_volume_error: float = 1.0e-9,
        incompressibility_tolerance: float = 1.0e-10,
    ):
        if (
            not isinstance(cell, PeriodicCell)
            or not cell.fully_periodic
            or cell.rank != 3
        ):
            raise ValueError(
                "Evolving flow requires a fully periodic three-dimensional cell."
            )
        condition = float(maximum_condition_number)
        volume_error = float(maximum_relative_volume_error)
        trace_tolerance = float(incompressibility_tolerance)
        if (
            not math.isfinite(condition)
            or condition < cell.condition_number
            or not math.isfinite(volume_error)
            or volume_error <= 0.0
            or not math.isfinite(trace_tolerance)
            or trace_tolerance < 0.0
        ):
            raise ValueError("Evolving-cell certification tolerances are invalid.")
        self.cell = cell
        self.maximum_condition_number = condition
        self.maximum_relative_volume_error = volume_error
        self.incompressibility_tolerance = trace_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "evolving-flow-cell",
                "cell": cell.cell_id,
                "maximum_condition_number": condition,
                "maximum_relative_volume_error": volume_error,
                "incompressibility_tolerance": trace_tolerance,
            }
        )

    def initialize(self) -> EvolvingFlowCellState:
        dtype = self.cell.vectors.dtype
        return EvolvingFlowCellState(
            self.cell.vectors,
            jnp.eye(3, dtype=dtype),
            jnp.eye(3, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
            self.plan_id,
        )

    def step(
        self,
        state: EvolvingFlowCellState,
        velocity_gradient: ArrayLike,
        time_step: float,
        /,
    ) -> EvolvingFlowCellStepResult:
        if not isinstance(state, EvolvingFlowCellState) or state.plan_id != self.plan_id:
            raise ValueError("Flow-cell state does not belong to this plan.")
        gradient = jnp.asarray(velocity_gradient, dtype=state.vectors.dtype)
        step = float(time_step)
        if gradient.shape != (3, 3):
            raise ValueError("velocity_gradient must have shape (3, 3).")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        operator = DenseLinearOperator(gradient)

        def evolve(column):
            result = matrix_exponential_action(operator, column, step)
            return result.value, result.converged

        evolved_vectors, vector_converged = jax.vmap(evolve, in_axes=1, out_axes=(1, 0))(
            state.vectors.T
        )
        candidate_vectors = evolved_vectors.T
        candidate_deformation, deformation_converged = jax.vmap(
            evolve, in_axes=1, out_axes=(1, 0)
        )(state.deformation_gradient)
        volume = jnp.abs(jnp.linalg.det(candidate_vectors))
        singular = jnp.linalg.svd(candidate_vectors, compute_uv=False)
        condition = singular[0] / singular[-1]
        incompressibility = jnp.abs(jnp.trace(gradient))
        relative_volume = jnp.abs(volume - self.cell.volume) / self.cell.volume
        exponential_converged = jnp.all(vector_converged) & jnp.all(deformation_converged)
        finite = (
            jnp.all(jnp.isfinite(candidate_vectors))
            & jnp.all(jnp.isfinite(candidate_deformation))
            & jnp.isfinite(condition)
            & jnp.isfinite(volume)
        )
        accepted = (
            state.successful
            & finite
            & exponential_converged
            & (condition <= self.maximum_condition_number)
            & (incompressibility <= self.incompressibility_tolerance)
            & (relative_volume <= self.maximum_relative_volume_error)
        )
        candidate = EvolvingFlowCellState(
            candidate_vectors,
            candidate_deformation,
            state.lattice_transform,
            state.time + step,
            state.step_index + 1,
            state.remap_count,
            state.successful & accepted,
            self.plan_id,
        )
        accepted_state = EvolvingFlowCellState(
            jnp.where(accepted, candidate.vectors, state.vectors),
            jnp.where(
                accepted, candidate.deformation_gradient, state.deformation_gradient
            ),
            state.lattice_transform,
            jnp.where(accepted, candidate.time, state.time),
            jnp.where(accepted, candidate.step_index, state.step_index),
            state.remap_count,
            state.successful,
            self.plan_id,
        )
        return EvolvingFlowCellStepResult(
            candidate,
            accepted_state,
            volume,
            condition,
            incompressibility,
            exponential_converged,
            accepted,
            accepted,
            self.plan_id,
        )


class FlowCellRemapResult(StrictModule):
    candidate_state: EvolvingFlowCellState
    accepted_state: EvolvingFlowCellState
    candidate_positions: Array
    accepted_positions: Array
    candidate_image_counts: Array
    accepted_image_counts: Array
    remap_matrix: Array
    cartesian_residual: Array
    unimodularity_residual: Array
    remapped: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _apply_remap(
    cell_plan: EvolvingFlowCellPlan,
    state: EvolvingFlowCellState,
    positions: Array,
    image_counts: Array,
    remap_matrix: Array,
    remapped: Array,
    /,
) -> FlowCellRemapResult:
    matrix = jnp.asarray(remap_matrix, dtype=jnp.int32)
    determinant = jnp.linalg.det(matrix.astype(state.vectors.dtype))
    unimodularity = jnp.abs(jnp.abs(determinant) - 1.0)
    unwrapped = positions + contract(
        "ni,ij->nj", image_counts.astype(positions.dtype), state.vectors
    )
    candidate_vectors = contract(
        "ij,jk->ik", matrix.astype(state.vectors.dtype), state.vectors
    )
    wrapped, images = cell_plan.cell.wrap_with_vectors(unwrapped, candidate_vectors)
    reconstructed = wrapped + contract(
        "ni,ij->nj", images.astype(wrapped.dtype), candidate_vectors
    )
    cartesian_residual = jnp.max(jnp.abs(reconstructed - unwrapped))
    tolerance = (
        256.0
        * jnp.finfo(positions.dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(unwrapped)))
    )
    accepted = (
        remapped
        & state.successful
        & (unimodularity == 0.0)
        & (cartesian_residual <= tolerance)
        & jnp.all(jnp.isfinite(candidate_vectors))
    )
    candidate = EvolvingFlowCellState(
        candidate_vectors,
        state.deformation_gradient,
        contract("ij,jk->ik", matrix, state.lattice_transform),
        state.time,
        state.step_index,
        state.remap_count + remapped.astype(jnp.int32),
        state.successful & accepted,
        state.plan_id,
    )
    accepted_state = EvolvingFlowCellState(
        jnp.where(accepted, candidate.vectors, state.vectors),
        state.deformation_gradient,
        jnp.where(accepted, candidate.lattice_transform, state.lattice_transform),
        state.time,
        state.step_index,
        jnp.where(accepted, candidate.remap_count, state.remap_count),
        state.successful,
        state.plan_id,
    )
    return FlowCellRemapResult(
        candidate,
        accepted_state,
        wrapped,
        jnp.where(accepted, wrapped, positions),
        images,
        jnp.where(accepted, images, image_counts),
        matrix,
        cartesian_residual,
        unimodularity,
        remapped,
        state.successful & (~remapped | accepted),
        state.plan_id,
    )


class LeesEdwardsRemapPlan(StrictModule, NonTrainableState):
    cell_plan: EvolvingFlowCellPlan
    flow_axis: int = eqx.field(static=True)
    gradient_axis: int = eqx.field(static=True)
    maximum_reduced_tilt: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_plan: EvolvingFlowCellPlan,
        /,
        *,
        flow_axis: int = 0,
        gradient_axis: int = 1,
        maximum_reduced_tilt: float = 0.5,
    ):
        if not isinstance(cell_plan, EvolvingFlowCellPlan):
            raise TypeError("cell_plan must be EvolvingFlowCellPlan.")
        flow = int(flow_axis)
        gradient = int(gradient_axis)
        maximum = float(maximum_reduced_tilt)
        if flow not in range(3) or gradient not in range(3) or flow == gradient:
            raise ValueError("Lees-Edwards axes must be distinct three-dimensional axes.")
        if not math.isfinite(maximum) or not 0.0 < maximum <= 0.5:
            raise ValueError("maximum_reduced_tilt must lie in (0, 0.5].")
        self.cell_plan = cell_plan
        self.flow_axis = flow
        self.gradient_axis = gradient
        self.maximum_reduced_tilt = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lees-edwards-remap",
                "cell_plan": cell_plan.plan_id,
                "flow_axis": flow,
                "gradient_axis": gradient,
                "maximum_reduced_tilt": maximum,
            }
        )

    def apply(
        self,
        state: EvolvingFlowCellState,
        positions: ArrayLike,
        image_counts: ArrayLike,
        /,
    ) -> FlowCellRemapResult:
        coordinate = jnp.asarray(positions)
        images = jnp.asarray(image_counts, dtype=jnp.int32)
        flow_length = state.vectors[self.flow_axis, self.flow_axis]
        tilt = state.vectors[self.gradient_axis, self.flow_axis]
        reduced = tilt / flow_length
        shift = jnp.rint(reduced).astype(jnp.int32)
        should_remap = jnp.abs(reduced) > self.maximum_reduced_tilt
        matrix = (
            jnp.eye(3, dtype=jnp.int32).at[self.gradient_axis, self.flow_axis].set(-shift)
        )
        result = _apply_remap(
            self.cell_plan,
            state,
            coordinate,
            images,
            matrix,
            should_remap,
        )
        return FlowCellRemapResult(
            result.candidate_state,
            result.accepted_state,
            result.candidate_positions,
            result.accepted_positions,
            result.candidate_image_counts,
            result.accepted_image_counts,
            result.remap_matrix,
            result.cartesian_residual,
            result.unimodularity_residual,
            result.remapped,
            result.successful,
            self.plan_id,
        )


class GeneralizedKraynikReineltPlan(StrictModule, NonTrainableState):
    cell_plan: EvolvingFlowCellPlan
    velocity_gradient: Array
    period: float = eqx.field(static=True)
    remap_matrix: Array
    certification_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_vectors: ArrayLike,
        velocity_gradient: ArrayLike,
        period: float,
        remap_matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_condition_number: float = 7.0,
    ):
        vectors = np.asarray(reference_vectors, dtype=np.float64)
        gradient = np.asarray(velocity_gradient, dtype=np.float64)
        matrix = np.asarray(remap_matrix, dtype=np.int64)
        recurrence = float(period)
        tolerance_ = float(tolerance)
        condition_limit = float(maximum_condition_number)
        if vectors.shape != (3, 3) or gradient.shape != (3, 3) or matrix.shape != (3, 3):
            raise ValueError("Generalized KR arrays must all have shape (3, 3).")
        if (
            not math.isfinite(condition_limit)
            or condition_limit < float(np.linalg.cond(vectors))
            or condition_limit > 16.0
        ):
            raise ValueError(
                "Generalized KR maximum_condition_number must cover the reference "
                "lattice and not exceed the bounded value 16."
            )
        determinant = round(float(np.linalg.det(matrix)))
        if determinant not in (-1, 1) or not np.array_equal(
            matrix @ np.rint(np.linalg.inv(matrix)).astype(np.int64),
            np.eye(3, dtype=np.int64),
        ):
            raise ValueError("remap_matrix must be an exactly unimodular integer matrix.")
        if not math.isfinite(recurrence) or recurrence <= 0.0:
            raise ValueError("KR period must be finite and positive.")
        if abs(float(np.trace(gradient))) > tolerance_:
            raise ValueError("Generalized KR flow must be incompressible.")
        operator = DenseLinearOperator(jnp.asarray(gradient))
        actions = tuple(
            matrix_exponential_action(
                operator, jnp.asarray(vectors.T[:, column]), recurrence
            )
            for column in range(3)
        )
        if not all(bool(np.asarray(action.converged)) for action in actions):
            raise ValueError("Native KR recurrence exponential did not converge.")
        evolved = np.asarray(
            jnp.stack(tuple(action.value for action in actions), axis=1).T
        )
        residual = float(
            np.linalg.norm(matrix @ evolved - vectors)
            / max(np.linalg.norm(vectors), np.finfo(np.float64).tiny)
        )
        if not math.isfinite(residual) or residual > tolerance_:
            raise ValueError("Reference lattice, flow, period, and remap do not recur.")
        image_extent = max(1, math.ceil(0.5 + condition_limit * math.sqrt(3.0)))
        cell = PeriodicCell(
            vectors,
            maximum_condition_number=condition_limit,
            maximum_image_count=(2 * image_extent + 1) ** 3,
        )
        self.cell_plan = EvolvingFlowCellPlan(
            cell,
            maximum_condition_number=condition_limit,
            maximum_relative_volume_error=max(tolerance_, 1.0e-12),
            incompressibility_tolerance=tolerance_,
        )
        self.velocity_gradient = jnp.asarray(gradient, dtype=cell.vectors.dtype)
        self.period = recurrence
        self.remap_matrix = jnp.asarray(matrix, dtype=jnp.int32)
        self.certification_residual = residual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "generalized-kraynik-reinelt",
                "arrays": array_tree_fingerprint((vectors, gradient, matrix)),
                "period": recurrence,
                "residual": residual,
                "cell_plan": self.cell_plan.plan_id,
            }
        )

    def initialize(self) -> EvolvingFlowCellState:
        state = self.cell_plan.initialize()
        return EvolvingFlowCellState(
            state.vectors,
            state.deformation_gradient,
            state.lattice_transform,
            state.time,
            state.step_index,
            state.remap_count,
            state.successful,
            self.plan_id,
        )

    def step(
        self, state: EvolvingFlowCellState, time_step: float, /
    ) -> EvolvingFlowCellStepResult:
        local_state = EvolvingFlowCellState(
            state.vectors,
            state.deformation_gradient,
            state.lattice_transform,
            state.time,
            state.step_index,
            state.remap_count,
            state.successful,
            self.cell_plan.plan_id,
        )
        result = self.cell_plan.step(local_state, self.velocity_gradient, time_step)
        candidate = _with_plan_id(result.candidate_state, self.plan_id)
        accepted = _with_plan_id(result.accepted_state, self.plan_id)
        return EvolvingFlowCellStepResult(
            candidate,
            accepted,
            result.volume,
            result.condition_number,
            result.incompressibility_residual,
            result.exponential_converged,
            result.accepted,
            result.successful,
            self.plan_id,
        )

    def remap_if_due(
        self,
        state: EvolvingFlowCellState,
        positions: ArrayLike,
        image_counts: ArrayLike,
        /,
    ) -> FlowCellRemapResult:
        target_count = jnp.floor(
            (state.time + 128.0 * jnp.finfo(state.time.dtype).eps) / self.period
        ).astype(jnp.int32)
        due = target_count > state.remap_count
        no_skip = target_count <= state.remap_count + 1
        local_state = EvolvingFlowCellState(
            state.vectors,
            state.deformation_gradient,
            state.lattice_transform,
            state.time,
            state.step_index,
            state.remap_count,
            state.successful & no_skip,
            self.cell_plan.plan_id,
        )
        result = _apply_remap(
            self.cell_plan,
            local_state,
            jnp.asarray(positions),
            jnp.asarray(image_counts, dtype=jnp.int32),
            self.remap_matrix,
            due,
        )
        candidate = _with_plan_id(result.candidate_state, self.plan_id)
        accepted = _with_plan_id(result.accepted_state, self.plan_id)
        return FlowCellRemapResult(
            candidate,
            accepted,
            result.candidate_positions,
            result.accepted_positions,
            result.candidate_image_counts,
            result.accepted_image_counts,
            result.remap_matrix,
            result.cartesian_residual,
            result.unimodularity_residual,
            result.remapped,
            result.successful,
            self.plan_id,
        )


class PlanarKraynikReineltPlan(StrictModule, NonTrainableState):
    generalized: GeneralizedKraynikReineltPlan
    extensional_rate: float = eqx.field(static=True)
    planar_area: float = eqx.field(static=True)
    transverse_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        extensional_rate: float,
        planar_area: float,
        transverse_length: float,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_condition_number: float = 7.0,
    ):
        rate = float(extensional_rate)
        area = float(planar_area)
        length = float(transverse_length)
        if any(
            not math.isfinite(value) or value <= 0.0 for value in (rate, area, length)
        ):
            raise ValueError(
                "Planar KR rate and cell scales must be finite and positive."
            )
        automorphism = np.asarray([[2, 1], [1, 1]], dtype=np.int64)
        eigenvalues, eigenvectors = np.linalg.eigh(automorphism.astype("float64"))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        planar = eigenvectors
        planar *= math.sqrt(area / abs(float(np.linalg.det(planar))))
        vectors = np.eye(3)
        vectors[:2, :2] = planar
        vectors[2, 2] = length
        gradient = np.diag([rate, -rate, 0.0])
        period = math.log(float(eigenvalues[0])) / rate
        remap = np.eye(3, dtype=np.int64)
        remap[:2, :2] = np.rint(np.linalg.inv(automorphism)).astype(np.int64)
        generalized = GeneralizedKraynikReineltPlan(
            vectors,
            gradient,
            period,
            remap,
            tolerance=tolerance,
            maximum_condition_number=maximum_condition_number,
        )
        self.generalized = generalized
        self.extensional_rate = rate
        self.planar_area = area
        self.transverse_length = length
        self.plan_id = canonical_fingerprint(
            {
                "kind": "planar-kraynik-reinelt",
                "generalized": generalized.plan_id,
                "rate": rate,
                "area": area,
                "transverse_length": length,
            }
        )

    def initialize(self) -> EvolvingFlowCellState:
        return _with_plan_id(self.generalized.initialize(), self.plan_id)

    def step(
        self, state: EvolvingFlowCellState, time_step: float, /
    ) -> EvolvingFlowCellStepResult:
        generalized_state = _with_plan_id(state, self.generalized.plan_id)
        result = self.generalized.step(generalized_state, time_step)
        candidate = _with_plan_id(result.candidate_state, self.plan_id)
        accepted = _with_plan_id(result.accepted_state, self.plan_id)
        return EvolvingFlowCellStepResult(
            candidate,
            accepted,
            result.volume,
            result.condition_number,
            result.incompressibility_residual,
            result.exponential_converged,
            result.accepted,
            result.successful,
            self.plan_id,
        )

    def remap_if_due(
        self,
        state: EvolvingFlowCellState,
        positions: ArrayLike,
        image_counts: ArrayLike,
        /,
    ) -> FlowCellRemapResult:
        generalized_state = _with_plan_id(state, self.generalized.plan_id)
        result = self.generalized.remap_if_due(generalized_state, positions, image_counts)
        candidate = _with_plan_id(result.candidate_state, self.plan_id)
        accepted = _with_plan_id(result.accepted_state, self.plan_id)
        return FlowCellRemapResult(
            candidate,
            accepted,
            result.candidate_positions,
            result.accepted_positions,
            result.candidate_image_counts,
            result.accepted_image_counts,
            result.remap_matrix,
            result.cartesian_residual,
            result.unimodularity_residual,
            result.remapped,
            result.successful,
            self.plan_id,
        )


__all__ = [
    "EvolvingFlowCellPlan",
    "EvolvingFlowCellState",
    "EvolvingFlowCellStepResult",
    "FlowCellRemapResult",
    "GeneralizedKraynikReineltPlan",
    "LeesEdwardsRemapPlan",
    "PlanarKraynikReineltPlan",
]
