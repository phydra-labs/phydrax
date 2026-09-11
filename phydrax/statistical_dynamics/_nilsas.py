#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import (
    EvolutionArgumentJacobianAction,
    EvolutionJacobianAction,
    EvolutionTrajectory,
)
from ..dynamics._grid import IterationGrid, TimeGrid
from ..dynamics.analysis import ShadowingSensitivityProblem
from ..dynamics.analysis._shadowing import _quadrature_weights
from ._shadowing_solve import (
    AbstractShadowingSolvePlan,
    orthonormalize_shadowing_basis,
    shadowing_plan_id,
    ShadowingMemoryMode,
    ShadowingSolveCost,
    ShadowingSolveStatus,
    solve_reduced_shadowing,
    validate_shadowing_plan,
)


def _tree_add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_finite(tree: PyTree[Array], /) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _observable_state_gradient(
    problem: ShadowingSensitivityProblem,
    coordinate: Array,
    state: Array,
    args: Any,
    /,
) -> Array:
    if problem.observable_state_gradient is None:
        return jax.grad(lambda value: problem.observable(coordinate, value, args))(state)
    return jnp.asarray(problem.observable_state_gradient(coordinate, state, args))


def _observable_parameter_pullback(
    problem: ShadowingSensitivityProblem,
    coordinate: Array,
    state: Array,
    args: PyTree[Array],
    weight: Array,
    /,
) -> PyTree[Array]:
    value, pullback = jax.vjp(
        lambda current_args: problem.observable(coordinate, state, current_args),
        args,
    )
    if jnp.asarray(value).shape != ():
        raise ValueError("Shadowing observable must return a scalar.")
    return pullback(weight)[0]


def _transpose_columns(action: EvolutionJacobianAction, basis: Array, /) -> Array:
    if basis.shape[1] == 0:
        return basis
    columns = eqx.filter_vmap(action.transpose_mv)(basis.T)
    return columns.T


class NILSASPlan(AbstractShadowingSolvePlan):
    """Segmented non-intrusive least-squares adjoint shadowing policy."""

    method: str = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    unstable_dimension: int = eqx.field(static=True)
    basis_dimension: int = eqx.field(static=True)
    segment_steps: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    memory_mode: ShadowingMemoryMode = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_dimension: int,
        unstable_dimension: int,
        basis_dimension: int,
        segment_steps: int,
        segment_count: int,
        /,
        *,
        regularization: float = 0.0,
        rank_tolerance: float = 1.0e-10,
        memory_mode: ShadowingMemoryMode = "store",
        maximum_retained_bytes: int = 2 * 1024 * 1024 * 1024,
        maximum_workspace_bytes: int = 4 * 1024 * 1024 * 1024,
    ):
        values = validate_shadowing_plan(
            "nilsas",
            state_dimension,
            unstable_dimension,
            basis_dimension,
            segment_steps,
            segment_count,
            regularization,
            rank_tolerance,
            memory_mode,
            maximum_retained_bytes,
            maximum_workspace_bytes,
        )
        (
            self.state_dimension,
            self.unstable_dimension,
            self.basis_dimension,
            self.segment_steps,
            self.segment_count,
            self.regularization,
            self.rank_tolerance,
            self.maximum_retained_bytes,
            self.maximum_workspace_bytes,
        ) = values
        self.method = "nilsas"
        self.memory_mode = memory_mode
        self.plan_id = shadowing_plan_id("nilsas", values, memory_mode)

    def prepare(
        self,
        problem: ShadowingSensitivityProblem,
        trajectory: EvolutionTrajectory,
        /,
        *,
        args: PyTree[Array],
        terminal_basis: ArrayLike | None = None,
        key: Array | None = None,
    ) -> "PreparedNILSAS":
        if not isinstance(problem, ShadowingSensitivityProblem):
            raise TypeError("problem must be a ShadowingSensitivityProblem.")
        if not isinstance(trajectory, EvolutionTrajectory):
            raise TypeError("trajectory must be an EvolutionTrajectory.")
        if problem.evolution.eventful:
            raise ValueError("NILSAS requires event-free evolution segments.")
        if problem.evolution.stochastic:
            raise ValueError("NILSAS requires deterministic evolution segments.")
        if trajectory.evolution_id != problem.evolution.evolution_id:
            raise ValueError("Trajectory and shadowing evolution IDs must match.")
        if trajectory.grid.num_steps != self.horizon_steps:
            raise ValueError("Trajectory horizon does not match the NILSAS plan.")
        if trajectory.state_layout.shape != (self.state_dimension,):
            raise ValueError("NILSAS requires the declared one-dimensional state vector.")
        if not trajectory.state_layout.geometry.trivial:
            raise ValueError("NILSAS initially requires Euclidean state geometry.")
        if not jnp.issubdtype(trajectory.states.dtype, jnp.floating):
            raise TypeError("NILSAS requires a real floating trajectory.")
        argument_leaves = jax.tree.leaves(args)
        if not argument_leaves or any(
            not eqx.is_inexact_array(leaf) for leaf in argument_leaves
        ):
            raise TypeError("NILSAS args must be a nonempty PyTree of inexact arrays.")
        if isinstance(trajectory.grid, IterationGrid):
            if problem.time_dilation != "none":
                raise ValueError("Discrete-map NILSAS does not use flow time dilation.")
        elif isinstance(trajectory.grid, TimeGrid):
            if problem.time_dilation != "flow":
                raise ValueError("Time-grid NILSAS requires flow time dilation.")
            if self.basis_dimension < self.unstable_dimension + 1:
                raise ValueError(
                    "Flow NILSAS basis_dimension must include the neutral adjoint."
                )
        else:
            raise TypeError("NILSAS requires an IterationGrid or TimeGrid trajectory.")

        if self.basis_dimension == 0:
            if terminal_basis is not None or key is not None:
                raise ValueError(
                    "A zero-dimensional NILSAS basis accepts no initializer."
                )
            basis = jnp.zeros((self.state_dimension, 0), dtype=trajectory.states.dtype)
            terminal_defect = jnp.asarray(0.0, dtype=trajectory.states.dtype)
        else:
            if (terminal_basis is None) == (key is None):
                raise ValueError("Provide exactly one of terminal_basis and key.")
            basis = (
                jax.random.normal(
                    key,
                    (self.state_dimension, self.basis_dimension),
                    dtype=trajectory.states.dtype,
                )
                if terminal_basis is None
                else jnp.asarray(terminal_basis, dtype=trajectory.states.dtype)
            )
            if basis.shape != (self.state_dimension, self.basis_dimension):
                raise ValueError("terminal_basis has an incompatible shape.")
            basis, _, terminal_defect, basis_valid = orthonormalize_shadowing_basis(
                basis,
                rank_tolerance=self.rank_tolerance,
            )
            if not bool(np.asarray(basis_valid)):
                raise ValueError("NILSAS terminal basis lost numerical rank.")

        itemsize = np.dtype(trajectory.states.dtype).itemsize
        horizon = self.horizon_steps
        boundary_storage = (
            self.segment_count * self.state_dimension * (self.basis_dimension + 1)
        )
        path_storage = (
            horizon * self.state_dimension * 2 * (self.basis_dimension + 1)
            if self.memory_mode == "store"
            else 0
        )
        reduced_storage = self.segment_count * (
            self.basis_dimension**2 + 3 * self.basis_dimension + 1
        )
        retained = itemsize * (boundary_storage + path_storage + reduced_storage)
        variables = self.segment_count * self.basis_dimension
        constraints = max(self.segment_count - 1, 0) * self.basis_dimension
        if problem.time_dilation == "flow":
            constraints += 1
        workspace = retained + itemsize * (variables + constraints) ** 2
        if retained > self.maximum_retained_bytes:
            raise MemoryError("NILSAS state exceeds maximum_retained_bytes.")
        if workspace > self.maximum_workspace_bytes:
            raise MemoryError("NILSAS reduced solve exceeds maximum_workspace_bytes.")
        cost = ShadowingSolveCost(
            method=self.method,
            state_dimension=self.state_dimension,
            unstable_dimension=self.unstable_dimension,
            basis_dimension=self.basis_dimension,
            horizon_steps=horizon,
            segment_count=self.segment_count,
            input_trajectory_bytes=int(trajectory.states.size) * itemsize,
            retained_bytes=retained,
            workspace_bytes=workspace,
            maximum_retained_bytes=self.maximum_retained_bytes,
            maximum_workspace_bytes=self.maximum_workspace_bytes,
        )
        return PreparedNILSAS(
            plan=self,
            cost=cost,
            problem=problem,
            trajectory=trajectory,
            args=args,
            terminal_basis=basis,
            terminal_basis_defect=terminal_defect,
        )


class PreparedNILSAS(StrictModule, NonTrainableState):
    plan: NILSASPlan
    cost: ShadowingSolveCost
    problem: ShadowingSensitivityProblem
    trajectory: EvolutionTrajectory
    args: PyTree[Array]
    terminal_basis: Array
    terminal_basis_defect: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: NILSASPlan,
        cost: ShadowingSolveCost,
        problem: ShadowingSensitivityProblem,
        trajectory: EvolutionTrajectory,
        args: PyTree[Array],
        terminal_basis: Array,
        terminal_basis_defect: ArrayLike,
    ):
        self.plan = plan
        self.cost = cost
        self.problem = problem
        self.trajectory = trajectory
        self.args = args
        self.terminal_basis = terminal_basis
        self.terminal_basis_defect = jnp.asarray(terminal_basis_defect)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-nilsas",
                "plan": plan.plan_id,
                "problem": problem.problem_id,
                "trajectory": trajectory.evolution_id,
                "args": array_tree_fingerprint(args),
                "terminal_basis": array_tree_fingerprint(terminal_basis),
            }
        )

    @eqx.filter_jit
    def solve(self, /) -> "NILSASResult":
        plan = self.plan
        problem = self.problem
        trajectory = self.trajectory
        weights = _quadrature_weights(trajectory)
        dtype = trajectory.states.dtype
        n = plan.state_dimension
        m = plan.basis_dimension
        horizon = plan.horizon_steps
        segments = plan.segment_count

        objective_values = jax.vmap(
            lambda coordinate, state: jnp.asarray(
                problem.observable(coordinate, state, self.args)
            )
        )(trajectory.grid.coordinates, trajectory.states)
        if objective_values.shape != (horizon + 1,):
            raise ValueError("NILSAS observable must return one scalar per node.")
        terminal_gradient = _observable_state_gradient(
            problem,
            trajectory.grid.coordinates[-1],
            trajectory.states[-1],
            self.args,
        )
        basis = self.terminal_basis
        inhomogeneous = weights[-1] * terminal_gradient
        terminal_basis_seeds = jnp.zeros((segments, n, m), dtype=dtype)
        terminal_inhomogeneous_seeds = jnp.zeros((segments, n), dtype=dtype)
        covariances = jnp.zeros((segments, m, m), dtype=dtype)
        linear_terms = jnp.zeros((segments, m), dtype=dtype)
        neutral_basis = jnp.zeros((segments, m), dtype=dtype)
        neutral_offsets = jnp.zeros((segments,), dtype=dtype)
        relations = jnp.zeros((max(segments - 1, 0), m, m), dtype=dtype)
        offsets = jnp.zeros((max(segments - 1, 0), m), dtype=dtype)
        endpoint_bases = jnp.zeros((horizon, n, m), dtype=dtype)
        endpoint_inhomogeneous = jnp.zeros((horizon, n), dtype=dtype)
        node_bases = jnp.zeros((horizon, n, m), dtype=dtype)
        node_inhomogeneous = jnp.zeros((horizon, n), dtype=dtype)
        valid_values: list[Array] = []
        orthogonality_defects: list[Array] = [self.terminal_basis_defect]

        for segment in range(segments - 1, -1, -1):
            terminal_basis_seeds = terminal_basis_seeds.at[segment].set(basis)
            terminal_inhomogeneous_seeds = terminal_inhomogeneous_seeds.at[segment].set(
                inhomogeneous
            )
            covariance = jnp.zeros((m, m), dtype=dtype)
            linear = jnp.zeros((m,), dtype=dtype)
            neutral_row = jnp.zeros((m,), dtype=dtype)
            neutral_value = jnp.asarray(0.0, dtype=dtype)
            if segment == segments - 1:
                covariance = (
                    covariance
                    + weights[-1] * ein.contract("ia,ib->ab", jnp.conj(basis), basis).real
                )
                linear = (
                    linear
                    + weights[-1]
                    * ein.contract("ia,i->a", jnp.conj(basis), inhomogeneous).real
                )
                if problem.time_dilation == "flow":
                    assert problem.neutral_direction is not None
                    neutral = jnp.asarray(
                        problem.neutral_direction(
                            trajectory.grid.coordinates[-1],
                            trajectory.states[-1],
                            self.args,
                        )
                    )
                    neutral_row = (
                        neutral_row
                        + weights[-1]
                        * ein.contract("ia,i->a", jnp.conj(basis), neutral).real
                    )
                    neutral_value = (
                        neutral_value
                        + weights[-1] * jnp.vdot(inhomogeneous, neutral).real
                    )

            start = segment * plan.segment_steps
            stop = (segment + 1) * plan.segment_steps
            for index in range(stop - 1, start - 1, -1):
                if plan.memory_mode == "store":
                    endpoint_bases = endpoint_bases.at[index].set(basis)
                    endpoint_inhomogeneous = endpoint_inhomogeneous.at[index].set(
                        inhomogeneous
                    )
                action = EvolutionJacobianAction(
                    problem.evolution,
                    trajectory.states[index],
                    trajectory.grid.coordinates[index],
                    trajectory.grid.coordinates[index + 1],
                    args=self.args,
                )
                previous_basis = _transpose_columns(action, basis)
                state_gradient = _observable_state_gradient(
                    problem,
                    trajectory.grid.coordinates[index],
                    trajectory.states[index],
                    self.args,
                )
                previous_inhomogeneous = (
                    action.transpose_mv(inhomogeneous) + weights[index] * state_gradient
                )
                if plan.memory_mode == "store":
                    node_bases = node_bases.at[index].set(previous_basis)
                    node_inhomogeneous = node_inhomogeneous.at[index].set(
                        previous_inhomogeneous
                    )
                covariance = (
                    covariance
                    + weights[index]
                    * ein.contract(
                        "ia,ib->ab", jnp.conj(previous_basis), previous_basis
                    ).real
                )
                linear = (
                    linear
                    + weights[index]
                    * ein.contract(
                        "ia,i->a", jnp.conj(previous_basis), previous_inhomogeneous
                    ).real
                )
                step_valid = (
                    action.primal.valid
                    & jnp.all(jnp.isfinite(previous_basis))
                    & jnp.all(jnp.isfinite(previous_inhomogeneous))
                    & jnp.all(jnp.isfinite(state_gradient))
                )
                if problem.time_dilation == "flow":
                    assert problem.neutral_direction is not None
                    neutral = jnp.asarray(
                        problem.neutral_direction(
                            trajectory.grid.coordinates[index],
                            trajectory.states[index],
                            self.args,
                        )
                    )
                    neutral_row = (
                        neutral_row
                        + weights[index]
                        * ein.contract("ia,i->a", jnp.conj(previous_basis), neutral).real
                    )
                    neutral_value = (
                        neutral_value
                        + weights[index] * jnp.vdot(previous_inhomogeneous, neutral).real
                    )
                    step_valid = step_valid & jnp.all(jnp.isfinite(neutral))
                valid_values.append(step_valid)
                basis = previous_basis
                inhomogeneous = previous_inhomogeneous

            covariances = covariances.at[segment].set(covariance)
            linear_terms = linear_terms.at[segment].set(linear)
            neutral_basis = neutral_basis.at[segment].set(neutral_row)
            neutral_offsets = neutral_offsets.at[segment].set(neutral_value)
            if segment > 0:
                basis, relation, defect, rank_valid = orthonormalize_shadowing_basis(
                    basis,
                    rank_tolerance=plan.rank_tolerance,
                )
                offset = (
                    ein.contract("ia,i->a", jnp.conj(basis), inhomogeneous).real
                    if m
                    else jnp.zeros((0,), dtype=dtype)
                )
                inhomogeneous = inhomogeneous - (
                    ein.contract("ia,a->i", basis, offset)
                    if m
                    else jnp.zeros_like(inhomogeneous)
                )
                relations = relations.at[segment - 1].set(relation)
                offsets = offsets.at[segment - 1].set(offset)
                orthogonality_defects.append(defect)
                valid_values.append(rank_valid)

        reduced = solve_reduced_shadowing(
            covariances,
            linear_terms,
            relations,
            offsets,
            regularization=plan.regularization,
            neutral_basis=neutral_basis if problem.time_dilation == "flow" else None,
            neutral_offset=(
                jnp.sum(neutral_offsets) if problem.time_dilation == "flow" else None
            ),
            relation_orientation="backward",
            solve_id=self.prepared_id,
        )

        parameter_gradient = jax.tree.map(jnp.zeros_like, self.args)
        for index in range(horizon + 1):
            direct = _observable_parameter_pullback(
                problem,
                trajectory.grid.coordinates[index],
                trajectory.states[index],
                self.args,
                weights[index],
            )
            parameter_gradient = _tree_add(parameter_gradient, direct)

        if plan.memory_mode == "store":
            shadowing_path = jnp.zeros((horizon + 1, n), dtype=dtype)
            for index in range(horizon):
                segment = min(index // plan.segment_steps, segments - 1)
                coefficient = reduced.coefficients[segment]
                shadowing_path = shadowing_path.at[index].set(
                    node_inhomogeneous[index]
                    + (
                        ein.contract("ia,a->i", node_bases[index], coefficient)
                        if m
                        else jnp.zeros((n,), dtype=dtype)
                    )
                )
                endpoint = endpoint_inhomogeneous[index] + (
                    ein.contract("ia,a->i", endpoint_bases[index], coefficient)
                    if m
                    else jnp.zeros((n,), dtype=dtype)
                )
                parameter_action = EvolutionArgumentJacobianAction(
                    problem.evolution,
                    trajectory.states[index],
                    trajectory.grid.coordinates[index],
                    trajectory.grid.coordinates[index + 1],
                    self.args,
                )
                parameter_gradient = _tree_add(
                    parameter_gradient,
                    parameter_action.transpose_mv(endpoint),
                )
            final_coefficient = reduced.coefficients[-1]
            shadowing_path = shadowing_path.at[-1].set(
                terminal_inhomogeneous_seeds[-1]
                + (
                    ein.contract("ia,a->i", terminal_basis_seeds[-1], final_coefficient)
                    if m
                    else jnp.zeros((n,), dtype=dtype)
                )
            )
        else:
            shadowing_path = jnp.zeros((0, n), dtype=dtype)
            for segment in range(segments - 1, -1, -1):
                basis = terminal_basis_seeds[segment]
                inhomogeneous = terminal_inhomogeneous_seeds[segment]
                coefficient = reduced.coefficients[segment]
                start = segment * plan.segment_steps
                stop = (segment + 1) * plan.segment_steps
                for index in range(stop - 1, start - 1, -1):
                    endpoint = inhomogeneous + (
                        ein.contract("ia,a->i", basis, coefficient)
                        if m
                        else jnp.zeros((n,), dtype=dtype)
                    )
                    parameter_action = EvolutionArgumentJacobianAction(
                        problem.evolution,
                        trajectory.states[index],
                        trajectory.grid.coordinates[index],
                        trajectory.grid.coordinates[index + 1],
                        self.args,
                    )
                    parameter_gradient = _tree_add(
                        parameter_gradient,
                        parameter_action.transpose_mv(endpoint),
                    )
                    action = EvolutionJacobianAction(
                        problem.evolution,
                        trajectory.states[index],
                        trajectory.grid.coordinates[index],
                        trajectory.grid.coordinates[index + 1],
                        args=self.args,
                    )
                    basis = _transpose_columns(action, basis)
                    inhomogeneous = action.transpose_mv(inhomogeneous) + weights[
                        index
                    ] * _observable_state_gradient(
                        problem,
                        trajectory.grid.coordinates[index],
                        trajectory.states[index],
                        self.args,
                    )

        maximum_orthogonality_defect = jnp.max(
            jnp.stack(tuple(orthogonality_defects)), initial=0.0
        )
        propagation_valid = jnp.all(jnp.stack(tuple(valid_values)))
        finite = (
            _tree_finite(self.args)
            & _tree_finite(parameter_gradient)
            & jnp.all(jnp.isfinite(reduced.coefficients))
            & jnp.all(jnp.isfinite(objective_values))
            & jnp.isfinite(maximum_orthogonality_defect)
        )
        tolerance = 100.0 * plan.rank_tolerance
        neutral_valid = (
            reduced.neutral_residual <= tolerance
            if problem.time_dilation == "flow"
            else jnp.asarray(True)
        )
        successful = (
            trajectory.successful
            & propagation_valid
            & reduced.successful
            & finite
            & (reduced.continuity_residual <= tolerance)
            & (maximum_orthogonality_defect <= tolerance)
            & neutral_valid
        )
        status = jnp.where(
            ~trajectory.successful,
            int(ShadowingSolveStatus.TRAJECTORY_INVALID),
            jnp.where(
                ~propagation_valid,
                int(ShadowingSolveStatus.BASIS_RANK_LOST),
                jnp.where(
                    ~reduced.successful,
                    int(ShadowingSolveStatus.LINEAR_SOLVE_FAILED),
                    jnp.where(
                        ~finite,
                        int(ShadowingSolveStatus.NONFINITE),
                        jnp.where(
                            reduced.continuity_residual > tolerance,
                            int(ShadowingSolveStatus.CONTINUITY_FAILED),
                            jnp.where(
                                ~neutral_valid,
                                int(ShadowingSolveStatus.NEUTRAL_CONSTRAINT_FAILED),
                                int(ShadowingSolveStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        base_actions = horizon * (m + 1)
        replay_actions = base_actions if plan.memory_mode == "recompute" else 0
        return NILSASResult(
            trajectory=trajectory,
            parameter_gradient=parameter_gradient,
            adjoint_shadowing_path=shadowing_path,
            segment_coefficients=reduced.coefficients,
            objective_average=jnp.sum(weights * objective_values),
            continuity_residual=reduced.continuity_residual,
            neutral_constraint_residual=reduced.neutral_residual,
            maximum_orthogonality_defect=maximum_orthogonality_defect,
            finite=finite,
            successful=successful,
            status=status,
            linear_status=reduced.linear_status,
            linear_rank=reduced.rank,
            linear_condition_estimate=reduced.condition_estimate,
            adjoint_action_count=jnp.asarray(
                base_actions + replay_actions, dtype=jnp.int32
            ),
            parameter_action_count=jnp.asarray(horizon, dtype=jnp.int32),
            replay_action_count=jnp.asarray(replay_actions, dtype=jnp.int32),
            cost=self.cost,
            prepared_id=self.prepared_id,
            problem_id=problem.problem_id,
            parameter_id=problem.parameter_id,
            objective_id=problem.observable_id,
            approximation=(
                "finite-horizon-discrete-adjoint-shadowing"
                if isinstance(trajectory.grid, IterationGrid)
                else "finite-horizon-time-discrete-flow-adjoint-shadowing"
            ),
        )


class NILSASResult(StrictModule):
    trajectory: EvolutionTrajectory
    parameter_gradient: PyTree[Array]
    adjoint_shadowing_path: Array
    segment_coefficients: Array
    objective_average: Array
    continuity_residual: Array
    neutral_constraint_residual: Array
    maximum_orthogonality_defect: Array
    finite: Array
    successful: Array
    status: Array
    linear_status: Array
    linear_rank: Array
    linear_condition_estimate: Array
    adjoint_action_count: Array
    parameter_action_count: Array
    replay_action_count: Array
    cost: ShadowingSolveCost
    prepared_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)


__all__ = ["NILSASPlan", "NILSASResult", "PreparedNILSAS"]
