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
from ..dynamics import EvolutionArgumentJacobianAction, EvolutionTrajectory
from ..dynamics._grid import IterationGrid, TimeGrid
from ..dynamics.analysis import (
    evaluate_shadowing_candidate,
    ShadowingCandidateResult,
    ShadowingSensitivityProblem,
)
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


def _tree_finite(tree: PyTree[Array], /) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _propagate_columns(
    problem: ShadowingSensitivityProblem,
    state: Array,
    basis: Array,
    source: Array,
    target: Array,
    args: Any,
    /,
) -> tuple[Array, Array]:
    if basis.shape[1] == 0:
        return basis, jnp.asarray(True)

    def propagate(column):
        result = problem.evolution.tangent_action(
            state,
            column,
            source,
            target,
            args,
        )
        return result.tangent, result.valid

    columns, valid = eqx.filter_vmap(propagate)(basis.T)
    return columns.T, jnp.all(valid)


def _project_flow_tangents(
    basis: Array,
    inhomogeneous: Array,
    neutral: Array,
    /,
    *,
    tolerance: float,
) -> tuple[Array, Array, Array, Array, Array]:
    denominator = jnp.vdot(neutral, neutral).real
    safe = jnp.where(denominator > tolerance, denominator, 1.0)
    basis_coefficients = (
        ein.contract("i,ia->a", jnp.conj(neutral), basis).real / safe
        if basis.shape[1]
        else jnp.zeros((0,), dtype=inhomogeneous.dtype)
    )
    inhomogeneous_coefficient = jnp.vdot(neutral, inhomogeneous).real / safe
    projected_basis = basis - neutral[:, None] * basis_coefficients[None, :]
    projected_inhomogeneous = inhomogeneous - neutral * inhomogeneous_coefficient
    valid = (
        (denominator > tolerance)
        & jnp.all(jnp.isfinite(projected_basis))
        & jnp.all(jnp.isfinite(projected_inhomogeneous))
    )
    return (
        projected_basis,
        projected_inhomogeneous,
        -basis_coefficients,
        -inhomogeneous_coefficient,
        valid,
    )


class NILSSPlan(AbstractShadowingSolvePlan):
    """Matrix-free segmented non-intrusive least-squares shadowing policy."""

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
        maximum_retained_bytes: int = 2 * 1024 * 1024 * 1024,
        maximum_workspace_bytes: int = 4 * 1024 * 1024 * 1024,
    ):
        values = validate_shadowing_plan(
            "nilss",
            state_dimension,
            unstable_dimension,
            basis_dimension,
            segment_steps,
            segment_count,
            regularization,
            rank_tolerance,
            "store",
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
        self.method = "nilss"
        self.memory_mode = "store"
        self.plan_id = shadowing_plan_id("nilss", values, self.memory_mode)

    def prepare(
        self,
        problem: ShadowingSensitivityProblem,
        trajectory: EvolutionTrajectory,
        parameter_direction: PyTree[Any],
        /,
        *,
        args: PyTree[Array],
        initial_basis: ArrayLike | None = None,
        key: Array | None = None,
    ) -> "PreparedNILSS":
        if not isinstance(problem, ShadowingSensitivityProblem):
            raise TypeError("problem must be a ShadowingSensitivityProblem.")
        if not isinstance(trajectory, EvolutionTrajectory):
            raise TypeError("trajectory must be an EvolutionTrajectory.")
        if problem.evolution.eventful:
            raise ValueError("NILSS requires event-free evolution segments.")
        if problem.evolution.stochastic:
            raise ValueError("NILSS requires deterministic evolution segments.")
        if trajectory.evolution_id != problem.evolution.evolution_id:
            raise ValueError("Trajectory and shadowing evolution IDs must match.")
        if trajectory.grid.num_steps != self.horizon_steps:
            raise ValueError("Trajectory horizon does not match the NILSS plan.")
        if trajectory.state_layout.shape != (self.state_dimension,):
            raise ValueError("NILSS requires the declared one-dimensional state vector.")
        if not trajectory.state_layout.geometry.trivial:
            raise ValueError("NILSS initially requires Euclidean state geometry.")
        if not jnp.issubdtype(trajectory.states.dtype, jnp.floating):
            raise TypeError("NILSS requires a real floating trajectory.")
        if jax.tree.structure(args) != jax.tree.structure(parameter_direction):
            raise ValueError("Parameter direction must match the argument PyTree.")
        argument_leaves = jax.tree.leaves(args)
        direction_leaves = jax.tree.leaves(parameter_direction)
        if not argument_leaves or any(
            not eqx.is_inexact_array(leaf) for leaf in argument_leaves
        ):
            raise TypeError("NILSS args must be a nonempty PyTree of inexact arrays.")
        if any(
            argument.shape != direction.shape or argument.dtype != direction.dtype
            for argument, direction in zip(argument_leaves, direction_leaves, strict=True)
        ):
            raise ValueError(
                "Parameter directions must match argument shapes and dtypes."
            )
        if isinstance(trajectory.grid, IterationGrid):
            if problem.time_dilation != "none":
                raise ValueError("Discrete-map NILSS does not use flow time dilation.")
        elif isinstance(trajectory.grid, TimeGrid):
            if problem.time_dilation != "flow":
                raise ValueError("Time-grid NILSS requires flow time dilation.")
        else:
            raise TypeError("NILSS requires an IterationGrid or TimeGrid trajectory.")

        if self.basis_dimension == 0:
            if initial_basis is not None or key is not None:
                raise ValueError("A zero-dimensional NILSS basis accepts no initializer.")
            basis = jnp.zeros((self.state_dimension, 0), dtype=trajectory.states.dtype)
            initial_defect = jnp.asarray(0.0, dtype=trajectory.states.dtype)
        else:
            if (initial_basis is None) == (key is None):
                raise ValueError("Provide exactly one of initial_basis and key.")
            basis = (
                jax.random.normal(
                    key,
                    (self.state_dimension, self.basis_dimension),
                    dtype=trajectory.states.dtype,
                )
                if initial_basis is None
                else jnp.asarray(initial_basis, dtype=trajectory.states.dtype)
            )
            if basis.shape != (self.state_dimension, self.basis_dimension):
                raise ValueError("initial_basis has an incompatible shape.")
            if problem.time_dilation == "flow":
                assert problem.neutral_direction is not None
                neutral = jnp.asarray(
                    problem.neutral_direction(
                        trajectory.grid.coordinates[0],
                        trajectory.states[0],
                        args,
                    )
                )
                basis, _, _, _, neutral_valid = _project_flow_tangents(
                    basis,
                    jnp.zeros((self.state_dimension,), dtype=basis.dtype),
                    neutral,
                    tolerance=self.rank_tolerance,
                )
                if not bool(np.asarray(neutral_valid)):
                    raise ValueError("Initial NILSS neutral direction is singular.")
            basis, _, initial_defect, basis_valid = orthonormalize_shadowing_basis(
                basis,
                rank_tolerance=self.rank_tolerance,
            )
            if not bool(np.asarray(basis_valid)):
                raise ValueError("NILSS initial basis lost numerical rank.")

        itemsize = np.dtype(trajectory.states.dtype).itemsize
        horizon = self.horizon_steps
        retained = itemsize * (
            (horizon + 1) * self.state_dimension * (self.basis_dimension + 2)
            + horizon * (self.basis_dimension + 1)
            + self.segment_count * self.basis_dimension**2
        )
        variables = self.segment_count * self.basis_dimension
        constraints = max(self.segment_count - 1, 0) * self.basis_dimension
        workspace = retained + itemsize * (variables + constraints) ** 2
        if retained > self.maximum_retained_bytes:
            raise MemoryError("NILSS trajectory exceeds maximum_retained_bytes.")
        if workspace > self.maximum_workspace_bytes:
            raise MemoryError("NILSS reduced solve exceeds maximum_workspace_bytes.")
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
        return PreparedNILSS(
            plan=self,
            cost=cost,
            problem=problem,
            trajectory=trajectory,
            args=args,
            direction=parameter_direction,
            initial_basis=basis,
            initial_basis_defect=initial_defect,
        )


class PreparedNILSS(StrictModule, NonTrainableState):
    plan: NILSSPlan
    cost: ShadowingSolveCost
    problem: ShadowingSensitivityProblem
    trajectory: EvolutionTrajectory
    args: PyTree[Array]
    direction: PyTree[Array]
    initial_basis: Array
    initial_basis_defect: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: NILSSPlan,
        cost: ShadowingSolveCost,
        problem: ShadowingSensitivityProblem,
        trajectory: EvolutionTrajectory,
        args: PyTree[Array],
        direction: PyTree[Array],
        initial_basis: Array,
        initial_basis_defect: ArrayLike,
    ):
        self.plan = plan
        self.cost = cost
        self.problem = problem
        self.trajectory = trajectory
        self.args = args
        self.direction = direction
        self.initial_basis = initial_basis
        self.initial_basis_defect = jnp.asarray(initial_basis_defect)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-nilss",
                "plan": plan.plan_id,
                "problem": problem.problem_id,
                "trajectory": trajectory.evolution_id,
                "args": array_tree_fingerprint(args),
                "direction": array_tree_fingerprint(direction),
                "initial_basis": array_tree_fingerprint(initial_basis),
            }
        )

    @eqx.filter_jit
    def solve(self, /) -> "NILSSResult":
        plan = self.plan
        problem = self.problem
        trajectory = self.trajectory
        basis = self.initial_basis
        inhomogeneous = jnp.zeros((plan.state_dimension,), dtype=trajectory.states.dtype)
        basis_samples: list[Array] = []
        inhomogeneous_samples: list[Array] = []
        dilation_basis_samples: list[Array] = []
        dilation_inhomogeneous_samples: list[Array] = []
        relations: list[Array] = []
        offsets: list[Array] = []
        orthogonality_defects: list[Array] = [self.initial_basis_defect]
        valid_values: list[Array] = []
        primal_defects: list[Array] = []

        for segment in range(plan.segment_count):
            for local_step in range(plan.segment_steps):
                index = segment * plan.segment_steps + local_step
                state = trajectory.states[index]
                source = trajectory.grid.coordinates[index]
                target = trajectory.grid.coordinates[index + 1]
                basis_samples.append(basis)
                inhomogeneous_samples.append(inhomogeneous)
                propagated_basis, basis_valid = _propagate_columns(
                    problem,
                    state,
                    basis,
                    source,
                    target,
                    self.args,
                )
                inhomogeneous_step = problem.evolution.tangent_action(
                    state,
                    inhomogeneous,
                    source,
                    target,
                    self.args,
                )
                parameter_action = EvolutionArgumentJacobianAction(
                    problem.evolution,
                    state,
                    source,
                    target,
                    self.args,
                )
                propagated_inhomogeneous = inhomogeneous_step.tangent + jnp.asarray(
                    parameter_action.mv(self.direction)
                )
                endpoint_defect = jnp.max(
                    jnp.abs(
                        jnp.asarray(parameter_action.primal)
                        - trajectory.states[index + 1]
                    ),
                    initial=0.0,
                )
                scale = jnp.maximum(
                    1.0,
                    jnp.max(jnp.abs(trajectory.states[index + 1]), initial=0.0),
                )
                step_valid = (
                    basis_valid
                    & inhomogeneous_step.valid
                    & jnp.all(jnp.isfinite(propagated_basis))
                    & jnp.all(jnp.isfinite(propagated_inhomogeneous))
                    & (endpoint_defect <= 100.0 * plan.rank_tolerance * scale)
                )
                if problem.time_dilation == "flow":
                    assert problem.neutral_direction is not None
                    neutral = jnp.asarray(
                        problem.neutral_direction(
                            target,
                            trajectory.states[index + 1],
                            self.args,
                        )
                    )
                    (
                        propagated_basis,
                        propagated_inhomogeneous,
                        dilation_basis,
                        dilation_inhomogeneous,
                        projection_valid,
                    ) = _project_flow_tangents(
                        propagated_basis,
                        propagated_inhomogeneous,
                        neutral,
                        tolerance=plan.rank_tolerance,
                    )
                    step_valid = step_valid & projection_valid
                else:
                    dilation_basis = jnp.zeros(
                        (plan.basis_dimension,), dtype=trajectory.states.dtype
                    )
                    dilation_inhomogeneous = jnp.asarray(
                        0.0, dtype=trajectory.states.dtype
                    )
                dilation_basis_samples.append(dilation_basis)
                dilation_inhomogeneous_samples.append(dilation_inhomogeneous)
                valid_values.append(step_valid)
                primal_defects.append(endpoint_defect)
                basis = propagated_basis
                inhomogeneous = propagated_inhomogeneous

            if segment + 1 < plan.segment_count:
                basis, relation, defect, rank_valid = orthonormalize_shadowing_basis(
                    basis,
                    rank_tolerance=plan.rank_tolerance,
                )
                offset = (
                    ein.contract("ia,i->a", jnp.conj(basis), inhomogeneous).real
                    if plan.basis_dimension
                    else jnp.zeros((0,), dtype=inhomogeneous.dtype)
                )
                inhomogeneous = inhomogeneous - (
                    ein.contract("ia,a->i", basis, offset)
                    if plan.basis_dimension
                    else jnp.zeros_like(inhomogeneous)
                )
                relations.append(relation)
                offsets.append(offset)
                orthogonality_defects.append(defect)
                valid_values.append(rank_valid)

        basis_samples.append(basis)
        inhomogeneous_samples.append(inhomogeneous)
        bases = jnp.stack(tuple(basis_samples), axis=0)
        inhomogeneous_values = jnp.stack(tuple(inhomogeneous_samples), axis=0)
        dilation_bases = jnp.stack(tuple(dilation_basis_samples), axis=0)
        dilation_offsets = jnp.stack(tuple(dilation_inhomogeneous_samples), axis=0)
        relation_values = (
            jnp.stack(tuple(relations), axis=0)
            if relations
            else jnp.zeros(
                (0, plan.basis_dimension, plan.basis_dimension),
                dtype=trajectory.states.dtype,
            )
        )
        offset_values = (
            jnp.stack(tuple(offsets), axis=0)
            if offsets
            else jnp.zeros((0, plan.basis_dimension), dtype=trajectory.states.dtype)
        )
        weights = _quadrature_weights(trajectory)
        covariances = []
        linear_terms = []
        for segment in range(plan.segment_count):
            start = segment * plan.segment_steps
            stop = (
                (segment + 1) * plan.segment_steps
                if segment + 1 < plan.segment_count
                else plan.horizon_steps + 1
            )
            local_basis = bases[start:stop]
            local_inhomogeneous = inhomogeneous_values[start:stop]
            local_weights = weights[start:stop]
            covariances.append(
                ein.contract(
                    "t,tia,tib->ab",
                    local_weights,
                    jnp.conj(local_basis),
                    local_basis,
                ).real
            )
            linear_terms.append(
                ein.contract(
                    "t,tia,ti->a",
                    local_weights,
                    jnp.conj(local_basis),
                    local_inhomogeneous,
                ).real
            )
        covariance_values = jnp.stack(tuple(covariances), axis=0)
        linear_values = jnp.stack(tuple(linear_terms), axis=0)

        neutral_basis = None
        neutral_offset = None
        reduced = solve_reduced_shadowing(
            covariance_values,
            linear_values,
            relation_values,
            offset_values,
            regularization=plan.regularization,
            neutral_basis=neutral_basis,
            neutral_offset=neutral_offset,
            solve_id=self.prepared_id,
        )
        node_segments = jnp.minimum(
            jnp.arange(plan.horizon_steps + 1) // plan.segment_steps,
            plan.segment_count - 1,
        )
        selected_coefficients = reduced.coefficients[node_segments]
        tangent_path = inhomogeneous_values + ein.contract(
            "tia,ta->ti", bases, selected_coefficients
        )
        step_segments = jnp.arange(plan.horizon_steps) // plan.segment_steps
        step_coefficients = reduced.coefficients[step_segments]
        dilation = dilation_offsets + ein.contract(
            "ta,ta->t", dilation_bases, step_coefficients
        )
        candidate = evaluate_shadowing_candidate(
            problem,
            trajectory,
            tangent_path,
            self.direction,
            args=self.args,
            time_dilation=dilation,
            boundary="free",
        )
        maximum_orthogonality_defect = jnp.max(
            jnp.stack(tuple(orthogonality_defects)), initial=0.0
        )
        propagation_valid = jnp.all(jnp.stack(tuple(valid_values)))
        maximum_primal_defect = jnp.max(jnp.stack(tuple(primal_defects)), initial=0.0)
        finite = (
            candidate.valid
            & jnp.all(jnp.isfinite(tangent_path))
            & jnp.all(jnp.isfinite(dilation))
            & jnp.all(jnp.isfinite(reduced.coefficients))
            & jnp.isfinite(maximum_orthogonality_defect)
            & jnp.isfinite(maximum_primal_defect)
        )
        tolerance = 100.0 * plan.rank_tolerance
        successful = (
            trajectory.successful
            & propagation_valid
            & reduced.successful
            & candidate.valid
            & finite
            & (reduced.continuity_residual <= tolerance)
            & (reduced.neutral_residual <= tolerance)
            & (maximum_orthogonality_defect <= tolerance)
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
                                reduced.neutral_residual > tolerance,
                                int(ShadowingSolveStatus.NEUTRAL_CONSTRAINT_FAILED),
                                int(ShadowingSolveStatus.SUCCESS),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return NILSSResult(
            trajectory=trajectory,
            candidate=candidate,
            shadowing_tangent=tangent_path,
            time_dilation=dilation,
            segment_coefficients=reduced.coefficients,
            objective_average=candidate.observable_mean,
            directional_gradient=candidate.mean_directional_response,
            continuity_residual=reduced.continuity_residual,
            neutral_constraint_residual=reduced.neutral_residual,
            maximum_orthogonality_defect=maximum_orthogonality_defect,
            maximum_primal_defect=maximum_primal_defect,
            finite=finite,
            successful=successful,
            status=status,
            linear_status=reduced.linear_status,
            tangent_evaluations=jnp.asarray(
                plan.horizon_steps * (plan.basis_dimension + 1), dtype=jnp.int32
            ),
            parameter_action_count=jnp.asarray(plan.horizon_steps, dtype=jnp.int32),
            cost=self.cost,
            prepared_id=self.prepared_id,
            problem_id=problem.problem_id,
            parameter_id=problem.parameter_id,
            objective_id=problem.observable_id,
        )


class NILSSResult(StrictModule):
    trajectory: EvolutionTrajectory
    candidate: ShadowingCandidateResult
    shadowing_tangent: Array
    time_dilation: Array
    segment_coefficients: Array
    objective_average: Array
    directional_gradient: Array
    continuity_residual: Array
    neutral_constraint_residual: Array
    maximum_orthogonality_defect: Array
    maximum_primal_defect: Array
    finite: Array
    successful: Array
    status: Array
    linear_status: Array
    tangent_evaluations: Array
    parameter_action_count: Array
    cost: ShadowingSolveCost
    prepared_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)


__all__ = ["NILSSPlan", "NILSSResult", "PreparedNILSS"]
