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
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, HermitianSpectrum, LinearSystem, solve


StepFunction = Callable[[Array, PyTree[Any]], Array]
ObjectiveFunction = Callable[[Array, PyTree[Any]], Array]


def _orthonormalize(
    basis: Array,
    /,
    *,
    rank_tolerance: float,
) -> tuple[Array, Array, Array]:
    real_dtype = jnp.empty((), dtype=basis.dtype).real.dtype
    effective_tolerance = max(
        rank_tolerance,
        10.0 * np.finfo(np.dtype(real_dtype)).eps * max(basis.shape),
    )
    gram = oe.contract("ia,ib->ab", jnp.conj(basis), basis)
    spectrum = HermitianSpectrum(gram, tolerance=effective_tolerance)
    values = spectrum.eigenvalues
    scale = jnp.maximum(jnp.max(jnp.abs(values), initial=0.0), 1.0)
    threshold = effective_tolerance * scale
    valid = spectrum.valid & jnp.all(values > threshold)
    if not bool(np.asarray(valid)):
        raise ValueError("NILSS homogeneous tangent basis lost numerical rank.")
    vectors = spectrum.eigenvectors
    inverse_sqrt = vectors * (1.0 / jnp.sqrt(values))[None, :]
    inverse_sqrt = oe.contract("ia,ja->ij", inverse_sqrt, jnp.conj(vectors))
    square_root = vectors * jnp.sqrt(values)[None, :]
    square_root = oe.contract("ia,ja->ij", square_root, jnp.conj(vectors))
    orthonormal = oe.contract("ia,ab->ib", basis, inverse_sqrt)
    relation = square_root
    defect = jnp.max(
        jnp.abs(
            oe.contract("ia,ib->ab", jnp.conj(orthonormal), orthonormal)
            - jnp.eye(values.size, dtype=basis.dtype)
        ),
        initial=0.0,
    )
    return orthonormal, relation, defect


class NILSSCost(StrictModule, NonTrainableState):
    state_dimension: int = eqx.field(static=True)
    unstable_dimension: int = eqx.field(static=True)
    horizon_steps: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)


class NILSSPlan(StrictModule, NonTrainableState):
    """Segmented non-intrusive least-squares shadowing policy.

    Homogeneous and inhomogeneous tangents are obtained only by differentiating
    the supplied time-step map.  Segment-boundary coefficients satisfy the
    exact QR-equivalent continuity constraints, and one constrained global
    least-squares problem is solved; this is not a set of independently scored
    perturbation candidates.
    """

    state_dimension: int = eqx.field(static=True)
    unstable_dimension: int = eqx.field(static=True)
    segment_steps: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_dimension: int,
        unstable_dimension: int,
        segment_steps: int,
        segment_count: int,
        /,
        *,
        regularization: float = 0.0,
        rank_tolerance: float = 1.0e-10,
        maximum_retained_bytes: int = 2 * 1024 * 1024 * 1024,
        maximum_workspace_bytes: int = 4 * 1024 * 1024 * 1024,
    ):
        dimension = int(state_dimension)
        unstable = int(unstable_dimension)
        length = int(segment_steps)
        count = int(segment_count)
        regularization_ = float(regularization)
        rank = float(rank_tolerance)
        retained_limit = int(maximum_retained_bytes)
        workspace_limit = int(maximum_workspace_bytes)
        if (
            dimension < 1
            or unstable < 1
            or unstable > dimension
            or length < 1
            or count < 1
            or not np.isfinite(regularization_)
            or regularization_ < 0.0
            or not np.isfinite(rank)
            or rank <= 0.0
            or retained_limit <= 0
            or workspace_limit <= 0
        ):
            raise ValueError("NILSS dimensions, tolerances, or resources are invalid.")
        self.state_dimension = dimension
        self.unstable_dimension = unstable
        self.segment_steps = length
        self.segment_count = count
        self.regularization = regularization_
        self.rank_tolerance = rank
        self.maximum_retained_bytes = retained_limit
        self.maximum_workspace_bytes = workspace_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nilss-plan",
                "state_dimension": dimension,
                "unstable_dimension": unstable,
                "segment_steps": length,
                "segment_count": count,
                "regularization": regularization_,
                "rank_tolerance": rank,
                "maximum_retained_bytes": retained_limit,
                "maximum_workspace_bytes": workspace_limit,
                "formulation": "segmented-constrained-least-squares-shadowing",
            }
        )

    @property
    def horizon_steps(self) -> int:
        return self.segment_steps * self.segment_count

    def prepare(
        self,
        step: StepFunction,
        objective: ObjectiveFunction,
        initial_state: ArrayLike,
        parameters: PyTree[Any],
        direction: PyTree[Any],
        /,
        *,
        dynamics_id: str,
        objective_id: str,
        initial_basis: ArrayLike | None = None,
    ) -> "PreparedNILSS":
        if not callable(step) or not callable(objective):
            raise TypeError("step and objective must be callable.")
        dynamics_identifier = str(dynamics_id)
        objective_identifier = str(objective_id)
        if not dynamics_identifier or not objective_identifier:
            raise ValueError("dynamics_id and objective_id must be non-empty.")
        initial = jnp.asarray(initial_state)
        if initial.shape != (self.state_dimension,) or not jnp.issubdtype(
            initial.dtype, jnp.floating
        ):
            raise ValueError("NILSS initial_state must be a real state vector.")
        if not bool(np.asarray(jnp.all(jnp.isfinite(initial)))):
            raise ValueError("NILSS initial_state must be finite.")
        parameter_structure = jax.tree.structure(parameters)
        if parameter_structure != jax.tree.structure(direction):
            raise ValueError("Parameter direction must match the parameter PyTree.")
        basis = (
            jnp.eye(self.state_dimension, self.unstable_dimension, dtype=initial.dtype)
            if initial_basis is None
            else jnp.asarray(initial_basis, dtype=initial.dtype)
        )
        if basis.shape != (self.state_dimension, self.unstable_dimension):
            raise ValueError("initial_basis has an incompatible shape.")
        basis, _, defect = _orthonormalize(basis, rank_tolerance=self.rank_tolerance)
        if not bool(np.asarray(jnp.all(jnp.isfinite(basis)))):
            raise ValueError("NILSS initial_basis must be finite.")
        itemsize = np.dtype(initial.dtype).itemsize
        horizon = self.horizon_steps
        retained = itemsize * (
            (horizon + 1) * self.state_dimension
            + horizon * self.state_dimension * (self.unstable_dimension + 1)
            + self.segment_count * self.unstable_dimension**2
        )
        variables = self.segment_count * self.unstable_dimension
        constraints = (self.segment_count - 1) * self.unstable_dimension
        workspace = retained + itemsize * (variables + constraints) ** 2
        if retained > self.maximum_retained_bytes:
            raise MemoryError("NILSS trajectory exceeds maximum_retained_bytes.")
        if workspace > self.maximum_workspace_bytes:
            raise MemoryError("NILSS constrained solve exceeds maximum_workspace_bytes.")
        cost = NILSSCost(
            state_dimension=self.state_dimension,
            unstable_dimension=self.unstable_dimension,
            horizon_steps=horizon,
            segment_count=self.segment_count,
            retained_bytes=retained,
            workspace_bytes=workspace,
            maximum_retained_bytes=self.maximum_retained_bytes,
            maximum_workspace_bytes=self.maximum_workspace_bytes,
        )
        return PreparedNILSS(
            self,
            cost,
            step,
            objective,
            initial,
            parameters,
            direction,
            basis,
            dynamics_id=dynamics_identifier,
            objective_id=objective_identifier,
            initial_basis_defect=defect,
        )


class PreparedNILSS(StrictModule, NonTrainableState):
    plan: NILSSPlan
    cost: NILSSCost
    step_function: StepFunction = eqx.field(static=True)
    objective_function: ObjectiveFunction = eqx.field(static=True)
    initial_state: Array
    parameters: PyTree[Any]
    direction: PyTree[Any]
    initial_basis: Array
    initial_basis_defect: Array
    dynamics_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: NILSSPlan,
        cost: NILSSCost,
        step: StepFunction,
        objective: ObjectiveFunction,
        initial_state: Array,
        parameters: PyTree[Any],
        direction: PyTree[Any],
        initial_basis: Array,
        /,
        *,
        dynamics_id: str,
        objective_id: str,
        initial_basis_defect: ArrayLike,
    ):
        self.plan = plan
        self.cost = cost
        self.step_function = step
        self.objective_function = objective
        self.initial_state = initial_state
        self.parameters = parameters
        self.direction = direction
        self.initial_basis = initial_basis
        self.initial_basis_defect = jnp.asarray(initial_basis_defect)
        self.dynamics_id = dynamics_id
        self.objective_id = objective_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-nilss",
                "plan": plan.plan_id,
                "dynamics": dynamics_id,
                "objective": objective_id,
                "initial_state": array_tree_fingerprint(initial_state),
                "parameters": array_tree_fingerprint(parameters),
                "direction": array_tree_fingerprint(direction),
                "initial_basis": array_tree_fingerprint(initial_basis),
            }
        )

    def solve(self, /) -> "NILSSResult":
        plan = self.plan
        m = plan.unstable_dimension
        segments = plan.segment_count
        x = self.initial_state
        basis = self.initial_basis
        inhomogeneous = jnp.zeros_like(x)
        trajectory: list[Array] = [x]
        local_bases: list[Array] = []
        local_inhomogeneous: list[Array] = []
        local_objectives: list[Array] = []
        boundary_relations: list[Array] = []
        boundary_offsets: list[Array] = []
        orthogonality_defects: list[Array] = [self.initial_basis_defect]
        for segment in range(segments):
            for _ in range(plan.segment_steps):
                local_bases.append(basis)
                local_inhomogeneous.append(inhomogeneous)
                objective_value = jnp.asarray(self.objective_function(x, self.parameters))
                if objective_value.shape != ():
                    raise ValueError("NILSS objective must return a scalar.")
                local_objectives.append(objective_value)
                state_jacobian = jax.jacfwd(
                    lambda state: self.step_function(state, self.parameters)
                )(x)
                if state_jacobian.shape != (plan.state_dimension, plan.state_dimension):
                    raise ValueError(
                        "NILSS step function must preserve the state vector shape."
                    )
                next_state, parameter_tangent = jax.jvp(
                    lambda parameters: self.step_function(x, parameters),
                    (self.parameters,),
                    (self.direction,),
                )
                inhomogeneous = (
                    oe.contract("ij,j->i", state_jacobian, inhomogeneous)
                    + parameter_tangent
                )
                basis = oe.contract("ij,ja->ia", state_jacobian, basis)
                x = jnp.asarray(next_state)
                if x.shape != (plan.state_dimension,):
                    raise ValueError(
                        "NILSS step function returned an incompatible state."
                    )
                trajectory.append(x)
            if segment + 1 < segments:
                next_basis, relation, defect = _orthonormalize(
                    basis,
                    rank_tolerance=plan.rank_tolerance,
                )
                offset = oe.contract("ia,i->a", jnp.conj(next_basis), inhomogeneous)
                inhomogeneous = inhomogeneous - oe.contract("ia,a->i", next_basis, offset)
                basis = next_basis
                boundary_relations.append(relation)
                boundary_offsets.append(offset)
                orthogonality_defects.append(defect)
        basis_samples = jnp.stack(tuple(local_bases), axis=0)
        inhomogeneous_samples = jnp.stack(tuple(local_inhomogeneous), axis=0)
        objective_samples = jnp.stack(tuple(local_objectives), axis=0)
        coefficient_count = segments * m
        constraint_count = (segments - 1) * m
        hessian = np.zeros(
            (coefficient_count, coefficient_count), dtype=np.asarray(x).dtype
        )
        gradient = np.zeros((coefficient_count,), dtype=np.asarray(x).dtype)
        for segment in range(segments):
            start = segment * plan.segment_steps
            stop = start + plan.segment_steps
            samples = basis_samples[start:stop]
            offsets = inhomogeneous_samples[start:stop]
            block_hessian = np.asarray(
                oe.contract("tia,tib->ab", jnp.conj(samples), samples)
            )
            block_gradient = np.asarray(
                oe.contract("tia,ti->a", jnp.conj(samples), offsets)
            )
            block = slice(segment * m, (segment + 1) * m)
            hessian[block, block] = block_hessian + plan.regularization * np.eye(m)
            gradient[block] = block_gradient
        constraints = np.zeros(
            (constraint_count, coefficient_count), dtype=np.asarray(x).dtype
        )
        constraint_rhs = np.zeros((constraint_count,), dtype=np.asarray(x).dtype)
        for boundary, (relation, offset) in enumerate(
            zip(boundary_relations, boundary_offsets, strict=True)
        ):
            row = slice(boundary * m, (boundary + 1) * m)
            left = slice(boundary * m, (boundary + 1) * m)
            right = slice((boundary + 1) * m, (boundary + 2) * m)
            constraints[row, left] = -np.asarray(relation)
            constraints[row, right] = np.eye(m)
            constraint_rhs[row] = np.asarray(offset)
        if constraint_count:
            kkt = np.block(
                [
                    [hessian, constraints.T],
                    [
                        constraints,
                        np.zeros(
                            (constraint_count, constraint_count), dtype=hessian.dtype
                        ),
                    ],
                ]
            )
            rhs = np.concatenate((-gradient, constraint_rhs))
        else:
            kkt = hessian
            rhs = -gradient
        operator = DenseLinearOperator(
            jnp.asarray(kkt),
            operator_id=canonical_fingerprint(
                {
                    "kind": "nilss-kkt",
                    "prepared": self.prepared_id,
                    "dimension": int(kkt.shape[0]),
                }
            ),
        )
        linear_result = solve(LinearSystem(operator), jnp.asarray(rhs))
        if not bool(np.asarray(linear_result.successful)):
            raise ValueError("NILSS constrained least-squares solve did not converge.")
        coefficients = jnp.asarray(linear_result.value[:coefficient_count]).reshape(
            (segments, m)
        )
        tangent_samples: list[Array] = []
        for sample in range(plan.horizon_steps):
            segment = sample // plan.segment_steps
            tangent_samples.append(
                inhomogeneous_samples[sample]
                + oe.contract("ia,a->i", basis_samples[sample], coefficients[segment])
            )
        tangents = jnp.stack(tuple(tangent_samples), axis=0)
        direct_derivatives: list[Array] = []
        state_derivatives: list[Array] = []
        for state, tangent in zip(trajectory[:-1], tangents, strict=True):
            direct = jax.jvp(
                lambda parameters: self.objective_function(state, parameters),
                (self.parameters,),
                (self.direction,),
            )[1]
            state_gradient = jax.grad(
                lambda current: self.objective_function(current, self.parameters)
            )(state)
            state_derivatives.append(oe.contract("i,i->", state_gradient, tangent))
            direct_derivatives.append(direct)
        instantaneous = jnp.stack(tuple(state_derivatives)) + jnp.stack(
            tuple(direct_derivatives)
        )
        directional_gradient = jnp.mean(instantaneous)
        continuity_residual = (
            jnp.max(
                jnp.abs(
                    jnp.asarray(constraints) @ coefficients.reshape((-1,))
                    - jnp.asarray(constraint_rhs)
                ),
                initial=0.0,
            )
            if constraint_count
            else jnp.asarray(0.0, dtype=x.dtype)
        )
        maximum_orthogonality_defect = jnp.max(
            jnp.stack(tuple(orthogonality_defects)), initial=0.0
        )
        finite = (
            jnp.all(jnp.isfinite(jnp.stack(tuple(trajectory))))
            & jnp.all(jnp.isfinite(tangents))
            & jnp.isfinite(directional_gradient)
        )
        real_dtype = jnp.empty((), dtype=x.dtype).real.dtype
        effective_tolerance = max(
            plan.rank_tolerance,
            10.0
            * np.finfo(np.dtype(real_dtype)).eps
            * max(plan.state_dimension, plan.unstable_dimension),
        )
        successful = (
            finite
            & (continuity_residual <= 100.0 * effective_tolerance)
            & (maximum_orthogonality_defect <= 100.0 * effective_tolerance)
        )
        return NILSSResult(
            trajectory=jnp.stack(tuple(trajectory)),
            shadowing_tangent=tangents,
            segment_coefficients=coefficients,
            instantaneous_directional_derivative=instantaneous,
            objective_average=jnp.mean(objective_samples),
            directional_gradient=directional_gradient,
            continuity_residual=continuity_residual,
            maximum_orthogonality_defect=maximum_orthogonality_defect,
            finite=finite,
            successful=successful,
            linear_status=linear_result.status,
            prepared_id=self.prepared_id,
            dynamics_id=self.dynamics_id,
            objective_id=self.objective_id,
        )


class NILSSResult(StrictModule):
    trajectory: Array
    shadowing_tangent: Array
    segment_coefficients: Array
    instantaneous_directional_derivative: Array
    objective_average: Array
    directional_gradient: Array
    continuity_residual: Array
    maximum_orthogonality_defect: Array
    finite: Array
    successful: Array
    linear_status: Array
    prepared_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)


__all__ = ["NILSSCost", "NILSSPlan", "NILSSResult", "PreparedNILSS"]
