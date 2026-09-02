#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from ..._strict import StrictModule
from ...continuation import ParameterContinuationProblem
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    OperatorProperties,
)
from ...linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
)
from ._force_density import (
    _internal_nodal_forces,
    _nodal_loads,
    _validated_force_densities,
    ForceDensityInputs,
    ForceDensityProblem,
    ForceDensityState,
)
from ._force_density_topology import ForceDensityStructure


class ForceDensityMechanismResult(StrictModule):
    """Rigidity spectrum, mechanism modes, and self-stress modes."""

    rigidity_matrix: Array
    mechanism_eigenvalues: Array
    mechanism_modes: Array
    mechanism_mask: Array
    self_stress_eigenvalues: Array
    self_stress_modes: Array
    self_stress_mask: Array
    mechanism_count: Array
    self_stress_count: Array
    cutoff: Array
    mechanism_solve: EigenSolveResult
    self_stress_solve: EigenSolveResult

    @property
    def successful(self) -> Array:
        return self.mechanism_solve.successful & self.self_stress_solve.successful


class ForceDensityTangentStabilityResult(StrictModule):
    """Material/geometric tangent spectrum under supplied axial rigidities."""

    tangent_matrix: Array
    eigenvalues: Array
    modes: Array
    stable: Array
    minimum_eigenvalue: Array
    cutoff: Array
    eigen_solve: EigenSolveResult

    @property
    def successful(self) -> Array:
        return self.eigen_solve.successful


def force_density_rigidity_matrix(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
    *,
    minimum_length: float = 1.0e-12,
) -> Array:
    """Return active-member length derivatives with respect to free coordinates."""
    xyz = jnp.asarray(positions)
    expected = (structure.node_count, structure.dimension)
    if xyz.shape != expected:
        raise ValueError(f"positions must have shape {expected}; got {xyz.shape}.")
    margin = float(minimum_length)
    if not isfinite(margin) or margin <= 0.0:
        raise ValueError("minimum_length must be finite and positive.")
    active = np.flatnonzero(np.asarray(structure.member_valid, dtype=bool))
    senders = structure.senders[active]
    receivers = structure.receivers[active]
    vectors = xyz[receivers] - xyz[senders]
    lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    vectors = eqx.error_if(
        vectors,
        jnp.any(lengths <= margin),
        "Rigidity analysis requires nondegenerate active members.",
    )
    directions = vectors / lengths[:, None]
    full = jnp.zeros((active.size, structure.full_dof_count), dtype=xyz.dtype)
    coordinates = jnp.arange(structure.dimension, dtype=jnp.int32)
    sender_dofs = senders[:, None] * structure.dimension + coordinates[None, :]
    receiver_dofs = receivers[:, None] * structure.dimension + coordinates[None, :]
    rows = jnp.arange(active.size, dtype=jnp.int32)[:, None]
    full = full.at[rows, sender_dofs].add(-directions)
    full = full.at[rows, receiver_dofs].add(directions)
    if structure.affine_constraints:
        if structure.affine_prolongation is None:
            raise RuntimeError("Affine prolongation is unavailable.")
        return full @ structure.affine_prolongation
    return full[:, structure.free_dof_indices]


def _gram_eigensolve(matrix: Array, problem_id: str, /) -> EigenSolveResult:
    dimension = int(matrix.shape[0])
    if dimension <= 0:
        raise ValueError("Spectral mechanism analysis requires a nonempty space.")
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{problem_id}:operator",
    )
    return eigensolve(
        Eigenproblem(operator, problem_id=problem_id),
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=dimension,
            which="smallest-algebraic",
        ),
    )


def analyze_force_density_mechanisms(
    structure: ForceDensityStructure,
    positions: ArrayLike,
    /,
    *,
    relative_cutoff: float = 1.0e-8,
    absolute_cutoff: float = 1.0e-12,
) -> ForceDensityMechanismResult:
    """Return infinitesimal mechanisms and axial self-stress subspaces."""
    relative = float(relative_cutoff)
    absolute = float(absolute_cutoff)
    if (
        not isfinite(relative)
        or not isfinite(absolute)
        or relative < 0.0
        or absolute < 0.0
    ):
        raise ValueError("Mechanism cutoffs must be finite and nonnegative.")
    rigidity = force_density_rigidity_matrix(structure, positions)
    mechanism_gram = ein.contract("mi,mj->ij", rigidity, rigidity)
    stress_gram = ein.contract("mi,ni->mn", rigidity, rigidity)
    mechanism = _gram_eigensolve(mechanism_gram, f"{structure.structure_id}:mechanisms")
    self_stress = _gram_eigensolve(stress_gram, f"{structure.structure_id}:self-stress")
    scale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(mechanism.eigenvalues), initial=0.0),
            jnp.max(jnp.abs(self_stress.eigenvalues), initial=0.0),
        ),
        1.0,
    )
    cutoff = absolute + relative * scale
    mechanism_mask = mechanism.eigenvalues <= cutoff
    self_stress_mask = self_stress.eigenvalues <= cutoff
    return ForceDensityMechanismResult(
        rigidity,
        mechanism.eigenvalues,
        mechanism.eigenvectors,
        mechanism_mask,
        self_stress.eigenvalues,
        self_stress.eigenvectors,
        self_stress_mask,
        jnp.sum(mechanism_mask, dtype=jnp.int32),
        jnp.sum(self_stress_mask, dtype=jnp.int32),
        cutoff,
        mechanism,
        self_stress,
    )


def force_density_tangent_matrix(
    structure: ForceDensityStructure,
    state: ForceDensityState,
    axial_rigidities: ArrayLike,
    /,
) -> Array:
    """Assemble constrained truss tangent from material and prestress terms."""
    rigidities = jnp.asarray(axial_rigidities)
    if rigidities.shape != (structure.member_count,):
        raise ValueError("axial_rigidities must contain one value per member.")
    if rigidities.dtype != state.positions.dtype:
        raise TypeError("axial_rigidities must share the state coordinate dtype.")
    rigidities = eqx.error_if(
        rigidities,
        jnp.any(
            structure.member_valid & (~jnp.isfinite(rigidities) | (rigidities <= 0.0))
        ),
        "Active axial rigidities must be finite and positive.",
    )
    active = np.flatnonzero(np.asarray(structure.member_valid, dtype=bool))
    matrix = jnp.zeros(
        (structure.full_dof_count, structure.full_dof_count),
        dtype=state.positions.dtype,
    )
    identity = jnp.eye(structure.dimension, dtype=state.positions.dtype)
    for member in active:
        vector = state.member_vectors[member]
        length = state.member_lengths[member]
        direction = vector / length
        projector = direction[:, None] * direction[None, :]
        material = (rigidities[member] / length) * projector
        geometric = state.force_densities[member] * (identity - projector)
        local = material + geometric
        sender = int(np.asarray(structure.senders[member]))
        receiver = int(np.asarray(structure.receivers[member]))
        sender_dofs = sender * structure.dimension + np.arange(structure.dimension)
        receiver_dofs = receiver * structure.dimension + np.arange(structure.dimension)
        matrix = matrix.at[np.ix_(sender_dofs, sender_dofs)].add(local)
        matrix = matrix.at[np.ix_(receiver_dofs, receiver_dofs)].add(local)
        matrix = matrix.at[np.ix_(sender_dofs, receiver_dofs)].add(-local)
        matrix = matrix.at[np.ix_(receiver_dofs, sender_dofs)].add(-local)
    if structure.affine_constraints:
        if structure.affine_prolongation is None:
            raise RuntimeError("Affine prolongation is unavailable.")
        return structure.affine_prolongation.T @ matrix @ structure.affine_prolongation
    free = structure.free_dof_indices
    return matrix[free[:, None], free[None, :]]


def analyze_force_density_tangent_stability(
    structure: ForceDensityStructure,
    state: ForceDensityState,
    axial_rigidities: ArrayLike,
    /,
    *,
    relative_cutoff: float = 1.0e-8,
    absolute_cutoff: float = 1.0e-12,
) -> ForceDensityTangentStabilityResult:
    """Certify tangent positivity only when constitutive axial rigidity is supplied."""
    tangent = force_density_tangent_matrix(structure, state, axial_rigidities)
    dimension = int(tangent.shape[0])
    operator = DenseLinearOperator(
        tangent,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=f"{structure.structure_id}:constitutive-tangent",
    )
    solved = eigensolve(
        Eigenproblem(
            operator,
            problem_id=f"{structure.structure_id}:tangent-stability",
        ),
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=dimension,
            which="smallest-algebraic",
        ),
    )
    relative = float(relative_cutoff)
    absolute = float(absolute_cutoff)
    scale = jnp.maximum(jnp.max(jnp.abs(solved.eigenvalues), initial=0.0), 1.0)
    cutoff = absolute + relative * scale
    minimum = jnp.min(solved.eigenvalues)
    return ForceDensityTangentStabilityResult(
        tangent,
        solved.eigenvalues,
        solved.eigenvectors,
        solved.successful & (minimum > cutoff),
        minimum,
        cutoff,
        solved,
    )


def force_density_continuation_problem(
    problem: ForceDensityProblem,
    decode_inputs: Callable[[Array, Any], ForceDensityInputs],
    /,
    *,
    parameter_lower: float = -jnp.inf,
    parameter_upper: float = jnp.inf,
    problem_id: str | None = None,
) -> ParameterContinuationProblem:
    """Expose a scalar force/load/support path to native branch continuation."""
    if not isinstance(problem, ForceDensityProblem):
        raise TypeError("problem must be a ForceDensityProblem.")
    if not callable(decode_inputs):
        raise TypeError("decode_inputs must be callable.")
    structure = problem.structure

    def residual(reduced: PyTree[Any], coordinate: Array, args: Any):
        inputs = decode_inputs(coordinate, args)
        if not isinstance(inputs, ForceDensityInputs):
            raise TypeError("decode_inputs must return ForceDensityInputs.")
        force_densities = _validated_force_densities(problem, inputs)
        positions = structure.expand(jnp.asarray(reduced), inputs.prescribed_values)
        loads = _nodal_loads(problem, inputs, positions, force_densities.dtype)
        internal = _internal_nodal_forces(structure, force_densities, positions)
        return structure.reduce(internal - loads)

    space = ArraySpace((structure.free_dof_count,), dtype=jnp.float64)
    return ParameterContinuationProblem(
        residual,
        parameter_lower=parameter_lower,
        parameter_upper=parameter_upper,
        state_space=space,
        residual_space=space,
        problem_id=(
            f"{problem.problem_id}:continuation"
            if problem_id is None
            else str(problem_id)
        ),
    )


__all__ = [
    "ForceDensityMechanismResult",
    "ForceDensityTangentStabilityResult",
    "analyze_force_density_mechanisms",
    "analyze_force_density_tangent_stability",
    "force_density_continuation_problem",
    "force_density_rigidity_matrix",
    "force_density_tangent_matrix",
]
