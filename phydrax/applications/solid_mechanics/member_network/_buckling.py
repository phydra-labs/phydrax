#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....continuation import ParameterContinuationProblem
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
    GeneralizedEigenproblem,
)
from ._equilibrium import (
    _residual,
    MemberNetworkInputs,
    MemberNetworkProblem,
)
from ._reference import MemberKinematics, MemberNetworkDefinition


class BucklingEvidenceLevel(IntEnum):
    CURRENT_TANGENT = 0
    LINEARIZED_LOAD_FACTOR = 1
    NONLINEAR_CRITICAL_POINT = 2
    POSTBUCKLING_BRANCH = 3


class LocalMemberBucklingResult(StrictModule):
    critical_load: Array
    compression_demand: Array
    utilization: Array
    margin: Array
    effective_length_factor: Array
    governing_axis: Array
    valid: Array


class TangentStabilityResult(StrictModule):
    tangent: Array
    eigenvalues: Array
    modes: Array
    minimum_eigenvalue: Array
    stable: Array
    eigen_result: EigenSolveResult


class LinearBucklingResult(StrictModule):
    load_factors: Array
    modes: Array
    positive_mask: Array
    critical_factor: Array
    successful: Array
    proportional_load_verified: bool = eqx.field(static=True)
    conservative_verified: bool = eqx.field(static=True)
    eigen_result: EigenSolveResult


def local_euler_buckling(
    definition: MemberNetworkDefinition,
    axial_forces: ArrayLike,
    effective_length_factors: ArrayLike,
    current_lengths: ArrayLike,
    /,
) -> LocalMemberBucklingResult:
    """Return local Euler flexural buckling evidence with explicit K factors."""
    forces = jnp.asarray(axial_forces)
    factors = jnp.asarray(effective_length_factors, dtype=forces.dtype)
    lengths = jnp.asarray(current_lengths, dtype=forces.dtype)
    count = definition.structure.member_count
    if forces.shape != (count,) or factors.shape != (count,) or lengths.shape != (count,):
        raise ValueError("Local buckling arrays must match the member count.")
    if bool(jnp.any(~jnp.isfinite(factors) | (factors <= 0.0))):
        raise ValueError("Effective-length factors must be finite and positive.")
    properties = definition.properties.structural_arrays()
    minimum_inertia = jnp.minimum(properties["inertia_y"], properties["inertia_z"])
    governing_axis = jnp.where(
        properties["inertia_y"] <= properties["inertia_z"], 0, 1
    ).astype(jnp.int32)
    critical = pi**2 * properties["young"] * minimum_inertia / (factors * lengths) ** 2
    compression = jnp.maximum(-forces, 0.0)
    utilization = compression / jnp.maximum(critical, jnp.finfo(forces.dtype).tiny)
    valid = (
        jnp.isfinite(critical)
        & (critical > 0.0)
        & jnp.isfinite(lengths)
        & (lengths > 0.0)
        & (minimum_inertia > 0.0)
    )
    return LocalMemberBucklingResult(
        critical,
        compression,
        utilization,
        1.0 - utilization,
        factors,
        governing_axis,
        valid,
    )


def member_network_tangent(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    kinematics: MemberKinematics,
    /,
) -> Array:
    reduced = problem.definition.dofs.reduce(
        kinematics.positions, kinematics.rotation_vectors
    )
    return jax.jacfwd(lambda value: _residual(problem, value, inputs))(reduced)


def tangent_stability(
    problem: MemberNetworkProblem,
    inputs: MemberNetworkInputs,
    kinematics: MemberKinematics,
    /,
    *,
    cutoff: float = 1.0e-8,
) -> TangentStabilityResult:
    """Return complete symmetric tangent spectrum at one equilibrium state."""
    tangent = member_network_tangent(problem, inputs, kinematics)
    asymmetry = jnp.max(jnp.abs(tangent - tangent.T), initial=0.0)
    tangent = eqx.error_if(
        tangent,
        asymmetry > cutoff,
        "Tangent stability requires a symmetric conservative tangent.",
    )
    operator = DenseLinearOperator(
        0.5 * (tangent + tangent.T),
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        operator_id=f"{problem.problem_id}:tangent",
    )
    dimension = problem.definition.dofs.reduced_size
    result = eigensolve(
        Eigenproblem(operator, problem_id=f"{problem.problem_id}:tangent-spectrum"),
        policy=EigenSolvePolicy(DenseEigh(), count=dimension, which="smallest-algebraic"),
    )
    minimum = jnp.min(result.eigenvalues)
    return TangentStabilityResult(
        tangent,
        result.eigenvalues,
        result.eigenvectors,
        minimum,
        result.successful & (minimum > cutoff),
        result,
    )


def linearized_buckling(
    material_tangent: ArrayLike,
    geometric_tangent: ArrayLike,
    /,
    *,
    count: int | None = None,
    proportional_load_verified: bool,
    conservative_verified: bool,
    problem_id: str = "member-network-linear-buckling",
) -> LinearBucklingResult:
    """Solve (-Kgeo) phi = mu Kmaterial phi and return lambda = 1 / mu."""
    material = jnp.asarray(material_tangent)
    geometric = jnp.asarray(geometric_tangent, dtype=material.dtype)
    if (
        material.ndim != 2
        or material.shape[0] != material.shape[1]
        or geometric.shape != material.shape
    ):
        raise ValueError("Buckling tangents must be aligned square matrices.")
    if not proportional_load_verified or not conservative_verified:
        raise ValueError(
            "Linear buckling requires verified proportional conservative loading."
        )
    tolerance = 1.0e-8
    material = eqx.error_if(
        material,
        jnp.max(jnp.abs(material - material.T), initial=0.0) > tolerance,
        "Material tangent must be symmetric.",
    )
    geometric = eqx.error_if(
        geometric,
        jnp.max(jnp.abs(geometric - geometric.T), initial=0.0) > tolerance,
        "Geometric tangent must be symmetric.",
    )
    material_operator = DenseLinearOperator(
        material,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "asserted",
            },
        ),
        operator_id=f"{problem_id}:material",
    )
    geometric_operator = DenseLinearOperator(
        -geometric,
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        operator_id=f"{problem_id}:negative-geometric",
    )
    dimension = material.shape[0]
    requested = dimension if count is None else min(int(count), dimension)
    result = eigensolve(
        GeneralizedEigenproblem(
            geometric_operator,
            material_operator,
            problem_id=problem_id,
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=requested, which="largest-algebraic"),
    )
    positive = result.eigenvalues > 0.0
    factors = jnp.where(positive, 1.0 / result.eigenvalues, jnp.inf)
    critical = jnp.min(factors)
    return LinearBucklingResult(
        factors,
        result.eigenvectors,
        positive,
        critical,
        result.successful & jnp.any(positive),
        True,
        True,
        result,
    )


def member_network_continuation_problem(
    problem: MemberNetworkProblem,
    base_inputs: MemberNetworkInputs,
    /,
    *,
    problem_id: str | None = None,
) -> ParameterContinuationProblem:
    """Scale declared nodal force/moment loads along one continuation coordinate."""
    definition = problem.definition

    def residual(reduced, coordinate, args):
        del args
        scaled = eqx.tree_at(
            lambda selected: (selected.nodal_forces, selected.nodal_moments),
            base_inputs,
            (
                coordinate * base_inputs.nodal_forces,
                coordinate * base_inputs.nodal_moments,
            ),
        )
        return _residual(problem, reduced, scaled)

    dtype = base_inputs.rest_lengths.dtype
    from ....linalg import ArraySpace

    space = ArraySpace((definition.dofs.reduced_size,), dtype=dtype)
    return ParameterContinuationProblem(
        residual,
        parameter_lower=0.0,
        parameter_upper=jnp.inf,
        state_space=space,
        residual_space=space,
        problem_id=problem_id or f"{problem.problem_id}:load-continuation",
    )


__all__ = [
    "BucklingEvidenceLevel",
    "LinearBucklingResult",
    "LocalMemberBucklingResult",
    "TangentStabilityResult",
    "linearized_buckling",
    "local_euler_buckling",
    "member_network_continuation_problem",
    "member_network_tangent",
    "tangent_stability",
]
