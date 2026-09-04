#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite, pi

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._strict import StrictModule
from ....continuation import ParameterContinuationProblem
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    EigenSolveResult,
    EigenSolveStatus,
    GeneralizedEigenproblem,
    HermitianEigenspaceTrackingPlan,
    HermitianEigenspaceTrackingResult,
    track_hermitian_eigenspaces,
)
from ._equilibrium import (
    _residual,
    MemberNetworkInputs,
    MemberNetworkProblem,
    MemberNetworkResult,
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
    """Accepted constrained tangent and optional physical modal evidence."""

    tangent: Array
    mass: Array | None
    eigenvalues: Array
    modes: Array
    angular_frequencies: Array | None
    minimum_eigenvalue: Array
    stable: Array
    equilibrium_accepted: Array
    physical_tangent: Array
    minimum_mass_eigenvalue: Array
    mass_positive: Array
    eigen_residual: Array
    mass_orthogonality_error: Array
    rigid_modes_valid: Array
    mode_gap_valid: Array
    modal_valid: Array
    mode_derivatives_available: Array
    rigid_mode_count: int = eqx.field(static=True)
    tracking: HermitianEigenspaceTrackingResult | None
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
    equilibrium: MemberNetworkResult,
    /,
    *,
    mass: ArrayLike | None = None,
    rigid_mode_count: int = 0,
    tracking_plan: HermitianEigenspaceTrackingPlan | None = None,
    reference_modes: ArrayLike | None = None,
    cutoff: float = 1.0e-8,
    mode_gap_cutoff: float | None = None,
    differentiate_eigenvalues: bool = False,
) -> TangentStabilityResult:
    """Certify a constrained tangent and, when mass is supplied, physical modes.

    The equilibrium result is the sole state and numerical-input authority. Modal
    evidence is available only for a positive-definite reduced mass matrix.
    ``reference_modes`` uses the same physical coordinates and current mass metric
    as the returned modes.
    """
    cutoff = float(cutoff)
    if not isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be finite and nonnegative.")
    if not isinstance(equilibrium, MemberNetworkResult):
        raise TypeError("equilibrium must be a MemberNetworkResult.")
    if (
        equilibrium.provenance.problem_id != problem.problem_id
        or equilibrium.provenance.definition_id != problem.definition.definition_id
        or equilibrium.provenance.assembly_id != problem.assembly.assembly_id
    ):
        raise ValueError("Equilibrium provenance does not match the mechanics problem.")
    dimension = problem.definition.dofs.reduced_size
    if dimension < 1:
        raise ValueError("Tangent stability requires at least one free physical DOF.")
    rigid_count = int(rigid_mode_count)
    if rigid_count < 0 or rigid_count >= dimension:
        raise ValueError("rigid_mode_count must lie in [0, reduced_size).")
    if (tracking_plan is None) != (reference_modes is None):
        raise ValueError("tracking_plan and reference_modes must be supplied together.")
    if mass is None and tracking_plan is not None:
        raise ValueError("Mode tracking requires a physical mass matrix.")

    tangent = member_network_tangent(
        problem,
        equilibrium.inputs,
        equilibrium.state.kinematics,
    )
    asymmetry = jnp.max(jnp.abs(tangent - tangent.T), initial=0.0)
    tangent = eqx.error_if(
        tangent,
        asymmetry > cutoff,
        "Tangent stability requires a symmetric conservative tangent.",
    )
    tangent = 0.5 * (tangent + tangent.T)
    tangent_operator = DenseLinearOperator(
        tangent,
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        operator_id=f"{problem.problem_id}:constrained-tangent",
    )
    differentiation = "eigenvalues" if differentiate_eigenvalues else "none"
    policy = EigenSolvePolicy(
        DenseEigh(),
        count=dimension,
        which="smallest-algebraic",
        differentiation=differentiation,
    )
    mass_array = None
    mass_positive = jnp.asarray(False)
    mass_orthogonality_error = jnp.asarray(jnp.inf, dtype=tangent.dtype)
    angular_frequencies = None
    minimum_mass_eigenvalue = jnp.asarray(-jnp.inf, dtype=tangent.dtype)
    tracking = None

    if mass is None:
        result = eigensolve(
            Eigenproblem(
                tangent_operator,
                problem_id=f"{problem.problem_id}:tangent-spectrum",
            ),
            policy=policy,
        )
        eigenvalues = result.eigenvalues
        modes = result.eigenvectors
        rigid_modes_valid = jnp.asarray(rigid_count == 0)
        modal_valid = jnp.asarray(False)
        tracking_valid = jnp.asarray(True)
    else:
        mass_array = jnp.asarray(mass, dtype=tangent.dtype)
        if mass_array.shape != tangent.shape:
            raise ValueError("mass must have the reduced tangent shape.")
        mass_asymmetry = jnp.max(
            jnp.abs(mass_array - mass_array.T),
            initial=0.0,
        )
        mass_array = eqx.error_if(
            mass_array,
            mass_asymmetry > cutoff,
            "Modal stability requires a symmetric physical mass.",
        )
        mass_array = 0.5 * (mass_array + mass_array.T)
        mass_operator = DenseLinearOperator(
            mass_array,
            properties=OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=f"{problem.problem_id}:constrained-mass-check",
        )
        mass_result = eigensolve(
            Eigenproblem(
                mass_operator,
                problem_id=f"{problem.problem_id}:mass-spectrum",
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=dimension,
                which="smallest-algebraic",
            ),
        )
        mass_positive = mass_result.successful & jnp.all(mass_result.eigenvalues > cutoff)
        minimum_mass_eigenvalue = jnp.min(mass_result.eigenvalues)
        mass_array = eqx.error_if(
            mass_array,
            ~mass_positive,
            "Modal stability requires positive-definite physical mass.",
        )
        metric_operator = DenseLinearOperator(
            mass_array,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "verified",
                },
            ),
            operator_id=f"{problem.problem_id}:constrained-mass",
        )
        result = eigensolve(
            GeneralizedEigenproblem(
                tangent_operator,
                metric_operator,
                problem_id=f"{problem.problem_id}:physical-modes",
            ),
            policy=policy,
        )
        eigenvalues = result.eigenvalues
        modes = result.eigenvectors
        angular_frequencies = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
        gram = contract("ai,ab,bj->ij", jnp.conj(modes), mass_array, modes)
        mass_orthogonality_error = jnp.max(
            jnp.abs(gram - jnp.eye(dimension, dtype=gram.dtype))
        )
        rigid_modes_valid = jnp.all(
            jnp.abs(eigenvalues[:rigid_count]) <= cutoff
        ) & jnp.all(eigenvalues[rigid_count:] > cutoff)
        tracking_valid = jnp.asarray(True)
        if tracking_plan is not None:
            if tracking_plan.dimension != dimension:
                raise ValueError(
                    "tracking_plan dimension must equal the reduced modal dimension."
                )
            reference = jnp.asarray(reference_modes, dtype=modes.dtype)
            if reference.shape != modes.shape:
                raise ValueError("reference_modes must have the physical mode shape.")
            mass_vectors = mass_result.eigenvectors
            square_roots = jnp.sqrt(mass_result.eigenvalues)
            mass_square_root = (mass_vectors * square_roots[None, :]) @ jnp.conj(
                mass_vectors.T
            )
            inverse_mass_square_root = (
                mass_vectors * (1.0 / square_roots)[None, :]
            ) @ jnp.conj(mass_vectors.T)
            tracking = track_hermitian_eigenspaces(
                tracking_plan,
                mass_square_root @ reference,
                eigenvalues,
                mass_square_root @ modes,
            )
            tracking_valid = tracking.successful
            eigenvalues = tracking.values
            modes = inverse_mass_square_root @ tracking.vectors
            angular_frequencies = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
        modal_valid = (
            mass_positive
            & rigid_modes_valid
            & (mass_orthogonality_error <= cutoff)
            & tracking_valid
        )

    gap_limit = cutoff if mode_gap_cutoff is None else float(mode_gap_cutoff)
    if not isfinite(gap_limit) or gap_limit < 0.0:
        raise ValueError("mode_gap_cutoff must be finite and nonnegative.")
    isolation_gaps = result.diagnostics.isolation_gaps
    mode_gap_valid = jnp.all(
        ~jnp.isnan(isolation_gaps[rigid_count:])
        & (isolation_gaps[rigid_count:] > gap_limit)
    )
    equilibrium_accepted = (
        equilibrium.successful & equilibrium.diagnostics.equilibrium_valid
    )
    physical_tangent = (
        equilibrium_accepted & jnp.all(jnp.isfinite(tangent)) & (asymmetry <= cutoff)
    )
    eigen_residual = jnp.max(result.relative_residuals, initial=0.0)
    minimum = jnp.min(eigenvalues)
    spectral_valid = result.successful | (
        result.status == int(EigenSolveStatus.DIFFERENTIATION_REJECTED)
    )
    stable = (
        physical_tangent
        & spectral_valid
        & rigid_modes_valid
        & jnp.all(eigenvalues[rigid_count:] > cutoff)
    )
    modal_valid = (
        modal_valid
        & physical_tangent
        & spectral_valid
        & mode_gap_valid
        & (eigen_residual <= cutoff)
    )
    derivatives_available = (
        jnp.asarray(differentiate_eigenvalues)
        & result.successful
        & modal_valid
        & mode_gap_valid
        & tracking_valid
    )
    return TangentStabilityResult(
        tangent,
        mass_array,
        eigenvalues,
        modes,
        angular_frequencies,
        minimum,
        stable,
        equilibrium_accepted,
        physical_tangent,
        minimum_mass_eigenvalue,
        mass_positive,
        eigen_residual,
        mass_orthogonality_error,
        rigid_modes_valid,
        mode_gap_valid,
        modal_valid,
        derivatives_available,
        rigid_count,
        tracking,
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
