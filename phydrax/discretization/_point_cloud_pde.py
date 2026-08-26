#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._point_cloud import PreparedPointCloudDiscretization


PointBoundaryKind: TypeAlias = Literal["dirichlet", "neumann", "robin"]


class PointBoundaryPlan(StrictModule):
    kind: PointBoundaryKind = eqx.field(static=True)
    values: Array
    robin_coefficient: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PointBoundaryKind,
        values: ArrayLike,
        /,
        *,
        robin_coefficient: ArrayLike | None = None,
    ):
        if kind not in ("dirichlet", "neumann", "robin"):
            raise ValueError("Unknown point-cloud boundary kind.")
        values_ = jnp.asarray(values)
        if values_.ndim != 1:
            raise ValueError("Point boundary values must be a vector.")
        if kind == "robin":
            if robin_coefficient is None:
                raise ValueError("Robin boundaries require robin_coefficient.")
            coefficient = jnp.broadcast_to(jnp.asarray(robin_coefficient), values_.shape)
            coefficient = eqx.error_if(
                coefficient,
                jnp.any(~jnp.isfinite(coefficient)) | jnp.any(coefficient < 0.0),
                "Robin coefficients must be finite and nonnegative.",
            )
        else:
            if robin_coefficient is not None:
                raise ValueError("Only Robin boundaries accept robin_coefficient.")
            coefficient = None
        self.kind = kind
        self.values = values_
        self.robin_coefficient = coefficient
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-boundary-plan",
                "boundary_kind": kind,
                "values": array_tree_fingerprint(values_),
                "coefficient": None
                if coefficient is None
                else array_tree_fingerprint(coefficient),
            }
        )


class PointSBPReport(StrictModule, NonTrainableState):
    maximum_green_residual: float = eqx.field(static=True)
    maximum_conservation_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


def point_sbp_report(
    discretization: PreparedPointCloudDiscretization,
    /,
    *,
    tolerance: float = 1e-8,
) -> PointSBPReport:
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("SBP tolerance must be finite and nonnegative.")
    boundary_weights = discretization.plan.boundary_quadrature_weights
    if boundary_weights is None:
        raise ValueError("Point SBP evidence requires boundary_quadrature_weights.")
    count = discretization.state_shape[0]
    mass = np.diag(np.asarray(discretization.quadrature_weights))
    boundary = np.diag(np.asarray(boundary_weights))
    maximum_green = 0.0
    maximum_conservation = 0.0
    for axis in range(discretization.spatial_dimension):
        identity = jnp.eye(count)
        derivative = np.asarray(
            jax.vmap(
                lambda column, _axis=axis: discretization.partial_derivative(
                    column,
                    axis=_axis,
                ),
                in_axes=1,
                out_axes=1,
            )(identity)
        )
        normal = np.asarray(discretization.plan.boundary_normals)[:, axis]
        boundary_form = boundary @ np.diag(normal)
        green = mass @ derivative + derivative.T @ mass - boundary_form
        conservation = np.ones(count) @ mass @ derivative
        maximum_green = max(maximum_green, float(np.max(np.abs(green))))
        maximum_conservation = max(
            maximum_conservation, float(np.max(np.abs(conservation)))
        )
    passed = maximum_green <= threshold and maximum_conservation <= threshold
    report_id = canonical_fingerprint(
        {
            "kind": "point-sbp-report",
            "discretization": discretization.prepared_id,
            "green": maximum_green,
            "conservation": maximum_conservation,
            "tolerance": threshold,
        }
    )
    return PointSBPReport(
        maximum_green,
        maximum_conservation,
        passed,
        report_id,
    )


class DissipativePointDiffusion(StrictModule, NonTrainableState):
    discretization: PreparedPointCloudDiscretization
    gradient_matrices: tuple[Array, ...]
    diffusivity: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedPointCloudDiscretization,
        diffusivity: ArrayLike = 1.0,
        /,
    ):
        count = discretization.state_shape[0]
        coefficient = jnp.broadcast_to(jnp.asarray(diffusivity, dtype=float), (count,))
        coefficient = eqx.error_if(
            coefficient,
            jnp.any(~jnp.isfinite(coefficient)) | jnp.any(coefficient <= 0.0),
            "Point diffusivity must be finite and positive.",
        )
        identity = jnp.eye(count)
        gradients = tuple(
            jax.vmap(
                lambda column, axis=axis: discretization.partial_derivative(
                    column, axis=axis
                ),
                in_axes=1,
                out_axes=1,
            )(identity)
            for axis in range(discretization.spatial_dimension)
        )
        self.discretization = discretization
        self.gradient_matrices = gradients
        self.diffusivity = coefficient
        self.operator_id = canonical_fingerprint(
            {
                "kind": "dissipative-point-diffusion",
                "discretization": discretization.prepared_id,
                "diffusivity": array_tree_fingerprint(coefficient),
            }
        )

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim < 1 or value.shape[0] != self.discretization.state_shape[0]:
            raise ValueError("Point diffusion values must begin with the point count.")
        payload_rank = value.ndim - 1
        reshape = (self.discretization.state_shape[0],) + (1,) * payload_rank
        mass = self.discretization.quadrature_weights.reshape(reshape)
        diffusivity = self.diffusivity.reshape(reshape)
        output = jnp.zeros_like(value)
        for gradient in self.gradient_matrices:
            flux = diffusivity * (gradient @ value)
            output = output - (jnp.conj(gradient.T) @ (mass * flux)) / mass
        return output

    def energy_rate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        reshape = (self.discretization.state_shape[0],) + (1,) * (value.ndim - 1)
        mass = self.discretization.quadrature_weights.reshape(reshape)
        return jnp.real(jnp.vdot(value, mass * self.mv(value)))


class PointCloudPoissonResult(StrictModule):
    values: Array
    residual_norm: Array
    compatible: Array


def solve_point_cloud_poisson(
    discretization: PreparedPointCloudDiscretization,
    source: ArrayLike,
    boundary: PointBoundaryPlan,
    /,
    *,
    diffusivity: ArrayLike = 1.0,
) -> PointCloudPoissonResult:
    if not isinstance(discretization, PreparedPointCloudDiscretization):
        raise TypeError("discretization must be PreparedPointCloudDiscretization.")
    if not isinstance(boundary, PointBoundaryPlan):
        raise TypeError("boundary must be PointBoundaryPlan.")
    count = discretization.state_shape[0]
    dtype = jnp.result_type(source, boundary.values, float)
    source_ = jnp.asarray(source, dtype=dtype)
    if source_.shape != (count,) or boundary.values.shape != (count,):
        raise ValueError("Point Poisson source/boundary values must match point count.")
    boundary_mask = discretization.plan.boundary_mask
    diffusion = DissipativePointDiffusion(discretization, diffusivity)
    identity = jnp.eye(count, dtype=dtype)
    matrix = jax.vmap(diffusion.mv, in_axes=1, out_axes=1)(identity).astype(dtype)
    rhs = source_
    augmented = None
    augmented_rhs = None
    if boundary.kind == "dirichlet":
        matrix = jnp.where(boundary_mask[:, None], identity, matrix)
        rhs = jnp.where(boundary_mask, boundary.values.astype(dtype), rhs)
        compatibility = jnp.asarray(0.0, dtype=source_.real.dtype)
    else:
        normal_derivative = jnp.sum(
            discretization.gradient(identity)
            * discretization.plan.boundary_normals[:, None, :],
            axis=-1,
        )
        boundary_matrix = normal_derivative
        if boundary.kind == "robin":
            assert boundary.robin_coefficient is not None
            boundary_matrix = (
                boundary_matrix
                + boundary.robin_coefficient[:, None].astype(dtype) * identity
            )
        if boundary.kind == "neumann":
            boundary_weights = discretization.plan.boundary_quadrature_weights
            if boundary_weights is None:
                raise ValueError(
                    "Neumann point Poisson requires boundary_quadrature_weights."
                )
            volume_weights = discretization.quadrature_weights.astype(dtype)
            boundary_weights_ = boundary_weights.astype(dtype)
            mismatch = jnp.sum(volume_weights * source_) + jnp.sum(
                boundary_weights_ * boundary.values.astype(dtype)
            )
            compatibility = jnp.abs(mismatch)
            corrected_source = source_ - mismatch / jnp.sum(volume_weights)
            rhs = jnp.where(
                boundary_mask,
                boundary.values.astype(dtype),
                corrected_source,
            )
            matrix = jnp.where(boundary_mask[:, None], boundary_matrix, matrix)
            augmented = jnp.zeros((count + 1, count + 1), dtype=dtype)
            augmented = augmented.at[:count, :count].set(matrix)
            augmented = augmented.at[:count, count].set(volume_weights)
            augmented = augmented.at[count, :count].set(volume_weights)
            augmented_rhs = jnp.concatenate((rhs, jnp.zeros((1,), dtype=dtype)))
        else:
            matrix = jnp.where(boundary_mask[:, None], boundary_matrix, matrix)
            rhs = jnp.where(boundary_mask, boundary.values.astype(dtype), rhs)
            compatibility = jnp.asarray(0.0, dtype=source_.real.dtype)
    solve_matrix = matrix if augmented is None else augmented
    solve_rhs = rhs if augmented is None else augmented_rhs
    if solve_rhs is None:
        raise RuntimeError("Point Poisson right-hand side was not prepared.")
    solved = solve(
        LinearSystem(
            DenseLinearOperator(solve_matrix),
            problem_id=f"{discretization.prepared_id}:point-poisson",
        ),
        solve_rhs,
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    )
    values = solved.value if augmented is None else solved.value[:count]
    residual = solved.diagnostics.residual_norm
    return PointCloudPoissonResult(values, residual, compatibility <= 1e-8)


class PointConormalInterface(StrictModule):
    left_indices: Array
    right_indices: Array
    left_diffusivity: Array
    right_diffusivity: Array
    jump: Array
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_indices: ArrayLike,
        right_indices: ArrayLike,
        left_diffusivity: ArrayLike,
        right_diffusivity: ArrayLike,
        /,
        *,
        jump: ArrayLike = 0.0,
    ):
        left_host = np.asarray(left_indices)
        right_host = np.asarray(right_indices)
        if (
            left_host.ndim != 1
            or right_host.shape != left_host.shape
            or not np.issubdtype(left_host.dtype, np.signedinteger)
            or not np.issubdtype(right_host.dtype, np.signedinteger)
            or np.any(left_host < 0)
            or np.any(right_host < 0)
        ):
            raise ValueError(
                "Point interface indices must be paired nonnegative signed integers."
            )
        left = jnp.asarray(left_host, dtype=jnp.int32)
        right = jnp.asarray(right_host, dtype=jnp.int32)
        left_k = jnp.broadcast_to(jnp.asarray(left_diffusivity), left.shape)
        right_k = jnp.broadcast_to(jnp.asarray(right_diffusivity), left.shape)
        jump_ = jnp.broadcast_to(jnp.asarray(jump), left.shape)
        if bool(
            jnp.any(~jnp.isfinite(left_k) | (left_k <= 0.0))
            | jnp.any(~jnp.isfinite(right_k) | (right_k <= 0.0))
            | jnp.any(~jnp.isfinite(jump_))
        ):
            raise ValueError(
                "Point interface diffusivities must be finite and positive and "
                "jumps must be finite."
            )
        self.left_indices = left
        self.right_indices = right
        self.left_diffusivity = left_k
        self.right_diffusivity = right_k
        self.jump = jump_
        self.interface_id = canonical_fingerprint(
            {
                "kind": "point-conormal-interface",
                "left": array_tree_fingerprint(left),
                "right": array_tree_fingerprint(right),
                "left_diffusivity": array_tree_fingerprint(left_k),
                "right_diffusivity": array_tree_fingerprint(right_k),
                "jump": array_tree_fingerprint(jump_),
            }
        )

    def residual(self, left_flux: ArrayLike, right_flux: ArrayLike, /) -> Array:
        left = jnp.asarray(left_flux)[self.left_indices]
        right = jnp.asarray(right_flux)[self.right_indices]
        return self.right_diffusivity * right - self.left_diffusivity * left - self.jump


class DistributedPointPartition(StrictModule, NonTrainableState):
    owners: Array
    partition_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, owners: ArrayLike, partition_count: int, /):
        owners_host = np.asarray(owners)
        count = int(partition_count)
        if (
            owners_host.ndim != 1
            or count <= 0
            or not np.issubdtype(owners_host.dtype, np.signedinteger)
        ):
            raise ValueError(
                "Distributed point owners must be a signed-integer vector and "
                "partition_count must be positive."
            )
        owners_ = jnp.asarray(owners_host, dtype=jnp.int32)
        if bool(jnp.any((owners_ < 0) | (owners_ >= count))):
            raise ValueError("Point owners are outside partition_count.")
        self.owners = owners_
        self.partition_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "distributed-point-partition",
                "owners": array_tree_fingerprint(owners_),
                "partition_count": count,
            }
        )

    def halo_routes(
        self,
        discretization: PreparedPointCloudDiscretization,
        /,
    ) -> tuple[Array, Array]:
        source = discretization.relation.source_indices
        target_owner = self.owners[:, None]
        source_owner = self.owners[source]
        remote = source_owner != target_owner
        return source[remote], jnp.broadcast_to(target_owner, source.shape)[remote]


__all__ = [
    "DissipativePointDiffusion",
    "DistributedPointPartition",
    "PointBoundaryKind",
    "PointBoundaryPlan",
    "PointCloudPoissonResult",
    "PointConormalInterface",
    "PointSBPReport",
    "point_sbp_report",
    "solve_point_cloud_poisson",
]
