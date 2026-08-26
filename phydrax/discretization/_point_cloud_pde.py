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
    count = discretization.state_shape[0]
    mass = np.diag(np.asarray(discretization.quadrature_weights))
    boundary = np.diag(np.asarray(discretization.plan.boundary_mask, dtype=float))
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
    passed = maximum_green <= tolerance and maximum_conservation <= tolerance
    report_id = canonical_fingerprint(
        {
            "kind": "point-sbp-report",
            "discretization": discretization.prepared_id,
            "green": maximum_green,
            "conservation": maximum_conservation,
            "tolerance": float(tolerance),
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
        mass = self.discretization.quadrature_weights
        output = jnp.zeros_like(value)
        for gradient in self.gradient_matrices:
            flux = self.diffusivity * (gradient @ value)
            output = output - (jnp.conj(gradient.T) @ (mass * flux)) / mass
        return output

    def energy_rate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        return jnp.real(
            jnp.vdot(value, self.discretization.quadrature_weights * self.mv(value))
        )


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
    count = discretization.state_shape[0]
    source_ = jnp.asarray(source)
    if source_.shape != (count,) or boundary.values.shape != (count,):
        raise ValueError("Point Poisson source/boundary values must match point count.")
    boundary_mask = discretization.plan.boundary_mask
    diffusion = DissipativePointDiffusion(discretization, diffusivity)
    identity = jnp.eye(count)
    matrix = jax.vmap(diffusion.mv, in_axes=1, out_axes=1)(identity)
    rhs = source_
    if boundary.kind == "dirichlet":
        matrix = jnp.where(boundary_mask[:, None], identity, matrix)
        rhs = jnp.where(boundary_mask, boundary.values, rhs)
    else:
        normal_derivative = jnp.sum(
            discretization.gradient(identity)
            * discretization.plan.boundary_normals[:, None, :],
            axis=-1,
        )
        boundary_matrix = normal_derivative
        if boundary.kind == "robin":
            boundary_matrix = (
                boundary_matrix + boundary.robin_coefficient[:, None] * identity
            )
        matrix = jnp.where(boundary_mask[:, None], boundary_matrix, matrix)
        rhs = jnp.where(boundary_mask, boundary.values, rhs)
        if boundary.kind == "neumann":
            weights = discretization.quadrature_weights
            compatibility = jnp.abs(jnp.sum(weights * rhs))
            rhs = rhs - jnp.sum(weights * rhs) / jnp.sum(weights)
            matrix = matrix.at[0].set(weights / jnp.sum(weights))
            rhs = rhs.at[0].set(0.0)
        else:
            compatibility = jnp.asarray(0.0)
    if boundary.kind == "dirichlet":
        compatibility = jnp.asarray(0.0)
    values = jnp.linalg.solve(matrix, rhs)
    residual = jnp.linalg.norm(matrix @ values - rhs)
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
        left = jnp.asarray(left_indices, dtype=jnp.int32)
        right = jnp.asarray(right_indices, dtype=jnp.int32)
        if left.ndim != 1 or right.shape != left.shape:
            raise ValueError("Point interface indices must be paired vectors.")
        left_k = jnp.broadcast_to(jnp.asarray(left_diffusivity), left.shape)
        right_k = jnp.broadcast_to(jnp.asarray(right_diffusivity), left.shape)
        jump_ = jnp.broadcast_to(jnp.asarray(jump), left.shape)
        if bool(jnp.any(left_k <= 0.0)) or bool(jnp.any(right_k <= 0.0)):
            raise ValueError("Point interface diffusivities must be positive.")
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
        owners_ = jnp.asarray(owners, dtype=jnp.int32)
        count = int(partition_count)
        if owners_.ndim != 1 or count <= 0:
            raise ValueError("Distributed point partition inputs are invalid.")
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
