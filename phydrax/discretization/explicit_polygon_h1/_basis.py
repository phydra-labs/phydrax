#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    inverse_small_linear,
    prepare_local_block_factorization,
    SmallLinearSolvePlan,
    solve_local_blocks,
)
from .._polygon_geometry import PolygonGeometry, PolygonTriangulation
from ._precision import ExplicitPolygonH1PrecisionPolicy
from ._spec import (
    ExplicitPolygonH1QuadraturePolicy,
    ExplicitPolygonH1QualificationPolicy,
)


class ExplicitPolygonH1BasisEvidence(StrictModule):
    """Per-cell geometry, reproduction, factorization, and spectrum evidence."""

    geometry_valid: Array
    fan_area_partition_error: Array
    minimum_fan_measure: Array
    factorization_valid: Array
    minimum_private_pivot: Array
    condensation_residual: Array
    boundary_identity_error: Array
    partition_error: Array
    partition_gradient_error: Array
    affine_value_error: Array
    affine_gradient_error: Array
    stiffness_symmetry_error: Array
    mass_symmetry_error: Array
    stiffness_rank: Array
    minimum_positive_stiffness_eigenvalue: Array
    mass_minimum_eigenvalue: Array
    stiffness_condition: Array
    mass_condition: Array
    finite: Array
    passed: Array


class ExplicitPolygonH1BlockData(StrictModule):
    """Runtime basis, fan metric, quadrature, and evidence for one arity block."""

    witness: Array
    fan_points: Array
    fan_measures: Array
    prolongation: Array
    basis_values: Array
    reference_gradients: Array
    physical_points: Array
    physical_weights: Array
    jacobians: Array
    inverse_jacobians: Array
    reference_points: Array
    reference_weights: Array
    evidence: ExplicitPolygonH1BasisEvidence
    arity: int = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)


def _scatter_triangle_matrices(
    matrices: Array,
    arity: int,
    /,
) -> Array:
    cells = matrices.shape[0]
    result = jnp.zeros((cells, arity + 1, arity + 1), dtype=matrices.dtype)
    for triangle in range(arity):
        routes = (arity, triangle, (triangle + 1) % arity)
        for local_row, row in enumerate(routes):
            for local_column, column in enumerate(routes):
                result = result.at[:, row, column].add(
                    matrices[:, triangle, local_row, local_column]
                )
    return result


def prepare_explicit_polygon_h1_basis(
    geometry: PolygonGeometry,
    triangulation: PolygonTriangulation,
    local_width: int,
    quadrature: ExplicitPolygonH1QuadraturePolicy,
    precision: ExplicitPolygonH1PrecisionPolicy,
    qualification: ExplicitPolygonH1QualificationPolicy,
    /,
) -> ExplicitPolygonH1BlockData:
    """Build one arity bucket of condensed discrete-harmonic P1 bases."""
    if not isinstance(geometry, PolygonGeometry):
        raise TypeError("geometry must be PolygonGeometry.")
    if not isinstance(triangulation, PolygonTriangulation):
        raise TypeError("triangulation must be PolygonTriangulation.")
    vertices = precision.geometry(geometry.vertices)
    cells, arity, dimension = vertices.shape
    width = int(local_width)
    if dimension != 2 or width < arity:
        raise ValueError("Explicit polygon basis requires planar padded vertex width.")
    witness_weights = precision.geometry(triangulation.witness_weights)
    witness = oe.contract("ci,cid->cd", witness_weights, vertices)
    following = jnp.roll(vertices, -1, axis=1)
    witness_rows = jnp.broadcast_to(witness[:, None, :], vertices.shape)
    fan_points = jnp.stack((witness_rows, vertices, following), axis=2)
    axis_one = vertices - witness_rows
    axis_two = following - witness_rows
    jacobians_by_triangle = jnp.stack((axis_one, axis_two), axis=-1)
    inverse_result = inverse_small_linear(
        SmallLinearSolvePlan(
            2,
            singular_tolerance=float(
                qualification.tolerance_multiplier
                * jnp.finfo(precision.factorization_dtype).eps
            ),
            maximum_condition=qualification.maximum_condition_number,
        ),
        precision.factorization(jacobians_by_triangle),
    )
    inverse_by_triangle = inverse_result.value
    determinants = inverse_result.determinant
    fan_measures = 0.5 * determinants

    from ...integration import (
        GaussLegendreRule,
        reference_rule_data,
        ReferenceTriangleRule,
    )

    reference_data = reference_rule_data(
        ReferenceTriangleRule(GaussLegendreRule(quadrature.cell_order))
    )
    reference_points = precision.basis(reference_data.points)
    reference_weights = precision.basis(reference_data.weights)
    reference_values = jnp.stack(
        (
            1.0 - reference_points[:, 0] - reference_points[:, 1],
            reference_points[:, 0],
            reference_points[:, 1],
        ),
        axis=-1,
    )
    reference_gradient = precision.basis(
        jnp.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    )
    physical_gradients = oe.contract(
        "ar,cnrd->cnad", reference_gradient, inverse_by_triangle
    )
    triangle_stiffness = fan_measures[:, :, None, None] * oe.contract(
        "cnad,cnbd->cnab", physical_gradients, physical_gradients
    )
    fine_stiffness = _scatter_triangle_matrices(
        precision.accumulation(triangle_stiffness), arity
    )
    mass_template = precision.accumulation(
        jnp.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)))
    )
    triangle_mass = fan_measures[:, :, None, None] * mass_template / 12.0
    fine_mass = _scatter_triangle_matrices(triangle_mass, arity)

    private = precision.factorization(
        fine_stiffness[:, arity : arity + 1, arity : arity + 1]
    )
    coupling = precision.factorization(fine_stiffness[:, arity : arity + 1, :arity])
    factorization = prepare_local_block_factorization(private, positive_definite=True)
    solved, solve_failed = solve_local_blocks(factorization, coupling)
    extension = -solved[:, 0, :]
    prolongation = jnp.zeros((cells, arity + 1, arity), dtype=extension.dtype)
    prolongation = prolongation.at[:, :arity, :].set(
        jnp.broadcast_to(jnp.eye(arity, dtype=extension.dtype), (cells, arity, arity))
    )
    prolongation = prolongation.at[:, arity, :].set(extension)

    basis_blocks = []
    gradient_blocks = []
    point_blocks = []
    weight_blocks = []
    jacobian_blocks = []
    inverse_blocks = []
    for triangle in range(arity):
        routes = jnp.asarray((arity, triangle, (triangle + 1) % arity))
        local_prolongation = prolongation[:, routes, :]
        basis_blocks.append(
            oe.contract("qa,can->cqn", reference_values, local_prolongation)
        )
        local_reference_gradient = oe.contract(
            "ar,can->cnr", reference_gradient, local_prolongation
        )
        gradient_blocks.append(
            jnp.broadcast_to(
                local_reference_gradient[:, None, :, :],
                (cells, reference_points.shape[0], arity, 2),
            )
        )
        point_blocks.append(
            witness[:, None, :]
            + oe.contract(
                "qr,cdr->cqd", reference_points, jacobians_by_triangle[:, triangle]
            )
        )
        weight_blocks.append(determinants[:, triangle, None] * reference_weights[None, :])
        jacobian_blocks.append(
            jnp.broadcast_to(
                jacobians_by_triangle[:, triangle, None, :, :],
                (cells, reference_points.shape[0], 2, 2),
            )
        )
        inverse_blocks.append(
            jnp.broadcast_to(
                inverse_by_triangle[:, triangle, None, :, :],
                (cells, reference_points.shape[0], 2, 2),
            )
        )
    basis = jnp.stack(tuple(basis_blocks), axis=1).reshape((cells, -1, arity))
    reference_gradients = jnp.stack(tuple(gradient_blocks), axis=1).reshape(
        (cells, -1, arity, 2)
    )
    physical_points = jnp.stack(tuple(point_blocks), axis=1).reshape((cells, -1, 2))
    physical_weights = jnp.stack(tuple(weight_blocks), axis=1).reshape((cells, -1))
    jacobians = jnp.stack(tuple(jacobian_blocks), axis=1).reshape((cells, -1, 2, 2))
    inverse_jacobians = jnp.stack(tuple(inverse_blocks), axis=1).reshape(
        (cells, -1, 2, 2)
    )
    if width > arity:
        basis = jnp.pad(basis, ((0, 0), (0, 0), (0, width - arity)))
        reference_gradients = jnp.pad(
            reference_gradients,
            ((0, 0), (0, 0), (0, width - arity), (0, 0)),
        )

    physical_basis_gradients = oe.contract(
        "cqnr,cqrd->cqnd", reference_gradients, inverse_jacobians
    )
    active_basis = basis[:, :, :arity]
    active_gradients = physical_basis_gradients[:, :, :arity]
    ones = jnp.ones((cells, arity), dtype=active_basis.dtype)
    reproduced = oe.contract("cqn,cnd->cqd", active_basis, vertices)
    reproduced_gradient = oe.contract("cnd,cqne->cqde", vertices, active_gradients)
    coarse_stiffness = oe.contract(
        "cai,cab,cbj->cij", prolongation, fine_stiffness, prolongation
    )
    coarse_mass = oe.contract("cai,cab,cbj->cij", prolongation, fine_mass, prolongation)
    stiffness_eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (coarse_stiffness + jnp.swapaxes(coarse_stiffness, -1, -2))
    )
    mass_eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (coarse_mass + jnp.swapaxes(coarse_mass, -1, -2))
    )
    scale = jnp.maximum(jnp.max(jnp.abs(coarse_stiffness), axis=(-2, -1)), 1.0)
    mass_scale = jnp.maximum(jnp.max(jnp.abs(coarse_mass), axis=(-2, -1)), 1.0)
    tolerance = (
        qualification.tolerance_multiplier * jnp.finfo(precision.certification_dtype).eps
    )
    positive_floor = tolerance * scale
    stiffness_rank = jnp.sum(stiffness_eigenvalues > positive_floor[:, None], axis=-1)
    positive_eigenvalue = stiffness_eigenvalues[:, 1]
    stiffness_condition = jnp.max(stiffness_eigenvalues, axis=-1) / jnp.maximum(
        positive_eigenvalue, jnp.finfo(stiffness_eigenvalues.dtype).tiny
    )
    mass_minimum = mass_eigenvalues[:, 0]
    mass_condition = jnp.max(mass_eigenvalues, axis=-1) / jnp.maximum(
        mass_minimum, jnp.finfo(mass_eigenvalues.dtype).tiny
    )
    area_scale = jnp.maximum(jnp.abs(geometry.areas), 1.0)
    condensation_scale = jnp.maximum(jnp.max(jnp.abs(coupling), axis=(-2, -1)), 1.0)
    condensation_residual = (
        jnp.max(jnp.abs(private * extension[:, None, :] + coupling), axis=(-2, -1))
        / condensation_scale
    )
    boundary_identity = jnp.max(
        jnp.abs(
            prolongation[:, :arity, :] - jnp.eye(arity, dtype=prolongation.dtype)[None]
        ),
        axis=(-2, -1),
    )
    partition_error = jnp.max(
        jnp.abs(oe.contract("cqn,cn->cq", active_basis, ones) - 1.0), axis=-1
    )
    partition_gradient_error = jnp.max(
        jnp.abs(jnp.sum(active_gradients, axis=2)), axis=(-2, -1)
    ) * jnp.maximum(geometry.diameters, 1.0)
    affine_value_error = jnp.max(
        jnp.abs(reproduced - physical_points), axis=(-2, -1)
    ) / jnp.maximum(geometry.diameters, 1.0)
    affine_gradient_error = jnp.max(
        jnp.abs(reproduced_gradient - jnp.eye(2)[None, None]), axis=(-3, -2, -1)
    )
    stiffness_symmetry = (
        jnp.max(
            jnp.abs(coarse_stiffness - jnp.swapaxes(coarse_stiffness, -1, -2)),
            axis=(-2, -1),
        )
        / scale
    )
    mass_symmetry = (
        jnp.max(jnp.abs(coarse_mass - jnp.swapaxes(coarse_mass, -1, -2)), axis=(-2, -1))
        / mass_scale
    )
    area_error = jnp.abs(jnp.sum(fan_measures, axis=1) - geometry.areas) / area_scale
    finite = (
        jnp.all(jnp.isfinite(prolongation), axis=(-2, -1))
        & jnp.all(jnp.isfinite(basis), axis=(-2, -1))
        & jnp.all(jnp.isfinite(reference_gradients), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(coarse_stiffness), axis=(-2, -1))
        & jnp.all(jnp.isfinite(coarse_mass), axis=(-2, -1))
    )
    factorization_valid = (~factorization.failed_blocks) & (~solve_failed)
    geometry_valid = geometry.evidence.valid & jnp.all(inverse_result.successful, axis=1)
    passed = (
        geometry_valid
        & factorization_valid
        & finite
        & jnp.all(fan_measures > tolerance * area_scale[:, None], axis=1)
        & (area_error <= tolerance)
        & (condensation_residual <= tolerance)
        & (boundary_identity <= tolerance)
        & (partition_error <= tolerance)
        & (partition_gradient_error <= tolerance)
        & (affine_value_error <= tolerance)
        & (affine_gradient_error <= tolerance)
        & (stiffness_symmetry <= tolerance)
        & (mass_symmetry <= tolerance)
        & (stiffness_rank == arity - 1)
        & (positive_eigenvalue > positive_floor)
        & (mass_minimum > tolerance * mass_scale)
        & (stiffness_condition <= qualification.maximum_condition_number)
        & (mass_condition <= qualification.maximum_condition_number)
    )
    evidence = ExplicitPolygonH1BasisEvidence(
        geometry_valid=geometry_valid,
        fan_area_partition_error=area_error,
        minimum_fan_measure=jnp.min(fan_measures, axis=1),
        factorization_valid=factorization_valid,
        minimum_private_pivot=private[:, 0, 0],
        condensation_residual=condensation_residual,
        boundary_identity_error=boundary_identity,
        partition_error=partition_error,
        partition_gradient_error=partition_gradient_error,
        affine_value_error=affine_value_error,
        affine_gradient_error=affine_gradient_error,
        stiffness_symmetry_error=stiffness_symmetry,
        mass_symmetry_error=mass_symmetry,
        stiffness_rank=stiffness_rank,
        minimum_positive_stiffness_eigenvalue=positive_eigenvalue,
        mass_minimum_eigenvalue=mass_minimum,
        stiffness_condition=stiffness_condition,
        mass_condition=mass_condition,
        finite=finite,
        passed=passed,
    )
    return ExplicitPolygonH1BlockData(
        witness=witness,
        fan_points=fan_points,
        fan_measures=fan_measures,
        prolongation=prolongation,
        basis_values=precision.basis(basis),
        reference_gradients=precision.basis(reference_gradients),
        physical_points=precision.geometry(physical_points),
        physical_weights=precision.basis(physical_weights),
        jacobians=precision.geometry(jacobians),
        inverse_jacobians=precision.geometry(inverse_jacobians),
        reference_points=reference_points,
        reference_weights=reference_weights,
        evidence=evidence,
        arity=arity,
        local_width=width,
        point_count=int(arity * reference_points.shape[0]),
        basis_id=canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-basis",
                "geometry": geometry.geometry_id,
                "triangulation": triangulation.triangulation_id,
                "quadrature": quadrature.policy_id,
                "precision": precision.policy_id,
                "qualification": qualification.policy_id,
                "arity": arity,
                "local_width": width,
            }
        ),
    )


__all__ = [
    "ExplicitPolygonH1BasisEvidence",
    "ExplicitPolygonH1BlockData",
    "prepare_explicit_polygon_h1_basis",
]
