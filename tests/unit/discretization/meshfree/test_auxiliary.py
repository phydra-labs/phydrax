#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    isotropic_elasticity_coefficients,
    LocalStencilPolicy,
    MechanicsStatus,
    meshfree_auxiliary_builder,
    meshfree_auxiliary_stiffness,
    MeshfreeElasticityPlan,
    MeshfreeNearNullspace,
    PointBlockSystemPlan,
    PointGhostLayerPlan,
)
from phydrax.operators.mechanics import LinearElasticityTensor
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


_LAMBDA, _MU = 1.0, 0.5


def _jittered_square(side: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.meshgrid(*(np.linspace(0.0, 1.0, side),) * 2, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    rng = np.random.default_rng(7)
    points[~boundary] += rng.uniform(-0.15, 0.15, (np.count_nonzero(~boundary), 2)) / (
        side - 1
    )
    return points, boundary


def _cloud(side: int) -> PreparedPointCloudDiscretization:
    points, boundary = _jittered_square(side)
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    return PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()


def _coefficients() -> Array:
    return isotropic_elasticity_coefficients(_LAMBDA, _MU, 2)


def _dense(operator: la.AbstractLinearOperator) -> np.ndarray:
    size = operator.source.size
    return np.asarray(jax.vmap(operator.mv, in_axes=1, out_axes=1)(jnp.eye(size)))


def _sparse(matrix: np.ndarray, space: la.ArraySpace) -> SparseCoordinateOperator:
    rows, columns = np.nonzero(matrix)
    return SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=matrix.shape[1],
            target_size=matrix.shape[0],
        ),
        jnp.asarray(matrix[rows, columns]),
        source=space,
        target=space,
    )


def _block(field: np.ndarray) -> np.ndarray:
    """Component-major coordinates of a ``(points, 2)`` field."""
    return field.T.reshape(-1)


def test_auxiliary_stiffness_is_exact_for_affine_energy_and_rigid_kernel() -> None:
    points, _ = _jittered_square(9)
    count = points.shape[0]
    space = la.ArraySpace((2 * count,))
    stiffness = meshfree_auxiliary_stiffness(
        points,
        _coefficients(),
        space,
    )
    matrix = _dense(stiffness.operator)
    gradient = np.asarray([[0.3, -0.2], [0.5, 0.1]])
    strain = 0.5 * (gradient + gradient.T)
    stress = _LAMBDA * np.trace(strain) * np.eye(2) + 2.0 * _MU * strain
    affine = _block(points @ gradient.T)
    rotation = _block(np.stack((-points[:, 1], points[:, 0]), axis=1))
    translations = (
        _block(np.tile([1.0, 0.0], (count, 1))),
        _block(np.tile([0.0, 1.0], (count, 1))),
    )

    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
    # Continuous P1 fields are exact for affine displacements: the energy is
    # area * sigma : eps over the triangulated unit square.
    np.testing.assert_allclose(
        affine @ matrix @ affine, np.sum(stress * strain), rtol=1e-12
    )
    for mode in (rotation, *translations):
        np.testing.assert_allclose(matrix @ mode, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(np.asarray(stiffness.lumped_mass)), 1.0, rtol=1e-12)


def test_eliminated_auxiliary_stiffness_is_positive_definite() -> None:
    points, boundary = _jittered_square(9)
    count = points.shape[0]
    eliminated = np.concatenate((boundary, boundary))
    stiffness = meshfree_auxiliary_stiffness(
        points,
        _coefficients(),
        la.ArraySpace((2 * count,)),
        eliminated=eliminated,
    )
    matrix = _dense(stiffness.operator)

    np.testing.assert_array_equal(
        matrix[eliminated][:, eliminated], np.eye(eliminated.sum())
    )
    np.testing.assert_array_equal(matrix[eliminated][:, ~eliminated], 0.0)
    assert np.linalg.eigvalsh(matrix)[0] > 0.0


def test_auxiliary_action_is_the_block_triangular_schur_inverse() -> None:
    rng = np.random.default_rng(3)
    points, boundary = _jittered_square(6)
    count = points.shape[0]
    space = la.ArraySpace((count,))
    stiffness = meshfree_auxiliary_stiffness(
        points, jnp.eye(2)[None, None], space, eliminated=boundary & (points[:, 0] < 0.5)
    )
    auxiliary = _dense(stiffness.operator)
    trace = boundary & (points[:, 0] >= 0.5)
    # A nonsymmetric system with the auxiliary sparsity plus a perturbation.
    system = auxiliary + np.where(
        auxiliary != 0, rng.uniform(-0.05, 0.05, auxiliary.shape), 0.0
    )
    measure = rng.uniform(0.5, 2.0, count)
    builder = meshfree_auxiliary_builder(
        stiffness,
        trace=jnp.asarray(trace),
        row_measure=jnp.asarray(measure),
        interior=la.SparseFactorizationPreconditionerBuilder(),
    )
    materialization = la.MaterializationPolicy()
    action = builder.prepare(_sparse(system, space), materialization=materialization)
    residual = rng.normal(size=count)
    scale = np.where(
        np.asarray(stiffness.eliminated), 1.0, np.asarray(stiffness.lumped_mass) / measure
    )

    def expected(matrix: np.ndarray) -> np.ndarray:
        interior = np.linalg.solve(auxiliary, np.where(trace, 0.0, scale * residual))
        interior[trace] = 0.0
        interior[trace] = np.linalg.solve(
            matrix[np.ix_(trace, trace)], residual[trace] - matrix[trace] @ interior
        )
        return interior

    np.testing.assert_allclose(
        action.apply(jnp.asarray(residual)), expected(system), atol=1e-10
    )
    updated = system + np.where(system != 0, 0.1 * np.abs(system), 0.0)
    refreshed = builder.refresh(
        action, _sparse(updated, space), materialization=materialization
    )
    np.testing.assert_allclose(
        refreshed.apply(jnp.asarray(residual)), expected(updated), atol=1e-10
    )
    with pytest.raises(ValueError, match="positive"):
        meshfree_auxiliary_builder(stiffness, row_measure=jnp.zeros(count))


def _stress(displacement: Callable[[Array], Array]) -> Callable[[Array], Array]:
    def stress(x: Array) -> Array:
        gradient = jax.jacfwd(displacement)(x)
        strain = 0.5 * (gradient + gradient.T)
        return _LAMBDA * jnp.trace(strain) * jnp.eye(2) + 2.0 * _MU * strain

    return stress


def _smooth(x: Array) -> Array:
    return jnp.stack(
        (0.1 * jnp.sin(2.0 * x[0]) * jnp.exp(x[1]), 0.05 * jnp.cos(x[0] + 2.0 * x[1]))
    )


def _traction_problem(cloud: PreparedPointCloudDiscretization) -> PointBoundaryPlan:
    points = cloud.points
    exact = jax.vmap(_smooth)(points)
    right = np.flatnonzero(np.isclose(np.asarray(points[:, 0]), 1.0))
    rest = np.flatnonzero(
        np.asarray(cloud.plan.boundary_mask) & ~np.isclose(np.asarray(points[:, 0]), 1.0)
    )
    traction = jax.vmap(_stress(_smooth))(points[right])[:, :, 0]
    normals = np.tile([[1.0, 0.0]], (right.size, 1))
    conditions = []
    for a in range(2):
        conditions.append(
            PointBoundaryCondition(
                "neumann",
                right,
                traction[:, a],
                label=f"t{a}",
                component=a,
                normals=normals,
            )
        )
        conditions.append(
            PointBoundaryCondition(
                "dirichlet", rest, exact[rest, a], label=f"d{a}", component=a
            )
        )
    return PointBoundaryPlan(conditions, row_count=points.shape[0], components=2)


def test_default_elasticity_iterations_stay_bounded_under_refinement() -> None:
    # Collocated elasticity with a traction face (default ghost route): the
    # auxiliary-space default needs a bounded GMRES iteration count as the
    # cloud is refined (measured 57, 79, 70 at N = 289, 1089, 4225; an exact
    # interior solve of the auxiliary stiffness needs 50 and 64), while the
    # solution converges to the analytic displacement. Collocated multigrid
    # and ILU are refused from N ~ 1000 on rollers.
    iterations, errors = [], []
    tensor = LinearElasticityTensor.isotropic(2, lame_lambda=_LAMBDA, shear_modulus=_MU)
    for side in (17, 33):
        cloud = _cloud(side)
        plan = MeshfreeElasticityPlan(cloud, _traction_problem(cloud), tensor)
        body = jax.vmap(
            lambda x: -jnp.trace(jax.jacfwd(_stress(_smooth))(x), axis1=1, axis2=2)
        )
        result = plan.prepare().solve(body(cloud.points))
        assert int(result.status) == MechanicsStatus.ACCEPTED
        iterations.append(int(result.block.linear_result.diagnostics.iterations))
        errors.append(
            float(jnp.max(jnp.abs(result.displacement - jax.vmap(_smooth)(cloud.points))))
        )

    assert iterations[1] <= 1.5 * iterations[0]
    assert errors[1] < 0.6 * errors[0]


def test_auxiliary_route_refuses_unanchored_components_and_conflicts() -> None:
    cloud = _cloud(7)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    normals = np.asarray(cloud.plan.boundary_normals)[rows]
    half_free = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", rows, 0.0, label="x", component=0),
            PointBoundaryCondition(
                "robin",
                rows,
                0.0,
                label="y",
                component=1,
                normals=normals,
                robin_coefficient=1.0,
            ),
        ),
        row_count=cloud.state_shape[0],
        components=2,
    )
    tensor = LinearElasticityTensor.isotropic(2, lame_lambda=_LAMBDA, shear_modulus=_MU)
    with pytest.raises(ValueError, match="Dirichlet rows in component 1"):
        MeshfreeElasticityPlan(cloud, half_free, tensor)
    with pytest.raises(ValueError, match="owns its preconditioning"):
        MeshfreeElasticityPlan(
            cloud,
            _traction_problem(cloud),
            tensor,
            linear_policy=la.LinearSolvePolicy(la.GMRES()),
            preconditioner="ilu",
        )
    # The collocated multigrid route remains available for such boundaries.
    MeshfreeElasticityPlan(cloud, half_free, tensor, preconditioner="multigrid")


def test_coefficient_refresh_rebuilds_the_auxiliary_hierarchy() -> None:
    cloud = _cloud(13)
    boundary = _traction_problem(cloud)
    # Ghost-extended traction rows: the auxiliary stiffness triangulates the
    # cloud and ghost points, and the ghost condition rows are its trace rows.
    plan = PointBlockSystemPlan(
        cloud,
        boundary,
        components=("x", "y"),
        auxiliary=MeshfreeNearNullspace("rigid-body"),
        ghosts=PointGhostLayerPlan(boundary).prepare(cloud),
    )
    body = jax.vmap(
        lambda x: -jnp.trace(jax.jacfwd(_stress(_smooth))(x), axis1=1, axis2=2)
    )
    source = body(cloud.points)
    prepared = plan.prepare(_coefficients())
    stiffer = prepared.refresh(2.0 * _coefficients())
    first = prepared.solve(source)
    second = stiffer.solve(source)

    assert (
        stiffer.auxiliary_stiffness is not None
        and prepared.auxiliary_stiffness is not None
    )
    free = ~np.asarray(prepared.auxiliary_stiffness.eliminated)
    np.testing.assert_allclose(
        _dense(stiffer.auxiliary_stiffness.operator)[np.ix_(free, free)],
        2.0 * _dense(prepared.auxiliary_stiffness.operator)[np.ix_(free, free)],
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(first.successful) and bool(second.successful)
    # The rebuilt preconditioner solves the refreshed original equations.
    assert float(second.residual_norm) <= float(second.residual_tolerance)
