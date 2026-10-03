#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    isotropic_elasticity_coefficients,
    LocalStencilPolicy,
    PointBlockSystemPlan,
    PointGhostLayerPlan,
)


def _square(side: int = 11) -> tuple[np.ndarray, PreparedPointCloudDiscretization]:
    grid = np.meshgrid(
        np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side), indexing="ij"
    )
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    rng = np.random.default_rng(7)
    points[~boundary] += rng.uniform(-0.15, 0.15, (np.count_nonzero(~boundary), 2)) / (
        side - 1
    )
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    return points, cloud


def _coupling(p: Array, cross: float = 1.0) -> Array:
    k0 = jnp.asarray([[1.5, 0.2], [0.2, 1.0]])
    k1 = jnp.asarray([[1.0, -0.1], [-0.1, 1.3]])
    c = cross * (0.4 + 0.1 * p[0]) * jnp.eye(2)
    return jnp.stack((jnp.stack((k0, c)), jnp.stack((c, k1))))


def _field(p: Array) -> Array:
    return jnp.stack((jnp.sin(p[0]) * jnp.exp(0.5 * p[1]), jnp.cos(p[0] + p[1])))


def _flux(
    coupling: Callable[[Array], Array], field: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Independent ``sigma[a, i] = sum_bj C_abij d_j u_b`` by autodiff."""

    def flux(p: Array) -> Array:
        gradient = jax.jacfwd(field)(p)
        return jnp.sum(coupling(p) * gradient[None, :, None, :], axis=(1, 3))

    return flux


def _source(flux: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return lambda p: -jnp.trace(jax.jacfwd(flux)(p), axis1=1, axis2=2)


def _faces(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.flatnonzero(np.isclose(points[:, 0], 1.0))
    boundary = np.any(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    rest = np.flatnonzero(boundary & ~np.isclose(points[:, 0], 1.0))
    return right, rest


def test_cross_coupled_anisotropic_vector_system_is_genuinely_coupled() -> None:
    points, cloud = _square()
    p = jnp.asarray(points)
    exact = jax.vmap(_field)(p)
    flux = _flux(_coupling, _field)
    source = jax.vmap(_source(flux))(p)
    right, rest = _faces(points)
    walls = np.concatenate((right, rest))
    normal = np.tile([[1.0, 0.0]], (right.size, 1))
    traction = jax.vmap(flux)(p[right])[:, 1, 0]
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", walls, exact[walls, 0], label="u-walls"),
            PointBoundaryCondition(
                "neumann", right, traction, label="v-right", component=1, normals=normal
            ),
            PointBoundaryCondition(
                "dirichlet", rest, exact[rest, 1], label="v-walls", component=1
            ),
        ),
        row_count=points.shape[0],
        components=2,
    )
    plan = PointBlockSystemPlan(cloud, boundary, components=("u", "v"))
    result = plan.prepare(jax.vmap(_coupling)(p)).solve(source)
    error = float(jnp.max(jnp.abs(result.values - exact)))
    assert bool(result.successful)
    assert error < 5e-3
    assert float(result.coupling_evidence.minimum_eigenvalue) > 0.0
    # Channel-wise application (cross blocks removed) is a different PDE.
    channelwise = plan.prepare(jax.vmap(lambda q: _coupling(q, 0.0))(p)).solve(source)
    assert float(jnp.max(jnp.abs(channelwise.values - exact))) > max(10 * error, 1e-2)


def test_linear_elasticity_patch_with_traction_face() -> None:
    points, cloud = _square(9)
    p = jnp.asarray(points)
    gradient = jnp.asarray([[0.1, 0.2], [-0.05, 0.3]])
    exact = p @ gradient.T + jnp.asarray([0.01, -0.02])
    coefficients = isotropic_elasticity_coefficients(1.0, 0.5, 2)
    stress = jnp.sum(coefficients * gradient[None, :, None, :], axis=(1, 3))
    right, rest = _faces(points)
    traction = jnp.broadcast_to(stress[:, 0], (right.size, 2))
    normal = np.tile([[1.0, 0.0]], (right.size, 1))
    conditions = []
    for component, name in enumerate(("x", "y")):
        conditions.append(
            PointBoundaryCondition(
                "neumann",
                right,
                traction[:, component],
                label=f"traction-{name}",
                component=component,
                normals=normal,
            )
        )
        conditions.append(
            PointBoundaryCondition(
                "dirichlet",
                rest,
                exact[rest, component],
                label=f"clamp-{name}",
                component=component,
            )
        )
    boundary = PointBoundaryPlan(conditions, row_count=points.shape[0], components=2)
    # Traction rows on the ghost route: PDE at every free point, the traction
    # condition in the ghost's row; the default spectral assessment admits it.
    ghosts = PointGhostLayerPlan(boundary).prepare(cloud)
    plan = PointBlockSystemPlan(cloud, boundary, components=("ux", "uy"), ghosts=ghosts)
    prepared = plan.prepare(coefficients)
    result = prepared.solve(jnp.zeros_like(exact))
    assert prepared.stability.outcome == "admitted"
    assert bool(result.successful)
    np.testing.assert_allclose(result.values, exact, atol=1e-8)
    # The ghost values continue the affine field outside the domain.
    assert result.ghost_values is not None and result.ghost_extension_defect is not None
    outside = ghosts.points[cloud.state_shape[0] :]
    np.testing.assert_allclose(
        result.ghost_values, outside @ gradient.T + jnp.asarray([0.01, -0.02]), atol=1e-8
    )
    assert float(result.ghost_extension_defect) < 1e-8


def test_nonlinear_coupled_reaction_uses_prepared_newton() -> None:
    points, cloud = _square(9)
    p = jnp.asarray(points)
    exact = jax.vmap(_field)(p)
    constant = _coupling(jnp.zeros(2))
    flux = _flux(lambda _: constant, _field)
    source = jax.vmap(_source(flux))(p) + exact**3
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    boundary = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "dirichlet", rows, exact[rows, a], label=f"walls-{a}", component=a
            )
            for a in range(2)
        ),
        row_count=points.shape[0],
        components=2,
    )
    prepared = PointBlockSystemPlan(cloud, boundary, components=("u", "v")).prepare(
        constant
    )
    result = prepared.solve_nonlinear(lambda values, _: values**3, source)
    assert bool(result.successful)
    assert float(result.residual_norm) < 1e-7
    assert float(jnp.max(jnp.abs(result.values - exact))) < 5e-3
    # Dropping the reaction solves a different equation.
    linear = prepared.solve(source)
    assert float(jnp.max(jnp.abs(linear.values - exact))) > 1e-2


def test_floating_components_and_indefinite_coupling_are_refused() -> None:
    points, cloud = _square(7)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    normals = np.asarray(cloud.plan.boundary_normals)[rows]
    floating = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", rows, 0.0, label="u"),
            PointBoundaryCondition(
                "neumann", rows, 0.0, label="v", component=1, normals=normals
            ),
        ),
        row_count=points.shape[0],
        components=2,
    )
    with pytest.raises(ValueError, match="floats"):
        PointBlockSystemPlan(cloud, floating, components=("u", "v"))
    anchored = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition("dirichlet", rows, 0.0, label=f"w{a}", component=a)
            for a in range(2)
        ),
        row_count=points.shape[0],
        components=2,
    )
    plan = PointBlockSystemPlan(cloud, anchored, components=("u", "v"))
    indefinite = _coupling(jnp.zeros(2), cross=5.0)
    with pytest.raises(Exception, match="positive semidefinite"):
        plan.prepare(indefinite)
