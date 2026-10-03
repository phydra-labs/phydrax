#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.discretization import (
    FacetTraceRule,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCollocationStabilityRefusal,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    PointGhostLayerPlan,
    PointInterfaceCondition,
    PointSideSupportPlan,
    prepare_point_sbp_derivatives,
    sample_boundary_atlas,
)
from phydrax.geometry import circle_boundary_atlas
from phydrax.optim import ConvexProgramStatus


def _bimaterial() -> tuple[Array, PreparedPointCloudDiscretization, np.ndarray]:
    x = np.linspace(0.0, 1.0, 21)
    boundary = np.isin(np.arange(21), (0, 20))
    normals = np.zeros((21, 1))
    normals[0, 0], normals[20, 0] = -1.0, 1.0
    cloud = PointCloudPlan(
        x[:, None],
        np.full(21, 1.0 / 21),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    membership = np.stack((x <= 0.5, x >= 0.5), axis=1)
    return jnp.asarray(x), cloud, membership


def test_discontinuous_coefficient_uses_one_sided_support_and_flux_transmission() -> None:
    x, cloud, membership = _bimaterial()
    # k=1 on [0, 1/2], k=4 on [1/2, 1]. u_L = x² + x and
    # u_R = (x-1/2)² + (x-1/2)/2 + 3/4 are continuous with equal flux
    # 1 * u_L'(1/2) = 4 * u_R'(1/2) = 2, and -(k u')' = -2 / -8.
    exact = jnp.where(x <= 0.5, x * x + x, (x - 0.5) ** 2 + 0.5 * (x - 0.5) + 0.75)
    source = jnp.where(x <= 0.5, -2.0, -8.0)
    support = PointSideSupportPlan(cloud, membership, sides=("left", "right")).prepare()
    evidence = support.evidence
    assert evidence.admitted
    assert np.asarray(evidence.one_sided).tolist() == [i == 10 for i in range(21)]
    left = support.family("left")
    used = np.asarray(left.relation.source_indices)[np.asarray(left.relation.valid)]
    assert used.max() <= 10
    right = support.family("right")
    right_rows = np.asarray(right.relation.source_indices)[10][
        np.asarray(right.relation.valid)[10]
    ]
    assert right_rows.min() >= 10
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet",
                np.asarray([0, 20]),
                exact[jnp.asarray([0, 20])],
                label="ends",
            ),
        ),
        row_count=21,
    )
    interface = PointInterfaceCondition(
        np.asarray([10]),
        np.asarray([[1.0]]),
        label="material",
        minus="left",
        plus="right",
    )
    plan = PointCloudPoissonPlan(cloud, boundary, sides=support, interfaces=(interface,))
    result = plan.prepare((jnp.ones(21), jnp.full(21, 4.0))).solve(source)
    assert bool(result.successful)
    np.testing.assert_allclose(result.values, exact, atol=1e-9)
    # Fault adequacy: one global product-rule stencil across the jump fails.
    jump = jnp.where(x <= 0.5, 1.0, 4.0)
    smeared = PointCloudPoissonPlan(cloud, boundary).prepare(jump).solve(source)
    assert float(jnp.max(jnp.abs(smeared.values - exact))) > 1e-2


def test_side_support_refuses_undersampled_sides_and_unowned_shared_rows() -> None:
    x, cloud, membership = _bimaterial()
    thin = np.stack((np.ones(21, dtype=np.bool_), np.asarray(x) >= 0.95), axis=1)
    with pytest.raises(ValueError, match="fewer points than the polynomial basis"):
        PointSideSupportPlan(cloud, thin, sides=("left", "right"))
    support = PointSideSupportPlan(cloud, membership, sides=("left", "right")).prepare()
    boundary = PointBoundaryPlan(
        (PointBoundaryCondition("dirichlet", np.asarray([0, 20]), 0.0, label="ends"),),
        row_count=21,
    )
    with pytest.raises(ValueError, match="need an interface or boundary condition"):
        PointCloudPoissonPlan(cloud, boundary, sides=support)
    with pytest.raises(ValueError, match="belong to both of its sides"):
        PointCloudPoissonPlan(
            cloud,
            boundary,
            sides=support,
            interfaces=(
                PointInterfaceCondition(
                    np.asarray([9]),
                    np.asarray([[1.0]]),
                    label="wrong",
                    minus="left",
                    plus="right",
                ),
            ),
        )


def _exact(p: Array) -> Array:
    return jnp.exp(0.5 * p[0]) * jnp.cos(p[1])


def _k(p: Array) -> Array:
    return 1.0 + 0.2 * p[0]


def test_atlas_boundary_samples_drive_robin_disk_problem() -> None:
    atlas = circle_boundary_atlas(jnp.zeros(2), jnp.asarray(1.0), source_id="unit-disk")
    samples = sample_boundary_atlas(
        atlas, FacetTraceRule("gauss-legendre", points=10), "edge"
    )
    boundary_points = np.asarray(samples.points)
    np.testing.assert_allclose(np.sum(np.asarray(samples.measure)), 2 * np.pi, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(samples.normals), boundary_points, atol=1e-12)
    count = 220
    index = np.arange(count) + 0.5
    radius = 0.94 * np.sqrt(index / count)
    angle = index * np.pi * (3.0 - np.sqrt(5.0))
    interior = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    points = np.concatenate((boundary_points, interior))
    rows = np.arange(boundary_points.shape[0])
    total = points.shape[0]
    normals = np.zeros_like(points)
    normals[rows] = np.asarray(samples.normals)
    weights = np.zeros(total)
    weights[rows] = np.asarray(samples.measure)
    cloud = PointCloudPlan(
        points,
        np.full(total, np.pi / total),
        boundary_mask=np.isin(np.arange(total), rows),
        boundary_normals=normals,
        boundary_quadrature_weights=weights,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    p = jnp.asarray(points)
    exact = jax.vmap(_exact)(p)
    k = jax.vmap(_k)(p)
    source = jax.vmap(
        lambda q: -jnp.trace(jax.jacfwd(lambda r: _k(r) * jax.grad(_exact)(r))(q))
    )(p)
    conormal = k[rows] * jnp.sum(
        jax.vmap(jax.grad(_exact))(p[rows]) * samples.normals, axis=1
    )
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "robin",
                rows,
                conormal + exact[rows],
                label="circle",
                normals=samples.normals,
                measure=samples.measure,
                robin_coefficient=1.0,
            ),
        ),
        row_count=total,
    )
    result = PointCloudPoissonPlan(cloud, boundary).prepare(k).solve(source)
    assert bool(result.successful)
    assert float(jnp.max(jnp.abs(result.values - exact))) < 1e-2


def _segment(mass: np.ndarray) -> PreparedPointCloudDiscretization:
    x = np.linspace(0.0, 1.0, mass.size)
    boundary = np.isin(np.arange(mass.size), (0, mass.size - 1))
    normals = np.zeros((mass.size, 1))
    normals[0, 0], normals[-1, 0] = -1.0, 1.0
    return PointCloudPlan(
        x[:, None],
        mass,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary.astype(np.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=3,
    ).prepare()


def test_constrained_sbp_derivative_satisfies_full_identity_when_feasible() -> None:
    count = 11
    h = 1.0 / (count - 1)
    trapezoid = np.full(count, h)
    trapezoid[[0, -1]] = h / 2
    cloud = _segment(trapezoid)
    result = prepare_point_sbp_derivatives(cloud, reproduction_degree=1)
    assert int(result.status[0]) == int(ConvexProgramStatus.OPTIMAL)
    assert bool(result.successful)
    # Independent dense audit of M D + Dᵀ M = B and a manufactured boundary flux.
    indices = np.asarray(result.relation.source_indices)
    valid = np.asarray(result.relation.valid)
    dense = np.zeros((count, count))
    np.add.at(
        dense,
        (
            np.repeat(np.arange(count), indices.shape[1])[valid.reshape(-1)],
            indices[valid],
        ),
        np.asarray(result.weights[0])[valid],
    )
    mass = np.diag(trapezoid)
    flux = np.zeros((count, count))
    flux[0, 0], flux[-1, -1] = -1.0, 1.0
    np.testing.assert_allclose(mass @ dense + dense.T @ mass, flux, atol=1e-6)
    x = np.linspace(0.0, 1.0, count)
    np.testing.assert_allclose(dense @ x, np.ones(count), atol=1e-6)
    u, v = np.sin(2 * x), np.cos(3 * x)
    pairing = u @ mass @ (dense @ v) + v @ mass @ (dense @ u)
    np.testing.assert_allclose(pairing, u[-1] * v[-1] - u[0] * v[0], atol=1e-6)


def test_constrained_sbp_reports_infeasible_cubature_instead_of_relabeling() -> None:
    count = 11
    # Uniform weights integrate neither 1 nor x exactly on [0, 1]; a diagonal-
    # norm first-order SBP derivative cannot exist for this cubature.
    cloud = _segment(np.full(count, 1.0 / (count - 1)))
    result = prepare_point_sbp_derivatives(cloud, reproduction_degree=1)
    assert int(result.status[0]) == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    assert bool(result.certificate_valid[0])
    assert not bool(result.successful)


# Ghost-layer PDE+BC route. References are independent analytic fields
# differentiated by jax autodiff; spectra come from an independent numpy eig.

_PHS = LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3)


def _exact(x: Array) -> Array:
    return jnp.exp(0.5 * x[0]) * jnp.sin(1.0 + jnp.sum(x[1:]))


def _conductivity(x: Array) -> Array:
    return 1.0 + 0.3 * x[0] + 0.2 * jnp.sum(x[1:])


def _face_problem(
    dimension: int, side: int, *, robin: bool = False, flip: float = 1.0
) -> tuple[PreparedPointCloudDiscretization, PointBoundaryPlan, Array, Array, Array]:
    """Unit cube, flux condition on the open x=1 face, Dirichlet elsewhere."""
    axis = np.linspace(0.0, 1.0, side + 1)
    grid = np.meshgrid(*([axis] * dimension), indexing="ij")
    points = np.stack([coordinate.reshape(-1) for coordinate in grid], axis=1)
    on_face = np.isclose(points, 0.0) | np.isclose(points, 1.0)
    boundary = np.any(on_face, axis=1)
    rng = np.random.default_rng(3)
    points[~boundary] += (
        rng.uniform(-0.15, 0.15, (np.count_nonzero(~boundary), dimension)) / side
    )
    count = points.shape[0]
    x = jnp.asarray(points)
    exact = jax.vmap(_exact)(x)
    conductivity = jax.vmap(_conductivity)(x)

    def flux(p: Array) -> Array:
        return _conductivity(p) * jax.grad(_exact)(p)

    source = jax.vmap(lambda p: -jnp.trace(jax.jacfwd(flux)(p)))(x)
    face = np.flatnonzero(np.isclose(points[:, 0], 1.0) & ~np.any(on_face[:, 1:], axis=1))
    walls = np.flatnonzero(boundary & ~np.isin(np.arange(count), face))
    normals = np.zeros((face.size, dimension))
    normals[:, 0] = 1.0
    conormal = jax.vmap(flux)(x)[face, 0]
    cloud = PointCloudPlan(
        x,
        jnp.full((count,), 1.0 / count),
        boundary_mask=boundary,
        boundary_normals=np.where(
            boundary[:, None], np.full(points.shape, 1.0 / np.sqrt(dimension)), 0.0
        ),
        stencil=_PHS,
    ).prepare()
    condition = (
        PointBoundaryCondition(
            "robin",
            face,
            conormal + 2.0 * exact[face],
            label="face",
            normals=flip * normals,
            robin_coefficient=2.0,
        )
        if robin
        else PointBoundaryCondition(
            "neumann", face, conormal, label="face", normals=flip * normals
        )
    )
    plan = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", walls, exact[walls], label="walls"),
            condition,
        ),
        row_count=count,
    )
    return cloud, plan, conductivity, exact, source


def _face_error(dimension: int, side: int, *, robin: bool) -> float:
    cloud, boundary, conductivity, exact, source = _face_problem(
        dimension, side, robin=robin
    )
    ghosts = PointGhostLayerPlan(boundary).prepare(cloud)
    result = (
        PointCloudPoissonPlan(cloud, boundary, ghosts=ghosts)
        .prepare(conductivity)
        .solve(source)
    )
    assert bool(result.successful)
    assert result.ghost_values is not None and result.ghost_values.shape == (
        ghosts.ghost_count,
    )
    return float(jnp.max(jnp.abs(result.values - exact)))


@pytest.mark.parametrize(
    "dimension,sides,robin",
    (((2, (12, 24), False)), ((2, (12, 24), True)), ((3, (6, 10), False))),
    ids=("2d-neumann", "2d-robin", "3d-neumann"),
)
def test_ghost_route_converges_at_second_order_on_flux_faces(
    dimension: int, sides: tuple[int, int], robin: bool
) -> None:
    coarse = _face_error(dimension, sides[0], robin=robin)
    fine = _face_error(dimension, sides[1], robin=robin)
    order = np.log(coarse / fine) / np.log(sides[1] / sides[0])
    # Measured orders 1.81 (2-D Neumann), 1.83 (2-D Robin), 2.14 (3-D Neumann)
    # for cubic-augmented PHS whose Laplacian is second order.
    assert order > 1.6


def _hexagonal_disk(spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Jittered hexagonal interior plus a ring of matching spacing."""
    rng = np.random.default_rng(0)
    m = int(np.ceil(1.2 / spacing)) + 2
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    interior = np.stack(
        (((i + 0.5 * (j % 2)) * spacing).ravel(), (j * spacing * np.sqrt(3) / 2).ravel()),
        axis=1,
    )
    interior = interior + rng.uniform(-0.2 * spacing, 0.2 * spacing, interior.shape)
    interior = interior[np.linalg.norm(interior, axis=1) < 1 - 0.5 * spacing]
    ring_count = int(round(2 * np.pi / spacing))
    angles = np.arange(ring_count) * (2 * np.pi / ring_count)
    points = np.concatenate(
        (np.stack((np.cos(angles), np.sin(angles)), axis=1), interior)
    )
    return points, np.arange(points.shape[0]) < ring_count


def _cubic_ball(spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Jittered cubic interior plus a Fibonacci sphere of matching spacing."""
    rng = np.random.default_rng(0)
    m = int(np.ceil(1.2 / spacing)) + 1
    axis = np.arange(-m, m + 1) * spacing
    interior = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    interior = interior + rng.uniform(-0.2 * spacing, 0.2 * spacing, interior.shape)
    interior = interior[np.linalg.norm(interior, axis=1) < 1 - 0.5 * spacing]
    surface_count = int(round(4 * np.pi / spacing**2))
    index = np.arange(surface_count) + 0.5
    polar = np.arccos(1 - 2 * index / surface_count)
    azimuth = np.pi * (1 + 5**0.5) * index
    surface = np.stack(
        (np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)),
        axis=1,
    )
    points = np.concatenate((surface, interior))
    return points, np.arange(points.shape[0]) < surface_count


def _neumann_ball(
    points: np.ndarray, boundary: np.ndarray, neighbors: int
) -> tuple[PreparedPointCloudDiscretization, PointBoundaryPlan, Array, Array]:
    """Pure Neumann -Δu = f on the unit ball, u = exp(sum(x)/d)."""
    count, dimension = points.shape
    x = jnp.asarray(points)
    rows = np.flatnonzero(boundary)
    normals = points[rows] / np.linalg.norm(points[rows], axis=1, keepdims=True)
    exact = jnp.exp(jnp.sum(x, axis=1) / dimension)
    cloud = PointCloudPlan(
        x,
        jnp.full((count,), 1.0 / count),
        boundary_mask=boundary,
        boundary_normals=np.where(boundary[:, None], points, 0.0),
        neighbors=neighbors,
        stencil=_PHS,
    ).prepare()
    flux = exact[rows] * jnp.sum(jnp.asarray(normals), axis=1) / dimension
    plan = PointBoundaryPlan(
        (PointBoundaryCondition("neumann", rows, flux, label="wall", normals=normals),),
        row_count=count,
    )
    return cloud, plan, exact, -exact / dimension


@pytest.mark.parametrize(
    "dimension,neighbors,eigenvalue",
    ((2, 20, 1.8412**2), (3, 40, 2.0816**2)),
    ids=("2d-disk", "3d-ball"),
)
def test_ghost_route_has_only_the_gauge_mode_where_square_neumann_is_refused(
    dimension: int, neighbors: int, eigenvalue: float
) -> None:
    points, boundary = (
        _hexagonal_disk(np.sqrt(np.pi / (1024 * 0.866)))
        if dimension == 2
        else _cubic_ball(0.175)
    )
    assert 1000 <= points.shape[0] <= 1100
    cloud, plan, exact, source = _neumann_ball(points, boundary, neighbors)
    if dimension == 2:
        # Fault adequacy: the flux-only boundary rows of square collocation
        # carry spurious nonpositive modes on this cloud and are refused.
        with pytest.raises(PointCollocationStabilityRefusal):
            PointCloudPoissonPlan(cloud, plan, compatibility="project").prepare()
    ghosts = PointGhostLayerPlan(plan).prepare(cloud)
    prepared = PointCloudPoissonPlan(
        cloud, plan, ghosts=ghosts, compatibility="project"
    ).prepare()
    assert prepared.stability is not None and prepared.stability.admitted
    dense = np.asarray(
        la.materialize(
            prepared.physical_assembly.operator,
            la.MaterializationPolicy(max_entries=4_000_000, max_bytes=1 << 26),
        )
    )
    count = points.shape[0]
    spectrum = np.linalg.eigvals(dense)
    scale = np.max(np.abs(spectrum))
    assert np.count_nonzero(spectrum.real <= 1e-8 * scale) == 1
    assert np.count_nonzero(np.abs(spectrum) <= 1e-8 * scale) == 1
    # Eliminating the ghosts through the boundary rows leaves the cloud
    # operator of the heat equation; past the gauge its lowest eigenvalue is
    # the continuum Neumann eigenvalue of the unit disk or ball.
    schur = dense[:count, :count] - dense[:count, count:] @ np.linalg.solve(
        dense[count:, count:], dense[count:, :count]
    )
    lowest = np.sort(np.linalg.eigvals(schur).real)
    assert abs(lowest[0]) <= 1e-8 * scale
    assert abs(lowest[1] / eigenvalue - 1.0) < 0.03
    result = prepared.solve(source)
    assert bool(result.successful)
    error = np.asarray(result.values - exact)
    assert np.max(np.abs(error - error.mean())) < (5e-5 if dimension == 2 else 2e-4)
    assert result.ghost_extension_defect is not None
    assert float(result.ghost_extension_defect) < 1e-3


def test_ghost_route_default_multigrid_iterations_stay_bounded() -> None:
    points, boundary = _hexagonal_disk(0.0301)
    assert 4000 <= points.shape[0] <= 4200
    cloud, plan, exact, source = _neumann_ball(points, boundary, 20)
    poisson = PointCloudPoissonPlan(
        cloud,
        plan,
        ghosts=PointGhostLayerPlan(plan).prepare(cloud),
        compatibility="project",
    )
    assert poisson.hierarchy is not None
    result = poisson.prepare().solve(source)
    assert bool(result.successful)
    # Measured multigrid-GMRES iterations 12/13/15 at about 1k/4k/16k points
    # (square flux-row collocation needed 147 at 4k with 30 neighbors).
    assert int(result.diagnostics.iterations) <= 20
    error = np.asarray(result.values - exact)
    assert np.max(np.abs(error - error.mean())) < 2e-5


def test_ghost_layer_refuses_inward_normals_foreign_clouds_and_other_boundaries() -> None:
    cloud, plan, _, _, _ = _face_problem(2, 12, flip=-1.0)
    # Inward normals put ghosts among the cloud's own samples.
    with pytest.raises(ValueError, match="minimum_separation"):
        PointGhostLayerPlan(plan).prepare(cloud)
    cloud, plan, _, _, _ = _face_problem(2, 12)
    other, other_plan, _, _, _ = _face_problem(2, 8)
    with pytest.raises(ValueError, match="another cloud"):
        PointCloudPoissonPlan(
            cloud, plan, ghosts=PointGhostLayerPlan(other_plan).prepare(other)
        )
    walls = plan.condition("walls")
    dirichlet_only = PointBoundaryPlan(
        (
            walls,
            PointBoundaryCondition(
                "dirichlet", plan.condition("face").rows, 0.0, label="face"
            ),
        ),
        row_count=plan.row_count,
    )
    with pytest.raises(ValueError, match="exactly this boundary"):
        PointCloudPoissonPlan(
            cloud, dirichlet_only, ghosts=PointGhostLayerPlan(plan).prepare(cloud)
        )
    with pytest.raises(ValueError, match="no Neumann or Robin rows"):
        PointGhostLayerPlan(dirichlet_only)
