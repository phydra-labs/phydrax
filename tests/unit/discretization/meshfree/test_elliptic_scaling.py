#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.discretization import (
    collocation_stability_assessment,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCollocationStabilityRefusal,
    PointDiffusionOperator,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeSchwarzPlan,
    MeshfreeSchwarzPolicy,
    MeshfreeSchwarzProlongation,
)


_PHS = LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3)


def _random_disk(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random interior (not quasi-uniform) plus an equispaced ring."""
    rng = np.random.default_rng(0)
    ring_count = int(np.ceil(4 * np.sqrt(count)))
    angles = np.arange(ring_count) * (2 * np.pi / ring_count)
    theta = rng.uniform(0, 2 * np.pi, count - ring_count)
    radius = 0.95 * np.sqrt(rng.uniform(0, 1, count - ring_count))
    points = np.concatenate(
        (
            np.stack((np.cos(angles), np.sin(angles)), axis=1),
            radius[:, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1),
        )
    )
    return points, np.arange(count) < ring_count


def _hexagonal_disk(spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Jittered hexagonal interior (quasi-uniform) plus a ring of matching spacing."""
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


def _dirichlet_disk(
    points: np.ndarray,
    boundary: np.ndarray,
    stencil: LocalStencilPolicy,
    neighbors: int,
) -> tuple[PreparedPointCloudDiscretization, PointBoundaryPlan, Array, Array, Array]:
    """Independent manufactured u=exp((x+y)/2), k=2+0.1(x+y), f=-u(k/2+0.1)."""
    count = points.shape[0]
    xy = jnp.asarray(points)
    ring = np.flatnonzero(boundary)
    cloud = PointCloudPlan(
        xy,
        jnp.full((count,), np.pi / count),
        boundary_mask=jnp.asarray(boundary),
        boundary_normals=jnp.where(jnp.asarray(boundary)[:, None], xy, 0.0),
        neighbors=neighbors,
        stencil=stencil,
    ).prepare()
    diffusivity = 2.0 + 0.1 * jnp.sum(xy, axis=1)
    exact = jnp.exp(jnp.sum(xy, axis=1) / 2)
    source = -exact * (diffusivity / 2 + 0.1)
    boundary_plan = PointBoundaryPlan(
        (PointBoundaryCondition("dirichlet", ring, exact[ring], label="ring"),),
        row_count=count,
    )
    return cloud, boundary_plan, diffusivity, exact, source


def _interior_spectrum(
    cloud: PreparedPointCloudDiscretization, diffusivity: Array, boundary: np.ndarray
) -> np.ndarray:
    """Dense eigenvalues of the positive-sign collocated operator's interior block."""
    stiffness = la.assemble_sparse(
        PointDiffusionOperator(cloud, diffusivity, form="collocated").stiffness()
    )
    dense = np.asarray(
        la.materialize(
            stiffness, la.MaterializationPolicy(max_entries=2_000_000, max_bytes=1 << 26)
        )
    )
    interior = ~boundary
    return np.linalg.eigvals(dense[np.ix_(interior, interior)])


def test_default_policy_solves_a_4096_point_dirichlet_disk_scalably() -> None:
    points, boundary = _hexagonal_disk(0.0301)
    assert 4000 <= points.shape[0] <= 4200
    cloud, boundary_plan, diffusivity, exact, source = _dirichlet_disk(
        points, boundary, _PHS, 20
    )
    plan = PointCloudPoissonPlan(cloud, boundary_plan)
    # Large square collocation defaults to the native meshfree multigrid cycle.
    assert plan.hierarchy is not None
    assert len(plan.hierarchy.spaces) >= 3
    prepared = plan.prepare(diffusivity)
    result = prepared.solve(source)
    assert bool(result.successful)
    assert float(result.residual_norm) <= float(result.residual_tolerance)
    # Declared iteration envelope: measured 14 multigrid-preconditioned GMRES
    # iterations (ILU needs 31 here and 64 at four times the points).
    assert int(result.diagnostics.iterations) <= 20
    # Second-order PHS-RBF-FD accuracy against the independent exact field.
    assert float(jnp.max(jnp.abs(result.values - exact))) < 3e-5
    assert prepared.stability is not None and prepared.stability.admitted


def test_spectrally_unstable_gmls_collocation_is_refused_with_evidence() -> None:
    points, boundary = _random_disk(1024)
    cloud, boundary_plan, diffusivity, _, _ = _dirichlet_disk(
        points, boundary, LocalStencilPolicy(polynomial_degree=2), 16
    )
    # Independent oracle: least-squares rows at nearly coincident random points
    # produce eigenvalues with nonpositive real part near zero.
    spectrum = _interior_spectrum(cloud, diffusivity, boundary)
    diagnostic = PointCloudPoissonPlan(
        cloud, boundary_plan, stability="diagnostic"
    ).prepare(diffusivity)
    stability = diagnostic.stability
    assert stability is not None and stability.spectrum is not None
    assert stability.outcome == "nonpositive-real-part"
    assert stability.assessed_rows == np.count_nonzero(~boundary)
    found = np.asarray(stability.spectrum.eigenvalues)
    assert bool(np.all(np.asarray(stability.spectrum.diagnostics.converged_mask)))
    # Every estimate is a dense eigenvalue, and the Re <= 0 count matches the
    # dense count among the eigenvalues no farther from zero.
    assert np.max(np.min(np.abs(found[:, None] - spectrum[None, :]), axis=1)) < 1e-6
    radius = np.max(np.abs(found)) * (1 + 1e-9)
    nearest = spectrum[np.abs(spectrum) <= radius]
    assert nearest.size == found.size
    assert stability.nonpositive_count == np.count_nonzero(nearest.real <= 0.0) > 0
    with pytest.raises(PointCollocationStabilityRefusal, match="nonpositive real part"):
        PointCloudPoissonPlan(cloud, boundary_plan).prepare(diffusivity)


def test_refresh_reuses_evidence_only_within_its_certified_perturbation() -> None:
    points, boundary = _random_disk(1024)
    cloud, boundary_plan, diffusivity, _, _ = _dirichlet_disk(
        points, boundary, LocalStencilPolicy(polynomial_degree=2), 16
    )
    x = jnp.asarray(points[:, 0])
    perturbed = diffusivity * (1.0 + 1e-3 * x)
    # Independent oracle for the refreshed operator.
    spectrum = _interior_spectrum(cloud, perturbed, boundary)
    prepared = PointCloudPoissonPlan(
        cloud,
        boundary_plan,
        stability="diagnostic",
        stability_refresh="reuse-within-perturbation",
    ).prepare(diffusivity)
    assert prepared.stability is not None and not prepared.stability.reused
    # Unchanged coefficients: zero perturbation, the refusal evidence carries.
    same = prepared.refresh(diffusivity)
    assert same.stability is not None and same.stability.reused
    assert same.stability.perturbation_bound == 0.0
    assert same.stability.outcome == "nonpositive-real-part"
    assert same.stability.nonpositive_count == prepared.stability.nonpositive_count
    # A 1e-3 coefficient change exceeds every carried pair's tolerance margin,
    # so the operator is reassessed (warm-started) instead of reused.
    changed = prepared.refresh(perturbed)
    assert changed.stability is not None and not changed.stability.reused
    assert changed.stability.outcome == "nonpositive-real-part"
    assert changed.stability.spectrum is not None
    found = np.asarray(changed.stability.spectrum.eigenvalues)
    # The warm-started reassessment matches the refreshed dense spectrum.
    assert np.max(np.min(np.abs(found[:, None] - spectrum[None, :]), axis=1)) < 1e-6


def test_phs_collocation_on_the_same_random_cloud_is_admitted_and_stable() -> None:
    points, boundary = _random_disk(1024)
    cloud, boundary_plan, diffusivity, exact, source = _dirichlet_disk(
        points, boundary, _PHS, 30
    )
    spectrum = _interior_spectrum(cloud, diffusivity, boundary)
    # The smallest real part approximates k * λ1 of the unit disk (≈ 2 * 5.78).
    assert np.min(spectrum.real) > 10.0
    prepared = PointCloudPoissonPlan(cloud, boundary_plan).prepare(diffusivity)
    stability = prepared.stability
    assert stability is not None and stability.admitted
    assert stability.nonpositive_count == 0
    np.testing.assert_allclose(
        stability.minimum_real_part, np.min(spectrum.real), rtol=1e-8
    )
    result = prepared.solve(source)
    assert bool(result.successful)
    assert float(jnp.max(jnp.abs(result.values - exact))) < 1e-4


def test_stable_random_cloud_with_a_nonpositive_diagonal_row_is_admitted() -> None:
    points, boundary = _random_disk(4096)
    cloud, boundary_plan, diffusivity, _, _ = _dirichlet_disk(points, boundary, _PHS, 20)
    prepared = PointCloudPoissonPlan(
        cloud, boundary_plan, linear_policy=la.LinearSolvePolicy(la.GMRES())
    ).prepare(diffusivity)
    stability = prepared.stability
    assert stability is not None and stability.admitted
    # The row-sign heuristic would refuse this cloud; its spectrum is stable
    # (dense reference: minimum real part 11.560 ≈ k λ1, none with Re <= 0).
    assert stability.nonpositive_diagonal_rows >= 1
    np.testing.assert_allclose(stability.minimum_real_part, 11.560193, rtol=1e-6)


def test_iterative_assessment_reuses_the_solve_preconditioner_with_the_lu_outcome() -> (
    None
):
    # Capacity selection: large preconditioned plans assess iteratively, small
    # or unpreconditioned ones through one sparse LU.
    assert isinstance(
        collocation_stability_assessment(
            16606, 20, 2, preconditioned=True
        ).transform_solve,
        la.LinearSolvePolicy,
    )
    assert isinstance(
        collocation_stability_assessment(
            16606, 20, 2, preconditioned=False
        ).transform_solve,
        la.SparseFactorizationPolicy,
    )
    points, boundary = _random_disk(4096)
    cloud, boundary_plan, diffusivity, exact, source = _dirichlet_disk(
        points, boundary, _PHS, 20
    )
    # The default multigrid plan, assessed through GMRES preconditioned by its
    # own V-cycle restricted to the assessed rows instead of a sparse LU.
    prepared = PointCloudPoissonPlan(
        cloud,
        boundary_plan,
        stability_assessment=collocation_stability_assessment(
            100_000, 20, 2, preconditioned=True
        ),
    ).prepare(diffusivity)
    stability = prepared.stability
    assert stability is not None and stability.admitted
    assert stability.spectrum is not None
    assert int(stability.spectrum.diagnostics.factorization_status) == 0
    # Same minimum real part as the LU route and the dense reference above.
    np.testing.assert_allclose(stability.minimum_real_part, 11.560193, rtol=1e-6)
    refreshed = prepared.refresh(2.0 * diffusivity)
    assert refreshed.stability is not None
    np.testing.assert_allclose(
        refreshed.stability.minimum_real_part, 2 * 11.560193, rtol=1e-6
    )
    result = refreshed.solve(2.0 * source)
    assert bool(result.successful)
    assert float(jnp.max(jnp.abs(result.values - exact))) < 1e-4


@pytest.mark.parametrize("prolongation", ["partition_of_unity", "adjoint"])
def test_batched_schwarz_term_equals_per_patch_additive_schwarz(
    prolongation: MeshfreeSchwarzProlongation,
) -> None:
    points, boundary = _random_disk(256)
    cloud, boundary_plan, diffusivity, _, _ = _dirichlet_disk(points, boundary, _PHS, 30)
    plan = PointCloudPoissonPlan(cloud, boundary_plan)
    operator = plan.prepare(diffusivity).assembly.operator
    schwarz = MeshfreeSchwarzPlan(
        cloud.points,
        cloud.relation,
        policy=MeshfreeSchwarzPolicy(core_points=64, overlap_layers=1),
    ).prepare(plan.solve_space)
    policy = la.MaterializationPolicy()
    per_patch = la.AdditiveSubspaceCorrectionBuilder(
        schwarz.terms(prolongation=prolongation)
    ).prepare(operator, materialization=policy)
    batched = la.AdditiveSubspaceCorrectionBuilder(
        (schwarz.block_term(prolongation=prolongation),)
    ).prepare(operator, materialization=policy)
    vector = jnp.asarray(np.random.default_rng(1).standard_normal(points.shape[0]))
    expected = per_patch.apply(vector)
    # Dense batched LU versus per-patch sparse LU: equal up to rounding.
    np.testing.assert_allclose(
        batched.apply(vector),
        expected,
        rtol=0.0,
        atol=1e-12 * float(jnp.max(jnp.abs(expected))),
    )
