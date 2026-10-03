# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    meshfree_coarse_correction_term,
    MeshfreePartitionOfUnity,
    MeshfreeSchwarzPlan,
    MeshfreeSchwarzPolicy,
    MeshfreeSchwarzProlongation,
    PreparedMeshfreeSchwarz,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


_MATERIALIZATION = la.MaterializationPolicy(max_entries=1 << 16, max_bytes=1 << 22)


def _disk_points(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    boundary_count = max(8, int(np.ceil(4 * np.sqrt(count))))
    angles = np.arange(boundary_count) * (2 * np.pi / boundary_count)
    ring = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    theta = rng.uniform(0, 2 * np.pi, count - boundary_count)
    radius = 0.95 * np.sqrt(rng.uniform(0, 1, count - boundary_count))
    interior = radius[:, None] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    return np.concatenate((ring, interior)), np.arange(count) < boundary_count


def _knn_problem(
    count: int, *, seed: int = 0, neighbors: int = 6
) -> tuple[np.ndarray, EdgeRelation, la.AbstractSparseLinearOperator, np.ndarray]:
    """Nonsymmetric k-nearest-neighbor graph diffusion with identity boundary rows.

    The dense matrix is returned as the independent NumPy reference.
    """
    points, boundary = _disk_points(count, seed)
    distance = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, 1 : neighbors + 1]
    rows = np.repeat(np.arange(count), neighbors)
    columns = nearest.reshape(-1)
    weights = np.where(boundary[rows], 0.0, 1.0 / distance[rows, columns])
    dense = np.zeros((count, count))
    np.add.at(dense, (rows, columns), -weights)
    dense[np.arange(count), np.arange(count)] = np.where(
        boundary, 1.0, np.bincount(rows, weights=weights, minlength=count)
    )
    space = la.ArraySpace((count,), dtype=jnp.float64, space_id="knn-diffusion")
    target, source = np.nonzero(dense)
    operator = SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(source), jnp.asarray(target), source_size=count, target_size=count
        ),
        jnp.asarray(dense[target, source]),
        source=space,
        target=space,
    )
    adjacency = EdgeRelation(
        jnp.asarray(columns), jnp.asarray(rows), source_size=count, target_size=count
    )
    return points, adjacency, la.assemble_sparse(operator), dense


def _patches(schwarz: PreparedMeshfreeSchwarz) -> list[np.ndarray]:
    offsets = np.asarray(schwarz.patch_offsets)
    members = np.asarray(schwarz.patch_points)
    return [members[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


def _reference_schwarz(
    dense: np.ndarray,
    schwarz: PreparedMeshfreeSchwarz,
    residual: np.ndarray,
    *,
    weighted: bool,
    multiplicative: bool,
) -> np.ndarray:
    """Textbook (restricted) Schwarz with exact local blocks R A Rᵀ."""
    offsets = np.asarray(schwarz.patch_offsets)
    weights = np.asarray(schwarz.partition_weights)
    correction = np.zeros_like(residual)
    for patch, indices in enumerate(_patches(schwarz)):
        defect = residual - dense @ correction if multiplicative else residual
        local = np.linalg.solve(dense[np.ix_(indices, indices)], defect[indices])
        if weighted:
            local = local * weights[offsets[patch] : offsets[patch + 1]]
        correction[indices] += local
    return correction


@pytest.mark.parametrize("partition", ["ownership", "layer_decay"])
def test_overlapping_patches_cover_by_graph_layers_with_exact_partition_of_unity(
    partition: MeshfreePartitionOfUnity,
) -> None:
    points, adjacency, _, _ = _knn_problem(96)
    layers = 2
    schwarz = MeshfreeSchwarzPlan(
        points,
        adjacency,
        policy=MeshfreeSchwarzPolicy(
            core_points=20,
            overlap_layers=layers,
            partition=partition,
        ),
    ).prepare(la.ArraySpace((96,), dtype=jnp.float64, space_id="knn-diffusion"))
    graph = np.zeros((96, 96), dtype=bool)
    graph[np.asarray(adjacency.target_indices), np.asarray(adjacency.source_indices)] = (
        True
    )
    graph |= graph.T
    offsets = np.asarray(schwarz.patch_offsets)
    distance = np.asarray(schwarz.patch_layers)
    weights = np.asarray(schwarz.partition_weights)
    cores = []
    total = np.zeros(96)
    for patch, indices in enumerate(_patches(schwarz)):
        layer = distance[offsets[patch] : offsets[patch + 1]]
        core = set(indices[layer == 0].tolist())
        cores.append(core)
        # Independent breadth-first closure of the core in the symmetrized graph.
        closure = set(core)
        for _ in range(layers):
            closure |= set(np.flatnonzero(graph[sorted(closure)].any(axis=0)).tolist())
        assert set(indices.tolist()) == closure
        np.add.at(total, indices, weights[offsets[patch] : offsets[patch + 1]])
    assert sorted(point for core in cores for point in core) == list(range(96))
    np.testing.assert_allclose(total, 1.0, atol=1e-14)
    evidence = schwarz.evidence
    assert evidence.partition_residual < 1e-14
    assert evidence.patch_sizes == tuple(np.diff(offsets).tolist())
    assert evidence.overlap_entries == sum(evidence.patch_sizes) - 96
    assert evidence.maximum_multiplicity > 1


def test_patches_depend_on_stable_ids_not_storage_order() -> None:
    points, adjacency, _, _ = _knn_problem(80, seed=3)
    policy = MeshfreeSchwarzPolicy(core_points=16, overlap_layers=1)
    space = la.ArraySpace((80,), dtype=jnp.float64)
    reference = MeshfreeSchwarzPlan(points, adjacency, policy=policy).prepare(space)
    permutation = np.random.default_rng(7).permutation(80)
    inverse = np.argsort(permutation)
    permuted_adjacency = EdgeRelation(
        jnp.asarray(inverse[np.asarray(adjacency.source_indices)]),
        jnp.asarray(inverse[np.asarray(adjacency.target_indices)]),
        source_size=80,
        target_size=80,
    )
    permuted = MeshfreeSchwarzPlan(
        points[permutation],
        permuted_adjacency,
        stable_ids=jnp.asarray(permutation),
        policy=policy,
    ).prepare(space)
    expected = [indices.tolist() for indices in _patches(reference)]
    observed = [permutation[indices].tolist() for indices in _patches(permuted)]
    assert observed == expected
    np.testing.assert_array_equal(permuted.patch_layers, reference.patch_layers)


@pytest.mark.parametrize(
    ("prolongation", "multiplicative"),
    [
        ("partition_of_unity", False),
        ("adjoint", False),
        ("partition_of_unity", True),
    ],
    ids=["restricted-additive", "classical-additive", "restricted-multiplicative"],
)
def test_native_correction_matches_textbook_schwarz_with_unweighted_local_blocks(
    prolongation: MeshfreeSchwarzProlongation, multiplicative: bool
) -> None:
    points, adjacency, operator, dense = _knn_problem(64, seed=1)
    assert isinstance(operator.source, la.ArraySpace)
    schwarz = MeshfreeSchwarzPlan(
        points,
        adjacency,
        policy=MeshfreeSchwarzPolicy(core_points=16, overlap_layers=2),
    ).prepare(operator.source)
    terms = schwarz.terms(prolongation=prolongation)
    builder = (
        la.MultiplicativeSubspaceCorrectionBuilder(terms)
        if multiplicative
        else la.AdditiveSubspaceCorrectionBuilder(terms)
    )
    preconditioner = builder.prepare(operator, materialization=_MATERIALIZATION)
    residual = np.random.default_rng(2).standard_normal(64)
    expected = _reference_schwarz(
        dense,
        schwarz,
        residual,
        weighted=prolongation == "partition_of_unity",
        multiplicative=multiplicative,
    )
    np.testing.assert_allclose(
        preconditioner.apply(jnp.asarray(residual)), expected, rtol=1e-10, atol=1e-12
    )


def test_classical_schwarz_with_cholesky_local_solves_is_a_symmetric_correction() -> None:
    points, adjacency, _, dense = _knn_problem(48, seed=4)
    coupling = dense + dense.T
    np.fill_diagonal(coupling, 0.0)
    # Strict diagonal dominance with positive diagonal: symmetric positive definite.
    symmetric = coupling + np.diag(np.abs(coupling).sum(axis=1) + 1.0)
    space = la.ArraySpace((48,), dtype=jnp.float64)
    target, source = np.nonzero(symmetric)
    operator = SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(source), jnp.asarray(target), source_size=48, target_size=48
        ),
        jnp.asarray(symmetric[target, source]),
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    schwarz = MeshfreeSchwarzPlan(
        points, adjacency, policy=MeshfreeSchwarzPolicy(core_points=12)
    ).prepare(space)
    cholesky = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("cholesky")
    )
    classical = la.AdditiveSubspaceCorrectionBuilder(
        schwarz.terms(cholesky, prolongation="adjoint")
    )
    restricted = la.AdditiveSubspaceCorrectionBuilder(schwarz.terms(cholesky))
    assert classical.properties_for(operator).certifies("self_adjoint")
    assert not restricted.properties_for(operator).certifies("self_adjoint")
    action = classical.prepare(operator, materialization=_MATERIALIZATION)
    rng = np.random.default_rng(6)
    left, right = (jnp.asarray(rng.standard_normal(48)) for _ in range(2))
    np.testing.assert_allclose(
        jnp.vdot(left, action.apply(right)),
        jnp.vdot(right, action.apply(left)),
        rtol=1e-12,
    )


def test_patch_capacities_and_local_factor_resources_are_refused() -> None:
    points, adjacency, operator, _ = _knn_problem(64, seed=5)
    space = operator.source
    assert isinstance(space, la.ArraySpace)
    with pytest.raises(la.LinearCapabilityError, match="maximum_patch_points"):
        MeshfreeSchwarzPlan(
            points,
            adjacency,
            policy=MeshfreeSchwarzPolicy(
                core_points=16, overlap_layers=2, maximum_patch_points=16
            ),
        ).prepare(space)
    with pytest.raises(la.LinearCapabilityError, match="maximum_patches"):
        MeshfreeSchwarzPlan(
            points,
            adjacency,
            policy=MeshfreeSchwarzPolicy(core_points=8, maximum_patches=7),
        ).prepare(space)
    with pytest.raises(la.LinearCapabilityError, match="maximum_transfer_entries"):
        MeshfreeSchwarzPlan(
            points,
            adjacency,
            policy=MeshfreeSchwarzPolicy(
                core_points=16, overlap_layers=3, maximum_transfer_entries=80
            ),
        ).prepare(space)
    schwarz = MeshfreeSchwarzPlan(
        points, adjacency, policy=MeshfreeSchwarzPolicy(core_points=16)
    ).prepare(space)
    bounded = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy(max_factor_nnz=8)
    )
    with pytest.raises(ValueError, match="infeasible"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.GMRES(restart=20),
                preconditioning=la.PreconditioningPolicy(
                    la.AdditiveSubspaceCorrectionBuilder(schwarz.terms(bounded))
                ),
            ),
        )


def _poisson_cloud(
    count: int,
) -> tuple[Array, np.ndarray, PreparedPointCloudDiscretization]:
    points, boundary = _disk_points(count, 0)
    points_ = jnp.asarray(points, dtype=jnp.float64)
    boundary_ = jnp.asarray(boundary)
    cloud = PointCloudPlan(
        points_,
        jnp.full((count,), np.pi / count),
        boundary_mask=boundary_,
        boundary_normals=jnp.where(boundary_[:, None], points_, 0.0),
        neighbors=16,
        stencil=LocalStencilPolicy(polynomial_degree=2),
    ).prepare()
    return points_, np.flatnonzero(boundary), cloud


@pytest.mark.parametrize("composition", ["additive", "multiplicative", "two-level"])
def test_point_cloud_poisson_with_schwarz_reaches_the_original_tolerance(
    composition: str,
) -> None:
    # Independent manufactured solution: u = 1 - |x|², k = 2 + 0.1 Σx,
    # -div(k grad u) = 4k + 0.2 Σx on the unit disk.
    points, ring, cloud = _poisson_cloud(256)
    exact = 1.0 - jnp.sum(points * points, axis=1)
    diffusivity = 2.0 + 0.1 * jnp.sum(points, axis=1)
    source = 4.0 * diffusivity + 0.2 * jnp.sum(points, axis=1)
    tolerance = la.TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=2000)

    def solve(
        preconditioning: la.PreconditioningPolicy | None, method: la.AbstractLinearMethod
    ) -> tuple[int, PointCloudPoissonPlan]:
        plan = PointCloudPoissonPlan(
            cloud,
            PointBoundaryPlan(
                (PointBoundaryCondition("dirichlet", ring, exact[ring], label="ring"),),
                row_count=256,
            ),
            linear_policy=la.LinearSolvePolicy(
                method, tolerance=tolerance, preconditioning=preconditioning
            ),
            # Degree-2 GMLS on this random disk has spurious Re <= 0 eigenvalues;
            # the quadratic exact field is still reproduced, and this test is
            # about preconditioning the algebraic system, so the assessment is
            # recorded rather than enforced.
            stability="diagnostic",
        )
        prepared = plan.prepare(diffusivity)
        assert prepared.stability is not None
        assert prepared.stability.outcome == "nonpositive-real-part"
        result = prepared.solve(source)
        assert bool(result.successful)
        assert float(result.residual_norm) <= float(result.residual_tolerance)
        np.testing.assert_allclose(result.values, exact, atol=1e-6)
        return int(result.diagnostics.iterations), plan

    # Unrestarted Krylov at N=256 so the unpreconditioned reference converges.
    baseline, plan = solve(None, la.GMRES(restart=256))
    schwarz = MeshfreeSchwarzPlan(
        points,
        cloud.relation,
        policy=MeshfreeSchwarzPolicy(core_points=64, overlap_layers=1),
    ).prepare(plan.solve_space)
    terms = schwarz.terms()
    match composition:
        case "additive":
            builder = la.AdditiveSubspaceCorrectionBuilder(terms)
            method: la.AbstractLinearMethod = la.GMRES(restart=256)
        case "multiplicative":
            builder = la.MultiplicativeSubspaceCorrectionBuilder(terms)
            method = la.GMRES(restart=256)
        case _:
            # Hybrid two-level: an exact Galerkin coarse correction on the first
            # hierarchy level, then the restricted patch sweep on its defect.
            hierarchy = plan.hierarchy_plan().prepare(plan.solve_space)
            builder = la.MultiplicativeSubspaceCorrectionBuilder(
                (meshfree_coarse_correction_term(hierarchy, level=1),) + terms
            )
            method = la.GMRES(restart=256)
    iterations, _ = solve(la.PreconditioningPolicy(builder), method)
    assert iterations < baseline
