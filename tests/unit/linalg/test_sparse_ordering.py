#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owned symmetric fill-reducing orderings and their factorization reuse.

Fill is measured independently of the package: ``nnz(L)`` of the Cholesky
factor of ``P A Pᵀ`` comes from the elimination tree and row subtrees (Liu,
SIAM J. Matrix Anal. Appl. 11, 1990) evaluated here on the SciPy pattern, and
is cross-checked against the factorization plan's own symbolic elimination.
Patterns are symmetrized k-nearest-neighbor graph Laplacians of seeded uniform
point clouds, shifted to be positive definite.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import phydrax as phx


jax.config.update("jax_enable_x64", True)

la = phx.linalg

_METHODS = (
    "natural",
    "reverse-cuthill-mckee",
    "approximate-minimum-degree",
    "nested-dissection",
)


def _knn_laplacian(
    count: int, dimension: int, /, *, seed: int, neighbors: int = 10
) -> tuple[np.ndarray, sp.csr_matrix]:
    points = np.random.default_rng(seed).random((count, dimension))
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, 1 : neighbors + 1]
    rows = np.repeat(np.arange(count), neighbors)
    graph = sp.csr_matrix(
        (np.ones(rows.size), (rows, nearest.ravel())), shape=(count, count)
    )
    # Unit-weight symmetrization: both triangles hold ones.
    graph = sp.csr_matrix(graph.maximum(graph.T))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    laplacian = sp.csr_matrix(sp.diags(degree + 0.5) - graph)
    laplacian.sort_indices()
    return points, laplacian


def _operator(matrix: sp.csr_matrix, /, *, spd: bool = True) -> Any:
    coordinates = matrix.tocoo()
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(coordinates.col, dtype=jnp.int32),
        jnp.asarray(coordinates.row, dtype=jnp.int32),
        source_size=matrix.shape[1],
        target_size=matrix.shape[0],
    )
    properties = (
        la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        if spd
        else None
    )
    return phx.sparse.SparseLinearMap(
        relation, jnp.asarray(coordinates.data), properties=properties
    )


def _cholesky_nnz(matrix: sp.csr_matrix, permutation: np.ndarray, /) -> int:
    """``nnz(L)`` with diagonal of ``chol(P A Pᵀ)`` from etree row subtrees."""
    permuted = sp.csr_matrix(matrix[permutation][:, permutation])
    permuted.sort_indices()
    size = permuted.shape[0]
    parent = [-1] * size
    ancestor = [-1] * size
    for row in range(size):
        for column in permuted.indices[permuted.indptr[row] : permuted.indptr[row + 1]]:
            node = int(column)
            while node != -1 and node < row:
                following = ancestor[node]
                ancestor[node] = row
                if following == -1:
                    parent[node] = row
                node = following
    count = 0
    mark = [-1] * size
    for row in range(size):
        mark[row] = row
        count += 1
        for column in permuted.indices[permuted.indptr[row] : permuted.indptr[row + 1]]:
            node = int(column)
            while node < row and mark[node] != row:
                mark[node] = row
                count += 1
                node = parent[node]
    return count


def _ordering(
    operator: Any,
    method: la.SparseOrdering,
    coordinates: np.ndarray | None = None,
    /,
    **capacities: int,
) -> Any:
    return la.prepare_sparse_ordering(
        operator,
        la.SparseOrderingPolicy(method, **capacities),
        coordinates=coordinates,
    )


@pytest.mark.parametrize(
    ("method", "geometric"),
    [
        ("natural", False),
        ("reverse-cuthill-mckee", False),
        ("approximate-minimum-degree", False),
        ("nested-dissection", False),
        ("nested-dissection", True),
    ],
    ids=["natural", "rcm", "amd", "nd-graph", "nd-geometric"],
)
def test_orderings_are_deterministic_validated_permutations(
    method: la.SparseOrdering, geometric: bool
) -> None:
    points, matrix = _knn_laplacian(400, 3, seed=5)
    # A second component exercises disconnected dissection.
    second_points, second_matrix = _knn_laplacian(90, 3, seed=6)
    block = sp.csr_matrix(sp.block_diag((matrix, second_matrix), format="csr"))
    coordinates = np.concatenate((points, second_points + 2.0))
    operator = _operator(block)

    first = _ordering(operator, method, coordinates if geometric else None)
    second = _ordering(operator, method, coordinates if geometric else None)

    size = block.shape[0]
    assert np.array_equal(np.sort(first.permutation), np.arange(size))
    assert np.array_equal(first.inverse_permutation[first.permutation], np.arange(size))
    assert np.array_equal(first.permutation, second.permutation)
    assert first.ordering_id == second.ordering_id
    assert first.method == method
    if method == "natural":
        assert np.array_equal(first.permutation, np.arange(size))
        assert first.work == 0
    else:
        assert first.work > 0
    if method == "nested-dissection":
        evidence = first.dissection
        assert evidence is not None
        assert evidence.route == ("geometric" if geometric else "graph")
        assert geometric == (first.coordinates_id is not None)
        # Separators and leaves partition the vertices exactly.
        assert evidence.separator_sizes.sum() + evidence.leaf_sizes.sum() == size
        assert evidence.leaf_sizes.max() <= evidence.leaf_capacity
        assert evidence.separator_sizes.max() <= evidence.separator_capacity
    else:
        assert first.dissection is None


def test_geometric_and_graph_dissections_have_distinct_identities() -> None:
    points, matrix = _knn_laplacian(300, 2, seed=8)
    operator = _operator(matrix)

    graph = _ordering(operator, "nested-dissection")
    geometric = _ordering(operator, "nested-dissection", points)
    shifted = _ordering(operator, "nested-dissection", points + 1.0)

    assert graph.ordering_id != geometric.ordering_id
    assert geometric.coordinates_id != shifted.coordinates_id
    # A uniform translation does not change the median cuts.
    assert np.array_equal(geometric.permutation, shifted.permutation)


@pytest.mark.parametrize("dimension", [2, 3], ids=["2d", "3d"])
def test_minimum_degree_and_dissection_reduce_knn_laplacian_fill(
    dimension: int,
) -> None:
    points, matrix = _knn_laplacian(512, dimension, seed=512 + dimension)
    operator = _operator(matrix)
    fill = {
        "natural": _cholesky_nnz(matrix, np.arange(512)),
        "rcm": _cholesky_nnz(
            matrix, _ordering(operator, "reverse-cuthill-mckee").permutation
        ),
        "amd": _cholesky_nnz(
            matrix, _ordering(operator, "approximate-minimum-degree").permutation
        ),
        "nd-graph": _cholesky_nnz(
            matrix, _ordering(operator, "nested-dissection").permutation
        ),
        "nd-geometric": _cholesky_nnz(
            matrix, _ordering(operator, "nested-dissection", points).permutation
        ),
    }

    assert fill["amd"] < 0.5 * fill["natural"]
    assert fill["amd"] < fill["rcm"]
    assert fill["nd-graph"] < 0.6 * fill["natural"]
    assert fill["nd-geometric"] < 0.6 * fill["natural"]
    assert fill["nd-graph"] < fill["rcm"]
    assert fill["nd-geometric"] < fill["rcm"]


@pytest.mark.parametrize("method", _METHODS[1:])
def test_plan_fill_matches_the_independent_elimination_tree_count(
    method: la.SparseOrdering,
) -> None:
    _, matrix = _knn_laplacian(256, 3, seed=11)
    operator = _operator(matrix)
    ordering = _ordering(operator, method)

    plan = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy("cholesky", ordering=method),
        ordering=ordering,
    )

    assert np.array_equal(np.asarray(plan.permutation), ordering.permutation)
    assert plan.factor_nnz == _cholesky_nnz(matrix, ordering.permutation)
    assert plan.ordering_id == ordering.ordering_id
    assert plan.ordering_work == ordering.work


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("kind", ["cholesky", "lu"])
def test_reordered_factor_solves_the_original_system(
    method: la.SparseOrdering, kind: la.SparseFactorizationKind
) -> None:
    _, matrix = _knn_laplacian(160, 3, seed=13)
    if kind == "lu":
        # Same symmetric pattern, nonsymmetric values; still diagonally
        # dominant by rows and columns, so static pivots stay admissible.
        upper = sp.triu(matrix, k=1)
        matrix = sp.csr_matrix(matrix - 0.3 * upper)
        matrix.sort_indices()
    operator = _operator(matrix, spd=kind == "cholesky")
    right_hand_side = np.random.default_rng(17).standard_normal(160)

    factor = la.factorize_sparse(
        operator,
        la.SparseFactorizationPolicy(kind, ordering=method),
    )
    result = factor.solve(jnp.asarray(right_hand_side))

    assert bool(result.success)
    value = np.asarray(result.value)
    reference = spsolve(sp.csc_matrix(matrix), right_hand_side)
    residual = np.linalg.norm(matrix @ value - right_hand_side)
    assert residual <= 1e-11 * np.linalg.norm(right_hand_side)
    np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-12)


def test_ordering_work_is_refused_before_any_factor_entry() -> None:
    _, matrix = _knn_laplacian(200, 3, seed=19)
    operator = _operator(matrix)
    ordering = _ordering(operator, "approximate-minimum-degree")
    limit = ordering.work - 1

    with pytest.raises(
        la.LinearCapabilityError,
        match=rf"ordering_work requires {ordering.work}, exceeding limit {limit}",
    ):
        la.prepare_sparse_ordering(
            operator,
            la.SparseOrderingPolicy(
                "approximate-minimum-degree", max_ordering_work=limit
            ),
        )
    policy = la.SparseFactorizationPolicy(
        "cholesky", ordering="approximate-minimum-degree", max_symbolic_work=limit
    )
    # Internally prepared and reused orderings refuse identically, with no
    # diagonal or fill entry reserved yet.
    expected = (
        rf"symbolic_work requires {ordering.work}, exceeding limit {limit}; factor_nnz=0/"
    )
    with pytest.raises(la.LinearCapabilityError, match=expected):
        la.prepare_sparse_factorization(operator, policy)
    with pytest.raises(la.LinearCapabilityError, match=expected):
        la.prepare_sparse_factorization(operator, policy, ordering=ordering)
    admitted = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy(
            "cholesky",
            ordering="approximate-minimum-degree",
            max_symbolic_work=ordering.work + 10 * matrix.nnz * matrix.shape[0],
        ),
    )
    assert admitted.ordering_work == ordering.work
    assert admitted.symbolic_work > ordering.work


def test_separator_capacity_refuses_nested_dissection() -> None:
    _, matrix = _knn_laplacian(300, 2, seed=23)
    operator = _operator(matrix)
    admitted = _ordering(operator, "nested-dissection")
    largest = int(admitted.dissection.separator_sizes.max())

    with pytest.raises(
        la.LinearCapabilityError,
        match=rf"separator of {largest} vertices exceeds separator_capacity {largest - 1}",
    ):
        _ordering(operator, "nested-dissection", separator_capacity=largest - 1)
    tight = _ordering(operator, "nested-dissection", separator_capacity=largest)
    assert np.array_equal(tight.permutation, admitted.permutation)


def test_prepared_ordering_is_reused_by_plans_and_numeric_refresh() -> None:
    _, matrix = _knn_laplacian(240, 3, seed=29)
    operator = _operator(matrix)
    method = "nested-dissection"
    ordering = _ordering(operator, method)
    policy = la.SparseFactorizationPolicy("cholesky", ordering=method)

    internal = la.prepare_sparse_factorization(operator, policy)
    reused = la.prepare_sparse_factorization(operator, policy, ordering=ordering)
    assert reused.plan_id == internal.plan_id
    assert reused.symbolic_work == internal.symbolic_work

    # New values on the frozen pattern: refresh keeps the plan and ordering.
    refreshed_matrix = sp.csr_matrix(2.0 * matrix + sp.eye(240))
    refreshed_matrix.sort_indices()
    refreshed = la.refresh_sparse_factorization(reused, _operator(refreshed_matrix))
    assert refreshed.plan.plan_id == internal.plan_id
    right_hand_side = np.random.default_rng(31).standard_normal(240)
    value = np.asarray(refreshed.solve(jnp.asarray(right_hand_side)).value)
    assert np.linalg.norm(
        refreshed_matrix @ value - right_hand_side
    ) <= 1e-11 * np.linalg.norm(right_hand_side)

    other = _operator(_knn_laplacian(240, 3, seed=30)[1])
    with pytest.raises(ValueError, match="different sparse pattern"):
        la.prepare_sparse_factorization(other, policy, ordering=ordering)
    with pytest.raises(ValueError, match="differs from the policy ordering"):
        la.prepare_sparse_factorization(
            operator,
            la.SparseFactorizationPolicy(
                "cholesky", ordering="approximate-minimum-degree"
            ),
            ordering=ordering,
        )


def test_ordering_policy_and_coordinates_are_validated() -> None:
    _, matrix = _knn_laplacian(40, 2, seed=37)
    operator = _operator(matrix)

    with pytest.raises(ValueError, match="method"):
        la.SparseOrderingPolicy("metis")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="apply only to nested-dissection"):
        la.SparseOrderingPolicy("approximate-minimum-degree", leaf_capacity=8)
    with pytest.raises(ValueError, match="positive"):
        la.SparseOrderingPolicy("nested-dissection", leaf_capacity=0)
    with pytest.raises(ValueError, match="coordinates apply only"):
        _ordering(operator, "approximate-minimum-degree", np.zeros((40, 2)))
    with pytest.raises(ValueError):
        _ordering(operator, "nested-dissection", np.zeros((39, 2)))
    with pytest.raises(ValueError, match="finite"):
        _ordering(operator, "nested-dissection", np.full((40, 2), np.nan))
    with pytest.raises(ValueError, match="bijection"):
        la.PreparedSparseOrdering(
            np.asarray([0, 0, 2]),
            policy=la.SparseOrderingPolicy("natural"),
            pattern_id="pattern",
            work=0,
        )
