#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.linalg._forest_bounds import (
    _certify_with_coefficients,
    _local_coefficients,
    BlockForestRankCertificate,
    BlockForestRankStatus,
    BlockForestTopology,
    certify_block_forest_rank,
    prepare_block_forest,
    PreparedBlockForest,
    refresh_block_forest,
)


pytestmark = pytest.mark.strict_jax

type FloatArray = NDArray[np.float64] | NDArray[np.float32]


def _case(
    parents: list[int],
    rows: list[int],
    width: int,
    seed: int,
    *,
    coupling: float = 0.5,
    dtype: type[np.floating] = np.float64,
) -> tuple[BlockForestTopology, FloatArray, FloatArray]:
    """Diagonally dominant child blocks and random parent couplings."""
    rng = np.random.default_rng(seed)
    count = len(parents)
    child = np.zeros((count, width, width))
    parent = np.zeros((count, width, width))
    for vertex, (above, rows_) in enumerate(zip(parents, rows, strict=True)):
        child[vertex, :rows_] = np.eye(width)[:rows_] + 0.2 * rng.normal(
            size=(rows_, width)
        )
        if above >= 0:
            parent[vertex, :rows_] = coupling * rng.normal(size=(rows_, width))
    topology = BlockForestTopology(
        np.asarray(parents), np.asarray(rows), column_width=width
    )
    return topology, child.astype(dtype), parent.astype(dtype)


def _dense(
    topology: BlockForestTopology, child: FloatArray, parent: FloatArray
) -> NDArray[np.float64]:
    """Independent assembly of the represented operator, one row block per edge."""
    parents = np.asarray(topology.parent_vertices)
    counts = np.asarray(topology.row_mask).sum(axis=1)
    count, _, width = child.shape
    blocks = []
    for vertex in range(count):
        rows = int(counts[vertex])
        if rows == 0:
            continue
        block = np.zeros((rows, count * width))
        block[:, vertex * width : (vertex + 1) * width] = child[vertex, :rows]
        above = int(parents[vertex])
        if above >= 0:
            block[:, above * width : (above + 1) * width] = parent[vertex, :rows]
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def _singular_extremes(matrix: NDArray[np.float64]) -> tuple[mp.mpf, mp.mpf]:
    """Smallest row and largest singular values in 50-digit arithmetic."""
    with mp.workdps(50):
        values = mp.svd_r(mp.matrix(matrix.tolist()), compute_uv=False)
        spectrum = [values[index] for index in range(values.rows)]
        return min(spectrum), max(spectrum)


def _spectral_norm(matrix: NDArray[np.float64]) -> mp.mpf:
    with mp.workdps(50):
        values = mp.svd_r(mp.matrix(matrix.tolist()), compute_uv=False)
        return max(values[index] for index in range(values.rows))


def _certify(
    topology: BlockForestTopology,
    child: FloatArray,
    parent: FloatArray,
    *,
    relative: float,
    absolute: float = 0.0,
) -> tuple[PreparedBlockForest, BlockForestRankCertificate]:
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))

    @jax.jit
    def certify(value: PreparedBlockForest) -> BlockForestRankCertificate:
        return certify_block_forest_rank(
            value, relative_cutoff=relative, absolute_cutoff=absolute
        )

    return prepared, certify(prepared)


_TOPOLOGIES = {
    "free-chain": ([-1, 0, 1, 2, 3, 4], [0, 3, 3, 3, 3, 3]),
    "anchored-star": ([-1, -1, -1, -1, -1], [3, 2, 3, 1, 3]),
    "mixed-star": ([-1, 0, 0, 0, 0, 0], [0, 3, 2, 3, 1, 3]),
    "disconnected": ([-1, 0, 1, -1, 3, -1, -1, 6], [3, 3, 2, 0, 3, 0, 3, 3]),
}


@pytest.mark.parametrize("name", sorted(_TOPOLOGIES))
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_certificate_encloses_high_precision_spectrum(
    name: str, dtype: type[np.floating]
) -> None:
    parents, rows = _TOPOLOGIES[name]
    topology, child, parent = _case(parents, rows, 3, seed=17, dtype=dtype)
    _, certificate = _certify(topology, child, parent, relative=1.0e-4)
    smallest, largest = _singular_extremes(_dense(topology, child, parent))
    lower = float(certificate.minimum_singular_value_lower_bound)
    upper = float(certificate.maximum_singular_value_upper_bound)
    assert bool(certificate.certified)
    assert int(certificate.status) == BlockForestRankStatus.CERTIFIED
    assert int(certificate.rank_lower_bound) == sum(rows) == certificate.row_count
    assert 0.0 < lower <= smallest
    assert largest <= upper
    assert largest / smallest <= float(certificate.condition_upper_bound)
    assert float(certificate.residual_norm_upper_bound) < 1.0
    assert lower > float(certificate.threshold_upper_bound) >= 1.0e-4 * float(largest)


def test_represented_right_inverse_and_residual_are_dominated() -> None:
    parents, rows = _TOPOLOGIES["disconnected"]
    topology, child, parent = _case(parents, rows, 3, seed=5, coupling=1.5)
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))
    right, propagation = _local_coefficients(
        prepared.child_blocks, prepared.parent_blocks, topology.row_mask
    )
    # Perturb the stored coefficients: the bounds must follow the actual R.
    rng = np.random.default_rng(3)
    right = right * (1.0 + 1.0e-3 * jnp.asarray(rng.normal(size=right.shape)))
    right = jnp.where(topology.row_mask[:, None, :], right, 0.0)
    certificate = _certify_with_coefficients(
        prepared, right, propagation, relative_cutoff=0.0, absolute_cutoff=0.0
    )
    q, c = np.asarray(right), np.asarray(propagation)
    counts = np.asarray(topology.row_mask).sum(axis=1)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    order = np.argsort(np.asarray(_depths(parents)), kind="stable")
    matrix = _dense(topology, child, parent)
    total = matrix.shape[0]
    candidate = np.zeros((matrix.shape[1], total))
    for column in range(total):
        y = np.zeros(total)
        y[column] = 1.0
        x = np.zeros((len(parents), 3))
        for vertex in order:
            if counts[vertex] == 0:
                continue
            local = y[offsets[vertex] : offsets[vertex + 1]]
            x[vertex] = q[vertex, :, : counts[vertex]] @ local
            if parents[vertex] >= 0:
                x[vertex] -= c[vertex] @ x[parents[vertex]]
        candidate[:, column] = x.reshape(-1)
    residual = matrix @ candidate - np.eye(total)
    assert _spectral_norm(candidate) <= float(certificate.right_inverse_norm_upper_bound)
    assert _spectral_norm(residual) <= float(certificate.residual_norm_upper_bound)
    smallest, _ = _singular_extremes(matrix)
    assert float(certificate.minimum_singular_value_lower_bound) <= smallest
    exact = certify_block_forest_rank(prepared, relative_cutoff=0.0)
    assert float(certificate.residual_norm_upper_bound) > float(
        exact.residual_norm_upper_bound
    )


def _depths(parents: list[int]) -> list[int]:
    depths = []
    for vertex in range(len(parents)):
        depth, current = 0, parents[vertex]
        while current >= 0:
            depth, current = depth + 1, parents[current]
        depths.append(depth)
    return depths


def test_noncontractive_stored_inverse_is_inconclusive() -> None:
    topology, child, parent = _case([-1, 0, 1], [0, 3, 3], 3, seed=2)
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))
    right, propagation = _local_coefficients(
        prepared.child_blocks, prepared.parent_blocks, topology.row_mask
    )
    certificate = _certify_with_coefficients(
        prepared,
        0.0 * right,
        propagation,
        relative_cutoff=0.0,
        absolute_cutoff=0.0,
    )
    assert not bool(certificate.certified)
    assert int(certificate.status) == BlockForestRankStatus.RESIDUAL_NOT_CONTRACTIVE
    assert float(certificate.residual_norm_upper_bound) >= 1.0
    assert float(certificate.minimum_singular_value_lower_bound) == 0.0
    assert int(certificate.rank_lower_bound) == 0


def test_unit_local_pivots_do_not_establish_global_rank() -> None:
    count, beta = 8, 4.0
    parents = [-1, *range(count - 1)]
    rows = [3] * count
    child = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    parent = np.zeros((count, 3, 3))
    parent[1:] = -beta * np.eye(3)
    topology = BlockForestTopology(np.asarray(parents), np.asarray(rows), column_width=3)
    smallest, largest = _singular_extremes(_dense(topology, child, parent))
    assert smallest / largest < 1.0e-4
    _, coarse = _certify(topology, child, parent, relative=1.0e-3)
    assert not bool(coarse.certified)
    assert int(coarse.status) == BlockForestRankStatus.THRESHOLD_NOT_SEPARATED
    assert float(coarse.minimum_singular_value_lower_bound) <= smallest
    _, fine = _certify(topology, child, parent, relative=1.0e-12)
    assert bool(fine.certified)
    assert float(fine.minimum_singular_value_lower_bound) <= smallest


def test_unequal_component_scales() -> None:
    parents, rows = [-1, 0, 1, -1, 3, 4], [0, 3, 3, 0, 3, 3]
    topology, child, parent = _case(parents, rows, 3, seed=11)
    child[3:] *= 1.0e-7
    parent[3:] *= 1.0e-7
    smallest, largest = _singular_extremes(_dense(topology, child, parent))
    assert smallest < 1.0e-6 * largest
    _, separated = _certify(topology, child, parent, relative=1.0e-10)
    assert bool(separated.certified)
    assert float(separated.minimum_singular_value_lower_bound) <= smallest
    _, blocked = _certify(topology, child, parent, relative=1.0e-6)
    assert not bool(blocked.certified)
    assert int(blocked.rank_lower_bound) == 0


def test_threshold_equality_is_not_certified() -> None:
    topology, child, parent = _case([-1, 0, 0], [0, 3, 2], 3, seed=8)
    _, reference = _certify(topology, child, parent, relative=0.0)
    lower = float(reference.minimum_singular_value_lower_bound)
    _, equal = _certify(topology, child, parent, relative=0.0, absolute=lower)
    assert float(equal.minimum_singular_value_lower_bound) == lower
    assert not bool(equal.certified)
    assert int(equal.status) == BlockForestRankStatus.THRESHOLD_NOT_SEPARATED
    assert float(equal.separation_lower_bound) <= 0.0


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        ("nonfinite", BlockForestRankStatus.INVALID_BLOCKS),
        ("outside-structure", BlockForestRankStatus.INVALID_BLOCKS),
        ("singular-pivot", BlockForestRankStatus.LOCAL_BOUND_UNRESOLVED),
        ("overflow", BlockForestRankStatus.GLOBAL_BOUND_OVERFLOW),
    ],
)
def test_failures_are_inconclusive(mutation: str, status: BlockForestRankStatus) -> None:
    topology, child, parent = _case([-1, 0, 1, 2], [3, 3, 2, 3], 3, seed=4)
    match mutation:
        case "nonfinite":
            child[1, 0, 0] = np.nan
        case "outside-structure":
            child[2, 2, 0] = 1.0
        case "singular-pivot":
            child[2, 1] = 0.0
        case "overflow":
            # Local blocks stay finite; the depth-3 path products overflow.
            child[1:] = np.eye(3)
            parent[1:] = -1.0e120 * np.eye(3)
            child[2, 2] = 0.0
            parent[2, 2] = 0.0
        case _:
            raise AssertionError(mutation)
    _, certificate = _certify(topology, child, parent, relative=0.0)
    assert int(certificate.status) == status
    assert not bool(certificate.certified)
    assert int(certificate.rank_lower_bound) == 0
    assert float(certificate.minimum_singular_value_lower_bound) == 0.0


def test_ambiguous_pivot_never_claims_deficiency() -> None:
    topology, child, parent = _case([-1, 0, 1], [0, 3, 3], 3, seed=9)
    # Both blocks of one row must be near-dependent: a near-singular child block
    # alone is compensated by its parent coupling and stays globally well posed.
    child[2, 2] = child[2, 1] + 1.0e-9 * np.asarray([1.0, -2.0, 0.5])
    parent[2, 2] = parent[2, 1]
    smallest, largest = _singular_extremes(_dense(topology, child, parent))
    assert smallest < 1.0e-6 * largest
    _, certificate = _certify(topology, child, parent, relative=1.0e-6)
    assert not bool(certificate.certified)
    assert float(certificate.minimum_singular_value_lower_bound) <= smallest
    assert int(certificate.status) in {
        BlockForestRankStatus.THRESHOLD_NOT_SEPARATED,
        BlockForestRankStatus.RESIDUAL_NOT_CONTRACTIVE,
        BlockForestRankStatus.LOCAL_BOUND_UNRESOLVED,
    }


@pytest.mark.parametrize(
    ("parents", "rows", "width", "message"),
    [
        ([1, 0], [3, 3], 3, "cycle"),
        ([0], [3], 3, "another vertex"),
        ([-1, 0], [0, 0], 3, "at least one row"),
        ([-1, 0], [0, 4], 3, "row_capacity"),
    ],
)
def test_unsupported_topologies_are_refused(
    parents: list[int], rows: list[int], width: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BlockForestTopology(np.asarray(parents), np.asarray(rows), column_width=width)


def test_wider_row_capacity_than_columns_is_refused() -> None:
    with pytest.raises(ValueError, match="no right inverse"):
        BlockForestTopology(
            np.asarray([-1, 0]), np.asarray([0, 2]), column_width=2, row_capacity=3
        )


def test_refresh_binds_new_numeric_state() -> None:
    topology, child, parent = _case([-1, 0, 1], [0, 3, 3], 3, seed=21)
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))
    assert int(prepared.numeric_version) == 0
    degenerate = child.copy()
    degenerate[2] = 0.0
    refreshed = refresh_block_forest(prepared, jnp.asarray(degenerate), jnp.asarray(parent))
    assert int(refreshed.numeric_version) == 1
    before = certify_block_forest_rank(prepared, relative_cutoff=1.0e-8)
    after = certify_block_forest_rank(refreshed, relative_cutoff=1.0e-8)
    assert bool(before.certified) and int(before.numeric_version) == 0
    assert not bool(after.certified) and int(after.numeric_version) == 1
    assert before.topology_id == after.topology_id == topology.topology_id
    with pytest.raises(ValueError, match="shape"):
        refresh_block_forest(prepared, jnp.asarray(child[:2]), jnp.asarray(parent[:2]))
    with pytest.raises(ValueError, match="dtype"):
        refresh_block_forest(
            prepared,
            jnp.asarray(child, dtype=jnp.float32),
            jnp.asarray(parent, dtype=jnp.float32),
        )


def test_workspace_budget_is_admitted_before_tracing() -> None:
    topology, child, parent = _case([-1, 0, 1], [0, 3, 3], 3, seed=1)
    with pytest.raises(ValueError, match="workspace"):
        prepare_block_forest(
            topology, jnp.asarray(child), jnp.asarray(parent), workspace_bytes=1
        )
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))
    assert prepared.retained_bytes >= 2 * child.size * child.itemsize
    assert prepared.workspace_bytes > 0


def test_zero_row_forest_keeps_rank_zero_condition_one() -> None:
    topology = BlockForestTopology(np.asarray([-1, -1]), np.asarray([0, 0]), column_width=3)
    zeros = jnp.zeros((2, 3, 3))
    certificate = certify_block_forest_rank(
        prepare_block_forest(topology, zeros, zeros), relative_cutoff=1.0e-10
    )
    assert bool(certificate.certified)
    assert int(certificate.rank_lower_bound) == 0
    assert float(certificate.condition_upper_bound) == 1.0
    assert certificate.row_count == 0


def test_bound_derivatives_away_from_decision_boundaries() -> None:
    parents, rows = _TOPOLOGIES["mixed-star"]
    topology, child, parent = _case(parents, rows, 3, seed=31)
    prepared = prepare_block_forest(topology, jnp.asarray(child), jnp.asarray(parent))
    direction = np.random.default_rng(6).normal(size=child.shape)
    direction = np.where(np.asarray(topology.row_mask)[:, :, None], direction, 0.0)

    def lower(blocks: jax.Array) -> jax.Array:
        bound = refresh_block_forest(prepared, blocks, prepared.parent_blocks)
        return certify_block_forest_rank(
            bound, relative_cutoff=1.0e-6
        ).minimum_singular_value_lower_bound

    _, tangent = jax.jvp(lower, (jnp.asarray(child),), (jnp.asarray(direction),))
    step = 1.0e-6
    central = (
        lower(jnp.asarray(child + step * direction))
        - lower(jnp.asarray(child - step * direction))
    ) / (2.0 * step)
    assert np.isfinite(float(tangent))
    assert float(tangent) == pytest.approx(float(central), rel=1.0e-5, abs=1.0e-9)
