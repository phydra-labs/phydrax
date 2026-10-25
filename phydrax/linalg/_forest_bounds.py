#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Verified right-inverse lower bounds for the smallest singular value of block forests.

A block forest operator ``J`` has one column block of width ``c`` per vertex and,
for every non-root vertex ``w``, one row block ``A_w x_w + B_w x_parent(w)``.
``B_w`` is absent when the incoming edge ends at an anchor without columns.

For every edge the certificate stores an approximate local right inverse ``Q_w``
of ``A_w`` and the propagation coefficient ``C_w = Q_w B_w`` actually held in
memory. The represented global right-inverse candidate ``R`` is
``x_root = 0`` and ``x_w = Q_w y_w - C_w x_parent(w)``. With verified upper bounds
``a_w >= ||Q_w||``, ``b_w >= ||C_w||``, ``u_w >= ||A_w Q_w - I||`` and
``v_w >= ||B_w - A_w C_w||`` the block-norm domination matrices of ``R`` and
``E = J R - I`` satisfy the forest recurrences

* ``h_w = a_w + b_w h_parent(w)`` (``h`` is ``0`` at roots and anchored parents),
* ``t_v = 1 + sum_children b_w t_w``,
* ``d_v = sum_children (v_w + b_w d_w)``,

so ``||R|| <= V = sqrt(max h * max_w a_w t_w)`` and
``||E|| <= eps = sqrt(max_w(u_w + v_w h_parent(w)) * max_w(u_w + a_w d_w))``.
If ``eps < 1``, ``R (I + E)^-1`` is an exact right inverse of norm at most
``V / (1 - eps)``, hence ``sigma_min(J) >= L = (1 - eps) / V`` for the row
singular values. ``sigma_max(J) <= U`` comes from the Schur row/column-sum test
of the represented blocks. Full numerical row rank under the strict rule
``sigma > absolute + relative * sigma_max`` is certified only when
``L > absolute + relative * U``; failure of this sufficient test is inconclusive.

Every scalar is carried as a rigorous bound: each floating operation group is
followed by an outward pad under the model documented on :func:`_pad`. The
recurrences use pointer doubling (``Sum_k B^k = Prod_j (I + B^(2^j))``), so the
device work is ``O(n log depth)`` and no dense matrix is formed.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import as_array, Bool, checked, Dim, Float, Int32, Scalar, Scope


class ForestVertexDim(Dim):
    """Vertices of a block forest; vertex ``w`` owns one column block."""


class ForestRowDim(Dim):
    """Padded rows of the incoming row block of a vertex."""


class ForestColumnDim(Dim):
    """Columns of one vertex column block."""


class ForestLevelDim(Dim):
    """Pointer-doubling levels; level ``j`` addresses the ``2^j``-th ancestor."""


class BlockForestRankStatus(IntEnum):
    """Certificate outcome; every non-certified outcome is inconclusive.

    Ordered by priority: the first failing stage is reported.
    """

    CERTIFIED = 0
    THRESHOLD_NOT_SEPARATED = 1
    RESIDUAL_NOT_CONTRACTIVE = 2
    GLOBAL_BOUND_OVERFLOW = 3
    LOCAL_BOUND_UNRESOLVED = 4
    INVALID_BLOCKS = 5


# The outward pad below is valid while every declared operation count k satisfies
# 16 k eps <= 1/4; preparation refuses topologies whose counts exceed this.
_ROUNDING_CAPACITY = 0.25


def _pad(value: Array, operations: int) -> Array:
    """Outward pad for ``operations`` floating operations on nonnegative data.

    Model: every elementary operation (``+ - * / sqrt``, including reduction
    reassociation, fused multiply-add contraction, approximate device division or
    square root within four units in the last place, and flush-to-zero underflow)
    returns ``x (1 + delta) + eta`` with ``|delta| <= 2 eps`` and
    ``|eta| <= tiny``. A nonnegative sum of ``k + 1`` terms, or one monotone
    operation for ``k = 1``, then satisfies
    ``exact <= computed (1 + 4 k eps (1 + 4 k eps)) + 2 k tiny``; the pad
    ``16 k eps |x| + 8 k tiny`` dominates this and the rounding of the pad itself.
    """
    info = jnp.finfo(value.dtype)
    return (16 * operations) * info.eps * value + (8 * operations) * info.tiny


def _up(value: Array, operations: int) -> Array:
    return value + _pad(value, operations)


def _down(value: Array, operations: int) -> Array:
    return value - _pad(jnp.abs(value), operations)


@final
class BlockForestTopology(StrictModule):
    """Host-validated rooted forest of row/column blocks.

    ``parent_vertices[w]`` is the parent vertex of the incoming edge of ``w`` or
    ``-1``. ``row_counts[w]`` is the row count of that incoming edge; zero marks a
    root. A vertex with ``parent -1`` and positive rows has an anchored edge whose
    other endpoint carries no columns. Every edge has at most ``column_width``
    rows, since a wider block has no right inverse.
    """

    __strict_contract__ = True
    parent_vertices: Int32[ForestVertexDim]
    row_mask: Bool[ForestVertexDim, ForestRowDim]
    ancestors: Int32[ForestLevelDim, ForestVertexDim]
    column_width: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    level_multiplicities: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent_vertices: ArrayLike,
        row_counts: ArrayLike,
        /,
        *,
        column_width: int,
        row_capacity: int | None = None,
    ) -> None:
        parents = np.asarray(parent_vertices)
        rows = np.asarray(row_counts)
        if parents.ndim != 1 or rows.shape != parents.shape or parents.size < 1:
            raise ValueError(
                "parent_vertices and row_counts must be matching nonempty vectors."
            )
        if not (
            np.issubdtype(parents.dtype, np.integer)
            and np.issubdtype(rows.dtype, np.integer)
        ):
            raise TypeError("parent_vertices and row_counts must be integers.")
        width = _positive(column_width, "column_width")
        capacity = width if row_capacity is None else _positive(row_capacity, "row_capacity")
        if capacity > width:
            raise ValueError(
                "row_capacity exceeds column_width; such a block has no right inverse."
            )
        count = parents.size
        parents = parents.astype(np.int64)
        rows = rows.astype(np.int64)
        if np.any((parents < -1) | (parents >= count)) or np.any(
            parents == np.arange(count)
        ):
            raise ValueError("parent_vertices must name another vertex or -1.")
        if np.any((rows < 0) | (rows > capacity)):
            raise ValueError("row_counts must lie in [0, row_capacity].")
        if np.any((parents >= 0) & (rows == 0)):
            raise ValueError("An edge to a parent vertex must carry at least one row.")
        table = _ancestor_table(parents)
        depth = _depth(parents, table)
        levels = depth.bit_length()
        ancestors = table[:levels]
        multiplicities = tuple(
            int(np.max(np.bincount(level[level >= 0], minlength=count), initial=0))
            for level in ancestors
        )
        self.parent_vertices = jnp.asarray(parents.astype(np.int32))
        self.row_mask = jnp.asarray(np.arange(capacity)[None, :] < rows[:, None])
        self.ancestors = jnp.asarray(ancestors.reshape(levels, count).astype(np.int32))
        self.column_width = width
        self.row_count = int(np.sum(rows))
        self.depth = depth
        self.level_multiplicities = multiplicities
        self.topology_id = canonical_fingerprint(
            {
                "kind": "block-forest-topology",
                "column_width": width,
                "row_capacity": capacity,
                "values": array_tree_fingerprint({"parents": parents, "rows": rows}),
            }
        )

    @property
    def vertex_count(self) -> int:
        return self.parent_vertices.shape[0]

    @property
    def row_capacity(self) -> int:
        return self.row_mask.shape[1]

    @property
    def child_multiplicity(self) -> int:
        """Largest number of children of one vertex."""
        return self.level_multiplicities[0] if self.level_multiplicities else 0

    def rounding_operations(self) -> int:
        """Largest operation count charged to one outward pad."""
        width, rows = self.column_width, self.row_capacity
        return max(
            width * rows,
            width * width,
            rows * rows,
            2 * width + 1,
            (self.child_multiplicity + 1) * rows + 1,
            max(self.level_multiplicities, default=0) + 1,
        )


def _positive(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _ancestor_table(parents: np.ndarray) -> np.ndarray:
    """``2^j``-th ancestors; an acyclic parent map reaches ``-1`` within ``n`` steps."""
    count = parents.size
    levels = max(count.bit_length(), 1)
    table = np.empty((levels + 1, count), dtype=np.int64)
    table[0] = parents
    for level in range(1, levels + 1):
        previous = table[level - 1]
        table[level] = np.where(previous >= 0, previous[np.maximum(previous, 0)], -1)
    if np.any(table[levels] >= 0):
        raise ValueError("parent_vertices contain a cycle; a forest is required.")
    return table


def _depth(parents: np.ndarray, table: np.ndarray) -> int:
    """Largest number of parent steps from any vertex to its root."""
    current = np.arange(parents.size)
    depth = np.zeros(parents.size, dtype=np.int64)
    for level in range(table.shape[0] - 1, -1, -1):
        following = table[level][current]
        moved = following >= 0
        current = np.where(moved, following, current)
        depth += np.where(moved, 1 << level, 0)
    return int(np.max(depth))


@final
class PreparedBlockForest(StrictModule, NonTrainableState):
    """One numeric binding of the represented blocks to a validated topology.

    ``child_blocks[w]`` is ``A_w`` and ``parent_blocks[w]`` is ``B_w``; rows run
    over the row capacity with padded rows zero. ``numeric_version`` is ``0`` at
    preparation and advances on every refresh. ``retained_bytes`` is the exact
    logical size of the device leaves; ``workspace_bytes`` is the logical size of
    the certificate intermediates (not a compiler measurement).
    """

    __strict_contract__ = True
    topology: BlockForestTopology
    child_blocks: Float[ForestVertexDim, ForestRowDim, ForestColumnDim]
    parent_blocks: Float[ForestVertexDim, ForestRowDim, ForestColumnDim]
    numeric_version: Int32[Scalar]
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


def _blocks(
    topology: BlockForestTopology, child: ArrayLike, parent: ArrayLike, /
) -> tuple[Array, Array]:
    scope = Scope()
    form = Float[ForestVertexDim, ForestRowDim, ForestColumnDim]
    child_ = as_array(child, form, "child_blocks", scope=scope)
    parent_ = as_array(parent, form, "parent_blocks", scope=scope)
    expected = (topology.vertex_count, topology.row_capacity, topology.column_width)
    if child_.shape != expected:
        raise ValueError(f"Block forest blocks must have shape {expected}.")
    if child_.dtype != parent_.dtype or child_.dtype.itemsize < 4:
        raise ValueError("Block forest blocks need one real float32 or wider dtype.")
    return child_, parent_


def _workspace_bytes(topology: BlockForestTopology, itemsize: int, /) -> int:
    width, rows = topology.column_width, topology.row_capacity
    per_vertex = (
        rows * width * rows  # local product terms
        + rows * width * width  # local coupling terms
        + 3 * rows * rows
        + 4 * rows * width
        + 2 * width * width
        + 16 + 2 * len(topology.level_multiplicities)  # sweep vectors and weights
    )
    return topology.vertex_count * per_vertex * itemsize


@checked
def prepare_block_forest(
    topology: BlockForestTopology,
    child_blocks: ArrayLike,
    parent_blocks: ArrayLike,
    /,
    *,
    workspace_bytes: int = 512 * 1024 * 1024,
) -> PreparedBlockForest:
    """Bind the first numeric blocks after host admission of rounding and memory."""
    child, parent = _blocks(topology, child_blocks, parent_blocks)
    eps = float(np.finfo(child.dtype).eps)
    if 16 * topology.rounding_operations() * eps > _ROUNDING_CAPACITY:
        raise ValueError(
            "Block forest fan-in or block width exceeds the rounding-enclosure model "
            "for this dtype."
        )
    itemsize = child.dtype.itemsize
    workspace = _workspace_bytes(topology, itemsize)
    if isinstance(workspace_bytes, bool) or workspace_bytes < 0:
        raise ValueError("workspace_bytes must be a nonnegative integer.")
    if workspace > workspace_bytes:
        raise ValueError(
            f"Block forest certificate workspace {workspace} bytes exceeds budget "
            f"{workspace_bytes}."
        )
    retained = 2 * child.size * itemsize + 4 * (
        topology.parent_vertices.size + topology.ancestors.size + 1
    ) + topology.row_mask.size
    return PreparedBlockForest(
        topology,
        child,
        parent,
        jnp.asarray(0, dtype=jnp.int32),
        retained,
        workspace,
    )


@checked
def refresh_block_forest(
    prepared: PreparedBlockForest,
    child_blocks: ArrayLike,
    parent_blocks: ArrayLike,
    /,
) -> PreparedBlockForest:
    """Bind new numeric blocks to the same topology, dtype and admitted budgets."""
    child, parent = _blocks(prepared.topology, child_blocks, parent_blocks)
    if child.dtype != prepared.child_blocks.dtype:
        raise ValueError("Block forest refresh changed the block dtype.")
    return PreparedBlockForest(
        prepared.topology,
        child,
        parent,
        prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        prepared.retained_bytes,
        prepared.workspace_bytes,
    )


@final
class BlockForestRankCertificate(StrictModule):
    """Rigorous singular-value bounds and the full-row-rank decision they support.

    ``minimum_singular_value_lower_bound`` is always a valid lower bound (``0``
    when no contractive right inverse was verified). ``condition_upper_bound`` is
    ``U / L`` and is infinite when ``L`` is zero; neither is a computed singular
    value or condition number. ``rank_lower_bound`` equals ``row_count`` exactly
    when ``certified``; an uncertified result never claims rank deficiency.
    ``separation_lower_bound`` bounds ``L - threshold_upper_bound`` from below.
    ``numeric_version`` and ``topology_id`` identify the bound binding.
    """

    __strict_contract__ = True
    status: Int32[Scalar]
    certified: Bool[Scalar]
    rank_lower_bound: Int32[Scalar]
    minimum_singular_value_lower_bound: Float[Scalar]
    maximum_singular_value_upper_bound: Float[Scalar]
    condition_upper_bound: Float[Scalar]
    right_inverse_norm_upper_bound: Float[Scalar]
    residual_norm_upper_bound: Float[Scalar]
    residual_row_sum_upper_bound: Float[Scalar]
    residual_column_sum_upper_bound: Float[Scalar]
    threshold_upper_bound: Float[Scalar]
    separation_lower_bound: Float[Scalar]
    unresolved_local_blocks: Int32[Scalar]
    numeric_version: Int32[Scalar]
    row_count: int = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    relative_cutoff: float = eqx.field(static=True)
    absolute_cutoff: float = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)


def _local_coefficients(
    child: Array, parent: Array, row_mask: Array, /
) -> tuple[Array, Array]:
    """Approximate ``Q = A^T (A A^T)^-1`` and the stored ``C = Q B``.

    Accuracy only affects tightness: the bounds use the residuals of the stored
    coefficients. Explicit products and sums keep IEEE single-rounding semantics
    (a matrix-unit lowering may use reduced-precision passes).
    """
    rows = row_mask.shape[-1]
    real = row_mask.astype(child.dtype)
    identity = jnp.eye(rows, dtype=child.dtype)[None]
    gram = jnp.sum(child[:, :, None, :] * child[:, None, :, :], axis=-1)
    gram = gram + identity * (1 - real)[:, :, None]
    factor = jnp.linalg.cholesky(gram)

    def solve(cholesky: Array, right_hand_side: Array) -> Array:
        lower = jsp.linalg.solve_triangular(cholesky, right_hand_side, lower=True)
        return jsp.linalg.solve_triangular(cholesky.T, lower, lower=False)

    right = jnp.swapaxes(jax.vmap(solve)(factor, child), -1, -2)
    right = jnp.where(row_mask[:, None, :], right, 0)
    # One Newton-Schulz step Q <- Q (2I - A Q) squares the right-inverse defect.
    defect = identity * real[:, :, None] * real[:, None, :] - jnp.sum(
        child[:, :, :, None] * right[:, None, :, :], axis=2
    )
    right = right + jnp.sum(right[:, :, :, None] * defect[:, None, :, :], axis=2)
    right = jnp.where(row_mask[:, None, :], right, 0)
    propagation = jnp.sum(right[:, :, :, None] * parent[:, None, :, :], axis=2)
    return right, propagation


def _spectral_upper(magnitude: Array) -> Array:
    """Upper bound of the spectral norm of each trailing nonnegative matrix.

    ``min(||M||_F, sqrt(||M||_1 ||M||_inf))``; both dominate ``||M||_2``, and
    ``||M||_2 <= || |M| ||_2`` is monotone under entrywise domination.
    """
    rows, columns = magnitude.shape[-2:]
    frobenius = _up(
        jnp.sqrt(_up(jnp.sum(magnitude * magnitude, axis=(-2, -1)), rows * columns)),
        1,
    )
    row_sum = _up(jnp.sum(magnitude, axis=-1), columns)
    column_sum = _up(jnp.sum(magnitude, axis=-2), rows)
    product = _up(jnp.max(row_sum, axis=-1) * jnp.max(column_sum, axis=-1), 1)
    return jnp.minimum(frobenius, _up(jnp.sqrt(product), 1))


def _local_bounds(
    child: Array, parent: Array, row_mask: Array, right: Array, propagation: Array, /
) -> tuple[Array, Array, Array, Array]:
    """Verified ``a >= ||Q||``, ``b >= ||C||``, ``u >= ||AQ - I||``, ``v >= ||B - AC||``.

    A computed entry ``s = sum_k p_k - delta`` of ``c`` products differs from the
    exact entry by at most ``_pad(sum_k |p_k| + |delta|, c + 1)``; the entrywise
    dominating matrices add that enclosure to the computed magnitudes.
    """
    columns = child.shape[-1]
    real = row_mask.astype(child.dtype)
    square_mask = row_mask[:, :, None] & row_mask[:, None, :]
    identity = jnp.eye(row_mask.shape[-1], dtype=child.dtype)[None] * real[:, :, None]
    product_terms = child[:, :, :, None] * right[:, None, :, :]
    inverse_defect = jnp.sum(product_terms, axis=2) - identity
    inverse_scale = jnp.sum(jnp.abs(product_terms), axis=2) + identity
    inverse_bound = _up(
        jnp.abs(inverse_defect) + _pad(inverse_scale, columns + 1), 2
    )
    coupling_terms = child[:, :, :, None] * propagation[:, None, :, :]
    coupling_defect = parent - jnp.sum(coupling_terms, axis=2)
    coupling_scale = jnp.abs(parent) + jnp.sum(jnp.abs(coupling_terms), axis=2)
    coupling_bound = _up(
        jnp.abs(coupling_defect) + _pad(coupling_scale, columns + 1), 2
    )
    return (
        _spectral_upper(jnp.abs(right)),
        _spectral_upper(jnp.abs(propagation)),
        _spectral_upper(jnp.where(square_mask, inverse_bound, 0)),
        _spectral_upper(jnp.where(row_mask[:, :, None], coupling_bound, 0)),
    )


def _gather(values: Array, index: Array, /) -> Array:
    return jnp.where(index >= 0, values[jnp.maximum(index, 0)], 0)


def _scatter(values: Array, index: Array, /) -> Array:
    """Sum leading-axis entries into ``index``; ``-1`` entries go to a dropped segment."""
    count = values.shape[0]
    segments = jnp.where(index >= 0, index, count)
    return jax.ops.segment_sum(values, segments, num_segments=count + 1)[:count]


def _subtree_sum(
    source: Array, weights: list[Array], topology: BlockForestTopology, /
) -> Array:
    """``Sum_k B^k source`` with ``(B x)_v = sum_{parent(w)=v} b_w x_w``."""
    value = source
    for level, multiplicity in enumerate(topology.level_multiplicities):
        ancestors = topology.ancestors[level]
        moved = _scatter(_up(weights[level] * value, 1), ancestors)
        value = _up(value + moved, multiplicity + 1)
    return value


def _path_sum(source: Array, weights: list[Array], topology: BlockForestTopology) -> Array:
    """``Sum_k (B^T)^k source`` with ``(B^T x)_w = b_w x_parent(w)``."""
    value = source
    for level in range(len(topology.level_multiplicities)):
        moved = _up(weights[level] * _gather(value, topology.ancestors[level]), 1)
        value = _up(value + moved, 1)
    return value


def _operator_upper(
    child: Array, parent: Array, topology: BlockForestTopology, /
) -> Array:
    """``sigma_max(J) <= sqrt(max row sum * max column sum)`` of the blocks."""
    width, rows = topology.column_width, topology.row_capacity
    child_magnitude = jnp.abs(child)
    parent_magnitude = jnp.abs(parent)
    row_sum = _up(
        jnp.sum(child_magnitude, axis=-1) + jnp.sum(parent_magnitude, axis=-1),
        2 * width + 1,
    )
    column_sum = jnp.sum(child_magnitude, axis=1) + _scatter(
        jnp.sum(parent_magnitude, axis=1), topology.parent_vertices
    )
    column_sum = _up(column_sum, (topology.child_multiplicity + 1) * rows + 1)
    product = _up(jnp.max(row_sum) * jnp.max(column_sum), 1)
    return _up(jnp.sqrt(product), 1)


def _empty_certificate(
    prepared: PreparedBlockForest, relative_cutoff: float, absolute_cutoff: float, /
) -> BlockForestRankCertificate:
    """The zero-row convention: rank zero, condition one, no singular values."""
    dtype = prepared.child_blocks.dtype
    zero = jnp.asarray(0, dtype=dtype)
    return BlockForestRankCertificate(
        jnp.asarray(int(BlockForestRankStatus.CERTIFIED), dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        zero,
        zero,
        jnp.asarray(1, dtype=dtype),
        zero,
        zero,
        zero,
        zero,
        jnp.asarray(absolute_cutoff, dtype=dtype),
        zero,
        jnp.asarray(0, dtype=jnp.int32),
        prepared.numeric_version,
        0,
        prepared.topology.vertex_count * prepared.topology.column_width,
        relative_cutoff,
        absolute_cutoff,
        prepared.topology.topology_id,
    )


def _certify_with_coefficients(
    prepared: PreparedBlockForest,
    right: Array,
    propagation: Array,
    /,
    *,
    relative_cutoff: float,
    absolute_cutoff: float,
) -> BlockForestRankCertificate:
    """Certificate for the represented candidate built from the stored ``Q, C``."""
    topology = prepared.topology
    child, parent = prepared.child_blocks, prepared.parent_blocks
    dtype = child.dtype
    row_mask = topology.row_mask
    has_edge = jnp.any(row_mask, axis=-1)
    parent_vertex = topology.parent_vertices
    structured = jnp.all(jnp.where(row_mask[:, :, None], True, child == 0)) & jnp.all(
        jnp.where(row_mask[:, :, None] & (parent_vertex >= 0)[:, None, None], True, parent == 0)
    )
    blocks_valid = jnp.all(jnp.isfinite(child)) & jnp.all(jnp.isfinite(parent)) & structured

    inverse, coupling_norm, inverse_defect, coupling_defect = _local_bounds(
        child, parent, row_mask, right, propagation
    )
    local = jnp.stack((inverse, coupling_norm, inverse_defect, coupling_defect))
    resolved = jnp.all(jnp.isfinite(local), axis=0) | ~has_edge
    unresolved = jnp.sum(~resolved, dtype=jnp.int32)
    inverse, coupling_norm, inverse_defect, coupling_defect = (
        jnp.where(has_edge, value, 0) for value in local
    )
    # An anchored edge has no parent columns; its propagation weight is zero.
    coupling_norm = jnp.where(parent_vertex >= 0, coupling_norm, 0)

    weights = [coupling_norm]
    for level in range(1, len(topology.level_multiplicities)):
        previous = weights[-1]
        weights.append(_up(previous * _gather(previous, topology.ancestors[level - 1]), 1))
    path = _path_sum(inverse, weights, topology)
    subtree = _subtree_sum(jnp.ones_like(inverse), weights, topology)
    leaving = _up(_scatter(coupling_defect, parent_vertex), max(topology.child_multiplicity, 1))
    descent = _subtree_sum(leaving, weights, topology)

    column_load = jnp.max(jnp.where(has_edge, _up(inverse * subtree, 1), 0))
    right_inverse = _up(jnp.sqrt(_up(jnp.max(path) * column_load, 1)), 1)
    parent_path = _gather(path, parent_vertex)
    row_residual = jnp.max(
        jnp.where(has_edge, _up(inverse_defect + _up(coupling_defect * parent_path, 1), 1), 0)
    )
    column_residual = jnp.max(
        jnp.where(has_edge, _up(inverse_defect + _up(inverse * descent, 1), 1), 0)
    )
    residual = _up(jnp.sqrt(_up(row_residual * column_residual, 1)), 1)
    upper = _operator_upper(child, parent, topology)

    global_finite = (
        jnp.isfinite(right_inverse) & jnp.isfinite(residual) & jnp.isfinite(upper)
    )
    contractive = residual < 1
    valid = blocks_valid & (unresolved == 0) & global_finite & contractive
    valid = valid & (right_inverse > 0)
    safe_inverse = jnp.where(valid, right_inverse, 1)
    margin = _down(1 - jnp.where(valid, residual, 0), 1)
    lower = jnp.where(valid, jnp.maximum(_down(margin / safe_inverse, 1), 0), 0)
    relative = _up(jnp.asarray(relative_cutoff, dtype=dtype), 1)
    absolute = _up(jnp.asarray(absolute_cutoff, dtype=dtype), 1)
    threshold = _up(absolute + _up(relative * upper, 1), 1)
    separated = lower > threshold
    certified = valid & separated
    condition = jnp.where(
        lower > 0, _up(upper / jnp.where(lower > 0, lower, 1), 1), jnp.inf
    )
    status = jnp.select(
        (~blocks_valid, unresolved > 0, ~global_finite, ~contractive, ~separated),
        (
            int(BlockForestRankStatus.INVALID_BLOCKS),
            int(BlockForestRankStatus.LOCAL_BOUND_UNRESOLVED),
            int(BlockForestRankStatus.GLOBAL_BOUND_OVERFLOW),
            int(BlockForestRankStatus.RESIDUAL_NOT_CONTRACTIVE),
            int(BlockForestRankStatus.THRESHOLD_NOT_SEPARATED),
        ),
        int(BlockForestRankStatus.CERTIFIED),
    ).astype(jnp.int32)
    return BlockForestRankCertificate(
        status,
        certified,
        jnp.where(certified, topology.row_count, 0).astype(jnp.int32),
        lower,
        upper,
        condition.astype(dtype),
        right_inverse,
        residual,
        row_residual,
        column_residual,
        threshold,
        _down(lower - threshold, 1),
        unresolved,
        prepared.numeric_version,
        topology.row_count,
        topology.vertex_count * topology.column_width,
        relative_cutoff,
        absolute_cutoff,
        topology.topology_id,
    )


def _cutoff(value: float, name: str, /) -> float:
    cutoff = float(value)
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return cutoff


@checked
def certify_block_forest_rank(
    prepared: PreparedBlockForest,
    /,
    *,
    relative_cutoff: float,
    absolute_cutoff: float = 0.0,
) -> BlockForestRankCertificate:
    """Certify ``sigma_min(J) > absolute_cutoff + relative_cutoff * sigma_max(J)``.

    The decision concerns the exact operator assembled from the represented
    blocks. A dense SVD of a separately formed matrix agrees with it only when the
    reported separation exceeds that SVD's own backward error and any difference
    between the two matrix formations.
    """
    relative = _cutoff(relative_cutoff, "relative_cutoff")
    absolute = _cutoff(absolute_cutoff, "absolute_cutoff")
    if prepared.topology.row_count == 0:
        return _empty_certificate(prepared, relative, absolute)
    right, propagation = _local_coefficients(
        prepared.child_blocks, prepared.parent_blocks, prepared.topology.row_mask
    )
    return _certify_with_coefficients(
        prepared,
        right,
        propagation,
        relative_cutoff=relative,
        absolute_cutoff=absolute,
    )


__all__ = [
    "BlockForestRankCertificate",
    "BlockForestRankStatus",
    "BlockForestTopology",
    "certify_block_forest_rank",
    "ForestColumnDim",
    "ForestLevelDim",
    "ForestRowDim",
    "ForestVertexDim",
    "prepare_block_forest",
    "PreparedBlockForest",
    "refresh_block_forest",
]
