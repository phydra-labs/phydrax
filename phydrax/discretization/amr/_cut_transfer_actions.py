"""Bounded component-SVD actions for cut-complex preparation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from scipy.sparse.csgraph import connected_components

from ..._strict import StrictModule


class SparseAction(StrictModule):
    rows: Array
    columns: Array
    values: Array
    output_size: int

    def __init__(self, matrix: sp.csr_matrix) -> None:
        entries = matrix.tocoo()
        self.rows = jnp.asarray(entries.row, dtype=jnp.int32)
        self.columns = jnp.asarray(entries.col, dtype=jnp.int32)
        self.values = jnp.asarray(entries.data, dtype=jnp.float64)
        self.output_size = matrix.shape[0]

    def __call__(self, value: Array) -> Array:
        return (
            jnp.zeros((self.output_size,), dtype=value.dtype)
            .at[self.rows]
            .add(self.values * value[self.columns])
        )


class ComponentMinimumNorm(StrictModule):
    rows: tuple[Array, ...]
    columns: tuple[Array, ...]
    left: tuple[Array, ...]
    right: tuple[Array, ...]
    reciprocal: tuple[Array, ...]
    output_size: int
    rank: int
    component_shapes: tuple[tuple[int, int], ...]

    def __init__(
        self, matrix: sp.csr_matrix, tolerance: float, workspace_bytes: int
    ) -> None:
        rows, columns = matrix.shape
        entries = matrix.tocoo()
        graph_rows = np.concatenate((entries.row, rows + entries.col))
        graph_columns = np.concatenate((rows + entries.col, entries.row))
        graph = sp.csr_matrix(
            (np.ones(graph_rows.size, dtype=np.float64), (graph_rows, graph_columns)),
            shape=(rows + columns, rows + columns),
        )
        _, labels = connected_components(graph, directed=False)
        blocks = []
        for label in np.unique(labels):
            row = np.flatnonzero(labels[:rows] == label)
            column = np.flatnonzero(labels[rows:] == label)
            if row.size == 0 or column.size == 0:
                continue
            m, n = row.size, column.size
            # Include input copies, economy factors, and divide-and-conquer work.
            required = 8 * (6 * m * n + 8 * min(m, n) ** 2 + 8 * (m + n))
            if required > workspace_bytes:
                raise ValueError(
                    f"Cut transfer component ({m}, {n}) requires {required} workspace bytes; budget is {workspace_bytes}."
                )
            dense = matrix[row][:, column].toarray()
            left, singular, right = np.linalg.svd(dense, full_matrices=False)
            blocks.append((row, column, left, singular, right))
        maximum = max((float(s[0]) for _, _, _, s, _ in blocks if s.size), default=0.0)
        # A block-local cutoff would change the rank of small-scale components.
        retained = [s > tolerance * maximum for _, _, _, s, _ in blocks]
        self.rows = tuple(jnp.asarray(r, dtype=jnp.int32) for r, _, _, _, _ in blocks)
        self.columns = tuple(jnp.asarray(c, dtype=jnp.int32) for _, c, _, _, _ in blocks)
        self.left = tuple(jnp.asarray(u, dtype=jnp.float64) for _, _, u, _, _ in blocks)
        self.right = tuple(jnp.asarray(v, dtype=jnp.float64) for _, _, _, _, v in blocks)
        self.reciprocal = tuple(
            jnp.asarray(
                np.divide(1.0, s, out=np.zeros_like(s), where=keep), dtype=jnp.float64
            )
            for (_, _, _, s, _), keep in zip(blocks, retained, strict=True)
        )
        self.output_size = columns
        self.rank = sum(np.count_nonzero(keep).item() for keep in retained)
        self.component_shapes = tuple((r.size, c.size) for r, c, _, _, _ in blocks)

    def __call__(self, value: Array) -> Array:
        output = jnp.zeros((self.output_size,), dtype=value.dtype)
        for rows, columns, left, right, reciprocal in zip(
            self.rows, self.columns, self.left, self.right, self.reciprocal, strict=True
        ):
            output = output.at[columns].set(
                right.T @ (reciprocal * (left.T @ value[rows])),
                unique_indices=True,
                indices_are_sorted=True,
            )
        return output


class CommutingAction(StrictModule):
    upper: SparseAction | CommutingAction
    source_d: SparseAction
    target_d: SparseAction
    geometric: SparseAction
    solve: ComponentMinimumNorm

    def __call__(self, value: Array) -> Array:
        geometric = self.geometric(value)
        return geometric + self.solve(
            self.upper(self.source_d(value)) - self.target_d(geometric)
        )
