from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._jax_config import x64_enabled
from .._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage


class _PreparedHostReferenceRules:
    """Private numerical operands of one preparation, never scientific evidence."""

    def __init__(self, storage: CoordinateLiveStorage, /) -> None:
        storage.set_bound(4096)
        self._storage = storage
        self._retained_bytes = 4096
        self._rules: dict[tuple[str, int, str, bool], tuple[np.ndarray, np.ndarray]] = {}

    def rule(
        self, cell_kind: str, polynomial_degree: int, /
    ) -> tuple[np.ndarray, np.ndarray]:
        # The host owner is always float64; the original JAX round trip rounds
        # to the configured precision before any NumPy mapped operation.
        key = (cell_kind, polynomial_degree, np.dtype(np.float64).str, x64_enabled())
        existing = self._rules.get(key)
        if existing is not None:
            return existing
        if cell_kind.startswith(("simplex:", "tensor:")):
            dimension = int(cell_kind.split(":")[1])
            count = max(2, polynomial_degree + dimension)
        else:
            dimension = {
                "interval": 1,
                "triangle": 2,
                "quadrilateral": 2,
                "tetrahedron": 3,
                "prism": 3,
                "pyramid": 3,
                "hexahedron": 3,
            }[cell_kind]
            count = max(2, polynomial_degree + (2 if cell_kind == "tetrahedron" else 1))
        points_count = count**dimension
        # Admit before leggauss/meshgrid, including their dense eigensolver,
        # grid/stack temporaries, the transfer and private host copies. Existing
        # rules remain live during this miss. This is the caller's ledger, not
        # an independent allowance or a reset of its resources.
        temporary_bytes = (
            16_384
            + 1024 * dimension
            + 128 * count**2
            + 8 * (6 * dimension + 12) * points_count
        )
        self._storage.set_bound(
            self._retained_bytes + temporary_bytes,
            work=count**2 + points_count * (dimension + 1),
        )
        try:
            points, weights = degree_aware_reference_rule(cell_kind, polynomial_degree)
            host_points, host_weights = (
                np.array(points, copy=True), np.array(weights, copy=True)
            )
            host_points.setflags(write=False)
            host_weights.setflags(write=False)
            self._rules[key] = host_points, host_weights
            del points, weights
            # Includes dictionary capacity, key/tuple/scalar metadata and array
            # headers; the arrays own their buffers (no unaccounted base bank).
            self._retained_bytes += 2048 + host_points.nbytes + host_weights.nbytes
        finally:
            self._storage.set_bound(self._retained_bytes)
        return host_points, host_weights


@contextmanager
def _prepared_host_reference_rules(
    budget: CoordinateEnclosureBudget, /
) -> Iterator[_PreparedHostReferenceRules]:
    """Retain private, precision-qualified rules only during this actual call.

    Consumers must not publish these operands. Mapped points and densities are
    always evaluated afresh from the fully authenticated current geometry.
    """
    with budget.live_storage() as storage:
        rules = _PreparedHostReferenceRules(storage)
        try:
            yield rules
        finally:
            rules._rules.clear()


def host_degree_aware_reference_rule(
    cell_kind: str, polynomial_degree: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare positive-weight reference quadrature on the host in rule order."""
    if cell_kind.startswith(("simplex:", "tensor:")):
        kind, dimension_text = cell_kind.split(":")
        dimension = int(dimension_text)
        count = max(2, polynomial_degree + dimension)
        axis, weights = np.polynomial.legendre.leggauss(count)
        axis, weights = 0.5 * (axis + 1), 0.5 * weights
        grids = np.meshgrid(*((axis,) * dimension), indexing="ij")
        weight_grids = np.meshgrid(*((weights,) * dimension), indexing="ij")
        points = np.stack(grids, axis=-1)
        combined = np.prod(np.stack(weight_grids, axis=-1), axis=-1)
        if kind == "simplex":
            scale = np.ones_like(grids[0])
            for coordinate in range(dimension):
                points[..., coordinate] = scale * grids[coordinate]
                combined *= (1 - grids[coordinate]) ** (dimension - coordinate - 1)
                scale *= 1 - grids[coordinate]
        return np.asarray(points.reshape((-1, dimension)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    count = max(2, int(polynomial_degree) + 1)
    if cell_kind == "tetrahedron":
        count = max(count, int(polynomial_degree) + 2)
    axis, weights = np.polynomial.legendre.leggauss(count)
    axis = 0.5 * (axis + 1.0)
    weights = 0.5 * weights
    if cell_kind == "interval":
        return np.asarray(axis[:, None], dtype=np.float64), np.asarray(
            weights, dtype=np.float64
        )
    if cell_kind == "triangle":
        first, second = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack((first, (1.0 - first) * second), axis=-1)
        combined = weights[:, None] * weights[None, :] * (1.0 - first)
        return np.asarray(points.reshape((-1, 2)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    if cell_kind == "quadrilateral":
        first, second = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack((first, second), axis=-1)
        combined = weights[:, None] * weights[None, :]
        return np.asarray(points.reshape((-1, 2)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    if cell_kind == "tetrahedron":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        one_minus_first = 1.0 - first
        one_minus_second = 1.0 - second
        points = np.stack(
            (
                first,
                one_minus_first * second,
                one_minus_first * one_minus_second * third,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * one_minus_first**2
            * one_minus_second
        )
        return np.asarray(points.reshape((-1, 3)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    if cell_kind == "prism":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack((first, (1.0 - first) * second, third), axis=-1)
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * (1.0 - first)
        )
        return np.asarray(points.reshape((-1, 3)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    if cell_kind == "pyramid":
        first, second, height = np.meshgrid(axis, axis, axis, indexing="ij")
        scale = 1.0 - height
        points = np.stack(
            (
                scale * first + 0.5 * height,
                scale * second + 0.5 * height,
                height,
            ),
            axis=-1,
        )
        combined = (
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * scale**2
        )
        return np.asarray(points.reshape((-1, 3)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    if cell_kind == "hexahedron":
        first, second, third = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.stack((first, second, third), axis=-1)
        combined = (
            weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        )
        return np.asarray(points.reshape((-1, 3)), dtype=np.float64), np.asarray(
            combined.reshape((-1,)), dtype=np.float64
        )
    raise ValueError(
        "host_degree_aware_reference_rule.cell_kind must name a supported "
        f"finite-element cell kind; got {cell_kind!r}"
    )


def degree_aware_reference_rule(
    cell_kind: str, polynomial_degree: int, /
) -> tuple[Array, Array]:
    """Transfer the host reference rule once to the configured JAX precision."""
    points, weights = host_degree_aware_reference_rule(cell_kind, polynomial_degree)
    return jnp.asarray(points), jnp.asarray(weights)
