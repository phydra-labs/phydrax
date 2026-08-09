#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._tensor import TensorFieldLayout


def _matrix_index(matrices: np.ndarray, candidate: np.ndarray, tolerance: float) -> int:
    distances = np.max(np.abs(matrices - candidate[None, ...]), axis=(1, 2))
    matches = np.flatnonzero(distances <= tolerance)
    if matches.size != 1:
        raise ValueError(
            "Finite group matrices are not uniquely closed under composition."
        )
    return int(matches[0])


def _signed_permutation_data(
    matrix: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    absolute = np.abs(matrix)
    nonzero = absolute > tolerance
    if not (
        np.all(np.sum(nonzero, axis=0) == 1)
        and np.all(np.sum(nonzero, axis=1) == 1)
        and np.all(np.abs(absolute[nonzero] - 1.0) <= tolerance)
        and np.all(absolute[~nonzero] <= tolerance)
    ):
        return None
    permutation = tuple(
        int(np.flatnonzero(nonzero[row])[0]) for row in range(matrix.shape[0])
    )
    signs = tuple(
        int(np.sign(matrix[row, permutation[row]])) for row in range(matrix.shape[0])
    )
    return permutation, signs


def _canonical_signed_permutations(
    dimension: int,
    /,
    *,
    proper_only: bool,
) -> np.ndarray:
    matrices = []
    for permutation in itertools.permutations(range(dimension)):
        for signs in itertools.product((-1, 1), repeat=dimension):
            matrix = np.zeros((dimension, dimension), dtype=float)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if not proper_only or round(float(np.linalg.det(matrix))) == 1:
                matrices.append(matrix)
    identity = np.eye(dimension)
    matrices.sort(
        key=lambda matrix: (
            0 if np.array_equal(matrix, identity) else 1,
            tuple(int(value) for value in matrix.reshape(-1)),
        )
    )
    return np.stack(matrices, axis=0)


class FiniteOrthogonalGroup(StrictModule, NonTrainableState):
    """Validated finite subgroup of O(d) with exact composition metadata."""

    name: str
    matrices: Array
    dimension: int
    identity_index: int
    multiplication_table: tuple[tuple[int, ...], ...]
    inverse_indices: tuple[int, ...]
    lattice_permutations: tuple[tuple[int, ...], ...] | None
    lattice_signs: tuple[tuple[int, ...], ...] | None
    fingerprint: str

    def __init__(
        self,
        name: str,
        matrices: Array | Sequence[Array],
        /,
        *,
        tolerance: float = 1e-6,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Finite group names must be non-empty.")
        tolerance_value = float(tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        host = np.asarray(matrices)
        if host.ndim != 3 or host.shape[0] == 0 or host.shape[1] != host.shape[2]:
            raise ValueError("matrices must have shape (order, dimension, dimension).")
        if host.shape[1] == 0:
            raise ValueError("Finite group dimension must be positive.")
        if not np.issubdtype(host.dtype, np.floating):
            host = host.astype(float)
        if np.any(~np.isfinite(host)):
            raise ValueError("Finite group matrices must be finite.")
        order, dimension, _ = host.shape
        identity = np.eye(dimension, dtype=host.dtype)
        orthogonality_defect = np.max(
            np.abs(np.einsum("gji,gjk->gik", host, host) - identity[None, ...])
        )
        if orthogonality_defect > tolerance_value:
            raise ValueError("Every finite group matrix must be orthogonal.")
        determinants = np.linalg.det(host)
        if np.max(np.abs(np.abs(determinants) - 1.0)) > tolerance_value:
            raise ValueError(
                "Finite orthogonal matrices must have determinant magnitude one."
            )
        pairwise = np.max(
            np.abs(host[:, None, :, :] - host[None, :, :, :]),
            axis=(2, 3),
        )
        duplicate_mask = (pairwise <= tolerance_value) & ~np.eye(order, dtype=bool)
        if np.any(duplicate_mask):
            raise ValueError("Finite group matrices must be unique.")
        identity_index = _matrix_index(host, identity, tolerance_value)
        table = tuple(
            tuple(
                _matrix_index(host, host[left] @ host[right], tolerance_value)
                for right in range(order)
            )
            for left in range(order)
        )
        inverses = tuple(
            _matrix_index(host, host[index].T, tolerance_value) for index in range(order)
        )
        lattice_data = tuple(
            _signed_permutation_data(matrix, tolerance_value) for matrix in host
        )
        if all(value is not None for value in lattice_data):
            lattice_permutations = tuple(
                value[0] for value in lattice_data if value is not None
            )
            lattice_signs = tuple(value[1] for value in lattice_data if value is not None)
        else:
            lattice_permutations = None
            lattice_signs = None
        rounded = np.round(host.astype(np.float64), decimals=12)
        digest = hashlib.sha256()
        digest.update(resolved_name.encode("utf-8"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(rounded.tobytes())

        self.name = resolved_name
        self.matrices = jnp.asarray(host)
        self.dimension = int(dimension)
        self.identity_index = identity_index
        self.multiplication_table = table
        self.inverse_indices = inverses
        self.lattice_permutations = lattice_permutations
        self.lattice_signs = lattice_signs
        self.fingerprint = digest.hexdigest()

    @property
    def order(self) -> int:
        return int(self.matrices.shape[0])

    @property
    def is_proper(self) -> bool:
        return bool(np.all(np.linalg.det(np.asarray(self.matrices)) > 0.0))

    @property
    def supports_lattice_action(self) -> bool:
        return self.lattice_permutations is not None

    def compose(self, left: int, right: int, /) -> int:
        return self.multiplication_table[int(left)][int(right)]

    def inverse(self, element: int, /) -> int:
        return self.inverse_indices[int(element)]

    def spatial_action(
        self,
        values: Array,
        element: int,
        /,
        *,
        spatial_axes: Sequence[int] | None = None,
    ) -> Array:
        """Apply ``f(x) -> f(g^-1 x)`` on a periodic signed-permutation lattice."""
        if self.lattice_permutations is None or self.lattice_signs is None:
            raise ValueError("This finite group does not preserve a Cartesian lattice.")
        index = int(element)
        if not 0 <= index < self.order:
            raise IndexError("Finite group element index is out of range.")
        array = jnp.asarray(values)
        axes = (
            tuple(range(self.dimension))
            if spatial_axes is None
            else tuple(int(axis) for axis in spatial_axes)
        )
        if len(axes) != self.dimension or len(set(axes)) != len(axes):
            raise ValueError(
                "spatial_axes must uniquely identify every spatial dimension."
            )
        normalized_axes = tuple(axis % array.ndim for axis in axes)
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("spatial_axes must be unique after normalization.")
        moved = jnp.moveaxis(array, normalized_axes, tuple(range(self.dimension)))
        permutation = self.lattice_permutations[index]
        signs = self.lattice_signs[index]
        spatial_shape = moved.shape[: self.dimension]
        if any(
            spatial_shape[axis] != spatial_shape[permutation[axis]]
            for axis in range(self.dimension)
        ):
            raise ValueError(
                "Axis-permuting group actions require matching lattice sizes."
            )
        transposed = jnp.transpose(
            moved,
            permutation + tuple(range(self.dimension, moved.ndim)),
        )
        for axis, sign in enumerate(signs):
            if sign < 0:
                indices = (-jnp.arange(transposed.shape[axis])) % transposed.shape[axis]
                transposed = jnp.take(transposed, indices, axis=axis)
        return jnp.moveaxis(transposed, tuple(range(self.dimension)), normalized_axes)

    def field_action(
        self,
        values: Array,
        layout: TensorFieldLayout,
        element: int,
        /,
        *,
        spatial_axes: Sequence[int] | None = None,
    ) -> Array:
        """Apply spatial pullback followed by the declared tensor channel action."""
        if not isinstance(layout, TensorFieldLayout):
            raise TypeError("layout must be a TensorFieldLayout.")
        if layout.dimension != self.dimension:
            raise ValueError("Tensor layout and finite group dimensions must agree.")
        spatial = self.spatial_action(values, element, spatial_axes=spatial_axes)
        return layout.transform(spatial, self.matrices[int(element)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matrices": np.asarray(self.matrices).tolist(),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "FiniteOrthogonalGroup":
        group = cls(str(value["name"]), value["matrices"])
        expected = value.get("fingerprint")
        if expected is not None and str(expected) != group.fingerprint:
            raise ValueError("Finite group fingerprint does not match its matrices.")
        return group

    @classmethod
    def c4(cls, /) -> "FiniteOrthogonalGroup":
        generator = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        matrices = np.stack(
            (np.eye(2), generator, generator @ generator, generator.T),
            axis=0,
        )
        return cls("C4", matrices)

    @classmethod
    def d4(cls, /) -> "FiniteOrthogonalGroup":
        rotations = np.asarray(cls.c4().matrices)
        reflection = np.asarray([[1.0, 0.0], [0.0, -1.0]])
        matrices = np.concatenate((rotations, rotations @ reflection), axis=0)
        return cls("D4", matrices)

    @classmethod
    def cube_rotations(cls, /) -> "FiniteOrthogonalGroup":
        return cls(
            "cube_rotations",
            _canonical_signed_permutations(3, proper_only=True),
        )

    @classmethod
    def cube_orthogonal(cls, /) -> "FiniteOrthogonalGroup":
        return cls(
            "cube_orthogonal",
            _canonical_signed_permutations(3, proper_only=False),
        )


__all__ = ["FiniteOrthogonalGroup"]
