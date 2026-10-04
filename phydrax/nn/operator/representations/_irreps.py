#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""General real O(3) irrep layouts and exact real coupling coefficients.

The native component basis of every general block is the real harmonic basis of
`phydrax.special.RealCartesianHarmonics`: orders ``m = -l, ..., l`` of real
orthonormal harmonics without the Condon--Shortley phase, so degree one is
ordered ``(y, z, x)``. Packed values concatenate blocks in declaration order and
store each block multiplicity-major, ``(multiplicity, 2 l + 1)``.

Real coupling coefficients are derived from the exact Condon--Shortley SU(2)
coefficients at doubled spins ``2 l`` through the exact complex-to-real map
``S = U Y``:

``C_real[o, i, j] = g * sum U_out[o, M] C[M1, M2, M] conj(U_left[i, M1]) conj(U_right[j, M2])``

with gauge ``g = 1`` for even ``l1 + l2 + l3`` and ``g = -i`` for odd, the only
values for which the result is real. With this gauge the vector cross-product
path ``(1, 1) -> 1`` is ``+(a x b) / sqrt(2)`` in Cartesian components. Each real
entry is a sum of at most two terms of equal or distinct exact magnitude, so
reality and vanishing are decided in exact arithmetic; omitted entries are
mathematical zeros only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._model import register_artifact_value
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_integer,
)
from phydrax.axes import Axis, AxisKey
from phydrax.ein import contract
from phydrax.linalg import determinant_small_linear, SmallLinearSolvePlan
from phydrax.special._real_harmonic import _real_harmonic_basis_terms
from phydrax.typing import Dim, HostFloat64, HostInt32, parse


O3Parity: TypeAlias = Literal[-1, 1]

# Real basis position of each Cartesian axis of a degree-one block: (y, z, x).
_DEGREE_ONE_CARTESIAN_AXES = (1, 2, 0)


def _orthogonal_frame(orthogonal: ArrayLike, /) -> tuple[Array, Array]:
    """Validate a 3-D orthogonal frame change and return it with its determinant."""
    matrix = jnp.asarray(orthogonal)
    if matrix.shape != (3, 3):
        raise ValueError("O(3) transforms require a (3, 3) matrix.")
    if not isinstance(matrix, jax_core.Tracer):
        error = jnp.max(jnp.abs(matrix.T @ matrix - jnp.eye(3, dtype=matrix.dtype)))
        if float(error) > 1e-6:
            raise ValueError("O(3) transform matrix must be orthogonal.")
    return matrix, determinant_small_linear(SmallLinearSolvePlan(3), matrix)


@final
class O3IrrepBlock(StrictModule):
    """One named real O(3) irrep ``(l, parity)`` repeated over channels.

    Parity is the sign acquired under spatial inversion; spherical harmonics of
    degree ``l`` have parity ``(-1)**l``. Blocks are identified by name, never by
    equal degree, parity, or dimension.
    """

    name: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    parity: O3Parity = eqx.field(static=True)
    multiplicity: int = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        degree: int,
        parity: O3Parity,
        /,
        *,
        multiplicity: int = 1,
    ) -> None:
        name_ = canonical_identifier(name, "O3IrrepBlock name")
        degree_ = nonnegative_integer(degree, "O3IrrepBlock degree")
        parity_ = parse(parity, O3Parity, "parity")
        multiplicity_ = positive_integer(multiplicity, "O3IrrepBlock multiplicity")
        self.name = name_
        self.degree = degree_
        self.parity = parity_
        self.multiplicity = multiplicity_

    @property
    def dimension(self) -> int:
        """Number of real components, ``2 l + 1``."""
        return 2 * self.degree + 1

    @property
    def size(self) -> int:
        """Packed width, ``multiplicity * (2 l + 1)``."""
        return self.multiplicity * self.dimension


@final
class O3IrrepLayout(StrictModule, NonTrainableState):
    """Ordered packing of named real O(3) irrep blocks of arbitrary degree.

    Declaration order is the packed order and is never sorted or merged:
    blocks sharing degree and parity remain distinct. Each block is stored
    multiplicity-major with real harmonic components ``m = -l..l``.
    """

    blocks: tuple[O3IrrepBlock, ...]
    layout_id: str = eqx.field(static=True)

    def __init__(self, blocks: Sequence[O3IrrepBlock], /) -> None:
        resolved = tuple(blocks)
        if not resolved:
            raise ValueError("O3IrrepLayout needs at least one block.")
        if any(not isinstance(block, O3IrrepBlock) for block in resolved):
            raise TypeError("O3IrrepLayout blocks must be O3IrrepBlock values.")
        names = tuple(block.name for block in resolved)
        if len(set(names)) != len(names):
            raise ValueError("O3IrrepLayout block names must be unique.")
        self.blocks = resolved
        self.layout_id = canonical_fingerprint(
            {
                "kind": "o3-irrep-layout",
                "basis": "real-spherical",
                "packing": "multiplicity-major",
                "blocks": [
                    {
                        "name": block.name,
                        "degree": block.degree,
                        "parity": block.parity,
                        "multiplicity": block.multiplicity,
                    }
                    for block in resolved
                ],
            }
        )

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    @property
    def packed_size(self) -> int:
        return sum(block.size for block in self.blocks)

    @property
    def channel_count(self) -> int:
        """Number of irreducible channels, the total multiplicity."""
        return sum(block.multiplicity for block in self.blocks)

    def index(self, name: str, /) -> int:
        """Declared position of one named block."""
        names = self.block_names
        if name not in names:
            raise ValueError(f"O3IrrepLayout has no block named {name!r}.")
        return names.index(name)

    def offset(self, name: str, /) -> int:
        """Packed offset of one named block."""
        return sum(block.size for block in self.blocks[: self.index(name)])

    def multiplicity_axis(self, name: str, /) -> Axis:
        """Semantic multiplicity axis owned by one block of this layout."""
        block = self.blocks[self.index(name)]
        return Axis(
            AxisKey(f"o3-irrep-layout:{self.layout_id}", name), block.multiplicity
        )

    def component_axis(self, name: str, /) -> Axis:
        """Semantic real-harmonic component axis shared by every block of a degree."""
        degree = self.blocks[self.index(name)].degree
        return Axis(
            AxisKey("o3-real-spherical-component", f"degree-{degree}"),
            2 * degree + 1,
            labels=tuple(f"m={order}" for order in range(-degree, degree + 1)),
        )

    def split(self, values: ArrayLike, /) -> tuple[Array, ...]:
        """Unpack values into one ``(..., multiplicity, 2 l + 1)`` array per block."""
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.packed_size:
            raise ValueError(
                f"O(3) irrep values require packed size {self.packed_size}; got {array.shape}."
            )
        leading = array.shape[:-1]
        blocks: list[Array] = []
        start = 0
        for block in self.blocks:
            stop = start + block.size
            blocks.append(
                array[..., start:stop].reshape(
                    leading + (block.multiplicity, block.dimension)
                )
            )
            start = stop
        return tuple(blocks)

    def join(self, blocks: Sequence[ArrayLike], /) -> Array:
        """Pack one ``(..., multiplicity, 2 l + 1)`` array per declared block."""
        arrays = tuple(jnp.asarray(value) for value in blocks)
        if len(arrays) != len(self.blocks):
            raise ValueError("O(3) irrep block sequences must align with the layout.")
        leading: tuple[int, ...] | None = None
        flattened: list[Array] = []
        for block, array in zip(self.blocks, arrays, strict=True):
            shape = (block.multiplicity, block.dimension)
            if array.ndim < 2 or array.shape[-2:] != shape:
                raise ValueError(
                    f"O(3) irrep block {block.name!r} must end in {shape}; got {array.shape}."
                )
            current = array.shape[:-2]
            if leading is None:
                leading = current
            elif current != leading:
                raise ValueError("Every O(3) irrep block must share its leading shape.")
            flattened.append(array.reshape(current + (block.size,)))
        return jnp.concatenate(flattened, axis=-1)

    def representation_matrix(self, orthogonal: ArrayLike, /) -> Array:
        """Packed action ``(packed_size, packed_size)`` of one O(3) frame change."""
        matrix, determinant = _orthogonal_frame(orthogonal)
        actions = _degree_actions(self.blocks, matrix, determinant)
        size = self.packed_size
        packed = jnp.zeros((size, size), dtype=matrix.dtype)
        start = 0
        for block in self.blocks:
            action = _parity_factor(block.parity, determinant) * actions[block.degree]
            stop = start + block.size
            packed = packed.at[start:stop, start:stop].set(
                jnp.kron(jnp.eye(block.multiplicity, dtype=matrix.dtype), action)
            )
            start = stop
        return packed

    def transform(self, values: ArrayLike, orthogonal: ArrayLike, /) -> Array:
        """Apply one orthogonal frame change, including reflections, to packed values."""
        matrix, determinant = _orthogonal_frame(orthogonal)
        actions = _degree_actions(self.blocks, matrix, determinant)
        transformed = [
            _parity_factor(block.parity, determinant)
            * contract("ab,...ub->...ua", actions[block.degree], array)
            for block, array in zip(self.blocks, self.split(values), strict=True)
        ]
        return self.join(transformed)


def _parity_factor(parity: int, determinant: Array, /) -> Array:
    return jnp.where(determinant < 0, parity, 1).astype(determinant.dtype)


def _degree_actions(
    blocks: Sequence[O3IrrepBlock], matrix: Array, determinant: Array, /
) -> dict[int, Array]:
    maximum = max(block.degree for block in blocks)
    actions = _proper_actions(determinant * matrix, maximum)
    return {block.degree: actions[block.degree] for block in blocks}


def _proper_actions(rotation: Array, maximum_degree: int, /) -> tuple[Array, ...]:
    """Real Wigner matrices ``D_0..D_L`` of one proper rotation.

    ``D_1`` is the rotation in ``(y, z, x)`` order. Higher degrees use the
    equivariant isometry ``C`` of ``(l - 1) x 1 -> l``:
    ``D_l = C (D_{l-1} kron D_1) C^T``, valid because the component-normalized
    coupling rows are orthonormal.
    """
    dtype = rotation.dtype
    axes = jnp.asarray(_DEGREE_ONE_CARTESIAN_AXES)
    actions = [jnp.ones((1, 1), dtype=dtype)]
    if maximum_degree == 0:
        return tuple(actions)
    first = rotation[axes][:, axes]
    actions.append(first)
    for degree in range(2, maximum_degree + 1):
        coupling = jnp.asarray(
            o3_real_coupling(degree - 1, 1, degree).dense(), dtype=dtype
        )
        actions.append(
            contract("oij,ia,jb,pab->op", coupling, actions[degree - 1], first, coupling)
        )
    return tuple(actions)


def o3_irrep_action(orthogonal: ArrayLike, degree: int, parity: O3Parity, /) -> Array:
    """Real Wigner matrix of one irrep under an orthogonal frame change.

    An improper transform ``Q`` acts as ``D_l(det(Q) Q)`` times ``parity``.
    The result is differentiable in ``orthogonal``.
    """
    degree_ = nonnegative_integer(degree, "degree")
    parity_ = parse(parity, O3Parity, "parity")
    matrix, determinant = _orthogonal_frame(orthogonal)
    action = _proper_actions(determinant * matrix, degree_)[degree_]
    return _parity_factor(parity_, determinant) * action


class _CouplingEntryDim(Dim):
    """Exactly nonzero real coupling entries."""


@final
class O3RealCoupling(StrictModule, NonTrainableState):
    """Exact sparse component-normalized real coupling ``(l1, l2) -> l3``.

    ``indices`` columns are ``(output, left, right)`` real-component positions
    (``m + l``); ``values`` are the matching coefficients. Every output component
    has unit coefficient norm, ``sum_ij C[o, i, j]**2 = 1``.
    """

    __strict_contract__ = True

    left_degree: int = eqx.field(static=True)
    right_degree: int = eqx.field(static=True)
    output_degree: int = eqx.field(static=True)
    indices: HostInt32[_CouplingEntryDim, Literal[3]]
    values: HostFloat64[_CouplingEntryDim]

    def dense(self) -> np.ndarray:
        """Dense host table ``(2 l3 + 1, 2 l1 + 1, 2 l2 + 1)``, output-major."""
        table = np.zeros(
            (
                2 * self.output_degree + 1,
                2 * self.left_degree + 1,
                2 * self.right_degree + 1,
            ),
            dtype=np.float64,
        )
        table[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]] = self.values
        return table


# Gaussian units i**k as integer (real, imaginary) pairs.
_UNITS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _real_coupling_entries(
    left_degree: int, right_degree: int, output_degree: int, /
) -> list[tuple[int, int, int, float]]:
    # The tensor-network facade eagerly imports every network family; resolve
    # the coefficient owner only when a coupling is first prepared.
    from phydrax.tensor_network._su2 import _su2_clebsch_gordan_entries

    left_terms = _complex_rows(_real_harmonic_basis_terms(left_degree))
    right_terms = _complex_rows(_real_harmonic_basis_terms(right_degree))
    output_terms = _complex_rows(_real_harmonic_basis_terms(output_degree))
    gauge = 3 if (left_degree + right_degree + output_degree) % 2 else 0
    terms: dict[tuple[int, int, int], list[tuple[int, Fraction]]] = {}
    for entry in _su2_clebsch_gordan_entries(
        2 * left_degree, 2 * right_degree, 2 * output_degree
    ):
        sign_turns = 0 if entry.sign > 0 else 2
        for output_row, output_turns, output_halves in output_terms[entry.output]:
            for left_row, left_turns, left_halves in left_terms[entry.left]:
                for right_row, right_turns, right_halves in right_terms[entry.right]:
                    # Conjugated input rows contribute negative quarter turns.
                    turns = (
                        output_turns - left_turns - right_turns + sign_turns + gauge
                    ) % 4
                    halves = output_halves + left_halves + right_halves
                    terms.setdefault((output_row, left_row, right_row), []).append(
                        (turns, entry.square / 2**halves)
                    )
    entries: list[tuple[int, int, int, float]] = []
    for key in sorted(terms):
        value = _exact_real_value(terms[key])
        if value is not None:
            entries.append((*key, value))
    return entries


def _complex_rows(
    rows: tuple[tuple[tuple[int, int, int], ...], ...], /
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Invert the complex-to-real map: complex index -> (real row, turns, halves)."""
    inverse: dict[int, list[tuple[int, int, int]]] = {}
    for row, row_terms in enumerate(rows):
        for column, turns, halves in row_terms:
            inverse.setdefault(column, []).append((row, turns, halves))
    return {column: tuple(values) for column, values in inverse.items()}


def _exact_real_value(terms: Sequence[tuple[int, Fraction]], /) -> float | None:
    """Sum ``i**turns * sqrt(square)`` exactly; ``None`` for a mathematical zero.

    Real-basis entries combine the ``(M1, M2, M)`` and ``(-M1, -M2, -M)`` complex
    coefficients only, so at most two terms occur. Terms with distinct exact
    squares cannot cancel, so grouping by equal squares decides vanishing and
    reality exactly.
    """
    if len(terms) > 2:
        raise RuntimeError("Real O(3) coupling entry has more than two complex terms.")
    groups: dict[Fraction, tuple[int, int]] = {}
    for turns, square in terms:
        real, imaginary = groups.get(square, (0, 0))
        unit_real, unit_imaginary = _UNITS[turns]
        groups[square] = (real + unit_real, imaginary + unit_imaginary)
    if any(imaginary != 0 for _, imaginary in groups.values()):
        raise RuntimeError("Real O(3) coupling gauge produced an imaginary coefficient.")
    nonzero = [(real, square) for square, (real, _) in groups.items() if real != 0]
    if not nonzero:
        return None
    return math.fsum(
        real * math.sqrt(square.numerator / square.denominator)
        for real, square in nonzero
    )


def o3_real_coupling(
    left_degree: int, right_degree: int, output_degree: int, /
) -> O3RealCoupling:
    """Exact component-normalized real coupling of one degree triangle.

    The coupling ignores parity; a path is legal only when
    ``|l1 - l2| <= l3 <= l1 + l2`` and ``p3 = p1 * p2``. Results are immutable
    and prepared once per degree triangle.
    """
    left = nonnegative_integer(left_degree, "left_degree")
    right = nonnegative_integer(right_degree, "right_degree")
    output = nonnegative_integer(output_degree, "output_degree")
    if not abs(left - right) <= output <= left + right:
        raise ValueError(f"Degrees ({left}, {right}) cannot couple to degree {output}.")
    return _prepared_real_coupling(left, right, output)


@lru_cache(maxsize=None)
def _prepared_real_coupling(left: int, right: int, output: int, /) -> O3RealCoupling:
    entries = _real_coupling_entries(left, right, output)
    indices = np.asarray([entry[:3] for entry in entries], dtype=np.int32).reshape(
        (len(entries), 3)
    )
    values = np.asarray([entry[3] for entry in entries], dtype=np.float64)
    indices.setflags(write=False)
    values.setflags(write=False)
    return O3RealCoupling(
        left_degree=left,
        right_degree=right,
        output_degree=output,
        indices=indices,
        values=values,
    )


register_artifact_value("phydrax.nn.operator:O3IrrepBlock", O3IrrepBlock)
register_artifact_value("phydrax.nn.operator:O3IrrepLayout", O3IrrepLayout)


__all__ = [
    "O3IrrepBlock",
    "O3IrrepLayout",
    "O3Parity",
    "O3RealCoupling",
    "o3_irrep_action",
    "o3_real_coupling",
]
