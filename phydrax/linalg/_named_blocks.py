#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact named and path-based coordinate maps on existing block spaces.

Every map in this module is a coordinate identification: selected canonical source
coordinates are copied, embedded, or regrouped without arithmetic. Coordinate
transposes are the reversed maps. Hilbert adjoints are always evaluated against the
declared source and target pairings, so a map between differently paired spaces never
confuses its transpose with its adjoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..typing import checked
from ._operators import (
    _assemble_operator_diagonal,
    _generic_adjoint,
    AbstractLinearOperator,
    BlockLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    TransposeLinearOperator,
)
from ._properties import LinearCapabilityError, OperatorCapabilities, OperatorProperties
from ._spaces import _coordinate_dtype, AbstractVectorSpace, ArraySpace, BlockSpace


type BlockPath = tuple[str, ...]
type CoordinateRange = tuple[int, int]
type CoordinateRanges = tuple[CoordinateRange, ...]
type BlockMember = BlockPath | tuple[BlockPath, ...] | CoordinateBlock


def _exact_int(value: object, name: str, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer.")
    return value


def _block_path(value: object, /) -> BlockPath:
    if not isinstance(value, tuple) or not value:
        raise TypeError("A block path must be a non-empty tuple of block names.")
    if not all(isinstance(name, str) and name for name in value):
        raise TypeError("Block path entries must be non-empty strings.")
    return value


def _path_node(
    space: AbstractVectorSpace, path: BlockPath, /
) -> tuple[AbstractVectorSpace, CoordinateRange]:
    """Resolve one path to its member space and canonical coordinate span."""
    current = space
    start = 0
    for depth, name in enumerate(path):
        if not isinstance(current, BlockSpace):
            raise ValueError(
                f"Block path {path!r} descends below a non-block member at depth {depth}."
            )
        if name not in current.names:
            raise ValueError(f"Block path {path!r} names unknown block {name!r}.")
        index = current.names.index(name)
        start += sum(member.size for member in current.spaces[:index])
        current = current.spaces[index]
    return current, (start, start + current.size)


def _member_offsets(space: BlockSpace, /) -> tuple[int, ...]:
    offsets = [0]
    for member in space.spaces[:-1]:
        offsets.append(offsets[-1] + member.size)
    return tuple(offsets)


def _range_length(ranges: CoordinateRanges, /) -> int:
    return sum(stop - start for start, stop in ranges)


def _positions(ranges: CoordinateRanges, /) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in ranges]
    )


def _gather(coordinates: Array, ranges: CoordinateRanges, /) -> Array:
    pieces = tuple(coordinates[start:stop] for start, stop in ranges)
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def _scatter(
    placements: Sequence[tuple[int, Array]],
    size: int,
    dtype: np.dtype,
    /,
) -> Array:
    """Place disjoint coordinate segments into a zero vector without scatter updates."""
    pieces: list[Array] = []
    cursor = 0
    for start, values in sorted(placements, key=lambda item: item[0]):
        if start > cursor:
            pieces.append(jnp.zeros((start - cursor,), dtype=dtype))
        pieces.append(values)
        cursor = start + values.shape[0]
    if cursor < size:
        pieces.append(jnp.zeros((size - cursor,), dtype=dtype))
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def _scatter_ranges(
    coordinates: Array, ranges: CoordinateRanges, size: int, dtype: np.dtype, /
) -> Array:
    placements: list[tuple[int, Array]] = []
    offset = 0
    for start, stop in ranges:
        placements.append((start, coordinates[offset : offset + stop - start]))
        offset += stop - start
    return _scatter(placements, size, dtype)


def _validate_disjoint(ranges: Sequence[CoordinateRange], size: int, /) -> None:
    cursor = 0
    for start, stop in sorted(ranges):
        if start < cursor:
            raise ValueError("Selected coordinate ranges must be disjoint.")
        if stop > size:
            raise ValueError(
                f"Selected coordinate range {(start, stop)} exceeds size {size}."
            )
        cursor = stop


def _identity_ordered(ranges: Sequence[CoordinateRanges], size: int, /) -> bool:
    flat = tuple(value for member in ranges for value in member)
    cursor = 0
    for start, stop in flat:
        if start != cursor:
            return False
        cursor = stop
    return cursor == size


class CoordinateBlock(StrictModule):
    """Selection member identified with explicit canonical source coordinate ranges.

    The member vector's canonical coordinates are the concatenation of the listed
    source ranges, in order. The member space may carry a different pairing and
    space identity from the source; the selection is a coordinate identification.
    """

    space: AbstractVectorSpace
    ranges: CoordinateRanges = eqx.field(static=True)

    @checked
    def __init__(
        self,
        space: AbstractVectorSpace,
        ranges: Sequence[CoordinateRange],
        /,
    ) -> None:
        parsed: list[CoordinateRange] = []
        for value in ranges:
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("Coordinate ranges must be (start, stop) tuples.")
            start = _exact_int(value[0], "Coordinate range start")
            stop = _exact_int(value[1], "Coordinate range stop")
            if start < 0 or stop <= start:
                raise ValueError("Coordinate ranges require 0 <= start < stop.")
            parsed.append((start, stop))
        if not parsed:
            raise ValueError("A coordinate block requires at least one range.")
        ranges_ = tuple(parsed)
        if _range_length(ranges_) != space.size:
            raise ValueError(
                "Coordinate block ranges must contain exactly the member space size."
            )
        self.space = space
        self.ranges = ranges_


def _selection_member(
    source: AbstractVectorSpace,
    member: object,
    /,
) -> tuple[AbstractVectorSpace, CoordinateRanges, tuple[BlockPath, ...] | None]:
    if isinstance(member, CoordinateBlock):
        return member.space, member.ranges, None
    if not isinstance(member, tuple) or not member:
        raise TypeError(
            "Selection members must be block paths, tuples of block paths, or "
            "CoordinateBlock values."
        )
    if not isinstance(source, BlockSpace):
        raise TypeError("Path-based selection members require a BlockSpace source.")
    if all(isinstance(value, str) for value in member):
        path = _block_path(member)
        space, span = _path_node(source, path)
        return space, (span,), (path,)
    paths = tuple(_block_path(value) for value in member)
    nodes = tuple(_path_node(source, path) for path in paths)
    group = BlockSpace(
        tuple(space for space, _ in nodes),
        names=tuple("/".join(path) for path in paths),
    )
    return group, tuple(span for _, span in nodes), paths


class BlockSelection(StrictModule):
    """Exact named selection of canonical source coordinates into one ``BlockSpace``.

    Each member is a block path of a ``BlockSpace`` source, a tuple of block paths
    (a named group whose inner names are the ``"/"``-joined paths), or an explicit
    ``CoordinateBlock``. Path members retain the source member spaces and therefore
    their pairings; the orthogonal-sum pairing of a ``BlockSpace`` makes their
    prolongation the Hilbert adjoint of their restriction.
    """

    source: AbstractVectorSpace
    target: BlockSpace
    member_ranges: tuple[CoordinateRanges, ...] = eqx.field(static=True)
    member_paths: tuple[tuple[BlockPath, ...] | None, ...] = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: AbstractVectorSpace,
        members: Sequence[tuple[str, BlockMember]],
        /,
    ) -> None:
        entries = tuple(members)
        if not entries:
            raise ValueError("A block selection requires at least one member.")
        names: list[str] = []
        spaces: list[AbstractVectorSpace] = []
        ranges: list[CoordinateRanges] = []
        paths: list[tuple[BlockPath, ...] | None] = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("Selection members must be (name, member) pairs.")
            name, member = entry
            if not isinstance(name, str) or not name:
                raise TypeError("Selection member names must be non-empty strings.")
            space, member_ranges, member_paths = _selection_member(source, member)
            names.append(name)
            spaces.append(space)
            ranges.append(member_ranges)
            paths.append(member_paths)
        _validate_disjoint(
            tuple(value for member in ranges for value in member), source.size
        )
        target = BlockSpace(tuple(spaces), names=tuple(names))
        if _coordinate_dtype(target) != _coordinate_dtype(source):
            raise TypeError("Selected members must share the source coordinate dtype.")
        self.source = source
        self.target = target
        self.member_ranges = tuple(ranges)
        self.member_paths = tuple(paths)
        self.selection_id = canonical_fingerprint(
            {
                "kind": "block-selection",
                "source": source.space_id,
                "target": target.space_id,
                "ranges": [[list(value) for value in member] for member in ranges],
            }
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self.target.names

    @property
    def covers_source(self) -> bool:
        """Whether the selection is a bijective coordinate identification."""
        return self.target.size == self.source.size

    @property
    def preserves_pairing(self) -> bool:
        """Whether every member is a path of the source ``BlockSpace``."""
        return all(paths is not None for paths in self.member_paths)

    def restrict(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        """Return the selected members of one source vector."""
        coordinates = self.source.flatten(vector)
        return tuple(
            space.unflatten(_gather(coordinates, ranges))
            for space, ranges in zip(self.target.spaces, self.member_ranges, strict=True)
        )

    def prolong(self, vector: PyTree[Any], /) -> PyTree[Array]:
        """Embed one selected vector into the source, with zeros elsewhere."""
        values = self.target.validate(vector)
        placements: list[tuple[int, Array]] = []
        for space, value, ranges in zip(
            self.target.spaces, values, self.member_ranges, strict=True
        ):
            coordinates = space.flatten(value)
            offset = 0
            for start, stop in ranges:
                placements.append((start, coordinates[offset : offset + stop - start]))
                offset += stop - start
        return self.source.unflatten(
            _scatter(placements, self.source.size, _coordinate_dtype(self.source))
        )


def _selection_matrix(selection: BlockSelection, /) -> Array:
    rows = np.arange(selection.target.size, dtype=np.int64)
    columns = _positions(
        tuple(value for member in selection.member_ranges for value in member)
    )
    matrix = np.zeros(
        (selection.target.size, selection.source.size),
        dtype=_coordinate_dtype(selection.source),
    )
    matrix[rows, columns] = 1
    return jnp.asarray(matrix)


def _selection_properties(selection: BlockSelection, /) -> OperatorProperties:
    return OperatorProperties(
        rank=selection.target.size,
        evidence={"rank": "construction"},
    )


class BlockRestrictionLinearOperator(AbstractLinearOperator):
    """Exact restriction of source vectors onto one named block selection."""

    selection: BlockSelection

    @checked
    def __init__(self, selection: BlockSelection, /) -> None:
        self.selection = selection
        self.source = selection.source
        self.target = selection.target
        self.properties = _selection_properties(selection)
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {"kind": "block-restriction", "selection": selection.selection_id}
        )

    def mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        return self.selection.restrict(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.selection.prolong(vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return _selection_matrix(self.selection)


class BlockProlongationLinearOperator(AbstractLinearOperator):
    """Exact zero-filled embedding of one named block selection into its source."""

    selection: BlockSelection

    @checked
    def __init__(self, selection: BlockSelection, /) -> None:
        self.selection = selection
        self.source = selection.target
        self.target = selection.source
        self.properties = _selection_properties(selection)
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {"kind": "block-prolongation", "selection": selection.selection_id}
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.selection.prolong(vector)

    def transpose_mv(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        return self.selection.restrict(vector)

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        return _selection_matrix(self.selection).T


type CoordinatePiece = tuple[int, int, int]


class _CoordinateTransferLinearOperator(AbstractLinearOperator):
    """Partial coordinate identity ``target[t:t+n] = source[s:s+n]`` between spaces."""

    pieces: tuple[CoordinatePiece, ...] = eqx.field(static=True)

    def __init__(
        self,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        pieces: tuple[CoordinatePiece, ...],
        /,
    ) -> None:
        if _coordinate_dtype(source) != _coordinate_dtype(target):
            raise TypeError("Coordinate transfers require one coordinate dtype.")
        _validate_disjoint(
            tuple((start, start + length) for start, _, length in pieces), source.size
        )
        _validate_disjoint(
            tuple((start, start + length) for _, start, length in pieces), target.size
        )
        self.source = source
        self.target = target
        self.pieces = pieces
        self.properties = OperatorProperties(
            rank=sum(length for _, _, length in pieces),
            evidence={"rank": "construction"},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "coordinate-transfer",
                "source": source.space_id,
                "target": target.space_id,
                "pieces": [list(piece) for piece in pieces],
            }
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.source.flatten(vector)
        placements = tuple(
            (target_start, coordinates[source_start : source_start + length])
            for source_start, target_start, length in self.pieces
        )
        return self.target.unflatten(
            _scatter(placements, self.target.size, _coordinate_dtype(self.target))
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.target.flatten(vector)
        placements = tuple(
            (source_start, coordinates[target_start : target_start + length])
            for source_start, target_start, length in self.pieces
        )
        return self.source.unflatten(
            _scatter(placements, self.source.size, _coordinate_dtype(self.source))
        )

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        matrix = np.zeros(
            (self.target.size, self.source.size), dtype=_coordinate_dtype(self.source)
        )
        for source_start, target_start, length in self.pieces:
            index = np.arange(length, dtype=np.int64)
            matrix[target_start + index, source_start + index] = 1
        return jnp.asarray(matrix)


def _coordinate_identity(
    source: AbstractVectorSpace, target: AbstractVectorSpace, /
) -> AbstractLinearOperator:
    """Identify the complete canonical coordinates of two equal-size spaces."""
    if source.size != target.size:
        raise ValueError("A coordinate identity requires equal space sizes.")
    return _CoordinateTransferLinearOperator(source, target, ((0, 0, source.size),))


class _SelectedBlockLinearOperator(AbstractLinearOperator):
    """Exact ``gather(rows) @ operator @ scatter(columns)`` with identified spaces."""

    operator: AbstractLinearOperator
    row_ranges: CoordinateRanges = eqx.field(static=True)
    column_ranges: CoordinateRanges = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        row_ranges: CoordinateRanges,
        row_space: AbstractVectorSpace,
        column_ranges: CoordinateRanges,
        column_space: AbstractVectorSpace,
        /,
    ) -> None:
        if _range_length(row_ranges) != row_space.size:
            raise ValueError("Selected rows must match the requested row space size.")
        if _range_length(column_ranges) != column_space.size:
            raise ValueError(
                "Selected columns must match the requested column space size."
            )
        self.operator = operator
        self.row_ranges = row_ranges
        self.column_ranges = column_ranges
        self.source = column_space
        self.target = row_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.transpose,
            materialize=operator.capabilities.materialize,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "selected-block",
                "operator": operator.operator_id,
                "rows": [list(value) for value in row_ranges],
                "row_space": row_space.space_id,
                "columns": [list(value) for value in column_ranges],
                "column_space": column_space.space_id,
            }
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        inner = self.operator
        embedded = _scatter_ranges(
            self.source.flatten(vector),
            self.column_ranges,
            inner.source.size,
            _coordinate_dtype(inner.source),
        )
        image = inner.target.flatten(inner.mv(inner.source.unflatten(embedded)))
        return self.target.unflatten(_gather(image, self.row_ranges))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        inner = self.operator
        embedded = _scatter_ranges(
            self.target.flatten(vector),
            self.row_ranges,
            inner.target.size,
            _coordinate_dtype(inner.target),
        )
        image = inner.source.flatten(inner.transpose_mv(inner.target.unflatten(embedded)))
        return self.source.unflatten(_gather(image, self.column_ranges))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        matrix = self.operator._materialize()
        rows = jnp.asarray(_positions(self.row_ranges))
        columns = jnp.asarray(_positions(self.column_ranges))
        return matrix[rows[:, None], columns[None, :]]


class _CoordinateRebasedLinearOperator(AbstractLinearOperator):
    """Canonical coordinate endomorphism for an equal-size structured map.

    The rebased operator shares the canonical coordinates of the wrapped operator's
    source and target, so named block extraction descends through it unchanged.
    """

    operator: AbstractLinearOperator
    coordinate_space: ArraySpace

    @checked
    def __init__(self, operator: AbstractLinearOperator, /) -> None:
        if operator.batch_shape:
            raise ValueError("Coordinate rebasing requires an unbatched operator.")
        if operator.source.size != operator.target.size:
            raise ValueError(
                "Coordinate rebasing requires equal source and target dimensions."
            )
        source_coordinates = operator.source.flatten(operator.source.zeros())
        target_coordinates = operator.target.flatten(operator.target.zeros())
        if source_coordinates.dtype != target_coordinates.dtype:
            raise TypeError(
                "Coordinate rebasing requires matching source and target dtypes."
            )
        coordinate_space = ArraySpace(
            (operator.source.size,),
            dtype=source_coordinates.dtype,
        )
        self.operator = operator
        self.coordinate_space = coordinate_space
        self.source = coordinate_space
        self.target = coordinate_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.adjoint,
            materialize=False,
            diagonal_assembly=operator.capabilities.diagonal_assembly,
        )
        self.batch_shape = ()
        self.operator_id = f"{operator.operator_id}/canonical-coordinate-rebase"

    def flatten_target(self, vector: PyTree[Any], /) -> Array:
        return self.coordinate_space.validate(self.operator.target.flatten(vector))

    def unflatten_source(self, coordinates: Array, /) -> PyTree[Array]:
        value = self.coordinate_space.validate(coordinates)
        return self.operator.source.unflatten(value)

    def mv(self, vector: PyTree[Any], /) -> Array:
        state_direction = self.unflatten_source(vector)
        return self.flatten_target(self.operator.mv(state_direction))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        coordinates = self.coordinate_space.validate(vector)
        residual_direction = self.operator.target.unflatten(coordinates)
        transposed = self.operator.transpose_mv(residual_direction)
        return self.coordinate_space.validate(self.operator.source.flatten(transposed))

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        coordinates = self.coordinate_space.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(coordinates)))

    def _assemble_diagonal(self, /) -> Array:
        return _assemble_operator_diagonal(self.operator)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "A coordinate-rebased Jacobian is matrix-free and cannot materialize."
        )


def _validate_coordinate_map(selection: object, name: str, /) -> BlockSelection:
    if not isinstance(selection, BlockSelection):
        raise TypeError(f"{name} must be a BlockSelection.")
    if not selection.covers_source:
        raise ValueError(f"{name} must identify every source coordinate exactly once.")
    return selection


def _transposed_blocks(operator: AbstractLinearOperator, /) -> AbstractLinearOperator:
    if not isinstance(operator, BlockLinearOperator):
        return TransposeLinearOperator(operator)
    blocks = tuple(
        tuple(_transposed_entry(row[column]) for row in operator.blocks)
        for column in range(len(operator.source.spaces))
    )
    return BlockLinearOperator(blocks, source=operator.target, target=operator.source)


def _transposed_entry(
    block: AbstractLinearOperator | None, /
) -> AbstractLinearOperator | None:
    return None if block is None else _transposed_blocks(block)


def _identity_piece(
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    row_start: int,
    column_start: int,
    weight: Array,
    /,
) -> AbstractLinearOperator | None:
    low = max(row_start, column_start)
    high = min(row_start + target.size, column_start + source.size)
    if low >= high:
        return None
    transfer = _CoordinateTransferLinearOperator(
        source, target, ((low - column_start, low - row_start, high - low),)
    )
    return ScaledLinearOperator(transfer, weight)


def _switched_blocks(
    operator: AbstractLinearOperator,
    active: Array,
    inactive: Array,
    row_start: int,
    column_start: int,
    /,
) -> AbstractLinearOperator:
    """Blend each leaf with the canonical identity piece at the same coordinates."""
    if isinstance(operator, BlockLinearOperator):
        row_offsets = _member_offsets(operator.target)
        column_offsets = _member_offsets(operator.source)
        blocks = []
        for row_index, target in enumerate(operator.target.spaces):
            row: list[AbstractLinearOperator | None] = []
            for column_index, source in enumerate(operator.source.spaces):
                block = operator.blocks[row_index][column_index]
                row_offset = row_start + row_offsets[row_index]
                column_offset = column_start + column_offsets[column_index]
                row.append(
                    _identity_piece(source, target, row_offset, column_offset, inactive)
                    if block is None
                    else _switched_blocks(
                        block, active, inactive, row_offset, column_offset
                    )
                )
            blocks.append(tuple(row))
        return BlockLinearOperator(
            tuple(blocks), source=operator.source, target=operator.target
        )
    weighted = ScaledLinearOperator(operator, active)
    identity = _identity_piece(
        operator.source, operator.target, row_start, column_start, inactive
    )
    return weighted if identity is None else SumLinearOperator(weighted, identity)


class MappedBlockLinearOperator(AbstractLinearOperator):
    """Named block operator acting on native coordinates through explicit maps.

    ``column_map`` identifies the native source coordinates with the block operator
    source; ``row_map`` identifies the native target coordinates with the block
    operator target. The action is ``prolong(rows) @ block @ restrict(columns)``; the
    coordinate transpose applies the reversed maps around the block transpose. The
    Hilbert adjoint is evaluated against the native source and target pairings and is
    never substituted by the transpose.
    """

    block_operator: AbstractLinearOperator
    row_map: BlockSelection
    column_map: BlockSelection
    identity_ordered: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        block_operator: AbstractLinearOperator,
        /,
        *,
        row_map: BlockSelection,
        column_map: BlockSelection,
    ) -> None:
        row_map_ = _validate_coordinate_map(row_map, "row_map")
        column_map_ = _validate_coordinate_map(column_map, "column_map")
        if block_operator.batch_shape:
            raise ValueError("Mapped block operators must be unbatched.")
        if not block_operator.source.compatible(column_map_.target):
            raise ValueError("The block operator source must match the column map.")
        if not block_operator.target.compatible(row_map_.target):
            raise ValueError("The block operator target must match the row map.")
        rank = block_operator.properties.rank
        self.block_operator = block_operator
        self.row_map = row_map_
        self.column_map = column_map_
        self.identity_ordered = _identity_ordered(
            row_map_.member_ranges, row_map_.source.size
        ) and _identity_ordered(column_map_.member_ranges, column_map_.source.size)
        self.source = column_map_.source
        self.target = row_map_.source
        self.properties = OperatorProperties(
            rank=rank if block_operator.properties.certifies("rank") else None,
            evidence=(
                {"rank": "transformed"}
                if block_operator.properties.certifies("rank")
                else {}
            ),
        )
        self.capabilities = OperatorCapabilities(
            transpose=block_operator.capabilities.transpose,
            adjoint=block_operator.capabilities.transpose,
            materialize=block_operator.capabilities.materialize,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "mapped-block",
                "block": block_operator.operator_id,
                "rows": row_map_.selection_id,
                "columns": column_map_.selection_id,
            }
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        blocks = self.column_map.restrict(vector)
        return self.row_map.prolong(self.block_operator.mv(blocks))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        blocks = self.row_map.restrict(vector)
        return self.column_map.prolong(self.block_operator.transpose_mv(blocks))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        matrix = self.block_operator._materialize()
        if self.identity_ordered:
            return matrix
        rows = _positions(
            tuple(value for member in self.row_map.member_ranges for value in member)
        )
        columns = _positions(
            tuple(value for member in self.column_map.member_ranges for value in member)
        )
        placed = jnp.zeros((self.target.size, self.source.size), dtype=matrix.dtype)
        return placed.at[jnp.asarray(rows)[:, None], jnp.asarray(columns)[None, :]].set(
            matrix
        )

    def coordinate_transpose(self) -> MappedBlockLinearOperator:
        """Return the explicit transpose with reversed row and column maps."""
        return MappedBlockLinearOperator(
            _transposed_blocks(self.block_operator),
            row_map=self.column_map,
            column_map=self.row_map,
        )

    def with_inactive_identity(
        self,
        active: ArrayLike,
        /,
        *,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
    ) -> MappedBlockLinearOperator:
        """Return this operator where ``active`` and the coordinate identity elsewhere.

        The blend is applied to every explicit block leaf, so named block structure
        survives an inactive replay root. The maps are rebound to the supplied native
        ``source`` and ``target`` spaces, which must carry this operator's space
        identities; guarded evaluation may have replaced the numeric leaves of the
        spaces retained by this operator. Identity-ordered maps of equal size are
        required.
        """
        if not self.identity_ordered or self.source.size != self.target.size:
            raise ValueError(
                "An inactive identity requires identity-ordered maps of equal size."
            )
        if not source.compatible(self.source) or not target.compatible(self.target):
            raise ValueError("Rebound native spaces must keep the operator identities.")
        flag = jnp.asarray(active)
        if flag.shape != () or flag.dtype != jnp.bool_:
            raise ValueError("active must be one Boolean scalar.")
        real = np.finfo(_coordinate_dtype(self.source)).dtype
        weight = flag.astype(real)
        return MappedBlockLinearOperator(
            _switched_blocks(self.block_operator, weight, 1.0 - weight, 0, 0),
            row_map=_rebound_selection(self.row_map, target),
            column_map=_rebound_selection(self.column_map, source),
        )


def _rebound_selection(
    selection: BlockSelection, source: AbstractVectorSpace, /
) -> BlockSelection:
    """Return the same member spaces and ranges over another source instance."""
    return BlockSelection(
        source,
        tuple(
            (name, CoordinateBlock(space, ranges))
            for name, space, ranges in zip(
                selection.names,
                selection.target.spaces,
                selection.member_ranges,
                strict=True,
            )
        ),
    )


def _member_pieces(
    space: BlockSpace, ranges: CoordinateRanges, /
) -> tuple[tuple[int, CoordinateRanges], ...]:
    """Split canonical ranges at member boundaries, preserving their order."""
    offsets = _member_offsets(space)
    pieces: list[tuple[int, list[CoordinateRange]]] = []
    for start, stop in ranges:
        cursor = start
        while cursor < stop:
            index = next(
                index
                for index, (offset, member) in enumerate(
                    zip(offsets, space.spaces, strict=True)
                )
                if offset <= cursor < offset + member.size
            )
            end = min(stop, offsets[index] + space.spaces[index].size)
            local = (cursor - offsets[index], end - offsets[index])
            if pieces and pieces[-1][0] == index and pieces[-1][1][-1][1] == local[0]:
                pieces[-1][1][-1] = (pieces[-1][1][-1][0], local[1])
            elif pieces and pieces[-1][0] == index:
                pieces[-1][1].append(local)
            else:
                pieces.append((index, [local]))
            cursor = end
    return tuple((index, tuple(local)) for index, local in pieces)


def _node_for_range(
    space: AbstractVectorSpace, span: CoordinateRange, /
) -> AbstractVectorSpace | None:
    """Return the nested block member whose canonical span is exactly ``span``."""
    start, stop = span
    if span == (0, space.size):
        return space
    if not isinstance(space, BlockSpace):
        return None
    for offset, member in zip(_member_offsets(space), space.spaces, strict=True):
        if offset <= start and stop <= offset + member.size:
            return _node_for_range(member, (start - offset, stop - offset))
    return None


def _piece_space(
    space: BlockSpace, index: int, local: CoordinateRanges, /
) -> AbstractVectorSpace:
    member = space.spaces[index]
    node = _node_for_range(member, local[0]) if len(local) == 1 else None
    if node is not None:
        return node
    return ArraySpace(
        (_range_length(local),),
        dtype=_coordinate_dtype(member),
        space_id=canonical_fingerprint(
            {
                "kind": "canonical-block-piece",
                "space": space.space_id,
                "member": space.names[index],
                "ranges": [list(value) for value in local],
            }
        ),
    )


def _reuses_members(
    space: AbstractVectorSpace, pieces: tuple[AbstractVectorSpace, ...], /
) -> bool:
    return (
        isinstance(space, BlockSpace)
        and len(space.spaces) == len(pieces)
        and all(
            member.compatible(piece)
            for member, piece in zip(space.spaces, pieces, strict=True)
        )
    )


def _piece_map(
    space: AbstractVectorSpace, pieces: tuple[AbstractVectorSpace, ...], /
) -> BlockSelection:
    members: list[tuple[str, BlockMember]] = []
    offset = 0
    for index, piece in enumerate(pieces):
        members.append(
            (str(index), CoordinateBlock(piece, ((offset, offset + piece.size),)))
        )
        offset += piece.size
    return BlockSelection(space, members)


def _grid_range_block(
    operator: BlockLinearOperator,
    row_ranges: CoordinateRanges,
    row_space: AbstractVectorSpace,
    column_ranges: CoordinateRanges,
    column_space: AbstractVectorSpace,
    /,
) -> AbstractLinearOperator | None:
    row_pieces = _member_pieces(operator.target, row_ranges)
    column_pieces = _member_pieces(operator.source, column_ranges)
    if len(row_pieces) == 1 and len(column_pieces) == 1:
        (row_index, local_rows), (column_index, local_columns) = (
            row_pieces[0],
            column_pieces[0],
        )
        return _range_entry(
            operator.blocks[row_index][column_index],
            local_rows,
            row_space,
            local_columns,
            column_space,
        )
    row_spaces = tuple(
        _piece_space(operator.target, index, local) for index, local in row_pieces
    )
    column_spaces = tuple(
        _piece_space(operator.source, index, local) for index, local in column_pieces
    )
    grid = tuple(
        tuple(
            _range_entry(
                operator.blocks[row_index][column_index],
                local_rows,
                piece_row,
                local_columns,
                piece_column,
            )
            for (column_index, local_columns), piece_column in zip(
                column_pieces, column_spaces, strict=True
            )
        )
        for (row_index, local_rows), piece_row in zip(row_pieces, row_spaces, strict=True)
    )
    if all(block is None for row in grid for block in row):
        return None
    if _reuses_members(row_space, row_spaces) and _reuses_members(
        column_space, column_spaces
    ):
        if not isinstance(row_space, BlockSpace) or not isinstance(
            column_space, BlockSpace
        ):
            raise RuntimeError("Reused block pieces require BlockSpace members.")
        return BlockLinearOperator(grid, source=column_space, target=row_space)
    target = BlockSpace(row_spaces, names=tuple(str(k) for k in range(len(row_spaces))))
    source = BlockSpace(
        column_spaces, names=tuple(str(k) for k in range(len(column_spaces)))
    )
    return MappedBlockLinearOperator(
        BlockLinearOperator(grid, source=source, target=target),
        row_map=_piece_map(row_space, row_spaces),
        column_map=_piece_map(column_space, column_spaces),
    )


def _range_entry(
    block: AbstractLinearOperator | None,
    row_ranges: CoordinateRanges,
    row_space: AbstractVectorSpace,
    column_ranges: CoordinateRanges,
    column_space: AbstractVectorSpace,
    /,
) -> AbstractLinearOperator | None:
    if block is None:
        return None
    return _range_block(block, row_ranges, row_space, column_ranges, column_space)


def _range_block(
    operator: AbstractLinearOperator,
    row_ranges: CoordinateRanges,
    row_space: AbstractVectorSpace,
    column_ranges: CoordinateRanges,
    column_space: AbstractVectorSpace,
    /,
) -> AbstractLinearOperator | None:
    """Return ``gather(rows) @ operator @ scatter(columns)`` preserving structure."""
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return _range_block(
            operator.operator, row_ranges, row_space, column_ranges, column_space
        )
    if isinstance(operator, MappedBlockLinearOperator) and operator.identity_ordered:
        return _range_block(
            operator.block_operator, row_ranges, row_space, column_ranges, column_space
        )
    if isinstance(operator, BlockLinearOperator):
        return _grid_range_block(
            operator, row_ranges, row_space, column_ranges, column_space
        )
    if (
        row_ranges == ((0, operator.target.size),)
        and column_ranges == ((0, operator.source.size),)
        and row_space.compatible(operator.target)
        and column_space.compatible(operator.source)
    ):
        return operator
    return _SelectedBlockLinearOperator(
        operator, row_ranges, row_space, column_ranges, column_space
    )


def _compression_properties(
    operator: AbstractLinearOperator,
    rows: BlockSelection,
    columns: BlockSelection,
    blocks: tuple[tuple[AbstractLinearOperator | None, ...], ...],
    /,
) -> OperatorProperties | None:
    """Return inherited claims of a principal compression ``R A R*``.

    A path selection keeps the source member pairings, so its prolongation is the
    Hilbert adjoint of its surjective restriction and the compression of a
    self-adjoint (definite) operator is self-adjoint (definite).
    """
    if (
        rows.selection_id != columns.selection_id
        or not rows.preserves_pairing
        or not operator.source.compatible(operator.target)
        or not operator.properties.certifies("self_adjoint")
    ):
        return None
    positive_definite = operator.properties.certifies("positive_definite")
    positive_semidefinite = operator.properties.certifies("positive_semidefinite")
    block_diagonal = all(
        row_index == column_index or block is None
        for row_index, row in enumerate(blocks)
        for column_index, block in enumerate(row)
    )
    claims = {
        "self_adjoint": True,
        "positive_definite": positive_definite,
        "positive_semidefinite": positive_semidefinite,
        "block_diagonal": block_diagonal,
    }
    return OperatorProperties(
        **claims,
        evidence={
            name: "construction" if name == "block_diagonal" else "transformed"
            for name, claimed in claims.items()
            if claimed
        },
    )


def select_block_operator(
    operator: AbstractLinearOperator,
    rows: BlockSelection,
    columns: BlockSelection,
    /,
) -> BlockLinearOperator:
    """Return the exact named block grid ``restrict(rows) @ operator @ prolong(columns)``.

    Explicit structure is preserved: block grids are indexed rather than composed,
    identity-ordered mapped operators and canonical coordinate views are traversed,
    and only opaque leaves are wrapped as exact coordinate selections. Every block
    retains the capabilities of the operator it came from. A principal compression
    by one path selection inherits self-adjoint and definiteness claims.
    """
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not isinstance(rows, BlockSelection) or not isinstance(columns, BlockSelection):
        raise TypeError("rows and columns must be BlockSelection values.")
    if operator.batch_shape:
        raise ValueError("Named block selection requires an unbatched operator.")
    if not rows.source.compatible(operator.target):
        raise ValueError("Row selection source must be the operator target space.")
    if not columns.source.compatible(operator.source):
        raise ValueError("Column selection source must be the operator source space.")
    blocks = tuple(
        tuple(
            _range_block(operator, row_ranges, row_space, column_ranges, column_space)
            for column_space, column_ranges in zip(
                columns.target.spaces, columns.member_ranges, strict=True
            )
        )
        for row_space, row_ranges in zip(
            rows.target.spaces, rows.member_ranges, strict=True
        )
    )
    return BlockLinearOperator(
        blocks,
        source=columns.target,
        target=rows.target,
        properties=_compression_properties(operator, rows, columns, blocks),
    )


type BlockEntry = tuple[BlockPath, BlockPath, AbstractLinearOperator]


def _lifted_entry(
    row_path: tuple[str, ...],
    column_path: tuple[str, ...],
    operator: AbstractLinearOperator,
    row_space: AbstractVectorSpace,
    column_space: AbstractVectorSpace,
    /,
) -> AbstractLinearOperator:
    result = operator
    if column_path:
        node, (start, _) = _path_node(column_space, column_path)
        result = result @ _CoordinateTransferLinearOperator(
            column_space, node, ((start, 0, node.size),)
        )
    if row_path:
        node, (start, _) = _path_node(row_space, row_path)
        result = (
            _CoordinateTransferLinearOperator(node, row_space, ((0, start, node.size),))
            @ result
        )
    return result


def _assembled_grid(
    entries: tuple[BlockEntry, ...], source: BlockSpace, target: BlockSpace, /
) -> BlockLinearOperator:
    grouped: dict[
        tuple[int, int],
        list[tuple[tuple[str, ...], tuple[str, ...], AbstractLinearOperator]],
    ] = {}
    for row_path, column_path, operator in entries:
        key = (target.names.index(row_path[0]), source.names.index(column_path[0]))
        grouped.setdefault(key, []).append((row_path[1:], column_path[1:], operator))
    grid: list[list[AbstractLinearOperator | None]] = [
        [None] * len(source.spaces) for _ in target.spaces
    ]
    for (row_index, column_index), members in grouped.items():
        row_space = target.spaces[row_index]
        column_space = source.spaces[column_index]
        nested = tuple(
            (row_path, column_path, operator)
            for row_path, column_path, operator in members
            if row_path and column_path
        )
        terms = [
            _lifted_entry(row_path, column_path, operator, row_space, column_space)
            for row_path, column_path, operator in members
            if not (row_path and column_path)
        ]
        if nested:
            if not isinstance(row_space, BlockSpace) or not isinstance(
                column_space, BlockSpace
            ):
                raise RuntimeError("Nested block entries require BlockSpace members.")
            terms.append(_assembled_grid(nested, column_space, row_space))
        total = terms[0]
        for term in terms[1:]:
            total = SumLinearOperator(total, term)
        grid[row_index][column_index] = total
    return BlockLinearOperator(
        tuple(tuple(row) for row in grid), source=source, target=target
    )


def assemble_block_operator(
    entries: Sequence[BlockEntry],
    /,
    *,
    source: BlockSpace,
    target: BlockSpace,
) -> BlockLinearOperator:
    """Assemble one explicit block operator from ``(row_path, column_path, operator)``.

    Entries at equal nesting depth become explicit nested grids. An entry whose row
    and column paths end at different depths is embedded exactly with coordinate
    identities. Repeated block addresses are refused; entries landing in one block
    at different depths are summed in declaration order.
    """
    if not isinstance(source, BlockSpace) or not isinstance(target, BlockSpace):
        raise TypeError("source and target must be BlockSpace values.")
    parsed: list[BlockEntry] = []
    seen: set[tuple[BlockPath, BlockPath]] = set()
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError("Block entries must be (row_path, column_path, operator).")
        row_path = _block_path(entry[0])
        column_path = _block_path(entry[1])
        operator = entry[2]
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError(
                "Block entry operators must be AbstractLinearOperator values."
            )
        if operator.batch_shape:
            raise ValueError("Block entry operators must be unbatched.")
        if (row_path, column_path) in seen:
            raise ValueError(f"Block entry {(row_path, column_path)!r} is repeated.")
        row_node, _ = _path_node(target, row_path)
        column_node, _ = _path_node(source, column_path)
        if not operator.target.compatible(row_node):
            raise ValueError(f"Block entry at row {row_path!r} has the wrong target.")
        if not operator.source.compatible(column_node):
            raise ValueError(
                f"Block entry at column {column_path!r} has the wrong source."
            )
        seen.add((row_path, column_path))
        parsed.append((row_path, column_path, operator))
    return _assembled_grid(tuple(parsed), source, target)


__all__ = [
    "assemble_block_operator",
    "BlockPath",
    "BlockProlongationLinearOperator",
    "BlockRestrictionLinearOperator",
    "BlockSelection",
    "CoordinateBlock",
    "MappedBlockLinearOperator",
    "select_block_operator",
]
