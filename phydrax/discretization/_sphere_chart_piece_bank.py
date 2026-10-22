#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical array storage for exact sphere-chart overlap rows.

The bank is an artifact, not a proof. Every read authenticates all public array
bytes. Only immutable, owned bytes and the exact atoms decoded from those bytes
are cached; neither a scientific verdict nor a caller's mutable row is cached.
"""

from __future__ import annotations

import hashlib
import operator
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import cast, overload, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._strict import mark_strict_initialized, StrictModule
from .._trainable import NonTrainableState
from ..geometry._sphere_material_atlas import (
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..linalg._hermitian_spectral import fraction_storage_upper
from ._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage


if TYPE_CHECKING:
    from ._sphere_chart_records import _Triangle, PreparedSphereChartPiece


# All integers have an explicit little-endian representation. Atoms store a
# sign byte followed by minimal unsigned numerator and denominator magnitudes;
# their three offsets are (start, numerator end, denominator end).
_ARRAY_SCHEMA = (
    ("atom_bytes", "u1", 1, ()),
    ("atom_offsets", "<i8", 2, (3,)),
    ("text_bytes", "u1", 1, ()),
    ("text_offsets", "<i8", 1, ()),
    ("occurrence_rows", "<i8", 2, (3,)),
    ("occurrence_paths", "<i8", 1, ()),
    ("map_rows", "<i8", 2, (19,)),
    ("map_representatives", "<f8", 4, (3, 3, 3)),
    ("bound_rows", "<i8", 2, (4,)),
    ("piece_rows", "<i8", 2, (30,)),
    ("boundary_labels", "<i8", 1, ()),
    ("reference_vertices", "<f8", 4, (2, 3, 2)),
    ("displacement_bounds", "<f8", 1, ()),
)


@dataclass(frozen=True, slots=True)
class _PieceBankAtoms:
    """Owned immutable numerical inputs; deliberately contains no proof token."""

    digest: str
    buffers: tuple[bytes, ...]
    shapes: tuple[tuple[int, ...], ...]
    atoms: tuple[Fraction, ...]
    texts: tuple[str, ...]

    def array(self, index: int, /) -> np.ndarray:
        return np.frombuffer(self.buffers[index], dtype=_ARRAY_SCHEMA[index][1]).reshape(
            self.shapes[index]
        )


# The weak owner is not retained by its entry. The derived cache is never an
# Equinox field, so cold recipes contain only the canonical public arrays.
_ATOMS: dict[int, tuple[weakref.ReferenceType[SphereChartPieceBank], _PieceBankAtoms]] = {}


def _digest(buffers: tuple[bytes, ...], shapes: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256(b"sphere-chart-piece-bank-v1\0")
    for (name, dtype, _, _), shape, data in zip(
        _ARRAY_SCHEMA, shapes, buffers, strict=True
    ):
        digest.update(name.encode("ascii") + b"\0" + dtype.encode("ascii") + b"\0")
        digest.update(len(shape).to_bytes(8, "little"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "little"))
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _invalid() -> ValueError:
    return ValueError(
        "sphere_chart.piece_bank must retain its canonical exact arrays and identity; "
        "got malformed or changed sphere piece-bank bytes."
    )


def _decode(
    digest: str,
    buffers: tuple[bytes, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> _PieceBankAtoms:
    arrays = tuple(
        np.frombuffer(data, dtype=schema[1]).reshape(shape)
        for schema, shape, data in zip(_ARRAY_SCHEMA, shapes, buffers, strict=True)
    )
    atom_data, offsets, text_data, text_offsets = arrays[:4]
    atoms: list[Fraction] = []
    end = 0
    for start, middle, stop in offsets:
        start, middle, stop = int(start), int(middle), int(stop)
        if start != end or middle <= start + 1 or stop <= middle or stop > atom_data.size:
            raise _invalid()
        sign = int(atom_data[start])
        numerator_bytes = buffers[0][start + 1 : middle]
        denominator_bytes = buffers[0][middle:stop]
        if (
            sign not in (0, 1)
            or (len(numerator_bytes) > 1 and numerator_bytes[0] == 0)
            or (len(denominator_bytes) > 1 and denominator_bytes[0] == 0)
        ):
            raise _invalid()
        numerator = int.from_bytes(numerator_bytes, "big")
        denominator = int.from_bytes(denominator_bytes, "big")
        if not denominator or (sign and not numerator):
            raise _invalid()
        numerator = -numerator if sign else numerator
        atom = Fraction(numerator, denominator)
        if atom.numerator != numerator or atom.denominator != denominator:
            raise _invalid()
        atoms.append(atom)
        end = stop
    if end != atom_data.size or len(set(atoms)) != len(atoms):
        raise _invalid()
    if (
        not text_offsets.size
        or int(text_offsets[0]) != 0
        or int(text_offsets[-1]) != text_data.size
        or np.any(text_offsets[1:] < text_offsets[:-1])
    ):
        raise _invalid()
    try:
        texts = tuple(
            buffers[2][int(start) : int(stop)].decode("utf-8")
            for start, stop in zip(text_offsets[:-1], text_offsets[1:], strict=True)
        )
    except UnicodeDecodeError as error:
        raise _invalid() from error
    if len(set(texts)) != len(texts):
        raise _invalid()

    def indices(values: np.ndarray, size: int) -> None:
        if values.size and (np.any(values < 0) or np.any(values >= size)):
            raise _invalid()

    occurrences, paths, maps, map_values, bounds, rows, labels, vertices, displacement = arrays[4:]
    indices(paths, len(texts))
    indices(occurrences[:, 0], len(texts))
    path_end = 0
    for _, start, count in occurrences:
        if int(start) != path_end or count < 0 or start + count > paths.size:
            raise _invalid()
        path_end = int(start + count)
    if path_end != paths.size:
        raise _invalid()
    indices(maps[:, :2], len(texts))
    indices(maps[:, 4], len(occurrences))
    indices(maps[:, 5], len(texts))
    indices(maps[:, 6:], len(atoms))
    indices(bounds, len(atoms))
    indices(rows[:, 6], len(occurrences))
    indices(rows[:, 7], len(maps))
    indices(rows[:, 8], len(bounds))
    indices(rows[:, 9:23], len(atoms))
    indices(rows[:, 28:30], len(texts))
    if (
        map_values.shape[0] != maps.shape[0]
        or vertices.shape[0] != rows.shape[0]
        or displacement.shape[0] != rows.shape[0]
        or np.any(rows[:, :6] < 0)
        or np.any(rows[:, 25:28] < 0)
        or not np.all(np.isfinite(map_values))
        or not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(displacement))
        or np.any(displacement < 0)
    ):
        raise _invalid()
    occurrence_keys = {
        (int(entity), tuple(int(value) for value in paths[int(start) : int(start + count)]))
        for entity, start, count in occurrences
    }
    map_keys = {
        (tuple(int(value) for value in row), values.tobytes())
        for row, values in zip(maps, map_values, strict=True)
    }
    if (
        len(occurrence_keys) != len(occurrences)
        or len(map_keys) != len(maps)
        or len({tuple(int(value) for value in row) for row in bounds}) != len(bounds)
    ):
        raise _invalid()
    label_end = 0
    for row in rows:
        start, count = int(row[23]), int(row[24])
        if start != label_end or count < 0 or start + count > labels.size:
            raise _invalid()
        label_end = start + count
        mapping = maps[int(row[7])]
        # Map identity is not merely a matrix: it names the actual endpoint
        # cells and the unmerged scientific occurrence of this row.
        if tuple(row[:2]) != tuple(mapping[2:4]) or row[6] != mapping[4]:
            raise _invalid()
    if label_end != labels.size:
        raise _invalid()
    return _PieceBankAtoms(digest, buffers, shapes, tuple(atoms), texts)


class SphereChartPieceBank(StrictModule, NonTrainableState):
    """Array-artifact sequence of canonical exact sphere overlap pieces.

    ``authenticate`` and ``iter_pieces`` charge the explicit request ledger.
    Python sequence access is fully authenticated artifact introspection, not a
    scientific proof and not an implicit independently funded request.
    """

    atom_bytes: NDArray[np.uint8]
    atom_offsets: NDArray[np.int64]
    text_bytes: NDArray[np.uint8]
    text_offsets: NDArray[np.int64]
    occurrence_rows: NDArray[np.int64]
    occurrence_paths: NDArray[np.int64]
    map_rows: NDArray[np.int64]
    map_representatives: NDArray[np.float64]
    bound_rows: NDArray[np.int64]
    piece_rows: NDArray[np.int64]
    boundary_labels: NDArray[np.int64]
    reference_vertices: NDArray[np.float64]
    displacement_bounds: NDArray[np.float64]
    bank_id: str = eqx.field(static=True)

    @classmethod
    def from_pieces(
        cls,
        pieces: Iterable[PreparedSphereChartPiece],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SphereChartPieceBank:
        """Intern an input stream without retaining its logical row objects."""
        builder = SphereChartPieceBankBuilder(budget=budget)
        try:
            for piece in pieces:
                builder.append(piece)
            return builder.finish()
        finally:
            builder.close()

    @classmethod
    def from_stored_fields(
        cls,
        value: SphereChartPieceBank,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SphereChartPieceBank:
        """Restore numerical storage, without inventing scientific evidence."""
        value._atoms(budget)
        return value

    @property
    def retained_bytes_upper(self) -> int:
        """Bound public arrays and the private immutable numerical atom cache."""
        atoms = self._atoms(None)
        return (
            2 * sum(len(buffer) for buffer in atoms.buffers)
            + 4 * len(atoms.buffers[0])
            + 4 * len(atoms.buffers[2])
            + 512 * len(atoms.atoms)
            + 128 * len(atoms.texts)
            + 8192
        )

    def _snapshot(
        self, budget: CoordinateEnclosureBudget | None
    ) -> tuple[tuple[bytes, ...], tuple[tuple[int, ...], ...], str]:
        arrays: list[np.ndarray] = []
        for name, dtype, ndim, tail in _ARRAY_SCHEMA:
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(dtype)
                or value.ndim != ndim
                or (tail and value.shape[1:] != tail)
            ):
                raise _invalid()
            arrays.append(value)
        size = sum(value.nbytes for value in arrays)
        if budget is not None:
            # Copy and SHA-256 both visit the complete stream. A work unit is
            # one 64-byte block visit, plus each array's framing visit.
            budget.reserve(2 * ((size + 63) // 64) + len(arrays), size + 4096)
        buffers = tuple(value.tobytes(order="C") for value in arrays)
        shapes = tuple(value.shape for value in arrays)
        digest = _digest(buffers, shapes)
        if digest != self.bank_id:
            raise _invalid()
        return buffers, shapes, digest

    def _atoms(self, budget: CoordinateEnclosureBudget | None) -> _PieceBankAtoms:
        scope = nullcontext() if budget is None else budget.temporary_scope()
        with scope:
            buffers, shapes, digest = self._snapshot(budget)
            entry = _ATOMS.get(id(self))
            candidate = (
                entry[1]
                if entry is not None and entry[0]() is self and entry[1].digest == digest
                else None
            )
            # Artifact introspection cannot prime a new request with free
            # exact decoding. A ledger adopts/decodes its numerical inputs
            # once; later accesses still copy and authenticate every byte.
            if candidate is not None and (
                budget is None
                or budget.retained_objects.get(id(candidate)) is candidate
            ):
                result = candidate
            else:
                if budget is not None:
                    size = sum(len(buffer) for buffer in buffers)
                    index_count = sum(
                        len(buffer) // 8
                        for schema, buffer in zip(_ARRAY_SCHEMA, buffers, strict=True)
                        if schema[1] == "<i8"
                    )
                    budget.reserve(
                        index_count + (size + 63) // 64,
                        5 * size + 512 * shapes[1][0] + 128 * shapes[3][0] + 4096,
                    )
                result = _decode(digest, buffers, shapes)
                identifier = id(self)
                _ATOMS[identifier] = (
                    weakref.ref(self, lambda _owner: _ATOMS.pop(identifier, None)),
                    result,
                )
            if budget is not None and budget.retained_objects.get(id(result)) is not result:
                # This is retained numerical input storage, not a cached verdict.
                budget.retain_basis((result,))
        return result

    def authenticate(self, *, budget: CoordinateEnclosureBudget) -> None:
        """Authenticate every mutable public byte before derived-atom reuse."""
        self._atoms(budget)

    def iter_pieces(
        self, *, budget: CoordinateEnclosureBudget
    ) -> Iterator[PreparedSphereChartPiece]:
        """Read one fully authenticated immutable snapshot on the caller ledger."""
        atoms = self._atoms(budget)
        return _iter_pieces(atoms, budget)

    def _iter_kernel_pieces(
        self, *, budget: CoordinateEnclosureBudget
    ) -> Iterator[PreparedSphereChartPiece]:
        """Borrow private numerical operands; never publish rows or their atoms.

        Only internal consumers whose outputs contain no borrowed Fraction or
        row may use this path. Public row access always makes independent atoms.
        Authentication still covers every current stored byte on every access.
        """
        return _iter_pieces(self._atoms(budget), budget, kernel=True)

    def _iter_kernel_source_areas(
        self, *, budget: CoordinateEnclosureBudget
    ) -> Iterator[tuple[int, Fraction]]:
        """Borrow source IDs/areas without allocating unused numerical fields."""
        atoms = self._atoms(budget)
        rows = atoms.array(9)
        budget.reserve(3 * rows.shape[0], 512)
        return (
            (int(row[0]), atoms.atoms[int(row[21])]) for row in rows
        )

    def metadata_rows(
        self, *, budget: CoordinateEnclosureBudget
    ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Read IDs and occurrences without constructing numerical row objects."""
        atoms = self._atoms(budget)
        occurrences, paths, rows = atoms.array(4), atoms.array(5), atoms.array(9)
        budget.reserve(
            3 * occurrences.shape[0] + paths.size + 3 * rows.shape[0],
            128 * (occurrences.shape[0] + paths.size + rows.shape[0]) + 4096,
        )
        groups = tuple(
            (
                atoms.texts[int(entity)],
                tuple(
                    atoms.texts[int(value)]
                    for value in paths[int(start) : int(start + count)]
                ),
            )
            for entity, start, count in occurrences
        )
        return tuple(
            (atoms.texts[int(row[29])], *groups[int(row[6])]) for row in rows
        )

    def __len__(self) -> int:
        # Shape access authenticates bytes but never decodes exact atoms.
        return self._snapshot(None)[1][9][0]

    def __iter__(self) -> Iterator[PreparedSphereChartPiece]:
        atoms = self._atoms(None)
        return (_piece(atoms, index) for index in range(atoms.shapes[9][0]))

    @overload
    def __getitem__(self, index: int) -> PreparedSphereChartPiece: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PreparedSphereChartPiece, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PreparedSphereChartPiece | tuple[PreparedSphereChartPiece, ...]:
        atoms = self._atoms(None)
        count = atoms.shapes[9][0]
        if isinstance(index, slice):
            return tuple(_piece(atoms, item) for item in range(*index.indices(count)))
        item = operator.index(index)
        if item < 0:
            item += count
        if not 0 <= item < count:
            raise IndexError("sphere chart piece-bank index out of range")
        return _piece(atoms, item)


def _iter_pieces(
    atoms: _PieceBankAtoms, budget: CoordinateEnclosureBudget, *, kernel: bool = False
) -> Iterator[PreparedSphereChartPiece]:
    storage = CoordinateLiveStorage(budget)
    storage.set_bound(2 * 8192)
    try:
        for index in range(atoms.shapes[9][0]):
            # Do not leave a temporary_scope suspended across yield: that would
            # release a consumer's still-live exact image/proof allocations.
            budget.reserve(49)
            yield _piece(atoms, index, storage=storage, kernel=kernel)
    finally:
        storage.close()


def _piece(
    data: _PieceBankAtoms,
    index: int,
    *,
    storage: CoordinateLiveStorage | None = None,
    kernel: bool = False,
) -> PreparedSphereChartPiece:
    from ._sphere_chart_records import PreparedSphereChartPiece

    row = data.array(9)[index]
    texts = data.texts

    def occurrence(index: int) -> tuple[str, tuple[str, ...]]:
        entity, start, count = (int(value) for value in data.array(4)[index])
        return texts[entity], tuple(
            texts[int(value)] for value in data.array(5)[start : start + count]
        )

    mapping = data.array(6)[int(row[7])]
    bound_row = data.array(8)[int(row[8])]
    atoms: tuple[Fraction, ...] | dict[int, Fraction]
    if kernel:
        # These atoms were constructed privately from the fully authenticated
        # byte snapshot; no caller/decoded Fraction object is trusted. Borrowing
        # them inside kernels avoids repeated public-row normalizations.
        atoms = data.atoms
    else:
        # Fraction's private slots are writable in CPython. Never return a
        # retained atom through public row access.
        references = {
            int(value) for values in (mapping[6:], bound_row, row[9:23]) for value in values
        }
        bits = max(
            (
                max(
                    abs(data.atoms[atom].numerator).bit_length(),
                    data.atoms[atom].denominator.bit_length(),
                    1,
                )
                for atom in references
            ),
            default=1,
        )
        if storage is not None:
            storage.set_bound(
                2 * (8192 + fraction_storage_upper(len(references), bits)),
                work=3 * len(references),
            )
        atoms = {
            atom: Fraction(data.atoms[atom].numerator, data.atoms[atom].denominator)
            for atom in references
        }
    map_entity, map_path = occurrence(int(mapping[4]))
    exact = tuple(atoms[int(value)] for value in mapping[6:])
    map_values = data.array(7)[int(row[7])]
    # Like the artifact substrate, reconstruct stored fields, not a constructor
    # which would silently prepare a new map and earn a different proof/ID.
    projective = object.__new__(SphereProjectiveReferenceMap)
    for name, value in (
        ("source_atlas_id", texts[int(mapping[0])]),
        ("target_atlas_id", texts[int(mapping[1])]),
        ("source_cell_global_id", int(mapping[2])),
        ("target_cell_global_id", int(mapping[3])),
        ("geometry_entity_id", map_entity),
        ("occurrence_path", map_path),
        ("exact_coefficients", tuple(tuple(exact[3 * r : 3 * r + 3]) for r in range(3))),
        ("exact_denominator", exact[9:12]),
        ("orientation_ratio", exact[12]),
        ("coefficients", cast(Array, map_values[0])),
        ("coefficient_lower", cast(Array, map_values[1])),
        ("coefficient_upper", cast(Array, map_values[2])),
        ("map_id", texts[int(mapping[5])]),
    ):
        object.__setattr__(projective, name, value)
    mark_strict_initialized(projective)
    vertices = tuple(atoms[int(value)] for value in row[9:21])
    triangles = tuple(
        tuple(tuple(vertices[start + 2 * r : start + 2 * r + 2]) for r in range(3))
        for start in (0, 6)
    )
    bounds = SphereProjectiveTriangleBounds(
        *(atoms[int(value)] for value in bound_row)
    )
    entity, path = occurrence(int(row[6]))
    representatives = data.array(11)[index]
    start, count = int(row[23]), int(row[24])
    return PreparedSphereChartPiece(
        source_cell_global_id=int(row[0]),
        target_cell_global_id=int(row[1]),
        source_row=int(row[2]),
        target_row=int(row[3]),
        source_atlas_row=int(row[4]),
        target_atlas_row=int(row[5]),
        geometry_entity_id=entity,
        occurrence_path=path,
        projective_map=projective,
        exact_source_reference_vertices=cast("_Triangle", triangles[0]),
        exact_target_reference_vertices=cast("_Triangle", triangles[1]),
        source_reference_vertices=cast(Array, representatives[0]),
        target_reference_vertices=cast(Array, representatives[1]),
        projective_bounds=bounds,
        source_reference_area=atoms[int(row[21])],
        target_reference_area=atoms[int(row[22])],
        native_boundary_labels=tuple(int(value) for value in data.array(10)[start : start + count]),
        native_polygon_vertex_indices=(int(row[25]), int(row[26]), int(row[27])),
        native_clip_id=texts[int(row[28])],
        displacement_bound=float(data.array(12)[index]),
        piece_id=texts[int(row[29])],
    )


@dataclass(frozen=True, slots=True)
class _SphereKernelMap:
    """Builder-local canonical map operands, never a scientific proof."""

    owner: weakref.ReferenceType[SphereChartPieceBankBuilder]
    index: int
    occurrence: int
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class _SphereKernelPoint:
    """Owned point operands interned only within one live canonical builder.

    ``exact`` is suitable for the existing exact-point identity payload. Neither
    it nor the canonical float64 bytes retains a caller-owned Fraction.
    """

    owner: weakref.ReferenceType[SphereChartPieceBankBuilder]
    index: int
    atoms: tuple[int, int]
    exact: tuple[tuple[int, int], tuple[int, int]]
    representatives: bytes


class SphereChartPieceBankBuilder:
    """Incrementally own only interned atoms/metadata and compact numerical rows."""

    def __init__(
        self,
        *,
        budget: CoordinateEnclosureBudget,
        maximum_memory_bytes: int | None = None,
    ) -> None:
        if maximum_memory_bytes is not None and (
            type(maximum_memory_bytes) is not int or maximum_memory_bytes < 0
        ):
            raise ValueError("sphere chart piece-bank memory limit must be nonnegative")
        self._budget = budget
        self._maximum_memory_bytes = maximum_memory_bytes
        self._storage = CoordinateLiveStorage(budget)
        self._closed = False
        self._temporary_bytes = 0
        self._check_memory(4096)
        self._storage.set_bound(4096)
        self._atom_index: dict[tuple[int, int], int] = {}
        self._atom_bytes = bytearray()
        self._atom_offsets: list[tuple[int, int, int]] = []
        self._text_index: dict[str, int] = {}
        self._text_bytes = bytearray()
        self._text_offsets = [0]
        self._occurrence_index: dict[tuple[int, tuple[int, ...]], int] = {}
        self._occurrences: list[tuple[int, int, int]] = []
        self._paths: list[int] = []
        self._map_index: dict[tuple[tuple[int, ...], bytes], int] = {}
        self._maps: list[tuple[int, ...]] = []
        self._map_values: list[bytes] = []
        self._bound_index: dict[tuple[int, ...], int] = {}
        self._bounds: list[tuple[int, ...]] = []
        self._kernel_maps: dict[int, _SphereKernelMap] = {}
        self._point_index: dict[tuple[int, int], _SphereKernelPoint] = {}
        self._points: list[_SphereKernelPoint] = []
        # Ordered numerical chunks support both single public rows and whole
        # private fans without retaining a Python 30-field record per triangle.
        self._rows: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._vertices: list[np.ndarray] = []
        self._displacement: list[np.ndarray] = []
        self._row_count = 0
        self._label_count = 0

    def _check_memory(self, additional: int) -> None:
        limit = self._maximum_memory_bytes
        total = self._storage.bound + self._temporary_bytes + additional
        if limit is not None and total > limit:
            raise ValueError(
                "sphere_chart.piece_bank.storage must fit its admitted preparation "
                f"memory; got {total} > {limit} bytes"
            )

    def _grow(self, size: int, *, work: int = 0) -> None:
        self._check_memory(size + 8192)
        self._storage.set_bound(self._storage.bound + size, work=work)

    @contextmanager
    def _fan_temporaries(self, size: int) -> Iterator[None]:
        # Interning can grow while fan buffers are live. Keep that coexisting
        # storage visible to every nested admission against the caller's cap.
        self._check_memory(size)
        self._temporary_bytes += size
        try:
            with self._budget.temporary_scope():
                yield
        finally:
            self._temporary_bytes -= size

    def set_memory_limit(self, maximum_memory_bytes: int) -> None:
        """Tighten the live cap as the caller's other arrangement storage grows."""
        if type(maximum_memory_bytes) is not int or maximum_memory_bytes < 0:
            raise ValueError("sphere chart piece-bank memory limit must be nonnegative")
        self._maximum_memory_bytes = maximum_memory_bytes
        self._check_memory(0)

    def _atom(self, value: Fraction) -> int:
        if not isinstance(value, Fraction):
            raise _invalid()
        # Snapshot immutable integers, never retain a caller-owned Fraction key:
        # its CPython private numerator/denominator slots can be reassigned.
        key = (value.numerator, value.denominator)
        if type(key[0]) is not int or type(key[1]) is not int or key[1] <= 0:
            raise _invalid()
        return self._atom_pair(key)

    def _atom_pair(self, key: tuple[int, int]) -> int:
        """Intern an already inspected immutable numerator/denominator pair."""
        bits = abs(key[0]).bit_length() + key[1].bit_length()
        self._budget.reserve(2 + (bits + 63) // 64)
        existing = self._atom_index.get(key)
        if existing is not None:
            return existing
        numerator = abs(key[0])
        numerator_size = max(1, (numerator.bit_length() + 7) // 8)
        denominator_size = max(1, (key[1].bit_length() + 7) // 8)
        size = 1 + numerator_size + denominator_size
        self._grow(512 + 4 * size, work=1 + (size + 63) // 64)
        start = len(self._atom_bytes)
        self._atom_bytes.append(int(key[0] < 0))
        self._atom_bytes.extend(numerator.to_bytes(numerator_size, "big"))
        middle = len(self._atom_bytes)
        self._atom_bytes.extend(key[1].to_bytes(denominator_size, "big"))
        index = len(self._atom_offsets)
        self._atom_offsets.append((start, middle, len(self._atom_bytes)))
        self._atom_index[key] = index
        return index

    def _text(self, text: str) -> int:
        if not isinstance(text, str):
            raise _invalid()
        self._budget.reserve(1 + (len(text) + 15) // 16)
        existing = self._text_index.get(text)
        if existing is not None:
            return existing
        self._grow(256 + 8 * len(text))
        encoded = text.encode("utf-8")
        index = len(self._text_offsets) - 1
        self._text_bytes.extend(encoded)
        self._text_offsets.append(len(self._text_bytes))
        self._text_index[text] = index
        return index

    def _occurrence(self, entity: str, path: tuple[str, ...]) -> int:
        key = (self._text(entity), tuple(self._text(item) for item in path))
        existing = self._occurrence_index.get(key)
        if existing is not None:
            return existing
        self._grow(512 + 64 * len(path), work=1)
        index = len(self._occurrences)
        self._occurrences.append((key[0], len(self._paths), len(key[1])))
        self._paths.extend(key[1])
        self._occurrence_index[key] = index
        return index

    @property
    def retained_bytes_upper(self) -> int:
        """Live builder storage, including its actual interning dictionaries."""
        return self._storage.bound

    def __len__(self) -> int:
        return self._row_count

    def append(self, piece: PreparedSphereChartPiece) -> None:
        self._append_piece(piece, self._prepare_kernel_map(piece.projective_map))

    def _prepare_kernel_map(
        self, mapping: SphereProjectiveReferenceMap
    ) -> _SphereKernelMap:
        """Snapshot an actual map once before its private fan is published."""
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        self._check_memory(8192)
        with self._budget.temporary_scope():
            self._budget.reserve(32, 8192)
            if (
                len(mapping.exact_coefficients) != 3
                or any(len(row) != 3 for row in mapping.exact_coefficients)
                or len(mapping.exact_denominator) != 3
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not -(2**63) <= value < 2**63
                    for value in (
                        mapping.source_cell_global_id, mapping.target_cell_global_id
                    )
                )
            ):
                raise _invalid()
            occurrence = self._occurrence(mapping.geometry_entity_id, mapping.occurrence_path)
            map_atoms = tuple(
                self._atom(value)
                for value in (
                    *(value for row in mapping.exact_coefficients for value in row),
                    *mapping.exact_denominator,
                    mapping.orientation_ratio,
                )
            )
            map_row = (
                self._text(mapping.source_atlas_id), self._text(mapping.target_atlas_id),
                mapping.source_cell_global_id, mapping.target_cell_global_id,
                occurrence, self._text(mapping.map_id), *map_atoms,
            )
            values = np.asarray(
                (mapping.coefficients, mapping.coefficient_lower, mapping.coefficient_upper),
                dtype="<f8",
            )
            if values.shape != (3, 3, 3) or not np.all(np.isfinite(values)):
                raise _invalid()
            key = (map_row, values.tobytes())
            map_index = self._map_index.get(key)
            if map_index is None:
                self._grow(2048 + len(key[1]), work=1)
                map_index = len(self._maps)
                self._map_index[key] = map_index
                self._maps.append(map_row)
                self._map_values.append(key[1])
            # Cover the token while the caller consumes the entire fan, not
            # merely the temporary snapshot's lexical allocation scope.
            self._grow(256)
            token = _SphereKernelMap(
                weakref.ref(self), map_index, occurrence,
                mapping.source_cell_global_id, mapping.target_cell_global_id,
            )
            self._kernel_maps[id(token)] = token
            return token

    def _prepare_kernel_points(
        self, points: Iterable[tuple[Fraction, Fraction] | _SphereKernelPoint]
    ) -> tuple[_SphereKernelPoint, ...]:
        """Snapshot actual polygon points once, before publishing their fan.

        The returned tuple is builder-owned until ``close``. Each distinct atom
        pair owns one token and one canonical float representation, even when a
        point recurs in many triangles or polygons. Public row ingestion does
        not use this private numerical-operand path.
        """
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        self._grow(128, work=1)
        prepared: list[_SphereKernelPoint] = []
        for point in points:
            # Include the returned polygon's reference slots, even on an
            # intern hit. They coexist with the token/index banks through fan
            # publication, rather than living in a released temporary scope.
            self._grow(32, work=4)
            if isinstance(point, _SphereKernelPoint):
                if (
                    point.owner() is not self
                    or not 0 <= point.index < len(self._points)
                    or self._points[point.index] is not point
                ):
                    raise _invalid()
                prepared.append(point)
                continue
            self._check_memory(8192)
            with self._budget.temporary_scope():
                self._budget.reserve(0, 8192)
                if len(point) != 2 or any(
                    not isinstance(value, Fraction) for value in point
                ):
                    raise _invalid()
                exact = tuple((value.numerator, value.denominator) for value in point)
                if any(
                    type(numerator) is not int
                    or type(denominator) is not int
                    or denominator <= 0
                    for numerator, denominator in exact
                ):
                    raise _invalid()
                atoms = cast(
                    tuple[int, int], tuple(self._atom_pair(pair) for pair in exact)
                )
                existing = self._point_index.get(atoms)
                if existing is None:
                    bits = max(
                        max(abs(numerator).bit_length(), denominator.bit_length(), 1)
                        for numerator, denominator in exact
                    )
                    temporary = fraction_storage_upper(2, bits) + 1024
                    self._check_memory(8192 + temporary)
                    self._budget.reserve(2 * (2 + (2 * bits + 63) // 64), temporary)
                    # Round fresh privately reconstructed operands, never the
                    # mutable Fraction instances supplied by the producer.
                    representatives = np.asarray(
                        [float(Fraction(*pair)) for pair in exact], dtype="<f8"
                    )
                    if not np.all(np.isfinite(representatives)):
                        raise _invalid()
                    self._grow(768, work=1)
                    existing = _SphereKernelPoint(
                        weakref.ref(self), len(self._points), atoms,
                        cast(tuple[tuple[int, int], tuple[int, int]], exact),
                        representatives.tobytes(),
                    )
                    self._points.append(existing)
                    self._point_index[atoms] = existing
                prepared.append(existing)
        return tuple(prepared)

    def _append_kernel_fan(
        self,
        mapping: _SphereKernelMap,
        source_points: tuple[_SphereKernelPoint, ...],
        target_points: tuple[_SphereKernelPoint, ...],
        *,
        source_cell_global_id: int,
        target_cell_global_id: int,
        source_row: int,
        target_row: int,
        source_atlas_row: int,
        target_atlas_row: int,
        geometry_entity_id: str,
        occurrence_path: tuple[str, ...],
        projective_bounds: tuple[SphereProjectiveTriangleBounds, ...],
        source_reference_areas: tuple[Fraction, ...],
        target_reference_areas: tuple[Fraction, ...],
        native_boundary_labels: tuple[int, ...],
        native_clip_id: str,
        displacement_bounds: tuple[float, ...],
        piece_ids: tuple[str, ...],
    ) -> None:
        """Append one complete ordered fan from privately owned polygon inputs.

        Triangle ``slot - 1`` uses polygon indices ``(0, slot, slot + 1)``.
        Bounds, areas, displacement and IDs remain actual per-triangle inputs;
        the map and each polygon point must be live registered builder tokens.
        Fixed metadata is inspected once, then broadcast into numerical buffers.
        This is numerical construction only, not a retained scientific verdict.
        """
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        count = len(source_points)
        triangles = count - 2
        labels = len(native_boundary_labels)
        if (
            count < 3
            or len(target_points) != count
            or len(projective_bounds) != triangles
            or len(source_reference_areas) != triangles
            or len(target_reference_areas) != triangles
            or len(displacement_bounds) != triangles
            or len(piece_ids) != triangles
        ):
            raise _invalid()
        # Owned output chunks coexist with polygon/gather temporaries. Include
        # NumPy headers, chunk-list slots and Python loop operands before any
        # buffers are allocated. Array writes/copies use the existing 64-byte
        # block convention; Python scalar inspection remains explicitly charged.
        output_size = triangles * (30 * 8 + 12 * 8 + 8 + labels * 8)
        temporary_size = 8192 + 64 * count + 256 * triangles + 8 * labels
        self._grow(2048 + output_size)
        with self._fan_temporaries(temporary_size):
            self._budget.reserve(
                16 + 6 + 2 * labels + 12 * count + 2 * triangles
                + (2 * output_size + 128 * count + 72 * triangles + 63) // 64,
                temporary_size,
            )
            fixed = (
                source_cell_global_id, target_cell_global_id,
                source_row, target_row, source_atlas_row, target_atlas_row,
            )
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value < 2**63
                for value in fixed
            ) or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not -(2**63) <= value < 2**63
                for value in native_boundary_labels
            ):
                raise _invalid()
            occurrence = self._occurrence(geometry_entity_id, occurrence_path)
            if (
                not isinstance(mapping, _SphereKernelMap)
                or self._kernel_maps.get(id(mapping)) is not mapping
                or mapping.owner() is not self
                or mapping.source != source_cell_global_id
                or mapping.target != target_cell_global_id
                or occurrence != mapping.occurrence
            ):
                raise _invalid()
            # One owner/registry visit per polygon position, not six visits
            # for every output triangle. No exposed exact Fraction is authority.
            for polygon in (source_points, target_points):
                for point in polygon:
                    if (
                        not isinstance(point, _SphereKernelPoint)
                        or point.owner() is not self
                        or not 0 <= point.index < len(self._points)
                        or self._points[point.index] is not point
                    ):
                        raise _invalid()
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0
                for value in displacement_bounds
            ):
                raise _invalid()
            clip = self._text(native_clip_id)
            rows = np.empty((triangles, 30), dtype="<i8")
            rows[:, :6] = fixed
            rows[:, 6] = occurrence
            rows[:, 7] = mapping.index
            rows[:, 23] = self._label_count + labels * np.arange(triangles, dtype="<i8")
            rows[:, 24] = labels
            rows[:, 25] = 0
            rows[:, 26] = np.arange(1, count - 1, dtype="<i8")
            rows[:, 27] = np.arange(2, count, dtype="<i8")
            rows[:, 28] = clip
            vertices = np.empty((triangles, 2, 3, 2), dtype="<f8")
            for side, polygon in enumerate((source_points, target_points)):
                atoms = np.asarray([point.atoms for point in polygon], dtype="<i8")
                representatives = np.frombuffer(
                    b"".join(point.representatives for point in polygon), dtype="<f8"
                ).reshape(count, 2)
                start = 9 + 6 * side
                rows[:, start : start + 2] = atoms[0]
                rows[:, start + 2 : start + 4] = atoms[1:-1]
                rows[:, start + 4 : start + 6] = atoms[2:]
                vertices[:, side, 0] = representatives[0]
                vertices[:, side, 1] = representatives[1:-1]
                vertices[:, side, 2] = representatives[2:]
            # These visits cannot be lifted: every triangle owns its own exact
            # bounds, areas and logical piece identity.
            self._budget.reserve(9 * triangles)
            for index in range(triangles):
                bound = projective_bounds[index]
                bound_row = tuple(self._atom(value) for value in (
                    bound.denominator_lower, bound.denominator_upper,
                    bound.jacobian_lower, bound.jacobian_upper,
                ))
                bound_index = self._bound_index.get(bound_row)
                if bound_index is None:
                    self._grow(512, work=1)
                    bound_index = len(self._bounds)
                    self._bound_index[bound_row] = bound_index
                    self._bounds.append(bound_row)
                rows[index, 8] = bound_index
                rows[index, 21] = self._atom(source_reference_areas[index])
                rows[index, 22] = self._atom(target_reference_areas[index])
                rows[index, 29] = self._text(piece_ids[index])
            label_values = np.tile(
                np.asarray(native_boundary_labels, dtype="<i8"), triangles
            )
            displacement = np.asarray(displacement_bounds, dtype="<f8")
            self._rows.append(rows)
            self._labels.append(label_values)
            self._vertices.append(vertices)
            self._displacement.append(displacement)
            self._row_count += triangles
            self._label_count += labels * triangles

    def _store_row(
        self,
        row: tuple[int, ...],
        labels: tuple[int, ...],
        vertices: bytes,
        displacement: float,
    ) -> None:
        """Keep legacy single-row ingestion in the same ordered chunk banks."""
        self._budget.reserve(43 + len(labels))
        self._rows.append(np.asarray(row, dtype="<i8").reshape(1, 30))
        self._labels.append(np.asarray(labels, dtype="<i8"))
        self._vertices.append(
            np.frombuffer(vertices, dtype="<f8").reshape(1, 2, 3, 2)
        )
        self._displacement.append(np.asarray([displacement], dtype="<f8"))
        self._row_count += 1
        self._label_count += len(labels)

    def _append_kernel_piece(
        self,
        mapping: _SphereKernelMap,
        source_points: tuple[_SphereKernelPoint, _SphereKernelPoint, _SphereKernelPoint],
        target_points: tuple[_SphereKernelPoint, _SphereKernelPoint, _SphereKernelPoint],
        *,
        source_cell_global_id: int,
        target_cell_global_id: int,
        source_row: int,
        target_row: int,
        source_atlas_row: int,
        target_atlas_row: int,
        geometry_entity_id: str,
        occurrence_path: tuple[str, ...],
        projective_bounds: SphereProjectiveTriangleBounds,
        source_reference_area: Fraction,
        target_reference_area: Fraction,
        native_boundary_labels: tuple[int, ...],
        native_polygon_vertex_indices: tuple[int, int, int],
        native_clip_id: str,
        displacement_bound: float,
        piece_id: str,
    ) -> None:
        """Publish a private fan row from earned map and point operands.

        All scalar metadata, areas and bounds remain actual row inputs and are
        inspected/interned exactly as on public append. Only polygon point
        inspection, large-integer hashing and float rounding have already been
        performed by this builder; no mutable logical row supplies authority.
        """
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        self._check_memory(8192)
        with self._budget.temporary_scope():
            self._budget.reserve(32, 8192)
            occurrence = self._occurrence(geometry_entity_id, occurrence_path)
            integer_values = (
                source_cell_global_id, target_cell_global_id, source_row, target_row,
                source_atlas_row, target_atlas_row, *native_polygon_vertex_indices,
                *native_boundary_labels,
            )
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not -(2**63) <= value < 2**63
                for value in integer_values
            ):
                raise _invalid()
            if (
                not isinstance(mapping, _SphereKernelMap)
                or mapping.owner() is not self
                or mapping.source != source_cell_global_id
                or mapping.target != target_cell_global_id
                or occurrence != mapping.occurrence
                or len(source_points) != 3
                or len(target_points) != 3
                or len(native_polygon_vertex_indices) != 3
            ):
                raise _invalid()
            points = (*source_points, *target_points)
            if any(
                not isinstance(point, _SphereKernelPoint)
                or point.owner() is not self
                or not 0 <= point.index < len(self._points)
                or self._points[point.index] is not point
                for point in points
            ):
                raise _invalid()
            bound = projective_bounds
            bound_row = tuple(self._atom(value) for value in (
                bound.denominator_lower, bound.denominator_upper,
                bound.jacobian_lower, bound.jacobian_upper,
            ))
            bound_index = self._bound_index.get(bound_row)
            if bound_index is None:
                self._grow(512, work=1)
                bound_index = len(self._bounds)
                self._bound_index[bound_row] = bound_index
                self._bounds.append(bound_row)
            vertices = tuple(atom for point in points for atom in point.atoms)
            source_area = self._atom(source_reference_area)
            target_area = self._atom(target_reference_area)
            clip, identity = self._text(native_clip_id), self._text(piece_id)
            self._grow(2048 + 64 * len(native_boundary_labels), work=1)
            self._store_row((
                source_cell_global_id, target_cell_global_id,
                source_row, target_row, source_atlas_row, target_atlas_row,
                occurrence, mapping.index, bound_index, *vertices, source_area, target_area,
                self._label_count, len(native_boundary_labels),
                *native_polygon_vertex_indices, clip, identity,
            ), native_boundary_labels,
                b"".join(point.representatives for point in points), displacement_bound)

    def _append_piece(
        self, piece: PreparedSphereChartPiece, mapping: _SphereKernelMap
    ) -> None:
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        self._check_memory(8192)
        # Admit the transient logical row conversion before allocating arrays.
        with self._budget.temporary_scope():
            self._budget.reserve(32, 8192)
            occurrence = self._occurrence(piece.geometry_entity_id, piece.occurrence_path)
            integer_values = (
                piece.source_cell_global_id, piece.target_cell_global_id,
                piece.source_row, piece.target_row, piece.source_atlas_row,
                piece.target_atlas_row, *piece.native_polygon_vertex_indices,
                *piece.native_boundary_labels,
            )
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not -(2**63) <= value < 2**63
                for value in integer_values
            ):
                raise _invalid()
            if (
                any(
                    len(triangle) != 3 or any(len(point) != 2 for point in triangle)
                    for triangle in (
                        piece.exact_source_reference_vertices,
                        piece.exact_target_reference_vertices,
                    )
                )
            ):
                raise _invalid()
            if (
                mapping.owner() is not self
                or mapping.source != piece.source_cell_global_id
                or mapping.target != piece.target_cell_global_id
                or occurrence != mapping.occurrence
            ):
                raise _invalid()
            map_index = mapping.index
            bound = piece.projective_bounds
            bound_row = tuple(self._atom(value) for value in (
                bound.denominator_lower, bound.denominator_upper,
                bound.jacobian_lower, bound.jacobian_upper,
            ))
            bound_index = self._bound_index.get(bound_row)
            if bound_index is None:
                self._grow(512, work=1)
                bound_index = len(self._bounds)
                self._bound_index[bound_row] = bound_index
                self._bounds.append(bound_row)
            vertices = tuple(self._atom(value) for triangle in (
                piece.exact_source_reference_vertices, piece.exact_target_reference_vertices
            ) for point in triangle for value in point)
            if len(vertices) != 12 or len(piece.native_polygon_vertex_indices) != 3:
                raise _invalid()
            representatives = np.asarray(
                (piece.source_reference_vertices, piece.target_reference_vertices), dtype="<f8"
            )
            if representatives.shape != (2, 3, 2) or not np.all(np.isfinite(representatives)):
                raise _invalid()
            source_area, target_area = self._atom(piece.source_reference_area), self._atom(piece.target_reference_area)
            clip, identity = self._text(piece.native_clip_id), self._text(piece.piece_id)
            self._grow(2048 + 64 * len(piece.native_boundary_labels), work=1)
            self._store_row((
                piece.source_cell_global_id, piece.target_cell_global_id,
                piece.source_row, piece.target_row, piece.source_atlas_row, piece.target_atlas_row,
                occurrence, map_index, bound_index, *vertices, source_area, target_area,
                self._label_count, len(piece.native_boundary_labels),
                *piece.native_polygon_vertex_indices, clip, identity,
            ), piece.native_boundary_labels, representatives.tobytes(), piece.displacement_bound)

    def finish(self) -> SphereChartPieceBank:
        if self._closed:
            raise RuntimeError("sphere chart piece-bank builder is closed")
        output_size = (
            len(self._atom_bytes) + 24 * len(self._atom_offsets)
            + len(self._text_bytes) + 8 * len(self._text_offsets)
            + 24 * len(self._occurrences) + 8 * len(self._paths)
            + (19 * 8 + 3 * 3 * 3 * 8) * len(self._maps)
            + 32 * len(self._bounds) + (30 * 8 + 2 * 3 * 2 * 8 + 8) * self._row_count
            + 8 * self._label_count
        )
        # The builder, output arrays, byte snapshots, exact decoded atoms and
        # retention traversal coexist at publication; enforce the caller's
        # narrower preparation cap before any of those allocations.
        self._check_memory(
            10 * output_size + 512 * len(self._atom_offsets)
            + 128 * len(self._text_offsets) + 32768
        )
        with self._budget.temporary_scope():
            self._budget.reserve((output_size + 63) // 64, 3 * output_size + 8192)
            arrays = (
                np.frombuffer(bytes(self._atom_bytes), dtype="u1").copy(),
                np.asarray(self._atom_offsets, dtype="<i8").reshape(-1, 3),
                np.frombuffer(bytes(self._text_bytes), dtype="u1").copy(),
                np.asarray(self._text_offsets, dtype="<i8"),
                np.asarray(self._occurrences, dtype="<i8").reshape(-1, 3),
                np.asarray(self._paths, dtype="<i8"),
                np.asarray(self._maps, dtype="<i8").reshape(-1, 19),
                np.frombuffer(b"".join(self._map_values), dtype="<f8").copy().reshape(-1, 3, 3, 3),
                np.asarray(self._bounds, dtype="<i8").reshape(-1, 4),
                np.concatenate(self._rows) if self._rows else np.empty((0, 30), dtype="<i8"),
                np.concatenate(self._labels) if self._labels else np.empty(0, dtype="<i8"),
                np.concatenate(self._vertices) if self._vertices else np.empty((0, 2, 3, 2), dtype="<f8"),
                np.concatenate(self._displacement) if self._displacement else np.empty(0, dtype="<f8"),
            )
            buffers = tuple(array.tobytes() for array in arrays)
            shapes = tuple(array.shape for array in arrays)
            identity = _digest(buffers, shapes)
            bank = SphereChartPieceBank(**dict(zip(
                (schema[0] for schema in _ARRAY_SCHEMA), arrays, strict=True
            )), bank_id=identity)
            bank.authenticate(budget=self._budget)
            self._budget.retain_basis((bank,))
        self.close()
        return bank

    def close(self) -> None:
        """Release builder-only interning storage (also safe after a refusal)."""
        if not self._closed:
            self._atom_index.clear()
            self._atom_bytes.clear()
            self._atom_offsets.clear()
            self._text_index.clear()
            self._text_bytes.clear()
            self._text_offsets.clear()
            self._occurrence_index.clear()
            self._occurrences.clear()
            self._paths.clear()
            self._map_index.clear()
            self._maps.clear()
            self._map_values.clear()
            self._bound_index.clear()
            self._bounds.clear()
            self._kernel_maps.clear()
            self._point_index.clear()
            self._points.clear()
            self._rows.clear()
            self._labels.clear()
            self._vertices.clear()
            self._displacement.clear()
            self._row_count = 0
            self._label_count = 0
            self._storage.close()
            self._closed = True
