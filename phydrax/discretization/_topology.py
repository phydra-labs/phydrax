#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, RowRelation, SparseLinearMap, SparseRelation
from ..typing import checked, ConvertibleToArray
from ._core import nonempty_identifier, resolved_identifier
from .spatial._dyadic import DyadicCellTopology


def _array_digest(value: object, /) -> dict[str, object]:
    return array_tree_fingerprint(value)


def _bool_array(name: str, value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return jnp.asarray(array)


def _column_extremes(rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Per-column int64 minima and maxima of nonempty rank-2 integer rows.

    One strided reduction per column: an axis-0 reduction of narrow row-major
    rows runs NumPy's inner loop over only the row width.
    """

    lows = np.asarray([rows[:, column].min() for column in range(rows.shape[1])])
    highs = np.asarray([rows[:, column].max() for column in range(rows.shape[1])])
    return lows.astype(np.int64), highs.astype(np.int64)


def _exact_spans(lows: np.ndarray, highs: np.ndarray, /) -> np.ndarray | None:
    """Per-column value spans, or ``None`` when their product exceeds int64.

    The spans are formed in Python integers: ``high - low + 1`` wraps in int64
    for columns reaching both a padding sentinel near the int64 maximum and
    small values.
    """

    exact = [int(high) - int(low) + 1 for low, high in zip(lows, highs, strict=True)]
    if math.prod(exact) > np.iinfo(np.int64).max:
        return None
    return np.asarray(exact, dtype=np.int64)


def _row_value_keys(
    rows: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Order-preserving mixed-radix int64 keys of nonempty rank-2 integer rows.

    Returns the keys, column lows and spans and the span product, or ``None``
    when the product of the per-column value spans exceeds int64.
    """

    lows, highs = _column_extremes(rows)
    spans = _exact_spans(lows, highs)
    if spans is None:
        return None
    span_product = math.prod(int(span) for span in spans)
    key = np.zeros((rows.shape[0],), dtype=np.int64)
    for column, low, span in zip(rows.T, lows, spans, strict=True):
        key *= span
        key += column.astype(np.int64) - low
    return key, lows, spans, span_product


def _row_order(rows: np.ndarray, /) -> np.ndarray:
    """Return the stable lexicographic order of rank-2 integer rows.

    Rows whose per-column value spans fit one mixed-radix int64 key sort in a
    single pass: with the row index in the key's low bits when both fit one
    uint64, so a plain value sort of distinct keys yields the stable order,
    else by a stable argsort. Wider rows fall back to a column-wise
    lexicographic sort.
    """

    if rows.size == 0:
        return np.arange(rows.shape[0], dtype=np.int64)
    values = _row_value_keys(rows)
    if values is None:
        return np.lexsort(rows.T[::-1])
    key, _, _, span_product = values
    index_bits = (rows.shape[0] - 1).bit_length()
    largest_packed = ((span_product - 1) << index_bits) | ((1 << index_bits) - 1)
    if largest_packed > np.iinfo(np.uint64).max:
        return np.argsort(key, kind="stable")
    # The key is nonnegative, so its uint64 view orders like the int64 key.
    packed = key.view(np.uint64)
    packed <<= np.uint64(index_bits)
    packed |= np.arange(rows.shape[0], dtype=np.uint64)
    packed.sort()
    packed &= np.uint64((1 << index_bits) - 1)
    return key


def _row_run_starts(ordered: np.ndarray, /) -> np.ndarray:
    starts = np.ones((ordered.shape[0],), dtype=np.bool_)
    np.any(ordered[1:] != ordered[:-1], axis=1, out=starts[1:])
    return starts


def has_repeated_values(values: np.ndarray, /) -> bool:
    """Return whether a one-dimensional integer array repeats any value.

    Strictly increasing arrays (the common identity bank) answer in one pass;
    otherwise one value sort finds equal neighbors.
    """

    if values.size < 2 or np.all(values[1:] > values[:-1]):
        return False
    ordered = np.sort(values)
    return bool(np.any(ordered[1:] == ordered[:-1]))


def _has_duplicate_rows(rows: np.ndarray, /) -> bool:
    """Return whether any two rows of a rank-2 integer array are equal."""

    values = _row_value_keys(rows) if rows.size else None
    if values is not None:
        # Equal rows are exactly equal keys; their order is immaterial.
        key = values[0]
        key.sort()
        return bool(np.any(key[1:] == key[:-1]))
    # Equal rows have equal wrapped polynomial hashes, so distinct hashes
    # prove distinct rows; only a hash collision needs the exact row order.
    hashed = np.zeros((rows.shape[0],), dtype=np.uint64)
    for column in rows.T:
        hashed *= np.uint64(0x9E3779B97F4A7C15)
        hashed += column.astype(np.uint64)
    hashed.sort()
    if not np.any(hashed[1:] == hashed[:-1]):
        return False
    return not np.all(_row_run_starts(rows[_row_order(rows)]))


def sorted_unique_rows(rows: np.ndarray, /) -> np.ndarray:
    """The distinct rows of a rank-2 integer array in lexicographic order.

    Equal to ``np.unique(rows, axis=0)``, through one packed-key sort.
    """

    values = _row_value_keys(rows) if rows.size else None
    if values is None:
        ordered = rows[_row_order(rows)]
        return ordered[_row_run_starts(ordered)] if ordered.shape[0] else ordered
    # Keys order rows lexicographically, so the sorted distinct keys decode to
    # the sorted distinct rows.
    key, lows, spans, _ = values
    key.sort()
    starts = np.ones((key.size,), dtype=np.bool_)
    np.not_equal(key[1:], key[:-1], out=starts[1:])
    key = key[starts]
    result = np.empty((key.size, rows.shape[1]), dtype=rows.dtype)
    for column in range(rows.shape[1] - 1, -1, -1):
        key, digit = np.divmod(key, spans[column])
        result[:, column] = digit + lows[column]
    return result


def sorted_rows(rows: np.ndarray, /) -> np.ndarray:
    """The rows of a rank-2 integer array in lexicographic order, with repeats.

    Equal to ``rows[_row_order(rows)]``: equal rows are indistinguishable, so
    one value sort of the packed keys decodes to the same array.
    """

    values = _row_value_keys(rows) if rows.size else None
    if values is None:
        return rows[_row_order(rows)]
    key, lows, spans, _ = values
    key.sort()
    result = np.empty(rows.shape, dtype=rows.dtype)
    for column in range(rows.shape[1] - 1, -1, -1):
        key, digit = np.divmod(key, spans[column])
        result[:, column] = digit + lows[column]
    return result


def rows_increase(rows: np.ndarray, /, *, strict: bool) -> bool:
    """Whether rank-2 integer rows (strictly) increase lexicographically."""

    if rows.shape[0] < 2:
        return True
    greater = np.zeros((rows.shape[0] - 1,), dtype=np.bool_)
    equal = np.ones((rows.shape[0] - 1,), dtype=np.bool_)
    for column in range(rows.shape[1]):
        upper = rows[1:, column]
        lower = rows[:-1, column]
        greater |= equal & (upper > lower)
        equal &= upper == lower
    return bool(np.all(greater if strict else greater | equal))


def unique_rows_inverse(rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``np.unique(rows, axis=0, return_inverse=True, return_counts=True)``.

    The same lexicographically sorted distinct rows, flat inverse and counts,
    through one packed-key sort.
    """

    order = _row_order(rows)
    ordered = rows[order]
    if not ordered.shape[0]:
        return ordered, np.empty((0,), np.int64), np.empty((0,), np.int64)
    starts = _row_run_starts(ordered)
    groups = np.cumsum(starts, dtype=np.int64) - 1
    inverse = np.empty((rows.shape[0],), dtype=np.int64)
    inverse[order] = groups
    return ordered[starts], inverse, np.bincount(groups)


def _row_keys(
    table: np.ndarray, query: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Order-preserving int64 keys of two nonempty arrays' rows, if spans fit."""

    table_lows, table_highs = _column_extremes(table)
    query_lows, query_highs = _column_extremes(query)
    lows = np.minimum(table_lows, query_lows)
    highs = np.maximum(table_highs, query_highs)
    spans = _exact_spans(lows, highs)
    if spans is None:
        return None
    table_key = np.zeros((table.shape[0],), dtype=np.int64)
    query_key = np.zeros((query.shape[0],), dtype=np.int64)
    for column, low, span in zip(range(table.shape[1]), lows, spans, strict=True):
        table_key *= span
        table_key += table[:, column].astype(np.int64) - low
        query_key *= span
        query_key += query[:, column].astype(np.int64) - low
    return table_key, query_key


def row_positions(table: np.ndarray, query: np.ndarray, /) -> np.ndarray:
    """Row of ``table`` equal to each ``query`` row, or -1 (distinct table rows)."""

    if table.ndim != 2 or query.ndim != 2 or table.shape[1] != query.shape[1]:
        raise ValueError("Row lookup requires rank-2 arrays of equal width.")
    if not table.shape[0] or not query.shape[0]:
        return np.full((query.shape[0],), -1, dtype=np.int64)
    keys = _row_keys(table, query)
    if keys is None:
        _, inverse = np.unique(
            np.concatenate((table, query)), axis=0, return_inverse=True
        )
        inverse = inverse.reshape((-1,))
        owners = np.full((int(inverse.max()) + 1,), -1, dtype=np.int64)
        owners[inverse[: table.shape[0]]] = np.arange(table.shape[0], dtype=np.int64)
        return owners[inverse[table.shape[0] :]]
    table_keys, query_keys = keys
    if np.all(table_keys[1:] > table_keys[:-1]):
        # Lexicographically sorted tables are their own order; a sorted table of
        # consecutive keys locates each key by its offset.
        if table_keys[-1] - table_keys[0] == table_keys.size - 1:
            position = query_keys - table_keys[0]
            return np.where(
                (position >= 0) & (position < table_keys.size), position, -1
            ).astype(np.int64)
        position = np.minimum(
            np.searchsorted(table_keys, query_keys), table_keys.size - 1
        )
        return np.where(table_keys[position] == query_keys, position, -1).astype(np.int64)
    order = np.argsort(table_keys, kind="stable")
    ordered = table_keys[order]
    position = np.minimum(np.searchsorted(ordered, query_keys), ordered.size - 1)
    return np.where(ordered[position] == query_keys, order[position], -1).astype(np.int64)


class EntitySubset(StrictModule, NonTrainableState):
    """Named subset of one fixed-capacity entity set."""

    name: str = eqx.field(static=True)
    mask: Array
    subset_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mask: ArrayLike,
        /,
        *,
        subset_id: str | None = None,
    ) -> None:
        name_ = nonempty_identifier("name", name)
        mask_ = np.asarray(mask, dtype=np.bool_)
        if mask_.ndim != 1:
            raise ValueError("Entity subset masks must be rank-1.")
        self.name = name_
        self.mask = jax.device_put(mask_)
        self.subset_id = resolved_identifier(
            "subset_id",
            subset_id,
            {
                "kind": "entity-subset",
                "name": name_,
                "mask": _array_digest(mask_),
            },
        )


class EntitySet(StrictModule, NonTrainableState):
    """Fixed-capacity, identity-bearing topological entities."""

    name: str = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    count: int = eqx.field(static=True)
    entity_ids: Array
    active_mask: Array
    subsets: tuple[EntitySubset, ...]
    entity_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        intrinsic_dimension: int,
        entity_ids: ConvertibleToArray,
        /,
        *,
        active_mask: ArrayLike | None = None,
        subsets: Sequence[EntitySubset] = (),
        entity_set_id: str | None = None,
    ) -> None:
        name_ = nonempty_identifier("name", name)
        dimension = int(intrinsic_dimension)
        if dimension < 0:
            raise ValueError("intrinsic_dimension must be non-negative.")
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("entity_ids must be one rank-1 integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        count = identifiers.shape[0]
        active = (
            np.ones((count,), dtype=np.bool_)
            if active_mask is None
            else np.asarray(active_mask, dtype=np.bool_)
        )
        if active.shape != (count,):
            raise ValueError(
                f"active_mask must have shape {(count,)}; got {active.shape}."
            )
        active_ids = identifiers[active]
        if np.any(active_ids < 0):
            raise ValueError("Active entity IDs must be non-negative.")
        if has_repeated_values(active_ids):
            raise ValueError("Active entity IDs must be unique.")
        subsets_ = tuple(subsets)
        if not all(isinstance(subset, EntitySubset) for subset in subsets_):
            raise TypeError("subsets must contain EntitySubset values.")
        names = tuple(subset.name for subset in subsets_)
        if len(set(names)) != len(names):
            raise ValueError("Entity subset names must be unique.")
        for subset in subsets_:
            subset_mask = np.asarray(subset.mask, dtype=np.bool_)
            if subset_mask.shape != (count,):
                raise ValueError(
                    f"Subset {subset.name!r} must have shape {(count,)}; got {subset_mask.shape}."
                )
            if np.any(subset_mask & ~active):
                raise ValueError("Entity subsets cannot include inactive entities.")
        self.name = name_
        self.intrinsic_dimension = dimension
        self.count = count
        self.entity_ids = jax.device_put(identifiers)
        self.active_mask = jax.device_put(active)
        self.subsets = subsets_
        self.entity_set_id = resolved_identifier(
            "entity_set_id",
            entity_set_id,
            {
                "kind": "entity-set",
                "name": name_,
                "intrinsic_dimension": dimension,
                "entity_ids": _array_digest(identifiers),
                "active_mask": _array_digest(active),
                "subsets": [subset.subset_id for subset in subsets_],
            },
        )

    @property
    def num_active(self) -> int:
        return int(np.count_nonzero(np.asarray(self.active_mask)))

    def subset(self, name: str, /) -> EntitySubset:
        for subset in self.subsets:
            if subset.name == name:
                return subset
        raise KeyError(f"Unknown entity subset {name!r} on {self.name!r}.")


class EntitySelection(StrictModule, NonTrainableState):
    """Canonical composable selection over one exact entity set."""

    entity_set_id: str = eqx.field(static=True)
    mask: Array
    active_mask: Array
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        entities: EntitySet | str,
        mask: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        selection_id: str | None = None,
    ) -> None:
        if isinstance(entities, EntitySet):
            entity_set_id = entities.entity_set_id
            active = np.asarray(entities.active_mask, dtype=np.bool_)
        else:
            entity_set_id = nonempty_identifier("entity_set_id", entities)
            if active_mask is None:
                raise ValueError(
                    "active_mask is required when constructing from an entity-set ID."
                )
            active = np.asarray(active_mask, dtype=np.bool_)
        mask_ = np.asarray(mask, dtype=np.bool_)
        if active.ndim != 1 or mask_.shape != active.shape:
            raise ValueError(
                "Entity selection and active masks must share one rank-1 shape."
            )
        if np.any(mask_ & ~active):
            raise ValueError("Entity selections cannot include inactive entities.")
        self.entity_set_id = entity_set_id
        self.mask = jax.device_put(mask_)
        self.active_mask = jax.device_put(active)
        self.selection_id = resolved_identifier(
            "selection_id",
            selection_id,
            {
                "kind": "entity-selection",
                "entity_set": entity_set_id,
                "mask": _array_digest(mask_),
            },
        )

    @classmethod
    def from_subset(
        cls,
        entities: EntitySet,
        subset_name: str,
        /,
    ) -> EntitySelection:
        return cls(entities, entities.subset(subset_name).mask)

    @checked
    def _binary(
        self,
        other: EntitySelection,
        operation: str,
        /,
    ) -> EntitySelection:
        if self.entity_set_id != other.entity_set_id:
            raise ValueError("Entity selections must share one entity set.")
        if operation == "union":
            mask = self.mask | other.mask
        elif operation == "intersection":
            mask = self.mask & other.mask
        elif operation == "difference":
            mask = self.mask & ~other.mask
        else:
            raise ValueError("Unknown entity selection operation.")
        return _selection_from_masks(
            self.entity_set_id,
            mask,
            self.active_mask,
            operation,
            (self.selection_id, other.selection_id),
        )

    def union(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "union")

    def intersection(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "intersection")

    def difference(self, other: EntitySelection, /) -> EntitySelection:
        return self._binary(other, "difference")

    def complement(self, /) -> EntitySelection:
        return _selection_from_masks(
            self.entity_set_id,
            self.active_mask & ~self.mask,
            self.active_mask,
            "complement",
            (self.selection_id,),
        )


def _selection_from_masks(
    entity_set_id: str,
    mask: ArrayLike,
    active_mask: ArrayLike,
    operation: str,
    operands: tuple[str, ...],
    /,
) -> EntitySelection:
    return EntitySelection(
        entity_set_id,
        mask,
        active_mask=active_mask,
        selection_id=canonical_fingerprint(
            {
                "kind": "composed-entity-selection",
                "entity_set": entity_set_id,
                "operation": operation,
                "operands": list(operands),
                "mask": _array_digest(mask),
            }
        ),
    )


class OrientedIncidence(StrictModule, NonTrainableState):
    """Sparse signed boundary incidence from lower to upper entities.

    ``canonical_order`` may supply the permutation of the active routes that
    sorts their ``(lower ID, upper ID, sign)`` rows; it is checked.
    """

    degree: int = eqx.field(static=True)
    lower_entity_set_id: str = eqx.field(static=True)
    upper_entity_set_id: str = eqx.field(static=True)
    relation: EdgeRelation
    signs: Array
    incidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        degree: int,
        lower: EntitySet,
        upper: EntitySet,
        relation: EdgeRelation,
        signs: ArrayLike,
        /,
        *,
        incidence_id: str | None = None,
        canonical_order: ArrayLike | None = None,
    ) -> None:
        degree_ = int(degree)
        if degree_ <= 0:
            raise ValueError("Incidence degree must be positive.")
        if lower.intrinsic_dimension != degree_ - 1:
            raise ValueError("Lower entity dimension must equal degree - 1.")
        if upper.intrinsic_dimension != degree_:
            raise ValueError("Upper entity dimension must equal degree.")
        if relation.source_size != lower.count or relation.target_size != upper.count:
            raise ValueError("Incidence relation sizes must match the entity sets.")
        coefficients = np.asarray(signs)
        if coefficients.shape != relation.route_shape:
            raise ValueError(
                f"Incidence signs must have shape {relation.route_shape}; got {coefficients.shape}."
            )
        valid = np.asarray(relation.valid, dtype=np.bool_)
        source = np.asarray(relation.source_indices)
        target = np.asarray(relation.target_indices)
        if np.all(valid):
            active_coefficients = coefficients.reshape((-1,))
            source = source.reshape((-1,))
            target = target.reshape((-1,))
        else:
            active_coefficients = coefficients[valid]
            source = source[valid]
            target = target[valid]
        if not np.all((active_coefficients == 1) | (active_coefficients == -1)):
            if np.any(~np.isfinite(active_coefficients)):
                raise ValueError("Active incidence signs must be finite.")
            raise ValueError("Active incidence signs must be ±1.")
        # Preserve attaching-face multiplicity, including repeated signed routes.
        if canonical_order is not None:
            order = np.asarray(canonical_order)
            count = active_coefficients.shape[0]
            if (
                order.shape != (count,)
                or not np.issubdtype(order.dtype, np.integer)
                or (count and (order.min() < 0 or order.max() >= count))
            ):
                raise ValueError("canonical_order must index the active incidence rows.")
            seen = np.zeros((count,), dtype=np.bool_)
            seen[order] = True
            if not np.all(seen):
                raise ValueError(
                    "canonical_order must permute the active incidence rows."
                )
            source = source[order]
            target = target[order]
            active_coefficients = active_coefficients[order]
        canonical_incidence = np.stack(
            (
                np.asarray(lower.entity_ids, dtype=np.int64)[source],
                np.asarray(upper.entity_ids, dtype=np.int64)[target],
                active_coefficients.astype(np.int64),
            ),
            axis=1,
        )
        if canonical_order is None:
            canonical_incidence = sorted_rows(canonical_incidence)
        elif not rows_increase(canonical_incidence, strict=False):
            raise ValueError("canonical_order must sort the canonical incidence rows.")
        self.degree = degree_
        self.lower_entity_set_id = lower.entity_set_id
        self.upper_entity_set_id = upper.entity_set_id
        self.relation = relation
        self.signs = jax.device_put(coefficients.astype(np.float64, copy=False))
        self.incidence_id = resolved_identifier(
            "incidence_id",
            incidence_id,
            {
                "kind": "oriented-incidence",
                "degree": degree_,
                "lower": lower.entity_set_id,
                "upper": upper.entity_set_id,
                "canonical_incidence": _array_digest(canonical_incidence),
            },
        )

    def exterior_derivative(self, /) -> SparseLinearMap:
        """Return the lower-to-upper transpose-boundary action."""
        return SparseLinearMap(
            self.relation,
            self.signs,
            operator_id=canonical_fingerprint(
                {"kind": "exterior-derivative", "incidence": self.incidence_id}
            ),
        )

    def boundary(self, /) -> SparseLinearMap:
        """Return the upper-to-lower boundary action."""
        return SparseLinearMap(
            self.relation.transpose(),
            self.signs,
            operator_id=canonical_fingerprint(
                {"kind": "boundary", "incidence": self.incidence_id}
            ),
        )

    def scipy_boundary(self, /) -> sp.csr_matrix:
        """Return a host-side sparse lower-by-upper boundary matrix."""
        valid = np.asarray(self.relation.valid, dtype=np.bool_)
        return sp.coo_matrix(
            (
                np.asarray(self.signs)[valid],
                (
                    np.asarray(self.relation.source_indices)[valid],
                    np.asarray(self.relation.target_indices)[valid],
                ),
            ),
            shape=(self.relation.source_size, self.relation.target_size),
        ).tocsr()


class TensorTopology(StrictModule, NonTrainableState):
    """Implicit tensor-product topology without materialized graph routes."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_sizes: tuple[int, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    active_mask: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        axis_sizes: Sequence[int],
        /,
        *,
        periodic: Sequence[bool] | None = None,
        active_mask: ArrayLike | None = None,
        topology_id: str | None = None,
    ) -> None:
        names = tuple(str(name) for name in axis_names)
        sizes = tuple(axis_sizes)
        if not names or any(not name for name in names):
            raise ValueError("Tensor topology requires non-empty axis names.")
        if len(set(names)) != len(names):
            raise ValueError("Tensor topology axis names must be unique.")
        if len(sizes) != len(names) or any(size <= 0 for size in sizes):
            raise ValueError("Tensor topology requires one positive size per axis.")
        periodic_ = (
            (False,) * len(names)
            if periodic is None
            else tuple(bool(value) for value in periodic)
        )
        if len(periodic_) != len(names):
            raise ValueError("periodic must provide one value per axis.")
        active = (
            np.ones(sizes, dtype=np.bool_)
            if active_mask is None
            else np.asarray(active_mask, dtype=np.bool_)
        )
        if active.shape != sizes:
            raise ValueError(f"active_mask must have shape {sizes}; got {active.shape}.")
        if not np.any(active):
            raise ValueError("Tensor topology requires at least one active site.")
        self.axis_names = names
        self.axis_sizes = sizes
        self.periodic = periodic_
        self.active_mask = jnp.asarray(active)
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "tensor-topology",
                "axis_names": list(names),
                "axis_sizes": list(sizes),
                "periodic": list(periodic_),
                "active_mask": _array_digest(active),
            },
        )


class CellComplexTopology(StrictModule, NonTrainableState):
    """Validated oriented finite cell complex with one entity set per degree."""

    entity_sets: tuple[EntitySet, ...]
    incidences: tuple[OrientedIncidence, ...]
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_sets: Sequence[EntitySet],
        incidences: Sequence[OrientedIncidence],
        /,
        *,
        topology_id: str | None = None,
        validate: bool = True,
    ) -> None:
        entities = tuple(entity_sets)
        if not entities or not all(isinstance(value, EntitySet) for value in entities):
            raise TypeError("entity_sets must contain one or more EntitySet values.")
        dimensions = tuple(value.intrinsic_dimension for value in entities)
        if dimensions != tuple(range(len(entities))):
            raise ValueError(
                "Cell-complex entity sets must cover contiguous dimensions from 0."
            )
        incidences_ = tuple(incidences)
        if len(incidences_) != len(entities) - 1:
            raise ValueError("One incidence is required between each consecutive degree.")
        for degree, incidence in enumerate(incidences_, start=1):
            if not isinstance(incidence, OrientedIncidence):
                raise TypeError("incidences must contain OrientedIncidence values.")
            if incidence.degree != degree:
                raise ValueError("Incidence degrees must be contiguous and ordered.")
            if (
                incidence.lower_entity_set_id != entities[degree - 1].entity_set_id
                or incidence.upper_entity_set_id != entities[degree].entity_set_id
            ):
                raise ValueError(
                    "Incidence endpoints must match consecutive entity sets."
                )
        if validate:
            self._validate_chain(incidences_)
        self.entity_sets = entities
        self.incidences = incidences_
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "cell-complex-topology",
                "entity_sets": [value.entity_set_id for value in entities],
                "incidences": [value.incidence_id for value in incidences_],
            },
        )

    @staticmethod
    def _validate_chain(incidences: tuple[OrientedIncidence, ...], /) -> None:
        for lower, upper in zip(incidences[:-1], incidences[1:], strict=True):
            product = lower.scipy_boundary() @ upper.scipy_boundary()
            product.eliminate_zeros()
            if product.nnz:
                raise ValueError(
                    "Cell-complex incidences violate boundary-of-boundary zero."
                )

    @property
    def dimension(self) -> int:
        return len(self.entity_sets) - 1

    def entities(self, degree: int, /) -> EntitySet:
        index = int(degree)
        if index < 0 or index >= len(self.entity_sets):
            raise ValueError(f"degree must lie in [0, {self.dimension}].")
        return self.entity_sets[index]


class PointTopology(StrictModule, NonTrainableState):
    """Point support with an optional fixed-capacity neighborhood relation."""

    points: EntitySet
    neighborhoods: SparseRelation | None
    refreshable_neighborhoods: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: EntitySet,
        /,
        *,
        neighborhoods: SparseRelation | None = None,
        refreshable_neighborhoods: bool = False,
        topology_id: str | None = None,
    ) -> None:
        if not isinstance(points, EntitySet) or points.intrinsic_dimension != 0:
            raise TypeError("points must be a zero-dimensional EntitySet.")
        if neighborhoods is not None:
            if not isinstance(neighborhoods, (EdgeRelation, RowRelation)):
                raise TypeError(
                    "neighborhoods must be an EdgeRelation, RowRelation, or None."
                )
            if neighborhoods.source_size != points.count:
                raise ValueError("Neighborhood sources must index the point entity set.")
            if (
                isinstance(neighborhoods, EdgeRelation)
                and neighborhoods.target_size != points.count
            ):
                raise ValueError(
                    "Edge neighborhoods must target the same point entity set."
                )
        neighborhood_payload = None
        if neighborhoods is not None:
            neighborhood_payload = {
                "source": _array_digest(neighborhoods.source_indices),
                "valid": _array_digest(neighborhoods.valid),
                "source_size": neighborhoods.source_size,
                "output_shape": list(neighborhoods.output_shape),
            }
            if isinstance(neighborhoods, EdgeRelation):
                neighborhood_payload["target"] = _array_digest(
                    neighborhoods.target_indices
                )
        self.points = points
        self.neighborhoods = neighborhoods
        self.refreshable_neighborhoods = bool(refreshable_neighborhoods)
        self.topology_id = resolved_identifier(
            "topology_id",
            topology_id,
            {
                "kind": "point-topology",
                "points": points.entity_set_id,
                "neighborhoods": neighborhood_payload,
                "refreshable": bool(refreshable_neighborhoods),
            },
        )


DiscreteTopology: TypeAlias = (
    TensorTopology | CellComplexTopology | PointTopology | DyadicCellTopology
)


__all__ = [
    "CellComplexTopology",
    "DiscreteTopology",
    "DyadicCellTopology",
    "EntitySet",
    "EntitySubset",
    "OrientedIncidence",
    "PointTopology",
    "TensorTopology",
]
