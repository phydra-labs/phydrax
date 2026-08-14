#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._relation import EdgeRelation


SparsePatternOrigin: TypeAlias = Literal["declared", "structural", "asdex"]
_PATTERN_SCHEMA_VERSION = 1


class SparsePattern(StrictModule, NonTrainableState):
    """Canonical structural nonzero pattern for one coordinate matrix."""

    relation: EdgeRelation
    symmetric: bool = eqx.field(static=True)
    origin: SparsePatternOrigin = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        /,
        *,
        symmetric: bool = False,
        origin: SparsePatternOrigin = "declared",
    ):
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        if origin not in ("declared", "structural", "asdex"):
            raise ValueError(f"Unknown sparse-pattern origin {origin!r}.")

        source = np.asarray(relation.source_indices, dtype=np.int64)
        target = np.asarray(relation.target_indices, dtype=np.int64)
        valid = np.asarray(relation.valid, dtype=bool)
        source = source[valid]
        target = target[valid]
        if source.size:
            coordinates = np.stack((target, source), axis=1)
            coordinates = np.unique(coordinates, axis=0)
            rows = coordinates[:, 0]
            cols = coordinates[:, 1]
        else:
            rows = np.empty((0,), dtype=np.int64)
            cols = np.empty((0,), dtype=np.int64)

        symmetric_ = bool(symmetric)
        if symmetric_:
            if relation.source_size != relation.target_size:
                raise ValueError("A symmetric sparse pattern must be square.")
            entries = set(zip(rows.tolist(), cols.tolist(), strict=True))
            if any((column, row) not in entries for row, column in entries):
                raise ValueError(
                    "A symmetric sparse pattern must explicitly contain every transpose entry."
                )

        canonical_relation = EdgeRelation(
            jnp.asarray(cols, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=relation.source_size,
            target_size=relation.target_size,
        )
        payload = {
            "kind": "sparse-pattern",
            "shape": [relation.target_size, relation.source_size],
            "rows": rows.tolist(),
            "cols": cols.tolist(),
            "symmetric": symmetric_,
            "origin": origin,
        }
        self.relation = canonical_relation
        self.symmetric = symmetric_
        self.origin = origin
        self.pattern_id = canonical_fingerprint(payload)

    @classmethod
    def from_coo(
        cls,
        rows: ArrayLike | Sequence[int],
        cols: ArrayLike | Sequence[int],
        shape: tuple[int, int],
        /,
        *,
        symmetric: bool = False,
        origin: SparsePatternOrigin = "declared",
    ) -> "SparsePattern":
        """Construct a canonical pattern from matrix row and column coordinates."""

        if len(shape) != 2:
            raise ValueError("shape must contain exactly two dimensions.")
        target_size, source_size = (int(size) for size in shape)
        if target_size < 0 or source_size < 0:
            raise ValueError("Sparse-pattern dimensions must be non-negative.")
        row_array = np.asarray(rows)
        col_array = np.asarray(cols)
        if row_array.ndim != 1 or col_array.ndim != 1:
            raise ValueError("Sparse-pattern rows and columns must be rank-1.")
        if row_array.shape != col_array.shape:
            raise ValueError("Sparse-pattern rows and columns must have equal shape.")
        if row_array.size and not np.issubdtype(row_array.dtype, np.integer):
            raise TypeError("Sparse-pattern rows must have an integer dtype.")
        if col_array.size and not np.issubdtype(col_array.dtype, np.integer):
            raise TypeError("Sparse-pattern columns must have an integer dtype.")
        if np.any((row_array < 0) | (row_array >= target_size)):
            raise ValueError("Sparse-pattern rows lie outside the declared shape.")
        if np.any((col_array < 0) | (col_array >= source_size)):
            raise ValueError("Sparse-pattern columns lie outside the declared shape.")
        index_maximum = np.iinfo(np.int32).max
        if np.any(row_array > index_maximum) or np.any(col_array > index_maximum):
            raise ValueError("Sparse-pattern coordinates must fit in int32.")
        relation = EdgeRelation(
            jnp.asarray(col_array, dtype=jnp.int32),
            jnp.asarray(row_array, dtype=jnp.int32),
            source_size=source_size,
            target_size=target_size,
        )
        return cls(relation, symmetric=symmetric, origin=origin)

    @property
    def rows(self) -> Array:
        return self.relation.target_indices

    @property
    def cols(self) -> Array:
        return self.relation.source_indices

    @property
    def shape(self) -> tuple[int, int]:
        return (self.relation.target_size, self.relation.source_size)

    @property
    def target_size(self) -> int:
        return self.relation.target_size

    @property
    def source_size(self) -> int:
        return self.relation.source_size

    @property
    def nnz(self) -> int:
        return self.relation.capacity

    @property
    def density(self) -> float:
        total = self.target_size * self.source_size
        return self.nnz / total if total else 0.0

    def to_dict(self, /) -> dict[str, Any]:
        """Return a versioned JSON-compatible structural artifact."""

        return {
            "schema_version": _PATTERN_SCHEMA_VERSION,
            "shape": list(self.shape),
            "rows": np.asarray(self.rows).tolist(),
            "cols": np.asarray(self.cols).tolist(),
            "symmetric": self.symmetric,
            "origin": self.origin,
            "pattern_id": self.pattern_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "SparsePattern":
        """Restore and validate one structural artifact."""

        if not isinstance(value, Mapping):
            raise TypeError("Serialized sparse pattern must be a mapping.")
        expected = {
            "schema_version",
            "shape",
            "rows",
            "cols",
            "symmetric",
            "origin",
            "pattern_id",
        }
        fields = set(value)
        if fields != expected:
            missing = sorted(expected - fields)
            unknown = sorted(fields - expected)
            raise ValueError(
                f"Invalid sparse-pattern fields; missing={missing}, unknown={unknown}."
            )
        schema_version = value["schema_version"]
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != _PATTERN_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported sparse-pattern schema {schema_version!r}.")
        shape_value = value["shape"]
        if (
            not isinstance(shape_value, list)
            or len(shape_value) != 2
            or any(
                not isinstance(size, int) or isinstance(size, bool)
                for size in shape_value
            )
        ):
            raise ValueError(
                "Serialized sparse-pattern shape must contain two integer dimensions."
            )
        for name in ("rows", "cols"):
            indices = value[name]
            if not isinstance(indices, list) or any(
                not isinstance(index, int) or isinstance(index, bool) for index in indices
            ):
                raise ValueError(
                    f"Serialized sparse-pattern {name} must be a list of integers."
                )
        if not isinstance(value["symmetric"], bool):
            raise ValueError("Serialized sparse-pattern symmetric must be boolean.")
        if not isinstance(value["origin"], str):
            raise ValueError("Serialized sparse-pattern origin must be a string.")
        if not isinstance(value["pattern_id"], str):
            raise ValueError("Serialized sparse-pattern pattern_id must be a string.")
        pattern = cls.from_coo(
            value["rows"],
            value["cols"],
            (shape_value[0], shape_value[1]),
            symmetric=value["symmetric"],
            origin=value["origin"],
        )
        if value["pattern_id"] != pattern.pattern_id:
            raise ValueError(
                "Serialized sparse-pattern fingerprint does not match its data."
            )
        return pattern


__all__ = ["SparsePattern", "SparsePatternOrigin"]
