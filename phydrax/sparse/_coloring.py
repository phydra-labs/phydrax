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
from ._pattern import SparsePattern


SparseDerivativeKind: TypeAlias = Literal["jacobian", "hessian"]
SparseJacobianMode: TypeAlias = Literal["fwd", "rev"]
SparseHessianMode: TypeAlias = Literal[
    "fwd_over_rev",
    "rev_over_fwd",
    "rev_over_rev",
]
SparseDerivativeMode: TypeAlias = SparseJacobianMode | SparseHessianMode
SparseDerivativeCompiler: TypeAlias = Literal["auto", "native", "asdex"]
SparseColoringCompiler: TypeAlias = Literal["native", "asdex"]
_COLORING_SCHEMA_VERSION = 1
_JACOBIAN_MODES = ("fwd", "rev")
_HESSIAN_MODES = ("fwd_over_rev", "rev_over_fwd", "rev_over_rev")


class SparseColoring(StrictModule, NonTrainableState):
    """Provider-neutral seed and extraction structure for sparse derivatives."""

    pattern: SparsePattern
    colors: Array
    gather_colors: Array
    gather_elements: Array
    num_colors: int = eqx.field(static=True)
    mode: SparseDerivativeMode = eqx.field(static=True)
    symmetric: bool = eqx.field(static=True)
    compiler: SparseColoringCompiler = eqx.field(static=True)
    coloring_id: str = eqx.field(static=True)

    def __init__(
        self,
        pattern: SparsePattern,
        colors: ArrayLike | Sequence[int],
        gather_colors: ArrayLike | Sequence[int],
        gather_elements: ArrayLike | Sequence[int],
        /,
        *,
        mode: SparseDerivativeMode,
        symmetric: bool = False,
        compiler: SparseColoringCompiler = "native",
        num_colors: int | None = None,
    ):
        if not isinstance(pattern, SparsePattern):
            raise TypeError("pattern must be a SparsePattern.")
        if mode not in (*_JACOBIAN_MODES, *_HESSIAN_MODES):
            raise ValueError(f"Unknown sparse derivative mode {mode!r}.")
        if compiler not in ("native", "asdex"):
            raise ValueError(f"Unknown sparse coloring compiler {compiler!r}.")

        colors_host = _integer_vector("colors", colors)
        expected_colors = pattern.target_size if mode == "rev" else pattern.source_size
        if colors_host.shape != (expected_colors,):
            raise ValueError(
                f"Color vector must have shape {(expected_colors,)}; "
                f"got {colors_host.shape}."
            )
        if np.any(colors_host < -1):
            raise ValueError("Color values must be -1 or non-negative.")
        active = np.unique(colors_host[colors_host >= 0])
        inferred_num_colors = 0 if active.size == 0 else int(active[-1]) + 1
        if not np.array_equal(active, np.arange(inferred_num_colors)):
            raise ValueError("Active sparse colors must be contiguous from zero.")
        color_count = inferred_num_colors if num_colors is None else int(num_colors)
        if color_count != inferred_num_colors:
            raise ValueError(
                "num_colors must equal the number of contiguous active colors."
            )

        gather_colors_host = _integer_vector("gather_colors", gather_colors)
        gather_elements_host = _integer_vector("gather_elements", gather_elements)
        expected_gather_shape = (pattern.nnz,)
        if gather_colors_host.shape != expected_gather_shape:
            raise ValueError(
                f"gather_colors must have shape {expected_gather_shape}; "
                f"got {gather_colors_host.shape}."
            )
        if gather_elements_host.shape != expected_gather_shape:
            raise ValueError(
                f"gather_elements must have shape {expected_gather_shape}; "
                f"got {gather_elements_host.shape}."
            )
        if pattern.nnz:
            if color_count == 0 or np.any(
                (gather_colors_host < 0) | (gather_colors_host >= color_count)
            ):
                raise ValueError("Every sparse route must reference an active color.")
            compressed_dimension = (
                pattern.source_size if mode == "rev" else pattern.target_size
            )
            if np.any(
                (gather_elements_host < 0)
                | (gather_elements_host >= compressed_dimension)
            ):
                raise ValueError(
                    "Every sparse route must reference a valid compressed coordinate."
                )

        symmetric_ = bool(symmetric)
        if symmetric_ and not pattern.symmetric:
            raise ValueError("Symmetric coloring requires a symmetric sparse pattern.")
        if symmetric_:
            _validate_symmetric_coloring(
                pattern,
                colors_host,
                gather_colors_host,
                gather_elements_host,
                mode,
            )
        else:
            _validate_ordinary_coloring(pattern, colors_host, mode)
            expected_gather_colors, expected_gather_elements = _ordinary_gathers(
                pattern, colors_host, mode
            )
            if not np.array_equal(
                gather_colors_host, expected_gather_colors
            ) or not np.array_equal(gather_elements_host, expected_gather_elements):
                raise ValueError(
                    "Nonsymmetric extraction indices do not match the declared coloring."
                )

        payload = {
            "kind": "sparse-coloring",
            "pattern_id": pattern.pattern_id,
            "colors": colors_host.tolist(),
            "gather_colors": gather_colors_host.tolist(),
            "gather_elements": gather_elements_host.tolist(),
            "num_colors": color_count,
            "mode": mode,
            "symmetric": symmetric_,
            "compiler": compiler,
        }
        self.pattern = pattern
        self.colors = jnp.asarray(colors_host, dtype=jnp.int32)
        self.gather_colors = jnp.asarray(gather_colors_host, dtype=jnp.int32)
        self.gather_elements = jnp.asarray(gather_elements_host, dtype=jnp.int32)
        self.num_colors = color_count
        self.mode = mode
        self.symmetric = symmetric_
        self.compiler = compiler
        self.coloring_id = canonical_fingerprint(payload)

    @property
    def seed_dimension(self) -> int:
        return (
            self.pattern.target_size if self.mode == "rev" else self.pattern.source_size
        )

    @property
    def compressed_dimension(self) -> int:
        return (
            self.pattern.source_size if self.mode == "rev" else self.pattern.target_size
        )

    def to_dict(self, /) -> dict[str, Any]:
        """Return a versioned JSON-compatible coloring artifact."""

        return {
            "schema_version": _COLORING_SCHEMA_VERSION,
            "pattern": self.pattern.to_dict(),
            "colors": np.asarray(self.colors).tolist(),
            "gather_colors": np.asarray(self.gather_colors).tolist(),
            "gather_elements": np.asarray(self.gather_elements).tolist(),
            "num_colors": self.num_colors,
            "mode": self.mode,
            "symmetric": self.symmetric,
            "compiler": self.compiler,
            "coloring_id": self.coloring_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "SparseColoring":
        """Restore and validate one coloring artifact."""

        if not isinstance(value, Mapping):
            raise TypeError("Serialized sparse coloring must be a mapping.")
        expected = {
            "schema_version",
            "pattern",
            "colors",
            "gather_colors",
            "gather_elements",
            "num_colors",
            "mode",
            "symmetric",
            "compiler",
            "coloring_id",
        }
        fields = set(value)
        if fields != expected:
            missing = sorted(expected - fields)
            unknown = sorted(fields - expected)
            raise ValueError(
                f"Invalid sparse-coloring fields; missing={missing}, unknown={unknown}."
            )
        schema_version = value["schema_version"]
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != _COLORING_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported sparse-coloring schema {schema_version!r}.")
        for name in ("colors", "gather_colors", "gather_elements"):
            indices = value[name]
            if not isinstance(indices, list) or any(
                not isinstance(index, int) or isinstance(index, bool) for index in indices
            ):
                raise ValueError(
                    f"Serialized sparse-coloring {name} must be a list of integers."
                )
        num_colors = value["num_colors"]
        if (
            not isinstance(num_colors, int)
            or isinstance(num_colors, bool)
            or num_colors < 0
        ):
            raise ValueError(
                "Serialized sparse-coloring num_colors must be a non-negative integer."
            )
        if not isinstance(value["mode"], str):
            raise ValueError("Serialized sparse-coloring mode must be a string.")
        if not isinstance(value["symmetric"], bool):
            raise ValueError("Serialized sparse-coloring symmetric must be boolean.")
        if not isinstance(value["compiler"], str):
            raise ValueError("Serialized sparse-coloring compiler must be a string.")
        if not isinstance(value["coloring_id"], str):
            raise ValueError("Serialized sparse-coloring coloring_id must be a string.")
        coloring = cls(
            SparsePattern.from_dict(value["pattern"]),
            value["colors"],
            value["gather_colors"],
            value["gather_elements"],
            mode=value["mode"],
            symmetric=value["symmetric"],
            compiler=value["compiler"],
            num_colors=num_colors,
        )
        if value["coloring_id"] != coloring.coloring_id:
            raise ValueError(
                "Serialized sparse-coloring fingerprint does not match its data."
            )
        return coloring


def native_coloring(
    pattern: SparsePattern,
    /,
    *,
    derivative_kind: SparseDerivativeKind,
    mode: SparseDerivativeMode | None = None,
) -> SparseColoring:
    """Color a known pattern with deterministic dependency-local greedy coloring."""

    if not isinstance(pattern, SparsePattern):
        raise TypeError("pattern must be a SparsePattern.")
    if derivative_kind == "jacobian":
        if mode is not None and mode not in _JACOBIAN_MODES:
            raise ValueError("Jacobian mode must be 'fwd', 'rev', or None.")
        if mode == "fwd":
            return _native_forward_coloring(pattern, "fwd")
        if mode == "rev":
            return _native_reverse_coloring(pattern)
        forward = _native_forward_coloring(pattern, "fwd")
        reverse = _native_reverse_coloring(pattern)
        return forward if forward.num_colors <= reverse.num_colors else reverse
    if derivative_kind == "hessian":
        resolved_mode: SparseDerivativeMode = "fwd_over_rev" if mode is None else mode
        if resolved_mode not in _HESSIAN_MODES:
            raise ValueError(
                "Hessian mode must be 'fwd_over_rev', 'rev_over_fwd', "
                "'rev_over_rev', or None."
            )
        if not pattern.symmetric:
            raise ValueError("Hessian coloring requires a symmetric sparse pattern.")
        return _native_forward_coloring(pattern, resolved_mode)
    raise ValueError(f"Unknown sparse derivative kind {derivative_kind!r}.")


def _native_forward_coloring(
    pattern: SparsePattern, mode: SparseDerivativeMode, /
) -> SparseColoring:
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    colors = _greedy_colors(
        pattern.source_size,
        pattern.target_size,
        group_indices=rows,
        vertex_indices=cols,
    )
    gather_colors, gather_elements = _ordinary_gathers(pattern, colors, mode)
    return SparseColoring(
        pattern,
        colors,
        gather_colors,
        gather_elements,
        mode=mode,
        compiler="native",
    )


def _native_reverse_coloring(pattern: SparsePattern, /) -> SparseColoring:
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    colors = _greedy_colors(
        pattern.target_size,
        pattern.source_size,
        group_indices=cols,
        vertex_indices=rows,
    )
    gather_colors, gather_elements = _ordinary_gathers(pattern, colors, "rev")
    return SparseColoring(
        pattern,
        colors,
        gather_colors,
        gather_elements,
        mode="rev",
        compiler="native",
    )


def _greedy_colors(
    num_vertices: int,
    num_groups: int,
    /,
    *,
    group_indices: np.ndarray,
    vertex_indices: np.ndarray,
) -> np.ndarray:
    memberships: list[list[int]] = [[] for _ in range(num_vertices)]
    for group, vertex in zip(group_indices, vertex_indices, strict=True):
        memberships[int(vertex)].append(int(group))
    group_colors: list[set[int]] = [set() for _ in range(num_groups)]
    colors = np.full((num_vertices,), -1, dtype=np.int32)
    order = sorted(
        (vertex for vertex in range(num_vertices) if memberships[vertex]),
        key=lambda vertex: (-len(memberships[vertex]), vertex),
    )
    for vertex in order:
        forbidden: set[int] = set()
        for group in memberships[vertex]:
            forbidden.update(group_colors[group])
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
        for group in memberships[vertex]:
            group_colors[group].add(color)
    return colors


def _ordinary_gathers(
    pattern: SparsePattern,
    colors: np.ndarray,
    mode: SparseDerivativeMode,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    if mode == "rev":
        return colors[rows].astype(np.int32), cols.astype(np.int32)
    return colors[cols].astype(np.int32), rows.astype(np.int32)


def _validate_ordinary_coloring(
    pattern: SparsePattern,
    colors: np.ndarray,
    mode: SparseDerivativeMode,
    /,
) -> None:
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    groups = cols if mode == "rev" else rows
    vertices = rows if mode == "rev" else cols
    group_count = pattern.source_size if mode == "rev" else pattern.target_size
    for group in range(group_count):
        assigned = colors[vertices[groups == group]]
        if np.any(assigned < 0) or np.unique(assigned).size != assigned.size:
            raise ValueError("Sparse coloring contains a structural seed collision.")


def _validate_symmetric_coloring(
    pattern: SparsePattern,
    colors: np.ndarray,
    gather_colors: np.ndarray,
    gather_elements: np.ndarray,
    mode: SparseDerivativeMode,
    /,
) -> None:
    rows = np.asarray(pattern.rows, dtype=np.int64)
    cols = np.asarray(pattern.cols, dtype=np.int64)
    compressed_entries: dict[tuple[int, int], list[int]] = {}
    if mode == "rev":
        groups = cols
        vertices = rows
    else:
        groups = rows
        vertices = cols
    for group, vertex in zip(groups, vertices, strict=True):
        key = (int(group), int(colors[vertex]))
        compressed_entries.setdefault(key, []).append(int(vertex))

    for row, column, color, element in zip(
        rows,
        cols,
        gather_colors,
        gather_elements,
        strict=True,
    ):
        direct = int(color) == int(colors[column]) and int(element) == int(row)
        transposed = int(color) == int(colors[row]) and int(element) == int(column)
        if not direct and not transposed:
            raise ValueError(
                "Symmetric extraction must address one endpoint color at the "
                "opposite endpoint coordinate."
            )
        expected_vertex = int(column) if direct else int(row)
        contributors = compressed_entries.get((int(element), int(color)), [])
        if contributors != [expected_vertex]:
            raise ValueError(
                "Symmetric sparse coloring contains an unresolved seed collision."
            )


def _integer_vector(name: str, value: ArrayLike | Sequence[int], /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1.")
    if array.size == 0:
        return np.empty((0,), dtype=np.int32)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    limits = np.iinfo(np.int32)
    if np.any((array < limits.min) | (array > limits.max)):
        raise ValueError(f"{name} values must fit in int32.")
    return array.astype(np.int32, copy=False)


__all__ = [
    "SparseColoring",
    "SparseDerivativeCompiler",
    "SparseDerivativeKind",
    "SparseDerivativeMode",
    "SparseHessianMode",
    "SparseJacobianMode",
    "native_coloring",
]
