#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


_FINITE_SPACE_VERSION = 1
_FINITE_SEARCH_METHOD_ID = "finite-exhaustive-v1"


def _is_finite_axis(value: Any, /) -> bool:
    return isinstance(value, FiniteAxis)


def _path_name(path: tuple[Any, ...], /) -> str:
    return jax.tree_util.keystr(path) or "<root>"


def _checked_integer_array(value: Any, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.bool_) or not jnp.issubdtype(
        array.dtype, jnp.integer
    ):
        raise TypeError(f"{name} must contain integer indices.")
    return array


def _checked_index_bounds(
    value: Array,
    invalid: Array,
    message: str,
    /,
) -> Array:
    if isinstance(invalid, jax_core.Tracer):
        return eqx.error_if(value, invalid, message)
    if bool(invalid):
        raise IndexError(message)
    return value


class FiniteAxis(StrictModule):
    """One correlated finite axis with array-valued candidate payloads."""

    values: PyTree[Array]
    size: int = eqx.field(static=True)
    tree_definition: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    payload_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, values: PyTree[Any], /):
        path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(values)
        if not path_leaves:
            raise ValueError("FiniteAxis requires at least one array leaf.")

        arrays: list[Array] = []
        paths: list[str] = []
        payload_shapes: list[tuple[int, ...]] = []
        dtypes: list[str] = []
        axis_size: int | None = None
        for path, raw in path_leaves:
            if isinstance(raw, (str, bytes)):
                raise TypeError("FiniteAxis leaves must be numerical or boolean arrays.")
            array = jnp.asarray(raw)
            if np.dtype(array.dtype).kind not in "biufc":
                raise TypeError("FiniteAxis leaves must be numerical or boolean arrays.")
            if array.ndim == 0:
                raise ValueError(
                    f"FiniteAxis leaf {_path_name(path)} must have a leading "
                    "candidate dimension."
                )
            leading = int(array.shape[0])
            if leading == 0:
                raise ValueError("FiniteAxis candidate dimensions must be nonempty.")
            if axis_size is None:
                axis_size = leading
            elif leading != axis_size:
                raise ValueError(
                    "Every FiniteAxis leaf must have the same leading candidate "
                    f"dimension; {_path_name(path)} has {leading}, expected {axis_size}."
                )
            arrays.append(array)
            paths.append(_path_name(path))
            payload_shapes.append(tuple(int(size) for size in array.shape[1:]))
            dtypes.append(str(array.dtype))

        assert axis_size is not None
        self.values = tree_definition.unflatten(arrays)
        self.size = axis_size
        self.tree_definition = tree_definition
        self.paths = tuple(paths)
        self.payload_shapes = tuple(payload_shapes)
        self.dtypes = tuple(dtypes)

    def point_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        """Return the shape and dtype of one selected axis payload."""
        leaves = tuple(
            jax.ShapeDtypeStruct(shape, np.dtype(dtype))
            for shape, dtype in zip(self.payload_shapes, self.dtypes, strict=True)
        )
        return self.tree_definition.unflatten(leaves)

    def _take_unchecked(self, index: Array, /) -> PyTree[Array]:
        return jax.tree_util.tree_map(
            lambda values: values[index],
            self.values,
        )


class FiniteProductSpace(StrictModule):
    """Lazy Cartesian product of explicit correlated finite axes."""

    axes: PyTree[FiniteAxis]
    product_shape: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    axis_paths: tuple[str, ...] = eqx.field(static=True)
    axis_tree_definition: Any = eqx.field(static=True)
    point_tree_definition: Any = eqx.field(static=True)

    def __init__(self, axes: PyTree[FiniteAxis], /):
        path_axes, axis_tree_definition = jax.tree_util.tree_flatten_with_path(
            axes,
            is_leaf=_is_finite_axis,
        )
        if not path_axes:
            raise ValueError("FiniteProductSpace requires at least one FiniteAxis.")
        if any(not isinstance(axis, FiniteAxis) for _, axis in path_axes):
            raise TypeError(
                "Every FiniteProductSpace outer PyTree leaf must be a FiniteAxis."
            )

        axis_blocks = tuple(axis for _, axis in path_axes)
        product_shape = tuple(axis.size for axis in axis_blocks)
        size = math.prod(product_shape)
        if size > np.iinfo(np.int64).max:
            raise OverflowError(
                "FiniteProductSpace cardinality exceeds signed 64-bit indexing."
            )
        point = axis_tree_definition.unflatten(
            tuple(axis.point_spec() for axis in axis_blocks)
        )

        self.axes = axes
        self.product_shape = product_shape
        self.size = size
        self.axis_paths = tuple(_path_name(path) for path, _ in path_axes)
        self.axis_tree_definition = axis_tree_definition
        self.point_tree_definition = jax.tree_util.tree_structure(point)

    def _axis_blocks(self, /) -> tuple[FiniteAxis, ...]:
        return tuple(jax.tree_util.tree_leaves(self.axes, is_leaf=_is_finite_axis))

    def point_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        """Return the shape and dtype PyTree of one product candidate."""
        return self.axis_tree_definition.unflatten(
            tuple(axis.point_spec() for axis in self._axis_blocks())
        )

    def signature(self, /) -> str:
        """Return a content-sensitive identity for this finite candidate space."""
        axes = []
        for path, axis in zip(
            self.axis_paths,
            self._axis_blocks(),
            strict=True,
        ):
            axes.append(
                {
                    "path": path,
                    "size": axis.size,
                    "treedef": str(axis.tree_definition),
                    "values": array_tree_fingerprint(axis.values),
                }
            )
        return canonical_fingerprint(
            {
                "kind": "finite-product-space",
                "version": _FINITE_SPACE_VERSION,
                "axis_treedef": str(self.axis_tree_definition),
                "point_treedef": str(self.point_tree_definition),
                "product_shape": list(self.product_shape),
                "axes": axes,
            }
        )

    def _unravel_unchecked(self, flat_index: Array, /) -> tuple[Array, ...]:
        remaining = jnp.asarray(flat_index, dtype=jnp.int64)
        reversed_indices: list[Array] = []
        for axis_size in reversed(self.product_shape):
            reversed_indices.append(jnp.mod(remaining, axis_size))
            remaining = jnp.floor_divide(remaining, axis_size)
        return tuple(reversed(reversed_indices))

    def unravel_index(self, flat_index: Any, /) -> tuple[Array, ...]:
        """Convert checked row-major flat indices to product-axis indices."""
        raw = _checked_integer_array(flat_index, "flat_index")
        checked = _checked_index_bounds(
            raw,
            jnp.any(raw < 0) | jnp.any(raw >= self.size),
            "FiniteProductSpace flat index is out of range.",
        )
        return self._unravel_unchecked(checked.astype(jnp.int64))

    def ravel_index(self, product_index: Sequence[Any], /) -> Array:
        """Convert checked product-axis indices to row-major flat indices."""
        if not isinstance(product_index, Sequence) or isinstance(
            product_index, (str, bytes)
        ):
            raise TypeError("product_index must be a sequence of axis indices.")
        if len(product_index) != len(self.product_shape):
            raise ValueError(
                "product_index length must equal the number of finite product axes."
            )
        raw_indices = tuple(
            _checked_integer_array(index, f"product_index[{axis}]")
            for axis, index in enumerate(product_index)
        )
        broadcast_shape = jnp.broadcast_shapes(*(index.shape for index in raw_indices))
        flat = jnp.zeros(broadcast_shape, dtype=jnp.int64)
        for axis, (raw, axis_size) in enumerate(
            zip(raw_indices, self.product_shape, strict=True)
        ):
            raw = jnp.broadcast_to(raw, broadcast_shape)
            checked = _checked_index_bounds(
                raw,
                jnp.any(raw < 0) | jnp.any(raw >= axis_size),
                f"FiniteProductSpace product index for axis {axis} is out of range.",
            ).astype(jnp.int64)
            flat = flat * axis_size + checked
        return flat

    def _take_product_unchecked(
        self,
        product_index: tuple[Array, ...],
        /,
    ) -> PyTree[Array]:
        selected = tuple(
            axis._take_unchecked(index)
            for axis, index in zip(
                self._axis_blocks(),
                product_index,
                strict=True,
            )
        )
        return self.axis_tree_definition.unflatten(selected)

    def _take_unchecked(self, flat_index: Array, /) -> PyTree[Array]:
        return self._take_product_unchecked(self._unravel_unchecked(flat_index))

    def take(self, flat_index: Any, /) -> PyTree[Array]:
        """Select one or more product candidates by checked flat index."""
        raw = _checked_integer_array(flat_index, "flat_index")
        checked = _checked_index_bounds(
            raw,
            jnp.any(raw < 0) | jnp.any(raw >= self.size),
            "FiniteProductSpace flat index is out of range.",
        )
        return self._take_unchecked(checked.astype(jnp.int64))


class FiniteExhaustiveSearch(StrictModule):
    """Exact finite enumeration with bounded transient evaluation batches."""

    batch_size: int | None = eqx.field(static=True)

    def __init__(self, batch_size: int | None = None):
        if batch_size is None:
            self.batch_size = None
            return
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer or None.")
        resolved = int(batch_size)
        if resolved <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = resolved

    @property
    def method_id(self) -> str:
        return _FINITE_SEARCH_METHOD_ID

    def effective_batch_size(self, candidate_count: int, /) -> int:
        count = int(candidate_count)
        if count <= 0:
            raise ValueError("candidate_count must be positive.")
        requested = 1 if self.batch_size is None else self.batch_size
        return min(requested, count)


class _FiniteMinimumState(StrictModule):
    minimum: Array
    valid: Array
    flat_index: Array
    attempted_evaluations: Array
    invalid_evaluations: Array


class _FiniteMinimumEvidence(StrictModule):
    minimum: Array
    valid: Array
    flat_index: Array
    product_index: tuple[Array, ...]
    attempted_evaluations: Array
    invalid_evaluations: Array


FiniteEvaluator = Callable[[PyTree[Array]], tuple[Array, Array]]


def _merge_minimum(
    left: _FiniteMinimumState,
    right: _FiniteMinimumState,
    /,
) -> _FiniteMinimumState:
    right_wins = right.valid & (
        ~left.valid
        | (right.minimum < left.minimum)
        | ((right.minimum == left.minimum) & (right.flat_index < left.flat_index))
    )
    return _FiniteMinimumState(
        minimum=jnp.where(right_wins, right.minimum, left.minimum),
        valid=left.valid | right.valid,
        flat_index=jnp.where(right_wins, right.flat_index, left.flat_index),
        attempted_evaluations=(left.attempted_evaluations + right.attempted_evaluations),
        invalid_evaluations=left.invalid_evaluations + right.invalid_evaluations,
    )


def _batch_minimum(
    scores: Array,
    declared_valid: Array,
    flat_indices: Array,
    /,
) -> _FiniteMinimumState:
    effective_valid = declared_valid & jnp.isfinite(scores)
    safe_scores = jnp.where(effective_valid, scores, jnp.inf)
    local_index = jnp.argmin(safe_scores, axis=0)
    minimum = jnp.take_along_axis(
        safe_scores,
        local_index[None, ...],
        axis=0,
    )[0]
    valid = jnp.any(effective_valid, axis=0)
    flat_index = flat_indices[0] + local_index.astype(jnp.int64)
    return _FiniteMinimumState(
        minimum=minimum,
        valid=valid,
        flat_index=jnp.where(valid, flat_index, -1),
        attempted_evaluations=jnp.asarray(scores.shape[0], dtype=jnp.int64),
        invalid_evaluations=jnp.sum(~effective_valid, axis=0, dtype=jnp.int64),
    )


def _evaluate_finite_batch(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    flat_indices: Array,
    /,
) -> _FiniteMinimumState:
    points = space._take_unchecked(flat_indices)
    points = jax.tree_util.tree_map(jax.lax.stop_gradient, points)
    scores, declared_valid = jax.vmap(evaluator)(points)
    return _batch_minimum(scores, declared_valid, flat_indices)


@eqx.filter_jit
def _run_exhaustive_minimum(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    /,
    *,
    batch_size: int,
    output_shape: tuple[int, ...],
    output_dtype: str,
) -> _FiniteMinimumEvidence:
    state = _FiniteMinimumState(
        minimum=jnp.full(output_shape, jnp.inf, dtype=np.dtype(output_dtype)),
        valid=jnp.zeros(output_shape, dtype=bool),
        flat_index=jnp.full(output_shape, -1, dtype=jnp.int64),
        attempted_evaluations=jnp.asarray(0, dtype=jnp.int64),
        invalid_evaluations=jnp.zeros(output_shape, dtype=jnp.int64),
    )
    full_batches, remainder = divmod(space.size, batch_size)

    def body(batch_index, current):
        start = jnp.asarray(batch_index, dtype=jnp.int64) * batch_size
        indices = start + jnp.arange(batch_size, dtype=jnp.int64)
        batch = _evaluate_finite_batch(evaluator, space, indices)
        return _merge_minimum(current, batch)

    state = jax.lax.fori_loop(0, full_batches, body, state)
    if remainder:
        start = full_batches * batch_size
        indices = start + jnp.arange(remainder, dtype=jnp.int64)
        state = _merge_minimum(
            state,
            _evaluate_finite_batch(evaluator, space, indices),
        )

    safe_flat_index = jnp.where(state.valid, state.flat_index, 0)
    product_index = tuple(
        jnp.where(state.valid, index, -1)
        for index in space._unravel_unchecked(safe_flat_index)
    )
    return _FiniteMinimumEvidence(
        minimum=jax.lax.stop_gradient(jnp.where(state.valid, state.minimum, jnp.nan)),
        valid=jax.lax.stop_gradient(state.valid),
        flat_index=jax.lax.stop_gradient(state.flat_index),
        product_index=jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            product_index,
        ),
        attempted_evaluations=state.attempted_evaluations,
        invalid_evaluations=state.invalid_evaluations,
    )


def _exhaustive_minimum(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    search: FiniteExhaustiveSearch,
    /,
) -> _FiniteMinimumEvidence:
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    if not isinstance(space, FiniteProductSpace):
        raise TypeError("space must be a FiniteProductSpace.")
    if not isinstance(search, FiniteExhaustiveSearch):
        raise TypeError("search must be a FiniteExhaustiveSearch.")

    output = eqx.filter_eval_shape(evaluator, space.point_spec())
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Finite evaluators must return a (score, valid) tuple.")
    score_spec, valid_spec = output
    if not isinstance(score_spec, jax.ShapeDtypeStruct) or not isinstance(
        valid_spec, jax.ShapeDtypeStruct
    ):
        raise TypeError("Finite evaluator scores and validity must be arrays.")
    if not np.issubdtype(np.dtype(score_spec.dtype), np.floating):
        raise TypeError("Finite evaluator scores must be real floating-point arrays.")
    if np.dtype(valid_spec.dtype) != np.dtype(bool):
        raise TypeError("Finite evaluator validity must be boolean.")
    if score_spec.shape != valid_spec.shape:
        raise ValueError("Finite evaluator score and validity shapes must match.")
    if any(size == 0 for size in score_spec.shape):
        raise ValueError("Finite evaluator output dimensions must be nonempty.")

    return _run_exhaustive_minimum(
        evaluator,
        space,
        batch_size=search.effective_batch_size(space.size),
        output_shape=tuple(int(size) for size in score_spec.shape),
        output_dtype=str(np.dtype(score_spec.dtype)),
    )


__all__ = [
    "FiniteAxis",
    "FiniteExhaustiveSearch",
    "FiniteProductSpace",
]
