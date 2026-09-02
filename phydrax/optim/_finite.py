#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from enum import IntEnum
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._branch_and_bound import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundStatus,
)


_FINITE_SPACE_VERSION = 1
_FINITE_SEARCH_METHOD_ID = "finite-exhaustive-v1"
FiniteEvaluator = Callable[[PyTree[Array]], tuple[Array, Array]]


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
    space_id: str = eqx.field(static=True)

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
        self.space_id = self._compute_space_id()

    def _axis_blocks(self, /) -> tuple[FiniteAxis, ...]:
        return tuple(jax.tree_util.tree_leaves(self.axes, is_leaf=_is_finite_axis))

    def point_spec(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        """Return the shape and dtype PyTree of one product candidate."""
        return self.axis_tree_definition.unflatten(
            tuple(axis.point_spec() for axis in self._axis_blocks())
        )

    def _compute_space_id(self, /) -> str:
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

    def signature(self, /) -> str:
        """Return the content-sensitive identity of this finite candidate space."""
        return self.space_id

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


class FiniteSearchStatus(IntEnum):
    """Observable completion state for one finite search."""

    COMPLETE = 0
    NO_VALID_CANDIDATES = 1
    FRONTIER_CAPACITY_EXCEEDED = 2
    STOPPED = 3


class FiniteMinimum(StrictModule):
    """Exact scalar minimum with stable flat-index tie breaking."""


class FiniteTopK(StrictModule):
    """Exact fixed-capacity scalar top-k reducer."""

    k: int = eqx.field(static=True)

    def __init__(self, k: int, /):
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be a positive integer.")
        resolved = int(k)
        if resolved <= 0:
            raise ValueError("k must be positive.")
        self.k = resolved


class FinitePareto(StrictModule):
    """Exact nondominated archive with a declared static capacity."""

    objective_count: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    def __init__(self, objective_count: int, capacity: int, /):
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (objective_count, capacity)
        ):
            raise TypeError("objective_count and capacity must be positive integers.")
        objectives = int(objective_count)
        capacity_ = int(capacity)
        if objectives <= 0 or capacity_ <= 0:
            raise ValueError("objective_count and capacity must be positive.")
        self.objective_count = objectives
        self.capacity = capacity_


FiniteReducer = FiniteMinimum | FiniteTopK | FinitePareto


class FiniteLandscapePolicy(StrictModule):
    """Explicit storage budget for index-aligned finite objective landscapes."""

    retain: bool = eqx.field(static=True)
    maximum_entries: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        retain: bool = False,
        maximum_entries: int = 1_000_000,
        maximum_bytes: int = 64 * 1024 * 1024,
    ):
        if not isinstance(retain, bool):
            raise TypeError("retain must be a bool.")
        entries = int(maximum_entries)
        bytes_ = int(maximum_bytes)
        if entries <= 0 or bytes_ <= 0:
            raise ValueError("Landscape storage budgets must be positive.")
        self.retain = retain
        self.maximum_entries = entries
        self.maximum_bytes = bytes_


class FiniteSearchProgress(StrictModule):
    """Immutable host-callback snapshot between compiled evaluation batches."""

    attempted_evaluations: Array
    invalid_evaluations: Array
    retained_candidates: Array
    total_candidates: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)


@runtime_checkable
class FiniteSearchCallback(Protocol):
    """Host-only progress callback; return true to request an explicit stop."""

    def __call__(self, progress: FiniteSearchProgress, /) -> bool: ...


class FiniteLocalRefinement(StrictModule):
    """Typed continuous refinement composed after, never inside, finite search."""

    encode: Callable[[Any], Array] = eqx.field(static=True)
    decode: Callable[[Array], Any] = eqx.field(static=True)
    solve: Callable[[Array], tuple[Array, Any]] = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)

    def __init__(
        self,
        encode: Callable[[Any], Array],
        decode: Callable[[Array], Any],
        solve: Callable[[Array], tuple[Array, Any]],
        /,
        *,
        refinement_id: str = "finite-local-refinement",
    ):
        if not all(callable(value) for value in (encode, decode, solve)):
            raise TypeError("encode, decode, and solve must be callable.")
        identifier = str(refinement_id)
        if not identifier:
            raise ValueError("refinement_id must be non-empty.")
        self.encode = encode
        self.decode = decode
        self.solve = solve
        self.refinement_id = identifier


class FiniteSearchResult(StrictModule):
    """Fixed-capacity reducer output and exactness/storage evidence."""

    points: PyTree[Array]
    scores: Array
    valid: Array
    flat_indices: Array
    product_indices: tuple[Array, ...]
    landscape_scores: Array | None
    landscape_valid: Array | None
    landscape_evaluated: Array | None
    attempted_evaluations: Array
    invalid_evaluations: Array
    status: Array
    exact: Array
    refined_point: Any
    refined_score: Array | None
    refinement_evidence: Any
    space_id: str = eqx.field(static=True)
    reducer_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteSearchStatus.COMPLETE)


def _reducer_shape(
    reducer: FiniteReducer,
    score_spec: jax.ShapeDtypeStruct,
    space_size: int,
    /,
) -> tuple[int, int, str]:
    if isinstance(reducer, FiniteMinimum):
        if score_spec.shape != ():
            raise ValueError("FiniteMinimum requires one scalar objective.")
        return 1, 1, "minimum"
    if isinstance(reducer, FiniteTopK):
        if score_spec.shape != ():
            raise ValueError("FiniteTopK requires one scalar objective.")
        if reducer.k > space_size:
            raise ValueError("FiniteTopK.k cannot exceed the finite space size.")
        return reducer.k, 1, f"top-k:{reducer.k}"
    if isinstance(reducer, FinitePareto):
        if score_spec.shape != (reducer.objective_count,):
            raise ValueError(
                "FinitePareto evaluator scores must have trailing shape "
                f"({reducer.objective_count},)."
            )
        return (
            reducer.capacity,
            reducer.objective_count,
            (f"pareto:{reducer.objective_count}:{reducer.capacity}"),
        )
    raise TypeError("reducer must be FiniteMinimum, FiniteTopK, or FinitePareto.")


def _stable_scalar_reduce(
    scores: Array,
    valid: Array,
    indices: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array]:
    safe = jnp.where(valid, scores, jnp.inf)
    index_order = jnp.argsort(indices, stable=True)
    safe = safe[index_order]
    valid = valid[index_order]
    indices = indices[index_order]
    score_order = jnp.argsort(safe, stable=True)
    selected = score_order[:capacity]
    selected_valid = valid[selected] & jnp.isfinite(safe[selected])
    return (
        jnp.where(selected_valid, safe[selected], jnp.nan),
        selected_valid,
        jnp.where(selected_valid, indices[selected], -1),
    )


def _pareto_reduce(
    scores: Array,
    valid: Array,
    indices: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    finite = valid & jnp.all(jnp.isfinite(scores), axis=-1)
    candidate = scores[:, None, :]
    competitor = scores[None, :, :]
    dominates = (
        finite[None, :]
        & jnp.all(competitor <= candidate, axis=-1)
        & jnp.any(competitor < candidate, axis=-1)
    )
    nondominated = finite & ~jnp.any(dominates, axis=-1)
    order = jnp.argsort(indices, stable=True)
    ordered_nondominated = nondominated[order]
    selected_positions = jnp.nonzero(
        ordered_nondominated,
        size=capacity,
        fill_value=0,
    )[0]
    selected = order[selected_positions]
    count = jnp.sum(nondominated, dtype=jnp.int32)
    slot = jnp.arange(capacity, dtype=jnp.int32)
    selected_valid = slot < count
    return (
        jnp.where(selected_valid[:, None], scores[selected], jnp.nan),
        selected_valid,
        jnp.where(selected_valid, indices[selected], -1),
        count > capacity,
    )


def _finite_evaluator_contract(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    reducer: FiniteReducer,
    /,
) -> tuple[jax.ShapeDtypeStruct, int, int, str]:
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    output = eqx.filter_eval_shape(evaluator, space.point_spec())
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("Finite evaluators must return a (score, valid) tuple.")
    score_spec, valid_spec = output
    if not isinstance(score_spec, jax.ShapeDtypeStruct) or not isinstance(
        valid_spec, jax.ShapeDtypeStruct
    ):
        raise TypeError("Finite evaluator scores and validity must be arrays.")
    if not np.issubdtype(np.dtype(score_spec.dtype), np.floating):
        raise TypeError("Finite evaluator scores must use a real floating dtype.")
    if np.dtype(valid_spec.dtype) != np.dtype(bool) or valid_spec.shape != ():
        raise ValueError("Finite evaluator validity must be one boolean scalar.")
    capacity, objectives, reducer_id = _reducer_shape(reducer, score_spec, space.size)
    return score_spec, capacity, objectives, reducer_id


class FiniteCertifiedLowerBound(StrictModule, abc.ABC):
    """Caller-owned certified lower bound over half-open flat-index boxes."""

    __strict_abstract__ = True

    certificate_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def lower_bound(
        self,
        space: FiniteProductSpace,
        start: int,
        stop: int,
        /,
    ) -> float:
        raise NotImplementedError


class FiniteAdaptiveSearch(StrictModule):
    """Certified adaptive finite minimum through the shared branch-and-bound engine."""

    bound: FiniteCertifiedLowerBound
    policy: BranchAndBoundPolicy

    def __init__(
        self,
        bound: FiniteCertifiedLowerBound,
        /,
        *,
        policy: BranchAndBoundPolicy | None = None,
    ):
        if not isinstance(bound, FiniteCertifiedLowerBound):
            raise TypeError("bound must be a FiniteCertifiedLowerBound.")
        identifier = str(bound.certificate_id)
        if not identifier:
            raise ValueError("Finite lower-bound certificate_id must be non-empty.")
        policy_ = BranchAndBoundPolicy() if policy is None else policy
        if not isinstance(policy_, BranchAndBoundPolicy):
            raise TypeError("policy must be a BranchAndBoundPolicy.")
        self.bound = bound
        self.policy = policy_


class _FiniteAdaptiveProblem(AbstractBranchAndBoundProblem):
    evaluator: FiniteEvaluator = eqx.field(static=True)
    space: FiniteProductSpace
    bound: FiniteCertifiedLowerBound

    def __init__(
        self,
        evaluator: FiniteEvaluator,
        space: FiniteProductSpace,
        bound: FiniteCertifiedLowerBound,
        /,
    ):
        self.evaluator = evaluator
        self.space = space
        self.bound = bound
        self.problem_id = canonical_fingerprint(
            {
                "kind": "finite-adaptive-search",
                "space": space.space_id,
                "bound": bound.certificate_id,
            }
        )

    def root(self, /):
        return (0, self.space.size)

    def node_id(self, node, /) -> str:
        return f"{int(node[0]):020d}:{int(node[1]):020d}"

    def lower_bound(self, node, /) -> float:
        start, stop = (int(node[0]), int(node[1]))
        value = float(self.bound.lower_bound(self.space, start, stop))
        if not np.isfinite(value):
            return value
        if stop - start == 1:
            score, valid = self.evaluator(self.space.take(start))
            score_ = float(np.asarray(score))
            valid_ = bool(np.asarray(valid)) and np.isfinite(score_)
            if valid_ and value > score_:
                raise ValueError(
                    "Finite lower-bound certificate exceeds a singleton objective."
                )
        return value

    def feasible(self, node, /) -> bool:
        start, stop = (int(node[0]), int(node[1]))
        return 0 <= start < stop <= self.space.size

    def complete(self, node, /) -> bool:
        return int(node[1]) - int(node[0]) == 1

    def objective(self, node, /) -> float:
        score, valid = self.evaluator(self.space.take(int(node[0])))
        value = float(np.asarray(score))
        return value if bool(np.asarray(valid)) and np.isfinite(value) else math.inf

    def branch(self, node, /):
        start, stop = (int(node[0]), int(node[1]))
        midpoint = start + (stop - start) // 2
        return ((start, midpoint), (midpoint, stop))


def _search_finite_adaptive(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    search: FiniteAdaptiveSearch,
    /,
) -> FiniteSearchResult:
    score_spec, capacity, objectives, reducer_id = _finite_evaluator_contract(
        evaluator, space, FiniteMinimum()
    )
    del score_spec, capacity, objectives
    execution = branch_and_bound(
        _FiniteAdaptiveProblem(evaluator, space, search.bound),
        policy=search.policy,
    )
    valid = execution.incumbent is not None and np.isfinite(
        float(np.asarray(execution.objective))
    )
    index = int(execution.incumbent[0]) if valid else 0
    flat_indices = jnp.asarray([index if valid else -1], dtype=jnp.int64)
    mask = jnp.asarray([valid])
    points = space._take_unchecked(jnp.asarray([index], dtype=jnp.int64))
    product_indices = tuple(
        jnp.where(mask, value, -1)
        for value in space._unravel_unchecked(jnp.asarray([index], dtype=jnp.int64))
    )
    exact = execution.status == int(BranchAndBoundStatus.OPTIMAL)
    status = jnp.where(
        valid,
        jnp.where(
            exact,
            int(FiniteSearchStatus.COMPLETE),
            int(FiniteSearchStatus.STOPPED),
        ),
        int(FiniteSearchStatus.NO_VALID_CANDIDATES),
    ).astype(jnp.int32)
    return FiniteSearchResult(
        points,
        jnp.asarray([float(np.asarray(execution.objective)) if valid else jnp.nan]),
        mask,
        flat_indices,
        product_indices,
        None,
        None,
        None,
        execution.explored_nodes.astype(jnp.int64),
        jnp.asarray(0, dtype=jnp.int64),
        status,
        exact & valid,
        None,
        None,
        execution,
        space.space_id,
        reducer_id,
    )


def search_finite(
    evaluator: FiniteEvaluator,
    space: FiniteProductSpace,
    reducer: FiniteReducer | None = None,
    /,
    *,
    search: FiniteExhaustiveSearch | FiniteAdaptiveSearch | None = None,
    landscape: FiniteLandscapePolicy | None = None,
    callback: FiniteSearchCallback | None = None,
    refinement: FiniteLocalRefinement | None = None,
) -> FiniteSearchResult:
    """Stream exact finite reducers over deterministic flat indices.

    Callback orchestration is deliberately host-side and occurs only between
    compiled, fixed-shape batches. Selection and optional refinement are
    nondifferentiable; refinement evidence never changes the finite exactness claim.
    """

    reducer_ = FiniteMinimum() if reducer is None else reducer
    search_ = FiniteExhaustiveSearch() if search is None else search
    landscape_ = FiniteLandscapePolicy() if landscape is None else landscape
    if not isinstance(landscape_, FiniteLandscapePolicy):
        raise TypeError("landscape must be a FiniteLandscapePolicy.")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None.")
    if refinement is not None and not isinstance(refinement, FiniteLocalRefinement):
        raise TypeError("refinement must be a FiniteLocalRefinement or None.")
    if isinstance(search_, FiniteAdaptiveSearch):
        if not isinstance(reducer_, FiniteMinimum):
            raise TypeError("FiniteAdaptiveSearch supports FiniteMinimum only.")
        if landscape_.retain or callback is not None or refinement is not None:
            raise ValueError(
                "Adaptive search does not support landscape, callback, or refinement."
            )
        return _search_finite_adaptive(evaluator, space, search_)
    if not isinstance(search_, FiniteExhaustiveSearch):
        raise TypeError(
            "search must be a FiniteExhaustiveSearch or FiniteAdaptiveSearch."
        )
    score_spec, capacity, objectives, reducer_id = _finite_evaluator_contract(
        evaluator, space, reducer_
    )
    entries = space.size * objectives
    bytes_ = entries * np.dtype(score_spec.dtype).itemsize
    if landscape_.retain and (
        entries > landscape_.maximum_entries or bytes_ > landscape_.maximum_bytes
    ):
        raise ValueError(
            "Finite landscape exceeds its explicit entry or byte storage budget."
        )

    retained_scores = jnp.full((capacity, objectives), jnp.nan, dtype=score_spec.dtype)
    retained_valid = jnp.zeros((capacity,), dtype=bool)
    retained_indices = jnp.full((capacity,), -1, dtype=jnp.int64)
    landscape_scores = (
        jnp.full((space.size, objectives), jnp.nan, dtype=score_spec.dtype)
        if landscape_.retain
        else None
    )
    landscape_valid = jnp.zeros((space.size,), dtype=bool) if landscape_.retain else None
    landscape_evaluated = (
        jnp.zeros((space.size,), dtype=bool) if landscape_.retain else None
    )
    attempted = jnp.asarray(0, dtype=jnp.int64)
    invalid = jnp.asarray(0, dtype=jnp.int64)
    overflow = jnp.asarray(False)
    stopped = False
    batch_size = search_.effective_batch_size(space.size)

    @eqx.filter_jit
    def evaluate_batch(indices):
        points = space._take_unchecked(indices)
        points = jax.tree_util.tree_map(jax.lax.stop_gradient, points)
        return jax.vmap(evaluator)(points)

    for start in range(0, space.size, batch_size):
        stop = min(start + batch_size, space.size)
        indices = jnp.arange(start, stop, dtype=jnp.int64)
        batch_scores, declared_valid = evaluate_batch(indices)
        batch_scores = batch_scores[:, None] if objectives == 1 else batch_scores
        effective_valid = declared_valid & jnp.all(jnp.isfinite(batch_scores), axis=-1)
        attempted = attempted + jnp.asarray(stop - start, dtype=jnp.int64)
        invalid = invalid + jnp.sum(~effective_valid, dtype=jnp.int64)
        if landscape_.retain:
            landscape_scores = landscape_scores.at[start:stop].set(batch_scores)
            landscape_valid = landscape_valid.at[start:stop].set(effective_valid)
            landscape_evaluated = landscape_evaluated.at[start:stop].set(True)
        combined_scores = jnp.concatenate((retained_scores, batch_scores), axis=0)
        combined_valid = jnp.concatenate((retained_valid, effective_valid), axis=0)
        combined_indices = jnp.concatenate((retained_indices, indices), axis=0)
        if isinstance(reducer_, (FiniteMinimum, FiniteTopK)):
            scalar, retained_valid, retained_indices = _stable_scalar_reduce(
                combined_scores[:, 0],
                combined_valid,
                combined_indices,
                capacity,
            )
            retained_scores = scalar[:, None]
        else:
            (
                retained_scores,
                retained_valid,
                retained_indices,
                batch_overflow,
            ) = _pareto_reduce(
                combined_scores,
                combined_valid,
                combined_indices,
                capacity,
            )
            overflow = overflow | batch_overflow
        if callback is not None:
            stopped = bool(
                callback(
                    FiniteSearchProgress(
                        jnp.asarray(attempted, dtype=jnp.int64),
                        jnp.asarray(invalid, dtype=jnp.int64),
                        jnp.sum(retained_valid, dtype=jnp.int32),
                        space.size,
                        stop == space.size,
                    )
                )
            )
            if stopped:
                break

    any_valid = jnp.any(retained_valid)
    status = jnp.where(
        stopped,
        int(FiniteSearchStatus.STOPPED),
        jnp.where(
            overflow,
            int(FiniteSearchStatus.FRONTIER_CAPACITY_EXCEEDED),
            jnp.where(
                any_valid,
                int(FiniteSearchStatus.COMPLETE),
                int(FiniteSearchStatus.NO_VALID_CANDIDATES),
            ),
        ),
    ).astype(jnp.int32)
    exact = status == int(FiniteSearchStatus.COMPLETE)
    safe_indices = jnp.where(retained_valid, retained_indices, 0)
    points = space._take_unchecked(safe_indices)
    product_indices = tuple(
        jnp.where(retained_valid, value, -1)
        for value in space._unravel_unchecked(safe_indices)
    )
    refined_point = None
    refined_score = None
    refinement_evidence = None
    if refinement is not None:
        if not bool(np.asarray(retained_valid[0])):
            raise ValueError("Local refinement requires a valid finite seed.")
        seed = jax.tree_util.tree_map(lambda value: value[0], points)
        encoded = jnp.asarray(refinement.encode(seed))
        if encoded.ndim != 1 or not jnp.issubdtype(encoded.dtype, jnp.floating):
            raise TypeError(
                "FiniteLocalRefinement.encode must return one floating vector."
            )
        refined_coordinates, refinement_evidence = refinement.solve(encoded)
        refined_point = refinement.decode(jnp.asarray(refined_coordinates))
        candidate_score, candidate_valid = evaluator(refined_point)
        if jnp.asarray(candidate_score).shape != score_spec.shape:
            raise ValueError("Refined objective shape differs from the finite objective.")
        refined_score = jnp.where(candidate_valid, candidate_score, jnp.nan)

    return FiniteSearchResult(
        jax.tree_util.tree_map(jax.lax.stop_gradient, points),
        jax.lax.stop_gradient(
            retained_scores[:, 0] if objectives == 1 else retained_scores
        ),
        jax.lax.stop_gradient(retained_valid),
        jax.lax.stop_gradient(retained_indices),
        jax.tree_util.tree_map(jax.lax.stop_gradient, product_indices),
        None if landscape_scores is None else jax.lax.stop_gradient(landscape_scores),
        None if landscape_valid is None else jax.lax.stop_gradient(landscape_valid),
        (
            None
            if landscape_evaluated is None
            else jax.lax.stop_gradient(landscape_evaluated)
        ),
        jnp.asarray(attempted, dtype=jnp.int64),
        jnp.asarray(invalid, dtype=jnp.int64),
        status,
        exact,
        refined_point,
        refined_score,
        refinement_evidence,
        space.space_id,
        reducer_id,
    )


__all__ = [
    "FiniteAdaptiveSearch",
    "FiniteAxis",
    "FiniteCertifiedLowerBound",
    "FiniteExhaustiveSearch",
    "FiniteLandscapePolicy",
    "FiniteLocalRefinement",
    "FiniteMinimum",
    "FinitePareto",
    "FiniteProductSpace",
    "FiniteSearchCallback",
    "FiniteSearchProgress",
    "FiniteSearchResult",
    "FiniteSearchStatus",
    "FiniteTopK",
    "search_finite",
]
