#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral numerical algebraic-geometry data contracts.

These records retain numerical evidence.  A completed trace test or a transitive
monodromy action is not an exact proof of irreducibility, radicality, or
completeness of the underlying algebraic set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from math import isfinite
from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._grading import PolynomialVariableGroup


class PathStatus(str, Enum):
    """Observed disposition of one requested continuation path."""

    SUCCESS = "success"
    TRACKING_FAILED = "tracking-failed"
    SINGULAR = "singular"
    DIVERGED = "diverged"
    INVALID_ENDPOINT = "invalid-endpoint"
    NOT_ATTEMPTED = "not-attempted"


class DecompositionStatus(str, Enum):
    """Qualification of finite numerical decomposition evidence."""

    EVIDENCE_COMPLETE = "evidence-complete"
    INCOMPLETE = "incomplete"
    PARTIAL_PATH_FAILURE = "partial-path-failure"
    TRACE_TEST_FAILED = "trace-test-failed"
    BUDGET_EXHAUSTED = "budget-exhausted"


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _nonnegative_integer(value: Any, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        result = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: Any, name: str, /) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_array(value: ArrayLike, name: str, ndim: int, /) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}.")
    if result.dtype.kind not in "fciu" or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real or complex array.")
    return result.astype("float64", copy=False) if result.dtype.kind in "iu" else result


def _real_nonnegative_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 1
        or result.dtype.kind not in "fiu"
        or np.any(~np.isfinite(result))
        or np.any(result < 0)
    ):
        raise ValueError(f"{name} must be a finite nonnegative real vector.")
    return result.astype("float64", copy=False)


def _canonical_indices(
    values: Sequence[int], upper: int, name: str, /
) -> tuple[int, ...]:
    result = tuple(_nonnegative_integer(value, name) for value in values)
    if len(set(result)) != len(result) or any(value >= upper for value in result):
        raise ValueError(f"{name} must be unique indices in [0, {upper}).")
    return tuple(sorted(result))


class AffineSlice(StrictModule, NonTrainableState):
    """An ordered finite complex-affine slice ``linear @ x + offset = 0``."""

    linear: Array
    offset: Array
    ambient_dimension: int = eqx.field(static=True)
    codimension: int = eqx.field(static=True)
    slice_id: str = eqx.field(static=True)

    def __init__(self, linear: ArrayLike, offset: ArrayLike, /):
        linear_ = _finite_array(linear, "linear", 2)
        offset_ = _finite_array(offset, "offset", 1)
        if offset_.shape != (linear_.shape[0],):
            raise ValueError("Slice offset length must equal the number of slice rows.")
        dtype = np.result_type(linear_.dtype, offset_.dtype)
        self.linear = jnp.asarray(linear_, dtype=dtype)
        self.offset = jnp.asarray(offset_, dtype=dtype)
        self.ambient_dimension = linear_.shape[1]
        self.codimension = linear_.shape[0]
        self.slice_id = canonical_fingerprint(
            {
                "kind": "polynomial-affine-slice",
                "linear": linear_.astype(dtype, copy=False),
                "offset": offset_.astype(dtype, copy=False),
            }
        )


class PathRecord(StrictModule, NonTrainableState):
    """One path's source/endpoint inventory entry and observed disposition."""

    path_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    target_index: int | None = eqx.field(static=True)
    status: PathStatus = eqx.field(static=True)
    residual_norm: float | None = eqx.field(static=True)
    diagnostic: str = eqx.field(static=True)

    def __init__(
        self,
        path_id: str,
        batch_id: str,
        source_index: int,
        status: PathStatus | str,
        /,
        *,
        target_index: int | None = None,
        residual_norm: float | None = None,
        diagnostic: str = "",
    ):
        path = _identifier(path_id, "path_id")
        batch = _identifier(batch_id, "batch_id")
        source = _nonnegative_integer(source_index, "source_index")
        try:
            status_ = PathStatus(status)
        except ValueError as error:
            raise ValueError(f"Unknown continuation path status {status!r}.") from error
        target = (
            None
            if target_index is None
            else _nonnegative_integer(target_index, "target_index")
        )
        residual = None if residual_norm is None else float(residual_norm)
        if residual is not None and (not isfinite(residual) or residual < 0.0):
            raise ValueError("residual_norm must be finite and nonnegative or None.")
        diagnostic_ = str(diagnostic)
        if status_ is PathStatus.SUCCESS:
            if target is None or residual is None:
                raise ValueError(
                    "Successful paths require a target index and residual norm."
                )
        elif target is not None:
            raise ValueError("Unsuccessful paths cannot assert a target index.")
        if status_ is PathStatus.NOT_ATTEMPTED and residual is not None:
            raise ValueError("Unattempted paths cannot carry a residual norm.")
        self.path_id = path
        self.batch_id = batch
        self.source_index = source
        self.target_index = target
        self.status = status_
        self.residual_norm = residual
        self.diagnostic = diagnostic_


class PathInventory(StrictModule, NonTrainableState):
    """Exact disposition inventory for a bounded set of requested paths."""

    expected_path_ids: tuple[str, ...] = eqx.field(static=True)
    records: tuple[PathRecord, ...]
    path_capacity: int = eqx.field(static=True)
    budget_exhausted: bool = eqx.field(static=True)
    inventory_id: str = eqx.field(static=True)

    def __init__(
        self,
        expected_path_ids: Sequence[str],
        records: Sequence[PathRecord],
        /,
        *,
        path_capacity: int,
        budget_exhausted: bool = False,
    ):
        expected = tuple(
            _identifier(value, "expected path ID") for value in expected_path_ids
        )
        records_ = tuple(records)
        capacity = _positive_integer(path_capacity, "path_capacity")
        if len(set(expected)) != len(expected):
            raise ValueError("Expected path IDs must be unique.")
        if len(expected) > capacity:
            raise ValueError("Expected path inventory exceeds path_capacity.")
        if any(not isinstance(record, PathRecord) for record in records_):
            raise TypeError("records must contain PathRecord values.")
        actual = tuple(record.path_id for record in records_)
        if len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError(
                "Path records must exactly inventory every expected path ID."
            )
        by_id = {record.path_id: record for record in records_}
        ordered = tuple(by_id[value] for value in expected)
        exhausted = bool(budget_exhausted)
        has_unattempted = any(
            record.status is PathStatus.NOT_ATTEMPTED for record in ordered
        )
        if has_unattempted and not exhausted:
            raise ValueError("Unattempted paths require explicit budget exhaustion.")
        if exhausted and not has_unattempted:
            raise ValueError(
                "budget_exhausted requires at least one unattempted path record."
            )
        self.expected_path_ids = expected
        self.records = ordered
        self.path_capacity = capacity
        self.budget_exhausted = exhausted
        self.inventory_id = canonical_fingerprint(
            {
                "kind": "continuation-path-inventory",
                "expected": expected,
                "capacity": capacity,
                "budget_exhausted": exhausted,
                "records": [
                    {
                        "path_id": record.path_id,
                        "batch_id": record.batch_id,
                        "source_index": record.source_index,
                        "target_index": record.target_index,
                        "status": record.status.value,
                        "residual_norm": record.residual_norm,
                        "diagnostic": record.diagnostic,
                    }
                    for record in ordered
                ],
            }
        )

    @property
    def successful(self) -> bool:
        return not self.budget_exhausted and all(
            record.status is PathStatus.SUCCESS for record in self.records
        )

    @property
    def successful_count(self) -> int:
        return sum(record.status is PathStatus.SUCCESS for record in self.records)


class WitnessSet(StrictModule, NonTrainableState):
    """A finite numerical intersection with one declared generic affine slice."""

    system_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    slice: AffineSlice
    points: Array
    residual_norms: Array
    degree: int = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        system_id: str,
        dimension: int,
        slice_matrix: ArrayLike,
        slice_offset: ArrayLike,
        points: ArrayLike,
        residual_norms: ArrayLike,
        /,
    ):
        system = _identifier(system_id, "system_id")
        dimension_ = _nonnegative_integer(dimension, "dimension")
        slice_ = AffineSlice(slice_matrix, slice_offset)
        points_ = _finite_array(points, "points", 2)
        residuals = _real_nonnegative_vector(residual_norms, "residual_norms")
        if dimension_ > slice_.ambient_dimension:
            raise ValueError("Witness dimension cannot exceed ambient dimension.")
        if slice_.codimension != dimension_:
            raise ValueError(
                "A dimension-d witness set requires exactly d affine slice rows."
            )
        if points_.shape[1] != slice_.ambient_dimension:
            raise ValueError("Witness point width must equal slice ambient dimension.")
        if points_.shape[0] < 1 or residuals.shape != (points_.shape[0],):
            raise ValueError(
                "Witness sets require nonempty points and one residual per point."
            )
        slice_residual = points_ @ np.asarray(slice_.linear).T + np.asarray(slice_.offset)
        scale = 1.0 + np.max(np.abs(points_), axis=1, initial=0.0)
        tolerance = 64.0 * np.finfo(np.asarray(points_).real.dtype).eps
        if slice_.codimension and np.any(
            np.max(np.abs(slice_residual), axis=1) > tolerance * scale
        ):
            raise ValueError("Witness points do not lie on the declared affine slice.")
        dtype = np.result_type(points_.dtype, np.asarray(slice_.linear).dtype)
        self.system_id = system
        self.dimension = dimension_
        self.ambient_dimension = slice_.ambient_dimension
        self.slice = slice_
        self.points = jnp.asarray(points_, dtype=dtype)
        self.residual_norms = jnp.asarray(residuals)
        self.degree = points_.shape[0]
        self.witness_id = canonical_fingerprint(
            {
                "kind": "polynomial-witness-set",
                "system": system,
                "dimension": dimension_,
                "slice": slice_.slice_id,
                "points": points_.astype(dtype, copy=False),
                "residual_norms": residuals,
            }
        )


class MultigradedWitnessCollection(StrictModule, NonTrainableState):
    """Witness sets indexed by multidegrees for one variable-group partition."""

    system_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    groups: tuple[PolynomialVariableGroup, ...]
    multidegrees: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    witness_sets: tuple[WitnessSet, ...]
    collection_id: str = eqx.field(static=True)

    def __init__(
        self,
        system_id: str,
        ambient_dimension: int,
        groups: Sequence[PolynomialVariableGroup],
        entries: Sequence[tuple[Sequence[int], WitnessSet]],
        /,
    ):
        system = _identifier(system_id, "system_id")
        ambient = _positive_integer(ambient_dimension, "ambient_dimension")
        groups_ = tuple(groups)
        entries_ = tuple(entries)
        if not groups_ or any(
            not isinstance(group, PolynomialVariableGroup) for group in groups_
        ):
            raise TypeError("groups must be nonempty PolynomialVariableGroup values.")
        labels = tuple(group.label for group in groups_)
        if len(set(labels)) != len(labels):
            raise ValueError("Polynomial variable-group labels must be unique.")
        partition = tuple(
            variable for group in groups_ for variable in group.variable_indices
        )
        if sorted(partition) != list(range(ambient)):
            raise ValueError("Variable groups must partition the ambient coordinates.")
        if not entries_:
            raise ValueError("A witness collection must contain at least one entry.")
        multidegrees: list[tuple[int, ...]] = []
        witnesses: list[WitnessSet] = []
        for multidegree, witness in entries_:
            if not isinstance(witness, WitnessSet):
                raise TypeError(
                    "Witness collection entries must contain WitnessSet values."
                )
            multi = tuple(
                _nonnegative_integer(value, "multidegree") for value in multidegree
            )
            if (
                len(multi) != len(groups_)
                or sum(multi) != witness.dimension
                or any(
                    count > group.dimension
                    for count, group in zip(multi, groups_, strict=True)
                )
            ):
                raise ValueError(
                    "Each multidegree must respect group dimensions and sum to the witness dimension."
                )
            if witness.system_id != system or witness.ambient_dimension != ambient:
                raise ValueError(
                    "Witness collection system or ambient identity mismatch."
                )
            row = 0
            linear = np.asarray(witness.slice.linear)
            offset = np.asarray(witness.slice.offset)
            points = np.asarray(witness.points)
            for count, group in zip(multi, groups_, strict=True):
                allowed = set(group.variable_indices)
                for row_index in range(row, row + count):
                    values = linear[row_index]
                    if any(
                        coordinate not in allowed and values[coordinate] != 0
                        for coordinate in range(ambient)
                    ):
                        raise ValueError(
                            "Multigraded slice rows must be supported on their variable group."
                        )
                    if group.geometry == "projective" and offset[row_index] != 0:
                        raise ValueError(
                            "Projective multigraded slice rows must be homogeneous."
                        )
                if group.geometry == "projective" and np.any(
                    np.all(points[:, group.variable_indices] == 0, axis=1)
                ):
                    raise ValueError(
                        "Projective witness points require a nonzero homogeneous block."
                    )
                row += count
            multidegrees.append(multi)
            witnesses.append(witness)
        if len(set(multidegrees)) != len(multidegrees):
            raise ValueError("Witness collection multidegrees must be unique.")
        order = tuple(sorted(range(len(multidegrees)), key=multidegrees.__getitem__))
        multidegrees_ = tuple(multidegrees[position] for position in order)
        witnesses_ = tuple(witnesses[position] for position in order)
        self.system_id = system
        self.ambient_dimension = ambient
        self.groups = groups_
        self.multidegrees = multidegrees_
        self.witness_sets = witnesses_
        self.collection_id = canonical_fingerprint(
            {
                "kind": "multigraded-witness-collection",
                "system": system,
                "ambient_dimension": ambient,
                "groups": [
                    {
                        "label": group.label,
                        "variables": group.variable_indices,
                        "geometry": group.geometry,
                    }
                    for group in groups_
                ],
                "entries": [
                    {"multidegree": multi, "witness": witness.witness_id}
                    for multi, witness in zip(multidegrees_, witnesses_, strict=True)
                ],
            }
        )

    def witness(self, multidegree: Sequence[int], /) -> WitnessSet:
        key = tuple(index(value) for value in multidegree)
        for candidate, witness in zip(self.multidegrees, self.witness_sets, strict=True):
            if candidate == key:
                return witness
        raise KeyError(key)


class PseudoWitnessSet(StrictModule, NonTrainableState):
    """Finite graph slice supporting a numerical image-degree observation."""

    source_system_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    source_dimension: int = eqx.field(static=True)
    image_dimension: int = eqx.field(static=True)
    source_ambient_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    source_slice: AffineSlice
    image_slice: AffineSlice
    source_points: Array
    image_points: Array
    residual_norms: Array
    image_degree: int = eqx.field(static=True)
    fiber_degree: int = eqx.field(static=True)
    pseudo_witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_system_id: str,
        map_id: str,
        source_dimension: int,
        image_dimension: int,
        source_slice_matrix: ArrayLike,
        source_slice_offset: ArrayLike,
        image_slice_matrix: ArrayLike,
        image_slice_offset: ArrayLike,
        source_points: ArrayLike,
        image_points: ArrayLike,
        residual_norms: ArrayLike,
        /,
        *,
        image_degree: int,
    ):
        system = _identifier(source_system_id, "source_system_id")
        map_ = _identifier(map_id, "map_id")
        source_dimension_ = _nonnegative_integer(source_dimension, "source_dimension")
        image_dimension_ = _nonnegative_integer(image_dimension, "image_dimension")
        if image_dimension_ > source_dimension_:
            raise ValueError("image_dimension cannot exceed source_dimension.")
        source_slice = AffineSlice(source_slice_matrix, source_slice_offset)
        image_slice = AffineSlice(image_slice_matrix, image_slice_offset)
        source_points_ = _finite_array(source_points, "source_points", 2)
        image_points_ = _finite_array(image_points, "image_points", 2)
        residuals = _real_nonnegative_vector(residual_norms, "residual_norms")
        point_count = source_points_.shape[0]
        degree = _positive_integer(image_degree, "image_degree")
        fiber_dimension = source_dimension_ - image_dimension_
        if source_slice.codimension != fiber_dimension:
            raise ValueError(
                "Source slice codimension must equal generic fiber dimension."
            )
        if image_slice.codimension != image_dimension_:
            raise ValueError("Image slice codimension must equal image_dimension.")
        if source_points_.shape[1] != source_slice.ambient_dimension:
            raise ValueError("Source point and source slice dimensions disagree.")
        if image_points_.shape[1] != image_slice.ambient_dimension:
            raise ValueError("Image point and image slice dimensions disagree.")
        if (
            point_count < 1
            or image_points_.shape[0] != point_count
            or residuals.shape != (point_count,)
            or point_count % degree != 0
        ):
            raise ValueError(
                "Pseudo-witness points/residuals must align and factor by image_degree."
            )
        source_error = source_points_ @ np.asarray(source_slice.linear).T + np.asarray(
            source_slice.offset
        )
        image_error = image_points_ @ np.asarray(image_slice.linear).T + np.asarray(
            image_slice.offset
        )
        epsilon = (
            64.0
            * np.finfo(
                np.result_type(source_points_.real.dtype, image_points_.real.dtype)
            ).eps
        )
        if (
            source_error.size
            and np.max(np.abs(source_error))
            > epsilon * (1 + np.max(np.abs(source_points_)))
        ) or (
            image_error.size
            and np.max(np.abs(image_error))
            > epsilon * (1 + np.max(np.abs(image_points_)))
        ):
            raise ValueError("Pseudo-witness points do not lie on the declared slices.")
        self.source_system_id = system
        self.map_id = map_
        self.source_dimension = source_dimension_
        self.image_dimension = image_dimension_
        self.source_ambient_dimension = source_slice.ambient_dimension
        self.target_dimension = image_slice.ambient_dimension
        self.source_slice = source_slice
        self.image_slice = image_slice
        self.source_points = jnp.asarray(source_points_)
        self.image_points = jnp.asarray(image_points_)
        self.residual_norms = jnp.asarray(residuals)
        self.image_degree = degree
        self.fiber_degree = point_count // degree
        self.pseudo_witness_id = canonical_fingerprint(
            {
                "kind": "polynomial-pseudo-witness-set",
                "source_system": system,
                "map": map_,
                "source_dimension": source_dimension_,
                "image_dimension": image_dimension_,
                "source_slice": source_slice.slice_id,
                "image_slice": image_slice.slice_id,
                "source_points": source_points_,
                "image_points": image_points_,
                "residual_norms": residuals,
                "image_degree": degree,
            }
        )


class MonodromyEvidence(StrictModule, NonTrainableState):
    """Validated loop permutations and path evidence, not an irreducibility proof."""

    witness_set_id: str = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    attempted_loop_ids: tuple[str, ...] = eqx.field(static=True)
    completed_loop_ids: tuple[str, ...] = eqx.field(static=True)
    permutations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    orbits: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    paths: PathInventory
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        witness_set_id: str,
        point_count: int,
        attempted_loop_ids: Sequence[str],
        completed: Sequence[tuple[str, Sequence[int]]],
        paths: PathInventory,
        /,
    ):
        witness = _identifier(witness_set_id, "witness_set_id")
        count = _positive_integer(point_count, "point_count")
        attempted = tuple(
            _identifier(loop, "attempted loop ID") for loop in attempted_loop_ids
        )
        completed_ = tuple(completed)
        if len(set(attempted)) != len(attempted):
            raise ValueError("Attempted monodromy loop IDs must be unique.")
        if not isinstance(paths, PathInventory):
            raise TypeError("paths must be a PathInventory.")
        completed_ids: list[str] = []
        permutations: list[tuple[int, ...]] = []
        for loop_id, permutation in completed_:
            loop = _identifier(loop_id, "completed loop ID")
            permutation_ = tuple(index(value) for value in permutation)
            if (
                loop not in attempted
                or loop in completed_ids
                or len(permutation_) != count
                or sorted(permutation_) != list(range(count))
            ):
                raise ValueError(
                    "Completed loops require unique attempted IDs and valid permutations."
                )
            records = tuple(record for record in paths.records if record.batch_id == loop)
            if (
                len(records) != count
                or sorted(record.source_index for record in records) != list(range(count))
                or any(record.status is not PathStatus.SUCCESS for record in records)
            ):
                raise ValueError(
                    "Each completed permutation requires one successful path per point."
                )
            by_source = {record.source_index: record.target_index for record in records}
            if tuple(by_source[source] for source in range(count)) != permutation_:
                raise ValueError(
                    "Permutation targets disagree with path endpoint inventory."
                )
            completed_ids.append(loop)
            permutations.append(permutation_)
        if any(record.batch_id not in attempted for record in paths.records):
            raise ValueError("Monodromy path batches must name attempted loop IDs.")
        expected_count = count * len(attempted)
        if len(paths.records) != expected_count:
            raise ValueError("Monodromy inventory requires one path per point and loop.")
        adjacency = [set((value,)) for value in range(count)]
        for permutation in permutations:
            for source, target in enumerate(permutation):
                adjacency[source].add(target)
                adjacency[target].add(source)
        unseen = set(range(count))
        orbits: list[tuple[int, ...]] = []
        while unseen:
            root = min(unseen)
            stack = [root]
            orbit: set[int] = set()
            while stack:
                current = stack.pop()
                if current in orbit:
                    continue
                orbit.add(current)
                stack.extend(adjacency[current] - orbit)
            unseen -= orbit
            orbits.append(tuple(sorted(orbit)))
        orbits_ = tuple(sorted(orbits, key=lambda values: values[0]))
        completed_ids_ = tuple(completed_ids)
        permutations_ = tuple(permutations)
        self.witness_set_id = witness
        self.point_count = count
        self.attempted_loop_ids = attempted
        self.completed_loop_ids = completed_ids_
        self.permutations = permutations_
        self.orbits = orbits_
        self.paths = paths
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-monodromy-evidence",
                "witness": witness,
                "point_count": count,
                "attempted_loops": attempted,
                "completed_loops": completed_ids_,
                "permutations": permutations_,
                "orbits": orbits_,
                "paths": paths.inventory_id,
            }
        )

    @property
    def transitive(self) -> bool:
        return len(self.orbits) == 1


class TraceTestEvidence(StrictModule, NonTrainableState):
    """Observed affine trace residual for one proposed witness-point subset."""

    witness_set_id: str = eqx.field(static=True)
    point_indices: tuple[int, ...] = eqx.field(static=True)
    sample_parameters: Array
    trace_values: Array
    affine_fit_residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    paths: PathInventory
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        witness_set_id: str,
        point_indices: Sequence[int],
        sample_parameters: ArrayLike,
        trace_values: ArrayLike,
        affine_fit_residual: float,
        tolerance: float,
        paths: PathInventory,
        /,
    ):
        witness = _identifier(witness_set_id, "witness_set_id")
        points = tuple(
            sorted(_nonnegative_integer(v, "point index") for v in point_indices)
        )
        if not points or len(set(points)) != len(points):
            raise ValueError("Trace-test point indices must be nonempty and unique.")
        parameters = _finite_array(sample_parameters, "sample_parameters", 1)
        traces = _finite_array(trace_values, "trace_values", 2)
        residual = float(affine_fit_residual)
        tolerance_ = float(tolerance)
        if parameters.size < 3 or traces.shape[0] != parameters.size:
            raise ValueError("Trace tests require at least three aligned samples.")
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("affine_fit_residual must be finite and nonnegative.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if not isinstance(paths, PathInventory):
            raise TypeError("paths must be a PathInventory.")
        batches: defaultdict[str, list[int]] = defaultdict(list)
        for record in paths.records:
            batches[record.batch_id].append(record.source_index)
        if len(batches) != parameters.size or any(
            tuple(sorted(source_indices)) != points for source_indices in batches.values()
        ):
            raise ValueError(
                "Trace-test paths require one exact point-subset batch per sample."
            )
        self.witness_set_id = witness
        self.point_indices = points
        self.sample_parameters = jnp.asarray(parameters)
        self.trace_values = jnp.asarray(traces)
        self.affine_fit_residual = residual
        self.tolerance = tolerance_
        self.finite = True
        self.passed = paths.successful and residual <= tolerance_
        self.paths = paths
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-trace-test-evidence",
                "witness": witness,
                "point_indices": points,
                "sample_parameters": parameters,
                "trace_values": traces,
                "affine_fit_residual": residual,
                "tolerance": tolerance_,
                "paths": paths.inventory_id,
                "passed": self.passed,
            }
        )


class NumericalComponent(StrictModule, NonTrainableState):
    """A proposed numerical component; no exact irreducibility claim is implied."""

    witness_set_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    point_indices: tuple[int, ...] = eqx.field(static=True)
    monodromy_evidence_id: str = eqx.field(static=True)
    trace_evidence_id: str = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        witness_set: WitnessSet,
        point_indices: Sequence[int],
        monodromy: MonodromyEvidence,
        trace_test: TraceTestEvidence,
        /,
    ):
        if not isinstance(witness_set, WitnessSet):
            raise TypeError("witness_set must be a WitnessSet.")
        if not isinstance(monodromy, MonodromyEvidence):
            raise TypeError("monodromy must be MonodromyEvidence.")
        if not isinstance(trace_test, TraceTestEvidence):
            raise TypeError("trace_test must be TraceTestEvidence.")
        points = _canonical_indices(
            point_indices, witness_set.degree, "component point indices"
        )
        if not points:
            raise ValueError("A numerical component requires at least one witness point.")
        if (
            monodromy.witness_set_id != witness_set.witness_id
            or monodromy.point_count != witness_set.degree
            or trace_test.witness_set_id != witness_set.witness_id
            or trace_test.point_indices != points
        ):
            raise ValueError(
                "Component witness and numerical evidence identities disagree."
            )
        if points not in monodromy.orbits:
            raise ValueError(
                "Component point indices must be one observed monodromy orbit."
            )
        self.witness_set_id = witness_set.witness_id
        self.system_id = witness_set.system_id
        self.dimension = witness_set.dimension
        self.point_indices = points
        self.monodromy_evidence_id = monodromy.evidence_id
        self.trace_evidence_id = trace_test.evidence_id
        self.component_id = canonical_fingerprint(
            {
                "kind": "candidate-numerical-component",
                "witness": witness_set.witness_id,
                "points": points,
                "monodromy": monodromy.evidence_id,
                "trace_test": trace_test.evidence_id,
            }
        )

    @property
    def degree(self) -> int:
        return len(self.point_indices)


class RegenerationStage(StrictModule, NonTrainableState):
    """One equation subset and expected dimension in a regeneration plan."""

    label: str = eqx.field(static=True)
    equation_indices: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        equation_indices: Sequence[int],
        dimension: int,
        /,
    ):
        label_ = _identifier(label, "stage label")
        equations = tuple(
            sorted(
                _nonnegative_integer(value, "equation index")
                for value in equation_indices
            )
        )
        if len(set(equations)) != len(equations):
            raise ValueError("Regeneration stage equations must be unique.")
        dimension_ = _nonnegative_integer(dimension, "dimension")
        self.label = label_
        self.equation_indices = equations
        self.dimension = dimension_
        self.stage_id = canonical_fingerprint(
            {
                "kind": "polynomial-regeneration-stage",
                "label": label_,
                "equations": equations,
                "dimension": dimension_,
            }
        )


class RegenerationEdge(StrictModule, NonTrainableState):
    """One bounded equation-addition transition between regeneration stages."""

    source_stage_id: str = eqx.field(static=True)
    target_stage_id: str = eqx.field(static=True)
    equation_index: int = eqx.field(static=True)
    expected_path_ids: tuple[str, ...] = eqx.field(static=True)
    path_capacity: int = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: RegenerationStage,
        target: RegenerationStage,
        equation_index: int,
        expected_path_ids: Sequence[str],
        /,
        *,
        path_capacity: int,
    ):
        if not isinstance(source, RegenerationStage) or not isinstance(
            target, RegenerationStage
        ):
            raise TypeError("source and target must be RegenerationStage values.")
        equation = _nonnegative_integer(equation_index, "equation_index")
        expected = tuple(_identifier(value, "path ID") for value in expected_path_ids)
        capacity = _positive_integer(path_capacity, "path_capacity")
        if source.stage_id == target.stage_id:
            raise ValueError("Regeneration edges require distinct stages.")
        if (
            equation in source.equation_indices
            or tuple(sorted((*source.equation_indices, equation)))
            != target.equation_indices
        ):
            raise ValueError("A regeneration edge must add exactly its named equation.")
        if target.dimension not in (source.dimension, max(0, source.dimension - 1)):
            raise ValueError(
                "A regeneration edge may lower expected dimension by at most one."
            )
        if (
            not expected
            or len(set(expected)) != len(expected)
            or len(expected) > capacity
        ):
            raise ValueError(
                "Regeneration path IDs must be unique, nonempty, and bounded."
            )
        self.source_stage_id = source.stage_id
        self.target_stage_id = target.stage_id
        self.equation_index = equation
        self.expected_path_ids = expected
        self.path_capacity = capacity
        self.edge_id = canonical_fingerprint(
            {
                "kind": "polynomial-regeneration-edge",
                "source": source.stage_id,
                "target": target.stage_id,
                "equation": equation,
                "expected_paths": expected,
                "path_capacity": capacity,
            }
        )


class RegenerationPlan(StrictModule, NonTrainableState):
    """A finite acyclic equation-by-equation regeneration graph."""

    system_id: str = eqx.field(static=True)
    stages: tuple[RegenerationStage, ...]
    edges: tuple[RegenerationEdge, ...]
    path_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system_id: str,
        stages: Sequence[RegenerationStage],
        edges: Sequence[RegenerationEdge],
        /,
        *,
        path_capacity: int,
    ):
        system = _identifier(system_id, "system_id")
        stages_ = tuple(stages)
        edges_ = tuple(edges)
        capacity = _positive_integer(path_capacity, "path_capacity")
        if not stages_ or any(
            not isinstance(stage, RegenerationStage) for stage in stages_
        ):
            raise TypeError("stages must contain RegenerationStage values.")
        if any(not isinstance(edge, RegenerationEdge) for edge in edges_):
            raise TypeError("edges must contain RegenerationEdge values.")
        stage_ids = tuple(stage.stage_id for stage in stages_)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("Regeneration stage identities must be unique.")
        positions = {stage_id: position for position, stage_id in enumerate(stage_ids)}
        incoming: defaultdict[str, int] = defaultdict(int)
        path_ids: list[str] = []
        for edge in edges_:
            if (
                edge.source_stage_id not in positions
                or edge.target_stage_id not in positions
            ):
                raise ValueError("Regeneration edges must reference plan stages.")
            if positions[edge.source_stage_id] >= positions[edge.target_stage_id]:
                raise ValueError("Regeneration stages must be in topological order.")
            incoming[edge.target_stage_id] += 1
            path_ids.extend(edge.expected_path_ids)
        if any(incoming[stage_id] == 0 for stage_id in stage_ids[1:]):
            raise ValueError(
                "Every noninitial regeneration stage requires an incoming edge."
            )
        if incoming[stage_ids[0]]:
            raise ValueError(
                "The initial regeneration stage cannot have an incoming edge."
            )
        if len(set(path_ids)) != len(path_ids) or len(path_ids) > capacity:
            raise ValueError(
                "Plan-wide regeneration path inventory is not unique or bounded."
            )
        self.system_id = system
        self.stages = stages_
        self.edges = edges_
        self.path_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-regeneration-plan",
                "system": system,
                "stages": stage_ids,
                "edges": [edge.edge_id for edge in edges_],
                "path_capacity": capacity,
            }
        )


class NumericalDecompositionResult(StrictModule, NonTrainableState):
    """Audited numerical decomposition evidence with deliberately qualified claims."""

    witness_collection: MultigradedWitnessCollection
    components: tuple[NumericalComponent, ...]
    monodromy: tuple[MonodromyEvidence, ...]
    trace_tests: tuple[TraceTestEvidence, ...]
    path_inventories: tuple[PathInventory, ...]
    status: DecompositionStatus = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        witness_collection: MultigradedWitnessCollection,
        components: Sequence[NumericalComponent],
        monodromy: Sequence[MonodromyEvidence],
        trace_tests: Sequence[TraceTestEvidence],
        path_inventories: Sequence[PathInventory] = (),
        /,
    ):
        if not isinstance(witness_collection, MultigradedWitnessCollection):
            raise TypeError("witness_collection must be MultigradedWitnessCollection.")
        components_ = tuple(components)
        monodromy_ = tuple(monodromy)
        traces_ = tuple(trace_tests)
        inventories = tuple(path_inventories)
        if any(not isinstance(value, NumericalComponent) for value in components_):
            raise TypeError("components must contain NumericalComponent values.")
        if any(not isinstance(value, MonodromyEvidence) for value in monodromy_):
            raise TypeError("monodromy must contain MonodromyEvidence values.")
        if any(not isinstance(value, TraceTestEvidence) for value in traces_):
            raise TypeError("trace_tests must contain TraceTestEvidence values.")
        if any(not isinstance(value, PathInventory) for value in inventories):
            raise TypeError("path_inventories must contain PathInventory values.")
        witnesses = {
            witness.witness_id: witness for witness in witness_collection.witness_sets
        }
        monodromy_by_witness = {value.witness_set_id: value for value in monodromy_}
        if len(monodromy_by_witness) != len(monodromy_):
            raise ValueError("At most one monodromy record is allowed per witness set.")
        traces_by_id = {value.evidence_id: value for value in traces_}
        if len(traces_by_id) != len(traces_):
            raise ValueError("Trace-test evidence identities must be unique.")
        partition: defaultdict[str, list[int]] = defaultdict(list)
        for component in components_:
            witness = witnesses.get(component.witness_set_id)
            monodromy_record = monodromy_by_witness.get(component.witness_set_id)
            trace = traces_by_id.get(component.trace_evidence_id)
            if (
                witness is None
                or monodromy_record is None
                or trace is None
                or component.system_id != witness_collection.system_id
                or component.monodromy_evidence_id != monodromy_record.evidence_id
            ):
                raise ValueError("Numerical component evidence identity mismatch.")
            partition[component.witness_set_id].extend(component.point_indices)
        partition_complete = all(
            sorted(partition[witness.witness_id]) == list(range(witness.degree))
            for witness in witness_collection.witness_sets
        )
        evidence_paths_by_id: dict[str, PathInventory] = {}
        for inventory in (
            *(value.paths for value in monodromy_),
            *(value.paths for value in traces_),
            *inventories,
        ):
            evidence_paths_by_id.setdefault(inventory.inventory_id, inventory)
        evidence_paths = tuple(evidence_paths_by_id.values())
        if any(value.budget_exhausted for value in evidence_paths):
            status = DecompositionStatus.BUDGET_EXHAUSTED
        elif any(not value.successful for value in evidence_paths):
            status = DecompositionStatus.PARTIAL_PATH_FAILURE
        elif any(not value.passed for value in traces_):
            status = DecompositionStatus.TRACE_TEST_FAILED
        elif (
            not partition_complete
            or set(monodromy_by_witness) != set(witnesses)
            or len(traces_) != len(components_)
        ):
            status = DecompositionStatus.INCOMPLETE
        else:
            status = DecompositionStatus.EVIDENCE_COMPLETE
        self.witness_collection = witness_collection
        self.components = components_
        self.monodromy = monodromy_
        self.trace_tests = traces_
        self.path_inventories = evidence_paths
        self.status = status
        self.result_id = canonical_fingerprint(
            {
                "kind": "qualified-numerical-decomposition-result",
                "witness_collection": witness_collection.collection_id,
                "components": [value.component_id for value in components_],
                "monodromy": [value.evidence_id for value in monodromy_],
                "trace_tests": [value.evidence_id for value in traces_],
                "path_inventories": [value.inventory_id for value in evidence_paths],
                "status": status.value,
            }
        )

    @property
    def evidence_complete(self) -> bool:
        """Whether all declared numerical checks completed and passed."""

        return self.status is DecompositionStatus.EVIDENCE_COMPLETE

    @property
    def claim(self) -> str:
        return "numerical-decomposition-evidence-not-exact-irreducibility-or-completeness"


__all__ = [
    "AffineSlice",
    "DecompositionStatus",
    "MonodromyEvidence",
    "MultigradedWitnessCollection",
    "NumericalComponent",
    "NumericalDecompositionResult",
    "PathInventory",
    "PathRecord",
    "PathStatus",
    "PseudoWitnessSet",
    "RegenerationEdge",
    "RegenerationPlan",
    "RegenerationStage",
    "TraceTestEvidence",
    "WitnessSet",
]
