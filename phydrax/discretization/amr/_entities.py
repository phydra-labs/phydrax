#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded canonical cubical entity complexes over variable AMR patch unions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from .._topology import CellComplexTopology, EntitySet, OrientedIncidence
from ._patches import BlockHierarchyCapacityPlan
from ._variable import VariablePatchHierarchyTopology


EntityKey = tuple[tuple[int, ...], tuple[int, ...]]


def _entity_shape(
    envelope_shape: tuple[int, ...], orientation: tuple[int, ...], /
) -> tuple[int, ...]:
    tangent = frozenset(orientation)
    return tuple(
        extent if axis in tangent else extent + 1
        for axis, extent in enumerate(envelope_shape)
    )


def _entity_key(orientation: tuple[int, ...], coordinate: Sequence[int], /) -> EntityKey:
    return tuple(orientation), tuple(coordinate)


def _canonical_entity_key(
    orientation: tuple[int, ...],
    coordinate: Sequence[int],
    global_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> EntityKey:
    values = list(int(value) for value in coordinate)
    tangent = frozenset(orientation)
    for axis, extent in enumerate(global_shape):
        if axis in tangent:
            if values[axis] < 0 or values[axis] >= extent:
                raise ValueError(
                    "Tangential entity origins must lie in the cell lattice."
                )
        elif periodic[axis]:
            values[axis] %= extent
        elif values[axis] < 0 or values[axis] > extent:
            raise ValueError("Point-like entity coordinates are outside the grid.")
    return _entity_key(orientation, values)


def _stable_entity_id(
    level: int,
    orientation: tuple[int, ...],
    coordinate: tuple[int, ...],
    global_shape: tuple[int, ...],
    level_count: int,
    /,
) -> int:
    """Encode one level/orientation/reference-coordinate key without collisions."""
    dimension = len(global_shape)
    if (
        len(coordinate) != dimension
        or any(axis < 0 or axis >= dimension for axis in orientation)
        or tuple(sorted(set(orientation))) != orientation
        or any(
            value < 0 or value > global_shape[axis]
            for axis, value in enumerate(coordinate)
        )
    ):
        raise ValueError("Variable patch entity key is outside the reference domain.")
    level_bits = max(1, (int(level_count) - 1).bit_length())
    coordinate_bits = tuple(max(1, int(maximum).bit_length()) for maximum in global_shape)
    total_bits = level_bits + dimension + sum(coordinate_bits)
    if total_bits > 62:
        raise ValueError("Variable patch semantic entity IDs exceed signed int64 range.")
    orientation_mask = sum(1 << axis for axis in orientation)
    identifier = int(level)
    identifier = (identifier << dimension) | orientation_mask
    for value, bits in zip(coordinate, coordinate_bits, strict=True):
        identifier = (identifier << bits) | int(value)
    return identifier


def _cell_entities(
    cell: tuple[int, ...],
    dimension: int,
    global_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> tuple[tuple[int, EntityKey], ...]:
    result: list[tuple[int, EntityKey]] = []
    for degree in range(dimension + 1):
        for orientation in combinations(range(dimension), degree):
            transverse = tuple(
                axis for axis in range(dimension) if axis not in orientation
            )
            for bits in np.ndindex((2,) * len(transverse)):
                coordinate = list(cell)
                for axis, bit in zip(transverse, bits, strict=True):
                    coordinate[axis] += bit
                result.append(
                    (
                        degree,
                        _canonical_entity_key(
                            tuple(orientation),
                            coordinate,
                            global_shape,
                            periodic,
                        ),
                    )
                )
    return tuple(result)


class VariablePatchEntityBucketView(StrictModule, NonTrainableState):
    """Static local bucket view into one canonical degree/orientation entity set."""

    level: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    orientation: tuple[int, ...] = eqx.field(static=True)
    bucket: int = eqx.field(static=True)
    global_indices: Array
    orientation_signs: Array
    owned: Array
    valid: Array
    view_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        degree: int,
        orientation: Sequence[int],
        bucket: int,
        global_indices: Array,
        orientation_signs: Array,
        owned: Array,
        valid: Array,
        /,
    ):
        indices = jnp.asarray(global_indices, dtype=jnp.int32)
        signs = jnp.asarray(orientation_signs)
        owned_ = jnp.asarray(owned, dtype=jnp.bool_)
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        if (
            indices.shape != signs.shape
            or indices.shape != owned_.shape
            or indices.shape != valid_.shape
        ):
            raise ValueError("Variable patch entity view arrays must share one shape.")
        if bool(jnp.any(valid_ & (indices < 0))):
            raise ValueError("Valid patch entity routes require global indices.")
        if bool(jnp.any(~valid_ & (indices != -1))):
            raise ValueError("Inactive patch entity routes require -1 indices.")
        if bool(jnp.any(valid_ & (jnp.abs(signs) != 1))):
            raise ValueError("Valid patch entity orientations must be ±1.")
        if bool(jnp.any(owned_ & ~valid_)):
            raise ValueError("Only valid patch entities may be owners.")
        self.level = int(level)
        self.degree = int(degree)
        self.orientation = tuple(orientation)
        self.bucket = int(bucket)
        self.global_indices = indices
        self.orientation_signs = signs
        self.owned = owned_
        self.valid = valid_
        self.view_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-view",
                "level": self.level,
                "degree": self.degree,
                "orientation": self.orientation,
                "bucket": self.bucket,
                "shape": list(indices.shape),
                "indices": np.asarray(indices),
                "owned": np.asarray(owned_),
            }
        )


class VariablePatchEntityComplex(StrictModule, NonTrainableState):
    """One bounded canonical entity complex at one physical AMR level."""

    topology: VariablePatchHierarchyTopology
    level: int = eqx.field(static=True)
    complex: CellComplexTopology
    entity_keys: tuple[tuple[EntityKey | None, ...], ...] = eqx.field(static=True)
    views: tuple[VariablePatchEntityBucketView, ...]
    capacity: tuple[int, ...] = eqx.field(static=True)
    incidence_capacity: tuple[int, ...] = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        level: int,
        entity_capacity: Sequence[int],
        incidence_capacity: Sequence[int],
        /,
    ):
        if not isinstance(topology, VariablePatchHierarchyTopology):
            raise TypeError("Variable patch entity complex requires realized topology.")
        level_ = int(level)
        if level_ < 0 or level_ >= len(topology.levels):
            raise ValueError("Variable patch entity level is out of range.")
        dimension = topology.plan.levels[level_].dimension
        capacity = tuple(entity_capacity)
        if len(capacity) != dimension + 1 or any(value <= 0 for value in capacity):
            raise ValueError("Variable patch entity capacities require every degree.")
        incidence_bounds = tuple(incidence_capacity)
        if len(incidence_bounds) != dimension or any(
            value <= 0 for value in incidence_bounds
        ):
            raise ValueError(
                "Variable patch entity incidence capacities require every positive degree."
            )
        metadata = topology.levels[level_]
        global_shape = topology.plan.global_cell_shapes[level_]
        periodic = topology.plan.periodic_axes
        cells = tuple(
            coordinate
            for _, _, box in metadata.active_boxes()
            for coordinate in (
                tuple(
                    start + offset for start, offset in zip(box.lower, local, strict=True)
                )
                for local in np.ndindex(box.extent)
            )
        )
        if len(set(cells)) != len(cells):
            raise ValueError(
                "Variable patch cells must be uniquely owned before entity lowering."
            )
        keys_by_degree: list[set[EntityKey]] = [set() for _ in range(dimension + 1)]
        for cell in cells:
            for degree, key in _cell_entities(
                cell,
                dimension,
                global_shape,
                periodic,
            ):
                keys_by_degree[degree].add(key)
        ordered_keys = tuple(
            tuple(sorted(keys, key=lambda value: (value[0], value[1])))
            for keys in keys_by_degree
        )
        if any(
            len(keys) > bound for keys, bound in zip(ordered_keys, capacity, strict=True)
        ):
            raise ValueError("Variable patch entity capacity is exceeded.")
        entity_sets = []
        padded_keys: list[tuple[EntityKey | None, ...]] = []
        key_to_index: list[dict[EntityKey, int]] = []
        for degree, (keys, bound) in enumerate(zip(ordered_keys, capacity, strict=True)):
            active = np.zeros((bound,), dtype=np.bool_)
            active[: len(keys)] = True
            identifiers = np.full((bound,), -1, dtype=np.int64)
            identifiers[: len(keys)] = [
                _stable_entity_id(
                    level_,
                    key[0],
                    key[1],
                    topology.plan.global_cell_shapes[level_],
                    len(topology.levels),
                )
                for key in keys
            ]
            entity_sets.append(
                EntitySet(
                    f"variable-patch-level-{level_}-degree-{degree}",
                    degree,
                    identifiers,
                    active_mask=active,
                )
            )
            padded_keys.append(keys + (None,) * (bound - len(keys)))
            key_to_index.append({key: index for index, key in enumerate(keys)})
        incidences = []
        for degree in range(1, dimension + 1):
            source: list[int] = []
            target: list[int] = []
            incidence_signs: list[float] = []
            for upper_index, key in enumerate(ordered_keys[degree]):
                orientation, coordinate = key
                for position, axis in enumerate(orientation):
                    lower_orientation = tuple(
                        value for value in orientation if value != axis
                    )
                    lower_coordinate = list(coordinate)
                    upper_coordinate = list(coordinate)
                    upper_coordinate[axis] += 1
                    for face_coordinate, sign in (
                        (lower_coordinate, (-1.0) ** (position + 1)),
                        (upper_coordinate, (-1.0) ** position),
                    ):
                        lower_key = _canonical_entity_key(
                            lower_orientation,
                            face_coordinate,
                            global_shape,
                            periodic,
                        )
                        lower_index = key_to_index[degree - 1].get(lower_key)
                        if lower_index is None:
                            raise RuntimeError(
                                "Variable patch entity incidence lost a boundary key."
                            )
                        source.append(lower_index)
                        target.append(upper_index)
                        incidence_signs.append(sign)
            bound = incidence_bounds[degree - 1]
            if len(source) > bound:
                raise ValueError("Variable patch incidence route capacity is exceeded.")
            source_array = np.zeros((bound,), dtype=np.int32)
            target_array = np.zeros((bound,), dtype=np.int32)
            signs_array = np.zeros((bound,), dtype=np.float64)
            valid = np.zeros((bound,), dtype=np.bool_)
            source_array[: len(source)] = source
            target_array[: len(target)] = target
            signs_array[: len(incidence_signs)] = incidence_signs
            valid[: len(source)] = True
            incidences.append(
                OrientedIncidence(
                    degree,
                    entity_sets[degree - 1],
                    entity_sets[degree],
                    EdgeRelation(
                        source_array,
                        target_array,
                        source_size=capacity[degree - 1],
                        target_size=capacity[degree],
                        valid=valid,
                    ),
                    signs_array,
                )
            )
        complex_ = CellComplexTopology(entity_sets, incidences)
        owner_by_key: dict[EntityKey, str] = {}
        for _, _, box in metadata.active_boxes():
            for cell in (
                tuple(
                    start + offset for start, offset in zip(box.lower, local, strict=True)
                )
                for local in np.ndindex(box.extent)
            ):
                for _, key in _cell_entities(
                    cell,
                    dimension,
                    global_shape,
                    periodic,
                ):
                    owner_by_key[key] = min(
                        owner_by_key.get(key, box.box_id),
                        box.box_id,
                    )
        views: list[VariablePatchEntityBucketView] = []
        for degree in range(dimension + 1):
            for orientation in combinations(range(dimension), degree):
                for bucket_index, bucket in enumerate(
                    topology.plan.levels[level_].buckets
                ):
                    local_shape = _entity_shape(
                        bucket.signature.envelope_shape,
                        tuple(orientation),
                    )
                    indices = np.full(
                        (bucket.lane_capacity,) + local_shape,
                        -1,
                        dtype=np.int32,
                    )
                    orientation_signs = np.ones(indices.shape, dtype=np.float64)
                    owned = np.zeros(indices.shape, dtype=np.bool_)
                    valid = np.zeros(indices.shape, dtype=np.bool_)
                    for lane, box in enumerate(metadata.boxes[bucket_index]):
                        if box is None:
                            continue
                        actual_shape = _entity_shape(box.extent, tuple(orientation))
                        for local in np.ndindex(actual_shape):
                            coordinate = tuple(
                                start + offset
                                for start, offset in zip(box.lower, local, strict=True)
                            )
                            key = _canonical_entity_key(
                                tuple(orientation),
                                coordinate,
                                global_shape,
                                periodic,
                            )
                            index = key_to_index[degree][key]
                            route = (lane,) + local
                            indices[route] = index
                            valid[route] = True
                            owned[route] = owner_by_key[key] == box.box_id
                    views.append(
                        VariablePatchEntityBucketView(
                            level_,
                            degree,
                            orientation,
                            bucket_index,
                            indices,
                            orientation_signs,
                            owned,
                            valid,
                        )
                    )
        self.topology = topology
        self.level = level_
        self.complex = complex_
        self.entity_keys = tuple(padded_keys)
        self.views = tuple(views)
        self.capacity = capacity
        self.incidence_capacity = incidence_bounds
        self.complex_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-complex",
                "epoch": topology.epoch.epoch_id,
                "level": level_,
                "complex": complex_.topology_id,
                "views": [view.view_id for view in views],
                "incidence_capacity": incidence_bounds,
            }
        )

    def view(
        self,
        degree: int,
        orientation: Sequence[int],
        bucket: int,
        /,
    ) -> VariablePatchEntityBucketView:
        key = (int(degree), tuple(orientation), int(bucket))
        for view in self.views:
            if (view.degree, view.orientation, view.bucket) == key:
                return view
        raise KeyError("Unknown variable patch entity bucket view.")


class VariablePatchEntityComplexPlan(StrictModule, NonTrainableState):
    """Bounded entity and incidence policy for every variable AMR level."""

    capacity_plan: BlockHierarchyCapacityPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity_plan: BlockHierarchyCapacityPlan, /):
        if not isinstance(capacity_plan, BlockHierarchyCapacityPlan):
            raise TypeError(
                "Variable patch entity complex requires BlockHierarchyCapacityPlan."
            )
        required_routes = sum(
            max(0, len(capacity) - 1) for capacity in capacity_plan.entity_capacities
        )
        if len(capacity_plan.route_capacities) != required_routes:
            raise ValueError(
                "Entity complex route capacities require one entry per level/positive degree."
            )
        self.capacity_plan = capacity_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-entity-complex-plan",
                "capacity": capacity_plan.capacity_id,
            }
        )

    def prepare(
        self, topology: VariablePatchHierarchyTopology, /
    ) -> tuple[VariablePatchEntityComplex, ...]:
        capacities = self.capacity_plan.entity_capacities
        if (
            not isinstance(topology, VariablePatchHierarchyTopology)
            or len(capacities) != len(topology.levels)
            or any(
                len(capacity) != topology.plan.levels[level].dimension + 1
                for level, capacity in enumerate(capacities)
            )
        ):
            raise ValueError(
                "Entity complex capacities do not match patch hierarchy levels."
            )
        route_offset = 0
        result = []
        for level, capacity in enumerate(capacities):
            degree_count = len(capacity) - 1
            incidence = self.capacity_plan.route_capacities[
                route_offset : route_offset + degree_count
            ]
            route_offset += degree_count
            result.append(
                VariablePatchEntityComplex(
                    topology,
                    level,
                    capacity,
                    incidence,
                )
            )
        return tuple(result)


__all__ = [
    "VariablePatchEntityBucketView",
    "VariablePatchEntityComplex",
    "VariablePatchEntityComplexPlan",
]
