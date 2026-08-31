#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CellComplexPair, CellSubcomplex
from ._filtration import CellFiltration
from ._homology import compute_homology, HomologyResult
from ._induced import compute_induced_topology_map, induced_homology_coordinates
from ._integer import ExactIntegerCOO
from ._maps import CellularChainMap, CellularPairMap, chain_coordinate_id
from ._resources import TopologyResourcePolicy


ExtendedComponentKind = Literal[
    "ordinary", "relative", "extended_positive", "extended_negative"
]


class ExtendedPersistenceComponent(StrictModule, NonTrainableState):
    """One phase-qualified interval component of exact extended persistence."""

    degrees: Array
    birth_values: Array
    death_values: Array
    birth_nodes: Array
    death_nodes: Array
    kind: ExtendedComponentKind = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ExtendedComponentKind,
        degrees: ArrayLike,
        birth_values: ArrayLike,
        death_values: ArrayLike,
        birth_nodes: ArrayLike,
        death_nodes: ArrayLike,
        /,
    ):
        if kind not in (
            "ordinary",
            "relative",
            "extended_positive",
            "extended_negative",
        ):
            raise ValueError("Unknown extended-persistence component kind.")
        arrays = tuple(
            np.asarray(value)
            for value in (degrees, birth_values, death_values, birth_nodes, death_nodes)
        )
        if (
            any(value.ndim != 1 for value in arrays)
            or len({value.shape for value in arrays}) != 1
        ):
            raise ValueError("Extended-persistence component arrays must align.")
        self.degrees = jnp.asarray(arrays[0], dtype=jnp.int32)
        self.birth_values = jnp.asarray(arrays[1])
        self.death_values = jnp.asarray(arrays[2])
        self.birth_nodes = jnp.asarray(arrays[3], dtype=jnp.int32)
        self.death_nodes = jnp.asarray(arrays[4], dtype=jnp.int32)
        self.kind = kind
        self.component_id = canonical_fingerprint(
            {
                "kind": "extended-persistence-component",
                "component": kind,
                "degrees": array_tree_fingerprint(arrays[0]),
                "births": array_tree_fingerprint(arrays[1]),
                "deaths": array_tree_fingerprint(arrays[2]),
                "birth_nodes": array_tree_fingerprint(arrays[3]),
                "death_nodes": array_tree_fingerprint(arrays[4]),
            }
        )

    @property
    def interval_count(self) -> int:
        return int(self.degrees.shape[0])


class ExtendedPersistenceResult(StrictModule, NonTrainableState):
    """Exact interval decomposition of the sublevel-to-relative extended module."""

    ordinary: ExtendedPersistenceComponent
    relative: ExtendedPersistenceComponent
    extended_positive: ExtendedPersistenceComponent
    extended_negative: ExtendedPersistenceComponent
    filtration_id: str = eqx.field(static=True)
    field: PrimeField
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        ordinary: ExtendedPersistenceComponent,
        relative: ExtendedPersistenceComponent,
        extended_positive: ExtendedPersistenceComponent,
        extended_negative: ExtendedPersistenceComponent,
        /,
        *,
        filtration_id: str,
        field: PrimeField,
    ):
        self.ordinary = ordinary
        self.relative = relative
        self.extended_positive = extended_positive
        self.extended_negative = extended_negative
        self.filtration_id = str(filtration_id)
        self.field = field
        self.result_id = canonical_fingerprint(
            {
                "kind": "extended-persistence-result",
                "filtration": self.filtration_id,
                "field": field.field_id,
                "components": [
                    ordinary.component_id,
                    relative.component_id,
                    extended_positive.component_id,
                    extended_negative.component_id,
                ],
            }
        )


def _empty_subcomplex(complex: CellSubcomplex, /) -> CellSubcomplex:
    return CellSubcomplex(
        complex.topology,
        tuple(np.zeros_like(np.asarray(mask), dtype=bool) for mask in complex.masks),
        subcomplex_id=f"{complex.subcomplex_id}:empty",
    )


def _prefixes(filtration: CellFiltration, /):
    canonical = tuple(
        np.asarray(value if filtration.direction == "sublevel" else -value)
        for value in filtration.values
    )
    selected_values = np.concatenate(
        tuple(
            values[np.asarray(mask, dtype=bool)]
            for values, mask in zip(canonical, filtration.complex.masks, strict=True)
        )
    )
    levels = np.unique(selected_values)
    prefixes = [_empty_subcomplex(filtration.complex)]
    for level in levels:
        masks = tuple(
            np.asarray(base, dtype=bool) & (values <= level)
            for base, values in zip(filtration.complex.masks, canonical, strict=True)
        )
        prefixes.append(CellSubcomplex(filtration.complex.topology, masks))
    if prefixes[-1].subcomplex_id != filtration.complex.subcomplex_id:
        prefixes[-1] = filtration.complex
    return tuple(prefixes), levels


def _inclusion(source: CellSubcomplex, target: CellSubcomplex, /) -> CellularChainMap:
    maps = []
    for degree in range(source.max_degree + 1):
        target_inverse = np.asarray(target.layout.ambient_to_compact[degree])
        source_ambient = np.asarray(source.layout.compact_to_ambient[degree])
        rows = target_inverse[source_ambient]
        if np.any(rows < 0):
            raise ValueError("Extended-persistence prefix inclusion is invalid.")
        columns = np.arange(source.layout.counts[degree], dtype=np.int32)
        maps.append(
            ExactIntegerCOO(
                target.layout.counts[degree],
                source.layout.counts[degree],
                rows,
                columns,
                (1,) * source.layout.counts[degree],
                source_id=chain_coordinate_id(source.subcomplex_id, degree),
                target_id=chain_coordinate_id(target.subcomplex_id, degree),
            )
        )
    return CellularChainMap(source, target, maps)


def _modular_rank(matrix: np.ndarray, field: PrimeField, /) -> int:
    values = np.asarray(matrix, dtype=object) % field.modulus
    if values.ndim != 2:
        raise ValueError("Persistence-module map must be a matrix.")
    row = 0
    for column in range(values.shape[1]):
        pivot = next(
            (
                index
                for index in range(row, values.shape[0])
                if int(values[index, column]) % field.modulus
            ),
            None,
        )
        if pivot is None:
            continue
        values[[row, pivot]] = values[[pivot, row]]
        inverse = field.inverse(int(values[row, column]))
        values[row] = [field.multiply(int(value), inverse) for value in values[row]]
        for other in range(values.shape[0]):
            if other == row:
                continue
            factor = int(values[other, column]) % field.modulus
            if factor:
                values[other] = [
                    field.subtract(int(left), field.multiply(factor, int(right)))
                    for left, right in zip(values[other], values[row], strict=True)
                ]
        row += 1
        if row == values.shape[0]:
            break
    return row


def _intervals(dimensions, maps, field: PrimeField, /):
    node_count = len(dimensions)
    ranks = np.zeros((node_count, node_count), dtype=np.int32)
    for start in range(node_count):
        ranks[start, start] = dimensions[start]
        product = np.eye(dimensions[start], dtype=np.int64)
        for end in range(start + 1, node_count):
            product = np.asarray(maps[end - 1], dtype=object) @ product
            product = np.asarray(product, dtype=object) % field.modulus
            ranks[start, end] = _modular_rank(product, field)
    output = []
    for birth in range(node_count):
        for death in range(birth, node_count):
            multiplicity = int(ranks[birth, death])
            if birth:
                multiplicity -= int(ranks[birth - 1, death])
            if death + 1 < node_count:
                multiplicity -= int(ranks[birth, death + 1])
                if birth:
                    multiplicity += int(ranks[birth - 1, death + 1])
            if multiplicity < 0:
                raise RuntimeError("Extended persistence rank invariant is inconsistent.")
            output.extend((birth, death) for _ in range(multiplicity))
    return output


def compute_extended_persistence(
    filtration: CellFiltration,
    /,
    *,
    coefficients: PrimeField,
    max_degree: int | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> ExtendedPersistenceResult:
    """Compute the exact finite extended module by induced-map decomposition."""
    policy = TopologyResourcePolicy() if resources is None else resources
    maximum = filtration.max_degree if max_degree is None else int(max_degree)
    if maximum < 0 or maximum > filtration.max_degree:
        raise ValueError("Extended-persistence max_degree is outside the complex.")
    prefixes, levels = _prefixes(filtration)
    ordinary_results = tuple(
        compute_homology(
            prefix,
            coefficients=coefficients,
            degrees=tuple(range(maximum + 1)),
            representatives="both",
            resources=policy,
        )
        for prefix in prefixes
    )
    ordinary_maps = []
    for source_prefix, target_prefix, source_result, target_result in zip(
        prefixes[:-1],
        prefixes[1:],
        ordinary_results[:-1],
        ordinary_results[1:],
        strict=True,
    ):
        induced = compute_induced_topology_map(
            _inclusion(source_prefix, target_prefix), source_result, target_result
        )
        ordinary_maps.append(
            {value.degree: np.asarray(value.matrix) for value in induced.homology_maps}
        )
    ambient = filtration.complex
    relative_pairs = tuple(CellComplexPair(ambient, prefix) for prefix in prefixes[1:])
    relative_results = tuple(
        compute_homology(
            pair,
            coefficients=coefficients,
            degrees=tuple(range(maximum + 1)),
            representatives="both",
            resources=policy,
        )
        for pair in relative_pairs
    )
    empty_pair = CellComplexPair(ambient, prefixes[0])
    empty_result = compute_homology(
        empty_pair,
        coefficients=coefficients,
        degrees=tuple(range(maximum + 1)),
        representatives="both",
        resources=policy,
    )
    identity = CellularChainMap.identity(ambient)
    pair_maps = []
    previous_pair = empty_pair
    previous_result: HomologyResult = empty_result
    for pair, result in zip(relative_pairs, relative_results, strict=True):
        pair_map = CellularPairMap(previous_pair, pair, identity)
        coordinate_maps = induced_homology_coordinates(
            pair_map.quotient_maps,
            previous_pair.quotient_layout,
            pair.quotient_layout,
            previous_result,
            result,
        )
        pair_maps.append(
            {value.degree: np.asarray(value.matrix) for value in coordinate_maps}
        )
        previous_pair = pair
        previous_result = result
    node_results = ordinary_results + relative_results
    adjacent = ordinary_maps + pair_maps
    center = len(ordinary_results) - 1
    node_values = np.concatenate(
        (
            np.asarray([levels[0] if levels.size else 0.0] + levels.tolist()),
            levels,
        )
    )
    buckets: dict[str, list[tuple[int, float, float, int, int]]] = {
        "ordinary": [],
        "relative": [],
        "extended_positive": [],
        "extended_negative": [],
    }
    for degree in range(maximum + 1):
        dimensions = [result.degree(degree).dimension for result in node_results]
        maps = [
            values.get(
                degree,
                np.zeros((dimensions[index + 1], dimensions[index]), dtype=np.int64),
            )
            for index, values in enumerate(adjacent)
        ]
        for birth, death in _intervals(dimensions, maps, coefficients):
            if death < center:
                kind = "ordinary"
            elif birth > center:
                kind = "relative"
            else:
                kind = (
                    "extended_positive"
                    if node_values[birth] <= node_values[death]
                    else "extended_negative"
                )
            birth_value = float(node_values[birth])
            death_value = float(node_values[death])
            if filtration.direction == "superlevel":
                birth_value = -birth_value
                death_value = -death_value
            buckets[kind].append((degree, birth_value, death_value, birth, death))
    components = {}
    for kind, values in buckets.items():
        components[kind] = ExtendedPersistenceComponent(
            kind,
            np.asarray([value[0] for value in values], dtype=np.int32),
            np.asarray([value[1] for value in values]),
            np.asarray([value[2] for value in values]),
            np.asarray([value[3] for value in values], dtype=np.int32),
            np.asarray([value[4] for value in values], dtype=np.int32),
        )
    return ExtendedPersistenceResult(
        components["ordinary"],
        components["relative"],
        components["extended_positive"],
        components["extended_negative"],
        filtration_id=filtration.filtration_id,
        field=coefficients,
    )


__all__ = [
    "ExtendedPersistenceComponent",
    "ExtendedPersistenceResult",
    "compute_extended_persistence",
]
