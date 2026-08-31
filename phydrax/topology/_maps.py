#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._complex import CellComplexPair, CellSubcomplex, compact_boundary
from ._filtration import CellFiltration
from ._integer import ExactIntegerCOO


def chain_coordinate_id(source_id: str, degree: int, /) -> str:
    return f"{source_id}:degree:{int(degree)}"


def _boundaries(complex: CellSubcomplex, /) -> tuple[ExactIntegerCOO, ...]:
    return tuple(
        ExactIntegerCOO.from_boundary(compact_boundary(complex, degree))
        for degree in range(complex.max_degree + 1)
    )


class CellularChainMap(StrictModule, NonTrainableState):
    """Exact covariant chain map between compact cellular chain coordinates."""

    source: CellSubcomplex
    target: CellSubcomplex
    degree_maps: tuple[ExactIntegerCOO, ...]
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellSubcomplex,
        target: CellSubcomplex,
        degree_maps: Sequence[ExactIntegerCOO],
        /,
        *,
        map_id: str | None = None,
    ):
        if not isinstance(source, CellSubcomplex) or not isinstance(
            target, CellSubcomplex
        ):
            raise TypeError("Cellular chain maps require two CellSubcomplex values.")
        if source.max_degree != target.max_degree:
            raise ValueError("Cellular chain maps require equal represented max degree.")
        values = tuple(degree_maps)
        maximum = max(source.max_degree, target.max_degree)
        if len(values) != maximum + 1:
            raise ValueError("One cellular chain map is required per represented degree.")
        for degree, matrix in enumerate(values):
            if not isinstance(matrix, ExactIntegerCOO):
                raise TypeError("Cellular degree maps must be ExactIntegerCOO values.")
            source_count = (
                source.layout.counts[degree] if degree <= source.max_degree else 0
            )
            target_count = (
                target.layout.counts[degree] if degree <= target.max_degree else 0
            )
            if matrix.column_count != source_count or matrix.row_count != target_count:
                raise ValueError("Cellular degree map dimensions do not match complexes.")
            if matrix.source_id != chain_coordinate_id(source.subcomplex_id, degree):
                raise ValueError("Cellular map source coordinate ID is incorrect.")
            if matrix.target_id != chain_coordinate_id(target.subcomplex_id, degree):
                raise ValueError("Cellular map target coordinate ID is incorrect.")
        source_boundaries = _boundaries(source)
        target_boundaries = _boundaries(target)
        for degree in range(1, maximum + 1):
            source_boundary = (
                source_boundaries[degree]
                if degree <= source.max_degree
                else ExactIntegerCOO.zero(
                    0,
                    0,
                    source_id=chain_coordinate_id(source.subcomplex_id, degree),
                    target_id=chain_coordinate_id(source.subcomplex_id, degree - 1),
                )
            )
            target_boundary = (
                target_boundaries[degree]
                if degree <= target.max_degree
                else ExactIntegerCOO.zero(
                    0,
                    0,
                    source_id=chain_coordinate_id(target.subcomplex_id, degree),
                    target_id=chain_coordinate_id(target.subcomplex_id, degree - 1),
                )
            )
            left = target_boundary.compose(values[degree])
            right = values[degree - 1].compose(source_boundary)
            if not left.equals(right):
                raise ValueError(
                    f"Cellular map violates boundary commutation in degree {degree}."
                )
        declared = None if map_id is None else str(map_id)
        if declared is not None and not declared:
            raise ValueError("map_id must be non-empty when provided.")
        self.source = source
        self.target = target
        self.degree_maps = values
        self.map_id = canonical_fingerprint(
            {
                "kind": "cellular-chain-map",
                "source": source.subcomplex_id,
                "target": target.subcomplex_id,
                "matrices": [value.matrix_id for value in values],
                "declared": declared,
            }
        )

    @classmethod
    def identity(cls, complex: CellSubcomplex, /) -> "CellularChainMap":
        return cls(
            complex,
            complex,
            tuple(
                ExactIntegerCOO.identity(
                    count,
                    coordinate_id=chain_coordinate_id(complex.subcomplex_id, degree),
                )
                for degree, count in enumerate(complex.layout.counts)
            ),
            map_id=f"identity:{complex.subcomplex_id}",
        )

    def compose(self, right: "CellularChainMap", /) -> "CellularChainMap":
        """Return ``self ∘ right``."""
        if not isinstance(right, CellularChainMap):
            raise TypeError("Cellular map composition requires CellularChainMap.")
        if right.target.subcomplex_id != self.source.subcomplex_id:
            raise ValueError("Cellular map composition complexes do not align.")
        return CellularChainMap(
            right.source,
            self.target,
            tuple(
                left.compose(right_value)
                for left, right_value in zip(
                    self.degree_maps, right.degree_maps, strict=True
                )
            ),
            map_id=f"{self.map_id}:after:{right.map_id}",
        )

    @property
    def cochain_pullbacks(self) -> tuple[ExactIntegerCOO, ...]:
        return tuple(value.transpose() for value in self.degree_maps)


class CellularPairMap(StrictModule, NonTrainableState):
    """Exact cellular map of pairs and its induced quotient chain map."""

    source: CellComplexPair
    target: CellComplexPair
    ambient_map: CellularChainMap
    quotient_maps: tuple[ExactIntegerCOO, ...]
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellComplexPair,
        target: CellComplexPair,
        ambient_map: CellularChainMap,
        /,
    ):
        if not isinstance(source, CellComplexPair) or not isinstance(
            target, CellComplexPair
        ):
            raise TypeError("Cellular pair maps require two CellComplexPair values.")
        if ambient_map.source.subcomplex_id != source.ambient.subcomplex_id:
            raise ValueError("Pair-map ambient source does not match the source pair.")
        if ambient_map.target.subcomplex_id != target.ambient.subcomplex_id:
            raise ValueError("Pair-map ambient target does not match the target pair.")
        quotient_values = []
        for degree, matrix in enumerate(ambient_map.degree_maps):
            source_ambient = np.asarray(
                source.ambient.layout.compact_to_ambient[degree], dtype=np.int32
            )
            target_ambient = np.asarray(
                target.ambient.layout.compact_to_ambient[degree], dtype=np.int32
            )
            source_relative_mask = np.asarray(source.relative.masks[degree], dtype=bool)
            target_relative_mask = np.asarray(target.relative.masks[degree], dtype=bool)
            source_quotient_map = np.asarray(
                source.quotient_layout.ambient_to_compact[degree], dtype=np.int32
            )
            target_quotient_map = np.asarray(
                target.quotient_layout.ambient_to_compact[degree], dtype=np.int32
            )
            rows = []
            columns = []
            coefficients = []
            for row, column, coefficient in matrix.entries():
                source_slot = int(source_ambient[column])
                target_slot = int(target_ambient[row])
                if (
                    source_relative_mask[source_slot]
                    and not target_relative_mask[target_slot]
                ):
                    raise ValueError(
                        "Cellular pair map does not send the relative subcomplex "
                        "into the target relative subcomplex."
                    )
                quotient_column = int(source_quotient_map[source_slot])
                quotient_row = int(target_quotient_map[target_slot])
                if quotient_column >= 0 and quotient_row >= 0:
                    rows.append(quotient_row)
                    columns.append(quotient_column)
                    coefficients.append(coefficient)
            quotient_values.append(
                ExactIntegerCOO(
                    target.quotient_layout.counts[degree],
                    source.quotient_layout.counts[degree],
                    np.asarray(rows, dtype=np.int32),
                    np.asarray(columns, dtype=np.int32),
                    coefficients,
                    source_id=chain_coordinate_id(source.pair_id, degree),
                    target_id=chain_coordinate_id(target.pair_id, degree),
                )
            )
        source_boundaries = tuple(
            ExactIntegerCOO.from_boundary(compact_boundary(source, degree))
            for degree in range(source.max_degree + 1)
        )
        target_boundaries = tuple(
            ExactIntegerCOO.from_boundary(compact_boundary(target, degree))
            for degree in range(target.max_degree + 1)
        )
        for degree in range(1, source.max_degree + 1):
            left = target_boundaries[degree].compose(quotient_values[degree])
            right = quotient_values[degree - 1].compose(source_boundaries[degree])
            if not left.equals(right):
                raise ValueError("Cellular pair quotient map violates chain commutation.")
        self.source = source
        self.target = target
        self.ambient_map = ambient_map
        self.quotient_maps = tuple(quotient_values)
        self.map_id = canonical_fingerprint(
            {
                "kind": "cellular-pair-map",
                "source": source.pair_id,
                "target": target.pair_id,
                "ambient": ambient_map.map_id,
                "quotient": [value.matrix_id for value in quotient_values],
            }
        )


class FilteredCellularChainMap(StrictModule, NonTrainableState):
    """Exact chain map with an explicit filtration-shift bound."""

    chain_map: CellularChainMap
    source_filtration: CellFiltration
    target_filtration: CellFiltration
    epsilon: float = eqx.field(static=True)
    filtered_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        chain_map: CellularChainMap,
        source_filtration: CellFiltration,
        target_filtration: CellFiltration,
        /,
        *,
        epsilon: float = 0.0,
    ):
        if source_filtration.complex.subcomplex_id != chain_map.source.subcomplex_id:
            raise ValueError("Source filtration does not match the cellular map.")
        if target_filtration.complex.subcomplex_id != chain_map.target.subcomplex_id:
            raise ValueError("Target filtration does not match the cellular map.")
        if source_filtration.direction != target_filtration.direction:
            raise ValueError("Filtered maps require a shared filtration direction.")
        epsilon_ = float(epsilon)
        if not np.isfinite(epsilon_) or epsilon_ < 0.0:
            raise ValueError("Filtered-map epsilon must be finite and non-negative.")
        source_values = np.asarray(source_filtration.canonical_compact_values())
        target_values = np.asarray(target_filtration.canonical_compact_values())
        source_offsets = np.cumsum(
            (0,) + chain_map.source.layout.counts[:-1], dtype=np.int64
        )
        target_offsets = np.cumsum(
            (0,) + chain_map.target.layout.counts[:-1], dtype=np.int64
        )
        for degree, matrix in enumerate(chain_map.degree_maps):
            for row, column, _ in matrix.entries():
                source_value = source_values[int(source_offsets[degree]) + column]
                target_value = target_values[int(target_offsets[degree]) + row]
                if target_value > source_value + epsilon_:
                    raise ValueError("Cellular map exceeds its filtration-shift bound.")
        self.chain_map = chain_map
        self.source_filtration = source_filtration
        self.target_filtration = target_filtration
        self.epsilon = epsilon_
        self.filtered_map_id = canonical_fingerprint(
            {
                "kind": "filtered-cellular-chain-map",
                "map": chain_map.map_id,
                "source_filtration": source_filtration.filtration_id,
                "target_filtration": target_filtration.filtration_id,
                "epsilon": epsilon_,
            }
        )


class CellularChainContraction(StrictModule, NonTrainableState):
    """Exact chain contraction with inclusion, projection, and chain homotopy."""

    large: CellSubcomplex
    small: CellSubcomplex
    projection: CellularChainMap
    inclusion: CellularChainMap
    homotopies: tuple[ExactIntegerCOO, ...]
    contraction_id: str = eqx.field(static=True)

    def __init__(
        self,
        large: CellSubcomplex,
        small: CellSubcomplex,
        projection: CellularChainMap,
        inclusion: CellularChainMap,
        homotopies: Sequence[ExactIntegerCOO],
        /,
    ):
        if projection.source.subcomplex_id != large.subcomplex_id:
            raise ValueError("Contraction projection must start on the large complex.")
        if projection.target.subcomplex_id != small.subcomplex_id:
            raise ValueError("Contraction projection must end on the small complex.")
        if inclusion.source.subcomplex_id != small.subcomplex_id:
            raise ValueError("Contraction inclusion must start on the small complex.")
        if inclusion.target.subcomplex_id != large.subcomplex_id:
            raise ValueError("Contraction inclusion must end on the large complex.")
        small_identity = CellularChainMap.identity(small)
        if any(
            not actual.equals(expected)
            for actual, expected in zip(
                projection.compose(inclusion).degree_maps,
                small_identity.degree_maps,
                strict=True,
            )
        ):
            raise ValueError("Chain contraction violates projection-inclusion identity.")
        values = tuple(homotopies)
        if len(values) != large.max_degree + 1:
            raise ValueError("One chain homotopy value is required per large degree.")
        boundaries = _boundaries(large)
        large_identity = CellularChainMap.identity(large)
        inclusion_projection = inclusion.compose(projection)
        for degree in range(large.max_degree + 1):
            target_count = (
                large.layout.counts[degree + 1] if degree < large.max_degree else 0
            )
            homotopy = values[degree]
            if (
                homotopy.column_count != large.layout.counts[degree]
                or homotopy.row_count != target_count
                or homotopy.source_id != chain_coordinate_id(large.subcomplex_id, degree)
                or homotopy.target_id
                != chain_coordinate_id(large.subcomplex_id, degree + 1)
            ):
                raise ValueError("Chain homotopy coordinates do not align.")
            left = large_identity.degree_maps[degree].add(
                inclusion_projection.degree_maps[degree], scale=-1
            )
            if degree < large.max_degree:
                right = boundaries[degree + 1].compose(homotopy)
            else:
                right = ExactIntegerCOO.zero(
                    large.layout.counts[degree],
                    large.layout.counts[degree],
                    source_id=chain_coordinate_id(large.subcomplex_id, degree),
                    target_id=chain_coordinate_id(large.subcomplex_id, degree),
                )
            if degree:
                right = right.add(values[degree - 1].compose(boundaries[degree]))
            if not left.equals(right):
                raise ValueError("Chain contraction violates its homotopy identity.")
        self.large = large
        self.small = small
        self.projection = projection
        self.inclusion = inclusion
        self.homotopies = values
        self.contraction_id = canonical_fingerprint(
            {
                "kind": "cellular-chain-contraction",
                "large": large.subcomplex_id,
                "small": small.subcomplex_id,
                "projection": projection.map_id,
                "inclusion": inclusion.map_id,
                "homotopies": [value.matrix_id for value in values],
            }
        )


class FilteredCellularChainContraction(StrictModule, NonTrainableState):
    """Exact contraction whose projection, inclusion, and homotopy respect filtration."""

    contraction: CellularChainContraction
    large_filtration: CellFiltration
    small_filtration: CellFiltration
    epsilon: float = eqx.field(static=True)
    filtered_contraction_id: str = eqx.field(static=True)

    def __init__(
        self,
        contraction: CellularChainContraction,
        large_filtration: CellFiltration,
        small_filtration: CellFiltration,
        /,
        *,
        epsilon: float = 0.0,
    ):
        epsilon_ = float(epsilon)
        FilteredCellularChainMap(
            contraction.projection,
            large_filtration,
            small_filtration,
            epsilon=epsilon_,
        )
        FilteredCellularChainMap(
            contraction.inclusion,
            small_filtration,
            large_filtration,
            epsilon=epsilon_,
        )
        large_values = tuple(
            np.asarray(value if large_filtration.direction == "sublevel" else -value)[
                np.asarray(ambient)
            ]
            for value, ambient in zip(
                large_filtration.values,
                large_filtration.complex.layout.compact_to_ambient,
                strict=True,
            )
        )
        for degree, homotopy in enumerate(contraction.homotopies[:-1]):
            for row, column, _ in homotopy.entries():
                if (
                    large_values[degree + 1][row]
                    > large_values[degree][column] + epsilon_
                ):
                    raise ValueError(
                        "Chain homotopy exceeds the declared filtration shift."
                    )
        self.contraction = contraction
        self.large_filtration = large_filtration
        self.small_filtration = small_filtration
        self.epsilon = epsilon_
        self.filtered_contraction_id = canonical_fingerprint(
            {
                "kind": "filtered-cellular-chain-contraction",
                "contraction": contraction.contraction_id,
                "large_filtration": large_filtration.filtration_id,
                "small_filtration": small_filtration.filtration_id,
                "epsilon": epsilon_,
            }
        )


__all__ = [
    "CellularChainContraction",
    "CellularChainMap",
    "CellularPairMap",
    "FilteredCellularChainMap",
    "FilteredCellularChainContraction",
    "chain_coordinate_id",
]
