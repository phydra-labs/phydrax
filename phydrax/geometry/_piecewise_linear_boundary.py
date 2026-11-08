#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact chart and metric queries over an owned piecewise-linear boundary."""

from __future__ import annotations

from fractions import Fraction
from typing import final, Protocol

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import Dim, HostInt64
from ._affine_plc_domain import AffinePlcDomain
from ._domain_coverage import PiecewiseLinearDomain
from ._rational import fraction_endpoint, rational_vector
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
)


class _PiecewiseLinearBoundaryOwner(Protocol):
    @property
    def domain(self) -> PiecewiseLinearDomain | AffinePlcDomain: ...

    @property
    def edges(self) -> np.ndarray: ...

    @property
    def envelope_id(self) -> str: ...

    @property
    def scope_entity_ids_by_mesh_row(self) -> tuple[np.ndarray, ...]: ...

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    def require_current(self, /) -> None: ...

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray: ...

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]: ...


@final
class PiecewiseLinearDomainBoundaryOwner(StrictModule, NonTrainableState):
    """Retain a declared PWL domain and its canonical occurrence-aware strata."""

    domain: PiecewiseLinearDomain
    edges: np.ndarray
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)
    _scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    _entity_keys: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = eqx.field(
        static=True
    )

    def __init__(self, domain: PiecewiseLinearDomain, /) -> None:
        if not isinstance(domain, PiecewiseLinearDomain) or domain.vertices.shape[1] != 3:
            raise TypeError(
                "A PWL domain boundary owner requires a declared 3D PWL domain."
            )
        edge_rows = {
            tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
            for face in np.asarray(domain.facets, dtype=np.int64)
            for index in range(3)
        }
        edges = np.asarray(sorted(edge_rows), dtype=np.int64).reshape((-1, 2))
        rows = (
            np.arange(domain.vertices.shape[0], dtype=np.int64),
            np.arange(edges.shape[0], dtype=np.int64),
            np.arange(domain.facets.shape[0], dtype=np.int64),
            np.arange(len(domain.region_ids), dtype=np.int64),
        )
        edges.setflags(write=False)
        for bank in rows:
            bank.setflags(write=False)
        self.domain = domain
        self.edges = edges
        self.scope_entity_ids_by_mesh_row = rows
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.envelope_id = domain.domain_id
        self._scope_entity_set_ids = tuple(
            f"{domain.domain_id}:pwl:{dimension}" for dimension in range(4)
        )
        self._entity_keys = (
            tuple(
                (f"{domain.domain_id}:vertex:{row}", ()) for row in range(rows[0].size)
            ),
            tuple(
                (f"{domain.domain_id}:edge:{int(edge[0])}:{int(edge[1])}", ())
                for edge in edges
            ),
            tuple((f"{domain.domain_id}:facet:{row}", ()) for row in range(rows[2].size)),
            tuple(zip(domain.region_ids, domain.region_occurrence_paths, strict=True)),
        )

    def require_current(self, /) -> None:
        if (
            self.domain.domain_id,
            self.domain.source_id,
            self.domain.source_revision,
        ) != (self.envelope_id, self.source_id, self.source_revision):
            raise ValueError("The retained PWL source domain changed after preparation.")

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        return self.scope_entity_ids_by_mesh_row[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        return self._scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        identifiers = self.scope_entity_ids_by_mesh_row[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError(
                "The requested PWL entity is outside its retained source bank."
            )
        return self._entity_keys[dimension][position]


class _PwlBoundaryEntityDim(Dim, minimum=1):
    """Selected source entities and their aligned owner rows."""


@final
class PiecewiseLinearBoundarySource(StrictModule, NonTrainableState):
    """One source-owned stratum query over an exact retained PWL boundary.

    The source owner supplies stable entity keys and cold revalidation; the
    PiecewiseLinearDomain and its retained edge bank supply actual represented
    coordinates. No topology or metric claim comes from a sampled proxy.
    """

    __strict_contract__ = True

    owner: _PiecewiseLinearBoundaryOwner
    domain: PiecewiseLinearDomain | AffinePlcDomain
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_query_id: str = eqx.field(static=True)
    entity_dimension: int = eqx.field(static=True)
    entity_ids: HostInt64[_PwlBoundaryEntityDim]
    source_rows: HostInt64[_PwlBoundaryEntityDim]
    source_entity_keys: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    chart_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        owner: _PiecewiseLinearBoundaryOwner,
        dimension: int,
        /,
        *,
        entity_ids: ArrayLike | None = None,
    ) -> None:
        from ..meshing._scope import _SourceEntityScopes

        if not isinstance(owner, _SourceEntityScopes):
            raise TypeError(
                "owner must implement the canonical source-entity scope authority."
            )
        domain_value: object = getattr(owner, "domain", None)
        if not isinstance(domain_value, (PiecewiseLinearDomain, AffinePlcDomain)):
            raise TypeError("owner must retain a PLC domain.")
        domain = domain_value
        source_id = canonical_identifier(owner.source_id, "source_id")
        source_revision = canonical_identifier(owner.source_revision, "source_revision")
        if (domain.source_id, domain.source_revision) != (source_id, source_revision):
            raise ValueError(
                "The piecewise-linear domain changes the retained owner identity."
            )
        if type(dimension) is not int or dimension not in (0, 1, 2):
            raise ValueError("PWL boundary stratum dimension must be 0, 1, or 2.")
        if domain.vertices.shape[1] != 3 or domain.facets.shape[1] != 3:
            raise ValueError(
                "PWL boundary queries require triangular surfaces in three-space."
            )
        owner_id = canonical_identifier(
            getattr(owner, "envelope_id", ""), "piecewise_linear_owner_id"
        )
        if not callable(getattr(owner, "require_current", None)):
            raise TypeError(
                "owner must provide cold mathematical revalidation through require_current()."
            )
        row_banks_value: object = getattr(owner, "scope_entity_ids_by_mesh_row", None)
        if not isinstance(row_banks_value, tuple) or len(row_banks_value) <= dimension:
            raise ValueError("owner must retain source entity IDs in mesh-row order.")
        row_banks = row_banks_value
        match dimension:
            case 0:
                source_elements = np.arange(
                    domain.vertices.shape[0], dtype=np.int64
                ).reshape((-1, 1))
            case 1:
                source_elements = np.asarray(owner.edges)
                if (
                    source_elements.dtype.kind not in "iu"
                    or source_elements.ndim != 2
                    or source_elements.shape[1] != 2
                    or np.any(source_elements < 0)
                    or np.any(source_elements >= domain.vertices.shape[0])
                ):
                    raise ValueError(
                        "owner must retain canonical PWL edge-to-vertex rows."
                    )
                source_elements = source_elements.astype(np.int64, copy=False)
            case 2:
                source_elements = np.asarray(domain.facets, dtype=np.int64)
        scoped_rows = np.asarray(row_banks[dimension], dtype=np.int64)
        if (
            scoped_rows.shape != (source_elements.shape[0],)
            or np.unique(scoped_rows).size != scoped_rows.size
            or set(scoped_rows.tolist())
            != set(np.asarray(owner.scope_entity_ids(dimension), dtype=np.int64).tolist())
        ):
            raise ValueError(
                "PWL source entity scopes do not match their retained element rows."
            )
        if entity_ids is None:
            selected = scoped_rows
        else:
            raw = np.asarray(entity_ids)
            if raw.dtype.kind not in "iu" or raw.ndim != 1:
                raise TypeError(
                    "entity_ids must be a one-dimensional integer source-scope bank."
                )
            selected_set = {int(value) for value in raw.tolist()}
            if len(selected_set) != raw.size or not selected_set <= set(
                scoped_rows.tolist()
            ):
                raise ValueError(
                    "entity_ids must uniquely select retained PWL source elements."
                )
            selected = np.asarray(
                [
                    int(value)
                    for value in scoped_rows.tolist()
                    if int(value) in selected_set
                ],
                dtype=np.int64,
            )
        if selected.size == 0:
            raise ValueError("A PWL boundary query requires at least one source entity.")
        row_by_id = {int(value): row for row, value in enumerate(scoped_rows.tolist())}
        selected_rows = np.asarray(
            [row_by_id[int(value)] for value in selected], dtype=np.int64
        )
        keys = tuple(
            owner.source_entity_key_for_scope(dimension, int(identifier))
            for identifier in selected.tolist()
        )
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not isinstance(key[1], tuple)
            or not all(isinstance(value, str) for value in key[1])
            for key in keys
        ) or len(set(keys)) != len(keys):
            raise ValueError(
                "PWL entity keys must be unique and retain occurrence paths."
            )
        query_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-boundary-query",
                "source": (source_id, source_revision),
                "owner": owner_id,
                "domain": domain.domain_id,
                "dimension": dimension,
                "entity_ids": array_tree_fingerprint(selected),
                "entity_keys": keys,
            }
        )
        chart_ids = (
            tuple(
                canonical_fingerprint(
                    {
                        "kind": "piecewise-linear-boundary-chart",
                        "query": query_id,
                        "dimension": dimension,
                        "entity_id": int(identifier),
                        "key": key,
                        "source_row": int(row),
                    }
                )
                for identifier, key, row in zip(
                    selected.tolist(), keys, selected_rows.tolist(), strict=True
                )
            )
            if dimension in (1, 2)
            else ()
        )
        selected_copy = np.array(selected, dtype=np.int64, copy=True)
        rows_copy = np.array(selected_rows, dtype=np.int64, copy=True)
        selected_copy.setflags(write=False)
        rows_copy.setflags(write=False)
        self.owner = owner
        self.domain = domain
        self.source_id = source_id
        self.source_revision = source_revision
        self.source_query_id = query_id
        self.entity_dimension = dimension
        self.entity_ids = selected_copy
        self.source_rows = rows_copy
        self.source_entity_keys = tuple(key[0] for key in keys)
        self.source_occurrence_paths = tuple(key[1] for key in keys)
        self.chart_ids = chart_ids
        self.require_current()

    @property
    def query_id(self) -> str:
        return self.source_query_id

    @property
    def ambient_dimension(self) -> int:
        return int(self.domain.vertices.shape[1])

    @property
    def is_full_scope(self) -> bool:
        return set(self.entity_ids.tolist()) == set(
            np.asarray(
                self.owner.scope_entity_ids(self.entity_dimension), dtype=np.int64
            ).tolist()
        )

    def require_current(self, /) -> None:
        owner = self.owner
        owner.require_current()
        domain_value: object = getattr(owner, "domain", None)
        if (
            not isinstance(domain_value, (PiecewiseLinearDomain, AffinePlcDomain))
            or domain_value.domain_id != self.domain.domain_id
            or (owner.source_id, owner.source_revision)
            != (self.source_id, self.source_revision)
        ):
            raise ValueError(
                "The retained PWL source owner no longer reproduces this boundary query."
            )
        row_banks_value: object = getattr(owner, "scope_entity_ids_by_mesh_row", None)
        if (
            not isinstance(row_banks_value, tuple)
            or len(row_banks_value) <= self.entity_dimension
        ):
            raise ValueError("The PWL owner lost its source entity ancestry bank.")
        current_rows = np.asarray(
            owner.scope_entity_ids_by_mesh_row[self.entity_dimension], dtype=np.int64
        )
        if not set(self.entity_ids.tolist()) <= set(current_rows.tolist()):
            raise ValueError(
                "The PWL source query no longer names retained boundary elements."
            )
        if self.entity_dimension == 1:
            current_edges = np.asarray(owner.edges, dtype=np.int64)
            if current_edges.shape != (current_rows.size, 2):
                raise ValueError("The PWL owner changed its source edge row shape.")
        current_keys = tuple(
            owner.source_entity_key_for_scope(self.entity_dimension, int(identifier))
            for identifier in self.entity_ids.tolist()
        )
        expected_keys = tuple(
            zip(self.source_entity_keys, self.source_occurrence_paths, strict=True)
        )
        if current_keys != expected_keys:
            raise ValueError(
                "The PWL source query changes retained entity keys or occurrences."
            )

    def selected_source_keys(
        self, dimension: int, /
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if dimension != self.entity_dimension:
            raise ValueError(
                "PWL query scope dimension differs from its retained stratum."
            )
        return tuple(
            sorted(
                zip(self.source_entity_keys, self.source_occurrence_paths, strict=True)
            )
        )

    def source_row_for_entity_key(
        self, dimension: int, key: tuple[str, tuple[str, ...]], /
    ) -> int:
        if dimension != self.entity_dimension:
            raise ValueError(
                "PWL entity-key dimension differs from its retained stratum."
            )
        try:
            return tuple(
                zip(self.source_entity_keys, self.source_occurrence_paths, strict=True)
            ).index(key)
        except ValueError as error:
            raise ValueError(
                "Source entity key is outside the selected PWL query scope."
            ) from error

    def chart_row(self, chart_dimension: int, chart_id: str, /) -> int:
        if chart_dimension != self.entity_dimension or chart_dimension not in (1, 2):
            raise ValueError(
                "PWL chart dimension differs from the selected source stratum."
            )
        wanted = canonical_identifier(chart_id, "source_chart_id")
        try:
            return self.chart_ids.index(wanted)
        except ValueError as error:
            raise ValueError(
                "PWL source chart is outside the retained query scope."
            ) from error

    def chart_id_for_entity_id(self, entity_id: int, /) -> str:
        if self.entity_dimension not in (1, 2):
            raise ValueError("PWL point scopes do not have chart IDs.")
        matches = np.flatnonzero(self.entity_ids == int(entity_id))
        if matches.size != 1:
            raise ValueError("Entity ID is not exactly one selected PWL source chart.")
        return self.chart_ids[int(matches[0])]

    def chart_source_key(
        self, chart_dimension: int, chart_id: str, /
    ) -> tuple[str, tuple[str, ...]]:
        row = self.chart_row(chart_dimension, chart_id)
        return self.source_entity_keys[row], self.source_occurrence_paths[row]

    def chart_coordinate_expressions(
        self, chart_dimension: int, chart_id: str, /
    ) -> tuple[dict[tuple[int, ...], Fraction], ...]:
        row = self.chart_row(chart_dimension, chart_id)
        source_row = int(self.source_rows[row])
        if chart_dimension == 2:
            element = self.domain.facets[source_row]
        else:
            element = np.asarray(self.owner.edges, dtype=np.int64)[source_row]
        exact = tuple(self._exact_vertex(int(vertex)) for vertex in element)
        if chart_dimension == 1:
            return tuple(
                {(0,): exact[0][axis], (1,): exact[1][axis] - exact[0][axis]}
                for axis in range(3)
            )
        return tuple(
            {
                (0, 0): exact[0][axis],
                (1, 0): exact[1][axis] - exact[0][axis],
                (0, 1): exact[2][axis] - exact[0][axis],
            }
            for axis in range(3)
        )

    def point_coordinates(
        self,
        entity_id: str,
        occurrence_path: tuple[str, ...],
        /,
    ) -> tuple[Fraction, Fraction, Fraction]:
        if self.entity_dimension != 0:
            raise ValueError(
                "PWL point coordinates require a zero-dimensional source query."
            )
        row = self.source_row_for_entity_key(0, (entity_id, occurrence_path))
        source_row = int(self.source_rows[row])
        return self._exact_vertex(source_row)

    def _exact_vertex(self, row: int, /) -> tuple[Fraction, Fraction, Fraction]:
        if isinstance(self.domain, AffinePlcDomain):
            point = self.domain.exact_vertices[row]
        else:
            point = rational_vector(self.domain.vertices[row])
        return point[0], point[1], point[2]

    def validate_chart_witness(self, witness: SourceBoundaryChartWitness, /) -> None:
        if not isinstance(witness, SourceBoundaryChartWitness):
            raise TypeError("PWL source charts require SourceBoundaryChartWitness.")
        self.require_current()
        if (
            self.entity_dimension not in (1, 2)
            or witness.source_id != self.source_id
            or witness.source_revision != self.source_revision
            or witness.source_query_id != self.source_query_id
            or witness.chart_dimension != self.entity_dimension
            or witness.implicit_cell_keys.shape != (0, 4)
            or witness.implicit_root_edge_keys.shape != (0, 3, 2, 3)
            or set(witness.source_chart_ids) != set(self.chart_ids)
        ):
            raise ValueError(
                "PWL source chart witness changes its retained source query."
            )
        for row, chart_id in enumerate(witness.source_chart_ids):
            source_row = self.chart_row(self.entity_dimension, chart_id)
            if (
                witness.source_entity_ids[row] != self.source_entity_keys[source_row]
                or witness.occurrence_paths[row]
                != self.source_occurrence_paths[source_row]
            ):
                raise ValueError(
                    "PWL source chart witness changes source entity or occurrence ancestry."
                )
            parameters = witness.source_parameter_chart(row)
            if self.entity_dimension == 1:
                if any(not 0 <= point[0] <= 1 for point in parameters):
                    raise ValueError(
                        "PWL source edge parameters leave the exact unit interval."
                    )
            elif any(u < 0 or v < 0 or u + v > 1 for u, v in parameters):
                raise ValueError(
                    "PWL source face parameters leave the exact facet simplex."
                )

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        if self.entity_dimension != 2:
            raise ValueError(
                "Only a complete PWL facet query provides triangle chart coverage."
            )
        self.require_current()
        facets = self.domain.facets[self.source_rows]
        triangles = self.domain.vertices[facets]
        error = max(
            (
                sum(
                    (
                        (value - Fraction(float(rounded))) ** 2
                        for value, rounded in zip(
                            self._exact_vertex(int(vertex)),
                            self.domain.vertices[int(vertex)],
                            strict=True,
                        )
                    ),
                    Fraction(0),
                )
                for facet in facets
                for vertex in facet
            ),
            default=Fraction(0),
        )
        _, error_upper = fraction_sqrt_interval(error)
        bound = fraction_endpoint(error_upper, upper=True, saturate=True)
        return SourceBoundaryChartCover(
            np.asarray(triangles, dtype=np.float64),
            np.full((triangles.shape[0],), bound, dtype=np.float64),
            "certified",
            True,
            self.source_id,
            self.source_revision,
            restriction_ids=(self.source_query_id,),
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or np.any(~np.isfinite(values)):
            raise ValueError("PWL boundary distance points must be finite three-vectors.")
        self.require_current()
        from ._domain_boundary import _point_segment_squared, _point_triangle_squared

        edges = (
            None
            if self.entity_dimension != 1
            else np.asarray(self.owner.edges, dtype=np.int64)
        )
        lower = np.empty((values.shape[0],), dtype=np.float64)
        upper = np.empty_like(lower)
        for point_row in range(values.shape[0]):
            point_coordinates = tuple(
                Fraction.from_float(float(values[point_row, axis])) for axis in range(3)
            )
            exact_point: tuple[Fraction, Fraction, Fraction] = (
                point_coordinates[0],
                point_coordinates[1],
                point_coordinates[2],
            )
            squared: list[Fraction] = []
            for source_row in self.source_rows.tolist():
                if self.entity_dimension == 0:
                    source_point = self._exact_vertex(int(source_row))
                    squared.append(
                        sum(
                            (
                                (a - b) ** 2
                                for a, b in zip(exact_point, source_point, strict=True)
                            ),
                            Fraction(0),
                        )
                    )
                elif self.entity_dimension == 1:
                    if edges is None:
                        raise RuntimeError(
                            "A one-dimensional PWL query lost its edge bank."
                        )
                    edge = edges[int(source_row)]
                    segment = tuple(self._exact_vertex(int(vertex)) for vertex in edge)
                    squared.append(
                        _point_segment_squared(exact_point, segment[0], segment[1])
                    )
                else:
                    vertices = self.domain.facets[int(source_row)]
                    first = self._exact_vertex(int(vertices[0]))
                    second = self._exact_vertex(int(vertices[1]))
                    third = self._exact_vertex(int(vertices[2]))
                    triangle = (
                        (first[0], first[1], first[2]),
                        (second[0], second[1], second[2]),
                        (third[0], third[1], third[2]),
                    )
                    squared.append(_point_triangle_squared(exact_point, triangle))
            distance_lower, distance_upper = fraction_sqrt_interval(min(squared))
            lower[point_row] = fraction_endpoint(
                distance_lower, upper=False, saturate=True
            )
            upper[point_row] = fraction_endpoint(
                distance_upper, upper=True, saturate=True
            )
        return SourceBoundaryDistance(lower, upper, "certified")


__all__ = ["PiecewiseLinearBoundarySource"]
