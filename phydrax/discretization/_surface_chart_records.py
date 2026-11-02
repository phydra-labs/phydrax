from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Protocol, runtime_checkable, Sequence, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._archive_columns import (
    ExactTensorArchiveRecipe,
    IntegerRowsArchiveRecipe,
    StaticColumnArchiveRecipe,
    TextColumnArchiveRecipe,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import checked, Identifier
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from ._cell_mesh import CellMesh


class _DomainIdentity(Protocol):
    domain_id: str


class _BoundaryChartCover(Protocol):
    complete: bool


class _CertificateBinding(Protocol):
    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        source: MeshingDomainBoundarySource | None = None,
    ) -> None: ...


if TYPE_CHECKING:
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
else:

    @runtime_checkable
    class _MeshingDomainBoundarySource(Protocol):
        domain: _DomainIdentity

        def boundary_chart_cover(self, /) -> _BoundaryChartCover: ...

    @runtime_checkable
    class _GlobalEmbeddingCertificate(Protocol):
        binding: _CertificateBinding
        certificate_id: str
        validity_certificate_id: str
        status: str

        def require_current(
            self,
            mesh: CellMesh,
            geometry: CellGeometrySpec,
            /,
            *,
            validity: CellValidityCertificate | None = None,
        ) -> None: ...

    @runtime_checkable
    class _SourceFidelityCertificate(Protocol):
        binding: _CertificateBinding
        certificate_id: str
        deviation_bounds: np.ndarray
        source_cell_global_ids: tuple[int, ...]
        status: str

        def require_current(
            self,
            mesh: CellMesh,
            geometry: CellGeometrySpec,
            source: _MeshingDomainBoundarySource,
            /,
        ) -> None: ...

        def revalidate(
            self,
            mesh: CellMesh,
            geometry: CellGeometrySpec,
            source: _MeshingDomainBoundarySource,
            /,
        ) -> None: ...

    MeshingDomainBoundarySource = _MeshingDomainBoundarySource
    GlobalEmbeddingCertificate = _GlobalEmbeddingCertificate
    SourceFidelityCertificate = _SourceFidelityCertificate

_ExactPoint = tuple[Fraction, Fraction]


class SurfaceRootMeshCellBinding(NamedTuple):
    """One exact chart-row binding to an actual root CellMesh cell."""

    root_geometry_id: str
    root_topology_id: str
    root_cell_global_id: int


_ChartRationalToken = tuple[int, int]
_ChartRationalPoint = tuple[_ChartRationalToken, _ChartRationalToken]
_ChartRationalCell = tuple[_ChartRationalPoint, ...]
_ChartRationalBank = tuple[_ChartRationalCell, ...]
_ExactTriangle = tuple[_ExactPoint, _ExactPoint, _ExactPoint]
_ExactChart = tuple[_ExactPoint, ...]


def _chart_corner_count(cell_kind: str, /) -> int:
    if cell_kind == "triangle":
        return 3
    if cell_kind == "quadrilateral":
        return 4
    raise ValueError("Surface charts require a triangle or quadrilateral cell kind.")


def material_triangle(chart: _ExactChart, /) -> _ExactTriangle:
    """Admit one canonical chart to the triangle-only material partition."""
    if len(chart) != 3:
        raise ValueError("Surface material deformation requires triangular charts.")
    return chart[0], chart[1], chart[2]


def _area(points: tuple[_ExactPoint, ...], /) -> Fraction:
    if len(points) < 3:
        return Fraction(0)
    return (
        sum(
            (
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(points, (*points[1:], points[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _exact_fraction(value: object, /) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Exact chart coordinates cannot be boolean values.")
    if isinstance(value, (float, np.floating)):
        return Fraction(float(value))
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    raise TypeError("Exact chart coordinates must be Fractions or binary numeric values.")


def exact_chart_bank(
    values: ArrayLike, count: int, /, *, cell_kind: str = "triangle"
) -> tuple[_ExactChart, ...]:
    corners = _chart_corner_count(cell_kind)
    bank = np.asarray(values, dtype=object)
    if bank.shape != (count, corners, 2):
        raise ValueError("Material charts must align with their scientific cell rows.")
    charts: list[_ExactChart] = []
    for cell in bank:
        chart = tuple(
            (_exact_fraction(point[0]), _exact_fraction(point[1])) for point in cell
        )
        if not _area(chart):
            raise ValueError("Material charts must be nondegenerate exact cells.")
        charts.append(chart)
    return tuple(charts)


def rational_chart_tokens(
    values: Sequence[Sequence[Sequence[Fraction | int | np.integer]]],
    /,
    *,
    cell_kind: str = "triangle",
) -> _ChartRationalBank:
    """Encode exact chart coordinates as reduced rational tokens."""
    corners = _chart_corner_count(cell_kind)
    rows: list[_ChartRationalCell] = []
    for row in values:
        if len(row) != corners:
            raise ValueError("Material chart corner count must match its cell kind.")
        points: list[_ChartRationalPoint] = []
        for point in row:
            if len(point) != 2:
                raise ValueError("Material chart corners must be two-dimensional.")
            coordinates: list[_ChartRationalToken] = []
            for value in point:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (Fraction, int, np.integer)
                ):
                    raise TypeError(
                        "Exact material charts require Fraction or integer coordinates."
                    )
                exact = value if isinstance(value, Fraction) else Fraction(int(value))
                coordinates.append((exact.numerator, exact.denominator))
            points.append((coordinates[0], coordinates[1]))
        rows.append(tuple(points))
    return tuple(rows)


def _exact_chart_from_tokens(bank: _ChartRationalBank, row: int, /) -> _ExactChart:
    tokens = bank[row]

    def point(value: _ChartRationalPoint, /) -> _ExactPoint:
        first, second = value
        return (
            Fraction(first[0], first[1]),
            Fraction(second[0], second[1]),
        )

    return tuple(point(value) for value in tokens)


def _validated_chart_rational_tokens(
    supplied: _ChartRationalBank | None,
    realization: np.ndarray,
    count: int,
    /,
    *,
    cell_kind: str = "triangle",
) -> _ChartRationalBank:
    corners = _chart_corner_count(cell_kind)
    if supplied is None:
        exact = exact_chart_bank(realization, count, cell_kind=cell_kind)
        return rational_chart_tokens(exact, cell_kind=cell_kind)
    if not isinstance(supplied, tuple) or len(supplied) != count:
        raise ValueError("Rational chart tokens must align with their cell rows.")
    exact_rows: list[_ExactChart] = []
    for row in supplied:
        if not isinstance(row, tuple) or len(row) != corners:
            raise ValueError(
                "Rational chart token corner count must match its cell kind."
            )
        points: list[_ExactPoint] = []
        for point in row:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError(
                    "Rational chart token corners must have two coordinates."
                )
            coordinates: list[Fraction] = []
            for token in point:
                if (
                    not isinstance(token, tuple)
                    or len(token) != 2
                    or any(
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer))
                        for value in token
                    )
                ):
                    raise TypeError(
                        "Chart coordinates must be numerator/denominator integer pairs."
                    )
                numerator, denominator = (int(value) for value in token)
                if denominator <= 0:
                    raise ValueError("Chart token denominators must be positive.")
                exact = Fraction(numerator, denominator)
                if (exact.numerator, exact.denominator) != (numerator, denominator):
                    raise ValueError("Chart tokens must be reduced rational pairs.")
                coordinates.append(exact)
            points.append((coordinates[0], coordinates[1]))
        exact_rows.append(tuple(points))
    canonical = rational_chart_tokens(exact_rows, cell_kind=cell_kind)
    rounded = np.asarray(
        [
            [
                [float(value) for value in point]
                for point in _exact_chart_from_tokens(canonical, row)
            ]
            for row in range(count)
        ],
        dtype=np.float64,
    )
    if not np.array_equal(rounded, realization):
        raise ValueError(
            "Rational chart tokens do not round to the retained chart array."
        )
    return canonical


def material_point(chart: _ExactTriangle, point: _ExactPoint, /) -> _ExactPoint:
    x, y = point
    weight0 = Fraction(1) - x - y
    return (
        weight0 * chart[0][0] + x * chart[1][0] + y * chart[2][0],
        weight0 * chart[0][1] + x * chart[1][1] + y * chart[2][1],
    )


def material_reference(chart: _ExactTriangle, point: _ExactPoint, /) -> _ExactPoint:
    first = tuple(chart[1][axis] - chart[0][axis] for axis in range(2))
    second = tuple(chart[2][axis] - chart[0][axis] for axis in range(2))
    delta = tuple(point[axis] - chart[0][axis] for axis in range(2))
    determinant = first[0] * second[1] - first[1] * second[0]
    if not determinant:
        raise ValueError("A source material chart is singular.")
    return (
        (delta[0] * second[1] - delta[1] * second[0]) / determinant,
        (first[0] * delta[1] - first[1] * delta[0]) / determinant,
    )


def material_piece_id(
    source_id: int,
    target_id: int,
    source_vertices: _ExactTriangle,
    target_vertices: _ExactTriangle,
    polygon: tuple[_ExactPoint, ...],
    labels: tuple[int, ...],
    fan: tuple[int, int, int],
    /,
) -> str:
    def tokens(
        points: tuple[_ExactPoint, ...],
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(
            tuple((value.numerator, value.denominator) for value in point)
            for point in points
        )

    return canonical_fingerprint(
        {
            "kind": "native-exact-surface-material-piece",
            "source": source_id,
            "target": target_id,
            "source_reference": tokens(source_vertices),
            "target_reference": tokens(target_vertices),
            "native_polygon": tokens(polygon),
            "supports": labels,
            "fan": fan,
        }
    )


class SurfaceChartWitness(StrictModule, NonTrainableState):
    """Cell-ID charts with separate source-atlas and actual root-mesh ancestry."""

    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    cell_global_ids: Array
    patches: Array
    charts: Array
    source_root_cell_ids: Array
    source_piece_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)
    chart_rational_tokens: _ChartRationalBank = eqx.field(static=True)
    root_mesh_cell_bindings: tuple[SurfaceRootMeshCellBinding | None, ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        *,
        geometry_id: str,
        topology_id: str,
        domain_id: str,
        cell_kind: str = "triangle",
        cell_global_ids: ArrayLike,
        patches: ArrayLike,
        charts: ArrayLike,
        source_root_cell_ids: ArrayLike,
        source_piece_ids: tuple[tuple[int, ...], ...],
        geometry_entity_ids: tuple[str, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
        chart_rational_tokens: _ChartRationalBank | None = None,
        root_mesh_cell_bindings: Sequence[SurfaceRootMeshCellBinding | None],
    ) -> None:
        bindings_raw = tuple(root_mesh_cell_bindings)
        bindings: list[SurfaceRootMeshCellBinding | None] = []
        for binding in bindings_raw:
            if binding is None:
                bindings.append(None)
                continue
            if not isinstance(binding, SurfaceRootMeshCellBinding):
                raise TypeError(
                    "Each root mesh chart row must be a SurfaceRootMeshCellBinding or None."
                )
            geometry_identity = canonical_identifier(
                binding.root_geometry_id, "root_geometry_id"
            )
            topology_identity = canonical_identifier(
                binding.root_topology_id, "root_topology_id"
            )
            root_cell = binding.root_cell_global_id
            if isinstance(root_cell, (bool, np.bool_)) or not isinstance(
                root_cell, (int, np.integer)
            ):
                raise TypeError(
                    "root_cell_global_id must be an integer global CellMesh cell ID."
                )
            root_cell_id = int(root_cell)
            if root_cell_id < 0 or root_cell_id > np.iinfo(np.int64).max:
                raise ValueError("root_cell_global_id must be a nonnegative int64 value.")
            bindings.append(
                SurfaceRootMeshCellBinding(
                    geometry_identity, topology_identity, root_cell_id
                )
            )
        ids = np.asarray(cell_global_ids)
        patch = np.asarray(patches)
        chart = np.asarray(charts, dtype=np.float64)
        source_roots = np.asarray(source_root_cell_ids)
        pieces = tuple(tuple(row) for row in source_piece_ids)
        if (
            ids.ndim != 1
            or not ids.size
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or patch.shape != ids.shape
            or patch.dtype.kind not in "iu"
            or np.any(patch < 0)
            or source_roots.shape != ids.shape
            or source_roots.dtype.kind not in "iu"
            or np.any(source_roots < 0)
            or np.any(source_roots > np.iinfo(np.int64).max)
            or chart.shape != (ids.size, _chart_corner_count(cell_kind), 2)
            or not np.all(np.isfinite(chart))
            or len(bindings) != ids.size
            or len(pieces) != ids.size
            or any(
                not row
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    or int(value) > np.iinfo(np.int64).max
                    for value in row
                )
                for row in pieces
            )
            or any(
                int(row[0]) != int(patch_value)
                for row, patch_value in zip(pieces, patch.tolist(), strict=True)
            )
            or len(geometry_entity_ids) != ids.size
            or len(occurrence_paths) != ids.size
            or any(
                not isinstance(value, str) or not value for value in geometry_entity_ids
            )
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in occurrence_paths
            )
            or any(
                not isinstance(value, str) or not value
                for value in (geometry_id, topology_id, domain_id)
            )
        ):
            raise ValueError(
                "Surface chart witness rows and source identities must align."
            )
        rational_bank = _validated_chart_rational_tokens(
            chart_rational_tokens, chart, ids.size, cell_kind=cell_kind
        )
        charts_exact = tuple(
            _exact_chart_from_tokens(rational_bank, row) for row in range(ids.size)
        )
        if any(not _area(value) for value in charts_exact):
            raise ValueError("Surface chart witness contains a degenerate material cell.")
        order = np.argsort(ids, kind="stable")
        sorted_source_roots = np.asarray(source_roots[order], dtype=np.int64)
        sorted_source_pieces = tuple(pieces[int(row)] for row in order)
        sorted_ids = np.asarray(ids[order], dtype=np.int64)
        sorted_patches = np.asarray(patch[order], dtype=np.int32)
        sorted_charts = np.asarray(chart[order], dtype=np.float64)
        sorted_chart_tokens = tuple(rational_bank[int(row)] for row in order)
        sorted_root_mesh_bindings = tuple(bindings[int(row)] for row in order)
        self.geometry_id = geometry_id
        self.topology_id = topology_id
        self.domain_id = domain_id
        self.cell_kind = cell_kind
        self.cell_global_ids = jnp.asarray(sorted_ids)
        self.source_root_cell_ids = jnp.asarray(sorted_source_roots)
        self.source_piece_ids = sorted_source_pieces
        self.patches = jnp.asarray(sorted_patches)
        self.charts = jnp.asarray(sorted_charts)
        self.geometry_entity_ids = tuple(geometry_entity_ids[int(row)] for row in order)
        self.occurrence_paths = tuple(occurrence_paths[int(row)] for row in order)
        self.chart_rational_tokens = sorted_chart_tokens
        self.root_mesh_cell_bindings = sorted_root_mesh_bindings
        self.witness_id = canonical_fingerprint(
            {
                "kind": "surface-chart-witness",
                "geometry": geometry_id,
                "topology": topology_id,
                "domain": domain_id,
                "cell_kind": cell_kind,
                "cells": array_tree_fingerprint(sorted_ids),
                "source_root_cells": array_tree_fingerprint(sorted_source_roots),
                "source_pieces": sorted_source_pieces,
                "patches": array_tree_fingerprint(sorted_patches),
                "charts": array_tree_fingerprint(sorted_charts),
                "entities": self.geometry_entity_ids,
                "occurrences": self.occurrence_paths,
                "chart_rational_tokens": self.chart_rational_tokens,
                "root_mesh_cell_bindings": tuple(
                    None
                    if binding is None
                    else (
                        binding.root_geometry_id,
                        binding.root_topology_id,
                        binding.root_cell_global_id,
                    )
                    for binding in self.root_mesh_cell_bindings
                ),
            }
        )

    def exact_chart(self, row: int, /) -> _ExactChart:
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < len(self.source_piece_ids)
        ):
            raise IndexError("surface chart row is out of range.")
        return _exact_chart_from_tokens(self.chart_rational_tokens, row)


class SurfaceChartWitnessArchiveRecipe(NamedTuple):
    """Archive form of a surface chart witness with its per-row banks as columns.

    Source pieces, geometry entities, occurrence paths, exact chart tokens and
    root-mesh bindings are one static value per chart row; the archive stores
    them as array columns instead of one manifest node per row. Root bindings
    split into a ``(root_geometry_id, root_topology_id)`` value table (``None``
    for unbound rows) and int64 root cell IDs (``-1`` where unbound).
    :meth:`restore` rebuilds through the witness constructor and requires the
    recomputed witness identity to equal the stored one.
    """

    geometry_id: str
    topology_id: str
    domain_id: str
    cell_kind: str
    cell_global_ids: Array
    patches: Array
    charts: Array
    source_root_cell_ids: Array
    source_piece_ids: IntegerRowsArchiveRecipe
    geometry_entity_ids: TextColumnArchiveRecipe
    occurrence_paths: StaticColumnArchiveRecipe
    chart_rational_tokens: ExactTensorArchiveRecipe
    root_mesh_owners: StaticColumnArchiveRecipe
    root_mesh_cell_ids: np.ndarray
    witness_id: str

    @classmethod
    def from_witness(
        cls, value: SurfaceChartWitness, /
    ) -> SurfaceChartWitnessArchiveRecipe:
        if type(value) is not SurfaceChartWitness:
            raise TypeError("value must be SurfaceChartWitness.")
        bindings = value.root_mesh_cell_bindings
        return cls(
            value.geometry_id,
            value.topology_id,
            value.domain_id,
            value.cell_kind,
            value.cell_global_ids,
            value.patches,
            value.charts,
            value.source_root_cell_ids,
            IntegerRowsArchiveRecipe.encode(value.source_piece_ids),
            TextColumnArchiveRecipe.encode(value.geometry_entity_ids),
            StaticColumnArchiveRecipe.encode(value.occurrence_paths),
            ExactTensorArchiveRecipe.encode(value.chart_rational_tokens, rational=False),
            StaticColumnArchiveRecipe.encode(
                tuple(
                    None
                    if binding is None
                    else (binding.root_geometry_id, binding.root_topology_id)
                    for binding in bindings
                )
            ),
            np.asarray(
                [
                    -1 if binding is None else binding.root_cell_global_id
                    for binding in bindings
                ],
                dtype=np.int64,
            ).reshape((len(bindings),)),
            value.witness_id,
        )

    def restore(self) -> SurfaceChartWitness:
        owners = self.root_mesh_owners.decode()
        cells = np.asarray(self.root_mesh_cell_ids)
        if cells.dtype != np.int64 or cells.shape != (len(owners),):
            raise TypeError("Stored root-mesh cell IDs must be one int64 row per owner.")
        bindings: list[SurfaceRootMeshCellBinding | None] = []
        for owner, cell in zip(owners, cells.tolist(), strict=True):
            if owner is None:
                if cell != -1:
                    raise ValueError("An unbound chart row cannot name a root cell.")
                bindings.append(None)
                continue
            if not isinstance(owner, tuple) or len(owner) != 2:
                raise TypeError("Stored root-mesh owners must be identity pairs.")
            bindings.append(SurfaceRootMeshCellBinding(owner[0], owner[1], cell))
        tokens = self.chart_rational_tokens.decode()
        restored = SurfaceChartWitness(
            geometry_id=self.geometry_id,
            topology_id=self.topology_id,
            domain_id=self.domain_id,
            cell_kind=self.cell_kind,
            cell_global_ids=self.cell_global_ids,
            patches=self.patches,
            charts=self.charts,
            source_root_cell_ids=self.source_root_cell_ids,
            source_piece_ids=self.source_piece_ids.decode(),
            geometry_entity_ids=self.geometry_entity_ids.decode(),
            occurrence_paths=self.occurrence_paths.decode(),
            chart_rational_tokens=tokens,
            root_mesh_cell_bindings=bindings,
        )
        if restored.witness_id != self.witness_id:
            raise ValueError(
                "Stored surface chart witness differs from its recomputed identity."
            )
        return restored


class PreparedSurfaceChartPiece(NamedTuple):
    source_row: int
    target_row: int
    source_cell_global_id: int
    target_cell_global_id: int
    exact_source_reference_vertices: _ExactTriangle
    exact_target_reference_vertices: _ExactTriangle
    source_reference_area: Fraction
    target_reference_area: Fraction
    native_boundary_labels: tuple[int, ...]
    exact_native_source_polygon: tuple[_ExactPoint, ...]
    native_polygon_vertex_indices: tuple[int, int, int]
    piece_id: str


class PreparedSurfaceChartOccurrence(NamedTuple):
    patch: int
    geometry_entity_id: str
    occurrence_path: tuple[str, ...]
    source_rows: Array
    target_rows: Array
    exact_source_material_charts: tuple[_ExactTriangle, ...]
    exact_target_material_charts: tuple[_ExactTriangle, ...]
    pieces: tuple[PreparedSurfaceChartPiece, ...]
    root_pieces: tuple[PreparedSurfaceChartPiece, ...]
    material_root_witness_id: str
    displacement_bounds: Array
    source_domain_coverage: str
    target_domain_coverage: str


class PreparedSurfaceChartDeformation(StrictModule, NonTrainableState):
    """Exact native material partition and endpoint-bound surface deformation."""

    __strict_contract__ = True

    source_witness: SurfaceChartWitness
    target_witness: SurfaceChartWitness
    material_root_witness: SurfaceChartWitness
    source_chart_cover: MeshingDomainBoundarySource
    material_root_chart_cover: MeshingDomainBoundarySource | None
    target_chart_cover: MeshingDomainBoundarySource
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    occurrences: tuple[PreparedSurfaceChartOccurrence, ...]
    source_geometry_id: Identifier = eqx.field(static=True)
    target_geometry_id: Identifier = eqx.field(static=True)
    source_topology_id: Identifier = eqx.field(static=True)
    target_topology_id: Identifier = eqx.field(static=True)
    domain_id: Identifier = eqx.field(static=True)
    source_fidelity_bounds: Array
    target_fidelity_bounds: Array
    source_polynomial_chord_bounds: Array
    source_corner_enclosure_bounds: Array
    source_continuous_chord_bounds: Array
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    source_fidelity_certificate: SourceFidelityCertificate
    target_fidelity_certificate: SourceFidelityCertificate
    maximum_displacement_bound: float = eqx.field(static=True)
    deformation_id: Identifier = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_domain_coverage(self) -> str:
        return "certified" if self._coverage_is_complete(target=False) else "unresolved"

    @property
    def target_domain_coverage(self) -> str:
        return "certified" if self._coverage_is_complete(target=True) else "unresolved"

    def material_charts(self, *, target: bool) -> np.ndarray:
        mesh = self.target_mesh if target else self.source_mesh
        ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        result = np.empty((ids.size, 3, 2), dtype=object)
        seen = np.zeros(ids.size, dtype=np.bool_)
        for occurrence in self.occurrences:
            rows = np.asarray(
                occurrence.target_rows if target else occurrence.source_rows,
                dtype=np.int64,
            )
            charts = (
                occurrence.exact_target_material_charts
                if target
                else occurrence.exact_source_material_charts
            )
            if (
                rows.shape != (len(charts),)
                or np.any(rows < 0)
                or np.any(rows >= ids.size)
                or np.any(seen[rows])
            ):
                raise ValueError(
                    "Material chart occurrences duplicate or exceed physical cell rows."
                )
            result[rows] = np.asarray(charts, dtype=object)
            seen[rows] = True
        if not np.all(seen):
            raise ValueError(
                "Material chart occurrences do not cover every physical cell."
            )
        return result

    def _coverage_is_complete(self, *, target: bool) -> bool:
        try:
            self._require_partitions()
            cover = self.target_chart_cover if target else self.source_chart_cover
            if cover is None:
                return False
            retained = cover.boundary_chart_cover()
            return bool(retained.complete)
        except (TypeError, ValueError, IndexError):
            return False

    def _require_partitions(self) -> None:
        if not self.occurrences:
            raise ValueError(
                "A material deformation needs at least one patch occurrence."
            )
        source_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in self.source_mesh.blocks
            ]
        )
        target_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in self.target_mesh.blocks
            ]
        )
        maximum_displacement = 0.0
        source_covered = []
        target_covered = []
        patches = set()
        for occurrence in self.occurrences:
            if (
                occurrence.patch in patches
                or occurrence.material_root_witness_id
                != self.material_root_witness.witness_id
            ):
                raise ValueError(
                    "Surface chart occurrences duplicate a patch or change material root."
                )
            patches.add(occurrence.patch)
            old_rows = np.asarray(occurrence.source_rows, dtype=np.int64)
            new_rows = np.asarray(occurrence.target_rows, dtype=np.int64)
            old_charts = occurrence.exact_source_material_charts
            new_charts = occurrence.exact_target_material_charts
            if (
                old_rows.shape != (len(old_charts),)
                or new_rows.shape != (len(new_charts),)
                or np.any(old_rows < 0)
                or np.any(old_rows >= source_ids.size)
                or np.any(new_rows < 0)
                or np.any(new_rows >= target_ids.size)
            ):
                raise ValueError("Surface chart occurrence rows are misaligned.")
            displacement = np.asarray(occurrence.displacement_bounds, dtype=np.float64)
            if (
                displacement.shape != (len(occurrence.pieces),)
                or not np.all(np.isfinite(displacement))
                or np.any(displacement < 0.0)
            ):
                raise ValueError(
                    "Surface occurrence displacement bounds must align with exact pieces."
                )
            if displacement.size:
                maximum_displacement = max(
                    maximum_displacement, float(np.max(displacement))
                )
            source_covered.extend(old_rows.tolist())
            target_covered.extend(new_rows.tolist())
            root_rows = np.flatnonzero(
                np.asarray(self.material_root_witness.patches) == occurrence.patch
            )
            root_charts = tuple(
                material_triangle(self.material_root_witness.exact_chart(int(row)))
                for row in root_rows
            )
            root_ids = np.asarray(
                self.material_root_witness.cell_global_ids, dtype=np.int64
            )[root_rows]
            old_chart_ids = source_ids[old_rows]
            if occurrence.root_pieces:
                require_material_partition(
                    occurrence.root_pieces,
                    root_charts,
                    old_charts,
                    root_ids,
                    old_chart_ids,
                )
            elif root_charts != old_charts or not np.array_equal(root_ids, old_chart_ids):
                raise ValueError(
                    "Surface source charts lose their retained material-root partition."
                )
            require_material_partition(
                occurrence.pieces,
                old_charts,
                new_charts,
                old_chart_ids,
                target_ids[new_rows],
            )
            self._require_occurrence_witness(
                self.source_witness, self.source_mesh, occurrence, old_rows, target=False
            )
            self._require_occurrence_witness(
                self.target_witness, self.target_mesh, occurrence, new_rows, target=True
            )
        if sorted(source_covered) != list(range(source_ids.size)) or sorted(
            target_covered
        ) != list(range(target_ids.size)):
            raise ValueError(
                "Surface chart occurrences must cover every physical cell exactly once."
            )
        if patches != set(map(int, np.asarray(self.source_witness.patches))):
            raise ValueError("Surface chart deformation omits source patches.")
        if maximum_displacement > self.maximum_displacement_bound:
            raise ValueError(
                "Surface deformation bound is smaller than a piecewise displacement."
            )

    @staticmethod
    def _require_occurrence_witness(
        witness: SurfaceChartWitness,
        mesh: CellMesh,
        occurrence: PreparedSurfaceChartOccurrence,
        rows: np.ndarray,
        /,
        *,
        target: bool,
    ) -> None:
        physical_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        declared_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
        charts = (
            occurrence.exact_target_material_charts
            if target
            else occurrence.exact_source_material_charts
        )
        witness_rows = np.searchsorted(declared_ids, physical_ids[rows])
        if (
            len(charts) != rows.size
            or np.any(witness_rows >= declared_ids.size)
            or not np.array_equal(declared_ids[witness_rows], physical_ids[rows])
            or np.any(np.asarray(witness.patches)[witness_rows] != occurrence.patch)
            or any(
                witness.geometry_entity_ids[index] != occurrence.geometry_entity_id
                or witness.occurrence_paths[index] != occurrence.occurrence_path
                for index in witness_rows
            )
        ):
            raise ValueError(
                "Surface material rows change source entity or occurrence identity."
            )

    @checked
    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        if (
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
            source_mesh.topology_id,
            target_mesh.topology_id,
        ) != (
            self.source_mesh.mesh_id,
            self.target_mesh.mesh_id,
            self.source_geometry_id,
            self.target_geometry_id,
            self.source_topology_id,
            self.target_topology_id,
        ):
            raise ValueError(
                "Surface deformation is stale for these endpoint meshes or maps."
            )
        if (
            self.source_witness.domain_id != self.domain_id
            or self.target_witness.domain_id != self.domain_id
            or self.material_root_witness.domain_id != self.domain_id
        ):
            raise ValueError(
                "Surface deformation witnesses disagree on their source domain."
            )
        from ..geometry._domain_boundary import (
            MeshingDomainBoundarySource as BoundarySource,
        )
        from ..geometry._global_embedding import (
            GlobalEmbeddingCertificate as EmbeddingCertificate,
        )
        from ..geometry._source_fidelity_certificate import (
            SourceFidelityCertificate as FidelityCertificate,
        )

        if (
            not isinstance(self.source_chart_cover, BoundarySource)
            or not isinstance(self.material_root_chart_cover, BoundarySource)
            or not isinstance(self.target_chart_cover, BoundarySource)
        ):
            raise TypeError(
                "Surface deformation covers require MeshingDomainBoundarySource."
            )
        if not isinstance(self.source_embedding, EmbeddingCertificate) or not isinstance(
            self.target_embedding, EmbeddingCertificate
        ):
            raise TypeError(
                "Surface deformation embeddings require GlobalEmbeddingCertificate."
            )
        if not isinstance(
            self.source_fidelity_certificate, FidelityCertificate
        ) or not isinstance(self.target_fidelity_certificate, FidelityCertificate):
            raise TypeError(
                "Surface deformation fidelity requires SourceFidelityCertificate."
            )
        self._require_partitions()
        for cover in (
            self.source_chart_cover,
            self.material_root_chart_cover,
            self.target_chart_cover,
        ):
            if not cover.boundary_chart_cover().complete:
                raise ValueError(
                    "Surface deformation lacks complete source boundary chart coverage."
                )
        for mesh, geometry, validity, embedding, fidelity, source in (
            (
                source_mesh,
                source_geometry,
                self.source_validity,
                self.source_embedding,
                self.source_fidelity_certificate,
                self.source_chart_cover,
            ),
            (
                target_mesh,
                target_geometry,
                self.target_validity,
                self.target_embedding,
                self.target_fidelity_certificate,
                self.target_chart_cover,
            ),
        ):
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            if embedding.status != "certified":
                raise ValueError(
                    "Surface deformation requires globally embedded endpoints."
                )
            fidelity.binding.require(mesh, geometry, source=source)
            if fidelity.status != "certified":
                raise ValueError(
                    "Surface deformation requires certified source fidelity."
                )
        for values in (
            self.source_fidelity_bounds,
            self.target_fidelity_bounds,
            self.source_polynomial_chord_bounds,
            self.source_corner_enclosure_bounds,
            self.source_continuous_chord_bounds,
            self.source_cell_measures,
            self.target_cell_measures,
            self.source_measure_errors,
            self.target_measure_errors,
        ):
            host = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(host)) or np.any(host < 0.0):
                raise ValueError(
                    "Surface deformation bounds and measures must be finite and nonnegative."
                )
        if np.any(np.asarray(self.source_cell_measures) <= 0.0) or np.any(
            np.asarray(self.target_cell_measures) <= 0.0
        ):
            raise ValueError("Surface physical cell measures must be strictly positive.")
        if (
            not np.isfinite(self.maximum_displacement_bound)
            or self.maximum_displacement_bound < 0.0
        ):
            raise ValueError("Surface displacement bound must be finite and nonnegative.")
        self.source_geometry.resolve(source_mesh)
        self.target_geometry.resolve(target_mesh)
        expected_id = canonical_fingerprint(
            {
                "kind": "prepared-surface-chart-deformation",
                "source": self.source_witness.witness_id,
                "target": self.target_witness.witness_id,
                "material_root": self.material_root_witness.witness_id,
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "source_mesh": source_mesh.mesh_id,
                "target_mesh": target_mesh.mesh_id,
                "domain": self.domain_id,
                "overlaps": tuple(
                    tuple(piece.piece_id for piece in item.pieces)
                    for item in self.occurrences
                ),
                "root_partitions": tuple(
                    tuple(piece.piece_id for piece in item.root_pieces)
                    for item in self.occurrences
                ),
                "certificates": (
                    self.source_validity.certificate_id,
                    self.target_validity.certificate_id,
                    self.source_embedding.certificate_id,
                    self.target_embedding.certificate_id,
                    self.source_fidelity_certificate.certificate_id,
                    self.target_fidelity_certificate.certificate_id,
                ),
                "source_bounds": array_tree_fingerprint(self.source_fidelity_bounds),
                "target_bounds": array_tree_fingerprint(self.target_fidelity_bounds),
                "source_polynomial_chord_bounds": array_tree_fingerprint(
                    self.source_polynomial_chord_bounds
                ),
                "source_corner_enclosure_bounds": array_tree_fingerprint(
                    self.source_corner_enclosure_bounds
                ),
                "source_continuous_chord_bounds": array_tree_fingerprint(
                    self.source_continuous_chord_bounds
                ),
                "source_cell_measures": array_tree_fingerprint(self.source_cell_measures),
                "target_cell_measures": array_tree_fingerprint(self.target_cell_measures),
                "source_measure_errors": array_tree_fingerprint(
                    self.source_measure_errors
                ),
                "target_measure_errors": array_tree_fingerprint(
                    self.target_measure_errors
                ),
                "occurrences": tuple(
                    (
                        item.patch,
                        item.geometry_entity_id,
                        item.occurrence_path,
                        array_tree_fingerprint(item.displacement_bounds),
                    )
                    for item in self.occurrences
                ),
                "displacement": self.maximum_displacement_bound,
            }
        )
        if expected_id != self.deformation_id:
            raise ValueError("Retained surface chart deformation identity is stale.")

    @classmethod
    def from_stored_fields(
        cls, value: PreparedSurfaceChartDeformation, /
    ) -> PreparedSurfaceChartDeformation:
        value.require_bound(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
        )
        # Stored identities are not proof authority: renew both proofs.
        value.source_fidelity_certificate.revalidate(
            value.source_mesh,
            value.source_geometry,
            value.source_fidelity_certificate.source,
        )
        value.target_fidelity_certificate.revalidate(
            value.target_mesh,
            value.target_geometry,
            value.target_fidelity_certificate.source,
        )
        return value


def require_material_partition(
    pieces: tuple[PreparedSurfaceChartPiece, ...],
    source: tuple[_ExactTriangle, ...],
    target: tuple[_ExactTriangle, ...],
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    /,
) -> None:
    source_areas = [Fraction(0)] * len(source)
    target_areas = [Fraction(0)] * len(target)
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    for piece in pieces:
        old, new = piece.source_row, piece.target_row
        if (
            not 0 <= old < len(source)
            or not 0 <= new < len(target)
            or piece.source_cell_global_id != int(source_ids[old])
            or piece.target_cell_global_id != int(target_ids[new])
        ):
            raise ValueError("Material piece changes its scientific cell binding.")
        mapped: _ExactTriangle = (
            material_reference(target[new], source[old][0]),
            material_reference(target[new], source[old][1]),
            material_reference(target[new], source[old][2]),
        )
        origin = mapped[0]
        first = tuple(mapped[1][axis] - origin[axis] for axis in range(2))
        second = tuple(mapped[2][axis] - origin[axis] for axis in range(2))
        planes = (
            (Fraction(-1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(-1), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
            (-first[0], -second[0], origin[0]),
            (-first[1], -second[1], origin[1]),
            (first[0] + first[1], second[0] + second[1], 1 - origin[0] - origin[1]),
        )
        polygon, labels = piece.exact_native_source_polygon, piece.native_boundary_labels
        if (
            len(polygon) < 3
            or len(set(polygon)) != len(polygon)
            or len(labels) != len(polygon)
            or _area(polygon) <= 0
        ):
            raise ValueError(
                "Native material polygon must be positively oriented with distinct vertices."
            )
        for slot, label in enumerate(labels):
            if not 0 <= label < len(planes):
                raise ValueError(
                    "Native material boundary names an absent supporting face."
                )
            point, following = polygon[slot], polygon[(slot + 1) % len(polygon)]
            previous = polygon[slot - 1]
            if (point[0] - previous[0]) * (following[1] - point[1]) - (
                point[1] - previous[1]
            ) * (following[0] - point[0]) <= 0:
                raise ValueError("Native material polygon must be strictly convex.")
            if any(a * point[0] + b * point[1] > c for a, b, c in planes):
                raise ValueError(
                    "Native material polygon leaves a source chart halfplane."
                )
            a, b, c = planes[label]
            if any(a * value[0] + b * value[1] != c for value in (point, following)):
                raise ValueError(
                    "Native material boundary changes its exact supporting face."
                )
        fan = piece.native_polygon_vertex_indices
        if fan[0] != 0 or fan[2] != fan[1] + 1 or not 1 <= fan[1] < len(polygon) - 1:
            raise ValueError("Material piece loses its ordered polygon fan partition.")
        key = (old, new, fan)
        if key in seen:
            raise ValueError("Native material fan piece is duplicated.")
        seen.add(key)
        old_vertices = (polygon[fan[0]], polygon[fan[1]], polygon[fan[2]])
        new_vertices: _ExactTriangle = (
            material_point(mapped, old_vertices[0]),
            material_point(mapped, old_vertices[1]),
            material_point(mapped, old_vertices[2]),
        )
        if (
            old_vertices != piece.exact_source_reference_vertices
            or new_vertices != piece.exact_target_reference_vertices
        ):
            raise ValueError(
                "Material piece changes its exact source/target reference map."
            )
        old_area = abs(_area(old_vertices))
        new_area = abs(_area(new_vertices))
        if (
            not old_area
            or not new_area
            or old_area != piece.source_reference_area
            or new_area != piece.target_reference_area
        ):
            raise ValueError("Material piece changes its positive reference measure.")
        if piece.piece_id != material_piece_id(
            piece.source_cell_global_id,
            piece.target_cell_global_id,
            old_vertices,
            new_vertices,
            polygon,
            labels,
            fan,
        ):
            raise ValueError("Material piece source identity is stale.")
        source_areas[old] += old_area
        target_areas[new] += new_area
    if any(value != Fraction(1, 2) for value in (*source_areas, *target_areas)):
        raise ValueError(
            "Material pieces do not form complete exact reference partitions."
        )


__all__ = [
    "PreparedSurfaceChartDeformation",
    "PreparedSurfaceChartOccurrence",
    "PreparedSurfaceChartPiece",
    "SurfaceChartWitness",
    "SurfaceChartWitnessArchiveRecipe",
    "SurfaceRootMeshCellBinding",
    "exact_chart_bank",
    "rational_chart_tokens",
    "material_point",
    "material_reference",
    "material_piece_id",
    "require_material_partition",
]
