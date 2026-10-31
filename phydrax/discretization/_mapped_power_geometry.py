#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical images of exact reference power vertices under mapped source roots."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._exact import solve_exact
from ._exact_algebraic_geometry import coefficient_payload
from ._exact_power_geometry import ExactPoint3, ExactPowerCellGeometrySource


type CarrierRootChart = tuple[ExactPoint3, ExactPoint3, ExactPoint3, ExactPoint3]


def _integer_rows(value: Any, name: str, /) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer vector.")
    return rows.astype(np.int64, copy=False)


def _renewed_mapped_domain(stored: Any, /) -> Any:
    """Rebuild a stored mapped reference domain through its constructors.

    Archived coordinate elements are canonical recipes, not executable
    elements, so the root maps are only usable after the reference mesh, the
    root coordinate geometry and the reference domain are renewed. The renewed
    domain identity covers all of them and must equal the stored one.
    """
    from ..geometry._domain_coverage import PiecewiseLinearDomain
    from ..geometry._mapped_reference_domain import MappedReferenceDomain
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh

    if type(stored) is not MappedReferenceDomain:
        raise TypeError("Stored mapped power geometry lost its mapped reference domain.")
    reference_mesh = CellMesh.from_stored_fields(stored.reference_mesh)
    stored_geometry = stored.source_geometry
    source_geometry = CellGeometrySpec.from_stored_fields(
        stored_geometry,
        source_mesh=(None if stored_geometry.source_mesh_id is None else reference_mesh),
        source_map=stored_geometry.source_map,
        restriction_source=stored_geometry.restriction_source,
        elements=dict(
            zip(stored_geometry.block_names, stored_geometry.elements, strict=True)
        ),
    )
    stored_reference = stored.reference_domain
    reference_domain = PiecewiseLinearDomain(
        stored_reference.vertices,
        stored_reference.facets,
        stored_reference.facet_regions,
        stored_reference.region_ids,
        source_id=stored_reference.source_id,
        region_occurrence_paths=stored_reference.region_occurrence_paths,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        source_geometry,
        stored.cell_regions,
        source_id=stored.source_id,
        source_revision=stored.source_revision,
        coordinate_contract=stored.coordinate_contract,
        root_vertex_occurrence_paths=stored.root_vertex_occurrence_paths,
        root_cell_occurrence_paths=stored.root_cell_occurrence_paths,
        reference_domain_vertex_root_ids=(
            None
            if stored.reference_domain_vertex_root_ids is None
            else np.asarray(stored.reference_domain_vertex_root_ids, dtype=np.int64)
        ),
    )
    if (
        reference_domain.domain_id != stored_reference.domain_id
        or domain.domain_id != stored.domain_id
    ):
        raise ValueError("Stored mapped reference domain differs from its renewal.")
    return domain


@final
class MappedPowerCellGeometrySource(StrictModule, NonTrainableState):
    """Exact physical power vertices bound to their reference power cells.

    Restricted power cells are built in the reference coordinates of a mapped
    PLC carrier. Every carrier tetrahedron retains the exact root-reference
    images of its four corners, so a reference power vertex with exact carrier
    barycentrics has one exact root-reference point. Its physical coordinate is
    the root map evaluated there. Each restricted piece keeps its target cell
    and carrier tetrahedron; chart certification uses those rows to restrict
    the carrier root chart to every published polyhedron.
    """

    reference_source: ExactPowerCellGeometrySource
    carrier_root_cell_ids: jnp.ndarray
    carrier_root_charts: tuple[CarrierRootChart, ...] = eqx.field(static=True)
    piece_cells: jnp.ndarray
    piece_tets: jnp.ndarray
    runtime_vertices: jnp.ndarray
    exact_vertices: tuple[ExactPoint3, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_source: ExactPowerCellGeometrySource,
        carrier_root_cell_ids: Any,
        carrier_root_charts: tuple[CarrierRootChart, ...],
        piece_cells: Any,
        piece_tets: Any,
        cell_count: int,
        /,
    ) -> None:
        from ..geometry._mapped_reference_domain import MappedReferenceDomain

        if not isinstance(reference_source, ExactPowerCellGeometrySource):
            raise TypeError("reference_source must be ExactPowerCellGeometrySource.")
        domain = reference_source.domain_source
        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError(
                "Mapped power geometry requires a reference source bound to its MappedReferenceDomain."
            )
        if reference_source.periodic_source_context is not None:
            raise ValueError(
                "Mapped power geometry does not admit periodic power sources."
            )
        if (reference_source.source_id, reference_source.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError(
                "Reference power vertices and the mapped domain bind different sources."
            )
        carrier_tets = np.asarray(reference_source.carrier_tets, dtype=np.int64)
        root_ids = _integer_rows(carrier_root_cell_ids, "carrier_root_cell_ids")
        cells = _integer_rows(piece_cells, "piece_cells")
        tets = _integer_rows(piece_tets, "piece_tets")
        if (
            root_ids.shape != (carrier_tets.shape[0],)
            or len(carrier_root_charts) != carrier_tets.shape[0]
            or any(
                len(chart) != 4
                or any(
                    len(point) != 3
                    or any(not isinstance(value, Fraction) for value in point)
                    for point in chart
                )
                for chart in carrier_root_charts
            )
        ):
            raise ValueError(
                "Every carrier tetrahedron needs one root cell and four exact root-reference corners."
            )
        known_roots = {int(value) for value in domain.root_cell_ids}
        if any(int(value) not in known_roots for value in root_ids):
            raise ValueError("A carrier tetrahedron names an absent mapped root cell.")
        if (
            type(cell_count) is not int
            or cell_count < 1
            or cells.shape != tets.shape
            or np.any(cells < 0)
            or np.any(cells >= cell_count)
            or np.any(tets < 0)
            or np.any(tets >= carrier_tets.shape[0])
            or np.unique(cells).size != cell_count
        ):
            raise ValueError(
                "Restricted pieces must name published cells and carrier tetrahedra."
            )

        incident: dict[int, list[int]] = {}
        for tet_row, tet in enumerate(carrier_tets.tolist()):
            for vertex in tet:
                incident.setdefault(int(vertex), []).append(tet_row)
        carriers = np.asarray(reference_source.vertex_carriers, dtype=np.int64)
        output_rows = np.asarray(reference_source.output_vertex_indices, dtype=np.int64)
        reference_points: list[tuple[ExactPoint3, ...]] = []
        root_rows: list[int] = []
        for output_row, vertex in enumerate(output_rows.tolist()):
            carrier_row = carriers[vertex]
            carrier_ids = tuple(int(value) for value in carrier_row[carrier_row >= 0])
            weights = reference_source.carrier_barycentrics[output_row]
            containing = next(
                (
                    tet_row
                    for tet_row in incident.get(carrier_ids[0], ())
                    if set(carrier_ids) <= set(carrier_tets[tet_row].tolist())
                ),
                None,
            )
            if containing is None:
                raise ValueError(
                    "A reference power vertex carrier lies in no carrier tetrahedron."
                )
            corners = carrier_tets[containing].tolist()
            chart = carrier_root_charts[containing]
            reference_point: ExactPoint3 = (Fraction(0), Fraction(0), Fraction(0))
            for weight, carrier_id in zip(
                weights[: len(carrier_ids)], carrier_ids, strict=True
            ):
                corner = chart[corners.index(carrier_id)]
                reference_point = (
                    reference_point[0] + weight * corner[0],
                    reference_point[1] + weight * corner[1],
                    reference_point[2] + weight * corner[2],
                )
            reference_points.append((reference_point,))
            root_rows.append(int(root_ids[containing]))
        physical = domain.source_geometry.evaluate_exact(
            domain.reference_mesh,
            np.asarray(root_rows, dtype=np.int64),
            tuple(reference_points),
        )
        exact_vertices: list[ExactPoint3] = []
        for row in physical:
            point = row[0]
            if len(point) != 3 or any(not isinstance(value, Fraction) for value in point):
                raise ValueError(
                    "Mapped power vertices require exact rational root-map images."
                )
            exact_vertices.append(
                (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))
            )
        runtime = np.asarray(exact_vertices, dtype=np.float64)
        if not np.all(np.isfinite(runtime)):
            raise ValueError("Mapped power vertices must have finite binary64 shadows.")
        identity = canonical_fingerprint(
            {
                "kind": "mapped-power-cell-geometry-source",
                "reference_source": reference_source.source_map_id,
                "domain": domain.domain_id,
                "carrier_root_cell_ids": array_tree_fingerprint(root_ids),
                "carrier_root_charts": tuple(
                    tuple(
                        tuple(coefficient_payload(value) for value in point)
                        for point in chart
                    )
                    for chart in carrier_root_charts
                ),
                "pieces": array_tree_fingerprint((cells, tets)),
                "exact_vertices": tuple(
                    tuple(coefficient_payload(value) for value in point)
                    for point in exact_vertices
                ),
            }
        )
        self.reference_source = reference_source
        self.carrier_root_cell_ids = jnp.asarray(root_ids, dtype=jnp.int64)
        self.carrier_root_charts = carrier_root_charts
        self.piece_cells = jnp.asarray(cells, dtype=jnp.int64)
        self.piece_tets = jnp.asarray(tets, dtype=jnp.int64)
        self.runtime_vertices = jnp.asarray(runtime)
        self.exact_vertices = tuple(exact_vertices)
        self.source_id = reference_source.source_id
        self.source_revision = reference_source.source_revision
        self.domain_id = domain.domain_id
        self.source_map_id = identity

    @classmethod
    def from_stored_fields(
        cls,
        value: MappedPowerCellGeometrySource,
        /,
        *,
        reference_source: ExactPowerCellGeometrySource | None = None,
    ) -> MappedPowerCellGeometrySource:
        """Re-place every stored vertex through the renewed reference power map.

        The reference power source is rebuilt by its owner (or supplied already
        renewed by a caller that also renews its mapped domain), and the
        constructor re-derives the exact physical vertices from the stored root
        charts and restricted pieces. The renewed identity must match the stored
        one, so a changed chart, piece row or vertex image is refused.
        """
        if type(value) is not cls:
            raise TypeError("value must be MappedPowerCellGeometrySource.")
        if reference_source is None:
            if type(value.reference_source) is not ExactPowerCellGeometrySource:
                raise TypeError(
                    "Stored mapped power geometry lost its exact reference power source."
                )
            reference_source = ExactPowerCellGeometrySource.from_stored_fields(
                value.reference_source,
                domain_source=_renewed_mapped_domain(
                    value.reference_source.domain_source
                ),
            )
        cells = _integer_rows(np.asarray(value.piece_cells), "piece_cells")
        if cells.size == 0:
            raise ValueError("Stored mapped power geometry has no restricted pieces.")
        renewed = cls(
            reference_source,
            np.asarray(value.carrier_root_cell_ids),
            value.carrier_root_charts,
            cells,
            np.asarray(value.piece_tets),
            int(np.max(cells)) + 1,
        )
        if (
            renewed.source_map_id != value.source_map_id
            or renewed.source_id != value.source_id
            or renewed.source_revision != value.source_revision
            or renewed.domain_id != value.domain_id
            or renewed.exact_vertices != tuple(value.exact_vertices)
            or not np.array_equal(
                np.asarray(renewed.runtime_vertices), np.asarray(value.runtime_vertices)
            )
        ):
            raise ValueError(
                "Stored mapped power geometry differs from its renewed root-chart images."
            )
        return renewed

    @property
    def domain(self) -> Any:
        """The MappedReferenceDomain whose root maps place every vertex."""
        return self.reference_source.domain_source

    @property
    def rounded_vertices(self) -> np.ndarray:
        return np.asarray(self.runtime_vertices)

    def source_coordinates(self, indices: Any = None, /) -> tuple[ExactPoint3, ...]:
        """Return exact physical coordinates for selected output vertex rows."""
        if indices is None:
            return self.exact_vertices
        rows = _integer_rows(indices, "Mapped power coordinate selection")
        if np.any(rows < 0) or np.any(rows >= len(self.exact_vertices)):
            raise IndexError(
                "Mapped power coordinate selection is outside the vertex bank."
            )
        return tuple(self.exact_vertices[int(row)] for row in rows.tolist())

    def cell_piece_tetrahedra(self, /) -> tuple[tuple[int, ...], ...]:
        """Group the carrier tetrahedra of restricted pieces by published cell row."""
        cells = np.asarray(self.piece_cells, dtype=np.int64)
        tets = np.asarray(self.piece_tets, dtype=np.int64)
        order = np.argsort(cells, kind="stable")
        counts = np.bincount(cells, minlength=int(np.max(cells)) + 1)
        groups = np.split(tets[order], np.cumsum(counts)[:-1])
        return tuple(tuple(int(value) for value in group) for group in groups)

    def restricted_root_chart(
        self,
        carrier_tetrahedra: tuple[int, ...],
        reference_rows: Any,
        /,
    ) -> tuple[int, tuple[ExactPoint3, ...]]:
        """Map reference vertex rows through the one root chart of these pieces.

        Every carrier tetrahedron is affine in reference coordinates, so its
        four exact root-reference corners define one affine chart. The pieces
        of one cell must share their root cell and agree on that chart: the
        chart of the first tetrahedron must reproduce every other carrier
        tetrahedron's root-reference corners exactly.
        """
        rows = _integer_rows(reference_rows, "reference_rows")
        if not carrier_tetrahedra:
            raise ValueError("A published polyhedron has no restricted carrier piece.")
        roots = np.asarray(self.carrier_root_cell_ids, dtype=np.int64)
        root_ids = {int(roots[tet]) for tet in carrier_tetrahedra}
        if len(root_ids) != 1:
            raise ValueError(
                "A polyhedron's restricted pieces lie in more than one mapped root cell."
            )
        reference = self.reference_source
        carrier_tets = np.asarray(reference.carrier_tets, dtype=np.int64)
        carrier_points = np.asarray(reference.carrier_points, dtype=np.float64)
        reference_vertices = reference.source_coordinates()

        def carrier_point(row: int, /) -> ExactPoint3:
            x, y, z = (Fraction(float(value)) for value in carrier_points[row])
            return x, y, z

        first = carrier_tets[carrier_tetrahedra[0]].tolist()
        base = carrier_point(first[0])
        edges = tuple(
            tuple(carrier_point(first[corner])[axis] - base[axis] for corner in (1, 2, 3))
            for axis in range(3)
        )
        corners = self.carrier_root_charts[carrier_tetrahedra[0]]
        expected = tuple(
            (carrier_point(vertex), self.carrier_root_charts[tet][slot])
            for tet in carrier_tetrahedra[1:]
            for slot, vertex in enumerate(carrier_tets[tet].tolist())
        )
        queries = (
            *(reference_vertices[int(row)] for row in rows.tolist()),
            *(point for point, _ in expected),
        )
        barycentric = solve_exact(
            edges,
            tuple(
                tuple(query[axis] - base[axis] for query in queries) for axis in range(3)
            ),
            singular_message="A carrier tetrahedron of a mapped polyhedron is degenerate.",
        )
        images: list[ExactPoint3] = []
        for column in range(len(queries)):
            x, y, z = (
                corners[0][axis]
                + sum(
                    (
                        barycentric[corner][column]
                        * (corners[corner + 1][axis] - corners[0][axis])
                        for corner in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            images.append((x, y, z))
        if any(
            image != corner
            for image, (_, corner) in zip(images[rows.size :], expected, strict=True)
        ):
            raise ValueError(
                "The restricted pieces of one polyhedron carry different carrier root charts."
            )
        return next(iter(root_ids)), tuple(images[: rows.size])


__all__ = ["MappedPowerCellGeometrySource"]
