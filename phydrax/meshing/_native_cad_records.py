"""Source occurrence and association records of a native CAD solid.

``cad_mesh_solid`` classifies every surface vertex (corner, curve or face row
with its parameters), every boundary triangle (source face and the regions on
its two sides) and every curve edge (curve row and the curve parameters at its
ends). This module resolves those native arrays through the ``MeshingDomain``
occurrence tables into

- the B-Rep associations of every mesh dimension (vertices, edges, faces and
  cells) (:func:`native_cad_associations`), and
- :class:`NativeCadRecords`, the curve-edge authority a native CAD solid
  carries in its certification inputs: the constructed curve edges (roots),
  the exact curve association that partitions them, and the pole slivers.

Adaptation (which keeps the boundary) carries the records and the native
chart realization to its target (:meth:`NativeCadRecords.retarget`,
:func:`retarget_native_cad_surface`); the native verifiers re-prove the chart.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import CadChartResult, CadSurfaceResult, MeshcoreStatus
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry.brep._projection_contracts import brep_entity_id
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._native_cad_certificate import NativeCadSurfaceWitness


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._meshing_domain import MeshingDomain
    from ._lineage import MeshLineage, VertexInterpolationStencil
    from ._result import CellMeshingResult


_ROLES = (
    GeometrySourceEntityRole.VERTEX,
    GeometrySourceEntityRole.EDGE,
    GeometrySourceEntityRole.FACET,
    GeometrySourceEntityRole.REGION,
)
# Edges (k, k + 1) of a chart triangle, the edge_chord column order.
_CHART_EDGES = ((0, 1), (1, 2), (2, 0))


def _readonly(value: object, dtype: type[np.generic], /) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _root_id(
    source_id: str, source_revision: str, roots: np.ndarray, parameters: np.ndarray, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "native-cad-curve-roots",
            "source": (source_id, source_revision),
            "roots": array_tree_fingerprint((roots, parameters)),
        }
    )


def _curve_interval(
    first: float, second: float, /
) -> tuple[tuple[Fraction, Fraction], int]:
    """Exact source interval and the orientation of ``first -> second``."""
    start, end = Fraction.from_float(first), Fraction.from_float(second)
    if start == end:
        raise ValueError("A native CAD curve edge has a degenerate parameter interval.")
    return (min(start, end), max(start, end)), 1 if end > start else -1


class _Occurrences:
    """Domain rows resolved once into B-Rep index, entity id and occurrence."""

    __slots__ = ("_domain", "_resolved")

    def __init__(self, domain: MeshingDomain, /) -> None:
        self._domain = domain
        self._resolved: dict[tuple[int, int], tuple[int, str, tuple[str, ...]]] = {}

    def __call__(self, dimension: int, row: int, /) -> tuple[int, str, tuple[str, ...]]:
        key = (dimension, row)
        resolved = self._resolved.get(key)
        if resolved is None:
            domain = self._domain
            if dimension == 3:
                indices = domain.region_source_indices
                paths = domain.region_source_occurrences
            else:
                indices = domain.source_indices[dimension]
                paths = domain.source_occurrences[dimension]
            if not 0 <= row < len(indices):
                raise ValueError("A native CAD row names an absent domain entity.")
            path = tuple(paths[row])
            if (
                domain.resolve_entity_id(
                    dimension, domain.entity_id(dimension, row), occurrence_path=path
                )
                != row
            ):
                raise ValueError(
                    "A native CAD row does not resolve through one domain occurrence."
                )
            index = int(indices[row])
            resolved = (
                index,
                brep_entity_id(
                    domain.source_id,
                    domain.source_revision,
                    dimension,
                    index,
                    occurrence_path=path,
                ),
                path,
            )
            self._resolved[key] = resolved
        return resolved


def _domain_curve_rows(domain: MeshingDomain, /) -> np.ndarray:
    """Domain curve of every native model curve row."""
    from .providers._native_cad_bundle import cad_geometry_bundle

    edge_curve_rows = np.asarray(
        cad_geometry_bundle(domain).edge_curve_rows, dtype=np.int64
    )
    regular = np.flatnonzero(edge_curve_rows >= 0)
    rows = np.empty((regular.size,), dtype=np.int64)
    rows[edge_curve_rows[regular]] = regular
    return rows


def _sorted_rows(entities: np.ndarray, /) -> dict[tuple[int, ...], int]:
    return {
        tuple(int(value) for value in key): row
        for row, key in enumerate(np.sort(entities, axis=1))
    }


class NativeCadRecords(StrictModule, NonTrainableState):
    """Curve-edge authority and pole slivers of one native CAD solid mesh.

    ``root_curves``/``root_parameters`` are the constructed curve edges
    (domain curve row; curve parameter at their first and second canonical
    vertex). ``curve`` maps the current curve edges to exact source intervals:
    its ``parent_ids`` name roots, ``parent_association_id`` is ``root_id``,
    and its children partition every root interval exactly.
    ``sliver_vertices`` are the pole slivers' ``{P, Q}`` vertex global ids.
    """

    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    root_curves: np.ndarray
    root_parameters: np.ndarray
    root_id: str = eqx.field(static=True)
    curve: GeometryAssociation | None
    sliver_vertices: np.ndarray
    records_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        source_id: str,
        source_revision: str,
        root_curves: object,
        root_parameters: object,
        curve: GeometryAssociation | None,
        sliver_vertices: object,
        /,
    ) -> None:
        roots = _readonly(root_curves, np.int64)
        parameters = _readonly(root_parameters, np.float64)
        slivers = _readonly(sliver_vertices, np.int64)
        if (
            roots.ndim != 1
            or parameters.shape != (roots.size, 2)
            or not np.all(np.isfinite(parameters))
            or np.any(roots < 0)
            or slivers.ndim != 2
            or slivers.shape[1] != 2
            or np.any(slivers[:, 0] >= slivers[:, 1])
        ):
            raise ValueError("Native CAD records are misaligned.")
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.source_id = source_id
        self.source_revision = source_revision
        self.root_curves = roots
        self.root_parameters = parameters
        self.root_id = _root_id(source_id, source_revision, roots, parameters)
        self.curve = curve
        self.sliver_vertices = slivers
        self.records_id = canonical_fingerprint(
            {
                "kind": "native-cad-records",
                "mesh": self.mesh_id,
                "topology": self.topology_id,
                "roots": self.root_id,
                "curve": None if curve is None else curve.association_id,
                "slivers": array_tree_fingerprint(slivers),
            }
        )
        self.require_mesh(mesh)

    def require_mesh(self, mesh: CellMesh, /) -> None:
        """Require these records to bind ``mesh`` and partition every root."""
        if mesh.mesh_id != self.mesh_id or mesh.topology_id != self.topology_id:
            raise ValueError("Native CAD records bind another mesh.")
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        if not np.all(np.isin(self.sliver_vertices, vertex_ids)):
            raise ValueError("A native CAD pole sliver names an absent vertex.")
        curve = self.curve
        if curve is None:
            if self.root_curves.size:
                raise ValueError("Native CAD curve roots lost their curve association.")
            return
        intervals = curve.source_parameter_intervals
        if (
            curve.association_kind is not GeometryAssociationKind.CURVE
            or (curve.source_id, curve.source_revision)
            != (self.source_id, self.source_revision)
            or curve.provenance is not GeometryAssociationProvenance.LINEAGE
            or curve.parent_association_id != self.root_id
            or np.any(np.asarray(curve.parent_dimensions) != 1)
            or np.any(np.asarray(curve.source_dimensions) != 1)
        ):
            raise ValueError("Native CAD curve association lost its root authority.")
        curve.validate_target(mesh.entity_set(1))
        parents = np.asarray(curve.parent_ids, dtype=np.int64)
        if np.any(parents < 0) or np.any(parents >= self.root_curves.size):
            raise ValueError("A native CAD curve edge names an absent root.")
        children: dict[int, list[tuple[Fraction, Fraction]]] = {}
        identities: dict[int, set[tuple[str, tuple[str, ...]]]] = {}
        for row, parent in enumerate(parents.tolist()):
            interval = intervals[row]
            if interval is None:
                raise ValueError("A native CAD curve edge lacks its exact interval.")
            children.setdefault(parent, []).append(interval)
            identities.setdefault(parent, set()).add(
                (curve.source_entity_ids[row], curve.source_occurrence_paths[row])
            )
        for root, (first, second) in enumerate(self.root_parameters.tolist()):
            expected, _ = _curve_interval(first, second)
            parts = sorted(children.get(root, ()))
            if (
                not parts
                or len(identities[root]) != 1
                or parts[0][0] != expected[0]
                or parts[-1][1] != expected[1]
                or any(left[1] != right[0] for left, right in zip(parts, parts[1:]))
            ):
                raise ValueError(
                    "Native CAD curve edges do not exactly partition their root."
                )

    def require_curve(self, curve: GeometryAssociation | None, /) -> None:
        """Require ``curve`` to be these records' curve association."""
        if (None if curve is None else curve.association_id) != (
            None if self.curve is None else self.curve.association_id
        ):
            raise ValueError("CAD curve association differs from its native records.")

    def require_associations(
        self, associations: tuple[GeometryAssociation, ...], /
    ) -> None:
        """Require associations of this source to carry exactly ``curve``."""
        own = tuple(
            item
            for item in associations
            if (item.source_id, item.source_revision)
            == (self.source_id, self.source_revision)
        )
        if not own:
            return
        curves = tuple(
            item for item in own if item.association_kind is GeometryAssociationKind.CURVE
        )
        if len(curves) > 1:
            raise ValueError("A native CAD solid carries one curve association.")
        self.require_curve(curves[0] if curves else None)

    def retarget(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        stencil: VertexInterpolationStencil | None,
        /,
    ) -> NativeCadRecords:
        """These records on an adapted ``target`` through its exact lineage.

        The curve association follows the curve-edge lineage (a split curve
        edge's children take the exact halves of its source interval, the
        source occurrence inherited); the pole slivers keep their vertex ids,
        and a split pole segment is refused (see
        :func:`retarget_native_cad_surface`).
        """
        from ._surface_association_transfer import _curve_edges_with_root_ancestry

        self.require_mesh(source.mesh)
        curve = self.curve
        if target.topology_id == source.mesh.topology_id:
            return NativeCadRecords(
                target,
                self.source_id,
                self.source_revision,
                self.root_curves,
                self.root_parameters,
                curve,
                self.sliver_vertices,
            )
        if stencil is None:
            raise ValueError("Native CAD records require exact target vertex ancestry.")
        if curve is not None:
            curve = _curve_edges_with_root_ancestry(
                source, lineage, target, stencil, curve
            )
        weights = _stencil_weights(source.mesh, target, stencil)
        source_rows = {
            int(value): row
            for row, value in enumerate(
                np.asarray(source.mesh.vertex_global_ids).tolist()
            )
        }
        for first, second in self.sliver_vertices.tolist():
            segment = {source_rows[first], source_rows[second]}
            if any(set(row) == segment for row in weights):
                raise ValueError(
                    "Native CAD records refuse a split pole segment "
                    f"{(first, second)}: a non-pole point has no collapsed chart image."
                )
        return NativeCadRecords(
            target,
            self.source_id,
            self.source_revision,
            self.root_curves,
            self.root_parameters,
            curve,
            self.sliver_vertices,
        )


def native_cad_records(
    mesh: CellMesh,
    surface: CadSurfaceResult,
    domain: MeshingDomain,
    tolerance: float,
    /,
) -> NativeCadRecords:
    """The records of a ``cad_mesh_solid`` mesh (surface vertices first).

    Curve-edge residuals are the request's source tolerance, the bound the
    native certificate proves.
    """
    from .providers._native_plc_geometry import simplex_volume_entities

    edges, _ = simplex_volume_entities(mesh)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_rows = _sorted_rows(edges)
    curve_rows = _domain_curve_rows(domain)
    occurrences = _Occurrences(domain)
    curve_edges = np.asarray(surface.curve_edges, dtype=np.int64).reshape((-1, 2))
    model_curves = np.asarray(surface.curve_edge_curves, dtype=np.int64)
    parameters = np.asarray(surface.curve_edge_parameters, dtype=np.float64)
    count = curve_edges.shape[0]
    roots = np.empty((count,), dtype=np.int64)
    root_parameters = np.empty((count, 2), dtype=np.float64)
    targets = np.empty((count,), dtype=np.int64)
    indices = np.empty((count,), dtype=np.int32)
    orientations = np.empty((count,), dtype=np.int8)
    entity_ids: list[str] = []
    paths: list[tuple[str, ...]] = []
    intervals: list[tuple[Fraction, Fraction]] = []
    for row, ((a, b), model_curve, (first, second)) in enumerate(
        zip(curve_edges.tolist(), model_curves.tolist(), parameters.tolist(), strict=True)
    ):
        edge_row = edge_rows.get((min(a, b), max(a, b)))
        if edge_row is None:
            raise ValueError("A native CAD curve edge is not a mesh edge.")
        if int(edges[edge_row, 0]) != a:
            first, second = second, first
        curve = int(curve_rows[model_curve])
        interval, orientation = _curve_interval(first, second)
        index, entity_id, path = occurrences(1, curve)
        roots[row] = curve
        root_parameters[row] = (first, second)
        targets[row] = edge_ids[edge_row]
        indices[row] = index
        orientations[row] = orientation
        entity_ids.append(entity_id)
        paths.append(path)
        intervals.append(interval)
    curve_association = (
        GeometryAssociation(
            GeometryAssociationKind.CURVE,
            domain.source_id,
            domain.source_revision,
            mesh.entity_set(1).entity_set_id,
            targets,
            tuple(entity_ids),
            np.full((count,), tolerance, dtype=np.float64),
            source_dimensions=np.ones((count,), dtype=np.int8),
            source_indices=indices,
            source_roles=(GeometrySourceEntityRole.EDGE,) * count,
            source_occurrence_paths=tuple(paths),
            source_parameter_intervals=tuple(intervals),
            orientations=orientations,
            parent_dimensions=np.ones((count,), dtype=np.int8),
            parent_ids=np.arange(count, dtype=np.int64),
            parent_association_id=_root_id(
                domain.source_id, domain.source_revision, roots, root_parameters
            ),
            provenance=GeometryAssociationProvenance.LINEAGE,
            exact=False,
        )
        if count
        else None
    )
    chart = surface.chart
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    slivers = {
        tuple(sorted({int(vertex_ids[vertex]) for vertex in row}))
        for row, target in zip(
            np.asarray(chart.triangles, dtype=np.int64).tolist(),
            np.asarray(chart.chart_to_mesh, dtype=np.int64).tolist(),
            strict=True,
        )
        if target < 0
    }
    if any(len(pair) != 2 for pair in slivers):
        raise ValueError("A native CAD pole sliver does not pair two vertices.")
    return NativeCadRecords(
        mesh,
        domain.source_id,
        domain.source_revision,
        roots,
        root_parameters,
        curve_association,
        np.asarray(sorted(slivers), dtype=np.int64).reshape((-1, 2)),
    )


def native_cad_associations(
    mesh: CellMesh,
    surface: CadSurfaceResult,
    domain: MeshingDomain,
    cell_regions: np.ndarray,
    tolerance: float,
    records: NativeCadRecords,
    /,
) -> tuple[GeometryAssociation, ...]:
    """B-Rep associations of every mesh dimension, then the curve association.

    Surface vertices, edges and triangles take their native corner, curve or
    face classification; every other entity lies in the region of its cells.
    Boundary rows carry the source tolerance as residual (the bound the
    native certificate proves), region rows zero.
    """
    records.require_mesh(mesh)
    from .providers._native_plc_geometry import simplex_volume_entities

    edges, faces = simplex_volume_entities(mesh)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    regions = np.asarray(cell_regions, dtype=np.int64)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    patches = np.asarray(surface.triangle_patches, dtype=np.int64)
    facet_regions = np.asarray(surface.facet_regions, dtype=np.int64)
    if regions.shape != (cells.shape[0],) or np.any(regions < 0):
        raise ValueError("Native CAD cell regions are misaligned.")
    occurrences = _Occurrences(domain)
    curve_rows = _domain_curve_rows(domain)
    region_keys = domain.stratum_entity_ids(3)
    point_count = mesh.coordinates.shape[0]

    # Source stratum (dimension, domain row) of every entity; regions first.
    vertex_class = np.empty((point_count, 2), dtype=np.int64)
    vertex_class[:, 0] = 3
    vertex_class[cells.reshape(-1), 1] = np.repeat(regions, 4)
    vertex_parameters = np.full((point_count, 2), np.nan, dtype=np.float64)
    surface_count = np.asarray(surface.vertex_dimensions).shape[0]
    dimensions = np.asarray(surface.vertex_dimensions, dtype=np.int64)
    entities = np.asarray(surface.vertex_entities, dtype=np.int64)
    native_parameters = np.asarray(surface.vertex_parameters, dtype=np.float64)
    vertex_class[:surface_count, 0] = dimensions
    vertex_class[:surface_count, 1] = entities
    on_curve = np.flatnonzero(dimensions == 1)
    on_face = np.flatnonzero(dimensions == 2)
    vertex_class[on_curve, 1] = curve_rows[entities[on_curve]]
    vertex_parameters[on_curve, 0] = native_parameters[on_curve, 0]
    vertex_parameters[on_face] = native_parameters[on_face]

    edge_rows = _sorted_rows(edges)
    edge_class = np.empty((edges.shape[0], 2), dtype=np.int64)
    edge_class[:, 0] = 3
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pairs = np.sort(cells[:, [a, b]], axis=1)
        rows = [edge_rows[(int(x), int(y))] for x, y in pairs.tolist()]
        edge_class[rows, 1] = regions
    edge_orientations = np.zeros((edges.shape[0],), dtype=np.int8)
    for triangle, patch in zip(triangles.tolist(), patches.tolist(), strict=True):
        for a, b in _CHART_EDGES:
            row = edge_rows[
                (min(triangle[a], triangle[b]), max(triangle[a], triangle[b]))
            ]
            edge_class[row] = (2, patch)
    curve = records.curve
    if curve is not None:
        edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        edge_by_id = {int(value): row for row, value in enumerate(edge_ids.tolist())}
        for target, parent, orientation in zip(
            np.asarray(curve.target_global_ids, dtype=np.int64).tolist(),
            np.asarray(curve.parent_ids, dtype=np.int64).tolist(),
            np.asarray(curve.orientations, dtype=np.int8).tolist(),
            strict=True,
        ):
            row = edge_by_id[target]
            edge_class[row] = (1, int(records.root_curves[parent]))
            edge_orientations[row] = orientation

    face_rows = _sorted_rows(faces)
    face_class = np.empty((faces.shape[0], 2), dtype=np.int64)
    face_class[:, 0] = 3
    for a, b, c in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
        triples = np.sort(cells[:, [a, b, c]], axis=1)
        rows = [face_rows[tuple(int(value) for value in key)] for key in triples.tolist()]
        face_class[rows, 1] = regions
    face_orientations = np.zeros((faces.shape[0],), dtype=np.int8)
    face_sides: list[tuple[str | None, str | None]] = [(None, None)] * faces.shape[0]
    for triangle, patch, sides in zip(
        triangles.tolist(), patches.tolist(), facet_regions.tolist(), strict=True
    ):
        row = face_rows[tuple(sorted(triangle))]
        face_class[row] = (2, patch)
        # The canonical face order is an even permutation of the outward
        # triangle exactly when it is a rotation of it.
        canonical = [int(value) for value in faces[row]]
        rotations = {tuple(triangle[k:] + triangle[:k]) for k in range(3)}
        face_orientations[row] = 1 if tuple(canonical) in rotations else -1
        face_sides[row] = (
            None if sides[0] < 0 else region_keys[sides[0]],
            None if sides[1] < 0 else region_keys[sides[1]],
        )
    cell_class = np.stack((np.full_like(regions, 3), regions), axis=1)

    def association(
        dimension: int,
        classes: np.ndarray,
        parameters: np.ndarray | None,
        orientations: np.ndarray | None,
        sides: list[tuple[str | None, str | None]] | None,
        /,
    ) -> GeometryAssociation:
        resolved = [occurrences(int(stratum), int(row)) for stratum, row in classes]
        count = classes.shape[0]
        return GeometryAssociation(
            GeometryAssociationKind.BREP,
            domain.source_id,
            domain.source_revision,
            mesh.entity_set(dimension).entity_set_id,
            np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64),
            tuple(value[1] for value in resolved),
            np.where(classes[:, 0] < 3, tolerance, 0.0),
            resolved=np.ones((count,), dtype=np.bool_),
            source_dimensions=classes[:, 0].astype(np.int8),
            source_indices=np.asarray([value[0] for value in resolved], dtype=np.int32),
            source_roles=tuple(_ROLES[int(stratum)] for stratum in classes[:, 0]),
            source_occurrence_paths=tuple(value[2] for value in resolved),
            source_region_sides=(
                tuple((None, None) for _ in range(count))
                if sides is None
                else tuple(sides)
            ),
            parameters=parameters,
            orientations=(
                np.zeros((count,), dtype=np.int8)
                if orientations is None
                else orientations
            ),
            exact=False,
        )

    associations = (
        association(0, vertex_class, vertex_parameters, None, None),
        association(1, edge_class, None, edge_orientations, None),
        association(2, face_class, None, face_orientations, face_sides),
        association(3, cell_class, None, None, None),
    )
    return associations if curve is None else (*associations, curve)


def _stencil_weights(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil | None,
    /,
) -> list[dict[int, Fraction]]:
    """Exact source-vertex-row weights of every target vertex row.

    A target vertex without a stencil row keeps its source vertex (same
    global id, same coordinates).
    """
    source_rows = {
        int(value): row
        for row, value in enumerate(np.asarray(source.vertex_global_ids).tolist())
    }
    rows: dict[int, dict[int, Fraction]] = {}
    if stencil is not None:
        for target_id, sources, valid, weights in zip(
            np.asarray(stencil.target_global_ids, dtype=np.int64).tolist(),
            np.asarray(stencil.source_global_ids, dtype=np.int64).tolist(),
            np.asarray(stencil.valid, dtype=np.bool_).tolist(),
            stencil.exact_weights,
            strict=True,
        ):
            row: dict[int, Fraction] = {}
            for source_id, is_valid, weight in zip(sources, valid, weights, strict=True):
                if not is_valid or weight == 0:
                    continue
                if not isinstance(weight, Fraction) or source_id not in source_rows:
                    raise ValueError(
                        "Native CAD chart transfer refuses a target vertex whose "
                        "ancestry is not a rational combination of source vertices."
                    )
                row[source_rows[source_id]] = weight
            rows[int(target_id)] = row
    source_points = np.asarray(source.coordinates, dtype=np.float64)
    target_points = np.asarray(target.coordinates, dtype=np.float64)
    output: list[dict[int, Fraction]] = []
    for target_row, identifier in enumerate(
        np.asarray(target.vertex_global_ids, dtype=np.int64).tolist()
    ):
        stencil_row = rows.get(identifier)
        if stencil_row is None:
            source_row = source_rows.get(identifier)
            if source_row is None or not np.array_equal(
                target_points[target_row], source_points[source_row]
            ):
                raise ValueError(
                    "Native CAD chart transfer refuses a target vertex without "
                    "exact source ancestry."
                )
            stencil_row = {source_row: Fraction(1)}
        output.append(stencil_row)
    return output


def _split_point(
    corners: np.ndarray,
    weights: tuple[Fraction, Fraction, Fraction],
    order: list[int],
    /,
) -> np.ndarray:
    """Chart point of exact corner weights, summed in a canonical corner order.

    A point on a chart edge depends only on that edge's two corners (taken in
    source vertex order), so rows sharing the edge reproduce it bitwise.
    """
    active = [k for k in order if weights[k] != 0]
    if len(active) == 1:
        return corners[active[0]].copy()
    if len(active) == 2:
        first, second = active
        return corners[first] + float(weights[second]) * (
            corners[second] - corners[first]
        )
    point = float(weights[order[0]]) * corners[order[0]]
    for k in order[1:]:
        point = point + float(weights[k]) * corners[k]
    return point


def retarget_native_cad_surface(
    witness: NativeCadSurfaceWitness,
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil | None,
    /,
) -> NativeCadSurfaceWitness:
    """The chart realization of ``witness`` on an adapted ``target``.

    The native chart realizes each boundary triangle as the affine image of
    its chart triangle (V3 bounds exactly that correspondence). Adaptation
    inserts a boundary vertex as an exact rational combination of its source
    triangle's corners (bisection: the physical midpoint of a boundary edge,
    weights 1/2, 1/2), so each target boundary triangle is the affine image
    of the chart sub-triangle with the same corner weights: on a face edge
    the chart midpoint of the row's edge, on a curve edge additionally the
    chord's parameter midpoint (chords split linearly with their ranges).
    No point is projected: every target boundary vertex must lie at its
    stencil combination, and the native verifiers re-prove the target
    realization from scratch. Refused, with the reason: ancestry that is not
    rational or not within one source chart row, a vertex off its stencil
    combination, and a split pole segment (a pole sliver's P-Q edge; its
    midpoint would need a collapsed chart edge off the pole, which the chart
    model does not admit).
    """
    if witness.mesh_id != source.mesh_id:
        raise ValueError("Native CAD surface witness binds another source mesh.")
    weights = _stencil_weights(source, target, stencil)
    source_points = np.asarray(source.coordinates, dtype=np.float64)
    target_points = np.asarray(target.coordinates, dtype=np.float64)
    chart_triangles = np.asarray(witness.chart_triangles, dtype=np.int64)
    chart_uv = np.asarray(witness.chart_uv, dtype=np.float64)
    chart_patches = np.asarray(witness.chart_patches, dtype=np.int64)
    chart_to_mesh = np.asarray(witness.chart_to_mesh, dtype=np.int64)
    edge_chord = np.asarray(witness.edge_chord, dtype=np.int64)
    chord_uv = np.asarray(witness.chord_uv, dtype=np.float64)
    chord_range = np.asarray(witness.chord_range, dtype=np.float64)
    surface_triangles = np.asarray(witness.surface_triangles, dtype=np.int64)
    for corners, mesh_row in zip(
        chart_triangles.tolist(), chart_to_mesh.tolist(), strict=True
    ):
        if mesh_row < 0:
            continue
        triangle = surface_triangles[mesh_row].tolist()
        if corners not in (
            triangle,
            triangle[1:] + triangle[:1],
            triangle[2:] + triangle[:2],
        ):
            raise ValueError(
                "A native CAD chart row does not follow its boundary triangle."
            )
    rows_by_corners = {
        frozenset(corners): row
        for row, (corners, mesh_row) in enumerate(
            zip(chart_triangles.tolist(), chart_to_mesh.tolist(), strict=True)
        )
        if mesh_row >= 0
    }

    target_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in target.blocks]
    )
    faces, counts = np.unique(
        np.sort(
            np.concatenate(
                [
                    target_cells[:, list(face)]
                    for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
                ]
            ),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    # Target chart pieces: (source row, target vertex rows, corner weights).
    pieces: list[tuple[int, list[int], list[tuple[Fraction, Fraction, Fraction]]]] = []

    def corner_weights(row: int, vertex: int, /) -> tuple[Fraction, Fraction, Fraction]:
        corners = chart_triangles[row].tolist()
        values = [Fraction(0)] * 3
        for source_row, weight in weights[vertex].items():
            values[corners.index(source_row)] = weight
        return values[0], values[1], values[2]

    for key in faces[counts == 1].tolist():
        support = frozenset(row for vertex in key for row in weights[vertex])
        row = rows_by_corners.get(support)
        if row is None:
            raise ValueError(
                "Native CAD chart transfer refuses a target boundary triangle whose "
                f"vertices descend from source vertices {sorted(support)}, which "
                "bound no single source chart triangle."
            )
        vertices = [int(value) for value in key]
        # Map exactly: the target point must be its stencil combination of
        # the source corners (up to the rounding of that combination).
        for vertex in vertices:
            expected = sum(
                float(weight) * source_points[source_row]
                for source_row, weight in weights[vertex].items()
            )
            scale = max(1.0, float(np.max(np.abs(target_points[vertex]))))
            if (
                np.max(np.abs(target_points[vertex] - expected))
                > 8 * np.finfo(np.float64).eps * scale
            ):
                raise ValueError(
                    "Native CAD chart transfer refuses target boundary vertex row "
                    f"{vertex}: it does not lie at its stencil combination of the "
                    "source chart corners."
                )
        barycentric = [corner_weights(row, vertex) for vertex in vertices]
        (a0, a1, _), (b0, b1, _), (c0, c1, _) = barycentric
        area = (b0 - a0) * (c1 - a1) - (b1 - a1) * (c0 - a0)
        if area == 0:
            raise ValueError("A target boundary triangle is degenerate in its chart.")
        if area < 0:
            vertices[1], vertices[2] = vertices[2], vertices[1]
            barycentric[1], barycentric[2] = barycentric[2], barycentric[1]
        pieces.append((row, vertices, barycentric))
    surface_count = len(pieces)
    for row in np.flatnonzero(chart_to_mesh < 0).tolist():
        corners = chart_triangles[row].tolist()
        single = next(k for k in range(3) if corners.count(corners[k]) == 1)
        p, q = corners[(single + 1) % 3], corners[single]
        # A sliver pairs two chart images of the pole P with Q; its segment
        # PQ is one mesh edge. Splitting it at M would give M two chart
        # points joined by a collapsed edge off the pole, which the chart
        # model (V3's chart-mesh link) does not admit.
        on_segment = [
            vertex
            for vertex, vertex_weights in enumerate(weights)
            if set(vertex_weights) <= {p, q} and len(vertex_weights) == 2
        ]
        if on_segment:
            raise ValueError(
                "Native CAD chart transfer refuses a split pole segment: target "
                f"vertex rows {on_segment} lie inside the sliver segment of source "
                f"vertices {sorted((p, q))}, and a non-pole point has no collapsed "
                "chart image."
            )
        vertices = []
        for corner in corners:
            (target_row,) = (
                vertex
                for vertex, vertex_weights in enumerate(weights)
                if vertex_weights == {corner: Fraction(1)}
            )
            vertices.append(target_row)
        identity = (Fraction(1), Fraction(0), Fraction(0))
        pieces.append(
            (
                row,
                vertices,
                [identity, identity[2:] + identity[:2], identity[1:] + identity[:1]],
            )
        )

    triangles = np.empty((len(pieces), 3), dtype=np.int64)
    uv = np.empty((len(pieces), 3, 2), dtype=np.float64)
    patches = np.empty((len(pieces),), dtype=np.int32)
    to_mesh = np.full((len(pieces),), -1, dtype=np.int64)
    to_mesh[:surface_count] = np.arange(surface_count, dtype=np.int64)
    for index, (row, vertices, barycentric) in enumerate(pieces):
        order = sorted(range(3), key=lambda k: int(chart_triangles[row, k]))
        triangles[index] = vertices
        patches[index] = chart_patches[row]
        for k in range(3):
            uv[index, k] = _split_point(chart_uv[row], barycentric[k], order)

    # Split points of every chord (fractions along its direction) with the
    # chart point of each, and the chord of every piece edge on it.
    chord_points: dict[int, dict[Fraction, np.ndarray]] = {}
    piece_chords: list[tuple[int, int, int, Fraction]] = []
    for index, (row, _, barycentric) in enumerate(pieces):
        corners = chart_uv[row]
        for piece_edge, (a, b) in enumerate(_CHART_EDGES):
            for edge, (u, v) in enumerate(_CHART_EDGES):
                other = 3 - u - v
                chord = int(edge_chord[row, edge])
                if chord < 0 or barycentric[a][other] != 0 or barycentric[b][other] != 0:
                    continue
                forward = np.linalg.norm(chord_uv[chord, 0] - corners[u]) <= (
                    np.linalg.norm(chord_uv[chord, 0] - corners[v])
                )
                s_a, s_b = barycentric[a][v], barycentric[b][v]
                if not forward:
                    s_a, s_b = 1 - s_a, 1 - s_b
                points = chord_points.setdefault(chord, {})
                points.setdefault(s_a, uv[index, a])
                points.setdefault(s_b, uv[index, b])
                piece_chords.append((index, piece_edge, chord, min(s_a, s_b)))
    chord_use = np.asarray(witness.chord_use, dtype=np.int64)
    chord_loop = np.asarray(witness.chord_loop, dtype=np.int64)
    new_use: list[int] = []
    new_loop: list[int] = []
    new_uv: list[np.ndarray] = []
    new_range: list[np.ndarray] = []
    chord_rows: dict[tuple[int, Fraction], int] = {}
    for chord in range(chord_use.size):
        low, high = chord_range[chord]
        points = {
            **chord_points.get(chord, {}),
            Fraction(0): chord_uv[chord, 0],
            Fraction(1): chord_uv[chord, 1],
        }
        ordered = sorted(points)
        for left, right in zip(ordered, ordered[1:], strict=False):
            chord_rows[(chord, left)] = len(new_use)
            new_use.append(int(chord_use[chord]))
            new_loop.append(int(chord_loop[chord]))
            new_uv.append(np.stack((points[left], points[right])))
            new_range.append(
                np.asarray(
                    [
                        low
                        if s == 0
                        else high
                        if s == 1
                        else low + float(s) * (high - low)
                        for s in (left, right)
                    ],
                    dtype=np.float64,
                )
            )
    chords = np.full((len(pieces), 3), -1, dtype=np.int64)
    for index, piece_edge, chord, left in piece_chords:
        chords[index, piece_edge] = chord_rows[(chord, left)]
    chart = CadChartResult(
        triangles,
        patches,
        uv,
        to_mesh,
        chords,
        np.asarray(witness.use_patch),
        np.asarray(witness.use_range),
        np.asarray(new_use, dtype=np.int64),
        np.asarray(new_loop, dtype=np.int64),
        np.asarray(new_uv, dtype=np.float64).reshape((-1, 2, 2)),
        np.asarray(new_range, dtype=np.float64).reshape((-1, 2)),
    )
    # The witness reads the boundary triangles and the chart; the other
    # construction outputs of a surface result do not exist after adaptation.
    empty = np.empty((0,))
    surface_result = CadSurfaceResult(
        status=MeshcoreStatus.OK,
        points=np.asarray(target.coordinates, dtype=np.float64),
        vertex_dimensions=empty,
        vertex_entities=empty,
        vertex_parameters=empty,
        triangles=triangles[:surface_count],
        triangle_patches=empty,
        triangle_uv=empty,
        triangle_edge_uses=empty,
        facet_regions=empty,
        sliver_uv=empty,
        sliver_patches=empty,
        curve_edges=empty,
        curve_edge_curves=empty,
        curve_edge_parameters=empty,
        face_stats=empty,
        curve_offsets=empty,
        curve_parameters=empty,
        curve_sag_splits=empty,
        curve_failures=empty,
        curve_statistics=empty,
        face_failures=empty,
        pole_sectors=empty,
        chart=chart,
    )
    return NativeCadSurfaceWitness(target, surface_result, witness.loop_count)


__all__ = [
    "native_cad_associations",
    "native_cad_records",
    "NativeCadRecords",
    "retarget_native_cad_surface",
]
