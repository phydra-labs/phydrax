"""Source occurrence and association records of a native CAD solid.

The native chart, the actual B-Rep occurrences and the tetrahedral mesh
determine the B-Rep tables. Curve roots are issued by the original native
chart. An adapted mesh retains that one origin, not a chain of association
tables, so an archive can replay exact curve ancestry from independent
operands instead of certifying the tables it is asked to restore.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import CadChartResult, CadSurfaceResult, MeshcoreStatus
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._meshing_domain import MeshingDomain, PatchCurveUse
from ..geometry.brep._projection_contracts import brep_entity_id
from ._association import (
    _fraction_rows_fingerprint,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._native_cad_certificate import NativeCadSurfaceWitness


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ._lineage import MeshLineage, VertexInterpolationStencil
    from ._result import CellMeshingResult


_ROLES = (
    GeometrySourceEntityRole.VERTEX,
    GeometrySourceEntityRole.EDGE,
    GeometrySourceEntityRole.FACET,
    GeometrySourceEntityRole.REGION,
)

type _CurveSplit = tuple[int, int, int, Fraction]
# Edges (k, k + 1) of a chart triangle, the edge_chord column order.
_CHART_EDGES = ((0, 1), (1, 2), (2, 0))


def _readonly(value: object, dtype: type[np.generic], /) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _root_id(
    source_id: str,
    source_revision: str,
    roots: np.ndarray,
    parameters: np.ndarray,
    vertices: np.ndarray,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "native-cad-curve-roots",
            "source": (source_id, source_revision),
            "roots": array_tree_fingerprint((roots, parameters, vertices)),
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


def _sorted_rows(entities: np.ndarray, /) -> dict[tuple[int, ...], int]:
    return {
        tuple(int(value) for value in key): row
        for row, key in enumerate(np.sort(entities, axis=1))
    }


class NativeCadRecords(StrictModule, NonTrainableState):
    """Complete native source association authority for one CAD solid mesh.

    The roots name original curve segments by source curve, oriented endpoint
    parameters and endpoint global IDs. Adapted records retain the checked
    origin mesh/chart needed to replay this historical claim on archive read.
    ``association_keys`` bind the full B-Rep and curve tables, including
    residuals (not part of ``GeometryAssociation.association_id``).
    ``sliver_vertices`` are the pole slivers' ``{P, Q}`` vertex global ids.
    """

    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    origin_mesh: CellMesh | None
    origin_chart: NativeCadSurfaceWitness | None
    curve_splits: tuple[_CurveSplit, ...] = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    root_curves: np.ndarray
    root_parameters: np.ndarray
    root_vertices: np.ndarray
    root_id: str = eqx.field(static=True)
    curve: GeometryAssociation | None
    sliver_vertices: np.ndarray
    association_keys: tuple[str, ...] = eqx.field(static=True)
    records_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        domain: MeshingDomain,
        witness: NativeCadSurfaceWitness,
        tolerance: float,
        regions: np.ndarray,
        origin_mesh: CellMesh | None,
        origin_chart: NativeCadSurfaceWitness | None,
        curve_splits: tuple[_CurveSplit, ...],
        associations_out: list[tuple[GeometryAssociation, ...]],
        /,
    ) -> None:
        roots, curve, slivers, tables = _derive_native_cad_tables(
            mesh,
            domain,
            regions,
            witness,
            tolerance,
            origin_mesh,
            origin_chart,
            curve_splits,
        )
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.witness_id = witness.witness_id
        self.tolerance = tolerance
        self.origin_mesh = origin_mesh
        self.origin_chart = origin_chart
        self.curve_splits = curve_splits
        self.root_curves, self.root_parameters, self.root_vertices = roots
        self.root_id = _root_id(domain.source_id, domain.source_revision, *roots)
        self.curve = curve
        self.sliver_vertices = slivers
        self.association_keys = tuple(_association_key(item) for item in tables)
        self.records_id = canonical_fingerprint(
            {
                "kind": "native-cad-records",
                "mesh": self.mesh_id,
                "tolerance": self.tolerance,
                "topology": self.topology_id,
                "witness": self.witness_id,
                "origin": None if origin_chart is None else origin_chart.witness_id,
                "curve_splits": tuple(
                    (vertex, first, second, weight.numerator, weight.denominator)
                    for vertex, first, second, weight in curve_splits
                ),
                "roots": self.root_id,
                "associations": self.association_keys,
                "slivers": array_tree_fingerprint(slivers),
            }
        )
        associations_out.append(tables)

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
        if (None if curve is None else _association_key(curve)) != (
            None if self.curve is None else self.association_keys[-1]
        ):
            raise ValueError("CAD curve association differs from its native records.")

    def require_associations(
        self, mesh: CellMesh, associations: tuple[GeometryAssociation, ...], /
    ) -> None:
        """Require every owned table, and only those tables, before consuming rows."""
        if (mesh.mesh_id, mesh.topology_id) != (self.mesh_id, self.topology_id):
            raise ValueError("Native CAD associations bind another mesh.")
        targets = tuple(mesh.entity_set(dimension) for dimension in range(4))
        slots = {
            (GeometryAssociationKind.BREP, entity_set.entity_set_id): dimension
            for dimension, entity_set in enumerate(targets)
        }
        if len(self.association_keys) == 5:
            slots[(GeometryAssociationKind.CURVE, targets[1].entity_set_id)] = 4
        seen: set[int] = set()
        for table in associations:
            if (table.source_id, table.source_revision) != (
                self.source_id,
                self.source_revision,
            ):
                continue
            slot = slots.get((table.association_kind, table.target_entity_set_id))
            if slot is None or slot in seen:
                raise ValueError(
                    "Native CAD associations have an unexpected or duplicate table."
                )
            if slot < 4 and not np.array_equal(
                np.asarray(table.target_global_ids), np.asarray(targets[slot].entity_ids)
            ):
                raise ValueError(
                    "Native CAD association does not cover its exact target rows."
                )
            if _association_key(table) != self.association_keys[slot]:
                raise ValueError(
                    "Native CAD association differs from its native records."
                )
            seen.add(slot)
        if len(seen) != len(slots):
            raise ValueError("Native CAD associations are missing an owned source table.")

    def require_premises(
        self,
        mesh: CellMesh,
        domain: MeshingDomain,
        regions: np.ndarray,
        witness: NativeCadSurfaceWitness,
        tolerance: float,
        /,
    ) -> None:
        _require_solid_region(mesh, domain, regions)
        witness.require_binding(mesh)
        self.require_mesh(mesh)
        if (self.source_id, self.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ) or (self.witness_id, self.tolerance) != (witness.witness_id, tolerance):
            raise ValueError(
                "Native CAD association records bind another source chart or tolerance."
            )

    @classmethod
    def from_stored_fields(
        cls,
        stored: NativeCadRecords,
        mesh: CellMesh,
        domain: MeshingDomain,
        regions: np.ndarray,
        witness: NativeCadSurfaceWitness,
        tolerance: float,
        /,
    ) -> NativeCadRecords:
        """Replay native ownership; archived table contents and IDs are not premises."""
        from ..discretization._cell_geometry import CellGeometrySpec
        from ..discretization._cell_mesh import CellMesh
        from ._native_cad_certificate import _native_findings

        witness.require_binding(mesh)
        origin = stored.origin_mesh
        origin_chart = stored.origin_chart
        if origin is not None and origin_chart is not None:
            # The original tetrahedra are used by the same independent
            # embedding/tiling proof as a root result. No parent result or
            # previously authored association is needed to replay the root.
            origin = CellMesh.from_stored_fields(origin)
            origin_chart.require_binding(origin)
            findings, _, _ = _native_findings(
                origin,
                CellGeometrySpec.affine(origin),
                domain,
                origin_chart,
                tolerance,
                False,
                False,
            )
            if findings:
                raise ValueError(
                    "Native CAD association archive has no verified original chart."
                )
        restored, _ = _issue_records(
            mesh,
            domain,
            regions,
            witness,
            tolerance,
            origin,
            origin_chart,
            stored.curve_splits,
        )
        if (
            (stored.records_id, stored.root_id, stored.association_keys)
            != (restored.records_id, restored.root_id, restored.association_keys)
            or (stored.mesh_id, stored.topology_id, stored.witness_id, stored.tolerance)
            != (
                restored.mesh_id,
                restored.topology_id,
                restored.witness_id,
                restored.tolerance,
            )
            or (stored.source_id, stored.source_revision)
            != (domain.source_id, domain.source_revision)
            or not np.array_equal(stored.root_curves, restored.root_curves)
            or not np.array_equal(stored.root_parameters, restored.root_parameters)
            or not np.array_equal(stored.root_vertices, restored.root_vertices)
            or not np.array_equal(stored.sliver_vertices, restored.sliver_vertices)
        ):
            raise ValueError(
                "Native CAD association archive differs from replayed source authority."
            )
        restored.require_curve(stored.curve)
        return restored

    def retarget(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        regions: np.ndarray,
        stencil: VertexInterpolationStencil | None,
        /,
    ) -> tuple[
        NativeCadRecords, NativeCadSurfaceWitness, tuple[GeometryAssociation, ...]
    ]:
        """Rebuild one target association bank from checked chart and edge ancestry."""
        from ._association_routes import require_transfer_lineage
        from ._surface_association_transfer import _curve_edges_with_root_ancestry

        self.require_associations(source.mesh, source.associations)
        report = source.certification
        if report is None or report.inputs.native_cad_records is not self:
            raise ValueError(
                "Native CAD association transfer needs its certified source records."
            )
        inputs = report.inputs
        domain, chart = inputs.domain, inputs.native_cad_surface
        if (
            not isinstance(domain, MeshingDomain)
            or chart is None
            or inputs.cell_regions is None
        ):
            raise ValueError(
                "Native CAD association transfer needs its source domain and chart."
            )
        self.require_premises(
            source.mesh, domain, inputs.cell_regions, chart, self.tolerance
        )
        report.require_passed()
        if report.native_cad is None or report.native_cad.records_id != self.records_id:
            raise ValueError(
                "Native CAD association transfer has no matching accepted records."
            )
        require_transfer_lineage(source.mesh, lineage, target, dimensions=(0, 1))
        target_chart = (
            chart
            if target.mesh_id == source.mesh.mesh_id
            else retarget_native_cad_surface(chart, source.mesh, target, stencil)
        )
        origin_mesh = self.origin_mesh
        origin_chart = self.origin_chart
        if target.mesh_id != source.mesh.mesh_id and origin_mesh is None:
            origin_mesh, origin_chart = source.mesh, chart
        curve_splits = (
            self.curve_splits
            if target.topology_id == source.mesh.topology_id
            else _capture_curve_splits(
                source.mesh,
                _chart_curve_segments(target, domain, target_chart),
                stencil,
                self.curve_splits,
            )
        )
        records, tables = _issue_records(
            target,
            domain,
            regions,
            target_chart,
            self.tolerance,
            origin_mesh,
            origin_chart,
            curve_splits,
        )
        if self.root_id != records.root_id:
            raise ValueError(
                "Native CAD association transfer changed its original curve roots."
            )
        if target.topology_id != source.mesh.topology_id and self.curve is not None:
            if stencil is None or records.curve is None:
                raise ValueError(
                    "Native CAD curve association requires exact vertex ancestry."
                )
            inherited = _curve_edges_with_root_ancestry(
                source, lineage, target, stencil, self.curve
            )
            if _curve_lineage_rows(records.curve) != _curve_lineage_rows(inherited):
                raise ValueError(
                    "Native CAD curve association disagrees with exact transfer lineage."
                )
        return records, target_chart, tables


def _association_key(table: GeometryAssociation, /) -> str:
    """Commit to actual rows even when a retained association ID is stale."""
    count = len(table.target_global_ids)
    if table.source_boundary_witness_id is not None or any(
        len(bank) != count or any(row is not None for row in bank)
        for bank in (
            table.target_reference_pieces,
            table.target_parameter_intervals,
            table.source_parameter_points,
        )
    ):
        raise ValueError("Native CAD associations do not carry partial target supports.")
    return canonical_fingerprint(
        {
            "kind": "native-cad-association",
            "association": table.association_id,
            "table": (table.association_kind.value, table.target_entity_set_id),
            "source": (table.source_id, table.source_revision),
            "entities": table.source_entity_ids,
            "roles": tuple(
                None if role is None else role.value for role in table.source_roles
            ),
            "paths": table.source_occurrence_paths,
            "sides": table.source_region_sides,
            "intervals": _fraction_rows_fingerprint(table.source_parameter_intervals),
            "lineage": (table.exact, table.provenance.value, table.parent_association_id),
            "arrays": array_tree_fingerprint(table),
        }
    )


def _curve_lineage_rows(
    curve: GeometryAssociation, /
) -> dict[int, tuple[int, int, tuple[Fraction, Fraction] | None]]:
    return dict(
        zip(
            np.asarray(curve.target_global_ids, dtype=np.int64).tolist(),
            zip(
                np.asarray(curve.parent_ids, dtype=np.int64).tolist(),
                np.asarray(curve.orientations, dtype=np.int8).tolist(),
                curve.source_parameter_intervals,
                strict=True,
            ),
            strict=True,
        )
    )


def _require_solid_region(
    mesh: CellMesh, domain: MeshingDomain, regions: np.ndarray, /
) -> None:
    if (
        len(domain.regions) != 1
        or regions.shape != (mesh.entity_set(3).count,)
        or np.any(regions != 0)
    ):
        raise ValueError(
            "Native CAD associations require the sole solid to own every cell."
        )


class _CurveSegment(NamedTuple):
    curve: int
    low: float
    high: float
    start: int
    stop: int


def _chart_curve_segments(
    mesh: CellMesh, domain: MeshingDomain, chart: NativeCadSurfaceWitness, /
) -> tuple[_CurveSegment, ...]:
    """In the curve owner's parameter gauge, directed from low to high."""
    owners = {}
    use_row = 0
    for patch, face in enumerate(domain.patches):
        for loop, uses in enumerate(face.loops):
            for offset, use in enumerate(uses):
                if isinstance(use, PatchCurveUse) and tuple(
                    domain.curve_owners[use.curve]
                ) == (patch, loop, offset):
                    owners[use_row] = use.curve
                use_row += 1
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    edge_chord = np.asarray(chart.edge_chord)
    chord_use = np.asarray(chart.chord_use)
    by_chord: dict[int, _CurveSegment] = {}
    for row, local in zip(*np.nonzero(edge_chord >= 0), strict=True):
        chord = int(edge_chord[row, local])
        if not 0 <= chord < chord_use.size:
            raise ValueError("Native CAD curve association names an absent chart chord.")
        curve = owners.get(int(chord_use[chord]))
        if curve is None:
            continue
        a, b = int(local), (int(local) + 1) % 3
        uv = chart.chart_uv[row, [a, b]]
        chord_uv = chart.chord_uv[chord]
        if np.array_equal(uv, chord_uv[::-1]):
            a, b = b, a
        elif not np.array_equal(uv, chord_uv):
            raise ValueError(
                "Native CAD curve association chord endpoints differ from the chart."
            )
        start, stop = (
            int(vertex_ids[vertex]) for vertex in chart.chart_triangles[row, [a, b]]
        )
        first, last = (float(value) for value in chart.chord_range[chord])
        if (
            not np.isfinite(first)
            or not np.isfinite(last)
            or first == last
            or start == stop
        ):
            raise ValueError(
                "Native CAD curve association has a degenerate chart segment."
            )
        segment = (
            _CurveSegment(curve, first, last, start, stop)
            if first < last
            else _CurveSegment(curve, last, first, stop, start)
        )
        if by_chord.setdefault(chord, segment) != segment:
            raise ValueError(
                "Native CAD curve association chord has inconsistent mesh endpoints."
            )
    if sum(int(use) in owners for use in chord_use) != len(by_chord):
        raise ValueError("Native CAD curve association is missing chart chord support.")
    return tuple(sorted(by_chord.values()))


def _root_banks(
    mesh: CellMesh, segments: tuple[_CurveSegment, ...], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Issue canonical root rows from realized original curve chord edges."""
    from .providers._native_plc_geometry import simplex_volume_entities

    edges, _ = simplex_volume_entities(mesh)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ordered = {frozenset(pair): tuple(pair) for pair in vertex_ids[edges].tolist()}
    parameters = []
    vertices = []
    for segment in segments:
        pair = ordered.get(frozenset((segment.start, segment.stop)))
        if pair is None:
            raise ValueError("Native CAD curve association root is not a mesh edge.")
        vertices.append(pair)
        parameters.append(
            (segment.low, segment.high)
            if pair[0] == segment.start
            else (segment.high, segment.low)
        )
    return (
        _readonly([item.curve for item in segments], np.int64),
        _readonly(np.asarray(parameters, dtype=np.float64).reshape((-1, 2)), np.float64),
        _readonly(np.asarray(vertices, dtype=np.int64).reshape((-1, 2)), np.int64),
    )


def _capture_curve_splits(
    source: CellMesh,
    segments: tuple[_CurveSegment, ...],
    stencil: VertexInterpolationStencil | None,
    retained: tuple[_CurveSplit, ...],
    /,
) -> tuple[_CurveSplit, ...]:
    """Keep the represented stencil actions needed by live curve vertices."""
    live = {vertex for segment in segments for vertex in (segment.start, segment.stop)}
    created = live.difference(np.asarray(source.vertex_global_ids).tolist())
    additions: list[_CurveSplit] = []
    if created and stencil is not None:
        sources = np.asarray(stencil.source_global_ids)
        valid = np.asarray(stencil.valid)
        for row, vertex in enumerate(np.asarray(stencil.target_global_ids).tolist()):
            if vertex not in created:
                continue
            operands = sorted(
                (parent, weight)
                for parent, weight, active in zip(
                    sources[row].tolist(),
                    stencil.exact_weights[row],
                    valid[row].tolist(),
                    strict=True,
                )
                if active and weight != 0
            )
            if len(operands) != 2:
                raise ValueError(
                    "Native CAD curve association requires two source endpoints."
                )
            (first, left), (second, right) = operands
            if (
                not isinstance(left, Fraction)
                or not isinstance(right, Fraction)
                or left + right != 1
            ):
                raise ValueError(
                    "Native CAD curve association requires affine dyadic weights."
                )
            additions.append((vertex, first, second, right))
    if len(additions) != len(created):
        raise ValueError("Native CAD curve association lost a created vertex's stencil.")
    kept: list[_CurveSplit] = []
    for split in reversed((*retained, *additions)):
        vertex, first, second, _ = split
        if vertex in live:
            live.update((first, second))
            kept.append(split)
    return tuple(reversed(kept))


def _is_curve_piece(
    low: Fraction, high: Fraction, first: Fraction, last: Fraction, /
) -> bool:
    """Whether [low, high] is an aligned power-of-two subdivision of the root."""
    if not first <= low < high <= last:
        return False
    width = high - low
    ratio = (last - first) / width
    return (
        ratio.denominator == 1
        and ratio.numerator & (ratio.numerator - 1) == 0
        and ((low - first) / width).denominator == 1
    )


def _replay_curve_splits(
    original: tuple[_CurveSegment, ...],
    origin_points: dict[int, np.ndarray],
    splits: tuple[_CurveSplit, ...],
    /,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, Fraction]]]:
    """Recreate represented coordinates and exact curve addresses from one origin."""
    result: dict[int, np.ndarray] = {}
    addresses: dict[int, dict[int, Fraction]] = {}
    root_intervals = tuple(_curve_interval(root.low, root.high)[0] for root in original)
    for parent, (root, interval) in enumerate(zip(original, root_intervals, strict=True)):
        for vertex, parameter in zip((root.start, root.stop), interval, strict=True):
            result[vertex] = origin_points[vertex]
            addresses.setdefault(vertex, {})[parent] = parameter
    for vertex, first, second, right in splits:
        if (
            vertex in result
            or first >= second
            or first not in result
            or second not in result
            or not isinstance(right, Fraction)
            or not 0 < right < 1
            or right.denominator & (right.denominator - 1)
        ):
            raise ValueError("Native CAD curve association has an invalid split stencil.")
        shared = addresses[first].keys() & addresses[second].keys()
        if len(shared) != 1:
            raise ValueError("Native CAD curve association split crosses original roots.")
        (parent,) = shared
        a, b = addresses[first][parent], addresses[second][parent]
        if not _is_curve_piece(min(a, b), max(a, b), *root_intervals[parent]):
            raise ValueError(
                "Native CAD curve association split lacks a dyadic parent edge."
            )
        left = 1 - right
        # Bisection sums two represented weighted endpoint coordinates; zero
        # slots do not change the numeric result. Preserve the epoch grouping.
        result[vertex] = float(left) * result[first] + float(right) * result[second]
        addresses[vertex] = {parent: left * a + right * b}
    return result, addresses


def _replay_curve_intervals(
    parent: int,
    root: _CurveSegment,
    parts: tuple[_CurveSegment, ...],
    points: dict[int, np.ndarray],
    expected_points: dict[int, np.ndarray],
    addresses: dict[int, dict[int, Fraction]],
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Match every current curve edge to its represented and exact root ancestry."""
    if (
        not parts
        or (parts[0].low, parts[0].start, parts[-1].high, parts[-1].stop)
        != (root.low, root.start, root.high, root.stop)
        or any(
            (left.high, left.stop) != (right.low, right.start)
            for left, right in zip(parts, parts[1:], strict=False)
        )
    ):
        raise ValueError(
            "Native CAD curve association does not partition its exact root endpoints."
        )
    root_low, root_high = _curve_interval(root.low, root.high)[0]
    error = 8 * np.finfo(np.float64).eps * (abs(root.low) + abs(root.high))
    intervals: list[tuple[Fraction, Fraction]] = []
    for part in parts:
        low, high = (
            addresses.get(vertex, {}).get(parent) for vertex in (part.start, part.stop)
        )
        if (
            low is None
            or high is None
            or not _is_curve_piece(low, high, root_low, root_high)
        ):
            raise ValueError(
                "Native CAD curve association has no exact bisection ancestry."
            )
        # Interpolated chart parameter shadows may round; positions do not
        # use this budget and must match their represented stencil exactly.
        if abs(part.low - float(low)) > error or abs(part.high - float(high)) > error:
            raise ValueError(
                "Native CAD curve association has inconsistent parameter shadows."
            )
        for vertex in (part.start, part.stop):
            if not np.array_equal(points[vertex], expected_points[vertex]):
                if vertex in (root.start, root.stop):
                    raise ValueError(
                        "Native CAD curve association moved a retained root endpoint."
                    )
                raise ValueError(
                    "Native CAD curve association vertex does not lie at its stencil combination."
                )
        intervals.append((low, high))
    return tuple(intervals)


def _curve_association(
    mesh: CellMesh,
    domain: MeshingDomain,
    tolerance: float,
    original_mesh: CellMesh,
    original: tuple[_CurveSegment, ...],
    current: tuple[_CurveSegment, ...],
    roots: tuple[np.ndarray, np.ndarray, np.ndarray],
    curve_splits: tuple[_CurveSplit, ...],
    /,
) -> GeometryAssociation | None:
    from .providers._native_plc_geometry import simplex_volume_entities

    if not original and not current:
        return None
    edges, _ = simplex_volume_entities(mesh)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ordered = {
        frozenset(pair): (int(edge), pair[0])
        for edge, pair in zip(
            np.asarray(mesh.entity_set(1).entity_ids).tolist(),
            vertex_ids[edges].tolist(),
            strict=True,
        )
    }
    points = dict(zip(vertex_ids.tolist(), np.asarray(mesh.coordinates), strict=True))
    original_points = dict(
        zip(
            np.asarray(original_mesh.vertex_global_ids, dtype=np.int64).tolist(),
            np.asarray(original_mesh.coordinates),
            strict=True,
        )
    )
    expected_points, addresses = _replay_curve_splits(
        original, original_points, curve_splits
    )
    occurrences = _Occurrences(domain)
    targets, indices, orientations, entity_ids, paths, intervals, parents = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    offset = 0
    for parent, root in enumerate(original):
        end = offset
        while (
            end < len(current)
            and current[end].curve == root.curve
            and current[end].high <= root.high
        ):
            end += 1
        parts = current[offset:end]
        exact_parts = _replay_curve_intervals(
            parent, root, parts, points, expected_points, addresses
        )
        offset = end
        index, entity_id, path = occurrences(1, root.curve)
        for part, interval in zip(parts, exact_parts, strict=True):
            edge = ordered.get(frozenset((part.start, part.stop)))
            if edge is None:
                raise ValueError(
                    "Native CAD curve association chord is not a target mesh edge."
                )
            target, first = edge
            targets.append(target)
            indices.append(index)
            orientations.append(1 if first == part.start else -1)
            entity_ids.append(entity_id)
            paths.append(path)
            intervals.append(interval)
            parents.append(parent)
    if offset != len(current):
        raise ValueError(
            "Native CAD curve association contains a chord outside its original roots."
        )
    count = len(targets)
    return GeometryAssociation(
        GeometryAssociationKind.CURVE,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(1).entity_set_id,
        np.asarray(targets, dtype=np.int64),
        tuple(entity_ids),
        np.full(count, tolerance, dtype=np.float64),
        source_dimensions=np.ones(count, dtype=np.int8),
        source_indices=np.asarray(indices, dtype=np.int32),
        source_roles=(GeometrySourceEntityRole.EDGE,) * count,
        source_occurrence_paths=tuple(paths),
        source_parameter_intervals=tuple(intervals),
        orientations=np.asarray(orientations, dtype=np.int8),
        parent_dimensions=np.ones(count, dtype=np.int8),
        parent_ids=np.asarray(parents, dtype=np.int64),
        parent_association_id=_root_id(domain.source_id, domain.source_revision, *roots),
        provenance=GeometryAssociationProvenance.LINEAGE,
        exact=False,
    )


def _derive_native_cad_tables(
    mesh: CellMesh,
    domain: MeshingDomain,
    regions: np.ndarray,
    witness: NativeCadSurfaceWitness,
    tolerance: float,
    origin_mesh: CellMesh | None,
    origin_chart: NativeCadSurfaceWitness | None,
    curve_splits: tuple[_CurveSplit, ...],
    /,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    GeometryAssociation | None,
    np.ndarray,
    tuple[GeometryAssociation, ...],
]:
    _require_solid_region(mesh, domain, regions)
    if (
        witness.mesh_id != mesh.mesh_id
        or (origin_mesh is None) != (origin_chart is None)
        or (origin_mesh is None and curve_splits)
    ):
        raise ValueError("Native CAD association chart or origin binds another mesh.")
    faces = _chart_face_rows(mesh, witness, domain)
    current = _chart_curve_segments(mesh, domain, witness)
    root_mesh = mesh if origin_mesh is None else origin_mesh
    original = (
        current
        if origin_chart is None
        else _chart_curve_segments(root_mesh, domain, origin_chart)
    )
    roots = _root_banks(root_mesh, original)
    curve = _curve_association(
        mesh, domain, tolerance, root_mesh, original, current, roots, curve_splits
    )
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    slivers = {
        tuple(sorted({int(ids[vertex]) for vertex in row}))
        for row in np.asarray(witness.chart_triangles)[
            np.asarray(witness.chart_to_mesh) < 0
        ].tolist()
    }
    if any(len(pair) != 2 for pair in slivers):
        raise ValueError("A native CAD pole sliver does not pair two vertices.")
    brep = _native_cad_associations(
        mesh, witness, domain, tolerance, current, curve, roots[0], faces
    )
    sliver_vertices = _readonly(
        np.asarray(sorted(slivers), dtype=np.int64).reshape((-1, 2)), np.int64
    )
    return roots, curve, sliver_vertices, brep if curve is None else (*brep, curve)


def _issue_records(
    mesh: CellMesh,
    domain: MeshingDomain,
    regions: np.ndarray,
    witness: NativeCadSurfaceWitness,
    tolerance: float,
    origin_mesh: CellMesh | None,
    origin_chart: NativeCadSurfaceWitness | None,
    curve_splits: tuple[_CurveSplit, ...],
    /,
) -> tuple[NativeCadRecords, tuple[GeometryAssociation, ...]]:
    tables: list[tuple[GeometryAssociation, ...]] = []
    records = NativeCadRecords(
        mesh,
        domain,
        witness,
        tolerance,
        regions,
        origin_mesh,
        origin_chart,
        curve_splits,
        tables,
    )
    return records, tables[0]


def native_cad_records(
    mesh: CellMesh,
    domain: MeshingDomain,
    regions: np.ndarray,
    witness: NativeCadSurfaceWitness,
    tolerance: float,
    /,
) -> tuple[NativeCadRecords, tuple[GeometryAssociation, ...]]:
    """Issue native association authority from original construction operands."""
    return _issue_records(mesh, domain, regions, witness, tolerance, None, None, ())


def _chart_face_rows(
    mesh: CellMesh, witness: NativeCadSurfaceWitness, domain: MeshingDomain, /
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve each boundary triangle's source patch from its chart row."""
    triangles = np.asarray(witness.surface_triangles, dtype=np.int64)
    charts = np.asarray(witness.chart_triangles, dtype=np.int64)
    mapping = np.asarray(witness.chart_to_mesh, dtype=np.int64)
    patch_rows = np.asarray(witness.chart_patches, dtype=np.int64)
    uv = np.asarray(witness.chart_uv, dtype=np.float64)
    if (
        charts.ndim != 2
        or charts.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or mapping.shape != (charts.shape[0],)
        or patch_rows.shape != mapping.shape
        or uv.shape != (*charts.shape, 2)
        or np.asarray(witness.edge_chord).shape != charts.shape
        or np.any(charts < 0)
        or np.any(charts >= mesh.coordinates.shape[0])
        or np.any(mapping < -1)
        or np.any(patch_rows < 0)
        or np.any(patch_rows >= len(domain.patches))
        or not np.all(np.isfinite(uv))
    ):
        raise ValueError("Native CAD association chart rows are misaligned.")
    physical = np.flatnonzero(mapping >= 0)
    targets = mapping[physical]
    if not np.array_equal(
        np.sort(targets), np.arange(triangles.shape[0], dtype=np.int64)
    ):
        raise ValueError(
            "Native CAD association chart must cover each boundary triangle."
        )
    # Face orientation comes from the physical triangle. V4 separately decides
    # whether the chart is oriented correctly, without changing proof dispatch.
    if not np.array_equal(
        np.sort(charts[physical], axis=1), np.sort(triangles[targets], axis=1)
    ):
        raise ValueError("Native CAD association chart changes its boundary triangle.")
    patches = np.empty((triangles.shape[0],), dtype=np.int64)
    patches[targets] = patch_rows[physical]
    return triangles, patches


def _bind_chart_corners(
    mesh: CellMesh,
    witness: NativeCadSurfaceWitness,
    domain: MeshingDomain,
    segments: tuple[_CurveSegment, ...],
    classes: np.ndarray,
    parameters: np.ndarray,
    /,
) -> None:
    """Classify corners from source-use endpoint incidence, not position lookup."""
    vertex_rows = {
        identifier: row
        for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids).tolist())
    }
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners: dict[int, int] = {}
    rows_by_corner: dict[int, int] = {}

    def bind(vertex: int, corner: int, /) -> None:
        if (
            corners.setdefault(vertex, corner) != corner
            or rows_by_corner.setdefault(corner, vertex) != vertex
            or not np.array_equal(points[vertex], domain.corner_points[corner])
        ):
            raise ValueError(
                "Native CAD association corner disagrees with source incidence."
            )

    for segment in segments:
        patch, loop, use = domain.curve_owners[segment.curve]
        owner = domain.patches[patch].loops[loop][use]
        source = domain.curves[segment.curve]
        if owner.first < owner.last:
            if segment.low == owner.first:
                bind(vertex_rows[segment.start], source.start)
            if segment.high == owner.last:
                bind(vertex_rows[segment.stop], source.end)
        else:
            if segment.high == owner.first:
                bind(vertex_rows[segment.stop], source.start)
            if segment.low == owner.last:
                bind(vertex_rows[segment.start], source.end)

    uses = tuple(use for patch in domain.patches for loop in patch.loops for use in loop)
    for row, local in zip(*np.nonzero(witness.edge_chord >= 0), strict=True):
        chord = witness.edge_chord[row, local]
        use = uses[witness.chord_use[chord]]
        if isinstance(use, PatchCurveUse):
            continue
        for vertex in witness.chart_triangles[row, [local, (local + 1) % 3]]:
            bind(vertex, use.corner)

    for vertex, corner in corners.items():
        classes[vertex] = (0, corner)
        parameters[vertex] = np.nan


def _vertex_sources(
    mesh: CellMesh,
    witness: NativeCadSurfaceWitness,
    domain: MeshingDomain,
    segments: tuple[_CurveSegment, ...],
    curve: GeometryAssociation | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    classes = np.zeros((points.shape[0], 2), dtype=np.int64)
    classes[:, 0] = 3
    parameters = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    triangles = np.asarray(witness.chart_triangles, dtype=np.int64)
    patches = np.asarray(witness.chart_patches, dtype=np.int64)
    classes[triangles.ravel(), 0] = 2
    classes[triangles.ravel(), 1] = np.repeat(patches, 3)
    parameters[triangles.ravel()] = np.asarray(witness.chart_uv).reshape((-1, 2))
    _bind_chart_corners(mesh, witness, domain, segments, classes, parameters)
    if curve is not None:
        id_to_row = {
            identifier: row
            for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids).tolist())
        }
        for segment, interval in zip(
            segments, curve.source_parameter_intervals, strict=True
        ):
            if interval is None:
                raise ValueError(
                    "Native CAD curve association is missing its exact interval."
                )
            for identifier, parameter in zip(
                (segment.start, segment.stop), interval, strict=True
            ):
                vertex = id_to_row[identifier]
                dimension, entity = classes[vertex]
                if dimension == 0:
                    continue
                if dimension == 1 and (
                    entity != segment.curve or parameters[vertex, 0] != float(parameter)
                ):
                    raise ValueError(
                        "Native CAD association has conflicting curve vertex parameters."
                    )
                classes[vertex] = (1, segment.curve)
                parameters[vertex] = (float(parameter), np.nan)
    return classes, parameters


def _native_cad_associations(
    mesh: CellMesh,
    witness: NativeCadSurfaceWitness,
    domain: MeshingDomain,
    tolerance: float,
    segments: tuple[_CurveSegment, ...],
    curve: GeometryAssociation | None,
    root_curves: np.ndarray,
    boundary: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[GeometryAssociation, ...]:
    """B-Rep rows tied to the actual chart, source occurrences and material side."""
    from .providers._native_plc_geometry import simplex_volume_entities

    edges, faces = simplex_volume_entities(mesh)
    triangles, patches = boundary
    occurrences = _Occurrences(domain)
    region_key = domain.stratum_entity_ids(3)[0]
    patch_sides = {
        patch: (None, region_key) if orientation > 0 else (region_key, None)
        for patch, orientation in domain.regions[0].boundary
    }
    vertex_class, vertex_parameters = _vertex_sources(
        mesh, witness, domain, segments, curve
    )

    edge_rows = _sorted_rows(edges)
    edge_class = np.zeros((edges.shape[0], 2), dtype=np.int64)
    edge_class[:, 0] = 3
    edge_orientations = np.zeros((edges.shape[0],), dtype=np.int8)
    for triangle, patch in zip(triangles.tolist(), patches.tolist(), strict=True):
        for a, b in _CHART_EDGES:
            row = edge_rows[
                (min(triangle[a], triangle[b]), max(triangle[a], triangle[b]))
            ]
            edge_class[row] = (2, patch)
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
            edge_class[row] = (1, int(root_curves[parent]))
            edge_orientations[row] = orientation

    face_rows = _sorted_rows(faces)
    face_class = np.zeros((faces.shape[0], 2), dtype=np.int64)
    face_class[:, 0] = 3
    face_orientations = np.zeros((faces.shape[0],), dtype=np.int8)
    face_sides: list[tuple[str | None, str | None]] = [(None, None)] * faces.shape[0]
    for triangle, patch in zip(triangles.tolist(), patches.tolist(), strict=True):
        row = face_rows[tuple(sorted(triangle))]
        face_class[row] = (2, patch)
        # The canonical face order is an even permutation of the outward
        # triangle exactly when it is a rotation of it.
        canonical = [int(value) for value in faces[row]]
        rotations = {tuple(triangle[k:] + triangle[:k]) for k in range(3)}
        face_orientations[row] = 1 if tuple(canonical) in rotations else -1
        if patch not in patch_sides:
            raise ValueError("Native CAD association patch does not bound the solid.")
        face_sides[row] = patch_sides[patch]
    cell_class = np.zeros((mesh.entity_set(3).count, 2), dtype=np.int64)
    cell_class[:, 0] = 3

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

    return (
        association(0, vertex_class, vertex_parameters, None, None),
        association(1, edge_class, None, edge_orientations, None),
        association(2, face_class, None, face_orientations, face_sides),
        association(3, cell_class, None, None, None),
    )


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
    "native_cad_records",
    "NativeCadRecords",
    "retarget_native_cad_surface",
]
