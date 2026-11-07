from __future__ import annotations

import os
import traceback
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellGeometrySpec,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    PreparedFieldReconstruction,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.geometry._exact_polyhedral_geometry import _segment_contact_points2
from phydrax.geometry._overset_interfaces import AtomicEdge
from phydrax.geometry._overset_partition import OversetPartition
from phydrax.geometry._planar_primitives import cross
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.lifecycle import (
    Composition,
    CompositionEntry,
    CompositionTransport,
)
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshQualityTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._motion_source import NativePlanarMotionSource
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativePlanarSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


_BACKGROUND = "background"
_BACKGROUND_HALF_WIDTH = 3.0
_RESOLUTION = 4
_INTERPOLATION_TOLERANCE = 1.0e-10
_ORPHAN_COUNT_TOLERANCE = 0
_CONSERVATIVE_CONTENT_TOLERANCE = 1.0e-10
_PDE_RESIDUAL_TOLERANCE = 1.0e-9
_FLUX_BALANCE_RELATIVE_TOLERANCE = 1.0e-8
_CONTENT_ENTRY_ID = "qualification:background-extensive-content"
_CONTRACT = SpatialCoordinateContract.si()


class _Body(NamedTuple):
    name: str
    center: tuple[float, float]
    half_width: float
    cavity_half_width: float
    refinement: int
    offsets: tuple[tuple[float, float], ...]
    offset_rule: str

    def offset(self, step: int, /) -> np.ndarray:
        if self.offset_rule == "cycle":
            value = self.offsets[step % len(self.offsets)]
        elif self.offset_rule == "first-then-repeat-last":
            value = self.offsets[min(step, len(self.offsets) - 1)]
        else:
            raise ValueError(f"Unknown native overset motion rule {self.offset_rule!r}.")
        return np.asarray(value, dtype=np.float64)

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center": list(self.center),
            "half_width": self.half_width,
            "cavity_half_width": self.cavity_half_width,
            "refinement": self.refinement,
            "offsets": [list(value) for value in self.offsets],
            "offset_rule": self.offset_rule,
        }


class _Profile(NamedTuple):
    name: str
    fringe_layers: int
    bodies: tuple[_Body, ...]

    def record(self) -> dict[str, object]:
        return {
            "fringe_layers": self.fringe_layers,
            "bodies": [body.record() for body in self.bodies],
        }


_PROFILES = (
    _Profile(
        "routine",
        1,
        (
            _Body(
                "moving",
                (0.0, 0.0),
                2.25,
                0.25,
                1,
                ((0.55, 0.0), (0.05, 0.0)),
                "first-then-repeat-last",
            ),
        ),
    ),
    _Profile(
        "combined",
        2,
        (
            _Body(
                "alpha",
                (-1.15, -0.3),
                1.7,
                0.3,
                1,
                ((0.0, 0.45), (0.0, -0.45)),
                "cycle",
            ),
            _Body(
                "beta",
                (1.55, 0.5),
                1.3,
                0.2,
                2,
                ((-0.05, -0.4), (0.05, 0.4)),
                "cycle",
            ),
        ),
    ),
    _Profile(
        "larger",
        1,
        (
            _Body(
                "port",
                (-1.5, 0.0),
                1.25,
                0.25,
                2,
                ((0.0, 0.5), (0.0, -0.5)),
                "cycle",
            ),
            _Body(
                "starboard",
                (1.5, 0.0),
                1.25,
                0.25,
                2,
                ((0.0, -0.5), (0.0, 0.5)),
                "cycle",
            ),
        ),
    ),
)


def _square_region(half_width: float, feature_id: str, /) -> PlanarMeshRegion:
    return PlanarMeshRegion(
        np.asarray(
            (
                (-half_width, -half_width),
                (half_width, -half_width),
                (half_width, half_width),
                (-half_width, half_width),
            ),
            dtype=np.float64,
        ),
        ((0, 1, 2, 3),),
        feature_id=feature_id,
    )


def _annulus_region(body: _Body, /) -> PlanarMeshRegion:
    cx, cy = body.center
    half, cavity = body.half_width, body.cavity_half_width
    vertices = np.asarray(
        (
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
            (cx - cavity, cy - cavity),
            (cx - cavity, cy + cavity),
            (cx + cavity, cy + cavity),
            (cx + cavity, cy - cavity),
        ),
        dtype=np.float64,
    )
    return PlanarMeshRegion(
        vertices,
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id=body.name,
    )


def _source_scope(
    source: NativePlanarSource,
    dimension: int,
    entity_indices: np.ndarray,
    /,
) -> MeshingScope:
    return MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        source.scope_entity_set_id(dimension),
        entity_indices,
    )


def _native_part(
    name: str,
    region: PlanarMeshRegion,
    /,
    *,
    body: _Body | None,
    resolution_scale: int = 1,
) -> tuple[MeshPart, NativePlanarSource, object, dict[str, object]]:
    loops = ((0, 1, 2, 3),) if body is None else ((0, 1, 2, 3), (4, 5, 6, 7))
    revision = canonical_fingerprint(
        {
            "kind": "native-moving-overset-source-r1",
            "part": name,
            "profile_body": None if body is None else body.record(),
            "background_half_width": _BACKGROUND_HALF_WIDTH if body is None else None,
            "vertices": np.asarray(region.vertices, dtype=np.float64).tolist(),
            "loops": loops,
        }
    )
    vertex_ids = np.arange(len(region.vertices), dtype=np.int64)
    edge_ids = np.arange(len(region.edges), dtype=np.int64)
    source = NativePlanarSource(
        region,
        revision,
        vertex_entity_ids=vertex_ids,
        edge_entity_ids=edge_ids,
        region_scope_entity_ids=np.asarray((0,), dtype=np.int64),
    )
    fluid = _source_scope(
        source,
        2,
        source.scope_entity_ids(2),
    )
    exterior = _source_scope(
        source,
        1,
        edge_ids[:4],
    )
    protected = [
        ProtectedFeature(
            exterior,
            FeatureKind.CURVE,
            maximum_deviation=1.0e-10,
            hard=True,
        )
    ]
    if body is not None:
        wall = _source_scope(
            source,
            1,
            edge_ids[4:8],
        )
        protected.append(
            ProtectedFeature(
                wall,
                FeatureKind.CURVE,
                maximum_deviation=1.0e-10,
                hard=True,
            )
        )
    refinement = 1 if body is None else body.refinement
    size = 0.8 / (_RESOLUTION * refinement * resolution_scale)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        fluid,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            UniformSizeControl(
                fluid,
                size,
                maximum_size=1.5 * size,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        ),
        protected_features=tuple(protected),
        quality_target=MeshQualityTarget(minimum_angle=np.radians(25.0), hard=True),
    )
    plan = NativeMeshingProvider(
        NativeMeshingOptions("planar_constrained_delaunay")
    ).plan(source, specification, coordinate_contract=_CONTRACT)
    result = plan.execute()
    part = MeshPart(name, result)
    generated = accepted_result_record(result)
    generated.update(
        {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "source_binding_id": source.binding_id,
            "request_id": specification.specification_id,
            "prepared_plan_id": plan.plan_id,
            "input_vertices": np.asarray(region.vertices, dtype=np.float64).tolist(),
            "input_loops": [list(loop) for loop in loops],
            "stable_vertex_ids": np.asarray(
                result.mesh.vertex_global_ids, dtype=np.int64
            ).tolist(),
            "stable_cell_ids_by_block": {
                block.name: np.asarray(block.global_ids, dtype=np.int64).tolist()
                for block in result.mesh.blocks
            },
            "stable_edge_ids": np.asarray(
                result.mesh.entity_set(1).entity_ids, dtype=np.int64
            ).tolist(),
            "source_associated_edge_ids": {
                "exterior": _source_edge_ids(part, source, (0, 1, 2, 3)).tolist(),
                "cavity_wall": []
                if body is None
                else _source_edge_ids(part, source, (4, 5, 6, 7)).tolist(),
            },
            "size_control": {
                "resolution": _RESOLUTION,
                "resolution_scale": resolution_scale,
                "refinement": refinement,
                "target_size": size,
                "maximum_size": 1.5 * size,
                "strength": "hard",
                "p50_relative_tolerance": 0.5,
                "compliance_passed": result.compliance.passed,
                "compliance_requested": dict(result.compliance.requested),
                "compliance_achieved": dict(result.compliance.achieved),
            },
            "protected_feature_maximum_deviation_m": 1.0e-10,
            "hard_minimum_angle_degrees": 25.0,
        }
    )
    return part, source, plan, generated


def _part_result(part: MeshPart, /) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            f"Native moving overset requires a CellMeshingResult carrier for {part.name!r}."
        )
    return carrier


def _part_connectivity(part: MeshPart, /) -> PolygonalConnectivity:
    result = _part_result(part)
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            f"Native moving overset requires polygonal connectivity for {part.name!r}."
        )
    return connectivity


def _motion_result(advance: phx.meshing.MeshMotionAdvance, /) -> CellMeshingResult:
    result = advance.result
    if result is None:
        raise RuntimeError("Accepted native mesh motion omitted its CellMeshingResult.")
    return result


def _source_edge_ids(
    part: MeshPart,
    source: NativePlanarSource,
    edge_indices: tuple[int, ...],
    /,
) -> np.ndarray:
    carrier = _part_result(part)
    edge_set = carrier.mesh.entity_set(1)
    requested = {
        source.source_entity_key_for_scope(1, int(source.edge_entity_ids[edge]))[0]: edge
        for edge in edge_indices
    }
    selected: list[int] = []
    covered: set[int] = set()
    for association in carrier.associations:
        if (
            association.association_kind is not phx.meshing.GeometryAssociationKind.CURVE
            or association.source_id != source.source_id
            or association.source_revision != source.source_revision
            or association.target_entity_set_id != edge_set.entity_set_id
        ):
            continue
        for identifier, entity, role in zip(
            np.asarray(association.target_global_ids),
            association.source_entity_ids,
            association.source_roles,
            strict=True,
        ):
            if role is GeometrySourceEntityRole.EDGE and entity in requested:
                selected.append(int(identifier))
                covered.add(requested[entity])
    if covered != set(edge_indices):
        raise RuntimeError(
            f"Native overset source association omitted loop edges {sorted(set(edge_indices) - covered)!r}."
        )
    return np.unique(np.asarray(selected, dtype=np.int64))


def _part_source_measures(
    part: MeshPart,
    body: _Body | None,
    source: NativePlanarSource,
    /,
) -> dict[str, object]:
    result = _part_result(part)
    mesh = result.mesh
    connectivity = _part_connectivity(part)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    triangles = points[cells]
    areas = 0.5 * np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    if body is None:
        expected_area = 4.0 * _BACKGROUND_HALF_WIDTH**2
        outer_expected = 8.0 * _BACKGROUND_HALF_WIDTH
        wall_edges = np.empty((0,), dtype=np.int64)
    else:
        expected_area = 4.0 * (body.half_width**2 - body.cavity_half_width**2)
        outer_expected = 8.0 * body.half_width
        wall_edges = _source_edge_ids(part, source, (4, 5, 6, 7))
    exterior_edges = _source_edge_ids(part, source, (0, 1, 2, 3))
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_rows = np.asarray(connectivity.edges, dtype=np.int64)
    exterior_rows = np.flatnonzero(np.isin(edge_ids, exterior_edges))
    wall_rows = np.flatnonzero(np.isin(edge_ids, wall_edges))
    edge_points = points[edge_rows]
    lengths = np.linalg.norm(edge_points[:, 1] - edge_points[:, 0], axis=1)
    measured_area = float(np.sum(areas))
    exterior_length = float(np.sum(lengths[exterior_rows]))
    wall_length = float(np.sum(lengths[wall_rows]))
    wall_expected = 0.0 if body is None else 8.0 * body.cavity_half_width
    if (
        np.any(areas <= 0.0)
        or abs(measured_area - expected_area) > 1.0e-9
        or abs(exterior_length - outer_expected) > 1.0e-9
        or abs(wall_length - wall_expected) > 1.0e-9
    ):
        raise RuntimeError(
            f"Native overset source measure/association oracle failed for {part.name!r}: "
            f"{measured_area=}, {expected_area=}, {exterior_length=}, "
            f"{outer_expected=}, {wall_length=}, {wall_expected=}."
        )
    return {
        "expected_area": expected_area,
        "independent_triangle_area": measured_area,
        "expected_exterior_length": outer_expected,
        "associated_exterior_length": exterior_length,
        "expected_cavity_wall_length": wall_expected,
        "associated_cavity_wall_length": wall_length,
        "triangle_count": int(cells.shape[0]),
        "all_triangle_measures_positive": bool(np.all(areas > 0.0)),
        "source_association_edges": {
            "exterior_count": int(exterior_edges.size),
            "cavity_wall_count": int(wall_edges.size),
        },
        "analytic_measure_tolerance_absolute": 1.0e-9,
    }


def _make_connectivity(
    profile: _Profile,
    parts: dict[str, MeshPart],
    sources: dict[str, NativePlanarSource],
    /,
) -> phx.meshing.OversetConnectivity:
    specifications = [phx.meshing.OversetPartSpec(_BACKGROUND)]
    for body in profile.bodies:
        part = parts[body.name]
        source = sources[body.name]
        specifications.append(
            phx.meshing.OversetPartSpec(
                body.name,
                wall=part.scope(1, _source_edge_ids(part, source, (4, 5, 6, 7))),
                boundary=part.scope(
                    1,
                    _source_edge_ids(part, source, (0, 1, 2, 3)),
                ),
            )
        )
    return phx.meshing.prepare_overset_connectivity(
        MeshAssembly(
            tuple(
                parts[name]
                for name in (_BACKGROUND, *(body.name for body in profile.bodies))
            )
        ),
        tuple(specifications),
        policy=phx.meshing.OversetPolicy(fringe_layers=profile.fringe_layers),
        partition_priority=(*(body.name for body in profile.bodies), _BACKGROUND),
        partition_sources={body.name: sources[body.name] for body in profile.bodies},
    )


def _inside_closed_rectangle(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    return np.all(
        (points >= lower[None, :] - 1.0e-12) & (points <= upper[None, :] + 1.0e-12),
        axis=1,
    )


def _wall_segments(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        (
            (lower[0], lower[1], lower[0], upper[1]),
            (lower[0], upper[1], upper[0], upper[1]),
            (upper[0], lower[1], upper[0], upper[1]),
            (lower[0], lower[1], upper[0], lower[1]),
        ),
        dtype=np.float64,
    ).reshape((4, 2, 2))


def _segment_meets_triangle(segment: np.ndarray, triangle: np.ndarray, /) -> bool:
    endpoints = tuple(
        (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
        for point in segment
    )
    vertices = tuple(
        (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
        for point in triangle
    )
    edges = tuple(zip(vertices, vertices[1:] + vertices[:1], strict=True))
    for endpoint in endpoints:
        signs = tuple(cross(a, b, endpoint) for a, b in edges)
        if all(value >= 0 for value in signs) or all(value <= 0 for value in signs):
            return True
    return any(
        _segment_contact_points2(endpoints[0], endpoints[1], a, b) for a, b in edges
    )


def _independent_blanking(
    connectivity: phx.meshing.OversetConnectivity,
    profile: _Profile,
    sources: dict[str, NativePlanarSource],
    /,
) -> dict[str, object]:
    connectivity.require_complete()
    partition = connectivity.bound_partition
    if partition is None:
        raise RuntimeError(
            "Independent blanking requires the source-bound exact partition."
        )
    cell_proofs = {(cell.part_name, cell.cell_index): cell for cell in partition.cells}
    owned_areas: dict[tuple[str, int], Fraction] = {}
    for piece in partition.pieces:
        key = (piece.part_name, piece.cell_index)
        owned_areas[key] = owned_areas.get(key, Fraction(0)) + piece.area
    parts = {part.name: part for part in connectivity.assembly.parts}
    specs = {spec.part_name: spec for spec in connectivity.specs}
    walls = {
        body.name: (np.min(polygon, axis=0), np.max(polygon, axis=0))
        for body, polygon in zip(
            profile.bodies, _cavity_polygons(profile, connectivity, sources), strict=True
        )
    }
    reports: dict[str, object] = {}
    for name, part in parts.items():
        result = _part_result(part)
        mesh = result.mesh
        mesh_connectivity = _part_connectivity(part)
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
        holes = np.zeros(points.shape[0], dtype=np.bool_)
        segments: list[np.ndarray] = []
        for other_name, (lower, upper) in walls.items():
            if other_name == name:
                continue
            holes |= _inside_closed_rectangle(points, lower, upper)
            segments.extend(_wall_segments(lower, upper))
        cell_lower = np.min(points[cells], axis=1)
        cell_upper = np.max(points[cells], axis=1)
        cut = np.zeros(cells.shape[0], dtype=np.bool_)
        for edge in segments:
            edge_lower = np.min(edge, axis=0)
            edge_upper = np.max(edge, axis=0)
            candidates = np.flatnonzero(
                np.all(
                    (cell_upper >= edge_lower[None, :])
                    & (cell_lower <= edge_upper[None, :]),
                    axis=1,
                )
            )
            for row in candidates:
                if not cut[row] and _segment_meets_triangle(edge, points[cells[row]]):
                    cut[row] = True
        cell_holes = np.any(holes[cells], axis=1) | cut
        boundary_vertices = np.zeros(points.shape[0], dtype=np.bool_)
        protected_vertices = np.zeros(points.shape[0], dtype=np.bool_)
        edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        edge_rows = np.asarray(mesh_connectivity.edges, dtype=np.int64)
        boundary_scope = specs[name].boundary
        if boundary_scope is not None:
            boundary_rows = np.flatnonzero(
                np.isin(edge_ids, np.asarray(boundary_scope.entity_ids, dtype=np.int64))
            )
            boundary_vertices[np.unique(edge_rows[boundary_rows])] = True
        wall_scope = specs[name].wall
        if wall_scope is not None:
            wall_rows = np.flatnonzero(
                np.isin(edge_ids, np.asarray(wall_scope.entity_ids, dtype=np.int64))
            )
            protected_vertices[np.unique(edge_rows[wall_rows])] = True
        touched = np.zeros(points.shape[0], dtype=np.bool_)
        if np.any(cell_holes):
            touched[cells[cell_holes].reshape((-1,))] = True
        frontier = (touched | boundary_vertices) & ~holes
        layers = np.zeros(points.shape[0], dtype=np.int32)
        layers[frontier] = 1
        for depth in range(2, profile.fringe_layers + 1):
            active = np.flatnonzero(~cell_holes & np.any(frontier[cells], axis=1))
            grown = np.zeros(points.shape[0], dtype=np.bool_)
            if active.size:
                grown[cells[active].reshape((-1,))] = True
            frontier = grown & ~holes & (layers == 0)
            layers[frontier] = depth
        layers[protected_vertices & ~holes] = 0
        vertex_status = np.select(
            (holes, layers > 0),
            (
                int(phx.meshing.OversetVertexStatus.HOLE),
                int(phx.meshing.OversetVertexStatus.RECEPTOR),
            ),
            int(phx.meshing.OversetVertexStatus.ACTIVE),
        ).astype(np.int8)
        cell_status = np.select(
            (cell_holes, np.any(layers[cells] > 0, axis=1)),
            (
                int(phx.meshing.OversetCellStatus.HOLE),
                int(phx.meshing.OversetCellStatus.FRINGE),
            ),
            int(phx.meshing.OversetCellStatus.ACTIVE),
        ).astype(np.int8)
        actual = connectivity.blanking_of(name)
        expected_arrays = (
            ("vertex_status", vertex_status, np.asarray(actual.vertex_status)),
            ("fringe_layer", layers, np.asarray(actual.fringe_layer)),
            ("cell_status", cell_status, np.asarray(actual.cell_status)),
        )
        for field_name, expected, observed in expected_arrays:
            if not np.array_equal(expected, observed):
                raise RuntimeError(
                    f"Independent overset {field_name} oracle disagrees for {name!r}."
                )
        ambiguous = cut & ~np.any(holes[cells], axis=1)
        if not np.array_equal(ambiguous, np.asarray(actual.ambiguous_cells)):
            raise RuntimeError(
                f"Independent overset ambiguous-cell oracle disagrees for {name!r}."
            )
        if np.any(np.asarray(actual.protected_conflict)):
            raise RuntimeError(f"Native overset protected-wall conflict for {name!r}.")
        for row in np.flatnonzero(ambiguous):
            proof = cell_proofs.get((name, int(row)))
            triangle = tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in points[cells[row]]
            )
            twice_area = abs(
                (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
                - (triangle[1][1] - triangle[0][1]) * (triangle[2][0] - triangle[0][0])
            )
            if (
                proof is None
                or proof.source_area != twice_area / 2
                or not 0 <= proof.cavity_area <= proof.source_area
                or not 0 <= proof.owned_area <= proof.source_area - proof.cavity_area
                or owned_areas.get((name, int(row)), Fraction(0)) != proof.owned_area
            ):
                raise RuntimeError(
                    f"Ambiguous cell {name!r}:{row} lacks exact partition proof."
                )
        reports[name] = {
            "exact_status_match": True,
            "hole_vertices": int(
                np.count_nonzero(
                    vertex_status == int(phx.meshing.OversetVertexStatus.HOLE)
                )
            ),
            "receptor_vertices": int(
                np.count_nonzero(
                    vertex_status == int(phx.meshing.OversetVertexStatus.RECEPTOR)
                )
            ),
            "hole_cells": int(
                np.count_nonzero(cell_status == int(phx.meshing.OversetCellStatus.HOLE))
            ),
            "fringe_cells": int(
                np.count_nonzero(cell_status == int(phx.meshing.OversetCellStatus.FRINGE))
            ),
            "fringe_layers": profile.fringe_layers,
            "ambiguous_cells": int(np.count_nonzero(ambiguous)),
            "ambiguous_cells_proved_by_exact_partition": int(np.count_nonzero(ambiguous)),
            "protected_conflicts": 0,
        }
    return reports


def _independent_donor_oracle(
    connectivity: phx.meshing.OversetConnectivity,
    /,
) -> dict[str, object]:
    names = connectivity.assembly.global_part_names
    parts = {part.name: part for part in connectivity.assembly.parts}
    donor_data: dict[str, tuple[dict[int, np.ndarray], np.ndarray, set[int]]] = {}
    for name, part in parts.items():
        mesh = _part_result(part).mesh
        rows: dict[int, np.ndarray] = {}
        for block in mesh.blocks:
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            vertices = np.asarray(block.vertices, dtype=np.int64)
            rows.update(
                (int(identifier), vertices[row])
                for row, identifier in enumerate(identifiers)
            )
        blanking = connectivity.blanking_of(name)
        cell_ids = np.asarray(blanking.cell_ids, dtype=np.int64)
        cell_status = np.asarray(blanking.cell_status, dtype=np.int8)
        holes = set(
            map(int, cell_ids[cell_status == int(phx.meshing.OversetCellStatus.HOLE)])
        )
        donor_data[name] = (rows, np.asarray(mesh.coordinates, dtype=np.float64), holes)
    max_residual = 0.0
    routed = 0
    per_part: dict[str, object] = {}
    for evidence in connectivity.receptors:
        target = parts[evidence.part_name]
        target_result = _part_result(target)
        target_mesh = target_result.mesh
        target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
        target_rows = {int(identifier): row for row, identifier in enumerate(target_ids)}
        target_points = np.asarray(target_mesh.coordinates, dtype=np.float64)
        statuses = np.asarray(evidence.status, dtype=np.int32)
        donors = np.asarray(evidence.donor_parts, dtype=np.int32)
        donor_cells = np.asarray(evidence.donor_cells, dtype=np.int64)
        receptor_ids = np.asarray(evidence.receptor_ids, dtype=np.int64)
        residuals = np.asarray(evidence.residuals, dtype=np.float64)
        candidates = np.asarray(evidence.candidate_parts, dtype=np.int32)
        if evidence.orphan_ids.size or np.any(
            statuses != int(phx.meshing.CouplingSearchStatus.FOUND)
        ):
            raise RuntimeError(
                f"Native donor search orphaned receptors on {evidence.part_name!r}."
            )
        part_residual = 0.0
        for receptor_id, donor_index, donor_cell_id, residual, candidate_count in zip(
            receptor_ids,
            donors,
            donor_cells,
            residuals,
            candidates,
            strict=True,
        ):
            if donor_index < 0 or donor_index >= len(names) or candidate_count < 1:
                raise RuntimeError("Native donor route omitted a source-part candidate.")
            donor_name = names[int(donor_index)]
            donor_rows, donor_points, hole_cells = donor_data[donor_name]
            if int(donor_cell_id) not in donor_rows:
                raise RuntimeError(
                    "Native donor route references an absent stable cell ID."
                )
            vertices = donor_rows[int(donor_cell_id)]
            point = target_points[target_rows[int(receptor_id)]]
            corners = donor_points[vertices]
            jacobian = np.column_stack((corners[1] - corners[0], corners[2] - corners[0]))
            reference = np.linalg.solve(jacobian, point - corners[0])
            weights = np.asarray((1.0 - reference.sum(), reference[0], reference[1]))
            defect = float(np.linalg.norm(weights @ corners - point))
            if np.min(weights) < -1.0e-10 or np.max(weights) > 1.0 + 1.0e-10:
                raise RuntimeError("Independent barycentric donor containment failed.")
            if defect > _INTERPOLATION_TOLERANCE or residual > _INTERPOLATION_TOLERANCE:
                raise RuntimeError(
                    "Independent donor reconstruction exceeded its residual tolerance."
                )
            if int(donor_cell_id) in hole_cells:
                raise RuntimeError("Native donor search selected a blanked hole cell.")
            part_residual = max(part_residual, defect, float(residual))
            max_residual = max(max_residual, defect, float(residual))
            routed += 1
        per_part[evidence.part_name] = {
            "receptor_count": int(receptor_ids.size),
            "found_count": int(
                np.count_nonzero(statuses == int(phx.meshing.CouplingSearchStatus.FOUND))
            ),
            "orphan_count": int(evidence.orphan_ids.size),
            "independent_barycentric_maximum_residual": part_residual,
        }
    if connectivity.orphan_count != _ORPHAN_COUNT_TOLERANCE:
        raise RuntimeError(
            f"Accepted overset epoch has {connectivity.orphan_count} donor orphans."
        )
    return {
        "complete": connectivity.complete,
        "donor_search_complete": connectivity.donor_search_complete,
        "orphan_count": connectivity.orphan_count,
        "accepted_orphan_count_tolerance": _ORPHAN_COUNT_TOLERANCE,
        "receptor_routes_independently_contained": routed,
        "maximum_independent_donor_residual": max_residual,
        "residual_tolerance": _INTERPOLATION_TOLERANCE,
        "per_part": per_part,
    }


def _field_transfer(
    connectivity: phx.meshing.OversetConnectivity,
    /,
) -> dict[str, object]:
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    reconstructions: dict[str, PreparedFieldReconstruction] = {}
    coefficients: dict[str, Array] = {}
    for part in connectivity.assembly.parts:
        carrier = _part_result(part)
        geometry = carrier.geometry
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                f"Native planar overset carrier has unsupported FE geometry {type(geometry).__name__}."
            )
        space = FiniteElementPlan(
            carrier.mesh,
            field,
            coordinate_spec=geometry,
        ).prepare()
        reconstruction = prepare_finite_element_field_reconstruction(space, "u")
        dof_points = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
        values = 1.0 + dof_points[:, 0] + 2.0 * dof_points[:, 1]
        reconstructions[part.name] = reconstruction
        coefficients[part.name] = jnp.asarray(values)
    transfer = phx.meshing.prepare_overset_field_transfer(
        connectivity,
        reconstructions,
        "u",
    )
    values = transfer.apply(coefficients)
    errors: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, received in values.items():
        receptor = next(item for item in connectivity.receptors if item.part_name == name)
        part = next(item for item in connectivity.assembly.parts if item.name == name)
        points = np.asarray(_part_result(part).mesh.coordinates, dtype=np.float64)[
            np.asarray(receptor.receptor_rows, dtype=np.int64)
        ]
        exact = 1.0 + points[:, 0] + 2.0 * points[:, 1]
        defect = float(
            np.max(np.abs(np.asarray(received, dtype=np.float64) - exact), initial=0.0)
        )
        if defect > _INTERPOLATION_TOLERANCE:
            raise RuntimeError(
                f"Overset affine interpolation failed on {name!r}: {defect=}."
            )
        errors[name] = defect
        counts[name] = int(np.asarray(received).size)
    return {
        "transfer_id": transfer.transfer_id,
        "field": "u=1+x+2y",
        "route": "native-P1-FE-interpolation-nonconservative",
        "per_receptor_part_maximum_affine_error": errors,
        "per_receptor_part_dof_count": counts,
        "maximum_affine_error": max(errors.values(), default=0.0),
        "tolerance_absolute": _INTERPOLATION_TOLERANCE,
    }


def _motion_advance(
    part: MeshPart,
    target_coordinates: np.ndarray,
    fixed_vertices: np.ndarray,
    numeric_version: str,
    /,
    *,
    source_transition: NativePlanarMotionSource | None = None,
) -> phx.meshing.MeshMotionAdvance:
    source = _part_result(part)
    advance = phx.meshing.advance_mesh_motion(
        phx.meshing.MeshMotionMonitor(source.mesh),
        source,
        target_coordinates,
        boundary_residual=0.0,
        fixed_vertices=fixed_vertices,
        numeric_version=numeric_version,
        source_transition=source_transition,
    )
    if not advance.accepted:
        raise RuntimeError(
            f"Source-bound mesh motion was rejected for {part.name!r}: "
            f"{[(item.reasons, item.failure_detail) for item in advance.assessments]!r}."
        )
    target = advance.result
    if target is None:
        raise RuntimeError(f"Accepted mesh motion omitted its result for {part.name!r}.")
    if (
        advance.source_mesh_id != source.mesh.mesh_id
        or advance.source_result_id != source.result_id
        or target.mesh.topology_id != source.mesh.topology_id
    ):
        raise RuntimeError(
            f"Source-bound mesh motion changed identity for {part.name!r}."
        )
    return advance


def _rigid_body_motion(
    part: MeshPart,
    body: _Body,
    source: NativePlanarSource,
    step: int,
    /,
) -> phx.meshing.MeshMotionAdvance:
    mesh = _part_result(part).mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    target = coordinates + body.offset(step)
    return _motion_advance(
        part,
        target,
        np.zeros((coordinates.shape[0],), dtype=np.bool_),
        f"native-overset-{body.name}-rigid-motion-{step}",
        source_transition=NativePlanarMotionSource(source, body.offset(step)),
    )


def _background_rezone(
    part: MeshPart,
    profile: _Profile,
    step: int,
    /,
) -> phx.meshing.MeshMotionAdvance:
    mesh = _part_result(part).mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    direction = profile.bodies[0].offset(step)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise RuntimeError("The authored profile has no nonzero rigid-motion direction.")
    direction = direction / norm
    scale = 1.0e-3
    factor = (1.0 - (coordinates[:, 0] / _BACKGROUND_HALF_WIDTH) ** 2) * (
        1.0 - (coordinates[:, 1] / _BACKGROUND_HALF_WIDTH) ** 2
    )
    target = coordinates + (scale * factor[:, None]) * direction[None, :]
    fixed = np.asarray(mesh.topology.entities(0).subset("boundary").mask, dtype=np.bool_)
    target[fixed] = coordinates[fixed]
    if not np.any(np.any(target != coordinates, axis=1) & ~fixed):
        raise RuntimeError(
            "The native background has no free vertex for its accepted rezone."
        )
    return _motion_advance(
        part,
        target,
        fixed,
        f"native-overset-{profile.name}-background-rezone-{step}",
    )


def _triangle_measures(
    result: phx.meshing.CellMeshingResult, /
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    cells = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    triangles = points[cells]
    measures = 0.5 * np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    return np.mean(triangles, axis=1), measures


def _manufactured_solution(points: Array, /) -> Array:
    points = jnp.asarray(points)
    x, y = points[..., 0], points[..., 1]
    return 5.0 + x + y - (x * x + y * y) / 4.0


def _source_center(source: NativePlanarSource, /) -> np.ndarray:
    """Derive the current square's center from its authored exterior loop."""
    region = source.region
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    edges = np.asarray(region.edges, dtype=np.int64)
    vertices = np.asarray(region.vertices, dtype=np.float64)
    return np.mean(vertices[edges[offsets[0] : offsets[1], 0]], axis=0)


def _source_cavity(source: NativePlanarSource, /) -> np.ndarray:
    region = source.region
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    edges = np.asarray(region.edges, dtype=np.int64)
    if offsets.shape != (3,):
        raise ValueError("Moving body source must retain its one authored cavity loop.")
    rows = edges[offsets[1] : offsets[2], 0]
    return np.asarray(region.vertices, dtype=np.float64)[rows]


def _cavity_polygons(
    profile: _Profile,
    connectivity: phx.meshing.OversetConnectivity,
    sources: dict[str, NativePlanarSource],
    /,
) -> tuple[np.ndarray, ...]:
    polygons = []
    for body in profile.bodies:
        part = connectivity.assembly.part(body.name)
        source = sources[body.name]
        specification = next(
            spec for spec in connectivity.specs if spec.part_name == body.name
        )
        if specification.wall is None:
            raise ValueError(f"Missing protected wall for {body.name!r}.")
        associated = _source_edge_ids(part, source, (4, 5, 6, 7))
        if not np.array_equal(
            np.sort(np.asarray(specification.wall.entity_ids)), associated
        ):
            raise ValueError(
                f"Protected wall is not bound to the current source for {body.name!r}."
            )
        polygons.append(_source_cavity(source))
    return tuple(polygons)


def _boundary_flux_oracle(
    profile: _Profile,
    cavities: tuple[np.ndarray, ...],
    boundary_fluxes: tuple[tuple[AtomicEdge, float], ...],
    /,
) -> dict[str, object]:
    """Compare each actual boundary group's numerical flux to exact ∇u·n."""
    extent = Fraction.from_float(_BACKGROUND_HALF_WIDTH)
    boxes: list[tuple[str, tuple[Fraction, Fraction], tuple[Fraction, Fraction]]] = [
        ("exterior", (-extent, -extent), (extent, extent))
    ]
    for body, polygon in zip(profile.bodies, cavities, strict=True):
        lower = np.min(polygon, axis=0)
        upper = np.max(polygon, axis=0)
        boxes.append(
            (
                body.name,
                (
                    Fraction.from_float(float(lower[0])),
                    Fraction.from_float(float(lower[1])),
                ),
                (
                    Fraction.from_float(float(upper[0])),
                    Fraction.from_float(float(upper[1])),
                ),
            )
        )
    numerical = {name: 0.0 for name, _, _ in boxes}
    exact = {name: Fraction(0) for name, _, _ in boxes}
    counts = {name: 0 for name, _, _ in boxes}
    for edge, value in boundary_fluxes:
        matches = []
        for name, lower, upper in boxes:
            if any(
                edge.start[axis] == edge.end[axis] == bound
                and lower[1 - axis] <= edge.start[1 - axis] <= upper[1 - axis]
                and lower[1 - axis] <= edge.end[1 - axis] <= upper[1 - axis]
                for axis in (0, 1)
                for bound in (lower[axis], upper[axis])
            ):
                matches.append(name)
        if len(matches) != 1 or len(edge.owners) != 1:
            raise RuntimeError("Physical edge has ambiguous exterior/cavity group.")
        name = matches[0]
        midpoint = tuple((a + b) / 2 for a, b in zip(edge.start, edge.end, strict=True))
        normal = tuple(
            component * edge.owners[0].orientation for component in edge.normal
        )
        exact[name] += (1 - midpoint[0] / 2) * normal[0] + (1 - midpoint[1] / 2) * normal[
            1
        ]
        numerical[name] += value
        counts[name] += 1
    if any(count == 0 for count in counts.values()):
        raise RuntimeError("Physical boundary group has no integrated edges.")
    errors = {name: numerical[name] - float(exact[name]) for name in numerical}
    return {
        "integration": "exact rational midpoint integral of manufactured gradient on actual atomic edges",
        "groups": {
            name: {
                "numerical_outward_flux": numerical[name],
                "manufactured_outward_flux": float(exact[name]),
                "signed_error": errors[name],
                "edge_count": counts[name],
            }
            for name in numerical
        },
        "group_flux_error_l2": float(np.linalg.norm(list(errors.values()))),
    }


def _composite_pde_oracle(
    profile: _Profile,
    centers: dict[str, np.ndarray],
    partition: OversetPartition,
    parts: tuple[phx.solver.MovingOversetPoissonPart, ...],
    /,
) -> dict[str, object]:
    """Integrate the P1 solution and quartic squared error over owned polygons."""
    # This six-point triangle rule integrates polynomials through degree four.
    barycentric = np.asarray(
        (
            (0.445948490915965, 0.445948490915965, 0.108103018168070),
            (0.445948490915965, 0.108103018168070, 0.445948490915965),
            (0.108103018168070, 0.445948490915965, 0.445948490915965),
            (0.091576213509771, 0.091576213509771, 0.816847572980459),
            (0.091576213509771, 0.816847572980459, 0.091576213509771),
            (0.816847572980459, 0.091576213509771, 0.091576213509771),
        ),
        dtype=np.float64,
    )
    weights = np.asarray(
        (0.223381589678011,) * 3 + (0.109951743655322,) * 3,
        dtype=np.float64,
    )
    by_name = {part.part_name: part for part in parts}
    arrays = {
        part.part_name: (
            np.asarray(part.discretization.mesh.coordinates, dtype=np.float64),
            np.asarray(part.discretization.mesh.blocks[0].vertices, dtype=np.int64),
            np.asarray(part.discretization.dof_maps[0].cell_dofs[0], dtype=np.int64),
            np.asarray(part.coefficients, dtype=np.float64),
        )
        for part in parts
    }
    piece_counts = {name: 0 for name in by_name}
    numerical_integral = 0.0
    exact_integral_quadrature = 0.0
    squared_error = 0.0
    physical_area = 0.0
    for piece in partition.pieces:
        mesh_points, cells, cell_dofs, field_values = arrays[piece.part_name]
        source = mesh_points[cells[piece.cell_index]]
        coefficients = field_values[cell_dofs[piece.cell_index]]
        matrix = np.column_stack((source[1] - source[0], source[2] - source[0]))
        polygon = np.asarray(piece.vertices, dtype=np.float64)
        piece_counts[piece.part_name] += 1
        for index in range(1, polygon.shape[0] - 1):
            triangle = polygon[[0, index, index + 1]]
            edges = triangle[1:] - triangle[0]
            area = abs(float(edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0])) / 2.0
            if area == 0.0:
                continue
            points = barycentric @ triangle
            coordinates = np.linalg.solve(matrix, (points - source[0]).T).T
            values = (
                (1.0 - coordinates[:, 0] - coordinates[:, 1]) * coefficients[0]
                + coordinates[:, 0] * coefficients[1]
                + coordinates[:, 1] * coefficients[2]
            )
            x, y = points[:, 0], points[:, 1]
            exact = 5.0 + x + y - (x * x + y * y) / 4.0
            numerical_integral += area * float(weights @ values)
            exact_integral_quadrature += area * float(weights @ exact)
            squared_error += area * float(weights @ ((values - exact) ** 2))
            physical_area += area
    expected_area = 4.0 * _BACKGROUND_HALF_WIDTH**2
    expected_integral = expected_area * (5.0 - _BACKGROUND_HALF_WIDTH**2 / 6.0)
    for body in profile.bodies:
        x, y = centers[body.name]
        cavity_area = 4.0 * body.cavity_half_width**2
        expected_area -= cavity_area
        expected_integral -= cavity_area * (
            5.0 + x + y - (x * x + y * y) / 4.0 - body.cavity_half_width**2 / 6.0
        )
    if (
        abs(physical_area - expected_area) > 1.0e-9
        or abs(exact_integral_quadrature - expected_integral) > 1.0e-9
        or any(count == 0 for count in piece_counts.values())
    ):
        raise RuntimeError(
            "Composite physical partition failed independent area/QoI checks."
        )
    return {
        "support": "priority-owned physical polygons, cavities excluded",
        "integration": "degree-four six-point triangle quadrature on polygon fans",
        "physical_area": physical_area,
        "independent_analytic_area": expected_area,
        "per_part_owned_polygon_count": piece_counts,
        "manufactured_l2_error": float(np.sqrt(max(squared_error, 0.0))),
        "qoi_integral": numerical_integral,
        "independent_exact_qoi_integral": expected_integral,
        "qoi_absolute_error": abs(numerical_integral - expected_integral),
        "manufactured_quadrature_integral": exact_integral_quadrature,
    }


def _initial_pde_entries(
    connectivity: phx.meshing.OversetConnectivity, /
) -> tuple[dict[str, FiniteElementPlan], tuple[CompositionEntry, ...]]:
    """Bind each original mesh's P1 space and state to the accepted assembly."""
    part_entries = {entry.entry_id: entry for entry in connectivity.composition_entries()}
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    plans: dict[str, FiniteElementPlan] = {}
    entries: list[CompositionEntry] = []
    for part in connectivity.assembly.parts:
        result = _part_result(part)
        if not isinstance(result.geometry, CellGeometrySpec):
            raise TypeError("Composite Poisson requires affine cell geometry.")
        plan = FiniteElementPlan(result.mesh, field, coordinate_spec=result.geometry)
        discretization = plan.prepare()
        dof_map = discretization.dof_maps[0]
        fe_entry = CompositionEntry(
            discretization,
            entry_id=f"fe:{part.name}",
            role="discretization",
            owner_id="native-moving-overset-matrix",
            structure_id=dof_map.dof_map_id,
            revision_id=discretization.prepared_id,
            semantics_id="scalar-P1-Poisson-space",
            dependencies=(part_entries[f"overset:part:{part.name}"].binding("revision"),),
        )
        values = jnp.zeros((dof_map.global_dof_count,), dtype=jnp.float64)
        state_entry = CompositionEntry(
            values,
            entry_id=f"poisson:{part.name}",
            role="physical-state",
            owner_id="native-moving-overset-matrix",
            structure_id=dof_map.dof_map_id,
            revision_id=canonical_fingerprint(
                {
                    "kind": "native-moving-overset-poisson-initial-state",
                    "part": part.part_id,
                    "space": discretization.prepared_id,
                }
            ),
            semantics_id="scalar-Poisson-potential",
            dependencies=(fe_entry.binding("revision"),),
        )
        plans[part.name] = plan
        entries.extend((fe_entry, state_entry))
    return plans, tuple(entries)


def _initial_inventory_entry(
    connectivity: phx.meshing.OversetConnectivity,
    background: MeshPart,
    /,
) -> tuple[CompositionEntry, np.ndarray]:
    background_result = _part_result(background)
    centers, measures = _triangle_measures(background_result)
    intensive = 2.0 + centers[:, 0] / 8.0 - centers[:, 1] / 10.0
    content = (measures * intensive)[:, None]
    total = np.sum(content, axis=0)
    if abs(float(total[0]) - 72.0) > _CONSERVATIVE_CONTENT_TOLERANCE:
        raise RuntimeError(
            "Native background inventory failed its exact square integral."
        )
    background_registration = {
        entry.entry_id: entry for entry in connectivity.composition_entries()
    }[f"overset:part:{_BACKGROUND}"]
    entry = CompositionEntry(
        jnp.asarray(content),
        entry_id=_CONTENT_ENTRY_ID,
        role="physical-state",
        owner_id="native-moving-overset-matrix",
        structure_id=background_result.mesh.topology_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "native-moving-overset-background-content",
                "result": background_result.result_id,
                "content": content.tolist(),
            }
        ),
        semantics_id="affine-density-extensive-cell-content",
        dependencies=(background_registration.binding("revision"),),
    )
    return entry, content


def _conservative_epoch_transport(
    connectivity: phx.meshing.OversetConnectivity,
    background_advance: phx.meshing.MeshMotionAdvance,
    composition: Composition,
    /,
) -> tuple[CompositionTransport, dict[str, object]]:
    source_part = connectivity.assembly.part(_BACKGROUND)
    target_result = _motion_result(background_advance)
    target_part = MeshPart(
        _BACKGROUND, target_result, coordinate_contract=source_part.coordinate_contract
    )
    source_result = _part_result(source_part)
    remap = phx.meshing.prepare_overset_conservative_remap(source_part, target_part)
    if not remap.complete:
        raise RuntimeError(
            "Moving native background common-refinement coverage is incomplete."
        )
    source_entry = composition.entry(_CONTENT_ENTRY_ID)
    source_content = np.asarray(source_entry.value, dtype=np.float64)
    target_content = np.asarray(
        remap.apply_extensive(jnp.asarray(source_content)), dtype=np.float64
    )
    source_total = np.sum(source_content, axis=0)
    target_total = np.sum(target_content, axis=0)
    centers, measures = _triangle_measures(target_result)
    independent_exact_total = float(
        np.sum(measures * (2.0 + centers[:, 0] / 8.0 - centers[:, 1] / 10.0))
    )
    content_defect = float(np.max(np.abs(target_total - source_total), initial=0.0))
    if (
        abs(float(source_total[0]) - 72.0) > _CONSERVATIVE_CONTENT_TOLERANCE
        or abs(float(target_total[0]) - 72.0) > _CONSERVATIVE_CONTENT_TOLERANCE
        or abs(independent_exact_total - 72.0) > _CONSERVATIVE_CONTENT_TOLERANCE
        or content_defect > _CONSERVATIVE_CONTENT_TOLERANCE
    ):
        raise RuntimeError(
            f"Moving conservative overlap failed its independent content oracle: "
            f"{source_total=}, {target_total=}, {independent_exact_total=}, {content_defect=}."
        )
    source_registration = composition.entry(f"overset:part:{_BACKGROUND}")
    target_registration = CompositionEntry(
        target_part,
        entry_id=source_registration.entry_id,
        role=source_registration.role,
        owner_id=source_registration.owner_id,
        structure_id=target_result.mesh.topology_id,
        revision_id=target_part.part_id,
        semantics_id=source_registration.semantics_id,
        dependencies=source_registration.dependencies,
    )
    target_entry = CompositionEntry(
        jnp.asarray(target_content),
        entry_id=_CONTENT_ENTRY_ID,
        role="physical-state",
        owner_id="native-moving-overset-matrix",
        structure_id=target_result.mesh.topology_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "native-moving-overset-background-content",
                "result": target_result.result_id,
                "content": target_content.tolist(),
            }
        ),
        semantics_id="affine-density-extensive-cell-content",
        dependencies=(target_registration.binding("revision"),),
    )
    transport = CompositionTransport(
        "physical-remap",
        (_CONTENT_ENTRY_ID,),
        (target_entry,),
        source_structure_ids=(source_entry.structure_id,),
        route_id=remap.transfer_id,
        successful=jnp.asarray(remap.complete),
        source_content=jnp.asarray(source_total),
        target_content=jnp.asarray(target_total),
        content_tolerance=jnp.full(source_total.shape, _CONSERVATIVE_CONTENT_TOLERANCE),
    )
    return transport, {
        "remap_id": remap.transfer_id,
        "complete": remap.complete,
        "source_mesh_result_id": source_result.result_id,
        "target_mesh_result_id": target_result.result_id,
        "source_extensive_content": source_total.tolist(),
        "target_extensive_content": target_total.tolist(),
        "independent_exact_target_content": independent_exact_total,
        "maximum_content_defect": content_defect,
        "tolerance_absolute": _CONSERVATIVE_CONTENT_TOLERANCE,
        "staged_in_same_accepted_epoch_rebind": True,
    }


def _accepted_orphan_refusal(
    profile: _Profile,
    connectivity: phx.meshing.OversetConnectivity,
    composition: Composition,
    plans: dict[str, FiniteElementPlan],
    sources: dict[str, NativePlanarSource],
    /,
) -> dict[str, object]:
    body = profile.bodies[0]
    part = connectivity.assembly.part(body.name)
    mesh = _part_result(part).mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    orphan_motion = _motion_advance(
        part,
        coordinates + np.asarray((8.0, 0.0), dtype=np.float64),
        np.zeros((coordinates.shape[0],), dtype=np.bool_),
        f"native-overset-{profile.name}-orphan-proposal",
        source_transition=NativePlanarMotionSource(
            sources[body.name], np.asarray((8.0, 0.0), dtype=np.float64)
        ),
    )
    candidate = connectivity.moved_from_advances({body.name: orphan_motion})
    orphan_sources = dict(sources)
    if orphan_motion.source_transition is None:
        raise RuntimeError("Orphan proposal omitted its successor source authority.")
    orphan_sources[body.name] = orphan_motion.source_transition.successor
    composition_id = composition.composition_id
    source_records = tuple(entry.record_id for entry in composition.entries)
    try:
        phx.solver.prepare_moving_overset_poisson_update(
            composition,
            connectivity,
            {body.name: orphan_motion},
            plans,
            "u",
            boundary_values=_manufactured_solution,
            cavities=_cavity_polygons(profile, candidate, orphan_sources),
            part_priority=(*(item.name for item in profile.bodies), _BACKGROUND),
        )
    except phx.meshing.OversetConnectivityError:
        rejected = True
    else:
        rejected = False
    unchanged = (
        composition.composition_id == composition_id
        and tuple(entry.record_id for entry in composition.entries) == source_records
    )
    if candidate.complete or candidate.orphan_count <= 0 or not rejected or not unchanged:
        raise RuntimeError(
            "An orphaned native overset epoch failed to roll back atomically."
        )
    return {
        "motion_accepted_source_bound": orphan_motion.accepted,
        "candidate_connectivity_id": candidate.connectivity_id,
        "candidate_complete": candidate.complete,
        "orphan_count": candidate.orphan_count,
        "publication_rejected": rejected,
        "composition_identity_unchanged": unchanged,
        "composition_id": composition.composition_id,
    }


def _run_profile(
    profile: _Profile, /, *, resolution_scale: int = 1, checkpoint: str | None = None
) -> dict[str, object]:
    parts: dict[str, MeshPart] = {}
    sources: dict[str, NativePlanarSource] = {}
    generation: dict[str, dict[str, object]] = {}
    parts[_BACKGROUND], sources[_BACKGROUND], _, generation[_BACKGROUND] = _native_part(
        _BACKGROUND,
        _square_region(_BACKGROUND_HALF_WIDTH, _BACKGROUND),
        body=None,
        resolution_scale=resolution_scale,
    )
    analytic_measures: dict[str, object] = {
        _BACKGROUND: _part_source_measures(parts[_BACKGROUND], None, sources[_BACKGROUND])
    }
    for body in profile.bodies:
        parts[body.name], sources[body.name], _, generation[body.name] = _native_part(
            body.name,
            _annulus_region(body),
            body=body,
            resolution_scale=resolution_scale,
        )
        analytic_measures[body.name] = _part_source_measures(
            parts[body.name], body, sources[body.name]
        )
    profile_record: dict[str, object] = {
        "profile": profile.name,
        "source_profile_id": {
            "routine": "native-overset-lifecycle",
            "combined": "native-overset-lifecycle:combined",
            "larger": "native-overset-lifecycle:larger",
        }[profile.name],
        "profile_controls": {
            "resolution_scale": resolution_scale,
            "resolution": _RESOLUTION,
            "minimum_adaptation_rounds": 0,
            "overset_profile": profile.name,
            "background_half_width": _BACKGROUND_HALF_WIDTH,
            **profile.record(),
        },
        "source_provenance": {
            "authority": "native-planar-source-with-stable-source-revision-and-entity-IDs",
            "donor_case_registry": "tools/_meshing_cases.py::CORPUS",
            "donor_source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
            "provider": "phydrax-native/planar_constrained_delaunay",
            "coordinate_contract": "SI",
            "resource_quotas_used": False,
            "optional_engines_used": False,
        },
        "native_generation": generation,
        "independent_source_oracle": analytic_measures,
        "tolerances": {
            "interpolation_absolute": _INTERPOLATION_TOLERANCE,
            "orphan_count_on_accepted_epoch": _ORPHAN_COUNT_TOLERANCE,
            "conservative_content_absolute": _CONSERVATIVE_CONTENT_TOLERANCE,
            "pde_normalized_residual": _PDE_RESIDUAL_TOLERANCE,
            "flux_balance_relative": _FLUX_BALANCE_RELATIVE_TOLERANCE,
        },
        "epochs": [],
        "status": "failed",
    }
    connectivity = _make_connectivity(profile, parts, sources)
    if not connectivity.complete or not connectivity.donor_search_complete:
        raise RuntimeError("Native overset wall cutting or donor search was incomplete.")
    centers = {body.name: _source_center(sources[body.name]) for body in profile.bodies}
    independent_blanking = _independent_blanking(connectivity, profile, sources)
    donor_oracle = _independent_donor_oracle(connectivity)
    field_oracle = _field_transfer(connectivity)
    initial_background = parts[_BACKGROUND]
    inventory_entry, initial_content = _initial_inventory_entry(
        connectivity, initial_background
    )
    plans, pde_entries = _initial_pde_entries(connectivity)
    composition = Composition(
        (*connectivity.composition_entries(), inventory_entry, *pde_entries),
        boundary_id=f"native-moving-overset-{profile.name}-accepted-epoch-0",
    )
    profile_record["initial_registration"] = {
        "connectivity_id": connectivity.connectivity_id,
        "epoch": connectivity.epoch,
        "complete": connectivity.complete,
        "part_count": len(connectivity.assembly.parts),
        "independent_blanking": independent_blanking,
        "independent_donor_oracle": donor_oracle,
        "field_transfer": field_oracle,
        "initial_composition_id": composition.composition_id,
        "initial_background_content": float(np.sum(initial_content)),
    }
    return _continue_profile(
        profile,
        profile_record,
        connectivity,
        centers,
        sources,
        composition,
        plans,
        checkpoint=checkpoint,
    )


def _continue_profile(
    profile: _Profile,
    profile_record: dict[str, object],
    connectivity: phx.meshing.OversetConnectivity,
    centers: dict[str, np.ndarray],
    sources: dict[str, NativePlanarSource],
    composition: Composition,
    plans: dict[str, FiniteElementPlan],
    /,
    *,
    start_step: int = 0,
    checkpoint: str | None = None,
) -> dict[str, object]:
    epochs = profile_record["epochs"]
    if not isinstance(epochs, list):
        raise ValueError("Profile epochs must be a list.")
    for step in range(start_step, 2):
        advances: dict[str, phx.meshing.MeshMotionAdvance] = {}
        motion_records: dict[str, object] = {}
        for body in profile.bodies:
            predecessor = connectivity.assembly.part(body.name)
            advance = _rigid_body_motion(predecessor, body, sources[body.name], step)
            advances[body.name] = advance
            transition = advance.source_transition
            if transition is None:
                raise RuntimeError("Body motion omitted its successor source authority.")
            centers[body.name] = _source_center(transition.successor)
            motion_records[body.name] = {
                "accepted": advance.accepted,
                "advance_id": advance.advance_id,
                "source_mesh_id": advance.source_mesh_id,
                "source_result_id": advance.source_result_id,
                "target_result_id": _motion_result(advance).result_id,
                "rigid_translation": body.offset(step).tolist(),
                "source_bound": True,
            }
        background = connectivity.assembly.part(_BACKGROUND)
        background_motion = _background_rezone(background, profile, step)
        advances[_BACKGROUND] = background_motion
        motion_records[_BACKGROUND] = {
            "accepted": background_motion.accepted,
            "advance_id": background_motion.advance_id,
            "source_mesh_id": background_motion.source_mesh_id,
            "source_result_id": background_motion.source_result_id,
            "target_result_id": _motion_result(background_motion).result_id,
            "motion": "fixed-exterior-interior-rezone",
            "source_bound": True,
        }
        candidate_sources = dict(sources)
        for body in profile.bodies:
            transition = advances[body.name].source_transition
            if transition is None:
                raise RuntimeError("Body motion omitted its successor source authority.")
            candidate_sources[body.name] = transition.successor
        transport, conservative = _conservative_epoch_transport(
            connectivity,
            background_motion,
            composition,
        )
        predecessor_composition = composition
        predecessor_composition_id = composition.composition_id
        cavities = tuple(
            _source_cavity(candidate_sources[body.name]) for body in profile.bodies
        )
        staged = phx.solver.prepare_moving_overset_poisson_update(
            predecessor_composition,
            connectivity,
            advances,
            plans,
            "u",
            boundary_values=_manufactured_solution,
            cavities=cavities,
            part_priority=(*(body.name for body in profile.bodies), _BACKGROUND),
            solve_tolerance=_PDE_RESIDUAL_TOLERANCE,
            flux_tolerance=_FLUX_BALANCE_RELATIVE_TOLERANCE,
            transports=(transport,),
        )
        candidate = staged.candidate
        if not candidate.complete or candidate.orphan_count != _ORPHAN_COUNT_TOLERANCE:
            raise RuntimeError(
                "Accepted rigid-motion profile epoch lost receptor coverage: "
                f"{candidate.orphan_count=}."
            )
        if (
            _part_result(candidate.assembly.part(_BACKGROUND)).result_id
            != _motion_result(background_motion).result_id
        ):
            raise RuntimeError(
                "Conservative overlap endpoint is not the accepted background motion."
            )
        target_registration = next(
            entry
            for entry in candidate.composition_entries()
            if entry.entry_id == f"overset:part:{_BACKGROUND}"
        )
        if transport.targets[0].dependencies != (
            target_registration.binding("revision"),
        ):
            raise RuntimeError(
                "Conservative transport does not bind the candidate registration."
            )
        associated_cavities = _cavity_polygons(profile, candidate, candidate_sources)
        if (
            len(cavities) != len(candidate.partition_cavities)
            or any(
                not np.array_equal(actual, bound)
                for actual, bound in zip(
                    cavities, candidate.partition_cavities, strict=True
                )
            )
            or any(
                not np.array_equal(actual, associated)
                for actual, associated in zip(cavities, associated_cavities, strict=True)
            )
        ):
            raise RuntimeError(
                "Candidate partition cavities differ from the current source walls."
            )
        conservative["accepted_connectivity_id"] = candidate.connectivity_id
        independent_blanking = _independent_blanking(
            candidate, profile, candidate_sources
        )
        donor_oracle = _independent_donor_oracle(candidate)
        field_oracle = _field_transfer(candidate)
        if not staged.accepted or staged.partition is None or staged.composite is None:
            raise RuntimeError(
                "Composite Poisson reanalysis did not produce accepted evidence."
            )
        if staged.partition is not staged.candidate.bound_partition:
            raise RuntimeError(
                "Composite solve did not use the registered physical partition."
            )
        pde_oracle = _composite_pde_oracle(
            profile, centers, staged.partition, staged.parts
        )
        pde = staged.composite
        boundary_oracle = _boundary_flux_oracle(
            profile, cavities, pde.physical_boundary_fluxes
        )
        expected_area = 4.0 * _BACKGROUND_HALF_WIDTH**2 - sum(
            4.0 * body.cavity_half_width**2 for body in profile.bodies
        )
        global_flux_defect = abs(pde.boundary_flux + expected_area) / expected_area
        if (
            pde.normalized_residual > _PDE_RESIDUAL_TOLERANCE
            or global_flux_defect > _FLUX_BALANCE_RELATIVE_TOLERANCE
        ):
            raise RuntimeError(
                "Composite Poisson residual or physical flux oracle failed."
            )
        refused = staged.commit(accepted_boundary=False)
        if refused.published or refused.composition is not predecessor_composition:
            raise RuntimeError(
                "Rejected overset composition boundary was not rolled back."
            )
        accepted = staged.commit(accepted_boundary=True)
        if not accepted.published or accepted.composition is predecessor_composition:
            raise RuntimeError(
                "Accepted overset motion did not publish its candidate composition."
            )
        composition = accepted.composition
        sources = candidate_sources
        epochs.append(
            {
                "epoch": candidate.epoch,
                "predecessor_connectivity_id": connectivity.connectivity_id,
                "connectivity_id": candidate.connectivity_id,
                "registration_complete": candidate.complete,
                "orphan_count": candidate.orphan_count,
                "motion_advances": motion_records,
                "independent_blanking": independent_blanking,
                "independent_donor_oracle": donor_oracle,
                "field_transfer": field_oracle,
                "conservative_overlap_transport": conservative,
                "composite_pde": {
                    "status": "passed",
                    "equation": "u=5+x+y-(x^2+y^2)/4, -Laplace(u)=1",
                    "physical_dirichlet": "weak symmetric Nitsche on background exterior and every protected cavity wall",
                    "operator": "one sparse coupled P1 finite-element system with symmetric Nitsche interfaces and cut-dependent coercive penalty",
                    "priority": [*(body.name for body in profile.bodies), _BACKGROUND],
                    "overlap_semantics": "priority-owned physical partition; each point integrated once",
                    "gauge": "none; physical Dirichlet boundary anchors the solution",
                    "native_linear_status": np.asarray(pde.solve_result.status).tolist(),
                    "native_linear_successful": bool(
                        jnp.all(pde.solve_result.successful)
                    ),
                    "absolute_linear_residual": pde.linear_residual,
                    "normalized_residual": pde.normalized_residual,
                    "physical_boundary_flux": pde.boundary_flux,
                    "physical_flux_definition": "outward symmetric-Nitsche numerical flux, integral of grad(u_h).n - gamma*(u_h-g)",
                    "raw_gradient_flux_by_part": dict(pde.raw_gradient_flux_by_name),
                    "source_content": pde.source_content,
                    "global_flux_balance_relative": global_flux_defect,
                    "independent_boundary_flux_oracle": boundary_oracle,
                    "independent_physical_oracle": pde_oracle,
                    "update_id": staged.update_id,
                },
                "predecessor_composition_id": predecessor_composition_id,
                "rejected_boundary_rollback": (
                    not refused.published
                    and refused.composition is predecessor_composition
                ),
                "accepted_rebind_published": accepted.published,
                "accepted_composition_id": composition.composition_id,
            }
        )
        for name, old_plan in plans.items():
            result = _part_result(candidate.assembly.part(name))
            plans[name] = FiniteElementPlan(
                result.mesh,
                old_plan.fields,
                precision_policy=old_plan.precision_policy,
                coordinate_spec=result.geometry,
                coefficient_dtype=old_plan.coefficient_dtype,
                coordinate_measure=old_plan.coordinate_measure,
            )
        connectivity = candidate
        if checkpoint is not None and step == 0:
            profile_record["cold_checkpoint"] = _write_cold_checkpoint(
                checkpoint, connectivity, composition, plans, sources
            )
            return profile_record
    orphan = _accepted_orphan_refusal(profile, connectivity, composition, plans, sources)
    profile_record["orphan_failure"] = orphan
    profile_record["status"] = "passed"
    return profile_record


def _require_refinement_convergence(
    original: dict[str, object], refined: dict[str, object], /
) -> list[dict[str, object]]:
    """Require decreasing physical errors at both identical motion epochs."""
    coarse_epochs, fine_epochs = original.get("epochs"), refined.get("epochs")
    if not isinstance(coarse_epochs, list) or not isinstance(fine_epochs, list):
        raise RuntimeError("Missing original or refined composite motion epochs.")
    if len(coarse_epochs) != 2 or len(fine_epochs) != 2:
        raise RuntimeError("Composite refinement must cover both authored motion epochs.")
    comparisons: list[dict[str, object]] = []
    for coarse, fine in zip(coarse_epochs, fine_epochs, strict=True):
        if not isinstance(coarse, dict) or not isinstance(fine, dict):
            raise TypeError("Composite epoch evidence must be a mapping.")
        coarse_pde, fine_pde = coarse.get("composite_pde"), fine.get("composite_pde")
        if not isinstance(coarse_pde, dict) or not isinstance(fine_pde, dict):
            raise RuntimeError("Composite epoch omitted its PDE evidence.")
        coarse_physical = coarse_pde.get("independent_physical_oracle")
        fine_physical = fine_pde.get("independent_physical_oracle")
        coarse_flux = coarse_pde.get("independent_boundary_flux_oracle")
        fine_flux = fine_pde.get("independent_boundary_flux_oracle")
        if not all(
            isinstance(value, dict)
            for value in (coarse_physical, fine_physical, coarse_flux, fine_flux)
        ):
            raise RuntimeError("Composite epoch omitted its independent physical oracle.")
        assert isinstance(coarse_physical, dict) and isinstance(fine_physical, dict)
        assert isinstance(coarse_flux, dict) and isinstance(fine_flux, dict)
        entries = (
            ("manufactured_l2_error", coarse_physical, fine_physical),
            ("qoi_absolute_error", coarse_physical, fine_physical),
            ("group_flux_error_l2", coarse_flux, fine_flux),
        )
        metrics: dict[str, object] = {}
        for key, coarse_evidence, fine_evidence in entries:
            first, second = coarse_evidence.get(key), fine_evidence.get(key)
            if not isinstance(first, (float, int)) or not isinstance(
                second, (float, int)
            ):
                raise TypeError(f"Missing numeric convergence evidence {key!r}.")
            if (
                not np.isfinite(first)
                or not np.isfinite(second)
                or first < 0
                or second < 0
            ):
                raise RuntimeError(f"Invalid convergence evidence {key!r}.")
            passed = bool(second < first)
            metrics[key] = {"original": first, "refined": second, "decreased": passed}
            if not passed:
                raise RuntimeError(
                    f"Composite refinement did not decrease {key!r} at epoch "
                    f"{coarse.get('epoch')!r}: original={first!r}, refined={second!r}."
                )
        comparisons.append({"epoch": coarse.get("epoch"), "metrics": metrics})
    return comparisons


def _entry_with_value(entry: CompositionEntry, value: object, /) -> CompositionEntry:
    return CompositionEntry(
        value,
        entry_id=entry.entry_id,
        role=entry.role,
        owner_id=entry.owner_id,
        structure_id=entry.structure_id,
        revision_id=entry.revision_id,
        semantics_id=entry.semantics_id,
        dependencies=entry.dependencies,
    )


def _write_cold_checkpoint(
    path: str,
    connectivity: phx.meshing.OversetConnectivity,
    composition: Composition,
    plans: dict[str, FiniteElementPlan],
    sources: dict[str, NativePlanarSource],
    /,
) -> dict[str, object]:
    from phydrax.meshing._model_codec import (
        finite_element_field_banks_recipe,
        model_archive_content_id,
        overset_registration_recipe,
        write_model_artifact,
    )
    from phydrax.solver._finite_element_schedule import (
        FiniteElementAcceptedFieldLayout,
        FiniteElementFieldTransferSemantics,
    )

    if any(
        sources[name].binding_id != source.binding_id
        for name, source in connectivity.partition_sources.items()
    ):
        raise ValueError("Checkpoint sources differ from accepted registration.")
    values: dict[str, object] = {
        "overset:connectivity": overset_registration_recipe(connectivity.registration())
    }
    for part in connectivity.assembly.parts:
        name = part.name
        values[f"overset:part:{name}"] = _part_result(part)
        layout = FiniteElementAcceptedFieldLayout(
            plans[name].fields[0],
            transfer_semantics=FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
        )
        values[f"fe:{name}"] = finite_element_field_banks_recipe(
            composition.value(f"fe:{name}"),
            (layout,),
            (composition.value(f"poisson:{name}"),),
        )
    projected = Composition(
        tuple(
            _entry_with_value(entry, values.get(entry.entry_id, entry.value))
            for entry in composition.entries
        ),
        boundary_id=composition.boundary_id,
    )
    if projected.composition_id != composition.composition_id:
        raise RuntimeError(
            "Checkpoint projection changed the accepted composition identity."
        )
    write_model_artifact(path, projected, limits=None)
    return {
        "archive_content_id": model_archive_content_id(path, limits=None),
        "composition_id": composition.composition_id,
        "connectivity_id": connectivity.connectivity_id,
        "partition_digest": connectivity.partition_digest,
        "epoch": connectivity.epoch,
    }


def _resume_cold_profile(
    profile: _Profile, record: dict[str, object], path: str, receipt: dict[str, object], /
) -> dict[str, object]:
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_finite_element_field_banks,
        restore_model_archive_value,
        restore_overset_registration,
    )

    if record.get("profile") != profile.name or record.get("cold_checkpoint") != receipt:
        raise ValueError("Cold producer record does not match its trusted receipt.")
    epochs = record.get("epochs")
    if (
        not isinstance(epochs, list)
        or len(epochs) != 1
        or epochs[0].get("accepted_composition_id") != receipt["composition_id"]
        or epochs[0].get("connectivity_id") != receipt["connectivity_id"]
    ):
        raise ValueError("Cold receipt does not bind the accepted predecessor epoch.")
    archive_content_id = receipt["archive_content_id"]
    if not isinstance(archive_content_id, str):
        raise ValueError("Cold receipt must contain an archive content identifier.")
    projected = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=None,
        expected_content_id=archive_content_id,
    )
    if projected.composition_id != receipt["composition_id"]:
        raise ValueError("Cold archive changed the accepted composition.")
    parts = tuple(
        MeshPart(name, projected.value(f"overset:part:{name}"))
        for name in (_BACKGROUND, *(body.name for body in profile.bodies))
    )
    registration = restore_overset_registration(
        projected.value("overset:connectivity"), parts
    )
    connectivity = registration.prepare()
    if (
        connectivity.connectivity_id != receipt["connectivity_id"]
        or connectivity.partition_digest != receipt["partition_digest"]
        or connectivity.epoch != receipt["epoch"]
    ):
        raise ValueError("Cold registration changed its accepted partition or epoch.")
    connectivity.require_complete()
    plans: dict[str, FiniteElementPlan] = {}
    replacements: dict[str, object] = {"overset:connectivity": connectivity}
    for part in connectivity.assembly.parts:
        name = part.name
        archived_result = projected.value(f"overset:part:{name}")
        result = _part_result(part)
        if archived_result.result_id != result.result_id:
            raise ValueError("Cold registration changed an accepted part result.")
        recipe = projected.value(f"fe:{name}")
        fields = tuple(layout.rebuild_field_spec() for layout in recipe.field_layouts)
        plan = FiniteElementPlan(result.mesh, fields, coordinate_spec=result.geometry)
        space = plan.prepare(numeric_version=f"overset-motion:{connectivity.epoch}")
        _, banks = restore_finite_element_field_banks(recipe, space)
        if not np.array_equal(
            np.asarray(banks[0]), np.asarray(projected.value(f"poisson:{name}"))
        ):
            raise ValueError("Cold field bank differs from the accepted physical state.")
        plans[name] = plan
        replacements[f"overset:part:{name}"] = part
        replacements[f"fe:{name}"] = space
    composition = Composition(
        tuple(
            _entry_with_value(entry, replacements.get(entry.entry_id, entry.value))
            for entry in projected.entries
        ),
        boundary_id=projected.boundary_id,
    )
    if composition.composition_id != receipt["composition_id"]:
        raise ValueError("Cold reconstruction changed the accepted composition identity.")
    centers = {
        body.name: _source_center(registration.partition_sources[body.name])
        for body in profile.bodies
    }
    result = _continue_profile(
        profile,
        record,
        connectivity,
        centers,
        dict(registration.partition_sources),
        composition,
        plans,
        start_step=1,
    )
    continued_epochs = result["epochs"]
    if not isinstance(continued_epochs, list) or not isinstance(
        continued_epochs[1], dict
    ):
        raise ValueError("Cold continuation omitted its epoch record.")
    continued = continued_epochs[1]
    if (
        continued["predecessor_composition_id"] != composition.composition_id
        or continued["predecessor_connectivity_id"] != connectivity.connectivity_id
    ):
        raise RuntimeError(
            "Cold continuation did not consume the restored accepted predecessor."
        )
    result["cold_continuation"] = {
        "archive_content_id": receipt["archive_content_id"],
        "restored_composition_id": composition.composition_id,
        "restored_connectivity_id": connectivity.connectivity_id,
        "restored_partition_digest": connectivity.partition_digest,
        "continued_epoch": continued["epoch"],
        "continued_composition_id": continued["accepted_composition_id"],
        "status": "passed",
    }
    return result


def _run_cold_profile(
    profile: _Profile, /, *, resolution_scale: int = 1
) -> dict[str, object]:
    """Publish the first accepted epoch, exit, then continue it in a new process."""
    from .cold_restore import _run_child

    index = _PROFILES.index(profile)
    with TemporaryDirectory(prefix="phydrax-moving-overset-cold-") as temporary:
        archive = str(Path(temporary) / "accepted.phx")
        record_path = str(Path(temporary) / "producer.json")
        producer = _run_child(
            f"""
import json, os
from pathlib import Path
from tools.meshing_qualification.scenarios.moving_overset_matrix import _PROFILES, _run_profile
record = _run_profile(_PROFILES[{index}], resolution_scale={resolution_scale}, checkpoint={archive!r})
Path({record_path!r}).write_text(json.dumps(record))
print(json.dumps({{"pid": os.getpid(), "checkpoint": record["cold_checkpoint"]}}))
""",
            environment=dict(os.environ),
        )
        consumer = _run_child(
            f"""
import json, os
from pathlib import Path
from tools.meshing_qualification.scenarios.moving_overset_matrix import _PROFILES, _resume_cold_profile
record = json.loads(Path({record_path!r}).read_text())
result = _resume_cold_profile(_PROFILES[{index}], record, {archive!r}, {producer["checkpoint"]!r})
result["cold_continuation"]["consumer_pid"] = os.getpid()
print(json.dumps(result))
""",
            environment=dict(os.environ),
        )
    cold = consumer.get("cold_continuation")
    if (
        not isinstance(cold, dict)
        or cold.get("status") != "passed"
        or not isinstance(cold.get("consumer_pid"), int)
        or not isinstance(producer.get("pid"), int)
        or cold["consumer_pid"] == producer["pid"]
    ):
        raise RuntimeError("Moving overset continuation did not use a fresh process.")
    cold["producer_pid"] = producer["pid"]
    cold["producer_process_exited_before_consumer"] = True
    return consumer


def native_moving_overset() -> dict[str, object]:
    """Run all authored moving-overset profiles and both motion epochs."""
    meshcore = require_meshcore()
    profiles: dict[str, object] = {}
    for profile in _PROFILES:
        original: dict[str, object] | None = None
        refined: dict[str, object] | None = None
        try:
            original = _run_cold_profile(profile)
            refined = _run_profile(profile, resolution_scale=2)
            convergence = _require_refinement_convergence(original, refined)
            original["refinement_verification"] = {
                "method": "same authored geometry and motion; target element size halved",
                "resolution_scale": 2,
                "epochs": convergence,
                "refined_profile": refined,
            }
            profiles[profile.name] = original
        except Exception as error:
            profiles[profile.name] = {
                "profile": profile.name,
                "source_profile_id": {
                    "routine": "native-overset-lifecycle",
                    "combined": "native-overset-lifecycle:combined",
                    "larger": "native-overset-lifecycle:larger",
                }[profile.name],
                "status": "failed",
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "detail": str(error),
                "original_profile": original,
                "refined_profile": refined,
                "profile_controls": {
                    "overset_profile": profile.name,
                    "resolution_scale": 1,
                    "resolution": _RESOLUTION,
                    "minimum_adaptation_rounds": 0,
                    "background_half_width": _BACKGROUND_HALF_WIDTH,
                    **profile.record(),
                },
                "attempted_input": {
                    "background": {
                        "shape": "native square",
                        "half_width": _BACKGROUND_HALF_WIDTH,
                    },
                    "bodies": [body.record() for body in profile.bodies],
                    "motion_steps": 2,
                },
                "source_provenance": {
                    "donor_case_registry": "tools/_meshing_cases.py::CORPUS",
                    "donor_source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
                    "provider": "phydrax-native/planar_constrained_delaunay",
                    "resource_quotas_used": False,
                    "optional_engines_used": False,
                },
                "tolerances": {
                    "interpolation_absolute": _INTERPOLATION_TOLERANCE,
                    "orphan_count_on_accepted_epoch": _ORPHAN_COUNT_TOLERANCE,
                    "conservative_content_absolute": _CONSERVATIVE_CONTENT_TOLERANCE,
                    "pde_normalized_residual": _PDE_RESIDUAL_TOLERANCE,
                    "flux_balance_relative": _FLUX_BALANCE_RELATIVE_TOLERANCE,
                },
                "partial_matrix_pass": False,
            }
    passed = all(
        isinstance(value, dict) and value.get("status") == "passed"
        for value in profiles.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "workflow": "native square-annulus/background generation -> source-bound motion -> independent hole/fringe/donor and affine transfer oracles -> coupled composite FE Poisson reanalysis -> atomic conservative overlap rebind",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#moving-overset",
        "donor_profile_registry": "tools/_meshing_cases.py::CORPUS",
        "donor_source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
        "profiles": profiles,
        "profile_matrix_passed": passed,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
