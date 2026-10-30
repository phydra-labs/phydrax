from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellGeometrySpec,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    PreparedFieldReconstruction,
    lagrange_element,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.lifecycle import (
    Composition,
    CompositionEntry,
    CompositionTransport,
)
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshQualityTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
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
    name: str,
    revision: str,
    dimension: int,
    scope_name: str,
    entity_indices: np.ndarray,
    /,
) -> MeshingScope:
    return MeshingScope(
        name,
        revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        scope_name,
        entity_indices,
    )


def _native_part(
    name: str,
    region: PlanarMeshRegion,
    /,
    *,
    body: _Body | None,
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
    source = NativePlanarSource(region, revision)
    fluid = _source_scope(
        name,
        revision,
        2,
        "overset-fluid-region",
        np.asarray((0,), dtype=np.int64),
    )
    exterior = _source_scope(
        name,
        revision,
        1,
        "overset-exterior",
        np.arange(4, dtype=np.int64),
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
            name,
            revision,
            1,
            "overset-cavity-wall",
            np.arange(4, 8, dtype=np.int64),
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
    size = 0.8 / (_RESOLUTION * refinement)
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
                "resolution_scale": 1,
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
    requested = {f"{source.source_revision}:edge:{edge}": edge for edge in edge_indices}
    selected: list[int] = []
    covered: set[int] = set()
    for association in carrier.associations:
        if (
            association.association_kind
            is not phx.meshing.GeometryAssociationKind.PIECEWISE_LINEAR
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
            if role is phx.meshing.GeometrySourceEntityRole.EDGE and entity in requested:
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
    )


def _wall_rectangle(center: np.ndarray, half: float, /) -> tuple[np.ndarray, np.ndarray]:
    return center - half, center + half


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


def _independent_blanking(
    connectivity: phx.meshing.OversetConnectivity,
    profile: _Profile,
    centers: dict[str, np.ndarray],
    /,
) -> dict[str, object]:
    parts = {part.name: part for part in connectivity.assembly.parts}
    specs = {spec.part_name: spec for spec in connectivity.specs}
    walls = {
        body.name: _wall_rectangle(centers[body.name], body.cavity_half_width)
        for body in profile.bodies
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
            cut |= np.all(
                (cell_upper >= edge_lower[None, :] - 1.0e-12)
                & (cell_lower <= edge_upper[None, :] + 1.0e-12),
                axis=1,
            )
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
        if np.any(np.asarray(actual.ambiguous_cells)) or np.any(
            np.asarray(actual.protected_conflict)
        ):
            raise RuntimeError(f"Native overset blanking is ambiguous for {name!r}.")
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
            "ambiguous_cells": 0,
            "protected_conflicts": 0,
        }
    return reports


def _independent_donor_oracle(
    connectivity: phx.meshing.OversetConnectivity,
    /,
) -> dict[str, object]:
    names = connectivity.assembly.global_part_names
    parts = {part.name: part for part in connectivity.assembly.parts}
    max_residual = 0.0
    routed = 0
    per_part: dict[str, object] = {}
    for evidence in connectivity.receptors:
        target = parts[evidence.part_name]
        target_result = _part_result(target)
        target_mesh = target_result.mesh
        target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
        target_rows = {int(identifier): row for row, identifier in enumerate(target_ids)}
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
            donor_mesh = _part_result(parts[donor_name]).mesh
            donor_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for block in donor_mesh.blocks:
                for row, identifier in enumerate(
                    np.asarray(block.global_ids, dtype=np.int64)
                ):
                    donor_rows[int(identifier)] = (
                        np.asarray(block.vertices[row], dtype=np.int64),
                        np.asarray(donor_mesh.coordinates, dtype=np.float64),
                    )
            if int(donor_cell_id) not in donor_rows:
                raise RuntimeError(
                    "Native donor route references an absent stable cell ID."
                )
            vertices, donor_points = donor_rows[int(donor_cell_id)]
            point = np.asarray(target_mesh.coordinates, dtype=np.float64)[
                target_rows[int(receptor_id)]
            ]
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
            if np.any(
                np.asarray(connectivity.blanking_of(donor_name).cell_status)
                == int(phx.meshing.OversetCellStatus.HOLE)
            ) and int(donor_cell_id) in connectivity.blanking_of(
                donor_name
            ).cell_ids_with(int(phx.meshing.OversetCellStatus.HOLE)):
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
    spaces: dict[str, FiniteElementDiscretization] = {}
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
        spaces[part.name] = space
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
        points = np.asarray(spaces[name].dof_maps[0].dof_coordinates, dtype=np.float64)
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
) -> phx.meshing.MeshMotionAdvance:
    source = _part_result(part)
    advance = phx.meshing.advance_mesh_motion(
        phx.meshing.MeshMotionMonitor(source.mesh),
        source,
        target_coordinates,
        boundary_residual=0.0,
        fixed_vertices=fixed_vertices,
        numeric_version=numeric_version,
    )
    if not advance.accepted:
        raise RuntimeError(f"Source-bound mesh motion was rejected for {part.name!r}.")
    target = advance.result
    if target is None:
        raise RuntimeError(f"Accepted mesh motion omitted its result for {part.name!r}.")
    if (
        advance.source_mesh_id != source.mesh.mesh_id
        or advance.source_result_id != source.result_id
        or target.mesh.topology_id != source.mesh.topology_id
    ):
        raise RuntimeError(f"Source-bound mesh motion changed identity for {part.name!r}.")
    return advance


def _rigid_body_motion(
    part: MeshPart,
    body: _Body,
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
    candidate: phx.meshing.OversetConnectivity,
    background_advance: phx.meshing.MeshMotionAdvance,
    composition: Composition,
    /,
) -> tuple[CompositionTransport, dict[str, object]]:
    source_part = connectivity.assembly.part(_BACKGROUND)
    target_part = candidate.assembly.part(_BACKGROUND)
    source_result = _part_result(source_part)
    target_result = _part_result(target_part)
    motion_result = _motion_result(background_advance)
    if target_result.result_id != motion_result.result_id:
        raise RuntimeError(
            "Conservative overlap endpoint is not the accepted background motion."
        )
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
    registrations = {entry.entry_id: entry for entry in candidate.composition_entries()}
    target_registration = registrations[f"overset:part:{_BACKGROUND}"]
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
        "accepted_connectivity_id": candidate.connectivity_id,
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
    )
    candidate = connectivity.moved_from_advances({body.name: orphan_motion})
    staged = phx.meshing.prepare_overset_motion_rebind(
        composition,
        connectivity,
        candidate,
    )
    receipt = staged.commit(accepted_boundary=True)
    if (
        candidate.complete
        or candidate.orphan_count <= 0
        or receipt.published
        or receipt.composition is not composition
    ):
        raise RuntimeError(
            "An orphaned native overset epoch failed to roll back atomically."
        )
    return {
        "motion_accepted_source_bound": orphan_motion.accepted,
        "candidate_connectivity_id": candidate.connectivity_id,
        "candidate_complete": candidate.complete,
        "orphan_count": candidate.orphan_count,
        "publication_rejected": not receipt.published,
        "composition_identity_unchanged": receipt.composition is composition,
        "composition_id": composition.composition_id,
    }


def _run_profile(profile: _Profile, /) -> dict[str, object]:
    parts: dict[str, MeshPart] = {}
    sources: dict[str, NativePlanarSource] = {}
    generation: dict[str, dict[str, object]] = {}
    parts[_BACKGROUND], sources[_BACKGROUND], _, generation[_BACKGROUND] = _native_part(
        _BACKGROUND,
        _square_region(_BACKGROUND_HALF_WIDTH, _BACKGROUND),
        body=None,
    )
    analytic_measures: dict[str, object] = {
        _BACKGROUND: _part_source_measures(parts[_BACKGROUND], None, sources[_BACKGROUND])
    }
    for body in profile.bodies:
        parts[body.name], sources[body.name], _, generation[body.name] = _native_part(
            body.name,
            _annulus_region(body),
            body=body,
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
            "resolution_scale": 1,
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
    centers = {
        body.name: np.asarray(body.center, dtype=np.float64) for body in profile.bodies
    }
    independent_blanking = _independent_blanking(connectivity, profile, centers)
    donor_oracle = _independent_donor_oracle(connectivity)
    field_oracle = _field_transfer(connectivity)
    initial_background = parts[_BACKGROUND]
    inventory_entry, initial_content = _initial_inventory_entry(
        connectivity, initial_background
    )
    composition = Composition(
        (*connectivity.composition_entries(), inventory_entry),
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
    for step in range(2):
        advances: dict[str, phx.meshing.MeshMotionAdvance] = {}
        motion_records: dict[str, object] = {}
        for body in profile.bodies:
            predecessor = connectivity.assembly.part(body.name)
            advance = _rigid_body_motion(predecessor, body, step)
            advances[body.name] = advance
            centers[body.name] = centers[body.name] + body.offset(step)
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
        candidate = connectivity.moved_from_advances(advances)
        if not candidate.complete or candidate.orphan_count != _ORPHAN_COUNT_TOLERANCE:
            raise RuntimeError(
                f"Accepted rigid-motion profile epoch lost receptor coverage: "
                f"{candidate.orphan_count=}."
            )
        independent_blanking = _independent_blanking(candidate, profile, centers)
        donor_oracle = _independent_donor_oracle(candidate)
        field_oracle = _field_transfer(candidate)
        transport, conservative = _conservative_epoch_transport(
            connectivity,
            candidate,
            background_motion,
            composition,
        )
        predecessor_composition = composition
        predecessor_composition_id = composition.composition_id
        staged = phx.meshing.prepare_overset_motion_rebind(
            predecessor_composition,
            connectivity,
            candidate,
            transports=(transport,),
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
        profile_record["epochs"].append(
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
                "predecessor_composition_id": predecessor_composition_id,
                "rejected_boundary_rollback": (
                    not refused.published
                    and refused.composition is predecessor_composition
                ),
                "accepted_rebind_published": accepted.published,
                "accepted_composition_id": composition.composition_id,
            }
        )
        connectivity = candidate
    orphan = _accepted_orphan_refusal(profile, connectivity, composition)
    profile_record["orphan_failure"] = orphan
    missing_common = {
        "status": "missing-production-owner",
        "attempted_equation": "u=5+x+y-(x^2+y^2)/4, -Laplace(u)=1",
        "required_boundary_data": (
            "Dirichlet on background exterior and every protected cavity wall"
        ),
        "required_error_quantity": "independent-manufactured-composite-overset-L2-error",
        "available_non_equivalent_apis": [
            "phydrax.discretization.fem.solve_hdg_poisson solves one mesh-local HDG problem",
            "phydrax.solver.PreparedFiniteVolumeRuntime.commit_moving_overset_update is an FV conservation update, not the authored FE Poisson problem",
        ],
        "missing_owner_api": "phydrax.meshing.qualification.prepare_poisson_update",
        "missing_reason": (
            "The current product tree has no production composite Poisson owner/API. "
            "The former prepare_poisson_update route belonged to the historical "
            "qualification scenario and cannot authorize a positive composite solve."
        ),
        "pde_normalized_residual": None,
        "flux_balance_relative": None,
    }
    if len(profile.bodies) > 1:
        missing_common["available_non_equivalent_apis"] = [
            *missing_common["available_non_equivalent_apis"],
            "phydrax.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan.overset binds one donor/receptor geometry pair",
            "phydrax.solver.PreparedFiniteVolumeRuntime.commit_moving_overset_update resolves one moved FV mesh part",
        ]
        missing_common["missing_reason"] = (
            f"The {len(profile.bodies)}-body profile requires one accepted composite "
            "Poisson solve across overlapping bodies and the background; current "
            "production APIs do not own this multi-part FE operator or its physical L2/QoI reanalysis."
        )
    profile_record["composite_pde"] = missing_common
    profile_record["status"] = "failed"
    return profile_record


def native_moving_overset() -> dict[str, object]:
    """Run every authored moving-overset source profile and report all missing gates."""
    meshcore = require_meshcore()
    profiles: dict[str, object] = {}
    for profile in _PROFILES:
        try:
            profiles[profile.name] = _run_profile(profile)
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
                "detail": str(error),
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
                "composite_pde": {
                    "status": "missing-production-owner",
                    "missing_owner_api": "phydrax.meshing.qualification.prepare_poisson_update",
                    "missing_reason": (
                        "Current production APIs expose no accepted composite FE Poisson "
                        "owner for the authored manufactured solution and L2/QoI reanalysis."
                    ),
                    "pde_normalized_residual": None,
                    "flux_balance_relative": None,
                },
                "partial_matrix_pass": False,
            }
    return {
        "status": "failed",
        "workflow": "native square-annulus/background generation -> source-bound rigid motion -> independent hole/fringe/donor oracle -> affine FE transfer -> accepted conservative overlap rebind",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#moving-overset",
        "donor_profile_registry": "tools/_meshing_cases.py::CORPUS",
        "donor_source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
        "profiles": profiles,
        "missing_production_contracts": {
            name: value.get("composite_pde")
            if isinstance(value, dict)
            else "profile execution did not produce a composite PDE record"
            for name, value in profiles.items()
        },
        "profile_matrix_passed": False,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
