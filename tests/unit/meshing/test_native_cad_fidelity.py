# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent source-coefficient checks for native CAD fidelity."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import (
    CadFidelityChords,
    CadFidelityMesh,
    MeshcoreStatus,
    verify_cad_fidelity,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_interpolation import certify_domain_surface_charts
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    ConePatch,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from phydrax.meshing.providers._native_cad_bundle import (
    cad_geometry_bundle,
    coefficient_bounds,
)


def _surface_domain(surface: AbstractSurfacePatch) -> MeshingDomain:
    """Two oppositely oriented occurrences of one source chart."""
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))

    def loop(sense: int) -> tuple[PatchCurveUse, ...]:
        uses = tuple(
            PatchCurveUse(
                curve,
                LineCurve(
                    np.asarray(corners[first], dtype=np.float64),
                    np.asarray(corners[last], dtype=np.float64)
                    - np.asarray(corners[first], dtype=np.float64),
                ),
                0.0,
                1.0,
                sense=sense,
            )
            for curve, (first, last) in enumerate(pairs)
        )
        return uses if sense > 0 else tuple(reversed(uses))

    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (loop(1),)),
         MeshingSurfacePatch(surface, (loop(-1),), orientation=-1)),
        tuple(MeshingDomainCurve(first, last) for first, last in pairs),
        4,
        source_id="qualification:native-fidelity-cylinder",
        source_revision="r1",
    )
    return domain


def _cylinder_domain() -> MeshingDomain:
    surface = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), float(Fraction(1, 3)), radius_exact=Fraction(1, 3),
    )
    return _surface_domain(surface)


def test_lazy_cylinder_coefficient_bounds() -> None:
    """Each exact cylinder coefficient, not its rounded shadow, is enclosed."""
    domain = _cylinder_domain()
    bundle = cad_geometry_bundle(domain)
    before = bundle.model.node_reals.copy()
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.array_equal(bundle.model.node_reals, before)
    assert not lower.flags.writeable and not upper.flags.writeable
    cylinders = 0
    for patch, node in zip(domain.patches, bundle.model.surface_nodes, strict=True):
        if not isinstance(patch.surface, CylinderPatch):
            continue
        cylinders += 1
        surface = patch.surface
        exact = (
            *surface.origin_exact,
            *surface.first_axis_exact,
            *surface.second_axis_exact,
            *surface.axis_exact,
            surface.radius_exact,
        )
        offset = bundle.model.node_real_offsets[node]
        for index, value in enumerate(exact):
            assert Fraction.from_float(lower[offset + index]) <= value
            assert value <= Fraction.from_float(upper[offset + index])
    assert cylinders > 0


def _analytic_surface(kind: str) -> AbstractSurfacePatch:
    origin = (0.0, 0.0, 0.0)
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    match kind:
        case "plane":
            return PlanePatch(origin, x, y)
        case "cylinder":
            return CylinderPatch(origin, x, y, z, 1.0 / 3)
        case "cone":
            return ConePatch(origin, x, y, z, 0.4, 0.2)
        case "sphere":
            return SpherePatch(origin, x, y, z, 1.0 / 3)
        case "torus":
            return TorusPatch(origin, x, y, z, 0.7, 0.2)
        case _:
            raise ValueError(f"Unknown analytic fixture {kind}.")


def _analytic_values(kind: str, uv: np.ndarray) -> np.ndarray:
    u, v = uv[..., 0], uv[..., 1]
    match kind:
        case "plane":
            return np.stack((u, v, np.zeros_like(u)), axis=-1)
        case "cylinder":
            return np.stack((np.cos(u) / 3, np.sin(u) / 3, v), axis=-1)
        case "cone":
            radius = 0.4 + v * np.tan(0.2)
            return np.stack((radius * np.cos(u), radius * np.sin(u), v), axis=-1)
        case "sphere":
            return np.stack(
                (np.cos(v) * np.cos(u) / 3, np.cos(v) * np.sin(u) / 3, np.sin(v) / 3),
                axis=-1,
            )
        case "torus":
            radius = 0.7 + 0.2 * np.cos(v)
            return np.stack((radius * np.cos(u), radius * np.sin(u), 0.2 * np.sin(v)), axis=-1)
        case _:
            raise ValueError(f"Unknown analytic fixture {kind}.")


def _cylinder_mesh(
    domain: MeshingDomain, kind: str = "cylinder",
) -> tuple[CadFidelityMesh, CadFidelityChords]:
    uv_points = np.asarray(
        [(0.5 * i, 0.5 * j) for j in range(3) for i in range(3)], dtype=np.float64
    )
    points = _analytic_values(kind, uv_points)
    triangles: list[tuple[int, int, int]] = []
    for j in range(2):
        for i in range(2):
            a = 3 * j + i
            triangles.extend(((a, a + 1, a + 4), (a, a + 4, a + 3)))
    cells = np.asarray(triangles, dtype=np.int64)
    rows = np.zeros(len(triangles), dtype=np.int32)
    edge_rows = np.full((len(triangles), 3), -1, dtype=np.int64)
    owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, cell in enumerate(triangles):
        for edge in range(3):
            a, b = cell[edge], cell[(edge + 1) % 3]
            owners.setdefault((min(a, b), max(a, b)), []).append((row, edge))
    chord_use: list[int] = []
    chord_uv: list[np.ndarray] = []
    chord_range: list[tuple[float, float]] = []
    for occurrences in owners.values():
        if len(occurrences) != 1:
            continue
        row, edge = occurrences[0]
        a, b = triangles[row][edge], triangles[row][(edge + 1) % 3]
        start, end = uv_points[a], uv_points[b]
        if start[1] == end[1] == 0:
            use, first = 0, start[0]
        elif start[0] == end[0] == 1:
            use, first = 1, start[1]
        elif start[1] == end[1] == 1:
            use, first = 2, 1.0 - start[0]
        else:
            use, first = 3, 1.0 - start[1]
        edge_rows[row, edge] = len(chord_use)
        chord_use.append(use)
        chord_uv.append(np.stack((start, end)))
        chord_range.append((float(first), float(first + 0.5)))
    mesh = CadFidelityMesh(points, cells, rows, uv_points[cells], edge_rows)
    bundle = cad_geometry_bundle(domain)
    use_patch = np.repeat(
        np.arange(bundle.model.surface_nodes.size, dtype=np.int64),
        np.diff(bundle.model.loop_use_offsets),
    )
    chords = CadFidelityChords(
        use_patch,
        bundle.model.use_ranges,
        bundle.model.loop_surfaces.size,
        np.asarray(chord_use, dtype=np.int64),
        np.zeros(len(chord_use), dtype=np.int64),
        np.asarray(chord_uv, dtype=np.float64),
        np.asarray(chord_range, dtype=np.float64),
    )
    return mesh, chords


def _python_chart_decisions(
    domain: MeshingDomain, native_mesh: CadFidelityMesh, rows: np.ndarray, tolerance: float
) -> tuple[bool, ...]:
    """Invoke the existing exact-source chart certifier on the selected faces."""
    ids = np.arange(101, 101 + rows.size, dtype=np.int64)
    target = CellMesh(
        native_mesh.points,
        (CellBlock("surface", "triangle", native_mesh.triangles[rows], global_ids=ids),),
    )
    geometry = CellGeometrySpec.affine(target)
    source = MeshingDomainBoundarySource(domain, (0,))
    entity = domain.entity_id(2, 0)
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    witness = SourceBoundaryChartWitness(
        mesh_id=target.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=target.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=ids,
        entity_global_ids=ids,
        local_entity_indices=np.full(rows.size, -1, dtype=np.int32),
        target_reference_vertices=np.tile(reference[None], (rows.size, 1, 1)),
        source_chart_ids=(entity,) * rows.size,
        source_entity_ids=(entity,) * rows.size,
        occurrence_paths=((),) * rows.size,
        source_parameters=native_mesh.uv[rows],
        orientations=np.ones(rows.size, dtype=np.int8),
    )
    evidence = certify_domain_surface_charts(
        target,
        geometry,
        source,
        witness,
        np.zeros(rows.size, dtype=np.int64),
        tuple((0, i) for i in range(rows.size)),
        tolerance=tolerance,
    )
    failed = {
        int(finding.entity_ids[0])
        for finding in evidence.findings
        if finding.check == "source_fidelity_tolerance"
    }
    assert all(finding.check == "source_fidelity_tolerance" for finding in evidence.findings)
    return tuple(int(value) not in failed for value in ids)


@pytest.mark.parametrize("kind", ("plane", "cylinder", "cone", "sphere", "torus"))
def test_interior_bounds_against_python_and_sampling(kind: str) -> None:
    """Enclose the original triangle interpolation after subdivision."""
    domain = _surface_domain(_analytic_surface(kind))
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    mesh, chords = _cylinder_mesh(domain, kind)
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.UNRESOLVED
    assert result.findings.size == 0
    interior = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    assert interior.size > 0
    assert _python_chart_decisions(domain, mesh, interior, 0.5) == tuple(
        status == MeshcoreStatus.OK for status in result.item_status[interior]
    )
    ratios: list[float] = []
    for row in interior:
        assert result.item_status[row] == MeshcoreStatus.OK
        chart = tuple(
            (Fraction.from_float(float(vertex[0])), Fraction.from_float(float(vertex[1])))
            for vertex in mesh.uv[row]
        )
        reference = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 2),))[0]
        assert reference is not None
        if reference > Fraction(1, 2**100):
            ratios.append(result.upper[row] / float(reference))
        else:
            # A plane's exact interpolation error is zero; an ulp-scale
            # absolute bound is meaningful where a relative ratio is not.
            assert result.upper[row] < 1e-12
        vertices = mesh.points[mesh.triangles[row]]
        for i in range(21):
            for j in range(21 - i):
                weights = np.asarray((1 - (i + j) / 20, i / 20, j / 20), dtype=np.float64)
                sample_uv = weights @ mesh.uv[row]
                exact = _analytic_values(kind, sample_uv)
                assert np.linalg.norm(exact - weights @ vertices) <= result.upper[row]
    print(f"{kind} native/Python adaptive bound ratios:", ratios)
    assert all(0.99 <= ratio <= 1.01 for ratio in ratios)
    assert np.all(result.reasons[np.any(mesh.edge_chord >= 0, axis=1)] == 3)


def test_chord_association_findings_are_returned() -> None:
    domain = _cylinder_domain()
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    mesh, chords = _cylinder_mesh(domain)
    edges = mesh.edge_chord.copy()
    row, edge = np.argwhere(edges >= 0)[0]
    chord = edges[row, edge]
    edges[row, edge] = -1
    verdict = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(edge_chord=edges), chords, 0.1, 4
    )
    assert verdict.status == MeshcoreStatus.UNRESOLVED
    assert np.any((verdict.findings["kind"] == 3) & (verdict.findings["chord"] == chord))
    assert np.any(
        (verdict.findings["kind"] == 4)
        & (verdict.findings["triangle"] == row)
        & (verdict.findings["edge"] == edge)
    )


def test_perturbed_vertex_and_uv_are_unresolved() -> None:
    """Endpoint inconsistencies cannot be certified by small leaf Hessians."""
    domain = _cylinder_domain()
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    mesh, chords = _cylinder_mesh(domain)
    interior = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    points = mesh.points.copy()
    points[4, 2] += 0.25
    moved = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(points=points), chords, 0.1, 3
    )
    assert np.all(moved.item_status[interior] == MeshcoreStatus.UNRESOLVED)
    assert np.all(moved.reasons[interior] == 1)
    assert _python_chart_decisions(domain, mesh._replace(points=points), interior, 0.1) == tuple(
        status == MeshcoreStatus.OK for status in moved.item_status[interior]
    )
    uv = mesh.uv.copy()
    uv[:, :, 0][mesh.triangles == 4] += 0.75
    perturbed = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(uv=uv), chords, 0.1, 3
    )
    assert perturbed.findings.size == 0
    assert np.all(perturbed.item_status[interior] == MeshcoreStatus.UNRESOLVED)
    assert np.all(perturbed.reasons[interior] == 1)
    assert _python_chart_decisions(domain, mesh._replace(uv=uv), interior, 0.1) == tuple(
        status == MeshcoreStatus.OK for status in perturbed.item_status[interior]
    )
