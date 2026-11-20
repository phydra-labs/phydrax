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
    PatchPoleUse,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_interpolation import certify_domain_surface_charts
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    CircleCurve,
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

def test_lazy_bspline_coefficient_bounds() -> None:
    """The flat bank encloses exact knots, controls and rational weights."""
    third = Fraction(1, 3)
    controls = (
        ((Fraction(0), Fraction(0), third), (Fraction(0), Fraction(1), third)),
        ((Fraction(1), Fraction(0), third), (Fraction(1), Fraction(1), third)),
    )
    weights = ((Fraction(1), third), (Fraction(2, 3), Fraction(1)))
    knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    surface = BSplineSurfacePatch(
        np.asarray(controls, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        1,
        1,
        exact_control_points=controls,
        exact_weights=weights,
        exact_u_knots=knots,
        exact_v_knots=knots,
    )
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    offset = bundle.model.node_real_offsets[bundle.model.surface_nodes[0]]
    expected = (
        *knots,
        *knots,
        *(value for row in controls for point in row for value in point),
        *(value for row in weights for value in row),
    )
    for index, value in enumerate(expected):
        assert Fraction.from_float(lower[offset + index]) <= value
        assert value <= Fraction.from_float(upper[offset + index])


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
    mesh = CadFidelityMesh(
        points, cells, rows, uv_points[cells], cells, np.arange(len(cells), dtype=np.int64),
        edge_rows,
    )
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
    assert result.status == MeshcoreStatus.OK
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
    assert np.all(result.item_status == MeshcoreStatus.OK)


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


@pytest.mark.parametrize("kind", ("cylinder", "cone", "sphere"))
def test_curved_trim_lens_contains_sampled_arc(kind: str) -> None:
    """A curved source p-curve can depart from its target chord."""
    surface = _analytic_surface(kind)
    half_pi = float(np.pi / 2)
    angles = np.asarray((half_pi, half_pi / 2, 0.0), dtype=np.float64)
    arc = np.stack((1.0 - np.cos(angles), np.sin(angles)), axis=1)
    polygon = np.vstack((arc, (2.0, 0.0), (2.0, 1.0)))
    circular = CircleCurve((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), 1.0)
    straight = tuple(
        LineCurve(polygon[i], polygon[(i + 1) % len(polygon)] - polygon[i])
        for i in range(2, 5)
    )

    def loop(sense: int) -> tuple[PatchCurveUse, ...]:
        uses = (
            PatchCurveUse(0, circular, 0.0, half_pi, sense=-sense),
            *(PatchCurveUse(i + 1, curve, 0.0, 1.0, sense=sense)
              for i, curve in enumerate(straight)),
        )
        return uses if sense > 0 else tuple(reversed(uses))

    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (loop(1),)),
         MeshingSurfacePatch(surface, (loop(-1),), orientation=-1)),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(0, 2),
         MeshingDomainCurve(2, 3), MeshingDomainCurve(3, 1)),
        4,
        tolerance=1e-12,
        source_id=f"qualification:native-curved-trim-{kind}",
        source_revision="r1",
    )
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    chart_points = np.vstack(((1.45, 0.5), polygon))
    indices = np.asarray(
        [(0, i + 1, (i + 1) % len(polygon) + 1) for i in range(len(polygon))],
        dtype=np.int64,
    )
    edges = np.full((len(indices), 3), -1, dtype=np.int64)
    edges[:, 1] = np.arange(len(indices), dtype=np.int64)
    mesh = CadFidelityMesh(
        _analytic_values(kind, chart_points),
        indices,
        np.zeros(len(indices), dtype=np.int32),
        chart_points[indices],
        indices,
        np.arange(len(indices), dtype=np.int64),
        edges,
    )
    chord_range = np.asarray(
        ((half_pi, half_pi / 2), (half_pi / 2, 0.0), (0.0, 1.0),
         (0.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    use_patch = np.repeat(
        np.arange(bundle.model.surface_nodes.size, dtype=np.int64),
        np.diff(bundle.model.loop_use_offsets),
    )
    chords = CadFidelityChords(
        use_patch,
        bundle.model.use_ranges,
        bundle.model.loop_surfaces.size,
        np.asarray((0, 0, 1, 2, 3), dtype=np.int64),
        np.zeros(len(indices), dtype=np.int64),
        mesh.uv[:, 1:3, :],
        chord_range,
    )
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.5, 5)
    assert result.status == MeshcoreStatus.OK
    assert result.findings.size == 0
    rows = np.arange(len(indices), dtype=np.int64)
    assert _python_chart_decisions(domain, mesh, rows, 0.5) == tuple(
        status == MeshcoreStatus.OK for status in result.item_status
    )
    for row in (0, 1):
        for fraction in np.linspace(0.0, 1.0, 101):
            t = (1.0 - fraction) * chord_range[row, 0] + fraction * chord_range[row, 1]
            uv = np.asarray((1.0 - np.cos(t), np.sin(t)), dtype=np.float64)
            source = _analytic_values(kind, uv)
            first, last = mesh.points[indices[row, 1:3]]
            chord = (1.0 - fraction) * first + fraction * last
            assert np.linalg.norm(source - chord) <= result.upper[row]


def _sphere_pole_domain(*, pole: bool) -> MeshingDomain:
    latitude = float(np.pi / 2)
    corners = np.asarray(
        ((0.0, latitude - 0.125), (1.0, latitude - 0.125),
         (1.0, latitude), (0.0, latitude)),
        dtype=np.float64,
    )
    pcurves = tuple(
        LineCurve(corners[i], corners[(i + 1) % 4] - corners[i]) for i in range(4)
    )
    surface = _analytic_surface("sphere")

    def loop(sense: int) -> tuple[PatchCurveUse | PatchPoleUse, ...]:
        uses: tuple[PatchCurveUse | PatchPoleUse, ...]
        if pole:
            uses = (
                PatchCurveUse(0, pcurves[0], 0.0, 1.0, sense=sense),
                PatchCurveUse(1, pcurves[1], 0.0, 1.0, sense=sense),
                PatchPoleUse(2, pcurves[2], 0.0, 1.0, sense=sense),
                PatchCurveUse(2, pcurves[3], 0.0, 1.0, sense=sense),
            )
        else:
            uses = tuple(
                PatchCurveUse(i, curve, 0.0, 1.0, sense=sense)
                for i, curve in enumerate(pcurves)
            )
        return uses if sense > 0 else tuple(reversed(uses))

    curves = (
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(2, 0))
        if pole
        else tuple(MeshingDomainCurve(i, (i + 1) % 4) for i in range(4))
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop(1),)),
         MeshingSurfacePatch(surface, (loop(-1),), orientation=-1)),
        curves,
        3 if pole else 4,
        source_id=f"qualification:native-sphere-pole-{pole}",
        source_revision="r1",
        tolerance=1e-12,
    )


def test_symbolic_pole_token_is_widened_only_for_pole_use() -> None:
    """The sphere pole owner, not a coincidental float token, controls widening."""
    shadow = float(np.pi / 2)
    for pole in (True, False):
        domain = _sphere_pole_domain(pole=pole)
        bundle = cad_geometry_bundle(domain)
        lower, upper = coefficient_bounds(bundle, domain)
        node = bundle.model.use_pcurve_nodes[2]
        latitude = bundle.model.node_real_offsets[node] + 1
        if pole:
            from phydrax.geometry.brep._trim_helpers import (
                period_offset,
                period_scalar_bounds,
            )

            lo, hi = period_scalar_bounds(period_offset(Fraction(0), Fraction(1, 4)))
            assert lower[latitude] <= float(lo)
            assert upper[latitude] >= float(hi)
            assert lower[latitude] < shadow < upper[latitude]
        else:
            assert lower[latitude] == shadow == upper[latitude]


def test_sphere_pole_sliver_and_chart_mesh_link() -> None:
    """The chart has a pole sliver; the physical mesh omits it."""
    domain = _sphere_pole_domain(pole=True)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    latitude = float(np.pi / 2)
    chart = np.asarray(
        ((0.0, latitude - 0.125), (1.0, latitude - 0.125),
         (1.0, latitude), (0.0, latitude)),
        dtype=np.float64,
    )
    points = np.vstack((_analytic_values("sphere", chart[:2]), (0.0, 0.0, 1.0 / 3)))
    chart_indices = np.asarray(((0, 1, 2), (0, 2, 2)), dtype=np.int64)
    chart_uv = np.asarray((chart[[0, 1, 2]], chart[[0, 2, 3]]), dtype=np.float64)
    mesh = CadFidelityMesh(
        points,
        chart_indices,
        np.zeros(2, dtype=np.int32),
        chart_uv,
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((0, -1), dtype=np.int64),
        np.asarray(((0, 1, -1), (-1, 2, 3)), dtype=np.int64),
    )
    chord_uv = np.asarray(
        tuple((chart[i], chart[(i + 1) % 4]) for i in range(4)), dtype=np.float64
    )
    chords = CadFidelityChords(
        np.repeat(np.arange(2, dtype=np.int64), np.diff(bundle.model.loop_use_offsets)),
        bundle.model.use_ranges,
        bundle.model.loop_surfaces.size,
        np.arange(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        chord_uv,
        np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (4, 1)),
    )
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.OK
    assert result.findings.size == 0

    def exact_uv(index: int) -> tuple[Fraction, Fraction]:
        vertex = chart_uv[1, index]
        return Fraction.from_float(float(vertex[0])), Fraction.from_float(float(vertex[1]))

    exact_sliver = (exact_uv(0), exact_uv(1), exact_uv(2))
    python_cap = MeshingDomainBoundarySource(domain, (0,)).pole_sliver_cap_bound(
        0, exact_sliver
    )
    assert (python_cap <= Fraction(1, 2)) == (result.item_status[1] == MeshcoreStatus.OK)
    for i in range(21):
        for j in range(21 - i):
            weights = np.asarray((1 - (i + j) / 20, i / 20, j / 20), dtype=np.float64)
            source = _analytic_values("sphere", weights @ chart_uv[1])
            assert np.linalg.norm(source - points[2]) <= result.upper[1]
    bad_mapping = mesh.chart_to_mesh.copy()
    bad_mapping[1] = 0
    broken = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(chart_to_mesh=bad_mapping), chords, 0.5, 4
    )
    assert broken.status == MeshcoreStatus.UNRESOLVED
    assert np.any((broken.findings["kind"] == 5) & (broken.findings["triangle"] == 1))
