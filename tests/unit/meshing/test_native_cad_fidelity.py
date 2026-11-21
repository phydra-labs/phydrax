# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent source-coefficient checks for native CAD fidelity."""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from phydrax._meshcore import (
    CadFidelityChords,
    CadFidelityMesh,
    CadFidelityResult,
    CadModel,
    MeshcoreStatus,
    verify_cad_fidelity as native_verify_cad_fidelity,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    _span_enclosures_for_chart,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
    SourcePreparation,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_interpolation import certify_domain_surface_charts
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from phydrax.meshing.providers._native_cad_bundle import (
    cad_geometry_bundle,
    coefficient_bounds,
)


def verify_cad_fidelity(
    model: CadModel,
    lower: np.ndarray,
    upper: np.ndarray,
    mesh: CadFidelityMesh,
    chords: CadFidelityChords,
    tolerance: float,
    max_depth: int,
) -> CadFidelityResult:
    """V3 on a fixture that meshes one occurrence of a two-sided source.

    V3 checks V4's use coverage and loop closure, so every model loop needs
    chords. A loop without chords whose uses are the reversed p-curves of a
    meshed loop gets that loop's rows mirrored (reversed triangles and
    chords); each use range is given in the order of its chords. Results are
    reported for the fixture's own rows.
    """
    offsets = model.loop_use_offsets
    rows = mesh.triangles.shape[0]
    use_range = np.asarray(model.use_ranges, dtype=np.float64).copy()
    for use in np.unique(chords.chord_use):
        first = np.flatnonzero(chords.chord_use == use)
        if chords.chord_range[first[0], 0] > chords.chord_range[first[0], 1]:
            use_range[use] = use_range[use, ::-1]
    triangles = [mesh.triangles]
    faces = [mesh.face_rows]
    uv = [mesh.uv]
    mesh_triangles = [mesh.mesh_triangles]
    chart_to_mesh = [mesh.chart_to_mesh]
    edge_chord = [mesh.edge_chord]
    chord_use = [chords.chord_use]
    chord_loop = [chords.chord_loop]
    chord_uv = [chords.chord_uv]
    chord_range = [chords.chord_range]
    chord_count = chords.chord_use.size
    mesh_count = mesh.mesh_triangles.shape[0]
    covered = set(np.unique(chords.chord_loop).tolist())
    for loop in range(model.loop_surfaces.size):
        if loop in covered:
            continue
        count = offsets[loop + 1] - offsets[loop]
        node = model.surface_nodes[model.loop_surfaces[loop]]
        source = next(
            other for other in covered
            if offsets[other + 1] - offsets[other] == count
            and model.surface_nodes[model.loop_surfaces[other]] == node
        )
        selected = np.flatnonzero(chords.chord_loop == source)
        mirrored = np.full(chord_count, -1, dtype=np.int64)
        mirrored[selected] = chord_count + np.arange(selected.size, dtype=np.int64)
        uses = offsets[loop] + (count - 1 - (chords.chord_use[selected] - offsets[source]))
        chord_use.append(uses.astype(np.int64))
        chord_loop.append(np.full(selected.size, loop, dtype=np.int64))
        chord_uv.append(chords.chord_uv[selected][:, ::-1])
        chord_range.append(chords.chord_range[selected][:, ::-1])
        for use in np.unique(uses):
            use_range[use] = use_range[offsets[source] + count - 1 - (use - offsets[loop])][::-1]
        chord_count += selected.size
        patch = model.loop_surfaces[source]
        own = np.flatnonzero(mesh.face_rows == patch)
        triangles.append(mesh.triangles[own][:, [0, 2, 1]])
        faces.append(np.full(own.size, model.loop_surfaces[loop], dtype=np.int32))
        uv.append(mesh.uv[own][:, [0, 2, 1]])
        edges = mesh.edge_chord[own][:, [2, 1, 0]]
        edge_chord.append(np.where(edges >= 0, mirrored[np.maximum(edges, 0)], -1))
        mapped = mesh.chart_to_mesh[own]
        physical = mapped >= 0
        mesh_triangles.append(mesh.mesh_triangles[mapped[physical]][:, [0, 2, 1]])
        image = np.full(own.size, -1, dtype=np.int64)
        image[physical] = mesh_count + np.arange(np.count_nonzero(physical), dtype=np.int64)
        chart_to_mesh.append(image)
        mesh_count += int(np.count_nonzero(physical))
    completed = CadFidelityMesh(
        mesh.points, np.concatenate(triangles), np.concatenate(faces), np.concatenate(uv),
        np.concatenate(mesh_triangles), np.concatenate(chart_to_mesh), np.concatenate(edge_chord),
    )
    table = CadFidelityChords(
        chords.use_patch, use_range, chords.loop_count, np.concatenate(chord_use),
        np.concatenate(chord_loop), np.concatenate(chord_uv), np.concatenate(chord_range),
    )
    result = native_verify_cad_fidelity(
        model, lower, upper, completed, table, tolerance, max_depth
    )
    return result._replace(
        upper=result.upper[:rows], item_status=result.item_status[:rows],
        reasons=result.reasons[:rows], subdivisions=result.subdivisions[:rows],
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


def test_offset_cylinder_computed_exact_radius() -> None:
    """A rounded sum of offset and base radius must enclose the exact sum."""
    base = _cylinder_domain().patches[0].surface
    assert isinstance(base, CylinderPatch)
    surface = OffsetSurface(base, Fraction(1, 7))
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    node = bundle.model.surface_nodes[0]
    index = bundle.model.node_real_offsets[node] + 12
    exact = Fraction(1, 3) + surface.frame_sign * Fraction(1, 7)
    assert Fraction.from_float(lower[index]) <= exact <= Fraction.from_float(upper[index])
    mesh, chords = _cylinder_mesh(domain)
    points = mesh.points.copy()
    points[:, :2] *= 3.0 * float(exact)
    candidate = mesh._replace(points=points)
    result = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.OK
    assert np.all(result.item_status == MeshcoreStatus.OK)


def test_offset_plane_irrational_normal_coefficients() -> None:
    plane = PlanePatch((0, 0, 0), (1, 1, 0), (0, 1, 1))
    surface = OffsetSurface(plane, Fraction(1, 7))
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    start = bundle.model.node_real_offsets[bundle.model.surface_nodes[0]]
    with localcontext() as context:
        context.prec = 100
        shift = Decimal(surface.frame_sign or 1) / (Decimal(7) * Decimal(3).sqrt())
        expected = (shift, -shift, shift, Decimal(1), Decimal(1), Decimal(0),
                    Decimal(0), Decimal(1), Decimal(1))
        for index, exact in enumerate(expected):
            assert Decimal.from_float(float(lower[start + index])) <= exact
            assert exact <= Decimal.from_float(float(upper[start + index]))


_EXACT_CARRIER_REFUSAL = (
    "An exact-backed source carrier has no retained exact coordinate map; "
    "rounded dynamic values are not a proof."
)


@pytest.mark.parametrize("distance", [Fraction(1, 7), Fraction(1, 8)])
def test_generic_extrusion_offset_fidelity(distance: Fraction) -> None:
    """Offset of an extruded circle: native uses the exact distance.

    Declared capability difference for d = 1/7: Python refuses at
    ``_source_interpolation.certify_domain_surface_charts`` because
    ``_has_unexpanded_exact_surface_carrier`` reports the non-binary
    ``OffsetSurface.distance_exact``; native encloses 1/7 outward and the
    stored binary64 circle and direction exactly. With dyadic d = 1/8 both
    certify the same rows and native is no looser than Python.
    """
    circle = CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0)
    surface = OffsetSurface(ExtrusionSurface(circle, (0, 0, 1)), distance)
    radius = 1 + (surface.frame_sign or 1) * distance
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    mesh, chords = _cylinder_mesh(domain)
    points = mesh.points.copy()
    points[:, :2] *= 3.0 * float(radius)
    rows = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    dyadic = distance.denominator & (distance.denominator - 1) == 0
    for broken in (False, True):
        candidate_points = points.copy()
        if broken:
            candidate_points[4, 2] += 1000.0
        candidate = mesh._replace(points=candidate_points)
        result = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords, 100.0, 4)
        native = tuple(status == MeshcoreStatus.OK for status in result.item_status[rows])
        python = _python_chart_decisions(
            domain, candidate, rows, 100.0,
            refusal=None if dyadic else _EXACT_CARRIER_REFUSAL,
        )
        expected = tuple(not (broken and 4 in candidate.triangles[row]) for row in rows)
        assert native == expected
        assert python == (expected if dyadic else (False,) * rows.size)
        # Native refines until its bound meets the target, so the parity check
        # asks native to certify within Python's own adaptive bound.
        for row in rows:
            vertices = candidate_points[candidate.triangles[row]]
            if dyadic and not broken:
                chart = tuple(
                    (Fraction.from_float(float(a)), Fraction.from_float(float(b)))
                    for a, b in candidate.uv[row]
                )
                reference = domain.adaptive_interpolation_bounds(
                    0, (chart,), (Fraction(1, 2),)
                )[0]
                assert reference is not None
                tight = verify_cad_fidelity(
                    bundle.model, lower, upper, candidate, chords, float(reference), 6
                )
                assert tight.item_status[row] == MeshcoreStatus.OK
                assert tight.upper[row] <= float(reference)
                print("extrusion offset native/Python", tight.upper[row] / float(reference),
                      "subdivisions", tight.subdivisions[row])
            for i in range(11):
                for j in range(11 - i):
                    barycentric = np.asarray((1 - (i + j) / 10, i / 10, j / 10))
                    uv = barycentric @ candidate.uv[row]
                    exact = float(radius) * np.asarray((np.cos(uv[0]), np.sin(uv[0]), 0.0))
                    exact[2] = uv[1]
                    assert np.linalg.norm(exact - barycentric @ vertices) <= result.upper[row]


def _rational_spline_surface() -> BSplineSurfacePatch:
    third = Fraction(1, 3)
    controls = (
        ((Fraction(0), Fraction(0), third), (Fraction(0), Fraction(1), third)),
        ((Fraction(1), Fraction(0), third), (Fraction(1), Fraction(1), third)),
    )
    weights = ((Fraction(1), third), (Fraction(2, 3), Fraction(1)))
    knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    return BSplineSurfacePatch(
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


def test_lazy_bspline_coefficient_bounds() -> None:
    """The flat bank encloses exact knots, controls and rational weights."""
    surface = _rational_spline_surface()
    controls = surface.exact_control_points
    weights = surface.exact_weights
    knots = surface.exact_u_knots
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
    domain: MeshingDomain,
    native_mesh: CadFidelityMesh,
    rows: np.ndarray,
    tolerance: float,
    *,
    refusal: str | None = None,
) -> tuple[bool, ...]:
    """Invoke the existing exact-source chart certifier on the selected faces.

    ``refusal`` names a declared Python capability refusal: every row must
    then be refused with exactly that detail.
    """
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
    if refusal is not None:
        assert len(evidence.findings) == rows.size
        assert all(
            finding.status == "unresolved" and finding.detail == refusal
            for finding in evidence.findings
        )
        return (False,) * rows.size
    failed = {
        int(finding.entity_ids[0])
        for finding in evidence.findings
        if finding.check == "source_fidelity_tolerance"
    }
    assert all(finding.check == "source_fidelity_tolerance" for finding in evidence.findings)
    return tuple(int(value) not in failed for value in ids)


def _rational_spline_values(uv: np.ndarray) -> np.ndarray:
    u, v = uv[..., 0], uv[..., 1]
    weights = (
        (1 - u) * (1 - v),
        (1 - u) * v / 3,
        2 * u * (1 - v) / 3,
        u * v,
    )
    denominator = sum(weights)
    return np.stack(
        ((weights[2] + weights[3]) / denominator,
         (weights[1] + weights[3]) / denominator,
         np.full_like(u, 1 / 3)),
        axis=-1,
    )


def test_rational_spline_verdict_and_dense_bounds() -> None:
    """Adaptive native bounds are tighter than Python's unsubdivided spline bounds.

    Rows 3/4: native 0.123267/0.264703, Python 1.090123/7.954951,
    sampled-and-locally-maximized distance 0.042045/0.020825. Python bounds
    exact homogeneous Bernstein derivative numerator nets on the original box,
    whereas native interval Cox--de Boor/quotient bounds lose dependencies
    but subdivide. See _bspline_extraction.py:1723-1740,1778-1814 and
    _meshing_domain.py:4092,4128.
    """
    domain = _surface_domain(_rational_spline_surface())
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    mesh, chords = _cylinder_mesh(domain)
    uv = np.asarray([(0.5 * i, 0.5 * j) for j in range(3) for i in range(3)])
    mesh = mesh._replace(points=_rational_spline_values(uv))
    rows = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    fixture_path = Path(__file__).parents[3] / "native/meshcore/tests/spline_hessian_fixture.inc"
    numbers = re.findall(r"-?0x[0-9a-f.]+p[+-][0-9]+", fixture_path.read_text())
    fixture = np.asarray([float.fromhex(value) for value in numbers]).reshape((2, 20))
    spans = SourcePreparation(domain).surface_spans(domain.patches[0].surface)
    assert spans is not None
    for position, row in enumerate(rows):
        chart = tuple(
            (Fraction.from_float(float(a)), Fraction.from_float(float(b)))
            for a, b in mesh.uv[row]
        )
        enclosure = _span_enclosures_for_chart(spans, chart)[0]
        pairs = (
            enclosure.denominator_bounds,
            *enclosure.duu_bounds,
            *enclosure.duv_bounds,
            *enclosure.dvv_bounds,
        )
        for index, (lo, hi) in enumerate(pairs):
            assert Fraction.from_float(fixture[position, 2 * index]) <= lo
            assert hi <= Fraction.from_float(fixture[position, 2 * index + 1])
    tolerance = 0.5
    for broken in (False, True):
        points = mesh.points.copy()
        if broken:
            points[4, 2] += 10.0
        candidate = mesh._replace(points=points)
        result = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords, tolerance, 4)
        decisions = _python_chart_decisions(domain, candidate, rows, tolerance)
        if broken:
            assert decisions == tuple(status == MeshcoreStatus.OK for status in result.item_status[rows])
        else:
            assert decisions == (False, False)
            assert np.all(result.item_status[rows] == MeshcoreStatus.OK)
            assert np.all(result.upper[rows] <= np.asarray((1.0901229543292632, 7.954951288348677)))
            assert np.all(result.upper[rows] >= np.asarray((0.04204418698947044, 0.020824323571858087)))
        for row in rows:
            vertices = points[candidate.triangles[row]]
            for i in range(21):
                for j in range(21 - i):
                    barycentric = np.asarray((1 - (i + j) / 20, i / 20, j / 20))
                    sample = _rational_spline_values(barycentric @ candidate.uv[row])
                    assert np.linalg.norm(sample - barycentric @ vertices) <= result.upper[row]


def test_python_spline_kink_requires_nonzero_interpolation_bound() -> None:
    """A continuous piecewise-linear tent is one unit above its corner plane."""
    controls = np.asarray(
        [[[0, 0, 0], [0, 1, 0]], [[0.5, 0, 1], [0.5, 1, 1]], [[1, 0, 0], [1, 1, 0]]],
        dtype=np.float64,
    )
    surface = BSplineSurfacePatch(
        controls, np.ones((3, 2), dtype=np.float64),
        np.asarray((0, 0, 0.5, 1, 1), dtype=np.float64),
        np.asarray((0, 0, 1, 1), dtype=np.float64), 1, 1,
    )
    domain = _surface_domain(surface)
    chart = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
             (Fraction(0), Fraction(1)))
    bound = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 2),))[0]
    assert bound is not None and bound >= 1


def _degree_one_surface(u_height: float, v_height: float) -> BSplineSurfacePatch:
    """Degree-1 tents in u and v; u = 1/2 and v = 1/2 are C0 knot lines."""
    knots = np.asarray((0, 0, 0.5, 1, 1), dtype=np.float64)
    controls = np.asarray(
        [[(0.5 * i, 0.5 * j, u_height * (i == 1) + v_height * (j == 1)) for j in range(3)]
         for i in range(3)],
        dtype=np.float64,
    )
    return BSplineSurfacePatch(controls, np.ones((3, 3), dtype=np.float64), knots, knots, 1, 1)


def _tent_values(uv: np.ndarray, u_height: float, v_height: float) -> np.ndarray:
    """Independent closed form of `_degree_one_surface`."""
    u, v = uv[..., 0], uv[..., 1]
    z = u_height * (1 - np.abs(2 * u - 1)) + v_height * (1 - np.abs(2 * v - 1))
    return np.stack((u, v, z), axis=-1)


def _square_mesh(
    domain: MeshingDomain, uv_points: np.ndarray, triangles: list[tuple[int, int, int]],
    points: np.ndarray,
) -> tuple[CadFidelityMesh, CadFidelityChords]:
    """A triangulation of the unit square chart with its boundary chords."""
    cells = np.asarray(triangles, dtype=np.int64)
    owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, cell in enumerate(triangles):
        for edge in range(3):
            a, b = cell[edge], cell[(edge + 1) % 3]
            owners.setdefault((min(a, b), max(a, b)), []).append((row, edge))
    edge_rows = np.full((len(triangles), 3), -1, dtype=np.int64)
    chord_use: list[int] = []
    chord_uv: list[np.ndarray] = []
    chord_range: list[tuple[float, float]] = []
    for occurrences in owners.values():
        if len(occurrences) != 1:
            continue
        row, edge = occurrences[0]
        start = uv_points[triangles[row][edge]]
        end = uv_points[triangles[row][(edge + 1) % 3]]
        if start[1] == end[1] == 0:
            use, first, last = 0, start[0], end[0]
        elif start[0] == end[0] == 1:
            use, first, last = 1, start[1], end[1]
        elif start[1] == end[1] == 1:
            use, first, last = 2, 1.0 - start[0], 1.0 - end[0]
        else:
            use, first, last = 3, 1.0 - start[1], 1.0 - end[1]
        edge_rows[row, edge] = len(chord_use)
        chord_use.append(use)
        chord_uv.append(np.stack((start, end)))
        chord_range.append((float(first), float(last)))
    mesh = CadFidelityMesh(
        points, cells, np.zeros(len(cells), dtype=np.int32), uv_points[cells], cells,
        np.arange(len(cells), dtype=np.int64), edge_rows,
    )
    bundle = cad_geometry_bundle(domain)
    use_patch = np.repeat(
        np.arange(bundle.model.surface_nodes.size, dtype=np.int64),
        np.diff(bundle.model.loop_use_offsets),
    )
    chords = CadFidelityChords(
        use_patch, bundle.model.use_ranges, bundle.model.loop_surfaces.size,
        np.asarray(chord_use, dtype=np.int64), np.zeros(len(chord_use), dtype=np.int64),
        np.asarray(chord_uv, dtype=np.float64), np.asarray(chord_range, dtype=np.float64),
    )
    return mesh, chords


def _dense_distance(mesh: CadFidelityMesh, row: int, u_height: float, v_height: float) -> float:
    """Sampled distance, including the triangle's crossings of the knot lines."""
    steps = 64
    weights = [
        (1 - (i + j) / steps, i / steps, j / steps)
        for i in range(steps + 1) for j in range(steps + 1 - i)
    ]
    uv = mesh.uv[row]
    for a in range(3):
        p, q = uv[a], uv[(a + 1) % 3]
        for axis in range(2):
            if (p[axis] - 0.5) * (q[axis] - 0.5) < 0:
                s = (0.5 - p[axis]) / (q[axis] - p[axis])
                weight = [0.0, 0.0, 0.0]
                weight[a], weight[(a + 1) % 3] = 1 - s, s
                weights.append((weight[0], weight[1], weight[2]))
    barycentric = np.asarray(weights, dtype=np.float64)
    exact = _tent_values(barycentric @ uv, u_height, v_height)
    linear = barycentric @ mesh.points[mesh.triangles[row]]
    return float(np.max(np.linalg.norm(exact - linear, axis=1)))


def test_folded_degree_one_spline_uses_clipped_pieces() -> None:
    """The folded degree-1 spline: clipped pieces give the exact distance 0.01.

    Fault caught: bounding the leaf Hessian across the fold (zero) or the
    first-order fallback (about 0.5 here) instead of the clipped pieces.
    """
    height = 0.01
    domain = _surface_domain(_degree_one_surface(height, 0.0))
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    uv_points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    mesh, chords = _square_mesh(
        domain, uv_points, [(0, 1, 2), (0, 2, 3)], _tent_values(uv_points, height, 0.0)
    )
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.1, 0)
    assert result.status == MeshcoreStatus.OK, (result.reasons, result.upper)
    for row in range(2):
        assert _dense_distance(mesh, row, height, 0.0) == pytest.approx(height, rel=1e-12)
        assert height <= result.upper[row] <= height * (1 + 1e-9)
        chart = tuple(
            (Fraction.from_float(float(a)), Fraction.from_float(float(b)))
            for a, b in mesh.uv[row]
        )
        python = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 10),))[0]
        assert python is not None
        print("folded native", result.upper[row], "Python", float(python))
    assert _python_chart_decisions(domain, mesh, np.arange(2), 0.1) == (True, True)


def test_knot_edge_on_line_and_near_parallel_edges() -> None:
    """Edges on a knot line, and near-parallel edges on both sides of the
    clip-conditioning threshold.

    Faults caught: rounding a clip point instead of enclosing it (a missed
    crossing underestimates), and clipping an ill-conditioned crossing.
    """
    height = 0.01
    # Edge on the knot line u = 1/2: each triangle lies in one closed cell.
    domain = _surface_domain(_degree_one_surface(height, 0.0))
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    uv_points = np.asarray(
        ((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)), dtype=np.float64
    )
    triangles = [(0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)]
    mesh, chords = _square_mesh(domain, uv_points, triangles,
                                _tent_values(uv_points, height, 0.0))
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.1, 0)
    assert result.status == MeshcoreStatus.OK, (result.reasons, result.upper)
    for row in range(len(triangles)):
        assert _dense_distance(mesh, row, height, 0.0) <= result.upper[row] <= 1e-12
    # Near-parallel crossings of u = 1/2 and then v = 1/2 by a thin triangle
    # C, B, A. With coarse offsets the crossings are exact or narrowly
    # enclosed and the triangle is clipped at depth 0. With offsets of a few
    # ulps, the u-crossing of AB, 1/2 + 13 * 2^-55, is not a double: its
    # enclosure is a few ulps wide, and the v-crossing of the nearly parallel
    # edge from it to A is enclosed wider than the threshold, so that leaf
    # takes the first-order bound and subdivides.
    domain = _surface_domain(_degree_one_surface(height, height))
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    for a_v, b_v, c_v, conditioned in (
        (0.5 - 2.0**-30, 0.5 + 2.0**-28, 0.5 - 2.0**-26, True),
        (0.5 - 3 * 2.0**-54, 0.5 + 2.0**-50, 0.5 - 2.0**-48, False),
    ):
        uv_points = np.asarray(
            ((0, 0), (1, 0), (1, b_v), (1, 1), (0, 1), (0, a_v), (0, c_v)), dtype=np.float64
        )
        triangles = [(6, 2, 5), (0, 1, 2), (0, 2, 6), (5, 2, 3), (5, 3, 4)]
        mesh, chords = _square_mesh(domain, uv_points, triangles,
                                    _tent_values(uv_points, height, height))
        result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.1, 12)
        assert result.status == MeshcoreStatus.OK, (result.reasons, result.upper)
        for row in range(len(triangles)):
            assert _dense_distance(mesh, row, height, height) <= result.upper[row]
        print("near-parallel", conditioned, "bound", result.upper[0],
              "subdivisions", result.subdivisions[0])
        assert (result.subdivisions[0] == 0) == conditioned


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
    # Native takes the Taylor remainder about the leaf box centre (one quarter
    # of Python's box-extent remainder), so it is never looser than Python;
    # the sampled check above keeps it an enclosure.
    assert all(ratio <= 1.01 for ratio in ratios)
    assert np.all(result.item_status == MeshcoreStatus.OK)


def test_unbounded_coefficients_are_refused() -> None:
    """The lazy bound builder's unhandled nodes get (-inf, inf): every row refuses.

    Fault caught: a rounded binary64 coefficient used where no exact
    enclosure was built.
    """
    domain = _cylinder_domain()
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    node = bundle.model.surface_nodes[0]
    first, last = bundle.model.node_real_offsets[node], bundle.model.node_real_offsets[node + 1]
    lower, upper = lower.copy(), upper.copy()
    lower[first:last], upper[first:last] = -np.inf, np.inf
    mesh, chords = _cylinder_mesh(domain)
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.UNRESOLVED
    assert np.all(result.item_status == MeshcoreStatus.UNRESOLVED)
    assert np.all(np.isin(result.reasons, (2, 3)))


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
    # Moving vertex 4 by 0.75 flips its triangles: V4's orientation, checked
    # by V3, refuses it as input; 0.45 keeps the orientation.
    flipped = mesh.uv.copy()
    flipped[:, :, 0][mesh.triangles == 4] += 0.75
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_cad_fidelity(bundle.model, lower, upper, mesh._replace(uv=flipped), chords, 0.1, 3)
    uv = mesh.uv.copy()
    uv[:, :, 0][mesh.triangles == 4] += 0.45
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
    # The sliver bound is the interpolation bound on the chart triangle: it
    # covers |S(x) - I_T(x)|, and I_T(x) lies on the mesh edge from the pole.
    sliver_points = points[chart_indices[1]]
    for i in range(21):
        for j in range(21 - i):
            weights = np.asarray((1 - (i + j) / 20, i / 20, j / 20), dtype=np.float64)
            source = _analytic_values("sphere", weights @ chart_uv[1])
            assert np.linalg.norm(source - weights @ sliver_points) <= result.upper[1]
    bad_mapping = mesh.chart_to_mesh.copy()
    bad_mapping[1] = 0
    broken = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(chart_to_mesh=bad_mapping), chords, 0.5, 4
    )
    assert broken.status == MeshcoreStatus.UNRESOLVED
    assert np.any((broken.findings["kind"] == 5) & (broken.findings["triangle"] == 1))


_HALF_PI = np.float64(np.pi / 2)
# pi/2 - rho*, rho* the positive root of rho = sin 2 rho, rounded down.
_CAP_LATITUDE = float.fromhex("0x1.3f004dc9db074p-1")


def _pole_region(uv: np.ndarray) -> np.ndarray:
    """V3's pole-region rows: +1 / -1 by pole, 0 elsewhere (the native predicate).

    A row is in a pole's region when a corner lies on the pole line or any
    corner lies in the cap sigma v >= _CAP_LATITUDE (the native kCapLatitude).
    """
    region = np.zeros(uv.shape[0], dtype=np.int64)
    for sigma in (1, -1):
        on = (uv[:, :, 1] == sigma * _HALF_PI).any(axis=1)
        cap = (sigma * uv[:, :, 1] >= _CAP_LATITUDE).any(axis=1)
        region[on | cap] = sigma
    return region


def _pole_chart_deviation(
    surface: Callable[[np.ndarray], np.ndarray],
    uv: np.ndarray,
    vertices: np.ndarray,
    sigma: int,
    n: int,
    pole_v: float | None = None,
) -> float:
    """Sampled sup |S - I_T| over the pole-chart triangle of one row.

    Corners map to rho (cos u, sin u), rho = pi/2 - sigma v on a sphere, or
    rho = sigma (v - pole_v) at a disk centre (`pole_v` given); a pole corner
    maps to the origin. S at a pole-chart point is the surface at its chart
    point.
    """
    if pole_v is None:
        rho = _HALF_PI - sigma * uv[:, 1]
        pole = uv[:, 1] == sigma * _HALF_PI
    else:
        rho = sigma * (uv[:, 1] - pole_v)
        pole = uv[:, 1] == pole_v
    corners = np.where(pole[:, None], 0.0, rho[:, None] * np.stack(
        (np.cos(uv[:, 0]), np.sin(uv[:, 0])), axis=1))
    worst = 0.0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            weights = np.asarray((1 - (i + j) / n, i / n, j / n))
            x, y = weights @ corners
            radius = np.hypot(x, y)
            v = sigma * (_HALF_PI - radius) if pole_v is None else pole_v + sigma * radius
            exact = np.asarray(surface(np.asarray((np.arctan2(y, x), v))), dtype=np.float64)
            worst = max(worst, float(np.linalg.norm(exact.reshape(3) - weights @ vertices)))
    return worst


def _brep_sphere_grid(
    columns: int, rows: int
) -> tuple[MeshingDomain, CadFidelityMesh, CadFidelityChords]:
    """A chart grid of the BRep unit sphere with rooted pole p-curves."""
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_sphere

    domain = MeshingDomain.from_brep(
        brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    )
    tau, half = 2.0 * np.pi, 0.5 * np.pi
    u = np.asarray([tau * i / columns for i in range(columns)] + [tau], dtype=np.float64)
    v = np.asarray([-half + np.pi * j / rows for j in range(rows)] + [half], dtype=np.float64)
    south, north = 0, 1

    def physical(i: int, j: int) -> int:
        if j == 0:
            return south
        if j == rows:
            return north
        return 2 + (j - 1) * columns + i % columns

    points = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    for j in range(1, rows):
        for i in range(columns):
            points.append((-np.sin(u[i]) * np.cos(v[j]), np.cos(u[i]) * np.cos(v[j]),
                           np.sin(v[j])))
    chart: list[tuple[int, int, int]] = []
    chart_uv: list[np.ndarray] = []
    edge_chord: list[list[int]] = []
    chord_use: list[int] = []
    chord_uv: list[np.ndarray] = []
    chord_range: list[tuple[float, float]] = []
    mesh_rows: list[tuple[int, int, int]] = []
    chart_to_mesh: list[int] = []

    def add_chord(use: int, start: np.ndarray, end: np.ndarray, first: float, last: float) -> int:
        chord_use.append(use)
        chord_uv.append(np.stack((start, end)))
        chord_range.append((first, last))
        return len(chord_use) - 1

    for j in range(rows):
        for i in range(columns):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            for local in ((0, 1, 2), (0, 2, 3)):
                cell = tuple(corners[k] for k in local)
                uv = np.asarray([(u[a], v[b]) for a, b in cell], dtype=np.float64)
                ids = (physical(*cell[0]), physical(*cell[1]), physical(*cell[2]))
                chords = [-1, -1, -1]
                for edge in range(3):
                    (a0, b0), (a1, b1) = cell[edge], cell[(edge + 1) % 3]
                    start, end = uv[edge], uv[(edge + 1) % 3]
                    if b0 == b1 == 0:
                        chords[edge] = add_chord(0, start, end, u[a0], u[a1])
                    elif a0 == a1 == columns:
                        chords[edge] = add_chord(1, start, end, v[b0], v[b1])
                    elif b0 == b1 == rows:
                        chords[edge] = add_chord(2, start, end, u[a0], u[a1])
                    elif a0 == a1 == 0:
                        chords[edge] = add_chord(3, start, end, v[b0], v[b1])
                chart.append(ids)
                chart_uv.append(uv)
                edge_chord.append(chords)
                if len(set(ids)) == 3:
                    chart_to_mesh.append(len(mesh_rows))
                    mesh_rows.append(ids)
                else:
                    chart_to_mesh.append(-1)
    mesh = CadFidelityMesh(
        np.asarray(points, dtype=np.float64),
        np.asarray(chart, dtype=np.int64),
        np.zeros(len(chart), dtype=np.int32),
        np.asarray(chart_uv, dtype=np.float64),
        np.asarray(mesh_rows, dtype=np.int64),
        np.asarray(chart_to_mesh, dtype=np.int64),
        np.asarray(edge_chord, dtype=np.int64),
    )
    bundle = cad_geometry_bundle(domain)
    chords_table = CadFidelityChords(
        np.zeros(bundle.model.use_ranges.shape[0], dtype=np.int64),
        bundle.model.use_ranges,
        bundle.model.loop_surfaces.size,
        np.asarray(chord_use, dtype=np.int64),
        np.zeros(len(chord_use), dtype=np.int64),
        np.asarray(chord_uv, dtype=np.float64),
        np.asarray(chord_range, dtype=np.float64),
    )
    return domain, mesh, chords_table


def test_brep_sphere_rooted_pole_pcurves() -> None:
    """Rooted pole latitudes and 2π seam gauges are enclosed, then certified."""
    domain, mesh, chords = _brep_sphere_grid(8, 12)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    offsets = bundle.model.node_real_offsets
    for use, sign in ((0, -1), (2, 1)):
        node = bundle.model.use_pcurve_nodes[use]
        child = bundle.model.node_integers[bundle.model.node_integer_offsets[node]]
        angle = offsets[child]
        # The rooted latitude encloses the exact ±π/2, not its binary64 shadow.
        assert Fraction.from_float(float(lower[angle])) < sign * Fraction(
            "1.5707963267948966192313216916397514"
        ) < Fraction.from_float(float(upper[angle]))
    seam = bundle.model.use_pcurve_nodes[1]
    shift = offsets[seam] + 4
    tau = 2 * Fraction("3.14159265358979323846264338327950288")
    assert Fraction.from_float(float(lower[shift])) < tau < Fraction.from_float(float(upper[shift]))
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.5, 4)
    assert result.findings.size == 0, result.findings
    assert result.status == MeshcoreStatus.OK
    assert np.all(result.item_status == MeshcoreStatus.OK)

    def sphere(uv: np.ndarray) -> np.ndarray:
        return np.asarray((-np.sin(uv[0]) * np.cos(uv[1]), np.cos(uv[0]) * np.cos(uv[1]),
                           np.sin(uv[1])))

    # Pole slivers too: I_T interpolates the collapsed corners, so it lies on
    # the mesh edge from the pole. Pole-region rows are associated in the pole
    # chart, so their same-parameter reference is taken there.
    region = _pole_region(mesh.uv)
    for row in range(mesh.triangles.shape[0]):
        vertices = mesh.points[mesh.triangles[row]]
        if region[row] != 0:
            sampled = _pole_chart_deviation(sphere, mesh.uv[row], vertices, int(region[row]), 10)
            assert sampled <= result.upper[row]
            continue
        for i in range(11):
            for j in range(11 - i):
                weights = np.asarray((1 - (i + j) / 10, i / 10, j / 10))
                exact = sphere(weights @ mesh.uv[row])
                assert np.linalg.norm(exact - weights @ vertices) <= result.upper[row]
    interior = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    assert _python_chart_decisions(domain, mesh, interior, 0.5) == tuple(
        status == MeshcoreStatus.OK for status in result.item_status[interior]
    )


@pytest.mark.parametrize("distance", [Fraction(1, 7), Fraction(1, 8)])
def test_offset_cone_encloses_sine_and_cosine(distance: Fraction) -> None:
    """Offset cone coefficients enclose s d sin(a), s d cos(a) of the exact a."""
    import mpmath

    cone = _analytic_surface("cone")
    assert isinstance(cone, ConePatch)
    surface = OffsetSurface(cone, distance)
    sign = surface.frame_sign
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    start = bundle.model.node_real_offsets[bundle.model.surface_nodes[0]]
    mpmath.mp.prec = 200
    angle = mpmath.mpf(Fraction.from_float(0.2).numerator) / Fraction.from_float(0.2).denominator
    scaled = mpmath.mpf(sign * distance.numerator) / distance.denominator
    expected = {2: -scaled * mpmath.sin(angle), 12: mpmath.mpf(0.4) + scaled * mpmath.cos(angle)}
    for index, value in expected.items():
        assert mpmath.mpf(float(lower[start + index])) <= value <= mpmath.mpf(float(upper[start + index]))
    assert lower[start + 13] == upper[start + 13] == 0.2
    mesh, chords = _cylinder_mesh(domain, "cone")
    radius_shift = float(sign * distance) * np.cos(0.2)
    height_shift = -float(sign * distance) * np.sin(0.2)

    def offset_values(uv: np.ndarray) -> np.ndarray:
        radius = 0.4 + uv[..., 1] * np.tan(0.2) + radius_shift
        return np.stack((radius * np.cos(uv[..., 0]), radius * np.sin(uv[..., 0]),
                         uv[..., 1] + height_shift), axis=-1)

    uv_points = np.zeros((mesh.points.shape[0], 2))
    uv_points[mesh.triangles.reshape(-1)] = mesh.uv.reshape((-1, 2))
    candidate = mesh._replace(points=offset_values(uv_points))
    result = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.OK
    rows = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    dyadic = distance.denominator & (distance.denominator - 1) == 0
    python = _python_chart_decisions(
        domain, candidate, rows, 0.5, refusal=None if dyadic else _EXACT_CARRIER_REFUSAL
    )
    assert python == ((True,) * rows.size if dyadic else (False,) * rows.size)
    for row in rows:
        vertices = candidate.points[candidate.triangles[row]]
        for i in range(11):
            for j in range(11 - i):
                weights = np.asarray((1 - (i + j) / 10, i / 10, j / 10))
                exact = offset_values(weights @ candidate.uv[row])
                assert np.linalg.norm(exact - weights @ vertices) <= result.upper[row]


def test_offset_line_and_rooted_pcurve_enclosures() -> None:
    """Line offsets enclose irrational unit normals; rooted phases enclose 2π turns."""
    from decimal import Decimal

    from phydrax.geometry.brep._boolean_core import AffinePCurve
    from phydrax.geometry.brep._patches import (
        ConicRootLinePCurve,
        ExactLineCurve,
        OffsetCurve,
        TorusRootedCirclePCurve,
    )
    from phydrax.meshing.providers._native_cad_bundle import (
        _computed_coefficient_intervals,
        _NodeBank,
    )

    def interval(entity: object) -> tuple[tuple[Fraction, Fraction], ...]:
        bank = _NodeBank()
        index = bank.node(entity)
        computed = _computed_coefficient_intervals(bank, entity, index)
        assert computed is not None
        pairs = [
            (lo, hi) for lo, hi in computed[index]
            if isinstance(lo, Fraction) and isinstance(hi, Fraction)
        ]
        assert len(pairs) == len(computed[index])
        return tuple(pairs)

    with localcontext() as context:
        context.prec = 60
        root2 = Decimal(2).sqrt()
        planar = interval(OffsetCurve(LineCurve((0, 0), (1, 1)), Fraction(1, 7)))
        shift = Decimal(1) / (Decimal(7) * root2)
        for (lo, hi), exact in zip(planar, (shift, -shift, Decimal(1), Decimal(1)), strict=True):
            assert Decimal(lo.numerator) / lo.denominator <= exact
            assert exact <= Decimal(hi.numerator) / hi.denominator
        spatial = interval(
            OffsetCurve(ExactLineCurve((0, 0, 0), (1, 1, 0)), Fraction(1, 7), (0, 0, 1))
        )
        for (lo, hi), exact in zip(
            spatial, (shift, -shift, Decimal(0), 1 + shift, 1 - shift, Decimal(0)), strict=True
        ):
            assert Decimal(lo.numerator) / lo.denominator <= exact
            assert exact <= Decimal(hi.numerator) / hi.denominator
    domain, _, _ = _brep_sphere_grid(4, 4)
    pole = domain.patches[0].loops[0][2].pcurve
    if not (isinstance(pole, AffinePCurve) and isinstance(pole.curve, ConicRootLinePCurve)):
        pytest.fail("A BRep sphere pole p-curve is an affine conic-root line.")
    root = pole.curve.root
    rooted = interval(TorusRootedCirclePCurve(root, Fraction(3, 4), -1))
    pi = Fraction("3.14159265358979323846264338327950288")
    assert rooted[0][0] < Fraction(3, 2) * pi < rooted[0][1]
    assert rooted[1][0] < pi / 2 < rooted[1][1]


def test_brep_torus_coefficients_are_finite() -> None:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_torus

    domain = MeshingDomain.from_brep(
        brep_torus(1.0, 0.25, coordinate_contract=SpatialCoordinateContract.si())
    )
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= bundle.model.node_reals)
    assert np.all(bundle.model.node_reals <= upper)


def _tspline_surface() -> AbstractSurfacePatch:
    """A rational cubic T-spline on non-dyadic knots k/3 with rational weights."""
    from phydrax._interpolation import TMesh2D
    from phydrax.geometry.brep._tspline import TSplineSurfacePatch

    knots = tuple(Fraction(index, 3) for index in range(4))
    mesh = TMesh2D.tensor_product(knots, knots)
    parameters = tuple(anchor.local_knots.greville_parameter for anchor in mesh.anchors)
    points = tuple((u, v, (u - Fraction(1, 2)) ** 2 + u * v / 3) for u, v in parameters)
    weights = tuple(1 + u / 3 + v / 5 for u, v in parameters)
    return TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in points]),
        np.asarray([float(value) for value in weights]),
        exact_control_points=points,
        exact_weights=weights,
    )


def test_tspline_exact_coefficients_and_fidelity() -> None:
    """Exact non-dyadic extraction data; native bound >= dense distance; Python parity."""
    import jax.numpy as jnp

    from phydrax.geometry.brep._tspline import TSplineSurfacePatch

    surface = _tspline_surface()
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    node = bundle.model.surface_nodes[0]
    start, stop = bundle.model.node_real_offsets[node], bundle.model.node_real_offsets[node + 1]
    assert np.all(np.isfinite(lower[start:stop])) and np.all(np.isfinite(upper[start:stop]))
    if not isinstance(surface, TSplineSurfacePatch):
        pytest.fail("The fixture is a T-spline surface.")
    realization = surface.realization
    exact_bounds = [value for piece in realization.extractions for value in piece.parameter_bounds]
    assert any(value.denominator == 3 for value in exact_bounds)
    for offset, value in enumerate(exact_bounds):
        assert Fraction.from_float(float(lower[start + offset])) <= value
        assert value <= Fraction.from_float(float(upper[start + offset]))
    operators = start + len(exact_bounds)
    first = realization.extractions[0].exact_extraction_operator
    for row, values in enumerate(first):
        for column, value in enumerate(values):
            index = operators + 16 * row + column
            assert Fraction.from_float(float(lower[index])) <= value
            assert value <= Fraction.from_float(float(upper[index]))
    mesh, chords = _cylinder_mesh(domain)
    uv_points = np.zeros((mesh.points.shape[0], 2))
    uv_points[mesh.triangles.reshape(-1)] = mesh.uv.reshape((-1, 2))
    candidate = mesh._replace(points=np.asarray(surface.evaluate(jnp.asarray(uv_points))))
    result = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords, 0.5, 4)
    assert result.status == MeshcoreStatus.OK
    assert result.findings.size == 0
    rows = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    assert _python_chart_decisions(domain, candidate, rows, 0.5) == tuple(
        status == MeshcoreStatus.OK for status in result.item_status[rows]
    )
    for row in range(mesh.triangles.shape[0]):
        vertices = candidate.points[candidate.triangles[row]]
        weights = np.asarray(
            [(1 - (i + j) / 20, i / 20, j / 20) for i in range(21) for j in range(21 - i)]
        )
        exact = np.asarray(surface.evaluate(jnp.asarray(weights @ candidate.uv[row])))
        distance = np.linalg.norm(exact - weights @ vertices, axis=1)
        assert np.all(distance <= result.upper[row])
    for row in rows:
        chart = tuple(
            (Fraction.from_float(float(a)), Fraction.from_float(float(b)))
            for a, b in candidate.uv[row]
        )
        reference = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 2),))[0]
        assert reference is not None
        # Native refines until it meets the target: certify within Python's bound.
        tight = verify_cad_fidelity(bundle.model, lower, upper, candidate, chords,
                                    float(reference), 6)
        assert tight.item_status[row] == MeshcoreStatus.OK
        assert tight.upper[row] <= float(reference)
        print("tspline native/Python depth 0", result.upper[row] / float(reference),
              "within Python bound", tight.upper[row] / float(reference),
              "subdivisions", tight.subdivisions[row])


def test_intersection_pcurve_trimmed_face() -> None:
    """A face bounded by one traced cylinder/cylinder intersection p-curve."""
    import jax.numpy as jnp

    from phydrax.geometry.brep._intersection_curve import IntersectionPCurve

    domain = MeshingDomain.from_brep(_crossing_cylinders_model())
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= bundle.model.node_reals) and np.all(bundle.model.node_reals <= upper)
    patch = domain.patches[0]
    (use,) = patch.loops[0]
    assert isinstance(use.pcurve, IntersectionPCurve) and use.sense == -1
    samples = 32
    parameters = np.linspace(use.first, use.last, samples + 1)[:-1]
    # Loop order: a reversed use runs the p-curve from last to first.
    parameters = parameters[::-1]
    boundary = np.asarray(use.pcurve.evaluate(jnp.asarray(parameters)))
    centre = boundary.mean(axis=0)
    inner = centre + 0.5 * (boundary - centre)
    uv_points = np.vstack((centre[None], inner, boundary))
    points = np.asarray(patch.surface.evaluate(jnp.asarray(uv_points)))
    triangles: list[tuple[int, int, int]] = []
    edge_chord: list[list[int]] = []
    chord_uv: list[np.ndarray] = []
    chord_range: list[tuple[float, float]] = []
    for i in range(samples):
        j = (i + 1) % samples
        r_i, r_j, p_i, p_j = 1 + i, 1 + j, 1 + samples + i, 1 + samples + j
        triangles += [(0, r_i, r_j), (r_i, p_i, p_j), (r_i, p_j, r_j)]
        edge_chord += [[-1, -1, -1], [-1, len(chord_uv), -1], [-1, -1, -1]]
        # The closing chord wraps from t = first to the same point at t = last.
        start = use.last if parameters[i] == use.first else parameters[i]
        chord_uv.append(np.stack((uv_points[p_i], uv_points[p_j])))
        chord_range.append((float(start), float(parameters[j])))
    cells = np.asarray(triangles, dtype=np.int64)
    mesh = CadFidelityMesh(
        points, cells, np.zeros(len(cells), dtype=np.int32), uv_points[cells], cells,
        np.arange(len(cells), dtype=np.int64), np.asarray(edge_chord, dtype=np.int64),
    )
    # Only face 0 is meshed: V3 checks every model use and surface is
    # covered, so it is given the model of face 0 alone (same nodes and
    # coefficients, no regions).
    count = int(bundle.model.loop_use_offsets[1])
    model = bundle.model._replace(
        surface_nodes=bundle.model.surface_nodes[:1],
        surface_orientations=bundle.model.surface_orientations[:1],
        surface_periods=bundle.model.surface_periods[:1],
        surface_parameter_domains=bundle.model.surface_parameter_domains[:1],
        loop_use_offsets=bundle.model.loop_use_offsets[:2],
        loop_surfaces=bundle.model.loop_surfaces[:1],
        use_kinds=bundle.model.use_kinds[:count],
        use_entities=bundle.model.use_entities[:count],
        use_senses=bundle.model.use_senses[:count],
        use_ranges=bundle.model.use_ranges[:count],
        use_pcurve_nodes=bundle.model.use_pcurve_nodes[:count],
        region_surface_offsets=bundle.model.region_surface_offsets[:1],
        region_surfaces=bundle.model.region_surfaces[:0],
        region_orientations=bundle.model.region_orientations[:0],
    )
    chords = CadFidelityChords(
        np.zeros(count, dtype=np.int64), model.use_ranges, 1,
        np.zeros(len(chord_uv), dtype=np.int64), np.zeros(len(chord_uv), dtype=np.int64),
        np.asarray(chord_uv, dtype=np.float64), np.asarray(chord_range, dtype=np.float64),
    )
    result = verify_cad_fidelity(model, lower, upper, mesh, chords, 0.1, 6)
    assert result.findings.size == 0, result.findings
    assert result.status == MeshcoreStatus.OK, (result.reasons, result.upper)
    rows = np.arange(len(cells), dtype=np.int64)
    assert _python_chart_decisions(domain, mesh, rows, 0.1) == tuple(
        status == MeshcoreStatus.OK for status in result.item_status
    )
    weights = np.asarray(
        [(1 - (a + b) / 12, a / 12, b / 12) for a in range(13) for b in range(13 - a)]
    )
    for row, cell in enumerate(cells):
        exact = np.asarray(patch.surface.evaluate(jnp.asarray(weights @ mesh.uv[row])))
        assert np.all(np.linalg.norm(exact - weights @ points[cell], axis=1) <= result.upper[row])
    for chord, (first, last) in enumerate(chord_range):
        row = 3 * chord + 1
        fractions = np.linspace(0.0, 1.0, 33)
        arc_uv = np.asarray(use.pcurve.evaluate(jnp.asarray(first + fractions * (last - first))))
        arc = np.asarray(patch.surface.evaluate(jnp.asarray(arc_uv)))
        a, b = points[cells[row][1]], points[cells[row][2]]
        straight = a[None] + fractions[:, None] * (b - a)[None]
        assert np.all(np.linalg.norm(arc - straight, axis=1) <= result.upper[row])


def _crossing_cylinders_model() -> BRepModel:
    """Intersection of a vertical and a transverse cylinder (traced p-curves)."""
    import phydrax as phx

    contract = phx.SpatialCoordinateContract.si()
    vertical = phx.geometry.brep.brep_cylinder(
        1.0, 2.4, base_center=(0.0, 0.0, -1.2), axis=(0.0, 0.0, 1.0),
        coordinate_contract=contract, source_id="qualification:native-fidelity-vertical",
    )
    transverse = phx.geometry.brep.brep_cylinder(
        0.6, 3.0, base_center=(-1.5, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
        coordinate_contract=contract, source_id="qualification:native-fidelity-transverse",
    )
    return phx.geometry.brep.boolean_brep(
        vertical, transverse, "intersection", policy=phx.geometry.brep.BRepBooleanPolicy()
    ).model


@pytest.mark.parametrize("kind", ["sphere", "cone", "cylinder"])
def test_pole_sliver_bound_is_tight(kind: str) -> None:
    """Collapsed pole slivers of real face output certify within 1.5x of sampling.

    A sliver bound below the sampled |S - I_T| would be unsound; above 1.5x it
    is the old first-order cap (6-8x on the h = 0.2 sphere). Sphere slivers lie
    in V3's pole region, so their reference is sampled in the pole chart.
    Fault caught: the first-order radial cap or a missing tight refinement.
    """
    import jax.numpy as jnp

    from phydrax._meshcore import cad_mesh
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_cone, brep_cylinder, brep_sphere

    contract = SpatialCoordinateContract.si()
    brep, h = {
        "sphere": (brep_sphere(1.0, coordinate_contract=contract), 0.2),
        "cone": (brep_cone(1.0, 0.0, 1.0, coordinate_contract=contract), 0.15),
        "cylinder": (brep_cylinder(1.0, 1.0, coordinate_contract=contract), 0.2),
    }[kind]
    domain = MeshingDomain.from_brep(brep)
    bundle = cad_geometry_bundle(domain)
    model = bundle.model
    faces = model.surface_nodes.size
    meshed = cad_mesh(model, np.full(faces, h), np.full(faces, 1e6),
                      np.full(model.curve_nodes.size, np.inf))
    assert meshed.status == MeshcoreStatus.OK
    chart = meshed.chart
    slivers = np.flatnonzero(chart.chart_to_mesh < 0)
    assert slivers.size > 0
    n = 24
    weights = np.asarray(
        [(1 - (i + j) / n, i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)],
        dtype=np.float64,
    )
    sampled = np.zeros(slivers.size, dtype=np.float64)
    region = _pole_region(chart.uv)
    flat = np.zeros(slivers.size, dtype=bool)
    for position, row in enumerate(slivers):
        surface = domain.patches[int(chart.patches[row])].surface
        vertices = meshed.points[chart.triangles[row]]

        def evaluate(c: np.ndarray, s: AbstractSurfacePatch = surface) -> np.ndarray:
            return np.asarray(s.evaluate(jnp.asarray(c)))

        if kind == "sphere":
            assert region[row] != 0
            sampled[position] = _pole_chart_deviation(
                evaluate, chart.uv[row], vertices, int(region[row]), n
            )
            continue
        # A disk centre (planar face; V3 certifies the frame exactly): the
        # sliver's two pole corners share v; its reference is the
        # disk-centre chart, where the deviation is rounding only.
        ids = chart.triangles[row]
        pair = [(a, b) for a in range(3) for b in range(a + 1, 3) if ids[a] == ids[b]][0]
        pole_v = float(chart.uv[row, pair[0], 1])
        other = chart.uv[row, 3 - pair[0] - pair[1], 1]
        sigma = 1 if other > pole_v else -1
        probe = np.asarray(evaluate(np.asarray([[0.0, pole_v], [0.0, pole_v + sigma * 0.1],
                                                [1.0, pole_v + sigma * 0.1],
                                                [2.0, pole_v + sigma * 0.2]])))
        normal = np.cross(probe[1] - probe[0], probe[2] - probe[0])
        if abs(float(np.dot(normal, probe[3] - probe[0]))) <= 1e-12 * float(np.linalg.norm(normal)):
            flat[position] = True
            sampled[position] = _pole_chart_deviation(
                evaluate, chart.uv[row], vertices, sigma, n, pole_v)
            continue
        source = np.asarray(surface.evaluate(weights @ chart.uv[row]), dtype=np.float64)
        linear = weights @ vertices
        sampled[position] = np.max(np.linalg.norm(source - linear, axis=1))
    lower, upper = coefficient_bounds(bundle, domain)
    mesh = CadFidelityMesh(meshed.points, chart.triangles, chart.patches, chart.uv,
                           meshed.triangles.astype(np.int64), chart.chart_to_mesh,
                           chart.edge_chord)
    chords = CadFidelityChords(chart.use_patch, chart.use_range, len(model.loop_surfaces),
                               chart.chord_use, chart.chord_loop, chart.chord_uv,
                               chart.chord_range)
    result = verify_cad_fidelity(model, lower, upper, mesh, chords, 1.0, 8)
    assert result.status == MeshcoreStatus.OK
    assert np.all(result.upper[slivers] >= sampled)
    # Disk-centre slivers are rounding-level both ways; the tightness check
    # applies to curved slivers.
    ratio = result.upper[slivers][~flat] / sampled[~flat]
    assert np.all(ratio <= 1.5), ratio


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_sphere_pole_chart_rows(theta: float) -> None:
    """Fan and band rows of the h = 0.2 unit sphere certify in the pole chart.

    At tolerance h^2 / 4 the uv chart refused every fan row (3.85x the
    tolerance) and the ring band (3.18x). In the pole chart every pole-region
    row certifies, and its bound covers both the dense same-parameter
    deviation in the pole chart and the exact distance | |p| - 1 | of its
    mesh points. A displaced pole vertex and a displaced band vertex refuse
    their rows. Fault caught: a pole chart that skips rows, uses other
    corners than the row's own, or certifies a displaced vertex.
    """
    import jax.numpy as jnp

    from phydrax._meshcore import cad_mesh
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_sphere

    domain = MeshingDomain.from_brep(
        brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    )
    bundle = cad_geometry_bundle(domain)
    model = bundle.model
    h, tolerance = 0.2, 0.01
    meshed = cad_mesh(model, np.full(model.surface_nodes.size, h),
                      np.full(model.surface_nodes.size, tolerance),
                      np.full(model.curve_nodes.size, tolerance), tolerance_fraction=theta)
    assert meshed.status == MeshcoreStatus.OK
    chart = meshed.chart
    lower, upper = coefficient_bounds(bundle, domain)
    chords = CadFidelityChords(chart.use_patch, chart.use_range, len(model.loop_surfaces),
                               chart.chord_use, chart.chord_loop, chart.chord_uv,
                               chart.chord_range)

    def verify(points: np.ndarray) -> CadFidelityResult:
        mesh = CadFidelityMesh(points, chart.triangles, chart.patches, chart.uv,
                               meshed.triangles.astype(np.int64), chart.chart_to_mesh,
                               chart.edge_chord)
        return verify_cad_fidelity(model, lower, upper, mesh, chords, tolerance, 8)

    result = verify(meshed.points)
    region = _pole_region(chart.uv)
    rows = np.flatnonzero(region != 0)
    fan = rows[(chart.uv[rows, :, 1] == region[rows, None] * _HALF_PI).any(axis=1)]
    band = np.setdiff1d(rows, fan)
    assert fan.size > 0 and band.size > 0
    assert np.all(result.item_status[rows] == MeshcoreStatus.OK)
    assert np.all(result.upper[rows] <= tolerance)
    if theta == 0.5:
        assert result.status == MeshcoreStatus.OK
    n = 16
    weights = np.asarray(
        [(1 - (i + j) / n, i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)],
        dtype=np.float64,
    )
    sigma = region[rows].astype(np.float64)
    uv = chart.uv[rows]
    rho = _HALF_PI - sigma[:, None] * uv[:, :, 1]
    pole = uv[:, :, 1] == sigma[:, None] * _HALF_PI
    x = np.where(pole, 0.0, rho * np.cos(uv[:, :, 0])) @ weights.T
    y = np.where(pole, 0.0, rho * np.sin(uv[:, :, 0])) @ weights.T
    points = np.stack((np.arctan2(y, x), sigma[:, None] * (_HALF_PI - np.hypot(x, y))), axis=-1)
    surface = domain.patches[0].surface
    exact = np.asarray(surface.evaluate(jnp.asarray(points.reshape(-1, 2))), dtype=np.float64).reshape(
        rows.size, weights.shape[0], 3)
    linear = np.einsum("kv,rvd->rkd", weights, meshed.points[chart.triangles[rows]])
    sampled = np.linalg.norm(exact - linear, axis=-1).max(axis=1)
    distance = np.abs(np.linalg.norm(linear, axis=-1) - 1.0).max(axis=1)
    assert np.all(result.upper[rows] >= sampled)
    assert np.all(result.upper[rows] >= distance)
    # Displaced vertices: the north pole (fan rows) and a band vertex.
    north_row = fan[region[fan] == 1][0]
    north = int(chart.triangles[north_row][np.flatnonzero(chart.uv[north_row, :, 1] == _HALF_PI)[0]])
    ring = int(np.setdiff1d(chart.triangles[band[region[band] == 1]], chart.triangles[fan])[0])
    for vertex in (north, ring):
        shifted = meshed.points.copy()
        shifted[vertex] *= 1.05
        moved = verify(shifted)
        touching = np.flatnonzero((chart.triangles == vertex).any(axis=1))
        assert np.all(moved.item_status[touching] != MeshcoreStatus.OK)
