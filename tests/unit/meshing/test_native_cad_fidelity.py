# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent source-coefficient checks for native CAD fidelity."""

from __future__ import annotations

import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

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
            tuple(Fraction.from_float(float(value)) for value in vertex)
            for vertex in mesh.uv[row]
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
                ids = tuple(physical(a, b) for a, b in cell)
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

    for row in range(mesh.triangles.shape[0]):
        vertices = mesh.points[mesh.triangles[row]]
        for i in range(11):
            for j in range(11 - i):
                weights = np.asarray((1 - (i + j) / 10, i / 10, j / 10))
                exact = sphere(weights @ mesh.uv[row])
                if mesh.chart_to_mesh[row] < 0:
                    reference = vertices[0] if np.all(vertices[0] == vertices[1]) else vertices[2]
                    distance = np.linalg.norm(exact - reference)
                else:
                    distance = np.linalg.norm(exact - weights @ vertices)
                assert distance <= result.upper[row]
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

    from phydrax.geometry.brep._patches import (
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
        return tuple(
            (Fraction(lo), Fraction(hi)) for lo, hi in computed[index]  # type: ignore[arg-type]
        )

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
    root = pole.curve.root  # type: ignore[attr-defined]
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
    surface = _tspline_surface()
    domain = _surface_domain(surface)
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    node = bundle.model.surface_nodes[0]
    start, stop = bundle.model.node_real_offsets[node], bundle.model.node_real_offsets[node + 1]
    assert np.all(np.isfinite(lower[start:stop])) and np.all(np.isfinite(upper[start:stop]))
    realization = surface.realization  # type: ignore[attr-defined]
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
    candidate = mesh._replace(points=np.asarray(surface.evaluate(uv_points)))
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
        exact = np.asarray(surface.evaluate(weights @ candidate.uv[row]))
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
