# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent source-coefficient checks for native CAD fidelity."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax._meshcore import (
    CadFidelityChords,
    CadFidelityMesh,
    MeshcoreStatus,
    verify_cad_fidelity,
)
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import CylinderPatch, LineCurve
from phydrax.meshing.providers._native_cad_bundle import (
    cad_geometry_bundle,
    coefficient_bounds,
)


def _cylinder_domain() -> MeshingDomain:
    """Two oppositely oriented occurrences with exact cylinder coefficients."""
    surface = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), float(Fraction(1, 3)), radius_exact=Fraction(1, 3),
    )
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


def _cylinder_mesh(
    domain: MeshingDomain,
) -> tuple[CadFidelityMesh, CadFidelityChords]:
    uv_points = np.asarray(
        [(0.5 * i, 0.5 * j) for j in range(3) for i in range(3)], dtype=np.float64
    )
    points = np.column_stack(
        (np.cos(uv_points[:, 0]) / 3, np.sin(uv_points[:, 0]) / 3, uv_points[:, 1])
    )
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


def test_interior_cylinder_bounds_against_python_and_sampling() -> None:
    """Enclose the original triangle interpolation after subdivision."""
    domain = _cylinder_domain()
    bundle = cad_geometry_bundle(domain)
    lower, upper = coefficient_bounds(bundle, domain)
    mesh, chords = _cylinder_mesh(domain)
    result = verify_cad_fidelity(bundle.model, lower, upper, mesh, chords, 0.1, 4)
    assert result.status == MeshcoreStatus.UNRESOLVED
    assert result.findings.size == 0
    interior = np.flatnonzero(np.all(mesh.edge_chord < 0, axis=1))
    assert interior.size > 0
    ratios: list[float] = []
    for row in interior:
        assert result.item_status[row] == MeshcoreStatus.OK
        chart = tuple(
            (Fraction.from_float(float(vertex[0])), Fraction.from_float(float(vertex[1])))
            for vertex in mesh.uv[row]
        )
        reference = domain.adaptive_interpolation_bounds(0, (chart,), (Fraction(1, 10),))[0]
        assert reference is not None and reference > 0
        ratios.append(result.upper[row] / float(reference))
        vertices = mesh.points[mesh.triangles[row]]
        for i in range(21):
            for j in range(21 - i):
                weights = np.asarray((1 - (i + j) / 20, i / 20, j / 20), dtype=np.float64)
                u, v = weights @ mesh.uv[row]
                exact = np.asarray((np.cos(u) / 3, np.sin(u) / 3, v))
                assert np.linalg.norm(exact - weights @ vertices) <= result.upper[row]
    print("native/Python adaptive bound ratios:", ratios)
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
    uv = mesh.uv.copy()
    uv[:, :, 0][mesh.triangles == 4] += 0.75
    perturbed = verify_cad_fidelity(
        bundle.model, lower, upper, mesh._replace(uv=uv), chords, 0.1, 3
    )
    assert perturbed.findings.size == 0
    assert np.all(perturbed.item_status[interior] == MeshcoreStatus.UNRESOLVED)
    assert np.all(perturbed.reasons[interior] == 1)
