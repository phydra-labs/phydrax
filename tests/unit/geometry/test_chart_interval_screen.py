"""Public source-fidelity decisions checked against exact affine-chart geometry."""

from fractions import Fraction

import numpy as np

from phydrax.discretization import lagrange_element
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
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry._source_fidelity_certificate import SourceFidelityCertificate
from phydrax.geometry.brep._patches import LineCurve, PlanePatch


_TRIANGLES = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
_IDS = np.asarray((101, 102, 103, 104), dtype=np.int64)
_REFERENCE = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))


def _case(
    center: tuple[float, float],
    heights: np.ndarray,
    order: np.ndarray,
    *,
    curvature: float = 0.0,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    MeshingDomainBoundarySource,
    SourceBoundaryChartWitness,
]:
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1), center), dtype=np.float64)
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(0, 3),
        ),
        4,
        source_id="interval-screen-square",
        source_revision="interval-screen-square:1",
        corner_points=np.column_stack((uv[:4], np.zeros(4))),
    )
    source = MeshingDomainBoundarySource(domain, (0,))
    mesh = CellMesh(
        np.column_stack((uv, heights)),
        (CellBlock("surface", "triangle", _TRIANGLES, global_ids=_IDS),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    if curvature:
        element = lagrange_element("triangle", 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        coordinates = []
        for triangle in _TRIANGLES:
            p = uv[triangle]
            xy = p[0] + nodes[:, :1] * (p[1] - p[0]) + nodes[:, 1:] * (p[2] - p[0])
            z = curvature * xy[:, 0] * (1.0 - xy[:, 0])
            coordinates.extend(np.column_stack((xy, z)))
        geometry = CellGeometrySpec(
            {"surface": element},
            {"surface": np.arange(4 * len(nodes)).reshape(4, len(nodes))},
            np.asarray(coordinates),
        )
    entity = domain.entity_id(2, 0)
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=_IDS[order],
        entity_global_ids=_IDS[order],
        local_entity_indices=np.full(4, -1, dtype=np.int32),
        target_reference_vertices=np.tile(_REFERENCE[None], (4, 1, 1)),
        source_chart_ids=(entity,) * 4,
        source_entity_ids=(entity,) * 4,
        occurrence_paths=((),) * 4,
        source_parameters=uv[_TRIANGLES[order]],
        orientations=np.ones(4, dtype=np.int8),
    )
    return mesh, geometry, source, witness


def _certificate(
    center: tuple[float, float],
    heights: np.ndarray,
    order: np.ndarray,
    tolerance: float,
    *,
    curvature: float = 0.0,
) -> SourceFidelityCertificate:
    mesh, geometry, source, witness = _case(center, heights, order, curvature=curvature)
    return certify_source_fidelity(
        mesh, geometry, source, tolerance=tolerance, witness=witness
    )


def _exact_row_bounds(heights: np.ndarray) -> tuple[Fraction, ...]:
    # |z| attains its maximum at a vertex of each affine triangle.
    return tuple(max(abs(Fraction(float(heights[i]))) for i in row) for row in _TRIANGLES)


def _check_oracle(
    certificate: SourceFidelityCertificate,
    heights: np.ndarray,
    tolerance: float,
) -> None:
    expected = _exact_row_bounds(heights)
    for row, identifier in enumerate(certificate.source_cell_global_ids):
        exact = expected[int(identifier) - 101]
        assert Fraction(float(certificate.deviation_bounds[row, 1])) >= exact
    maximum = max(expected)
    assert Fraction(certificate.mesh_to_source_upper) >= maximum
    assert Fraction(certificate.source_to_mesh_upper) >= maximum
    assert (certificate.status == "certified") == (maximum <= Fraction(tolerance))


def test_randomized_affine_chart_decisions_match_exact_oracle() -> None:
    rng = np.random.default_rng(92761)
    order = np.arange(4)
    for _ in range(12):
        center = tuple(rng.integers(2, 15, size=2) / 16.0)
        heights = rng.integers(-16, 17, size=5).astype(np.float64) / 64.0
        maximum = max(_exact_row_bounds(heights))
        # Include both decisions and equality, which cannot be settled by a
        # strictly outward floating enclosure straddling the exact threshold.
        for tolerance in (
            float(maximum),
            float(maximum + Fraction(1, 64)),
            float(maximum - Fraction(1, 128)),
        ):
            if tolerance < 0:
                continue
            _check_oracle(
                _certificate(center, heights, order, tolerance), heights, tolerance
            )


def test_chart_results_do_not_depend_on_other_rows_or_row_order() -> None:
    center = (0.375, 0.625)
    first = np.asarray((0.0625, -0.0625, 0.0625, 0.0625, 0.0625))
    changed = first.copy()
    changed[2] = 0.25  # Vertex 2 does not belong to row 101.
    tolerance = 0.125
    a = _certificate(center, first, np.arange(4), tolerance)
    b = _certificate(center, changed, np.asarray((2, 0, 3, 1)), tolerance)
    _check_oracle(a, first, tolerance)
    _check_oracle(b, changed, tolerance)
    ai = a.source_cell_global_ids.index(101)
    bi = b.source_cell_global_ids.index(101)
    np.testing.assert_array_equal(a.deviation_bounds[ai], b.deviation_bounds[bi])
    af = tuple((f.check, f.status) for f in a.findings if f.entity_ids == (101,))
    bf = tuple((f.check, f.status) for f in b.findings if f.entity_ids == (101,))
    assert af == bf


def _quadratic_bounds(
    center: tuple[float, float], curvature: float
) -> tuple[Fraction, ...]:
    # For z = a*x*(1-x), the chord residual is a times the variance of
    # vertex x-coordinates. Its degree-two Bernstein controls are zero at
    # vertices and a*(x_i-x_j)^2/2 on edges. These are the exact controls
    # used by the original polynomial-hull decision, not sampled distances.
    xs = (Fraction(0), Fraction(1), Fraction(1), Fraction(0), Fraction(center[0]))
    a = Fraction(curvature)
    result = []
    for triangle in _TRIANGLES:
        x = tuple(xs[int(i)] for i in triangle)
        endpoint = max(a * value * (1 - value) for value in x)
        residual = max(a * (x[i] - x[j]) ** 2 / 2 for i in range(3) for j in range(i))
        result.append(endpoint + residual)
    return tuple(result)


def _quadratic_certificate(
    center: tuple[float, float],
    curvature: float,
    tolerance: float,
    order: np.ndarray,
) -> SourceFidelityCertificate:
    heights = np.asarray((0.0, 0.0, 0.0, 0.0, curvature * center[0] * (1.0 - center[0])))
    return _certificate(center, heights, order, tolerance, curvature=curvature)


def _check_quadratic(
    certificate: SourceFidelityCertificate,
    center: tuple[float, float],
    curvature: float,
    tolerance: float,
) -> None:
    exact = _quadratic_bounds(center, curvature)
    for row, identifier in enumerate(certificate.source_cell_global_ids):
        assert (
            Fraction(float(certificate.deviation_bounds[row, 1]))
            >= exact[identifier - 101]
        )
    assert (certificate.status == "certified") == (max(exact) <= Fraction(tolerance))


def test_randomized_quadratic_screen_matches_exact_bernstein_decisions() -> None:
    rng = np.random.default_rng(61273)
    for _ in range(8):
        center = tuple(rng.integers(2, 15, size=2) / 16.0)
        curvature = float(rng.integers(1, 9) / 16.0)
        maximum = max(_quadratic_bounds(center, curvature))
        exact_float = float(maximum)
        for tolerance in (
            np.nextafter(exact_float, -np.inf),
            exact_float,
            np.nextafter(exact_float, np.inf),
            exact_float + 0.125,
        ):
            result = _quadratic_certificate(center, curvature, tolerance, np.arange(4))
            _check_quadratic(result, center, curvature, tolerance)


def test_quadratic_screen_batch_permutation_preserves_rows() -> None:
    center = (0.375, 0.625)
    curvature = 0.5
    tolerance = 1.0
    first = _quadratic_certificate(center, curvature, tolerance, np.arange(4))
    permuted = _quadratic_certificate(
        center, curvature, tolerance, np.asarray((2, 0, 3, 1))
    )
    _check_quadratic(first, center, curvature, tolerance)
    _check_quadratic(permuted, center, curvature, tolerance)
    for identifier in first.source_cell_global_ids:
        a = first.source_cell_global_ids.index(identifier)
        b = permuted.source_cell_global_ids.index(identifier)
        np.testing.assert_array_equal(
            first.deviation_bounds[a], permuted.deviation_bounds[b]
        )


def test_quadratic_screen_encloses_exact_endpoint() -> None:
    center = (0.5, 0.5)
    curvature = 0.5
    tolerance = float(np.nextafter(0.375, -np.inf))
    certificate = _quadratic_certificate(center, curvature, tolerance, np.arange(4))
    _check_quadratic(certificate, center, curvature, tolerance)
    exact = _quadratic_bounds(center, curvature)
    for row, identifier in enumerate(certificate.source_cell_global_ids):
        endpoint = exact[identifier - 101]
        inward = np.nextafter(float(endpoint), -np.inf)
        assert Fraction(float(inward)) < endpoint
        assert Fraction(float(certificate.deviation_bounds[row, 1])) > Fraction(float(inward))
