"""Exact dyadic witnesses for periodic face projection certificates."""

import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._patches import CylinderPatch, LineCurve, RevolutionSurface
from phydrax.geometry.brep._projection import NativeBRepProjection, prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import BRepProjectionStatus, brep_entity_id
from tests.unit.geometry.test_brep_projection import _instanced_box, _rectangle_face_model
from tests.unit.geometry.test_native_source_projection import _nearby_stationary_root_surface_model


def _trig(value: Fraction, *, cosine: bool) -> tuple[Fraction, Fraction]:
    first, degree = (0, 60) if cosine else (1, 61)
    total = sum(
        (Fraction((-1) ** ((n - first) // 2), math.factorial(n)) * value**n
         for n in range(first, degree + 1, 2)),
        Fraction(0),
    )
    error = abs(value) ** (degree + 1) / math.factorial(degree + 1)
    return total - error, total + error


def _projection(kind: str, lower: float, upper: float) -> NativeBRepProjection:
    origin, axis = (0, 0, 0), (0, 0, 1)
    if kind == "cylinder":
        patch = CylinderPatch(origin, (1, 0, 0), (0, 1, 0), axis, 1)
        bounds = ((lower, -1.0), (upper, 1.0))
    else:
        patch = RevolutionSurface(LineCurve((1, 0, -1), (0, 0, 2)), origin, axis)
        bounds = ((lower, 0.0), (upper, 1.0))
    return prepare_brep_projection(_rectangle_face_model(patch, bounds))


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
@pytest.mark.parametrize(
    ("x", "y", "edge_hex", "side"),
    (
        (Fraction(1, 8), Fraction(3, 16), "0x1.f730bd281f69bp-1", "upper"),
        (Fraction(3, 16), Fraction(1, 4), "0x1.dac670561bb4fp-1", "upper"),
        (Fraction(1, 8), Fraction(3, 16), "0x1.f730bd281f69cp-1", "lower"),
        (Fraction(3, 16), Fraction(1, 4), "0x1.dac670561bb50p-1", "lower"),
        (Fraction(3, 16), Fraction(-1, 4), "-0x1.dac670561bb50p-1", "upper"),
    ),
)
def test_outside_carrier_foot_is_not_certified(
    kind: str, x: Fraction, y: Fraction, edge_hex: str, side: str
) -> None:
    edge = float.fromhex(edge_hex)
    sine, cosine = _trig(Fraction(edge), cosine=False), _trig(Fraction(edge), cosine=True)
    products = tuple(y * c - x * s for c in cosine for s in sine)
    if side == "upper":
        assert min(products) > 0
        lower, upper = (-2.0, edge) if y < 0 else (0.0, edge)
    else:
        assert max(products) < 0
        lower, upper = edge, 2.0
    result = _projection(kind, lower, upper).project(
        np.asarray(((float(x), float(y), 0.25),), dtype=np.float64),
        np.asarray((2,), dtype=np.int64), np.asarray((0,), dtype=np.int64),
    )
    face_squared = tuple((x - c) ** 2 + (y - s) ** 2 for c in cosine for s in sine)
    assert Fraction(float(result.distance_lower_bounds[0])) ** 2 <= min(face_squared)
    upper_distance = float(result.residuals[0])
    assert math.isinf(upper_distance) or Fraction(upper_distance) ** 2 >= max(face_squared)
    parameter = float(result.parameters[0, 0])
    if math.isnan(parameter):
        assert int(result.status[0]) == BRepProjectionStatus.UNRESOLVED
    else:
        assert lower <= parameter <= upper

@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_interior_foot_remains_unique(kind: str) -> None:
    # cos(2) < 0 < sin(2), so 0 < pi/4 < pi/2 < 2.
    assert _trig(Fraction(2), cosine=True)[1] < 0
    assert _trig(Fraction(2), cosine=False)[0] > 0
    result = _projection(kind, 0.0, 2.0).project(
        np.asarray(((1.0, 1.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE
    lower = Fraction(float(result.distance_lower_bounds[0]))
    upper = Fraction(float(result.residuals[0]))
    assert (lower + 1) ** 2 <= 2 <= (upper + 1) ** 2


@pytest.mark.parametrize("kind", ("cylinder", "revolution"))
def test_full_turn_retains_interior_and_seam(kind: str) -> None:
    projection = _projection(kind, -math.pi, math.pi)
    interior = projection.project(
        np.asarray(((1.0, 1.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(interior.status[0]) == BRepProjectionStatus.UNIQUE
    seam = projection.project(
        np.asarray(((-1.0, 0.0, 0.25),)), np.asarray((2,)), np.asarray((0,))
    )
    assert int(seam.status[0]) == BRepProjectionStatus.SEAM


def test_placed_point_residual_covers_exact_distance() -> None:
    placement = BRepAffinePlacement.from_matrix(
        ((Fraction(3, 5), Fraction(-4, 5), 0, Fraction(1, 3)),
         (Fraction(4, 5), Fraction(3, 5), 0, 0),
         (0, 0, 1, 0), (0, 0, 0, 1))
    )
    projection = prepare_brep_projection(_instanced_box(placement))
    point = np.asarray(((float(Fraction(17, 60)), float(Fraction(7, 20)), -0.125),))
    result = projection.project(point, np.asarray((2,)), np.asarray((0,)),
                                occurrence_paths=(("pair", "right"),))
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE
    lower = Fraction(float(result.distance_lower_bounds[0]))
    upper = Fraction(float(result.residuals[0]))
    assert lower <= Fraction(1, 8) <= upper
    squared = sum((Fraction(float(a)) - Fraction(float(b))) ** 2
                  for a, b in zip(point[0], np.asarray(result.points)[0], strict=True))
    assert upper**2 >= squared


def test_boundary_union_includes_unresolved_and_resolved_bounds() -> None:
    model, _ = _nearby_stationary_root_surface_model()
    projection = prepare_brep_projection(model)
    query = np.asarray(((0.0, 0.5, 0.0),), dtype=np.float64)
    face = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    edge_ids = tuple(
        brep_entity_id(model.source_id, model.source_revision, 1, index)
        for index in range(model.topology.num_edges)
    )
    face_result = projection.project_boundary(query, entity_ids=(face,))
    edge_results = tuple(
        projection.project_boundary(query, entity_ids=(edge,)) for edge in edge_ids
    )
    edge_index = min(range(len(edge_ids)), key=lambda i: float(edge_results[i].residuals[0]))
    edge_result = edge_results[edge_index]
    assert int(face_result.status[0]) == BRepProjectionStatus.UNRESOLVED
    assert float(edge_result.residuals[0]) < float(face_result.residuals[0])
    for ids in ((face, edge_ids[edge_index]), (edge_ids[edge_index], face)):
        result = projection.project_boundary(query, entity_ids=ids)
        assert int(result.status[0]) == BRepProjectionStatus.UNRESOLVED
        assert float(result.distance_lower_bounds[0]) == min(
            float(face_result.distance_lower_bounds[0]),
            float(edge_result.distance_lower_bounds[0]),
        )
        assert float(result.residuals[0]) == min(
            float(face_result.residuals[0]), float(edge_result.residuals[0])
        )


def test_uncertified_polygon_membership_keeps_base_projection() -> None:
    import jax.numpy as jnp

    from phydrax.geometry.brep._model import BRepGeometry, BRepImportReport, BRepModel
    from phydrax.geometry.brep._patches import SurfaceMappedCurve
    from phydrax.geometry.brep._root_bindings import BRepPlacedVertex, BRepVertexRoot
    from tests.unit.geometry.test_brep_projection import _CONTRACT

    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
    corners = ((0.0, -1.0), (2.0, -1.0), (1.0, 1.0), (0.0, 1.0))
    points = np.asarray(patch.evaluate(jnp.asarray(corners, dtype=jnp.float64)))
    pcurves = tuple(
        LineCurve(
            corners[i],
            tuple(corners[(i + 1) % 4][j] - corners[i][j] for j in range(2)),
        )
        for i in range(4)
    )
    geometry = BRepGeometry(
        vertex_points=points,
        vertex_roots=tuple(
            BRepVertexRoot(
                BRepPlacedVertex(point, np.eye(3), np.zeros(3), f"probe-{i}")
            )
            for i, point in enumerate(points)
        ),
        curves=tuple(SurfaceMappedCurve(patch, curve) for curve in pcurves),
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (4, 1)),
        edge_vertices=((0, 1), (1, 2), (2, 3), (3, 0)),
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    report = BRepImportReport(
        source_id="projection-evidence-probe",
        source_digest="e" * 64,
        source_format="native",
        coordinate_contract=_CONTRACT,
        import_policy_id="f" * 64,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    model = BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray((((0.0, -1.0), (2.0, 1.0)),), dtype=np.float64),
        orientation=np.asarray((1.0,), dtype=np.float64),
        physical_tags=("face",),
        coordinate_contract=_CONTRACT,
        report=report,
    )
    result = prepare_brep_projection(model).project(
        np.asarray(((1.0, 0.5, 0.0),), dtype=np.float64),
        np.asarray((2,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
    )
    # Pinned from the clean 6e73f0d3c base: membership uses the polygon fallback.
    assert int(result.status[0]) == BRepProjectionStatus.UNIQUE
    assert float(result.distance_lower_bounds[0]).hex() == "0x1.e3779b97f4a7ap-4"
    assert float(result.residuals[0]).hex() == "0x1.e3779b97f4a80p-4"
