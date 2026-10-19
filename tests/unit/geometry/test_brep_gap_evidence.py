from typing import get_args

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry.brep import (
    BRepGapEvidence,
    BRepGapKind,
    BRepGapLimitError,
    BRepGeometry,
    BRepModel,
    certify_brep_gaps,
    LineCurve,
)
from tests._support.cad_models import native_solid


@pytest.fixture(scope="module")
def box() -> BRepModel:
    return native_solid("box")


@pytest.fixture(scope="module")
def box_evidence(box: BRepModel) -> BRepGapEvidence:
    return certify_brep_gaps(box)


@pytest.mark.parametrize("kind", get_args(BRepGapKind.__value__))
def test_gap_kinds_publish_finite_physical_bounds(
    box: BRepModel,
    box_evidence: BRepGapEvidence,
    kind: BRepGapKind,
) -> None:
    selected = np.asarray([item == kind for item in box_evidence.kinds], dtype=np.bool_)
    assert np.any(selected)
    assert np.all(np.isfinite(box_evidence.upper_bounds[selected]))
    assert np.all(box_evidence.upper_bounds[selected] >= 0.0)
    assert box_evidence.model_id == box.model_id
    assert all(
        entity.source_revision == box.source_revision for entity in box_evidence.entities
    )


def test_gap_evidence_does_not_change_model_identity(box: BRepModel) -> None:
    before = box.model_id, box.tessellation_id, box.source_revision
    certify_brep_gaps(box)
    assert (box.model_id, box.tessellation_id, box.source_revision) == before
    restored = box.evolve()
    assert (
        restored.model_id,
        restored.tessellation_id,
        restored.source_revision,
    ) == before


def test_gap_limit_refusal_preserves_measured_bound(box: BRepModel) -> None:
    geometry = box.require_geometry()
    curve_index = geometry.edge_curves[0]
    curve = geometry.curves[curve_index]
    if not isinstance(curve, LineCurve):
        raise TypeError(f"box.curve must be LineCurve; got {type(curve).__name__}")
    # An exactly representable displacement below the pre-existing endpoint
    # admission tolerance isolates measurement from construction refusal.
    displacement = 2.0**-34
    changed = LineCurve(
        np.asarray(curve.origin) + (displacement, 0.0, 0.0), curve.direction
    )
    curves = tuple(
        changed if index == curve_index else item
        for index, item in enumerate(geometry.curves)
    )
    displaced = BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        occurrences=geometry.occurrences,
        assembly_containers=geometry.assembly_containers,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    model = box.evolve(geometry=displaced, coedge_deviation_bounds=None)
    evidence = certify_brep_gaps(model)
    bound = evidence.bound(model.edge_ids[0], kind="edge-face")
    assert bound >= displacement
    before = evidence.upper_bounds.copy()
    with pytest.raises(BRepGapLimitError) as refused:
        evidence.require_limit(displacement / 2.0)
    assert refused.value.measured > refused.value.limit
    assert refused.value.limit == displacement / 2.0
    assert refused.value.entity.source_revision == model.source_revision
    np.testing.assert_array_equal(evidence.upper_bounds, before)


def test_gap_evidence_reserves_before_processing(box: BRepModel) -> None:
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with pytest.raises(CoordinateEnclosureResourceError) as refused:
            certify_brep_gaps(box, budget=budget)
        assert refused.value.completed == 0
        assert refused.value.phase_evidence is not None
        assert refused.value.phase_evidence.work_units == 0


def test_face_export_bound_includes_incident_gap_evidence(
    box: BRepModel,
    box_evidence: BRepGapEvidence,
) -> None:
    face = box.face_ids[0]
    selected = [
        index for index, entity in enumerate(box_evidence.partners) if entity == face
    ]
    assert selected
    assert box_evidence.bound(face) == float(np.max(box_evidence.upper_bounds[selected]))


@pytest.mark.parametrize("kind", ("edge-face", "edge-trim", "edge-edge"))
def test_shared_box_support_has_zero_coupled_gap(
    box_evidence: BRepGapEvidence,
    kind: BRepGapKind,
) -> None:
    selected = np.asarray([item == kind for item in box_evidence.kinds], dtype=np.bool_)
    assert np.any(selected)
    np.testing.assert_array_equal(box_evidence.upper_bounds[selected], 0.0)


@pytest.mark.parametrize("name", ("cylinder", "sphere", "torus"))
def test_periodic_and_pole_gap_evidence_is_finite(name: str) -> None:
    model = native_solid(name, catalog="geometry")
    evidence = certify_brep_gaps(model)
    assert np.all(np.isfinite(evidence.upper_bounds))
    assert evidence.model_id == model.model_id


def test_published_gap_bounds_cannot_be_grown(box_evidence: BRepGapEvidence) -> None:
    before = box_evidence.upper_bounds.copy()
    with pytest.raises(ValueError, match="read-only"):
        box_evidence.upper_bounds[0] = 1.0
    np.testing.assert_array_equal(box_evidence.upper_bounds, before)


def _converted_affine_lamina(*, width: float, recompute: bool) -> BRepModel:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import (
        BRepTessellationPolicy,
        Builder,
        finish,
    )
    from phydrax.geometry.brep._coordinate_request import coordinate_source_request
    from phydrax.geometry.brep._patches import BSplineCurve, ExtrusionSurface
    from phydrax.geometry.brep._planar_offset import planar_carrier

    sweep = ExtrusionSurface(
        BSplineCurve(
            np.asarray(((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), dtype=np.float64),
            np.ones(2, dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
            1,
        ),
        (0.0, 2.0, 0.0),
    )
    plane = planar_carrier(sweep)
    assert plane is not None
    builder = Builder()
    builder.recompute_endpoint_gaps = recompute
    points = np.asarray(
        ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    vertices = [builder.vertex(point) for point in points]
    parameters = np.asarray(
        ((0.0, 0.0), (width / 3.0, 0.0), (width / 3.0, 0.5), (0.0, 0.5)), dtype=np.float64
    )
    with coordinate_source_request((sweep, points)) as budget:
        uses = []
        for first in range(4):
            last = (first + 1) % 4
            edge = builder.edge(
                LineCurve(points[first], points[last] - points[first]),
                0.0,
                1.0,
                vertices[first],
                vertices[last],
            )
            uses.append(
                builder.coedge(
                    edge,
                    1,
                    LineCurve(parameters[first], parameters[last] - parameters[first]),
                    budget=budget,
                )
            )
        builder.face(
            plane,
            np.asarray(((0.0, 0.0), (width / 3.0, 0.5)), dtype=np.float64),
            [uses],
            "converted-plane",
            None,
            1,
        )
        return finish(
            builder,
            solid=False,
            coordinate_contract=SpatialCoordinateContract.si(),
            source_id="converted-affine-lamina",
            tessellation=BRepTessellationPolicy(realize=False),
            budget=budget,
        )


def test_finish_certifies_face_only_rounding_with_exact_endpoints() -> None:
    from fractions import Fraction

    model = _converted_affine_lamina(width=1.0, recompute=True)
    geometry = model.require_geometry()
    assert geometry.edge_vertex_gap_bounds is None
    bounds = geometry.edge_face_gap_bounds
    assert bounds is not None
    # Exact rational lift of the stored binary64 chart coordinate. This oracle
    # does not call the publication or gap-measurement implementation.
    distance = abs(3 * Fraction(1.0 / 3.0) - 1)
    assert distance > 0
    assert Fraction(bounds[0]) >= distance
    assert Fraction(bounds[1]) >= distance
    assert Fraction(bounds[2]) >= distance
    assert bounds[3] == 0.0


def test_gap_free_recomputation_preserves_geometry_and_model_ids() -> None:
    before = _converted_affine_lamina(width=3.0, recompute=False)
    after = _converted_affine_lamina(width=3.0, recompute=True)
    assert before.require_geometry().edge_vertex_gap_bounds is None
    assert after.require_geometry().edge_vertex_gap_bounds is None
    assert after.require_geometry().edge_face_gap_bounds is None
    assert before.require_geometry().geometry_id == after.require_geometry().geometry_id
    assert before.model_id == after.model_id
    assert before.source_revision == after.source_revision
