"""Independently authored S2-029 construction and contact corpus.

All fixtures use native constructors, not an imported CAD corpus. Volumes are
checked against rational closed forms and an independent Machin-series enclosure
of pi. Refusals retain their scientific reason or BX-17 endpoint certificate;
an unrelated exception never counts as a successful corpus outcome.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from math import tau
from pathlib import Path

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._boolean_core import (
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanResult,
)
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_offset,
    brep_revolution,
    brep_sphere,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._fillet import brep_fillet
from phydrax.geometry.brep._measure import brep_mass_properties, BRepMassProperties
from phydrax.geometry.brep._model import BRepEntityId, BRepModel
from phydrax.geometry.brep._parameter_range import refusal_parameter_range
from phydrax.geometry.brep._partition import (
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    partition_brep,
)
from phydrax.geometry.brep._patches import PlanePatch
from phydrax.geometry.brep._section_clearance import brep_section


_CONTRACT = SpatialCoordinateContract.si()
_LAZY = BRepTessellationPolicy(realize=False)
_AXIAL_PLANE = ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _atan_enclosure(value: Fraction) -> tuple[Fraction, Fraction]:
    # Even truncation of this decreasing alternating series is a lower bound;
    # adding its first omitted term bounds the infinite tail above.
    lower = sum(
        ((-1) ** n * value ** (2 * n + 1) / (2 * n + 1) for n in range(32)),
        Fraction(),
    )
    return lower, lower + value**65 / 65


_FIVE_LOWER, _FIVE_UPPER = _atan_enclosure(Fraction(1, 5))
_OTHER_LOWER, _OTHER_UPPER = _atan_enclosure(Fraction(1, 239))
_PI_LOWER = 16 * _FIVE_LOWER - 4 * _OTHER_UPPER
_PI_UPPER = 16 * _FIVE_UPPER - 4 * _OTHER_LOWER


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=1_000_000_000
        )
        with budget.activate():
            yield budget


def _assert_volume(
    model: BRepModel, reference: tuple[Fraction, Fraction]
) -> BRepMassProperties:
    with _budget() as budget:
        measured = brep_mass_properties(
            model, model.require_geometry(), subdivisions=12, coordinate_budget=budget
        )
    assert measured.model_id == model.model_id
    bounds = np.asarray(measured.volume_enclosures)
    assert bounds.shape == (model.topology.num_solids, 2)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 0] > 0.0)
    # Sum binary endpoints exactly; no rounded reduction can hide a bad bound.
    lower = sum((Fraction(float(row[0])) for row in bounds), Fraction())
    upper = sum((Fraction(float(row[1])) for row in bounds), Fraction())
    assert lower <= reference[0] <= reference[1] <= upper
    return measured


def _assert_empty_boolean(result: BRepBooleanResult) -> None:
    # The independent zero-volume oracle is topological: the intersection has
    # no open three-dimensional component, not a small numerical volume.
    assert result.empty
    assert result.model.topology.num_solids == result.model.topology.num_faces == 0
    assert result.model.require_geometry().face_loops == ()
    graph = result.association_graph
    assert result.certificate_id
    assert graph.transaction.coverage.source_exhaustive
    assert graph.transaction.coverage.target_exhaustive
    assert not graph.target_revision.occurrences
    assert set(result.deleted_source_ids) == {
        occurrence.occurrence_id for occurrence in graph.source_revision.occurrences
    }


def _box() -> BRepModel:
    return brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )


def _axial_edge(model: BRepModel) -> BRepEntityId:
    geometry = model.require_geometry()
    points = np.asarray(geometry.vertex_points)
    for index, (first, last) in enumerate(geometry.edge_vertices):
        if set(map(tuple, points[[first, last]])) == {(0.0, 0.0, 0.0), (0.0, 0.0, 4.0)}:
            return model.edge_ids[index]
    raise AssertionError("corpus.box must contain its authored z-axis edge; got no match")


@pytest.mark.parametrize(
    ("corners", "pi_coefficient", "centroid_height", "pole_count"),
    [
        pytest.param(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)),
            12,
            1.5,
            0,
            id="axis-edge-cylinder",
        ),
        pytest.param(
            ((0.0, 0.0), (2.0, 0.0), (0.0, 3.0)),
            4,
            0.75,
            1,
            id="axis-apex-cone",
        ),
    ],
)
def test_axis_touching_revolution_has_certified_volume(
    corners: tuple[tuple[float, float], ...],
    pi_coefficient: int,
    centroid_height: float,
    pole_count: int,
) -> None:
    profile = PlanarProfile(_AXIAL_PLANE, ProfileLoop.polygon(corners))
    model = brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    geometry = model.require_geometry()
    assert model.topology.num_solids == 1
    assert sum(geometry.degenerate_edges) == pole_count
    assert all(
        balance == 0
        for balance, degenerate in zip(
            geometry.edge_use_balance(model.orientation),
            geometry.degenerate_edges,
            strict=True,
        )
        if not degenerate
    )
    measured = _assert_volume(
        model, (pi_coefficient * _PI_LOWER, pi_coefficient * _PI_UPPER)
    )
    centroid = np.asarray((0.0, 0.0, centroid_height), dtype=np.float64)
    bounds = np.asarray(measured.centroid_enclosures)[0]
    assert np.all(bounds[0] <= centroid)
    assert np.all(centroid <= bounds[1])


@pytest.mark.parametrize(
    "corners",
    [
        pytest.param(
            ((-1.0, 0.0), (2.0, 0.0), (2.0, 3.0), (-1.0, 3.0)),
            id="crossing-rectangle",
        ),
        pytest.param(
            ((-1.0, 0.0), (2.0, 0.0), (0.0, 3.0)),
            id="crossing-with-axis-vertex",
        ),
    ],
)
def test_axis_crossing_revolution_refuses_half_plane_domain(
    corners: tuple[tuple[float, float], ...],
) -> None:
    profile = PlanarProfile(_AXIAL_PLANE, ProfileLoop.polygon(corners))
    # Authored x signs prove both half-planes are occupied. The public
    # ValueError is a domain refusal, not a license to accept any exception.
    assert min(point[0] for point in corners) < 0 < max(point[0] for point in corners)
    message = (
        "brep_revolution.profile must lie on one side of its revolution axis; "
        "got an axis-crossing source profile"
    )
    with pytest.raises(ValueError, match="axis-crossing source profile") as failure:
        brep_revolution(
            profile,
            (0.0, 0.0),
            (0.0, 1.0),
            coordinate_contract=_CONTRACT,
            tessellation=_LAZY,
        )
    assert type(failure.value) is ValueError
    assert failure.value.args == (message,)


@pytest.mark.parametrize(
    "angle",
    [pytest.param(None, id="native-full360"), pytest.param(tau, id="literal-two-pi")],
)
def test_full_turn_annular_sweep_preserves_phase_and_volume(angle: float | None) -> None:
    profile = PlanarProfile(
        _AXIAL_PLANE,
        ProfileLoop.polygon(((1.0, 0.0), (2.0, 0.0), (2.0, 3.0), (1.0, 3.0))),
    )
    model = brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        angle=angle,
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    # A native turn identifies its seam; a numeric binary64 angle is a literal
    # sector and retains two caps even when its displayed value is 2*pi.
    assert model.topology.num_faces == (4 if angle is None else 6)
    assert all(
        balance == 0
        for balance in model.require_geometry().edge_use_balance(model.orientation)
    )
    expected = (
        (9 * _PI_LOWER, 9 * _PI_UPPER)
        if angle is None
        else (Fraction(9, 2) * Fraction(angle), Fraction(9, 2) * Fraction(angle))
    )
    _assert_volume(model, expected)


@pytest.mark.parametrize(
    "radius",
    [
        pytest.param(2.0, id="exact-limit"),
        pytest.param(float(np.nextafter(2.0, np.inf)), id="one-ulp-beyond"),
    ],
)
def test_fillet_feasible_endpoint_refuses_with_exact_range(radius: float) -> None:
    source = _box()
    identity = source.model_id
    edge = _axial_edge(source)
    result = brep_fillet(source, (edge,), radius, tessellation=_LAZY)
    assert result.status == "unsupported"
    assert result.reason == (
        "planar blend refuses: setback reaches a neighboring edge or face vanishes"
    )
    certificate = result.parameter_range
    assert certificate is not None
    # The quarter circle consumes radius along x and y, not along z.
    assert certificate.parameter == "radius"
    assert certificate.interval == (0.0, 2.0)
    assert certificate.end_status == ("attained_exactly", "attained_exactly")
    assert certificate.constraints[1] == "face_vanishes"
    assert certificate.entity_ids[1]
    assert all(entity in source.edge_ids for entity in certificate.entity_ids[1])
    assert certificate.evidence
    assert source.model_id == identity


def test_fillet_inside_feasible_endpoint_encloses_removed_corner_volume() -> None:
    source = _box()
    radius = 2.0 - 2.0**-12
    result = brep_fillet(
        source,
        (_axial_edge(source),),
        radius,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    assert result.model is not None
    assert result.parameter_range is not None
    assert result.parameter_range.interval == (0.0, 2.0)
    squared = Fraction(radius) ** 2
    # Remove (square minus quarter disk) times edge length four.
    _assert_volume(
        result.model,
        (24 - 4 * squared + squared * _PI_LOWER, 24 - 4 * squared + squared * _PI_UPPER),
    )


@pytest.mark.parametrize(
    "distance",
    [
        pytest.param(-1.0, id="exact-limit"),
        pytest.param(float(np.nextafter(-1.0, -np.inf)), id="one-ulp-beyond"),
    ],
)
def test_offset_feasible_endpoint_refuses_with_exact_range(distance: float) -> None:
    source = _box()
    identity = source.model_id
    with pytest.raises(
        ValueError, match="face vanishes|collision pair|reversed edge"
    ) as failure:
        brep_offset(source, distance, tessellation=_LAZY)
    assert type(failure.value) is ValueError
    certificate = refusal_parameter_range(failure.value)
    # Opposing x faces each advance |d|: 2 + 2*d reaches zero at -1.
    assert certificate.parameter == "distance"
    assert certificate.interval[0] == -1.0
    assert certificate.end_status[0] == "attained_exactly"
    assert certificate.constraints[0] == "trim_vanishes"
    assert certificate.entity_ids[0]
    assert all(entity in source.edge_ids for entity in certificate.entity_ids[0])
    assert certificate.evidence
    assert source.model_id == identity


def test_offset_inside_feasible_endpoint_encloses_remaining_box_volume() -> None:
    source = _box()
    distance = -0.875
    result = brep_offset(
        source, distance, tessellation=_LAZY, include_parameter_range=True
    )
    assert result.evidence.volume_status == "certified"
    assert result.parameter_range is not None
    assert result.parameter_range.interval[0] == -1.0
    exact = (
        (Fraction(2) + 2 * Fraction(distance))
        * (Fraction(3) + 2 * Fraction(distance))
        * (Fraction(4) + 2 * Fraction(distance))
    )
    bounds = result.evidence.volume_enclosure
    assert bounds is not None
    assert Fraction(float(bounds[0])) <= exact <= Fraction(float(bounds[1]))
    _assert_volume(result.model, (exact, exact))


@pytest.mark.parametrize(
    "maximum",
    [
        pytest.param((2.0, 3.0, 4.0), id="coincident-removal"),
        pytest.param((3.0, 4.0, 5.0), id="enclosing-removal"),
    ],
)
def test_empty_material_split_has_typed_refusal_and_no_publication(
    maximum: tuple[float, float, float], tmp_path: Path
) -> None:
    body = _box()
    cutter = brep_box(
        (0.0, 0.0, 0.0),
        maximum,
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    identities = body.model_id, cutter.model_id
    plan = BRepPartitionPlan(
        _CONTRACT,
        (
            BRepPartitionOperand("body", body, BRepPartitionRole.REGION),
            BRepPartitionOperand(
                "cut", cutter, BRepPartitionRole.VOID, target_region_ids=("body",)
            ),
        ),
        BRepPartitionPolicy(("body",)),
    )
    destination = tmp_path / "empty-split.phx"
    with pytest.raises(BRepBooleanFailure) as failure:
        partition_brep(plan, destination=destination)
    # Removing a containing closed box leaves no material. Partition's typed
    # unresolved refusal must explain that zero set, not an unrelated failure.
    assert failure.value.reason == (
        "BRepPartition.regions must contain surviving material; got no surviving material region"
    )
    assert "Native CAD Boolean unresolved:" in str(failure.value)
    assert not destination.exists()
    assert (body.model_id, cutter.model_id) == identities


@pytest.mark.parametrize(
    "minimum",
    [
        pytest.param((1.0, 0.0, 0.0), id="face-contact"),
        pytest.param((1.0, 1.0, 0.0), id="edge-contact"),
        pytest.param((1.0, 1.0, 1.0), id="vertex-contact"),
    ],
)
def test_tangent_boxes_have_certified_regularized_empty_intersection(
    minimum: tuple[float, float, float],
) -> None:
    first = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    second = brep_box(
        minimum,
        tuple(value + 1.0 for value in minimum),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    identities = first.model_id, second.model_id
    # At least one interval overlap has length zero; there is no 3D cell.
    assert any(value == 1.0 for value in minimum)
    _assert_empty_boolean(boolean_brep(first, second, "intersection"))
    assert (first.model_id, second.model_id) == identities


@pytest.mark.parametrize(
    ("operation", "volume"),
    [
        pytest.param("union", 2, id="union"),
        pytest.param("difference", 1, id="difference"),
    ],
)
def test_face_tangent_boxes_preserve_certified_nonempty_volume(
    operation: BRepBooleanOperation, volume: int
) -> None:
    first = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    second = brep_box(
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    result = boolean_brep(first, second, operation)
    assert not result.empty
    assert result.association_graph.transaction.coverage.source_exhaustive
    assert result.association_graph.transaction.coverage.target_exhaustive
    _assert_volume(result.model, (Fraction(volume), Fraction(volume)))


def test_sphere_tangent_plane_has_certified_isolated_contact() -> None:
    sphere = brep_sphere(1.0, coordinate_contract=_CONTRACT, tessellation=_LAZY)
    plane = PlanePatch((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    result = brep_section(sphere, plane)
    assert result.status == "certified", result.reason
    assert len(result.intersections) == 1
    intersection = result.intersections[0]
    assert intersection.complete
    assert not intersection.curves
    assert not intersection.coincident
    assert not intersection.unresolved
    assert len(intersection.points) == 1
    contact = intersection.points[0]
    assert contact.kind == "tangent_point"
    assert contact.certificate == "analytic-contact"
    assert contact.branch_count == 0
    # x^2+y^2+z^2=1 and y=1 force x=z=0, independently of the atlas.
    np.testing.assert_array_equal(
        contact.point, np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    )
    assert np.isfinite(contact.gap_bound) and contact.gap_bound >= 0.0
    assert Fraction(float(contact.parameter_lower[0])) <= _PI_LOWER / 2
    assert _PI_UPPER / 2 <= Fraction(float(contact.parameter_upper[0]))
    assert np.all(contact.parameter_lower[1:] <= 0.0)
    assert np.all(contact.parameter_upper[1:] >= 0.0)


def test_sphere_tangent_plane_boolean_is_empty_or_explicitly_unresolved() -> None:
    sphere = brep_sphere(1.0, coordinate_contract=_CONTRACT, tessellation=_LAZY)
    # A Boolean operand must be a solid: this box's y=1 face supplies the
    # tangent plane, with the entire box on the exterior side of the sphere.
    exterior = brep_box(
        (-2.0, 1.0, -2.0),
        (2.0, 2.0, 2.0),
        coordinate_contract=_CONTRACT,
        tessellation=_LAZY,
    )
    identities = sphere.model_id, exterior.model_id
    try:
        result = boolean_brep(sphere, exterior, "intersection")
    except BRepBooleanFailure as failure:
        # This class is the scientific unresolved outcome. ValueError,
        # RuntimeError and every other arbitrary exception still fail the test.
        assert failure.reason
        assert "must" in failure.reason and "; got " in failure.reason
        assert str(failure).startswith("Native CAD Boolean unresolved: ")
    else:
        _assert_empty_boolean(result)
    assert (sphere.model_id, exterior.model_id) == identities
