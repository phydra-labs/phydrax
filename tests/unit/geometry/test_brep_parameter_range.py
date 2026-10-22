"""Independent extent and collision-pair oracles for construction ranges."""

import fractions
import sys
from collections.abc import Callable
from fractions import Fraction
from itertools import combinations
from types import FrameType

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry.brep import _planar_offset as planar_stages
from phydrax.geometry.brep._parameter_range import (
    attach_parameter_range,
    CertifiedParameterRange,
    refusal_parameter_range,
    unknown_parameter_range,
)
from tests._support.coordinate_references import coordinate_reference_budget


_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)


def _prism(*, concave: bool = False) -> phx.geometry.BRepModel:
    points = (
        ((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))
        if concave
        else ((0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0))
    )
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(), phx.geometry.ProfileLoop.polygon(points)
    )
    return phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 4.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )


def test_prism_range_matches_half_minimum_extent_and_preserves_boundary_refusals() -> (
    None
):
    source = _prism()
    result = phx.geometry.brep_offset(
        source, -0.25, tessellation=_LAZY, include_parameter_range=True
    )
    certificate = result.parameter_range
    assert certificate is not None
    # The x extent is 2; inward translation consumes it from both sides.
    assert certificate.parameter == "distance"
    assert certificate.interval[0] == -1.0
    assert certificate.end_status[0] == "attained_exactly"
    assert certificate.constraints[0] == "trim_vanishes"
    assert certificate.entity_ids[0]
    without = phx.geometry.brep_offset(source, -0.25, tessellation=_LAZY)
    assert without.parameter_range is None
    assert without.model.model_id == result.model.model_id
    assert (
        without.model.require_geometry().geometry_id
        == result.model.require_geometry().geometry_id
    )
    assert (
        phx.geometry.brep_offset(source, -0.875, tessellation=_LAZY).parameter_range
        is None
    )
    with pytest.raises(ValueError, match="face vanishes") as failure:
        phx.geometry.brep_offset(source, -1.0, tessellation=_LAZY)
    assert type(failure.value) is ValueError
    assert refusal_parameter_range(failure.value).interval == certificate.interval
    with pytest.raises(ValueError, match="collision pair|reversed edge") as outside:
        phx.geometry.brep_offset(source, -1.125, tessellation=_LAZY)
    assert refusal_parameter_range(outside.value).interval[0] == -1.0


def test_l_prism_range_reports_arm_collapse_and_unbounded_growth() -> None:
    source = _prism(concave=True)
    result = phx.geometry.brep_offset(
        source, -0.125, tessellation=_LAZY, include_parameter_range=True
    )
    certificate = result.parameter_range
    assert certificate is not None
    # Both unit-width arms collapse at -1/2. Outward, every trim keeps or grows
    # its length and no two non-incident features ever meet.
    assert certificate.interval == (-0.5, np.inf)
    assert certificate.end_status[0] == "attained_exactly"
    assert certificate.constraints[0] == "trim_vanishes"
    assert all(entity in source.edge_ids for entity in certificate.entity_ids[0])
    with pytest.raises(ValueError, match="face vanishes") as failure:
        phx.geometry.brep_offset(source, -0.5, tessellation=_LAZY)
    assert refusal_parameter_range(failure.value).interval == certificate.interval
    distance = 8.0
    grown = phx.geometry.brep_offset(source, distance, tessellation=_LAZY)
    # Five convex and one reflex right-angle corner: area 5 + 12 d + 4 d^2.
    exact = (5 + 12 * distance + 4 * distance**2) * (4 + 2 * distance)
    enclosure = grown.evidence.volume_enclosure
    assert grown.evidence.volume_status == "certified"
    assert enclosure is not None
    assert float(enclosure[0]) <= exact <= float(enclosure[1])


def _notch(wall: float) -> phx.geometry.BRepModel:
    # Notch walls have rational unit normals (-13, +-84)/85, so the reflex tip
    # at the origin moves along +x at 85/13 per unit inward distance.
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(
            (
                (-2.625, -4.0),
                (wall, -4.0),
                (wall, 4.0),
                (-2.625, 4.0),
                (-2.625, 0.40625),
                (0.0, 0.0),
                (-2.625, -0.40625),
            )
        ),
    )
    return phx.geometry.brep_extrusion(
        profile, (0.0, 0.0, 1.0), coordinate_contract=_CONTRACT, tessellation=_LAZY
    )


def _assert_tip_wall_bound(certificate: CertifiedParameterRange, wall: float) -> None:
    # Tip x = -85 d / 13 meets the opposite wall x = wall + d at
    # d = -13 wall / 98, before the unit height collapses at d = -1/2.
    contact = Fraction(-13, 98) * Fraction(wall)
    lower = certificate.interval[0]
    assert lower is not None
    assert certificate.constraints[0] == "edges_collide"
    assert certificate.end_status[0] == "certified_inner_bound"
    assert Fraction(float(np.nextafter(lower, -np.inf))) < contact < Fraction(lower)
    assert len(certificate.entity_ids[0]) == 2


def _orientation(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], third: tuple[Fraction, ...]
) -> int:
    value = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )
    return (value > 0) - (value < 0)


def _between(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], point: tuple[Fraction, ...]
) -> bool:
    return all(
        min(first[axis], second[axis]) <= point[axis] <= max(first[axis], second[axis])
        for axis in range(2)
    )


def _cap_trim_contacts(model: phx.geometry.BRepModel) -> int:
    """Count touching nonadjacent published top-cap trims by exact orientation."""
    geometry = model.require_geometry()
    points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(geometry.vertex_points)
    )
    top = max(point[2] for point in points)
    trims = tuple(
        (first, last)
        for first, last in geometry.edge_vertices
        if points[first][2] == top == points[last][2]
    )
    contacts = 0
    for (a, b), (c, d) in combinations(trims, 2):
        if {a, b} & {c, d}:
            continue
        p, q, r, s = points[a], points[b], points[c], points[d]
        signs = (
            _orientation(p, q, r),
            _orientation(p, q, s),
            _orientation(r, s, p),
            _orientation(r, s, q),
        )
        if (
            (signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0)
            or (signs[0] == 0 and _between(p, q, r))
            or (signs[1] == 0 and _between(p, q, s))
            or (signs[2] == 0 and _between(r, s, p))
            or (signs[3] == 0 and _between(r, s, q))
        ):
            contacts += 1
    return contacts


def test_acute_notch_offset_past_tip_wall_contact_refuses() -> None:
    source = _notch(0.5)
    assert _cap_trim_contacts(source) == 0
    # The tip travels 85/104 > 3/8, crossing the offset wall: no publication.
    with pytest.raises(ValueError, match="collision pair") as failure:
        phx.geometry.brep_offset(source, -0.125, tessellation=_LAZY)
    assert type(failure.value) is ValueError
    certificate = refusal_parameter_range(failure.value)
    _assert_tip_wall_bound(certificate, 0.5)
    entities = (*source.vertex_ids, *source.edge_ids, *source.face_ids)
    assert all(entity in entities for entity in certificate.entity_ids[0])


@pytest.mark.parametrize(
    ("wall", "distance"),
    [
        pytest.param(0.5, -0.0625, id="before-contact"),
        pytest.param(1.5, -0.125, id="distant-wall"),
    ],
)
def test_acute_notch_offset_before_tip_wall_contact_publishes_simple_cap(
    wall: float, distance: float
) -> None:
    result = phx.geometry.brep_offset(
        _notch(wall), distance, tessellation=_LAZY, include_parameter_range=True
    )
    assert result.evidence.volume_status == "certified"
    assert result.parameter_range is not None
    _assert_tip_wall_bound(result.parameter_range, wall)
    assert _cap_trim_contacts(result.model) == 0


# Python-level ``Fraction`` operations; reflected and mixed operators dispatch
# to these, and ordering comparisons share ``_richcmp``.
_FRACTION_OPERATIONS = frozenset(
    {
        "_add",
        "_sub",
        "_mul",
        "_div",
        "_richcmp",
        "__eq__",
        "__bool__",
        "__neg__",
        "__abs__",
    }
)


def _counted_fraction_operations[T](
    call: Callable[[], T],
) -> tuple[T | CoordinateEnclosureResourceError, int]:
    """Run ``call`` and count the rational operations it performs."""
    operations = 0

    def profile(frame: FrameType, event: str, argument: object) -> None:
        nonlocal operations
        code = frame.f_code
        if (
            event == "call"
            and code.co_filename == fractions.__file__
            and code.co_name in _FRACTION_OPERATIONS
        ):
            operations += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        outcome = call()
    except CoordinateEnclosureResourceError as error:
        outcome = error
    finally:
        sys.setprofile(previous)
    return outcome, operations


type _Motion = planar_stages._PlanarMotion


def _notch_motion() -> _Motion:
    """Return the rational affine trajectories of the wall-0.5 notch boundary."""
    source = _notch(0.5)
    geometry = source.require_geometry()
    carriers = tuple(planar_stages.planar_carrier(patch) for patch in source.patches)
    with coordinate_reference_budget() as budget:
        planes = planar_stages._offset_planes(source, carriers, budget=budget)
        zero = planar_stages._offset_vertices(geometry, planes, Fraction(), budget=budget)
        one = planar_stages._offset_vertices(geometry, planes, Fraction(1), budget=budget)
    velocity = tuple(
        tuple(b - a for a, b in zip(start, end, strict=True))
        for start, end in zip(zero, one, strict=True)
    )
    return planar_stages._planar_motion(source, planes, zero, velocity)


def _source_clearance(motion: _Motion, budget: CoordinateEnclosureBudget) -> object:
    return planar_stages._clearance_violation(
        motion, Fraction(), Fraction(), budget=budget
    )


def _margin_clearance(motion: _Motion, budget: CoordinateEnclosureBudget) -> object:
    return planar_stages._clearance_violation(
        motion, Fraction(-1, 16), Fraction(1, 2**40), budget=budget
    )


def _inward_contacts(motion: _Motion, budget: CoordinateEnclosureBudget) -> object:
    return planar_stages._first_contacts(motion, Fraction(-1, 2), None, budget=budget)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        pytest.param(_source_clearance, None, id="source-clearance"),
        pytest.param(_margin_clearance, None, id="margin-clearance"),
        pytest.param(_inward_contacts, Fraction(-13, 196), id="tip-wall-contact"),
    ],
)
def test_offset_certificate_stage_admits_its_arithmetic_before_forming_values(
    stage: Callable[[_Motion, CoordinateEnclosureBudget], object],
    expected: Fraction | None,
) -> None:
    motion = _notch_motion()
    with coordinate_reference_budget() as budget:
        outcome, performed = _counted_fraction_operations(lambda: stage(motion, budget))
        charged = budget.work_units
    # The profiler observes the arithmetic itself, independently of the ledger:
    # every rational operation the stage performs must have been admitted.
    assert 0 < performed <= charged
    if expected is None:
        assert outcome is None
    else:
        assert isinstance(outcome, tuple)
        lower = outcome[0]
        assert isinstance(lower, tuple)
        assert lower[0] == expected
    # One unit short, the admission refuses before any rational operation.
    limits = MeshCertificateLimits(maximum_work_units=charged - 1)
    with coordinate_reference_budget(limits) as starved:
        refused, starved_operations = _counted_fraction_operations(
            lambda: stage(motion, starved)
        )
        assert starved.work_units <= charged - 1
    assert type(refused) is CoordinateEnclosureResourceError
    assert starved_operations == 0


def test_unsupported_offset_reports_unknown_range_without_message_change() -> None:
    source = phx.geometry.brep_cylinder(
        2.0, 3.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    with pytest.raises(ValueError, match="sharp|cross-face|blend") as failure:
        phx.geometry.brep_offset(source, 0.125, tessellation=_LAZY)
    certificate = refusal_parameter_range(failure.value)
    assert certificate.interval == (None, None)
    assert certificate.end_status == ("unknown", "unknown")


def test_successful_smooth_offset_explicitly_reports_unknown_interval() -> None:
    source = phx.geometry.brep_sphere(
        2.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
    )
    result = phx.geometry.brep_offset(
        source, 0.125, tessellation=_LAZY, include_parameter_range=True
    )
    certificate = result.parameter_range
    assert certificate is not None
    assert certificate.interval == (None, None)
    assert certificate.end_status == ("unknown", "unknown")


def test_shared_range_rejects_uncertified_numeric_endpoints() -> None:
    unknown = unknown_parameter_range("radius", "unsupported analytic family")
    assert unknown.interval == (None, None)
    with pytest.raises(ValueError, match="unknown ends"):
        CertifiedParameterRange(
            "radius",
            (0.0, 1.0),
            end_status=("attained_exactly", "unknown"),
            constraints=("parameter_domain", "unknown"),
            evidence=("invalid claim",),
        )
    with pytest.raises(ValueError, match="unknown ends"):
        CertifiedParameterRange(
            "radius",
            (0.0, np.nan),
            end_status=("attained_exactly", "attained_exactly"),
            constraints=("parameter_domain", "reach"),
            evidence=("invalid claim",),
        )


def test_refusal_enrichment_preserves_exception_identity_args_and_message() -> None:
    error = ValueError(
        "construction.parameter must have a supported carrier; got an unsupported family"
    )
    message, args = str(error), error.args
    certificate = unknown_parameter_range("radius", "unsupported analytic family")
    assert attach_parameter_range(error, certificate) is error
    assert type(error) is ValueError
    assert error.args == args
    assert str(error) == message
    assert refusal_parameter_range(error) is certificate
