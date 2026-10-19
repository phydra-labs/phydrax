"""Independent extent and collision-pair oracles for construction ranges."""

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep._parameter_range import (
    attach_parameter_range,
    CertifiedParameterRange,
    refusal_parameter_range,
    unknown_parameter_range,
)


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


def test_l_prism_range_is_bounded_by_certified_collision_pair() -> None:
    source = _prism(concave=True)
    result = phx.geometry.brep_offset(
        source, -0.125, tessellation=_LAZY, include_parameter_range=True
    )
    certificate = result.parameter_range
    assert certificate is not None
    # The two arms each have width one: the source face-box separation
    # certificate cannot continue past half that width in either direction.
    assert certificate.interval == (-0.5, 0.5)
    assert certificate.end_status[0] in ("attained_exactly", "certified_inner_bound")
    assert certificate.constraints[1] == "edges_collide"
    assert len(certificate.entity_ids[1]) == 2
    assert all(entity in source.face_ids for entity in certificate.entity_ids[1])
    with pytest.raises(ValueError, match="collision pair|face vanishes") as failure:
        phx.geometry.brep_offset(source, -0.5, tessellation=_LAZY)
    assert refusal_parameter_range(failure.value).interval == certificate.interval


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
