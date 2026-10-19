"""Closed-form meridian-integral oracles for circular analytic edge shaping."""

import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.brep._differentiable import _FixedTopologyBRepKernel
from phydrax.geometry.brep._fillet import brep_chamfer, brep_fillet
from phydrax.geometry.brep._patches import (
    ConePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
)


# Empirical `test_closed_seam_area_and_volume_match_independent_closed_forms`: matched max |delta|=9.76996e-14; joint allowance fraction=0.000488498, margin=99.9512%.
_CLOSED_SEAM_VOLUME_CHANGE_ATOL = 2e-10
# Empirical `test_closed_seam_area_and_volume_match_independent_closed_forms`: matched max |delta|=3.64153e-14; joint allowance fraction=0.000182077, margin=99.9818%.
_CLOSED_SEAM_AREA_CHANGE_ATOL = 2e-10
# Empirical `test_closed_seam_radius_derivative_matches_independent_fd`: matched max |delta|=3.29779e-09; joint allowance fraction=0.0164889, margin=98.3511%.
_CLOSED_SEAM_RADIUS_DERIVATIVE_ATOL = 2e-7


Family: TypeAlias = Literal["cylinder", "cone", "sphere"]
_CONTRACT = phx.SpatialCoordinateContract.si()
_LAZY = phx.geometry.BRepTessellationPolicy(realize=False)


def _source(family: Family) -> tuple[phx.geometry.BRepModel, int, float, float]:
    if family == "cylinder":
        model = phx.geometry.brep_cylinder(
            3.0, 3.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
        )
        z0, top_radius = 3.0, 3.0
    elif family == "cone":
        model = phx.geometry.brep_cone(
            3.0, 2.0, 3.0, coordinate_contract=_CONTRACT, tessellation=_LAZY
        )
        z0, top_radius = 3.0, 2.0
    else:
        profile = phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            phx.geometry.ProfileLoop(
                ((0.0, -5.0), (4.0, 3.0), (0.0, 3.0)),
                (
                    phx.geometry.ProfileArc((0.0, 0.0)),
                    phx.geometry.ProfileLine(),
                    phx.geometry.ProfileLine(),
                ),
            ),
        )
        model = phx.geometry.brep_revolution(
            profile,
            (0.0, 0.0),
            (0.0, 1.0),
            coordinate_contract=_CONTRACT,
            tessellation=_LAZY,
        )
        z0, top_radius = 3.0, 4.0
    geometry = model.require_geometry()
    for index, (first, last) in enumerate(geometry.edge_vertices):
        uses = [
            face
            for face, loops in enumerate(geometry.face_loops)
            if any(
                geometry.coedge_edges[coedge] == index
                for loop in loops
                for coedge in loop
            )
        ]
        if (
            first == last
            and len(uses) == 2
            and any(isinstance(model.patches[face], PlanePatch) for face in uses)
            and any(
                isinstance(model.patches[face], (CylinderPatch, ConePatch, SpherePatch))
                for face in uses
            )
            and float(geometry.vertex_points[first, 2]) == z0
        ):
            return model, index, z0, top_radius
    raise RuntimeError(
        "ClosedSeamFixture.edge must contain a top circular seam; got no matching incidence"
    )


def _old_integrals(family: Family, first: float, last: float, /) -> tuple[float, float]:
    if family == "sphere":
        return math.pi * (
            25.0 * (last - first) - (last**3 - first**3) / 3.0
        ), 10.0 * math.pi * (last - first)
    slope = 0.0 if family == "cylinder" else -1.0 / 3.0
    volume = math.pi * (
        9.0 * (last - first)
        + 3.0 * slope * (last**2 - first**2)
        + slope**2 * (last**3 - first**3) / 3.0
    )
    area = (
        2.0
        * math.pi
        * math.hypot(1.0, slope)
        * (3.0 * (last - first) + slope * (last**2 - first**2) / 2.0)
    )
    return volume, area


def _fillet_delta(
    family: Family, z0: float, top_radius: float, radius: float, /
) -> tuple[float, float]:
    center_z = z0 - radius
    if family == "cylinder":
        center_r, contact_z, theta = 3.0 - radius, center_z, 0.0
    elif family == "cone":
        slope = -1.0 / 3.0
        norm = math.hypot(1.0, slope)
        center_r = 3.0 + slope * center_z - radius * norm
        contact_z = center_z - radius * slope / norm
        theta = -math.atan(slope)
    else:
        center_r = math.sqrt((5.0 - radius) ** 2 - center_z**2)
        contact_z = 5.0 * center_z / (5.0 - radius)
        theta = math.atan2(center_z, center_r)

    def primitive(angle: float) -> float:
        sine = math.sin(angle)
        return (
            center_r**2 * sine
            + center_r * radius * (angle + math.sin(2.0 * angle) / 2.0)
            + radius**2 * (sine - sine**3 / 3.0)
        )

    new_volume = math.pi * radius * (primitive(math.pi / 2.0) - primitive(theta))
    new_area = (
        2.0
        * math.pi
        * radius
        * (center_r * (math.pi / 2.0 - theta) + radius * (1.0 - math.sin(theta)))
    )
    old_volume, old_area = _old_integrals(family, contact_z, z0)
    return new_volume - old_volume, new_area - old_area + math.pi * (
        center_r**2 - top_radius**2
    )


def _chamfer_delta(
    family: Family, z0: float, top_radius: float, a: float, b: float, /
) -> tuple[float, float]:
    plane_r = top_radius - a
    if family == "cylinder":
        contact_r, contact_z = top_radius, z0 - b
    elif family == "cone":
        contact_z = z0 - b / math.hypot(1.0, -1.0 / 3.0)
        contact_r = 3.0 - contact_z / 3.0
    else:
        angle = math.atan2(z0, top_radius) - b / 5.0
        contact_r, contact_z = 5.0 * math.cos(angle), 5.0 * math.sin(angle)
    new_volume = (
        math.pi
        * (z0 - contact_z)
        * (plane_r**2 + plane_r * contact_r + contact_r**2)
        / 3.0
    )
    new_area = (
        math.pi * (plane_r + contact_r) * math.hypot(plane_r - contact_r, z0 - contact_z)
    )
    old_volume, old_area = _old_integrals(family, contact_z, z0)
    return new_volume - old_volume, new_area - old_area + math.pi * (
        plane_r**2 - top_radius**2
    )


@pytest.mark.parametrize("family", ["cylinder", "cone", "sphere"])
@pytest.mark.parametrize("kind", ["fillet", "chamfer"])
def test_closed_seam_area_and_volume_match_independent_closed_forms(
    family: Family, kind: str
) -> None:
    source, edge, z0, top_radius = _source(family)
    old = phx.geometry.prepare_brep_query(source).measures
    if kind == "fillet":
        result = brep_fillet(source, (source.edge_ids[edge],), 0.25, tessellation=_LAZY)
        volume_change, area_change = _fillet_delta(family, z0, top_radius, 0.25)
    else:
        result = brep_chamfer(
            source, (source.edge_ids[edge],), (0.25, 0.375), tessellation=_LAZY
        )
        volume_change, area_change = _chamfer_delta(family, z0, top_radius, 0.25, 0.375)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    new = phx.geometry.prepare_brep_query(model).measures
    assert float(new.solid_volumes[0] - old.solid_volumes[0]) == pytest.approx(
        volume_change, abs=_CLOSED_SEAM_VOLUME_CHANGE_ATOL
    )
    assert float(np.sum(new.face_areas) - np.sum(old.face_areas)) == pytest.approx(
        area_change, abs=_CLOSED_SEAM_AREA_CHANGE_ATOL
    )
    if kind == "fillet":
        assert result.g1_angle_bounds.shape == (2, 2)
        assert np.all(np.asarray(result.g1_angle_bounds)[:, 0] <= 0.0)
        assert np.all(np.asarray(result.g1_angle_bounds)[:, 1] < 1e-6)


@pytest.mark.parametrize("family", ["cylinder", "cone", "sphere"])
def test_closed_seam_radius_derivative_matches_independent_fd(family: Family) -> None:
    source, edge, z0, top_radius = _source(family)
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert result.status == "certified", result.reason
    radius_source = result.source
    assert radius_source is not None
    compiled = radius_source.compile(coordinate_unit=_CONTRACT.length_unit)
    kernel = compiled.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    index = compiled.schema.index(compiled.schema.parameter_ids[0])

    def volume(radius: Array) -> Array:
        return kernel.measure(compiled.state.replace_at(index, radius))

    step = 2.0**-15
    samples = []
    for radius in (0.25 - step, 0.25, 0.25 + step):
        state = compiled.state.replace_at(index, jnp.asarray(radius, dtype=jnp.float64))
        realized = kernel.realize(state)
        assert float(realized.seam_residual) <= float(realized.seam_bound)
        samples.append(float(kernel.measure(state)))
    difference = (samples[2] - samples[0]) / (2.0 * step)
    independent = (
        _fillet_delta(family, z0, top_radius, 0.25 + step)[0]
        - _fillet_delta(family, z0, top_radius, 0.25 - step)[0]
    ) / (2.0 * step)
    automatic = float(jax.grad(volume)(jnp.asarray(0.25, dtype=jnp.float64)))
    assert automatic == pytest.approx(difference, abs=_CLOSED_SEAM_RADIUS_DERIVATIVE_ATOL)
    assert automatic == pytest.approx(
        independent, abs=_CLOSED_SEAM_RADIUS_DERIVATIVE_ATOL
    )


@pytest.mark.parametrize("family", ["cylinder", "cone", "sphere"])
def test_closed_seam_requested_fillet_range_retains_meridian_chart_certificate(
    family: Family,
) -> None:
    source, edge, _, _ = _source(family)
    result = brep_fillet(
        source,
        (source.edge_ids[edge],),
        0.25,
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    certificate = result.parameter_range
    assert certificate is not None and certificate.parameter == "radius"
    lower, upper = certificate.interval
    assert lower == 0.0 and upper is not None and upper > 0.25
    assert certificate.end_status == ("attained_exactly", "certified_inner_bound")


@pytest.mark.parametrize("family", ["cylinder", "cone", "sphere"])
def test_closed_seam_requested_chamfer_range_never_estimates_missing_sign_proof(
    family: Family,
) -> None:
    source, edge, _, _ = _source(family)
    result = brep_chamfer(
        source,
        (source.edge_ids[edge],),
        (0.25, 0.375),
        tessellation=_LAZY,
        include_parameter_range=True,
    )
    assert result.status == "certified", result.reason
    certificate = result.parameter_range
    assert certificate is not None and certificate.parameter == "setback_scale"
    if family == "sphere":
        assert certificate.interval == (None, None)
        assert certificate.end_status == ("unknown", "unknown")
        assert "generatrix sign" in certificate.evidence[0]
    else:
        lower, upper = certificate.interval
        assert lower == 0.0 and upper is not None and upper > 1.0
        assert certificate.end_status == ("attained_exactly", "certified_inner_bound")
