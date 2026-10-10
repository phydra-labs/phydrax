#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_G = phx.geometry
_UNIT = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


def _discover(geometry: Any, domain: Any, **policy: Any) -> Any:
    return _G.discover_adaptive_implicit_surface(
        geometry,
        domain=domain,
        policy=_G.AdaptiveImplicitSurfacePolicy(**policy),
        source_id="adaptive",
    )


def _signed_volume(mesh: Any) -> float:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                vertices[faces[:, 0]],
                np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
            )
        )
        / 6.0
    )


def _inside_any(boxes: np.ndarray, point: np.ndarray) -> bool:
    return bool(np.any(np.all((boxes[:, 0] <= point) & (point <= boxes[:, 1]), axis=1)))


def test_certified_discovery_finds_component_inside_one_coarse_cell() -> None:
    # Radius 0.04 inside the level-2 cell [0, .5] x [0, .5] x [-.5, 0], away from
    # its corners and center: no coarse sample changes sign.
    geometry = _G.Sphere((0.37, 0.13, -0.36), 0.04).compile()
    coarse = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(_UNIT))
    with pytest.raises(ValueError, match="no surface crossings"):
        _G.discover_implicit_surface(geometry, coarse, source_id="fixed")

    result = _discover(
        geometry, _UNIT, initial_level=2, minimum_surface_level=2, maximum_level=8
    )

    assert result.evidence.accuracy == "certified"
    assert result.evidence.unresolved_count == 0
    assert result.mesh.topology.num_face_components == 1
    assert result.mesh.topology.euler_characteristic == 2
    assert result.cover.complete and result.cover.certified
    assert result.topology.certified and result.topology.established
    vertices = np.asarray(result.mesh.vertices)
    assert (
        np.max(np.abs(np.linalg.norm(vertices - (0.37, 0.13, -0.36), axis=1) - 0.04))
        < 0.01
    )


def test_sampled_enclosure_is_reported_as_sampled_and_can_miss_components() -> None:
    geometry = _G.Sphere((0.37, 0.13, -0.36), 0.04).compile()

    result = _discover(
        geometry,
        _UNIT,
        enclosure="sampled",
        initial_level=2,
        minimum_surface_level=2,
        maximum_level=8,
    )

    assert result.evidence.accuracy == "sampled"
    assert result.evidence.enclosure == "sampled"
    assert result.mesh is None
    assert result.cover is None and result.topology is None
    assert _G.AdaptiveImplicitSurfaceStatus.NO_SURFACE in result.evidence.status_flags
    query = result.volume.classify_points(np.asarray([[0.37, 0.13, -0.36]]))
    assert not query.certified


def test_tangential_zero_without_sign_change_is_reported_not_empty() -> None:
    # Two externally tangent balls intersect in the single point 0: the field is
    # nonnegative everywhere, so no sample can change sign.
    geometry = (
        _G.Sphere((-0.3, 0.0, 0.0), 0.3) & _G.Sphere((0.3, 0.0, 0.0), 0.3)
    ).compile()

    result = _discover(geometry, _UNIT + 0.0137, maximum_level=7)

    evidence = result.evidence
    assert result.mesh is None
    assert evidence.accuracy == "enclosed"
    assert _G.AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES in evidence.status_flags
    assert _inside_any(evidence.unresolved_boxes, np.zeros(3))
    assert np.any(evidence.unresolved_issues & int(_G.AdaptiveImplicitBoxIssue.SINGULAR))
    assert result.cover.complete and not result.cover.certified


def test_certified_torus_has_genus_one() -> None:
    geometry = _G.Torus((0.0123, -0.0071, 0.0173), 0.3, 0.9).compile()

    result = _discover(geometry, 1.3 * _UNIT, maximum_level=8)

    topology = result.mesh.topology
    assert result.evidence.accuracy == "certified"
    assert topology.watertight and topology.num_face_components == 1
    assert 1 - topology.euler_characteristic // 2 == 1
    assert result.topology.certified


def test_nearly_touching_spheres_are_separated_when_resolved() -> None:
    geometry = (
        _G.Sphere((-0.41, 0.013, 0.007), 0.4) | _G.Sphere((0.41, 0.013, 0.007), 0.4)
    ).compile()

    result = _discover(geometry, 1.1 * _UNIT, maximum_level=10)

    assert result.evidence.accuracy == "certified"
    assert result.mesh.topology.num_face_components == 2
    assert result.mesh.topology.euler_characteristic == 4


def test_nearly_touching_spheres_are_reported_when_unresolved() -> None:
    geometry = (
        _G.Sphere((-0.41, 0.013, 0.007), 0.4) | _G.Sphere((0.41, 0.013, 0.007), 0.4)
    ).compile()

    result = _discover(geometry, 1.1 * _UNIT, maximum_level=5)

    evidence = result.evidence
    assert evidence.accuracy == "enclosed"
    assert _G.AdaptiveImplicitSurfaceStatus.MAXIMUM_LEVEL_REACHED in evidence.status_flags
    # The unresolved leaves include the 0.02-wide gap between the balls.
    assert _inside_any(evidence.unresolved_boxes, np.asarray([0.0, 0.013, 0.007]))


def test_exact_zero_lattice_samples_are_certified_by_perturbation() -> None:
    # |(0.25, 0.25, 0.25)| is the radius in floating point, so eight lattice
    # vertices lie exactly on the zero set.
    radius = float(np.sqrt(0.1875))
    geometry = _G.Sphere((0.0, 0.0, 0.0), radius).compile()
    lattice = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(9) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(_UNIT))
    with pytest.raises(ValueError, match="ambiguous zero"):
        _G.discover_implicit_surface(geometry, lattice, source_id="fixed")

    result = _discover(geometry, _UNIT, maximum_level=7)

    assert result.evidence.accuracy == "certified"
    assert result.evidence.zero_vertex_count == 8
    assert result.mesh.topology.euler_characteristic == 2
    assert _signed_volume(result.mesh) == pytest.approx(
        4.0 / 3.0 * math.pi * radius**3, rel=0.1
    )


@pytest.mark.parametrize(
    ("budget", "flag"),
    [
        ({"maximum_boxes": 300}, "BOX_BUDGET_EXHAUSTED"),
        ({"maximum_evaluations": 6000}, "EVALUATION_BUDGET_EXHAUSTED"),
    ],
    ids=["boxes", "evaluations"],
)
def test_budget_exhaustion_reports_unresolved_boxes(budget: Any, flag: str) -> None:
    geometry = _G.Torus((0.0123, -0.0071, 0.0173), 0.3, 0.9).compile()

    result = _discover(geometry, 1.3 * _UNIT, maximum_level=8, **budget)

    evidence = result.evidence
    assert _G.AdaptiveImplicitSurfaceStatus[flag] in evidence.status_flags
    assert _G.AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES in evidence.status_flags
    assert evidence.accuracy == "enclosed"
    assert evidence.unresolved_count > 0
    assert evidence.unresolved_boxes.shape == (evidence.unresolved_count, 2, 3)
    assert np.all(evidence.unresolved_issues != 0)
    assert result.topology is None
    assert evidence.leaf_count <= budget.get("maximum_boxes", evidence.leaf_count)


def test_certified_sphere_is_outward_oriented_with_bounded_residuals() -> None:
    geometry = _G.Sphere((0.013, -0.021, 0.007), 0.75).compile()

    result = _discover(geometry, 1.4 * _UNIT, maximum_level=7, flatness_tolerance=0.02)

    evidence = result.evidence
    assert evidence.accuracy == "certified"
    assert evidence.intersection_checked and evidence.intersection_free
    assert evidence.maximum_flatness_bound <= 0.02
    assert evidence.maximum_anchor_residual <= 1.0e-9
    assert evidence.maximum_vertex_residual <= evidence.maximum_surface_box_diagonal
    assert _signed_volume(result.mesh) == pytest.approx(
        4.0 / 3.0 * math.pi * 0.75**3, rel=0.02
    )


def test_volume_query_classifies_points_and_boxes_with_enclosures() -> None:
    geometry = _G.Sphere((0.013, -0.021, 0.007), 0.75).compile()
    volume = _discover(geometry, 1.4 * _UNIT, maximum_level=6).volume

    points = volume.classify_points(
        np.asarray([[0.0, 0.0, 0.0], [1.2, 1.2, 1.2], [0.763, -0.021, 0.007]])
    )
    boxes = volume.classify_boxes(
        np.asarray([[-0.1, -0.1, -0.1], [0.7, -0.1, -0.1]]),
        np.asarray([[0.1, 0.1, 0.1], [0.9, 0.1, 0.1]]),
    )

    classes = _G.ImplicitVolumeClass
    assert points.certified and boxes.certified
    assert points.classes.tolist() == [classes.INSIDE, classes.OUTSIDE, classes.UNKNOWN]
    assert points.value_lower[2] <= 0.0 <= points.value_upper[2]
    assert boxes.classes.tolist() == [classes.INSIDE, classes.UNKNOWN]
    assert np.all(boxes.value_lower <= boxes.value_upper)


def test_sharp_union_crease_is_meshed_with_features_and_reported() -> None:
    geometry = (
        _G.Sphere((-0.2, 0.013, 0.007), 0.45) | _G.Sphere((0.25, 0.013, 0.007), 0.4)
    ).compile()

    result = _discover(geometry, 1.1 * _UNIT, maximum_level=7)

    evidence = result.evidence
    mesh = result.mesh
    assert mesh.topology.watertight and mesh.topology.num_face_components == 1
    assert mesh.topology.euler_characteristic == 2
    assert evidence.feature_vertex_count > 0
    assert result.feature_edges.shape[0] > 0
    # Whatever stays unresolved lies on the intersection circle x = 0.0722,
    # radius 0.358, of the two spheres.
    centers = 0.5 * (evidence.unresolved_boxes[:, 0] + evidence.unresolved_boxes[:, 1])
    radial = np.linalg.norm(centers[:, 1:] - (0.013, 0.007), axis=1)
    assert np.all(np.hypot(centers[:, 0] - 0.0722, radial - 0.358) < 0.05)
    assert evidence.accuracy in ("certified", "enclosed")


def test_surface_reaching_the_domain_boundary_is_refused_as_open() -> None:
    geometry = _G.Sphere((0.0, 0.0, 0.0), 0.75).compile()

    result = _discover(geometry, np.asarray([[-0.5, -1.0, -1.0], [1.0, 1.0, 1.0]]))

    evidence = result.evidence
    assert result.mesh is None
    assert (
        _G.AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING in evidence.status_flags
    )
    assert np.any(
        evidence.unresolved_issues & int(_G.AdaptiveImplicitBoxIssue.DOMAIN_BOUNDARY)
    )
    assert evidence.accuracy == "enclosed"


def test_lipschitz_enclosure_certifies_cover_but_not_topology() -> None:
    geometry = _G.Sphere((0.013, -0.021, 0.007), 0.75).compile()

    result = _discover(geometry, 1.4 * _UNIT, enclosure="lipschitz", maximum_level=6)

    assert result.evidence.accuracy == "enclosed"
    assert result.cover.complete
    assert result.topology is None
    assert result.mesh.topology.euler_characteristic == 2


def test_enclosure_selection_refuses_sources_without_valid_bounds() -> None:
    geometry = (
        _G.Sphere((-0.2, 0.0, 0.0), 0.45) | _G.Sphere((0.25, 0.0, 0.0), 0.4)
    ).compile()

    with pytest.raises(ValueError, match="sampled"):
        _discover(geometry, _UNIT, enclosure="lipschitz")


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"enclosure": "exact"}, ValueError),
        ({"initial_level": 4, "minimum_surface_level": 3}, ValueError),
        ({"maximum_level": 17}, ValueError),
        ({"maximum_boxes": 0}, ValueError),
        ({"maximum_level": 6.0}, TypeError),
        ({"flatness_tolerance": -1.0}, ValueError),
    ],
    ids=["selector", "level-order", "level-limit", "budget", "level-type", "flatness"],
)
def test_adaptive_policy_refuses_invalid_settings(
    options: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _G.AdaptiveImplicitSurfacePolicy(**options)
