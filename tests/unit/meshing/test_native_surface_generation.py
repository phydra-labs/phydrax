#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from examples._native_surface_sources import (
    AnalyticSurface,
    annulus,
    capped_cylinder,
    cylinder_sheet,
    folded_plates,
    sphere,
    torus,
)
from phydrax.discretization import CellMesh
from phydrax.geometry import (
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
)
from phydrax.geometry._chart_restriction import (
    ExactRationalChartRestriction,
    validate_chart_restrictions,
)
from phydrax.geometry.brep._patches import AbstractSurfacePatch
from phydrax.geometry.brep._placed import PlacedSurface
from phydrax.geometry.surface import InterfaceSide


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()
_CASES = {
    "sphere": (sphere, 0.35),
    "torus": (torus, 0.45),
    "capped_cylinder": (capped_cylinder, 0.3),
    "annulus": (annulus, 0.2),
    "cylinder_sheet": (cylinder_sheet, 0.25),
    "folded_plates": (folded_plates, 0.3),
}
# Largest principal curvature of each analytic case (independent of meshing).
_CURVATURE = {
    "sphere": 1.0,
    "torus": 1.0 / 0.6,
    "capped_cylinder": 1.0 / 0.8,
    "annulus": 1.0 / 0.4,
    "cylinder_sheet": 1.0,
    "folded_plates": 0.0,
}


def _scope(domain: Any, dimension: int, entities: Any) -> Any:
    return M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(entities, dtype=np.int64),
    )


def _specification(domain: Any, size: float, **arguments: Any) -> Any:
    scope = _scope(domain, 2, np.arange(len(domain.patches)))
    return M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, size, strength=M.SizeControlStrength.SOFT),
        ),
        **arguments,
    )


def _provider() -> Any:
    return M.NativeMeshingProvider(M.NativeMeshingOptions("parametric_surface"))


def _mesh(domain: Any, specification: Any) -> Any:
    plan = _provider().plan(
        M.NativeSurfaceSource(domain), specification, coordinate_contract=_SI
    )
    return plan.execute()


@functools.cache
def _case(name: str) -> tuple[AnalyticSurface, float, Any]:
    builder, size = _CASES[name]
    surface = builder()
    return surface, size, _mesh(surface.domain, _specification(surface.domain, size))


def _arrays(result: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(result.mesh.coordinates, dtype=np.float64)
    faces = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    return vertices, faces


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def _edge_uses(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges and how many faces use each (independent counting)."""

    return np.unique(np.sort(_directed_edges(faces), axis=1), axis=0, return_counts=True)


def _area(vertices: np.ndarray, faces: np.ndarray) -> float:
    corners = vertices[faces]
    return float(
        0.5
        * np.sum(
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
    )


def _minimum_angle(vertices: np.ndarray, faces: np.ndarray) -> float:
    corners = vertices[faces]
    first = np.roll(corners, -1, axis=1) - corners
    second = np.roll(corners, -2, axis=1) - corners
    cosine = np.sum(first * second, axis=2) / (
        np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2)
    )
    return float(np.min(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _area_tolerance(name: str, size: float, area: float) -> float:
    # Inscribed flat triangles lose O((kappa h)^2) relative area; feature curves
    # are chords, so straight features are exact up to rounding.
    return area * max(0.25 * (_CURVATURE[name] * size) ** 2, 1.0e-12)


@pytest.mark.parametrize("name", ["sphere", "torus", "capped_cylinder"])
def test_closed_surfaces_are_watertight_outward_and_measured(name: str) -> None:
    surface, size, result = _case(name)
    vertices, faces = _arrays(result)
    edges, uses = _edge_uses(faces)
    directed = _directed_edges(faces)
    corners = vertices[faces]
    signed_volume = float(
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2])) / 6.0
    )

    assert result.audit.passed
    assert np.all(uses == 2)
    assert np.unique(directed, axis=0).shape[0] == directed.shape[0]
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == (
        surface.euler_characteristic
    )
    assert signed_volume > 0.0
    assert np.unique(vertices, axis=0).shape[0] == vertices.shape[0]
    assert float(np.max(surface.distance(vertices))) <= 1.0e-12
    assert abs(_area(vertices, faces) - surface.area) <= _area_tolerance(
        name, size, surface.area
    )


@pytest.mark.parametrize(
    ("name", "boundary_loops"),
    [("annulus", 2), ("cylinder_sheet", 1), ("folded_plates", 1)],
)
def test_open_sheets_keep_boundaries_and_orientation(
    name: str, boundary_loops: int
) -> None:
    surface, size, result = _case(name)
    vertices, faces = _arrays(result)
    edges, uses = _edge_uses(faces)
    directed = _directed_edges(faces)
    boundary = edges[uses == 1]
    boundary_vertices = np.unique(boundary)
    # Boundary loops = components of the boundary edge graph (each vertex has
    # two boundary edges on a manifold sheet).
    degree = np.bincount(boundary.reshape(-1), minlength=vertices.shape[0])
    euler_boundary = boundary_vertices.size - boundary.shape[0]

    assert result.audit.passed
    assert np.all(uses <= 2)
    assert np.unique(directed, axis=0).shape[0] == directed.shape[0]
    assert np.all(degree[boundary_vertices] == 2)
    assert euler_boundary == 0
    # Genus-zero sheets: V - E + F = 2 - (boundary loops).
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 2 - boundary_loops
    assert float(np.max(surface.distance(vertices))) <= 1.0e-12
    assert abs(_area(vertices, faces) - surface.area) <= _area_tolerance(
        name, size, surface.area
    )


def test_shared_curve_vertices_are_identical_across_patches() -> None:
    surface, _, result = _case("folded_plates")
    vertices, faces = _arrays(result)
    face_patches = {
        patch.name: set(np.asarray(patch.scope.entity_ids).tolist())
        for patch in result.patches
    }
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids)
    first = np.isin(face_ids, sorted(face_patches["surface:0"]))
    edges, uses = _edge_uses(faces)
    on_fold = np.all(
        (np.abs(vertices[edges][..., 0]) <= 1.0e-12)
        & (np.abs(vertices[edges][..., 2]) <= 1.0e-12),
        axis=1,
    )
    first_edges = {tuple(edge) for edge in np.sort(_directed_edges(faces[first]), axis=1)}
    second_edges = {
        tuple(edge) for edge in np.sort(_directed_edges(faces[~first]), axis=1)
    }
    fold = [tuple(edge) for edge in edges[on_fold]]

    assert surface.closed is False
    assert len(fold) >= 4
    assert np.all(uses[on_fold] == 2)
    assert all(edge in first_edges and edge in second_edges for edge in fold)
    assert np.unique(vertices, axis=0).shape[0] == vertices.shape[0]


def test_hard_minimum_angle_request_is_met_on_the_physical_surface() -> None:
    surface = sphere()
    angle = np.radians(20.0)
    result = _mesh(
        surface.domain,
        _specification(
            surface.domain, 0.4, quality_target=M.MeshQualityTarget(minimum_angle=angle)
        ),
    )
    vertices, faces = _arrays(result)

    assert _minimum_angle(vertices, faces) >= angle
    assert dict(result.compliance.requested)["minimum_angle"] == angle


def test_exact_rational_chart_restriction_requires_unique_source_authority() -> None:
    charts = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (float(Fraction(1, 3)),) * 2),
        dtype=np.float64,
    )
    required = np.asarray((False, False, True), dtype=np.bool_)
    with pytest.raises(ValueError, match="exactly one canonical rational authority"):
        validate_chart_restrictions(
            charts,
            required,
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            "rational-chart-source",
            0,
        )

    exact, restrictions, errors = validate_chart_restrictions(
        charts,
        required,
        np.asarray((2,), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        charts[np.asarray(((0, 1),), dtype=np.int64)],
        np.asarray(((1, 3),), dtype=np.int64),
        "rational-chart-source",
        0,
    )
    assert exact[2, 0] == exact[2, 1] == Fraction(1, 3)
    assert Fraction(float(charts[2, 0])) != Fraction(1, 3)
    assert np.all(errors[2] > 0.0)
    assert len(restrictions) == 1 and restrictions[0].restriction_id
    with pytest.raises(ValueError, match="primitive"):
        ExactRationalChartRestriction(
            "rational-chart-source",
            0,
            2,
            (0, 1),
            ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(1))),
            Fraction(1, 3),
            charts[2],
            root_coefficients=(0, 0),
            isolation=(Fraction(0), Fraction(1)),
        )


def test_unrecoverable_local_trim_ribbon_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = annulus()

    def unrecoverable(
        self: MeshingDomain,
        patch: int,
        use: PatchCurveUse,
        parameters: np.ndarray,
        points: np.ndarray,
        /,
        *,
        intervals: np.ndarray | None = None,
    ) -> np.ndarray:
        del self, patch, use, points
        count = parameters.size - 1 if intervals is None else intervals.size
        return np.full((count,), 1.0, dtype=np.float64)

    monkeypatch.setattr(MeshingDomain, "trim_ribbon_bounds", unrecoverable)
    feature = M.ProtectedFeature(
        _scope(surface.domain, 2, (0,)),
        M.FeatureKind.SURFACE,
        maximum_deviation=1.0e-3,
    )
    with pytest.raises(
        M.MeshingFailure, match="unrecoverable local trim ribbon"
    ) as refused:
        _mesh(
            surface.domain,
            _specification(
                surface.domain,
                0.2,
                protected_features=(feature,),
            ),
        )
    assert refused.value.category is M.MeshingFailureCategory.UNSUPPORTED_COMBINATION


def test_surface_fidelity_request_bounds_sampled_deviation() -> None:
    surface = cylinder_sheet()
    bound = 2.0e-3
    feature = M.ProtectedFeature(
        _scope(surface.domain, 2, (0,)), M.FeatureKind.SURFACE, maximum_deviation=bound
    )
    result = _mesh(
        surface.domain,
        _specification(surface.domain, 0.4, protected_features=(feature,)),
    )
    vertices, faces = _arrays(result)
    corners = vertices[faces]
    weights = np.asarray([[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [4, 1, 1]]) / 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    samples = np.matmul(weights[None], corners).reshape((-1, 3))

    assert float(np.max(surface.distance(samples))) <= 1.1 * bound
    assert result.associations[0].complete


def test_stale_source_revision_is_refused_before_work() -> None:
    current = sphere(revision="r2")
    stale = _specification(sphere(revision="r1").domain, 0.4)

    with pytest.raises(M.MeshingFailure) as refused:
        _provider().plan(
            M.NativeSurfaceSource(current.domain), stale, coordinate_contract=_SI
        )
    assert refused.value.category is M.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED


def test_curvature_only_source_request_meets_continuous_normal_bound() -> None:
    surface = cylinder_sheet()
    scope = _scope(surface.domain, 2, (0,))
    angle = 0.4
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(M.CurvatureSizeControl(scope, angle),),
    )
    result = _mesh(surface.domain, specification)
    achieved = dict(result.compliance.achieved)
    key = f"source_size:{specification.size_controls[0].control_id}:normal_angle"
    assert achieved[key] <= angle
    vertices, faces = _arrays(result)
    edges, _ = _edge_uses(faces)
    assert np.max(
        np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    ) <= 2 * np.sin(angle / 2) * (1 + 1e-8)
    assert result.certification is not None and result.certification.passed


def test_exhausted_work_budget_refuses_publication() -> None:
    surface = sphere()
    with pytest.raises(M.MeshingFailure) as refused:
        _mesh(
            surface.domain,
            _specification(
                surface.domain, 0.2, limits=M.MeshingLimits(maximum_work_units=2_000)
            ),
        )
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED


@pytest.mark.parametrize(
    "limit_name", ("maximum_geometry_queries", "maximum_scratch_bytes")
)
def test_public_surface_source_capacity_refuses_without_changing_authority(
    limit_name: str,
) -> None:
    domain = sphere().domain
    original_id = domain.domain_id
    corners = np.asarray(domain.corner_points).copy()
    limits = M.MeshingLimits(**{limit_name: 1})
    with pytest.raises(M.MeshingFailure) as refused:
        _mesh(domain, _specification(domain, 0.2, limits=limits))
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert domain.domain_id == original_id
    np.testing.assert_array_equal(domain.corner_points, corners)


def test_native_surface_supports_sided_interface_attachment() -> None:
    surface, _, result = _case("sphere")
    patch = next(value for value in result.patches if value.name == "surface:0")
    faces = next(
        value
        for value in result.associations
        if value.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
    )
    part = M.MeshPart("shell", result)
    attachment = M.MeshInterfaceAttachment(
        part,
        patch,
        faces,
        (surface.domain.entity_id(2, 0),),
        tolerance=0.05,
        carrier_side=InterfaceSide.MINUS,
    )
    vertices, triangles = _arrays(result)
    corners = vertices[triangles]
    outward = np.sum(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        * corners.mean(axis=1),
        axis=1,
    )

    assert np.all(outward > 0.0)
    assert attachment.orientation == 1
    assert attachment.geometry_source_revision == "r1"


@pytest.mark.parametrize("name", ["sphere", "cone"], ids=["smooth-poles", "conical-pole"])
def test_native_brep_poles_keep_hard_angular_fidelity_and_source_identity(
    name: str,
) -> None:
    policy = phx.geometry.BRepTessellationPolicy(
        linear_deflection=0.03,
        angular_deflection=0.3,
        maximum_triangles=20_000,
    )
    if name == "sphere":
        model = phx.geometry.brep_sphere(
            1.0, coordinate_contract=_SI, tessellation=policy
        )
        pole_coordinates = np.asarray(
            ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), dtype=np.float64
        )
    else:
        model = phx.geometry.brep_cone(
            1.0, 0.0, 2.0, coordinate_contract=_SI, tessellation=policy
        )
        pole_coordinates = np.asarray(((0.0, 0.0, 2.0),), dtype=np.float64)
    domain = phx.geometry.MeshingDomain.from_brep(model)
    result = _mesh(
        domain,
        _specification(
            domain,
            0.35,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, np.arange(len(domain.patches))),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.03,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    _, uses = _edge_uses(faces)
    assert np.all(uses == 2)
    assert (
        np.max(np.asarray(model.tessellation_deviation_bounds))
        <= policy.linear_deflection
    )
    assert (
        np.max(np.asarray(model.tessellation_normal_bounds)) <= policy.angular_deflection
    )
    # There is one physical vertex per authored pole, not one per UV copy.
    poles = np.any(np.all(vertices[:, None] == pole_coordinates[None], axis=-1), axis=1)
    assert np.count_nonzero(poles) == pole_coordinates.shape[0]
    assert np.unique(vertices[poles], axis=0).shape[0] == pole_coordinates.shape[0]
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete
    assert all(association.complete for association in result.associations)


def test_hard_sphere_size_and_accuracy_are_met_together() -> None:
    surface = sphere()
    scope = _scope(surface.domain, 2, (0,))
    size, deviation = 0.4, 0.04
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(
                scope,
                size,
                maximum_size=0.6,
                strength=M.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=M.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            M.ProtectedFeature(
                scope,
                M.FeatureKind.SURFACE,
                maximum_deviation=deviation,
            ),
        ),
    )
    result = _mesh(surface.domain, specification)
    points, faces = _arrays(result)
    edges, counts = _edge_uses(faces)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    assert 0.2 <= np.median(lengths) <= 0.6
    assert np.max(lengths) <= 0.6
    assert np.all(counts == 2)
    assert result.certification is not None and result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.mesh_to_source_upper <= deviation
    assert result.certification.fidelity.source_to_mesh_upper <= deviation


def test_partial_circle_trim_has_continuous_source_fidelity() -> None:
    from phydrax.geometry.brep._intersection import NativePeriodEndpoint

    geometry = phx.geometry
    circle = geometry.CircleCurve(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        1.0,
    )
    loop = (
        geometry.PatchCurveUse(
            0,
            circle,
            0.0,
            np.pi,
            last_root=NativePeriodEndpoint(circle, None, None, turns=Fraction(1, 2)),
        ),
        geometry.PatchCurveUse(1, geometry.LineCurve((-1.0, 0.0), (2.0, 0.0)), 0.0, 1.0),
    )
    domain = geometry.MeshingDomain(
        (
            geometry.MeshingSurfacePatch(
                geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                (loop,),
            ),
        ),
        (geometry.MeshingDomainCurve(0, 1), geometry.MeshingDomainCurve(1, 0)),
        2,
        source_id="partial-circle-face",
        source_revision="r1",
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.2,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    assert np.all(vertices[:, 1] >= -1e-12)
    assert abs(_area(vertices, faces) - 0.5 * np.pi) < 0.03
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert all(association.complete for association in result.associations)


@pytest.mark.parametrize(
    "middle_weight", [0.75, 1.0], ids=["rational-trim", "polynomial-trim"]
)
def test_spline_trim_cover_is_bound_to_actual_prescribed_chords(
    middle_weight: float,
) -> None:
    geometry = phx.geometry
    controls = np.asarray(
        (
            ((1, 0), (1, 1), (0, 1)),
            ((0, 1), (-1, 1), (-1, 0)),
            ((-1, 0), (-1, -1), (0, -1)),
            ((0, -1), (1, -1), (1, 0)),
        ),
        dtype=np.float64,
    )
    loop = tuple(
        geometry.PatchCurveUse(
            index,
            geometry.BSplineCurve.bezier(points, (1.0, middle_weight, 1.0)),
            0.0,
            1.0,
        )
        for index, points in enumerate(controls)
    )
    domain = geometry.MeshingDomain(
        (
            geometry.MeshingSurfacePatch(
                geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                (loop,),
            ),
        ),
        tuple(geometry.MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id=f"spline-trim-{middle_weight}",
        source_revision="r1",
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.2,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    assert result.certification is not None
    assert result.certification.fidelity is not None
    fidelity = result.certification.fidelity
    assert fidelity.status == "certified"
    assert fidelity.chart_coverage is not None
    assert fidelity.chart_coverage.complete
    assert np.max(fidelity.chart_coverage.deviation_bounds) <= 0.01
    assert all(association.complete for association in result.associations)


def _quadratic_band(middle_offset: float) -> MeshingDomain:
    """Lower quadratic wall and its .05 translate, joined by straight ends."""
    lower = ((0.0, 0.0), (0.5, 0.2), (1.0, 0.0))
    upper = ((1.0, 0.05), (0.5, 0.2 + middle_offset), (0.0, 0.05))
    edges = (lower, (lower[-1], upper[0]), upper, (upper[-1], lower[0]))
    loop = tuple(
        phx.geometry.PatchCurveUse(
            index,
            phx.geometry.BSplineCurve.bezier(
                np.asarray(points, dtype=np.float64),
                np.ones(len(points), dtype=np.float64),
            ),
            0.0,
            1.0,
        )
        for index, points in enumerate(edges)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id=f"quadratic-narrow-band-{middle_offset}",
        source_revision="r1",
    )


def test_coarse_curved_narrow_band_proves_its_original_chord_chain() -> None:
    # One coarse arc per curve half has overlapping lower/upper enclosures
    # although the source band is .05 wide; the request is not tightened.
    domain = _quadratic_band(0.05)
    result = _mesh(domain, _specification(domain, 2.0))
    assert result.certification is not None and result.certification.passed
    vertices, faces = _arrays(result)
    edges, uses = _edge_uses(faces)
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 1
    boundary = vertices[np.unique(edges[uses == 1])]
    wall = 0.4 * boundary[:, 0] * (1.0 - boundary[:, 0])
    assert np.all(
        np.isclose(boundary[:, 1], wall, rtol=0.0, atol=1.0e-12)
        | np.isclose(boundary[:, 1], wall + 0.05, rtol=0.0, atol=1.0e-12)
        | np.isin(boundary[:, 0], (0.0, 1.0))
    )
    assert all(association.complete for association in result.associations)


def test_self_crossing_curved_band_is_refused_before_publication() -> None:
    # The upper curve dips through the lower one: the exact chart chain of the
    # source loop crosses itself, which no trim proof may repair.
    domain = _quadratic_band(-0.5)
    with pytest.raises(ValueError, match="cross"):
        _mesh(domain, _specification(domain, 2.0))


def test_trimmed_annulus_certifies_both_hole_and_outer_source_chains() -> None:
    surface = annulus()
    domain = surface.domain
    result = _mesh(
        domain,
        _specification(
            domain,
            0.2,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    edges, uses = _edge_uses(faces)
    boundary = edges[uses == 1]
    radii = np.linalg.norm(vertices[boundary], axis=2)
    assert np.all((radii < 0.41) | (radii > 0.99))
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 0
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete


@pytest.mark.parametrize(
    "declared_pole", [False, True], ids=["ordinary-coincidence", "source-pole"]
)
def test_radial_split_retains_a_collapsed_child_only_for_a_declared_pole(
    declared_pole: bool,
) -> None:
    from phydrax._meshcore import MeshcoreStatus, surface_reconnect

    charts = np.asarray(((0, 0), (1, 0), (0.5, 1), (-0.5, 1)), dtype=np.float64)
    points = np.asarray(
        (
            (0, 0, 0),
            (0, 0, 0),
            (np.cos(0.5), np.sin(0.5), 0),
            (np.cos(-0.5), np.sin(-0.5), 0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    constraints = np.asarray(((1, 0, 1), (1, 1, 0)), dtype=np.bool_)
    normals = np.tile(np.asarray((0, 0, -1), dtype=np.float64), (4, 1))
    split_chart = np.asarray(((0.25, 0.5),), dtype=np.float64)
    split_point = np.asarray(
        ((0.5 * np.cos(0.25), 0.5 * np.sin(0.25), 0),), dtype=np.float64
    )
    pole_ids = np.asarray((0, 0, -1, -1), dtype=np.int64) if declared_pole else None
    triangles, _, ids, status, _ = surface_reconnect(
        charts,
        points,
        normals,
        cells,
        constraints,
        split_chart,
        split_point,
        normals[:1],
        np.asarray((0.0,), dtype=np.float64),
        np.asarray((0,), dtype=np.int32),
        sweep=False,
        max_triangles=4,
        work_limit=1_000,
        vertex_pole_ids=pole_ids,
        insert_edges=None,
    )
    if not declared_pole:
        assert ids[0] == -1
        assert status[0] == MeshcoreStatus.DEGENERATE_INPUT
        np.testing.assert_array_equal(triangles, cells)
        return
    assert status[0] == MeshcoreStatus.OK
    assert ids[0] == 4
    xyz = np.concatenate((points, split_point))[triangles]
    physical_normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    assert np.count_nonzero(np.linalg.norm(physical_normals, axis=1) == 0) == 1
    assert np.all(physical_normals[:, 2] <= 0)
    uv = np.concatenate((charts, split_chart))[triangles]
    chart_determinants = (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1]) - (
        uv[:, 1, 1] - uv[:, 0, 1]
    ) * (uv[:, 2, 0] - uv[:, 0, 0])
    assert np.all(chart_determinants > 0)
    assert np.sum(chart_determinants) == 2.0


@pytest.mark.parametrize("edge_intent", ["paired", "constrained", "stale"])
def test_explicit_rounded_surface_edge_uses_complete_pair_or_refuses_unchanged(
    edge_intent: str,
) -> None:
    from phydrax._meshcore import exact_orient2d, MeshcoreStatus, surface_reconnect

    charts = np.asarray(
        (
            (0.24543692606170273, 0.024543692606170175),
            (0.3436116964863838, -0.024543692606170175),
            (0.3, 0.1),
            (0.3, -0.1),
        ),
        dtype=np.float64,
    )
    midpoint = np.mean(charts[:2], axis=0)[None]
    assert exact_orient2d(charts[:1], charts[1:2], midpoint)[0] != 0
    points = np.column_stack((charts, np.zeros(4, dtype=np.float64)))
    normals = np.tile(np.asarray((0.0, 0.0, 1.0), dtype=np.float64), (4, 1))
    cells = np.asarray(((0, 1, 2), (1, 0, 3)), dtype=np.int32)
    constraints = np.asarray(((True, True, False), (True, True, False)), dtype=np.bool_)
    if edge_intent == "constrained":
        constraints[:, 2] = True
    selected = np.asarray(((2, 3) if edge_intent == "stale" else (0, 1),), dtype=np.int32)
    original = tuple(
        value.copy() for value in (charts, points, normals, cells, constraints)
    )
    triangles, flags, ids, status, counters = surface_reconnect(
        charts,
        points,
        normals,
        cells,
        constraints,
        midpoint,
        np.column_stack((midpoint, np.zeros(1, dtype=np.float64))),
        normals[:1],
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.int32),
        insert_edges=selected,
        sweep=False,
        max_triangles=4,
        work_limit=1_000,
    )
    for value, before in zip((charts, points, normals, cells, constraints), original):
        np.testing.assert_array_equal(value, before)
    if edge_intent != "paired":
        assert ids[0] == -1
        expected = (
            MeshcoreStatus.CONSTRAINT_INTERSECTION
            if edge_intent == "constrained"
            else MeshcoreStatus.INVALID_INPUT
        )
        assert status[0] == expected
        np.testing.assert_array_equal(triangles, cells)
        np.testing.assert_array_equal(flags, constraints)
        return
    assert status[0] == MeshcoreStatus.OK
    assert ids.tolist() == [4]
    assert triangles.shape == (4, 3)
    assert counters[2] == 0
    exact_charts = [tuple(Fraction(float(value)) for value in point) for point in charts]
    exact_charts.append(
        tuple(
            (first + second) / 2
            for first, second in zip(exact_charts[0], exact_charts[1], strict=True)
        )
    )
    for triangle in triangles:
        first, second, third = (exact_charts[int(vertex)] for vertex in triangle)
        determinant = (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])
        assert determinant > 0
    uv = np.concatenate((charts, midpoint))[triangles]
    assert np.all(exact_orient2d(uv[:, 0], uv[:, 1], uv[:, 2]) > 0)
    edges = [
        tuple(sorted((int(triangle[(edge + 1) % 3]), int(triangle[(edge + 2) % 3]))))
        for triangle in triangles
        for edge in range(3)
    ]
    assert (0, 1) not in edges
    assert edges.count((0, 4)) == edges.count((1, 4)) == 2
    boundary = {
        tuple(sorted((int(triangle[(edge + 1) % 3]), int(triangle[(edge + 2) % 3]))))
        for triangle, constrained in zip(triangles, flags)
        for edge in range(3)
        if constrained[edge]
    }
    assert boundary == {(0, 2), (1, 2), (0, 3), (1, 3)}


@pytest.mark.parametrize(
    "edges, error",
    [
        pytest.param([[0.0, 1.0]], TypeError, id="floating-endpoints"),
        pytest.param([[False, True]], TypeError, id="boolean-endpoints"),
        pytest.param([[0, -1]], ValueError, id="mixed-sentinel"),
        pytest.param([[0, 0]], ValueError, id="repeated-endpoint"),
        pytest.param([[0, 4]], ValueError, id="outside-source"),
        pytest.param([[0, 1, 2]], ValueError, id="wrong-rank"),
    ],
)
def test_surface_selected_edge_validation_rejects_untruthful_rows(
    edges: object, error: type[Exception]
) -> None:
    from phydrax._meshcore import surface_reconnect

    charts = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    points = np.column_stack((charts, np.zeros(3, dtype=np.float64)))
    with pytest.raises(error, match="insert_edges"):
        surface_reconnect(
            charts,
            points,
            np.tile(np.asarray((0.0, 0.0, 1.0), dtype=np.float64), (3, 1)),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.zeros((1, 3), dtype=np.bool_),
            np.asarray(((0.25, 0.25),), dtype=np.float64),
            np.asarray(((0.25, 0.25, 0.0),), dtype=np.float64),
            np.asarray(((0.0, 0.0, 1.0),), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.int32),
            insert_edges=edges,
            sweep=False,
            max_triangles=3,
            work_limit=1_000,
        )


def test_implicit_trim_root_uncertainty_keeps_exact_junctions_and_a_certified_cover() -> (
    None
):
    from phydrax.geometry.brep._intersection import TrimIntersectionRoot, TrimRootEndpoint
    from phydrax.geometry.brep._intersection_curve import CurveTrimSegment
    from phydrax.geometry.brep._root_bindings import BRepRootSupport, BRepVertexRoot

    geometry = phx.geometry
    plane = geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    circle = geometry.CircleCurve((0, 0), (1, 0), (0, 1), 1.0)
    line = geometry.LineCurve((0.3, 0), (0, 1))
    circle_trim = CurveTrimSegment(circle, 0.0, 2 * np.pi)
    line_trim = CurveTrimSegment(line, -2.0, 2.0)
    angle, height = np.arccos(0.3), np.sqrt(0.91)
    upper_parameters = np.asarray(
        (angle / (2 * np.pi), (height + 2) / 4), dtype=np.float64
    )
    lower_parameters = np.asarray(
        (1 - angle / (2 * np.pi), (2 - height) / 4), dtype=np.float64
    )
    # Asymmetric isolating boxes deliberately make their bounded midpoint
    # realizations disagree, although the implicit junction is one exact root.
    before = np.asarray((2e-6, 3e-6), dtype=np.float64)
    after = np.asarray((5e-6, 1e-6), dtype=np.float64)
    upper = TrimIntersectionRoot(
        circle_trim,
        line_trim,
        parameter_lower=upper_parameters - before,
        parameter_upper=upper_parameters + after,
    )
    lower = TrimIntersectionRoot(
        circle_trim,
        line_trim,
        parameter_lower=lower_parameters - before,
        parameter_upper=lower_parameters + after,
    )
    upper_circle, lower_circle = (
        TrimRootEndpoint(upper, "first"),
        TrimRootEndpoint(lower, "first"),
    )
    upper_line, lower_line = (
        TrimRootEndpoint(upper, "second"),
        TrimRootEndpoint(lower, "second"),
    )
    upper_vertex = BRepVertexRoot(BRepRootSupport(plane, upper))
    lower_vertex = BRepVertexRoot(BRepRootSupport(plane, lower))
    loop = (
        geometry.PatchCurveUse(
            0,
            circle,
            upper_circle.parameter,
            lower_circle.parameter,
            first_root=upper_circle,
            last_root=lower_circle,
            start_vertex_root=upper_vertex,
            end_vertex_root=lower_vertex,
        ),
        geometry.PatchCurveUse(
            1,
            line,
            lower_line.parameter,
            upper_line.parameter,
            first_root=lower_line,
            last_root=upper_line,
            start_vertex_root=lower_vertex,
            end_vertex_root=upper_vertex,
        ),
    )
    domain = geometry.MeshingDomain(
        (geometry.MeshingSurfacePatch(plane, (loop,)),),
        (geometry.MeshingDomainCurve(0, 1), geometry.MeshingDomainCurve(1, 0)),
        2,
        source_id="implicit-root-circle-cut",
        source_revision="r1",
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.2,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    expected_area = np.pi - np.arccos(0.3) + 0.3 * np.sqrt(0.91)
    assert abs(_area(vertices, faces) - expected_area) < 0.03
    assert np.max(vertices[:, 0]) <= 0.30001
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete
    assert all(association.complete for association in result.associations)


def _source_square(
    curves: tuple[int, int, int, int], height: float, reversed: bool = False
) -> MeshingSurfacePatch:
    bottom, right, top, left = curves
    return MeshingSurfacePatch(
        PlanePatch((0, 0, height), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(bottom, LineCurve((0, 0), (1, 0)), 0, 1),
                PatchCurveUse(right, LineCurve((1, 0), (0, 1)), 0, 1),
                PatchCurveUse(top, LineCurve((0, 1), (1, 0)), 1, 0),
                PatchCurveUse(left, LineCurve((0, 0), (0, 1)), 1, 0),
            ),
        ),
        reversed=reversed,
    )


def _square_source_domain() -> MeshingDomain:
    return MeshingDomain(
        (_source_square((0, 1, 2, 3), 0),),
        tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0))),
        4,
        source_id="metric-square",
        source_revision="r1",
    )


def test_source_thin_gap_is_sized_from_physical_plane_lower_bounds() -> None:
    gap = 0.2
    domain = MeshingDomain(
        (_source_square((0, 1, 2, 3), 0), _source_square((4, 5, 6, 7), gap, True)),
        tuple(
            MeshingDomainCurve(*ends)
            for ends in (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            )
        ),
        8,
        source_id="thin-gap",
        source_revision="r1",
    )
    scope = _scope(domain, 2, (0, 1))
    control = M.ProximitySizeControl(_scope(domain, 2, (0,)), _scope(domain, 2, (1,)), 2)
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 0.4, strength=M.SizeControlStrength.SOFT),
            control,
        ),
    )
    result = _mesh(domain, specification)
    vertices, faces = _arrays(result)
    edges, _ = _edge_uses(faces)
    assert np.max(
        np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    ) <= gap / 2 * (1 + 1e-8)
    assert np.all((vertices[:, 2] == 0) | (vertices[:, 2] == gap))
    assert result.compliance.passed


@pytest.mark.parametrize("gap", (0.01, 2.0), ids=("thin-disconnected", "separated"))
def test_disconnected_source_sheets_publish_independent_exact_coverage(
    gap: float,
) -> None:
    domain = MeshingDomain(
        (_source_square((0, 1, 2, 3), 0), _source_square((4, 5, 6, 7), gap, True)),
        tuple(
            MeshingDomainCurve(*ends)
            for ends in (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            )
        ),
        8,
        source_id="disconnected-sheets",
        source_revision="r1",
    )
    scope = _scope(domain, 2, (0, 1))
    result = _mesh(
        domain,
        _specification(
            domain,
            0.3,
            protected_features=(
                M.ProtectedFeature(
                    scope,
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    corner_heights = vertices[faces, 2]
    assert np.all(
        np.all(corner_heights == 0, axis=1) | np.all(corner_heights == gap, axis=1)
    )
    for height, orientation in ((0.0, 1.0), (gap, -1.0)):
        selected = faces[np.all(corner_heights == height, axis=1)]
        assert selected.shape[0] > 0
        edges, uses = _edge_uses(selected)
        assert np.all((uses == 1) | (uses == 2))
        assert np.unique(selected).size - edges.shape[0] + selected.shape[0] == 1
        np.testing.assert_allclose(_area(vertices, selected), 1.0, rtol=0.0, atol=1.0e-12)
        corners = vertices[selected]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        assert np.all(orientation * normals[:, 2] > 0)
        assert np.all((corners[:, :, :2] >= 0) & (corners[:, :, :2] <= 1))
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete
    assert all(association.complete for association in result.associations)


def test_strong_physical_source_metric_drives_anisotropic_interior_edges() -> None:
    domain = _square_source_domain()
    background = CellMesh.from_triangles(
        np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    vertex_set = background.entity_set(0)
    metric_scope = M.MeshingScope(
        background.mesh_id,
        background.numeric_version,
        M.MeshingEntityKind.MESH,
        0,
        vertex_set.entity_set_id,
        vertex_set.entity_ids,
    )
    tensor = np.diag(np.asarray([400, 16, 16], dtype=np.float64))
    metric = M.MeshMetricField(
        metric_scope,
        np.broadcast_to(tensor, (4, 3, 3)),
        minimum_size=0.04,
        maximum_size=0.3,
        maximum_anisotropy=6,
    )
    control = M.BackgroundMetricControl(
        background, metric, _SI, maximum_metric_edge_length=1
    )
    result = _mesh(domain, _specification(domain, 0.4, background_metric=control))
    vertices, faces = _arrays(result)
    edges, _ = _edge_uses(faces)
    offset = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    metric_lengths = np.sqrt(np.sum(offset * (offset @ tensor), axis=1))
    assert np.max(metric_lengths) <= 1 + 1e-8
    midpoint = np.mean(vertices[edges], axis=1)
    interior = np.all((midpoint[:, :2] > 0.15) & (midpoint[:, :2] < 0.85), axis=1)
    assert np.median(np.abs(offset[interior, 0])) < 0.6 * np.median(
        np.abs(offset[interior, 1])
    )
    assert dict(result.compliance.achieved)["metric_compliance_location_pairs"] > 0


def test_two_source_surface_regions_preserve_materials_and_exact_shared_interface() -> (
    None
):
    domain = folded_plates().domain
    horizontal = M.RegionControl(
        _scope(domain, 2, (0,)), "horizontal", "fluid", M.RegionRole.FLUID
    )
    vertical = M.RegionControl(
        _scope(domain, 2, (1,)), "vertical", "wall", M.RegionRole.SOLID
    )
    interface = M.PatchControl(
        "shared-interface", _scope(domain, 1, (0,)), ("horizontal", "vertical")
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.3,
            region_controls=(horizontal, vertical),
            patch_controls=(interface,),
        ),
    )
    zones = {zone.name: zone for zone in result.zones}
    assert zones["horizontal"].material_id == "fluid"
    assert zones["vertical"].material_id == "wall"
    assert not set(np.asarray(zones["horizontal"].scope.entity_ids)).intersection(
        np.asarray(zones["vertical"].scope.entity_ids)
    )
    patch = next(value for value in result.patches if value.name == "shared-interface")
    source_curve = next(value for value in result.labels if value.name == "curve:0")
    np.testing.assert_array_equal(patch.scope.entity_ids, source_curve.scope.entity_ids)
    assert set(patch.adjacent_zone_ids) == {zone.zone_id for zone in result.zones}


def test_source_volume_region_is_an_oriented_overlapping_boundary_not_fake_cells() -> (
    None
):
    domain = sphere().domain
    region_scope = _scope(domain, 3, (0,))
    region = M.RegionControl(region_scope, "fluid-volume", "water", M.RegionRole.FLUID)
    wall = M.PatchControl("wall", _scope(domain, 2, (0,)), ("fluid-volume",))
    result = _mesh(
        domain,
        _specification(domain, 0.4, region_controls=(region,), patch_controls=(wall,)),
    )
    assert not result.zones
    evidence = result.region_boundary_evidence[0]
    assert evidence.source_scope.scope_id == region_scope.scope_id
    assert evidence.material_id == "water"
    assert evidence.role is M.RegionRole.FLUID
    patch = next(value for value in result.patches if value.name == "wall")
    assert patch.source_adjacent_region_ids == (domain.entity_id(3, 0), None)
    np.testing.assert_array_equal(
        evidence.boundary_label.scope.entity_ids, result.mesh.entity_set(2).entity_ids
    )
    evidence.require_current(result.mesh, result.labels, result.patches)


def test_surface_cavity_budget_refuses_before_native_growth() -> None:
    domain = _square_source_domain()
    with pytest.raises(M.MeshingFailure) as refused:
        _mesh(
            domain,
            _specification(domain, 0.3, limits=M.MeshingLimits(maximum_cavity_cells=1)),
        )
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_spline_knot_corner_is_preserved_before_trim_triangulation() -> None:
    geometry = phx.geometry
    spline = geometry.BSplineCurve(
        ((1.0, 0.0), (1.3, 0.5), (1.0, 1.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.5, 1.0, 1.0),
        1,
    )
    loop = (
        geometry.PatchCurveUse(0, geometry.LineCurve((0, 0), (1, 0)), 0.0, 1.0),
        geometry.PatchCurveUse(1, spline, 0.0, 1.0),
        geometry.PatchCurveUse(2, geometry.LineCurve((1, 1), (-1, 0)), 0.0, 1.0),
        geometry.PatchCurveUse(3, geometry.LineCurve((0, 1), (0, -1)), 0.0, 1.0),
    )
    domain = geometry.MeshingDomain(
        (
            geometry.MeshingSurfacePatch(
                geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                (loop,),
            ),
        ),
        tuple(geometry.MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="spline-knot-corner",
        source_revision="r1",
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.4,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.01,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    assert np.any(np.all(np.abs(vertices - np.asarray((1.3, 0.5, 0.0))) < 1e-12, axis=1))
    assert abs(_area(vertices, faces) - 1.15) < 1e-12
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"


def _normal_extrusion_domain(surface: AbstractSurfacePatch) -> MeshingDomain:
    loop = (
        PatchCurveUse(0, LineCurve((0, 0), (1, 0)), 0.0, 2.0),
        PatchCurveUse(1, LineCurve((2, 0), (0, 1)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((0, 1), (1, 0)), 2.0, 0.0),
        PatchCurveUse(3, LineCurve((0, 0), (0, 1)), 1.0, 0.0),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="closed-source-normal-knot",
        source_revision="authored-normal-source",
    )


@pytest.mark.parametrize(
    "interval",
    ((0.9, 1.0), (1.0, 1.1), (0.9, 1.1)),
    ids=("left-knot-touch", "right-knot-touch", "knot-crossing"),
)
@pytest.mark.parametrize("placement", ("unplaced", "quarter-turn", "nonbinary-placed"))
def test_c0_source_normal_diameter_encloses_both_one_sided_world_normals(
    interval: tuple[float, float],
    placement: str,
) -> None:
    curve = phx.geometry.BSplineCurve(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.01, 0.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 2.0, 2.0),
        1,
    )
    source = phx.geometry.ExtrusionSurface(curve, (0.0, 0.0, 1.0))
    rotation = np.eye(3, dtype=np.float64)
    if placement == "quarter-turn":
        rotation = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
        )
    elif placement == "nonbinary-placed":
        angle = 0.37
        rotation = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, np.cos(angle), -np.sin(angle)),
                (0.0, np.sin(angle), np.cos(angle)),
            ),
            dtype=np.float64,
        )
    surface = (
        source
        if placement == "unplaced"
        else PlacedSurface(source, rotation, (0.1, 0.2, 0.3))
    )
    domain = _normal_extrusion_domain(surface)
    first, last = interval
    charts = np.asarray((((first, 0.0), (last, 0.0), (last, 1.0)),), dtype=np.float64)
    box = np.asarray((((first, 0.0), (last, 1.0)),), dtype=np.float64)
    second_lower, second_upper = surface.derivative_bounds_batch(box, order=2)
    assert np.all(np.isneginf(second_lower[0, :, 0, 0]))
    assert np.all(np.isposinf(second_upper[0, :, 0, 0]))
    bound = domain.normal_turn_bounds(0, charts)[0]
    # Independent geometric oracle: these are two ruled planes with the
    # authored tangents, not a discretized or second-derivative approximation.
    sweep = rotation @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    normals = np.asarray(
        [
            np.cross(rotation @ tangent, sweep)
            for tangent in np.asarray(
                ((1.0, 0.0, 0.0), (1.0, 0.01, 0.0)), dtype=np.float64
            )
        ]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    angle = np.arccos(np.clip(normals[0] @ normals[1], -1.0, 1.0))
    assert np.isfinite(bound)
    assert 0.0 < angle <= bound < 0.02


def test_zero_source_normal_remains_unresolved_without_a_gauss_cone() -> None:
    curve = phx.geometry.BSplineCurve(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 2.0, 2.0),
        1,
    )
    surface = phx.geometry.ExtrusionSurface(curve, (0.0, 0.0, 1.0))
    domain = _normal_extrusion_domain(surface)
    charts = np.asarray((((0.9, 0.0), (1.1, 0.0), (1.1, 1.0)),), dtype=np.float64)
    assert np.isposinf(domain.normal_turn_bounds(0, charts)[0])


@pytest.mark.parametrize(
    "triangle",
    (
        ((0.5, 0.0), (1.0, 0.0), (0.75, 1.0)),
        ((1.0, 0.0), (1.5, 0.0), (1.25, 1.0)),
        ((0.5, 0.0), (1.5, 0.0), (1.2, 1.0)),
        ((0.5, 0.0), (1.5, 0.0), (1.5, 0.0)),
    ),
    ids=("left-closed-stratum", "right-closed-stratum", "crossing-knot", "crossing-edge"),
)
@pytest.mark.parametrize("placed", (False, True), ids=("source", "placed-world-source"))
def test_c0_extrusion_interpolation_bounds_original_barycentric_chord(
    triangle: tuple[tuple[float, float], ...],
    placed: bool,
) -> None:
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    curve = phx.geometry.BSplineCurve(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.1, 0.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 2.0, 2.0),
        1,
    )
    source = phx.geometry.ExtrusionSurface(curve, (0.0, 0.0, 1.0))
    angle = 0.37
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        ),
        dtype=np.float64,
    )
    translation = np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
    surface = PlacedSurface(source, rotation, translation) if placed else source
    domain = _normal_extrusion_domain(surface)
    charts = np.asarray((triangle,), dtype=np.float64)
    owner = CoordinateEnclosureBudget(2_000_000, 81_920_000)
    bound = domain.interpolation_bounds(0, charts, budget=owner)[0]
    weights = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (1 / 3, 1 / 3, 1 / 3),
            (0.2, 0.7, 0.1),
        ),
        dtype=np.float64,
    )
    parameters = weights @ charts[0]
    # Independent point-set oracle: the profile is a continuous two-plane
    # hinge, and the original consuming triangle uses its original vertices.
    points = np.column_stack(
        (
            parameters[:, 0],
            0.1 * np.maximum(parameters[:, 0] - 1.0, 0.0),
            parameters[:, 1],
        )
    )
    vertices = np.column_stack(
        (charts[0, :, 0], 0.1 * np.maximum(charts[0, :, 0] - 1.0, 0.0), charts[0, :, 1])
    )
    if placed:
        points, vertices = (
            points @ rotation.T + translation,
            vertices @ rotation.T + translation,
        )
    error = np.linalg.norm(points - weights @ vertices, axis=1)
    assert np.isfinite(bound) and np.max(error) <= bound + 256 * np.finfo(np.float64).eps
    assert bound < 0.026
    assert owner.work_units > 0 and owner.peak_bytes_upper > 0
    retained = owner.retained_basis_bytes
    repeated = domain.interpolation_bounds(0, charts, budget=owner)
    np.testing.assert_array_equal(repeated, (bound,))
    assert owner.retained_basis_bytes == retained


def test_c0_extrusion_interpolation_borrows_original_zero_allowance() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    curve = phx.geometry.BSplineCurve(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.1, 0.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 2.0, 2.0),
        1,
    )
    domain = _normal_extrusion_domain(
        phx.geometry.ExtrusionSurface(curve, (0.0, 0.0, 1.0))
    )
    charts = np.asarray((((0.5, 0.0), (1.5, 0.0), (1.2, 1.0)),), dtype=np.float64)
    owner = CoordinateEnclosureBudget(0, 81_920_000)
    with owner.activate(), pytest.raises(CoordinateEnclosureResourceError) as refused:
        domain.interpolation_bounds(0, charts)
    assert owner.work_units == 0
    assert refused.value.resource == "coefficient_work" and refused.value.limit == 0


def test_nonrational_torus_plane_trims_generate_a_certified_annular_face() -> None:
    import jax.numpy as jnp

    geometry = phx.geometry
    plane = geometry.PlanePatch((-3, -3, -0.35), (6, 0, 0.9), (0, 6, 0))
    torus_patch = geometry.TorusPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        2.0,
        0.6,
    )
    intersection = geometry.intersect_surface_regions(
        geometry.SurfaceRegion(plane, np.asarray(((0.0, 0.0), (1.0, 1.0)))),
        geometry.SurfaceRegion(
            torus_patch, np.asarray(((0.0, 0.0), (2 * np.pi, 2 * np.pi)))
        ),
    )
    assert intersection.complete
    assert len(intersection.curves) == 2
    pcurves, areas = [], []
    for curve in intersection.curves:
        assert curve.closed and curve.fully_certified
        pcurve = curve.p_curve("first")
        uv = np.asarray(pcurve.evaluate(jnp.linspace(0.0, curve.num_charts, 129)))
        areas.append(float(np.sum(uv[:-1, 0] * uv[1:, 1] - uv[:-1, 1] * uv[1:, 0])))
        pcurves.append(pcurve)
    outer, inner = np.argsort(-np.abs(np.asarray(areas)))
    loops = []
    for index, orientation in ((int(outer), 1), (int(inner), -1)):
        curve = intersection.curves[index]
        first, last = 0.0, float(curve.num_charts)
        if areas[index] * orientation < 0:
            first, last = last, first
        loops.append((geometry.PatchCurveUse(index, pcurves[index], first, last),))
    domain = geometry.MeshingDomain(
        (geometry.MeshingSurfacePatch(plane, tuple(loops)),),
        (geometry.MeshingDomainCurve(0, 0), geometry.MeshingDomainCurve(1, 1)),
        2,
        source_id="nonrational-offset-torus-section",
        source_revision="r1",
    )
    result = _mesh(
        domain,
        _specification(
            domain,
            0.2,
            protected_features=(
                M.ProtectedFeature(
                    _scope(domain, 2, (0,)),
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.02,
                ),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    edges, uses = _edge_uses(faces)
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 0
    assert np.all(uses <= 2)
    np.testing.assert_allclose(vertices[:, 2], 0.15 * vertices[:, 0] + 0.1, atol=1e-12)
    boundary = np.unique(edges[uses == 1])
    boundary_points = vertices[boundary]
    # Independent implicit torus equation, not a rational fit to its section.
    radial = np.hypot(boundary_points[:, 0], boundary_points[:, 1])
    residual = (radial - 2.0) ** 2 + boundary_points[:, 2] ** 2 - 0.6**2
    assert np.max(np.abs(residual)) < 1e-9
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete
    assert all(association.complete for association in result.associations)


def test_rational_curve_density_roots_follow_selected_chart_and_live_coefficients() -> (
    None
):
    """Source restriction must preserve chart identity without freezing coefficients."""
    import jax.numpy as jnp

    from phydrax.geometry import BoundaryAtlas, BSplineCurve
    from phydrax.geometry._meshing_domain import _PhysicalCurveMap
    from phydrax.meshing.providers._native_curve import (
        _CurveCharts,
        _CurveSizing,
        _placement_roots,
    )

    controls = (
        np.asarray(((0.0, 0.0, 0.0), (0.3, 0.6, 0.0), (0.7, -0.2, 0.0), (1.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 0.0), (0.3, 0.2, 0.0), (0.7, 0.5, 0.0), (1.0, 0.0, 0.0))),
    )
    rational_weights = (
        np.asarray((1.0, 0.8, 1.2, 1.0)),
        np.asarray((1.0, 1.2, 0.7, 1.0)),
    )
    sources = tuple(
        BSplineCurve.bezier(points, weights)
        for points, weights in zip(controls, rational_weights, strict=True)
    )
    sizing = _CurveSizing(
        np.asarray((0.3,)),
        np.asarray((np.inf,)),
        np.asarray((0.0,)),
        np.asarray((np.inf,)),
        np.asarray((0.0,)),
    )
    nodes, weights = np.polynomial.legendre.leggauss(8)
    oracle_nodes, oracle_weights = np.polynomial.legendre.leggauss(128)
    fractions = np.linspace(0.1, 0.9, 32)

    def arc_integral(source_index: int, stop: np.ndarray | float) -> np.ndarray:
        parameter = 0.5 * np.asarray(stop)[..., None] * (oracle_nodes + 1.0)
        complement = 1.0 - parameter
        basis = np.stack(
            (
                complement**3,
                3.0 * parameter * complement**2,
                3.0 * parameter**2 * complement,
                parameter**3,
            ),
            axis=-1,
        )
        derivative = np.stack(
            (
                -3.0 * complement**2,
                3.0 * complement**2 - 6.0 * parameter * complement,
                6.0 * parameter * complement - 3.0 * parameter**2,
                3.0 * parameter**2,
            ),
            axis=-1,
        )
        weighted = basis * rational_weights[source_index]
        weighted_derivative = derivative * rational_weights[source_index]
        numerator = weighted @ controls[source_index]
        denominator = np.sum(weighted, axis=-1)
        velocity = (
            (weighted_derivative @ controls[source_index]) * denominator[..., None]
            - numerator * np.sum(weighted_derivative, axis=-1)[..., None]
        ) / denominator[..., None] ** 2
        return (
            0.5
            * np.asarray(stop)
            * np.sum(np.linalg.norm(velocity, axis=-1) * oracle_weights, axis=-1)
        )

    # Different chart counts/order and changed same-shape scientific arrays all
    # exercise the same compiled density program, with independent arclength.
    worksets: tuple[tuple[tuple[int, ...], int], ...] = (
        ((0, 1, 0), 0),
        ((1, 0), 1),
        ((1,), 0),
    )
    for order, selected in worksets:
        mapping = _PhysicalCurveMap(
            tuple(sources[index] for index in order),
            np.tile(((0.0, 1.0),), (len(order), 1)),
        )
        atlas = BoundaryAtlas(
            mapping,
            source_entity_ids=jnp.arange(len(order), dtype=jnp.int32),
            source_id="rational-root-charts",
            physical_tags=tuple("curve" for _ in order),
        )
        charts = _CurveCharts(atlas, jnp.arange(len(order), dtype=jnp.int32)).select(
            selected
        )
        roots, successful, _ = _placement_roots(
            charts,
            sizing,
            jnp.zeros((32,)),
            jnp.ones((32,)),
            jnp.asarray(fractions),
            jnp.asarray(nodes),
            jnp.asarray(weights),
        )
        roots = np.asarray(roots)
        assert np.all(np.asarray(successful))
        assert np.all((roots > 0.0) & (roots < 1.0))
        assert np.all(np.diff(roots) > 0.0)
        selected_source = int(np.asarray(order, dtype=np.int64)[selected])
        np.testing.assert_allclose(
            arc_integral(selected_source, roots) / arc_integral(selected_source, 1.0),
            fractions,
            rtol=0.0,
            atol=2.0e-5,
        )


def test_repeated_brep_definition_keeps_qualified_surface_and_vertex_identity() -> None:
    from hashlib import sha256

    from phydrax.geometry.brep._constructors import (
        assemble_brep_model,
        brep_box,
        BRepTessellationPolicy,
    )
    from phydrax.geometry.brep._model import BRepGeometry, BRepOccurrence

    policy = BRepTessellationPolicy(
        linear_deflection=0.05,
        angular_deflection=0.5,
        maximum_triangles=20_000,
    )
    definition = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_SI,
        tessellation=policy,
    )
    source = definition.geometry
    assert source is not None
    source_fields = (
        "vertex_points",
        "curves",
        "edge_curves",
        "edge_ranges",
        "edge_vertices",
        "pcurves",
        "coedge_edges",
        "coedge_senses",
        "face_loops",
        "shell_faces",
        "shell_orientations",
        "solid_shells",
        "vertex_roots",
        "edge_endpoint_roots",
        "coedge_endpoint_roots",
    )
    paths = (("root", "left"), ("root", "right"))
    geometry = BRepGeometry(
        **{name: getattr(source, name) for name in source_fields},
        occurrences=(
            BRepOccurrence(paths[0], 0),
            BRepOccurrence(paths[1], 0, translation=np.asarray((3.0, 0.0, 0.0))),
        ),
    )
    model = assemble_brep_model(
        geometry,
        definition.patches,
        definition.parameter_bounds,
        definition.orientation,
        definition.physical_tags,
        coordinate_contract=_SI,
        source_id="qualified-box-pair",
        source_format=definition.report.source_format,
        source_digest=geometry.geometry_id,
        import_policy_id=sha256(b"qualified-surface-identity-test").hexdigest(),
        tessellation=policy,
    )
    domain = MeshingDomain.from_brep(model)
    scope = _scope(domain, 2, tuple(domain.scope_indices(2)))
    result = _mesh(
        domain,
        M.SurfaceMeshingSpec(
            M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                M.UniformSizeControl(
                    scope,
                    0.5,
                    strength=M.SizeControlStrength.SOFT,
                ),
            ),
            protected_features=(
                M.ProtectedFeature(
                    scope,
                    M.FeatureKind.SURFACE,
                    maximum_deviation=0.05,
                ),
            ),
        ),
    )
    points, faces = _arrays(result)
    assert np.all((points[:, 0] <= 1.0) | (points[:, 0] >= 3.0))
    assert np.all(
        np.all(points[faces, 0] <= 1.0, axis=1) | np.all(points[faces, 0] >= 3.0, axis=1)
    )
    _, uses = _edge_uses(faces)
    assert np.all(uses == 2)
    associated_paths = {
        path
        for association in result.associations
        for path in association.source_occurrence_paths
    }
    assert associated_paths == set(paths)
    assert result.audit.passed
    assert result.certification is not None and result.certification.passed


def test_tilted_planar_source_uses_physical_area_normal_and_original_corners() -> None:
    root = np.sqrt(0.5)
    origin = np.asarray((0.125, -0.25, 0.375))
    first = np.asarray((root, 0.0, root))
    second = np.asarray((0.0, 1.0, 0.0))
    square = _source_square((0, 1, 2, 3), 0.0)
    domain = MeshingDomain(
        (MeshingSurfacePatch(PlanePatch(origin, first, second), square.loops),),
        tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0))),
        4,
        source_id="tilted-physical-square",
        source_revision="r1",
    )
    scope = _scope(domain, 2, (0,))
    result = _mesh(
        domain,
        _specification(
            domain,
            0.3,
            protected_features=(
                M.ProtectedFeature(scope, M.FeatureKind.SURFACE, maximum_deviation=0.01),
                M.ProtectedFeature(_scope(domain, 0, (0, 1, 2, 3)), M.FeatureKind.CORNER),
            ),
        ),
    )
    vertices, faces = _arrays(result)
    normal = np.cross(first, second)
    np.testing.assert_allclose((vertices - origin) @ normal, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(_area(vertices, faces), 1.0, rtol=0.0, atol=1.0e-12)
    corners = vertices[faces]
    assert np.all(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) @ normal
        > 0.0
    )
    for authored in (origin, origin + first, origin + first + second, origin + second):
        assert np.any(np.all(vertices == authored, axis=1))
    edges, uses = _edge_uses(faces)
    assert np.all((uses == 1) | (uses == 2))
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 1
    assert result.certification is not None
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.chart_coverage is not None
    assert result.certification.fidelity.chart_coverage.complete


def test_authored_nondyadic_curve_interval_retains_exact_boundary_provenance() -> None:
    first, last = 0.1, -0.2
    corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    uses = []
    for row, head in enumerate(corners):
        tail = corners[(row + 1) % corners.shape[0]]
        direction = (tail - head) / (last - first)
        uses.append(
            PatchCurveUse(
                row,
                LineCurve(head - first * direction, direction),
                first,
                last,
            )
        )
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), (tuple(uses),)
            ),
        ),
        tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="authored-nondyadic-boundary",
        source_revision="original-source-intervals",
    )
    result = _mesh(domain, _specification(domain, 0.4))
    vertices, faces = _arrays(result)
    assert abs(_area(vertices, faces) - 1.0) < 1e-12
    assert result.compliance.passed
    assert all(association.complete for association in result.associations)


@pytest.mark.parametrize(
    "changed_bank", ("uv", "patch", "cell-sci", "source-definition", "source-boundary")
)
def test_public_native_source_charts_authenticate_and_refuse_changed_banks(
    changed_bank: str,
) -> None:
    import equinox as eqx

    from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport

    original, _, result = _case("cylinder_sheet")
    charts = result.surface_source
    if charts is None or result.certification is None:
        raise ValueError(
            "The actual public cylinder source must retain its original chart and source certificate."
        )
    charts.require_bound(
        result.mesh, result.geometry, result.certification.request.source
    )
    support = PreparedSurfaceSourceSupport(
        original.domain,
        result,
        charts.root_parameters,
        charts.boundary_source,
        maximum_support_queries=charts.maximum_support_queries,
    )
    support.require_current()
    if changed_bank == "uv":
        values = charts.root_parameters.copy()
        values[0, 0, 0] = np.nextafter(values[0, 0, 0], np.inf)
        changed = eqx.tree_at(lambda node: node.root_parameters, charts, values)
    elif changed_bank == "patch":
        values = charts.root_patches.copy()
        values[0] += 1
        changed = eqx.tree_at(lambda node: node.root_patches, charts, values)
    elif changed_bank == "cell-sci":
        values = charts.mesh.blocks[0].global_ids.at[0].add(1)
        changed = eqx.tree_at(lambda node: node.mesh.blocks[0].global_ids, charts, values)
    elif changed_bank == "source-definition":
        changed = eqx.tree_at(
            lambda node: node.domain.patches[0].surface.radius,
            charts,
            charts.domain.patches[0].surface.radius * 1.01,
        )
    else:
        values = charts.boundary_source.chart_triangulations[0][1].copy()
        values[0, 0] = np.nextafter(values[0, 0], np.inf)
        changed = eqx.tree_at(
            lambda node: node.boundary_source.chart_triangulations[0][1],
            charts,
            values,
        )
    with pytest.raises(ValueError, match="changed"):
        changed.require_bound(
            result.mesh, result.geometry, result.certification.request.source
        )
    charts.require_bound(
        result.mesh, result.geometry, result.certification.request.source
    )


@pytest.mark.parametrize("premise", ("domain", "boundary-domain"))
def test_original_surface_support_refuses_changed_supplied_source_definition(
    premise: str,
) -> None:
    import equinox as eqx

    from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport

    original, _, result = _case("cylinder_sheet")
    charts = result.surface_source
    if charts is None or result.certification is None:
        raise ValueError(
            "Actual cylinder publication lacks its original source chart premise."
        )
    surface = original.domain.patches[0].surface
    if not isinstance(surface, phx.geometry.CylinderPatch):
        raise RuntimeError("Cylinder source fixture lost its native patch.")
    changed_domain = eqx.tree_at(
        lambda node: node.patches[0].surface.radius,
        original.domain,
        surface.radius * 1.01,
    )
    domain = changed_domain if premise == "domain" else original.domain
    boundary = (
        charts.boundary_source
        if premise == "domain"
        else eqx.tree_at(lambda node: node.domain, charts.boundary_source, changed_domain)
    )
    with pytest.raises(ValueError, match="changed authority"):
        PreparedSurfaceSourceSupport(
            domain,
            result,
            charts.root_parameters,
            boundary,
            maximum_support_queries=charts.maximum_support_queries,
        )
    charts.require_bound(
        result.mesh, result.geometry, result.certification.request.source
    )


def test_public_native_surface_default_archive_retains_original_chart_boundary(
    tmp_path: Any,
) -> None:
    from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing import MeshPart

    original, size, result = _case("cylinder_sheet")
    charts = result.surface_source
    if charts is None or result.certification is None:
        raise ValueError(
            "The actual public cylinder source must retain its original chart premise."
        )
    receipt = write_meshing_source_closure(
        tmp_path / "native-cylinder-source.zip",
        {
            "certification_inputs": result.certification.request,
            "report": result.certification,
            "associations": result.associations,
            "generation_part": MeshPart(original.domain.source_id, result),
            "generation_source": M.NativeSurfaceSource(original.domain),
            "generation_specification": _specification(original.domain, size),
            "generation_options": M.NativeMeshingOptions("parametric_surface"),
        },
    )
    reopened = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    restored = reopened["generation_part"].carrier
    if (
        not isinstance(restored, M.CellMeshingResult)
        or restored.surface_source is None
        or restored.certification is None
    ):
        raise ValueError(
            "Default source restoration lost the accepted actual surface carrier."
        )
    restored_charts = restored.surface_source
    restored_charts.require_bound(
        restored.mesh, restored.geometry, restored.certification.request.source
    )
    np.testing.assert_array_equal(restored_charts.root_parameters, charts.root_parameters)
    np.testing.assert_array_equal(restored_charts.root_patches, charts.root_patches)
    np.testing.assert_array_equal(
        restored.mesh.vertex_global_ids, result.mesh.vertex_global_ids
    )
    for before, after in zip(
        charts.boundary_source.chart_triangulations,
        restored_charts.boundary_source.chart_triangulations,
        strict=True,
    ):
        assert before[0] == after[0]
        for old, new in zip(before[1:], after[1:], strict=True):
            np.testing.assert_array_equal(old, new)
    support = PreparedSurfaceSourceSupport(
        restored_charts.domain,
        restored,
        restored_charts.root_parameters,
        restored_charts.boundary_source,
        maximum_support_queries=restored_charts.maximum_support_queries,
    )
    support.require_current()


def test_native_spline_normal_bounds_bind_original_scratch_owner_and_refuse_overflow(
    monkeypatch: Any,
) -> None:
    from phydrax.geometry.brep import _patches as source_bounds
    from phydrax.meshing import _surface_generation as generation

    corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    loop = tuple(
        PatchCurveUse(
            row,
            LineCurve(head, corners[(row + 1) % 4] - head),
            0.0,
            1.0,
        )
        for row, head in enumerate(corners)
    )
    patch = phx.geometry.BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.05]], [[1.0, 0.0, 0.1], [1.0, 1.0, 0.2]]],
        [[1.0, 0.9], [1.1, 1.0]],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="native-spline-normal-owner",
        source_revision="authored",
    )
    seen = []
    prefix = source_bounds._BernsteinRestrictionBank.prefix
    metrics_init = generation._Metrics.__init__
    metrics_calls = []
    source_works = []

    def observe_metrics(
        metrics: Any,
        domain: Any,
        state: Any,
        ledger: Any,
        compiled: Any,
        cache: Any,
        **kwargs: Any,
    ) -> None:
        owner = ledger.interpolation_budget
        before_source = 0 if owner is None else owner.work_units
        before_work = ledger.work
        before_temporary = 0 if owner is None else owner.temporary_bytes_upper
        metrics_init(metrics, domain, state, ledger, compiled, cache, **kwargs)
        owner = ledger.interpolation_budget
        source_work = owner.work_units - before_source
        # Cells retained from an earlier round reuse their source bounds.
        source_works.append(source_work)
        assert ledger.work - before_work == source_work
        assert ledger.limits.maximum_work_units - ledger.work == (
            ledger.limits.maximum_work_units - before_work - source_work
        )
        assert owner.temporary_bytes_upper == before_temporary
        metrics_calls.append((domain, state, ledger, compiled, kwargs))

    monkeypatch.setattr(generation._Metrics, "__init__", observe_metrics)

    def observe(bank: Any, piece: Any, order: int) -> Any:
        assert bank.budget is not None
        assert bank.budget.maximum_memory_bytes <= 2_000_000
        seen.append(order)
        return prefix(bank, piece, order)

    monkeypatch.setattr(source_bounds._BernsteinRestrictionBank, "prefix", observe)
    with pytest.raises(M.MeshingFailure) as refused:
        _mesh(
            domain,
            _specification(
                domain,
                0.4,
                limits=M.MeshingLimits(maximum_scratch_bytes=2_000_000),
            ),
        )
    assert refused.value.category in (
        M.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        M.MeshingFailureCategory.AUDIT_FAILED,
    )
    requested = dict(refused.value.evidence.requested)
    storage_limits = [
        value
        for name, value in requested.items()
        if name == "maximum_scratch_bytes"
        or name.endswith("resource:polynomial_storage:limit")
    ]
    assert storage_limits and max(storage_limits) <= 2_000_000
    assert seen and 1 in seen and 2 in seen
    assert metrics_calls and source_works[0] > 0
    domain, state, ledger, compiled, kwargs = metrics_calls[-1]
    owner = ledger.interpolation_budget
    before_temporary = owner.temporary_bytes_upper
    initialize = generation._Metrics._initialize

    def fail_after_source(metrics: Any, *args: Any, **kwargs: Any) -> None:
        initialize(metrics, *args, **kwargs)
        raise RuntimeError("post-source metrics failure")

    monkeypatch.setattr(generation._Metrics, "_initialize", fail_after_source)
    with pytest.raises(RuntimeError, match="post-source metrics failure"):
        metrics_init(
            object.__new__(generation._Metrics),
            domain,
            state,
            ledger,
            compiled,
            generation._MetricsCache(),
            **kwargs,
        )
    assert owner.temporary_bytes_upper == before_temporary
    assert source_bounds._BERNSTEIN_RESTRICTION_BANK.get() is None
