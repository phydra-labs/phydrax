#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

import numpy as np
import pytest

import phydrax as phx


# Analytic references: the unit sphere and the torus with major radius 1 and
# minor radius 0.4, sampled independently of the reconstruction under test.
_MAJOR = 1.0
_MINOR = 0.4


def _sphere_samples(rng: np.random.Generator, count: int) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _nonuniform_noisy_sphere(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _sphere_samples(rng, 2500)
    cap = _sphere_samples(rng, 2000)
    points = np.concatenate((base, cap[cap[:, 2] > 0.3]), axis=0)
    return points * (1.0 + 0.005 * rng.normal(size=(points.shape[0], 1)))


def _torus_samples(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    toroidal = rng.uniform(0.0, 2.0 * np.pi, count)
    poloidal = rng.uniform(0.0, 2.0 * np.pi, count)
    ring = _MAJOR + _MINOR * np.cos(poloidal)
    points = np.stack(
        (ring * np.cos(toroidal), ring * np.sin(toroidal), _MINOR * np.sin(poloidal)),
        axis=1,
    )
    normals = np.stack(
        (
            np.cos(poloidal) * np.cos(toroidal),
            np.cos(poloidal) * np.sin(toroidal),
            np.sin(poloidal),
        ),
        axis=1,
    )
    return points, normals


def _torus_distance(points: np.ndarray) -> np.ndarray:
    radial = np.hypot(points[:, 0], points[:, 1]) - _MAJOR
    return np.abs(np.hypot(radial, points[:, 2]) - _MINOR)


def _world_vertices(source: phx.geometry.ReconstructedGeometrySource) -> np.ndarray:
    mesh = source.source
    assert isinstance(mesh, phx.geometry.MeshRegion)
    offset = np.asarray(source.report.recenter_offset, dtype=np.float64)
    return np.asarray(mesh.vertices, dtype=np.float64) + offset


def test_noisy_nonuniform_sphere_reconstructs_closed_genus_zero_surface() -> None:
    points = _nonuniform_noisy_sphere(0)

    source = phx.geometry.reconstruct_surface_region(points)
    report = source.report

    assert report.algorithm == "native_screened_poisson"
    assert report.watertight and report.winding_consistent
    assert report.connected_components == 1
    assert report.euler_characteristic == 2
    poisson = report.poisson_evidence
    assert poisson is not None and poisson.solver_converged
    assert poisson.solver_message == "success"
    spacing = poisson.grid_spacing
    vertices = _world_vertices(source)
    assert np.max(np.abs(np.linalg.norm(vertices, axis=1) - 1.0)) <= 0.5 * spacing
    reference = _sphere_samples(np.random.default_rng(1), 500)
    nearest = np.min(
        np.linalg.norm(reference[:, None, :] - vertices[None, :, :], axis=-1), axis=1
    )
    assert np.max(nearest) <= spacing
    deviation = report.deviation_evidence
    assert deviation is not None
    assert deviation.sample_to_surface_max <= 0.5 * spacing
    # Independent brute-force vertex-to-sample distances; the maximum is bounded
    # by the covering radius of the random samples, not by the cell width alone.
    nearest_sample = np.concatenate(
        [
            np.min(
                np.linalg.norm(block[:, None, :] - points[None, :, :], axis=-1), axis=1
            )
            for block in np.array_split(vertices, 16)
        ]
    )
    assert deviation.surface_to_sample_max == pytest.approx(
        float(np.max(nearest_sample)), rel=1e-9
    )
    assert deviation.surface_to_sample_max <= 1.5 * spacing
    domain = phx.domain.GeometryDomain(source.compile())
    assert float(domain.volume) == pytest.approx(4.0 * np.pi / 3.0, rel=0.03)
    assert report.outlier_evidence is None
    assert report.thin_feature_evidence is not None
    assert report.thin_feature_evidence.thin_samples == 0
    assert report.coverage_evidence is not None
    assert report.coverage_evidence.unsupported_triangles == 0
    assert report.component_evidence is not None
    assert report.component_evidence.unresolved_components == ()


def test_inconsistent_supplied_normals_are_oriented_before_poisson() -> None:
    points, outward = _torus_samples(2, 5000)
    flipped = np.random.default_rng(3).uniform(size=points.shape[0]) < 0.4
    supplied = np.where(flipped[:, None], -outward, outward)

    oriented = phx.geometry.estimate_point_normals(points, normals=supplied)

    assert np.all(np.sum(oriented.normals * outward, axis=1) > 0.99)
    evidence = oriented.evidence
    assert evidence.reoriented_count == np.count_nonzero(flipped)
    assert evidence.conflicting_neighbor_edges == 0
    assert evidence.graph_components == 1

    source = phx.geometry.reconstruct_surface_region(points, normals=supplied)
    report = source.report
    assert report.watertight and report.connected_components == 1
    assert report.euler_characteristic == 0
    assert report.normal_evidence == evidence
    assert report.poisson_evidence is not None
    spacing = report.poisson_evidence.grid_spacing
    assert np.max(_torus_distance(_world_vertices(source))) <= 0.5 * spacing


def test_supplied_orientation_keeps_inconsistent_normals_and_reports_conflicts() -> None:
    points, outward = _torus_samples(4, 3000)
    flipped = np.random.default_rng(5).uniform(size=points.shape[0]) < 0.3
    supplied = np.where(flipped[:, None], -outward, outward)

    kept = phx.geometry.estimate_point_normals(
        points, normals=supplied, orientation="supplied"
    )

    np.testing.assert_allclose(kept.normals, supplied, atol=1e-12)
    assert kept.evidence.reoriented_count == 0
    assert kept.evidence.conflicting_neighbor_edges > 0


def test_estimated_normals_reconstruct_torus_with_genus_one() -> None:
    points, outward = _torus_samples(6, 5000)
    points = points + 0.004 * np.random.default_rng(7).normal(size=points.shape)

    estimated = phx.geometry.estimate_point_normals(points)
    source = phx.geometry.reconstruct_surface_region(points)

    assert np.mean(np.sum(estimated.normals * outward, axis=1) > 0.9) > 0.99
    assert estimated.evidence.route == "bvh-knn-pca-mst-propagation"
    report = source.report
    assert report.connected_components == 1
    assert report.euler_characteristic == 0
    assert report.poisson_evidence is not None
    spacing = report.poisson_evidence.grid_spacing
    assert np.max(_torus_distance(_world_vertices(source))) <= 0.5 * spacing


def test_disjoint_sample_components_reconstruct_separate_closed_surfaces() -> None:
    rng = np.random.default_rng(8)
    first = _sphere_samples(rng, 1500)
    second = 0.6 * _sphere_samples(rng, 1200) + np.asarray((3.0, 0.0, 0.0))
    points = np.concatenate((first, second), axis=0)

    source = phx.geometry.reconstruct_surface_region(points)

    report = source.report
    assert report.normal_evidence is not None
    assert report.normal_evidence.graph_components == 2
    assert report.connected_components == 2
    assert report.euler_characteristic == 4
    domain = phx.domain.GeometryDomain(source.compile())
    expected = 4.0 * np.pi / 3.0 * (1.0 + 0.6**3)
    assert float(domain.volume) == pytest.approx(expected, rel=0.04)


@pytest.mark.parametrize("discretization", ["octree", "regular"])
def test_poisson_unknown_budget_is_refused_before_assembly(
    discretization: phx.geometry.PoissonDiscretization,
) -> None:
    points = _nonuniform_noisy_sphere(9)

    with pytest.raises(ValueError, match="maximum_grid_nodes"):
        phx.geometry.reconstruct_surface_region(
            points,
            sample_spacing=0.01,
            maximum_grid_nodes=1 << 16,
            discretization=discretization,
        )


def _torus_case() -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], int]:
    points, _ = _torus_samples(12, 12000)
    return points, _torus_distance, 0


def _sphere_case() -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], int]:
    points = _sphere_samples(np.random.default_rng(13), 8000)
    return points, lambda vertices: np.abs(np.linalg.norm(vertices, axis=1) - 1.0), 2


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_sphere_case, id="sphere"),
        pytest.param(_torus_case, id="torus"),
    ],
)
def test_octree_matches_regular_grid_accuracy_with_fewer_unknowns(
    case: Callable[[], tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], int]],
) -> None:
    points, distance, euler = case()

    reports = {}
    errors = {}
    for discretization in ("regular", "octree"):
        source = phx.geometry.reconstruct_surface_region(
            points, normals=points if euler == 2 else None, discretization=discretization
        )
        reports[discretization] = source.report
        errors[discretization] = float(np.max(distance(_world_vertices(source))))

    octree = reports["octree"].poisson_evidence
    regular = reports["regular"].poisson_evidence
    assert octree is not None and regular is not None
    assert octree.discretization == "octree" and regular.discretization == "regular"
    assert octree.grid_spacing == pytest.approx(regular.grid_spacing)
    assert octree.hanging_nodes > 0
    assert 3 * octree.unknowns < regular.unknowns
    assert reports["octree"].watertight
    assert reports["octree"].euler_characteristic == euler
    assert reports["octree"].connected_components == 1
    assert errors["octree"] <= 0.5 * octree.grid_spacing
    assert errors["octree"] <= 1.25 * errors["regular"]


def test_octree_level_set_is_closed_across_refinement_levels() -> None:
    points = _sphere_samples(np.random.default_rng(14), 1500)

    source = phx.geometry.reconstruct_surface_region(
        points, normals=points, sample_spacing=0.03
    )

    report = source.report
    poisson = report.poisson_evidence
    assert poisson is not None and poisson.hanging_nodes > 0
    assert report.watertight and report.connected_components == 1
    assert report.euler_characteristic == 2
    domain = phx.domain.GeometryDomain(source.compile())
    assert float(domain.volume) == pytest.approx(4.0 * np.pi / 3.0, rel=0.03)


def test_statistical_outlier_removal_drops_exactly_the_injected_outliers() -> None:
    rng = np.random.default_rng(20)
    surface = _sphere_samples(rng, 3000)
    candidates = rng.uniform(-1.8, 1.8, size=(4000, 3))
    far = np.abs(np.linalg.norm(candidates, axis=1) - 1.0) > 0.4
    points = np.concatenate((surface, candidates[far][:60]), axis=0)

    source = phx.geometry.reconstruct_surface_region(
        points,
        robustness=phx.geometry.ReconstructionRobustness(outlier_std_ratio=2.0),
    )

    report = source.report
    outliers = report.outlier_evidence
    assert outliers is not None
    assert outliers.removed_indices == tuple(range(3000, 3060))
    assert outliers.threshold == pytest.approx(
        outliers.mean_distance + 2.0 * outliers.std_distance
    )
    assert report.input_points == 3060 and report.retained_points == 3000
    assert report.watertight and report.euler_characteristic == 2
    domain = phx.domain.GeometryDomain(source.compile())
    assert float(domain.volume) == pytest.approx(4.0 * np.pi / 3.0, rel=0.03)


def _capless_sphere() -> tuple[np.ndarray, float]:
    """Unit-sphere samples below ``z = 0.7`` and the analytic unsupported polar angle.

    With the default coverage distance ``d = 2 h`` the extracted closure is
    unsupported beyond chord ``d`` from the rim at polar angle ``acos(0.7)``.
    """

    points = _sphere_samples(np.random.default_rng(21), 3000)
    return points[points[:, 2] < 0.7], float(np.arccos(0.7))


def test_incompletely_sampled_region_is_reported_on_the_closed_surface() -> None:
    points, rim = _capless_sphere()

    source = phx.geometry.reconstruct_surface_region(points)

    report = source.report
    coverage = report.coverage_evidence
    poisson = report.poisson_evidence
    assert coverage is not None and poisson is not None
    assert report.watertight and report.euler_characteristic == 2
    assert coverage.coverage_distance == pytest.approx(2.0 * poisson.grid_spacing)
    assert coverage.unsupported_patches == 1
    angle = rim - 2.0 * np.arcsin(0.5 * coverage.coverage_distance)
    expected = 2.0 * np.pi * (1.0 - np.cos(angle))
    assert coverage.unsupported_area == pytest.approx(expected, rel=0.2)
    assert any("farther than" in warning for warning in report.warnings)


def test_incompletely_sampled_region_is_refused_under_refuse_policy() -> None:
    points, _ = _capless_sphere()

    with pytest.raises(phx.geometry.ReconstructionFailure, match="Incomplete") as error:
        phx.geometry.reconstruct_surface_region(
            points,
            robustness=phx.geometry.ReconstructionRobustness(
                incomplete_sampling="refuse"
            ),
        )

    coverage = error.value.report.coverage_evidence
    assert coverage is not None and coverage.unsupported_patches == 1


def test_density_trimming_opens_the_unsampled_cap() -> None:
    points, rim = _capless_sphere()

    trimmed = phx.geometry.reconstruct_trimmed_surface(points)

    report = trimmed.report
    coverage = report.coverage_evidence
    assert coverage is not None
    assert not report.watertight
    assert report.connected_components == 1 and report.euler_characteristic == 1
    vertices = np.asarray(trimmed.surface.mesh.vertices) + np.asarray(
        report.recenter_offset
    )
    angle = rim - 2.0 * np.arcsin(0.5 * coverage.coverage_distance)
    assert np.max(vertices[:, 2]) <= np.cos(angle) + 0.5 * coverage.coverage_distance
    assert np.max(np.abs(np.linalg.norm(vertices, axis=1) - 1.0)) <= 0.05
    expected = 4.0 * np.pi - 2.0 * np.pi * (1.0 - np.cos(angle))
    assert float(trimmed.surface.measure) == pytest.approx(expected, rel=0.03)


def test_unresolvable_component_is_refused_with_fit_evidence() -> None:
    rng = np.random.default_rng(22)
    resolved = _sphere_samples(rng, 2500)
    unresolved = 0.02 * _sphere_samples(rng, 200) + np.asarray((3.0, 0.0, 0.0))
    points = np.concatenate((resolved, unresolved), axis=0)

    with pytest.raises(phx.geometry.ReconstructionFailure, match="not fit") as error:
        phx.geometry.reconstruct_surface_region(points)

    fit = error.value.report.component_evidence
    assert fit is not None
    assert fit.sample_counts == (2500, 200)
    assert fit.unresolved_components == (1,)
    assert fit.spreads[0] <= fit.tolerance < fit.spreads[1]
    assert fit.mean_offsets[1] > 0.0


def test_thin_plate_is_reported_with_its_sheet_separation() -> None:
    thickness = 0.03
    rng = np.random.default_rng(24)
    plan = rng.uniform(-0.5, 0.5, size=(6000, 2))
    side = np.where(np.arange(6000) < 3000, 1.0, -1.0)
    points = np.column_stack((plan, 0.5 * thickness * side))
    normals = np.column_stack((np.zeros((6000, 2)), side))

    source = phx.geometry.reconstruct_surface_region(
        points, normals=normals, normal_orientation="supplied"
    )

    report = source.report
    thin = report.thin_feature_evidence
    poisson = report.poisson_evidence
    assert thin is not None and poisson is not None
    assert thin.thin_feature_distance == pytest.approx(2.0 * poisson.grid_spacing)
    assert thickness < thin.thin_feature_distance
    assert thin.minimum_separation == pytest.approx(thickness, rel=1e-9)
    assert thin.thin_samples > 5400
    assert any("stacked sheet" in warning for warning in report.warnings)


def _invalid_outlier_ratio() -> object:
    return phx.geometry.ReconstructionRobustness(outlier_std_ratio=-1.0)


def _invalid_sampling_policy() -> object:
    return phx.geometry.ReconstructionRobustness(
        incomplete_sampling="drop",  # ty: ignore[invalid-argument-type]
    )


def _invalid_discretization() -> object:
    return phx.geometry.reconstruct_surface_region(
        _sphere_samples(np.random.default_rng(23), 64),
        discretization="adaptive",  # ty: ignore[invalid-argument-type]
    )


@pytest.mark.parametrize(
    ("request_reconstruction", "message"),
    [
        pytest.param(_invalid_outlier_ratio, "outlier_std_ratio", id="outlier-ratio"),
        pytest.param(_invalid_sampling_policy, "incomplete_sampling", id="policy"),
        pytest.param(_invalid_discretization, "discretization", id="discretization"),
    ],
)
def test_reconstruction_refuses_invalid_robustness_requests(
    request_reconstruction: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        request_reconstruction()


def _without_normals_supplied(points: np.ndarray) -> object:
    return phx.geometry.estimate_point_normals(points, orientation="supplied")


def _unknown_orientation(points: np.ndarray) -> object:
    return phx.geometry.estimate_point_normals(
        points,
        orientation="nearest",  # ty: ignore[invalid-argument-type]
    )


def _empty_neighborhood(points: np.ndarray) -> object:
    return phx.geometry.estimate_point_normals(points, neighborhood_size=0)


def _zero_supplied_normals(points: np.ndarray) -> object:
    return phx.geometry.estimate_point_normals(points, normals=np.zeros_like(points))


@pytest.mark.parametrize(
    ("request_normals", "message"),
    [
        pytest.param(_without_normals_supplied, "requires normals", id="no-normals"),
        pytest.param(_unknown_orientation, "orientation", id="bad-selector"),
        pytest.param(_empty_neighborhood, "positive", id="empty-neighborhood"),
        pytest.param(_zero_supplied_normals, "nonzero", id="zero-supplied-normals"),
    ],
)
def test_normal_estimation_refuses_invalid_requests(
    request_normals: Callable[[np.ndarray], object], message: str
) -> None:
    points = _sphere_samples(np.random.default_rng(10), 64)

    with pytest.raises(ValueError, match=message):
        request_normals(points)


def test_planar_reconstruction_covers_alpha_filtered_concave_domain() -> None:
    lattice = np.stack(
        np.meshgrid(np.linspace(0.0, 2.0, 21), np.linspace(0.0, 2.0, 21)), axis=-1
    ).reshape((-1, 2))
    keep = (lattice[:, 0] <= 1.0 + 1e-12) | (lattice[:, 1] <= 1.0 + 1e-12)
    points = lattice[keep] + 0.01 * np.random.default_rng(11).uniform(
        -1.0, 1.0, size=(np.count_nonzero(keep), 2)
    )

    source = phx.geometry.reconstruct_planar_region(points, alpha=0.2)

    report = source.report
    assert report.algorithm == "native_delaunay_2d_native_boundary"
    assert report.parameters == (("alpha", "0.2"), ("tolerance", "1e-05"))
    domain = phx.domain.GeometryDomain(source.compile())
    assert float(domain.volume) == pytest.approx(3.0, abs=0.08)


def test_dem_reconstruction_extrudes_terrain_into_exact_prism_volume() -> None:
    x = np.linspace(-2.0, 2.0, 25)
    y = np.linspace(-1.5, 1.5, 19)
    heights = np.sin(x)[None, :] * np.cos(y)[:, None]

    source = phx.geometry.reconstruct_dem_region(heights, x=x, y=y, extrude_depth=0.25)

    report = source.report
    assert report.watertight
    assert report.connected_components == 1
    assert report.euler_characteristic == 2
    assert report.deviation_evidence is None
    domain = phx.domain.GeometryDomain(source.compile())
    assert float(domain.volume) == pytest.approx(0.25 * 4.0 * 3.0, rel=1e-10)


def test_native_reconstruction_loads_no_external_reconstruction_engine() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        import numpy as np

        import phydrax as phx

        # Module import time is outside this contract (lazy facades defer it);
        # resolve the owner first so only reconstruction calls are measured.
        phx.geometry.reconstruct_surface_region

        BLOCKED = ("pyvista", "vtk", "vtkmodules", "scipy.spatial")

        def blocked(name):
            return any(name == root or name.startswith(root + ".") for root in BLOCKED)

        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if blocked(name):
                    raise ImportError(f"native reconstruction imported {name}")
                return None

        for name in [name for name in sys.modules if blocked(name)]:
            del sys.modules[name]
        sys.meta_path.insert(0, Refuse())

        rng = np.random.default_rng(0)
        sphere = rng.normal(size=(800, 3))
        sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
        surface = phx.geometry.reconstruct_surface_region(sphere)
        planar = phx.geometry.reconstruct_planar_region(rng.uniform(size=(200, 2)))
        grid = np.sin(np.linspace(0.0, 3.0, 12))[None, :] * np.ones((10, 1))
        terrain = phx.geometry.reconstruct_dem_region(grid)
        assert surface.report.euler_characteristic == 2
        assert not [name for name in sys.modules if blocked(name)]
        print("isolated")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("isolated")
