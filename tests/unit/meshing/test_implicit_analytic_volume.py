"""Analytic source profiles, atlas coverage and revision-bound restoration."""

from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreStatus, NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry.design._schema import ParameterId
from phydrax.geometry.implicit._analytic_profile import (
    AnalyticBoundaryCoverCapacityError,
    AnalyticImplicitProfile,
)
from phydrax.geometry.implicit._enclosure import INTERVAL_PROGRAMS
from phydrax.geometry.implicit._projection_cover import (
    compute_implicit_projection_cover,
    implicit_facet_projection_bounds,
)
from phydrax.geometry.multiregion_surface import seed_sphere
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._implicit_volume import implicit_source_budget
from phydrax.meshing._volume_generation import native_volume_execution_budget
from phydrax.units import CENTIMETER, METER
from tests._support.implicit_meshing import NATIVE, source_budget as source_budget


# Unchanged empirical allowances from the independent sphere/torus signed-
# distance archive/cache fixtures and Decimal reach reference in this subject.
ANALYTIC_ENCLOSURE_COMPARISON_ATOL = 1e-15
ANALYTIC_DISTANCE_COMPARISON_ATOL = 1e-14
REACH_LOWER_COMPARISON_FACTOR = 0.999999999999


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_nominal_profiles_establish_actual_reach_and_distance_brackets(
    family: str,
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(
            coordinate_unit=METER
        )
        points = np.asarray(
            ((0.0, 0.0, 0.0), (0.371, 0.0, 0.0), (0.571, 0.0, 0.0)), dtype=np.float64
        )
        reach, genus = 0.371, 0
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile(
            coordinate_unit=METER
        )
        points = np.asarray(
            ((0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)), dtype=np.float64
        )
        reach, genus = 0.2, 1
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    assert REACH_LOWER_COMPARISON_FACTOR * reach <= profile.reach_lower <= reach
    assert profile.genus == genus and profile.component_count == 1
    distances = profile.boundary_distance(points)
    with localcontext() as context:
        context.prec = 80
        for index, point in enumerate(points):
            x, y, z = (Decimal.from_float(float(value)) for value in point)
            radius = Decimal.from_float(reach)
            if family == "sphere":
                exact = abs((x * x + y * y + z * z).sqrt() - radius)
            else:
                major = Decimal.from_float(0.8)
                exact = abs(
                    (((x * x + y * y).sqrt() - major) ** 2 + z * z).sqrt() - radius
                )
            assert Decimal.from_float(float(distances.lower[index])) <= exact
            assert exact <= Decimal.from_float(float(distances.upper[index]))


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_analytic_atlas_cover_contains_independent_dense_source_probes(
    family: str,
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(
            coordinate_unit=METER
        )
        first = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False, dtype=np.float64)
        second = np.linspace(0.0, np.pi, 19, dtype=np.float64)
        u, v = np.meshgrid(first, second, indexing="ij")
        probes = 0.371 * np.stack(
            (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)), axis=-1
        ).reshape((-1, 3))
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile(
            coordinate_unit=METER
        )
        axis = np.linspace(0.0, 2.0 * np.pi, 37, endpoint=False, dtype=np.float64)
        u, v = np.meshgrid(axis, axis, indexing="ij")
        probes = np.stack(
            (
                (0.8 + 0.2 * np.cos(v)) * np.cos(u),
                (0.8 + 0.2 * np.cos(v)) * np.sin(u),
                0.2 * np.sin(v),
            ),
            axis=-1,
        ).reshape((-1, 3))
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    cover = profile.boundary_cover(0.05, 5000)
    assert cover.complete and cover.semantics == "certified"
    assert np.max(cover.covering_radius) <= 0.05
    for chunk in np.array_split(probes, 8):
        separation = np.linalg.norm(chunk[:, None] - cover.points[None], axis=2)
        assert np.all(np.min(separation - cover.covering_radius[None], axis=1) <= 0.0)


def test_profile_refuses_changed_radius_with_unchanged_source_metadata() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="profile-sphere"
    ).compile(coordinate_unit=METER)
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry, contract, source_id="external", source_revision="stable"
    )
    changed = geometry.with_parameters({ParameterId("profile-sphere", "radius"): 0.4})
    with pytest.raises(ValueError, match=r"AnalyticImplicitProfile\.binding"):
        profile.require_bound(changed, contract)


@pytest.mark.parametrize("change", ("frame", "unit"))
def test_profile_refuses_changed_physical_coordinate_contract(change: str) -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(geometry, contract)
    changed = (
        phx.SpatialCoordinateContract(contract.length_unit, reference_frame="scanner")
        if change == "frame"
        else phx.SpatialCoordinateContract(CENTIMETER)
    )
    with pytest.raises(ValueError, match=r"AnalyticImplicitProfile\.binding"):
        profile.require_bound(geometry, changed)


@pytest.mark.parametrize("invalidity", ("changed-angle", "horn"))
def test_profile_refuses_invalid_full_torus_state(invalidity: str) -> None:
    geometry = phx.geometry.Torus(
        (0.0, 0.0, 0.0), 0.6, 1.0, feature_id="profile-torus"
    ).compile(coordinate_unit=METER)
    changed = geometry.with_parameters(
        {ParameterId("profile-torus", "angle"): np.pi}
        if invalidity == "changed-angle"
        else {ParameterId("profile-torus", "minor_radius"): 0.8}
    )
    with pytest.raises(ValueError, match=r"AnalyticImplicitProfile\.(geometry|kernel)"):
        AnalyticImplicitProfile(changed, phx.SpatialCoordinateContract.si())


def test_analytic_cover_capacity_refusal_preserves_achieved_bound() -> None:
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
    )
    with pytest.raises(AnalyticBoundaryCoverCapacityError) as failure:
        profile.boundary_cover(1.0e-4, 8)
    assert failure.value.requested_radius == 1.0e-4
    assert failure.value.achieved_radius > failure.value.requested_radius
    assert failure.value.maximum_samples == 8


def test_nonanalytic_same_zero_set_cannot_inherit_nominal_source_facts() -> None:
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371)
    equivalent_field = (sphere | sphere).compile(coordinate_unit=METER)
    with pytest.raises(ValueError, match=r"AnalyticImplicitProfile\.kernel"):
        AnalyticImplicitProfile(equivalent_field, phx.SpatialCoordinateContract.si())


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_analytic_source_archive_rebuilds_cold_queries(
    tmp_path: Path, family: str
) -> None:
    if family == "sphere":
        geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(
            coordinate_unit=METER
        )
        points = np.asarray(
            ((0.0, 0.0, 0.0), (0.371, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
        )
        expected = np.asarray((0.371, 0.0, 0.629), dtype=np.float64)
    else:
        geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 0.6, 1.0).compile(
            coordinate_unit=METER
        )
        points = np.asarray(
            ((0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float64
        )
        expected = np.asarray((0.2, 0.0, 0.6), dtype=np.float64)
    profile = AnalyticImplicitProfile(
        geometry, phx.SpatialCoordinateContract.si(), cover_radius=0.05
    )
    path = tmp_path / "analytic-source.zip"
    receipt = write_meshing_source_closure(path, profile)
    INTERVAL_PROGRAMS.clear()
    restored = read_meshing_source_closure(path, expected_content_id=receipt.content_id)
    assert isinstance(restored, AnalyticImplicitProfile)
    restored.validate_source_integrity()
    distances = restored.boundary_distance(points)
    assert np.all(distances.lower <= expected + ANALYTIC_ENCLOSURE_COMPARISON_ATOL)
    assert np.all(distances.upper >= expected - ANALYTIC_ENCLOSURE_COMPARISON_ATOL)
    np.testing.assert_allclose(
        distances.upper, expected, rtol=0.0, atol=ANALYTIC_DISTANCE_COMPARISON_ATOL
    )
    cover = restored.boundary_cover(0.05, 5000)
    assert cover.complete and cover.semantics == "certified"
    assert np.max(cover.covering_radius) <= 0.05


def test_interval_execution_cache_binds_actual_state_not_equal_shape() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="cached-sphere"
    ).compile(coordinate_unit=METER)
    first = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    changed = geometry.with_parameters(
        {
            phx.geometry.ParameterId("cached-sphere", "radius"): 0.613,
        }
    )
    second = AnalyticImplicitProfile(changed, phx.SpatialCoordinateContract.si())
    point = np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64)
    initial = first.boundary_distance(point)
    updated = second.boundary_distance(point)
    np.testing.assert_allclose(
        initial.upper, (0.629,), rtol=0.0, atol=ANALYTIC_DISTANCE_COMPARISON_ATOL
    )
    np.testing.assert_allclose(
        updated.upper, (0.387,), rtol=0.0, atol=ANALYTIC_DISTANCE_COMPARISON_ATOL
    )


def test_restored_profile_rejects_divergent_bound_state_before_queries() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.371, feature_id="bound-sphere"
    ).compile(coordinate_unit=METER)
    profile = AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())
    radius = geometry.schema.index(phx.geometry.ParameterId("bound-sphere", "radius"))
    divergent = eqx.tree_at(
        lambda value: value.field_bounds.state.values[radius],
        profile,
        jnp.asarray(0.613, dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="inconsistent scientific facts"):
        divergent.boundary_distance(np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64))


@NATIVE
def test_analytic_cover_refuses_cumulative_queries_before_returning_samples() -> None:
    from phydrax._meshcore import MeshcoreError

    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
    )
    published: list[str] = []
    with pytest.raises(MeshcoreError) as caught:
        limits = MeshingLimits(
            maximum_work_units=1_000_000,
            maximum_geometry_queries=2,
            maximum_cavity_cells=100,
            maximum_scratch_bytes=8 * 1024**2,
            maximum_wall_seconds=120.0,
        )
        with NativeExecutionScope(limits.native_limits()) as execution:
            with execution.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_scratch_bytes,
            ) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                with budget.activate():
                    profile.boundary_samples(1, budget=budget)
                    published.append("inadmissible")
    assert not published
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert execution.evidence is not None
    assert int(execution.evidence.work_counters.source_queries) == 2


@pytest.mark.parametrize("tolerance", (0.02, 0.001))
def test_continuous_facet_bounds_do_not_substitute_axis_box_or_vertex_samples(
    tolerance: float,
    source_budget: CoordinateEnclosureBudget,
) -> None:
    """One physical facet may need proof subdivision, not smaller mesh edges."""
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
    )
    triangles = np.asarray(
        (((0.0, 0.0, 1.0), (0.25, 0.0, np.sqrt(0.9375)), (0.0, 0.25, np.sqrt(0.9375))),),
        dtype=np.float64,
    )
    original = triangles.copy()
    box = profile.field_bounds.boxes(np.min(triangles, axis=1), np.max(triangles, axis=1))
    assert max(abs(box.value_lower[0]), abs(box.value_upper[0])) > 0.02
    before = source_budget.ticket.remaining_geometry_queries
    deviation, oriented, queries = implicit_facet_projection_bounds(
        triangles, profile, tolerance, MeshCertificateLimits(), budget=source_budget
    )
    assert queries > 1
    assert before - source_budget.ticket.remaining_geometry_queries == 2 * queries
    np.testing.assert_array_equal(triangles, original)
    barycentric = np.asarray(
        [(i / 64, j / 64, 1 - (i + j) / 64) for i in range(65) for j in range(65 - i)],
        dtype=np.float64,
    )
    independent = np.max(np.abs(np.linalg.norm(barycentric @ triangles[0], axis=1) - 1.0))
    assert independent <= deviation[0]
    assert bool(oriented[0])
    assert bool(deviation[0] <= tolerance) is (tolerance == 0.02)


def test_continuous_facet_bounds_preserve_orientation_and_query_refusal() -> None:
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
    )
    triangle = np.asarray(
        (((0.0, 0.0, 1.0), (0.3, 0.0, np.sqrt(0.91)), (0.0, 0.3, np.sqrt(0.91))),),
        dtype=np.float64,
    )
    _, oriented, _ = implicit_facet_projection_bounds(
        triangle[:, ::-1], profile, 0.02, MeshCertificateLimits()
    )
    assert not bool(oriented[0])
    deviation, _, queries = implicit_facet_projection_bounds(
        triangle,
        profile,
        0.02,
        MeshCertificateLimits(maximum_source_samples=1),
    )
    assert np.isposinf(deviation[0]) and queries == 0


@NATIVE
def test_final_projection_consumes_the_original_query_ticket_and_refuses_its_last_query() -> (
    None
):
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile(coordinate_unit=METER),
        contract,
    )
    seed = seed_sphere(1.0, subdivisions=0)
    mesh = phx.geometry.TriangleMesh(seed.positions, seed.faces).as_cell_mesh()
    target = phx.meshing.certify_cell_mesh(mesh, contract)

    def cover(maximum_queries: int) -> int:
        limits = MeshingLimits(
            maximum_work_units=1_000_000,
            maximum_geometry_queries=maximum_queries,
        )
        with native_volume_execution_budget(limits) as execution:
            with implicit_source_budget(
                limits, scope=execution, queries_upper=maximum_queries
            ) as budget:
                before = budget.ticket.remaining_geometry_queries
                with budget.ticket.native_target():
                    proof = compute_implicit_projection_cover(
                        mesh,
                        target.geometry,
                        profile,
                        tolerance=0.24,
                        limits=MeshCertificateLimits(),
                        budget=budget,
                    )
                assert proof.status == "certified"
                return before - budget.ticket.remaining_geometry_queries

    consumed = cover(100_000)
    assert consumed > 4
    with pytest.raises(phx.meshing.MeshingFailure) as refusal:
        cover(consumed - 1)
    assert refusal.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
