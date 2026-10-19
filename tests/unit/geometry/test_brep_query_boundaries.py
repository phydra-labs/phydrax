#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer contracts across NumPy preparation and compiled B-Rep queries."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations
from tests._support.cad_models import native_rational_spline_face


@pytest.fixture(scope="module")
def sphere_query() -> phx.geometry.PreparedBRepQuery:
    model = phx.geometry.brep_sphere(
        1.0,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    return phx.geometry.prepare_brep_query(model)


def test_host_and_compiled_sphere_closest_points_match_radial_reference(
    sphere_query: phx.geometry.PreparedBRepQuery,
) -> None:
    points = np.asarray(((2.0, 0.5, 0.25), (-0.5, 2.0, 0.75)), dtype=np.float64)
    radii = np.linalg.norm(points, axis=1)
    expected = points / radii[:, None]
    host = sphere_query.closest_point(points)
    device = eqx.filter_jit(sphere_query.closest_point)(jnp.asarray(points))
    # The projection policy owns the requested geometric error, rather than an
    # independently chosen test tolerance.
    tolerance = sphere_query.projection_policy.ambiguity_tolerance
    for result in (host, device):
        np.testing.assert_allclose(result.points, expected, atol=tolerance, rtol=0.0)
        np.testing.assert_allclose(
            result.distances, radii - 1.0, atol=tolerance, rtol=0.0
        )
        assert not np.any(
            np.asarray(result.status) == phx.geometry.BRepProjectionStatus.FAILED
        )


def test_host_sphere_membership_retains_directed_boundary_distance(
    sphere_query: phx.geometry.PreparedBRepQuery,
) -> None:
    points = np.asarray(((0.25, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float64)
    result = sphere_query.contains(points)
    np.testing.assert_array_equal(result.inside, (True, False))
    np.testing.assert_array_equal(result.unresolved, (False, False))
    distances = np.asarray((0.75, 1.0), dtype=np.float64)
    assert np.all(np.asarray(result.distance_lower_bounds) <= distances)
    assert np.all(np.asarray(result.distance_upper_bounds) >= distances)
    assert not np.any(np.asarray(result.resource_exhausted))


def test_query_policy_refuses_boolean_budget_metadata() -> None:
    with pytest.raises(TypeError, match="seed_resolution"):
        phx.geometry.BRepQueryPolicy(seed_resolution=True)


def test_measure_error_estimates_are_finite_nonnegative(
    sphere_query: phx.geometry.PreparedBRepQuery,
) -> None:
    measures = sphere_query.measures
    assert np.all(np.isfinite(np.asarray(measures.face_area_error_estimates)))
    assert np.all(np.asarray(measures.face_area_error_estimates) >= 0.0)
    assert np.all(np.isfinite(np.asarray(measures.solid_volume_error_estimates)))
    assert np.all(np.asarray(measures.solid_volume_error_estimates) >= 0.0)


def test_trimmed_spline_query_refuses_unadmitted_caller_preparation_work() -> None:
    model = native_rational_spline_face()
    policy = phx.geometry.BRepQueryPolicy()
    projection_policy = phx.geometry.BRepProjectionPolicy()
    inputs = (model, policy, projection_policy)
    preparations = TrimRequestPreparations()
    with coordinate_source_request(inputs) as budget:
        # The rational sheet's restricted coefficient banks require work.
        # The caller admits none of that preparation in this stage.
        with budget.bound_stage(0, budget.maximum_memory_bytes):
            with pytest.raises(CoordinateEnclosureResourceError) as refusal:
                phx.geometry.prepare_brep_query(
                    model,
                    policy=policy,
                    projection_policy=projection_policy,
                    budget=budget,
                    preparations=preparations,
                )
            assert refusal.value.resource == "coefficient_work"
            assert refusal.value.limit == 0
            assert refusal.value.requested > refusal.value.limit
            assert refusal.value.completed == budget.work_units == 0
