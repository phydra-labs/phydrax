#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry.analytic import RigidFrame
from phydrax.optics.geometric._paraxial import (
    DifferentialRayMap,
    linearize_sequential_optics,
    ParaxialOpticsPlan,
    ParaxialOpticsStatus,
)
from phydrax.optics.geometric._sequential import SequentialOpticsPlan


def _frame(z: float, rotation=None) -> RigidFrame:
    return RigidFrame(
        np.eye(3) if rotation is None else np.asarray(rotation),
        np.asarray((0.0, 0.0, z)),
    )


def _single_plan(
    *,
    kind: str = "plane",
    interaction: str = "transmit",
    surface_z: float = 1.0,
    curvature: float = 0.0,
    aperture: float = 2.0,
    indices: tuple[float, float] = (1.0, 1.0),
) -> SequentialOpticsPlan:
    return SequentialOpticsPlan(
        (_frame(surface_z),),
        (kind,),
        (interaction,),
        np.asarray((curvature,)),
        np.zeros((1,)),
        np.zeros((1, 0)),
        np.zeros((1, 0), dtype=bool),
        np.asarray((aperture,)),
        np.asarray((True,)),
        np.asarray((20.0,)),
        np.asarray(indices),
    )


def _explicit_map(
    exact,
    reference=(0.0, 0.0, 0.0, 0.0),
    *,
    input_frame=None,
    output_frame=None,
    input_index=1.0,
    output_index=1.0,
) -> DifferentialRayMap:
    input_frame = _frame(0.0) if input_frame is None else input_frame
    output_frame = _frame(1.0) if output_frame is None else output_frame
    return linearize_sequential_optics(
        exact,
        jnp.asarray(reference),
        input_frame=input_frame,
        output_frame=output_frame,
        input_refractive_index=input_index,
        output_refractive_index=output_index,
    )


def _independent_exact_coordinates(
    exact,
    coordinates,
    *,
    input_frame,
    output_frame,
    input_index,
    output_index,
):
    coordinates = jnp.asarray(coordinates)
    local_origin = jnp.asarray((coordinates[0], coordinates[1], 0.0))
    theta = coordinates[2:] / input_index
    local_direction = jnp.asarray((jnp.tan(theta[0]), jnp.tan(theta[1]), 1.0))
    local_direction = local_direction / jnp.linalg.norm(local_direction)
    origin = input_frame.apply(local_origin)
    direction = local_direction @ input_frame.rotation.T
    result = exact.execute(origin, direction)
    output_origin = output_frame.inverse_apply(result.rays.origins)
    output_direction = result.rays.directions @ output_frame.rotation
    distance = -output_origin[2] / output_direction[2]
    point = output_origin + distance * output_direction
    return jnp.asarray(
        (
            point[0],
            point[1],
            output_index * jnp.atan2(output_direction[0], output_direction[2]),
            output_index * jnp.atan2(output_direction[1], output_direction[2]),
        )
    )


def test_canonical_coordinate_translation_has_analytic_matrix():
    distance = 2.5
    refractive_index = 1.4
    exact = _single_plan(
        surface_z=distance, indices=(refractive_index, refractive_index)
    ).prepare()
    ray_map = _explicit_map(
        exact,
        input_frame=_frame(0.0),
        output_frame=_frame(distance),
        input_index=refractive_index,
        output_index=refractive_index,
    )
    expected = np.eye(4)
    expected[0, 2] = distance / refractive_index
    expected[1, 3] = distance / refractive_index

    assert ray_map.coordinate_convention == "(u,v,nθu,nθv)"
    assert bool(ray_map.valid)
    np.testing.assert_allclose(ray_map.input_reference, np.zeros((4,)), atol=1.0e-12)
    np.testing.assert_allclose(ray_map.output_reference, np.zeros((4,)), atol=1.0e-12)
    np.testing.assert_allclose(ray_map.jacobian, expected, rtol=2.0e-9, atol=2.0e-9)


def test_spherical_refraction_matches_first_order_power_limit():
    distance = 1.0
    curvature = 0.2
    incident_index = 1.0
    transmitted_index = 1.5
    exact = _single_plan(
        kind="sphere",
        surface_z=distance,
        curvature=curvature,
        indices=(incident_index, transmitted_index),
    ).prepare()
    ray_map = _explicit_map(
        exact,
        input_frame=_frame(0.0),
        output_frame=_frame(distance),
        input_index=incident_index,
        output_index=transmitted_index,
    )
    power = (incident_index - transmitted_index) * curvature
    expected = np.asarray(
        (
            (1.0, 0.0, distance / incident_index, 0.0),
            (0.0, 1.0, 0.0, distance / incident_index),
            (power, 0.0, 1.0 + power * distance / incident_index, 0.0),
            (0.0, power, 0.0, 1.0 + power * distance / incident_index),
        )
    )

    assert bool(ray_map.valid)
    np.testing.assert_allclose(ray_map.jacobian, expected, rtol=2.0e-8, atol=2.0e-8)


def test_explicit_and_cached_maps_are_identical_and_affine():
    exact_plan = _single_plan(surface_z=2.0)
    exact = exact_plan.prepare()
    explicit = _explicit_map(exact, output_frame=_frame(2.0))
    plan = ParaxialOpticsPlan(
        exact_plan,
        _frame(0.0),
        _frame(2.0),
        jnp.zeros((4,)),
        input_refractive_index=1.0,
        output_refractive_index=1.0,
        maximum_transverse_perturbation=0.2,
        maximum_angular_perturbation=0.1,
    )
    prepared = plan.prepare(exact)

    np.testing.assert_allclose(prepared.differential_map.jacobian, explicit.jacobian)
    np.testing.assert_allclose(
        prepared.differential_map.output_reference, explicit.output_reference
    )
    coordinates = jnp.asarray((0.1, -0.05, 0.02, -0.01))
    result = prepared.execute(coordinates)
    expected = explicit.output_reference + explicit.jacobian @ coordinates
    assert bool(result.successful)
    np.testing.assert_allclose(result.coordinates, expected, atol=1.0e-12)


def test_centered_finite_difference_matches_fixed_branch_jacobian():
    input_frame = _frame(0.0)
    output_frame = _frame(1.0)
    exact = _single_plan(
        kind="sphere", curvature=0.15, surface_z=1.0, indices=(1.0, 1.6)
    ).prepare()
    reference = jnp.asarray((0.03, -0.02, 0.01, -0.015))
    ray_map = _explicit_map(
        exact,
        reference,
        input_frame=input_frame,
        output_frame=output_frame,
        input_index=1.0,
        output_index=1.6,
    )
    step = 1.0e-5
    columns = []
    for index in range(4):
        perturbation = np.zeros((4,))
        perturbation[index] = step
        plus = _independent_exact_coordinates(
            exact,
            reference + perturbation,
            input_frame=input_frame,
            output_frame=output_frame,
            input_index=1.0,
            output_index=1.6,
        )
        minus = _independent_exact_coordinates(
            exact,
            reference - perturbation,
            input_frame=input_frame,
            output_frame=output_frame,
            input_index=1.0,
            output_index=1.6,
        )
        columns.append((plus - minus) / (2.0 * step))
    finite_difference = jnp.stack(columns, axis=1)
    exact_reference = _independent_exact_coordinates(
        exact,
        reference,
        input_frame=input_frame,
        output_frame=output_frame,
        input_index=1.0,
        output_index=1.6,
    )

    assert bool(ray_map.valid)
    np.testing.assert_allclose(ray_map.output_reference, exact_reference, atol=1.0e-12)
    np.testing.assert_allclose(
        ray_map.jacobian, finite_difference, rtol=2.0e-6, atol=2.0e-7
    )


def test_reflection_uses_declared_output_frame_and_canonical_signs():
    distance = 1.5
    input_index = 1.2
    mirror = _single_plan(
        interaction="reflect",
        surface_z=distance,
        aperture=4.0,
        indices=(input_index, input_index),
    )
    exact = mirror.prepare()
    reverse_frame = _frame(0.0, np.diag((1.0, -1.0, -1.0)))
    ray_map = _explicit_map(
        exact,
        input_frame=_frame(0.0),
        output_frame=reverse_frame,
        input_index=input_index,
        output_index=input_index,
    )
    expected = np.asarray(
        (
            (1.0, 0.0, 2.0 * distance / input_index, 0.0),
            (0.0, -1.0, 0.0, -2.0 * distance / input_index),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
        )
    )

    assert bool(ray_map.valid)
    np.testing.assert_allclose(ray_map.jacobian, expected, rtol=2.0e-9, atol=2.0e-9)
    traced = exact.execute(jnp.asarray((0.0, 0.0, 0.0)), jnp.asarray((0.0, 0.0, 1.0)))
    assert float(traced.rays.directions[2]) < 0.0
    wrong_orientation = _explicit_map(
        exact,
        input_frame=_frame(0.0),
        output_frame=_frame(0.0),
        input_index=input_index,
        output_index=input_index,
    )
    assert not bool(wrong_orientation.valid)
    assert int(wrong_orientation.status) == int(
        ParaxialOpticsStatus.BRANCH_MARGIN_VIOLATION
    )


def test_aperture_edge_invalidates_differential_map_without_invalidating_chief_hit():
    exact_plan = _single_plan(aperture=0.5)
    exact = exact_plan.prepare()
    reference = jnp.asarray((0.5, 0.0, 0.0, 0.0))
    ray_map = _explicit_map(exact, reference=reference)

    assert not bool(ray_map.valid)
    assert float(ray_map.branch_margin) == 0.0
    assert int(ray_map.status) == int(ParaxialOpticsStatus.BRANCH_MARGIN_VIOLATION)
    cached = ParaxialOpticsPlan(
        exact_plan,
        _frame(0.0),
        _frame(1.0),
        reference,
        input_refractive_index=1.0,
        output_refractive_index=1.0,
        maximum_transverse_perturbation=0.1,
        maximum_angular_perturbation=0.1,
    ).prepare(exact)
    cached_result = cached.execute(reference)
    assert int(cached_result.status) == int(ParaxialOpticsStatus.INVALID_DIFFERENTIAL_MAP)


def test_failed_chief_ray_produces_invalid_differential_evidence():
    incident_index = 1.5
    exact = _single_plan(aperture=10.0, indices=(incident_index, 1.0)).prepare()
    reference = jnp.asarray((0.0, 0.0, incident_index * np.deg2rad(60.0), 0.0))
    ray_map = _explicit_map(
        exact,
        reference=reference,
        input_index=incident_index,
        output_index=1.0,
    )

    assert not bool(ray_map.valid)
    assert int(ray_map.status) == int(ParaxialOpticsStatus.CHIEF_RAY_FAILURE)


def test_cached_paraxial_map_refuses_outside_trust_envelope():
    exact_plan = _single_plan(surface_z=1.0)
    prepared = ParaxialOpticsPlan(
        exact_plan,
        _frame(0.0),
        _frame(1.0),
        jnp.zeros((4,)),
        input_refractive_index=1.0,
        output_refractive_index=1.0,
        maximum_transverse_perturbation=0.1,
        maximum_angular_perturbation=0.05,
    ).prepare()
    values = jnp.asarray(
        (
            (0.05, 0.0, 0.02, 0.0),
            (0.11, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.051, 0.0),
            (jnp.nan, 0.0, 0.0, 0.0),
        )
    )
    result = prepared.execute(values)

    np.testing.assert_array_equal(result.successful, (True, False, False, False))
    np.testing.assert_array_equal(
        result.status,
        (
            ParaxialOpticsStatus.SUCCESS,
            ParaxialOpticsStatus.OUTSIDE_TRUST_ENVELOPE,
            ParaxialOpticsStatus.OUTSIDE_TRUST_ENVELOPE,
            ParaxialOpticsStatus.NONFINITE_INPUT,
        ),
    )
    np.testing.assert_array_equal(result.within_envelope, (True, False, False, False))


def test_cached_execution_is_jittable_and_vmappable():
    exact_plan = _single_plan(surface_z=1.0)
    prepared = ParaxialOpticsPlan(
        exact_plan,
        _frame(0.0),
        _frame(1.0),
        jnp.zeros((4,)),
        input_refractive_index=1.0,
        output_refractive_index=1.0,
        maximum_transverse_perturbation=0.2,
        maximum_angular_perturbation=0.1,
    ).prepare()
    values = jnp.asarray(((0.01, 0.02, 0.03, 0.04), (-0.02, 0.01, 0.0, -0.03)))

    compiled = jax.jit(prepared.execute)(values)
    mapped = jax.vmap(prepared.execute)(values)

    np.testing.assert_allclose(compiled.coordinates, mapped.coordinates, atol=1.0e-12)
    np.testing.assert_array_equal(compiled.status, mapped.status)
