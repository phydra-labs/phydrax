#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.analytic import RigidFrame
from phydrax.optics.geometric._sequential import (
    PreparedSequentialOptics,
    SequentialOpticsPlan,
    SequentialOpticsStatus,
)


def _frame(z: float) -> RigidFrame:
    return RigidFrame(np.eye(3), np.asarray((0.0, 0.0, z)))


def _one_surface(
    kind: str,
    *,
    interaction: str = "transmit",
    z: float = 1.0,
    curvature: float = 0.0,
    conic: float = 0.0,
    coefficient: float = 0.0,
    aperture: float | None = 2.0,
    indices: tuple[float, float] = (1.0, 1.0),
    bracket_sample_count: int = 64,
    root_iteration_count: int = 40,
    intersection_tolerance: float = 1.0e-10,
    maximum_distance: float = 30.0,
) -> SequentialOpticsPlan:
    asphere = kind == "even-asphere"
    coefficients = np.asarray(((coefficient if asphere else 0.0,),))
    coefficient_active = np.asarray(((asphere,),))
    return SequentialOpticsPlan(
        (_frame(z),),
        (kind,),
        (interaction,),
        np.asarray((curvature,)),
        np.asarray((conic,)),
        coefficients,
        coefficient_active,
        np.asarray((0.0 if aperture is None else aperture,)),
        np.asarray((aperture is not None,)),
        np.asarray((maximum_distance,)),
        np.asarray(indices),
        bracket_sample_count=bracket_sample_count,
        root_iteration_count=root_iteration_count,
        intersection_tolerance=intersection_tolerance,
    )


@pytest.mark.parametrize(
    ("kind", "curvature", "conic", "coefficient"),
    (
        ("plane", 0.0, 0.0, 0.0),
        ("sphere", 0.2, 0.0, 0.0),
        ("conic", 0.2, -0.5, 0.0),
        ("even-asphere", 0.2, -0.5, 0.01),
    ),
)
def test_every_surface_kind_hits_its_vertex_connected_branch(
    kind, curvature, conic, coefficient
):
    prepared = _one_surface(
        kind,
        curvature=curvature,
        conic=conic,
        coefficient=coefficient,
    ).prepare()
    result = prepared.execute(jnp.asarray((0.1, 0.0, 0.0)), jnp.asarray((0.0, 0.0, 1.0)))

    assert bool(result.successful)
    assert int(result.status) == int(SequentialOpticsStatus.SUCCESS)
    assert int(result.traversed_surfaces) == 1
    assert np.isfinite(float(result.maximum_intersection_residual))
    assert float(result.maximum_intersection_residual) <= 1.0e-9
    if kind != "plane":
        expected_sag = (
            curvature * 0.01 / (1.0 + np.sqrt(1.0 - (1.0 + conic) * curvature**2 * 0.01))
        )
        if kind == "even-asphere":
            expected_sag += coefficient * 0.1**4
        np.testing.assert_allclose(
            result.rays.origins[2], 1.0 + expected_sag, atol=2.0e-9
        )


def test_sphere_selects_nearest_forward_vertex_branch():
    prepared = _one_surface(
        "sphere", curvature=0.2, z=1.0, aperture=None, maximum_distance=20.0
    ).prepare()
    result = prepared.execute(jnp.asarray((0.0, 0.0, -1.0)), jnp.asarray((0.0, 0.0, 1.0)))

    assert bool(result.successful)
    np.testing.assert_allclose(result.rays.origins, (0.0, 0.0, 1.0), atol=1.0e-11)
    np.testing.assert_allclose(result.rays.geometric_path_lengths, 2.0, atol=1.0e-11)


def test_bounded_asphere_solver_selects_nearest_of_two_forward_roots():
    prepared = _one_surface(
        "even-asphere",
        curvature=0.0,
        conic=0.0,
        coefficient=1.0,
        aperture=2.0,
        maximum_distance=3.0,
    ).prepare()
    result = prepared.execute(jnp.asarray((0.0, 0.0, 0.9)), jnp.asarray((1.0, 0.0, 1.0)))
    positive_roots = sorted(
        root.real
        for root in np.roots((1.0, 0.0, 0.0, -1.0, 0.1))
        if abs(root.imag) < 1.0e-12 and root.real > 0.0
    )
    nearest = positive_roots[0]

    assert bool(result.successful)
    assert len(positive_roots) == 2
    np.testing.assert_allclose(
        result.rays.origins,
        (nearest, 0.0, 0.9 + nearest),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert float(result.rays.origins[0]) < 0.2


def test_circular_aperture_boundary_and_clipping_are_explicit():
    prepared = _one_surface("plane", aperture=0.5).prepare()
    origins = jnp.asarray(((0.5, 0.0, 0.0), (0.5001, 0.0, 0.0)))
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), origins.shape)
    result = prepared.execute(origins, directions)

    np.testing.assert_array_equal(result.successful, (True, False))
    np.testing.assert_array_equal(
        result.status,
        (SequentialOpticsStatus.SUCCESS, SequentialOpticsStatus.APERTURE_CLIPPED),
    )
    np.testing.assert_allclose(
        result.minimum_aperture_margin, (0.0, -1.0e-4), atol=1.0e-12
    )
    np.testing.assert_allclose(result.rays.origins[1], origins[1])
    assert int(result.traversed_surfaces[1]) == 0


def test_plane_hit_failures_have_distinct_statuses():
    prepared = _one_surface("plane", aperture=None, maximum_distance=0.5).prepare()
    origins = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0),
            (jnp.nan, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    directions = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    result = prepared.execute(origins, directions)

    np.testing.assert_array_equal(
        result.status,
        (
            SequentialOpticsStatus.PARALLEL,
            SequentialOpticsStatus.COPLANAR,
            SequentialOpticsStatus.BEHIND_RAY,
            SequentialOpticsStatus.MISSED_SURFACE,
            SequentialOpticsStatus.NONFINITE_INPUT,
            SequentialOpticsStatus.INVALID_DIRECTION,
        ),
    )
    assert not np.any(np.asarray(result.successful))


def test_forward_hit_from_wrong_normal_side_is_rejected():
    prepared = _one_surface("plane", aperture=2.0).prepare()
    result = prepared.execute(jnp.asarray((0.0, 0.0, 2.0)), jnp.asarray((0.0, 0.0, -1.0)))

    assert not bool(result.successful)
    assert int(result.status) == int(SequentialOpticsStatus.WRONG_SIDE_INCIDENCE)
    assert int(result.traversed_surfaces) == 0
    np.testing.assert_allclose(result.rays.origins, (0.0, 0.0, 2.0))


def test_sphere_miss_and_tangency_are_not_root_fallbacks():
    prepared = _one_surface(
        "sphere", curvature=1.0, z=1.0, aperture=None, maximum_distance=5.0
    ).prepare()
    origins = jnp.asarray(((2.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    directions = jnp.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    result = prepared.execute(origins, directions)

    np.testing.assert_array_equal(
        result.status,
        (SequentialOpticsStatus.MISSED_SURFACE, SequentialOpticsStatus.TANGENT_SURFACE),
    )


def test_conic_tangent_is_detected_without_a_sign_change():
    prepared = _one_surface(
        "conic",
        curvature=1.0,
        conic=-1.0,
        aperture=1.0,
        maximum_distance=3.0,
    ).prepare()
    result = prepared.execute(
        jnp.asarray((-0.5, 0.0, 0.625)), jnp.asarray((1.0, 0.0, 0.5))
    )

    assert not bool(result.successful)
    assert int(result.status) == int(SequentialOpticsStatus.TANGENT_SURFACE)


def test_invalid_sag_domain_and_solver_exhaustion_are_distinct():
    invalid_domain = _one_surface(
        "conic", curvature=1.0, conic=0.0, aperture=0.9, maximum_distance=3.0
    ).prepare()
    domain_result = invalid_domain.execute(
        jnp.asarray((1.1, 0.0, 0.0)), jnp.asarray((0.0, 0.0, 1.0))
    )
    assert int(domain_result.status) == int(SequentialOpticsStatus.INVALID_SAG_DOMAIN)

    exhausted = _one_surface(
        "even-asphere",
        curvature=0.4,
        conic=-0.5,
        coefficient=0.4,
        aperture=2.0,
        maximum_distance=2.0,
        bracket_sample_count=2,
        root_iteration_count=1,
        intersection_tolerance=1.0e-15,
    ).prepare()
    exhausted_result = exhausted.execute(
        jnp.asarray((0.7, 0.0, 0.0)), jnp.asarray((0.2, 0.0, 1.0))
    )
    assert int(exhausted_result.status) == int(SequentialOpticsStatus.ROOT_NONCONVERGENCE)


def test_transmit_tir_fails_but_declared_reflection_succeeds():
    theta = np.deg2rad(60.0)
    direction = jnp.asarray((np.sin(theta), 0.0, np.cos(theta)))
    transmit = _one_surface(
        "plane", indices=(1.5, 1.0), aperture=10.0, maximum_distance=5.0
    ).prepare()
    transmitted = transmit.execute(jnp.asarray((0.0, 0.0, 0.0)), direction)
    assert int(transmitted.status) == int(
        SequentialOpticsStatus.TOTAL_INTERNAL_REFLECTION
    )
    assert int(transmitted.traversed_surfaces) == 0
    np.testing.assert_allclose(transmitted.rays.origins, (0.0, 0.0, 0.0))

    reflect = _one_surface(
        "plane",
        interaction="reflect",
        indices=(1.5, 1.5),
        aperture=10.0,
        maximum_distance=5.0,
    ).prepare()
    reflected = reflect.execute(jnp.asarray((0.0, 0.0, 0.0)), direction)
    assert bool(reflected.successful)
    assert float(reflected.rays.directions[2]) < 0.0
    np.testing.assert_allclose(reflected.rays.refractive_indices, 1.5)


def test_multiple_surfaces_accumulate_geometric_and_optical_path_only_on_success():
    plan = SequentialOpticsPlan(
        (_frame(1.0), _frame(3.0)),
        ("plane", "plane"),
        ("transmit", "transmit"),
        np.zeros((2,)),
        np.zeros((2,)),
        np.zeros((2, 0)),
        np.zeros((2, 0), dtype=bool),
        np.zeros((2,)),
        np.zeros((2,), dtype=bool),
        np.asarray((2.0, 3.0)),
        np.asarray((1.0, 1.5, 1.0)),
    )
    result = plan.prepare().execute(
        jnp.asarray((0.0, 0.0, 0.0)), jnp.asarray((0.0, 0.0, 1.0))
    )

    assert bool(result.successful)
    assert int(result.traversed_surfaces) == 2
    np.testing.assert_allclose(result.rays.geometric_path_lengths, 3.0)
    np.testing.assert_allclose(result.rays.optical_path_lengths, 4.0)
    np.testing.assert_allclose(result.rays.refractive_indices, 1.0)


def test_fixed_layout_validation_rejects_non_neutral_rows_and_index_crossing():
    with pytest.raises(ValueError, match="Plane rows"):
        _one_surface("plane", curvature=0.1)
    with pytest.raises(ValueError, match="Sphere rows"):
        _one_surface("sphere", curvature=0.1, conic=-1.0)
    with pytest.raises(ValueError, match="retain"):
        _one_surface("plane", interaction="reflect", indices=(1.0, 1.5))
    with pytest.raises(ValueError, match="sag domain"):
        _one_surface("conic", curvature=1.0, conic=0.0, aperture=1.1)


def test_prepared_work_bounds_are_static_and_exact():
    plan = SequentialOpticsPlan(
        (_frame(1.0), _frame(2.0), _frame(3.0), _frame(4.0)),
        ("plane", "sphere", "conic", "even-asphere"),
        ("transmit",) * 4,
        np.asarray((0.0, 0.1, 0.1, 0.1)),
        np.asarray((0.0, 0.0, -1.0, -1.0)),
        np.asarray(((0.0,), (0.0,), (0.0,), (0.01,))),
        np.asarray(((False,), (False,), (False,), (True,))),
        np.asarray((0.0, 0.0, 1.0, 1.0)),
        np.asarray((False, False, True, True)),
        np.full((4,), 4.0),
        np.ones((5,)),
        bracket_sample_count=10,
        root_iteration_count=7,
    )
    prepared = plan.prepare()

    assert isinstance(prepared, PreparedSequentialOptics)
    assert prepared.worst_case_surface_evaluations == 4
    assert prepared.worst_case_root_evaluations == 2 * (11 + 3 * 7 + 2)
    assert prepared.kind_tags == (0, 1, 2, 3)
    assert prepared.interaction_tags == (0, 0, 0, 0)


def test_mixed_batch_jit_and_vmap_preserve_lane_statuses():
    prepared = _one_surface("plane", aperture=0.75).prepare()
    origins = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    directions = jnp.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, 1.0)))

    compiled = jax.jit(prepared.execute)(origins, directions)
    mapped = jax.vmap(lambda o, d: prepared.execute(o, d))(origins, directions)

    np.testing.assert_array_equal(compiled.status, mapped.status)
    np.testing.assert_array_equal(
        compiled.status,
        (SequentialOpticsStatus.SUCCESS, SequentialOpticsStatus.APERTURE_CLIPPED),
    )
    np.testing.assert_allclose(compiled.rays.directions[0], (0.0, 0.0, 1.0))


def test_ordinary_trace_does_not_request_a_jacobian(monkeypatch):
    prepared = _one_surface("plane").prepare()

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary tracing must not form a Jacobian")

    monkeypatch.setattr(jax, "jacfwd", forbidden)
    monkeypatch.setattr(jax, "jacrev", forbidden)
    result = prepared.execute(jnp.asarray((0.0, 0.0, 0.0)), jnp.asarray((0.0, 0.0, 1.0)))
    assert bool(result.successful)
