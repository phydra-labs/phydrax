import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.metrix._black_hole_metrics import (
    boyer_lindquist_to_ingoing_kerr_transition,
    ExactMetricDomainStatus,
    ingoing_kerr_domain_evidence,
    ingoing_kerr_metric,
    ingoing_schwarzschild_domain_evidence,
    ingoing_schwarzschild_metric,
    kerr_boyer_lindquist_domain_evidence,
    kerr_boyer_lindquist_metric,
    kerr_ergosurface_radii,
    kerr_horizon_radii,
    kerr_kretschmann_scalar,
    kerr_pontryagin_scalar,
)
from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._lorentzian import schwarzschild_metric


jax.config.update("jax_enable_x64", True)


def _charts():
    boyer_lindquist = CoordinateChart(
        "kerr-boyer-lindquist",
        ("t", "r", "theta", "phi"),
    )
    ingoing = CoordinateChart(
        "kerr-future-ingoing",
        ("v", "r", "theta", "phi_tilde"),
    )
    return boyer_lindquist, ingoing


def test_exact_metrics_have_schwarzschild_limits_and_future_horizon_regularity():
    boyer_lindquist, ingoing = _charts()
    mass = 1.7
    exterior = jnp.array([0.2, 6.0, 0.9, -0.4])

    static_schwarzschild = schwarzschild_metric(
        mass,
        chart=boyer_lindquist,
    )(exterior)
    zero_spin_kerr = kerr_boyer_lindquist_metric(
        mass,
        0.0,
        chart=boyer_lindquist,
    )(exterior)
    np.testing.assert_allclose(zero_spin_kerr, static_schwarzschild, rtol=2e-15)

    ingoing_schwarzschild = ingoing_schwarzschild_metric(
        mass,
        chart=ingoing,
    )
    zero_spin_ingoing_kerr = ingoing_kerr_metric(
        mass,
        0.0,
        chart=ingoing,
    )
    sample = jnp.array([0.3, 2.6, 1.1, 0.7])
    np.testing.assert_allclose(
        zero_spin_ingoing_kerr(sample),
        ingoing_schwarzschild(sample),
        rtol=2e-15,
        atol=0,
    )

    horizon = jnp.array([0.3, 2.0 * mass, 1.1, 0.7])
    matrix = ingoing_schwarzschild(horizon)
    assert bool(jnp.all(jnp.isfinite(matrix)))
    np.testing.assert_allclose(matrix[0, 0], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(matrix[0, 1], 1.0, rtol=0, atol=0)
    expected_determinant = -((2.0 * mass) ** 4) * np.sin(1.1) ** 2
    np.testing.assert_allclose(jnp.linalg.det(matrix), expected_determinant)


def test_ingoing_kerr_is_regular_on_the_future_horizon_and_retains_spin_sign():
    boyer_lindquist, ingoing = _charts()
    mass = 2.0
    spin = 1.1
    outer = kerr_horizon_radii(mass, spin)[1]
    horizon = jnp.array([0.4, outer, 1.0, -0.2])

    future_metric = ingoing_kerr_metric(mass, -spin, chart=ingoing)
    horizon_matrix = future_metric(horizon)
    sigma = outer**2 + spin**2 * jnp.cos(horizon[2]) ** 2
    expected_determinant = -((sigma * jnp.sin(horizon[2])) ** 2)
    assert bool(jnp.all(jnp.isfinite(horizon_matrix)))
    np.testing.assert_allclose(jnp.linalg.det(horizon_matrix), expected_determinant)

    point = jnp.array([0.1, 5.0, 1.2, 0.3])
    ingoing_positive = ingoing_kerr_metric(mass, spin, chart=ingoing)(point)
    ingoing_negative = ingoing_kerr_metric(mass, -spin, chart=ingoing)(point)
    np.testing.assert_allclose(jnp.diag(ingoing_negative), jnp.diag(ingoing_positive))
    np.testing.assert_allclose(ingoing_negative[0, 3], -ingoing_positive[0, 3])
    np.testing.assert_allclose(ingoing_negative[1, 3], -ingoing_positive[1, 3])

    boyer_positive = kerr_boyer_lindquist_metric(
        mass,
        spin,
        chart=boyer_lindquist,
    )(point)
    boyer_negative = kerr_boyer_lindquist_metric(
        mass,
        -spin,
        chart=boyer_lindquist,
    )(point)
    np.testing.assert_allclose(jnp.diag(boyer_negative), jnp.diag(boyer_positive))
    np.testing.assert_allclose(boyer_negative[0, 3], -boyer_positive[0, 3])
    np.testing.assert_allclose(
        ingoing_kerr_metric(
            mass,
            spin,
            chart=ingoing,
            convention="mostly_minus",
        )(point),
        -ingoing_positive,
    )


def test_boyer_lindquist_to_ingoing_transition_has_exact_jacobian_and_pullback():
    boyer_lindquist, ingoing = _charts()
    mass = 2.0
    point = jnp.array([0.3, 7.0, 1.1, -0.2])

    for spin in (0.0, -0.7, mass):
        transition = boyer_lindquist_to_ingoing_kerr_transition(
            mass,
            spin,
            source=boyer_lindquist,
            target=ingoing,
        )
        mapped = transition(point)
        jacobian = transition.jacobian(point)
        radii = kerr_horizon_radii(mass, spin)
        delta = (point[1] - radii[1]) * (point[1] - radii[0])
        expected = jnp.array(
            [
                [1.0, (point[1] ** 2 + spin**2) / delta, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, spin / delta, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(jacobian, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(
            transition.inverse(mapped),
            point,
            rtol=2e-14,
            atol=2e-14,
        )

        ingoing_matrix = ingoing_kerr_metric(
            mass,
            spin,
            chart=ingoing,
        )(mapped)
        pulled_back = jacobian.T @ ingoing_matrix @ jacobian
        boyer_matrix = kerr_boyer_lindquist_metric(
            mass,
            spin,
            chart=boyer_lindquist,
        )(point)
        np.testing.assert_allclose(
            pulled_back,
            boyer_matrix,
            rtol=3e-14,
            atol=3e-14,
        )


def test_kerr_roots_are_stable_and_overextremal_inputs_are_not_clipped():
    near_extremal_spin = np.nextafter(1.0, 0.0)
    expected_gap = np.sqrt((1.0 - near_extremal_spin) * (1.0 + near_extremal_spin))
    near_extremal = kerr_horizon_radii(1.0, near_extremal_spin)
    np.testing.assert_allclose(
        near_extremal[1] - 1.0,
        expected_gap,
        rtol=2e-15,
    )

    small_spin = 1.0e-12
    small_spin_radii = kerr_horizon_radii(1.0, small_spin)
    assert float(small_spin_radii[0]) > 0.0
    np.testing.assert_allclose(
        small_spin_radii[0] * small_spin_radii[1],
        small_spin**2,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        kerr_horizon_radii(2.0, -0.8),
        kerr_horizon_radii(2.0, 0.8),
        rtol=0,
        atol=0,
    )

    horizons = kerr_horizon_radii(1.0, 1.01)
    equatorial_ergosurfaces = kerr_ergosurface_radii(
        1.0,
        1.01,
        0.5 * jnp.pi,
    )
    polar_ergosurfaces = kerr_ergosurface_radii(1.0, 1.01, 0.0)
    assert bool(jnp.all(jnp.isnan(horizons)))
    np.testing.assert_allclose(
        equatorial_ergosurfaces,
        jnp.array([0.0, 2.0]),
        atol=2e-16,
    )
    assert bool(jnp.all(jnp.isnan(polar_ergosurfaces)))

    mass = 2.0
    spin = -0.8
    np.testing.assert_allclose(
        kerr_ergosurface_radii(mass, spin, 0.0),
        kerr_horizon_radii(mass, spin),
    )
    np.testing.assert_allclose(
        kerr_ergosurface_radii(mass, spin, 0.5 * jnp.pi),
        jnp.array([0.0, 2.0 * mass]),
        atol=2e-16,
    )


def test_metric_domains_distinguish_axis_ring_horizon_and_invalid_parameters():
    boyer_lindquist, ingoing = _charts()
    mass = 1.0
    spin = 0.6
    outer = kerr_horizon_radii(mass, spin)[1]
    points = jnp.array(
        [
            [0.0, 3.0, 0.8, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.5 * jnp.pi, 0.0],
            [0.0, outer, 0.8, 0.0],
            [0.0, jnp.nan, 0.8, 0.0],
        ]
    )

    ingoing_evidence = ingoing_kerr_domain_evidence(
        mass,
        spin,
        points,
        chart=ingoing,
    )
    expected_ingoing_status = jnp.array(
        [
            ExactMetricDomainStatus.VALID,
            ExactMetricDomainStatus.AXIS,
            ExactMetricDomainStatus.RING,
            ExactMetricDomainStatus.VALID,
            ExactMetricDomainStatus.NONFINITE,
        ],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(ingoing_evidence.status, expected_ingoing_status)
    np.testing.assert_array_equal(
        ingoing_evidence.valid,
        jnp.array([True, False, False, True, False]),
    )
    assert bool(ingoing_evidence.derivative_valid[0])
    assert bool(ingoing_evidence.derivative_valid[3])
    assert not bool(ingoing_evidence.finite[4])

    boyer_evidence = kerr_boyer_lindquist_domain_evidence(
        mass,
        spin,
        points[3],
        chart=boyer_lindquist,
    )
    assert not bool(boyer_evidence.valid)
    assert int(boyer_evidence.status) == int(
        ExactMetricDomainStatus.BOYER_LINDQUIST_HORIZON
    )

    overextremal = ingoing_kerr_domain_evidence(
        mass,
        1.01,
        points[0],
        chart=ingoing,
    )
    assert bool(overextremal.physically_valid)
    assert bool(overextremal.derivative_valid)
    assert int(overextremal.status) == int(ExactMetricDomainStatus.VALID)
    assert bool(
        jnp.all(jnp.isfinite(ingoing_kerr_metric(mass, 1.01, chart=ingoing)(points[0])))
    )
    assert bool(
        jnp.all(
            jnp.isfinite(
                kerr_boyer_lindquist_metric(
                    mass,
                    1.01,
                    chart=boyer_lindquist,
                )(points[0])
            )
        )
    )

    invalid_parameters = ingoing_kerr_domain_evidence(
        0.0,
        spin,
        points[0],
        chart=ingoing,
    )
    assert not bool(invalid_parameters.physically_valid)
    assert int(invalid_parameters.status) == int(
        ExactMetricDomainStatus.INVALID_PARAMETERS
    )

    schwarzschild_axis = ingoing_schwarzschild_domain_evidence(
        mass,
        points[1],
        chart=ingoing,
    )
    assert int(schwarzschild_axis.status) == int(ExactMetricDomainStatus.AXIS)


def test_stationary_invariants_and_metric_maps_support_jit_vmap_and_grad():
    _, ingoing = _charts()
    mass = 1.3
    radius = 5.0
    polar = 0.7
    expected_schwarzschild = 48.0 * mass**2 / radius**6
    np.testing.assert_allclose(
        kerr_kretschmann_scalar(mass, 0.0, radius, polar),
        expected_schwarzschild,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        kerr_pontryagin_scalar(mass, 0.0, radius, polar),
        0.0,
        rtol=0,
        atol=0,
    )

    spin = 0.4
    positive_k = kerr_kretschmann_scalar(mass, spin, radius, polar)
    negative_k = kerr_kretschmann_scalar(mass, -spin, radius, polar)
    positive_p = kerr_pontryagin_scalar(mass, spin, radius, polar)
    negative_p = kerr_pontryagin_scalar(mass, -spin, radius, polar)
    assert bool(jnp.isfinite(kerr_kretschmann_scalar(mass, 2.0 * mass, radius, polar)))
    np.testing.assert_allclose(negative_k, positive_k, rtol=2e-15)
    np.testing.assert_allclose(negative_p, -positive_p, rtol=2e-15)

    radii = jnp.array([4.0, 5.0, 6.0])
    batched_invariant = jax.jit(
        jax.vmap(lambda value: kerr_kretschmann_scalar(mass, spin, value, polar))
    )(radii)
    assert batched_invariant.shape == radii.shape
    assert bool(jnp.all(jnp.isfinite(batched_invariant)))
    assert bool(
        jnp.isfinite(
            jax.grad(lambda value: kerr_kretschmann_scalar(mass, spin, value, polar))(
                radius
            )
        )
    )

    metric = ingoing_kerr_metric(mass, spin, chart=ingoing)
    points = jnp.array(
        [
            [0.0, 4.0, 0.8, 0.1],
            [0.2, 5.0, 1.1, -0.3],
        ]
    )
    compiled = eqx.filter_jit(metric)(points)
    np.testing.assert_allclose(compiled, metric(points), rtol=2e-15)
    spin_derivative = jax.grad(
        lambda value: ingoing_kerr_metric(
            mass,
            value,
            chart=ingoing,
        )(points[0])[0, 3]
    )(spin)
    assert bool(jnp.isfinite(spin_derivative))
