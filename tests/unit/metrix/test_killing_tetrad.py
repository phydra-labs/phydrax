#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax._trainable import NonTrainableState
from phydrax.metrix._black_hole_metrics import (
    ingoing_schwarzschild_metric,
    kerr_boyer_lindquist_metric,
)
from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._killing_tetrad import (
    axial_killing_vector,
    kerr_principal_null_tetrad,
    maximum_killing_equation_residual,
    orthonormal_tetrad,
    stationary_axial_inner_product_evidence,
    stationary_killing_vector,
    tetrad_dual,
    tetrad_parallel_transport_evidence,
    tetrad_project_covector,
    tetrad_project_vector,
    tetrad_reconstruct_covector,
    tetrad_reconstruct_vector,
    zamo_observer_tetrad,
)
from phydrax.metrix._lorentzian import minkowski_metric, schwarzschild_metric
from phydrax.metrix._metric import LorentzianMetric
from phydrax.metrix._metric_domain import MetricDomainStatus
from phydrax.metrix._spacetime_conventions import RelativityConvention


def _spherical_chart(name: str = "spherical") -> CoordinateChart:
    return CoordinateChart(name, ("t", "r", "theta", "phi"))


def _spherical_minkowski_metric(coordinates):
    _, radius, polar, _ = coordinates
    return jnp.diag(jnp.asarray((-1.0, 1.0, radius**2, radius**2 * jnp.sin(polar) ** 2)))


@pytest.mark.parametrize("metric_signature", ("mostly_plus", "mostly_minus"))
def test_stationary_axial_killing_fields_and_inner_product_evidence(
    metric_signature,
):
    chart = _spherical_chart("kerr-killing")
    metric = kerr_boyer_lindquist_metric(
        1.0, 0.7, chart=chart, convention=metric_signature
    )
    points = jnp.asarray(((0.0, 4.0, 0.9, 0.2), (0.3, 6.0, 1.2, -0.4)))
    stationary = stationary_killing_vector(points)
    axial = axial_killing_vector(points)
    evidence = stationary_axial_inner_product_evidence(metric, points)

    assert stationary.shape == (2, 4)
    assert axial.shape == (2, 4)
    assert jnp.all(stationary[:, 0] == 1.0)
    assert jnp.all(axial[:, 3] == 1.0)
    assert jnp.all(evidence.finite)
    assert jnp.all(evidence.stationary_timelike)
    assert jnp.all(evidence.axial_spacelike)
    assert jnp.all(evidence.orbit_plane_lorentzian)
    assert jnp.all(evidence.qualified)
    assert jnp.all(evidence.derivative_valid)

    stationary_residual = maximum_killing_equation_residual(
        metric, stationary_killing_vector, points
    )
    axial_residual = maximum_killing_equation_residual(
        metric, axial_killing_vector, points
    )
    assert jnp.allclose(stationary_residual, 0.0, atol=2e-12)
    assert jnp.allclose(axial_residual, 0.0, atol=2e-12)


def test_ergoregion_classification_does_not_invalidate_zamo_orbit_plane():
    chart = _spherical_chart("kerr-ergoregion")
    metric = kerr_boyer_lindquist_metric(1.0, 0.8, chart=chart)
    point = jnp.asarray((0.0, 1.8, jnp.pi / 2.0, 0.0))

    killing = stationary_axial_inner_product_evidence(metric, point)
    tetrad = zamo_observer_tetrad(metric, point, source_id="kerr-M1-a0.8")

    assert not killing.stationary_timelike
    assert killing.axial_spacelike
    assert killing.orbit_plane_lorentzian
    assert killing.physically_valid
    assert tetrad.domain.physically_valid
    assert tetrad.qualified
    assert tetrad.inner_products.maximum_absolute_residual < 2e-12
    assert tetrad.time_vector[0] > 0.0


@pytest.mark.parametrize("metric_signature", ("mostly_plus", "mostly_minus"))
def test_schwarzschild_zamo_limit_orientation_dual_and_projection_round_trips(
    metric_signature,
):
    chart = _spherical_chart("schwarzschild-zamo")
    mass = 1.3
    radius = 5.2
    polar = 1.1
    point = jnp.asarray((0.0, radius, polar, 0.2))
    metric = schwarzschild_metric(mass, chart=chart, convention=metric_signature)
    tetrad = zamo_observer_tetrad(
        metric, point, source_id=f"schwarzschild-{metric_signature}"
    )
    factor = 1.0 - 2.0 * mass / radius
    expected = jnp.diag(
        jnp.asarray(
            (
                1.0 / jnp.sqrt(factor),
                jnp.sqrt(factor),
                1.0 / radius,
                1.0 / (radius * jnp.sin(polar)),
            )
        )
    )

    assert jnp.allclose(tetrad.vectors, expected, atol=2e-12)
    assert jnp.allclose(tetrad.inner_products.maximum_absolute_residual, 0.0, atol=2e-12)
    assert tetrad.orientation == 1.0
    assert tetrad.orientation_residual == 0.0
    assert tetrad.time_direction_residual == 0.0
    assert tetrad.finite
    assert tetrad.physically_valid
    assert tetrad.qualified
    assert tetrad.derivative_valid
    assert jnp.allclose(
        jnp.einsum("ai,bi->ab", tetrad_dual(tetrad), tetrad.vectors),
        jnp.eye(4),
        atol=2e-12,
    )

    vector = jnp.asarray((0.4, -0.3, 0.2, 0.7))
    covector = jnp.asarray((-0.5, 0.6, -0.1, 0.9))
    vector_components = tetrad_project_vector(tetrad, vector)
    covector_components = tetrad_project_covector(tetrad, covector)
    assert jnp.allclose(
        tetrad_reconstruct_vector(tetrad, vector_components), vector, atol=2e-12
    )
    assert jnp.allclose(
        tetrad_reconstruct_covector(tetrad, covector_components),
        covector,
        atol=2e-12,
    )


def test_declared_time_and_spacetime_orientations_control_tetrad_evidence():
    chart = CoordinateChart("oriented-flat", ("t", "x", "y", "z"))
    metric = minkowski_metric(chart)
    convention = RelativityConvention(
        future_time_orientation=-1,
        spacetime_orientation=-1,
        azimuthal_orientation=-1,
    )
    point = jnp.zeros((4,))
    vectors = jnp.diag(jnp.asarray((-1.0, 1.0, 1.0, 1.0)))
    tetrad = orthonormal_tetrad(
        metric,
        vectors,
        point,
        convention=convention,
        source_id="past-oriented-flat-frame",
    )

    assert tetrad.orientation == -1.0
    assert tetrad.orientation_residual == 0.0
    assert tetrad.time_direction_residual == 0.0
    assert tetrad.qualified
    assert tetrad.convention.convention_id == convention.convention_id


def test_zamo_reports_axis_and_regular_horizon_as_observer_domain_failures():
    chart = _spherical_chart("observer-domain")
    spherical_flat = LorentzianMetric(_spherical_minkowski_metric, chart=chart)
    axis = zamo_observer_tetrad(
        spherical_flat,
        jnp.asarray((0.0, 2.0, 0.0, 0.0)),
        source_id="spherical-flat",
    )
    assert axis.finite
    assert not axis.domain_valid
    assert not axis.physically_valid
    assert not axis.qualified
    assert not axis.derivative_valid
    assert axis.domain.status == int(MetricDomainStatus.REJECTED)
    assert jnp.all(axis.vectors == 0.0)

    ingoing_chart = CoordinateChart("ingoing-schwarzschild", ("v", "r", "theta", "phi"))
    horizon_metric = ingoing_schwarzschild_metric(1.0, chart=ingoing_chart)
    horizon = zamo_observer_tetrad(
        horizon_metric,
        jnp.asarray((0.0, 2.0, 1.0, 0.0)),
        source_id="ingoing-schwarzschild-M1",
    )
    assert jnp.all(jnp.isfinite(horizon_metric(jnp.asarray((0.0, 2.0, 1.0, 0.0)))))
    assert horizon.finite
    assert not horizon.domain_valid
    assert not horizon.physically_valid
    assert not horizon.qualified
    assert not horizon.derivative_valid
    assert horizon.domain.margin == 0.0
    assert horizon.domain.status == int(MetricDomainStatus.REJECTED)


@pytest.mark.parametrize("metric_signature", ("mostly_plus", "mostly_minus"))
@pytest.mark.parametrize("spin", (0.6, -0.6))
def test_kerr_principal_null_tetrad_normalization_and_dual(metric_signature, spin):
    chart = _spherical_chart("kerr-principal")
    mass = 1.4
    metric = kerr_boyer_lindquist_metric(
        mass, spin, chart=chart, convention=metric_signature
    )
    point = jnp.asarray((0.2, 5.0, 1.0, -0.4))
    tetrad = kerr_principal_null_tetrad(
        metric,
        mass,
        spin,
        point,
        source_id=f"kerr-M{mass}-a{spin}-{metric_signature}",
    )

    assert tetrad.finite
    assert tetrad.domain_valid
    assert tetrad.physically_valid
    assert tetrad.qualified
    assert tetrad.derivative_valid
    assert jnp.sign(jnp.real(tetrad.outgoing[3])) == jnp.sign(spin)
    assert jnp.allclose(tetrad.polarization_conjugate, jnp.conj(tetrad.polarization))
    assert tetrad.inner_products.maximum_absolute_residual < 3e-12
    assert jnp.allclose(
        jnp.einsum("ai,bi->ab", tetrad.dual_covectors, tetrad.vectors),
        jnp.eye(4),
        atol=3e-12,
    )

    vector = jnp.asarray((0.4, -0.3, 0.2, 0.7), dtype="complex128")
    components = tetrad_project_vector(tetrad, vector)
    assert jnp.allclose(tetrad_reconstruct_vector(tetrad, components), vector, atol=3e-12)


def test_principal_null_tetrad_has_exact_schwarzschild_limit_and_domain_mask():
    chart = _spherical_chart("schwarzschild-principal")
    mass = 1.0
    radius = 4.0
    polar = 1.2
    point = jnp.asarray((0.0, radius, polar, 0.0))
    metric = kerr_boyer_lindquist_metric(mass, 0.0, chart=chart)
    tetrad = kerr_principal_null_tetrad(metric, mass, 0.0, point, source_id="kerr-M1-a0")
    factor = 1.0 - 2.0 * mass / radius

    assert jnp.allclose(
        tetrad.outgoing,
        jnp.asarray((1.0 / factor, 1.0, 0.0, 0.0)),
        atol=2e-12,
    )
    assert jnp.allclose(
        tetrad.ingoing,
        jnp.asarray((0.5, -0.5 * factor, 0.0, 0.0)),
        atol=2e-12,
    )
    assert jnp.allclose(
        tetrad.polarization,
        jnp.asarray((0.0, 0.0, 1.0, 1j / jnp.sin(polar)), dtype="complex128")
        / (jnp.sqrt(2.0) * radius),
        atol=2e-12,
    )

    ingoing_chart = CoordinateChart("principal-horizon-probe", ("v", "r", "theta", "phi"))
    horizon_regular_metric = ingoing_schwarzschild_metric(mass, chart=ingoing_chart)
    horizon = kerr_principal_null_tetrad(
        horizon_regular_metric,
        mass,
        0.0,
        jnp.asarray((0.0, 2.0, polar, 0.0)),
        source_id="BL-principal-domain-probe",
    )
    assert horizon.finite
    assert not horizon.domain_valid
    assert not horizon.qualified
    assert not horizon.derivative_valid
    assert jnp.all(horizon.vectors == 0.0)


def test_overextremal_kerr_still_has_local_principal_null_directions():
    chart = _spherical_chart("overextremal-principal")
    metric = kerr_boyer_lindquist_metric(1.0, 1.2, chart=chart)
    tetrad = kerr_principal_null_tetrad(
        metric,
        1.0,
        1.2,
        jnp.asarray((0.0, 2.5, 1.0, 0.0)),
        source_id="overextremal-kerr",
    )

    assert tetrad.domain_valid
    assert tetrad.qualified
    assert tetrad.inner_products.maximum_absolute_residual < 3e-12


def test_identities_are_immutable_and_bind_source_and_convention():
    chart = _spherical_chart("identity")
    metric = schwarzschild_metric(1.0, chart=chart)
    point = jnp.asarray((0.0, 4.0, 1.0, 0.0))
    first = zamo_observer_tetrad(metric, point, source_id="geometry-A")
    repeated = zamo_observer_tetrad(metric, point, source_id="geometry-A")
    different = zamo_observer_tetrad(metric, point, source_id="geometry-B")

    assert isinstance(first, NonTrainableState)
    assert first.tetrad_id == repeated.tetrad_id
    assert first.domain.domain_id == repeated.domain.domain_id
    assert first.tetrad_id != different.tetrad_id
    assert first.domain.domain_id != different.domain.domain_id
    with pytest.raises(AttributeError):
        first.tetrad_id = "tampered"
    with pytest.raises(AttributeError):
        first.vectors = jnp.zeros((4, 4))


def test_batched_jit_and_parallel_transport_evidence_are_fixed_shape():
    spherical_chart = _spherical_chart("batched-kerr")
    kerr_metric = kerr_boyer_lindquist_metric(1.0, 0.5, chart=spherical_chart)
    points = jnp.asarray(
        (
            (0.0, 3.0, 0.8, 0.0),
            (0.1, 4.0, 1.1, 0.2),
            (-0.2, 6.0, 1.4, -0.3),
        )
    )

    vectors, qualified, margins = jax.jit(
        lambda values: (
            zamo_observer_tetrad(kerr_metric, values, source_id="batched-kerr").vectors,
            zamo_observer_tetrad(kerr_metric, values, source_id="batched-kerr").qualified,
            zamo_observer_tetrad(
                kerr_metric, values, source_id="batched-kerr"
            ).domain.margin,
        )
    )(points)
    null_vectors, null_qualified = jax.jit(
        lambda values: (
            kerr_principal_null_tetrad(
                kerr_metric,
                1.0,
                0.5,
                values,
                source_id="batched-kerr",
            ).vectors,
            kerr_principal_null_tetrad(
                kerr_metric,
                1.0,
                0.5,
                values,
                source_id="batched-kerr",
            ).qualified,
        )
    )(points)
    assert vectors.shape == (3, 4, 4)
    assert qualified.shape == (3,)
    assert margins.shape == (3,)
    assert null_vectors.shape == (3, 4, 4)
    assert jnp.all(qualified)
    assert jnp.all(null_qualified)

    cartesian_chart = CoordinateChart("cartesian-flat", ("t", "x", "y", "z"))
    flat_metric = minkowski_metric(cartesian_chart)

    def constant_frame(point):
        return orthonormal_tetrad(
            flat_metric,
            jnp.eye(4),
            point,
            source_id="cartesian-coordinate-frame",
        )

    cartesian_points = jnp.asarray(((0.0, 0.0, 0.0, 0.0), (0.2, -0.3, 0.4, 0.1)))
    directions = jnp.asarray(((1.0, 0.2, 0.0, 0.0), (0.7, 0.0, -0.1, 0.3)))
    transport = jax.jit(
        lambda values, tangents: tetrad_parallel_transport_evidence(
            flat_metric,
            constant_frame,
            values,
            tangents,
            tolerance=1e-12,
            source_id="constant-cartesian-frame",
        )
    )(cartesian_points, directions)
    assert transport.residuals.shape == (2, 4, 4)
    assert jnp.allclose(transport.residuals, 0.0, atol=1e-12)
    assert jnp.all(transport.finite)
    assert jnp.all(transport.physically_valid)
    assert jnp.all(transport.qualified)
    assert jnp.all(transport.derivative_valid)
