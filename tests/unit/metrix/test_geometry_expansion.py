#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_immersion_pullback_map_geometry_and_density_calculus():
    source = phx.metrix.CoordinateChart("plane", ("u", "v"))
    target = phx.metrix.CoordinateChart("space", ("x", "y", "z"))
    immersion = phx.metrix.Immersion(
        source,
        target,
        lambda q: jnp.asarray([q[0], q[1], q[0] * q[1]]),
    )
    point = jnp.asarray([0.2, -0.3])
    report = phx.metrix.validate_immersion(immersion, point)
    ambient = phx.metrix.euclidean_metric(target)
    induced = phx.metrix.pullback_metric(ambient, immersion)
    geometry = phx.metrix.RiemannianMapGeometry(immersion, induced, ambient)

    assert bool(report.valid)
    assert jnp.allclose(
        induced(point),
        immersion.jacobian(point).T @ immersion.jacobian(point),
    )
    assert jnp.allclose(geometry.isometry_residual(point), 0.0, atol=1e-6)
    assert geometry.energy_density(point) == pytest.approx(1.0)

    density = phx.metrix.VolumeDensity(
        lambda q: jnp.exp(q[0]),
        chart=source,
        log_coefficient=lambda q: q[0],
    )
    assert bool(phx.metrix.validate_volume_density(density, point).valid)

    coefficients = jnp.zeros((2, 2, 2)).at[0, 0, 0].set(1.0)
    connection = phx.metrix.CallableAffineConnection(
        lambda q: coefficients,
        chart=source,
    )
    derivative = phx.metrix.affine_covariant_derivative(
        lambda q: jnp.exp(q[0]),
        connection,
        phx.metrix.TensorType(density_weight=1.0),
        jnp.zeros((2,)),
    )
    assert jnp.allclose(derivative, 0.0)


def test_weighted_measure_boundary_and_numerical_geodesic():
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.asarray([4.0, 1.0]),
        chart=chart,
    )
    measure = phx.metrix.WeightedRiemannianMeasure(
        metric,
        lambda q: -0.5 * jnp.dot(q, q),
    )
    point = jnp.asarray([0.3, -0.2])
    value = measure.laplacian(lambda q: q[0] ** 2 + q[1] ** 2, point)
    expected = 2.0 / 4.0 + 2.0 - (point[0] ** 2 / 2.0 + 2.0 * point[1] ** 2)
    assert jnp.allclose(value, expected)

    boundary = phx.metrix.RiemannianHypersurface(
        metric,
        lambda q: jnp.asarray([1.0, 0.0]),
    )
    assert jnp.allclose(boundary.unit_conormal(point), jnp.asarray([2.0, 0.0]))
    assert jnp.allclose(boundary.unit_normal(point), jnp.asarray([0.5, 0.0]))

    euclidean = phx.metrix.euclidean_metric(chart)
    result = phx.metrix.integrate_metric_geodesic(
        euclidean,
        point,
        jnp.asarray([0.4, -0.1]),
        steps=4,
    )
    assert jnp.allclose(result.endpoint, point + jnp.asarray([0.4, -0.1]))
    assert jnp.allclose(result.final_velocity, jnp.asarray([0.4, -0.1]))


def test_exact_geodesic_manifolds_and_intrinsic_statistics():
    sphere = phx.metrix.SphereManifold(3)
    point = jnp.asarray([1.0, 0.0, 0.0])
    tangent = jnp.asarray([0.0, 0.3, 0.0])
    destination = sphere.exp(point, tangent)
    assert jnp.allclose(sphere.log(point, destination), tangent, atol=1e-6)
    assert sphere.squared_distance(point, destination) == pytest.approx(0.09)

    samples = jnp.asarray([[1.0, 0.0, 0.0], [jnp.cos(0.2), jnp.sin(0.2), 0.0]])
    mean = phx.metrix.frechet_mean(sphere, samples, iterations=8)
    assert jnp.allclose(
        mean.point, jnp.asarray([jnp.cos(0.1), jnp.sin(0.1), 0.0]), atol=1e-5
    )

    spd = phx.metrix.AffineInvariantSPDManifold(2)
    matrix = jnp.eye(2)
    step = jnp.diag(jnp.asarray([0.2, -0.1]))
    target = spd.exp(matrix, step)
    assert jnp.allclose(spd.log(matrix, target), step, atol=1e-6)


def test_complex_projective_unitary_and_hpd_geometry():
    projective = phx.metrix.ComplexProjectiveManifold(2)
    point = jnp.asarray([1.0 + 0.0j, 0.0j])
    tangent = jnp.asarray([0.0j, 0.25 + 0.1j])
    destination = projective.exp(point, tangent)
    phase = jnp.exp(0.7j)
    assert jnp.allclose(
        projective.squared_distance(point, destination),
        projective.squared_distance(phase * point, destination),
    )
    assert jnp.allclose(projective.log(point, destination), tangent, atol=1e-6)
    parameters = {"psi": point}
    parameter_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['psi']": projective},
    )
    gradient = {"psi": tangent}
    update = parameter_geometry.retract(
        parameters,
        parameter_geometry.egrad_to_rgrad(parameters, gradient),
    )
    assert bool(projective.contains(update["psi"]))

    unitary = phx.metrix.UnitaryGroup(2)
    algebra = jnp.asarray([[0.0j, 0.2j], [0.2j, 0.0j]])
    element = unitary.exp(algebra)
    assert bool(unitary.contains(element))
    assert jnp.allclose(unitary.exp(unitary.log(element)), element, atol=1e-5)

    special = phx.metrix.SpecialUnitaryGroup(2)
    special_element = special.exp(algebra)
    assert bool(special.contains(special_element))

    hpd = phx.metrix.AffineInvariantHPDManifold(2)
    matrix = jnp.asarray([[2.0 + 0.0j, 0.2j], [-0.2j, 1.2 + 0.0j]])
    step = jnp.asarray([[0.1 + 0.0j, 0.03j], [-0.03j, -0.05 + 0.0j]])
    destination = hpd.exp(matrix, step)
    assert bool(hpd.contains(destination))
    assert jnp.allclose(hpd.log(matrix, destination), step, atol=2e-5)


def test_flat_kahler_local_calabi_yau_and_wirtinger_calculus():
    chart = phx.metrix.CoordinateChart("complex-line", ("x", "y"))
    convention = phx.metrix.ComplexCoordinateConvention(chart)
    complex_structure = phx.metrix.standard_complex_structure(convention)
    metric = phx.metrix.euclidean_metric(chart)
    hermitian = phx.metrix.HermitianStructure(metric, complex_structure)
    kahler = phx.metrix.KahlerStructure(hermitian)
    points = jnp.asarray([[0.1, -0.2], [0.4, 0.3]])

    assert bool(phx.metrix.validate_kahler_structure(kahler, points).valid)
    derivative, conjugate_derivative = phx.metrix.wirtinger_derivatives(
        lambda q: q[0] + 1j * q[1],
        convention,
        jnp.asarray([0.2, -0.1]),
    )
    assert jnp.allclose(derivative, jnp.asarray([1.0]))
    assert jnp.allclose(conjugate_derivative, 0.0)

    volume = phx.metrix.DifferentialForm(
        lambda q: jnp.asarray([1.0 + 0.0j, 1.0j]),
        chart=chart,
        degree=1,
    )
    structure = phx.metrix.LocalCalabiYauStructure(kahler, volume)
    report = phx.metrix.validate_local_calabi_yau_structure(structure, points)
    assert bool(report.valid)
    assert report.maximum_ricci_residual < 1e-8


def test_hessian_bundle_and_atlas_geometry():
    chart = phx.metrix.CoordinateChart("source", ("x", "y"))
    hessian = phx.metrix.HessianGeometry(
        lambda q: 0.5 * jnp.dot(q, q),
        chart=chart,
    )
    point = jnp.asarray([0.2, -0.1])
    assert bool(phx.metrix.validate_hessian_geometry(hessian, point).valid)
    assert jnp.allclose(
        hessian.bregman_divergence(jnp.asarray([1.0, 0.0]), jnp.zeros((2,))),
        0.5,
    )

    connection = phx.metrix.VectorBundleConnection(
        lambda q: jnp.zeros((1, 1, 2), dtype=complex),
        chart=chart,
        fiber_dimension=1,
    )
    gauge = lambda q: jnp.asarray([[jnp.exp(1j * q[0])]])
    assert jnp.allclose(
        phx.metrix.gauge_curvature_residual(connection, gauge, point),
        0.0,
        atol=1e-6,
    )

    target = phx.metrix.CoordinateChart("target", ("u", "v"))
    transition = phx.metrix.ChartTransition(
        chart,
        target,
        lambda q: jnp.asarray([2.0 * q[0], 3.0 * q[1]]),
        inverse=lambda q: jnp.asarray([q[0] / 2.0, q[1] / 3.0]),
    )
    atlas = phx.metrix.CoordinateAtlas((chart, target), (transition,))
    report = phx.metrix.validate_coordinate_atlas(atlas, (point,))
    assert bool(report.valid)
    field = phx.metrix.PatchwiseScalarField(
        atlas,
        (lambda q: q[0] + q[1], lambda q: q[0] / 2.0 + q[1] / 3.0),
    )
    assert jnp.allclose(field.transition_residual(0, 1, point), 0.0)


def test_sub_riemannian_control_metric_and_hamiltonian_rhs():
    chart = phx.metrix.CoordinateChart("control", ("x", "y"))
    cometric = phx.metrix.HorizontalCometric(
        lambda q: jnp.eye(2),
        chart,
        2,
        control_metric=lambda q: jnp.diag(jnp.asarray([2.0, 4.0])),
    )
    point = jnp.asarray([0.0, 0.0])
    assert jnp.allclose(cometric(point), jnp.diag(jnp.asarray([0.5, 0.25])))
    state = jnp.asarray([0.0, 0.0, 2.0, 4.0])
    rhs = phx.metrix.sub_riemannian_hamiltonian_rhs(cometric, state)
    assert jnp.allclose(rhs, jnp.asarray([1.0, 1.0, 0.0, 0.0]))
    compiled = jax.jit(
        lambda value: phx.metrix.sub_riemannian_hamiltonian_rhs(cometric, value)
    )
    assert jnp.all(jnp.isfinite(compiled(state)))
