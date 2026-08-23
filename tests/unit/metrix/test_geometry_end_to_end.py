#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_weighted_density_reference_operators_and_normalization():
    domain = phx.domain.HyperRectangle([-2.0, -2.0], [2.0, 2.0], label="x")
    chart = phx.metrix.CoordinateChart("weighted-plane", ("x", "y"))
    measure = phx.metrix.WeightedRiemannianMeasure(
        phx.metrix.euclidean_metric(chart), lambda point: -0.5 * jnp.dot(point, point)
    )
    field = domain.Function("x")(lambda point: jnp.dot(point, point))
    laplacian = phx.operators.weighted_laplacian(field, measure, var="x")
    point = jnp.asarray([0.3, -0.2])
    assert jnp.allclose(laplacian.func(point), 4.0 - 2.0 * jnp.dot(point, point))

    referenced = phx.domain.ReferencedDensityField(
        domain.Function("x")(lambda point: jnp.asarray(1.0)),
        reference="weighted-riemannian-volume",
        state_var="x",
        measure=measure,
    )
    coordinate_value = referenced.to_coordinate_value(1.0, point)
    assert jnp.allclose(referenced.from_coordinate_value(coordinate_value, point), 1.0)

    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    normalization = phx.integration.normalize_metric_measure(
        measure, points, jnp.asarray([0.5, 0.5])
    )
    assert bool(normalization.valid)
    assert normalization.mass > 0.0


def test_information_geometry_operator_hessian_and_pullback():
    chart = phx.metrix.CoordinateChart("hessian", ("x", "y"))
    hessian = phx.metrix.HessianGeometry(
        lambda point: 0.5 * jnp.dot(point, point), chart=chart
    )
    point = jnp.asarray([0.2, -0.1])
    operator = hessian.information_operator(point, damping=0.1)
    assert jnp.allclose(operator.mv(jnp.asarray([1.0, 2.0])), jnp.asarray([1.1, 2.2]))

    target = phx.metrix.InformationMetricOperator(
        lambda vector: 2.0 * vector,
        jnp.zeros((2,)),
        metric_id="target-information",
    )
    pulled = phx.metrix.pulled_back_information_operator(
        lambda parameters: jnp.asarray([parameters[0], 2.0 * parameters[0]]),
        jnp.asarray([0.3]),
        target,
    )
    assert jnp.allclose(pulled.mv(jnp.asarray([1.0])), jnp.asarray([10.0]))

    family = phx.uq.BernoulliFamily()
    geometry = phx.uq.ExponentialFamilyInformationGeometry(family)
    natural = family.natural(jnp.asarray([0.0]))
    assert jnp.allclose(
        geometry.information_operator(natural).materialize(), jnp.asarray([[0.25]])
    )
    assert jnp.allclose(
        geometry.natural_gradient(natural, jnp.asarray([1.0])), jnp.asarray([4.0])
    )


def test_geodesic_interpolant_manifold_metric_and_matrix_cost():
    sphere = phx.metrix.SphereManifold(3)
    transport = phx.transport.ManifoldTransportGeometry(sphere)
    source = jnp.asarray([1.0, 0.0, 0.0])
    target = jnp.asarray([0.0, 1.0, 0.0])
    midpoint = transport.interpolant.evaluate(0.5, source, target)
    expected = jnp.asarray([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
    assert bool(midpoint.valid)
    assert jnp.allclose(midpoint.state, expected)
    assert jnp.allclose(jnp.dot(midpoint.state, midpoint.conditional_velocity), 0.0)
    assert (
        transport.metric(
            midpoint.state,
            jnp.zeros_like(midpoint.conditional_velocity),
            midpoint.conditional_velocity,
        )
        > 0.0
    )
    assert bool(transport.state_geometry.contains(midpoint.state))

    spd = phx.metrix.AffineInvariantSPDManifold(2)
    cost = phx.transport.IntrinsicSquaredDistanceCost(spd)
    points = jnp.stack((jnp.eye(2), 2.0 * jnp.eye(2)))
    matrix = cost.matrix(points, points)
    assert matrix.shape == (2, 2)
    assert jnp.allclose(jnp.diag(matrix), 0.0)


def test_projective_atlas_patchwise_metric_and_integration():
    projective = phx.geometry.complex.ComplexProjectiveAtlas(1)
    point = jnp.asarray([2.0, 0.0])
    mapped = projective.cover.atlas.transition(0, 1)(point)
    assert jnp.allclose(mapped, jnp.asarray([0.5, 0.0]))

    metrics = phx.metrix.PatchwiseMetric(
        projective.cover, tuple(projective.metric(index) for index in range(2))
    )
    assert metrics.transition_residual(0, 1, point) < 1e-8

    target = phx.integration.AtlasIntegrationTarget(
        projective.cover,
        (
            phx.integration.AtlasPatchQuadrature(
                0,
                jnp.asarray([[0.0, 0.0]]),
                jnp.asarray([1.0]),
                ownership_weights=jnp.asarray([1.0]),
                patch_id="origin",
            ),
            phx.integration.AtlasPatchQuadrature(
                1,
                jnp.asarray([[0.0, 0.0]]),
                jnp.asarray([1.0]),
                ownership_weights=jnp.asarray([0.0]),
                patch_id="infinity",
            ),
        ),
        target_id="CP1-point-cover",
    )
    result = phx.integration.integrate_atlas_scalar(
        target, (lambda points: jnp.ones(points.shape[0]),) * 2
    )
    assert bool(result.valid)
    assert jnp.allclose(result.value, 1.0)


def test_dolbeault_chern_berry_and_projective_hypersurface():
    chart = phx.metrix.CoordinateChart("complex-plane", ("x0", "x1", "y0", "y1"))
    convention = phx.metrix.ComplexCoordinateConvention(chart)
    scalar = phx.metrix.BigradedForm(
        lambda point: jnp.prod(convention.to_complex(point)),
        convention=convention,
        bidegree=(0, 0),
    )
    point = jnp.asarray([0.2, -0.1, 0.3, 0.4])
    assert jnp.allclose(phx.metrix.partial_bar(scalar)(point), 0.0)
    assert jnp.allclose(phx.metrix.partial(phx.metrix.partial(scalar))(point), 0.0)

    line_chart = phx.metrix.CoordinateChart("line", ("x", "y"))
    line_convention = phx.metrix.ComplexCoordinateConvention(line_chart)
    frame = phx.metrix.HolomorphicBundleFrame(
        line_convention,
        lambda coordinates: jnp.asarray([[jnp.exp(jnp.dot(coordinates, coordinates))]]),
        1,
        frame_id="positive-line",
    )
    curvature = phx.metrix.ChernConnection(frame).curvature(jnp.asarray([0.1, -0.2]))
    assert curvature.shape == (1, 1, 1, 1)
    assert jnp.all(jnp.isfinite(curvature))

    qgt = phx.operators.quantum.quantum_geometric_tensor(
        lambda parameters: jnp.asarray(
            [jnp.cos(parameters[0] / 2.0), jnp.sin(parameters[0] / 2.0)]
        ).astype(complex),
        jnp.asarray([0.4]),
    )
    assert bool(qgt.valid)
    assert jnp.allclose(qgt.metric, jnp.asarray([[0.25]]))

    fermat = phx.geometry.complex.fermat_hypersurface(1)
    root = jnp.asarray([0.0, 1.0])
    assert fermat.residual(0, root) < 1e-8
    assert fermat.local_smoothness_margin(0, root) > 0.0


def test_flat_torus_kahler_potential_and_local_su_structure():
    torus = phx.geometry.complex.FlatComplexTorus(1)
    geometry = phx.metrix.KahlerPotentialGeometry(
        torus.metric, torus.convention, lambda point: jnp.asarray(0.0)
    )
    point = jnp.asarray([0.2, -0.3])
    assert geometry.positivity_margin(point) > 0.0
    assert jnp.allclose(
        geometry.monge_ampere_residual(lambda coordinates: jnp.asarray(0.0), point),
        0.0,
    )

    volume = phx.metrix.DifferentialForm(
        lambda coordinates: jnp.asarray([1.0 + 0.0j, 1.0j]),
        chart=torus.convention.chart,
        degree=1,
    )
    structure = phx.metrix.LocalSUNStructure(
        torus.kahler,
        volume,
        volume_bidegree=(1, 0),
    )
    report = phx.metrix.validate_local_su_structure(structure, point)
    assert bool(report.valid)
