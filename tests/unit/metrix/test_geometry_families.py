#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import comb

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_lorentzian_metric_causality_wave_operator_and_curvature():
    chart = phx.metrix.CoordinateChart("spacetime", ("t", "x", "y", "z"))
    metric = phx.metrix.minkowski_metric(chart)
    point = jnp.array([0.2, -0.3, 0.1, 0.4])

    report = phx.metrix.validate_lorentzian_metric(metric, point)

    def field(q):
        return -(q[0] ** 2) + jnp.sum(q[1:] ** 2)

    assert bool(report.valid)
    assert (
        phx.metrix.causal_character(metric, point, jnp.array([1.0, 0.0, 0.0, 0.0])) == -1
    )
    assert jnp.allclose(phx.metrix.dalembertian(field, metric, point), 8.0)
    assert jnp.max(jnp.abs(phx.metrix.riemann_tensor(metric, point))) < 1e-10
    assert jnp.allclose(
        jax.jit(lambda q: phx.metrix.dalembertian(field, metric, q))(point),
        8.0,
    )

    def future_reference(q):
        return jnp.array([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)

    orientation = phx.metrix.TimeOrientation(metric, future_reference)
    future = jnp.array([2.0, 0.1, 0.0, 0.0])
    past = -future
    zero = jnp.zeros((4,))
    assert bool(orientation.is_future_directed(point, future))
    assert bool(orientation.is_past_directed(point, past))
    assert not bool(orientation.is_future_directed(point, zero))
    assert not bool(orientation.is_past_directed(point, zero))
    assert jnp.allclose(
        phx.metrix.proper_time_rate(metric, point, future), jnp.sqrt(3.99)
    )
    batched_points = jnp.stack((point, point + 0.1))
    assert jnp.all(
        orientation.is_future_directed(
            batched_points,
            jnp.stack((future, future)),
        )
    )


def test_signed_tensor_and_levi_civita_operations_are_signature_independent():
    chart = phx.metrix.CoordinateChart("minkowski", ("t", "x", "y", "z"))
    metric = phx.metrix.minkowski_metric(chart)
    point = jnp.array([0.2, -0.3, 0.1, 0.4])
    vector = jnp.array([2.0, 3.0, -1.0, 4.0])

    lowered = phx.metrix.lower_index(
        vector,
        metric,
        point,
        tensor_type=phx.metrix.VECTOR_TENSOR,
    )
    raised = phx.metrix.raise_index(
        lowered,
        metric,
        point,
        tensor_type=phx.metrix.COVECTOR_TENSOR,
    )

    assert jnp.array_equal(lowered, jnp.array([-2.0, 3.0, -1.0, 4.0]))
    assert jnp.array_equal(raised, vector)
    assert jnp.allclose(phx.metrix.inner_product(vector, vector, metric, point), 22.0)
    assert jnp.allclose(
        phx.metrix.tensor_norm_squared(
            jnp.array([2.0, 0.0, 0.0, 0.0]),
            metric,
            phx.metrix.VECTOR_TENSOR,
            point,
        ),
        -4.0,
    )

    scalar = lambda q: q[0] ** 2 + 2.0 * q[1] ** 2
    assert jnp.allclose(
        phx.metrix.covariant_hessian(scalar, metric, point),
        jnp.diag(jnp.array([2.0, 4.0, 0.0, 0.0])),
    )
    assert jnp.allclose(
        phx.metrix.divergence(lambda q: q, metric, point),
        4.0,
    )
    assert jnp.allclose(
        phx.metrix.covariant_derivative(
            lambda q: metric(q),
            metric,
            phx.metrix.TensorType(("covariant", "covariant")),
            point,
        ),
        0.0,
    )


def test_signed_codifferential_and_hodge_square_obey_index_signs():
    chart = phx.metrix.CoordinateChart("minkowski_forms", ("t", "x", "y", "z"))
    metric = phx.metrix.minkowski_metric(chart)
    point = jnp.array([0.2, -0.3, 0.1, 0.4])

    for degree in range(chart.dimension + 1):
        coefficient_count = comb(chart.dimension, degree)
        coefficients = jnp.arange(1, coefficient_count + 1, dtype=point.dtype)
        form = phx.metrix.DifferentialForm(
            lambda q, values=coefficients: values,
            chart=chart,
            degree=degree,
        )
        expected_sign = (
            -1
            if (degree * (chart.dimension - degree) + metric.signature.index) % 2
            else 1
        )
        assert jnp.allclose(
            phx.metrix.hodge_star(
                phx.metrix.hodge_star(form, metric),
                metric,
            )(point),
            expected_sign * coefficients,
        )

    covector = phx.metrix.DifferentialForm(
        lambda q: q,
        chart=chart,
        degree=1,
    )
    scalar = phx.metrix.DifferentialForm(
        lambda q: -(q[0] ** 2) + jnp.sum(q[1:] ** 2),
        chart=chart,
        degree=0,
    )
    two_form = phx.metrix.DifferentialForm(
        lambda q: jnp.array(
            [q[0] * q[1], q[0] * q[2], q[0] * q[3], q[1], q[2], q[3]]
        ),
        chart=chart,
        degree=2,
    )

    assert jnp.allclose(phx.metrix.codifferential(covector, metric)(point), -2.0)
    assert jnp.allclose(
        phx.metrix.hodge_laplacian(scalar, metric)(point),
        jnp.array([-8.0]),
    )
    assert jnp.allclose(
        phx.metrix.codifferential(
            phx.metrix.codifferential(two_form, metric),
            metric,
        )(point),
        jnp.array([0.0]),
        atol=1e-10,
    )

def test_signed_metric_constructors_validate_declared_signatures():
    chart = phx.metrix.CoordinateChart(
        "spherical_spacetime",
        ("t", "r", "theta", "phi"),
    )
    point = jnp.array([0.3, 4.0, 1.1, -0.2])

    def scale_factor(time):
        return jnp.exp(0.1 * time)

    def lapse(q):
        return 1.5 + 0.0 * q[0]

    def shift(q):
        return jnp.array([0.1, -0.2, 0.05], dtype=q.dtype)

    def spatial_metric(q):
        return jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=q.dtype))

    metrics = (
        phx.metrix.flrw_metric(scale_factor, chart=chart),
        phx.metrix.adm_metric(
            lapse,
            shift,
            spatial_metric,
            chart=chart,
        ),
        phx.metrix.schwarzschild_metric(1.0, chart=chart),
    )
    assert all(
        bool(phx.metrix.validate_lorentzian_metric(metric, point).valid)
        for metric in metrics
    )

    reversed_metric = phx.metrix.minkowski_metric(
        chart,
        convention="mostly_minus",
    )
    reversed_report = phx.metrix.validate_lorentzian_metric(
        reversed_metric,
        jnp.stack((point, point + 0.1)),
    )
    assert bool(reversed_report.valid)
    assert jnp.all(reversed_report.observed_positive == 1)
    assert jnp.all(reversed_report.observed_negative == 3)

    plane = phx.metrix.CoordinateChart("signed_plane", ("x", "y"))
    mismatched = phx.metrix.SemiRiemannianMetric(
        lambda q: jnp.diag(jnp.array([-1.0, 1.0], dtype=q.dtype)),
        chart=plane,
        signature=phx.metrix.MetricSignature(2, 0),
    )
    assert not bool(
        phx.metrix.validate_semi_riemannian_metric(
            mismatched,
            jnp.zeros((2,)),
        ).valid
    )


def test_differential_forms_obey_nilpotency_hodge_and_pullback_laws():
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(2), chart=chart)
    one_form = phx.metrix.DifferentialForm(
        lambda q: jnp.array([-q[1], q[0]]),
        chart=chart,
        degree=1,
    )
    point = jnp.array([0.2, 0.3])
    exterior = phx.metrix.exterior_derivative(one_form)

    assert jnp.allclose(exterior(point), jnp.array([2.0]))
    assert jnp.allclose(jax.jit(exterior)(point), jnp.array([2.0]))
    scalar = phx.metrix.DifferentialForm(
        lambda q: q[0] ** 2 + q[1] ** 2,
        chart=chart,
        degree=0,
    )
    mismatched_metric = phx.metrix.RiemannianMetric(
        lambda q: jnp.eye(2),
        chart=phx.metrix.CoordinateChart("other-plane", ("u", "v")),
    )
    with pytest.raises(ValueError, match="charts do not match"):
        phx.metrix.hodge_laplacian(scalar, mismatched_metric)
    assert jnp.allclose(
        phx.metrix.exterior_derivative(phx.metrix.exterior_derivative(scalar))(point),
        jnp.array([0.0]),
    )
    assert jnp.allclose(
        phx.metrix.hodge_star(phx.metrix.hodge_star(one_form, metric), metric)(point),
        -one_form(point),
    )

    target = phx.metrix.CoordinateChart("target", ("u", "v"))
    map = phx.metrix.DifferentiableMap(
        chart, target, lambda q: jnp.array([2.0 * q[0], 3.0 * q[1]])
    )
    area = phx.metrix.DifferentialForm(lambda q: jnp.array([1.0]), chart=target, degree=2)
    assert jnp.allclose(phx.metrix.pullback_form(area, map)(point), jnp.array([6.0]))
    second_one_form = phx.metrix.DifferentialForm(
        lambda q: jnp.array([q[0], 2.0 * q[1]]),
        chart=chart,
        degree=1,
    )
    assert jnp.allclose(
        phx.metrix.wedge(one_form, second_one_form)(point),
        -phx.metrix.wedge(second_one_form, one_form)(point),
    )
    target_one_form = phx.metrix.DifferentialForm(
        lambda q: jnp.array([-q[1], q[0]]),
        chart=target,
        degree=1,
    )
    pulled_one_form = phx.metrix.pullback_form(target_one_form, map)
    assert jnp.allclose(
        phx.metrix.exterior_derivative(pulled_one_form)(point),
        phx.metrix.pullback_form(
            phx.metrix.exterior_derivative(target_one_form),
            map,
        )(point),
    )
    assert jnp.allclose(
        phx.metrix.hodge_laplacian(scalar, metric)(point),
        jnp.array([-4.0]),
    )


def test_lie_group_and_poisson_structures_satisfy_defining_laws():
    group = phx.metrix.SpecialOrthogonalGroup(3)
    algebra = jnp.array([0.2, -0.1, 0.3])
    element = group.exp(group.hat(algebra))

    assert bool(group.contains(element))
    assert jnp.allclose(group.vee(group.log(element)), algebra, atol=1e-10)
    assert jnp.allclose(
        group.compose(element, group.inverse(element)),
        group.identity(),
        atol=1e-10,
    )

    chart = phx.metrix.CoordinateChart("phase", ("q", "p"))
    symplectic = phx.metrix.canonical_symplectic_form(chart)
    poisson = phx.metrix.symplectic_to_poisson(symplectic)
    point = jnp.array([0.4, -0.2])

    def left(z):
        return z[0]

    def right(z):
        return z[1]

    assert bool(phx.metrix.validate_symplectic_form(symplectic, point).valid)
    assert bool(phx.metrix.validate_poisson_structure(poisson, point).valid)
    assert jnp.allclose(phx.metrix.poisson_bracket(left, right, poisson, point), 1.0)
    assert jnp.allclose(
        jax.jit(
            lambda q: phx.metrix.hamiltonian_vector_field(
                lambda z: 0.5 * jnp.dot(z, z), poisson, q
            )
        )(point),
        jnp.array([-0.2, -0.4]),
    )
    assert jnp.allclose(phx.metrix.poisson_bracket(left, left, poisson, point), 0.0)
    assert jnp.max(jnp.abs(phx.metrix.casimir_residual(left, poisson, point))) > 0.0


def test_lie_group_logs_cover_identity_and_reject_the_pi_cut_locus():
    rotations = phx.metrix.SpecialOrthogonalGroup(3)
    tiny = jnp.array([1e-9, 0.0, 0.0])
    assert jnp.allclose(
        rotations.vee(rotations.log(rotations.exp(rotations.hat(tiny)))),
        tiny,
        atol=1e-14,
    )

    pi_rotation = jnp.diag(jnp.array([1.0, -1.0, -1.0]))
    with pytest.raises(Exception, match="ill-conditioned"):
        rotations.log(pi_rotation)

    rigid_motions = phx.metrix.SpecialEuclideanGroup(3)
    algebra = jnp.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1])
    assert jnp.allclose(
        rigid_motions.vee(
            rigid_motions.log(rigid_motions.exp(rigid_motions.hat(algebra)))
        ),
        algebra,
        atol=1e-10,
    )


def test_lie_group_batches_preserve_exp_log_and_inverse_laws():
    rotations = phx.metrix.SpecialOrthogonalGroup(3)
    rotation_coordinates = jnp.array([[0.2, -0.1, 0.3], [-0.15, 0.25, 0.05]])
    rotation_elements = rotations.exp(rotations.hat(rotation_coordinates))

    assert bool(rotations.contains(rotation_elements))
    assert jnp.allclose(
        rotations.vee(rotations.log(rotation_elements)),
        rotation_coordinates,
        atol=1e-10,
    )
    assert jnp.allclose(
        rotations.compose(rotation_elements, rotations.inverse(rotation_elements)),
        jnp.broadcast_to(rotations.identity(), rotation_elements.shape),
        atol=1e-10,
    )

    rigid_motions = phx.metrix.SpecialEuclideanGroup(3)
    motion_coordinates = jnp.array(
        [
            [0.2, -0.1, 0.3, 0.1, -0.2, 0.05],
            [-0.3, 0.4, 0.1, -0.15, 0.05, 0.2],
        ]
    )
    motion_elements = rigid_motions.exp(rigid_motions.hat(motion_coordinates))

    assert bool(rigid_motions.contains(motion_elements))
    assert jnp.allclose(
        rigid_motions.vee(rigid_motions.log(motion_elements)),
        motion_coordinates,
        atol=1e-10,
    )
    assert jnp.allclose(
        rigid_motions.compose(
            motion_elements,
            rigid_motions.inverse(motion_elements),
        ),
        jnp.broadcast_to(rigid_motions.identity(), motion_elements.shape),
        atol=1e-10,
    )
    assert jnp.allclose(
        jax.jit(
            lambda coordinates: rigid_motions.vee(
                rigid_motions.log(rigid_motions.exp(rigid_motions.hat(coordinates)))
            )
        )(motion_coordinates),
        motion_coordinates,
        atol=1e-10,
    )


def test_nonconstant_poisson_structure_obeys_jacobi_and_detects_failure():
    chart = phx.metrix.CoordinateChart("lie_poisson", ("x", "y", "z"))

    def radial_bivector(q):
        x, y, z = q
        return jnp.array([[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]])

    poisson = phx.metrix.PoissonStructure(radial_bivector, chart=chart)
    point = jnp.array([0.2, 0.3, 0.4])

    def first(q):
        return q[0]

    def second(q):
        return q[1]

    def third(q):
        return q[2]

    def bracket(left, right):
        return lambda q: phx.metrix.poisson_bracket(left, right, poisson, q)

    jacobi = (
        phx.metrix.poisson_bracket(first, bracket(second, third), poisson, point)
        + phx.metrix.poisson_bracket(second, bracket(third, first), poisson, point)
        + phx.metrix.poisson_bracket(third, bracket(first, second), poisson, point)
    )

    def casimir(q):
        return jnp.dot(q, q)

    assert bool(phx.metrix.validate_poisson_structure(poisson, point).valid)
    assert jnp.allclose(jacobi, 0.0, atol=1e-12)
    assert jnp.allclose(phx.metrix.casimir_residual(casimir, poisson, point), 0.0)
    assert jnp.allclose(
        phx.metrix.poisson_bracket(first, second, poisson, point),
        -phx.metrix.poisson_bracket(second, first, poisson, point),
    )

    def invalid_bivector(q):
        x, y, z = q
        return jnp.array([[0.0, x, -z], [-x, 0.0, y], [z, -y, 0.0]])

    invalid = phx.metrix.PoissonStructure(invalid_bivector, chart=chart)
    assert not bool(phx.metrix.validate_poisson_structure(invalid, point).valid)


def test_heisenberg_horizontal_cometric_is_step_two_bracket_generating():
    chart = phx.metrix.CoordinateChart("heisenberg", ("x", "y", "z"))
    cometric = phx.metrix.HorizontalCometric(
        lambda q: jnp.array([[1.0, 0.0], [0.0, 1.0], [-0.5 * q[1], 0.5 * q[0]]]),
        chart,
        2,
    )
    point = jnp.array([0.2, 0.3, -0.1])

    def field(q):
        return q[0] ** 2 + q[1] ** 2

    report = phx.metrix.validate_horizontal_cometric(
        cometric,
        point,
        require_step_two_bracket_generating=True,
    )

    assert bool(report.valid)
    assert report.step_two_rank == 3
    assert jnp.allclose(phx.metrix.sub_laplacian(field, cometric, point), 4.0)
    assert jnp.allclose(
        jax.jit(lambda q: phx.metrix.sub_laplacian(field, cometric, q))(point),
        4.0,
    )
    assert jnp.allclose(
        phx.metrix.horizontal_hamiltonian(jnp.array([1.0, 2.0, 0.0]), cometric, point),
        2.5,
    )
