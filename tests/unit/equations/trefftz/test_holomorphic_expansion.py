#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_exact_disk_trace_matches_entire_finite_fourier_boundary():
    plan = phx.equations.DiskHolomorphicTracePlan(
        3,
        center=0.2 - 0.1j,
        radius=1.4,
    )
    cosine = jnp.asarray([1.0, 0.4, -0.2, 0.15])
    sine = jnp.asarray([0.0, -0.3, 0.25, 0.1])
    lift = plan.lift(cosine, sine)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 97, endpoint=False)
    points = plan.center + plan.radius * jnp.exp(1j * angles)
    expected = cosine[0] + sum(
        cosine[mode] * jnp.cos(mode * angles) + sine[mode] * jnp.sin(mode * angles)
        for mode in range(1, 4)
    )
    actual = jax.vmap(lambda point: jnp.real(lift(point)[0]))(points)
    assert jnp.allclose(actual, expected, atol=2e-12)
    certificate = lift.trace_certificate()
    assert certificate.evidence_kind == "continuous-subspace-exact"
    assert certificate.residual_bound == 0.0


def test_contour_period_and_fixed_meromorphic_domain_evidence():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 128, endpoint=False)
    nodes = jnp.exp(1j * angles)
    dz = 1j * nodes * (2.0 * jnp.pi / angles.size)
    period = phx.equations.holomorphic_period_functional(nodes, dz)
    assert jnp.linalg.norm(period.assemble_row(frame)) < 2e-12

    poles = phx.equations.PoleSet(jnp.asarray([2.0 + 0.0j, -2.5 + 0.4j]), (1, 2))
    meromorphic_frame = phx.equations.MeromorphicLinearFrame(2, poles)
    functionals = (
        phx.equations.HolomorphicPointFunctional.value(0.0),
        phx.equations.HolomorphicPointFunctional.value(0.5),
    )
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        meromorphic_frame,
        functionals,
    ).prepare()
    coefficient_map = operator.affine_map(jnp.asarray([0.0, 0.25]))
    potential = phx.equations.ConstrainedMeromorphicPotential(coefficient_map)
    domain_certificate = potential.certify_on_disk(
        center=0.0j,
        radius=1.0,
        required_clearance=0.2,
    )
    assert domain_certificate.domain_id
    assert potential.meromorphic_certificate().pole_set_id == poles.pole_set_id
    assert (
        jnp.linalg.norm(coefficient_map.residual(potential.free_coordinates))
        <= coefficient_map.evidence.tolerance
    )


def test_meromorphic_variable_projection_recovers_linear_coefficients():
    coordinates = jnp.linspace(-0.8, 0.8, 21).astype(jnp.complex128) + 0.1j
    pole = 1.7 + 0.4j
    observations = (
        0.3 - 0.2j + (0.6 + 0.1j) * coordinates + (1.2 - 0.5j) / (coordinates - pole)
    )
    plan = phx.equations.MeromorphicVariableProjectionPlan(
        coordinates,
        observations,
        1,
        (1,),
    )
    parameters = jnp.asarray([jnp.real(pole), jnp.imag(pole)])
    solution, residual, rank, singular_values = plan.problem().linear_solution(parameters)
    assert solution.shape == (6,)
    assert jnp.linalg.norm(residual) < 2e-11
    assert rank == 6
    assert jnp.all(singular_values > 0.0)
    continuation = plan.continuation_problem(observations)
    stationarity = continuation.residual(parameters, jnp.asarray(0.5))
    assert jnp.linalg.norm(stationarity) < 2e-9


def test_multi_index_resource_guard_fails_before_expansion():
    with pytest.raises(ValueError, match="exceeds"):
        phx.equations.HolomorphicMultiIndexSet.total_degree(
            4,
            5,
            maximum_count=10,
        )


def test_holomorphic_mlp_multijets_match_complex_ad():
    model = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(4,),
        key=jr.key(0),
    )
    coordinate = jnp.asarray([0.2 - 0.1j, -0.3 + 0.25j])
    index_set = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    jet = model.multi_jet(coordinate, index_set)
    jacobian = jax.jacfwd(model, holomorphic=True)(coordinate)
    hessian = jax.jacfwd(
        jax.jacfwd(model, holomorphic=True),
        holomorphic=True,
    )(coordinate)
    assert jnp.allclose(jet.value, model(coordinate), atol=2e-12)
    assert jnp.allclose(jet.derivative((1, 0)), jacobian[:, 0], atol=2e-12)
    assert jnp.allclose(jet.derivative((0, 1)), jacobian[:, 1], atol=2e-12)
    assert jnp.allclose(jet.derivative((1, 1)), hessian[:, 0, 1], atol=3e-11)


def test_product_multijets_match_complex_ad():
    first = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(3,),
        key=jr.key(1),
    )
    second = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(3,),
        key=jr.key(2),
    )
    product = phx.equations.HolomorphicProductPotential(
        (first, second),
        latent_rank=2,
        branches=1,
    )
    coordinate = jnp.asarray([0.1 + 0.2j, -0.2 + 0.1j])
    index_set = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    jet = product.multi_jet(coordinate, index_set)
    hessian = jax.jacfwd(
        jax.jacfwd(product, holomorphic=True),
        holomorphic=True,
    )(coordinate)
    assert jnp.allclose(jet.value, product(coordinate), atol=2e-12)
    assert jnp.allclose(jet.derivative((1, 1)), hessian[:, 0, 1], atol=4e-11)


def test_pluriharmonic_derivatives_and_kahler_gauge_are_invariant():
    provider = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=1,
        hidden_sizes=(4,),
        key=jr.key(3),
    )
    potential = phx.equations.PluriharmonicPotential(provider)
    point = jnp.asarray([0.2, -0.3, 0.1, 0.25])
    assert jnp.allclose(potential.gradient(point), jax.grad(potential)(point), atol=3e-11)
    assert jnp.allclose(
        potential.hessian(point), jax.hessian(potential)(point), atol=4e-10
    )
    assert jnp.allclose(potential.laplacian(point), 0.0, atol=4e-10)

    chart = phx.metrix.CoordinateChart("complex-two-space", ("x0", "x1", "y0", "y1"))
    convention = phx.metrix.ComplexCoordinateConvention(chart)
    base = phx.metrix.KahlerPotentialGeometry(
        phx.metrix.euclidean_metric(chart),
        convention,
        lambda coordinates: jnp.asarray(0.0),
    )
    gauge = phx.metrix.KahlerHolomorphicGauge(base, provider)
    report = gauge.invariance_report(jnp.stack((point, -0.5 * point)), tolerance=1e-9)
    assert report.valid
    assert report.maximum_complex_hessian_change < 1e-9
