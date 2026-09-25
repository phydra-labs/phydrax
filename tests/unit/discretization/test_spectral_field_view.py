#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import DiscreteFieldFunctionView, FieldQueryStatus
from phydrax.discretization.spectral import prepare_spectral_field_reconstruction


jax.config.update("jax_enable_x64", True)

D = phx.discretization


def _space(*pairs):
    plans = tuple(plan for plan, _ in pairs)
    domains = tuple(domain for _, domain in pairs)
    names = ("x", "y", "z")[: len(plans)]
    return D.TensorSpectralPlan(plans, axis_names=names, field_name="u").prepare(domains)


def _view(reconstruction, coefficients):
    domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return DiscreteFieldFunctionView(reconstruction, coefficients, domain, variable="x")


_FAMILIES = {
    "fourier": (D.FourierBasisPlan(12), D.AxisDomain.periodic(0.25, 1.75)),
    "sine": (D.SineBasisPlan(10), D.AxisDomain.interval(0.5, 2.0)),
    "cosine": (D.CosineBasisPlan(9), D.AxisDomain.interval(-1.0, 0.5)),
    "chebyshev": (D.ChebyshevBasisPlan(9), D.AxisDomain.interval(-0.5, 1.5)),
    "legendre": (D.LegendreBasisPlan(8), D.AxisDomain.interval(0.0, 3.0)),
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_point_synthesis_equals_grid_reconstruction_and_derivatives(family):
    space = _space(_FAMILIES[family])
    rng = np.random.default_rng(3)
    # A generic (non-Hermitian) modal state exercises every mode, including
    # Nyquist and the highest sine/cosine modes.
    count = space.modal_shape[0]
    coefficients = jnp.asarray(rng.normal(size=count) + 1j * rng.normal(size=count))
    nodes = space.axes[0].nodes[:, None]

    np.testing.assert_allclose(
        space.evaluate(coefficients, nodes), space.reconstruct(coefficients), atol=1e-13
    )
    np.testing.assert_allclose(
        space.evaluate(coefficients, nodes, real_output=False),
        space.reconstruct(coefficients, real_output=False),
        atol=1e-13,
    )
    for order in (1, 2, 3):
        reference = space.derivative_values(coefficients, axis=0, order=order)
        scale = max(1.0, float(jnp.max(jnp.abs(reference))))
        np.testing.assert_allclose(
            space.derivative_at(coefficients, nodes, (order,)),
            reference,
            atol=1e-13 * scale,
        )


def test_analytic_fields_and_derivatives_at_arbitrary_points():
    rng = np.random.default_rng(0)
    fourier = _space((D.FourierBasisPlan(16), D.AxisDomain.periodic(0.0, 1.0)))
    x = fourier.axes[0].nodes
    k = 2.0 * jnp.pi
    coefficients = fourier.project(jnp.sin(k * x) + 0.3 * jnp.cos(2.0 * k * x))
    points = jnp.asarray(rng.uniform(0.0, 1.0, size=(7, 1)))
    p = points[:, 0]
    np.testing.assert_allclose(
        fourier.evaluate(coefficients, points),
        jnp.sin(k * p) + 0.3 * jnp.cos(2.0 * k * p),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fourier.derivative_at(coefficients, points, (1,)),
        k * jnp.cos(k * p) - 0.6 * k * jnp.sin(2.0 * k * p),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        fourier.derivative_at(coefficients, points, (2,)),
        -(k**2) * jnp.sin(k * p) - 1.2 * k**2 * jnp.cos(2.0 * k * p),
        atol=1e-10,
    )
    # Periodic axes wrap coordinates by the periodic cell.
    np.testing.assert_allclose(
        fourier.evaluate(coefficients, points + 3.0),
        fourier.evaluate(coefficients, points),
        atol=1e-12,
    )

    chebyshev = _space((D.ChebyshevBasisPlan(6), D.AxisDomain.interval(-1.0, 2.0)))
    y = chebyshev.axes[0].nodes
    polynomial = chebyshev.project(y**3 - 2.0 * y + 0.5)
    points = jnp.asarray(rng.uniform(-1.0, 2.0, size=(7, 1)))
    q = points[:, 0]
    np.testing.assert_allclose(
        chebyshev.evaluate(polynomial, points), q**3 - 2.0 * q + 0.5, atol=1e-12
    )
    np.testing.assert_allclose(
        chebyshev.derivative_at(polynomial, points, (1,)),
        3.0 * q**2 - 2.0,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        chebyshev.derivative_at(polynomial, points, (2,)), 6.0 * q, atol=1e-10
    )
    # Bounded axes refuse instead of extrapolating the polynomial.
    with pytest.raises(Exception, match="do not extrapolate"):
        chebyshev.evaluate(polynomial, jnp.asarray(((2.5,),))).block_until_ready()

    sine = _space((D.SineBasisPlan(12), D.AxisDomain.interval(0.0, 2.0)))
    z = sine.axes[0].nodes
    wave = 1.5 * jnp.pi
    mode = sine.project(jnp.sin(wave * z))
    points = jnp.asarray(rng.uniform(0.0, 2.0, size=(7, 1)))
    r = points[:, 0]
    np.testing.assert_allclose(sine.evaluate(mode, points), jnp.sin(wave * r), atol=1e-12)
    np.testing.assert_allclose(
        sine.derivative_at(mode, points, (1,)), wave * jnp.cos(wave * r), atol=1e-11
    )
    np.testing.assert_allclose(
        sine.derivative_at(mode, points, (2,)),
        -(wave**2) * jnp.sin(wave * r),
        atol=1e-10,
    )


def _mixed():
    # u(x, y) = sin(2 pi x) (y^2 + y) on the periodic x cell [0, 1) and y in [-1, 1].
    space = _space(
        (D.FourierBasisPlan(12), D.AxisDomain.periodic(0.0, 1.0)),
        (D.ChebyshevBasisPlan(6), D.AxisDomain.interval(-1.0, 1.0)),
    )
    x, y = jnp.meshgrid(space.axes[0].nodes, space.axes[1].nodes, indexing="ij")
    coefficients = space.project(jnp.sin(2.0 * jnp.pi * x) * (y**2 + y))
    return space, coefficients


_POINTS = jnp.asarray(((0.1, -0.3), (0.45, 0.8), (0.9, 0.05), (0.7, -0.95)))


def test_view_domain_function_gradient_and_laplacian_are_exact():
    space, coefficients = _mixed()
    reconstruction = prepare_spectral_field_reconstruction(space)
    field = _view(reconstruction, coefficients).as_domain_function()
    x, y = _POINTS[:, 0], _POINTS[:, 1]
    k = 2.0 * jnp.pi

    assert reconstruction.regularity == phx.DerivativeRegularity.smooth()
    assert reconstruction.trace_policy.kind == "single-valued"
    np.testing.assert_allclose(
        jax.vmap(field.func)(_POINTS), jnp.sin(k * x) * (y**2 + y), atol=1e-12
    )
    gradient = phx.operators.grad(field, var="x")
    np.testing.assert_allclose(
        jax.vmap(gradient.func)(_POINTS),
        jnp.stack(
            (k * jnp.cos(k * x) * (y**2 + y), jnp.sin(k * x) * (2.0 * y + 1.0)), -1
        ),
        atol=1e-11,
    )
    laplacian = phx.operators.laplacian(field, var="x")
    np.testing.assert_allclose(
        jax.vmap(laplacian.func)(_POINTS),
        -(k**2) * jnp.sin(k * x) * (y**2 + y) + 2.0 * jnp.sin(k * x),
        atol=1e-9,
    )
    mixed = reconstruction.derivative(coefficients, _POINTS, (1, 1))
    assert bool(jnp.all(mixed.valid))
    np.testing.assert_allclose(
        mixed.values, k * jnp.cos(k * x) * (2.0 * y + 1.0), atol=1e-10
    )
    # Values agree with the discretization's own point synthesis, also under jit.
    np.testing.assert_allclose(
        eqx.filter_jit(eqx.filter_vmap(field.func))(_POINTS),
        space.evaluate(coefficients, _POINTS),
        atol=1e-13,
    )


def test_derivatives_beyond_the_prepared_order_are_refused():
    space, coefficients = _mixed()
    reconstruction = prepare_spectral_field_reconstruction(
        space, maximum_derivative_order=1
    )
    field = _view(reconstruction, coefficients).as_domain_function()

    with pytest.raises(ValueError, match="maximum_derivative_order=1"):
        reconstruction.derivative(coefficients, _POINTS, (1, 1))
    with pytest.raises(ValueError, match="maximum_derivative_order=1"):
        phx.operators.partial_n(field, var="x", axis=1, order=2)


def test_queries_outside_the_axis_box_fail_closed_with_evidence():
    space, coefficients = _mixed()
    reconstruction = prepare_spectral_field_reconstruction(space)
    view = _view(reconstruction, coefficients)
    # Outside the bounded y axis, outside the periodic x cell, inside, on the box.
    points = jnp.asarray(((0.5, 1.2), (1.3, 0.0), (0.5, 0.0), (0.0, 1.0)))

    evidence = reconstruction.validity(points)
    assert evidence.status.tolist() == [
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.VALID),
    ]
    nonfinite = reconstruction.validity(jnp.asarray(((jnp.nan, 0.0),)))
    assert nonfinite.status.tolist() == [int(FieldQueryStatus.NONFINITE)]
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        view.as_domain_function().func(points[0])
    with pytest.raises(Exception, match="invalid"):
        eqx.filter_jit(view.as_domain_function().func)(points[:2]).block_until_ready()


def test_coefficient_transpose_is_the_exact_adjoint_of_the_synthesis():
    space, coefficients = _mixed()
    reconstruction = prepare_spectral_field_reconstruction(space)
    cotangent = jnp.asarray((1.0, -2.0, 0.5, 3.0))
    rng = np.random.default_rng(5)
    generic = coefficients + jnp.asarray(
        rng.normal(size=space.modal_shape) + 1j * rng.normal(size=space.modal_shape)
    )

    for derivative in (None, (1, 0), (0, 2)):
        evidence = reconstruction.duality_evidence(
            generic, _POINTS, cotangent, derivative=derivative
        )
        assert bool(evidence.valid)
    _, pullback = jax.vjp(
        lambda c: reconstruction.derivative(c, _POINTS, (1, 0)).values, generic
    )
    np.testing.assert_allclose(
        pullback(cotangent)[0],
        reconstruction.transpose(_POINTS, cotangent, derivative=(1, 0)),
        atol=1e-12,
    )


def test_invalid_query_routes_have_no_transpose():
    space, coefficients = _mixed()
    reconstruction = prepare_spectral_field_reconstruction(space)
    # One valid point and one outside the bounded y axis.
    points = jnp.asarray(((0.5, 0.0), (0.5, 1.2)))
    cotangent = jnp.asarray((1.0, 1.0))

    with pytest.raises(ValueError, match=r"Invalid points \[1\].*OUTSIDE_SUPPORT"):
        reconstruction.transpose(points, cotangent)
    with pytest.raises(ValueError, match=r"Invalid points \[1\].*OUTSIDE_SUPPORT"):
        reconstruction.duality_evidence(coefficients, points, cotangent)
    with pytest.raises(Exception, match="invalid"):
        eqx.filter_jit(reconstruction.transpose)(points, cotangent).block_until_ready()


def test_vector_components_follow_the_declared_value_port():
    space, coefficients = _mixed()
    port = phx.ValuePort(
        "velocity",
        event_shape=(2,),
        component_ids=("velocity[0]", "velocity[1]"),
        representation="spectral-field",
    )
    reconstruction = prepare_spectral_field_reconstruction(space, value_port=port)
    stacked = jnp.stack((coefficients, 2.0 * coefficients), axis=-1)

    assert reconstruction.coefficient_shape == (*space.modal_shape, 2)
    values = reconstruction.evaluate(stacked, _POINTS).values
    expected = space.evaluate(coefficients, _POINTS)
    np.testing.assert_allclose(
        values, jnp.stack((expected, 2.0 * expected), -1), atol=1e-13
    )


def test_support_geometry_must_be_the_axis_box():
    space, coefficients = _mixed()
    box = phx.geometry.Orthotope((0.5, 0.0), (1.0, 2.0), feature_id="user-box").compile()
    reconstruction = prepare_spectral_field_reconstruction(space, support_geometry=box)
    view = _view(reconstruction, coefficients)
    np.testing.assert_allclose(
        view.query(_POINTS).values, space.evaluate(coefficients, _POINTS), atol=1e-13
    )

    shrunk = phx.geometry.Orthotope((0.5, 0.0), (1.0, 1.5)).compile()
    with pytest.raises(ValueError, match="reconstruction box"):
        prepare_spectral_field_reconstruction(space, support_geometry=shrunk)


def test_axes_without_point_synthesis_or_box_support_are_refused():
    constrained = _space(
        (
            D.ConstrainedBasisPlan(
                D.ChebyshevBasisPlan(8), D.SpectralBoundaryConditionPlan.dirichlet()
            ),
            D.AxisDomain.interval(-1.0, 1.0),
        )
    )
    with pytest.raises(ValueError, match="arbitrary-point synthesis"):
        prepare_spectral_field_reconstruction(constrained)
    with pytest.raises(ValueError, match="arbitrary-point synthesis"):
        constrained.evaluate(
            jnp.zeros(constrained.modal_shape, dtype=jnp.complex128), jnp.zeros((1, 1))
        )

    rational = _space((D.RationalChebyshevLineBasisPlan(8), D.AxisDomain.real_line()))
    with pytest.raises(ValueError, match="unbounded"):
        prepare_spectral_field_reconstruction(rational)
