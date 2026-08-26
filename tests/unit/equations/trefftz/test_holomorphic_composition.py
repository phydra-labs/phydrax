#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _polynomial(coefficients):
    values = jnp.asarray(coefficients, dtype=float)
    potential = phx.equations.HolomorphicPolynomialPotential(
        values.shape[0],
        values.shape[1] - 1,
    )
    return eqx.tree_at(
        lambda item: (item.coefficient_real, item.coefficient_imag),
        potential,
        (values, jnp.zeros_like(values)),
    )


def test_certificate_binds_canonical_complex_algebra_and_construction_dependencies():
    potential = _polynomial([[1.0, 2.0]])
    certificate = potential.holomorphic_certificate()
    assert certificate.complex_algebra_id == (
        phx.metrix.algebra.ComplexAlgebraSpec().algebra_id
    )
    assert len(certificate.construction_dependencies) == 1
    with pytest.raises(ValueError, match="canonical real two-coordinate complex"):
        phx.equations.HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=1,
            construction="invalid-algebra",
            normalization_id=certificate.normalization_id,
            maximum_derivative_order=1,
            operations=("complex-affine",),
            parameter_coverage="finite-subspace",
            linear_in_parameters=True,
            complex_algebra_id="not-the-complex-algebra",
        )


def test_branch_bundle_concatenates_values_jets_and_linearity():
    first = _polynomial([[1.0, 2.0, -0.5]])
    second = _polynomial([[0.3, -1.0], [2.0, 0.25]])
    bundle = phx.equations.HolomorphicBranchBundle((first, second))
    z = jnp.asarray(0.2 - 0.1j)
    jet = bundle.jet(z, 2)

    assert jnp.allclose(bundle(z), jnp.concatenate((first(z), second(z))))
    assert jnp.allclose(
        jet.derivative(1),
        jnp.concatenate((first.jet(z, 1).derivative(1), second.jet(z, 1).derivative(1))),
    )
    certificate = bundle.holomorphic_certificate()
    assert certificate.complex_output_size == 3
    assert certificate.parameter_coverage == "finite-subspace"
    assert certificate.linear_in_parameters
    assert bundle.factorization.gauge_kind == "none"


def test_branch_bundle_composes_with_biharmonic_physical_wrapper():
    phi = _polynomial([[0.0, 1.0, 0.2]])
    psi = _polynomial([[0.1, -0.3, 0.4]])
    bundle = phx.equations.HolomorphicBranchBundle((phi, psi))
    field = phx.equations.BiharmonicPotential2D(bundle)
    point = jnp.asarray([0.2, -0.4])

    def laplacian(value):
        return jnp.trace(jax.hessian(field)(value))

    residual = jnp.trace(jax.hessian(laplacian)(point))

    assert jnp.abs(residual) < 1e-10
    assert field.model_metadata()["trial_space_certificate"].linear_in_coefficients


def test_product_potential_analytic_jets_match_direct_holomorphic_ad():
    first = _polynomial([[1.0, 1.0], [2.0, -1.0]])
    second = _polynomial([[3.0, 1.0], [1.0, 2.0]])
    product = phx.equations.HolomorphicProductPotential(
        (first, second),
        latent_rank=2,
        branches=1,
    )
    z = jnp.asarray(0.2 + 0.1j)
    jet = product.jet(z, 4)

    def scalar(value):
        return product(value)[0]

    derivative = scalar
    expected = [scalar(z)]
    for _ in range(4):
        derivative = jax.jacfwd(derivative, holomorphic=True)
        expected.append(derivative(z))
    for order, value in enumerate(expected):
        assert jnp.allclose(jet.derivative(order)[0], value)

    certificate = product.holomorphic_certificate()
    assert certificate.parameter_coverage == "finite-parametric-family"
    assert not certificate.linear_in_parameters
    assert product.factorization.gauge_kind == "multiplicative-factor-scale"


def test_product_potential_is_harmonic_and_reports_factor_gauge():
    first = _polynomial([[1.0, 0.5], [0.2, -0.1]])
    second = _polynomial([[0.3, 0.8], [1.2, 0.4]])
    product = phx.equations.HolomorphicProductPotential(
        (first, second),
        latent_rank=2,
        branches=1,
    )
    harmonic = phx.equations.HarmonicPotential2D(product)
    point = jnp.asarray([0.15, -0.25])

    assert jnp.abs(jnp.trace(jax.hessian(harmonic)(point))) < 1e-10
    gauge = product.gauge_report(point[0] + 1j * point[1])
    assert bool(gauge.finite)
    assert float(gauge.imbalance_ratio) >= 1.0

    scaled_first = eqx.tree_at(
        lambda item: (item.coefficient_real, item.coefficient_imag),
        first,
        (2.0 * first.coefficient_real, 2.0 * first.coefficient_imag),
    )
    scaled_second = eqx.tree_at(
        lambda item: (item.coefficient_real, item.coefficient_imag),
        second,
        (0.5 * second.coefficient_real, 0.5 * second.coefficient_imag),
    )
    scaled = phx.equations.HolomorphicProductPotential(
        (scaled_first, scaled_second),
        latent_rank=2,
        branches=1,
    )
    z = point[0] + 1j * point[1]
    assert jnp.allclose(product(z), scaled(z))


def test_product_and_bundle_validate_child_contracts():
    factor = _polynomial([[1.0, 1.0]])
    with pytest.raises(ValueError, match=r"latent_rank \* branches"):
        phx.equations.HolomorphicProductPotential(
            (factor,),
            latent_rank=2,
            branches=1,
        )

    generic = phx.nn.models.Separable(
        in_size="scalar",
        out_size=1,
        latent_size=1,
        models=(
            phx.nn.models.HolomorphicMLP(
                in_size=1,
                out_size=1,
                hidden_sizes=(2,),
                key=jr.key(0),
            ),
        ),
        keep_outputs_complex=True,
    )
    with pytest.raises(TypeError, match="HolomorphicPotentialProvider"):
        phx.equations.HarmonicPotential2D(generic)
