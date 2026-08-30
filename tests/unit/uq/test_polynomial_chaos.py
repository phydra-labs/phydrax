#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _factors():
    uniform = phx.domain.ProbabilityDomain(
        phx.uq.Uniform(2.0, 6.0), label="conductivity"
    )
    normal = phx.domain.ProbabilityDomain(
        phx.uq.Normal(-1.5, 2.5), label="forcing"
    )
    return uniform, normal


def _basis(degree=3):
    return phx.uq.PolynomialChaosBasis(_factors(), degree)


def _projection_plan(basis, order=5, **kwargs):
    return phx.uq.PolynomialChaosProjectionPlan(
        basis,
        phx.integration.ProductIntegrationPlan(
            {
                "conductivity": phx.integration.FixedQuadraturePlan(
                    phx.integration.GaussLegendreRule(order)
                ),
                "forcing": phx.integration.FixedQuadraturePlan(
                    phx.integration.GaussHermiteRule(order)
                ),
            }
        ),
        **kwargs,
    )


def test_total_degree_multiindices_are_graded_deterministic_and_content_addressed():
    indices = phx.uq.PolynomialMultiIndexSet(2, 3)
    repeated = phx.uq.PolynomialMultiIndexSet(2, 3)

    assert indices.feature_count == 10
    assert np.array_equal(
        indices.indices,
        np.asarray(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [1, 1],
                [0, 2],
                [3, 0],
                [2, 1],
                [1, 2],
                [0, 3],
            ]
        ),
    )
    assert indices.content_id == repeated.content_id
    assert indices.storage_bytes == indices.indices.size * indices.indices.dtype.itemsize


def test_degree_zero_is_a_valid_constant_basis_and_regression():
    factor = phx.domain.ProbabilityDomain(phx.uq.Uniform(-4.0, 8.0), label="x")
    basis = phx.uq.PolynomialChaosBasis(factor, 0)
    values = basis.evaluate(jnp.asarray([[-4.0], [2.0], [8.0]]))
    fit = phx.uq.PolynomialChaosRegressionPlan(basis).fit(
        jnp.asarray([[0.0], [4.0]]), jnp.asarray([3.25, 3.25])
    )

    assert basis.feature_count == 1
    assert values.shape == (3, 1)
    assert jnp.all(values == 1.0)
    assert jnp.allclose(fit.expansion(jnp.asarray([[-2.0], [7.0]])), 3.25)
    assert jnp.allclose(fit.expansion.variance, 0.0)
    assert jnp.allclose(fit.expansion.total_order_sobol["x"], 0.0)


def test_basis_identity_and_modes_preserve_factor_labels_and_order():
    factors = _factors()
    forward = phx.uq.PolynomialChaosBasis(factors, 1)
    reversed_basis = phx.uq.PolynomialChaosBasis(tuple(reversed(factors)), 1)
    labeled_point = {
        "conductivity": jnp.asarray(5.0),
        "forcing": jnp.asarray(1.0),
    }

    assert forward.labels == ("conductivity", "forcing")
    assert reversed_basis.labels == ("forcing", "conductivity")
    assert forward.basis_id != reversed_basis.basis_id
    assert jnp.allclose(
        forward.evaluate(labeled_point)[1:],
        reversed_basis.evaluate(labeled_point)[1:][::-1],
    )


def test_basis_is_orthonormal_for_nonstandard_uniform_and_normal_laws():
    basis = _basis(3)
    legendre = phx.integration.GaussLegendreRule(8).data()
    hermite = phx.integration.GaussHermiteRule(8).data()
    uniform = basis.factors[0].from_reference(legendre.nodes)
    normal = basis.factors[1].from_reference(hermite.nodes)
    first, second = jnp.meshgrid(uniform, normal, indexing="ij")
    weights = (0.5 * legendre.weights[:, None] * hermite.weights[None, :]).reshape((-1,))
    vandermonde = basis.evaluate(
        {
            "conductivity": first.reshape((-1,)),
            "forcing": second.reshape((-1,)),
        }
    )
    gram = vandermonde.T @ (weights[:, None] * vandermonde)

    assert jnp.allclose(gram, jnp.eye(basis.feature_count), rtol=2e-11, atol=2e-11)
    point = basis.evaluate(
        {"forcing": jnp.asarray(-1.5), "conductivity": jnp.asarray(4.0)}
    )
    assert jnp.allclose(point[:3], jnp.asarray([1.0, 0.0, 0.0]))


def test_regression_recovers_exact_span_and_reports_out_of_span_residual():
    basis = _basis(2)
    uniform = jnp.linspace(2.1, 5.9, 5)
    normal = jnp.asarray([-4.0, -2.5, -1.0, 0.5, 2.0])
    first, second = jnp.meshgrid(uniform, normal, indexing="ij")
    points = jnp.stack((first.reshape((-1,)), second.reshape((-1,))), axis=-1)
    design = basis.evaluate(points)
    expected = jnp.asarray([1.25, -0.5, 2.0, 0.75, -1.0, 0.3])
    exact_values = design @ expected

    exact = phx.uq.PolynomialChaosRegressionPlan(basis).fit(points, exact_values)
    misspecified = phx.uq.PolynomialChaosRegressionPlan(basis).fit(
        points,
        exact_values + 0.2 * ((points[:, 0] - 4.0) / 2.0) ** 3,
    )

    assert exact.method == "regression-least-squares"
    assert exact.rank == basis.feature_count
    assert jnp.allclose(exact.expansion.coefficients, expected, rtol=2e-10, atol=2e-10)
    assert jnp.allclose(exact.residual_norm, 0.0, atol=2e-10)
    assert float(misspecified.residual_norm) > 1e-4


def test_projection_and_regression_match_for_an_exact_mixed_span():
    basis = _basis(3)

    def model(conductivity, forcing):
        x = (conductivity - 4.0) / 2.0
        z = (forcing + 1.5) / 2.5
        return 2.0 + x - 0.5 * z + 0.75 * x * z + 0.2 * z**3

    projection = _projection_plan(basis, order=5).fit(model)
    uniform = jnp.linspace(2.05, 5.95, 8)
    normal = jnp.linspace(-5.0, 2.0, 8)
    first, second = jnp.meshgrid(uniform, normal, indexing="ij")
    points = jnp.stack((first.reshape((-1,)), second.reshape((-1,))), axis=-1)
    regression = phx.uq.PolynomialChaosRegressionPlan(basis).fit(
        points, jax.vmap(model)(points[:, 0], points[:, 1])
    )

    assert projection.method == "projection"
    assert projection.model_evaluations == 25
    assert jnp.allclose(
        projection.expansion.coefficients,
        regression.expansion.coefficients,
        rtol=2e-9,
        atol=2e-9,
    )


def test_pytree_and_field_outputs_preserve_structure_and_physical_axes():
    basis = _basis(2)
    first, second = jnp.meshgrid(
        jnp.asarray([2.0, 4.0, 6.0]),
        jnp.asarray([-4.0, -1.0, 3.0]),
        indexing="ij",
    )
    points = jnp.stack(
        (first.reshape((-1,)), second.reshape((-1,))), axis=-1
    )
    scalar = points[:, 0] + points[:, 1]
    field_data = jnp.stack((scalar, points[:, 0] - points[:, 1]), axis=-1)
    outputs = {
        "scalar": scalar,
        "field": cx.Field(field_data, dims=("sample", "channel")),
    }
    fit = phx.uq.PolynomialChaosRegressionPlan(basis).fit(points, outputs)
    predicted = fit.expansion(jnp.asarray([3.25, -0.75]))
    aligned = fit.expansion(
        {
            "conductivity": cx.Field(
                jnp.asarray([3.25, 4.5]), dims=("draw",)
            ),
            "forcing": cx.Field(
                jnp.asarray([-0.75, 0.25]), dims=("draw",)
            ),
        }
    )

    assert set(predicted) == {"field", "scalar"}
    assert isinstance(predicted["field"], cx.Field)
    assert predicted["field"].dims == ("channel",)
    assert predicted["field"].shape == (2,)
    assert jnp.allclose(predicted["scalar"], 2.5)
    assert isinstance(fit.expansion.mean["field"], cx.Field)
    assert fit.expansion.variance["field"].dims == ("channel",)
    assert aligned["field"].dims == ("draw", "channel")


def test_projection_supports_pytree_and_field_model_outputs():
    basis = _basis(1)

    def model(conductivity, forcing):
        value = conductivity + forcing
        return {
            "scalar": value,
            "field": cx.Field(
                jnp.stack((value, conductivity - forcing)),
                dims=("channel",),
            ),
        }

    result = _projection_plan(basis, order=3).fit(model)
    evaluated = result.expansion(
        {"conductivity": jnp.asarray(4.0), "forcing": jnp.asarray(-1.5)}
    )

    assert jnp.allclose(evaluated["scalar"], 2.5)
    assert evaluated["field"].dims == ("channel",)
    assert jnp.allclose(evaluated["field"].data, jnp.asarray([2.5, 5.5]))


def test_coefficient_moments_and_sobol_effects_are_analytic_and_axis_preserving():
    basis = _basis(2)
    coefficients = jnp.zeros((basis.feature_count, 2))
    coefficients = coefficients.at[0].set(jnp.asarray([3.0, -2.0]))
    coefficients = coefficients.at[1].set(jnp.asarray([1.0, 2.0]))
    coefficients = coefficients.at[2].set(jnp.asarray([2.0, 0.0]))
    coefficients = coefficients.at[4].set(jnp.asarray([3.0, 1.0]))
    expansion = phx.uq.PolynomialChaosExpansion(
        basis, cx.Field(coefficients, dims=(None, "channel"))
    )

    assert expansion.coefficients.dims == (
        "__phydra_uq_polynomial_mode",
        "channel",
    )
    assert expansion.mean.dims == ("channel",)
    assert jnp.allclose(expansion.mean.data, jnp.asarray([3.0, -2.0]))
    assert jnp.allclose(expansion.variance.data, jnp.asarray([14.0, 5.0]))
    assert jnp.allclose(
        expansion.first_order_sobol["conductivity"].data,
        jnp.asarray([1.0 / 14.0, 4.0 / 5.0]),
    )
    assert jnp.allclose(
        expansion.first_order_sobol["forcing"].data,
        jnp.asarray([4.0 / 14.0, 0.0]),
    )
    assert jnp.allclose(
        expansion.total_order_sobol["conductivity"].data,
        jnp.asarray([10.0 / 14.0, 1.0]),
    )
    assert jnp.allclose(
        expansion.total_order_sobol["forcing"].data,
        jnp.asarray([13.0 / 14.0, 1.0 / 5.0]),
    )


def test_rank_deficiency_and_nonfinite_inputs_fail_without_repair():
    factor = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="x")
    basis = phx.uq.PolynomialChaosBasis(factor, 2)
    plan = phx.uq.PolynomialChaosRegressionPlan(basis)

    with pytest.raises(ValueError, match="solve failed"):
        plan.fit(jnp.zeros((3, 1)), jnp.asarray([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        plan.fit(
            jnp.asarray([[-1.0], [0.0], [jnp.nan]]),
            jnp.asarray([1.0, 2.0, 3.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        plan.fit(
            jnp.asarray([[-1.0], [0.0], [1.0]]),
            jnp.asarray([1.0, jnp.inf, 3.0]),
        )


def test_explicit_native_svd_policy_can_select_rank_deficient_pseudoinverse():
    factor = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="x")
    basis = phx.uq.PolynomialChaosBasis(factor, 2)
    selected_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseSVD(),
        rank=phx.linalg.RankPolicy(require_full_rank=False),
    )
    result = phx.uq.PolynomialChaosRegressionPlan(
        basis, least_squares_policy=selected_policy
    ).fit(jnp.zeros((4, 1)), jnp.ones((4,)))

    assert result.rank == 1
    assert jnp.all(jnp.isfinite(result.expansion.coefficients))
    assert result.provenance["linear_methods"] == ("dense-svd",)


def test_unsupported_or_nonindependent_laws_are_rejected_during_basis_construction():
    lognormal = phx.domain.ProbabilityDomain(
        phx.uq.LogNormal(0.0, 0.5), label="positive"
    )
    first, second = _factors()

    with pytest.raises(TypeError, match="Uniform.*Normal"):
        phx.uq.PolynomialChaosBasis(lognormal, 2)
    with pytest.raises(TypeError, match="independent scalar"):
        phx.uq.PolynomialChaosBasis((first, phx.domain.ProductDomain(first, second)), 2)


def test_feature_storage_evaluation_and_design_capacity_guards_fail_closed():
    with pytest.raises(ValueError, match="maximum_features"):
        phx.uq.PolynomialMultiIndexSet(8, 8, maximum_features=100)
    with pytest.raises(ValueError, match="maximum_storage_bytes"):
        phx.uq.PolynomialMultiIndexSet(4, 4, maximum_storage_bytes=8)

    basis = _basis(2)
    with pytest.raises(ValueError, match="maximum_model_evaluations"):
        _projection_plan(basis, order=3, maximum_model_evaluations=8).fit(
            lambda x, z: x + z
        )
    with pytest.raises(ValueError, match="maximum_samples"):
        phx.uq.PolynomialChaosRegressionPlan(basis, maximum_samples=5).fit(
            jnp.ones((6, 2)), jnp.ones((6,))
        )
    with pytest.raises(ValueError, match="maximum_design_bytes"):
        phx.uq.PolynomialChaosRegressionPlan(
            basis, maximum_design_bytes=8
        ).fit(jnp.ones((6, 2)), jnp.ones((6,)))


def test_projection_honors_evaluation_accumulation_and_output_precision():
    basis = _basis(2)
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype=jnp.float32,
        accumulation_dtype=jnp.float64,
        decision_dtype=jnp.float64,
        output_dtype=jnp.float64,
    )
    result = _projection_plan(
        basis, order=4, precision=precision
    ).fit(lambda conductivity, forcing: conductivity * forcing)

    assert result.expansion.coefficients.dtype == jnp.float64
    assert result.evidence["precision_policy_id"] == precision.policy_id


def test_expansion_is_jittable_differentiable_and_preserves_coefficient_precision():
    factor = phx.domain.ProbabilityDomain(phx.uq.Normal(1.0, 2.0), label="x")
    basis = phx.uq.PolynomialChaosBasis(factor, 2)
    coefficients = jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float64)
    expansion = phx.uq.PolynomialChaosExpansion(basis, coefficients)
    point = jnp.asarray([1.25], dtype=jnp.float64)

    value = jax.jit(expansion)(point)
    derivative = jax.jit(jax.grad(lambda scalar: expansion(scalar[None])))(point)

    assert value.dtype == jnp.float64
    assert derivative.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.allclose(value, expansion(point))


def test_polynomial_chaos_fit_result_is_portably_exported(tmp_path):
    basis = _basis(1)
    points = jnp.asarray(
        [[2.0, -2.0], [3.0, -1.0], [5.0, 0.0], [6.0, 1.0]]
    )
    result = phx.uq.PolynomialChaosRegressionPlan(basis).fit(
        points, points[:, 0] + points[:, 1]
    )
    path = phx.uq.export_result(result, tmp_path / "polynomial-chaos.npz")
    archive = phx.uq.read_result_archive(path)

    assert archive.kind == "polynomial_chaos_fit"
    assert archive.metadata["basis_id"] == basis.basis_id
    assert archive.metadata["labels"] == ["conductivity", "forcing"]
    assert "<root>" in archive.tree("coefficients")
