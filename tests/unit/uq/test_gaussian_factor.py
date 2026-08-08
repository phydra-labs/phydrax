#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_rectangular_factor_prediction_matches_dense_covariance_algebra():
    mean = jnp.asarray([0.5, -1.0, 0.25])
    root = jnp.asarray([[1.0, 0.0], [0.5, 0.75], [-0.25, 0.4]])
    factor = phx.uq.GaussianFactor(root, factor_id="rectangular-prior")
    matrix = jnp.asarray([[1.0, -0.5, 0.25], [0.2, 1.5, -1.0]])
    offset = jnp.asarray([0.1, -0.3])
    noise = phx.uq.GaussianFactor(
        jnp.asarray([[0.2], [-0.1]]), factor_id="independent-noise"
    )

    moments = phx.uq.predict_affine_gaussian(
        mean,
        factor,
        matrix,
        offset,
        noise,
        moments_id="dense-equivalence",
    )
    dense_covariance = root @ root.T
    expected_covariance = (
        matrix @ dense_covariance @ matrix.T + noise.factor @ noise.factor.T
    )

    assert factor.rank == 2
    assert factor.event_size == 3
    assert moments.factor.rank == 2
    assert jnp.allclose(moments.mean, matrix @ mean + offset)
    assert jnp.allclose(moments.covariance, expected_covariance)
    assert jnp.allclose(moments.cross_covariance, dense_covariance @ matrix.T)
    assert moments.factor.resolved_method == "qr-compression"
    assert bool(moments.valid)


def test_rectangular_regression_evaluation_has_zero_input_output_cross_covariance():
    regression = phx.uq.GaussianRegression(
        jnp.asarray([[1.0, -0.5, 0.25], [0.2, 0.75, -1.0]]),
        jnp.asarray([0.1, -0.2]),
        phx.uq.GaussianFactor(jnp.asarray([[0.3], [-0.15]])),
    )
    value = jnp.asarray([0.4, -0.7, 0.2])
    evaluated = jax.jit(lambda argument: regression(argument))(value)

    assert evaluated.cross_covariance.shape == (3, 2)
    assert jnp.array_equal(evaluated.cross_covariance, jnp.zeros((3, 2)))
    assert jnp.allclose(evaluated.mean, regression.matrix @ value + regression.offset)
    assert jnp.allclose(evaluated.covariance, regression.noise_factor.covariance)
    assert bool(evaluated.valid)


def test_independent_factor_addition_and_explicit_regularization_are_preserved():
    covariance = jnp.asarray([[1.5, 0.25], [0.25, 0.5]])
    regularized = phx.uq.gaussian_factor_from_covariance(
        covariance,
        regularization=0.2,
        rank_tolerance=1e-12,
        hermitian_tolerance=1e-12,
        factor_id="regularized",
    )
    independent = phx.uq.GaussianFactor(jnp.asarray([[0.0], [0.3]]), regularization=0.1)
    summed = phx.uq.add_independent_gaussian_factors(
        regularized, independent, compress=False
    )

    assert jnp.allclose(regularized.covariance, covariance + 0.2 * jnp.eye(2))
    assert jnp.allclose(
        summed.covariance, regularized.covariance + independent.covariance
    )
    assert jnp.allclose(summed.regularization, 0.3)
    assert summed.rank == 3
    assert bool(regularized.valid)
    assert int(regularized.status) == phx.uq.GAUSSIAN_FACTOR_SUCCESS


def test_factor_addition_preserves_invalid_operand_provenance_and_metadata():
    nonhermitian = phx.uq.gaussian_factor_from_covariance(
        jnp.asarray([[1.0, 0.5], [0.0, 1.0]])
    )
    valid = phx.uq.GaussianFactor(jnp.eye(2))
    invalid_tolerance = phx.uq.GaussianFactor(
        jnp.eye(2),
        rank_tolerance=-1.0,
    )
    invalid_regularization = phx.uq.GaussianFactor(
        jnp.eye(2),
        regularization=-0.25,
    )

    left_failure = phx.uq.add_independent_gaussian_factors(
        nonhermitian,
        invalid_tolerance,
    )
    right_failure = phx.uq.add_independent_gaussian_factors(
        valid,
        invalid_tolerance,
        compress=False,
    )
    regularization_failure = phx.uq.add_independent_gaussian_factors(
        invalid_regularization,
        phx.uq.GaussianFactor(jnp.eye(2), regularization=0.5),
    )

    assert not bool(left_failure.valid)
    assert int(left_failure.status) == phx.uq.GAUSSIAN_FACTOR_NON_HERMITIAN
    assert not bool(right_failure.valid)
    assert int(right_failure.status) == phx.uq.GAUSSIAN_FACTOR_INVALID_REGULARIZATION
    assert right_failure.rank_tolerance == -1.0
    assert not bool(regularization_failure.valid)
    assert (
        int(regularization_failure.status)
        == phx.uq.GAUSSIAN_FACTOR_INVALID_REGULARIZATION
    )
    assert regularization_failure.regularization == -0.25


def test_invalid_rank_tolerance_precedes_psd_diagnostics():
    covariance = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])

    for tolerance in (-1.0, jnp.nan):
        factor = phx.uq.gaussian_factor_from_covariance(
            covariance,
            rank_tolerance=tolerance,
        )
        assert not bool(factor.valid)
        assert int(factor.status) == phx.uq.GAUSSIAN_FACTOR_INVALID_REGULARIZATION


def test_singular_and_zero_noise_conditioning_remain_exact():
    prior = phx.uq.GaussianFactor(jnp.asarray([[1.0, 0.0], [0.0, 0.0]]))
    predicted = phx.uq.predict_affine_gaussian(
        jnp.asarray([0.0, 0.0]),
        prior,
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([0.0]),
        None,
    )
    conditioned = phx.uq.condition_gaussian(
        jnp.asarray([0.0, 0.0]),
        prior,
        predicted,
        jnp.asarray([2.0]),
        rank_tolerance=1e-12,
        support_tolerance=1e-12,
    )

    assert prior.numerical_rank == 1
    assert jnp.isneginf(phx.uq.gaussian_factor_log_determinant(prior))
    assert jnp.allclose(
        phx.uq.gaussian_factor_quadratic_form(
            prior, jnp.asarray([2.0, 0.0]), support_tolerance=1e-12
        ),
        4.0,
    )
    assert jnp.isposinf(
        phx.uq.gaussian_factor_quadratic_form(
            prior, jnp.asarray([0.0, 1.0]), support_tolerance=1e-12
        )
    )
    assert jnp.allclose(conditioned.mean, jnp.asarray([2.0, 0.0]))
    assert jnp.allclose(conditioned.covariance, jnp.zeros((2, 2)), atol=1e-12)
    assert bool(conditioned.valid)

    zero = phx.uq.GaussianFactor(jnp.empty((2, 0)))
    assert zero.rank == 0
    assert zero.numerical_rank == 0
    assert jnp.array_equal(zero.covariance, jnp.zeros((2, 2)))
    assert phx.uq.gaussian_factor_quadratic_form(zero, jnp.zeros(2)) == 0.0
    assert jnp.isposinf(phx.uq.gaussian_factor_quadratic_form(zero, jnp.ones(2)))


def test_singular_condition_rejects_observations_outside_the_gaussian_support():
    prior = phx.uq.GaussianFactor(jnp.asarray([[1.0], [0.0]]))
    output = phx.uq.predict_affine_gaussian(
        jnp.zeros(2),
        prior,
        jnp.eye(2),
        jnp.zeros(2),
        None,
    )
    conditioned = phx.uq.condition_gaussian(
        jnp.zeros(2),
        prior,
        output,
        jnp.asarray([0.0, 1.0]),
        rank_tolerance=1e-12,
        support_tolerance=1e-12,
    )

    assert not bool(conditioned.valid)
    assert int(conditioned.status) == phx.uq.CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION


def test_conditioning_propagates_invalid_input_and_output_factors():
    invalid_input = phx.uq.gaussian_factor_from_covariance(
        jnp.asarray([[1.0, 0.25], [0.0, 1.0]]),
        hermitian_tolerance=0.0,
    )
    valid_output_factor = phx.uq.GaussianFactor(jnp.asarray([[0.7]]))
    valid_output = phx.uq.ConditionalGaussianMoments(
        jnp.zeros(1),
        valid_output_factor,
        jnp.zeros((2, 1)),
    )
    valid_input = phx.uq.GaussianFactor(jnp.eye(2))
    invalid_output_factor = phx.uq.gaussian_factor_from_covariance(
        jnp.asarray([[1.0, -0.4], [0.0, 1.0]]),
        hermitian_tolerance=0.0,
    )
    invalid_output = phx.uq.ConditionalGaussianMoments(
        jnp.zeros(2),
        invalid_output_factor,
        jnp.zeros((2, 2)),
    )

    for input_factor, output_moments, observed in (
        (invalid_input, valid_output, jnp.zeros(1)),
        (valid_input, invalid_output, jnp.zeros(2)),
    ):
        regression = phx.uq.GaussianRegression.from_moments(
            jnp.zeros(2),
            input_factor,
            output_moments,
        )
        evaluated = regression(observed)
        conditioned = phx.uq.condition_gaussian(
            jnp.zeros(2),
            input_factor,
            output_moments,
            observed,
        )

        assert not bool(regression.valid)
        assert int(regression.status) == phx.uq.CONDITIONAL_GAUSSIAN_INVALID_FACTOR
        assert not bool(evaluated.valid)
        assert int(evaluated.status) == phx.uq.CONDITIONAL_GAUSSIAN_INVALID_FACTOR
        assert not bool(conditioned.valid)
        assert int(conditioned.status) == phx.uq.CONDITIONAL_GAUSSIAN_INVALID_FACTOR


def test_complex_factors_use_hermitian_adjoints_everywhere():
    root = jnp.asarray(
        [[1.0 + 0.5j, 0.2 - 0.1j], [0.3j, 0.8 + 0.4j]],
        dtype=jnp.complex128,
    )
    factor = phx.uq.GaussianFactor(root)
    covariance = root @ jnp.conj(root.T)
    reconstructed = phx.uq.gaussian_factor_from_covariance(
        covariance,
        rank_tolerance=1e-12,
        hermitian_tolerance=1e-12,
    )
    right = phx.uq.GaussianFactor(jnp.asarray([[0.5 - 0.2j, 0.1j]], dtype=jnp.complex128))
    residual = jnp.asarray([0.4 + 0.1j, -0.2 + 0.3j])

    assert jnp.allclose(factor.covariance, covariance)
    assert jnp.allclose(reconstructed.covariance, covariance)
    assert jnp.allclose(
        phx.uq.gaussian_cross_covariance(factor, right),
        root @ jnp.conj(right.factor.T),
    )
    assert jnp.allclose(
        phx.uq.gaussian_factor_log_determinant(factor),
        jnp.linalg.slogdet(covariance)[1],
    )
    assert jnp.allclose(
        phx.uq.gaussian_factor_quadratic_form(factor, residual, support_tolerance=1e-12),
        jnp.real(jnp.vdot(residual, jnp.linalg.solve(covariance, residual))),
    )
    assert bool(reconstructed.valid)


def test_nonhermitian_dense_input_is_diagnosed_without_silent_symmetrization():
    covariance = jnp.asarray([[1.0 + 0.0j, 0.0 + 0.5j], [0.0 + 0.5j, 1.0 + 0.0j]])
    factor = phx.uq.gaussian_factor_from_covariance(covariance, hermitian_tolerance=0.0)

    assert not bool(factor.valid)
    assert int(factor.status) == phx.uq.GAUSSIAN_FACTOR_NON_HERMITIAN


def test_qr_compression_has_dense_equivalent_gradients():
    root = jnp.asarray([[1.0, 0.25, -0.4, 0.8], [0.1, 1.2, 0.3, -0.2]])

    def compressed_loss(value):
        compressed = phx.uq.compress_gaussian_factor(phx.uq.GaussianFactor(value))
        return jnp.sum(compressed.covariance**2)

    def dense_loss(value):
        covariance = value @ value.T
        return jnp.sum(covariance**2)

    compressed = phx.uq.compress_gaussian_factor(phx.uq.GaussianFactor(root))
    assert compressed.factor.shape == (2, 2)
    assert jnp.allclose(compressed.covariance, root @ root.T)
    assert jnp.allclose(
        jax.grad(compressed_loss)(root),
        jax.grad(dense_loss)(root),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gaussian_modules_are_jittable_pytrees_with_exact_reconstruction():
    factor = phx.uq.GaussianFactor(
        jnp.asarray([[1.0, 0.0], [0.25, 0.5]]), factor_id="tree-factor"
    )
    moments = phx.uq.ConditionalGaussianMoments(
        jnp.asarray([0.2, -0.1]),
        factor,
        factor.covariance,
        moments_id="tree-moments",
    )
    regression = phx.uq.GaussianRegression(
        jnp.asarray([[1.0, 0.2], [-0.3, 0.5]]),
        jnp.asarray([0.1, -0.2]),
        factor,
        regression_id="tree-regression",
    )

    leaves, structure = jax.tree_util.tree_flatten(regression)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    compiled_covariance = jax.jit(lambda value: phx.uq.GaussianFactor(value).covariance)(
        factor.factor
    )
    compiled_factorization = jax.jit(
        lambda covariance, regularization: phx.uq.gaussian_factor_from_covariance(
            covariance,
            regularization=regularization,
            rank_tolerance=jnp.asarray(1e-12),
            hermitian_tolerance=jnp.asarray(1e-12),
        )
    )(factor.covariance, jnp.asarray(0.1))

    assert isinstance(rebuilt, phx.uq.GaussianRegression)
    assert rebuilt.regression_id == "tree-regression"
    assert rebuilt.noise_factor.factor_id == "tree-factor"
    assert jnp.array_equal(rebuilt.matrix, regression.matrix)
    assert jnp.array_equal(rebuilt.noise_factor.factor, factor.factor)
    assert jnp.allclose(compiled_covariance, factor.covariance)
    assert bool(compiled_factorization.valid)
    assert jnp.allclose(
        compiled_factorization.covariance, factor.covariance + 0.1 * jnp.eye(2)
    )
    moment_leaves, moment_structure = jax.tree_util.tree_flatten(moments)
    rebuilt_moments = jax.tree_util.tree_unflatten(moment_structure, moment_leaves)
    assert rebuilt_moments.moments_id == "tree-moments"
    assert jnp.array_equal(rebuilt_moments.cross_covariance, factor.covariance)


def test_conditioning_matches_dense_gaussian_formula():
    mean = jnp.asarray([0.4, -0.2])
    covariance = jnp.asarray([[1.2, 0.3], [0.3, 0.8]])
    prior = phx.uq.gaussian_factor_from_covariance(
        covariance, rank_tolerance=1e-12, hermitian_tolerance=1e-12
    )
    matrix = jnp.asarray([[1.0, -0.5], [0.25, 0.75]])
    offset = jnp.asarray([0.1, -0.3])
    noise_covariance = jnp.asarray([[0.2, 0.05], [0.05, 0.15]])
    noise = phx.uq.gaussian_factor_from_covariance(
        noise_covariance, rank_tolerance=1e-12, hermitian_tolerance=1e-12
    )
    observed = jnp.asarray([0.7, -0.4])
    output = phx.uq.predict_affine_gaussian(mean, prior, matrix, offset, noise)
    conditioned = phx.uq.condition_gaussian(
        mean,
        prior,
        output,
        observed,
        rank_tolerance=1e-10,
        support_tolerance=1e-10,
    )

    output_mean = matrix @ mean + offset
    innovation_covariance = matrix @ covariance @ matrix.T + noise_covariance
    gain = covariance @ matrix.T @ jnp.linalg.inv(innovation_covariance)
    expected_mean = mean + gain @ (observed - output_mean)
    expected_covariance = covariance - gain @ matrix @ covariance

    assert jnp.allclose(conditioned.mean, expected_mean, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(conditioned.covariance, expected_covariance, rtol=1e-7, atol=1e-7)
    assert bool(conditioned.valid)


def test_rank_aware_triangular_solve_handles_full_rank_singular_and_complex_cases():
    full = jnp.asarray([[2.0, 0.0], [1.0, 3.0]])
    right = jnp.asarray([4.0, 7.0])
    singular = jnp.asarray([[2.0, 0.0], [0.0, 0.0]])
    complex_upper = jnp.asarray([[1.0 + 0.5j, 0.2 - 0.1j], [0.0 + 0.0j, 2.0 - 0.3j]])
    complex_right = jnp.asarray([0.5 + 0.2j, -0.3j])

    assert jnp.allclose(
        phx.uq.solve_triangular_rank_aware(full, right, lower=True),
        jnp.linalg.solve(full, right),
    )
    assert jnp.allclose(
        phx.uq.solve_triangular_rank_aware(
            singular, jnp.asarray([4.0, 3.0]), rank_tolerance=1e-12
        ),
        jnp.asarray([2.0, 0.0]),
    )
    assert jnp.allclose(
        phx.uq.solve_triangular_rank_aware(
            complex_upper,
            complex_right,
            lower=False,
            conjugate_transpose=True,
        ),
        jnp.linalg.solve(jnp.conj(complex_upper.T), complex_right),
    )


def test_rank_deficient_triangular_solve_has_finite_reverse_rhs_gradient():
    singular = jnp.asarray([[2.0, 0.0], [0.0, 0.0]])

    def objective(right):
        solution = phx.uq.solve_triangular_rank_aware(
            singular,
            right,
            rank_tolerance=0.0,
        )
        return jnp.sum(solution)

    gradient = jax.grad(objective)(jnp.asarray([4.0, 3.0]))

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, jnp.asarray([0.5, 0.0]))


def test_gaussian_smoothing_regression_composition_matches_direct_moments():
    inner_noise = phx.uq.GaussianFactor(jnp.asarray([[0.3], [0.1]]))
    outer_noise = phx.uq.GaussianFactor(jnp.asarray([[0.2], [-0.15]]))
    inner = phx.uq.GaussianRegression(
        jnp.asarray([[1.0, 0.25], [-0.5, 0.75]]),
        jnp.asarray([0.2, -0.1]),
        inner_noise,
        regression_id="inner-smoothing-step",
    )
    outer = phx.uq.GaussianRegression(
        jnp.asarray([[0.8, -0.3], [0.4, 1.1]]),
        jnp.asarray([-0.2, 0.5]),
        outer_noise,
        regression_id="outer-smoothing-step",
    )
    composed = phx.uq.compose_gaussian_regressions(outer, inner)
    value = jnp.asarray([0.7, -0.4])
    evaluated = composed(value)

    expected_matrix = outer.matrix @ inner.matrix
    expected_offset = outer.matrix @ inner.offset + outer.offset
    expected_covariance = (
        outer.matrix @ inner_noise.covariance @ outer.matrix.T + outer_noise.covariance
    )

    assert jnp.allclose(composed.matrix, expected_matrix)
    assert jnp.allclose(composed.offset, expected_offset)
    assert jnp.allclose(composed.noise_factor.covariance, expected_covariance)
    assert jnp.allclose(evaluated.mean, expected_matrix @ value + expected_offset)
    assert jnp.allclose(evaluated.covariance, expected_covariance)
    assert composed.resolved_method == "independent-affine-regression-composition"
