#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.bioinformatics.omics._assay import (
    ContinuousAssay,
    CountAssay,
    IMPLICIT_OBSERVED_ZERO,
)
from phydrax.bioinformatics.omics._count_models import (
    fit_negative_binomial_glm,
    GLM_ALL_ZERO,
    negative_binomial_log_probability,
)
from phydrax.bioinformatics.omics._design import (
    build_experimental_design,
    DESIGN_RANK_DEFICIENT,
    ExperimentalDesign,
    pairwise_condition_contrast,
    TERM_INTERACTION,
    TERM_NESTED_BATCH,
)
from phydrax.bioinformatics.omics._differential import (
    likelihood_ratio_test,
    TEST_INVALID_FIT,
    wald_test,
)
from phydrax.bioinformatics.omics._dispersion import (
    DISPERSION_BOUNDARY,
    estimate_feature_dispersion,
    fit_dispersion_trend,
    shrink_dispersion,
)
from phydrax.bioinformatics.omics._multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
)
from phydrax.bioinformatics.omics._normalization import (
    library_size_normalization,
    median_ratio_normalization,
)
from phydrax.bioinformatics.omics._pseudobulk import pseudobulk_counts


def test_dense_and_fixed_sparse_assays_preserve_three_cell_states():
    dense = CountAssay(
        jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        missing=jnp.array([[False, True, False], [False, True, False]]),
        structural_absence=jnp.array([[False, False, True], [False, False, True]]),
    )
    sparse = CountAssay.from_fixed_sparse(
        jnp.array([[0, 1, 2], [0, 1, 2]]),
        jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        num_features=3,
        missing=jnp.array([[False, True, False], [False, True, False]]),
        structural_absence=jnp.array([[False, False, True], [False, False, True]]),
        implicit_state=IMPLICIT_OBSERVED_ZERO,
    )
    assert jnp.array_equal(sparse.dense_values, dense.dense_values)
    assert jnp.array_equal(sparse.observed_mask, dense.observed_mask)
    assert jnp.array_equal(sparse.structural_absence_mask, dense.structural_absence_mask)
    assert jnp.array_equal(sparse.missing_mask, dense.missing_mask)
    assert jnp.array_equal(
        sparse.zero_mask,
        jnp.array([[True, False, False], [False, False, False]]),
    )

    continuous = ContinuousAssay(
        jnp.array([[0.0, jnp.nan, 3.0]]),
        structural_absence=jnp.array([[False, False, True]]),
    )
    continuous_sparse = ContinuousAssay.from_fixed_sparse(
        jnp.array([[1, 2]]),
        jnp.array([[0.0, 0.0]]),
        num_features=3,
        missing=jnp.array([[True, False]]),
        structural_absence=jnp.array([[False, True]]),
    )
    assert jnp.array_equal(continuous_sparse.observed_mask, continuous.observed_mask)
    assert jnp.array_equal(continuous_sparse.zero_mask, jnp.array([[True, False, False]]))


def test_fixed_sparse_capacity_rejects_duplicate_routes():
    with pytest.raises(ValueError, match="unique"):
        CountAssay.from_fixed_sparse(
            jnp.array([[1, 1]]),
            jnp.array([[2, 3]]),
            num_features=3,
        )


def test_pseudobulk_sums_technical_counts_without_conflating_missingness():
    assay = CountAssay(
        jnp.array(
            [
                [2, 0, 0],
                [3, 5, 0],
                [7, 0, 0],
                [11, 0, 0],
            ]
        ),
        missing=jnp.array(
            [
                [False, True, False],
                [False, False, False],
                [False, True, False],
                [False, True, False],
            ]
        ),
        structural_absence=jnp.array(
            [
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
            ]
        ),
    )
    result = pseudobulk_counts(assay, jnp.array([0, 0, 1, 1]), num_units=2)
    assert jnp.array_equal(result.assay.dense_values[:, 0], jnp.array([5, 18]))
    assert result.assay.dense_values[0, 1] == 5
    assert result.assay.observed_mask[0, 1]
    assert result.assay.missing_mask[1, 1]
    assert jnp.all(result.assay.structural_absence_mask[:, 2])
    assert jnp.array_equal(result.contributing_cells, jnp.array([2, 2]))


def test_paired_nested_and_interaction_designs_are_explicit_and_estimable():
    paired = build_experimental_design(
        jnp.array([0, 1, 0, 1, 0, 1]),
        num_conditions=2,
        donor_indices=jnp.array([0, 0, 1, 1, 2, 2]),
        num_donors=3,
        paired=True,
    )
    contrast = pairwise_condition_contrast(paired, 1, 0)
    assert paired.valid
    assert contrast.estimable
    assert paired.rank == 4

    condition = jnp.tile(jnp.array([0, 1]), 4)
    donor = jnp.repeat(jnp.array([0, 0, 1, 1]), 2)
    batch = jnp.repeat(jnp.array([0, 1, 0, 1]), 2)
    nested = build_experimental_design(
        condition,
        num_conditions=2,
        donor_indices=donor,
        num_donors=2,
        batch_indices=batch,
        num_batches=2,
        batch_nested_in_donor=True,
    )
    assert nested.valid
    assert jnp.sum(nested.coefficient_terms == TERM_NESTED_BATCH) == 2

    interaction = build_experimental_design(
        condition,
        num_conditions=2,
        batch_indices=jnp.tile(jnp.array([0, 0, 1, 1]), 2),
        num_batches=2,
        include_batch=True,
        condition_batch_interaction=True,
    )
    assert interaction.valid
    assert jnp.sum(interaction.coefficient_terms == TERM_INTERACTION) == 1


def test_rank_deficiency_is_observable_and_contrast_is_not_overclaimed():
    design = ExperimentalDesign(
        jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    assert not design.valid
    assert design.status == DESIGN_RANK_DEFICIENT


def test_depth_and_composition_normalizations_make_distinct_claims():
    assay = CountAssay(jnp.array([[10, 10, 10, 10], [20, 20, 20, 2000]]))
    depth = library_size_normalization(assay)
    robust = median_ratio_normalization(assay)
    assert depth.size_factors[1] / depth.size_factors[0] > 20.0
    assert robust.size_factors[1] / robust.size_factors[0] == pytest.approx(
        2.0, rel=1.0e-5
    )
    assert jnp.all(robust.valid)


def test_dispersion_estimation_marks_all_zero_low_count_and_boundaries():
    assay = CountAssay(
        jnp.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 0, 2, 20],
                [0, 1, 1, 1],
                [0, 0, 2, 20],
                [0, 1, 1, 1],
            ]
        )
    )
    estimates = estimate_feature_dispersion(assay, minimum=1.0e-6, maximum=1.0)
    assert not estimates.valid[0]
    assert estimates.valid[1]
    assert estimates.status[1] == DISPERSION_BOUNDARY
    assert estimates.at_lower_bound[1]
    assert estimates.at_upper_bound[3]
    trend = fit_dispersion_trend(estimates)
    shrunk = shrink_dispersion(estimates, trend, prior_degrees_of_freedom=4.0)
    assert jnp.all(
        (shrunk.dispersion >= estimates.minimum)
        & (shrunk.dispersion <= estimates.maximum)
    )


def test_nb_likelihood_has_finite_poisson_and_small_dispersion_boundaries():
    counts = jnp.array([0.0, 1.0, 8.0])
    mean = jnp.array([1.0, 1.0, 8.0])
    poisson = negative_binomial_log_probability(counts, mean, 0.0)
    near_poisson = negative_binomial_log_probability(counts, mean, 1.0e-6)
    assert poisson[0] == pytest.approx(-1.0)
    assert jnp.all(jnp.isfinite(near_poisson))
    assert jnp.max(jnp.abs(poisson - near_poisson)) < 1.0e-3


def test_glm_handles_all_zero_and_low_count_features_with_offsets():
    assay = CountAssay(
        jnp.array(
            [
                [0, 0],
                [1, 0],
                [0, 0],
                [2, 0],
                [1, 0],
                [2, 0],
            ]
        )
    )
    design = ExperimentalDesign(jnp.ones((6, 1)))
    fit = fit_negative_binomial_glm(
        assay,
        design,
        jnp.array([0.2, 0.2]),
        offsets=jnp.log(jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])),
        maximum_steps=80,
    )
    assert fit.valid[0]
    assert fit.status[1] == GLM_ALL_ZERO
    assert not fit.valid[1]


def test_glm_rejects_nonfinite_offsets_on_observed_rows():
    assay = CountAssay(jnp.array([[2], [3], [4]]))
    design = ExperimentalDesign(jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="offsets must be finite"):
        fit_negative_binomial_glm(
            assay,
            design,
            0.1,
            offsets=jnp.asarray((0.0, jnp.nan, 0.0)),
        )


def test_wald_and_lrt_null_tails_and_p_values_are_nondifferentiable():
    assay = CountAssay(jnp.array([[10], [11], [9], [10], [11], [9]]))
    full_design = build_experimental_design(
        jnp.array([0, 0, 0, 1, 1, 1]), num_conditions=2
    )
    reduced_design = ExperimentalDesign(jnp.ones((6, 1)))
    full = fit_negative_binomial_glm(assay, full_design, 0.1, maximum_steps=100)
    reduced = fit_negative_binomial_glm(assay, reduced_design, 0.1, maximum_steps=100)
    contrast = pairwise_condition_contrast(full_design, 1, 0)
    wald = wald_test(full, contrast)
    lrt = likelihood_ratio_test(full, reduced)
    assert wald.valid[0]
    assert wald.p_value[0] > 0.9
    assert lrt.valid[0]
    assert lrt.p_value[0] > 0.9
    different_assay = CountAssay(jnp.array([[10], [11], [9], [10], [11], [10]]))
    different = fit_negative_binomial_glm(
        different_assay,
        reduced_design,
        0.1,
        maximum_steps=100,
    )
    with pytest.raises(ValueError, match="identical counts"):
        likelihood_ratio_test(full, different)

    gradient = jax.grad(
        lambda value: jnp.nansum(
            benjamini_hochberg(value, jnp.array([True, True])).adjusted_p_values
        )
    )(jnp.array([0.2, 0.8]))
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


def test_explicit_bh_by_family_excludes_untested_values_and_preserves_tails():
    p_values = jnp.array([0.01, 0.04, 0.03, 0.5, 1.0])
    tested = jnp.array([True, True, True, False, True])
    bh = benjamini_hochberg(p_values, tested)
    by = benjamini_yekutieli(p_values, tested)
    assert bh.family_size == 4
    assert jnp.isnan(bh.adjusted_p_values[3])
    assert bh.adjusted_p_values[4] == 1.0
    assert jnp.all(by.adjusted_p_values[tested] >= bh.adjusted_p_values[tested])


def test_technical_cell_duplication_does_not_increase_biological_residual_df():
    cell_assay = CountAssay(jnp.array([[3], [4], [5], [6], [7], [8]]))
    pseudobulk = pseudobulk_counts(
        cell_assay,
        jnp.array([0, 0, 1, 1, 2, 2]),
        num_units=3,
    )
    biological_design = ExperimentalDesign(jnp.ones((3, 1)))
    fit = fit_negative_binomial_glm(
        pseudobulk.assay,
        biological_design,
        0.1,
        maximum_steps=80,
    )
    assert fit.sample_count[0] == 3
    assert fit.residual_degrees_of_freedom[0] == 2

    duplicated_column_design = ExperimentalDesign(
        jnp.column_stack((jnp.ones(3), jnp.ones(3)))
    )
    duplicated = fit_negative_binomial_glm(
        pseudobulk.assay,
        duplicated_column_design,
        0.1,
        maximum_steps=80,
    )
    comparison = likelihood_ratio_test(duplicated, fit)
    assert comparison.degrees_of_freedom[0] == 0
    assert comparison.status[0] == TEST_INVALID_FIT
