#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_gaussian_and_student_t_likelihoods_are_finite_and_sample_shapes_match():
    gaussian = phx.uq.GaussianLikelihood(0.5)
    student = phx.uq.StudentTLikelihood(df=4.0, scale=0.75)
    location = jnp.asarray([0.0, 1.0])
    target = jnp.asarray([0.2, 0.7])

    assert jnp.all(jnp.isfinite(gaussian.log_prob(location, target)))
    assert jnp.all(jnp.isfinite(student.log_prob(location, target)))
    assert gaussian.sample(jr.key(0), location).shape == location.shape
    assert student.sample(jr.key(1), location).shape == location.shape


def test_proper_scores_match_reference_identities():
    gaussian_at_center = phx.uq.gaussian_crps(0.0, 1.0, 0.0)
    expected = (jnp.sqrt(2.0) - 1.0) / jnp.sqrt(jnp.pi)
    assert jnp.allclose(gaussian_at_center, expected)

    samples = jnp.asarray([-1.0, 0.0, 2.0, 4.0])
    target = 0.5
    brute = jnp.mean(jnp.abs(samples - target)) - 0.5 * jnp.mean(
        jnp.abs(samples[:, None] - samples[None, :])
    )
    assert jnp.allclose(phx.uq.ensemble_crps(samples, target), brute)
    assert phx.uq.student_t_crps(0.0, 1.0, 4.0, 0.2) >= 0.0


def test_metric_masks_remove_nonfinite_padding():
    width = phx.uq.interval_width(
        jnp.asarray([0.0, jnp.nan]),
        jnp.asarray([2.0, jnp.nan]),
        mask=jnp.asarray([True, False]),
    )
    assert width == 2.0


def test_gaussian_scale_calibrator_uses_closed_form_held_out_optimum():
    center = jnp.zeros((16,))
    target = jnp.full((16,), 2.0)
    scale = jnp.ones((16,))
    calibrator = phx.uq.GaussianScaleCalibrator.fit(center, scale, target)

    assert jnp.allclose(calibrator.scale_multiplier, 2.0)
    before = phx.uq.negative_log_likelihood(
        phx.uq.GaussianLikelihood(1.0), center, target
    )
    after = phx.uq.negative_log_likelihood(
        phx.uq.GaussianLikelihood(calibrator.scale_multiplier), center, target
    )
    assert after < before


def test_operator_valued_likelihood_constraint_scores_transformed_observation():
    data = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    domain = phx.domain.DatasetDomain(data)

    @domain.Function("data")
    def field(row):
        return row[0] + row[1]

    targets = 2.0 * (data[:, 0] + data[:, 1])
    constraint = phx.constraints.SupervisedLikelihoodConstraint(
        "u",
        domain.component(),
        targets,
        phx.uq.GaussianLikelihood(0.5),
        sampling=phx.domain.PointSampling(12, design="uniform"),
        observation_operator=lambda u: 2.0 * u,
    )
    loss = constraint.loss({"u": field}, key=jr.key(2))
    expected = jnp.log(0.5) + 0.5 * jnp.log(2.0 * jnp.pi)
    assert jnp.allclose(loss, expected)


def test_split_conformal_uses_exact_finite_sample_rank_and_rejects_impossible_rank():
    center = jnp.zeros((9,))
    target = jnp.arange(9.0)
    calibrator = phx.uq.SplitConformal.calibrate(center, target, alpha=0.2)
    assert calibrator.radius == 7.0

    with pytest.raises(ValueError, match="too small"):
        phx.uq.SplitConformal.calibrate(center, target, alpha=0.01)


def test_functional_conformal_aggregates_one_masked_score_per_case():
    center = jnp.zeros((9, 4))
    target = jnp.arange(9.0)[:, None] * jnp.ones((1, 4))
    mask = jnp.ones_like(target, dtype=bool).at[:, -1].set(False)
    calibrator = phx.uq.FunctionalConformal.calibrate(
        center,
        target,
        alpha=0.2,
        case_dim=0,
        mask=mask,
    )
    interval = calibrator.interval(jnp.zeros((4,)))

    assert calibrator.radius == 7.0
    assert interval.simultaneous
    assert interval.calibrated


def test_conformal_rejects_invalid_scale_axes_and_l2_box_intervals():
    center = cx.Field(jnp.zeros((9, 2)), dims=("case", "x"))
    target = cx.Field(jnp.ones((9, 2)), dims=("case", "y"))
    with pytest.raises(ValueError, match="matching shapes and dimensions"):
        phx.uq.FunctionalConformal.calibrate(center, target, alpha=0.2, case_dim="case")
    with pytest.raises(ValueError, match="out of bounds"):
        phx.uq.SplitConformal.calibrate(
            jnp.zeros((9,)), jnp.ones((9,)), alpha=0.2, case_dim=2
        )
    with pytest.raises(ValueError, match="non-negative"):
        phx.uq.NormalizedConformal.calibrate(
            jnp.zeros((9,)),
            -jnp.ones((9,)),
            jnp.ones((9,)),
            alpha=0.2,
        )
    l2 = phx.uq.FunctionalConformal.calibrate(
        jnp.zeros((9, 2)),
        jnp.ones((9, 2)),
        alpha=0.2,
        score="l2",
    )
    with pytest.raises(ValueError, match="norm ball"):
        l2.interval(jnp.zeros((2,)))


def test_three_way_split_is_disjoint_complete_and_nonempty():
    train, calibration, test = phx.data_utils.train_calibration_test_split_indices(
        17,
        calibration_fraction=0.2,
        test_fraction=0.3,
        key=jr.key(3),
    )
    all_indices = jnp.concatenate((train, calibration, test))

    assert train.size > 0 and calibration.size > 0 and test.size > 0
    assert jnp.unique(all_indices).size == 17
    assert jnp.array_equal(jnp.sort(all_indices), jnp.arange(17))
