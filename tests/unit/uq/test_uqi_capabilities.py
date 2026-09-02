#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_complex_likelihoods_are_normalized_real_and_dtype_preserving():
    circular = phx.uq.CircularComplexGaussianLikelihood(jnp.asarray(2.0))
    location = jnp.asarray([1.0 + 2.0j])
    target = jnp.asarray([2.0 + 4.0j])
    expected = -5.0 / 4.0 - jnp.log(4.0 * jnp.pi)
    assert jnp.allclose(circular.log_prob(location, target), expected)
    assert jnp.issubdtype(circular.log_prob(location, target).dtype, jnp.floating)
    assert jnp.issubdtype(
        circular.sample(jax.random.key(0), location).dtype, jnp.complexfloating
    )
    with pytest.raises(TypeError, match="complex"):
        phx.uq.GaussianLikelihood(1.0).log_prob(location, target)

    covariance = jnp.asarray([[2.0 + 0.0j, 0.3 + 0.2j], [0.3 - 0.2j, 1.0 + 0.0j]])
    pseudo = jnp.asarray([[0.1 + 0.0j, 0.02 + 0.03j], [0.02 + 0.03j, -0.05 + 0.0j]])
    dense = phx.uq.ComplexGaussianLikelihood.from_covariances(
        covariance,
        pseudo,
        hermitian_tolerance=1e-12,
        symmetry_tolerance=1e-12,
    )
    value = dense.log_prob(jnp.zeros((2,), dtype=complex), jnp.zeros((2,), dtype=complex))
    assert value.shape == ()
    assert jnp.isfinite(value)


def test_likelihood_batch_uses_explicit_nonuniform_estimator_weights():
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.MinibatchPosteriorProblem(
        space,
        lambda physical, batch: -0.5 * (batch.data - physical) ** 2,
        num_factors=4,
    )
    batch = phx.uq.LikelihoodBatch(
        jnp.asarray([1.0, 3.0, 1.0]),
        jnp.asarray([True, True, True]),
        factor_ids=jnp.asarray([0, 2, 0]),
        sampling_probabilities=jnp.asarray([0.2, 0.5, 0.2]),
        estimator_weights=jnp.asarray([1.0 / 0.6, 1.0 / 1.5, 1.0 / 0.6]),
    )
    factors = problem.log_likelihood_factors(jnp.asarray(0.0), batch)
    assert jnp.allclose(
        problem.log_likelihood_estimate(jnp.asarray(0.0), batch),
        jnp.sum(batch.estimator_weights * factors),
    )


def test_residual_penalty_noise_mapping_matches_real_and_complex_quadratics():
    real = phx.uq.ResidualPenaltyNoiseModel(
        coefficients=jnp.asarray([0.25, 0.0, 2.0]),
        penalty_scale=3.0,
        field="real",
        interpretation_id="real-test",
    )
    complex_model = phx.uq.ResidualPenaltyNoiseModel(
        coefficients=jnp.asarray([0.25, 0.0, 2.0]),
        penalty_scale=3.0,
        field="proper_complex",
        interpretation_id="complex-test",
    )
    assert jnp.allclose(real.variance, 1.0 / (6.0 * jnp.asarray([0.25, 2.0])))
    assert jnp.allclose(complex_model.variance, 1.0 / (3.0 * jnp.asarray([0.25, 2.0])))
    assert jnp.array_equal(real.active_indices, jnp.asarray([0, 2]))


def test_mc_dropout_calibration_matches_closed_form_scale_and_conformal_rank():
    samples = cx.Field(
        jnp.asarray(
            [
                [0.0, 1.0, 2.0],
                [2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        dims=("draw", "case"),
    )
    predictive = phx.uq.PredictiveField(
        samples,
        (phx.uq.SampleAxis("draw", "epistemic"),),
    )
    target = cx.Field(jnp.asarray([2.0, 0.0, 4.0]), dims=("case",))
    scale = phx.uq.MCDropoutCalibration.fit(
        predictive,
        target,
        nominal_coverage=0.8,
        method="gaussian_scale",
        split_identity="heldout-A",
    )
    center = predictive.mean(sources="epistemic").data
    std = predictive.std(sources="epistemic").data
    expected = jnp.sqrt(jnp.mean(((target.data - center) / std) ** 2))
    assert jnp.allclose(scale.coefficient, expected)
    conformal = phx.uq.MCDropoutCalibration.fit(
        predictive,
        target,
        nominal_coverage=0.5,
        method="normalized_conformal",
        split_identity="heldout-A",
    )
    assert conformal.evidence.calibration_count == 3
    assert conformal.interval(predictive).calibrated


def test_mc_dropout_functional_coverage_requires_every_active_point_per_case():
    samples = cx.Field(
        jnp.stack((-jnp.ones((4, 2)), jnp.ones((4, 2)))),
        dims=("draw", "case", "x"),
    )
    predictive = phx.uq.PredictiveField(
        samples,
        (phx.uq.SampleAxis("draw", "epistemic"),),
    )
    target = cx.Field(
        jnp.asarray(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
                [3.0, 0.0],
            ]
        ),
        dims=("case", "x"),
    )
    calibration = phx.uq.MCDropoutCalibration.fit(
        predictive,
        target,
        nominal_coverage=0.5,
        method="functional_conformal",
        split_identity="functional-heldout",
        case_dim="case",
    )
    assert jnp.allclose(calibration.coefficient, 2.0)
    assert jnp.allclose(calibration.evidence.empirical_heldout_coverage, 0.75)


def test_swag_welford_ring_and_sampling_are_fixed_capacity():
    state = phx.uq.SWAGState.initialize(
        jnp.zeros((2,)),
        snapshot_capacity=2,
        parameter_paths=("[0]",),
        accumulation_precision=jnp.float64,
    )
    state = phx.uq.update_swag_state(state, jnp.asarray([1.0, 2.0]), solver_step=1)
    state = phx.uq.update_swag_state(state, jnp.asarray([3.0, 4.0]), solver_step=2)
    state = phx.uq.update_swag_state(state, jnp.asarray([5.0, 6.0]), solver_step=3)
    assert state.count == 3
    assert state.active_snapshot_count == 2
    assert jnp.allclose(state.mean, jnp.asarray([3.0, 4.0]))
    draw = phx.uq.sample_swag_vector(state, jax.random.key(1))
    assert draw.shape == (2,)
    assert jnp.all(jnp.isfinite(draw))


def test_svgp_kl_and_step_schedules_have_direct_references():
    state = phx.uq.SparseVariationalGaussianState(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([0.5, -0.25]),
        jnp.zeros((2, 2)),
    )
    assert jnp.allclose(state.kl_standard_normal, 0.5 * (0.5**2 + 0.25**2))
    constant = phx.uq.SGMCMCStepSchedule.constant(0.1)
    polynomial = phx.uq.SGMCMCStepSchedule.polynomial(0.2, 10.0, 0.75)
    assert constant(100) == 0.1
    assert polynomial(2) < polynomial(1)
    with pytest.raises(ValueError, match="1/2"):
        phx.uq.SGMCMCStepSchedule.polynomial(0.2, 10.0, 0.5)


def test_structured_kinetic_actions_and_momentum_are_finite():
    reference = {"a": jnp.zeros((2,)), "b": jnp.zeros((1,))}
    diagonal = phx.uq.prepare_mcmc_kinetic(
        reference,
        phx.uq.MCMCMassAdaptationPlan.diagonal(),
        diagonal=jnp.asarray([1.0, 2.0, 3.0]),
    )
    value = jnp.asarray([1.0, -1.0, 2.0])
    assert jnp.allclose(
        diagonal.inverse_mass_action_vector(value),
        jnp.asarray([1.0, -2.0, 6.0]),
    )
    low_rank = phx.uq.prepare_mcmc_kinetic(
        reference,
        phx.uq.MCMCMassAdaptationPlan.diagonal_low_rank(1),
        diagonal=jnp.ones((3,)),
        low_rank_factor=jnp.asarray([[1.0], [0.0], [0.5]]),
    )
    momentum = low_rank.sample_momentum_vector(jax.random.key(2))
    assert momentum.shape == (3,)
    assert jnp.isfinite(low_rank.kinetic_energy_vector(momentum))


def test_nested_periodic_and_phantom_state_preserve_bounded_semantics():
    periodic = phx.uq.PeriodicNestedCoordinate("angle", -jnp.pi, 2.0 * jnp.pi)
    assert jnp.allclose(periodic.wrap(3.0 * jnp.pi), -jnp.pi)
    assert jnp.allclose(periodic.displacement(0.9 * jnp.pi, -0.9 * jnp.pi), 0.2 * jnp.pi)
    phantom = phx.uq.PhantomNestedState.initialize(2, 1, dtype=jnp.float64)
    phantom = phantom.add(
        jnp.asarray([0.5]),
        log_likelihood=2.0,
        birth_log_likelihood=0.0,
        proposal_epoch=1,
        ancestry=3,
    )
    assert jnp.array_equal(phantom.eligible(1.0), jnp.asarray([True, False]))
    capacity = phx.uq.NestedSamplingCapacity(
        max_live=20,
        max_dead_points=100,
        max_likelihood_evaluations=1000,
        max_dynamic_batches=4,
        max_clusters=3,
        max_phantoms=10,
    )
    prior = phx.uq.NestedPriorPlan(
        continuous_paths=("angle",),
        periodic=(periodic,),
    )
    proposal = phx.uq.NestedProposalPlan(periodic_slice=True)
    plan = phx.uq.NestedSamplingPlan(
        capacity,
        prior,
        proposal,
        initial_live=10,
        dynamic=phx.uq.DynamicNestedPolicy(
            pilot_dead_points=10,
            additional_live_per_batch=2,
            allocation_cadence=5,
        ),
    )
    assert plan.initial_live == 10


def test_structured_hmc_and_causal_nuts_execute_production_routes():
    space = phx.uq.ParameterSpace(
        jnp.zeros((2,)), priors=phx.uq.Normal(jnp.zeros((2,)), jnp.ones((2,)))
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * (4.0 * value[0] ** 2 + (value[1] - 0.5 * value[0]) ** 2),
    )
    blocks = phx.uq.MCMCMassAdaptationPlan.blocks(
        (("",),), max_block_size=2, memory_cap_bytes=4096
    )
    hmc = phx.uq.sample_hmc(
        problem,
        key=jax.random.key(91),
        num_integration_steps=2,
        num_chains=2,
        num_warmup=4,
        num_samples=4,
        kinetic=blocks,
    )
    assert hmc.samples.shape == (2, 4, 2)
    causal = phx.uq.sample_nuts(
        problem,
        key=jax.random.key(92),
        num_chains=2,
        num_warmup=4,
        num_samples=4,
        max_num_doublings=2,
        kinetic=blocks,
        trajectory="causal",
        causal_config=phx.uq.CausalNUTSConfig(
            max_num_doublings=2,
            recurrence=phx.uq.CausalHMCConfig(
                linearization="dense-exact",
                maximum_outer_iterations=16,
            ),
        ),
    )
    assert causal.trajectory_method == "causal"
    assert jnp.all(causal.causal_diagnostics.converged)


def test_prepared_nested_production_dynamic_phantom_and_proposal_lifecycle():
    space = phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(space, lambda value: -0.5 * (value - 1.0) ** 2)
    capacity = phx.uq.NestedSamplingCapacity(
        max_live=8,
        max_dead_points=12,
        max_likelihood_evaluations=256,
        max_dynamic_batches=2,
        max_clusters=2,
        max_phantoms=4,
    )
    prior = phx.uq.NestedPriorPlan(continuous_paths=("<root>",))
    proposal = phx.uq.NestedProposalPlan(
        ellipsoid=True,
        phantom_recycling=True,
        learned_flow=True,
        gradient_guided=True,
        maximum_attempts=32,
    )
    plan = phx.uq.NestedSamplingPlan(
        capacity,
        prior,
        proposal,
        initial_live=4,
        dynamic=phx.uq.DynamicNestedPolicy(
            pilot_dead_points=2,
            additional_live_per_batch=1,
            allocation_cadence=2,
        ),
    )
    result = phx.uq.sample_nested(
        problem,
        key=jax.random.key(93),
        plan=plan,
        remaining_evidence_tolerance=0.9,
    )
    assert isinstance(result.final_state, phx.uq.PreparedNestedState)
    assert result.num_live <= capacity.max_live
    assert result.num_dead <= capacity.max_dead_points
    assert result.num_likelihood_evaluations <= capacity.max_likelihood_evaluations
    assert jnp.isfinite(result.log_evidence)
    assert result.final_state.adaptation.ellipsoid_attempts > 0
    assert result.final_state.adaptation.flow_attempts > 0
    assert result.final_state.adaptation.gradient_attempts > 0
