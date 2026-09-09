import jax.numpy as jnp

from phydrax.finance.core import PhysicalLaw
from phydrax.finance.econometrics._state_space import (
    FinancialStateSpaceDefinition,
    fit_financial_state_space,
    summarize_financial_state_space,
)
from phydrax.stochastic._state_space import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    ObservationSequence,
    StateSpaceModel,
    StateSpaceProblem,
)
from phydrax.uq._state_space_inference import exact_state_space_log_likelihood


def test_financial_state_adapter_preserves_native_kalman_likelihood_and_masks():
    sequence = ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[1.0], [0.0], [2.0]]),
        observation_mask=jnp.asarray([[True], [False], [True]]),
        case_ids=("only",),
        sequence_id="finance-state-sequence",
    )
    prior = GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="finance-state-prior",
    )
    transition = LinearGaussianTransitionKernel(
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="finance-state-transition",
    )
    observation = LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = StateSpaceModel(prior, transition, observation, model_id="finance-state")
    problem = StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        problem_id="finance-state-problem",
    )
    law = PhysicalLaw("historical", "synthetic", "state-factor", "observed")

    fit = fit_financial_state_space(
        problem,
        "masked-observations",
        FinancialStateSpaceDefinition("finance-state", smooth=True),
        law,
    )
    result = summarize_financial_state_space(fit)
    native = exact_state_space_log_likelihood(problem, method="kalman")

    assert jnp.allclose(fit.log_likelihood, native.total_log_likelihood)
    assert jnp.array_equal(fit.likelihood.step_valid, native.step_valid)
    assert result.filtered_state_mean.shape == (3, 1)
    assert result.smoothed_state_mean.shape == (3, 1)
    assert fit.law_id == law.law_id
