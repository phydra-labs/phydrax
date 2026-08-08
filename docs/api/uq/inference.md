# Bayesian inference and ensembles

## Posterior problems

::: phydrax.uq.ParameterSpace
    options:
        members:
            - __init__
            - constrain
            - unconstrain
            - log_abs_det_jacobian
            - log_prior
            - unconstrained_log_prior
            - sample_prior

---

::: phydrax.uq.ParameterSubspace
    options:
        members:
            - __init__
            - reconstruct
            - array_leaf_paths
            - from_leaf_paths
            - from_subtree_paths
            - last_layer

---

::: phydrax.uq.PosteriorProblem
    options:
        members:
            - __init__
            - log_likelihood
            - log_density
            - negative_log_density
            - predict
            - conditional_observation_variance
            - sample_observation
            - gauss_newton_residual
            - validate


### Stochastic likelihood contracts

`MinibatchPosteriorProblem` requires one normalized likelihood contribution per
statistical factor. `MinibatchSource` epochs must cover every factor exactly once;
padded entries are excluded by `LikelihoodBatch.factor_mask`. The stochastic
estimate scales the active-factor mean by the declared population size and adds the
prior and bijector Jacobian exactly once.

For operator data, a factor is one complete physical case. Query points, channels,
geometry, masks, and quadrature remain inside that factor and are never interpreted
as independent observations.

::: phydrax.uq.LikelihoodBatch
    options:
        members:
            - __init__
            - capacity
            - factor_count

---

::: phydrax.uq.MinibatchSource

---

::: phydrax.uq.ArrayMinibatchSource
    options:
        members:
            - __init__
            - num_factors
            - batch_capacity
            - batches_per_epoch
            - fingerprint
            - configuration
            - epoch

---

::: phydrax.uq.MinibatchPosteriorProblem
    options:
        members:
            - __init__
            - log_likelihood_factors
            - log_likelihood_estimate
            - log_density_estimate
            - full_log_likelihood
            - full_log_density
            - predict
            - conditional_observation_variance
            - sample_observation
            - validate

---

::: phydrax.uq.diagnose_minibatch_posterior

---

::: phydrax.uq.MinibatchPosteriorCapabilities
    options:
        members:
            - as_dict

---

::: phydrax.uq.MinibatchPosteriorDiagnostics
    options:
        members:
            - passed
            - as_dict


### Posterior inspection

::: phydrax.uq.diagnose_posterior

---

::: phydrax.uq.PosteriorCapabilities
    options:
        members:
            - as_dict

---

::: phydrax.uq.PosteriorDiagnostics
    options:
        members:
            - passed
            - as_dict


### Normalized posterior terms

::: phydrax.uq.FixedObservationLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob

---

::: phydrax.uq.FixedResidualLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob

---

::: phydrax.uq.GaussianProcessMarginalLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob

---

::: phydrax.uq.FixedSupervisedLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob

---

::: phydrax.uq.CompositePosteriorLikelihood
    options:
        members:
            - __init__
            - log_prob


### Linearized errors in variables

`LinearizedGaussianMeasurementLikelihood` is a normalized joint Gaussian
likelihood for uncertain predictors and uncertain observations. For case \(i\),
it evaluates the physical prediction \(f(\theta, x_i)\), pushes the declared
input covariance through the local input derivative \(J_i\), and uses

\[
\Sigma_i(\theta)
= \Sigma_{y,i}(\theta)
+ J_i(\theta)\Sigma_{x,i}(\theta)J_i(\theta)^\mathsf{T}.
\]

The log determinant of \(\Sigma_i(\theta)\) is part of every factor. Omitting it
changes the posterior whenever the effective covariance depends on parameters.
Input and observation covariance may be fixed arrays or parameter-dependent
callbacks. Both shared and explicit per-case batching are supported. Output
events are intentionally bounded by `max_output_dimension`, because normalized
dense Gaussian factors require a Cholesky factorization.

`log_prob_cases(...)` evaluates the same term on an external deterministic
minibatch. Put measured inputs, targets, and original case indices in an
`ArrayMinibatchSource`; use `log_prob_cases` as the
`MinibatchPosteriorProblem` factor callable. This reuses the full-data
likelihood exactly rather than defining a second stochastic objective.

::: phydrax.uq.LinearizedGaussianMeasurementLikelihood
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob_cases
            - log_prob

---


## State-space completion and estimation

### Exact finite-state completion

`exact_state_space_log_likelihood(..., method="finite-state")` returns an
`ExactStateSpaceLikelihood` whose `backend` is the `FiniteStateFilterResult` consumed
below. Exact completion requires a `CategoricalStatePrior` aligned with a closed
`FiniteStateTransitionKernel`; it never substitutes another backend or repairs a
failed transition.

`finite_state_backward_smoother` returns fixed-interval state marginals and
`transition_probabilities[..., step, i, j]`, the posterior mass from state `i` at the
physical interval's start to state `j` at its endpoint. Step zero starts at
`problem.initial_time`. A padded interval has exactly zero transition mass and
contributes zero to counts and sufficient statistics. An observation channel masked
by `observation_mask` contributes no emission likelihood, while `step_valid` excludes
the entire padded schedule entry. Physical case axes and IDs are never pooled.

`finite_state_viterbi` returns the maximum joint path, including the state before the
first observation. A zero prior or transition probability is exact negative infinity,
not an epsilon floor; deterministic ties choose the lowest state index.
`finite_state_expected_transition_counts` sums adjacent-pair posterior mass per case
and over cases. `finite_state_expected_sufficient_statistics` evaluates
`statistic(previous_state, state, t0, t1, context)` for each state pair, with the
canonical context in the final positional slot, and preserves an arbitrary
shape-stable PyTree at per-step, per-case, and total reductions.

Every result retains `step_valid`, algorithm `valid` and `status`, and its source
result. The finite-state records also retain `input_id`; model, problem, sequence,
process, approximation, execution, case-ID, and case-shape provenance remains
reachable through the nested filter result. Inspect validity before reducing:
padding is inactive capacity, not a successful inferred transition.

::: phydrax.uq.exact_state_space_log_likelihood

---

::: phydrax.uq.ExactStateSpaceLikelihood

---

::: phydrax.uq.FiniteStateFilterResult

---

::: phydrax.uq.finite_state_backward_smoother

---

::: phydrax.uq.FiniteStateSmootherResult

---

::: phydrax.uq.finite_state_viterbi

---

::: phydrax.uq.FiniteStateViterbiResult

---

::: phydrax.uq.finite_state_expected_transition_counts

---

::: phydrax.uq.FiniteStateTransitionCountResult

---

::: phydrax.uq.finite_state_expected_sufficient_statistics

---

::: phydrax.uq.FiniteStateSufficientStatisticsResult

### Full particle smoothing and transition information

`full_particle_smoother` traces the complete realized resampling genealogy from each
case's last active step. It needs no transition density, but its empirical full-interval
marginals can exhibit genealogical collapse. `fixed_lag_particle_smoother` instead
conditions through at most the declared number of future steps; it is a bounded-lag
genealogical approximation, not a full smoother.

`particle_backward_smoother` is the density-based full FFBSm alternative. It requires
the transition kernel to expose a normalized `log_prob`; a sampler-only transition is
rejected rather than approximated by a hidden density. It stores normalized backward
kernels and adjacent-particle pair weights. `particle_backward_simulation` uses those
kernels to draw coherent FFBSi paths and retains the sampled particle indices.
Semantic keys include the physical case ID, step, and sample member, so changes to an
unrelated case do not renumber a path.

All three full-result families preserve physical case axes and IDs, schedule masks and
padding, per-step validity/status, model/problem/sequence/input IDs, and method
provenance. Density-based results additionally retain transition `process_id` and
`approximation_id`; genealogical results retain resampling method and policy. Padded
path slots remain invalid rather than becoming observations. Discrete genealogy and
sampled particle indices are nondifferentiable and explicitly use
`ancestry_gradient="stop"`; Phydrax does not provide a hidden differentiable ancestry
surrogate.

`particle_fisher_score` differentiates the stored normalized transition log density
under stop-gradient FFBSm pair weights. It estimates the transition contribution only,
not prior or observation scores, and returns both the transition-parameter PyTree and
flat physical-case scores. `particle_fisher_information` averages the outer products
of valid physical-case score vectors. Both retain the source smoother and its
process, approximation, model, problem, sequence, and input provenance. These are
particle approximations, not exact observed information.

::: phydrax.uq.full_particle_smoother

---

::: phydrax.uq.ParticleSmootherResult

---

::: phydrax.uq.fixed_lag_particle_smoother

---

::: phydrax.uq.FixedLagParticleSmootherResult

---

::: phydrax.uq.particle_backward_smoother

---

::: phydrax.uq.ParticleBackwardSmootherResult

---

::: phydrax.uq.particle_backward_simulation

---

::: phydrax.uq.ParticleBackwardSimulationResult

---

::: phydrax.uq.particle_fisher_score

---

::: phydrax.uq.ParticleFisherScoreResult

---

::: phydrax.uq.particle_fisher_information

---

::: phydrax.uq.ParticleFisherInformationResult

### Multi-experiment state-space estimation

`StateSpaceExperiment` binds one parameterized problem builder to a named experiment
and an explicit physical case contract. Every evaluation must reproduce the declared
`case_axes`, `case_shape`, and unique `case_ids`. Its exact path selects Kalman or
finite-state likelihood explicitly; a custom particle, guided-particle, or ensemble
likelihood must carry a non-exact `likelihood_id`. Custom likelihoods default to
`transform_safe=False`, so local MAP, global-then-local MAP, and Laplace reject them
before tracing. Set `transform_safe=True` only when the callback and retained backend
are JAX-transform-safe; this is an explicit caller assertion, not an inferred fallback.
Derivative-free global MAP and direct likelihood evaluation remain available otherwise.
There is no silent backend fallback.

`MultiExperimentStateSpaceLikelihood` evaluates each experiment independently, sums
all physical-case log likelihoods, and returns both the flattened per-case vector and
each untouched `ExperimentStateSpaceLikelihood`. Native schedule padding, observation
masks, validity, status, case layout, backend diagnostics, and `input_id` therefore
remain attached to the experiment that produced them. Each record also names the
likelihood and resolved temporal method, approximation, model/problem/sequence IDs,
model and observation discretizations, and any covariance regularization. Experiment
IDs qualify case IDs, so differently shaped experiments compose without treating a
padded slot as data or colliding identities.

`StateSpaceEstimation` places that likelihood into one `PosteriorProblem`; the
parameter prior and bijector Jacobian are added once, while each experiment likelihood
is added once. Its local/global MAP, Laplace, and sampler methods reuse the existing
algorithm results and attach likelihood diagnostics at the selected reference point.
Global initialization remains bounded stochastic search and is not a guarantee of a
global optimum. Failures and approximation provenance are returned, not repaired.

::: phydrax.uq.StateSpaceExperiment

---

::: phydrax.uq.ExperimentStateSpaceLikelihood

---

::: phydrax.uq.MultiExperimentStateSpaceLikelihood

---

::: phydrax.uq.MultiExperimentStateSpaceLikelihoodResult

---

::: phydrax.uq.StateSpaceEstimation

---

::: phydrax.uq.StateSpaceMAPWorkflowResult

---

::: phydrax.uq.StateSpaceLaplaceWorkflowResult

---

The exported composition typing contracts are
`ApproximateStateSpaceLikelihoodResult`, `StateSpaceLikelihoodBackend`,
`StateSpaceLikelihoodFunction`, and `StateSpaceSampler`.

::: phydrax.uq.ApproximateStateSpaceLikelihoodResult

---


::: phydrax.uq.StateSpaceSamplingWorkflowResult

## MAP estimation

`search_map` performs bounded stochastic global initialization in unconstrained
posterior-position coordinates. It evaluates the complete
`PosteriorProblem.negative_log_density`, preserves the population and exact
accounting, and never interprets that population as posterior samples. Local
stationarity remains a separate `find_map` phase.

::: phydrax.uq.search_map

---

::: phydrax.uq.MAPSearchResult

---

::: phydrax.uq.find_map

---

::: phydrax.uq.MAPResult
    options:
        members:
            - compilation_seconds
            - execution_seconds
            - mean_step_seconds

---

::: phydrax.uq.MAPConvergenceError

### Parameter bijectors

::: phydrax.uq.IdentityBijector

---

::: phydrax.uq.ExpBijector

---

::: phydrax.uq.SigmoidIntervalBijector
    options:
        members:
            - __init__

## MCMC

::: phydrax.uq.sample_nuts

---

::: phydrax.uq.sample_hmc

---

### Fixed-step stochastic-gradient MCMC

`sample_sgld` and `sample_sgnht` consume deterministic `MinibatchSource` epochs
through a shared fixed-step runtime. They return approximate, unadjusted production
draws: burn-in is discarded, but no Metropolis correction or automatic step-size
adaptation is implied. Report step-size sensitivity and compare with NUTS or Laplace
on a tractable reference before interpreting these draws quantitatively.

::: phydrax.uq.sample_sgld

---

::: phydrax.uq.sample_sgnht

---

::: phydrax.uq.build_sgmcmc_control_variate

---

::: phydrax.uq.SGMCMCControlVariate

---

::: phydrax.uq.SGMCMCResult
    options:
        members:
            - num_chains
            - num_draws
            - batch_fraction
            - source_configuration
            - predict
            - predict_observations
            - mixing_report

---

::: phydrax.uq.SGMCMCDiagnostics

---

::: phydrax.uq.SGMCMCMixingThresholds
    options:
        members:
            - __init__

---

::: phydrax.uq.SGMCMCMixingReport
    options:
        members:
            - raise_for_failure
            - as_dict

---

::: phydrax.uq.SGMCMCMixingError



### Flow-assisted NUTS

::: phydrax.uq.sample_flow_nuts

---

::: phydrax.uq.FlowNUTSConfig
    options:
        members:
            - __init__
            - as_dict

---

::: phydrax.uq.FlowNUTSResult
    options:
        members:
            - predict
            - predict_observations
            - convergence_report

---

::: phydrax.uq.MCMCResult
    options:
        members:
            - predict
            - predict_observations
            - convergence_report


### Posterior compression

::: phydrax.uq.thin_posterior

---

::: phydrax.uq.SteinThinning

---

::: phydrax.uq.PosteriorCoreset
    options:
        members:
            - predict
            - predict_observations
---

::: phydrax.uq.MCMCChainWarmup

---

::: phydrax.uq.MCMCDiagnostics

::: phydrax.uq.MCMCConvergenceThresholds

---

::: phydrax.uq.MCMCConvergenceReport
    options:
        members:
            - raise_on_failure
            - as_dict

---

::: phydrax.uq.MCMCConvergenceError

## Laplace approximation

::: phydrax.uq.fit_laplace

---

::: phydrax.uq.LaplaceResult
    options:
        members:
            - sample_unconstrained
            - sample
            - predict
            - linearized_predict
            - physical_covariance
            - physical_correlation
            - predict_observations

---

::: phydrax.uq.StructuredLaplaceResult
    options:
        members:
            - covariance_vector_product
            - physical_covariance_vector_product
            - sample_unconstrained
            - sample
            - predict
            - linearized_predict
            - predict_observations

---

::: phydrax.uq.LaplaceCurvatureError

::: phydrax.uq.GaussianPriorWhitening
    options:
        members:
            - from_parameter_space
            - whiten
            - unwhiten
            - whiten_vector
            - unwhiten_vector

## Pathfinder

::: phydrax.uq.fit_pathfinder

---

::: phydrax.uq.PathfinderResult
    options:
        members:
            - sample_approximation
            - predict
            - predict_observations

## Tempered SMC

::: phydrax.uq.sample_tempered_smc

---

::: phydrax.uq.TemperedSMCResult
    options:
        members:
            - predict
            - predict_observations

## Conditional SMC and particle MCMC

`conditional_particle_filter` preserves one reference trajectory while running the
ordinary state-space transition and observation contracts. Ancestor sampling is
explicit. `particle_gibbs` alternates conditional filtering and coherent path draws.
`particle_marginal_metropolis_hastings` uses an unbiased particle likelihood estimate
inside a parameter proposal and reports every filter evaluation, acceptance decision,
and failure. Both algorithms retain the seed lineage required to replay their
auxiliary randomness.

::: phydrax.uq.conditional_particle_filter

---

::: phydrax.uq.particle_gibbs

---

::: phydrax.uq.ParticleGibbsResult

---

::: phydrax.uq.particle_marginal_metropolis_hastings

---

::: phydrax.uq.ParticleMarginalMetropolisHastingsResult

## Ensemble Kalman inversion

::: phydrax.uq.fit_eki

---

::: phydrax.uq.EnsembleKalmanResult
    options:
        members:
            - ensemble_size
            - mean
            - unconstrained_mean
            - predict
            - predict_observations

---

::: phydrax.uq.EnsembleKalmanDiagnostics

---

::: phydrax.uq.EnsembleKalmanConvergenceError

## Checkpoints and result interchange

::: phydrax.uq.export_result

---

::: phydrax.uq.read_result_archive

---

::: phydrax.uq.UQResultArchive
    options:
        members:
            - array
            - tree

---

::: phydrax.uq.to_arviz

---

::: phydrax.uq.CheckpointError

---

::: phydrax.uq.CheckpointCompatibilityError

---

::: phydrax.uq.CheckpointCorruptionError

## Model discrepancy

::: phydrax.uq.ExactGaussianProcessDiscrepancy
    options:
        members:
            - __init__
            - residual
            - factor
            - log_marginal_likelihood
            - condition

---

::: phydrax.uq.ExactGaussianProcessFactor
    options:
        members:
            - __init__
            - factor_storage_elements
            - log_probability
            - conditioner
            - condition

---

::: phydrax.uq.GaussianProcessConditioner
    options:
        members:
            - condition

---

::: phydrax.uq.GaussianProcessCondition
    options:
        members:
            - sample
            - predictive_field

---

::: phydrax.uq.MultiOutputGaussianProcessDiscrepancy
    options:
        members:
            - __init__
            - residual
            - log_marginal_likelihood
            - condition

---

::: phydrax.uq.MultiOutputGaussianProcessCondition
    options:
        members:
            - sample
            - predictive_field

---

::: phydrax.uq.SparseGaussianProcessDiscrepancy
    options:
        members:
            - __init__
            - from_evenly_spaced_subset
            - residual
            - factor
            - log_marginal_likelihood
            - condition


---

::: phydrax.uq.SparseGaussianProcessFactor
    options:
        members:
            - __init__
            - factor_storage_elements
            - log_probability
            - conditioner
            - condition


---

::: phydrax.uq.select_inducing_points

---

::: phydrax.uq.InducingPointSelection

### Identifiability gates

::: phydrax.uq.discrepancy_identifiability_report

---

::: phydrax.uq.DiscrepancyIdentifiabilityThresholds

---

::: phydrax.uq.DiscrepancyIdentifiabilityReport
    options:
        members:
            - raise_on_failure
            - as_dict

## Stochastic neural inference

## Dropout

::: phydrax.nn.Dropout
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.inference_mode

## Ensembles

::: phydrax.uq.HomogeneousFunctionEnsemble
    options:
        members:
            - __init__
            - from_factory
            - from_members
            - predict
            - predict_many
            - predict_operator

---

::: phydrax.uq.HeterogeneousFunctionEnsemble
    options:
        members:
            - __init__
            - predict
            - predict_many
            - predict_operator

---

::: phydrax.uq.fit_ensemble

---

::: phydrax.uq.EnsembleMemberDiagnostics

---

::: phydrax.uq.EnsembleFitResult

---

::: phydrax.uq.EnsembleFitError

## Randomized priors

::: phydrax.uq.FrozenModel
    options:
        members: []

---

::: phydrax.uq.RandomizedPriorModel
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.uq.randomized_prior_ensemble
