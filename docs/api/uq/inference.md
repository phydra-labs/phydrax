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

::: phydrax.uq.FixedConstraintLikelihood
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
