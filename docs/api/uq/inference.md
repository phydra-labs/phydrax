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

## MAP estimation

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
