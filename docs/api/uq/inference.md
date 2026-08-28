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

::: phydrax.nn.parameters.ParameterSubspace
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
finite-state likelihood explicitly. Bellman pseudo-likelihood enters through
`StateSpaceLaplaceLikelihood`; conditionally linear particle likelihood enters through
`RaoBlackwellizedFilterLikelihood`. Both are explicit approximate backends and retain
their native filter result and diagnostics. The Rao--Blackwellized backend requires a
`RaoBlackwellizedStateSpaceProblem`; the built-in exact path rejects that problem type
rather than inventing an exact likelihood.

Any Bellman, Rao--Blackwellized, particle, guided-particle, or ensemble backend must
carry a non-exact `likelihood_id`. Approximate likelihoods default to
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

::: phydrax.uq.StateSpaceLaplaceLikelihood

---

::: phydrax.uq.RaoBlackwellizedFilterLikelihood

---

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

`search_map_candidates` performs deterministic screening over an explicitly declared
finite set of unconstrained posterior positions. Candidate points must exactly match
`PosteriorProblem.initial_position` in PyTree structure, trailing leaf shapes, and leaf
dtypes. One `FiniteAxis` may hold complete correlated positions; multiple axes form a
Cartesian product of independently chosen position blocks.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx


    parameter_space = phx.uq.ParameterSpace(
        jnp.zeros((2,)),
        log_prior=lambda _: jnp.zeros(()),
    )
    problem = phx.uq.PosteriorProblem(
        parameter_space,
        lambda value: -jnp.sum((value - jnp.asarray([1.2, 2.2])) ** 2),
    )
    candidates = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(
            jnp.asarray(
                [
                    [-2.0, -2.0],
                    [1.0, 2.0],
                    [3.0, 3.0],
                ]
            )
        )
    )

    screened = phx.uq.search_map_candidates(
        problem,
        candidates,
        search=phx.optim.FiniteExhaustiveSearch(batch_size=2),
    )
    if not screened.valid:
        raise RuntimeError(screened.termination_reason)

    # Optional continuous local refinement; a separate numerical claim.
    local = phx.uq.find_map(problem, screened.position)
    ```

Each candidate is evaluated once with the complete
`PosteriorProblem.negative_log_density`, including likelihood, prior, bijector
Jacobian, and any parameter-space transformation. The selected constrained
`parameters` are reconstructed only after the finite reduction. A valid result reports
the exact finite minimum, flat and per-axis indices, deterministic axis paths,
candidate signature, and exact attempted/valid/invalid counts. If every candidate is
invalid or nonfinite, `valid=False`, `position=None`, `parameters=None`, indices are
`-1`, and objective and log density are `NaN`; no arbitrary candidate is returned.

Selection is nondifferentiable and the winning position is detached. Use the finite
result as a deterministic coarse initializer only when a subsequent call to `find_map`
is required. The finite guarantee does not extend to positions outside the catalog.
`export_result` stores both valid and all-invalid candidate-search evidence as
`map_candidate_search`, excluding the live `PosteriorProblem` and search object while
retaining method, layout, signature, and count provenance.

`search_map` provides two bounded initializers in unconstrained posterior-position
coordinates. `DifferentialEvolutionSearch` preserves final-population and generation
evidence. `GaussianProcessMAPSearch` performs sequential expected improvement and
preserves every evaluated position, raw negative log density, validity flag, proposal
kind, and running best value. Neither archive is a posterior sample, convergence
certificate, or stationarity claim. Local stationarity remains a separate `find_map`
phase.

The GP surrogate is fitted after standardizing observed negative log densities.
`GaussianProcessLikelihoodState.noise_scale` remains in raw negative-log-density
units and is divided by the active objective scale before covariance construction.
`jitter` acts directly in standardized covariance units. Search positions are modeled
in an affine unit box; physical parameter reconstruction still follows the
`ParameterSpace` bijectors.

::: phydrax.uq.search_map_candidates

---

::: phydrax.uq.MAPCandidateSearchResult

---


::: phydrax.uq.search_map

---

::: phydrax.uq.MAPSearchResult

::: phydrax.uq.GaussianProcessMAPSearch

---

::: phydrax.uq.GaussianProcessMAPSearchResult

---

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

## Gaussian-process discrepancy

Scalar GP models separate observations from the covariance/noise state. A
`GaussianProcessLikelihoodState` contains one shared `phydrax.kernels` expression,
observation-noise scale, and numerical factorization jitter. Pass a state explicitly
to likelihood, factorization, and conditioning calls. Fixed states can be factored
once; a state callback on `GaussianProcessMarginalLikelihood` keeps kernel and noise
parameters differentiable posterior leaves.

If the kernel exposes the `AbstractFiniteFeatureKernel` capability and its rank is
strictly smaller than the observation count, `ExactGaussianProcessDiscrepancy.factor`
selects `FiniteFeatureGaussianProcessFactor` automatically. Supported amplitudes,
finite sums, and input transforms preserve that capability. Products and general
scales remain on the conservative dense path unless they declare an exact feature
factorization. Equal or larger feature rank also uses the dense path. Both paths
implement the same declared covariance and expose comparable log-probability,
conditioning, and storage diagnostics.

Scalar value-observation GPs accept any design rank declared by
`state.kernel.input_ndim`. A signature-kernel observation design is
`(observation, knot, channel)`; inducing and query paths may use different knot
counts when the kernel supports rectangular cross-evaluation. Exact and FITC
factorizations preserve these axes and return one scalar latent value per
leading design row.

```python
import jax.numpy as jnp
import phydrax as phx

time = jnp.linspace(0.0, 1.0, 6)
path = jnp.stack((time, time**2), axis=-1)
paths = jnp.stack((path, -path, 0.5 * path, 1.5 * path))
observations = paths[:, -1, 0]
physical_mean = jnp.zeros_like(observations)
query_time = jnp.linspace(0.0, 1.0, 4)
query_path = jnp.stack((query_time, query_time**2), axis=-1)
query_paths = jnp.stack((query_path, -query_path))
path_kernel = phx.kernels.SignaturePDEKernel(
    phx.kernels.LinearKernel(),
    polynomial_order=5,
)
model = phx.uq.ExactGaussianProcessDiscrepancy(paths, observations)
state = phx.uq.GaussianProcessLikelihoodState(
    kernel=path_kernel,
    noise_scale=0.05,
)
conditioned = model.condition(
    physical_mean,
    query_paths,
    state=state,
    output_dim="trajectory",
)
```

Path kernels are not coordinate-functional kernels. `FunctionalDesign`,
coordinate partial derivatives, and differential-functional GP states require
`input_ndim == 1` and reject path kernels explicitly.

::: phydrax.uq.GaussianProcessLikelihoodState

---

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

::: phydrax.uq.FiniteFeatureGaussianProcessFactor
    options:
        members:
            - factor_storage_elements
            - log_probability
            - conditioner
            - condition

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

### Computation-aware scalar GPs

Computation-aware factors condition on a native linear action operator and retain
unobserved prior directions in the latent covariance. The statistical
`GaussianProcessLikelihoodState` remains independent of action selection and
execution resources. Use `latent_moments` for a mean and diagonal variance; full
`condition` calls obey the query-covariance byte limit in
`GaussianProcessComputationPolicy`.

`ComputationAwareGaussianProcessELBO` is a full-data variational lower bound.
It equals exact marginal likelihood only for a complete independent action basis.

::: phydrax.uq.AbstractGaussianProcessActionPolicy

---

::: phydrax.uq.FixedGaussianProcessActionPolicy
    options:
        members:
            - __init__

---

::: phydrax.uq.BlockSparseGaussianProcessActionPolicy
    options:
        members:
            - __init__
            - from_random

---

::: phydrax.uq.PseudoInputGaussianProcessActionPolicy
    options:
        members:
            - __init__

---

::: phydrax.uq.GaussianProcessComputationPolicy
    options:
        members:
            - __init__

---

::: phydrax.uq.ComputationAwareGaussianProcessDiscrepancy
    options:
        members:
            - __init__
            - residual
            - factor
            - elbo
            - condition

---

::: phydrax.uq.ComputationAwareGaussianProcessFactor
    options:
        members:
            - __init__
            - factor_storage_elements
            - elbo
            - latent_moments
            - conditioner
            - condition

---

::: phydrax.uq.ComputationAwareGaussianProcessConditioner
    options:
        members:
            - storage_elements
            - condition

---

::: phydrax.uq.ComputationAwareGaussianProcessDiagnostics

---

::: phydrax.uq.ComputationAwareGaussianProcessELBO
    options:
        members:
            - __init__
            - per_case_log_prob
            - log_prob

### Correlated outputs

`MultiOutputDesign` stores one flat row per observed point/channel pair. Construct it
from dense values with an explicit mask for heterotopic data. Output covariance is
represented only through PSD-preserving `Coregionalization` factors. ICM uses one
spatial kernel; LMC combines multiple spatial kernels and coregionalizations.

::: phydrax.uq.MultiOutputDesign
    options:
        members:
            - from_dense
            - flatten
            - dense

---

::: phydrax.uq.Coregionalization

---

::: phydrax.uq.AbstractMultiOutputKernel

---

::: phydrax.uq.IntrinsicCoregionalizationKernel

---

::: phydrax.uq.LinearModelCoregionalizationKernel

---

::: phydrax.uq.MultiOutputGaussianProcessLikelihoodState

---

::: phydrax.uq.MultiOutputGaussianProcessDiscrepancy
    options:
        members:
            - __init__
            - from_dense
            - residual
            - log_marginal_likelihood
            - condition

---

::: phydrax.uq.MultiOutputGaussianProcessCondition
    options:
        members:
            - dense_mean
            - dense_variance
            - sample
            - predictive_field

### Linear-functional observations

Functional GP designs mix value, partial-derivative, directional-derivative, and
Laplacian blocks in one covariance system. Operator coefficients may be dynamic JAX
values. The shared spatial kernel's derivative certificate is checked against every
block. Supplying `inducing_design` on the likelihood state selects interdomain FITC;
omitting it selects exact inference.

::: phydrax.uq.LinearDifferentialFunctional

---

::: phydrax.uq.FunctionalObservationBlock

---

::: phydrax.uq.FunctionalDesign
    options:
        members:
            - from_points
            - flatten
            - split

---

::: phydrax.uq.value_functional

---

::: phydrax.uq.partial_derivative_functional

---

::: phydrax.uq.directional_derivative_functional

---

::: phydrax.uq.laplacian_functional

---

::: phydrax.uq.functional_kernel_matrix

---

::: phydrax.uq.functional_kernel_diagonal

---

::: phydrax.uq.FunctionalGaussianProcessLikelihoodState

---

::: phydrax.uq.FunctionalGaussianProcessDiscrepancy
    options:
        members:
            - residual
            - log_marginal_likelihood
            - condition

---

::: phydrax.uq.FunctionalGaussianProcessCondition
    options:
        members:
            - split_mean
            - split_variance
            - sample

### Inducing-point selection

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

::: phydrax.nn.layers.Dropout
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.inference_mode

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

## Variational posterior inference

`fit_variational` minimizes reverse KL in the unconstrained coordinates of a
`PosteriorProblem`. The target therefore includes the physical prior and each
bijector Jacobian exactly once. `MeanFieldGaussianFamily` is the default normalized
family. It is scalable but cannot represent posterior correlation or disconnected
modes and commonly underestimates uncertainty.

`fit_flow_variational` first obtains a mean-field approximation, initializes a
FlowJAX spline distribution from its draws, and then optimizes the flow against the
target density. This reverse-KL objective is distinct from the sample maximum-
likelihood objective used by flow-assisted NUTS.
`ContinuousFlowLaw` is a different, solver-backed probability-law adapter. It is
restricted to small full-dimensional real Euclidean events and computes exact
state-Jacobian divergence. `estimate_continuous_flow_log_prob` uses keyed
Hutchinson probes and reports their Monte Carlo error, but that estimate is not an
exact normalized density and must not be used as an ordinary Metropolis acceptance
target. FlowJAX remains the default finite-dimensional variational and proposal
family.


Both results retain unconstrained and physical draws, target and family log
densities, ELBO and gradient histories, deterministic root-key lineage, portable
checkpoint state, prediction methods, memory, duration, and approximation identity.

::: phydrax.uq.MeanFieldGaussianFamily

---

::: phydrax.uq.VariationalConfig

---

::: phydrax.uq.VariationalResult

---

::: phydrax.uq.fit_variational

---

::: phydrax.uq.FlowVariationalConfig

---

::: phydrax.uq.FlowVariationalFamily

---

::: phydrax.uq.fit_flow_variational

## Full-path and amortized state-space inference

`state_space_path_log_density` evaluates a normalized latent path containing the
initial state followed by one state per observation step. It returns initial,
transition, and observation terms separately. Inactive padded steps contribute zero
and must preserve their predecessor exactly.

`GaussianMarkovVariationalFamily` is a normalized directed Gaussian Markov path
with dense affine transitions and diagonal innovations.
`fit_state_space_variational` is the fixed-model full-path reference.
`AmortizedGaussianMarkovFamily` replaces case-specific free parameters with one
shared bidirectional context encoder and can be rebound to a compatible observation
sequence without retraining.

`fit_buffered_state_space_variational` samples fixed-length target intervals.
`StateSpaceWindowPlan` records exact edge inclusion probabilities; target terms use
inverse-inclusion weights. Left and right buffers control only encoder context.
This remains an explicitly identified approximation because its boundary states are
provided by the amortized family rather than an exact full-data smoother.

::: phydrax.uq.StateSpacePathLogDensity

---

::: phydrax.uq.GaussianMarkovVariationalFamily

---

::: phydrax.uq.fit_state_space_variational

---

::: phydrax.uq.AmortizedGaussianMarkovFamily

---

::: phydrax.uq.fit_amortized_state_space_variational

---

::: phydrax.uq.StateSpaceWindowPlan

---

::: phydrax.uq.BufferedStateSpaceVariationalResult

---

::: phydrax.uq.fit_buffered_state_space_variational

## Causal fixed-trajectory HMC

`sample_hmc(..., trajectory_method="causal")` keeps BlackJAX warmup, momentum
generation, energy evaluation, momentum flip, and Metropolis decision unchanged.
Only the fixed-length velocity-Verlet trajectory is solved through the causal
nonlinear recurrence. Production supports diagonal adapted inverse mass matrices,
static leapfrog counts, dense-exact or position--momentum pair Hutchinson
linearization, and explicit trajectory blocking.

Every block must converge to the sequential trajectory before the proposal is
treated as ordinary HMC. `failure_policy="raise"` is the default; an explicit
sequential fallback is recorded in `CausalHMCDiagnostics`. NUTS is not supported
because dynamic tree construction does not satisfy the fixed causal layout.

::: phydrax.uq.CausalHMCConfig

---

::: phydrax.uq.CausalHMCDiagnostics

## Particle genealogical scores and stochastic gradients

`particle_genealogical_score` propagates normalized prior, transition, and
observation score increments through realized stopped-gradient ancestry. Its stored
state scales as `O(TN)`, unlike the existing density-based pair smoother's
`O(TN²)` score. The lower cost trades for greater resampling and genealogy variance;
the existing Fisher score remains available.

`ParameterizedStateSpaceProblem` binds unconstrained global coordinates into the
existing `StateSpaceStepContext.args` contract without defining a second model
hierarchy. `ParticleGenealogicalGradientEstimator` uses its complete-sequence
particle score plus the exact parameter prior and can drive `sample_sgld` or
`sample_sgnht` through the common `AbstractStochasticGradientEstimator` interface.
The existing autodiff minibatch estimator remains the default and replay-compatible.

Buffered particle SG-MCMC is deliberately not exposed: the current buffered
variational boundary law has not established a sufficiently accurate particle-score
boundary correction. Complete-sequence particle gradients are supported.

::: phydrax.uq.ParticleGenealogicalScoreResult

---

::: phydrax.uq.particle_genealogical_score

---

::: phydrax.uq.ParameterizedStateSpaceProblem

---

::: phydrax.uq.AbstractStochasticGradientEstimator

---

::: phydrax.uq.ParticleGenealogicalGradientEstimator
