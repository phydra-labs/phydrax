import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _linear_problem(
    *,
    problem_id,
    sequence_id,
    case_axes,
    case_shape,
    case_ids,
    values,
    times,
    step_valid=None,
    observation_mask=None,
):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray(times),
        jnp.asarray(values),
        case_axes=case_axes,
        case_shape=case_shape,
        case_ids=case_ids,
        step_valid=step_valid,
        observation_mask=observation_mask,
        sequence_id=sequence_id,
    )
    mean = jnp.zeros(case_shape + (1,))
    covariance = jnp.broadcast_to(jnp.asarray(((1.0,),)), case_shape + (1, 1))
    prior = phx.stochastic.GaussianStatePrior(
        mean,
        covariance,
        state_shape=(1,),
        prior_id=f"{problem_id}:prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.15,),)),
        state_shape=(1,),
        process_id=f"{problem_id}:process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.25,),)),
        state_shape=(1,),
        observation_shape=(1,),
        offset=jnp.asarray(0.0),
        observation_id=f"{problem_id}:observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id=f"{problem_id}:model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=jnp.zeros(case_shape),
        problem_id=problem_id,
    )


def _templates():
    replicated = _linear_problem(
        problem_id="replicated-problem",
        sequence_id="replicated-sequence",
        case_axes=("replicate",),
        case_shape=(2,),
        case_ids=("replicate-a", "replicate-b"),
        times=jnp.asarray((1.0, 2.0, 3.0)),
        values=jnp.asarray(
            (
                ((0.8,), (0.0,), (1.1,)),
                ((1.4,), (1.2,), (0.0,)),
            )
        ),
        step_valid=jnp.asarray(((True, True, True), (True, True, False))),
        observation_mask=jnp.asarray(
            (
                ((True,), (False,), (True,)),
                ((True,), (True,), (False,)),
            )
        ),
    )
    independent = _linear_problem(
        problem_id="independent-problem",
        sequence_id="independent-sequence",
        case_axes=(),
        case_shape=(),
        case_ids=("single",),
        times=jnp.asarray((0.5, 1.5)),
        values=jnp.asarray(((0.6,), (1.0,))),
    )
    return replicated, independent


def _experiments():
    replicated, independent = _templates()

    def replicated_problem(parameters):
        return eqx.tree_at(
            lambda problem: problem.model.observation.offset,
            replicated,
            parameters["offset"],
        )

    def independent_problem(parameters):
        return eqx.tree_at(
            lambda problem: problem.model.observation.offset,
            independent,
            parameters["offset"],
        )

    return (
        phx.uq.StateSpaceExperiment(
            replicated_problem,
            experiment_id="replicated",
            case_axes=("replicate",),
            case_shape=(2,),
            case_ids=("replicate-a", "replicate-b"),
        ),
        phx.uq.StateSpaceExperiment(
            independent_problem,
            experiment_id="independent",
            case_axes=(),
            case_shape=(),
            case_ids=("single",),
        ),
    )


def _estimation(initial=-0.5):
    parameter_space = phx.uq.ParameterSpace(
        {"offset": jnp.asarray(initial)},
        log_prior=lambda parameters: -0.5 * (parameters["offset"] / 3.0) ** 2,
    )
    return phx.uq.StateSpaceEstimation(parameter_space, _experiments())


def _particle_experiment():
    template, _ = _templates()

    def problem(parameters):
        return eqx.tree_at(
            lambda value: value.model.observation.offset,
            template,
            parameters["offset"],
        )

    def particle_likelihood(state_space_problem):
        return phx.uq.bootstrap_particle_filter(
            jr.key(17),
            state_space_problem,
            num_particles=32,
            resampling_policy="never",
        )

    return phx.uq.StateSpaceExperiment(
        problem,
        experiment_id="particle-replicated",
        case_axes=("replicate",),
        case_shape=(2,),
        case_ids=("replicate-a", "replicate-b"),
        likelihood=particle_likelihood,
        likelihood_id="bootstrap-32",
    )


def _reference_sampler(problem, *, position):
    return problem.log_density(position)


def test_multi_experiment_likelihood_and_gradient_equal_separate_exact_terms():
    experiments = _experiments()
    likelihood = phx.uq.MultiExperimentStateSpaceLikelihood(experiments)
    parameters = {"offset": jnp.asarray(0.2)}
    result = likelihood.evaluate(parameters)
    separate = tuple(experiment.evaluate(parameters) for experiment in experiments)
    gradient = jax.grad(likelihood.log_prob)(parameters)
    separate_gradient = sum(
        (
            jax.grad(
                lambda value, item=experiment: item.evaluate(value).total_log_likelihood
            )(parameters)["offset"]
            for experiment in experiments
        ),
        jnp.asarray(0.0),
    )

    assert result.experiment_ids == ("replicated", "independent")
    assert result.per_experiment_log_likelihood.shape == (2,)
    assert result.per_case_log_likelihood.shape == (3,)
    assert result.total_log_likelihood == pytest.approx(
        sum(float(item.total_log_likelihood) for item in separate)
    )
    assert gradient["offset"] == pytest.approx(separate_gradient)
    assert jnp.isfinite(gradient["offset"])


def test_experiment_diagnostics_preserve_cases_masks_status_and_backend():
    result = _estimation().evaluate_likelihood({"offset": jnp.asarray(0.0)})
    replicated = result.experiment("replicated")
    independent = result.experiment("independent")

    assert replicated.case_axes == ("replicate",)
    assert replicated.case_shape == (2,)
    assert replicated.case_ids == ("replicate-a", "replicate-b")
    assert replicated.per_case_log_likelihood.shape == (2,)
    assert replicated.incremental_log_likelihood.shape == (2, 3)
    assert jnp.array_equal(
        replicated.step_valid,
        jnp.asarray(((True, True, True), (True, True, False))),
    )
    assert jnp.array_equal(
        replicated.problem.observations.observation_mask,
        jnp.asarray(
            (
                ((True,), (False,), (True,)),
                ((True,), (True,), (False,)),
            )
        ),
    )
    assert replicated.backend.problem is replicated.problem
    assert replicated.method == "kalman"
    assert replicated.approximation_id == "exact-linear-gaussian"
    assert replicated.model_id == "replicated-problem:model"
    assert replicated.problem_id == "replicated-problem"
    assert replicated.sequence_id == "replicated-sequence"
    assert replicated.input_id is None
    assert replicated.covariance_regularization == 0.0
    assert independent.case_axes == ()
    assert independent.case_ids == ("single",)
    assert independent.per_case_log_likelihood.shape == ()
    assert jnp.all(result.successful)
    with pytest.raises(KeyError, match="missing"):
        result.experiment("missing")


def test_approximate_experiment_composes_without_discarding_particle_diagnostics():
    experiment = _particle_experiment()
    result = phx.uq.MultiExperimentStateSpaceLikelihood((experiment,)).evaluate(
        {"offset": jnp.asarray(0.0)}
    )
    diagnostic = result.experiment("particle-replicated")

    assert isinstance(diagnostic.backend, phx.uq.ParticleFilterResult)
    assert diagnostic.method == "bootstrap-particle"
    assert diagnostic.approximation_id == "particle:32"
    assert diagnostic.backend.num_particles == 32
    assert diagnostic.backend.case_ids == diagnostic.case_ids
    assert jnp.allclose(
        diagnostic.per_case_log_likelihood,
        diagnostic.backend.cumulative_log_likelihood[..., -1],
    )
    assert result.total_log_likelihood == pytest.approx(
        jnp.sum(diagnostic.per_case_log_likelihood)
    )


def test_bellman_and_rao_blackwellized_likelihood_backends_retain_diagnostics():
    _, template = _templates()
    bellman_experiment = phx.uq.StateSpaceExperiment(
        lambda parameters: eqx.tree_at(
            lambda problem: problem.model.observation.offset,
            template,
            parameters["offset"],
        ),
        experiment_id="bellman",
        case_axes=(),
        case_shape=(),
        case_ids=("single",),
        likelihood=phx.uq.StateSpaceLaplaceLikelihood(),
        likelihood_id="bellman-pseudo",
    )

    nonlinear_prior = phx.stochastic.CategoricalStatePrior(
        jnp.asarray([[0]]), jnp.asarray([1.0]), prior_id="estimation-mode-prior"
    )
    nonlinear_transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state,
        state_shape=(1,),
        process_id="estimation-mode",
        approximation_id="constant-mode",
    )
    rb_model = phx.uq.RaoBlackwellizedStateSpaceModel(
        nonlinear_prior,
        nonlinear_transition,
        lambda mode, args: (jnp.asarray([args["offset"]]), jnp.eye(1)),
        lambda previous_mode, mode, t0, t1, context: (
            jnp.eye(1),
            jnp.zeros(1),
            jnp.asarray([[0.15]]),
        ),
        lambda mode, time, context: (
            jnp.eye(1),
            jnp.zeros(1),
            jnp.asarray([[0.25]]),
        ),
        linear_state_shape=(1,),
        observation_shape=(1,),
        model_id="estimation-rb-model",
    )

    def rb_problem(parameters):
        return phx.uq.RaoBlackwellizedStateSpaceProblem(
            rb_model,
            template.observations,
            initial_time=0.0,
            problem_id="estimation-rb-problem",
            args={"offset": parameters["offset"]},
        )

    rb_experiment = phx.uq.StateSpaceExperiment(
        rb_problem,
        experiment_id="rao-blackwellized",
        case_axes=(),
        case_shape=(),
        case_ids=("single",),
        likelihood=phx.uq.RaoBlackwellizedFilterLikelihood(
            jr.key(18),
            num_particles=4,
            resampling_policy="never",
        ),
        likelihood_id="rao-blackwellized-4",
    )
    parameters = {"offset": jnp.asarray(0.1)}
    bellman = bellman_experiment.evaluate(parameters)
    rao = rb_experiment.evaluate(parameters)

    assert isinstance(bellman.backend, phx.uq.BellmanFilterResult)
    assert bellman.method == "bellman-pseudo"
    assert bellman.curvature_damping == 0.0
    assert jnp.allclose(
        bellman.per_case_log_likelihood,
        bellman.backend.cumulative_pseudo_log_likelihood[..., -1],
    )
    assert isinstance(rao.backend, phx.uq.RaoBlackwellizedFilterResult)
    assert rao.method == "rao-blackwellized-particle"
    assert rao.problem is rao.backend.problem
    assert jnp.allclose(
        rao.per_case_log_likelihood,
        rao.backend.cumulative_log_likelihood[..., -1],
    )

    exact_only = phx.uq.StateSpaceExperiment(
        rb_problem,
        experiment_id="rao-without-backend",
        case_axes=(),
        case_shape=(),
        case_ids=("single",),
    )
    with pytest.raises(TypeError, match="requires a custom likelihood"):
        exact_only.evaluate(parameters)


@pytest.mark.parametrize(
    "workflow",
    ("local_map", "global_then_local_map", "laplace"),
)
def test_gradient_workflows_reject_custom_approximate_backend_before_tracing(workflow):
    parameter_space = phx.uq.ParameterSpace(
        {"offset": jnp.asarray(0.0)},
        log_prior=lambda parameters: -0.5 * parameters["offset"] ** 2,
    )
    estimation = phx.uq.StateSpaceEstimation(
        parameter_space,
        (_particle_experiment(),),
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"{workflow} requires explicitly transform-safe likelihood backends.*"
            "particle-replicated.*bootstrap-32"
        ),
    ):
        if workflow == "local_map":
            estimation.local_map(max_steps=1)
        elif workflow == "global_then_local_map":
            estimation.global_then_local_map(
                phx.optim.DifferentialEvolutionSearch(4, 1),
                key=jr.key(21),
                position_bounds=(
                    {"offset": jnp.asarray(-1.0)},
                    {"offset": jnp.asarray(1.0)},
                ),
                max_steps=1,
            )
        else:
            estimation.laplace(stationarity_tolerance=None)


def test_declared_transform_safe_custom_likelihood_supports_local_map():
    template, _ = _templates()

    def problem(parameters):
        return eqx.tree_at(
            lambda value: value.model.observation.offset,
            template,
            parameters["offset"],
        )

    experiment = phx.uq.StateSpaceExperiment(
        problem,
        experiment_id="custom-exact",
        case_axes=("replicate",),
        case_shape=(2,),
        case_ids=("replicate-a", "replicate-b"),
        likelihood=phx.uq.exact_state_space_log_likelihood,
        likelihood_id="custom-exact",
        transform_safe=True,
    )
    estimation = phx.uq.StateSpaceEstimation(
        phx.uq.ParameterSpace(
            {"offset": jnp.asarray(0.0)},
            log_prior=lambda parameters: -0.5 * parameters["offset"] ** 2,
        ),
        (experiment,),
    )

    result = estimation.local_map(max_steps=1, raise_on_failure=False)

    assert result.workflow == "local"
    assert result.likelihood.experiment_ids == ("custom-exact",)


def test_custom_likelihood_rejects_cached_backend_with_matching_user_ids():
    template, _ = _templates()

    def problem(parameters):
        return eqx.tree_at(
            lambda value: value.model.observation.offset,
            template,
            parameters["offset"],
        )

    cached_problem = problem({"offset": jnp.asarray(-0.5)})
    cached_backend = phx.uq.exact_state_space_log_likelihood(cached_problem)
    experiment = phx.uq.StateSpaceExperiment(
        problem,
        experiment_id="cached-exact",
        case_axes=("replicate",),
        case_shape=(2,),
        case_ids=("replicate-a", "replicate-b"),
        likelihood=lambda _: cached_backend,
        likelihood_id="cached-exact",
    )

    with pytest.raises(
        ValueError,
        match="exact evaluated StateSpaceProblem.*cached or relabelled",
    ):
        experiment.evaluate({"offset": jnp.asarray(0.5)})


def test_local_global_map_laplace_and_sampler_composition_preserve_diagnostics():
    estimation = _estimation(initial=-0.75)
    local = estimation.local_map(max_steps=50, raise_on_failure=False)
    search = phx.optim.DifferentialEvolutionSearch(
        8,
        4,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    combined = estimation.global_then_local_map(
        search,
        key=jr.key(8),
        position_bounds=(
            {"offset": jnp.asarray(-2.0)},
            {"offset": jnp.asarray(2.0)},
        ),
        max_steps=50,
        raise_on_failure=False,
    )
    laplace = estimation.laplace(
        combined,
        stationarity_tolerance=None,
        damping=0.0,
    )
    samples = laplace.approximation.sample(jr.key(9), num_samples=4)
    sampled = estimation.sample(
        _reference_sampler,
        sampler_id="reference-log-density",
        reference_position=combined.position,
        position=combined.position,
    )

    assert local.workflow == "local"
    assert local.local_map is not None
    assert local.global_search is None
    assert local.likelihood.experiment_ids == ("replicated", "independent")
    assert combined.workflow == "global-local"
    assert combined.global_search is not None
    assert combined.local_map is not None
    assert combined.log_density >= combined.global_search.log_density - 1e-5
    assert combined.likelihood.experiment_ids == ("replicated", "independent")
    assert laplace.source_map is combined
    assert laplace.likelihood.experiment_ids == combined.likelihood.experiment_ids
    assert samples["offset"].shape == (4,)
    assert sampled.sampler_id == "reference-log-density"
    assert sampled.result == pytest.approx(combined.log_density)
    assert sampled.reference_likelihood.experiment_ids == (
        "replicated",
        "independent",
    )


def test_state_space_global_then_local_map_accepts_gp_initializer():
    estimation = _estimation(initial=-0.75)
    search = phx.uq.GaussianProcessBayesianOptimization(
        8,
        objective_surrogate=phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.Matern52Kernel(length_scale=0.25),
            noise_scale=0.0,
        ),
        initial_evaluations=4,
        candidate_tuple_count=32,
        fantasy_count=8,
    )

    result = estimation.global_then_local_map(
        search,
        key=jr.key(25),
        position_bounds=(
            {"offset": jnp.asarray(-2.0)},
            {"offset": jnp.asarray(2.0)},
        ),
        max_steps=20,
        raise_on_failure=False,
    )

    assert isinstance(result.global_search, phx.uq.BayesianOptimizationMAPResult)
    assert result.global_search.valid
    assert result.local_map is not None
    assert result.local_map.objective <= result.global_search.objective + 1e-8


def test_experiment_rejects_changed_case_semantics():
    template, _ = _templates()
    experiment = phx.uq.StateSpaceExperiment(
        lambda parameters: template,
        experiment_id="wrong-contract",
        case_axes=("replicate",),
        case_shape=(2,),
        case_ids=("other-a", "other-b"),
    )

    with pytest.raises(ValueError, match="different case IDs"):
        experiment.evaluate({"offset": jnp.asarray(0.0)})
