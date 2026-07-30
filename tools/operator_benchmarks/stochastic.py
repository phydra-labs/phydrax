from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax as phx


@dataclass(frozen=True)
class StochasticTransitionData:
    """Repeated transition realizations for one fixed spatial discretization."""

    axis: phx.nn.OperatorAxis
    initial_states: Array
    final_states: Array
    duration: float
    drift_matrix: Array
    noise_matrix: Array
    analytic_mean: Array | None = None
    analytic_covariance: Array | None = None

    @property
    def num_cases(self) -> int:
        return int(self.initial_states.shape[0])

    @property
    def num_realizations(self) -> int:
        return int(self.final_states.shape[1])

    @property
    def grid_size(self) -> int:
        return int(self.initial_states.shape[-1])

    def evaluation_batch(
        self, case_indices: Sequence[int] | None = None
    ) -> phx.nn.OperatorBatch:
        indices = (
            jnp.arange(self.num_cases, dtype=jnp.int32)
            if case_indices is None
            else jnp.asarray(tuple(case_indices), dtype=jnp.int32)
        )
        states = self.initial_states[indices]
        durations = jnp.full_like(states, self.duration)
        return phx.nn.OperatorBatch(
            inputs={
                "state": phx.nn.FunctionSamples(values=states, axes=(self.axis,)),
                "duration": phx.nn.FunctionSamples(values=durations, axes=(self.axis,)),
            },
            queries={"query": phx.nn.FunctionSamples(values=None, axes=(self.axis,))},
            case_axes=("case",),
        )

    def operator_dataset(
        self,
        case_indices: Sequence[int] | None = None,
    ) -> phx.nn.OperatorDataset:
        indices = (
            tuple(range(self.num_cases))
            if case_indices is None
            else tuple(int(index) for index in case_indices)
        )
        states = self.initial_states[jnp.asarray(indices, dtype=jnp.int32)]
        targets = self.final_states[jnp.asarray(indices, dtype=jnp.int32)]
        repeated_states = jnp.repeat(states, self.num_realizations, axis=0)
        flat_targets = targets.reshape((-1, self.grid_size))
        durations = jnp.full_like(repeated_states, self.duration)
        provenance = tuple(
            phx.nn.OperatorCaseProvenance(
                f"state:{case_index}:draw:{draw}",
                identities={"initial_state": f"state:{case_index}"},
            )
            for case_index in indices
            for draw in range(self.num_realizations)
        )
        return phx.nn.operator_dataset_from_arrays(
            {"state": repeated_states, "duration": durations},
            {"output": flat_targets},
            source_axes={"state": (self.axis,), "duration": (self.axis,)},
            query_axes=(self.axis,),
            provenance=provenance,
        )


def _smooth_initial_states(key: Key[Array, ""], cases: int, size: int, /) -> Array:
    x = jnp.linspace(0.0, 1.0, size, endpoint=False)
    coefficients = jr.normal(key, (cases, 5))
    basis = jnp.stack(
        (
            jnp.ones_like(x),
            jnp.sin(2.0 * jnp.pi * x),
            jnp.cos(2.0 * jnp.pi * x),
            jnp.sin(4.0 * jnp.pi * x),
            jnp.cos(4.0 * jnp.pi * x),
        )
    )
    return coefficients @ basis / jnp.sqrt(5.0)


def _linear_transition_moments(
    drift_matrix: Array,
    noise_matrix: Array,
    initial_states: Array,
    duration: float,
    /,
) -> tuple[Array, Array]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(drift_matrix)
    propagator = (eigenvectors * jnp.exp(eigenvalues * duration)) @ eigenvectors.T
    mean = initial_states @ propagator.T
    forcing = eigenvectors.T @ (noise_matrix @ noise_matrix.T) @ eigenvectors
    sums = eigenvalues[:, None] + eigenvalues[None, :]
    integral = jnp.where(
        jnp.abs(sums) > 1e-12,
        jnp.expm1(sums * duration) / sums,
        duration,
    )
    covariance = eigenvectors @ (forcing * integral) @ eigenvectors.T
    return mean, 0.5 * (covariance + covariance.T)


def stochastic_heat_transition_data(
    key: Key[Array, ""],
    /,
    *,
    grid_size: int = 16,
    num_cases: int = 8,
    num_realizations: int = 64,
    duration: float = 0.1,
    diffusivity: float = 0.02,
    noise_rank: int = 3,
    noise_scale: float = 0.35,
    dt0: float = 2e-3,
) -> StochasticTransitionData:
    """Generate semidiscrete stochastic-heat transitions through Diffrax."""
    size, cases, draws = int(grid_size), int(num_cases), int(num_realizations)
    if size < 3 or cases <= 0 or draws <= 0:
        raise ValueError(
            "grid_size >= 3 and positive case/realization counts are required."
        )
    if duration <= 0.0 or diffusivity <= 0.0 or dt0 <= 0.0:
        raise ValueError("duration, diffusivity, and dt0 must be positive.")
    initial_key, driver_key = jr.split(key)
    initial = _smooth_initial_states(initial_key, cases, size)
    axis_discretization = phx.domain.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis_discretization,))
    noise_basis = phx.solver.SpatialNoiseBasis.from_spectrum(
        spatial,
        float(noise_scale) ** 2 / float(size),
        rank=int(noise_rank),
    )
    drift = float(diffusivity) * spatial.laplacian_matrix()
    noise = noise_basis.diffusion_matrix
    save_time = jnp.asarray([duration])
    final: list[Array] = []
    for case in range(cases):
        spde = phx.solver.semidiscretize_reaction_diffusion(
            initial[case],
            spatial,
            t0=0.0,
            t1=duration,
            kappa=float(diffusivity),
            noise_basis=noise_basis,
        )
        driver = spde.wiener_driver(
            jr.fold_in(driver_key, case),
            tolerance=min(float(dt0), 1e-3),
            realization_id=f"heat-case-{case}",
        )
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=save_time,
            driver=driver,
            num_paths=draws,
            dt0=dt0,
        )
        final.append(solution.states[:, 0, :])
    mean, covariance = _linear_transition_moments(
        drift,
        noise,
        initial,
        float(duration),
    )
    axis = phx.nn.OperatorAxis(
        "x",
        axis_discretization.nodes,
        quadrature_weights=axis_discretization.quad_weights,
        periodic=True,
    )
    return StochasticTransitionData(
        axis,
        initial,
        jnp.stack(tuple(final)),
        float(duration),
        drift,
        noise,
        mean,
        covariance,
    )


def allen_cahn_transition_data(
    key: Key[Array, ""],
    /,
    *,
    grid_size: int = 8,
    num_cases: int = 12,
    num_realizations: int = 16,
    duration: float = 0.2,
    diffusivity: float = 0.01,
    noise_rank: int = 3,
    noise_scale: float = 0.7,
    dt0: float = 5e-3,
) -> StochasticTransitionData:
    """Generate fixed-grid stochastic Allen--Cahn transition realizations."""
    size, cases, draws = int(grid_size), int(num_cases), int(num_realizations)
    initial_key, driver_key = jr.split(key)
    initial = 0.35 * _smooth_initial_states(initial_key, cases, size)
    axis_discretization = phx.domain.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis_discretization,))
    noise_basis = phx.solver.SpatialNoiseBasis.from_spectrum(
        spatial,
        float(noise_scale) ** 2 / float(size),
        rank=int(noise_rank),
    )
    drift = float(diffusivity) * spatial.laplacian_matrix()
    noise = noise_basis.diffusion_matrix
    final: list[Array] = []
    for case in range(cases):
        spde = phx.solver.semidiscretize_reaction_diffusion(
            initial[case],
            spatial,
            t0=0.0,
            t1=duration,
            kappa=float(diffusivity),
            reaction=lambda t, state, args: state - state**3,
            noise_basis=noise_basis,
        )
        driver = spde.wiener_driver(
            jr.fold_in(driver_key, case),
            tolerance=min(float(dt0), 1e-3),
            realization_id=f"allen-cahn-case-{case}",
        )
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=jnp.asarray([duration]),
            driver=driver,
            num_paths=draws,
            dt0=dt0,
        )
        final.append(solution.states[:, 0, :])
    axis = phx.nn.OperatorAxis(
        "x",
        axis_discretization.nodes,
        quadrature_weights=axis_discretization.quad_weights,
        periodic=True,
    )
    return StochasticTransitionData(
        axis,
        initial,
        jnp.stack(tuple(final)),
        float(duration),
        drift,
        noise,
    )


@dataclass(frozen=True)
class StochasticHeatGaussianBenchmarkResult:
    diagonal_energy_distance: float
    low_rank_energy_distance: float
    diagonal_covariance_error: float
    low_rank_covariance_error: float
    location_rmse: float
    fine_grid_finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.low_rank_energy_distance < self.diagonal_energy_distance
            and self.low_rank_covariance_error < self.diagonal_covariance_error
            and self.fine_grid_finite
        )


def _operator_predictive(samples, batch, sample_dim):
    return phx.uq.operator_predictive_from_samples(
        samples,
        batch,
        phx.nn.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis(sample_dim, "process"),),
        field_name="output",
        query_name="query",
    )


def run_stochastic_heat_gaussian_benchmark(
    key: Key[Array, ""],
    /,
    *,
    data: StochasticTransitionData | None = None,
    evaluation_samples: int = 256,
    jitter: float = 1e-4,
) -> StochasticHeatGaussianBenchmarkResult:
    """Compare diagonal and coherent low-rank Gaussian transition baselines."""
    dataset = (
        stochastic_heat_transition_data(jr.fold_in(key, 0)) if data is None else data
    )
    if dataset.analytic_mean is None or dataset.analytic_covariance is None:
        raise ValueError("The stochastic-heat benchmark requires analytic moments.")
    batch = dataset.evaluation_batch()
    query = batch.require_single_query()
    covariance = dataset.analytic_covariance
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    keep = eigenvalues > max(float(jitter) ** 2, 1e-12)
    factors = eigenvectors[:, keep] * jnp.sqrt(jnp.maximum(eigenvalues[keep], 0.0))
    factors = jnp.broadcast_to(
        factors,
        (dataset.num_cases,) + factors.shape,
    )
    fixed_scale = jnp.full_like(dataset.analytic_mean, float(jitter))
    low_rank = phx.nn.GaussianOperatorDistribution(
        mean=dataset.analytic_mean,
        scale=fixed_scale,
        factors=factors,
        query=query,
        output_spec=phx.nn.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
        uncertainty_source="process",
    )
    diagonal_scale = jnp.broadcast_to(
        jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0) + float(jitter) ** 2),
        dataset.analytic_mean.shape,
    )
    diagonal = phx.nn.GaussianOperatorDistribution(
        mean=dataset.analytic_mean,
        scale=diagonal_scale,
        factors=None,
        query=query,
        output_spec=phx.nn.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
        uncertainty_source="process",
    )
    diagonal_key, low_rank_key = jr.split(jr.fold_in(key, 1))
    count = int(evaluation_samples)
    reference = jnp.moveaxis(dataset.final_states, 1, 0)
    if reference.shape[0] > count:
        reference = reference[:count]
    diagonal_samples = diagonal.sample(diagonal_key, (reference.shape[0],))
    low_rank_samples = low_rank.sample(low_rank_key, (reference.shape[0],))
    reference_predictive = _operator_predictive(reference, batch, "reference")
    diagonal_predictive = _operator_predictive(diagonal_samples, batch, "diagonal")
    low_rank_predictive = _operator_predictive(low_rank_samples, batch, "low_rank")
    diagonal_distance = phx.uq.operator_ensemble_energy_distance(
        diagonal_predictive,
        reference_predictive,
    )
    low_rank_distance = phx.uq.operator_ensemble_energy_distance(
        low_rank_predictive,
        reference_predictive,
    )
    true_covariance = covariance + float(jitter) ** 2 * jnp.eye(dataset.grid_size)
    diagonal_error = jnp.linalg.norm(diagonal.dense_covariance()[0] - true_covariance)
    low_rank_error = jnp.linalg.norm(low_rank.dense_covariance()[0] - true_covariance)
    empirical_mean = jnp.mean(dataset.final_states, axis=1)
    location_rmse = jnp.sqrt(jnp.mean((empirical_mean - dataset.analytic_mean) ** 2))
    fine_data = stochastic_heat_transition_data(
        jr.fold_in(key, 2),
        grid_size=2 * dataset.grid_size,
        num_cases=1,
        num_realizations=2,
        duration=dataset.duration,
        noise_rank=int(dataset.noise_matrix.shape[-1]),
        dt0=min(dataset.duration / 10.0, 5e-3),
    )
    fine_covariance = fine_data.analytic_covariance
    if fine_covariance is None:
        raise AssertionError(
            "Stochastic-heat data unexpectedly omitted analytic covariance."
        )
    fine_finite = bool(
        jnp.all(jnp.isfinite(fine_data.final_states))
        & jnp.all(jnp.isfinite(fine_covariance))
    )
    return StochasticHeatGaussianBenchmarkResult(
        float(diagonal_distance),
        float(low_rank_distance),
        float(diagonal_error),
        float(low_rank_error),
        float(location_rmse),
        fine_finite,
    )


@dataclass(frozen=True)
class AllenCahnFlowBenchmarkTrial:
    seed: int
    gaussian_initial_nll: float
    gaussian_final_nll: float
    flow_initial_nll: float
    flow_final_nll: float
    gaussian_energy_distance: float
    flow_energy_distance: float

    @property
    def finite(self) -> bool:
        values = jnp.asarray(
            [
                self.gaussian_initial_nll,
                self.gaussian_final_nll,
                self.flow_initial_nll,
                self.flow_final_nll,
                self.gaussian_energy_distance,
                self.flow_energy_distance,
            ]
        )
        return bool(jnp.all(jnp.isfinite(values)))

    @property
    def won(self) -> bool:
        return self.finite and (
            self.flow_final_nll < self.gaussian_final_nll
            or self.flow_energy_distance < self.gaussian_energy_distance
        )


@dataclass(frozen=True)
class AllenCahnFlowBenchmarkResult:
    trials: tuple[AllenCahnFlowBenchmarkTrial, ...]

    @property
    def passed(self) -> bool:
        required = ceil(2.0 * len(self.trials) / 3.0)
        return sum(trial.won for trial in self.trials) >= required


def _fit_allen_cahn_trial(
    data: StochasticTransitionData,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    evaluation_samples: int,
) -> AllenCahnFlowBenchmarkTrial:
    split = max(1, data.num_cases - max(1, data.num_cases // 4))
    train = data.operator_dataset(range(split))
    evaluation_indices = tuple(range(split, data.num_cases))
    evaluation_batch = data.evaluation_batch(evaluation_indices)
    reference = jnp.moveaxis(data.final_states[jnp.asarray(evaluation_indices)], 1, 0)
    if reference.shape[0] > evaluation_samples:
        reference = reference[:evaluation_samples]
    location = phx.nn.FNO(
        n_modes=(max(1, data.grid_size // 3),),
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(seed),
    )
    encoder = phx.nn.FixedBranchEncoder(
        phx.nn.MLP(
            in_size=data.grid_size,
            out_size=8,
            width_size=16,
            depth=1,
            key=jr.key(seed + 1),
        ),
        8,
    )
    conditioner = phx.nn.OperatorBatchConditioner({"state": encoder})
    flow = phx.nn.conditional_coupling_flow_operator(
        jr.key(seed + 2),
        location_model=location,
        conditioner=conditioner,
        reference_query=evaluation_batch.require_single_query(),
        uncertainty_source="process",
        flow_layers=3,
        nn_width=16,
        nn_depth=1,
    )
    gaussian_base = phx.nn.FNO(
        n_modes=(max(1, data.grid_size // 3),),
        in_channels="scalar",
        out_channels=4,
        width=8,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(seed + 3),
    )
    gaussian = phx.nn.GaussianFunctionOperator(
        gaussian_base,
        out_channels="scalar",
        factor_rank=2,
        min_scale=1e-3,
        uncertainty_source="process",
    )

    def heldout_nll(model):
        distribution = model.distribution(evaluation_batch)
        return float(-jnp.mean(jax.vmap(distribution.log_prob)(reference)))

    gaussian_initial_nll = heldout_nll(gaussian)
    flow_initial_nll = heldout_nll(flow)
    loss = (phx.nn.OperatorDistributionNLL(),)
    gaussian_fit = phx.nn.fit_operator(
        gaussian,
        train,
        loss_terms=loss,
        learning_rate=2e-3,
        epochs=max(1, int(steps)),
        steps=int(steps),
        batch_size=int(batch_size),
        seed=seed,
        jit=True,
    )
    flow_fit = phx.nn.fit_operator(
        flow,
        train,
        loss_terms=loss,
        learning_rate=2e-3,
        epochs=max(1, int(steps)),
        steps=int(steps),
        batch_size=int(batch_size),
        seed=seed + 10,
        jit=True,
    )
    fitted_gaussian = gaussian_fit.execution_model
    fitted_flow = flow_fit.execution_model
    if not isinstance(
        fitted_gaussian,
        phx.nn.AbstractProbabilisticOperatorModel,
    ) or not isinstance(
        fitted_flow,
        phx.nn.AbstractProbabilisticOperatorModel,
    ):
        raise TypeError(
            "Distributional training must preserve probabilistic operator models."
        )
    gaussian_final_nll = heldout_nll(fitted_gaussian)
    flow_final_nll = heldout_nll(fitted_flow)
    gaussian_samples = fitted_gaussian.sample(
        evaluation_batch,
        num_samples=int(reference.shape[0]),
        key=jr.key(seed + 20),
    )
    flow_samples = fitted_flow.sample(
        evaluation_batch,
        num_samples=int(reference.shape[0]),
        key=jr.key(seed + 21),
    )
    reference_predictive = _operator_predictive(reference, evaluation_batch, "reference")
    gaussian_predictive = _operator_predictive(
        gaussian_samples,
        evaluation_batch,
        "gaussian",
    )
    flow_predictive = _operator_predictive(flow_samples, evaluation_batch, "flow")
    return AllenCahnFlowBenchmarkTrial(
        seed,
        gaussian_initial_nll,
        gaussian_final_nll,
        flow_initial_nll,
        flow_final_nll,
        float(
            phx.uq.operator_ensemble_energy_distance(
                gaussian_predictive,
                reference_predictive,
            )
        ),
        float(
            phx.uq.operator_ensemble_energy_distance(
                flow_predictive,
                reference_predictive,
            )
        ),
    )


def run_allen_cahn_flow_benchmark(
    key: Key[Array, ""],
    /,
    *,
    data: StochasticTransitionData | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = 100,
    batch_size: int = 32,
    evaluation_samples: int = 16,
) -> AllenCahnFlowBenchmarkResult:
    """Train Gaussian and conditional-flow transitions on stochastic Allen--Cahn data."""
    dataset = allen_cahn_transition_data(key) if data is None else data
    resolved_seeds = tuple(int(seed) for seed in seeds)
    if not resolved_seeds:
        raise ValueError("At least one benchmark seed is required.")
    trials = tuple(
        _fit_allen_cahn_trial(
            dataset,
            seed,
            steps=int(steps),
            batch_size=int(batch_size),
            evaluation_samples=int(evaluation_samples),
        )
        for seed in resolved_seeds
    )
    return AllenCahnFlowBenchmarkResult(trials)


__all__ = [
    "AllenCahnFlowBenchmarkResult",
    "AllenCahnFlowBenchmarkTrial",
    "StochasticHeatGaussianBenchmarkResult",
    "StochasticTransitionData",
    "allen_cahn_transition_data",
    "run_allen_cahn_flow_benchmark",
    "run_stochastic_heat_gaussian_benchmark",
    "stochastic_heat_transition_data",
]
