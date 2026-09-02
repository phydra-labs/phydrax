from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

import phydrax as phx


def _metadata(
    data: phx.stochastic.StochasticTransitionView,
    name: str,
):
    if name not in data.metadata:
        raise ValueError(f"Stochastic benchmark metadata is missing {name!r}.")
    return data.metadata[name]


def _axis(
    data: phx.stochastic.StochasticTransitionView,
) -> phx.nn.operator.OperatorAxis:
    value = _metadata(data, "operator_axis")
    if not isinstance(value, phx.nn.operator.OperatorAxis):
        raise TypeError("Stochastic benchmark operator_axis metadata is invalid.")
    return value


def _physical_initial_states(
    data: phx.stochastic.StochasticTransitionView,
) -> Array:
    values = data.source_states.reshape(
        (
            data.num_cases,
            data.num_realizations,
            data.num_pairs,
        )
        + data.trajectory.state_shape
    )
    reference = values[:, 0, 0]
    if not bool(jnp.allclose(values, reference[:, None, None])):
        raise ValueError(
            "Benchmark evaluation requires one shared source state per physical case."
        )
    return reference


def _final_states(
    data: phx.stochastic.StochasticTransitionView,
) -> Array:
    if data.num_pairs != 1:
        raise ValueError("Stochastic benchmarks require exactly one transition per path.")
    return data.target_states.reshape(
        (data.num_cases, data.num_realizations) + data.trajectory.state_shape
    )


def _evaluation_batch(
    data: phx.stochastic.StochasticTransitionView,
    case_indices: Sequence[int] | None = None,
) -> phx.nn.operator.OperatorBatch:
    indices = (
        jnp.arange(data.num_cases, dtype=jnp.int32)
        if case_indices is None
        else jnp.asarray(tuple(case_indices), dtype=jnp.int32)
    )
    states = _physical_initial_states(data)[indices]
    durations = jnp.full_like(states, data.duration)
    axis = _axis(data)
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=states, axes=(axis,)),
            "duration": phx.nn.operator.FunctionSamples(values=durations, axes=(axis,)),
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
    )


def _operator_dataset(
    data: phx.stochastic.StochasticTransitionView,
    case_indices: Sequence[int] | None = None,
) -> phx.nn.operator.training.OperatorDataset:
    dataset = data.operator_dataset(source_axes=(_axis(data),))
    if case_indices is None:
        return dataset
    block = data.num_realizations * data.num_pairs
    selected = jnp.asarray(
        tuple(case * block + offset for case in case_indices for offset in range(block)),
        dtype=jnp.int32,
    )
    return dataset.take(selected)


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


class LinearGaussianReferenceOperator(phx.nn.operator.AbstractProbabilisticOperatorModel):
    """Exact transition law for a finite-dimensional self-adjoint linear SDE."""

    drift_eigenvalues: Array
    drift_eigenvectors: Array
    modal_noise_covariance: Array
    diagonal_jitter: float
    in_size: str
    out_size: str

    def __init__(
        self,
        drift_matrix: Array,
        noise_matrix: Array,
        /,
        *,
        diagonal_jitter: float = 1e-6,
    ):
        drift = jnp.asarray(drift_matrix)
        noise = jnp.asarray(noise_matrix)
        if drift.ndim != 2 or drift.shape[0] != drift.shape[1]:
            raise ValueError("drift_matrix must be square.")
        if noise.ndim != 2 or noise.shape[0] != drift.shape[0]:
            raise ValueError(
                "noise_matrix must have one row per drift_matrix state dimension."
            )
        if not bool(jnp.allclose(drift, drift.T, rtol=1e-10, atol=1e-12)):
            raise ValueError(
                "LinearGaussianReferenceOperator requires self-adjoint drift."
            )
        if float(diagonal_jitter) <= 0.0:
            raise ValueError("diagonal_jitter must be positive.")
        eigenvalues, eigenvectors = jnp.linalg.eigh(drift)
        transformed_noise = eigenvectors.T @ noise
        self.drift_eigenvalues = eigenvalues
        self.drift_eigenvectors = eigenvectors
        self.modal_noise_covariance = transformed_noise @ transformed_noise.T
        self.diagonal_jitter = float(diagonal_jitter)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_output_specs(self):
        return {"output": phx.nn.operator.OperatorOutputSpec("scalar")}

    def distribution(self, batch, /, *, key=None):
        del key
        states = batch.input("state").values
        durations = batch.input("duration").values
        if states is None or durations is None:
            raise ValueError("Reference transitions require state and duration values.")
        size = int(self.drift_eigenvalues.shape[0])
        if states.shape != batch.case_shape + (size,):
            raise ValueError(
                "Reference transition state shape must be "
                f"{batch.case_shape + (size,)}; got {states.shape}."
            )
        flat_states = states.reshape((-1, size))
        flat_durations = durations[..., 0].reshape((-1,))

        def transition_moments(state, duration):
            decay = jnp.exp(self.drift_eigenvalues * duration)
            mean = self.drift_eigenvectors @ (decay * (self.drift_eigenvectors.T @ state))
            sums = self.drift_eigenvalues[:, None] + self.drift_eigenvalues[None, :]
            integral = jnp.where(
                jnp.abs(sums) > 1e-12,
                jnp.expm1(sums * duration) / sums,
                duration,
            )
            covariance = (
                self.drift_eigenvectors
                @ (self.modal_noise_covariance * integral)
                @ self.drift_eigenvectors.T
            )
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
            factors = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
            return mean, factors

        means, factors = jax.vmap(transition_moments)(
            flat_states,
            flat_durations,
        )
        mean = means.reshape(batch.case_shape + (size,))
        factor = factors.reshape(batch.case_shape + (size, size))
        return phx.nn.operator.GaussianOperatorDistribution(
            mean=mean,
            scale=jnp.full_like(mean, self.diagonal_jitter),
            factors=factor,
            query=batch.require_single_query(),
            output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
            uncertainty_source="process",
        )


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
) -> phx.stochastic.StochasticTransitionView:
    """Generate canonical stochastic-heat trajectory transitions."""
    size, cases, draws = int(grid_size), int(num_cases), int(num_realizations)
    if size < 3 or cases <= 0 or draws <= 0:
        raise ValueError(
            "grid_size >= 3 and positive case/realization counts are required."
        )
    if duration <= 0.0 or diffusivity <= 0.0 or dt0 <= 0.0:
        raise ValueError("duration, diffusivity, and dt0 must be positive.")
    initial_key, realization_key = jr.split(key)
    initial = _smooth_initial_states(initial_key, cases, size)
    axis_discretization = phx.discretization.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    spatial = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((axis_discretization,), axis_names=("x",))
    )
    noise_basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        spatial,
        float(noise_scale) ** 2 / float(size),
        rank=int(noise_rank),
    )
    drift = float(diffusivity) * spatial.laplacian_matrix()
    noise = noise_basis.diffusion_matrix
    save_time = jnp.asarray([duration])
    trajectories: list[phx.stochastic.StochasticTrajectory] = []
    for case in range(cases):
        spde = phx.solver.semidiscretize_reaction_diffusion(
            initial[case],
            spatial,
            t0=0.0,
            t1=duration,
            kappa=float(diffusivity),
            noise_basis=noise_basis,
        )
        realization = spde.wiener_realization(
            jr.fold_in(realization_key, case),
            sample_shape=(draws,),
            tolerance=min(0.5 * float(dt0), 1e-3),
            label=f"heat-case-{case}",
        )
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=save_time,
            realization=realization,
            dt0=dt0,
        )
        trajectories.append(
            solution.to_stochastic_trajectory(
                initial_state=initial[case],
                initial_time=0.0,
                realization_axes=("realization",),
                state_axes=("x",),
                case_id=f"heat-case:{case}",
                discretization_id=spatial.discretization_id,
                basis_id=noise_basis.basis_id,
            )
        )
    mean, covariance = _linear_transition_moments(
        drift,
        noise,
        initial,
        float(duration),
    )
    axis = phx.nn.operator.OperatorAxis(
        "x",
        axis_discretization.nodes,
        quadrature_weights=axis_discretization.quad_weights,
        periodic=True,
    )
    trajectory = phx.stochastic.StochasticTrajectory.stack_cases(
        trajectories,
        case_axis="case",
        metadata={
            "operator_axis": axis,
            "drift_matrix": drift,
            "noise_matrix": noise,
            "analytic_mean": mean,
            "analytic_covariance": covariance,
        },
    )
    return trajectory.adjacent_transitions()


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
) -> phx.stochastic.StochasticTransitionView:
    """Generate canonical stochastic Allen--Cahn trajectory transitions."""
    size, cases, draws = int(grid_size), int(num_cases), int(num_realizations)
    initial_key, realization_key = jr.split(key)
    initial = 0.35 * _smooth_initial_states(initial_key, cases, size)
    axis_discretization = phx.discretization.UniformAxisSpec(
        size,
        endpoint=False,
        periodic=True,
    ).materialize(0.0, 1.0)
    spatial = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((axis_discretization,), axis_names=("x",))
    )
    noise_basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        spatial,
        float(noise_scale) ** 2 / float(size),
        rank=int(noise_rank),
    )
    drift = float(diffusivity) * spatial.laplacian_matrix()
    noise = noise_basis.diffusion_matrix
    trajectories: list[phx.stochastic.StochasticTrajectory] = []
    for case in range(cases):
        spde = phx.solver.semidiscretize_reaction_diffusion(
            initial[case],
            spatial,
            t0=0.0,
            t1=duration,
            kappa=float(diffusivity),
            reaction=lambda t, state, args: state - state**3,
            reaction_id="allen-cahn-cubic-reaction-v1",
            noise_basis=noise_basis,
        )
        realization = spde.wiener_realization(
            jr.fold_in(realization_key, case),
            sample_shape=(draws,),
            tolerance=min(0.5 * float(dt0), 1e-3),
            label=f"allen-cahn-case-{case}",
        )
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=jnp.asarray([duration]),
            realization=realization,
            dt0=dt0,
        )
        trajectories.append(
            solution.to_stochastic_trajectory(
                initial_state=initial[case],
                initial_time=0.0,
                realization_axes=("realization",),
                state_axes=("x",),
                case_id=f"allen-cahn-case:{case}",
                discretization_id=spatial.discretization_id,
                basis_id=noise_basis.basis_id,
            )
        )
    axis = phx.nn.operator.OperatorAxis(
        "x",
        axis_discretization.nodes,
        quadrature_weights=axis_discretization.quad_weights,
        periodic=True,
    )
    trajectory = phx.stochastic.StochasticTrajectory.stack_cases(
        trajectories,
        case_axis="case",
        metadata={
            "operator_axis": axis,
            "drift_matrix": drift,
            "noise_matrix": noise,
        },
    )
    return trajectory.adjacent_transitions()


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
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis(sample_dim, "process"),),
        field_name="output",
        query_name="query",
    )


def run_stochastic_heat_gaussian_benchmark(
    key: Key[Array, ""],
    /,
    *,
    data: phx.stochastic.StochasticTransitionView | None = None,
    evaluation_samples: int = 256,
    jitter: float = 1e-4,
) -> StochasticHeatGaussianBenchmarkResult:
    """Compare diagonal and coherent low-rank Gaussian transition baselines."""
    dataset = (
        stochastic_heat_transition_data(jr.fold_in(key, 0)) if data is None else data
    )
    mean = _metadata(dataset, "analytic_mean")
    covariance = _metadata(dataset, "analytic_covariance")
    if mean is None or covariance is None:
        raise ValueError("The stochastic-heat benchmark requires analytic moments.")
    batch = _evaluation_batch(dataset)
    query = batch.require_single_query()
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    keep = eigenvalues > max(float(jitter) ** 2, 1e-12)
    factors = eigenvectors[:, keep] * jnp.sqrt(jnp.maximum(eigenvalues[keep], 0.0))
    factors = jnp.broadcast_to(
        factors,
        (dataset.num_cases,) + factors.shape,
    )
    fixed_scale = jnp.full_like(mean, float(jitter))
    low_rank = phx.nn.operator.GaussianOperatorDistribution(
        mean=mean,
        scale=fixed_scale,
        factors=factors,
        query=query,
        output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
        uncertainty_source="process",
    )
    diagonal_scale = jnp.broadcast_to(
        jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0) + float(jitter) ** 2),
        mean.shape,
    )
    diagonal = phx.nn.operator.GaussianOperatorDistribution(
        mean=mean,
        scale=diagonal_scale,
        factors=None,
        query=query,
        output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
        uncertainty_source="process",
    )
    diagonal_key, low_rank_key = jr.split(jr.fold_in(key, 1))
    count = int(evaluation_samples)
    reference = jnp.moveaxis(_final_states(dataset), 1, 0)
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
    grid_size = _axis(dataset).size
    true_covariance = covariance + float(jitter) ** 2 * jnp.eye(grid_size)
    diagonal_error = jnp.linalg.norm(diagonal.dense_covariance()[0] - true_covariance)
    low_rank_error = jnp.linalg.norm(low_rank.dense_covariance()[0] - true_covariance)
    final_states = _final_states(dataset)
    empirical_mean = jnp.mean(final_states, axis=1)
    location_rmse = jnp.sqrt(jnp.mean((empirical_mean - mean) ** 2))
    fine_data = stochastic_heat_transition_data(
        jr.fold_in(key, 2),
        grid_size=2 * grid_size,
        num_cases=1,
        num_realizations=2,
        duration=dataset.duration,
        noise_rank=int(_metadata(dataset, "noise_matrix").shape[-1]),
        dt0=min(dataset.duration / 10.0, 5e-3),
    )
    fine_covariance = _metadata(fine_data, "analytic_covariance")
    if fine_covariance is None:
        raise AssertionError(
            "Stochastic-heat data unexpectedly omitted analytic covariance."
        )
    fine_finite = bool(
        jnp.all(jnp.isfinite(_final_states(fine_data)))
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
class StochasticHeatProcessBenchmarkResult:
    direct_mean_relative_error: float
    direct_covariance_relative_error: float
    rollout_mean_relative_error: float
    rollout_covariance_relative_error: float
    semigroup_error: float
    replay_exact: bool
    predictive_process_axis: bool

    @property
    def passed(self) -> bool:
        return (
            self.direct_mean_relative_error < 1e-5
            and self.direct_covariance_relative_error < 1e-5
            and self.rollout_mean_relative_error < 0.12
            and self.rollout_covariance_relative_error < 0.25
            and self.semigroup_error < 0.2
            and self.replay_exact
            and self.predictive_process_axis
        )


def run_stochastic_heat_process_benchmark(
    key: Key[Array, ""],
    /,
    *,
    data: phx.stochastic.StochasticTransitionView | None = None,
    num_realizations: int = 2048,
) -> StochasticHeatProcessBenchmarkResult:
    """Validate an exact stochastic-heat transition as one coherent process."""
    dataset = (
        stochastic_heat_transition_data(
            jr.fold_in(key, 0),
            grid_size=8,
            num_cases=3,
            num_realizations=32,
        )
        if data is None
        else data
    )
    count = int(num_realizations)
    if count < 2:
        raise ValueError("num_realizations must be at least two.")
    drift = _metadata(dataset, "drift_matrix")
    noise = _metadata(dataset, "noise_matrix")
    true_mean = _metadata(dataset, "analytic_mean")
    true_covariance = _metadata(dataset, "analytic_covariance")
    if any(value is None for value in (drift, noise, true_mean, true_covariance)):
        raise ValueError("Stochastic-heat process validation requires analytic metadata.")

    batch = _evaluation_batch(dataset)
    initial_states = _physical_initial_states(dataset)
    transition = phx.nn.operator.training.OperatorMarginalTransition(
        LinearGaussianReferenceOperator(drift, noise),
        batch,
        phx.nn.operator.training.OperatorTransitionSpec(
            phx.nn.operator.OperatorOutputSpec("scalar")
        ),
        process_id="analytic-stochastic-heat",
    )
    direct = transition.marginal_transition(
        initial_states,
        t0=0.0,
        t1=dataset.duration,
    )
    operator_distribution = direct.operator_distribution
    if not isinstance(
        operator_distribution, phx.nn.operator.GaussianOperatorDistribution
    ):
        raise TypeError("Analytic heat transition must return a Gaussian distribution.")
    direct_covariance = operator_distribution.dense_covariance()
    expected_covariance = jnp.broadcast_to(
        true_covariance,
        direct_covariance.shape,
    )

    def relative_error(value, reference):
        return jnp.linalg.norm(value - reference) / jnp.maximum(
            jnp.linalg.norm(reference),
            1e-12,
        )

    times = jnp.asarray([0.0, 0.5 * dataset.duration, dataset.duration])
    rollout_key, objective_key = jr.split(jr.fold_in(key, 1))
    rollout = phx.nn.operator.training.marginal_operator_rollout(
        transition,
        times,
        initial_state=initial_states,
        key=rollout_key,
        num_realizations=count,
    )
    replay = phx.nn.operator.training.marginal_operator_rollout(
        transition,
        times,
        initial_state=initial_states,
        key=rollout_key,
        num_realizations=count,
    )
    final_states = rollout.states[:, :, -1]
    empirical_mean = jnp.mean(final_states, axis=1)
    centered = final_states - empirical_mean[:, None]
    empirical_covariance = oe.contract(
        "cri,crj->cij",
        centered,
        centered,
    ) / float(count - 1)
    predictive = rollout.to_predictive()
    semigroup_error = phx.stochastic.semigroup_objective(
        transition,
        initial_states,
        t0=0.0,
        tmid=0.5 * dataset.duration,
        t1=dataset.duration,
        key=objective_key,
        num_samples=min(count, 1024),
    )
    return StochasticHeatProcessBenchmarkResult(
        float(relative_error(direct.location, true_mean)),
        float(relative_error(direct_covariance, expected_covariance)),
        float(relative_error(empirical_mean, true_mean)),
        float(relative_error(empirical_covariance, expected_covariance)),
        float(semigroup_error),
        bool(jnp.array_equal(rollout.states, replay.states)),
        (
            len(predictive.sample_axes) == 1
            and predictive.sample_axes[0].source == "process"
        ),
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
    data: phx.stochastic.StochasticTransitionView,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    evaluation_samples: int,
) -> AllenCahnFlowBenchmarkTrial:
    split = max(1, data.num_cases - max(1, data.num_cases // 4))
    train = _operator_dataset(data, range(split))
    evaluation_indices = tuple(range(split, data.num_cases))
    evaluation_batch = _evaluation_batch(data, evaluation_indices)
    reference = jnp.moveaxis(_final_states(data)[jnp.asarray(evaluation_indices)], 1, 0)
    if reference.shape[0] > evaluation_samples:
        reference = reference[:evaluation_samples]
    grid_size = _axis(data).size
    location = phx.nn.operator.architectures.FNO(
        n_modes=(max(1, grid_size // 3),),
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(seed),
    )
    encoder = phx.nn.operator.architectures.FixedBranchEncoder(
        phx.nn.models.MLP(
            in_size=grid_size,
            out_size=8,
            width_size=16,
            depth=1,
            key=jr.key(seed + 1),
        ),
        8,
    )
    conditioner = phx.nn.operator.architectures.OperatorBatchConditioner(
        {"state": encoder}
    )
    flow = phx.nn.operator.architectures.conditional_coupling_flow_operator(
        jr.key(seed + 2),
        location_model=location,
        conditioner=conditioner,
        reference_query=evaluation_batch.require_single_query(),
        uncertainty_source="process",
        flow_layers=3,
        nn_width=16,
        nn_depth=1,
    )
    gaussian_base = phx.nn.operator.architectures.FNO(
        n_modes=(max(1, grid_size // 3),),
        in_channels="scalar",
        out_channels=4,
        width=8,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(seed + 3),
    )
    gaussian = phx.nn.operator.architectures.GaussianFunctionOperator(
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
    loss = (phx.nn.operator.training.OperatorDistributionNLL(),)
    gaussian_fit = phx.nn.operator.training.fit_operator(
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
    flow_fit = phx.nn.operator.training.fit_operator(
        flow,
        train,
        loss_terms=loss,
        learning_rate=1e-3,
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
        phx.nn.operator.AbstractProbabilisticOperatorModel,
    ) or not isinstance(
        fitted_flow,
        phx.nn.operator.AbstractProbabilisticOperatorModel,
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
    data: phx.stochastic.StochasticTransitionView | None = None,
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
    "LinearGaussianReferenceOperator",
    "StochasticHeatGaussianBenchmarkResult",
    "StochasticHeatProcessBenchmarkResult",
    "allen_cahn_transition_data",
    "run_allen_cahn_flow_benchmark",
    "run_stochastic_heat_gaussian_benchmark",
    "run_stochastic_heat_process_benchmark",
    "stochastic_heat_transition_data",
]
