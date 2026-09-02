from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import cast

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

import phydrax as phx


@dataclass(frozen=True)
class StochasticHeatConvergenceBenchmarkResult:
    temporal: phx.solver.SPDEConvergenceStudy
    spatial: phx.solver.SPDEConvergenceStudy
    noise_truncation: phx.solver.NoiseTruncationStudy
    ensemble: phx.solver.SPDEConvergenceStudy
    finite_time_covariance_relative_error: float
    stationary_covariance_relative_error: float

    @property
    def passed(self) -> bool:
        return (
            self.temporal.regression_rate("strong") > 0.7
            and self.spatial.regression_rate("strong") > 1.7
            and self.ensemble.regression_rate("sampling") > 0.5
            and self.finite_time_covariance_relative_error < 0.2
            and self.stationary_covariance_relative_error < 0.2
            and all(
                left.finite_horizon_solution_residual
                >= right.finite_horizon_solution_residual
                for left, right in zip(
                    self.noise_truncation.levels,
                    self.noise_truncation.levels[1:],
                )
            )
        )


def _periodic_heat_problem(
    size: int,
    /,
    *,
    diffusivity: float,
    noise_rank: int,
    noise_scale: float,
):
    axis = phx.discretization.UniformAxisSpec(
        int(size), endpoint=False, periodic=True
    ).materialize(0.0, 1.0)
    spatial = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    )
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        spatial,
        float(noise_scale) ** 2 / float(size),
        rank=int(noise_rank),
    )
    return axis, spatial, basis


def _modal_coefficients(states: Array, modes: Array, weights: Array, /) -> Array:
    return oe.contract("pr,p,np->nr", modes, weights, states)


def run_stochastic_heat_convergence_benchmark(
    key: Key[Array, ""],
    /,
    *,
    temporal_paths: int = 256,
    moment_paths: int = 2048,
) -> StochasticHeatConvergenceBenchmarkResult:
    """Exercise time, space, noise-rank, and ensemble refinement independently."""
    duration = 0.05
    diffusivity = 0.01
    axis, spatial, basis = _periodic_heat_problem(
        8,
        diffusivity=diffusivity,
        noise_rank=3,
        noise_scale=0.25,
    )
    spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros(spatial.state_shape),
        spatial,
        t0=0.0,
        t1=duration,
        kappa=diffusivity,
        noise_basis=basis,
    )
    realization = spde.wiener_realization(
        jr.fold_in(key, 0),
        sample_shape=(int(temporal_paths),),
        tolerance=2e-5,
        label="heat-time-convergence",
    )
    reference_dt = 3.125e-4
    reference = phx.solver.solve_diffrax_ensemble(
        spde.problem,
        save_times=jnp.asarray([duration]),
        realization=realization,
        dt0=reference_dt,
    ).states[:, -1]
    reference_energy = float(jnp.mean(jnp.mean(reference**2, axis=-1)))
    temporal_levels: list[phx.solver.SPDEConvergenceLevel] = []
    for step in (2.5e-3, 1.25e-3, 6.25e-4):
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=jnp.asarray([duration]),
            realization=realization,
            dt0=step,
        )
        terminal = solution.states[:, -1]
        strong = phx.solver.coupled_strong_error(
            terminal,
            reference,
            quadrature_weights=spatial.quadrature_weights,
        )
        per_path = jnp.sqrt(jnp.mean((terminal - reference) ** 2, axis=-1))
        squared_path_error = per_path**2
        strong_standard_error = float(
            jnp.std(squared_path_error, ddof=1)
            / (2.0 * max(strong, 1e-14) * jnp.sqrt(float(squared_path_error.size)))
        )
        weak = phx.solver.weak_observable_estimate(
            terminal,
            lambda value: jnp.mean(value**2),
            reference_energy,
            name="energy",
        )
        temporal_levels.append(
            phx.solver.SPDEConvergenceLevel(
                step,
                work=duration / step,
                strong_error=strong,
                pathwise_error=float(jnp.max(per_path)),
                weak_estimates=(weak,),
                error_budget=phx.solver.SPDEErrorBudget(
                    temporal=strong,
                    sampling=strong_standard_error,
                ),
                mean_square=float(jnp.mean(terminal**2)),
                realization_id=realization.realization_id,
                coupling_id=realization.coupling_id,
                provenance={
                    "discretization": spatial.discretization_id,
                    "basis": basis.basis_id,
                },
            )
        )
    temporal = phx.solver.SPDEConvergenceStudy(
        "time",
        temporal_levels,
        reference_id=f"diffrax-euler:{reference_dt}",
    )

    spatial_levels: list[phx.solver.SPDEConvergenceLevel] = []
    for size in (8, 16, 32):
        spatial_axis = phx.discretization.UniformAxisSpec(
            size, endpoint=False, periodic=True
        ).materialize(0.0, 1.0)
        discretization = phx.discretization.periodic_finite_difference(
            phx.discretization.PreparedTensorGrid((spatial_axis,), axis_names=("x",))
        )
        points = spatial_axis.nodes
        initial = jnp.sin(2.0 * jnp.pi * points)
        deterministic = phx.solver.semidiscretize_reaction_diffusion(
            initial,
            discretization,
            t0=0.0,
            t1=0.2,
            kappa=0.03,
        )
        numerical = phx.solver.solve_semilinear_spde(
            deterministic,
            save_times=jnp.asarray([0.2]),
            dt=0.2,
            matrix_function_policy=phx.linalg.MatrixFunctionPolicy(
                "lanczos", max_dimension=size
            ),
        ).states[-1]
        exact = jnp.exp(-0.03 * (2.0 * jnp.pi) ** 2 * 0.2) * initial
        error = float(
            jnp.sqrt(jnp.sum(spatial_axis.quad_weights * (numerical - exact) ** 2))
        )
        spatial_levels.append(
            phx.solver.SPDEConvergenceLevel(
                1.0 / float(size),
                work=float(size),
                strong_error=error,
                error_budget=phx.solver.SPDEErrorBudget(spatial=error),
                mean_square=float(jnp.sum(spatial_axis.quad_weights * numerical**2)),
                realization_id="deterministic-periodic-mode",
                coupling_id="deterministic-periodic-mode",
                provenance={"discretization": discretization.discretization_id},
            )
        )
    spatial_study = phx.solver.SPDEConvergenceStudy(
        "space",
        spatial_levels,
        reference_id="continuous-periodic-heat-mode",
    )

    frequencies = jnp.arange(64, dtype=float)
    laplacian_spectrum = (2.0 * jnp.pi * frequencies) ** 2
    covariance_spectrum = 0.02 / (1.0 + laplacian_spectrum) ** 1.25
    linear_spectrum = -0.03 * laplacian_spectrum
    noise_study = phx.solver.NoiseTruncationStudy.from_compatible_spectrum(
        covariance_spectrum,
        linear_spectrum,
        (2, 4, 8, 16, 32, 64),
        horizon=0.2,
        operator_id="continuous-periodic-heat",
        basis_id="analytic-periodic-Q",
        observable_mode_weights={
            "spatial_mean": jnp.concatenate((jnp.ones((1,)), jnp.zeros((63,))))
        },
    )

    modal_eigenvalues, modal_modes = spatial.eigenpairs(rank=basis.rank)
    linear_modes = -diffusivity * modal_eigenvalues
    factors = jnp.where(
        jnp.abs(linear_modes) > 1e-12,
        jnp.expm1(2.0 * linear_modes * duration) / (2.0 * linear_modes),
        duration,
    )
    analytic_variance = basis.eigenvalues * factors
    normals = jr.normal(
        jr.fold_in(key, 1),
        (int(moment_paths), basis.rank),
    )
    modal_samples = normals * jnp.sqrt(analytic_variance)
    ensemble_levels: list[phx.solver.SPDEConvergenceLevel] = []
    exact_energy = float(jnp.sum(analytic_variance))
    for count in (64, 256, min(1024, int(moment_paths))):
        sample = modal_samples[:count]
        empirical_variance = jnp.var(sample, axis=0)
        error = float(
            jnp.linalg.norm(empirical_variance - analytic_variance)
            / jnp.maximum(jnp.linalg.norm(analytic_variance), 1e-14)
        )
        weak = phx.solver.weak_observable_estimate(
            sample,
            lambda value: jnp.sum(value**2),
            exact_energy,
            name="modal_energy",
        )
        ensemble_levels.append(
            phx.solver.SPDEConvergenceLevel(
                1.0 / sqrt(float(count)),
                work=float(count),
                weak_estimates=(weak,),
                error_budget=phx.solver.SPDEErrorBudget(sampling=error),
                mean_square=float(jnp.mean(jnp.sum(sample**2, axis=-1))),
                provenance={"prefix_ensemble": "heat-modal-samples"},
            )
        )
    ensemble = phx.solver.SPDEConvergenceStudy(
        "ensemble",
        ensemble_levels,
        reference_id="analytic-modal-covariance",
    )
    finite_covariance_error = ensemble.levels[-1].error_budget
    assert finite_covariance_error is not None

    stationary_q = basis.eigenvalues.at[0].set(0.0)
    stationary_basis = phx.stochastic.SpatialNoiseBasis.from_modes(
        modal_modes,
        stationary_q,
        quadrature_weights=spatial.quadrature_weights,
        state_shape=spatial.state_shape,
        mode_ids=tuple(f"stationary:{index}" for index in range(basis.rank)),
        field_space_id=spatial.field_spaces[0].field_space_id,
    )
    stationary_spde = phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros(spatial.state_shape),
        spatial,
        t0=0.0,
        t1=8.0,
        kappa=diffusivity,
        noise_basis=stationary_basis,
    )
    stationary_realization = stationary_spde.wiener_realization(
        jr.fold_in(key, 2),
        sample_shape=(int(moment_paths),),
        label="heat-stationary",
    )
    stationary_solution = phx.solver.solve_semilinear_spde(
        stationary_spde,
        save_times=jnp.asarray([8.0]),
        realization=stationary_realization,
        dt=8.0,
        matrix_function_policy=phx.linalg.MatrixFunctionPolicy(
            "lanczos", max_dimension=spatial.num_points
        ),
    )
    stationary_coefficients = _modal_coefficients(
        stationary_solution.states[:, -1],
        modal_modes,
        spatial.quadrature_weights,
    )
    empirical_stationary = jnp.var(stationary_coefficients[:, 1:], axis=0)
    analytic_stationary = stationary_q[1:] / (-2.0 * linear_modes[1:])
    stationary_error = float(
        jnp.linalg.norm(empirical_stationary - analytic_stationary)
        / jnp.linalg.norm(analytic_stationary)
    )
    return StochasticHeatConvergenceBenchmarkResult(
        temporal,
        spatial_study,
        noise_study,
        ensemble,
        finite_covariance_error.sampling,
        stationary_error,
    )


@dataclass(frozen=True)
class StochasticAdvectionDiffusionBenchmarkResult:
    mean_relative_error: float
    covariance_relative_error: float
    phase_error: float
    derivative_noise_variance_error: float
    stratonovich_correction_error: float

    @property
    def passed(self) -> bool:
        return (
            self.mean_relative_error < 0.04
            and self.covariance_relative_error < 0.08
            and self.phase_error < 0.04
            and self.derivative_noise_variance_error < 0.08
            and self.stratonovich_correction_error < 1e-10
        )


def run_stochastic_advection_diffusion_benchmark(
    key: Key[Array, ""],
    /,
    *,
    num_samples: int = 32768,
) -> StochasticAdvectionDiffusionBenchmarkResult:
    """Periodic phase transport with additive derivative noise and drift conversion."""
    wavenumber = 2.0 * jnp.pi
    advection = 0.7
    viscosity = 0.08
    noise_scale = 0.3
    duration = 0.4
    decay = viscosity * wavenumber**2
    frequency = advection * wavenumber
    drift = jnp.asarray([[-decay, -frequency], [frequency, -decay]])
    initial = jnp.asarray([1.0, -0.2])
    propagator = jax.scipy.linalg.expm(duration * drift)
    exact_mean = propagator @ initial
    variance = (
        noise_scale**2
        * wavenumber**2
        * (-jnp.expm1(-2.0 * decay * duration))
        / (2.0 * decay)
    )
    samples = exact_mean + jnp.sqrt(variance) * jr.normal(key, (int(num_samples), 2))
    empirical_mean = jnp.mean(samples, axis=0)
    empirical_covariance = jnp.cov(samples, rowvar=False)
    exact_covariance = variance * jnp.eye(2)
    mean_error = float(
        jnp.linalg.norm(empirical_mean - exact_mean) / jnp.linalg.norm(exact_mean)
    )
    covariance_error = float(
        jnp.linalg.norm(empirical_covariance - exact_covariance)
        / jnp.linalg.norm(exact_covariance)
    )
    exact_phase = jnp.arctan2(exact_mean[1], exact_mean[0])
    empirical_phase = jnp.arctan2(empirical_mean[1], empirical_mean[0])
    phase_error = float(jnp.abs(jnp.angle(jnp.exp(1j * (empirical_phase - exact_phase)))))
    derivative_variance_error = float(
        jnp.abs(jnp.trace(empirical_covariance) - 2.0 * variance) / (2.0 * variance)
    )

    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=4.0).compile()
    )
    zero_drift = domain.Function("x")(lambda x: jnp.zeros((2,)))
    rotation = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    angular_noise = 0.6
    diffusion = domain.Function("x")(lambda x: (angular_noise * (rotation @ x))[:, None])
    corrected = phx.operators.stratonovich_to_ito_drift(zero_drift, diffusion)
    point = jnp.asarray([0.4, -0.7])
    expected_correction = -0.5 * angular_noise**2 * point
    correction_error = float(jnp.linalg.norm(corrected.func(point) - expected_correction))
    return StochasticAdvectionDiffusionBenchmarkResult(
        mean_error,
        covariance_error,
        phase_error,
        derivative_variance_error,
        correction_error,
    )


@dataclass(frozen=True)
class MultiplicativeReactionDiffusionBenchmarkResult:
    temporal: phx.solver.SPDEConvergenceStudy
    analytic_mean_relative_error: float
    analytic_second_moment_relative_error: float
    positive_fraction: float

    @property
    def passed(self) -> bool:
        return (
            self.temporal.regression_rate("strong") > 0.35
            and self.temporal.regression_rate(observable="mean") > 0.5
            and self.analytic_mean_relative_error < 0.08
            and self.analytic_second_moment_relative_error < 0.15
            and self.positive_fraction > 0.999
        )


def _brownian_terminal_values(
    realization: phx.stochastic.WienerRealization,
    duration: float,
    /,
) -> Array:
    keys = realization.path_keys.reshape((-1,))
    signs = realization.path_signs.reshape((-1,))

    def one(path_key, sign):
        tree = dfx.VirtualBrownianTree(
            t0=realization.support[0],
            t1=realization.support[1],
            tol=realization.tolerance,
            shape=jax.ShapeDtypeStruct(realization.noise_shape, jnp.float64),
            key=path_key,
            levy_area=dfx.BrownianIncrement,
        )
        return sign * cast(Array, tree.evaluate(0.0, duration))[0]

    return jax.vmap(one)(keys, signs).reshape(realization.sample_shape)


def run_multiplicative_reaction_diffusion_benchmark(
    key: Key[Array, ""],
    /,
    *,
    num_paths: int = 1024,
) -> MultiplicativeReactionDiffusionBenchmarkResult:
    """Lognormal invariant-subspace benchmark for multiplicative spatial noise."""
    size = 8
    duration = 0.2
    diffusivity = 0.02
    growth = 0.15
    noise_scale = 0.45
    axis = phx.discretization.UniformAxisSpec(
        size, endpoint=False, periodic=True
    ).materialize(0.0, 1.0)
    spatial = phx.discretization.periodic_finite_difference(
        phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    )
    mode = 1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * axis.nodes)
    constant = jnp.ones((size, 1))
    basis = phx.stochastic.SpatialNoiseBasis.from_modes(
        constant,
        jnp.ones((1,)),
        quadrature_weights=spatial.quadrature_weights,
        state_shape=spatial.state_shape,
        mode_ids=("global-amplitude",),
        field_space_id=spatial.field_spaces[0].field_space_id,
    )
    laplacian_mode = spatial.laplacian(mode)
    # The constant offset and cosine have different eigenvalues. Cancel diffusion
    # pointwise so the chosen positive profile spans one exact scalar amplitude law.
    reaction_shape = growth * mode - diffusivity * laplacian_mode
    spde = phx.solver.semidiscretize_reaction_diffusion(
        mode,
        spatial,
        t0=0.0,
        t1=duration,
        kappa=diffusivity,
        reaction=lambda t, state, args: (reaction_shape / mode) * state,
        reaction_id="lognormal-mode-linear-reaction-v1",
        noise_basis=basis,
        noise_amplitude=lambda t, state, args: noise_scale * state,
        noise_structure="commutative",
    )
    realization = spde.wiener_realization(
        key,
        sample_shape=(int(num_paths),),
        tolerance=2e-5,
        label="lognormal-reaction-diffusion",
    )
    brownian = _brownian_terminal_values(realization, duration)
    exact_amplitude = jnp.exp(
        (growth - 0.5 * noise_scale**2) * duration + noise_scale * brownian
    )
    exact_states = exact_amplitude[:, None] * mode[None, :]
    analytic_mean = jnp.exp(growth * duration)
    analytic_second = jnp.exp((2.0 * growth + noise_scale**2) * duration)
    levels: list[phx.solver.SPDEConvergenceLevel] = []
    terminal = None
    for step in (0.02, 0.01, 0.005):
        solution = phx.solver.solve_diffrax_ensemble(
            spde.problem,
            save_times=jnp.asarray([duration]),
            realization=realization,
            dt0=step,
        )
        terminal = solution.states[:, -1]
        amplitude = jnp.mean(terminal / mode[None, :], axis=-1)
        strong = phx.solver.coupled_strong_error(
            terminal,
            exact_states,
            quadrature_weights=spatial.quadrature_weights,
        )
        num_steps = int(round(duration / step))
        euler_mean = (1.0 + growth * step) ** num_steps
        euler_second = ((1.0 + growth * step) ** 2 + noise_scale**2 * step) ** num_steps
        mean_estimate = phx.solver.WeakObservableEstimate(
            "mean",
            euler_mean,
            float(analytic_mean),
            0.0,
            int(num_paths),
        )
        second_estimate = phx.solver.WeakObservableEstimate(
            "second_moment",
            euler_second,
            float(analytic_second),
            0.0,
            int(num_paths),
        )
        levels.append(
            phx.solver.SPDEConvergenceLevel(
                step,
                work=duration / step,
                strong_error=strong,
                pathwise_error=float(
                    jnp.max(jnp.sqrt(jnp.mean((terminal - exact_states) ** 2, axis=-1)))
                ),
                weak_estimates=(mean_estimate, second_estimate),
                error_budget=phx.solver.SPDEErrorBudget(
                    temporal=strong,
                    sampling=max(
                        float(jnp.std(amplitude, ddof=1) / jnp.sqrt(float(num_paths))),
                        float(jnp.std(amplitude**2, ddof=1) / jnp.sqrt(float(num_paths))),
                    ),
                ),
                mean_square=float(jnp.mean(terminal**2)),
                realization_id=realization.realization_id,
                coupling_id=realization.coupling_id,
                provenance={"basis": basis.basis_id},
            )
        )
    assert terminal is not None
    numerical_amplitude = jnp.mean(terminal / mode[None, :], axis=-1)
    analytic_mean = jnp.asarray(analytic_mean)
    analytic_second = jnp.asarray(analytic_second)
    return MultiplicativeReactionDiffusionBenchmarkResult(
        phx.solver.SPDEConvergenceStudy(
            "time",
            levels,
            reference_id="exact-lognormal-amplitude",
        ),
        float(jnp.abs(jnp.mean(numerical_amplitude) - analytic_mean) / analytic_mean),
        float(
            jnp.abs(jnp.mean(numerical_amplitude**2) - analytic_second) / analytic_second
        ),
        float(jnp.mean(numerical_amplitude > 0.0)),
    )


@dataclass(frozen=True)
class CommutativeNoiseBenchmarkResult:
    commutative_bracket_norm: float
    noncommutative_bracket_norm: float
    commutative_flow_order_error: float
    noncommutative_flow_order_error: float

    @property
    def passed(self) -> bool:
        return (
            self.commutative_bracket_norm < 1e-12
            and self.commutative_flow_order_error < 1e-12
            and self.noncommutative_bracket_norm > 0.5
            and self.noncommutative_flow_order_error > 1e-3
        )


def run_commutative_noise_benchmark() -> CommutativeNoiseBenchmarkResult:
    """Distinguish commuting noise flows from systems requiring Lévy area."""
    commuting_left = jnp.diag(jnp.asarray([0.4, -0.2]))
    commuting_right = jnp.diag(jnp.asarray([-0.3, 0.5]))
    noncommuting_left = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    noncommuting_right = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    commuting_bracket = (
        commuting_left @ commuting_right - commuting_right @ commuting_left
    )
    noncommuting_bracket = (
        noncommuting_left @ noncommuting_right - noncommuting_right @ noncommuting_left
    )
    state = jnp.asarray([0.7, -0.4])
    first_increment, second_increment = 0.3, -0.2

    def order_error(left, right):
        left_flow = jax.scipy.linalg.expm(first_increment * left)
        right_flow = jax.scipy.linalg.expm(second_increment * right)
        return jnp.linalg.norm(
            left_flow @ right_flow @ state - right_flow @ left_flow @ state
        )

    return CommutativeNoiseBenchmarkResult(
        float(jnp.linalg.norm(commuting_bracket)),
        float(jnp.linalg.norm(noncommuting_bracket)),
        float(order_error(commuting_left, commuting_right)),
        float(order_error(noncommuting_left, noncommuting_right)),
    )


@dataclass(frozen=True)
class MultilevelMonteCarloBenchmarkResult:
    estimate: phx.integration.IntegrationEstimate
    exact_expectation: float
    absolute_error: float
    equivalent_fine_paths: int
    ordinary_monte_carlo_error: float
    ordinary_monte_carlo_standard_error: float

    @property
    def passed(self) -> bool:
        diagnostics = self.estimate.diagnostics
        return (
            bool(self.estimate.successful)
            and self.absolute_error <= 3.0 * float(diagnostics.rmse_estimate)
            and bool(
                diagnostics.correction_variance_norms[-1]
                < diagnostics.correction_variance_norms[0]
            )
            and bool(jnp.all(jnp.diff(diagnostics.mean_costs) > 0.0))
        )


def run_multilevel_monte_carlo_benchmark(
    key: Key[Array, ""],
    /,
    *,
    target_rmse: float = 0.04,
    initial_samples: int = 64,
) -> MultilevelMonteCarloBenchmarkResult:
    """Compare coupled MLMC with equal-work finest-level Monte Carlo for geometric SDE."""
    drift = 0.3
    diffusion = 0.45
    duration = 1.0
    steps = (4, 8, 16, 32)
    levels = tuple(
        phx.stochastic.StochasticLevelSpec(
            f"gbm-{index}",
            index,
            refinement_axes=("time",),
            resolutions=(duration / count,),
            state_shape=(1,),
            problem_id="geometric-brownian-motion",
            observable_id="terminal-state",
            solver_id="euler-maruyama",
            approximation_id=f"steps-{count}",
            parent_level_id=None if index == 0 else f"gbm-{index - 1}",
        )
        for index, count in enumerate(steps)
    )
    hierarchy = phx.stochastic.StochasticCouplingPlan(
        levels,
        hierarchy_id="gbm-mlmc-benchmark",
    )

    def euler(normals, step):
        return jnp.prod(
            1.0 + drift * step + diffusion * jnp.sqrt(step) * normals,
            axis=-1,
        )

    def sampler(level_index, sample_indices, root_key):
        level_key = jr.fold_in(root_key, level_index)
        keys = jax.vmap(lambda index: jr.fold_in(level_key, index))(sample_indices)
        fine_steps = steps[level_index]
        fine_step = duration / fine_steps
        normals = jax.vmap(lambda sample_key: jr.normal(sample_key, (fine_steps,)))(keys)
        fine = euler(normals, fine_step)
        coarse = None
        work = fine_steps
        if level_index > 0:
            coarse_normals = jnp.sum(
                normals.reshape((normals.shape[0], fine_steps // 2, 2)),
                axis=-1,
            ) / sqrt(2.0)
            coarse = euler(coarse_normals, 2.0 * fine_step)
            work += fine_steps // 2
        return phx.integration.MultilevelSampleBatch(
            fine,
            coarse,
            sample_indices,
            jnp.full(sample_indices.shape, float(work)),
            level_index=level_index,
            pair_ids=sample_indices,
            provenance="gbm-euler-prefix-sampler",
        )

    target = phx.integration.multilevel(
        hierarchy,
        sampler,
        sampler_id="gbm-euler-prefix-sampler",
    )
    estimate = phx.integration.integrate(
        lambda samples, level: samples,
        target,
        phx.integration.MultilevelMonteCarloPlan(
            initial_samples=initial_samples,
            target_rmse=target_rmse,
            max_samples_per_level=200_000,
            batch_size=16_384,
            max_rounds=12,
        ),
        key=key,
    )
    diagnostics = estimate.diagnostics
    total_work = float(jnp.sum(diagnostics.attempted_counts * diagnostics.mean_costs))
    equivalent_paths = max(int(total_work // steps[-1]), 2)
    baseline_key = jr.fold_in(key, 2**31 - 1)
    baseline_normals = jr.normal(baseline_key, (equivalent_paths, steps[-1]))
    baseline_values = euler(baseline_normals, duration / steps[-1])
    baseline_mean = jnp.mean(baseline_values)
    baseline_standard_error = jnp.std(baseline_values, ddof=1) / sqrt(equivalent_paths)
    exact = float(jnp.exp(drift * duration))
    return MultilevelMonteCarloBenchmarkResult(
        estimate,
        exact,
        abs(float(estimate.value) - exact),
        equivalent_paths,
        abs(float(baseline_mean) - exact),
        float(baseline_standard_error),
    )


@dataclass(frozen=True)
class RoughLogODEConvergenceBenchmarkResult:
    interval_counts: tuple[int, ...]
    terminal_errors: tuple[float, ...]
    general_linear_relative_error: float
    depth_three_hurst_successful: bool
    accepted_logode_steps: int

    @property
    def passed(self) -> bool:
        return (
            self.terminal_errors[-1] < self.terminal_errors[0]
            and self.general_linear_relative_error < 2e-6
            and self.depth_three_hurst_successful
            and self.accepted_logode_steps > 0
        )


def run_rough_logode_convergence_benchmark(
    key: Key[Array, ""],
    /,
    *,
    fine_steps: int = 32,
) -> RoughLogODEConvergenceBenchmarkResult:
    """Instrument depth-3 log-ODE refinement for one H=0.3 rough driver."""
    steps = int(fine_steps)
    if steps < 16 or steps % 8:
        raise ValueError("fine_steps must be a multiple of eight and at least 16.")
    realization = phx.stochastic.FractionalGaussianRealization(
        phx.stochastic.FractionalGaussianProcess(
            0.3,
            0.25,
            dimension=2,
            process_id="rough-logode-convergence-driver",
        ),
        key,
        jnp.linspace(0.0, 1.0, steps + 1),
    )
    left = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    right = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])

    def vector_fields(time, state, args):
        del time, args
        return jnp.stack((left @ state, right @ state), axis=-1)

    problem = phx.solver.RoughDifferentialProblem(
        vector_fields,
        jnp.asarray([1.0, -0.25]),
        driver_dimension=2,
        problem_id="noncommuting-linear-logode-convergence",
    )
    matrix_policy = phx.linalg.MatrixFunctionPolicy("arnoldi", max_dimension=2)
    linear_solver = phx.solver.LinearLogODE(
        (left, right),
        matrix_function_policy=matrix_policy,
    )
    reference_control = phx.stochastic.LogSignatureControl.from_fractional_gaussian(
        realization,
        depth=3,
    )
    reference = phx.solver.solve_rough_differential(
        problem,
        reference_control,
        save_times=jnp.asarray([1.0]),
        solver=linear_solver,
    ).states[-1]
    interval_counts = (2, 4, 8)
    errors: list[float] = []
    coarse_control = None
    coarse_linear = None
    for count in interval_counts:
        stride = steps // count
        indices = tuple(range(0, steps + 1, stride))
        control = phx.stochastic.LogSignatureControl.from_fractional_gaussian(
            realization,
            depth=3,
            coarse_indices=indices,
        )
        solution = phx.solver.solve_rough_differential(
            problem,
            control,
            save_times=jnp.asarray([1.0]),
            solver=linear_solver,
        )
        errors.append(float(jnp.linalg.norm(solution.states[-1] - reference)))
        if count == interval_counts[0]:
            coarse_control = control
            coarse_linear = solution
    if coarse_control is None or coarse_linear is None:
        raise RuntimeError(
            "The rough log-ODE benchmark did not construct a coarse level."
        )
    general = phx.solver.solve_rough_differential(
        problem,
        coarse_control,
        save_times=jnp.asarray([1.0]),
        solver=phx.solver.LogODE(),
    )
    disagreement = float(
        jnp.linalg.norm(general.states[-1] - coarse_linear.states[-1])
        / jnp.maximum(jnp.linalg.norm(coarse_linear.states[-1]), 1e-14)
    )
    return RoughLogODEConvergenceBenchmarkResult(
        interval_counts,
        tuple(errors),
        disagreement,
        bool(general.successful),
        int(jnp.sum(general.statistics["num_accepted_steps"])),
    )


__all__ = [
    "CommutativeNoiseBenchmarkResult",
    "MultilevelMonteCarloBenchmarkResult",
    "MultiplicativeReactionDiffusionBenchmarkResult",
    "StochasticAdvectionDiffusionBenchmarkResult",
    "RoughLogODEConvergenceBenchmarkResult",
    "StochasticHeatConvergenceBenchmarkResult",
    "run_commutative_noise_benchmark",
    "run_multilevel_monte_carlo_benchmark",
    "run_rough_logode_convergence_benchmark",
    "run_multiplicative_reaction_diffusion_benchmark",
    "run_stochastic_advection_diffusion_benchmark",
    "run_stochastic_heat_convergence_benchmark",
]
