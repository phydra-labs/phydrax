from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Key

import phydrax as phx


@dataclass(frozen=True)
class LevyStableReferenceBenchmarkResult:
    jump_count_mean_relative_error: float
    jump_count_variance_relative_error: float
    characteristic_function_max_error: float
    complete_path_fraction: float

    @property
    def passed(self) -> bool:
        return (
            self.jump_count_mean_relative_error < 0.06
            and self.jump_count_variance_relative_error < 0.08
            and self.characteristic_function_max_error < 0.035
            and self.complete_path_fraction == 1.0
        )


@dataclass(frozen=True)
class FractionalRoughReferenceBenchmarkResult:
    covariance_relative_error: float
    self_similarity_relative_error: float
    rough_linear_relative_rmse: float
    chen_identity_error: float

    @property
    def passed(self) -> bool:
        return (
            self.covariance_relative_error < 0.12
            and self.self_similarity_relative_error < 0.08
            and self.rough_linear_relative_rmse < 4e-3
            and self.chen_identity_error < 1e-11
        )


@dataclass(frozen=True)
class MemoryParticleReferenceBenchmarkResult:
    volterra_mean_error: float
    volterra_variance_relative_error: float
    delay_max_error: float
    particle_mean_error: float
    particle_contraction_error: float

    @property
    def passed(self) -> bool:
        return (
            self.volterra_mean_error < 0.04
            and self.volterra_variance_relative_error < 0.1
            and self.delay_max_error < 2.1e-3
            and self.particle_mean_error < 1e-12
            and self.particle_contraction_error < 1e-12
        )


def run_levy_stable_reference_benchmark(
    key: Key[Array, ""],
    /,
    *,
    num_paths: int = 8192,
) -> LevyStableReferenceBenchmarkResult:
    """Check Poisson tails and Gaussian-closed increments against stable-law formulas."""
    duration = 0.75
    cutoff = 0.06
    process = phx.stochastic.SymmetricStableLevyProcess(
        1.35,
        0.3,
        drift=0.08,
        process_id="stable-reference-driver",
    )
    realization = phx.stochastic.LevyProcessRealization.from_process(
        process,
        key,
        support=(0.0, duration),
        max_terms=128,
        sample_shape=(int(num_paths),),
        gaussian_tolerance=1e-5,
    )
    series = realization.series(process)
    counts = series.num_jumps_above(cutoff)
    expected_count = (
        duration * process.total_tail_coefficient * cutoff ** (-process.alpha)
    )
    count_mean_error = float(jnp.abs(jnp.mean(counts) - expected_count) / expected_count)
    count_variance_error = float(
        jnp.abs(jnp.var(counts, ddof=1) - expected_count) / expected_count
    )
    truncated = realization.truncated_increments(
        process,
        jnp.asarray([0.0]),
        jnp.asarray([duration]),
        cutoff=cutoff,
    )[:, 0, 0]
    gaussian = realization.gaussian_realization().increments(
        jnp.asarray([0.0]),
        jnp.asarray([duration]),
    )[:, 0, 0]
    small_variance_rate = process.small_jump_covariance(cutoff)[0, 0]
    closed_increment = truncated + jnp.sqrt(small_variance_rate) * gaussian
    frequencies = jnp.asarray([0.4, 0.8, 1.2])
    empirical = jnp.mean(
        jnp.exp(1j * closed_increment[:, None] * frequencies[None, :]),
        axis=0,
    )
    exact = jnp.exp(duration * process.characteristic_exponent(frequencies[:, None]))
    characteristic_error = float(jnp.max(jnp.abs(empirical - exact)))
    complete_fraction = float(jnp.mean(series.complete_above(cutoff)))
    return LevyStableReferenceBenchmarkResult(
        count_mean_error,
        count_variance_error,
        characteristic_error,
        complete_fraction,
    )


def run_fractional_rough_reference_benchmark(
    key: Key[Array, ""],
    /,
    *,
    num_paths: int = 1024,
) -> FractionalRoughReferenceBenchmarkResult:
    """Check fractional covariance, self-similarity, Chen composition, and an exact RDE."""
    hurst = 0.7
    scale = 0.6
    grid = jnp.linspace(0.0, 1.0, 65)
    process = phx.stochastic.FractionalGaussianProcess(
        hurst,
        scale,
        process_id="fractional-reference-driver",
    )
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        key,
        grid,
        sample_shape=(int(num_paths),),
    )
    selected_indices = jnp.asarray([0, 16, 32, 48, 64])
    selected_times = grid[selected_indices]
    selected_values = realization.values[:, selected_indices, 0]
    empirical_covariance = jnp.cov(selected_values, rowvar=False)
    exact_covariance = process.covariance(
        selected_times[:, None],
        selected_times[None, :],
    )[..., 0, 0]
    covariance_error = float(
        jnp.linalg.norm(empirical_covariance - exact_covariance)
        / jnp.linalg.norm(exact_covariance)
    )
    variance_half = jnp.var(realization.values[:, 32, 0], ddof=1)
    variance_one = jnp.var(realization.values[:, 64, 0], ddof=1)
    exact_ratio = 2.0 ** (2.0 * hurst)
    self_similarity_error = float(
        jnp.abs(variance_one / variance_half - exact_ratio) / exact_ratio
    )
    rough_path = phx.stochastic.GeometricRoughPath.from_fractional_gaussian(realization)
    rate = 0.55
    initial = 1.2
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: rate * state[..., None],
        jnp.asarray([initial]),
        driver_dimension=1,
        problem_id="fractional-linear-reference-rde",
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        rough_path,
        save_times=jnp.asarray([1.0]),
    )
    exact_terminal = initial * jnp.exp(rate * realization.values[:, -1, 0])
    rough_error = float(
        jnp.sqrt(jnp.mean((solution.states[:, -1, 0] - exact_terminal) ** 2))
        / jnp.sqrt(jnp.mean(exact_terminal**2))
    )
    midpoint = rough_path.num_steps // 2
    left = rough_path.signature(0, midpoint)
    right = rough_path.signature(midpoint, rough_path.num_steps)
    composed = phx.stochastic.compose_rough_path_segments(*left, *right)
    terminal = rough_path.terminal_signature
    chen_error = float(
        jnp.maximum(
            jnp.max(jnp.abs(composed[0] - terminal[0])),
            jnp.max(jnp.abs(composed[1] - terminal[1])),
        )
    )
    return FractionalRoughReferenceBenchmarkResult(
        covariance_error,
        self_similarity_error,
        rough_error,
        chen_error,
    )


def run_memory_particle_reference_benchmark(
    key: Key[Array, ""],
    /,
    *,
    num_paths: int = 4096,
) -> MemoryParticleReferenceBenchmarkResult:
    """Check a solvable stochastic convolution, delay equation, and mean-field flow."""
    volterra_times = jnp.linspace(0.0, 1.0, 33)
    exponent = 0.25
    realization = phx.stochastic.WienerRealization(
        key,
        (1,),
        support=(0.0, 1.0),
        sample_shape=(int(num_paths),),
        tolerance=1e-5,
        noise_id="volterra-reference-noise",
    )
    volterra_problem = phx.solver.StochasticVolterraProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.3]),
        t0=0.0,
        t1=1.0,
        diffusion=lambda time, state, args: jnp.ones((1, 1)),
        diffusion_kernel=lambda target, source, args: (target - source) ** exponent,
        noise_shape=(1,),
        noise_id="volterra-reference-noise",
        problem_id="power-kernel-volterra-reference",
    )
    volterra = phx.solver.solve_stochastic_volterra(
        volterra_problem,
        times=volterra_times,
        realization=realization,
    )
    terminal = volterra.states[:, -1, 0]
    exact_variance = 1.0 / (2.0 * exponent + 1.0)
    mean_error = float(jnp.abs(jnp.mean(terminal) - 0.3))
    variance_error = float(
        jnp.abs(jnp.var(terminal, ddof=1) - exact_variance) / exact_variance
    )

    delay = 0.4
    delay_times = jnp.linspace(0.0, 0.8, 81)
    delay_problem = phx.solver.StochasticDelayProblem(
        lambda time, state, delayed, args: delayed[0],
        lambda time, args: jnp.asarray([1.0]),
        jnp.asarray([delay]),
        t0=0.0,
        t1=0.8,
        problem_id="piecewise-polynomial-delay-reference",
    )
    delay_solution = phx.solver.solve_stochastic_delay(
        delay_problem,
        times=delay_times,
    )
    exact_delay = 1.0 + delay_times + 0.5 * jnp.maximum(delay_times - delay, 0.0) ** 2
    delay_error = float(jnp.max(jnp.abs(delay_solution.states[:, 0] - exact_delay)))

    initial_particles = jnp.asarray([[-1.0], [0.0], [2.0], [3.0]])
    particle_times = jnp.linspace(0.0, 1.0, 11)
    particle_problem = phx.solver.InteractingParticleProblem(
        lambda time, state, law, args: law.mean - state,
        initial_particles,
        t0=0.0,
        t1=1.0,
        problem_id="mean-attraction-reference",
    )
    particle_solution = phx.solver.solve_interacting_particles(
        particle_problem,
        times=particle_times,
    )
    initial_mean = jnp.mean(initial_particles, axis=0)
    exact_particles = initial_mean + 0.9**10 * (initial_particles - initial_mean)
    particle_mean_error = float(jnp.max(jnp.abs(particle_solution.means - initial_mean)))
    particle_contraction_error = float(
        jnp.max(jnp.abs(particle_solution.particles[-1] - exact_particles))
    )
    return MemoryParticleReferenceBenchmarkResult(
        mean_error,
        variance_error,
        delay_error,
        particle_mean_error,
        particle_contraction_error,
    )


__all__ = [
    "FractionalRoughReferenceBenchmarkResult",
    "LevyStableReferenceBenchmarkResult",
    "MemoryParticleReferenceBenchmarkResult",
    "run_fractional_rough_reference_benchmark",
    "run_levy_stable_reference_benchmark",
    "run_memory_particle_reference_benchmark",
]
