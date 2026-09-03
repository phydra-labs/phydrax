"""Deterministic qualification producer for OU-forced periodic ETDRK flow."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax.applications.incompressible_flow._forcing import (
    SolenoidalHermitianFourierBasis,
    SolenoidalOUForcingPlan,
)
from phydrax.applications.incompressible_flow._production import (
    prepare_ou_forced_periodic_method,
)
from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    HermitianSpectralCoordinates,
    PaddingDealiasingPlan,
    PeriodicLerayProjector,
    PseudospectralMethodPlan,
    TensorSpectralPlan,
)
from phydrax.equations import (
    compile_periodic_incompressible_flow,
    IncompressibleFlowProblem,
)
from phydrax.solver import ETDRKMethod
from phydrax.stochastic import OrnsteinUhlenbeckRealization


jax.config.update("jax_enable_x64", True)


def _periodic_problem(*, count: int = 4, viscosity: float = 0.04):
    space = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(tuple(AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3)))
    compiled = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(3, viscosity),
        space,
        PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2)),
    )
    coordinates = HermitianSpectralCoordinates(space, component_shape=(3,))
    base = ETDRKMethod(4).prepare(
        compiled.semilinear_drift,
        coordinates=coordinates,
    )
    basis = SolenoidalHermitianFourierBasis(
        PeriodicLerayProjector(space), maximum_wavenumber=1.1
    )
    forcing = SolenoidalOUForcingPlan(
        basis,
        correlation_time=0.4,
        rms_acceleration=0.12,
    )
    return compiled, coordinates, base, basis, forcing


def _coefficient_and_subdivision(*, sample_count: int) -> dict[str, object]:
    coordinate_count = 6
    correlation = jnp.asarray(0.4)
    end = jnp.asarray(1.2)
    ensemble = OrnsteinUhlenbeckRealization(
        jax.random.key(1301),
        (coordinate_count,),
        support=(0.0, 2.0),
        sample_shape=(sample_count,),
        tolerance=1.0e-7,
        noise_id="ou-coefficient-qualification",
    )
    zeros = jnp.zeros((sample_count, coordinate_count))
    samples = ensemble.transition(zeros, jnp.asarray(0.0), end, correlation)
    centered = samples - jnp.mean(samples, axis=0)
    covariance = centered.T @ centered / jnp.asarray(sample_count - 1)
    expected_variance = 1.0 - jnp.exp(-2.0 * end / correlation)
    covariance_error = jnp.max(
        jnp.abs(covariance - expected_variance * jnp.eye(coordinate_count))
    )

    path = OrnsteinUhlenbeckRealization(
        jax.random.key(1302),
        (coordinate_count,),
        support=(0.0, 2.0),
        tolerance=1.0e-7,
        noise_id="ou-subdivision-qualification",
    )
    initial = jnp.linspace(-0.3, 0.4, coordinate_count)
    whole = path.transition(initial, jnp.asarray(0.0), end, correlation)
    midpoint = jnp.asarray(0.45)
    subdivided = path.transition(
        path.transition(initial, jnp.asarray(0.0), midpoint, correlation),
        midpoint,
        end,
        correlation,
    )
    subdivision_error = jnp.max(jnp.abs(whole - subdivided))
    covariance_tolerance = 5.0 / jnp.sqrt(jnp.asarray(sample_count))
    passed = (covariance_error <= covariance_tolerance) & (subdivision_error <= 5.0e-11)
    return {
        "sample_count": sample_count,
        "expected_variance": float(expected_variance),
        "maximum_covariance_error": float(covariance_error),
        "covariance_tolerance": float(covariance_tolerance),
        "subdivision_error": float(subdivision_error),
        "passed": bool(passed),
    }


def _basis_spectrum_isotropy(
    basis: SolenoidalHermitianFourierBasis,
    forcing: SolenoidalOUForcingPlan,
    *,
    sample_count: int,
) -> dict[str, object]:
    realization = OrnsteinUhlenbeckRealization(
        jax.random.key(1303),
        (basis.coordinate_size,),
        support=(0.0, 2.0),
        sample_shape=(sample_count,),
        tolerance=1.0e-7,
        noise_id="ou-basis-spectrum-qualification",
    )
    zeros = jnp.zeros((sample_count, basis.coordinate_size))
    coefficients = realization.transition(
        zeros, jnp.asarray(0.0), jnp.asarray(1.8), jnp.asarray(0.4)
    )
    probe = coefficients[0]
    reconstructed = basis.evaluate(probe)
    roundtrip_error = jnp.max(jnp.abs(basis.analyze(reconstructed) - probe))
    divergence = basis.projector.divergence_norm(reconstructed)
    forbidden_energy = jnp.sum(jnp.abs(reconstructed[~basis.forced_mask]) ** 2)
    total_energy = jnp.sum(jnp.abs(reconstructed) ** 2)
    forbidden_fraction = forbidden_energy / jnp.maximum(total_energy, 1.0e-30)
    representative = reconstructed.reshape((-1, 3))[basis.representative_indices]
    polarization_coordinates = (
        jnp.abs(
            jnp.sum(
                representative[:, None, :]
                * basis.polarizations.astype(representative.dtype),
                axis=-1,
            )
        )
        ** 2
    )
    spectrum_spread = jnp.std(polarization_coordinates) / jnp.maximum(
        jnp.mean(polarization_coordinates), 1.0e-30
    )
    variances = jnp.var(forcing.coefficient_scale * coefficients, axis=0, ddof=1)
    isotropy_spread = jnp.std(variances) / jnp.maximum(jnp.mean(variances), 1.0e-30)
    forcing_fields = jax.vmap(
        lambda values: forcing.coefficient_scale * basis.evaluate(values)
    )(coefficients)
    measured_mean_square = jnp.mean(
        jnp.sum(jnp.abs(forcing_fields) ** 2, axis=(1, 2, 3, 4)) / basis.volume
    )
    stationary_fraction = 1.0 - jnp.exp(-2.0 * 1.8 / forcing.correlation_time)
    expected_mean_square = forcing.rms_acceleration**2 * stationary_fraction
    rms_normalization_error = jnp.abs(
        measured_mean_square - expected_mean_square
    ) / jnp.maximum(expected_mean_square, 1.0e-30)
    passed = (
        (roundtrip_error <= 2.0e-12)
        & (divergence <= 2.0e-12)
        & (forbidden_fraction <= 2.0e-24)
        & (isotropy_spread <= 0.15)
        & (rms_normalization_error <= 0.1)
    )
    return {
        "basis_id": basis.basis_id,
        "forcing_id": forcing.forcing_id,
        "coordinate_size": basis.coordinate_size,
        "roundtrip_error": float(roundtrip_error),
        "divergence_norm": float(divergence),
        "forbidden_spectral_energy_fraction": float(forbidden_fraction),
        "single_realization_polarization_spread": float(spectrum_spread),
        "ensemble_isotropy_spread": float(isotropy_spread),
        "measured_mean_square_acceleration": float(measured_mean_square),
        "expected_mean_square_acceleration": float(expected_mean_square),
        "rms_normalization_relative_error": float(rms_normalization_error),
        "passed": bool(passed),
    }


def _logical_randomness(*, path_count: int) -> dict[str, object]:
    device_count = max(1, min(len(jax.devices()), path_count))
    paths = max(path_count, device_count)
    paths += (-paths) % device_count
    realization = OrnsteinUhlenbeckRealization(
        jax.random.key(1304),
        (5,),
        support=(0.0, 1.0),
        sample_shape=(paths,),
        tolerance=1.0e-7,
        noise_id="ou-logical-sharding-qualification",
    )
    starts = jnp.asarray(0.1)
    ends = jnp.asarray(0.8)
    correlation = jnp.asarray(0.3)
    unsharded = realization.innovations(starts, ends, correlation)
    mesh = Mesh(np.asarray(jax.devices()[:device_count]), ("logical_path",))
    sharding = NamedSharding(mesh, PartitionSpec("logical_path", None))
    sharded = jax.jit(
        lambda: realization.innovations(starts, ends, correlation),
        out_shardings=sharding,
    )()
    maximum_error = jnp.max(jnp.abs(unsharded - sharded))
    passed = maximum_error == 0.0
    return {
        "logical_path_count": paths,
        "device_count": device_count,
        "coupling_id": realization.coupling_id,
        "maximum_sharding_difference": float(maximum_error),
        "passed": bool(passed),
    }


def _initial_velocity(basis: SolenoidalHermitianFourierBasis) -> jax.Array:
    return basis.evaluate(
        0.02 * jnp.sin(jnp.arange(basis.coordinate_size, dtype=jnp.float64) + 0.3)
    )


def _rollout(method, initial, *, step_size: float, end_time: float):
    count = int(round(end_time / step_size))
    state = method.initial_state(initial, 0.0)
    energies = []
    powers = []
    dissipations = []
    successes = []
    k_squared = method.forcing.basis.projector.wavenumber_squared
    viscosity = jnp.asarray(0.04, dtype=k_squared.dtype)
    for step_index in range(count):
        time = state.forcing_state.time
        result = method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            time,
            state,
            jnp.asarray(step_size),
            None,
        )
        state = result.accepted_state
        forcing_value = method.forcing.evaluate(state.forcing_state)
        modal_energy = 0.5 * jnp.sum(jnp.abs(state.velocity) ** 2)
        forcing_power = jnp.real(jnp.sum(jnp.conj(state.velocity) * forcing_value))
        viscous_dissipation = (
            2.0
            * viscosity
            * jnp.sum(k_squared * 0.5 * jnp.sum(jnp.abs(state.velocity) ** 2, axis=-1))
        )
        energies.append(modal_energy)
        powers.append(forcing_power)
        dissipations.append(viscous_dissipation)
        successes.append(result.successful)
    return (
        state,
        jnp.stack(tuple(energies)),
        jnp.stack(tuple(powers)),
        jnp.stack(tuple(dissipations)),
        jnp.all(jnp.stack(tuple(successes))),
    )


def _temporal_restart_and_budgets(
    *, stationary_steps: int
) -> tuple[dict[str, object], dict[str, object]]:
    _, _, base, basis, forcing = _periodic_problem()
    support_end = max(8.0, 0.02 * stationary_steps + 1.0)
    realization = OrnsteinUhlenbeckRealization(
        jax.random.key(1305),
        (basis.coordinate_size,),
        support=(0.0, support_end),
        tolerance=1.0e-7,
        noise_id="ou-fluid-fixed-path-qualification",
    )
    method = prepare_ou_forced_periodic_method(base, forcing, realization)
    initial = _initial_velocity(basis)
    horizon = 0.16
    coarse, _, _, _, coarse_success = _rollout(
        method, initial, step_size=0.04, end_time=horizon
    )
    medium, _, _, _, medium_success = _rollout(
        method, initial, step_size=0.02, end_time=horizon
    )
    fine, _, _, _, fine_success = _rollout(
        method, initial, step_size=0.01, end_time=horizon
    )
    coarse_error = jnp.sqrt(jnp.sum(jnp.abs(coarse.velocity - fine.velocity) ** 2))
    medium_error = jnp.sqrt(jnp.sum(jnp.abs(medium.velocity - fine.velocity) ** 2))
    refinement_ratio = coarse_error / jnp.maximum(medium_error, 1.0e-30)

    state = method.initial_state(initial, 0.0)
    half_steps = 4
    restart_success = jnp.asarray(True)
    for step_index in range(half_steps):
        advance = method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            state.forcing_state.time,
            state,
            jnp.asarray(0.01),
            None,
        )
        restart_success = restart_success & advance.successful
        state = advance.accepted_state
    checkpoint_velocity = jnp.array(state.velocity)
    checkpoint_coefficients = jnp.array(state.forcing_state.coefficients)
    restarted = method.initial_state(
        checkpoint_velocity,
        state.forcing_state.time,
        coefficients=checkpoint_coefficients,
    )
    direct = state
    for offset in range(half_steps):
        step_index = half_steps + offset
        time = direct.forcing_state.time
        direct_advance = method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            time,
            direct,
            jnp.asarray(0.01),
            None,
        )
        restarted_advance = method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            time,
            restarted,
            jnp.asarray(0.01),
            None,
        )
        restart_success = (
            restart_success & direct_advance.successful & restarted_advance.successful
        )
        direct = direct_advance.accepted_state
        restarted = restarted_advance.accepted_state
    restart_velocity_error = jnp.max(jnp.abs(direct.velocity - restarted.velocity))
    restart_forcing_error = jnp.max(
        jnp.abs(direct.forcing_state.coefficients - restarted.forcing_state.coefficients)
    )
    temporal_passed = (
        coarse_success
        & medium_success
        & fine_success
        & restart_success
        & (medium_error < coarse_error)
        & (refinement_ratio > 1.2)
        & (restart_velocity_error <= 2.0e-12)
        & (restart_forcing_error == 0.0)
    )
    temporal = {
        "fixed_path_coupling_id": realization.coupling_id,
        "coarse_to_fine_error": float(coarse_error),
        "medium_to_fine_error": float(medium_error),
        "refinement_ratio": float(refinement_ratio),
        "restart_velocity_error": float(restart_velocity_error),
        "restart_forcing_error": float(restart_forcing_error),
        "all_steps_successful": bool(
            coarse_success & medium_success & fine_success & restart_success
        ),
        "passed": bool(temporal_passed),
    }

    duration = 0.02 * stationary_steps
    _, energies, powers, dissipations, stationary_success = _rollout(
        method, initial, step_size=0.02, end_time=duration
    )
    burn = stationary_steps // 2
    energies = energies[burn:]
    net = powers[burn:] - dissipations[burn:]
    block_count = min(8, max(2, energies.size // 8))
    block_size = energies.size // block_count
    retained = block_count * block_size
    energy_blocks = jnp.mean(
        energies[:retained].reshape((block_count, block_size)), axis=1
    )
    net_blocks = jnp.mean(net[:retained].reshape((block_count, block_size)), axis=1)
    energy_standard_error = jnp.std(energy_blocks, ddof=1) / jnp.sqrt(block_count)
    net_standard_error = jnp.std(net_blocks, ddof=1) / jnp.sqrt(block_count)
    elapsed = jnp.asarray(0.02 * max(energies.size - 1, 1))
    energy_slope = (energies[-1] - energies[0]) / elapsed
    mean_net = jnp.mean(net)
    mean_power = jnp.mean(powers[burn:])
    mean_dissipation = jnp.mean(dissipations[burn:])
    budget_defect = mean_net - energy_slope
    budget_tolerance = (
        4.0 * net_standard_error
        + 0.1 * (jnp.abs(mean_power) + jnp.abs(mean_dissipation))
        + 1.0e-10
    )
    budget_passed = (
        stationary_success
        & jnp.all(jnp.isfinite(energies))
        & jnp.all(jnp.isfinite(net))
        & (jnp.abs(budget_defect) <= budget_tolerance)
    )
    budgets = {
        "retained_samples": int(energies.size),
        "block_count": int(block_count),
        "block_size": int(block_size),
        "mean_energy": float(jnp.mean(energies)),
        "energy_block_standard_error": float(energy_standard_error),
        "mean_forcing_power": float(mean_power),
        "mean_viscous_dissipation": float(mean_dissipation),
        "mean_power_minus_dissipation": float(mean_net),
        "net_budget_block_standard_error": float(net_standard_error),
        "energy_slope": float(energy_slope),
        "stationary_budget_defect": float(budget_defect),
        "budget_tolerance": float(budget_tolerance),
        "all_steps_successful": bool(stationary_success),
        "passed": bool(budget_passed),
    }
    return temporal, budgets


def _fixed_realization_derivatives() -> dict[str, object]:
    _, _, base, basis, forcing = _periodic_problem()
    realization = OrnsteinUhlenbeckRealization(
        jax.random.key(1306),
        (basis.coordinate_size,),
        support=(0.0, 1.0),
        tolerance=1.0e-7,
        noise_id="ou-fixed-realization-derivative-qualification",
    )
    method = prepare_ou_forced_periodic_method(base, forcing, realization)
    seed_velocity = _initial_velocity(basis)

    def terminal(amplitude):
        state = method.initial_state(amplitude * seed_velocity, 0.0)
        successful = jnp.asarray(True)
        for step_index in range(4):
            advance = method.step(
                jnp.asarray(step_index, dtype=jnp.int32),
                jnp.asarray(0.01 * step_index),
                state,
                jnp.asarray(0.01),
                None,
            )
            successful = successful & advance.successful
            state = advance.accepted_state
        energy = 0.5 * jnp.real(jnp.sum(jnp.conj(state.velocity) * state.velocity))
        return energy, successful

    def terminal_energy(amplitude):
        return terminal(amplitude)[0]

    primal, tangent = jax.jvp(terminal_energy, (jnp.asarray(1.0),), (jnp.asarray(0.25),))
    value, pullback = jax.vjp(terminal_energy, jnp.asarray(1.0))
    reverse = pullback(jnp.asarray(1.0))[0]
    duality_error = jnp.abs(tangent - 0.25 * reverse)
    _, path_successful = terminal(jnp.asarray(1.0))
    passed = (
        path_successful
        & jnp.isfinite(primal)
        & jnp.isfinite(tangent)
        & jnp.isfinite(value)
        & jnp.isfinite(reverse)
        & (duality_error <= 5.0e-11 * (1.0 + jnp.abs(tangent)))
    )
    return {
        "realization_id": realization.realization_id,
        "primal": float(primal),
        "jvp": float(tangent),
        "vjp": float(reverse),
        "duality_error": float(duality_error),
        "all_steps_successful": bool(path_successful),
        "passed": bool(passed),
    }


def qualification(
    *,
    sample_count: int = 2048,
    stationary_steps: int = 128,
) -> dict[str, object]:
    if sample_count < 256 or stationary_steps < 32:
        raise ValueError("OU qualification sample and stationary windows are too small.")
    coefficient = _coefficient_and_subdivision(sample_count=sample_count)
    _, _, _, basis, forcing = _periodic_problem()
    basis_spectrum = _basis_spectrum_isotropy(basis, forcing, sample_count=sample_count)
    logical = _logical_randomness(path_count=max(16, len(jax.devices()) * 4))
    temporal, budgets = _temporal_restart_and_budgets(stationary_steps=stationary_steps)
    derivatives = _fixed_realization_derivatives()
    gates = {
        "coefficient_covariance_and_subdivision": coefficient["passed"],
        "basis_spectrum_and_isotropy": basis_spectrum["passed"],
        "logical_randomness": logical["passed"],
        "fluid_temporal_refinement_and_restart": temporal["passed"],
        "stationary_energy_budget_and_block_uncertainty": budgets["passed"],
        "fixed_realization_jvp_vjp": derivatives["passed"],
    }
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
            "x64": bool(jax.config.x64_enabled),
        },
        "coefficient_covariance_and_subdivision": coefficient,
        "basis_spectrum_and_isotropy": basis_spectrum,
        "logical_randomness": logical,
        "fluid_temporal_refinement_and_restart": temporal,
        "stationary_energy_budget_and_block_uncertainty": budgets,
        "fixed_realization_jvp_vjp": derivatives,
        "gates": gates,
        "passed": all(bool(value) for value in gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/incompressible_ou_qualification.json"),
    )
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--stationary-steps", type=int, default=128)
    arguments = parser.parse_args()
    payload = qualification(
        sample_count=arguments.sample_count,
        stationary_steps=arguments.stationary_steps,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
