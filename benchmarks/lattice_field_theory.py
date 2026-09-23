#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from _runtime import capture_environment, logical_array_bytes, measure_repeated

import phydrax as phx


def _triangle():
    return phx.discretization.polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)


def _free_scalar_case(draws: int) -> dict:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    cochain = phx.discretization.StructuredCochainBridge(grid).cochain
    action = phx.operators.path_integral.Phi4LatticeAction(
        cochain,
        mass_squared=1.5,
        quartic_coupling=0.0,
    )
    dimension = action.configuration_shape[0]
    zero = jnp.zeros(action.configuration_shape)
    precision = jax.hessian(action.action)(zero)
    exact_covariance = phx.linalg.inverse(
        precision,
        phx.linalg.FactorizationPolicy("cholesky"),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "verified",
                "positive_definite": "verified",
            },
        ),
    ).value
    kernel = phx.sampling.prepare_hamiltonian_kernel(
        lambda value: -action.action(value),
        jnp.eye(dimension),
        step_size=0.1,
        leapfrog_steps=6,
        target_id=f"{action.action_id}:free-scalar",
    )
    initial = 0.02 * jnp.reshape(
        jnp.arange(4 * dimension, dtype="float64"),
        (4, dimension),
    )
    state = phx.sampling.initialize_hamiltonian_state(kernel, initial)
    root_key = jax.random.key(5)
    adaptation_key, sampling_key = jax.random.split(root_key)
    adaptation = phx.sampling.adapt_hamiltonian_kernel(
        kernel,
        state,
        phx.sampling.HamiltonianAdaptationPlan(
            warmup_steps=32,
            minimum_step_size=0.01,
            maximum_step_size=0.3,
        ),
        key=adaptation_key,
    )
    result = phx.sampling.sample_hamiltonian(
        adaptation.kernel,
        adaptation.final_state,
        key=sampling_key,
        num_draws=max(128, draws),
    )
    samples = result.samples.reshape((-1, dimension))
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    empirical_covariance = centered.T @ centered / samples.shape[0]
    covariance_scale = jnp.maximum(jnp.max(jnp.abs(exact_covariance)), 1.0)
    return {
        "sites": dimension,
        "rng": {
            "root_seed": 5,
            "adaptation_stream": 0,
            "sampling_stream": 1,
        },
        "draws": result.samples.shape[1],
        "adaptation_valid": bool(jnp.all(adaptation.valid)),
        "acceptance_rate": [float(value) for value in jnp.mean(result.accepted, axis=1)],
        "mean_residual": float(jnp.max(jnp.abs(jnp.mean(samples, axis=0)))),
        "relative_covariance_residual": float(
            jnp.max(jnp.abs(empirical_covariance - exact_covariance)) / covariance_scale
        ),
    }


def _phi4_case(repeats: int, draws: int) -> dict:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    cochain = phx.discretization.StructuredCochainBridge(grid).cochain
    action = phx.operators.path_integral.Phi4LatticeAction(
        cochain,
        mass_squared=0.6,
        quartic_coupling=0.4,
    )
    local = phx.operators.path_integral.prepare_local_phi4_action(action)
    field = 0.2 * jnp.sin(jnp.arange(action.configuration_shape[0]))
    value, timing = measure_repeated(
        lambda: action.action(field), warmup=1, repeats=repeats
    )
    _, cache = local.initialize_incremental(field)
    proposal = field.at[7].add(0.1)
    payload = phx.sampling.SingleCoordinateProposalPayload(
        index=jnp.asarray(7, dtype=jnp.int32),
        displacement=jnp.asarray(0.1),
    )
    delta, local_timing = measure_repeated(
        lambda: local.propose_incremental(field, cache, proposal, payload)[0],
        warmup=1,
        repeats=repeats,
    )
    exact_delta = action.action(proposal) - action.action(field)
    target = phx.operators.path_integral.incremental_target_from_lattice_action(
        local,
        refresh_cadence=16,
    )
    kernel = phx.sampling.MetropolisHastings(
        phx.sampling.SingleCoordinateGaussianProposal(0.2)
    )
    state = kernel.initialize(
        target,
        jnp.zeros((4,) + action.configuration_shape),
    )
    chain = phx.sampling.sample_markov(
        target,
        kernel,
        state,
        key=jax.random.key(6),
        warmup_steps=max(32, draws),
        num_draws=draws,
    )
    magnetization_plan = phx.operators.path_integral.phi4_observable_plans(action)[1]
    magnetization = jax.vmap(jax.vmap(magnetization_plan.evaluate))(chain.samples)
    magnetization_diagnostics = phx.uq.correlated_observable_diagnostics(magnetization)
    exact_log_target = jax.vmap(jax.vmap(lambda sample: -action.action(sample)))(
        chain.samples
    )
    return {
        "sites": action.configuration_shape[0],
        "edges": cochain.cell_counts[1],
        "logical_input_bytes": logical_array_bytes(field),
        "action": float(value),
        "local_delta_residual": float(jnp.abs(delta - exact_delta)),
        "full_action_execution": timing.to_milliseconds_dict(),
        "local_delta_execution": local_timing.to_milliseconds_dict(),
        "acceptance_rate": [float(value) for value in chain.acceptance_rate],
        "maximum_log_target_residual": float(
            jnp.max(jnp.abs(chain.log_target - exact_log_target))
        ),
        "magnetization_mean": float(magnetization_diagnostics.mean),
        "magnetization_tau_int": float(
            magnetization_diagnostics.integrated_autocorrelation_time
        ),
    }


def _u1_case(draws: int) -> dict:
    action = phx.operators.path_integral.CompactU1GaugeMeasure(_triangle(), beta=0.7)
    target = phx.operators.path_integral.incremental_target_from_lattice_action(
        action,
        refresh_cadence=16,
    )
    kernel = phx.sampling.MetropolisHastings(
        phx.sampling.SingleCoordinatePeriodicProposal(2.0 * jnp.pi, 0.8)
    )
    state = kernel.initialize(target, jnp.zeros((4, action.num_edges)))
    result = phx.sampling.sample_markov(
        target,
        kernel,
        state,
        key=jax.random.key(10),
        warmup_steps=max(32, draws),
        num_draws=draws,
    )
    plaquette = jax.vmap(jax.vmap(action.plaquette_angles))(result.samples)[..., 0]
    observable = jnp.cos(plaquette)
    diagnostics = phx.uq.correlated_observable_diagnostics(observable)
    exact = jsp.special.i1e(action.beta) / jsp.special.i0e(action.beta)
    error = jnp.abs(diagnostics.mean - exact)
    uncertainty = 4.0 * diagnostics.standard_error
    return {
        "chains": result.num_chains,
        "draws": result.num_draws,
        "acceptance_rate": [float(value) for value in result.acceptance_rate],
        "plaquette_mean": float(diagnostics.mean),
        "plaquette_reference": float(exact),
        "plaquette_absolute_error": float(error),
        "monte_carlo_standard_error": float(diagnostics.standard_error),
        "within_four_standard_errors": bool(error <= uncertainty),
        "integrated_autocorrelation_time": float(
            diagnostics.integrated_autocorrelation_time
        ),
    }


def _sun_case(dimension: int, draws: int, repeats: int) -> dict:
    topology = _triangle()
    boundaries = phx.discretization.prepare_cell_boundary_paths(topology)
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(dimension),
    )
    action = phx.operators.path_integral.WilsonGaugeAction(
        space,
        boundaries,
        plaquette_couplings=0.8,
    )
    target = phx.operators.path_integral.compact_geometric_target_from_lattice_action(
        action
    )
    kernel = phx.sampling.prepare_compact_group_hamiltonian_kernel(
        target,
        step_size=0.08 if dimension == 2 else 0.05,
        leapfrog_steps=4,
    )
    identity = space.identity()
    second = action.geometry.retract(
        identity,
        jnp.full(action.local_coordinate_shape, 0.02),
    )
    state = phx.sampling.initialize_compact_group_hamiltonian_state(
        kernel,
        jnp.stack((identity, second)),
    )
    result, timing = measure_repeated(
        lambda: phx.sampling.sample_compact_group_hamiltonian(
            kernel,
            state,
            key=jax.random.key(10 + dimension),
            num_draws=draws,
        ),
        warmup=1,
        repeats=repeats,
    )
    samples = result.samples.reshape((-1,) + action.configuration_shape)
    membership = jax.vmap(action.geometry.contains)(samples)
    return {
        "chains": result.num_chains,
        "draws": result.num_draws,
        "group": space.group.group_id,
        "algebra_dimension": kernel.coordinate_metric.dimension,
        "metric_bytes": logical_array_bytes(kernel.coordinate_metric.gram),
        "acceptance_rate": [float(value) for value in result.acceptance_rate],
        "maximum_absolute_energy_error": float(jnp.max(jnp.abs(result.energy_error))),
        "all_samples_in_group": bool(jnp.all(membership)),
        "membership_failure_count": int(jnp.sum(result.membership_failure)),
        "trajectory_execution": timing.to_milliseconds_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.draws < 8 or arguments.repeats < 1:
        raise ValueError("draws must be at least eight and repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "free_scalar": _free_scalar_case(arguments.draws),
        "phi4": _phi4_case(arguments.repeats, arguments.draws),
        "compact_u1": _u1_case(arguments.draws),
        "su2_wilson_hmc": _sun_case(2, arguments.draws, arguments.repeats),
        "su3_wilson_hmc": _sun_case(3, arguments.draws, arguments.repeats),
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
