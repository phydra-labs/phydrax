"""Evaluate a frozen-law response and an independently induced-law fixed point."""

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


PARTICLE_COUNT = 16
TIME_COUNT = 5
PROBLEM_ID = "affine-induced-law-candidate"
times = jnp.linspace(0.0, 1.0, TIME_COUNT)
particle_offsets = jnp.linspace(-0.15, 0.15, PARTICLE_COUNT)
initial_particles = jnp.broadcast_to(
    particle_offsets[:, None, None],
    (PARTICLE_COUNT, TIME_COUNT, 1),
)
initial_flow = phx.stochastic.EmpiricalMeanField(
    times,
    initial_particles,
    sample_shape=(PARTICLE_COUNT,),
    state_shape=(1,),
    mean_field_id="affine-law:initial",
    source_path_id="affine-law:initial-forward-paths",
)


def law_means(flow):
    """Return the weighted mean at every represented time node."""

    return jax.vmap(lambda time: flow.snapshot(time).mean)(flow.times)


def frozen_response(flow, args):
    """Evaluate one candidate response against exactly ``flow``."""

    del args
    path_id = f"frozen-response-paths:{flow.mean_field_id}"
    paths = phx.stochastic.BSDEPathBatch(
        flow.times,
        flow.particles,
        jnp.zeros(flow.sample_shape + (flow.times.shape[0] - 1, 1)),
        sample_shape=flow.sample_shape,
        state_shape=flow.state_shape,
        noise_shape=(1,),
        path_id=path_id,
        process_id="affine-frozen-response-process",
    )
    adapter = phx.stochastic.MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, adapter_args: -z.reshape((1,)),
        lambda time, state, law, action, adapter_args: 0.5 * action**2,
        lambda time, state, law, action, adapter_args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="affine-frozen-response-adapter",
    )
    base_problem = phx.stochastic.adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, base_args: jnp.zeros((1,)),
        lambda time, state, law, base_args: jnp.ones((1, 1)),
        lambda state, law, base_args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"affine-base:{flow.mean_field_id}",
        process_id=paths.process_id,
    )
    problem = phx.control.games.FrozenLawBestResponseProblem(
        base_problem,
        adapter,
        supplied_law_id=f"supplied:{flow.mean_field_id}",
        problem_id=f"frozen:{flow.mean_field_id}",
    )
    return phx.control.games.solve_frozen_law_best_response(
        problem,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
        key=jr.key(0),
        minimum_effective_sample_size=2.0,
    )


def independently_induced_flow(response, args):
    """Construct a new forward-law sample rather than returning the frozen input."""

    del args
    current = response.mean_field
    induced_means = 0.25 + 0.5 * law_means(current)
    particles = induced_means[None, ...] + particle_offsets[:, None, None]
    return phx.stochastic.EmpiricalMeanField(
        current.times,
        particles,
        sample_shape=current.sample_shape,
        state_shape=current.state_shape,
        mean_field_id=f"induced:{response.flow_id}",
        weights=current.weights,
        source_path_id=f"independent-forward-paths:{response.flow_id}",
    )


def maximum_mean_distance(current, induced, args):
    del args
    return jnp.max(jnp.abs(law_means(current) - law_means(induced)))


problem = phx.control.games.MeanFieldGameFixedPointProblem(
    initial_flow,
    frozen_response,
    independently_induced_flow,
    maximum_mean_distance,
    best_response_id="frozen-bsde-candidate-response",
    induced_flow_id="independent-affine-forward-law",
    law_distance_id="maximum-time-node-mean-distance",
    problem_id=PROBLEM_ID,
)
plan = phx.control.games.MeanFieldGameFixedPointPlan(
    maximum_iterations=8,
    consistency_tolerance=2.0e-3,
    damping=1.0,
    minimum_effective_sample_size=2.0,
    problem_id=PROBLEM_ID,
)
result = phx.control.games.solve_mean_field_game_fixed_point(problem, plan)

if not bool(result.valid):
    raise RuntimeError(
        "fixed-point candidate evaluation failed: "
        f"status={int(result.status)}, distance={float(result.final_distance)}"
    )
if result.best_response_result is None or not bool(result.best_response_result.valid):
    raise RuntimeError("the accepted result does not retain a valid frozen-law response")
if result.induced_flow is None:
    raise RuntimeError(
        "the accepted result does not retain its independently induced law"
    )
if (
    result.mean_field_game_equilibrium_claimed
    or result.best_response_optimality_evaluated
    or result.mean_field_control_optimum_claimed
    or result.finite_population_game_claimed
    or result.common_noise_equilibrium_claimed
):
    raise RuntimeError("candidate evidence crossed its declared claim boundary")

accepted = int(result.accepted_iterations)
print(
    {
        "certificate": result.certificate_label,
        "candidate_evaluation_only": result.candidate_evaluation_only,
        "iterations": int(result.iterations),
        "law_distance_history": result.distance_history[:accepted].tolist(),
        "final_law_distance": float(result.final_distance),
        "minimum_current_ess": float(
            jnp.min(result.current_effective_sample_size_history[:accepted])
        ),
        "minimum_induced_ess": float(
            jnp.min(result.induced_effective_sample_size_history[:accepted])
        ),
        "claim_boundary": {
            "best_response_optimality_evaluated": result.best_response_optimality_evaluated,
            "mean_field_game_equilibrium_claimed": result.mean_field_game_equilibrium_claimed,
            "mean_field_control_optimum_claimed": result.mean_field_control_optimum_claimed,
            "finite_population_game_claimed": result.finite_population_game_claimed,
            "common_noise_equilibrium_claimed": result.common_noise_equilibrium_claimed,
        },
    }
)
