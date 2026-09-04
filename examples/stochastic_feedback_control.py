"""Evaluate one state-feedback policy on disjoint prepared noise bundles."""

import jax.numpy as jnp

import phydrax as phx


def transition(context, state, action, noise, args):
    """Advance after the action is chosen, exposing noise only to the dynamics."""
    return state + context.duration * (args["decay"] * state + action) + noise


def stage_cost(context, state, action, args):
    del args
    return context.duration * (state @ state + 0.25 * (action @ action))


def terminal_cost(time, state, args):
    del time, args
    return state @ state


def full_state_policy(context, state, args):
    """Use only the current full state and time context--never noise or a key."""
    del context
    return args["feedback_gain"] * state


def prepared_noise(prefix, coupling_id, increments, labels):
    path_count = int(increments.shape[0])
    return phx.control.stochastic.PreparedControlledNoise(
        increments,
        valid=jnp.ones((path_count,), dtype=bool),
        realization_ids=tuple(f"{prefix}:{index}" for index in range(path_count)),
        coupling_id=coupling_id,
        independence_labels=labels,
        noise_shape=(1,),
    )


time_grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 1.0, 5),
    time_id="example-stochastic-feedback",
)
problem = phx.control.stochastic.ControlledTransitionProblem(
    transition,
    time_grid,
    jnp.asarray([0.75]),
    state_shape=(1,),
    action_shape=(1,),
    noise_shape=(1,),
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    args={"decay": -0.2, "feedback_gain": -0.6},
    problem_id="example-stochastic-feedback",
)
training_noise = prepared_noise(
    "training",
    "example-stochastic-feedback:training",
    jnp.asarray(
        [
            [[0.03], [-0.01], [0.02], [-0.02]],
            [[-0.02], [0.01], [-0.03], [0.02]],
            [[0.01], [0.02], [-0.01], [-0.02]],
            [[-0.01], [-0.02], [0.01], [0.02]],
        ]
    ),
    jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
)
holdout_noise = prepared_noise(
    "holdout",
    "example-stochastic-feedback:holdout",
    jnp.asarray(
        [
            [[0.02], [0.00], [-0.01], [0.01]],
            [[-0.01], [0.01], [0.00], [-0.02]],
            [[0.00], [-0.02], [0.02], [0.01]],
            [[0.01], [0.00], [0.01], [-0.01]],
            [[-0.02], [0.02], [-0.01], [0.00]],
            [[0.00], [0.01], [-0.02], [0.02]],
        ]
    ),
    jnp.asarray([0, 1, 2, 3, 4, 5], dtype=jnp.int32),
)
if not set(training_noise.realization_ids).isdisjoint(holdout_noise.realization_ids):
    raise RuntimeError("training and holdout realization IDs must be disjoint")
if training_noise.coupling_id == holdout_noise.coupling_id:
    raise RuntimeError("training and holdout bundles must identify independent coupling")

training = phx.control.stochastic.evaluate_feedback_policy(
    problem,
    full_state_policy,
    training_noise,
    policy_id="example-full-state-linear-feedback",
    method="asymptotic-normal",
    sample_role="training",
)
holdout = phx.control.stochastic.evaluate_feedback_policy(
    problem,
    full_state_policy,
    holdout_noise,
    policy_id="example-full-state-linear-feedback",
    method="asymptotic-normal",
    sample_role="holdout",
)
if not bool(training.valid) or not bool(holdout.valid):
    raise RuntimeError(
        "prepared-noise policy evaluation failed: "
        f"training_status={int(training.status)}, holdout_status={int(holdout.status)}"
    )
if training.evidence.has_coverage_claim:
    raise RuntimeError("training reuse must not produce a coverage claim")
if not holdout.evidence.has_coverage_claim:
    raise RuntimeError("the independent holdout must retain its requested evidence")

print(
    {
        "policy_information": "full state and step context only; no noise and no random key",
        "training_empirical_cost": float(training.empirical_risk),
        "training_coverage": training.evidence.coverage,
        "holdout_empirical_cost": float(holdout.empirical_risk),
        "holdout_interval": [
            float(holdout.evidence.lower),
            float(holdout.evidence.upper),
        ],
        "holdout_confidence": holdout.evidence.confidence,
        "holdout_independent_clusters": int(holdout.evidence.independent_cluster_count),
        "coverage_assumptions": list(holdout.evidence.coverage_assumptions),
        "claim_scope": (
            "fixed-policy asymptotic expectation evidence on the independent "
            "holdout clusters; no policy-optimality claim"
        ),
    }
)
