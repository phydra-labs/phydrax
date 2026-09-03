"""Solve a convex shared-resource game with one common VE multiplier."""

import jax.numpy as jnp

import phydrax as phx


partition = phx.control.games.PlayerControlPartition(("one", "two"), (1, 1))


def path_inequality(
    function,
    constraint_id,
    *,
    scope,
    participants,
    owner,
    control_dependencies,
):
    return phx.control.games.GameConstraintBlock(
        phx.control.BoundedPathConstraint(
            function,
            lower=-jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=phx.control.games.GameConstraintSite.PATH,
        equality=False,
        residual_shape=(),
        time_dependent=False,
        state_dependent=False,
        control_dependencies=control_dependencies,
    )


constraints = phx.control.games.OpenLoopGameConstraints(
    partition,
    (
        path_inequality(
            lambda time, state, control, args: -control[0],
            "one-nonnegative",
            scope=phx.control.games.GameConstraintScope.PLAYER_LOCAL,
            participants=("one",),
            owner="one",
            control_dependencies=("one",),
        ),
        path_inequality(
            lambda time, state, control, args: -control[1],
            "two-nonnegative",
            scope=phx.control.games.GameConstraintScope.PLAYER_LOCAL,
            participants=("two",),
            owner="two",
            control_dependencies=("two",),
        ),
        path_inequality(
            lambda time, state, control, args: control[0] + control[1] - 1.0,
            "shared-resource",
            scope=phx.control.games.GameConstraintScope.SHARED,
            participants=("one", "two"),
            owner=None,
            control_dependencies=("one", "two"),
        ),
    ),
)
problem = phx.control.games.FiniteHorizonLQOpenLoopVEProblem(
    jnp.zeros((1, 1, 1)),
    jnp.zeros((1, 1, 2)),
    jnp.zeros((1,)),
    jnp.zeros((2, 1, 1, 1)),
    jnp.asarray(
        (
            (((1.0, 0.0), (0.0, 0.0)),),
            (((0.0, 0.0), (0.0, 1.0)),),
        )
    ),
    jnp.zeros((2, 1, 1)),
    partition,
    constraints=constraints,
    control_linear=jnp.asarray((((-2.0, 0.0),), ((0.0, -2.0),))),
    problem_id="example-open-loop-shared-resource-ve",
)
solution = phx.control.games.solve_open_loop_ve(
    problem,
    jnp.asarray(((0.2, 0.2),)),
)
if not bool(solution.valid):
    raise RuntimeError(f"open-loop VE solve failed: status={int(solution.status)}")
if not bool(solution.convexity_certified) or not bool(solution.monotone):
    raise RuntimeError("the reported open-loop VE lacks the required convexity evidence")
if solution.shared_multipliers.shape != (1,):
    raise RuntimeError("the shared resource must carry exactly one common multiplier")

print(
    {
        "certificate": solution.certificate_label,
        "claim": solution.certification_claim,
        "claim_boundary": "convex open-loop variational equilibrium; not feedback Nash",
        "controls": solution.controls.tolist(),
        "common_shared_multiplier": solution.shared_multipliers.tolist(),
        "private_multipliers": [value.tolist() for value in solution.private_multipliers],
        "original_kkt_residual": float(solution.original_kkt_residual),
        "natural_residual": float(solution.natural_residual),
        "status": int(solution.status),
    }
)
