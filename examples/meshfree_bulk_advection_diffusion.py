# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Semidiscrete meshfree advection-diffusion with native temporal methods and motion epochs.

One periodic GMLS point cloud carries the exact advection-diffusion solution
``c = 1 + 0.5 exp(-8 pi^2 k t) sin(2 pi (x - u_x t)) cos(2 pi (y - u_y t))``.
The same prepared semidiscrete problem is integrated by a fixed SSP Runge-Kutta
method and by an additive IMEX method with native implicit stage solves. An ALE
motion within the fixed-support trust is accepted (stage-correct refresh, exact
free-stream/GCL ledger); a motion beyond it is refused with the state held, and
the support is re-anchored as a new epoch before continuing.
"""

from __future__ import annotations

import argparse
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import PointCloudPlan
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeDiffusionLaw,
    MeshfreeEvolutionPlan,
    MeshfreeEvolutionStatus,
    MeshfreeMotion,
    PreparedMeshfreeEvolution,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.solver import FixedStepProblem, solve_fixed_step


# The unit torus: the closed square with both coordinate seams identified;
# the meshfree address is derived from the canonical identifications.
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)


class BulkAdvectionDiffusionMetrics(TypedDict):
    point_count: int
    steps: int
    ssprk33_error: float
    ssprk33_mass_drift: float
    imex_error: float
    imex_mass_drift: float
    imex_stage_iterations: int
    motion_accepted: bool
    motion_concentration_error: float
    motion_content_drift: float
    motion_points_crossing_seam: int
    refused_status: str
    refused_state_held: bool
    rebased_step_accepted: bool
    support_trust: float
    oracle_provenance: str


_VELOCITY = np.asarray([1.0, 0.5])
_DIFFUSIVITY = 0.02


def _exact(time: float, points: Array) -> Array:
    shifted = points - time * jnp.asarray(_VELOCITY)
    decay = jnp.exp(-8 * jnp.pi**2 * _DIFFUSIVITY * time)
    return 1.0 + 0.5 * decay * jnp.sin(2 * jnp.pi * shifted[:, 0]) * jnp.cos(
        2 * jnp.pi * shifted[:, 1]
    )


def _velocity(time: Array, points: Array, args: object) -> Array:
    del time, args
    return jnp.broadcast_to(jnp.asarray(_VELOCITY), points.shape)


def _cloud(size: int, seed: int) -> PointCloudPlan:
    """Jittered periodic lattice; 21 neighbors close a lattice distance shell.

    Selecting complete shells keeps a positive neighbor-selection gap, so the
    fixed-support refresh trusts motion of a fraction of the spacing. The last
    column sits just inside the periodic seam so admitted motion crosses it.
    """
    spacing = 1.0 / size
    axis = (np.arange(size) + 0.98) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape)
    points = np.mod(points + 0.005 * spacing * jitter, 1.0)
    return PointCloudPlan(
        points,
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=21,
        address=_PERIODIC,
    )


def _rollout(
    evolution: PreparedMeshfreeEvolution,
    method_name: str,
    initial: Array,
    final_time: float,
    steps: int,
) -> tuple[Array, bool, int]:
    method = (
        evolution.ssprk_method("ssprk33")
        if method_name == "ssprk33"
        else evolution.imex_method("ars-222")
    )
    solution = solve_fixed_step(
        FixedStepProblem(
            method, initial, t0=0.0, t1=final_time, step_size=final_time / steps
        ),
        save_every=steps,
        evidence_retention="steps",
    )
    evidence = solution.evidence
    # Only the IMEX stages solve; SSP evidence is the explicit step's admission.
    iterations = (
        0
        if method_name == "ssprk33" or evidence is None or evidence.steps is None
        else int(np.sum(np.asarray(evidence.steps["stage_iterations"])))
    )
    return solution.states[-1], bool(solution.successful), iterations


def run_workflow(
    *, size: int = 12, steps: int = 10, seed: int = 0
) -> BulkAdvectionDiffusionMetrics:
    cloud = _cloud(size, seed).prepare()
    points = cloud.points
    weights = cloud.quadrature_weights
    final_time = 0.1
    eulerian = MeshfreeEvolutionPlan(
        cloud,
        velocity=_velocity,
        diffusion=MeshfreeDiffusionLaw(_DIFFUSIVITY, law_id="isotropic-k"),
        plan_id="bulk-advection-diffusion",
    ).prepare()
    initial = eulerian.initial_state(_exact(0.0, points))
    mass = jnp.sum(weights * initial)
    exact = _exact(final_time, points)
    explicit, explicit_ok, _ = _rollout(eulerian, "ssprk33", initial, final_time, steps)
    implicit, implicit_ok, iterations = _rollout(
        eulerian, "ars-222", initial, final_time, steps
    )
    if not (explicit_ok and implicit_ok):
        raise RuntimeError("An admitted fixed-support rollout was refused.")

    trust = float(jnp.min(cloud.trust_radius))
    drift = np.asarray([0.5 * trust, 0.0])

    def mesh(time: Array, coordinates: Array, args: object) -> Array:
        del time, args
        return jnp.broadcast_to(jnp.asarray(drift), coordinates.shape)

    moving = MeshfreeEvolutionPlan(
        cloud,
        velocity=_velocity,
        diffusion=MeshfreeDiffusionLaw(_DIFFUSIVITY, law_id="isotropic-k"),
        motion=MeshfreeMotion("ale", mesh_velocity=mesh, law_id="seam-translation"),
        plan_id="bulk-advection-diffusion-ale",
    ).prepare()
    moving_initial = moving.initial_state(_exact(0.0, points))
    motion = solve_fixed_step(
        FixedStepProblem(
            moving.ssprk_method("ssprk33"),
            moving_initial,
            t0=0.0,
            t1=1.0,
            step_size=1.0 / steps,
        ),
        save_every=steps,
    )
    moved = moving.fields(motion.states[-1])
    moved_exact = _exact(1.0, moved.points)

    far = np.asarray([4.0 * trust, 0.0])

    def far_mesh(time: Array, coordinates: Array, args: object) -> Array:
        del time, args
        return jnp.broadcast_to(jnp.asarray(far), coordinates.shape)

    refused_plan = MeshfreeEvolutionPlan(
        cloud,
        velocity=_velocity,
        diffusion=MeshfreeDiffusionLaw(_DIFFUSIVITY, law_id="isotropic-k"),
        motion=MeshfreeMotion("ale", mesh_velocity=far_mesh, law_id="beyond-trust"),
        plan_id="bulk-advection-diffusion-refused",
    ).prepare()
    state = refused_plan.initial_state(_exact(0.0, points))
    attempt = refused_plan.ssprk_method("ssprk33").step(
        jnp.asarray(0), jnp.asarray(0.0), state, jnp.asarray(1.0), None
    )
    admission = refused_plan.admission(attempt.candidate_state)
    held = bool(jnp.all(attempt.accepted_state == state))
    # Support epoch: re-anchor identical nodes, then take a step inside the new trust.
    rebased = refused_plan.rebase(attempt.accepted_state)
    continued = rebased.ssprk_method("ssprk33").step(
        jnp.asarray(1),
        jnp.asarray(0.0),
        refused_plan.repacked(attempt.accepted_state),
        jnp.asarray(0.1 * float(rebased.capacity.support_trust) / far[0]),
        None,
    )
    return {
        "point_count": points.shape[0],
        "steps": steps,
        "ssprk33_error": float(jnp.max(jnp.abs(explicit - exact))),
        "ssprk33_mass_drift": float(jnp.abs(jnp.sum(weights * explicit) - mass) / mass),
        "imex_error": float(jnp.max(jnp.abs(implicit - exact))),
        "imex_mass_drift": float(jnp.abs(jnp.sum(weights * implicit) - mass) / mass),
        "imex_stage_iterations": iterations,
        "motion_accepted": bool(motion.successful),
        "motion_concentration_error": float(
            jnp.max(jnp.abs(moved.concentration - moved_exact))
        ),
        "motion_content_drift": float(
            jnp.abs(
                moving.total_content(motion.states[-1])
                - moving.total_content(moving_initial)
            )
            / moving.total_content(moving_initial)
        ),
        "motion_points_crossing_seam": int(jnp.sum(moved.points[:, 0] >= 1.0)),
        "refused_status": MeshfreeEvolutionStatus(int(admission.status)).name,
        "refused_state_held": held and not bool(attempt.successful),
        "rebased_step_accepted": bool(continued.successful),
        "support_trust": trust,
        "oracle_provenance": "exact periodic advection-diffusion mode, independent of the discretization",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, value in run_workflow(
        size=args.size, steps=args.steps, seed=args.seed
    ).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
