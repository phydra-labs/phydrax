#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compare fixed USL/USF/MUSL and adaptive explicit MPM."""

import jax.numpy as jnp

import phydrax as phx


def _compile(schedule):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    volume = jnp.full((4,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "schedule-example",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(schedule=schedule),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.04, -0.01)), position.shape)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    return compiled, arguments, state


def run():
    results = {}
    for schedule in (
        phx.discretization.USLMPMSchedule(),
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
    ):
        compiled, arguments, state = _compile(schedule)
        detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
        results[schedule.common_name] = {
            "successful": bool(detail.successful),
            "mass_defect": float(detail.diagnostics.transfer.relative_mass_defect),
            "energy_defect": float(detail.diagnostics.energy.balance_defect),
        }
    compiled, arguments, state = _compile(phx.discretization.USLMPMSchedule())
    adaptive = phx.solver.AdaptiveMPMRolloutPlan(
        compiled.dynamics,
        phx.solver.MPMAdaptivePolicy(16, maximum_retries=8),
        final_time=0.02,
        initial_step_size=1.0,
    ).rollout(state, arguments)
    results["adaptive"] = {
        "completed": bool(adaptive.completed),
        "attempts": int(adaptive.journal.attempt_count),
        "accepted": int(adaptive.journal.accepted_count),
    }
    return results


if __name__ == "__main__":
    print(run())
