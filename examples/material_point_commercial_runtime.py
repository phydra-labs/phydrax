#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Claim-bound MPM execution with durable checkpoint and output."""

from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def run(directory="commercial_mpm_run"):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    claim = phx.discretization.MPMClaimTuple(
        equation_family="solid-mechanics",
        dimension=2,
        kinematics="plane-strain",
        grid_assignment="quadratic-bspline",
        source_domain="point",
        transfer="apic",
        schedule="usl-minus",
        material="neo-hookean",
        field_contact="single-field-none",
        fracture="none",
        integrator="explicit-fixed",
        storage_backend="dense-cpu-f64-deterministic",
        precision_accumulation="f64-deterministic",
        capacity_envelope="example",
        derivative_mode="branchwise",
    )
    intended = phx.discretization.MPMIntendedUse(
        "demonstrate durable commercial runtime",
        phenomena=("elasticity",),
        target_observables=("particle position",),
        risk_class="example-only",
        geometry_loading_scope="periodic unit square",
        material_parameter_scope="example parameters",
        accuracy_uq_goal="demonstrate transactional runtime",
    )
    decision = phx.discretization.MPMSupportDecision(
        claim,
        phx.discretization.MPMClaimOutcome.SUPPORTED,
        reason="example tuple",
        required_profile="example",
    )
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38], [0.36, 0.52]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    problem = phx.equations.MaterialPointProblemIR(
        "commercial-example",
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        intended_use=intended,
        claim=claim,
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        support_decision=decision,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.02, -0.01)), position.shape),
        volume,
        arguments,
    )
    checkpoint = phx.solver.MPMCheckpointPlan(compiled, state)
    output = phx.solver.MPMOutputPlan(compiled, target / "trajectory.h5")
    supervisor = phx.solver.MPMRunSupervisor(
        compiled.dynamics,
        state,
        arguments,
        checkpoint_plan=checkpoint,
        checkpoint_directory=target / "checkpoints",
        output_plan=output,
    )
    result = supervisor.advance(0.001)
    return {
        "successful": bool(result.numerical_result.successful),
        "claim_id": compiled.claim_id,
        "generation": result.generation,
        "output_complete": result.output_complete,
        "snapshot": supervisor.snapshot(),
    }


if __name__ == "__main__":
    print(run())
