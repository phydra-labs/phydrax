"""Run the source-complete GASAM material and its exact mixed rest solve."""

from __future__ import annotations

import jax.numpy as jnp

from phydrax.applications.skeletal_muscle import continuum
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def main():
    fibers = continuum.UniformFiberArchitecturePlan("demo-x-fibers").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    plan = continuum.EngelhardtGasam2025Plan("demo-muscle")
    parameters = continuum.EngelhardtGasam2025Parameters.published_multiload_fit()
    passive = plan.prepare(parameters, fibers, 0.0)
    active_commit = passive.propose_activation(0.75).commit()
    active = passive.with_commit(active_commit)

    stretch = 1.05
    deformation = jnp.diag(jnp.asarray((stretch, stretch**-0.5, stretch**-0.5)))
    response = active.evaluate(deformation, 0.0)
    fiber_nominal_stress = (
        fibers.reference_direction
        @ response.first_piola
        @ fibers.reference_direction
    )
    print(f"fiber stretch: {float(response.evidence.fiber_stretch):.4f}")
    print(
        "activation weight omega_a: "
        f"{float(response.evidence.activation_weight):.6f}"
    )
    print(
        f"fiber nominal stress: {float(fiber_nominal_stress) / 1000.0:.3f} kPa"
    )

    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    qualified = passive.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero"))
    )
    manufactured = continuum.solve_manufactured_rest(qualified).commit()
    print(f"mixed pair: {qualified.qualification.pair_names[0]}")
    print(f"manufactured rest committed: {manufactured.committed}")
    print(
        "final residual norm: "
        f"{float(manufactured.evidence.final_residual_norm):.3e}"
    )


if __name__ == "__main__":
    main()
