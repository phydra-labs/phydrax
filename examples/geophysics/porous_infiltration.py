"""Two-cell conservative unsaturated infiltration with implicit sensitivity."""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()
    exterior = np.flatnonzero(np.asarray(discretization.neighbour_cells) < 0)
    boundaries = phx.applications.porous_media.PorousBoundaryConditions(
        discretization,
        pressure_Pa={int(face): -2.0e4 for face in exterior},
    )
    plan = phx.applications.porous_media.RichardsPlan(
        discretization,
        phx.applications.porous_media.PorousMaterial(0.3, 1.0e-12),
        phx.applications.porous_media.VanGenuchtenMualem(1.0e-5, 2.0),
        boundaries,
        gravity_m_s2=(0.0, 0.0, 0.0),
        method=phx.nonlinear.NewtonKrylov(
            linear_policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
        ),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=40,
        ),
    )
    initial = plan.initialize(-2.0e4)

    def final_mass(injection):
        result = plan.step(initial, 5.0, source_kg_s=jnp.asarray((injection, 0.0)))
        return jnp.sum(result.state.water_mass_kg)

    result = plan.step(initial, 5.0, source_kg_s=jnp.asarray((1.0e-5, 0.0)))
    sensitivity = jax.grad(final_mass)(jnp.asarray(1.0e-5))
    print(
        {
            "successful": bool(result.successful),
            "water_mass_kg": np.asarray(result.state.water_mass_kg).tolist(),
            "maximum_residual": float(jnp.max(jnp.abs(result.residual))),
            "total_mass_sensitivity_s": float(sensitivity),
        }
    )


if __name__ == "__main__":
    main()
