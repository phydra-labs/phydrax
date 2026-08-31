#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

import phydrax as phx


class FourierMode(eqx.Module):
    coefficients: Array

    def __call__(self, x: Array, /) -> Array:
        phase = 2.0 * jnp.pi * x[0]
        return self.coefficients[0] * jnp.sin(phase) + self.coefficients[1] * jnp.cos(
            phase
        )


def main() -> None:
    speed = 0.5
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Function("x")(FourierMode(jnp.asarray([1.0, 0.0])))
    component = domain.component()
    batch = component.sample(
        phx.domain.PointSampling(
            128,
            layout=phx.domain.SampleLayout((("x",),)),
            design="sobol_scrambled",
        ),
        key=jr.key(0),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )

    def rate(_time, functions, _args):
        return {"u": -speed * phx.operators.partial_n(functions["u"], var="x", order=1)}

    problem = phx.solver.NeuralGalerkinProblem(
        {"u": field},
        rate,
        (phx.solver.FieldProjectionMetric("u", realization),),
        problem_id="periodic-fourier-advection",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, 11),
        time_id="periodic-advection-time",
    )
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        tangent=phx.solver.NeuralTangentSolvePolicy(damping=1e-10),
        rtol=1e-7,
        atol=1e-9,
        dense=True,
    )
    final = result.field_at(grid.num_times - 1, "u")(batch)
    points = jnp.asarray(batch.points["x"].data)
    exact = jnp.sin(2.0 * jnp.pi * (points[..., 0] - speed * grid.t1))
    error_norm = jnp.sqrt(jnp.sum(jnp.abs(final.data - exact) ** 2))
    exact_norm = jnp.sqrt(jnp.sum(jnp.abs(exact) ** 2))
    relative_error = error_norm / exact_norm
    print(
        {
            "successful": bool(result.successful),
            "relative_l2_error": float(relative_error),
            "accepted_steps": int(result.parameter_solution.stats["num_accepted_steps"]),
            "max_projection_defect": float(
                jnp.max(result.audit.relative_projection_defect)
            ),
        }
    )


if __name__ == "__main__":
    main()
