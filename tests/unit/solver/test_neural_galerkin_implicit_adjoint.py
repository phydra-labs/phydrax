#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.solver._neural_galerkin import _NeuralGalerkinVectorField


def _growth_problem() -> Any:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    batch = component.sample(
        phx.domain.PointSampling(
            8,
            layout=phx.domain.SampleLayout((("x",),)),
            design="uniform",
        ),
        key=jr.key(0),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
    return phx.solver.NeuralGalerkinProblem(
        {"u": domain.Parameter(jnp.asarray(1.0))},
        lambda _time, functions, _args: {"u": functions["u"]},
        (phx.solver.FieldProjectionMetric("u", realization),),
        problem_id="implicit-adjoint-growth",
    )


def test_certified_backsolve_gradient_matches_recursive_checkpoint() -> None:
    for tangent in [
        phx.solver.NeuralTangentSolvePolicy("gram", damping=1e-6),
        phx.solver.NeuralTangentSolvePolicy("rectangular", damping=1e-6),
        phx.solver.NeuralTangentSolvePolicy("rectangular", damping=0.0),
    ]:
        problem = _growth_problem()
        time = jnp.asarray(0.0)
        initial = problem.parameter_subspace.pack()

        def rate_gradient(mode: Any) -> Any:
            field = _NeuralGalerkinVectorField(
                problem,
                tangent,
                phx.solver.NeuralGalerkinAdjointPolicy(mode),
            )
            return jax.grad(lambda parameters: jnp.sum(field(time, parameters, None)))(
                initial
            )

        implicit = rate_gradient("certified_backsolve")
        unrolled = rate_gradient("recursive_checkpoint")

        assert jnp.all(jnp.isfinite(implicit))
        assert jnp.allclose(implicit, unrolled, rtol=1e-6, atol=1e-9)
