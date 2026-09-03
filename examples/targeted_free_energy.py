import jax.numpy as jnp
from flowjax.bijections import Affine

import phydrax as phx


source = phx.uq.CallableReducedPotential(
    lambda value: 0.5 * value[0] ** 2, (1,), "unit-normal"
)
target = phx.uq.CallableReducedPotential(
    lambda value: 0.5 * ((value[0] - 2.0) / 0.5) ** 2,
    (1,),
    "shifted-normal",
)
bijection = phx.uq.FlowJAXBijectionAdapter(
    Affine(loc=jnp.asarray([2.0]), scale=jnp.asarray([0.5])),
    architecture_id="affine-targeted-map",
)
mapping = phx.uq.TargetedMapPlan(bijection, (1,), architecture_id="affine-targeted-map")
problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
samples = jnp.linspace(-2.0, 2.0, 64)[:, None]
evaluation = phx.uq.evaluate_targeted_work(problem, samples)
estimate = phx.uq.free_energy_perturbation(evaluation.forward_work)
if not bool(evaluation.valid & estimate.converged):
    raise RuntimeError("Targeted free-energy example did not qualify.")
print(float(estimate.free_energies[-1]), float(jnp.std(evaluation.forward_work)))
