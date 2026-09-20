import jax
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
samples = jax.random.normal(jax.random.key(17), (1024, 1))
evaluation = phx.uq.evaluate_targeted_work(problem, samples)
count = samples.shape[0]
active = jnp.ones((count,), dtype="bool")
work = phx.uq.ReducedWorkDataset(
    evaluation.forward_work,
    active,
    active,
    jnp.zeros((count,), dtype=jnp.int32),
    jnp.ones((count,), dtype=jnp.int32),
    jnp.zeros((count,), dtype=jnp.int32),
    jnp.arange(count, dtype=jnp.int32),
    jnp.zeros((count,), dtype=jnp.int32),
    jnp.zeros((count,), dtype=jnp.int32),
    state_ids=("source", "target"),
    potential_ids=(source.potential_id, target.potential_id),
    measure_ids=("real-line", "real-line"),
    producer_id=problem.problem_id,
    run_id="targeted-free-energy-example",
    work_id="affine-targeted-work",
    work_kind="targeted-map",
    qualification_id="iid-normal-source-sampling",
    sampling_exact=True,
    sampling_bias_bound=0.0,
    mapping_id=mapping.map_id,
    unit_id="1",
)
estimate = phx.uq.free_energy_perturbation(work)
if not bool(jnp.all(evaluation.valid) & estimate.successful):
    raise RuntimeError("Targeted free-energy example did not qualify.")
print(float(estimate.free_energies[-1]), float(jnp.std(evaluation.forward_work)))
