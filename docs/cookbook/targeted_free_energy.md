# Improve free-energy overlap with an exact map

Keep map construction, generalized-work evaluation, authenticated work data, and
free-energy estimation as separate steps. This complete identity-map example has an
exact zero free-energy difference:

```python
import jax.numpy as jnp
import phydrax as phx

source = phx.uq.CallableReducedPotential(
    lambda value: 0.5 * jnp.sum(value**2),
    (1,),
    "unit-normal-source",
)
target = phx.uq.CallableReducedPotential(
    lambda value: 0.5 * jnp.sum(value**2),
    (1,),
    "unit-normal-target",
)
mapping = phx.uq.TargetedMapPlan(
    phx.uq.IdentityBijector(),
    (1,),
    architecture_id="identity-endpoint-map",
)
problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
samples = jnp.linspace(-2.0, 2.0, 16)[:, None]
work = phx.uq.evaluate_targeted_work(
    problem,
    samples,
    target_samples=samples,
)
count = samples.shape[0]
active = jnp.ones((2 * count,), dtype="bool")
dataset = phx.uq.ReducedWorkDataset(
    jnp.concatenate((work.forward_work, work.reverse_work)),
    active,
    active,
    jnp.concatenate(
        (
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.ones((count,), dtype=jnp.int32),
        )
    ),
    jnp.concatenate(
        (
            jnp.ones((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.int32),
        )
    ),
    jnp.zeros((2 * count,), dtype=jnp.int32),
    jnp.tile(jnp.arange(count, dtype=jnp.int32), 2),
    jnp.zeros((2 * count,), dtype=jnp.int32),
    jnp.zeros((2 * count,), dtype=jnp.int32),
    state_ids=("source", "target"),
    potential_ids=(source.potential_id, target.potential_id),
    measure_ids=("real-line", "real-line"),
    producer_id=problem.problem_id,
    run_id="identity-targeted-run",
    work_id="identity-targeted-work",
    qualification_id="analytic-iid-sampling",
    sampling_exact=True,
    sampling_bias_bound=0.0,
    work_kind="targeted-map",
    mapping_id=mapping.map_id,
    unit_id="1",
)
estimate = phx.uq.bennett_acceptance_ratio(dataset)
if not bool(jnp.all(work.valid) & estimate.successful):
    raise RuntimeError("Targeted free-energy calculation did not qualify")
```

Use `CenterOfMassPreservingBijector` when a finite molecular problem should leave
translation untouched. `ControlledHamiltonianReducedPotential` admits only one
nonperiodic state on common normalized Cartesian support; controls that decouple a
region from its environment are rejected. Approximate divergence or Jacobian estimates
remain outside this API.
