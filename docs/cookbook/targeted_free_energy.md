# Improve free-energy overlap with an exact map

Keep map construction, generalized-work evaluation, and free-energy estimation as
separate steps.

```python
import phydrax as phx

mapping = phx.uq.TargetedMapPlan(
    exact_bijector,
    source.event_shape,
    architecture_id="endpoint-map",
)
problem = phx.uq.TargetedFreeEnergyProblem(source, target, mapping)
fit = phx.uq.fit_targeted_free_energy_map(
    problem,
    source_training_samples,
    key,
    target_samples=target_training_samples,
    validation_source=source_validation_samples,
    validation_target=target_validation_samples,
    policy=phx.uq.TargetedMapTrainingPolicy(maximum_steps=1000),
)
validated = phx.uq.TargetedFreeEnergyProblem(source, target, fit.mapping)
work = phx.uq.evaluate_targeted_work(
    validated,
    source_validation_samples,
    target_samples=target_validation_samples,
)
estimate = phx.uq.bennett_acceptance_ratio(work.forward_work, work.reverse_work)
if not bool(fit.valid & work.valid & estimate.converged):
    raise RuntimeError("Targeted free-energy calculation did not qualify")
```

Use `CenterOfMassPreservingBijector` when a finite molecular problem should leave
translation untouched. `AlchemicalEndpointReducedPotential` rejects endpoint dummy
support because an unrestrained dummy coordinate does not define a normalized density.
Approximate divergence or Jacobian estimates are outside this API.
