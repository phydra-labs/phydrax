# Persistent Markov measures

This example samples a parameterized one-dimensional Boltzmann density and lowers the
correlated draws into the ordinary Phydrax integration API. The same persistent-state
pattern is used by variational Monte Carlo, but nothing here is quantum-specific.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

proposal = phx.sampling.GaussianRandomWalkProposal(0.35)
kernel = phx.sampling.MetropolisHastings(proposal)


def normal_target(precision):
    return phx.sampling.FullMarkovTarget(
        lambda position: -0.5 * precision * jnp.sum(position**2),
        target_id=f"unit-normal-precision-{float(precision).hex()}",
    )


initial = jnp.asarray([[-1.0], [0.0], [1.0], [2.0]])
unit_target = normal_target(1.0)
state = kernel.initialize(unit_target, initial)
first = phx.sampling.sample_markov(
    unit_target,
    kernel,
    state,
    key=jr.key(0),
    num_draws=512,
    steps_per_draw=2,
    warmup_steps=128,
)

measure = phx.integration.markov_chain_measure(first)
second_moment = phx.integration.integrate(lambda value: value**2, measure)
if second_moment.error_estimate is not None or measure.independent:
    raise RuntimeError("Correlated Markov measure evidence is inconsistent")

# Keep the positions, but rebind to a changed target identity before sampling.
concentrated_target = normal_target(2.0)
rebound = kernel.rebind(concentrated_target, first.final_state)
second = phx.sampling.sample_markov(
    concentrated_target,
    kernel,
    rebound,
    key=jr.key(1),
    num_draws=512,
    steps_per_draw=2,
)
```

`warmup_steps` discards transitions but does not adapt the proposal. If parameters
change, retained positions are valid warm starts only after `rebind`; `refresh` is
reserved for auditing the same `target_id`. Draws from the old target must not be
relabeled as draws from the new one.

The equal-weight integration estimate describes the realized correlated measure. It
does not claim IID uncertainty. For a reportable final estimate, freeze the target,
run multiple sufficiently long chains, and compute chain convergence diagnostics
before applying any release threshold.
