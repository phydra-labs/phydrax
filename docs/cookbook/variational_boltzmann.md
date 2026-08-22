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


def log_target(precision):
    return lambda position: -0.5 * precision * jnp.sum(position**2)


initial = jnp.asarray([[-1.0], [0.0], [1.0], [2.0]])
state = kernel.initialize(log_target(1.0), initial)
first = phx.sampling.sample_markov(
    log_target(1.0),
    kernel,
    state,
    key=jr.key(0),
    num_draws=512,
    steps_per_draw=2,
    warmup_steps=128,
)

measure = phx.integration.markov_chain_measure(first)
second_moment = phx.integration.integrate(lambda value: value**2, measure)
assert second_moment.error_estimate is None
assert measure.independent is False

# Keep the positions, but refresh their target values before sampling a changed law.
refreshed = kernel.refresh(log_target(2.0), first.final_state)
second = phx.sampling.sample_markov(
    log_target(2.0),
    kernel,
    refreshed,
    key=jr.key(1),
    num_draws=512,
    steps_per_draw=2,
)
```

`warmup_steps` discards transitions but does not adapt the proposal. If parameters
change, retained positions are valid warm starts only after `refresh`; draws from the
old target must not be relabeled as draws from the new one.

The equal-weight integration estimate describes the realized correlated measure. It
does not claim IID uncertainty. For a reportable final estimate, freeze the target,
run multiple sufficiently long chains, and compute chain convergence diagnostics
before applying any release threshold.
