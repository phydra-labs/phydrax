# Discrete variational Monte Carlo

This recipe minimizes the energy of a two-spin transverse-field Ising model,

`H = −Z₀Z₁ − 0.5 (X₀ + X₁)`,

using a user-defined amplitude table, persistent single-spin-flip chains, and damped
stochastic reconfiguration. The Hamiltonian is never materialized by the solver.

```python
import equinox as eqx
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx


class TableAmplitude(eqx.Module):
    log_values: jax.Array

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        return phx.operators.LogAmplitude(
            self.log_values[index],
            1.0 + 0.0j,
        )


def diagonal(configurations):
    return -configurations[..., 0] * configurations[..., 1]


def connections(configurations):
    first = configurations.at[..., 0].multiply(-1)
    second = configurations.at[..., 1].multiply(-1)
    connected = jnp.stack((first, second), axis=-2)
    shape = configurations.shape[:-1] + (2,)
    return phx.operators.ConnectedConfigurations(
        connected,
        -0.5 * jnp.ones(shape),
        jnp.ones(shape, dtype=bool),
        configuration_shape=(2,),
    )


operator = phx.operators.CallableDiscreteQuantumOperator(
    diagonal,
    connections,
    configuration_shape=(2,),
    operator_id="two-spin-ising",
)


def flip(key, configuration):
    site = jr.randint(key, (), 0, configuration.shape[0])
    return configuration.at[site].multiply(-1)


def flip_log_prob(_proposed, configuration):
    return -jnp.log(float(configuration.shape[0]))


proposal = phx.sampling.CallableProposal(
    flip,
    flip_log_prob,
    proposal_id="single-spin-flip",
)
kernel = phx.sampling.MetropolisHastings(proposal)
initial = jnp.asarray(
    [[1, 1], [1, -1], [-1, 1], [-1, -1]],
    dtype=jnp.int32,
)
problem = phx.solver.VariationalMonteCarloProblem(
    TableAmplitude(jnp.asarray([0.2, -0.1, 0.1, -0.2])),
    operator,
    kernel,
    initial,
    complex_parameter_mode="real",
)
policy = phx.solver.VariationalMonteCarloPolicy(
    num_iterations=20,
    draws_per_iteration=128,
    steps_per_draw=2,
    warmup_steps=64,
    final_evaluation_draws=512,
    learning_rate=0.03,
    damping=0.1,
    max_update_norm=5.0,
)
result = phx.solver.solve_variational_monte_carlo(
    problem,
    policy,
    key=jr.key(0),
)

print(result.energy_history)
print(result.final_estimate.physical_energy)
print(result.final_estimate.variance)
print(result.final_estimate.acceptance_rate)
print(result.final_estimate.chain_diagnostics.max_rhat)

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "vmc-state.zip"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint, problem, policy, result.final_state
    )
    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint, problem, policy
    )
    assert restored.iteration == result.final_state.iteration
```

`ConnectedConfigurations.matrix_elements` stores `H[current, connected]`. Padded
connection slots must have `valid=False`; their configuration and matrix-element
payloads are ignored. Fermionic signs, when applicable in a downstream operator, are
part of the matrix elements rather than a separate solver convention.

The amplitude representation separates log magnitude and unit phase. A proposed zero
amplitude has a valid zero ratio, while a zero current amplitude cannot be used as a
ratio denominator. No epsilon is inserted at nodes.

The final estimate is generated after the last update with a frozen model and a
separate chain run. It remains a correlated-chain estimate, not an IID estimate. For a
publication/release result, increase chain count and length and apply the UQ convergence
diagnostics to the frozen-target draws.
