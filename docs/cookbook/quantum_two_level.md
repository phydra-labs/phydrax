# Two-level closed-system quantum dynamics

This recipe checks a complex Schrödinger trajectory through the same sampled
`ResidualPenalty` and `FunctionalSolver` path used for PDE residuals.

Take

$$
H=\frac{\omega}{2}\sigma_z,
\qquad
\psi(0)=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}.
$$

With $\hbar=1$, the exact state is

$$
\psi(t)=\frac{1}{\sqrt2}
\begin{bmatrix}e^{-i\omega t/2}\\e^{i\omega t/2}\end{bmatrix},
$$

and satisfies $i\partial_t\psi-H\psi=0$.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
time = phx.domain.TimeInterval(0.0, 1.0)
omega = 1.7
H = time.Function()(0.5 * omega * sigma_z)

@time.Function("t")
def psi(t):
    return jnp.asarray(
        [jnp.exp(-0.5j * omega * t), jnp.exp(0.5j * omega * t)]
    ) / jnp.sqrt(2.0)

@time.Function("t")
def perturbed_psi(t):
    return jnp.asarray(
        [jnp.exp(-0.3j * omega * t), jnp.exp(0.3j * omega * t)]
    ) / jnp.sqrt(2.0)

component = time.component()
condition = phx.conditions.Residual(
    "psi",
    component,
    lambda state: phx.operators.schrodinger_residual(state, H),
    label="schrodinger",
)
source = phx.integration.per_step(
    phx.integration.mean_over(condition.on),
    phx.domain.PointSampling(
        32,
        layout=phx.domain.SampleLayout((("t",),)),
    ),
)
term = phx.terms.ResidualPenalty(condition, source, scale=1.0)

exact = phx.solver.FunctionalSolver(
    functions={"psi": psi},
    terms=[term],
)
perturbed = phx.solver.FunctionalSolver(
    functions={"psi": perturbed_psi},
    terms=[term],
)

exact_loss = exact.loss(key=jr.key(0))
perturbed_loss = perturbed.loss(key=jr.key(1))
assert jnp.isrealobj(exact_loss)
assert exact_loss < 1e-20
assert perturbed_loss > 1e-4

@time.Function("t")
def density_factor(t):
    return psi.func(t)[:, None]

density = phx.operators.density_from_factor(density_factor)
Sx = time.Function()(sigma_x)
state_expectation = phx.operators.state_expectation(psi, Sx)
density_expectation = phx.operators.density_expectation(density, Sx)

assert jnp.allclose(phx.operators.state_norm_residual(psi).func(0.4), 0.0)
assert jnp.allclose(state_expectation.func(0.4), jnp.cos(omega * 0.4))
assert jnp.allclose(density_expectation.func(0.4), state_expectation.func(0.4))
assert jnp.allclose(
    phx.operators.observable_variance(psi, H).func(0.4),
    (0.5 * omega) ** 2,
)
assert jnp.allclose(phx.operators.hermiticity_residual(density).func(0.4), 0.0)
assert jnp.allclose(phx.operators.unit_trace_residual(density).func(0.4), 0.0)
assert jnp.allclose(phx.operators.von_neumann_residual(density, H).func(0.4), 0.0)
```

The residual penalty uses $\sum_i\overline{r_i}r_i$, not $\sum_i r_i^2$; the result is
therefore real and nonnegative for complex wavefunctions. The zero loss only checks the
differential equation. In a learned solve, initial or boundary data must also select the
intended solution, and normalization or positivity should be enforced by suitable
observation penalties or parameterization.

For a spatial wavefunction, replace the matrix `H` with a callable Hamiltonian action:

```python
import phydrax as phx

hbar = 1.0
mass = 1.0


def free_particle_hamiltonian(state):
    return -(hbar**2 / (2.0 * mass)) * phx.operators.laplacian(state, var="x")
```

Pass `free_particle_hamiltonian` as the second argument to
`schrodinger_residual`. The callable must return a `DomainFunction` with the same value
shape as $\partial_t\psi$.
