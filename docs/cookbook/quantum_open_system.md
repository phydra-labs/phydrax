# Open-system amplitude damping

This recipe verifies a two-level Lindblad master equation through a complex residual
condition and penalty. Let $|0\rangle$ be the ground state, $|1\rangle$ the excited
state, and

$$
L=\sqrt{\gamma}\,|0\rangle\langle1|.
$$

Starting from $\rho(0)=|1\rangle\langle1|$ with $H=0$, the exact amplitude-damping
trajectory is

$$
\rho(t)=
\begin{bmatrix}
1-e^{-\gamma t} & 0\\
0 & e^{-\gamma t}
\end{bmatrix}.
$$

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

lowering = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
excited_projector = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
time = phx.domain.TimeInterval(0.0, 2.0)
gamma = 0.8
H = time.Function()(jnp.zeros((2, 2), dtype=complex))
L = time.Function()(jnp.sqrt(gamma) * lowering)
Pe = time.Function()(excited_projector)

@time.Function("t")
def rho(t):
    excited = jnp.exp(-gamma * t)
    return jnp.asarray(
        [[1.0 - excited, 0.0], [0.0, excited]],
        dtype=complex,
    )

@time.Function("t")
def perturbed_rho(t):
    excited = jnp.exp(-0.6 * gamma * t)
    return jnp.asarray(
        [[1.0 - excited, 0.0], [0.0, excited]],
        dtype=complex,
    )

dissipator = phx.operators.lindblad_dissipator(rho, L)
residual = phx.operators.lindblad_residual(rho, H, (L,))
population = phx.operators.density_expectation(rho, Pe)

assert jnp.allclose(residual.func(0.4), 0.0, atol=1e-11)
assert jnp.allclose(jnp.trace(dissipator.func(0.4)), 0.0, atol=1e-12)
assert jnp.allclose(phx.operators.hermiticity_residual(rho).func(0.4), 0.0)
assert jnp.allclose(phx.operators.unit_trace_residual(rho).func(0.4), 0.0)
assert jnp.allclose(population.func(0.4), jnp.exp(-gamma * 0.4))
assert jnp.real(population.func(1.2)) < jnp.real(population.func(0.2))

component = time.component()
condition = phx.conditions.Residual(
    "rho",
    component,
    lambda density: phx.operators.lindblad_residual(
        density,
        H,
        (L,),
    ),
    label="lindblad",
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
    functions={"rho": rho},
    terms=[term],
)
perturbed = phx.solver.FunctionalSolver(
    functions={"rho": perturbed_rho},
    terms=[term],
)

exact_loss = exact.loss(key=jr.key(0))
perturbed_loss = perturbed.loss(key=jr.key(1))
assert jnp.isrealobj(exact_loss)
assert exact_loss < 1e-20
assert perturbed_loss > 1e-4
```

`lindblad_dissipator` does not add rates independently: the conventional
$\sqrt{\gamma_k}$ factor belongs in each collapse operator $L_k$. A single
`DomainFunction`, a sequence of collapse operators, or an empty sequence is accepted.
An empty sequence reduces `lindblad_residual` to closed-system von Neumann evolution.

Trace and Hermiticity preservation are algebraic properties of the Lindblad generator,
but a freely learned matrix field is not thereby guaranteed to be a physical density
at every point. Use `density_from_factor` when positive semidefiniteness and unit trace
must hold by construction.
