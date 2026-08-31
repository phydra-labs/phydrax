# Lagrangian and Hamiltonian mechanics

Phydrax represents mechanics with the same objects used for PDEs: labeled domains,
`DomainFunction`s, differentiable operators, sampled constraints, and scalar objectives.
The labels carry the semantics:

- `"q"`: configuration,
- `"v"`: velocity,
- `"p"`: canonical momentum,
- `"t"`: time.

The operators are continuous-time constructions. They do **not** select a numerical
time integrator and do not make an arbitrary solver symplectic.

## Labeled composition and pullbacks

A Lagrangian or Hamiltonian is a function on state space, while a trajectory is a
function of time. `pullback` composes them by label. If
$H=H(q,p,t)$, then

$$
\operatorname{pullback}(H,\{q\mapsto q(t),p\mapsto p(t)\})
=H(q(t),p(t),t).
$$

Unsubstituted dependencies such as `"t"` pass through when the target domain has the
same label.

```python
import jax.numpy as jnp
import phydrax as phx

q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
time = phx.domain.TimeInterval(0.1, 2.0)
phase_time = phx.domain.ProductDomain(q_space, p_space, time)


@phase_time.Function("q", "p", "t")
def H(q, p, t):
    return 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, p) + t


@time.Function("t")
def q(t):
    return jnp.asarray([jnp.cos(t)])


@time.Function("t")
def p(t):
    return jnp.asarray([-jnp.sin(t)])


energy_along_path = phx.operators.pullback(H, {"q": q, "p": p})
assert jnp.allclose(energy_along_path.func(0.7), 1.2)
```

## Lagrangian mechanics

For a scalar Lagrangian $L(q,v,t)$, `canonical_momentum` constructs

$$
p=\frac{\partial L}{\partial v}.
$$

`euler_lagrange` pulls the velocity derivative back to a trajectory and returns

$$
r_L(t)=\frac{d}{dt}\frac{\partial L}{\partial v}
        -\frac{\partial L}{\partial q}-Q(t).
$$

`Q` is an optional generalized force. The residual has the same trailing shape as
the trajectory value, so coupled vector systems use the same API as scalar systems.

```python
import jax.numpy as jnp
import phydrax as phx

q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
v_space = phx.domain.HyperRectangle([-2.0], [2.0], label="v")
time = phx.domain.TimeInterval(0.0, 2.0)
tangent = phx.domain.ProductDomain(q_space, v_space)


@tangent.Function("q", "v")
def L(q, v):
    return 0.5 * jnp.dot(v, v) - 0.5 * jnp.dot(q, q)


@time.Function("t")
def q(t):
    return jnp.asarray([jnp.cos(t)])


p_of_qv = phx.operators.canonical_momentum(L)
el_residual = phx.operators.euler_lagrange(q, L)
assert jnp.allclose(
    p_of_qv.func(jnp.asarray([0.2]), jnp.asarray([-0.3])),
    jnp.asarray([-0.3]),
)
assert jnp.allclose(el_residual.func(0.4), jnp.asarray([0.0]), atol=1e-10)
```

The Lagrangian must be real and scalar-valued at each point. The configuration and
velocity factors must have equal dimensions. The implementation differentiates the
scalar density with respect to **state values**; `grad(L, var="q")` would instead mean
differentiation with respect to an independent domain coordinate.

## Hamiltonian mechanics

For canonical coordinates $z=(q,p)$ and scalar $H(q,p,t)$,
`canonical_hamiltonian_vector_field` returns

$$
X_H=(H_p,-H_q).
$$

`canonical_hamiltonian_residual` evaluates the two trajectory equations as one
stacked residual:

$$
r_H(t)=\begin{bmatrix}\dot q-H_p\\\dot p+H_q\end{bmatrix}.
$$

The canonical Poisson bracket is

$$
\{F,G\}=F_q\cdot G_p-F_p\cdot G_q.
$$

`hamilton_jacobi_residual` constructs

$$
\partial_t S(x,t)+H\bigl(x,\nabla_xS(x,t),t\bigr).
$$

```python
import jax.numpy as jnp
import phydrax as phx

q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
time = phx.domain.TimeInterval(0.0, 2.0)
phase = phx.domain.ProductDomain(q_space, p_space)


@phase.Function("q", "p")
def H(q, p):
    return 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, p)


@time.Function("t")
def q(t):
    return jnp.asarray([jnp.cos(t)])


@time.Function("t")
def p(t):
    return jnp.asarray([-jnp.sin(t)])


flow = phx.operators.canonical_hamiltonian_vector_field(H)
residual = phx.operators.canonical_hamiltonian_residual(q, p, H)
self_bracket = phx.operators.canonical_poisson_bracket(H, H)
assert jnp.allclose(
    flow.func(jnp.asarray([0.2]), jnp.asarray([0.3])),
    jnp.asarray([0.3, -0.2]),
)
assert jnp.allclose(residual.func(0.7), jnp.asarray([0.0, 0.0]), atol=1e-10)
assert jnp.allclose(self_bracket.func(jnp.asarray([0.2]), jnp.asarray([0.3])), 0.0)
```

## Residual penalties versus signed objectives

These are different mathematical operations:

1. **Strong residual minimization** represents an Euler–Lagrange or Hamiltonian
   residual with `phydrax.conditions.Residual`, samples it through an explicit
   `phydrax.integration` source, and evaluates it with
   `phydrax.terms.ResidualPenalty`. The penalty reduces a nonnegative pointwise
   squared magnitude.
2. **Energy minimization** puts a signed density in
   `phydrax.terms.IntegralFunctional`. The returned integral is not squared.
3. **Stationary action** asks for a stationary point of
   $\mathcal S[q]=\int L(q,\dot q,t)\,dt$. A stationary point need not minimize the
   action, so sending raw action to a gradient-minimizing solver is not generally valid.

For example, a Deep Ritz objective has the form

$$
\mathcal J[u]=\int_\Omega\left(\frac12|\nabla u|^2-fu\right)\,dx.
$$

Use `phydrax.terms.IntegralFunctional` for this signed integral and enforce
essential boundary conditions in the ansatz. See the
[mechanics cookbook](cookbook/mechanics.md) for an end-to-end example.

## Learned Lagrangians and Hamiltonians

A learned scalar `DomainFunction` can be used directly as `lagrangian` or
`hamiltonian`; include that field in `FunctionalSolver.functions` so its parameters are
part of the trainable PyTree. The mechanics operators remain differentiable with
respect to those parameters. Identifiability still belongs to the model and data:
Hamiltonians may be ambiguous up to constants, and learned Lagrangians may have gauge
freedom.

## Discrete continuum mechanics

The [Material Point Method](guides_material_point_method.md) is a separate
updated-Lagrangian particle-grid discretization with an explicit fixed-temporal
solver. Its APIC, first-Piola, stability, and replay contracts do not follow
automatically from the continuous canonical operators on this page.

## Scope

The current API covers continuous-time canonical mechanics. Deferred topics include
Legendre-transform inversion, singular Lagrangians, noncanonical Poisson tensors,
Dirac constraints, variational integrators, and automatic Noether currents.
