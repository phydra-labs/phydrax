# Quantum operators and dynamics

Phydrax represents finite-dimensional quantum states, observables, density operators,
and Hamiltonians as complex-valued `DomainFunction`s. Quantum algebra and evolution
residuals therefore compose with the same labeled domains, differentiation backends,
constraints, and solver used for real-valued PDEs.

The current API targets closed-system dynamics and matrix operator algebra. It does not
choose a time integrator or make an arbitrary learned trajectory unitary.

## Three distinct brackets

Three operations in Phydrax are Lie brackets, but they act on different objects:

| Setting | Objects | Phydrax operator | Definition |
| --- | --- | --- | --- |
| Canonical mechanics | Scalar phase-space functions | `poisson_bracket(F, G)` | $F_q\cdot G_p-F_p\cdot G_q$ |
| Quantum mechanics | Square matrix operators | `commutator(A, B)` | $AB-BA$ |
| Differential geometry | Vector fields | `lie_bracket(X, Y)` | $D_XY-D_YX$ |

`quantum_bracket(A, B, hbar=...)` is the scaled matrix commutator
$[A,B]/(i\hbar)$. It is not an automatic quantization map from classical observables
to operators. In particular, Phydrax does not claim that replacing every Poisson
bracket by a commutator preserves arbitrary classical identities.

## Complex matrix fields

The linear-algebra helpers `conjugate`, `adjoint`, `real_part`, and `imag_part` act
pointwise on `DomainFunction`s. `adjoint` conjugates and swaps the final two value axes;
it therefore requires matrix-valued output. Quantum algebra additionally requires
square matrices of equal size.

For Pauli matrices,

$$
[\sigma_x,\sigma_y]=2i\sigma_z,
\qquad
\{\sigma_x,\sigma_y\}_+=0.
$$

```python
import jax.numpy as jnp
import phydrax as phx

sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

time = phx.domain.TimeInterval(0.0, 1.0)
Sx = time.Function()(sigma_x)
Sy = time.Function()(sigma_y)

comm = phx.operators.commutator(Sx, Sy)
anti = phx.operators.anticommutator(Sx, Sy)
scaled = phx.operators.quantum_bracket(Sx, Sy)

assert jnp.allclose(comm.func(), 2.0j * sigma_z)
assert jnp.allclose(anti.func(), jnp.zeros((2, 2)))
assert jnp.allclose(scaled.func(), 2.0 * sigma_z)
```

`hbar` must be a positive real scalar. It defaults to one, corresponding to units with
$\hbar=1$.

## States, expectations, and physical density operators

State-vector and density-operator expectations are explicit, separate operations:

$$
\begin{aligned}
\operatorname{state\_expectation}(\psi,A)&=\langle\psi|A|\psi\rangle,\\
\operatorname{density\_expectation}(\rho,A)&=\operatorname{tr}(\rho A),\\
\operatorname{observable\_variance}(\psi,A)
  &=\langle\psi|A^2|\psi\rangle-\langle\psi|A|\psi\rangle^2.
\end{aligned}
$$

The state APIs assume a normalized vector but do not normalize it silently. Use
`state_norm_residual(psi)` to construct
$\langle\psi|\psi\rangle-1$. The expectation and variance operators likewise do not
assume that $A$ is Hermitian: a non-Hermitian input may produce a complex result.

For learned density operators, `density_from_factor(T)` constructs

$$
\rho=\frac{TT^\dagger}{\operatorname{tr}(TT^\dagger)}.
$$

The factor may have rectangular value shape $(n,r)$, which permits rank-limited
density operators. The result is Hermitian, positive semidefinite, and unit trace by
construction. A zero factor is rejected; no hidden diagonal regularizer is added.
This parameterization is preferable to an eigenvalue penalty when physicality must
hold exactly.

## Quantum information measures

Quantum-information operators remain pointwise `DomainFunction` transformations:

$$
\begin{aligned}
\operatorname{purity}(\rho)&=\operatorname{tr}(\rho^2),\\
S(\rho)&=-\operatorname{tr}(\rho\log_b\rho),\\
F(\psi,\phi)&=|\langle\psi|\phi\rangle|^2,\\
F(\rho,\sigma)&=\left\|\sqrt{\rho}\sqrt{\sigma}\right\|_1^2,\\
D(\rho,\sigma)&=\tfrac12\|\rho-\sigma\|_1.
\end{aligned}
$$

`von_neumann_entropy` defaults to base two and therefore returns bits. Entropy and
the squared Uhlmann `density_fidelity` require nonempty Hermitian
positive-semidefinite inputs. They reject invalid spectra rather than silently
regularizing them. Unit trace is assumed for the usual entropy, fidelity, and
distance bounds but is not imposed implicitly; construct learned densities with
`density_from_factor` when physicality must hold throughout training.

Pure-state normalization is likewise assumed by `state_fidelity`. An infidelity such
as `1.0 - state_fidelity(state, target)` is a real scalar `DomainFunction` and can be
used directly as a residual or raw objective. A `FunctionalConstraint` squares that
residual; use an `IntegralFunctional` when the unsquared infidelity itself is the
intended signed objective.

## Composite Hilbert spaces

`tensor_product(*factors)` constructs pointwise Kronecker products. Every factor must
be vector-valued or every factor must be square-matrix-valued; mixed products are
rejected rather than assigned ambiguous semantics.

For a composite density operator, use
`partial_trace(density, subsystem_dims=(2, 2), trace_out=1)`.

The subsystem dimensions are required even when the total array dimension is known:
a four-dimensional Hilbert space could be a single four-level system or two qubits.
Untraced subsystems remain in their original order. Tracing all subsystems returns the
scalar total trace, while `trace_out=()` returns the original matrix value.

`embed_operator(A, subsystem=i, subsystem_dims=dims)` inserts identity operators on
every subsystem except `i`. This is the readable way to construct local Hamiltonian,
observable, and collapse-operator terms. See the
[Bell-state recipe](cookbook/quantum_composite.md) for reduced states and correlations.

## Closed-system evolution residuals

### Schrödinger picture

For a state vector $\psi(t)$,

$$
r_\psi=i\hbar\,\partial_t\psi-\hat H\psi.
$$

`schrodinger_residual(psi, hamiltonian)` accepts either:

- a matrix-valued Hamiltonian `DomainFunction`, evaluated as $H\psi$; or
- a callable `hamiltonian(psi) -> DomainFunction`, for differential operators such as
  $-\hbar^2\Delta/(2m)+V$.

The callable form is important for wavefunctions on spatial domains: a differential
Hamiltonian is an action on a field, not merely a matrix-valued field.

### Heisenberg picture

For an observable $A(t)$ with no separate explicit-time source term,

$$
r_A=\partial_tA-\frac{[A,H]}{i\hbar}.
$$

Construct this with `heisenberg_residual(A, H)`. An explicitly time-dependent
observable can still be represented, but the current helper treats `dt(A)` as the
total derivative of the supplied field; it does not accept a second source term.

### Density-operator picture

Closed-system von Neumann evolution is

$$
r_\rho=\partial_t\rho-\frac{[H,\rho]}{i\hbar}.
$$

Construct it with `von_neumann_residual(rho, H)`. The helpers
`hermiticity_residual(A)` and `unit_trace_residual(rho)` provide structural residuals
$A-A^\dagger$ and $\operatorname{tr}(\rho)-1$.

Hermiticity and unit trace do **not** imply positive semidefiniteness. If a learned
density operator must remain physical, enforce positivity through its
parameterization; do not substitute an eigenvalue penalty and assume exact validity.
Alternatively, construct a physical learned density directly with
`density_from_factor`.

## Open-system Lindblad dynamics

For Markovian open-system evolution with collapse operators $L_k$, the dissipator is

$$
\mathcal D(\rho)=\sum_k\left(
L_k\rho L_k^\dagger
-\frac12\{L_k^\dagger L_k,\rho\}_+
\right),
$$

and the master-equation residual is

$$
r_\rho=\partial_t\rho-\frac{[H,\rho]}{i\hbar}-\mathcal D(\rho).
$$

Use `lindblad_dissipator(rho, collapse_operators)` for $\mathcal D(\rho)$ and
`lindblad_residual(rho, H, collapse_operators)` for the complete residual. A single
collapse-operator `DomainFunction` or a sequence is accepted. The rate is part of
each operator, conventionally $L_k=\sqrt{\gamma_k}C_k$; Phydrax does not insert or
infer rates. An empty sequence is valid and reduces the residual to von Neumann
evolution.

For square operators of matching dimension, the dissipator is trace preserving. It
also maps Hermitian densities to Hermitian derivatives. These generator properties do
not make an arbitrary learned matrix field positive semidefinite; use
`density_from_factor` when density physicality must hold pointwise. See the
[open-system amplitude-damping recipe](cookbook/quantum_open_system.md).

## Quantum residuals in constraints

A complex residual belongs in a residual constraint, not in a raw signed objective.
Sampled Phydrax constraints use the Hermitian squared Frobenius norm

$$
\|r\|_F^2=\sum_i \overline{r_i}r_i,
$$

so their losses remain real and nonnegative. `IntegralFunctional`, by contrast,
represents a real signed scalar objective and rejects complex output. Select an
explicit real quantity with `real_part(...)` when that is mathematically intended.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
time = phx.domain.TimeInterval(0.0, 1.0)
omega = 1.4
H = time.Function()(0.5 * omega * sigma_z)

@time.Function("t")
def psi(t):
    return jnp.asarray([jnp.exp(-0.5j * omega * t), 0.0j])

constraint = phx.constraints.FunctionalConstraint.from_operator(
    component=time.component(),
    operator=lambda state: phx.operators.schrodinger_residual(state, H),
    constraint_vars="psi",
    num_points=16,
    structure=phx.domain.ProductStructure((("t",),)),
    reduction="mean",
)
solver = phx.solver.FunctionalSolver(
    functions={"psi": psi},
    constraints=[constraint],
)
assert solver.loss(key=jr.key(0)) < 1e-20
```

Learned matrix and state fields remain ordinary trainable `DomainFunction`s, and the
quantum operators preserve JAX tracing and parameter gradients. Include a learned
Hamiltonian, state, observable, or density factor in `FunctionalSolver.functions` to
expose its trainable leaves. Identifiability remains a responsibility of the model.

## Scope

The current quantum API intentionally excludes non-Markovian master equations,
creation–annihilation operators, Moyal/star products, and unitary or
positivity-preserving time integrators. Composite factorization remains explicit
rather than being inferred from array shapes. These deferred features require
additional algebraic, memory, or integration contracts.
