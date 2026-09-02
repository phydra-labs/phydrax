# Quantum operators and dynamics

Phydrax represents finite-dimensional quantum states, observables, density operators,
and Hamiltonians as complex-valued `DomainFunction`s. Quantum algebra and evolution
residuals therefore compose with the same labeled domains, differentiation backends,
constraints, and solver used for real-valued PDEs.

The current API targets closed-system dynamics and matrix operator algebra. Dense
Hermitian Hamiltonians may be integrated through a structure-preserving U(n)
propagator; learned residual trajectories remain ordinary functions unless they use
that explicit propagation contract.

Finite-register programs additionally support exact dense and bounded
tensor-network execution, local observables, angle-bound Pauli templates,
parameter-shift gradients, quantum feature models, and exact fidelity kernels.
These APIs retain a separate local-map lifecycle; see
[Local quantum programs](guides_quantum_programs.md).

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
as `1.0 - state_fidelity(state, target)` is a real scalar `DomainFunction` and can
define either a `Residual` condition or a raw objective. A `ResidualPenalty` evaluates
the condition through an explicit integration source and squares its magnitude; use
an `IntegralFunctional` when the unsquared infidelity itself is the intended signed
objective.

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

## Quantum residual penalties

A complex residual belongs in a `Residual` condition evaluated by a
`ResidualPenalty`, not in a raw signed objective. Sampled residual penalties use the
Hermitian squared Frobenius norm

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


component = time.component()
condition = phx.conditions.Residual(
    "psi",
    component,
    lambda state: phx.operators.schrodinger_residual(state, H),
)
source = phx.integration.per_step(
    phx.integration.mean_over(condition.on),
    phx.domain.PointSampling(
        16,
        layout=phx.domain.SampleLayout((("t",),)),
    ),
)
term = phx.terms.ResidualPenalty(condition, source, scale=1.0)
solver = phx.solver.FunctionalSolver(
    functions={"psi": psi},
    terms=[term],
)
assert solver.loss(key=jr.key(0)) < 1e-20
```

Learned matrix and state fields remain ordinary trainable `DomainFunction`s, and the
quantum operators preserve JAX tracing and parameter gradients. Include a learned
Hamiltonian, state, observable, or density factor in `FunctionalSolver.functions` to
expose its trainable leaves. Identifiability remains a responsibility of the model.

## Dense unitary propagation

`UnitaryPropagatorProblem` binds a finite dense Hermitian Hamiltonian to a
right-trivialized U(n) state geometry. `solve_unitary_propagator` uses the
commutator-free geometric solver and returns the complete propagator trajectory
with unitarity and Hamiltonian-Hermiticity evidence.

U(n) is the default. Selecting SU(n) explicitly removes and archives the trace
generator rather than silently discarding global phase. State evolution applies
`U psi`; density evolution applies `U rho U†`, preserving trace and spectrum up
to numerical error.

This path is separate from matrix-free VMC/TDVP and from residual-only
Schrödinger conditions.

## Local-operator variational Monte Carlo

Finite-configuration and continuum-electron VMC share one local-operator contract.
An amplitude maps one configuration to `LogAmplitude(log_abs, phase)`, and the
sampler targets the real log density `2 * log_abs`. Every
`AbstractLocalQuantumOperator` returns `LocalOperatorEstimate`: local value,
validity, portable status, operator-specific work count, configuration shape,
operator identity, compute dtype, and method identity.

For `AbstractDiscreteQuantumOperator`, the unchanged connected-configuration
algorithm evaluates

$$
E_{\mathrm{loc}}(x)=\sum_{x'}H_{x,x'}\frac{\psi(x')}{\psi(x)}.
$$

`connections(x)` returns fixed-capacity connected configurations, matrix elements,
and a validity mask. Fermionic ordering signs remain the discrete operator's
responsibility. Padded slots are ignored; an active invalid amplitude ratio or
nonfinite matrix element fails closed. `work_count` is the active connection count.

`ElectronicCoulombHamiltonian` implements the nonrelativistic Born--Oppenheimer
molecular Hamiltonian in three dimensions,

$$
H=-\frac12\sum_i\nabla_i^2
  +\sum_{i<j}\frac1{r_{ij}}
  -\sum_{iA}\frac{Z_A}{r_{iA}}
  +\sum_{A<B}\frac{Z_A Z_B}{R_{AB}}.
$$

Nuclei and unit conversion come from `AtomicStructure`. The electronic entry
points require the scale's reference to be Bohr/Hartree: Bohr and Hartree use
unit factors, while angstrom/electronvolt require the encoded physical factors
`1 Å = 1.8897261254578281 Bohr` and
`1 eV = 0.03674932217565499 Hartree`; arbitrary reference factors are rejected.
Electron configurations have shape `(electron_count, 3)`. Active pair masks
exclude only exact self pairs; exact coincident particles return
`SINGULAR_CONFIGURATION`, and distances are never clipped.
`ElectronicCoulombHamiltonian` remains the finite nonperiodic molecular route;
periodic calculations use the separately named finite-resolution Ewald and
`PeriodicFermiNet` contracts.

`ElectronicVMCResourcePlan` admits each finite electron/determinant case from
pair-stream storage, determinant work, kinetic method, dtype, and caller
limits. It replaces the former global four-electron ceiling but does not claim
unrestricted scaling. The division-free polynomial determinant remains the
zero-reactivating route; resource and conditioning failure remain explicit.

`ElectronicKineticPolicy` offers deterministic `exact` and `chunked-exact`
coordinate Hessian traces. `StochasticElectronicKineticPolicy` offers finite
Hutchinson or orthogonal-Hutchinson probes with semantic replay, estimator
variance, count, and exhaustion evidence. Probe variance is within-configuration
uncertainty and is not treated as additional independent walkers.

`ElectronicIntegralHamiltonian` covers a declared finite spin-orbital basis.
The `four-component-no-pair` label requires an explicit positive-energy
projector identity; it is not QED or a continuum Dirac-sea claim. Periodic
Coulomb evidence records real/reciprocal resolution and requires neutrality or
an explicit uniform background.

`phydrax.nn.quantum.FermiNet` supplies a canonical `LogAmplitude` for this
Hamiltonian. It uses shared one- and two-electron streams, a static leading
spin-up/trailing spin-down partition, full generalized Slater determinants,
row/column-scaled log envelopes chosen from the combined nonzero orbital
magnitudes, a recursive signed-log product with higher-order-correct zero and
subnormal tangents, polynomial scaled determinants whose singular terms retain
mixture derivatives, a signed linear determinant mixture shifted by actual
nonzero coefficient–determinant product magnitudes with coefficient- and
singularity-reactivation fallbacks, and envelope decays with a
strictly positive configurable physical floor. Same-spin coordinate exchange is
antisymmetric. The full generalized determinant does not impose a standalone
spatial sign rule on an opposite-spin exchange.

`electronic_initial_walkers` draws replayable finite-molecule chains around active
nuclei. `harmonic_mean_electron_proposal` scales each electron's Gaussian move by
the harmonic mean of its electron--nucleus distances. The proposal is
state-dependent, so its `log_prob` implements both directions and the existing
`MetropolisHastings` kernel applies the exact proposal-density correction.

`VariationalMonteCarloProblem` combines the amplitude, any local operator, fixed
`MetropolisHastings` kernel, initial chains, and explicit complex-parameter mode.
`solve_variational_monte_carlo` preserves chain state, refreshes stored target
values after parameter updates, and uses the shared training lifecycle. It builds
the score as a `JacobianLinearOperator`, the centered stochastic-reconfiguration
metric as `EmpiricalGramLinearOperator`, and solves through `phydrax.linalg`; it
does not materialize a sample-by-parameter Jacobian.

The parameter modes are `real`, `holomorphic`, and `nonholomorphic` (independent
real coordinates for complex parameters). `FiniteSignedPermutationSymmetry` and
`SymmetryProjectedAmplitude` remain available for finite discrete sectors.
`solve_variational_tdvp` retains the established fixed-step VMC path.
`solve_adaptive_tdvp` adds a fixed-attempt Euler/Heun temporal controller with
common-random-number stage addresses and separate sampling uncertainty; its
midpoint route is symmetric but does not claim exact generic conservation.
Only `solve_finite_subspace_tdvp` may report norm/energy preservation, for a
declared finite linear subspace with positive overlap and Hermitian,
time-independent Hamiltonian.

Samples are correlated. `markov_chain_measure` marks them non-IID, and the final
frozen-model chain can record ESS and rank-normalized R-hat diagnostics. A finite
energy estimate is not labeled a variational upper bound unless the estimator and
trial-domain conditions have been established separately.

See the [VMC cookbook](cookbook/quantum_vmc.md) and
[solver API](api/solver/variational_monte_carlo.md).

## Canonical finite channels and circuit devices

`FiniteCPTPMap` stores explicit input/output dimensions and a canonical Choi
action. Kraus, Choi, superoperator, unitary, local-program, tensor-local, and
memory-map adapters retain CP, TP, reconstruction, capacity, and source
evidence. Choi-to-Kraus cleanup is a preparation-only explicit policy; runtime
application never clips a density matrix. `FiniteLindbladChannelPlan`
exponentiates each fixed interval and composes certified maps rather than
falling back to Euler state repair.

PR #236's `HilbertRegisterLayout`, local unitary/Kraus operations,
`QuantumProgram`, and dense prepared executor are the canonical circuit IR and
dense device. QPV adds finite POVM exact/fixed-shot measurement, bounded
mid-circuit outcome branches, and a nearest-neighbor tensor-network executor.
Nonlocal tensor gates require an explicit SWAP rewrite; no device registry or
implicit decomposition exists.

The bounded ansatz catalog contains named Jastrow, RBM, autoregressive,
Slater-Jastrow, circuit, MPS, periodic determinant, and FermiNet amplitudes.
Jastrow/RBM flip caches use the root `IncrementalMarkovTarget`; parameter
updates require cache refresh and mismatch fails closed.

## Finite open-system claims

Open-system certificates name the represented dimension, saved nodes,
truncation/refinement sequence, and assumptions. Fock and HEOM stabilization
over declared epochs is an estimate unless a certified tail/contraction
hypothesis is supplied. LPDO compression remains PSD by compressing a
purification factor and reports trace loss and a trace-distance upper bound.
Process-memory projection requires positive retained initial weight and
Kraus-subspace leakage evidence. Finite steady-state uniqueness requires
Liouvillian nullity one and a physical trace-constrained density; a generic
finite trajectory window is not a uniqueness proof.

## Scope

The quantum surface includes dense closed-system propagation, explicit finite
Hilbert-factor programs, Fock operators, Lindblad and Gaussian open systems,
trajectories, HEOM, pseudomodes, memory kernels, process tensors, MPS/TEBD, and
locally purified evolution. These remain representation-specific and retain their
own approximation and physicality evidence; they are not folded into a universal
quantum array or backend.

Local deterministic programs are documented in
[Dense local quantum programs](guides_quantum_programs.md). Composite factorization
remains explicit rather than inferred from array shapes. Arbitrary learned
trajectories are not silently repaired to become unitary, completely positive, or
trace preserving.
