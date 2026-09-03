# Local quantum programs

Phydrax executes immutable ordered local operations on an explicit
finite-dimensional Hilbert factorization. Dense, MPS, and locally purified
executors are separate representation contracts over the same `QuantumProgram`;
none is a fallback for another. Quantum programs remain separate from classical
`phydrax.circuit`: they compose quantum maps in time order rather than solving a
network conservation equation.

## Explicit register factorization

`HilbertRegisterLayout` records ordered wire IDs and local dimensions. Factorization
is never inferred from a flat state dimension.

```python
import jax.numpy as jnp
import phydrax as phx

q = phx.operators.quantum
layout = q.HilbertRegisterLayout(("qubit", "qutrit"), (2, 3))
```

A local matrix is flattened in the exact order of its target IDs. For targets
`("qutrit", "qubit")`, the first local matrix factor belongs to `qutrit`; changing
target order changes the operation.

## State-vector programs

Local execution reshapes the state to subsystem axes, contracts only the selected
axes with `opt_einsum.contract`, restores the register order, and returns the original
flat state shape. It never constructs the global embedded operator.

```python
x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
program = q.QuantumProgram(
    layout,
    (q.LocalUnitaryOperation(x, ("qubit",)),),
    state_kind="state-vector",
)
prepared = phx.solver.prepare_dense_quantum_program(program)
initial = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
result = phx.solver.execute_dense_quantum_program(prepared, initial)
```

State vectors may carry leading batch axes. Local operation matrices are exactly
rank two. Parameter batches use an outer `jax.vmap` over program refresh and
execution; they are not encoded as ambiguous leading matrix axes.

## Density programs and local channels

Density programs admit local unitaries and Kraus channels. A Kraus stack has exact
shape `(K, dT, dT)` and is completely positive by construction. Preparation checks
finiteness and the trace-preservation residual `max(abs(sum(K†K) - I))`; it does not
allocate a Choi matrix.

```python
gamma = jnp.asarray(0.1)
kraus = jnp.stack(
    (
        jnp.asarray(
            [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]],
            dtype=jnp.complex128,
        ),
        jnp.asarray(
            [[0.0, jnp.sqrt(gamma)], [0.0, 0.0]],
            dtype=jnp.complex128,
        ),
    )
)
channel_program = q.QuantumProgram(
    q.HilbertRegisterLayout(("qubit",), (2,)),
    (q.LocalKrausChannelOperation(kraus, ("qubit",)),),
    state_kind="density-matrix",
)
```

Kraus branches are accumulated through a fixed-capacity loop. No global Kraus
operator or superoperator is materialized.

## Plan, prepare, refresh, execute

`plan_dense_quantum_program` resolves target routes, validates local matrix
dimensions and state capability, and rejects state, operation, or workspace resource
overflow. `prepare_dense_quantum_program` binds numerical matrices and records
unitarity or trace-preservation evidence. `refresh_dense_quantum_program` accepts
new numerical values only when layout, operation order and variants, targets, matrix
shapes, Kraus capacity, dtype, state kind, and policy are unchanged.

A refresh preserves `prepared_id`, increments `numeric_version`, and remains in the
JAX numerical path. This supports `jax.jit`, outer `jax.vmap`, and real-scalar
objectives differentiated through refreshed local matrices.

## Parameterized programs and local observables

`QuantumProgramTemplate` is a host-constructed lowering specification, not a
second executable circuit representation. Fixed entries are ordinary
`LocalUnitaryOperation` or `LocalKrausChannelOperation` values.
`PauliRotationInstruction` binds a one- or two-qubit Pauli rotation to one
coordinate of an angle vector. Materialization produces an ordinary
`QuantumProgram`, so planning, physicality checks, refresh, and execution retain
their existing contracts.

```python
import jax.numpy as jnp
import phydrax as phx

q = phx.operators.quantum
layout = q.HilbertRegisterLayout(("q",), (2,))
template = q.QuantumProgramTemplate(
    layout,
    (q.PauliRotationInstruction(("X",), ("q",), 0),),
    state_kind="state-vector",
)
prepared = phx.solver.prepare_dense_quantum_template(template)
observable = q.LocalObservable(
    jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128),
    ("q",),
)
observable_plan = phx.solver.plan_dense_quantum_observables(
    prepared.prepared_program,
    (observable,),
)
initial = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
result = phx.solver.execute_dense_quantum_template(
    prepared,
    jnp.asarray([0.3], dtype=jnp.float64),
    initial,
)
expectation = phx.solver.evaluate_dense_quantum_observables(
    observable_plan,
    result,
)
```

Angle coordinates are real and have the precision paired with the template's
complex dtype. One angle may occur in several gates. A template never accepts a
leading angle batch; use an outer `jax.vmap`. Observable plans group matrices
with identical ordered targets and reuse the reduced state for that target
group. They preserve caller output order and report program validity,
Hermiticity, finiteness, and imaginary residuals. `real_values` is available
only when the result is a certified real expectation.

## Exact circuit gradients

Dense template execution remains differentiable through JAX. The separate
`ParameterShiftPlan` provides the portable exact first derivative for
`PauliRotationInstruction`. For each gate occurrence it evaluates the plus and
minus π/2 shifts with coefficients plus and minus 1/2. If one angle is shared
by several gates, occurrences are shifted separately and accumulated into the
same angle coordinate.

```python
shift_plan = phx.solver.plan_parameter_shift(template)
gradient = phx.solver.evaluate_parameter_shift_jacobian(
    prepared,
    observable_plan,
    shift_plan,
    jnp.asarray([0.3], dtype=jnp.float64),
    initial,
)
```

`DenseCircuitExpectationModel` may select `"autodiff"` or
`"parameter-shift"`; both have the same primal output. Parameter-shift mode
uses a custom VJP with respect to the angle vector, so JAX composes the circuit
derivative through an arbitrary differentiable classical angle model.
Second-order shift derivatives, arbitrary generators, shot estimates, and
MPS/LPDO circuit gradients are not certified.

## Exact quantum feature models and kernels

`phydrax.ml.quantum` supplies exact dense circuit state and expectation models.
The standard builders cover an IQP state map, projected IQP X/Y/Z features,
and trainable affine data re-uploading. Entanglement is an explicit edge tuple;
an empty tuple is the separable control.

`ExactQuantumStateFidelityKernel` evaluates
`|<psi(x), psi(y)>|**2`. It requires finite normalized pure states and performs
no normalization. Its matrix path materializes each input state once before
forming the Gram matrix. Projected-observable kernels use the existing
`InputTransformedKernel`, or a `CircuitFeatureTransformRecipe` followed by an
ordinary kernel method inside a leakage-safe `Pipeline`.

`VariationalCircuitClassifierRecipe` trains a binary circuit expectation model
and linear logit head with exact full-batch gradients. It requires an explicit
key, scalar binary targets, and one `MLBatch` case. Multiclass composition uses
the existing one-vs-rest or one-vs-one recipes. Hardware providers, shot-based
features, sampled fidelity matrices, and hidden positive-semidefinite Gram
repairs are deliberately absent.

## Open-chain MPS and LPDO execution

`plan_mps_quantum_program` binds a state-vector program to one template MPS
structure and tensor precision policy. `plan_lpdo_quantum_program` does the same
for a density program and a template locally purified density. The template is
required because the Hilbert layout determines physical dimensions but not
virtual bonds or purification capacities.

Targets are resolved once against the layout. MPS and LPDO execution accepts only
one-site or adjacent two-site operations and preserves the exact declared target
factor order. Non-nearest operations require an explicit compilation result with
caller-visible SWAP or other declared routing; the executor never contracts a
hidden long interval or inserts hidden SWAPs.

MPS execution accepts unitaries only. LPDO execution accepts adjacent unitaries
and one-site Kraus channels. Purified construction preserves positive
semidefiniteness, while bond or purification truncation can perturb trace. The
executor reports raw trace drift and never normalizes, projects, symmetrizes, or
clips the result.

Both lifecycles preserve prepared identity under numeric refresh and reject any
change to layout, representation, template structure, operation order or kind,
ordered targets, route, shape, dtype, Kraus capacity, or resource policy.

## Failure and physicality

Structural errors raise during construction, planning, or refresh. Numerical
invalidity remains explicit in preparation evidence and execution status. The
executor never normalizes a state, symmetrizes or projects a density, clips an
eigenvalue, promotes a state vector to a density, or silently converts dtype.

The default density policy audits initial and final positivity. The optional
`density_positivity_audit="construction"` mode still audits the initial density,
then propagates the CP/TP-by-construction certificate and checks final trace and
Hermiticity; the omitted final numerical eigenvalue audit is reported as unaudited.

## Instruments, experiments, and deliberate limits

`QuantumPOVM` and `QuantumInstrument` validate finite outcomes, positivity, and
completeness. `QuantumExperimentProgram` adds bounded classical registers,
static feed-forward tables, exact branch enumeration, and address-derived typed
shot randomness. Sampling replay is invariant to shot batching. Mixed
multi-Kraus branches require LPDO execution; pure MPS branches fail closed.

Hardware compilation records logical placement, native decomposition, and every
SWAP edge. Continuous controls use `FixedGridLocalHamiltonian` and the local
Hamiltonian evolution lifecycle; they are not silently materialized as gate programs.

The platform does not provide unbounded branch graphs, arbitrary Python
callbacks, hidden provider calls, global entropy, automatic gate factorization,
implicit routing, or a universal executor. Dense state-vector and density
storage still scale with the full Hilbert dimension. MPS and LPDO execution is
finite, open-boundary, fixed-capacity, and reports every approximation.
