# Dense local quantum programs

Phydrax executes immutable ordered local operations on an explicit finite-dimensional
Hilbert factorization. The dense program layer is separate from classical
`phydrax.circuit`: it composes quantum maps in time order rather than solving a
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

## Failure and physicality

Structural errors raise during construction, planning, or refresh. Numerical
invalidity remains explicit in preparation evidence and execution status. The
executor never normalizes a state, symmetrizes or projects a density, clips an
eigenvalue, promotes a state vector to a density, or silently converts dtype.

The default density policy audits initial and final positivity. The optional
`density_positivity_audit="construction"` mode still audits the initial density,
then propagates the CP/TP-by-construction certificate and checks final trace and
Hermiticity; the omitted final numerical eigenvalue audit is reported as unaudited.

## Deliberate limits

The current layer is deterministic and dense. It does not provide measurement,
mid-circuit classical control, a string gate registry, parameter-shift rules,
implicit SWAP networks, MPS/LPDO fallback, continuous Hamiltonian segments, or
automatic factorization. Dense storage remains proportional to the total Hilbert
dimension for state vectors and its square for density matrices.
