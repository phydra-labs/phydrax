# Circuit-QED device modeling

Phydrax models reduced superconducting modes as a composition of qualified quantum
operators, sparse topology, local Hamiltonian terms, sampled controls, and an explicit
logical subspace. It does not infer a circuit Hamiltonian from an arbitrary electrical
netlist.

## Conventions

Circuit parameters and control generators use the same Hamiltonian units. With
`hbar=1`, rates are angular frequencies. A value measured in cycles per unit time must
be multiplied by `2π` before it is supplied as an angular rate.

`external_phase` is a reduced phase. It is not a flux in webers and is not a value in
units of the flux quantum. Convert a normalized flux `Φ/Φ₀` to reduced phase with
`2π Φ/Φ₀`.

Every time-dependent solve retains one fixed coordinate basis. Re-diagonalizing into an
instantaneous basis would require the associated frame-connection term and is never done
implicitly.

## Local modes and basis reduction

A circuit model first produces a generic `ModeReductionProblem`. The reduction lifecycle
then computes the retained eigenspace, projects every named operator with the same
isometry, and reports residual, gap, and continuation evidence.

```python
import jax.numpy as jnp
import phydrax as phx

q = phx.operators.quantum

basis = q.ChargeBasis(12)
problem = q.transmon_mode_problem(
    q.TransmonParameters(
        2.0 * jnp.pi * 0.25,
        2.0 * jnp.pi * 8.0,
        2.0 * jnp.pi * 7.5,
        external_phase=0.2,
    ),
    basis,
)
prepared = q.prepare_mode_reduction(
    problem,
    policy=q.ModeReductionPolicy(
        4,
        minimum_boundary_gap=1e-3,
    ),
)

if not bool(prepared.diagnostics.valid):
    raise RuntimeError("The retained circuit mode is not qualified.")
```

The public reduced models are:

- `transmon_mode_problem` with a finite integer-charge basis;
- `fluxonium_mode_problem` with a fixed-reference oscillator basis;
- `harmonic_mode_problem` with explicit oscillator quadrature scaling.

The transmon constructor uses two junction rates directly. It therefore retains the
phase shift of an asymmetric SQUID instead of reducing the junction pair to one effective
magnitude. Fluxonium uses a fixed oscillator scale so an optimized physical parameter does
not silently redefine the coordinate basis.

### Basis qualification

One cutoff is not evidence of convergence. Prepare two independent resolutions and
compare them:

```python
report = q.compare_mode_resolutions(
    coarse,
    fine,
    policy=q.ModeResolutionPolicy(
        energy_absolute=1e-7,
        energy_relative=1e-6,
        operator_absolute=1e-5,
    ),
)
```

The comparison reports energy and projected-operator drift. An optional explicit
coarse-to-fine embedding adds retained-subspace overlap evidence.

`refresh_mode_reduction` preserves the original static structure and follows the nominal
retained eigenspace. Matching is one-to-one. Degeneracies are aligned as subspaces;
ambiguous crossings invalidate the result instead of silently relabeling states.

## Device topology and parameter sharing

`CircuitQEDDeviceSpec` binds typed mode placements and interactions to one unbatched
`GraphIR`. Graph nodes define mode order. Graph edges define pairwise interaction order.
Additional explicitly targeted product interactions may be supplied separately.

A placement references a numerical parameter block by integer index. Repeated indices are
exact parameter sharing. Independent fabrication samples use independent blocks or an
outer `jax.vmap`; the device compiler does not draw hidden random variations.

```python
import jax.numpy as jnp
import phydrax as phx

q = phx.operators.quantum
s = phx.solver

raw_basis = q.OscillatorBasis(10)
reduction = q.ModeReductionPolicy(3)
topology = phx.graph.GraphIR(
    n_node=jnp.asarray([2]),
    n_edge=jnp.asarray([1]),
    senders=jnp.asarray([0]),
    receivers=jnp.asarray([1]),
)
placements = (
    s.CircuitModePlacement("q0", "harmonic", raw_basis, 0, reduction),
    s.CircuitModePlacement("q1", "harmonic", raw_basis, 1, reduction),
)
interaction = s.CircuitInteraction(
    (0, 1),
    ("phase", "phase"),
    0,
)
port = s.CircuitDrivePort(0, "charge", 0)
spec = s.CircuitQEDDeviceSpec(
    topology,
    placements,
    (interaction,),
    drive_ports=(port,),
)
parameters = s.CircuitQEDDeviceParameters(
    (
        q.HarmonicModeParameters(4.8),
        q.HarmonicModeParameters(5.1),
    ),
    interaction_strengths=jnp.asarray([0.03]),
    drive_scales=jnp.asarray([1.0]),
)
device = s.prepare_circuit_qed_device(spec, parameters)
```

Preparation returns:

- one aligned reduction per placement;
- the ordered `HilbertRegisterLayout`;
- a constant drift `LocalHamiltonian`;
- ordered drive-port terms;
- dense, state, and prepared-storage costs;
- aggregated validity evidence.

Dense admissibility is reported independently from local-state admissibility. A device may
remain valid for local evolution while its dense Hamiltonian is disallowed.

## Controls

A quantum control line combines one existing scalar control parameterization with:

- trainable I coefficients;
- optional trainable Q coefficients;
- angular carrier rate and phase;
- whole-waveform delay;
- an explicit compact support interval.

Values outside support are exactly zero. Boundary values are not extended by clipping.
The line-to-drive transfer matrix captures direct actuation and crosstalk.

```python
control_grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 20.0, 201),
    time_id="gate-grid",
)
parameterization = phx.control.PiecewiseLinearControlParameterization(
    control_grid,
    (),
    parameterization_id="q0-iq-envelope",
)
line = s.QuantumControlLine(
    parameterization,
    jnp.zeros(parameterization.parameter_shape),
    quadrature_coefficients=jnp.zeros(parameterization.parameter_shape),
    carrier=s.QuantumCarrier(angular_rate=4.8, phase=0.0),
    support_start=0.0,
    support_stop=20.0,
)
control = s.QuantumControlSchedule(
    (line,),
    s.LinearQuantumControlTransfer(jnp.asarray([[1.0]])),
)
sampled = s.sample_quantum_control_schedule(control, control_grid.times)
schedule = s.assemble_circuit_qed_hamiltonian(device, sampled)
```

The sampled result contains line values and final Hamiltonian-term coefficients. Drift
coefficients are inserted as explicit columns of ones when the driven Hamiltonian is
assembled.

## Exact-state local evolution

`FixedGridLocalHamiltonian` is a continuous-time Hamiltonian schedule. It is not a
`QuantumProgram` and is not silently compiled into a gate list.

```python
initial = jnp.zeros((device.plan.layout.dimension,), dtype=jnp.complex128)
initial = initial.at[0].set(1.0)
prepared_evolution = s.prepare_local_hamiltonian_evolution(
    schedule,
    policy=s.LocalHamiltonianEvolutionPolicy(order=2),
)
result = s.solve_local_hamiltonian_evolution(prepared_evolution, initial)
```

The default second-order symmetric product formula exponentiates only each term's local
matrix and applies it directly to selected state-tensor axes. The solver never constructs
the full Hamiltonian. It reports local-unitarity, state-norm, finite-value, resource, and
schedule evidence.

Use `order=1` only as an explicit baseline. Establish time-step adequacy by comparing a
refined grid against the same physical observable.

### Differentiation

`differentiation="autodiff"` differentiates the executed algorithm normally.

`differentiation="reversible-product-formula"` implements a memory-bounded reverse pass:
it reconstructs preceding unitary states in reverse interval order and accumulates the
VJP of each discrete product-formula step. It differentiates the discrete solver, not an
unqualified continuous equation. It rejects saved intermediate states and fails if
backward reconstruction exceeds tolerance.

The reversible route requires Hermitian generators and cannot be used for dissipative or
truncating tensor-network evolution.

## Dressed and computational subspaces

Whole-device diagonalization is explicit and bounded:

```python
labels = tuple(
    s.DressedStateLabel(levels)
    for levels in ((0, 0), (0, 1), (1, 0), (1, 1))
)
dressed = s.prepare_dressed_spectrum(device, labels=labels)
zz = (
    dressed.energy((1, 1))
    - dressed.energy((1, 0))
    - dressed.energy((0, 1))
    + dressed.energy((0, 0))
)
logical_subspace = s.dressed_quantum_subspace(dressed, labels)
```

Product labels are matched one-to-one by overlap. A hard assignment is not differentiated.
Near crossings, use the overlap and assignment-margin diagnostics rather than assuming
that sorted eigenvalues retain a physical identity.

For a product basis, `BasisStateSubspace` stores flat indices and uses gather/scatter.
For dressed states, `DenseQuantumSubspace` stores and validates an isometry.

## Fidelity and leakage

Gate-quality functions report separate quantities:

- average subspace survival;
- leakage;
- unconditional average gate fidelity;
- conditional fidelity when survival is nonzero;
- effective logical map or Choi matrix;
- target, source, and subspace validity.

```python
quality = q.unitary_gate_quality(
    physical_unitary,
    target_unitary,
    logical_subspace,
)
```

`finite_channel_gate_quality` applies the same logical restriction to a canonical
`FiniteCPTPMap`. The projected operation may be trace-decreasing and is not relabeled
CPTP.

No local frame compensation is hidden in these metrics. Apply a declared frame correction
to the target or effective operation and evaluate it as a separate result.

## Tensor-network lowering

A factored `LocalHamiltonianTerm` lowers exactly into an MPO:

```python
lowering = s.lower_local_hamiltonian_to_mpo(
    device.drift,
    chain_order=("q0", "q1"),
)
coefficient_basis = s.fixed_grid_local_hamiltonian_mpo_coefficients(
    schedule,
    lowering,
)
```

Heterogeneous local dimensions are supported. Noncontiguous product terms receive explicit
identity factors between their targets. An unfactored multi-site block is rejected; it is
never silently approximated by an undeclared tensor decomposition.

The resulting MPO and coefficient basis compose with the existing TDVP/MPS stack. Tensor
truncation, bond capacity, and time-integration evidence remain the responsibility of the
selected tensor solver.

## Open systems

Circuit-mode reductions expose projected charge, phase, lowering, and number operators.
Use them to construct physically declared collapse operators and pass those to existing
finite Lindblad, trajectory, LPDO, HEOM, pseudomode, or memory-kernel APIs. Circuit-QED
modeling does not add another channel or Liouvillian representation.

## Failure boundaries

Construction raises for structural errors: incompatible dimensions, unknown modes,
invalid topology, missing projected operators, impossible parameter references, or denied
resource requests.

Numerical operations return validity evidence for:

- nonfinite values;
- Hamiltonian Hermiticity;
- eigenpair and orthogonality residuals;
- retained/discarded spectral separation;
- eigenspace correspondence;
- norm and local-unitarity preservation;
- dense and MPO resource admissibility.

A scalar objective should consume these diagnostics explicitly. Invalid physics is never
normalized, relabeled, or projected into apparent validity.
