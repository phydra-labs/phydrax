# Atomistic alchemy, elastic networks, and external fields

PhydraX represents an alchemical calculation as one canonical
`AtomisticForceFieldPlan` plus typed controls over its prepared potential program.
There is no endpoint-only molecule, topology, unit, trajectory, or estimator model.
Host preparation validates the complete controlled route before compiled execution.

## Controlled Hamiltonians

`AlchemicalControlSchedulePlan` owns ordered thermodynamic-state IDs, typed control
IDs, and a finite state-by-control table. Each control has distinct exact binary
endpoints and a monotone path. Runtime kernels select numeric control rows; strings are
never inspected inside a compiled calculation.

`AlchemicalInteractionPartitionPlan` binds every control to a stable-particle-ID region
and one interaction mode:

- cross-region interactions only;
- every interaction touching the region;
- internal region interactions only.

Preparation resolves those regions against the existing `PreparedAtomisticSystem` and
topology. State-dependent masses, constraints, virtual-site geometry, unknown or
inactive particles, overlapping route ownership, changing unsupported force-field
terms, and charge-changing controlled regions fail before execution.

`ControlledHamiltonianPlan` binds the prepared force field, schedule, partition, and
`SoftCorePolicy`. `PreparedControlledHamiltonian` evaluates one cell-aware scalar with
the canonical potential program. The resulting `ControlledHamiltonianEvaluation`
contains total, term, and atom energies; Cartesian forces; virial; control derivatives;
control values; state index; and complete success evidence. Forces and control
derivatives are differentiated from the same scalar energy.

Controlled harmonic bonds, angles, proper/improper torsions, Lennard-Jones routes, and
direct Coulomb routes use the canonical term implementations. Soft-core regularization
is applied only to changing pair routes. Unchanged interactions retain the ordinary
potential-program path. Reciprocal electrostatic and reciprocal/tail dispersion controls
are rejected until their real, reciprocal, self, background, exception, and correction
terms can change as one endpoint-exact Hamiltonian.

`AlchemicalReducedPotentialEvaluation` carries state-major dimensionless reduced
potentials, physical energies, coverage, ordered state/potential IDs, measure identity,
unit-system identity, and source identities. Convert it with
`phydrax.uq.reduced_potential_dataset_from_alchemical_evaluation`; do not construct an
anonymous matrix.

## Switching and protocol values

`AlchemicalSwitchingPlan` evaluates an explicit source-to-destination control path and
records oriented protocol work, coverage, state/potential identities, and sample
lineage. Convert committed switching records with
`phydrax.uq.reduced_work_dataset_from_alchemical_switching` before FEP or BAR.

The `phydrax.atomistic.free_energy` package provides covariance-aware value contracts,
not a second scheduler:

- `NeutralAbsoluteSolvationPlan` combines vacuum and solvent decoupling legs.
- `AbsoluteBindingPlan` combines solvent and complex legs with explicit restraint,
  standard-state, and symmetry corrections.
- `SeparatedTopologyPlan` defines one environment decoupling result without an atom
  mapping.
- `MappedRelativeSolvationPlan` and `MappedRelativeBindingPlan` require one
  authenticated stable-ID mapping shared by both legs.

Every state and leg follows destination minus source. Leg and correction covariance is
propagated through the declared contrast; a missing, failed, mismatched, charge-changing,
or differently normalized component is rejected. Initial absolute-solvation and binding
contracts are neutral-only. They consume already prepared native force fields and
qualified multistate results; they do not parameterize, solvate, or infer molecular
chemistry.

See [Native alchemical free energy](guides_atomistic_alchemical_free_energy.md) for the
measure, sampling, sign, correction, and release boundaries.

## Reference-derived elastic networks

`ElasticNetworkPlan(cutoff, stiffness, edge_capacity)` selects every active stable-ID pair whose reference minimum-image distance is at most the cutoff. Selected edges are lexicographically ordered by stable particle IDs. The prepared identity includes the system, plan, reference provenance, padded edge tensors, and validity mask. Only active reference coordinates must be finite; ignored padding is canonicalized before fingerprinting. Preparation fails if the cutoff graph exceeds edge capacity or a selected reference edge has zero length.

```python
network = phx.atomistic.ElasticNetworkPlan(
    cutoff=0.8,
    stiffness=500.0,
    edge_capacity=4096,
).prepare(system, reference_positions, reference_id="equilibrated-frame-2000")
evaluation = network.evaluate(positions)
```

Each spring contributes ½ k (r − r₀)². `evaluate` returns its fixed-capacity edge ledger, total energy, and equal-and-opposite conservative forces. Invalid padded routes are masked before geometry and force assembly, so nonfinite inactive coordinate sentinels cannot contaminate an evaluation. A valid edge that collapses to zero length produces finite fail-closed output with `successful=False`, because its conservative direction is undefined. Translation and rotation leave a nonperiodic network invariant; periodic systems use the system cell's native minimum-image operation. `preparation` records selected and reserved edge counts and the reference identity.

## Scalar and vector gridded fields

`GriddedExternalFieldPlan` accepts a regular three-dimensional scalar grid `(nx, ny, nz)` or vector grid `(nx, ny, nz, components)`, with at least two nodes per axis. Coordinate frame, coordinate unit, and value unit are mandatory parts of field identity. Preparation fixes the grid shape and compiles multilinear interpolation without dynamic stencil allocation.

```python
field = phx.atomistic.GriddedExternalFieldPlan(
    origin=[-1.0, -1.0, -1.0],
    spacing=[0.1, 0.1, 0.1],
    values=potential_grid,
    boundary_policy=phx.atomistic.ExternalFieldBoundaryPolicy.FAIL,
    coordinate_frame="laboratory",
    coordinate_unit="nm",
    value_unit="kJ/mol",
).prepare()

sample = field.evaluate(query_points)
force = field.energy_and_forces(atom_positions, coupling=particle_couplings)
```

`evaluate` returns interpolated values, coordinate Jacobians, and per-point out-of-domain evidence. A scalar field additionally supports `energy_and_forces`: the total energy is the coupling-weighted sum of particle values, and forces are the negative gradient of that same energy.

Boundary policies are explicit:

- `PERIODIC` wraps each axis with a period equal to its node count times spacing. The evidence still records which original queries were outside the principal grid domain.
- `CLAMP` evaluates at the nearest boundary and sets the derivative along every clamped coordinate to zero.
- `FAIL` emits NaN values and Jacobians for offending points and marks the evaluation unsuccessful without changing array shape, so compiled callers can fail closed from evidence.

All policies report `out_of_domain`, `out_of_domain_count`, `finite`, and `successful`. Periodic and clamped out-of-domain queries can be successful; a failed-domain query cannot.
