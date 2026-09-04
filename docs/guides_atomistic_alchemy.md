# Atomistic alchemy, elastic networks, and external fields

PhydraX represents an alchemical calculation as an immutable plan followed by a prepared, fixed-shape runtime. Endpoint construction and mapping are host-side operations. Energy, force, derivative, work, cross-evaluation, elastic-network, and grid-interpolation operations use JAX arrays and retain their shapes under compilation.

## Endpoint alchemy

An `AlchemicalEndpointPlan` carries stable particle IDs, aligned atom type,
charge, Lennard-Jones sigma/epsilon and harmonic-bond tensors, and the complete
`AtomisticUnitSystem`. A dummy atom is explicit: its dummy mask is true, its
charge and epsilon are zero, and no active bond may reference it. Both endpoints
must have the same exact unit-system identity; the Coulomb coefficient is derived
from that descriptor rather than supplied independently.

```python
import jax.numpy as jnp
import phydrax as phx
units = phx.atomistic.AtomisticUnitSystem.reduced()

initial = phx.atomistic.AlchemicalEndpointPlan(
    [10, 20],
    [0, 1],
    jnp.asarray([1.0, -1.0]),
    jnp.asarray([0.30, 0.32]),
    jnp.asarray([0.20, 0.15]),
    bond_particle_ids=[[10, 20]],
    bond_stiffness=jnp.asarray([800.0]),
    bond_equilibrium_lengths=jnp.asarray([0.11]),
    units=units,
)
final = phx.atomistic.AlchemicalEndpointPlan(
    [110, 120],
    [2, 3],
    jnp.asarray([0.5, -0.5]),
    jnp.asarray([0.31, 0.34]),
    jnp.asarray([0.18, 0.12]),
    bond_particle_ids=[[110, 120]],
    bond_stiffness=jnp.asarray([700.0]),
    bond_equilibrium_lengths=jnp.asarray([0.12]),
    units=units,
)
```

Mapping is an explicit one-to-one table of endpoint particle IDs. Preparation orders mapped atoms deterministically, appends endpoint-only atoms in stable-ID order, creates dummy slots at the opposite endpoint, and pads atoms and bonds to declared capacities. Capacity overflow and unknown or duplicate mapping IDs fail during preparation. A bond between atoms that remain non-dummy at both endpoints must exist at both endpoints; disappearance of such a mapped-core bond is rejected rather than silently changing topology.

```python
schedule = phx.atomistic.LambdaSchedulePlan(
    jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
    jnp.asarray([0.0, 0.10, 0.5, 0.90, 1.0]),
)
plan = phx.atomistic.AlchemicalTransformationPlan(
    initial,
    final,
    atom_mapping=[[10, 110], [20, 120]],
    atom_capacity=2,
    bond_capacity=1,
    schedule=schedule,
    soft_core=phx.atomistic.SoftCorePolicy(
        lennard_jones_alpha=0.5,
        electrostatic_alpha=0.5,
        coupling_power=2,
    ),
    beta=1.0,
)
transformation = plan.prepare()
```

`interpolate(lambda_value)` exposes fixed-shape interpolated atom and bond tensors and requires one scalar schedule coordinate. Integer atom types select the nearest endpoint while continuous physical parameters and occupancy weights follow the schedule coupling. The potential is endpoint exact: λ = 0 evaluates the initial endpoint and λ = 1 evaluates the final endpoint. Lennard-Jones and electrostatic soft-core shifts are gated independently when the corresponding interaction term changes activity, including zero-ε or zero-charge transitions, and vanish at the endpoint where that term is physical.

`evaluate(positions, lambda_value)` returns total energy and forces, plus component energies and λ derivatives ordered as harmonic bond, Coulomb, and Lennard-Jones. Forces are the negative coordinate derivative of the same scalar energy. `dudlambda` is the sum of `component_dudlambda`; it includes the derivative of a nonlinear piecewise-linear coupling schedule.

`work(positions, lambda_initial, lambda_final)` computes instantaneous work as the final energy minus the initial energy at fixed coordinates, both in total and by component. `cycle_work` accepts a finite in-range λ path whose first and last values are equal and reports telescoping closure explicitly.

### Cross-evaluation for free-energy estimators

For sample coordinates with shape `(samples, atom_capacity, 3)`, `cross_evaluate` returns state-by-sample energy and reduced-potential matrices. Rows follow the supplied λ vector, or the prepared schedule when it is omitted. `values` equals β times `energies` and has shape `(states, samples)`, directly matching `phydrax.uq.ReducedPotentialSamples`:

```python
cross = transformation.cross_evaluate(sample_positions)
samples = phx.uq.ReducedPotentialSamples(
    cross.values,
    state_counts,
    origin_states,
    source_id=cross.prepared_id,
)
result = phx.uq.multistate_bennett_acceptance_ratio(samples)
```

FEP uses differences between two rows, BAR uses forward and reverse row differences, thermodynamic integration consumes sampled `dudlambda`, and MBAR consumes the entire matrix. Applications must check `successful`; it is false for out-of-range λ values, any nonfinite energy, or a zero-length bond carrying nonzero endpoint weight.

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
